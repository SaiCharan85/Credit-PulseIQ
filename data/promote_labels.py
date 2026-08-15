"""Promote reviewed discovery candidates into the Chapter 11 label set.

Discovery (``data/discover.py``) emits candidates with a verdict; this step
turns the confirmed ones into labels. Kept as a separate, scripted step so the
promotion criteria are auditable rather than buried in a manual edit -- and so
``needs_review`` rows can never slip in silently.

Only ``verdict == confirmed`` is promoted. Rows quarantined for review (an 8-K
that concerns a prior case, a mis-parsed date) stay out until a human resolves
them.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Sequence
from datetime import date
from pathlib import Path

from data.labels import CHAPTER11_CSV, EVENT_CHAPTER11

CANDIDATES_CSV = Path("data/labels/candidates_chapter11.csv")

COHORT_RECENT = "recent_2025_2026"
COHORT_PRIOR = "prior_2021_2024"
COHORT_HISTORICAL = "historical_pre_2021"

FIELDS = [
    "cik",
    "company",
    "event_type",
    "event_date",
    "as_of_date",
    "date_basis",
    "source_accession",
    "source_url",
    "cohort",
    "verification",
]


def cohort_for(event_date: date) -> str:
    if event_date >= date(2025, 1, 1):
        return COHORT_RECENT
    if event_date >= date(2021, 1, 1):
        return COHORT_PRIOR
    return COHORT_HISTORICAL


def _registrant_not_a_debtor(client, row: dict) -> bool:
    """Fourth check: is this a parent reporting a *subsidiary's* bankruptcy?

    Found when the window widened to 2018. FirstEnergy Corp ($42bn, healthy
    today) filed an item 1.03 describing the Chapter 11 of FirstEnergy
    Solutions. All three earlier checks pass, because the filing is entirely
    truthful -- just not about the filer.

    Reads the cached filing text via ``discover.registrant_is_debtor``. An
    earlier attempt inferred this from post-petition filing *behaviour*
    ("did it keep reporting normally?") and was wrong: it flagged Diebold
    Nixdorf, Core Scientific, CBL and Ferrellgas -- genuine filers that
    reorganised and resumed reporting -- which would have systematically
    deleted the successful reorganisations from the positive class.
    """
    from data.discover import _strip_html, registrant_is_debtor

    doc = (row.get("source_url") or "").rsplit("/", 1)[-1]
    if not doc:
        return False
    try:
        text = _strip_html(
            client.fetch_filing_document(int(row["cik"]), row["item103_accession"], doc)
        ).lower()
    except Exception:  # noqa: BLE001 - uncached document: keep, do not silently drop
        return False
    return not registrant_is_debtor(text)


def promote(
    candidates_paths: Sequence[Path],
    out_path: Path,
    max_historical_share: float = 0.10,
    affiliate_guard: bool = True,
) -> int:
    rows: list[dict] = []
    for p in candidates_paths:
        if p.exists():
            rows.extend(csv.DictReader(p.open(encoding="utf8")))
    confirmed = [r for r in rows if r["verdict"] == "confirmed" and r["petition_date"]]

    if affiliate_guard:
        from data.edgar import EdgarClient

        client = EdgarClient(offline=True)
        kept, quarantined = [], []
        for r in confirmed:
            (quarantined if _registrant_not_a_debtor(client, r) else kept).append(r)
        if quarantined:
            print(
                f"quarantined {len(quarantined)} filing(s) describing a subsidiary's "
                f"bankruptcy rather than the registrant's:",
                file=sys.stderr,
            )
            for r in quarantined:
                print(f"    {r['company'][:44]}", file=sys.stderr)
        confirmed = kept

    # Affiliates co-filing one case appear as separate registrants reporting
    # identical financials -- iHeartMedia filed as three. Keeping all of them
    # would count one credit event three times in precision/recall.
    seen_cofiling: set[tuple[str, str]] = set()
    deduped = []
    for r in sorted(confirmed, key=lambda x: x["company"]):
        key = (r["petition_date"], r["total_assets"] or "")
        if key[1] and key in seen_cofiling:
            continue
        seen_cofiling.add(key)
        deduped.append(r)
    if len(deduped) < len(confirmed):
        print(
            f"collapsed {len(confirmed) - len(deduped)} affiliate co-filing(s) "
            f"reporting identical financials on the same date",
            file=sys.stderr,
        )
    confirmed = deduped

    labels = []
    for r in confirmed:
        event_date = date.fromisoformat(r["petition_date"])
        labels.append(
            {
                "cik": r["cik"],
                "company": r["company"],
                "event_type": EVENT_CHAPTER11,
                "event_date": r["petition_date"],
                "as_of_date": r["item103_date"],
                "date_basis": r["date_basis"],
                "source_accession": r["item103_accession"],
                "source_url": r["source_url"],
                "cohort": cohort_for(event_date),
                "verification": r["verification_string"]
                if r.get("verification_string")
                else "item_1.03+chapter11_text+voluntary_petition",
            }
        )

    # One event per filer: keep the earliest. A filer that reorganises and files
    # again would otherwise contribute two positives for one credit story.
    by_cik: dict[str, dict] = {}
    for row in sorted(labels, key=lambda x: x["event_date"]):
        by_cik.setdefault(row["cik"], row)
    labels = sorted(by_cik.values(), key=lambda x: x["event_date"], reverse=True)

    # Keep the pre-2021 tail to roughly a tenth of the set: those filings sit in
    # a different XBRL-coverage and rate environment, and mixing eras silently
    # is how a backtest flatters itself.
    historical = [x for x in labels if x["cohort"] == COHORT_HISTORICAL]
    # Cap is derived from the *retained* total, not the pre-drop total: solving
    # h / (rest + h) = share gives h = rest * share / (1 - share). Using the
    # pre-drop count inflates the tail (it produced 15% for a 10% target).
    rest = len(labels) - len(historical)
    cap = int(rest * max_historical_share / (1 - max_historical_share))
    if len(historical) > cap:
        drop = {id(x) for x in historical[cap:]}
        labels = [x for x in labels if id(x) not in drop]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(labels)

    counts: dict[str, int] = {}
    for x in labels:
        counts[x["cohort"]] = counts.get(x["cohort"], 0) + 1
    print(f"candidates: {len(rows)}  confirmed: {len(confirmed)}  promoted: {len(labels)}", file=sys.stderr)
    for cohort in (COHORT_RECENT, COHORT_PRIOR, COHORT_HISTORICAL):
        print(f"  {cohort:22} {counts.get(cohort, 0)}", file=sys.stderr)
    print(f"written to {out_path}", file=sys.stderr)
    return len(labels)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--candidates", type=Path, nargs="+", default=[CANDIDATES_CSV])
    parser.add_argument("--out", type=Path, default=CHAPTER11_CSV)
    parser.add_argument("--max-historical-share", type=float, default=0.10)
    parser.add_argument("--no-affiliate-guard", action="store_true")
    args = parser.parse_args(argv)
    promote(
        args.candidates,
        args.out,
        args.max_historical_share,
        affiliate_guard=not args.no_affiliate_guard,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
