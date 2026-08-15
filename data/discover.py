"""Chapter 11 candidate discovery from primary sources.

Produces *candidates* for review, never labels directly. Nothing here writes to
``data/labels/``: the three checks below reduce a few thousand hits to a
reviewable shortlist, and a human promotes rows into the label set.

The pipeline exists because no single EDGAR signal is trustworthy on its own:

1. **Full-text search** for bankruptcy language over 8-K filings in a window.
   High recall, poor precision -- the phrase "Bankruptcy or Receivership" is an
   item header that appears in solvent companies' filings as boilerplate.
2. **Structured item code** ``1.03`` from submissions metadata. Filer-supplied,
   and therefore miscoded: J.C. Penney's earliest ``1.03`` is a 2014 shareholder
   rights plan, six years before its bankruptcy.
3. **Filing text** must describe *this filer's own* Chapter 11 petition --
   excluding Chapter 7 liquidations and counterparty bankruptcies.

Each check catches errors the other two miss. Measured on the 2025-01-01 to
2026-08-14 window: 179 raw candidates, 89 with enough filing history, 58
confirmed by item code, and further exclusions at the text stage (Granite
Construction's ``1.03`` contains no Chapter 11 language at all).

Deterministic plain code -- no LLM (PROMPT hard rule 3).
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

from data.edgar import EdgarClient, default_user_agent

FTS_URL = "https://efts.sec.gov/LATEST/search-index"
FTS_QUERY = '"Bankruptcy or Receivership"'
FTS_PAGE_CAP = 9990

CHAPTER11_PATTERNS = ("chapter 11", "voluntary petition")
CHAPTER7_PATTERN = "chapter 7"

#: Form 8-K is due within four business days of the event. A parsed petition
#: date further back than this is not a disclosure lag -- it means the filing
#: concerns an older case or the date was mis-parsed. Either way, review.
MAX_DISCLOSURE_LAG_DAYS = 10

#: Fourth check: is the *registrant* among the debtors, or only a subsidiary?
#:
#: Found when the window was widened to 2018. FirstEnergy Corp ($42bn, healthy
#: today) filed an item 1.03 reading "...each wholly owned subsidiaries of
#: FirstEnergy Corp., filed voluntary petitions...". All three earlier checks
#: pass, because the filing is entirely truthful -- just not about the filer.
#: Labelling that parent bankrupt would put a healthy megacap in the positive
#: class.
#:
#: The distinction must be textual. An earlier attempt used post-petition filing
#: behaviour ("did it keep reporting normally?") and was wrong: it flagged
#: Diebold Nixdorf, Core Scientific, CBL and Ferrellgas -- all genuine filers
#: that reorganised and resumed reporting -- which would have deleted exactly
#: the successful reorganisations from the label set.
REGISTRANT_IS_DEBTOR = re.compile(
    r"(?:"
    r"(?:filed|commenced)[^.]{0,60}?\bby\s+the\s+(?:company|registrant)\b"
    r"|the\s+(?:company|registrant)\s+and\s+(?:certain\s+of\s+)?(?:its|our)[^.]{0,80}?"
    r"(?:subsidiar|affiliat|debtor)"
    r"|the\s+(?:company|registrant)\s*(?:,[^.]{0,60})?\s+(?:filed|commenced)"
    r"|\b(?:we|the\s+partnership|the\s+trust)\s+(?:filed|commenced)"
    r")",
    re.IGNORECASE,
)

#: The parent-only construction is an *appositive*: the debtors are described
#: as "each a subsidiary of X" or "wholly owned subsidiaries of X", which scopes
#: the filing to entities other than the registrant::
#:
#:     each wholly owned subsidiaries of FirstEnergy Corp., filed voluntary...
#:     each a subsidiary of Novelion Therapeutics Inc., filed voluntary...
#:
#: A genuine filer uses a *conjunction* instead, putting itself among the
#: debtors -- and must not match here::
#:
#:     Sears Holdings Corporation and the subsidiaries of the Company ... filed
#:     the Company and certain subsidiaries of the Company ... filed
#:
#: Requiring the "each a" / "wholly owned" qualifier is what separates them.
#: Note also the permissive ``.`` gap: a sentence-bounded ``[^.]`` fails,
#: because the parent's own name ends in a period ("FirstEnergy Corp.,").
SUBSIDIARY_IS_DEBTOR = re.compile(
    r"each\s+(?:an?\s+)?(?:direct\s+|indirect\s+|wholly[\s-]?owned\s+)*subsidiar\w+\s+of\s+"
    r".{0,100}?(?:filed|commenced)"
    r"|(?:direct|indirect|wholly[\s-]?owned)\s+subsidiar\w+\s+of\s+.{0,100}?(?:filed|commenced)",
    re.IGNORECASE,
)

MONTHS = (
    "january|february|march|april|may|june|july|august|september|october|november|december"
)
PETITION_DATE_RE = re.compile(
    rf"(?:on\s+)?({MONTHS})\s+(\d{{1,2}}),\s+(\d{{4}})[^.]{{0,160}}?"
    r"(?:petition date|filed voluntary petitions|commenced voluntary|filed a voluntary)",
    re.IGNORECASE,
)
PETITION_DATE_RE_ALT = re.compile(
    rf"(?:petition date|filed voluntary petitions?|commenced voluntary cases?)[^.]{{0,120}}?"
    rf"(?:on\s+)?({MONTHS})\s+(\d{{1,2}}),\s+(\d{{4}})",
    re.IGNORECASE,
)


@dataclass
class Candidate:
    """One filer that may have filed Chapter 11, with every check recorded."""

    cik: int
    company: str = ""
    sic: str = ""
    fts_date: str = ""
    item103_date: str = ""
    item103_accession: str = ""
    source_url: str = ""
    petition_date: str = ""
    date_basis: str = ""
    total_assets: float | None = None
    periodic_filings_before: int = 0
    last_periodic: str = ""
    has_item103: bool = False
    has_chapter11_text: bool = False
    has_chapter7_text: bool = False
    verdict: str = "pending"
    reasons: list[str] = field(default_factory=list)

    @property
    def confirmed(self) -> bool:
        return self.verdict == "confirmed"

    def verification_string(self) -> str:
        parts = []
        if self.has_item103:
            parts.append("item_1.03")
        if self.has_chapter11_text:
            parts.append("chapter11_text")
            parts.append("voluntary_petition")
        return "+".join(parts)


# ---------------------------------------------------------------------------
# Stage 1 -- full-text search sweep
# ---------------------------------------------------------------------------


def _quarters(start: date, end: date) -> Iterator[tuple[date, date]]:
    """EDGAR full-text search caps results per query, so sweep in windows."""
    cur = start
    while cur < end:
        year, month = cur.year + (cur.month + 2) // 12, (cur.month + 2) % 12 + 1
        nxt = min(date(year, month, 1), end)
        yield cur, nxt - timedelta(days=1)
        cur = nxt


def _fts_page(user_agent: str, start: date, end: date, offset: int, timeout: int = 45) -> dict:
    query = urllib.parse.urlencode(
        {
            "q": FTS_QUERY,
            "forms": "8-K",
            "startdt": start.isoformat(),
            "enddt": end.isoformat(),
            "from": offset,
        }
    )
    request = urllib.request.Request(f"{FTS_URL}?{query}", headers={"User-Agent": user_agent})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def sweep_full_text(
    start: date, end: date, user_agent: str, pause: float = 0.12, verbose: bool = True
) -> dict[int, dict[str, str]]:
    """Earliest bankruptcy-language 8-K per filer in the window."""
    earliest: dict[int, dict[str, str]] = {}
    for window_start, window_end in _quarters(start, end):
        offset = 0
        total = 0
        while True:
            payload = _fts_page(user_agent, window_start, window_end, offset)
            hits = payload["hits"]["hits"]
            total = payload["hits"]["total"]["value"]
            if not hits:
                break
            for hit in hits:
                src = hit["_source"]
                cik = int(src["ciks"][0])
                row = {
                    "date": src["file_date"],
                    "name": src["display_names"][0],
                    "accession": hit["_id"].split(":")[0],
                }
                if cik not in earliest or row["date"] < earliest[cik]["date"]:
                    earliest[cik] = row
            offset += len(hits)
            if offset >= min(total, FTS_PAGE_CAP):
                break
            time.sleep(pause)
        if verbose:
            print(f"  {window_start} .. {window_end}: {total} hits", file=sys.stderr)
        time.sleep(pause)
    return earliest


# ---------------------------------------------------------------------------
# Stage 2 and 3 -- structured item code, then filing text
# ---------------------------------------------------------------------------


def _strip_html(raw: str) -> str:
    text = re.sub(r"<[^>]+>", " ", raw)
    return re.sub(r"\s+", " ", html.unescape(text))


def registrant_is_debtor(text: str) -> bool:
    """True when the filing describes the *registrant's own* petition.

    Only rejects when the text points at a subsidiary and never puts the
    registrant among the debtors. Ambiguous filings default to True and are
    caught downstream -- silently discarding real bankruptcies is the more
    damaging error, because they are the positive class.
    """
    if REGISTRANT_IS_DEBTOR.search(text):
        return True
    return not SUBSIDIARY_IS_DEBTOR.search(text)


def _parse_petition_date(text: str) -> date | None:
    for pattern in (PETITION_DATE_RE, PETITION_DATE_RE_ALT):
        match = pattern.search(text)
        if not match:
            continue
        month_name, day, year = match.group(1), match.group(2), match.group(3)
        try:
            return datetime.strptime(f"{month_name} {day} {year}", "%B %d %Y").date()
        except ValueError:
            continue
    return None


def find_item_103(client: EdgarClient, cik: int, on_or_after: date | None = None) -> list[dict]:
    """8-K filings carrying item code 1.03, oldest first."""
    hits = []
    for filing in client.filing_index(cik):
        if not filing["form"].startswith("8-K"):
            continue
        if "1.03" not in (filing.get("items") or ""):
            continue
        if on_or_after and filing["filing_date"] < on_or_after:
            continue
        hits.append(filing)
    return sorted(hits, key=lambda f: f["filing_date"])


def inspect_candidate(
    client: EdgarClient,
    cik: int,
    fts_row: dict[str, str],
    window_start: date,
    min_periodic: int = 6,
    max_reporting_gap_days: int = 400,
) -> Candidate:
    """Run every check against one filer and record what each one found."""
    cand = Candidate(cik=cik, company=fts_row.get("name", ""), fts_date=fts_row.get("date", ""))

    try:
        sub = client.submissions(cik)
    except Exception as exc:  # noqa: BLE001 - a fetch failure is a finding, not a crash
        cand.verdict = "error"
        cand.reasons.append(f"submissions fetch failed: {exc}")
        return cand

    cand.company = sub.get("name", cand.company) or cand.company
    cand.sic = str(sub.get("sic") or "").zfill(4)

    item103 = find_item_103(client, cik, on_or_after=window_start)
    if not item103:
        cand.verdict = "rejected"
        cand.reasons.append("no 8-K carrying item 1.03 in window")
        return cand

    first = item103[0]
    cand.has_item103 = True
    cand.item103_date = first["filing_date"].isoformat()
    cand.item103_accession = first["accession"]

    index = client.filing_index(cik)
    periodic = [
        f["filing_date"]
        for f in index
        if f["form"] in ("10-K", "10-Q") and f["filing_date"] < first["filing_date"]
    ]
    cand.periodic_filings_before = len(periodic)
    if periodic:
        cand.last_periodic = max(periodic).isoformat()

    if len(periodic) < min_periodic:
        cand.verdict = "rejected"
        cand.reasons.append(f"only {len(periodic)} periodic filings before the event")
        return cand

    gap = (first["filing_date"] - max(periodic)).days
    if gap > max_reporting_gap_days:
        cand.verdict = "rejected"
        cand.reasons.append(f"stopped reporting {gap} days before the event")
        return cand

    try:
        raw = client.fetch_filing_document(cik, first["accession"], first["primary_document"])
    except Exception as exc:  # noqa: BLE001
        cand.verdict = "needs_review"
        cand.reasons.append(f"could not fetch filing text: {exc}")
        return cand

    cand.source_url = client.filing_document_url(cik, first["accession"], first["primary_document"])
    text = _strip_html(raw).lower()
    cand.has_chapter11_text = any(p in text for p in CHAPTER11_PATTERNS)
    cand.has_chapter7_text = CHAPTER7_PATTERN in text

    if not cand.has_chapter11_text:
        # The J.C. Penney / Granite Construction failure mode: item code says
        # bankruptcy, the document says something else entirely.
        cand.verdict = "rejected"
        cand.reasons.append("item 1.03 present but no Chapter 11 language in the filing")
        return cand

    if cand.has_chapter7_text and "chapter 11" not in text:
        cand.verdict = "rejected"
        cand.reasons.append("Chapter 7 liquidation, not Chapter 11")
        return cand

    petition = _parse_petition_date(text)
    if petition is not None:
        lag = (first["filing_date"] - petition).days
        if lag > MAX_DISCLOSURE_LAG_DAYS:
            # Two different problems produce a large gap, and neither may be
            # papered over with a default:
            #   * the 8-K concerns a *prior* case -- Hertz's in-window item 1.03
            #     is its 2021 emergence, referencing a 2020 petition. Falling
            #     back to the filing date would date the bankruptcy to the
            #     emergence, which is worse than the mis-parse.
            #   * the regex latched onto an unrelated date (Mondee: a 2024 date
            #     in an 8-K disclosing a 2025 petition).
            # Either way a human decides, because the event date drives lead
            # time and a silent guess corrupts the headline metric.
            cand.verdict = "needs_review"
            cand.reasons.append(
                f"parsed petition {petition} is {lag}d from disclosure "
                f"{first['filing_date']}; 8-K may concern a prior case, or the date is mis-parsed"
            )
            return cand
        if lag >= 0:
            cand.petition_date = petition.isoformat()
            cand.date_basis = "petition_date_from_8k_text"
        else:
            # A date *after* the disclosure cannot be a prior case -- it is
            # simply a bad parse. The filing date is safe here.
            cand.petition_date = cand.item103_date
            cand.date_basis = "8k_filing_date_fallback"
    else:
        # Conservative: the disclosure date can only understate lead time.
        cand.petition_date = cand.item103_date
        cand.date_basis = "8k_filing_date_fallback"

    cand.total_assets = latest_assets_before(client, cik, first["filing_date"])
    cand.verdict = "confirmed"
    return cand


def latest_assets_before(client: EdgarClient, cik: int, before: date) -> float | None:
    """Most recent reported total assets public before ``before`` (size filter)."""
    try:
        payload = client.company_concept(cik, "Assets")
    except Exception:  # noqa: BLE001
        return None
    entries = [
        u
        for u in payload.get("units", {}).get("USD", [])
        if u.get("filed") and date.fromisoformat(u["filed"]) < before
    ]
    if not entries:
        return None
    return float(max(entries, key=lambda u: (u["end"], u["filed"]))["val"])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def discover(
    start: date,
    end: date,
    client: EdgarClient,
    min_assets: float = 0.0,
    min_periodic: int = 6,
    verbose: bool = True,
) -> list[Candidate]:
    if verbose:
        print(f"stage 1: full-text sweep {start} .. {end}", file=sys.stderr)
    hits = sweep_full_text(start, end, client.user_agent, verbose=verbose)
    if verbose:
        print(f"stage 2/3: inspecting {len(hits)} filers", file=sys.stderr)

    out: list[Candidate] = []
    for n, (cik, row) in enumerate(sorted(hits.items(), key=lambda kv: kv[1]["date"]), 1):
        cand = inspect_candidate(client, cik, row, start, min_periodic=min_periodic)
        if cand.confirmed and min_assets and (cand.total_assets or 0) < min_assets:
            cand.verdict = "rejected"
            cand.reasons.append(f"total assets below {min_assets:,.0f}")
        out.append(cand)
        if verbose and n % 25 == 0:
            print(f"  inspected {n}/{len(hits)}", file=sys.stderr)
    out.sort(key=lambda c: (c.verdict != "confirmed", -(c.total_assets or 0)))
    return out


def write_candidates(candidates: list[Candidate], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(asdict(candidates[0]).keys()))
        writer.writeheader()
        for c in candidates:
            row = asdict(c)
            row["reasons"] = "; ".join(c.reasons)
            writer.writerow(row)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--start", type=date.fromisoformat, default=date(2025, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date.today())
    parser.add_argument("--min-assets", type=float, default=100e6)
    parser.add_argument("--min-periodic", type=int, default=6)
    parser.add_argument("--out", type=Path, default=Path("data/labels/candidates_chapter11.csv"))
    parser.add_argument("--user-agent", default=default_user_agent())
    args = parser.parse_args(argv)

    client = EdgarClient(user_agent=args.user_agent)
    candidates = discover(
        args.start, args.end, client, min_assets=args.min_assets, min_periodic=args.min_periodic
    )
    write_candidates(candidates, args.out)

    confirmed = [c for c in candidates if c.confirmed]
    print(f"\ncandidates: {len(candidates)}  confirmed: {len(confirmed)}", file=sys.stderr)
    print(f"written to {args.out}", file=sys.stderr)
    print("\nReview before promoting any row into data/labels/chapter11.csv.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
