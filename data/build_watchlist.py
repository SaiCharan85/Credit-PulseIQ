"""Build the monitored universe: labelled distress names + matched survivors.

Survivors are not hand-picked. They are drawn from the filers that were
*actually still reporting* in a recent XBRL frame, then stratified to match the
sector profile of the distress cohort. Two reasons that matters:

* **Peer groups need members.** A percentile is only meaningful against a
  populated distribution, so survivors are allocated by SIC division to the
  divisions the distress names actually occupy.
* **Hand-picking negatives biases the backtest.** Choosing obviously-healthy
  megacaps as the negative class makes the task artificially easy. Sampling the
  active-filer population by size band keeps the comparison honest.

"Survivor" here means *no Chapter 11 on record*, which is a censored
observation rather than a proven negative -- a name may simply not have failed
yet (SPEC 13).

Deterministic plain code, no LLM.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.request
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

from data.edgar import EdgarClient, default_user_agent
from data.labels import load_chapter11_labels
from data.watchlist import COHORT_DISTRESS, COHORT_SURVIVOR, WATCHLIST_CSV

FRAMES_URL = "https://data.sec.gov/api/xbrl/frames/us-gaap/Assets/USD/{frame}.json"

#: A survivor must have enough history for the investigator to see a trend.
MIN_PERIODIC_FILINGS = 8

#: ...and must still be filing. Appearing in a recent frame already implies
#: this; the explicit check guards against a stale frame.
MAX_REPORTING_GAP_DAYS = 400


def active_filers(frame: str, user_agent: str, min_assets: float) -> list[tuple[int, float]]:
    """(cik, total_assets) for every filer reporting Assets in ``frame``.

    One request for the whole population, rather than thousands of per-company
    lookups.
    """
    request = urllib.request.Request(
        FRAMES_URL.format(frame=frame), headers={"User-Agent": user_agent}
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        payload = json.load(response)
    rows = [(int(r["cik"]), float(r["val"])) for r in payload["data"] if r.get("val")]
    return sorted((r for r in rows if r[1] >= min_assets), key=lambda r: -r[1])


def sic_division(sic: str | int | None) -> str:
    s = str(sic or "").strip().zfill(4)
    return s[:2] if s and s != "0000" else ""


def profile_distress(client: EdgarClient, ciks: list[int]) -> dict[int, dict]:
    """Resolve SIC and size for the labelled distress names."""
    out: dict[int, dict] = {}
    for cik in ciks:
        try:
            sub = client.submissions(cik)
        except Exception as exc:  # noqa: BLE001
            print(f"  warn: {cik} submissions failed: {exc}", file=sys.stderr)
            continue
        out[cik] = {
            "name": sub.get("name", ""),
            "sic": str(sub.get("sic") or "").zfill(4),
            "division": sic_division(sub.get("sic")),
        }
    return out


def target_allocation(divisions: Counter, total: int) -> dict[str, int]:
    """Allocate survivor slots across divisions, proportional but floored.

    Every division containing a distress name gets at least
    ``DEFAULT_MIN_PEERS + 1`` survivors so its peer group can actually form.
    """
    floor = 4
    known = [d for d in divisions if d]
    if not known:
        return {}
    alloc = {d: floor for d in known}
    remaining = total - sum(alloc.values())
    if remaining > 0:
        weight_total = sum(divisions[d] for d in known)
        for d in known:
            alloc[d] += int(remaining * divisions[d] / weight_total)
    return alloc


def collect_survivors(
    client: EdgarClient,
    pool: list[tuple[int, float]],
    allocation: dict[str, int],
    exclude: set[int],
    max_lookups: int,
    today: date,
) -> list[dict]:
    """Walk the active-filer pool, keeping names that fill an open slot.

    Largest first, so survivors are substantive operating companies rather than
    micro-caps, and stopping as soon as quotas fill to bound the request count.
    """
    need = dict(allocation)
    found: list[dict] = []
    lookups = 0
    for cik, _assets in pool:
        if lookups >= max_lookups or not any(v > 0 for v in need.values()):
            break
        if cik in exclude:
            continue
        lookups += 1
        try:
            sub = client.submissions(cik)
        except Exception:  # noqa: BLE001
            continue
        division = sic_division(sub.get("sic"))
        if need.get(division, 0) <= 0:
            continue
        index = client.filing_index(cik)
        periodic = [f["filing_date"] for f in index if f["form"] in ("10-K", "10-Q")]
        if len(periodic) < MIN_PERIODIC_FILINGS:
            continue
        if (today - max(periodic)).days > MAX_REPORTING_GAP_DAYS:
            continue
        need[division] -= 1
        found.append(
            {
                "cik": cik,
                "name": sub.get("name", ""),
                "cohort": COHORT_SURVIVOR,
                "sector_hint": str(sub.get("sic") or "").zfill(4),
            }
        )
        if len(found) % 25 == 0:
            print(f"  {len(found)} survivors ({lookups} lookups)", file=sys.stderr)
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--survivors", type=int, default=250)
    parser.add_argument("--frame", default="CY2026Q1I")
    parser.add_argument("--min-assets", type=float, default=50e6)
    parser.add_argument("--max-lookups", type=int, default=1500)
    parser.add_argument("--out", type=Path, default=WATCHLIST_CSV)
    parser.add_argument("--user-agent", default=default_user_agent())
    args = parser.parse_args(argv)

    client = EdgarClient(user_agent=args.user_agent)
    labels = load_chapter11_labels()
    distress_ciks = sorted({x.cik for x in labels})
    print(f"distress cohort: {len(distress_ciks)} labelled filers", file=sys.stderr)

    profiles = profile_distress(client, distress_ciks)
    divisions = Counter(p["division"] for p in profiles.values() if p["division"])
    allocation = target_allocation(divisions, args.survivors)
    print(
        f"sector divisions in distress cohort: {len(divisions)}; "
        f"survivor slots allocated: {sum(allocation.values())}",
        file=sys.stderr,
    )

    pool = active_filers(args.frame, client.user_agent, args.min_assets)
    print(
        f"active filers in {args.frame} with assets >= {args.min_assets:,.0f}: {len(pool)}",
        file=sys.stderr,
    )

    survivors = collect_survivors(
        client, pool, allocation, set(distress_ciks), args.max_lookups, date.today()
    )

    rows = [
        {
            "cik": cik,
            "name": profiles.get(cik, {}).get("name", ""),
            "cohort": COHORT_DISTRESS,
            "sector_hint": profiles.get(cik, {}).get("sic", ""),
        }
        for cik in distress_ciks
    ]
    rows += survivors

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["cik", "name", "cohort", "sector_hint"])
        writer.writeheader()
        writer.writerows(rows)

    by_division: dict[str, int] = defaultdict(int)
    for r in survivors:
        by_division[sic_division(r["sector_hint"])] += 1
    print(
        f"\nwrote {len(rows)} names to {args.out} "
        f"({len(distress_ciks)} distress / {len(survivors)} survivors)",
        file=sys.stderr,
    )
    print(f"survivor divisions covered: {len(by_division)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
