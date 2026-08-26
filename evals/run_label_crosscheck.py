"""Cross-check our bankruptcy labels against court records.

Every number this project publishes rests on 156 Chapter 11 positives, and
until now nothing had checked them against a second source. They were derived
from EDGAR 8-K item 1.03 -- a company telling the SEC it filed -- which is
good evidence and still only one witness. A wrong label is worse than a wrong
model: it corrupts the measurement silently, and every arm inherits the error
equally so no comparison reveals it.

BankruptcyObserver's court dockets are an independent record of the same
events. Agreement is corroboration; disagreement is a bug worth finding.

Three outcomes, and the middle one is the interesting one:

``confirmed``  a court case for that debtor within the window, same date
``off by N``   found, but the court date differs from ours -- usually the 8-K
               was filed a day or two after the petition, which is expected and
               harmless, or we anchored on the wrong affiliate, which is not
``not found``  no court record. Could be a naming mismatch, a subsidiary
               filing under a different debtor name, or a label we should not
               have.

**This is eval-side only.** The module it calls documents at length why court
outcome data must never reach the investigator; nothing here is importable from
the serving path.

    python -m evals.run_label_crosscheck --limit 25
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import date
from pathlib import Path

from data.bko import best_match

EVENTS = Path("data/labels/distress_events.csv")

#: Days of slack between the petition and the 8-K that reports it. A company
#: has four business days to file an item 1.03, so a small positive offset is
#: the expected case, not a discrepancy.
EXPECTED_LAG_DAYS = 6


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=25)
    ap.add_argument("--events", default=str(EVENTS))
    args = ap.parse_args(argv)

    path = Path(args.events)
    if not path.exists():
        print(f"no events file at {path}")
        return 1

    from data.company_search import load_directory

    names: dict[int, str] = {}
    try:
        for row in load_directory():
            names[row["cik"]] = row["name"]
    except Exception:  # noqa: BLE001
        pass

    petitions = [
        r for r in csv.DictReader(path.open(encoding="utf8"))
        if (r.get("signal") or "").strip() == "chapter11_petition"
    ]
    petitions.sort(key=lambda r: r.get("event_date", ""), reverse=True)
    sample = petitions[: args.limit]

    print("Label cross-check -- our 8-K derived labels against court dockets")
    print(f"{len(petitions)} Chapter 11 labels on file, checking {len(sample)}\n")

    tally: Counter[str] = Counter()
    offsets: list[int] = []
    for row in sample:
        cik = int(row["cik"])
        ours = date.fromisoformat(row["event_date"])
        name = names.get(cik, "")
        if not name:
            tally["no name"] += 1
            print(f"  {'?':<9} CIK {cik:<9} {ours}  (not in the SEC directory)")
            continue

        case = best_match(name, near=ours)
        if case is None or case.date_filed is None:
            tally["not found"] += 1
            print(f"  {'NOT FOUND':<9} {name[:34]:<34} ours {ours}")
            continue

        delta = (case.date_filed - ours).days
        offsets.append(delta)
        if delta == 0:
            tally["exact"] += 1
            mark = "exact"
        elif abs(delta) <= EXPECTED_LAG_DAYS:
            tally["within lag"] += 1
            mark = f"{delta:+d}d"
        else:
            tally["mismatch"] += 1
            mark = f"{delta:+d}d !"
        print(f"  {mark:<9} {name[:34]:<34} ours {ours}  court {case.date_filed} "
              f"ch{case.chapter or '?'}  {case.case_number}")

    print("\n  " + "  ".join(f"{k}: {v}" for k, v in sorted(tally.items())))
    checked = tally["exact"] + tally["within lag"] + tally["mismatch"]
    if checked:
        agree = tally["exact"] + tally["within lag"]
        print(f"  corroborated: {agree}/{checked} ({agree / checked:.0%}) "
              f"within {EXPECTED_LAG_DAYS} days of the court date")
    print(
        "\n  A small positive offset is expected: a company has four business days\n"
        "  to file the 8-K that reports its own petition. A large one, or a miss,\n"
        "  is a label to look at by hand before it goes on being measured."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
