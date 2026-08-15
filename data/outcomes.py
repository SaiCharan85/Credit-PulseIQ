"""What happened *after* the petition: did the filer survive?

The Chapter 11 label answers "did it default". This answers "did it come back",
which is a different question and a different metric. In the current universe
84% did not: 87 of 152 deregistered outright, 40 stopped filing, and only 19
kept reporting.

This does **not** affect the distress leg's prediction target. The event being
predicted is the petition itself, and that is the same event whether the filer
later emerges or liquidates -- probability of default, not loss given default.
The distinction is recorded because it is cheap, CIK-native, and lets the
backtest slice results by severity later.

Determined from post-petition filing behaviour:

``deregistered``
    Filed Form 15 (deregistration). The company stopped being a public filer --
    typically liquidation or going private through the case.
``went_dark``
    Stopped filing without a Form 15. Usually the same outcome, less tidily.
``emerged``
    Kept filing 10-K/10-Q afterwards -- reorganised and continued reporting.
``in_process``
    Still filing something recently; the case has not resolved.

These are inferences from filing behaviour, not court records. A filer that
emerges but deregisters because it went private reads as ``deregistered`` here.
For court-accurate dispositions the CourtListener/PACER docket is the source;
that linkage is name-keyed and out of scope for now.

Deterministic plain code, no LLM.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Sequence
from datetime import date
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from data.edgar import EdgarClient, default_user_agent
from data.labels import LabelRecord, load_chapter11_labels

OUTCOMES_CSV = Path("data/labels/chapter11_outcomes.csv")

OUTCOME_DEREGISTERED = "deregistered"
OUTCOME_WENT_DARK = "went_dark"
OUTCOME_EMERGED = "emerged"
OUTCOME_IN_PROCESS = "in_process"
OUTCOME_UNKNOWN = "unknown"

#: A filer with at least this many periodic reports after the petition is
#: treated as having resumed normal reporting.
MIN_PERIODIC_AFTER = 2

#: Recent activity that keeps a case "in process" rather than dark.
RECENT_ACTIVITY_DAYS = 200

NON_SURVIVOR = frozenset({OUTCOME_DEREGISTERED, OUTCOME_WENT_DARK})


class BankruptcyOutcome(BaseModel):
    model_config = ConfigDict(frozen=True)

    cik: int
    company: str
    event_date: date
    outcome: str
    periodic_filings_after: int = 0
    deregistration_date: date | None = None
    last_filing_date: date | None = None

    @property
    def survived(self) -> bool:
        return self.outcome == OUTCOME_EMERGED


def classify(
    client: EdgarClient, label: LabelRecord, today: date | None = None
) -> BankruptcyOutcome:
    today = today or date.today()
    try:
        index = client.filing_index(label.cik)
    except Exception:  # noqa: BLE001
        return BankruptcyOutcome(
            cik=label.cik,
            company=label.company,
            event_date=label.event_date,
            outcome=OUTCOME_UNKNOWN,
        )

    after = [f for f in index if f["filing_date"] and f["filing_date"] > label.event_date]
    periodic = [f for f in after if f["form"] in ("10-K", "10-Q")]
    dereg = [f for f in after if f["form"].startswith("15")]
    last = max((f["filing_date"] for f in after), default=None)

    if dereg:
        outcome = OUTCOME_DEREGISTERED
    elif len(periodic) >= MIN_PERIODIC_AFTER:
        outcome = OUTCOME_EMERGED
    elif last is not None and (today - last).days < RECENT_ACTIVITY_DAYS:
        outcome = OUTCOME_IN_PROCESS
    else:
        outcome = OUTCOME_WENT_DARK

    return BankruptcyOutcome(
        cik=label.cik,
        company=label.company,
        event_date=label.event_date,
        outcome=outcome,
        periodic_filings_after=len(periodic),
        deregistration_date=min(f["filing_date"] for f in dereg) if dereg else None,
        last_filing_date=last,
    )


def classify_all(
    client: EdgarClient, labels: Sequence[LabelRecord], today: date | None = None
) -> list[BankruptcyOutcome]:
    return [classify(client, x, today) for x in labels]


def write_outcomes(rows: Sequence[BankruptcyOutcome], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "cik",
        "company",
        "event_date",
        "outcome",
        "survived",
        "periodic_filings_after",
        "deregistration_date",
        "last_filing_date",
    ]
    with path.open("w", encoding="utf8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "cik": r.cik,
                    "company": r.company,
                    "event_date": r.event_date.isoformat(),
                    "outcome": r.outcome,
                    "survived": int(r.survived),
                    "periodic_filings_after": r.periodic_filings_after,
                    "deregistration_date": r.deregistration_date.isoformat()
                    if r.deregistration_date
                    else "",
                    "last_filing_date": r.last_filing_date.isoformat() if r.last_filing_date else "",
                }
            )


def load_outcomes(path: Path | str = OUTCOMES_CSV) -> list[BankruptcyOutcome]:
    path = Path(path)
    if not path.exists():
        return []
    out = []
    with path.open(encoding="utf8", newline="") as fh:
        for row in csv.DictReader(fh):
            out.append(
                BankruptcyOutcome(
                    cik=int(row["cik"]),
                    company=row["company"],
                    event_date=date.fromisoformat(row["event_date"]),
                    outcome=row["outcome"],
                    periodic_filings_after=int(row["periodic_filings_after"] or 0),
                    deregistration_date=date.fromisoformat(row["deregistration_date"])
                    if row["deregistration_date"]
                    else None,
                    last_filing_date=date.fromisoformat(row["last_filing_date"])
                    if row["last_filing_date"]
                    else None,
                )
            )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=OUTCOMES_CSV)
    parser.add_argument("--user-agent", default=default_user_agent())
    args = parser.parse_args(argv)

    client = EdgarClient(user_agent=args.user_agent)
    labels = load_chapter11_labels()
    rows = classify_all(client, labels)
    write_outcomes(rows, args.out)

    counts: dict[str, int] = {}
    for r in rows:
        counts[r.outcome] = counts.get(r.outcome, 0) + 1
    non_survivors = sum(v for k, v in counts.items() if k in NON_SURVIVOR)
    print(f"classified {len(rows)} Chapter 11 filers -> {args.out}", file=sys.stderr)
    for k in sorted(counts, key=lambda k: -counts[k]):
        print(f"  {k:16} {counts[k]:>4}", file=sys.stderr)
    print(
        f"\nnon-survivors: {non_survivors}/{len(rows)} "
        f"({100 * non_survivors / max(len(rows), 1):.0f}%)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
