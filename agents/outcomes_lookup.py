"""What actually happened afterwards -- for the reader, never for the model.

Asked "did the company survive?", the Q&A answered that the assessment does
not say. Correct, and useless: the assessment is dated 2024-07-01 and cannot
see past it, but the filing is in our own label set and happened 348 days
later. The system was holding the answer and declining to give it.

The distinction is the point, and it must not blur:

**The investigator may never see this.** Every figure it reads is filtered to
the prediction date, and an outcome is by definition after it. Feeding this
into an assessment would be the cardinal sin the whole backtest rests on
avoiding -- it would not be a small leak, it would be the label.

**The reader may.** Someone asking today is asking about history. Withholding
a Chapter 11 filing from 2025 because a memo dated 2024 could not have known
it confuses the model's constraint with the user's.

So this module is reachable only from the answering path, never from
``ToolBox`` or the orchestrator, and everything it returns is labelled as
hindsight so the two can never be read as one claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path


@dataclass(frozen=True)
class Outcome:
    """A recorded event, with its distance from the prediction date."""

    cik: int
    event_type: str
    event_date: date
    days_after: int

    @property
    def within_horizon(self) -> bool:
        """Whether it fell inside the 12 months the model was asked about."""
        return 0 <= self.days_after <= 365

    def describe(self, as_of: date) -> str:
        if self.days_after < 0:
            return (
                f"This filer had already filed for bankruptcy protection on "
                f"{self.event_date}, before the {as_of} prediction date."
            )
        window = (
            "inside the twelve-month window the assessment was asked about"
            if self.within_horizon
            else "after the twelve-month window the assessment was asked about"
        )
        return (
            f"It did not survive. This filer entered Chapter 11 on "
            f"{self.event_date}, {self.days_after} days after the {as_of} "
            f"prediction date -- {window}. That is recorded in the outcome "
            f"labels and was not visible to the assessment, which sees only "
            f"filings public on or before {as_of}."
        )


def load_outcomes(path: Path | str = "data/labels/chapter11.csv") -> dict[int, date]:
    """Verified Chapter 11 filings by CIK. Earliest event per filer."""
    import csv

    p = Path(path)
    if not p.exists():
        return {}
    out: dict[int, date] = {}
    for row in csv.DictReader(p.open(encoding="utf8")):
        try:
            cik = int(str(row["cik"]).strip().lstrip("0") or 0)
            when = date.fromisoformat(row["event_date"].strip())
        except (KeyError, ValueError):
            continue
        if cik not in out or when < out[cik]:
            out[cik] = when
    return out


def outcome_for(cik: int, as_of: date, outcomes: dict[int, date]) -> Outcome | None:
    """The recorded outcome for one filer, or None if it never filed."""
    when = outcomes.get(cik)
    if when is None:
        return None
    return Outcome(cik=cik, event_type="chapter_11", event_date=when,
                   days_after=(when - as_of).days)


def survived_note(cik: int, as_of: date, outcomes: dict[int, date]) -> str:
    """Plain-language hindsight for the reader, always labelled as such."""
    found = outcome_for(cik, as_of, outcomes)
    if found is None:
        return (
            "No bankruptcy filing is recorded for this filer in the outcome "
            "labels. That is not proof it is healthy today -- the label set "
            "covers verified Chapter 11 filings in the monitored universe, and "
            "absence means no such filing was found, not that none exists."
        )
    return found.describe(as_of)
