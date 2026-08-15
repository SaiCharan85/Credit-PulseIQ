"""Outcome labels (Chapter 11, AAER) with as-of dating.

Two dates per label, and the difference between them is load-bearing:

``event_date``
    When the thing happened (the bankruptcy petition was filed).
``as_of_date``
    When it became *publicly knowable* (the 8-K disclosing it hit EDGAR).

These are not the same. Bed Bath & Beyond petitioned 2023-04-23 and disclosed
2023-04-24; Hertz petitioned 2020-05-22 and disclosed 2020-05-26. Lead time is
measured to the event; visibility is governed by the as-of date.

Note the asymmetry in how this module is used, because it is easy to get
backwards:

* :func:`labels_visible_as_of` filters to what was knowable -- use it whenever a
  label feeds the agent as an *input* (e.g. a prior enforcement action).
* :func:`outcome_within_horizon` deliberately looks into the future. It is the
  grader, not an input. Scoring a prediction requires knowing what happened
  after it; that is not leakage as long as the result never reaches the agent.

Plain deterministic code (PROMPT hard rule 3).
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Sequence
from datetime import date, datetime, timedelta
from pathlib import Path

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

LABELS_DIR = Path(__file__).parent / "labels"
CHAPTER11_CSV = LABELS_DIR / "chapter11.csv"
AAER_CSV = LABELS_DIR / "aaer.csv"

EVENT_CHAPTER11 = "chapter11"
EVENT_AAER = "aaer"


class LabelRecord(BaseModel):
    """One outcome event, primary-sourced and dated twice."""

    model_config = ConfigDict(frozen=True)

    cik: int
    company: str
    event_type: str
    event_date: date
    as_of_date: date
    date_basis: str = ""
    source_accession: str | None = None
    source_url: str | None = None
    cohort: str = ""
    verification: str = ""
    # AAER-only: the fiscal window the misstatement covered.
    misstatement_start: date | None = None
    misstatement_end: date | None = None

    @field_validator("cik", mode="before")
    @classmethod
    def _coerce_cik(cls, v: object) -> int:
        return int(str(v).strip().lstrip("0") or 0)

    @model_validator(mode="after")
    def _check_dating(self) -> LabelRecord:
        if self.as_of_date < self.event_date:
            raise ValueError(
                f"{self.company} ({self.cik}): as_of_date {self.as_of_date} precedes "
                f"event_date {self.event_date}. A label cannot be public before it happens."
            )
        return self

    @property
    def lag_days(self) -> int:
        """Disclosure lag: event to public knowledge."""
        return (self.as_of_date - self.event_date).days


def _parse_date(s: str | None) -> date | None:
    s = (s or "").strip()
    if not s:
        return None
    return datetime.strptime(s, "%Y-%m-%d").date()


def _load_csv(path: Path, event_type: str) -> list[LabelRecord]:
    if not path.exists():
        return []
    out: list[LabelRecord] = []
    with path.open(encoding="utf8", newline="") as fh:
        for row in csv.DictReader(fh):
            if not (row.get("cik") or "").strip():
                continue
            out.append(
                LabelRecord(
                    cik=row["cik"],
                    company=row.get("company", ""),
                    event_type=row.get("event_type") or event_type,
                    event_date=_parse_date(row["event_date"]),
                    as_of_date=_parse_date(row.get("as_of_date")) or _parse_date(row["event_date"]),
                    date_basis=row.get("date_basis", ""),
                    source_accession=row.get("source_accession") or None,
                    source_url=row.get("source_url") or None,
                    cohort=row.get("cohort", ""),
                    verification=row.get("verification", ""),
                    misstatement_start=_parse_date(row.get("misstatement_start")),
                    misstatement_end=_parse_date(row.get("misstatement_end")),
                )
            )
    return sorted(out, key=lambda r: r.event_date, reverse=True)


def load_chapter11_labels(path: Path | str = CHAPTER11_CSV) -> list[LabelRecord]:
    """Chapter 11 petitions, verified against 8-K Item 1.03 + filing text."""
    return _load_csv(Path(path), EVENT_CHAPTER11)


def load_aaer_labels(path: Path | str = AAER_CSV) -> list[LabelRecord]:
    """SEC Accounting & Auditing Enforcement Releases.

    Not yet populated -- see ``data/labels/aaer.csv``. The earnings-quality leg
    is gated behind the distress leg's calibration curve (PROMPT hard rule 8),
    so this returning ``[]`` is the expected Phase 1-2 state, not a bug.
    """
    return _load_csv(Path(path), EVENT_AAER)


def labels_visible_as_of(labels: Iterable[LabelRecord], as_of: date) -> list[LabelRecord]:
    """Labels publicly knowable strictly before ``as_of``.

    For labels used as agent *inputs*. Mirrors ``facts.visible_as_of``.
    """
    return [x for x in labels if x.as_of_date < as_of]


def outcome_within_horizon(
    labels: Sequence[LabelRecord],
    cik: int,
    prediction_date: date,
    horizon_days: int,
    event_type: str = EVENT_CHAPTER11,
) -> LabelRecord | None:
    """The event for ``cik`` in ``(prediction_date, prediction_date + horizon]``.

    LOOKS AHEAD BY DESIGN. This is the backtest grader. Never expose its result
    to the agent or to anything the agent can read.
    """
    end = prediction_date + timedelta(days=horizon_days)
    hits = [
        x
        for x in labels
        if x.cik == cik
        and x.event_type == event_type
        and prediction_date < x.event_date <= end
    ]
    return min(hits, key=lambda x: x.event_date) if hits else None


def lead_time_days(prediction_date: date, event_date: date) -> int:
    """Days from a prediction to the event it anticipated. Negative if after."""
    return (event_date - prediction_date).days


def labelled_ciks(labels: Iterable[LabelRecord]) -> set[int]:
    return {x.cik for x in labels}
