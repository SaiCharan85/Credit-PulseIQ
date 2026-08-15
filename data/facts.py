"""Normalized XBRL facts and the as-of dating controls.

This module owns the single most important correctness control in the project:
a fact is only *visible* to a prediction once the filing that reported it was
public. Everything downstream (compute, investigators, backtest) reads facts
through :func:`visible_as_of`, never through a raw EDGAR payload.

Plain deterministic code. No LLM, no agency (SPEC 3, PROMPT hard rule 3).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import date

from pydantic import BaseModel, ConfigDict, Field


class XbrlFact(BaseModel):
    """One reported value for one concept, one period, one filing.

    The same (tag, period) is typically reported many times -- once in the
    original filing and again as a comparative in later filings, sometimes
    restated. Each of those is a separate ``XbrlFact`` with its own ``filed``
    date. Collapsing them loses the vintage information the backtest needs,
    so we keep them all and select explicitly (:func:`latest_vintage`).
    """

    model_config = ConfigDict(frozen=True)

    cik: int
    taxonomy: str
    tag: str
    unit: str
    value: float
    period_start: date | None = None
    period_end: date
    fy: int | None = None
    fp: str | None = None
    form: str
    accession: str
    filed: date = Field(
        ...,
        description="Date the filing became public on EDGAR. This is the as-of date.",
    )
    frame: str | None = None

    @property
    def as_of_date(self) -> date:
        """When this value became knowable. Alias for ``filed``."""
        return self.filed

    @property
    def is_instant(self) -> bool:
        """Balance-sheet (point-in-time) fact rather than a flow over a period."""
        return self.period_start is None

    @property
    def duration_days(self) -> int | None:
        if self.period_start is None:
            return None
        return (self.period_end - self.period_start).days

    @property
    def source_url(self) -> str:
        accn = self.accession.replace("-", "")
        return f"https://www.sec.gov/Archives/edgar/data/{self.cik}/{accn}/"


def visible_as_of(facts: Iterable[XbrlFact], as_of: date) -> list[XbrlFact]:
    """Facts whose filing date strictly precedes ``as_of``.

    Strict ``<`` is deliberate. A filing made *on* the prediction date is not
    treated as available: intraday timing is not in the data, so same-day
    availability cannot be established, and the conservative direction is the
    one that cannot manufacture lookahead.

    This is the cardinal-sin guard (PROMPT hard rule 4). The backtest asserts
    against it; it is not advisory.
    """
    return [f for f in facts if f.filed < as_of]


def latest_vintage(facts: Iterable[XbrlFact]) -> list[XbrlFact]:
    """Collapse restatements: keep the most recently filed value per series.

    Grouped by (tag, unit, period_start, period_end). When two filings report
    the same period, the later filing wins -- that is the best information an
    analyst would have had. Ties on ``filed`` are broken by accession number so
    the result is deterministic rather than dependent on input ordering.

    Call this *after* :func:`visible_as_of`, never before: picking the latest
    vintage over unfiltered facts would select a restatement from the future.
    """
    best: dict[tuple, XbrlFact] = {}
    for f in facts:
        key = (f.tag, f.unit, f.period_start, f.period_end)
        cur = best.get(key)
        if cur is None or (f.filed, f.accession) > (cur.filed, cur.accession):
            best[key] = f
    return sorted(best.values(), key=lambda f: (f.tag, f.period_end))


def as_of_view(facts: Iterable[XbrlFact], as_of: date) -> list[XbrlFact]:
    """The complete point-in-time view: visible facts, latest vintage each.

    This is the only function the investigator's tools should use to reach
    financial data.
    """
    return latest_vintage(visible_as_of(facts, as_of))


def annual_facts(facts: Sequence[XbrlFact], tag: str) -> list[XbrlFact]:
    """Annual-duration facts for a tag, oldest first.

    Filters flow facts to ~1 year (350-380 days) so that quarterly and annual
    values are never mixed into the same trend, which would produce a fake
    75% "decline" every fourth period.
    """
    out = [
        f
        for f in facts
        if f.tag == tag
        and f.duration_days is not None
        and 350 <= f.duration_days <= 380
    ]
    return sorted(out, key=lambda f: f.period_end)


def instant_facts(facts: Sequence[XbrlFact], tag: str) -> list[XbrlFact]:
    """Balance-sheet facts for a tag, oldest first."""
    out = [f for f in facts if f.tag == tag and f.is_instant]
    return sorted(out, key=lambda f: f.period_end)
