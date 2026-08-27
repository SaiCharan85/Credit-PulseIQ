"""L0: report the as-of view, and say when the date has not happened.

Two bugs found by sweeping endpoints rather than by a reader, which is the
order it should happen in.

**The fact count was the whole fetch.** The UI prints it as "figures
available", and it reported 36,659 for a date at which 16,123 were visible.
Twice the truth, on the one number whose entire job is to say how much the
system could see -- and it made two different prediction dates show the same
figure, which quietly contradicts the point of having an as-of date at all.

**A future prediction date passed in silence.** as_of=2030 returned a
confident healthy reading. The staleness note fired but described the newest
filing as "1462 days before the prediction date", which is true and misses the
point: the date has not happened, so nothing was withheld and the as-of filter
did nothing.

Neither is refused -- asking about today is ordinary and the boundary is fuzzy.
Both are reported, which is what the rest of this system does with anything a
reader should weigh.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from serve import FUTURE_DATE_SLACK_DAYS, _future_date_note, _visible_fact_count


class _Fact:
    """Minimal stand-in: the as-of filter reads ``filed``."""

    def __init__(self, filed: date):
        self.filed = filed
        self.cik = 1
        self.taxonomy = "us-gaap"
        self.tag = "Assets"
        self.unit = "USD"
        self.value = 1.0
        self.period_start = None
        self.period_end = filed
        self.fy = None
        self.fp = None
        self.form = "10-K"
        self.accession = "0000000000-00-000000"
        self.frame = None


def test_the_count_is_the_as_of_view_not_the_fetch() -> None:
    """The bug exactly: a filtered count, not the total."""
    facts = [_Fact(date(2020, 1, 1)), _Fact(date(2022, 1, 1)), _Fact(date(2024, 1, 1))]
    assert _visible_fact_count(facts, date(2021, 1, 1)) == 1
    assert _visible_fact_count(facts, date(2023, 1, 1)) == 2
    assert _visible_fact_count(facts, date(2025, 1, 1)) == 3


def test_the_count_moves_when_the_date_moves() -> None:
    """The property that was broken. Two dates showing the same figure is how
    the bug hid: it looked plausible until you compared."""
    facts = [_Fact(date(2018 + i, 6, 1)) for i in range(6)]
    counts = [_visible_fact_count(facts, date(y, 1, 1)) for y in (2019, 2021, 2024)]
    assert counts == sorted(counts)
    assert len(set(counts)) > 1, "the count must depend on the as-of date"


def test_a_broken_filter_degrades_rather_than_failing() -> None:
    """A count is not worth failing an assessment over."""
    assert _visible_fact_count(["not a fact"], date(2024, 1, 1)) == 1


@pytest.mark.parametrize("ahead", [30, 365, 4000])
def test_a_future_date_is_reported(ahead: int) -> None:
    note = _future_date_note(date.today() + timedelta(days=ahead))
    assert note
    assert "future" in note
    assert "as-of filter is doing nothing" in note, (
        "the note must say what is actually wrong -- that nothing was withheld"
    )


@pytest.mark.parametrize("ago", [0, 1, 400, 4000])
def test_today_and_the_past_are_not_flagged(ago: int) -> None:
    """Asking about today is the ordinary case, not a mistake."""
    assert _future_date_note(date.today() - timedelta(days=ago)) == ""


def test_a_day_or_two_ahead_is_tolerated() -> None:
    """A timezone, not a misunderstanding. Flagging it would train readers to
    ignore the flag."""
    assert _future_date_note(date.today() + timedelta(days=FUTURE_DATE_SLACK_DAYS)) == ""
    assert _future_date_note(date.today() + timedelta(days=FUTURE_DATE_SLACK_DAYS + 1))
