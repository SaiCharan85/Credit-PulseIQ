"""L0 -- as-of dating and lookahead controls.

Lookahead leakage is the cardinal sin (PROMPT hard rule 4). These tests pin the
guarantee that everything else rests on: no fact reaches a prediction unless the
filing that reported it was already public.
"""

from __future__ import annotations

from datetime import date

import pytest

from compute.ratios import compute_metric
from data.facts import as_of_view, latest_vintage, visible_as_of
from evals.conftest import DEFAULT_PERIOD_END, DISTRESSED, make_fact, make_facts

MARCH = date(2025, 3, 1)


class TestVisibleAsOf:
    def test_future_filings_are_excluded(self) -> None:
        facts = [
            make_fact("total_assets", 1000.0, filed=date(2025, 2, 20)),
            make_fact("total_assets", 1100.0, filed=date(2025, 6, 1)),
        ]
        visible = visible_as_of(facts, MARCH)
        assert [f.value for f in visible] == [1000.0]

    def test_same_day_filing_is_excluded(self) -> None:
        """Strict ``<``. Intraday timing is not in the data, so a filing made on
        the prediction date cannot be shown to have preceded the prediction.
        The conservative direction is the one that cannot manufacture lookahead.
        """
        facts = [make_fact("total_assets", 1000.0, filed=MARCH)]
        assert visible_as_of(facts, MARCH) == []

    def test_empty_when_nothing_yet_filed(self) -> None:
        facts = make_facts(DISTRESSED, filed=date(2025, 2, 20))
        assert visible_as_of(facts, date(2024, 1, 1)) == []


class TestLatestVintage:
    def test_restatement_wins_over_original(self) -> None:
        original = make_fact("total_assets", 1000.0, filed=date(2025, 2, 20))
        restated = make_fact("total_assets", 900.0, filed=date(2026, 2, 20))
        assert [f.value for f in latest_vintage([original, restated])] == [900.0]

    def test_result_is_order_independent(self) -> None:
        a = make_fact("total_assets", 1000.0, filed=date(2025, 2, 20))
        b = make_fact("total_assets", 900.0, filed=date(2026, 2, 20))
        assert latest_vintage([a, b]) == latest_vintage([b, a])

    def test_ties_broken_deterministically_by_accession(self) -> None:
        a = make_fact("total_assets", 1.0, filed=MARCH, accession="0000000000-00-000001")
        b = make_fact("total_assets", 2.0, filed=MARCH, accession="0000000000-00-000002")
        assert [f.value for f in latest_vintage([a, b])] == [2.0]
        assert latest_vintage([a, b]) == latest_vintage([b, a])

    def test_different_periods_are_kept_separate(self) -> None:
        a = make_fact("total_assets", 1000.0, period_end=date(2023, 12, 31))
        b = make_fact("total_assets", 1100.0, period_end=date(2024, 12, 31))
        assert len(latest_vintage([a, b])) == 2


class TestAsOfView:
    def test_restatement_from_the_future_does_not_leak(self) -> None:
        """The ordering of the two operations is the whole point.

        Filtering *then* selecting the latest vintage yields the number an
        analyst had at the time. Selecting the latest vintage *then* filtering
        would surface a restatement that had not happened yet.
        """
        original = make_fact("total_assets", 1000.0, filed=date(2025, 2, 20))
        restated = make_fact("total_assets", 900.0, filed=date(2026, 2, 20))
        view = as_of_view([original, restated], MARCH)
        assert [f.value for f in view] == [1000.0]

    def test_metric_computed_on_as_of_view_uses_period_appropriate_values(self) -> None:
        facts = make_facts(DISTRESSED, filed=date(2025, 2, 20))
        facts += make_facts({**DISTRESSED, "total_assets": 5000.0}, filed=date(2026, 2, 20))
        cv = compute_metric("debt_to_assets", as_of_view(facts, MARCH), DEFAULT_PERIOD_END)
        assert cv.value == pytest.approx(0.70)  # 700 / 1000, not 700 / 5000

    def test_as_of_never_reaches_forward(self) -> None:
        facts = make_facts(DISTRESSED, filed=date(2025, 2, 20))
        for f in as_of_view(facts, MARCH):
            assert f.filed < MARCH


class TestAnnualPeriodDiscipline:
    def test_quarterly_facts_are_not_mixed_into_annual_metrics(self) -> None:
        """A quarter dropped into an annual series reads as a ~75% collapse."""
        from compute.lineitems import resolve

        quarterly = make_fact("revenue", 500.0)
        quarterly = quarterly.model_copy(
            update={"period_start": DEFAULT_PERIOD_END.replace(month=10, day=1)}
        )
        assert resolve("revenue", [quarterly], DEFAULT_PERIOD_END) is None

    def test_annual_duration_resolves(self) -> None:
        from compute.lineitems import resolve

        annual = make_fact("revenue", 2000.0)
        assert resolve("revenue", [annual], DEFAULT_PERIOD_END) is not None
