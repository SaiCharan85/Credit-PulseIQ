"""L0 -- trend correctness (also L2: trend construction over real periods)."""

from __future__ import annotations

from datetime import date

import pytest

from compute.trends import (
    DIRECTION_DETERIORATING,
    DIRECTION_FLAT,
    DIRECTION_IMPROVING,
    DIRECTION_UNKNOWN,
    build_trend,
)
from evals.conftest import DISTRESSED, make_facts

PERIODS = [date(y, 12, 31) for y in (2021, 2022, 2023, 2024)]


def facts_for(series: dict[str, list[float]]) -> list:
    """Build facts across PERIODS from per-concept value series."""
    out = []
    for i, pe in enumerate(PERIODS):
        values = {**DISTRESSED}
        for concept, vals in series.items():
            values[concept] = vals[i]
        out += make_facts(values, period_end=pe, filed=date(pe.year + 1, 2, 20))
    return out


class TestDirection:
    def test_rising_leverage_is_deterioration(self) -> None:
        """Sign correction matters: for leverage, up is bad."""
        facts = facts_for({"long_term_debt": [200.0, 400.0, 600.0, 800.0]})
        trend = build_trend("debt_to_assets", facts, PERIODS)
        assert trend.direction == DIRECTION_DETERIORATING
        assert trend.values == pytest.approx([0.3, 0.5, 0.7, 0.9])

    def test_falling_coverage_is_deterioration(self) -> None:
        facts = facts_for({"operating_income": [400.0, 300.0, 200.0, 100.0]})
        trend = build_trend("interest_coverage", facts, PERIODS)
        assert trend.direction == DIRECTION_DETERIORATING

    def test_rising_coverage_is_improvement(self) -> None:
        facts = facts_for({"operating_income": [100.0, 200.0, 300.0, 400.0]})
        assert build_trend("interest_coverage", facts, PERIODS).direction == DIRECTION_IMPROVING

    def test_flat_series_is_flat(self) -> None:
        facts = facts_for({"operating_income": [100.0] * 4})
        assert build_trend("interest_coverage", facts, PERIODS).direction == DIRECTION_FLAT

    def test_single_point_direction_is_unknown(self) -> None:
        facts = facts_for({"operating_income": [100.0] * 4})
        assert build_trend("interest_coverage", facts, PERIODS[:1]).direction == DIRECTION_UNKNOWN

    def test_direction_reflects_endpoints_not_average_slope(self) -> None:
        """A series that dips and recovers to above its start is improving,
        even though a naive mid-series read would call it deterioration."""
        facts = facts_for({"operating_income": [200.0, 50.0, 60.0, 400.0]})
        assert build_trend("interest_coverage", facts, PERIODS).direction == DIRECTION_IMPROVING


class TestOrdering:
    def test_points_are_ordered_oldest_first_regardless_of_input_order(self) -> None:
        facts = facts_for({"long_term_debt": [200.0, 400.0, 600.0, 800.0]})
        trend = build_trend("debt_to_assets", facts, list(reversed(PERIODS)))
        assert trend.period_ends == PERIODS
        assert trend.values == pytest.approx([0.3, 0.5, 0.7, 0.9])


class TestConsecutiveDeteriorations:
    def test_unbroken_slide_counts_every_step(self) -> None:
        facts = facts_for({"long_term_debt": [200.0, 400.0, 600.0, 800.0]})
        assert build_trend("debt_to_assets", facts, PERIODS).consecutive_deteriorations == 3

    def test_run_counts_only_from_the_most_recent_end(self) -> None:
        """Three bad years with a recovery is a different signal from an
        unbroken three-year slide."""
        facts = facts_for({"long_term_debt": [800.0, 600.0, 400.0, 600.0]})
        assert build_trend("debt_to_assets", facts, PERIODS).consecutive_deteriorations == 1

    def test_improving_series_has_no_run(self) -> None:
        facts = facts_for({"long_term_debt": [800.0, 600.0, 400.0, 200.0]})
        assert build_trend("debt_to_assets", facts, PERIODS).consecutive_deteriorations == 0


class TestSummaryStatistics:
    def test_change_abs_and_pct(self) -> None:
        facts = facts_for({"long_term_debt": [200.0, 400.0, 600.0, 800.0]})
        trend = build_trend("debt_to_assets", facts, PERIODS)
        assert trend.change_abs == pytest.approx(0.6)  # 0.9 - 0.3
        assert trend.change_pct == pytest.approx(2.0)  # +200%

    def test_slope_is_positive_for_rising_series(self) -> None:
        facts = facts_for({"long_term_debt": [200.0, 400.0, 600.0, 800.0]})
        # values 0.3, 0.5, 0.7, 0.9 -> exactly +0.2 per period
        assert build_trend("debt_to_assets", facts, PERIODS).slope_per_period == pytest.approx(0.2)

    def test_as_of_is_latest_filing_across_all_points(self) -> None:
        facts = facts_for({"long_term_debt": [200.0, 400.0, 600.0, 800.0]})
        assert build_trend("debt_to_assets", facts, PERIODS).as_of == date(2025, 2, 20)


class TestUndefinedPoints:
    def test_undefined_points_are_recorded_not_dropped(self) -> None:
        """A metric that stops computing is a data problem the agent must see,
        not a gap to quietly interpolate over."""
        facts = facts_for({"interest_expense": [80.0, 80.0, 0.0, 80.0]})
        trend = build_trend("interest_coverage", facts, PERIODS)
        assert len(trend.points) == 4
        assert len(trend.defined_points) == 3
        assert trend.coverage == pytest.approx(0.75)
        assert trend.notes and "2023-12-31" in trend.notes[0]

    def test_all_undefined_yields_unknown_direction(self) -> None:
        facts = facts_for({"interest_expense": [0.0] * 4})
        trend = build_trend("interest_coverage", facts, PERIODS)
        assert trend.coverage == 0.0
        assert trend.direction == DIRECTION_UNKNOWN
