"""L2: peer-group construction and trend correctness, at the tool boundary.

L0 tests the pure functions on synthetic inputs. L2 tests what the *agent*
actually gets: peer comparison and trends assembled from several filers'
real fact sets, through ``ToolBox``, where the pieces meet.

The leakage tests are the reason this level exists. ``as_of`` is enforced in
one place for the subject filer, and it is easy to assume that covers a peer
comparison -- it does not, because peer facts arrive through a separate path.
A peer's figures filed after the prediction date would leak the future into
the subject's percentile just as surely as the subject's own would, and no
test above this level would notice.
"""

from __future__ import annotations

from datetime import date

import pytest

from agents.tools import ToolBox
from compute.peers import (
    LEVEL_INSUFFICIENT,
    LEVEL_SIC2,
    LEVEL_SIC4,
    build_peer_group,
)
from compute.trends import (
    DIRECTION_DETERIORATING,
    DIRECTION_IMPROVING,
    build_trend,
)
from data.facts import XbrlFact
from evals.conftest import make_fact

AS_OF = date(2024, 6, 1)


def fact(concept: str, value: float, end: date, filed: date, cik: int = 1) -> XbrlFact:
    """Canonical-concept fact. Delegates so the instant/duration handling and
    tag resolution stay in one place -- a hand-rolled copy drifts."""
    return make_fact(concept, value, end, filed, cik)


def balance_sheet(cik: int, end: date, filed: date, current_assets: float) -> list[XbrlFact]:
    """Minimum facts for current_ratio, plus a duration fact.

    An annual period end requires *both* a balance sheet (``total_assets``) and
    an income statement (``revenue`` or ``net_income``). A fact set carrying
    only the two ratio inputs yields no periods at all, and every lookup then
    returns "no visible period" rather than anything about the ratio.
    """
    return [
        fact("current_assets", current_assets, end, filed, cik),
        fact("current_liabilities", 100.0, end, filed, cik),
        fact("total_assets", 1000.0, end, filed, cik),
        fact("revenue", 1000.0, end, filed, cik),
    ]


class TestPeerFactsRespectAsOf:
    """A peer's future filing must not reach the subject's percentile."""

    def test_a_peer_figure_filed_after_as_of_is_excluded(self) -> None:
        subject = balance_sheet(1, date(2023, 12, 31), date(2024, 3, 1), 50.0)
        # Each peer has an old, visible figure and a newer one filed after AS_OF.
        peers = {}
        for cik in (2, 3, 4):
            peers[cik] = balance_sheet(cik, date(2023, 12, 31), date(2024, 3, 1), 200.0) + \
                balance_sheet(cik, date(2024, 3, 31), date(2024, 9, 1), 1.0)
        sic = {1: "3578", 2: "3578", 3: "3578", 4: "3578"}
        tb = ToolBox(cik=1, as_of=AS_OF, facts=subject, peer_facts=peers, sic_by_cik=sic)
        result = tb.get_peer_comparison("current_ratio")
        # Visible peer current_ratio is 2.0; the leaked one would be 0.01.
        assert result["peer_median"] == pytest.approx(2.0)

    def test_a_peer_with_only_future_filings_is_skipped_not_zeroed(self) -> None:
        """Absent must not be read as a peer whose ratio is zero."""
        subject = balance_sheet(1, date(2023, 12, 31), date(2024, 3, 1), 50.0)
        peers = {
            2: balance_sheet(2, date(2023, 12, 31), date(2024, 3, 1), 200.0),
            3: balance_sheet(3, date(2023, 12, 31), date(2024, 3, 1), 200.0),
            4: balance_sheet(4, date(2024, 3, 31), date(2024, 9, 1), 200.0),
        }
        sic = {c: "3578" for c in (1, 2, 3, 4)}
        tb = ToolBox(cik=1, as_of=AS_OF, facts=subject, peer_facts=peers, sic_by_cik=sic)
        result = tb.get_peer_comparison("current_ratio")
        assert result["peer_count"] == 2

    def test_as_of_is_not_a_peer_tool_argument(self) -> None:
        from agents.tools import tool_schemas

        schema = next(
            t for t in tool_schemas() if t["function"]["name"] == "get_peer_comparison"
        )
        assert "as_of" not in schema["function"]["parameters"]["properties"]


class TestPeerGroupConstruction:
    def test_the_subject_is_never_its_own_peer(self) -> None:
        sic = {1: "3578", 2: "3578", 3: "3578", 4: "3578"}
        assert 1 not in build_peer_group(1, sic).members

    def test_exact_sic4_is_preferred(self) -> None:
        sic = {1: "3578", 2: "3578", 3: "3578", 4: "3578", 5: "3500"}
        group = build_peer_group(1, sic)
        assert group.level == LEVEL_SIC4
        assert 5 not in group.members

    def test_it_widens_when_too_few_exact_matches(self) -> None:
        """Real SIC assignment is uneven, so exact matching leaves groups empty."""
        sic = {1: "3578", 2: "3500", 3: "3510", 4: "3520"}
        group = build_peer_group(1, sic)
        assert group.level == LEVEL_SIC2
        assert group.notes and "widened" in group.notes[0]

    def test_widening_is_recorded_so_it_can_be_discounted(self) -> None:
        sic = {1: "3578", 2: "3500", 3: "3510", 4: "3520"}
        assert build_peer_group(1, sic).sic_prefix == "35"

    def test_a_filer_with_no_sic_gets_no_group(self) -> None:
        group = build_peer_group(1, {1: None, 2: "3578", 3: "3578", 4: "3578"})
        assert group.level == LEVEL_INSUFFICIENT
        assert group.members == []

    def test_explicit_exclusions_are_honoured(self) -> None:
        sic = {c: "3578" for c in (1, 2, 3, 4, 5)}
        assert 2 not in build_peer_group(1, sic, exclude=[2]).members


class TestPeerComparisonUsability:
    def test_a_thin_group_is_marked_unusable(self) -> None:
        """Two peers is not a distribution, and must not be reported as one."""
        subject = balance_sheet(1, date(2023, 12, 31), date(2024, 3, 1), 50.0)
        peers = {2: balance_sheet(2, date(2023, 12, 31), date(2024, 3, 1), 200.0)}
        sic = {1: "3578", 2: "3578"}
        tb = ToolBox(cik=1, as_of=AS_OF, facts=subject, peer_facts=peers, sic_by_cik=sic)
        result = tb.get_peer_comparison("current_ratio")
        assert result["usable"] is False

    def test_absent_peer_data_is_an_error_not_an_empty_distribution(self) -> None:
        subject = balance_sheet(1, date(2023, 12, 31), date(2024, 3, 1), 50.0)
        tb = ToolBox(cik=1, as_of=AS_OF, facts=subject)
        assert "error" in tb.get_peer_comparison("current_ratio")

    def test_percentile_places_the_subject_correctly(self) -> None:
        subject = balance_sheet(1, date(2023, 12, 31), date(2024, 3, 1), 10.0)
        peers = {
            c: balance_sheet(c, date(2023, 12, 31), date(2024, 3, 1), v)
            for c, v in ((2, 200.0), (3, 300.0), (4, 400.0))
        }
        sic = {c: "3578" for c in (1, 2, 3, 4)}
        tb = ToolBox(cik=1, as_of=AS_OF, facts=subject, peer_facts=peers, sic_by_cik=sic)
        result = tb.get_peer_comparison("current_ratio")
        assert result["percentile"] == pytest.approx(0.0)


class TestTrendsOverRealSequences:
    def _series(self, values: list[tuple[date, float]], filed_offset_days: int = 60):
        from datetime import timedelta

        facts: list[XbrlFact] = []
        for end, v in values:
            filed = end + timedelta(days=filed_offset_days)
            facts += [
                fact("current_assets", v, end, filed),
                fact("current_liabilities", 100.0, end, filed),
                fact("total_assets", 1000.0, end, filed),
                fact("revenue", 1000.0, end, filed),
            ]
        return facts

    def test_direction_follows_higher_is_better(self) -> None:
        ends = [date(2021, 12, 31), date(2022, 12, 31), date(2023, 12, 31)]
        rising = self._series(list(zip(ends, [100.0, 150.0, 200.0], strict=True)))
        assert build_trend("current_ratio", rising, ends).direction == DIRECTION_IMPROVING
        falling = self._series(list(zip(ends, [200.0, 150.0, 100.0], strict=True)))
        assert build_trend("current_ratio", falling, ends).direction == DIRECTION_DETERIORATING

    def test_points_are_ordered_oldest_first_regardless_of_input_order(self) -> None:
        ends = [date(2023, 12, 31), date(2021, 12, 31), date(2022, 12, 31)]
        facts = self._series([(date(2021, 12, 31), 100.0), (date(2022, 12, 31), 150.0),
                              (date(2023, 12, 31), 200.0)])
        trend = build_trend("current_ratio", facts, ends)
        assert [p.period_end for p in trend.points] == sorted(ends)

    def test_a_gap_is_reported_not_interpolated(self) -> None:
        """A missing period must show as undefined, never smoothed over."""
        ends = [date(2021, 12, 31), date(2022, 12, 31), date(2023, 12, 31)]
        facts = self._series([(date(2021, 12, 31), 100.0), (date(2023, 12, 31), 200.0)])
        trend = build_trend("current_ratio", facts, ends)
        assert any(not p.is_defined for p in trend.points)
        assert trend.notes and "undefined" in trend.notes[0]

    def test_a_restated_period_uses_the_later_filing(self) -> None:
        """Two filings for one period end: the amended one is what a reader sees."""
        from datetime import timedelta

        end = date(2022, 12, 31)
        facts = [
            fact("current_assets", 100.0, end, end + timedelta(days=60)),
            fact("current_liabilities", 100.0, end, end + timedelta(days=60)),
            fact("total_assets", 1000.0, end, end + timedelta(days=60)),
            fact("revenue", 1000.0, end, end + timedelta(days=60)),
            fact("current_assets", 500.0, end, end + timedelta(days=400)),
            fact("current_liabilities", 100.0, end, end + timedelta(days=400)),
        ]
        trend = build_trend("current_ratio", facts, [end])
        assert trend.points[0].value == pytest.approx(5.0)
