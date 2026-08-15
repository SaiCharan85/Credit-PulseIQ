"""L0/L2 -- peer-group construction and peer-relative percentiles."""

from __future__ import annotations

from datetime import date

import pytest

from compute.peers import (
    DEFAULT_MIN_PEERS,
    LEVEL_INSUFFICIENT,
    LEVEL_SIC2,
    LEVEL_SIC3,
    LEVEL_SIC4,
    PeerGroup,
    build_peer_group,
    compare_to_peers,
    median,
    normalize_sic,
    percentile_rank,
)
from compute.provenance import UNIT_RATIO, ComputedValue, FactRef

PERIOD = date(2024, 12, 31)
FILED = date(2025, 2, 20)
AS_OF = date(2025, 6, 1)


def cv(value: float | None, filed: date = FILED, metric: str = "debt_to_assets") -> ComputedValue:
    return ComputedValue(
        metric=metric,
        formula=metric,
        value=value,
        unit=UNIT_RATIO,
        period_end=PERIOD,
        inputs={
            "total_assets": FactRef(
                concept="total_assets", tag="Assets", value=1000.0, period_end=PERIOD, filed=filed
            )
        },
    )


class TestNormalizeSic:
    def test_leading_zeros_restored(self) -> None:
        """SIC codes lose leading zeros when stored as integers -- agriculture
        (0100) becomes 100 and would never match a prefix."""
        assert normalize_sic(100) == "0100"
        assert normalize_sic("5331") == "5331"
        assert normalize_sic(None) == ""


class TestPeerGroupWidening:
    SIC = {1: "5331", 2: "5331", 3: "5331", 4: "5331", 5: "5912", 6: "2821"}

    def test_exact_sic4_group_preferred(self) -> None:
        g = build_peer_group(1, self.SIC)
        assert g.level == LEVEL_SIC4
        assert g.members == [2, 3, 4]

    def test_target_excluded_from_own_group(self) -> None:
        assert 1 not in build_peer_group(1, self.SIC).members

    def test_widens_to_sic3_when_thin(self) -> None:
        sic = {1: "5331", 2: "5332", 3: "5333", 4: "5334", 5: "2821"}
        g = build_peer_group(1, sic)
        assert g.level == LEVEL_SIC3
        assert g.sic_prefix == "533"
        assert g.notes and "widened" in g.notes[0]

    def test_widens_to_sic2_when_still_thin(self) -> None:
        # Same division (53) but no shared 3- or 4-digit prefix.
        sic = {1: "5331", 2: "5399", 3: "5312", 4: "5340", 5: "2821"}
        g = build_peer_group(1, sic)
        assert g.level == LEVEL_SIC2
        assert g.sic_prefix == "53"
        assert g.members == [2, 3, 4]

    def test_no_peers_in_division_is_insufficient(self) -> None:
        """Widening stops at the 2-digit division. A retailer is not comparable
        to a chemicals producer just because the universe is small."""
        sic = {1: "5331", 2: "5411", 3: "5651", 4: "5912", 5: "2821"}
        assert build_peer_group(1, sic).level == LEVEL_INSUFFICIENT

    def test_insufficient_when_universe_too_small(self) -> None:
        """A percentile against two peers is noise. Say so rather than
        publishing a statistic that cannot support the weight."""
        g = build_peer_group(1, {1: "5331", 2: "5331"})
        assert g.level == LEVEL_INSUFFICIENT
        assert not g.is_usable

    def test_missing_sic_is_insufficient(self) -> None:
        assert not build_peer_group(1, {1: None, 2: "5331", 3: "5331", 4: "5331"}).is_usable


class TestPercentileRank:
    def test_mid_rank_convention_for_ties(self) -> None:
        """Ties get half credit so identical values do not land on 0 or 100
        depending on comparison order."""
        assert percentile_rank(5.0, [5.0, 5.0, 5.0, 5.0]) == pytest.approx(50.0)

    def test_lowest_and_highest(self) -> None:
        assert percentile_rank(0.0, [1.0, 2.0, 3.0]) == pytest.approx(0.0)
        assert percentile_rank(9.0, [1.0, 2.0, 3.0]) == pytest.approx(100.0)

    def test_middle_value(self) -> None:
        assert percentile_rank(2.0, [1.0, 3.0]) == pytest.approx(50.0)

    def test_empty_peers_is_none(self) -> None:
        assert percentile_rank(1.0, []) is None

    def test_median_even_and_odd(self) -> None:
        assert median([1.0, 2.0, 3.0]) == 2.0
        assert median([1.0, 2.0, 3.0, 4.0]) == 2.5
        assert median([]) is None


class TestVintagePinning:
    """Peer values must predate the prediction, or "peer-relative" becomes the
    back door lookahead walks through."""

    GROUP = PeerGroup(target_cik=1, members=[2, 3, 4], level=LEVEL_SIC4, sic_prefix="5331")

    def test_peer_not_yet_public_is_dropped(self) -> None:
        peers = {2: cv(0.1), 3: cv(0.2), 4: cv(0.3, filed=date(2025, 9, 1))}
        result = compare_to_peers("debt_to_assets", cv(0.9), peers, self.GROUP, as_of=AS_OF)
        assert result.peer_count == 2
        assert any("not yet public" in n for n in result.notes)

    def test_percentile_withheld_when_too_few_peers_survive(self) -> None:
        peers = {2: cv(0.1), 3: cv(0.2, filed=date(2025, 9, 1)), 4: cv(0.3, filed=date(2025, 9, 1))}
        result = compare_to_peers("debt_to_assets", cv(0.9), peers, self.GROUP, as_of=AS_OF)
        assert result.percentile is None
        assert not result.is_usable

    def test_all_visible_peers_are_ranked(self) -> None:
        peers = {2: cv(0.1), 3: cv(0.2), 4: cv(0.3)}
        result = compare_to_peers("debt_to_assets", cv(0.9), peers, self.GROUP, as_of=AS_OF)
        assert result.peer_count == 3
        assert result.percentile == pytest.approx(100.0)
        assert result.peer_median == pytest.approx(0.2)

    def test_undefined_target_yields_no_percentile(self) -> None:
        peers = {2: cv(0.1), 3: cv(0.2), 4: cv(0.3)}
        result = compare_to_peers("debt_to_assets", cv(None), peers, self.GROUP, as_of=AS_OF)
        assert result.percentile is None
        assert any("undefined" in n for n in result.notes)

    def test_undefined_peers_are_skipped(self) -> None:
        peers = {2: cv(0.1), 3: cv(None), 4: cv(0.3)}
        result = compare_to_peers("debt_to_assets", cv(0.2), peers, self.GROUP, as_of=AS_OF)
        assert result.peer_count == 2

    def test_unusable_group_never_produces_a_percentile(self) -> None:
        thin = PeerGroup(target_cik=1, members=[2], level=LEVEL_INSUFFICIENT)
        result = compare_to_peers("debt_to_assets", cv(0.5), {2: cv(0.1)}, thin, as_of=AS_OF)
        assert result.percentile is None


class TestMinPeersContract:
    def test_default_minimum_is_three(self) -> None:
        assert DEFAULT_MIN_PEERS == 3
