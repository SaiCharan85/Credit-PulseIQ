"""L1 -- XBRL extraction accuracy against real filings.

Fixtures are trimmed but otherwise untouched SEC ``companyfacts`` payloads for
Sleep Number (CIK 827187, Chapter 11 on 2026-06-12) and its sector peer
La-Z-Boy (CIK 57131, SIC 2510, still filing).

Values asserted here were read from the filings, not produced by this code.
L0 proves the arithmetic; L1 proves we are feeding it the right numbers.
"""

from __future__ import annotations

from datetime import date

import pytest

from compute.lineitems import annual_period_ends, period_ends, resolve
from compute.peers import build_peer_group, compare_to_peers
from compute.ratios import compute_all, compute_metric
from compute.trends import DIRECTION_DETERIORATING, build_trend
from data.facts import as_of_view
from verify.recompute import verify

# Sleep Number FY2025 (52/53-week retail year ending 2026-01-03), from the
# 10-K filed 2026-05-12 / 2026-03-12.
SN_FY2025 = date(2026, 1, 3)
SN_EXPECTED = {
    "total_assets": 680_060_000.0,
    "total_liabilities": 1_258_535_000.0,
    "current_assets": 149_957_000.0,
    "current_liabilities": 912_546_000.0,
    "cash": 1_693_000.0,
    "inventory": 82_233_000.0,
    "equity": -578_475_000.0,
    "short_term_debt": 588_200_000.0,
    "retained_earnings": -611_158_000.0,
    "revenue": 1_411_450_000.0,
    "operating_income": -46_592_000.0,
    "net_income": -132_000_000.0,
    "cash_from_operations": -3_283_000.0,
}

LZB_FY2026 = date(2026, 4, 25)


class TestExtractionAccuracy:
    @pytest.mark.parametrize("concept,expected", sorted(SN_EXPECTED.items()))
    def test_line_item_matches_filing(self, sleep_number_facts, concept, expected) -> None:
        ref = resolve(concept, sleep_number_facts, SN_FY2025)
        assert ref is not None, f"{concept} did not resolve"
        assert ref.value == pytest.approx(expected)

    def test_every_fact_carries_a_filing_date(self, sleep_number_facts) -> None:
        """No fact without a filed date -- that date is the as-of control."""
        assert sleep_number_facts
        for f in sleep_number_facts:
            assert f.filed is not None and f.accession

    def test_provenance_names_the_real_accession(self, sleep_number_facts) -> None:
        ref = resolve("total_assets", sleep_number_facts, SN_FY2025)
        assert ref.tag == "Assets"
        assert ref.accession.startswith("0000")
        assert ref.form in {"10-K", "10-Q", "8-K", "20-F"}


class TestTagFallbackChains:
    def test_falls_back_when_primary_tag_absent(self, sleep_number_facts) -> None:
        """Neither company tags an annual ``InterestExpense``; both fall through
        to ``InterestExpenseNonoperating``. Filers do not agree on tags, which
        is why resolution walks a chain and records what it actually read."""
        ref = resolve("interest_expense", sleep_number_facts, SN_FY2025)
        assert ref is not None
        assert ref.tag == "InterestExpenseNonoperating"

    def test_peer_uses_a_different_receivables_tag(self, la_z_boy_facts) -> None:
        ref = resolve("receivables", la_z_boy_facts, LZB_FY2026)
        assert ref is not None and ref.tag == "ReceivablesNetCurrent"


class TestRealWorldCoverageGaps:
    """Gaps found in real filings, pinned so a regression is visible."""

    def test_distressed_filer_reports_no_long_term_debt(self, sleep_number_facts) -> None:
        """By FY2025 Sleep Number's debt is entirely current -- the
        reclassification that happens when maturities accelerate."""
        assert resolve("long_term_debt", sleep_number_facts, SN_FY2025) is None
        assert resolve("short_term_debt", sleep_number_facts, SN_FY2025) is not None

    def test_leverage_still_computes_via_assumed_zero(self, sleep_number_facts) -> None:
        """Total debt = 0 + 588.2m. The assumption is recorded, not silent."""
        cv = compute_metric("total_debt", sleep_number_facts, SN_FY2025)
        assert cv.value == pytest.approx(588_200_000.0)
        assert any("long_term_debt" in n for n in cv.notes)

    def test_debt_is_never_assumed_away_entirely(self, la_z_boy_facts) -> None:
        """La-Z-Boy tags neither debt concept at FY2026. Assuming both zero
        would report an unlevered balance sheet; undefined is the honest
        answer."""
        assert resolve("long_term_debt", la_z_boy_facts, LZB_FY2026) is None
        assert resolve("short_term_debt", la_z_boy_facts, LZB_FY2026) is None
        cv = compute_metric("total_debt", la_z_boy_facts, LZB_FY2026)
        assert cv.value is None
        assert "missing line items" in cv.notes[0]

    def test_filer_that_does_not_tag_total_liabilities(self, la_z_boy_facts) -> None:
        """La-Z-Boy never tags ``Liabilities``. Deriving it as assets minus
        equity would be wrong in the presence of noncontrolling interests, so
        dependent metrics report undefined with a reason instead.

        Known limitation: this costs Altman Z'' coverage on such filers.
        """
        assert resolve("total_liabilities", la_z_boy_facts, LZB_FY2026) is None
        cv = compute_metric("liabilities_to_assets", la_z_boy_facts, LZB_FY2026)
        assert cv.value is None
        assert "total_liabilities" in cv.notes[0]


class TestPeriodDiscovery:
    def test_annual_period_ends_are_descending_and_annual(self, sleep_number_facts) -> None:
        ends = annual_period_ends(sleep_number_facts, limit=5)
        assert ends == sorted(ends, reverse=True)
        assert SN_FY2025 in ends

    def test_fiscal_years_are_not_calendar_years(self, sleep_number_facts) -> None:
        """A 52/53-week retail year ends on 2026-01-03. Assuming December 31
        would silently drop the most important period."""
        assert SN_FY2025 not in {date(y, 12, 31) for y in range(2018, 2027)}

    def test_balance_sheet_periods_exist(self, sleep_number_facts) -> None:
        assert period_ends(sleep_number_facts, "total_assets")


class TestRealDistressSignals:
    """The FY2025 filing should look like what it was: a company months from
    Chapter 11."""

    def test_equity_is_negative(self, sleep_number_facts) -> None:
        cv = compute_metric("liabilities_to_assets", sleep_number_facts, SN_FY2025)
        assert cv.value > 1.0  # liabilities exceed assets

    def test_current_ratio_far_below_one(self, sleep_number_facts) -> None:
        cv = compute_metric("current_ratio", sleep_number_facts, SN_FY2025)
        assert cv.value == pytest.approx(149_957_000.0 / 912_546_000.0)
        assert cv.value < 0.2

    def test_operating_income_does_not_cover_interest(self, sleep_number_facts) -> None:
        cv = compute_metric("interest_coverage", sleep_number_facts, SN_FY2025)
        assert cv.value < 0  # operating loss

    def test_altman_z_in_distress_zone(self, sleep_number_facts) -> None:
        cv = compute_metric("altman_z_double_prime", sleep_number_facts, SN_FY2025)
        assert cv.is_defined and cv.value < 1.1

    def test_peer_is_not_in_distress(self, la_z_boy_facts) -> None:
        """Control: the same code on a healthy filer must not flag distress."""
        assert compute_metric("current_ratio", la_z_boy_facts, LZB_FY2026).value > 1.5
        assert compute_metric("net_margin", la_z_boy_facts, LZB_FY2026).value > 0

    def test_leverage_deteriorates_over_time(self, sleep_number_facts) -> None:
        ends = sorted(annual_period_ends(sleep_number_facts))[-4:]
        trend = build_trend("liabilities_to_assets", sleep_number_facts, ends)
        assert trend.direction == DIRECTION_DETERIORATING
        assert trend.consecutive_deteriorations >= 1


class TestEndToEndOnRealData:
    def test_every_computed_metric_verifies(self, sleep_number_facts) -> None:
        values = compute_all(sleep_number_facts, SN_FY2025)
        report = verify(values.values())
        assert report.passed, report.defect_summary()

    def test_as_of_view_excludes_the_final_filing(self, sleep_number_facts) -> None:
        """Standing at 2026-03-01, the 10-K filed 2026-05-12 must be invisible."""
        cutoff = date(2026, 3, 1)
        view = as_of_view(sleep_number_facts, cutoff)
        assert view and all(f.filed < cutoff for f in view)
        assert len(view) < len(sleep_number_facts)

    def test_peer_comparison_between_real_filers(self, sleep_number_facts, la_z_boy_facts) -> None:
        """Two-company universe is deliberately too thin: the percentile must
        be withheld rather than computed against a single peer."""
        sic = {827187: "2510", 57131: "2510"}
        group = build_peer_group(827187, sic)
        target = compute_metric("current_ratio", sleep_number_facts, SN_FY2025)
        peer = compute_metric("current_ratio", la_z_boy_facts, LZB_FY2026)
        result = compare_to_peers("current_ratio", target, {57131: peer}, group)
        assert result.percentile is None
        assert not group.is_usable
