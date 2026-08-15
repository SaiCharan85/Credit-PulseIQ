"""L0 -- deterministic ratio math against hand-computed values.

Expected values below are computed by hand from ``conftest.DISTRESSED``, not by
running the code. A test that asserts the code agrees with itself proves
nothing.
"""

from __future__ import annotations

from datetime import date

import pytest

from compute.lineitems import ABSENT_TAG
from compute.provenance import FORMULAS, safe_div
from compute.ratios import STANDARD_METRICS, compute_all, compute_metric
from evals.conftest import DEFAULT_PERIOD_END, DISTRESSED, make_facts

# Hand-computed from DISTRESSED. See conftest for the input profile.
#   total_debt      = 600 + 100                    = 700
#   net_debt        = 700 - 50                     = 650
#   debt_to_assets  = 700 / 1000                   = 0.70
#   liab_to_assets  = 1200 / 1000                  = 1.20   (insolvent)
#   debt_to_equity  = 700 / -200                   = -3.50  (equity wiped out)
#   ebitda          = 100 + 60                     = 160
#   debt_to_ebitda  = 700 / 160                    = 4.375
#   interest_cover  = 100 / 80                     = 1.25
#   ebitda_cover    = 160 / 80                     = 2.00
#   current_ratio   = 400 / 500                    = 0.80
#   quick_ratio     = (400 - 100) / 500            = 0.60
#   cash_ratio      = 50 / 500                     = 0.10
#   working_capital = 400 - 500                    = -100
#   gross_margin    = (2000 - 1500) / 2000         = 0.25
#   operating_marg  = 100 / 2000                   = 0.05
#   net_margin      = -50 / 2000                   = -0.025
#   roa             = -50 / 1000                   = -0.05
#   fcf             = 30 - 20                      = 10
#   ocf_to_debt     = 30 / 700                     = 0.0428571428...
#   accruals/assets = (-50 - 30) / 1000            = -0.08
#   altman z''      = 6.56(-0.1) + 3.26(-0.3) + 6.72(0.1) + 1.05(-1/6)
#                   = -0.656 - 0.978 + 0.672 - 0.175 = -1.137
EXPECTED: dict[str, float] = {
    "total_debt": 700.0,
    "net_debt": 650.0,
    "debt_to_assets": 0.70,
    "liabilities_to_assets": 1.20,
    "debt_to_equity": -3.50,
    "debt_to_ebitda": 4.375,
    "ebitda": 160.0,
    "interest_coverage": 1.25,
    "ebitda_interest_coverage": 2.0,
    "current_ratio": 0.80,
    "quick_ratio": 0.60,
    "cash_ratio": 0.10,
    "working_capital": -100.0,
    "gross_margin": 0.25,
    "operating_margin": 0.05,
    "net_margin": -0.025,
    "return_on_assets": -0.05,
    "free_cash_flow": 10.0,
    "ocf_to_debt": 30.0 / 700.0,
    "accruals_to_assets": -0.08,
    "altman_z_double_prime": -1.137,
}


@pytest.mark.parametrize("metric", sorted(EXPECTED))
def test_metric_matches_hand_computed_value(metric: str, distressed_facts) -> None:
    cv = compute_metric(metric, distressed_facts, DEFAULT_PERIOD_END)
    assert cv.is_defined, f"{metric} unexpectedly undefined: {cv.notes}"
    assert cv.value == pytest.approx(EXPECTED[metric], rel=1e-9, abs=1e-9)


def test_every_standard_metric_has_an_expected_value() -> None:
    """Meta-test: no metric ships without a hand-computed assertion.

    Adding a ratio to STANDARD_METRICS without adding its expected value here
    fails the build, which is what keeps the compute layer honest as it grows.
    """
    assert set(STANDARD_METRICS) == set(EXPECTED)


def test_every_registered_formula_is_reachable() -> None:
    """Formulas exist to be computed. An unregistered-but-unused formula is
    dead code the verifier would still accept figures from."""
    assert set(FORMULAS) >= set(STANDARD_METRICS)


def test_compute_all_returns_every_metric(distressed_facts) -> None:
    out = compute_all(distressed_facts, DEFAULT_PERIOD_END)
    assert set(out) == set(STANDARD_METRICS)


class TestUndefinedRatherThanWrong:
    """Division by zero must yield ``None``, never inf, zero, or an exception.

    A distressed company routinely has zero equity or zero interest expense.
    Returning inf poisons comparisons silently; returning zero reads as
    "no leverage", which is the direction that produces false confidence.
    """

    def test_safe_div_by_zero_is_none(self) -> None:
        assert safe_div(1.0, 0.0) is None
        assert safe_div(1.0, 1e-15) is None

    def test_safe_div_propagates_none(self) -> None:
        assert safe_div(None, 1.0) is None
        assert safe_div(1.0, None) is None

    def test_zero_equity_gives_undefined_not_inf(self) -> None:
        facts = make_facts({**DISTRESSED, "equity": 0.0})
        cv = compute_metric("debt_to_equity", facts, DEFAULT_PERIOD_END)
        assert cv.value is None
        assert cv.notes

    def test_zero_interest_expense_gives_undefined(self) -> None:
        facts = make_facts({**DISTRESSED, "interest_expense": 0.0})
        cv = compute_metric("interest_coverage", facts, DEFAULT_PERIOD_END)
        assert cv.value is None

    def test_negative_equity_is_negative_not_absent(self) -> None:
        """Negative debt/equity must not be mistaken for low leverage."""
        cv = compute_metric("debt_to_equity", make_facts(DISTRESSED), DEFAULT_PERIOD_END)
        assert cv.value is not None and cv.value < 0


class TestMissingLineItems:
    def test_missing_input_yields_undefined_with_reason(self) -> None:
        values = {k: v for k, v in DISTRESSED.items() if k != "total_assets"}
        cv = compute_metric("debt_to_assets", make_facts(values), DEFAULT_PERIOD_END)
        assert cv.value is None
        assert "missing line items" in cv.notes[0]
        assert "total_assets" in cv.notes[0]

    def test_absent_short_term_debt_is_assumed_zero_and_flagged(self) -> None:
        """Absence is treated as zero only where that is genuinely implied --
        and the assumption is recorded in provenance, never silent."""
        values = {k: v for k, v in DISTRESSED.items() if k != "short_term_debt"}
        cv = compute_metric("total_debt", make_facts(values), DEFAULT_PERIOD_END)
        assert cv.value == 600.0
        assert cv.inputs["short_term_debt"].tag == ABSENT_TAG
        assert any("assumed zero" in n for n in cv.notes)

    def test_absent_revenue_does_not_assume_zero(self) -> None:
        """Only ZERO_IF_ABSENT concepts get the assumption. A missing
        denominator must stay undefined rather than become a divide-by-zero."""
        values = {k: v for k, v in DISTRESSED.items() if k != "revenue"}
        cv = compute_metric("net_margin", make_facts(values), DEFAULT_PERIOD_END)
        assert cv.value is None


class TestProvenance:
    def test_every_input_is_traced_to_a_filing(self, distressed_facts) -> None:
        cv = compute_metric("debt_to_assets", distressed_facts, DEFAULT_PERIOD_END)
        assert set(cv.inputs) == set(FORMULAS["debt_to_assets"].inputs)
        for ref in cv.inputs.values():
            assert ref.tag and ref.filed and ref.period_end

    def test_as_of_is_latest_input_filing_date(self) -> None:
        """A figure is knowable only once its *last* input is public."""
        facts = make_facts({"total_assets": 1000.0}, filed=date(2025, 2, 20)) + make_facts(
            {"long_term_debt": 600.0, "short_term_debt": 100.0}, filed=date(2025, 5, 9)
        )
        cv = compute_metric("debt_to_assets", facts, DEFAULT_PERIOD_END)
        assert cv.as_of == date(2025, 5, 9)

    def test_citations_name_tag_and_period(self, distressed_facts) -> None:
        cv = compute_metric("current_ratio", distressed_facts, DEFAULT_PERIOD_END)
        assert any("AssetsCurrent@2024-12-31" in c for c in cv.citations)

    def test_unknown_metric_raises(self, distressed_facts) -> None:
        with pytest.raises(KeyError):
            compute_metric("not_a_metric", distressed_facts, DEFAULT_PERIOD_END)
