"""L0 -- composite scores (runway, working-capital cycle, Piotroski, Beneish, Ohlson).

Expected values are hand-computed from the fixtures below, not produced by
running the code.

For Beneish the test uses a *neutral* company -- every year-on-year input
identical, and net income equal to operating cash flow. Every index collapses
to 1.0 and TATA to 0, so the score reduces to the sum of the published
coefficients and pins the transcription exactly:

    -4.84 + 0.920 + 0.528 + 0.404 + 0.892 + 0.115 - 0.172 + 0 - 0.327 = -2.48
"""

from __future__ import annotations

from datetime import date

import pytest

from compute.provenance import FORMULAS
from compute.ratios import compute_metric
from compute.scores import (
    ALL_SCORES,
    SINGLE_PERIOD_SCORES,
    TWO_PERIOD_SCORES,
    compute_scores,
    compute_two_period_score,
)
from evals.conftest import make_facts
from verify.recompute import verify

CURRENT = date(2024, 12, 31)
PRIOR = date(2023, 12, 31)

#: Current-year profile. Extends conftest.DISTRESSED with the line items the
#: composite scores need.
CUR: dict[str, float] = {
    "total_assets": 1000.0,
    "total_liabilities": 1200.0,
    "current_assets": 400.0,
    "current_liabilities": 500.0,
    "cash": 50.0,
    "inventory": 100.0,
    "receivables": 200.0,
    "accounts_payable": 150.0,
    "ppe_net": 300.0,
    "equity": -200.0,
    "long_term_debt": 600.0,
    "short_term_debt": 100.0,
    "retained_earnings": -300.0,
    "revenue": 2000.0,
    "cost_of_revenue": 1500.0,
    "sga_expense": 250.0,
    "operating_income": 100.0,
    "net_income": -50.0,
    "interest_expense": 80.0,
    "depreciation_amortization": 60.0,
    "cash_from_operations": 30.0,
    "capex": 20.0,
    "shares_outstanding": 1000.0,
}

#: Prior year: healthier on every dimension, so Piotroski scores near zero.
PRIOR_VALUES: dict[str, float] = {
    **CUR,
    "total_assets": 900.0,
    "current_assets": 500.0,
    "current_liabilities": 400.0,
    "long_term_debt": 400.0,
    "revenue": 1800.0,
    "cost_of_revenue": 1200.0,
    "net_income": 100.0,
    "cash_from_operations": 80.0,
    "shares_outstanding": 900.0,
}


def two_period_facts(cur: dict | None = None, prior: dict | None = None):
    return make_facts(cur or CUR, period_end=CURRENT, filed=date(2025, 2, 20)) + make_facts(
        prior or PRIOR_VALUES, period_end=PRIOR, filed=date(2024, 2, 20)
    )


class TestWorkingCapitalCycle:
    """DSO = 200/2000*365 = 36.5;  DIO = 100/1500*365 = 24.3333;
    DPO = 150/1500*365 = 36.5;  CCC = 36.5 + 24.3333 - 36.5 = 24.3333"""

    def test_days_sales_outstanding(self) -> None:
        cv = compute_metric("days_sales_outstanding", make_facts(CUR), CURRENT)
        assert cv.value == pytest.approx(36.5)

    def test_days_inventory_outstanding(self) -> None:
        cv = compute_metric("days_inventory_outstanding", make_facts(CUR), CURRENT)
        assert cv.value == pytest.approx(100.0 / 1500.0 * 365.0)

    def test_days_payables_outstanding(self) -> None:
        cv = compute_metric("days_payables_outstanding", make_facts(CUR), CURRENT)
        assert cv.value == pytest.approx(36.5)

    def test_cash_conversion_cycle(self) -> None:
        cv = compute_metric("cash_conversion_cycle", make_facts(CUR), CURRENT)
        assert cv.value == pytest.approx(100.0 / 1500.0 * 365.0)


class TestCashRunway:
    def test_runway_when_burning_cash(self) -> None:
        """cash 50 / (60/12 per month) = 10 months."""
        facts = make_facts({**CUR, "cash_from_operations": -60.0})
        cv = compute_metric("cash_runway_months", facts, CURRENT)
        assert cv.value == pytest.approx(10.0)

    def test_undefined_when_generating_cash(self) -> None:
        """A company that is not burning has no runway limit. Reporting a huge
        number would read as a quantified strength rather than as inapplicable."""
        cv = compute_metric("cash_runway_months", make_facts(CUR), CURRENT)
        assert cv.value is None

    def test_zero_operating_cash_flow_is_undefined(self) -> None:
        facts = make_facts({**CUR, "cash_from_operations": 0.0})
        assert compute_metric("cash_runway_months", facts, CURRENT).value is None


class TestPiotroski:
    """Hand-scored against CUR vs PRIOR_VALUES:

    ==  =====================================  =====
    1   ROA > 0            -0.05              0
    2   CFO > 0            30                 1
    3   ROA > ROA_prior    -0.05 vs 0.1111    0
    4   CFO > NI           30 > -50           1
    5   LTD/TA fell        0.60 vs 0.4444     0
    6   current ratio rose 0.80 vs 1.25       0
    7   no dilution        1000 vs 900        0
    8   gross margin rose  0.25 vs 0.3333     0
    9   asset turnover     2.00 vs 2.00       0
    ==  =====================================  =====
                                        total  2
    """

    def test_score_matches_hand_count(self) -> None:
        cv = compute_two_period_score("piotroski_f_score", two_period_facts(), CURRENT, PRIOR)
        assert cv.value == pytest.approx(2.0)

    def test_healthy_company_scores_high(self) -> None:
        """Improving on every dimension should score 9/9."""
        prior = {**CUR, "total_assets": 1200.0, "current_assets": 300.0,
                 "current_liabilities": 600.0, "long_term_debt": 900.0,
                 "revenue": 1000.0, "cost_of_revenue": 900.0,
                 "net_income": -200.0, "cash_from_operations": -100.0,
                 "shares_outstanding": 1200.0}
        cur = {**CUR, "net_income": 100.0, "cash_from_operations": 200.0}
        cv = compute_two_period_score(
            "piotroski_f_score", two_period_facts(cur, prior), CURRENT, PRIOR
        )
        assert cv.value == pytest.approx(9.0)

    def test_score_is_bounded(self) -> None:
        cv = compute_two_period_score("piotroski_f_score", two_period_facts(), CURRENT, PRIOR)
        assert 0.0 <= cv.value <= 9.0


class TestBeneish:
    def test_neutral_company_equals_coefficient_sum(self) -> None:
        """Every index 1.0 and TATA 0 -> M = -2.48 exactly.

        This pins the published coefficients: a transcription slip in any one
        of the eight moves the result.
        """
        neutral = {**CUR, "net_income": 30.0, "cash_from_operations": 30.0}
        facts = two_period_facts(neutral, neutral)
        cv = compute_two_period_score("beneish_m_score", facts, CURRENT, PRIOR)
        assert cv.value == pytest.approx(-2.48, abs=1e-9)

    def test_accruals_raise_the_score(self) -> None:
        """Earnings well above operating cash flow is the dominant term
        (TATA carries the largest coefficient, 4.679)."""
        neutral = {**CUR, "net_income": 30.0, "cash_from_operations": 30.0}
        accruing = {**CUR, "net_income": 130.0, "cash_from_operations": 30.0}
        base = compute_two_period_score(
            "beneish_m_score", two_period_facts(neutral, neutral), CURRENT, PRIOR
        )
        flagged = compute_two_period_score(
            "beneish_m_score", two_period_facts(accruing, neutral), CURRENT, PRIOR
        )
        assert flagged.value > base.value
        assert flagged.value == pytest.approx(base.value + 4.679 * (100.0 / 1000.0))


class TestOhlson:
    def test_matches_hand_computed_value(self) -> None:
        """ln(1000)=6.907755279; TL/TA=1.2; WC/TA=-0.1; CL/CA=1.25;
        OENEG=1; NI/TA=-0.05; CFO/TL=0.025; INTWO=0; CHIN=-1.0

        O = -1.32 - 0.407(6.907755279) + 6.03(1.2) - 1.43(-0.1) + 0.0757(1.25)
            - 1.72(1) - 2.37(-0.05) - 1.83(0.025) + 0.285(0) - 0.521(-1.0)
          = 2.2159186
        """
        cv = compute_two_period_score("ohlson_o_score", two_period_facts(), CURRENT, PRIOR)
        assert cv.value == pytest.approx(2.2159186, rel=1e-6)

    def test_insolvency_flag_fires(self) -> None:
        """OENEG subtracts 1.72 when liabilities exceed assets."""
        solvent = {**CUR, "total_liabilities": 900.0}
        insolvent = {**CUR, "total_liabilities": 900.0 + 1e-6, "total_assets": 900.0 - 1e-6}
        a = compute_two_period_score(
            "ohlson_o_score", two_period_facts(solvent, PRIOR_VALUES), CURRENT, PRIOR
        )
        b = compute_two_period_score(
            "ohlson_o_score", two_period_facts(insolvent, PRIOR_VALUES), CURRENT, PRIOR
        )
        assert b.value < a.value  # the -1.72 term dominates the small deltas

    def test_two_loss_years_flag(self) -> None:
        losses = {**PRIOR_VALUES, "net_income": -100.0}
        cv = compute_two_period_score(
            "ohlson_o_score", two_period_facts(CUR, losses), CURRENT, PRIOR
        )
        assert cv.is_defined

    def test_non_positive_assets_is_undefined(self) -> None:
        """ln() of a non-positive figure has no value -- undefined, not a crash."""
        broken = {**CUR, "total_assets": 0.0}
        cv = compute_two_period_score(
            "ohlson_o_score", two_period_facts(broken, PRIOR_VALUES), CURRENT, PRIOR
        )
        assert cv.value is None


class TestTwoPeriodMachinery:
    def test_prior_inputs_are_namespaced_and_traced(self) -> None:
        cv = compute_two_period_score("ohlson_o_score", two_period_facts(), CURRENT, PRIOR)
        assert cv.inputs["net_income"].period_end == CURRENT
        assert cv.inputs["net_income_prior"].period_end == PRIOR

    def test_as_of_is_the_later_filing(self) -> None:
        """A year-on-year score is knowable only once the current year is filed."""
        cv = compute_two_period_score("ohlson_o_score", two_period_facts(), CURRENT, PRIOR)
        assert cv.as_of == date(2025, 2, 20)

    def test_missing_prior_year_is_undefined_not_imputed(self) -> None:
        """No prior year means no year-on-year score. Substituting the current
        year would silently report zero change."""
        facts = make_facts(CUR, period_end=CURRENT)
        cv = compute_two_period_score("piotroski_f_score", facts, CURRENT, PRIOR)
        assert cv.value is None
        assert "missing line items" in cv.notes[0]

    def test_two_period_scores_omitted_without_prior(self) -> None:
        out = compute_scores(make_facts(CUR), CURRENT)
        assert set(out) == set(SINGLE_PERIOD_SCORES)

    def test_all_scores_present_with_prior(self) -> None:
        out = compute_scores(two_period_facts(), CURRENT, PRIOR)
        assert set(out) == set(ALL_SCORES)

    def test_unknown_metric_raises(self) -> None:
        with pytest.raises(KeyError):
            compute_two_period_score("nope", two_period_facts(), CURRENT, PRIOR)


class TestScoresVerify:
    def test_every_score_recomputes(self) -> None:
        """The verifier must reproduce two-period scores from provenance alone."""
        out = compute_scores(two_period_facts(), CURRENT, PRIOR)
        report = verify(out.values())
        assert report.passed, report.defect_summary()

    def test_tampered_score_is_caught(self) -> None:
        cv = compute_two_period_score("ohlson_o_score", two_period_facts(), CURRENT, PRIOR)
        tampered = cv.model_copy(update={"value": -5.0})
        assert not verify([tampered]).passed

    def test_all_scores_are_registered_formulas(self) -> None:
        for metric in ALL_SCORES:
            assert metric in FORMULAS

    def test_two_period_formulas_declare_prior_inputs(self) -> None:
        for metric in TWO_PERIOD_SCORES:
            assert any(k.endswith("_prior") for k in FORMULAS[metric].inputs)
