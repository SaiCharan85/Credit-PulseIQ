"""L0 -- independent numeric verification.

The verifier is the enforcement point for "the LLM never does arithmetic"
(PROMPT hard rule 1). These tests prove it actually catches a wrong number
rather than rubber-stamping whatever it is handed.
"""

from __future__ import annotations

from datetime import date

import pytest

from compute.provenance import ComputedValue
from compute.ratios import compute_all, compute_metric
from evals.conftest import DEFAULT_PERIOD_END, DISTRESSED, make_facts
from verify.recompute import (
    FAIL_UNKNOWN_FORMULA,
    FAIL_VALUE_MISMATCH,
    recompute,
    verify,
    verify_or_raise,
)

FILED = date(2025, 2, 20)
LATER = date(2025, 6, 1)


class TestRecompute:
    def test_honest_value_verifies(self, distressed_facts) -> None:
        cv = compute_metric("debt_to_assets", distressed_facts, DEFAULT_PERIOD_END)
        assert recompute(cv).agrees

    def test_tampered_value_is_caught(self, distressed_facts) -> None:
        """The core guarantee: a figure that does not follow from its inputs
        fails, however plausible it looks."""
        cv = compute_metric("debt_to_assets", distressed_facts, DEFAULT_PERIOD_END)
        tampered = cv.model_copy(update={"value": 0.35})  # halved: looks healthy
        result = recompute(tampered)
        assert result.failed
        assert result.reason == FAIL_VALUE_MISMATCH
        assert result.recomputed == pytest.approx(0.70)

    def test_subtle_tampering_is_caught(self, distressed_facts) -> None:
        cv = compute_metric("interest_coverage", distressed_facts, DEFAULT_PERIOD_END)
        tampered = cv.model_copy(update={"value": cv.value * (1 + 1e-6)})
        assert recompute(tampered).failed

    def test_floating_point_noise_is_tolerated(self, distressed_facts) -> None:
        cv = compute_metric("ocf_to_debt", distressed_facts, DEFAULT_PERIOD_END)
        jittered = cv.model_copy(update={"value": cv.value * (1 + 1e-12)})
        assert recompute(jittered).agrees

    def test_unknown_formula_fails_closed(self) -> None:
        """An unrecognized formula must fail, not pass by default. Otherwise a
        fabricated metric name would bypass verification entirely."""
        cv = ComputedValue(
            metric="made_up",
            formula="made_up",
            value=1.0,
            unit="ratio",
            period_end=DEFAULT_PERIOD_END,
        )
        result = recompute(cv)
        assert result.failed and result.reason == FAIL_UNKNOWN_FORMULA

    def test_undefined_with_missing_inputs_is_consistent(self) -> None:
        values = {k: v for k, v in DISTRESSED.items() if k != "total_assets"}
        cv = compute_metric("debt_to_assets", make_facts(values), DEFAULT_PERIOD_END)
        assert recompute(cv).agrees

    def test_value_claimed_without_inputs_fails(self) -> None:
        cv = ComputedValue(
            metric="debt_to_assets",
            formula="debt_to_assets",
            value=0.42,
            unit="ratio",
            period_end=DEFAULT_PERIOD_END,
            inputs={},
        )
        assert recompute(cv).failed

    def test_all_standard_metrics_verify(self, distressed_facts) -> None:
        values = compute_all(distressed_facts, DEFAULT_PERIOD_END)
        report = verify(values.values())
        assert report.passed, report.defect_summary()


class TestLookaheadGuard:
    def test_input_from_the_future_is_flagged(self) -> None:
        cv = compute_metric("debt_to_assets", make_facts(DISTRESSED, filed=LATER), DEFAULT_PERIOD_END)
        report = verify([cv], as_of=date(2025, 3, 1))
        assert not report.passed
        assert report.lookahead_violations

    def test_input_filed_before_prediction_passes(self) -> None:
        cv = compute_metric("debt_to_assets", make_facts(DISTRESSED, filed=FILED), DEFAULT_PERIOD_END)
        assert verify([cv], as_of=date(2025, 3, 1)).passed

    def test_same_day_filing_is_a_violation(self) -> None:
        cv = compute_metric("debt_to_assets", make_facts(DISTRESSED, filed=FILED), DEFAULT_PERIOD_END)
        assert verify([cv], as_of=FILED).lookahead_violations


class TestStaleness:
    def test_old_inputs_warn_but_do_not_block(self) -> None:
        """Stale data is surfaced, not fatal -- a filer gone quiet is itself a
        signal the investigator should reason about (SPEC 8)."""
        cv = compute_metric("debt_to_assets", make_facts(DISTRESSED, filed=FILED), DEFAULT_PERIOD_END)
        report = verify([cv], as_of=date(2027, 1, 1))
        assert report.staleness_warnings
        assert report.passed

    def test_fresh_inputs_do_not_warn(self) -> None:
        cv = compute_metric("debt_to_assets", make_facts(DISTRESSED, filed=FILED), DEFAULT_PERIOD_END)
        assert not verify([cv], as_of=date(2025, 4, 1)).staleness_warnings


class TestReportContract:
    def test_defect_summary_names_the_specific_figure(self, distressed_facts) -> None:
        """The retry feedback must be actionable. "verification failed" gives a
        reviser nothing to work with (SPEC feedback loops)."""
        cv = compute_metric("current_ratio", distressed_facts, DEFAULT_PERIOD_END)
        report = verify([cv.model_copy(update={"value": 9.9})])
        summary = report.defect_summary()
        assert "current_ratio" in summary and "9.9" in summary

    def test_verify_or_raise_raises_on_failure(self, distressed_facts) -> None:
        cv = compute_metric("current_ratio", distressed_facts, DEFAULT_PERIOD_END)
        with pytest.raises(AssertionError):
            verify_or_raise([cv.model_copy(update={"value": 9.9})])
