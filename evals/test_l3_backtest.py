"""L3 -- the backtest harness and the rule-based control agent.

Metric definitions are pinned against hand-built cases, because these are the
numbers the project reports. The false-confidence rate in particular is the
designated kill-signal metric (SPEC 10) and must not drift.
"""

from __future__ import annotations

from datetime import date

import pytest

from agents.rulebased import CASCADE, MIN_DEFINED_METRICS, RuleBasedInvestigator
from agents.schemas import (
    SIGNAL_ELEVATED,
    SIGNAL_HEALTHY,
    SIGNAL_INSUFFICIENT,
    SIGNAL_SEVERE,
    SIGNAL_WATCH,
    InvestigatorOutput,
)
from data.edgar import EdgarClient
from evals.backtest import (
    CaseResult,
    assert_no_lookahead,
    grade,
    reliability_curve,
    run_backtest,
)
from models.panel import PanelRow

SLEEP_NUMBER = 827187
BEFORE = date(2026, 3, 1)


@pytest.fixture(scope="module")
def facts():
    return EdgarClient(offline=True).facts(SLEEP_NUMBER)


def case(label: int, signal: str, confidence: float, days: int | None = None) -> CaseResult:
    return CaseResult(
        cik=1,
        as_of=BEFORE,
        label=label,
        days_to_event=days,
        output=InvestigatorOutput(
            cik=1, as_of=BEFORE, signal=signal, confidence=confidence
        ),
    )


class TestGrading:
    def test_precision_and_recall(self) -> None:
        results = [
            case(1, SIGNAL_SEVERE, 0.9, days=100),   # TP
            case(1, SIGNAL_ELEVATED, 0.7, days=200), # TP
            case(0, SIGNAL_ELEVATED, 0.7),           # FP
            case(1, SIGNAL_HEALTHY, 0.5),            # FN
            case(0, SIGNAL_HEALTHY, 0.8),            # TN
        ]
        report = grade(results)
        assert report.true_positives == 2
        assert report.false_positives == 1
        assert report.false_negatives == 1
        assert report.precision == pytest.approx(2 / 3)
        assert report.recall == pytest.approx(2 / 3)

    def test_only_elevated_and_severe_are_positive_predictions(self) -> None:
        report = grade([case(1, SIGNAL_WATCH, 0.6)])
        assert report.true_positives == 0
        assert report.false_negatives == 1

    def test_median_lead_time_uses_correct_flags_only(self) -> None:
        results = [
            case(1, SIGNAL_SEVERE, 0.9, days=100),
            case(1, SIGNAL_SEVERE, 0.9, days=300),
            case(0, SIGNAL_SEVERE, 0.9),  # false positive contributes no lead time
        ]
        assert grade(results).median_lead_time_days == pytest.approx(200)

    def test_abstentions_are_neither_hit_nor_miss(self) -> None:
        """An honest "insufficient evidence" is not a wrong answer. Folding it
        into the error count would punish the behaviour guardrails encourage."""
        results = [case(1, SIGNAL_INSUFFICIENT, 0.0), case(0, SIGNAL_INSUFFICIENT, 0.0)]
        report = grade(results)
        assert report.n_abstained == 2
        assert report.true_positives == report.false_positives == report.false_negatives == 0
        assert report.precision is None
        assert report.abstention_rate == pytest.approx(1.0)


class TestFalseConfidence:
    """The catastrophic-error metric: confidently benign on a name that failed."""

    def test_counted_when_confidently_healthy_and_company_failed(self) -> None:
        report = grade([case(1, SIGNAL_HEALTHY, 0.9)])
        assert report.false_confidence_cases == 1
        assert report.false_confidence_rate == pytest.approx(1.0)

    def test_not_counted_when_unsure(self) -> None:
        assert grade([case(1, SIGNAL_HEALTHY, 0.4)]).false_confidence_cases == 0

    def test_not_counted_when_company_survived(self) -> None:
        assert grade([case(0, SIGNAL_HEALTHY, 0.95)]).false_confidence_cases == 0

    def test_abstention_is_never_false_confidence(self) -> None:
        assert grade([case(1, SIGNAL_INSUFFICIENT, 0.0)]).false_confidence_cases == 0

    def test_watch_at_high_confidence_still_counts(self) -> None:
        """"Watch" is a benign call: it does not flag risk."""
        assert grade([case(1, SIGNAL_WATCH, 0.9)]).false_confidence_cases == 1

    def test_rate_is_over_failures_not_over_all_cases(self) -> None:
        results = [case(1, SIGNAL_HEALTHY, 0.9)] + [case(0, SIGNAL_HEALTHY, 0.9)] * 9
        report = grade(results)
        assert report.n_positives == 1
        assert report.false_confidence_rate == pytest.approx(1.0)


class TestCalibrationMapping:
    def test_signal_order_is_preserved(self) -> None:
        """Each verdict maps into its own band, so the four levels stay ordered.

        The previous mapping used only direction -- `confidence if flags_risk
        else 1 - confidence` -- which scored "watch at 0.9" identically to
        "healthy at 0.9". Four verdicts collapsed to two numbers, and the ties
        cost AUC the model had actually earned.
        """
        healthy = case(0, SIGNAL_HEALTHY, 0.9).risk_probability
        watch = case(0, SIGNAL_WATCH, 0.9).risk_probability
        elevated = case(1, SIGNAL_ELEVATED, 0.9).risk_probability
        severe = case(1, SIGNAL_SEVERE, 0.9).risk_probability
        assert healthy < watch < elevated < severe

    def test_confidence_positions_within_the_band(self) -> None:
        """A confident healthy call is less risky than a hesitant one."""
        assert case(0, SIGNAL_HEALTHY, 0.9).risk_probability < case(
            0, SIGNAL_HEALTHY, 0.4
        ).risk_probability

    def test_abstention_is_neutral(self) -> None:
        assert case(1, SIGNAL_INSUFFICIENT, 0.0).risk_probability == pytest.approx(0.5)

    def test_reliability_curve_bins(self) -> None:
        results = [case(1, SIGNAL_SEVERE, 0.9)] * 5 + [case(0, SIGNAL_HEALTHY, 0.9)] * 5
        curve = reliability_curve(results, bins=5)
        assert curve
        assert all(0 <= row["observed_rate"] <= 1 for row in curve)

    def test_perfect_calibration_has_zero_ece(self) -> None:
        results = [case(1, SIGNAL_SEVERE, 1.0)] * 5 + [case(0, SIGNAL_HEALTHY, 1.0)] * 5
        assert grade(results).ece == pytest.approx(0.0)


class TestLookaheadAssertion:
    def test_mismatched_as_of_is_caught(self) -> None:
        bad = CaseResult(
            cik=1,
            as_of=date(2024, 1, 1),
            label=0,
            days_to_event=None,
            output=InvestigatorOutput(
                cik=1, as_of=date(2025, 1, 1), signal=SIGNAL_HEALTHY, confidence=0.5
            ),
        )
        with pytest.raises(AssertionError):
            assert_no_lookahead([bad])

    def test_matching_as_of_passes(self) -> None:
        assert_no_lookahead([case(0, SIGNAL_HEALTHY, 0.5)])


class TestRuleBasedControl:
    def test_flags_a_company_months_before_it_filed(self, facts) -> None:
        """Sleep Number petitioned 2026-06-11; standing at 2026-03-01 the
        control should already see severe distress."""
        out = RuleBasedInvestigator().run(SLEEP_NUMBER, BEFORE, facts)
        assert out.signal in (SIGNAL_ELEVATED, SIGNAL_SEVERE)
        assert out.flags_risk

    def test_output_passes_its_own_critic(self, facts) -> None:
        """The control is verified on the same terms as the ReAct loop, so a
        comparison is not flattered by skipping verification."""
        out = RuleBasedInvestigator().run(SLEEP_NUMBER, BEFORE, facts)
        assert out.verification_passed, out.verification_defects

    def test_abstains_when_too_little_is_computable(self, facts) -> None:
        early = RuleBasedInvestigator().run(SLEEP_NUMBER, date(2009, 1, 1), facts)
        assert early.signal == SIGNAL_INSUFFICIENT

    def test_evidence_is_cited(self, facts) -> None:
        out = RuleBasedInvestigator().run(SLEEP_NUMBER, BEFORE, facts)
        assert out.evidence
        assert all(e.value is not None for e in out.evidence)

    def test_residual_caps_confidence(self, facts) -> None:
        out = RuleBasedInvestigator().run(SLEEP_NUMBER, BEFORE, facts)
        if out.residual:
            assert out.confidence <= 0.6

    def test_cascade_covers_the_core_dimensions(self) -> None:
        assert MIN_DEFINED_METRICS <= len(CASCADE)
        assert "liabilities_to_assets" in CASCADE and "current_ratio" in CASCADE


class TestRunBacktest:
    def test_end_to_end_over_panel_cases(self, facts) -> None:
        cases = [
            PanelRow(cik=SLEEP_NUMBER, observation_date=BEFORE, label=1, days_to_event=102),
            PanelRow(cik=SLEEP_NUMBER, observation_date=date(2025, 6, 1), label=0),
        ]
        results = run_backtest(RuleBasedInvestigator(), cases, lambda _cik: facts)
        assert len(results) == 2
        assert_no_lookahead(results)
        report = grade(results)
        assert report.n_cases == 2
        assert report.mean_tool_calls > 0

    def test_investigator_receives_the_case_date(self, facts) -> None:
        seen: list[date] = []

        class Spy:
            def run(self, cik, as_of, facts, **kw):
                seen.append(as_of)
                return InvestigatorOutput(
                    cik=cik, as_of=as_of, signal=SIGNAL_HEALTHY, confidence=0.5
                )

        cases = [
            PanelRow(cik=1, observation_date=date(2024, 1, 1), label=0),
            PanelRow(cik=1, observation_date=date(2024, 4, 1), label=0),
        ]
        run_backtest(Spy(), cases, lambda _cik: facts)
        assert seen == [date(2024, 1, 1), date(2024, 4, 1)]


class TestStratifiedSampling:
    """Positives are scarce and each agent case costs an API call, so
    negatives are thinned -- and the resulting precision is corrected back."""

    CASES = [PanelRow(cik=i, observation_date=BEFORE, label=1) for i in range(10)] + [
        PanelRow(cik=100 + i, observation_date=BEFORE, label=0) for i in range(90)
    ]

    def test_every_positive_is_kept(self) -> None:
        from evals.backtest import stratified_sample

        s = stratified_sample(self.CASES, max_negatives=30)
        assert s.n_positives == 10
        assert sum(c.label for c in s.cases) == 10

    def test_negatives_are_capped(self) -> None:
        from evals.backtest import stratified_sample

        s = stratified_sample(self.CASES, max_negatives=30)
        assert s.n_negatives_kept == 30
        assert s.negative_fraction == pytest.approx(30 / 90)

    def test_sampling_is_deterministic(self) -> None:
        from evals.backtest import stratified_sample

        a = stratified_sample(self.CASES, max_negatives=30, seed=7)
        b = stratified_sample(self.CASES, max_negatives=30, seed=7)
        assert [c.cik for c in a.cases] == [c.cik for c in b.cases]

    def test_no_sampling_when_cap_exceeds_population(self) -> None:
        from evals.backtest import stratified_sample

        s = stratified_sample(self.CASES, max_negatives=1000)
        assert s.negative_fraction == 1.0
        assert len(s.cases) == len(self.CASES)

    def test_precision_is_corrected_back_to_the_population(self) -> None:
        """Thinning negatives inflates raw precision; the correction removes it.

        Verified end to end against the real panel: the 400-case sample reads
        0.460 raw and 0.186 corrected, matching the full 1,217-case run.
        """
        results = [case(1, SIGNAL_SEVERE, 0.8) for _ in range(10)]
        results += [case(0, SIGNAL_SEVERE, 0.8) for _ in range(10)]
        report = grade(results, negative_fraction=0.25)
        assert report.precision == pytest.approx(0.5)
        # 10 sampled FPs represent 40 in the population -> 10 / (10 + 40)
        assert report.corrected_precision == pytest.approx(0.2)

    def test_recall_is_unaffected_by_sampling(self) -> None:
        """Recall is computed over positives only, which are never thinned."""
        results = [case(1, SIGNAL_SEVERE, 0.8), case(1, SIGNAL_HEALTHY, 0.5)]
        assert grade(results, negative_fraction=0.25).recall == pytest.approx(0.5)


class TestProtocolFailureSplit:
    """A model too weak to hold the protocol yields the same verdict as one
    exercising judgment. Conflating them lets incompetence read as caution."""

    def _abstention(self, reason: str) -> CaseResult:
        return CaseResult(
            cik=1,
            as_of=BEFORE,
            label=0,
            days_to_event=None,
            output=InvestigatorOutput(
                cik=1,
                as_of=BEFORE,
                signal=SIGNAL_INSUFFICIENT,
                confidence=0.0,
                terminated_because=reason,
            ),
        )

    def test_model_choosing_to_abstain_is_honest(self) -> None:
        report = grade([self._abstention("model_finished")])
        assert report.honest_abstentions == 1
        assert report.protocol_failures == 0

    @pytest.mark.parametrize(
        "reason", ["step_budget_exhausted", "retries_exhausted", "unparseable_response"]
    )
    def test_incomplete_runs_are_protocol_failures(self, reason: str) -> None:
        report = grade([self._abstention(reason)])
        assert report.protocol_failures == 1
        assert report.honest_abstentions == 0

    def test_rates_sum_to_the_abstention_rate(self) -> None:
        results = [
            self._abstention("model_finished"),
            self._abstention("retries_exhausted"),
            case(0, SIGNAL_HEALTHY, 0.6),
        ]
        report = grade(results)
        assert report.honest_abstentions + report.protocol_failures == report.n_abstained
        assert report.protocol_failure_rate == pytest.approx(1 / 3)


class TestResponseCache:
    def test_identical_conversation_is_replayed(self, tmp_path) -> None:
        """Providers are not bit-reproducible at temperature 0, so a backtest
        without a cache cannot distinguish a regression from provider drift."""
        from agents.llm import CachingClient, ScriptedClient

        inner = ScriptedClient(script=['{"a": 1}', '{"b": 2}'])
        client = CachingClient(inner=inner, cache_dir=tmp_path)
        messages = [{"role": "user", "content": "hello"}]
        first = client.complete(messages)
        second = client.complete(messages)
        assert first == second == '{"a": 1}'
        assert client.stats() == {"hits": 1, "misses": 1}

    def test_different_conversations_are_not_confused(self, tmp_path) -> None:
        from agents.llm import CachingClient, ScriptedClient

        client = CachingClient(
            inner=ScriptedClient(script=['{"a": 1}', '{"b": 2}']), cache_dir=tmp_path
        )
        assert client.complete([{"role": "user", "content": "one"}]) == '{"a": 1}'
        assert client.complete([{"role": "user", "content": "two"}]) == '{"b": 2}'

    def test_cache_survives_a_new_client(self, tmp_path) -> None:
        from agents.llm import CachingClient, ScriptedClient

        messages = [{"role": "user", "content": "hello"}]
        CachingClient(inner=ScriptedClient(script=['{"a": 1}']), cache_dir=tmp_path).complete(messages)
        fresh = CachingClient(inner=ScriptedClient(script=[]), cache_dir=tmp_path)
        assert fresh.complete(messages) == '{"a": 1}'
        assert fresh.stats()["hits"] == 1
