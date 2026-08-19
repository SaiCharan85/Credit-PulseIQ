"""L3 -- the distress investigator's loop, tools and critic.

Run against a **scripted model**: the loop really parses replies, really
dispatches tools against real filings, and really decides when to stop; only
token generation is fixed. That is what lets the agentic behaviour itself be
tested deterministically and offline, with no endpoint.

The behaviours pinned here are the ones that would silently rot:

* the loop branches on observations rather than running a fixed sequence,
* it cannot reach data from the future however it is prompted,
* it abstains rather than guessing,
* and the deterministic critic — not the judge — is what triggers a retry.
"""

from __future__ import annotations

import json
from datetime import date

import pytest

from agents.critic import (
    DEFECT_CALIBRATION,
    DEFECT_LOOKAHEAD,
    DEFECT_NUMERIC,
    DEFECT_SCOPE,
    review,
)
from agents.distress import (
    TERMINATED_BUDGET,
    TERMINATED_MODEL,
    TERMINATED_RETRIES,
    DistressInvestigator,
)
from agents.llm import ScriptedClient, extract_json, judge_client
from agents.schemas import (
    SIGNAL_HEALTHY,
    SIGNAL_INSUFFICIENT,
    SIGNAL_SEVERE,
    Evidence,
    InvestigatorOutput,
    scope_violations,
)
from agents.tools import STATUS_SEVERE, THRESHOLDS, ToolBox
from compute.lineitems import FactIndex
from compute.ratios import compute_metric
from data.edgar import EdgarClient
from data.facts import as_of_view

SLEEP_NUMBER = 827187
BEFORE_BANKRUPTCY = date(2026, 3, 1)  # petition was 2026-06-11


@pytest.fixture(scope="module")
def facts():
    return EdgarClient(offline=True).facts(SLEEP_NUMBER)


def tool_call(tool: str, **arguments) -> str:
    return json.dumps({"thought": "t", "action": "call_tool", "tool": tool, "arguments": arguments})


def finish(**fields) -> str:
    return json.dumps({"thought": "t", "action": "finish", **fields})


# Cites only ``current_ratio``: the critic rejects any figure no tool produced,
# so every script below must actually fetch what its conclusion cites.
GOOD_FINISH = finish(
    signal=SIGNAL_SEVERE,
    confidence=0.85,
    rationale="Liabilities exceed assets and current ratio is far below 1.0.",
    evidence=[{"metric": "current_ratio", "value": 0.1996, "note": "far below 1.0"}],
)

FETCH_CURRENT_RATIO = tool_call("get_metric", metric="current_ratio")


class TestToolBoxAsOfEnforcement:
    def test_as_of_is_not_a_tool_argument(self, facts) -> None:
        """The agent cannot ask for the future because there is no parameter
        for it -- the box is bound to one date at construction."""
        tools = ToolBox(SLEEP_NUMBER, BEFORE_BANKRUPTCY, facts)
        import inspect

        for name in ("get_metric", "get_trend", "get_line_item", "check_threshold"):
            params = inspect.signature(getattr(tools, name)).parameters
            assert "as_of" not in params

    def test_periods_exclude_filings_not_yet_public(self, facts) -> None:
        tools = ToolBox(SLEEP_NUMBER, BEFORE_BANKRUPTCY, facts)
        periods = tools.available_periods()["periods"]
        assert periods
        # FY2025 ended 2026-01-03 but was filed after this date.
        assert "2026-01-03" not in periods

    def test_later_as_of_reveals_more(self, facts) -> None:
        early = ToolBox(SLEEP_NUMBER, date(2025, 1, 1), facts).available_periods()["count"]
        late = ToolBox(SLEEP_NUMBER, date(2026, 8, 1), facts).available_periods()["count"]
        assert late > early

    def test_requesting_an_invisible_period_is_refused(self, facts) -> None:
        tools = ToolBox(SLEEP_NUMBER, BEFORE_BANKRUPTCY, facts)
        result = tools.get_metric("current_ratio", period="2026-01-03")
        assert "error" in result


class TestToolBoxContracts:
    def test_unknown_metric_returns_an_error_not_an_exception(self, facts) -> None:
        """A bad argument must be an observation the agent can recover from."""
        result = ToolBox(SLEEP_NUMBER, BEFORE_BANKRUPTCY, facts).get_metric("nonsense")
        assert "error" in result and "unknown metric" in result["error"]

    def test_metric_carries_citations(self, facts) -> None:
        result = ToolBox(SLEEP_NUMBER, BEFORE_BANKRUPTCY, facts).get_metric("current_ratio")
        assert result["defined"] and result["citations"]

    def test_threshold_fetches_its_own_value(self, facts) -> None:
        """One fewer place for a number to drift between calls."""
        result = ToolBox(SLEEP_NUMBER, BEFORE_BANKRUPTCY, facts).check_threshold("current_ratio")
        assert result["value"] is not None
        assert result["status"] == STATUS_SEVERE

    def test_every_threshold_names_a_direction(self) -> None:
        for metric, rule in THRESHOLDS.items():
            assert rule["worse"] in ("higher", "lower"), metric

    def test_calls_are_recorded_for_the_audit_trail(self, facts) -> None:
        tools = ToolBox(SLEEP_NUMBER, BEFORE_BANKRUPTCY, facts)
        tools.get_metric("current_ratio")
        tools.get_metric("nonsense")
        trail = tools.audit_trail()
        assert [c["tool"] for c in trail] == ["get_metric", "get_metric"]
        assert trail[0]["ok"] and not trail[1]["ok"]

    def test_peer_comparison_without_peers_is_an_error(self, facts) -> None:
        result = ToolBox(SLEEP_NUMBER, BEFORE_BANKRUPTCY, facts).get_peer_comparison("current_ratio")
        assert "error" in result


class TestReActLoop:
    def test_loop_branches_on_observations(self, facts) -> None:
        """A real loop, not one prompted call: four tools chosen in sequence."""
        script = [
            tool_call("available_periods"),
            tool_call("get_metric", metric="liabilities_to_assets"),
            tool_call("check_threshold", metric="current_ratio"),
            tool_call("get_trend", metric="liabilities_to_assets", n_periods=4),
            GOOD_FINISH,
        ]
        out = DistressInvestigator(ScriptedClient(script=script)).run(
            SLEEP_NUMBER, BEFORE_BANKRUPTCY, facts
        )
        assert out.signal == SIGNAL_SEVERE
        assert out.terminated_because == TERMINATED_MODEL
        assert [c["tool"] for c in out.audit_trail] == [
            "available_periods",
            "get_metric",
            "check_threshold",
            "get_trend",
        ]

    def test_model_sees_observations_between_calls(self, facts) -> None:
        client = ScriptedClient(script=[tool_call("get_metric", metric="current_ratio"), GOOD_FINISH])
        DistressInvestigator(client).run(SLEEP_NUMBER, BEFORE_BANKRUPTCY, facts)
        last_prompt = client.calls[-1]
        assert any("Observation" in m["content"] for m in last_prompt)

    def test_loop_recovers_from_a_tool_error(self, facts) -> None:
        """A bad call is an observation, not a crash; the loop continues."""
        script = [
            tool_call("get_metric", metric="not_a_real_metric"),
            FETCH_CURRENT_RATIO,
            GOOD_FINISH,
        ]
        out = DistressInvestigator(ScriptedClient(script=script)).run(
            SLEEP_NUMBER, BEFORE_BANKRUPTCY, facts
        )
        assert out.signal == SIGNAL_SEVERE
        assert out.audit_trail[0]["ok"] is False

    def test_unknown_tool_is_reported_back(self, facts) -> None:
        script = [tool_call("hack_the_database"), FETCH_CURRENT_RATIO, GOOD_FINISH]
        out = DistressInvestigator(ScriptedClient(script=script)).run(
            SLEEP_NUMBER, BEFORE_BANKRUPTCY, facts
        )
        assert out.signal == SIGNAL_SEVERE  # recovered

    def test_malformed_json_is_survivable(self, facts) -> None:
        script = ["I think the company looks risky.", FETCH_CURRENT_RATIO, GOOD_FINISH]
        out = DistressInvestigator(ScriptedClient(script=script)).run(
            SLEEP_NUMBER, BEFORE_BANKRUPTCY, facts
        )
        assert out.signal == SIGNAL_SEVERE

    def test_step_budget_forces_abstention(self, facts) -> None:
        """Never an unbounded loop, and never a guess when it runs out."""
        script = [tool_call("get_metric", metric="current_ratio")] * 20
        out = DistressInvestigator(ScriptedClient(script=script), max_steps=3).run(
            SLEEP_NUMBER, BEFORE_BANKRUPTCY, facts
        )
        assert out.signal == SIGNAL_INSUFFICIENT
        assert out.terminated_because == TERMINATED_BUDGET

    def test_model_may_abstain_on_its_own_judgment(self, facts) -> None:
        script = [
            tool_call("get_metric", metric="interest_coverage"),
            finish(
                signal=SIGNAL_INSUFFICIENT,
                confidence=0.3,
                rationale="Coverage is undefined and no substitute was available.",
            ),
        ]
        out = DistressInvestigator(ScriptedClient(script=script)).run(
            SLEEP_NUMBER, BEFORE_BANKRUPTCY, facts
        )
        assert out.abstained
        assert out.terminated_because == TERMINATED_MODEL

    def test_fabricated_metric_in_evidence_is_rejected(self, facts) -> None:
        """A figure that no tool produced cannot be cited, so the response
        fails the critic and, after bounded retries, abstains."""
        script = [
            tool_call("get_metric", metric="current_ratio"),
            finish(
                signal=SIGNAL_SEVERE,
                confidence=0.9,
                rationale="Debt to EBITDA is 12x.",
                evidence=[{"metric": "debt_to_ebitda", "value": 12.0}],
            ),
        ] * 4
        out = DistressInvestigator(ScriptedClient(script=script)).run(
            SLEEP_NUMBER, BEFORE_BANKRUPTCY, facts
        )
        assert out.signal == SIGNAL_INSUFFICIENT
        assert out.terminated_because == TERMINATED_RETRIES

    def test_retry_feeds_back_the_specific_defect(self, facts) -> None:
        bad = finish(
            signal=SIGNAL_SEVERE,
            confidence=0.95,
            rationale="Investors should sell this position immediately.",
        )
        client = ScriptedClient(script=[FETCH_CURRENT_RATIO, bad, GOOD_FINISH])
        out = DistressInvestigator(client).run(SLEEP_NUMBER, BEFORE_BANKRUPTCY, facts)
        assert out.signal == SIGNAL_SEVERE
        assert out.retries == 1
        feedback = [m for call in client.calls for m in call if "failed these deterministic" in m["content"]]
        assert feedback and "scope" in feedback[-1]["content"]


class TestCritic:
    def _output(self, **fields) -> InvestigatorOutput:
        base = dict(
            cik=SLEEP_NUMBER,
            as_of=BEFORE_BANKRUPTCY,
            signal=SIGNAL_SEVERE,
            confidence=0.8,
            rationale="Leverage is elevated.",
        )
        base.update(fields)
        return InvestigatorOutput(**base)

    def visible(self, facts, as_of=BEFORE_BANKRUPTCY):
        """Compute the way the ToolBox does -- from the as-of view.

        Using raw facts here made the critic (correctly) flag lookahead: the
        FY2024 balance sheet was restated in a 10-K filed after the prediction
        date.
        """
        view = FactIndex(as_of_view(facts, as_of))
        return compute_metric("current_ratio", view, date(2024, 12, 28))

    def test_clean_output_passes(self, facts) -> None:
        cv = self.visible(facts)
        assert review(self._output(), [cv], BEFORE_BANKRUPTCY).passed

    def test_unreproducible_figure_is_a_hard_fail(self, facts) -> None:
        cv = self.visible(facts)
        tampered = cv.model_copy(update={"value": 3.0})
        report = review(self._output(), [tampered], BEFORE_BANKRUPTCY)
        assert not report.passed
        assert any(d.kind == DEFECT_NUMERIC for d in report.defects)

    def test_lookahead_is_a_hard_fail(self, facts) -> None:
        cv = self.visible(facts)
        report = review(self._output(), [cv], date(2024, 1, 1))
        assert any(d.kind == DEFECT_LOOKAHEAD for d in report.defects)

    def test_decision_framing_is_a_hard_fail(self) -> None:
        report = review(
            self._output(rationale="We recommend investing in this name."), [], BEFORE_BANKRUPTCY
        )
        assert any(d.kind == DEFECT_SCOPE for d in report.defects)

    def test_confidence_capped_when_residual_reported(self) -> None:
        report = review(
            self._output(confidence=0.95, residual="coverage could not be computed"),
            [],
            BEFORE_BANKRUPTCY,
        )
        assert any(d.kind == DEFECT_CALIBRATION for d in report.defects)

    def test_confident_abstention_is_a_contradiction(self) -> None:
        report = review(
            self._output(signal=SIGNAL_INSUFFICIENT, confidence=0.9), [], BEFORE_BANKRUPTCY
        )
        assert any(d.kind == DEFECT_CALIBRATION for d in report.defects)

    def test_uncited_evidence_is_rejected(self) -> None:
        report = review(
            self._output(evidence=[Evidence(metric="debt_to_ebitda", value=9.9)]),
            [],
            BEFORE_BANKRUPTCY,
        )
        assert any(d.kind == DEFECT_NUMERIC for d in report.defects)

    def test_staleness_warns_but_does_not_block(self, facts) -> None:
        cv = self.visible(facts)
        report = review(self._output(), [cv], date(2027, 6, 1))
        assert report.warnings
        assert report.passed

    def test_feedback_is_specific(self) -> None:
        report = review(self._output(rationale="You should sell this."), [], BEFORE_BANKRUPTCY)
        assert "scope" in report.feedback()


class TestOutputContract:
    def test_unknown_signal_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            InvestigatorOutput(
                cik=1, as_of=BEFORE_BANKRUPTCY, signal="probably_fine", confidence=0.5
            )

    def test_confidence_is_bounded(self) -> None:
        with pytest.raises(ValueError):
            InvestigatorOutput(cik=1, as_of=BEFORE_BANKRUPTCY, signal=SIGNAL_HEALTHY, confidence=1.4)

    def test_false_confidence_definition(self) -> None:
        """High-confidence healthy on a name that later failed -- the
        catastrophic-error metric."""
        healthy = InvestigatorOutput(
            cik=1, as_of=BEFORE_BANKRUPTCY, signal=SIGNAL_HEALTHY, confidence=0.9
        )
        assert healthy.is_false_confidence(failed=True)
        assert not healthy.is_false_confidence(failed=False)

    def test_unsure_healthy_is_not_false_confidence(self) -> None:
        unsure = InvestigatorOutput(
            cik=1, as_of=BEFORE_BANKRUPTCY, signal=SIGNAL_HEALTHY, confidence=0.4
        )
        assert not unsure.is_false_confidence(failed=True)

    def test_abstention_is_never_false_confidence(self) -> None:
        abstain = InvestigatorOutput(
            cik=1, as_of=BEFORE_BANKRUPTCY, signal=SIGNAL_INSUFFICIENT, confidence=0.0
        )
        assert not abstain.is_false_confidence(failed=True)

    def test_only_elevated_and_severe_count_as_positives(self) -> None:
        for signal, expected in (
            (SIGNAL_HEALTHY, False),
            ("watch", False),
            ("elevated_risk", True),
            (SIGNAL_SEVERE, True),
            (SIGNAL_INSUFFICIENT, False),
        ):
            out = InvestigatorOutput(
                cik=1, as_of=BEFORE_BANKRUPTCY, signal=signal, confidence=0.5
            )
            assert out.flags_risk is expected

    def test_scope_violations_detected(self) -> None:
        assert scope_violations("You should sell this position")
        assert not scope_violations("Leverage is elevated and coverage is thin.")


class TestJudgeSeparation:
    def test_judge_must_differ_from_the_agent_model(self, monkeypatch) -> None:
        """Grading an output with the model that produced it invites
        self-preference bias (hard rule 7)."""
        monkeypatch.setenv("CREDITPULSE_LLM_MODEL", "same-model")
        monkeypatch.setenv("CREDITPULSE_JUDGE_MODEL", "same-model")
        with pytest.raises(RuntimeError, match="must differ"):
            judge_client()

    def test_missing_judge_is_an_explicit_error(self, monkeypatch) -> None:
        monkeypatch.delenv("CREDITPULSE_JUDGE_MODEL", raising=False)
        with pytest.raises(RuntimeError):
            judge_client()


class TestJsonExtraction:
    def test_plain_object(self) -> None:
        assert extract_json('{"a": 1}') == {"a": 1}

    def test_fenced_block(self) -> None:
        assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}

    def test_object_with_surrounding_prose(self) -> None:
        assert extract_json('Sure! {"a": 1} hope that helps') == {"a": 1}

    def test_nested_braces_and_strings(self) -> None:
        assert extract_json('{"a": {"b": "}"}}') == {"a": {"b": "}"}}

    def test_no_json_returns_none(self) -> None:
        assert extract_json("no json here") is None

    def test_malformed_returns_none(self) -> None:
        assert extract_json('{"a": }') is None


class TestTwoPeriodScoresWithoutAPriorYear:
    """Regression: a year-on-year score with only one visible period.

    Crashed a live 400-case backtest 139 calls in. ``get_metric`` fell through
    to the single-period path, whose formulas declare ``<concept>_prior``
    inputs -- not line items -- so concept resolution raised. A tool must hand
    the agent a result it can reason about, never an exception.
    """

    def one_period_facts(self):
        from evals.conftest import DISTRESSED, make_facts

        return make_facts(DISTRESSED, period_end=date(2024, 12, 31), filed=date(2025, 2, 20))

    def toolbox(self):
        return ToolBox(1, date(2025, 6, 1), self.one_period_facts())

    def test_only_one_period_is_visible(self) -> None:
        assert self.toolbox().available_periods()["count"] == 1

    @pytest.mark.parametrize("metric", ["ohlson_o_score", "piotroski_f_score", "beneish_m_score"])
    def test_get_metric_returns_undefined_not_an_exception(self, metric: str) -> None:
        result = self.toolbox().get_metric(metric)
        assert "error" not in result
        assert result["defined"] is False
        assert "prior fiscal year" in " ".join(result["notes"])

    @pytest.mark.parametrize("metric", ["ohlson_o_score", "piotroski_f_score"])
    def test_check_threshold_reports_unknown(self, metric: str) -> None:
        result = self.toolbox().check_threshold(metric)
        assert "error" not in result
        assert result["status"] == "unknown"
        assert result["value"] is None

    def test_single_period_metrics_still_compute(self) -> None:
        """The guard must not disable ordinary single-period metrics."""
        assert self.toolbox().get_metric("current_ratio")["defined"] is True

    def test_prior_year_is_never_the_same_period(self) -> None:
        """check_threshold previously passed the period as its own prior,
        silently producing year-on-year ratios of 1.0 rather than failing."""
        from evals.conftest import DISTRESSED, make_facts

        facts = make_facts(DISTRESSED, period_end=date(2024, 12, 31), filed=date(2025, 2, 20))
        facts += make_facts(DISTRESSED, period_end=date(2023, 12, 31), filed=date(2024, 2, 20))
        cv = ToolBox(1, date(2025, 6, 1), facts).get_metric("ohlson_o_score")
        assert cv["defined"] is True
        assert "2023-12-31" in " ".join(cv["notes"])


class TestTrendsOfYearOnYearScores:
    """Second regression from the same live crash, via a different caller.

    ``get_trend`` reached ``compute_metric`` with a two-period score, which
    raised deep in concept resolution. Fixed in two places: ``compute_metric``
    now refuses such formulas outright (protecting every caller), and
    ``get_trend`` pairs consecutive periods so the trend is actually computed.
    """

    @pytest.mark.parametrize("metric", ["ohlson_o_score", "piotroski_f_score", "beneish_m_score"])
    def test_trend_does_not_raise(self, facts, metric: str) -> None:
        result = ToolBox(SLEEP_NUMBER, BEFORE_BANKRUPTCY, facts).get_trend(metric, 4)
        assert "values" in result or "error" in result

    def test_ohlson_trend_has_real_points(self, facts) -> None:
        result = ToolBox(SLEEP_NUMBER, BEFORE_BANKRUPTCY, facts).get_trend("ohlson_o_score", 4)
        assert len(result["values"]) >= 2
        assert result["direction"] in ("improving", "deteriorating", "flat")

    def test_each_point_uses_its_own_prior_year(self, facts) -> None:
        """Not one shared prior: every point compares consecutive years."""
        result = ToolBox(SLEEP_NUMBER, BEFORE_BANKRUPTCY, facts).get_trend("ohlson_o_score", 4)
        assert len(result["periods"]) == len(set(result["periods"]))

    def test_compute_metric_refuses_two_period_formulas(self) -> None:
        """The safety net that protects callers we have not written yet."""
        from compute.ratios import compute_metric
        from evals.conftest import DISTRESSED, make_facts

        cv = compute_metric("ohlson_o_score", make_facts(DISTRESSED), date(2024, 12, 31))
        assert not cv.is_defined
        assert "two fiscal years" in cv.notes[0]

    def test_single_period_trends_are_unaffected(self, facts) -> None:
        result = ToolBox(SLEEP_NUMBER, BEFORE_BANKRUPTCY, facts).get_trend("current_ratio", 4)
        assert len(result["values"]) == 4


class TestForcedFinish:
    """The loop asks for a verdict before the budget runs out.

    Measured on a real 200-case run: gemma-4-31b-it averaged 13.3 of 14 steps
    and never called finish on 75% of cases. Those became step-budget
    abstentions and were recorded as protocol failures -- a truncated
    investigation reported as the model declining to answer.
    """

    def test_model_is_warned_before_the_budget_runs_out(self, facts) -> None:
        client = ScriptedClient(script=[FETCH_CURRENT_RATIO] * 20)
        DistressInvestigator(client, max_steps=4).run(SLEEP_NUMBER, BEFORE_BANKRUPTCY, facts)
        prompts = [m["content"] for call in client.calls for m in call if m["role"] == "user"]
        assert any("one step left" in p for p in prompts)

    def test_warning_arrives_with_a_turn_left_to_act_on_it(self, facts) -> None:
        """Warning on the final step would leave no room to comply."""
        client = ScriptedClient(script=[FETCH_CURRENT_RATIO] * 20)
        DistressInvestigator(client, max_steps=6).run(SLEEP_NUMBER, BEFORE_BANKRUPTCY, facts)
        warned_at = [
            i for i, call in enumerate(client.calls)
            if any("one step left" in m["content"] for m in call if m["role"] == "user")
        ]
        assert warned_at and warned_at[0] < len(client.calls) - 1

    def test_a_model_that_complies_produces_a_verdict(self, facts) -> None:
        script = [FETCH_CURRENT_RATIO, FETCH_CURRENT_RATIO, GOOD_FINISH]
        out = DistressInvestigator(ScriptedClient(script=script), max_steps=4).run(
            SLEEP_NUMBER, BEFORE_BANKRUPTCY, facts
        )
        assert out.signal == SIGNAL_SEVERE
        assert out.terminated_because == TERMINATED_MODEL

    def test_warning_is_issued_once(self, facts) -> None:
        client = ScriptedClient(script=[FETCH_CURRENT_RATIO] * 20)
        DistressInvestigator(client, max_steps=5).run(SLEEP_NUMBER, BEFORE_BANKRUPTCY, facts)
        last = client.calls[-1]
        warnings = [m for m in last if m["role"] == "user" and "one step left" in m["content"]]
        assert len(warnings) == 1


class TestRiskScoreAndBaseline:
    """Two changes aimed at the measured gap to the hazard model.

    The first run's scores clustered: 146 of 198 cases in two buckets, because
    a five-level signal times a few round confidences yields ~12 distinct
    values. AUC is a ranking metric, so ties cap it regardless of reasoning.
    """

    def test_risk_score_is_used_for_ranking_when_present(self) -> None:
        from evals.backtest import CaseResult

        out = InvestigatorOutput(
            cik=1, as_of=BEFORE_BANKRUPTCY, signal=SIGNAL_SEVERE,
            confidence=0.85, risk_score=93.0,
        )
        case = CaseResult(cik=1, as_of=BEFORE_BANKRUPTCY, label=1, days_to_event=10, output=out)
        assert case.risk_probability == pytest.approx(0.93)

    def test_band_mapping_applies_without_a_score(self) -> None:
        """Falls back to the ordinal band, not the old direction-only mapping."""
        from evals.backtest import SIGNAL_BANDS, CaseResult

        out = InvestigatorOutput(
            cik=1, as_of=BEFORE_BANKRUPTCY, signal=SIGNAL_HEALTHY, confidence=0.9
        )
        case = CaseResult(cik=1, as_of=BEFORE_BANKRUPTCY, label=0, days_to_event=None, output=out)
        low, high = SIGNAL_BANDS[SIGNAL_HEALTHY]
        assert low <= case.risk_probability <= high

    def test_risk_score_breaks_ties_the_signal_cannot(self) -> None:
        """Two severe calls at equal confidence must be rankable."""
        from evals.backtest import CaseResult

        def sev(score):
            return CaseResult(
                cik=1, as_of=BEFORE_BANKRUPTCY, label=1, days_to_event=1,
                output=InvestigatorOutput(
                    cik=1, as_of=BEFORE_BANKRUPTCY, signal=SIGNAL_SEVERE,
                    confidence=0.85, risk_score=score,
                ),
            )
        assert sev(95).risk_probability > sev(72).risk_probability

    def test_risk_score_is_bounded(self) -> None:
        with pytest.raises(ValueError):
            InvestigatorOutput(
                cik=1, as_of=BEFORE_BANKRUPTCY, signal=SIGNAL_SEVERE,
                confidence=0.5, risk_score=140.0,
            )

    def test_baseline_tool_returns_the_score(self, facts) -> None:
        tools = ToolBox(SLEEP_NUMBER, BEFORE_BANKRUPTCY, facts, model_score=0.87)
        result = tools.get_model_score()
        assert result["baseline_probability"] == pytest.approx(0.87)
        assert "disagree" in result["note"].lower()

    def test_baseline_tool_errors_when_not_supplied(self, facts) -> None:
        """Absent baseline must be a recoverable observation, not a crash."""
        result = ToolBox(SLEEP_NUMBER, BEFORE_BANKRUPTCY, facts).get_model_score()
        assert "error" in result

    def test_loop_can_call_the_baseline_tool(self, facts) -> None:
        script = [tool_call("get_model_score"), FETCH_CURRENT_RATIO, GOOD_FINISH]
        out = DistressInvestigator(ScriptedClient(script=script)).run(
            SLEEP_NUMBER, BEFORE_BANKRUPTCY, facts, model_score=0.9
        )
        assert "get_model_score" in [c["tool"] for c in out.audit_trail]
        assert out.signal == SIGNAL_SEVERE
