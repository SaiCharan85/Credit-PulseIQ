"""L3 -- run resilience: outages, quotas, pacing, and error attribution.

These paths exist so that an environment failure is never reported as an agent
failure. That distinction has been wrong three separate times in this project:
a 402 recorded as protocol failure, raw line items rejected as fabricated, and
a dropped connection about to be scored as the model refusing to answer. Each
would have produced a confident, wrong conclusion about model quality.

Everything here runs offline against stub clients.
"""

from __future__ import annotations

from datetime import date

import pytest

import evals.backtest as bt
from agents.llm import (
    BudgetExhausted,
    CachingClient,
    InfrastructureError,
    RateLimitedClient,
    ScriptedClient,
)
from agents.schemas import SIGNAL_HEALTHY, SIGNAL_INSUFFICIENT, InvestigatorOutput
from evals.backtest import (
    MAX_CONSECUTIVE_FAILURES,
    PROTOCOL_FAILURE_REASONS,
    grade,
    run_backtest,
)
from models.panel import PanelRow

TODAY = date(2024, 7, 1)


def cases(n: int) -> list[PanelRow]:
    return [PanelRow(cik=i, observation_date=TODAY, label=i % 2) for i in range(n)]


def healthy(cik: int, as_of: date) -> InvestigatorOutput:
    return InvestigatorOutput(cik=cik, as_of=as_of, signal=SIGNAL_HEALTHY, confidence=0.5)


class FlakyInvestigator:
    """Fails for the first ``down_for`` attempts, then works."""

    def __init__(self, down_for: int, error: type[Exception] = ConnectionError):
        self.down_for = down_for
        self.error = error
        self.attempts = 0

    def run(self, cik, as_of, facts, **kw):
        self.attempts += 1
        if self.attempts <= self.down_for:
            raise self.error("Connection error.")
        return healthy(cik, as_of)


@pytest.fixture(autouse=True)
def _fast_backoff(monkeypatch):
    """Keep outage backoff instant so the suite stays fast."""
    monkeypatch.setattr(bt, "OUTAGE_RETRY_SECONDS", 0.001)


class TestOutageResume:
    def test_run_resumes_after_the_network_returns(self) -> None:
        """The whole point: an outage must not cost us the remaining cases."""
        agent = FlakyInvestigator(down_for=12)
        results = run_backtest(agent, cases(10), lambda c: [])
        assert len(results) == 10
        assert all(not r.abstained for r in results)

    def test_the_interrupted_case_is_retried_not_skipped(self) -> None:
        """Skipping would silently drop a case from the sample."""
        agent = FlakyInvestigator(down_for=8)
        results = run_backtest(agent, cases(6), lambda c: [])
        assert [r.cik for r in results] == [0, 1, 2, 3, 4, 5]

    def test_isolated_failures_do_not_trigger_the_outage_path(self) -> None:
        """One bad filer is recorded and skipped; the sweep continues."""

        class OneBadCase:
            def run(self, cik, as_of, facts, **kw):
                if cik == 3:
                    raise ValueError("malformed filing")
                return healthy(cik, as_of)

        results = run_backtest(OneBadCase(), cases(8), lambda c: [])
        assert len(results) == 8
        bad = [r for r in results if r.cik == 3]
        assert bad[0].output.terminated_because == "case_error"

    def test_gives_up_after_the_maximum_outage_wait(self, monkeypatch) -> None:
        monkeypatch.setattr(bt, "MAX_OUTAGE_WAIT_SECONDS", 0.0)
        agent = FlakyInvestigator(down_for=10_000)
        results = run_backtest(agent, cases(20), lambda c: [])
        assert len(results) < 20  # stopped rather than fabricating 20 failures

    def test_partial_results_are_still_gradeable(self, monkeypatch) -> None:
        monkeypatch.setattr(bt, "MAX_OUTAGE_WAIT_SECONDS", 0.0)

        class DiesHalfway:
            def __init__(self):
                self.n = 0

            def run(self, cik, as_of, facts, **kw):
                self.n += 1
                if self.n > 6:
                    raise ConnectionError("Connection error.")
                return healthy(cik, as_of)

        results = run_backtest(DiesHalfway(), cases(30), lambda c: [])
        report = grade(results)
        assert 0 < report.n_cases < 30
        assert report.n_cases == len(results)


class TestErrorAttribution:
    """An environment failure must never be scored as an agent failure."""

    def test_infrastructure_errors_abort_rather_than_grade(self) -> None:
        class NoCredit:
            def run(self, cik, as_of, facts, **kw):
                raise InfrastructureError("402 insufficient credits")

        with pytest.raises(InfrastructureError):
            run_backtest(NoCredit(), cases(5), lambda c: [])

    def test_quota_stop_grades_what_completed(self) -> None:
        class QuotaAfterThree:
            def __init__(self):
                self.n = 0

            def run(self, cik, as_of, facts, **kw):
                self.n += 1
                if self.n > 3:
                    raise BudgetExhausted("daily limit")
                return healthy(cik, as_of)

        results = run_backtest(QuotaAfterThree(), cases(20), lambda c: [])
        assert len(results) == 3
        assert grade(results).n_cases == 3

    def test_case_error_is_classified_as_protocol_failure(self) -> None:
        assert "case_error" in PROTOCOL_FAILURE_REASONS

    def test_consecutive_threshold_is_above_one(self) -> None:
        """A single blip must not be mistaken for an outage."""
        assert MAX_CONSECUTIVE_FAILURES >= 3


class TestTokenPacing:
    def test_pacing_spaces_calls_under_the_ceiling(self, monkeypatch) -> None:
        """Reactive backoff alone idled at half the allowance; pacing sleeps
        just enough to stay beneath it instead of provoking a 429.

        ``time.sleep`` is stubbed: the point is that it *would* wait, not that
        the suite should spend a minute proving it.
        """
        slept: list[float] = []
        import agents.llm as llm_mod

        monkeypatch.setattr(llm_mod.time, "sleep", slept.append)
        client = RateLimitedClient(inner=ScriptedClient(script=["ok"] * 50), tokens_per_minute=1000)
        big = [{"role": "user", "content": "x" * 3200}]  # ~800 tokens + headroom
        for _ in range(2):
            client.complete(big)
        assert slept, "second call should have been paced"
        assert client.stats()["paced_seconds"] > 0

    def test_pacing_is_off_by_default(self) -> None:
        client = RateLimitedClient(inner=ScriptedClient(script=["ok"] * 5))
        for _ in range(5):
            client.complete([{"role": "user", "content": "x" * 4000}])
        assert client.stats()["paced_seconds"] == 0

    def test_token_estimate_counts_tools_and_content(self) -> None:
        small = RateLimitedClient._estimate_tokens([{"role": "user", "content": "hi"}], None)
        large = RateLimitedClient._estimate_tokens(
            [{"role": "user", "content": "hi"}], [{"a": "x" * 4000}]
        )
        assert large > small


class TestCacheSurvivesRestart:
    def test_a_restart_replays_without_new_calls(self, tmp_path) -> None:
        """Process death must not cost the work already paid for."""
        messages = [{"role": "user", "content": "case-1"}]
        first = CachingClient(inner=ScriptedClient(script=['{"a":1}']), cache_dir=tmp_path)
        first.complete(messages)

        # Restart with a client that has nothing left to give.
        restarted = CachingClient(inner=ScriptedClient(script=[]), cache_dir=tmp_path)
        assert restarted.complete(messages) == '{"a":1}'
        assert restarted.stats() == {"hits": 1, "misses": 0}

    def test_cache_is_keyed_by_model(self, tmp_path) -> None:
        """Switching models must not replay the previous model's answers."""
        messages = [{"role": "user", "content": "same question"}]
        a = CachingClient(inner=ScriptedClient(script=['{"from":"a"}'], name="model-a"), cache_dir=tmp_path)
        b = CachingClient(inner=ScriptedClient(script=['{"from":"b"}'], name="model-b"), cache_dir=tmp_path)
        assert a.complete(messages) != b.complete(messages)


class TestLineItemsAreCitable:
    """Regression: the critic rejected an agent for citing data it fetched."""

    def test_raw_line_items_count_as_cited_evidence(self) -> None:
        from agents.critic import review
        from agents.schemas import SIGNAL_SEVERE, Evidence

        output = InvestigatorOutput(
            cik=1,
            as_of=TODAY,
            signal=SIGNAL_SEVERE,
            confidence=0.8,
            rationale="Equity is negative.",
            evidence=[Evidence(metric="equity", value=-4.5e8)],
        )
        rejected = review(output, [], TODAY)
        accepted = review(output, [], TODAY, cited_line_items={"equity"})
        assert not rejected.passed
        assert accepted.passed

    def test_genuinely_uncited_figures_are_still_rejected(self) -> None:
        from agents.critic import review
        from agents.schemas import SIGNAL_SEVERE, Evidence

        output = InvestigatorOutput(
            cik=1,
            as_of=TODAY,
            signal=SIGNAL_SEVERE,
            confidence=0.8,
            rationale="Leverage is 12x.",
            evidence=[Evidence(metric="debt_to_ebitda", value=12.0)],
        )
        assert not review(output, [], TODAY, cited_line_items={"equity"}).passed


class TestAbstentionAccounting:
    def test_environment_failures_never_look_like_judgment(self) -> None:
        """An honest abstention and a crashed case must not be conflated."""
        honest = InvestigatorOutput(
            cik=1, as_of=TODAY, signal=SIGNAL_INSUFFICIENT, confidence=0.0,
            terminated_because="model_finished",
        )
        crashed = InvestigatorOutput(
            cik=2, as_of=TODAY, signal=SIGNAL_INSUFFICIENT, confidence=0.0,
            terminated_because="case_error",
        )
        results = [
            bt.CaseResult(cik=1, as_of=TODAY, label=0, days_to_event=None, output=honest),
            bt.CaseResult(cik=2, as_of=TODAY, label=0, days_to_event=None, output=crashed),
        ]
        report = grade(results)
        assert report.honest_abstentions == 1
        assert report.protocol_failures == 1
