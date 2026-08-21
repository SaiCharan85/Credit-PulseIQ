"""L4: end-to-end, filer facts in -> cited memo out (SPEC 7 ladder).

Runs the orchestrator over a whole filer, so it covers what nothing below it
does: routing on findings, the guard gate, and memo assembly acting together.

The scope-labelling tests are load-bearing. Only the distress leg is
backtested; the earnings-quality leg measured 0.51-0.61 across five attempts
and is attached as context. A memo that presented both alike would let a reader
give the unmeasured one the credibility the measured one earned, which is the
exact failure the honest-scoping rule exists to prevent -- and it would be
invisible, because the words would look the same.
"""

from __future__ import annotations

from datetime import date

from agents.memo import TIER_BACKTESTED, TIER_CONTEXT
from agents.orchestrator import (
    DEPTH_DEEP,
    DEPTH_SHALLOW,
    Orchestrator,
    triage,
)
from agents.rulebased import RuleBasedInvestigator
from evals.conftest import make_fact

AS_OF = date(2024, 6, 1)


def filer(
    current_assets=500.0,
    current_liabilities=100.0,
    total_assets=2000.0,
    total_liabilities=400.0,
    revenue=1000.0,
    net_income=200.0,
    end=date(2023, 12, 31),
    filed=date(2024, 3, 1),
):
    values = {
        "current_assets": current_assets,
        "current_liabilities": current_liabilities,
        "total_assets": total_assets,
        "total_liabilities": total_liabilities,
        "revenue": revenue,
        "net_income": net_income,
    }
    return [make_fact(c, v, end, filed) for c, v in values.items()]


HEALTHY = filer()
STRESSED = filer(current_assets=50.0, current_liabilities=500.0,
                 total_assets=1000.0, total_liabilities=900.0, net_income=-300.0)


class TestRoutingOnFindings:
    def test_a_clean_profile_gets_a_shallow_pass(self) -> None:
        plan = triage(HEALTHY, AS_OF)
        assert plan.depth == DEPTH_SHALLOW
        assert plan.steps < 14

    def test_a_stressed_profile_gets_the_full_budget(self) -> None:
        plan = triage(STRESSED, AS_OF)
        assert plan.depth == DEPTH_DEEP
        assert plan.steps == 14

    def test_the_reason_for_the_route_is_recorded(self) -> None:
        """Routing must be auditable, not a number that appeared."""
        assert any("quick_ratio" in r for r in triage(STRESSED, AS_OF).reasons)

    def test_no_visible_period_escalates_rather_than_skipping(self) -> None:
        """An unreadable filer is the last one to give a shallow pass."""
        plan = triage([], AS_OF)
        assert plan.depth == DEPTH_DEEP

    def test_a_debt_free_filer_is_not_escalated_for_lacking_interest_coverage(self) -> None:
        """Absence escalates only where absence is abnormal.

        Every operating filer reports a balance sheet, so a missing quick ratio
        means tags went away. A company with no borrowings legitimately has no
        interest coverage, and escalating that routes healthy filers deep for
        the crime of having no debt.
        """
        plan = triage(HEALTHY, AS_OF)
        assert "interest_coverage" in plan.uncomputable
        assert plan.depth == DEPTH_SHALLOW

    def test_the_step_budget_is_restored_after_the_run(self) -> None:
        inv = RuleBasedInvestigator()
        before = getattr(inv, "max_steps", None)
        Orchestrator(inv).run(1, AS_OF, STRESSED)
        assert getattr(inv, "max_steps", None) == before


class TestMemoAssembly:
    def _memo(self, facts=STRESSED, notes=("Accruals exceed operating cash flow.",)):
        result = Orchestrator(RuleBasedInvestigator(), context_notes=notes).run(1, AS_OF, facts)
        assert result.shipped, result.blocked_reason
        return result.memo

    def test_a_memo_is_produced_end_to_end(self) -> None:
        memo = self._memo()
        assert memo.cik == 1
        assert memo.as_of == AS_OF
        assert memo.signal

    def test_every_cited_figure_reaches_the_rendered_memo(self) -> None:
        memo = self._memo()
        text = memo.render()
        assert memo.sections[0].evidence
        for e in memo.sections[0].evidence:
            assert e.metric in text

    def test_the_distress_section_is_labelled_backtested(self) -> None:
        assert self._memo().sections[0].tier == TIER_BACKTESTED

    def test_earnings_observations_are_labelled_context_only(self) -> None:
        memo = self._memo()
        context = [s for s in memo.sections if s.tier == TIER_CONTEXT]
        assert context, "context notes should appear as their own section"
        assert not context[0].moves_the_signal

    def test_graded_evidence_excludes_context_sections(self) -> None:
        """The distinction must be machine-readable, not only printed."""
        memo = self._memo()
        graded = memo.graded_evidence
        context_evidence = [e for s in memo.sections if s.tier == TIER_CONTEXT for e in s.evidence]
        for e in context_evidence:
            assert e not in graded

    def test_the_rendered_memo_states_its_tiers(self) -> None:
        text = self._memo().render()
        assert TIER_BACKTESTED in text
        assert TIER_CONTEXT in text

    def test_routing_is_visible_to_the_reader(self) -> None:
        assert "ROUTING" in self._memo().render()

    def test_the_advisory_disclaimer_is_always_present(self) -> None:
        assert "not a recommendation" in self._memo().render()

    def test_the_audit_trail_travels_with_the_memo(self) -> None:
        memo = self._memo()
        assert memo.tool_calls == len(memo.audit_trail)
        assert memo.tool_calls > 0

    def test_a_memo_with_no_context_notes_has_no_context_section(self) -> None:
        memo = self._memo(notes=())
        assert all(s.tier != TIER_CONTEXT for s in memo.sections)


class TestDataLimitationsSurface:
    def test_uncomputable_metrics_are_reported_not_hidden(self) -> None:
        result = Orchestrator(RuleBasedInvestigator()).run(1, AS_OF, HEALTHY)
        assert result.shipped
        assert any("uncomputable" in item for item in result.memo.limitations)
        assert "DATA LIMITATIONS" in result.memo.render()

    def test_a_stale_filer_is_flagged_in_the_memo(self) -> None:
        old = filer(end=date(2019, 12, 31), filed=date(2020, 3, 1))
        result = Orchestrator(RuleBasedInvestigator()).run(1, AS_OF, old)
        assert result.shipped
        assert any("annual period ends" in item for item in result.memo.limitations)
