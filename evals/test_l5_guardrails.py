"""L5: guardrails and adversarial behaviour (SPEC 8).

Every test here describes a memo that must not reach a human, or a caveat that
must not be dropped on the way. The guards are the last thing standing between
a plausible-looking assessment and a reader who will act on it.

Two guards block and two annotate, and the asymmetry is the design: a
fabricated number or a decision recommendation makes a memo *wrong*, so it
stops. Stale data makes a memo *conditional*, so it travels attached to the
memo instead of silently deciding for the reader that it did not matter.
"""

from __future__ import annotations

from datetime import date

import pytest

from agents.guards import (
    GUARD_NUMERIC,
    GUARD_SCOPE,
    STALE_AFTER_DAYS,
    check_freshness,
    run_guards,
)
from agents.orchestrator import Orchestrator
from agents.schemas import (
    SIGNAL_HEALTHY,
    SIGNAL_INSUFFICIENT,
    SIGNAL_SEVERE,
    InvestigatorOutput,
)

AS_OF = date(2024, 6, 1)
RECENT = date(2023, 12, 31)


def output(**kw) -> InvestigatorOutput:
    base = dict(
        cik=1,
        as_of=AS_OF,
        signal=SIGNAL_SEVERE,
        confidence=0.9,
        risk_score=85.0,
        rationale="Liquidity has deteriorated for four consecutive quarters.",
        verification_passed=True,
    )
    base.update(kw)
    return InvestigatorOutput(**base)


class TestScopeGuardBlocks:
    """The output states risk. It never states an action."""

    @pytest.mark.parametrize(
        "rationale",
        [
            "Risk is elevated, so deny the loan application.",
            "We recommend investing in this issuer.",
            "Short the equity ahead of the filing.",
            "Our price target implies further downside.",
        ],
    )
    def test_decision_framing_blocks_the_memo(self, rationale: str) -> None:
        report = run_guards(output(rationale=rationale), [], AS_OF, RECENT)
        assert not report.may_ship
        assert any(GUARD_SCOPE in b for b in report.blocked)

    def test_a_risk_statement_is_allowed(self) -> None:
        report = run_guards(output(), [], AS_OF, RECENT)
        assert report.may_ship

    def test_a_blocked_assessment_produces_no_memo_at_all(self) -> None:
        """Blocking must withhold the artifact, not merely annotate it."""

        class Recommender:
            max_steps = 6

            def run(self, cik, as_of, facts, **kw):
                return output(cik=cik, as_of=as_of, rationale="Deny the loan.")

        result = Orchestrator(Recommender()).run(1, AS_OF, [])
        assert result.memo is None
        assert not result.shipped
        assert GUARD_SCOPE in result.blocked_reason


class TestNumericGuardBlocks:
    def test_a_failed_verification_blocks_the_memo(self) -> None:
        bad = output(verification_passed=False, verification_defects="stated 1.4, recomputed 0.4")
        report = run_guards(bad, [], AS_OF, RECENT)
        assert not report.may_ship
        assert any(GUARD_NUMERIC in b for b in report.blocked)

    def test_the_defect_detail_survives_into_the_reason(self) -> None:
        bad = output(verification_passed=False, verification_defects="stated 1.4, recomputed 0.4")
        report = run_guards(bad, [], AS_OF, RECENT)
        assert "recomputed 0.4" in report.summary()

    def test_no_memo_is_rendered_when_numbers_do_not_reproduce(self) -> None:
        class Fabricator:
            max_steps = 6

            def run(self, cik, as_of, facts, **kw):
                return output(cik=cik, as_of=as_of, verification_passed=False,
                              verification_defects="quick_ratio unreproducible")

        result = Orchestrator(Fabricator()).run(1, AS_OF, [])
        assert result.memo is None


class TestFreshnessGuardAnnotates:
    """Stale data conditions a memo; it does not invalidate it."""

    def test_a_recent_filing_raises_nothing(self) -> None:
        assert check_freshness(RECENT, AS_OF) == []

    def test_a_stale_filing_is_surfaced(self) -> None:
        stale = date(2020, 12, 31)
        items = check_freshness(stale, AS_OF)
        assert items and "2020-12-31" in items[0]

    def test_absent_periods_are_surfaced_rather_than_assumed_fine(self) -> None:
        assert check_freshness(None, AS_OF)

    def test_the_boundary_does_not_cry_wolf_on_ordinary_filing_lag(self) -> None:
        """A filer one day inside the window is not flagged."""
        from datetime import timedelta

        assert check_freshness(AS_OF - timedelta(days=STALE_AFTER_DAYS - 1), AS_OF) == []
        assert check_freshness(AS_OF - timedelta(days=STALE_AFTER_DAYS + 1), AS_OF)

    def test_staleness_never_blocks(self) -> None:
        report = run_guards(output(), [], AS_OF, date(2018, 1, 1))
        assert report.may_ship
        assert report.limitations


class TestAbstentionIsNotSafety:
    def test_an_abstention_is_marked_as_such(self) -> None:
        """'Insufficient evidence' must never read as 'low risk'."""
        report = run_guards(
            output(signal=SIGNAL_INSUFFICIENT, confidence=0.0, risk_score=None),
            [],
            AS_OF,
            RECENT,
        )
        assert any("abstention" in item for item in report.limitations)

    def test_an_abstention_still_ships(self) -> None:
        """It is a valid terminal state, not a failure to suppress."""
        report = run_guards(
            output(signal=SIGNAL_INSUFFICIENT, confidence=0.0, risk_score=None),
            [],
            AS_OF,
            RECENT,
        )
        assert report.may_ship

    def test_a_healthy_verdict_carries_no_abstention_caveat(self) -> None:
        report = run_guards(output(signal=SIGNAL_HEALTHY, confidence=0.8, risk_score=10.0),
                            [], AS_OF, RECENT)
        assert not any("abstention" in item for item in report.limitations)


class TestGuardsCannotBeBypassed:
    def test_every_shipped_memo_passed_the_gate(self) -> None:
        class Sneaky:
            max_steps = 6

            def run(self, cik, as_of, facts, **kw):
                return output(cik=cik, as_of=as_of, rationale="Approve the loan.")

        result = Orchestrator(Sneaky()).run(1, AS_OF, [])
        assert result.guards is not None
        assert result.shipped is (result.guards.may_ship is True)
        assert not result.shipped
