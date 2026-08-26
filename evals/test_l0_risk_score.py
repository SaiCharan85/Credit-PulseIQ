"""L0: the control arm's ordinal score, and why it cannot move the measurement.

The rule-based arm emitted no ``risk_score`` at all. "What is its risk score"
is a fair question and it was unanswerable for every filer this arm assessed,
because five signal levels were the entire output and five levels cannot
separate two filers inside the same band.

The property that makes adding one safe is **band containment**: the score is
derived from the band its own signal implies, then positioned inside it. So

* it can never contradict the signal -- the critic's consistency check has
  nothing to catch, rather than catching it after the fact;
* ranking by score is a *refinement* of ranking by signal. Two filers in
  different bands keep their order; two in the same band get separated. A
  refinement cannot reduce rank AUC, so the published 0.885 stands as a floor
  rather than being invalidated by this change.

The first version derived the score independently and hoped the two agreed. It
scored an ``elevated_risk`` filer at 36 against a band of [45, 75] and the
guard blocked the memo -- correctly, and that is the failure this design
removes by construction.
"""

from __future__ import annotations

import itertools

import pytest

from agents.critic import SIGNAL_BANDS
from agents.rulebased import CASCADE, risk_score
from agents.schemas import (
    SIGNAL_ELEVATED,
    SIGNAL_HEALTHY,
    SIGNAL_INSUFFICIENT,
    SIGNAL_SEVERE,
    SIGNAL_WATCH,
)

ORDER = [SIGNAL_HEALTHY, SIGNAL_WATCH, SIGNAL_ELEVATED, SIGNAL_SEVERE]


def _counts(n_severe: int, n_flag: int) -> tuple[list[str], list[str]]:
    return list(CASCADE[:n_severe]), list(CASCADE[:n_flag])


@pytest.mark.parametrize("signal", ORDER)
@pytest.mark.parametrize("n_severe", range(len(CASCADE) + 1))
@pytest.mark.parametrize("n_flag", range(len(CASCADE) + 1))
def test_the_score_always_lands_inside_its_own_band(
    signal: str, n_severe: int, n_flag: int
) -> None:
    """Band containment, over every breach combination the cascade can produce.
    If this fails the critic blocks the memo, so it is not a style point."""
    low, high = SIGNAL_BANDS[signal]
    score = risk_score(signal, *_counts(n_severe, n_flag))
    assert low <= score <= high, f"{signal}: {score} outside [{low}, {high}]"


@pytest.mark.parametrize("signal", ORDER)
def test_the_score_stays_off_the_band_edges(signal: str) -> None:
    """A filer sitting exactly on a boundary reads as belonging to either
    neighbour, and boundaries are where readers look hardest."""
    low, high = SIGNAL_BANDS[signal]
    extremes = [risk_score(signal, *_counts(0, 0)),
                risk_score(signal, *_counts(len(CASCADE), len(CASCADE)))]
    for score in extremes:
        assert low < score < high, f"{signal}: {score} sits on a band edge"


def test_ranking_by_score_never_contradicts_ranking_by_signal() -> None:
    """The property the published figure rests on. Across bands the order is
    preserved exactly, so ranking by score is a refinement of ranking by
    signal -- and a refinement cannot reduce rank AUC."""
    worst = (len(CASCADE), len(CASCADE))
    best = (0, 0)
    for lower, higher in itertools.combinations(ORDER, 2):
        # Even the *worst* filer in the lower band must score below the *best*
        # filer in the higher one.
        assert risk_score(lower, *_counts(*worst)) < risk_score(higher, *_counts(*best)), (
            f"{lower} at its worst outranks {higher} at its best"
        )


def test_more_breaches_score_higher_within_a_band() -> None:
    """The point of having a score at all: separating filers the signal cannot."""
    for signal in ORDER:
        scores = [risk_score(signal, *_counts(n, 0)) for n in range(len(CASCADE) + 1)]
        assert scores == sorted(scores)
        assert scores[0] < scores[-1], f"{signal}: the score never moves"


def test_a_severe_breach_outweighs_a_flag() -> None:
    """A measure past the severe level is a stronger statement than one merely
    near it; collapsing them would put five near-misses level with insolvency."""
    assert risk_score(SIGNAL_SEVERE, *_counts(2, 0)) > risk_score(SIGNAL_SEVERE, *_counts(0, 2))


def test_an_unknown_signal_does_not_raise() -> None:
    """Insufficient-evidence has no band. It must return something harmless
    rather than throw inside an assessment."""
    assert risk_score(SIGNAL_INSUFFICIENT, *_counts(0, 0)) == 0.0
    assert risk_score("nonsense", *_counts(3, 1)) == 0.0


def test_the_arm_now_emits_a_score_end_to_end() -> None:
    from datetime import date

    from agents.orchestrator import Orchestrator
    from agents.rulebased import RuleBasedInvestigator
    from evals.test_l4_memo import STRESSED

    result = Orchestrator(RuleBasedInvestigator()).run(1, date(2024, 6, 1), STRESSED)
    assert result.shipped, result.blocked_reason
    assert result.memo.risk_score is not None, "the question that started this"
    low, high = SIGNAL_BANDS[result.memo.signal]
    assert low <= result.memo.risk_score <= high
