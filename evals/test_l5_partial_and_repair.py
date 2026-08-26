"""L5: a bad figure costs the claim, not the whole answer.

All-or-nothing was costing real answers. The case that forced this:

    "if this company is healthy, why was it at default in 2020"
    -> ANSWER WITHHELD BY THE GUARD

A fundamental question, and the analyst got a wall. The figure that failed was
one of several; the rest of the answer verified perfectly well.

Two recoveries, tried in order, both bounded:

**Repair.** The model asserted where a figure came from and was wrong. Told
exactly which claim failed and what the source says, it usually fixes it. One
attempt, and the repair is accepted only if it verifies completely -- a second
bad answer is not progress, and preferring it would hide that the model cannot
source the claim at all. Same shape as the investigator's bounded retry against
the numeric critic.

**Partial.** If the repair fails, drop the sentences carrying bad figures and
keep the rest, with a visible note naming what was removed. The unit is the
sentence, because a figure's claim lives in its sentence -- cutting the number
alone leaves "coverage is of the interest it owes".

Two limits keep this from becoming a way to launder bad output. Below half the
answer surviving it refuses outright, because a mangled paragraph serves nobody.
And neither recovery applies to the scope guard: an answer that recommended an
action is refused, never coached into a compliant rephrasing of the same advice.
"""

from __future__ import annotations

from agents.qa import (
    MIN_SURVIVING_SHARE,
    PARTIAL_NOTICE,
    build_context,
    check_answer,
    redact_bad_claims,
)

PAYLOAD = {
    "cik": 28823,
    "as_of": "2024-07-01",
    "triage": {"latest_period_end": "2022-12-31"},
    "memo": {
        "signal": "severe_risk", "confidence": 0.8, "risk_score": 97.1,
        "summary": "", "limitations": [], "routing": [], "audit_trail": [],
        "sections": [{"title": "Credit distress", "tier": "backtested", "body": "",
                      "evidence": [
                          {"metric": "current_ratio", "value": 1.10,
                           "period_end": "2022-12-31", "note": "flag"},
                          {"metric": "interest_coverage", "value": -1.06,
                           "period_end": "2022-12-31", "note": "severe"},
                          {"metric": "net_margin", "value": -0.17,
                           "period_end": "2022-12-31", "note": "severe"},
                      ]}],
    },
}


def _ctx():
    return build_context(PAYLOAD)


# ---- redaction, in isolation ----------------------------------------------

def test_only_the_sentence_carrying_the_bad_figure_is_dropped() -> None:
    text = (
        "The company cannot cover its interest. "
        "Coverage stands at 9.99 for the period. "
        "It also loses money on what it sells."
    )
    kept, dropped = redact_bad_claims(text, ["9.99 cites interest_coverage"])
    assert "9.99" not in kept
    assert "cannot cover its interest" in kept
    assert "loses money on what it sells" in kept
    assert len(dropped) == 1


def test_nothing_is_dropped_when_nothing_is_wrong() -> None:
    text = "The company cannot cover its interest. It loses money."
    kept, dropped = redact_bad_claims(text, [])
    assert kept == text and dropped == []


def test_every_sentence_carrying_the_value_goes() -> None:
    """A figure repeated in two sentences is wrong in both."""
    text = "Coverage is 9.99 here. Elsewhere it is also 9.99. Leverage is high."
    kept, dropped = redact_bad_claims(text, ["9.99 cites interest_coverage"])
    assert "9.99" not in kept and len(dropped) == 2
    assert "Leverage is high." in kept


# ---- the policy, through check_answer -------------------------------------

def test_a_mostly_good_answer_survives_as_a_partial() -> None:
    ctx = _ctx()
    text = (
        "The company holds 1.10 [current_ratio 2022-12-31] against its "
        "short-term bills. "
        "Its coverage is 1.10 [interest_coverage 2022-12-31]. "
        "It loses money on what it sells at -0.17 [net_margin 2022-12-31]. "
        "The distress is broad and long-standing across the balance sheet."
    )
    answer = check_answer(text, ctx)
    assert answer.allowed, "three good sentences should not be thrown away"
    assert answer.partial
    assert "1.10" in answer.text          # the correct use of 1.10 survives
    assert answer.dropped_claims
    assert PARTIAL_NOTICE.split("{")[0].strip()[:20] in answer.text, (
        "a partial answer must say that something was removed"
    )


def test_the_notice_names_what_was_wrong() -> None:
    """Silent repair is the failure mode this whole file guards against."""
    ctx = _ctx()
    text = (
        "Coverage is 1.10 [interest_coverage 2022-12-31]. "
        "The company loses money at -0.17 [net_margin 2022-12-31]. "
        "Liquidity is thin and the position has been deteriorating for years. "
        "None of this is a forecast about what happens next."
    )
    answer = check_answer(text, ctx)
    assert answer.partial
    assert "interest_coverage" in answer.text


def test_an_answer_that_is_mostly_bad_is_refused_outright() -> None:
    """Below the surviving threshold a partial is incoherent, and an analyst is
    better served by a clean refusal than a mangled paragraph."""
    ctx = _ctx()
    text = "Coverage is 1.10 [interest_coverage 2022-12-31]."
    answer = check_answer(text, ctx)
    assert not answer.allowed
    assert not answer.partial


def test_the_surviving_threshold_is_a_real_fraction() -> None:
    assert 0.0 < MIN_SURVIVING_SHARE < 1.0


def test_a_clean_answer_is_untouched() -> None:
    ctx = _ctx()
    text = "It holds 1.10 [current_ratio 2022-12-31] against short-term bills."
    answer = check_answer(text, ctx)
    assert answer.allowed and not answer.partial and not answer.dropped_claims


def test_advice_is_refused_and_never_partialled() -> None:
    """The scope guard returns before any recovery runs. An answer that
    recommended an action must not be trimmed into a compliant rephrasing of
    the same advice."""
    ctx = _ctx()
    text = (
        "You should sell this position immediately. "
        "It holds 1.10 [current_ratio 2022-12-31] against short-term bills. "
        "The balance sheet is stretched and coverage is negative."
    )
    answer = check_answer(text, ctx)
    assert not answer.allowed
    assert not answer.partial, "advice must be refused, not edited down"
