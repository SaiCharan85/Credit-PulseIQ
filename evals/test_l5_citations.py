"""L5: the model declares where each figure came from, and Python checks it.

Three guards on numbers, in increasing strength:

1. **presence** -- the digits appear somewhere in the assessment. Catches
   invention. Blind to a real figure quoted against the wrong measure.
2. **inferred attribution** -- parse the measure out of the prose near the
   number. Measured at **0 of 38 numerals** on real answers: the prompt tells
   the model to write for a non-specialist, so it says "the cushion for
   short-term bills" and names no measure at all. A guard that only fires on
   phrasing the system forbids is not a guard.
3. **declared attribution**, this file -- the model writes
   ``1.10 [current_ratio 2022-12-31]`` and Python checks the measure *and* the
   period against the source, then strips the bracket. Measured at **30 of 35**
   on the same answers, the other five being thresholds like "1.00" that name
   no measure because they are not claims about one.

A wrong bracket blocks the answer: the model asserted a provenance and was
wrong, which is worse than staying silent. A missing bracket does not block --
it falls back to guard 1 and is *reported* as unverified, because withholding
a good answer over a missing bracket trains people to switch the guard off.
"""

from __future__ import annotations

import pytest

from agents.qa import REFUSAL_MISATTRIBUTED, check_answer, verify_citations

CONTEXT = """\
CIK 28823, prediction date 2024-07-01
Latest annual period visible: 2022-12-31

[Credit distress] (backtested)
  - current_ratio = 1.10 at period end 2022-12-31 (flag)
  - interest_coverage = -1.06 at period end 2022-12-31 (severe)
  - net_margin = -0.17 at period end 2022-12-31 (severe)

[Series] current_ratio
  direction over the period: deteriorating
  2017-12-31: 1.38
  2022-12-31: 1.10
"""

#: The same context plus the overall score, which is rendered with a colon
#: rather than an equals sign -- the punctuation difference that made it
#: uncitable and withheld a real answer.
def _scored():
    """A real rendered Context, bindings included -- the production path.

    The string form of this context is what broke: ``risk_score`` is
    written with a colon, the fallback parser only reads ``metric =
    value``, so the score was absent from the bindings and citing it
    withheld the answer.
    """
    from agents.qa import build_context

    return build_context({
        "cik": 28823,
        "as_of": "2024-07-01",
        "triage": {"latest_period_end": "2022-12-31"},
        "memo": {
            "signal": "healthy", "confidence": 0.9, "risk_score": 88.0,
            "summary": "", "limitations": [], "routing": [], "audit_trail": [],
            "sections": [{"title": "Credit distress", "tier": "backtested",
                          "body": "", "evidence": [
                {"metric": "current_ratio", "value": 1.10,
                 "period_end": "2022-12-31", "note": "flag"}]}],
        },
    })


def test_a_correct_citation_verifies_and_is_stripped() -> None:
    c = verify_citations("It holds 1.10 [current_ratio 2022-12-31] in assets.", CONTEXT)
    assert c.verified == 1
    assert not c.bad
    assert "[current_ratio" not in c.text, "the bracket must not reach the reader"
    assert c.text == "It holds 1.10 (2022) in assets."


def test_the_period_survives_the_strip() -> None:
    """Removing the whole bracket took the dates out with it, and a trend
    answer came back as "it began at 1.38 and reached 1.40" -- every figure
    verified against a period the reader could no longer see."""
    c = verify_citations(
        "It began at 1.38 [current_ratio 2017-12-31] and is now "
        "1.10 [current_ratio 2022-12-31].",
        CONTEXT,
    )
    assert c.verified == 2
    assert "(2017)" in c.text and "(2022)" in c.text


def test_the_year_is_not_repeated_when_the_prose_already_says_it() -> None:
    c = verify_citations("In 2017 it stood at 1.38 [current_ratio 2017-12-31].", CONTEXT)
    assert c.text == "In 2017 it stood at 1.38."


def test_a_date_is_not_parsed_as_a_cited_value() -> None:
    """Without a lookbehind the value group matches "-31" off the tail of a
    date, so a model that wrote the period out longhand before its own bracket
    was reported as citing -31 -- and a correct answer was withheld."""
    c = verify_citations(
        "As at 2022-12-31 [current_ratio 2022-12-31] the ratio held.", CONTEXT
    )
    assert not c.bad, f"a date was misread as a figure: {c.bad}"


def test_the_wrong_measure_is_caught() -> None:
    c = verify_citations("Coverage is 1.10 [interest_coverage 2022-12-31].", CONTEXT)
    assert c.bad and "whose value is -1.06" in c.bad[0]
    assert not check_answer("Coverage is 1.10 [interest_coverage 2022-12-31].", CONTEXT).allowed


def test_the_wrong_period_is_caught() -> None:
    """The failure the earlier guards could not see at all: right measure,
    right value, wrong year. 1.38 is the 2017 reading, not the 2022 one."""
    c = verify_citations("The ratio was 1.38 [current_ratio 2022-12-31].", CONTEXT)
    assert c.bad, "a value from the wrong period must not pass"
    assert "1.38" in c.bad[0]
    # ...and the same value cited against its own period is fine.
    assert verify_citations("It was 1.38 [current_ratio 2017-12-31].", CONTEXT).verified == 1


def test_a_measure_not_in_the_assessment_is_caught() -> None:
    c = verify_citations("Debt to equity is 2.40 [debt_to_equity 2022-12-31].", CONTEXT)
    assert c.bad and "not in the assessment" in c.bad[0]


def test_a_bad_citation_blocks_with_its_own_reason() -> None:
    """Distinct from the ungrounded message: a reader told "that figure is not
    in the assessment" about a figure that *is* would rightly lose trust in
    the guard."""
    answer = check_answer("Coverage is 1.10 [interest_coverage 2022-12-31].", CONTEXT)
    assert not answer.allowed
    assert answer.reason == REFUSAL_MISATTRIBUTED


def test_untagged_figures_are_reported_not_blocked() -> None:
    """Falling back rather than refusing. A missing bracket is a reporting gap,
    not evidence of anything wrong -- and blocking on it would make the guard
    something people route around."""
    answer = check_answer("It holds 1.10 in short-term assets.", CONTEXT)
    assert answer.allowed
    assert answer.figures_verified == 0
    assert answer.figures_untagged == 1


def test_a_verified_figure_is_not_also_counted_as_untagged() -> None:
    """Stripping the bracket leaves the numeral in the prose, where a naive
    sweep counts it again -- which reported four cited figures as "4 verified,
    4 untagged" and halved the coverage number."""
    c = verify_citations(
        "It holds 1.10 [current_ratio 2022-12-31] against -1.06 "
        "[interest_coverage 2022-12-31] of cover.",
        CONTEXT,
    )
    assert (c.verified, c.untagged) == (2, 0)


@pytest.mark.parametrize(
    "text",
    [
        "It holds 1.10 [current_ratio 2022-12-31] for every 1.00 it owes.",
        "Anything below 1.00 is thin; here it is -1.06 [interest_coverage 2022-12-31].",
    ],
)
def test_thresholds_are_untagged_without_being_wrong(text: str) -> None:
    """A comparator names no measure because it is not a claim about one. It
    should count as untagged, never as a bad citation."""
    c = verify_citations(text, CONTEXT)
    assert not c.bad
    assert c.verified == 1
    assert c.untagged == 1


def test_dates_and_years_are_not_counted_as_figures() -> None:
    c = verify_citations("Between 2017 and 2022 it fell to 1.10 [current_ratio 2022-12-31].",
                         CONTEXT)
    assert (c.verified, c.untagged) == (1, 0)


def test_a_period_the_context_never_binds_is_reported_clearly() -> None:
    c = verify_citations("It was 1.10 [current_ratio 2019-12-31].", CONTEXT)
    assert c.bad
    assert "2017-12-31" in c.bad[0] and "2022-12-31" in c.bad[0], (
        "the message should name the periods that do exist, or a reader "
        "cannot tell whether the model or the data is at fault"
    )


def test_an_empty_context_disables_checking_rather_than_blocking() -> None:
    c = verify_citations("It holds 1.10 [current_ratio 2022-12-31].", "")
    assert not c.bad and c.verified == 0


def test_confidence_as_a_probability_is_blocked_by_framing_not_by_ban() -> None:
    """Found by the adversarial probe: asked for a probability of default, the
    model cited the system's own confidence, which would have read as an 80%
    chance of failure.

    The first fix banned the field outright and promptly blocked this, from a
    real analyst: "if this company is healthy, why was it at default in 2020"
    -> withheld, "3 cites risk_score, which is not in the assessment". The
    score *was* in the assessment and the question was entirely fair. Banning a
    field to stop one misuse is the crude fix; the framing is what is wrong, so
    the framing is what is checked."""
    c = verify_citations("The probability of default is 0.80 [confidence].", CONTEXT)
    assert c.bad
    assert "not a chance of default" in c.bad[0]


def test_the_score_itself_is_quotable() -> None:
    """The regression the ban caused. Describing the score must work."""
    ctx = _scored()
    c = verify_citations("Its risk score is 88 [risk_score].", ctx)
    assert not c.bad, f"blocked a plain statement of the score: {c.bad}"
    assert c.verified == 1


def test_the_risk_score_is_not_a_percentage() -> None:
    ctx = _scored()
    c = verify_citations("There is an 88 [risk_score] percent chance of failure.", ctx)
    assert c.bad
    assert "not a chance of default" in c.bad[0]


def test_a_real_measure_is_unaffected_by_that_rule() -> None:
    assert verify_citations("It holds 1.10 [current_ratio 2022-12-31].", CONTEXT).verified == 1


def test_a_bracket_on_the_wrong_number_is_forgiven_not_refused() -> None:
    """"3 out of 100 [risk_score]" attaches the bracket to 100. The model was
    told to put it immediately after its figure and did not quite; refusing the
    whole answer over bracket placement punishes the reader for that."""
    ctx = _scored()
    c = verify_citations("The score is 88 out of 100 [risk_score].", ctx)
    assert not c.bad
    assert c.verified == 1


def test_a_legal_reference_is_not_counted_as_a_figure() -> None:
    """"Chapter 11" is the name of a law, not a measurement. Counting its 11
    as an unverified figure understated citation coverage and tripped the
    figure audit, which reported it as a number on screen with no source."""
    c = verify_citations(
        "It holds 1.10 [current_ratio 2022-12-31] and has filed for Chapter 11.",
        CONTEXT,
    )
    assert c.verified == 1
    assert c.untagged == 0, f"a legal reference was counted: {c.untagged}"


def test_form_and_item_references_are_also_excluded() -> None:
    for text in ("reported on Form 10-K", "disclosed under Item 1.03",
                 "an 8-K filed that week", "Section 13 requires it"):
        c = verify_citations(text, CONTEXT)
        assert c.untagged == 0, f"counted a reference as a figure: {text!r}"
