"""L5: a real figure on the wrong measure.

The grounding check asks whether a number appears in the context. That catches
invention and misses swapping, and swapping is the worse failure:

    context:  current_ratio = 1.10      interest_coverage = -1.06
    answer:   "Interest coverage is 1.10."

Every digit in that sentence traces to the assessment. It passes the
digits-present check cleanly, it is internally consistent, and it is false. A
reader has no way to catch it -- checking the number against the memo confirms
it, because the number is genuinely there.

The check added here is deliberately narrow: a value is only reported when a
measure is named close before it *and* the value provably belongs to a
different measure. Both halves matter. Without the second, every threshold and
comparator in a sentence ("1.10 against the 1.00 level") becomes a false
positive, and a guard that cries wolf gets the answer withheld for being
correct -- which is how a safety control ends up switched off.

Attribution in prose that names no measure is not checked and cannot be. That
is recorded in the last test rather than hidden.
"""

from __future__ import annotations

import pytest

from agents.qa import REFUSAL_MISATTRIBUTED, attribution_violations, check_answer

CONTEXT = """\
CIK 28823, prediction date 2024-07-01
SIGNAL: severe_risk  confidence 0.8

[Credit distress] (backtested)
  - current_ratio = 1.10 at period end 2022-12-31 (flag) -- short-term assets \
against short-term obligations; near or below 1 means little cushion
  - interest_coverage = -1.06 at period end 2022-12-31 (severe) -- operating \
income against interest owed; below 1 means not earning enough
  - liabilities_to_assets = 1.45 at period end 2022-12-31 (severe) -- above 0.7 \
is heavily levered
  - net_margin = -0.17 at period end 2022-12-31 (severe)

[Series] current_ratio -- short-term assets against short-term obligations
  direction over the period: deteriorating
  2017-12-31: 1.38
  2018-12-31: 1.40
  2022-12-31: 1.10
"""

SWAPPED = [
    "Interest coverage is 1.10, so it cannot service its debt.",
    "The current ratio of -1.06 shows there is no cushion.",
    "Its net margin stands at 1.45, which is severe.",
    "The Altman distress score is 1.10.",
    "Interest coverage was 1.38 at the end of 2017.",
]

#: Swaps this check does **not** catch, recorded rather than hidden.
#:
#: Catching these needs loose aliases -- "leverage", "cushion", "obligations",
#: "danger zone" -- and those are the words a long answer uses while explaining
#: something else. With them in, five correct answers were withheld on the live
#: battery, including a paragraph about the Altman score followed later by the
#: current ratio, which charged 1.10 to Altman. The narrow check catches every
#: swap that names a measure precisely, and misses ones phrased loosely. That
#: is the trade taken, on the grounds that a guard which blocks correct output
#: gets switched off and then protects nothing.
NOT_CAUGHT = [
    "Leverage sits at -0.17 of the balance sheet.",
    "The cushion for short-term bills is -1.06.",
]

CORRECT = [
    "Interest coverage is -1.06, so it cannot service its debt.",
    "The current ratio of 1.10 leaves very little cushion.",
    "Its net margin stands at -0.17, meaning it loses money on what it sells.",
    "Obligations are 1.45 times the value of everything it owns.",
    "The cushion for short-term bills fell from 1.38 in 2017 to 1.10 in 2022.",
]


@pytest.mark.parametrize("text", SWAPPED)
def test_a_real_figure_on_the_wrong_measure_is_caught(text: str) -> None:
    assert attribution_violations(text, CONTEXT), f"swap went undetected: {text!r}"
    answer = check_answer(text, CONTEXT)
    assert not answer.allowed
    assert answer.reason == REFUSAL_MISATTRIBUTED


@pytest.mark.parametrize("text", CORRECT)
def test_correct_attribution_passes(text: str) -> None:
    found = attribution_violations(text, CONTEXT)
    assert not found, f"correct sentence blocked: {text!r} -> {found}"
    assert check_answer(text, CONTEXT).allowed


def test_the_digits_check_alone_would_have_missed_every_swap() -> None:
    """The reason this file exists. Each swapped sentence uses only numbers the
    context genuinely contains, so the older check clears all of them."""
    from agents.qa import _numerals

    known = _numerals(CONTEXT)
    for text in SWAPPED:
        stray = [n for n in _numerals(text) if not any(n in k or k in n for k in known)]
        assert not stray, f"expected the digits check to be fooled by {text!r}"


THRESHOLDS_AND_COMPARATORS = [
    "It has 1.10 in short-term assets for every 1.00 it owes.",
    "Anything above 0.7 counts as heavily levered, and this sits at 1.45.",
    "Coverage below 1 means interest is not covered; here it is -1.06.",
    "The figure is 1.10, against a conventional level of 1.00.",
]


@pytest.mark.parametrize("text", THRESHOLDS_AND_COMPARATORS)
def test_thresholds_and_comparators_are_not_flagged(text: str) -> None:
    """A guard that blocks correct answers gets disabled, and then protects
    nothing. Numbers with no wrong owner to name are left alone."""
    found = attribution_violations(text, CONTEXT)
    assert not found, f"comparator flagged as a swap: {text!r} -> {found}"


def test_series_values_are_owned_by_their_metric() -> None:
    """Dated points carry attribution too, or the trend feature reopens the
    hole it was built to avoid."""
    assert attribution_violations("Interest coverage was 1.40 in 2018.", CONTEXT)
    assert not attribution_violations("The current ratio was 1.40 in 2018.", CONTEXT)


def test_a_distant_metric_name_does_not_claim_a_number() -> None:
    """The window is bounded so a subject two sentences back cannot be read as
    claiming a later figure."""
    text = (
        "Interest coverage is the measure that matters most for a levered "
        "borrower, and this is a company with a great deal of debt on its "
        "balance sheet relative to its size. Obligations are 1.45 times assets."
    )
    assert not attribution_violations(text, CONTEXT)


@pytest.mark.parametrize("text", NOT_CAUGHT)
def test_loosely_phrased_swaps_are_a_known_gap(text: str) -> None:
    """Fails loudly if someone widens the aliases without re-running the live
    battery. Turning one of these green is progress only if the false-positive
    rate stays at zero -- move it into SWAPPED and delete it from here."""
    assert not attribution_violations(text, CONTEXT), (
        f"{text!r} is now caught -- good, if and only if "
        "python -m evals.run_response_eval still withholds nothing"
    )


def test_an_intervening_figure_takes_the_name() -> None:
    """The rule that removed the false positives. A measure named before some
    other number was claiming that one, not a later one."""
    text = "The Altman distress score is -1.93, and it holds 1.10 in short-term assets."
    assert not attribution_violations(text, CONTEXT)
    # ...but with nothing in between, the claim stands and the swap is caught.
    assert attribution_violations("The Altman distress score is 1.10.", CONTEXT)


def test_prose_naming_no_measure_is_not_checked() -> None:
    """A stated limit. Where the sentence names nothing, attribution cannot be
    verified deterministically and this returns clean -- the digits check is
    the only guarantee that remains."""
    assert not attribution_violations("The figure is 1.10 and that is the concern.", CONTEXT)


def test_an_empty_context_disables_the_check_rather_than_blocking() -> None:
    assert attribution_violations("Interest coverage is 1.10.", "") == []
