"""L5: the answer's shape is read off the question, not fixed in the prompt.

The bug this file pins: a reader asked for "at least 400 words" and got sixty.
The prose prompt carried "Plain prose, two to four sentences" as a standing
rule, so the reader's instruction was arguing with the system prompt, and the
system prompt won every time.

Two layers now decide the shape, and the split is the point:

* **Detection**, tested here, for what the reader said outright -- a word
  count, "in detail", "briefly", "in bullet points". A reader who names a
  number has not left anything to infer, and inferring anyway is how you end
  up ignoring them.
* **Judgement**, which cannot be unit-tested because it lives in the prompt:
  everything else falls to a clause telling the model to size the answer to
  the question's scope.

Length and format are independent -- "give me a detailed breakdown in points"
asks for both -- so they are chosen separately and substituted into one
coherent prompt rather than stacked on top of each other.
"""

from __future__ import annotations

import pytest

from agents.qa import (
    LENGTH_BRIEF,
    LENGTH_DETAILED,
    length_clause,
    requested_words,
    wants_detail,
    wants_points,
)

DETAIL_REQUESTS = [
    "can you in detail in at least 400 words abou this",  # verbatim, typo and all
    "explain this in detail",
    "give me a detailed answer",
    "elaborate on the liquidity position",
    "I want a comprehensive picture",
    "walk me through the whole thing",
    "can you expand on that",
    "give me the full explanation",
    "go in depth on the leverage",
    "tell me everything the assessment shows",
    "a thorough answer please",
    "longer answer than that",
]

BRIEF_REQUESTS = [
    "briefly, why is it at risk?",
    "in brief, what is wrong here",
    "short answer: is coverage below one?",
    "tl;dr",
    "give me the gist",
    "in a nutshell what is the problem",
    "one-sentence summary please",
]

NEUTRAL_REQUESTS = [
    "why is this company at risk?",
    "what did the auditor say?",
    "how has the current ratio moved?",
    "what would change your reading?",
]


@pytest.mark.parametrize("q", DETAIL_REQUESTS)
def test_detail_requests_are_detected(q: str) -> None:
    assert wants_detail(q), f"asked for depth and was not heard: {q!r}"
    assert length_clause(q).startswith("The reader has asked for a full")


@pytest.mark.parametrize("q", BRIEF_REQUESTS)
def test_brief_requests_are_detected(q: str) -> None:
    assert not wants_detail(q)
    assert length_clause(q) == LENGTH_BRIEF, f"asked for brevity and was not heard: {q!r}"


@pytest.mark.parametrize("q", NEUTRAL_REQUESTS)
def test_neutral_questions_are_left_to_judgement(q: str) -> None:
    """No explicit instruction means the model sizes it, not a regex."""
    clause = length_clause(q)
    assert clause not in (LENGTH_BRIEF,)
    assert not clause.startswith("The reader has asked for a full")
    assert "Match the length of the answer" in clause


@pytest.mark.parametrize(
    ("q", "n"),
    [
        ("at least 400 words", 400),
        ("in about 250 words please", 250),
        ("write 1000 words on this", 1000),
        ("minimum of 150 words", 150),
        ("~300 words", 300),
        ("500+ words", 500),
    ],
)
def test_word_targets_are_read_literally(q: str, n: int) -> None:
    assert requested_words(q) == n
    assert f"at least {n} words" in length_clause(q)


def test_a_word_count_beats_a_brevity_word() -> None:
    """"a short 500 words" names a number; the number is not a guess."""
    assert requested_words("give me a short 500 words on this") == 500
    assert wants_detail("give me a short 500 words on this")


def test_detail_beats_brevity_when_both_appear() -> None:
    """Checked in that order deliberately: "briefly but in detail" is a reader
    asking for depth without waffle, not a contradiction to refuse."""
    assert wants_detail("briefly but in detail, why is it at risk")


def test_length_and_format_are_independent() -> None:
    q = "give me a detailed breakdown in points"
    assert wants_detail(q) and wants_points(q), "both dimensions must register"


def test_stray_numbers_are_not_read_as_word_counts() -> None:
    """A figure in a question is usually a figure, not a length instruction."""
    for q in [
        "why is the current ratio 0.62",
        "what happened in 2024",
        "is coverage below 1.5",
        "explain the 1.45 leverage figure",
    ]:
        assert requested_words(q) is None, f"misread a figure as a word count: {q!r}"


def test_the_detailed_clause_forbids_padding() -> None:
    """A reader asking for length invites invention, in a system whose entire
    claim is that every figure traces to a filing. The clause has to say what
    legitimate expansion is."""
    clause = LENGTH_DETAILED.format(target="")
    assert "never by inventing" in clause
    assert "not in the assessment" in clause
    assert "does not carry more" in clause, "must license a short honest answer"


def test_prompts_still_format_cleanly() -> None:
    """Both prompts carry the slot; a missing one would raise at answer time."""
    from agents.qa import FIGURE_RULE, POINTS_PROMPT, SYSTEM_PROMPT

    for prompt in (SYSTEM_PROMPT, POINTS_PROMPT):
        rendered = prompt.format(length=LENGTH_BRIEF, figures=FIGURE_RULE)
        assert LENGTH_BRIEF in rendered
        assert "{length}" not in rendered
        assert "{figures}" not in rendered


def test_worked_examples_demonstrate_the_citation_format() -> None:
    """Rules alone do not change this model's output -- that is recorded in the
    module docstring, about bullet points. The examples have to show the
    bracket or the model will not write one, and every figure comes back
    untagged."""
    from agents.qa import POINTS_PROMPT, SYSTEM_PROMPT

    for prompt in (SYSTEM_PROMPT, POINTS_PROMPT):
        assert "[current_ratio 2022-12-31]" in prompt or "[net_margin 2022-12-31]" in prompt
