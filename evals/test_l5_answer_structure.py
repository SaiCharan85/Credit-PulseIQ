"""L5: structure follows the shape of the information, not its length.

The binary this replaced -- prose or a flat list -- could not express the
commonest real request: *"give me the subtopics and a brief understanding of
each"*. That wants an opening sentence, then a heading per subject, then
whichever of prose or bullets suits what sits under that heading. A flat list
loses the grouping; prose loses the scannability; neither is the answer.

Four shapes, each earning its place:

``prose``     a causal chain, where the connections between findings are the
              content and bullets would cut them
``bullets``   parallel, independent items a reader wants to scan
``sections``  several genuinely different subjects, each formatted on its own
              terms -- the mixed case
``table``     several items compared on the same few columns

The rule underneath all four: shape mirrors the information, never its length.
A long answer is not automatically a list, and a short one is not automatically
a sentence -- that assumption is what failed a correctly formatted nine-line
breakdown for having lines over twenty-two words.
"""

from __future__ import annotations

import pytest

from agents.qa import (
    SECTIONED_PROMPT,
    TABLE_DIRECTIVE,
    wants_points,
    wants_sections,
    wants_table,
)

SECTIONED = [
    "Break this down by subtopic with a brief understanding of each.",
    "give me the subtopics",
    "break it down by category",
    "group them by area",
    "walk through the different aspects",
    "structure the answer for me",
    "answer with headings",
    "explain each part",
]

TABULAR = [
    "Show me the key measures in a table.",
    "put it in tabular form",
    "line them up side by side",
    "give me the figures in columns",
]

FLAT = [
    "Give me the main risks as bullet points.",
    "list the key figures",
    "point wise please",
]

PLAIN = [
    "Why is this company at risk?",
    "What did the auditor say?",
    "Is coverage below one?",
]


@pytest.mark.parametrize("q", SECTIONED)
def test_multi_subject_requests_get_sections(q: str) -> None:
    assert wants_sections(q), f"would have been flattened into one list: {q!r}"


@pytest.mark.parametrize("q", TABULAR)
def test_tabular_requests_are_detected(q: str) -> None:
    assert wants_table(q)


@pytest.mark.parametrize("q", FLAT)
def test_a_flat_list_request_is_not_promoted_to_sections(q: str) -> None:
    """Headings on a single-subject answer are noise. "List the key figures"
    wants one list, not four headed groups of one item each."""
    assert wants_points(q)
    assert not wants_sections(q)


@pytest.mark.parametrize("q", PLAIN)
def test_plain_questions_stay_prose(q: str) -> None:
    assert not wants_sections(q) and not wants_points(q) and not wants_table(q)


def test_sections_and_points_can_be_asked_for_together() -> None:
    """"Break it down by subtopic in points" asks for both. Sections win,
    because the sectioned shape can still use bullets inside a section while a
    flat list cannot recover the grouping."""
    q = "break it down by category, in points"
    assert wants_sections(q) and wants_points(q)


def test_the_sectioned_example_demonstrates_mixed_shapes() -> None:
    """The example is the instruction. Its two sections are formatted
    *differently from each other* -- one prose, one bullets -- because stating
    that as a rule did not produce it and showing it does."""
    body = SECTIONED_PROMPT
    assert body.count("##") >= 2, "needs at least two headings to show grouping"
    assert "- obligations exceed" in body, "one section must demonstrate bullets"
    # ...and the prose section must not be a list.
    prose_section = body.split("## Ability to service debt")[1].split("##")[0]
    assert "- " not in prose_section, "the prose section must stay prose"


def test_the_guidance_names_when_each_shape_applies() -> None:
    for cue in ("Prose", "Bullets", "heading", "chain", "parallel"):
        assert cue in SECTIONED_PROMPT, f"missing guidance on {cue}"


def test_the_table_directive_demands_consistent_columns() -> None:
    """A table whose rows disagree about what the columns mean is worse than a
    list -- the reader scans a column and gets nonsense."""
    import re

    # Normalised: the directive is wrapped, so the phrase spans a line break.
    flat = re.sub(r"\s+", " ", TABLE_DIRECTIVE)
    assert "same columns for every row" in flat
    assert "not in a ragged extra column" in flat


def test_both_prompts_still_render() -> None:
    from agents.qa import FIGURE_RULE, LENGTH_NORMAL

    out = SECTIONED_PROMPT.format(length=LENGTH_NORMAL, figures=FIGURE_RULE)
    assert "{length}" not in out and "{figures}" not in out


def test_nesting_is_offered_but_bounded() -> None:
    """Two levels, and only where an item genuinely contains sub-items. Deeper
    hierarchies read as structure the evidence does not support, and nesting to
    show emphasis is the commonest way that happens."""
    import re

    flat = re.sub(r"\s+", " ", SECTIONED_PROMPT)
    assert "Two levels at most" in flat
    assert "not part of its parent" in flat
    # the example has to show a real parent/child pair, not just describe one
    assert "- 1.10 [current_ratio 2022-12-31]" in SECTIONED_PROMPT
