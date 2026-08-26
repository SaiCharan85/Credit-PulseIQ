"""L5: a question about movement gets the movement loaded, or gets nothing.

The hallucination pressure this removes is specific and worth naming. A memo
holds one figure per measure. Ask it for "a detailed report with the trends"
and three things could happen:

1. the model says nothing about trends -- the reader's question is ignored;
2. the model invents a direction -- the failure this whole system exists to
   prevent, and the most convincing kind, because "deteriorating since 2021"
   reads exactly like a fact;
3. the model guesses and the grounding check blocks the whole answer -- safe,
   and useless.

None of those is an answer. So a trend question triggers the real series being
fetched *before* the model is asked, from the same as-of filtered facts the
memo came from. Then "expand" is served by more evidence rather than more
prose, and the grounding check has something true to validate against.

The rule that keeps it honest in the other direction is tested last: where no
series exists, the prompt requires the model to say the assessment shows a
single period rather than characterise a direction. One figure has no
direction.
"""

from __future__ import annotations

import pytest

from agents.qa import LENGTH_DETAILED, memo_context, wants_detail, wants_trends

TREND_QUESTIONS = [
    "how has the current ratio moved over time?",
    "what is the trend here",
    "show me the history of its leverage",
    "has coverage improved or worsened",
    "what has happened over the last few years",
    "what is the trajectory",
    "is it getting worse",
    "how has this changed since 2019",
    "year-on-year, what does this look like",
]

SNAPSHOT_QUESTIONS = [
    "why is this company at risk?",
    "what did the auditor say?",
    "is coverage below one?",
    "what is the biggest problem",
]


@pytest.mark.parametrize("q", TREND_QUESTIONS)
def test_movement_questions_load_the_series(q: str) -> None:
    assert wants_trends(q), f"would have had to invent a trend for: {q!r}"


@pytest.mark.parametrize("q", SNAPSHOT_QUESTIONS)
def test_snapshot_questions_do_not(q: str) -> None:
    assert not wants_trends(q)


def test_a_detailed_report_request_counts_as_needing_history() -> None:
    """"A detailed report" implies the shape of the position over time, and a
    reader who asks for one and gets a single-period snapshot has been given a
    partial answer without being told it was partial."""
    q = "Give me a detailed report on this company with the trends over the years."
    assert wants_trends(q) and wants_detail(q)
    # detail alone should not trigger a fetch; "report" alone should not either
    assert not wants_trends("explain this in detail")
    assert not wants_trends("give me the report")


def _payload(trends=None):
    return {
        "cik": 28823,
        "as_of": "2024-07-01",
        "memo": {
            "signal": "severe_risk",
            "confidence": 0.8,
            "summary": "several measures past the severe threshold",
            "sections": [
                {
                    "title": "Credit distress",
                    "tier": "backtested",
                    "body": "",
                    "evidence": [
                        {"metric": "current_ratio", "value": 1.10,
                         "period_end": "2022-12-31", "note": "flag"},
                    ],
                }
            ],
            "limitations": [],
            "routing": [],
            "audit_trail": [],
        },
        **({"trends": trends} if trends is not None else {}),
    }


SERIES = [
    {
        "metric": "current_ratio",
        "direction": "deteriorating",
        "points": [
            {"period_end": "2017-12-31", "value": 1.38},
            {"period_end": "2018-12-31", "value": 1.40},
            {"period_end": "2019-12-31", "value": None},
            {"period_end": "2022-12-31", "value": 1.10},
        ],
    }
]


def test_series_reach_the_model_as_dated_values() -> None:
    ctx = memo_context(_payload(SERIES))
    assert "[Series] current_ratio" in ctx
    assert "direction over the period: deteriorating" in ctx
    assert "2017-12-31: 1.38" in ctx
    assert "2022-12-31: 1.10" in ctx


def test_an_unreported_period_is_stated_not_skipped() -> None:
    """A filer that stops reporting a line item is often a filer in trouble.
    Dropping the row would hide that and, worse, would make the series look
    continuous across a gap it does not cover."""
    ctx = memo_context(_payload(SERIES))
    assert "2019-12-31: not reported" in ctx


def test_context_without_trends_is_unchanged() -> None:
    """Loading history must be additive: a snapshot question sees exactly what
    it saw before, so no answer changes shape because of this feature."""
    assert "[Series]" not in memo_context(_payload())


def test_a_one_point_series_is_dropped() -> None:
    """One point is not a trend, and presenting it under a heading that says
    'Series' invites the model to describe a direction it cannot see."""
    thin = [{"metric": "current_ratio", "direction": "flat",
             "points": [{"period_end": "2022-12-31", "value": 1.10}]}]
    assert "[Series]" not in memo_context(_payload(thin))


def test_every_series_number_is_groundable() -> None:
    """The series values must appear in the context verbatim, or the grounding
    check will block the model for quoting the evidence we handed it."""
    from agents.qa import check_answer

    ctx = memo_context(_payload(SERIES))
    answer = "The cushion fell from 1.38 at the end of 2017 to 1.10 by the end of 2022."
    assert check_answer(answer, ctx).allowed


def test_the_prompt_forbids_direction_without_a_series() -> None:
    clause = LENGTH_DETAILED.format(target="")
    assert "[Series]" in clause
    assert "one figure has no direction" in clause.lower()
    assert "not reported" in clause
