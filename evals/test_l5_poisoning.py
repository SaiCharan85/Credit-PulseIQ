"""L5: numbers may only be grounded in things this system computed.

Two write paths into the trusted number set, both real, neither obvious.

**The filer.** A going-concern passage is text the assessed company wrote. Put
it in the context and its numbers become groundable: "management believes the
current ratio of 9.99 demonstrates ample liquidity" is an adversarial write,
and the check confirms 9.99 for a company whose ratio we computed at 1.10. The
subject of an investigation should not be able to edit its own evidence file.

**The model.** A section body is the agent's own narrative. A figure invented
there is never cited as evidence, so the memo's numeric critic -- which
recomputes *cited* figures -- has nothing to check, and the figure survives.
Ask a follow-up question and that prose is now context: the model's own
hallucination, laundered into verified ground by the next turn. This is the
one that would have gone unnoticed, because nothing about it looks like an
attack.

The fix is not detection. It is refusing to ground a number in prose at all --
:func:`trusted_context` keeps the structured evidence, series rows, score and
limitation lines, and drops model narrative and quoted filing text. Evidence
entries carry the same figures in checkable form, so nothing legitimate is
lost.
"""

from __future__ import annotations

from agents.qa import check_answer, memo_context, trusted_context

POISONED_QUOTE = (
    'Going-concern language found: "management believes the current ratio of '
    '9.99 and interest coverage of 42.00 demonstrate ample liquidity for the '
    'coming year"'
)
POISONED_NARRATIVE = (
    "The company is under pressure, though its debt service coverage of 7.77 "
    "provides some offset against near-term maturities."
)


def _payload(body: str) -> dict:
    return {
        "cik": 28823,
        "as_of": "2024-07-01",
        "triage": {"latest_period_end": "2022-12-31"},
        "memo": {
            "signal": "severe_risk",
            "confidence": 0.8,
            "risk_score": 88.0,
            "summary": "6 metrics past the severe threshold",
            "sections": [
                {
                    "title": "Credit distress",
                    "tier": "backtested",
                    "body": body,
                    "evidence": [
                        {"metric": "current_ratio", "value": 1.10,
                         "period_end": "2022-12-31", "note": "flag"},
                        {"metric": "interest_coverage", "value": -1.06,
                         "period_end": "2022-12-31", "note": "severe"},
                    ],
                }
            ],
            "limitations": ["most recent annual period ends 2022-12-31 -- 548 days before"],
            "routing": [],
            "audit_trail": [],
        },
    }


def test_a_figure_quoted_from_the_filing_cannot_ground_a_claim() -> None:
    payload = _payload(POISONED_QUOTE)
    # The model still *sees* the quote -- withholding evidence from it would be
    # a different kind of wrong. It simply cannot assert the number as ours.
    assert "9.99" in memo_context(payload)
    assert "9.99" not in trusted_context(payload)

    answer = check_answer("The current ratio is 9.99.", memo_context(payload),
                          trusted=trusted_context(payload))
    assert not answer.allowed, "the filer wrote that number, not us"
    assert "9.99" in answer.ungrounded_numbers


def test_a_figure_the_model_invented_in_prose_cannot_ground_a_later_answer() -> None:
    """The quieter failure: no attacker, just the agent's own narrative
    becoming trusted context on the next question."""
    payload = _payload(POISONED_NARRATIVE)
    assert "7.77" in memo_context(payload)
    assert "7.77" not in trusted_context(payload)

    answer = check_answer("Debt service coverage is 7.77.", memo_context(payload),
                          trusted=trusted_context(payload))
    assert not answer.allowed


def test_computed_evidence_still_grounds_normally() -> None:
    """The check has to keep passing everything real, or it gets removed."""
    payload = _payload(POISONED_QUOTE)
    trusted = trusted_context(payload)
    for claim in [
        "It holds 1.10 in short-term assets for every unit it owes.",
        "It covers -1.06 of the interest it owes.",
        "The risk score is 88 out of 100.",
    ]:
        answer = check_answer(claim, memo_context(payload), trusted=trusted)
        assert answer.allowed, f"blocked a real figure: {claim} -> {answer.ungrounded_numbers}"


def test_system_generated_limitations_still_ground() -> None:
    """Limitation lines are written by this system, not by the filer, so the
    dates and day counts in them remain quotable."""
    payload = _payload(POISONED_QUOTE)
    answer = check_answer(
        "The figures are from a period ending 2022-12-31, some 548 days earlier.",
        memo_context(payload), trusted=trusted_context(payload),
    )
    assert answer.allowed


def test_the_citation_check_catches_it_too_when_the_model_tags() -> None:
    """Defence in depth: a tagged figure is checked against the measure's real
    value, so a poisoned number fails there as well as at grounding."""
    payload = _payload(POISONED_QUOTE)
    answer = check_answer(
        "The current ratio is 9.99 [current_ratio 2022-12-31].",
        memo_context(payload), trusted=trusted_context(payload),
    )
    assert not answer.allowed


def test_series_rows_survive_the_projection() -> None:
    payload = _payload(POISONED_QUOTE)
    payload["trends"] = [{
        "metric": "current_ratio", "direction": "deteriorating",
        "points": [{"period_end": "2017-12-31", "value": 1.38},
                   {"period_end": "2022-12-31", "value": 1.10}],
    }]
    trusted = trusted_context(payload)
    assert "1.38" in trusted, "dated series rows are computed, not narrated"
    assert "9.99" not in trusted
