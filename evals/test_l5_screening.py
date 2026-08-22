"""L5: screening answers must be dated measurements, never forecasts.

"Give me the best companies to survive this year" bundles an answerable
request with an unanswerable one. Ranking filers by measured evidence is
information; saying who will survive is a forecast this system cannot make and
the part that creates liability.

These tests pin the separation: the ranking is emitted in the past tense with
its as-of date, the non-forecast statement is attached rather than optional,
and the system's own miss rate goes out with the list.
"""

from __future__ import annotations

from datetime import date

import pytest

from agents.screening import (
    FALSE_CONFIDENCE_RATE,
    build,
    is_forward_looking,
    is_screening,
    rank_rows,
)

AS_OF = date(2024, 7, 1)


def row(cik: int, signal: str, score: float | None, status: str = "ok") -> dict:
    return {"cik": cik, "status": status, "signal": signal,
            "risk_score": score, "confidence": 0.8}


ROWS = [
    row(1, "severe_risk", 95.0),
    row(2, "healthy", 8.0),
    row(3, "elevated_risk", 60.0),
    row(4, "watch", 30.0),
    row(5, "healthy", 5.0),
    row(6, "healthy", None, status="unavailable"),
]


class TestIntentDetection:
    @pytest.mark.parametrize(
        "q",
        [
            "give the list of best companies to survive this year",
            "which companies are safest",
            "rank my companies by risk",
            "give me names of the least risky",
            "what are the worst names across my portfolio",
        ],
    )
    def test_screening_questions_are_recognised(self, q: str) -> None:
        assert is_screening(q)

    @pytest.mark.parametrize(
        "q", ["why is this company at risk?", "what does the auditor say?"]
    )
    def test_single_company_questions_are_not(self, q: str) -> None:
        assert not is_screening(q)

    @pytest.mark.parametrize(
        "q",
        [
            "which companies will survive this year",
            "who is going to default next quarter",
            "what is the outlook",
            "predict which names fail",
        ],
    )
    def test_forward_looking_questions_are_flagged(self, q: str) -> None:
        assert is_forward_looking(q)

    def test_a_dated_backward_question_is_not_flagged(self) -> None:
        assert not is_forward_looking("which companies scored lowest as of last June")


class TestRanking:
    def test_ascending_puts_the_healthiest_first(self) -> None:
        assert rank_rows(ROWS, ascending=True)[0]["signal"] == "healthy"

    def test_descending_puts_the_worst_first(self) -> None:
        assert rank_rows(ROWS, ascending=False)[0]["signal"] == "severe_risk"

    def test_unreadable_filers_are_excluded_not_ranked_safe(self) -> None:
        """A filer we could not read must never appear at the safe end."""
        ranked = rank_rows(ROWS, ascending=True)
        assert 6 not in [r["cik"] for r in ranked]

    def test_score_breaks_ties_within_a_signal(self) -> None:
        ranked = rank_rows(ROWS, ascending=True)
        healthy = [r["cik"] for r in ranked if r["signal"] == "healthy"]
        assert healthy[0] == 5  # score 5.0 ranks safer than 8.0


class TestTheAnswerCannotReadAsAForecast:
    def test_the_as_of_date_is_stated(self) -> None:
        assert str(AS_OF) in build("safest companies", ROWS, AS_OF).text

    def test_the_non_forecast_statement_is_always_attached(self) -> None:
        """Not conditional on being asked -- the reader who does not think to
        ask is the one who assumes it is a prediction."""
        answer = build("which companies are safest", ROWS, AS_OF)
        assert any("does not forecast" in c for c in answer.caveats)

    def test_a_forward_looking_question_is_refused_first(self) -> None:
        answer = build("best companies to survive this year", ROWS, AS_OF)
        assert answer.refused_forecast
        assert "not answered" in answer.caveats[0]

    def test_the_miss_rate_travels_with_the_list(self) -> None:
        answer = build("safest companies", ROWS, AS_OF)
        assert any(f"{FALSE_CONFIDENCE_RATE:.0%}" in c for c in answer.caveats)

    def test_unreadable_filers_are_disclosed(self) -> None:
        answer = build("safest companies", ROWS, AS_OF)
        assert any("could not be read" in c for c in answer.caveats)

    def test_an_empty_set_says_so_rather_than_ranking_nothing(self) -> None:
        answer = build("safest", [row(9, "healthy", None, status="unavailable")], AS_OF)
        assert "nothing to rank" in answer.text
        assert answer.rows == []

    def test_the_text_is_past_tense_about_a_dated_record(self) -> None:
        text = build("riskiest companies", ROWS, AS_OF).text
        assert "As of" in text and "public at that date" in text
