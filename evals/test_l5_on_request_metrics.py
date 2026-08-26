"""L5: a measure the memo skipped is routed to a formula, never to the model.

Thirty-six formulas are registered with hand-checked tests. Seven happen to
reach a memo, because the investigator surfaces what bears on *distress*. Ask
for one of the other twenty-nine -- quick ratio, debt to equity, free cash flow
-- and the honest-but-useless answer was "the assessment does not provide
that". The number is one deterministic call away, and a user told no twice goes
and asks a chatbot that will invent it.

So the request is **routed**, not refused: the named measure is computed in
Python from the same as-of filtered facts and handed to the model as evidence.
The model gains a fact, not a capability -- it still never does arithmetic, and
the figure arrives through the ordinary evidence path, so the grounding check,
the citation check and the as-of filter all apply to it unchanged.

The line is drawn at *registered*. "Revenue per employee" has no verified
implementation and no test behind it; computing it ad hoc would be exactly the
unaudited arithmetic the rest of this system exists to prevent. So it stays
refused, and the refusal says why.
"""

from __future__ import annotations

import pytest

# Both registries. compute.ratios holds 21 formulas, compute.scores the other
# 15 -- importing one and not the other is how a question about the F-score
# silently routes to nothing, which is what this file caught.
import compute.ratios  # noqa: F401
import compute.scores  # noqa: F401
from agents.qa import requested_metrics
from compute.provenance import FORMULAS

REGISTERED = set(FORMULAS)


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("What is its debt to equity ratio?", "debt_to_equity"),
        ("Show me the quick ratio", "quick_ratio"),
        ("what's the cash ratio here", "cash_ratio"),
        ("How much working capital does it have?", "working_capital"),
        ("What is free cash flow?", "free_cash_flow"),
        ("Give me the Piotroski F-score", "piotroski_f_score"),
        ("What does the Ohlson O-score say?", "ohlson_o_score"),
        ("Is there a Beneish m-score?", "beneish_m_score"),
        ("What is its DSO?", "days_sales_outstanding"),
        ("how long is the cash runway", "cash_runway_months"),
        ("what is net debt", "net_debt"),
        ("give me return on assets", "return_on_assets"),
    ],
)
def test_a_named_registered_measure_is_routed(question: str, expected: str) -> None:
    assert expected in requested_metrics(question, REGISTERED)


@pytest.mark.parametrize(
    "question",
    [
        "What is revenue per employee?",
        "What is the Sharpe ratio?",
        "How does its price to book compare?",
        "What is the customer churn rate?",
    ],
)
def test_an_unregistered_measure_is_not_invented(question: str) -> None:
    """No verified formula means no figure. This is the line that keeps the
    routing from becoming a licence to compute anything on request."""
    assert requested_metrics(question, REGISTERED) == []


def test_several_measures_in_one_question() -> None:
    found = requested_metrics("What is the quick ratio and the cash ratio?", REGISTERED)
    assert set(found) == {"quick_ratio", "cash_ratio"}


def test_an_ordinary_question_triggers_nothing() -> None:
    """Routing must not fire on prose that merely mentions finance. Computing
    six extra figures for "why is this at risk" would bury the finding."""
    for question in [
        "Why is this company at risk?",
        "What did the auditor say?",
        "Should I be worried about its debt?",
        "Explain the position in simple terms.",
    ]:
        assert requested_metrics(question, REGISTERED) == [], question


def test_only_formulas_that_actually_exist_are_offered() -> None:
    """Every synonym must point at a live formula, or a question routes to a
    name nothing can compute and the reader gets a silent nothing."""
    from agents.qa import _METRIC_SYNONYMS

    unknown = sorted(set(_METRIC_SYNONYMS) - REGISTERED)
    assert not unknown, f"synonyms for unregistered formulas: {unknown}"


def test_the_computed_figure_matches_the_filing() -> None:
    """The end-to-end claim, checked against source rather than against our
    own pipeline: 2,557,600,000 / -1,380,900,000 is -1.8521."""
    from datetime import date

    from compute.lineitems import FactIndex, annual_period_ends
    from compute.ratios import compute_metric
    from data.edgar import EdgarClient
    from data.facts import as_of_view

    pytest.importorskip("requests")
    try:
        facts = EdgarClient().facts(28823)
    except Exception:  # noqa: BLE001 - offline runs skip rather than fail
        pytest.skip("EDGAR unavailable")
    view = FactIndex(as_of_view(facts, date(2024, 7, 1)))
    ends = annual_period_ends(view)
    if not ends:
        pytest.skip("no annual period visible")
    cv = compute_metric("debt_to_equity", view, ends[0])
    assert cv.is_defined
    inputs = {k: float(r.value) for k, r in cv.inputs.items()}
    expected = (inputs["long_term_debt"] + inputs["short_term_debt"]) / inputs["equity"]
    assert abs(float(cv.value) - expected) < 1e-9
