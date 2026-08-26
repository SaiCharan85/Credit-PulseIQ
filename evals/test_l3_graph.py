"""L3: the pipeline as a graph, with the measured loop left alone.

LangGraph orchestrates the stages around the investigator -- load, triage,
investigate, rank, guard, assemble -- and declares the branch at the guard gate
that used to be an early return.

What it does *not* do is re-express the ReAct loop as nodes. The step budget,
the forced abstention and the bounded critic retry are the behaviours the
0.963 backtest measured across 200 cases; rebuilding them inside a graph would
be a rewrite of the thing under measurement, and the number would stop
describing what ships. That boundary is asserted here so a later refactor
cannot quietly cross it.
"""

from __future__ import annotations

from datetime import date

import pytest

from agents.graph import build_graph
from agents.rulebased import RuleBasedInvestigator

pytest.importorskip("langgraph")
AS_OF = date(2024, 6, 1)


def _facts():
    from evals.test_l4_memo import STRESSED

    return STRESSED


class _Edgar:
    """Serves fixture facts, so the graph is tested without a network."""

    def __init__(self, facts):
        self._facts = facts

    def facts(self, cik: int):
        return self._facts


def test_the_pipeline_runs_end_to_end() -> None:
    graph = build_graph(RuleBasedInvestigator(), edgar=_Edgar(_facts()))
    out = graph.invoke({"cik": 1, "as_of": AS_OF})
    assert [s["node"] for s in out["steps"]] == [
        "load_facts", "triage", "investigate", "rank", "guard", "assemble",
    ]
    assert out["memo"] is not None
    assert not out.get("blocked_reason")


def test_a_filer_with_no_facts_stops_at_the_first_node() -> None:
    """The error edge exists so a missing filer is an outcome, not an exception
    thrown three stages later."""
    graph = build_graph(RuleBasedInvestigator(), edgar=_Edgar([]))
    out = graph.invoke({"cik": 1, "as_of": AS_OF})
    assert out.get("error")
    assert "memo" not in out or out.get("memo") is None
    assert [s["node"] for s in out.get("steps", [])] == []


def test_the_ranker_never_gates_the_memo() -> None:
    """The statistical view sits beside the narrative one. A missing ranker
    costs a column, never the assessment."""
    import models.ranker as ranker

    original = ranker.rank
    ranker.rank = lambda *a, **k: None
    try:
        graph = build_graph(RuleBasedInvestigator(), edgar=_Edgar(_facts()))
        out = graph.invoke({"cik": 1, "as_of": AS_OF})
        assert out["memo"] is not None
        assert out.get("ranking") is None
    finally:
        ranker.rank = original


def test_the_react_loop_is_a_single_node() -> None:
    """The boundary that keeps the backtest meaningful.

    If a refactor splits the investigator across nodes, the 0.963 figure stops
    describing the shipped agent and this fails."""
    import inspect

    import agents.graph as g

    source = inspect.getsource(g.build_graph)
    for owned_by_the_loop in ("max_steps", "investigator.run"):
        assert owned_by_the_loop in source
    # ...and the graph must not be re-implementing tool dispatch itself.
    for not_the_graphs_job in ("_dispatch", "tool_schemas", "check_threshold"):
        assert not_the_graphs_job not in source, (
            f"{not_the_graphs_job!r} belongs to the investigator, not the graph"
        )


def test_streaming_emits_one_update_per_node() -> None:
    """What the SSE endpoint was assembling by hand from a callback."""
    graph = build_graph(RuleBasedInvestigator(), edgar=_Edgar(_facts()))
    seen = [list(chunk)[0] for chunk in graph.stream({"cik": 1, "as_of": AS_OF})]
    assert seen == ["load_facts", "triage", "investigate", "rank", "guard", "assemble"]


def test_checkpointing_makes_a_run_resumable() -> None:
    from langgraph.checkpoint.memory import MemorySaver

    saver = MemorySaver()
    graph = build_graph(RuleBasedInvestigator(), edgar=_Edgar(_facts()),
                        checkpointer=saver)
    config = {"configurable": {"thread_id": "t1"}}
    out = graph.invoke({"cik": 1, "as_of": AS_OF}, config)
    assert out["memo"] is not None
    # The state survives the call, which is what resuming rests on.
    assert graph.get_state(config).values.get("memo") is not None
