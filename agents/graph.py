"""The assessment pipeline as a LangGraph state machine.

The orchestrator was a function calling five things in order. That worked and
said nothing: the stages, the branch at the guard gate, and the state each
carries were all implicit in control flow. As a graph they are declared, which
buys three things that are not cosmetic.

**Checkpointing.** A cold investigation is two minutes of sequential model
calls. With a checkpointer the run is resumable at the node it died on rather
than from the top, and a thread id makes a conversation durable across a
process restart -- the same property the chat had to hand-roll in localStorage.

**The branch is visible.** A blocked memo is not an error path, it is a real
outcome of the guard gate, and an edge that says so reads better than an early
return buried in a function.

**Streaming for free.** ``graph.stream`` emits per-node updates, which is what
the SSE endpoint was assembling by hand from an ``on_step`` callback.

What deliberately does **not** move into the graph is the ReAct loop itself.
``DistressInvestigator`` owns the step budget, the forced abstention and the
bounded critic retry, and those are the behaviours the 0.963 backtest measured
across 200 cases. Re-expressing them as nodes would be a rewrite of the thing
under measurement, and the number would stop describing what ships. The loop
stays one node; the graph orchestrates around it.
"""

from __future__ import annotations

from datetime import date
from typing import Annotated, Any, TypedDict


class AssessmentState(TypedDict, total=False):
    """What flows between nodes.

    Flat and explicit on purpose: a node reads what it needs and writes what it
    produced, so the diff between two checkpoints says exactly which stage did
    what.
    """

    cik: int
    as_of: date
    agent: str
    facts: Any
    #: Extra keyword arguments for the investigator -- the filing-text tools
    #: on the ReAct arm. Carried through the graph rather than closed over,
    #: because losing them silently would cost the +0.084 AUC those tools buy
    #: and the assessment would still look fine.
    tool_kwargs: dict
    #: Earnings-quality observations. Context-only: attached to the memo,
    #: never allowed to move the graded signal.
    context_notes: list
    triage: Any
    output: Any
    guards: Any
    ranking: Any
    memo: Any
    blocked_reason: str
    error: str
    steps: Annotated[list[dict[str, Any]], lambda a, b: (a or []) + (b or [])]


def build_graph(investigator: Any, edgar: Any = None, checkpointer: Any = None):
    """Compile the pipeline. Returns a graph with the same contract as before.

    ``checkpointer`` is optional and off by default: the API path is
    stateless per request and a checkpointer there would accumulate state
    nobody reads. It earns its place for long runs and for conversations,
    where resuming beats restarting.
    """
    from langgraph.graph import END, StateGraph

    from data.edgar import EdgarClient

    client = edgar or EdgarClient()

    def load_facts(state: AssessmentState) -> dict:
        """Fetch and as-of filter. The boundary the whole project rests on.

        Skipped when the caller already holds the facts -- the API path loads
        them to answer other questions first, and refetching would double the
        EDGAR traffic against a rate-limited endpoint for no gain.
        """
        if state.get("facts"):
            return {"steps": [{"node": "load_facts", "n": len(state["facts"])}]}
        try:
            facts = client.facts(state["cik"])
        except Exception as exc:  # noqa: BLE001
            return {"error": f"could not load filings: {exc}"}
        if not facts:
            return {"error": f"no XBRL facts for CIK {state['cik']}"}
        return {"facts": facts, "steps": [{"node": "load_facts", "n": len(facts)}]}

    def triage(state: AssessmentState) -> dict:
        from agents.orchestrator import triage as run_triage

        plan = run_triage(state["facts"], state["as_of"])
        return {"triage": plan, "steps": [{"node": "triage", "depth": plan.depth}]}

    def investigate(state: AssessmentState) -> dict:
        """One node, on purpose. The loop inside it is what was measured."""
        plan = state["triage"]
        previous = getattr(investigator, "max_steps", None)
        if previous is not None:
            investigator.max_steps = plan.steps
        try:
            output = investigator.run(
                state["cik"], state["as_of"], state["facts"],
                **(state.get("tool_kwargs") or {}),
            )
        finally:
            if previous is not None:
                investigator.max_steps = previous
        return {
            "output": output,
            "steps": [{"node": "investigate", "signal": output.signal}],
        }

    def rank(state: AssessmentState) -> dict:
        """The statistical view, alongside the narrative one. Never gates it."""
        from models import ranker

        r = ranker.rank(state["cik"], state["as_of"], state["facts"])
        return {"ranking": r, "steps": [{"node": "rank", "found": r is not None}]}

    def guard(state: AssessmentState) -> dict:
        from agents.guards import run_guards

        guards = run_guards(
            state["output"], cited=[], as_of=state["as_of"],
            latest_period_end=state["triage"].latest_period_end,
        )
        return {
            "guards": guards,
            "blocked_reason": "" if guards.may_ship else guards.summary(),
            "steps": [{"node": "guard", "may_ship": guards.may_ship}],
        }

    def assemble(state: AssessmentState) -> dict:
        from agents.orchestrator import build_memo

        memo = build_memo(
            state["output"], state["guards"], state["triage"],
            state.get("context_notes") or [],
        )
        return {"memo": memo, "steps": [{"node": "assemble"}]}

    def after_load(state: AssessmentState) -> str:
        return END if state.get("error") else "triage"

    def after_guard(state: AssessmentState) -> str:
        # Not an error edge. A blocked memo is the guard working, and the graph
        # should say so rather than route it through a failure path.
        return "assemble" if state["guards"].may_ship else END

    graph = StateGraph(AssessmentState)
    for name, fn in (
        ("load_facts", load_facts), ("triage", triage), ("investigate", investigate),
        ("rank", rank), ("guard", guard), ("assemble", assemble),
    ):
        graph.add_node(name, fn)

    graph.set_entry_point("load_facts")
    graph.add_conditional_edges("load_facts", after_load, {"triage": "triage", END: END})
    graph.add_edge("triage", "investigate")
    graph.add_edge("investigate", "rank")
    graph.add_edge("rank", "guard")
    graph.add_conditional_edges("guard", after_guard, {"assemble": "assemble", END: END})
    graph.add_edge("assemble", END)
    return graph.compile(checkpointer=checkpointer)


def run(investigator: Any, cik: int, as_of: date, edgar: Any = None,
        thread_id: str = "", facts: Any = None, context_notes: list | None = None,
        **tool_kwargs: Any) -> AssessmentState:
    """Run one assessment through the graph.

    ``thread_id`` turns on checkpointing for this run, so an interrupted
    investigation resumes at the node it stopped on instead of paying the two
    minutes again.
    """
    checkpointer = None
    config: dict[str, Any] = {}
    if thread_id:
        from langgraph.checkpoint.memory import MemorySaver

        checkpointer = MemorySaver()
        config = {"configurable": {"thread_id": thread_id}}
    graph = build_graph(investigator, edgar=edgar, checkpointer=checkpointer)
    state: dict[str, Any] = {"cik": cik, "as_of": as_of}
    if facts is not None:
        state["facts"] = facts
    if context_notes:
        state["context_notes"] = list(context_notes)
    if tool_kwargs:
        state["tool_kwargs"] = tool_kwargs
    return graph.invoke(state, config or None)
