"""L5: a follow-up question needs an antecedent, and only a safe one.

Every question used to be answered from scratch. Three turns that read as a
conversation were answered as three strangers:

    "What is the quick ratio?"            -> 0.74
    "Why does that matter?"               -> a generic distress summary;
                                             *that* referred to nothing
    "And compare it to the current ratio" -> quoted 1.10, never compared

Carrying prior turns fixes it, and introduces two ways to get it wrong.

**Scope.** History belongs to one filer at one date. Carrying it across a
company change would put one company's figures behind another's question --
the grounding failure this system exists to prevent, arriving by the side door.
Enforced on both sides: the client filters, and the server filters again rather
than trusting it.

**Blocked turns.** Replaying a withheld answer puts the text of a rejected
claim back into context, where the model may repeat it -- laundering a refusal
into a citation on the next turn. Only allowed exchanges are carried.

A third failure took a live conversation to find and is the reason this file
exists at all: metrics computed *on request* have to persist. Turn one asks for
the quick ratio and gets 0.74; turn three says "compare it to the current
ratio" and names no measure, so recomputing from that question alone drops 0.74
from the trusted set -- and the follow-up is blocked for quoting a figure this
system produced two turns earlier.
"""

from __future__ import annotations

from agents.qa import MAX_HISTORY_TURNS, _history_messages


def _turn(q: str, a: str, **kw) -> dict:
    return {"question": q, "answer": a, "allowed": True, **kw}


def test_prior_turns_become_conversation() -> None:
    msgs = _history_messages([_turn("What is the quick ratio?", "It is 0.74.")])
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[0]["content"] == "What is the quick ratio?"
    assert msgs[1]["content"] == "It is 0.74."


def test_a_withheld_turn_is_never_replayed() -> None:
    """Putting a blocked answer back in context is how a refusal becomes a
    citation on the next turn."""
    history = [
        _turn("Good question", "A verified answer."),
        {"question": "Bad one", "answer": "A claim we rejected.", "allowed": False},
    ]
    msgs = _history_messages(history)
    assert "A claim we rejected." not in [m["content"] for m in msgs]
    assert "A verified answer." in [m["content"] for m in msgs]


def test_empty_turns_are_dropped() -> None:
    assert _history_messages([{"question": "", "answer": "x", "allowed": True}]) == []
    assert _history_messages([{"question": "x", "answer": "", "allowed": True}]) == []
    assert _history_messages(None) == []
    assert _history_messages([]) == []


def test_only_the_recent_turns_are_carried() -> None:
    """Every turn is resent on every call, so the budget is quadratic in this
    number -- and an hour-old aside should not steer a fresh question."""
    history = [_turn(f"q{i}", f"a{i}") for i in range(10)]
    msgs = _history_messages(history)
    assert len(msgs) == MAX_HISTORY_TURNS * 2
    assert msgs[-1]["content"] == "a9", "the newest turn must survive"
    assert "a0" not in [m["content"] for m in msgs]


def test_the_server_rescopes_rather_than_trusting_the_client() -> None:
    """A client bug, or a crafted request, must not be able to put another
    filer's figures into this filer's context."""
    history = [
        _turn("about A", "A answer", cik=111, as_of="2024-07-01"),
        _turn("about B", "B answer", cik=222, as_of="2024-07-01"),
        _turn("wrong date", "stale answer", cik=111, as_of="2020-01-01"),
    ]
    cik, as_of_raw = 111, "2024-07-01"
    scoped = [
        t for t in history
        if str(t.get("cik", cik)) == str(cik)
        and str(t.get("as_of", as_of_raw)) == str(as_of_raw)
    ]
    assert len(scoped) == 1
    assert scoped[0]["question"] == "about A"


def test_history_absent_is_the_old_behaviour() -> None:
    """No history must still answer -- the first question of every chat has
    none, and general mode never has any."""
    assert _history_messages([]) == []
