"""Intent by shape: does the answer look like the question deserved?

Every other eval here asks whether an answer is *true*. None asked whether it
was *usable*, and that gap is why six formatting defects in a row were found by
a reader rather than by the harness. A one-line answer to "break this down by
subtopic" passes the grounding check, the citation check and the figure audit,
and is still the wrong answer.

So this declares the mapping the system is supposed to know, and tests it:

    intent          what the reader is doing        the shape that serves it
    -----------     ---------------------------     ------------------------
    lookup          wants one number                one or two sentences
    explain         wants the causal chain          prose, several sentences
    degree          "is that bad?"                  verdict plus the comparator
    trend           wants direction over time       dated points, a direction
    breakdown       several subjects at once        headings, mixed shapes
    list            wants to scan                   framing plus real bullets
    table           wants to compare on columns     a markdown table
    rank            where it sits                   placement plus the caveat
    history         what happened, and when         dated events
    limits          what is missing                 the gap, named
    definition      what a term means               concept, no company claim
    compare_company two filers side by side         both named, both assessed
    compound        several requests in one         a section per request

The checks are deliberately structural. Whether the figures are right is
answered by ``run_figure_audit``; whether they are grounded is answered by
``run_response_eval``. This asks the third question, which is whether a person
reading the reply gets what they asked for.

    python -m evals.run_answer_quality
    python -m evals.run_answer_quality breakdown list
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date

SUBJECT_CIK = 28823
AS_OF = date(2024, 7, 1)

_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+", re.M)
_HEADING = re.compile(r"^#{2,4}\s+\S|^\*\*[^*]+\*\*\s*$", re.M)
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$", re.M)
_DATED = re.compile(r"\b(?:19|20)\d{2}\b")


def _words(t: str) -> int:
    return len(t.split())


def _sentences(t: str) -> int:
    return len([s for s in re.split(r"(?<=[.!?])\s+", t.strip()) if s])


# ---- shape predicates -------------------------------------------------------
# Each returns "" when the shape is right, or what is wrong with it.

def brief(max_sentences: int = 2) -> Callable[[str], str]:
    return lambda t: (
        "" if _sentences(t) <= max_sentences
        else f"a lookup answered in {_sentences(t)} sentences"
    )


def prose(min_words: int = 30) -> Callable[[str], str]:
    def check(t: str) -> str:
        if _words(t) < min_words:
            return f"an explanation in {_words(t)} words"
        lines = [ln for ln in t.splitlines() if ln.strip()]
        if lines and len(_BULLET.findall(t)) == len(lines):
            return "a causal chain delivered as disconnected bullets"
        return ""

    return check


def bulleted(min_points: int = 3) -> Callable[[str], str]:
    def check(t: str) -> str:
        marked = len(_BULLET.findall(t))
        if marked >= min_points:
            return ""
        lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
        return (
            "" if len(lines) >= min_points + 1
            else f"asked to scan, given {marked} marked points in {len(lines)} lines"
        )

    return check


def sectioned(min_headings: int = 2) -> Callable[[str], str]:
    def check(t: str) -> str:
        heads = len(_HEADING.findall(t))
        if heads < min_headings:
            return f"several subjects run together under {heads} headings"
        if not _BULLET.search(t) and _words(t) > 200:
            return "long sectioned answer with nothing scannable in it"
        return ""

    return check


def tabular(t: str) -> str:
    rows = _TABLE_ROW.findall(t)
    if len(rows) < 3:
        return f"asked for a table, got {len(rows)} rows"
    widths = {r.count("|") for r in rows}
    return "" if len(widths) == 1 else "table rows disagree about their columns"


def dated(min_years: int = 2) -> Callable[[str], str]:
    return lambda t: (
        "" if len(set(_DATED.findall(t))) >= min_years
        else f"movement described across {len(set(_DATED.findall(t)))} dated points"
    )


def mentions(*words: str) -> Callable[[str], str]:
    def check(t: str) -> str:
        low = t.lower()
        return "" if any(w in low for w in words) else f"never mentions {words[0]!r}"

    return check


def all_of(*checks: Callable[[str], str]) -> Callable[[str], str]:
    def check(t: str) -> str:
        return next((msg for msg in (c(t) for c in checks) if msg), "")

    return check


@dataclass
class Case:
    intent: str
    question: str
    shape: Callable[[str], str]
    note: str = ""


CASES: list[Case] = [
    Case("lookup", "What is the quick ratio?", brief(2),
         "a figure request answered with an essay is a worse answer"),
    Case("lookup", "Is coverage below one?", brief(2)),

    Case("explain", "Why is this company at risk?", prose(40),
         "the connections between findings are the content"),
    Case("explain", "What is the single biggest problem here?", prose(30)),

    Case("degree", "How leveraged is it, slightly or heavily?",
         all_of(mentions("heavil", "severe", "high", "well above"), brief(4)),
         "a degree question wants a verdict, not the raw number again"),

    Case("trend", "How has the current ratio moved over the years?", dated(3),
         "a direction with no dated points behind it is an assertion"),

    Case("breakdown", "Break this down by subtopic with a brief understanding of each.",
         sectioned(2), "the request a flat list cannot serve"),
    Case("breakdown", "Group the problems by area and explain each.", sectioned(2)),

    Case("list", "Give me the main risks as points.", bulleted(3)),
    Case("list", "List the key figures as a numbered list.", bulleted(3)),

    Case("table", "Show me the key measures in a table.", tabular),

    Case("rank", "At what percentile does this rank?",
         all_of(mentions("percentile", "top", "bottom", "above"),
                mentions("not a probability", "ordering", "rank")),
         "a placement without the caveat invites it being read as a chance"),

    Case("history", "If it looks healthy now, was it ever in trouble?",
         dated(1), "an outcome question wants the date it happened"),

    Case("limits", "What could you not see in these filings?",
         mentions("not report", "cannot", "could not", "old", "year", "before",
                  "latest", "missing", "not reflect")),

    Case("definition", "What does a going concern warning mean?",
         all_of(mentions("auditor", "doubt", "survive", "continue", "twelve"),
                prose(25)),
         "a definitional aside mid-conversation should just be answered"),

    Case("compare_company", "Compare this to Valaris.",
         all_of(mentions("valaris"), mentions("diebold", "28823")),
         "both filers named, both assessed -- never compared from memory"),

    Case("compound",
         "At what percentile does it rank, and also compare it to Valaris, "
         "and explain the main problems with subtopics and bullets under each.",
         all_of(sectioned(2), mentions("percentile", "top", "above"),
                mentions("valaris")),
         "people do not ask one thing at a time"),

    Case("compound",
         "I'm doing a credit review. First what is driving the risk, second how "
         "bad is the leverage against normal levels, and third has it been "
         "getting worse? Organised with headings please.",
         all_of(sectioned(2), bulleted(2))),
]


def main(argv: list[str]) -> int:
    from agents.llm import load_env_file

    load_env_file()
    import serve

    wanted = set(argv[1:])
    cases = [c for c in CASES if not wanted or c.intent in wanted]

    payload, status = serve._assess(
        serve.AssessRequest(cik=SUBJECT_CIK, as_of=AS_OF, agent="rules")
    )
    if status != 200 or not payload.get("memo"):
        print("could not build a subject assessment", file=sys.stderr)
        return 1

    results = []
    for i, case in enumerate(cases, 1):
        print(f"  [{i:>2}/{len(cases)}] {case.intent:<16}{case.question[:52]}", flush=True)
        body = json.loads(
            serve.ask_question(
                serve.AskRequest(assessment=payload, question=case.question,
                                 mode="company")
            ).body
        )
        text = str(body.get("answer") or "")
        problem = "answer withheld" if not text else case.shape(text)
        results.append((case, text, problem))

    print("\n" + "=" * 78)
    print("ANSWER QUALITY -- does the shape match the intent?")
    print("=" * 78)
    by_intent: dict[str, list] = {}
    for case, text, problem in results:
        by_intent.setdefault(case.intent, []).append((case, text, problem))

    for intent in sorted(by_intent):
        rows = by_intent[intent]
        ok = sum(1 for _, _, p in rows if not p)
        print(f"\n{intent.upper():<18} {ok}/{len(rows)}")
        for case, text, problem in rows:
            mark = "pass" if not problem else "FAIL"
            print(f"  [{mark}] {_words(text):>4}w  {case.question[:50]}")
            if problem:
                print(f"         -> {problem}")
                if case.note:
                    print(f"            ({case.note})")

    passed = sum(1 for _, _, p in results if not p)
    print(f"\nTOTAL {passed}/{len(results)} answers took the shape the question asked for")
    print(
        "\n  Structural only. Whether the figures are right is run_figure_audit;\n"
        "  whether they are grounded is run_response_eval. This is the third\n"
        "  question -- whether a person reading the reply got what they asked for."
    )
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
