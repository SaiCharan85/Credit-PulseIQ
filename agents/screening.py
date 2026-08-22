"""Screening questions: ranked evidence, never a forecast.

"Give me the best companies to survive this year" is two requests wearing one
sentence. The screening part -- rank filers by measured distress evidence -- is
answerable and useful. The forecast part -- who will survive -- is not
answerable by this system and is the part that creates liability.

Separating them is the whole job here:

**A ranking is a dated measurement, not a prediction.** "As of 2024-07-01, on
filings public at that date, these ten scored lowest" is a statement about the
past that can be checked. "These ten will survive" is a claim about the future
that cannot. The tense is the control, so every ranked answer is emitted in the
past tense with its as-of date attached.

**The disclaimer is attached, not available on request.** A reader who does not
think to ask whether a ranking is a forecast is exactly the reader who will
assume it is. It rides with the output.

**Our own error rate goes out with the list.** The 200-case backtest missed 8%
of companies that subsequently failed, at high confidence. Publishing a
"lowest risk" list without that number is the version that misleads;
publishing it with the number is a measured statement about measured evidence.

What is still refused: choosing from the list. Ranking evidence is
information; picking names out of it for a particular reader is selection
advice, and the redirect fires there exactly as it does for a single company.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

#: Questions asking across companies rather than about one.
_SCREENING = re.compile(
    # Plural, or an explicit multi-filer word. "What is the current ratio of
    # this company" contains "what ... company" and is emphatically not a
    # screen; requiring plurality is what separates the two.
    r"\b(?:list|rank|ranking|screen|compare|which|what) .{0,40}\b(?:companies|filers|names|issuers)\b"
    r"|\b(?:companies|filers|issuers)\b.{0,30}\b(?:list|rank|ranked|safest|riskiest|best|worst)\b"
    r"|\b(?:best|worst|safest|riskiest|top|bottom|least risky|most risky)\b"
    r".{0,30}\b(?:compan|filer|name|issuer|stock)"
    r"|\bgive me .{0,20}(?:names|list)\b"
    r"|\bacross (?:my |the )?(?:portfolio|holdings|names)\b",
    re.I,
)

#: Language that makes a question about the future.
_FORWARD = re.compile(
    r"\b(?:will|going to|gonna) (?:be|survive|fail|default|go under|make it)\b"
    r"|\bsurvive\b|\bnext (?:year|quarter|month|12 months)\b"
    r"|\bthis year\b|\bgoing forward\b|\bin (?:the )?future\b"
    r"|\bpredict|\bforecast|\boutlook\b|\bexpect(?:ed)? to\b"
    r"|\bwho (?:will|is going to)\b",
    re.I,
)

NOT_A_FORECAST = (
    "This system does not forecast. It reports what filings said at a chosen "
    "past date, so a ranking describes measured evidence as of that date and "
    "not what will happen next."
)

#: From the 200-case L3 backtest, arm 2. Printed with every ranking.
FALSE_CONFIDENCE_RATE = 0.080


def is_screening(question: str) -> bool:
    """Whether the question spans companies rather than asking about one."""
    return bool(_SCREENING.search(question))


def is_forward_looking(question: str) -> bool:
    """Whether the question asks about the future rather than the record."""
    return bool(_FORWARD.search(question))


@dataclass
class ScreenAnswer:
    """A ranked answer with the caveats that must travel with it."""

    text: str = ""
    rows: list[dict] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    refused_forecast: bool = False


def rank_rows(rows: list[dict], ascending: bool) -> list[dict]:
    """Order assessed filers by measured risk. Unreadable filers are dropped
    from the ranking but counted, because a list that silently omits the names
    it could not read overstates its own coverage."""
    from agents.schemas import SIGNAL_ORDER

    ok = [r for r in rows if r.get("status") == "ok"]

    def key(r: dict) -> tuple:
        sev = SIGNAL_ORDER.index(r["signal"]) if r.get("signal") in SIGNAL_ORDER else -1
        score = r.get("risk_score")
        return (sev, score if score is not None else -1.0, r.get("confidence", 0.0))

    return sorted(ok, key=key, reverse=not ascending)


def build(
    question: str,
    rows: list[dict],
    as_of: date,
    limit: int = 10,
) -> ScreenAnswer:
    """A ranked screening answer, in the past tense, with caveats attached."""
    ascending = bool(
        re.search(r"\b(?:best|safest|least risk|lowest|strongest|healthiest)\b", question, re.I)
    )
    ranked = rank_rows(rows, ascending)
    unreadable = sum(1 for r in rows if r.get("status") != "ok")
    answer = ScreenAnswer(rows=ranked[:limit])

    if not ranked:
        answer.text = "No filer in this set could be assessed, so there is nothing to rank."
        return answer

    direction = "lowest" if ascending else "highest"
    lines = [
        f"As of {as_of}, on filings public at that date, these are the "
        f"{direction}-scoring of the {len(ranked)} filers assessed."
    ]
    # The rule-based control emits no risk_score, so printing "no score" beside
    # every row reads as a fault rather than as the arm's design. The signal
    # already carries the ranking in that case; the score is shown only when
    # there is one.
    for r in answer.rows:
        score = f"  {r['risk_score']:.0f}/100" if r.get("risk_score") is not None else ""
        lines.append(f"  CIK {r['cik']}  {r['signal'].replace('_', ' ')}{score}")
    if all(r.get("risk_score") is None for r in answer.rows):
        lines.append(
            "  (ranked by signal; the deterministic control emits no 0-100 score)"
        )
    answer.text = "\n".join(lines)

    answer.caveats.append(NOT_A_FORECAST)
    answer.caveats.append(
        f"On the 200-case backtest this system called {FALSE_CONFIDENCE_RATE:.0%} of "
        "companies that subsequently failed low-risk at high confidence, so a low "
        "score is not a clean bill of health."
    )
    if unreadable:
        answer.caveats.append(
            f"{unreadable} filer(s) could not be read and are absent from the ranking "
            "rather than ranked as safe."
        )
    if is_forward_looking(question):
        answer.refused_forecast = True
        answer.caveats.insert(
            0,
            "The question asked about the future. That part is not answered: what "
            "follows is the measured record at the date above.",
        )
    return answer
