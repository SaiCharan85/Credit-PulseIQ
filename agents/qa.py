"""Grounded question answering over a finished assessment.

A chat box on top of a verified memo is the easiest way to give back
everything the rest of this system protects. Ask a model "what's the debt-to-
EBITDA?" and it will happily compute one; ask "should we lend?" and it will
answer. Either response looks exactly like the memo's verified prose and
carries none of its guarantees.

So this is not a chat over the filings. It is a **reader** for one assessment
that has already passed the guard gate, under three rules:

**No new numbers.** Every figure in an answer must already appear in the memo.
The check is mechanical -- numerals are extracted from the response and matched
against the memo's evidence and text, and an unmatched one blocks the answer.
The model cannot smuggle in arithmetic, because arithmetic produces numbers
that were not there before.

**No new evidence.** The context is the memo, not the filings. A question the
assessment does not cover is answered with "the assessment does not cover
this", which is a useful answer -- it tells a reader where the analysis stops.

**Same scope guard.** An answer that recommends an action is refused, exactly
as a memo would be. The guard does not weaken because the output is
conversational.

The first two make this narrow on purpose. A reader gets a faster route into
an assessment they could have read themselves; they do not get a second,
unverified analyst sitting behind the first one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from agents.schemas import scope_violations

SYSTEM_PROMPT = """\
You answer questions about ONE credit assessment that has already been produced \
and verified. The assessment is given below. It is your only source.

RULES
1. Never state a number that is not already in the assessment. Never compute, \
sum, average, or convert. If a question needs a figure that is not there, say \
it is not in the assessment.
2. Never recommend an action. No lending, investing, trading or pricing advice. \
You explain risk; the reader decides.
3. If the assessment does not answer the question, say so plainly. That is a \
useful answer, not a failure.
4. Be concise and concrete. Quote the assessment's own wording where it helps.
"""

#: Numerals that are structural rather than claims -- years, CIKs, list markers.
#: Excluded from the "no new numbers" check so ordinary prose is not blocked.
_SAFE = re.compile(r"^(19|20)\d\d$|^[0-9]$|^[0-9]{7,}$")
_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")

#: Several models emit their scratchpad inline. It must be removed before the
#: answer is shown *and* before it is checked -- a figure the model considered
#: and discarded while reasoning is not a claim, and treating it as one would
#: block honest answers for numbers the reader never sees.
_THOUGHT = re.compile(
    r"<thought>.*?</thought>"
    r"|<think>.*?</think>"
    r"|<reasoning>.*?</reasoning>"
    r"|<scratchpad>.*?</scratchpad>",
    re.I | re.S,
)
#: An unterminated opener swallows the rest of the reply; matched separately so
#: the closed-tag case above stays exact.
_THOUGHT_OPEN = re.compile(r"<(?:thought|think|reasoning|scratchpad)>", re.I)


def strip_reasoning(text: str) -> str:
    """Remove inline scratchpad blocks and leading bullet-form deliberation."""
    cleaned = _THOUGHT.sub("", text).strip()
    if _THOUGHT_OPEN.search(cleaned):
        # Opener with no closer: keep only what follows the last tag.
        cleaned = _THOUGHT_OPEN.split(cleaned)[-1].strip()
    return cleaned or text.strip()

REFUSAL_UNGROUNDED = (
    "That answer referenced a figure not present in the assessment, so it was "
    "withheld. Every number here has to trace to a verified memo entry."
)
REFUSAL_SCOPE = (
    "That answer recommended an action. This system assesses risk and leaves "
    "the decision to you."
)


@dataclass
class Answer:
    """A model reply plus the verdict of the checks it had to pass."""

    text: str = ""
    allowed: bool = True
    reason: str = ""
    ungrounded_numbers: list[str] = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        return not self.allowed


def _numerals(text: str) -> set[str]:
    out = set()
    for raw in _NUMBER.findall(text):
        token = raw.strip().rstrip(".").replace(",", "")
        if not token or _SAFE.match(token):
            continue
        out.add(token.lstrip("-"))
    return out


def memo_context(payload: dict[str, Any]) -> str:
    """Flatten an assessment into the only text the model may read."""
    memo = payload.get("memo") or {}
    lines = [
        f"CIK {payload.get('cik')}, prediction date {payload.get('as_of')}",
        f"SIGNAL: {memo.get('signal')}  confidence {memo.get('confidence')}",
    ]
    if memo.get("risk_score") is not None:
        lines.append(f"risk_score: {memo['risk_score']} out of 100")
    if memo.get("summary"):
        lines.append(f"Summary: {memo['summary']}")
    for section in memo.get("sections", []):
        lines.append(f"\n[{section['title']}] ({section['tier']})")
        if section.get("body"):
            lines.append(section["body"])
        for e in section.get("evidence", []):
            period = f" at period end {e['period_end']}" if e.get("period_end") else ""
            lines.append(f"  - {e['metric']} = {e['value']}{period} ({e.get('note', '')})")
    if memo.get("residual"):
        lines.append(f"\nResidual uncertainty: {memo['residual']}")
    for item in memo.get("limitations", []):
        lines.append(f"Data limitation: {item}")
    for step in memo.get("routing", []):
        lines.append(f"Routing: {step}")
    trail = memo.get("audit_trail", [])
    if trail:
        lines.append(
            "\nTools called: " + ", ".join(c.get("tool", "") for c in trail)
        )
    return "\n".join(lines)


def check_answer(text: str, context: str) -> Answer:
    """Enforce the two rules a model cannot be trusted to keep on its own."""
    answer = Answer(text=text)

    if scope_violations(text):
        answer.allowed = False
        answer.reason = REFUSAL_SCOPE
        return answer

    known = _numerals(context)
    # Percentages are a common restatement of a ratio already present, so a
    # figure is accepted when its digits appear in the context in any form.
    stray = sorted(n for n in _numerals(text) if not any(n in k or k in n for k in known))
    if stray:
        answer.allowed = False
        answer.reason = REFUSAL_UNGROUNDED
        answer.ungrounded_numbers = stray
    return answer


def ask(client: Any, payload: dict[str, Any], question: str) -> Answer:
    """Answer one question about one assessment, then check the answer."""
    if not (payload.get("memo")):
        return Answer(
            text="No memo was produced for this assessment, so there is nothing to explain.",
            allowed=True,
        )
    context = memo_context(payload)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"ASSESSMENT\n{context}\n\nQUESTION\n{question}"},
    ]
    reply = strip_reasoning(client.complete(messages))
    return check_answer(reply, context)
