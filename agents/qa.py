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

#: Two failure modes shaped this prompt, both found by using it.
#:
#: Instructions alone did not stop bullet points. Three prompts forbade them
#: and the model produced them every time; showing a bad answer beside a good
#: one stopped it immediately. Models imitate examples far more reliably than
#: they obey rules.
#:
#: But a single example produced a single answer. "Why is this at risk", "what
#: is the biggest problem" and "should I lend" all returned the same paragraph
#: -- the model had learned one shape and recited it regardless of what was
#: asked. Several examples, answering *different* questions from the *same*
#: assessment, are what make the reply track the question.
SYSTEM_PROMPT = """\
You are a credit analyst explaining a finished assessment to a company \
executive who does not read financial statements. They understand cash, bills, \
lenders and trouble. They do not know what a coverage ratio is.

ANSWER THE QUESTION ASKED. Do not summarise the whole assessment every time. \
Mention only the findings that bear on this particular question. If the \
question is narrow, the answer is narrow.

Plain prose, two to four sentences. No bullet points, no lists, no metric \
names in bold, no ratio jargon. At most one or two figures, rounded to two \
decimals, and only where a figure earns its place.

Worked examples, all from the same assessment. Note how little they overlap.

Q: Why is this company at risk?
A: This business is not earning enough to cover the interest on its
   borrowings, and it is burning cash rather than generating it, so the debt
   is being serviced from something other than trading. On the standard
   measure used to flag companies heading for insolvency it sits well inside
   the danger zone. Taken together this is a company that cannot fund itself
   from its own operations.

Q: What is the single biggest problem here?
A: That the company cannot service its debt out of what it earns. Everything
   else follows from it -- the cash drain, the thin cover for near-term bills,
   the distress score. A business can survive weak margins for a while; it
   cannot survive not covering its interest.

Q: How urgent is this?
A: The assessment does not put a timeline on it. What it does show is that the
   pressure is on near-term obligations rather than distant ones, and that the
   figures come from filings that are now well over a year old, so the current
   position could be materially different in either direction.

Q: What does the auditor say?
A: The assessment does not record an auditor opinion for this filer. It rests
   on figures taken from the filed statements rather than on anything the
   auditor wrote.

RULES
1. Only facts from the assessment. Never compute, sum, average or convert. If \
a figure is not there, say it is not in the assessment.
2. Never recommend an action -- no lending, investing, trading or pricing \
advice. You explain the situation; the reader decides.
3. If the assessment does not answer the question, say so in one sentence and \
then say what it does show. Do not pad with unrelated findings.
4. Never open with the signal name or the confidence number.
"""

#: The reader asking for structure. Prose is the default because the model
#: dumped metric lists when nobody asked, but banning lists outright was an
#: over-correction: someone who asks for point-wise wants points, and giving
#: them prose is the same failure in the other direction. Format follows the
#: request.
_WANTS_POINTS = re.compile(
    r"\bpoint[\s-]?wise\b|\bbullet|\bin points\b|\blist (?:them|out|the)\b"
    r"|\bitemi[sz]e|\bbreak (?:it|this) down\b|\bstep[\s-]?by[\s-]?step\b"
    r"|\bone by one\b|\benumerate\b",
    re.I,
)

#: A request to see the shape of something over time, not to read about it.
_WANTS_CHART = re.compile(
    r"\b(?:chart|graph|plot|visuali[sz]e|show me the trend|trend line)\b"
    r"|\bover time\b.{0,20}\b(?:show|see|view)\b"
    r"|\b(?:show|draw|display)\b.{0,24}\b(?:trend|history|over time)\b",
    re.I,
)

#: This prompt was rewritten three times. Rules did not work: the model read
#: them as a checklist, verified each in turn, and filled its whole budget with
#: deliberation -- twelve thousand characters of self-checking, the tag closed,
#: and no answer after it. What works is the same shape as the prose prompt: a
#: worked question-and-answer pair, and almost no rules.
POINTS_PROMPT = """\
You are a credit analyst writing for a company executive who does not read \
financial statements.

Q: Give me the risk parameters point wise where it is going wrong.
A: This business cannot fund itself from what it earns, and several measures
   say so at once.

   It is deep inside the danger zone for insolvency (-1.93)
   Its obligations are larger than everything it owns (1.45)
   It is not earning enough to pay the interest on its debt (-1.06)
   It is losing money on what it sells (-0.17)
   It is burning cash rather than generating it (-0.15)

   Together these describe a company servicing its debt from something other
   than trading.

Answer in exactly that shape, using the assessment below. Write the answer \
straight out with no working. Use only figures already in the assessment, \
rounded to two decimals, and never recommend an action.
"""

#: Metric names a reader might ask to see plotted, mapped to the series the
#: trend endpoint serves.
CHARTABLE = {
    "current ratio": "current_ratio",
    "quick ratio": "quick_ratio",
    "liabilities": "liabilities_to_assets",
    "leverage": "liabilities_to_assets",
    "interest coverage": "interest_coverage",
    "coverage": "interest_coverage",
    "margin": "net_margin",
    "net margin": "net_margin",
    "return on assets": "return_on_assets",
    "cash flow": "ocf_to_debt",
    "altman": "altman_z_double_prime",
    "distress score": "altman_z_double_prime",
}


def wants_points(question: str) -> bool:
    """Whether the reader asked for a structured answer rather than prose."""
    return bool(_WANTS_POINTS.search(question))


def wants_chart(question: str) -> str | None:
    """The metric the reader asked to see plotted, if any."""
    if not _WANTS_CHART.search(question):
        return None
    low = question.lower()
    for phrase, metric in sorted(CHARTABLE.items(), key=lambda kv: -len(kv[0])):
        if phrase in low:
            return metric
    # A chart was asked for without naming a metric; the caller picks a default
    # rather than guessing at one here.
    return "current_ratio"


#: Questions that ask what to *do* rather than what is *true*.
#:
#: Blocking these after the fact returns nothing useful. SPEC 8 says refuse
#: *or redirect*, and only the refusal was built: a reader who asks "should I
#: lend to this" got a guard message and no evidence, which is a safe answer
#: and a bad one. Detected on the way in, the turn is steered instead --
#: decline the decision, give the evidence bearing on it, and say what the
#: system cannot see. That last part is the most useful thing in the reply and
#: is still not advice.
_ADVICE_SOUGHT = re.compile(
    r"\bshould (?:i|we|they|he|she|you)\b"
    r"|\b(?:do|would) you recommend\b"
    r"|\bwhat (?:should|would) (?:i|we|you) do\b"
    r"|\b(?:is|are) (?:it|this|they) a (?:good|bad|safe|smart) (?:idea|investment|bet|buy)\b"
    r"|\b(?:buy|sell|short|invest in|lend to|avoid|exit|hold)\b[^?]{0,25}\?"
    r"|\bworth (?:buying|investing|lending|the risk)\b"
    r"|\bhow much should\b"
    r"|\btell me what to do\b"
    r"|\bwhat would you do\b",
    re.I,
)

#: Instruction-shaped text in the *question*. ``data/sanitize.py`` guards
#: filing text, which is attacker-controlled; the question is too, and it
#: reaches the model by a different path that guard never covered.
#:
#: The output scope guard is a backstop, not a substitute. It catches
#: "deny the loan" and misses "your position would be better served by" --
#: same advice, no banned phrase. Detecting the attempt on the way in also
#: lets the reply say what happened rather than silently complying.
_QUESTION_INJECTION = re.compile(
    r"ignore (?:all |any |the )?(?:previous|prior|above|earlier|your) "
    r"(?:instruction|prompt|rule|guideline)"
    r"|disregard (?:all |any |the )?(?:previous|prior|above|your)"
    r"|you are (?:now |no longer )"
    r"|new instructions?:"
    r"|(?:^|\n)\s*(?:system|assistant|developer)\s*:"
    r"|</?(?:system|assistant|instruction)\s*>"
    r"|pretend (?:you are|to be)"
    r"|(?:forget|drop|bypass|override) (?:your |the )?(?:rules|guardrails|restrictions|scope)",
    re.I,
)

REDIRECT_DIRECTIVE = """
The reader has asked what to DO. Do not answer that, and do not lecture \
them about it. Structure the reply exactly so:

First sentence: say plainly that you set out evidence rather than decisions.
Then: the two or three findings that bear most on their question, in plain \
language.
Last: what this assessment CANNOT tell them that matters for the decision \
-- their exposure, security or recovery position, pricing, the company's \
private financing, or anything after the prediction date. Be specific \
about the gaps.

Still no bullet points, still prose, still no recommendation.
"""

INJECTION_NOTICE = (
    "That question contained text shaped like an instruction to the "
    "assistant rather than a question about the company. It was not acted "
    "on. Ask about the assessment and it will be answered from the "
    "verified findings."
)


def question_is_injection(question: str) -> bool:
    """Whether the question is trying to reprogram the assistant."""
    return bool(_QUESTION_INJECTION.search(question))


def asks_for_advice(question: str) -> bool:
    """Whether the question wants a decision rather than a fact."""
    return bool(_ADVICE_SOUGHT.search(question))


#: What each metric means, in the terms a credit analyst would use.
#:
#: Without this the model has nothing to explain *with*: a context of bare
#: names and numbers can only be read back as a list, which is exactly what it
#: did. Asking for insight while supplying none is not a prompt problem.
GLOSS: dict[str, str] = {
    "interest_coverage": "operating income against interest owed; below 1 means "
        "the company is not earning enough to pay the interest on its debt",
    "ocf_to_debt": "cash generated by operations against total debt; negative means "
        "the business is consuming cash rather than producing it to service borrowings",
    "altman_z_double_prime": "the Altman distress score; below roughly 1.1 is the "
        "distress zone, and negative is deep inside it",
    "current_ratio": "short-term assets against short-term obligations; near or "
        "below 1 means little cushion to meet the next year's bills",
    "quick_ratio": "the same but excluding inventory, so the cash-like cushion",
    "liabilities_to_assets": "share of the balance sheet funded by obligations "
        "rather than equity; above 0.7 is heavily levered",
    "debt_to_assets": "borrowings as a share of assets",
    "net_margin": "profit per unit of revenue; negative means losses",
    "operating_margin": "profit from operations before financing",
    "return_on_assets": "profit generated per unit of assets employed",
    "accruals_to_assets": "reported profit not yet backed by cash",
    "ohlson_o_score": "a fitted bankruptcy-probability score; higher is worse",
    "piotroski_f_score": "nine fundamental health checks; low means few are passing",
    "cash_ratio": "cash alone against short-term obligations",
    "days_sales_outstanding": "how long revenue sits uncollected",
    "cash_conversion_cycle": "days between paying suppliers and collecting from customers",
}

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


def _dedent(text: str) -> str:
    """Drop the leading indentation the worked example carries into replies."""
    return "\n".join(ln.strip() for ln in text.splitlines()).strip()


def strip_reasoning(text: str) -> str:
    """Remove inline scratchpad blocks and leading bullet-form deliberation."""
    had_block = bool(_THOUGHT.search(text)) or bool(_THOUGHT_OPEN.search(text))
    cleaned = _THOUGHT.sub("", text).strip()
    if had_block and not cleaned:
        # A scratchpad and nothing else: the model ran out of room before
        # writing an answer. Falling back to the original text here is what
        # published a sixty-line reasoning trace to a reader.
        return ""
    cleaned = _dedent(cleaned)
    if _THOUGHT_OPEN.search(cleaned):
        # An opener with no closer means the model ran out of room while still
        # deliberating. Everything after the tag is scratchpad, so keeping it
        # publishes the deliberation as the answer -- which is how a 60-line
        # trace reached a reader. Nothing usable was produced.
        return ""
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
        lines.append(f"risk_score: {float(memo['risk_score']):.0f} out of 100")
    if memo.get("summary"):
        lines.append(f"Summary: {memo['summary']}")
    for section in memo.get("sections", []):
        lines.append(f"\n[{section['title']}] ({section['tier']})")
        if section.get("body"):
            lines.append(section["body"])
        for e in section.get("evidence", []):
            period = f" at period end {e['period_end']}" if e.get("period_end") else ""
            # Rounded here, not in the prompt. A model shown
            # -0.43269186181312314 will quote it back, and the grounding check
            # then *rewards* the long form because any rounding risks looking
            # like a number that was not in the source.
            value = e.get("value")
            shown = f"{float(value):,.2f}" if isinstance(value, (int, float)) else value
            gloss = GLOSS.get(e["metric"], "")
            meaning = f" -- {gloss}" if gloss else ""
            lines.append(
                f"  - {e['metric']} = {shown}{period} ({e.get('note', '')}){meaning}"
            )
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
    # Refused before the model sees it. Letting an injection through and
    # relying on the output guard covers only the phrasings that guard knows,
    # and leaves the reader with a blocked answer and no idea why. The model
    # happens to resist these prompts today, which is robustness rather than a
    # control -- a control does not depend on the model's mood.
    if question_is_injection(question):
        return Answer(text=INJECTION_NOTICE, allowed=True, reason="question_injection")

    context = memo_context(payload)
    # Chosen, not layered. An earlier version appended a "give points"
    # directive to a prompt whose rules forbade lists, and the model spent its
    # entire budget litigating the contradiction -- the reply was a 60-line
    # reasoning trace that never reached an answer.
    if wants_points(question):
        system = POINTS_PROMPT
    elif asks_for_advice(question):
        # Redirect rather than refuse (SPEC 8). Only the refusal half was
        # built, so a reader asking "should I lend to this" got a guard
        # message and no evidence -- safe, and useless.
        system = SYSTEM_PROMPT + REDIRECT_DIRECTIVE
    else:
        system = SYSTEM_PROMPT
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"ASSESSMENT\n{context}\n\nQUESTION\n{question}"},
    ]
    # This model emits its scratchpad inline and those tokens count against
    # the budget. The default 1600 is comfortable for a prose answer and not
    # for a point-wise one, where ordering seven findings produced a long
    # deliberation that consumed the whole allowance before the answer began.
    # The same failure is documented for preflight in agents/llm.py.
    reply = strip_reasoning(client.complete(messages, max_tokens=4000))
    if not reply:
        # The model produced only a scratchpad. One retry, told plainly not to
        # deliberate -- the failure is over-thinking rather than difficulty,
        # and a second pass with that said usually lands.
        retry = list(messages)
        retry.append({
            "role": "user",
            "content": "Write the answer only. No working, no checking, no preamble.",
        })
        reply = strip_reasoning(client.complete(retry, max_tokens=4000))
    if not reply:
        return Answer(
            text=(
                "The model did not finish composing an answer to that. Try asking "
                "it more narrowly -- one finding at a time rather than all of them."
            ),
            allowed=True,
            reason="truncated",
        )
    return check_answer(reply, context)
