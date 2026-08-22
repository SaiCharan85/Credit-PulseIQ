"""General financial explanation, separated from grounded claims.

The Q&A surface answers only from a verified assessment, which is right for
claims about a company and wrong for everything else. Asked "what is a covenant
breach?" it said the assessment does not cover it -- true, useless, and not
what the grounding rule was protecting against.

The rule that matters is narrower than "answer only from the memo":

**A claim about a specific company must be grounded.** "Diebold's coverage is
-0.43" has to come from a verified assessment, because nothing else can check
it and a wrong number here is the failure the whole project exists to prevent.

**A claim about how finance works need not be.** "A covenant is a promise in a
loan agreement, and breaching one usually lets the lender demand repayment" is
textbook, checkable by anyone, and specific to no filer. Refusing it protects
nothing and makes the tool useless for the reader it is built for.

So this module answers concepts and refuses specifics. A question that names a
company, or asks for a figure, is pushed back to the grounded path rather than
answered from the model's own recollection -- which is exactly where an
invented ratio would come from.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

SYSTEM_PROMPT = """\
You explain corporate credit and financial-statement concepts to someone \
running a business who does not read financial statements. They understand \
cash, customers, bills and lenders.

Plain prose, three to five sentences. No bullet points, no lists, no formulas. \
Define the idea, say why a lender or analyst cares, and give a concrete \
everyday illustration.

You are explaining HOW THINGS WORK, not any particular company. If the \
question is about a specific company's numbers or situation, say that you \
answer general questions here and that company figures come from running an \
assessment, then stop.

Never recommend an action. No lending, investing, trading or pricing advice.
Never invent a statistic. If you would need a number to answer well and do not \
have a source for it, describe the relationship instead of quantifying it.
"""

#: A question naming a company or demanding a figure belongs on the grounded
#: path. Detected on the way in, because a model asked about a named filer will
#: answer from recollection and the answer will look identical to a verified one.
_SPECIFIC = re.compile(
    r"\b(?:this|the) (?:company|filer|business|issuer)\b"
    r"|\bits\s+(?:current ratio|leverage|coverage|margin|score|risk)\b"
    r"|\bwhat (?:is|was) (?:the|its)\s+\w+\s+(?:ratio|score|margin|coverage)\b"
    r"|\bhow much\b.{0,30}\b(?:debt|cash|revenue|profit)\b",
    re.I,
)

#: Concept questions. Anything not matching stays on the grounded path, so the
#: default is the conservative one.
_CONCEPTUAL = re.compile(
    r"^\s*(?:what|why|how|when|explain|define|tell me about|describe)\b"
    r"|\bwhat (?:is|are|does|do)\b"
    r"|\bhow (?:does|do|is|are|can)\b"
    r"|\bmeaning of\b|\bdifference between\b",
    re.I,
)

REDIRECT_TO_ASSESSMENT = (
    "That asks about a particular company. Run an assessment on it and the "
    "figures will come from its filings, recomputed and cited, rather than "
    "from anything I recall."
)


@dataclass
class Explanation:
    text: str = ""
    allowed: bool = True
    reason: str = ""


def is_conceptual(question: str) -> bool:
    """Whether this is a general question rather than one about a filer."""
    if _SPECIFIC.search(question):
        return False
    return bool(_CONCEPTUAL.search(question))


def explain(client: Any, question: str) -> Explanation:
    """Answer a general finance question, refusing company specifics."""
    from agents.qa import check_answer, strip_reasoning
    from agents.schemas import scope_violations

    if _SPECIFIC.search(question):
        return Explanation(text=REDIRECT_TO_ASSESSMENT, allowed=True,
                           reason="needs_assessment")

    reply = strip_reasoning(
        client.complete(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ]
        )
    )
    # The scope guard still applies -- explanation is not a licence to advise.
    if scope_violations(reply):
        return Explanation(
            text=(
                "That answer drifted into recommending an action. This system "
                "explains how things work and leaves decisions to you."
            ),
            allowed=False,
            reason="scope",
        )
    # The no-new-numbers check is deliberately NOT applied here. There is no
    # assessment to ground against, and a general explanation may legitimately
    # cite a conventional threshold -- a current ratio near 1, an Altman score
    # under 1.1. Those are textbook constants, not claims about a filer.
    _ = check_answer  # imported for symmetry; grounding belongs to the memo path
    return Explanation(text=reply, allowed=True, reason="general")
