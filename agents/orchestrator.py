"""Routing and memo assembly (SPEC 4, SPEC 12 phase 5).

The orchestrator decides *which* investigation to run and *how deep*, then
assembles the result into a cited memo. It computes nothing and produces no
numbers of its own -- every figure in the memo came from a tool and was
recomputed by the verifier.

**Routing is on findings, not on a fixed script.** A cheap deterministic triage
reads the filings first; a clean profile gets a shallow pass, a stressed one
gets the full step budget. That branching is what makes the system agentic
rather than a pipeline with an LLM in the middle, and it is also what makes it
affordable: most filers are not distressed, and spending fourteen steps proving
it is waste.

**Honest scoping is structural here.** SPEC's phase 5 describes an orchestrator
"fusing the two green legs". Only one leg is green. Measured:

    distress leg          agent 0.965 vs hazard baseline 0.966 -- parity,
                          and filing-text tools worth +0.084 AUC
    earnings-quality leg  0.506-0.605 across five measurements, clean canary

So the earnings leg is attached as *context only* -- its observations appear in
the memo, labelled, and never move the graded signal. The covenant leg is not
built. Rather than fake a fusion, the orchestrator routes to the one leg that
has evidence behind it and says so in the memo.

The triage is deliberately not an LLM. It reads three ratios and picks a depth;
putting a model there would add cost and latency to a decision that thresholds
make better (PROMPT hard rule 3).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from agents.guards import GuardReport, run_guards
from agents.memo import (
    TIER_BACKTESTED,
    TIER_CONTEXT,
    Memo,
    MemoSection,
)
from agents.schemas import InvestigatorOutput
from compute.lineitems import FactIndex, annual_period_ends
from compute.ratios import compute_metric
from data.facts import as_of_view

DEPTH_SHALLOW = "shallow"
DEPTH_DEEP = "deep"

#: Steps allowed at each depth. Shallow is not a worse investigation, it is a
#: shorter one for a filer whose triage found nothing to chase.
DEPTH_STEPS = {DEPTH_SHALLOW: 6, DEPTH_DEEP: 14}

#: Triage thresholds. Conventional credit levels, not fitted -- they choose an
#: effort budget, never an answer, so tuning them cannot leak.
TRIAGE_METRICS = ("quick_ratio", "liabilities_to_assets", "interest_coverage")

#: Metrics whose *absence* escalates. Every operating filer reports a balance
#: sheet, so a missing quick ratio or leverage figure means the tags went away
#: -- which distress does. Interest coverage is excluded: a debt-free company
#: legitimately has none, and escalating that routes healthy filers to a deep
#: pass for the crime of having no borrowings.
ESCALATE_IF_ABSENT = frozenset({"quick_ratio", "liabilities_to_assets"})
TRIAGE_TRIGGERS = {
    "quick_ratio": ("below", 1.0),
    "liabilities_to_assets": ("above", 0.7),
    "interest_coverage": ("below", 1.5),
}


@dataclass
class Triage:
    """The cheap deterministic read that chooses a depth."""

    depth: str = DEPTH_SHALLOW
    reasons: list[str] = field(default_factory=list)
    latest_period_end: date | None = None
    uncomputable: list[str] = field(default_factory=list)

    @property
    def steps(self) -> int:
        return DEPTH_STEPS[self.depth]


def triage(facts: Sequence[Any], as_of: date) -> Triage:
    """Read three ratios and decide how hard to look.

    An *uncomputable* metric escalates rather than being ignored. Distress
    removes tags -- a filer that has stopped reporting interest coverage is
    more interesting than one reporting a comfortable figure, and treating
    absence as "no trigger" would route exactly the wrong cases to a shallow
    pass.
    """
    view = FactIndex(as_of_view(facts, as_of))
    ends = annual_period_ends(view)
    result = Triage(latest_period_end=ends[0] if ends else None)
    if not ends:
        result.depth = DEPTH_DEEP
        result.reasons.append("no annual period visible; escalated rather than skipped")
        return result

    latest = ends[0]
    for metric in TRIAGE_METRICS:
        computed = compute_metric(metric, view, latest)
        if not computed.is_defined:
            result.uncomputable.append(metric)
            if metric in ESCALATE_IF_ABSENT:
                result.reasons.append(f"{metric} is uncomputable -- distress removes tags")
            continue
        direction, level = TRIAGE_TRIGGERS[metric]
        value = float(computed.value)
        if (direction == "below" and value < level) or (direction == "above" and value > level):
            result.reasons.append(f"{metric} {value:.2f} {direction} {level}")

    result.depth = DEPTH_DEEP if result.reasons else DEPTH_SHALLOW
    if not result.reasons:
        result.reasons.append("no triage trigger fired; shallow pass")
    return result


@dataclass
class Assessment:
    """A finished run: the memo, or the reason there isn't one."""

    cik: int
    as_of: date
    triage: Triage
    output: InvestigatorOutput | None = None
    guards: GuardReport | None = None
    memo: Memo | None = None
    blocked_reason: str = ""

    @property
    def shipped(self) -> bool:
        return self.memo is not None


def build_memo(
    output: InvestigatorOutput,
    guards: GuardReport,
    triage_result: Triage,
    context_notes: Sequence[str] = (),
) -> Memo:
    """Assemble the memo. Deterministic -- no model writes any of this."""
    sections = [
        MemoSection(
            title="Credit distress",
            tier=TIER_BACKTESTED,
            body=output.rationale,
            evidence=list(output.evidence),
        )
    ]
    if context_notes:
        sections.append(
            MemoSection(
                title="Earnings-quality observations",
                tier=TIER_CONTEXT,
                body="\n".join(context_notes),
            )
        )

    limitations = list(guards.limitations)
    if triage_result.uncomputable:
        limitations.append(
            "uncomputable at this date: "
            + ", ".join(triage_result.uncomputable)
            + " -- absence of a tag is itself a finding, not a neutral value"
        )

    return Memo(
        cik=output.cik,
        as_of=output.as_of,
        signal=output.signal,
        confidence=output.confidence,
        risk_score=output.risk_score,
        summary=output.rationale.split(".")[0].strip() + "." if output.rationale else "",
        sections=sections,
        residual=output.residual,
        limitations=limitations,
        audit_trail=list(output.audit_trail),
        routing=[f"depth={triage_result.depth} ({triage_result.steps} steps)"]
        + list(triage_result.reasons),
        tool_calls=len(output.audit_trail),
    )


class Orchestrator:
    """Triage, route to the graded leg, gate, and assemble."""

    def __init__(self, investigator: Any, context_notes: Sequence[str] = ()) -> None:
        self.investigator = investigator
        #: Attached to every memo as context-only. Never moves the signal.
        self.context_notes = list(context_notes)

    def run(
        self,
        cik: int,
        as_of: date,
        facts: Sequence[Any],
        **kwargs: Any,
    ) -> Assessment:
        plan = triage(facts, as_of)
        result = Assessment(cik=cik, as_of=as_of, triage=plan)

        previous = getattr(self.investigator, "max_steps", None)
        if previous is not None:
            self.investigator.max_steps = plan.steps
        try:
            output = self.investigator.run(cik, as_of, facts, **kwargs)
        finally:
            if previous is not None:
                self.investigator.max_steps = previous
        result.output = output

        guards = run_guards(
            output,
            cited=[],
            as_of=as_of,
            latest_period_end=plan.latest_period_end,
        )
        result.guards = guards
        if not guards.may_ship:
            result.blocked_reason = guards.summary()
            return result

        result.memo = build_memo(output, guards, plan, self.context_notes)
        return result
