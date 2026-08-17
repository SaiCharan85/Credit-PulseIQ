"""Deterministic critic for the within-request feedback loop (SPEC 6, SPEC 8).

The critic runs hard checks over a finished investigator response and returns a
specific, actionable defect list. It is the **only** thing permitted to trigger
a retry (PROMPT hard rule 5): retries are deterministic, bounded, and terminate
in abstention. The LLM judge never enters this loop, because looping an agent
against its grader optimises the grader.

Four hard checks, mapping to the guardrail suite:

``numeric``
    Every cited figure must be reproducible from its provenance. Unreproducible
    is a hard fail, not a warning.
``lookahead``
    No cited figure may rest on a filing published on or after the prediction
    date.
``scope``
    The output is a risk assessment, never a decision ("sell", "deny the loan").
``calibration``
    Confidence must be capped when the agent itself reports high residual
    uncertainty, and *insufficient evidence* must not be stated confidently.

Staleness is reported as a warning, not a failure: a filer gone quiet is itself
a signal the investigator should be reasoning about.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date

from pydantic import BaseModel, ConfigDict, Field

from agents.schemas import (
    SIGNAL_INSUFFICIENT,
    InvestigatorOutput,
    scope_violations,
)
from compute.provenance import ComputedValue
from verify.recompute import verify

#: Confidence ceiling once the agent has declared meaningful residual doubt.
RESIDUAL_CONFIDENCE_CAP = 0.6

#: An abstention is a statement of not knowing; it cannot be highly confident.
ABSTENTION_CONFIDENCE_CAP = 0.5

DEFECT_NUMERIC = "numeric"
DEFECT_LOOKAHEAD = "lookahead"
DEFECT_SCOPE = "scope"
DEFECT_CALIBRATION = "calibration"


class Defect(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: str
    detail: str


class CriticReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    defects: list[Defect] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.defects

    def feedback(self) -> str:
        """Concrete instruction for the reviser.

        Deliberately specific: "verification failed" gives a reviser nothing to
        act on, so each defect names the figure and what was wrong with it.
        """
        lines = [f"- [{d.kind}] {d.detail}" for d in self.defects]
        return (
            "Your previous answer failed these deterministic checks. "
            "Correct them and answer again.\n" + "\n".join(lines)
        )


def review(
    output: InvestigatorOutput,
    cited: list[ComputedValue],
    as_of: date,
    cited_line_items: Iterable[str] = (),
) -> CriticReport:
    """Run every hard check over a finished response."""
    defects: list[Defect] = []
    warnings: list[str] = []

    report = verify(cited, as_of=as_of)
    for failure in report.failures:
        defects.append(
            Defect(
                kind=DEFECT_NUMERIC,
                detail=(
                    f"{failure.metric}: stated {failure.stated}, recomputed "
                    f"{failure.recomputed} via {failure.formula} ({failure.reason})"
                ),
            )
        )
    for violation in report.lookahead_violations:
        defects.append(Defect(kind=DEFECT_LOOKAHEAD, detail=violation))
    warnings.extend(report.staleness_warnings)

    for violation in scope_violations(output.rationale):
        defects.append(
            Defect(
                kind=DEFECT_SCOPE,
                detail=(
                    f"rationale contains decision framing ('{violation.phrase}'). "
                    "State risk, never an action to take."
                ),
            )
        )

    if output.residual.strip() and output.confidence > RESIDUAL_CONFIDENCE_CAP:
        defects.append(
            Defect(
                kind=DEFECT_CALIBRATION,
                detail=(
                    f"confidence {output.confidence:.2f} exceeds the "
                    f"{RESIDUAL_CONFIDENCE_CAP} cap while residual uncertainty is "
                    f"reported ('{output.residual[:80]}')"
                ),
            )
        )

    if output.signal == SIGNAL_INSUFFICIENT and output.confidence > ABSTENTION_CONFIDENCE_CAP:
        defects.append(
            Defect(
                kind=DEFECT_CALIBRATION,
                detail=(
                    f"confidence {output.confidence:.2f} is too high for an "
                    "insufficient-evidence verdict"
                ),
            )
        )

    # A figure asserted in the evidence list that was never produced by a tool
    # cannot be verified, so it cannot be cited.
    # Raw line items count as cited: they are filed values with provenance,
    # not derived figures needing recomputation. Omitting them made the critic
    # reject an agent for citing data it had correctly fetched.
    computed = {c.metric for c in cited} | set(cited_line_items)
    for item in output.evidence:
        if item.metric not in computed:
            defects.append(
                Defect(
                    kind=DEFECT_NUMERIC,
                    detail=(
                        f"evidence cites '{item.metric}' but no tool call produced it; "
                        "every figure must come from a tool"
                    ),
                )
            )

    return CriticReport(defects=defects, warnings=warnings)
