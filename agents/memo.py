"""The cited memo: what a human actually receives (SPEC 6, SPEC 8).

The memo is the deliverable, and its structure encodes the project's two
central commitments.

**Every figure is traceable.** Each cited number carries its metric, value,
period end and the fact that it was recomputed from source. A memo cannot be
rendered from numbers the verifier did not reproduce, because the guard gate
runs first and a numeric failure is a hard stop.

**Scope is stated per leg, not implied.** The distress leg is backtested
against real Chapter 11 outcomes. The earnings-quality leg is not -- five
measurements put it at 0.51-0.61 AUC, so its observations appear as context
that never moves the graded signal. Labelling that inside the memo is the
honest-scoping rule made mechanical: a reader cannot mistake which claim
carries evidence, because the memo says so next to the claim.

The memo is advisory. It states risk and never an action -- no "reduce the
line", no "deny the facility". The scope guard enforces this before rendering,
and it is not a stylistic preference: an automated system recommending credit
action is a regulated activity, and the counterfactual needed to validate such
advice does not exist in outcome data.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field

from agents.schemas import Evidence

#: How much weight a section's evidence carries. Printed next to the section.
TIER_BACKTESTED = "backtested"
TIER_DEMO = "demo-grade"
TIER_CONTEXT = "context-only"

TIER_NOTE = {
    TIER_BACKTESTED: "measured against real outcomes on a held-out window",
    TIER_DEMO: "not backtested; illustrative only",
    TIER_CONTEXT: "narrative context; does not affect the graded signal",
}


class MemoSection(BaseModel):
    """One leg's contribution, carrying its own evidential standing."""

    model_config = ConfigDict(frozen=True)

    title: str
    tier: str
    body: str = ""
    evidence: list[Evidence] = Field(default_factory=list)

    @property
    def moves_the_signal(self) -> bool:
        return self.tier == TIER_BACKTESTED


class Memo(BaseModel):
    """A cited, confidence-scored risk assessment for one filer at one date."""

    model_config = ConfigDict(frozen=True)

    cik: int
    as_of: date
    signal: str
    confidence: float = Field(ge=0.0, le=1.0)
    risk_score: float | None = Field(default=None, ge=0.0, le=100.0)
    summary: str = ""
    sections: list[MemoSection] = Field(default_factory=list)
    #: Residual uncertainty the investigator itself reported.
    residual: str = ""
    #: Data problems the guards surfaced -- stale filings, absent tags.
    #: Present in the memo rather than swallowed, because reasoning on
    #: incomplete data without saying so is the failure this project exists to
    #: prevent.
    limitations: list[str] = Field(default_factory=list)
    #: Every tool call made, in order.
    audit_trail: list[dict] = Field(default_factory=list)
    #: Which routing decisions were taken and why.
    routing: list[str] = Field(default_factory=list)
    tool_calls: int = 0

    @property
    def graded_evidence(self) -> list[Evidence]:
        """Evidence from backtested legs only."""
        return [e for s in self.sections if s.moves_the_signal for e in s.evidence]

    def render(self) -> str:
        """Plain-text memo. Deterministic -- no model writes this."""
        lines: list[str] = []
        score = f"  risk_score {self.risk_score:.0f}/100" if self.risk_score is not None else ""
        lines.append(f"CREDIT DISTRESS ASSESSMENT -- CIK {self.cik}, as of {self.as_of}")
        lines.append("=" * 72)
        lines.append(f"SIGNAL: {self.signal.upper()}   confidence {self.confidence:.2f}{score}")
        lines.append("")
        if self.summary:
            lines.append(self.summary.strip())
            lines.append("")

        for section in self.sections:
            lines.append(f"-- {section.title}  [{section.tier}: {TIER_NOTE[section.tier]}]")
            if section.body:
                lines.append(f"   {section.body.strip()}")
            for e in section.evidence:
                period = f" @ {e.period_end}" if e.period_end else ""
                value = "n/a" if e.value is None else f"{e.value:,.4g}"
                note = f"  ({e.note})" if e.note else ""
                lines.append(f"   * {e.metric} = {value}{period}{note}")
            lines.append("")

        if self.residual:
            lines.append(f"RESIDUAL UNCERTAINTY: {self.residual.strip()}")
            lines.append("")
        if self.limitations:
            lines.append("DATA LIMITATIONS")
            for item in self.limitations:
                lines.append(f"   ! {item}")
            lines.append("")
        if self.routing:
            lines.append("ROUTING")
            for step in self.routing:
                lines.append(f"   - {step}")
            lines.append("")

        lines.append(
            f"Every figure above was recomputed from the filed source. "
            f"{self.tool_calls} tool call(s) recorded in the audit trail."
        )
        lines.append(
            "This is a risk assessment for a human reviewer, not a recommendation "
            "to take any action."
        )
        return "\n".join(lines)
