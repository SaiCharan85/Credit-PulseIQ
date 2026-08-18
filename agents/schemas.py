"""Typed schemas for the investigator (SPEC 5, "Output schema").

The investigator returns signal, confidence, cited evidence, and a residual
field that can trigger escalation. Pydantic validation is the first guardrail:
a response that will not parse never reaches the verifier.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

SIGNAL_HEALTHY = "healthy"
SIGNAL_WATCH = "watch"
SIGNAL_ELEVATED = "elevated_risk"
SIGNAL_SEVERE = "severe_risk"
SIGNAL_INSUFFICIENT = "insufficient_evidence"

#: Ordered worst-last, so a fusion step can compare severity.
SIGNAL_ORDER = [SIGNAL_HEALTHY, SIGNAL_WATCH, SIGNAL_ELEVATED, SIGNAL_SEVERE]

VALID_SIGNALS = set(SIGNAL_ORDER) | {SIGNAL_INSUFFICIENT}

#: Scope guard (SPEC 8): the output is a risk assessment, never a decision.
#: Matched case-insensitively against the rationale.
FORBIDDEN_PHRASES = (
    "buy ",
    "sell ",
    "short the",
    "deny the loan",
    "approve the loan",
    "we recommend investing",
    "price target",
    "should be purchased",
)


class Evidence(BaseModel):
    """One cited figure supporting the assessment."""

    model_config = ConfigDict(frozen=True)

    metric: str
    value: float | None
    period_end: date | None = None
    note: str = ""


class InvestigatorOutput(BaseModel):
    """The distress investigator's typed verdict."""

    model_config = ConfigDict(frozen=True)

    cik: int
    as_of: date
    signal: str
    confidence: float = Field(ge=0.0, le=1.0)
    #: Continuous 0-100 distress score, finer-grained than the five-level
    #: signal.
    #:
    #: AUC is a ranking metric, and the signal plus a confidence that clusters
    #: on round values gave roughly a dozen distinct scores across 200 cases --
    #: 146 of them in just two buckets. Ties cap discrimination structurally,
    #: independently of reasoning quality. This lets the agent separate cases
    #: it already distinguishes but had no way to express.
    risk_score: float | None = Field(default=None, ge=0.0, le=100.0)
    rationale: str = ""
    evidence: list[Evidence] = Field(default_factory=list)
    residual: str = ""
    steps_taken: int = 0
    terminated_because: str = ""
    audit_trail: list[dict[str, Any]] = Field(default_factory=list)
    verification_passed: bool = True
    verification_defects: str = ""
    retries: int = 0

    @field_validator("signal")
    @classmethod
    def _known_signal(cls, v: str) -> str:
        if v not in VALID_SIGNALS:
            raise ValueError(f"unknown signal '{v}'; expected one of {sorted(VALID_SIGNALS)}")
        return v

    @property
    def abstained(self) -> bool:
        return self.signal == SIGNAL_INSUFFICIENT

    @property
    def flags_risk(self) -> bool:
        """Whether this counts as a positive prediction for precision/recall."""
        return self.signal in (SIGNAL_ELEVATED, SIGNAL_SEVERE)

    @property
    def severity(self) -> int:
        return SIGNAL_ORDER.index(self.signal) if self.signal in SIGNAL_ORDER else -1

    def is_false_confidence(self, failed: bool) -> bool:
        """High-confidence "healthy" on a name that later failed.

        The catastrophic-error metric (SPEC 7), tracked first-class because it
        is qualitatively worse than being unsure.
        """
        return failed and self.signal in (SIGNAL_HEALTHY, SIGNAL_WATCH) and self.confidence >= 0.7


class ScopeViolation(BaseModel):
    model_config = ConfigDict(frozen=True)

    phrase: str
    where: str


def scope_violations(text: str) -> list[ScopeViolation]:
    """Decision framing the memo must never contain (SPEC 8, scope guard)."""
    lowered = f" {text.lower()} "
    return [
        ScopeViolation(phrase=p.strip(), where="rationale")
        for p in FORBIDDEN_PHRASES
        if p in lowered
    ]
