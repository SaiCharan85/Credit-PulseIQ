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

#: Where a continuous risk_score starts counting as a positive prediction.
#: 45 is the lower edge of the ``elevated_risk`` band in ``evals/backtest.py``,
#: so score-based and signal-based flagging agree at the boundary.
FLAG_RISK_SCORE = 45.0

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
        """Whether this counts as a positive prediction for precision/recall.

        Prefers ``risk_score`` and falls back to the signal. On the 169 graded
        cases this is a **no-op**: the two agree on all 169, flag the same 82,
        and give the same weighted F1 of 0.870. The agent keeps its score and
        its signal consistent, so there is no resolution hiding in one that the
        other lacks.

        It is kept because the score is the finer-grained field and a future
        model may not keep them aligned, not because it was measured to help.

        A caution recorded here because it nearly became a claim: weighted F1
        does rise to 0.917 if 94 cases are flagged instead of 82. That is a
        different *operating point*, not a different field, and 94 was chosen
        by looking at the test set. Selecting a threshold that way is tuning
        against the answer, so the flag rate stays where the model puts it.
        """
        if self.risk_score is not None:
            return self.risk_score >= FLAG_RISK_SCORE
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
