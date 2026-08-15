"""Independent numeric verification (SPEC 6, SPEC 8).

The verifier re-executes a formula against the recorded raw inputs and compares
the result to the stated value. It never trusts ``ComputedValue.value``. A
figure that cannot be reproduced is a hard failure, not a warning -- that is
what makes the memo auditable and the evals gradeable (README, "Deterministic
verification").

Plain deterministic code. This is a critic, not an agent (PROMPT hard rule 3),
and it is the *only* thing permitted to trigger a within-request retry
(hard rule 5).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import date

from pydantic import BaseModel, ConfigDict, Field

from compute.provenance import FORMULAS, ComputedValue

#: Relative tolerance for float comparison. Tight on purpose: the verifier
#: re-runs the same arithmetic on the same inputs, so anything beyond
#: floating-point noise means the value did not come from this formula.
REL_TOLERANCE = 1e-9

FAIL_UNKNOWN_FORMULA = "unknown_formula"
FAIL_MISSING_INPUTS = "missing_inputs"
FAIL_VALUE_MISMATCH = "value_mismatch"
FAIL_UNDEFINED_MISMATCH = "undefined_mismatch"
FAIL_LOOKAHEAD = "lookahead"
FAIL_STALE = "stale"


class RecomputeResult(BaseModel):
    """Outcome of re-deriving one figure from its provenance."""

    model_config = ConfigDict(frozen=True)

    metric: str
    formula: str
    stated: float | None
    recomputed: float | None
    agrees: bool
    reason: str = ""

    @property
    def failed(self) -> bool:
        return not self.agrees


class VerificationReport(BaseModel):
    """Aggregate verdict over every figure a response cites."""

    model_config = ConfigDict(frozen=True)

    results: list[RecomputeResult] = Field(default_factory=list)
    lookahead_violations: list[str] = Field(default_factory=list)
    staleness_warnings: list[str] = Field(default_factory=list)

    @property
    def failures(self) -> list[RecomputeResult]:
        return [r for r in self.results if r.failed]

    @property
    def passed(self) -> bool:
        """Hard pass/fail. Staleness warns; unreproducible figures and
        lookahead violations block."""
        return not self.failures and not self.lookahead_violations

    def defect_summary(self) -> str:
        """The specific defect text fed back to a bounded retry (SPEC feedback
        loops). Deliberately concrete -- "recompute failed" is not actionable."""
        parts = [
            f"{r.metric}: stated {r.stated}, recomputed {r.recomputed} via {r.formula}"
            for r in self.failures
        ]
        parts.extend(self.lookahead_violations)
        return "; ".join(parts)


def _agrees(stated: float | None, recomputed: float | None) -> bool:
    if stated is None or recomputed is None:
        return stated is None and recomputed is None
    if stated == recomputed:
        return True
    scale = max(abs(stated), abs(recomputed), 1e-12)
    return abs(stated - recomputed) / scale <= REL_TOLERANCE


def recompute(value: ComputedValue) -> RecomputeResult:
    """Re-derive one figure from its recorded inputs."""
    if value.formula not in FORMULAS:
        return RecomputeResult(
            metric=value.metric,
            formula=value.formula,
            stated=value.value,
            recomputed=None,
            agrees=False,
            reason=FAIL_UNKNOWN_FORMULA,
        )

    f = FORMULAS[value.formula]
    missing = [k for k in f.inputs if k not in value.inputs]
    if missing:
        # An undefined value that declares missing inputs is self-consistent:
        # the metric correctly reports that it could not be computed.
        if value.value is None:
            return RecomputeResult(
                metric=value.metric,
                formula=value.formula,
                stated=None,
                recomputed=None,
                agrees=True,
                reason="undefined with missing inputs (consistent)",
            )
        return RecomputeResult(
            metric=value.metric,
            formula=value.formula,
            stated=value.value,
            recomputed=None,
            agrees=False,
            reason=f"{FAIL_MISSING_INPUTS}: {', '.join(sorted(missing))}",
        )

    recomputed = f(**{k: value.inputs[k].value for k in f.inputs})
    ok = _agrees(value.value, recomputed)
    reason = ""
    if not ok:
        reason = FAIL_UNDEFINED_MISMATCH if None in (value.value, recomputed) else FAIL_VALUE_MISMATCH
    return RecomputeResult(
        metric=value.metric,
        formula=value.formula,
        stated=value.value,
        recomputed=recomputed,
        agrees=ok,
        reason=reason,
    )


def check_lookahead(value: ComputedValue, as_of: date) -> list[str]:
    """Inputs must have been public strictly before ``as_of``.

    Belt and braces: the data layer already filters, but a figure can reach the
    verifier through paths the harness did not construct (a cached tool result,
    a hand-built fixture). The cardinal sin gets checked at the boundary too.
    """
    bad = []
    for name, ref in value.inputs.items():
        if ref.filed >= as_of:
            bad.append(
                f"{value.metric}.{name} uses {ref.tag}@{ref.period_end} filed {ref.filed}, "
                f"not public before {as_of}"
            )
    return bad


def check_staleness(value: ComputedValue, as_of: date, max_age_days: int = 550) -> list[str]:
    """Warn when the newest input is old enough that the picture may have moved.

    Default 550 days ~ 18 months: past the point where the next annual filing
    should have arrived. A filer that has gone quiet is itself a distress
    signal, so this surfaces rather than silently reasoning on stale data
    (SPEC 8, data-freshness guard).
    """
    stamp = value.as_of
    if stamp is None:
        return [f"{value.metric}: no dated inputs"]
    age = (as_of - stamp).days
    if age > max_age_days:
        return [f"{value.metric}: newest input filed {stamp} ({age} days before {as_of})"]
    return []


def verify(
    values: Iterable[ComputedValue],
    as_of: date | None = None,
    max_age_days: int = 550,
) -> VerificationReport:
    """Verify every figure: reproducible, not from the future, not stale."""
    values = list(values)
    results = [recompute(v) for v in values]
    lookahead: list[str] = []
    stale: list[str] = []
    if as_of is not None:
        for v in values:
            lookahead.extend(check_lookahead(v, as_of))
            stale.extend(check_staleness(v, as_of, max_age_days))
    return VerificationReport(
        results=results, lookahead_violations=lookahead, staleness_warnings=stale
    )


def verify_or_raise(values: Sequence[ComputedValue], as_of: date | None = None) -> VerificationReport:
    """Verify and raise on hard failure. For CI and harness assertions."""
    report = verify(values, as_of=as_of)
    if not report.passed:
        raise AssertionError(f"verification failed: {report.defect_summary()}")
    return report
