"""The four guards that run before any memo ships (SPEC 8).

The critic already enforces three of these inside the investigator's retry
loop. This module is the *gate*: the last check before a memo reaches a human,
composing all four and returning a single verdict with its reasons.

Two are hard failures -- a memo that trips them is not rendered:

* **Numeric verification.** Any figure the verifier cannot reproduce from the
  filed source blocks the memo. Not a warning; the whole value proposition is
  that every number is checkable.
* **Scope.** The output states risk, never an action. Decision framing is
  refused rather than softened.

Two shape the memo rather than blocking it:

* **Calibration.** Confidence is capped where residual uncertainty is
  reported, and *insufficient evidence* is a valid terminal state rather than
  a failure to be retried away.
* **Data freshness.** Stale or absent filings are *surfaced in the memo*, not
  swallowed. This is the guard the critic did not cover, and it is the one
  that matters most for silent failure: a filer whose last annual report is
  three years old will still yield ratios, and those ratios will look like
  ordinary numbers unless something says otherwise.

The asymmetry is deliberate. Fabrication and scope violations make a memo
wrong, so they stop it. Stale data makes a memo *conditional*, so it travels
with the memo instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from agents.critic import (
    DEFECT_LOOKAHEAD,
    DEFECT_NUMERIC,
    DEFECT_SCOPE,
    CriticReport,
    review,
)
from agents.schemas import InvestigatorOutput, scope_violations
from compute.provenance import ComputedValue

#: Beyond this, an annual report is old enough that a reader must be told.
#: 15 months: a filer on an annual cycle should have reported within 12, and
#: the extra quarter absorbs ordinary filing lag without crying wolf.
STALE_AFTER_DAYS = 456

GUARD_NUMERIC = "numeric_verification"
GUARD_SCOPE = "scope"
GUARD_CALIBRATION = "calibration"
GUARD_FRESHNESS = "data_freshness"

#: Guards whose failure blocks the memo entirely.
BLOCKING = frozenset({GUARD_NUMERIC, GUARD_SCOPE})


@dataclass
class GuardReport:
    """The gate's verdict, with everything a reader needs to judge it."""

    blocked: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    critic: CriticReport | None = None

    @property
    def may_ship(self) -> bool:
        return not self.blocked

    def summary(self) -> str:
        if self.may_ship:
            n = len(self.limitations)
            return f"guards passed ({n} limitation(s) surfaced)"
        return f"BLOCKED: {'; '.join(self.blocked)}"


def check_freshness(
    latest_period_end: date | None, as_of: date, stale_after_days: int = STALE_AFTER_DAYS
) -> list[str]:
    """Say so when the newest usable filing is old, or absent entirely.

    Returned as limitations rather than defects. The figures are real -- they
    are simply about a world that may have moved, and a reader can only weigh
    that if the memo tells them.
    """
    if latest_period_end is None:
        return ["no annual period could be resolved from the filings visible at this date"]
    age = (as_of - latest_period_end).days
    if age > stale_after_days:
        years = age / 365.25
        return [
            f"most recent annual period ends {latest_period_end} -- "
            f"{age} days ({years:.1f} years) before the prediction date; "
            "every ratio here describes that vintage"
        ]
    return []


def run_guards(
    output: InvestigatorOutput,
    cited: list[ComputedValue],
    as_of: date,
    latest_period_end: date | None = None,
    cited_line_items: tuple[str, ...] = (),
) -> GuardReport:
    """Run all four guards over a finished investigation.

    ``cited`` is the investigator's recomputed values. When it is empty the
    numeric guard *enforces* the verdict the investigator already reached
    rather than re-deriving it: re-verification needs the toolbox, and running
    ``review`` against no cited values would report every figure as unsourced
    and block every memo. Enforcing a verdict reached with the data beats
    recomputing one without it.
    """
    report = GuardReport()

    if not cited:
        if not output.verification_passed:
            report.blocked.append(
                f"{GUARD_NUMERIC}: {output.verification_defects or 'verification failed'}"
            )
        for violation in scope_violations(output.rationale):
            report.blocked.append(
                f"{GUARD_SCOPE}: rationale contains decision framing "
                f"('{violation.phrase}')"
            )
        report.limitations.extend(check_freshness(latest_period_end, as_of))
        if output.abstained:
            report.limitations.append(
                "the investigator reported insufficient evidence; this is an "
                "abstention, not an assessment of low risk"
            )
        return report

    critic = review(output, cited, as_of, cited_line_items=cited_line_items)
    report.critic = critic

    for defect in critic.defects:
        if defect.kind in (DEFECT_NUMERIC, DEFECT_LOOKAHEAD):
            report.blocked.append(f"{GUARD_NUMERIC}: {defect.detail}")
        elif defect.kind == DEFECT_SCOPE:
            report.blocked.append(f"{GUARD_SCOPE}: {defect.detail}")
        else:
            # Calibration and self-contradiction shape the memo; the retry loop
            # already had its chance to fix them.
            report.warnings.append(f"{GUARD_CALIBRATION}: {defect.detail}")

    report.warnings.extend(critic.warnings)
    report.limitations.extend(check_freshness(latest_period_end, as_of))

    # An abstention is a valid terminal state, not a defect -- but a reader
    # must not mistake it for a clean bill of health.
    if output.abstained:
        report.limitations.append(
            "the investigator reported insufficient evidence; this is an "
            "abstention, not an assessment of low risk"
        )
    return report
