"""L3 -- the outcome-based backtest harness (SPEC 7).

The differentiator. Scores an investigator's *reasoning* against real Chapter 11
outcomes under strict as-of cutoffs, and reports the four metrics that matter:

precision / recall
    Against real outcomes, treating ``elevated_risk`` and ``severe_risk`` as
    positive predictions.
lead time
    How far ahead a correct flag was raised. A call made three weeks before the
    petition is worth much less than one made a year out.
false-confidence rate
    High-confidence "healthy" on a name that later failed. Tracked first-class
    as a catastrophic error, and the designated kill-signal metric (SPEC 10).
calibration
    ECE plus a reliability curve.

Cases come from :mod:`models.panel`, so the backtest and the hazard baseline see
*identical* observations under identical as-of rules -- otherwise a comparison
between agent and baseline would be measuring the data split as much as the
reasoning.

Abstentions are reported separately and never counted as either a hit or a
miss. An honest "insufficient evidence" is a different thing from a wrong
answer, and folding them together would punish the behaviour the guardrails
exist to encourage.

Not a pytest module -- this is the harness those tests drive.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Protocol

from agents.schemas import (
    SIGNAL_INSUFFICIENT,
    InvestigatorOutput,
)
from models.hazard import expected_calibration_error, roc_auc
from models.panel import PanelRow

#: Confidence at or above which a benign call counts as *false confidence*.
FALSE_CONFIDENCE_THRESHOLD = 0.7


class Investigator(Protocol):
    def run(self, cik: int, as_of: date, facts: Sequence[Any], **kwargs: Any) -> InvestigatorOutput: ...


@dataclass
class CaseResult:
    """One graded prediction."""

    cik: int
    as_of: date
    label: int
    days_to_event: int | None
    output: InvestigatorOutput

    @property
    def abstained(self) -> bool:
        return self.output.signal == SIGNAL_INSUFFICIENT

    @property
    def predicted_positive(self) -> bool:
        return self.output.flags_risk

    @property
    def true_positive(self) -> bool:
        return self.predicted_positive and self.label == 1

    @property
    def false_positive(self) -> bool:
        return self.predicted_positive and self.label == 0

    @property
    def false_negative(self) -> bool:
        return not self.predicted_positive and not self.abstained and self.label == 1

    @property
    def false_confidence(self) -> bool:
        return self.output.is_false_confidence(failed=self.label == 1)

    @property
    def risk_probability(self) -> float:
        """Confidence mapped onto P(distress) for calibration.

        The agent states confidence *in its assessment*, not a probability of
        bankruptcy, so a benign call at 0.9 confidence is read as 0.1 risk.
        The mapping is stated because it determines the reliability curve.
        """
        if self.abstained:
            return 0.5
        return self.output.confidence if self.predicted_positive else 1.0 - self.output.confidence


@dataclass
class Sample:
    """A stratified subsample and the correction it implies.

    Positives are scarce (100 in the test split) and each agent case costs an
    API call, so negatives are subsampled while every positive is kept. That
    inflates the apparent base rate, which inflates precision -- so the
    sampling fraction is carried alongside the cases and used to correct it.
    Recall, lead time and the false-confidence rate are unaffected, because
    they are computed over positives only.
    """

    cases: list[PanelRow]
    negative_fraction: float = 1.0
    n_positives: int = 0
    n_negatives_kept: int = 0
    n_negatives_total: int = 0


def stratified_sample(
    cases: Sequence[PanelRow], max_negatives: int, seed: int = 0, max_positives: int = 0
) -> Sample:
    """Keep every positive, sample negatives down to ``max_negatives``.

    ``max_positives`` caps positives too, for a pilot run under a token budget.
    It weakens every metric, so it is opt-in and reported.
    """
    import random

    positives = [c for c in cases if c.label == 1]
    if max_positives and max_positives < len(positives):
        positives = random.Random(seed).sample(positives, max_positives)
    negatives = [c for c in cases if c.label == 0]
    if max_negatives <= 0 or max_negatives >= len(negatives):
        kept = negatives
        fraction = 1.0
    else:
        kept = random.Random(seed).sample(negatives, max_negatives)
        fraction = max_negatives / len(negatives)
    selected = sorted(positives + kept, key=lambda c: (c.observation_date, c.cik))
    return Sample(
        cases=selected,
        negative_fraction=fraction,
        n_positives=len(positives),
        n_negatives_kept=len(kept),
        n_negatives_total=len(negatives),
    )


@dataclass
class L3Report:
    """Aggregate backtest metrics."""

    n_cases: int = 0
    n_positives: int = 0
    n_abstained: int = 0
    honest_abstentions: int = 0
    protocol_failures: int = 0
    negative_fraction: float = 1.0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    false_confidence_cases: int = 0
    lead_times: list[int] = field(default_factory=list)
    auc: float | None = None
    ece: float | None = None
    mean_steps: float = 0.0
    mean_tool_calls: float = 0.0
    verification_failures: int = 0
    signal_counts: dict[str, int] = field(default_factory=dict)

    @property
    def graded(self) -> int:
        """Cases that produced a verdict. Abstentions are excluded."""
        return self.n_cases - self.n_abstained

    @property
    def precision(self) -> float | None:
        flagged = self.true_positives + self.false_positives
        return self.true_positives / flagged if flagged else None

    @property
    def recall(self) -> float | None:
        actual = self.true_positives + self.false_negatives
        return self.true_positives / actual if actual else None

    @property
    def f1(self) -> float | None:
        p, r = self.precision, self.recall
        if not p or not r:
            return None
        return 2 * p * r / (p + r)

    @property
    def corrected_precision(self) -> float | None:
        """Precision adjusted back to the full negative population.

        Under subsampling only false positives are undercounted, by the
        sampling fraction, so scaling them back recovers the population value.
        Without this a 400-case sample would report a flattering precision
        purely because negatives were thinned.
        """
        if self.negative_fraction >= 1.0:
            return self.precision
        scaled_fp = self.false_positives / self.negative_fraction
        denominator = self.true_positives + scaled_fp
        return self.true_positives / denominator if denominator else None

    @property
    def abstention_rate(self) -> float:
        return self.n_abstained / self.n_cases if self.n_cases else 0.0

    @property
    def protocol_failure_rate(self) -> float:
        """Share of cases the agent could not complete.

        Kept apart from honest abstention. A model too weak to hold the tool
        protocol produces the same ``insufficient_evidence`` verdict as a model
        exercising good judgment, and conflating them would let incompetence
        read as admirable caution -- exactly what happened with the local 1.5B.
        """
        return self.protocol_failures / self.n_cases if self.n_cases else 0.0

    @property
    def false_confidence_rate(self) -> float | None:
        """Share of names that failed where the agent said healthy, confidently."""
        return self.false_confidence_cases / self.n_positives if self.n_positives else None

    @property
    def median_lead_time_days(self) -> float | None:
        return statistics.median(self.lead_times) if self.lead_times else None

    def summary(self) -> str:
        def pct(x: float | None) -> str:
            return f"{x:.3f}" if x is not None else "n/a"

        lead = self.median_lead_time_days
        corrected = (
            ""
            if self.negative_fraction >= 1.0
            else f"  [base-rate corrected {pct(self.corrected_precision)}]"
        )
        return "\n".join(
            [
                f"cases                 {self.n_cases}  (positives {self.n_positives})",
                f"abstained             {self.n_abstained}  ({self.abstention_rate:.1%})"
                f"  honest {self.honest_abstentions} / protocol-failure {self.protocol_failures}",
                f"precision / recall    {pct(self.precision)} / {pct(self.recall)}"
                f"   F1 {pct(self.f1)}{corrected}",
                f"median lead time      {f'{lead:.0f} days' if lead is not None else 'n/a'}",
                f"FALSE-CONFIDENCE RATE {pct(self.false_confidence_rate)}"
                f"  ({self.false_confidence_cases}/{self.n_positives})",
                f"calibration (ECE)     {pct(self.ece)}    AUC {pct(self.auc)}",
                f"mean steps / calls    {self.mean_steps:.1f} / {self.mean_tool_calls:.1f}",
                f"verification failures {self.verification_failures}",
                f"signals               {dict(sorted(self.signal_counts.items()))}",
            ]
        )


#: Termination reasons that mean the agent could not complete the protocol,
#: as opposed to deciding it lacked evidence.
PROTOCOL_FAILURE_REASONS = frozenset(
    {"step_budget_exhausted", "retries_exhausted", "unparseable_response", "case_error"}
)


def grade(results: Sequence[CaseResult], negative_fraction: float = 1.0) -> L3Report:
    """Aggregate graded predictions into the L3 report."""
    report = L3Report(n_cases=len(results), negative_fraction=negative_fraction)
    if not results:
        return report

    for r in results:
        report.signal_counts[r.output.signal] = report.signal_counts.get(r.output.signal, 0) + 1
        if r.label == 1:
            report.n_positives += 1
        if r.abstained:
            report.n_abstained += 1
            if r.output.terminated_because in PROTOCOL_FAILURE_REASONS:
                report.protocol_failures += 1
            else:
                report.honest_abstentions += 1
            continue
        if r.true_positive:
            report.true_positives += 1
            if r.days_to_event is not None:
                report.lead_times.append(r.days_to_event)
        elif r.false_positive:
            report.false_positives += 1
        elif r.false_negative:
            report.false_negatives += 1
        if r.false_confidence:
            report.false_confidence_cases += 1
        if not r.output.verification_passed:
            report.verification_failures += 1

    graded = [r for r in results if not r.abstained]
    if graded:
        y = [r.label for r in graded]
        p = [r.risk_probability for r in graded]
        report.auc = roc_auc(y, p)
        report.ece = expected_calibration_error(y, p)
    report.mean_steps = sum(r.output.steps_taken for r in results) / len(results)
    report.mean_tool_calls = sum(len(r.output.audit_trail) for r in results) / len(results)
    return report


def reliability_curve(
    results: Sequence[CaseResult], bins: int = 5
) -> list[dict[str, float | int]]:
    """Binned confidence vs realised outcome frequency -- the calibration curve."""
    graded = [r for r in results if not r.abstained]
    out: list[dict[str, float | int]] = []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        chunk = [
            r
            for r in graded
            if r.risk_probability >= lo and (r.risk_probability < hi or b == bins - 1)
        ]
        if not chunk:
            continue
        out.append(
            {
                "bin_low": lo,
                "bin_high": hi,
                "n": len(chunk),
                "mean_confidence": sum(r.risk_probability for r in chunk) / len(chunk),
                "observed_rate": sum(r.label for r in chunk) / len(chunk),
            }
        )
    return out


def run_backtest(
    investigator: Investigator,
    cases: Sequence[PanelRow],
    facts_for: Callable[[int], Sequence[Any]],
    on_case: Callable[[int, CaseResult], None] | None = None,
    **run_kwargs: Any,
) -> list[CaseResult]:
    """Run the investigator over panel cases and grade each prediction.

    ``cases`` carry their own ``observation_date``; the investigator is given
    that date and nothing else, so the as-of discipline is inherited from the
    panel rather than reimplemented here.
    """
    from agents.llm import BudgetExhausted, InfrastructureError

    results: list[CaseResult] = []
    for n, case in enumerate(cases, 1):
        facts = facts_for(case.cik)
        try:
            output = investigator.run(case.cik, case.observation_date, facts, **run_kwargs)
        except InfrastructureError:
            # Not an agent failure and not a partial result worth grading: an
            # unusable endpoint means no case ran properly.
            raise
        except BudgetExhausted as exc:
            # A quota stop is not a failure. Everything completed so far is
            # real and already cached, so return it rather than discarding a
            # partial run -- the metrics computed over positives (lead time,
            # false-confidence rate) are valid on any subset.
            import sys

            print(f"\nquota reached after {len(results)} cases: {exc}", file=sys.stderr)
            break
        except Exception as exc:  # noqa: BLE001
            # One bad case must not destroy the run. A three-hour sweep will
            # hit provider errors and malformed edge cases; each becomes a
            # recorded protocol failure so the other 399 still produce numbers.
            output = InvestigatorOutput(
                cik=case.cik,
                as_of=case.observation_date,
                signal=SIGNAL_INSUFFICIENT,
                confidence=0.0,
                rationale=f"case failed: {type(exc).__name__}: {str(exc)[:200]}",
                terminated_because="case_error",
                verification_passed=False,
            )
        result = CaseResult(
            cik=case.cik,
            as_of=case.observation_date,
            label=case.label,
            days_to_event=case.days_to_event,
            output=output,
        )
        results.append(result)
        if on_case:
            on_case(n, result)
    return results


def assert_no_lookahead(results: Sequence[CaseResult]) -> None:
    """Every cited figure must predate its own prediction date.

    Asserted in the harness as well as enforced in the data layer: the cardinal
    sin gets checked wherever it could still appear (PROMPT hard rule 4).
    """
    for r in results:
        for call in r.output.audit_trail:
            filed = call.get("arguments", {}).get("filed")
            if filed and str(filed) >= r.as_of.isoformat():
                raise AssertionError(
                    f"lookahead: cik={r.cik} as_of={r.as_of} used a filing dated {filed}"
                )
        if r.output.as_of != r.as_of:
            raise AssertionError(f"case as_of mismatch for cik={r.cik}")
