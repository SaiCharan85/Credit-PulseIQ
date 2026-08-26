"""A deterministic rule-based investigator -- the control agent.

Same interface and same tools as the ReAct loop, but the policy is a fixed
threshold cascade with no model in it. It exists for two reasons:

* **The harness runs without an endpoint.** L3 produces real numbers on real
  outcomes today, so the backtest is exercised end to end rather than sitting
  untested until a GPU appears.
* **It is the control the LLM has to beat.** An agent that cannot outperform a
  threshold cascade over the same tools is not earning its inference cost, and
  reporting that honestly is the point of the eval ladder.

It is deliberately *not* an agent: no branching on findings beyond the cascade,
no judgment about when to stop. That is the difference the L3 comparison
measures.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from typing import Any

from agents.critic import review
from agents.schemas import (
    SIGNAL_ELEVATED,
    SIGNAL_HEALTHY,
    SIGNAL_INSUFFICIENT,
    SIGNAL_SEVERE,
    SIGNAL_WATCH,
    Evidence,
    InvestigatorOutput,
)
from agents.tools import STATUS_FLAG, STATUS_SEVERE, ToolBox

#: Checked in order; each contributes to the severity tally.
CASCADE: tuple[str, ...] = (
    "liabilities_to_assets",
    "current_ratio",
    "interest_coverage",
    "ocf_to_debt",
    "altman_z_double_prime",
    "net_margin",
    "debt_to_assets",
)

#: Severe findings needed for each verdict.
SEVERE_FOR_SEVERE_RISK = 3
SEVERE_FOR_ELEVATED = 1
FLAGS_FOR_WATCH = 2

#: Below this many computable metrics there is not enough to judge on.
MIN_DEFINED_METRICS = 3

#: A severe breach counts double a flag. Both matter; a measure past the severe
#: level is a stronger statement than one merely near it, and collapsing them
#: would put a filer with five near-misses level with one already insolvent.
SEVERE_WEIGHT = 2


def risk_score(signal: str, severe: Sequence[str], flagged: Sequence[str]) -> float:
    """An ordinal 0-100 severity, positioned *inside* the signal's own band.

    This arm emitted no score at all, which left "what is its risk score" --
    an entirely reasonable question -- unanswerable for every filer it
    assessed. The five-level signal was the only output, and five levels cannot
    separate two filers sitting in the same band.

    The score is derived from the band the signal already implies, then placed
    within it by how many thresholds broke. Deriving it independently and
    hoping the two agreed is exactly what the critic's consistency check exists
    to catch -- a first attempt at this scored an ``elevated_risk`` filer at 36
    against a band of [45, 75], and the guard blocked the memo, correctly. A
    score that cannot contradict its own signal is better than one that is
    checked for contradiction afterwards.

    Deliberately a *rank*, not a probability. The test universe is enriched
    with distressed filers far above the population base rate, so an absolute
    probability drawn from it would not transfer. The Q&A guard enforces that
    separately, refusing any answer that frames this number as a chance of
    default.
    """
    from agents.critic import SIGNAL_BANDS

    low, high = SIGNAL_BANDS.get(signal, (0.0, 0.0))
    if high <= low:
        return low
    worst = SEVERE_WEIGHT * len(CASCADE)
    weighted = min(SEVERE_WEIGHT * len(severe) + len(flagged), worst)
    # Kept off the band edges: a filer exactly on a boundary reads as belonging
    # to either neighbour, and the boundaries are where readers look hardest.
    span = (high - low) * 0.9
    return round(low + (high - low) * 0.05 + span * (weighted / worst), 1)


class RuleBasedInvestigator:
    """Threshold cascade over the same typed tools."""

    name = "rule-based"

    def run(
        self,
        cik: int,
        as_of: date,
        facts: Sequence[Any],
        peer_facts: dict[int, Sequence[Any]] | None = None,
        sic_by_cik: dict[int, str] | None = None,
        events: Sequence[Any] = (),
    ) -> InvestigatorOutput:
        tools = ToolBox(
            cik=cik,
            as_of=as_of,
            facts=facts,
            peer_facts=peer_facts,
            sic_by_cik=sic_by_cik,
            events=events,
        )
        tools.available_periods()

        severe: list[str] = []
        flagged: list[str] = []
        evidence: list[Evidence] = []
        defined = 0

        for metric in CASCADE:
            result = tools.check_threshold(metric)
            if "error" in result:
                continue
            value = result.get("value")
            if value is None:
                continue
            defined += 1
            status = result["status"]
            if status == STATUS_SEVERE:
                severe.append(metric)
            elif status == STATUS_FLAG:
                flagged.append(metric)
            if status in (STATUS_SEVERE, STATUS_FLAG):
                evidence.append(Evidence(metric=metric, value=value, note=status))

        if defined < MIN_DEFINED_METRICS:
            return InvestigatorOutput(
                cik=cik,
                as_of=as_of,
                signal=SIGNAL_INSUFFICIENT,
                confidence=0.0,
                rationale=(
                    f"only {defined} of {len(CASCADE)} distress metrics could be computed "
                    f"from filings public as of {as_of}"
                ),
                residual="insufficient computable metrics",
                steps_taken=tools.call_count(),
                terminated_because="insufficient_data",
                audit_trail=tools.audit_trail(),
            )

        if len(severe) >= SEVERE_FOR_SEVERE_RISK:
            signal, confidence = SIGNAL_SEVERE, 0.80
        elif len(severe) >= SEVERE_FOR_ELEVATED:
            signal, confidence = SIGNAL_ELEVATED, 0.65
        elif len(flagged) >= FLAGS_FOR_WATCH:
            signal, confidence = SIGNAL_WATCH, 0.55
        else:
            signal, confidence = SIGNAL_HEALTHY, 0.60

        # Coverage discounts confidence: a verdict on half the metrics is a
        # weaker verdict, and saying so is cheaper than being wrong.
        coverage = defined / len(CASCADE)
        confidence = round(confidence * (0.6 + 0.4 * coverage), 2)

        rationale = (
            f"{len(severe)} metric(s) past the severe threshold "
            f"({', '.join(severe) or 'none'}); {len(flagged)} flagged "
            f"({', '.join(flagged) or 'none'}); {defined}/{len(CASCADE)} computable."
        )
        residual = (
            f"{len(CASCADE) - defined} metric(s) could not be computed"
            if defined < len(CASCADE)
            else ""
        )
        if residual:
            confidence = min(confidence, 0.6)

        output = InvestigatorOutput(
            cik=cik,
            as_of=as_of,
            signal=signal,
            confidence=confidence,
            risk_score=risk_score(signal, severe, flagged),
            rationale=rationale,
            evidence=evidence,
            residual=residual,
            steps_taken=tools.call_count(),
            terminated_because="cascade_complete",
            audit_trail=tools.audit_trail(),
        )

        # The control agent passes through the same critic as the ReAct loop,
        # so a comparison is not flattered by skipping verification.
        critic = review(output, tools.cited, as_of, tools.cited_line_items)
        return output.model_copy(
            update={
                "verification_passed": critic.passed,
                "verification_defects": "; ".join(d.detail for d in critic.defects),
            }
        )
