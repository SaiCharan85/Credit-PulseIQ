"""Typed tool contracts for the investigators (SPEC 5).

The agent gets tools, never the raw database, and never arithmetic. Every tool
returns a structured, provenanced result computed by ``compute/``.

Two properties of this design are load-bearing:

**The as-of date is not a tool argument.** A :class:`ToolBox` is constructed for
one filer at one prediction date, and every call is filtered to that date
automatically. If the agent could pass an as-of date it could ask for the
future, and the cardinal sin would be one hallucinated parameter away. The
model cannot request data it should not see because there is no argument for it.

**Every call is recorded.** The audit trail underpins both the cited memo
(SPEC 6) and the agent-trajectory eval (SPEC 7): did it call the right tools, in
a sensible order, and stop when it should?

Plain deterministic code -- the tools are not agents (PROMPT hard rule 3).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from compute import lineitems
from compute.lineitems import FactIndex, annual_period_ends
from compute.peers import PeerComparison, build_peer_group, compare_to_peers
from compute.provenance import FORMULAS, ComputedValue, undefined
from compute.ratios import STANDARD_METRICS, compute_metric
from compute.scores import ALL_SCORES, TWO_PERIOD_SCORES, compute_two_period_score
from compute.trends import Trend, build_trend
from data.distress_events import DistressEvent, events_visible_as_of
from data.facts import as_of_view

#: Reference thresholds for ``check_threshold``. Conventional credit-analysis
#: levels, not fitted parameters -- they exist so the agent compares against a
#: fixed rule rather than inventing a cutoff mid-reasoning. ``worse`` records
#: which direction is bad.
THRESHOLDS: dict[str, dict[str, Any]] = {
    "debt_to_assets": {"flag": 0.60, "severe": 0.80, "worse": "higher"},
    "liabilities_to_assets": {"flag": 0.90, "severe": 1.00, "worse": "higher"},
    "debt_to_ebitda": {"flag": 4.0, "severe": 6.0, "worse": "higher"},
    "interest_coverage": {"flag": 2.0, "severe": 1.0, "worse": "lower"},
    "ebitda_interest_coverage": {"flag": 3.0, "severe": 1.5, "worse": "lower"},
    "current_ratio": {"flag": 1.2, "severe": 1.0, "worse": "lower"},
    "quick_ratio": {"flag": 0.8, "severe": 0.5, "worse": "lower"},
    "cash_ratio": {"flag": 0.3, "severe": 0.1, "worse": "lower"},
    "operating_margin": {"flag": 0.02, "severe": 0.0, "worse": "lower"},
    "net_margin": {"flag": 0.0, "severe": -0.10, "worse": "lower"},
    "return_on_assets": {"flag": 0.0, "severe": -0.05, "worse": "lower"},
    "ocf_to_debt": {"flag": 0.10, "severe": 0.0, "worse": "lower"},
    "altman_z_double_prime": {"flag": 2.6, "severe": 1.1, "worse": "lower"},
    "ohlson_o_score": {"flag": 0.5, "severe": 2.0, "worse": "higher"},
    "piotroski_f_score": {"flag": 4.0, "severe": 2.0, "worse": "lower"},
    "cash_runway_months": {"flag": 18.0, "severe": 6.0, "worse": "lower"},
    "accruals_to_assets": {"flag": 0.05, "severe": 0.10, "worse": "higher"},
}

STATUS_OK = "ok"
STATUS_FLAG = "flag"
STATUS_SEVERE = "severe"
STATUS_UNKNOWN = "unknown"


class ToolCall(BaseModel):
    """One recorded tool invocation, for the audit trail."""

    model_config = ConfigDict(frozen=True)

    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    ok: bool = True
    error: str = ""
    summary: str = ""


class ToolError(BaseModel):
    model_config = ConfigDict(frozen=True)

    tool: str
    error: str

    def as_dict(self) -> dict[str, Any]:
        return {"error": self.error, "tool": self.tool}


class LineItemResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    concept: str
    tag: str
    value: float
    unit: str
    period_end: date
    form: str
    accession: str
    filed: date


class MetricResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    metric: str
    value: float | None
    unit: str
    period_end: date
    defined: bool
    formula: str
    citations: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ThresholdResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    metric: str
    value: float | None
    status: str
    flag_at: float | None = None
    severe_at: float | None = None
    worse_when: str = ""
    note: str = ""


class ToolBox:
    """Tools bound to one filer at one prediction date.

    Construction fixes ``cik`` and ``as_of``; the agent cannot vary either.
    """

    def __init__(
        self,
        cik: int,
        as_of: date,
        facts: Sequence[Any],
        peer_facts: dict[int, Sequence[Any]] | None = None,
        sic_by_cik: dict[int, str] | None = None,
        events: Sequence[DistressEvent] = (),
    ) -> None:
        self.cik = cik
        self.as_of = as_of
        self._view = FactIndex(as_of_view(facts, as_of))
        self._peer_facts = peer_facts or {}
        self._sic_by_cik = sic_by_cik or {}
        self._events = list(events)
        self.calls: list[ToolCall] = []
        self.cited: list[ComputedValue] = []
        self._periods = annual_period_ends(self._view)

    # ---- helpers -------------------------------------------------------

    def _record(self, tool: str, arguments: dict, ok: bool, summary: str, error: str = "") -> None:
        self.calls.append(
            ToolCall(tool=tool, arguments=arguments, ok=ok, summary=summary, error=error)
        )

    def _resolve_period(self, period: str | date | None) -> date | None:
        """Accept an ISO date, ``"latest"``, or an offset like ``"latest-1"``."""
        if not self._periods:
            return None
        if period is None or period == "latest":
            return self._periods[0]
        if isinstance(period, date):
            return period if period in self._periods else None
        text = str(period).strip()
        if text.startswith("latest-"):
            try:
                back = int(text.split("-", 1)[1])
            except ValueError:
                return None
            return self._periods[back] if back < len(self._periods) else None
        try:
            parsed = datetime.strptime(text, "%Y-%m-%d").date()
        except ValueError:
            return None
        return parsed if parsed in self._periods else None

    def _no_prior_period(self, metric: str, period_end: date) -> ComputedValue:
        """A year-on-year score with only one visible period.

        Cannot fall through to the single-period path: those formulas declare
        ``<concept>_prior`` inputs, which are not line items, so resolution
        raises. Returning an undefined value keeps the tool contract intact --
        the agent sees "not computable and why", which it can act on, rather
        than an exception it cannot.
        """
        return undefined(
            metric,
            metric,
            period_end,
            "requires a prior fiscal year; only one period is visible as of "
            f"{self.as_of.isoformat()}",
        )

    # ---- the tool surface ----------------------------------------------

    def available_periods(self) -> dict[str, Any]:
        """Fiscal periods visible as of the prediction date."""
        out = {
            "as_of": self.as_of.isoformat(),
            "periods": [d.isoformat() for d in self._periods],
            "count": len(self._periods),
        }
        self._record("available_periods", {}, True, f"{len(self._periods)} periods")
        return out

    def get_line_item(self, concept: str, period: str | None = "latest") -> dict[str, Any]:
        """A raw XBRL value with its provenance."""
        args = {"concept": concept, "period": period}
        if concept not in lineitems.CONCEPTS:
            err = f"unknown concept '{concept}'; known: {', '.join(sorted(lineitems.CONCEPTS))}"
            self._record("get_line_item", args, False, "", err)
            return ToolError(tool="get_line_item", error=err).as_dict()
        pe = self._resolve_period(period)
        if pe is None:
            err = f"no visible period matching '{period}'"
            self._record("get_line_item", args, False, "", err)
            return ToolError(tool="get_line_item", error=err).as_dict()
        ref = lineitems.resolve(concept, self._view, pe)
        if ref is None:
            err = f"{concept} is not tagged in the {pe.isoformat()} filing"
            self._record("get_line_item", args, False, "", err)
            return ToolError(tool="get_line_item", error=err).as_dict()
        result = LineItemResult(
            concept=concept,
            tag=ref.tag,
            value=ref.value,
            unit=ref.unit,
            period_end=ref.period_end,
            form=ref.form,
            accession=ref.accession,
            filed=ref.filed,
        )
        self._record("get_line_item", args, True, f"{concept}={ref.value:,.0f}")
        return result.model_dump(mode="json")

    def get_metric(self, metric: str, period: str | None = "latest") -> dict[str, Any]:
        """A computed ratio or score. The agent's only source of numbers."""
        args = {"metric": metric, "period": period}
        if metric not in FORMULAS:
            err = f"unknown metric '{metric}'; known: {', '.join(sorted(STANDARD_METRICS + ALL_SCORES))}"
            self._record("get_metric", args, False, "", err)
            return ToolError(tool="get_metric", error=err).as_dict()
        pe = self._resolve_period(period)
        if pe is None:
            err = f"no visible period matching '{period}'"
            self._record("get_metric", args, False, "", err)
            return ToolError(tool="get_metric", error=err).as_dict()

        if metric in TWO_PERIOD_SCORES:
            idx = self._periods.index(pe)
            prior = self._periods[idx + 1] if idx + 1 < len(self._periods) else None
            cv = (
                compute_two_period_score(metric, self._view, pe, prior)
                if prior is not None
                else self._no_prior_period(metric, pe)
            )
        else:
            cv = compute_metric(metric, self._view, pe)

        self.cited.append(cv)
        result = MetricResult(
            metric=metric,
            value=cv.value,
            unit=cv.unit,
            period_end=cv.period_end,
            defined=cv.is_defined,
            formula=cv.formula,
            citations=cv.citations,
            notes=cv.notes,
        )
        shown = f"{cv.value:,.4f}" if cv.is_defined else "undefined"
        self._record("get_metric", args, True, f"{metric}={shown}")
        return result.model_dump(mode="json")

    def get_trend(self, metric: str, n_periods: int = 4) -> dict[str, Any]:
        """A metric across the last ``n_periods`` visible fiscal years."""
        args = {"metric": metric, "n_periods": n_periods}
        if metric not in FORMULAS:
            err = f"unknown metric '{metric}'"
            self._record("get_trend", args, False, "", err)
            return ToolError(tool="get_trend", error=err).as_dict()
        n = max(2, min(int(n_periods), len(self._periods)))
        periods = self._periods[:n]
        if len(periods) < 2:
            err = "fewer than two visible periods; no trend can be computed"
            self._record("get_trend", args, False, "", err)
            return ToolError(tool="get_trend", error=err).as_dict()
        if metric in TWO_PERIOD_SCORES:
            # Each point needs its own prior year, so pair consecutive periods
            # and drop the oldest, which has no predecessor.
            pairs = list(zip(periods[:-1], periods[1:], strict=True))
            if not pairs:
                err = "not enough periods for a year-on-year score trend"
                self._record("get_trend", args, False, "", err)
                return ToolError(tool="get_trend", error=err).as_dict()
            points = [
                compute_two_period_score(metric, self._view, current, prior)
                for current, prior in reversed(pairs)
            ]
            trend = Trend(metric=metric, points=points)
        else:
            trend = build_trend(metric, self._view, periods)
        self.cited.extend(trend.points)
        summary = trend.summary()
        self._record("get_trend", args, True, f"{metric} {trend.direction}")
        return summary

    def get_peer_comparison(self, metric: str, period: str | None = "latest") -> dict[str, Any]:
        """Where this filer sits in its peer distribution, same vintage."""
        args = {"metric": metric, "period": period}
        if not self._peer_facts:
            err = "no peer data was loaded for this run"
            self._record("get_peer_comparison", args, False, "", err)
            return ToolError(tool="get_peer_comparison", error=err).as_dict()
        pe = self._resolve_period(period)
        if pe is None:
            err = f"no visible period matching '{period}'"
            self._record("get_peer_comparison", args, False, "", err)
            return ToolError(tool="get_peer_comparison", error=err).as_dict()

        group = build_peer_group(self.cik, self._sic_by_cik)
        target = compute_metric(metric, self._view, pe)
        peer_values: dict[int, ComputedValue] = {}
        for cik in group.members:
            facts = self._peer_facts.get(cik)
            if not facts:
                continue
            view = FactIndex(as_of_view(facts, self.as_of))
            ends = annual_period_ends(view)
            if not ends:
                continue
            peer_values[cik] = compute_metric(metric, view, ends[0])

        comparison: PeerComparison = compare_to_peers(
            metric, target, peer_values, group, as_of=self.as_of
        )
        self.cited.append(target)
        self._record(
            "get_peer_comparison",
            args,
            True,
            f"{metric} pct={comparison.percentile} n={comparison.peer_count}",
        )
        return {
            "metric": metric,
            "value": comparison.value,
            "percentile": comparison.percentile,
            "peer_median": comparison.peer_median,
            "peer_count": comparison.peer_count,
            "peer_group_level": comparison.group.level,
            "usable": comparison.is_usable,
            "notes": comparison.notes,
        }

    def check_threshold(self, metric: str, value: float | None = None) -> dict[str, Any]:
        """Compare a metric against fixed reference levels.

        Fetches the value itself when not supplied, so the agent never has to
        carry a number between calls -- one fewer place for a figure to drift.
        """
        args = {"metric": metric, "value": value}
        rule = THRESHOLDS.get(metric)
        if rule is None:
            err = f"no reference threshold for '{metric}'"
            self._record("check_threshold", args, False, "", err)
            return ToolError(tool="check_threshold", error=err).as_dict()

        if value is None:
            pe = self._resolve_period("latest")
            if pe is None:
                err = "no visible period"
                self._record("check_threshold", args, False, "", err)
                return ToolError(tool="check_threshold", error=err).as_dict()
            if metric in TWO_PERIOD_SCORES:
                cv = (
                    compute_two_period_score(metric, self._view, pe, self._periods[1])
                    if len(self._periods) > 1
                    else self._no_prior_period(metric, pe)
                )
            else:
                cv = compute_metric(metric, self._view, pe)
            self.cited.append(cv)
            value = cv.value

        if value is None:
            result = ThresholdResult(
                metric=metric,
                value=None,
                status=STATUS_UNKNOWN,
                flag_at=rule["flag"],
                severe_at=rule["severe"],
                worse_when=rule["worse"],
                note="metric undefined; cannot be compared",
            )
        else:
            worse_higher = rule["worse"] == "higher"
            if (value >= rule["severe"]) if worse_higher else (value <= rule["severe"]):
                status = STATUS_SEVERE
            elif (value >= rule["flag"]) if worse_higher else (value <= rule["flag"]):
                status = STATUS_FLAG
            else:
                status = STATUS_OK
            result = ThresholdResult(
                metric=metric,
                value=value,
                status=status,
                flag_at=rule["flag"],
                severe_at=rule["severe"],
                worse_when=rule["worse"],
            )
        self._record("check_threshold", args, True, f"{metric} -> {result.status}")
        return result.model_dump(mode="json")

    def get_prior_distress_events(self) -> dict[str, Any]:
        """Distress events already public at the prediction date.

        Filtered through ``events_visible_as_of``: a prior restatement is
        legitimate evidence only if it had already been disclosed.
        """
        visible = events_visible_as_of(
            [e for e in self._events if e.cik == self.cik], self.as_of
        )
        rows = [
            {
                "tier": e.tier,
                "signal": e.signal,
                "date": e.event_date.isoformat(),
                "noisy": e.noisy,
            }
            for e in sorted(visible, key=lambda e: e.event_date, reverse=True)[:20]
        ]
        self._record("get_prior_distress_events", {}, True, f"{len(rows)} prior events")
        return {"count": len(rows), "events": rows}

    # ---- audit ---------------------------------------------------------

    def audit_trail(self) -> list[dict[str, Any]]:
        return [c.model_dump(mode="json") for c in self.calls]

    def call_count(self) -> int:
        return len(self.calls)
