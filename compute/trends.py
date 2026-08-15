"""Deterministic trends over computed metrics (SPEC 3).

A trend is a sequence of :class:`ComputedValue`, so provenance survives
aggregation: every point in a five-year leverage trend still names the filing
it came from.

Two rules the investigator depends on:

* Periods are annual only. Mixing a quarter into an annual series manufactures
  a ~75% "collapse" every fourth point.
* An undefined point is not skipped, it is *recorded*. A coverage ratio that
  becomes undefined because interest expense stopped being tagged is a data
  problem the agent must see, not a gap to interpolate over.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

from pydantic import BaseModel, ConfigDict, Field

from compute.provenance import ComputedValue
from compute.ratios import compute_metric
from data.facts import XbrlFact

DIRECTION_IMPROVING = "improving"
DIRECTION_DETERIORATING = "deteriorating"
DIRECTION_FLAT = "flat"
DIRECTION_UNKNOWN = "unknown"

#: Metrics where a falling value is the deteriorating direction. For leverage
#: metrics the sign is inverted -- rising debt/assets is deterioration.
HIGHER_IS_BETTER = frozenset(
    {
        "interest_coverage",
        "ebitda_interest_coverage",
        "current_ratio",
        "quick_ratio",
        "cash_ratio",
        "working_capital",
        "gross_margin",
        "operating_margin",
        "net_margin",
        "return_on_assets",
        "free_cash_flow",
        "ocf_to_debt",
        "ebitda",
        "altman_z_double_prime",
    }
)

FLAT_TOLERANCE = 1e-9


class Trend(BaseModel):
    """An ordered metric series with deterministic summary statistics."""

    model_config = ConfigDict(frozen=True)

    metric: str
    points: list[ComputedValue] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @property
    def defined_points(self) -> list[ComputedValue]:
        return [p for p in self.points if p.is_defined]

    @property
    def period_ends(self) -> list[date]:
        return [p.period_end for p in self.points]

    @property
    def values(self) -> list[float]:
        return [p.value for p in self.defined_points]  # type: ignore[misc]

    @property
    def as_of(self) -> date | None:
        """Latest filing date across every input of every point."""
        stamps = [p.as_of for p in self.points if p.as_of is not None]
        return max(stamps) if stamps else None

    @property
    def coverage(self) -> float:
        """Fraction of requested periods that actually computed."""
        return len(self.defined_points) / len(self.points) if self.points else 0.0

    @property
    def latest(self) -> ComputedValue | None:
        return self.points[-1] if self.points else None

    @property
    def change_abs(self) -> float | None:
        v = self.values
        return v[-1] - v[0] if len(v) >= 2 else None

    @property
    def change_pct(self) -> float | None:
        v = self.values
        if len(v) < 2 or abs(v[0]) < 1e-12:
            return None
        return (v[-1] - v[0]) / abs(v[0])

    @property
    def direction(self) -> str:
        """Deterioration/improvement, sign-corrected per metric.

        Uses first-to-last change, not the slope, so it answers "worse than
        when we started?" rather than "trending down on average?" -- the
        latter can read as improving while the latest value is the worst.
        """
        change = self.change_abs
        if change is None:
            return DIRECTION_UNKNOWN
        if abs(change) <= FLAT_TOLERANCE:
            return DIRECTION_FLAT
        better_when_up = self.metric in HIGHER_IS_BETTER
        improving = change > 0 if better_when_up else change < 0
        return DIRECTION_IMPROVING if improving else DIRECTION_DETERIORATING

    @property
    def consecutive_deteriorations(self) -> int:
        """Length of the current run of period-on-period worsening.

        Counted from the most recent end. A three-year unbroken slide is a
        different signal from three bad years with a recovery in between.
        """
        v = self.values
        if len(v) < 2:
            return 0
        better_when_up = self.metric in HIGHER_IS_BETTER
        run = 0
        for prev, cur in zip(reversed(v[:-1]), reversed(v[1:]), strict=True):
            delta = cur - prev
            worse = delta < 0 if better_when_up else delta > 0
            if worse and abs(delta) > FLAT_TOLERANCE:
                run += 1
            else:
                break
        return run

    @property
    def slope_per_period(self) -> float | None:
        """Ordinary least-squares slope over period index.

        Index-based rather than date-based: fiscal years are near-uniform, and
        using the index keeps the statistic stable when a filer shifts its year
        end by a few days.
        """
        v = self.values
        n = len(v)
        if n < 2:
            return None
        mean_x = (n - 1) / 2
        mean_y = sum(v) / n
        num = sum((i - mean_x) * (y - mean_y) for i, y in enumerate(v))
        den = sum((i - mean_x) ** 2 for i in range(n))
        if abs(den) < 1e-12:
            return None
        return num / den

    def summary(self) -> dict[str, object]:
        return {
            "metric": self.metric,
            "periods": [d.isoformat() for d in self.period_ends],
            "values": self.values,
            "direction": self.direction,
            "change_abs": self.change_abs,
            "change_pct": self.change_pct,
            "consecutive_deteriorations": self.consecutive_deteriorations,
            "slope_per_period": self.slope_per_period,
            "coverage": self.coverage,
            "as_of": self.as_of.isoformat() if self.as_of else None,
        }


def build_trend(
    metric: str, facts: Sequence[XbrlFact], period_ends: Sequence[date]
) -> Trend:
    """Compute ``metric`` across ``period_ends``, oldest first.

    ``facts`` must already be an as-of view; this function adds no date
    filtering of its own (the lookahead control lives in one place).
    """
    ordered = sorted(period_ends)
    points = [compute_metric(metric, facts, pe) for pe in ordered]
    notes: list[str] = []
    undefined_periods = [p.period_end.isoformat() for p in points if not p.is_defined]
    if undefined_periods:
        notes.append(f"undefined in {len(undefined_periods)} period(s): {', '.join(undefined_periods)}")
    return Trend(metric=metric, points=points, notes=notes)


def build_trends(
    metrics: Sequence[str], facts: Sequence[XbrlFact], period_ends: Sequence[date]
) -> dict[str, Trend]:
    return {m: build_trend(m, facts, period_ends) for m in metrics}
