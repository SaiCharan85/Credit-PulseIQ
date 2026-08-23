"""A corporate distress diagnostic: past and present states only.

The screen this serves is a *diagnosis of filings already made*, not a view on
what happens next, and three choices in here enforce that rather than merely
asserting it in a disclaimer.

**Every state is dated.** There is no "current" reading. A filer's latest
visible annual period may be eighteen months before the date being asked
about, and labelling that "current" implies live data nobody has. The vintage
travels with the number.

**Reported facts and calculated indices are separated.** ``total_assets`` is a
figure the company filed. ``altman_z_double_prime`` is arithmetic performed on
several such figures. A reader can check the first against the filing and can
only check the second by trusting the formula, so the two are kept in
different columns rather than mixed into one table of "metrics".

**The historical series is recomputed, not remembered.** Each point on the
timeline is the diagnostic as it would have read on that date, using only
filings public by then. It is not today's model applied to old numbers, which
would import knowledge the period did not have.

No zone here is named for an outcome. "Elevated stress" describes a balance
sheet; "likely to fail" describes a future, and this module has no basis for
the second.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

#: Diagnostic zones, ordered. Named for the condition observed, never for a
#: consequence -- "monitored" is a state a reader can verify, "at risk of
#: bankruptcy" is a claim about the future.
ZONES = (
    ("stable", "Stable", "No threshold breached in the filed figures."),
    ("monitored", "Monitored", "One or more measures near a conventional limit."),
    ("elevated", "Elevated stress", "Several measures past conventional limits."),
    ("severe", "Severe stress", "Core solvency measures breached together."),
    ("insufficient", "Insufficient data", "Too few figures to form a diagnosis."),
)

#: Our five-level signal maps onto the diagnostic zones one-for-one.
SIGNAL_TO_ZONE = {
    "healthy": "stable",
    "watch": "monitored",
    "elevated_risk": "elevated",
    "severe_risk": "severe",
    "insufficient_evidence": "insufficient",
}

#: Figures the company filed. A reader can open the 10-K and check these.
REPORTED = (
    ("total_assets", "Total assets"),
    ("total_liabilities", "Total liabilities"),
    ("current_assets", "Current assets"),
    ("current_liabilities", "Current liabilities"),
    ("revenue", "Revenue"),
    ("net_income", "Net income"),
    ("cash_from_operations", "Cash from operations"),
)

#: Arithmetic performed on the above. Checkable only by trusting the formula,
#: so shown separately.
CALCULATED = (
    ("current_ratio", "Current ratio", "1.00"),
    ("quick_ratio", "Quick ratio", "1.00"),
    ("liabilities_to_assets", "Liabilities to assets", "0.70"),
    ("debt_to_assets", "Debt to assets", "0.60"),
    ("interest_coverage", "Interest coverage", "1.50"),
    ("ocf_to_debt", "Operating cash flow to debt", "0.10"),
    ("net_margin", "Net margin", "0.00"),
    ("altman_z_double_prime", "Altman Z'' score", "1.10"),
)


@dataclass
class Reading:
    """One figure, with whether it was filed or derived."""

    key: str
    label: str
    value: float | None
    computable: bool
    threshold: str = ""
    breached: bool = False


@dataclass
class TimelinePoint:
    """The diagnosis as it would have read on one past date."""

    as_of: date
    period_end: date | None
    zone: str
    breached: int
    computable: int


@dataclass
class Diagnostic:
    cik: int
    name: str = ""
    ticker: str = ""
    exchange: str = ""
    sic: str = ""
    sic_description: str = ""
    state: str = ""
    as_of: date | None = None
    period_end: date | None = None
    filing_age_days: int | None = None
    zone: str = "insufficient"
    reported: list[Reading] = field(default_factory=list)
    calculated: list[Reading] = field(default_factory=list)
    timeline: list[TimelinePoint] = field(default_factory=list)

    @property
    def zone_label(self) -> str:
        return next((lbl for k, lbl, _ in ZONES if k == self.zone), self.zone)

    @property
    def stale(self) -> bool:
        """Whether the newest usable filing is old enough to say so."""
        return self.filing_age_days is not None and self.filing_age_days > 456


def _breaches(key: str, value: float) -> bool:
    """Whether a calculated figure sits past its conventional limit.

    Conventional levels from credit practice, not fitted to our outcomes --
    fitting them would make this a model of our labels rather than a reading
    of the balance sheet.
    """
    limits = {
        "current_ratio": lambda v: v < 1.0,
        "quick_ratio": lambda v: v < 1.0,
        "liabilities_to_assets": lambda v: v > 0.7,
        "debt_to_assets": lambda v: v > 0.6,
        "interest_coverage": lambda v: v < 1.5,
        "ocf_to_debt": lambda v: v < 0.1,
        "net_margin": lambda v: v < 0.0,
        "altman_z_double_prime": lambda v: v < 1.1,
    }
    test = limits.get(key)
    return bool(test and test(value))


def build(
    cik: int,
    as_of: date,
    facts: Any,
    submissions: dict[str, Any] | None = None,
    history: int = 6,
) -> Diagnostic:
    """Assemble the diagnostic for one filer at one date."""
    from compute import lineitems
    from compute.lineitems import FactIndex, annual_period_ends
    from compute.ratios import compute_metric
    from data.facts import as_of_view

    sub = submissions or {}
    out = Diagnostic(
        cik=cik,
        name=str(sub.get("name") or ""),
        ticker=(sub.get("tickers") or [""])[0] if sub.get("tickers") else "",
        exchange=(sub.get("exchanges") or [""])[0] if sub.get("exchanges") else "",
        sic=str(sub.get("sic") or ""),
        sic_description=str(sub.get("sicDescription") or ""),
        state=str(sub.get("stateOfIncorporation") or ""),
        as_of=as_of,
    )

    view = FactIndex(as_of_view(facts, as_of))
    ends = annual_period_ends(view)
    if not ends:
        return out
    latest = ends[0]
    out.period_end = latest
    out.filing_age_days = (as_of - latest).days

    for key, label in REPORTED:
        # resolve() returns the fact reference the filer actually filed --
        # a value a reader can find in the document, unlike a computed ratio.
        ref = lineitems.resolve(key, view, latest)
        out.reported.append(
            Reading(key, label, float(ref.value) if ref else None, ref is not None)
        )

    breached = 0
    for key, label, limit in CALCULATED:
        cv = compute_metric(key, view, latest)
        ok = cv.is_defined
        value = float(cv.value) if ok else None
        hit = bool(ok and _breaches(key, value))
        breached += hit
        out.calculated.append(Reading(key, label, value, ok, limit, hit))

    computable = sum(1 for r in out.calculated if r.computable)
    out.zone = _zone_from(breached, computable)

    # Recompute the diagnosis at each earlier period end, using only what was
    # public by then. Not today's reading applied backwards.
    for end in ends[1 : history + 1]:
        past_view = FactIndex(as_of_view(facts, end))
        past_ends = annual_period_ends(past_view)
        if not past_ends:
            continue
        pb = pc = 0
        for key, _lbl, _lim in CALCULATED:
            cv = compute_metric(key, past_view, past_ends[0])
            if cv.is_defined:
                pc += 1
                pb += _breaches(key, float(cv.value))
        out.timeline.append(
            TimelinePoint(end, past_ends[0], _zone_from(pb, pc), pb, pc)
        )
    out.timeline.reverse()
    return out


def _zone_from(breached: int, computable: int) -> str:
    """Zone from how many conventional limits the filed figures breach.

    Counting breaches rather than fitting a score keeps this a reading of the
    balance sheet. A fitted threshold would encode our own outcome labels and
    make the gauge a quiet prediction.
    """
    if computable < 3:
        return "insufficient"
    share = breached / computable
    if breached == 0:
        return "stable"
    if share < 0.25:
        return "monitored"
    if share < 0.5:
        return "elevated"
    return "severe"
