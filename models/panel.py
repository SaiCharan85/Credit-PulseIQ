"""Firm-period panel construction for the hazard baseline (SPEC 7).

One row per firm per observation date. Features are computed strictly from
filings public *before* that date; the label is whether a terminal event
follows within the horizon.

This shape is required by Shumway (2001), whose central result is that static
single-period bankruptcy models -- Altman, Ohlson -- are biased and
inconsistent because they take one observation per firm and ignore that firms
deteriorate over time. A firm contributes a row for every period it survives
and then exits the panel, which is also what makes the survivors *censored*
rather than proven negatives.

Three controls are load-bearing:

**As-of features.** Every row's features come from ``as_of_view(facts, date)``,
so nothing filed on or after the observation date can enter. The panel is the
one place where a leak would be invisible and fatal.

**Post-event rows are dropped.** A firm that filed Chapter 11 in March 2023
contributes no rows after March 2023; predicting a bankruptcy that already
happened is not prediction.

**Missingness is a feature, not a gap.** Our data is MNAR -- values go missing
*because* of distress (a distressed filer reclassifies its debt, or stops
filing altogether). Imputing them would erase the signal, so each metric
carries a companion ``<metric>__missing`` indicator.

Deterministic plain code, no LLM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

from compute.lineitems import FactIndex, annual_period_ends
from compute.ratios import compute_metric
from compute.scores import compute_two_period_score
from data.distress_events import SEVERITY, TIER_DEFAULT, DistressEvent, worst_tier_within
from data.edgar import EdgarClient
from data.facts import as_of_view

#: Single-period metrics used as hazard covariates. Chosen to span the classic
#: dimensions -- leverage, liquidity, profitability, coverage, cash flow -- and
#: to overlap the Altman/Ohlson inputs so the baseline is comparable.
FEATURES: tuple[str, ...] = (
    "debt_to_assets",
    "liabilities_to_assets",
    "current_ratio",
    "quick_ratio",
    "cash_ratio",
    "interest_coverage",
    "operating_margin",
    "net_margin",
    "return_on_assets",
    "ocf_to_debt",
    "accruals_to_assets",
    "altman_z_double_prime",
)

#: Year-on-year scores, computed only where a prior year is visible.
TWO_PERIOD_FEATURES: tuple[str, ...] = ("ohlson_o_score", "piotroski_f_score")

MISSING_SUFFIX = "__missing"

DEFAULT_HORIZON_DAYS = 365

#: Winsorisation bound. Financial ratios have unbounded tails (a near-zero
#: denominator produces values in the thousands) which would dominate a linear
#: model. Clipping is stated rather than hidden because it changes coefficients.
CLIP_ABS = 25.0


@dataclass
class PanelRow:
    """One firm-period observation."""

    cik: int
    observation_date: date
    features: dict[str, float] = field(default_factory=dict)
    missing: dict[str, bool] = field(default_factory=dict)
    label: int = 0
    event_date: date | None = None
    days_to_event: int | None = None
    latest_period_end: date | None = None

    def vector(self, names: list[str]) -> list[float]:
        return [self.features.get(n, 0.0) for n in names]


def _clip(value: float) -> float:
    return max(-CLIP_ABS, min(CLIP_ABS, value))


def observation_dates(start: date, end: date, months: int = 3) -> list[date]:
    """Regular observation grid. Quarterly by default."""
    out: list[date] = []
    cur = start
    while cur <= end:
        out.append(cur)
        year = cur.year + (cur.month - 1 + months) // 12
        month = (cur.month - 1 + months) % 12 + 1
        day = min(cur.day, 28)
        cur = date(year, month, day)
    return out


def build_firm_rows(
    facts,
    cik: int,
    events: list[DistressEvent],
    dates: list[date],
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    min_history_days: int = 400,
    features: list[str] | None = None,
    two_period: list[str] | None = None,
) -> list[PanelRow]:
    """Rows for one firm across the observation grid.

    ``facts`` is the firm's full fact set; each row re-filters it to its own
    as-of view rather than trusting a pre-filtered input.

    ``features``/``two_period`` default to the distress covariates. The
    earnings-quality leg passes its own set -- notably Beneish M -- rather than
    widening these, because adding a covariate here would silently change the
    fitted hazard baseline the distress leg has already been measured against.
    """
    features = FEATURES if features is None else features
    two_period = TWO_PERIOD_FEATURES if two_period is None else two_period
    terminal = [e for e in events if e.cik == cik and e.tier == TIER_DEFAULT]
    event_date = min((e.event_date for e in terminal), default=None)

    rows: list[PanelRow] = []
    for obs in dates:
        # A firm exits the panel once it has filed. Predicting an event that
        # already happened is not prediction.
        if event_date is not None and obs > event_date:
            continue

        view = FactIndex(as_of_view(facts, obs))
        if not view:
            continue
        period_ends = annual_period_ends(view)
        if not period_ends:
            continue
        latest = period_ends[0]
        # Require reasonably fresh data; a filer years stale has no signal to
        # read, and its silence is captured by the ladder instead.
        if (obs - latest).days > min_history_days + 365:
            continue

        prior = period_ends[1] if len(period_ends) > 1 else None

        row = PanelRow(cik=cik, observation_date=obs, latest_period_end=latest)
        for name in features:
            cv = compute_metric(name, view, latest)
            if cv.is_defined:
                row.features[name] = _clip(float(cv.value))
                row.missing[name] = False
            else:
                row.features[name] = 0.0
                row.missing[name] = True
            row.features[f"{name}{MISSING_SUFFIX}"] = 1.0 if row.missing[name] else 0.0

        for name in two_period:
            cv = (
                compute_two_period_score(name, view, latest, prior)
                if prior is not None
                else None
            )
            if cv is not None and cv.is_defined:
                row.features[name] = _clip(float(cv.value))
                row.missing[name] = False
            else:
                row.features[name] = 0.0
                row.missing[name] = True
            row.features[f"{name}{MISSING_SUFFIX}"] = 1.0 if row.missing[name] else 0.0

        hit = worst_tier_within(
            events,
            cik,
            obs,
            obs + timedelta(days=horizon_days),
            min_severity=SEVERITY[TIER_DEFAULT],
        )
        row.label = 1 if hit else 0
        if hit:
            row.event_date = hit.event_date
            row.days_to_event = (hit.event_date - obs).days
        rows.append(row)
    return rows


def feature_names(
    features: list[str] | None = None, two_period: list[str] | None = None
) -> list[str]:
    """Covariate order: every metric followed by its missingness indicator."""
    features = FEATURES if features is None else features
    two_period = TWO_PERIOD_FEATURES if two_period is None else two_period
    names: list[str] = []
    for base in features + two_period:
        names.append(base)
        names.append(f"{base}{MISSING_SUFFIX}")
    return names


def build_panel(
    client: EdgarClient,
    ciks: list[int],
    events: list[DistressEvent],
    dates: list[date],
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    verbose: bool = True,
) -> list[PanelRow]:
    rows: list[PanelRow] = []
    for n, cik in enumerate(ciks, 1):
        try:
            facts = client.facts(cik)
        except Exception:  # noqa: BLE001 - an unavailable filer is skipped, not fatal
            continue
        rows.extend(build_firm_rows(facts, cik, events, dates, horizon_days))
        if verbose and n % 25 == 0:
            import sys

            print(f"  {n}/{len(ciks)} filers, {len(rows)} rows", file=sys.stderr)
    return rows


def save_panel(rows: list[PanelRow], path, names: list[str] | None = None) -> None:
    """Persist the panel so evaluation can iterate without refetching.

    ``names`` must match the feature set the rows were built with. Defaulting
    to the distress covariates silently dropped the earnings-quality columns
    on write -- the file loaded back clean, with Beneish M simply absent.
    """
    import csv
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = feature_names() if names is None else list(names)
    with path.open("w", encoding="utf8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["cik", "observation_date", "label", "days_to_event", "latest_period_end"] + names)
        for r in rows:
            writer.writerow(
                [
                    r.cik,
                    r.observation_date.isoformat(),
                    r.label,
                    r.days_to_event if r.days_to_event is not None else "",
                    r.latest_period_end.isoformat() if r.latest_period_end else "",
                ]
                + [r.features.get(n, 0.0) for n in names]
            )


def load_panel(path, names: list[str] | None = None) -> list[PanelRow]:
    import csv
    from pathlib import Path

    path = Path(path)
    if not path.exists():
        return []
    names = feature_names() if names is None else list(names)
    rows: list[PanelRow] = []
    with path.open(encoding="utf8", newline="") as fh:
        for rec in csv.DictReader(fh):
            row = PanelRow(
                cik=int(rec["cik"]),
                observation_date=date.fromisoformat(rec["observation_date"]),
                label=int(rec["label"]),
                days_to_event=int(rec["days_to_event"]) if rec["days_to_event"] else None,
                latest_period_end=date.fromisoformat(rec["latest_period_end"])
                if rec["latest_period_end"]
                else None,
            )
            for n in names:
                row.features[n] = float(rec[n])
            for n in names:
                if n.endswith(MISSING_SUFFIX):
                    row.missing[n[: -len(MISSING_SUFFIX)]] = row.features[n] == 1.0
            rows.append(row)
    return rows


def split_by_date(rows: list[PanelRow], cutoff: date) -> tuple[list[PanelRow], list[PanelRow]]:
    """Temporal split: train strictly before ``cutoff``, test on or after.

    A random split would leak, because the same firm's adjacent quarters are
    nearly identical and would land on both sides.
    """
    train = [r for r in rows if r.observation_date < cutoff]
    test = [r for r in rows if r.observation_date >= cutoff]
    return train, test
