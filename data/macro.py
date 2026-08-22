"""Macroeconomic covariates from FRED, with point-in-time discipline.

Corporate default is strongly cyclical. Firm ratios describe a company; they
say nothing about whether credit is cheap or the cycle is turning, and the
default literature (Duffie, Saita and Wang 2007) finds macro covariates add
real explanatory power over firm-level variables alone. Our hazard baseline
uses firm ratios only, so this is the obvious gap.

**Only non-revised series are used, and that is the whole design.** FRED
publishes two kinds of data. Market prices -- credit spreads, the term
structure, implied volatility -- are observed and never revised: the value
printed for 3 June 2024 is the value a reader saw on 3 June 2024, forever.
Survey and accounting aggregates -- unemployment, GDP, industrial production --
are *revised*, sometimes substantially, for months afterwards. Pulling today's
UNRATE series and reading off a 2019 value hands the model a number nobody had
in 2019, which is lookahead of exactly the kind this project spends its effort
preventing, and it would be invisible because the date column looks correct.

Point-in-time unemployment is obtainable from ALFRED's vintage archive, and if
this leg is ever extended that is the route. Until then the series here are
restricted to market data, where the distinction does not arise.

The as-of rule matches the rest of the system: a value counts only if its
observation date is on or before the prediction date. Markets publish
same-day, so same-day values are legitimately visible -- unlike a filing,
which becomes public when it is filed rather than when the period ends.
"""

from __future__ import annotations

import csv
import io
import json
import urllib.parse
import urllib.request
from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"

#: Market-priced series only. Each is observed, never revised.
SERIES: dict[str, str] = {
    #: ICE BofA US High Yield option-adjusted spread. The single most direct
    #: market read on corporate credit risk -- what lenders charge weak
    #: borrowers, in real time.
    "credit_spread": "BAMLH0A0HYM2",
    #: 10-year minus 3-month Treasury. Inverts ahead of recessions and is the
    #: classic cycle-turn indicator.
    "term_spread": "T10Y3M",
    #: CBOE implied volatility -- risk appetite, priced.
    "vix": "VIXCLS",
    #: Investment-grade spread. Carries the same signal as high yield at a
    #: different point in the credit stack, and moves earlier for large filers.
    "ig_spread": "BAMLC0A0CM",
}

#: Lookbacks for change features. A spread's *level* says where the cycle is;
#: its *change* says which way it is moving, and the two are different signals.
CHANGE_WINDOWS = (90, 365)

#: How stale a macro reading may be before it is treated as absent. Markets
#: close for holidays and the series carry gaps, but a month-old "current"
#: spread is not a current spread.
MAX_STALENESS_DAYS = 21


@dataclass
class MacroSeries:
    """One FRED series as a sorted, queryable history."""

    name: str
    dates: list[date]
    values: list[float]

    def as_of(self, when: date, max_staleness_days: int = MAX_STALENESS_DAYS) -> float | None:
        """Most recent observation on or before ``when``, or None if stale.

        The bisect is over observation dates, so nothing after ``when`` is
        reachable -- lookahead is prevented by construction rather than by a
        filter that could be forgotten.
        """
        if not self.dates:
            return None
        idx = bisect_right(self.dates, when) - 1
        if idx < 0:
            return None
        if (when - self.dates[idx]).days > max_staleness_days:
            return None
        return self.values[idx]

    def change(self, when: date, days: int, **kw) -> float | None:
        """Change in the level over the trailing window."""
        now = self.as_of(when, **kw)
        then = self.as_of(when - timedelta(days=days), **kw)
        if now is None or then is None:
            return None
        return now - then


def parse_fred_csv(text: str) -> MacroSeries:
    """FRED CSV to a series. Missing observations are '.' and are dropped."""
    reader = csv.reader(io.StringIO(text))
    header = next(reader, None)
    name = header[1] if header and len(header) > 1 else "series"
    dates: list[date] = []
    values: list[float] = []
    for row in reader:
        if len(row) < 2:
            continue
        try:
            when = date.fromisoformat(row[0].strip())
            value = float(row[1].strip())
        except ValueError:
            continue  # '.' marks a non-trading day
        dates.append(when)
        values.append(value)
    order = sorted(range(len(dates)), key=lambda i: dates[i])
    return MacroSeries(name, [dates[i] for i in order], [values[i] for i in order])


def fetch_series(
    series_id: str,
    start: date,
    end: date,
    cache_dir: Path | str = "data/cache/macro",
    fetch=None,
) -> MacroSeries:
    """One FRED series, cached on disk. No API key is required for this endpoint."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{series_id}_{start.isoformat()}_{end.isoformat()}.csv"
    if path.exists():
        return parse_fred_csv(path.read_text(encoding="utf8"))

    url = f"{FRED_CSV}?id={series_id}&cosd={start.isoformat()}&coed={end.isoformat()}"
    if fetch is None:
        # A decade of daily observations is a slow render on FRED's side; the
        # default 45s was not enough and the failure surfaced as an empty
        # series rather than an error.
        request = urllib.request.Request(url, headers={"User-Agent": "CreditPulse IQ research"})
        last: Exception | None = None
        for _attempt in range(2):
            try:
                with urllib.request.urlopen(request, timeout=90) as response:
                    text = response.read().decode("utf8", errors="replace")
                break
            except Exception as exc:  # noqa: BLE001
                last = exc
        else:
            raise RuntimeError(f"FRED fetch failed for {series_id}: {last}") from last
    else:
        text = fetch(url)
    path.write_text(text, encoding="utf8")
    return parse_fred_csv(text)


# ---------------------------------------------------------------------------
# Alternate providers
# ---------------------------------------------------------------------------
#
# FRED became unreachable mid-project -- every endpoint, every range, 40s
# timeouts, while EDGAR continued to serve thousands of requests. Rather than
# park the whole leg on one host, two of the four series are available
# elsewhere. Both replacements are market data, so the non-revised property
# that makes point-in-time lookup safe still holds.
#
# The high-yield credit spread has no free non-FRED source and stays blocked.
# It is the most informative of the four, so this is a real reduction in the
# hypothesis being tested, not a like-for-like substitution.

TREASURY_CSV = (
    "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
    "daily-treasury-rates.csv/{year}/all?type=daily_treasury_yield_curve&"
    "field_tdr_date_value={year}&page&_format=csv"
)
YAHOO_CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range={range}&interval=1d"

_BROWSER_UA = {"User-Agent": "Mozilla/5.0 (CreditPulse IQ research)"}


def _get(url: str, timeout: int = 40) -> str:
    request = urllib.request.Request(url, headers=_BROWSER_UA)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf8", errors="replace")


def fetch_term_spread(
    years: Sequence[int], cache_dir: Path | str = "data/cache/macro"
) -> MacroSeries:
    """10-year minus 3-month Treasury, from the Treasury's own daily curve."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    dates: list[date] = []
    values: list[float] = []
    for year in years:
        path = cache_dir / f"treasury_{year}.csv"
        if path.exists():
            text = path.read_text(encoding="utf8")
        else:
            text = _get(TREASURY_CSV.format(year=year))
            path.write_text(text, encoding="utf8")
        reader = csv.DictReader(io.StringIO(text))
        for row in reader:
            try:
                when = datetime.strptime(row["Date"].strip(), "%m/%d/%Y").date()
                ten = float(row["10 Yr"])
                three = float(row["3 Mo"])
            except (KeyError, ValueError, TypeError):
                continue
            dates.append(when)
            values.append(ten - three)
    order = sorted(range(len(dates)), key=lambda i: dates[i])
    return MacroSeries("term_spread", [dates[i] for i in order], [values[i] for i in order])


def fetch_yahoo(
    symbol: str, name: str, span: str = "10y", cache_dir: Path | str = "data/cache/macro"
) -> MacroSeries:
    """Daily closes for a market index. Prices are observed, never revised."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"yahoo_{name}_{span}.json"
    text = path.read_text(encoding="utf8") if path.exists() else _get(
        YAHOO_CHART.format(symbol=urllib.parse.quote(symbol), range=span)
    )
    if not path.exists():
        path.write_text(text, encoding="utf8")
    payload = json.loads(text)
    result = payload["chart"]["result"][0]
    stamps = result["timestamp"]
    closes = result["indicators"]["quote"][0]["close"]
    dates: list[date] = []
    values: list[float] = []
    for ts, close in zip(stamps, closes, strict=False):
        if close is None:
            continue
        dates.append(datetime.utcfromtimestamp(ts).date())
        values.append(float(close))
    return MacroSeries(name, dates, values)


def load_macro_available(
    years: Sequence[int] = tuple(range(2018, 2027)),
    cache_dir: Path | str = "data/cache/macro",
) -> dict[str, MacroSeries]:
    """Whatever can actually be fetched today, with failures reported."""
    out: dict[str, MacroSeries] = {}
    failed: list[str] = []
    try:
        out["term_spread"] = fetch_term_spread(years, cache_dir)
    except Exception as exc:  # noqa: BLE001
        failed.append(f"term_spread (Treasury): {exc}")
    try:
        out["vix"] = fetch_yahoo("^VIX", "vix", cache_dir=cache_dir)
    except Exception as exc:  # noqa: BLE001
        failed.append(f"vix (Yahoo): {exc}")
    for name in ("credit_spread", "ig_spread"):
        try:
            out[name] = fetch_series(SERIES[name], date(2018, 1, 1), date(2026, 8, 31), cache_dir)
        except Exception as exc:  # noqa: BLE001
            failed.append(f"{name} (FRED): {str(exc)[:60]}")
    if failed:
        import sys

        print("macro series unavailable: " + "; ".join(failed), file=sys.stderr)
    return out


def load_macro(
    start: date = date(2015, 1, 1),
    end: date = date(2026, 12, 31),
    cache_dir: Path | str = "data/cache/macro",
    fetch=None,
) -> dict[str, MacroSeries]:
    """Every configured series, keyed by our name rather than the FRED id."""
    out: dict[str, MacroSeries] = {}
    failed: list[str] = []
    for name, series_id in SERIES.items():
        try:
            out[name] = fetch_series(series_id, start, end, cache_dir, fetch)
        except Exception as exc:  # noqa: BLE001
            # Recorded, not swallowed. An earlier version returned an empty
            # series here and the caller saw four covariates quietly reading
            # zero -- the exact silent-failure-on-missing-data mode this
            # project exists to catch, committed in its own codebase.
            failed.append(f"{name} ({series_id}): {exc}")
            out[name] = MacroSeries(series_id, [], [])
    if failed:
        import sys

        print("macro series unavailable: " + "; ".join(failed), file=sys.stderr)
    return out


def macro_feature_names() -> list[str]:
    """Level and change features, each followed by its missingness indicator."""
    names: list[str] = []
    for base in SERIES:
        names += [f"macro_{base}", f"macro_{base}__missing"]
        for window in CHANGE_WINDOWS:
            names += [f"macro_{base}_chg{window}", f"macro_{base}_chg{window}__missing"]
    return names


def macro_features(series: dict[str, MacroSeries], when: date) -> dict[str, float]:
    """Every macro covariate as of one date, with missingness flags.

    A stale or absent reading becomes 0.0 with its flag set, matching how the
    panel treats every other uncomputable metric -- absence is carried as a
    feature rather than imputed away.
    """
    out: dict[str, float] = {}
    for base, s in series.items():
        level = s.as_of(when)
        out[f"macro_{base}"] = float(level) if level is not None else 0.0
        out[f"macro_{base}__missing"] = 0.0 if level is not None else 1.0
        for window in CHANGE_WINDOWS:
            delta = s.change(when, window)
            out[f"macro_{base}_chg{window}"] = float(delta) if delta is not None else 0.0
            out[f"macro_{base}_chg{window}__missing"] = 0.0 if delta is not None else 1.0
    return out
