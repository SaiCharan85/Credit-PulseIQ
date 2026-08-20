"""Earnings-quality leg: panel, labels and baselines (SPEC 12, phase 4).

The question is not "will this company fail" but **"are the financial statements
a reader can see right now going to be declared unreliable?"** The label is an
8-K item 4.02 landing within the horizon, and every prediction is made strictly
before that filing.

Three things differ from the distress leg, and each is deliberate:

**A different feature set.** Distress is about solvency; manipulation is about
the gap between accrual earnings and cash. So Beneish M joins the covariates,
along with the working-capital cycle metrics that carry the classic red flags --
receivables growing faster than sales, inventory building, the cash conversion
cycle stretching. The distress covariates are kept too, because a company under
solvency pressure has the motive.

**Beneish M as the Tier 0 baseline.** It is the field's published manipulation
score with fixed 1999 coefficients, fitted to nothing here, which makes it the
honest analogue of Altman Z'' on the distress leg. Its conventional threshold is
-1.78: above that, "likely manipulator".

**A company does not exit the panel after the event.** A bankruptcy is terminal;
a restatement is not. A firm that restates in 2019 and again in 2023 supplies
genuine observations in between, so only the horizon window is labelled positive.

One honest limit stated up front: the universe is the distress watchlist, which
is enriched with troubled filers, and troubled filers restate far more often
than the population. The base rate here is roughly 14% against a population rate
nearer 1-2%. Ranking transfers; absolute probabilities do not.
"""

from __future__ import annotations

from datetime import date, timedelta

from compute.scores import BENEISH_COMPONENTS
from data.distress_events import DistressEvent
from data.restatements import RestatementCandidate
from models.panel import (
    FEATURES,
    TWO_PERIOD_FEATURES,
    PanelRow,
    build_firm_rows,
    feature_names,
)

#: Working-capital red flags on top of the solvency covariates.
EQ_FEATURES: tuple[str, ...] = FEATURES + (
    "days_sales_outstanding",
    "days_inventory_outstanding",
    "cash_conversion_cycle",
)

#: Beneish M needs two periods, like Ohlson and Piotroski -- and so do its
#: components, which are carried individually.
#:
#: The composite alone was computable on 17% of wide-panel rows: it returns
#: nothing unless all eight terms resolve, so one untagged line item discards
#: the other seven signals. Each term needs only its own inputs, and the
#: panel's missingness indicators let a model use whatever a filer did tag.
#: TATA is absent because it is arithmetically identical to
#: ``accruals_to_assets``, already carried at 93% coverage.
EQ_TWO_PERIOD: tuple[str, ...] = (
    TWO_PERIOD_FEATURES + ("beneish_m_score",) + BENEISH_COMPONENTS
)

#: Beneish (1999): above this, the model classes a filer a likely manipulator.
BENEISH_THRESHOLD = -1.78

#: The tier string the panel builder expects. Restatements are not distress
#: events, but the container is generic and reusing it is what makes the
#: harness pay off -- the alternative is a parallel panel builder that would
#: drift out of step with the one already tested.
EQ_TIER = "default"


def restatement_events(
    candidates: list[RestatementCandidate], names: dict[int, str] | None = None
) -> list[DistressEvent]:
    """Kept restatement candidates as dated events the panel builder accepts."""
    names = names or {}
    out = []
    for c in candidates:
        if not c.kept:
            continue
        out.append(
            DistressEvent(
                cik=c.cik,
                tier=EQ_TIER,
                signal="non_reliance_8k",
                event_date=c.filing_date,
                # An 8-K is public the day it is filed, so the two dates
                # coincide. Kept explicit rather than implied.
                as_of_date=c.filing_date,
                source_form="8-K",
                source_accession=c.accession,
            )
        )
    return out


def eq_feature_names() -> list[str]:
    return feature_names(list(EQ_FEATURES), list(EQ_TWO_PERIOD))


def save_eq_panel(rows: list[PanelRow], path) -> None:
    from models.panel import save_panel

    save_panel(rows, path, eq_feature_names())


def load_eq_panel(path) -> list[PanelRow]:
    from models.panel import load_panel

    return load_panel(path, eq_feature_names())


def build_eq_rows(
    facts,
    cik: int,
    events: list[DistressEvent],
    dates: list[date],
    horizon_days: int = 365,
) -> list[PanelRow]:
    """Rows for one filer, labelled by whether a restatement follows.

    ``build_firm_rows`` drops observations after a terminal event, which is
    right for bankruptcy and wrong here. So each restatement is presented only
    to the window it can legitimately label, and the rows are recombined.
    """
    mine = sorted(
        (e for e in events if e.cik == cik), key=lambda e: e.event_date
    )
    if not mine:
        return build_firm_rows(
            facts, cik, [], dates, horizon_days,
            features=list(EQ_FEATURES), two_period=list(EQ_TWO_PERIOD),
        )

    rows: list[PanelRow] = []
    seen: set[date] = set()
    for event in mine:
        window = [
            d
            for d in dates
            if event.event_date - timedelta(days=horizon_days) <= d <= event.event_date
        ]
        for r in build_firm_rows(
            facts, cik, [event], window, horizon_days,
            features=list(EQ_FEATURES), two_period=list(EQ_TWO_PERIOD),
        ):
            if r.observation_date not in seen:
                seen.add(r.observation_date)
                rows.append(r)

    # Everything outside any event window is a genuine negative observation.
    rest = [d for d in dates if d not in seen]
    rows.extend(
        build_firm_rows(
            facts, cik, [], rest, horizon_days,
            features=list(EQ_FEATURES), two_period=list(EQ_TWO_PERIOD),
        )
    )
    rows.sort(key=lambda r: r.observation_date)
    return rows


class BeneishBaseline:
    """Tier 0: the published M-score, fitted to nothing.

    The analogue of Altman Z'' on the distress leg -- a fixed-coefficient
    formula from the literature, so it needs no training data and cannot leak.
    """

    METRIC = "beneish_m_score"
    name = "beneish_m"

    def fit(self, rows: list[PanelRow]) -> BeneishBaseline:
        return self

    def predict_proba(self, rows: list[PanelRow]) -> list[float]:
        import math

        out: list[float] = []
        for r in rows:
            if r.missing.get(self.METRIC, True):
                # Uncomputable M is a neutral rank, never a confident pass. A
                # filer that does not tag the inputs is not thereby clean.
                out.append(0.5)
                continue
            m = r.features.get(self.METRIC, BENEISH_THRESHOLD)
            z = max(-30.0, min(30.0, m - BENEISH_THRESHOLD))
            out.append(1.0 / (1.0 + math.exp(-z)))
        return out

    def evaluate(self, rows: list[PanelRow]):
        from models.hazard import evaluate

        return evaluate([r.label for r in rows], self.predict_proba(rows))
