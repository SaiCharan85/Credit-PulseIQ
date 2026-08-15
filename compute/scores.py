"""Composite distress and earnings-quality scores (SPEC 3).

Extends the single-period ratios in ``compute/ratios.py`` with:

* **Liquidity runway** and the **working-capital cycle** — single period.
* **Piotroski F**, **Beneish M**, **Ohlson O** — year-on-year, so they need two
  periods.

Two-period formulas keep the same contract as everything else: inputs are
primitive filing values only, and prior-year inputs are namespaced with a
``_prior`` suffix. A formula still re-executes from its recorded inputs alone,
so ``verify/recompute.py`` works unchanged and every score remains auditable in
one hop back to tagged line items.

These are published scores with published coefficients; the point of
implementing them here is that they are *deterministic* and *provenanced*, so
the investigator can cite one without the LLM doing arithmetic.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from math import log

from compute import lineitems
from compute.provenance import (
    UNIT_DAYS,
    UNIT_SCORE,
    ComputedValue,
    FactRef,
    build,
    formula,
    safe_div,
    undefined,
)
from data.facts import XbrlFact

PRIOR_SUFFIX = "_prior"
DAYS_IN_YEAR = 365.0

# --------------------------------------------------------------------------
# Liquidity runway
# --------------------------------------------------------------------------


@formula(
    "cash_runway_months",
    inputs=("cash", "cash_from_operations"),
    unit=UNIT_DAYS,
    expression="cash / (-cash_from_operations / 12)",
)
def _cash_runway_months(cash: float, cash_from_operations: float) -> float | None:
    """Months of cash left at the current burn rate.

    Undefined when operations generate cash: a company that is not burning has
    no runway *limit*, and reporting a huge number would read as a quantified
    strength rather than an inapplicable metric.
    """
    if cash_from_operations >= 0:
        return None
    return safe_div(cash, -cash_from_operations / 12.0)


# --------------------------------------------------------------------------
# Working-capital cycle
# --------------------------------------------------------------------------


@formula(
    "days_sales_outstanding",
    inputs=("receivables", "revenue"),
    unit=UNIT_DAYS,
    expression="receivables / revenue * 365",
)
def _dso(receivables: float, revenue: float) -> float | None:
    """Rising DSO means customers are paying more slowly -- or revenue is being
    recognised ahead of collection."""
    r = safe_div(receivables, revenue)
    return None if r is None else r * DAYS_IN_YEAR


@formula(
    "days_inventory_outstanding",
    inputs=("inventory", "cost_of_revenue"),
    unit=UNIT_DAYS,
    expression="inventory / cost_of_revenue * 365",
)
def _dio(inventory: float, cost_of_revenue: float) -> float | None:
    r = safe_div(inventory, cost_of_revenue)
    return None if r is None else r * DAYS_IN_YEAR


@formula(
    "days_payables_outstanding",
    inputs=("accounts_payable", "cost_of_revenue"),
    unit=UNIT_DAYS,
    expression="accounts_payable / cost_of_revenue * 365",
)
def _dpo(accounts_payable: float, cost_of_revenue: float) -> float | None:
    """Stretching payables is one of the most reliable late-stage distress
    tells: a company short of cash pays its suppliers later."""
    r = safe_div(accounts_payable, cost_of_revenue)
    return None if r is None else r * DAYS_IN_YEAR


@formula(
    "cash_conversion_cycle",
    inputs=("receivables", "inventory", "accounts_payable", "revenue", "cost_of_revenue"),
    unit=UNIT_DAYS,
    expression="DSO + DIO - DPO",
)
def _ccc(
    receivables: float,
    inventory: float,
    accounts_payable: float,
    revenue: float,
    cost_of_revenue: float,
) -> float | None:
    dso = safe_div(receivables, revenue)
    dio = safe_div(inventory, cost_of_revenue)
    dpo = safe_div(accounts_payable, cost_of_revenue)
    if dso is None or dio is None or dpo is None:
        return None
    return (dso + dio - dpo) * DAYS_IN_YEAR


# --------------------------------------------------------------------------
# Piotroski F-score (two period)
# --------------------------------------------------------------------------


@formula(
    "piotroski_f_score",
    inputs=(
        "net_income",
        "cash_from_operations",
        "total_assets",
        "long_term_debt",
        "current_assets",
        "current_liabilities",
        "revenue",
        "cost_of_revenue",
        "shares_outstanding",
        "net_income_prior",
        "cash_from_operations_prior",
        "total_assets_prior",
        "long_term_debt_prior",
        "current_assets_prior",
        "current_liabilities_prior",
        "revenue_prior",
        "cost_of_revenue_prior",
        "shares_outstanding_prior",
    ),
    unit=UNIT_SCORE,
    expression="sum of 9 binary fundamental tests (0-9)",
)
def _piotroski(
    net_income: float,
    cash_from_operations: float,
    total_assets: float,
    long_term_debt: float,
    current_assets: float,
    current_liabilities: float,
    revenue: float,
    cost_of_revenue: float,
    shares_outstanding: float,
    net_income_prior: float,
    cash_from_operations_prior: float,
    total_assets_prior: float,
    long_term_debt_prior: float,
    current_assets_prior: float,
    current_liabilities_prior: float,
    revenue_prior: float,
    cost_of_revenue_prior: float,
    shares_outstanding_prior: float,
) -> float | None:
    """Piotroski F-score, 0-9. Low scores indicate weak fundamentals.

    Nine binary tests across profitability, leverage/liquidity and efficiency.
    Deliberately coarse: each signal contributes one point, which is what makes
    it robust to any single noisy input.
    """
    roa = safe_div(net_income, total_assets)
    roa_prior = safe_div(net_income_prior, total_assets_prior)
    ltd_ratio = safe_div(long_term_debt, total_assets)
    ltd_ratio_prior = safe_div(long_term_debt_prior, total_assets_prior)
    curr = safe_div(current_assets, current_liabilities)
    curr_prior = safe_div(current_assets_prior, current_liabilities_prior)
    gm = safe_div(revenue - cost_of_revenue, revenue)
    gm_prior = safe_div(revenue_prior - cost_of_revenue_prior, revenue_prior)
    turnover = safe_div(revenue, total_assets)
    turnover_prior = safe_div(revenue_prior, total_assets_prior)

    if None in (roa, roa_prior, ltd_ratio, ltd_ratio_prior, curr, curr_prior, gm, gm_prior,
                turnover, turnover_prior):
        return None

    score = 0
    score += 1 if roa > 0 else 0
    score += 1 if cash_from_operations > 0 else 0
    score += 1 if roa > roa_prior else 0
    score += 1 if cash_from_operations > net_income else 0  # accrual quality
    score += 1 if ltd_ratio < ltd_ratio_prior else 0
    score += 1 if curr > curr_prior else 0
    score += 1 if shares_outstanding <= shares_outstanding_prior else 0  # no dilution
    score += 1 if gm > gm_prior else 0
    score += 1 if turnover > turnover_prior else 0
    return float(score)


# --------------------------------------------------------------------------
# Beneish M-score (two period)
# --------------------------------------------------------------------------


@formula(
    "beneish_m_score",
    inputs=(
        "receivables",
        "revenue",
        "cost_of_revenue",
        "current_assets",
        "ppe_net",
        "total_assets",
        "depreciation_amortization",
        "sga_expense",
        "long_term_debt",
        "current_liabilities",
        "net_income",
        "cash_from_operations",
        "receivables_prior",
        "revenue_prior",
        "cost_of_revenue_prior",
        "current_assets_prior",
        "ppe_net_prior",
        "total_assets_prior",
        "depreciation_amortization_prior",
        "sga_expense_prior",
        "long_term_debt_prior",
        "current_liabilities_prior",
    ),
    unit=UNIT_SCORE,
    expression=(
        "-4.84 + 0.920*DSRI + 0.528*GMI + 0.404*AQI + 0.892*SGI + 0.115*DEPI "
        "- 0.172*SGAI + 4.679*TATA - 0.327*LVGI"
    ),
)
def _beneish(
    receivables: float,
    revenue: float,
    cost_of_revenue: float,
    current_assets: float,
    ppe_net: float,
    total_assets: float,
    depreciation_amortization: float,
    sga_expense: float,
    long_term_debt: float,
    current_liabilities: float,
    net_income: float,
    cash_from_operations: float,
    receivables_prior: float,
    revenue_prior: float,
    cost_of_revenue_prior: float,
    current_assets_prior: float,
    ppe_net_prior: float,
    total_assets_prior: float,
    depreciation_amortization_prior: float,
    sga_expense_prior: float,
    long_term_debt_prior: float,
    current_liabilities_prior: float,
) -> float | None:
    """Beneish M-score. Above roughly -1.78 flags possible manipulation.

    An earnings-quality signal, computed here because it is deterministic. The
    investigator that reasons about it is gated behind the distress leg
    (PROMPT hard rule 8).
    """
    def ratio(a: float, b: float) -> float | None:
        return safe_div(a, b)

    dsr = ratio(receivables, revenue)
    dsr_prior = ratio(receivables_prior, revenue_prior)
    dsri = safe_div(dsr, dsr_prior) if None not in (dsr, dsr_prior) else None

    gm = ratio(revenue - cost_of_revenue, revenue)
    gm_prior = ratio(revenue_prior - cost_of_revenue_prior, revenue_prior)
    gmi = safe_div(gm_prior, gm) if None not in (gm, gm_prior) else None

    aq = ratio(total_assets - current_assets - ppe_net, total_assets)
    aq_prior = ratio(total_assets_prior - current_assets_prior - ppe_net_prior, total_assets_prior)
    aqi = safe_div(aq, aq_prior) if None not in (aq, aq_prior) else None

    sgi = safe_div(revenue, revenue_prior)

    dep = ratio(depreciation_amortization, depreciation_amortization + ppe_net)
    dep_prior = ratio(
        depreciation_amortization_prior, depreciation_amortization_prior + ppe_net_prior
    )
    depi = safe_div(dep_prior, dep) if None not in (dep, dep_prior) else None

    sga = ratio(sga_expense, revenue)
    sga_prior = ratio(sga_expense_prior, revenue_prior)
    sgai = safe_div(sga, sga_prior) if None not in (sga, sga_prior) else None

    lvg = ratio(long_term_debt + current_liabilities, total_assets)
    lvg_prior = ratio(long_term_debt_prior + current_liabilities_prior, total_assets_prior)
    lvgi = safe_div(lvg, lvg_prior) if None not in (lvg, lvg_prior) else None

    tata = ratio(net_income - cash_from_operations, total_assets)

    if None in (dsri, gmi, aqi, sgi, depi, sgai, lvgi, tata):
        return None
    return (
        -4.84
        + 0.920 * dsri
        + 0.528 * gmi
        + 0.404 * aqi
        + 0.892 * sgi
        + 0.115 * depi
        - 0.172 * sgai
        + 4.679 * tata
        - 0.327 * lvgi
    )


# --------------------------------------------------------------------------
# Ohlson O-score (two period)
# --------------------------------------------------------------------------


@formula(
    "ohlson_o_score",
    inputs=(
        "total_assets",
        "total_liabilities",
        "current_assets",
        "current_liabilities",
        "net_income",
        "cash_from_operations",
        "net_income_prior",
    ),
    unit=UNIT_SCORE,
    expression=(
        "-1.32 - 0.407*ln(TA) + 6.03*(TL/TA) - 1.43*(WC/TA) + 0.0757*(CL/CA) "
        "- 1.72*OENEG - 2.37*(NI/TA) - 1.83*(CFO/TL) + 0.285*INTWO "
        "- 0.521*((NI-NI_prior)/(|NI|+|NI_prior|))"
    ),
)
def _ohlson(
    total_assets: float,
    total_liabilities: float,
    current_assets: float,
    current_liabilities: float,
    net_income: float,
    cash_from_operations: float,
    net_income_prior: float,
) -> float | None:
    """Ohlson O-score. Higher means higher modelled default probability.

    Deviation from the 1980 original, stated rather than hidden: the size term
    is ``ln(total assets)`` instead of assets deflated by the GNP price index.
    The deflator would make scores incomparable across the 2021-2026 window
    unless pinned to an as-of vintage of an external series -- an extra
    lookahead surface for a term that only rescales size. Absolute values are
    therefore not comparable to published tables; relative ranking is.
    """
    if total_assets <= 0 or abs(total_liabilities) < 1e-12 or abs(current_assets) < 1e-12:
        return None
    working_capital = current_assets - current_liabilities
    oeneg = 1.0 if total_liabilities > total_assets else 0.0
    intwo = 1.0 if (net_income < 0 and net_income_prior < 0) else 0.0
    ni_scale = abs(net_income) + abs(net_income_prior)
    chin = (net_income - net_income_prior) / ni_scale if ni_scale > 1e-12 else 0.0
    return (
        -1.32
        - 0.407 * log(total_assets)
        + 6.03 * (total_liabilities / total_assets)
        - 1.43 * (working_capital / total_assets)
        + 0.0757 * (current_liabilities / current_assets)
        - 1.72 * oeneg
        - 2.37 * (net_income / total_assets)
        - 1.83 * (cash_from_operations / total_liabilities)
        + 0.285 * intwo
        - 0.521 * chin
    )


# --------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------

SINGLE_PERIOD_SCORES: tuple[str, ...] = (
    "cash_runway_months",
    "days_sales_outstanding",
    "days_inventory_outstanding",
    "days_payables_outstanding",
    "cash_conversion_cycle",
)

TWO_PERIOD_SCORES: tuple[str, ...] = (
    "piotroski_f_score",
    "beneish_m_score",
    "ohlson_o_score",
)

ALL_SCORES: tuple[str, ...] = SINGLE_PERIOD_SCORES + TWO_PERIOD_SCORES


def compute_two_period_score(
    metric: str,
    facts: Sequence[XbrlFact],
    period_end: date,
    prior_period_end: date,
) -> ComputedValue:
    """Resolve current and prior-year inputs, then execute the formula.

    Prior-year concepts are namespaced with ``_prior``. Provenance keeps both
    vintages, so an auditor can see which two filings a year-on-year score
    compared.
    """
    from compute.provenance import FORMULAS

    if metric not in FORMULAS:
        raise KeyError(f"unknown metric: {metric}")
    needed = FORMULAS[metric].inputs
    current_concepts = [c for c in needed if not c.endswith(PRIOR_SUFFIX)]
    prior_concepts = [c[: -len(PRIOR_SUFFIX)] for c in needed if c.endswith(PRIOR_SUFFIX)]

    refs: dict[str, FactRef] = {}
    refs.update(lineitems.resolve_many(current_concepts, facts, period_end))
    for concept, ref in lineitems.resolve_many(prior_concepts, facts, prior_period_end).items():
        refs[f"{concept}{PRIOR_SUFFIX}"] = ref

    missing = [c for c in needed if c not in refs]
    if missing:
        return undefined(
            metric,
            metric,
            period_end,
            f"missing line items: {', '.join(sorted(missing))}",
            refs,
        )
    return build(
        metric,
        metric,
        period_end,
        refs,
        [f"year-on-year vs {prior_period_end.isoformat()}"],
    )


def compute_scores(
    facts: Sequence[XbrlFact],
    period_end: date,
    prior_period_end: date | None = None,
) -> dict[str, ComputedValue]:
    """Every score available for a period.

    Two-period scores are omitted (not faked) when no prior year is supplied.
    """
    from compute.ratios import compute_metric

    out = {m: compute_metric(m, facts, period_end) for m in SINGLE_PERIOD_SCORES}
    if prior_period_end is not None:
        for m in TWO_PERIOD_SCORES:
            out[m] = compute_two_period_score(m, facts, period_end, prior_period_end)
    return out
