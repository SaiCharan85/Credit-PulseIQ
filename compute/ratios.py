"""Deterministic ratios (SPEC 3). Pure functions, full provenance, no LLM.

Every formula takes **primitive filing values only** -- never another computed
ratio. Chaining derived values would mean an audit of one number required
auditing a tree of intermediates; keeping formulas flat means every
:class:`ComputedValue` traces in one hop to tagged line items in a specific
filing, and ``verify/`` can re-execute it against those raw values alone.

Sign conventions follow the filings: ``interest_expense`` and ``capex`` are
reported as positive magnitudes, losses are negative.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

from compute import lineitems
from compute.provenance import (
    UNIT_RATIO,
    UNIT_SCORE,
    UNIT_USD,
    ComputedValue,
    FactRef,
    build,
    formula,
    safe_div,
    undefined,
)
from data.facts import XbrlFact

# --------------------------------------------------------------------------
# Leverage
# --------------------------------------------------------------------------


@formula(
    "total_debt",
    inputs=("long_term_debt", "short_term_debt"),
    unit=UNIT_USD,
    expression="long_term_debt + short_term_debt",
)
def _total_debt(long_term_debt: float, short_term_debt: float) -> float:
    """Interest-bearing debt. Excludes operating liabilities such as payables."""
    return long_term_debt + short_term_debt


@formula(
    "net_debt",
    inputs=("long_term_debt", "short_term_debt", "cash"),
    unit=UNIT_USD,
    expression="long_term_debt + short_term_debt - cash",
)
def _net_debt(long_term_debt: float, short_term_debt: float, cash: float) -> float:
    return long_term_debt + short_term_debt - cash


@formula(
    "debt_to_assets",
    inputs=("long_term_debt", "short_term_debt", "total_assets"),
    unit=UNIT_RATIO,
    expression="(long_term_debt + short_term_debt) / total_assets",
)
def _debt_to_assets(long_term_debt: float, short_term_debt: float, total_assets: float) -> float | None:
    return safe_div(long_term_debt + short_term_debt, total_assets)


@formula(
    "liabilities_to_assets",
    inputs=("total_liabilities", "total_assets"),
    unit=UNIT_RATIO,
    expression="total_liabilities / total_assets",
)
def _liabilities_to_assets(total_liabilities: float, total_assets: float) -> float | None:
    """Above 1.0 means liabilities exceed assets -- balance-sheet insolvency."""
    return safe_div(total_liabilities, total_assets)


@formula(
    "debt_to_equity",
    inputs=("long_term_debt", "short_term_debt", "equity"),
    unit=UNIT_RATIO,
    expression="(long_term_debt + short_term_debt) / equity",
)
def _debt_to_equity(long_term_debt: float, short_term_debt: float, equity: float) -> float | None:
    """Undefined at zero equity, and *negative* at negative equity.

    A negative reading is not a low-leverage signal -- it means equity is wiped
    out. Downstream thresholds must not treat it as "below limit".
    """
    return safe_div(long_term_debt + short_term_debt, equity)


@formula(
    "debt_to_ebitda",
    inputs=("long_term_debt", "short_term_debt", "operating_income", "depreciation_amortization"),
    unit=UNIT_RATIO,
    expression="(long_term_debt + short_term_debt) / (operating_income + depreciation_amortization)",
)
def _debt_to_ebitda(
    long_term_debt: float, short_term_debt: float, operating_income: float, depreciation_amortization: float
) -> float | None:
    return safe_div(long_term_debt + short_term_debt, operating_income + depreciation_amortization)


# --------------------------------------------------------------------------
# Coverage
# --------------------------------------------------------------------------


@formula(
    "ebitda",
    inputs=("operating_income", "depreciation_amortization"),
    unit=UNIT_USD,
    expression="operating_income + depreciation_amortization",
)
def _ebitda(operating_income: float, depreciation_amortization: float) -> float:
    return operating_income + depreciation_amortization


@formula(
    "interest_coverage",
    inputs=("operating_income", "interest_expense"),
    unit=UNIT_RATIO,
    expression="operating_income / interest_expense",
)
def _interest_coverage(operating_income: float, interest_expense: float) -> float | None:
    """EBIT / interest. Below ~1.0 means operations do not cover interest."""
    return safe_div(operating_income, interest_expense)


@formula(
    "ebitda_interest_coverage",
    inputs=("operating_income", "depreciation_amortization", "interest_expense"),
    unit=UNIT_RATIO,
    expression="(operating_income + depreciation_amortization) / interest_expense",
)
def _ebitda_interest_coverage(
    operating_income: float, depreciation_amortization: float, interest_expense: float
) -> float | None:
    return safe_div(operating_income + depreciation_amortization, interest_expense)


# --------------------------------------------------------------------------
# Liquidity
# --------------------------------------------------------------------------


@formula(
    "current_ratio",
    inputs=("current_assets", "current_liabilities"),
    unit=UNIT_RATIO,
    expression="current_assets / current_liabilities",
)
def _current_ratio(current_assets: float, current_liabilities: float) -> float | None:
    return safe_div(current_assets, current_liabilities)


@formula(
    "quick_ratio",
    inputs=("current_assets", "inventory", "current_liabilities"),
    unit=UNIT_RATIO,
    expression="(current_assets - inventory) / current_liabilities",
)
def _quick_ratio(current_assets: float, inventory: float, current_liabilities: float) -> float | None:
    """Liquidity excluding inventory -- the asset that stops converting first
    when a retailer's demand falls."""
    return safe_div(current_assets - inventory, current_liabilities)


@formula(
    "cash_ratio",
    inputs=("cash", "current_liabilities"),
    unit=UNIT_RATIO,
    expression="cash / current_liabilities",
)
def _cash_ratio(cash: float, current_liabilities: float) -> float | None:
    return safe_div(cash, current_liabilities)


@formula(
    "working_capital",
    inputs=("current_assets", "current_liabilities"),
    unit=UNIT_USD,
    expression="current_assets - current_liabilities",
)
def _working_capital(current_assets: float, current_liabilities: float) -> float:
    return current_assets - current_liabilities


# --------------------------------------------------------------------------
# Margins and returns
# --------------------------------------------------------------------------


@formula(
    "gross_margin",
    inputs=("revenue", "cost_of_revenue"),
    unit=UNIT_RATIO,
    expression="(revenue - cost_of_revenue) / revenue",
)
def _gross_margin(revenue: float, cost_of_revenue: float) -> float | None:
    return safe_div(revenue - cost_of_revenue, revenue)


@formula(
    "operating_margin",
    inputs=("operating_income", "revenue"),
    unit=UNIT_RATIO,
    expression="operating_income / revenue",
)
def _operating_margin(operating_income: float, revenue: float) -> float | None:
    return safe_div(operating_income, revenue)


@formula(
    "net_margin",
    inputs=("net_income", "revenue"),
    unit=UNIT_RATIO,
    expression="net_income / revenue",
)
def _net_margin(net_income: float, revenue: float) -> float | None:
    return safe_div(net_income, revenue)


@formula(
    "return_on_assets",
    inputs=("net_income", "total_assets"),
    unit=UNIT_RATIO,
    expression="net_income / total_assets",
)
def _return_on_assets(net_income: float, total_assets: float) -> float | None:
    return safe_div(net_income, total_assets)


# --------------------------------------------------------------------------
# Cash flow
# --------------------------------------------------------------------------


@formula(
    "free_cash_flow",
    inputs=("cash_from_operations", "capex"),
    unit=UNIT_USD,
    expression="cash_from_operations - capex",
)
def _free_cash_flow(cash_from_operations: float, capex: float) -> float:
    """Capex is reported as a positive outflow, so it is subtracted."""
    return cash_from_operations - capex


@formula(
    "ocf_to_debt",
    inputs=("cash_from_operations", "long_term_debt", "short_term_debt"),
    unit=UNIT_RATIO,
    expression="cash_from_operations / (long_term_debt + short_term_debt)",
)
def _ocf_to_debt(cash_from_operations: float, long_term_debt: float, short_term_debt: float) -> float | None:
    return safe_div(cash_from_operations, long_term_debt + short_term_debt)


@formula(
    "accruals_to_assets",
    inputs=("net_income", "cash_from_operations", "total_assets"),
    unit=UNIT_RATIO,
    expression="(net_income - cash_from_operations) / total_assets",
)
def _accruals_to_assets(net_income: float, cash_from_operations: float, total_assets: float) -> float | None:
    """Total accruals scaled by assets (Sloan). Earnings that persistently
    exceed cash generation are the classic earnings-quality flag.

    Computed here because it is deterministic arithmetic. The *earnings-quality
    investigator* that reasons about it is gated behind the distress leg
    (PROMPT hard rule 8) -- having the number available is not the same as
    building the second worker.
    """
    return safe_div(net_income - cash_from_operations, total_assets)


# --------------------------------------------------------------------------
# Composite
# --------------------------------------------------------------------------


@formula(
    "altman_z_double_prime",
    inputs=(
        "current_assets",
        "current_liabilities",
        "retained_earnings",
        "operating_income",
        "equity",
        "total_liabilities",
        "total_assets",
    ),
    unit=UNIT_SCORE,
    expression=(
        "6.56*((CA-CL)/TA) + 3.26*(RE/TA) + 6.72*(EBIT/TA) + 1.05*(BVE/TL)"
    ),
)
def _altman_z_double_prime(
    current_assets: float,
    current_liabilities: float,
    retained_earnings: float,
    operating_income: float,
    equity: float,
    total_liabilities: float,
    total_assets: float,
) -> float | None:
    """Altman Z''-score.

    The Z'' variant is used deliberately: the original Z-score needs market
    value of equity, which is not in the filings and would have to come from a
    price feed with its own as-of problems. Z'' substitutes book equity over
    total liabilities, so the whole score is computable from a single filing.

    Conventional reading: below ~1.1 distress, above ~2.6 safe. Treated as one
    deterministic signal for the investigator to weigh, not a verdict.
    """
    if abs(total_assets) < 1e-12 or abs(total_liabilities) < 1e-12:
        return None
    x1 = (current_assets - current_liabilities) / total_assets
    x2 = retained_earnings / total_assets
    x3 = operating_income / total_assets
    x4 = equity / total_liabilities
    return 6.56 * x1 + 3.26 * x2 + 6.72 * x3 + 1.05 * x4


# --------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------

#: Metrics computed for every period. Keys are metric names, values are formulas.
STANDARD_METRICS: tuple[str, ...] = (
    "total_debt",
    "net_debt",
    "debt_to_assets",
    "liabilities_to_assets",
    "debt_to_equity",
    "debt_to_ebitda",
    "ebitda",
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
    "accruals_to_assets",
    "altman_z_double_prime",
)

#: Concepts where absence genuinely means zero rather than missing data.
#:
#: ``long_term_debt`` is here because of what distress does to a balance sheet:
#: once maturities accelerate or covenants trip, debt is reclassified as
#: current. Sleep Number's final pre-bankruptcy 10-K tags no long-term debt at
#: all -- its entire $588m sits in ``DebtCurrent``. Treating that as "missing"
#: would leave leverage undefined on exactly the companies the distress leg
#: exists to catch.
ZERO_IF_ABSENT = ("long_term_debt", "short_term_debt", "inventory")

#: Concepts that may only be assumed zero when a sibling was actually observed.
#: Assuming *all* debt is zero because no debt tag resolved would silently
#: report an unlevered balance sheet -- the exact direction that produces
#: high-confidence "healthy" on a company that later fails.
ZERO_REQUIRES_SIBLING: dict[str, tuple[str, ...]] = {
    "long_term_debt": ("short_term_debt",),
    "short_term_debt": ("long_term_debt",),
}


def _inputs_for(
    metric: str, facts: Sequence[XbrlFact], period_end: date
) -> tuple[dict[str, FactRef], list[str]]:
    from compute.provenance import FORMULAS

    needed = FORMULAS[metric].inputs
    refs = lineitems.resolve_many(needed, facts, period_end)
    observed = set(refs)
    notes: list[str] = []
    for concept in needed:
        if concept in refs or concept not in ZERO_IF_ABSENT:
            continue
        siblings = ZERO_REQUIRES_SIBLING.get(concept)
        if siblings and not any(s in observed for s in siblings):
            continue
        anchor = next(iter(refs.values()), None)
        if anchor is None:
            continue
        refs[concept] = lineitems.assumed_zero(concept, anchor)
        notes.append(f"{concept} not tagged by filer; assumed zero")
    return refs, notes


def compute_metric(metric: str, facts: Sequence[XbrlFact], period_end: date) -> ComputedValue:
    """One metric for one period. ``facts`` must be an as-of filtered view."""
    from compute.provenance import FORMULAS

    if metric not in FORMULAS:
        raise KeyError(f"unknown metric: {metric}")
    refs, notes = _inputs_for(metric, facts, period_end)
    missing = [c for c in FORMULAS[metric].inputs if c not in refs]
    if missing:
        return undefined(
            metric,
            metric,
            period_end,
            f"missing line items: {', '.join(sorted(missing))}",
            refs,
        )
    return build(metric, metric, period_end, refs, notes)


def compute_all(
    facts: Sequence[XbrlFact], period_end: date, metrics: Sequence[str] = STANDARD_METRICS
) -> dict[str, ComputedValue]:
    """The full deterministic signal set for one period.

    Undefined metrics are included rather than dropped: "we could not compute
    coverage because interest expense is untagged" is information the
    investigator needs, and silently omitting it invites a confident read of an
    incomplete picture (SPEC 8, data-freshness guard).
    """
    return {m: compute_metric(m, facts, period_end) for m in metrics}
