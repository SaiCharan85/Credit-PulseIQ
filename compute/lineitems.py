"""Canonical concepts and their XBRL tag fallback chains.

Filers do not agree on tags. Revenue alone appears as at least four different
us-gaap concepts depending on filer, era and ASC 606 adoption. Resolution walks
an ordered chain and records which tag actually matched, so provenance reflects
what was read rather than what was hoped for.

Plain deterministic code. No LLM anywhere near tag selection.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import date

from compute.provenance import FactRef
from data.facts import XbrlFact

# Ordered fallback chains. Earlier tags are preferred: they are the more
# specific or more modern concept.
CONCEPTS: dict[str, tuple[str, ...]] = {
    # Balance sheet (instant)
    "total_assets": ("Assets",),
    "total_liabilities": ("Liabilities",),
    "current_assets": ("AssetsCurrent",),
    "current_liabilities": ("LiabilitiesCurrent",),
    "cash": (
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
        "CashAndCashEquivalentsAtCarryingValueIncludingDiscontinuedOperations",
    ),
    "inventory": ("InventoryNet",),
    "receivables": (
        "AccountsReceivableNetCurrent",
        "ReceivablesNetCurrent",
        "AccountsReceivableNet",
    ),
    "equity": (
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ),
    "long_term_debt": (
        "LongTermDebtNoncurrent",
        "LongTermDebt",
        "LongTermDebtAndCapitalLeaseObligations",
    ),
    "short_term_debt": (
        "LongTermDebtCurrent",
        "DebtCurrent",
        "ShortTermBorrowings",
        "LongTermDebtAndCapitalLeaseObligationsCurrent",
    ),
    "retained_earnings": ("RetainedEarningsAccumulatedDeficit",),
    "accounts_payable": (
        "AccountsPayableCurrent",
        "AccountsPayableTradeCurrent",
        "AccountsPayableAndAccruedLiabilitiesCurrent",
    ),
    "ppe_net": (
        "PropertyPlantAndEquipmentNet",
        "PropertyPlantAndEquipmentAndFinanceLeaseRightOfUseAssetAfterAccumulatedDepreciationAndAmortization",
    ),
    "shares_outstanding": (
        "CommonStockSharesOutstanding",
        "CommonStockSharesIssued",
        "EntityCommonStockSharesOutstanding",
    ),
    # Income statement (annual duration)
    "revenue": (
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
    ),
    "cost_of_revenue": (
        "CostOfGoodsAndServicesSold",
        "CostOfRevenue",
        "CostOfGoodsSold",
    ),
    "gross_profit": ("GrossProfit",),
    "operating_income": ("OperatingIncomeLoss",),
    "net_income": ("NetIncomeLoss", "ProfitLoss"),
    # Deliberately excludes the *net* interest tags
    # (``InterestIncomeExpenseNonoperatingNet``, ``InterestIncomeExpenseNet``)
    # and the cash-flow item ``InterestPaidNet``, even though they are the most
    # common alternatives on filers that skip these tags. Net tags fold interest
    # income in and flip sign between filers; a wrongly-signed coverage ratio
    # reads as healthy, which is the one direction that must never happen
    # silently. Those filers get an undefined coverage ratio instead.
    "interest_expense": (
        "InterestExpense",
        "InterestExpenseDebt",
        "InterestExpenseNonoperating",
        "InterestAndDebtExpense",
        "InterestExpenseOther",
    ),
    "depreciation_amortization": (
        "DepreciationDepletionAndAmortization",
        "DepreciationAmortizationAndAccretionNet",
        "DepreciationAndAmortization",
    ),
    "income_tax_expense": ("IncomeTaxExpenseBenefit",),
    "sga_expense": (
        "SellingGeneralAndAdministrativeExpense",
        "GeneralAndAdministrativeExpense",
    ),
    "pretax_income": (
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesDomestic",
    ),
    # Cash flow (annual duration)
    "cash_from_operations": ("NetCashProvidedByUsedInOperatingActivities",),
    "capex": (
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets",
    ),
}

INSTANT_CONCEPTS = frozenset(
    {
        "total_assets",
        "total_liabilities",
        "current_assets",
        "current_liabilities",
        "cash",
        "inventory",
        "receivables",
        "equity",
        "long_term_debt",
        "short_term_debt",
        "retained_earnings",
        "accounts_payable",
        "ppe_net",
        "shares_outstanding",
    }
)

ANNUAL_MIN_DAYS = 350
ANNUAL_MAX_DAYS = 380

ABSENT_TAG = "ABSENT:assumed_zero"


class ConceptError(KeyError):
    pass


def is_instant(concept: str) -> bool:
    if concept not in CONCEPTS:
        raise ConceptError(concept)
    return concept in INSTANT_CONCEPTS


class FactIndex:
    """A fact set indexed by tag.

    Resolution otherwise rescans every fact for every candidate tag, which is
    fine for one lookup and quadratic for a backtest panel: 369 filers x 19
    observation dates x ~30 metrics, each scanning tens of thousands of facts.
    Indexing once per as-of view turns that into a dict lookup.

    Behaves as a sequence so it can be passed anywhere a fact list is expected.
    """

    __slots__ = ("_facts", "_by_tag")

    def __init__(self, facts: Sequence[XbrlFact]) -> None:
        self._facts = list(facts)
        self._by_tag: dict[str, list[XbrlFact]] = {}
        for f in self._facts:
            self._by_tag.setdefault(f.tag, []).append(f)

    def __iter__(self):
        return iter(self._facts)

    def __len__(self) -> int:
        return len(self._facts)

    def __getitem__(self, i):
        return self._facts[i]

    def for_tag(self, tag: str) -> Sequence[XbrlFact]:
        return self._by_tag.get(tag, ())


def _scan(facts: Sequence[XbrlFact], tag: str) -> Sequence[XbrlFact]:
    if isinstance(facts, FactIndex):
        return facts.for_tag(tag)
    return facts


def _candidates(facts: Sequence[XbrlFact], tag: str, period_end: date, instant: bool) -> list[XbrlFact]:
    out = []
    for f in _scan(facts, tag):
        if f.tag != tag or f.period_end != period_end:
            continue
        if instant:
            if f.is_instant:
                out.append(f)
        else:
            d = f.duration_days
            if d is not None and ANNUAL_MIN_DAYS <= d <= ANNUAL_MAX_DAYS:
                out.append(f)
    return out


def resolve(concept: str, facts: Sequence[XbrlFact], period_end: date) -> FactRef | None:
    """First tag in the chain with a fact at ``period_end``, or ``None``.

    ``facts`` must already be filtered to the as-of view
    (``data.facts.as_of_view``). This function does no date filtering of its
    own by design -- putting the lookahead control in one place keeps it
    auditable, and a second implementation here could drift out of agreement
    with it.
    """
    if concept not in CONCEPTS:
        raise ConceptError(concept)
    instant = is_instant(concept)
    for tag in CONCEPTS[concept]:
        hits = _candidates(facts, tag, period_end, instant)
        if hits:
            # Latest filed wins (restatement); accession breaks ties.
            best = max(hits, key=lambda f: (f.filed, f.accession))
            return FactRef.from_fact(concept, best)
    return None


def resolve_many(
    concepts: Iterable[str], facts: Sequence[XbrlFact], period_end: date
) -> dict[str, FactRef]:
    """Resolve several concepts; absent ones are simply omitted."""
    out: dict[str, FactRef] = {}
    for c in concepts:
        ref = resolve(c, facts, period_end)
        if ref is not None:
            out[c] = ref
    return out


def assumed_zero(concept: str, like: FactRef) -> FactRef:
    """An explicit zero for a concept the filer did not tag.

    Used only where absence genuinely implies zero (a company with no current
    portion of long-term debt often omits the tag entirely). The synthetic tag
    ``ABSENT:assumed_zero`` keeps this visible in the audit trail rather than
    letting a substituted zero masquerade as reported data -- an invisible zero
    in a debt total would understate leverage, which is the direction that
    causes false confidence.
    """
    return FactRef(
        concept=concept,
        tag=ABSENT_TAG,
        taxonomy=like.taxonomy,
        unit=like.unit,
        value=0.0,
        period_end=like.period_end,
        period_start=like.period_start,
        form=like.form,
        accession=like.accession,
        filed=like.filed,
    )


def period_ends(
    facts: Sequence[XbrlFact], concept: str = "total_assets", limit: int | None = None
) -> list[date]:
    """Period ends available for a concept, most recent first."""
    if concept not in CONCEPTS:
        raise ConceptError(concept)
    instant = is_instant(concept)
    seen: set[date] = set()
    if isinstance(facts, FactIndex):
        pool = [f for tag in CONCEPTS[concept] for f in facts.for_tag(tag)]
    else:
        pool = [f for f in facts if f.tag in CONCEPTS[concept]]
    for f in pool:
        if instant and f.is_instant:
            seen.add(f.period_end)
        elif not instant:
            d = f.duration_days
            if d is not None and ANNUAL_MIN_DAYS <= d <= ANNUAL_MAX_DAYS:
                seen.add(f.period_end)
    out = sorted(seen, reverse=True)
    return out[:limit] if limit else out


def annual_period_ends(facts: Sequence[XbrlFact], limit: int | None = None) -> list[date]:
    """Fiscal year ends where both a balance sheet and an annual income
    statement exist -- the periods a full ratio set can be computed for."""
    bs = set(period_ends(facts, "total_assets"))
    is_ = set(period_ends(facts, "revenue")) | set(period_ends(facts, "net_income"))
    out = sorted(bs & is_, reverse=True)
    return out[:limit] if limit else out
