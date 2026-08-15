"""Shared fixtures for the eval ladder.

Everything here runs offline. Network-dependent checks are marked ``network``
and deselected by default so CI gates on determinism, not on SEC availability.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pytest

from compute.lineitems import CONCEPTS, INSTANT_CONCEPTS
from data.edgar import EdgarClient
from data.facts import XbrlFact

FIXTURES = Path(__file__).parent / "fixtures"

SLEEP_NUMBER_CIK = 827187  # filed Chapter 11 2026-06-12
LA_Z_BOY_CIK = 57131  # sector peer (SIC 2510), still filing


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "network: hits SEC EDGAR; deselected by default")


def _load_fixture(client: EdgarClient, slug: str, cik: int) -> list[XbrlFact]:
    payload = json.loads((FIXTURES / f"companyfacts_{slug}.json").read_text(encoding="utf8"))
    return client.facts_from_payload(cik, payload)


@pytest.fixture(scope="session")
def client(tmp_path_factory: pytest.TempPathFactory) -> EdgarClient:
    return EdgarClient(cache_dir=tmp_path_factory.mktemp("edgar_cache"), offline=True)


@pytest.fixture(scope="session")
def sleep_number_facts(client: EdgarClient) -> list[XbrlFact]:
    return _load_fixture(client, "sleep_number", SLEEP_NUMBER_CIK)


@pytest.fixture(scope="session")
def la_z_boy_facts(client: EdgarClient) -> list[XbrlFact]:
    return _load_fixture(client, "la_z_boy", LA_Z_BOY_CIK)


# ---------------------------------------------------------------------------
# Synthetic builders
#
# Hand-built facts with hand-computed expected values are how L0 pins the
# arithmetic. Real filings verify extraction (L1); they cannot verify a ratio,
# because checking a ratio against a value the same code produced is circular.
# ---------------------------------------------------------------------------

DEFAULT_PERIOD_END = date(2024, 12, 31)
DEFAULT_FILED = date(2025, 2, 20)

#: Imported, never re-listed. A hand-copied duplicate of this set drifted the
#: moment new balance-sheet concepts were added, and silently produced
#: duration facts that no instant lookup could resolve.
INSTANT = INSTANT_CONCEPTS


def make_fact(
    concept: str,
    value: float,
    period_end: date = DEFAULT_PERIOD_END,
    filed: date = DEFAULT_FILED,
    cik: int = 1,
    form: str = "10-K",
    accession: str = "0000000000-00-000000",
    tag: str | None = None,
) -> XbrlFact:
    """One fact for a canonical concept, using its primary tag by default."""
    resolved_tag = tag or CONCEPTS[concept][0]
    is_instant = concept in INSTANT
    return XbrlFact(
        cik=cik,
        taxonomy="us-gaap",
        tag=resolved_tag,
        unit="USD",
        value=value,
        period_start=None if is_instant else period_end - timedelta(days=364),
        period_end=period_end,
        form=form,
        accession=accession,
        filed=filed,
    )


def make_facts(
    values: dict[str, float],
    period_end: date = DEFAULT_PERIOD_END,
    filed: date = DEFAULT_FILED,
    cik: int = 1,
) -> list[XbrlFact]:
    return [make_fact(c, v, period_end, filed, cik) for c, v in values.items()]


#: A distressed profile: negative equity, coverage barely above 1, negative
#: working capital, accumulated deficit. Every expected value in
#: ``test_l0_ratios`` is hand-computed from these numbers.
DISTRESSED: dict[str, float] = {
    "total_assets": 1000.0,
    "total_liabilities": 1200.0,
    "current_assets": 400.0,
    "current_liabilities": 500.0,
    "cash": 50.0,
    "inventory": 100.0,
    "equity": -200.0,
    "long_term_debt": 600.0,
    "short_term_debt": 100.0,
    "retained_earnings": -300.0,
    "revenue": 2000.0,
    "cost_of_revenue": 1500.0,
    "operating_income": 100.0,
    "net_income": -50.0,
    "interest_expense": 80.0,
    "depreciation_amortization": 60.0,
    "cash_from_operations": 30.0,
    "capex": 20.0,
}


@pytest.fixture
def distressed_facts() -> list[XbrlFact]:
    return make_facts(DISTRESSED)
