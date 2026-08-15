"""L0 -- Chapter 11 discovery logic.

Exercises the three-signal filter against the real failure modes it was built
from, using stubbed EDGAR responses so the tests stay offline and fast.
"""

from __future__ import annotations

from datetime import date, timedelta

from data.discover import (
    Candidate,
    _parse_petition_date,
    _strip_html,
    find_item_103,
    inspect_candidate,
)

WINDOW_START = date(2025, 1, 1)

REAL_CH11 = """
    <p>On June 11, 2026 (the &ldquo;Petition Date&rdquo;), Sleep Number Corporation
    and its subsidiaries filed voluntary petitions for relief under chapter 11 of
    title 11 of the United States Code in the United States Bankruptcy Court.</p>
"""

RIGHTS_PLAN = """
    <p>The Company entered into an Amended and Restated Rights Agreement,
    declaring a dividend of one preferred share purchase right.</p>
"""

CHAPTER_7 = """
    <p>On November 14, 2025, the Company filed voluntary petitions to commence
    proceedings under chapter 7 of title 11 of the United States Code.</p>
"""


class StubClient:
    """Minimal EdgarClient stand-in."""

    def __init__(self, filings, document=REAL_CH11, name="Test Corp", sic=5331, assets=5e8):
        self._filings = filings
        self._document = document
        self._name = name
        self._sic = sic
        self._assets = assets

    def submissions(self, cik):
        return {"name": self._name, "sic": self._sic}

    def filing_index(self, cik):
        return self._filings

    def fetch_filing_document(self, cik, accession, primary_document):
        return self._document

    def filing_document_url(self, cik, accession, primary_document):
        return f"https://www.sec.gov/Archives/edgar/data/{cik}/x/{primary_document}"

    def company_concept(self, cik, tag, taxonomy="us-gaap"):
        return {"units": {"USD": [{"end": "2025-12-31", "filed": "2026-02-01", "val": self._assets}]}}


def filing(form, filing_date, items="", accession="0000000000-26-000001", doc="a.htm"):
    return {
        "form": form,
        "items": items,
        "filing_date": filing_date,
        "accession": accession,
        "primary_document": doc,
    }


def history(n=8, end=date(2026, 4, 1)):
    """n quarterly filings running up to ``end`` -- a filer still reporting."""
    return [
        filing("10-Q", end - timedelta(days=91 * i), accession=f"acc-{i}") for i in range(n)
    ]


class TestItemCodeScan:
    def test_finds_item_103_filings_oldest_first(self) -> None:
        filings = [
            filing("8-K", date(2026, 6, 12), items="1.03,9.01"),
            filing("8-K", date(2025, 3, 1), items="1.03"),
            filing("8-K", date(2025, 4, 1), items="2.02"),
        ]
        hits = find_item_103(StubClient(filings), 1, on_or_after=WINDOW_START)
        assert [h["filing_date"] for h in hits] == [date(2025, 3, 1), date(2026, 6, 12)]

    def test_filings_before_the_window_are_ignored(self) -> None:
        """Cumulus Media's only item 1.03 is from its 2017 bankruptcy."""
        filings = [filing("8-K", date(2017, 11, 30), items="1.03")]
        assert find_item_103(StubClient(filings), 1, on_or_after=WINDOW_START) == []

    def test_non_8k_forms_are_ignored(self) -> None:
        filings = [filing("10-K", date(2025, 3, 1), items="1.03")]
        assert find_item_103(StubClient(filings), 1, on_or_after=WINDOW_START) == []


class TestThreeSignalFilter:
    def test_genuine_bankruptcy_is_confirmed(self) -> None:
        filings = history() + [filing("8-K", date(2026, 6, 12), items="1.03,9.01")]
        c = inspect_candidate(StubClient(filings), 1, {"date": "2026-06-12"}, WINDOW_START)
        assert c.verdict == "confirmed"
        assert c.has_item103 and c.has_chapter11_text
        assert c.verification_string() == "item_1.03+chapter11_text+voluntary_petition"

    def test_miscoded_item_103_is_rejected(self) -> None:
        """The J.C. Penney / Granite Construction failure mode: the item code
        says bankruptcy, the document is a shareholder rights plan."""
        filings = history(end=date(2026, 1, 15)) + [filing("8-K", date(2026, 2, 18), items="1.03")]
        client = StubClient(filings, document=RIGHTS_PLAN)
        c = inspect_candidate(client, 1, {"date": "2026-02-18"}, WINDOW_START)
        assert c.verdict == "rejected"
        assert "no Chapter 11 language" in c.reasons[0]

    def test_text_match_without_item_code_is_rejected(self) -> None:
        """Solvent filers match 'Bankruptcy or Receivership' as boilerplate --
        WEX, Howard Hughes, SM Energy and Opendoor all did."""
        filings = history(end=date(2025, 2, 1)) + [filing("8-K", date(2025, 3, 6), items="2.02,9.01")]
        c = inspect_candidate(StubClient(filings), 1, {"date": "2025-03-06"}, WINDOW_START)
        assert c.verdict == "rejected"
        assert "no 8-K carrying item 1.03" in c.reasons[0]

    def test_chapter_7_is_rejected(self) -> None:
        """Canoo and Sonder liquidated. A different event type, not a Ch. 11."""
        filings = history(end=date(2025, 10, 1)) + [filing("8-K", date(2025, 11, 14), items="1.03")]
        client = StubClient(filings, document=CHAPTER_7)
        c = inspect_candidate(client, 1, {"date": "2025-11-14"}, WINDOW_START)
        assert c.verdict == "rejected"
        assert "Chapter 7" in c.reasons[0]

    def test_shell_without_filing_history_is_rejected(self) -> None:
        """A credit model needs financials to reason over."""
        filings = history(n=2, end=date(2025, 5, 1)) + [filing("8-K", date(2025, 6, 1), items="1.03")]
        c = inspect_candidate(StubClient(filings), 1, {"date": "2025-06-01"}, WINDOW_START)
        assert c.verdict == "rejected"
        assert "periodic filings" in c.reasons[0]

    def test_filer_that_went_dark_is_rejected(self) -> None:
        """Stopped reporting years before the event: no as-of data to predict from."""
        old = [filing("10-Q", date(2019, 3, 1), accession=f"a{i}") for i in range(8)]
        filings = old + [filing("8-K", date(2025, 6, 1), items="1.03")]
        c = inspect_candidate(StubClient(filings), 1, {"date": "2025-06-01"}, WINDOW_START)
        assert c.verdict == "rejected"
        assert "stopped reporting" in c.reasons[0]


class TestPetitionDateParsing:
    def test_parses_petition_date_from_text(self) -> None:
        assert _parse_petition_date(_strip_html(REAL_CH11).lower()) == date(2026, 6, 11)

    def test_falls_back_to_filing_date_when_unparseable(self) -> None:
        vague = "<p>The Company commenced chapter 11 proceedings. A voluntary petition was filed.</p>"
        filings = history() + [filing("8-K", date(2026, 6, 12), items="1.03")]
        c = inspect_candidate(StubClient(filings, document=vague), 1, {"date": "2026-06-12"}, WINDOW_START)
        assert c.verdict == "confirmed"
        assert c.date_basis == "8k_filing_date_fallback"
        assert c.petition_date == "2026-06-12"

    def test_emergence_filing_is_quarantined_not_misdated(self) -> None:
        """Hertz's in-window Item 1.03 is its 2021 *emergence*, which cites the
        2020 petition. Dating the bankruptcy to the emergence would be worse
        than the mis-parse, so a large gap goes to review."""
        emergence = (
            "<p>On May 22, 2020 (the Petition Date), the Company filed voluntary "
            "petitions under chapter 11. The Plan became effective today.</p>"
        )
        filings = history(end=date(2021, 5, 1)) + [filing("8-K", date(2021, 6, 16), items="1.03")]
        c = inspect_candidate(
            StubClient(filings, document=emergence), 1, {"date": "2021-06-16"}, date(2021, 1, 1)
        )
        assert c.verdict == "needs_review"
        assert "prior case" in c.reasons[0]

    def test_misparsed_date_is_quarantined(self) -> None:
        """Mondee: a 2024 date lifted out of an 8-K disclosing a 2025 petition."""
        text = (
            "<p>On January 14, 2024, the Company filed a voluntary petition under "
            "chapter 11 of the Bankruptcy Code.</p>"
        )
        filings = history(end=date(2024, 12, 1)) + [filing("8-K", date(2025, 1, 15), items="1.03")]
        c = inspect_candidate(
            StubClient(filings, document=text), 1, {"date": "2025-01-15"}, date(2021, 1, 1)
        )
        assert c.verdict == "needs_review"

    def test_normal_disclosure_lag_is_accepted(self) -> None:
        """A few days between petition and 8-K is the expected case."""
        filings = history() + [filing("8-K", date(2026, 6, 12), items="1.03")]
        c = inspect_candidate(StubClient(filings), 1, {"date": "2026-06-12"}, WINDOW_START)
        assert c.verdict == "confirmed"
        assert c.petition_date == "2026-06-11"

    def test_petition_date_after_filing_date_is_not_trusted(self) -> None:
        """A future date in the text is a parse error, not a petition date.
        Falling back can only understate lead time; trusting it would inflate."""
        future = (
            "<p>On December 31, 2099 (the Petition Date), the Company filed "
            "voluntary petitions under chapter 11.</p>"
        )
        filings = history() + [filing("8-K", date(2026, 6, 12), items="1.03")]
        c = inspect_candidate(StubClient(filings, document=future), 1, {"date": "2026-06-12"}, WINDOW_START)
        assert c.date_basis == "8k_filing_date_fallback"


class TestCandidateContract:
    def test_discovery_never_emits_a_label_directly(self) -> None:
        """Candidates carry a verdict for review; promotion is a human step."""
        c = Candidate(cik=1)
        assert c.verdict == "pending"
        assert not c.confirmed

    def test_rejected_candidates_record_why(self) -> None:
        filings = history(n=1, end=date(2025, 5, 1)) + [filing("8-K", date(2025, 6, 1), items="1.03")]
        c = inspect_candidate(StubClient(filings), 1, {"date": "2025-06-01"}, WINDOW_START)
        assert c.reasons
