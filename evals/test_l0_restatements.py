"""L0: restatement (8-K item 4.02) discovery and classification.

The exclusion tests carry the weight. A label set that quietly includes the
2021 SPAC warrant wave would teach an earnings-quality model to recognise a
market-wide accounting rule change, and it would score well while learning
nothing about accounting quality.
"""

from __future__ import annotations

from datetime import date

from data.restatements import (
    KEEP,
    REJECT_NO_TEXT,
    REJECT_NOT_OWN,
    REJECT_SPAC_SHARES,
    REJECT_SPAC_WARRANT,
    classify,
    discover,
    find_non_reliance_filings,
    first_event_per_company,
    is_spac_share_reclassification,
    is_spac_warrant_restatement,
)

GENUINE = (
    "On March 3, 2023 the Audit Committee concluded that the previously issued "
    "financial statements for the year ended December 31, 2021 should no longer "
    "be relied upon due to errors in revenue recognition."
)
SPAC = (
    "Following the Staff Statement on Accounting and Reporting Considerations "
    "for Warrants Issued by Special Purpose Acquisition Companies, the Company "
    "concluded its previously issued financial statements should no longer be "
    "relied upon, and will reclassify its warrants from equity to a liability."
)
OPERATING_WARRANT = (
    "The Company determined that its previously issued financial statements "
    "should no longer be relied upon because warrants issued to a lender were "
    "misvalued as a result of an error in the option pricing model."
)


def row(items: str, filed: date, form: str = "8-K") -> dict:
    return {
        "form": form,
        "items": items,
        "filing_date": filed,
        "accession": "0001-23-000001",
        "primary_document": "d.htm",
    }


class TestFindingTheFilings:
    def test_item_402_is_selected(self) -> None:
        assert len(find_non_reliance_filings([row("4.02,9.01", date(2023, 3, 3))])) == 1

    def test_other_items_are_ignored(self) -> None:
        assert find_non_reliance_filings([row("2.04,9.01", date(2023, 3, 3))]) == []

    def test_substring_does_not_match(self) -> None:
        """'14.02' and '4.021' must not be read as item 4.02."""
        assert find_non_reliance_filings([row("14.02", date(2023, 3, 3))]) == []

    def test_non_8k_forms_are_ignored(self) -> None:
        assert find_non_reliance_filings([row("4.02", date(2023, 3, 3), form="10-K")]) == []

    def test_amended_8ks_still_count(self) -> None:
        assert len(find_non_reliance_filings([row("4.02", date(2023, 3, 3), form="8-K/A")])) == 1

    def test_since_filters_older_events(self) -> None:
        rows = [row("4.02", date(2015, 1, 1)), row("4.02", date(2023, 3, 3))]
        found = find_non_reliance_filings(rows, since=date(2019, 1, 1))
        assert [f["filing_date"] for f in found] == [date(2023, 3, 3)]


class TestClassification:
    def test_a_genuine_restatement_is_kept(self) -> None:
        verdict, detail = classify(GENUINE)
        assert verdict == KEEP
        assert "relied upon" in detail

    def test_the_spac_warrant_wave_is_excluded(self) -> None:
        verdict, detail = classify(SPAC)
        assert verdict == REJECT_SPAC_WARRANT
        assert "warrant" in detail

    def test_an_operating_company_warrant_error_is_kept(self) -> None:
        """Warrants alone are not disqualifying -- only the SPAC reclassification.

        A real company misvaluing warrants is a genuine accounting failure, and
        excluding it would throw away true positives to remove the 2021 wave.
        """
        assert classify(OPERATING_WARRANT)[0] == KEEP

    def test_an_empty_filing_is_rejected(self) -> None:
        assert classify("")[0] == REJECT_NO_TEXT

    def test_a_bare_item_header_is_rejected(self) -> None:
        """The item header appears with no disclosure attached."""
        text = "Item 4.02. Non-Reliance. Item 9.01. Exhibits. Signature."
        assert classify(text)[0] in (KEEP, REJECT_NOT_OWN)

    def test_unrelated_text_is_rejected(self) -> None:
        assert classify("The Company announced a new Chief Marketing Officer.")[0] == REJECT_NOT_OWN

    def test_spac_detector_needs_more_than_the_word_warrant(self) -> None:
        assert is_spac_warrant_restatement("The Company issued warrants to a lender.") is False
        assert is_spac_warrant_restatement(SPAC) is True


SPAC_SHARES = (
    "The Company concluded its previously issued financial statements should no "
    "longer be relied upon. All Class A ordinary shares subject to possible "
    "redemption are reclassified to temporary equity. The Trust Account holds "
    "the proceeds of the initial public offering."
)
REDEEMABLE_PREFERRED = (
    "Previously issued financial statements should no longer be relied upon: the "
    "Company misclassified its redeemable preferred stock within temporary equity "
    "as a result of an error in analysing the redemption feature."
)


class TestSecondSpacWave:
    """Late 2021: Class A shares reclassified to temporary equity.

    Uses no warrant language, so the warrant rule passes it straight through.
    After that rule alone, 2021 still held 299 events against a 2019-2020
    baseline near 65.
    """

    def test_the_share_reclassification_wave_is_excluded(self) -> None:
        assert classify(SPAC_SHARES)[0] == REJECT_SPAC_SHARES

    def test_the_warrant_rule_alone_would_have_missed_it(self) -> None:
        assert is_spac_warrant_restatement(SPAC_SHARES) is False
        assert is_spac_share_reclassification(SPAC_SHARES) is True

    def test_an_operating_company_temporary_equity_error_is_kept(self) -> None:
        """Redeemable preferred also lives in temporary equity, and restating
        it is a genuine accounting failure."""
        assert classify(REDEEMABLE_PREFERRED)[0] == KEEP

    def test_redemption_language_alone_is_not_enough(self) -> None:
        assert is_spac_share_reclassification("shares subject to possible redemption") is False


class TestDiscovery:
    def test_classifies_and_sorts(self) -> None:
        idx = {
            1: [row("4.02", date(2023, 3, 3))],
            2: [row("4.02", date(2021, 5, 1))],
        }
        text = {1: GENUINE, 2: SPAC}
        found = discover([1, 2], lambda c: idx[c], lambda c, a, d: text[c])
        assert [c.cik for c in found] == [2, 1]  # date order
        assert found[0].verdict == REJECT_SPAC_WARRANT
        assert found[1].verdict == KEEP

    def test_an_unreachable_index_does_not_abort_the_scan(self) -> None:
        def index_for(cik):
            if cik == 1:
                raise RuntimeError("503")
            return [row("4.02", date(2023, 3, 3))]

        found = discover([1, 2], index_for, lambda c, a, d: GENUINE)
        assert [c.cik for c in found] == [2]

    def test_a_failed_text_fetch_is_recorded_not_assumed_clean(self) -> None:
        def boom(cik, accession, document):
            raise RuntimeError("timeout")

        found = discover([1], lambda c: [row("4.02", date(2023, 3, 3))], boom)
        assert found[0].verdict == REJECT_NO_TEXT
        assert not found[0].kept


class TestDeduplication:
    def test_only_the_earliest_event_per_company_is_kept(self) -> None:
        """One restatement episode often spans several 8-Ks."""
        cands = discover(
            [1],
            lambda c: [row("4.02", date(2023, 3, 3)), row("4.02", date(2023, 6, 1))],
            lambda c, a, d: GENUINE,
        )
        first = first_event_per_company(cands)
        assert len(first) == 1
        assert first[0].filing_date == date(2023, 3, 3)

    def test_rejected_candidates_never_become_events(self) -> None:
        cands = discover([1], lambda c: [row("4.02", date(2021, 5, 1))], lambda c, a, d: SPAC)
        assert first_event_per_company(cands) == []
