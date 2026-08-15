"""L0 -- label-set integrity and the grader/input asymmetry.

Labels are the entire edge (README, "Scope discipline"), so the label file
itself is tested as rigorously as the code: dating, provenance, verification
level, and the boundary between what the agent may see and what only the
grader may see.
"""

from __future__ import annotations

from datetime import date

import pytest

from data.labels import (
    EVENT_CHAPTER11,
    LabelRecord,
    labels_visible_as_of,
    lead_time_days,
    load_aaer_labels,
    load_chapter11_labels,
    outcome_within_horizon,
)
from data.watchlist import COHORT_DISTRESS, COHORT_SURVIVOR, by_cohort, load_watchlist

COHORTS = {"recent_2025_2026", "prior_2021_2024", "historical_pre_2021"}
REQUIRED_SIGNALS = ("item_1.03", "chapter11_text", "voluntary_petition")


@pytest.fixture(scope="module")
def ch11() -> list[LabelRecord]:
    return load_chapter11_labels()


class TestLabelDating:
    def test_disclosure_never_precedes_the_event(self, ch11) -> None:
        for x in ch11:
            assert x.as_of_date >= x.event_date, x.company

    def test_disclosure_lag_is_plausible(self, ch11) -> None:
        """Form 8-K is due within four business days. A large lag means the
        event date was probably taken from the wrong filing."""
        for x in ch11:
            assert 0 <= x.lag_days <= 10, f"{x.company}: {x.lag_days}d"

    def test_event_and_disclosure_dates_differ_where_expected(self, ch11) -> None:
        """If every lag were zero, event_date would just be the filing date --
        i.e. the petition dates were never actually parsed."""
        assert any(x.lag_days > 0 for x in ch11)

    def test_no_event_in_the_future(self, ch11) -> None:
        for x in ch11:
            assert x.event_date <= date.today()

    def test_as_of_before_event_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            LabelRecord(
                cik=1,
                company="Impossible Corp",
                event_type=EVENT_CHAPTER11,
                event_date=date(2025, 6, 1),
                as_of_date=date(2025, 5, 1),
            )


class TestLabelProvenance:
    def test_every_label_passed_all_three_checks(self, ch11) -> None:
        """Item code, Chapter 11 text, and own-petition subject. Each caught
        errors the others missed -- see data/labels/README.md."""
        for x in ch11:
            for signal in REQUIRED_SIGNALS:
                assert signal in x.verification, f"{x.company} missing {signal}"

    def test_every_label_cites_a_source_filing(self, ch11) -> None:
        for x in ch11:
            assert x.source_accession and x.source_url
            assert x.source_url.startswith("https://www.sec.gov/Archives/")

    def test_date_basis_is_recorded(self, ch11) -> None:
        for x in ch11:
            assert x.date_basis in {"petition_date_from_8k_text", "8k_filing_date_fallback"}

    def test_fallback_dating_is_the_minority(self, ch11) -> None:
        """The fallback can only understate lead time, but if most labels used
        it the lead-time metric would be systematically compressed."""
        parsed = sum(1 for x in ch11 if x.date_basis == "petition_date_from_8k_text")
        assert parsed > len(ch11) / 2


class TestLabelSetShape:
    def test_no_duplicate_ciks(self, ch11) -> None:
        """One event per filer. Related registrants filing the same case (QVC
        Group and Old QVC Group) would otherwise double-count one outcome."""
        ciks = [x.cik for x in ch11]
        assert len(ciks) == len(set(ciks))

    def test_cohorts_are_known(self, ch11) -> None:
        assert {x.cohort for x in ch11} <= COHORTS

    def test_cohort_windows_are_consistent(self, ch11) -> None:
        for x in ch11:
            if x.cohort == "recent_2025_2026":
                assert x.event_date >= date(2025, 1, 1)
            elif x.cohort == "prior_2021_2024":
                assert date(2021, 1, 1) <= x.event_date < date(2025, 1, 1)
            else:
                assert x.event_date < date(2021, 1, 1)

    def test_recent_cohort_is_represented(self, ch11) -> None:
        """Recency was prioritised in *discovery*, not forced in the label mix.

        The 2023-24 bankruptcy wave genuinely dominates the window, so asserting
        that 2025-26 outnumbers it would encode a wish rather than a policy and
        could only be satisfied by discarding real events.
        """
        recent = [x for x in ch11 if x.cohort == "recent_2025_2026"]
        assert len(recent) >= 15

    def test_historical_tail_is_capped_near_ten_percent(self, ch11) -> None:
        """Pre-2021 filings sit in a different XBRL-coverage and rate
        environment; the tail is kept small so eras are not mixed silently."""
        old = [x for x in ch11 if x.cohort == "historical_pre_2021"]
        assert 0 < len(old) <= round(0.12 * len(ch11))

    def test_universe_is_large_enough_to_report(self, ch11) -> None:
        """Statistical power is a gate (SPEC 7). Below ~100 positives a recall
        estimate carries a CI too wide to publish."""
        assert len(ch11) >= 100


class TestWatchlistConsistency:
    def test_every_labelled_filer_is_monitored(self, ch11) -> None:
        monitored = {e.cik for e in load_watchlist()}
        for x in ch11:
            assert x.cik in monitored, f"{x.company} labelled but not on the watchlist"

    def test_distress_cohort_matches_the_label_set(self, ch11) -> None:
        distress = {e.cik for e in by_cohort(load_watchlist(), COHORT_DISTRESS)}
        assert distress == {x.cik for x in ch11}

    def test_survivors_have_no_chapter11_label(self, ch11) -> None:
        survivors = {e.cik for e in by_cohort(load_watchlist(), COHORT_SURVIVOR)}
        assert survivors.isdisjoint({x.cik for x in ch11})

    def test_universe_has_both_classes(self) -> None:
        entries = load_watchlist()
        assert by_cohort(entries, COHORT_DISTRESS) and by_cohort(entries, COHORT_SURVIVOR)

    def test_no_duplicate_watchlist_entries(self) -> None:
        ciks = [e.cik for e in load_watchlist()]
        assert len(ciks) == len(set(ciks))


class TestVisibilityVersusGrading:
    """The asymmetry that makes a backtest valid.

    ``labels_visible_as_of`` filters to what was knowable and feeds the agent.
    ``outcome_within_horizon`` looks ahead on purpose and feeds only the grader.
    """

    LABELS = [
        LabelRecord(
            cik=1,
            company="Fails Later Inc",
            event_type=EVENT_CHAPTER11,
            event_date=date(2025, 6, 1),
            as_of_date=date(2025, 6, 3),
        )
    ]

    def test_label_is_invisible_before_disclosure(self) -> None:
        assert labels_visible_as_of(self.LABELS, date(2025, 6, 1)) == []
        assert labels_visible_as_of(self.LABELS, date(2025, 6, 3)) == []
        assert len(labels_visible_as_of(self.LABELS, date(2025, 6, 4))) == 1

    def test_grader_sees_the_future_event(self) -> None:
        hit = outcome_within_horizon(self.LABELS, 1, date(2025, 1, 1), horizon_days=365)
        assert hit is not None and hit.event_date == date(2025, 6, 1)

    def test_event_outside_horizon_is_not_counted(self) -> None:
        assert outcome_within_horizon(self.LABELS, 1, date(2025, 1, 1), horizon_days=30) is None

    def test_event_before_prediction_is_not_counted(self) -> None:
        """Predicting a bankruptcy that already happened is not a prediction."""
        assert outcome_within_horizon(self.LABELS, 1, date(2025, 7, 1), horizon_days=365) is None

    def test_horizon_boundary_is_inclusive_at_the_end(self) -> None:
        assert outcome_within_horizon(self.LABELS, 1, date(2024, 6, 1), horizon_days=365) is not None

    def test_unlabelled_filer_has_no_outcome(self) -> None:
        assert outcome_within_horizon(self.LABELS, 999, date(2025, 1, 1), horizon_days=365) is None

    def test_lead_time_is_positive_before_the_event(self) -> None:
        assert lead_time_days(date(2025, 1, 1), date(2025, 6, 1)) == 151
        assert lead_time_days(date(2025, 7, 1), date(2025, 6, 1)) < 0


class TestAaerPlaceholder:
    def test_aaer_set_is_empty_but_loadable(self) -> None:
        """The earnings-quality leg is gated behind the distress leg's
        calibration curve. An empty label set is the honest state; a fabricated
        one would defeat the project."""
        assert load_aaer_labels() == []
