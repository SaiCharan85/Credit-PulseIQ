"""L0 -- the severity ladder, bankruptcy outcomes, and the subsidiary check."""

from __future__ import annotations

from datetime import date

import pytest

from data.discover import registrant_is_debtor
from data.distress_events import (
    SEVERITY,
    TIER_DEFAULT,
    TIER_EARLY_WARNING,
    TIER_NEAR_DEFAULT,
    TIER_STRESS,
    DistressEvent,
    default_tier_events,
    events_visible_as_of,
    extract_events,
    first_event_at_tier,
    load_events,
    merged_event_set,
    worst_tier_within,
)
from data.labels import EVENT_CHAPTER11, LabelRecord, load_chapter11_labels
from data.outcomes import (
    NON_SURVIVOR,
    OUTCOME_DEREGISTERED,
    OUTCOME_EMERGED,
    OUTCOME_IN_PROCESS,
    OUTCOME_WENT_DARK,
    classify,
    load_outcomes,
)


def filing(form, filing_date, items="", accession="acc", doc="a.htm"):
    return {
        "form": form,
        "items": items,
        "filing_date": filing_date,
        "accession": accession,
        "primary_document": doc,
    }


class StubClient:
    def __init__(self, filings):
        self._filings = filings

    def filing_index(self, cik):
        return self._filings


def event(cik=1, tier=TIER_STRESS, signal="s", day=date(2024, 6, 1), noisy=False):
    return DistressEvent(
        cik=cik,
        tier=tier,
        signal=signal,
        event_date=day,
        as_of_date=day,
        source_form="8-K",
        source_accession="acc",
        noisy=noisy,
    )


class TestLadderExtraction:
    def test_item_codes_map_to_tiers(self) -> None:
        filings = [
            filing("8-K", date(2024, 1, 5), items="2.04"),
            filing("8-K", date(2024, 2, 5), items="3.01"),
            filing("8-K", date(2024, 3, 5), items="4.02"),
            filing("8-K", date(2024, 4, 5), items="4.01"),
        ]
        events = extract_events(StubClient(filings), 1)
        by_signal = {e.signal: e.tier for e in events}
        assert by_signal["debt_acceleration"] == TIER_NEAR_DEFAULT
        assert by_signal["listing_rule_failure"] == TIER_STRESS
        assert by_signal["restatement_non_reliance"] == TIER_STRESS
        assert by_signal["auditor_change"] == TIER_EARLY_WARNING

    def test_forms_map_to_tiers(self) -> None:
        filings = [
            filing("25-NSE", date(2024, 5, 1)),
            filing("NT 10-K", date(2024, 6, 1)),
            filing("NT 10-Q", date(2024, 7, 1)),
        ]
        by_signal = {e.signal: e.tier for e in extract_events(StubClient(filings), 1)}
        assert by_signal["delisting_by_exchange"] == TIER_NEAR_DEFAULT
        assert by_signal["late_annual_report"] == TIER_EARLY_WARNING
        assert by_signal["late_quarterly_report"] == TIER_EARLY_WARNING

    def test_item_103_is_not_a_ladder_signal(self) -> None:
        """The terminal tier comes from the verified label set, never from an
        item code -- that is the code filers miscode."""
        filings = [filing("8-K", date(2024, 1, 5), items="1.03")]
        assert extract_events(StubClient(filings), 1) == []

    def test_one_filing_can_raise_several_signals(self) -> None:
        filings = [filing("8-K", date(2024, 1, 5), items="2.04,3.01,9.01")]
        assert len(extract_events(StubClient(filings), 1)) == 2

    def test_since_filter_excludes_older_filings(self) -> None:
        filings = [filing("8-K", date(2014, 1, 5), items="3.01")]
        assert extract_events(StubClient(filings), 1, since=date(2015, 1, 1)) == []

    def test_routine_signals_are_marked_noisy(self) -> None:
        """Auditor changes and restructuring charges happen at healthy
        companies too, so they are flagged rather than weighted equally."""
        filings = [filing("8-K", date(2024, 1, 5), items="4.01")]
        assert extract_events(StubClient(filings), 1)[0].noisy

    def test_debt_acceleration_is_not_noisy(self) -> None:
        filings = [filing("8-K", date(2024, 1, 5), items="2.04")]
        assert not extract_events(StubClient(filings), 1)[0].noisy


class TestSeverityOrdering:
    def test_tiers_are_strictly_ordered(self) -> None:
        assert (
            SEVERITY[TIER_EARLY_WARNING]
            < SEVERITY[TIER_STRESS]
            < SEVERITY[TIER_NEAR_DEFAULT]
            < SEVERITY[TIER_DEFAULT]
        )

    def test_worst_tier_wins_over_earlier_milder_event(self) -> None:
        events = [
            event(tier=TIER_EARLY_WARNING, day=date(2024, 1, 1)),
            event(tier=TIER_NEAR_DEFAULT, day=date(2024, 6, 1)),
            event(tier=TIER_STRESS, day=date(2024, 3, 1)),
        ]
        worst = worst_tier_within(events, 1, date(2023, 1, 1), date(2025, 1, 1))
        assert worst.tier == TIER_NEAR_DEFAULT

    def test_min_severity_filters_out_weak_signals(self) -> None:
        events = [event(tier=TIER_EARLY_WARNING)]
        assert worst_tier_within(events, 1, date(2023, 1, 1), date(2025, 1, 1)) is not None
        assert (
            worst_tier_within(
                events, 1, date(2023, 1, 1), date(2025, 1, 1), min_severity=SEVERITY[TIER_STRESS]
            )
            is None
        )

    def test_noisy_events_can_be_excluded(self) -> None:
        events = [event(tier=TIER_EARLY_WARNING, noisy=True)]
        assert worst_tier_within(events, 1, date(2023, 1, 1), date(2025, 1, 1)) is not None
        assert (
            worst_tier_within(events, 1, date(2023, 1, 1), date(2025, 1, 1), exclude_noisy=True)
            is None
        )

    def test_window_boundaries(self) -> None:
        events = [event(day=date(2024, 6, 1))]
        assert worst_tier_within(events, 1, date(2024, 6, 1), date(2024, 12, 1)) is None  # exclusive
        assert worst_tier_within(events, 1, date(2024, 1, 1), date(2024, 6, 1)) is not None  # inclusive

    def test_first_event_at_tier_finds_earliest_escalation(self) -> None:
        events = [
            event(tier=TIER_STRESS, day=date(2024, 3, 1)),
            event(tier=TIER_NEAR_DEFAULT, day=date(2024, 9, 1)),
            event(tier=TIER_EARLY_WARNING, day=date(2023, 1, 1)),
        ]
        assert first_event_at_tier(events, 1, TIER_STRESS).event_date == date(2024, 3, 1)
        assert first_event_at_tier(events, 1, TIER_DEFAULT) is None


class TestLadderVisibility:
    def test_events_are_hidden_until_disclosed(self) -> None:
        events = [event(day=date(2024, 6, 1))]
        assert events_visible_as_of(events, date(2024, 6, 1)) == []
        assert len(events_visible_as_of(events, date(2024, 6, 2))) == 1


class TestTerminalTier:
    LABEL = LabelRecord(
        cik=7,
        company="Gone Inc",
        event_type=EVENT_CHAPTER11,
        event_date=date(2025, 3, 1),
        as_of_date=date(2025, 3, 3),
        source_accession="0001-25-000001",
    )

    def test_labels_become_default_tier_events(self) -> None:
        events = default_tier_events([self.LABEL])
        assert len(events) == 1
        assert events[0].tier == TIER_DEFAULT
        assert events[0].signal == "chapter11_petition"

    def test_terminal_event_keeps_both_dates(self) -> None:
        e = default_tier_events([self.LABEL])[0]
        assert e.event_date == date(2025, 3, 1)
        assert e.as_of_date == date(2025, 3, 3)

    def test_merged_set_contains_all_tiers(self) -> None:
        merged = merged_event_set([event(cik=7, tier=TIER_STRESS)], [self.LABEL])
        assert {e.tier for e in merged} == {TIER_STRESS, TIER_DEFAULT}


class TestOutcomeClassification:
    LABEL = LabelRecord(
        cik=1,
        company="Test Corp",
        event_type=EVENT_CHAPTER11,
        event_date=date(2024, 1, 15),
        as_of_date=date(2024, 1, 16),
    )
    TODAY = date(2026, 8, 15)

    def test_form_15_means_deregistered(self) -> None:
        filings = [filing("15-12B", date(2024, 6, 1))]
        assert classify(StubClient(filings), self.LABEL, self.TODAY).outcome == OUTCOME_DEREGISTERED

    def test_resumed_reporting_means_emerged(self) -> None:
        """A genuine filer that reorganises resumes 10-K/10-Q reporting."""
        filings = [filing("10-Q", date(2024, 6, 1)), filing("10-K", date(2025, 3, 1))]
        out = classify(StubClient(filings), self.LABEL, self.TODAY)
        assert out.outcome == OUTCOME_EMERGED
        assert out.survived

    def test_silence_means_went_dark(self) -> None:
        filings = [filing("8-K", date(2024, 3, 1))]
        assert classify(StubClient(filings), self.LABEL, self.TODAY).outcome == OUTCOME_WENT_DARK

    def test_recent_activity_means_in_process(self) -> None:
        filings = [filing("8-K", date(2026, 7, 1))]
        assert classify(StubClient(filings), self.LABEL, self.TODAY).outcome == OUTCOME_IN_PROCESS

    def test_deregistration_beats_resumed_reporting(self) -> None:
        """Filing 10-Ks then deregistering is still a death."""
        filings = [
            filing("10-Q", date(2024, 6, 1)),
            filing("10-K", date(2025, 3, 1)),
            filing("15-12B", date(2025, 6, 1)),
        ]
        assert classify(StubClient(filings), self.LABEL, self.TODAY).outcome == OUTCOME_DEREGISTERED

    def test_pre_petition_filings_are_ignored(self) -> None:
        filings = [filing("10-K", date(2023, 3, 1)), filing("10-Q", date(2023, 6, 1))]
        assert classify(StubClient(filings), self.LABEL, self.TODAY).outcome == OUTCOME_WENT_DARK

    def test_non_survivor_set(self) -> None:
        assert NON_SURVIVOR == {OUTCOME_DEREGISTERED, OUTCOME_WENT_DARK}


class TestRegistrantIsDebtor:
    """The fourth check, built from real filing text.

    A genuine filer joins itself to the debtors with a *conjunction*; a parent
    reporting a subsidiary's case uses an *appositive*.
    """

    SEARS = (
        'sears holdings corporation (the "company") and the subsidiaries of the company '
        'listed in exhibit 99.1 (collectively, the "debtors") filed voluntary petitions'
    )
    ENVIVA = (
        'on the petition date, the company and certain subsidiaries of the company '
        '(collectively, the "debtors") filed voluntary petitions for reorganization'
    )
    DIEBOLD = (
        "a pre-packaged chapter 11 plan of reorganization to be filed by the company "
        'and certain of its subsidiaries (collectively, the "debtors")'
    )
    FIRSTENERGY = (
        'each wholly owned subsidiaries of firstenergy corp. ("firstenergy"), '
        "filed voluntary petitions for bankruptcy protection under chapter 11"
    )
    NOVELION = (
        'aegerion pharmaceuticals, inc. (together, the "debtors"), each a subsidiary of '
        'novelion therapeutics inc. (the "company"), filed voluntary petitions'
    )
    RVL = (
        'rvl pharmaceuticals, inc. (the "debtors"), each an indirect subsidiary of '
        "rvl pharmaceuticals plc, filed voluntary petitions"
    )

    @pytest.mark.parametrize("text", [SEARS, ENVIVA, DIEBOLD])
    def test_registrant_among_debtors_is_kept(self, text: str) -> None:
        assert registrant_is_debtor(text)

    @pytest.mark.parametrize("text", [FIRSTENERGY, NOVELION, RVL])
    def test_subsidiary_only_filing_is_rejected(self, text: str) -> None:
        assert not registrant_is_debtor(text)

    def test_ambiguous_text_defaults_to_keeping(self) -> None:
        """Silently discarding a real bankruptcy is the costlier error: it is
        the positive class."""
        assert registrant_is_debtor("the debtors filed voluntary petitions under chapter 11")


class TestBuiltArtifacts:
    """The committed ladder and outcome files must stay coherent."""

    def test_every_label_has_a_terminal_event(self) -> None:
        events = load_events()
        if not events:
            pytest.skip("distress_events.csv not built")
        default_ciks = {e.cik for e in events if e.tier == TIER_DEFAULT}
        assert {x.cik for x in load_chapter11_labels()} == default_ciks

    def test_all_four_tiers_are_populated(self) -> None:
        events = load_events()
        if not events:
            pytest.skip("distress_events.csv not built")
        assert {e.tier for e in events} == set(SEVERITY)

    def test_outcomes_cover_every_label(self) -> None:
        outcomes = load_outcomes()
        if not outcomes:
            pytest.skip("chapter11_outcomes.csv not built")
        assert {o.cik for o in outcomes} == {x.cik for x in load_chapter11_labels()}

    def test_most_bankruptcies_are_not_survived(self) -> None:
        outcomes = load_outcomes()
        if not outcomes:
            pytest.skip("chapter11_outcomes.csv not built")
        non_survivors = [o for o in outcomes if o.outcome in NON_SURVIVOR]
        assert len(non_survivors) > len(outcomes) / 2
