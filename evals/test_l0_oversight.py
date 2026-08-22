"""L0: regulator and governance signals, and their as-of discipline.

The leakage risk here is sharper than elsewhere, because the event being
predicted is the same *kind* of event used as a predictor. A filer's own
restatement must never appear among the features that forecast it, and a
comment letter published after the prediction date must be invisible even
though its correspondence began before.
"""

from __future__ import annotations

from datetime import date

from data.oversight import (
    LOOKBACK_DAYS,
    OversightSignals,
    oversight_feature_names,
    oversight_signals,
)

AS_OF = date(2024, 6, 1)


def row(form: str, filed: date, items: str = "") -> dict:
    return {"form": form, "items": items, "filing_date": filed,
            "accession": "0001-24-1", "primary_document": "d.htm"}


class TestAsOfDiscipline:
    def test_a_letter_published_after_as_of_is_invisible(self) -> None:
        """Correspondence may predate publication; only publication is knowable."""
        s = oversight_signals([row("UPLOAD", date(2024, 8, 1))], AS_OF)
        assert s.comment_letters == 0

    def test_a_letter_published_on_the_day_counts(self) -> None:
        assert oversight_signals([row("UPLOAD", AS_OF)], AS_OF).comment_letters == 1

    def test_events_before_the_lookback_are_excluded(self) -> None:
        old = row("UPLOAD", AS_OF.replace(year=AS_OF.year - 5))
        assert oversight_signals([old], AS_OF).comment_letters == 0

    def test_the_lookback_window_is_two_years(self) -> None:
        from datetime import timedelta

        inside = row("UPLOAD", AS_OF - timedelta(days=LOOKBACK_DAYS - 1))
        outside = row("UPLOAD", AS_OF - timedelta(days=LOOKBACK_DAYS + 1))
        assert oversight_signals([inside], AS_OF).comment_letters == 1
        assert oversight_signals([outside], AS_OF).comment_letters == 0


class TestPriorRestatementsCannotLeak:
    def test_the_event_being_predicted_is_excluded(self) -> None:
        """A restatement on or after the prediction date is the outcome."""
        s = oversight_signals([], AS_OF, restatement_dates=[AS_OF, date(2024, 9, 1)])
        assert s.prior_restatements == 0

    def test_a_genuinely_prior_restatement_counts(self) -> None:
        s = oversight_signals([], AS_OF, restatement_dates=[date(2022, 3, 1)])
        assert s.prior_restatements == 1
        assert s.days_since_restatement == (AS_OF - date(2022, 3, 1)).days

    def test_repeat_offenders_are_counted(self) -> None:
        s = oversight_signals(
            [], AS_OF, restatement_dates=[date(2019, 1, 1), date(2022, 3, 1)]
        )
        assert s.prior_restatements == 2
        # recency measured from the most recent, not the first
        assert s.days_since_restatement == (AS_OF - date(2022, 3, 1)).days


class TestExtraction:
    def test_letters_replies_and_officer_events_are_separated(self) -> None:
        index = [
            row("UPLOAD", date(2024, 1, 5)),
            row("CORRESP", date(2024, 2, 5)),
            row("8-K", date(2024, 3, 5), "5.02"),
            row("8-K", date(2024, 3, 6), "2.02"),
        ]
        s = oversight_signals(index, AS_OF)
        assert (s.comment_letters, s.comment_replies, s.officer_events) == (1, 1, 1)

    def test_recency_uses_the_most_recent_event(self) -> None:
        index = [row("UPLOAD", date(2023, 1, 1)), row("UPLOAD", date(2024, 5, 1))]
        s = oversight_signals(index, AS_OF)
        assert s.days_since_comment_letter == (AS_OF - date(2024, 5, 1)).days


class TestFeatureVector:
    def test_never_happened_is_flagged_not_zero(self) -> None:
        """Zero days since an event and never having one are different facts."""
        f = OversightSignals().as_features()
        assert f["ovs_days_since_comment_letter"] == 0.0
        assert f["ovs_days_since_comment_letter__missing"] == 1.0

    def test_a_real_gap_is_unflagged(self) -> None:
        f = OversightSignals(days_since_comment_letter=30).as_features()
        assert f["ovs_days_since_comment_letter"] == 30.0
        assert f["ovs_days_since_comment_letter__missing"] == 0.0

    def test_names_are_stable_and_complete(self) -> None:
        names = oversight_feature_names()
        assert names == sorted(OversightSignals().as_features())
        assert "ovs_prior_restatements" in names
