"""L0: macro covariates and their point-in-time discipline.

Macro data is a quieter leakage route than filings, and a more dangerous one,
because the date column always looks right. A series pulled today and indexed
by observation date will happily hand a model the value for 2019 -- but if that
series gets revised, the number returned is not the number anyone had in 2019.
These tests pin the two defences: only non-revised market series are used, and
a lookup can never reach past its own as-of date.
"""

from __future__ import annotations

from datetime import date

from data.macro import (
    SERIES,
    MacroSeries,
    macro_feature_names,
    macro_features,
    parse_fred_csv,
)

CSV = """observation_date,BAMLH0A0HYM2
2024-01-02,3.50
2024-02-01,.
2024-03-01,4.00
2024-06-03,5.00
2024-09-02,9.99
"""


def series() -> MacroSeries:
    return parse_fred_csv(CSV)


class TestParsing:
    def test_missing_observations_are_dropped(self) -> None:
        """FRED writes '.' for non-trading days."""
        s = series()
        assert date(2024, 2, 1) not in s.dates
        assert len(s.dates) == 4

    def test_values_are_sorted_by_date(self) -> None:
        s = series()
        assert s.dates == sorted(s.dates)

    def test_a_garbage_payload_yields_an_empty_series(self) -> None:
        assert parse_fred_csv("not,a,series\nnope").values == []


class TestAsOfDiscipline:
    def test_a_future_observation_is_unreachable(self) -> None:
        """The 2024-09-02 value must not leak into a June prediction."""
        assert series().as_of(date(2024, 6, 10)) == 5.00

    def test_the_same_day_value_is_visible(self) -> None:
        """Markets publish same-day, unlike a filing."""
        assert series().as_of(date(2024, 6, 3)) == 5.00

    def test_before_the_series_starts_is_none(self) -> None:
        assert series().as_of(date(2023, 1, 1)) is None

    def test_a_stale_reading_is_none_not_a_stale_number(self) -> None:
        """Three months after the last print is not a current spread."""
        assert series().as_of(date(2024, 5, 30)) is None  # last print 2024-03-01
        assert series().as_of(date(2024, 3, 15)) == 4.00

    def test_an_empty_series_never_raises(self) -> None:
        assert MacroSeries("x", [], []).as_of(date(2024, 6, 1)) is None


class TestChange:
    def test_change_is_measured_backwards_only(self) -> None:
        s = series()
        # 2024-06-03 level 5.00; ~90 days earlier resolves to 2024-03-01 at 4.00
        assert s.change(date(2024, 6, 3), 90) == 1.00

    def test_change_is_none_when_either_end_is_missing(self) -> None:
        assert series().change(date(2024, 1, 2), 365) is None


class TestFeatureVector:
    def test_every_feature_has_a_missingness_flag(self) -> None:
        names = macro_feature_names()
        bases = [n for n in names if not n.endswith("__missing")]
        for b in bases:
            assert f"{b}__missing" in names

    def test_absent_series_are_zero_and_flagged_not_imputed(self) -> None:
        empty = {k: MacroSeries(k, [], []) for k in SERIES}
        feats = macro_features(empty, date(2024, 6, 1))
        assert feats["macro_credit_spread"] == 0.0
        assert feats["macro_credit_spread__missing"] == 1.0

    def test_a_present_series_is_unflagged(self) -> None:
        feats = macro_features({"credit_spread": series()}, date(2024, 6, 3))
        assert feats["macro_credit_spread"] == 5.00
        assert feats["macro_credit_spread__missing"] == 0.0

    def test_feature_order_is_stable(self) -> None:
        """The covariate vector is positional; a reordering silently
        mismatches a fitted model against its own coefficients."""
        assert macro_feature_names() == macro_feature_names()


class TestOnlyNonRevisedSeries:
    def test_no_revised_survey_series_are_configured(self) -> None:
        """UNRATE, GDP and friends are revised for months after publication.

        Reading today's vintage at a 2019 date hands the model a number nobody
        had in 2019, and the date column would look entirely correct.
        """
        revised = {"UNRATE", "GDP", "GDPC1", "INDPRO", "PAYEMS", "CPIAUCSL", "USREC"}
        assert not (set(SERIES.values()) & revised)
