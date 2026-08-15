"""L0 -- panel construction and baseline evaluation metrics.

The panel is where a lookahead leak would be invisible and fatal, so its as-of
discipline is tested directly rather than assumed from the layers beneath it.
Metric implementations are checked against hand-computed values because these
numbers go in the README.
"""

from __future__ import annotations

from datetime import date

import pytest

from data.distress_events import TIER_DEFAULT, TIER_STRESS, DistressEvent
from evals.conftest import DISTRESSED, make_facts
from models.hazard import (
    AltmanBaseline,
    HazardBaseline,
    brier_score,
    evaluate,
    expected_calibration_error,
    precision_recall_at_k,
    roc_auc,
)
from models.panel import (
    MISSING_SUFFIX,
    build_firm_rows,
    feature_names,
    observation_dates,
    split_by_date,
)

FY2022 = date(2022, 12, 31)
FY2023 = date(2023, 12, 31)


def facts_two_years(cik: int = 1, filed_2023=date(2024, 2, 20)):
    return make_facts(DISTRESSED, period_end=FY2022, filed=date(2023, 2, 20), cik=cik) + make_facts(
        DISTRESSED, period_end=FY2023, filed=filed_2023, cik=cik
    )


def terminal(cik: int, day: date) -> DistressEvent:
    return DistressEvent(
        cik=cik,
        tier=TIER_DEFAULT,
        signal="chapter11_petition",
        event_date=day,
        as_of_date=day,
        source_form="8-K",
        source_accession="acc",
    )


class TestObservationGrid:
    def test_quarterly_grid(self) -> None:
        grid = observation_dates(date(2024, 1, 1), date(2024, 12, 31), months=3)
        assert grid == [date(2024, 1, 1), date(2024, 4, 1), date(2024, 7, 1), date(2024, 10, 1)]

    def test_annual_grid(self) -> None:
        grid = observation_dates(date(2023, 1, 1), date(2025, 1, 1), months=12)
        assert grid == [date(2023, 1, 1), date(2024, 1, 1), date(2025, 1, 1)]


class TestPanelAsOfDiscipline:
    def test_features_never_use_future_filings(self) -> None:
        """Standing at 2024-01-01, the FY2023 10-K filed in February is invisible,
        so the row must be built from FY2022."""
        rows = build_firm_rows(facts_two_years(), 1, [], [date(2024, 1, 1)])
        assert len(rows) == 1
        assert rows[0].latest_period_end == FY2022

    def test_later_observation_sees_the_new_filing(self) -> None:
        rows = build_firm_rows(facts_two_years(), 1, [], [date(2024, 6, 1)])
        assert rows[0].latest_period_end == FY2023

    def test_no_rows_before_any_filing_exists(self) -> None:
        assert build_firm_rows(facts_two_years(), 1, [], [date(2020, 1, 1)]) == []

    def test_rows_after_the_event_are_dropped(self) -> None:
        """Predicting a bankruptcy that already happened is not prediction."""
        events = [terminal(1, date(2024, 5, 1))]
        grid = [date(2024, 1, 1), date(2024, 4, 1), date(2024, 7, 1), date(2024, 10, 1)]
        rows = build_firm_rows(facts_two_years(), 1, events, grid)
        assert [r.observation_date for r in rows] == [date(2024, 1, 1), date(2024, 4, 1)]


class TestPanelLabels:
    EVENTS = [terminal(1, date(2024, 9, 1))]

    def test_label_is_one_inside_the_horizon(self) -> None:
        rows = build_firm_rows(facts_two_years(), 1, self.EVENTS, [date(2024, 1, 1)], horizon_days=365)
        assert rows[0].label == 1
        assert rows[0].days_to_event == 244

    def test_label_is_zero_outside_the_horizon(self) -> None:
        rows = build_firm_rows(facts_two_years(), 1, self.EVENTS, [date(2024, 1, 1)], horizon_days=90)
        assert rows[0].label == 0
        assert rows[0].days_to_event is None

    def test_non_terminal_events_do_not_set_the_label(self) -> None:
        """The hazard target is default, not any ladder event."""
        stress = DistressEvent(
            cik=1,
            tier=TIER_STRESS,
            signal="listing_rule_failure",
            event_date=date(2024, 6, 1),
            as_of_date=date(2024, 6, 1),
            source_form="8-K",
            source_accession="a",
        )
        rows = build_firm_rows(facts_two_years(), 1, [stress], [date(2024, 1, 1)])
        assert rows[0].label == 0

    def test_other_firms_events_are_ignored(self) -> None:
        rows = build_firm_rows(facts_two_years(cik=2), 2, self.EVENTS, [date(2024, 1, 1)])
        assert rows[0].label == 0


class TestMissingnessIsAFeature:
    """Data here is MNAR -- values go missing *because* of distress. Imputing
    them would erase the signal, so absence is encoded explicitly."""

    def test_indicator_present_for_every_feature(self) -> None:
        rows = build_firm_rows(facts_two_years(), 1, [], [date(2024, 6, 1)])
        names = feature_names()
        for name in names:
            assert name in rows[0].features

    def test_indicator_set_when_metric_undefined(self) -> None:
        values = {k: v for k, v in DISTRESSED.items() if k != "total_liabilities"}
        facts = make_facts(values, period_end=FY2023, filed=date(2024, 2, 20))
        rows = build_firm_rows(facts, 1, [], [date(2024, 6, 1)])
        assert rows[0].missing["liabilities_to_assets"]
        assert rows[0].features[f"liabilities_to_assets{MISSING_SUFFIX}"] == 1.0

    def test_missing_value_is_zero_not_imputed(self) -> None:
        values = {k: v for k, v in DISTRESSED.items() if k != "total_liabilities"}
        facts = make_facts(values, period_end=FY2023, filed=date(2024, 2, 20))
        rows = build_firm_rows(facts, 1, [], [date(2024, 6, 1)])
        assert rows[0].features["liabilities_to_assets"] == 0.0

    def test_present_metric_clears_the_indicator(self) -> None:
        rows = build_firm_rows(facts_two_years(), 1, [], [date(2024, 6, 1)])
        assert not rows[0].missing["current_ratio"]
        assert rows[0].features[f"current_ratio{MISSING_SUFFIX}"] == 0.0

    def test_feature_names_pair_each_metric_with_its_indicator(self) -> None:
        names = feature_names()
        bases = [n for n in names if not n.endswith(MISSING_SUFFIX)]
        for base in bases:
            assert f"{base}{MISSING_SUFFIX}" in names


class TestTemporalSplit:
    def test_split_is_by_date_not_random(self) -> None:
        rows = build_firm_rows(
            facts_two_years(), 1, [], observation_dates(date(2024, 1, 1), date(2024, 12, 31))
        )
        train, test = split_by_date(rows, date(2024, 7, 1))
        assert all(r.observation_date < date(2024, 7, 1) for r in train)
        assert all(r.observation_date >= date(2024, 7, 1) for r in test)
        assert len(train) + len(test) == len(rows)


class TestMetrics:
    def test_auc_perfect_separation(self) -> None:
        assert roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)

    def test_auc_inverted(self) -> None:
        assert roc_auc([1, 1, 0, 0], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(0.0)

    def test_auc_all_ties_is_a_coin_flip(self) -> None:
        assert roc_auc([0, 1, 0, 1], [0.5] * 4) == pytest.approx(0.5)

    def test_auc_undefined_without_both_classes(self) -> None:
        assert roc_auc([1, 1], [0.2, 0.8]) is None

    def test_auc_partial_ordering(self) -> None:
        # one positive ranked below one negative out of 2x2 pairs -> 0.75
        assert roc_auc([0, 1, 0, 1], [0.1, 0.2, 0.3, 0.9]) == pytest.approx(0.75)

    def test_brier_perfect_and_worst(self) -> None:
        assert brier_score([1, 0], [1.0, 0.0]) == pytest.approx(0.0)
        assert brier_score([1, 0], [0.0, 1.0]) == pytest.approx(1.0)

    def test_ece_zero_when_calibrated(self) -> None:
        y = [1] * 5 + [0] * 5
        p = [0.95] * 5 + [0.05] * 5
        assert expected_calibration_error(y, p) == pytest.approx(0.05, abs=1e-9)

    def test_ece_large_when_overconfident(self) -> None:
        """The false-confidence direction: certain, and wrong."""
        assert expected_calibration_error([0] * 10, [0.99] * 10) == pytest.approx(0.99)

    def test_precision_at_k(self) -> None:
        p, r = precision_recall_at_k([1, 1, 0, 0], [0.9, 0.8, 0.7, 0.6], ks=(2, 4))
        assert p[2] == pytest.approx(1.0)
        assert p[4] == pytest.approx(0.5)
        assert r[2] == pytest.approx(1.0)

    def test_k_larger_than_sample_is_skipped(self) -> None:
        p, _ = precision_recall_at_k([1, 0], [0.9, 0.1], ks=(10,))
        assert p == {}

    def test_evaluate_reports_base_rate(self) -> None:
        m = evaluate([1, 0, 0, 0], [0.9, 0.1, 0.2, 0.3])
        assert m.base_rate == pytest.approx(0.25)
        assert m.positives == 1
        assert "AUC" in m.summary()


class TestBaselines:
    def panel(self):
        rows = []
        for cik in range(1, 21):
            healthy = cik % 2 == 0
            values = dict(DISTRESSED)
            if healthy:
                values.update(
                    {"total_liabilities": 400.0, "current_assets": 900.0,
                     "current_liabilities": 300.0, "net_income": 150.0,
                     "retained_earnings": 400.0, "equity": 600.0}
                )
            facts = make_facts(values, period_end=FY2023, filed=date(2024, 2, 20), cik=cik)
            events = [] if healthy else [terminal(cik, date(2024, 9, 1))]
            rows += build_firm_rows(facts, cik, events, [date(2024, 6, 1)])
        return rows

    def test_altman_ranks_distress_higher(self) -> None:
        """Z'' is inverted: a lower score must produce a higher risk rank."""
        rows = self.panel()
        scores = AltmanBaseline().predict_proba(rows)
        distressed = [s for s, r in zip(scores, rows, strict=True) if r.label == 1]
        healthy = [s for s, r in zip(scores, rows, strict=True) if r.label == 0]
        assert min(distressed) > max(healthy)

    def test_altman_is_neutral_when_score_missing(self) -> None:
        values = {k: v for k, v in DISTRESSED.items() if k != "total_liabilities"}
        facts = make_facts(values, period_end=FY2023, filed=date(2024, 2, 20))
        rows = build_firm_rows(facts, 1, [], [date(2024, 6, 1)])
        assert AltmanBaseline().predict_proba(rows) == [0.5]

    def test_hazard_fits_and_discriminates(self) -> None:
        rows = self.panel()
        model = HazardBaseline().fit(rows)
        assert model.evaluate(rows).auc == pytest.approx(1.0)

    def test_hazard_refuses_single_class_training(self) -> None:
        rows = build_firm_rows(facts_two_years(), 1, [], [date(2024, 6, 1)])
        with pytest.raises(ValueError):
            HazardBaseline().fit(rows)

    def test_hazard_requires_fitting_before_predicting(self) -> None:
        with pytest.raises(RuntimeError):
            HazardBaseline().predict_proba(self.panel())

    def test_coefficients_are_named_and_ordered(self) -> None:
        model = HazardBaseline().fit(self.panel())
        coefs = model.coefficients()
        assert set(coefs) == set(feature_names())
        values = [abs(v) for v in coefs.values()]
        assert values == sorted(values, reverse=True)
