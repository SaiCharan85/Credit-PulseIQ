"""L0 -- probability recalibration.

Calibration is one of the four L3 metrics, and the reported ECE depends
entirely on this machinery being right. The properties pinned here are the ones
that make a calibrated number trustworthy: the map is monotonic (so it cannot
launder a ranking improvement), it is fitted on a fold the model never saw, and
it genuinely reduces error on overconfident input.
"""

from __future__ import annotations

import random
from datetime import date

import pytest

from models.calibration import (
    IdentityCalibrator,
    IsotonicCalibrator,
    PlattCalibrator,
    max_calibration_error,
    reliability_table,
    split_for_calibration,
)
from models.hazard import expected_calibration_error, roc_auc
from models.panel import PanelRow


def overconfident(n: int = 400, seed: int = 0) -> tuple[list[float], list[int]]:
    """Scores pushed towards the extremes relative to the true rate.

    The failure mode the hazard baseline actually shows: it states 0.09 where
    the event happens 0.3% of the time.
    """
    rng = random.Random(seed)
    scores: list[float] = []
    labels: list[int] = []
    for _ in range(n):
        true_p = rng.uniform(0.05, 0.95)
        labels.append(1 if rng.random() < true_p else 0)
        # Stated confidence is the truth pushed outward.
        stated = true_p**0.45 if true_p > 0.5 else true_p**2.2
        scores.append(min(0.999, max(0.001, stated)))
    return scores, labels


class TestPlatt:
    def test_reduces_calibration_error(self) -> None:
        scores, labels = overconfident()
        fitted = PlattCalibrator().fit(scores, labels)
        before = expected_calibration_error(labels, scores)
        after = expected_calibration_error(labels, fitted.transform(scores))
        assert after < before

    def test_preserves_ranking(self) -> None:
        """Monotonic by construction, so AUC cannot move. A calibrator that
        changed AUC would be smuggling in a model change."""
        scores, labels = overconfident()
        mapped = PlattCalibrator().fit(scores, labels).transform(scores)
        assert roc_auc(labels, mapped) == pytest.approx(roc_auc(labels, scores), abs=1e-9)

    def test_output_is_a_probability(self) -> None:
        scores, labels = overconfident()
        mapped = PlattCalibrator().fit(scores, labels).transform(scores)
        assert all(0.0 <= p <= 1.0 for p in mapped)

    def test_single_class_fold_is_refused(self) -> None:
        with pytest.raises(ValueError):
            PlattCalibrator().fit([0.1, 0.2, 0.3], [0, 0, 0])

    def test_transform_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError):
            PlattCalibrator().transform([0.5])


class TestIsotonic:
    def test_reduces_calibration_error(self) -> None:
        scores, labels = overconfident()
        fitted = IsotonicCalibrator().fit(scores, labels)
        before = expected_calibration_error(labels, scores)
        after = expected_calibration_error(labels, fitted.transform(scores))
        assert after < before

    def test_is_monotonic(self) -> None:
        scores, labels = overconfident()
        fitted = IsotonicCalibrator().fit(scores, labels)
        probe = [i / 100 for i in range(101)]
        mapped = fitted.transform(probe)
        assert all(b >= a - 1e-9 for a, b in zip(mapped[:-1], mapped[1:], strict=True))

    def test_single_class_fold_is_refused(self) -> None:
        with pytest.raises(ValueError):
            IsotonicCalibrator().fit([0.1, 0.9], [1, 1])


class TestIdentity:
    def test_returns_input_unchanged(self) -> None:
        """"Uncalibrated" is a first-class comparison arm, not an absence."""
        scores = [0.1, 0.5, 0.9]
        assert IdentityCalibrator().fit(scores, [0, 1, 1]).transform(scores) == scores


class TestFoldDiscipline:
    """A calibrator fitted on the model's own training rows learns its
    training-set optimism and reports calibration that does not hold."""

    ROWS = [
        PanelRow(cik=1, observation_date=date(2022, 1, 1), label=0),
        PanelRow(cik=1, observation_date=date(2023, 1, 1), label=0),
        PanelRow(cik=1, observation_date=date(2023, 9, 1), label=1),
        PanelRow(cik=1, observation_date=date(2024, 1, 1), label=1),
    ]

    def test_split_is_temporal(self) -> None:
        fit_rows, calib_rows = split_for_calibration(self.ROWS, date(2023, 6, 1))
        assert all(r.observation_date < date(2023, 6, 1) for r in fit_rows)
        assert all(r.observation_date >= date(2023, 6, 1) for r in calib_rows)

    def test_folds_are_disjoint_and_complete(self) -> None:
        fit_rows, calib_rows = split_for_calibration(self.ROWS, date(2023, 6, 1))
        assert len(fit_rows) + len(calib_rows) == len(self.ROWS)


class TestReliabilityReporting:
    def test_perfectly_calibrated_input_sits_on_the_diagonal(self) -> None:
        scores = [0.0] * 50 + [1.0] * 50
        labels = [0] * 50 + [1] * 50
        for row in reliability_table(scores, labels, bins=5):
            assert row["stated"] == pytest.approx(row["observed"], abs=1e-9)

    def test_max_error_catches_a_single_bad_bin(self) -> None:
        """ECE averages, so one badly wrong bin can hide inside an acceptable
        mean. For distress the confident bin is the one that matters."""
        scores = [0.05] * 99 + [0.95]
        labels = [0] * 99 + [0]
        assert expected_calibration_error(labels, scores) < 0.1
        assert max_calibration_error(scores, labels) == pytest.approx(0.95)

    def test_empty_input_is_none(self) -> None:
        assert max_calibration_error([], []) is None
        assert reliability_table([], []) == []
