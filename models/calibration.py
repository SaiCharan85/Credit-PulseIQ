"""Probability recalibration (SPEC 7, calibration).

Measuring ECE says a model is overconfident; it does not fix it. This module
fits a monotonic map from raw score to calibrated probability, so that a stated
0.8 means the event happens about 80% of the time.

Two methods, both order-preserving so **AUC is unchanged** -- recalibration
moves probabilities, never the ranking:

:class:`PlattCalibrator`
    Logistic regression on the score. Two parameters, so it is stable on small
    samples. The default here, because the calibration fold has ~100 positives.
:class:`IsotonicCalibrator`
    Non-parametric monotonic fit. More flexible, needs more data, and will
    happily overfit a small fold.

**The fold discipline is the whole point.** A calibrator fitted on the same
rows the model was trained on learns the model's training-set optimism and
reports calibration that does not hold out of sample. So the training window is
split again: the model fits on the earlier part, the calibrator on the later
part, and the test set stays untouched. Temporally, never randomly -- adjacent
quarters of one firm are near-identical.

One honest limit: the universe is enriched with bankrupt filers far above the
population base rate (Zmijewski 1984). Calibration here is to *this* base rate.
Absolute probabilities still do not transfer to the wider population; a prior
shift would be needed for that, and is not pretended here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from models.panel import PanelRow


def _clip01(p: float, eps: float = 1e-6) -> float:
    return max(eps, min(1.0 - eps, p))


def _logit(p: float) -> float:
    p = _clip01(p)
    return math.log(p / (1.0 - p))


@dataclass
class PlattCalibrator:
    """Logistic recalibration: sigmoid(a * logit(p) + b)."""

    a: float = 1.0
    b: float = 0.0
    fitted: bool = False
    name: str = "platt"

    def fit(self, scores: list[float], labels: list[int]) -> PlattCalibrator:
        from sklearn.linear_model import LogisticRegression

        if len(set(labels)) < 2:
            raise ValueError("calibration fold contains a single class")
        x = [[_logit(s)] for s in scores]
        model = LogisticRegression(C=1e6, max_iter=1000)  # near-unpenalised: 2 params
        model.fit(x, labels)
        self.a = float(model.coef_[0][0])
        self.b = float(model.intercept_[0])
        self.fitted = True
        return self

    def transform(self, scores: list[float]) -> list[float]:
        if not self.fitted:
            raise RuntimeError("calibrator is not fitted")
        return [1.0 / (1.0 + math.exp(-(self.a * _logit(s) + self.b))) for s in scores]


@dataclass
class IsotonicCalibrator:
    """Non-parametric monotonic recalibration."""

    fitted: bool = False
    name: str = "isotonic"
    _model: object = None

    def fit(self, scores: list[float], labels: list[int]) -> IsotonicCalibrator:
        from sklearn.isotonic import IsotonicRegression

        if len(set(labels)) < 2:
            raise ValueError("calibration fold contains a single class")
        model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        model.fit(scores, labels)
        self._model = model
        self.fitted = True
        return self

    def transform(self, scores: list[float]) -> list[float]:
        if not self.fitted:
            raise RuntimeError("calibrator is not fitted")
        return [float(p) for p in self._model.predict(scores)]


class IdentityCalibrator:
    """No-op, so "uncalibrated" is a first-class comparison arm."""

    name = "identity"
    fitted = True

    def fit(self, scores: list[float], labels: list[int]) -> IdentityCalibrator:
        return self

    def transform(self, scores: list[float]) -> list[float]:
        return list(scores)


def split_for_calibration(
    train: list[PanelRow], calibration_cutoff: date
) -> tuple[list[PanelRow], list[PanelRow]]:
    """Split the training window into a model fold and a calibration fold.

    Temporal, for the same reason the train/test split is: a firm's adjacent
    quarters are nearly identical and a random split would put the same
    observation on both sides.
    """
    fit_rows = [r for r in train if r.observation_date < calibration_cutoff]
    calib_rows = [r for r in train if r.observation_date >= calibration_cutoff]
    return fit_rows, calib_rows


def reliability_table(
    scores: list[float], labels: list[int], bins: int = 10
) -> list[dict[str, float | int]]:
    """Binned stated-probability vs observed frequency."""
    out: list[dict[str, float | int]] = []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, s in enumerate(scores) if s >= lo and (s < hi or b == bins - 1)]
        if not idx:
            continue
        out.append(
            {
                "bin_low": lo,
                "bin_high": hi,
                "n": len(idx),
                "stated": sum(scores[i] for i in idx) / len(idx),
                "observed": sum(labels[i] for i in idx) / len(idx),
            }
        )
    return out


def max_calibration_error(scores: list[float], labels: list[int], bins: int = 10) -> float | None:
    """Worst single-bin gap.

    Reported next to ECE because ECE averages, and an average can look
    acceptable while one bin is badly wrong. For credit distress the bin that
    matters is the confident one.
    """
    table = reliability_table(scores, labels, bins)
    if not table:
        return None
    return max(abs(row["stated"] - row["observed"]) for row in table)
