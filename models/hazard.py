"""Discrete-time hazard baseline (Shumway 2001) and its evaluation metrics.

This is a **baseline the agent must beat**, not a deliverable in itself. Without
one, an L3 precision figure is uninterpretable: if the ReAct investigator cannot
outperform a 1980-vintage logit on the same as-of data, that is a finding worth
reporting honestly rather than hiding.

Why a hazard model rather than a classifier: firms are observed repeatedly until
they either fail or are censored, and Shumway showed that collapsing that to one
observation per firm biases the estimates. A discrete-time hazard is just logistic
regression over firm-period observations, which is what ``models/panel.py`` builds.

Two properties of this data drive the implementation:

* **Choice-based sample bias** (Zmijewski 1984). The universe is enriched with
  bankrupt firms far above the population base rate, so fitted intercepts do not
  transfer. Discrimination metrics (AUC, precision@k) are reported; absolute
  probabilities are not treated as population default rates.
* **Extreme imbalance.** Accuracy is meaningless here. AUC, precision@k, Brier
  score and calibration error are reported instead.

Deterministic given a seed. No LLM.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from models.panel import PanelRow, feature_names


@dataclass
class Metrics:
    """Discrimination and calibration for a set of predictions."""

    n: int = 0
    positives: int = 0
    base_rate: float = 0.0
    auc: float | None = None
    brier: float | None = None
    ece: float | None = None
    precision_at_k: dict[int, float] = field(default_factory=dict)
    recall_at_k: dict[int, float] = field(default_factory=dict)

    def summary(self) -> str:
        auc = f"{self.auc:.3f}" if self.auc is not None else "n/a"
        brier = f"{self.brier:.4f}" if self.brier is not None else "n/a"
        ece = f"{self.ece:.4f}" if self.ece is not None else "n/a"
        parts = [
            f"n={self.n}",
            f"positives={self.positives}",
            f"base_rate={self.base_rate:.4f}",
            f"AUC={auc}",
            f"Brier={brier}",
            f"ECE={ece}",
        ]
        for k in sorted(self.precision_at_k):
            parts.append(f"P@{k}={self.precision_at_k[k]:.3f}")
        return "  ".join(parts)


def roc_auc(y_true: list[int], y_score: list[float]) -> float | None:
    """AUC via the rank (Mann-Whitney U) identity, with tie correction.

    Implemented directly so the metric has no hidden library conventions --
    this number goes in the README.
    """
    pos = sum(y_true)
    neg = len(y_true) - pos
    if pos == 0 or neg == 0:
        return None
    order = sorted(range(len(y_score)), key=lambda i: y_score[i])
    ranks = [0.0] * len(y_score)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and y_score[order[j + 1]] == y_score[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # average rank, 1-indexed
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    rank_sum = sum(r for r, y in zip(ranks, y_true, strict=True) if y == 1)
    return (rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)


def brier_score(y_true: list[int], y_prob: list[float]) -> float | None:
    if not y_true:
        return None
    return sum((p - y) ** 2 for p, y in zip(y_prob, y_true, strict=True)) / len(y_true)


def expected_calibration_error(
    y_true: list[int], y_prob: list[float], bins: int = 10
) -> float | None:
    """ECE: mean |confidence - accuracy| across equal-width probability bins.

    Reported alongside the reliability curve. With few positives the bins are
    sparse, which is why SPEC 7 gates reporting on universe size.
    """
    if not y_true:
        return None
    total = 0.0
    n = len(y_true)
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, p in enumerate(y_prob) if (p >= lo and (p < hi or (b == bins - 1 and p <= hi)))]
        if not idx:
            continue
        conf = sum(y_prob[i] for i in idx) / len(idx)
        acc = sum(y_true[i] for i in idx) / len(idx)
        total += len(idx) / n * abs(conf - acc)
    return total


def precision_recall_at_k(
    y_true: list[int], y_score: list[float], ks: tuple[int, ...] = (10, 25, 50, 100)
) -> tuple[dict[int, float], dict[int, float]]:
    """Top-k precision/recall -- the analyst-relevant framing.

    A watch-list has finite capacity; what matters is how many of the top k
    flagged names actually failed.
    """
    order = sorted(range(len(y_score)), key=lambda i: -y_score[i])
    total_pos = sum(y_true)
    precision: dict[int, float] = {}
    recall: dict[int, float] = {}
    for k in ks:
        if k > len(order):
            continue
        hits = sum(y_true[i] for i in order[:k])
        precision[k] = hits / k
        recall[k] = hits / total_pos if total_pos else 0.0
    return precision, recall


def evaluate(y_true: list[int], y_prob: list[float]) -> Metrics:
    p_at_k, r_at_k = precision_recall_at_k(y_true, y_prob)
    return Metrics(
        n=len(y_true),
        positives=sum(y_true),
        base_rate=sum(y_true) / len(y_true) if y_true else 0.0,
        auc=roc_auc(y_true, y_prob),
        brier=brier_score(y_true, y_prob),
        ece=expected_calibration_error(y_true, y_prob),
        precision_at_k=p_at_k,
        recall_at_k=r_at_k,
    )


class HazardBaseline:
    """Discrete-time hazard: penalised logistic regression over firm-periods."""

    def __init__(self, c: float = 1.0, seed: int = 0, class_weight: str | None = "balanced"):
        self.c = c
        self.seed = seed
        self.class_weight = class_weight
        self.names = feature_names()
        self._model = None
        self._mean: list[float] = []
        self._scale: list[float] = []

    def _standardise(self, x: list[list[float]], fit: bool = False) -> list[list[float]]:
        if fit:
            n = len(x)
            cols = len(self.names)
            self._mean = [sum(row[j] for row in x) / n for j in range(cols)]
            self._scale = []
            for j in range(cols):
                var = sum((row[j] - self._mean[j]) ** 2 for row in x) / max(n - 1, 1)
                sd = math.sqrt(var)
                self._scale.append(sd if sd > 1e-9 else 1.0)
        return [
            [(row[j] - self._mean[j]) / self._scale[j] for j in range(len(self.names))] for row in x
        ]

    def fit(self, rows: list[PanelRow]) -> HazardBaseline:
        from sklearn.linear_model import LogisticRegression

        x = [r.vector(self.names) for r in rows]
        y = [r.label for r in rows]
        if len(set(y)) < 2:
            raise ValueError("training panel contains a single class")
        xs = self._standardise(x, fit=True)
        self._model = LogisticRegression(
            C=self.c,
            max_iter=2000,
            random_state=self.seed,
            class_weight=self.class_weight,
        )
        self._model.fit(xs, y)
        return self

    def predict_proba(self, rows: list[PanelRow]) -> list[float]:
        if self._model is None:
            raise RuntimeError("model is not fitted")
        xs = self._standardise([r.vector(self.names) for r in rows])
        return [float(p[1]) for p in self._model.predict_proba(xs)]

    def coefficients(self) -> dict[str, float]:
        """Standardised coefficients, largest absolute effect first.

        Interpretable by construction -- part of why the hazard model is the
        primary baseline rather than a black box.
        """
        if self._model is None:
            raise RuntimeError("model is not fitted")
        pairs = dict(zip(self.names, (float(c) for c in self._model.coef_[0]), strict=True))
        return dict(sorted(pairs.items(), key=lambda kv: -abs(kv[1])))

    def evaluate(self, rows: list[PanelRow]) -> Metrics:
        return evaluate([r.label for r in rows], self.predict_proba(rows))


class AltmanBaseline:
    """Tier-0 reference: rank by Altman Z'' alone, no fitting.

    The floor any model must clear. Z is inverted because a *lower* score means
    more distress, and the panel scores higher-is-riskier.
    """

    METRIC = "altman_z_double_prime"

    def fit(self, rows: list[PanelRow]) -> AltmanBaseline:
        return self

    def predict_proba(self, rows: list[PanelRow]) -> list[float]:
        out = []
        for r in rows:
            if r.missing.get(self.METRIC, True):
                out.append(0.5)  # no score: neutral rank, never a confident pass
            else:
                z = r.features.get(self.METRIC, 0.0)
                out.append(1.0 / (1.0 + math.exp(max(-30.0, min(30.0, z)))))
        return out

    def evaluate(self, rows: list[PanelRow]) -> Metrics:
        return evaluate([r.label for r in rows], self.predict_proba(rows))
