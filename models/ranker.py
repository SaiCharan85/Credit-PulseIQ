"""The GBM as a *ranker* in the serving path -- and only as a ranker.

Two models, two jobs, and the split is not a compromise. Measured over 25
group-aware random splits of the panel:

    GBM,   whole firms held out    0.9187   95% of draws [0.8885, 0.9394]
    agent, firms resampled         0.9651   95% of draws [0.9337, 0.9874]

They tie on discrimination. What separates them is what else they produce. The
GBM emits one number and cannot say why; the agent cites every figure back to
a filing. So the GBM sorts the queue and the agent writes the memo, and neither
is asked to do the other's work.

**Ranked, never calibrated.** The score is for ordering filers against each
other. It is not a probability of default, and the reason is measurable: the
same model scored 0.9082 on a 2022 window and 0.9914 on a 2025 one, and the
published 0.9768 came from the easiest window in the panel -- outside the
entire range of 25 random draws. The *ordering* survives resampling; the level
does not. Serving the level as a probability would be selling the part that
does not generalise.

**The disagreement is the product.** A filer the statistics rank high while the
agent reads healthy is the most interesting row on any screen -- either the
model is seeing something in the covariates the narrative missed, or the agent
has context the features cannot encode. Both are worth a human's attention, so
the two scores are shown side by side and their disagreement is surfaced rather
than reconciled away.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

MODEL_PATH = Path("data/cache/ranker.txt")
META_PATH = Path("data/cache/ranker.json")

#: Percentile gap at which the two views are called out as disagreeing. Wide on
#: purpose: flagging every small difference would train readers to ignore it.
DISAGREEMENT_PCT = 40

_model: Any = None
_meta: dict[str, Any] | None = None


@dataclass
class Ranking:
    """Where this filer sits against the population, and how sure we are."""

    score: float
    percentile: float
    #: The window the model was trained on, so a reader can see how stale it is.
    trained_through: str = ""
    #: The honest resampled figure, not the single-window one.
    resampled_auc: float = 0.0
    resampled_range: tuple[float, float] = (0.0, 0.0)


def available() -> bool:
    """Whether a trained ranker is on disk. The app runs fine without one."""
    return MODEL_PATH.exists() and META_PATH.exists()


def _load() -> tuple[Any, dict[str, Any]] | tuple[None, None]:
    global _model, _meta
    if _model is not None and _meta is not None:
        return _model, _meta
    if not available():
        return None, None
    try:
        import lightgbm as lgb

        _model = lgb.Booster(model_file=str(MODEL_PATH))
        _meta = json.loads(META_PATH.read_text(encoding="utf8"))
    except Exception:  # noqa: BLE001 - a missing ranker degrades, never breaks
        return None, None
    return _model, _meta


def rank(cik: int, as_of: date, facts: Any) -> Ranking | None:
    """Score one filer from the same as-of view the agent reads.

    Returns None when no ranker is trained, when the filer has too little
    history to build a feature row, or when anything at all goes wrong. A
    missing rank costs a column; a raised exception would cost the assessment.
    """
    model, meta = _load()
    if model is None or meta is None:
        return None
    try:
        from models.panel import build_firm_rows, feature_names

        rows = build_firm_rows(facts, cik, [], [as_of])
        if not rows:
            return None
        names = meta.get("features") or feature_names()
        x = [[rows[0].features.get(n, 0.0) for n in names]]
        score = float(model.predict(x)[0])
    except Exception:  # noqa: BLE001
        return None

    # Percentile against the training distribution, because a raw margin means
    # nothing to a reader. "Above 94% of filers" is a statement they can use.
    grid = meta.get("score_grid") or []
    pct = 100.0 * sum(1 for g in grid if g <= score) / len(grid) if grid else 0.0
    lo, hi = meta.get("resampled_range", [0.0, 0.0])
    return Ranking(
        score=round(score, 4),
        percentile=round(pct, 1),
        trained_through=str(meta.get("trained_through", "")),
        resampled_auc=float(meta.get("resampled_auc", 0.0)),
        resampled_range=(float(lo), float(hi)),
    )


#: Where each signal typically sits on the ranked percentile, so a disagreement
#: can be detected without pretending the two scales are the same thing.
SIGNAL_PERCENTILE = {
    "healthy": 20.0,
    "watch": 45.0,
    "elevated_risk": 70.0,
    "severe_risk": 90.0,
}


def disagreement(signal: str, percentile: float) -> str:
    """A one-line note when the ranker and the agent tell different stories.

    Deliberately not a reconciliation. Neither view is authoritative -- they
    tie on discrimination -- so the honest move is to say they differ and let
    a person look, rather than to average two numbers that measure different
    things.
    """
    expected = SIGNAL_PERCENTILE.get(signal)
    if expected is None:
        return ""
    gap = percentile - expected
    if abs(gap) < DISAGREEMENT_PCT:
        return ""
    if gap > 0:
        return (
            f"The statistical ranker places this filer above {percentile:.0f}% of "
            f"the population while the investigation reads '{signal.replace('_', ' ')}'. "
            "The covariates are seeing something the narrative did not surface -- "
            "worth a look before relying on either."
        )
    return (
        f"The investigation reads '{signal.replace('_', ' ')}' while the ranker "
        f"places this filer at only the {percentile:.0f}th percentile. The agent "
        "may be reading filing language the covariates cannot encode."
    )
