"""Calibrate the agent, and fuse it with the hazard model (SPEC 7).

Two separate defects, two separate remedies, and it matters not to confuse
them:

* The agent's *reliability* is bad -- it under-calls distress everywhere below
  0.8, which is what produces false-confidence misses. Platt fixes that, and
  because it is strictly monotonic it leaves AUC untouched.
* The agent's *ranking* trails the hazard model. Only new information can move
  that, so calibration cannot help and fusion is the honest attempt.

**The fold discipline is the whole result.** Fitting a calibrator or a fusion
weight on the test predictions and then reporting on those same predictions is
test-set tuning; the numbers would be optimistic by an unknown amount and the
backtest would be worthless. So the graded cases are split *temporally*: the
earlier part fits, the later part scores, and the later part is never looked at
while choosing anything.

A parameter-free ensemble is reported alongside the fitted one. It has nothing
to overfit with, so where the two agree the gain is real rather than borrowed
from the fold.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from datetime import date
from pathlib import Path

from models.calibration import PlattCalibrator
from models.hazard import (
    AltmanBaseline,
    HazardBaseline,
    expected_calibration_error,
    roc_auc,
)
from models.panel import load_panel, split_by_date


def _logit(p: float, eps: float = 1e-6) -> float:
    p = max(eps, min(1.0 - eps, p))
    return math.log(p / (1.0 - p))


def _ranks(xs: list[float]) -> list[float]:
    """Fractional ranks in [0, 1]. Ties share the average rank."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    out = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0
        for k in range(i, j + 1):
            out[order[k]] = avg / max(1, len(xs) - 1)
        i = j + 1
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", type=Path, default=Path("data/cache/l3_partial_grade.csv"))
    ap.add_argument("--panel", type=Path, default=Path("data/cache/panel.csv"))
    ap.add_argument("--cutoff", type=date.fromisoformat, default=date(2024, 6, 1))
    ap.add_argument(
        "--fold",
        type=float,
        default=0.5,
        help="fraction of graded cases (earliest first) used to fit; rest scores",
    )
    args = ap.parse_args(argv)

    agent_rows = list(csv.DictReader(args.results.open(encoding="utf8")))
    if not agent_rows:
        print(f"no graded cases in {args.results}", file=sys.stderr)
        return 1
    if any(r.get("baseline_seen") == "1" for r in agent_rows):
        print("REFUSING: these predictions saw the hazard score; fusion would be circular", file=sys.stderr)
        return 2

    panel = load_panel(args.panel)
    train, test = split_by_date(panel, args.cutoff)
    by_key = {(r.cik, r.observation_date): r for r in test}

    hazard = HazardBaseline().fit(train)
    altman = AltmanBaseline()

    matched = []
    for r in agent_rows:
        key = (int(r["cik"]), date.fromisoformat(r["as_of"]))
        row = by_key.get(key)
        if row is None:
            continue
        matched.append((key[1], row, float(r["risk_probability"]), int(r["label"])))
    matched.sort(key=lambda t: (t[0], t[1].cik))

    rows = [m[1] for m in matched]
    haz = hazard.predict_proba(rows)
    alt = altman.predict_proba(rows)
    agent = [m[2] for m in matched]
    y = [m[3] for m in matched]

    # Temporal split, so the fit fold never contains a period later than the
    # scoring fold. A random split would put adjacent quarters of one firm on
    # both sides and leak.
    cut = int(len(matched) * args.fold)
    while cut < len(matched) and matched[cut][0] == matched[cut - 1][0]:
        cut += 1  # never split a single observation date across folds

    def fold(xs, lo, hi):
        return xs[lo:hi]

    fit_y, ev_y = fold(y, 0, cut), fold(y, cut, len(y))
    if len(set(fit_y)) < 2 or len(set(ev_y)) < 2:
        print("a fold is single-class; adjust --fold", file=sys.stderr)
        return 3

    fit_a, ev_a = fold(agent, 0, cut), fold(agent, cut, len(y))
    fit_h, ev_h = fold(haz, 0, cut), fold(haz, cut, len(y))
    ev_alt = fold(alt, cut, len(y))

    print(
        f"fit fold: {len(fit_y)} cases ({sum(fit_y)} positives, "
        f"through {matched[cut - 1][0]})",
        file=sys.stderr,
    )
    print(
        f"eval fold: {len(ev_y)} cases ({sum(ev_y)} positives, "
        f"from {matched[cut][0]}) -- never used to fit anything",
        file=sys.stderr,
    )

    platt = PlattCalibrator().fit(fit_a, fit_y)
    ev_a_cal = platt.transform(ev_a)

    # Fitted fusion: logistic on the two logits. Two parameters plus an
    # intercept, fitted on the fit fold only.
    from sklearn.linear_model import LogisticRegression

    lr = LogisticRegression(C=1.0, max_iter=1000)
    lr.fit([[_logit(a), _logit(h)] for a, h in zip(fit_a, fit_h, strict=True)], fit_y)
    ev_fused = [
        float(p)
        for p in lr.predict_proba(
            [[_logit(a), _logit(h)] for a, h in zip(ev_a, ev_h, strict=True)]
        )[:, 1]
    ]

    # Parameter-free control: average of within-fold ranks. Nothing is fitted,
    # so it cannot borrow anything from the fit fold.
    ra, rh = _ranks(ev_a), _ranks(ev_h)
    ev_rank = [(x + z) / 2.0 for x, z in zip(ra, rh, strict=True)]

    arms = [
        ("agent raw", ev_a),
        ("agent + Platt", ev_a_cal),
        ("Altman Z''", ev_alt),
        ("hazard alone", ev_h),
        ("agent + hazard (fitted)", ev_fused),
        ("agent + hazard (rank avg, no fit)", ev_rank),
    ]
    print(f"\n=== eval fold: {len(ev_y)} cases, {sum(ev_y)} positives ===", file=sys.stderr)
    print(f"{'arm':<36} {'AUC':>7} {'ECE':>7}", file=sys.stderr)
    for name, p in arms:
        ece = expected_calibration_error(ev_y, p)
        print(f"{name:<36} {roc_auc(ev_y, p):>7.4f} {ece:>7.4f}", file=sys.stderr)

    print(
        "\nECE for the rank-average arm is meaningless (ranks are not "
        "probabilities); read only its AUC.",
        file=sys.stderr,
    )
    print(
        f"fusion weights on the fit fold: agent {lr.coef_[0][0]:+.3f}, "
        f"hazard {lr.coef_[0][1]:+.3f}. A near-zero agent weight means the "
        "hazard model already contains what the agent found.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
