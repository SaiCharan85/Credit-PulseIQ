"""Is the hazard baseline strong enough to be a fair comparison?

Every model in this project is ``LogisticRegression(C=1.0)`` -- the sklearn
default, never tuned, never a different family. The headline claim is that the
agent *ties* the hazard baseline at 0.965 against 0.966. If a properly
specified baseline scores higher, that sentence becomes "the agent loses to a
real baseline", and the result gets weaker rather than stronger.

That is the reason to run this. A baseline nobody tried to make strong is not a
baseline, it is a strawman, and the credibility of the whole comparison rests
on it being fair. This test can only hurt our own headline, which is precisely
why it should exist.

A linear model also cannot represent interactions. Distress plausibly involves
them -- thin liquidity matters far more when leverage is high than when it is
not -- and a logit can only add the two together. Gradient boosting captures
that and non-linear thresholds besides.

**Tuning happens on an inner validation split carved out of the training
window, never on the test fold.** The test fold is scored exactly once per
arm, at the end. Anything else would tune against the answer.
"""

from __future__ import annotations

import argparse
import random
import sys
from datetime import date
from pathlib import Path

from models.hazard import HazardBaseline, roc_auc
from models.panel import feature_names, load_panel, split_by_date

#: Small grid. Deliberately coarse: with a few hundred positives a fine search
#: fits the validation split rather than the problem.
GRID = (
    {"n_estimators": 200, "learning_rate": 0.05, "num_leaves": 15, "min_child_samples": 40},
    {"n_estimators": 400, "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 20},
    {"n_estimators": 200, "learning_rate": 0.10, "num_leaves": 31, "min_child_samples": 20},
    {"n_estimators": 600, "learning_rate": 0.03, "num_leaves": 15, "min_child_samples": 40},
    {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 10},
)


def matrix(rows, names):
    return [[r.features.get(n, 0.0) for n in names] for r in rows]


def fit_gbm(train, names, params):
    import lightgbm as lgb

    model = lgb.LGBMClassifier(
        objective="binary",
        class_weight="balanced",
        random_state=0,
        verbose=-1,
        **params,
    )
    model.fit(matrix(train, names), [r.label for r in train])
    return model


def paired_bootstrap(y, a, b, label, alpha=0.05, n_boot=6000):
    rng = random.Random(0)
    diffs = []
    n = len(y)
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        yy = [y[i] for i in idx]
        if len(set(yy)) < 2:
            continue
        diffs.append(roc_auc(yy, [a[i] for i in idx]) - roc_auc(yy, [b[i] for i in idx]))
    diffs.sort()
    lo, hi = diffs[int(alpha / 2 * len(diffs))], diffs[int((1 - alpha / 2) * len(diffs))]
    delta = roc_auc(y, a) - roc_auc(y, b)
    verdict = "excludes 0" if lo > 0 else ("excludes 0 (worse)" if hi < 0 else "INCLUDES 0")
    print(f"  {label}: {delta:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  {verdict}", file=sys.stderr)
    return delta, lo, hi


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--panel", type=Path, default=Path("data/cache/panel.csv"))
    ap.add_argument("--cutoff", type=date.fromisoformat, default=date(2024, 6, 1))
    ap.add_argument(
        "--inner-cutoff",
        type=date.fromisoformat,
        default=date(2023, 6, 1),
        help="splits the training window into fit and validation; test is untouched",
    )
    args = ap.parse_args(argv)

    rows = load_panel(args.panel)
    if not rows:
        print(f"no panel at {args.panel}", file=sys.stderr)
        return 1
    train, test = split_by_date(rows, args.cutoff)
    inner_fit, inner_val = split_by_date(train, args.inner_cutoff)
    names = feature_names()
    y_test = [r.label for r in test]
    print(
        f"train {len(train)} ({sum(r.label for r in train)} pos)"
        f" -> fit {len(inner_fit)} ({sum(r.label for r in inner_fit)} pos)"
        f" / val {len(inner_val)} ({sum(r.label for r in inner_val)} pos)",
        file=sys.stderr,
    )
    print(f"test {len(test)} ({sum(y_test)} pos) -- scored once per arm", file=sys.stderr)
    if sum(r.label for r in inner_val) < 20:
        print("validation fold too thin to select on", file=sys.stderr)
        return 2

    print("\n=== tuning on the inner validation split ===", file=sys.stderr)
    y_val = [r.label for r in inner_val]
    best, best_auc = None, -1.0
    for params in GRID:
        model = fit_gbm(inner_fit, names, params)
        auc = roc_auc(y_val, [float(p) for p in model.predict_proba(matrix(inner_val, names))[:, 1]])
        print(f"  {params} -> val AUC {auc:.4f}", file=sys.stderr)
        if auc > best_auc:
            best, best_auc = params, auc
    print(f"  chosen: {best}  (val {best_auc:.4f})", file=sys.stderr)

    # Refit on the full training window with the chosen shape, then score once.
    gbm = fit_gbm(train, names, best)
    p_gbm = [float(p) for p in gbm.predict_proba(matrix(test, names))[:, 1]]
    p_logit = HazardBaseline().fit(train).predict_proba(test)

    print("\n=== test fold, scored once ===", file=sys.stderr)
    print(f"  hazard, logistic C=1.0 (current baseline) : AUC {roc_auc(y_test, p_logit):.4f}",
          file=sys.stderr)
    print(f"  hazard, tuned LightGBM                    : AUC {roc_auc(y_test, p_gbm):.4f}",
          file=sys.stderr)
    paired_bootstrap(y_test, p_gbm, p_logit, "GBM - logistic  ")

    print(
        "\nIf the GBM is materially better, the published claim that the agent "
        "ties 'the hazard baseline' was measured against a weak one, and the "
        "comparison needs restating rather than defending.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
