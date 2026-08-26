"""Repeated group-aware random splits: is 0.977 the model, or the window?

The published GBM number came from one held-out fold -- the most recent window
in the panel. The walk-forward table shows why that matters:

    2022-07-01   logistic 0.7500   GBM 0.9082
    2025-01-01   logistic 0.9788   GBM 0.9914

*Every* method scores higher on later windows. Logistic gains 0.23 AUC across
the same six folds without changing at all, so a large part of "0.98" is the
window being easy, not the model being good. A single test fold cannot separate
those two explanations.

So this resamples: many random splits of the same panel, and reports the
distribution rather than a point. A model that is genuinely strong scores well
across draws; one that got a friendly split shows a wide spread with the
published number sitting at the top of it.

**Group-aware throughout.** The panel repeats each firm quarterly, so a random
row split puts the same company on both sides and every score inflates by
roughly 0.09. Splitting is by firm, always -- that is not a refinement, it is
the difference between a measurement and a fiction.

    python -m models.run_random_splits --repeats 30
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys

from models.panel import feature_names, load_panel

#: Capacity-constrained on purpose. The unconstrained model scored 0.99 on
#: train and was memorising; these are the parameters that raised *test* AUC.
PARAMS = {
    "n_estimators": 150, "learning_rate": 0.05, "num_leaves": 3,
    "min_child_samples": 200, "subsample": 0.6, "subsample_freq": 1,
    "colsample_bytree": 0.4, "reg_lambda": 50.0,
}

TEST_SHARE = 0.25


def auc(y: list[int], s: list[float]) -> float | None:
    pos = [a for a, b in zip(s, y, strict=True) if b == 1]
    neg = [a for a, b in zip(s, y, strict=True) if b == 0]
    if not pos or not neg:
        return None
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def split_by_firm(rows, rng) -> tuple[list, list]:
    """Hold out whole firms, never rows. The cardinal split rule here."""
    firms = sorted({r.cik for r in rows})
    rng.shuffle(firms)
    cut = int(len(firms) * (1 - TEST_SHARE))
    train_firms = set(firms[:cut])
    train = [r for r in rows if r.cik in train_firms]
    test = [r for r in rows if r.cik not in train_firms]
    return train, test


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--panel", default="data/cache/panel.csv")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    import lightgbm as lgb
    import numpy as np

    rows = load_panel(args.panel)
    names = feature_names()
    firms = len({r.cik for r in rows})
    print("Repeated group-aware random splits -- is the number the model or the window?")
    print(f"panel {len(rows)} rows, {firms} firms, "
          f"{sum(int(r.label) for r in rows)} positives")
    print(f"{args.repeats} draws, {TEST_SHARE:.0%} of firms held out each time\n")

    scores, leaks, sizes, gaps = [], [], [], []
    rng = random.Random(args.seed)
    for i in range(args.repeats):
        train, test = split_by_firm(rows, rng)
        overlap = {r.cik for r in train} & {r.cik for r in test}
        leaks.append(len(overlap))
        ytr = [int(r.label) for r in train]
        yte = [int(r.label) for r in test]
        if not (0 < sum(yte) < len(yte)) or not (0 < sum(ytr) < len(ytr)):
            continue
        xtr = np.array([[r.features.get(n, 0.0) for n in names] for r in train], dtype=float)
        xte = np.array([[r.features.get(n, 0.0) for n in names] for r in test], dtype=float)
        model = lgb.LGBMClassifier(objective="binary", class_weight="balanced",
                                   verbose=-1, random_state=args.seed + i, **PARAMS)
        model.fit(xtr, ytr)
        a = auc(yte, list(model.predict_proba(xte)[:, 1]))
        # Logistic on the identical split. The level of both may be window
        # dependent; whether the *gap* survives resampling is the question the
        # single fold could not answer.
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        lr = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced"),
        )
        lr.fit(np.nan_to_num(xtr), ytr)
        b = auc(yte, list(lr.predict_proba(np.nan_to_num(xte))[:, 1]))
        if a is None:
            continue
        if b is not None:
            gaps.append(a - b)
        scores.append(a)
        sizes.append((len(test), sum(yte)))
        print(f"  draw {i + 1:>2}  test {len(test):>5} rows "
              f"({sum(yte):>3} pos)   AUC {a:.4f}")

    if not scores:
        print("no usable draws", file=sys.stderr)
        return 1
    scores.sort()
    lo = scores[int(0.025 * len(scores))]
    hi = scores[min(len(scores) - 1, int(0.975 * len(scores)))]
    print(f"\n  draws scored     : {len(scores)}")
    print(f"  firm leakage     : {max(leaks)} (must be 0)")
    print(f"  mean AUC         : {statistics.mean(scores):.4f}")
    print(f"  median           : {statistics.median(scores):.4f}")
    print(f"  spread           : {scores[0]:.4f} to {scores[-1]:.4f}")
    print(f"  95% of draws     : [{lo:.4f}, {hi:.4f}]")
    print("  published figure : 0.9768  (one late window)")
    above = sum(1 for s in scores if s >= 0.9768)
    print(f"  draws >= 0.9768  : {above}/{len(scores)} ({above / len(scores):.0%})")
    if gaps:
        wins = sum(1 for g in gaps if g > 0)
        print("\n  GBM minus logistic, same splits")
        print(f"    mean gap       : {statistics.mean(gaps):+.4f}")
        print(f"    GBM better in  : {wins}/{len(gaps)} draws")
        print(f"    range          : {min(gaps):+.4f} to {max(gaps):+.4f}")
    print(
        "\n  Read the spread, not the mean. If the published number sits near the\n"
        "  top of this range it was a friendly window; if it sits in the middle,\n"
        "  the model is doing the work."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
