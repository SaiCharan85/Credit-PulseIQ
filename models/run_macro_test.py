"""Do macro covariates improve the distress baseline? (paired)

The hazard model reads firm ratios only. Corporate default is cyclical, and
Duffie, Saita and Wang (2007) find macro covariates add real explanatory power
over firm-level variables alone, so this is the obvious gap.

Only series with **full coverage of the panel window** are used, and that
exclusion is doing real work. The high-yield credit spread would be the most
informative covariate here, but the available download starts 2023-08-22 while
the panel starts 2019. A feature present throughout the test fold and absent
from most of training does not merely fit badly: its missingness indicator
becomes a proxy for "is this an old observation", and the model can learn the
calendar instead of the credit cycle. That is close enough to leakage to
exclude on principle rather than measure and hope.

Paired, for the same reason as everywhere else in this project: both arms fit
on the same rows and are scored on the same rows, differing only in the feature
set, so anything that flatters one flatters both.

Group-aware splitting is applied. Every fitted baseline in this project was
inflated by roughly 0.09 AUC of firm memorisation before that was caught, and
a macro effect measured on top of memorisation would be measuring the wrong
thing.
"""

from __future__ import annotations

import argparse
import random
import sys
import warnings
from datetime import date
from pathlib import Path

from data.macro import load_macro_available, macro_features
from models.hazard import roc_auc
from models.panel import feature_names, load_panel, split_by_date

#: Constrained, per the regularisation finding on this leg.
GBM_PARAMS = {
    "n_estimators": 150, "learning_rate": 0.05, "num_leaves": 3,
    "min_child_samples": 200, "subsample": 0.6, "subsample_freq": 1,
    "colsample_bytree": 0.4, "reg_lambda": 50.0,
}

#: A series must cover this share of observation dates to be admitted.
MIN_COVERAGE = 0.95


def usable_series(series: dict, dates: list[date]) -> tuple[list[str], list[str]]:
    """Split configured series into those covering the window and those not."""
    keep, drop = [], []
    for name, s in series.items():
        hit = sum(1 for d in dates if s.as_of(d) is not None)
        share = hit / len(dates) if dates else 0.0
        (keep if share >= MIN_COVERAGE else drop).append(f"{name} ({share:.0%})")
    return keep, drop


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--panel", type=Path, default=Path("data/cache/panel.csv"))
    ap.add_argument("--cutoff", type=date.fromisoformat, default=date(2024, 6, 1))
    args = ap.parse_args(argv)

    warnings.filterwarnings("ignore")
    import lightgbm as lgb

    rows = load_panel(args.panel)
    if not rows:
        print(f"no panel at {args.panel}", file=sys.stderr)
        return 1
    dates = sorted({r.observation_date for r in rows})
    series = load_macro_available()

    keep, drop = usable_series(series, dates)
    print(f"panel {len(rows)} rows over {len(dates)} observation dates", file=sys.stderr)
    print(f"  admitted : {', '.join(keep) or 'none'}", file=sys.stderr)
    print(f"  excluded : {', '.join(drop) or 'none'}  (below {MIN_COVERAGE:.0%} coverage)",
          file=sys.stderr)
    admitted = {k: v for k, v in series.items() if any(f.startswith(k + " ") for f in keep)}
    if not admitted:
        print("\nno macro series covers the panel window; nothing to test", file=sys.stderr)
        return 2

    for r in rows:
        r.features.update(macro_features(admitted, r.observation_date))
    macro_names = sorted(
        n for n in next(iter(rows)).features if n.startswith("macro_")
    )
    base = feature_names()

    train, test = split_by_date(rows, args.cutoff)
    test_firms = {r.cik for r in test}
    clean = [r for r in train if r.cik not in test_firms]
    y = [r.label for r in test]
    print(f"\ngroup-aware training {len(clean)} rows ({sum(r.label for r in clean)} pos); "
          f"test {len(test)} ({sum(y)} pos)", file=sys.stderr)

    def fit(names):
        m = lgb.LGBMClassifier(objective="binary", class_weight="balanced",
                               random_state=0, verbose=-1, **GBM_PARAMS)
        m.fit([[r.features.get(n, 0.0) for n in names] for r in clean],
              [r.label for r in clean])
        return [float(p) for p in
                m.predict_proba([[r.features.get(n, 0.0) for n in names] for r in test])[:, 1]]

    p_base, p_macro = fit(base), fit(base + macro_names)
    a_base, a_macro = roc_auc(y, p_base), roc_auc(y, p_macro)
    print(f"\n  firm ratios only        : AUC {a_base:.4f}", file=sys.stderr)
    print(f"  + {len(macro_names) // 2} macro covariates    : AUC {a_macro:.4f}", file=sys.stderr)

    rng = random.Random(0)
    diffs = []
    n = len(y)
    for _ in range(6000):
        i = [rng.randrange(n) for _ in range(n)]
        yy = [y[j] for j in i]
        if len(set(yy)) < 2:
            continue
        diffs.append(roc_auc(yy, [p_macro[j] for j in i]) - roc_auc(yy, [p_base[j] for j in i]))
    diffs.sort()
    lo, hi = diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs))]
    delta = a_macro - a_base
    verdict = ("REAL" if lo > 0 else "worse" if hi < 0 else "NO EFFECT")
    print(f"  difference              : {delta:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  {verdict}",
          file=sys.stderr)

    if drop:
        print(
            "\nThe most informative covariate is among the excluded. A high-yield "
            "credit spread covering 2019 onward would make this a stronger test; "
            "the available download starts 2023-08-22.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
