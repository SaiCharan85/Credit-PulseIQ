"""Attempt eight: a better estimator, and an honest split, on the earnings leg.

Seven attempts have all used ``LogisticRegression(C=1.0)``. On the distress leg
a regularised gradient-boosted model beat that same logit by +0.057 under
group-aware splitting, so trying it here is not another slice of the same data
-- it is a different axis, and one where the improvement is already measured
elsewhere.

Two things are tested at once, because they pull in opposite directions and
reporting either alone would mislead:

**A stronger estimator.** A logit cannot represent interactions, and accounting
red flags plausibly interact -- a comment letter matters more at a filer whose
accruals are already high than at one whose are not.

**An honest split.** The earnings panel has the same structure that inflated
every distress baseline: quarterly observations of the same companies, so a
test filer has typically appeared many times in training with near-identical
rows. The seven prior earnings numbers were all measured with that overlap
left in. Removing it will *lower* the score, and the difference is the
memorisation premium that has been sitting inside every earnings result so far.

Capacity is constrained from the start rather than tuned upward. On the
distress leg, selecting on validation AUC alone chose a configuration scoring
0.9946 on its own training rows, and constraining it *raised* test AUC. That
lesson is imported rather than relearned.

Pre-committed, and this is the eighth attempt so the bar is unchanged and the
correction is stricter:

    delta >= +0.05 and Bonferroni CI (alpha 0.05/8) excludes zero  -> real
    CI excludes zero but delta < +0.05                             -> marginal
    CI includes zero                                               -> closed
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import warnings
from collections import defaultdict
from datetime import date
from pathlib import Path

from agents.llm import load_env_file
from data.edgar import EdgarClient
from data.oversight import oversight_feature_names, oversight_signals
from models.earnings import eq_feature_names, load_eq_panel
from models.hazard import HazardBaseline, roc_auc
from models.panel import split_by_date

#: Constrained from the outset. See the distress leg's regularisation finding.
GBM_PARAMS = {
    "n_estimators": 150,
    "learning_rate": 0.05,
    "num_leaves": 3,
    "min_child_samples": 200,
    "subsample": 0.6,
    "subsample_freq": 1,
    "colsample_bytree": 0.4,
    "reg_lambda": 50.0,
}


def matrix(rows, names):
    return [[r.features.get(n, 0.0) for n in names] for r in rows]


def fit_gbm(train, names):
    import lightgbm as lgb

    m = lgb.LGBMClassifier(
        objective="binary", class_weight="balanced", random_state=0, verbose=-1, **GBM_PARAMS
    )
    m.fit(matrix(train, names), [r.label for r in train])
    return m


def predict(kind, train, test, names):
    if kind == "logistic":
        return HazardBaseline(names=names).fit(train).predict_proba(test)
    m = fit_gbm(train, names)
    return [float(p) for p in m.predict_proba(matrix(test, names))[:, 1]]


def boot(y, a, b, alpha):
    rng = random.Random(0)
    diffs = []
    n = len(y)
    for _ in range(6000):
        i = [rng.randrange(n) for _ in range(n)]
        yy = [y[j] for j in i]
        if len(set(yy)) < 2:
            continue
        diffs.append(roc_auc(yy, [a[j] for j in i]) - roc_auc(yy, [b[j] for j in i]))
    diffs.sort()
    return diffs[int(alpha / 2 * len(diffs))], diffs[int((1 - alpha / 2) * len(diffs))]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--panel", type=Path, default=Path("data/cache/eq_panel_severe.csv"))
    ap.add_argument(
        "--events", type=Path, default=Path("data/labels/candidates_restatements_wide.csv")
    )
    ap.add_argument("--cutoff", type=date.fromisoformat, default=date(2023, 7, 1))
    args = ap.parse_args(argv)

    warnings.filterwarnings("ignore")
    load_env_file()
    rows = load_eq_panel(args.panel)
    if not rows:
        print(f"no panel at {args.panel}", file=sys.stderr)
        return 1

    prior: dict[int, list[date]] = defaultdict(list)
    for r in csv.DictReader(args.events.open(encoding="utf8")):
        if r.get("verdict") == "keep":
            prior[int(r["cik"])].append(date.fromisoformat(r["filing_date"]))

    edgar = EdgarClient()
    ciks = sorted({r.cik for r in rows})
    index: dict[int, list] = {}
    for n, cik in enumerate(ciks, 1):
        try:
            index[cik] = edgar.filing_index(cik)
        except Exception:  # noqa: BLE001
            index[cik] = []
        if n % 300 == 0:
            print(f"  index {n}/{len(ciks)}", file=sys.stderr)
    for r in rows:
        r.features.update(
            oversight_signals(index.get(r.cik, []), r.observation_date, prior.get(r.cik, ())).as_features()
        )

    # The prior-restatement features are inverted by label construction: a
    # positive row precedes a filer's *first* restatement and so has none.
    # Excluded, as in attempt seven.
    names = eq_feature_names() + [n for n in oversight_feature_names() if "restatement" not in n]

    train, test = split_by_date(rows, args.cutoff)
    y = [r.label for r in test]
    test_firms = {r.cik for r in test}
    clean = [r for r in train if r.cik not in test_firms]
    overlap = sum(1 for r in test if r.cik in {t.cik for t in train})
    print(
        f"\ntest {len(test)} rows ({sum(y)} pos); {overlap}/{len(test)} "
        f"({overlap / len(test):.0%}) are firms already in training",
        file=sys.stderr,
    )
    print(f"training {len(train)} -> {len(clean)} rows once test firms are removed "
          f"({sum(r.label for r in clean)} pos)\n", file=sys.stderr)

    print(f"{'split':<14}{'estimator':<12}{'train':>9}{'test':>9}{'gap':>9}", file=sys.stderr)
    scores: dict[tuple, list[float]] = {}
    for split_label, tr in (("firms overlap", train), ("group-aware", clean)):
        if sum(r.label for r in tr) < 20:
            print(f"{split_label:<14} too few positives", file=sys.stderr)
            continue
        for kind in ("logistic", "gbm"):
            p_te = predict(kind, tr, test, names)
            p_tr = predict(kind, tr, tr, names)
            scores[(split_label, kind)] = p_te
            a_tr, a_te = roc_auc([r.label for r in tr], p_tr), roc_auc(y, p_te)
            print(f"{split_label:<14}{kind:<12}{a_tr:>9.4f}{a_te:>9.4f}{a_tr - a_te:>+9.4f}",
                  file=sys.stderr)

    print("\n=== does the estimator help? (group-aware, paired) ===", file=sys.stderr)
    if ("group-aware", "gbm") in scores:
        g, lgt = scores[("group-aware", "gbm")], scores[("group-aware", "logistic")]
        delta = roc_auc(y, g) - roc_auc(y, lgt)
        lo95, hi95 = boot(y, g, lgt, 0.05)
        lob, hib = boot(y, g, lgt, 0.05 / 8)
        print(f"  GBM - logistic: {delta:+.4f}", file=sys.stderr)
        print(f"    95% CI        [{lo95:+.4f}, {hi95:+.4f}]", file=sys.stderr)
        print(f"    Bonferroni /8 [{lob:+.4f}, {hib:+.4f}]", file=sys.stderr)
        if lob > 0 and delta >= 0.05:
            v = "REAL -- the leg is worth reopening"
        elif lob > 0:
            v = "MARGINAL -- report, do not build on it"
        else:
            v = "NO EFFECT -- the leg stays closed"
        print(f"    verdict       {v}", file=sys.stderr)

    if ("firms overlap", "gbm") in scores:
        premium = roc_auc(y, scores[("firms overlap", "gbm")]) - roc_auc(y, scores[("group-aware", "gbm")])
        print(f"\n  memorisation premium inside every earlier earnings number: "
              f"{premium:+.4f} AUC", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
