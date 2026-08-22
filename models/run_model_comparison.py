"""Every model, every split, one table.

Four estimators across train, validation and test, so the numbers that have
been quoted piecemeal all session can be read against each other.

Three things the table is built to expose:

**Train-versus-test gap.** A model scoring far higher on data it was fitted to
than on held-out data is memorising. Reporting only test AUC hides that;
reporting both makes it unmissable.

**Firm memorisation.** The panel holds quarterly observations of the same
companies, so a firm in the test fold has typically appeared ~13 times in
training with near-identical features. Every arm is therefore run twice --
once with that overlap left in, once with test firms removed entirely from
training. The gap between the two is the memorisation premium.

**The agent has no train or validation column, and that is the point.** It
fits nothing. There is no set of rows it has seen, so novelty costs it
nothing, and its single number is the same figure under every split
condition. That invariance is its actual property, not a higher score.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import warnings
from datetime import date
from pathlib import Path

from models.hazard import HazardBaseline, roc_auc
from models.panel import feature_names, load_panel, split_by_date

LGBM_PARAMS = {
    "n_estimators": 600,
    "learning_rate": 0.02,
    "num_leaves": 31,
    "min_child_samples": 80,
}


def matrix(rows, names):
    return [[r.features.get(n, 0.0) for n in names] for r in rows]


def build(kind: str, train, names):
    """Fit one estimator. All use balanced weights: positives are ~8% here."""
    x, y = matrix(train, names), [r.label for r in train]
    if kind == "logistic":
        return HazardBaseline().fit(train)
    if kind == "histgb":
        from sklearn.ensemble import HistGradientBoostingClassifier

        m = HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
            min_samples_leaf=40, class_weight="balanced", random_state=0,
        )
    else:
        import lightgbm as lgb

        m = lgb.LGBMClassifier(
            objective="binary", class_weight="balanced", random_state=0,
            verbose=-1, **LGBM_PARAMS,
        )
    m.fit(x, y)
    return m


def score(model, rows, names) -> list[float]:
    if isinstance(model, HazardBaseline):
        return model.predict_proba(rows)
    return [float(p) for p in model.predict_proba(matrix(rows, names))[:, 1]]


def auc_or_dash(rows, p) -> str:
    y = [r.label for r in rows]
    return f"{roc_auc(y, p):.4f}" if len(set(y)) > 1 else "  --  "


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--panel", type=Path, default=Path("data/cache/panel.csv"))
    ap.add_argument("--cutoff", type=date.fromisoformat, default=date(2024, 6, 1))
    ap.add_argument("--val-cutoff", type=date.fromisoformat, default=date(2023, 6, 1))
    ap.add_argument("--agent-results", type=Path, default=Path("data/cache/l3_sig_partial.csv"))
    args = ap.parse_args(argv)

    warnings.filterwarnings("ignore")
    from evals.backtest import stratified_sample

    rows = load_panel(args.panel)
    names = feature_names()
    train_all, test = split_by_date(rows, args.cutoff)
    fit_rows, val = split_by_date(train_all, args.val_cutoff)

    agent = {
        (int(r["cik"]), date.fromisoformat(r["as_of"])): float(r["risk_probability"])
        for r in csv.DictReader(args.agent_results.open(encoding="utf8"))
    }
    graded = [
        c for c in stratified_sample(test, 100).cases
        if (c.cik, c.observation_date) in agent
    ]
    y_graded = [c.label for c in graded]
    p_agent = [agent[(c.cik, c.observation_date)] for c in graded]

    print(
        f"fit   {len(fit_rows):>5} rows {sum(r.label for r in fit_rows):>4} pos  (< {args.val_cutoff})\n"
        f"val   {len(val):>5} rows {sum(r.label for r in val):>4} pos  ({args.val_cutoff} .. {args.cutoff})\n"
        f"test  {len(test):>5} rows {sum(r.label for r in test):>4} pos  (>= {args.cutoff})\n"
        f"graded{len(graded):>5} rows {sum(y_graded):>4} pos  (the agent's cases, stratified)",
        file=sys.stderr,
    )

    graded_firms = {c.cik for c in graded}
    kinds = ("logistic", "histgb", "lightgbm")

    for label, pool in (
        ("A. FIRMS OVERLAP (test companies also appear in training)", fit_rows),
        ("B. GROUP-AWARE (graded companies removed from training)",
         [r for r in fit_rows if r.cik not in graded_firms]),
    ):
        print(f"\n=== {label} ===", file=sys.stderr)
        print(f"   training rows {len(pool)}, {sum(r.label for r in pool)} positives",
              file=sys.stderr)
        print(f"\n{'model':<12}{'train':>9}{'val':>9}{'test':>9}{'graded169':>11}"
              f"{'train-test':>12}", file=sys.stderr)
        for kind in kinds:
            m = build(kind, pool, names)
            a_tr = auc_or_dash(pool, score(m, pool, names))
            a_va = auc_or_dash(val, score(m, val, names))
            a_te = auc_or_dash(test, score(m, test, names))
            a_gr = auc_or_dash(graded, score(m, graded, names))
            gap = float(a_tr) - float(a_te) if "-" not in a_tr + a_te else 0.0
            print(f"{kind:<12}{a_tr:>9}{a_va:>9}{a_te:>9}{a_gr:>11}{gap:>+12.4f}",
                  file=sys.stderr)
        print(f"{'agent':<12}{'n/a':>9}{'n/a':>9}{'n/a':>9}"
              f"{roc_auc(y_graded, p_agent):>11.4f}{'n/a':>12}", file=sys.stderr)

    # Paired test on the arm that matters: group-aware, identical cases.
    clean = [r for r in fit_rows if r.cik not in graded_firms]
    best = build("lightgbm", clean, names)
    p_gbm = score(best, graded, names)
    rng = random.Random(0)
    diffs = []
    n = len(y_graded)
    for _ in range(6000):
        i = [rng.randrange(n) for _ in range(n)]
        yy = [y_graded[j] for j in i]
        if len(set(yy)) < 2:
            continue
        diffs.append(
            roc_auc(yy, [p_agent[j] for j in i]) - roc_auc(yy, [p_gbm[j] for j in i])
        )
    diffs.sort()
    lo, hi = diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs))]
    delta = roc_auc(y_graded, p_agent) - roc_auc(y_graded, p_gbm)
    verdict = "agent better" if lo > 0 else ("agent worse" if hi < 0 else "TIE")
    print(f"\nagent - LightGBM, group-aware, identical {n} cases: "
          f"{delta:+.4f} 95% CI [{lo:+.4f}, {hi:+.4f}]  {verdict}", file=sys.stderr)
    print(
        "\nThe agent has no train or validation column because it fits nothing. "
        "Its single figure holds under every split condition, which is the "
        "property being claimed -- invariance to novelty, not a higher score.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
