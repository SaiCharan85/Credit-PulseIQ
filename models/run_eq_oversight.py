"""Attempt seven: regulator and governance signals, paired against base.

Six approaches have measured the earnings leg between 0.506 and 0.605, and all
six drew from the same well -- filed ratios and eight event flags. This one uses
information none of them had, which is the distinction between a new hypothesis
and a seventh slice of the same data:

* **SEC staff comment letters** (``UPLOAD``/``CORRESP``). The regulator formally
  questioning a company's accounting, before anything is restated.
* **Executive departures** (8-K item 5.02). A CFO leaving shortly before a
  restatement is among the better-documented precursors in the literature.
* **Prior restatement history.** Already on disk as labels, never used as a
  feature. A filer that has restated before is materially likelier to again.

Run on the *income-decreasing* label at filers over $100M -- the narrower
population from attempt six, which strips out much of the clerical-error
population that no model can predict. Restatements caused by a mistyped tax
rate have no precursor; the hope is only ever to find the deliberate ones.

The comparison is paired for the same reason as attempt six: the fold has been
looked at repeatedly and no untouched slice exists, so both arms fit on the
same window and score the same rows, differing only in the feature set. Fold
reuse moves them together and cannot manufacture a difference.

Pre-committed reading, fixed before the run:

    CI excludes zero and delta >= +0.05   -> real, worth an investigator
    CI excludes zero but delta < +0.05    -> marginal, report, do not build on it
    CI includes zero                      -> the leg stays closed, permanently
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

from agents.llm import load_env_file
from data.edgar import EdgarClient
from data.oversight import oversight_feature_names, oversight_signals
from models.earnings import eq_feature_names, load_eq_panel
from models.hazard import roc_auc
from models.panel import split_by_date


def restatement_dates_by_cik(path: Path) -> dict[int, list[date]]:
    """Every kept 4.02 event per filer, for the prior-history feature."""
    out: dict[int, list[date]] = defaultdict(list)
    if not path.exists():
        return out
    for r in csv.DictReader(path.open(encoding="utf8")):
        if r.get("verdict") != "keep":
            continue
        out[int(r["cik"])].append(date.fromisoformat(r["filing_date"]))
    for v in out.values():
        v.sort()
    return out


def fit_predict(train, test, names) -> list[float]:
    from sklearn.linear_model import LogisticRegression

    x = [[r.features.get(n, 0.0) for n in names] for r in train]
    y = [r.label for r in train]
    model = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(x, y)
    xt = [[r.features.get(n, 0.0) for n in names] for r in test]
    return [float(p) for p in model.predict_proba(xt)[:, 1]]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--panel", type=Path, default=Path("data/cache/eq_panel_severe.csv"))
    ap.add_argument(
        "--events", type=Path, default=Path("data/labels/candidates_restatements_wide.csv")
    )
    ap.add_argument("--cutoff", type=date.fromisoformat, default=date(2023, 7, 1))
    args = ap.parse_args(argv)

    load_env_file()
    rows = load_eq_panel(args.panel)
    base = eq_feature_names()
    if not rows:
        print(f"no panel at {args.panel}", file=sys.stderr)
        return 1
    print(f"panel {len(rows)} rows, {len(base)} base covariates", file=sys.stderr)

    prior = restatement_dates_by_cik(args.events)
    print(f"prior-restatement history for {len(prior)} filers", file=sys.stderr)

    edgar = EdgarClient()
    ciks = sorted({r.cik for r in rows})
    index: dict[int, list] = {}
    unavailable = 0
    for n, cik in enumerate(ciks, 1):
        try:
            index[cik] = edgar.filing_index(cik)
        except Exception:  # noqa: BLE001 - recorded below, not silently zeroed
            index[cik] = []
            unavailable += 1
        if n % 200 == 0:
            print(f"  index {n}/{len(ciks)}", file=sys.stderr)
    print(f"filing index unavailable for {unavailable}/{len(ciks)} filers", file=sys.stderr)

    new = oversight_feature_names()
    for r in rows:
        signals = oversight_signals(
            index.get(r.cik, []), r.observation_date, prior.get(r.cik, ())
        )
        feats = signals.as_features()
        for k, v in feats.items():
            r.features[k] = v
            r.missing[k] = bool(feats.get(f"{k}__missing", 0.0))

    train, test = split_by_date(rows, args.cutoff)
    y = [r.label for r in test]
    print(
        f"train {len(train)} ({sum(r.label for r in train)} pos) | "
        f"test {len(test)} ({sum(y)} pos)",
        file=sys.stderr,
    )
    if sum(y) < 100:
        print("fewer than 100 test positives; result would not be interpretable", file=sys.stderr)

    # How often does each new signal actually fire? A feature present on 1% of
    # rows cannot move an aggregate no matter how predictive it is when it does.
    print("\n  signal prevalence on the test fold:", file=sys.stderr)
    for name in ("ovs_comment_letters", "ovs_officer_events", "ovs_prior_restatements"):
        nonzero = sum(1 for r in test if r.features.get(name, 0.0) > 0)
        pos = sum(1 for r in test if r.label == 1 and r.features.get(name, 0.0) > 0)
        npos = max(1, sum(y))
        print(
            f"    {name:<26} {nonzero / len(test):>6.1%} of rows | "
            f"{pos / npos:>6.1%} of restaters",
            file=sys.stderr,
        )

    p_base = fit_predict(train, test, base)
    p_full = fit_predict(train, test, base + new)
    a_base, a_full = roc_auc(y, p_base), roc_auc(y, p_full)
    print("\n=== paired on identical rows ===", file=sys.stderr)
    print(f"  base features               : AUC {a_base:.4f}", file=sys.stderr)
    print(f"  base + oversight signals    : AUC {a_full:.4f}", file=sys.stderr)

    rng = random.Random(0)
    diffs = []
    n = len(y)
    for _ in range(6000):
        idx = [rng.randrange(n) for _ in range(n)]
        yy = [y[i] for i in idx]
        if len(set(yy)) < 2:
            continue
        diffs.append(
            roc_auc(yy, [p_full[i] for i in idx]) - roc_auc(yy, [p_base[i] for i in idx])
        )
    diffs.sort()
    lo, hi = diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs))]
    delta = a_full - a_base
    print(f"  difference                  : {delta:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]",
          file=sys.stderr)

    if lo > 0 and delta >= 0.05:
        verdict = "REAL -- worth an investigator"
    elif lo > 0:
        verdict = "MARGINAL -- report, do not build on it"
    else:
        verdict = "NO EFFECT -- the leg stays closed"
    print(f"  verdict                     : {verdict}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
