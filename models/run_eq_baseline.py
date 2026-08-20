"""Baselines for the earnings-quality leg (SPEC 12, phase 4).

The counterpart of ``models/run_baseline.py``. Same two tiers, same temporal
split discipline, different question: not "will this company fail" but "are the
statements a reader can see right now going to be declared unreliable".

Tier 0 is Beneish M, the field's published manipulation score with fixed 1999
coefficients. It is the honest analogue of Altman Z'' -- fitted to nothing, so
it cannot leak, and it works on a filer with no history.

Tier 1 is the same penalised logistic estimator the distress leg uses, over the
earnings covariates. It is fitted only on observations before the cutoff.

Reported next to both is a leak canary: the same model on shuffled labels. If
that does not land near 0.5, something in the pipeline is carrying the answer
and every other number on the page is worthless.
"""

from __future__ import annotations

import argparse
import random
import sys
from datetime import date
from pathlib import Path

from models.earnings import BeneishBaseline, eq_feature_names, load_eq_panel
from models.hazard import HazardBaseline, roc_auc
from models.panel import split_by_date


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--panel", type=Path, default=Path("data/cache/eq_panel_wide.csv"))
    ap.add_argument("--cutoff", type=date.fromisoformat, default=date(2023, 7, 1))
    ap.add_argument(
        "--negative-fraction",
        type=float,
        default=1.0,
        help="sampling fraction for base-rate correction of precision",
    )
    args = ap.parse_args(argv)

    rows = load_eq_panel(args.panel)
    if not rows:
        print(f"no panel at {args.panel}", file=sys.stderr)
        return 1
    train, test = split_by_date(rows, args.cutoff)
    ptr, pte = sum(r.label for r in train), sum(r.label for r in test)
    print(
        f"panel {len(rows)} rows | train {len(train)} ({ptr} pos) "
        f"| test {len(test)} ({pte} pos, {pte / max(1, len(test)):.2%})",
        file=sys.stderr,
    )
    if ptr < 20 or pte < 20:
        print("too few positives in a fold to say anything", file=sys.stderr)
        return 2

    names = eq_feature_names()
    print("\n=== earnings-quality baselines ===", file=sys.stderr)
    print(f"  Tier 0 Beneish M : {BeneishBaseline().evaluate(test).summary()}", file=sys.stderr)
    fitted = HazardBaseline(names=names).fit(train)
    print(f"  Tier 1 logistic  : {fitted.evaluate(test).summary()}", file=sys.stderr)

    # Leak canary. Shuffling the training labels destroys the signal; anything
    # the model still finds is coming from the pipeline rather than the data.
    shuffled = list(train)
    labels = [r.label for r in shuffled]
    random.Random(0).shuffle(labels)
    for r, y in zip(shuffled, labels, strict=True):
        r.label = y
    canary = HazardBaseline(names=names).fit(shuffled)
    canary_auc = roc_auc([r.label for r in test], canary.predict_proba(test))
    verdict = "ok" if 0.42 <= canary_auc <= 0.58 else "*** LEAK ***"
    print(f"  leak canary      : AUC {canary_auc:.4f} on shuffled labels -- {verdict}", file=sys.stderr)

    top = sorted(zip(names, fitted._model.coef_[0], strict=True), key=lambda t: -abs(t[1]))[:8]
    print("\n  strongest covariates:", file=sys.stderr)
    for n, c in top:
        print(f"    {n:<34} {c:+.3f}", file=sys.stderr)

    beneish_have = sum(1 for r in test if not r.missing.get("beneish_m_score", True))
    print(
        f"\n  Beneish M computable on {beneish_have}/{len(test)} test rows "
        f"({beneish_have / len(test):.0%}). It needs eight ratios across two "
        "periods; the rest score a neutral 0.5, which caps its discrimination.",
        file=sys.stderr,
    )
    if args.negative_fraction < 1.0:
        print(
            f"\n  precision is inflated by choice-based sampling "
            f"(fraction {args.negative_fraction:.4f}); ranking and recall are not.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
