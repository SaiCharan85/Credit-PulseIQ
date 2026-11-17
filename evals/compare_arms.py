"""Paired comparison of two backtest arms on the cases both completed.

    python -m evals.compare_arms data/cache/l3_react_full.csv data/cache/l3_single.csv

Only the intersection is compared -- the same filers at the same dates -- and
the uncertainty is a *paired* bootstrap over those cases: resample cases, recompute
both AUCs on the same resample, take the difference. Pairing matters because the
arms share cases; treating them as independent samples would overstate the noise
and hide a real difference, or the reverse.

The interval is the point. Two arms near AUC 0.96 on a couple of hundred cases
differ by noise more often than not, and a single number invites reading a
0.005 gap as a result.

Abstentions are scored 0.5, the backtest's convention, so an arm cannot improve
its AUC by declining the hard cases.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

from evals.to_langfuse import auc, key, probability


def load(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf8") as fh:
        return {key(r["cik"], r["as_of"]): r for r in csv.DictReader(fh)}


def paired_bootstrap(a: list[float], b: list[float], y: list[int], n: int = 2000, seed: int = 7):
    rng = random.Random(seed)
    idx = list(range(len(y)))
    diffs = []
    for _ in range(n):
        s = [rng.choice(idx) for _ in idx]
        ys = [y[i] for i in s]
        if 0 < sum(ys) < len(ys):
            da = auc([(a[i], y[i]) for i in s])
            db = auc([(b[i], y[i]) for i in s])
            diffs.append(db - da)
    diffs.sort()
    lo, hi = diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs)) - 1]
    return lo, hi, sum(d > 0 for d in diffs) / len(diffs)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("a", type=Path, help="arm A results CSV (e.g. the tool loop)")
    ap.add_argument("b", type=Path, help="arm B results CSV (e.g. single-pass)")
    ap.add_argument("--out", type=Path, default=Path("results/ab_comparison.log"))
    args = ap.parse_args(argv)

    A, B = load(args.a), load(args.b)
    common = sorted(set(A) & set(B))
    y = [int(A[k]["label"]) for k in common]
    pa = [probability(A[k]) for k in common]
    pb = [probability(B[k]) for k in common]

    def summary(rows: dict, ps: list[float]) -> str:
        abst = sum(rows[k]["signal"] == "insufficient_evidence" for k in common)
        steps = [float(rows[k]["steps"] or 0) for k in common]
        a = auc(list(zip(ps, y, strict=True)))
        return (f"AUC {a:.4f} | abstained {abst}/{len(common)} | "
                f"mean steps {sum(steps) / len(steps):.1f}")

    lo, hi, p_better = paired_bootstrap(pa, pb, y)
    text = "\n".join([
        f"A/B on the {len(common)} cases both arms completed "
        f"({sum(y)} bankruptcies, {len(y) - sum(y)} healthy)",
        f"  A {args.a.name:<22} {summary(A, pa)}",
        f"  B {args.b.name:<22} {summary(B, pb)}",
        f"  AUC(B) - AUC(A): {auc(list(zip(pb, y, strict=True))) - auc(list(zip(pa, y, strict=True))):+.4f}"
        f"   paired-bootstrap 95% CI [{lo:+.4f}, {hi:+.4f}]   P(B better) {p_better:.2f}",
        "  An interval spanning 0 means the arms are not distinguishable on these cases.",
        f"  only in A: {len(set(A) - set(B))}   only in B: {len(set(B) - set(A))}",
    ])
    print(text)
    args.out.write_text(text + "\n", encoding="utf8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
