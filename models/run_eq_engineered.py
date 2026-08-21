"""Attempt six on the earnings leg: engineered features, paired against base.

Five approaches have already been measured on this test fold, so an absolute
AUC here would be the sixth look at the same data and worth correspondingly
less. There is no untouched slice to retreat to either: every observation after
2023-07-01 was inside the test fold of all five, and the panel cannot be
extended forward because a 12-month label window needs outcomes that have not
happened yet.

So the question is posed as a *paired* one instead. Both arms fit on the same
training window and are scored on the same rows; the only difference is the
feature set. Bootstrapping the difference asks "do these features add
anything", which fold reuse cannot flatter -- reusing the fold moves both arms
together.

Three families, none of which the earlier attempts used, because all five
worked from point-in-time levels:

**Industry-adjusted.** Raw accruals differ structurally by sector; a software
firm and a utility are not comparable on the level. The accounting literature
uses industry-relative measures for exactly this reason. Each metric becomes a
percentile within its 2-digit SIC group *at the same observation date*, so the
comparison is cross-sectional and cannot see the future.

**Change.** Sloan's anomaly is about accruals persisting, not their level once.
Each metric gets its year-over-year change for the same firm.

**Volatility.** Dispersion of each metric over the firm's trailing history, a
standard manipulation correlate that is not computed anywhere in the project.

Pre-committed reading, fixed before the run:

    CI on the difference excludes zero and delta >= +0.05   -> real, worth an investigator
    CI excludes zero but delta < +0.05                      -> marginal, report, do not build
    CI includes zero                                        -> the leg is closed
"""

from __future__ import annotations

import argparse
import random
import statistics as st
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

from agents.llm import load_env_file
from data.edgar import EdgarClient
from models.earnings import eq_feature_names, load_eq_panel
from models.hazard import roc_auc
from models.panel import PanelRow, split_by_date

#: Metrics worth adjusting and differencing. The accounting-quality core.
CORE = (
    "accruals_to_assets",
    "days_sales_outstanding",
    "days_inventory_outstanding",
    "cash_conversion_cycle",
    "net_margin",
    "return_on_assets",
    "operating_margin",
    "beneish_sgi",
    "beneish_dsri",
)


def sic_map(ciks: list[int], edgar: EdgarClient) -> dict[int, str]:
    """2-digit SIC per filer, from cached submissions metadata."""
    out: dict[int, str] = {}
    for n, cik in enumerate(ciks, 1):
        try:
            sub = edgar.submissions(cik)
            sic = str(sub.get("sic") or "").strip()
        except Exception:  # noqa: BLE001
            sic = ""
        out[cik] = sic[:2] if len(sic) >= 2 else "??"
        if n % 400 == 0:
            print(f"  sic {n}/{len(ciks)}", file=sys.stderr)
    return out


def _pct_rank(value: float, pool: list[float]) -> float:
    if len(pool) < 5:
        return 0.5  # too thin a group to rank against; neutral, and flagged below
    below = sum(1 for v in pool if v < value)
    return below / len(pool)


def engineer(rows: list[PanelRow], sic: dict[int, str]) -> list[str]:
    """Add industry-adjusted, change and volatility features in place."""
    added: list[str] = []

    # --- industry-adjusted: percentile within (observation_date, SIC2) ---
    groups: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (r.observation_date, sic.get(r.cik, "??"))
        for m in CORE:
            if not r.missing.get(m, True):
                groups[key][m].append(r.features[m])
    for r in rows:
        key = (r.observation_date, sic.get(r.cik, "??"))
        for m in CORE:
            name = f"{m}__ind"
            if r.missing.get(m, True):
                r.features[name] = 0.5
                r.missing[name] = True
            else:
                r.features[name] = _pct_rank(r.features[m], groups[key][m])
                r.missing[name] = False
            r.features[f"{name}__missing"] = 1.0 if r.missing[name] else 0.0
    added += [f"{m}__ind" for m in CORE]

    # --- change and volatility: within firm, over its own history ---
    by_firm: dict[int, list[PanelRow]] = defaultdict(list)
    for r in rows:
        by_firm[r.cik].append(r)
    for firm_rows in by_firm.values():
        firm_rows.sort(key=lambda r: r.observation_date)
        for i, r in enumerate(firm_rows):
            past = firm_rows[max(0, i - 4) : i + 1]
            for m in CORE:
                delta, vol = f"{m}__chg", f"{m}__vol"
                series = [p.features[m] for p in past if not p.missing.get(m, True)]
                have = not r.missing.get(m, True) and len(series) >= 2
                r.features[delta] = (series[-1] - series[0]) if have else 0.0
                r.missing[delta] = not have
                r.features[f"{delta}__missing"] = 0.0 if have else 1.0
                enough = len(series) >= 3
                r.features[vol] = st.pstdev(series) if enough else 0.0
                r.missing[vol] = not enough
                r.features[f"{vol}__missing"] = 0.0 if enough else 1.0
    added += [f"{m}__chg" for m in CORE] + [f"{m}__vol" for m in CORE]

    names: list[str] = []
    for a in added:
        names += [a, f"{a}__missing"]
    return names


def fit_predict(train, test, names) -> list[float]:
    from sklearn.linear_model import LogisticRegression

    x = [[r.features.get(n, 0.0) for n in names] for r in train]
    y = [r.label for r in train]
    model = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(x, y)
    xt = [[r.features.get(n, 0.0) for n in names] for r in test]
    return [float(p) for p in model.predict_proba(xt)[:, 1]]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--panel", type=Path, default=Path("data/cache/eq_panel_wide2.csv"))
    ap.add_argument("--cutoff", type=date.fromisoformat, default=date(2023, 7, 1))
    args = ap.parse_args(argv)

    load_env_file()
    rows = load_eq_panel(args.panel)
    base = eq_feature_names()
    print(f"panel {len(rows)} rows, {len(base)} base covariates", file=sys.stderr)

    edgar = EdgarClient()
    ciks = sorted({r.cik for r in rows})
    sic = sic_map(ciks, edgar)
    known = sum(1 for v in sic.values() if v != "??")
    print(f"SIC resolved for {known}/{len(ciks)} filers", file=sys.stderr)

    new = engineer(rows, sic)
    print(f"engineered {len(new) // 2} features (+ missingness flags)", file=sys.stderr)

    train, test = split_by_date(rows, args.cutoff)
    y = [r.label for r in test]
    print(
        f"train {len(train)} ({sum(r.label for r in train)} pos) | "
        f"test {len(test)} ({sum(y)} pos)",
        file=sys.stderr,
    )

    p_base = fit_predict(train, test, base)
    p_full = fit_predict(train, test, base + new)
    a_base, a_full = roc_auc(y, p_base), roc_auc(y, p_full)
    print("\n=== paired on identical rows ===", file=sys.stderr)
    print(f"  base features            : AUC {a_base:.4f}", file=sys.stderr)
    print(f"  base + engineered        : AUC {a_full:.4f}", file=sys.stderr)

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
    print(f"  difference               : {delta:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]", file=sys.stderr)

    if lo > 0 and delta >= 0.05:
        verdict = "REAL -- worth an investigator"
    elif lo > 0:
        verdict = "MARGINAL -- report, do not build on it"
    else:
        verdict = "NO EFFECT -- the leg is closed"
    print(f"  verdict                  : {verdict}", file=sys.stderr)
    print(
        "\nPaired by construction: both arms fit on the same window and score the "
        "same rows, so reuse of this fold moves them together and cannot manufacture "
        "a difference.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
