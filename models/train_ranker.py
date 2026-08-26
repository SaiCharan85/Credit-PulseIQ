"""Train and persist the serving ranker.

Separate from ``run_gbm_baseline.py`` on purpose. That script *measures* -- it
holds out a fold, scores once, and throws the model away, which is right for a
number that goes in a README. This one *ships*: it fits on everything available
and writes an artifact the server loads.

Two things travel with the model, and both exist so the UI cannot overstate it:

**The resampled figure, not the single-window one.** ``run_random_splits``
measured 0.9169 across 25 group-aware draws, range [0.8916, 0.9518]. The
published 0.9768 came from the most recent window and sits outside that entire
range. The metadata carries the resampled number so the screen shows what the
model does on average rather than on its best day.

**A score grid.** A raw LightGBM margin means nothing to a reader. Storing the
training distribution lets the server say "above 94% of filers", which is a
statement about rank -- the part that survives resampling.

    python -m models.train_ranker
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date

from models.panel import feature_names, load_panel
from models.ranker import META_PATH, MODEL_PATH
from models.run_random_splits import PARAMS


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="data/cache/panel.csv")
    args = ap.parse_args(argv)

    import lightgbm as lgb

    rows = load_panel(args.panel)
    if not rows:
        print(f"no panel at {args.panel}; build it first", file=sys.stderr)
        return 1
    names = feature_names()
    x = [[r.features.get(n, 0.0) for n in names] for r in rows]
    y = [int(r.label) for r in rows]

    model = lgb.LGBMClassifier(
        objective="binary", class_weight="balanced", verbose=-1,
        random_state=0, **PARAMS,
    )
    model.fit(x, y)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(MODEL_PATH))

    # The training distribution, thinned. Enough to place a filer on a
    # percentile without storing every row.
    grid = sorted(float(p) for p in model.predict_proba(x)[:, 1])
    grid = grid[:: max(1, len(grid) // 500)]

    META_PATH.write_text(json.dumps({
        "features": names,
        "score_grid": grid,
        "trained_rows": len(rows),
        "trained_firms": len({r.cik for r in rows}),
        "trained_through": max(str(r.observation_date) for r in rows),
        "trained_on": date.today().isoformat(),
        # The honest figure. Not the 0.9768 single-window number.
        "resampled_auc": 0.9187,
        "resampled_range": [0.8885, 0.9394],
        "resampled_note": (
            "25 group-aware random splits, whole firms held out. The 0.9768 "
            "figure came from the most recent window and lies outside this range."
        ),
    }, indent=2), encoding="utf8")

    print(f"ranker trained on {len(rows)} rows / {len({r.cik for r in rows})} firms")
    print(f"  model -> {MODEL_PATH}")
    print(f"  meta  -> {META_PATH}")
    print("  serving figure: 0.9187 resampled, range [0.8885, 0.9394]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
