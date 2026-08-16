"""Run the L3 backtest and report it against the deterministic baselines.

Uses the same panel and the same temporal split as ``models/run_baseline.py``,
so agent and baseline are scored on identical observations under identical
as-of rules. Comparing across different splits would measure the split.

Default agent is the rule-based control, which needs no endpoint. Pass
``--agent react`` once a model is configured.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

from agents.rulebased import RuleBasedInvestigator
from data.edgar import EdgarClient
from evals.backtest import assert_no_lookahead, grade, reliability_curve, run_backtest
from models.hazard import AltmanBaseline, HazardBaseline
from models.panel import load_panel, split_by_date


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--panel", type=Path, default=Path("data/cache/panel.csv"))
    parser.add_argument("--cutoff", type=date.fromisoformat, default=date(2024, 6, 1))
    parser.add_argument("--agent", choices=("rules", "react"), default="rules")
    parser.add_argument("--limit", type=int, default=0, help="cap test cases (0 = all)")
    args = parser.parse_args(argv)

    rows = load_panel(args.panel)
    if not rows:
        print(f"no panel at {args.panel}; run `python -m models.run_baseline --rebuild`", file=sys.stderr)
        return 1
    train, test = split_by_date(rows, args.cutoff)
    if args.limit:
        test = test[: args.limit]

    if args.agent == "react":
        from agents.distress import DistressInvestigator
        from agents.llm import default_client

        investigator = DistressInvestigator(default_client())
        label = "ReAct investigator"
    else:
        investigator = RuleBasedInvestigator()
        label = "rule-based control"

    client = EdgarClient()
    cache: dict[int, list] = {}

    def facts_for(cik: int):
        if cik not in cache:
            try:
                cache[cik] = client.facts(cik)
            except Exception:  # noqa: BLE001
                cache[cik] = []
        return cache[cik]

    print(f"L3 backtest: {label} over {len(test)} test cases (cutoff {args.cutoff})", file=sys.stderr)

    def progress(n: int, _result) -> None:
        if n % 200 == 0:
            print(f"  {n}/{len(test)}", file=sys.stderr)

    results = run_backtest(investigator, test, facts_for, on_case=progress)
    assert_no_lookahead(results)

    report = grade(results)
    print(f"\n=== L3: {label} ===", file=sys.stderr)
    print(report.summary(), file=sys.stderr)

    print("\n  reliability curve (confidence -> observed failure rate):", file=sys.stderr)
    for row in reliability_curve(results):
        print(
            f"    [{row['bin_low']:.1f}-{row['bin_high']:.1f})  n={row['n']:>5}  "
            f"stated={row['mean_confidence']:.2f}  observed={row['observed_rate']:.2f}",
            file=sys.stderr,
        )

    print("\n=== deterministic baselines on the same test split ===", file=sys.stderr)
    print(f"  Tier 0 Altman Z'' : {AltmanBaseline().evaluate(test).summary()}", file=sys.stderr)
    hazard = HazardBaseline().fit(train)
    print(f"  Tier 1 hazard     : {hazard.evaluate(test).summary()}", file=sys.stderr)
    print(
        "\nAUC is comparable across rows; precision/recall are not directly "
        "comparable, because the agent may abstain and the baselines never do.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
