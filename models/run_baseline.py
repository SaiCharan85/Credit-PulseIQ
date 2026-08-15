"""Fit and report the Tier-0 and Tier-1 baselines.

Tier 0 is Altman Z'' used directly as a ranking, unfitted. Tier 1 is the
discrete-time hazard. Both are scored on a **temporal** hold-out: train on
observations before the cutoff, test on or after. A random split would leak,
because a firm's adjacent quarters are nearly identical and would land on both
sides.

These numbers are the bar the ReAct investigator has to clear in Phase 3.
"""

from __future__ import annotations

import argparse
import random
import sys
from datetime import date
from pathlib import Path

from data.distress_events import load_events
from data.edgar import EdgarClient, default_user_agent
from data.labels import load_chapter11_labels
from data.watchlist import load_watchlist
from models.hazard import AltmanBaseline, HazardBaseline
from models.panel import (
    build_panel,
    load_panel,
    observation_dates,
    save_panel,
    split_by_date,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--start", type=date.fromisoformat, default=date(2021, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2025, 8, 1))
    parser.add_argument("--cutoff", type=date.fromisoformat, default=date(2024, 6, 1))
    parser.add_argument("--horizon-days", type=int, default=365)
    parser.add_argument("--months", type=int, default=3)
    parser.add_argument("--user-agent", default=default_user_agent())
    parser.add_argument("--panel", type=Path, default=Path("data/cache/panel.csv"))
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument(
        "--shuffle-labels",
        action="store_true",
        help="leak canary: permute labels; a sound pipeline collapses to AUC ~0.5",
    )
    args = parser.parse_args(argv)

    rows = [] if args.rebuild else load_panel(args.panel)
    if rows:
        print(f"loaded cached panel: {len(rows)} rows from {args.panel}", file=sys.stderr)
    else:
        client = EdgarClient(user_agent=args.user_agent)
        events = load_events()
        if not events:
            from data.distress_events import merged_event_set

            events = merged_event_set([], load_chapter11_labels())
        ciks = [e.cik for e in load_watchlist()]
        grid = observation_dates(args.start, args.end, args.months)
        print(
            f"building panel: {len(ciks)} filers x {len(grid)} observation dates, "
            f"horizon {args.horizon_days}d",
            file=sys.stderr,
        )
        rows = build_panel(client, ciks, events, grid, args.horizon_days)
        save_panel(rows, args.panel)
        print(f"panel saved to {args.panel}", file=sys.stderr)

    if args.shuffle_labels:
        # Permute labels within the panel. Features, dates and split are
        # untouched, so any surviving signal is an artefact of the pipeline
        # rather than of the data.
        rng = random.Random(0)
        labels = [r.label for r in rows]
        rng.shuffle(labels)
        for r, y in zip(rows, labels, strict=True):
            r.label = y
        print("LEAK CANARY: labels shuffled; expect AUC ~ 0.5", file=sys.stderr)

    train, test = split_by_date(rows, args.cutoff)

    print(f"\npanel rows: {len(rows)}  ({len(train)} train / {len(test)} test)", file=sys.stderr)
    print(
        f"positives: {sum(r.label for r in rows)}  "
        f"({sum(r.label for r in train)} train / {sum(r.label for r in test)} test)",
        file=sys.stderr,
    )
    print(f"distinct filers with rows: {len({r.cik for r in rows})}", file=sys.stderr)

    if not train or not test or sum(r.label for r in train) == 0:
        print("\ninsufficient panel to fit; widen the window", file=sys.stderr)
        return 1

    print("\n--- Tier 0: Altman Z'' (unfitted ranking) ---", file=sys.stderr)
    altman = AltmanBaseline()
    print(f"  test : {altman.evaluate(test).summary()}", file=sys.stderr)

    print("\n--- Tier 1: discrete-time hazard (Shumway) ---", file=sys.stderr)
    hazard = HazardBaseline().fit(train)
    print(f"  train: {hazard.evaluate(train).summary()}", file=sys.stderr)
    print(f"  test : {hazard.evaluate(test).summary()}", file=sys.stderr)

    print("\n  top standardised coefficients:", file=sys.stderr)
    for name, coef in list(hazard.coefficients().items())[:12]:
        print(f"    {name:38} {coef:+.3f}", file=sys.stderr)

    print(
        "\nNote: the universe is enriched with bankrupt filers far above the "
        "population base rate (Zmijewski 1984), so these probabilities are not "
        "population default rates. Discrimination is the comparable quantity.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
