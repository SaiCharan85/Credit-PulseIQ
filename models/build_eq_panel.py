"""Build the earnings-quality panel over the wide, point-in-time universe.

Separated from ``models/earnings.py`` because it is a long-running fetch job,
not a library: several thousand ``companyfacts`` requests against EDGAR, cached
on disk, resumable by re-running.

The sample is choice-based on purpose. Every verified restatement is kept
because each one cost a full-text sweep and a text verification to find, while
the negative pool is thinned to what the fetch budget allows. That inflates the
base rate by construction (Zmijewski 1984), so precision must be corrected by
the sampling fraction downstream -- which ``evals/backtest.py`` already does.
Ranking and recall are unaffected.
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import date
from pathlib import Path

from agents.llm import load_env_file
from data.edgar import EdgarClient
from data.restatements import RestatementCandidate, first_event_per_company
from data.universe import IndexEntry, sample_universe
from models.earnings import build_eq_rows, restatement_events, save_eq_panel
from models.panel import observation_dates


def load_candidates(path: Path) -> list[RestatementCandidate]:
    out = []
    for r in csv.DictReader(path.open(encoding="utf8")):
        out.append(
            RestatementCandidate(
                cik=int(r["cik"]),
                filing_date=date.fromisoformat(r["filing_date"]),
                accession=r["accession"],
                primary_document=r["primary_document"],
                verdict=r["verdict"],
                detail=r["detail"],
            )
        )
    return out


def load_universe(cache_dir: Path) -> dict[int, IndexEntry]:
    """Rebuild the filer universe from the cached quarterly indexes."""
    from data.universe import parse_form_index

    found: dict[int, IndexEntry] = {}
    for path in sorted(cache_dir.glob("form_*.idx")):
        for e in parse_form_index(path.read_text(encoding="utf8", errors="replace")):
            if e.cik not in found or e.filed < found[e.cik].filed:
                found[e.cik] = e
    return found


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--candidates", type=Path, default=Path("data/labels/candidates_restatements_wide.csv")
    )
    ap.add_argument("--index-cache", type=Path, default=Path("data/cache/fullindex"))
    ap.add_argument("--out", type=Path, default=Path("data/cache/eq_panel_wide.csv"))
    ap.add_argument("--negatives", type=int, default=1500, help="negative filers to sample")
    ap.add_argument(
        "--min-total-assets",
        type=float,
        default=0.0,
        help="size floor at the observation date, e.g. 100e6",
    )
    ap.add_argument("--start", type=date.fromisoformat, default=date(2019, 1, 1))
    ap.add_argument("--end", type=date.fromisoformat, default=date(2025, 7, 1))
    args = ap.parse_args(argv)

    load_env_file()
    cands = load_candidates(args.candidates)
    events = restatement_events(first_event_per_company(cands))
    positives = sorted({e.cik for e in events})
    print(f"verified restatement events: {len(positives)} companies", file=sys.stderr)

    filers = load_universe(args.index_cache)
    print(f"point-in-time universe: {len(filers)} annual filers", file=sys.stderr)

    ciks = sample_universe(filers, keep=positives, n_sample=args.negatives)
    kept_neg = len(ciks) - len(set(positives) & set(filers))
    pool = len(filers) - len(set(positives) & set(filers))
    fraction = kept_neg / pool if pool else 1.0
    print(
        f"sampling {len(ciks)} filers "
        f"(all positives + {args.negatives} negatives; fraction {fraction:.4f})",
        file=sys.stderr,
    )

    dates = observation_dates(args.start, args.end, months=3)
    edgar = EdgarClient()
    rows = []
    missing = 0
    for n, cik in enumerate(ciks, 1):
        try:
            facts = edgar.facts(cik)
        except Exception:  # noqa: BLE001 - an unavailable filer is skipped, not fatal
            missing += 1
            continue
        rows.extend(build_eq_rows(facts, cik, events, dates, min_total_assets=args.min_total_assets))
        if n % 100 == 0:
            pos = sum(r.label for r in rows)
            print(f"  {n}/{len(ciks)} filers  rows={len(rows)}  pos={pos}", file=sys.stderr)

    pos = sum(r.label for r in rows)
    print(
        f"\nEQ panel: {len(rows)} rows, {pos} positive ({pos / max(1, len(rows)):.2%}), "
        f"{len({r.cik for r in rows})} filers ({missing} unavailable)",
        file=sys.stderr,
    )
    save_eq_panel(rows, args.out)
    print(f"-> {args.out}", file=sys.stderr)
    print(f"negative_fraction={fraction:.6f}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
