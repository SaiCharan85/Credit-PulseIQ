"""Ingest filing text for the backtest filers into the chunk store.

    python -m rag.ingest                    # 20-filer pilot, local store
    python -m rag.ingest --all              # every filer in the case file
    python -m rag.ingest --all --store cloud

**Incremental.** A filing already in the store is skipped, so a rerun -- after
a stop, a crash, or new filings arriving -- only does the new work, and the
pilot is simply the first part of the full run.

**Only what the cases can see.** For each filer, the periodic reports filed in
the window ``[earliest as-of - WINDOW_DAYS, latest as-of]``. Nothing filed
after the last prediction date is ingested at all, which is a second line of
defence behind the query-time filter, not a substitute for it.

**Stops before memory runs out** (see ``rag.memguard``).

**Pilot selection is deterministic** and deliberately skewed toward filers
whose text carries a warning -- going concern, material weakness, covenant
trouble -- because those are what the recall eval can score. A pilot of
healthy filers would measure nothing.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from data.edgar import EdgarClient, EdgarError
from data.signals import REPORT_FORMS
from rag import memguard
from rag.chunking import chunk_filing, make_splitter
from rag.store import WINDOW_DAYS, FilingStore, open_client

CASES = Path("results/signal_probe_200cases.csv")


def pick_pilot(cases: pd.DataFrame, n: int = 20) -> list[int]:
    """Filers with text-borne warnings first, then a few healthy ones."""
    picked: list[int] = []

    def take(frame: pd.DataFrame, k: int) -> None:
        for cik in sorted(frame.cik.unique()):
            if len(picked) >= n or k <= 0:
                return
            if int(cik) not in picked:
                picked.append(int(cik))
                k -= 1

    take(cases[cases.covenant == 1], 3)
    take(cases[cases.going_concern == 1], 7)
    take(cases[cases.material_weakness == 1], 5)
    take(cases[cases.label == 0], n)
    return picked[:n]


def reports_in_window(index: list[dict], start: date, end: date) -> list[dict]:
    return sorted(
        (
            r for r in index
            if isinstance(r.get("filing_date"), date)
            and start <= r["filing_date"] <= end
            and (r.get("form") or "").startswith(REPORT_FORMS)
            and r.get("primary_document")
        ),
        key=lambda r: r["filing_date"],
    )


def plan(ciks: list[int], cases: pd.DataFrame, store: FilingStore, edgar: EdgarClient, stats: dict) -> list:
    """(cik, report) pairs still to ingest, in order. Filing indexes are cached."""
    todo = []
    for cik in ciks:
        as_ofs = [date.fromisoformat(a) for a in cases[cases.cik == cik].as_of]
        start, end = min(as_ofs) - timedelta(days=WINDOW_DAYS), max(as_ofs)
        try:
            index = edgar.filing_index(cik)
        except EdgarError as exc:
            stats["errors"].append(f"{cik}: filing index: {exc}")
            continue
        stats["filers"] += 1
        for r in reports_in_window(index, start, end):
            if store.has_filing(r["accession"]):
                stats["filings_skipped"] += 1
            else:
                todo.append((cik, r))
    return todo


#: Filings downloaded ahead of the one being embedded. Downloading overlaps
#: embedding instead of alternating with it; three ahead bounds the raw text
#: held in memory to a few tens of MB.
PREFETCH = 3


def ingest(
    ciks: list[int], cases: pd.DataFrame, store: FilingStore, edgar: EdgarClient, cooldown: float = 0.0
) -> dict:
    splitter = make_splitter()
    stats = {"filers": 0, "filings_new": 0, "filings_skipped": 0, "chunks": 0, "errors": []}
    t0 = time.time()
    todo = plan(ciks, cases, store, edgar, stats)
    print(f"{len(todo)} filings to ingest ({stats['filings_skipped']} already stored)", flush=True)

    def fetch(item):
        cik, r = item
        return edgar.fetch_filing_document(cik, r["accession"], r["primary_document"])

    # One download thread: EDGAR's client throttles itself and is not built
    # for concurrent callers; one thread ahead is all the overlap needed.
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = deque(pool.submit(fetch, item) for item in todo[:PREFETCH])
        for n, (cik, r) in enumerate(todo):
            if n + PREFETCH < len(todo):
                pending.append(pool.submit(fetch, todo[n + PREFETCH]))
            future = pending.popleft()
            free = memguard.check()
            try:
                raw = future.result()
            except EdgarError as exc:
                stats["errors"].append(f"{cik} {r['accession']}: {exc}")
                continue
            chunks = chunk_filing(cik, r["accession"], r["form"], r["filing_date"], raw, splitter)
            del raw
            ts = time.time()
            store.add(chunks)
            busy = time.time() - ts
            stats["filings_new"] += 1
            stats["chunks"] += len(chunks)
            elapsed = time.time() - t0
            eta = elapsed / (n + 1) * (len(todo) - n - 1) / 60
            print(
                f"[{n + 1}/{len(todo)}] cik {cik} {r['form']:<7} {r['filing_date']} {len(chunks):>4} chunks "
                f"{busy:5.1f}s | RAM free {free:.1f} GB | ETA {eta:.0f} min",
                flush=True,
            )
            if cooldown:
                # Idle in proportion to the work just done, so the machine runs
                # at a duty cycle rather than flat out. Added after two hard
                # resets under sustained load while the charger's input kept
                # dropping in and out.
                time.sleep(busy * cooldown)
    stats["minutes"] = round((time.time() - t0) / 60, 1)
    return stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cases", type=Path, default=CASES)
    ap.add_argument("--pilot", type=int, default=20, help="number of filers in the pilot")
    ap.add_argument("--all", action="store_true", help="every filer in the case file")
    ap.add_argument("--store", choices=["local", "cloud"], default="local")
    ap.add_argument("--offline", action="store_true", help="use cached EDGAR documents only")
    ap.add_argument(
        "--pause-gb", type=float, default=memguard.PAUSE_GB, help="pause below this much free RAM"
    )
    ap.add_argument("--stop-gb", type=float, default=memguard.STOP_GB, help="stop below this much free RAM")
    ap.add_argument(
        "--cooldown", type=float, default=1.0,
        help="idle this many times each filing's embedding time (1.0 = ~50%% duty; 0 = flat out)",
    )
    args = ap.parse_args(argv)

    from agents.llm import load_env_file

    load_env_file()
    memguard.PAUSE_GB, memguard.STOP_GB = args.pause_gb, args.stop_gb
    cases = pd.read_csv(args.cases)
    ciks = sorted(int(c) for c in cases.cik.unique()) if args.all else pick_pilot(cases, args.pilot)
    print(f"RAM free {memguard.free_gb():.1f} GB; {len(ciks)} filers; store={args.store}", flush=True)

    from rag.embed import FastEmbedder

    embedder = FastEmbedder()
    print(f"embedding on {embedder.device}", flush=True)
    store = FilingStore(open_client(args.store), embedder)
    try:
        stats = ingest(ciks, cases, store, EdgarClient(offline=args.offline), cooldown=args.cooldown)
    except memguard.LowMemory as exc:
        print(f"\nSTOPPED: {exc}. Everything stored so far is kept; rerun to continue.")
        return 2
    print(
        f"\ndone: {stats['filers']} filers, {stats['filings_new']} new filings "
        f"({stats['filings_skipped']} already stored), {stats['chunks']} chunks, "
        f"{stats['minutes']} min, {len(stats['errors'])} errors"
    )
    for err in stats["errors"][:20]:
        print("  error:", err)
    return 0


if __name__ == "__main__":
    sys.exit(main())
