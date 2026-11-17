"""Finish the A/B after the overnight run: retry what infrastructure broke, then publish.

    python -m evals.ab_complete [--react-pid PID]   # logs to results/overnight.log

The overnight run left two kinds of gap, neither the agents' doing:

* single-pass: 55 cases recorded as abstentions because Google refused the
  call (429/503) and the client gave up after a minute. The client now
  retries those for ~10 minutes.
* tool loop: its run was still finishing the last 19 cases.

Re-running a backtest replays every completed call from the cache, so only
the failed cases cost API calls. Order: single-pass re-run alongside the
still-running tool loop (shared token ledger); then the tool loop again only
if its results still hold failed cases; then the paired comparison and the
Langfuse upload, once, on clean results.
"""

from __future__ import annotations

import argparse
import csv
import sys
import threading
import time
from pathlib import Path

from rag.overnight import LLM_CACHE, ONEDRIVE, _latest_llm_write, keep_awake, log, run

SINGLE_CSV = Path("data/cache/l3_single.csv")
REACT_CSV = Path("data/cache/l3_react_full.csv")
N_CASES = 199
STALL_MIN = 30
COMMON = ["--max-negatives", "100", "--tpm", "15000", "--workers", "4"]


def gaps(path: Path) -> int:
    """Cases missing or failed for infrastructure reasons (not agent abstentions)."""
    if not path.exists():
        return N_CASES
    rows = list(csv.DictReader(path.open(encoding="utf8")))
    return (N_CASES - len(rows)) + sum(r["terminated_because"] == "case_error" for r in rows)


def wait_for(pid: int) -> None:
    """Wait for a running backtest; kill it only if it goes silent while online."""
    import psutil

    from agents import netwait

    try:
        proc = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    log(f"WAIT  tool-loop backtest pid {pid} to finish")
    started = time.time()
    while proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE:
        time.sleep(60)
        last = max(_latest_llm_write(), started)
        if time.time() - last > STALL_MIN * 60 and netwait.online():
            log(f"STALL tool-loop pid {pid}: no LLM response for {STALL_MIN} min; stopping it to re-run")
            proc.kill()
            break
    log(f"END   tool-loop backtest pid {pid}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--react-pid", type=int, default=0)
    args = parser.parse_args(argv)
    assert LLM_CACHE.exists()

    keep_awake(True)
    log(f"A/B completion start (single-pass gaps {gaps(SINGLE_CSV)}, tool loop running)")
    try:
        single = threading.Thread(target=run, args=(
            "single-pass re-run (infra failures only; rest from cache)",
            ["evals.run_l3", "--agent", "single", "--store", "cloud", *COMMON,
             "--results", str(SINGLE_CSV)],
            "results/backtest3_single_rerun.log"), kwargs={"stall_minutes": STALL_MIN})
        single.start()

        if args.react_pid:
            wait_for(args.react_pid)
        for attempt in range(3):
            left = gaps(REACT_CSV)
            if not left:
                break
            log(f"tool loop has {left} missing/failed cases; re-running (attempt {attempt + 1})")
            run("tool-loop re-run", ["evals.run_l3", "--agent", "react", *COMMON,
                                     "--results", str(REACT_CSV)],
                "results/backtest3_react_rerun.log", stall_minutes=STALL_MIN)

        single.join()
        log(f"gaps after re-runs: single-pass {gaps(SINGLE_CSV)}, tool loop {gaps(REACT_CSV)}")
        run("A/B comparison (paired bootstrap)",
            ["evals.compare_arms", str(REACT_CSV), str(SINGLE_CSV)], "results/ab_comparison_run.log")
        run("Langfuse upload (both arms)",
            ["evals.to_langfuse", "--arm", f"tool-loop-full={REACT_CSV}",
             "--arm", f"single-pass-rag={SINGLE_CSV}"], "results/langfuse_upload.log")
    finally:
        if ONEDRIVE.exists():
            import subprocess

            subprocess.Popen([str(ONEDRIVE), "/background"])
            log("OneDrive restarted")
        keep_awake(False)
        log("A/B completion end")
    return 0


if __name__ == "__main__":
    sys.exit(main())
