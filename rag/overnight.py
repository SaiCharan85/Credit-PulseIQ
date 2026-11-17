"""Unattended pipeline: finish ingestion, evaluate retrieval, run the A/B backtest.

    python -m rag.overnight          # logs to results/overnight.log

Steps, each logged with timestamps, each continuing even if a later one fails:

1. Ingest the remaining filings to the cloud store, flat out, retrying when
   the memory guard stops it.
2. Recall@k over every case, against the cloud store.
3. A/B backtest on the identical stratified 200 cases: the tool loop replayed
   from the LLM cache (no API calls), then the single-pass arm live.
4. Restart OneDrive and let Windows sleep again.

Keeps Windows awake while running (a lid close may still sleep the machine,
depending on power settings) and pauses OneDrive, whose sync of freshly
downloaded filings held ~1.9 GB the memory guard needed.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import time
from pathlib import Path

LOG = Path("results/overnight.log")
ONEDRIVE = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "OneDrive" / "OneDrive.exe"
PY = sys.executable
#: Batch steps wait out internet outages instead of failing through them (agents/netwait.py).
#: Every step paces against one token ledger, so parallel arms share the quota.
ENV = {**os.environ, "PYTHONIOENCODING": "utf8", "CREDITPULSE_WAIT_FOR_NETWORK": "1",
       "CREDITPULSE_TPM_LEDGER": str(Path("data/cache/tpm_ledger.txt").resolve())}


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf8") as f:
        f.write(line + "\n")


LLM_CACHE = Path("data/cache/llm")


def _latest_llm_write() -> float:
    try:
        return max((p.stat().st_mtime for p in LLM_CACHE.iterdir()), default=0.0)
    except OSError:
        return 0.0


def run(name: str, args: list[str], out: str, stall_minutes: float = 0, restarts: int = 3) -> int:
    """Run one step. With ``stall_minutes``, a step that writes no new LLM
    response for that long is killed and restarted -- safe because completed
    calls are cached, so a restart resumes. Added after a backtest hung for five
    hours alive but silent."""
    from agents import netwait

    for attempt in range(restarts + 1):
        if not netwait.online():
            log(f"WAIT  {name}: internet down; pausing until it returns")
            netwait.wait_until_online(log)
        log(f"START {name}{f' (restart {attempt})' if attempt else ''}: {' '.join(args)}  -> {out}")
        with open(out, "a", encoding="utf8") as f:
            proc = subprocess.Popen([PY, "-m", *args], stdout=f, stderr=subprocess.STDOUT, env=ENV)
            started = time.time()
            stalled = False
            while proc.poll() is None:
                time.sleep(60)
                if stall_minutes:
                    last = max(_latest_llm_write(), started)
                    if time.time() - last > stall_minutes * 60 and netwait.online():
                        log(f"STALL {name}: no new LLM response for {stall_minutes:.0f} min; restarting")
                        proc.kill()
                        proc.wait()
                        stalled = True
                        break
        code = proc.returncode
        log(f"END   {name}: exit {code}{' (stalled)' if stalled else ''}")
        if not stalled:
            return code
    return code


def keep_awake(on: bool) -> None:
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))


def main() -> int:
    keep_awake(True)
    log("pipeline start")
    if ONEDRIVE.exists():
        subprocess.call([str(ONEDRIVE), "/shutdown"])
        log("OneDrive paused (closed); restarted at the end")
    try:
        for attempt in range(1, 7):
            code = run(f"ingest attempt {attempt}",
                       ["rag.ingest", "--all", "--store", "cloud", "--cooldown", "0",
                        "--pause-gb", "2.0", "--stop-gb", "1.2"], "results/rag_ingest_full.log")
            if code == 0:
                break
            time.sleep(60)
        run("recall eval",
            ["evals.run_rag_recall", "--store", "cloud", "--out", "results/rag_recall_full.log"],
            "results/rag_recall_full_run.log")
        run("A/B arm 1: tool loop, cache replay",
            ["evals.run_l3", "--agent", "react", "--max-negatives", "100", "--offline",
             "--results", "data/cache/l3_react_replay.csv"], "results/backtest3_react_replay.log")
        run("A/B arm 2: single-pass, live",
            ["evals.run_l3", "--agent", "single", "--store", "cloud", "--max-negatives", "100",
             "--tpm", "16000", "--results", "data/cache/l3_single.csv"], "results/backtest3_single_run.log")
    finally:
        if ONEDRIVE.exists():
            subprocess.Popen([str(ONEDRIVE), "/background"])
            log("OneDrive restarted")
        keep_awake(False)
        log("pipeline end")
    return 0


if __name__ == "__main__":
    sys.exit(main())
