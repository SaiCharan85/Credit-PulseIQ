"""Run the whole A/B unattended: both arms, every eval, then publish.

    python -m evals.ab_finish        # logs to results/overnight.log

Order, each step logged with timestamps and run even if an earlier one fails:

1+4. Both arms live on the stratified 199 cases, concurrently, sharing one
   token budget (the tool loop: 110 from the cache, 89 new).
2. Once single-pass ends: NLI faithfulness, context utilisation and answer
   relevance on its answers.
3. Retrieval Recall@5 and context precision@5 over every case.
   Then wait for the tool-loop arm.
5. Paired-bootstrap comparison on the cases both arms completed.
6. Both arms to Langfuse as runs on one dataset.

LLM steps run under a stall watchdog (rag.overnight.run): no new model response
for STALL_MIN minutes and the step is killed and restarted, resuming from the
cache. Batch steps wait out internet outages. Windows is kept awake, and
OneDrive -- closed earlier to free memory -- is restarted at the end.
"""

from __future__ import annotations

import subprocess
import sys

from rag.overnight import ONEDRIVE, keep_awake, log, run

#: Generous: the backtest spends ~10 minutes loading the panel before its first
#: model call, and a reasoning model can take a few minutes per reply.
STALL_MIN = 30


#: Both arms draw on one provider quota (~16K tokens/min per project). They
#: pace against a shared ledger (agents/llm.py) set a little under it, and each
#: runs WORKERS cases at once: reply latency, not tokens, was the bottleneck.
TPM = "15000"
WORKERS = "4"


def main() -> int:
    import threading

    keep_awake(True)
    log(f"A/B start (parallel arms, {WORKERS} workers each, shared {TPM} TPM ledger)")
    try:
        arms = [
            threading.Thread(target=run, args=(
                "single-pass arm, live",
                ["evals.run_l3", "--agent", "single", "--store", "cloud", "--max-negatives", "100",
                 "--tpm", TPM, "--workers", WORKERS, "--results", "data/cache/l3_single.csv"],
                "results/backtest3_single_run.log"), kwargs={"stall_minutes": STALL_MIN}),
            threading.Thread(target=run, args=(
                "tool-loop arm, live on all cases",
                ["evals.run_l3", "--agent", "react", "--max-negatives", "100", "--tpm", TPM,
                 "--workers", WORKERS, "--results", "data/cache/l3_react_full.csv"],
                "results/backtest3_react_full.log"), kwargs={"stall_minutes": STALL_MIN}),
        ]
        for t in arms:
            t.start()
        arms[0].join()
        run("faithfulness + relevance (NLI, no LLM)",
            ["evals.run_faithfulness", "--store", "cloud"], "results/faithfulness_run.log")
        run("recall + precision@5",
            ["evals.run_rag_recall", "--store", "cloud", "--out", "results/rag_recall_full.log"],
            "results/rag_recall_full_run.log")
        arms[1].join()
        run("A/B comparison (paired bootstrap)",
            ["evals.compare_arms", "data/cache/l3_react_full.csv", "data/cache/l3_single.csv"],
            "results/ab_comparison_run.log")
        run("Langfuse upload (both arms)",
            ["evals.to_langfuse", "--arm", "tool-loop-full=data/cache/l3_react_full.csv",
             "--arm", "single-pass-rag=data/cache/l3_single.csv"], "results/langfuse_upload.log")
    finally:
        if ONEDRIVE.exists():
            subprocess.Popen([str(ONEDRIVE), "/background"])
            log("OneDrive restarted")
        keep_awake(False)
        log("A/B end")
    return 0


if __name__ == "__main__":
    sys.exit(main())
