"""Stop before the machine runs out of memory, not after.

On this workstation the editor's helper processes are the first thing the OS
kills under memory pressure -- every one of them crashed in the same second at
0.3 GB free. A long ingestion that pushes free RAM that low takes the editor
(and the terminal running it) down with it.

The laptop also hard-reset once mid-ingestion, on battery, with every core
busy and 1.7 GB free; the thresholds were raised after that.

So ingestion checks free memory between filings. Below ``PAUSE_GB`` it waits
for memory to come back; below ``STOP_GB`` it stops cleanly. Ingestion skips
filings already stored, so a stop costs nothing but the rerun.
"""

from __future__ import annotations

import time

import psutil

PAUSE_GB = 2.5
STOP_GB = 1.5
WAIT_S = 10
#: How long to wait for memory before giving up. ``None`` waits indefinitely,
#: which lets a run resume by itself the moment the user frees memory; the
#: STOP_GB floor still ends it if memory keeps falling.
MAX_WAITS: int | None = None


class LowMemory(RuntimeError):
    pass


def free_gb() -> float:
    return psutil.virtual_memory().available / 2**30


def check(log=print) -> float:
    """Return free GB, pausing or raising ``LowMemory`` as thresholds require."""
    free = free_gb()
    waits = 0
    while free < PAUSE_GB:
        if free < STOP_GB:
            raise LowMemory(f"only {free:.2f} GB RAM free (stop threshold {STOP_GB} GB)")
        if MAX_WAITS is not None and waits >= MAX_WAITS:
            raise LowMemory(f"RAM stayed below {PAUSE_GB} GB for {WAIT_S * MAX_WAITS}s")
        if waits % 6 == 0:  # once a minute, not every ten seconds
            log(f"  [memguard] {free:.2f} GB free < {PAUSE_GB} GB; waiting "
                "(close browser tabs, pause OneDrive, or close editor windows to resume)")
        time.sleep(WAIT_S)
        waits += 1
        free = free_gb()
    return free
