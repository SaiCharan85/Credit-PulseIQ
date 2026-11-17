"""Wait out an internet outage instead of failing through it.

Unattended runs on this laptop meet real outages -- the Wi-Fi has dropped DNS
for minutes at a time. Retrying with backoff covers a blip, but a longer gap
exhausts the retries and the case is recorded as a failure, which reads as the
*agent* failing when the network did. That is the wrong conclusion stated
confidently, so batch runs pause until connectivity returns.

Opt-in (``CREDITPULSE_WAIT_FOR_NETWORK=1``) because an interactive request in
the web app must fail fast, not hang for as long as the outage lasts.
"""

from __future__ import annotations

import os
import socket
import time

ENV = "CREDITPULSE_WAIT_FOR_NETWORK"

#: Probed in turn; any one answering means we are online. DNS resolution is
#: part of the probe because DNS failing was the outage actually observed.
PROBES = (("www.google.com", 443), ("1.1.1.1", 443), ("www.microsoft.com", 443))
POLL_S = 30


def enabled() -> bool:
    return os.environ.get(ENV, "") not in ("", "0", "false")


def online(timeout: float = 5.0) -> bool:
    for host, port in PROBES:
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except OSError:
            continue
    return False


def wait_until_online(log=print) -> float:
    """Block until the internet answers again; return seconds waited."""
    start = time.monotonic()
    n = 0
    while not online():
        if n % 2 == 0:  # once a minute
            log(f"  [network] internet unreachable for {time.monotonic() - start:.0f}s; waiting")
        time.sleep(POLL_S)
        n += 1
    waited = time.monotonic() - start
    if waited:
        log(f"  [network] back online after {waited:.0f}s; resuming")
    return waited
