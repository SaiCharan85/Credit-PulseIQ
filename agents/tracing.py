"""Langfuse tracing for the agent loop, the guards and the Q&A path.

What this is for. The backtest tells you the agent scores 0.963 over 200 cases;
it tells you nothing about the run that just happened on the screen in front of
you. When a memo comes back blocked, or an investigation takes 130 seconds, or
an answer is withheld for a bad citation, the question is always "on this one,
what did it actually do" -- and that needs a trace, not an aggregate.

Three principles, each of which has a cost if broken:

**Optional, and silent when absent.** No keys configured means every call here
is a no-op. Observability that makes the product refuse to start is worse than
none, and this runs on a laptop with a `.env` that may not have Langfuse in it.

**Never raises.** A tracing backend being down, slow or misconfigured must not
fail an assessment. Every public function swallows its own exceptions. The one
thing worse than losing a trace is losing the answer the trace described.

**Nothing secret leaves.** Traces carry tool names, arguments, verdicts and
timings. They do not carry API keys, and the redaction is on the way in rather
than trusted to the backend.

Scores, not just spans. The interesting quantities here are not latency -- they
are whether the guard shipped the memo, how many cited figures verified against
their measure and period, and whether the investigation ended by concluding or
by running out of steps. Those go up as Langfuse scores so they can be tracked
across runs instead of read one trace at a time.

    LANGFUSE_PUBLIC_KEY=pk-...
    LANGFUSE_SECRET_KEY=sk-...
    LANGFUSE_HOST=https://cloud.langfuse.com     # or a self-hosted URL
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any

#: Substrings that mark a value as a secret. Redacted before anything is sent,
#: because a trace is a third-party service and a key in it is a leaked key.
_SECRET_HINTS = ("key", "token", "secret", "password", "authorization", "api_key")

_client: Any = None
_checked = False


def enabled() -> bool:
    """Whether tracing is configured. Cheap enough to call per request."""
    return client() is not None


def client() -> Any:
    """The Langfuse client, or None. Constructed once, never raises."""
    global _client, _checked
    if _checked:
        return _client
    _checked = True
    public = os.environ.get("LANGFUSE_PUBLIC_KEY")
    secret = os.environ.get("LANGFUSE_SECRET_KEY")
    if not (public and secret):
        return None
    try:
        from langfuse import Langfuse

        _client = Langfuse(
            public_key=public,
            secret_key=secret,
            host=os.environ.get("LANGFUSE_HOST") or "https://cloud.langfuse.com",
            environment=os.environ.get("CREDITPULSE_ENV") or "local",
        )
    except Exception:  # noqa: BLE001 - a broken backend must not break the app
        _client = None
    return _client


def reset() -> None:
    """Forget the cached client. For tests that change the environment."""
    global _client, _checked
    _client, _checked = None, False


def redact(value: Any) -> Any:
    """Strip anything that looks like a credential, at any depth."""
    if isinstance(value, dict):
        return {
            k: ("<redacted>" if any(h in str(k).lower() for h in _SECRET_HINTS)
                else redact(v))
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact(v) for v in value]
    return value


@contextmanager
def span(name: str, kind: str = "span", **fields: Any):
    """One observation. Yields a handle, or None when tracing is off.

    ``kind`` maps to Langfuse's observation types -- "agent" for the whole
    investigation, "tool" for a single tool call, "generation" for a model
    call, "guardrail" for a guard verdict. Using the real types rather than
    plain spans is what makes the trace legible as an *agent* run instead of a
    stack of timers.
    """
    lf = client()
    if lf is None:
        yield None
        return
    handle = None
    try:
        ctx = lf.start_as_current_observation(
            name=name, as_type=kind, **{k: redact(v) for k, v in fields.items()}
        )
        handle = ctx.__enter__()
    except Exception:  # noqa: BLE001
        yield None
        return
    try:
        yield handle
    finally:
        try:
            ctx.__exit__(None, None, None)
        except Exception:  # noqa: BLE001
            pass


def update(handle: Any, **fields: Any) -> None:
    """Attach output or metadata to an open observation."""
    if handle is None:
        return
    try:
        handle.update(**{k: redact(v) for k, v in fields.items()})
    except Exception:  # noqa: BLE001
        pass


def score(name: str, value: float | str | bool, comment: str = "") -> None:
    """Record a quality score against the current trace.

    Booleans are sent as Langfuse BOOLEAN scores rather than 0/1 numerics, so
    "did the guard ship this memo" reads as a pass rate in the dashboard
    instead of an average of ones and zeros.
    """
    lf = client()
    if lf is None:
        return
    try:
        if isinstance(value, bool):
            lf.score_current_trace(name=name, value=int(value), data_type="BOOLEAN",
                                   comment=comment or None)
        elif isinstance(value, str):
            lf.score_current_trace(name=name, value=value, data_type="CATEGORICAL",
                                   comment=comment or None)
        else:
            lf.score_current_trace(name=name, value=float(value), data_type="NUMERIC",
                                   comment=comment or None)
    except Exception:  # noqa: BLE001
        pass


def trace_url() -> str:
    """A link to the trace just recorded, for the UI to show. "" when off."""
    lf = client()
    if lf is None:
        return ""
    try:
        return lf.get_trace_url() or ""
    except Exception:  # noqa: BLE001
        return ""


def flush() -> None:
    """Push buffered events. Called at the end of a request, never blocking."""
    lf = client()
    if lf is None:
        return
    try:
        lf.flush()
    except Exception:  # noqa: BLE001
        pass
