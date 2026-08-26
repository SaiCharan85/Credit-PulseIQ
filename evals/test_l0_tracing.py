"""L0: tracing must be invisible when off and harmless when broken.

Observability earns its place only if it cannot take the product down with it.
Two properties are tested here and both have teeth:

* **Off by default.** No keys, no client, no network call, no behaviour change.
  The app has to start and answer on a laptop whose ``.env`` has never heard of
  Langfuse.
* **Never raises.** A backend that is down, slow, or misconfigured produces a
  lost trace, never a lost answer. Every entry point swallows its own
  exceptions, and the tests prove it by breaking the client on purpose.

Plus redaction, checked on the way in rather than trusted to the backend: a
trace goes to a third party, and a key inside one is a leaked key.
"""

from __future__ import annotations

import pytest

from agents import tracing


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"):
        monkeypatch.delenv(var, raising=False)
    tracing.reset()
    yield
    tracing.reset()


def test_disabled_without_keys():
    assert not tracing.enabled()
    assert tracing.client() is None
    assert tracing.trace_url() == ""


def test_every_entry_point_is_a_no_op_when_off():
    with tracing.span("assess", kind="agent", input={"cik": 1}) as handle:
        assert handle is None
        tracing.update(handle, output={"signal": "severe_risk"})
    tracing.score("shipped", True)
    tracing.score("confidence", 0.8)
    tracing.score("signal", "severe_risk")
    tracing.flush()


def test_partial_configuration_stays_off(monkeypatch):
    """A public key with no secret is a half-finished setup, not a licence to
    try the call and fail slowly on every request."""
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    tracing.reset()
    assert not tracing.enabled()


def test_a_broken_backend_loses_the_trace_not_the_answer(monkeypatch):
    class Exploding:
        def start_as_current_observation(self, **_kw):
            raise RuntimeError("backend down")

        def score_current_trace(self, **_kw):
            raise RuntimeError("backend down")

        def get_trace_url(self):
            raise RuntimeError("backend down")

        def flush(self):
            raise RuntimeError("backend down")

    monkeypatch.setattr(tracing, "_client", Exploding())
    monkeypatch.setattr(tracing, "_checked", True)

    with tracing.span("assess", kind="agent") as handle:
        assert handle is None          # degraded, not raised
        tracing.update(handle, output={"x": 1})
    tracing.score("shipped", True)
    assert tracing.trace_url() == ""
    tracing.flush()


def test_a_client_that_fails_to_construct_is_treated_as_absent(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    tracing.reset()
    import langfuse

    def boom(**_kw):
        raise RuntimeError("bad host")

    monkeypatch.setattr(langfuse, "Langfuse", boom)
    assert tracing.client() is None
    assert not tracing.enabled()


def test_the_client_is_built_once(monkeypatch):
    calls = []
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    tracing.reset()
    import langfuse

    monkeypatch.setattr(langfuse, "Langfuse", lambda **kw: calls.append(kw) or object())
    tracing.client()
    tracing.client()
    tracing.client()
    assert len(calls) == 1, "a client per request would open a connection per request"


# ---- redaction ------------------------------------------------------------

@pytest.mark.parametrize(
    "field",
    ["api_key", "API_KEY", "secret_key", "token", "Authorization", "password",
     "groq_api_key"],
)
def test_secrets_are_redacted_before_leaving(field):
    out = tracing.redact({field: "gsk_live_abcdef123456", "cik": 28823})
    assert out[field] == "<redacted>"
    assert out["cik"] == 28823


def test_redaction_reaches_nested_structures():
    out = tracing.redact(
        {"tools": [{"name": "get_metric", "headers": {"authorization": "Bearer x"}}]}
    )
    assert out["tools"][0]["headers"]["authorization"] == "<redacted>"
    assert out["tools"][0]["name"] == "get_metric"


def test_ordinary_arguments_survive_redaction():
    args = {"metric": "current_ratio", "period_end": "2022-12-31", "cik": 28823}
    assert tracing.redact(args) == args


def test_scores_accept_the_three_shapes_we_record(monkeypatch):
    """Booleans must not arrive as numerics: "did the guard ship this" should
    read as a pass rate, not an average of ones and zeros."""
    seen = []

    class Recorder:
        def score_current_trace(self, **kw):
            seen.append(kw)

    monkeypatch.setattr(tracing, "_client", Recorder())
    monkeypatch.setattr(tracing, "_checked", True)
    tracing.score("shipped", True)
    tracing.score("confidence", 0.8)
    tracing.score("signal", "severe_risk")
    assert [s["data_type"] for s in seen] == ["BOOLEAN", "NUMERIC", "CATEGORICAL"]
