"""The shared token ledger: one per-minute budget across clients, threads and processes."""

from __future__ import annotations

import threading

from agents import llm
from agents.llm import TPM_LEDGER_ENV, RateLimitedClient


class _Echo:
    name = "echo"

    def complete(self, messages, **kwargs):
        return "ok"


def _client(tpm: int) -> RateLimitedClient:
    return RateLimitedClient(inner=_Echo(), tokens_per_minute=tpm)


def test_two_clients_share_one_budget(tmp_path, monkeypatch):
    monkeypatch.setenv(TPM_LEDGER_ENV, str(tmp_path / "ledger.txt"))
    slept: list[float] = []

    class Stop(Exception):
        pass

    def fake_sleep(s):
        slept.append(s)
        raise Stop

    monkeypatch.setattr(llm.time, "sleep", fake_sleep)
    a, b = _client(1000), _client(1000)
    a._pace(700)
    # b is a different client (stand-in for another process) but sees a's spend.
    try:
        b._pace(700)
    except Stop:
        pass
    assert slept and 59 < slept[0] <= 60.5


def test_under_budget_does_not_wait(tmp_path, monkeypatch):
    monkeypatch.setenv(TPM_LEDGER_ENV, str(tmp_path / "ledger.txt"))
    monkeypatch.setattr(llm.time, "sleep", lambda s: (_ for _ in ()).throw(AssertionError("slept")))
    a, b = _client(1000), _client(1000)
    a._pace(400)
    b._pace(400)
    assert len((tmp_path / "ledger.txt").read_text().splitlines()) == 2


def test_oversized_request_alone_is_allowed(tmp_path, monkeypatch):
    monkeypatch.setenv(TPM_LEDGER_ENV, str(tmp_path / "ledger.txt"))
    monkeypatch.setattr(llm.time, "sleep", lambda s: (_ for _ in ()).throw(AssertionError("slept")))
    _client(1000)._pace(5000)  # bigger than the ceiling, but nothing else in the window


def test_threads_record_every_call(tmp_path, monkeypatch):
    ledger = tmp_path / "ledger.txt"
    monkeypatch.setenv(TPM_LEDGER_ENV, str(ledger))
    c = _client(10**9)
    threads = [threading.Thread(target=c._pace, args=(10,)) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(ledger.read_text().splitlines()) == 16
