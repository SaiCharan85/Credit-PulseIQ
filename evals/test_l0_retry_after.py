"""L0: honour the provider's stated wait on a rate limit.

This parser was broken and nobody noticed. A shell heredoc turned ``\\b`` into a
backspace, the pattern compiled fine, and it silently matched nothing -- so
every rate limit fell through to exponential backoff and the provider's own
"retry in 1.5s" was discarded.

That failure mode is the reason this file exists rather than only the
character sweep in ``test_l0_no_mangled_patterns``. The sweep catches *that*
corruption; it cannot catch a parser that is merely wrong. A regex returning
None on every input looks exactly like a provider that never sends a hint, and
backoff still "works" -- slower, and out of step with what we were told.

Getting it wrong costs in both directions. Waiting too long stalls a backtest
that is already hours long; retrying too soon earns another 429 and, on a
free tier, can burn the daily quota on rejected requests.
"""

from __future__ import annotations

import pytest

from agents.llm import RateLimitedClient

parse = RateLimitedClient._retry_after


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("Rate limit reached for model, please retry in 1.5s", 1.5),
        ("Rate limit reached, retry in 20s", 20.0),
        ("429 Too Many Requests. retry in 250ms", 0.25),
        ("retry in 1000ms", 1.0),
        ("Please retry in 0.5s and try again", 0.5),
        ("RETRY IN 3S", 3.0),
    ],
)
def test_a_stated_wait_is_read(message: str, expected: float) -> None:
    assert parse(Exception(message)) == pytest.approx(expected)


def test_milliseconds_and_seconds_are_not_confused() -> None:
    """250ms is a quarter second, not 250. Treating the unit as decoration
    turns a quarter-second pause into a four-minute one."""
    assert parse(Exception("retry in 250ms")) == pytest.approx(0.25)
    assert parse(Exception("retry in 250s")) == pytest.approx(250.0)


@pytest.mark.parametrize(
    "message",
    [
        "no hint at all here",
        "Rate limit exceeded",
        "429 Too Many Requests",
        "",
        "retry in a moment",
        "retry in soon",
    ],
)
def test_no_hint_returns_none_rather_than_a_guess(message: str) -> None:
    """None means "fall back to backoff". A fabricated number here would be a
    wait invented by us and attributed to the provider."""
    assert parse(Exception(message)) is None


def test_it_never_raises_on_an_odd_exception() -> None:
    """This runs inside the retry path. An exception raised while handling an
    exception loses the original error and the request with it."""

    class Odd(Exception):
        def __str__(self) -> str:
            return "\x00� retry in \x00"

    assert parse(Odd()) is None
    assert parse(Exception(str(None))) is None


def test_the_pattern_carries_no_control_characters() -> None:
    """The specific corruption that broke this, asserted where it broke."""
    import inspect

    source = inspect.getsource(RateLimitedClient._retry_after)
    for char in ("\x07", "\x08", "\x0b", "\x0c", "\x00"):
        assert char not in source, (
            f"a literal {char!r} is back in the retry parser -- it was almost "
            "certainly meant to be a regex escape, and the pattern will "
            "compile while matching nothing"
        )
