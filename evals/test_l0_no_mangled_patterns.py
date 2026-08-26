"""L0: no regex in this codebase contains a control character.

Written after the same bug landed three times in one session.

Editing a pattern through a shell heredoc turns ``\\b`` into 0x08 -- a literal
backspace. The pattern still compiles. It just stops matching word boundaries,
and therefore stops matching anything:

* ``_NEGATED`` silently never fired, so a correct refusal ("the assessment does
  not predict whether the company will go bankrupt") was graded as a forecast;
* ``_REFERS_TO_LOADED`` matched nothing, so every company question was routed
  to the general-definitions path and answered uselessly.

Neither failure raised. Both looked like a model problem and cost real time to
trace back to punctuation. A pattern that silently matches nothing is worse
than one that crashes, so this sweeps every compiled regex in the project and
fails loudly if a control character appears in one.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("agents", "data", "models", "compute", "evals", "verify")

#: Control characters that a mangled escape leaves behind. Tab and newline are
#: legitimate inside a verbose pattern, so they are not listed.
BAD = {
    "\x07": r"\a", "\x08": r"\b", "\x0b": r"\v", "\x0c": r"\f", "\x00": r"\0",
}


def _python_files() -> list[Path]:
    files = [ROOT / "serve.py"]
    for package in PACKAGES:
        files.extend(sorted((ROOT / package).rglob("*.py")))
    return [f for f in files if f.exists()]


@pytest.mark.parametrize("path", _python_files(), ids=lambda p: p.name)
def test_no_control_characters_in_source(path: Path) -> None:
    text = path.read_text(encoding="utf8")
    for char, meant in BAD.items():
        if char in text:
            line = text[: text.index(char)].count("\n") + 1
            pytest.fail(
                f"{path.relative_to(ROOT)}:{line} contains a literal "
                f"{char!r} where {meant} was almost certainly meant. A shell "
                "heredoc mangles regex escapes -- rewrite the pattern with the "
                "editor rather than through a shell."
            )


def test_the_patterns_that_were_broken_now_match() -> None:
    """Named explicitly, because a generic sweep would not have caught the
    original bugs before they shipped -- these are the two that did."""
    import serve
    from evals.run_response_eval import _NEGATED

    assert serve._REFERS_TO_LOADED.search("how leveraged is it")
    assert not serve._REFERS_TO_LOADED.search("what does going concern mean")
    assert _NEGATED.search("The assessment does not predict whether the company ")


def test_every_compiled_pattern_still_compiles() -> None:
    """Cheap belt-and-braces: import the modules that own the patterns and
    confirm the regex objects are real."""
    import agents.qa as qa
    import serve

    for module in (qa, serve):
        for name in dir(module):
            value = getattr(module, name)
            if isinstance(value, re.Pattern):
                assert value.pattern, f"{module.__name__}.{name} is empty"
                for char in BAD:
                    assert char not in value.pattern, (
                        f"{module.__name__}.{name} carries a literal {char!r}"
                    )
