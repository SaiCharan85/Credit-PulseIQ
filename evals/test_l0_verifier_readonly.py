"""L0: the verifier is read-only and model-free, and both are enforced.

Why this is deterministic code and not a verification *agent*:

A second model checking the first is appealing and wrong here. It is
non-deterministic, so the same memo verifies differently on Tuesday; it can
hallucinate an approval as readily as the generator hallucinated the figure;
and it cannot actually check arithmetic, only opine on it. Looping a generator
against a model grader also optimises the grader rather than the output, which
is why that loop is forbidden outright in this project.

What runs instead re-executes the formula against the recorded raw inputs and
compares, to a tolerance tight enough (1e-9) that anything past floating-point
noise means the value did not come from that formula. A reader can repeat the
check by hand.

Two properties make it trustworthy, so neither is left to inspection of the
source:

* **Read-only.** A critic that can edit the figure it is judging turns a failed
  check into a silent repair, and the memo then claims a verification it never
  passed.
* **Model-free.** Tested by breaking the LLM client and confirming a verdict
  still comes back. If a model ever creeps into this path, that test fails.
"""

from __future__ import annotations

import copy
from datetime import date

import pytest

import compute.ratios  # noqa: F401 - registers the formulas the verifier re-runs
from agents.critic import review
from agents.schemas import Evidence, InvestigatorOutput
from compute.provenance import ComputedValue, FactRef

#: The real Diebold figures: 1,770,900,000 / 1,604,900,000 = 1.1034...
CURRENT_ASSETS = 1_770_900_000.0
CURRENT_LIABILITIES = 1_604_900_000.0
TRUE_RATIO = CURRENT_ASSETS / CURRENT_LIABILITIES
PERIOD = date(2022, 12, 31)
AS_OF = date(2024, 7, 1)


def _fact(concept: str, value: float) -> FactRef:
    return FactRef(
        concept=concept,
        tag={"current_assets": "AssetsCurrent",
             "current_liabilities": "LiabilitiesCurrent"}[concept],
        taxonomy="us-gaap",
        unit="USD",
        value=value,
        period_end=PERIOD,
        form="10-K",
        accession="0000028823-24-000025",
        filed=date(2024, 3, 8),
    )


def _computed(value: float) -> ComputedValue:
    """A claim that current_ratio equals ``value``, with the real inputs
    attached. The verifier re-runs the formula over those inputs, so a wrong
    ``value`` here is exactly the fabrication it exists to catch."""
    return ComputedValue(
        metric="current_ratio",
        formula="current_ratio",
        value=value,
        unit="ratio",
        period_end=PERIOD,
        inputs={
            "current_assets": _fact("current_assets", CURRENT_ASSETS),
            "current_liabilities": _fact("current_liabilities", CURRENT_LIABILITIES),
        },
    )


def _output(value: float) -> InvestigatorOutput:
    return InvestigatorOutput(
        cik=28823,
        as_of=AS_OF,
        signal="severe_risk",
        # 0.6 not 0.8: the critic caps confidence when the output itself
        # reports residual uncertainty, and tripping that check here would
        # mask the numeric one these tests are about.
        confidence=0.6,
        rationale="Liabilities exceed assets and interest is not covered.",
        residual="The newest filing is over a year old.",
        evidence=[Evidence(metric="current_ratio", value=value,
                           period_end=PERIOD, note="flag")],
    )


def test_the_verifier_does_not_touch_what_it_inspects() -> None:
    """The basis of the whole guarantee."""
    output = _output(TRUE_RATIO)
    cited = [_computed(TRUE_RATIO)]
    before_out = copy.deepcopy(output.model_dump())
    before_cited = copy.deepcopy([c.model_dump() for c in cited])

    review(output, cited=cited, as_of=AS_OF)

    assert output.model_dump() == before_out, "the critic mutated the output it judged"
    assert [c.model_dump() for c in cited] == before_cited, "the critic mutated the evidence"


def test_a_figure_that_reproduces_passes() -> None:
    report = review(_output(TRUE_RATIO), cited=[_computed(TRUE_RATIO)], as_of=AS_OF)
    assert not [d for d in report.defects if d.kind == "numeric"]


@pytest.mark.parametrize("claimed", [9.99, 0.0, -1.0, 1.2, 1.11])
def test_no_wrong_value_survives_the_tolerance(claimed: float) -> None:
    """The tolerance is for floating-point noise, not for being close enough.
    1.11 against a true 1.1034 is a different number, and "nearly right" is the
    failure a loose tolerance was invented to hide."""
    report = review(_output(claimed), cited=[_computed(claimed)], as_of=AS_OF)
    assert not report.passed, f"{claimed} accepted for a figure that is {TRUE_RATIO:.4f}"
    assert any(d.kind == "numeric" for d in report.defects)


def test_the_defect_names_both_numbers() -> None:
    """A blocked memo has to say what failed, or the guard is unauditable and
    nobody can tell a bug from a catch."""
    report = review(_output(9.99), cited=[_computed(9.99)], as_of=AS_OF)
    detail = next(d.detail for d in report.defects if d.kind == "numeric")
    assert "9.99" in detail and "1.10" in detail


def test_the_same_input_verifies_identically_every_time() -> None:
    """Determinism is the property a model-based verifier cannot offer. Ten
    runs, one answer -- otherwise "verified" means "verified the day we
    happened to look"."""
    verdicts = {
        review(_output(TRUE_RATIO), cited=[_computed(TRUE_RATIO)], as_of=AS_OF).passed
        for _ in range(10)
    }
    assert verdicts == {True}


def test_verification_needs_no_model_at_all(monkeypatch) -> None:
    """Break the LLM client entirely; a verdict must still come back. If this
    ever fails, a model has crept into the verification path."""
    import agents.llm as llm

    def boom(*_a, **_k):
        raise AssertionError("the verifier must not call a model")

    for name in ("default_client", "judge_client", "preflight"):
        monkeypatch.setattr(llm, name, boom, raising=False)

    assert review(_output(TRUE_RATIO), cited=[_computed(TRUE_RATIO)], as_of=AS_OF).passed


def test_a_figure_resting_on_a_later_filing_is_a_lookahead_defect() -> None:
    """Read-only does not mean permissive: the cardinal sin still fails here.
    This input was filed 2024-03-08, so an assessment dated 2023-01-01 could
    not have seen it."""
    report = review(_output(TRUE_RATIO), cited=[_computed(TRUE_RATIO)],
                    as_of=date(2023, 1, 1))
    assert not report.passed
    assert any(d.kind == "lookahead" for d in report.defects)
