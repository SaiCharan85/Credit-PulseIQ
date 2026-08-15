"""Provenance types and the formula registry.

Every number in this system is a :class:`ComputedValue`: a value plus the exact
inputs and the named formula that produced it. Nothing downstream is allowed to
state a figure that is not one of these (SPEC 3, PROMPT hard rule 1).

The registry exists so that verification can be *independent*. ``verify/`` looks
up the formula by name and re-executes it against the recorded inputs; it never
trusts the stored ``value``. That is what makes a fabricated or drifted number a
hard failure rather than a matter of opinion.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from datetime import date

from pydantic import BaseModel, ConfigDict, Field

UNIT_RATIO = "ratio"
UNIT_USD = "USD"
UNIT_PERCENT = "percent"
UNIT_DAYS = "days"
UNIT_SCORE = "score"


class FactRef(BaseModel):
    """A single input value, traced to the filing it came from.

    Carries the actual XBRL tag used, not just the canonical concept, because
    concept resolution walks a fallback chain (see ``compute/lineitems.py``) and
    an auditor needs to know which tag was actually read.
    """

    model_config = ConfigDict(frozen=True)

    concept: str
    tag: str
    taxonomy: str = "us-gaap"
    unit: str = UNIT_USD
    value: float
    period_start: date | None = None
    period_end: date
    form: str = ""
    accession: str = ""
    filed: date

    @classmethod
    def from_fact(cls, concept: str, fact) -> FactRef:
        return cls(
            concept=concept,
            tag=fact.tag,
            taxonomy=fact.taxonomy,
            unit=fact.unit,
            value=fact.value,
            period_start=fact.period_start,
            period_end=fact.period_end,
            form=fact.form,
            accession=fact.accession,
            filed=fact.filed,
        )

    @property
    def citation(self) -> str:
        return f"{self.tag}@{self.period_end.isoformat()} ({self.form} {self.accession})"


class Formula:
    """A named, pure, deterministic computation."""

    __slots__ = ("name", "fn", "inputs", "unit", "expression", "doc")

    def __init__(
        self,
        name: str,
        fn: Callable[..., float | None],
        inputs: tuple[str, ...],
        unit: str,
        expression: str,
        doc: str = "",
    ) -> None:
        self.name = name
        self.fn = fn
        self.inputs = inputs
        self.unit = unit
        self.expression = expression
        self.doc = doc

    def __call__(self, **kwargs: float) -> float | None:
        return self.fn(**kwargs)


FORMULAS: dict[str, Formula] = {}


def formula(
    name: str, inputs: tuple[str, ...], unit: str, expression: str
) -> Callable[[Callable[..., float | None]], Formula]:
    """Register a pure function as a verifiable formula."""

    def wrap(fn: Callable[..., float | None]) -> Formula:
        if name in FORMULAS:
            raise ValueError(f"duplicate formula name: {name}")
        f = Formula(name, fn, inputs, unit, expression, (fn.__doc__ or "").strip())
        FORMULAS[name] = f
        return f

    return wrap


class ComputedValue(BaseModel):
    """A number this system is willing to state, with its full derivation."""

    model_config = ConfigDict(frozen=True)

    metric: str
    formula: str
    value: float | None
    unit: str
    period_end: date
    inputs: dict[str, FactRef] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)

    @property
    def as_of(self) -> date | None:
        """Earliest date this could have been computed.

        The latest filing date among its inputs. A ratio built from a balance
        sheet filed in March and earnings filed in May is not knowable until
        May -- taking the earliest, or the period end, would leak.
        """
        if not self.inputs:
            return None
        return max(ref.filed for ref in self.inputs.values())

    @property
    def is_defined(self) -> bool:
        return self.value is not None

    @property
    def citations(self) -> list[str]:
        return [ref.citation for ref in self.inputs.values()]

    def cited(self) -> str:
        """One-line rendering for a memo: value plus where it came from."""
        v = "undefined" if self.value is None else f"{self.value:,.4f}"
        return f"{self.metric}={v} [{self.formula}: {'; '.join(self.citations)}]"


def undefined(
    metric: str, formula_name: str, period_end: date, reason: str, inputs: Mapping[str, FactRef] | None = None
) -> ComputedValue:
    """A metric that could not be computed, and why.

    Returned instead of raising or substituting a default. A missing
    denominator is a finding the agent should see and can escalate on --
    silently coercing it to zero is how a distress signal gets erased.
    """
    unit = FORMULAS[formula_name].unit if formula_name in FORMULAS else UNIT_RATIO
    return ComputedValue(
        metric=metric,
        formula=formula_name,
        value=None,
        unit=unit,
        period_end=period_end,
        inputs=dict(inputs or {}),
        notes=[reason],
    )


def safe_div(numerator: float | None, denominator: float | None) -> float | None:
    """Division that returns ``None`` rather than raising or returning inf.

    Zero and near-zero denominators are common and meaningful in distressed
    financials (wiped-out equity, zero interest expense). ``None`` propagates
    as "undefined" and stays visible; ``inf`` would poison every downstream
    comparison silently.
    """
    if numerator is None or denominator is None:
        return None
    if abs(denominator) < 1e-12:
        return None
    return numerator / denominator


def build(
    metric: str,
    formula_name: str,
    period_end: date,
    inputs: Mapping[str, FactRef],
    notes: Iterable[str] = (),
) -> ComputedValue:
    """Execute a registered formula over resolved inputs, keeping provenance."""
    f = FORMULAS[formula_name]
    missing = [k for k in f.inputs if k not in inputs]
    if missing:
        return undefined(
            metric, formula_name, period_end, f"missing inputs: {', '.join(sorted(missing))}", inputs
        )
    value = f(**{k: inputs[k].value for k in f.inputs})
    note_list = list(notes)
    if value is None:
        note_list.append("undefined: division by zero or missing term")
    return ComputedValue(
        metric=metric,
        formula=formula_name,
        value=value,
        unit=f.unit,
        period_end=period_end,
        inputs={k: inputs[k] for k in f.inputs},
        notes=note_list,
    )
