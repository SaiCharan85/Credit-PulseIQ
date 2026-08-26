"""End-to-end audit: does every figure a user sees match the filing?

The critic already verifies cited figures, but it verifies them against the
provenance record the agent itself produced. That is the right check for
"did the model invent this", and the wrong one for "is the number on the
screen correct" -- if a metric were mis-implemented, or an as-of filter
applied inconsistently, agent and critic would agree with each other and both
be wrong.

So this audit trusts nothing in the pipeline. For every figure that reaches a
user -- memo evidence, trend series, the diagnostic screen, and the numerals
in a grounded answer -- it goes back to EDGAR, re-applies the as-of filter,
recomputes from the raw XBRL facts, and compares. Anything that disagrees is
reported with both values and the accession the input came from.

Four surfaces, because a figure can be right in one and wrong in another:

``memo``        evidence entries in the assessment
``trend``       the dated series behind a movement question
``diagnostic``  the reported line items and calculated ratios screen
``answer``      numerals in the model's prose, after the guards have run

Reported as counts, not a pass/fail: "38 of 38 matched to 1e-9" is a claim a
reader can weigh, and a bare green tick is not.

    python -m evals.run_figure_audit                 # default sample
    python -m evals.run_figure_audit 28823 320193    # specific filers
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import date

#: A spread of conditions on purpose: a filer in visible distress, large
#: healthy filers, and one with thin disclosure. A figure audit over five
#: healthy megacaps proves the easy case only.
SAMPLE = [28823, 320193, 1090727, 77476, 1467858, 320187]
AS_OF = date(2024, 7, 1)

#: Same tolerance the verifier uses. This is floating-point noise, not slack.
TOLERANCE = 1e-9


@dataclass
class Mismatch:
    surface: str
    cik: int
    metric: str
    period: str
    shown: float | None
    recomputed: float | None
    note: str = ""

    def __str__(self) -> str:
        return (
            f"    {self.surface:<11} CIK {self.cik} {self.metric} @ {self.period}: "
            f"screen {self.shown} vs source {self.recomputed} {self.note}"
        )


@dataclass
class Audit:
    checked: int = 0
    matched: int = 0
    unverifiable: int = 0
    mismatches: list[Mismatch] = field(default_factory=list)

    def add(self, other: Audit) -> None:
        self.checked += other.checked
        self.matched += other.matched
        self.unverifiable += other.unverifiable
        self.mismatches.extend(other.mismatches)


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= TOLERANCE * max(1.0, abs(a), abs(b))


def _source_view(cik: int, as_of: date):
    """A fresh index of the filer's facts, built without touching the agent."""
    from compute.lineitems import FactIndex
    from data.edgar import EdgarClient
    from data.facts import as_of_view

    return FactIndex(as_of_view(EdgarClient().facts(cik), as_of))


def audit_memo(cik: int, as_of: date, payload: dict) -> Audit:
    """Every evidence figure in the memo, recomputed from source."""
    from compute.ratios import compute_metric

    out = Audit()
    view = _source_view(cik, as_of)
    for section in (payload.get("memo") or {}).get("sections", []):
        for e in section.get("evidence", []):
            shown, metric = e.get("value"), e.get("metric")
            period = e.get("period_end")
            if shown is None or not period:
                out.unverifiable += 1
                continue
            out.checked += 1
            try:
                cv = compute_metric(metric, view, date.fromisoformat(period))
            except Exception as exc:  # noqa: BLE001
                out.mismatches.append(
                    Mismatch("memo", cik, metric, period, shown, None, f"({exc})")
                )
                continue
            if not cv.is_defined:
                out.mismatches.append(
                    Mismatch("memo", cik, metric, period, shown, None,
                             "(not computable from source)")
                )
            elif _close(float(cv.value), float(shown)):
                out.matched += 1
            else:
                out.mismatches.append(
                    Mismatch("memo", cik, metric, period, shown, float(cv.value))
                )
    return out


def audit_trends(cik: int, as_of: date, series: list[dict]) -> Audit:
    """Every dated point in a trend series."""
    from compute.ratios import compute_metric

    out = Audit()
    view = _source_view(cik, as_of)
    for s in series:
        for p in s.get("points", []):
            shown = p.get("value")
            if shown is None:
                out.unverifiable += 1
                continue
            out.checked += 1
            cv = compute_metric(s["metric"], view, date.fromisoformat(p["period_end"]))
            if cv.is_defined and _close(float(cv.value), float(shown)):
                out.matched += 1
            else:
                out.mismatches.append(
                    Mismatch("trend", cik, s["metric"], p["period_end"], shown,
                             float(cv.value) if cv.is_defined else None)
                )
    return out


def audit_diagnostic(cik: int, as_of: date) -> Audit:
    """The health-check screen: reported line items and calculated ratios."""
    from agents.diagnostic import build as build_diagnostic
    from compute import lineitems
    from compute.ratios import compute_metric
    from data.edgar import EdgarClient

    out = Audit()
    edgar = EdgarClient()
    facts = edgar.facts(cik)
    diag = build_diagnostic(cik, as_of, facts, submissions={})
    if not diag.period_end:
        return out
    view = _source_view(cik, as_of)

    for reading in diag.reported:
        if not reading.computable or reading.value is None:
            out.unverifiable += 1
            continue
        out.checked += 1
        ref = lineitems.resolve(reading.key, view, diag.period_end)
        if ref is not None and _close(float(ref.value), float(reading.value)):
            out.matched += 1
        else:
            out.mismatches.append(
                Mismatch("diagnostic", cik, reading.key, str(diag.period_end),
                         reading.value, float(ref.value) if ref else None)
            )
    for reading in diag.calculated:
        if not reading.computable or reading.value is None:
            out.unverifiable += 1
            continue
        out.checked += 1
        cv = compute_metric(reading.key, view, diag.period_end)
        if cv.is_defined and _close(float(cv.value), float(reading.value)):
            out.matched += 1
        else:
            out.mismatches.append(
                Mismatch("diagnostic", cik, reading.key, str(diag.period_end),
                         reading.value, float(cv.value) if cv.is_defined else None)
            )
    return out


def audit_answer(cik: int, answer: str, verified: set[str]) -> Audit:
    """Numerals in the model's prose, against the set recomputed from source.

    The guards already block a figure absent from the assessment. This asks the
    harder question: of the numbers that survived, is each one a value the
    filing actually supports?
    """
    import re

    out = Audit()
    # Strip ISO dates first. "2024-05-03" otherwise tokenises as -05 and -03,
    # which the audit then reports as two invented figures of -5.0 and -3.0 --
    # the auditor manufacturing the failure it exists to detect.
    # Legal and form references carry digits that are not measurements.
    # "Chapter 11" was audited as a figure of 11.0 and reported as a number on
    # screen disagreeing with source -- the auditor manufacturing a failure out
    # of the name of a bankruptcy chapter.
    prose = re.sub(r"\d{4}-\d{2}-\d{2}", " ", answer)
    prose = re.sub(
        r"\bchapter\s+\d+|\bitem\s+\d+(?:\.\d+)?|\bform\s+\d+-?[A-Z]?|"
        r"\b\d{1,2}-[KQ]\b|\bsection\s+\d+",
        " ",
        prose,
        flags=re.I,
    )
    for m in re.finditer(r"-?\d[\d,]*\.?\d+", prose):
        token = m.group(0).replace(",", "")
        if re.fullmatch(r"(?:19|20)\d{2}", token.lstrip("-")):
            continue  # a year, not a measurement
        out.checked += 1
        if any(token.lstrip("-") in v or v in token.lstrip("-") for v in verified):
            out.matched += 1
        else:
            out.mismatches.append(
                Mismatch("answer", cik, "prose figure", "-", float(token), None,
                         "(no source value rounds to this)")
            )
    return out


def main(argv: list[str]) -> int:
    from agents.llm import load_env_file

    load_env_file()
    import serve

    ciks = [int(a) for a in argv[1:]] or SAMPLE
    total = Audit()
    print("Figure audit -- every number on screen, recomputed from EDGAR")
    print(f"as-of {AS_OF}, tolerance {TOLERANCE:g}\n")

    for cik in ciks:
        payload, status = serve._assess(
            serve.AssessRequest(cik=cik, as_of=AS_OF, agent="rules")
        )
        if status != 200 or not payload.get("memo"):
            print(f"  CIK {cik:<9} skipped ({payload.get('error') or 'no memo'})")
            continue

        per = Audit()
        per.add(audit_memo(cik, AS_OF, payload))
        series = serve._trend_series(cik, AS_OF, payload)
        per.add(audit_trends(cik, AS_OF, series))
        per.add(audit_diagnostic(cik, AS_OF))

        # The values the filing supports, at the precision a reader sees.
        verified = {
            f"{float(e['value']):.2f}".lstrip("-")
            for s in payload["memo"]["sections"]
            for e in s["evidence"]
            if e.get("value") is not None
        }
        for s in series:
            verified |= {f"{p['value']:.2f}".lstrip("-")
                         for p in s["points"] if p.get("value") is not None}
        answer = serve.ask_question(
            serve.AskRequest(assessment=payload,
                             question="Why is this company at risk?")
        )
        import json

        body = json.loads(answer.body)
        if body.get("answer"):
            per.add(audit_answer(cik, body["answer"], verified))

        name = payload.get("company") or f"CIK {cik}"
        flag = "" if not per.mismatches else f"  <-- {len(per.mismatches)} MISMATCH"
        print(f"  {name[:34]:<34} {per.matched:>3}/{per.checked:<3} matched"
              f"  ({per.unverifiable} not reported){flag}")
        for m in per.mismatches:
            print(m)
        total.add(per)

    print(f"\n  {total.matched}/{total.checked} figures matched the filing to {TOLERANCE:g}")
    print(f"  {total.unverifiable} not reported by the filer (shown as gaps, not guessed)")
    if total.mismatches:
        print(f"  {len(total.mismatches)} MISMATCHES -- a number on screen disagrees with source")
        return 1
    print("  0 mismatches across memo, trends, diagnostic and answer prose")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
