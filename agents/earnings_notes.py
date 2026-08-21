"""Earnings-quality observations, as context and nothing more.

The earnings-quality leg was built to be a second backtested predictor and it
failed. Four approaches over 891 test positives with a clean leak canary:

    Beneish M (published)                 0.512
    fitted logistic on ratios             0.579
    plus Beneish decomposed into terms    0.585
    filing-text and 8-K event signals     0.579

Nothing above 0.605 even after narrowing to income-decreasing restatements at
filers over $100M. Restatements are not predictable from filed financials in
this universe, and the honest response is to stop claiming otherwise.

But the *metrics* are real, deterministic and useful to a human. An analyst
reading that earnings have exceeded operating cash flow for four straight
quarters is better informed, whether or not that fact forecasts a restatement.
So they appear in the memo as observations under a ``context-only`` tier that
``Memo.graded_evidence`` excludes by construction -- a reader cannot mistake
them for the backtested signal, because the data model will not let them.

Two rules follow from the measurement and are enforced here:

**No thresholds that imply prediction.** These notes report values and
directions. They do not say "elevated risk of restatement", because the
measurement says we cannot tell.

**Absence is reported, not skipped.** Beneish M is computable on only 17% of
filers -- it needs eight ratios across two periods. Saying so is more useful
than silently omitting the line, since a reader would otherwise assume it was
checked and found clean.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from typing import Any

from compute.lineitems import FactIndex, annual_period_ends
from compute.ratios import compute_metric
from compute.scores import compute_two_period_score
from data.facts import as_of_view

#: Single-period observations worth stating, with how to read each.
SINGLE = (
    ("accruals_to_assets", "accruals as a share of assets (Sloan): earnings not backed by cash"),
    ("days_sales_outstanding", "days sales outstanding: how long revenue sits uncollected"),
    ("days_inventory_outstanding", "days inventory outstanding"),
    ("cash_conversion_cycle", "cash conversion cycle in days"),
)

#: Two-period indices. Beneish's own terms, reported individually because the
#: composite resolves on a small minority of filers.
TWO_PERIOD = (
    ("beneish_m_score", "Beneish M composite (above -1.78 is the conventional flag)"),
    ("beneish_dsri", "receivables growth vs sales growth (DSRI); above 1 means receivables ran ahead"),
    ("beneish_sgi", "sales growth index (SGI)"),
    ("beneish_gmi", "gross margin index (GMI); above 1 means margins deteriorated"),
)
# TATA, Beneish's heaviest term, is reported above as accruals_to_assets --
# arithmetically the same figure, and computable on far more filers.

#: Printed with the section so the standing of these numbers travels with them.
DISCLAIMER = (
    "These are descriptive accounting-quality observations. Measured against "
    "real restatement outcomes they did not predict them (AUC 0.51-0.61 across "
    "four approaches, clean leak canary), so they carry no forecast and do not "
    "move the graded signal."
)


def _fmt(value: float, metric: str) -> str:
    if "days" in metric or "cycle" in metric:
        return f"{value:,.0f} days"
    return f"{value:,.3g}"


def earnings_notes(facts: Sequence[Any], as_of: date) -> list[str]:
    """Deterministic accounting-quality observations for one filer at one date.

    Reads the same as-of view the investigator does, so nothing here can see a
    filing the graded leg could not.
    """
    view = FactIndex(as_of_view(facts, as_of))
    ends = annual_period_ends(view)
    if not ends:
        return ["No annual period is visible at this date, so no accounting-quality "
                "observation can be made."]

    latest = ends[0]
    prior = ends[1] if len(ends) > 1 else None
    notes: list[str] = []
    absent: list[str] = []

    for metric, gloss in SINGLE:
        computed = compute_metric(metric, view, latest)
        if computed.is_defined:
            notes.append(f"{metric} = {_fmt(float(computed.value), metric)} -- {gloss}.")
        else:
            absent.append(metric)

    for metric, gloss in TWO_PERIOD:
        computed = (
            compute_two_period_score(metric, view, latest, prior) if prior is not None else None
        )
        if computed is not None and computed.is_defined:
            notes.append(f"{metric} = {float(computed.value):,.3g} -- {gloss}.")
        else:
            absent.append(metric)

    if prior is None:
        notes.append(
            "Only one annual period is visible, so no year-over-year index could be computed."
        )
    if absent:
        notes.append(
            "Not computable from the filed tags at this date: "
            + ", ".join(absent)
            + ". Reported rather than omitted, so absence is not read as a clean result."
        )
    notes.append(DISCLAIMER)
    return notes
