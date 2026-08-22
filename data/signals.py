"""Distress signals that live in filing text and metadata, not in ratios.

The hazard baseline is fitted on fourteen financial ratios, and the agent's
tools exposed the same registry -- so the agent was reading exactly the inputs
a logistic regression had already been fitted on, and re-deriving them by hand.
Measured: rank correlation 0.740 with the baseline, and 27 of 81 bankruptcies
missed that the baseline caught. Competing on those features is a losing game,
because the regression fits them directly against the outcome.

What a regression cannot read is what a filing *says*, and when it arrives.
These signals are absent from the panel by construction:

* **Late filings** (``NT 10-K``, ``NT 10-Q``). A company that cannot close its
  books on time is in operational trouble before the ratios show it.
* **Auditor dismissal or resignation** (8-K item 4.01).
* **Non-reliance on previously issued financials** (8-K item 4.02) -- a
  restatement, which means the ratios themselves were wrong.
* **Covenant breach or debt acceleration** (8-K item 2.04).
* **Delisting notice** (8-K item 3.01).
* **Material impairment** (8-K item 2.06).
* **Going-concern doubt** and **material weakness** in the audit opinion.

Two disciplines hold throughout:

**As-of.** Every function takes ``as_of`` and drops anything filed after it.
The date is supplied by the caller that constructed the toolbox, never chosen
by the model.

**Item 1.03 (bankruptcy) is deliberately excluded.** It is the outcome. An
as-of filter makes a prior 1.03 legitimate evidence, and
``get_prior_distress_events`` already reports it; repeating it here would
double-count the one signal closest to the label, which is exactly the shape a
leak takes even when the dates are honest.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

#: 8-K item code -> what it means for credit. Item 1.03 is excluded on purpose.
EVENT_ITEMS: dict[str, str] = {
    "2.04": "covenant breach or acceleration of a direct financial obligation",
    "3.01": "notice of delisting or failure to satisfy a listing rule",
    "4.01": "changes in registrant's certifying accountant",
    "4.02": "non-reliance on previously issued financial statements",
    "2.06": "material impairment",
}

#: Forms that announce a filing will be late.
LATE_FORMS = {"NT 10-K", "NT 10-Q", "NT 20-F", "NT 10-K/A", "NT 10-Q/A"}

#: Periodic reports whose text carries the audit opinion.
REPORT_FORMS = ("10-K", "10-Q", "20-F")

GOING_CONCERN_PATTERNS = (
    r"substantial\s+doubt\s+(?:exists\s+)?(?:about|as\s+to|regarding)[^.]{0,120}?going\s+concern",
    r"ability\s+to\s+continue\s+as\s+a\s+going\s+concern",
    r"going\s+concern\s+(?:qualification|uncertaint)",
)

MATERIAL_WEAKNESS_PATTERNS = (
    r"material\s+weakness(?:es)?\s+in\s+(?:our|its|the\s+Company'?s?)?\s*internal\s+control",
    r"internal\s+control\s+over\s+financial\s+reporting\s+(?:was|were)\s+not\s+effective",
)

_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


def strip_html(raw: str) -> str:
    """Tags out, whitespace normalised.

    Deliberately crude. The patterns below match prose that survives any
    reasonable stripping, and a real HTML parser would add a dependency for no
    gain in recall.
    """
    text = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", raw)
    text = _TAG.sub(" ", text)
    text = (
        text.replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&#8217;", "'")
        .replace("&rsquo;", "'")
    )
    return _WS.sub(" ", text)


def _matches(text: str, patterns: Sequence[str]) -> str:
    """First matching phrase, with a little surrounding context, or ""."""
    for pattern in patterns:
        found = re.search(pattern, text, re.IGNORECASE)
        if found:
            lo = max(0, found.start() - 120)
            hi = min(len(text), found.end() + 120)
            return text[lo:hi].strip()
    return ""


@dataclass(frozen=True)
class FilingEvent:
    """One dated, categorised filing event."""

    filing_date: date
    form: str
    code: str
    description: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "filing_date": self.filing_date.isoformat(),
            "form": self.form,
            "code": self.code,
            "description": self.description,
        }


def filing_events(
    index: Sequence[dict[str, Any]],
    as_of: date,
    lookback_days: int = 540,
) -> list[FilingEvent]:
    """Credit-relevant filing events in the window ending at ``as_of``.

    Anything filed after ``as_of`` is dropped, so this cannot see the future
    even when the index does.
    """
    start = as_of - timedelta(days=lookback_days)
    out: list[FilingEvent] = []
    for row in index:
        filed = row.get("filing_date")
        if not isinstance(filed, date) or filed > as_of or filed < start:
            continue
        form = (row.get("form") or "").strip()
        if form in LATE_FORMS:
            out.append(
                FilingEvent(filed, form, "late_filing", "notification of late filing")
            )
            continue
        if not form.startswith("8-K"):
            continue
        # `items` is a comma-separated list like "2.04,9.01".
        for raw in (row.get("items") or "").split(","):
            code = raw.strip()
            if code in EVENT_ITEMS:
                out.append(FilingEvent(filed, form, code, EVENT_ITEMS[code]))
    out.sort(key=lambda e: (e.filing_date, e.code))
    return out


def latest_report(
    index: Sequence[dict[str, Any]], as_of: date
) -> dict[str, Any] | None:
    """Most recent periodic report filed on or before ``as_of``."""
    candidates = [
        row
        for row in index
        if isinstance(row.get("filing_date"), date)
        and row["filing_date"] <= as_of
        and (row.get("form") or "").strip().startswith(REPORT_FORMS)
        and row.get("primary_document")
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda r: r["filing_date"])


def scan_report_text(text: str) -> dict[str, Any]:
    """Look for going-concern doubt and material weakness in one report.

    Quotes are sanitised before being returned, because they are lifted from a
    filing written by the company under assessment and handed to the model as
    context. Instruction-like spans are neutralised and the attempt reported --
    see ``data/sanitize.py`` for why the output guards cannot cover this.
    """
    from data.sanitize import sanitize

    plain = strip_html(text)
    concern = sanitize(_matches(plain, GOING_CONCERN_PATTERNS)[:400])
    weakness = sanitize(_matches(plain, MATERIAL_WEAKNESS_PATTERNS)[:400])
    suspicious = concern.findings + weakness.findings
    out: dict[str, Any] = {
        "going_concern_doubt": bool(concern.text),
        "going_concern_quote": concern.text,
        "material_weakness": bool(weakness.text),
        "material_weakness_quote": weakness.text,
    }
    if suspicious:
        out["injection_attempt"] = True
        out["injection_note"] = (concern.note or weakness.note)
    return out
