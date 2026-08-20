"""Restatement discovery: 8-K item 4.02, "non-reliance on previously issued
financial statements" (SPEC 12, phase 4 -- earnings-quality labels).

Produces *candidates* for review, never labels directly. Nothing here writes to
``data/labels/``; a human promotes rows, exactly as with Chapter 11.

Why 4.02 rather than AAER. SPEC names AAER, and AAER is the stronger claim --
SEC-confirmed accounting fraud. But an AAER is an enforcement release on
sec.gov, not a filing: its respondent is frequently an individual ("Steven W.
Hurd, CPA") with no CIK, the issuer must be recovered by fuzzy name matching,
and the misstatement window exists only as prose inside a PDF. Item 4.02 is
filed *by the company*, carries a structured item code in submissions metadata,
and is dated by EDGAR. The trade is scope, and it is stated rather than hidden:
this leg detects *accounting problems*, which includes honest error, not fraud.

**The label means: the financial statements a reader could see at time T were
later declared unreliable by the company itself.** That is the earnings-quality
question, and it is answerable without hindsight leaking in -- the 4.02 is the
moment it becomes public, and every prediction is made strictly before it.

Three checks, because the item code alone is not trustworthy:

1. **Item code 4.02** from submissions metadata. Filer-supplied, so it is both
   over- and under-applied.
2. **Filing text must describe this filer's own non-reliance** -- not a
   subsidiary's, not a boilerplate item header with no content.
3. **Regulatory-reclassification restatements are excluded** -- and there were
   *two* waves, which is why this check is two rules rather than one.

   In April 2021 the SEC ruled SPAC warrants must be classified as liabilities
   and essentially every SPAC restated at once. Market-wide that is 598 of 2,008
   events, 30%.

   The second wave is the trap. In late 2021 SPACs were pressed to reclassify
   Class A shares subject to possible redemption into temporary equity. It uses
   no warrant language at all, so a warrant filter passes it through untouched:
   after the first rule, 2021 still held 299 events against a 2019-2020 baseline
   near 65, and 6 of 14 sampled survivors from 2021Q4-2022Q1 were this
   reclassification, against 0 of 14 in a 2019-2020 control.

   Both are market-wide accounting rule changes, not company-specific failures.
   Keeping either would teach a model to recognise "was a SPAC in 2021" and
   report it as earnings quality.

Deterministic plain code -- no LLM (PROMPT hard rule 3).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

#: The 8-K item that announces a restatement.
NON_RELIANCE_ITEM = "4.02"

#: Reasons a candidate is rejected. Recorded rather than silently dropped, so
#: the exclusion rate is auditable and a bad rule shows up as a strange count.
KEEP = "keep"
REJECT_NO_TEXT = "no_filing_text"
REJECT_NOT_OWN = "not_this_filer"
REJECT_SPAC_WARRANT = "spac_warrant_reclassification"
REJECT_SPAC_SHARES = "spac_share_reclassification"

_NON_RELIANCE = re.compile(
    r"(should\s+no\s+longer\s+be\s+relied\s+upon"
    r"|non-?reliance"
    r"|shall\s+not\s+be\s+relied\s+upon"
    r"|will\s+(?:be\s+)?restat|has\s+restated|intends\s+to\s+restate"
    r"|previously\s+issued\s+financial\s+statements)",
    re.I,
)

_WARRANT = re.compile(r"warrant", re.I)

#: The *second* SPAC wave, and the one a warrant filter sails straight past.
#: In late 2021 the SEC pressed SPACs to reclassify Class A shares subject to
#: possible redemption out of permanent equity and into temporary equity.
#: Measured: 6 of 14 sampled 2021Q4-2022Q1 events that survived the warrant
#: check are this, against 0 of 14 in a 2019-2020 control.
_REDEMPTION = re.compile(
    r"subject\s+to\s+possible\s+redemption"
    r"|temporary\s+equity"
    r"|shares?\s+subject\s+to\s+redemption",
    re.I,
)
_SPAC = re.compile(
    r"blank\s+check|special\s+purpose\s+acquisition|SPAC\b|initial\s+business\s+combination",
    re.I,
)
#: A trust account holding IPO proceeds is a SPAC structure and little else.
_TRUST = re.compile(r"trust\s+account", re.I)

#: The SEC statement that triggered the wave.
_WARRANT_GUIDANCE = re.compile(
    r"(?:Staff\s+Statement|SEC\s+Statement)[^.]{0,120}?Warrant"
    r"|Warrants?\s+Issued\s+by\s+Special\s+Purpose\s+Acquisition"
    r"|equity\s+to\s+(?:a\s+)?liabilit|liabilit[^.]{0,40}rather\s+than\s+equity"
    r"|reclassif[^.]{0,60}warrant",
    re.I,
)


@dataclass(frozen=True)
class RestatementCandidate:
    """One 8-K item 4.02 event, classified but not yet promoted to a label."""

    cik: int
    filing_date: date
    accession: str
    primary_document: str
    verdict: str = KEEP
    detail: str = ""

    @property
    def kept(self) -> bool:
        return self.verdict == KEEP

    def as_dict(self) -> dict[str, Any]:
        return {
            "cik": self.cik,
            "filing_date": self.filing_date.isoformat(),
            "accession": self.accession,
            "primary_document": self.primary_document,
            "verdict": self.verdict,
            "detail": self.detail,
        }


def find_non_reliance_filings(
    index: Sequence[dict[str, Any]], since: date | None = None
) -> list[dict[str, Any]]:
    """Every 8-K carrying item 4.02, optionally from ``since`` onward."""
    out = []
    for row in index:
        form = (row.get("form") or "").strip()
        filed = row.get("filing_date")
        if not form.startswith("8-K") or not isinstance(filed, date):
            continue
        if since is not None and filed < since:
            continue
        items = [i.strip() for i in (row.get("items") or "").split(",")]
        if NON_RELIANCE_ITEM in items:
            out.append(row)
    return out


def is_spac_warrant_restatement(text: str) -> bool:
    """Whether this restatement is the 2021 SPAC warrant reclassification.

    Requires warrant language *and* either SPAC framing or the equity-to-
    liability reclassification wording. Warrants alone are not enough -- an
    operating company can restate for warrant accounting on its own merits, and
    that is a genuine accounting failure.
    """
    if not _WARRANT.search(text):
        return False
    return bool(_SPAC.search(text) or _WARRANT_GUIDANCE.search(text))


def is_spac_share_reclassification(text: str) -> bool:
    """Whether this is the late-2021 Class A share redemption reclassification.

    Needs redemption-accounting language *and* SPAC framing. An operating
    company with redeemable preferred stock also uses "temporary equity", and
    restating that is a genuine accounting failure.
    """
    if not _REDEMPTION.search(text):
        return False
    return bool(_SPAC.search(text) or _TRUST.search(text))


def classify(text: str) -> tuple[str, str]:
    """Judge one filing's text. Returns ``(verdict, supporting detail)``."""
    if not text or not text.strip():
        return REJECT_NO_TEXT, ""
    found = _NON_RELIANCE.search(text)
    if not found:
        # The item header can appear with no substantive disclosure attached.
        return REJECT_NOT_OWN, ""
    if is_spac_warrant_restatement(text):
        return REJECT_SPAC_WARRANT, "warrant reclassification per the 2021 SEC statement"
    if is_spac_share_reclassification(text):
        return REJECT_SPAC_SHARES, "Class A shares reclassified to temporary equity"
    lo = max(0, found.start() - 100)
    return KEEP, text[lo : found.end() + 200].strip()[:300]


def discover(
    ciks: Iterable[int],
    filing_index_for,
    fetch_text_for,
    since: date | None = None,
    on_progress=None,
) -> list[RestatementCandidate]:
    """Scan a universe for item 4.02 events and classify each one.

    ``filing_index_for(cik)`` and ``fetch_text_for(cik, accession, document)``
    are injected so this stays testable without touching the network.
    """
    out: list[RestatementCandidate] = []
    for n, cik in enumerate(ciks, 1):
        try:
            index = filing_index_for(cik)
        except Exception:  # noqa: BLE001
            index = []
        for row in find_non_reliance_filings(index, since):
            try:
                text = fetch_text_for(cik, row["accession"], row.get("primary_document", ""))
            except Exception:  # noqa: BLE001
                text = ""
            verdict, detail = classify(text)
            out.append(
                RestatementCandidate(
                    cik=cik,
                    filing_date=row["filing_date"],
                    accession=row["accession"],
                    primary_document=row.get("primary_document", ""),
                    verdict=verdict,
                    detail=detail,
                )
            )
        if on_progress:
            on_progress(n, cik)
    out.sort(key=lambda c: (c.filing_date, c.cik))
    return out


def first_event_per_company(
    candidates: Sequence[RestatementCandidate],
) -> list[RestatementCandidate]:
    """Earliest kept event per filer.

    A company that restates often files several 8-Ks about one episode. Counting
    each as an independent event would inflate the positive class and let one
    company dominate the metrics.
    """
    seen: dict[int, RestatementCandidate] = {}
    for c in candidates:
        if not c.kept:
            continue
        if c.cik not in seen or c.filing_date < seen[c.cik].filing_date:
            seen[c.cik] = c
    return sorted(seen.values(), key=lambda c: (c.filing_date, c.cik))
