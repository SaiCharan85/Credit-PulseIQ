"""Which SEC form is this, and can it even contain the signals we scan for?

Motivated by a real false reassurance. A user dropped in a **Form 4** -- an
insider's statement of changes in beneficial ownership, in this case Berkshire
buying Sirius XM shares -- and the scanner reported:

    Auditor doubts the company can survive the year ......... not found
    Serious weakness in internal financial controls ........ not found

Both true, and both worthless. A Form 4 is two pages of transaction rows. It
has no auditor's opinion in it, no internal-control report, and no management
discussion; those live in a 10-K or a 10-Q. "Not found" in a document that
could never contain the thing reads exactly like "checked and clean", and the
difference between those two is the whole point of the product.

So the scanner identifies the form first and says which of the two readings a
"not found" actually is. The check is deliberately cheap and textual: we are
labelling a document a user handed us, not parsing EDGAR metadata we do not
have.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Forms that carry audited or reviewed financial statements, and therefore can
#: carry an auditor's going-concern paragraph and an internal-control report.
#: A "not found" in one of these is a real negative finding.
FINANCIAL_FORMS = {
    "10-K": "Annual report",
    "10-K/A": "Annual report (amended)",
    "10-Q": "Quarterly report",
    "10-Q/A": "Quarterly report (amended)",
    "20-F": "Annual report, foreign private issuer",
    "40-F": "Annual report, Canadian issuer",
    "S-1": "Registration statement",
    "S-4": "Registration statement, business combination",
    "424B": "Prospectus",
}

#: Forms that cannot contain the disclosures we scan for. A "not found" here
#: says nothing about the company, only about the document.
NON_FINANCIAL_FORMS = {
    "3": "Initial statement of beneficial ownership",
    "4": "Statement of changes in beneficial ownership",
    "5": "Annual statement of changes in beneficial ownership",
    "SC 13D": "Beneficial ownership report, activist",
    "SC 13G": "Beneficial ownership report, passive",
    "13F-HR": "Institutional investment manager holdings",
    "DEF 14A": "Proxy statement",
    "8-K": "Current report",
}

#: What each non-financial form is actually good for, so the refusal points
#: somewhere rather than just declining.
INSTEAD = {
    "3": "It records who owns what, not how the business is doing.",
    "4": (
        "It records one insider's share transactions -- dates, amounts and prices. "
        "Nothing about the company's finances appears in it."
    ),
    "5": "It records an insider's yearly share transactions, not company finances.",
    "SC 13D": "It records a large shareholder's stake and intentions.",
    "SC 13G": "It records a large passive shareholder's stake.",
    "13F-HR": "It lists an investment manager's holdings, not any one company's accounts.",
    "DEF 14A": (
        "It covers voting matters and executive pay. Audited statements are "
        "usually incorporated by reference rather than printed in it."
    ),
    "8-K": (
        "It reports a single event. Going-concern and internal-control language "
        "appear in the annual or quarterly report, though an 8-K may point at one."
    ),
}

#: Ordered most specific first: "10-K/A" must win over "10-K", and the Form 3/4/5
#: patterns must not fire on the words "form 4" inside a 10-K's prose, so they
#: are anchored to the heading style those documents actually use.
_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("10-K/A", re.compile(r"\bFORM\s+10-K\s*/\s*A\b", re.I)),
    ("10-Q/A", re.compile(r"\bFORM\s+10-Q\s*/\s*A\b", re.I)),
    ("10-K", re.compile(r"\bFORM\s+10-K\b|\bANNUAL\s+REPORT\s+PURSUANT\s+TO\s+SECTION\s+13", re.I)),
    ("10-Q", re.compile(r"\bFORM\s+10-Q\b|\bQUARTERLY\s+REPORT\s+PURSUANT\s+TO\s+SECTION\s+13", re.I)),
    ("20-F", re.compile(r"\bFORM\s+20-F\b", re.I)),
    ("40-F", re.compile(r"\bFORM\s+40-F\b", re.I)),
    ("13F-HR", re.compile(r"\bFORM\s+13F\b", re.I)),
    ("SC 13D", re.compile(r"\bSCHEDULE\s+13D\b", re.I)),
    ("SC 13G", re.compile(r"\bSCHEDULE\s+13G\b", re.I)),
    ("DEF 14A", re.compile(r"\bSCHEDULE\s+14A\b|\bPROXY\s+STATEMENT\s+PURSUANT\b", re.I)),
    ("8-K", re.compile(r"\bFORM\s+8-K\b|\bCURRENT\s+REPORT\s+PURSUANT\s+TO\s+SECTION\s+13", re.I)),
    ("4", re.compile(r"\bFORM\s+4\b.{0,400}?STATEMENT\s+OF\s+CHANGES\s+IN\s+BENEFICIAL",
                     re.I | re.S)),
    ("3", re.compile(r"\bFORM\s+3\b.{0,400}?INITIAL\s+STATEMENT\s+OF\s+BENEFICIAL",
                     re.I | re.S)),
    ("5", re.compile(r"\bFORM\s+5\b.{0,400}?ANNUAL\s+STATEMENT\s+OF\s+CHANGES",
                     re.I | re.S)),
    ("S-4", re.compile(r"\bFORM\s+S-4\b", re.I)),
    ("S-1", re.compile(r"\bFORM\s+S-1\b", re.I)),
    ("424B", re.compile(r"\b424B[1-8]\b", re.I)),
)

#: Only the opening of a document is searched for its own form heading. A 10-K
#: mentions "Form 4" in its beneficial-ownership section, and matching that
#: would relabel the annual report as an insider filing.
HEAD_CHARS = 4000


@dataclass(frozen=True)
class FormType:
    code: str = ""
    name: str = ""
    #: Whether this form can carry an auditor's opinion and an internal-control
    #: report. False means a "not found" describes the document, not the company.
    carries_financials: bool = False
    #: Why not, when it cannot.
    instead: str = ""

    @property
    def identified(self) -> bool:
        return bool(self.code)

    @property
    def label(self) -> str:
        return f"{self.code} — {self.name}" if self.identified else "unrecognised form"


def identify(text: str) -> FormType:
    """Label the document from its own heading, or return an empty FormType.

    An unidentified document is *not* treated as unscannable. Plenty of real
    input is an excerpt, a printed page or a filing with the cover stripped,
    and refusing those would be worse than scanning them: the phrase match
    still works on whatever text is present.
    """
    head = text[:HEAD_CHARS]
    for code, pattern in _PATTERNS:
        if pattern.search(head):
            if code in FINANCIAL_FORMS:
                return FormType(code, FINANCIAL_FORMS[code], carries_financials=True)
            return FormType(
                code,
                NON_FINANCIAL_FORMS.get(code, ""),
                carries_financials=False,
                instead=INSTEAD.get(code, ""),
            )
    return FormType()
