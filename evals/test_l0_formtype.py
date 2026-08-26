"""L0: form identification, and the false all-clear it exists to prevent.

A Form 4 scanned for going-concern language reports "not found", which is true
and worthless -- the form has no auditor's opinion in it. The test that matters
is ``test_form_4_cannot_carry_the_signals_we_scan_for``: the scanner must be
able to tell "checked and clean" apart from "could never have said so".
"""

from __future__ import annotations

import pytest

from data import formtype

FORM_4 = """SEC Form 4
FORM 4 UNITED STATES SECURITIES AND EXCHANGE
COMMISSION
Washington, D.C. 20549
STATEMENT OF CHANGES IN BENEFICIAL OWNERSHIP
Filed pursuant to Section 16(a) of the Securities Exchange Act of 1934
1. Name and Address of Reporting Person*
BERKSHIRE HATHAWAY INC
2. Issuer Name and Ticker or Trading Symbol
SIRIUS XM HOLDINGS INC. [ SIRI ]
Common Stock 10/09/2024 P 869,800 A $23.5523(1) 106,024,829 I
"""

TEN_K = """UNITED STATES SECURITIES AND EXCHANGE COMMISSION
FORM 10-K
ANNUAL REPORT PURSUANT TO SECTION 13 OR 15(d) OF THE SECURITIES EXCHANGE ACT OF 1934
For the fiscal year ended December 31, 2023
"""

TEN_Q = """FORM 10-Q
QUARTERLY REPORT PURSUANT TO SECTION 13 OR 15(d) OF THE SECURITIES EXCHANGE ACT OF 1934
"""


def test_form_4_cannot_carry_the_signals_we_scan_for():
    form = formtype.identify(FORM_4)
    assert form.code == "4"
    assert not form.carries_financials, (
        "a Form 4 reported as scannable turns 'this form has no auditor's "
        "opinion' into 'the auditors raised no doubt'"
    )
    assert form.instead, "a refusal should say what the document is instead"


def test_annual_and_quarterly_reports_are_scannable():
    for text, code in ((TEN_K, "10-K"), (TEN_Q, "10-Q")):
        form = formtype.identify(text)
        assert form.code == code
        assert form.carries_financials


def test_amendments_beat_their_base_form():
    assert formtype.identify("FORM 10-K/A\nAmendment No. 1").code == "10-K/A"
    assert formtype.identify("FORM 10-Q/A\nAmendment No. 2").code == "10-Q/A"


def test_a_10k_mentioning_form_4_is_still_a_10k():
    """Annual reports discuss Section 16 filings in prose. Matching that would
    relabel the one document that actually carries the disclosures."""
    text = TEN_K + "\n" * 3 + (
        "Section 16(a) Beneficial Ownership Reporting Compliance. Our directors "
        "file a Form 4 with the Commission upon each transaction, and a "
        "Statement of Changes in Beneficial Ownership accompanies it."
    )
    form = formtype.identify(text)
    assert form.code == "10-K"
    assert form.carries_financials


def test_form_4_heading_beyond_the_cover_is_not_matched():
    """Only the opening is searched, so body prose cannot relabel a document."""
    padding = "Discussion of results of operations. " * 200
    assert len(padding) > formtype.HEAD_CHARS
    text = padding + FORM_4
    assert not formtype.identify(text).identified


@pytest.mark.parametrize(
    ("text", "code"),
    [
        ("SCHEDULE 13D under the Securities Exchange Act", "SC 13D"),
        ("SCHEDULE 13G under the Securities Exchange Act", "SC 13G"),
        ("FORM 8-K CURRENT REPORT PURSUANT TO SECTION 13", "8-K"),
        ("PROXY STATEMENT PURSUANT TO SECTION 14(a)", "DEF 14A"),
        ("FORM 20-F ANNUAL REPORT", "20-F"),
        ("FORM 13F-HR holdings report", "13F-HR"),
    ],
)
def test_recognises_the_common_forms(text, code):
    assert formtype.identify(text).code == code


def test_unrecognised_text_is_still_scanned():
    """An excerpt or a cover-stripped filing must not be refused: the phrase
    match still works on whatever text is present, and refusing would lose a
    real going-concern paragraph over a missing heading."""
    form = formtype.identify(
        "substantial doubt about the Company's ability to continue as a going concern"
    )
    assert not form.identified
    assert not form.carries_financials  # unknown, so not asserted as scannable
    assert form.label == "unrecognised form"


def test_labels_are_human_readable():
    assert formtype.identify(FORM_4).label.startswith("4 — ")
    assert "Annual report" in formtype.identify(TEN_K).label
