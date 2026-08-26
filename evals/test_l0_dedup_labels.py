"""L0: collapse one bankruptcy recorded twice, and nothing else.

A reorganisation can leave two registrants reporting the same petition -- QVC
Group, Inc. and Old QVC Group, Inc. both filed an item 1.03 for the Chapter 11
of 2026-04-16. That is one event and two positives.

The dangerous version of this fix treats every shared filing date as a
duplicate. The data refuses it: of twelve dates carrying more than one label,
exactly one is a real double-count. Tupperware and BurgerFi both filed on
2024-09-17; Chord Energy and Lonestar Resources both filed on 2020-09-30.
Collapsing those would delete real bankruptcies and shrink the positive class
-- corrupting the measurement in the name of cleaning it.

So both directions are tested, and the second list is the one that matters.
"""

from __future__ import annotations

import pytest

from data.dedup_labels import find_duplicates, normalise, same_business

#: Real pairs from the events file. One bankruptcy, two registrants.
DUPLICATES = [
    ("QVC Group, Inc.", "Old QVC Group, Inc."),
    ("Hertz Global Holdings, Inc.", "New Hertz Global Holdings Inc"),
    ("Party City Holdco Inc.", "Party City Holdco"),
]

#: Also real, and emphatically not duplicates -- companies that happened to
#: file on the same day.
COINCIDENCES = [
    ("TUPPERWARE BRANDS CORP", "BurgerFi International, Inc."),
    ("Chord Energy Corp", "Lonestar Resources US Inc."),
    ("Spirit Aviation Holdings, Inc.", "CareMax, Inc."),
    ("Acorda Therapeutics, Inc.", "Eiger BioPharmaceuticals, Inc."),
    ("CalAmp Corp.", "ISUN, INC."),
    ("2U, LLC", "Vintage Wine Estates, Inc."),
    ("NS Wind Down Co., Inc.", "Cano Health, Inc."),
]


@pytest.mark.parametrize(("a", "b"), DUPLICATES)
def test_one_business_under_two_names_is_matched(a: str, b: str) -> None:
    assert same_business(a, b), f"missed a real duplicate: {a!r} / {b!r}"


@pytest.mark.parametrize(("a", "b"), COINCIDENCES)
def test_different_companies_are_never_merged(a: str, b: str) -> None:
    """The expensive mistake. Merging these deletes real bankruptcies."""
    assert not same_business(a, b), f"would have deleted a real label: {a!r} / {b!r}"


def test_normalisation_strips_only_furniture() -> None:
    assert normalise("QVC Group, Inc.") == "qvc"
    assert normalise("Old QVC Group, Inc.") == "qvc"
    assert normalise("TUPPERWARE BRANDS CORP") == "tupperware brands"


def test_short_names_do_not_collapse() -> None:
    """A two-letter remainder matches half the universe by containment."""
    assert not same_business("BP p.l.c.", "BJ Services Company")
    assert not same_business("Inc.", "Incorporated")


def test_the_survivor_is_the_entity_that_carries_on() -> None:
    rows = [
        {"cik": "1254699", "signal": "chapter11_petition", "event_date": "2026-04-16"},
        {"cik": "1355096", "signal": "chapter11_petition", "event_date": "2026-04-16"},
    ]
    names = {1254699: "QVC Group, Inc.", 1355096: "Old QVC Group, Inc."}
    found = find_duplicates(rows, names)
    assert len(found) == 1
    assert found[0].keep_cik == 1254699, "the shell should be dropped, not the survivor"
    assert found[0].drop_cik == 1355096


def test_a_shared_date_alone_is_not_enough() -> None:
    rows = [
        {"cik": "1008654", "signal": "chapter11_petition", "event_date": "2024-09-17"},
        {"cik": "1723580", "signal": "chapter11_petition", "event_date": "2024-09-17"},
    ]
    names = {1008654: "TUPPERWARE BRANDS CORP", 1723580: "BurgerFi International, Inc."}
    assert find_duplicates(rows, names) == []


def test_the_same_name_on_different_dates_is_not_a_duplicate() -> None:
    """A company can file twice. Chapter 22 is a real thing and both events
    happened."""
    rows = [
        {"cik": "1", "signal": "chapter11_petition", "event_date": "2019-03-01"},
        {"cik": "2", "signal": "chapter11_petition", "event_date": "2024-03-01"},
    ]
    names = {1: "Acme Corp", 2: "Old Acme Corp"}
    assert find_duplicates(rows, names) == []


def test_an_unnamed_registrant_is_left_alone() -> None:
    """Most defunct filers are absent from the SEC's current directory. Missing
    a name is a coverage gap, never a licence to merge on date alone."""
    rows = [
        {"cik": "1", "signal": "chapter11_petition", "event_date": "2024-01-01"},
        {"cik": "2", "signal": "chapter11_petition", "event_date": "2024-01-01"},
    ]
    assert find_duplicates(rows, {1: "Acme Corp"}) == []
    assert find_duplicates(rows, {}) == []


def test_other_signals_are_untouched() -> None:
    """Only petitions collapse. A delisting recorded against the shell is a
    thing that genuinely happened to that filer."""
    rows = [
        {"cik": "1254699", "signal": "chapter11_petition", "event_date": "2026-04-16"},
        {"cik": "1355096", "signal": "delisting_filed", "event_date": "2026-04-16"},
    ]
    names = {1254699: "QVC Group, Inc.", 1355096: "Old QVC Group, Inc."}
    assert find_duplicates(rows, names) == []
