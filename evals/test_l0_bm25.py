"""L0: rank the candidates, and know what a candidate is.

Company resolution was a positional rule -- take capitalised words, resolve the
first that matches. It failed both ways at once: "Can yu tll me" matched Canaan
Inc. on three letters of a typo, and "comapre this to Valaris" matched CNA
Financial and never reached Valaris.

Both are the same missing thing: a **score**. A rule returns a match or nothing,
so nothing can be said about one candidate fitting better than another.

The instructive part is that BM25 alone made it *worse*. Fed whole questions it
found Here Group Ltd for "here", Wheels Up for "up" and SOUTHERN CO for "so" --
and it was not wrong, those genuinely are the best matches for those tokens. In
a corpus of ten thousand names every English word is somebody's company. The
failure was never ranking; it was deciding what to rank.

So extraction stays a rule and BM25 scores what it produces.
"""

from __future__ import annotations

import pytest

from data.bm25 import Index

ROWS = [
    {"cik": 28823, "name": "DIEBOLD NIXDORF, Inc", "ticker": "DBD"},
    {"cik": 314808, "name": "Valaris Ltd", "ticker": "VAL"},
    {"cik": 21175, "name": "CNA FINANCIAL CORP", "ticker": "CNA"},
    {"cik": 320193, "name": "Apple Inc.", "ticker": "AAPL"},
    {"cik": 1486159, "name": "Chord Energy Corp", "ticker": "CHRD"},
    {"cik": 1008654, "name": "TUPPERWARE BRANDS CORP", "ticker": "TUP"},
    {"cik": 9999, "name": "Here Group Ltd", "ticker": ""},
    {"cik": 8888, "name": "Wheels Up Experience Inc.", "ticker": "UP"},
    {"cik": 7777, "name": "SOUTHERN CO", "ticker": "SO"},
]


@pytest.fixture(scope="module")
def ix() -> Index:
    return Index().add_all(ROWS)


@pytest.mark.parametrize(
    ("phrase", "expected"),
    [
        ("Valaris", "Valaris Ltd"),
        ("Apple", "Apple Inc."),
        ("Chord Energy", "Chord Energy Corp"),
        ("Tupperware", "TUPPERWARE BRANDS CORP"),
        ("Diebold Nixdorf", "DIEBOLD NIXDORF, Inc"),
    ],
)
def test_a_named_company_resolves(ix: Index, phrase: str, expected: str) -> None:
    hit = ix.resolve(phrase)
    assert hit is not None and hit.name == expected


@pytest.mark.parametrize("phrase", ["here", "up", "so", "the", "it", "compare", "ok"])
def test_ordinary_words_resolve_to_nothing(ix: Index, phrase: str) -> None:
    """The failure BM25 introduced when it was handed whole sentences. Each of
    these is genuinely the best match for its token, and none of them is a
    company the reader named."""
    assert ix.resolve(phrase) is None


def test_the_best_candidate_wins_regardless_of_position(ix: Index) -> None:
    """"comapre this to Valaris" -- the old scan returned CNA because the typo
    came first. Scoring every candidate and taking the maximum is the fix."""
    candidates = ["Cna", "Valaris"]
    hit = ix.best_in("Cna you also comapre this to Valaris", candidates)
    assert hit is not None and hit.name == "Valaris Ltd"


def test_a_ticker_resolves_like_a_name(ix: Index) -> None:
    """Readers use both, and a ticker is the rarest possible term for a filer."""
    hit = ix.resolve("DBD")
    assert hit is not None and hit.name == "DIEBOLD NIXDORF, Inc"


def test_corporate_furniture_alone_matches_nothing(ix: Index) -> None:
    """"Corp" is in thousands of names. Matching it is not knowing which."""
    for phrase in ("Corp", "Inc", "Ltd", "Holdings Group"):
        assert ix.resolve(phrase) is None


def test_the_loaded_filer_is_excluded(ix: Index) -> None:
    assert ix.resolve("Diebold", exclude_cik=28823) is None
    assert ix.resolve("Valaris", exclude_cik=28823) is not None


def test_a_weak_match_is_refused_rather_than_guessed(ix: Index) -> None:
    """Below the evidence floor the match rests on common terms, and a wrong
    company named confidently is worse than admitting ambiguity."""
    hit = ix.resolve("xyzzy nonexistent")
    assert hit is None


def test_scores_are_ordered_and_reported(ix: Index) -> None:
    """The score is the point -- it is what a rule could never give."""
    hits = ix.search("Valaris")
    assert hits and hits[0].name == "Valaris Ltd"
    assert hits[0].score > 0
    assert all(hits[i].score >= hits[i + 1].score for i in range(len(hits) - 1))
