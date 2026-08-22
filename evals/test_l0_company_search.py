"""L0: resolving a company name or ticker to a CIK.

The ambiguity tests are the point. "apple" matches both Apple Inc. and Apple
Hospitality REIT, and silently returning the larger one would be a guess
dressed as an answer -- precisely the silent-wrong-answer failure this project
exists to prevent, on the one step where the user cannot check the result
because they came here not knowing the identifier.
"""

from __future__ import annotations

import pytest

from data.company_search import (
    EXACT_NAME,
    EXACT_TICKER,
    NAME_CONTAINS,
    TICKER_PREFIX,
    resolve,
    search,
)

DIRECTORY = [
    {"cik": 320193, "name": "Apple Inc.", "ticker": "AAPL"},
    {"cik": 1418121, "name": "Apple Hospitality REIT, Inc.", "ticker": "APLE"},
    {"cik": 789019, "name": "MICROSOFT CORP", "ticker": "MSFT"},
    {"cik": 1318605, "name": "Tesla, Inc.", "ticker": "TSLA"},
    {"cik": 28823, "name": "DIEBOLD NIXDORF, Inc", "ticker": "DBD"},
    {"cik": 1652044, "name": "Alphabet Inc.", "ticker": "GOOGL"},
]


class TestTickerAndName:
    def test_an_exact_ticker_resolves(self) -> None:
        m, _ = resolve("AAPL", DIRECTORY)
        assert m and m.cik == 320193 and m.how == EXACT_TICKER

    def test_tickers_are_case_insensitive(self) -> None:
        m, _ = resolve("tsla", DIRECTORY)
        assert m and m.cik == 1318605

    def test_an_exact_name_resolves(self) -> None:
        m, _ = resolve("Microsoft Corp", DIRECTORY)
        assert m and m.cik == 789019 and m.how == EXACT_NAME

    def test_legal_suffixes_are_ignored(self) -> None:
        """'Tesla' and 'Tesla, Inc.' are the same company to a human."""
        m, _ = resolve("Tesla", DIRECTORY)
        assert m and m.cik == 1318605

    def test_lowercase_name_still_matches(self) -> None:
        m, _ = resolve("microsoft", DIRECTORY)
        assert m and m.cik == 789019


class TestAmbiguityIsNotResolvedSilently:
    def test_a_shared_prefix_with_no_exact_hit_stays_ambiguous(self) -> None:
        """Neither 'Apple Hospitality' nor 'Alphabet' is an exact match for
        'al', so nothing may be auto-selected."""
        m, candidates = resolve("app", DIRECTORY)
        assert m is None, "must not guess when no candidate matches exactly"
        assert {c.cik for c in candidates} == {320193, 1418121}

    def test_a_normalised_exact_name_does_resolve(self) -> None:
        """'apple' equals 'Apple Inc.' once the legal suffix is stripped, and
        that is what a person typing it means -- so it resolves, and the
        REIT is still offered as an alternative."""
        m, candidates = resolve("apple", DIRECTORY)
        assert m and m.cik == 320193
        assert 1418121 in {c.cik for c in candidates}

    def test_candidates_are_returned_for_the_user_to_choose(self) -> None:
        _, candidates = resolve("apple", DIRECTORY)
        assert all(c.reason for c in candidates)

    def test_an_exact_hit_wins_even_when_others_match(self) -> None:
        """'Apple Inc.' is unambiguous despite 'apple' being shared."""
        m, _ = resolve("Apple Inc.", DIRECTORY)
        assert m and m.cik == 320193


class TestNoMatch:
    def test_an_unknown_name_returns_nothing(self) -> None:
        m, candidates = resolve("Wayne Enterprises", DIRECTORY)
        assert m is None and candidates == []

    def test_an_empty_query_returns_nothing(self) -> None:
        assert search("", DIRECTORY) == []

    def test_a_single_character_matches_nothing(self) -> None:
        """A one-letter query prefix-matches a large share of any real
        directory, which is noise rather than a result."""
        assert search("a", DIRECTORY) == []

    def test_two_characters_match_tickers_but_not_names(self) -> None:
        """Tickers are short enough that a two-character prefix is a real
        query; company names are not, so 'ap' finds APLE and not every
        company beginning with those letters."""
        hits = search("ap", DIRECTORY)
        assert all(h.how == TICKER_PREFIX for h in hits)
        assert 320193 not in {h.cik for h in hits}  # Apple Inc., ticker AAPL


class TestRanking:
    def test_exact_beats_substring(self) -> None:
        hits = search("Apple Inc.", DIRECTORY)
        assert hits[0].how <= NAME_CONTAINS
        assert hits[0].cik == 320193

    @pytest.mark.parametrize("q,cik", [("GOOGL", 1652044), ("DBD", 28823)])
    def test_known_tickers_resolve_to_the_right_filer(self, q: str, cik: int) -> None:
        m, _ = resolve(q, DIRECTORY)
        assert m and m.cik == cik
