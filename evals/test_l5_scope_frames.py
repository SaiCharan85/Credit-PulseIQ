"""L5: the scope guard must catch advice without blocking description.

A guard that fires on the word "sell" looks strict and is in fact both too
tight and too loose. Too tight, because a company being forced to sell assets
is the vocabulary of distress and a memo has to be able to say it -- the
regression that prompted this file was a plain definition of the current ratio
being withheld for the phrase "forced to sell off long-term equipment". Too
loose, because a recommendation phrased without that verb -- "exit the name",
"reduce the position" -- went straight through.

So both directions are tested here, and the second list is the one that keeps
the first honest.
"""

from __future__ import annotations

import pytest

from agents.schemas import scope_violations

#: Real advice. Every one of these must be blocked.
ADVICE = [
    "You should sell this position",
    "You should buy the stock while it is cheap",
    "Investors ought to avoid this issuer",
    "Clients may want to exit the name before the next filing",
    "We recommend reducing exposure here",
    "I would short the stock",
    "My advice is to wait for the restructuring",
    "We recommend selling into strength",
    "Our recommendation to buy stands",
    "Sell it now",
    "Short the bonds at these levels",
    "The committee should approve the loan",
    "We would deny the loan on these figures",
    "A price target of $12 looks right",
    "The shares should be purchased on weakness",
    "We recommend investing alongside the sponsor",
    "Lenders need to exit this credit",
]

#: Plain description. Every one of these must be allowed -- these are the
#: sentences a credit memo about a failing company is made of.
DESCRIPTION = [
    "Without cash it may be forced to sell off long-term equipment",
    "The company sold its European division to raise liquidity",
    "Management announced a plan to sell non-core assets",
    "Inventory it cannot sell has built up for three quarters",
    "The company continued to buy back shares while coverage fell below one",
    "Buyers of its paper demanded a wider spread",
    "It used the proceeds to buy inventory ahead of the season",
    "Leverage is elevated and coverage is thin",
    "The auditor raised substantial doubt about the going concern assumption",
    "A short-term facility matures within ninety days",
    "Short-dated debt makes up most of the maturity wall",
    "Holders of the notes agreed to extend the maturity",
    "The current ratio measures whether short-term assets cover the next year's bills",
    "Investors were told of the covenant breach in an 8-K",
    "It divested the packaging unit in 2021",
]


@pytest.mark.parametrize("text", ADVICE)
def test_advice_is_blocked(text: str) -> None:
    assert scope_violations(text), f"advice slipped through: {text!r}"


@pytest.mark.parametrize("text", DESCRIPTION)
def test_description_is_allowed(text: str) -> None:
    found = scope_violations(text)
    assert not found, f"description blocked as advice: {text!r} -> {[v.phrase for v in found]}"


def test_the_regression_that_prompted_this() -> None:
    """The exact answer that was withheld: a definition of the current ratio."""
    answer = (
        "The current ratio measures whether a business has enough short-term "
        "assets, like cash and money owed by customers, to cover the bills it "
        "must pay within the next year. Lenders care about this because it "
        "shows if a company can handle its immediate debts without running out "
        "of money or being forced to sell off long-term equipment."
    )
    assert not scope_violations(answer)


def test_advice_buried_in_description_is_still_caught() -> None:
    """The mixed case: a paragraph of legitimate description with a
    recommendation at the end is a violation, not a pass."""
    text = (
        "The company sold its European division to raise liquidity and coverage "
        "remains below one. You should sell this position before the next filing."
    )
    assert scope_violations(text)


def test_violation_reports_the_phrase_that_fired() -> None:
    """A blocked memo has to say what blocked it, or the guard is unauditable."""
    found = scope_violations("Investors ought to avoid this issuer")
    assert found
    assert "avoid" in found[0].phrase.lower()
