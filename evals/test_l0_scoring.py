"""L0: a case with no score must not be ranked as the safest one.

This is the regression test for a published error.

``risk_score`` is None when the agent cannot compute one. The CSV writes it
blank. A reader did ``float(row["risk_score"] or 0)``. Three of those blanks
were actual bankruptcies, so three failures were ranked as maximally safe,
subgroup AUC fell from 0.976 to 0.763, and that was shipped as a reliability
warning shown on every thinly-disclosed memo.

Every step was silent. The tests below fail if any of them can happen again:
a missing score returns None rather than a number, an unscorable positive is
counted rather than absorbed, and the count reaches the caller.
"""

from __future__ import annotations

import pytest

from evals.scoring import NEUTRAL, auc, ranked, risk_of


def test_a_blank_score_is_none_not_zero() -> None:
    """The whole bug in one assertion. 0.0 is the safest possible score, and
    handing it to a bankruptcy is worse than having no number at all."""
    assert risk_of({"risk_score": "", "risk_probability": ""}) is None
    assert risk_of({"risk_score": None}) is None
    assert risk_of({}) is None
    assert risk_of({"risk_score": "   "}) is None


def test_the_calibrated_column_is_preferred() -> None:
    """risk_probability is what the published figure ranks by. Where both
    exist they agree (score/100), so preferring it changes nothing there and
    keeps the fallback where the ordinal is missing."""
    assert risk_of({"risk_probability": "0.97", "risk_score": "97"}) == 0.97
    assert risk_of({"risk_score": "97"}) == 97.0
    assert risk_of({"confidence": "0.8"}) == 0.8


def test_a_malformed_value_falls_through_rather_than_crashing() -> None:
    assert risk_of({"risk_probability": "n/a", "risk_score": "42"}) == 42.0
    assert risk_of({"risk_probability": "n/a"}) is None


def test_unscorable_cases_are_counted_not_imputed() -> None:
    rows = [
        {"risk_probability": "0.9", "label": "1"},
        {"risk_probability": "", "label": "1"},   # a bankruptcy with no score
        {"risk_probability": "0.1", "label": "0"},
    ]
    out = ranked(rows)
    assert out.n == 2, "the unscorable case must not appear in the metric"
    assert out.unscorable == 1
    assert out.unscorable_positive == 1


def test_the_exclusion_reaches_the_caller() -> None:
    """A metric over 23 of 26 cases is publishable. One over 26 where 3 were
    invented is not, and the output has to tell them apart."""
    out = ranked([
        {"risk_probability": "", "label": "1"},
        {"risk_probability": "", "label": "1"},
        {"risk_probability": "0.5", "label": "0"},
    ])
    assert "2 case(s) had no score" in out.note
    assert "2 of them positive" in out.note


def test_a_clean_set_says_nothing() -> None:
    out = ranked([{"risk_probability": "0.9", "label": "1"},
                  {"risk_probability": "0.2", "label": "0"}])
    assert out.note == ""
    assert out.unscorable == 0


def test_the_original_failure_reproduces_and_is_now_prevented() -> None:
    """Thirteen positives, three of them unscorable, as in the real subgroup.

    Defaulting the blanks to 0.0 wrecks the metric; excluding them leaves it
    intact. Both halves are asserted so the fix cannot be undone quietly.
    """
    rows = [{"risk_probability": f"{0.6 + i * 0.02:.2f}", "label": "1"} for i in range(10)]
    rows += [{"risk_probability": "", "label": "1"} for _ in range(3)]
    rows += [{"risk_probability": f"{0.1 + i * 0.02:.2f}", "label": "0"} for i in range(13)]

    honest = ranked(rows)
    assert honest.unscorable_positive == 3
    assert auc(honest.scores, honest.labels) == 1.0

    # ...and what the old reader did with the same data.
    naive = [float(r["risk_probability"] or 0) for r in rows]
    labels = [int(r["label"]) for r in rows]
    assert auc(naive, labels) < 0.8, (
        "three positives defaulted to 0.0 should visibly wreck the metric -- "
        "if this no longer holds, the reproduction has drifted"
    )


def test_neutral_is_available_but_never_automatic() -> None:
    """0.5 is the right value for an abstention *when a caller asks for it*.
    It is not a default, because a default is how the original bug was
    invisible."""
    assert NEUTRAL == 0.5
    assert risk_of({"risk_score": ""}) is not NEUTRAL


@pytest.mark.parametrize(
    ("scores", "labels", "expected"),
    [
        ([1.0, 0.0], [1, 0], 1.0),
        ([0.0, 1.0], [1, 0], 0.0),
        ([0.5, 0.5], [1, 0], 0.5),
        ([1.0], [1], None),
    ],
)
def test_auc_basics(scores, labels, expected) -> None:
    assert auc(scores, labels) == expected
