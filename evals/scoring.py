"""One way to read a case's risk score, and no way to read it silently wrong.

A fairness gap was reported from this repo that did not exist. The chain:

1. ``risk_score`` is ``None`` when the agent cannot compute one -- an
   abstention, or too little data to place the filer in a band.
2. The backtest CSV writes that as an empty cell.
3. A reader did ``float(row["risk_score"] or 0)``.
4. Three of those blanks were **actual bankruptcies**.
5. All three landed at 0.0 -- the safest possible score -- at the bottom of the
   risk ranking.
6. Subgroup AUC fell from 0.976 to 0.763, and that was published as a
   reliability warning shown to readers on every thinly-disclosed memo.

No step raised anything. The defect is not that two score columns exist; it is
that **a missing score became maximum confidence in safety**, which is the most
dangerous direction for it to fail in and the least visible.

So :func:`risk_of` returns ``None`` rather than a number it does not have, and
:func:`ranked` reports how many cases it could not score. A metric computed
over 23 of 26 cases is a fine thing to publish. A metric computed over 26 where
3 were quietly invented is not, and nothing in the output distinguished them.

The column order is not arbitrary either. ``risk_probability`` is what the
published 0.963 ranks by; ``risk_score`` is the ordinal self-report and equals
it exactly (score/100) wherever both exist. Preferring the ordinal changes
nothing where it is present and loses the calibrated fallback where it is not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: In preference order. The first is what the headline figure ranks by; the
#: others only ever apply where it is absent.
SCORE_COLUMNS = ("risk_probability", "risk_score", "confidence")

#: What an abstention is worth for ranking: nothing either way. Used only where
#: a caller explicitly asks for it, never as a silent default.
NEUTRAL = 0.5


def risk_of(row: dict[str, Any]) -> float | None:
    """This case's risk score, or None when it genuinely has none.

    Returning None is the point. Every caller then has to decide what an
    unscorable case means for its own metric, and that decision becomes visible
    in the code rather than hiding in a default argument.
    """
    for key in SCORE_COLUMNS:
        raw = row.get(key)
        if raw is None:
            continue
        text = str(raw).strip()
        if not text:
            continue
        try:
            return float(text)
        except ValueError:
            continue
    return None


@dataclass
class Ranked:
    """Scores paired with labels, and an explicit count of what was dropped."""

    scores: list[float] = field(default_factory=list)
    labels: list[int] = field(default_factory=list)
    #: Cases with no usable score. Reported, never imputed.
    unscorable: int = 0
    #: How many of those were positives. A blank that is disproportionately a
    #: bankruptcy is the exact shape of the bug this module exists to prevent,
    #: so it is surfaced rather than left for someone to notice.
    unscorable_positive: int = 0

    @property
    def n(self) -> int:
        return len(self.scores)

    @property
    def note(self) -> str:
        if not self.unscorable:
            return ""
        return (
            f"{self.unscorable} case(s) had no score and were excluded"
            f"{f', {self.unscorable_positive} of them positive' if self.unscorable_positive else ''}"
        )


def ranked(rows: list[dict[str, Any]], label_key: str = "label") -> Ranked:
    """Scorable cases only, with the rest counted rather than defaulted."""
    out = Ranked()
    for row in rows:
        try:
            label = int(row[label_key])
        except (KeyError, TypeError, ValueError):
            continue
        score = risk_of(row)
        if score is None:
            out.unscorable += 1
            out.unscorable_positive += label
            continue
        out.scores.append(score)
        out.labels.append(label)
    return out


def auc(scores: list[float], labels: list[int]) -> float | None:
    """Rank-order AUC, ties counted as half. None when a class is absent."""
    pos = [s for s, y in zip(scores, labels, strict=True) if y == 1]
    neg = [s for s, y in zip(scores, labels, strict=True) if y == 0]
    if not pos or not neg:
        return None
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))
