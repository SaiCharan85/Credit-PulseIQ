"""Regulator and governance signals -- information outside the financials.

Six approaches to the earnings-quality leg all drew from the same well: filed
ratios and eight event flags. They landed between 0.506 and 0.605. This module
supplies information none of them had, which is what separates a seventh
hypothesis from a seventh slice of the same data.

**SEC staff comment letters** (``UPLOAD``, with the filer's reply in
``CORRESP``). The regulator writing to a company to question its accounting,
published to EDGAR after the correspondence closes. It is the closest thing in
public data to a disinterested third party saying "this looks wrong" before
anything is restated.

One property makes them subtle and it is handled explicitly below: a letter is
*written* months before it is *published*. Only the publication date is
knowable at prediction time, so that is the date used. Using the letter date
would be lookahead dressed as diligence.

**Executive departures** (8-K item 5.02). A CFO leaving shortly before a
restatement is among the better-documented precursors in the accounting
literature. The item covers appointments as well as exits, so it is noisy --
carried as a count rather than a flag, and left to the model to weight.

**Prior restatement history.** A filer that has already restated is materially
more likely to restate again. 887 dated events were already on disk and had
never been used as a *feature* -- only as labels.

Every function is as-of filtered on the same rule as the rest of the system:
visible means filed on or before the prediction date.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

#: SEC staff comment letter, and the registrant's reply.
COMMENT_LETTER_FORMS = ("UPLOAD",)
COMMENT_REPLY_FORMS = ("CORRESP",)

#: Departure or appointment of directors and principal officers.
OFFICER_ITEM = "5.02"

#: Default lookback. Two years: comment-letter cycles and executive turnover
#: play out over quarters, not weeks, and a 540-day window truncated real
#: sequences in testing.
LOOKBACK_DAYS = 730


@dataclass(frozen=True)
class OversightSignals:
    """Regulator and governance activity for one filer at one date."""

    comment_letters: int = 0
    comment_replies: int = 0
    days_since_comment_letter: int | None = None
    officer_events: int = 0
    days_since_officer_event: int | None = None
    prior_restatements: int = 0
    days_since_restatement: int | None = None

    def as_features(self) -> dict[str, float]:
        """Flat covariates, each with a missingness flag for the "never happened"
        case -- which is different from "happened a long time ago" and must not
        collapse to the same number."""
        out: dict[str, float] = {
            "ovs_comment_letters": float(self.comment_letters),
            "ovs_comment_replies": float(self.comment_replies),
            "ovs_officer_events": float(self.officer_events),
            "ovs_prior_restatements": float(self.prior_restatements),
        }
        for name, value in (
            ("ovs_days_since_comment_letter", self.days_since_comment_letter),
            ("ovs_days_since_officer_event", self.days_since_officer_event),
            ("ovs_days_since_restatement", self.days_since_restatement),
        ):
            out[name] = float(value) if value is not None else 0.0
            out[f"{name}__missing"] = 0.0 if value is not None else 1.0
        return out


def oversight_feature_names() -> list[str]:
    return sorted(OversightSignals().as_features())


def _visible(index: Sequence[dict[str, Any]], as_of: date, start: date):
    for row in index:
        filed = row.get("filing_date")
        if isinstance(filed, date) and start <= filed <= as_of:
            yield row, filed


def oversight_signals(
    index: Sequence[dict[str, Any]],
    as_of: date,
    restatement_dates: Sequence[date] = (),
    lookback_days: int = LOOKBACK_DAYS,
) -> OversightSignals:
    """Regulator and governance activity visible at ``as_of``.

    ``restatement_dates`` are this filer's own prior 4.02 events. Only those
    strictly before ``as_of`` count -- the event being predicted must never
    appear among its own predictors.
    """
    start = as_of - timedelta(days=lookback_days)
    letters: list[date] = []
    replies: list[date] = []
    officers: list[date] = []

    for row, filed in _visible(index, as_of, start):
        form = (row.get("form") or "").strip()
        if form in COMMENT_LETTER_FORMS:
            letters.append(filed)
        elif form in COMMENT_REPLY_FORMS:
            replies.append(filed)
        elif form.startswith("8-K"):
            items = [i.strip() for i in (row.get("items") or "").split(",")]
            if OFFICER_ITEM in items:
                officers.append(filed)

    prior = sorted(d for d in restatement_dates if d < as_of)

    def gap(dates: list[date]) -> int | None:
        return (as_of - max(dates)).days if dates else None

    return OversightSignals(
        comment_letters=len(letters),
        comment_replies=len(replies),
        days_since_comment_letter=gap(letters),
        officer_events=len(officers),
        days_since_officer_event=gap(officers),
        prior_restatements=len(prior),
        days_since_restatement=(as_of - prior[-1]).days if prior else None,
    )
