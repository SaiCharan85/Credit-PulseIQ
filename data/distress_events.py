"""Graded distress events — the severity ladder (SPEC 7, positives enrichment).

Chapter 11 is the cleanest label and the rarest. Waiting for it alone makes the
positive class tiny and the signal late. This module extracts the *intermediate*
distress events that precede it, each dated and traced to a filing, so the
backtest can measure early warning at several thresholds instead of one.

Four tiers, scored separately and never pooled:

===============  ====================================================
``default``      Chapter 11 / 7 petition
``near_default`` Debt acceleration, delisting effected
``stress``       Listing-rule failure, restatement (non-reliance)
``early_warning``Late filing, auditor change, impairment, restructuring
===============  ====================================================

Pooling the tiers into one binary would inflate the headline number: a late
filing is noisy and routine for some filers, a Chapter 11 is unambiguous.
Reporting precision/recall per tier keeps that visible.

Everything here comes from **structured submissions metadata** — item codes and
form types — so no document fetching is required and the whole universe can be
scanned from cache. The trade-off is deliberate: item codes are filer-supplied
and noisy (``data/discover.py`` documents the failure modes), which is exactly
why the terminal ``default`` tier still requires the three-signal text
verification and is sourced from ``data/labels/chapter11.csv`` rather than from
item codes alone.

Deterministic plain code, no LLM.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Iterable, Sequence
from datetime import date
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from data.edgar import EdgarClient, default_user_agent

EVENTS_CSV = Path("data/labels/distress_events.csv")

TIER_DEFAULT = "default"
TIER_NEAR_DEFAULT = "near_default"
TIER_STRESS = "stress"
TIER_EARLY_WARNING = "early_warning"

#: Higher is worse. Used for ordering and for "worst state in window".
SEVERITY: dict[str, int] = {
    TIER_EARLY_WARNING: 1,
    TIER_STRESS: 2,
    TIER_NEAR_DEFAULT: 3,
    TIER_DEFAULT: 4,
}

#: 8-K item code -> (tier, signal name). Item 1.03 is deliberately absent: the
#: terminal tier comes from the verified label set, not from an item code that
#: J.C. Penney and Granite Construction both got wrong.
ITEM_SIGNALS: dict[str, tuple[str, str]] = {
    "2.04": (TIER_NEAR_DEFAULT, "debt_acceleration"),
    "3.01": (TIER_STRESS, "listing_rule_failure"),
    "4.02": (TIER_STRESS, "restatement_non_reliance"),
    "4.01": (TIER_EARLY_WARNING, "auditor_change"),
    "2.06": (TIER_EARLY_WARNING, "material_impairment"),
    "2.05": (TIER_EARLY_WARNING, "exit_or_disposal_costs"),
}

#: Form type -> (tier, signal name).
FORM_SIGNALS: dict[str, tuple[str, str]] = {
    "25": (TIER_NEAR_DEFAULT, "delisting_filed"),
    "25-NSE": (TIER_NEAR_DEFAULT, "delisting_by_exchange"),
    "NT 10-K": (TIER_EARLY_WARNING, "late_annual_report"),
    "NT 10-Q": (TIER_EARLY_WARNING, "late_quarterly_report"),
}

#: Signals common enough at healthy companies that they carry little weight
#: alone. Kept, but flagged, so a consumer can require corroboration.
NOISY_SIGNALS = frozenset({"auditor_change", "exit_or_disposal_costs", "material_impairment"})


class DistressEvent(BaseModel):
    """One dated distress signal for one filer."""

    model_config = ConfigDict(frozen=True)

    cik: int
    tier: str
    signal: str
    event_date: date
    as_of_date: date
    source_form: str
    source_accession: str
    noisy: bool = False

    @property
    def severity(self) -> int:
        return SEVERITY[self.tier]


def extract_events(client: EdgarClient, cik: int, since: date | None = None) -> list[DistressEvent]:
    """Every ladder event for one filer, from submissions metadata.

    The filing date serves as both ``event_date`` and ``as_of_date``: for a
    notice, the disclosure *is* the event, and both are public the same day.
    """
    out: list[DistressEvent] = []
    for f in client.filing_index(cik):
        filed = f["filing_date"]
        if filed is None or (since and filed < since):
            continue
        form = f["form"]

        hit = FORM_SIGNALS.get(form)
        if hit:
            tier, signal = hit
            out.append(
                DistressEvent(
                    cik=cik,
                    tier=tier,
                    signal=signal,
                    event_date=filed,
                    as_of_date=filed,
                    source_form=form,
                    source_accession=f["accession"],
                    noisy=signal in NOISY_SIGNALS,
                )
            )

        if form.startswith("8-K"):
            items = f.get("items") or ""
            for code, (tier, signal) in ITEM_SIGNALS.items():
                if code in items:
                    out.append(
                        DistressEvent(
                            cik=cik,
                            tier=tier,
                            signal=signal,
                            event_date=filed,
                            as_of_date=filed,
                            source_form=form,
                            source_accession=f["accession"],
                            noisy=signal in NOISY_SIGNALS,
                        )
                    )
    return sorted(out, key=lambda e: (e.event_date, e.signal))


def default_tier_events(labels: Iterable) -> list[DistressEvent]:
    """Terminal events, sourced from the verified Chapter 11 label set.

    The ladder reads item codes, but item 1.03 is exactly the code that filers
    miscode (``data/discover.py``), so the terminal tier is not derived from it.
    These come from ``data/labels/chapter11.csv``, where every row passed the
    three-signal check.
    """
    return [
        DistressEvent(
            cik=x.cik,
            tier=TIER_DEFAULT,
            signal="chapter11_petition",
            event_date=x.event_date,
            as_of_date=x.as_of_date,
            source_form="8-K",
            source_accession=x.source_accession or "",
            noisy=False,
        )
        for x in labels
    ]


def merged_event_set(
    events: Sequence[DistressEvent], labels: Iterable
) -> list[DistressEvent]:
    """The full ladder: scanned intermediate events plus verified terminal ones."""
    combined = list(events) + default_tier_events(labels)
    return sorted(combined, key=lambda e: (e.cik, e.event_date, -e.severity))


def events_visible_as_of(events: Iterable[DistressEvent], as_of: date) -> list[DistressEvent]:
    """Events publicly knowable strictly before ``as_of`` — for agent inputs.

    Mirrors ``data.facts.visible_as_of``. A prior restatement is legitimate
    evidence *if it was already public*.
    """
    return [e for e in events if e.as_of_date < as_of]


def worst_tier_within(
    events: Sequence[DistressEvent],
    cik: int,
    start: date,
    end: date,
    min_severity: int = 1,
    exclude_noisy: bool = False,
) -> DistressEvent | None:
    """Most severe event for ``cik`` in ``(start, end]``, earliest on ties.

    LOOKS AHEAD BY DESIGN — this is a grader, like
    ``labels.outcome_within_horizon``. Never expose its result to the agent.
    """
    hits = [
        e
        for e in events
        if e.cik == cik
        and start < e.event_date <= end
        and e.severity >= min_severity
        and not (exclude_noisy and e.noisy)
    ]
    if not hits:
        return None
    return min(hits, key=lambda e: (-e.severity, e.event_date))


def first_event_at_tier(
    events: Sequence[DistressEvent], cik: int, tier: str
) -> DistressEvent | None:
    """Earliest event for ``cik`` at or above ``tier`` — for lead-time analysis."""
    threshold = SEVERITY[tier]
    hits = [e for e in events if e.cik == cik and e.severity >= threshold]
    return min(hits, key=lambda e: e.event_date) if hits else None


def build_event_set(
    client: EdgarClient, ciks: Sequence[int], since: date | None = None, verbose: bool = True
) -> list[DistressEvent]:
    out: list[DistressEvent] = []
    for n, cik in enumerate(ciks, 1):
        try:
            out.extend(extract_events(client, cik, since=since))
        except Exception as exc:  # noqa: BLE001
            print(f"  warn: {cik}: {exc}", file=sys.stderr)
        if verbose and n % 50 == 0:
            print(f"  {n}/{len(ciks)} filers, {len(out)} events", file=sys.stderr)
    return out


def write_events(events: Sequence[DistressEvent], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "cik",
        "tier",
        "severity",
        "signal",
        "event_date",
        "as_of_date",
        "source_form",
        "source_accession",
        "noisy",
    ]
    with path.open("w", encoding="utf8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for e in events:
            writer.writerow(
                {
                    "cik": e.cik,
                    "tier": e.tier,
                    "severity": e.severity,
                    "signal": e.signal,
                    "event_date": e.event_date.isoformat(),
                    "as_of_date": e.as_of_date.isoformat(),
                    "source_form": e.source_form,
                    "source_accession": e.source_accession,
                    "noisy": int(e.noisy),
                }
            )


def load_events(path: Path | str = EVENTS_CSV) -> list[DistressEvent]:
    path = Path(path)
    if not path.exists():
        return []
    out = []
    with path.open(encoding="utf8", newline="") as fh:
        for row in csv.DictReader(fh):
            out.append(
                DistressEvent(
                    cik=int(row["cik"]),
                    tier=row["tier"],
                    signal=row["signal"],
                    event_date=date.fromisoformat(row["event_date"]),
                    as_of_date=date.fromisoformat(row["as_of_date"]),
                    source_form=row["source_form"],
                    source_accession=row["source_accession"],
                    noisy=row["noisy"] in ("1", "True", "true"),
                )
            )
    return out


def main(argv: list[str] | None = None) -> int:
    from data.labels import load_chapter11_labels
    from data.watchlist import load_watchlist

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--since", type=date.fromisoformat, default=date(2015, 1, 1))
    parser.add_argument("--out", type=Path, default=EVENTS_CSV)
    parser.add_argument("--user-agent", default=default_user_agent())
    args = parser.parse_args(argv)

    client = EdgarClient(user_agent=args.user_agent)
    ciks = [e.cik for e in load_watchlist()]
    print(f"scanning {len(ciks)} filers for ladder events since {args.since}", file=sys.stderr)
    events = merged_event_set(build_event_set(client, ciks, since=args.since), load_chapter11_labels())
    write_events(events, args.out)

    by_tier: dict[str, int] = {}
    firms: dict[str, set] = {}
    for e in events:
        by_tier[e.tier] = by_tier.get(e.tier, 0) + 1
        firms.setdefault(e.tier, set()).add(e.cik)
    print(f"\n{len(events)} events written to {args.out}", file=sys.stderr)
    for tier in sorted(by_tier, key=lambda t: -SEVERITY[t]):
        print(f"  {tier:16} {by_tier[tier]:>6} events  {len(firms[tier]):>4} filers", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
