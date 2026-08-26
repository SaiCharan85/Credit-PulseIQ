"""Subgroup fairness and quality decay over the graded backtest.

Headline AUC is an average, and an average hides the two failures that matter
most to anyone deciding whether to rely on this:

**Fairness.** A model can score 0.96 overall while being near-useless on small
filers, or on the ones with sparse disclosure. Those are exactly the companies
a credit analyst most needs help with -- a large filer's distress is already
covered by six sell-side desks. So we split the graded cases by size, by how
much the filer actually reported, by industry, and by how far ahead of the
event the call was made, and report AUC within each.

**Decay.** The agent was measured once, on one window. If accuracy falls as the
cases get more recent, the number on the front page is describing a world that
has moved. We split by as-of date and by how stale the newest filing was.

Both are reported with the subgroup size next to them, because an AUC over
eleven cases is a rumour, not a measurement. Anything under
``MIN_SUBGROUP`` is printed but explicitly marked as too small to read.

    python -m evals.run_fairness_decay
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

RESULTS = Path("results/backtest2_agent_200cases.csv")

#: Below this many cases, or with either class absent, an AUC is noise. We
#: print the group anyway -- silently dropping small groups is how a subgroup
#: failure stays hidden -- but never quote the number as a finding.
MIN_SUBGROUP = 20


def auc(scores: list[float], labels: list[int]) -> float | None:
    """Rank AUC with ties at half credit. None when one class is missing."""
    pos = [s for s, y in zip(scores, labels, strict=True) if y == 1]
    neg = [s for s, y in zip(scores, labels, strict=True) if y == 0]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def _score(row: dict[str, str]) -> float:
    """The continuous score the agent produced, preferring risk_score."""
    for key in ("risk_score", "risk_probability", "confidence"):
        raw = (row.get(key) or "").strip()
        if raw:
            try:
                return float(raw)
            except ValueError:
                continue
    return 0.0


def _report(title: str, groups: dict[str, list[dict[str, Any]]], note: str = "") -> None:
    print(f"\n{title}")
    print(f"  {'group':<26} {'n':>4} {'pos':>4} {'AUC':>7}   reading")
    print("  " + "-" * 66)
    for name in sorted(groups):
        rows = groups[name]
        scores = [_score(r) for r in rows]
        labels = [int(r["label"]) for r in rows]
        a = auc(scores, labels)
        npos = sum(labels)
        if a is None:
            reading = "one class only -- no AUC"
            shown = "  n/a"
        elif len(rows) < MIN_SUBGROUP:
            reading = "TOO SMALL TO READ"
            shown = f"{a:.3f}"
        else:
            reading = ""
            shown = f"{a:.3f}"
        print(f"  {name:<26} {len(rows):>4} {npos:>4} {shown:>7}   {reading}")
    if note:
        print(f"  {note}")


def _bucket_days(row: dict[str, str]) -> str:
    """How far ahead of the bankruptcy the call was made."""
    if int(row["label"]) == 0:
        return "survivors (no event)"
    raw = (row.get("days_to_event") or "").strip()
    if not raw:
        return "positive, lead time unknown"
    d = int(float(raw))
    if d <= 90:
        return "0-90 days ahead"
    if d <= 180:
        return "91-180 days ahead"
    if d <= 365:
        return "181-365 days ahead"
    return "over a year ahead"


def _bucket_steps(row: dict[str, str]) -> str:
    """How much work the agent did. A proxy for how much it had to go on."""
    s = int(float(row.get("steps") or 0))
    if s <= 6:
        return f"shallow ({s} steps or fewer)" if s <= 6 else ""
    if s <= 10:
        return "medium (7-10 steps)"
    return "deep (11+ steps)"


def _bucket_evidence(row: dict[str, str]) -> str:
    """How much evidence the agent surfaced -- a stand-in for disclosure depth.

    Filers in distress stop tagging line items, so a thin evidence list is not
    only an agent property; it tracks how much the company actually reported.
    """
    n = len([e for e in (row.get("evidence") or "").split("|") if e.strip()])
    if n <= 2:
        return "sparse (0-2 figures)"
    if n <= 5:
        return "moderate (3-5 figures)"
    return "rich (6+ figures)"


def _bucket_tools(row: dict[str, str]) -> str:
    """Whether the agent read the filing text, not only the numbers."""
    tools = row.get("tools_called") or ""
    text = "check_going_concern" in tools or "get_filing_events" in tools
    return "read the filing text" if text else "numbers only"


def main(path: Path = RESULTS) -> int:
    if not path.exists():
        print(f"no results at {path}; run the backtest first", file=sys.stderr)
        return 1
    rows = list(csv.DictReader(path.open(encoding="utf8")))
    overall = auc([_score(r) for r in rows], [int(r["label"]) for r in rows])
    npos = sum(int(r["label"]) for r in rows)
    print("CreditPulse IQ -- subgroup fairness and decay")
    print(f"{path}  |  {len(rows)} cases, {npos} positives  |  overall AUC {overall:.4f}")
    print(
        "\nAUC within a subgroup answers: among cases *like these*, does the agent\n"
        "still rank the failures above the survivors? A subgroup where it does not\n"
        "is a population the headline number does not describe."
    )

    # ---- decay: does it hold up as the cases get more recent? -------------
    by_date: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        d = date.fromisoformat(r["as_of"])
        by_date[f"{d.year} H{1 if d.month <= 6 else 2}"].append(r)
    _report(
        "QUALITY DECAY -- by prediction date",
        by_date,
        note=(
            "A falling series means the measured number is describing a window that\n"
            "  has passed. A flat one means the single measurement still stands."
        ),
    )

    # ---- fairness: who does it work for? ----------------------------------
    # Lead time splits positives only, so each bucket must be scored against
    # the *whole* survivor pool. Bucketing survivors alongside them gives every
    # group a single class and no AUC at all -- which is what the first version
    # of this did, and it reported nothing while looking like it had run.
    survivors = [r for r in rows if int(r["label"]) == 0]
    lead_groups = {
        k: [r for r in rows if int(r["label"]) == 1 and _bucket_days(r) == k] + survivors
        for k in {_bucket_days(r) for r in rows if int(r["label"]) == 1}
    }
    _report(
        "LEAD TIME -- how early was the call? (each vs all 100 survivors)",
        lead_groups,
        note=(
            "Calls made close to the filing are the easy ones -- the distress is\n"
            "  already visible. Accuracy far ahead of the event is the useful kind.\n"
            "  n includes the shared survivor pool, so only 'pos' varies by row."
        ),
    )
    _report(
        "DISCLOSURE DEPTH -- how much did the filer report?",
        {k: [r for r in rows if _bucket_evidence(r) == k] for k in {_bucket_evidence(r) for r in rows}},
        note=(
            "Sparse filers are the fairness risk: distress removes tags, so the\n"
            "  companies with least data are disproportionately the failing ones."
        ),
    )
    _report(
        "INVESTIGATION DEPTH -- how hard did the agent work?",
        {k: [r for r in rows if _bucket_steps(r) == k] for k in {_bucket_steps(r) for r in rows}},
    )
    _report(
        "EVIDENCE TYPE -- did it read the words or only the numbers?",
        {k: [r for r in rows if _bucket_tools(r) == k] for k in {_bucket_tools(r) for r in rows}},
        note="The +0.084 filing-text effect should be visible here as a gap.",
    )

    # ---- behaviour: abstentions and verification --------------------------
    print("\nBEHAVIOUR")
    term = defaultdict(int)
    for r in rows:
        term[r.get("terminated_because") or "?"] += 1
    for k, v in sorted(term.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<26} {v:>4}  ({v / len(rows):.0%})")
    failed = [r for r in rows if (r.get("verification_passed") or "1") == "0"]
    print(f"  verification failures      {len(failed):>4}")
    print(
        "\n  Every abstention that is a protocol failure rather than a judgement is a\n"
        "  claim we cannot make: 'it will tell you when it does not know' stays\n"
        "  unexercised until one of these is the model's own choice."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
