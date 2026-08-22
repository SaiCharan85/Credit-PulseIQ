"""Does the agent find the evidence that is actually there? (L4)

Every metric in this project scores the agent's *number*. None scores its
*explanation* -- which, now that a tuned gradient-boosted model beats the agent
on ranking by 0.025, is the only thing it is still claimed to be good at. A
capability nobody measures is a capability nobody should believe in.

This measures it, against deterministic ground truth rather than a judge model.
``signal_probe.csv`` records, per case, which warnings genuinely existed in the
filings as of the prediction date -- going-concern language, covenant breaches,
delisting notices, late filings. So the question has a right answer:

    of the warnings that were really there, how many did the agent surface?

The split between failure modes is the point. A missed signal can mean two
very different things:

* **Never looked** -- the tool that would have found it was never called. A
  search failure, and a prompt can fix it.
* **Looked and ignored** -- the tool was called, the signal was there, and it
  did not reach the memo. A judgment failure, and only a better model fixes it.

Those need opposite remedies and have been indistinguishable all along, which
is why the question "would a bigger model help?" has gone unanswered.

Two honest limits. This scores only the eight enumerated signals, so it cannot
credit the agent for noticing something nobody thought to list. And it rewards
*finding* evidence, not *weighting* it correctly -- an agent that cites six
alarms and still says `healthy` scores perfectly here and is useless. It
belongs beside AUC, never instead of it.
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import date
from pathlib import Path

#: Ground-truth signal -> the tool that would surface it, and words that show
#: it reached the memo. Matching is deliberately generous: the agent is being
#: scored on whether it *surfaced* the finding, not on phrasing.
SIGNALS: dict[str, tuple[str, tuple[str, ...]]] = {
    "going_concern": ("check_going_concern", ("going concern", "going-concern", "substantial doubt")),
    "material_weakness": ("check_going_concern", ("material weakness", "internal control")),
    "late_filing": ("get_filing_events", ("late filing", "nt 10-", "late-filing", "failed to file")),
    "covenant": ("get_filing_events", ("covenant", "2.04", "acceleration", "default")),
    "delisting": ("get_filing_events", ("delist", "3.01", "listing")),
    "auditor_change": ("get_filing_events", ("auditor", "4.01", "accountant")),
    "restatement": ("get_filing_events", ("restat", "4.02", "non-reliance", "no longer be relied")),
    "impairment": ("get_filing_events", ("impairment", "2.06", "write-down", "writedown")),
}

FOUND = "found"
LOOKED_MISSED = "looked_and_missed"
NEVER_LOOKED = "never_looked"


def load(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open(encoding="utf8")))


def classify(signal: str, tools: str, memo_text: str) -> str:
    """Did the agent surface this signal, miss it, or never look?"""
    tool, phrases = SIGNALS[signal]
    if any(p in memo_text for p in phrases):
        return FOUND
    return LOOKED_MISSED if tool in tools else NEVER_LOOKED


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--truth", type=Path, default=Path("data/cache/signal_probe.csv"))
    ap.add_argument("--results", type=Path, default=Path("data/cache/l3_sig_partial.csv"))
    args = ap.parse_args(argv)

    truth = {(int(r["cik"]), date.fromisoformat(r["as_of"])): r for r in load(args.truth)}
    results = {(int(r["cik"]), date.fromisoformat(r["as_of"])): r for r in load(args.results)}
    keys = sorted(set(truth) & set(results))
    if not keys:
        print("no overlap between ground truth and results", file=sys.stderr)
        return 1
    print(f"scoring {len(keys)} cases present in both", file=sys.stderr)

    tally: dict[str, dict[str, int]] = {s: {FOUND: 0, LOOKED_MISSED: 0, NEVER_LOOKED: 0} for s in SIGNALS}
    per_case: list[tuple] = []

    for key in keys:
        t, r = truth[key], results[key]
        memo = " ".join([r.get("rationale", ""), r.get("evidence", ""), r.get("residual", "")]).lower()
        tools = r.get("tools_called", "")
        present = [s for s in SIGNALS if t.get(s) == "1"]
        if not present:
            continue
        outcomes = {s: classify(s, tools, memo) for s in present}
        for s, o in outcomes.items():
            tally[s][o] += 1
        found = sum(1 for o in outcomes.values() if o == FOUND)
        per_case.append((key[0], key[1], len(present), found, int(r["label"])))

    print(f"\n{'signal':<20}{'present':>8}{'found':>8}{'recall':>9}"
          f"{'looked/missed':>15}{'never looked':>14}", file=sys.stderr)
    tot = {FOUND: 0, LOOKED_MISSED: 0, NEVER_LOOKED: 0}
    for s, counts in tally.items():
        n = sum(counts.values())
        if not n:
            continue
        for k in tot:
            tot[k] += counts[k]
        print(f"{s:<20}{n:>8}{counts[FOUND]:>8}{counts[FOUND] / n:>8.0%}"
              f"{counts[LOOKED_MISSED]:>15}{counts[NEVER_LOOKED]:>14}", file=sys.stderr)
    n = sum(tot.values())
    if n:
        print(f"\n{'ALL SIGNALS':<20}{n:>8}{tot[FOUND]:>8}{tot[FOUND] / n:>8.0%}"
              f"{tot[LOOKED_MISSED]:>15}{tot[NEVER_LOOKED]:>14}", file=sys.stderr)
        missed = tot[LOOKED_MISSED] + tot[NEVER_LOOKED]
        if missed:
            share = tot[NEVER_LOOKED] / missed
            print(
                f"\nOf the {missed} signals missed, {share:.0%} were never looked for.\n"
                + (
                    "Mostly a search failure -- the tool was not called, which a prompt can fix."
                    if share > 0.6
                    else "Mostly a judgment failure -- the tool was called and the finding did "
                         "not reach the memo, which needs a better model, not a better prompt."
                ),
                file=sys.stderr,
            )

    if per_case:
        full = sum(1 for _, _, p, f, _ in per_case if f == p)
        none = sum(1 for _, _, p, f, _ in per_case if f == 0)
        print(f"\ncases with any signal present : {len(per_case)}", file=sys.stderr)
        print(f"  surfaced every one          : {full} ({full / len(per_case):.0%})", file=sys.stderr)
        print(f"  surfaced none               : {none} ({none / len(per_case):.0%})", file=sys.stderr)
        pos = [c for c in per_case if c[4] == 1]
        if pos:
            rec = sum(f for _, _, _, f, _ in pos) / max(1, sum(p for _, _, p, _, _ in pos))
            print(f"  recall on cases that failed : {rec:.0%}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
