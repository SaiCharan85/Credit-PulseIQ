"""One filer in, one cited memo out. The MVP entry point.

    python run_memo.py --cik 320193 --as-of 2024-06-01

Everything the backtest exercises, for a single company: as-of filtered facts,
triage, the investigator, the guard gate, and the rendered memo.

``--agent rules`` needs no model and no network beyond EDGAR, which makes it
the honest smoke test -- if the deterministic control cannot produce a memo for
a filer, neither will the ReAct loop, and the failure will be cheaper to read.

Refusing to print a blocked memo is the point, not an inconvenience. A memo
that trips the numeric or scope guard is withheld and the reason printed
instead, because a guard that degrades to a warning is not a guard.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date

from agents.guards import GUARD_NUMERIC, GUARD_SCOPE
from agents.llm import load_env_file
from agents.orchestrator import Orchestrator
from data.edgar import EdgarClient


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cik", type=int, required=True)
    ap.add_argument("--as-of", type=date.fromisoformat, required=True)
    ap.add_argument("--agent", choices=("rules", "react"), default="rules")
    ap.add_argument("--model", default="", help="override CREDITPULSE_LLM_MODEL")
    ap.add_argument("--base-url", default="", help="OpenAI-compatible endpoint")
    ap.add_argument("--show-trail", action="store_true", help="print every tool call")
    args = ap.parse_args(argv)

    load_env_file()
    edgar = EdgarClient()
    try:
        facts = edgar.facts(args.cik)
    except Exception as exc:  # noqa: BLE001
        print(f"could not load filings for CIK {args.cik}: {exc}", file=sys.stderr)
        return 2
    if not facts:
        print(f"no XBRL facts for CIK {args.cik}", file=sys.stderr)
        return 2
    print(f"loaded {len(facts)} facts for CIK {args.cik}", file=sys.stderr)

    if args.agent == "react":
        from agents.distress import DistressInvestigator
        from agents.llm import CachingClient, RateLimitedClient, default_client, preflight

        client = RateLimitedClient(
            inner=default_client(model=args.model, base_url=args.base_url)
        )
        ok, detail = preflight(client)
        if not ok:
            print(f"endpoint unavailable: {detail}", file=sys.stderr)
            return 3
        investigator = DistressInvestigator(CachingClient(inner=client))
    else:
        from agents.rulebased import RuleBasedInvestigator

        investigator = RuleBasedInvestigator()

    filing_index: list = []
    if args.agent == "react":
        try:
            filing_index = edgar.filing_index(args.cik)
        except Exception:  # noqa: BLE001
            print("filing index unavailable; text tools will report an error", file=sys.stderr)

    result = Orchestrator(investigator).run(
        args.cik,
        args.as_of,
        facts,
        **(
            {
                "filing_index": filing_index,
                "fetch_document": lambda a, d: edgar.fetch_filing_document(args.cik, a, d),
            }
            if args.agent == "react"
            else {}
        ),
    )

    print(
        f"triage: {result.triage.depth} ({result.triage.steps} steps) -- "
        f"{result.triage.reasons[0]}",
        file=sys.stderr,
    )

    if not result.shipped:
        print("\nNO MEMO ISSUED.", file=sys.stderr)
        print(result.blocked_reason, file=sys.stderr)
        blocking = GUARD_NUMERIC in result.blocked_reason or GUARD_SCOPE in result.blocked_reason
        print(
            "\nA blocked memo is the guard working. "
            + ("A figure could not be reproduced, or the output stated an action."
               if blocking else ""),
            file=sys.stderr,
        )
        return 1

    print()
    print(result.memo.render())
    if args.show_trail:
        print("\nAUDIT TRAIL")
        for n, call in enumerate(result.memo.audit_trail, 1):
            ok = "ok " if call.get("ok") else "ERR"
            print(f"  {n:>2}. [{ok}] {call.get('tool')}  {call.get('summary') or call.get('error')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
