"""Run the L3 backtest and report it against the deterministic baselines.

Uses the same panel and the same temporal split as ``models/run_baseline.py``,
so agent and baseline are scored on identical observations under identical
as-of rules. Comparing across different splits would measure the split.

Default agent is the rule-based control, which needs no endpoint. Pass
``--agent react`` once a model is configured.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

from agents.llm import (
    CachingClient,
    InfrastructureError,
    RateLimitedClient,
    load_env_file,
)
from agents.rulebased import RuleBasedInvestigator
from data.edgar import EdgarClient
from evals.backtest import (
    assert_no_lookahead,
    grade,
    reliability_curve,
    run_backtest,
    save_results,
    stratified_sample,
)
from models.hazard import AltmanBaseline, HazardBaseline, roc_auc
from models.panel import load_panel, split_by_date


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--panel", type=Path, default=Path("data/cache/panel.csv"))
    parser.add_argument("--results", type=Path, default=Path("data/cache/l3_results.csv"))
    parser.add_argument("--cutoff", type=date.fromisoformat, default=date(2024, 6, 1))
    parser.add_argument("--agent", choices=("rules", "react"), default="rules")
    parser.add_argument("--limit", type=int, default=0, help="cap test cases (0 = all)")
    parser.add_argument(
        "--max-negatives",
        type=int,
        default=0,
        help="stratified sample: keep all positives, cap negatives (0 = keep all)",
    )
    parser.add_argument("--no-cache", action="store_true", help="bypass the LLM response cache")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="grade only what the cache already holds; makes no API calls",
    )
    parser.add_argument("--model", default="", help="agent model id (overrides CREDITPULSE_LLM_MODEL)")
    parser.add_argument("--base-url", default="", help="OpenAI-compatible endpoint")
    parser.add_argument("--max-positives", type=int, default=0, help="cap positives (pilot runs)")
    parser.add_argument(
        "--with-baseline",
        action="store_true",
        help="let the agent consult the hazard baseline via get_model_score",
    )
    parser.add_argument(
        "--tpm",
        type=int,
        default=0,
        help="proactive tokens-per-minute ceiling (e.g. 16000 for gemma-4-31b-it)",
    )
    parser.add_argument(
        "--max-calls",
        type=int,
        default=0,
        help="self-imposed API call ceiling; stops cleanly, cached work is kept (0 = no cap)",
    )
    args = parser.parse_args(argv)

    loaded = load_env_file()
    if loaded:
        print(f"loaded from .env: {', '.join(loaded)}", file=sys.stderr)

    rows = load_panel(args.panel)
    if not rows:
        print(f"no panel at {args.panel}; run `python -m models.run_baseline --rebuild`", file=sys.stderr)
        return 1
    train, test = split_by_date(rows, args.cutoff)
    if args.limit:
        test = test[: args.limit]
    sample = stratified_sample(test, args.max_negatives, max_positives=args.max_positives)
    test = sample.cases
    if sample.negative_fraction < 1.0:
        print(
            f"stratified: {sample.n_positives} positives + "
            f"{sample.n_negatives_kept}/{sample.n_negatives_total} negatives "
            f"(fraction {sample.negative_fraction:.3f}); precision is base-rate corrected",
            file=sys.stderr,
        )

    if args.agent == "react":
        from agents.distress import DistressInvestigator
        from agents.llm import default_client, preflight

        # Wrap in the limiter *before* preflighting. An unwrapped preflight
        # dies on a transient per-minute 429, which is exactly the condition
        # the limiter exists to ride out -- it killed a restart once.
        client = RateLimitedClient(
            inner=default_client(model=args.model, base_url=args.base_url),
            max_calls=args.max_calls,
            tokens_per_minute=args.tpm,
        )
        if args.offline:
            # Build the identical client chain but never call through it. The
            # cache key includes the inner client's name, so a stand-in with a
            # different name would miss on every entry the real run wrote --
            # the replay would look like an empty cache rather than a hit.
            # Preflight is skipped: offline mode must touch no endpoint.
            client = CachingClient(inner=client, offline=True)
            print("offline replay: no API calls will be made", file=sys.stderr)
        else:
            ok, detail = preflight(client)
            if not ok:
                print(f"endpoint preflight failed: {detail}", file=sys.stderr)
                return 2
            print(f"endpoint ok: {detail}", file=sys.stderr)
            if not args.no_cache:
                # Cache outside the limiter: a cache hit costs no quota.
                client = CachingClient(inner=client)
        investigator = DistressInvestigator(client, with_baseline=args.with_baseline)
        label = f"ReAct investigator ({client.name})"
    else:
        investigator = RuleBasedInvestigator()
        label = "rule-based control"

    edgar = EdgarClient()
    cache: dict[int, list] = {}

    def facts_for(cik: int):
        """Facts for one filer, or a hard failure.

        This used to swallow the exception and substitute ``[]``. That is the
        worst possible handling: a transient EDGAR error silently handed the
        agent an empty universe, it investigated nothing and was then graded on
        the guess -- scoring a data outage as a judgment. It also made the run
        irreproducible, because the same case replayed differently depending on
        whether the fetch happened to succeed.
        """
        if cik not in cache:
            cache[cik] = edgar.facts(cik)
        return cache[cik]

    index_cache: dict[int, list] = {}

    def filings_for(cik: int) -> list:
        if cik not in index_cache:
            try:
                index_cache[cik] = edgar.filing_index(cik)
            except Exception:  # noqa: BLE001
                # Unlike facts, an absent index is survivable: the tools report
                # it as unavailable and the agent can say so, rather than being
                # handed a fabricated "no adverse events found".
                index_cache[cik] = []
        return index_cache[cik]

    def make_kwargs(row):
        extra = {
            "filing_index": filings_for(row.cik),
            "fetch_document": lambda acc, doc, _c=row.cik: edgar.fetch_filing_document(
                _c, acc, doc
            ),
        }
        if base_scores is not None:
            extra["model_score"] = base_scores[id(row)]
        return extra

    print(f"L3 backtest: {label} over {len(test)} test cases (cutoff {args.cutoff})", file=sys.stderr)

    def progress(n: int, _result) -> None:
        if n % 25 == 0:
            print(f"  {n}/{len(test)}", file=sys.stderr)

    # The agent may consult the baseline as one piece of evidence. It is fitted
    # only on the training window, so exposing it introduces no lookahead.
    base_scores = None
    if args.agent == "react" and args.with_baseline:
        scorer = HazardBaseline().fit(train)
        base_scores = dict(zip([id(c) for c in test], scorer.predict_proba(test), strict=True))
        print("agent has access to the hazard baseline via get_model_score", file=sys.stderr)
    case_kwargs = make_kwargs if args.agent == "react" else None

    try:
        results = run_backtest(
            investigator, test, facts_for, on_case=progress, case_kwargs=case_kwargs
        )
    except InfrastructureError as exc:
        print(f"\nABORTED -- endpoint unusable, no metrics are valid:\n  {exc}", file=sys.stderr)
        return 4
    if not results:
        print("\nno cases completed", file=sys.stderr)
        return 3
    assert_no_lookahead(results)
    save_results(results, args.results)
    print(f"per-case results -> {args.results}", file=sys.stderr)

    # A quota stop yields a partial run. Report it as such rather than
    # discarding it: metrics computed over positives stay valid on a subset.
    partial = len(results) < len(test)
    report = grade(results, negative_fraction=sample.negative_fraction)
    print(f"\n=== L3: {label}{' [PARTIAL]' if partial else ''} ===", file=sys.stderr)
    if partial:
        print(
            f"ran {len(results)}/{len(test)} cases before the quota stopped it. "
            "Recall, lead time and false-confidence are computed over positives "
            "and remain valid; precision carries wider error bars.",
            file=sys.stderr,
        )
    print(report.summary(), file=sys.stderr)

    print("\n  reliability curve (confidence -> observed failure rate):", file=sys.stderr)
    for row in reliability_curve(results):
        print(
            f"    [{row['bin_low']:.1f}-{row['bin_high']:.1f})  n={row['n']:>5}  "
            f"stated={row['mean_confidence']:.2f}  observed={row['observed_rate']:.2f}",
            file=sys.stderr,
        )

    # Score the baselines on *exactly* the rows the agent was graded on.
    #
    # Two ways this silently stopped being an apples-to-apples comparison:
    #
    # 1. The agent's AUC is computed over non-abstained cases only, because an
    #    abstention has no probability to rank. The baselines never abstain. So
    #    scoring them on the full split let the agent drop the cases it found
    #    hardest and keep the easy ones -- self-selection that flatters it.
    # 2. A run stopped by quota or outage grades a subset, while the baselines
    #    still saw every row. Different cases, so the difference partly
    #    measures which cases each side got.
    #
    # Both are fixed by matching on (cik, as_of) and reporting two arms.
    by_key = {(r.cik, r.observation_date): r for r in test}
    graded_cases = [c for c in results if not c.abstained]
    graded_rows = [by_key[(c.cik, c.as_of)] for c in graded_cases]
    completed_rows = [by_key[(c.cik, c.as_of)] for c in results]

    altman, hazard = AltmanBaseline(), HazardBaseline().fit(train)
    print(
        f"\n=== baselines on the IDENTICAL {len(graded_rows)} graded cases ===",
        file=sys.stderr,
    )
    agent_auc = f"{report.auc:.4f}" if report.auc is not None else "n/a"
    print(f"  agent {args.agent:<11}: auc={agent_auc}", file=sys.stderr)
    print(f"  Tier 0 Altman Z'' : {altman.evaluate(graded_rows).summary()}", file=sys.stderr)
    print(f"  Tier 1 hazard     : {hazard.evaluate(graded_rows).summary()}", file=sys.stderr)

    # Abstention must not be a free pass. Here every completed case counts, and
    # an abstention is scored as the uninformative 0.5 it is. This is the
    # stricter number, and the honest headline.
    if len(completed_rows) > len(graded_rows):
        y = [c.label for c in results]
        agent_auc_all = roc_auc(y, [c.risk_probability for c in results])
        print(
            f"\n=== all {len(results)} completed cases "
            f"(abstentions scored 0.5, not dropped) ===",
            file=sys.stderr,
        )
        print(f"  agent {args.agent:<11}: auc={agent_auc_all:.4f}", file=sys.stderr)
        print(f"  Tier 0 Altman Z'' : {altman.evaluate(completed_rows).summary()}", file=sys.stderr)
        print(f"  Tier 1 hazard     : {hazard.evaluate(completed_rows).summary()}", file=sys.stderr)

    print(
        "\nAUC is comparable across rows; precision/recall are not directly "
        "comparable, because the agent may abstain and the baselines never do.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
