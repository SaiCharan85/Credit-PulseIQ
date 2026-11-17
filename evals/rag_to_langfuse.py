"""Publish retrieval evals to Langfuse: one dataset, one run per retrieval variant.

    python -m evals.rag_to_langfuse results/rag_recall_full.items.jsonl results/rag_tune_terms.items.jsonl ...

Items are (warning family, filer, as-of) -- the same cases every variant is
scored on -- so Langfuse lines the variants up case by case. Scores are those
``run_rag_recall`` already computed (hybrid mode unless --mode says otherwise);
nothing is re-searched here. Per item: recall@5, reciprocal rank, precision@5,
nDCG@10. Per run, for each family and overall: their means.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

DATASET = "creditpulse-rag-retrieval"
METRICS = ("recall@5", "rr", "precision@5", "ndcg@10")


def load(path: Path, mode: str) -> tuple[str, dict[str, dict]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf8").splitlines() if line]
    rows = [r for r in rows if r["mode"] == mode]
    variant = rows[0]["variant"] if rows else path.stem
    return variant, {f"{r['family']}-{r['cik']}-{r['as_of']}": r for r in rows}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--mode", default="hybrid", choices=("hybrid", "dense", "sparse"))
    ap.add_argument("--dataset", default=DATASET)
    args = ap.parse_args(argv)

    from agents.llm import load_env_file

    load_env_file()
    from langfuse import Langfuse
    from langfuse.experiment import Evaluation

    runs = [load(p, args.mode) for p in args.files if p.exists()]
    if not runs:
        print("no item files found", file=sys.stderr)
        return 1
    lf = Langfuse()
    if not lf.auth_check():
        print("Langfuse credentials rejected -- check LANGFUSE_HOST region", file=sys.stderr)
        return 1

    try:
        lf.create_dataset(name=args.dataset, description="Filing-text retrieval cases (family, filer, as-of)")
    except Exception:  # noqa: BLE001 -- already exists
        pass
    cases: dict[str, dict] = {}
    for _, items in runs:
        for k, r in items.items():
            cases.setdefault(k, r)
    for k, r in cases.items():
        lf.create_dataset_item(
            dataset_name=args.dataset, id=k,
            input={"family": r["family"], "cik": r["cik"], "as_of": r["as_of"]},
            expected_output={"n_relevant": r["n_relevant"]},
        )
    lf.flush()
    dataset = lf.get_dataset(args.dataset)
    print(f"dataset {args.dataset}: {len(dataset.items)} items")

    for variant, items in runs:
        def task(*, item, items=items, **_):
            i = item.input
            r = items.get(f"{i['family']}-{i['cik']}-{i['as_of']}")
            return {"missing": True} if r is None else {m: r[m] for m in (*METRICS, "ranks")}

        def per_item(*, output, **_):
            if not output or output.get("missing"):
                return []
            return [Evaluation(name=m, value=round(float(output[m]), 4)) for m in METRICS]

        def run_level(*, item_results, **_):
            sums: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
            for res in item_results:
                out = res.output or {}
                if out.get("missing"):
                    continue
                for scope in (res.item.input["family"], "all"):
                    for m in METRICS:
                        sums[scope][m].append(float(out[m]))
            return [Evaluation(name=f"{scope}/{m}", value=round(sum(v) / len(v), 4))
                    for scope, by_m in sums.items() for m, v in by_m.items() if v]

        result = dataset.run_experiment(
            name=f"retrieval-{variant}", run_name=f"{variant} ({args.mode})",
            description=f"retrieval variant {variant}, {args.mode} search",
            task=task, evaluators=[per_item], run_evaluators=[run_level], max_concurrency=8,
            metadata={"variant": variant, "mode": args.mode},
        )
        scores = {e.name: e.value for e in (result.run_evaluations or []) if e.name.startswith("all/")}
        print(f"run {variant}: {len(items)} items  {scores}")
    lf.flush()
    print("open Langfuse > Datasets >", args.dataset, "to compare variants")
    return 0


if __name__ == "__main__":
    sys.exit(main())
