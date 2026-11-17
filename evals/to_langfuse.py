"""Publish backtest arms to Langfuse as one dataset with one experiment run per arm.

    python -m evals.to_langfuse                                   # both A/B arms
    python -m evals.to_langfuse --arm single=data/cache/l3_single.csv

Why a dataset and runs rather than loose traces: the arms are scored on the
*identical* cases, and Langfuse's experiment view lines runs up item by item.
A later run (say, after the retrieval query fix) lands as another run on the
same dataset and is compared against these without any bookkeeping.

Nothing is re-computed here. Outputs come from the results CSVs the backtest
wrote and the faithfulness detail file the NLI eval wrote, so uploading costs
no model calls and cannot drift from the numbers in ``results/``.

Per case: Brier error, correct-at-0.5, abstained, steps; for an arm with an
NLI detail file, faithfulness and contradicted-claim count. Per run: AUC with
abstentions scored 0.5 (the backtest's own convention), abstention rate, mean
Brier. Credentials come from .env (LANGFUSE_PUBLIC_KEY / _SECRET_KEY / _HOST).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

DATASET = "creditpulse-distress-ab"
DEFAULT_ARMS = {
    "tool-loop": Path("data/cache/l3_react_replay.csv"),
    "single-pass-rag": Path("data/cache/l3_single.csv"),
}
#: Arm -> NLI claim file. Empty on purpose: the small NLI model labelled plainly
#: correct figure-laden claims "contradicted" (8.6% supported), so its scores
#: measured the model, not the agent, and were removed from Langfuse. Re-enable
#: only with a checker validated against hand-labelled claims.
FAITHFULNESS: dict[str, Path] = {}


def key(cik, as_of) -> str:
    return f"{int(cik)}@{as_of}"


def load_rows(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf8") as fh:
        return {key(r["cik"], r["as_of"]): r for r in csv.DictReader(fh)}


def load_faithfulness(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf8").splitlines():
        rec = json.loads(line)
        claims = rec.get("claims", [])
        if claims:
            out[key(rec["cik"], rec["as_of"])] = {
                "faithfulness": sum(c["label"] == "supported" for c in claims) / len(claims),
                "contradicted": sum(c["label"] == "contradicted" for c in claims),
            }
    return out


def probability(row: dict) -> float:
    """The graded probability; abstentions are scored 0.5, as the backtest does."""
    if row["signal"] == "insufficient_evidence" or row.get("risk_probability", "") == "":
        return 0.5
    return float(row["risk_probability"])


def auc(pairs: list[tuple[float, int]]) -> float | None:
    pos = [p for p, y in pairs if y == 1]
    neg = [p for p, y in pairs if y == 0]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", action="append", default=[], help="name=path/to/results.csv (repeatable)")
    ap.add_argument("--dataset", default=DATASET)
    args = ap.parse_args(argv)

    from agents.llm import load_env_file

    load_env_file()
    from langfuse import Langfuse
    from langfuse.experiment import Evaluation

    arms = dict(a.split("=", 1) for a in args.arm) if args.arm else DEFAULT_ARMS
    arms = {k: Path(v) for k, v in arms.items() if Path(v).exists()}
    if not arms:
        print("no results files found", file=sys.stderr)
        return 1

    lf = Langfuse()
    if not lf.auth_check():
        print("Langfuse credentials rejected -- check LANGFUSE_HOST region", file=sys.stderr)
        return 1

    # The dataset is the union of cases across arms, keyed by (cik, as_of), so
    # re-running is idempotent: same ids, items upserted, not duplicated.
    cases: dict[str, dict] = {}
    for path in arms.values():
        for k, r in load_rows(path).items():
            cases.setdefault(k, {"cik": int(r["cik"]), "as_of": r["as_of"], "label": int(r["label"]),
                                 "days_to_event": r.get("days_to_event", "")})
    try:
        lf.create_dataset(name=args.dataset, description="Stratified distress backtest cases (A/B arms)")
    except Exception:  # noqa: BLE001 -- already exists
        pass
    for k, c in cases.items():
        lf.create_dataset_item(
            dataset_name=args.dataset, id=k.replace("@", "-"),
            input={"cik": c["cik"], "as_of": c["as_of"]},
            expected_output={"label": c["label"], "days_to_event": c["days_to_event"]},
        )
    lf.flush()
    dataset = lf.get_dataset(args.dataset)
    print(f"dataset {args.dataset}: {len(dataset.items)} items")

    for arm, path in arms.items():
        rows = load_rows(path)
        faith = load_faithfulness(FAITHFULNESS.get(arm, Path("-")))

        def task(*, item, rows=rows, **_):
            r = rows.get(key(item.input["cik"], item.input["as_of"]))
            if r is None:
                return {"missing": True}
            return {"signal": r["signal"], "risk_score": r["risk_score"], "probability": probability(r),
                    "confidence": r["confidence"], "steps": r["steps"],
                    "terminated_because": r["terminated_because"], "rationale": r["rationale"]}

        def per_case(*, input, output, expected_output, faith=faith, **_):
            if not output or output.get("missing"):
                return []
            y, p = int(expected_output["label"]), float(output["probability"])
            evals = [
                Evaluation(name="brier", value=round((p - y) ** 2, 4)),
                Evaluation(name="correct_at_0.5", value=(p >= 0.5) == bool(y), data_type="BOOLEAN"),
                Evaluation(name="abstained", value=output["signal"] == "insufficient_evidence",
                           data_type="BOOLEAN"),
                Evaluation(name="steps", value=float(output["steps"] or 0)),
            ]
            f = faith.get(key(input["cik"], input["as_of"]))
            if f:
                evals += [Evaluation(name="faithfulness_nli", value=round(f["faithfulness"], 3)),
                          Evaluation(name="contradicted_claims", value=float(f["contradicted"]))]
            return evals

        def run_level(*, item_results, **_):
            pairs, abst, brier = [], 0, []
            for res in item_results:
                out = res.output or {}
                if out.get("missing"):
                    continue
                y, p = int(res.item.expected_output["label"]), float(out["probability"])
                pairs.append((p, y))
                abst += out["signal"] == "insufficient_evidence"
                brier.append((p - y) ** 2)
            a = auc(pairs)
            return [e for e in (
                Evaluation(name="auc", value=round(a, 4)) if a is not None else None,
                Evaluation(name="abstention_rate", value=round(abst / len(pairs), 4)) if pairs else None,
                Evaluation(name="mean_brier", value=round(sum(brier) / len(brier), 4)) if brier else None,
            ) if e is not None]

        result = dataset.run_experiment(
            name=f"distress-{arm}", run_name=arm, description=f"results from {path}",
            task=task, evaluators=[per_case], run_evaluators=[run_level], max_concurrency=8,
            metadata={"results_file": str(path)},
        )
        scores = {e.name: e.value for e in (result.run_evaluations or [])}
        print(f"run {arm}: {len(rows)} cases  {scores}")
    lf.flush()
    print("open Langfuse > Datasets >", args.dataset, "to compare runs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
