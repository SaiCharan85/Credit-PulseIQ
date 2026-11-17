"""Are the single-pass answers faithful to their sources? (L5)

    python -m evals.run_faithfulness                  # all graded cases
    python -m evals.run_faithfulness --limit 20

Retrieval recall asks whether the right passage was *found*. This asks the
other half: given what the answer was shown, is each claim it makes backed by
those sources?

**No LLM grades this.** An answer graded by a model can be optimised against
that model. Faithfulness here is natural-language inference: a small local
cross-encoder (``cross-encoder/nli-deberta-v3-small``, ONNX, on CPU) reads
each (source, claim) pair and scores entailment, contradiction and neutral.
Same inputs, same scores, every run -- and it costs no API quota.

Each claim (a sentence of the rationale) is checked against every source the
answer was shown -- the retrieved passages and the cited figures, phrased as
sentences -- and takes the strongest verdict:

    supported     some source entails it          (entailment >= THRESHOLD)
    contradicted  some source contradicts it, none entails it
    neutral       no source entails or contradicts it

    faithfulness       = supported / claims
    contradiction rate = contradicted / claims

Two honest limits. *Neutral* includes legitimate inferences ("this is a
company that cannot fund itself") -- those are judgements, and NLI rightly
finds no source stating them -- so faithfulness is a floor, not a ceiling;
the contradiction rate is the sharper number. And a small NLI model misreads
some long, hedged filing sentences, so read the listed contradictions by
hand before acting on any of them.

Passages are re-retrieved rather than stored: retrieval is deterministic for a
fixed store, so the eval sees exactly what each answer saw.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from datetime import date
from functools import lru_cache
from pathlib import Path

import numpy as np

MODEL_REPO = "cross-encoder/nli-deberta-v3-small"
MODEL_DIR = Path("data/cache/nli")
#: Label order from the model's config.json.
CONTRADICTION, ENTAILMENT, NEUTRAL = 0, 1, 2
THRESHOLD = 0.5
MAX_TOKENS = 512

_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


def sentences(text: str) -> list[str]:
    """Claims: rationale sentences long enough to assert something."""
    return [s.strip() for s in _SENT.split(text or "") if len(s.split()) >= 4]


def figure_sentences(evidence: str) -> list[str]:
    """``metric=value | ...`` from the results CSV, as premises NLI can read."""
    out = []
    for part in (evidence or "").split("|"):
        if "=" in part:
            metric, value = part.split("=", 1)
            out.append(f"The {metric.strip().replace('_', ' ')} is {value.strip()}.")
    return out


class NLI:
    """ONNX cross-encoder: (premise, hypothesis) -> P(contradiction, entailment, neutral)."""

    def __init__(self, threads: int = 4) -> None:
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        paths = {f: hf_hub_download(MODEL_REPO, f, local_dir=str(MODEL_DIR))
                 for f in ("onnx/model.onnx", "tokenizer.json")}
        self.tok = Tokenizer.from_file(paths["tokenizer.json"])
        self.tok.enable_truncation(MAX_TOKENS, strategy="only_first")
        self.tok.enable_padding()
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(
            paths["onnx/model.onnx"], opts, providers=["CPUExecutionProvider"]
        )

    def probs(self, premises: list[str], hypothesis: str) -> np.ndarray:
        enc = self.tok.encode_batch([(p, hypothesis) for p in premises])
        feed = {
            "input_ids": np.array([e.ids for e in enc], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64),
        }
        logits = self.session.run(None, feed)[0]
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)


def verdict(probs: np.ndarray) -> tuple[str, int, float]:
    """Strongest verdict over all sources: (label, source index, probability)."""
    ent, con = probs[:, ENTAILMENT], probs[:, CONTRADICTION]
    i = int(ent.argmax())
    if ent[i] >= THRESHOLD:
        return "supported", i, float(ent[i])
    j = int(con.argmax())
    if con[j] >= THRESHOLD:
        return "contradicted", j, float(con[j])
    return "neutral", i, float(ent[i])


def check_case(nli, rationale: str, figures: list[str], passages: list[dict]) -> list[dict]:
    sources = [p["text"] for p in passages] + figures
    names = [p["id"] for p in passages] + [f"F{k + 1}" for k in range(len(figures))]
    out = []
    for claim in sentences(rationale):
        if not sources:
            out.append({"claim": claim, "label": "neutral", "source": None, "p": 0.0})
            continue
        label, idx, p = verdict(nli.probs(sources, claim))
        out.append({"claim": claim, "label": label, "source": names[idx], "p": round(p, 3)})
    return out


@lru_cache(maxsize=1)
def _nli() -> NLI:
    return NLI()


#: The question every distress answer is answering (agents/distress.py USER_TEMPLATE).
QUESTION = ("How likely is this company to file for bankruptcy protection within the next 12 months, "
            "judged from its filings?")


def answer_relevance(embedder, rationales: list[str]) -> list[float]:
    """Cosine between the question and each answer, with the retrieval embedder (bge).

    No LLM: RAGAS-style answer relevance uses a model to invent questions from
    the answer; a fixed task makes that unnecessary. Absolute values depend on
    the embedder, so read this comparatively -- arm against arm.
    """
    q = np.array(embedder.dense_query(QUESTION))
    out = []
    for i in range(0, len(rationales), 32):
        for v in embedder.dense(rationales[i : i + 32]):
            v = np.array(v)
            out.append(float(q @ v / (np.linalg.norm(q) * np.linalg.norm(v) or 1.0)))
    return out


def _rationales(path: Path) -> list[str]:
    with path.open(encoding="utf8") as fh:
        return [r["rationale"] for r in csv.DictReader(fh)
                if r["signal"] != "insufficient_evidence" and r.get("rationale")]


def relevance_lines(embedder, *paths: Path) -> list[str]:
    lines = []
    for p in paths:
        if not p.exists():
            continue
        scores = answer_relevance(embedder, _rationales(p))
        if scores:
            lines.append(
                f"  {p.name:<24} mean {np.mean(scores):.3f}  median {np.median(scores):.3f}  n={len(scores)}"
            )
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="NLI faithfulness of single-pass answers to their sources")
    ap.add_argument("--results", type=Path, default=Path("data/cache/l3_single.csv"))
    ap.add_argument("--store", choices=["local", "cloud"], default="cloud")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("results/faithfulness.log"))
    ap.add_argument("--detail", type=Path, default=Path("results/faithfulness_claims.jsonl"))
    ap.add_argument("--compare", type=Path, default=Path("data/cache/l3_react_replay.csv"),
                    help="other arm's results, for answer relevance side by side")
    args = ap.parse_args(argv)

    from agents.llm import load_env_file
    from agents.single_pass import retrieve
    from rag.embed import FastEmbedder
    from rag.store import FilingStore, open_client

    load_env_file()
    embedder = FastEmbedder()
    store = FilingStore(open_client(args.store), embedder)
    nli = _nli()

    with args.results.open(encoding="utf8") as fh:
        rows = [r for r in csv.DictReader(fh)
                if r["signal"] != "insufficient_evidence" and r.get("rationale")]
    if args.limit:
        rows = rows[: args.limit]

    totals: Counter = Counter()
    case_scores: list[float] = []
    contradicted: list[str] = []
    shown = used = 0
    with args.detail.open("w", encoding="utf8") as det:
        for n, r in enumerate(rows, 1):
            cik, as_of = int(r["cik"]), date.fromisoformat(r["as_of"])
            passages = retrieve(store, cik, as_of)
            claims = check_case(nli, r["rationale"], figure_sentences(r.get("evidence", "")), passages)
            c = Counter(x["label"] for x in claims)
            shown += len(passages)
            used += len({x["source"] for x in claims if x["label"] == "supported"
                         and str(x["source"]).startswith("P")})
            totals.update(c)
            if claims:
                case_scores.append(c["supported"] / len(claims))
            contradicted += [f"cik {cik} {as_of} vs {x['source']} (p={x['p']}): {x['claim'][:150]}"
                             for x in claims if x["label"] == "contradicted"]
            det.write(json.dumps({"cik": cik, "as_of": r["as_of"], "claims": claims}) + "\n")
            if n % 20 == 0:
                print(f"  {n}/{len(rows)}", flush=True)

    n_claims = sum(totals.values())
    lines = [
        f"Faithfulness of single-pass answers: {len(rows)} graded cases, {n_claims} claims",
        f"NLI model {MODEL_REPO} (local, deterministic, no LLM); threshold {THRESHOLD}",
        "",
        f"supported     {totals['supported']:>5}  ({totals['supported'] / n_claims:.1%})" if n_claims else "",
        (f"contradicted  {totals['contradicted']:>5}  ({totals['contradicted'] / n_claims:.1%})"
         if n_claims else ""),
        f"neutral       {totals['neutral']:>5}  ({totals['neutral'] / n_claims:.1%})"
        "  -- includes inferences; faithfulness is a floor" if n_claims else "",
        f"case mean faithfulness: {sum(case_scores) / len(case_scores):.3f}" if case_scores else "",
        f"context utilisation: {used}/{shown} passages shown support at least one claim"
        + (f" ({used / shown:.1%})" if shown else ""),
        "",
        "answer relevance (cosine of question vs answer, bge; compare arms, not absolutes):",
        *relevance_lines(embedder, args.results, args.compare),
        "",
        f"contradicted claims ({len(contradicted)}) -- read these by hand:",
        *[f"  {c}" for c in contradicted[:40]],
    ]
    text = "\n".join(lines)
    print(text)
    args.out.write_text(text + "\n", encoding="utf8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
