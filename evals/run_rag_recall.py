"""Does retrieval surface the warnings that are really in the filings? (L4)

    python -m evals.run_rag_recall               # local store, all ingested filers
    python -m evals.run_rag_recall --k 5 --store cloud

Ground truth is deterministic, not a judge. For each case (filer, as-of) and
each warning family, the *relevant* chunks are the stored chunks visible at
that date whose text matches the family's patterns -- the same going-concern
and material-weakness patterns the existing scan uses, plus covenant-trouble
patterns. The question then has a right answer:

    asked in plain words, does a top-k result contain a relevant chunk?

Queries are written the way an analyst or the agent would ask, *not* copied
from the patterns, so lexical overlap alone cannot carry the result. All three
search modes are scored on the same cases, which is how the hybrid design gets
justified rather than assumed.

Coverage is reported separately: of the cases where the signal probe found a
warning in the latest report, how many have at least one matching chunk in
the store? A coverage miss is an ingestion or section-splitting failure, and
no ranking can fix it -- which is why it must not be folded into recall.

Two honest limits. Pattern ground truth credits only passages that use the
standard wording, so a warning phrased unusually counts as neither found nor
missed. And recall measures *finding* the passage, not whether a model then
reads it correctly; that is the faithfulness eval's job.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import pandas as pd

from data.signals import GOING_CONCERN_PATTERNS, MATERIAL_WEAKNESS_PATTERNS
from rag.store import MODES, FilingStore, open_client

#: Covenant *trouble*, not covenant *boilerplate*. A first, looser version
#: matched "failure to comply with these covenants could result in default"
#: and "the agreement contains customary covenants and events of default ...
#: the Company was in compliance" -- hypothetical and routine text present in
#: nearly every debtor's filings -- so most "misses" were retrieval correctly
#: ignoring boilerplate. These require a stated breach, waiver, forbearance or
#: an explicit risk of non-compliance; hypotheticals are rejected below.
COVENANT_PATTERNS = (
    r"(?:was|were|is|are|has\s+been|have\s+been|remained?)\s+not\s+in\s+compliance\s+with[^.]{0,80}?covenants?",
    r"(?:at\s+risk\s+of|expects?\s+to\s+be\s+in|resulted\s+in|will\s+result\s+in)\s+(?:non-?compliance|a\s+breach)"
    r"[^.]{0,60}?covenants?",
    r"(?:obtained|received|entered\s+into|executed)\s+(?:a\s+|an\s+)?(?:limited\s+)?(?:waiver|forbearance)",
    r"forbearance\s+agreement",
    r"covenants?[^.]{0,60}?(?:were|was)\s+waived",
    r"(?:breached|violated)\s+(?:certain\s+|the\s+|its\s+|our\s+)?(?:financial\s+)?covenants?",
)

_HYPOTHETICAL = re.compile(r"\b(?:could|may|might|would)\s+(?:result|cause|lead|constitute|trigger)\b", re.I)

#: family -> (patterns defining relevance, plain-language query, probe column)
FAMILIES: dict[str, tuple[tuple[str, ...], str, str | None]] = {
    "going_concern": (
        GOING_CONCERN_PATTERNS,
        "doubt about whether the company can keep operating and meet its obligations over the next year",
        "going_concern",
    ),
    "material_weakness": (
        MATERIAL_WEAKNESS_PATTERNS,
        "problems with the company's internal controls over its financial reporting",
        "material_weakness",
    ),
    "covenant": (
        COVENANT_PATTERNS,
        "the company breached loan terms and needed lenders to waive defaults or forbear",
        None,  # the probe's covenant flag is an 8-K item code, not report text
    ),
}


#: Tuned retrieval for the two weak families, tried as variants against the
#: plain-language baseline above. Three levers, each measured separately:
#:
#: * **terms** -- several queries in the filings' own vocabulary, fused by rank.
#:   This deliberately shares words with the ground-truth patterns, so sparse
#:   gains here are partly lexical by construction; that is also how a real
#:   analyst searches, but read the *dense* column as the cleaner signal.
#: * **sections** -- search only where the disclosure lives. Material
#:   weakness: the controls item (plus risk factors, which repeat it). Covenant
#:   trouble: liquidity, MD&A, the debt notes and defaults. ``full`` and
#:   ``unlabeled`` stay in, so a filing whose sections could not be split is
#:   not silently excluded.
#: * **rerank** -- a cross-encoder reads query and passage together, which can
#:   tell "were not effective" from "were effective"; embeddings mostly cannot.
TUNED: dict[str, dict[str, tuple[str, ...]]] = {
    "material_weakness": {
        "queries": (
            "material weakness in internal control over financial reporting",
            "disclosure controls and procedures were not effective",
            "management concluded internal control over financial reporting was not effective",
            "remediation plan for the identified material weakness",
        ),
        "sections": ("controls", "risk_factors", "full", "unlabeled"),
    },
    "covenant": {
        "queries": (
            "not in compliance with the financial covenants of the credit agreement",
            "lenders granted a waiver of the covenant default",
            "forbearance agreement with lenders",
            "amendment to the credit agreement to waive the event of default",
        ),
        "sections": ("liquidity", "mda", "financials", "defaults", "full", "unlabeled"),
    },
}
#: Second covenant attempt. The first queries centred on waivers and
#: forbearance and scored *below* the plain baseline. Of the real covenant-breach
#: chunks, 76 of 189 sit in the financial-statement debt notes and 61 in
#: liquidity, which describe breaches as compliance status, amendments,
#: acceleration and debt reclassified as current -- so v2 asks in that voice,
#: and the sections widen to everywhere the chunks were actually found.
TUNED_V2: dict[str, dict[str, tuple[str, ...]]] = {
    "covenant": {
        "queries": (
            "the company was not in compliance with the financial covenants of its credit facility",
            "the lenders agreed to an amendment and waiver of the covenant violation",
            "event of default under the loan agreement allowing lenders to accelerate the debt",
            "long-term debt reclassified as current because of the covenant breach",
            "amendment to the credit agreement relaxing the leverage ratio covenant",
        ),
        "sections": ("financials", "liquidity", "risk_factors", "market_risk", "mda", "defaults",
                     "full", "unlabeled"),
    },
}
VARIANTS = ("baseline", "terms", "terms+sections", "terms+sections+rerank",
            "terms2", "terms2+sections", "terms2+sections+rerank")
RERANKER = "BAAI/bge-reranker-v2-m3-int8"
RRF_K = 60


def _fuse(lists: list[list]) -> list:
    """Reciprocal-rank fusion across several queries' hit lists, by chunk id."""
    score: dict[str, float] = defaultdict(float)
    first: dict[str, object] = {}
    for hits in lists:
        for rank, h in enumerate(hits, 1):
            score[h.id] += 1.0 / (RRF_K + rank)
            first.setdefault(h.id, h)
    return [first[i] for i in sorted(score, key=score.get, reverse=True)]


def tuned_search(store: FilingStore, cik: int, as_of: date, family: str, variant: str,
                 mode: str, n: int, reranker=None) -> list:
    """Hits for one family under one variant; ``baseline`` is the plain query."""
    if variant == "baseline" or family not in TUNED:
        return store.search(cik, as_of, FAMILIES[family][1], k=n, mode=mode)
    cfg = TUNED_V2.get(family, TUNED[family]) if variant.startswith("terms2") else TUNED[family]
    sections = cfg["sections"] if "sections" in variant else None
    pool = 40 if "rerank" in variant else n
    fused = _fuse([store.search(cik, as_of, q, k=pool, sections=sections, mode=mode)
                   for q in cfg["queries"]])
    if "rerank" in variant and reranker is not None and fused:
        scores = list(reranker.rerank(cfg["queries"][0], [h.text for h in fused]))
        fused = [h for _, h in sorted(zip(scores, fused, strict=True), key=lambda p: -p[0])]
    return fused[:n]


def relevant_ids(chunks: list[dict], patterns: tuple[str, ...]) -> set[str]:
    """Chunks whose text matches; a covenant match inside a hypothetical sentence does not count."""
    rx = [re.compile(p, re.IGNORECASE) for p in patterns]
    strict = patterns is COVENANT_PATTERNS
    out = set()
    for c in chunks:
        for r in rx:
            m = r.search(c["text"])
            if not m:
                continue
            if strict:
                text = c["text"]
                start = max(text.rfind(".", 0, m.start()), 0)
                end = text.find(".", m.end())
                if _HYPOTHETICAL.search(text[start : end if end != -1 else len(text)]):
                    continue
            out.add(c["id"])
            break
    return out


#: Grade 1 of the graded covenant key: covenant *stress* short of a stated
#: breach -- amendments to covenants, relief, warnings that compliance is in
#: doubt. Reading the misses showed retrieval ranking exactly these first while
#: the strict key (grade 2, COVENANT_PATTERNS) gave them no credit. Routine
#: wording ("customary covenants", "in compliance with all covenants") is
#: excluded so boilerplate earns nothing.
COVENANT_STRESS_PATTERNS = (
    r"amend(?:ed|ment)[^.]{0,120}?(?:financial\s+)?covenants?",
    r"covenants?[^.]{0,80}?(?:were|was)\s+amended",
    r"(?:absent|without)\s+(?:a\s+|an\s+)?(?:waiver|amendment)[^.]{0,120}?covenants?",
    r"(?:cannot|can\s*not|no)\s+(?:provide\s+)?assurances?[^.]{0,120}?covenants?",
    r"covenant\s+relief",
    r"(?:may|might|will)\s+not\s+(?:be\s+able\s+to\s+)?(?:remain\s+in\s+compliance|comply|meet)[^.]{0,80}?covenants?",
)
_ROUTINE = re.compile(r"customary|in\s+compliance\s+with\s+all", re.I)


def covenant_grades(chunks: list[dict]) -> dict[str, int]:
    """Chunk id -> 2 (stated breach/waiver/forbearance) or 1 (covenant stress)."""
    strict = relevant_ids(chunks, COVENANT_PATTERNS)
    rx = [re.compile(p, re.IGNORECASE) for p in COVENANT_STRESS_PATTERNS]
    grades = {i: 2 for i in strict}
    for c in chunks:
        if c["id"] in grades:
            continue
        for r in rx:
            m = r.search(c["text"])
            if not m:
                continue
            text = c["text"]
            start = max(text.rfind(".", 0, m.start()), 0)
            end = text.find(".", m.end())
            if not _ROUTINE.search(text[start : end if end != -1 else len(text)]):
                grades[c["id"]] = 1
                break
    return grades


def ndcg_graded(hit_grades: list[int], all_grades: list[int], cutoff: int = 10) -> float:
    """Graded nDCG@cutoff with gain 2^grade - 1 (breach 3, stress 1)."""
    import math

    dcg = sum((2 ** g - 1) / math.log2(i + 1) for i, g in enumerate(hit_grades[:cutoff], 1))
    best = sorted(all_grades, reverse=True)[:cutoff]
    ideal = sum((2 ** g - 1) / math.log2(i + 1) for i, g in enumerate(best, 1))
    return dcg / ideal if ideal else 0.0


NDCG_CUTOFF = 10


def ndcg(ranks: list[int], n_relevant: int, cutoff: int = NDCG_CUTOFF) -> float:
    """Binary-relevance nDCG@cutoff: each relevant hit earns 1/log2(rank+1),
    normalised by the best possible ranking (all relevant chunks first).
    Unlike MRR it credits every relevant chunk, not just the first."""
    import math

    dcg = sum(1.0 / math.log2(r + 1) for r in ranks if r <= cutoff)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(n_relevant, cutoff) + 1))
    return dcg / ideal if ideal else 0.0


def evaluate(store: FilingStore, cases: pd.DataFrame, k: int, variant: str = "baseline",
             families: tuple[str, ...] = tuple(FAMILIES), reranker=None, graded: bool = False) -> dict:
    found = defaultdict(lambda: defaultdict(int))  # family -> mode -> hits
    rr = defaultdict(lambda: defaultdict(float))  # reciprocal rank sums
    prec = defaultdict(lambda: defaultdict(float))  # precision@k sums
    nd = defaultdict(lambda: defaultdict(float))  # nDCG@10 sums
    scored = defaultdict(int)  # family -> cases with ground truth
    coverage = defaultdict(lambda: [0, 0])  # family -> [covered, probe positives]
    misses: list[str] = []
    items: list[dict] = []  # one record per (case, family, mode), for per-item tracking
    gsum = defaultdict(lambda: defaultdict(float))  # graded covenant metrics: metric -> mode -> sum
    gscored = 0

    for row in cases.itertuples():
        cik, as_of = int(row.cik), date.fromisoformat(row.as_of)
        visible = store.chunks_for(cik, as_of)
        if not visible:
            continue
        for family, (patterns, _query, probe_col) in FAMILIES.items():
            if family not in families:
                continue
            truth = relevant_ids(visible, patterns)
            grades = covenant_grades(visible) if graded and family == "covenant" else None
            if grades is not None and grades:
                gscored += 1
            if probe_col and getattr(row, probe_col) == 1:
                coverage[family][1] += 1
                coverage[family][0] += bool(truth)
            if not truth:
                continue
            scored[family] += 1
            for mode in MODES:
                hits = tuned_search(store, cik, as_of, family, variant, mode,
                                    max(k, NDCG_CUTOFF), reranker)
                ranks = [i for i, h in enumerate(hits, 1) if h.id in truth]
                nd[family][mode] += ndcg(ranks, len(truth))
                graded_rec = {}
                if grades:
                    hg = [grades.get(h.id, 0) for h in hits]
                    graded_rec = {
                        f"any_recall@{k}": float(any(hg[:k])),
                        f"any_precision@{k}": sum(g > 0 for g in hg[:k]) / len(hg[:k]) if hg else 0.0,
                        f"graded_ndcg@{NDCG_CUTOFF}": ndcg_graded(hg, list(grades.values()), NDCG_CUTOFF),
                        "n_stress": sum(g == 1 for g in grades.values()),
                    }
                    for name, v in graded_rec.items():
                        if name != "n_stress":
                            gsum[name][mode] += v
                items.append({
                    "family": family, "cik": cik, "as_of": as_of.isoformat(), "mode": mode,
                    "n_relevant": len(truth), "ranks": ranks[:NDCG_CUTOFF],
                    f"recall@{k}": float(bool(ranks) and ranks[0] <= k),
                    "rr": 1.0 / ranks[0] if ranks else 0.0,
                    f"precision@{k}": sum(h.id in truth for h in hits[:k]) / len(hits[:k]) if hits else 0.0,
                    f"ndcg@{NDCG_CUTOFF}": ndcg(ranks, len(truth)),
                    **graded_rec,
                })
                top = hits[:k]
                if top:
                    prec[family][mode] += sum(h.id in truth for h in top) / len(top)
                if ranks and ranks[0] <= k:
                    found[family][mode] += 1
                elif mode == "hybrid":
                    misses.append(
                        f"{family:<17} cik {cik} as of {as_of}: "
                        f"{len(truth)} relevant chunks, none in top {k}"
                    )
                if ranks:
                    rr[family][mode] += 1.0 / ranks[0]
    return {"found": found, "rr": rr, "prec": prec, "ndcg": nd, "scored": scored,
            "coverage": coverage, "misses": misses, "items": items,
            "graded": gsum, "graded_cases": gscored}


def report(res: dict, k: int) -> str:
    lines = [f"Recall@{k} (share of cases with a relevant chunk in the top {k}); MRR@10 in brackets", ""]
    lines.append(f"{'family':<18}{'cases':>6}  " + "".join(f"{m:>18}" for m in MODES))
    tot_found, tot_cases = defaultdict(int), 0
    for family in FAMILIES:
        n = res["scored"][family]
        tot_cases += n
        cells = []
        for mode in MODES:
            f = res["found"][family][mode]
            tot_found[mode] += f
            mrr = res["rr"][family][mode] / n if n else 0.0
            cells.append(f"{(f / n if n else 0):>9.2f} [{mrr:.2f}]  ")
        lines.append(f"{family:<18}{n:>6}  " + "".join(f"{c:>18}" for c in cells))
    lines.append(
        f"{'all':<18}{tot_cases:>6}  "
        + "".join(f"{(tot_found[m] / tot_cases if tot_cases else 0):>9.2f}{'':9}" for m in MODES)
    )
    lines += ["", f"Context precision@{k} (share of the top {k} that are relevant):"]
    for family in FAMILIES:
        n = res["scored"][family]
        lines.append(f"{family:<18}{n:>6}  " + "".join(
            f"{(res['prec'][family][m] / n if n else 0):>9.2f}{'':9}" for m in MODES))
    lines += ["", f"nDCG@{NDCG_CUTOFF} (binary relevance; credits every relevant chunk by rank):"]
    for family in FAMILIES:
        n = res["scored"][family]
        lines.append(f"{family:<18}{n:>6}  " + "".join(
            f"{(res['ndcg'][family][m] / n if n else 0):>9.2f}{'':9}" for m in MODES))
    if res.get("graded_cases"):
        n = res["scored"]["covenant"]  # same cases as the strict key; graded adds credit, not cases
        lines += ["", f"Covenant, GRADED key (2 = stated breach/waiver/forbearance, 1 = covenant stress), "
                      f"{n} cases:"]
        for metric in (f"any_recall@{k}", f"any_precision@{k}", f"graded_ndcg@{NDCG_CUTOFF}"):
            lines.append(f"{metric:<18}{n:>6}  " + "".join(
                f"{(res['graded'][metric][m] / n if n else 0):>9.2f}{'':9}" for m in MODES))
    lines += ["", "Coverage (probe found the warning in the latest report -> store has a matching chunk):"]
    for family, (cov, pos) in res["coverage"].items():
        lines.append(f"  {family:<17} {cov}/{pos}" + (f"  ({cov / pos:.0%})" if pos else ""))
    if res["misses"]:
        lines += ["", f"Hybrid misses ({len(res['misses'])}):"] + [f"  {m}" for m in res["misses"][:25]]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Recall@k of filing-text retrieval")
    ap.add_argument("--cases", type=Path, default=Path("results/signal_probe_200cases.csv"))
    ap.add_argument("--store", choices=["local", "cloud"], default="local")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", type=Path, default=Path("results/rag_recall.log"))
    ap.add_argument("--variant", choices=VARIANTS, default="baseline")
    ap.add_argument("--graded", action="store_true",
                    help="also score covenant on the graded key (breach 2, stress 1)")
    ap.add_argument("--families", default=",".join(FAMILIES),
                    help="comma-separated subset, e.g. material_weakness,covenant")
    args = ap.parse_args(argv)

    from rag.embed import FastEmbedder

    reranker = None
    if "rerank" in args.variant:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        from rag.embed import MODEL_CACHE

        reranker = TextCrossEncoder(RERANKER, cache_dir=str(MODEL_CACHE), threads=4)
    families = tuple(f for f in args.families.split(",") if f)
    store = FilingStore(open_client(args.store), FastEmbedder())
    res = evaluate(store, pd.read_csv(args.cases), args.k, args.variant, families, reranker,
                   graded=args.graded)
    text = f"variant: {args.variant}\n" + report(res, args.k)
    print(text)
    args.out.write_text(text + "\n", encoding="utf8")
    import json

    # Per-item records beside the summary, for Langfuse (evals/rag_to_langfuse.py).
    items_path = args.out.with_suffix(".items.jsonl")
    items_path.write_text(
        "".join(json.dumps({"variant": args.variant, **r}) + "\n" for r in res["items"]), encoding="utf8"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
