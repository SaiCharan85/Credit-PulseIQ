"""L0: the recall eval scores what it claims to.

An eval that miscounts is worse than none, because its number gets believed.
These pin the three things the Recall@k report rests on: pattern ground truth
picks the right chunks, a case with no ground truth is excluded rather than
counted as a miss, and coverage counts probe positives the store cannot match.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from evals.run_rag_recall import FAMILIES, evaluate, relevant_ids
from rag.chunking import Chunk
from rag.embed import HashEmbedder
from rag.store import FilingStore, open_client


def _store() -> FilingStore:
    s = FilingStore(open_client("memory"), HashEmbedder())
    texts = [
        "There is substantial doubt about the company's ability to continue as a going concern "
        "and to meet its obligations over the next year while it keeps operating.",
        "Revenue grew in every segment and margins improved.",
        "We were not in compliance with certain financial covenants "
        "and the lenders agreed to waive the defaults.",
    ]
    s.add(
        [
            Chunk(cik=1, accession="a1", form="10-K", filing_date=date(2024, 3, 1), section="financials",
                  chunk_no=i, text=t)
            for i, t in enumerate(texts)
        ]
    )
    return s


def test_pattern_ground_truth_selects_matching_chunks():
    chunks = [
        {"id": "x", "text": "ability to continue as a going concern"},
        {"id": "y", "text": "all is well"},
    ]
    assert relevant_ids(chunks, FAMILIES["going_concern"][0]) == {"x"}


def test_found_and_excluded_and_coverage():
    cases = pd.DataFrame(
        [
            {"cik": 1, "as_of": "2024-07-01", "going_concern": 1, "material_weakness": 1},
            {"cik": 99, "as_of": "2024-07-01", "going_concern": 1, "material_weakness": 0},  # not ingested
        ]
    )
    res = evaluate(_store(), cases, k=3)
    # going concern and covenant have ground truth; material weakness does not.
    assert res["scored"]["going_concern"] == 1
    assert res["scored"]["covenant"] == 1
    assert res["scored"]["material_weakness"] == 0
    assert res["found"]["going_concern"]["hybrid"] == 1
    # Probe said material weakness, store has no matching text: a coverage miss.
    assert res["coverage"]["material_weakness"] == [0, 1]
    assert res["coverage"]["going_concern"] == [1, 1]


def test_ndcg_matches_hand_computation():
    from evals.run_rag_recall import ndcg

    # relevant at ranks 2 and 4, two relevant chunks exist:
    # (1/log2 3 + 1/log2 5) / (1 + 1/log2 3) = (0.6309 + 0.4307) / 1.6309
    assert abs(ndcg([2, 4], 2) - 0.6509) < 1e-3
    assert ndcg([1, 2], 2) == 1.0
    assert ndcg([], 3) == 0.0
    assert ndcg([11], 1) == 0.0  # beyond the cutoff earns nothing


def test_covenant_grades_separate_breach_stress_and_boilerplate():
    from evals.run_rag_recall import covenant_grades

    chunks = [
        {"id": "breach", "text": "The Company was not in compliance with the leverage covenants."},
        {"id": "stress", "text": "We entered into an amendment that relaxed the financial covenants."},
        {"id": "routine", "text": "The amended agreement contains customary covenants."},
        {"id": "none", "text": "Revenue grew in the quarter."},
    ]
    assert covenant_grades(chunks) == {"breach": 2, "stress": 1}


def test_graded_ndcg_rewards_breach_above_stress():
    from evals.run_rag_recall import ndcg_graded

    grades = [2, 1]
    assert ndcg_graded([2, 1], grades) == 1.0
    assert ndcg_graded([1, 2], grades) < 1.0
    assert ndcg_graded([0, 0, 1], grades) < ndcg_graded([1, 0, 0], grades)
