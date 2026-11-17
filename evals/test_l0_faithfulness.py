"""L0: the NLI faithfulness eval counts what it claims to.

The verdict logic is tested on fixed probability arrays; one test runs the real
local model on obvious pairs, skipped if the model is not cached (it is never
downloaded during tests).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from evals.run_faithfulness import MODEL_DIR, figure_sentences, sentences, verdict

#             contradiction entailment neutral
SUPPORTS = [0.05, 0.90, 0.05]
DENIES = [0.85, 0.05, 0.10]
SILENT = [0.10, 0.10, 0.80]


def test_sentences_skip_fragments():
    text = "Leverage is high. Ok. The company obtained a covenant waiver in March."
    assert sentences(text) == ["The company obtained a covenant waiver in March."]


def test_figures_become_premises():
    assert figure_sentences("liabilities_to_assets=0.91 | quick_ratio=0.2") == [
        "The liabilities to assets is 0.91.", "The quick ratio is 0.2.",
    ]


def test_any_entailing_source_supports_the_claim():
    label, idx, _ = verdict(np.array([SILENT, SUPPORTS, DENIES]))
    assert (label, idx) == ("supported", 1)


def test_contradiction_only_when_nothing_entails():
    assert verdict(np.array([SILENT, DENIES]))[0] == "contradicted"


def test_silence_is_neutral_not_supported():
    assert verdict(np.array([SILENT, SILENT]))[0] == "neutral"


@pytest.mark.skipif(not (Path(MODEL_DIR) / "onnx/model.onnx").exists(), reason="NLI model not cached")
def test_real_model_on_obvious_pairs():
    from evals.run_faithfulness import NLI

    nli = NLI(threads=2)
    src = ["The Company obtained a waiver of the leverage covenant from its lenders in March 2024."]
    claim = "The company received a covenant waiver from its lenders."
    assert verdict(nli.probs(src, claim))[0] == "supported"
    assert verdict(nli.probs(src, "The lenders refused to waive the covenant."))[0] == "contradicted"
