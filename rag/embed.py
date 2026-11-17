"""Embeddings: dense (bge-small) for meaning, sparse (BM25) for exact wording.

Both, because filings are written in fixed formulae. "Substantial doubt",
"forbearance", "waiver" are terms of art a keyword match finds reliably and a
dense model can blur; paraphrased discussion of a liquidity squeeze is the
reverse. Qdrant fuses the two rankings.

Runs locally on CPU through ONNX (fastembed), with no torch: the installed
torch build is incompatible with NumPy 2, and pulling it in also loads
TensorFlow, which costs memory this machine does not have to spare.

``HashEmbedder`` exists for tests: deterministic, instant, no model download,
and good enough to check that filters -- not similarity -- decide what comes
back.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Protocol

DENSE_MODEL = "BAAI/bge-small-en-v1.5"
SPARSE_MODEL = "Qdrant/bm25"
DIM = 384

#: bge models are trained with this instruction on the query side only.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

#: Inside the gitignored cache, not %TEMP%, which Windows may clear.
MODEL_CACHE = Path("data/cache/fastembed")

#: Small batches keep peak memory low; throughput barely changes past ~16.
BATCH = 16

#: CPU threads for ONNX. Half the cores: all eight at 100% for half an hour
#: preceded an unexpected hard reset of this laptop (heat/power, on battery).
THREADS = 4

#: DirectML runs ONNX on any DirectX 12 GPU, AMD included (PyTorch's GPU path
#: is CUDA, NVIDIA-only). Measured on this laptop's Radeon RX 7700S: 3.8x the
#: 4-thread CPU speed, vectors identical to cosine 0.99999. Used when the
#: ``onnxruntime-directml`` build is installed; otherwise CPU. Set
#: CREDITPULSE_EMBED_CPU=1 to force CPU.
GPU_PROVIDERS = ["DmlExecutionProvider", "CPUExecutionProvider"]


def _gpu_available() -> bool:
    if os.environ.get("CREDITPULSE_EMBED_CPU"):
        return False
    try:
        import onnxruntime

        return "DmlExecutionProvider" in onnxruntime.get_available_providers()
    except Exception:  # noqa: BLE001
        return False


Sparse = tuple[list[int], list[float]]


class Embedder(Protocol):
    def dense(self, texts: list[str]) -> list[list[float]]: ...
    def dense_query(self, text: str) -> list[float]: ...
    def sparse(self, texts: list[str]) -> list[Sparse]: ...
    def sparse_query(self, text: str) -> Sparse: ...


class FastEmbedder:
    def __init__(self, threads: int | None = THREADS) -> None:
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        from fastembed import SparseTextEmbedding, TextEmbedding

        MODEL_CACHE.mkdir(parents=True, exist_ok=True)
        self.device = "gpu" if _gpu_available() else "cpu"
        kwargs = {"providers": GPU_PROVIDERS} if self.device == "gpu" else {"threads": threads}
        self._dense = TextEmbedding(DENSE_MODEL, cache_dir=str(MODEL_CACHE), **kwargs)
        self._sparse = SparseTextEmbedding(SPARSE_MODEL, cache_dir=str(MODEL_CACHE))

    def dense(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._dense.embed(texts, batch_size=BATCH)]

    def dense_query(self, text: str) -> list[float]:
        return next(iter(self._dense.embed([QUERY_PREFIX + text]))).tolist()

    def sparse(self, texts: list[str]) -> list[Sparse]:
        return [(v.indices.tolist(), v.values.tolist()) for v in self._sparse.embed(texts, batch_size=BATCH)]

    def sparse_query(self, text: str) -> Sparse:
        v = next(iter(self._sparse.query_embed(text)))
        return v.indices.tolist(), v.values.tolist()


_TOKEN = re.compile(r"[a-z0-9]+")


def _bucket(token: str, dim: int) -> int:
    return int(hashlib.md5(token.encode()).hexdigest()[:8], 16) % dim


class HashEmbedder:
    """Bag-of-words hashed into fixed dimensions. For tests only."""

    def __init__(self, dim: int = DIM) -> None:
        self.dim = dim

    def _counts(self, text: str) -> Counter:
        return Counter(_bucket(t, self.dim) for t in _TOKEN.findall(text.lower()))

    def dense(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vec = [0.0] * self.dim
            for i, c in self._counts(text).items():
                vec[i] = float(c)
            norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            out.append([x / norm for x in vec])
        return out

    def dense_query(self, text: str) -> list[float]:
        return self.dense([text])[0]

    def sparse(self, texts: list[str]) -> list[Sparse]:
        out = []
        for text in texts:
            counts = self._counts(text) or Counter({0: 0.0})
            idx = sorted(counts)
            out.append((idx, [float(counts[i]) for i in idx]))
        return out

    def sparse_query(self, text: str) -> Sparse:
        return self.sparse([text])[0]
