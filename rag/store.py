"""The filtered chunk store, local or cloud, behind one interface.

**The as-of filter is not optional and is not the caller's to build.**
``search`` takes ``as_of`` as a required argument and constructs the filter
itself; nothing above this module can loosen it. That mirrors the tool layer,
where the prediction date is fixed by whoever built the toolbox and never by
the model. A retrieval layer that let the date slip would leak the outcome in
words -- a 10-K filed after the petition says "Chapter 11" plainly -- which
is the same leak the ratio panel guards against, in a form that is harder to
spot.

The filter is enforced twice: once in the query, and again on what comes back.
The second check should never fire. It exists so that if a backend, a version
upgrade or a payload mistake ever returns a future chunk, the run fails loudly
instead of producing a quietly excellent backtest.

**Filter first, then rank.** One filer before one date is a few thousand
chunks, so the search is effectively exact. Qdrant switches to a full scan
when a filter is that selective (given payload indexes), which avoids the
recall loss HNSW suffers under strict filters.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client import models as m

from rag.chunking import Chunk
from rag.embed import DIM, Embedder

COLLECTION = "filing_chunks"
#: Outside the project folder on purpose. The project sits in a OneDrive-synced
#: folder that the editor also watches; a store rewriting hundreds of MB there
#: had OneDrive, the file watcher and the virus scanner all chasing every
#: write, and the editor's processes died of memory pressure mid-ingestion.
#: Override with CREDITPULSE_QDRANT_PATH.
LOCAL_PATH = Path(
    os.environ.get("CREDITPULSE_QDRANT_PATH")
    or Path(os.environ.get("LOCALAPPDATA", Path.home())) / "CreditPulseIQ" / "qdrant"
)

#: How far back retrieval looks from the prediction date. Eighteen months
#: covers the last 10-K and every 10-Q since, which is what an analyst reads.
WINDOW_DAYS = 548

#: Candidates fetched per ranking before fusion and de-duplication.
CANDIDATES = 40

MODES = ("hybrid", "dense", "sparse")


#: Waits between attempts when the network drops. A Wi-Fi blip (DNS failing
#: for a few seconds) killed one ingestion run mid-upload; ~2.5 minutes of
#: patience covers that without hiding a cluster that is really down.
RETRY_WAITS = (2, 5, 15, 30, 60, 60)


def _retry(call):
    """Run ``call``, retrying on connection errors. Upserts are idempotent (stable ids)."""
    import time

    for wait in RETRY_WAITS:
        try:
            return call()
        except Exception as exc:  # noqa: BLE001 -- qdrant wraps httpx errors in its own types
            if not any(s in repr(exc) for s in ("ConnectError", "getaddrinfo", "ReadTimeout",
                                                "ConnectTimeout", "RemoteProtocolError", "WriteError")):
                raise
            from agents import netwait

            if netwait.enabled() and not netwait.online():
                netwait.wait_until_online()  # an outage: wait it out, then retry afresh
                return _retry(call)
            print(f"  [network] {type(exc).__name__}; retrying in {wait}s", flush=True)
            time.sleep(wait)
    return call()


class LookaheadError(AssertionError):
    """A chunk filed after the prediction date reached a caller."""


@dataclass(frozen=True)
class Hit:
    id: str
    score: float
    text: str
    section: str
    form: str
    accession: str
    filing_date: date
    chunk_no: int
    injection: bool

    @property
    def citation(self) -> str:
        filed = self.filing_date.isoformat()
        return f"{self.form} {self.accession} filed {filed}, {self.section} #{self.chunk_no}"


def open_client(where: str = "local") -> QdrantClient:
    """``local`` (on disk, no server), ``memory`` (tests) or ``cloud`` (from .env)."""
    if where == "memory":
        return QdrantClient(":memory:")
    if where == "local":
        LOCAL_PATH.mkdir(parents=True, exist_ok=True)
        return QdrantClient(path=str(LOCAL_PATH))
    if where == "cloud":
        from agents.llm import load_env_file

        load_env_file()
        url = os.environ.get("CREDITPULSE_QDRANT_URL")
        key = os.environ.get("CREDITPULSE_QDRANT_API_KEY")
        if not url or not key:
            raise RuntimeError("CREDITPULSE_QDRANT_URL / CREDITPULSE_QDRANT_API_KEY not set in .env")
        return QdrantClient(url=url, api_key=key, timeout=60)
    raise ValueError(f"unknown store location {where!r}")


class FilingStore:
    def __init__(self, client: QdrantClient, embedder: Embedder, collection: str = COLLECTION) -> None:
        self.client = client
        self.embedder = embedder
        self.collection = collection
        self._ensure()

    def _ensure(self) -> None:
        if self.client.collection_exists(self.collection):
            return
        self.client.create_collection(
            self.collection,
            vectors_config={"dense": m.VectorParams(size=DIM, distance=m.Distance.COSINE)},
            sparse_vectors_config={"bm25": m.SparseVectorParams(modifier=m.Modifier.IDF)},
        )
        # Indexes make the filters fast and let Qdrant pick exact search when a
        # filter is selective. Local mode ignores them, harmlessly.
        for field, schema in (
            ("cik", m.PayloadSchemaType.INTEGER),
            ("filed_ord", m.PayloadSchemaType.INTEGER),
            ("section", m.PayloadSchemaType.KEYWORD),
            ("form", m.PayloadSchemaType.KEYWORD),
            ("accession", m.PayloadSchemaType.KEYWORD),
        ):
            self.client.create_payload_index(self.collection, field, schema)

    # ---- writing -------------------------------------------------------

    def has_filing(self, accession: str) -> bool:
        flt = m.Filter(must=[m.FieldCondition(key="accession", match=m.MatchValue(value=accession))])
        return _retry(lambda: self.client.count(self.collection, count_filter=flt, exact=True)).count > 0

    def add(self, chunks: Sequence[Chunk], batch: int = 64) -> int:
        for i in range(0, len(chunks), batch):
            part = list(chunks[i : i + batch])
            texts = [c.text for c in part]
            dense = self.embedder.dense(texts)
            sparse = self.embedder.sparse(texts)
            _retry(lambda part=part, dense=dense, sparse=sparse: self.client.upsert(
                self.collection,
                points=[
                    m.PointStruct(
                        id=c.id,
                        vector={"dense": d, "bm25": m.SparseVector(indices=s[0], values=s[1])},
                        payload=c.payload(),
                    )
                    for c, d, s in zip(part, dense, sparse, strict=True)
                ],
            ))
        return len(chunks)

    # ---- reading -------------------------------------------------------

    @staticmethod
    def _filter(cik: int, as_of: date, window_days: int, sections: Iterable[str] | None) -> m.Filter:
        must = [
            m.FieldCondition(key="cik", match=m.MatchValue(value=int(cik))),
            m.FieldCondition(
                key="filed_ord",
                range=m.Range(
                    lte=as_of.toordinal(),
                    gte=(as_of - timedelta(days=window_days)).toordinal(),
                ),
            ),
        ]
        if sections:
            must.append(m.FieldCondition(key="section", match=m.MatchAny(any=list(sections))))
        return m.Filter(must=must)

    def search(
        self,
        cik: int,
        as_of: date,
        query: str,
        k: int = 5,
        sections: Iterable[str] | None = None,
        window_days: int = WINDOW_DAYS,
        mode: str = "hybrid",
    ) -> list[Hit]:
        """Top ``k`` passages for one filer, filed on or before ``as_of``."""
        if not isinstance(as_of, date):
            raise TypeError("as_of must be a date; retrieval without a prediction date is refused")
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        flt = self._filter(cik, as_of, window_days, sections)
        dense = m.Prefetch(
            query=self.embedder.dense_query(query), using="dense", filter=flt, limit=CANDIDATES
        )
        idx, val = self.embedder.sparse_query(query)
        sparse = m.Prefetch(
            query=m.SparseVector(indices=idx, values=val), using="bm25", filter=flt, limit=CANDIDATES
        )
        if mode == "hybrid":
            res = _retry(lambda: self.client.query_points(
                self.collection, prefetch=[dense, sparse], query=m.FusionQuery(fusion=m.Fusion.RRF),
                limit=CANDIDATES, with_payload=True,
            ))
        else:
            p = dense if mode == "dense" else sparse
            res = _retry(lambda: self.client.query_points(
                self.collection, query=p.query, using=p.using, query_filter=flt,
                limit=CANDIDATES, with_payload=True,
            ))

        hits: list[Hit] = []
        seen: set[str] = set()
        for point in res.points:
            pl = point.payload or {}
            filed = date.fromisoformat(pl["filed"])
            if filed > as_of or int(pl["cik"]) != int(cik):
                raise LookaheadError(
                    f"store returned {pl.get('accession')} (cik {pl.get('cik')}, filed {filed}) "
                    f"for cik {cik} as of {as_of}"
                )
            # Risk factors are copied quarter to quarter; five copies of one
            # paragraph are one result, not five.
            if pl["text_hash"] in seen:
                continue
            seen.add(pl["text_hash"])
            hits.append(
                Hit(
                    id=str(point.id), score=float(point.score), text=pl["text"], section=pl["section"],
                    form=pl["form"], accession=pl["accession"], filing_date=filed,
                    chunk_no=int(pl["chunk_no"]), injection=bool(pl.get("injection")),
                )
            )
            if len(hits) >= k:
                break
        return hits

    def chunks_for(self, cik: int, as_of: date, window_days: int = WINDOW_DAYS) -> list[dict]:
        """Every stored chunk visible for one filer at one date (payloads, with ids)."""
        flt = self._filter(cik, as_of, window_days, None)
        out, offset = [], None
        while True:
            points, offset = _retry(lambda offset=offset: self.client.scroll(
                self.collection, scroll_filter=flt, limit=512, offset=offset, with_payload=True
            ))
            out.extend({"id": str(p.id), **(p.payload or {})} for p in points)
            if offset is None:
                return out
