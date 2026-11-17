"""Copy the chunk store from local mode to the cloud cluster, vectors included.

    python -m rag.migrate            # local -> cloud, then verify counts

Why this exists: Qdrant's local mode holds the whole collection in the Python
process. At ~50k chunks the ingest process was at 2.4 GB and growing on a
machine whose editor dies below ~1 GB free, so ingestion moved to the cloud
cluster, where the collection lives server-side. Copying the stored points
(with their vectors) avoids re-embedding the ~50k chunks already done.

Idempotent: point ids are stable, so a rerun overwrites rather than
duplicates, and an interrupted copy can simply be run again.
"""

from __future__ import annotations

import sys

from qdrant_client import models as m

from rag.embed import HashEmbedder
from rag.store import COLLECTION, FilingStore, open_client

BATCH = 256


def migrate(batch: int = BATCH) -> tuple[int, int]:
    src = open_client("local")
    # Creating the store through FilingStore gives the cloud collection the
    # same vector config and payload indexes; the embedder is never used here.
    dst = FilingStore(open_client("cloud"), HashEmbedder()).client
    total = src.count(COLLECTION).count
    copied, offset = 0, None
    while True:
        points, offset = src.scroll(
            COLLECTION, limit=batch, offset=offset, with_payload=True, with_vectors=True
        )
        if points:
            dst.upsert(
                COLLECTION,
                points=[m.PointStruct(id=p.id, vector=p.vector, payload=p.payload) for p in points],
                wait=True,
            )
            copied += len(points)
            print(f"  {copied}/{total}", flush=True)
        if offset is None:
            break
    src.close()
    return total, dst.count(COLLECTION, exact=True).count


def main() -> int:
    local, cloud = migrate()
    print(f"local {local} chunks; cloud now {cloud}")
    return 0 if cloud >= local else 1


if __name__ == "__main__":
    sys.exit(main())
