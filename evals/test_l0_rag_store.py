"""L0: retrieval can never see the future, and never sees another filer.

The as-of rule is the property every backtest number rests on. In the ratio
panel a leak looks like a suspiciously good AUC; in retrieval it looks like a
passage, because a 10-K filed after the petition says "Chapter 11" in plain
words. So these tests plant exactly that passage, worded to match the query
perfectly, and require that it never comes back -- in every search mode.

Uses an in-memory store and a hashing embedder: what is under test is the
filter, not the ranking, so no model download is needed.
"""

from __future__ import annotations

from datetime import date

import pytest

from rag.chunking import Chunk
from rag.embed import HashEmbedder
from rag.store import MODES, FilingStore, LookaheadError, open_client

AS_OF = date(2024, 7, 1)
CIK = 1111
OTHER = 2222

PETITION = "the Company filed a voluntary petition for relief under Chapter 11 of the Bankruptcy Code"


def _c(cik: int, filed: date, text: str, no: int, section: str = "mda", acc: str | None = None) -> Chunk:
    return Chunk(
        cik=cik, accession=acc or f"{cik}-{filed.isoformat()}", form="10-Q",
        filing_date=filed, section=section, chunk_no=no, text=text,
    )


@pytest.fixture()
def store() -> FilingStore:
    s = FilingStore(open_client("memory"), HashEmbedder())
    s.add(
        [
            _c(CIK, date(2024, 5, 10), "liquidity is tight and the revolver matures next year", 0),
            _c(CIK, date(2024, 5, 10),
               "substantial doubt about our ability to continue as a going concern", 1, "financials"),
            _c(CIK, date(2024, 7, 1), "covenant waiver obtained from lenders on the as-of date", 2),
            # Filed after the prediction date, worded to match the query exactly.
            _c(CIK, date(2024, 7, 2), PETITION, 3, acc="future-8k"),
            _c(CIK, date(2025, 1, 15), PETITION + " and the plan was confirmed", 4, acc="future-10k"),
            # Too old for the freshness window.
            _c(CIK, date(2021, 1, 5), PETITION + " in a prior decade", 5, acc="ancient"),
            # Another filer saying the same thing, before the date.
            _c(OTHER, date(2024, 3, 1), PETITION, 6, acc="other-filer"),
        ]
    )
    return s


@pytest.mark.parametrize("mode", MODES)
def test_nothing_filed_after_asof_is_returned(store, mode):
    hits = store.search(CIK, AS_OF, PETITION, k=10, mode=mode)
    assert hits, "the visible chunks should still be found"
    assert all(h.filing_date <= AS_OF for h in hits)
    assert not {"future-8k", "future-10k"} & {h.accession for h in hits}


@pytest.mark.parametrize("mode", MODES)
def test_other_filers_are_never_returned(store, mode):
    hits = store.search(CIK, AS_OF, PETITION, k=10, mode=mode)
    assert "other-filer" not in {h.accession for h in hits}


def test_filing_on_the_asof_date_is_visible(store):
    hits = store.search(CIK, AS_OF, "covenant waiver lenders", k=10)
    assert any(h.filing_date == AS_OF for h in hits)


def test_freshness_window_excludes_stale_filings(store):
    hits = store.search(CIK, AS_OF, PETITION, k=10)
    assert "ancient" not in {h.accession for h in hits}
    wide = store.search(CIK, AS_OF, PETITION, k=10, window_days=365 * 5)
    assert "ancient" in {h.accession for h in wide}


def test_section_filter(store):
    hits = store.search(CIK, AS_OF, "going concern doubt", k=10, sections=["financials"])
    assert hits and all(h.section == "financials" for h in hits)


def test_relevant_chunk_ranks_first(store):
    hits = store.search(CIK, AS_OF, "substantial doubt going concern", k=3)
    assert "going concern" in hits[0].text


def test_asof_is_required():
    s = FilingStore(open_client("memory"), HashEmbedder())
    with pytest.raises(TypeError):
        s.search(CIK, "2024-07-01", "anything")  # a string is not a date


def test_repeated_text_is_collapsed():
    s = FilingStore(open_client("memory"), HashEmbedder())
    same = "risk factors copied unchanged from quarter to quarter"
    s.add([_c(CIK, date(2024, m, 1), same, m, acc=f"q{m}") for m in (2, 4, 6)])
    hits = s.search(CIK, AS_OF, same, k=5)
    assert len(hits) == 1


def test_has_filing_supports_incremental_ingest(store):
    assert store.has_filing("future-8k")
    assert not store.has_filing("never-ingested")


def test_reingesting_overwrites_rather_than_duplicates(store):
    before = store.client.count(store.collection).count
    store.add([_c(CIK, date(2024, 5, 10), "liquidity is tight and the revolver matures next year", 0)])
    assert store.client.count(store.collection).count == before


def test_backend_returning_a_future_chunk_fails_loudly(store, monkeypatch):
    """Defence in depth: if the backend ever ignores the filter, the run must stop."""
    real = store._filter

    def broken(cik, as_of, window_days, sections):
        return real(cik, date(2030, 1, 1), window_days * 10, sections)

    monkeypatch.setattr(store, "_filter", broken)
    with pytest.raises(LookaheadError):
        store.search(CIK, AS_OF, PETITION, k=10)
