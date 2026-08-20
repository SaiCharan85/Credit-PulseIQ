"""Point-in-time filer universe from EDGAR's quarterly form indexes.

The earnings-quality leg was first built over the distress watchlist -- 354
companies selected because they went bankrupt or neighbour someone who did.
Restatements in that population do not resemble restatements generally, so any
number measured there describes a corner of the market rather than the market.

Widening it correctly is the whole difficulty, because the obvious way is
wrong. Taking today's ticker list and walking backwards silently drops every
company that delisted, merged or failed in between, which is exactly the
survivorship bias this project already fought once when pinning CIKs. A
universe assembled that way makes any model look good: the hardest names are
missing.

So the universe is built from ``full-index/YYYY/QTRn/form.idx`` -- EDGAR's
record of *every filing made in that quarter, as it was made*. A company that
filed a 10-K in 2019 and was delisted in 2021 appears in the 2019 index and
must appear in the 2019 universe. The index is immutable and contemporaneous,
which is the property that matters.

Membership is by *having filed a 10-K in the window*, not by being listed
today. That is the point-in-time definition, and it admits exactly the filers a
reader could have screened at the time.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from pathlib import Path

FULL_INDEX = "https://www.sec.gov/Archives/edgar/full-index"

#: Annual reports. 10-K variants only: a filer that has never filed one has no
#: audited annual statements to assess.
ANNUAL_FORMS = ("10-K", "10-K/A", "10-K405", "10-KSB", "20-F")

#: form.idx is fixed-width, but the column widths shift between years. Splitting
#: on runs of two-or-more spaces is stable where slicing by position is not.
_ROW = re.compile(
    r"^(\S[^ ]*(?: [^ ]+)*?)\s{2,}(.+?)\s{2,}(\d{1,10})\s{2,}"
    r"(\d{4}-\d{2}-\d{2})\s{2,}(\S+)\s*$"
)


@dataclass(frozen=True)
class IndexEntry:
    form: str
    company: str
    cik: int
    filed: date


def quarters(start: date, end: date) -> list[tuple[int, int]]:
    """(year, quarter) pairs covering the range inclusive."""
    out: list[tuple[int, int]] = []
    y, q = start.year, (start.month - 1) // 3 + 1
    ey, eq = end.year, (end.month - 1) // 3 + 1
    while (y, q) <= (ey, eq):
        out.append((y, q))
        q += 1
        if q > 4:
            y, q = y + 1, 1
    return out


def parse_form_index(text: str, forms: Iterable[str] = ANNUAL_FORMS) -> list[IndexEntry]:
    """Rows of one quarterly form.idx, filtered to ``forms``."""
    wanted = set(forms)
    out: list[IndexEntry] = []
    for line in text.splitlines():
        found = _ROW.match(line)
        if not found:
            continue
        form, company, cik, filed, _path = found.groups()
        if form not in wanted:
            continue
        try:
            out.append(
                IndexEntry(
                    form=form,
                    company=company.strip(),
                    cik=int(cik),
                    filed=date.fromisoformat(filed),
                )
            )
        except ValueError:
            continue
    return out


def fetch_form_index(year: int, quarter: int, fetch, cache_dir: Path | None = None) -> str:
    """One quarter's form.idx, cached on disk.

    These files are immutable once the quarter closes, so a cached copy is the
    same object forever -- unlike ``submissions``, which is rewritten as new
    filings land.
    """
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / f"form_{year}_QTR{quarter}.idx"
        if path.exists():
            return path.read_text(encoding="utf8", errors="replace")
    text = fetch(f"{FULL_INDEX}/{year}/QTR{quarter}/form.idx")
    if cache_dir is not None:
        (cache_dir / f"form_{year}_QTR{quarter}.idx").write_text(text, encoding="utf8")
    return text


def annual_filers(
    start: date,
    end: date,
    fetch,
    cache_dir: Path | None = None,
    on_progress=None,
) -> dict[int, IndexEntry]:
    """Every CIK that filed an annual report in the window, earliest first.

    The returned mapping *is* the point-in-time universe: membership was earned
    by filing, contemporaneously, and cannot be revoked by a later delisting.
    """
    found: dict[int, IndexEntry] = {}
    qs = quarters(start, end)
    for n, (year, quarter) in enumerate(qs, 1):
        try:
            text = fetch_form_index(year, quarter, fetch, cache_dir)
        except Exception:  # noqa: BLE001 - one missing quarter must not void the rest
            if on_progress:
                on_progress(n, len(qs), year, quarter, -1)
            continue
        entries = parse_form_index(text)
        for e in entries:
            if e.filed < start or e.filed > end:
                continue
            if e.cik not in found or e.filed < found[e.cik].filed:
                found[e.cik] = e
        if on_progress:
            on_progress(n, len(qs), year, quarter, len(entries))
    return found


def sample_universe(
    filers: dict[int, IndexEntry],
    keep: Iterable[int],
    n_sample: int,
    seed: int = 0,
) -> list[int]:
    """All of ``keep``, plus a random sample of the rest.

    Positives are scarce and every one is expensive to have found, so they are
    all retained while the negative pool is thinned to what the fetch budget
    allows. That makes this a choice-based sample (Zmijewski 1984): the base
    rate is inflated by construction, and precision must be corrected for the
    sampling fraction downstream. Recall and ranking are unaffected.
    """
    import random

    keep = {int(c) for c in keep}
    pool = sorted(set(filers) - keep)
    rng = random.Random(seed)
    take = pool if n_sample <= 0 or n_sample >= len(pool) else rng.sample(pool, n_sample)
    return sorted(keep | set(take))
