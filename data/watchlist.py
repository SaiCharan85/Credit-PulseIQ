"""The monitored universe.

The watchlist is *data* (``data/watchlist.csv``), not code, so Phase 3 can scale
it from ~50 names to a few hundred (SPEC 2) without touching logic.

Only the CIK and the cohort are stored. Everything else about a company --
name, ticker, SIC, size -- is resolved from EDGAR at load time, because those
are point-in-time attributes that drift: tickers get reused after bankruptcy
(BBBY's ticker now belongs to Overstock), and filers rename themselves into
shells ("Old COPPER Company" was J.C. Penney). Freezing them into a file would
bake in stale identity.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel, ConfigDict, field_validator

WATCHLIST_CSV = Path(__file__).parent / "watchlist.csv"

COHORT_DISTRESS = "distress"
COHORT_SURVIVOR = "survivor"


class WatchlistEntry(BaseModel):
    """A monitored filer.

    ``cohort`` records why the name is in the universe, not an outcome. The
    outcome lives in ``data/labels/``; a survivor is simply a name with no
    event on record, which is a censored observation rather than a proven
    negative (see SPEC 13, class imbalance).
    """

    model_config = ConfigDict(frozen=True)

    cik: int
    name: str
    cohort: str
    sector_hint: str = ""

    @field_validator("cik", mode="before")
    @classmethod
    def _coerce_cik(cls, v: object) -> int:
        return int(str(v).strip().lstrip("0") or 0)


def load_watchlist(path: Path | str = WATCHLIST_CSV) -> list[WatchlistEntry]:
    rows: list[WatchlistEntry] = []
    with Path(path).open(encoding="utf8", newline="") as fh:
        for row in csv.DictReader(fh):
            if not (row.get("cik") or "").strip():
                continue
            rows.append(
                WatchlistEntry(
                    cik=row["cik"],
                    name=row.get("name", ""),
                    cohort=row.get("cohort", ""),
                    sector_hint=row.get("sector_hint", ""),
                )
            )
    return rows


def by_cohort(entries: Iterable[WatchlistEntry], cohort: str) -> list[WatchlistEntry]:
    return [e for e in entries if e.cohort == cohort]


def watchlist_ciks(entries: Iterable[WatchlistEntry] | None = None) -> list[int]:
    return [e.cik for e in (entries if entries is not None else load_watchlist())]
