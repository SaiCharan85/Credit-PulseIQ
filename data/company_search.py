"""Find a filer by name or ticker, so nobody has to know a CIK.

The UI required a ten-digit CIK. Nobody knows CIK 867773; they know "the
packaging company" or "AAPL". Requiring the identifier made the tool unusable
for the person it is for.

Resolution is deliberately **not** fuzzy-matched to a single answer. Company
names collide -- "Apple Inc." and "Apple Hospitality REIT" both match "apple",
and quietly picking the larger one is the kind of silent wrong answer this
project spends its effort avoiding. Ambiguous queries return candidates and
let a human choose.

Ranking is by how the match was made rather than by string distance: an exact
ticker beats an exact name, which beats a prefix, which beats a substring. That
ordering is stable and explainable, where an edit-distance score is neither.
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

#: Match quality, best first. Also the sort order.
EXACT_TICKER = 0
EXACT_NAME = 1
NAME_PREFIX = 2
NAME_CONTAINS = 3
TICKER_PREFIX = 4

_LABEL = {
    EXACT_TICKER: "exact ticker",
    EXACT_NAME: "exact name",
    NAME_PREFIX: "name starts with",
    NAME_CONTAINS: "name contains",
    TICKER_PREFIX: "ticker starts with",
}

#: A one- or two-character query prefix-matches a large share of the
#: directory, which is noise rather than a result.
MIN_PREFIX = 3
MIN_SUBSTRING = 4

#: Suffixes that add nothing to a match and break prefix comparisons.
_NOISE = (
    " inc.", " inc", " corp.", " corp", " corporation", " co.", " company",
    " ltd.", " ltd", " llc", " plc", " l.p.", " lp", " holdings", " group",
    " the", ", the", " & co",
)


@dataclass(frozen=True)
class Match:
    cik: int
    name: str
    ticker: str
    how: int

    @property
    def reason(self) -> str:
        return _LABEL[self.how]


def _normalise(text: str) -> str:
    out = " ".join(text.lower().replace(",", " ").split())
    changed = True
    while changed:
        changed = False
        for suffix in _NOISE:
            if out.endswith(suffix):
                out = out[: -len(suffix)].strip()
                changed = True
    return out


def load_directory(
    cache_dir: Path | str = "data/cache", fetch=None
) -> list[dict]:
    """The SEC's ticker-to-CIK directory, cached on disk."""
    path = Path(cache_dir) / "company_tickers.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        text = path.read_text(encoding="utf8")
    elif fetch is not None:
        text = fetch(TICKERS_URL)
        path.write_text(text, encoding="utf8")
    else:
        # SEC rejects requests without a contact address (403). The same
        # declared user agent the EDGAR client uses.
        import os

        agent = os.environ.get("CREDITPULSE_SEC_UA", "").strip()
        if not agent:
            raise RuntimeError(
                "CREDITPULSE_SEC_UA is not set; SEC requires a contact address"
            )
        request = urllib.request.Request(TICKERS_URL, headers={"User-Agent": agent})
        with urllib.request.urlopen(request, timeout=45) as response:
            text = response.read().decode("utf8", errors="replace")
        path.write_text(text, encoding="utf8")
    payload = json.loads(text)
    rows = payload.values() if isinstance(payload, dict) else payload
    return [
        {
            "cik": int(r["cik_str"]),
            "name": str(r.get("title", "")),
            "ticker": str(r.get("ticker", "")).upper(),
        }
        for r in rows
    ]


def search(query: str, directory: list[dict], limit: int = 8) -> list[Match]:
    """Candidate filers for a name or ticker, best match first.

    Returns a list even when one match is obvious. The caller decides whether
    a single confident hit may be auto-selected; this function will not make
    that choice silently.
    """
    q = query.strip()
    if not q:
        return []
    qn, qu = _normalise(q), q.upper().strip()
    found: dict[int, Match] = {}

    for row in directory:
        name_n = _normalise(row["name"])
        ticker = row["ticker"]
        how: int | None = None
        if ticker and ticker == qu:
            how = EXACT_TICKER
        elif name_n == qn:
            how = EXACT_NAME
        elif len(qn) >= MIN_PREFIX and name_n.startswith(qn):
            how = NAME_PREFIX
        elif len(qn) >= MIN_SUBSTRING and qn in name_n:
            how = NAME_CONTAINS
        elif ticker and len(qu) >= 2 and ticker.startswith(qu):
            how = TICKER_PREFIX
        if how is None:
            continue
        existing = found.get(row["cik"])
        if existing is None or how < existing.how:
            found[row["cik"]] = Match(row["cik"], row["name"], ticker, how)

    return sorted(found.values(), key=lambda m: (m.how, len(m.name), m.name))[:limit]


def resolve(query: str, directory: list[dict]) -> tuple[Match | None, list[Match]]:
    """Return ``(unambiguous_match, candidates)``.

    A match is unambiguous only on an exact ticker or an exact name, or when
    a single candidate exists at all. "apple" hits both Apple Inc. and Apple
    Hospitality REIT, so it resolves to nothing and returns both -- choosing
    the bigger company would be a guess wearing the clothes of an answer.
    """
    hits = search(query, directory)
    if not hits:
        return None, []
    if hits[0].how in (EXACT_TICKER, EXACT_NAME) or len(hits) == 1:
        return hits[0], hits
    return None, hits
