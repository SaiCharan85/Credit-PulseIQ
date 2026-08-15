"""EDGAR client: XBRL company facts, submissions metadata, filing text.

Deterministic, cached, rate-limited. The client does not interpret anything --
it fetches and normalizes into :class:`~data.facts.XbrlFact`.

SEC requires a declaring User-Agent with contact info on every request
(https://www.sec.gov/os/webmaster-faq#developers) and throttles above ~10 req/s.
Set ``CREDITPULSE_SEC_UA`` e.g. "CreditPulse IQ research you@example.edu".
"""

from __future__ import annotations

import gzip
import json
import os
import time
import urllib.error
import urllib.request
import zlib
from collections.abc import Iterable
from datetime import date, datetime
from pathlib import Path
from typing import Any

from data.facts import XbrlFact

SEC_DATA = "https://data.sec.gov"
SEC_WWW = "https://www.sec.gov"
DEFAULT_CACHE = Path("data/cache")

# SEC asks for <= 10 req/s. We sit well under it; the backtest is not
# latency-sensitive and getting blocked costs far more than the wait.
MIN_INTERVAL_S = 0.15


class EdgarError(RuntimeError):
    pass


def _decode(body: bytes, content_encoding: str | None) -> str:
    """Decode a response, honouring the compression we asked for.

    We advertise gzip because company-facts payloads run to megabytes and the
    backtest sweeps the whole universe. SEC compresses large responses and
    leaves small ones alone, so the encoding must be read from the header
    rather than assumed either way.
    """
    encoding = (content_encoding or "").lower()
    if "gzip" in encoding:
        body = gzip.decompress(body)
    elif "deflate" in encoding:
        body = zlib.decompress(body, -zlib.MAX_WBITS)
    return body.decode("utf8")


def default_user_agent() -> str:
    """User-Agent from the environment. Empty is allowed here and rejected at
    request time, so offline/cached use needs no configuration."""
    return os.environ.get("CREDITPULSE_SEC_UA", "")


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    return datetime.strptime(s, "%Y-%m-%d").date()


class EdgarClient:
    """Cached EDGAR reader.

    Every response is cached on disk, so a backtest sweep re-reads from cache
    and is reproducible offline. ``data/cache/`` is gitignored.
    """

    def __init__(
        self,
        user_agent: str | None = None,
        cache_dir: Path | str = DEFAULT_CACHE,
        min_interval_s: float = MIN_INTERVAL_S,
        offline: bool = False,
    ) -> None:
        self.user_agent = user_agent or os.environ.get("CREDITPULSE_SEC_UA", "")
        self.cache_dir = Path(cache_dir)
        self.min_interval_s = min_interval_s
        self.offline = offline
        self._last_request = 0.0
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ---- transport -----------------------------------------------------

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _require_user_agent(self) -> str:
        if not self.user_agent:
            raise EdgarError(
                "SEC requires a User-Agent with contact info. "
                "Set CREDITPULSE_SEC_UA='CreditPulse IQ research you@example.edu'."
            )
        return self.user_agent

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request
        if elapsed < self.min_interval_s:
            time.sleep(self.min_interval_s - elapsed)
        self._last_request = time.monotonic()

    def _get_json(self, url: str, cache_key: str, retries: int = 3) -> dict[str, Any]:
        path = self._cache_path(cache_key)
        if path.exists():
            return json.loads(path.read_text(encoding="utf8"))
        if self.offline:
            raise EdgarError(f"offline and not cached: {cache_key}")
        self._require_user_agent()
        last: Exception | None = None
        for attempt in range(retries):
            self._throttle()
            try:
                req = urllib.request.Request(
                    url, headers={"User-Agent": self.user_agent, "Accept-Encoding": "gzip, deflate"}
                )
                with urllib.request.urlopen(req, timeout=60) as resp:
                    payload = json.loads(_decode(resp.read(), resp.headers.get("Content-Encoding")))
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(payload), encoding="utf8")
                return payload
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    raise EdgarError(f"not found: {url}") from exc
                last = exc
                time.sleep(1.5 * (attempt + 1))
            except Exception as exc:  # transient network/JSON
                last = exc
                time.sleep(1.5 * (attempt + 1))
        raise EdgarError(f"failed after {retries} attempts: {url}") from last

    # ---- endpoints -----------------------------------------------------

    def submissions(self, cik: int) -> dict[str, Any]:
        """Filer metadata + recent filing index (name, SIC, tickers, forms)."""
        return self._get_json(
            f"{SEC_DATA}/submissions/CIK{cik:010d}.json", f"submissions_{cik:010d}"
        )

    def company_facts(self, cik: int) -> dict[str, Any]:
        """All XBRL facts the filer has ever reported."""
        return self._get_json(
            f"{SEC_DATA}/api/xbrl/companyfacts/CIK{cik:010d}.json", f"companyfacts_{cik:010d}"
        )

    def company_concept(self, cik: int, tag: str, taxonomy: str = "us-gaap") -> dict[str, Any]:
        """All reported values for a single concept. Cheaper than companyfacts."""
        return self._get_json(
            f"{SEC_DATA}/api/xbrl/companyconcept/CIK{cik:010d}/{taxonomy}/{tag}.json",
            f"concept_{cik:010d}_{taxonomy}_{tag}",
        )

    # ---- normalization -------------------------------------------------

    def facts(self, cik: int, tags: Iterable[str] | None = None) -> list[XbrlFact]:
        """Normalize company facts into :class:`XbrlFact` records.

        ``tags`` restricts to specific concepts. Facts without a ``filed`` date
        are dropped -- without it there is no as-of date, and a fact we cannot
        date is a fact we cannot safely use.
        """
        payload = self.company_facts(cik)
        wanted = set(tags) if tags is not None else None
        out: list[XbrlFact] = []
        for taxonomy, concepts in payload.get("facts", {}).items():
            for tag, body in concepts.items():
                if wanted is not None and tag not in wanted:
                    continue
                for unit, entries in body.get("units", {}).items():
                    for e in entries:
                        if not e.get("filed") or e.get("val") is None:
                            continue
                        out.append(
                            XbrlFact(
                                cik=cik,
                                taxonomy=taxonomy,
                                tag=tag,
                                unit=unit,
                                value=float(e["val"]),
                                period_start=_parse_date(e.get("start")),
                                period_end=_parse_date(e["end"]),
                                fy=e.get("fy"),
                                fp=e.get("fp"),
                                form=e.get("form", ""),
                                accession=e.get("accn", ""),
                                filed=_parse_date(e["filed"]),
                                frame=e.get("frame"),
                            )
                        )
        return out

    def facts_from_payload(self, cik: int, payload: dict[str, Any]) -> list[XbrlFact]:
        """Normalize an already-loaded companyfacts payload (used by tests)."""
        saved = self._cache_path(f"companyfacts_{cik:010d}")
        saved.parent.mkdir(parents=True, exist_ok=True)
        saved.write_text(json.dumps(payload), encoding="utf8")
        return self.facts(cik)

    def filing_index(self, cik: int) -> list[dict[str, Any]]:
        """Flatten the submissions filing index, including the older overflow files."""
        sub = self.submissions(cik)
        blocks = [sub["filings"]["recent"]]
        for extra in sub["filings"].get("files", []):
            blocks.append(
                self._get_json(
                    f"{SEC_DATA}/submissions/{extra['name']}",
                    f"submissions_{extra['name'].replace('.json', '')}",
                )
            )
        rows: list[dict[str, Any]] = []
        for block in blocks:
            n = len(block.get("form", []))
            items = block.get("items", [""] * n)
            primary = block.get("primaryDocument", [""] * n)
            for i in range(n):
                rows.append(
                    {
                        "form": block["form"][i],
                        "items": items[i] if i < len(items) else "",
                        "filing_date": _parse_date(block["filingDate"][i]),
                        "accession": block["accessionNumber"][i],
                        "primary_document": primary[i] if i < len(primary) else "",
                    }
                )
        return rows

    # ---- filing documents ----------------------------------------------

    def filing_document_url(self, cik: int, accession: str, primary_document: str) -> str:
        """Canonical archive URL for a filing's primary document."""
        return f"{SEC_WWW}/Archives/edgar/data/{cik}/{accession.replace('-', '')}/{primary_document}"

    def fetch_filing_document(self, cik: int, accession: str, primary_document: str) -> str:
        """Raw filing text, cached.

        Used to read what a filing actually *says* -- structured metadata alone
        is not enough to establish a bankruptcy (see ``data/discover.py``).
        """
        key = f"doc_{accession.replace('-', '')}_{primary_document}".replace("/", "_")
        cached = self.cache_dir / f"{key}.txt"
        if cached.exists():
            return cached.read_text(encoding="utf8", errors="replace")
        if self.offline:
            raise EdgarError(f"offline and no cached document: {key}")
        url = self.filing_document_url(cik, accession, primary_document)
        self._throttle()
        request = urllib.request.Request(url, headers={"User-Agent": self._require_user_agent()})
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                text = response.read().decode("utf8", errors="replace")
        except urllib.error.URLError as exc:
            raise EdgarError(f"fetching {url}: {exc}") from exc
        cached.write_text(text, encoding="utf8")
        return text
