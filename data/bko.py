"""BankruptcyObserver MCP: court-side bankruptcy records.

**Read this before wiring it anywhere near the agent.**

This is *outcome* data. It answers "did this company file, and when" from court
dockets. Handing it to the investigator would not be an enrichment, it would be
telling the model the answer -- the single failure this project treats as
disqualifying. There is no as-of filter that makes a bankruptcy petition safe
to show an assessment dated before it.

So the boundary is structural, not advisory:

**Allowed.** Verifying our labels offline. Our 156 Chapter 11 positives were
derived from EDGAR 8-K item 1.03 filings -- a company telling the SEC it filed.
Court dockets are an independent record of the same event, so agreement is real
corroboration and disagreement is a label bug worth finding. The whole eval
rests on those labels being right and nothing until now has checked them
against a second source.

**Allowed.** Answering a reader's question about what actually happened, on the
path that already exists for that (``agents/outcomes_lookup``), which is
explicitly outside the as-of boundary and says so.

**Never.** The tool list the investigator can call. Not behind a flag, not with
a date filter, not "just for context".

The free tier gives exact/prefix name search with a limited field set -- case
number, court, filing date, chapter, asset/liability bands. That is enough for
verification. Everything richer is paid, and nothing here requires a key.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import date

ENDPOINT = "https://mcp.bankruptcyobserver.com/mcp"

#: The server advertises 30 requests per minute. Staying under it deliberately:
#: a verification sweep is not urgent and being rate-limited mid-run would
#: leave a partial answer that looks like missing data.
MIN_INTERVAL_SECONDS = 2.5

_last_call = 0.0


@dataclass(frozen=True)
class CourtCase:
    """One case as the court records it."""

    name: str
    case_number: str
    court: str
    date_filed: date | None
    chapter: int | None
    assets: str = ""
    liabilities: str = ""


def _call(tool: str, arguments: dict, timeout: int = 25) -> dict:
    """One JSON-RPC tool call, throttled. Raises on transport failure."""
    global _last_call
    wait = MIN_INTERVAL_SECONDS - (time.monotonic() - _last_call)
    if wait > 0:
        time.sleep(wait)
    _last_call = time.monotonic()

    body = json.dumps({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": tool, "arguments": arguments},
    }).encode()
    req = urllib.request.Request(
        ENDPOINT, data=body,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf8"))


def _parse_date(raw: str | None) -> date | None:
    if not raw:
        return None
    try:
        return date.fromisoformat(str(raw)[:10])
    except ValueError:
        return None


def search(name: str, limit: int = 5) -> list[CourtCase]:
    """Court cases matching a debtor name. Empty on any failure.

    Degrades rather than raises: this is a verification aid, and a sweep that
    dies on one bad name is less useful than one that reports the gap.
    """
    try:
        payload = _call("search_bankruptcy_cases_tool",
                        {"search_term": name, "limit": limit})
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return []

    blocks = (payload.get("result") or {}).get("content") or []
    out: list[CourtCase] = []
    for block in blocks:
        try:
            data = json.loads(block.get("text") or "{}")
        except json.JSONDecodeError:
            continue
        for row in data.get("results") or []:
            out.append(CourtCase(
                name=str(row.get("name") or ""),
                case_number=str(row.get("shortCaseNumber") or ""),
                court=str(row.get("court") or ""),
                date_filed=_parse_date(row.get("dateFiled")),
                chapter=row.get("chapter") if isinstance(row.get("chapter"), int) else None,
                assets=str(row.get("assetAmount") or ""),
                liabilities=str(row.get("liabAmount") or ""),
            ))
    return out


def best_match(name: str, near: date | None = None, window_days: int = 120
               ) -> CourtCase | None:
    """The case that best corresponds to a filer, optionally near a date.

    ``near`` matters more than it looks. Large debtors file many affiliated
    petitions -- parent, holding company, a dozen subsidiaries -- and the first
    name match is often a Dutch holding entity that filed Chapter 15 weeks
    apart from the operating company's Chapter 11. Anchoring on the date we
    already believe keeps the comparison honest rather than picking whichever
    affiliate happens to sort first.
    """
    cases = search(name)
    if not cases:
        return None
    if near is None:
        return cases[0]
    dated = [c for c in cases if c.date_filed]
    if not dated:
        return cases[0]
    closest = min(dated, key=lambda c: abs((c.date_filed - near).days))
    return closest if abs((closest.date_filed - near).days) <= window_days else None
