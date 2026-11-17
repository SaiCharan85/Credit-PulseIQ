"""Split a 10-K / 10-Q into the sections credit evidence lives in.

Three facts about real filings drive the design.

**A heading starts a line; a cross-reference does not.** "As discussed in Item
1A. Risk Factors of our annual report" is mid-sentence. With paragraph breaks
preserved (see ``chunking.html_to_text``), requiring a heading to open its
line removes almost every cross-reference at once.

**The table of contents repeats every heading.** Its entries sit a few lines
apart, so a run of headings with short gaps between them is the contents page
and is skipped. Picking "the longest span" instead fails on a common layout
where the financial statements follow the contents page with no heading of
their own: the last contents entry ("Item 6. Exhibits") then appears to own
sixty pages of statements.

**Sections are classified by title, not item number.** A 10-Q's Item 3 is
market risk in Part I and "Defaults Upon Senior Securities" in Part II; a
10-K's Item 3 is legal proceedings. The number is ambiguous and the title is
not.

Text between the contents page and the first body heading is labelled
``financials`` when it holds the notes to the statements -- that is where a
going-concern note lives -- and ``unlabeled`` otherwise. A filing with no
recognisable heading comes back whole as ``full``. Neither is dropped: an
irregular document is still evidence, and irregular filers are exactly the
ones distress produces.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Title pattern -> section name. First match wins; specific titles first.
TITLES: tuple[tuple[str, str], ...] = (
    ("mda", r"management[’'�]?s\s+discussion\s+and\s+analysis"),
    ("mda", r"md\s*&\s*a\b"),
    ("market_risk", r"quantitative\s+and\s+qualitative\s+disclosures?\s+about\s+market\s+risk"),
    ("risk_factors", r"risk\s+factors"),
    ("controls", r"controls\s+and\s+procedures"),
    ("defaults", r"defaults\s+upon\s+senior\s+securities"),
    ("financials", r"exhibits?,?\s+(?:and\s+)?financial\s+statement\s+schedules?"),
    ("financials", r"(?:consolidated\s+)?financial\s+statements"),
    ("market_equity", r"market\s+for\s+(?:the\s+)?registrant"),
    ("legal", r"legal\s+proceedings"),
    ("exhibits", r"exhibits"),
    ("business", r"business"),
)

#: Sections worth embedding. Business descriptions, properties and exhibit
#: lists are bulk without credit signal.
KEEP = frozenset(
    {
        "mda", "liquidity", "risk_factors", "financials", "controls", "market_risk",
        "defaults", "legal", "market_equity", "unlabeled", "full",
    }
)

#: The title is captured in a lookahead so it consumes nothing: contents
#: entries are shorter than the capture, and consuming it would swallow the
#: next entry. It may sit on the next line (its own table cell), hence DOTALL.
_HEADER = re.compile(
    r"^[ \t]*(?:part\s+i{1,3}\s*[,.\-–—]?\s*)?item\s+(\d{1,2}[a-c]?)\s*[.:\-–—]?\s*(?=(.{0,90}))",
    re.IGNORECASE | re.MULTILINE | re.DOTALL,
)

#: Headings closer together than this are contents entries, not sections.
TOC_GAP = 1000
#: A lone heading this early is cover-page noise before the contents page.
COVER_SHARE = 0.05
TOC_MIN_RUN = 3

_LIQUIDITY = re.compile(r"^[ \t]*liquidity\s+and\s+capital\s+resources[ \t]*:?[ \t]*$", re.I | re.M)
_NOTES = re.compile(
    r"notes\s+to\s+(?:the\s+)?(?:unaudited\s+)?(?:condensed\s+)?(?:consolidated\s+)?financial\s+statements",
    re.I,
)


@dataclass(frozen=True)
class Section:
    name: str
    text: str


def classify(title: str) -> str:
    for name, pattern in TITLES:
        if re.match(r"\s*" + pattern, title, re.IGNORECASE):
            return name
    return "other"


def _toc_mask(
    positions: list[int], doc_len: int, keys: list[tuple[str, str]] | None = None
) -> list[bool]:
    """True for headings in the contents page.

    The contents page is the run(s) of closely spaced headings *before the
    first isolated heading*. A cluster later in the document is a set of short
    body sections -- Part II items that each say "None." -- and is kept. A
    lone heading in the first few percent of the document is skipped too, but
    only if a contents run follows it -- otherwise it is the first real heading
    of a document that has no contents page.
    """
    n = len(positions)
    mask = [False] * n
    tentative: list[int] = []
    i = 0
    while i < n:
        j = i
        run_keys = {keys[i]} if keys else set()
        # A heading repeating a section already in the run is the body copy of
        # it, however close it sits to the contents page.
        while (
            j + 1 < n
            and positions[j + 1] - positions[j] < TOC_GAP
            and not (keys and keys[j + 1] in run_keys)
        ):
            j += 1
            if keys:
                run_keys.add(keys[j])
        if j - i + 1 >= TOC_MIN_RUN:
            for k in [*tentative, *range(i, j + 1)]:
                mask[k] = True
            tentative = []
        elif positions[i] <= COVER_SHARE * doc_len:
            tentative.extend(range(i, j + 1))  # cover-page mention, if a contents run follows
        else:
            break  # first isolated heading past the cover: the body has begun
        i = j + 1
    return mask


def split_sections(text: str) -> list[Section]:
    """Sections in document order. ``text`` is plain text with line breaks."""
    if not text.strip():
        return []
    found = [(m.start(), classify(m.group(2)), m.group(1).upper()) for m in _HEADER.finditer(text)]
    keys = [(name, item) if name == "other" else (name, "") for _, name, item in found]
    mask = _toc_mask([p for p, _, _ in found], len(text), keys)
    body = [h for h, toc in zip(found, mask, strict=True) if not toc]
    toc_end = max((p for (p, _, _), toc in zip(found, mask, strict=True) if toc), default=None)
    if not body:
        return [Section("full", text)]

    # One heading per section: the first body occurrence. Later repeats are
    # usually "Item 1A. Risk Factors" lines inside a Part II that says "None".
    starts: list[tuple[int, str]] = []
    seen: set[tuple[str, str]] = set()
    for pos, name, item in body:
        key = (name, item) if name == "other" else (name, "")
        if key in seen:
            continue
        seen.add(key)
        starts.append((pos, name))

    out: list[Section] = []
    # Unheaded text after the contents page (often the statements themselves).
    if toc_end is not None and starts[0][0] > toc_end:
        line_end = text.find("\n", toc_end)
        lead = text[line_end if line_end != -1 else toc_end : starts[0][0]].strip()
        if len(lead) > 1000:
            out.append(Section("financials" if _NOTES.search(lead) else "unlabeled", lead))

    for i, (pos, name) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else len(text)
        chunk = text[pos:end].strip()
        if not chunk:
            continue
        if name == "mda":
            out.extend(_split_liquidity(chunk))
        else:
            out.append(Section(name, chunk))
    return out


def _split_liquidity(mda: str) -> list[Section]:
    """Carve out the liquidity discussion where it has its own heading line."""
    m = _LIQUIDITY.search(mda)
    if not m:
        return [Section("mda", mda)]
    return [Section("mda", mda[: m.start()].strip()), Section("liquidity", mda[m.start() :].strip())]
