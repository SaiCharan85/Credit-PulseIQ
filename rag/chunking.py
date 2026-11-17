"""Filing -> chunks a reader can cite.

**Structure first, then size.** HTML is reduced to text *with its paragraph
breaks kept* -- block elements become blank lines -- because the paragraph is
the unit a filing is written in. Then a recursive splitter cuts at the
largest boundary that fits: paragraph, then line, then sentence, then clause,
then word. A chunk only ever breaks mid-sentence when a single sentence is
longer than the whole budget.

**Sized in the embedding model's own tokens.** bge-small reads 512 tokens and
silently truncates the rest, so a chunk measured in words can lose its tail
without anyone noticing. Counting with the model's tokenizer makes the budget
exact. 320 tokens with 64 of overlap (20%): long enough for a full disclosure
paragraph, short enough that one chunk is about one point, and the overlap
means a phrase like "substantial doubt about our ability to continue as a
going concern" that straddles a boundary survives whole on one side.

**Tables are dropped.** A stripped table is a run of figures; the numbers are
the XBRL side's job, and as prose they embed as noise.

**Sanitised at ingestion.** The text was written by the company under
assessment and will be put in front of a model. Neutralising instruction-like
spans once, before storage, means no retrieval path can skip it; the flag
survives into the payload because the attempt is itself a finding.

Each chunk carries what is needed to cite and to filter it, including the
*filing* date -- when it became public, which is what the as-of rule turns on,
never the period it describes.
"""

from __future__ import annotations

import hashlib
import html
import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path

from data.sanitize import sanitize
from rag.sections import KEEP, Section, split_sections
from rag.splitter import RecursiveSplitter

CHUNK_TOKENS = 320
OVERLAP_TOKENS = 64

#: Coarsest boundary first. The splitter only falls through to a finer one when
#: a piece is still over budget.
SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]

#: A line or chunk mostly made of figures is a stripped table.
MAX_NUMERIC_SHARE = 0.4

#: Below this share of the document in kept sections, embed it whole.
MIN_KEPT_SHARE = 0.3

_NAMESPACE = uuid.UUID("6f1d3a52-6b8e-4a51-9c3e-2b7f0f6c1e11")
_NUMERIC = re.compile(r"^[\$\(\)%,.\-—–\d]+$")
_BLOCK = re.compile(r"(?i)</?(?:p|div|br|tr|li|h[1-6]|table|section|ul|ol)\b[^>]*>")
_TAG = re.compile(r"<[^>]+>")


# ---- text ---------------------------------------------------------------


def html_to_text(raw: str) -> str:
    """HTML -> text, keeping paragraph breaks as blank lines."""
    text = re.sub(r"(?is)<(script|style|head)[^>]*>.*?</\1>", " ", raw)
    text = re.sub(r"(?is)<ix:header>.*?</ix:header>", " ", text)  # inline-XBRL metadata block
    text = _BLOCK.sub("\n\n", text)
    text = _TAG.sub(" ", text)
    text = html.unescape(text).replace("\xa0", " ")
    lines = [re.sub(r"[ \t\r\f\v]+", " ", line).strip() for line in text.split("\n")]
    text = "\n".join(line for line in lines if not _is_figures(line))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _is_figures(line: str) -> bool:
    tokens = line.split()
    return len(tokens) >= 2 and sum(1 for t in tokens if _NUMERIC.match(t)) / len(tokens) > 0.5


def is_tabular(text: str) -> bool:
    tokens = text.split()
    if not tokens:
        return True
    return sum(1 for t in tokens if _NUMERIC.match(t)) / len(tokens) > MAX_NUMERIC_SHARE


# ---- token counting -----------------------------------------------------


@lru_cache(maxsize=1)
def _tokenizer():
    """bge-small's tokenizer, from the local model cache, else fetched once."""
    from tokenizers import Tokenizer

    from rag.embed import MODEL_CACHE

    for root in (MODEL_CACHE, Path.home() / "AppData/Local/Temp/fastembed_cache"):
        for path in Path(root).glob("models--*bge-small-en-v1.5*/snapshots/*/tokenizer.json"):
            return Tokenizer.from_file(str(path))
    return Tokenizer.from_pretrained("BAAI/bge-small-en-v1.5")


def bge_tokens(text: str) -> int:
    return len(_tokenizer().encode(text, add_special_tokens=False).ids)


def approx_tokens(text: str) -> int:
    """English averages ~1.3 wordpiece tokens per word. For tests, no download."""
    return int(len(text.split()) * 1.3) + 1


def make_splitter(length: Callable[[str], int] = bge_tokens) -> RecursiveSplitter:
    return RecursiveSplitter(CHUNK_TOKENS, OVERLAP_TOKENS, length, SEPARATORS)


# ---- chunks -------------------------------------------------------------


@dataclass(frozen=True)
class Chunk:
    cik: int
    accession: str
    form: str
    filing_date: date
    section: str
    chunk_no: int
    text: str
    injection: bool = False

    @property
    def id(self) -> str:
        """Stable across runs, so re-ingesting a filing overwrites, never duplicates."""
        return str(uuid.uuid5(_NAMESPACE, f"{self.accession}/{self.chunk_no}"))

    @property
    def text_hash(self) -> str:
        """Identity of the words, for collapsing text repeated filing to filing."""
        norm = re.sub(r"\W+", " ", self.text.lower()).strip()
        return hashlib.sha1(norm.encode("utf8")).hexdigest()[:16]

    def payload(self) -> dict:
        return {
            "cik": self.cik,
            "accession": self.accession,
            "form": self.form,
            "filed": self.filing_date.isoformat(),
            "filed_ord": self.filing_date.toordinal(),
            "section": self.section,
            "chunk_no": self.chunk_no,
            "text": self.text,
            "text_hash": self.text_hash,
            "injection": self.injection,
        }


def chunk_filing(
    cik: int,
    accession: str,
    form: str,
    filing_date: date,
    raw: str,
    splitter=None,
) -> list[Chunk]:
    """Every citable chunk in one filing's primary document."""
    splitter = splitter or make_splitter()
    text = html_to_text(raw)
    sections = [s for s in split_sections(text) if s.name in KEEP]
    # If the headings we recognise cover little of the document, the layout
    # defeated the splitter; embed it whole rather than lose it.
    if text and sum(len(s.text) for s in sections) < MIN_KEPT_SHARE * len(text):
        sections = [Section("full", text)]
    out: list[Chunk] = []
    for section in sections:
        for piece in splitter.split_text(section.text):
            if len(piece.split()) < 12 or is_tabular(piece):
                continue  # headings, page furniture, table residue
            clean = sanitize(piece)
            out.append(
                Chunk(
                    cik=cik, accession=accession, form=form, filing_date=filing_date,
                    section=section.name, chunk_no=len(out), text=clean.text,
                    injection=clean.suspicious,
                )
            )
    return out
