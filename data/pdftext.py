"""Text extraction from PDF filings, with the empty case treated as a finding.

Companies file with the SEC in HTML, but the copy a person actually has on
their desk is usually a PDF -- an investor deck, a printed 10-K, a lender's
scan of a credit agreement. This module gets text out of those so the same
deterministic phrase scan can run over them.

One decision carries the module. **A PDF with no text layer must not scan
clean.** A scanned image of a page containing "substantial doubt about the
Company's ability to continue as a going concern" extracts as the empty
string, and an empty string matches no pattern, so a naive reader reports
"going-concern doubt: not found" on a document that says the opposite in
72-point type. That is a false negative manufactured by our own tooling, on
precisely the signal that mattered most in the backtest. So extraction
reports whether a usable text layer was present, and the caller is expected
to refuse rather than to scan.

Page numbers travel with the text. A quote that a reader cannot find in the
document is only marginally better than no quote, and "page 47" is the
difference between a citation and an assertion.
"""

from __future__ import annotations

import io
import re
from bisect import bisect_right
from dataclasses import dataclass, field

#: Below this many characters across the whole document we assume the pages are
#: images. Real filings run to hundreds of thousands of characters; a handful
#: of stray characters is what a scan yields from page furniture and stamps.
MIN_TEXT_LAYER = 200

#: Per-page floor for the same judgement, used to report how many pages of a
#: mixed document are image-only. A cover page legitimately holds very little.
MIN_PAGE_CHARS = 40

PDF_MAGIC = b"%PDF"


@dataclass
class PdfText:
    """Extracted text, plus what could not be extracted."""

    text: str = ""
    pages: list[str] = field(default_factory=list)
    n_pages: int = 0
    #: Pages that yielded almost nothing -- image-only in an otherwise text PDF.
    image_pages: list[int] = field(default_factory=list)
    encrypted: bool = False
    error: str = ""

    @property
    def has_text_layer(self) -> bool:
        """Whether enough text came out to scan honestly."""
        return len(self.text.strip()) >= MIN_TEXT_LAYER

    @property
    def n_characters(self) -> int:
        return len(self.text)

    def page_of(self, offset: int) -> int | None:
        """1-based page holding the character at ``offset`` in ``text``."""
        if not self._starts or offset < 0 or offset >= len(self.text):
            return None
        return bisect_right(self._starts, offset)

    def locate_span(self, needle: str) -> tuple[int, int] | None:
        """1-based (first, last) pages a passage covers, or None if unplaceable.

        A quote is a window with context either side of the matched phrase, so
        it routinely straddles a page break. Reporting only where the window
        opens sends a reader to the page *before* the sentence they came for --
        which is why this returns the span and the caller renders "pages 2-3".

        Whitespace is normalised on both sides: extraction inserts line breaks
        where the PDF had a column edge, so an exact match would fail on text
        that is plainly present.
        """
        if not needle.strip():
            return None
        flat, index = _flatten(self.text)
        target = _squeeze(needle)
        at = flat.find(target)
        if at < 0:
            return None
        first = self.page_of(index[at])
        last = self.page_of(index[at + len(target) - 1])
        if first is None or last is None:
            return None
        return first, last

    def locate(self, needle: str) -> int | None:
        """1-based page a passage starts on, or None if it cannot be placed."""
        span = self.locate_span(needle)
        return span[0] if span else None

    @property
    def _starts(self) -> list[int]:
        out, run = [], 0
        for p in self.pages:
            out.append(run)
            run += len(p) + 1  # the newline join() adds between pages
        return out


def is_pdf(data: bytes) -> bool:
    """Whether these bytes are a PDF, by magic number rather than filename.

    An uploaded file's extension is a claim by whoever named it. The first
    four bytes are the document itself.
    """
    return data[:1024].lstrip()[:4] == PDF_MAGIC


def extract(data: bytes) -> PdfText:
    """Pull text out of a PDF, page by page.

    Never raises for a malformed document: a corrupt PDF is a thing users hand
    us, and it should come back as a stated error rather than a 500.
    """
    try:
        from pypdf import PdfReader
    except ImportError:  # pragma: no cover - dependency is pinned
        return PdfText(error="PDF support requires pypdf")

    try:
        reader = PdfReader(io.BytesIO(data))
    except Exception as exc:  # noqa: BLE001 - any parse failure is the same to a caller
        return PdfText(error=f"could not read the PDF: {exc}")

    out = PdfText(encrypted=bool(getattr(reader, "is_encrypted", False)))
    if out.encrypted:
        # An owner password with an empty user password is common on filings
        # and decrypts silently; a real user password does not.
        try:
            reader.decrypt("")
        except Exception:  # noqa: BLE001
            out.error = "the PDF is password-protected"
            return out

    try:
        pages = list(reader.pages)
    except Exception as exc:  # noqa: BLE001
        out.error = f"could not read the PDF: {exc}"
        return out

    for i, page in enumerate(pages, start=1):
        try:
            body = page.extract_text() or ""
        except Exception:  # noqa: BLE001 - one bad page should not lose the rest
            body = ""
        out.pages.append(body)
        if len(body.strip()) < MIN_PAGE_CHARS:
            out.image_pages.append(i)

    out.n_pages = len(out.pages)
    out.text = "\n".join(out.pages)
    return out


def _squeeze(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _flatten(s: str) -> tuple[str, list[int]]:
    """Whitespace-normalised text, plus each character's offset in the original."""
    out: list[str] = []
    index: list[int] = []
    prev_space = True  # leading whitespace is dropped, as in _squeeze
    for i, ch in enumerate(s):
        if ch.isspace():
            if prev_space:
                continue
            out.append(" ")
            index.append(i)
            prev_space = True
        else:
            out.append(ch)
            index.append(i)
            prev_space = False
    while out and out[-1] == " ":
        out.pop()
        index.pop()
    return "".join(out), index
