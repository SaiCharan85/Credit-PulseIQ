"""L0: PDF text extraction, and the empty-scan trap.

The test that matters most here is ``test_image_only_pdf_is_not_clean``. Every
phrase pattern fails against the empty string, so an image-only PDF scans as
"no going-concern doubt found" unless something upstream refuses. That is a
false negative our own tooling would have manufactured, on the single signal
that separated bankrupt filers from survivors most sharply in the backtest.
"""

from __future__ import annotations

import io

import pytest

from data import pdftext


def _pdf(pages: list[str]) -> bytes:
    """Build a minimal single-font PDF holding one text run per page.

    Written by hand rather than with a library so the fixture has no dependency
    of its own, and so a reader can see exactly what the extractor is given --
    including the image-only case, which is simply a page with no text run.
    """
    objects: list[bytes] = []

    def add(body: bytes) -> int:
        objects.append(body)
        return len(objects)

    font = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    page_ids: list[int] = []
    content_ids: list[int] = []
    for text in pages:
        if text:
            escaped = text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
            stream = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode()
        else:
            stream = b""  # an image-only page: no text operators at all
        content_ids.append(add(b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream)))

    pages_id = len(objects) + len(pages) + 1
    for content in content_ids:
        page_ids.append(
            add(
                b"<< /Type /Page /Parent %d 0 R /MediaBox [0 0 612 792] "
                b"/Resources << /Font << /F1 %d 0 R >> >> /Contents %d 0 R >>"
                % (pages_id, font, content)
            )
        )
    kids = b" ".join(b"%d 0 R" % p for p in page_ids)
    add(b"<< /Type /Pages /Kids [%s] /Count %d >>" % (kids, len(page_ids)))
    root = add(b"<< /Type /Catalog /Pages %d 0 R >>" % pages_id)

    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets = [0]
    for i, body in enumerate(objects, start=1):
        offsets.append(out.tell())
        out.write(b"%d 0 obj\n%s\nendobj\n" % (i, body))
    start = out.tell()
    out.write(b"xref\n0 %d\n" % (len(objects) + 1))
    out.write(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        out.write(b"%010d 00000 n \n" % off)
    out.write(
        b"trailer\n<< /Size %d /Root %d 0 R >>\nstartxref\n%d\n%%%%EOF\n"
        % (len(objects) + 1, root, start)
    )
    return out.getvalue()


GOING_CONCERN = (
    "These conditions raise substantial doubt about the Company's ability to "
    "continue as a going concern for the twelve months following issuance."
)
FILLER = "Item 7. Management's discussion of results of operations for the period. " * 6


def test_is_pdf_reads_the_bytes_not_the_name():
    assert pdftext.is_pdf(_pdf(["hello"]))
    assert not pdftext.is_pdf(b"<html><body>substantial doubt</body></html>")
    # A file named .pdf that is really HTML must not be treated as a PDF.
    assert not pdftext.is_pdf(b"<!DOCTYPE html>\n<html>report.pdf</html>")


def test_extracts_text_and_counts_pages():
    doc = pdftext.extract(_pdf([FILLER, GOING_CONCERN, FILLER]))
    assert doc.error == ""
    assert doc.n_pages == 3
    assert doc.has_text_layer
    assert "going concern" in doc.text


def test_image_only_pdf_is_not_clean():
    """The trap: no text layer must be refusable, never scanned as negative."""
    doc = pdftext.extract(_pdf(["", "", ""]))
    assert doc.error == ""
    assert doc.n_pages == 3
    assert not doc.has_text_layer, "an image-only PDF must not look scannable"
    assert doc.image_pages == [1, 2, 3]

    # And confirm the failure it prevents is real: the scan itself is happy to
    # report a clean bill of health on the empty string.
    from data.signals import scan_report_text

    assert scan_report_text(doc.text)["going_concern_doubt"] is False


def test_mixed_document_names_the_image_pages():
    doc = pdftext.extract(_pdf([FILLER, "", GOING_CONCERN + FILLER]))
    assert doc.has_text_layer  # the text pages carry it
    assert doc.image_pages == [2]


def test_quote_locates_to_its_page():
    doc = pdftext.extract(_pdf([FILLER, FILLER, GOING_CONCERN]))
    assert doc.locate("substantial doubt about the Company's ability") == 3
    assert doc.locate("a phrase that is nowhere in this document") is None
    assert doc.locate("") is None


def test_passage_spanning_a_page_break_reports_both_pages():
    """A quote carries context either side of the phrase, so it straddles
    breaks. Reporting only where the window opens sends a reader to the page
    before the sentence they came for."""
    doc = pdftext.extract(_pdf([FILLER, GOING_CONCERN]))
    tail = FILLER[-60:]
    assert doc.locate_span(tail + " " + GOING_CONCERN[:60]) == (1, 2)
    # Wholly inside one page, the span collapses to that page.
    assert doc.locate_span(GOING_CONCERN[:40]) == (2, 2)
    assert doc.locate_span("nowhere in this document at all") is None


def test_locate_survives_extraction_line_breaks():
    """A quote broken across a column edge must still place, or the page number
    silently disappears on exactly the long passages worth citing."""
    doc = pdftext.extract(_pdf([FILLER, GOING_CONCERN]))
    broken = GOING_CONCERN.replace(" ", "\n   ", 3)
    assert doc.locate(broken) == 2


def test_corrupt_pdf_reports_rather_than_raises():
    doc = pdftext.extract(b"%PDF-1.4\nthis is not really a pdf at all")
    assert doc.error, "a malformed PDF should come back as a stated error"
    assert not doc.has_text_layer


def test_truncated_pdf_reports_rather_than_raises():
    good = _pdf([FILLER, GOING_CONCERN])
    doc = pdftext.extract(good[: len(good) // 2])
    assert not doc.has_text_layer
    assert doc.error or doc.n_pages == 0


@pytest.mark.parametrize("blob", [b"", b"   ", b"%PDF"])
def test_degenerate_input_is_handled(blob):
    doc = pdftext.extract(blob)
    assert not doc.has_text_layer
    assert doc.n_characters == 0


def test_page_of_covers_every_offset():
    doc = pdftext.extract(_pdf([FILLER, GOING_CONCERN, FILLER]))
    assert doc.page_of(0) == 1
    assert doc.page_of(len(doc.text) - 1) == doc.n_pages
    assert doc.page_of(-1) is None
    assert doc.page_of(len(doc.text)) is None
    seen = {doc.page_of(i) for i in range(len(doc.text))}
    assert seen == {1, 2, 3}
