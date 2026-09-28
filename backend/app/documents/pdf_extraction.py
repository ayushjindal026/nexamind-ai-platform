"""
PDF extraction, isolated from storage, database, and HTTP concerns.

This module's only job: given raw bytes, either hand back page text or raise
InvalidPDFError with a reason. It knows nothing about organizations,
documents, or requests — that separation is what lets Phase 4's chunking
build on `extract_pages()`'s output without this module changing at all.

Uses `pymupdf` (the modern import name; the older `fitz` alias still works
but is deprecated by the library itself as of the version pinned here).
"""

import pymupdf


class InvalidPDFError(Exception):
    """Raised when the bytes can't be opened as a PDF, or a specific page can't be read."""


def open_pdf(content: bytes) -> pymupdf.Document:
    """
    Raises InvalidPDFError if the bytes aren't a parseable PDF, or the PDF
    has zero pages. Caller is responsible for calling .close() on the
    returned document (a `finally` block in the service layer does this).
    """
    try:
        document = pymupdf.open(stream=content, filetype="pdf")
    except Exception as exc:  # pymupdf raises its own exception types on malformed input
        raise InvalidPDFError("File could not be parsed as a valid PDF") from exc

    if document.page_count == 0:
        document.close()
        raise InvalidPDFError("PDF contains no pages")

    return document


def extract_pages(document: pymupdf.Document) -> list[tuple[int, str]]:
    """
    Returns [(page_number, text), ...], 1-indexed. Raises InvalidPDFError if
    a specific page's content stream can't be read — this is a real, distinct
    failure mode from open_pdf() failing (the file opens fine, page count is
    fine, but one page is corrupt).
    """
    pages: list[tuple[int, str]] = []
    for zero_indexed_page_number in range(document.page_count):
        page_number = zero_indexed_page_number + 1
        try:
            page = document.load_page(zero_indexed_page_number)
            text = page.get_text()
        except Exception as exc:
            raise InvalidPDFError(f"Failed to extract text from page {page_number}") from exc
        pages.append((page_number, text))
    return pages
