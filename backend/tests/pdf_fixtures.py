"""
Builds PDF bytes at test time using PyMuPDF itself — no binary fixture files
committed to the repo.
"""

import pymupdf


def make_pdf_bytes(page_texts: list[str] | None = None) -> bytes:
    """A valid, openable PDF with one page per string in page_texts (default: one page)."""
    if page_texts is None:
        page_texts = ["Test document content."]

    document = pymupdf.open()
    for text in page_texts:
        page = document.new_page()
        page.insert_text((72, 72), text)
    content = document.tobytes()
    document.close()
    return content


def make_corrupt_pdf_bytes() -> bytes:
    """Passes the '%PDF-' magic-byte check but is not actually parseable."""
    return b"%PDF-1.7\n%garbage garbage garbage, not a real xref table\nnot a pdf" * 10
