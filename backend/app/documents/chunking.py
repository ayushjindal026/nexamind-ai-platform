"""
Chunking: converts a Document's already-extracted DocumentPage rows into the
units that get embedded and searched — DocumentChunk rows.

Current strategy: one chunk per page (chunk_index=0 for every chunk). This is
a deliberate simplification, not a placeholder pretending to be more than it
is — for the short pages in this assignment's scope, a whole page is a
reasonable retrieval unit, and it means never merging text across a page
break (a page's citation is unambiguous: "Document — Page N").

The schema already anticipates finer-grained chunking: `chunk_index` exists
specifically so a future change here (splitting a long page into multiple
chunks) requires no migration — only a change to this one function. Nothing
elsewhere in the codebase assumes "one chunk per page."
"""

from dataclasses import dataclass

from app.models.document import Document


@dataclass
class ChunkSpec:
    page_number: int
    chunk_index: int
    text: str


def build_chunk_specs(document: Document) -> list[ChunkSpec]:
    """
    Returns one ChunkSpec per page with non-empty text. A blank/image-only
    page (empty extracted text — not an error, see DocumentPage) produces no
    chunk, since an empty chunk has nothing to embed or match against.
    """
    return [
        ChunkSpec(page_number=page.page_number, chunk_index=0, text=page.text)
        for page in document.pages
        if page.text.strip()
    ]
