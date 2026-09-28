import uuid

from pydantic import BaseModel, Field, field_validator


class SearchRequest(BaseModel):
    query: str = Field(max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    min_similarity: float | None = Field(default=None, ge=-1.0, le=1.0)

    @field_validator("query")
    @classmethod
    def _query_must_not_be_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("query must not be empty")
        return stripped


class SearchResultItem(BaseModel):
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    document_name: str
    page_number: int
    text: str
    similarity: float
    # Ready-made citation string, e.g. "Scholarship_Rules.pdf — Page 4",
    # so later phases attach citations without reformatting.
    citation: str


class SearchResponse(BaseModel):
    results: list[SearchResultItem]
