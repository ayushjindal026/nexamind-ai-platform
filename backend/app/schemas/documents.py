import uuid
from datetime import datetime

from pydantic import BaseModel


class DocumentOut(BaseModel):
    id: uuid.UUID
    filename: str
    content_type: str
    file_size_bytes: int
    page_count: int
    status: str
    failure_reason: str | None
    created_at: datetime
    chunk_count: int = 0
    embedded_chunk_count: int = 0
