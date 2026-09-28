import uuid
from datetime import datetime

from pydantic import BaseModel, Field, field_validator
from app.decisions.schemas import DecisionResult


class AssistantStatusResponse(BaseModel):
    configured: bool
    assistant_id: uuid.UUID | None = None
    created_at: datetime | None = None


class AssistantTokenResponse(BaseModel):
    assistant_id: uuid.UUID
    assistant_token: str
    assistant_url: str
    embed_code: str


class AssistantUsageResponse(BaseModel):
    questions_asked: int
    questions_answered: int
    questions_unavailable: int


class AskRequest(BaseModel):
    assistant_token: str | None = Field(default=None, max_length=200)
    question: str = Field(max_length=2000)

    @field_validator("question")
    @classmethod
    def _question_must_not_be_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("question must not be empty")
        return stripped


class AnswerCitation(BaseModel):
    document_id: uuid.UUID
    document_name: str
    page_number: int
    citation: str


class AskResponse(BaseModel):
    answer: str
    citations: list[AnswerCitation]
    decisions: list[DecisionResult] = Field(default_factory=list)
    decision_outcome: str | None = None
