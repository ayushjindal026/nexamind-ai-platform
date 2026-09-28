"""Compose Phase 4 retrieval with Phase 6 grounded LLM answer generation."""

import uuid
from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.core.config import settings
from app.decisions.engine import DecisionEngineError, SystemOneProvider
from app.decisions.schemas import DecisionEvaluation
from app.decisions.service import evaluate_plan
from app.embeddings.provider import EMBEDDING_DIMENSION, EmbeddingProvider, EmbeddingProviderError
from app.llm.provider import LLMProvider
from app.retrieval.service import ChunkSearchResult, search_similar_chunks

NO_INFORMATION_ANSWER = "That information is not available in this organization's knowledge base."


@dataclass
class AnswerResult:
    answer: str
    citations: list[ChunkSearchResult]
    decision_evaluation: DecisionEvaluation | None = None


def answer_question(
    db: Session,
    organization_id: uuid.UUID,
    question: str,
    embedding_provider: EmbeddingProvider,
    llm_provider: LLMProvider,
    decision_engine: SystemOneProvider | None = None,
) -> AnswerResult:
    vectors = embedding_provider.embed([question])
    if len(vectors) != 1 or len(vectors[0]) != EMBEDDING_DIMENSION:
        raise EmbeddingProviderError("Embedding provider returned malformed output")

    matches = search_similar_chunks(
        db,
        organization_id,
        vectors[0],
        top_k=5,
        min_similarity=settings.answer_min_similarity,
    )
    if not matches:
        return AnswerResult(answer=NO_INFORMATION_ANSWER, citations=[])

    context = "\n\n".join(
        f"[{match.document_name} — Page {match.page_number}]\n{match.text}" for match in matches
    )
    decision_evaluation = None
    planner = getattr(llm_provider, "plan_decisions", None)
    if planner is not None:
        plan = planner(question, context)
        if plan.requires_decisions:
            if decision_engine is None:
                raise DecisionEngineError("Laya decision engine is not configured")
            decision_evaluation = evaluate_plan(context, plan, decision_engine)
            answer = llm_provider.generate_answer(
                question, context, decision_summary=decision_evaluation.summary()
            )
        else:
            answer = llm_provider.generate_answer(question, context)
    else:
        # Backward-compatible path for existing provider implementations.
        answer = llm_provider.generate_answer(question, context)

    # Return one source entry per document/page, even if multiple sub-page
    # chunks from that page were retrieved.
    citations: list[ChunkSearchResult] = []
    seen: set[tuple[uuid.UUID, int]] = set()
    for match in matches:
        key = (match.document_id, match.page_number)
        if key not in seen:
            seen.add(key)
            citations.append(match)
    return AnswerResult(answer=answer, citations=citations, decision_evaluation=decision_evaluation)
