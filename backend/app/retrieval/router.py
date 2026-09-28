"""
Retrieval endpoint: POST /api/retrieval/search.

This is the retrieval *building block*, not the assistant: it embeds the
query, runs the organization-scoped similarity search, and returns matching
chunks with citation info. No classification, no decision-making, no answer
generation — /api/ask (a later phase) composes this same service function
with those steps.

Authenticated dashboard route: organization_id comes from
get_current_membership, never from the request.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_membership
from app.db.session import get_db
from app.embeddings.provider import (
    EMBEDDING_DIMENSION,
    EmbeddingProvider,
    EmbeddingProviderError,
    get_embedding_provider,
)
from app.models.membership import Membership
from app.retrieval.service import search_similar_chunks
from app.schemas.retrieval import SearchRequest, SearchResponse, SearchResultItem

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/retrieval", tags=["retrieval"])


@router.post("/search", response_model=SearchResponse)
def search(
    payload: SearchRequest,
    membership: Membership = Depends(get_current_membership),
    db: Session = Depends(get_db),
    embedding_provider: EmbeddingProvider | None = Depends(get_embedding_provider),
) -> SearchResponse:
    if embedding_provider is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedding provider is not configured",
        )

    try:
        vectors = embedding_provider.embed([payload.query])
    except EmbeddingProviderError:
        logger.exception("Query embedding failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedding service is temporarily unavailable",
        )

    if len(vectors) != 1 or len(vectors[0]) != EMBEDDING_DIMENSION:
        logger.error("Embedding provider returned malformed output for a query")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedding service is temporarily unavailable",
        )

    try:
        results = search_similar_chunks(
            db,
            membership.organization_id,
            vectors[0],
            top_k=payload.top_k,
            min_similarity=payload.min_similarity,
        )
    except SQLAlchemyError:
        logger.exception("Similarity search failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Search failed",
        )

    return SearchResponse(
        results=[
            SearchResultItem(
                chunk_id=result.chunk_id,
                document_id=result.document_id,
                document_name=result.document_name,
                page_number=result.page_number,
                text=result.text,
                similarity=result.similarity,
                citation=f"{result.document_name} — Page {result.page_number}",
            )
            for result in results
        ]
    )
