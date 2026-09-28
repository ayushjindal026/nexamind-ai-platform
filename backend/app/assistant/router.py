"""Authenticated assistant management and public visitor question API."""

import json
import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import HTMLResponse
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.assistant.service import (
    AssistantAlreadyConfiguredError,
    AssistantNotConfiguredError,
    create_assistant,
    get_assistant_for_organization,
    get_assistant_for_token,
    regenerate_assistant_token,
)
from app.assistant.answering import answer_question
from app.decisions.engine import DecisionEngineError, SystemOneProvider, get_decision_engine
from app.auth.dependencies import get_current_membership
from app.db.session import get_db
from app.embeddings.provider import (
    EmbeddingProvider,
    EmbeddingProviderError,
    get_embedding_provider,
)
from app.llm.provider import LLMProvider, LLMProviderError, get_llm_provider
from app.models.membership import Membership
from app.schemas.assistant import (
    AnswerCitation,
    AskRequest,
    AskResponse,
    AssistantStatusResponse,
    AssistantTokenResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["assistant"])
management = APIRouter(prefix="/api/assistant")


def _embed_code(assistant_url: str) -> str:
    # The iframe keeps visitor traffic same-origin with the API, so the embed
    # works without broadening CORS or asking organizations to expose a host.
    return (
        f'<iframe src="{assistant_url}" title="Organization assistant" '
        'width="100%" height="560" style="border:0" loading="lazy"></iframe>'
    )


def _token_response(request: Request, assistant, token: str) -> AssistantTokenResponse:
    assistant_url = str(request.base_url).rstrip("/") + "/assistant?assistant_token=" + token
    return AssistantTokenResponse(
        assistant_id=assistant.id,
        assistant_token=token,
        assistant_url=assistant_url,
        embed_code=_embed_code(assistant_url),
    )


@management.get("", response_model=AssistantStatusResponse)
def get_assistant_status(
    membership: Membership = Depends(get_current_membership), db: Session = Depends(get_db)
) -> AssistantStatusResponse:
    assistant = get_assistant_for_organization(db, membership.organization_id)
    return AssistantStatusResponse(
        configured=assistant is not None,
        assistant_id=assistant.id if assistant else None,
        created_at=assistant.created_at if assistant else None,
    )


@management.post("", response_model=AssistantTokenResponse, status_code=status.HTTP_201_CREATED)
def create_assistant_endpoint(
    request: Request,
    membership: Membership = Depends(get_current_membership),
    db: Session = Depends(get_db),
) -> AssistantTokenResponse:
    try:
        assistant, token = create_assistant(db, membership.organization_id)
    except AssistantAlreadyConfiguredError:
        raise HTTPException(status_code=409, detail="Assistant is already configured")
    except SQLAlchemyError:
        db.rollback()
        logger.exception("Assistant creation failed")
        raise HTTPException(status_code=500, detail="Failed to create assistant")
    return _token_response(request, assistant, token)


@management.post("/regenerate", response_model=AssistantTokenResponse)
def regenerate_assistant_endpoint(
    request: Request,
    membership: Membership = Depends(get_current_membership),
    db: Session = Depends(get_db),
) -> AssistantTokenResponse:
    try:
        assistant, token = regenerate_assistant_token(db, membership.organization_id)
    except AssistantNotConfiguredError:
        raise HTTPException(status_code=404, detail="Assistant is not configured")
    except SQLAlchemyError:
        db.rollback()
        logger.exception("Assistant token regeneration failed")
        raise HTTPException(status_code=500, detail="Failed to regenerate assistant token")
    return _token_response(request, assistant, token)


@router.post("/api/ask", response_model=AskResponse)
def ask(
    payload: AskRequest,
    assistant_token_query: str | None = Query(default=None, alias="assistant_token", max_length=200),
    db: Session = Depends(get_db),
    embedding_provider: EmbeddingProvider | None = Depends(get_embedding_provider),
    llm_provider: LLMProvider | None = Depends(get_llm_provider),
    decision_engine: SystemOneProvider | None = Depends(get_decision_engine),
) -> AskResponse:
    token = payload.assistant_token or assistant_token_query
    assistant = get_assistant_for_token(db, token or "")
    if assistant is None:
        raise HTTPException(status_code=401, detail="Invalid assistant token")
    if embedding_provider is None:
        raise HTTPException(status_code=503, detail="Embedding provider is not configured")
    if llm_provider is None:
        raise HTTPException(status_code=503, detail="Language model provider is not configured")

    try:
        result = answer_question(
            db=db,
            organization_id=assistant.organization_id,
            question=payload.question,
            embedding_provider=embedding_provider,
            llm_provider=llm_provider,
            decision_engine=decision_engine,
        )
    except EmbeddingProviderError:
        logger.exception("Question embedding failed")
        raise HTTPException(status_code=503, detail="Assistant is temporarily unavailable")
    except LLMProviderError:
        logger.exception("Answer generation failed")
        raise HTTPException(status_code=503, detail="Assistant is temporarily unavailable")
    except DecisionEngineError:
        logger.exception("Structured decision evaluation failed or is unavailable")
        raise HTTPException(status_code=503, detail="Structured decision service is temporarily unavailable")
    except SQLAlchemyError:
        logger.exception("Assistant retrieval failed")
        raise HTTPException(status_code=500, detail="Assistant request failed")

    return AskResponse(
        answer=result.answer,
        citations=[
            AnswerCitation(
                document_id=item.document_id,
                document_name=item.document_name,
                page_number=item.page_number,
                citation=f"{item.document_name} — Page {item.page_number}",
            )
            for item in result.citations
        ],
        decisions=result.decision_evaluation.decisions if result.decision_evaluation else [],
        decision_outcome=result.decision_evaluation.outcome if result.decision_evaluation else None,
    )


@router.get("/assistant", response_class=HTMLResponse, include_in_schema=False)
def visitor_page(
    assistant_token: str = Query(min_length=1, max_length=200),
    db: Session = Depends(get_db),
) -> HTMLResponse:
    """Small standalone visitor surface so the generated public URL opens directly."""
    if get_assistant_for_token(db, assistant_token) is None:
        raise HTTPException(status_code=401, detail="Invalid assistant token")
    token_js = json.dumps(assistant_token).replace("<", "\\u003c")
    html = f'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Organization Assistant</title>
<main style="font:16px system-ui;max-width:42rem;margin:3rem auto;padding:0 1rem">
<h1>Ask the organization assistant</h1>
<form><label for="question">Your question</label><br><textarea id="question" rows="3" required maxlength="2000" style="width:100%;margin:.5rem 0"></textarea><br><button>Ask</button></form>
<p id="answer" aria-live="polite"></p><ul id="citations"></ul></main>
<script>
(() => {{
  const token = {token_js};
  const form = document.querySelector('form');
  const answer = document.querySelector('#answer');
  const citations = document.querySelector('#citations');
  form.addEventListener('submit', async (event) => {{
    event.preventDefault(); answer.textContent = 'Thinking…'; citations.replaceChildren();
    try {{
      const response = await fetch('/api/ask', {{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{assistant_token:token,question:document.querySelector('#question').value}})}});
      const result = await response.json(); if (!response.ok) throw new Error(result.detail || 'Request failed');
      answer.textContent = result.answer;
      for (const source of result.citations) {{ const li = document.createElement('li'); li.textContent = source.citation; citations.appendChild(li); }}
    }} catch (error) {{ answer.textContent = error.message || 'Assistant is unavailable'; }}
  }});
}})();
</script></html>'''
    return HTMLResponse(
        content=html,
        headers={"Referrer-Policy": "no-referrer", "X-Content-Type-Options": "nosniff"},
    )
