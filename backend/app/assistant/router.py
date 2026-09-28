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
    record_question_usage,
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
    AssistantUsageResponse,
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


@management.get("/usage", response_model=AssistantUsageResponse)
def get_assistant_usage(
    membership: Membership = Depends(get_current_membership), db: Session = Depends(get_db)
) -> AssistantUsageResponse:
    assistant = get_assistant_for_organization(db, membership.organization_id)
    if assistant is None:
        return AssistantUsageResponse(questions_asked=0, questions_answered=0, questions_unavailable=0)
    return AssistantUsageResponse(
        questions_asked=assistant.questions_asked,
        questions_answered=assistant.questions_answered,
        questions_unavailable=assistant.questions_unavailable,
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
    try:
        record_question_usage(db, assistant.id)
    except SQLAlchemyError:
        db.rollback()
        logger.exception("Assistant usage tracking failed")
        raise HTTPException(status_code=500, detail="Assistant request failed")
    if embedding_provider is None:
        record_question_usage(db, assistant.id, "unavailable")
        raise HTTPException(status_code=503, detail="Embedding provider is not configured")
    if llm_provider is None:
        record_question_usage(db, assistant.id, "unavailable")
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
        record_question_usage(db, assistant.id, "unavailable")
        logger.exception("Question embedding failed")
        raise HTTPException(status_code=503, detail="Assistant is temporarily unavailable")
    except LLMProviderError:
        record_question_usage(db, assistant.id, "unavailable")
        logger.exception("Answer generation failed")
        raise HTTPException(status_code=503, detail="Assistant is temporarily unavailable")
    except DecisionEngineError:
        record_question_usage(db, assistant.id, "unavailable")
        logger.exception("Structured decision evaluation failed or is unavailable")
        raise HTTPException(status_code=503, detail="Structured decision service is temporarily unavailable")
    except SQLAlchemyError:
        record_question_usage(db, assistant.id, "unavailable")
        logger.exception("Assistant retrieval failed")
        raise HTTPException(status_code=500, detail="Assistant request failed")

    try:
        record_question_usage(db, assistant.id, "answered" if result.citations else "unavailable")
    except SQLAlchemyError:
        db.rollback()
        logger.exception("Assistant usage result tracking failed")
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
    """Serve a standalone public chat surface for direct links and iframe embeds."""
    if get_assistant_for_token(db, assistant_token) is None:
        return HTMLResponse(
            content=(
                "<!doctype html><html lang='en'><meta charset='utf-8'>"
                "<meta name='viewport' content='width=device-width,initial-scale=1'>"
                "<title>Assistant link unavailable</title>"
                "<body style='font:15px system-ui;max-width:36rem;margin:12vh auto;padding:1rem;color:#25334d'>"
                "<h1>This assistant link is no longer available</h1>"
                "<p>Ask the organization for a current assistant link, then try again.</p></body></html>"
            ),
            status_code=status.HTTP_401_UNAUTHORIZED,
            headers={"Referrer-Policy": "no-referrer", "X-Content-Type-Options": "nosniff"},
        )
    token_js = json.dumps(assistant_token).replace("<", "\\u003c")
    html = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="referrer" content="no-referrer"><title>PanScience Assistant</title>
<style>
:root{{font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;color:#1b2940;background:#f5f7fb}}*{{box-sizing:border-box}}body{{margin:0;min-height:100vh;padding:24px;background:radial-gradient(ellipse at 85% 0,#e7edff 0,transparent 34%),#f5f7fb}}main{{max-width:620px;margin:4vh auto;background:white;border:1px solid #e7ebf3;border-radius:20px;box-shadow:0 24px 65px #27375b12;overflow:hidden}}header{{padding:22px 26px;border-bottom:1px solid #edf0f5;display:flex;align-items:center;gap:12px}}.mark{{width:35px;height:35px;border-radius:11px;background:#456bdb;color:white;display:grid;place-items:center;font-weight:800}}.brand{{font-size:14px;font-weight:800;letter-spacing:-.03em}}.subbrand{{font-size:10px;color:#8591a6;margin-top:3px}}.intro{{padding:30px 27px 19px}}.eyebrow{{font-size:9px;color:#5875ce;font-weight:800;letter-spacing:.13em}}h1{{font-size:22px;letter-spacing:-.04em;margin:8px 0;color:#1c2a44}}.intro p{{margin:0;color:#76839b;font-size:12px;line-height:1.6}}form{{padding:0 27px 20px}}label{{display:block;font-size:11px;font-weight:700;color:#4f5d75;margin-bottom:7px}}textarea{{width:100%;min-height:94px;resize:vertical;padding:12px;border:1px solid #e0e6f0;border-radius:10px;outline:none;color:#26344d;font:13px/1.5 inherit}}textarea:focus{{border-color:#849be6;box-shadow:0 0 0 3px #456bdb12}}.submit-row{{display:flex;justify-content:flex-end;margin-top:10px}}button{{border:0;border-radius:9px;padding:11px 17px;background:#456bdb;color:#fff;font-size:11px;font-weight:700;cursor:pointer}}button:disabled{{opacity:.55;cursor:wait}}.conversation{{padding:0 27px 26px}}.answer-card{{display:none;padding:16px;background:#f7f9fd;border:1px solid #edf0f6;border-radius:12px;color:#35435d;font-size:13px;line-height:1.7;white-space:pre-wrap}}.answer-card.visible{{display:block}}.answer-label{{font-size:9px;color:#8b97aa;font-weight:800;letter-spacing:.12em;margin-bottom:7px}}.citations{{display:grid;gap:7px;margin:13px 0 0;padding:0;list-style:none}}.citations:empty{{display:none}}.citations li{{font-size:10px;color:#576987;padding:9px 11px;background:#f1f5ff;border-radius:8px}}.error{{margin:0 27px 21px;padding:10px 12px;border:1px solid #f3d6d5;background:#fff3f2;color:#9f4846;border-radius:9px;font-size:11px;display:none}}.error.visible{{display:block}}footer{{padding:13px 27px;border-top:1px solid #edf0f5;color:#a1aabd;font-size:9px}}@media(max-width:520px){{body{{padding:12px}}main{{margin:1vh auto}}header{{padding:17px 18px}}.intro{{padding:24px 19px 16px}}form{{padding:0 19px 17px}}.conversation{{padding:0 19px 20px}}.error{{margin-left:19px;margin-right:19px}}footer{{padding-left:19px}}}}
</style></head><body><main><header><div class="mark">P</div><div><div class="brand">PanScience Assistant</div><div class="subbrand">Answers from your organization’s knowledge base</div></div></header>
<section class="intro"><div class="eyebrow">KNOWLEDGE ASSISTANT</div><h1>What would you like to know?</h1><p>Ask a question about the organization’s documents. Answers include source pages when relevant information is available.</p></section>
<form id="ask-form"><label for="question">Your question</label><textarea id="question" name="question" required maxlength="2000" placeholder="Ask about a policy, process, or requirement…"></textarea><div class="submit-row"><button id="submit" type="submit">Ask assistant ↗</button></div></form>
<div class="error" id="error" role="alert"></div><section class="conversation"><div class="answer-card" id="answer" aria-live="polite"><div class="answer-label">ANSWER</div><div id="answer-text"></div><ul class="citations" id="citations" aria-label="Sources"></ul></div></section><footer>Grounded in organization documents · Private and organization-specific</footer></main>
<script>(() => {{
  const token = {token_js}; const form = document.querySelector('#ask-form'); const button = document.querySelector('#submit');
  const answer = document.querySelector('#answer'); const answerText = document.querySelector('#answer-text'); const citations = document.querySelector('#citations'); const error = document.querySelector('#error'); const question = document.querySelector('#question');
  question.addEventListener('keydown', (event) => {{ if (event.key === 'Enter' && !event.shiftKey) {{ event.preventDefault(); form.requestSubmit(); }} }});
  form.addEventListener('submit', async (event) => {{ event.preventDefault(); if (!question.value.trim()) {{ question.focus(); return; }}
    button.disabled = true; button.textContent = 'Thinking…'; error.textContent = ''; error.classList.remove('visible'); answer.classList.add('visible'); answerText.textContent = 'Searching the organization’s knowledge base…'; citations.replaceChildren();
    try {{ const response = await fetch('/api/ask', {{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{assistant_token:token,question:question.value.trim()}})}}); const result = await response.json(); if (!response.ok) throw new Error(result.detail || 'The assistant could not complete your request.');
      answerText.textContent = result.answer; for (const source of result.citations || []) {{ const li = document.createElement('li'); li.textContent = source.citation || (source.document_name + ' — Page ' + source.page_number); citations.appendChild(li); }}
    }} catch (cause) {{ answer.classList.remove('visible'); error.textContent = cause.message || 'The assistant is temporarily unavailable. Please try again.'; error.classList.add('visible'); }}
    finally {{ button.disabled = false; button.textContent = 'Ask assistant ↗'; }}
  }});
}})();</script></body></html>'''
    return HTMLResponse(
        content=html,
        headers={"Referrer-Policy": "no-referrer", "X-Content-Type-Options": "nosniff"},
    )
