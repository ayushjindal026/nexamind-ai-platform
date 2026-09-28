"""Assistant token lifecycle and organization-scoped question answering."""

import hashlib
import secrets
import uuid

from sqlalchemy import update
from sqlalchemy.orm import Session

from app.models.assistant import Assistant


class AssistantAlreadyConfiguredError(Exception):
    pass


class AssistantNotConfiguredError(Exception):
    pass


def _new_token() -> str:
    return secrets.token_urlsafe(32)


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def get_assistant_for_organization(db: Session, organization_id: uuid.UUID) -> Assistant | None:
    return db.query(Assistant).filter(Assistant.organization_id == organization_id).one_or_none()


def create_assistant(db: Session, organization_id: uuid.UUID) -> tuple[Assistant, str]:
    if get_assistant_for_organization(db, organization_id) is not None:
        raise AssistantAlreadyConfiguredError()
    token = _new_token()
    assistant = Assistant(organization_id=organization_id, token_hash=_token_hash(token))
    db.add(assistant)
    db.commit()
    db.refresh(assistant)
    return assistant, token


def regenerate_assistant_token(db: Session, organization_id: uuid.UUID) -> tuple[Assistant, str]:
    assistant = get_assistant_for_organization(db, organization_id)
    if assistant is None:
        raise AssistantNotConfiguredError()
    token = _new_token()
    assistant.token_hash = _token_hash(token)
    db.commit()
    db.refresh(assistant)
    return assistant, token


def get_assistant_for_token(db: Session, token: str) -> Assistant | None:
    """Validate a visitor bearer token and return its server-owned assistant."""
    if not token:
        return None
    return db.query(Assistant).filter(Assistant.token_hash == _token_hash(token)).one_or_none()


def record_question_usage(db: Session, assistant_id: uuid.UUID, outcome: str | None = None) -> None:
    """Atomically count accepted visitor questions and their final outcomes."""
    values: dict[str, object] = {}
    if outcome is None:
        values["questions_asked"] = Assistant.questions_asked + 1
    elif outcome == "answered":
        values["questions_answered"] = Assistant.questions_answered + 1
    elif outcome == "unavailable":
        values["questions_unavailable"] = Assistant.questions_unavailable + 1
    else:
        raise ValueError("Unsupported question outcome")
    db.execute(update(Assistant).where(Assistant.id == assistant_id).values(**values))
    db.commit()
