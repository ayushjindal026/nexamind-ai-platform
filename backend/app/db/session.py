"""
SQLAlchemy engine and session factory.

Phase 1 added the engine and check_db_connection() for /health/db.
Phase 2 adds get_db(): the request-scoped session dependency every
tenant-scoped route depends on (directly or via the auth dependencies).
Declarative Base and the ORM models live in app/db/base.py and app/models/.
"""

from collections.abc import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings

engine = create_engine(settings.database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def check_db_connection() -> bool:
    """Used by /health/db. Raises if Postgres is unreachable; caller handles the error."""
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    return True


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency: one session per request, always closed afterward."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
