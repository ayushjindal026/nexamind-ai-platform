"""
App entrypoint.

Phase 1: /health, /health/db only.
Phase 2: adds the auth router (register/login/me).
Phase 3: adds the documents router (upload/list), plus a global exception
handler as a last line of defense — every router already translates its own
known exceptions into safe HTTP responses (see auth/router.py,
documents/router.py), but this handler ensures that ANY truly unexpected
exception anywhere still returns a generic message instead of a raw
traceback, filesystem path, or SQL error string.

Phase 4: adds the retrieval router (organization-scoped vector search).

Each future phase adds its own router module and one include_router line
here — this file never grows business logic itself.
"""

import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.auth.router import router as auth_router
from app.core.config import settings
from app.db.session import check_db_connection
from app.documents.router import router as documents_router
from app.retrieval.router import router as retrieval_router

logger = logging.getLogger(__name__)

app = FastAPI(title="AI Knowledge & Decision Assistant", version="0.4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(documents_router)
app.include_router(retrieval_router)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/health/db")
def health_db() -> dict:
    try:
        check_db_connection()
    except Exception as exc:  # noqa: BLE001 — a health/ops endpoint, not a business API;
        # revealing "unreachable: <reason>" here is a deliberate operator-facing
        # debugging signal, unlike the business endpoints elsewhere in the app.
        raise HTTPException(status_code=503, detail=f"database unreachable: {exc}") from exc
    return {"status": "ok", "database": "reachable"}
