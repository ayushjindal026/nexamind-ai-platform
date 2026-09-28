# Multi-Tenant AI Knowledge & Decision Assistant

PanScience Innovations SDE-1 take-home. Built incrementally, phase by phase — see `IMPLEMENTATION_LOG.md` (added from Phase 2 onward) for what exists at each stage.

## Status: Phase 4 — Embeddings, pgvector storage, and organization-scoped retrieval

Implements (on top of Phase 3's upload/extraction): `document_chunks` table with a pgvector `vector(1536)` column and HNSW cosine index, an `EmbeddingProvider` interface (OpenAI `text-embedding-3-small` implementation), best-effort chunking + embedding after upload, and `POST /api/retrieval/search` — the tenant-scoped similarity-search primitive later RAG phases build on. No LLM, Laya, or assistant/chat yet — see Phase log below.

## Architecture (final target — implemented incrementally)

```
Visitor (no login)                    Org Admin (dashboard)
      │                                      │
      ▼                                      ▼
 assistant_token in URL              JWT session (login)
      │                                      │
      ▼                                      ▼
              FastAPI backend (org_id always server-derived,
              never trusted from the client)
                      │
      ┌───────────────┼──────────────────────┐
      ▼                                       ▼
 /api/ask pipeline:                  /api/documents, /api/assistant,
 retrieve → classify →                /api/usage (dashboard CRUD)
 decompose (if needed) →
 System 1 (Laya) →
 combine (app logic) →
 final LLM answer →
 citations
```

### Tenant model (Phase 2)

```
User ──< Membership >── Organization
```

A user never has a direct `organization_id`. Every authenticated request does: JWT → decode → `User` → look up that user's `Membership` row → `organization_id` (server-derived, never accepted from the client). Current MVP invariant: exactly one membership per user, enforced by a `UNIQUE` constraint on `memberships.user_id` — see `app/models/membership.py` for the full rationale and what would need to change to support multi-org membership later.

### Document ingestion (Phase 3)

```
documents (organization_id, storage_key, status: processing|ready|failed)
    └──< document_pages (document_id only — ownership flows through the document, no denormalized organization_id)
```

Upload pipeline: validate raw bytes (empty / oversized / not-a-PDF by magic bytes) → open PDF → check page count → check organization quota → extract page text → create `documents` row (`processing`) → write to storage → persist `document_pages` + finalize to `ready`. Every step before the row is created is a `422` with nothing persisted; every step after is handled explicitly (see `app/documents/service.py`'s module docstring for the exact failure-mode behavior and its one documented residual risk — a row stuck at `processing` only if two database commits fail back-to-back within one request).

Storage is local filesystem (`LocalFilesystemStorage`) behind a 3-method `StorageBackend` protocol — swapping to S3 later means one new class, no service-layer changes. Keys are always server-generated (`{organization_id}/{document_id}.pdf`), never derived from the client-supplied filename, which is sanitized for display only.

### Embeddings & retrieval (Phase 4)

```
documents ──< document_pages (raw extraction, one row per page)
    └──< document_chunks (RAG unit: text + embedding vector(1536), NULL until embedded)
```

- **Chunking** (`app/documents/chunking.py`): one chunk per non-empty page today. `chunk_index` already exists so sub-page splitting later is a change to that one function, not a migration.
- **Embedding is an enhancement, never a precondition of upload.** After a successful upload the router chunks and embeds in one batched provider call. No provider configured, a provider failure, or malformed provider output all leave the document `ready` with un-embedded chunks (`embedded_chunk_count < chunk_count` in the API response). `embed_pending_chunks_for_organization()` embeds whatever is still pending — no queue or worker.
- **Retrieval** (`app/retrieval/service.py`): `search_similar_chunks(db, organization_id, query_embedding, top_k, min_similarity)`. `organization_id` is a required argument, every query joins through `documents` and filters on `documents.organization_id` (chunks carry no `organization_id` of their own — the document is the single ownership boundary, same as `document_pages`). Only embedded chunks of `ready` documents are searchable; `min_similarity` is the relevance-threshold hook the answer pipeline will use to say "not in the knowledge base" without calling an LLM with irrelevant context.
- **Provider abstraction**: only `app/embeddings/provider.py` knows about OpenAI. Tests inject a deterministic offline embedder (`tests/embedding_fixtures.py`, feature hashing) and the test suite deliberately **never** calls a real embedding API, even if `OPENAI_API_KEY` is set locally.
- **Why the dimension isn't a setting:** pgvector bakes the dimension into the column DDL, so changing it always means a migration plus re-embedding everything.

`POST /api/retrieval/search` — auth required; body `{"query": str, "top_k": 1-20 (default 5), "min_similarity": optional -1..1}`; returns `results[]` with `document_name`, `page_number`, `text`, `similarity`, and a ready-made `citation` (`"Scholarship_Rules.pdf — Page 4"`). `503` if no embedding provider is configured or it is failing (generic message; provider errors are logged, never returned).

## Running locally

```bash
cp .env.example .env
docker compose up --build
```

Ports are read from `.env` — the values below are this project's defaults; if you've changed them locally (e.g. because 5432/8000 were already taken), use your own `.env` values instead:

- Backend: http://localhost:8000/health and http://localhost:8000/health/db
- Frontend: http://localhost:5173
- Postgres: localhost:5432 (credentials in `.env`)

### Database migrations

Migrations are applied explicitly, never automatically on container boot:

```bash
docker compose exec backend alembic upgrade head
```

Future schema changes: `docker compose exec backend alembic revision --autogenerate -m "description"`, review the generated file under `backend/alembic/versions/`, then `alembic upgrade head` again.

### Manual API check

```bash
curl -X POST http://localhost:8000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"owner@acme-university.com","password":"correct-horse-battery","organization_name":"Acme University"}'

curl http://localhost:8000/api/auth/me -H "Authorization: Bearer <access_token from above>"

curl -X POST http://localhost:8000/api/documents \
  -H "Authorization: Bearer <access_token>" \
  -F "file=@/path/to/some.pdf;type=application/pdf"

curl http://localhost:8000/api/documents -H "Authorization: Bearer <access_token>"
```

## Configuration

All limits are configurable via environment variables (see `.env.example`), never hard-coded: `MAX_DOCUMENT_SIZE_BYTES` (default 10 MB), `MAX_DOCUMENT_PAGES` (default 20), `MAX_DOCUMENTS_PER_ORGANIZATION` (default 10). `ENVIRONMENT=production` requires an explicitly-set `JWT_SECRET_KEY` — the app refuses to start otherwise rather than silently using the checked-in development default.

## Known limitations

- **Approximate index + tenant filter:** the HNSW index is approximate and the organization filter is applied alongside it. At this project's scale Postgres plans an exact scan; at much larger scale a selective tenant filter can return fewer than `top_k` rows even when more matches exist. Standard mitigations (pgvector iterative scans, per-tenant partial indexes, partitioning) are production improvements, not MVP work.
- **Retrieval quality is only as good as the embedding model.** The offline test embedder is word-overlap only (no stemming or synonyms) — the suite asserts ranking and isolation, not semantic quality. Real semantic quality needs a real `OPENAI_API_KEY`, which this repo's tests intentionally do not exercise.
- Page-level chunks: a very long page becomes one large chunk until sub-page chunking is added.
- Embedding happens synchronously inside the upload request (no worker), so upload latency includes one embedding call.
- If a document's final database commit fails *and* the follow-up "mark as failed" commit also fails within the same request, the row is left at `processing` with no automatic reconciliation. No queue/worker exists in this MVP to retry it — documented, not hidden (see `app/documents/service.py`).
- Document storage is a local Docker volume; moving to S3 means implementing one more `StorageBackend`, not restructuring the document service.
- Only `POST` and `GET` exist for `/api/documents` — no single-document `GET /{id}` or `DELETE` yet (not required by any test or feature through Phase 3).

## Phase log

- **Phase 1**: repo structure, Docker Compose, health checks.
- **Phase 2**: `organizations`/`users`/`memberships` schema, Alembic, Argon2id hashing, JWT auth, server-derived tenant resolution.
- **Phase 3**: `documents`/`document_pages` schema, multipart upload with full validation, PyMuPDF extraction, local filesystem storage behind a swappable interface, organization-scoped upload/list, failure-mode handling with storage cleanup. Verified against real Postgres and a real running server — 29/29 tests passing, plus a manual walkthrough (upload, list, isolation, all rejection cases) confirmed against live HTTP requests.
- **Phase 4** (this phase): `document_chunks` + pgvector (`vector(1536)`, HNSW cosine index), `EmbeddingProvider` interface, best-effort chunk+embed after upload with graceful degradation and an org-scoped recovery function, and `POST /api/retrieval/search`. 59/59 tests pass against real PostgreSQL + real pgvector; tenant-isolation tests verified by mutation (removing the org filter makes exactly the isolation tests fail).
