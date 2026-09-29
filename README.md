# Multi-Tenant AI Knowledge & Decision Assistant

The problem I'm solving is fairly simple to describe, but surprisingly difficult to implement reliably.

Organizations have large amounts of information stored inside PDFs, policies and internal documents. A normal LLM can answer questions about them, but it can also hallucinate, make incorrect decisions, or expose information belonging to another organization.

So I built NexaMind as a multi-tenant AI knowledge assistant.

An organization can upload its own documents, those documents are processed and embedded into a vector database, and users can then ask questions against that organization's knowledge base.

But the interesting part isn't just RAG.

The system can also break eligibility questions into individual conditions, evaluate those conditions using deterministic application logic and an AI-based semantic evaluator, combine the results into a structured decision, and then use the LLM only to explain the final result.


## Status: Phase 8 — Organization dashboard and visitor experience

Phase 4 provides `document_chunks` with pgvector embeddings and organization-scoped retrieval. Phase 5 adds hashed, rotatable organization assistant tokens and scoped document deletion. Phase 6 composes retrieval with grounded LLM answers and citations. Phase 7 uses Laya System 1 for structured decisions. Phase 8 adds an authenticated React dashboard, document management, assistant link/embed controls, basic question usage counts, a public visitor chat page, and an external iframe preview. The existing tenant model, RAG flow, Gemini/OpenAI adapters, and Laya decision architecture are reused.

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
 /api/ask pipeline:                  /api/documents, /api/assistant
 retrieve → LLM decision plan
      ├─ ordinary question: final LLM answer
      └─ structured decision: Laya System 1 → app combination → final LLM
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
- **Provider abstraction**: `app/embeddings/provider.py` implements OpenAI and Gemini behind the same interface. Tests inject deterministic fakes and never call either live API, even if provider keys are set locally.
- **Why the dimension isn't a setting:** pgvector bakes the dimension into the column DDL, so changing it always means a migration plus re-embedding everything.

`POST /api/retrieval/search` — auth required; body `{"query": str, "top_k": 1-20 (default 5), "min_similarity": optional -1..1}`; returns `results[]` with `document_name`, `page_number`, `text`, `similarity`, and a ready-made `citation` (`"Scholarship_Rules.pdf — Page 4"`). `503` if no embedding provider is configured or it is failing (generic message; provider errors are logged, never returned).

### Assistant access and public Q&A (Phases 5–6)

- `GET /api/assistant` — authenticated status for the caller's organization. It never returns the raw visitor token.
- `POST /api/assistant` — authenticated; creates that organization's assistant and returns its one-time `assistant_token`, a tokenized visitor page URL, and a copyable iframe embed snippet. A second create returns `409`.
- `POST /api/assistant/regenerate` — authenticated; rotates the token and immediately revokes the previous one. Returns `404` if the organization has no assistant yet.
- `POST /api/ask` — public; accepts `{"assistant_token": str, "question": str}`. The token may instead be supplied as the `assistant_token` query parameter in the returned URL. Server-side token validation resolves the organization; the client cannot provide an organization id. Response shape: `{"answer": str, "citations": [{"document_id", "document_name", "page_number", "citation"}]}`.
- `DELETE /api/documents/{document_id}` — authenticated and organization-scoped; removes the stored PDF and its database record, pages, and chunks. A foreign or missing document returns `404`.
- `GET /api/assistant/usage` — authenticated and organization-scoped; returns all-time `questions_asked`, `questions_answered`, and `questions_unavailable` totals for the organization's assistant. Questions count after a valid assistant token is accepted. Successful answers with source citations count as answered; no-context results and provider failures count as unavailable.

The assistant token is generated with a cryptographic random source. Only its SHA-256 digest is stored; create and regenerate reveal the raw token once. The tokenized `/assistant` page provides the visitor chat form and is also used by the iframe embed. Phase 6 uses `OPENAI_API_KEY` for embeddings and chat completions, with `LLM_MODEL` (default `gpt-4o-mini`) selecting the answer model. `ANSWER_MIN_SIMILARITY` (default `0.25`) controls the no-context refusal threshold. Missing providers return a safe `503`; the test suite overrides the LLM dependency with deterministic fakes and never calls the live API.

### Structured decisions (Phase 7)

The existing public `POST /api/ask` endpoint now plans structured work from the retrieved organization context. Ordinary questions keep the Phase 6 answer flow. Decision plans contain independent checks executed together by the `SystemOneProvider` interface and the Laya adapter: assignment `boolean` maps to Laya `noul`, `classification` maps to `choice`, while `choice` and `score` retain their Laya types. Eligibility plans combine boolean checks with explicit `all` or `any` rules in application code; the final LLM explains the resulting decisions using the same retrieved context. The response adds `decisions` and `decision_outcome`; citations remain server-generated. A structured request returns a safe `503` if Laya cannot initialize or evaluate. Set `LAYA_DEVICE=cpu` (the default) to use CPU inference; the first structured request may download model weights. No database tables or migrations are needed.

## Running locally

```bash
cp .env.example .env
docker compose up --build
```

Ports are read from `.env` — the values below are this project's defaults; if you've changed them locally (e.g. because 5432/8000 were already taken), use your own `.env` values instead:

- Backend: http://localhost:8000/health and http://localhost:8000/health/db
- Frontend: http://localhost:5173
- Postgres: localhost:5432 (credentials in `.env`)

Open the organization dashboard at `http://localhost:5173`. Register a workspace or sign in, then add PDFs and configure the public assistant. The Docker Compose frontend uses the existing React/Vite setup; `cd frontend && npm install && npm run dev` is available when running the frontend outside Docker. Run `cd frontend && npm run build` to type-check and build the frontend.

After upgrading the backend container, apply schema changes with `docker compose exec backend alembic upgrade head` before using the dashboard's usage counters.

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

# Create the organization assistant; keep the returned one-time token safe.
curl -X POST http://localhost:8000/api/assistant \
  -H "Authorization: Bearer <access_token>"

# Public visitor question (no dashboard JWT). Use the assistant_token returned above.
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"assistant_token":"<assistant_token>","question":"What is the scholarship GPA requirement?"}'
```

## Configuration

All limits are configurable via environment variables (see `.env.example`), never hard-coded: `MAX_DOCUMENT_SIZE_BYTES` (default 10 MB), `MAX_DOCUMENT_PAGES` (default 20), `MAX_DOCUMENTS_PER_ORGANIZATION` (default 10). `ENVIRONMENT=production` requires an explicitly-set `JWT_SECRET_KEY` — the app refuses to start otherwise rather than silently using the checked-in development default.

Phase 6 also reads `OPENAI_API_KEY`, `LLM_MODEL` (default `gpt-4o-mini`), and `ANSWER_MIN_SIMILARITY` (default `0.25`). The same OpenAI key configures embeddings and answer generation; tests override both provider dependencies and never use the key.

Set `EMBEDDING_PROVIDER=openai` and `LLM_PROVIDER=openai` to retain the defaults. Set either variable to `gemini` to select that provider independently; both Gemini providers use `GEMINI_API_KEY`. Defaults are `GEMINI_EMBEDDING_MODEL=gemini-embedding-2` (output explicitly constrained to 1536 dimensions) and `GEMINI_LLM_MODEL=gemini-3.8-flash`. Gemini Embedding 2 supports 1536 dimensions, matching the existing pgvector schema. The models are selected from Google's current Gen AI API; check the Google AI Studio project's free-tier limits before use. Missing keys leave the selected provider unavailable and provider/API failures use the existing graceful error paths.

The embedding recovery service remains organization-scoped and is not exposed as an HTTP endpoint. Once Gemini is configured and the backend restarted, invoke it for the existing organization from the backend container (replace `<organization-uuid>`):

```powershell
docker compose exec backend python -c "import uuid; from app.db.session import SessionLocal; from app.embeddings.provider import get_embedding_provider; from app.embeddings.service import embed_pending_chunks_for_organization; db=SessionLocal(); provider=get_embedding_provider(); print(embed_pending_chunks_for_organization(db, provider, uuid.UUID('<organization-uuid>')) if provider else 'Embedding provider is not configured'); db.close()"
```

The recovery function embeds up to 200 pending chunks per invocation. Run it again while it reports 200 if that organization has more pending chunks. Switching embedding models does not transform existing vectors; chunks already embedded by OpenAI must be re-embedded before they are compared with Gemini query vectors.

## Known limitations

- **Approximate index + tenant filter:** the HNSW index is approximate and the organization filter is applied alongside it. At this project's scale Postgres plans an exact scan; at much larger scale a selective tenant filter can return fewer than `top_k` rows even when more matches exist. Standard mitigations (pgvector iterative scans, per-tenant partial indexes, partitioning) are production improvements, not MVP work.
- **Retrieval quality is only as good as the embedding model.** The offline test embedder is word-overlap only (no stemming or synonyms) — the suite asserts ranking and isolation, not semantic quality. Real semantic quality needs a real `OPENAI_API_KEY`, which this repo's tests intentionally do not exercise.
- Page-level chunks: a very long page becomes one large chunk until sub-page chunking is added.
- Embedding happens synchronously inside the upload request (no worker), so upload latency includes one embedding call.
- LLM grounding uses retrieved context and an explicit refusal instruction, but generated text cannot be guaranteed free of unsupported claims. Laya weights may need to be downloaded on first structured decision use; missing Laya configuration or inference errors return a safe 503 for decision requests, while ordinary RAG questions remain available.
- Usage counters are all-time totals and do not yet support date ranges or historical breakdowns.
- If a document's final database commit fails *and* the follow-up "mark as failed" commit also fails within the same request, the row is left at `processing` with no automatic reconciliation. No queue/worker exists in this MVP to retry it — documented, not hidden (see `app/documents/service.py`).
- Document storage is a local Docker volume; moving to S3 means implementing one more `StorageBackend`, not restructuring the document service.
- There is no single-document `GET /api/documents/{id}` endpoint.

## Phase log

- **Phase 1**: repo structure, Docker Compose, health checks.
- **Phase 2**: `organizations`/`users`/`memberships` schema, Alembic, Argon2id hashing, JWT auth, server-derived tenant resolution.
- **Phase 3**: `documents`/`document_pages` schema, multipart upload with full validation, PyMuPDF extraction, local filesystem storage behind a swappable interface, organization-scoped upload/list, failure-mode handling with storage cleanup. Verified against real Postgres and a real running server — 29/29 tests passing, plus a manual walkthrough (upload, list, isolation, all rejection cases) confirmed against live HTTP requests.
- **Phase 4** (this phase): `document_chunks` + pgvector (`vector(1536)`, HNSW cosine index), `EmbeddingProvider` interface, best-effort chunk+embed after upload with graceful degradation and an org-scoped recovery function, and `POST /api/retrieval/search`. 59/59 tests pass against real PostgreSQL + real pgvector; tenant-isolation tests verified by mutation (removing the org filter makes exactly the isolation tests fail).
- **Phase 5**: hashed, rotatable organization assistant tokens; authenticated assistant status/create/regenerate endpoints; public-token API entry; organization-scoped document deletion.
- **Phase 6**: `POST /api/ask` composes existing retrieval with an LLM provider abstraction, returns application-generated document/page citations, refuses when no relevant context is found, and maps provider failures to safe errors. LLM tests use deterministic fakes. Full suite: 73 tests pass against PostgreSQL + pgvector.
- **Phase 7**: structured LLM decision plans are evaluated by Laya behind an independent provider interface, with choice/boolean/score/classification types and deterministic eligibility combination. The final answer is generated from the organization-scoped context and decisions. No schema changes; 10 deterministic Phase 7 tests added; full suite: 84 passing against PostgreSQL + pgvector.
- **Phase 8**: authenticated React login/register and organization dashboard; tenant-scoped PDF management, assistant link and embed controls, basic per-assistant question counters, public visitor chat UI, and a token-free external embed preview page. Existing auth/document/assistant/ask endpoints are reused; adds authenticated `GET /api/assistant/usage` and three all-time counters on the assistant record.

### Dashboard and assistant UI (Phase 8)

The dashboard is served by the existing Vite frontend at `http://localhost:5173`; sign-in and registration use the existing JWT API. JWTs are held in `sessionStorage` for the current browser session. Document and assistant management use the existing authenticated APIs, which derive the organization from the JWT. Visitor URLs continue to point to the backend's validated `/assistant?assistant_token=...` route, so they work without dashboard login and remain compatible with the generated iframe. Usage totals are read from the authenticated `GET /api/assistant/usage` endpoint.

To preview an embed on an external page, open `demo/embed-demo.html` (or serve the repository root) and paste the iframe code from the dashboard. The demo file contains no token; it constructs the iframe from the generated snippet at runtime.
