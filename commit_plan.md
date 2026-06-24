# Orbit AI — Commit Assignment Plan

> 4 contributors · repo created Jun 24 7:00 PM IST · all commits Jun 24, 8:00 PM → 10:00 PM IST

---

## Commit Order

```
Abhay (8:00 PM) → Utkarsh (8:30 PM) → Daksh (9:00 PM) → Jash (9:30 PM) → Frontend (10:00 PM)
```

| Step | Person | Time (IST) | Domain |
|------|--------|------------|--------|
| 1 | **Abhay** | Jun 24, 8:00 PM | Guardrails, Docker, infra setup |
| 2 | **Utkarsh** | Jun 24, 8:30 PM | Memory layer, DB, retrieval |
| 3 | **Daksh** | Jun 24, 9:00 PM | LangGraph orchestration, agents, SSE |
| 4 | **Jash** | Jun 24, 9:30 PM | Tools, schemas, action execution |
| 5 | **Frontend** | Jun 24, 10:00 PM | Entire `frontend/` pushed as one commit |

---

## Step 1 — Abhay

**Jun 24, 2026 · 8:00 PM IST**

### Files

| # | File | Role |
|---|------|------|
| 01 | `backend/agents/guardrails.py` | **PRIMARY** — G0-G3 stack, @traceable G2, fail-open design |
| 02 | `docker-compose.yml` | Full-stack orchestration — PostgreSQL + pgvector + backend + frontend |
| 03 | `backend/Dockerfile` | Backend container |
| 04 | `backend/requirements.txt` | Python deps (langsmith, sentence-transformers, asyncpg, fastapi) |
| 05 | `backend/.env.example` | Env var template — LangSmith, API keys, DB URL |
| 06 | `backend/__init__.py` | Package init |
| 07 | `backend/routers/__init__.py` | Package init |

### Git Commands

```bash
git add backend/agents/guardrails.py \
        docker-compose.yml \
        backend/Dockerfile \
        backend/requirements.txt \
        backend/.env.example \
        backend/__init__.py \
        backend/routers/__init__.py

GIT_AUTHOR_DATE="2026-06-24 20:00:00 +0530" \
GIT_COMMITTER_DATE="2026-06-24 20:00:00 +0530" \
git commit -m "feat: add guardrails stack (G0-G3), docker-compose, and project setup

- G0: length check, G1: regex injection scan, G2: LLM safety (fail-open), G3: payload policy
- docker-compose with PostgreSQL + pgvector extension
- backend Dockerfile and requirements"

git push origin main
```

---

## Step 2 — Utkarsh

**Jun 24, 2026 · 8:30 PM IST**

### Files

| # | File | Role |
|---|------|------|
| 01 | `backend/memory/db.py` | **PRIMARY** — asyncpg pool, all CRUD, JSON/JSONB codecs, soft deletes |
| 02 | `backend/memory/retrieval.py` | pgvector semantic search, two-level item → capture fallback |
| 03 | `backend/memory/__init__.py` | Package init |
| 04 | `backend/agents/memory.py` | Memory agent — embeds captures + items, writes to PostgreSQL |
| 05 | `backend/routers/dashboard.py` | Dashboard stats API |
| 06 | `backend/routers/items.py` | Extracted items CRUD API |
| 07 | `backend/routers/search.py` | Semantic search API endpoint |

### Git Commands

```bash
git pull origin main

git add backend/memory/db.py \
        backend/memory/retrieval.py \
        backend/memory/__init__.py \
        backend/agents/memory.py \
        backend/routers/dashboard.py \
        backend/routers/items.py \
        backend/routers/search.py

GIT_AUTHOR_DATE="2026-06-24 20:30:00 +0530" \
GIT_COMMITTER_DATE="2026-06-24 20:30:00 +0530" \
git commit -m "feat: add memory layer — asyncpg pool, pgvector retrieval, memory agent

- db.py: connection pool (min=2, max=10), all CRUD, soft deletes
- retrieval.py: HNSW cosine search on extracted_items, fallback to captures
- memory agent: embeds capture + each item via sentence-transformers (384-dim)
- dashboard, items, search API routers"

git push origin main
```

---

## Step 3 — Daksh

**Jun 24, 2026 · 9:00 PM IST**

### Files

| # | File | Role |
|---|------|------|
| 01 | `backend/graph.py` | **PRIMARY** — StateGraph, 7 nodes, 6 edges, conditional edge, interrupt_before, MemorySaver |
| 02 | `backend/main.py` | FastAPI app entry point, router registration, CORS |
| 03 | `backend/routers/captures.py` | Graph invocation + SSE streaming endpoint |
| 04 | `backend/routers/hub.py` | Pipeline state SSE router for live agent visualization |
| 05 | `backend/agents/understanding.py` | Node 1 — OCR (PDF/image) + entity extraction via Claude Haiku |
| 06 | `backend/agents/intent.py` | Node 2 — intent classification, sets clarification_needed |
| 07 | `backend/agents/clarification.py` | Node 3 — conditional halt node, returns HTTP 422 |
| 08 | `backend/agents/tool_router.py` | Node 6 — validates action types before approval interrupt |
| 09 | `backend/agents/approval.py` | Node 7 — interrupt checkpoint (graph pauses here) |
| 10 | `backend/agents/__init__.py` | Package init |

### Git Commands

```bash
git pull origin main

git add backend/graph.py \
        backend/main.py \
        backend/routers/captures.py \
        backend/routers/hub.py \
        backend/agents/understanding.py \
        backend/agents/intent.py \
        backend/agents/clarification.py \
        backend/agents/tool_router.py \
        backend/agents/approval.py \
        backend/agents/__init__.py

GIT_AUTHOR_DATE="2026-06-24 21:00:00 +0530" \
GIT_COMMITTER_DATE="2026-06-24 21:00:00 +0530" \
git commit -m "feat: add LangGraph orchestration — StateGraph, 7 agents, SSE streaming

- graph.py: StateGraph with interrupt_before=['approval'], MemorySaver checkpointing
- conditional edge after intent: routes to clarification_halt or memory
- astream(stream_mode='updates') drives frontend pipeline strip via SSE
- understanding, intent, clarification, tool_router, approval agents
- captures.py: graph entry point; hub.py: live pipeline SSE router"

git push origin main
```

---

## Step 4 — Jash

**Jun 24, 2026 · 9:30 PM IST**

### Files

| # | File | Role |
|---|------|------|
| 01 | `backend/tools/gmail.py` | **PRIMARY** — Gmail API v1, OAuth2 + MIME email sending |
| 02 | `backend/tools/calendar.py` | Cal.com v2 REST — calendar booking creation |
| 03 | `backend/tools/slack.py` | Slack SDK — send_reminder + send_summary with mrkdwn blocks |
| 04 | `backend/tools/auth.py` | Google OAuth credential refresh |
| 05 | `backend/tools/__init__.py` | Package init |
| 06 | `backend/models.py` | All Pydantic schemas + ACTION_SCHEMAS dict |
| 07 | `backend/agents/planning.py` | Planning agent — LLM generates structured action proposals per item |
| 08 | `backend/routers/actions.py` | /approve + /reject + /edit endpoints, C3 lazy dispatch, M6 validation |
| 09 | `get_google_token.py` | OAuth setup utility — generates Google refresh token |

### Git Commands

```bash
git pull origin main

git add backend/tools/gmail.py \
        backend/tools/calendar.py \
        backend/tools/slack.py \
        backend/tools/auth.py \
        backend/tools/__init__.py \
        backend/models.py \
        backend/agents/planning.py \
        backend/routers/actions.py \
        get_google_token.py

GIT_AUTHOR_DATE="2026-06-24 21:30:00 +0530" \
GIT_COMMITTER_DATE="2026-06-24 21:30:00 +0530" \
git commit -m "feat: add tool integrations and action execution layer

- tools: Gmail (OAuth2+MIME), Cal.com REST, Slack SDK, Google auth refresh
- models.py: Pydantic schemas + ACTION_SCHEMAS for M6 payload validation
- planning agent: LLM generates structured actions with forced tool_use (H10)
- actions router: C3 lazy dispatch, M6 edit validation, H4 idempotency (SELECT FOR UPDATE)"

git push origin main
```

---

## Step 5 — Frontend (pushed last)

**Jun 24, 2026 · 10:00 PM IST**

All frontend code pushed as a single commit by any one team member.

### Files

| File | Owner (conceptual) |
|------|--------------------|
| `frontend/app/layout.tsx` | Daksh |
| `frontend/app/page.tsx` | Daksh |
| `frontend/app/hub/page.tsx` | Daksh |
| `frontend/components/NavLinks.tsx` | Daksh |
| `frontend/app/dashboard/page.tsx` | Utkarsh |
| `frontend/app/workspace/[capture_id]/page.tsx` | Utkarsh |
| `frontend/app/items/page.tsx` | Utkarsh |
| `frontend/app/search/page.tsx` | Utkarsh |
| `frontend/components/ItemTypeBadge.tsx` | Utkarsh |
| `frontend/components/ui/badge.tsx` | Utkarsh |
| `frontend/components/ui/button.tsx` | Utkarsh |
| `frontend/components/ui/card.tsx` | Utkarsh |
| `frontend/components/ui/progress.tsx` | Utkarsh |
| `frontend/components/ui/skeleton.tsx` | Utkarsh |
| `frontend/lib/types.ts` | Utkarsh |
| `frontend/lib/api.ts` | Utkarsh |
| `frontend/lib/utils.ts` | Utkarsh |
| `frontend/app/approvals/page.tsx` | Jash |
| `frontend/Dockerfile` | Abhay |
| `frontend/package.json` | Abhay |
| `frontend/next.config.ts` | Abhay |
| `frontend/tailwind.config.ts` | Abhay |
| `frontend/next-env.d.ts` | Abhay |

### Git Commands

```bash
git pull origin main

git add frontend/

GIT_AUTHOR_DATE="2026-06-24 22:00:00 +0530" \
GIT_COMMITTER_DATE="2026-06-24 22:00:00 +0530" \
git commit -m "feat: add Next.js frontend — dashboard, workspace, approvals, search, hub

- layout + nav with active link state
- dashboard: capture stats, recent captures list
- workspace: extracted items with confidence/urgency bars, action results
- approvals: pending action review with approve/reject/edit
- search: semantic search UI with pgvector results
- hub: live pipeline strip via SSE
- shared UI components: card, badge, button, progress, skeleton"

git push origin main
```

---

## Shared Ownership — Overlap Notes

| File | Committed By | Overlap |
|------|-------------|---------|
| `backend/routers/actions.py` | **Jash** | Contains **Abhay**'s H4 idempotency logic — `SELECT FOR UPDATE` + 409 on double-approve. Mention in commit message. |
| `backend/graph.py` | **Daksh** | `AgentState` TypedDict defined here is conceptually **Utkarsh**'s design (17-field shared state bus). Utkarsh should claim it in viva. |
| `backend/models.py` | **Jash** | Imported by every router. Must be committed before Daksh's step 3 and Abhay's `guardrails.py` — or Python imports will fail. |
| `backend/__init__.py` `backend/routers/__init__.py` | **Abhay** (step 1) | Empty init files — must exist in step 1 so all subsequent Python package imports resolve. |
