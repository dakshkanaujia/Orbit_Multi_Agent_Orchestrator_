# Orbit AI

> Multi-Agent Personal Chief-of-Staff — powered by LangGraph, FastAPI, PostgreSQL, and Claude Haiku

Orbit processes documents (PDFs, images, text pastes), extracts actionable items using a 7-agent LangGraph pipeline, proposes executable actions (emails, calendar bookings, Slack messages), and waits for human approval before executing anything.

---

## How It Works

```
Input (text / PDF / image)
        │
  [Guardrails]  ← length check → injection scan → LLM safety
        │
  understanding → intent ──→ memory → planning → tool_router
                     │                                  │
               clarification_halt              [PAUSE — human reviews]
                                                        │
                                          /approve  /reject  /edit
                                                        │
                                         Gmail · Cal.com · Slack
```

1. You paste text or upload a PDF/image
2. The pipeline extracts entities, classifies intent, stores embeddings, and plans actions
3. The graph pauses — you see the proposed actions in the UI
4. You approve, reject, or edit each action
5. Approved actions execute against real external APIs

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Orchestration | LangGraph 0.2 (StateGraph + MemorySaver) |
| LLM | Anthropic Claude Haiku |
| Backend | FastAPI + uvicorn (async) |
| Database | PostgreSQL 16 + pgvector (HNSW indexes) |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` (local, 384-dim) |
| Frontend | Next.js 14 (App Router) + Tailwind CSS |
| Observability | LangSmith (optional) |
| Infra | Docker Compose |

---

## Prerequisites

- Docker and Docker Compose
- API keys for the services you want to use (see `.env.example`)

---

## Setup

**1. Clone and configure environment**

```bash
git clone <repo-url>
cd Orbit
cp backend/.env.example backend/.env
```

Open `backend/.env` and fill in:

| Variable | Where to get it |
|----------|----------------|
| `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com) |
| `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET` | Google Cloud Console → APIs & Services → Credentials |
| `GOOGLE_REFRESH_TOKEN` | Run `python get_google_token.py` after setting client credentials |
| `CALCOM_API_KEY` | app.cal.com → Settings → Developer → API Keys |
| `CALCOM_EVENT_TYPE_ID` | URL when editing an event type: `app.cal.com/event-types/<ID>/edit` |
| `SLACK_BOT_TOKEN` | api.slack.com → Your App → OAuth & Permissions |
| `LANGCHAIN_API_KEY` | [smith.langchain.com](https://smith.langchain.com) → Settings → API Keys (optional) |

**2. Start the stack**

```bash
docker compose up --build
```

This starts:
- `orbit_db` — PostgreSQL 16 with pgvector on port `5432`
- `orbit_backend` — FastAPI on port `8000`
- `orbit_frontend` — Next.js on port `3000`

**3. Open the app**

```
http://localhost:3000
```

API docs available at `http://localhost:8000/docs`

---

## Project Structure

```
Orbit/
├── backend/
│   ├── agents/          # 7 LangGraph nodes (understanding, intent, memory, planning, ...)
│   ├── graph.py         # StateGraph wiring — nodes, edges, interrupt_before
│   ├── memory/          # asyncpg pool, pgvector retrieval
│   ├── routers/         # FastAPI routers (captures, actions, search, hub)
│   ├── tools/           # Gmail, Cal.com, Slack integrations
│   └── models.py        # Pydantic schemas + ACTION_SCHEMAS
└── frontend/
    ├── app/             # Next.js pages (dashboard, workspace, approvals, search, hub)
    ├── components/      # Shared UI components
    └── lib/             # API client, types, utilities
```

---

## Key Features

- **Multi-agent pipeline** — 7 specialized agents, each with one job and one failure mode
- **Human-in-the-loop** — graph pauses via `interrupt_before`, tools only execute after explicit approval
- **Semantic search** — pgvector HNSW indexes on all captured content and extracted items
- **Guardrails** — 4-layer input safety stack (length → regex → LLM classification → payload policy)
- **Live pipeline view** — SSE streaming shows which agent is running in real time
- **Edit before approve** — payloads can be modified and re-validated before execution

---

## Contributors

| Name | Domain |
|------|--------|
| **Daksh** | Agent Orchestration — LangGraph StateGraph, conditional routing, SSE streaming |
| **Utkarsh** | State & Memory — AgentState TypedDict, asyncpg pool, pgvector retrieval |
| **Jash** | Tool Execution — Gmail/Slack/Cal.com integrations, Pydantic schemas, lazy dispatch |
| **Abhay** | Guardrails & Reliability — G0-G3 safety stack, LangSmith tracing, idempotency |
