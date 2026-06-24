# Orbit AI — Commit Assignment Plan

> 4 contributors · 53 files mapped · commit in sequence

---

## Commit Order

```
Abhay (1) → Utkarsh (2) → Daksh (3) → Jash (4)
```

| Step | Person | Domain | Why First |
|------|--------|--------|-----------|
| 1 | **Abhay** | Guardrails & Infra | Docker, deps, env — nothing runs without these |
| 2 | **Utkarsh** | State & Memory | DB schema + AgentState must exist before agents import anything |
| 3 | **Daksh** | Orchestration | graph.py compiles after DB and guardrails are importable |
| 4 | **Jash** | Tools & Execution | Tools + actions router depend on graph and agents being wired |

---

## Abhay — Evaluation, Guardrails & Reliability

**9 files**

### Guardrails
| # | File | Role |
|---|------|------|
| 01 | `backend/agents/guardrails.py` | **PRIMARY** — G0-G3 stack, @traceable G2, fail-open design |

### Project Infrastructure
| # | File | Role |
|---|------|------|
| 02 | `docker-compose.yml` | Full-stack orchestration (PostgreSQL + pgvector + backend + frontend) |
| 03 | `backend/Dockerfile` | Backend container |
| 04 | `frontend/Dockerfile` | Frontend container |
| 05 | `backend/requirements.txt` | Python deps (includes langsmith, sentence-transformers, asyncpg) |
| 06 | `backend/.env.example` | Env var template — LangSmith config, API keys reference |

### Frontend Config
| # | File | Role |
|---|------|------|
| 07 | `frontend/package.json` | Node.js dependencies |
| 08 | `frontend/next.config.ts` | Next.js configuration |
| 09 | `frontend/tailwind.config.ts` | Tailwind CSS configuration |

> **Also commit:** `backend/__init__.py` and `backend/routers/__init__.py` — empty init files that establish Python package structure for all subsequent imports.

---

## Utkarsh — State & Memory Systems

**20 files**

### Memory Layer
| # | File | Role |
|---|------|------|
| 01 | `backend/memory/db.py` | **PRIMARY** — asyncpg pool, all CRUD operations, JSON/JSONB codecs, soft deletes |
| 02 | `backend/memory/retrieval.py` | pgvector semantic search, two-level item → capture fallback |
| 03 | `backend/memory/__init__.py` | Package init |

### Agent
| # | File | Role |
|---|------|------|
| 04 | `backend/agents/memory.py` | Memory agent — embeds captures + items, writes to PostgreSQL |

### API Routers
| # | File | Role |
|---|------|------|
| 05 | `backend/routers/dashboard.py` | Dashboard stats API |
| 06 | `backend/routers/items.py` | Extracted items CRUD API |
| 07 | `backend/routers/search.py` | Semantic search API endpoint |

### Frontend — Data Layer
| # | File | Role |
|---|------|------|
| 08 | `frontend/lib/types.ts` | TypeScript type definitions (mirrors AgentState, all API response shapes) |
| 09 | `frontend/lib/api.ts` | API client functions |
| 10 | `frontend/lib/utils.ts` | Shared utilities — ITEM_TYPE_COLORS, ITEM_TYPE_BORDER_COLORS, formatters |

### Frontend — Pages
| # | File | Role |
|---|------|------|
| 11 | `frontend/app/dashboard/page.tsx` | Dashboard UI — capture stats, recent captures |
| 12 | `frontend/app/workspace/[capture_id]/page.tsx` | Capture workspace — extracted items + action results |
| 13 | `frontend/app/items/page.tsx` | All items list with filters |
| 14 | `frontend/app/search/page.tsx` | Semantic search UI |

### Frontend — Components
| # | File | Role |
|---|------|------|
| 15 | `frontend/components/ItemTypeBadge.tsx` | Item type pill with icon per type |
| 16 | `frontend/components/ui/badge.tsx` | Shared badge component |
| 17 | `frontend/components/ui/button.tsx` | Shared button component |
| 18 | `frontend/components/ui/card.tsx` | Shared card component |
| 19 | `frontend/components/ui/progress.tsx` | Progress bar (supports blue/amber/auto color modes) |
| 20 | `frontend/components/ui/skeleton.tsx` | Skeleton loading component |

---

## Daksh — Agent Orchestration & LangGraph Control Flow

**14 files**

### Graph Core
| # | File | Role |
|---|------|------|
| 01 | `backend/graph.py` | **PRIMARY** — StateGraph, 7 nodes, 6 edges, 1 conditional edge, interrupt_before, MemorySaver |
| 02 | `backend/main.py` | FastAPI app, router registration, CORS |

### API Routers
| # | File | Role |
|---|------|------|
| 03 | `backend/routers/captures.py` | Graph invocation entry point + SSE streaming endpoint |
| 04 | `backend/routers/hub.py` | Pipeline state SSE router for live visualization |

### Agents
| # | File | Role |
|---|------|------|
| 05 | `backend/agents/understanding.py` | Node 1 — OCR (PDF/image) + entity extraction via Claude Haiku |
| 06 | `backend/agents/intent.py` | Node 2 — intent classification, sets clarification_needed |
| 07 | `backend/agents/clarification.py` | Node 3 — conditional halt node, returns HTTP 422 |
| 08 | `backend/agents/tool_router.py` | Node 6 — validates action types before approval interrupt |
| 09 | `backend/agents/approval.py` | Node 7 — interrupt checkpoint (graph pauses here) |
| 10 | `backend/agents/__init__.py` | Package init |

### Frontend — Nav & Layout
| # | File | Role |
|---|------|------|
| 11 | `frontend/app/layout.tsx` | Global nav layout, page shell |
| 12 | `frontend/app/page.tsx` | Root redirect |
| 13 | `frontend/components/NavLinks.tsx` | Nav links component with active state |
| 14 | `frontend/app/hub/page.tsx` | Live pipeline visualization — SSE consumer, shows agent progress |

---

## Jash — Tool Execution & External Integrations

**10 files**

### Tools
| # | File | Role |
|---|------|------|
| 01 | `backend/tools/gmail.py` | **PRIMARY** — Gmail API v1, OAuth2 + MIME email sending |
| 02 | `backend/tools/calendar.py` | Cal.com v2 REST — calendar booking creation |
| 03 | `backend/tools/slack.py` | Slack SDK — send_reminder + send_summary with mrkdwn blocks |
| 04 | `backend/tools/auth.py` | Google OAuth credential refresh |
| 05 | `backend/tools/__init__.py` | Package init |

### Schemas & Planning
| # | File | Role |
|---|------|------|
| 06 | `backend/models.py` | All Pydantic schemas + ACTION_SCHEMAS dict (maps action_type → schema class) |
| 07 | `backend/agents/planning.py` | Planning agent — LLM generates structured action proposals per item |

### Action Router & Dispatch
| # | File | Role |
|---|------|------|
| 08 | `backend/routers/actions.py` | /approve + /reject + /edit endpoints, C3 lazy dispatch, M6 validation |

### Frontend
| # | File | Role |
|---|------|------|
| 09 | `frontend/app/approvals/page.tsx` | Approvals UI — pending actions, approve/reject/edit controls |
| 10 | `get_google_token.py` | OAuth setup utility — generates Google refresh token |

---

## Shared Ownership — Overlap Notes

| File | Committed By | Overlap |
|------|-------------|---------|
| `backend/routers/actions.py` | **Jash** | Contains **Abhay**'s H4 idempotency logic — `SELECT FOR UPDATE` + 409 on double-approve. Note this in the commit message. |
| `backend/graph.py` | **Daksh** | `AgentState` TypedDict defined here is conceptually **Utkarsh**'s design (17-field shared state bus). Utkarsh should claim it in viva even though Daksh's file contains it. |
| `backend/models.py` | **Jash** | Imported by every router. Commit this before Daksh's `captures.py` and Abhay's `guardrails.py` are committed, or those files will have missing imports. |
| `backend/__init__.py` `backend/routers/__init__.py` | **Abhay** (step 1) | Empty init files — must exist before any Python imports resolve. Commit in the infra pass. |
