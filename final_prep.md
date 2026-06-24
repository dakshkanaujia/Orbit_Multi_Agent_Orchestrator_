# Orbit AI — Final Prep Guide

> Multi-Agent Personal Chief-of-Staff | LangGraph + FastAPI + PostgreSQL + Claude Haiku

---

## What is Orbit AI?

Orbit AI processes documents (PDFs, images, text pastes) and automatically extracts actionable items, proposes executable actions (emails, calendar bookings, Slack messages), and waits for human approval before executing anything.

**One-line pitch:** _"You give it a document, it tells you what needs to happen next — and does it after you say yes."_

---

## System Architecture

```
User Input (text / PDF / image)
        │
        ▼
  [Input Guardrails]  ← G0 length, G1 injection, G2 LLM safety
        │
        ▼
  ┌─────────────────────────────────────────────────────┐
  │              LangGraph Pipeline                     │
  │                                                     │
  │  understanding → intent ──┬──→ memory               │
  │                           │         │               │
  │                    clarification    ▼               │
  │                    _halt (422)  planning             │
  │                                     │               │
  │                               tool_router           │
  │                                     │               │
  │                               approval ◄── PAUSE   │
  └─────────────────────────────────────────────────────┘
                                         │
                              Human reviews actions
                                         │
                              POST /approve | /reject | /edit
                                         │
                              Tool executes (Gmail / Cal.com / Slack)
```

---

## The 7 Agents

| Agent | Job | Key Output |
|-------|-----|-----------|
| **understanding** | OCR (PDF/image) + extract named entities via Claude Haiku | `capture` dict, `extracted_entities` |
| **intent** | Find all actionable items in the document | `extracted_items[]` list |
| **clarification_halt** | Terminal node — fires if a deadline can't be parsed | HTTP 422 returned |
| **memory** | Embed content + persist to PostgreSQL with pgvector | `capture_id`, `extracted_item_ids[]` |
| **planning** | Generate actions per item (email / calendar / slack) | `actions[]`, `pending_action_ids[]` |
| **tool_router** | Validate action types, build final pending list | Validated `actions[]` |
| **approval** | Graph pauses here. Human decides via REST API | Execution happens outside graph |

---

## LangGraph — How It's Wired

```python
builder = StateGraph(AgentState)

# Add all 7 nodes
builder.add_node("understanding", understanding_agent)
builder.add_node("intent", intent_agent)
builder.add_node("clarification_halt", clarification_halt_node)
builder.add_node("memory", memory_agent)
builder.add_node("planning", planning_agent)
builder.add_node("tool_router", tool_router_agent)
builder.add_node("approval", approval_agent)

# Wire edges
builder.set_entry_point("understanding")
builder.add_edge("understanding", "intent")

# THE key conditional branch
builder.add_conditional_edges(
    "intent",
    lambda state: "clarification_halt" if state.get("clarification_needed") else "memory"
)

builder.add_edge("clarification_halt", END)
builder.add_edge("memory", "planning")
builder.add_edge("planning", "tool_router")
builder.add_edge("tool_router", "approval")
builder.add_edge("approval", END)

# interrupt_before pauses graph BEFORE approval node runs
app = builder.compile(checkpointer=MemorySaver(), interrupt_before=["approval"])
```

**Why `interrupt_before=["approval"]`?**  
The graph pauses after tool_router finishes. The human sees proposed actions on the frontend, then calls `/approve` or `/reject` via REST API. The tool actually executes there — not inside the graph.

---

## Shared State (AgentState)

Every agent reads from and writes to one shared TypedDict:

```python
class AgentState(TypedDict):
    input_content: str          # raw input
    input_type: str             # text | pdf | image
    run_id: str                 # links API → graph → DB → LangSmith
    capture: dict               # {modality, source, raw_content, metadata}
    extracted_entities: dict    # {people, dates, locations, deadlines, ...}
    extracted_items: list       # items found by intent agent
    capture_id: str             # DB ID after memory agent writes
    extracted_item_ids: list    # DB IDs after memory agent writes
    actions: list               # proposed actions from planning
    pending_action_ids: list    # awaiting human approval
    decided_action_ids: list    # already approved/rejected
    clarification_needed: bool  # triggers clarification_halt branch
    clarification_reason: str   # why clarification is needed
    trace: list                 # per-agent summary entries
```

Each agent returns `{**state, new_field: value}` — spreads existing state and adds its outputs.

---

## Database Schema (5 tables)

```
captures          extracted_items         actions
─────────         ───────────────         ───────
id (UUID)         id (UUID)               id (UUID)
run_id            capture_id ──FK──►      extracted_item_id ──FK──►
modality          item_type               action_type
raw_content       title                   payload (JSONB)
embedding         description             requires_approval
(vector 384)      confidence_score        status
metadata          urgency_score           depends_on_action_id
deleted_at        deadline
                  planning_status
                  embedding (vector 384)

decisions                   runs
─────────                   ────
id (UUID)                   id (UUID)
action_id ──FK──►           capture_id
decision                    status
edited_payload              trace (JSONB)
final_payload               created_at
execution_result
decided_at
```

**Key constraint:** `UNIQUE INDEX on decisions(action_id) WHERE action_id IS NOT NULL` — prevents double-approving the same action.

---

## Guardrails (4 layers)

```
Incoming text
     │
     ▼
[G0] Length check          → reject if < 3 chars
     │
     ▼
[G1] Regex injection scan  → 10 patterns: "ignore previous instructions",
     │                        "jailbreak", "DAN mode", "act as", etc.
     ▼                        No API call, < 1ms
[G2] LLM safety check      → Claude Haiku classifies:
     │                        safe / prompt_injection / harmful_request /
     ▼                        data_exfiltration / spam_campaign / self_harm
     │                        @traceable in LangSmith. FAIL-OPEN on error.
     ▼
[G3] Payload policy        → Applied in planning + again at approval:
                              Gmail: email regex, blocked domains, body ≤ 5000
                              Slack: message ≤ 4000 chars
                              Calendar: ISO 8601 datetime required
```

**Fail-open** means G2 API errors never block users — transient failures pass through.

---

## Human-in-the-Loop Flow

```
Graph pauses at approval checkpoint
           │
           ▼
   Frontend shows pending actions
           │
     ┌─────┴──────┐
     ▼            ▼            ▼
  /approve     /reject       /edit
     │                         │
  G3 check                  M6 Pydantic
  pre-execute               validation first,
     │                      then G3 check,
     ▼                      then execute
  _execute_tool()
     │
  Gmail / Cal.com / Slack
     │
  DB: insert decision row
  {edited_payload, final_payload, execution_result}
```

Concurrent requests handled by `SELECT ... FOR UPDATE` inside a transaction (row-level lock).

---

## Memory & Semantic Search

- **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2` (384-dim, runs locally)
- **Storage:** pgvector in PostgreSQL with HNSW indexes
- **Two-level search:**
  1. Search `extracted_items` by cosine similarity
  2. Fallback to `captures` for any uncovered content
- **Query:** `SELECT *, 1 - (embedding <=> $query) AS score FROM extracted_items ORDER BY score DESC`

---

## LangSmith Observability

```python
config = {
    "configurable": {"thread_id": run_id},
    "run_name": "orbit-pipeline",
    "metadata": {"orbit_run_id": run_id, "input_type": "text"},
    "tags": ["orbit"],
}
# Every LangGraph node appears as a child span automatically
# when LANGCHAIN_TRACING_V2=true in .env
```

G2 guardrail has `@traceable(run_type="chain", name="content-safety-check")` — appears as a named span in LangSmith even though it runs before the graph.

---

## Tools

| Tool | File | API Used | Returns |
|------|------|----------|---------|
| `gmail.send_email` | tools/gmail.py | Google Gmail API v1 | `{message_id, to, subject, status}` |
| `calendar.create_booking` | tools/calendar.py | Cal.com v2 REST | `{booking_uid, status, start}` |
| `slack.send_reminder` | tools/slack.py | Slack SDK | `{ts, channel, status}` |
| `slack.send_summary` | tools/slack.py | Slack SDK (mrkdwn blocks) | `{ts, channel, status}` |

**Lazy dispatch (C3):** `_get_handler(action_type)` resolves the handler at execution time via local imports. No function references stored in graph state — keeps state JSON-serializable.

---

## Individual Contributions

---

### Daksh — Agent Orchestration (LangGraph Control Flow)

**What he built:** The entire `graph.py` — StateGraph wiring, node registration, edge definitions, conditional routing, interrupt mechanism, and SSE streaming integration.

**Key contributions:**
- `StateGraph(AgentState)` with 7 nodes and 6 edges
- Conditional edge after intent: routes to `clarification_halt` or `memory` based on `clarification_needed`
- `interrupt_before=["approval"]` — pauses graph for human decision
- `astream(stream_mode="updates")` driving the frontend pipeline strip via SSE
- `run_id` as `thread_id` in config — links graph execution to LangSmith and DB

**Why LangGraph?** Explicit graph with typed state, built-in checkpointing, resumable execution, and deterministic routing. Unlike CrewAI (LLM-decided routing) or AutoGen (conversational), Orbit needs a strict sequential pipeline with one conditional branch.

**Key design decision:** Sequential pipeline (not parallel) because each node strictly depends on the previous node's output — understanding feeds intent, intent feeds memory, etc.

---

### Utkarsh — State & Memory Systems

**What he built:** `AgentState` TypedDict, all 5 DB tables (schema + migrations), `asyncpg` connection pool, `memory_agent`, and the `retrieval.py` semantic search module.

**Key contributions:**
- `AgentState` TypedDict with 17 fields — the communication bus for all agents
- `memory/db.py`: asyncpg pool (min=2, max=10), JSON/JSONB codecs, all CRUD operations, soft deletes
- `memory_agent`: embeds capture + each item, inserts to PostgreSQL, H2 assertion (`assert len(item_ids) == len(items)`)
- HNSW indexes on both `captures.embedding` and `extracted_items.embedding`
- Two-level retrieval: items by cosine similarity → captures as fallback

**Why pgvector?** No extra infrastructure — vector search lives in the same PostgreSQL instance as relational data. SQL joins and transactional consistency come free.

**Why TypedDict for state (not Pydantic)?** LangGraph expects dict-like objects. TypedDict is JSON-serializable without `.dict()` calls, has no `__init__` overhead, and checkpoints cleanly with MemorySaver.

---

### Jash — Tool Execution & Integrations

**What he built:** All 4 tool files (`gmail.py`, `calendar.py`, `slack.py`, `auth.py`), all Pydantic action schemas in `models.py`, the C3 lazy dispatch pattern in `actions.py`, and M6 payload validation.

**Key contributions:**
- 4 external API integrations: Gmail (OAuth2 + MIME), Cal.com (REST), Slack (SDK), Google Auth (refresh token)
- `ACTION_SCHEMAS` dict mapping action_type strings → Pydantic classes (CalComBookingSchema, GmailSendSchema, SlackReminderSchema, SlackSummarySchema)
- `_get_handler()` + `_execute_tool()` — C3 lazy dispatch with full error wrapping
- M6 validation: `schema_cls(**body.edited_payload)` before any edited payload executes
- H5 audit split: `edited_payload`, `final_payload`, `execution_result` as separate DB columns
- Structured output pattern (H10): all LLM calls use `tool_choice={"type":"tool"}` to force JSON

**Why lazy dispatch?** Storing callable objects in `AgentState` would break LangGraph's JSON checkpointing. Strings serialize cleanly; handlers are resolved at execution time.

---

### Abhay — Evaluation, Guardrails & Reliability

**What he built:** `agents/guardrails.py` (G0-G3 system), LangSmith integration config, human-in-the-loop approval endpoints with H4 idempotency, and the evaluation framework.

**Key contributions:**
- G0-G3 guardrail stack: length → regex injection → LLM classification → payload policy
- `@traceable(run_type="chain", name="content-safety-check")` on G2 for LangSmith visibility
- Fail-open design: G2 API errors never block users
- H4 idempotency: `SELECT ... FOR UPDATE` in transaction + partial unique index on `decisions(action_id)`
- M7 row-level locking: prevents concurrent double-execution of the same action
- LangSmith config in both capture endpoints: `run_name`, `metadata`, `tags`
- Double G3 application: at planning time (pre-store) + at approval time (pre-execute)

**Why fail-open on G2?** A transient Anthropic API error is far more likely than a genuine spam campaign. Blocking all users due to an API outage would break the product. False negatives in G2 are caught by G3 and human approval anyway.

---

## 10 Key Design Decisions (Quick Reference)

| Decision | Choice Made | Why |
|----------|-------------|-----|
| Orchestration | LangGraph StateGraph | Explicit graph, typed state, built-in checkpointing |
| Routing | Deterministic (lambda) | Predictable, testable — no LLM-decided routing |
| Human approval | `interrupt_before` + REST | Graph pauses cleanly; human acts via standard HTTP |
| Checkpointing | MemorySaver | Simple for demo; swap to Redis/Postgres for production |
| Embeddings | sentence-transformers local | No API cost, no rate limits, 384-dim sufficient |
| Vector store | pgvector in PostgreSQL | No extra infra, SQL joins, transactional consistency |
| Tool dispatch | C3 lazy (string → handler) | State stays JSON-serializable; no circular imports |
| Safety | G2 fail-open | API errors must never block legitimate users |
| Idempotency | DB lock + unique index | Two layers: app-level + database-level |
| LLM calls | Forced tool_use (H10) | Structured JSON guaranteed; free text impossible |

---

## End-to-End Example

**Input:** _"Email Sarah at sarah@company.com that the project deadline is moved to July 5th"_

1. **understanding** → entities: `{people: ["Sarah"], dates: ["July 5th"], emails: ["sarah@company.com"]}`
2. **intent** → 1 item: `{type: "communication", title: "Email Sarah about deadline", urgency: 0.7}`
3. **memory** → saved to DB, `capture_id` and `item_id` returned
4. **planning** → action: `{action_type: "gmail.send_email", payload: {to: "sarah@company.com", subject: "Project deadline update", body: "..."}}`
5. **tool_router** → validates action_type is in VALID_ACTION_TYPES ✓
6. **Graph pauses** at approval checkpoint
7. User clicks **Approve** in frontend
8. `POST /api/actions/{id}/approve` → G3 validates email → `send_email()` called → email sent
9. Decision recorded: `{approved, final_payload, execution_result: {message_id: "...", status: "sent"}}`

---

## Viva Quick-Fire Answers

**Q: Why multiple agents instead of one?**  
Each agent has one job, one prompt, one failure mode. A single agent doing OCR + entity extraction + item classification + action planning + validation is a 10,000-token context nightmare that hallucinates on complex documents.

**Q: Why LangGraph over CrewAI?**  
CrewAI uses LLM-decided routing — non-deterministic and hard to test. LangGraph gives an explicit graph with typed state, which is what a sequential pipeline with one conditional branch needs.

**Q: What happens if the understanding agent fails?**  
LangGraph propagates the exception to the caller. The `try/except` in `captures.py` catches it, calls `db.update_run(run_id, "failed", ...)`, and returns HTTP 500.

**Q: How is concurrent double-approval prevented?**  
`SELECT ... FOR UPDATE` locks the action row inside a transaction. A second request trying to lock the same row blocks until the first transaction commits, then sees the updated status and returns 409.

**Q: What does `interrupt_before` actually do?**  
The graph checkpoints state after `tool_router` completes, then stops before executing the `approval` node. MemorySaver stores this checkpoint keyed by `thread_id`. The graph does NOT resume — execution continues in the REST endpoint instead.

**Q: Why HNSW over IVFFlat for vectors?**  
IVFFlat requires a training phase (needs enough data to cluster). HNSW builds incrementally — works at any table size including 0 rows, and gives better recall.

**Q: What is the C3 pattern?**  
Lazy tool dispatch: only string action types (e.g. `"gmail.send_email"`) are stored in AgentState. The actual handler function is resolved at execution time via `_get_handler()`. This keeps state JSON-serializable for LangGraph checkpointing.
