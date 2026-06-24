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

## LangGraph — Theory & How It's Wired

### What is LangGraph?

LangGraph is a library for building **stateful, multi-actor applications** with LLMs. It models the workflow as a directed graph where:
- **Nodes** are agents or functions — each does one job
- **Edges** define the order of execution
- **State** is a shared TypedDict passed between all nodes
- **Checkpoints** allow the graph to pause and resume mid-execution

LangGraph's key advantage over simple chaining (like LangChain LCEL) is that it supports **cycles, conditional branching, and human interrupts** — making it suitable for agentic workflows where the next step depends on what happened in the previous one.

### Why a Graph, Not a Chain?

A plain chain runs A → B → C with no ability to branch or pause. Orbit needs:
1. **Conditional routing** after intent detection (clarification needed? → halt)
2. **Human-in-the-loop** pause before execution
3. **Resumable state** — the graph persists checkpoint between the pause and resume

LangGraph provides all three natively.

### The StateGraph (Orbit's Wiring)

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

### Key LangGraph Concepts in Orbit

| Concept | What It Is | How Orbit Uses It |
|---------|-----------|------------------|
| `StateGraph` | Graph where every node reads/writes shared state | All 7 agents share `AgentState` |
| `add_conditional_edges` | Route to different next nodes based on state | After intent: halt if unclear, else continue |
| `interrupt_before` | Pause graph execution before a named node | Pauses before `approval` for human review |
| `MemorySaver` | In-memory checkpointer — saves state at each step | Stores graph state between pause and resume |
| `thread_id` | Key that identifies a conversation/run | `run_id` links the graph to the DB and LangSmith |
| `astream(stream_mode="updates")` | Async generator — yields state after each node | Drives the frontend live pipeline strip via SSE |

**Why `interrupt_before=["approval"]`?**
The graph pauses after `tool_router` finishes — the state is checkpointed. The human sees proposed actions on the frontend, then calls `/approve` or `/reject` via REST API. The tool executes there — not inside the graph. This means tool execution is fully decoupled from graph execution, making each independently testable.

---

## Multi-Agent Design Principles (Theory)

### Why Multiple Agents?

A single agent given all tasks (OCR + entity extraction + intent classification + action planning + validation) would face:
- **Context overload** — 10,000+ token prompts degrade LLM reasoning quality
- **Single point of failure** — one error in OCR corrupts everything downstream
- **No separation of concerns** — impossible to test or improve one step without affecting others

Multi-agent design solves this by assigning **one agent = one job = one prompt = one failure mode**.

### Agent Design Patterns Used

| Pattern | Definition | Where in Orbit |
|---------|-----------|---------------|
| **Sequential pipeline** | Agents run one after another; each feeds the next | understanding → intent → memory → planning → tool_router |
| **Conditional branching** | Route to different agents based on state | intent → clarification_halt OR memory |
| **Human-in-the-loop** | Graph pauses for human decision | Interrupt before approval node |
| **Shared state bus** | All agents read/write one shared object | AgentState TypedDict |
| **Fail-open safety** | Errors in non-critical agents allow continuation | G2 LLM guardrail catches exceptions and passes |
| **Structured output** | All LLM calls forced to return JSON via tool_use | Prevents free-text responses from breaking parsing |

### Sequential vs Parallel Agents

Orbit uses a **purely sequential pipeline**. Each agent strictly depends on the previous agent's output:
- `memory` needs `extracted_items` from `intent`
- `planning` needs `capture_id` from `memory`
- `tool_router` needs `actions` from `planning`

Parallelizing these would break the dependency chain. The trade-off is latency (sequential is slower) vs correctness (parallel would require independent inputs).

### Deterministic Routing

Orbit's conditional edge uses a **deterministic lambda** — not an LLM-decided router:
```python
lambda state: "clarification_halt" if state.get("clarification_needed") else "memory"
```
This is a deliberate choice. LLM-based routing (like in CrewAI) is non-deterministic — the same input can route differently on different runs. For a production workflow that moves money or sends emails, determinism is non-negotiable.

---

## Shared State (AgentState)

Every agent reads from and writes to one shared TypedDict — the **state bus**:

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

**Why TypedDict?**
LangGraph checkpoints state as JSON. TypedDict is JSON-serializable without `.dict()` calls, has no constructor overhead, and works with LangGraph's MemorySaver directly. Pydantic models would require serialization/deserialization at every checkpoint.

Each agent returns `{**state, new_field: value}` — spreads existing state and adds its outputs. No agent modifies previous agents' fields.

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
     │                        Traced in LangSmith. FAIL-OPEN on error.
     ▼
[G3] Payload policy        → Applied in planning + again at approval:
                              Gmail: email regex, blocked domains, body ≤ 5000
                              Slack: message ≤ 4000 chars
                              Calendar: ISO 8601 datetime required
```

**Fail-open** means G2 API errors never block users — transient failures pass through. This is intentional: a transient Anthropic API error is far more likely than a genuine spam campaign.

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
  G3 check                  Pydantic
  pre-execute               validation first,
     │                      then G3 check,
     ▼                      then execute
  _execute_tool()
     │
  Gmail / Cal.com / Slack
```

This pattern is called **"human-in-the-loop with interrupt"** — the graph doesn't resume after approval. Instead, execution is handed off entirely to the REST layer. This separates orchestration (LangGraph) from execution (FastAPI), making each independently deployable and testable.

---

## Memory & Semantic Search

- **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2` (384-dim, runs locally)
- **Storage:** pgvector in PostgreSQL with HNSW indexes
- **Two-level search:**
  1. Search `extracted_items` by cosine similarity
  2. Fallback to `captures` for any uncovered content
- **Query:** `SELECT *, 1 - (embedding <=> $query) AS score FROM extracted_items ORDER BY score DESC`

**Why local embeddings?** No API cost, no rate limits, no data leaving the server. 384 dimensions is sufficient for semantic similarity of short action descriptions.

**Why HNSW over IVFFlat?** IVFFlat requires a training phase (clustering). HNSW builds incrementally — works at any table size including 0 rows, and gives better recall at query time.

---

## LangSmith Observability

```python
config = {
    "configurable": {"thread_id": run_id},
    "run_name": "orbit-pipeline",
    "metadata": {"orbit_run_id": run_id, "input_type": "text"},
    "tags": ["orbit"],
}
```

Every LangGraph node appears as a child span automatically when `LANGCHAIN_TRACING_V2=true`. The G2 guardrail has `@traceable` — it appears as a named span in LangSmith even though it runs before the graph starts.

---

## Tools

| Tool | API Used | Returns |
|------|----------|---------|
| `gmail.send_email` | Google Gmail API v1 | `{message_id, to, subject, status}` |
| `calendar.create_booking` | Cal.com v2 REST | `{booking_uid, status, start}` |
| `slack.send_reminder` | Slack SDK | `{ts, channel, status}` |
| `slack.send_summary` | Slack SDK (mrkdwn blocks) | `{ts, channel, status}` |

**Lazy dispatch (C3):** `_get_handler(action_type)` resolves the handler at execution time via local imports. No function references stored in graph state — keeps state JSON-serializable for LangGraph checkpointing.

---

## Individual Contributions

---

### Daksh — Agent Orchestration & LangGraph Control Flow

**Domain:** How agents are connected, sequenced, and controlled.

**What he built:**
The entire `graph.py` — StateGraph wiring, node registration, edge definitions, conditional routing, interrupt mechanism, and SSE streaming integration.

**Conceptual ownership:**
- **Graph topology design** — deciding which agents exist, in what order, and what triggers a branch. This is the core architectural decision: 7 nodes, 6 deterministic edges, 1 conditional edge.
- **Conditional routing logic** — after the intent agent, the graph either halts (clarification needed) or continues (all items are clear). This is implemented as a deterministic lambda over state, not an LLM decision — a deliberate choice for predictability.
- **interrupt_before mechanism** — understanding that `interrupt_before=["approval"]` doesn't pause inside the approval node; it prevents the node from starting at all. The graph checkpoints, stops, and waits for an external signal (the REST endpoint resuming it).
- **SSE streaming** — `astream(stream_mode="updates")` yields state deltas after each node. The frontend consumes these as server-sent events to show a live pipeline strip.
- **run_id as thread_id** — the single string that links the graph checkpoint, the DB row, and the LangSmith trace. Without this, debugging a failed pipeline run across three systems would be impossible.

**Key design decisions:**
- Sequential pipeline (not parallel) — because each agent strictly depends on the previous agent's output
- Deterministic routing (lambda, not LLM) — predictable, testable, production-safe
- `interrupt_before` over `interrupt_after` — gives human a chance to see proposed actions before any DB writes for execution happen

**Likely viva questions:**
- What is a StateGraph and how does it differ from a simple LangChain chain?
- How does `interrupt_before` work mechanically — what does "checkpoint" mean here?
- Why not let an LLM decide routing?
- What would happen if you used `interrupt_after=["tool_router"]` instead?

---

### Utkarsh — Shared State & Memory Systems

**Domain:** How agents communicate, how data persists, how past captures are recalled.

**What he built:**
`AgentState` TypedDict, the 5-table DB schema, the `memory_agent`, and the `retrieval.py` semantic search module.

**Conceptual ownership:**
- **AgentState as the communication bus** — the 17-field TypedDict is the single source of truth for the entire pipeline. Every agent reads from it and writes back to it. The design ensures no agent modifies another agent's fields — write-once semantics prevent silent overwrites.
- **Why TypedDict over Pydantic** — LangGraph checkpoints state as JSON. TypedDict is JSON-serializable by default, has no constructor overhead, and works with MemorySaver directly. Pydantic models require `.dict()` calls and custom serializers at every checkpoint.
- **Two-level semantic retrieval** — extracteditem-level search (granular) with capture-level fallback (broad). This means a query for "email about deadline" first finds the specific item, then falls back to the full document if no specific item matches closely enough.
- **HNSW indexing** — hash-navigable small world graphs allow approximate nearest-neighbor search in O(log n) time. Unlike IVFFlat, HNSW doesn't need a training phase, so it works even when the table is empty.
- **Soft deletes** — `deleted_at` column instead of physical DELETE. Allows audit trails, undo operations, and prevents FK violations if related rows still exist.

**Key design decisions:**
- Single shared state object for all agents (not message passing between agents)
- Local embeddings (sentence-transformers) to avoid API cost and data egress
- Two-level retrieval: item → capture fallback ensures no query returns empty unless the DB is genuinely empty

**Likely viva questions:**
- Why does every agent receive the full state instead of just what it needs?
- What is a vector embedding and why is cosine similarity used over Euclidean distance?
- What is HNSW and why is it better than a brute-force search at scale?
- What happens if the memory agent fails to write? Does the graph continue?

---

### Jash — Tool Execution & External Integrations

**Domain:** How proposed actions get validated, dispatched, and executed against real external APIs.

**What he built:**
All 4 tool files (Gmail, Calendar, Slack, Auth), all Pydantic action schemas, the C3 lazy dispatch pattern in `actions.py`, and M6 payload validation at the edit endpoint.

**Conceptual ownership:**
- **C3 Lazy dispatch pattern** — action types are stored in AgentState as plain strings (`"gmail.send_email"`). The actual function handler is resolved at execution time via `_get_handler()`. This is necessary because storing callable objects in state would break LangGraph's JSON checkpointing — functions cannot be serialized to JSON. Strings can.
- **M6 Pydantic validation** — before executing an edited payload, it's validated against a schema class: `ACTION_SCHEMAS.get(action_type)(**body.edited_payload)`. This catches type errors, missing fields, and invalid values before they reach the external API.
- **H5 audit split** — three separate DB columns: `edited_payload` (what the human changed), `final_payload` (what was actually sent), `execution_result` (what the API returned). This makes auditing and debugging each phase independently possible.
- **H10 structured output** — all LLM calls use `tool_choice={"type":"tool"}` to force the model to call a tool with a defined JSON schema. This prevents the LLM from returning free text that would fail JSON parsing downstream.
- **Tool abstraction** — each tool returns a consistent dict structure. The approval router doesn't know which external API was called — it only sees `{status: "sent", ...}`. This makes swapping tools (e.g. replacing Cal.com with Google Calendar) a single-file change.

**Key design decisions:**
- Lazy dispatch over storing function references — keeps state JSON-serializable
- Pydantic schemas per action type — catches bad edits before API calls, not after
- Unified result format — every tool returns `{status, ...}` regardless of underlying API

**Likely viva questions:**
- Why can't you store function references in LangGraph state?
- What is the C3 pattern and when would you need it?
- What happens if the Gmail API call fails — what does the user see?
- Why validate the payload twice (at edit AND at approval)?

---

### Abhay — Evaluation, Guardrails & Reliability

**Domain:** How the system defends against bad input, ensures safe execution, and remains observable.

**What he built:**
`agents/guardrails.py` (G0-G3 system), LangSmith tracing integration, human-in-the-loop idempotency (H4), and the evaluation framework.

**Conceptual ownership:**
- **Layered guardrail design** — G0-G3 are not redundant; they're a defense-in-depth stack. G0 catches obvious garbage in microseconds. G1 catches known injection strings without an API call. G2 uses LLM classification for subtle attacks. G3 validates the action payload immediately before execution. Each layer has a different cost/coverage trade-off.
- **Fail-open on G2** — when the Claude Haiku API call inside G2 throws an exception, the guardrail returns `passed=True`. This is intentional: a transient Anthropic API outage should never block legitimate users. False negatives in G2 are caught by G3 (payload validation) and by the human review step anyway.
- **H4 idempotency** — double-approving an action must be prevented. The solution uses two layers: (1) `SELECT ... FOR UPDATE` at the application level — locks the row inside a transaction so concurrent requests queue, not race; (2) a partial unique index at the database level — `UNIQUE INDEX on decisions(action_id) WHERE action_id IS NOT NULL` — a second insert with the same `action_id` fails at the DB layer even if the application-level check was somehow bypassed.
- **G3 double application** — G3 runs at planning time (before storing proposed actions) and again at approval time (before executing). This prevents a scenario where a valid-at-planning payload becomes invalid by the time a human approves it hours later (e.g., domain was added to blocklist in between).
- **LangSmith tracing** — the `run_name`, `metadata`, and `tags` in the graph config create structured, searchable traces. `@traceable` on G2 means the guardrail check appears as a named span even though it runs outside the graph, giving a complete picture of the pipeline in one view.

**Key design decisions:**
- Fail-open (not fail-closed) on G2 — uptime takes priority over edge-case safety
- Two-layer idempotency (app + DB) — belt-and-suspenders for a write-once operation
- G3 runs twice — because the world changes between planning and execution

**Likely viva questions:**
- What is fail-open vs fail-closed? When would you choose each?
- How does `SELECT FOR UPDATE` prevent double-execution?
- What is the partial unique index on `decisions` and why is it partial (not full)?
- What does `@traceable` add beyond LangGraph's automatic tracing?

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
2. **intent** → 1 item: `{type: "communication", title: "Email Sarah about deadline", urgency: 0.7}` — `clarification_needed: false`
3. **Conditional edge routes to memory** (not clarification_halt)
4. **memory** → saved to DB, `capture_id` and `item_id` written back to state
5. **planning** → action: `{action_type: "gmail.send_email", payload: {to: "sarah@company.com", subject: "Project deadline update", body: "..."}}`
6. **tool_router** → validates `action_type` is in `VALID_ACTION_TYPES` ✓
7. **Graph checkpoints and pauses** — `interrupt_before=["approval"]` fires
8. User clicks **Approve** in frontend
9. `POST /api/actions/{id}/approve` → G3 validates email → `send_email()` called → email sent
10. Decision recorded: `{approved, final_payload, execution_result: {message_id: "...", status: "sent"}}`

---

## Viva Quick-Fire Answers

**Q: Why multiple agents instead of one?**
Each agent has one job, one prompt, one failure mode. A single agent doing OCR + entity extraction + item classification + action planning + validation is a 10,000-token context nightmare that hallucinates on complex documents. Separation also means each agent can be improved, swapped, or tested independently.

**Q: Why LangGraph over CrewAI or AutoGen?**
CrewAI uses LLM-decided routing — non-deterministic and hard to test. AutoGen is conversational (agents talk to each other as messages). LangGraph gives an explicit graph with typed state and deterministic routing, which is what a sequential pipeline with one conditional branch needs. You can read the graph definition and know exactly what will happen.

**Q: What does `interrupt_before` actually do?**
After `tool_router` completes, LangGraph checkpoints the full state keyed by `thread_id`, then stops before executing the `approval` node. MemorySaver holds this checkpoint. The graph does NOT resume — execution continues in the REST endpoint instead, which calls `_execute_tool()` directly.

**Q: How is concurrent double-approval prevented?**
Two layers: (1) `SELECT ... FOR UPDATE` locks the action row inside a transaction — a second request trying to lock the same row blocks until the first commits, then sees the updated status and returns 409. (2) A partial unique index on `decisions(action_id)` at the DB level means even if the app-level lock was bypassed, the insert would fail.

**Q: What is the C3 pattern?**
Lazy tool dispatch: only string action types (e.g. `"gmail.send_email"`) are stored in AgentState. The actual handler function is resolved at execution time via `_get_handler()`. This keeps state JSON-serializable for LangGraph checkpointing — you can't serialize a Python function to JSON.

**Q: Why HNSW over IVFFlat for vectors?**
IVFFlat requires a training phase (needs enough data to cluster centroid points). HNSW builds incrementally — works at any table size including 0 rows, and gives better recall at query time.

**Q: What happens if the understanding agent fails?**
LangGraph propagates the exception to the caller. The `try/except` in `captures.py` catches it, marks the run as failed in the DB, and returns HTTP 500. No subsequent agents run.

**Q: Why is G2 fail-open instead of fail-closed?**
A transient Anthropic API error is orders of magnitude more likely than a genuine harmful request. Fail-closed would block all users during any API hiccup. False negatives from G2 are still caught by G3 (payload policy) and by the human review step — the system has defense in depth.
