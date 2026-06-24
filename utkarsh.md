# Utkarsh — Viva Preparation Guide
### State & Memory Systems · Orbit AI Multi-Agent Personal Chief-of-Staff

---

> **Scope of ownership:** AgentState design, PostgreSQL schema, asyncpg connection pool, memory agent, vector embeddings, RAG retrieval, checkpointing. Every piece of data that enters Orbit's brain passes through your code before any other agent sees it again.

---

## TABLE OF CONTENTS

1. Personal Ownership Summary
2. Deep Implementation Walkthrough
3. LangGraph Concepts — State-Focused
4. Multi-Agent Concepts — Memory & State Focus
5. Design Decisions and Defenses
6. Viva Question Bank (120 Questions)
   - 6.1 Beginner (20 Q&A)
   - 6.2 Intermediate (20 Q&A)
   - 6.3 Advanced (20 Q&A)
   - 6.4 LangGraph State (20 Q&A)
   - 6.5 Memory & RAG (20 Q&A)
   - 6.6 Database Design (20 Q&A)
7. Cross-Module Questions
8. Demonstration Script (3–5 minutes)
9. Examiner Challenge Scenarios

---

## 1. PERSONAL OWNERSHIP SUMMARY

### What You Built

You are responsible for the **persistent memory layer** of Orbit AI — the system that ensures nothing a user captures is ever lost, forgotten, or unreachable. Concretely, you built:

| Component | File | Role |
|-----------|------|------|
| AgentState schema | `models.py` | Shared contract for all 7 agents |
| Memory Agent | `agents/memory.py` | Embeds + persists captures and items |
| Database Layer | `memory/db.py` | asyncpg pool + all CRUD operations |
| Retrieval System | `memory/retrieval.py` | Two-level semantic search + context assembly |
| PostgreSQL Schema | migrations/schema | 5 tables, pgvector, HNSW indexes |

### Why Each Piece Is Needed

**AgentState** is the single source of truth that all 7 agents read from and write to. Without a well-defined TypedDict, different agents would invent their own key names, causing silent KeyError failures at runtime. The TypedDict is a contract — if `memory_agent` writes `capture_id`, every downstream agent can rely on it being there.

**Memory Agent** is the first DB write in the entire pipeline. Before memory_agent runs, everything exists only in RAM inside the LangGraph state object. After memory_agent runs, the capture and its items are durably stored in PostgreSQL. If the process crashes after this point, the data survives.

**Database Layer** wraps asyncpg with a stable API (`insert_capture`, `get_capture`, etc.) so that no agent ever writes raw SQL. This means if the schema changes, only `db.py` needs updating — the agents are insulated.

**Retrieval System** turns the database into a context engine. When a new capture arrives, instead of processing it in isolation, the system can pull semantically similar past captures and items to give downstream agents (and the LLM prompts) relevant history. This is what makes Orbit feel like it "remembers" you.

**Schema design** determines what questions you can answer later. The 5-table design with soft deletes, pgvector embeddings, and a separate decisions table means you can: reconstruct any past pipeline run, audit every human decision, search by meaning rather than keyword, and never lose a decision even if the action it belongs to is deleted.

### How It Connects to Every Other Module

```
User Input
    │
    ▼
[understanding agent]  ──── writes: capture, extracted_entities ──→ AgentState
    │
    ▼
[intent agent]         ──── writes: extracted_items ──────────────→ AgentState
    │
    ▼
[MEMORY AGENT ★]       ──── writes: capture_id, extracted_item_ids → AgentState
    │                   ──── first DB writes (captures + extracted_items tables)
    ▼
[planning agent]       ──── reads: extracted_item_ids
    │                   ──── writes: actions, pending_action_ids
    ▼
[tool_router agent]    ──── reads: actions
    │                   ──── writes: validated actions, pending_action_ids
    ▼
[approval agent]       ──── reads: pending_action_ids from DB
    │                   ──── writes: trace entry only (decisions stored in DB separately)
    ▼
[execution agent]      ──── reads: decided_action_ids
                        ──── writes: execution_results
```

**Connection to Daksh (orchestrator/routing):** Daksh's routing logic reads `clarification_needed` from AgentState — a field you defined. If `True`, the graph branches to a clarification path instead of proceeding to memory_agent. Your schema decision to include this field enables Daksh's conditional routing.

**Connection to Jash (planning):** Jash's planning agent reads `extracted_item_ids` — the list of UUIDs you write into AgentState during memory_agent. Without these IDs, planning cannot link actions to specific items in the DB.

**Connection to Abhay (approval/execution):** Abhay's approval endpoint reads `pending_action_ids` from the DB and from AgentState. Your `actions` table and `decisions` table are what Abhay queries when the human-in-the-loop webhook fires.

**Connection to LangSmith:** The `run_id` field you defined in AgentState doubles as the `thread_id` sent to LangSmith. This means every LangSmith trace for a run maps 1:1 to a row in your `runs` table.

**Connection to the Hub Page (AI Digest):** The retrieval system you built (`memory/retrieval.py`) is what powers the AI digest feature on the hub page. When the dashboard loads, it calls `assemble_retrieval_context` to pull recent semantically relevant items for display.

---

## 2. DEEP IMPLEMENTATION WALKTHROUGH

### 2.1 AgentState TypedDict — Every Field

```python
class AgentState(TypedDict):
    input_content: str          # raw text or file path
    input_type: str             # text | pdf | image
    run_id: str                 # thread_id for LangSmith + DB linkage
    capture: dict               # {modality, source, raw_content, file_path, metadata}
    extracted_entities: dict    # {people, dates, locations, deadlines, organizations, urls}
    extracted_items: list       # pre-DB item dicts from intent agent
    capture_id: str             # UUID after memory agent inserts to DB
    extracted_item_ids: list    # list of UUIDs after memory agent inserts items
    actions: list               # flat list of action dicts from planning + tool_router
    pending_action_ids: list    # IDs of actions awaiting human approval
    decided_action_ids: list    # IDs of actions with decisions (H3 multi-resume tracking)
    decisions: list             # decision records
    execution_results: list     # tool return values
    retrieval_context: list     # semantic search results for context enrichment
    clarification_needed: bool  # set by intent agent on validation failure
    clarification_reason: str   # human-readable reason for clarification
    trace: list                 # per-agent summary entries {agent, timestamp, ...}
```

**Field-by-field reasoning:**

`input_content` / `input_type` — Raw entry point. The understanding agent reads these to decide how to parse the input (text directly, PDF via extraction, image via OCR/vision).

`run_id` — **The correlation key.** This single UUID ties together: the LangSmith trace, the row in the `runs` table, every `capture` row, and every `extracted_item` row. If you want to reconstruct "what happened in run X," you query by `run_id`.

`capture` — A structured dict that the understanding agent assembles before writing to AgentState. It mirrors the `captures` table schema so memory_agent can insert it directly. Fields: `modality` (text/pdf/image), `source` (e.g. "user_input"), `raw_content` (the actual text), `file_path` (if file-based), `metadata` (arbitrary JSONB).

`extracted_entities` — Populated by the understanding agent. Structured extraction of named entities: people, dates, locations, deadlines, organizations, URLs. Used by the intent agent to validate and enrich items.

`extracted_items` — Populated by the intent agent. A list of dicts, each representing one actionable item identified in the capture. These are **pre-DB** — they have no IDs yet. Each dict has: `item_type`, `title`, `description`, `confidence_score`, `urgency_score`, `deadline`, `entities`, `metadata`.

`capture_id` — **Written by memory_agent.** UUID of the newly inserted row in `captures`. Downstream agents use this for linking.

`extracted_item_ids` — **Written by memory_agent.** List of UUIDs corresponding 1:1 with `extracted_items`. The H2 assertion `assert len(item_ids) == len(state["extracted_items"])` enforces this alignment.

`actions` — Populated by the planning agent. A flat list of action dicts. The "flat list" design is intentional — actions with dependencies are modeled via `depends_on_action_id` in the DB, not via nesting in state.

`pending_action_ids` — Actions that need human approval before execution. Set by tool_router after validating which actions `require_approval`.

`decided_action_ids` — Part of the H3 multi-resume pattern. When the graph is interrupted for approval and then resumed multiple times (one decision per resume), this list tracks which action IDs have already received decisions, preventing double-processing.

`decisions` — Decision records written back to state after the approval webhook fires. Each entry: `{action_id, decision, edited_payload, final_payload, decided_at}`.

`execution_results` — Return values from tool calls. Each entry corresponds to one executed action.

`retrieval_context` — Results from `assemble_retrieval_context`. A list of enriched item/capture dicts with `semantic_score` fields, sorted by relevance. Injected into LLM prompts by the planning and intent agents for context grounding.

`clarification_needed` / `clarification_reason` — Set by the intent agent when it cannot confidently parse the input. Daksh's router reads `clarification_needed`; if True, the graph branches to ask the user for more information instead of proceeding to memory_agent.

`trace` — A lightweight audit log built up across the entire run. Every agent appends one entry: `{agent, timestamp, ...agent-specific fields}`. **Critical pattern:** agents always append, never replace: `return {**state, "trace": state["trace"] + [new_entry]}`. The M4 pattern (lightweight trace) means only summary data goes here — not full state dumps — keeping the MemorySaver checkpoint small.

---

### 2.2 Memory Agent — Code Walkthrough

```python
_embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
```

**Lazy-load singleton pattern.** The `_embedder` variable is module-level. `SentenceTransformer` loads a 90MB model into RAM on first import. Making it a module-level singleton means: (a) it loads once per process, not once per request; (b) it's shared across all concurrent invocations; (c) the async event loop is never blocked by a model load mid-request (it loads at import time, before the server starts accepting requests).

```python
async def memory_agent(state: AgentState) -> AgentState:
```

The function is `async` because it calls `await db.insert_capture(...)`. It takes the full AgentState and returns the full AgentState (with additions). LangGraph calls this function as a graph node.

**Step 1 — Embed the capture:**
```python
capture_embedding = _embedder.encode(state["capture"]["raw_content"]).tolist()
```
`_embedder.encode()` is synchronous (runs in the calling thread). It returns a numpy array of shape `(384,)`. `.tolist()` converts to a Python list of floats, which asyncpg can serialize to pgvector's `vector(384)` type. The embedding represents the semantic meaning of the entire raw content.

**Step 2 — Insert to captures table:**
```python
capture_id = await db.insert_capture(
    id=str(uuid.uuid4()),
    run_id=state["run_id"],
    modality=state["capture"]["modality"],
    ...
    embedding=capture_embedding,
    metadata=state["capture"].get("metadata", {}),
)
```
A UUID is generated here (not in the DB) so that the application controls ID generation. `db.insert_capture` returns the UUID string of the inserted row.

**Step 3 — Embed and insert each item:**
```python
for item in state["extracted_items"]:
    item_text = f"{item['title']} {item['description']}"
    item_embedding = _embedder.encode(item_text).tolist()
    item_id = await db.insert_extracted_item(capture_id=capture_id, embedding=item_embedding, **item)
    item_ids.append(item_id)
```
Each item gets its own embedding. The text fed to the embedder is `title + " " + description` — a deliberate choice to give the model a complete semantic representation of the item (title alone is often too short; description adds context).

**Step 4 — H2 assertion:**
```python
assert len(item_ids) == len(state["extracted_items"]), "Item ID count mismatch"
```
This is a hard invariant check. If any insert silently failed or returned None, this assertion catches it immediately rather than letting downstream agents work with a corrupted state (mismatched indices between `extracted_items` and `extracted_item_ids`).

**Step 5 — Return updated state:**
```python
trace_entry = {"agent": "memory", "timestamp": now, "capture_id": capture_id, "item_count": len(item_ids)}
return {**state, "capture_id": capture_id, "extracted_item_ids": item_ids, "trace": state["trace"] + [trace_entry]}
```
The `{**state, ...}` spread pattern preserves all existing state fields and overwrites/adds only the new ones. This is idiomatic LangGraph node return style.

---

### 2.3 PostgreSQL Schema — Design Decisions

#### captures table
```sql
CREATE TABLE captures (
    id          UUID PRIMARY KEY,
    run_id      UUID NOT NULL,
    modality    TEXT NOT NULL,          -- text | pdf | image
    source      TEXT NOT NULL,          -- origin of the capture
    raw_content TEXT NOT NULL,          -- the actual text
    file_path   TEXT,                   -- nullable: only for file inputs
    embedding   vector(384),            -- pgvector column
    metadata    JSONB DEFAULT '{}',     -- flexible key-value store
    deleted_at  TIMESTAMP               -- soft delete: NULL = active
);
```

**Design notes:**
- `embedding vector(384)` uses the pgvector extension. The `384` matches `all-MiniLM-L6-v2`'s output dimension.
- `metadata JSONB` is intentionally flexible. Different capture sources (Slack, email, voice) may have different metadata shapes. Using JSONB avoids premature schema normalization.
- `deleted_at TIMESTAMP` implements soft delete (H7 pattern). Rows are never physically deleted. `get_capture` always filters `WHERE deleted_at IS NULL`.

#### extracted_items table
```sql
CREATE TABLE extracted_items (
    id               UUID PRIMARY KEY,
    capture_id       UUID REFERENCES captures(id),
    item_type        TEXT,
    title            TEXT,
    description      TEXT,
    confidence_score FLOAT,
    urgency_score    FLOAT,
    deadline         TIMESTAMP,
    entities         JSONB,
    metadata         JSONB,
    planning_status  TEXT DEFAULT 'pending',  -- pending | planned | skipped
    embedding        vector(384)
);
```

`planning_status` starts as `'pending'` and is updated by the planning agent via `update_extracted_item_planning_status`. This allows the hub page to show items that haven't been planned yet.

`get_items_by_capture` orders by `urgency_score DESC` — the planning agent sees the most urgent items first.

#### actions table
```sql
CREATE TABLE actions (
    id                   UUID PRIMARY KEY,
    extracted_item_id    UUID REFERENCES extracted_items(id),
    action_type          TEXT,
    payload              JSONB,
    requires_approval    BOOL DEFAULT false,
    status               TEXT DEFAULT 'pending',
    depends_on_action_id UUID REFERENCES actions(id)  -- self-referential FK, nullable
);
```

The self-referential `depends_on_action_id` models action dependencies in the DB while keeping the AgentState `actions` list flat. This separation is intentional: the graph state doesn't need to represent the dependency graph — the DB handles it for execution ordering.

#### decisions table
```sql
CREATE TABLE decisions (
    id               UUID PRIMARY KEY,
    action_id        UUID REFERENCES actions(id) ON DELETE SET NULL,
    decision         TEXT,              -- approved | rejected | edited
    edited_payload   JSONB,            -- what the human changed
    final_payload    JSONB,            -- the payload that was actually executed
    execution_result JSONB,
    decided_at       TIMESTAMP
);

CREATE UNIQUE INDEX decisions_action_id_unique 
    ON decisions(action_id) 
    WHERE action_id IS NOT NULL;
```

**H5 pattern — split audit fields:** The decisions table separates the audit trail (`edited_payload`, `final_payload`, `execution_result`) from the action itself. This means you can reconstruct exactly what a human changed and what was finally executed, independently of the action's current state.

**`ON DELETE SET NULL`:** If an action is deleted, its decisions are not deleted — only `action_id` is set to NULL. This preserves the audit trail even after action cleanup (H7).

**Partial unique index:** `WHERE action_id IS NOT NULL` means: enforce at-most-one decision per action, but allow multiple rows where `action_id IS NULL` (orphaned decisions from deleted actions). A regular unique index would prevent this because NULL values don't compare equal in most DBs, but the partial index explicitly handles it.

#### runs table
```sql
CREATE TABLE runs (
    id         UUID PRIMARY KEY,
    capture_id UUID REFERENCES captures(id),
    status     TEXT,          -- running | completed | failed
    trace      JSONB,         -- final trace array from AgentState
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);
```

The `trace` JSONB column stores the final M4 trace array from AgentState when the run completes. This is the persistent counterpart to the in-memory `trace` list in AgentState.

---

### 2.4 asyncpg Connection Pool

```python
# Initialization
pool = await asyncpg.create_pool(
    dsn=DATABASE_URL,
    min_size=2,
    max_size=10,
    init=_init_connection
)

async def _init_connection(conn):
    # Register JSON/JSONB codecs
    await conn.set_type_codec('json',  encoder=json.dumps, decoder=json.loads, schema='pg_catalog')
    await conn.set_type_codec('jsonb', encoder=json.dumps, decoder=json.loads, schema='pg_catalog')
```

**Why a pool?** Each asyncpg connection is a TCP connection to PostgreSQL. Creating a new connection per request costs ~50–100ms. A pool maintains a set of live connections and loans them to concurrent coroutines. `min_size=2` means 2 connections are always warm. `max_size=10` caps resource usage.

**Why register JSON/JSONB codecs?** By default, asyncpg returns JSONB columns as strings. Registering `json.loads` as the decoder means Python code receives actual dicts/lists from JSONB columns without manual parsing. The encoder converts Python dicts to JSON strings for insert.

**Usage pattern in db.py:**
```python
async def insert_capture(...) -> str:
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "INSERT INTO captures (...) VALUES (...) RETURNING id",
            ...values...
        )
        return str(row["id"])
```
`async with pool.acquire()` borrows a connection from the pool and automatically returns it when the block exits (even on exception). `fetchrow` returns a single `asyncpg.Record`. `.RETURNING id` avoids a second round-trip to retrieve the generated UUID.

---

### 2.5 Key CRUD Operations

**`get_capture(capture_id)`** — Always adds `WHERE deleted_at IS NULL` to filter soft-deleted rows. Returns None if not found.

**`list_captures(page, page_size)`** — Paginated with `LIMIT` and `OFFSET`. Excludes soft-deleted.

**`soft_delete_capture(capture_id)`** — `UPDATE captures SET deleted_at = NOW() WHERE id = $1`. Does not touch `extracted_items` — those remain for audit purposes (H7).

**`get_items_by_capture(capture_id)`** — `ORDER BY urgency_score DESC` so planning sees high-urgency items first.

**`update_extracted_item_planning_status(item_id, status)`** — Called by planning agent. Status transitions: `pending → planned` or `pending → skipped`.

**`get_pending_actions()`** — `WHERE status = 'pending' AND requires_approval = true`. Used by the approval webhook handler to know what to present to the human.

**`get_decision_by_action(action_id)`** — Used to check if an action already has a decision (idempotency guard during multi-resume).

**`insert_decision(...)`** — Inserts to the decisions table. If the partial unique index constraint fires (duplicate action_id), the insert raises `asyncpg.UniqueViolationError`, which the caller catches as an idempotency signal.

---

### 2.6 Retrieval System

```python
async def assemble_retrieval_context(query: str, ...) -> list:
    query_embedding = _embedder.encode(query).tolist()
    
    # Level 1: semantic search on extracted_items
    items = await _search_items(query_embedding, filters...)
    
    # Level 2: semantic search on captures (fallback)
    captures = await _search_captures(
        query_embedding, 
        exclude_capture_ids=[i["capture_id"] for i in items]
    )
    
    # Enrich, merge, deduplicate, sort by semantic_score DESC
    return merged_results
```

**Two-level search rationale:**
- Level 1 (items) is precise — it finds specific actionable items that match the query semantically.
- Level 2 (captures) is a fallback — if no items match well enough, search the raw captures. This ensures some context is always returned even for queries about topics where no items were extracted.

**The pgvector query:**
```sql
SELECT *, 1 - (embedding <=> $1) AS semantic_score 
FROM extracted_items 
WHERE 1 - (embedding <=> $1) > 0.3 
ORDER BY semantic_score DESC 
LIMIT $2
```

`<=>` is the pgvector cosine distance operator. Cosine distance ranges 0–2 (0 = identical, 2 = opposite). So `1 - distance` gives a similarity score where 1 = identical, -1 = opposite. The threshold `> 0.3` filters out weak matches (scores below 0.3 are likely unrelated). `ORDER BY semantic_score DESC` returns the most similar items first.

**Enrichment step:** After retrieval, each item dict is enriched with its parent capture (joined query) and any associated actions (separate query). This gives the LLM prompt full context: "this item came from this capture, and these actions were planned for it."

---

### 2.7 HNSW Indexes

```sql
CREATE INDEX ON captures USING hnsw (embedding vector_cosine_ops);
CREATE INDEX ON extracted_items USING hnsw (embedding vector_cosine_ops);
```

**HNSW = Hierarchical Navigable Small World.** A graph-based ANN (approximate nearest neighbor) index structure. At query time, instead of scanning all rows to compute cosine distance (O(n)), the index navigates a hierarchical graph to find approximate nearest neighbors in O(log n).

`vector_cosine_ops` tells pgvector to index for cosine distance (`<=>` operator). Other options: `vector_l2_ops` (Euclidean), `vector_ip_ops` (inner product). Cosine is appropriate here because sentence embeddings are unit-normalized — cosine similarity measures the angle between vectors regardless of magnitude.

**Why HNSW over IVFFlat (M5 pattern)?** IVFFlat requires a training phase (`VACUUM ANALYZE` after bulk insert) to build its clustering. HNSW builds incrementally — each inserted row updates the index immediately. At demo/prototype scale (hundreds to thousands of rows), IVFFlat may never be fully trained. HNSW just works at any table size.

---

## 3. LANGGRAPH CONCEPTS — STATE-FOCUSED

### 3.1 TypedDict as State Schema

LangGraph requires a state type that is dict-like. `TypedDict` is the standard choice because:

1. **It IS a dict at runtime.** `AgentState(input_content="hello", ...)` produces a regular Python dict. No `.dict()` call needed. LangGraph can serialize/deserialize it directly.
2. **Type hints are for developers, not runtime.** TypedDict adds zero runtime overhead — it's just `dict` with annotations. This means no `__init__` validation, no field coercion, no schema enforcement at runtime (unlike Pydantic).
3. **LangGraph expects dict-like objects.** The framework passes state dicts between nodes. TypedDict annotations help IDEs and type checkers catch bugs during development.

Pydantic BaseModel would require `.dict()` or `.model_dump()` at every node boundary, adding overhead and requiring all agents to know they're receiving a Pydantic model rather than a plain dict.

### 3.2 How StateGraph Validates State

LangGraph does not deeply validate state at runtime. When you call `graph.add_node("memory", memory_agent)`, LangGraph registers the function. When the node runs, LangGraph:
1. Passes the current state dict to the function
2. Receives the returned dict
3. Merges the returned dict into the current state using **reducers**

If a node returns a key that isn't in the TypedDict schema, LangGraph will still store it — TypedDict is advisory, not enforced. The real validation happens through your H2 assertions and the TypedDict type hints (caught by mypy/pyright during development).

### 3.3 Reducers — Default Replace

By default, LangGraph uses a **replace reducer**: when a node returns `{"capture_id": "some-uuid"}`, that value replaces whatever was in `state["capture_id"]` before. 

For list fields like `trace`, if you returned `{"trace": [new_entry]}`, it would **replace** the entire trace list with just the new entry. This is why every agent uses the append pattern:
```python
return {**state, "trace": state["trace"] + [new_entry]}
```

LangGraph supports custom reducers via `Annotated` types. For example, `Annotated[list, operator.add]` would auto-concatenate lists. Orbit uses manual append in each node instead, which makes the behavior explicit and auditable.

### 3.4 State Snapshots at Checkpoints

LangGraph creates a **checkpoint** after every node execution. A checkpoint is a snapshot of the full AgentState dict at that moment. Checkpoints are stored in the configured checkpointer (MemorySaver in Orbit's case).

When `interrupt_before=["approval"]` is set, LangGraph:
1. Runs all nodes up to (but not including) `approval`
2. Creates a checkpoint of the state just before `approval`
3. Pauses execution — the graph call returns with a special `INTERRUPT` state
4. When the approval webhook fires, LangGraph **resumes from that checkpoint**

This means the state is not re-computed from scratch on resume — it loads the snapshot and continues from where it stopped.

### 3.5 MemorySaver Internals

```python
from langgraph.checkpoint.memory import MemorySaver
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory, interrupt_before=["approval"])
```

MemorySaver is an in-memory dict keyed by `thread_id`. Its internal structure is roughly:
```python
{
    "thread_id_abc123": {
        "checkpoint_id_1": {state after node 1},
        "checkpoint_id_2": {state after node 2},
        ...
        "latest": "checkpoint_id_N"
    }
}
```

`thread_id` maps to `run_id` in AgentState. When you call `graph.invoke(state, config={"configurable": {"thread_id": run_id}})`, LangGraph uses `run_id` as the key to look up and store checkpoints.

**What gets checkpointed:** The full AgentState dict after each node. This means all 17 fields of AgentState are serialized to the MemorySaver after each agent runs. For large states (big `raw_content`, many items), this can be memory-intensive — which is why the M4 trace pattern keeps trace entries small.

**MemorySaver vs Redis/Postgres checkpointer:** MemorySaver lives in the process heap. If the server restarts, all checkpoints are lost. For Orbit's demo context, this is acceptable — a restart means all in-flight runs are lost, and users resubmit. A production system would use `AsyncPostgresSaver` or `AsyncRedisSaver` to persist checkpoints across restarts.

### 3.6 Resumption Mechanism

After an interrupt, resuming a graph:
```python
# Resume with the human's decision
result = await graph.ainvoke(
    {"decisions": [decision_record]},
    config={"configurable": {"thread_id": run_id}}
)
```

LangGraph:
1. Looks up the latest checkpoint for `thread_id = run_id`
2. Merges the input dict (`{"decisions": [...]}`) into the checkpointed state
3. Resumes execution from the interrupted node (`approval`)

The merged state is `{...checkpoint_state, "decisions": [decision_record]}`. This is how the human's decision gets into the graph without rerunning all previous nodes.

### 3.7 stream_mode="values" vs "updates"

**`stream_mode="values"`**: After each node runs, stream the **full** current AgentState. Every event contains the complete state. Good for displaying the full current state to a user in real time.

**`stream_mode="updates"`**: After each node runs, stream only the **diff** — the keys that changed. More bandwidth-efficient. Good for incremental UI updates.

In Orbit, `stream_mode="updates"` is used for the live pipeline visualization, so the frontend only receives the changed fields (e.g., after memory_agent: `{"capture_id": "...", "extracted_item_ids": [...], "trace": [...]}`).

### 3.8 How Nodes Communicate Via State (Not Direct Calls)

This is a fundamental LangGraph design principle. Nodes are **pure functions** of state. They do not call each other. They do not share any Python variables. The only way `planning_agent` knows what `memory_agent` did is by reading `extracted_item_ids` from the state dict.

This decoupling means:
- Nodes can run in any order defined by the graph edges
- Nodes can be tested in isolation by constructing a fake AgentState
- Adding a new agent doesn't require modifying existing agents — just define what it reads and writes

---

## 4. MULTI-AGENT CONCEPTS — MEMORY & STATE FOCUS

### 4.1 Shared State as Communication Bus

In Orbit, all 7 agents share a single AgentState dict. This is the **blackboard architecture** pattern: agents write their outputs to a shared board, and other agents read from it. No agent calls another directly.

The AgentState acts as an **asynchronous message bus** made synchronous by LangGraph's sequential execution. When memory_agent writes `capture_id`, it's like posting a message to a bus that planning_agent will later consume.

**Advantages over direct calls:**
- Agents remain loosely coupled — change memory_agent's internals without affecting planning_agent
- Easy to add logging/monitoring at the state level (the trace array)
- Easy to replay runs by replaying state transitions

**Disadvantages:**
- Large state objects can be memory-intensive with many agents
- No type enforcement at runtime — a typo in a key name fails silently

### 4.2 How Memory Agent Bridges In-Memory State to Persistent Store

Before memory_agent: everything is ephemeral (lives in LangGraph's in-memory state, lost on crash).

After memory_agent: the capture and its items are in PostgreSQL (survive crashes, restarts, and can be queried by the API independently of the graph run).

This bridge point is architecturally significant. It means:
- The API can serve captures and items independently of whether the LangGraph run is still in progress
- The hub page can display past captures even for completed runs
- The approval webhook can look up actions in the DB without accessing the graph state

The memory_agent is intentionally positioned early in the pipeline (after intent, before planning) so that the most fundamental data is persisted before any higher-level processing. If planning fails, you still have the capture and items.

### 4.3 State Accumulation Pattern — Trace Array

```python
# Pattern used by every agent
return {**state, "trace": state["trace"] + [new_entry]}
```

The trace array accumulates a lightweight audit log of what every agent did. By the end of the pipeline, `state["trace"]` contains one entry per agent:
```python
[
    {"agent": "understanding", "timestamp": "...", "modality": "text", ...},
    {"agent": "intent", "timestamp": "...", "item_count": 3, ...},
    {"agent": "memory", "timestamp": "...", "capture_id": "uuid", "item_count": 3},
    {"agent": "planning", "timestamp": "...", "action_count": 5, ...},
    ...
]
```

This trace is stored in the `runs.trace` JSONB column at the end of the run. It provides a human-readable audit log of the entire pipeline without storing full state dumps at each step (M4 pattern).

### 4.4 Context Grounding via retrieval_context

The `retrieval_context` field enables **context-augmented generation** for downstream agents. Instead of each LLM call operating only on the current capture, agents can include relevant past captures and items in their prompts.

For example, the planning agent might prompt:
```
Current items to plan: [extracted_items]
Relevant past context: [retrieval_context]
```

This means if a user captured "meeting with Sarah about Q3 budget" two weeks ago, and now captures "follow up with Sarah," the retrieval system surfaces the old capture as context, enabling the planning agent to generate a more informed action plan.

### 4.5 How run_id Enables Correlation

`run_id` = `thread_id` = LangSmith trace ID.

| System | How run_id is used |
|--------|-------------------|
| LangGraph | `config={"configurable": {"thread_id": run_id}}` — identifies the checkpoint |
| PostgreSQL | `captures.run_id` — find all captures for this run |
| PostgreSQL | `runs.id` — the run row itself |
| LangSmith | `thread_id` in trace config — links to the external trace |

This single UUID flowing through all systems means debugging is easy: given a run_id, you can find the LangSmith trace, the DB rows, and the checkpoint state.

---
