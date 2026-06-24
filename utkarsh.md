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

---

## 5. DESIGN DECISIONS AND DEFENSES

Each decision below is exam-ready: state the decision, the reasoning, and how to defend it against alternatives.

---

### 5.1 Why sentence-transformers not OpenAI embeddings?

**Decision:** Use `sentence-transformers/all-MiniLM-L6-v2` running locally.

**Reasoning:**
- **Free at runtime.** OpenAI embeddings cost ~$0.0001 per 1K tokens. For a personal chief-of-staff that embeds every capture, this adds up. Local embeddings have zero per-request cost.
- **No rate limits.** OpenAI has rate limits on the embeddings endpoint. Under concurrent load or batch processing, local embeddings never throttle.
- **No network dependency.** If OpenAI is down or the server has no internet access, the system still works.
- **384 dimensions is sufficient.** For Orbit's use case (semantic similarity of personal productivity items), 384-dim vectors provide adequate recall. OpenAI's `text-embedding-3-small` produces 1536 dims — more expensive to store and index with negligible real-world improvement at this scale.
- **Consistency.** The same model is used at write time (memory_agent) and at read time (retrieval). There is no risk of embedding drift from model version changes (which happens with OpenAI when they update models).

**Defense against "OpenAI embeddings are better quality":** For general-purpose NLP benchmarks, yes. For the specific domain of personal productivity (tasks, meetings, deadlines, follow-ups), `all-MiniLM-L6-v2` performs well. The quality gap doesn't justify the operational complexity and cost for a demo-scale system. In production, we could evaluate both and switch if recall metrics showed meaningful improvement.

---

### 5.2 Why pgvector not Pinecone/Weaviate/Chroma?

**Decision:** Store vectors in PostgreSQL via the `pgvector` extension.

**Reasoning:**
- **No extra infrastructure.** Orbit already requires PostgreSQL for relational data. Adding Pinecone means operating, securing, and paying for a second database service.
- **SQL joins.** With pgvector, a semantic search can join directly to relational data: `SELECT items.*, captures.source FROM extracted_items JOIN captures ON ...`. With a separate vector DB, you'd retrieve IDs from Pinecone, then query PostgreSQL for the related data — two round trips.
- **Transactional consistency.** An insert to `extracted_items` (including the embedding) is part of the same ACID transaction. With Pinecone, the vector insert is a separate call — if it fails after the SQL insert, you have an inconsistency (SQL row with no vector).
- **Simpler mental model.** Developers only need to know one database system. All schema, queries, indexes, and backups are in one place.

**Defense against "Pinecone scales better":** At demo scale (hundreds to low thousands of rows), HNSW in PostgreSQL is fast enough. The query latency for `LIMIT 10` with HNSW on 10,000 rows is in the low milliseconds. If Orbit ever reached millions of items, we'd re-evaluate — but that's a scale problem for a much later version.

---

### 5.3 Why HNSW not IVFFlat?

**Decision:** Use HNSW (Hierarchical Navigable Small World) indexes, not IVFFlat.

**Reasoning (M5 pattern):**
- **No training phase.** IVFFlat requires running `VACUUM ANALYZE` after a bulk insert to build its clustering. An IVFFlat index on an empty table is unusable. HNSW builds incrementally — the index is valid and useful from row 1.
- **Better recall at low table sizes.** IVFFlat's approximate nearest-neighbor accuracy degrades if the number of lists (`nlist`) is poorly tuned relative to table size. HNSW maintains consistent recall regardless of table size.
- **Dynamic updates.** HNSW handles inserts well. IVFFlat is optimized for static datasets; its recall can degrade as rows are added after index creation without retraining.

**The tradeoff:** HNSW uses more memory than IVFFlat (the graph structure is stored in-memory during queries). At millions of rows, this becomes significant. For Orbit's scale, it's irrelevant.

---

### 5.4 Why TypedDict not Pydantic BaseModel for AgentState?

**Decision:** Use `TypedDict` from Python's standard library.

**Reasoning:**
- **JSON-serializable without conversion.** A TypedDict instance is a plain `dict`. LangGraph serializes state to JSON for checkpointing — no `.dict()` or `.model_dump()` call needed.
- **LangGraph expects dict-like objects.** The `StateGraph` type parameter is `TypedDict`. Using a Pydantic model requires extra adapter code.
- **Zero `__init__` overhead.** Pydantic validates all fields on construction. For a state object that's constructed and mutated many times per second across agents, this adds measurable overhead.
- **Simpler update pattern.** `{**state, "capture_id": new_id}` works naturally with TypedDict. With Pydantic, you'd need `state.copy(update={"capture_id": new_id})` or `state.model_copy(...)`.

**Defense against "Pydantic gives you validation":** Runtime validation is a double-edged sword in a pipeline. If upstream agents produce slightly malformed data, Pydantic raises an exception. TypedDict lets the error surface where it actually causes a problem (e.g., when memory_agent tries to access a missing key). The explicit H2 assertion is a targeted invariant check at the one place where misalignment would cause silent corruption.

---

### 5.5 Why asyncpg not SQLAlchemy async?

**Decision:** Use `asyncpg` directly with raw SQL.

**Reasoning:**
- **Performance.** asyncpg is benchmarked as one of the fastest PostgreSQL clients available in Python. It communicates using the PostgreSQL binary protocol directly, avoiding parsing overhead.
- **Native async.** asyncpg was designed for async from the ground up — not bolted on. Every operation is a proper coroutine.
- **No ORM overhead.** SQLAlchemy adds a query builder, ORM object construction, relationship lazy-loading, and event hooks. For Orbit's straightforward CRUD operations, this is pure overhead.
- **Direct SQL control.** The pgvector `<=>` cosine distance operator is not abstracted by SQLAlchemy. Writing raw SQL for the vector search queries is cleaner than fighting the ORM.
- **Simple mental model.** `db.py` is a thin wrapper with explicit SQL. Any developer can read a function and know exactly what query will run.

**Defense against "SQLAlchemy handles schema migrations":** True — but Orbit uses Alembic-compatible migration files directly. ORM-level migration isn't needed when you control the SQL.

---

### 5.6 Why soft deletes not hard deletes?

**Decision:** Implement deletes as `UPDATE captures SET deleted_at = NOW()` rather than `DELETE FROM captures`.

**Reasoning (H7 pattern):**
- **Audit trail.** A deleted capture is still in the DB. You can query `WHERE deleted_at IS NOT NULL` to see what was deleted and when.
- **Decisions outlive captures.** If a capture is deleted but had associated decisions (via actions), those decisions remain. With hard deletes, the FK cascade would delete the entire chain. With soft deletes + `ON DELETE SET NULL`, decisions survive with `action_id = NULL`.
- **Accidental deletion recovery.** Restoring a soft-deleted row is `UPDATE captures SET deleted_at = NULL WHERE id = $1`. Restoring a hard-deleted row requires a DB backup restore.
- **Consistency with decisions table.** The decisions table uses `ON DELETE SET NULL` on `action_id`. This only works if captures are never hard-deleted (which would cascade to delete actions, which sets `action_id = NULL` on decisions). Soft deletes make this non-issue — actions are never hard-deleted either.

---

### 5.7 Why a separate decisions table?

**Decision:** Store human decisions in a `decisions` table, separate from the `actions` table.

**Reasoning (H5 pattern):**
- **Separation of concerns.** An action describes what to do. A decision describes what a human chose to do about it (approve, reject, edit). These are different entities with different lifecycles.
- **Actions can be re-decided.** If a decision is reversed (future feature), you can insert a new decisions row and archive the old one, without mutating the action.
- **Audit fields are rich.** Decisions store `edited_payload` (what the human changed), `final_payload` (what was actually executed), and `execution_result`. Cramming these into the actions table would mix operational state with audit history.
- **ON DELETE SET NULL semantics.** When an action is deleted, the decision row survives (with `action_id = NULL`). This preserves the audit trail — you know a human approved or rejected something, even if the action itself no longer exists.

---

### 5.8 Why partial unique index on decisions(action_id)?

**Decision:** `CREATE UNIQUE INDEX ... ON decisions(action_id) WHERE action_id IS NOT NULL`

**Reasoning:**
- **Enforce at-most-one decision per action.** The unique index prevents a race condition where two concurrent approval webhooks insert two decisions for the same action.
- **Allow multiple NULL rows.** A regular `UNIQUE` constraint treats NULL as distinct (so multiple NULL values are allowed in most databases). However, the partial index with `WHERE action_id IS NOT NULL` makes the intent explicit: we want uniqueness only for non-null action_ids.
- **Orphaned decisions.** When an action is deleted, `action_id` becomes NULL. Multiple orphaned decisions (from different deleted actions) would all have `action_id = NULL` — the partial index allows this correctly.

---

### 5.9 Why MemorySaver not Redis/Postgres checkpointer?

**Decision:** Use LangGraph's in-memory `MemorySaver` for checkpointing.

**Reasoning:**
- **Appropriate for demo scale.** MemorySaver works perfectly for a single-server demo. No additional infrastructure is needed.
- **Performance.** In-memory reads/writes are nanoseconds vs. milliseconds for Redis or Postgres. For rapid graph execution (no human-in-the-loop), MemorySaver adds zero latency.
- **Simplicity.** The tradeoff (no persistence across restarts) is acceptable for a demo. The data that matters (captures, items, actions, decisions) is persisted in PostgreSQL via the memory_agent. If the server restarts, in-flight runs are lost, but all completed runs' data is preserved.

**Defense against "What about production?":** In production, replace MemorySaver with `AsyncPostgresSaver` (LangGraph's PostgreSQL checkpointer) — it stores checkpoints in a `checkpoints` table. The rest of the code is unchanged because the checkpointer is injected at `graph.compile()` time.

---

### 5.10 Why 384-dim vectors?

**Decision:** Use `all-MiniLM-L6-v2` which produces 384-dimensional vectors.

**Reasoning:**
- **Model default.** `all-MiniLM-L6-v2` outputs 384 dims. This is the natural output of the chosen model — there's no additional truncation or projection.
- **Storage efficiency.** A 384-dim float32 vector is 384 × 4 bytes = 1,536 bytes per row. A 1536-dim vector (OpenAI) would be 6,144 bytes per row — 4x more storage, 4x more HNSW index memory.
- **HNSW memory.** HNSW index memory scales with vector dimension. Smaller vectors = smaller index = faster search.
- **Sufficient for the domain.** Semantic similarity in the personal productivity domain (tasks, meetings, projects) is well-captured at 384 dims. The marginal improvement from higher dimensions is small for this specific semantic space.

---

## 6. VIVA QUESTION BANK

---

### 6.1 BEGINNER QUESTIONS (20)

**Q1. What is AgentState and why does Orbit use TypedDict?**

AgentState is a TypedDict — a Python standard library type that annotates a regular dict with field names and types. Orbit uses it because LangGraph passes state as a plain dict between nodes. TypedDict gives developers type-checked access to state fields (caught by mypy) while remaining a plain dict at runtime — no serialization overhead, no `.dict()` calls needed.

---

**Q2. What does the memory agent do?**

The memory agent is the first node that writes to the database. It: (1) embeds the raw capture content using SentenceTransformer, (2) inserts the capture to PostgreSQL, (3) embeds and inserts each extracted item, (4) writes `capture_id` and `extracted_item_ids` back to AgentState, (5) appends a trace entry. Before memory_agent runs, all data is in-memory only. After it runs, data is durably persisted.

---

**Q3. What is pgvector?**

pgvector is a PostgreSQL extension that adds a `vector(n)` column type and approximate nearest-neighbor search operators. It enables storing and querying high-dimensional embeddings directly in PostgreSQL, eliminating the need for a separate vector database like Pinecone.

---

**Q4. What is a vector embedding?**

A vector embedding is a fixed-size array of floating-point numbers that represents the semantic meaning of a piece of text. Words or phrases with similar meanings produce similar vectors (small cosine distance). The embedding model (`all-MiniLM-L6-v2`) converts text to a 384-dimensional vector.

---

**Q5. What is the `run_id` field used for?**

`run_id` is a UUID that ties together a single pipeline execution across: LangGraph (as `thread_id` for the MemorySaver checkpoint), PostgreSQL (as the `run_id` FK in captures and as the PK in the runs table), and LangSmith (as the trace thread ID). It's the correlation key for debugging any specific run.

---

**Q6. What is a soft delete?**

A soft delete sets `deleted_at = NOW()` on a row instead of running `DELETE`. The row stays in the database but is excluded from normal queries via `WHERE deleted_at IS NULL`. Benefits: audit trail, easy recovery, and preservation of dependent records (decisions remain even after their parent actions' captures are "deleted").

---

**Q7. What does the asyncpg connection pool do?**

The pool maintains a set of live TCP connections to PostgreSQL (`min_size=2, max_size=10`). Instead of creating a new connection per database call (expensive: ~50–100ms), code borrows a connection from the pool, uses it, and returns it. Under concurrent load, up to 10 connections can be active simultaneously.

---

**Q8. What is HNSW?**

Hierarchical Navigable Small World — a graph-based data structure for approximate nearest-neighbor search. It allows finding vectors that are semantically similar to a query vector without scanning every row in the table. Query time is O(log n) instead of O(n). Used for the pgvector indexes on `captures` and `extracted_items`.

---

**Q9. What is the `trace` field in AgentState?**

A list of lightweight audit entries. Every agent appends one dict: `{agent, timestamp, ...summary fields}`. By pipeline end, `trace` contains a record of what every agent did. It's stored in the `runs.trace` JSONB column. It's deliberately lightweight (M4 pattern) — just summaries, not full state dumps.

---

**Q10. What are the 5 tables in Orbit's schema?**

1. `captures` — raw input captures with embeddings
2. `extracted_items` — structured items parsed from captures, with embeddings
3. `actions` — planned actions linked to items
4. `decisions` — human approval/rejection records for actions
5. `runs` — pipeline run metadata with final trace

---

**Q11. What is the H2 assertion in memory_agent?**

`assert len(item_ids) == len(state["extracted_items"])`. This verifies that every extracted item was successfully inserted and returned an ID. If any insert silently failed, this assertion catches it immediately, preventing a silent misalignment where `extracted_items[i]` doesn't correspond to `extracted_item_ids[i]`.

---

**Q12. What does `capture_id` represent?**

The UUID of the row inserted into the `captures` table by memory_agent. It's written to AgentState after the insert. Downstream agents (planning, approval) use it to link actions back to the original capture. It's also returned in API responses so the frontend can display the stored capture.

---

**Q13. What is `all-MiniLM-L6-v2`?**

A sentence embedding model from the `sentence-transformers` library. It's a small, fast transformer that converts text to 384-dimensional vectors. "MiniLM" indicates it's a distilled/compressed model. "L6" means 6 transformer layers. It's chosen for being free, local, and fast while providing good semantic similarity for English text.

---

**Q14. What is the difference between `extracted_items` (in state) and `extracted_items` (in the DB table)?**

In AgentState, `extracted_items` is a list of dicts produced by the intent agent — these are pre-DB, no IDs yet. The `extracted_items` DB table stores the same data post-insert, with UUIDs, embeddings, and `planning_status`. The memory agent bridges the two: it reads from state, inserts to DB, and writes the resulting IDs back to state.

---

**Q15. What is semantic search?**

Finding records that are semantically similar to a query, based on meaning rather than exact keyword matches. Implemented via: (1) embed the query text to a vector, (2) find DB rows whose embedding vector has small cosine distance to the query vector. Example: searching "follow up with Sarah" can surface a past capture "scheduled meeting with Sarah" even though no words match.

---

**Q16. What does `metadata JSONB` allow in the captures table?**

Flexible, schema-free key-value storage for capture-specific data. Different capture sources (email, Slack, voice memo) may have different metadata shapes (e.g., `{"sender": "...", "subject": "..."}` for email vs. `{"channel": "...", "thread_ts": "..."}` for Slack). JSONB avoids creating a column for every possible field.

---

**Q17. What is the append pattern for the trace array?**

```python
return {**state, "trace": state["trace"] + [new_entry]}
```
Since LangGraph's default reducer replaces values, returning just `[new_entry]` would erase previous trace entries. By spreading the existing state and building a new list with the old entries plus the new one, the trace accumulates across all agents.

---

**Q18. What does `planning_status TEXT DEFAULT 'pending'` in extracted_items do?**

It tracks whether the planning agent has processed an item. Starts as `'pending'`. The planning agent updates it to `'planned'` or `'skipped'` via `update_extracted_item_planning_status`. The hub page uses this to show items that haven't been actioned yet.

---

**Q19. What is a UUID and why use it for primary keys?**

A Universally Unique Identifier — a 128-bit random value, formatted as `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`. Used for PKs because: (1) IDs can be generated in the application layer (before the DB insert), enabling the ID to be placed in AgentState immediately; (2) UUIDs don't reveal row count or creation order; (3) no coordination needed between multiple app instances.

---

**Q20. What does `RETURNING id` do in asyncpg queries?**

It tells PostgreSQL to return the `id` column of the newly inserted row as part of the INSERT response. This avoids a second round-trip query to retrieve the UUID after insert. asyncpg's `conn.fetchrow("INSERT ... RETURNING id", ...)` returns an `asyncpg.Record` containing the new row's ID.

---

### 6.2 INTERMEDIATE QUESTIONS (20)

**Q21. Walk through what happens in memory_agent step by step.**

1. Load lazy-singleton embedder (no-op if already loaded).
2. Encode `capture["raw_content"]` → 384-dim numpy array → `.tolist()`.
3. `await db.insert_capture(...)` — inserts to `captures` table, returns UUID string.
4. For each item in `extracted_items`: encode `title + description`, `await db.insert_extracted_item(...)`.
5. Assert `len(item_ids) == len(extracted_items)`.
6. Build trace entry with agent name, timestamp, capture_id, item_count.
7. Return `{**state, capture_id=..., extracted_item_ids=..., trace=[...+new_entry]}`.

---

**Q22. How does the retrieval system decide which items are "relevant enough"?**

The cosine similarity threshold `> 0.3` filters results. Items with `1 - (embedding <=> query_embedding) <= 0.3` are considered semantically too distant to be useful. This value was chosen empirically — values below ~0.3 typically represent unrelated content. Above 0.3, there's meaningful semantic overlap. The system then sorts remaining results by `semantic_score DESC` and applies `LIMIT`.

---

**Q23. Explain the two-level retrieval design.**

Level 1 searches `extracted_items` — specific, structured, high-signal results. If a query matches known items semantically, those are the best context. Level 2 searches `captures` (excluding captures already represented in Level 1 results via `exclude_capture_ids`) — broader, raw-content results for when no items match well. Level 2 acts as a fallback ensuring some relevant context is always surfaced.

---

**Q24. How does `run_id` flow through the system?**

It's set when the pipeline is invoked (generated by the API layer before calling `graph.invoke`). It flows as: `AgentState.run_id` → `db.insert_capture(run_id=...)` → stored in `captures.run_id` → `db.insert_run(id=run_id, ...)` → stored in `runs.id`. Also used as `thread_id` in `graph.invoke(config={"configurable": {"thread_id": run_id}})`, which is the MemorySaver key and the LangSmith trace thread ID.

---

**Q25. What is the partial unique index on decisions, and what problem does it solve?**

```sql
CREATE UNIQUE INDEX decisions_action_id_unique 
ON decisions(action_id) 
WHERE action_id IS NOT NULL;
```
It enforces at-most-one decision per action (preventing duplicate approvals in race conditions) while allowing multiple rows with `action_id = NULL` (orphaned decisions from deleted actions). A regular unique index would prevent this because NULL uniqueness is inconsistent across databases. The `WHERE action_id IS NOT NULL` makes intent explicit.

---

**Q26. How does the memory agent handle an empty `extracted_items` list?**

The `for item in state["extracted_items"]` loop simply doesn't execute. `item_ids` remains `[]`. The H2 assertion `assert len([]) == len([])` passes. `extracted_item_ids` is set to `[]` in state. The capture is still inserted — a capture with zero items is valid (e.g., a purely informational note).

---

**Q27. What is the difference between `pending_action_ids` and `decided_action_ids`?**

`pending_action_ids`: actions that require human approval and have not yet been decided. Set by tool_router.
`decided_action_ids`: actions that have received a human decision (approve/reject/edit). Updated during multi-resume approval flows (H3 pattern).

The H3 pattern tracks both lists so that when the graph is resumed multiple times (one decision per resume), it can identify which actions still need decisions vs. which are already decided, preventing double-processing.

---

**Q28. Why is `_embedder` a module-level variable?**

Loading a SentenceTransformer model reads ~90MB from disk and initializes model weights in RAM. If `_embedder` were created inside `memory_agent`, it would reload on every invocation — hundreds of milliseconds of latency per request. As a module-level singleton, it loads once at import time and is reused. The "lazy" aspect means it loads on first import of the module, not at Python startup.

---

**Q29. What does `async with pool.acquire() as conn` do?**

It's an async context manager that: (1) waits for an available connection from the pool (or waits until one is returned if all are in use), (2) returns the connection as `conn`, (3) automatically returns the connection to the pool when the `with` block exits — even if an exception is raised. This ensures connections are never leaked.

---

**Q30. How does the decisions table handle the case where the associated action is deleted?**

The FK is defined as `REFERENCES actions(id) ON DELETE SET NULL`. When an action row is deleted, PostgreSQL automatically sets `decisions.action_id = NULL` for all related decision rows. The decision row itself is preserved. This means the audit record "a human approved this action" survives even if the action is cleaned up, but the link back to the specific action is severed.

---

**Q31. What is the `init` parameter in `asyncpg.create_pool`?**

`init=_init_connection` specifies an async function to call on each new connection before it's added to the pool. In Orbit, `_init_connection` registers JSON/JSONB codecs:
```python
await conn.set_type_codec('jsonb', encoder=json.dumps, decoder=json.loads, schema='pg_catalog')
```
This runs once per connection (not once per query), so the codec registration overhead is amortized.

---

**Q32. How would you explain the difference between cosine similarity and Euclidean distance for embeddings?**

Cosine similarity measures the angle between two vectors — it's 1 when vectors point in the same direction, 0 when perpendicular, -1 when opposite. It ignores magnitude. Euclidean distance measures straight-line distance — it depends on both angle and magnitude. For sentence embeddings, cosine is preferred because the models produce unit-normalized vectors (magnitude = 1), so direction (meaning) is what matters, not scale.

---

**Q33. What is the significance of `vector_cosine_ops` in the HNSW index creation?**

It tells pgvector which distance metric the index should optimize for. `vector_cosine_ops` matches the `<=>` cosine distance operator used in queries. Using the wrong `ops` (e.g., creating an `l2_ops` index but querying with `<=>`) causes PostgreSQL to skip the index and do a sequential scan — catastrophic for performance at scale.

---

**Q34. Explain the state spread pattern `{**state, "new_key": value}`.**

This creates a new dict containing all key-value pairs from `state`, with `"new_key"` added or overwritten. In LangGraph, nodes must return a dict of updates. The spread ensures all existing state fields are preserved while adding or updating specific fields. Without the spread, returning only `{"capture_id": "..."}` would cause LangGraph to merge only that key, potentially leaving other fields unset depending on the reducer.

---

**Q35. What does `get_items_by_capture` return and in what order?**

All extracted_items rows where `capture_id` matches the given ID, ordered by `urgency_score DESC`. High-urgency items appear first, so the planning agent (which reads this list) prioritizes them. Items are excluded if `deleted_at IS NOT NULL` (if soft-delete were applied to items — though in current schema only captures have `deleted_at`).

---

**Q36. How does the memory agent connect to LangSmith tracing?**

LangSmith traces are keyed by `thread_id`, which equals `run_id`. Every `await db.*` call made within a LangGraph node is automatically wrapped in the LangSmith trace (LangGraph instruments async calls within nodes). The memory agent's trace entry in AgentState (`{"agent": "memory", "capture_id": "...", ...}`) also gets stored in the `runs.trace` JSONB column, providing a parallel audit trail outside LangSmith.

---

**Q37. What happens to the retrieval_context if no items meet the similarity threshold?**

Level 1 returns an empty list. Level 2 then searches captures with no `exclude_capture_ids`. If captures also have low similarity, Level 2 also returns empty. The final `retrieval_context` is `[]`. Downstream agents must handle an empty `retrieval_context` gracefully — their prompts should not fail if no historical context is available. For the very first capture in a new account, this is always the case.

---

**Q38. Why is item embedding done on `title + description` rather than just `title`?**

Title alone is often too short for meaningful embedding. "Meeting with Sarah" has very similar embedding to "Call with Sarah" — the description disambiguates: "Meeting with Sarah to discuss Q3 budget" vs. "Call with Sarah about technical interview." Concatenating both fields gives the model enough context to produce discriminating embeddings. The description is the semantic payload; the title is just a label.

---

**Q39. What is the M4 trace pattern and why does it matter for MemorySaver?**

M4 (Minimal Trace) means each agent appends only a small summary dict to the trace, not a full state dump. Contrast with a naive approach: store the entire AgentState after every node. If `raw_content` is 10KB and there are 7 nodes, a naive trace would store 70KB per run in MemorySaver. With M4, the trace stores only metadata (agent name, timestamp, a few key IDs), keeping MemorySaver memory usage small and checkpoint serialization fast.

---

**Q40. What is the dependency chain between tables?**

```
captures (root)
    └── extracted_items (capture_id FK → captures.id)
            └── actions (extracted_item_id FK → extracted_items.id)
                    └── decisions (action_id FK → actions.id, ON DELETE SET NULL)
runs (capture_id FK → captures.id, loosely)
```
Deleting a capture soft-deletes it (no cascade). Actions have a self-referential `depends_on_action_id` FK, enabling dependency chains within the same table.

---

### 6.3 ADVANCED QUESTIONS (20)

**Q41. How would you handle embedding model version migration?**

The problem: if you upgrade from `all-MiniLM-L6-v2` to a new model with a different embedding space, all stored embeddings are incompatible with new query embeddings. Solutions: (1) **Re-embed**: run a migration script that re-encodes all stored `raw_content` and `title+description` fields with the new model. Requires a maintenance window. (2) **Dual-index**: maintain two embedding columns during a transition period, query both, merge results, then drop the old column. (3) **Version column**: add `embedding_model TEXT` to captures and items, filter queries to only compare against rows with the same model version. Option 3 is most operationally safe but adds query complexity.

---

**Q42. How would you scale the connection pool for high concurrent load?**

Current: `min_size=2, max_size=10`. Under high load: (1) Increase `max_size` — but PostgreSQL has a hard connection limit (~100–200 by default for `max_connections`). (2) Deploy **PgBouncer** (connection pooler at the PostgreSQL level) — allows many app connections to multiplex onto fewer PostgreSQL connections. (3) Use **read replicas** — route read queries (retrieval, get_capture) to replicas, write queries (insert_capture, insert_decision) to primary. (4) Tune `statement_timeout` and `connect_timeout` in asyncpg to fail fast under overload rather than piling up waiting connections.

---

**Q43. What are the consistency guarantees of memory_agent?**

Within memory_agent: all inserts happen sequentially in separate `pool.acquire()` blocks. There is no outer transaction wrapping capture + items. This means: if `insert_capture` succeeds but `insert_extracted_item` fails, you have a capture with no items in the DB. The H2 assertion catches this at the Python level, but the DB is already inconsistent. To fix: wrap all inserts in a single transaction using `async with pool.acquire() as conn: async with conn.transaction():`. The current design trades strict consistency for simplicity (acceptable for a demo).

---

**Q44. How does the H3 multi-resume pattern work in detail?**

The graph is interrupted before the `approval` node with `interrupt_before=["approval"]`. On first resume, the human provides a decision for action A. The approval node adds A to `decided_action_ids` and checks if all `pending_action_ids` are now decided. If not, it interrupts again. On second resume, the human provides a decision for action B. The approval node adds B to `decided_action_ids`. This continues until `set(pending_action_ids) == set(decided_action_ids)`. At that point, the approval node allows execution to proceed. `decided_action_ids` prevents re-processing already-decided actions on each resume.

---

**Q45. What are the implications of using `encode()` synchronously inside an async function?**

`_embedder.encode()` is a CPU-bound synchronous operation. When called inside `async def memory_agent`, it blocks the event loop for its duration (typically 10–50ms for short text on CPU). During this block, no other coroutines can run. For a single-user demo, this is acceptable. For concurrent multi-user scenarios: wrap the call in `asyncio.get_event_loop().run_in_executor(None, _embedder.encode, text)` to offload to a thread pool, freeing the event loop while encoding runs.

---

**Q46. How would you implement idempotent memory_agent invocations?**

Idempotency means running memory_agent twice with the same input produces the same outcome (no duplicate DB rows). Current implementation: not idempotent — two runs with the same `run_id` would insert duplicate captures. To fix: (1) Check `SELECT id FROM captures WHERE run_id = $1` before inserting. If exists, return the existing ID. (2) Use `INSERT ... ON CONFLICT (run_id) DO NOTHING RETURNING id` — but this requires a unique constraint on `run_id` in captures. (3) Hash the `raw_content` and use it as a deterministic ID. Option 2 is cleanest but requires the assumption that each run_id maps to exactly one capture.

---

**Q47. Describe the full lifecycle of a single extracted item from text input to semantic retrieval.**

1. User inputs text → understanding agent sets `capture = {raw_content: "meeting with Sarah tomorrow at 2pm about Q3"}`.
2. Intent agent parses → `extracted_items = [{title: "Meeting with Sarah", description: "Q3 discussion tomorrow 2pm", urgency_score: 0.8, deadline: ...}]`.
3. Memory agent encodes `"Meeting with Sarah Q3 discussion tomorrow 2pm"` → 384-dim vector.
4. Inserts to `extracted_items` table with embedding, returns UUID.
5. Writes UUID to `extracted_item_ids` in state.
6. Later: user inputs "what did I schedule with Sarah?" → retrieval system encodes query → pgvector `<=>` cosine search finds this item (high similarity) → returned in `retrieval_context` → used in planning agent's LLM prompt.

---

**Q48. How would you add full-text search alongside semantic search?**

PostgreSQL supports both. Add a `tsvector` column to `extracted_items`:
```sql
ALTER TABLE extracted_items ADD COLUMN search_vector tsvector 
    GENERATED ALWAYS AS (to_tsvector('english', title || ' ' || description)) STORED;
CREATE INDEX ON extracted_items USING gin(search_vector);
```
In retrieval: run both `WHERE search_vector @@ plainto_tsquery($1)` (keyword) and `ORDER BY embedding <=> $2` (semantic). Merge and rerank by combining scores. This handles cases where exact keyword matches are better than semantic similarity (e.g., searching for a specific proper noun that the embedding model might conflate with similar names).

---

**Q49. What would happen to the HNSW index if you bulk-inserted 100,000 rows at once?**

HNSW handles bulk inserts correctly — it inserts rows into the index one by one as they arrive. However, bulk inserts can be slow because each row updates the HNSW graph. Optimization: (1) Disable the index during bulk insert (`DROP INDEX`, bulk insert, `CREATE INDEX`). PostgreSQL builds HNSW from a full scan faster than incremental updates. (2) Use `SET maintenance_work_mem = '1GB'` before index creation to give HNSW more memory for building the graph. (3) Use `SET max_parallel_workers_per_gather = 4` to parallelize index build.

---

**Q50. How does AgentState flow relate to LangGraph's concept of "reducers"?**

LangGraph reducers define how a node's returned value is merged into the current state. Default: replace. If you annotate a field as `Annotated[list, operator.add]`, LangGraph auto-concatenates lists. Orbit uses manual append (`state["trace"] + [entry]`) for the trace instead of reducers, because: (1) it makes the behavior explicit — any developer reading the code knows the trace accumulates; (2) it avoids the `Annotated` syntax which requires importing from `langgraph.graph`; (3) it keeps all agents self-consistent without relying on graph-level configuration.

---

**Q51. What are the memory implications of storing embeddings in PostgreSQL vs. a vector DB?**

In PostgreSQL: vector data is stored on disk in the heap. The HNSW index is stored on disk but loaded into `shared_buffers` (PostgreSQL's buffer pool) during queries. For 10,000 rows × 384 dims × 4 bytes = ~15MB of raw vector data. HNSW index overhead is typically 2–5x the raw data size, so ~30–75MB. This is well within PostgreSQL's memory budget. For 10M rows, this becomes 15GB of raw vectors — at that point, a dedicated vector DB (with distributed storage, quantization, and tiered memory) becomes practical.

---

**Q52. How would you debug a situation where semantic search returns no results for a query that should match?**

Step 1: Verify the query embedding is non-zero and reasonable length — print `np.linalg.norm(query_embedding)` (should be ~1.0 for unit-normalized models).
Step 2: Run the raw pgvector query in psql with `EXPLAIN ANALYZE` — check if the HNSW index is being used.
Step 3: Lower the threshold temporarily (`> 0.1`) to see what scores are being produced — maybe the actual matches have score 0.25, just below 0.3.
Step 4: Verify the stored embeddings are valid — `SELECT id, array_length(embedding::float[], 1) FROM extracted_items LIMIT 5` — should all be 384.
Step 5: Verify the same model is used for storage and query by checking the module-level `_embedder` is the same instance.

---

**Q53. Explain how `ON DELETE SET NULL` is different from `ON DELETE CASCADE`.**

`ON DELETE CASCADE`: when a parent row is deleted, all child rows with a FK pointing to it are also deleted. Applied to decisions: deleting an action would delete its decisions — destroying the audit trail.
`ON DELETE SET NULL`: when a parent row is deleted, child rows' FK column is set to NULL. Applied to decisions: deleting an action sets `decisions.action_id = NULL` — the decision row survives, preserving the audit trail. The partial unique index accounts for multiple NULL values.

---

**Q54. How would you implement embedding caching?**

Two approaches: (1) **In-memory cache**: use `functools.lru_cache` on the embed function with text as the key — but strings are large and the cache fills quickly. Better: hash the text (`hashlib.sha256(text.encode()).hexdigest()`) and use the hash as the key. (2) **DB-level cache**: add a `content_hash TEXT, embedding vector(384)` to a separate `embedding_cache` table. Before encoding, query `SELECT embedding FROM embedding_cache WHERE content_hash = $1`. On miss, encode and insert. The DB cache persists across server restarts. Appropriate if the same content is captured multiple times (e.g., repeated meeting notes from the same template).

---

**Q55. What is the difference between `fetchrow`, `fetch`, and `execute` in asyncpg?**

`fetchrow(query, *args)` — returns one `asyncpg.Record` or None. Use for queries expected to return one row (e.g., `SELECT ... WHERE id = $1`).
`fetch(query, *args)` — returns a list of `asyncpg.Record`. Use for queries returning multiple rows (e.g., `SELECT * FROM extracted_items WHERE capture_id = $1`).
`execute(query, *args)` — executes the query and returns the status string (e.g., `"INSERT 0 1"`). Use for INSERT/UPDATE/DELETE where you don't need the returned data. `executemany` is for batch operations.

---

**Q56. How does pgvector handle NULL embeddings?**

If a row has `embedding = NULL`, it's excluded from ANN index scans (the HNSW index only indexes non-NULL values). Cosine distance with NULL is also NULL, so `WHERE 1 - (embedding <=> $1) > 0.3` filters out NULL-embedding rows naturally. In Orbit, every memory_agent insert includes an embedding — NULL embeddings shouldn't occur in normal operation. They could occur if the encode step fails silently (returns NaN or zero vector), which is why the SentenceTransformer is called synchronously and exceptions propagate to LangGraph's error handler.

---

**Q57. What is the purpose of `extracted_items.entities JSONB`?**

It stores structured entity references specific to each item. For example, a "meeting" item might have `entities = {"people": ["Sarah"], "location": "Conference Room B", "date": "2026-07-01"}`. This is distinct from `AgentState.extracted_entities`, which is the global entity extraction for the entire capture. The per-item entities allow fine-grained entity-to-item linking, enabling features like "show all items involving Sarah."

---

**Q58. How would you add rate limiting to the memory agent for high-throughput scenarios?**

The bottleneck is: (1) embedding (CPU-bound, blocks event loop), (2) DB inserts (I/O-bound, async). For rate limiting: use a semaphore to limit concurrent memory_agent invocations:
```python
_embed_semaphore = asyncio.Semaphore(4)  # max 4 concurrent embeddings
async def memory_agent(state):
    async with _embed_semaphore:
        embedding = await loop.run_in_executor(None, _embedder.encode, text)
```
For DB inserts, the asyncpg pool's `max_size=10` is the natural rate limit — the 11th concurrent request waits for a connection. To surface waiting time as a metric, wrap `pool.acquire()` with a timer.

---

**Q59. What happens to AgentState if the memory agent raises an exception?**

LangGraph propagates the exception up from the graph invocation. The state is not written back (the exception prevents `return {**state, ...}` from executing). The MemorySaver retains the checkpoint from just before memory_agent ran. The `runs` table row either wasn't inserted yet (if memory is the first DB write for the run) or is left in `status = 'running'`. The capture insert may or may not have completed depending on where the exception occurred — this is the atomicity gap described in Q43.

---

**Q60. How would you migrate from MemorySaver to AsyncPostgresSaver in production?**

1. Install `langgraph-checkpoint-postgres`.
2. Create the checkpoints table: `await AsyncPostgresSaver.acreate_tables(conn)`.
3. Change one line at graph compile time:
```python
# Before
checkpointer = MemorySaver()
# After  
checkpointer = await AsyncPostgresSaver.from_conn_string(DATABASE_URL)
```
4. All in-flight runs using MemorySaver are lost (they're in-memory). Since Orbit persists all meaningful data to PostgreSQL via memory_agent, the only loss is the checkpoint for resumption of interrupted runs. New runs will checkpoint to PostgreSQL.
5. No changes to any agent code — the checkpointer is injected at compile time, fully transparent to nodes.

---

### 6.4 LANGGRAPH STATE QUESTIONS (20)

**Q61. What is a StateGraph in LangGraph?**

A directed graph where nodes are Python functions (agents) and edges define execution flow. The graph takes a TypedDict as its state type. Each node receives the full state, executes, and returns a partial state dict. LangGraph merges the returned dict into the current state using reducers, then passes the updated state to the next node.

---

**Q62. How does LangGraph know which node to run first?**

By `graph.set_entry_point("understanding")` or `graph.add_edge(START, "understanding")`. The `START` node is a special built-in representing graph entry. The first real node in the execution order is whichever node `START` has an edge to.

---

**Q63. What is a conditional edge in LangGraph and where does Orbit use one?**

A conditional edge calls a Python function on the current state to determine which node to route to next. In Orbit, after the `intent` node:
```python
def route_after_intent(state: AgentState) -> str:
    if state["clarification_needed"]:
        return "clarification"
    return "memory"
```
If `clarification_needed` is True (set by intent agent), execution routes to the clarification node instead of memory_agent. This is the `clarification_needed` field in AgentState enabling Daksh's routing logic.

---

**Q64. What is the difference between `graph.invoke()` and `graph.stream()`?**

`invoke()` runs the graph to completion and returns the final state. `stream()` is an async generator that yields state updates after each node. `stream()` with `stream_mode="updates"` yields `{node_name: {changed_keys: values}}` after each node. This enables real-time pipeline visualization in the Orbit frontend — the UI can show "memory agent: done" as each agent completes.

---

**Q65. How does `interrupt_before` work in LangGraph?**

When `graph.compile(interrupt_before=["approval"])` is called, LangGraph checks before executing the `approval` node whether it should pause. It saves a checkpoint and raises an `Interrupt` signal. The calling code (the API endpoint) catches this signal, stores the run state, and returns a response to the client indicating approval is needed. The graph does not advance until explicitly resumed.

---

**Q66. What is a thread in LangGraph's context?**

A thread is an isolated execution context identified by `thread_id`. Multiple runs of the graph can be concurrent, each with their own thread (and thus their own checkpoint history in MemorySaver). `thread_id` in Orbit equals `run_id`. The `config={"configurable": {"thread_id": run_id}}` parameter passed to `graph.invoke()` tells LangGraph which thread's checkpoints to read/write.

---

**Q67. Can two nodes in a LangGraph graph write to the same state key? What happens?**

Yes, and the result depends on the reducer. With the default replace reducer: the last node to run wins. With `Annotated[list, operator.add]`: both nodes' values are concatenated. In Orbit, `trace` is manually appended by each node. If two nodes ran concurrently (LangGraph supports parallel branches), both could try to append to `trace` simultaneously — the merge would depend on execution order. In Orbit's sequential graph, this isn't an issue.

---

**Q68. What does `{**state}` do in a node's return statement?**

It spreads all existing state key-value pairs into a new dict. Combined with additional keys: `{**state, "capture_id": "uuid"}` produces a dict with all existing fields preserved plus `capture_id` set/overwritten. When LangGraph receives this from a node, it merges (replaces) all keys. Because all existing keys are present in the returned dict, their values are "replaced" with the same values — effectively a no-op for unchanged fields, and an update for new/changed fields.

---

**Q69. What happens to the state if you don't include `{**state}` in a node's return?**

With the default replace reducer: if a node returns `{"capture_id": "uuid"}`, LangGraph merges only `capture_id` into state — all other fields remain unchanged. So technically, in LangGraph's default behavior, you don't need `{**state}`. The spread is a defensive pattern that: (1) makes the full state explicit in the returned dict, (2) works correctly with both default and custom reducer configurations, (3) clearly communicates intent to other developers.

---

**Q70. How does LangGraph handle a node that returns None?**

If a node returns `None`, LangGraph treats it as an empty update dict `{}` — no state fields are changed. This is valid and means the node ran but made no state changes. In Orbit, every node appends to trace, so returning None would lose the trace entry. All agents correctly return at least `{**state, "trace": state["trace"] + [entry]}`.

---

**Q71. What is a checkpointer's role during normal (non-interrupted) execution?**

After each node completes, the checkpointer stores the current state snapshot. This enables: (1) observability — you can inspect the state after any node by looking at the checkpoint history; (2) resumption — if the process crashes mid-run, the graph can resume from the last checkpoint; (3) time-travel — LangGraph allows re-running from any previous checkpoint (useful for debugging).

---

**Q72. How would you inspect the state at a specific node during debugging?**

```python
# Get all checkpoints for a thread
checkpoints = list(graph.get_state_history(config={"configurable": {"thread_id": run_id}}))
# Each checkpoint has .values (the state), .next (which node runs next), .created_at
for cp in checkpoints:
    print(cp.next, list(cp.values.keys()))
```
This gives the full state snapshot after each node, enabling precise debugging of where a field was set incorrectly.

---

**Q73. What is the difference between MemorySaver and a persistent checkpointer for LangGraph?**

MemorySaver stores all checkpoints in a Python `dict` in heap memory. Data is lost when the process exits. A persistent checkpointer (e.g., `AsyncPostgresSaver`) stores checkpoints in an external database (PostgreSQL or Redis). Data survives process restarts, enabling long-running workflows (days/weeks) and horizontal scaling (multiple server instances sharing the same checkpoint store).

---

**Q74. How does LangGraph handle graph edges when using `interrupt_before`?**

`interrupt_before=["approval"]` modifies the graph's execution so that before the `approval` node's incoming edge is followed, LangGraph checks whether to interrupt. The interrupt is independent of the edge definition — edges are defined normally, but execution pauses before the specified node(s) run. This means the graph topology is unchanged; only the runtime behavior is modified by the compile-time configuration.

---

**Q75. What is `stream_mode="values"` useful for in Orbit?**

It streams the full AgentState after every node. Useful for: (1) debugging — log the complete state after each step; (2) loading screen that shows current values (e.g., "3 items extracted"); (3) testing — verify state accumulates correctly. Less useful for production UI (sends large payloads) vs. `stream_mode="updates"` which sends only diffs.

---

**Q76. How would adding a new agent to the pipeline affect AgentState?**

Add new fields to the AgentState TypedDict for the new agent's inputs and outputs. Existing agents are unaffected — they don't touch the new fields. The new agent reads existing fields and writes new ones. The graph definition adds a new node and edges. Since TypedDict fields have no default values in vanilla Python (they're all required), you'd need to either: (1) initialize new fields to `None`/`[]`/`{}` in the entry state, or (2) use `total=False` for optional fields.

---

**Q77. What is the `configurable` dict in LangGraph's config?**

A dict of runtime-configurable parameters passed to `graph.invoke(input, config={"configurable": {...}})`. The most important is `"thread_id"` — this tells the checkpointer which thread's checkpoints to load/save. Other configurable values can include `"recursion_limit"` (max node executions to prevent infinite loops) and custom values accessed via `config["configurable"]` inside nodes.

---

**Q78. Can LangGraph nodes be async? Does Orbit use async nodes?**

Yes — LangGraph supports both sync and async nodes. `graph.invoke()` runs sync nodes; `graph.ainvoke()` runs async nodes. Orbit uses `graph.ainvoke()` because memory_agent (and other agents) use `await db.*` calls. All nodes must be either all-sync or all-async within a single `ainvoke` call. Mixing requires wrapping sync nodes in `asyncio.to_thread()`.

---

**Q79. What does LangGraph's `compile()` method do?**

It validates the graph (checks for unreachable nodes, missing edges, etc.), initializes the checkpointer, and returns a `CompiledGraph` object with methods: `invoke`, `ainvoke`, `stream`, `astream`, `get_state`, `get_state_history`. The compiled graph is the runnable version. You should compile once at server startup, not on every request.

---

**Q80. How does LangGraph know when the graph has completed?**

The graph reaches the `END` node. `END` is a special built-in like `START`. You add it as an edge: `graph.add_edge("execution", END)`. When the execution node completes and LangGraph follows the edge to `END`, it stops and returns the final state. If no `END` edge is reachable (bug in graph definition), LangGraph raises an error at compile time.

---

### 6.5 MEMORY & RAG QUESTIONS (20)

**Q81. What is RAG? How does Orbit implement it?**

Retrieval-Augmented Generation: augment an LLM's prompt with retrieved context from a knowledge base, reducing hallucination and enabling the model to reference specific past information. Orbit implements RAG via `assemble_retrieval_context`: encode the query, search pgvector for similar items/captures, include top results in the `retrieval_context` state field, which agents include in their LLM prompts.

---

**Q82. What is the difference between RAG and fine-tuning for Orbit's use case?**

Fine-tuning bakes knowledge into model weights — expensive, requires large labeled datasets, and becomes stale as new captures arrive. RAG retrieves knowledge at inference time — inexpensive, always up-to-date, and works with any base LLM. For a personal chief-of-staff that ingests new data daily, RAG is the only practical choice. Fine-tuning would need to be redone every time the user captures new information.

---

**Q83. What is cosine distance and why is it appropriate for semantic search?**

Cosine distance = 1 - cos(θ) where θ is the angle between two vectors. It measures direction similarity, ignoring magnitude. Sentence embedding models like `all-MiniLM-L6-v2` normalize output vectors to unit length, so cosine similarity reduces to the dot product. Two texts with similar semantic meaning produce vectors pointing in similar directions → low cosine distance → high similarity score.

---

**Q84. What is the "semantic gap" problem and how does the threshold `> 0.3` address it?**

The semantic gap: two texts about very different topics might still have a non-trivial cosine similarity (e.g., 0.1) because they share common words in the embedding space. Setting a threshold `> 0.3` filters out low-quality matches where the similarity is noise rather than genuine semantic overlap. The specific value (0.3) is empirical — it was chosen to balance precision (not too many irrelevant results) vs. recall (not filtering out genuinely related items).

---

**Q85. How would you evaluate the quality of Orbit's retrieval system?**

Standard RAG evaluation metrics:
- **Recall@K**: for a set of test queries with known relevant items, what fraction of relevant items appear in the top K results?
- **Precision@K**: of the top K results, what fraction are actually relevant?
- **MRR (Mean Reciprocal Rank)**: average of 1/rank where rank is the position of the first relevant result.
- **Latency**: p50/p99 query time for the pgvector search.
Implementation: create a small labeled test set of (query, relevant_item_ids) pairs. Run `assemble_retrieval_context` for each query. Compare returned IDs against labeled relevant IDs.

---

**Q86. What is the difference between `_search_items` and `_search_captures` in the retrieval system?**

`_search_items` queries `extracted_items` — structured, parsed, high-signal results with item metadata (urgency, type, deadline). Returns specific actionable items.
`_search_captures` queries `captures` — raw content, broader results. Excludes captures already represented in Level 1 results (`exclude_capture_ids`). Returns whole captures, which are then enriched with their associated items.
Level 1 is preferred because items are more semantically focused; Level 2 is the fallback for broader context.

---

**Q87. How does enrichment work in the retrieval system?**

After retrieval, each item is enriched with:
1. Its parent capture: a JOIN query fetching `captures.*` for `item.capture_id`.
2. Its associated actions: a query fetching `actions.*` where `extracted_item_id = item.id`.
The enriched result dict contains: `item fields + parent_capture + actions`. This gives the LLM complete context: not just "Sarah meeting item" but also "from this capture, with these planned actions."

---

**Q88. What is the "context window" limitation in RAG and how does Orbit handle it?**

LLMs have a maximum context window (e.g., 128K tokens for Claude). If `retrieval_context` contains too many results, the prompt exceeds the limit. Orbit handles this via: (1) `LIMIT $2` in the pgvector query — caps the number of results. (2) The threshold `> 0.3` naturally filters low-relevance results. (3) M4 trace pattern — retrieval context contains summary-level enriched items, not full raw content (which could be very long).

---

**Q89. What is "embedding drift" and is it a concern for Orbit?**

Embedding drift occurs when the same text produces different vectors at different times. Causes: (1) model update (using a different version of `all-MiniLM-L6-v2`); (2) different input preprocessing (tokenization changes). In Orbit: drift is a concern if the model is ever upgraded. Current mitigation: the model version is pinned in `requirements.txt`. Long-term mitigation: add `embedding_model_version TEXT` to captures/items and re-embed on version change (see Q41).

---

**Q90. How would you add personalization to the retrieval system?**

Current system retrieves semantically similar items regardless of recency or user-specific patterns. Personalization layers:
1. **Recency boost**: modify the score as `final_score = 0.7 * semantic_score + 0.3 * recency_score` where `recency_score = 1 / (1 + days_since_created)`.
2. **User feedback**: if a user dismisses a retrieval result as irrelevant, decrease the weight for that capture's embedding. Requires a `feedback` table.
3. **Topic affinity**: cluster a user's captures by topic. Boost results from the user's most frequently accessed clusters.
4. **Temporal patterns**: if the user always captures meeting notes on Mondays, boost Monday-related past captures for Monday queries.

---

**Q91. What is the difference between dense retrieval (what Orbit uses) and sparse retrieval (BM25)?**

Dense retrieval: encode text to a dense vector, find nearest neighbors. Handles semantic similarity, synonyms, paraphrases. Slow to compute but fast at query time with HNSW.
Sparse retrieval (BM25/TF-IDF): count term frequencies, match exact or stemmed terms. Fast, interpretable, good for specific keywords and proper nouns.
Hybrid retrieval combines both: run sparse + dense in parallel, merge scores (Reciprocal Rank Fusion). In Orbit's Q48, adding a `tsvector` index alongside pgvector implements hybrid retrieval.

---

**Q92. How is `retrieval_context` populated in practice — who calls `assemble_retrieval_context`?**

In Orbit's pipeline, `retrieval_context` is populated early — either by a dedicated retrieval step or within the intent/planning agent before LLM calls. The query is typically derived from the current capture's `raw_content`. The goal is to have `retrieval_context` populated in AgentState before the planning agent runs, so planning LLM prompts can include historical context.

---

**Q93. What happens to retrieval quality as the database grows?**

Initially (small DB): few results, potentially no matches above threshold. Quality is limited by data sparsity.
As data grows: more relevant matches surface. Retrieval quality typically improves because more historical captures exist to match against.
At very large scale: too many results above threshold — the `LIMIT` clause becomes critical. Quality can degrade if the top results are dominated by very common topics. Solution: diversify results (maximal marginal relevance — penalize results similar to already-selected ones) or personalize (user-specific filtering).

---

**Q94. How does MemorySaver relate to RAG retrieval in Orbit?**

They solve different problems. MemorySaver: short-term, in-process state persistence for LangGraph's checkpoint/resume mechanism. It stores the full AgentState dict for the current and recent runs. It's not queryable by semantic similarity.

RAG retrieval: long-term, cross-run knowledge retrieval. Finds semantically similar content from any past run stored in PostgreSQL. It's queryable and survives process restarts.

MemorySaver = working memory. PostgreSQL + pgvector = long-term memory.

---

**Q95. What would you change about the retrieval system for production?**

1. **Async embedding**: run `_embedder.encode()` in a thread pool executor to not block the event loop.
2. **Caching**: cache embeddings for repeated queries (e.g., hub page polls the same "recent context" query every 30 seconds).
3. **Pagination**: add cursor-based pagination to retrieval results for infinite-scroll UI.
4. **Re-ranking**: use a cross-encoder model (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`) to re-rank top-K retrieval results for better precision.
5. **Monitoring**: log `semantic_score` distributions to detect drift or degradation.
6. **User scoping**: add `user_id` to captures/items and filter retrieval by `user_id` for multi-tenant deployment.

---

**Q96. What is the `exclude_capture_ids` parameter in `_search_captures`?**

After Level 1 retrieval (items), each retrieved item has a `capture_id`. Level 2 (captures) would redundantly return those same captures (they're already represented via their items). `exclude_capture_ids` prevents this by filtering: `WHERE id NOT IN ($1, $2, ...)`. This deduplicates the retrieval context — no information is presented twice.

---

**Q97. How would you implement time-bounded retrieval (e.g., "only retrieve from the last 30 days")?**

Add a `max_age_days: int = 90` parameter to `assemble_retrieval_context`. Build the filter dynamically:
```python
time_filter = f"AND created_at > NOW() - INTERVAL '{max_age_days} days'"
```
Include it in both `_search_items` and `_search_captures` queries. For Orbit's use case (personal chief-of-staff), recent context is usually more relevant. A 30-day default with override option balances recall vs. noise from stale history.

---

**Q98. What is the "lost in the middle" problem in RAG and does Orbit face it?**

When LLMs receive a long context, they attend better to content at the beginning and end than in the middle. If `retrieval_context` contains 20 items, the planning agent's LLM may effectively ignore items 5–15. Orbit's mitigation: sort `retrieval_context` by `semantic_score DESC` and pass only the top 3–5 results to the LLM. Fewer, higher-quality results beat many mediocre ones.

---

**Q99. What is maximal marginal relevance (MMR) and when would Orbit benefit from it?**

MMR selects retrieval results that are both relevant to the query AND diverse from each other. Standard retrieval might return 5 variations of "meeting with Sarah" when the user has many Sarah-related captures — all similar to the query, all similar to each other. MMR would select 1 Sarah result and 4 results about different topics. Orbit would benefit when a user has many captures about one recurring topic — without MMR, retrieval context would be dominated by that topic even for queries about other things.

---

**Q100. How does the memory agent's embedding step relate to the retrieval system's search step?**

They are mirror images: same model (`_embedder`), same dimension (384), same normalization. Memory agent: `encode(text) → store in DB`. Retrieval: `encode(query) → compare against stored vectors`. The `<=>` cosine distance is meaningful only because both sides used the same model. If you stored embeddings from model A and queried with model B, the vectors would live in different semantic spaces and distances would be meaningless.

---

### 6.6 DATABASE DESIGN QUESTIONS (20)

**Q101. Why is UUID used for primary keys instead of auto-increment integers?**

UUIDs: generated application-side (before DB insert), enabling the ID to be placed in AgentState before the insert completes. Globally unique across tables and services. Don't reveal table size or creation sequence.

Auto-increment: generated DB-side (requires a round-trip to know the ID). Reveals row count. Prone to collisions if IDs from different tables are mixed. For Orbit's use case (IDs passed around between agents, stored in AgentState, referenced in LangSmith), application-generated UUIDs are clearly superior.

---

**Q102. What is JSONB and when should you use it over TEXT?**

JSONB: PostgreSQL's binary JSON format. Stores JSON as parsed binary, enabling: (1) key/value queries with operators (`->>`, `@>`); (2) GIN indexing for fast JSON key searches; (3) automatic validation (invalid JSON fails on insert). TEXT: stores arbitrary strings with no built-in structure.

Use JSONB when: the data is structured but schema is variable (metadata, entities, payload). Use TEXT when: content is arbitrary and not queried by structure (raw_content).

---

**Q103. What is the purpose of `depends_on_action_id` in the actions table?**

A self-referential FK that models action dependency chains. Action B may depend on Action A completing first (e.g., "send email" depends on "draft email"). At execution time, the execution agent queries `WHERE depends_on_action_id IS NULL OR depends_on_action_id IN (completed_action_ids)` to find ready-to-execute actions. This encodes the dependency graph in the DB without nesting in AgentState.

---

**Q104. Why is `confidence_score` a FLOAT and not an INTEGER percentage?**

FLOATs allow finer granularity (0.847 vs. 85). In ML contexts, scores are typically floats from 0.0 to 1.0 — matching the output format of the intent classification model directly, without conversion. INTEGER percentages (85%) lose precision and require a multiplication step. The urgency_score field follows the same reasoning.

---

**Q105. What does `DEFAULT 'pending'` on `planning_status` accomplish?**

It ensures every newly inserted extracted_item starts with `planning_status = 'pending'` without the inserting code needing to explicitly set it. Memory_agent's `insert_extracted_item` doesn't pass `planning_status` — PostgreSQL fills it in automatically. This centralizes the default in the schema, not in application code.

---

**Q106. Why use separate `edited_payload` and `final_payload` columns in decisions?**

`edited_payload`: what the human changed from the original action payload (a diff). May be NULL if the human approved without changes.
`final_payload`: the actual payload used for execution (original merged with edits).

Storing both allows auditing: "the system proposed X, the human changed Y to Z, the final execution used W." If you only stored `final_payload`, you'd lose visibility into what was changed vs. what was original.

---

**Q107. What is a partial index in PostgreSQL?**

An index that only covers rows matching a `WHERE` clause. The partial unique index on decisions: `CREATE UNIQUE INDEX ... WHERE action_id IS NOT NULL` indexes only rows where `action_id` is non-null. Benefits: smaller index size (doesn't index NULL rows), semantically precise uniqueness constraint (only enforce for active, non-orphaned decisions).

---

**Q108. How would you add multi-tenancy (multiple users) to this schema?**

Add `user_id UUID NOT NULL REFERENCES users(id)` to the `captures`, `extracted_items`, `actions`, `decisions`, and `runs` tables. Add `CREATE INDEX ON captures(user_id)` for fast per-user queries. Update all queries to include `WHERE user_id = $1`. Update HNSW vector searches to include `AND user_id = $1` (pgvector supports pre-filtering). The retrieval system's context is now scoped per-user by default.

---

**Q109. What is a GIN index and when would you add one to Orbit's schema?**

GIN (Generalized Inverted Index): optimized for composite types like arrays, JSONB, and tsvector. Use for: (1) JSONB key queries: `CREATE INDEX ON captures USING gin(metadata)` — enables fast `WHERE metadata @> '{"source": "email"}'`; (2) Full-text search: `CREATE INDEX ON extracted_items USING gin(to_tsvector('english', title || description))`; (3) Array containment: `WHERE entities @> '["Sarah"]'` on a JSONB array. Add GIN when queries on JSONB columns appear in EXPLAIN as sequential scans.

---

**Q110. Why store the final trace in `runs.trace JSONB` if it's already in MemorySaver?**

MemorySaver is ephemeral — lost on restart. `runs.trace` persists permanently. After a run completes, the API writes `state["trace"]` to `runs.trace`. This enables: (1) historical run inspection via the API without LangGraph's state history API; (2) analytics (how many items did each run extract? which agents ran?); (3) debugging past runs after MemorySaver has been cleared.

---

**Q111. What is the purpose of `updated_at` in the runs table?**

Tracks when the run record was last modified. A run starts with `status = 'running'`. When it completes: `UPDATE runs SET status = 'completed', trace = $1, updated_at = NOW()`. `updated_at` enables: (1) detecting stale/hung runs (`WHERE status = 'running' AND updated_at < NOW() - INTERVAL '1 hour'` → likely crashed); (2) auditing when completion occurred.

---

**Q112. How would you implement pagination for `list_captures`?**

Current: `LIMIT $1 OFFSET $2` (offset-based).
Problem: `OFFSET 1000` scans and discards 1000 rows — inefficient for large tables.
Better: **keyset/cursor pagination**: `WHERE (created_at, id) < ($cursor_time, $cursor_id) ORDER BY created_at DESC, id DESC LIMIT $page_size`. The client passes the last seen `(created_at, id)` as the cursor. This is O(log n) via the index, regardless of page depth. Requires a composite index on `(created_at, id)`.

---

**Q113. What is a foreign key and what happens if you insert an extracted_item with a non-existent capture_id?**

A foreign key constrains `extracted_items.capture_id` to reference a valid `captures.id`. If you try to insert an item with `capture_id = 'non-existent-uuid'`, PostgreSQL raises a `ForeignKeyViolationError`. asyncpg propagates this as `asyncpg.ForeignKeyViolationError`. In memory_agent, this would only happen if `insert_capture` returned a fake UUID — which can't happen in practice since the capture is inserted first and the real UUID is used.

---

**Q114. Why would you NOT add a NOT NULL constraint to `captures.file_path`?**

`file_path` is only relevant for file-based inputs (PDF, image). Text inputs don't have a file path. Making it NOT NULL would require every text capture to supply a fake file path (e.g., empty string), which is semantically wrong. NULL correctly represents "this capture has no associated file." `get_capture` uses `file_path.get("file_path")` in Python, naturally handling None.

---

**Q115. What would happen to the schema if you needed to support multi-modal captures that reference multiple files?**

Current: `file_path TEXT` supports one file per capture. For multiple files: (1) Change to `file_paths TEXT[]` (PostgreSQL array) or `file_paths JSONB`. (2) Or create a separate `capture_files` table: `(id UUID PK, capture_id UUID FK, file_path TEXT, file_type TEXT, position INT)`. Option 2 is more normalized and enables per-file metadata. The embedding strategy would also need updating: embed each file separately and store multiple embeddings per capture (one per file), or combine files into one embedding.

---

**Q116. What is `vector_cosine_ops` vs `vector_l2_ops` vs `vector_ip_ops`?**

`vector_cosine_ops`: for the `<=>` cosine distance operator. Best for unit-normalized embeddings (like sentence-transformers output). Measures angle between vectors.
`vector_l2_ops`: for the `<->` Euclidean distance operator. Measures straight-line distance. Sensitive to vector magnitude.
`vector_ip_ops`: for the `<#>` negative inner product operator. `1 - (a · b)` — equivalent to cosine distance for unit vectors but computed differently.

Use `cosine_ops` when your embeddings are unit-normalized (most sentence transformer models). Use `l2_ops` for embeddings where magnitude carries meaning.

---

**Q117. What is an ACID transaction and does memory_agent use one?**

ACID: Atomicity (all-or-nothing), Consistency (constraints maintained), Isolation (concurrent transactions don't interfere), Durability (committed data survives crashes).

Memory_agent currently does NOT wrap all inserts in a single transaction. Each `pool.acquire()` block is its own implicit transaction. If `insert_capture` succeeds but `insert_extracted_item` fails, you have a partial write. For strict ACID, wrap the entire memory_agent's DB operations:
```python
async with pool.acquire() as conn:
    async with conn.transaction():
        capture_id = await conn.fetchval("INSERT INTO captures ... RETURNING id", ...)
        for item in items:
            await conn.fetchval("INSERT INTO extracted_items ... RETURNING id", ...)
```

---

**Q118. What is the purpose of `created_at` columns, given that `run_id` already links to a run?**

`created_at` gives each row its own timestamp independent of the run. Within a single run, captures and items might be created at slightly different times. `created_at` enables: (1) time-series analysis ("how many items were captured per hour?"); (2) recency-based retrieval filtering; (3) TTL-based archiving ("delete soft-deleted captures older than 1 year"). `run_id` links to the run context, not to creation time.

---

**Q119. Why does the `decisions` table have its own `decided_at TIMESTAMP` rather than using `updated_at`?**

`decided_at` is a semantic timestamp: "when the human made this decision." It's set once at insert time and never changes. An `updated_at` column implies the row can be updated. Decisions are immutable audit records — once a human decides, that record should not change. Having `decided_at` as an explicit, set-once field communicates this immutability. If a decision is reversed, a new row is inserted (with a new `decided_at`), not an update to the old row.

---

**Q120. How would you back up and restore Orbit's database with minimal data loss?**

**Backup strategy:**
1. **Continuous WAL archiving** (Write-Ahead Log): stream PostgreSQL's WAL to S3 using `pg_basebackup` + `archive_command`. Enables point-in-time recovery (PITR) to any second.
2. **Daily `pg_dump`**: logical backup for table-level restoration.
3. **pgvector indexes**: HNSW indexes are not backed up by logical dumps — they must be recreated from the data via `CREATE INDEX` after restore.

**Restore procedure:**
1. Restore base backup + WAL segments to target time.
2. Verify data integrity: `SELECT COUNT(*) FROM captures WHERE deleted_at IS NULL`.
3. Recreate HNSW indexes if needed.
4. Verify embedding dimensions: `SELECT array_length(embedding::float[], 1) FROM captures LIMIT 1` should return 384.

---
