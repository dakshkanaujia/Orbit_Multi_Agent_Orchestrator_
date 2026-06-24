# Daksh — Agent Orchestration Layer: Viva Preparation Guide
### Orbit AI · Multi-Agent Personal Chief-of-Staff

---

## 1. Personal Ownership Summary

### What Daksh Built

Daksh owns the **Agent Orchestration Layer** — the central nervous system of Orbit AI. Every user request that enters Orbit passes through code Daksh authored. Specifically:

- **`graph.py`** — The LangGraph `StateGraph` definition: 7 nodes, all edges (conditional and unconditional), the interrupt mechanism, and graph compilation.
- **`captures.py`** (orchestration surface) — The FastAPI endpoints that invoke the graph, including the `POST /api/captures/stream` SSE endpoint and `POST /api/captures` blocking endpoint.
- **The `generate()` async generator** — Parses raw LangGraph `astream` events and converts them into typed SSE frames the frontend can consume.
- **LangSmith integration config** — `run_name`, `metadata`, `tags`, and `run_id`→`thread_id` plumbing for full observability.

### Why It Was Needed

Without an orchestration layer, the 7 specialized agents (understanding, intent, clarification_halt, memory, planning, tool_router, approval) are just standalone async Python functions. They have no shared execution context, no conditional routing, no checkpoint/resume capability, and no streaming surface. Daksh's layer is what turns those 7 functions into a coherent, stateful, resumable pipeline that can:

1. Route conditionally (clarification path vs. normal path)
2. Pause mid-execution for human approval
3. Persist checkpoints so approval can happen asynchronously
4. Stream real-time progress to the frontend over SSE
5. Correlate all LangSmith traces under a single `run_id`

### How It Connects to Every Other Module

| Team Member | Their Domain | How Daksh's Graph Connects |
|---|---|---|
| Utkarsh | `AgentState` TypedDict + MemorySaver semantics | Daksh's `StateGraph(AgentState)` uses Utkarsh's state schema as the type contract. Every node receives and returns `AgentState` fields. |
| Jash | `planning_agent` + `tool_router_agent` + action generation | These are registered as nodes via `builder.add_node("planning", planning_agent)`. Daksh's graph calls them in sequence and propagates their output via shared state. |
| Abhay | G3 guardrail + `approval_agent` + human-in-the-loop | Abhay's G3 runs before `graph.invoke()` in captures.py. Abhay's `approval_agent` is the target of `interrupt_before=["approval"]`. The interrupt mechanism is Daksh's implementation. |
| Anushka | understanding_agent (OCR + NER) | Registered as the entry point node. Daksh's `set_entry_point("understanding")` is what makes it first. |
| SSE / Frontend | Pipeline visualization | Daksh's `generate()` generator in captures.py produces the SSE events the frontend reads to animate pipeline nodes. |

---

## 2. Deep Implementation Walkthrough

### Full Annotated `graph.py`

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
```

**`StateGraph`** is LangGraph's typed graph class. Unlike the lower-level `Graph`, `StateGraph` takes a TypedDict schema and enforces that every node function receives the full state and returns a dict of updates. `END` is a sentinel string (`"__end__"`) that signals graph termination — any edge that points to `END` terminates the run for that path.

**`MemorySaver`** is LangGraph's built-in in-memory checkpointer. It implements the `BaseCheckpointSaver` interface, storing checkpoint dicts in a plain Python dict keyed by `(thread_id, checkpoint_id)`. Every time LangGraph crosses a node boundary, it writes the current state to the checkpointer. This is what enables `interrupt_before` to work — when the graph is about to run `approval`, it writes state to MemorySaver and pauses, allowing a later `graph.invoke()` call with the same `thread_id` to resume.

---

```python
checkpointer = MemorySaver()
```

A single `MemorySaver` instance is created at module load time. It persists for the lifetime of the FastAPI process. This is intentional for a demo/dev context — if the server restarts, checkpoints are lost, which is acceptable because in-flight approvals are rare and sessions are short-lived. For production, this would be replaced with `AsyncSqliteSaver` or a Postgres-backed checkpointer.

---

```python
builder = StateGraph(AgentState)
```

`StateGraph` is parameterized by `AgentState` — the TypedDict Utkarsh designed. This means:
- Every node function signature is `async def node(state: AgentState) -> dict`
- LangGraph validates at compilation time that the return dict keys are valid `AgentState` fields
- The state is propagated between nodes via merge: return value is shallow-merged into the current state (default "replace" reducer semantics — no key appends, no lists merged, just overwrite)

---

```python
builder.add_node("understanding", understanding_agent)
builder.add_node("intent", intent_agent)
builder.add_node("clarification_halt", clarification_halt_node)
builder.add_node("memory", memory_agent)
builder.add_node("planning", planning_agent)
builder.add_node("tool_router", tool_router_agent)
builder.add_node("approval", approval_agent)
```

Each `add_node` call maps a **string name** to a **Python async function**. The string name is used in:
1. Edge definitions (`add_edge`, `add_conditional_edges`)
2. `interrupt_before` list
3. SSE event payloads (`agent: "understanding"`)
4. LangSmith span names
5. `astream` update dict keys

The mapping is: when the graph reaches node `"understanding"`, it calls `understanding_agent(state)`, takes the returned dict, merges it into `AgentState`, then follows outgoing edges.

Why async functions? Because all 7 agents call external services — Claude API (Haiku), PostgreSQL, REST APIs. Async lets FastAPI's event loop handle other requests while awaiting I/O. A sync function passed to `add_node` would block the event loop.

---

```python
builder.set_entry_point("understanding")
```

`set_entry_point` is syntactic sugar for `builder.add_edge(START, "understanding")` where `START` is the `"__start__"` sentinel. It designates `understanding_agent` as the first node to execute when `graph.invoke()` or `graph.astream()` is called. There can only be one entry point per graph.

---

```python
builder.add_edge("understanding", "intent")
```

An unconditional edge. After `understanding_agent` returns, the graph **always** moves to `intent_agent`. This is correct because:
- Understanding's job is OCR + entity extraction — it either succeeds or raises an exception
- There is no branching condition at this stage — every input that passes understanding goes to intent validation
- If understanding fails (exception), LangGraph propagates the exception to the caller, which FastAPI catches and returns as a 500

---

```python
builder.add_conditional_edges(
    "intent",
    lambda state: "clarification_halt" if state.get("clarification_needed") else "memory",
    {"clarification_halt": "clarification_halt", "memory": "memory"}
)
```

This is the only conditional edge in the graph. Breaking it down:

**Source node**: `"intent"` — the condition is evaluated after `intent_agent` completes.

**Router function**: `lambda state: "clarification_halt" if state.get("clarification_needed") else "memory"` — a pure function that reads `AgentState` and returns a string. The string must be a key in the routing dict. `state.get("clarification_needed")` returns `True` if the intent agent found ambiguous items (e.g., deadline impossible to parse, item too vague), `False`/`None` otherwise.

**Routing dict**: `{"clarification_halt": "clarification_halt", "memory": "memory"}` — maps router return values to node names. The dict serves two purposes: (1) it validates at compile time that all possible return values are handled, and (2) it allows aliasing (router returns `"A"`, dict maps `"A"` to node `"B"`). In this case the keys and values are identical, but the dict is still required.

Why this conditional is after `intent` not `understanding`:
- `understanding` does raw OCR/NER — it doesn't know what the user intends, it just extracts structure
- `intent` validates extracted items against business rules: deadline parsability, urgency truncation, item coherence
- Only after `intent` runs do we have the `clarification_needed` flag set with a `clarification_reason`
- Placing the conditional after `understanding` would require understanding to do intent-level reasoning, breaking single-responsibility

---

```python
builder.add_edge("clarification_halt", END)
```

`clarification_halt_node` is a terminal no-op. It sets no state, does nothing. Its only purpose is to give the graph a valid node to route to when `clarification_needed=True`. After it "runs" (returns immediately), the graph hits `END` and terminates. The FastAPI endpoint in captures.py checks `state.get("clarification_needed")` after graph completion and returns HTTP 422 with `clarification_reason`.

Why not route directly from `intent` to `END`? Because `add_conditional_edges` routing dict values must be node names, not `END`. Well — actually `END` is valid as a routing target. The reason for `clarification_halt` as an explicit node is observability: LangSmith records it as a span, the SSE stream emits `agent: "clarification_halt"` as an event, and the frontend can show a "Needs Clarification" state in the pipeline visualization.

---

```python
builder.add_edge("memory", "planning")
builder.add_edge("planning", "tool_router")
builder.add_edge("tool_router", "approval")
builder.add_edge("approval", END)
```

The normal-path chain. These are all unconditional because:
- `memory → planning`: memory agent persists entities and sets `retrieval_context`. Planning always needs retrieval context.
- `planning → tool_router`: planning generates `actions`. tool_router always validates and builds `pending_action_ids` from those actions.
- `tool_router → approval`: the pending action list is always reviewed before execution — no exception.
- `approval → END`: approval is the last step. After approval, execution results are in state and the run is complete.

---

```python
app = builder.compile(checkpointer=checkpointer, interrupt_before=["approval"])
```

`builder.compile()` performs graph validation and returns a `CompiledStateGraph`. Validation checks:
1. All nodes referenced in edges exist
2. All nodes are reachable from the entry point
3. All conditional routing dict values are valid node names or END
4. No cycles that could cause infinite loops (LangGraph does not validate this — it's the developer's responsibility)

**`checkpointer=checkpointer`**: Wires the `MemorySaver` instance into the compiled graph. Every node transition writes state to this checkpointer, keyed by `thread_id` (which Daksh maps from `run_id`). Without a checkpointer, `interrupt_before` cannot function.

**`interrupt_before=["approval"]`**: Tells LangGraph to:
1. Execute all nodes up to (but not including) `approval`
2. Write the current state to the checkpointer
3. Raise a `GraphInterrupt` exception internally
4. Return control to the caller

"Before" means: the approval node's function (`approval_agent`) has NOT been called yet. The state at checkpoint contains the output of `tool_router` — specifically `pending_action_ids` — which is exactly what the frontend needs to show the user their proposed actions before confirming.

To resume after interrupt, the caller invokes `graph.invoke(None, config={"configurable": {"thread_id": run_id}})` with `None` as input (the state is loaded from the checkpointer) and with `interrupt_before` excluded (or with `invoke` rather than streaming). LangGraph loads the checkpointed state, skips the already-executed nodes, and runs from `approval` onward.

---

### SSE Streaming — `captures.py`

#### `astream(stream_mode="updates")`

Used in `POST /api/captures/stream`.

When `stream_mode="updates"`, `astream` yields one dict per node completion. The dict format is:

```python
{"node_name": {<state_keys_that_changed>}}
```

For example, after `understanding_agent` completes:
```python
{"understanding": {"extracted_entities": [...], "capture": {...}}}
```

The `generate()` async generator in captures.py parses this:

```python
async def generate():
    async for event in app.astream(initial_state, config=config, stream_mode="updates"):
        node_name = list(event.keys())[0]  # extract first (only) key
        yield f"event: agent\ndata: {json.dumps({'agent': node_name, 'status': 'done'})}\n\n"
```

Each SSE frame has the format:
```
event: agent
data: {"agent": "understanding", "status": "done"}

```
(Note the double newline — that terminates an SSE event.)

The frontend reads these frames via `fetch()` + `ReadableStream`, parses each `data:` line as JSON, and animates the corresponding pipeline node.

#### `astream(stream_mode="values")`

Used in `POST /api/captures` (blocking endpoint, used for non-streaming clients or testing).

When `stream_mode="values"`, `astream` yields the **full `AgentState`** after each node. The caller only needs the final state, so it iterates the entire stream and takes the last value. Less efficient for streaming to the UI but useful for getting complete state after a run.

---

### `run_id` Flow

```
1. FastAPI POST /api/captures/stream receives request
2. Generates UUID: run_id = str(uuid4())
3. Constructs LangGraph config:
   config = {
       "configurable": {"thread_id": run_id},
       "run_name": "orbit-pipeline",
       "metadata": {"orbit_run_id": run_id, "input_type": input_type},
       "tags": ["orbit", "stream"]
   }
4. Calls app.astream(initial_state, config=config, stream_mode="updates")
5. LangGraph uses config["configurable"]["thread_id"] as the MemorySaver key
6. LangSmith uses run_name, metadata, tags to group all spans under one trace
7. run_id is returned to frontend in SSE stream (first event or response header)
8. Frontend uses run_id for the approval POST: POST /api/captures/{capture_id}/approve
9. Approval endpoint reconstructs config with same thread_id, resumes graph
```

Why `run_id` as `thread_id`? Because LangGraph's `MemorySaver` groups checkpoints by `thread_id`. Using the same ID that the frontend holds means the approval endpoint can reconstruct the exact checkpoint reference without a database lookup. It also makes LangSmith traces correlate — all spans from one user session appear under one trace root.

---

## 3. LangGraph Concepts: Beginner to Advanced

### StateGraph vs Graph

`Graph` is LangGraph's lower-level class. Nodes can pass arbitrary data; there is no shared typed state. You must manually pass outputs as inputs. `StateGraph` adds a shared TypedDict state that all nodes read from and write to. Every node receives the full current state and returns a partial update. `StateGraph` is the right choice for multi-agent pipelines where agents need to read each other's outputs. `MessageGraph` is a specialized subclass of `StateGraph` where the state is a list of `BaseMessage` objects — useful for chat, not for structured data pipelines like Orbit.

### Nodes: Sync vs Async

LangGraph supports both. Sync nodes run in a thread pool executor (via `asyncio.run_in_executor`). Async nodes run directly on the event loop. For Orbit, all nodes are async because they call external services. Mixing sync and async nodes in one graph is possible but adds complexity.

### Edges: Unconditional vs Conditional

`add_edge(A, B)` — always goes from A to B. `add_conditional_edges(A, fn, dict)` — calls `fn(state)` after A completes, uses the return value as a key into `dict` to find the next node. Conditional edges enable branching. Multiple conditional edges from the same node are not supported; use a single router function that returns different values for different conditions.

### `set_entry_point` vs `add_edge(START, ...)`

`set_entry_point("node")` is exactly equivalent to `add_edge(START, "node")` where `START = "__start__"`. The `set_entry_point` API is cleaner and more readable. `START` is useful when you need to add it explicitly to a routing dict or visualize the graph.

### `END` Sentinel

`END = "__end__"` is a special node name that terminates the graph. Any edge pointing to `END` causes the graph to stop after the source node. `END` cannot have outgoing edges. It is valid as a routing dict value in `add_conditional_edges`.

### MemorySaver Internals

`MemorySaver` is a dict-based implementation of `BaseCheckpointSaver`. Internally it stores:
```python
{
    thread_id: {
        checkpoint_id: {
            "v": 1,
            "ts": "2024-...",
            "id": checkpoint_id,
            "channel_values": {<AgentState fields>},
            "channel_versions": {<field: version_int>},
            "versions_seen": {...},
            "pending_sends": []
        }
    }
}
```
When `graph.invoke()` is called with a `thread_id`, LangGraph loads the latest checkpoint for that thread. When a node completes, LangGraph writes a new checkpoint. The checkpoint contains the full state, not just the delta — it's a snapshot, not a diff log.

### `interrupt_before` vs `interrupt_after`

`interrupt_before=["node"]`: Graph pauses **before** running `node`. The checkpoint state contains everything up to (and including) the previous node. The interrupting node has NOT run.

`interrupt_after=["node"]`: Graph pauses **after** running `node`. The checkpoint state includes that node's output. The interrupting node HAS run.

For Orbit's approval use case, `interrupt_before=["approval"]` is correct: the user sees the proposed actions (from `tool_router`'s output), decides to approve or reject, and then the graph resumes and `approval_agent` executes the approved actions. If we used `interrupt_after=["approval"]`, actions would already be executed before the user sees them.

### Thread IDs and Run Isolation

Each unique `thread_id` in the LangGraph config is a completely isolated execution context. Checkpoints from thread A never bleed into thread B. This is Orbit's per-request isolation: each API call generates a new `run_id` = new `thread_id`. Concurrent requests each have their own state, their own checkpoint history, and their own interrupt state.

### `astream` Modes

| Mode | Yields | When to use |
|---|---|---|
| `"values"` | Full `AgentState` after each node | When you need complete state at each step |
| `"updates"` | `{node_name: partial_state_update}` after each node | When you need to know which node just ran + what changed |
| `"debug"` | Verbose internal events (checkpoint writes, etc.) | Debugging only |
| `"messages"` | LangChain `BaseMessage` objects from LLM calls | Chat interfaces |

Orbit uses `"updates"` for SSE streaming (need node name for frontend events) and `"values"` for blocking endpoint (need final complete state).

### Conditional Routing: Lambda vs Named Function

The lambda `lambda state: "clarification_halt" if state.get("clarification_needed") else "memory"` is fine for a single condition. For complex routing with multiple conditions or reuse across graphs, a named function is cleaner:

```python
def route_after_intent(state: AgentState) -> str:
    if state.get("clarification_needed"):
        return "clarification_halt"
    return "memory"
```

Both work identically in LangGraph. The routing dict format is always `{return_value: node_name}`.

### Graph Compilation and Validation

`builder.compile()` does:
1. Builds the internal node adjacency list
2. Validates all referenced node names exist
3. Validates all routing dict values resolve to nodes or END
4. Wires in the checkpointer
5. Configures interrupt points
6. Returns a `CompiledStateGraph` with `invoke`, `ainvoke`, `stream`, `astream` methods

### Reducers and Default Replace Semantics

By default, `StateGraph` uses "replace" semantics: if a node returns `{"extracted_entities": [...]}`, the current value of `extracted_entities` in state is replaced entirely. There are no automatic appends or merges. To use append semantics (e.g., for a list that accumulates across nodes), you define a reducer in the TypedDict using `Annotated`:

```python
from typing import Annotated
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # append reducer
    extracted_entities: list  # replace reducer (default)
```

Orbit's `AgentState` uses replace semantics throughout — each agent replaces its output fields entirely.

### How State Propagates Between Nodes

1. Node A executes: `result = await node_a_fn(current_state)`
2. LangGraph merges: `new_state = {**current_state, **result}`
3. LangGraph writes checkpoint: `checkpointer.put(thread_id, new_state)`
4. LangGraph evaluates outgoing edges from A
5. Node B executes with `new_state` as input

This means Node B always sees all fields set by Node A (and all prior nodes). Nodes do not communicate via return values — they communicate via the shared state.

### Graph Visualization

`app.get_graph().draw_mermaid()` returns a Mermaid diagram string. `app.get_graph().draw_png()` requires `graphviz` installed. These are useful for documentation and debugging.

---

## 4. Multi-Agent Concepts (with Project Examples)

### Supervisor Pattern

In the supervisor pattern, one agent orchestrates others — it decides which sub-agents to call and in what order. In Orbit, `tool_router` acts as a lightweight supervisor: it receives the `actions` list from `planning`, validates each action type, and determines which actions are executable (`pending_action_ids`). It supervises the action plan before it goes to execution.

More broadly, the LangGraph graph itself is the supervisor — it manages the entire pipeline, decides routing (via conditional edges), and manages lifecycle (interrupt, resume, termination).

### Worker Pattern

Each of the 7 agents is a specialized worker with a single, well-defined responsibility:
- `understanding`: OCR + NER only
- `intent`: deadline validation + urgency truncation only
- `clarification_halt`: terminal signal only
- `memory`: embed + persist only
- `planning`: action generation only
- `tool_router`: action validation + queueing only
- `approval`: checkpoint + execution only

No agent does two jobs. This is the Worker pattern.

### Sequential Delegation Chain

`understanding → intent → memory → planning → tool_router → approval` is a sequential delegation chain. Each agent receives enriched state from all prior agents and adds its contribution. The chain guarantees ordering: you cannot run planning without memory (no retrieval context), and you cannot run tool_router without planning (no actions).

### Conditional Delegation

`intent` delegates to either `clarification_halt` (terminal) or `memory` (continue). This is conditional delegation — the delegating agent (intent) signals via state (`clarification_needed: True`), and the graph's conditional edge translates that signal into a routing decision. The intent agent itself does not call other agents; it only sets state. The orchestration logic lives in the graph, not the agent.

### Agent Communication via Shared State (Not Message Passing)

Orbit agents do not call each other. They do not have inboxes or message queues. They read from `AgentState` and write back to `AgentState`. This is shared-state communication, distinct from AutoGen's message-passing pattern where agents send messages to each other's inboxes. Shared-state communication is simpler to reason about, easier to debug, and naturally serializable (the state dict is a checkpoint).

### Context Propagation

`run_id` flows through the entire system as a first-class context identifier:
- FastAPI generates it
- Passed to LangGraph as `thread_id`
- Passed to LangSmith as `orbit_run_id` metadata
- Stored in the `capture` record in PostgreSQL
- Returned to frontend
- Used by frontend to approve: `POST /api/captures/{capture_id}/approve`
- Used to resume graph at the exact checkpoint

### Human-in-the-Loop as an Agent

The `approval` node is a full LangGraph node — it has the same signature as all other nodes, it appears in the graph as a node, and it receives the full state. But its execution is gated by `interrupt_before`: the graph never runs `approval_agent` until the human confirms. The human's decision (approve/reject specific actions) is injected into state via the resume call. This is LangGraph's canonical pattern for human-in-the-loop.

---

## 5. Design Decisions and Defenses

### Why LangGraph over raw function chains / FastAPI dependencies?

Raw async function chains have no checkpoint/resume capability. If you want to pause before approval and resume later (potentially after seconds or minutes, when the user checks their notification), you need serializable state and a resumption mechanism. FastAPI dependencies don't provide this — they're per-request synchronous validation, not stateful graph execution.

LangGraph gives us: conditional routing with a clean API, interrupt/resume with checkpointing, streaming via `astream`, LangSmith observability integration, and a graph compilation step that validates the topology. All of this would require hundreds of lines of custom infrastructure code to replicate.

### Why not CrewAI?

CrewAI is designed for LLM-driven task decomposition — the LLM decides which agents to call and in what order. For Orbit's pipeline, the routing is **deterministic**: we always want understanding before intent, we always want planning before tool_router. Introducing an LLM-based supervisor for routing adds non-determinism, latency (an extra LLM call per step), and cost — for no benefit, since the routing rules are already known at design time.

Additionally, CrewAI does not expose programmatic graph compilation. You cannot say "always pause before approval and wait for human input" with a single parameter. The interrupt mechanism would require significant custom tooling.

### Why not AutoGen?

AutoGen is built around conversational agent patterns — agents send messages to each other in a round-robin or dynamic topology. Orbit is a single-pass pipeline: a user submits a capture, it flows through 7 agents exactly once, and produces an output. There is no conversation. AutoGen's overhead (message routing, conversation history management, agent proxy objects) adds complexity with no benefit for a linear pipeline.

### Why 7 nodes specifically?

Each node has exactly one job:
1. **Understanding**: I/O concern (OCR, NER)
2. **Intent**: Validation concern (deadline parsing, urgency)
3. **Clarification halt**: Signal concern (terminal route)
4. **Memory**: Persistence concern (embedding, PostgreSQL)
5. **Planning**: Reasoning concern (action generation)
6. **Tool router**: Validation concern (action type validation)
7. **Approval**: Execution concern (human gate + REST calls)

Merging any two would violate single responsibility. For example, merging memory + planning means if memory fails, planning never runs — but you'd want to know specifically which step failed. Separate nodes give separate LangSmith spans, separate SSE events, and separate failure points for debugging.

### Why MemorySaver not SqliteSaver / Redis?

For a demo/capstone context, MemorySaver is appropriate. Requirements:
- No cross-process state sharing needed (single FastAPI worker)
- No persistence across restarts needed (sessions are short)
- No distributed state needed (single machine)
- Simple setup (zero configuration)

MemorySaver satisfies all of these. SqliteSaver would be appropriate if we needed persistence across restarts (e.g., user submits capture, server restarts, user returns and their approval is still pending). Redis-backed checkpointer would be appropriate for horizontal scaling (multiple FastAPI workers, load balancer). The upgrade path is a single line change in `graph.py`.

### Why `interrupt_before=["approval"]` not before memory or planning?

The interrupt is for **human review of proposed actions before execution**. The user needs to see the full list of pending actions (tool_router output) before deciding. Interrupting before memory means the user would see nothing — the pipeline hasn't even retrieved context yet. Interrupting before planning means they see raw entities, not actions. Only after tool_router has validated and built `pending_action_ids` does the data exist for the user to review.

### Why sequential not parallel?

Every node in the main path has a strict data dependency on the previous node:
- Intent needs `extracted_entities` (from understanding)
- Memory needs `extracted_items` (from intent)
- Planning needs `retrieval_context` (from memory)
- Tool router needs `actions` (from planning)
- Approval needs `pending_action_ids` (from tool_router)

There is no pair of nodes in the main path that could run in parallel without one waiting on the other. The dependency chain is total.

The only theoretical parallelism would be running multiple items through planning in parallel (fan-out per item), but this would require a Map-Reduce pattern in LangGraph (using `Send` API), which adds complexity beyond the project's scope.

### Why SSE not WebSockets?

The pipeline progress stream is **unidirectional**: the server pushes events to the client, and the client only reads. WebSockets are bidirectional — they're appropriate when the client also sends messages back over the same connection (e.g., chat). For a one-way "here's what the pipeline is doing" feed, SSE is simpler:
- No connection upgrade handshake
- Works over standard HTTP/1.1
- Automatic reconnection built into the browser's `EventSource` API
- No server-side socket state to manage
- Compatible with HTTP proxies and load balancers without special configuration

### Why conditional edge after intent not after understanding?

`understanding_agent` does OCR and NER — it produces `extracted_entities` and `extracted_items`. It doesn't evaluate whether those items are valid or actionable. The `clarification_needed` flag is set by `intent_agent` after validating deadlines, parsing urgency, and checking item coherence. Understanding cannot set `clarification_needed` because it doesn't do intent-level reasoning. Placing the conditional at the wrong node would require `understanding` to do work that belongs to `intent`.

### Why `run_id` as `thread_id`?

`thread_id` is LangGraph's isolation key for checkpoints. `run_id` is the frontend's reference for the current capture session. Using them as the same value means:
1. The frontend can construct the resume call without a separate lookup: it already holds `run_id`
2. LangSmith automatically groups all spans from the same request under one trace (same `thread_id` = same trace root)
3. The database record (`capture`) stores `run_id`, so any endpoint with `capture_id` can retrieve `run_id` and reconstruct the LangGraph config
4. Debugging is simpler: one ID correlates FastAPI logs, LangSmith traces, and MemorySaver checkpoints

---

## 6. Viva Question Bank

---

### Beginner Questions (20)

**Q1. What is LangGraph?**
LangGraph is a Python library built on top of LangChain that allows you to define stateful, multi-actor applications as directed graphs. Each node in the graph is a function (or agent), edges define the flow between nodes, and a shared state dict passes information between them. LangGraph handles execution order, conditional routing, checkpointing, and streaming.

**Q2. What is a node in LangGraph?**
A node is a Python function (sync or async) registered with `builder.add_node("name", fn)`. When the graph reaches that node, it calls `fn(state)` where `state` is the current `AgentState`, and merges the returned dict into the state. In Orbit, each agent is a node.

**Q3. What is an edge in LangGraph?**
An edge is a directed connection between two nodes. `add_edge(A, B)` means: after node A completes, always go to node B. Conditional edges (`add_conditional_edges`) choose the next node based on the current state.

**Q4. What is a StateGraph?**
`StateGraph` is LangGraph's graph class that uses a shared typed state (a TypedDict). All nodes read from and write to this shared state. This is in contrast to `Graph` where nodes pass data directly to each other.

**Q5. What is `END` in LangGraph?**
`END` is the string `"__end__"` — a sentinel that terminates graph execution. Any edge pointing to `END` causes the graph to stop. In Orbit, both `clarification_halt → END` and `approval → END` use this.

**Q6. How do you add a node in LangGraph?**
```python
builder.add_node("node_name", async_function)
```
The first argument is the string identifier used in edges and observability. The second is the callable.

**Q7. How do you add an unconditional edge?**
```python
builder.add_edge("source_node", "target_node")
```
After `source_node` completes, execution always moves to `target_node`.

**Q8. What does `compile()` do?**
`builder.compile()` validates the graph topology, wires in the checkpointer, configures interrupt points, and returns a `CompiledStateGraph` with `invoke`, `ainvoke`, `stream`, and `astream` methods. After compilation, the graph is immutable.

**Q9. What is a checkpointer?**
A checkpointer is a storage backend that saves the graph's state after each node execution. This enables interrupt/resume: the graph can be paused mid-execution and resumed later by loading the checkpoint.

**Q10. What is MemorySaver?**
`MemorySaver` is LangGraph's built-in in-memory checkpointer. It stores checkpoints in a Python dict for the lifetime of the process. It requires no external dependencies. Used in Orbit for dev/demo.

**Q11. How does the entry point work?**
`builder.set_entry_point("understanding")` designates `understanding_agent` as the first node to run when `graph.invoke()` is called. It's equivalent to `builder.add_edge(START, "understanding")`.

**Q12. What is AgentState?**
`AgentState` is a TypedDict that defines all the fields shared between agents: `input_content`, `input_type`, `run_id`, `capture`, `extracted_entities`, `extracted_items`, `capture_id`, `extracted_item_ids`, `actions`, `pending_action_ids`, `decided_action_ids`, `decisions`, `execution_results`, `retrieval_context`, `clarification_needed`, `clarification_reason`, and `trace`.

**Q13. How does a node update state?**
A node function returns a dict of the fields it wants to update. LangGraph merges this dict into the current state (shallow merge / replace). The node does not need to return the full state — only the fields it changed.

**Q14. What happens when the graph reaches `END`?**
Graph execution terminates. `graph.invoke()` returns the final `AgentState`. `graph.astream()` stops yielding events.

**Q15. What is `set_entry_point` vs `add_edge(START, ...)`?**
They are equivalent. `set_entry_point("X")` is shorthand for `add_edge(START, "X")`. `START = "__start__"` is a reserved sentinel. Use `set_entry_point` for clarity.

**Q16. Can a node have multiple outgoing edges?**
With `add_conditional_edges`, a node can route to multiple different next nodes. With `add_edge`, a node has exactly one next node. You cannot call `add_edge` twice from the same source — use conditional edges for multiple destinations.

**Q17. What is the difference between `invoke` and `ainvoke`?**
`invoke` is synchronous — it blocks the calling thread until the graph completes. `ainvoke` is async — it's awaitable and doesn't block the event loop. For FastAPI (async framework), always use `ainvoke` or `astream`.

**Q18. What is the difference between `stream` and `astream`?**
`stream` is a synchronous generator. `astream` is an async generator. Use `astream` with `async for` in async contexts.

**Q19. What does `interrupt_before=["approval"]` do?**
It tells LangGraph to pause execution before running the `approval` node. The graph saves state to the checkpointer and returns control to the caller. The graph can later be resumed by calling `invoke`/`astream` with the same `thread_id`.

**Q20. How many agents does Orbit AI have?**
Seven: understanding, intent, clarification_halt, memory, planning, tool_router, approval. Each is a separate LangGraph node with a single responsibility.

---

### Intermediate Questions (20)

**Q21. How does `add_conditional_edges` work in Orbit?**
After `intent_agent` completes, LangGraph calls the router function: `lambda state: "clarification_halt" if state.get("clarification_needed") else "memory"`. The returned string is used as a key in the routing dict `{"clarification_halt": "clarification_halt", "memory": "memory"}` to determine the next node. If `clarification_needed` is True, routes to `clarification_halt`; otherwise routes to `memory`.

**Q22. What does `interrupt_before` mean precisely?**
"Before" means the target node has NOT run. When `interrupt_before=["approval"]` triggers, the graph checkpoints the state that exists after `tool_router` completes (which includes `pending_action_ids`). The `approval_agent` function is never called during this invocation. On resume, it runs `approval_agent` with the checkpointed state.

**Q23. How does graph resumption work?**
After an interrupt:
1. Client calls `POST /api/captures/{id}/approve` with decisions
2. Endpoint loads `run_id` from database, reconstructs `config = {"configurable": {"thread_id": run_id}}`
3. Updates state with decisions: `graph.invoke({"decisions": {...}}, config=config)`
4. LangGraph loads the checkpoint for `thread_id`, merges `{"decisions": {...}}` into state
5. Skips already-executed nodes, runs from `approval_agent` onward
6. Returns final state

**Q24. What is `stream_mode="updates"` vs `"values"`?**
`"updates"`: yields `{node_name: changed_fields}` — tells you which node just ran and what changed. `"values"`: yields full `AgentState` — tells you the complete current state after each node. Orbit uses `"updates"` for SSE (need node name for UI) and `"values"` for blocking endpoint (need complete final state).

**Q25. How do nodes communicate with each other in Orbit?**
Through `AgentState`. Node A writes fields to state (returns dict), Node B reads those fields as part of the state it receives. There is no direct function call between nodes. The graph handles the handoff.

**Q26. What is the format of an `astream("updates")` event?**
A Python dict with one key: the node that just completed, mapped to a dict of the state fields that node changed. Example: `{"understanding": {"extracted_entities": [...], "capture": {...}}}`. After parsing: `node_name = list(event.keys())[0]`.

**Q27. What does `clarification_halt` actually do?**
It's a no-op node. The function returns an empty dict or `{}`. Its purpose is to provide a named graph node for the conditional edge to route to, so that LangSmith records it as a span and the SSE stream can emit an event for it. After it "runs", the graph hits `END`.

**Q28. How does the frontend know which agents have completed?**
Orbit's `generate()` async generator in captures.py yields one SSE event per completed node: `event: agent\ndata: {"agent": "understanding", "status": "done"}`. The frontend reads these events and marks each pipeline node as complete in the visualization.

**Q29. What happens in the graph when `clarification_needed=True`?**
1. `intent_agent` runs, sets `clarification_needed=True` and `clarification_reason="..."` in returned dict
2. LangGraph merges this into state
3. Conditional edge router evaluates: `state.get("clarification_needed")` → True
4. Routes to `clarification_halt`
5. `clarification_halt_node` runs (no-op)
6. Graph hits `END`
7. FastAPI endpoint checks state, finds `clarification_needed=True`, returns HTTP 422 with `clarification_reason`

**Q30. What is `thread_id` and why is it important?**
`thread_id` is the key under which MemorySaver stores checkpoints. Two invocations with the same `thread_id` share checkpoint history and can resume each other. Two invocations with different `thread_ids` are completely isolated. In Orbit, `thread_id = run_id` — one unique ID per user session.

**Q31. Can two graph invocations with the same `thread_id` run concurrently?**
No — this would cause a race condition on checkpoint writes. LangGraph assumes sequential invocations per `thread_id`. In Orbit this is safe: the first invocation (stream) runs to the interrupt point, then the second invocation (approve) resumes. They cannot overlap because the first invocation returns before the second starts (the interrupt returns control to FastAPI).

**Q32. What does `MemorySaver` store at each checkpoint?**
The full `AgentState` dict plus LangGraph metadata: checkpoint ID, timestamp, channel versions (version counters per state field), and pending sends. It's a complete snapshot, not a delta.

**Q33. How do you pass initial state to the graph?**
Via the first argument to `invoke`/`astream`:
```python
initial_state = {"input_content": content, "input_type": "image", "run_id": run_id}
app.astream(initial_state, config=config)
```
LangGraph merges this dict into a blank `AgentState` to create the starting state.

**Q34. What is LangSmith and how does Orbit configure it?**
LangSmith is Anthropic/LangChain's tracing platform. Orbit configures it via the LangGraph config:
```python
config = {
    "run_name": "orbit-pipeline",
    "metadata": {"orbit_run_id": run_id, "input_type": input_type},
    "tags": ["orbit", "stream"]
}
```
Each node execution becomes a LangSmith span. All spans from one `thread_id` are grouped under one trace. `run_name` labels the trace.

**Q35. What is the routing dict in `add_conditional_edges`?**
The third argument to `add_conditional_edges`. It maps the router function's return values to node names. Format: `{"router_returns_this": "go_to_this_node"}`. It serves as compile-time validation: LangGraph checks that all dict values are valid node names at compile time.

**Q36. What is the `trace` field in AgentState?**
A list that accumulates agent execution metadata throughout the pipeline (which agents ran, timing, intermediate outputs). Used for LangSmith enrichment and debugging. Each agent appends its trace entry.

**Q37. What happens if a node raises an exception?**
LangGraph propagates the exception to the `astream` caller. In captures.py, this causes the `generate()` generator to raise, which FastAPI translates to a 500 response. The checkpoint at the last successful node is still stored in MemorySaver (the exception node's checkpoint is not written).

**Q38. How does the approval endpoint know which `thread_id` to resume?**
The capture ID from the URL is looked up in the database to retrieve the `run_id`. Since `thread_id = run_id`, the endpoint can reconstruct: `config = {"configurable": {"thread_id": run_id}}`.

**Q39. Why does `clarification_halt` edge go to `END` not back to `understanding`?**
Because the clarification needs to come from the user — the system cannot auto-resolve it. Looping back to `understanding` would just re-process the same ambiguous input and produce the same `clarification_needed=True` result infinitely. The correct flow is: return 422 to the client, user provides clarification, client submits a new capture with the clarification included.

**Q40. How do you visualize the Orbit graph?**
```python
from graph import app
print(app.get_graph().draw_mermaid())
```
Outputs a Mermaid diagram string showing all nodes and edges, which can be rendered at mermaid.live.

---

### Advanced Questions (20)

**Q41. How does MemorySaver handle concurrent requests?**
MemorySaver is not thread-safe out of the box. For concurrent requests (different `thread_id`s), Python's GIL provides some protection, but the correct production solution is to use an async-safe checkpointer (AsyncSqliteSaver, AsyncRedisSaver) that uses proper locking per thread_id. In Orbit's demo context with a single worker, this is not a concern.

**Q42. What are reducers in LangGraph and does Orbit use them?**
Reducers define how new values are merged into existing state values. The default (no annotation) is replace: `new_state[key] = new_value`. Custom reducers are defined with `Annotated[type, reducer_fn]` in the TypedDict. Orbit uses default replace semantics throughout — no field accumulates across nodes.

**Q43. How would you add parallel execution in LangGraph?**
Using the `Send` API for fan-out:
```python
from langgraph.types import Send

def fan_out_to_items(state):
    return [Send("process_item", {"item": item}) for item in state["items"]]

builder.add_conditional_edges("planning", fan_out_to_items)
```
Each `Send` creates a parallel branch. Results are aggregated via a reducer on the state field. This would be appropriate if Orbit needed to process each captured item independently in parallel.

**Q44. How does resumption know which nodes to skip?**
LangGraph stores `channel_versions` in each checkpoint — a version counter per state field. When resuming, it replays from the checkpoint, which already has the output of all completed nodes baked into the state. The graph does not re-execute nodes whose output is already in the checkpointed state. The interrupt point acts as a cursor: nodes before the interrupt have their output in state, nodes after have not run.

**Q45. What are the internal LangGraph events in `stream_mode="debug"`?**
Debug mode yields events like: `on_chain_start`, `on_chain_end` (for each node), `on_checkpoint` (on each checkpoint write), `on_chain_stream` (for streamed node outputs). These are the raw LangGraph execution events, much more verbose than `"updates"`.

**Q46. How would you add dynamic routing where the number of next nodes is not known at compile time?**
Use `add_conditional_edges` with a router function that returns different values, and a routing dict that covers all cases. For truly dynamic fan-out (variable number of parallel branches), use the `Send` API where the router returns a list of `Send` objects instead of a string.

**Q47. What is the difference between `GraphInterrupt` and a regular exception?**
`GraphInterrupt` is LangGraph's internal signal for interrupt checkpoints. It is caught internally by LangGraph's execution loop, which writes the checkpoint and returns normally to the caller (no exception propagates). A regular exception in a node propagates up to the caller as an unhandled error.

**Q48. Can you have multiple `interrupt_before` points in one graph?**
Yes: `interrupt_before=["node_a", "node_b"]` adds interrupt points at both nodes. The graph pauses at whichever comes first in the execution path. For Orbit, only `approval` is an interrupt point, but you could add one before `memory` for a "confirm before storing" use case.

**Q49. How does LangGraph validate the graph at compile time?**
It checks: (1) all node names in edges exist in the node registry, (2) all routing dict values in conditional edges are valid node names or END, (3) the entry point is set, (4) there are no dangling nodes with no outgoing edges (except nodes that edge to END). It does NOT check for infinite loops.

**Q50. What happens to the checkpoint if the graph completes normally (no interrupt)?**
The final state is written as the last checkpoint. It remains in MemorySaver (or whatever checkpointer) indefinitely (until the process restarts for MemorySaver). Subsequent calls with the same `thread_id` would see this completed state. In Orbit, each request uses a new `run_id` = new `thread_id`, so completed states don't interfere with new requests.

**Q51. How do you inject state updates on resumption after an interrupt?**
Pass a dict as the first argument to `invoke`/`astream` on resumption:
```python
graph.invoke({"decisions": user_decisions}, config={"configurable": {"thread_id": run_id}})
```
LangGraph merges this dict into the checkpointed state before running the first post-interrupt node (`approval_agent`). This is how user decisions reach the approval agent.

**Q52. What is the difference between LangGraph and LangChain's LCEL chains?**
LCEL (LangChain Expression Language) chains are linear pipelines with `|` operator. They don't support conditional routing, shared state, checkpointing, or interrupt/resume. LangGraph uses LCEL internally for some node implementations but provides the graph layer on top. For Orbit's multi-agent pipeline with conditional routing and human-in-the-loop, LCEL alone would be insufficient.

**Q53. How would you add observability to see which nodes are slowest?**
LangGraph already emits LangSmith spans per node with timing. Additionally, each agent could set a `trace` entry with `time.time()` at start/end. For production, you'd add Prometheus metrics with a `node_execution_time` histogram labeled by node name, and instrument with middleware in the `generate()` generator.

**Q54. How does LangGraph handle state that is not in the TypedDict?**
It ignores it. If a node returns a key that is not in `AgentState`, LangGraph raises a `KeyError` at merge time (strict TypedDict validation). All state fields must be declared in the TypedDict.

**Q55. What is the `__start__` node and when does it matter?**
`__start__` (aliased as `START`) is the graph's implicit entry node. It receives the initial state passed to `invoke` and edges to the first real node. You typically interact with it only via `set_entry_point` or when explicitly building a routing dict that includes the initial dispatch.

**Q56. How do you test a LangGraph graph in isolation?**
Invoke individual nodes directly (they're just async functions). For end-to-end: call `graph.invoke(test_state, config={"configurable": {"thread_id": "test-thread-1"}})` with a mock checkpointer or MemorySaver. Assert on the returned state. For interrupt testing, invoke once (runs to interrupt), check checkpointed state, then invoke again with decisions (runs from approval).

**Q57. Can you swap out MemorySaver for a persistent checkpointer without changing graph.py?**
Yes — almost. Change:
```python
# from:
checkpointer = MemorySaver()
# to:
from langgraph.checkpoint.aiosqlite import AsyncSqliteSaver
checkpointer = AsyncSqliteSaver.from_conn_string("checkpoints.db")
```
The `builder.compile(checkpointer=checkpointer)` line is unchanged. All other code is unchanged. This is the upgrade path for production persistence.

**Q58. What is the graph's behavior if you call `invoke` with a `thread_id` that already has a completed state?**
LangGraph loads the completed checkpoint and, finding no pending interrupt and the graph already at END, returns the completed state immediately without re-executing any nodes. To re-run the graph for the same thread, you'd need a new `thread_id` or use `SubGraphs`.

**Q59. How does LangGraph handle the `async for` loop in astream when an interrupt fires?**
The async generator exits (stops yielding). The caller's `async for` loop terminates normally (no exception). The graph state at the interrupt point is saved to the checkpointer. In Orbit's `generate()`, the `async for event in app.astream(...)` loop simply ends at the interrupt, which causes `generate()` to return (closing the SSE stream).

**Q60. What are channel versions and why do they matter?**
Channel versions are per-field version counters in the checkpoint. They track which version of each state field is "current." On resumption, LangGraph uses these to determine which nodes' outputs are already incorporated and which nodes need to run. They are an implementation detail of LangGraph's MVCC-style state management.

---

### LangGraph-Specific Questions (20)

**Q61. What is the difference between StateGraph and MessageGraph?**
`MessageGraph` is a subclass of `StateGraph` where the state is exactly `list[BaseMessage]`. It has built-in reducers that append messages. It's designed for chat applications. `StateGraph` is general-purpose — the state can be any TypedDict. Orbit uses `StateGraph` because its state is a structured dict, not a message list.

**Q62. What is LangGraph vs LangChain?**
LangChain provides components: LLM wrappers, prompt templates, chains, tools. LangGraph provides the graph execution layer: stateful multi-agent orchestration, conditional routing, checkpointing, streaming. LangGraph is built on LangChain but solves a different problem. You can use LangGraph without LangChain (using raw LLM APIs in nodes), but Orbit uses Claude Haiku via LangChain's ChatAnthropic in some agents.

**Q63. What is the LangGraph `Send` API?**
`Send("node_name", state_dict)` creates a parallel branch to `node_name` with `state_dict` as its input. Used with `add_conditional_edges` to fan out to multiple parallel node executions. The results are merged back via reducers. Not used in Orbit (sequential pipeline), but the answer shows you know advanced LangGraph.

**Q64. How do you add a subgraph in LangGraph?**
A subgraph is a compiled `StateGraph` that is added as a node to a parent graph:
```python
subgraph = sub_builder.compile()
parent_builder.add_node("sub", subgraph)
```
The subgraph runs as a single node from the parent's perspective but internally executes its own node sequence.

**Q65. What is `CompiledStateGraph` vs `StateGraph`?**
`StateGraph` is the builder — you add nodes and edges but cannot execute. `builder.compile()` returns `CompiledStateGraph`, which has `invoke`, `ainvoke`, `stream`, `astream`, `get_state`, `update_state`, `get_graph` methods. After compilation, the graph topology is frozen.

**Q66. How does LangGraph's streaming compare to OpenAI's streaming?**
OpenAI's streaming (`stream=True`) streams individual LLM tokens. LangGraph's streaming (`astream`) streams node-level events — not individual tokens. LangGraph's `stream_mode="messages"` can stream individual LLM tokens from within nodes, combining both levels.

**Q67. What is `get_state` on a compiled graph?**
`app.get_state(config)` returns the current checkpoint state for a given `thread_id` without executing the graph. Useful for reading pending action state before an approval call. Returns a `StateSnapshot` with `.values` (the state dict) and `.next` (list of nodes to run on next invoke).

**Q68. What is `update_state` on a compiled graph?**
`app.update_state(config, values)` modifies the checkpointed state for a `thread_id` without running the graph. Used for human edits: the user can modify `decisions` in the checkpoint state before resuming. This is an alternative to passing updates as the first argument to `invoke` on resumption.

**Q69. What is a LangGraph `Pregel` execution model?**
LangGraph is internally built on Pregel, a Google graph processing model where all nodes in a "superstep" run in parallel, then state is synchronized. LangGraph adapts this for sequential graphs by running one node per superstep. This is why the internal events are called "supersteps" in debug mode.

**Q70. How does LangGraph handle cycles (loops)?**
LangGraph supports cycles. A node can have an edge back to an earlier node. To prevent infinite loops, you add a counter to `AgentState` and a conditional edge that terminates when the counter exceeds a limit. LangGraph does not automatically detect or prevent infinite loops.

**Q71. What is `interrupt_before` vs `human_in_the_loop` tool pattern?**
`interrupt_before` pauses the graph at a specific node and returns control to the caller. The "tool pattern" gives an agent a special tool call that signals the need for human input, which the runner detects. LangGraph's `interrupt_before` is cleaner for known-upfront checkpoints like Orbit's approval step.

**Q72. How do you pass configuration to nodes in LangGraph?**
Via `RunnableConfig` — the second argument to nodes if they accept it:
```python
async def node(state: AgentState, config: RunnableConfig) -> dict:
    thread_id = config["configurable"]["thread_id"]
```
This lets nodes access the `thread_id`, tags, metadata, and other config values set at invoke time.

**Q73. What LangGraph version does Orbit target?**
LangGraph 0.2.x. Key APIs: `StateGraph`, `MemorySaver`, `interrupt_before`, `astream(stream_mode="updates")`. These are stable APIs in the 0.2 series.

**Q74. How does LangGraph serialize state for checkpointing?**
MemorySaver stores Python objects directly in memory (no serialization). For persistent checkpointers (Sqlite, Postgres), LangGraph serializes state using msgpack or JSON. All values in `AgentState` must be serializable. Lists, dicts, strings, ints are fine. Custom objects need `__reduce__` or should be converted to dicts.

**Q75. What is the `__interrupt__` signal in LangGraph internals?**
When `interrupt_before` triggers, LangGraph raises `langgraph.errors.GraphInterrupt` internally. This is caught by the graph's execution loop (not by user code), which writes the checkpoint, serializes the interrupt state, and returns normally to the outer `invoke`/`astream` caller. User code never sees this exception.

**Q76. Can you have conditional edges AND unconditional edges from the same node?**
No. A node has exactly one outgoing edge definition. You choose either `add_edge` (unconditional) or `add_conditional_edges` (conditional). If you need "always go to A, sometimes also to B," you'd restructure so that the conditional happens at a subsequent node.

**Q77. How does LangGraph differ from Airflow/Prefect/Luigi for workflow orchestration?**
Airflow/Prefect/Luigi are for batch data pipelines — DAGs of tasks with retry logic, scheduling, and worker distribution. LangGraph is for real-time, stateful, LLM-powered agent execution — it handles shared state, LLM streaming, human-in-the-loop interrupts, and millisecond-latency node transitions. They solve different problems.

**Q78. How do you debug a LangGraph graph that's routing incorrectly?**
1. Check `stream_mode="debug"` output — see which nodes ran and their state at each step
2. Check LangSmith trace — see inputs/outputs per span
3. Add `print(state.get("clarification_needed"))` inside the router function
4. Call `app.get_state(config)` after the run to inspect final state
5. Use `app.get_graph().draw_mermaid()` to verify graph topology is correct

**Q79. What is `add_conditional_edges` routing dict aliasing used for?**
When you want the router to return a generic name that maps to a node with a different name:
```python
add_conditional_edges("node", router, {"CONTINUE": "memory", "STOP": "clarification_halt"})
```
The router returns `"CONTINUE"` or `"STOP"` (semantic labels), and the dict translates to actual node names. Makes the router function more readable. In Orbit, keys and values are identical for simplicity.

**Q80. What is the `interrupt_before` behavior when the interrupted node is inside a subgraph?**
If the `interrupt_before` node is in a subgraph, the interrupt pauses the subgraph's execution at that node. The parent graph sees the subgraph node as interrupted. Resumption works the same way — same `thread_id`, next `invoke` resumes the subgraph from the interrupt point.

---

### Multi-Agent Questions (20)

**Q81. What is the supervisor pattern in multi-agent systems?**
A supervisor agent orchestrates worker agents — it decides which workers to call, in what order, and when to stop. In Orbit, the LangGraph graph acts as the supervisor (deterministic routing), with `tool_router` acting as an intra-pipeline supervisor that validates and prioritizes actions from planning.

**Q82. What is the worker pattern?**
Each agent has a specialized capability and does not orchestrate others. Workers receive their input from the supervisor (or shared state), do their job, and return output. In Orbit, all 7 agents are workers: focused, single-responsibility, non-orchestrating.

**Q83. How does Orbit differ from AutoGen architecturally?**
AutoGen: agents have inboxes, send messages to each other, conversations can loop, routing is emergent from agent behavior. Orbit: agents communicate via shared TypedDict state, routing is deterministic (defined in graph.py), the pipeline runs exactly once per capture, no agent calls another agent. Orbit's pattern is appropriate for structured data extraction pipelines; AutoGen's is appropriate for open-ended collaborative reasoning.

**Q84. How does Orbit differ from CrewAI architecturally?**
CrewAI: an LLM-powered manager agent decides task assignment at runtime. Orbit: routing is programmatic (conditional edge lambda), no LLM decides which agent runs next. CrewAI is flexible for open-ended task decomposition; Orbit is correct for a fixed pipeline where the order is always known.

**Q85. What is agent communication via shared state vs message passing?**
Shared state: all agents read from and write to a central dict. Any agent can read any other agent's output. No explicit message routing. Simple, transparent, debuggable.
Message passing: agents send typed messages to other agents' queues. More decoupled, supports async agent-to-agent communication, harder to debug state.
Orbit uses shared state — appropriate for a linear pipeline where all agents need access to all prior outputs.

**Q86. How would you implement a retry pattern in Orbit's multi-agent pipeline?**
Add a `retry_count` field to `AgentState`. Add a conditional edge after the failing node:
```python
def route_after_planning(state):
    if not state.get("actions") and state.get("retry_count", 0) < 3:
        return "planning"  # retry
    return "tool_router"
builder.add_conditional_edges("planning", route_after_planning, {"planning": "planning", "tool_router": "tool_router"})
```
Increment `retry_count` inside the planning node on failure. This creates a loop, which LangGraph supports.

**Q87. What is human-in-the-loop and how does Orbit implement it?**
Human-in-the-loop: a human can inspect and modify agent behavior mid-pipeline. Orbit implements it via LangGraph's `interrupt_before=["approval"]`: the pipeline pauses before executing actions, the human reviews `pending_action_ids`, approves or rejects specific actions via the REST API, and the pipeline resumes with the human's `decisions` injected into state.

**Q88. What is the agent state schema in Orbit and who owns it?**
`AgentState` is a TypedDict. Utkarsh designed it. Key fields: `extracted_entities` (from understanding), `extracted_items` (from intent), `clarification_needed` + `clarification_reason` (from intent), `retrieval_context` (from memory), `actions` (from planning), `pending_action_ids` (from tool_router), `decisions` + `execution_results` (from approval).

**Q89. How does context propagate through Orbit's 7 agents?**
Via `AgentState`. `run_id` is in the initial state and available to every agent. `capture_id` is set by the understanding agent and used by downstream agents for database lookups. `retrieval_context` set by memory is read by planning. `actions` set by planning is read by tool_router. Every agent can read every prior agent's output.

**Q90. What is the difference between orchestrator agents and executor agents in Orbit?**
Orchestrator: manages flow (the LangGraph graph + tool_router supervisor). Executor: performs work (`approval_agent` makes REST API calls). In Orbit's architecture, most agents are analyzers/transformers; `approval_agent` is the only executor that actually changes external state.

**Q91. Why doesn't Orbit use a message bus (Kafka, RabbitMQ) between agents?**
Message buses are appropriate for distributed, decoupled microservices that run on different machines with independent scaling. Orbit's agents run in the same process, in sequence, with millisecond latency. A message bus would add network hops, serialization overhead, and operational complexity for no benefit. The LangGraph state dict is the appropriate coordination mechanism for co-located, synchronous agents.

**Q92. How would you add a new agent to Orbit's pipeline, say a `notification_agent` after approval?**
1. Write `async def notification_agent(state: AgentState) -> dict` with its logic
2. `builder.add_node("notification", notification_agent)` in graph.py
3. Change `builder.add_edge("approval", END)` to `builder.add_edge("approval", "notification")`
4. Add `builder.add_edge("notification", END)`
5. Add any new state fields to `AgentState` if needed
The rest of the graph is unchanged. LangGraph validates at compile time that all edges are valid.

**Q93. What is the sequential dependency chain in Orbit and why is it strict?**
`understanding → intent → memory → planning → tool_router → approval`
- Intent needs `extracted_entities` from understanding
- Memory needs `extracted_items` from intent (to know what to embed)
- Planning needs `retrieval_context` from memory (past context for action generation)
- Tool_router needs `actions` from planning (to validate)
- Approval needs `pending_action_ids` from tool_router (to present to user)
No pair of adjacent nodes can be parallelized without the downstream node waiting on the upstream node. The chain is total.

**Q94. How does Orbit handle agent failure gracefully?**
Currently: uncaught exceptions propagate to FastAPI, which returns a 500. For production: wrap each agent in try/except, set an `error` field in `AgentState`, and add conditional edges that route to an `error_handler` node. The error_handler could log to LangSmith, notify the user, and exit cleanly. The current approach is appropriate for a demo.

**Q95. What is the difference between an LLM call and an agent in Orbit?**
An LLM call is a single `client.chat.completions.create()` invocation. An agent is a LangGraph node function that may make zero, one, or multiple LLM calls, query databases, call REST APIs, and perform business logic. `understanding_agent` makes one Claude Haiku call. `memory_agent` makes an embedding API call and a PostgreSQL write. `clarification_halt` makes no calls.

**Q96. How does Orbit's pipeline ensure data integrity between agents?**
Via `AgentState` field contracts: each agent assumes specific fields are populated by prior agents. If `planning_agent` reads `state["retrieval_context"]` and memory failed to set it, planning will fail. In production, each agent should validate its required inputs at the start and raise descriptive errors. The TypedDict schema provides static typing guarantees at development time (via mypy), not runtime.

**Q97. What is the G3 guardrail and where does it sit relative to Daksh's graph?**
G3 is Abhay's guardrail that validates the request before the LangGraph graph runs. In captures.py, the flow is: (1) receive request, (2) run G3 check on `input_content`, (3) if G3 blocks → return 400 immediately, (4) if G3 passes → run `app.astream(...)`. G3 never enters the graph. This is a "pre-graph" guardrail, not a node.

**Q98. What multi-agent pattern does LangGraph call Orbit's approach?**
LangGraph documentation calls this the "sequential pipeline" or "chain" pattern — a fixed sequence of specialized agents, each handling one step. This contrasts with the "supervisor" pattern (dynamic routing by an LLM) and the "network" pattern (fully connected agents that can call each other).

**Q99. How does Orbit's approval node differ from a typical approval step in Airflow?**
Airflow approval: a DAG is paused, a human interface shows the task state, the human clicks "approve" in the Airflow UI, the DAG resumes. Orbit approval: LangGraph pauses at the interrupt point, the Orbit frontend (built specifically for Orbit) shows `pending_action_ids`, the user approves via the Orbit REST API, the graph resumes. Both are human-in-the-loop patterns; Orbit's is more tightly integrated with the application domain.

**Q100. How would you scale Orbit to handle 100 concurrent users?**
Key bottlenecks: (1) MemorySaver — replace with async PostgreSQL checkpointer (one row per checkpoint, thread_id as partition key), (2) FastAPI workers — deploy with `gunicorn -w 4 -k uvicorn.workers.UvicornWorker`, (3) LLM rate limits — add per-user rate limiting middleware, queue excess requests, (4) PostgreSQL connections — use `asyncpg` with a connection pool. The LangGraph graph code itself is stateless and scales horizontally.

---

### System Design Questions (20)

**Q101. How would you scale Orbit to 10,000 concurrent users?**
At 10K concurrent users, the key changes:
1. **Checkpointer**: Replace MemorySaver with a Redis-backed async checkpointer (Redis supports concurrent reads/writes, TTL for cleanup, horizontal scaling)
2. **LLM calls**: Implement request batching for Haiku calls, use async client with connection pooling, cache common entity extractions
3. **FastAPI**: Deploy behind a load balancer with 10+ Uvicorn workers or Kubernetes pods
4. **PostgreSQL**: Read replicas for retrieval, connection pooling via PgBouncer, database sharding by `user_id`
5. **SSE**: Use an SSE broker (Redis pub/sub) so any worker can push to any client's SSE connection regardless of which worker handled the initial request
6. **Approval state**: With Redis checkpointer, approval state persists across worker restarts and is accessible by any worker

**Q102. How do you add fault tolerance to Orbit's pipeline?**
1. **Retry logic**: Wrap LLM calls with `tenacity` (exponential backoff on 429/500)
2. **Circuit breaker**: Track failure rates per node, skip non-critical nodes (e.g., memory persistence) if downstream is failing
3. **Dead letter queue**: Failed captures go to a queue for manual review
4. **Idempotency**: Make each agent idempotent — re-running with same `run_id` should not duplicate database records
5. **Health checks**: FastAPI health endpoint checks LLM API reachability and PostgreSQL connectivity

**Q103. How would you replace LangGraph with a custom orchestrator?**
You'd need to implement:
1. Shared state dict passed between functions
2. Conditional routing logic (if/else)
3. State persistence per request (dict keyed by run_id)
4. Interrupt/resume (serialize state, reconstruct on resume)
5. Async generator for streaming
That's approximately 200-300 lines of infrastructure code. LangGraph provides all of this, plus LangSmith integration, graph validation, and a stable API. Replacing it would be justified only if you need capabilities LangGraph doesn't provide (e.g., distributed execution, custom serialization).

**Q104. How would you add monitoring to Orbit's graph?**
1. **LangSmith** (already configured): traces per run, per-node timing, LLM token counts
2. **Prometheus metrics**: `Counter` for runs started/completed/failed, `Histogram` for node execution times, `Gauge` for in-flight runs
3. **Node-level metrics**: instrument `generate()` to emit a metric after each SSE event
4. **Alerting**: Alert if `clarification_halt` rate exceeds threshold (input quality issue), if any node p95 latency exceeds SLO, if approval is pending > N minutes

**Q105. How would you add a new agent (e.g., `summarize_agent`) without breaking existing flow?**
1. Write the agent function
2. `builder.add_node("summarize", summarize_agent)` in graph.py
3. Decide where in the sequence it belongs (e.g., after `understanding`)
4. Change `builder.add_edge("understanding", "intent")` to `builder.add_edge("understanding", "summarize")` and add `builder.add_edge("summarize", "intent")`
5. Add new state fields to `AgentState` if needed
6. Recompile graph
Existing agents are unaffected — they read state fields that are still there. The only risk is if `summarize_agent` modifies a field that downstream agents depend on (e.g., overwrites `extracted_entities`). Code review for state field conflicts is the key risk mitigation.

**Q106. What would you change in production to make approval async (user approves via email link)?**
1. Replace MemorySaver with PostgreSQL checkpointer (state persists across restarts)
2. On interrupt, send email with approval link containing `run_id` + token
3. Email link hits `GET /approve?token=...` endpoint
4. Endpoint validates token, loads `run_id`, resumes graph
5. Graph completes, sends confirmation email
The LangGraph graph code is unchanged — only the checkpointer and the approval triggering mechanism change.

**Q107. How would you implement a feedback loop where planning can retry after tool_router rejects?**
Add `planning_retry_count` to `AgentState`. Add conditional edge after tool_router:
```python
def route_after_tool_router(state):
    if not state.get("pending_action_ids") and state.get("planning_retry_count", 0) < 2:
        return "planning"  # retry
    return "approval"
builder.add_conditional_edges("tool_router", route_after_tool_router, {"planning": "planning", "approval": "approval"})
```
In planning, increment `planning_retry_count` and use tool_router's rejection reason (stored in state) to generate better actions.

**Q108. How would you handle PII in Orbit's state?**
1. Mask PII fields (email, phone, name) in `AgentState` before writing to LangSmith via metadata filters
2. Encrypt `AgentState` at rest in the checkpointer
3. Add a `pii_detected` flag set by understanding_agent
4. Optionally add a `pii_redaction` node before memory persistence
5. Ensure `LANGCHAIN_HIDE_INPUTS=true` env var is set for sensitive deployments

**Q109. How would you A/B test different planning agents?**
1. Add an `experiment_group` field to `AgentState` (set at API entry)
2. Create `planning_agent_v2`
3. Add conditional edge after memory:
```python
def route_planning(state):
    return "planning_v2" if state.get("experiment_group") == "B" else "planning"
```
4. Track experiment group in LangSmith metadata
5. Compare action quality metrics across groups in LangSmith

**Q110. What is the blast radius if `tool_router_agent` has a bug?**
`tool_router` runs after `planning` and before `approval`. A bug could:
- **Incorrectly filter actions**: some valid actions never get `pending_action_ids` → never executed
- **Crash (exception)**: graph terminates at tool_router, FastAPI returns 500, user sees error
- **Corrupt state**: set `pending_action_ids` incorrectly → approval tries to execute invalid actions
Mitigation: thorough unit tests for `tool_router_agent` with known action inputs, integration tests for the full pipeline, and validation inside `approval_agent` that checks `pending_action_ids` are valid before executing.

**Q111. How would you implement distributed tracing across Orbit's graph?**
LangSmith already provides distributed tracing at the LLM call level. For full distributed tracing:
1. OpenTelemetry instrumentation on FastAPI (auto-instruments HTTP requests)
2. Pass `trace_context` (W3C TraceContext header) through `AgentState`
3. Instrument each agent function with `tracer.start_as_current_span(node_name)`
4. Export to Jaeger or Datadog
5. Correlate with LangSmith via `run_id` in span attributes

**Q112. How would you handle LangGraph's breaking changes in future versions?**
1. Pin LangGraph version in `requirements.txt`: `langgraph==0.2.x`
2. Maintain an integration test suite that tests the full graph end-to-end
3. Keep the graph compilation logic isolated in `graph.py` — only one file needs updating
4. Follow LangGraph's migration guides on version bumps
5. Evaluate breaking changes against the features actually used (StateGraph, MemorySaver, astream, interrupt_before) — Orbit uses stable core APIs

**Q113. How would you implement caching for repeated captures with identical content?**
1. Hash `input_content` → `content_hash`
2. Before running the graph, check Redis for `content_hash` → cached result
3. If hit: return cached state (skip entire graph)
4. If miss: run graph, store result in Redis with TTL, return result
5. LangGraph is unaffected — caching is at the API layer (captures.py), before `app.astream()`

**Q114. Explain the exact SSE byte format flowing over the connection.**
Each SSE event is:
```
event: agent\r\n
data: {"agent": "understanding", "status": "done"}\r\n
\r\n
```
The response `Content-Type` is `text/event-stream`. The HTTP response uses `Transfer-Encoding: chunked`. Each chunk contains one or more complete SSE events. The browser's built-in `EventSource` API parses these automatically. Orbit uses `fetch()` + `ReadableStream` directly (not EventSource), reading the response body as a stream of UTF-8 text, splitting on double newlines, and parsing the `data:` prefix.

**Q115. How would you implement rate limiting on Orbit's streaming endpoint?**
1. Add a Redis counter per user: `INCR orbit:ratelimit:{user_id}` with TTL of 60 seconds
2. If counter > limit (e.g., 10 requests/minute), return HTTP 429 before running graph
3. Implement as FastAPI middleware or dependency
4. Use sliding window algorithm for smoother rate limiting
5. Exempt admin users via JWT claims

**Q116. How would you add a circuit breaker for Claude API calls?**
Using `circuitbreaker` or `pybreaker` library:
```python
from circuitbreaker import circuit

@circuit(failure_threshold=5, recovery_timeout=30)
async def call_claude_haiku(prompt):
    return await claude_client.messages.create(...)
```
If Claude returns 5 consecutive errors, the circuit opens and all calls immediately raise `CircuitBreakerError` (no waiting). After 30 seconds, circuit half-opens to test recovery. This prevents cascade failures when Claude API is degraded.

**Q117. How would you implement a "dry run" mode for the graph?**
Add `dry_run: bool` to `AgentState`. In `approval_agent` and any state-mutating agents, check `state.get("dry_run")` and skip actual mutations. The graph runs all the way through, all agents analyze, planning generates actions, tool_router validates — but approval does not execute. Useful for testing and previewing what Orbit would do.

**Q118. How would you migrate from MemorySaver to a PostgreSQL checkpointer without downtime?**
1. Add AsyncPostgresSaver alongside MemorySaver
2. On each checkpoint write, write to both (dual-write)
3. On checkpoint read, try PostgreSQL first, fall back to MemorySaver
4. After all in-flight approvals complete (drain), switch to PostgreSQL-only read
5. Remove MemorySaver after validation period
This is the standard dual-write migration pattern. In-flight approvals during migration are handled by the fallback read.

**Q119. What happens to Orbit's graph under Python's GIL for concurrent async operations?**
Python's GIL is not a concern for async I/O. `asyncio` uses cooperative multitasking — nodes yield control at `await` points, not by releasing the GIL. Multiple concurrent graph executions run interleaved on a single event loop thread. CPU-bound operations (e.g., large embedding computations) would be GIL-constrained; these should run in a thread pool via `asyncio.run_in_executor`.

**Q120. How would you implement a time-to-live for pending approvals?**
1. When storing `run_id` in the database, store `approval_expires_at = now() + timedelta(hours=24)`
2. Add a background task (APScheduler or Celery beat) that queries for expired approvals
3. For expired approvals: call `app.invoke({"decisions": "expired"}, config=...)` to force graph to terminate
4. The `approval_agent` handles `decisions="expired"` by returning a cancellation result
5. Notify user that their approval window expired

---

## 7. Cross-Module Questions

### How does Daksh's graph receive AgentState that Utkarsh designed?

`StateGraph(AgentState)` — the TypedDict is passed as a type parameter at construction. Utkarsh defined all the fields (`input_content`, `input_type`, `run_id`, `capture`, `extracted_entities`, `extracted_items`, `capture_id`, `extracted_item_ids`, `actions`, `pending_action_ids`, `decided_action_ids`, `decisions`, `execution_results`, `retrieval_context`, `clarification_needed`, `clarification_reason`, `trace`). Daksh's graph uses this as the state contract. Every node in Daksh's graph receives and returns against this schema. If Utkarsh adds a field, Daksh's graph automatically makes it available to all nodes — no changes to graph.py needed.

### How does the planning agent (Jash's domain) get invoked inside Daksh's graph?

`builder.add_node("planning", planning_agent)` — Jash's `planning_agent` function is passed as a callable to Daksh's node registration. Inside the graph, after `memory_agent` completes and LangGraph merges its state updates, LangGraph calls `await planning_agent(current_state)`. The function returns a dict with `actions`, which is merged into `AgentState`. Jash owns what happens inside `planning_agent`; Daksh owns the fact that it is called after `memory` and before `tool_router`.

### How does Abhay's guardrail run BEFORE `graph.invoke()`?

In `captures.py`:
```python
@router.post("/api/captures/stream")
async def stream_capture(request: CaptureRequest):
    # Step 1: Abhay's G3 guardrail
    guardrail_result = await g3_guardrail(request.input_content)
    if guardrail_result.blocked:
        raise HTTPException(status_code=400, detail=guardrail_result.reason)
    
    # Step 2: Generate run_id
    run_id = str(uuid4())
    
    # Step 3: Daksh's graph
    async def generate():
        async for event in app.astream(initial_state, config=config, stream_mode="updates"):
            ...
    
    return StreamingResponse(generate(), media_type="text/event-stream")
```
G3 runs in the FastAPI request handler before `app.astream()` is called. If G3 blocks, `app.astream()` is never called. G3 is a pre-graph gate, not a node.

### What happens in the graph when G3 blocks an action inside planning?

This scenario addresses G3 as a per-action guardrail inside planning (not the pre-graph G3). Inside `planning_agent`, before adding an action to the `actions` list, it calls a guardrail check per action. If the guardrail blocks an action:
1. That action is not added to `actions`
2. `planning_agent` returns `actions` without the blocked action
3. LangGraph merges this into state: `AgentState.actions` = filtered list
4. `tool_router` receives only the non-blocked actions
5. The blocked action never reaches `approval`

The graph continues normally — the guardrail is internal to `planning_agent`, transparent to LangGraph's routing.

### How does the approval node enable Abhay's human-in-the-loop pattern?

`interrupt_before=["approval"]` is Daksh's graph configuration. What it does for Abhay's pattern:
1. Graph pauses before `approval_agent` runs
2. `pending_action_ids` (tool_router output) is checkpointed
3. Frontend shows user the pending actions
4. User (human) reviews and submits `decisions` (approve/reject per action)
5. `POST /api/captures/{id}/approve` endpoint calls `graph.invoke({"decisions": decisions}, config=...)` 
6. LangGraph loads checkpointed state, merges `decisions`, runs `approval_agent`
7. `approval_agent` (Abhay's code) executes only the approved actions

Daksh's `interrupt_before` is the mechanism; Abhay's `approval_agent` is the handler. Without Daksh's interrupt configuration, there would be no pause — approval would run immediately after tool_router with no human review.

### How does Utkarsh's MemorySaver checkpoint interact with Daksh's `interrupt_before`?

Utkarsh designed the `AgentState` schema and configured the MemorySaver instance. Daksh's `interrupt_before=["approval"]` relies on the checkpointer: when the interrupt fires, LangGraph calls `checkpointer.put(thread_id, current_state)` to save the state before returning. When the approval endpoint resumes the graph, LangGraph calls `checkpointer.get(thread_id)` to load the state. The interaction is:
- MemorySaver (Utkarsh's config) provides the persistence mechanism
- `interrupt_before` (Daksh's config) tells LangGraph when to use that persistence mechanism
- The `thread_id = run_id` mapping (Daksh's design) ensures the resume call loads the correct checkpoint

---

## 8. Demonstration Script (3–5 Minutes)

---

*Presenter speaks — word for word:*

"Let me walk you through what I built for Orbit AI's orchestration layer — the code that takes a user's captured content and turns it into an approved action plan.

When a user submits a capture — say, a photo of a handwritten to-do list — the request hits our FastAPI endpoint in captures.py. The first thing that happens is Abhay's G3 guardrail checks the content for safety violations. If it passes, I generate a UUID called `run_id`. This ID will follow the request all the way through — it becomes the LangGraph thread ID, the LangSmith trace ID, and the database record key.

[Point to graph.py]

Here's the graph I compiled. I used LangGraph's StateGraph class, parameterized with our AgentState TypedDict — that's the shared state schema that all 7 agents read from and write to.

The pipeline starts at `understanding_agent`: OCR plus named entity recognition using Claude Haiku. The output — `extracted_entities` and `extracted_items` — flows via shared state to `intent_agent`, which validates deadlines, truncates urgency labels, and — crucially — sets `clarification_needed` to True if anything is ambiguous.

[Point to conditional edge]

Here's the only conditional branch in the graph. After intent runs, a router function checks `state.get('clarification_needed')`. True routes to `clarification_halt` — a terminal no-op that causes the API to return HTTP 422 with a reason. False continues to `memory_agent`, which embeds the items and persists them to PostgreSQL.

From memory, we go to `planning_agent` — that's Jash's domain. Planning uses the retrieval context from memory to generate a structured action list. `tool_router` validates those actions and builds `pending_action_ids`.

[Point to compile call]

Now here's the key design decision: I compiled the graph with `interrupt_before=['approval']`. This means the graph executes up to and including tool_router, writes the full state to MemorySaver, and pauses. The user sees their proposed actions. They approve or reject specific items. That decision goes to a separate REST endpoint, which resumes the graph — LangGraph loads the checkpointed state, merges the user's decisions, and runs `approval_agent`, which executes the approved actions via REST API.

[Point to captures.py]

For real-time feedback, the streaming endpoint uses `astream(stream_mode='updates')`. Each time a node completes, LangGraph yields a dict like `{'understanding': {'extracted_entities': [...]}}`. My `generate()` async generator parses the node name and yields an SSE event: `event: agent / data: {"agent": "understanding", "status": "done"}`. The frontend reads this stream and animates the pipeline nodes one by one as they complete.

The result: 7 specialized agents, deterministic routing, human-gated execution, real-time progress — all in about 30 lines of graph definition code."

---

## 9. Examiner Challenge Scenarios with Strong Defenses

---

### "What happens if the understanding agent crashes mid-run?"

**Strong answer:**

`understanding_agent` is an async function in a LangGraph node. If it raises an unhandled exception (e.g., Claude API timeout, malformed OCR response), LangGraph does not catch it — it propagates the exception up through `app.astream()`. In Orbit's `generate()` generator, this causes the `async for` loop to raise. FastAPI's `StreamingResponse` handles this by closing the SSE connection with an error.

The state at that point: MemorySaver has a checkpoint from the previous successful node (or from the initial state if understanding is the first node). Since understanding is the entry point, the checkpoint would be the initial state — essentially, nothing useful was saved.

From the client's perspective: the SSE stream closes abruptly (or with an error event). The frontend should handle this with an error state in the pipeline visualization.

For production: wrap the Claude API call in `understanding_agent` with `tenacity` retry logic (3 retries, exponential backoff). Add a try/except that sets `error: "understanding_failed"` in state and routes to an `error_handler` node via a conditional edge. This gives graceful degradation instead of a hard crash.

---

### "Why not use a single Claude Opus call for everything?"

**Strong answer:**

Three reasons: cost, latency, and correctness.

**Cost**: A single Claude Opus call for OCR + NER + intent validation + memory retrieval + action planning + tool routing would be at Opus pricing for every request. Orbit uses Haiku for understanding (structured extraction, well-defined output format) and planning uses a lower-tier model for action generation. The pipeline costs are much lower.

**Latency**: A single Opus prompt that does everything sequentially has no opportunity for parallelism and the context window grows with each task. Seven focused prompts, each with a small, specific context, are faster individually. The sequential pipeline's total latency is roughly the sum of 7 small calls, which is comparable to one large Opus call.

**Correctness**: Prompting one model to do OCR + deadline validation + memory retrieval + action planning in one shot creates a massive prompt with competing concerns. Each concern degrades the others. Separation into specialized agents with specific prompts, specific tools, and specific outputs means each agent can be optimized, tested, and improved independently. If planning quality is poor, we tune planning's prompt without touching understanding. A monolithic prompt makes this impossible.

Additionally: Opus doesn't do OCR natively. Understanding uses Claude's vision/tool_use capabilities for structured extraction — that's a specific capability, not just general reasoning.

---

### "How would you add a new agent without breaking existing flow?"

**Strong answer:**

Say we want to add a `sentiment_agent` after `understanding` and before `intent`:

1. Write `async def sentiment_agent(state: AgentState) -> dict` — returns `{"sentiment": "positive/negative/neutral"}`
2. Add `sentiment: Optional[str]` to `AgentState` (Utkarsh's file)
3. In `graph.py`:
   - `builder.add_node("sentiment", sentiment_agent)`
   - Change `builder.add_edge("understanding", "intent")` to `builder.add_edge("understanding", "sentiment")`
   - Add `builder.add_edge("sentiment", "intent")`
4. Recompile graph

Existing agents are unaffected:
- `intent_agent` now receives a state with `sentiment` populated — it can ignore it or use it
- `memory_agent`, `planning_agent`, etc. are unchanged
- All existing edges are unchanged except the one we explicitly modified

Risk: if we accidentally modify a field that downstream agents depend on (e.g., `sentiment_agent` overwrites `extracted_entities`), downstream agents break. Mitigation: each agent only writes to fields it owns. Code review for state field ownership conflicts.

---

### "Your checkpoint is in memory — what happens on server restart?"

**Strong answer:**

All MemorySaver checkpoints are lost. This means:

1. Any in-flight approval (graph interrupted, waiting for user decision) is unresumable after restart
2. The user would submit their approval decision, the endpoint would try to load the checkpoint, find nothing for that `thread_id`, and LangGraph would either raise an error or start a new run with empty state

**Current mitigation for demo**: This is acceptable because Orbit is a demo/capstone project. The approval window is seconds to minutes (user is looking at the screen). Server restarts during an active session are rare.

**Production mitigation**: Replace MemorySaver with `AsyncPostgresSaver`:
```python
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
checkpointer = AsyncPostgresSaver.from_conn_string(DATABASE_URL)
app = builder.compile(checkpointer=checkpointer, interrupt_before=["approval"])
```
One line change. PostgreSQL persists across restarts. The approval can happen hours or days later. This is the documented LangGraph upgrade path.

**Defense of current choice**: For the capstone, MemorySaver is the right choice. It requires no database schema migrations, no connection management, and no external dependencies. The upgrade path is trivial and well-documented.

---

### "How does this scale to 10,000 concurrent users?"

**Strong answer:**

At 10K concurrent users with a streaming SSE pipeline, the bottlenecks are:

**1. MemorySaver**: Replace with Redis-backed checkpointer. Redis handles high concurrency with O(1) get/set, TTL for cleanup, pub/sub for SSE brokering across workers.

**2. SSE connections**: Each open SSE connection holds an async generator and an HTTP connection. At 10K concurrent, you need non-blocking async I/O throughout — Orbit already uses `asyncio` + `astream`, so this scales well. Deploy with `uvicorn --workers 8` or Kubernetes with horizontal pod autoscaling.

**3. LLM rate limits**: Claude API has per-account rate limits. At 10K users, you need request queuing (Redis queue + worker pool), retry with exponential backoff, and potentially multiple API keys with load balancing.

**4. PostgreSQL**: Connection pool via `asyncpg` pool (50 connections per worker), read replicas for retrieval-heavy queries, write batching for memory persistence.

**5. State size**: Each `AgentState` is kilobytes. At 10K concurrent, that's megabytes of in-memory state per worker — manageable. With Redis checkpointer, state is offloaded to Redis.

**6. SSE across workers**: If the initial stream and the approval are handled by different FastAPI workers (load balancer), the SSE connection to worker A is gone by the time approval hits worker B. Solution: SSE brokering via Redis pub/sub. Worker B publishes the "approval done" event to Redis channel, worker A's SSE generator (subscribed to that channel) pushes it to the client.

The graph code itself is stateless and horizontally scalable. The infrastructure changes are external to `graph.py`.

---

### "Explain the exact bytes flowing over the SSE connection"

**Strong answer:**

The HTTP response has:
- `Content-Type: text/event-stream`
- `Cache-Control: no-cache`
- `Connection: keep-alive`
- `Transfer-Encoding: chunked`

Each SSE event is transmitted as a chunk. For the `understanding` node completing:

```
38\r\n
event: agent\r\n
data: {"agent": "understanding", "status": "done"}\r\n
\r\n
\r\n
```

The `38` is the hex byte count of the chunk content. The double `\r\n` at the end terminates the SSE event.

In ASCII bytes:
```
65 76 65 6E 74 3A 20 61 67 65 6E 74 0D 0A   -> "event: agent\r\n"
64 61 74 61 3A 20 7B 22 61 67 65 6E 74 22   -> "data: {"agent""
3A 20 22 75 6E 64 65 72 73 74 61 6E 64 69   -> ": "understandi"
6E 67 22 2C 20 22 73 74 61 74 75 73 22 3A   -> "ng", "status":"
20 22 64 6F 6E 65 22 7D 0D 0A 0D 0A         -> " "done"}\r\n\r\n"
```

The browser's fetch API reads the response body as a `ReadableStream[Uint8Array]`. The frontend creates a `TextDecoder` to convert to UTF-8 string, splits on `\n\n`, and parses each segment for the `data:` prefix.

---

### "What happens when `clarification_halt` fires? Walk through state step by step"

**Strong answer:**

**Step 1: `understanding_agent` runs**
State before: `{input_content: "photo_bytes", input_type: "image", run_id: "abc-123"}`
State after merge: `{..., extracted_entities: [{entity: "call John", type: "task"}], extracted_items: [{description: "call John", raw_deadline: "soon"}], capture: {id: "cap-456"}}`
SSE event: `{"agent": "understanding", "status": "done"}`

**Step 2: `intent_agent` runs**
Receives above state. Validates `raw_deadline: "soon"` — cannot parse to a concrete date. Sets `clarification_needed: True`, `clarification_reason: "Deadline 'soon' could not be parsed to a specific date"`.
State after merge: `{..., clarification_needed: True, clarification_reason: "Deadline 'soon' could not be parsed..."}`
SSE event: `{"agent": "intent", "status": "done"}`

**Step 3: Conditional edge evaluates**
Router: `lambda state: "clarification_halt" if state.get("clarification_needed") else "memory"`
`state.get("clarification_needed")` → `True`
Returns: `"clarification_halt"`
Routing dict lookup: `{"clarification_halt": "clarification_halt"}` → next node = `"clarification_halt"`

**Step 4: `clarification_halt_node` runs**
No-op. Returns `{}` (empty dict). State unchanged.
SSE event: `{"agent": "clarification_halt", "status": "done"}`

**Step 5: Edge to END**
`clarification_halt → END`
Graph terminates. `astream` stops yielding. `generate()` returns.
SSE stream closes.

**Step 6: FastAPI endpoint post-processing**
After `generate()` returns, the endpoint function checks:
```python
final_state = await app.aget_state(config)
if final_state.values.get("clarification_needed"):
    # Log the clarification event
    # (SSE has already returned 422 signal or the stream is already closed)
```
Actually: with SSE, the 422 status cannot be set after headers are sent. The solution is to emit a special SSE event before closing: `event: error\ndata: {"type": "clarification_needed", "reason": "..."}\n\n`. The frontend reads this event and shows the clarification UI.

---

### "What if LangGraph releases a breaking change — how do you migrate?"

**Strong answer:**

LangGraph follows semantic versioning. Breaking changes are in major versions (e.g., 0.x → 1.x). Orbit pins to `langgraph==0.2.*` in requirements.

**Migration strategy:**

1. **Isolate LangGraph usage**: All LangGraph API calls are in `graph.py` (graph construction) and `captures.py` (astream, aget_state). These two files are the migration surface — not the 7 agent functions.

2. **Read migration guides**: LangGraph publishes MIGRATION.md in their GitHub releases. Common breaking changes: API renames (`add_conditional_edges` → different signature), checkpoint format changes (stored state schema changes), streaming event format changes.

3. **Test suite**: The integration test suite for the full pipeline (`test_graph.py`) runs `graph.invoke()` with mock agents. If the tests pass after the upgrade, the migration is complete.

4. **Checkpoint migration**: If the checkpoint format changes between versions, in-flight approvals stored in old format won't load. For MemorySaver (in-memory), this is zero risk — restart flushes checkpoints. For persistent checkpointers, LangGraph provides checkpoint migration utilities.

5. **Canary deploy**: Deploy the upgraded version to a staging environment with real traffic (10% canary). Monitor LangSmith for anomalous traces. Roll back if error rate spikes.

**Time estimate**: 2-4 hours for a minor breaking change in a codebase with Orbit's scope. The graph definition in `graph.py` is ~30 lines. The `captures.py` streaming code is ~50 lines of LangGraph interaction. Migrating those two files covers 100% of the LangGraph surface.

---

### "Can you run the understanding and intent agents in parallel?"

**Strong answer:**

**No, and here's exactly why:**

`intent_agent` requires `extracted_items` in state — that field is set by `understanding_agent`. The dependency is strict:

```
understanding → sets extracted_entities, extracted_items
intent → reads extracted_items to validate deadlines
```

If you ran them in parallel, `intent` would receive state where `extracted_items` is `None` (not yet set), and either crash or produce meaningless output.

**Theoretical parallelism within the pipeline:**

The only potential parallelism would be fan-out within a single node. For example, `planning_agent` generates an action per extracted item — these could be generated in parallel via `asyncio.gather()` inside `planning_agent`. This is parallelism within a node, not between nodes, and doesn't require changes to the graph structure.

For LangGraph-level parallelism with the `Send` API: you could fan out to parallel `process_item` nodes (one per item), each doing understanding + intent + planning for its item simultaneously. But this would require a Map-Reduce pattern: fan-out node → parallel per-item nodes → fan-in aggregator node. This adds graph complexity and is appropriate only if per-item processing is the bottleneck. For Orbit's typical input (1-5 items per capture), sequential processing is faster due to lower overhead.

**Bottom line**: Understanding and intent cannot be parallelized due to data dependency. For future scale, per-item parallelism within planning is the correct optimization point.

---

### "How would you add a feedback loop where planning can retry intent?"

**Strong answer:**

This requires adding a backward edge from planning to intent — a cycle in the graph. LangGraph supports cycles.

**Implementation:**

1. Add `planning_feedback: Optional[str]` and `planning_retry_count: int` to `AgentState`

2. In `planning_agent`: if the extracted items are insufficient for action generation, set `planning_feedback = "items_insufficient"` and return without setting `actions`

3. Add conditional edge after planning:
```python
def route_after_planning(state: AgentState) -> str:
    if (state.get("planning_feedback") == "items_insufficient" 
        and state.get("planning_retry_count", 0) < 2):
        return "intent"  # retry from intent
    return "tool_router"

builder.add_conditional_edges(
    "planning",
    route_after_planning,
    {"intent": "intent", "tool_router": "tool_router"}
)
```

4. In `intent_agent`: check `state.get("planning_feedback")` and adjust validation strictness if it's a retry

5. Add a guard in `planning_agent` to increment `planning_retry_count`:
```python
planning_retry_count = state.get("planning_retry_count", 0) + 1
```

**Safety**: The `planning_retry_count < 2` guard prevents infinite loops. After 2 retries, planning must proceed to tool_router regardless.

**LangGraph note**: Cycles work in LangGraph — the graph is not required to be a DAG. The execution will loop back to intent, re-run intent with the feedback in state, re-run memory (or skip it if we add a conditional edge), and re-run planning. Each cycle writes a new checkpoint, so the full execution history is traceable in LangSmith.

---

*End of Viva Preparation Guide*

---

> **Revision notes**: Review sections 3 (LangGraph concepts) and 6 (question bank) the evening before. Sections 5 (design decisions) and 9 (challenge scenarios) are the most likely to come up from examiners who have read the codebase. Have graph.py open on a second monitor during the viva — being able to point to specific lines is stronger than speaking from memory.
