# Orbit AI — Architecture Guide

Orbit is a multi-agent personal chief-of-staff. You drop in a document (email, PDF, screenshot, meeting notes) and it finds every actionable thing inside it, proposes actions across Google Calendar, Gmail, and Slack, then waits for your approval before doing anything.

---

## The Core Idea: One Capture, Many Items, Many Actions

Most AI assistants treat each upload as one thing. Orbit treats it as a container.

A single conference poster might contain:
- an **event** (the conference itself)
- a **deadline** (paper submission due date)
- a **reminder** (registration closes Friday)

Each of those becomes an `ExtractedItem`. Each item gets its own set of proposed `Actions`. Each action waits for a human decision before executing.

This is the capture-centric model:

```
Upload / Paste
     │
     ▼
  Capture  ──────────────────────────────────────┐
     │                                            │
     ├── ExtractedItem (event)                    │
     │        ├── Action: calendar.create_event   │  All stored in
     │        └── Action: slack.send_reminder     │  PostgreSQL
     │                                            │
     ├── ExtractedItem (deadline)                 │
     │        └── Action: gmail.draft_email       │
     │                                            │
     └── ExtractedItem (reminder)                 │
              └── Action: slack.send_reminder     │
                                                  │
     Each action → Decision (approve/reject/edit) ┘
```

---

## Database: 4 Tables

```
captures
  id, run_id, modality, source, raw_content, file_path
  embedding vector(384), metadata JSONB
  created_at, deleted_at   ← soft delete, audit trail preserved

    │ (one-to-many)
    ▼

extracted_items
  id, capture_id
  title, description, item_type, confidence_score, urgency_score
  entities JSONB, deadline TIMESTAMP
  embedding vector(384), metadata JSONB
  planning_status   ← pending | planned | skipped_low_confidence | skipped_no_actions

    │ (one-to-many)
    ▼

actions
  id, extracted_item_id
  action_type, payload JSONB
  status   ← pending | approved | rejected | executed | failed
  requires_approval BOOLEAN
  depends_on_action_id   ← FK to another action (e.g. send_email depends on draft_email)

    │ (one-to-one)
    ▼

decisions
  id, action_id
  decision   ← approved | rejected | edited
  edited_payload JSONB   ← what the user changed it to
  final_payload JSONB    ← what actually got sent to the tool
  execution_result JSONB ← what the tool returned
  decided_at
```

Embeddings use `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions).
Indexes use HNSW (`pgvector >= 0.5`) — no training phase required, works on any table size.

---

## The Pipeline: 6 Agents in a LangGraph

Every upload runs through a linear six-node state graph. The state is a Python `TypedDict` called `AgentState` — one dict that flows through all nodes, each node adding fields.

```
  [User uploads file or text]
            │
            ▼
   ┌─────────────────┐
   │  understanding  │  Extract raw text. OCR images + PDFs. Run NER.
   └────────┬────────┘
            │
            ▼
   ┌─────────────────┐
   │     intent      │  Find all actionable items. Validate deadlines.
   └────────┬────────┘
            │
     clarification_needed?
            │
     YES ───┴─── NO
      │              │
      ▼              ▼
  ┌──────────┐  ┌──────────┐
  │ halt     │  │  memory  │  Embed + write capture + items to DB.
  │ → 422    │  └────┬─────┘
  └──────────┘       │
                     ▼
              ┌─────────────┐
              │   planning  │  Generate 1-3 actions per item.
              └──────┬──────┘
                     │
                     ▼
              ┌─────────────┐
              │ tool_router │  Validate action types. Build pending_action_ids list.
              └──────┬──────┘
                     │
              ── GRAPH PAUSES ──    ← interrupt_before=["approval"]
                     │
                     ▼
              ┌─────────────┐
              │  approval   │  No-op checkpoint. Real decisions come via REST.
              └─────────────┘
```

The graph pauses before the `approval` node. From there, the frontend drives each action individually via `POST /api/actions/{id}/approve|reject|edit`.

---

## What Each Agent Does

### 1. Understanding Agent
**File:** `agents/understanding.py`

Converts whatever the user submitted into plain text and extracts named entities.

- **PDF** → `PyMuPDF` extracts text from each page + `pytesseract` OCRs any embedded images
- **Image** → `pytesseract` OCR
- **Text** → used as-is

Then calls Claude Haiku with `tool_use` mode (structured output) to extract:
`people`, `dates`, `locations`, `deadlines`, `organizations`, `urls`

Wraps user content in `<user_document>` XML tags to prevent prompt injection.

**Output added to state:** `capture` dict, `extracted_entities` dict

---

### 2. Intent Agent
**File:** `agents/intent.py`

Finds *all* distinct actionable things in the document — not just one.

Calls Claude Haiku with a strict JSON schema (via `tool_use`) to extract a list of items. Each item has:
`title`, `description`, `item_type`, `confidence_score`, `urgency_score`, `entities`, `deadline`

**Key behaviours:**
- Items are sorted by `urgency_score` before applying the `MAX_EXTRACTED_ITEMS_PER_CAPTURE` cap (default 10), so the highest-priority items are never dropped
- Truncated items count is stored in state metadata so the frontend can show a banner
- Each item's deadline is validated individually. A bad deadline on one item sets `deadline = null` and flags `clarification_needed` — it does not block other items
- If `clarification_needed` is true after this node, the graph routes to `clarification_halt` and returns HTTP 422 to the caller with the reason

**Output added to state:** `extracted_items` list, `clarification_needed`, `clarification_reason`

---

### 3. Memory Agent
**File:** `agents/memory.py`

Persists everything to PostgreSQL.

1. Embeds `capture.raw_content` → writes `captures` row (with `run_id` for LangGraph thread linkage)
2. For each extracted item: embeds `title + description` → writes `extracted_items` row

Uses `sentence-transformers/all-MiniLM-L6-v2`. Raises immediately on any DB write failure (does not silently continue). Asserts that the number of IDs written equals the number of items at the end.

**Output added to state:** `capture_id`, `extracted_item_ids`

---

### 4. Planning Agent
**File:** `agents/planning.py`

For each extracted item, calls Claude Haiku to generate 1–3 executable actions.

**Key behaviours:**
- Skips items with `confidence_score < 0.5` (marks them `skipped_low_confidence` in DB)
- Uses `tool_use` mode with a schema that includes `action_type` and `payload`
- Only allows known action types (`VALID_ACTION_TYPES` from `models.py`) — others are filtered out
- `gmail.send_email` is always a *separate* action from `gmail.draft_email`. The send action stores `depends_on_action_id` pointing to the draft action so the `draft_id` can be resolved at execution time
- Does **not** call calendar APIs at planning time — conflict checking happens at approval time

**Output added to state:** `actions` list, `pending_action_ids`

---

### 5. Tool Router Agent
**File:** `agents/tool_router.py`

Validates the `action_type` on each action against the known-valid set. Marks unknown types as `failed`. Populates `pending_action_ids` for actions that need human approval.

This node stores only strings in state — no callable references. Handler functions are resolved lazily at execution time in `routers/actions.py`.

**Output added to state:** validated `actions` list, `pending_action_ids`, `decided_action_ids`

---

### 6. Approval Agent
**File:** `agents/approval.py`

The graph pauses *before* this node. In practice, this node is a no-op — it logs to the trace but does no real work.

Real execution happens through the REST API (`routers/actions.py`). The agent node is kept in the graph so the LangGraph checkpointer can resume the run if needed.

---

## REST API: How Humans Drive Execution

**File:** `routers/actions.py`

The three human-action endpoints are where tools actually execute:

```
POST /api/actions/{id}/approve
  1. SELECT ... FOR UPDATE  ← prevents double-execution under concurrent requests
  2. Check for existing Decision  ← returns 409 if already decided
  3. If calendar.create_event: check for conflicts against live calendar
     → if conflict found: return {conflict_detected: true, suggested_alt: {...}}
       without executing (frontend shows the conflict and lets user edit)
  4. If gmail.send_email: resolve draft_id from the draft_email action's execution_result
  5. Execute the tool handler
  6. Write Decision row: final_payload + execution_result (stored separately)

POST /api/actions/{id}/reject
  1. Same lock + idempotency check
  2. Write Decision row (no execution)

POST /api/actions/{id}/edit
  1. Same lock + idempotency check
  2. Validate edited_payload against the action's Pydantic schema (e.g. CalendarCreateSchema)
  3. Execute with the edited payload
  4. Write Decision row
```

---

## The Tools

**`tools/calendar.py`** — Google Calendar
- `create_event(title, start_datetime, end_datetime, description)` → event link
- `read_availability(date)` → list of busy slots
- `suggest_alternate_slot(date, duration_minutes)` → `{start, end}` dict

**`tools/gmail.py`** — Gmail
- `draft_email(to, subject, body)` → `{draft_id, to, subject, status: "drafted"}`
- `send_email(draft_id)` → sends the existing draft (no re-composition)

The draft → send split is intentional: you review the draft, then approve the send as a separate action.

**`tools/slack.py`** — Slack
- `send_reminder(channel, message)`
- `send_summary(channel, summary)`

**`tools/auth.py`** — Shared Google credential refresh
- `get_google_credentials(scopes)` — auto-refreshes expired tokens via `google.auth.transport.requests.Request`

---

## How gmail.draft → gmail.send Works End-to-End

This is worth tracing in detail because it spans multiple agents and two approval steps.

```
Planning Agent
  ├── Action A: gmail.draft_email  {to, subject, body}     id=AAA
  └── Action B: gmail.send_email   {draft_id: "TBD"}       id=BBB
                                   depends_on_action_id=AAA

User approves Action A (draft_email):
  → tool executes: gmail.draft_email(to, subject, body)
  → execution_result = {draft_id: "r:abc123def", status: "drafted"}
  → Decision row written for AAA with execution_result stored

User approves Action B (send_email):
  → _resolve_send_email_payload():
      SELECT execution_result FROM decisions
      WHERE action_id = 'AAA'
      → finds {draft_id: "r:abc123def"}
  → tool executes: gmail.send_email(draft_id="r:abc123def")
  → email sent ✓
```

---

## How Calendar Conflict Detection Works

Conflict detection was deliberately moved from Planning to Approval time.

**Why:** At planning time, the schedule could be hours or days old. At approval time, you're deciding right now.

```
User clicks Approve on a calendar.create_event action
     │
     ▼
_check_calendar_conflict_at_approval()
     │
     ├── calls read_availability(date) ← live Google Calendar query
     │
     ├── overlap found?
     │      YES → suggests alternate slot via suggest_alternate_slot()
     │             → returns {conflict_detected: true, suggested_alt: {start, end}}
     │             → frontend shows the conflict banner with "Use Suggested Slot" button
     │             → nothing is executed yet
     │
     └── no overlap → execute create_event immediately
```

---

## Semantic Search

**File:** `memory/retrieval.py`

Two parallel pgvector queries on every search:

1. **Item search** — cosine similarity on `extracted_items.embedding`
   - Supports filters: `item_type`, `date_from`, `min_urgency`
   - Fetches parent capture and associated actions for each result

2. **Capture search** — cosine similarity on `captures.embedding`
   - Returns captures that didn't surface via item matches

Results are merged and sorted by semantic score, deduplicated by capture ID.

---

## Security & Reliability

**API auth (H9):** Every route is protected by `X-API-Key` header. Set `ORBIT_API_KEY` in `.env`. Leave it empty to disable auth in dev.

**Upload size (M8):** Two layers — HTTP middleware in `main.py` rejects before the request body is read; router also checks after reading the file. Default: 20 MB.

**Prompt injection (H10):** All user content is wrapped in `<user_document>` XML tags and passed via Anthropic's `tool_use` mode with `tool_choice={"type":"tool"}`. This forces structured JSON output — the model cannot emit arbitrary text in response to document content.

**Race conditions (M7):** Approve, reject, and edit all use `SELECT ... FOR UPDATE` inside a transaction. Concurrent clicks on the same action get a 409, not a double-execution.

**Idempotency (H4):** A partial unique index on `decisions(action_id)` prevents duplicate decisions at the database level, even if the application lock fails.

**Soft deletes (H7):** Deleting a capture sets `deleted_at`; it never hard-deletes. Decision rows keep their `action_id` reference (`ON DELETE SET NULL`) so the audit trail is always complete.

---

## Frontend: 6 Pages

```
/dashboard          Landing page. Upload/paste form. Recent captures list.
                    Pending actions badge. Item type breakdown chart.

/workspace/[id]     Per-capture view. Grid of ExtractedItem cards, each showing:
                    - Item type badge
                    - Planning status badge (shown when != "planned")
                    - Confidence + urgency progress bars
                    - Linked actions (expandable)
                    - Truncation banner (if intent capped the item list)

/items              Browse all items across all captures.
                    Filter by type, date range, min urgency.

/approvals          Pending actions queue, sorted by urgency.
                    Each card shows the action payload + parent item context.
                    Approve / Reject / Edit buttons.
                    If calendar conflict detected: shows conflict banner with
                    suggested alternate slot and "Use Suggested Slot" button.

/search             Semantic + structured search. Results show item + parent capture.

/trace/[run_id]     Per-run agent timeline. One row per agent, with summary fields.
```

---

## File Map

```
backend/
  main.py              FastAPI app, auth middleware, upload size middleware
  graph.py             LangGraph StateGraph definition
  models.py            AgentState TypedDict + Pydantic schemas + ACTION_SCHEMAS
  agents/
    understanding.py   OCR + NER extraction
    intent.py          Item list extraction
    clarification.py   Halt node for invalid inputs
    memory.py          DB writes + embeddings
    planning.py        Action generation
    tool_router.py     Action type validation
    approval.py        Checkpoint node (no-op)
  memory/
    db.py              asyncpg pool, all SQL helpers, schema creation
    retrieval.py       Two-level pgvector semantic search
  routers/
    captures.py        POST /captures, GET /captures/{id}/status, DELETE
    actions.py         approve / reject / edit endpoints (tool execution lives here)
    items.py           GET /items, GET /items/{id}
    search.py          GET /search
    trace.py           GET /trace/{run_id}
    dashboard.py       GET /dashboard
  tools/
    auth.py            Google credential refresh (shared)
    calendar.py        Google Calendar API
    gmail.py           Gmail API
    slack.py           Slack Web API

frontend/
  app/
    dashboard/page.tsx
    workspace/[capture_id]/page.tsx
    items/page.tsx
    approvals/page.tsx
    search/page.tsx
    trace/[run_id]/page.tsx
  components/
    ItemTypeBadge.tsx
    ui/  (card, badge, button, progress)
  lib/
    api.ts     All fetch calls to backend
    types.ts   TypeScript interfaces
    utils.ts   formatDate, truncate
```

---

## Running Locally

```bash
# 1. Start PostgreSQL with pgvector
docker-compose up -d db

# 2. Backend
cd backend
cp .env.example .env   # fill in API keys
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# 3. Frontend
cd frontend
npm install
npm run dev            # http://localhost:3000
```

Required env vars in `backend/.env`:
- `DATABASE_URL` — PostgreSQL connection string
- `ANTHROPIC_API_KEY` — for Claude Haiku (understanding, intent, planning agents)
- `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, `GOOGLE_REFRESH_TOKEN` — Calendar + Gmail
- `SLACK_BOT_TOKEN`, `SLACK_CHANNEL_ID` — Slack
- `ORBIT_API_KEY` — leave empty to skip auth in dev
