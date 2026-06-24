Here's a comprehensive implementation prompt you can hand directly to an AI coder:

---

## Orbit AI — Full Implementation Prompt

**You are being asked to build Orbit AI**, a multi-agent personal chief-of-staff system. A full PRD is attached. Below are precise implementation instructions.

---

### 🧱 Project Scaffold

Create a monorepo with this structure:

```
Orbit/
├── backend/          # FastAPI app
│   ├── agents/       # All LangGraph agents
│   ├── tools/        # Gmail, Calendar, Slack integrations
│   ├── memory/       # PostgreSQL + pgvector logic
│   │   ├── db.py     # Schema setup and DB connection
│   │   └── retrieval.py  # Two-level semantic search
│   ├── routers/      # FastAPI route handlers
│   ├── models.py     # Pydantic models + AgentState
│   ├── graph.py      # LangGraph workflow definition
│   └── main.py       # FastAPI entrypoint
├── frontend/         # Next.js app
│   ├── app/
│   │   ├── dashboard/
│   │   ├── workspace/[capture_id]/
│   │   ├── items/
│   │   ├── approvals/
│   │   ├── search/
│   │   └── trace/[run_id]/
│   └── components/
└── docker-compose.yml
```

---

### ⚙️ Backend: FastAPI + LangGraph

#### `models.py` — AgentState

```python
from typing import TypedDict, Optional

class AgentState(TypedDict):
    # Input
    input_content: str
    input_type: str               # "text" | "pdf" | "image"

    # Understanding Agent output
    capture: dict                 # {modality, source, raw_content, file_path, metadata}
    extracted_entities: dict      # {people, dates, locations, deadlines, organizations, urls}

    # Intent Agent output
    extracted_items: list         # list of ExtractedItem dicts (pre-DB write)

    # Memory Agent output (after DB writes)
    capture_id: str
    extracted_item_ids: list      # parallel to extracted_items

    # Planning Agent output
    actions: list                 # flat list of Action dicts across all items

    # Approval Agent output
    decisions: list               # list of Decision dicts

    # Retrieval
    retrieval_context: list

    # Execution
    execution_results: list

    # Guardrails
    clarification_needed: bool
    clarification_reason: str

    # Observability
    trace: list                   # [{agent, output, timestamp}]
```

**ExtractedItem dict shape** (produced by Intent Agent, before DB write):
```python
{
    "title": str,
    "description": str,
    "item_type": str,         # one of 9 valid types (see Intent Agent)
    "confidence_score": float,  # 0.0–1.0
    "urgency_score": float,     # 0.0–1.0
    "entities": dict,
    "deadline": str | None,   # ISO 8601 or null
    "metadata": dict
}
```

**Action dict shape** (produced by Planning Agent, before DB write):
```python
{
    "extracted_item_id": str,
    "action_type": str,        # e.g. "calendar.create_event"
    "payload": dict,
    "requires_approval": bool,
    "status": "pending"
}
```

---

#### `agents/` — Implement all 6 agents as standalone Python functions that accept and return `AgentState`

**Agent 1 — Understanding Agent** (`agents/understanding.py`)
- If `input_type == "pdf"`: use PyMuPDF (`fitz`) to extract text
- If `input_type == "image"`: use Tesseract OCR (`pytesseract`) to extract text
- If `input_type == "text"`: pass through directly
- Call Claude Haiku via the Anthropic SDK to extract entities (people, dates, locations, deadlines, organizations, URLs)
- Build the `capture` dict from the extracted content; detect `source` from the upload mechanism (`"upload"` for file, `"paste"` for text)
- Do NOT write to DB — Memory Agent owns all writes
- Return: `state["capture"]` (dict) + `state["extracted_entities"]` (dict)

**Agent 2 — Intent Agent** (`agents/intent.py`)
- Send `capture["raw_content"]` + `extracted_entities` to Claude Haiku
- Prompt instructs the model to identify **all** distinct actionable or noteworthy items in the content
- For each item: assign `item_type` from the 9-value enum: `event | deadline | task | communication | travel_interest | job_opportunity | meeting | reminder | knowledge`
- For each item: score `confidence_score` (0–1, certainty this is a real item) and `urgency_score` (0–1, time sensitivity)
- Parse `deadline` as ISO 8601; if a date string cannot be parsed, set `state["clarification_needed"] = True` and `state["clarification_reason"]` naming the item and field
- Soft cap: 10 items per capture (enforce via `MAX_EXTRACTED_ITEMS_PER_CAPTURE` env var in prompt)
- Return: `state["extracted_items"]` — a **list** of ExtractedItem dicts (minimum 1 item)

**Agent 3 — Memory Agent** (`agents/memory.py`)
- **Step 1 — Store capture:** Embed `capture["raw_content"]` using `sentence-transformers` (`all-MiniLM-L6-v2`) → write row to `captures` table → store returned `id` in `state["capture_id"]`
- **Step 2 — Store items:** For each item in `extracted_items`: embed `item["title"] + " " + item["description"]` → write row to `extracted_items` table → collect IDs into list → store as `state["extracted_item_ids"]`
- No LLM call in this agent — every capture is stored unconditionally; `confidence_score` on each item is the downstream filter

**Agent 4 — Planning Agent** (`agents/planning.py`)
- Iterate over `zip(state["extracted_items"], state["extracted_item_ids"])`
- Skip items with `confidence_score < 0.5`
- For each qualifying item: call Claude Haiku to generate 1–3 executable actions
- `gmail.send_email` must be generated as a **separate action** from `gmail.draft_email` (guardrail — never combine into one)
- For any `calendar.create_event` action: call `tools/calendar.py::read_availability` first; if the requested slot is busy, call `suggest_alternate_slot` and set `"suggested_alt"` in the action payload
- Write each action to `actions` table with `status = "pending"`
- Return: `state["actions"]` — flat list of Action dicts across all items

**Agent 5 — Tool Router Agent** (`agents/tool_router.py`)
- Iterate over `state["actions"]` and resolve each `action_type` to its handler function reference
- Dispatch table:
  - `calendar.create_event` → `tools/calendar.py::create_event`
  - `calendar.read_availability` → `tools/calendar.py::read_availability`
  - `gmail.draft_email` → `tools/gmail.py::draft_email`
  - `gmail.send_email` → `tools/gmail.py::send_email`
  - `slack.send_reminder` → `tools/slack.py::send_reminder`
  - `slack.send_summary` → `tools/slack.py::send_summary`
- Do NOT execute yet — attach resolved handler references and payloads to each action dict in state

**Agent 6 — Approval Agent** (`agents/approval.py`)
- Graph pauses before this node via `interrupt_before=["approval"]` (set in `graph.py`)
- Expose pending actions via API for the frontend Approval Center; each action is enriched with its parent `extracted_item` context (title, type, confidence, urgency)
- Three resume paths per action:
  - **Approve** → create Decision(`decision="approved"`, `final_action=original_payload`), execute handler, write result to `decisions.final_action`, update `actions.status = "executed"`
  - **Reject** → create Decision(`decision="rejected"`), update `actions.status = "rejected"`
  - **Edit + Approve** → create Decision(`decision="edited"`, `edited_payload=new_payload`, `final_action=new_payload`), execute handler with `edited_payload`, write result, update `actions.status = "executed"`
- Append each Decision to `state["decisions"]`; append execution result to `state["execution_results"]`

---

#### `graph.py` — LangGraph Workflow

```python
from langgraph.graph import StateGraph
from backend.models import AgentState
from backend.agents.understanding import understanding_agent
from backend.agents.intent import intent_agent
from backend.agents.memory import memory_agent
from backend.agents.planning import planning_agent
from backend.agents.tool_router import tool_router_agent
from backend.agents.approval import approval_agent

workflow = StateGraph(AgentState)
workflow.add_node("understanding", understanding_agent)
workflow.add_node("intent", intent_agent)
workflow.add_node("memory", memory_agent)
workflow.add_node("planning", planning_agent)
workflow.add_node("tool_router", tool_router_agent)
workflow.add_node("approval", approval_agent)

workflow.set_entry_point("understanding")
workflow.add_edge("understanding", "intent")
workflow.add_edge("intent", "memory")
workflow.add_edge("memory", "planning")
workflow.add_edge("planning", "tool_router")
workflow.add_edge("tool_router", "approval")

app = workflow.compile(interrupt_before=["approval"])
```

Append to `state["trace"]` at each node: `{ "agent": "...", "output": {...}, "timestamp": "..." }`

---

#### `routers/` — FastAPI Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/captures` | Accept file upload or text paste. Run the full graph. Return `run_id` + `capture_id`. |
| `GET` | `/api/captures` | List all captures (paginated). |
| `GET` | `/api/captures/{id}` | Return capture + all its extracted items + their actions. |
| `GET` | `/api/items` | List all extracted items. Filter by `?item_type=`, `?date_from=`, `?date_to=`, `?min_urgency=`. |
| `GET` | `/api/items/{id}` | Return a single extracted item + its actions. |
| `GET` | `/api/actions/pending` | List all pending actions enriched with parent item context. |
| `POST` | `/api/actions/{id}/approve` | Approve and execute a specific action. |
| `POST` | `/api/actions/{id}/reject` | Reject an action. |
| `POST` | `/api/actions/{id}/edit` | Approve with an edited payload (`{ "edited_payload": {...} }`). |
| `GET` | `/api/search` | Semantic + structured search. Params: `?query=`, `?item_type=`, `?date_from=`, `?min_urgency=`, `?limit=`. |
| `GET` | `/api/trace/{run_id}` | Return the full agent trace for a run. |
| `GET` | `/api/dashboard` | Return recent 10 captures, pending action count, item_type breakdown. |

For file uploads (`/api/captures`), accept `multipart/form-data` with a `file` field and a `source` field. Detect `modality` by MIME type.

---

#### `tools/` — External Integrations

**`tools/calendar.py`**
- Use `google-auth` + `googleapiclient`
- `create_event(title, start_datetime, end_datetime, description)` → returns event link
- `read_availability(date)` → returns list of busy slots
- `suggest_alternate_slot(date, duration_minutes)` → returns next free slot

**`tools/gmail.py`**
- `draft_email(to, subject, body)` → creates Gmail draft, returns draft ID
- `send_email(to, subject, body)` → sends email (requires explicit separate `send` action in approval)

**`tools/slack.py`**
- Use `slack_sdk`
- `send_reminder(channel, message)` → posts reminder
- `send_summary(channel, summary)` → posts summary

Store all OAuth tokens in environment variables. Use `.env` with `python-dotenv`.

---

#### Database Setup (`memory/db.py`)

Use `asyncpg` (or `psycopg2`) to connect to PostgreSQL. Run on startup:

```sql
CREATE EXTENSION IF NOT EXISTS vector;

-- Raw user input
CREATE TABLE IF NOT EXISTS captures (
  id          TEXT PRIMARY KEY,
  modality    TEXT NOT NULL CHECK (modality IN ('image', 'pdf', 'text')),
  source      TEXT NOT NULL CHECK (source IN ('upload', 'paste', 'email', 'screenshot')),
  raw_content TEXT,
  file_path   TEXT,
  embedding   vector(384),
  metadata    JSONB DEFAULT '{}',
  created_at  TIMESTAMP DEFAULT now()
);
CREATE INDEX IF NOT EXISTS captures_embedding_idx
  ON captures USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS captures_created_at_idx
  ON captures (created_at DESC);

-- One capture → many extracted items
CREATE TABLE IF NOT EXISTS extracted_items (
  id               TEXT PRIMARY KEY,
  capture_id       TEXT NOT NULL REFERENCES captures(id) ON DELETE CASCADE,
  title            TEXT NOT NULL,
  description      TEXT,
  item_type        TEXT NOT NULL CHECK (item_type IN (
                     'event','deadline','task','communication',
                     'travel_interest','job_opportunity','meeting','reminder','knowledge'
                   )),
  confidence_score FLOAT NOT NULL DEFAULT 0.0 CHECK (confidence_score BETWEEN 0.0 AND 1.0),
  urgency_score    FLOAT NOT NULL DEFAULT 0.0 CHECK (urgency_score BETWEEN 0.0 AND 1.0),
  entities         JSONB DEFAULT '{}',
  deadline         TIMESTAMP,
  embedding        vector(384),
  metadata         JSONB DEFAULT '{}',
  created_at       TIMESTAMP DEFAULT now()
);
CREATE INDEX IF NOT EXISTS extracted_items_capture_id_idx ON extracted_items (capture_id);
CREATE INDEX IF NOT EXISTS extracted_items_embedding_idx
  ON extracted_items USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS extracted_items_item_type_idx ON extracted_items (item_type);
CREATE INDEX IF NOT EXISTS extracted_items_deadline_idx
  ON extracted_items (deadline) WHERE deadline IS NOT NULL;
CREATE INDEX IF NOT EXISTS extracted_items_urgency_idx
  ON extracted_items (urgency_score DESC);

-- One extracted item → many actions
CREATE TABLE IF NOT EXISTS actions (
  id                TEXT PRIMARY KEY,
  extracted_item_id TEXT NOT NULL REFERENCES extracted_items(id) ON DELETE CASCADE,
  action_type       TEXT NOT NULL,
  payload           JSONB DEFAULT '{}',
  status            TEXT NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending','approved','rejected','executed','failed')),
  requires_approval BOOLEAN NOT NULL DEFAULT TRUE,
  created_at        TIMESTAMP DEFAULT now()
);
CREATE INDEX IF NOT EXISTS actions_extracted_item_id_idx ON actions (extracted_item_id);
CREATE INDEX IF NOT EXISTS actions_status_idx
  ON actions (status) WHERE status = 'pending';

-- Human-in-the-loop approval record per action
CREATE TABLE IF NOT EXISTS decisions (
  id             TEXT PRIMARY KEY,
  action_id      TEXT NOT NULL REFERENCES actions(id) ON DELETE CASCADE,
  decision       TEXT NOT NULL CHECK (decision IN ('approved','rejected','edited')),
  edited_payload JSONB,
  final_action   JSONB,   -- stores execution result after tool runs
  decided_at     TIMESTAMP DEFAULT now()
);
CREATE INDEX IF NOT EXISTS decisions_action_id_idx ON decisions (action_id);
```

---

#### Retrieval Architecture (`memory/retrieval.py`)

Two-level pgvector search serving `GET /api/search` and populating `state["retrieval_context"]`:

**Level 1 — Semantic search on `extracted_items`:**
```sql
SELECT ei.*, 1 - (ei.embedding <=> $1::vector) AS semantic_score
FROM extracted_items ei
WHERE ($2 IS NULL OR ei.item_type = $2)
  AND ($3 IS NULL OR ei.deadline >= $3)
  AND ($4 IS NULL OR ei.urgency_score >= $4)
ORDER BY ei.embedding <=> $1::vector
LIMIT $5;
```

**Level 2 — Semantic search on `captures`:**
```sql
SELECT c.*, 1 - (c.embedding <=> $1::vector) AS semantic_score
FROM captures c
ORDER BY c.embedding <=> $1::vector
LIMIT $2;
```

**Assembly logic (`retrieval.py::assemble_retrieval_context`):**
1. Embed query with `all-MiniLM-L6-v2`
2. Run Level 1 (items) + Level 2 (captures) in parallel
3. Fetch parent captures for all matched items
4. Fetch associated actions for all matched items
5. Build context list: `{item, parent_capture, actions, semantic_score}`
6. Merge in standalone capture results not already covered by item matches
7. Sort by `semantic_score` descending; return top `limit * 2` results

---

### 🖥️ Frontend: Next.js + Tailwind + shadcn/ui

Build 6 pages:

**1. `/dashboard`**
- `CaptureInput` component: dual-mode — file drag-and-drop (PDF/image) or text paste textarea; `source` selector; "Process" button POSTs to `/api/captures`
- `ProcessingStatus`: polls `run_id` status; on completion shows "N items extracted, M actions pending" with link to Capture Workspace
- `RecentCaptures` panel: last 10 captures (modality icon, item count, action count, timestamp)
- `PendingActionsCount` badge linking to Approval Center

**2. `/workspace/[capture_id]`** (shown after processing completes)
- `CaptureHeader`: modality, source, timestamp, raw_content preview
- `ExtractedItemsGrid`: grid of `ItemCard` components, one per extracted item
- `ItemCard`: `item_type` badge (color-coded), title, description, `confidence_score` progress bar, `urgency_score` progress bar, `deadline` chip, linked action count, "View Actions" expand button
- Pull from `GET /api/captures/{id}`

**3. `/items`**
- `FilterBar`: item_type multi-select, date range picker, urgency slider, sort controls
- `ItemsGrid`: paginated list of all extracted items across all captures
- `ItemDetailPanel` (slide-over): full item + its actions inline
- Pull from `GET /api/items` with query params

**4. `/approvals`**
- `PendingActionsQueue`: all pending actions sorted by parent item `urgency_score` desc
- `ActionCard` per action: shows `action_type`, `payload` (key-value), parent item context (title, type, confidence, urgency)
- Three buttons per card: Approve ✅ / Reject ❌ / Edit + Approve ✏️
- "Edit + Approve" opens inline payload editor (JSON or form fields) before calling `POST /api/actions/{id}/edit`
- Execution result shown inline after decision
- `DecisionHistory`: collapsible section showing resolved actions
- Pull from `GET /api/actions/pending`

**5. `/search`**
- `SearchBar`: free-text query input calling `GET /api/search?query=...`
- `FilterSidebar`: item_type filter, date range, urgency threshold
- `SearchResults`: ranked result cards showing type (capture vs item), semantic score badge, title, snippet, deadline, link to parent capture or item detail
- Supports semantic + structured search

**6. `/trace/[run_id]`**
- `AgentTimeline`: vertical timeline with 6 agent nodes; each shows name, duration, expandable JSON output
- Under intent node: list of all extracted items from that run
- Under planning node: list of all actions generated
- Under approval node: final decisions per action
- Pull from `GET /api/trace/{run_id}`

---

### 🔐 Guardrails (enforce in code)

1. No tool in `tools/` may execute without an explicit `Decision` row with `decision IN ('approved', 'edited')` for that action
2. `gmail.send_email` must always be generated as a separate action from `gmail.draft_email` — never merged into one
3. Validate all dates in Intent Agent — if a `deadline` string cannot be parsed to ISO 8601, set `clarification_needed = True` and `clarification_reason` naming the item and field; do not proceed to Memory Agent
4. Validate all email recipients in Planning Agent — if `to` field is missing or malformed, set `clarification_needed = True`
5. Never auto-create calendar events, send Slack messages, or send emails without a Decision record

---

### 📦 Dependencies

**Backend `requirements.txt`:**
```
fastapi
uvicorn[standard]
python-dotenv
python-multipart
langgraph
langchain
langchain-anthropic
anthropic
pymupdf
pytesseract
pillow
sentence-transformers
asyncpg
psycopg2-binary
google-auth
google-auth-oauthlib
google-api-python-client
slack_sdk
```

**Frontend `package.json` (key deps):**
```
next react react-dom tailwindcss @shadcn/ui lucide-react
@tanstack/react-query
zod
```

---

### 🌍 Environment Variables (`.env`)

```
ANTHROPIC_API_KEY=
GOOGLE_CLIENT_ID=
GOOGLE_CLIENT_SECRET=
GOOGLE_REDIRECT_URI=
SLACK_BOT_TOKEN=
SLACK_DEFAULT_CHANNEL=
DATABASE_URL=postgresql://user:password@localhost:5432/orbit
UPLOAD_DIR=/app/uploads
MAX_EXTRACTED_ITEMS_PER_CAPTURE=10
```

---

### ✅ Evaluation Scenarios to Verify

Each scenario traces the full pipeline: **Capture → Extracted Items → Actions → Decision → Execution**

**1. Conference Poster Image**
- Input: upload `poster.png`
- Expected extracted items (3): `event` (NeurIPS 2026, Dec 8), `deadline` (paper submission, Jul 15), `reminder` (registration opens Sep 1)
- Expected actions (4): `calendar.create_event`, `slack.send_reminder` (deadline), `gmail.draft_email` (co-author email), `slack.send_reminder` (registration)
- Verify: all 4 actions appear in Approval Center with parent item context; approving calendar action creates event link in Decision.final_action

**2. Job Description PDF**
- Input: upload `jd.pdf`
- Expected extracted items (3): `job_opportunity` (role + company), `deadline` (application deadline), `task` (prepare materials)
- Expected actions (4): `calendar.create_event` (deadline reminder), `slack.send_reminder`, `gmail.draft_email` (cover letter), `calendar.create_event` (prep session)
- Verify: user can Edit + Approve the email draft with modified body; Decision row shows `decision="edited"` + `edited_payload`

**3. Meeting Notes Text Paste**
- Input: paste meeting notes text
- Expected extracted items (3): `deadline` (API spec by Friday), `meeting` (kickoff call next Tuesday), `communication` (send design doc)
- Expected actions (4): `calendar.create_event`, `gmail.draft_email` (to Bob), `gmail.draft_email` (design doc), `slack.send_summary`
- Verify: `gmail.send_email` is NOT auto-generated — only drafts are created (guardrail enforced)

**4. Travel Destination Screenshot**
- Input: screenshot of Kyoto travel content
- Expected extracted items (2): `travel_interest` (Kyoto cherry blossom), `reminder` (book trip by December)
- Expected actions (2): `slack.send_reminder` (save to travel channel), `calendar.create_event` (book reminder)
- Verify: user can reject Slack action; rejection creates Decision row with `decision="rejected"` and skips execution

**5. Zoom Meeting Screenshot**
- Input: screenshot of Zoom meeting invite
- Expected extracted items (2): `meeting` (Q3 Planning Sync, July 2 2pm), `task` (prepare agenda)
- Expected actions (3): `calendar.create_event`, `gmail.draft_email` (agenda), `slack.send_reminder`
- Verify: Planning Agent detects calendar conflict at 2pm; `calendar.create_event` payload includes `"suggested_alt"` field; user uses Edit + Approve to change time before executing

---

**Build order: DB schema → models.py → agents (understanding → intent → memory → planning → tool_router → approval) → graph.py → routers/ → memory/retrieval.py → frontend (Dashboard → Workspace → Approvals → Items → Search → Trace)**
