import asyncpg
import json
import os
from typing import Optional, List, Any

_pool: Optional[asyncpg.Pool] = None

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://orbit:orbit_secret@localhost:5432/orbit")


def _j(value) -> dict:
    """Safely coerce a JSONB value to a dict — handles dict, JSON string, or None."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            result = json.loads(value)
            return result if isinstance(result, dict) else {}
        except (ValueError, TypeError):
            return {}
    return {}


# Export so routers can import it
safe_json = _j


async def _init_connection(conn):
    """Register JSON/JSONB codecs so asyncpg returns dicts, not raw strings."""
    await conn.set_type_codec("jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog")
    await conn.set_type_codec("json",  encoder=json.dumps, decoder=json.loads, schema="pg_catalog")


async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=10, init=_init_connection)
    return _pool


async def close_pool():
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


async def create_tables():
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
CREATE EXTENSION IF NOT EXISTS vector;

-- ── Captures ──────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS captures (
  id          TEXT PRIMARY KEY,
  run_id      TEXT,
  modality    TEXT NOT NULL CHECK (modality IN ('image', 'pdf', 'text')),
  source      TEXT NOT NULL CHECK (source IN ('upload', 'paste', 'email', 'screenshot')),
  raw_content TEXT,
  file_path   TEXT,
  embedding   vector(384),
  metadata    JSONB DEFAULT '{}',
  created_at  TIMESTAMP DEFAULT now(),
  deleted_at  TIMESTAMP DEFAULT NULL
);

-- M5: HNSW index — no training phase, works on any table size (pgvector >= 0.5)
CREATE INDEX IF NOT EXISTS captures_embedding_idx
  ON captures USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS captures_created_at_idx
  ON captures (created_at DESC);

-- ── Extracted Items ────────────────────────────────────────────────────────
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
  created_at       TIMESTAMP DEFAULT now(),
  -- M2: track why planning skipped an item
  planning_status  TEXT DEFAULT 'pending'
                   CHECK (planning_status IN ('pending','planned','skipped_low_confidence','skipped_no_actions'))
);

CREATE INDEX IF NOT EXISTS extracted_items_capture_id_idx ON extracted_items (capture_id);
-- M5: HNSW
CREATE INDEX IF NOT EXISTS extracted_items_embedding_idx
  ON extracted_items USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS extracted_items_item_type_idx ON extracted_items (item_type);
CREATE INDEX IF NOT EXISTS extracted_items_deadline_idx
  ON extracted_items (deadline) WHERE deadline IS NOT NULL;
CREATE INDEX IF NOT EXISTS extracted_items_urgency_idx
  ON extracted_items (urgency_score DESC);

-- ── Actions ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS actions (
  id                   TEXT PRIMARY KEY,
  extracted_item_id    TEXT NOT NULL REFERENCES extracted_items(id) ON DELETE CASCADE,
  action_type          TEXT NOT NULL,
  payload              JSONB DEFAULT '{}',
  status               TEXT NOT NULL DEFAULT 'pending'
                       CHECK (status IN ('pending','approved','rejected','executed','failed')),
  requires_approval    BOOLEAN NOT NULL DEFAULT TRUE,
  -- H6: link send_email → draft_email to resolve draft_id at execution time
  depends_on_action_id TEXT REFERENCES actions(id) ON DELETE SET NULL,
  created_at           TIMESTAMP DEFAULT now()
);

CREATE INDEX IF NOT EXISTS actions_extracted_item_id_idx ON actions (extracted_item_id);
CREATE INDEX IF NOT EXISTS actions_status_idx
  ON actions (status) WHERE status = 'pending';

-- ── Decisions ──────────────────────────────────────────────────────────────
-- H7: ON DELETE SET NULL preserves audit trail when action is cascade-deleted
-- H4: UNIQUE prevents double-execution
-- H5: split final_action into final_payload (what was approved) + execution_result (what the tool returned)
CREATE TABLE IF NOT EXISTS decisions (
  id               TEXT PRIMARY KEY,
  action_id        TEXT REFERENCES actions(id) ON DELETE SET NULL,
  decision         TEXT NOT NULL CHECK (decision IN ('approved','rejected','edited')),
  edited_payload   JSONB,
  final_payload    JSONB,
  execution_result JSONB,
  decided_at       TIMESTAMP DEFAULT now()
);

-- Partial unique index: only one decision per non-null action_id (nulls after cascade delete are fine)
CREATE UNIQUE INDEX IF NOT EXISTS decisions_action_id_unique_idx
  ON decisions (action_id) WHERE action_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS decisions_action_id_idx ON decisions (action_id);

-- ── Runs ───────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS runs (
  id          TEXT PRIMARY KEY,
  capture_id  TEXT REFERENCES captures(id) ON DELETE SET NULL,
  status      TEXT NOT NULL DEFAULT 'running'
              CHECK (status IN ('running','completed','interrupted','failed')),
  trace       JSONB DEFAULT '[]',
  created_at  TIMESTAMP DEFAULT now(),
  updated_at  TIMESTAMP DEFAULT now()
);
""")

        # Migrate existing tables: add new columns IF NOT EXISTS
        # (safe to run on both fresh and existing databases)
        migrations = [
            "ALTER TABLE captures ADD COLUMN IF NOT EXISTS run_id TEXT",
            "ALTER TABLE captures ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMP DEFAULT NULL",
            "ALTER TABLE extracted_items ADD COLUMN IF NOT EXISTS planning_status TEXT DEFAULT 'pending' CHECK (planning_status IN ('pending','planned','skipped_low_confidence','skipped_no_actions'))",
            "ALTER TABLE actions ADD COLUMN IF NOT EXISTS depends_on_action_id TEXT REFERENCES actions(id) ON DELETE SET NULL",
            "ALTER TABLE decisions ADD COLUMN IF NOT EXISTS final_payload JSONB",
            "ALTER TABLE decisions ADD COLUMN IF NOT EXISTS execution_result JSONB",
        ]
        for migration in migrations:
            try:
                await conn.execute(migration)
            except Exception:
                pass  # column already exists or constraint conflict — safe to ignore


# ── Captures ───────────────────────────────────────────────────────────────

async def insert_capture(
    id: str,
    modality: str,
    source: str,
    raw_content: Optional[str],
    file_path: Optional[str],
    embedding: Optional[list],
    metadata: dict,
    run_id: Optional[str] = None,  # C4
) -> dict:
    pool = await get_pool()
    emb_str = f"[{','.join(str(v) for v in embedding)}]" if embedding else None
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO captures (id, run_id, modality, source, raw_content, file_path, embedding, metadata)
            VALUES ($1, $2, $3, $4, $5, $6, $7::vector, $8::jsonb)
            RETURNING id, run_id, modality, source, raw_content, file_path, metadata, created_at
            """,
            id, run_id, modality, source, raw_content, file_path, emb_str, json.dumps(metadata),
        )
        return dict(row)


async def get_capture(capture_id: str) -> Optional[dict]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """SELECT id, run_id, modality, source, raw_content, file_path, metadata, created_at
               FROM captures WHERE id = $1 AND deleted_at IS NULL""",
            capture_id,
        )
        return dict(row) if row else None


async def list_captures(limit: int = 20, offset: int = 0) -> List[dict]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT id, run_id, modality, source, raw_content, file_path, metadata, created_at
               FROM captures WHERE deleted_at IS NULL
               ORDER BY created_at DESC LIMIT $1 OFFSET $2""",
            limit, offset,
        )
        return [dict(r) for r in rows]


async def soft_delete_capture(capture_id: str):
    """H7: soft delete — preserves decisions audit trail."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE captures SET deleted_at = now() WHERE id = $1",
            capture_id,
        )


async def count_extracted_items(capture_id: str) -> int:
    """H8: used by status endpoint."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.fetchval(
            "SELECT COUNT(*) FROM extracted_items WHERE capture_id = $1",
            capture_id,
        )


async def count_pending_actions_for_capture(capture_id: str) -> int:
    """H8: used by status endpoint."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.fetchval(
            """SELECT COUNT(*) FROM actions a
               JOIN extracted_items ei ON a.extracted_item_id = ei.id
               WHERE ei.capture_id = $1 AND a.status = 'pending'""",
            capture_id,
        )


# ── Extracted Items ────────────────────────────────────────────────────────

def _parse_deadline(deadline) -> Optional[Any]:
    """Convert ISO string deadline to datetime; pass through datetime objects; return None on failure."""
    if deadline is None:
        return None
    if hasattr(deadline, 'year'):  # already a date/datetime
        return deadline
    try:
        from datetime import datetime
        return datetime.fromisoformat(str(deadline).replace("Z", ""))
    except (ValueError, TypeError):
        return None


async def insert_extracted_item(
    id: str,
    capture_id: str,
    title: str,
    description: Optional[str],
    item_type: str,
    confidence_score: float,
    urgency_score: float,
    entities: dict,
    deadline: Optional[str],
    embedding: Optional[list],
    metadata: dict,
    planning_status: str = "pending",  # M2
) -> dict:
    pool = await get_pool()
    emb_str = f"[{','.join(str(v) for v in embedding)}]" if embedding else None
    deadline_dt = _parse_deadline(deadline)
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO extracted_items
              (id, capture_id, title, description, item_type, confidence_score, urgency_score,
               entities, deadline, embedding, metadata, planning_status)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8::jsonb,$9,$10::vector,$11::jsonb,$12)
            RETURNING id, capture_id, title, description, item_type, confidence_score,
                      urgency_score, entities, deadline, metadata, created_at, planning_status
            """,
            id, capture_id, title, description, item_type, confidence_score, urgency_score,
            json.dumps(entities), deadline_dt, emb_str, json.dumps(metadata), planning_status,
        )
        return dict(row)


async def update_extracted_item_planning_status(item_id: str, planning_status: str):
    """M2: record why an item was skipped by the Planning Agent."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE extracted_items SET planning_status = $1 WHERE id = $2",
            planning_status, item_id,
        )


async def get_extracted_item(item_id: str) -> Optional[dict]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """SELECT id, capture_id, title, description, item_type, confidence_score,
                      urgency_score, entities, deadline, metadata, created_at, planning_status
               FROM extracted_items WHERE id = $1""",
            item_id,
        )
        return dict(row) if row else None


async def get_items_by_capture(capture_id: str) -> List[dict]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT id, capture_id, title, description, item_type, confidence_score,
                      urgency_score, entities, deadline, metadata, created_at, planning_status
               FROM extracted_items WHERE capture_id = $1 ORDER BY urgency_score DESC""",
            capture_id,
        )
        return [dict(r) for r in rows]


async def list_extracted_items(
    item_type: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    min_urgency: Optional[float] = None,
    limit: int = 20,
    offset: int = 0,
) -> List[dict]:
    pool = await get_pool()
    conditions: List[str] = []
    params: List[Any] = []
    idx = 1

    if item_type:
        conditions.append(f"item_type = ${idx}")
        params.append(item_type)
        idx += 1
    if date_from:
        conditions.append(f"deadline >= ${idx}::timestamp")
        params.append(date_from)
        idx += 1
    if date_to:
        conditions.append(f"deadline <= ${idx}::timestamp")
        params.append(date_to)
        idx += 1
    if min_urgency is not None:
        conditions.append(f"urgency_score >= ${idx}")
        params.append(min_urgency)
        idx += 1

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    params.extend([limit, offset])
    query = f"""
        SELECT id, capture_id, title, description, item_type, confidence_score,
               urgency_score, entities, deadline, metadata, created_at, planning_status
        FROM extracted_items {where}
        ORDER BY urgency_score DESC, created_at DESC
        LIMIT ${idx} OFFSET ${idx+1}
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(query, *params)
        return [dict(r) for r in rows]


# ── Actions ────────────────────────────────────────────────────────────────

async def insert_action(
    id: str,
    extracted_item_id: str,
    action_type: str,
    payload: dict,
    requires_approval: bool = True,
    depends_on_action_id: Optional[str] = None,  # H6
) -> dict:
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO actions (id, extracted_item_id, action_type, payload, requires_approval, depends_on_action_id)
            VALUES ($1, $2, $3, $4::jsonb, $5, $6)
            RETURNING id, extracted_item_id, action_type, payload, status, requires_approval,
                      depends_on_action_id, created_at
            """,
            id, extracted_item_id, action_type, json.dumps(payload), requires_approval, depends_on_action_id,
        )
        return dict(row)


async def get_action(action_id: str) -> Optional[dict]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """SELECT id, extracted_item_id, action_type, payload, status, requires_approval,
                      depends_on_action_id, created_at
               FROM actions WHERE id = $1""",
            action_id,
        )
        return dict(row) if row else None


async def update_action_status(action_id: str, status: str):
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("UPDATE actions SET status = $1 WHERE id = $2", status, action_id)


async def get_actions_by_item(item_id: str) -> List[dict]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT id, extracted_item_id, action_type, payload, status, requires_approval,
                      depends_on_action_id, created_at
               FROM actions WHERE extracted_item_id = $1""",
            item_id,
        )
        return [dict(r) for r in rows]


async def get_pending_actions() -> List[dict]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT id, extracted_item_id, action_type, payload, status, requires_approval,
                      depends_on_action_id, created_at
               FROM actions WHERE status = 'pending' ORDER BY created_at ASC"""
        )
        return [dict(r) for r in rows]


# ── Decisions ──────────────────────────────────────────────────────────────

async def get_decision_by_action(action_id: str) -> Optional[dict]:
    """H4: idempotency check — returns existing decision if one exists."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id, action_id, decision, edited_payload, final_payload, execution_result, decided_at FROM decisions WHERE action_id = $1",
            action_id,
        )
        return dict(row) if row else None


async def insert_decision(
    id: str,
    action_id: str,
    decision: str,
    edited_payload: Optional[dict] = None,
    final_payload: Optional[dict] = None,   # H5: what was approved
    execution_result: Optional[dict] = None, # H5: what the tool returned
) -> dict:
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO decisions (id, action_id, decision, edited_payload, final_payload, execution_result)
            VALUES ($1, $2, $3, $4::jsonb, $5::jsonb, $6::jsonb)
            RETURNING id, action_id, decision, edited_payload, final_payload, execution_result, decided_at
            """,
            id, action_id, decision,
            json.dumps(edited_payload) if edited_payload else None,
            json.dumps(final_payload) if final_payload else None,
            json.dumps(execution_result) if execution_result else None,
        )
        return dict(row)


async def get_execution_result_for_action(action_id: str) -> Optional[dict]:
    """H6: resolve execution_result of a dependency (e.g. draft_id from draft_email)."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT execution_result FROM decisions WHERE action_id = $1",
            action_id,
        )
        if row and row["execution_result"]:
            result = row["execution_result"]
            return dict(result) if isinstance(result, dict) else json.loads(result)
        return None


# ── Runs ───────────────────────────────────────────────────────────────────

async def insert_run(run_id: str, capture_id: Optional[str] = None):
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO runs (id, capture_id) VALUES ($1, $2)",
            run_id, capture_id,
        )


async def update_run(run_id: str, status: str, trace: list, capture_id: Optional[str] = None):
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """UPDATE runs SET status = $1, capture_id = COALESCE($2, capture_id), updated_at = now()
               WHERE id = $3""",
            status, capture_id, run_id,
        )


async def get_run(run_id: str) -> Optional[dict]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id, capture_id, status, trace, created_at, updated_at FROM runs WHERE id = $1",
            run_id,
        )
        return dict(row) if row else None



async def get_pending_action_count() -> int:
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.fetchval("SELECT COUNT(*) FROM actions WHERE status = 'pending'")


async def get_item_type_breakdown() -> dict:
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT item_type, COUNT(*) as cnt FROM extracted_items GROUP BY item_type")
        return {r["item_type"]: r["cnt"] for r in rows}
