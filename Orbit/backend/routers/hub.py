"""
Hub router — Intelligence Hub
Returns three panels (upcoming, tasks, knowledge) + structured AI priorities + stats.
"""
import os
import anthropic
from datetime import datetime, timezone
from fastapi import APIRouter
from memory import db
from memory.db import safe_json

router = APIRouter(prefix="/api/hub", tags=["hub"])
_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

UPCOMING_TYPES = ("event", "meeting", "deadline", "reminder")
TASK_TYPES     = ("task", "job_opportunity", "communication", "travel_interest")

DIGEST_SCHEMA = {
    "name": "generate_digest",
    "description": "Generate a structured digest with priorities and a one-sentence summary.",
    "input_schema": {
        "type": "object",
        "properties": {
            "priorities": {
                "type": "array",
                "maxItems": 5,
                "description": "Up to 5 action items, most critical first",
                "items": {
                    "type": "object",
                    "properties": {
                        "level": {
                            "type": "string",
                            "enum": ["critical", "high", "medium", "low"],
                            "description": "critical=overdue/blocking, high=due today, medium=this week, low=on track",
                        },
                        "text": {
                            "type": "string",
                            "description": "Short actionable line under 80 characters",
                        },
                    },
                    "required": ["level", "text"],
                },
            },
            "summary": {
                "type": "string",
                "description": "One sentence overall status",
            },
        },
        "required": ["priorities", "summary"],
    },
}


def _group(deadline) -> str:
    if deadline is None:
        return "no_date"
    now = datetime.now(timezone.utc)
    dl = deadline.replace(tzinfo=timezone.utc) if deadline.tzinfo is None else deadline
    secs = (dl - now).total_seconds()
    if secs < 0:
        return "overdue"
    if secs < 86_400:
        return "today"
    if secs < 7 * 86_400:
        return "this_week"
    return "later"


def _serialize(row: dict) -> dict:
    d = dict(row)
    d["entities"] = safe_json(d.get("entities"))
    d["metadata"] = safe_json(d.get("metadata"))
    if d.get("deadline") and hasattr(d["deadline"], "isoformat"):
        d["deadline"] = d["deadline"].isoformat()
    return d


async def _query_items(conn, types: tuple, order: str, limit: int = 30) -> list:
    placeholders = ",".join(f"${i+1}" for i in range(len(types)))
    rows = await conn.fetch(
        f"""
        SELECT ei.id, ei.capture_id, ei.title, ei.description, ei.item_type,
               ei.confidence_score, ei.urgency_score, ei.entities, ei.deadline,
               ei.metadata, ei.created_at, ei.planning_status,
               COUNT(a.id) FILTER (WHERE a.status = 'pending')::int AS pending_actions,
               COUNT(a.id)::int                                      AS total_actions
        FROM extracted_items ei
        LEFT JOIN actions a ON a.extracted_item_id = ei.id
        WHERE ei.item_type IN ({placeholders})
        GROUP BY ei.id
        ORDER BY {order}
        LIMIT ${len(types) + 1}
        """,
        *types, limit,
    )
    return [_serialize(dict(r)) for r in rows]


def _compute_stats(upcoming: list, tasks: list, knowledge: list) -> dict:
    return {
        "overdue":         sum(1 for u in upcoming if u.get("group") == "overdue"),
        "today":           sum(1 for u in upcoming if u.get("group") == "today"),
        "this_week":       sum(1 for u in upcoming if u.get("group") == "this_week"),
        "total_tasks":     len(tasks),
        "knowledge_items": len(knowledge),
        "upcoming_total":  len(upcoming),
    }


def _generate_structured_digest(upcoming: list, tasks: list, knowledge: list) -> dict:
    overdue    = [u for u in upcoming if u.get("group") == "overdue"]
    today      = [u for u in upcoming if u.get("group") == "today"]
    this_week  = [u for u in upcoming if u.get("group") == "this_week"]
    high_tasks = [t for t in tasks if t.get("urgency_score", 0) >= 0.7]

    lines = []
    if overdue:
        lines.append(f"OVERDUE: {', '.join(x['title'] for x in overdue[:3])}")
    if today:
        lines.append(f"Due today: {', '.join(x['title'] for x in today[:3])}")
    if this_week:
        lines.append(f"This week: {', '.join(x['title'] for x in this_week[:3])}")
    if high_tasks:
        lines.append(f"High urgency tasks: {', '.join(x['title'] for x in high_tasks[:3])}")
    elif tasks:
        lines.append(f"Open tasks ({len(tasks)}): {', '.join(x['title'] for x in tasks[:3])}")
    if knowledge:
        lines.append(f"Learning goals: {', '.join(x['title'] for x in knowledge[:3])}")

    if not lines:
        return {
            "priorities": [],
            "summary": "Your hub is empty. Submit a capture on the Dashboard to get started.",
        }

    data_block = "\n".join(lines)
    prompt = (
        "You are a personal chief-of-staff AI. Generate a structured digest.\n"
        f"Current state:\n{data_block}\n\n"
        "Return up to 5 priority items (critical/high/medium/low) as short actionable lines, "
        "and a 1-sentence summary. critical=overdue/blocking, high=due today, "
        "medium=this week or important, low=on track/informational."
    )

    try:
        resp = _client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=400,
            tools=[DIGEST_SCHEMA],
            tool_choice={"type": "tool", "name": "generate_digest"},
            messages=[{"role": "user", "content": prompt}],
        )
        for block in resp.content:
            if block.type == "tool_use" and block.name == "generate_digest":
                return block.input
    except Exception:
        pass

    # Template fallback
    parts: list = []
    for u in overdue[:2]:
        parts.append({"level": "critical", "text": f"Resolve overdue: {u['title']}"})
    for u in today[:2]:
        parts.append({"level": "high", "text": u["title"]})
    for t in high_tasks[:2]:
        if not any(p["text"] == t["title"] for p in parts):
            parts.append({"level": "high", "text": t["title"]})
    for t in tasks[:2]:
        if not any(p["text"] == t["title"] for p in parts):
            parts.append({"level": "medium", "text": t["title"]})
    return {
        "priorities": parts[:5],
        "summary": f"{len(upcoming)} upcoming, {len(tasks)} tasks, {len(knowledge)} learning items.",
    }


@router.get("")
async def get_hub():
    pool = await db.get_pool()
    async with pool.acquire() as conn:
        upcoming_raw = await _query_items(
            conn, UPCOMING_TYPES,
            order="CASE WHEN ei.deadline IS NULL THEN 1 ELSE 0 END, ei.deadline ASC, ei.urgency_score DESC",
        )
        tasks = await _query_items(
            conn, TASK_TYPES,
            order="ei.urgency_score DESC, ei.created_at DESC",
        )
        knowledge = await _query_items(
            conn, ("knowledge",),
            order="ei.created_at DESC",
            limit=20,
        )

    # Annotate upcoming with time-group
    for item in upcoming_raw:
        dl = item.get("deadline")
        if dl:
            try:
                from datetime import datetime as dt2
                parsed = dt2.fromisoformat(dl.replace("Z", "+00:00"))
                item["group"] = _group(parsed)
            except ValueError:
                item["group"] = "no_date"
        else:
            item["group"] = "no_date"

    stats = _compute_stats(upcoming_raw, tasks, knowledge)
    digest = _generate_structured_digest(upcoming_raw, tasks, knowledge)

    return {
        "summary":    digest.get("summary", ""),
        "priorities": digest.get("priorities", []),
        "stats":      stats,
        "upcoming":   upcoming_raw,
        "tasks":      tasks,
        "knowledge":  knowledge,
    }
