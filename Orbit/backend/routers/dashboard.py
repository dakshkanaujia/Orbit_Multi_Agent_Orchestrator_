from fastapi import APIRouter
from memory import db
from memory.db import safe_json

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])


@router.get("")
async def get_dashboard():
    """Return recent captures, pending action count, and item type breakdown."""
    captures_raw = await db.list_captures(limit=10, offset=0)
    pending_count = await db.get_pending_action_count()
    item_type_breakdown = await db.get_item_type_breakdown()

    recent_captures = []
    for cap in captures_raw:
        items = await db.get_items_by_capture(cap["id"])
        actions = []
        for item in items:
            item_actions = await db.get_actions_by_item(item["id"])
            actions.extend(item_actions)

        pending_count = sum(1 for a in actions if a.get("status") == "pending")

        cap_dict = dict(cap)
        cap_dict["metadata"] = safe_json(cap_dict.get("metadata"))
        cap_dict["item_count"] = len(items)
        cap_dict["action_count"] = len(actions)
        cap_dict["pending_action_count"] = pending_count
        recent_captures.append(cap_dict)

    return {
        "recent_captures": recent_captures,
        "pending_count": pending_count,
        "item_type_breakdown": item_type_breakdown,
    }
