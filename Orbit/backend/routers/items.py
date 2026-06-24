from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from memory import db
from memory.db import safe_json

router = APIRouter(prefix="/api/items", tags=["items"])


@router.get("")
async def list_items(
    item_type: Optional[str] = Query(None),
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
    min_urgency: Optional[float] = Query(None, ge=0.0, le=1.0),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    items = await db.list_extracted_items(
        item_type=item_type,
        date_from=date_from,
        date_to=date_to,
        min_urgency=min_urgency,
        limit=limit,
        offset=offset,
    )
    result = []
    for item in items:
        item_dict = dict(item)
        item_dict["entities"] = safe_json(item_dict.get("entities"))
        item_dict["metadata"] = safe_json(item_dict.get("metadata"))
        result.append(item_dict)
    return result


@router.get("/{item_id}")
async def get_item(item_id: str):
    item = await db.get_extracted_item(item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")

    actions = await db.get_actions_by_item(item_id)
    item_dict = dict(item)
    item_dict["entities"] = safe_json(item_dict.get("entities"))
    item_dict["metadata"] = safe_json(item_dict.get("metadata"))
    item_dict["actions"] = [
        {**dict(a), "payload": safe_json(a.get("payload"))}
        for a in actions
    ]
    return item_dict
