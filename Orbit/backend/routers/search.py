from typing import Optional

from fastapi import APIRouter, Query, HTTPException

from memory.retrieval import assemble_retrieval_context
from memory.db import safe_json

router = APIRouter(prefix="/api/search", tags=["search"])


@router.get("")
async def search(
    query: str = Query(..., min_length=1),
    item_type: Optional[str] = Query(None),
    date_from: Optional[str] = Query(None),
    min_urgency: Optional[float] = Query(None, ge=0.0, le=1.0),
    limit: int = Query(10, ge=1, le=50),
):
    try:
        results = await assemble_retrieval_context(
            query=query,
            item_type=item_type,
            date_from=date_from,
            min_urgency=min_urgency,
            limit=limit,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

    serialized = []
    for r in results:
        entry = {"type": r.get("type"), "semantic_score": r.get("semantic_score", 0.0)}
        if r.get("type") == "item":
            item = dict(r.get("item") or {})
            item["entities"] = safe_json(item.get("entities"))
            item["metadata"] = safe_json(item.get("metadata"))
            parent = dict(r.get("parent_capture") or {})
            parent["metadata"] = safe_json(parent.get("metadata"))
            entry["item"] = item
            entry["parent_capture"] = parent
            entry["actions"] = [
                {**dict(a), "payload": safe_json(a.get("payload"))}
                for a in (r.get("actions") or [])
            ]
        else:
            cap = dict(r.get("capture") or {})
            cap["metadata"] = safe_json(cap.get("metadata"))
            entry["capture"] = cap
        serialized.append(entry)

    return serialized
