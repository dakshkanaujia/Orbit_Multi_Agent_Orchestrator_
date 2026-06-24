import asyncio
import json
from typing import Optional, List, Dict, Any

from memory.db import get_pool

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def _embed(text: str) -> list:
    model = _get_model()
    return model.encode(text, normalize_embeddings=True).tolist()


async def _search_items(
    conn,
    embedding: list,
    item_type: Optional[str],
    date_from: Optional[str],
    min_urgency: Optional[float],
    limit: int,
) -> List[dict]:
    emb_str = f"[{','.join(str(v) for v in embedding)}]"
    conditions = ["ei.embedding IS NOT NULL"]
    params: List[Any] = [emb_str]
    idx = 2

    if item_type:
        conditions.append(f"ei.item_type = ${idx}")
        params.append(item_type)
        idx += 1
    if date_from:
        conditions.append(f"ei.deadline >= ${idx}::timestamp")
        params.append(date_from)
        idx += 1
    if min_urgency is not None:
        conditions.append(f"ei.urgency_score >= ${idx}")
        params.append(min_urgency)
        idx += 1

    params.append(limit)
    where = " AND ".join(conditions)
    query = f"""
        SELECT ei.id, ei.capture_id, ei.title, ei.description, ei.item_type,
               ei.confidence_score, ei.urgency_score, ei.entities, ei.deadline,
               ei.metadata, ei.created_at,
               1 - (ei.embedding <=> ${1}::vector) AS semantic_score
        FROM extracted_items ei
        WHERE {where}
        ORDER BY ei.embedding <=> ${1}::vector
        LIMIT ${idx}
    """
    rows = await conn.fetch(query, *params)
    return [dict(r) for r in rows]


async def _search_captures(conn, embedding: list, limit: int) -> List[dict]:
    emb_str = f"[{','.join(str(v) for v in embedding)}]"
    rows = await conn.fetch(
        """
        SELECT c.id, c.modality, c.source, c.raw_content, c.file_path, c.metadata, c.created_at,
               1 - (c.embedding <=> $1::vector) AS semantic_score
        FROM captures c
        WHERE c.embedding IS NOT NULL
        ORDER BY c.embedding <=> $1::vector
        LIMIT $2
        """,
        emb_str, limit,
    )
    return [dict(r) for r in rows]


async def assemble_retrieval_context(
    query: str,
    item_type: Optional[str] = None,
    date_from: Optional[str] = None,
    min_urgency: Optional[float] = None,
    limit: int = 10,
) -> List[Dict]:
    embedding = _embed(query)
    pool = await get_pool()

    async def _fetch_items():
        async with pool.acquire() as conn:
            return await _search_items(conn, embedding, item_type, date_from, min_urgency, limit)

    async def _fetch_captures():
        async with pool.acquire() as conn:
            return await _search_captures(conn, embedding, max(3, limit // 2))

    items, captures = await asyncio.gather(_fetch_items(), _fetch_captures())

    async with pool.acquire() as conn:
        # Fetch parent captures for matched items
        capture_ids = list({i["capture_id"] for i in items})
        parent_captures: dict = {}
        if capture_ids:
            placeholders = ",".join(f"${i+1}" for i in range(len(capture_ids)))
            cap_rows = await conn.fetch(
                f"SELECT id, modality, source, raw_content, file_path, metadata, created_at FROM captures WHERE id IN ({placeholders})",
                *capture_ids,
            )
            parent_captures = {r["id"]: dict(r) for r in cap_rows}

        # Fetch actions for matched items
        item_ids = [i["id"] for i in items]
        item_actions: dict = {i["id"]: [] for i in items}
        if item_ids:
            placeholders = ",".join(f"${i+1}" for i in range(len(item_ids)))
            act_rows = await conn.fetch(
                f"SELECT id, extracted_item_id, action_type, payload, status, requires_approval, created_at FROM actions WHERE extracted_item_id IN ({placeholders})",
                *item_ids,
            )
            for r in act_rows:
                item_actions.setdefault(r["extracted_item_id"], []).append(dict(r))

    context = []
    seen_capture_ids = set()

    for item in items:
        parent = parent_captures.get(item["capture_id"], {})
        seen_capture_ids.add(item["capture_id"])
        context.append({
            "type": "item",
            "item": item,
            "parent_capture": parent,
            "actions": item_actions.get(item["id"], []),
            "semantic_score": float(item.get("semantic_score", 0.0)),
        })

    for cap in captures:
        if cap["id"] not in seen_capture_ids:
            context.append({
                "type": "capture",
                "capture": cap,
                "semantic_score": float(cap.get("semantic_score", 0.0)),
            })

    context.sort(key=lambda x: x["semantic_score"], reverse=True)
    return context[: limit * 2]
