"""
Memory Agent — fixes:
H2: Assert list lengths; raise immediately on DB write failure
C4: Store run_id on the captures row
"""
import uuid
from datetime import datetime
from typing import Optional

from sentence_transformers import SentenceTransformer

from models import AgentState
from memory import db

_model: Optional[SentenceTransformer] = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def _embed(text: str) -> list:
    model = _get_model()
    return model.encode(text, normalize_embeddings=True).tolist()


async def memory_agent(state: AgentState) -> AgentState:
    """Store capture + all extracted items in PostgreSQL with vector embeddings."""
    timestamp = datetime.utcnow().isoformat()

    capture = state.get("capture", {})
    items = state.get("extracted_items", [])
    run_id = state.get("run_id")  # C4

    # Step 1 — embed + store capture
    capture_id = str(uuid.uuid4())
    raw_content = capture.get("raw_content") or ""
    embedding = _embed(raw_content) if raw_content else None

    await db.insert_capture(
        id=capture_id,
        modality=capture.get("modality", "text"),
        source=capture.get("source", "paste"),
        raw_content=capture.get("raw_content"),
        file_path=capture.get("file_path"),
        embedding=embedding,
        metadata=capture.get("metadata", {}),
        run_id=run_id,  # C4
    )

    # Step 2 — embed + store each extracted item
    item_ids: list = []
    for i, item in enumerate(items):
        item_id = str(uuid.uuid4())
        text_to_embed = f"{item.get('title', '')} {item.get('description', '')}".strip()
        item_embedding = _embed(text_to_embed) if text_to_embed else None

        try:
            await db.insert_extracted_item(
                id=item_id,
                capture_id=capture_id,
                title=item.get("title", ""),
                description=item.get("description"),
                item_type=item.get("item_type", "knowledge"),
                confidence_score=item.get("confidence_score", 0.5),
                urgency_score=item.get("urgency_score", 0.0),
                entities=item.get("entities", {}),
                deadline=item.get("deadline"),
                embedding=item_embedding,
                metadata=item.get("metadata", {}),
            )
        except Exception as exc:
            # H2: raise immediately with context; do not silently continue
            raise RuntimeError(
                f"Memory Agent: failed to write item {i} ('{item.get('title')}') to DB: {exc}"
            ) from exc

        item_ids.append(item_id)

    # H2: assert alignment before returning
    assert len(item_ids) == len(items), (
        f"Memory Agent: item count mismatch — {len(items)} items provided, "
        f"{len(item_ids)} IDs written"
    )

    # M4: lightweight trace
    trace_entry = {
        "agent": "memory",
        "capture_id": capture_id,
        "items_stored": len(item_ids),
        "timestamp": timestamp,
    }

    return {
        **state,
        "capture_id": capture_id,
        "extracted_item_ids": item_ids,
        "trace": state.get("trace", []) + [trace_entry],
    }
