"""
Captures router — fixes:
C1: Return HTTP 422 when clarification_needed is True (no partial capture stored)
C4: run_id passed as thread_id and stored on capture
H8: Add GET /api/captures/{id}/status polling endpoint
H7: Add DELETE /api/captures/{id} soft-delete endpoint
M8: Upload size enforced via MAX_UPLOAD_BYTES env var (also set in main.py middleware)
"""
import json
import os
import uuid
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse

from models import ProcessResponse, CaptureStatusResponse
from memory import db
from memory.db import safe_json
from graph import app as graph_app
from agents.guardrails import run_input_guardrails

router = APIRouter(prefix="/api/captures", tags=["captures"])

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/app/uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(20 * 1024 * 1024)))  # 20 MB


@router.post("/stream")
async def stream_capture(
    content: Optional[str] = Form(None),
    source: str = Form("paste"),
):
    """Run the capture pipeline and stream SSE agent-completion events."""
    if not content or not content.strip():
        raise HTTPException(status_code=400, detail="'content' is required")

    # Input guardrails: G0 length + G1 injection + G2 LLM safety
    violation = await run_input_guardrails(content)
    if violation:
        raise HTTPException(
            status_code=422,
            detail={"error": "guardrail_violation", "rule": violation.rule, "reason": violation.reason},
        )

    run_id = str(uuid.uuid4())
    await db.insert_run(run_id)

    initial_state = {
        "input_content": content,
        "input_type": "text",
        "run_id": run_id,
        "capture": {},
        "extracted_entities": {},
        "extracted_items": [],
        "capture_id": "",
        "extracted_item_ids": [],
        "actions": [],
        "pending_action_ids": [],
        "decided_action_ids": [],
        "decisions": [],
        "retrieval_context": [],
        "execution_results": [],
        "clarification_needed": False,
        "clarification_reason": "",
        "trace": [],
    }
    # LangSmith: run_name + metadata appear in the trace UI when LANGCHAIN_TRACING_V2=true
    config = {
        "configurable": {"thread_id": run_id},
        "run_name": "orbit-pipeline",
        "metadata": {"orbit_run_id": run_id, "input_type": "text"},
        "tags": ["orbit", "stream"],
    }

    async def generate():
        final_state: dict = {}
        try:
            async for event in graph_app.astream(initial_state, config=config, stream_mode="updates"):
                node_name = list(event.keys())[0]
                node_update = event[node_name]
                final_state.update(node_update)

                if node_name == "understanding":
                    yield f"event: agent\ndata: {json.dumps({'agent': 'understanding', 'status': 'done'})}\n\n"
                elif node_name == "intent":
                    item_count = len(node_update.get("extracted_items") or [])
                    yield f"event: agent\ndata: {json.dumps({'agent': 'intent', 'status': 'done', 'item_count': item_count})}\n\n"
                elif node_name == "memory":
                    yield f"event: agent\ndata: {json.dumps({'agent': 'memory', 'status': 'done'})}\n\n"
                elif node_name == "planning":
                    action_count = len(node_update.get("actions") or [])
                    yield f"event: agent\ndata: {json.dumps({'agent': 'planning', 'status': 'done', 'action_count': action_count})}\n\n"
                elif node_name == "clarification_halt":
                    reason = final_state.get("clarification_reason", "")
                    await db.update_run(run_id, "failed", [], None)
                    yield f"event: error\ndata: {json.dumps({'error': 'clarification_required', 'reason': reason})}\n\n"
                    return

            capture_id = final_state.get("capture_id", "")
            actions_count = len(final_state.get("actions") or [])
            await db.update_run(run_id, "interrupted", [], capture_id)
            yield f"event: done\ndata: {json.dumps({'capture_id': capture_id, 'run_id': run_id, 'actions_count': actions_count})}\n\n"

        except Exception as e:
            await db.update_run(run_id, "failed", [], None)
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("", response_model=ProcessResponse)
async def create_capture(
    file: Optional[UploadFile] = File(None),
    content: Optional[str] = Form(None),
    source: str = Form("paste"),
):
    """Submit a file upload or text paste. Returns run_id + capture_id."""
    run_id = str(uuid.uuid4())

    if file is not None:
        file_bytes = await file.read()
        if len(file_bytes) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="File exceeds maximum allowed size.")
        ext = (file.filename or "upload").rsplit(".", 1)[-1].lower()
        file_path = os.path.join(UPLOAD_DIR, f"{run_id}.{ext}")
        with open(file_path, "wb") as f:
            f.write(file_bytes)
        content_type = file.content_type or ""
        input_type = "pdf" if ("pdf" in content_type or ext == "pdf") else "image"
        input_content = file_path
    elif content:
        input_type = "text"
        input_content = content
        file_path = None
    else:
        raise HTTPException(status_code=400, detail="Either 'file' or 'content' must be provided.")

    # Input guardrails: only on text (file uploads go through understanding agent first)
    if input_type == "text":
        violation = await run_input_guardrails(input_content)
        if violation:
            raise HTTPException(
                status_code=422,
                detail={"error": "guardrail_violation", "rule": violation.rule, "reason": violation.reason},
            )

    initial_state = {
        "input_content": input_content,
        "input_type": input_type,
        "run_id": run_id,      # C4
        "capture": {},
        "extracted_entities": {},
        "extracted_items": [],
        "capture_id": "",
        "extracted_item_ids": [],
        "actions": [],
        "pending_action_ids": [],
        "decided_action_ids": [],
        "decisions": [],
        "retrieval_context": [],
        "execution_results": [],
        "clarification_needed": False,
        "clarification_reason": "",
        "trace": [],
    }

    # LangSmith: run_name + metadata appear in the trace UI when LANGCHAIN_TRACING_V2=true
    config = {
        "configurable": {"thread_id": run_id},
        "run_name": "orbit-pipeline",
        "metadata": {"orbit_run_id": run_id, "input_type": input_type},
        "tags": ["orbit"],
    }
    await db.insert_run(run_id)

    try:
        final_state = None
        async for event in graph_app.astream(initial_state, config=config, stream_mode="values"):
            final_state = event
    except Exception as e:
        await db.update_run(run_id, "failed", [], None)
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

    if final_state is None:
        await db.update_run(run_id, "failed", [], None)
        raise HTTPException(status_code=500, detail="Pipeline returned no state.")

    # C1: if clarification needed, do NOT proceed — return 422 with reason
    if final_state.get("clarification_needed"):
        await db.update_run(run_id, "failed", final_state.get("trace", []), None)
        # Clean up uploaded file if it was created before intent stage failed
        if file is not None and file_path and os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(
            status_code=422,
            detail={
                "error": "clarification_required",
                "clarification_reason": final_state.get("clarification_reason", ""),
            },
        )

    capture_id = final_state.get("capture_id", "")
    actions = final_state.get("actions", [])
    trace = final_state.get("trace", [])

    await db.update_run(run_id, "interrupted", trace, capture_id)

    return ProcessResponse(
        run_id=run_id,
        capture_id=capture_id,
        extracted_count=len(final_state.get("extracted_item_ids", [])),
        actions_count=len(actions),
        clarification_needed=False,
        clarification_reason=None,
    )


@router.get("/{capture_id}/status", response_model=CaptureStatusResponse)
async def get_capture_status(capture_id: str):
    """H8: polling endpoint for frontend ProcessingStatus component."""
    capture = await db.get_capture(capture_id)
    if not capture:
        raise HTTPException(status_code=404, detail="Capture not found")
    item_count = await db.count_extracted_items(capture_id)
    pending_count = await db.count_pending_actions_for_capture(capture_id)
    return CaptureStatusResponse(
        capture_id=capture_id,
        status="complete" if item_count > 0 else "processing",
        item_count=item_count,
        pending_action_count=pending_count,
    )


@router.get("", response_model=list)
async def list_captures(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    captures = await db.list_captures(limit=limit, offset=offset)
    result = []
    for cap in captures:
        items = await db.get_items_by_capture(cap["id"])
        cap["extracted_items"] = items
        cap["metadata"] = safe_json(cap.get("metadata"))
        result.append(cap)
    return result


@router.get("/{capture_id}")
async def get_capture(capture_id: str):
    capture = await db.get_capture(capture_id)
    if not capture:
        raise HTTPException(status_code=404, detail="Capture not found")

    items = await db.get_items_by_capture(capture_id)
    enriched_items = []
    for item in items:
        actions = await db.get_actions_by_item(item["id"])
        item_dict = dict(item)
        item_dict["metadata"] = safe_json(item_dict.get("metadata"))
        item_dict["entities"] = safe_json(item_dict.get("entities"))
        item_dict["actions"] = [
            {**dict(a), "payload": safe_json(a.get("payload"))}
            for a in actions
        ]
        enriched_items.append(item_dict)

    capture["extracted_items"] = enriched_items
    capture["metadata"] = safe_json(capture.get("metadata"))
    return capture


@router.delete("/{capture_id}")
async def delete_capture(capture_id: str):
    """H7: soft delete — preserves decisions audit trail; removes uploaded file."""
    capture = await db.get_capture(capture_id)
    if not capture:
        raise HTTPException(status_code=404, detail="Capture not found")

    # M8: clean up uploaded file
    file_path = capture.get("file_path")
    if file_path and os.path.exists(file_path):
        try:
            os.remove(file_path)
        except OSError:
            pass

    await db.soft_delete_capture(capture_id)
    return {"deleted": True, "capture_id": capture_id}
