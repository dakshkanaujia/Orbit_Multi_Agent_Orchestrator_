"""
Actions router:
H4:  409 on double-approve (idempotency via decisions.action_id unique index)
M7:  SELECT FOR UPDATE to prevent race conditions between concurrent requests
M6:  Validate edited_payload against action-type Pydantic schema
H5:  Use final_payload + execution_result (not final_action)
"""
import uuid

from fastapi import APIRouter, HTTPException
from pydantic import ValidationError

from memory import db
from memory.db import safe_json
from models import EditRequest, RejectRequest, ACTION_SCHEMAS
from agents.guardrails import check_action_payload

router = APIRouter(prefix="/api/actions", tags=["actions"])


def _get_handler(action_type: str):
    """C3: resolve handler at execution time from string action_type."""
    if action_type == "calendar.create_booking":
        from tools.calendar import create_booking
        return create_booking
    elif action_type == "gmail.send_email":
        from tools.gmail import send_email
        return send_email
    elif action_type == "slack.send_reminder":
        from tools.slack import send_reminder
        return send_reminder
    elif action_type == "slack.send_summary":
        from tools.slack import send_summary
        return send_summary
    return None


def _execute_tool(action_type: str, payload: dict) -> dict:
    """Execute the resolved tool handler. Returns result dict."""
    handler = _get_handler(action_type)
    if handler is None:
        return {"status": "failed", "error": f"No handler for {action_type}"}
    try:
        result = handler(**payload)
        return result if isinstance(result, dict) else {"result": str(result), "status": "executed"}
    except Exception as e:
        return {"status": "failed", "error": str(e)}


@router.get("/pending")
async def get_pending_actions():
    """Pending actions enriched with parent item context."""
    actions = await db.get_pending_actions()
    result = []
    for action in actions:
        item = await db.get_extracted_item(action["extracted_item_id"])
        action_dict = dict(action)
        action_dict["payload"] = safe_json(action_dict.get("payload"))
        result.append({
            "action": action_dict,
            "extracted_item": {
                **dict(item),
                "entities": safe_json((item or {}).get("entities")),
                "metadata": safe_json((item or {}).get("metadata")),
            } if item else None,
        })
    return result


@router.post("/{action_id}/approve")
async def approve_action(action_id: str):
    """Approve + execute action. H4: 409 on double-approve. M7: DB-level lock."""
    pool = await db.get_pool()

    async with pool.acquire() as conn:
        async with conn.transaction():
            # M7: SELECT FOR UPDATE prevents race conditions
            action_row = await conn.fetchrow(
                "SELECT id, extracted_item_id, action_type, payload, status, depends_on_action_id "
                "FROM actions WHERE id = $1 AND status = 'pending' FOR UPDATE",
                action_id,
            )
            if not action_row:
                existing = await conn.fetchrow("SELECT status FROM actions WHERE id = $1", action_id)
                if existing:
                    raise HTTPException(status_code=409, detail=f"Action is already {existing['status']}")
                raise HTTPException(status_code=404, detail="Action not found")

            # H4: check for existing decision (unique index also enforces this at DB level)
            existing_decision = await db.get_decision_by_action(action_id)
            if existing_decision:
                raise HTTPException(status_code=409, detail="Action already has a decision")

            action = dict(action_row)
            action["payload"] = safe_json(action["payload"]) or {}

    final_payload = dict(action["payload"])

    # G3: last-mile payload check before execution
    guard = check_action_payload(action["action_type"], final_payload)
    if not guard.passed:
        raise HTTPException(
            status_code=422,
            detail={"error": "guardrail_violation", "rule": guard.rule, "reason": guard.reason},
        )

    execution_result = _execute_tool(action["action_type"], final_payload)
    final_status = "executed" if execution_result.get("status") != "failed" else "failed"

    await db.update_action_status(action_id, final_status)
    decision = await db.insert_decision(
        id=str(uuid.uuid4()),
        action_id=action_id,
        decision="approved",
        edited_payload=None,
        final_payload=final_payload,        # H5
        execution_result=execution_result,  # H5
    )

    return {"decision": decision, "execution_result": execution_result}


@router.post("/{action_id}/reject")
async def reject_action(action_id: str, body: RejectRequest = RejectRequest()):
    """Reject an action without executing it."""
    pool = await db.get_pool()

    async with pool.acquire() as conn:
        async with conn.transaction():
            action_row = await conn.fetchrow(
                "SELECT id, status FROM actions WHERE id = $1 AND status = 'pending' FOR UPDATE",
                action_id,
            )
            if not action_row:
                existing = await conn.fetchrow("SELECT status FROM actions WHERE id = $1", action_id)
                if existing:
                    raise HTTPException(status_code=409, detail=f"Action is already {existing['status']}")
                raise HTTPException(status_code=404, detail="Action not found")

    existing_decision = await db.get_decision_by_action(action_id)
    if existing_decision:
        raise HTTPException(status_code=409, detail="Action already has a decision")

    await db.update_action_status(action_id, "rejected")
    decision = await db.insert_decision(
        id=str(uuid.uuid4()),
        action_id=action_id,
        decision="rejected",
        final_payload=None,
        execution_result={"reason": body.reason} if body.reason else None,
    )

    return {"decision": decision}


@router.post("/{action_id}/edit")
async def edit_and_approve_action(action_id: str, body: EditRequest):
    """Edit payload + approve + execute. M6: validates payload against action schema."""
    pool = await db.get_pool()

    async with pool.acquire() as conn:
        async with conn.transaction():
            action_row = await conn.fetchrow(
                "SELECT id, action_type, status FROM actions WHERE id = $1 AND status = 'pending' FOR UPDATE",
                action_id,
            )
            if not action_row:
                existing = await conn.fetchrow("SELECT status FROM actions WHERE id = $1", action_id)
                if existing:
                    raise HTTPException(status_code=409, detail=f"Action is already {existing['status']}")
                raise HTTPException(status_code=404, detail="Action not found")

    existing_decision = await db.get_decision_by_action(action_id)
    if existing_decision:
        raise HTTPException(status_code=409, detail="Action already has a decision")

    action_type = action_row["action_type"]

    # M6: validate edited_payload against action-type schema
    schema_cls = ACTION_SCHEMAS.get(action_type)
    if schema_cls:
        try:
            schema_cls(**body.edited_payload)
        except ValidationError as e:
            raise HTTPException(status_code=422, detail=e.errors())

    # G3: check edited payload values before execution
    guard = check_action_payload(action_type, body.edited_payload)
    if not guard.passed:
        raise HTTPException(
            status_code=422,
            detail={"error": "guardrail_violation", "rule": guard.rule, "reason": guard.reason},
        )

    execution_result = _execute_tool(action_type, body.edited_payload)
    final_status = "executed" if execution_result.get("status") != "failed" else "failed"

    await db.update_action_status(action_id, final_status)
    decision = await db.insert_decision(
        id=str(uuid.uuid4()),
        action_id=action_id,
        decision="edited",
        edited_payload=body.edited_payload,
        final_payload=body.edited_payload,   # H5
        execution_result=execution_result,   # H5
    )

    return {"decision": decision, "execution_result": execution_result}
