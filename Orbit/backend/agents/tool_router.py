"""
Tool Router Agent — C3 fix:
Store ONLY string action_type in state. No callable/function references.
Handler resolution happens in approval execution (routers/actions.py).
This node validates action types and populates pending_action_ids.
"""
from datetime import datetime
from models import AgentState, VALID_ACTION_TYPES


def tool_router_agent(state: AgentState) -> AgentState:
    """Validate action types; attach pending_action_ids to state."""
    timestamp = datetime.utcnow().isoformat()
    actions = state.get("actions", [])
    invalid_count = 0

    for action in actions:
        if action.get("action_type") not in VALID_ACTION_TYPES:
            action["status"] = "failed"
            action.setdefault("metadata", {})["error"] = (
                f"Unknown action_type: {action.get('action_type')}"
            )
            invalid_count += 1

    pending_ids = [
        a["id"] for a in actions
        if a.get("status") == "pending" and a.get("requires_approval", True)
    ]

    # M4: lightweight trace
    trace_entry = {
        "agent": "tool_router",
        "resolved": len(actions) - invalid_count,
        "invalid": invalid_count,
        "pending_action_count": len(pending_ids),
        "timestamp": timestamp,
    }

    return {
        **state,
        "actions": actions,
        "pending_action_ids": pending_ids,
        "decided_action_ids": state.get("decided_action_ids", []),
        "trace": state.get("trace", []) + [trace_entry],
    }
