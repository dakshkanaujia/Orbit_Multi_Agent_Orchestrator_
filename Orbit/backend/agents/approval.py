"""
Approval Agent — the graph pauses BEFORE this node (interrupt_before=["approval"]).

In the REST API flow, human decisions arrive via:
  POST /api/actions/{id}/approve
  POST /api/actions/{id}/reject
  POST /api/actions/{id}/edit

These endpoints call _execute_decision() directly — they do NOT resume the graph.
This node only runs if the graph is explicitly resumed after ALL decisions are made.

C3:  DISPATCH table — handler resolved at execution time from string action_type
"""
from datetime import datetime
from models import AgentState


# C3: DISPATCH table — resolve handler from string at execution time
def _get_handler(action_type: str):
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


def approval_agent(state: AgentState) -> AgentState:
    """
    No-op checkpoint node — actual execution is handled by REST endpoints.
    Logs summary for the trace.
    """
    timestamp = datetime.utcnow().isoformat()
    decisions = state.get("decisions", [])
    execution_results = state.get("execution_results", [])

    trace_entry = {
        "agent": "approval",
        "decisions_processed": len(decisions),
        "execution_results": len(execution_results),
        "timestamp": timestamp,
    }

    return {
        **state,
        "trace": state.get("trace", []) + [trace_entry],
    }
