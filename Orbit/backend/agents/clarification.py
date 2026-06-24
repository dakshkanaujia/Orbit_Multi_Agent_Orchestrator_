"""C1: Clarification halt node — reached when clarification_needed == True."""
from models import AgentState


def clarification_halt_node(state: AgentState) -> AgentState:
    """
    Terminal node when the Intent Agent flags ambiguous content.
    Returns state as-is; the caller reads clarification_needed + clarification_reason.
    The POST /api/captures endpoint returns HTTP 422 with clarification_reason.
    """
    return state
