"""
C1: Conditional edge after intent node — routes to clarification_halt when
    state["clarification_needed"] is True instead of proceeding to memory.
C4: run_id is passed in initial_state and becomes the LangGraph thread_id.
"""
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from models import AgentState
from agents.understanding import understanding_agent
from agents.intent import intent_agent
from agents.memory import memory_agent
from agents.planning import planning_agent
from agents.tool_router import tool_router_agent
from agents.approval import approval_agent
from agents.clarification import clarification_halt_node

checkpointer = MemorySaver()

workflow = StateGraph(AgentState)

workflow.add_node("understanding", understanding_agent)
workflow.add_node("intent", intent_agent)
workflow.add_node("memory", memory_agent)
workflow.add_node("planning", planning_agent)
workflow.add_node("tool_router", tool_router_agent)
workflow.add_node("approval", approval_agent)
workflow.add_node("clarification_halt", clarification_halt_node)

workflow.set_entry_point("understanding")
workflow.add_edge("understanding", "intent")

# C1: branch after intent — halt if clarification needed
def route_after_intent(state: AgentState) -> str:
    if state.get("clarification_needed"):
        return "clarification_halt"
    return "memory"

workflow.add_conditional_edges(
    "intent",
    route_after_intent,
    {"memory": "memory", "clarification_halt": "clarification_halt"},
)
workflow.add_edge("clarification_halt", END)

workflow.add_edge("memory", "planning")
workflow.add_edge("planning", "tool_router")
workflow.add_edge("tool_router", "approval")
workflow.add_edge("approval", END)

# Graph pauses before approval — REST endpoints handle per-action decisions
app = workflow.compile(checkpointer=checkpointer, interrupt_before=["approval"])
