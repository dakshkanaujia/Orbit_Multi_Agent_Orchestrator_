from typing import TypedDict, Optional, List
from pydantic import BaseModel
from datetime import datetime


class AgentState(TypedDict):
    # Input
    input_content: str
    input_type: str               # "text" | "pdf" | "image"

    # C4: run_id is set before graph.invoke() and stored on captures
    run_id: str

    # Understanding Agent output
    capture: dict                 # {modality, source, raw_content, file_path, metadata}
    extracted_entities: dict      # {people, dates, locations, deadlines, organizations, urls}

    # Intent Agent output
    extracted_items: list         # list of ExtractedItem dicts (pre-DB write)

    # Memory Agent output (after DB writes)
    capture_id: str
    extracted_item_ids: list      # parallel to extracted_items (same order, asserted equal length)

    # Planning Agent output
    actions: list                 # flat list of Action dicts across all items

    # H3: per-action approval tracking for stateful multi-resume
    pending_action_ids: list      # populated by tool_router
    decided_action_ids: list      # appended to on each approval resume

    # Approval Agent output
    decisions: list               # list of Decision dicts
    execution_results: list

    # Retrieval
    retrieval_context: list

    # Guardrails
    clarification_needed: bool
    clarification_reason: str

    # M4: lightweight trace (summaries only, not full output blobs)
    trace: list                   # [{agent, timestamp, ...summary fields}]


# ── Action payload schemas (M6) ────────────────────────────────────────────

class CalComBookingSchema(BaseModel):
    start: str              # ISO 8601 e.g. "2026-07-01T10:00:00Z"
    attendee_name: str
    attendee_email: str
    event_type_id: int = 0  # falls back to CALCOM_EVENT_TYPE_ID env var when 0
    timezone: str = "UTC"


class GmailSendSchema(BaseModel):
    to: str
    subject: str
    body: str


class SlackReminderSchema(BaseModel):
    channel: str
    message: str


class SlackSummarySchema(BaseModel):
    channel: str
    summary: str


ACTION_SCHEMAS = {
    "calendar.create_booking": CalComBookingSchema,
    "gmail.send_email":        GmailSendSchema,
    "slack.send_reminder":     SlackReminderSchema,
    "slack.send_summary":      SlackSummarySchema,
}

VALID_ACTION_TYPES = set(ACTION_SCHEMAS.keys())


# ── Pydantic response models ───────────────────────────────────────────────

class ExtractedItemOut(BaseModel):
    id: str
    capture_id: str
    title: str
    description: Optional[str] = None
    item_type: str
    confidence_score: float
    urgency_score: float
    entities: dict = {}
    deadline: Optional[datetime] = None
    metadata: dict = {}
    created_at: datetime
    planning_status: str = "pending"


class ActionOut(BaseModel):
    id: str
    extracted_item_id: str
    action_type: str
    payload: dict = {}
    status: str
    requires_approval: bool
    depends_on_action_id: Optional[str] = None
    created_at: datetime


class DecisionOut(BaseModel):
    id: str
    action_id: Optional[str] = None
    decision: str
    edited_payload: Optional[dict] = None
    final_payload: Optional[dict] = None       # H5: approved payload
    execution_result: Optional[dict] = None    # H5: tool return value
    decided_at: datetime


class CaptureOut(BaseModel):
    id: str
    run_id: Optional[str] = None
    modality: str
    source: str
    raw_content: Optional[str] = None
    file_path: Optional[str] = None
    metadata: dict = {}
    created_at: datetime
    extracted_items: List[ExtractedItemOut] = []


class ActionWithContext(BaseModel):
    action: ActionOut
    extracted_item: Optional[ExtractedItemOut] = None


class SearchResult(BaseModel):
    item: Optional[ExtractedItemOut] = None
    parent_capture: Optional[CaptureOut] = None
    actions: List[ActionOut] = []
    semantic_score: float


class DashboardOut(BaseModel):
    recent_captures: List[CaptureOut]
    pending_count: int
    item_type_breakdown: dict


class ProcessResponse(BaseModel):
    run_id: str
    capture_id: str
    extracted_count: int
    actions_count: int
    clarification_needed: bool
    clarification_reason: Optional[str] = None


class CaptureStatusResponse(BaseModel):
    capture_id: str
    status: str  # "processing" | "complete"
    item_count: int
    pending_action_count: int


class ApproveRequest(BaseModel):
    pass


class RejectRequest(BaseModel):
    reason: Optional[str] = None


class EditRequest(BaseModel):
    edited_payload: dict
