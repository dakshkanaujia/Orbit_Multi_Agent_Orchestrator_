"""
Planning Agent — generates executable actions for each extracted item.
M2: Set planning_status = 'skipped_low_confidence' on skipped items
H10: Prompt injection guard + structured output
M4: Lightweight trace
"""
import json
import logging
import os
import uuid
from datetime import datetime

import anthropic

from models import AgentState, VALID_ACTION_TYPES
from memory import db
from agents.guardrails import check_action_payload

_log = logging.getLogger(__name__)

_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

ACTIONS_SCHEMA = {
    "name": "generate_actions",
    "description": "Generate executable actions for this extracted item.",
    "input_schema": {
        "type": "object",
        "properties": {
            "actions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "action_type":       {"type": "string"},
                        "payload":           {"type": "object"},
                        "requires_approval": {"type": "boolean"},
                    },
                    "required": ["action_type", "payload", "requires_approval"],
                },
            }
        },
        "required": ["actions"],
    },
}

PLANNING_PROMPT = """You are generating executable actions for a personal chief-of-staff.

Item:
- Title: {title}
- Type: {item_type}
- Description: {description}
- Urgency: {urgency_score}
- Deadline: {deadline}
- Entities: {entities}

Valid action_types: calendar.create_booking, gmail.send_email, slack.send_reminder, slack.send_summary

CRITICAL RULE — INTENT MATCHING:
Only generate actions that the user EXPLICITLY asked for. Do NOT infer or add supplementary actions.
- If the title/description mentions a calendar event or meeting → calendar.create_booking ONLY
- If the title/description mentions sending an email → gmail.send_email ONLY (single action)
- If the title/description mentions Slack → slack.send_reminder or slack.send_summary ONLY
- If the intent is unclear or mixed → generate the single most relevant action
- NEVER generate calendar + email + slack together unless the user asked for all three

ACTION RULES:
1. calendar.create_booking payload: {{start, attendee_name, attendee_email, timezone}} — start must be ISO 8601 (e.g. "2026-07-01T10:00:00Z").
2. gmail.send_email payload: {{to, subject, body}} — sends immediately on approval, no draft step.
3. Slack channel defaults to "{default_channel}".
4. If recipient email is unknown, use "unknown@example.com".
5. If attendee timezone is unknown, use "UTC".
6. requires_approval=true for all action types."""


async def planning_agent(state: AgentState) -> AgentState:
    items = state.get("extracted_items", [])
    item_ids = state.get("extracted_item_ids", [])
    timestamp = datetime.utcnow().isoformat()
    default_channel = os.getenv("SLACK_DEFAULT_CHANNEL", "general")

    all_actions: list = []
    clarification_needed = state.get("clarification_needed", False)
    clarification_reason = state.get("clarification_reason", "")

    for item, item_id in zip(items, item_ids):
        if item.get("confidence_score", 0.0) < 0.5:
            # M2: record why item was skipped
            await db.update_extracted_item_planning_status(item_id, "skipped_low_confidence")
            continue

        # Knowledge items are stored as information in the Hub — no external actions needed.
        if item.get("item_type") == "knowledge":
            await db.update_extracted_item_planning_status(item_id, "skipped_no_actions")
            continue

        prompt = PLANNING_PROMPT.format(
            title=item.get("title", ""),
            item_type=item.get("item_type", ""),
            description=item.get("description", ""),
            urgency_score=item.get("urgency_score", 0.0),
            deadline=item.get("deadline", "null"),
            entities=json.dumps(item.get("entities", {})),
            default_channel=default_channel,
        )

        response = _client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=2048,
            tools=[ACTIONS_SCHEMA],
            tool_choice={"type": "tool", "name": "generate_actions"},
            messages=[{"role": "user", "content": prompt}],
        )

        raw_actions = []
        for block in response.content:
            if block.type == "tool_use" and block.name == "generate_actions":
                raw_actions = block.input.get("actions", [])
                break

        # Filter to known action types only (C3)
        raw_actions = [a for a in raw_actions if a.get("action_type") in VALID_ACTION_TYPES]

        # Auto-fix: if send_email has no valid recipient, substitute placeholder.
        for action in raw_actions:
            if action.get("action_type") == "gmail.send_email":
                to_field = action.get("payload", {}).get("to", "")
                if not to_field or "@" not in str(to_field):
                    action.setdefault("payload", {})["to"] = "unknown@example.com"

        item_actions_written: list = []

        for action in raw_actions:
            # G3: validate payload before persisting — blocks bad values early
            guard = check_action_payload(action.get("action_type", ""), action.get("payload", {}))
            if not guard.passed:
                _log.warning("Planning Agent: action blocked by G3 — %s", guard.reason)
                continue

            action_id = str(uuid.uuid4())

            await db.insert_action(
                id=action_id,
                extracted_item_id=item_id,
                action_type=action.get("action_type", ""),
                payload=action.get("payload", {}),
                requires_approval=action.get("requires_approval", True),
                depends_on_action_id=None,
            )

            item_actions_written.append({
                "id": action_id,
                "extracted_item_id": item_id,
                "action_type": action.get("action_type", ""),
                "payload": action.get("payload", {}),
                "requires_approval": action.get("requires_approval", True),
                "depends_on_action_id": None,
                "status": "pending",
            })

        if item_actions_written:
            await db.update_extracted_item_planning_status(item_id, "planned")
        else:
            await db.update_extracted_item_planning_status(item_id, "skipped_no_actions")

        all_actions.extend(item_actions_written)

    # M4: lightweight trace
    trace_entry = {
        "agent": "planning",
        "actions_generated": len(all_actions),
        "action_types": [a["action_type"] for a in all_actions],
        "timestamp": timestamp,
    }

    return {
        **state,
        "actions": all_actions,
        "pending_action_ids": [a["id"] for a in all_actions if a.get("requires_approval")],
        "clarification_needed": clarification_needed,
        "clarification_reason": clarification_reason,
        "trace": state.get("trace", []) + [trace_entry],
    }
