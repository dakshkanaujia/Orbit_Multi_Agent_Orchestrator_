import os
from typing import Dict, Any

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError


def _get_client() -> WebClient:
    return WebClient(token=os.getenv("SLACK_BOT_TOKEN"))


def send_reminder(channel: str, message: str) -> Dict[str, Any]:
    """Post a reminder message to a Slack channel."""
    client = _get_client()
    try:
        response = client.chat_postMessage(channel=channel, text=f"⏰ Reminder: {message}")
        return {"ts": response["ts"], "channel": response["channel"], "status": "sent"}
    except SlackApiError as e:
        return {"error": str(e), "status": "failed"}


def send_summary(channel: str, summary: str) -> Dict[str, Any]:
    """Post a summary to a Slack channel."""
    client = _get_client()
    try:
        response = client.chat_postMessage(
            channel=channel,
            text=summary,
            blocks=[
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": f"📋 *Summary*\n{summary}"},
                }
            ],
        )
        return {"ts": response["ts"], "channel": response["channel"], "status": "sent"}
    except SlackApiError as e:
        return {"error": str(e), "status": "failed"}
