import base64
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from googleapiclient.discovery import build
from tools.auth import get_credentials


def send_email(to: str, subject: str, body: str) -> dict:
    """Build and send a Gmail message directly (no draft step)."""
    service = build("gmail", "v1", credentials=get_credentials())
    msg = MIMEMultipart("alternative")
    msg["To"] = to
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))
    msg.attach(MIMEText(f"<p>{body}</p>", "html"))
    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
    result = service.users().messages().send(
        userId="me", body={"raw": raw}
    ).execute()
    return {"message_id": result["id"], "to": to, "subject": subject, "status": "sent"}
