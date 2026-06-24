import os
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request


def get_credentials() -> Credentials:
    """Return a valid Google OAuth2 Credentials object, refreshing if needed."""
    creds = Credentials(
        token=None,
        refresh_token=os.getenv("GOOGLE_REFRESH_TOKEN"),
        token_uri="https://oauth2.googleapis.com/token",
        client_id=os.getenv("GOOGLE_CLIENT_ID"),
        client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
    )
    creds.refresh(Request())
    return creds
