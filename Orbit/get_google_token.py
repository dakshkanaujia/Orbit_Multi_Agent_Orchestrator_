"""
One-time script to generate a Google refresh token with the correct scopes.
Run this once: python generate_token.py
Then copy the printed refresh token into your .env as GOOGLE_REFRESH_TOKEN.
"""
import os
from google_auth_oauthlib.flow import InstalledAppFlow
from dotenv import load_dotenv

load_dotenv()

SCOPES = [
    "https://www.googleapis.com/auth/gmail.compose",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/calendar.events",
]

client_config = {
    "installed": {
        "client_id": os.getenv("GOOGLE_CLIENT_ID"),
        "client_secret": os.getenv("GOOGLE_CLIENT_SECRET"),
        "redirect_uris": ["http://localhost:8000/auth/google/callback", "urn:ietf:wg:oauth:2.0:oob"],
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
    }
}

flow = InstalledAppFlow.from_client_config(client_config, SCOPES)
creds = flow.run_local_server(port=8080)

print("\n✅ Done! Copy these into your .env file:\n")
print(f"GOOGLE_REFRESH_TOKEN={creds.refresh_token}")
print(f"GOOGLE_ACCESS_TOKEN={creds.token}")