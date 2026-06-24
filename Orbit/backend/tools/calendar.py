import os
import httpx

CALCOM_API_KEY = os.getenv("CALCOM_API_KEY", "")
CALCOM_DEFAULT_EVENT_TYPE_ID = int(os.getenv("CALCOM_EVENT_TYPE_ID", "0"))


def create_booking(
    start: str,
    attendee_name: str,
    attendee_email: str,
    event_type_id: int = 0,
    timezone: str = "UTC",
) -> dict:
    eid = event_type_id or CALCOM_DEFAULT_EVENT_TYPE_ID
    r = httpx.post(
        "https://api.cal.com/v2/bookings",
        json={
            "eventTypeId": eid,
            "start": start,
            "attendee": {
                "name": attendee_name,
                "email": attendee_email,
                "timeZone": timezone,
                "language": "en",
            },
        },
        headers={
            "Authorization": f"Bearer {CALCOM_API_KEY}",
            "cal-api-version": "2024-08-13",
            "Content-Type": "application/json",
        },
        timeout=15,
    )
    r.raise_for_status()
    data = r.json()
    booking = data.get("data", data)
    return {
        "booking_uid": booking.get("uid"),
        "status": booking.get("status", "created"),
        "start": start,
        "attendee_email": attendee_email,
    }
