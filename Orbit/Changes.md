# Orbit AI — Changes

## What Changed and Why

### Backend: Google OAuth Removed (original removal)

The original Google Calendar + Gmail OAuth setup required a complex multi-step credential flow (client ID, secret, refresh token). This was replaced with simpler integrations.

---

### Backend: Gmail (SMTP via App Password)

**File:** `backend/tools/gmail.py`

Sends email via Gmail SMTP using an App Password — no OAuth needed.

**Action type:** `gmail.send_email`
**Payload:**
```json
{ "to": "someone@example.com", "subject": "Hello", "body": "Message text" }
```

**Env vars required:**
```
GMAIL_ADDRESS=your_gmail@gmail.com
GMAIL_APP_PASSWORD=xxxx xxxx xxxx xxxx
```

To get an App Password: Google Account → Security → 2-Step Verification → App Passwords → generate one for "Mail".

---

### Backend: Cal.com (Calendar Bookings)

**File:** `backend/tools/calendar.py`

Creates bookings on Cal.com via their REST API.

**Action type:** `calendar.create_booking`
**Payload:**
```json
{
  "start": "2026-07-01T10:00:00Z",
  "attendee_name": "John Doe",
  "attendee_email": "john@example.com",
  "event_type_id": 0,
  "timezone": "UTC"
}
```
`event_type_id=0` falls back to `CALCOM_EVENT_TYPE_ID` env var.

**Env vars required:**
```
CALCOM_API_KEY=your_calcom_api_key_here
CALCOM_EVENT_TYPE_ID=123456
```

Get your API key at https://app.cal.com/settings/developer/api-keys. Find your event type ID in the Cal.com dashboard URL when editing an event type.

---

### Backend: Action Types

| Type | Payload fields | Description |
|------|---------------|-------------|
| `calendar.create_booking` | `start`, `attendee_name`, `attendee_email`, `timezone?` | Book a meeting via Cal.com |
| `gmail.send_email` | `to`, `subject`, `body` | Send email via Gmail SMTP |
| `slack.send_reminder` | `channel`, `message` | Post Slack reminder |
| `slack.send_summary` | `channel`, `summary` | Post Slack summary |

---

### Frontend: UI Overhaul

**Navigation:**
- Replaced emoji logo with `Zap` icon (lucide-react) + wordmark
- Active nav link highlighted (`bg-gray-100 text-gray-900`)
- Added Trace link to nav
- Nav is a separate `NavLinks` client component (needed for `usePathname`)

**Icons:**
- All emoji replaced with lucide-react SVG icons throughout
- `ItemTypeBadge` uses type-matched icons (Calendar, Clock, CheckSquare, etc.)
- Modality icons: FileText (PDF), Image, Type (text)
- Action status chips include CheckCircle / XCircle icons
- Approvals cards show CalendarCheck (cal.com) / Mail (gmail) / Bell (slack reminder)

**Loading states:** Animated skeleton cards replace "Loading…" text on all pages.

**Empty states:** Each page has a centered icon + message when there's no data.

**Other:** Progress bars taller (`h-2`), Filter icon in dropdowns, sort toggle on Items, Search icon inside input.

---

### How to Run

```bash
# 1. Fill in backend/.env:
#    GMAIL_ADDRESS, GMAIL_APP_PASSWORD
#    CALCOM_API_KEY, CALCOM_EVENT_TYPE_ID

# 2. Start
cd Orbit
docker compose up --build
```

Open http://localhost:3000/dashboard.
