import type {
  Capture,
  ExtractedItem,
  Action,
  ActionWithContext,
  SearchResult,
  ProcessResponse,
  DashboardData,
  HubData,
} from "./types";

const BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...init?.headers },
    ...init,
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`API ${res.status}: ${err}`);
  }
  return res.json() as Promise<T>;
}

// ── Captures ───────────────────────────────────────────────────────────────

export async function submitCapture(formData: FormData): Promise<ProcessResponse> {
  const res = await fetch(`${BASE}/api/captures`, { method: "POST", body: formData });
  if (!res.ok) throw new Error(`API ${res.status}: ${await res.text()}`);
  return res.json();
}

export function listCaptures(limit = 20, offset = 0): Promise<Capture[]> {
  return request(`/api/captures?limit=${limit}&offset=${offset}`);
}

export function getCapture(captureId: string): Promise<Capture> {
  return request(`/api/captures/${captureId}`);
}

// ── Items ──────────────────────────────────────────────────────────────────

export function listItems(params?: {
  item_type?: string;
  date_from?: string;
  date_to?: string;
  min_urgency?: number;
  limit?: number;
  offset?: number;
}): Promise<ExtractedItem[]> {
  const q = new URLSearchParams();
  if (params?.item_type) q.set("item_type", params.item_type);
  if (params?.date_from) q.set("date_from", params.date_from);
  if (params?.date_to) q.set("date_to", params.date_to);
  if (params?.min_urgency != null) q.set("min_urgency", String(params.min_urgency));
  if (params?.limit) q.set("limit", String(params.limit));
  if (params?.offset) q.set("offset", String(params.offset));
  return request(`/api/items?${q}`);
}

export function getItem(itemId: string): Promise<ExtractedItem> {
  return request(`/api/items/${itemId}`);
}

// ── Actions ────────────────────────────────────────────────────────────────

export function getPendingActions(): Promise<ActionWithContext[]> {
  return request("/api/actions/pending");
}

export function approveAction(actionId: string): Promise<unknown> {
  return request(`/api/actions/${actionId}/approve`, { method: "POST", body: "{}" });
}

export function rejectAction(actionId: string, reason?: string): Promise<unknown> {
  return request(`/api/actions/${actionId}/reject`, {
    method: "POST",
    body: JSON.stringify({ reason }),
  });
}

export function editAction(actionId: string, editedPayload: Record<string, unknown>): Promise<unknown> {
  return request(`/api/actions/${actionId}/edit`, {
    method: "POST",
    body: JSON.stringify({ edited_payload: editedPayload }),
  });
}

// ── Search ─────────────────────────────────────────────────────────────────

export function search(params: {
  query: string;
  item_type?: string;
  date_from?: string;
  min_urgency?: number;
  limit?: number;
}): Promise<SearchResult[]> {
  const q = new URLSearchParams({ query: params.query });
  if (params.item_type) q.set("item_type", params.item_type);
  if (params.date_from) q.set("date_from", params.date_from);
  if (params.min_urgency != null) q.set("min_urgency", String(params.min_urgency));
  if (params.limit) q.set("limit", String(params.limit));
  return request(`/api/search?${q}`);
}

// ── Dashboard ──────────────────────────────────────────────────────────────

export function getDashboard(): Promise<DashboardData> {
  return request("/api/dashboard");
}

// ── Hub ────────────────────────────────────────────────────────────────────

export function getHub(): Promise<HubData> {
  return request("/api/hub");
}
