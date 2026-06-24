export type PlanningStatus =
  | "pending"
  | "planned"
  | "skipped_low_confidence"
  | "skipped_no_actions";

export interface ExtractedItem {
  id: string;
  capture_id: string;
  title: string;
  description: string | null;
  item_type: ItemType;
  confidence_score: number;
  urgency_score: number;
  entities: Record<string, unknown>;
  deadline: string | null;
  // M2: planning pipeline status
  planning_status: PlanningStatus;
  metadata: Record<string, unknown>;
  created_at: string;
  actions?: Action[];
}

export type ItemType =
  | "event"
  | "deadline"
  | "task"
  | "communication"
  | "travel_interest"
  | "job_opportunity"
  | "meeting"
  | "reminder"
  | "knowledge";

export interface Action {
  id: string;
  extracted_item_id: string;
  action_type: string;
  payload: Record<string, unknown>;
  status: ActionStatus;
  requires_approval: boolean;
  created_at: string;
}

export type ActionStatus = "pending" | "approved" | "rejected" | "executed" | "failed";

export interface Decision {
  id: string;
  action_id: string;
  decision: "approved" | "rejected" | "edited";
  edited_payload: Record<string, unknown> | null;
  // H5: split final_action into payload + result
  final_payload: Record<string, unknown> | null;
  execution_result: Record<string, unknown> | null;
  decided_at: string;
}

export interface Capture {
  id: string;
  modality: "image" | "pdf" | "text";
  source: "upload" | "paste" | "email" | "screenshot";
  raw_content: string | null;
  file_path: string | null;
  metadata: Record<string, unknown>;
  created_at: string;
  extracted_items?: ExtractedItem[];
  item_count?: number;
  action_count?: number;
  // M1: truncation info in metadata
  items_truncated?: number;
}

export interface ApproveResponse {
  decision: Decision;
  execution_result: Record<string, unknown>;
}

export interface ActionWithContext {
  action: Action;
  extracted_item: ExtractedItem | null;
}

export interface SearchResult {
  type: "item" | "capture";
  semantic_score: number;
  item?: ExtractedItem;
  parent_capture?: Capture;
  actions?: Action[];
  capture?: Capture;
}

export interface ProcessResponse {
  run_id: string;
  capture_id: string;
  extracted_count: number;
  actions_count: number;
  clarification_needed: boolean;
  clarification_reason: string | null;
}

export type HubGroup = "overdue" | "today" | "this_week" | "later" | "no_date";
export type PriorityLevel = "critical" | "high" | "medium" | "low";

export interface HubItem extends ExtractedItem {
  group?: HubGroup;
  pending_actions: number;
  total_actions: number;
}

export interface HubPriority {
  level: PriorityLevel;
  text: string;
}

export interface HubStats {
  overdue: number;
  today: number;
  this_week: number;
  total_tasks: number;
  knowledge_items: number;
  upcoming_total: number;
}

export interface HubData {
  summary: string;
  priorities: HubPriority[];
  stats: HubStats;
  upcoming: HubItem[];
  tasks: HubItem[];
  knowledge: HubItem[];
}

export interface DashboardData {
  recent_captures: Capture[];
  pending_count: number;
  item_type_breakdown: Record<string, number>;
}

