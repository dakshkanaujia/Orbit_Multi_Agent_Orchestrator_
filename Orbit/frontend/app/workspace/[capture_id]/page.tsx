"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { getCapture } from "@/lib/api";
import { formatDate, formatDateTime, truncate } from "@/lib/utils";
import { ItemTypeBadge } from "@/components/ItemTypeBadge";
import { Progress } from "@/components/ui/progress";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { FileText, Image, Type, CalendarDays, CheckCircle, XCircle, Clock } from "lucide-react";
import type { Capture, ExtractedItem, Action, PlanningStatus } from "@/lib/types";

function ActionStatusBadge({ status }: { status: Action["status"] }) {
  const map: Record<string, { cls: string; Icon: React.ElementType }> = {
    pending:  { cls: "bg-yellow-100 text-yellow-800", Icon: Clock },
    approved: { cls: "bg-blue-100 text-blue-800",    Icon: CheckCircle },
    executed: { cls: "bg-green-100 text-green-800",  Icon: CheckCircle },
    rejected: { cls: "bg-gray-100 text-gray-500",    Icon: XCircle },
    failed:   { cls: "bg-red-100 text-red-800",      Icon: XCircle },
  };
  const { cls, Icon } = map[status] ?? map.pending;
  return (
    <span className={`inline-flex items-center gap-1 text-xs font-medium rounded-full px-2 py-0.5 ${cls}`}>
      <Icon className="w-3 h-3" />{status}
    </span>
  );
}

// M2: human-readable planning status badge
const PLANNING_STATUS_LABELS: Record<PlanningStatus, { label: string; className: string }> = {
  pending:                  { label: "Pending",          className: "bg-gray-100 text-gray-500" },
  planned:                  { label: "Planned",          className: "bg-green-100 text-green-700" },
  skipped_low_confidence:   { label: "Skipped — low confidence", className: "bg-yellow-100 text-yellow-700" },
  skipped_no_actions:       { label: "Skipped — no actions",     className: "bg-gray-100 text-gray-500" },
};

function PlanningStatusBadge({ status, confidence }: { status: PlanningStatus; confidence: number }) {
  const { label, className } = PLANNING_STATUS_LABELS[status] ?? PLANNING_STATUS_LABELS.pending;
  const withScore = status === "skipped_low_confidence"
    ? `${label} (${(confidence * 100).toFixed(0)}%)`
    : label;
  return (
    <span className={`text-xs font-medium rounded-full px-2 py-0.5 ${className}`}>
      {withScore}
    </span>
  );
}

function ItemCard({ item }: { item: ExtractedItem & { actions?: Action[] } }) {
  const [expanded, setExpanded] = useState(false);
  const actions = item.actions ?? [];

  return (
    <Card>
      <CardHeader>
        <div className="flex items-start justify-between gap-2">
          <div>
            <div className="flex items-center gap-2 flex-wrap">
              <ItemTypeBadge type={item.item_type} />
              {/* M2: show planning_status when not the default "planned" */}
              {item.planning_status && item.planning_status !== "planned" && (
                <PlanningStatusBadge status={item.planning_status} confidence={item.confidence_score} />
              )}
              {item.deadline && (
                <span className="inline-flex items-center gap-1 text-xs bg-amber-50 text-amber-700 rounded-full px-2 py-0.5 border border-amber-200">
                  <CalendarDays className="w-3 h-3" />{formatDate(item.deadline)}
                </span>
              )}
            </div>
            <p className="font-semibold text-gray-900 mt-2 text-sm">{item.title}</p>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        <p className="text-sm text-gray-600">{truncate(item.description, 160)}</p>
        <Progress value={item.confidence_score} label="Confidence" showScore />
        <Progress value={item.urgency_score} label="Urgency" showScore />
        {actions.length > 0 && (
          <div>
            <button
              className="text-xs text-blue-600 hover:underline"
              onClick={() => setExpanded((v) => !v)}
            >
              {expanded ? "Hide" : "Show"} {actions.length} action{actions.length !== 1 ? "s" : ""}
            </button>
            {expanded && (
              <div className="mt-2 space-y-1">
                {actions.map((a) => (
                  <div key={a.id} className="flex items-center justify-between text-xs bg-gray-50 rounded px-2 py-1.5">
                    <code className="text-gray-700">{a.action_type}</code>
                    <ActionStatusBadge status={a.status} />
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export default function WorkspacePage() {
  const { capture_id } = useParams<{ capture_id: string }>();
  const [capture, setCapture] = useState<Capture | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getCapture(capture_id)
      .then(setCapture)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [capture_id]);

  if (loading) return <p className="text-gray-500">Loading workspace…</p>;
  if (error) return <div className="text-red-600 bg-red-50 rounded p-4">{error}</div>;
  if (!capture) return <p className="text-gray-500">Capture not found.</p>;

  const items = (capture.extracted_items ?? []) as (ExtractedItem & { actions?: Action[] })[];
  const totalActions = items.reduce((sum, i) => sum + (i.actions?.length ?? 0), 0);
  const pendingActions = items.reduce(
    (sum, i) => sum + (i.actions?.filter((a) => a.status === "pending").length ?? 0),
    0
  );
  // M1: items_truncated stored in metadata by Intent Agent
  const itemsTruncated = (capture.metadata?.items_truncated as number | undefined) ?? 0;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <div className="flex items-center gap-2 text-sm text-gray-500 mb-1">
            <Link href="/dashboard" className="hover:text-gray-700">Dashboard</Link>
            <span>›</span>
            <span>Capture Workspace</span>
          </div>
          <h1 className="text-xl font-bold text-gray-900 flex items-center gap-2">
            {capture.modality === "pdf"
              ? <FileText className="w-5 h-5 text-red-500" />
              : capture.modality === "image"
              ? <Image className="w-5 h-5 text-blue-500" />
              : <Type className="w-5 h-5 text-gray-500" />}
            {capture.modality} · {capture.source}
          </h1>
          <p className="text-sm text-gray-500 mt-1">
            Captured {formatDateTime(capture.created_at)} · {items.length} items · {totalActions} actions
          </p>
        </div>
        {pendingActions > 0 && (
          <Link href="/approvals">
            <span className="bg-red-100 text-red-800 text-sm font-medium rounded-full px-3 py-1">
              {pendingActions} pending
            </span>
          </Link>
        )}
      </div>

      {/* M1: truncation banner — shown when Intent Agent hit MAX_EXTRACTED_ITEMS_PER_CAPTURE */}
      {itemsTruncated > 0 && (
        <div className="bg-amber-50 border border-amber-200 rounded-lg px-4 py-3 text-sm text-amber-800">
          <strong>{itemsTruncated}</strong> additional item{itemsTruncated !== 1 ? "s were" : " was"} detected in this capture but not extracted — extraction limit reached. Increase <code>MAX_EXTRACTED_ITEMS_PER_CAPTURE</code> to surface them.
        </div>
      )}

      {/* Raw content preview */}
      {capture.raw_content && (
        <Card>
          <CardHeader>
            <h2 className="text-sm font-semibold text-gray-700">Raw Content</h2>
          </CardHeader>
          <CardContent>
            <pre className="text-xs text-gray-600 whitespace-pre-wrap font-mono max-h-40 overflow-y-auto">
              {truncate(capture.raw_content, 800)}
            </pre>
          </CardContent>
        </Card>
      )}

      {/* Extracted Items Grid */}
      {items.length === 0 ? (
        <p className="text-gray-500 text-sm">No items extracted from this capture.</p>
      ) : (
        <div>
          <h2 className="text-lg font-semibold text-gray-900 mb-4">
            Extracted Items ({items.length})
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {items.map((item) => (
              <ItemCard key={item.id} item={item} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
