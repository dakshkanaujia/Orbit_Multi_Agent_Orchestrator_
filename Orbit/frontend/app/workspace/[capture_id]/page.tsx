"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { getCapture } from "@/lib/api";
import { formatDate, formatDateTime, formatRelativeTime, truncate, ITEM_TYPE_BORDER_COLORS } from "@/lib/utils";
import { ItemTypeBadge } from "@/components/ItemTypeBadge";
import { Progress } from "@/components/ui/progress";
import { Card, CardContent } from "@/components/ui/card";
import {
  FileText, Image, Type, CalendarDays, CheckCircle, XCircle,
  Clock, ChevronDown, ChevronUp, ArrowRight, AlertTriangle,
  Mail, MessageSquare, Bell, CalendarPlus, Zap, Eye, EyeOff,
} from "lucide-react";
import type { Capture, ExtractedItem, Action, PlanningStatus } from "@/lib/types";

// ─── Helpers ──────────────────────────────────────────────────────────────────

const ACTION_TYPE_LABELS: Record<string, { label: string; Icon: React.ElementType }> = {
  "gmail.send_email":        { label: "Send Email",        Icon: Mail },
  "calendar.create_booking": { label: "Calendar Booking",  Icon: CalendarPlus },
  "slack.send_reminder":     { label: "Slack Reminder",    Icon: Bell },
  "slack.send_summary":      { label: "Slack Summary",     Icon: MessageSquare },
};

function friendlyActionType(actionType: string) {
  return ACTION_TYPE_LABELS[actionType] ?? { label: actionType, Icon: Zap };
}

// ─── Action status badge ───────────────────────────────────────────────────────

function ActionStatusBadge({ status }: { status: Action["status"] }) {
  const map: Record<string, { cls: string; Icon: React.ElementType }> = {
    pending:  { cls: "bg-amber-50 text-amber-700 border border-amber-200",    Icon: Clock },
    approved: { cls: "bg-blue-50 text-blue-700 border border-blue-200",       Icon: CheckCircle },
    executed: { cls: "bg-green-50 text-green-700 border border-green-200",    Icon: CheckCircle },
    rejected: { cls: "bg-gray-50 text-gray-500 border border-gray-200",       Icon: XCircle },
    failed:   { cls: "bg-red-50 text-red-700 border border-red-200",          Icon: XCircle },
  };
  const { cls, Icon } = map[status] ?? map.pending;
  return (
    <span className={`inline-flex items-center gap-1 text-xs font-medium rounded-full px-2 py-0.5 ${cls}`}>
      <Icon className="w-3 h-3" />{status}
    </span>
  );
}

// ─── Planning status badge ─────────────────────────────────────────────────────

const PLANNING_STATUS_LABELS: Record<PlanningStatus, { label: string; className: string }> = {
  pending:                  { label: "Pending",               className: "bg-gray-100 text-gray-500" },
  planned:                  { label: "Planned",               className: "bg-green-100 text-green-700" },
  skipped_low_confidence:   { label: "Skipped — low confidence", className: "bg-amber-100 text-amber-700" },
  skipped_no_actions:       { label: "Skipped — no actions",  className: "bg-gray-100 text-gray-500" },
};

function PlanningStatusBadge({ status, confidence }: { status: PlanningStatus; confidence: number }) {
  const { label, className } = PLANNING_STATUS_LABELS[status] ?? PLANNING_STATUS_LABELS.pending;
  const withScore = status === "skipped_low_confidence"
    ? `${label} (${(confidence * 100).toFixed(0)}%)`
    : label;
  return (
    <span className={`text-xs font-medium rounded-full px-2 py-0.5 ${className}`}>{withScore}</span>
  );
}

// ─── Skeleton loader ───────────────────────────────────────────────────────────

function SkeletonCard() {
  return (
    <div className="rounded-xl border border-gray-100 bg-white p-5 space-y-3 animate-pulse">
      <div className="flex gap-2">
        <div className="h-5 w-24 bg-gray-100 rounded-full" />
        <div className="h-5 w-20 bg-gray-100 rounded-full" />
      </div>
      <div className="h-4 w-3/4 bg-gray-100 rounded" />
      <div className="h-3 w-full bg-gray-100 rounded" />
      <div className="h-3 w-5/6 bg-gray-100 rounded" />
      <div className="space-y-2 pt-1">
        <div className="h-2 w-full bg-gray-100 rounded-full" />
        <div className="h-2 w-full bg-gray-100 rounded-full" />
      </div>
    </div>
  );
}

// ─── Item card ─────────────────────────────────────────────────────────────────

function ItemCard({ item }: { item: ExtractedItem & { actions?: Action[] } }) {
  const [expanded, setExpanded] = useState(false);
  const actions = item.actions ?? [];
  const borderColor = ITEM_TYPE_BORDER_COLORS[item.item_type] ?? "border-l-gray-300";

  return (
    <div className={`rounded-xl border border-gray-100 bg-white shadow-sm border-l-4 ${borderColor} overflow-hidden`}>
      <div className="p-5 space-y-3">
        {/* Badges row */}
        <div className="flex items-center gap-2 flex-wrap">
          <ItemTypeBadge type={item.item_type} />
          {item.planning_status && item.planning_status !== "planned" && (
            <PlanningStatusBadge status={item.planning_status} confidence={item.confidence_score} />
          )}
          {item.deadline && (
            <span className="inline-flex items-center gap-1 text-xs bg-amber-50 text-amber-700 rounded-full px-2 py-0.5 border border-amber-100">
              <CalendarDays className="w-3 h-3" />{formatDate(item.deadline)}
            </span>
          )}
        </div>

        {/* Title */}
        <h3 className="font-semibold text-gray-900 text-sm leading-snug">{item.title}</h3>

        {/* Description */}
        <p className="text-sm text-gray-500 leading-relaxed">{truncate(item.description, 160)}</p>

        {/* Metrics */}
        <div className="space-y-2 pt-1">
          <Progress value={item.confidence_score} label="Confidence" showScore color="blue" />
          <Progress value={item.urgency_score}    label="Urgency"    showScore color="amber" />
        </div>

        {/* Actions toggle */}
        {actions.length > 0 && (
          <div>
            <button
              className="inline-flex items-center gap-1 text-xs font-medium text-gray-500 hover:text-gray-800 transition-colors pt-1"
              onClick={() => setExpanded((v) => !v)}
            >
              {expanded ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
              {expanded ? "Hide" : "Show"} {actions.length} action{actions.length !== 1 ? "s" : ""}
            </button>

            {expanded && (
              <div className="mt-2 space-y-1.5">
                {actions.map((a) => {
                  const { label, Icon } = friendlyActionType(a.action_type);
                  return (
                    <div key={a.id} className="flex items-center justify-between bg-gray-50 rounded-lg px-3 py-2">
                      <span className="inline-flex items-center gap-1.5 text-xs text-gray-700 font-medium">
                        <Icon className="w-3.5 h-3.5 text-gray-400" />
                        {label}
                      </span>
                      <ActionStatusBadge status={a.status} />
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

// ─── Raw content preview ───────────────────────────────────────────────────────

function RawContentCard({ content }: { content: string }) {
  const [visible, setVisible] = useState(false);
  const preview = truncate(content, 120);

  return (
    <div className="rounded-xl border border-gray-100 bg-white shadow-sm">
      <div className="flex items-center justify-between px-5 py-3.5 border-b border-gray-50">
        <span className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Source Input</span>
        <button
          onClick={() => setVisible((v) => !v)}
          className="inline-flex items-center gap-1 text-xs text-gray-400 hover:text-gray-700 transition-colors"
        >
          {visible ? <EyeOff className="w-3.5 h-3.5" /> : <Eye className="w-3.5 h-3.5" />}
          {visible ? "Hide" : "Show"}
        </button>
      </div>
      <div className="px-5 py-4">
        {visible ? (
          <pre className="text-xs text-gray-600 whitespace-pre-wrap font-mono leading-relaxed max-h-36 overflow-y-auto">
            {truncate(content, 800)}
          </pre>
        ) : (
          <p className="text-xs text-gray-400 font-mono leading-relaxed">{preview}</p>
        )}
      </div>
    </div>
  );
}

// ─── Page ──────────────────────────────────────────────────────────────────────

const MODALITY_ICONS: Record<string, React.ElementType> = {
  pdf:   FileText,
  image: Image,
  text:  Type,
};

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

  if (loading) {
    return (
      <div className="space-y-6">
        <div className="space-y-2 animate-pulse">
          <div className="h-3 w-40 bg-gray-100 rounded" />
          <div className="h-6 w-64 bg-gray-100 rounded" />
          <div className="h-3 w-48 bg-gray-100 rounded" />
        </div>
        <div className="h-24 bg-gray-50 rounded-xl" />
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          <SkeletonCard /><SkeletonCard /><SkeletonCard />
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-start gap-3 bg-red-50 border border-red-100 rounded-xl px-5 py-4">
        <XCircle className="w-4 h-4 text-red-500 mt-0.5 shrink-0" />
        <p className="text-sm text-red-700">{error}</p>
      </div>
    );
  }

  if (!capture) {
    return (
      <div className="text-center py-20 text-gray-400">
        <FileText className="w-10 h-10 mx-auto mb-3 opacity-30" />
        <p className="text-sm">Capture not found.</p>
      </div>
    );
  }

  const items = (capture.extracted_items ?? []) as (ExtractedItem & { actions?: Action[] })[];
  const totalActions = items.reduce((sum, i) => sum + (i.actions?.length ?? 0), 0);
  const pendingActions = items.reduce(
    (sum, i) => sum + (i.actions?.filter((a) => a.status === "pending").length ?? 0),
    0
  );
  const itemsTruncated = (capture.metadata?.items_truncated as number | undefined) ?? 0;

  const ModalityIcon = MODALITY_ICONS[capture.modality] ?? Type;

  return (
    <div className="space-y-7">

      {/* ── Header ── */}
      <div className="flex items-start justify-between gap-4">
        <div className="space-y-1">
          {/* Breadcrumb */}
          <div className="flex items-center gap-1.5 text-xs text-gray-400">
            <Link href="/dashboard" className="hover:text-gray-600 transition-colors">Dashboard</Link>
            <span>›</span>
            <span>Capture Workspace</span>
          </div>

          {/* Title */}
          <h1 className="text-xl font-bold text-gray-900 flex items-center gap-2.5">
            <span className="inline-flex items-center justify-center w-8 h-8 rounded-lg bg-gray-100">
              <ModalityIcon className="w-4 h-4 text-gray-500" />
            </span>
            <span className="capitalize">{capture.modality}</span>
            <span className="text-gray-300">·</span>
            <span className="text-gray-600 font-medium">{capture.source}</span>
          </h1>

          {/* Stat chips */}
          <div className="flex items-center gap-2 pt-0.5 flex-wrap">
            <span className="text-xs text-gray-400 bg-gray-50 border border-gray-100 rounded-full px-2.5 py-0.5">
              {formatDateTime(capture.created_at)}
            </span>
            <span className="text-xs text-gray-400 bg-gray-50 border border-gray-100 rounded-full px-2.5 py-0.5">
              {items.length} item{items.length !== 1 ? "s" : ""}
            </span>
            <span className="text-xs text-gray-400 bg-gray-50 border border-gray-100 rounded-full px-2.5 py-0.5">
              {totalActions} action{totalActions !== 1 ? "s" : ""}
            </span>
          </div>
        </div>

        {/* Pending CTA */}
        {pendingActions > 0 && (
          <Link href="/approvals">
            <span className="inline-flex items-center gap-1.5 bg-amber-500 hover:bg-amber-600 text-white text-sm font-medium rounded-lg px-4 py-2 transition-colors shadow-sm">
              <Clock className="w-4 h-4" />
              {pendingActions} pending
              <ArrowRight className="w-3.5 h-3.5" />
            </span>
          </Link>
        )}
      </div>

      {/* ── Truncation warning ── */}
      {itemsTruncated > 0 && (
        <div className="flex items-start gap-2.5 bg-amber-50 border border-amber-100 rounded-xl px-4 py-3">
          <AlertTriangle className="w-4 h-4 text-amber-500 mt-0.5 shrink-0" />
          <p className="text-sm text-amber-800">
            <strong>{itemsTruncated}</strong> additional item{itemsTruncated !== 1 ? "s were" : " was"} detected but not extracted — extraction limit reached.
            Increase <code className="bg-amber-100 px-1 py-0.5 rounded text-xs">MAX_EXTRACTED_ITEMS_PER_CAPTURE</code> to surface them.
          </p>
        </div>
      )}

      {/* ── Raw content ── */}
      {capture.raw_content && <RawContentCard content={capture.raw_content} />}

      {/* ── Extracted Items ── */}
      {items.length === 0 ? (
        <div className="text-center py-16 text-gray-400 border border-dashed border-gray-200 rounded-xl">
          <Zap className="w-8 h-8 mx-auto mb-3 opacity-30" />
          <p className="text-sm font-medium">No items extracted from this capture.</p>
          <p className="text-xs mt-1 text-gray-300">The pipeline ran but found nothing actionable.</p>
        </div>
      ) : (
        <div>
          <div className="flex items-center gap-2 mb-4">
            <h2 className="text-base font-semibold text-gray-900">Extracted Items</h2>
            <span className="inline-flex items-center justify-center min-w-[1.25rem] h-5 bg-gray-100 text-gray-600 text-xs font-semibold rounded-full px-1.5">
              {items.length}
            </span>
          </div>
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
