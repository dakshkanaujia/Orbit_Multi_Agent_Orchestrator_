"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { getHub } from "@/lib/api";
import { Skeleton } from "@/components/ui/skeleton";
import { ItemTypeBadge } from "@/components/ItemTypeBadge";
import {
  Sparkles, CalendarDays, CheckSquare, BookOpen,
  AlertCircle, Clock, CalendarClock, Target,
  RefreshCw, ArrowRight,
} from "lucide-react";
import type { HubData, HubItem, HubGroup, HubPriority, HubStats, PriorityLevel } from "@/lib/types";

// ── Urgency badge ───────────────────────────────────────────────────────────

function UrgencyBadge({ score }: { score: number }) {
  if (score >= 0.7)
    return <span className="text-xs font-semibold text-red-600 bg-red-50 border border-red-200 px-1.5 py-0.5 rounded-md whitespace-nowrap">High</span>;
  if (score >= 0.4)
    return <span className="text-xs font-semibold text-amber-600 bg-amber-50 border border-amber-200 px-1.5 py-0.5 rounded-md whitespace-nowrap">Med</span>;
  return <span className="text-xs font-semibold text-gray-400 bg-gray-50 border border-gray-200 px-1.5 py-0.5 rounded-md whitespace-nowrap">Low</span>;
}

// ── Priority item ───────────────────────────────────────────────────────────

const PRIORITY_CFG: Record<PriorityLevel, { dot: string; row: string; label: string; labelCls: string }> = {
  critical: { dot: "bg-red-500",    row: "bg-red-50 border-red-200",    label: "Critical", labelCls: "text-red-500"    },
  high:     { dot: "bg-orange-400", row: "bg-orange-50 border-orange-200", label: "High",  labelCls: "text-orange-500" },
  medium:   { dot: "bg-amber-400",  row: "bg-amber-50 border-amber-100",  label: "Medium", labelCls: "text-amber-500"  },
  low:      { dot: "bg-gray-300",   row: "bg-gray-50 border-gray-100",    label: "Low",    labelCls: "text-gray-400"   },
};

function PriorityRow({ p }: { p: HubPriority }) {
  const c = PRIORITY_CFG[p.level];
  return (
    <div className={`flex items-center gap-3 px-3 py-2 rounded-lg border ${c.row}`}>
      <div className={`w-2 h-2 rounded-full flex-shrink-0 ${c.dot}`} />
      <p className="text-sm text-gray-800 flex-1">{p.text}</p>
      <span className={`text-xs font-semibold ${c.labelCls}`}>{c.label}</span>
    </div>
  );
}

// ── Stat chip ───────────────────────────────────────────────────────────────

function StatChip({
  icon: Icon, label, value, cls,
}: { icon: React.ElementType; label: string; value: number; cls: string }) {
  return (
    <div className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border text-xs font-semibold ${cls}`}>
      <Icon className="w-3.5 h-3.5" />
      <span className="text-sm font-bold">{value}</span>
      <span className="opacity-70 font-medium">{label}</span>
    </div>
  );
}

// ── Left-border color helpers ───────────────────────────────────────────────

const GROUP_BORDER: Record<HubGroup, string> = {
  overdue:   "border-l-red-500",
  today:     "border-l-orange-400",
  this_week: "border-l-blue-400",
  later:     "border-l-slate-300",
  no_date:   "border-l-gray-200",
};

const TYPE_BORDER: Record<string, string> = {
  task:            "border-l-amber-400",
  communication:   "border-l-green-400",
  job_opportunity: "border-l-indigo-400",
  travel_interest: "border-l-sky-400",
};

const GROUP_LABEL: Record<HubGroup, { text: string; cls: string }> = {
  overdue:   { text: "Overdue",   cls: "text-red-600 bg-red-50 border-red-200"       },
  today:     { text: "Today",     cls: "text-orange-600 bg-orange-50 border-orange-200" },
  this_week: { text: "This week", cls: "text-blue-600 bg-blue-50 border-blue-200"    },
  later:     { text: "Later",     cls: "text-slate-500 bg-slate-50 border-slate-200" },
  no_date:   { text: "No date",   cls: "text-gray-400 bg-gray-50 border-gray-100"    },
};

// ── Compact item card ───────────────────────────────────────────────────────

function HubCard({ item, borderClass, showGroup = false }: {
  item: HubItem;
  borderClass: string;
  showGroup?: boolean;
}) {
  const group = item.group as HubGroup | undefined;
  const gl = group ? GROUP_LABEL[group] : null;

  return (
    <Link href={`/workspace/${item.capture_id}`} className="block group">
      <div className={`p-3 rounded-lg border border-gray-200 border-l-4 bg-white hover:shadow-sm hover:border-gray-300 transition-all ${borderClass}`}>
        {/* Row 1: type badge + group badge + urgency */}
        <div className="flex items-center gap-1.5 mb-1.5">
          <ItemTypeBadge type={item.item_type} />
          {showGroup && gl && (
            <span className={`text-xs font-medium px-1.5 py-0.5 rounded border leading-none ${gl.cls}`}>
              {gl.text}
            </span>
          )}
          <div className="ml-auto">
            <UrgencyBadge score={item.urgency_score} />
          </div>
        </div>

        {/* Row 2: title */}
        <p className="text-sm font-semibold text-gray-900 truncate group-hover:text-indigo-600 transition-colors">
          {item.title}
        </p>

        {/* Row 3: description */}
        {item.description && (
          <p className="text-xs text-gray-400 line-clamp-1 mt-0.5">{item.description}</p>
        )}

        {/* Row 4: meta */}
        <div className="flex items-center gap-2 mt-1.5 text-xs text-gray-400">
          <span>{new Date(item.created_at).toLocaleDateString()}</span>
          {item.deadline && (
            <>
              <span>·</span>
              <CalendarDays className="w-3 h-3 flex-shrink-0" />
              <span>{new Date(item.deadline).toLocaleDateString()}</span>
            </>
          )}
          {item.pending_actions > 0 && (
            <span className="ml-auto bg-amber-100 text-amber-700 px-1.5 py-0.5 rounded-full font-semibold">
              {item.pending_actions} pending
            </span>
          )}
        </div>
      </div>
    </Link>
  );
}

// ── Column panel ────────────────────────────────────────────────────────────

function Panel({
  title, icon: Icon, count, headerCls, countCls, emptyMsg, children,
}: {
  title: string;
  icon: React.ElementType;
  count: number;
  headerCls: string;
  countCls: string;
  emptyMsg: string;
  children: React.ReactNode;
}) {
  return (
    <div className="flex flex-col rounded-xl border border-gray-200 overflow-hidden bg-white">
      <div className={`flex items-center gap-2 px-4 py-3 border-b border-gray-100 ${headerCls}`}>
        <Icon className="w-4 h-4" />
        <span className="text-sm font-semibold">{title}</span>
        <span className={`ml-auto text-xs font-bold px-2 py-0.5 rounded-full ${countCls}`}>{count}</span>
      </div>
      <div className="flex-1 overflow-y-auto max-h-[62vh] p-3 space-y-2 bg-gray-50/40">
        {count === 0
          ? <p className="text-xs text-gray-400 text-center py-12">{emptyMsg}</p>
          : children}
      </div>
    </div>
  );
}

// ── Skeleton ────────────────────────────────────────────────────────────────

function HubSkeleton() {
  return (
    <div className="space-y-5">
      <div className="flex gap-2 flex-wrap">
        {[1, 2, 3, 4].map(i => <Skeleton key={i} className="h-8 w-24 rounded-lg" />)}
      </div>
      <Skeleton className="h-36 w-full rounded-xl" />
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {[1, 2, 3].map(i => (
          <div key={i} className="rounded-xl border overflow-hidden">
            <Skeleton className="h-11 w-full rounded-none" />
            <div className="p-3 space-y-2 bg-gray-50/40">
              {[1, 2, 3].map(j => <Skeleton key={j} className="h-[72px] w-full rounded-lg" />)}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Page ────────────────────────────────────────────────────────────────────

export default function HubPage() {
  const [data, setData] = useState<HubData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = () => {
    setLoading(true);
    setError(null);
    getHub()
      .then(setData)
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  };

  useEffect(() => { load(); }, []);

  return (
    <div className="space-y-5">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Sparkles className="w-5 h-5 text-indigo-500" />
          <h1 className="text-2xl font-bold text-gray-900">Intelligence Hub</h1>
        </div>
        <button
          onClick={load}
          disabled={loading}
          className="flex items-center gap-1.5 text-sm text-gray-500 hover:text-gray-800 px-3 py-1.5 rounded-lg border border-gray-200 hover:bg-gray-50 transition-colors disabled:opacity-40"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? "animate-spin" : ""}`} />
          Refresh
        </button>
      </div>

      {loading ? (
        <HubSkeleton />
      ) : error ? (
        <div className="flex flex-col items-center justify-center py-20 text-center">
          <AlertCircle className="w-8 h-8 text-red-400 mb-3" />
          <p className="text-sm text-red-600">{error}</p>
          <button onClick={load} className="mt-3 text-sm text-indigo-600 hover:underline">Retry</button>
        </div>
      ) : data && (
        <>
          {/* ── Stats strip ─────────────────────────────────────────────── */}
          <div className="flex flex-wrap gap-2">
            {data.stats.overdue > 0 && (
              <StatChip icon={AlertCircle} label="Overdue" value={data.stats.overdue}
                cls="text-red-600 bg-red-50 border-red-200" />
            )}
            {data.stats.today > 0 && (
              <StatChip icon={Clock} label="Today" value={data.stats.today}
                cls="text-orange-600 bg-orange-50 border-orange-200" />
            )}
            {data.stats.this_week > 0 && (
              <StatChip icon={CalendarClock} label="This week" value={data.stats.this_week}
                cls="text-blue-600 bg-blue-50 border-blue-200" />
            )}
            <StatChip icon={Target} label="Tasks" value={data.stats.total_tasks}
              cls="text-amber-600 bg-amber-50 border-amber-200" />
            <StatChip icon={BookOpen} label="Learning" value={data.stats.knowledge_items}
              cls="text-purple-600 bg-purple-50 border-purple-200" />
          </div>

          {/* ── AI Priorities ────────────────────────────────────────────── */}
          {(data.priorities.length > 0 || data.summary) && (
            <div className="rounded-xl border border-indigo-100 bg-gradient-to-br from-indigo-50/50 to-purple-50/30 overflow-hidden">
              <div className="flex items-center gap-2 px-4 py-2.5 border-b border-indigo-100/70">
                <div className="p-1 bg-indigo-100 rounded">
                  <Sparkles className="w-3.5 h-3.5 text-indigo-600" />
                </div>
                <span className="text-xs font-bold text-indigo-600 uppercase tracking-wider">
                  Today's Priorities
                </span>
              </div>
              <div className="p-4 space-y-2">
                {data.priorities.map((p, i) => (
                  <PriorityRow key={i} p={p} />
                ))}
                {data.summary && (
                  <p className="text-xs text-gray-500 mt-1 px-1 pt-1 border-t border-indigo-100/50">
                    {data.summary}
                  </p>
                )}
              </div>
            </div>
          )}

          {/* ── Three panels ─────────────────────────────────────────────── */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            {/* Upcoming */}
            <Panel
              title="Upcoming"
              icon={CalendarDays}
              count={data.upcoming.length}
              headerCls="bg-blue-50/80 text-blue-800"
              countCls="bg-blue-100 text-blue-700"
              emptyMsg="No upcoming events or meetings."
            >
              {data.upcoming.map(item => (
                <HubCard
                  key={item.id}
                  item={item}
                  borderClass={GROUP_BORDER[item.group as HubGroup] ?? "border-l-gray-200"}
                  showGroup
                />
              ))}
            </Panel>

            {/* Tasks & Goals */}
            <Panel
              title="Tasks & Goals"
              icon={CheckSquare}
              count={data.tasks.length}
              headerCls="bg-amber-50/80 text-amber-800"
              countCls="bg-amber-100 text-amber-700"
              emptyMsg="No open tasks."
            >
              {data.tasks.map(item => (
                <HubCard
                  key={item.id}
                  item={item}
                  borderClass={TYPE_BORDER[item.item_type] ?? "border-l-gray-300"}
                />
              ))}
            </Panel>

            {/* Learning */}
            <Panel
              title="Learning"
              icon={BookOpen}
              count={data.knowledge.length}
              headerCls="bg-purple-50/80 text-purple-800"
              countCls="bg-purple-100 text-purple-700"
              emptyMsg="No knowledge captured yet."
            >
              {data.knowledge.map(item => (
                <HubCard
                  key={item.id}
                  item={item}
                  borderClass="border-l-purple-400"
                />
              ))}
            </Panel>
          </div>

          {/* Footer link */}
          <div className="text-center pt-1">
            <Link
              href="/items"
              className="inline-flex items-center gap-1.5 text-sm text-gray-400 hover:text-indigo-600 transition-colors"
            >
              View all items <ArrowRight className="w-3.5 h-3.5" />
            </Link>
          </div>
        </>
      )}
    </div>
  );
}
