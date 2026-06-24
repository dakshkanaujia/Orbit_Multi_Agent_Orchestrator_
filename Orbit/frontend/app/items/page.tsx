"use client";

import { useEffect, useState } from "react";
import { listItems } from "@/lib/api";
import { formatDate, formatDateTime, truncate } from "@/lib/utils";
import { ItemTypeBadge } from "@/components/ItemTypeBadge";
import { Progress } from "@/components/ui/progress";
import { Skeleton } from "@/components/ui/skeleton";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  Filter, ArrowUpDown, Inbox, CalendarDays,
  CheckCircle, XCircle, Clock, X,
} from "lucide-react";
import type { ExtractedItem, ItemType, Action } from "@/lib/types";
import Link from "next/link";

const ITEM_TYPES: ItemType[] = [
  "event", "deadline", "task", "communication",
  "travel_interest", "job_opportunity", "meeting", "reminder", "knowledge",
];

function ActionStatusChip({ status }: { status: Action["status"] }) {
  const map: Record<string, { cls: string; Icon: React.ElementType }> = {
    pending:  { cls: "bg-yellow-100 text-yellow-800", Icon: Clock },
    approved: { cls: "bg-blue-100 text-blue-800",    Icon: CheckCircle },
    executed: { cls: "bg-green-100 text-green-800",  Icon: CheckCircle },
    rejected: { cls: "bg-gray-100 text-gray-500",    Icon: XCircle },
    failed:   { cls: "bg-red-100 text-red-700",      Icon: XCircle },
  };
  const { cls, Icon } = map[status] ?? map.pending;
  return (
    <span className={`inline-flex items-center gap-1 text-xs font-medium rounded-full px-2 py-0.5 ${cls}`}>
      <Icon className="w-3 h-3" />
      {status}
    </span>
  );
}

function ItemRow({ item, onSelect }: { item: ExtractedItem; onSelect: () => void }) {
  return (
    <Card className="hover:shadow-md transition-shadow cursor-pointer" onClick={onSelect}>
      <CardContent className="pt-4 pb-4">
        <div className="flex items-start gap-4">
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 flex-wrap">
              <ItemTypeBadge type={item.item_type} />
              {item.deadline && (
                <span className="inline-flex items-center gap-1 text-xs text-amber-700">
                  <CalendarDays className="w-3 h-3" />{formatDate(item.deadline)}
                </span>
              )}
            </div>
            <p className="font-medium text-gray-900 mt-1.5 text-sm">{item.title}</p>
            <p className="text-xs text-gray-500 mt-0.5">{truncate(item.description, 100)}</p>
          </div>
          <div className="shrink-0 w-28 space-y-1.5">
            <Progress value={item.confidence_score} label="Confidence" showScore />
            <Progress value={item.urgency_score} label="Urgency" showScore />
          </div>
          <div className="text-xs text-gray-400 shrink-0">{formatDateTime(item.created_at)}</div>
        </div>
      </CardContent>
    </Card>
  );
}

function ItemDetailPanel({
  item,
  onClose,
}: {
  item: ExtractedItem & { actions?: Action[] };
  onClose: () => void;
}) {
  const actions = item.actions ?? [];
  return (
    <div className="fixed inset-y-0 right-0 w-full max-w-md bg-white shadow-2xl z-50 flex flex-col">
      <div className="flex items-center justify-between px-5 py-4 border-b">
        <div className="flex items-center gap-2 min-w-0">
          <ItemTypeBadge type={item.item_type} />
          <h2 className="font-semibold text-gray-900 text-sm truncate">{item.title}</h2>
        </div>
        <button onClick={onClose} className="text-gray-400 hover:text-gray-600 shrink-0">
          <X className="w-5 h-5" />
        </button>
      </div>
      <div className="overflow-y-auto flex-1 p-5 space-y-4">
        {item.description && <p className="text-sm text-gray-600">{item.description}</p>}
        {item.deadline && (
          <p className="text-sm text-amber-700 flex items-center gap-1.5">
            <CalendarDays className="w-4 h-4" />Deadline: {formatDate(item.deadline)}
          </p>
        )}
        <Progress value={item.confidence_score} label="Confidence" showScore />
        <Progress value={item.urgency_score} label="Urgency" showScore />

        {Object.entries(item.entities || {})
          .filter(([, v]) => Array.isArray(v) && (v as unknown[]).length > 0)
          .map(([k, v]) => (
            <div key={k}>
              <p className="text-xs text-gray-500 uppercase tracking-wide mb-1">{k}</p>
              <div className="flex flex-wrap gap-1">
                {(v as string[]).map((e, i) => (
                  <span key={i} className="text-xs bg-gray-100 rounded px-2 py-0.5">{e}</span>
                ))}
              </div>
            </div>
          ))}

        {actions.length > 0 && (
          <div>
            <p className="text-xs text-gray-500 uppercase tracking-wide mb-2">Actions</p>
            <div className="space-y-2">
              {actions.map((a) => (
                <div key={a.id} className="bg-gray-50 rounded p-2.5 text-xs flex justify-between items-center">
                  <code className="text-gray-700 text-xs">{a.action_type}</code>
                  <ActionStatusChip status={a.status} />
                </div>
              ))}
            </div>
          </div>
        )}

        <Link href={`/workspace/${item.capture_id}`} className="block">
          <Button variant="outline" size="sm" className="w-full">View in Workspace</Button>
        </Link>
      </div>
    </div>
  );
}

export default function ItemsPage() {
  const [items, setItems] = useState<ExtractedItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedItem, setSelectedItem] = useState<ExtractedItem | null>(null);
  const [filter, setFilter] = useState<{ item_type?: string; min_urgency?: number }>({});
  const [sortUrgency, setSortUrgency] = useState<"desc" | "asc">("desc");

  const load = () => {
    setLoading(true);
    listItems({ ...filter, limit: 50 })
      .then(setItems)
      .finally(() => setLoading(false));
  };

  useEffect(() => { load(); }, [filter]);

  const sorted = [...items].sort((a, b) =>
    sortUrgency === "desc"
      ? b.urgency_score - a.urgency_score
      : a.urgency_score - b.urgency_score
  );

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-gray-900">Extracted Items</h1>
        <span className="text-sm text-gray-500">{items.length} items</span>
      </div>

      {/* Filter Bar */}
      <div className="flex flex-wrap gap-3 items-center">
        <div className="relative flex items-center">
          <Filter className="absolute left-2.5 w-3.5 h-3.5 text-gray-400 pointer-events-none" />
          <select
            className="border border-gray-200 rounded-md pl-8 pr-3 py-1.5 text-sm focus:outline-none appearance-none bg-white"
            value={filter.item_type ?? ""}
            onChange={(e) => setFilter((f) => ({ ...f, item_type: e.target.value || undefined }))}
          >
            <option value="">All types</option>
            {ITEM_TYPES.map((t) => <option key={t} value={t}>{t.replace(/_/g, " ")}</option>)}
          </select>
        </div>
        <select
          className="border border-gray-200 rounded-md px-3 py-1.5 text-sm focus:outline-none bg-white"
          value={filter.min_urgency ?? ""}
          onChange={(e) => setFilter((f) => ({ ...f, min_urgency: e.target.value ? Number(e.target.value) : undefined }))}
        >
          <option value="">Any urgency</option>
          <option value="0.7">High (≥70%)</option>
          <option value="0.4">Medium (≥40%)</option>
        </select>
        <Button
          size="sm"
          variant="outline"
          onClick={() => setSortUrgency((s) => s === "desc" ? "asc" : "desc")}
          className="flex items-center gap-1.5"
        >
          <ArrowUpDown className="w-3.5 h-3.5" />
          Urgency {sortUrgency === "desc" ? "↓" : "↑"}
        </Button>
        {(filter.item_type || filter.min_urgency) && (
          <Button size="sm" variant="ghost" onClick={() => setFilter({})} className="flex items-center gap-1">
            <X className="w-3.5 h-3.5" />Clear
          </Button>
        )}
      </div>

      {loading ? (
        <div className="space-y-3">
          {[1, 2, 3, 4].map((i) => <Skeleton key={i} className="h-20 w-full rounded-xl" />)}
        </div>
      ) : sorted.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-20 text-center">
          <div className="p-4 bg-gray-100 rounded-full mb-4">
            <Inbox className="w-8 h-8 text-gray-400" />
          </div>
          <h3 className="font-semibold text-gray-900 mb-1">No items yet</h3>
          <p className="text-sm text-gray-500">
            {filter.item_type || filter.min_urgency
              ? "No items match the current filters."
              : "Capture some content on the Dashboard to get started."}
          </p>
        </div>
      ) : (
        <div className="space-y-3">
          {sorted.map((item) => (
            <ItemRow key={item.id} item={item} onSelect={() => setSelectedItem(item)} />
          ))}
        </div>
      )}

      {selectedItem && (
        <>
          <div className="fixed inset-0 bg-black/20 z-40" onClick={() => setSelectedItem(null)} />
          <ItemDetailPanel item={selectedItem} onClose={() => setSelectedItem(null)} />
        </>
      )}
    </div>
  );
}
