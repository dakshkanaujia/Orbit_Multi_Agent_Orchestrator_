"use client";

import { useState } from "react";
import { search } from "@/lib/api";
import { ItemTypeBadge } from "@/components/ItemTypeBadge";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { formatDate, truncate, ITEM_TYPE_COLORS } from "@/lib/utils";
import { Search, SearchX, FileText, Image, Type, Filter } from "lucide-react";
import type { SearchResult, ItemType } from "@/lib/types";
import Link from "next/link";

const ITEM_TYPES: ItemType[] = [
  "event", "deadline", "task", "communication",
  "travel_interest", "job_opportunity", "meeting", "reminder", "knowledge",
];

function ScoreBar({ score }: { score: number }) {
  const pct = Math.round(score * 100);
  const color = pct >= 70 ? "bg-green-500" : pct >= 40 ? "bg-yellow-500" : "bg-gray-400";
  return (
    <div className="flex items-center gap-2">
      <div className="w-16 bg-gray-200 rounded-full h-1.5 overflow-hidden">
        <div className={`h-full rounded-full ${color}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-xs text-gray-500 tabular-nums">{pct}%</span>
    </div>
  );
}

function ModalityIcon({ modality }: { modality: string }) {
  if (modality === "pdf") return <FileText className="w-4 h-4 text-red-500" />;
  if (modality === "image") return <Image className="w-4 h-4 text-blue-500" />;
  return <Type className="w-4 h-4 text-gray-500" />;
}

function SearchResultCard({ result }: { result: SearchResult }) {
  if (result.type === "item" && result.item) {
    const { item, parent_capture, actions, semantic_score } = result;
    const borderColor = ITEM_TYPE_COLORS[item.item_type]?.includes("blue") ? "border-l-blue-400"
      : ITEM_TYPE_COLORS[item.item_type]?.includes("red") ? "border-l-red-400"
      : ITEM_TYPE_COLORS[item.item_type]?.includes("yellow") ? "border-l-yellow-400"
      : ITEM_TYPE_COLORS[item.item_type]?.includes("green") ? "border-l-green-400"
      : ITEM_TYPE_COLORS[item.item_type]?.includes("purple") ? "border-l-purple-400"
      : ITEM_TYPE_COLORS[item.item_type]?.includes("indigo") ? "border-l-indigo-400"
      : ITEM_TYPE_COLORS[item.item_type]?.includes("orange") ? "border-l-orange-400"
      : "border-l-gray-300";

    return (
      <Card className={`hover:shadow-md transition-shadow border-l-4 ${borderColor}`}>
        <CardContent className="pt-4 pb-4">
          <div className="flex items-start justify-between gap-3">
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 flex-wrap">
                <ItemTypeBadge type={item.item_type} />
                {item.deadline && (
                  <span className="text-xs text-amber-700">{formatDate(item.deadline)}</span>
                )}
              </div>
              <p className="font-semibold text-gray-900 mt-1.5">{item.title}</p>
              <p className="text-sm text-gray-500 mt-0.5">{truncate(item.description, 120)}</p>
              {parent_capture && (
                <p className="text-xs text-gray-400 mt-1">
                  From {parent_capture.modality} capture ·{" "}
                  <Link href={`/workspace/${parent_capture.id}`} className="text-indigo-500 hover:underline">
                    View workspace
                  </Link>
                </p>
              )}
            </div>
            <div className="shrink-0 flex flex-col items-end gap-1">
              <ScoreBar score={semantic_score} />
              <span className="text-xs text-gray-400">{actions?.length ?? 0} actions</span>
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (result.type === "capture" && result.capture) {
    const { capture, semantic_score } = result;
    return (
      <Card className="hover:shadow-md transition-shadow border-l-4 border-l-gray-300">
        <CardContent className="pt-4 pb-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-gray-100 rounded-lg">
                <ModalityIcon modality={capture.modality} />
              </div>
              <div>
                <p className="text-sm font-medium text-gray-900 capitalize">
                  {capture.modality} capture · {capture.source}
                </p>
                <p className="text-xs text-gray-500 mt-0.5">{truncate(capture.raw_content, 80)}</p>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <ScoreBar score={semantic_score} />
              <Link href={`/workspace/${capture.id}`}>
                <Button size="sm" variant="outline">View</Button>
              </Link>
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  return null;
}

export default function SearchPage() {
  const [query, setQuery] = useState("");
  const [filterType, setFilterType] = useState<string>("");
  const [minUrgency, setMinUrgency] = useState<number | undefined>();
  const [results, setResults] = useState<SearchResult[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSearch = async () => {
    if (!query.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const res = await search({ query, item_type: filterType || undefined, min_urgency: minUrgency, limit: 20 });
      setResults(res);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Search failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-gray-900">Search Vault</h1>

      {/* Search Bar */}
      <div className="flex gap-2">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400 pointer-events-none" />
          <input
            className="w-full border border-gray-200 rounded-md pl-10 pr-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-gray-300"
            placeholder="Search your captures and extracted items…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSearch()}
          />
        </div>
        <Button onClick={handleSearch} disabled={loading || !query.trim()}>
          {loading ? "Searching…" : "Search"}
        </Button>
      </div>

      {/* Filters */}
      <div className="flex flex-wrap gap-3 items-center">
        <div className="relative flex items-center">
          <Filter className="absolute left-2.5 w-3.5 h-3.5 text-gray-400 pointer-events-none" />
          <select
            className="border border-gray-200 rounded-md pl-8 pr-3 py-1.5 text-sm focus:outline-none appearance-none bg-white"
            value={filterType}
            onChange={(e) => setFilterType(e.target.value)}
          >
            <option value="">All types</option>
            {ITEM_TYPES.map((t) => <option key={t} value={t}>{t.replace(/_/g, " ")}</option>)}
          </select>
        </div>
        <select
          className="border border-gray-200 rounded-md px-3 py-1.5 text-sm focus:outline-none bg-white"
          value={minUrgency ?? ""}
          onChange={(e) => setMinUrgency(e.target.value ? Number(e.target.value) : undefined)}
        >
          <option value="">Any urgency</option>
          <option value="0.7">High (≥70%)</option>
          <option value="0.4">Medium (≥40%)</option>
        </select>
      </div>

      {error && <div className="bg-red-50 border border-red-200 rounded p-3 text-sm text-red-700">{error}</div>}

      {loading ? (
        <div className="space-y-3">
          {[1, 2, 3].map((i) => <Skeleton key={i} className="h-24 w-full rounded-xl" />)}
        </div>
      ) : results !== null ? (
        results.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-20 text-center">
            <div className="p-4 bg-gray-100 rounded-full mb-4">
              <SearchX className="w-8 h-8 text-gray-400" />
            </div>
            <h3 className="font-semibold text-gray-900 mb-1">No results</h3>
            <p className="text-sm text-gray-500">No matches found for &ldquo;{query}&rdquo;.</p>
          </div>
        ) : (
          <div className="space-y-3">
            <p className="text-sm text-gray-500">{results.length} results</p>
            {results.map((r, i) => <SearchResultCard key={i} result={r} />)}
          </div>
        )
      ) : null}
    </div>
  );
}
