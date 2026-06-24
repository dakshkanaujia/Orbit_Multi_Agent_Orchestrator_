"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { getDashboard } from "@/lib/api";
import { formatRelativeTime, truncate } from "@/lib/utils";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import {
  FileText, Image, Type, Clock, Database,
  LayoutGrid, AlertCircle, Inbox,
} from "lucide-react";
import type { DashboardData } from "@/lib/types";

// ── Pipeline strip ─────────────────────────────────────────────────────────

type StageStatus = "idle" | "active" | "done";
type PipelineStatuses = Record<string, StageStatus>;

const PIPELINE_NODES = [
  { key: "capture",       label: "Capture"    },
  { key: "understanding", label: "Understand" },
  { key: "intent",        label: "Extract"    },
  { key: "memory",        label: "Remember"   },
  { key: "planning",      label: "Plan"       },
];

const INITIAL_PIPELINE: PipelineStatuses = {
  capture: "idle", understanding: "idle", intent: "idle", memory: "idle", planning: "idle",
};

function PipelineStrip({ statuses }: { statuses: PipelineStatuses }) {
  const cells: React.ReactNode[] = [];
  PIPELINE_NODES.forEach((node, i) => {
    cells.push(
      <div key={node.key} className="flex flex-col items-center gap-1">
        <div className={`w-6 h-6 rounded-full flex items-center justify-center transition-all duration-300 ${
          statuses[node.key] === "done"   ? "bg-green-500" :
          statuses[node.key] === "active" ? "bg-indigo-500 animate-pulse" :
                                            "bg-gray-200"
        }`}>
          {statuses[node.key] === "done" && (
            <svg className="w-3.5 h-3.5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={3}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
            </svg>
          )}
        </div>
        <span className={`text-xs font-medium whitespace-nowrap transition-colors duration-300 ${
          statuses[node.key] === "done"   ? "text-green-600" :
          statuses[node.key] === "active" ? "text-indigo-600" :
                                            "text-gray-400"
        }`}>{node.label}</span>
      </div>
    );
    if (i < PIPELINE_NODES.length - 1) {
      cells.push(
        <div key={`line-${i}`} className="flex-1 flex items-start pt-3">
          <div className={`h-px w-full transition-colors duration-300 ${
            statuses[PIPELINE_NODES[i + 1].key] !== "idle" ? "bg-indigo-400" : "bg-gray-200"
          }`} />
        </div>
      );
    }
  });
  return <div className="flex items-start gap-0">{cells}</div>;
}

// ── Capture card helpers ───────────────────────────────────────────────────

const STRIPE: Record<string, string> = {
  text:  "border-indigo-400",
  pdf:   "border-red-400",
  image: "border-blue-400",
};

function ModalityIcon({ modality }: { modality: string }) {
  if (modality === "pdf")   return <FileText className="w-4 h-4 text-red-500" />;
  if (modality === "image") return <Image    className="w-4 h-4 text-blue-500" />;
  return <Type className="w-4 h-4 text-indigo-500" />;
}

// ── Skeleton ──────────────────────────────────────────────────────────────

function DashboardSkeleton() {
  return (
    <div className="space-y-8">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {[1, 2, 3].map((i) => (
          <Card key={i}>
            <CardContent className="pt-6">
              <Skeleton className="h-4 w-24 mb-3" />
              <Skeleton className="h-9 w-16" />
            </CardContent>
          </Card>
        ))}
      </div>
      <div className="space-y-3">
        {[1, 2, 3].map((i) => (
          <Skeleton key={i} className="h-20 w-full rounded-xl" />
        ))}
      </div>
    </div>
  );
}

// ── Page ──────────────────────────────────────────────────────────────────

export default function DashboardPage() {
  const router = useRouter();
  const [dashboard, setDashboard] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState(true);
  const [processing, setProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [textContent, setTextContent] = useState("");
  const [pipelineVisible, setPipelineVisible] = useState(false);
  const [pipelineStatuses, setPipelineStatuses] = useState<PipelineStatuses>(INITIAL_PIPELINE);

  useEffect(() => {
    getDashboard()
      .then(setDashboard)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  const handleProcess = async () => {
    if (!textContent.trim() || processing) return;
    setProcessing(true);
    setError(null);
    setPipelineVisible(true);
    setPipelineStatuses({ ...INITIAL_PIPELINE, capture: "active" });

    const BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

    try {
      const form = new FormData();
      form.append("content", textContent);
      form.append("source", "paste");

      const res = await fetch(`${BASE}/api/captures/stream`, { method: "POST", body: form });

      if (!res.ok) {
        throw new Error(`API ${res.status}: ${await res.text()}`);
      }

      // Response headers received → capture was accepted
      setPipelineStatuses((p) => ({ ...p, capture: "done", understanding: "active" }));

      const reader = res.body!.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      const NEXT_STAGE: Record<string, string | null> = {
        understanding: "intent",
        intent:        "memory",
        memory:        "planning",
        planning:      null,
      };

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split("\n\n");
        buffer = parts.pop() ?? "";

        for (const part of parts) {
          if (!part.trim()) continue;
          let eventType = "message";
          let eventData = "";
          for (const line of part.split("\n")) {
            if (line.startsWith("event: ")) eventType = line.slice(7).trim();
            else if (line.startsWith("data: ")) eventData = line.slice(6).trim();
          }
          if (!eventData) continue;

          const data = JSON.parse(eventData) as Record<string, unknown>;

          if (eventType === "agent") {
            const agent = data.agent as string;
            if (agent in NEXT_STAGE) {
              setPipelineStatuses((p) => {
                const next: PipelineStatuses = { ...p, [agent]: "done" };
                const nextKey = NEXT_STAGE[agent];
                if (nextKey) next[nextKey] = "active";
                return next;
              });
            }
          } else if (eventType === "done") {
            const captureId = data.capture_id as string;
            router.push(`/workspace/${captureId}`);
            return;
          } else if (eventType === "error") {
            throw new Error((data.error as string) || "Pipeline error");
          }
        }
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Processing failed");
      setPipelineVisible(false);
    } finally {
      setProcessing(false);
    }
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>
          <p className="text-gray-500 text-sm mt-1">Capture content and turn it into actions</p>
        </div>
        {dashboard && dashboard.pending_count > 0 && (
          <Link href="/approvals">
            <Badge className="bg-red-100 text-red-800 text-sm px-3 py-1 cursor-pointer hover:bg-red-200 flex items-center gap-1">
              <AlertCircle className="w-3.5 h-3.5" />
              {dashboard.pending_count} pending approval{dashboard.pending_count !== 1 ? "s" : ""}
            </Badge>
          </Link>
        )}
      </div>

      {/* Capture Input */}
      <Card>
        <CardHeader>
          <h2 className="font-semibold text-gray-900">New Capture</h2>
        </CardHeader>
        <CardContent className="space-y-4">
          <textarea
            className="w-full h-36 border border-gray-200 rounded-md p-3 text-sm focus:outline-none focus:ring-2 focus:ring-gray-300 resize-none disabled:opacity-60"
            placeholder="Paste meeting notes, job descriptions, conference info, emails…"
            value={textContent}
            onChange={(e) => setTextContent(e.target.value)}
            disabled={processing}
          />
          <Button onClick={handleProcess} disabled={processing || !textContent.trim()}>
            {processing ? "Processing…" : "Process"}
          </Button>

          {pipelineVisible && (
            <div className="pt-3 pb-1">
              <PipelineStrip statuses={pipelineStatuses} />
            </div>
          )}
        </CardContent>
      </Card>

      {error && (
        <div className="bg-red-50 border border-red-200 rounded-md p-4 text-sm text-red-700 flex items-center gap-2">
          <AlertCircle className="w-4 h-4 shrink-0" />
          {error}
        </div>
      )}

      {/* Stats */}
      {loading ? (
        <DashboardSkeleton />
      ) : dashboard ? (
        <>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Card>
              <CardContent className="pt-5 flex items-start gap-4">
                <div className="p-2 bg-red-100 rounded-lg">
                  <Clock className="w-5 h-5 text-red-600" />
                </div>
                <div>
                  <p className="text-sm text-gray-500">Pending Actions</p>
                  <p className="text-3xl font-bold text-red-600 mt-0.5">{dashboard.pending_count}</p>
                </div>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="pt-5 flex items-start gap-4">
                <div className="p-2 bg-indigo-100 rounded-lg">
                  <Database className="w-5 h-5 text-indigo-600" />
                </div>
                <div>
                  <p className="text-sm text-gray-500">Total Captures</p>
                  <p className="text-3xl font-bold text-gray-900 mt-0.5">{dashboard.recent_captures.length}</p>
                </div>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="pt-5 flex items-start gap-4">
                <div className="p-2 bg-green-100 rounded-lg">
                  <LayoutGrid className="w-5 h-5 text-green-600" />
                </div>
                <div>
                  <p className="text-sm text-gray-500">Item Types</p>
                  <div className="flex flex-wrap gap-1 mt-1.5">
                    {Object.entries(dashboard.item_type_breakdown).map(([type, count]) => (
                      <span key={type} className="text-xs bg-gray-100 rounded-full px-2 py-0.5">
                        {type.replace(/_/g, " ")} {count}
                      </span>
                    ))}
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Recent Captures */}
          {dashboard.recent_captures.length > 0 ? (
            <div>
              <h2 className="text-lg font-semibold text-gray-900 mb-4">Recent Captures</h2>
              <div className="space-y-3">
                {dashboard.recent_captures.map((cap) => {
                  const pendingCount = (cap as Record<string, unknown>).pending_action_count as number ?? 0;
                  const actionCount  = (cap as Record<string, unknown>).action_count  as number ?? 0;
                  const itemCount    = (cap as Record<string, unknown>).item_count    as number ?? 0;
                  const preview = truncate(cap.raw_content, 80);
                  const stripe = STRIPE[cap.modality] ?? "border-gray-300";

                  return (
                    <Link key={cap.id} href={`/workspace/${cap.id}`}>
                      <Card className={`hover:shadow-md transition-shadow cursor-pointer border-l-4 ${stripe}`}>
                        <CardContent className="pt-3 pb-3">
                          <div className="flex items-start justify-between gap-4">
                            <div className="flex items-start gap-3 min-w-0">
                              <div className="pt-0.5 shrink-0">
                                <ModalityIcon modality={cap.modality} />
                              </div>
                              <div className="min-w-0">
                                <p className="text-sm font-medium text-gray-900 capitalize">
                                  {cap.modality} · {cap.source}
                                </p>
                                {preview && (
                                  <p className="text-xs text-gray-500 mt-0.5 truncate">{preview}</p>
                                )}
                                <div className="flex items-center gap-1.5 mt-1.5">
                                  <span className="text-xs bg-gray-100 text-gray-600 rounded-full px-2 py-0.5">
                                    {itemCount} item{itemCount !== 1 ? "s" : ""}
                                  </span>
                                  {actionCount > 0 && (
                                    <span className={`text-xs rounded-full px-2 py-0.5 ${
                                      pendingCount > 0
                                        ? "bg-amber-100 text-amber-700"
                                        : "bg-green-100 text-green-700"
                                    }`}>
                                      {pendingCount > 0 ? `${pendingCount} pending` : `${actionCount} done`}
                                    </span>
                                  )}
                                </div>
                              </div>
                            </div>
                            <p className="text-xs text-gray-400 shrink-0 mt-0.5">
                              {formatRelativeTime(cap.created_at)}
                            </p>
                          </div>
                        </CardContent>
                      </Card>
                    </Link>
                  );
                })}
              </div>
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center py-16 text-center">
              <div className="p-4 bg-gray-100 rounded-full mb-4">
                <Inbox className="w-8 h-8 text-gray-400" />
              </div>
              <h3 className="font-semibold text-gray-900 mb-1">No captures yet</h3>
              <p className="text-sm text-gray-500 max-w-xs">
                Paste some text above to get started.
              </p>
            </div>
          )}
        </>
      ) : null}
    </div>
  );
}
