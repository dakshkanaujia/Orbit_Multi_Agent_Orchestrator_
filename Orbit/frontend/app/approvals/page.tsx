"use client";

import { useEffect, useState } from "react";
import { getPendingActions, approveAction, rejectAction, editAction } from "@/lib/api";
import { ItemTypeBadge } from "@/components/ItemTypeBadge";
import { Progress } from "@/components/ui/progress";
import { Skeleton } from "@/components/ui/skeleton";
import { Card, CardContent, CardFooter, CardHeader } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { formatDateTime } from "@/lib/utils";
import {
  Mail, CalendarDays, Bell, Hash,
  CheckCircle, XCircle, RefreshCw,
} from "lucide-react";
import type { ActionWithContext, ApproveResponse } from "@/lib/types";

function actionTypeIcon(actionType: string) {
  if (actionType.startsWith("calendar")) return <CalendarDays className="w-4 h-4" />;
  if (actionType.startsWith("gmail")) return <Mail className="w-4 h-4" />;
  if (actionType.includes("reminder")) return <Bell className="w-4 h-4" />;
  return <Hash className="w-4 h-4" />;
}

function PayloadView({ payload, label }: { payload: Record<string, unknown>; label?: string }) {
  return (
    <div>
      {label && <p className="text-xs text-gray-500 mb-1 font-medium">{label}</p>}
      <div className="bg-gray-50 rounded p-3 font-mono text-xs text-gray-700 overflow-x-auto max-h-40">
        {Object.entries(payload).map(([k, v]) => (
          <div key={k}>
            <span className="text-indigo-600">{k}</span>:{" "}
            <span>{typeof v === "object" ? JSON.stringify(v) : String(v ?? "—")}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function EditPayloadDialog({
  initial,
  onSave,
  onCancel,
}: {
  initial: Record<string, unknown>;
  onSave: (payload: Record<string, unknown>) => void;
  onCancel: () => void;
}) {
  const [raw, setRaw] = useState(JSON.stringify(initial, null, 2));
  const [parseError, setParseError] = useState<string | null>(null);

  const handleSave = () => {
    try {
      const parsed = JSON.parse(raw);
      setParseError(null);
      onSave(parsed);
    } catch {
      setParseError("Invalid JSON");
    }
  };

  return (
    <div className="fixed inset-0 bg-black/30 z-50 flex items-center justify-center p-4">
      <div className="bg-white rounded-lg shadow-2xl w-full max-w-lg">
        <div className="flex items-center justify-between px-5 py-4 border-b">
          <h3 className="font-semibold text-gray-900">Edit Payload</h3>
          <button onClick={onCancel} className="text-gray-400 hover:text-gray-600 text-xl leading-none">×</button>
        </div>
        <div className="p-5">
          <textarea
            className="w-full h-52 font-mono text-xs border border-gray-200 rounded p-3 focus:outline-none focus:ring-2 focus:ring-gray-300 resize-none"
            value={raw}
            onChange={(e) => setRaw(e.target.value)}
          />
          {parseError && <p className="text-red-600 text-xs mt-1">{parseError}</p>}
        </div>
        <div className="flex gap-2 justify-end px-5 pb-4">
          <Button size="sm" variant="outline" onClick={onCancel}>Cancel</Button>
          <Button size="sm" onClick={handleSave}>Save &amp; Approve</Button>
        </div>
      </div>
    </div>
  );
}

function ActionCard({
  item: { action, extracted_item },
  onDecision,
}: {
  item: ActionWithContext;
  onDecision: () => void;
}) {
  const [loading, setLoading] = useState<"approve" | "reject" | "edit" | null>(null);
  const [result, setResult] = useState<{ payload?: Record<string, unknown>; exec?: Record<string, unknown>; raw?: string; ok?: boolean } | null>(null);
  const [showEdit, setShowEdit] = useState(false);

  const handle = async (fn: () => Promise<unknown>, label: "approve" | "reject" | "edit") => {
    setLoading(label);
    try {
      const res = await fn() as ApproveResponse | Record<string, unknown>;
      if (res && ("decision" in res || "execution_result" in res)) {
        const typed = res as { decision?: Record<string, unknown>; execution_result?: Record<string, unknown> };
        setResult({
          payload: typed.decision?.final_payload as Record<string, unknown> | undefined,
          exec: (typed.execution_result ?? typed.decision?.execution_result) as Record<string, unknown> | undefined,
          ok: (typed.execution_result as Record<string, unknown> | undefined)?.status !== "failed",
        });
      } else {
        setResult({ raw: JSON.stringify(res, null, 2) });
      }
      onDecision();
    } catch (e: unknown) {
      setResult({ raw: `Error: ${e instanceof Error ? e.message : String(e)}`, ok: false });
    } finally {
      setLoading(null);
    }
  };

  if (result !== null) {
    return (
      <Card className="border-gray-200 opacity-80">
        <CardContent className="space-y-2 pt-4">
          <div className="flex items-center gap-2 mb-2">
            {result.ok !== false ? (
              <CheckCircle className="w-4 h-4 text-green-600" />
            ) : (
              <XCircle className="w-4 h-4 text-red-500" />
            )}
            <span className="text-xs font-medium text-gray-700">
              {result.ok !== false ? "Executed" : "Failed"}
            </span>
          </div>
          {result.raw ? (
            <p className="text-xs text-gray-500 font-mono whitespace-pre-wrap max-h-24 overflow-y-auto">{result.raw}</p>
          ) : (
            <>
              {result.payload && <PayloadView payload={result.payload} label="Final Payload" />}
              {result.exec && (
                <PayloadView
                  payload={result.exec}
                  label={result.ok === false ? "Execution Error" : "Execution Result"}
                />
              )}
            </>
          )}
        </CardContent>
      </Card>
    );
  }

  return (
    <>
      <Card>
        <CardHeader>
          <div className="flex items-start justify-between gap-2">
            <div className="flex items-start gap-2">
              <div className="p-1.5 bg-gray-100 rounded-md text-gray-600 mt-0.5">
                {actionTypeIcon(action.action_type)}
              </div>
              <div>
                <span className="inline-flex items-center bg-gray-100 text-gray-700 text-xs font-semibold rounded-full px-3 py-1">
                  {action.action_type}
                </span>
                <p className="text-xs text-gray-400 mt-1">{formatDateTime(action.created_at)}</p>
              </div>
            </div>
            {extracted_item && <ItemTypeBadge type={extracted_item.item_type} />}
          </div>
        </CardHeader>
        <CardContent className="space-y-3">
          <PayloadView payload={action.payload} />
          {extracted_item && (
            <div className="bg-gray-50 rounded p-3 space-y-2 text-xs">
              <p className="font-medium text-gray-700">{extracted_item.title}</p>
              <Progress value={extracted_item.confidence_score} label="Confidence" showScore />
              <Progress value={extracted_item.urgency_score} label="Urgency" showScore />
            </div>
          )}
        </CardContent>
        <CardFooter>
          <div className="flex gap-2 w-full">
            <Button
              size="sm"
              variant="success"
              className="flex-1"
              disabled={!!loading}
              onClick={() => handle(() => approveAction(action.id), "approve")}
            >
              <CheckCircle className="w-3.5 h-3.5 mr-1" />
              {loading === "approve" ? "…" : "Approve"}
            </Button>
            <Button
              size="sm"
              variant="destructive"
              className="flex-1"
              disabled={!!loading}
              onClick={() => handle(() => rejectAction(action.id), "reject")}
            >
              <XCircle className="w-3.5 h-3.5 mr-1" />
              {loading === "reject" ? "…" : "Reject"}
            </Button>
            <Button
              size="sm"
              variant="outline"
              className="flex-1"
              disabled={!!loading}
              onClick={() => setShowEdit(true)}
            >
              Edit
            </Button>
          </div>
        </CardFooter>
      </Card>

      {showEdit && (
        <EditPayloadDialog
          initial={action.payload}
          onCancel={() => setShowEdit(false)}
          onSave={(edited) => {
            setShowEdit(false);
            handle(() => editAction(action.id, edited), "edit");
          }}
        />
      )}
    </>
  );
}

export default function ApprovalsPage() {
  const [pending, setPending] = useState<ActionWithContext[]>([]);
  const [loading, setLoading] = useState(true);

  const load = () => {
    setLoading(true);
    getPendingActions()
      .then(setPending)
      .finally(() => setLoading(false));
  };

  useEffect(() => { load(); }, []);

  const sorted = [...pending].sort(
    (a, b) => (b.extracted_item?.urgency_score ?? 0) - (a.extracted_item?.urgency_score ?? 0)
  );

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-gray-900">Approval Center</h1>
        <div className="flex items-center gap-3">
          {pending.length > 0 && (
            <span className="bg-red-100 text-red-800 text-sm font-medium rounded-full px-3 py-1">
              {pending.length} pending
            </span>
          )}
          <Button size="sm" variant="outline" onClick={load} className="flex items-center gap-1.5">
            <RefreshCw className="w-3.5 h-3.5" />
            Refresh
          </Button>
        </div>
      </div>

      {loading ? (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {[1, 2, 3, 4].map((i) => (
            <Skeleton key={i} className="h-52 w-full rounded-xl" />
          ))}
        </div>
      ) : sorted.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-20 text-center">
          <div className="p-4 bg-green-100 rounded-full mb-4">
            <CheckCircle className="w-8 h-8 text-green-600" />
          </div>
          <h3 className="font-semibold text-gray-900 mb-1">All clear</h3>
          <p className="text-sm text-gray-500">No pending actions waiting for approval.</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {sorted.map((item) => (
            <ActionCard key={item.action.id} item={item} onDecision={load} />
          ))}
        </div>
      )}
    </div>
  );
}
