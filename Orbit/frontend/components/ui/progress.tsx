import { cn, scoreColor } from "@/lib/utils";

interface ProgressProps {
  value: number;
  className?: string;
  label?: string;
  showScore?: boolean;
}

export function Progress({ value, className, label, showScore = false }: ProgressProps) {
  const pct = Math.round(value * 100);
  return (
    <div className={cn("space-y-1", className)}>
      {label && (
        <div className="flex justify-between text-xs text-gray-500">
          <span>{label}</span>
          {showScore && <span>{pct}%</span>}
        </div>
      )}
      <div className="w-full bg-gray-200 rounded-full h-2 overflow-hidden">
        <div
          className={cn("h-full rounded-full transition-all", scoreColor(value))}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}
