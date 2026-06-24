import { cn, scoreColor } from "@/lib/utils";

type ProgressColor = "auto" | "blue" | "amber";

interface ProgressProps {
  value: number;
  className?: string;
  label?: string;
  showScore?: boolean;
  color?: ProgressColor;
}

const COLOR_CLASSES: Record<ProgressColor, (v: number) => string> = {
  auto:  (v) => scoreColor(v),
  blue:  ()  => "bg-blue-500",
  amber: ()  => "bg-amber-500",
};

export function Progress({ value, className, label, showScore = false, color = "auto" }: ProgressProps) {
  const pct = Math.round(value * 100);
  const barClass = COLOR_CLASSES[color](value);
  return (
    <div className={cn("space-y-1", className)}>
      {label && (
        <div className="flex justify-between text-xs text-gray-500">
          <span>{label}</span>
          {showScore && <span className="font-medium tabular-nums">{pct}%</span>}
        </div>
      )}
      <div className="w-full bg-gray-100 rounded-full h-2 overflow-hidden">
        <div
          className={cn("h-full rounded-full transition-all duration-300", barClass)}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}
