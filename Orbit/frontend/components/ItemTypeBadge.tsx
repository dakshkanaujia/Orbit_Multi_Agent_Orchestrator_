import {
  Calendar, Clock, CheckSquare, MessageCircle, Plane,
  Briefcase, Users, Bell, BookOpen, LucideIcon,
} from "lucide-react";
import { cn, ITEM_TYPE_COLORS } from "@/lib/utils";
import type { ItemType } from "@/lib/types";

const ITEM_TYPE_ICONS: Record<ItemType, LucideIcon> = {
  event: Calendar,
  deadline: Clock,
  task: CheckSquare,
  communication: MessageCircle,
  travel_interest: Plane,
  job_opportunity: Briefcase,
  meeting: Users,
  reminder: Bell,
  knowledge: BookOpen,
};

interface Props {
  type: ItemType;
  className?: string;
}

export function ItemTypeBadge({ type, className }: Props) {
  const Icon = ITEM_TYPE_ICONS[type] ?? BookOpen;
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-xs font-semibold",
        ITEM_TYPE_COLORS[type],
        className
      )}
    >
      <Icon className="w-3 h-3" />
      <span>{type.replace(/_/g, " ")}</span>
    </span>
  );
}
