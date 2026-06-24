import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";
import type { ItemType } from "./types";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export const ITEM_TYPE_COLORS: Record<ItemType, string> = {
  event: "bg-blue-100 text-blue-800",
  deadline: "bg-red-100 text-red-800",
  task: "bg-yellow-100 text-yellow-800",
  communication: "bg-purple-100 text-purple-800",
  travel_interest: "bg-teal-100 text-teal-800",
  job_opportunity: "bg-green-100 text-green-800",
  meeting: "bg-indigo-100 text-indigo-800",
  reminder: "bg-orange-100 text-orange-800",
  knowledge: "bg-gray-100 text-gray-800",
};

export const ITEM_TYPE_ICON_NAMES: Record<ItemType, string> = {
  event: "Calendar",
  deadline: "Clock",
  task: "CheckSquare",
  communication: "MessageCircle",
  travel_interest: "Plane",
  job_opportunity: "Briefcase",
  meeting: "Users",
  reminder: "Bell",
  knowledge: "BookOpen",
};

// Keep for backwards compat in places that still use string emoji
export const ITEM_TYPE_ICONS: Record<ItemType, string> = {
  event: "Cal",
  deadline: "Clk",
  task: "Chk",
  communication: "Msg",
  travel_interest: "Pln",
  job_opportunity: "Brf",
  meeting: "Usr",
  reminder: "Bll",
  knowledge: "Bk",
};

export function formatDate(iso: string | null | undefined): string {
  if (!iso) return "—";
  return new Date(iso).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}

export function formatDateTime(iso: string | null | undefined): string {
  if (!iso) return "—";
  return new Date(iso).toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function scoreColor(score: number): string {
  if (score >= 0.7) return "bg-green-500";
  if (score >= 0.4) return "bg-yellow-500";
  return "bg-red-500";
}

export function truncate(str: string | null | undefined, max = 120): string {
  if (!str) return "";
  return str.length > max ? str.slice(0, max) + "…" : str;
}

export function formatRelativeTime(iso: string | null | undefined): string {
  if (!iso) return "—";
  const diff = Date.now() - new Date(iso).getTime();
  const minutes = Math.floor(diff / 60_000);
  const hours = Math.floor(diff / 3_600_000);
  const days = Math.floor(diff / 86_400_000);
  if (minutes < 1) return "just now";
  if (minutes < 60) return `${minutes}m ago`;
  if (hours < 24) return `${hours}h ago`;
  if (days < 7) return `${days}d ago`;
  return formatDate(iso);
}
