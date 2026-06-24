"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import { LayoutDashboard, Inbox, CheckSquare, Search, Sparkles } from "lucide-react";

const NAV = [
  { href: "/dashboard", label: "Dashboard", icon: LayoutDashboard },
  { href: "/hub",       label: "Hub",       icon: Sparkles },
  { href: "/items",     label: "Items",     icon: Inbox },
  { href: "/approvals", label: "Approvals", icon: CheckSquare },
  { href: "/search",    label: "Search",    icon: Search },
];

export function NavLinks() {
  const pathname = usePathname();
  return (
    <div className="flex items-center gap-1">
      {NAV.map(({ href, label, icon: Icon }) => {
        const active = pathname === href || pathname.startsWith(href + "/");
        return (
          <Link
            key={href}
            href={href}
            className={cn(
              "flex items-center gap-1.5 px-3 py-2 text-sm font-medium rounded-md transition-colors",
              active
                ? "text-gray-900 bg-gray-100"
                : "text-gray-500 hover:text-gray-900 hover:bg-gray-50"
            )}
          >
            <Icon className="w-4 h-4" />
            <span>{label}</span>
          </Link>
        );
      })}
    </div>
  );
}
