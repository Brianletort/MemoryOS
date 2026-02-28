"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  MessageSquare,
  LayoutDashboard,
  Sun,
  Briefcase,
  Sparkles,
  BookOpen,
  Settings,
  Activity,
  FolderOpen,
  Terminal,
  HelpCircle,
  MoreHorizontal,
} from "lucide-react";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { cn } from "@/lib/utils";

const PRIMARY_NAV = [
  { href: "/", label: "Chat", icon: MessageSquare },
  { href: "/dashboard", label: "Dashboard", icon: LayoutDashboard },
  { href: "/myday", label: "My Day", icon: Sun },
  { href: "/work", label: "My Work", icon: Briefcase },
  { href: "/skills", label: "Skills", icon: Sparkles },
  { href: "/knowledge", label: "Knowledge", icon: BookOpen },
];

const MORE_NAV = [
  { href: "/pipeline", label: "Pipeline", icon: Activity },
  { href: "/browser", label: "File Browser", icon: FolderOpen },
  { href: "/logs", label: "Logs", icon: Terminal },
  { href: "/settings", label: "Settings", icon: Settings },
  { href: "/help", label: "Help", icon: HelpCircle },
];

export function NavBar() {
  const pathname = usePathname();
  const isMoreActive = MORE_NAV.some((item) =>
    item.href === "/" ? pathname === "/" : pathname.startsWith(item.href)
  );

  return (
    <nav className="flex h-12 shrink-0 items-center border-b border-border bg-card px-4">
      <Link href="/" className="mr-6 flex items-center gap-2 font-semibold">
        <div className="flex h-7 w-7 items-center justify-center rounded-md bg-primary text-primary-foreground text-xs font-bold">
          M
        </div>
        <span className="hidden sm:inline">MemoryOS</span>
      </Link>

      <div className="flex items-center gap-1">
        {PRIMARY_NAV.map((item) => {
          const isActive =
            item.href === "/"
              ? pathname === "/"
              : pathname.startsWith(item.href);
          return (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                "flex items-center gap-1.5 rounded-md px-3 py-1.5 text-sm transition-colors",
                isActive
                  ? "bg-accent text-accent-foreground font-medium"
                  : "text-muted-foreground hover:bg-accent/50 hover:text-foreground"
              )}
            >
              <item.icon className="h-4 w-4" />
              <span className="hidden md:inline">{item.label}</span>
            </Link>
          );
        })}

        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <button
              className={cn(
                "flex items-center gap-1.5 rounded-md px-3 py-1.5 text-sm transition-colors",
                isMoreActive
                  ? "bg-accent text-accent-foreground font-medium"
                  : "text-muted-foreground hover:bg-accent/50 hover:text-foreground"
              )}
            >
              <MoreHorizontal className="h-4 w-4" />
              <span className="hidden md:inline">More</span>
            </button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="start" className="w-48">
            {MORE_NAV.map((item) => {
              const isActive = pathname.startsWith(item.href);
              return (
                <DropdownMenuItem key={item.href} asChild>
                  <Link
                    href={item.href}
                    className={cn(
                      "flex items-center gap-2 cursor-pointer",
                      isActive && "text-primary font-medium"
                    )}
                  >
                    <item.icon className="h-4 w-4" />
                    {item.label}
                  </Link>
                </DropdownMenuItem>
              );
            })}
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
    </nav>
  );
}
