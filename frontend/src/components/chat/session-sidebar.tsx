"use client";

import { useState, useCallback } from "react";
import {
  Plus,
  Search,
  Pin,
  PinOff,
  Pencil,
  Trash2,
  PanelLeftClose,
  MoreHorizontal,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { useChatStore } from "@/stores/chat-store";
import { useChat } from "@/lib/use-chat";
import { cn } from "@/lib/utils";
import type { Session } from "@/lib/api";

function groupByDate(sessions: Session[]) {
  const now = new Date();
  const today = now.toDateString();
  const yesterday = new Date(now.getTime() - 86400000).toDateString();
  const weekAgo = new Date(now.getTime() - 7 * 86400000);

  const groups: { label: string; sessions: Session[] }[] = [
    { label: "Pinned", sessions: [] },
    { label: "Today", sessions: [] },
    { label: "Yesterday", sessions: [] },
    { label: "This Week", sessions: [] },
    { label: "Older", sessions: [] },
  ];

  for (const s of sessions) {
    if (s.pinned) {
      groups[0].sessions.push(s);
      continue;
    }
    const d = new Date(s.updated_at);
    if (d.toDateString() === today) groups[1].sessions.push(s);
    else if (d.toDateString() === yesterday) groups[2].sessions.push(s);
    else if (d > weekAgo) groups[3].sessions.push(s);
    else groups[4].sessions.push(s);
  }

  return groups.filter((g) => g.sessions.length > 0);
}

export function SessionSidebar() {
  const sessions = useChatStore((s) => s.sessions);
  const activeId = useChatStore((s) => s.activeSessionId);
  const toggleSidebar = useChatStore((s) => s.toggleSidebar);
  const { selectSession, newChat, removeSession, pinSession, renameSession } =
    useChat();

  const [search, setSearch] = useState("");
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editTitle, setEditTitle] = useState("");

  const filtered = search
    ? sessions.filter((s) =>
        s.title.toLowerCase().includes(search.toLowerCase())
      )
    : sessions;

  const groups = groupByDate(filtered);

  const startRename = useCallback((s: Session) => {
    setEditingId(s.id);
    setEditTitle(s.title);
  }, []);

  const commitRename = useCallback(async () => {
    if (editingId && editTitle.trim()) {
      await renameSession(editingId, editTitle.trim());
    }
    setEditingId(null);
  }, [editingId, editTitle, renameSession]);

  return (
    <div className="flex h-full flex-col bg-card">
      <div className="flex items-center gap-2 border-b border-border p-3">
        <Button
          variant="outline"
          size="sm"
          className="flex-1 justify-start gap-2"
          onClick={newChat}
        >
          <Plus className="h-4 w-4" />
          New Chat
        </Button>
        <Button variant="ghost" size="icon" className="h-8 w-8" onClick={toggleSidebar}>
          <PanelLeftClose className="h-4 w-4" />
        </Button>
      </div>

      <div className="px-3 py-2">
        <div className="relative">
          <Search className="absolute left-2.5 top-2.5 h-3.5 w-3.5 text-muted-foreground" />
          <Input
            placeholder="Search chats..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="h-8 pl-8 text-sm"
          />
        </div>
      </div>

      <ScrollArea className="flex-1">
        <div className="px-2 pb-4">
          {groups.map((group) => (
            <div key={group.label} className="mt-3">
              <div className="px-2 pb-1 text-xs font-medium text-muted-foreground">
                {group.label}
              </div>
              {group.sessions.map((s) => (
                <div
                  key={s.id}
                  className={cn(
                    "group flex items-center rounded-md px-2 py-1.5 text-sm cursor-pointer transition-colors",
                    s.id === activeId
                      ? "bg-accent text-accent-foreground"
                      : "hover:bg-accent/50"
                  )}
                  onClick={() => selectSession(s.id)}
                >
                  {editingId === s.id ? (
                    <input
                      className="flex-1 bg-transparent text-sm outline-none"
                      value={editTitle}
                      onChange={(e) => setEditTitle(e.target.value)}
                      onBlur={commitRename}
                      onKeyDown={(e) => {
                        if (e.key === "Enter") commitRename();
                        if (e.key === "Escape") setEditingId(null);
                      }}
                      autoFocus
                      onClick={(e) => e.stopPropagation()}
                    />
                  ) : (
                    <span className="flex-1 truncate">{s.title}</span>
                  )}

                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-6 w-6 opacity-0 group-hover:opacity-100"
                        onClick={(e) => e.stopPropagation()}
                      >
                        <MoreHorizontal className="h-3.5 w-3.5" />
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="end" className="w-40">
                      <DropdownMenuItem onClick={() => startRename(s)}>
                        <Pencil className="mr-2 h-3.5 w-3.5" />
                        Rename
                      </DropdownMenuItem>
                      <DropdownMenuItem
                        onClick={() => pinSession(s.id, !s.pinned)}
                      >
                        {s.pinned ? (
                          <PinOff className="mr-2 h-3.5 w-3.5" />
                        ) : (
                          <Pin className="mr-2 h-3.5 w-3.5" />
                        )}
                        {s.pinned ? "Unpin" : "Pin"}
                      </DropdownMenuItem>
                      <DropdownMenuItem
                        className="text-destructive"
                        onClick={() => removeSession(s.id)}
                      >
                        <Trash2 className="mr-2 h-3.5 w-3.5" />
                        Delete
                      </DropdownMenuItem>
                    </DropdownMenuContent>
                  </DropdownMenu>
                </div>
              ))}
            </div>
          ))}

          {groups.length === 0 && (
            <div className="px-2 py-8 text-center text-sm text-muted-foreground">
              {search ? "No matching chats" : "No conversations yet"}
            </div>
          )}
        </div>
      </ScrollArea>
    </div>
  );
}
