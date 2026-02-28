"use client";

import { useEffect } from "react";
import { SessionSidebar } from "@/components/chat/session-sidebar";
import { MessageThread } from "@/components/chat/message-thread";
import { ChatInput } from "@/components/chat/chat-input";
import { TracePanel } from "@/components/chat/trace-panel";
import { useChatStore } from "@/stores/chat-store";
import { useChat } from "@/lib/use-chat";

export default function ChatPage() {
  const { loadSessions } = useChat();
  const sidebarOpen = useChatStore((s) => s.sidebarOpen);
  const traceOpen = useChatStore((s) => s.traceOpen);

  useEffect(() => {
    loadSessions();
  }, [loadSessions]);

  return (
    <div className="flex h-full overflow-hidden">
      {sidebarOpen && (
        <div className="w-72 shrink-0 border-r border-border overflow-hidden flex flex-col">
          <SessionSidebar />
        </div>
      )}

      <div className="flex flex-1 flex-col min-w-0 h-full">
        <div className="flex-1 min-h-0 overflow-hidden">
          <MessageThread />
        </div>
        <div className="shrink-0">
          <ChatInput />
        </div>
      </div>

      {traceOpen && (
        <div className="w-80 shrink-0 border-l border-border overflow-hidden flex flex-col">
          <TracePanel />
        </div>
      )}
    </div>
  );
}
