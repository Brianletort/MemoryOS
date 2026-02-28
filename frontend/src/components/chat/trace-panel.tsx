"use client";

import { useEffect, useRef } from "react";
import {
  Wrench,
  ArrowRight,
  Activity,
  Loader2,
  CheckCircle2,
  Search,
  FileText,
  Brain,
  Globe,
} from "lucide-react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useChatStore } from "@/stores/chat-store";
import { cn } from "@/lib/utils";

const STATUS_ICONS: Record<string, typeof Search> = {
  "Loading context...": FileText,
  "Searching memory vault...": Search,
  "Analyzing query...": Brain,
  "Generating response...": Brain,
  "Searching the web...": Globe,
  "Building presentation...": FileText,
  "Running skill...": Wrench,
};

export function TracePanel() {
  const traceEvents = useChatStore((s) => s.traceEvents);
  const isStreaming = useChatStore((s) => s.isStreaming);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [traceEvents]);

  return (
    <div className="flex h-full flex-col bg-card">
      <div className="flex items-center gap-2 border-b border-border px-4 py-3">
        <Activity className="h-4 w-4 text-muted-foreground" />
        <span className="text-sm font-medium">Agent Trace</span>
        {isStreaming && (
          <span className="ml-auto flex items-center gap-1.5 text-xs text-primary">
            <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-primary" />
            Running
          </span>
        )}
      </div>

      <ScrollArea className="flex-1">
        <div className="space-y-1 p-3">
          {traceEvents.length === 0 && (
            <p className="text-center text-xs text-muted-foreground py-8">
              Agent activity will appear here during execution.
            </p>
          )}

          {traceEvents.map((event, i) => {
            if (event.type === "status") {
              const isLast = i === traceEvents.length - 1;
              return (
                <div key={i} className="flex items-center gap-2 py-1.5 px-2 rounded-md">
                  {isLast && isStreaming ? (
                    <Loader2 className="h-3.5 w-3.5 animate-spin text-primary shrink-0" />
                  ) : (
                    <CheckCircle2 className="h-3.5 w-3.5 text-green-500/70 shrink-0" />
                  )}
                  <span className={cn(
                    "text-xs",
                    isLast && isStreaming ? "text-foreground font-medium" : "text-muted-foreground"
                  )}>
                    {event.data}
                  </span>
                </div>
              );
            }

            if (event.type === "tool_call") {
              return (
                <div key={i} className="mt-2 rounded-lg border border-[rgba(108,124,255,.15)] bg-[rgba(108,124,255,.05)] p-2.5">
                  <div className="flex items-center gap-1.5">
                    <Wrench className="h-3.5 w-3.5 text-primary shrink-0" />
                    <span className="text-xs font-medium text-primary">
                      {event.name}
                    </span>
                  </div>
                  <pre className="mt-1.5 max-h-24 overflow-auto rounded bg-background/50 p-2 text-[10px] leading-relaxed text-muted-foreground">
                    {event.data.length > 300 ? event.data.slice(0, 300) + "..." : event.data}
                  </pre>
                </div>
              );
            }

            if (event.type === "tool_result") {
              return (
                <div key={i} className="rounded-lg border border-green-500/15 bg-green-500/5 p-2.5">
                  <div className="flex items-center gap-1.5">
                    <ArrowRight className="h-3.5 w-3.5 text-green-500 shrink-0" />
                    <span className="text-xs font-medium text-green-400">
                      {event.name} result
                    </span>
                  </div>
                  <pre className="mt-1.5 max-h-24 overflow-auto rounded bg-background/50 p-2 text-[10px] leading-relaxed text-muted-foreground">
                    {event.data.length > 300 ? event.data.slice(0, 300) + "..." : event.data}
                  </pre>
                </div>
              );
            }

            return null;
          })}

          <div ref={bottomRef} />
        </div>
      </ScrollArea>
    </div>
  );
}
