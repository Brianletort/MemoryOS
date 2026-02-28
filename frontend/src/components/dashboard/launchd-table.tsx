"use client";

import { useState, useCallback } from "react";
import { Play, Square, RotateCcw, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8765";

interface AgentInfo {
  pid: number | null;
  last_exit_code: number;
  loaded: boolean;
  healthy: boolean;
}

interface LaunchdTableProps {
  agents: Record<string, AgentInfo>;
  onRefresh: () => void;
}

export function LaunchdTable({ agents, onRefresh }: LaunchdTableProps) {
  const [acting, setActing] = useState<string | null>(null);

  const doAction = useCallback(async (name: string, action: string) => {
    setActing(`${name}:${action}`);
    try {
      await fetch(`${API}/api/agent/${name}/${action}`, { method: "POST" });
      setTimeout(onRefresh, 1000);
    } catch {}
    finally { setActing(null); }
  }, [onRefresh]);

  const entries = Object.entries(agents).sort(([a], [b]) => a.localeCompare(b));

  return (
    <div className="overflow-x-auto rounded-lg border border-border">
      <table className="w-full text-sm">
        <thead className="bg-muted/30">
          <tr>
            <th className="px-4 py-2.5 text-left text-xs font-semibold text-muted-foreground">Agent</th>
            <th className="px-4 py-2.5 text-center text-xs font-semibold text-muted-foreground">Status</th>
            <th className="px-4 py-2.5 text-center text-xs font-semibold text-muted-foreground">PID</th>
            <th className="px-4 py-2.5 text-center text-xs font-semibold text-muted-foreground">Exit</th>
            <th className="px-4 py-2.5 text-right text-xs font-semibold text-muted-foreground">Actions</th>
          </tr>
        </thead>
        <tbody>
          {entries.map(([name, info]) => {
            const shortName = name.replace("com.memoryos.", "");
            const isActing = acting?.startsWith(name);
            return (
              <tr key={name} className="border-t border-border hover:bg-muted/20 transition-colors">
                <td className="px-4 py-2 font-mono text-xs">{shortName}</td>
                <td className="px-4 py-2 text-center">
                  <span className={`inline-block rounded-full px-2 py-0.5 text-[10px] font-semibold ${
                    info.healthy ? "bg-green-400/15 text-green-400" :
                    info.loaded ? "bg-yellow-400/15 text-yellow-400" :
                    "bg-muted text-muted-foreground"
                  }`}>
                    {info.healthy ? "Healthy" : info.loaded ? "Loaded" : "Stopped"}
                  </span>
                </td>
                <td className="px-4 py-2 text-center font-mono text-xs text-muted-foreground">
                  {info.pid || "—"}
                </td>
                <td className="px-4 py-2 text-center font-mono text-xs text-muted-foreground">
                  {info.last_exit_code}
                </td>
                <td className="px-4 py-2 text-right">
                  <div className="flex items-center justify-end gap-1">
                    {isActing ? (
                      <Loader2 className="h-3.5 w-3.5 animate-spin text-muted-foreground" />
                    ) : (
                      <>
                        <Button variant="ghost" size="icon" className="h-7 w-7" onClick={() => doAction(name, "start")} title="Start">
                          <Play className="h-3 w-3" />
                        </Button>
                        <Button variant="ghost" size="icon" className="h-7 w-7" onClick={() => doAction(name, "stop")} title="Stop">
                          <Square className="h-3 w-3" />
                        </Button>
                        <Button variant="ghost" size="icon" className="h-7 w-7" onClick={() => doAction(name, "restart")} title="Restart">
                          <RotateCcw className="h-3 w-3" />
                        </Button>
                      </>
                    )}
                  </div>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
