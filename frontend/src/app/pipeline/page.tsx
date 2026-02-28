"use client";

import { useEffect, useState } from "react";
import { Activity, Loader2, FileText, Clock } from "lucide-react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { GlassCard, GradientValue } from "@/components/ui/glass-card";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8765";

interface FolderHealth {
  total_files: number;
  files_today: number;
  newest_file?: { name: string; path: string; modified: string };
}

export default function PipelinePage() {
  const [folders, setFolders] = useState<Record<string, FolderHealth>>({});
  const [loading, setLoading] = useState(true);
  const [ts, setTs] = useState("");

  useEffect(() => {
    fetch(`${API_BASE}/api/pipeline-health`)
      .then((r) => r.json())
      .then((d) => { setFolders(d.folders || {}); setTs(d.timestamp || ""); })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="flex items-center justify-center h-full"><Loader2 className="h-6 w-6 animate-spin text-muted-foreground" /></div>;

  return (
    <ScrollArea className="h-full">
      <div className="mx-auto max-w-6xl px-6 py-5 space-y-5">
        <div className="flex items-center gap-3">
          <Activity className="h-6 w-6 text-primary" />
          <h1 className="text-2xl font-semibold">Pipeline Health</h1>
          {ts && <span className="text-xs text-muted-foreground ml-auto">{new Date(ts).toLocaleTimeString()}</span>}
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {Object.entries(folders).map(([name, info]) => {
            const healthy = info.files_today > 0;
            return (
              <GlassCard key={name} label={name}>
                <div className="flex items-baseline gap-3 mb-3">
                  <GradientValue className="text-3xl">{info.total_files.toLocaleString()}</GradientValue>
                  <span className="text-xs text-muted-foreground">total files</span>
                </div>
                <div className="flex items-center justify-between text-sm mb-2">
                  <span className="text-muted-foreground">Today</span>
                  <span className={healthy ? "text-green-400 font-semibold" : "text-yellow-400"}>{info.files_today}</span>
                </div>
                {info.newest_file && (
                  <div className="flex items-start gap-2 text-xs text-muted-foreground mt-2 pt-2 border-t border-[rgba(255,255,255,.04)]">
                    <FileText className="h-3.5 w-3.5 shrink-0 mt-0.5" />
                    <div className="min-w-0">
                      <div className="truncate">{info.newest_file.name}</div>
                      <div className="flex items-center gap-1 mt-0.5"><Clock className="h-3 w-3" /> {info.newest_file.modified?.slice(0, 16).replace("T", " ")}</div>
                    </div>
                  </div>
                )}
                <div className="mt-3">
                  <span className={`rounded-full px-2.5 py-0.5 text-[0.7rem] font-semibold ${healthy ? "bg-green-400/15 text-green-400" : "bg-yellow-400/15 text-yellow-400"}`}>
                    {healthy ? "Active" : "Idle"}
                  </span>
                </div>
              </GlassCard>
            );
          })}
        </div>
      </div>
    </ScrollArea>
  );
}
