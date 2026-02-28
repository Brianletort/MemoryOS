"use client";

import { useState, useCallback } from "react";
import { Play, Loader2, Activity } from "lucide-react";
import { Button } from "@/components/ui/button";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8765";

interface ExtractorCardsProps {
  extractors: Record<string, Record<string, unknown>>;
  onRefresh: () => void;
}

const EXTRACTOR_NAMES: Record<string, string> = {
  screenpipe: "Screenpipe",
  outlook: "Outlook",
  onedrive: "OneDrive",
  "mail-app": "Mail.app",
  "calendar-app": "Calendar.app",
};

export function ExtractorCards({ extractors, onRefresh }: ExtractorCardsProps) {
  const [running, setRunning] = useState<string | null>(null);

  const runExtractor = useCallback(async (name: string) => {
    setRunning(name);
    try {
      await fetch(`${API}/api/run/${name}`, { method: "POST" });
      setTimeout(onRefresh, 2000);
    } catch {}
    finally { setRunning(null); }
  }, [onRefresh]);

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
      {Object.entries(extractors).map(([name, info]) => {
        const label = EXTRACTOR_NAMES[name] || name;
        const stats = Object.entries(info)
          .filter(([k]) => !k.startsWith("_"))
          .slice(0, 5);

        return (
          <div key={name} className="rounded-xl border border-border bg-card p-4 space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Activity className="h-4 w-4 text-primary" />
                <span className="font-medium text-sm">{label}</span>
              </div>
              <Button
                variant="outline"
                size="sm"
                className="h-7 gap-1.5 text-xs"
                onClick={() => runExtractor(name)}
                disabled={running === name}
              >
                {running === name ? (
                  <Loader2 className="h-3 w-3 animate-spin" />
                ) : (
                  <Play className="h-3 w-3" />
                )}
                Run
              </Button>
            </div>
            <div className="space-y-1">
              {stats.map(([k, v]) => (
                <div key={k} className="flex items-center justify-between text-xs">
                  <span className="text-muted-foreground">{k.replace(/_/g, " ")}</span>
                  <span className="font-mono">{String(v)}</span>
                </div>
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}
