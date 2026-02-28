"use client";

import { useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import { Loader2, Download, ExternalLink } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent } from "@/components/ui/dialog";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8765";

interface ReportViewerProps {
  skillName: string | null;
  date: string | null;
  onClose: () => void;
}

interface ReportJson {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  [key: string]: any;
}

function MetricCard({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="rounded-lg border border-border bg-card p-3 text-center">
      <div className="text-2xl font-bold text-primary">{value}</div>
      <div className="text-xs text-muted-foreground mt-1">{label}</div>
    </div>
  );
}

function renderJsonReport(skill: string, data: ReportJson) {
  if (skill === "morning-brief") {
    return (
      <div className="space-y-4">
        <div className="grid grid-cols-3 gap-3">
          <MetricCard label="Day Score" value={String(data.day_score ?? "—")} />
          <MetricCard label="Meetings" value={String(data.meeting_count ?? "—")} />
          <MetricCard label="Focus Hours" value={String(data.day_composition?.focus_percent ?? "—") + "%"} />
        </div>
        {data.day_summary && <p className="text-sm text-muted-foreground">{String(data.day_summary)}</p>}
        {Array.isArray(data.quick_wins) && data.quick_wins.length > 0 && (
          <div>
            <h3 className="text-sm font-semibold mb-2">Quick Wins</h3>
            <ul className="space-y-1">
              {data.quick_wins.map((w: { action?: string } | string, i: number) => (
                <li key={i} className="text-sm text-muted-foreground">
                  {typeof w === "string" ? w : w.action}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    );
  }

  if (skill === "plan-my-week") {
    return (
      <div className="space-y-4">
        {data.week_score != null && (
          <MetricCard label="Week Score" value={String(data.week_score)} />
        )}
        {Array.isArray(data.days) && (
          <div className="grid grid-cols-5 gap-2">
            {data.days.map((d: { day_name?: string; meeting_hours?: number; focus_hours?: number; capacity_percent?: number }, i: number) => (
              <div key={i} className="rounded-lg border border-border p-2 text-center text-xs">
                <div className="font-semibold">{d.day_name}</div>
                <div className="text-muted-foreground mt-1">{d.meeting_hours ?? 0}h mtg</div>
                <div className="text-muted-foreground">{d.focus_hours ?? 0}h focus</div>
                <div className="mt-1 text-primary font-mono">{d.capacity_percent ?? 0}%</div>
              </div>
            ))}
          </div>
        )}
      </div>
    );
  }

  return (
    <pre className="text-xs overflow-x-auto rounded-lg bg-muted p-4 max-h-[400px] overflow-y-auto">
      {JSON.stringify(data, null, 2)}
    </pre>
  );
}

export function ReportViewer({ skillName, date, onClose }: ReportViewerProps) {
  const [markdown, setMarkdown] = useState<string>("");
  const [jsonData, setJsonData] = useState<ReportJson | null>(null);
  const [loading, setLoading] = useState(false);
  const [viewMode, setViewMode] = useState<"rich" | "markdown">("rich");

  useEffect(() => {
    if (!skillName || !date) return;
    setLoading(true);
    setJsonData(null);
    setMarkdown("");

    fetch(`${API}/api/agents/reports/${skillName}/${date}/json`)
      .then((r) => r.ok ? r.json() : null)
      .then((d) => { if (d) setJsonData(d); })
      .catch(() => {});

    fetch(`${API}/api/agents/reports/${skillName}/${date}`)
      .then((r) => r.ok ? r.json() : null)
      .then((d) => { if (d?.content) setMarkdown(d.content); })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [skillName, date]);

  const legacyUrl = `${API}/report/${skillName}/${date}`;

  return (
    <Dialog open={!!skillName && !!date} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="max-w-3xl max-h-[85vh] overflow-y-auto">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold capitalize">{skillName} — {date}</h2>
          <div className="flex items-center gap-2">
            <Button variant="ghost" size="sm" className="text-xs gap-1" onClick={() => setViewMode(viewMode === "rich" ? "markdown" : "rich")}>
              {viewMode === "rich" ? "Markdown" : "Rich"}
            </Button>
            <a href={legacyUrl} target="_blank" rel="noopener noreferrer">
              <Button variant="ghost" size="sm" className="text-xs gap-1">
                <ExternalLink className="h-3 w-3" /> Full View
              </Button>
            </a>
            <Button variant="ghost" size="sm" className="text-xs gap-1" onClick={() => window.print()}>
              <Download className="h-3 w-3" /> PDF
            </Button>
          </div>
        </div>

        {loading ? (
          <div className="flex items-center justify-center py-12">
            <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
          </div>
        ) : viewMode === "rich" && jsonData ? (
          renderJsonReport(skillName || "", jsonData)
        ) : markdown ? (
          <div className="prose prose-sm dark:prose-invert max-w-none">
            <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeHighlight]}>
              {markdown}
            </ReactMarkdown>
          </div>
        ) : (
          <p className="text-sm text-muted-foreground text-center py-8">No report data available</p>
        )}
      </DialogContent>
    </Dialog>
  );
}
