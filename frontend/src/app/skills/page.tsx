"use client";

import { useEffect, useState, useCallback } from "react";
import { Sparkles, Play, Loader2, CheckCircle2, Clock, FileText, Eye, Calendar } from "lucide-react";
import { GlassCard } from "@/components/ui/glass-card";
import { Button } from "@/components/ui/button";
import { SkillDetailDialog } from "@/components/skills/skill-detail-dialog";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8765";

interface Skill {
  name: string;
  dir_name: string;
  description: string;
  sections?: string[];
}

interface ScheduledSkill {
  skill: string;
  label: string;
  schedule: string;
  last_run: string | null;
  last_report: string | null;
  launchd_loaded: boolean;
  plist_installed: boolean;
  status: string;
}

export default function SkillsPage() {
  const [skills, setSkills] = useState<Skill[]>([]);
  const [scheduled, setScheduled] = useState<ScheduledSkill[]>([]);
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState<string | null>(null);
  const [completed, setCompleted] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [detailSkill, setDetailSkill] = useState<string | null>(null);
  const [pollSkill, setPollSkill] = useState<string | null>(null);

  const loadData = useCallback(() => {
    fetch(`${API}/api/skills`)
      .then((r) => r.json())
      .then((data) => setSkills(Array.isArray(data) ? data : data.skills || []))
      .catch(() => setSkills([]));
    fetch(`${API}/api/agents/status`)
      .then((r) => r.json())
      .then((data) => setScheduled(Array.isArray(data) ? data : []))
      .catch(() => setScheduled([]));
  }, []);

  useEffect(() => {
    loadData();
    setLoading(false);
  }, [loadData]);

  useEffect(() => {
    if (!pollSkill) return;
    const iv = setInterval(async () => {
      try {
        const r = await fetch(`${API}/api/agents/run-status/${pollSkill}`);
        const d = await r.json();
        if (d.status === "complete" || d.status === "error") {
          setRunning(null);
          setPollSkill(null);
          if (d.status === "complete") {
            setCompleted(pollSkill);
            setTimeout(() => setCompleted(null), 5000);
          } else {
            setError(`${pollSkill}: ${d.error || "failed"}`);
          }
          loadData();
        }
      } catch {}
    }, 2000);
    return () => clearInterval(iv);
  }, [pollSkill, loadData]);

  const runSkill = async (dirName: string) => {
    setRunning(dirName);
    setCompleted(null);
    setError(null);
    try {
      const res = await fetch(`${API}/api/agents/run-skill/${dirName}`, { method: "POST" });
      const data = await res.json();
      if (data.error) {
        setError(`${dirName}: ${data.error}`);
        setRunning(null);
      } else {
        setPollSkill(dirName);
      }
    } catch (e) {
      setError(`${dirName}: ${(e as Error).message}`);
      setRunning(null);
    }
  };

  const scheduledMap = new Map(scheduled.map((s) => [s.skill, s]));

  const formatTime = (iso: string | null) => {
    if (!iso) return "Never";
    try {
      const d = new Date(iso);
      return d.toLocaleDateString("en-US", { month: "short", day: "numeric" }) + " " +
        d.toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit" });
    } catch { return iso; }
  };

  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto max-w-6xl px-6 py-6 space-y-6">
        <div className="flex items-center gap-3">
          <Sparkles className="h-6 w-6 text-primary" />
          <h1 className="text-2xl font-semibold">Agent Skills</h1>
          <span className="text-sm text-muted-foreground">{skills.length} installed</span>
        </div>

        {error && (
          <div className="rounded-xl border border-red-400/30 bg-red-400/10 px-4 py-3 text-sm text-red-300">{error}</div>
        )}

        {/* Scheduled Skills Status */}
        {scheduled.length > 0 && (
          <GlassCard label="Scheduled Skills">
            <div className="overflow-x-auto rounded-lg border border-border">
              <table className="w-full text-sm">
                <thead className="bg-muted/30">
                  <tr>
                    <th className="px-4 py-2 text-left text-xs font-semibold text-muted-foreground">Skill</th>
                    <th className="px-4 py-2 text-center text-xs font-semibold text-muted-foreground">Schedule</th>
                    <th className="px-4 py-2 text-center text-xs font-semibold text-muted-foreground">Status</th>
                    <th className="px-4 py-2 text-center text-xs font-semibold text-muted-foreground">Last Run</th>
                    <th className="px-4 py-2 text-right text-xs font-semibold text-muted-foreground">Report</th>
                  </tr>
                </thead>
                <tbody>
                  {scheduled.map((s) => (
                    <tr key={s.skill} className="border-t border-border hover:bg-muted/20 transition-colors">
                      <td className="px-4 py-2 font-medium text-xs">{s.skill}</td>
                      <td className="px-4 py-2 text-center text-xs text-muted-foreground">
                        <span className="inline-flex items-center gap-1">
                          <Calendar className="h-3 w-3" /> {s.schedule}
                        </span>
                      </td>
                      <td className="px-4 py-2 text-center">
                        <span className={`inline-block rounded-full px-2 py-0.5 text-[10px] font-semibold ${
                          s.status === "active" ? "bg-green-400/15 text-green-400" :
                          s.launchd_loaded ? "bg-yellow-400/15 text-yellow-400" :
                          "bg-muted text-muted-foreground"
                        }`}>
                          {s.status === "active" ? "Active" : s.launchd_loaded ? "Loaded" : "Inactive"}
                        </span>
                      </td>
                      <td className="px-4 py-2 text-center text-xs text-muted-foreground">
                        <span className="inline-flex items-center gap-1">
                          <Clock className="h-3 w-3" /> {formatTime(s.last_run)}
                        </span>
                      </td>
                      <td className="px-4 py-2 text-right">
                        {s.last_report ? (
                          <a
                            href={`${API}/report/${s.skill}/${s.last_report.replace(".md", "")}`}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="inline-flex items-center gap-1 text-xs text-primary hover:underline"
                          >
                            <FileText className="h-3 w-3" /> View
                          </a>
                        ) : (
                          <span className="text-xs text-muted-foreground">—</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </GlassCard>
        )}

        {/* All Skills Grid */}
        {loading ? (
          <div className="flex items-center justify-center py-20">
            <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
          </div>
        ) : (
          <div className="grid gap-4" style={{ gridTemplateColumns: "repeat(auto-fill, minmax(340px, 1fr))" }}>
            {skills.map((skill) => {
              const key = skill.dir_name || skill.name;
              const isRunning = running === key;
              const isCompleted = completed === key;
              const sched = scheduledMap.get(key);

              return (
                <GlassCard key={key} hover className="flex flex-col">
                  <div className="flex items-start justify-between mb-1.5">
                    <div className="text-[1.1rem] font-semibold text-primary">{skill.name}</div>
                    <Button variant="ghost" size="icon" className="h-7 w-7 shrink-0" onClick={() => setDetailSkill(key)} title="View details">
                      <Eye className="h-3.5 w-3.5" />
                    </Button>
                  </div>

                  <p className="text-[0.82rem] text-muted-foreground leading-relaxed mb-3 line-clamp-3 flex-1">
                    {skill.description}
                  </p>

                  {skill.sections && skill.sections.length > 0 && (
                    <div className="flex flex-wrap gap-1.5 mb-3">
                      {skill.sections.slice(0, 5).map((s) => (
                        <span key={s} className="inline-block rounded-md px-2 py-0.5 text-[0.7rem] font-medium bg-[rgba(108,124,255,.12)] text-primary">
                          {s}
                        </span>
                      ))}
                    </div>
                  )}

                  {sched && (
                    <div className="flex items-center gap-3 mb-3 text-xs text-muted-foreground">
                      <span className="inline-flex items-center gap-1">
                        <Calendar className="h-3 w-3" /> {sched.schedule}
                      </span>
                      {sched.last_run && (
                        <span className="inline-flex items-center gap-1">
                          <Clock className="h-3 w-3" /> {formatTime(sched.last_run)}
                        </span>
                      )}
                    </div>
                  )}

                  <div className="flex items-center gap-2 mt-auto">
                    <Button
                      variant="outline"
                      size="sm"
                      className="flex-1 gap-1.5 text-xs"
                      onClick={() => runSkill(key)}
                      disabled={isRunning}
                    >
                      {isRunning ? (
                        <Loader2 className="h-3.5 w-3.5 animate-spin" />
                      ) : isCompleted ? (
                        <CheckCircle2 className="h-3.5 w-3.5 text-green-400" />
                      ) : (
                        <Play className="h-3.5 w-3.5" />
                      )}
                      {isRunning ? "Running..." : isCompleted ? "Done" : "Run"}
                    </Button>
                    {sched?.last_report && (
                      <a
                        href={`${API}/report/${key}/${sched.last_report.replace(".md", "")}`}
                        target="_blank"
                        rel="noopener noreferrer"
                      >
                        <Button variant="ghost" size="sm" className="gap-1.5 text-xs">
                          <FileText className="h-3.5 w-3.5" /> Report
                        </Button>
                      </a>
                    )}
                  </div>
                </GlassCard>
              );
            })}
          </div>
        )}

        <SkillDetailDialog skillName={detailSkill} onClose={() => setDetailSkill(null)} />
      </div>
    </div>
  );
}
