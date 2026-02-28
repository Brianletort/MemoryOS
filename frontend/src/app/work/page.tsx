"use client";

import { useEffect, useState, useMemo, useCallback } from "react";
import { Briefcase, Loader2, ChevronDown, ChevronRight, Plus, Save, Trash2, ClipboardCheck, Check, X } from "lucide-react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { GlassCard } from "@/components/ui/glass-card";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8765";

interface Task { id: string; task: string; status: string; priority: string; due?: string; notes?: string; owner?: string; section?: string; subtasks?: Task[]; }
interface Project { name: string; tasks: Task[]; owner?: string; }
interface WaitingItem { what: string; who: string; since: string; followup: string; }
interface TasksData { projects: Project[]; waiting: WaitingItem[]; completed: Task[]; backlog: Task[]; priorities: string[]; project_names: string[]; }

const STATUS_OPTS = ["not_started", "in_progress", "complete", "waiting", "blocked"];
const PRI_OPTS = ["P0", "P1", "P2"];

function statusLabel(s: string) { return s.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase()); }

export default function WorkPage() {
  const [data, setData] = useState<TasksData | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [saveMsg, setSaveMsg] = useState<string | null>(null);
  const [hideDone, setHideDone] = useState(false);
  const [filterStatus, setFilterStatus] = useState("");
  const [filterPriority, setFilterPriority] = useState("");
  const [filterProject, setFilterProject] = useState("");
  const [collapsed, setCollapsed] = useState<Set<string>>(new Set());
  const [approvals, setApprovals] = useState<{ id: string; system: string; title: string; type: string; status: string; detected_at: string; url?: string }[]>([]);
  const [view, setView] = useState<"table" | "raw">("table");
  const [rawTasks, setRawTasks] = useState("");
  const [rawPri, setRawPri] = useState("");

  const loadData = useCallback(async () => {
    setLoading(true);
    try {
      const d = await fetch(`${API_BASE}/api/tasks`).then((r) => r.json());
      setData(d);
    } catch { setData(null); }
    finally { setLoading(false); }
    fetch(`${API_BASE}/api/approvals/state`)
      .then((r) => r.json())
      .then((d) => setApprovals(Array.isArray(d?.pending) ? d.pending : Array.isArray(d) ? d : []))
      .catch(() => setApprovals([]));
  }, []);

  useEffect(() => { loadData(); }, [loadData]);

  const loadRaw = useCallback(async () => {
    try {
      const [t, p] = await Promise.all([
        fetch(`${API_BASE}/api/context/file?name=tasks.md`).then((r) => r.json()),
        fetch(`${API_BASE}/api/context/file?name=priorities.md`).then((r) => r.json()),
      ]);
      setRawTasks(t.content || "");
      setRawPri(p.content || "");
    } catch { /* ignore */ }
  }, []);

  const toggleView = (v: "table" | "raw") => {
    setView(v);
    if (v === "raw") loadRaw();
  };

  const toggleCollapse = (name: string) => setCollapsed((prev) => { const n = new Set(prev); if (n.has(name)) n.delete(name); else n.add(name); return n; });

  const updateTask = (projIdx: number, taskIdx: number, field: keyof Task, value: string) => {
    if (!data) return;
    const next = structuredClone(data);
    (next.projects[projIdx].tasks[taskIdx] as unknown as Record<string, unknown>)[field] = value;
    setData(next);
  };

  const addTask = (projIdx: number) => {
    if (!data) return;
    const next = structuredClone(data);
    const id = `t_new_${Date.now()}`;
    next.projects[projIdx].tasks.push({ id, task: "", status: "not_started", priority: "P1", due: "", notes: "", owner: "" });
    setData(next);
  };

  const deleteTask = (projIdx: number, taskIdx: number) => {
    if (!data) return;
    const next = structuredClone(data);
    next.projects[projIdx].tasks.splice(taskIdx, 1);
    setData(next);
  };

  const saveAll = async () => {
    if (!data) return;
    setSaving(true);
    setSaveMsg(null);
    try {
      const res = await fetch(`${API_BASE}/api/tasks`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      });
      const d = await res.json();
      if (d.error) throw new Error(d.error);
      setSaveMsg("Saved!");
      setTimeout(() => setSaveMsg(null), 3000);
    } catch (e) {
      setSaveMsg(`Error: ${(e as Error).message}`);
    } finally { setSaving(false); }
  };

  const saveRaw = async (name: string, content: string) => {
    try {
      await fetch(`${API_BASE}/api/context/file`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, content }),
      });
      setSaveMsg(`${name} saved!`);
      setTimeout(() => setSaveMsg(null), 3000);
    } catch (e) { setSaveMsg(`Error: ${(e as Error).message}`); }
  };

  const filteredProjects = useMemo(() => {
    if (!data) return [];
    return data.projects
      .filter((p) => !filterProject || p.name === filterProject)
      .map((p) => ({ ...p, tasks: p.tasks.filter((t) => {
        if (hideDone && t.status === "complete") return false;
        if (filterStatus && t.status !== filterStatus) return false;
        if (filterPriority && t.priority !== filterPriority) return false;
        return true;
      }) }))
      .filter((p) => p.tasks.length > 0);
  }, [data, hideDone, filterStatus, filterPriority, filterProject]);

  const counts = useMemo(() => {
    if (!data) return { open: 0, prog: 0, done: 0 };
    const all = data.projects.flatMap((p) => p.tasks);
    return { open: all.filter((t) => t.status === "not_started").length, prog: all.filter((t) => t.status === "in_progress" || t.status === "in-progress").length, done: all.filter((t) => t.status === "complete").length };
  }, [data]);

  if (loading) return <div className="flex items-center justify-center h-full"><Loader2 className="h-6 w-6 animate-spin text-muted-foreground" /></div>;
  if (!data) return <div className="flex items-center justify-center h-full text-muted-foreground">Failed to load tasks</div>;

  return (
    <ScrollArea className="h-full">
      <div className="mx-auto max-w-6xl px-6 py-5 space-y-4">
        <div className="flex items-center gap-3">
          <Briefcase className="h-6 w-6 text-primary" />
          <h1 className="text-2xl font-semibold">My Work</h1>
          <span className="text-sm text-muted-foreground">Edits save to tasks.md in the vault</span>
        </div>

        {/* Top bar */}
        <div className="flex items-center gap-3 flex-wrap">
          <div className="inline-flex border border-border rounded-lg overflow-hidden">
            <button onClick={() => toggleView("table")} className={cn("px-3.5 py-1.5 text-xs font-medium transition-colors", view === "table" ? "bg-primary text-white" : "text-muted-foreground hover:text-foreground")}>Table</button>
            <button onClick={() => toggleView("raw")} className={cn("px-3.5 py-1.5 text-xs font-medium transition-colors", view === "raw" ? "bg-primary text-white" : "text-muted-foreground hover:text-foreground")}>Raw Markdown</button>
          </div>

          {view === "table" && (
            <>
              <label className="flex items-center gap-1.5 text-xs text-muted-foreground cursor-pointer select-none">
                <input type="checkbox" checked={hideDone} onChange={(e) => setHideDone(e.target.checked)} className="accent-primary" /> Hide complete
              </label>
              <select value={filterStatus} onChange={(e) => setFilterStatus(e.target.value)} className="rounded-md border border-border bg-background px-2 py-1 text-xs text-foreground">
                <option value="">All Status</option>
                {STATUS_OPTS.map((s) => <option key={s} value={s}>{statusLabel(s)}</option>)}
              </select>
              <select value={filterPriority} onChange={(e) => setFilterPriority(e.target.value)} className="rounded-md border border-border bg-background px-2 py-1 text-xs text-foreground">
                <option value="">All Priority</option>
                {PRI_OPTS.map((p) => <option key={p} value={p}>{p}</option>)}
              </select>
              <select value={filterProject} onChange={(e) => setFilterProject(e.target.value)} className="rounded-md border border-border bg-background px-2 py-1 text-xs text-foreground">
                <option value="">All Projects</option>
                {data.project_names?.map((n) => <option key={n} value={n}>{n}</option>)}
              </select>
            </>
          )}

          <div className="ml-auto flex items-center gap-3">
            {saveMsg && <span className={cn("text-xs font-medium", saveMsg.startsWith("Error") ? "text-red-400" : "text-green-400")}>{saveMsg}</span>}
            <button onClick={saveAll} disabled={saving} className="inline-flex items-center gap-1.5 rounded-lg border border-border bg-card px-4 py-1.5 text-xs font-medium text-primary transition-all hover:bg-primary hover:text-white hover:border-primary disabled:opacity-40">
              {saving ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Save className="h-3.5 w-3.5" />} Save All
            </button>
          </div>
        </div>

        {/* Task counts */}
        <div className="flex items-center gap-4 text-xs text-muted-foreground">
          <span><span className="text-foreground font-semibold">{counts.open}</span> open</span>
          <span><span className="text-primary font-semibold">{counts.prog}</span> in progress</span>
          <span><span className="text-green-400 font-semibold">{counts.done}</span> done</span>
        </div>

        {view === "table" ? (
          <GlassCard className="p-0 overflow-hidden">
            <div className="grid grid-cols-[1fr_100px_65px_105px_85px_1fr_32px] gap-0 px-3 py-2 text-[0.7rem] font-semibold uppercase tracking-[0.04em] text-muted-foreground border-b-2 border-border">
              <span>Task</span><span>Owner</span><span>Pri</span><span>Status</span><span>Due</span><span>Notes</span><span></span>
            </div>
            <div>
              {filteredProjects.map((project) => {
                const origPi = data.projects.findIndex((p) => p.name === project.name);
                const allTasks = data.projects[origPi]?.tasks || [];
                const doneN = allTasks.filter((t) => t.status === "complete").length;
                return (
                  <div key={project.name}>
                    <div role="button" tabIndex={0} onClick={() => toggleCollapse(project.name)} onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") toggleCollapse(project.name); }} className="w-full flex items-center gap-2 px-3 py-2 text-sm font-semibold text-primary bg-[rgba(108,124,255,.06)] hover:bg-[rgba(108,124,255,.1)] transition-colors border-b border-border cursor-pointer">
                      {collapsed.has(project.name) ? <ChevronRight className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />}
                      {project.name}
                      <span className="text-xs text-muted-foreground font-normal ml-1">{doneN}/{allTasks.length} done</span>
                      <button onClick={(e) => { e.stopPropagation(); addTask(origPi); }} className="ml-auto text-xs text-green-400 hover:text-green-300 flex items-center gap-1"><Plus className="h-3 w-3" /> Task</button>
                    </div>
                    {!collapsed.has(project.name) && project.tasks.map((task) => {
                      const ti = data.projects[origPi].tasks.findIndex((t) => t.id === task.id);
                      const isOverdue = task.due && new Date(task.due + " 2026") < new Date() && task.status !== "complete";
                      return (
                        <div key={task.id} className={cn("grid grid-cols-[1fr_100px_65px_105px_85px_1fr_32px] gap-0 px-3 py-1.5 text-[0.82rem] border-b border-border items-center", task.status === "complete" && "opacity-50")}>
                          <input value={task.task} onChange={(e) => updateTask(origPi, ti, "task", e.target.value)} className={cn("bg-transparent border border-transparent hover:border-border focus:border-primary focus:bg-background rounded px-1.5 py-1 text-sm outline-none w-full", task.status === "complete" && "line-through")} />
                          <input value={task.owner || ""} onChange={(e) => updateTask(origPi, ti, "owner", e.target.value)} className="bg-transparent border border-transparent hover:border-border focus:border-primary focus:bg-background rounded px-1.5 py-1 text-xs outline-none w-full text-muted-foreground" />
                          <select value={task.priority} onChange={(e) => updateTask(origPi, ti, "priority", e.target.value)} className="bg-transparent border border-transparent hover:border-border focus:border-primary rounded px-1 py-1 text-xs outline-none appearance-none cursor-pointer">
                            {PRI_OPTS.map((p) => <option key={p} value={p}>{p}</option>)}
                          </select>
                          <select value={task.status} onChange={(e) => updateTask(origPi, ti, "status", e.target.value)} className="bg-transparent border border-transparent hover:border-border focus:border-primary rounded px-1 py-1 text-xs outline-none appearance-none cursor-pointer">
                            {STATUS_OPTS.map((s) => <option key={s} value={s}>{statusLabel(s)}</option>)}
                          </select>
                          <input value={task.due || ""} onChange={(e) => updateTask(origPi, ti, "due", e.target.value)} className={cn("bg-transparent border border-transparent hover:border-border focus:border-primary focus:bg-background rounded px-1.5 py-1 text-xs outline-none w-full", isOverdue && "text-red-400 font-semibold")} />
                          <input value={task.notes || ""} onChange={(e) => updateTask(origPi, ti, "notes", e.target.value)} className="bg-transparent border border-transparent hover:border-border focus:border-primary focus:bg-background rounded px-1.5 py-1 text-xs outline-none w-full text-muted-foreground" placeholder="notes..." />
                          <button onClick={() => deleteTask(origPi, ti)} className="text-muted-foreground hover:text-red-400 hover:bg-red-400/10 rounded p-1 transition-colors"><Trash2 className="h-3.5 w-3.5" /></button>
                        </div>
                      );
                    })}
                  </div>
                );
              })}
              {filteredProjects.length === 0 && <div className="px-3 py-8 text-center text-sm text-muted-foreground">No tasks match filters</div>}
            </div>
          </GlassCard>
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <GlassCard label="Tasks (tasks.md)" className="flex flex-col">
              <textarea value={rawTasks} onChange={(e) => setRawTasks(e.target.value)} className="flex-1 min-h-[400px] bg-background border border-border rounded-lg p-3 font-mono text-xs leading-relaxed resize-y outline-none focus:border-primary" spellCheck={false} />
              <button onClick={() => saveRaw("tasks.md", rawTasks)} className="mt-3 self-start inline-flex items-center gap-1.5 rounded-lg border border-border bg-card px-4 py-1.5 text-xs font-medium text-primary hover:bg-primary hover:text-white hover:border-primary transition-all"><Save className="h-3.5 w-3.5" /> Save</button>
            </GlassCard>
            <GlassCard label="Priorities (priorities.md)" className="flex flex-col">
              <textarea value={rawPri} onChange={(e) => setRawPri(e.target.value)} className="flex-1 min-h-[400px] bg-background border border-border rounded-lg p-3 font-mono text-xs leading-relaxed resize-y outline-none focus:border-primary" spellCheck={false} />
              <button onClick={() => saveRaw("priorities.md", rawPri)} className="mt-3 self-start inline-flex items-center gap-1.5 rounded-lg border border-border bg-card px-4 py-1.5 text-xs font-medium text-primary hover:bg-primary hover:text-white hover:border-primary transition-all"><Save className="h-3.5 w-3.5" /> Save</button>
            </GlassCard>
          </div>
        )}

        {approvals.length > 0 && (
          <GlassCard label={`Approvals Queue (${approvals.length})`}>
            <div className="flex items-center gap-2 mb-3">
              <ClipboardCheck className="h-4 w-4 text-primary" />
              <p className="text-xs text-muted-foreground">Pending approvals detected from screen activity</p>
            </div>
            <div className="space-y-2">
              {approvals.map((a) => (
                <div key={a.id} className="flex items-center gap-3 rounded-lg border border-border p-3">
                  <div className="flex-1 min-w-0">
                    <div className="font-medium text-sm truncate">{a.title}</div>
                    <div className="text-xs text-muted-foreground mt-0.5">
                      {a.system} &middot; {a.type} &middot; {new Date(a.detected_at).toLocaleDateString()}
                    </div>
                  </div>
                  <div className="flex items-center gap-1.5 shrink-0">
                    {a.url && (
                      <a href={a.url} target="_blank" rel="noopener noreferrer">
                        <Button variant="outline" size="sm" className="h-7 text-xs">Open</Button>
                      </a>
                    )}
                    <Button variant="ghost" size="icon" className="h-7 w-7 text-green-400" onClick={async () => { await fetch(`${API_BASE}/api/approvals/action`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ id: a.id, action: "approve" }) }); loadData(); }} title="Mark approved">
                      <Check className="h-3.5 w-3.5" />
                    </Button>
                    <Button variant="ghost" size="icon" className="h-7 w-7 text-muted-foreground" onClick={async () => { await fetch(`${API_BASE}/api/approvals/action`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ id: a.id, action: "dismiss" }) }); loadData(); }} title="Dismiss">
                      <X className="h-3.5 w-3.5" />
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          </GlassCard>
        )}

        {data.waiting?.length > 0 && view === "table" && (
          <GlassCard label="Waiting On">
            <div className="grid grid-cols-[1fr_100px_85px_1fr] gap-0 px-1 py-1.5 text-[0.7rem] font-semibold uppercase tracking-[0.04em] text-muted-foreground border-b border-border">
              <span>What</span><span>Who</span><span>Since</span><span>Follow-up</span>
            </div>
            {data.waiting.map((w, i) => (
              <div key={i} className="grid grid-cols-[1fr_100px_85px_1fr] gap-0 px-1 py-2 text-sm border-b border-[rgba(255,255,255,.04)] last:border-0 text-orange-300/80">
                <span className="truncate pr-2">{w.what}</span><span className="text-xs">{w.who}</span><span className="text-xs">{w.since}</span><span className="text-xs truncate">{w.followup}</span>
              </div>
            ))}
          </GlassCard>
        )}
      </div>
    </ScrollArea>
  );
}
