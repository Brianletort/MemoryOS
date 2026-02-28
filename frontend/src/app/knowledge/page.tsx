"use client";

import { useEffect, useState, useCallback } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { BookOpen, FileText, Pin, Plus, Save, Loader2 } from "lucide-react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8765";

interface KnowledgeFile { name: string; path: string; size: number; modified: string; pinned?: boolean; }

export default function KnowledgePage() {
  const [files, setFiles] = useState<KnowledgeFile[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [content, setContent] = useState<string>("");
  const [editing, setEditing] = useState(false);
  const [editContent, setEditContent] = useState("");
  const [saving, setSaving] = useState(false);
  const [newName, setNewName] = useState("");
  const [showNew, setShowNew] = useState(false);

  const loadFiles = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/knowledge/files`);
      const d = await res.json();
      setFiles(Array.isArray(d) ? d : d.files || []);
    } catch { setFiles([]); }
  }, []);

  useEffect(() => { loadFiles(); }, [loadFiles]);

  const loadFile = async (path: string) => {
    setSelected(path);
    setEditing(false);
    try {
      const res = await fetch(`${API_BASE}/api/knowledge/file?path=${encodeURIComponent(path)}`);
      const d = await res.json();
      setContent(d.content || "");
      setEditContent(d.content || "");
    } catch { setContent("Failed to load"); }
  };

  const saveFile = async () => {
    if (!selected) return;
    setSaving(true);
    try {
      await fetch(`${API_BASE}/api/knowledge/file`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path: selected, content: editContent }),
      });
      setContent(editContent);
      setEditing(false);
    } catch { /* ignore */ }
    finally { setSaving(false); }
  };

  const createFile = async () => {
    if (!newName.trim()) return;
    try {
      await fetch(`${API_BASE}/api/knowledge/new`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: newName.trim() }),
      });
      setNewName("");
      setShowNew(false);
      loadFiles();
    } catch { /* ignore */ }
  };

  const togglePin = async (path: string) => {
    try {
      await fetch(`${API_BASE}/api/knowledge/pin`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path }),
      });
      loadFiles();
    } catch { /* ignore */ }
  };

  return (
    <div className="flex h-full">
      <div className="w-72 shrink-0 border-r border-border bg-card flex flex-col">
        <div className="flex items-center gap-2 border-b border-border p-4">
          <BookOpen className="h-5 w-5 text-primary" />
          <h1 className="text-lg font-semibold flex-1">Knowledge</h1>
          <button onClick={() => setShowNew(!showNew)} className="text-primary hover:text-foreground transition-colors"><Plus className="h-4 w-4" /></button>
        </div>
        {showNew && (
          <div className="flex gap-2 px-3 py-2 border-b border-border">
            <input value={newName} onChange={(e) => setNewName(e.target.value)} placeholder="filename.md" className="flex-1 rounded-md border border-border bg-background px-2 py-1 text-xs outline-none focus:border-primary" onKeyDown={(e) => e.key === "Enter" && createFile()} />
            <button onClick={createFile} className="text-xs text-primary font-medium">Create</button>
          </div>
        )}
        <ScrollArea className="flex-1">
          <div className="p-2 space-y-0.5">
            {files.map((f) => (
              <div key={f.path} className={cn("group flex items-center gap-2 rounded-md px-3 py-2 text-sm transition-colors cursor-pointer", selected === f.path ? "bg-accent text-accent-foreground" : "hover:bg-accent/50")}>
                <button onClick={() => loadFile(f.path)} className="flex items-center gap-2 flex-1 min-w-0 text-left">
                  <FileText className="h-4 w-4 shrink-0 text-muted-foreground" />
                  <span className="truncate">{f.name}</span>
                </button>
                <button onClick={() => togglePin(f.path)} className="opacity-0 group-hover:opacity-100 transition-opacity" title="Toggle pin">
                  <Pin className={cn("h-3.5 w-3.5", f.pinned ? "text-primary" : "text-muted-foreground")} />
                </button>
              </div>
            ))}
            {files.length === 0 && <p className="px-3 py-8 text-center text-sm text-muted-foreground">No knowledge files</p>}
          </div>
        </ScrollArea>
      </div>

      <div className="flex-1 flex flex-col">
        {selected && (
          <div className="flex items-center gap-2 px-6 py-2 border-b border-border">
            <span className="text-sm text-muted-foreground flex-1 truncate">{selected}</span>
            {editing ? (
              <button onClick={saveFile} disabled={saving} className="inline-flex items-center gap-1.5 rounded-lg border border-border px-3 py-1 text-xs font-medium text-primary hover:bg-primary hover:text-white hover:border-primary transition-all disabled:opacity-40">
                {saving ? <Loader2 className="h-3 w-3 animate-spin" /> : <Save className="h-3 w-3" />} Save
              </button>
            ) : (
              <button onClick={() => setEditing(true)} className="text-xs text-primary hover:underline">Edit</button>
            )}
          </div>
        )}
        <div className="flex-1 p-6">
          {selected ? (
            editing ? (
              <textarea value={editContent} onChange={(e) => setEditContent(e.target.value)} className="w-full h-full bg-background border border-border rounded-lg p-4 font-mono text-xs leading-relaxed resize-none outline-none focus:border-primary" spellCheck={false} />
            ) : (
              <ScrollArea className="h-full">
                <div className="prose prose-sm dark:prose-invert max-w-none">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
                </div>
              </ScrollArea>
            )
          ) : (
            <div className="flex h-full items-center justify-center text-muted-foreground">Select a knowledge file to view</div>
          )}
        </div>
      </div>
    </div>
  );
}
