"use client";

import { useEffect, useState, useCallback } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Folder, FileText, ChevronRight, Loader2, Home } from "lucide-react";
import { ScrollArea } from "@/components/ui/scroll-area";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8765";

interface BrowseData {
  current_path: string;
  breadcrumb: { name: string; path: string }[];
  folders: { name: string; path: string }[];
  files: { name: string; path: string; size: number; modified: string }[];
}

export default function BrowserPage() {
  const [, setPath] = useState("");
  const [data, setData] = useState<BrowseData | null>(null);
  const [fileContent, setFileContent] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const browse = useCallback(async (p: string) => {
    setLoading(true);
    setFileContent(null);
    try {
      const res = await fetch(`${API_BASE}/api/browse?path=${encodeURIComponent(p)}`);
      setData(await res.json());
      setPath(p);
    } catch { setData(null); }
    finally { setLoading(false); }
  }, []);

  const openFile = async (p: string) => {
    try {
      const res = await fetch(`${API_BASE}/api/file?path=${encodeURIComponent(p)}`);
      const d = await res.json();
      setFileContent(d.content || d.text || "");
    } catch { setFileContent("Failed to load file"); }
  };

  useEffect(() => { browse(""); }, [browse]);

  return (
    <div className="flex h-full flex-col">
      {/* Breadcrumb */}
      <div className="flex items-center gap-1 px-6 py-3 border-b border-border text-sm flex-wrap">
        <button onClick={() => browse("")} className="text-primary hover:underline flex items-center gap-1"><Home className="h-3.5 w-3.5" /> Vault</button>
        {data?.breadcrumb?.map((b, i) => (
          <span key={i} className="flex items-center gap-1">
            <ChevronRight className="h-3 w-3 text-muted-foreground" />
            <button onClick={() => browse(b.path)} className="text-primary hover:underline">{b.name}</button>
          </span>
        ))}
      </div>

      <div className="flex flex-1 min-h-0">
        {/* File list */}
        <div className="w-80 shrink-0 border-r border-border">
          <ScrollArea className="h-full">
            <div className="p-2 space-y-0.5">
              {loading && <div className="flex justify-center py-8"><Loader2 className="h-5 w-5 animate-spin text-muted-foreground" /></div>}
              {data?.folders?.map((f) => (
                <button key={f.path} onClick={() => browse(f.path)} className="w-full flex items-center gap-2 rounded-md px-3 py-2 text-sm hover:bg-accent/50 transition-colors text-left">
                  <Folder className="h-4 w-4 text-primary shrink-0" />
                  <span className="truncate">{f.name}</span>
                </button>
              ))}
              {data?.files?.map((f) => (
                <button key={f.path} onClick={() => openFile(f.path)} className="w-full flex items-center gap-2 rounded-md px-3 py-2 text-sm hover:bg-accent/50 transition-colors text-left">
                  <FileText className="h-4 w-4 text-muted-foreground shrink-0" />
                  <span className="truncate flex-1">{f.name}</span>
                  <span className="text-xs text-muted-foreground shrink-0">{(f.size / 1024).toFixed(0)}K</span>
                </button>
              ))}
              {data && !data.folders?.length && !data.files?.length && <div className="text-sm text-muted-foreground text-center py-8">Empty directory</div>}
            </div>
          </ScrollArea>
        </div>

        {/* Content preview */}
        <div className="flex-1 p-6">
          {fileContent !== null ? (
            <ScrollArea className="h-full">
              <div className="prose prose-sm dark:prose-invert max-w-none">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>{fileContent}</ReactMarkdown>
              </div>
            </ScrollArea>
          ) : (
            <div className="flex h-full items-center justify-center text-muted-foreground text-sm">Select a file to preview</div>
          )}
        </div>
      </div>
    </div>
  );
}
