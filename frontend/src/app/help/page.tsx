"use client";

import { HelpCircle, Wrench, MessageSquare } from "lucide-react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { GlassCard } from "@/components/ui/glass-card";

export default function HelpPage() {
  return (
    <ScrollArea className="h-full">
      <div className="mx-auto max-w-4xl px-6 py-5 space-y-5">
        <div className="flex items-center gap-3">
          <HelpCircle className="h-6 w-6 text-primary" />
          <h1 className="text-2xl font-semibold">Help &amp; Documentation</h1>
        </div>

        <GlassCard label="Getting Started">
          <ol className="space-y-3 text-sm text-muted-foreground list-decimal list-inside">
            <li><strong className="text-foreground">Clone the repo:</strong> <code className="rounded bg-background px-1.5 py-0.5 text-xs">git clone https://github.com/Brianletort/MemoryOS.git</code></li>
            <li><strong className="text-foreground">Run setup:</strong> <code className="rounded bg-background px-1.5 py-0.5 text-xs">cd MemoryOS && ./scripts/setup.sh</code></li>
            <li><strong className="text-foreground">Configure:</strong> Use the Setup Wizard to set your vault path, API keys, and extractor settings</li>
            <li><strong className="text-foreground">Activate:</strong> Install background agents and run your first extraction</li>
            <li><strong className="text-foreground">Chat:</strong> Open the Chat tab and ask about your emails, meetings, or run skills</li>
          </ol>
        </GlassCard>

        <GlassCard label="How It Works">
          <pre className="text-xs font-mono leading-relaxed text-muted-foreground bg-background rounded-lg p-4 overflow-x-auto">{`Interface Layer:   Next.js Frontend | Cursor (@context) | CLI | ChatGPT
                         |              |         |
Memory Layer:      SQLite FTS5 Index | Context Generator | Hot/Warm/Cold Tiers
                         |
Collection Layer:  Extractors -> Obsidian Vault (Markdown source of truth)
                   Screenpipe | Mail/Calendar | Outlook/Graph | OneDrive`}</pre>
        </GlassCard>

        <GlassCard label="Agent Tools">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {[
              { name: "memory_search", desc: "Full-text search across emails, meetings, transcripts, activity" },
              { name: "vault_read", desc: "Read a specific file from the Obsidian vault" },
              { name: "vault_browse", desc: "List files and directories in the vault" },
              { name: "vault_recent", desc: "Get recently created/modified documents" },
              { name: "context_summary", desc: "Today's auto-generated context (calendar, emails, activity)" },
              { name: "run_skill", desc: "Execute a MemoryOS skill (morning-brief, meeting-prep, etc.)" },
              { name: "web_search", desc: "Search the web via DuckDuckGo" },
              { name: "send_email", desc: "Send email via Mail.app or SMTP" },
              { name: "build_slides", desc: "Generate a PowerPoint deck from a structured spec" },
              { name: "analyze_file", desc: "Extract text from uploaded PDF, DOCX, CSV files" },
              { name: "vault_write", desc: "Create or update files in the vault" },
              { name: "meeting_overview", desc: "Get calendar + transcripts for a date" },
            ].map((t) => (
              <div key={t.name} className="flex items-start gap-2 text-sm">
                <Wrench className="h-3.5 w-3.5 mt-0.5 text-primary shrink-0" />
                <div><code className="text-xs text-primary">{t.name}</code><div className="text-xs text-muted-foreground mt-0.5">{t.desc}</div></div>
              </div>
            ))}
          </div>
        </GlassCard>

        <GlassCard label="Available Skills">
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 text-sm">
            {["morning-brief", "meeting-prep", "weekly-status", "news-pulse", "plan-my-week",
              "commitment-tracker", "project-brief", "focus-audit", "relationship-crm",
              "team-manager", "approvals-queue", "strategic-radar", "decision-log",
              "executive-package", "career-narrative", "catchup-brief", "pptx-builder"].map((s) => (
              <div key={s} className="flex items-center gap-1.5 text-muted-foreground">
                <MessageSquare className="h-3 w-3 text-primary" /> {s}
              </div>
            ))}
          </div>
        </GlassCard>

        <GlassCard label="Keyboard Shortcuts">
          <div className="space-y-2 text-sm">
            {[
              ["Enter", "Send message in chat"],
              ["Shift + Enter", "New line in chat input"],
              ["Click sidebar session", "Load session history"],
            ].map(([key, desc]) => (
              <div key={key} className="flex items-center gap-3">
                <kbd className="rounded bg-background border border-border px-2 py-0.5 text-xs font-mono">{key}</kbd>
                <span className="text-muted-foreground">{desc}</span>
              </div>
            ))}
          </div>
        </GlassCard>
      </div>
    </ScrollArea>
  );
}
