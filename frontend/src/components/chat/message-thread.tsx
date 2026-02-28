"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import { User, Bot, Download, Copy, Check, Loader2, CheckCircle2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { useChatStore } from "@/stores/chat-store";
import { downloadUrl } from "@/lib/api";
import { cn } from "@/lib/utils";
import type { Components } from "react-markdown";

const PROSE_CLASSES = [
  "prose dark:prose-invert max-w-none",
  "prose-headings:text-foreground prose-headings:font-semibold prose-headings:tracking-tight",
  "prose-h1:text-xl prose-h1:mt-6 prose-h1:mb-3",
  "prose-h2:text-lg prose-h2:mt-6 prose-h2:mb-2 prose-h2:border-b prose-h2:border-border/30 prose-h2:pb-1",
  "prose-h3:text-base prose-h3:mt-4 prose-h3:mb-2",
  "prose-p:text-sm prose-p:leading-7 prose-p:my-3 prose-p:text-foreground/90",
  "prose-li:text-sm prose-li:leading-7 prose-li:my-1 prose-li:text-foreground/90",
  "prose-ul:my-3 prose-ol:my-3",
  "prose-strong:text-foreground prose-strong:font-semibold",
  "prose-em:text-foreground/80",
  "prose-a:text-primary prose-a:underline prose-a:underline-offset-2 prose-a:decoration-primary/40 hover:prose-a:decoration-primary",
  "[&_pre]:bg-[#1a1d27] [&_pre]:rounded-xl [&_pre]:my-4 [&_pre]:p-4 [&_pre]:border [&_pre]:border-border/30",
  "[&_code]:text-[13px] [&_code]:font-mono",
  "[&_:not(pre)>code]:bg-[#22252f] [&_:not(pre)>code]:text-primary [&_:not(pre)>code]:px-1.5 [&_:not(pre)>code]:py-0.5 [&_:not(pre)>code]:rounded-md [&_:not(pre)>code]:text-[13px] [&_:not(pre)>code]:font-medium",
  "[&_blockquote]:border-l-primary/40 [&_blockquote]:bg-primary/5 [&_blockquote]:rounded-r-lg [&_blockquote]:py-1 [&_blockquote]:px-4 [&_blockquote]:text-foreground/70",
  "[&_hr]:border-border/40 [&_hr]:my-6",
  "[&_table]:text-sm [&_th]:text-left [&_th]:font-semibold [&_th]:px-3 [&_th]:py-2 [&_th]:bg-muted/50 [&_td]:px-3 [&_td]:py-2 [&_td]:border-t [&_td]:border-border/30",
  "[&_img]:rounded-xl [&_img]:border [&_img]:border-border/30 [&_img]:shadow-lg",
].join(" ");

const DOWNLOAD_RE = /\/api\/files\/download\/([^\s)"']+)/g;
const IMAGE_EXTENSIONS = /\.(png|jpg|jpeg|gif|webp)$/i;

function extractDownloads(content: string): { images: string[]; files: string[] } {
  const images: string[] = [];
  const files: string[] = [];
  let m: RegExpExecArray | null;
  DOWNLOAD_RE.lastIndex = 0;
  while ((m = DOWNLOAD_RE.exec(content)) !== null) {
    if (IMAGE_EXTENSIONS.test(m[1])) {
      images.push(m[1]);
    } else {
      files.push(m[1]);
    }
  }
  return { images, files };
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  const handleCopy = useCallback(() => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }, [text]);

  return (
    <button
      onClick={handleCopy}
      className="absolute right-2 top-2 rounded-md bg-background/80 p-1.5 text-muted-foreground opacity-0 transition-opacity group-hover:opacity-100 hover:text-foreground"
      title="Copy code"
    >
      {copied ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
    </button>
  );
}

const TOOL_LABELS: Record<string, string> = {
  memory_search: "Searching vault",
  vault_read: "Reading file",
  vault_browse: "Browsing vault",
  vault_recent: "Finding recent docs",
  context_summary: "Loading context",
  run_skill: "Running skill",
  web_search: "Searching the web",
  send_email: "Sending email",
  vault_write: "Writing to vault",
  meeting_overview: "Loading meetings",
  shell_exec: "Running command",
  build_slides: "Building slides",
  generate_image: "Generating image",
  analyze_file: "Analyzing file",
};

function InlineToolCalls() {
  const events = useChatStore((s) => s.streamingTraceEvents);
  if (events.length === 0) return null;

  const paired: { name: string; done: boolean }[] = [];
  const pending = new Map<string, number>();

  for (const ev of events) {
    if (ev.type === "tool_call") {
      pending.set(ev.name, paired.length);
      paired.push({ name: ev.name, done: false });
    } else if (ev.type === "tool_result") {
      const idx = pending.get(ev.name);
      if (idx !== undefined) {
        paired[idx].done = true;
        pending.delete(ev.name);
      }
    }
  }

  return (
    <div className="flex flex-col gap-1.5 mb-3">
      {paired.map((item, i) => (
        <div key={i} className="flex items-center gap-2 text-xs text-muted-foreground">
          {item.done ? (
            <CheckCircle2 className="h-3.5 w-3.5 text-green-500 shrink-0" />
          ) : (
            <Loader2 className="h-3.5 w-3.5 animate-spin shrink-0" />
          )}
          <span>{TOOL_LABELS[item.name] || item.name}...</span>
        </div>
      ))}
    </div>
  );
}

const markdownComponents: Components = {
  pre({ children, ...props }) {
    return (
      <div className="group relative">
        <pre {...props} className="overflow-x-auto rounded-lg bg-background/50 p-4 text-xs">
          {children}
        </pre>
      </div>
    );
  },
  code({ children, className, ...props }) {
    const isBlock = className?.startsWith("language-") || className?.startsWith("hljs");
    if (!isBlock) {
      return (
        <code className="rounded bg-background/50 px-1.5 py-0.5 text-xs font-mono" {...props}>
          {children}
        </code>
      );
    }
    const text = String(children).replace(/\n$/, "");
    const lang = className?.replace(/^language-/, "").replace(/^hljs\s*/, "") || "";
    return (
      <>
        {lang && (
          <div className="flex items-center justify-between rounded-t-lg bg-muted/50 px-4 py-1.5 text-[10px] font-medium text-muted-foreground -mb-2">
            {lang}
          </div>
        )}
        <code className={className} {...props}>
          {children}
        </code>
        <CopyButton text={text} />
      </>
    );
  },
  img({ src, alt, ...props }) {
    if (!src || typeof src !== "string") return null;
    const fullSrc = src.startsWith("/api/") ? downloadUrl(src.replace("/api/files/download/", "")) : src;
    return (
      <a href={fullSrc} target="_blank" rel="noopener noreferrer" className="block my-3">
        <img
          src={fullSrc}
          alt={alt || "Generated image"}
          className="max-w-full rounded-lg border border-border shadow-sm hover:shadow-md transition-shadow"
          loading="lazy"
          {...props}
        />
      </a>
    );
  },
  table({ children, ...props }) {
    return (
      <div className="my-3 overflow-x-auto rounded-lg border border-border">
        <table className="w-full text-sm" {...props}>
          {children}
        </table>
      </div>
    );
  },
  thead({ children, ...props }) {
    return (
      <thead className="bg-muted/50" {...props}>
        {children}
      </thead>
    );
  },
  th({ children, ...props }) {
    return (
      <th className="px-3 py-2 text-left text-xs font-semibold" {...props}>
        {children}
      </th>
    );
  },
  td({ children, ...props }) {
    return (
      <td className="border-t border-border px-3 py-2 text-xs" {...props}>
        {children}
      </td>
    );
  },
};

export function MessageThread() {
  const messages = useChatStore((s) => s.messages);
  const streamingContent = useChatStore((s) => s.streamingContent);
  const isStreaming = useChatStore((s) => s.isStreaming);
  const statusText = useChatStore((s) => s.statusText);
  const activeSessionId = useChatStore((s) => s.activeSessionId);
  const scrollRef = useRef<HTMLDivElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const isNearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 200;
    if (isNearBottom) {
      bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages, streamingContent]);

  if (!activeSessionId && messages.length === 0 && !isStreaming) {
    return (
      <div className="flex h-full items-center justify-center">
        <div className="max-w-md text-center">
          <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-primary/10">
            <Bot className="h-8 w-8 text-primary" />
          </div>
          <h2 className="mb-2 text-xl font-semibold">MemoryOS Chat</h2>
          <p className="text-sm text-muted-foreground">
            Ask about your emails, meetings, calendar, or run skills like
            morning-brief, meeting-prep, or build slides.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div ref={scrollRef} className="h-full overflow-y-auto">
      <div className="mx-auto max-w-3xl px-4 py-6 space-y-6">
        {messages.map((msg) => (
          <MessageBubble key={msg.id} role={msg.role} content={msg.content} />
        ))}

        {isStreaming && (
          <div className="flex gap-3">
            <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary/10">
              <Bot className="h-4 w-4 text-primary" />
            </div>
            <div className="min-w-0 max-w-[92%] py-2">
              <InlineToolCalls />
              {streamingContent ? (
                <div className={PROSE_CLASSES}>
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm]}
                    rehypePlugins={[rehypeHighlight]}
                    components={markdownComponents}
                  >
                    {streamingContent}
                  </ReactMarkdown>
                </div>
              ) : (
                <div className="space-y-2 pt-1">
                  {statusText ? (
                    <div className="flex items-center gap-2 text-xs text-muted-foreground">
                      <Loader2 className="h-3.5 w-3.5 animate-spin shrink-0" />
                      <span>{statusText}</span>
                    </div>
                  ) : (
                    <>
                      <Skeleton className="h-4 w-64" />
                      <Skeleton className="h-4 w-48" />
                      <Skeleton className="h-4 w-56" />
                    </>
                  )}
                </div>
              )}
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>
    </div>
  );
}

function MessageBubble({
  role,
  content,
}: {
  role: string;
  content: string;
}) {
  const isUser = role === "user";
  const { images, files } = extractDownloads(content);

  return (
    <div className={cn("flex gap-3", isUser && "flex-row-reverse")}>
      <div
        className={cn(
          "flex h-8 w-8 shrink-0 items-center justify-center rounded-full",
          isUser ? "bg-secondary" : "bg-primary/10"
        )}
      >
        {isUser ? (
          <User className="h-4 w-4" />
        ) : (
          <Bot className="h-4 w-4 text-primary" />
        )}
      </div>

      <div
        className={cn(
          "min-w-0",
          isUser
            ? "max-w-[80%] rounded-2xl bg-primary text-primary-foreground px-4 py-3 text-sm leading-relaxed"
            : "max-w-[92%] py-2"
        )}
      >
        {isUser ? (
          <p className="whitespace-pre-wrap">{content}</p>
        ) : (
          <div className={PROSE_CLASSES}>
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              rehypePlugins={[rehypeHighlight]}
              components={markdownComponents}
            >
              {content}
            </ReactMarkdown>
          </div>
        )}

        {files.length > 0 && (
          <div className="mt-3 space-y-2">
            {files.map((filename) => (
              <a
                key={filename}
                href={downloadUrl(filename)}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-2 rounded-lg border border-border bg-background px-3 py-2 text-sm transition-colors hover:bg-accent no-underline"
              >
                <Download className="h-4 w-4 text-primary" />
                <span className="truncate">{filename}</span>
                <Button variant="outline" size="sm" className="ml-auto h-7 text-xs">
                  Download
                </Button>
              </a>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
