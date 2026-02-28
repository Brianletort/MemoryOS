const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8765";

export interface Session {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  pinned: number;
}

export interface Message {
  id: string;
  session_id: string;
  role: "user" | "assistant" | "tool";
  content: string;
  tool_calls?: ToolCall[];
  created_at: string;
}

export interface ToolCall {
  name: string;
  args: Record<string, unknown>;
}

export interface SessionWithMessages extends Session {
  messages: Message[];
}

export interface UploadResult {
  file_id: string;
  filename: string;
  stored_name: string;
  size: number;
  text_preview: string;
  text_length: number;
}

export interface FileAttachment {
  stored_name: string;
  filename: string;
}

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...init?.headers,
    },
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API ${res.status}: ${text}`);
  }
  return res.json();
}

export async function listSessions(): Promise<Session[]> {
  return apiFetch("/api/sessions");
}

export async function createSession(title?: string): Promise<Session> {
  return apiFetch("/api/sessions", {
    method: "POST",
    body: JSON.stringify({ title: title || "New Chat" }),
  });
}

export async function getSession(id: string): Promise<SessionWithMessages> {
  return apiFetch(`/api/sessions/${id}`);
}

export async function updateSession(
  id: string,
  data: { title?: string; pinned?: number }
): Promise<Session> {
  return apiFetch(`/api/sessions/${id}`, {
    method: "PUT",
    body: JSON.stringify(data),
  });
}

export async function deleteSession(id: string): Promise<void> {
  await apiFetch(`/api/sessions/${id}`, { method: "DELETE" });
}

export async function uploadFile(file: File): Promise<UploadResult> {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch(`${API_BASE}/api/files/upload`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Upload failed: ${text}`);
  }
  return res.json();
}

export function downloadUrl(filename: string): string {
  return `${API_BASE}/api/files/download/${filename}`;
}

export interface ChatSSECallbacks {
  onSession?: (sessionId: string) => void;
  onStatus?: (text: string) => void;
  onToolCall?: (name: string, args: Record<string, unknown>) => void;
  onToolResult?: (name: string, output: string) => void;
  onContent?: (text: string) => void;
  onFileReady?: (url: string, filename: string) => void;
  onDone?: (sessionId: string) => void;
  onError?: (message: string) => void;
}

export interface ChatRequestOptions {
  model?: string;
  web_enabled?: boolean;
  modes?: string[];
}

export async function sendChatMessage(
  sessionId: string | null,
  message: string,
  attachments: FileAttachment[],
  callbacks: ChatSSECallbacks,
  signal?: AbortSignal,
  options?: ChatRequestOptions
): Promise<void> {
  const res = await fetch(`${API_BASE}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: sessionId,
      message,
      attachments,
      model: options?.model ?? "default",
      web_enabled: options?.web_enabled ?? true,
      modes: options?.modes ?? [],
    }),
    signal,
  });

  if (!res.ok) {
    const text = await res.text();
    callbacks.onError?.(`Chat error: ${text}`);
    return;
  }

  const reader = res.body?.getReader();
  if (!reader) {
    callbacks.onError?.("No response body");
    return;
  }

  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    let eventType = "";
    for (const line of lines) {
      if (line.startsWith("event: ")) {
        eventType = line.slice(7).trim();
      } else if (line.startsWith("data: ")) {
        const raw = line.slice(6);
        try {
          const data = JSON.parse(raw);
          switch (eventType) {
            case "session":
              callbacks.onSession?.(data.session_id);
              break;
            case "status":
              callbacks.onStatus?.(data.text);
              break;
            case "tool_call":
              callbacks.onToolCall?.(data.name, data.args);
              break;
            case "tool_result":
              callbacks.onToolResult?.(data.name, data.output);
              break;
            case "content":
              callbacks.onContent?.(data.text);
              break;
            case "file_ready":
              callbacks.onFileReady?.(data.url, data.filename);
              break;
            case "done":
              callbacks.onDone?.(data.session_id);
              break;
            case "error":
              callbacks.onError?.(data.message);
              break;
          }
        } catch {
          // skip malformed JSON
        }
      }
    }
  }
}
