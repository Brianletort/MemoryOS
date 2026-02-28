import { create } from "zustand";
import type { Message, Session, FileAttachment } from "@/lib/api";

export type ModelOption = "default" | "thinking" | "pro";

export interface TraceEvent {
  type: "tool_call" | "tool_result" | "status";
  name: string;
  data: string;
  timestamp: number;
}

interface ChatState {
  sessions: Session[];
  activeSessionId: string | null;
  messages: Message[];
  streamingContent: string;
  isStreaming: boolean;
  statusText: string;
  traceEvents: TraceEvent[];
  streamingTraceEvents: TraceEvent[];
  pendingAttachments: FileAttachment[];
  sidebarOpen: boolean;
  traceOpen: boolean;
  selectedModel: ModelOption;
  webEnabled: boolean;
  activeModes: string[];

  setSessions: (sessions: Session[]) => void;
  addSession: (session: Session) => void;
  removeSession: (id: string) => void;
  updateSessionInList: (id: string, data: Partial<Session>) => void;
  setActiveSession: (id: string | null) => void;
  setMessages: (messages: Message[]) => void;
  addMessage: (message: Message) => void;
  appendStreamContent: (text: string) => void;
  finalizeStream: (content: string) => void;
  setIsStreaming: (v: boolean) => void;
  setStatusText: (text: string) => void;
  addTraceEvent: (event: TraceEvent) => void;
  clearTrace: () => void;
  addStreamingTraceEvent: (event: TraceEvent) => void;
  clearStreamingTrace: () => void;
  addAttachment: (att: FileAttachment) => void;
  removeAttachment: (storedName: string) => void;
  clearAttachments: () => void;
  toggleSidebar: () => void;
  toggleTrace: () => void;
  setSelectedModel: (model: ModelOption) => void;
  toggleWeb: () => void;
  addMode: (mode: string) => void;
  removeMode: (mode: string) => void;
  clearModes: () => void;
}

export const useChatStore = create<ChatState>((set) => ({
  sessions: [],
  activeSessionId: null,
  messages: [],
  streamingContent: "",
  isStreaming: false,
  statusText: "",
  traceEvents: [],
  streamingTraceEvents: [],
  pendingAttachments: [],
  sidebarOpen: true,
  traceOpen: false,
  selectedModel: "default",
  webEnabled: true,
  activeModes: [],

  setSessions: (sessions) => set({ sessions }),

  addSession: (session) =>
    set((s) => ({ sessions: [session, ...s.sessions] })),

  removeSession: (id) =>
    set((s) => ({
      sessions: s.sessions.filter((x) => x.id !== id),
      activeSessionId: s.activeSessionId === id ? null : s.activeSessionId,
      messages: s.activeSessionId === id ? [] : s.messages,
    })),

  updateSessionInList: (id, data) =>
    set((s) => ({
      sessions: s.sessions.map((x) => (x.id === id ? { ...x, ...data } : x)),
    })),

  setActiveSession: (id) => set({ activeSessionId: id }),

  setMessages: (messages) => set({ messages }),

  addMessage: (message) =>
    set((s) => ({ messages: [...s.messages, message] })),

  appendStreamContent: (text) =>
    set((s) => ({ streamingContent: s.streamingContent + text })),

  finalizeStream: (content) =>
    set((s) => ({
      streamingContent: "",
      messages: [
        ...s.messages,
        {
          id: `stream-${Date.now()}`,
          session_id: s.activeSessionId || "",
          role: "assistant" as const,
          content,
          created_at: new Date().toISOString(),
        },
      ],
    })),

  setIsStreaming: (v) => set({ isStreaming: v }),

  setStatusText: (text) => set({ statusText: text }),

  addTraceEvent: (event) =>
    set((s) => ({ traceEvents: [...s.traceEvents, event] })),

  clearTrace: () => set({ traceEvents: [] }),

  addStreamingTraceEvent: (event) =>
    set((s) => ({ streamingTraceEvents: [...s.streamingTraceEvents, event] })),

  clearStreamingTrace: () => set({ streamingTraceEvents: [] }),

  addAttachment: (att) =>
    set((s) => ({ pendingAttachments: [...s.pendingAttachments, att] })),

  removeAttachment: (storedName) =>
    set((s) => ({
      pendingAttachments: s.pendingAttachments.filter(
        (a) => a.stored_name !== storedName
      ),
    })),

  clearAttachments: () => set({ pendingAttachments: [] }),

  toggleSidebar: () => set((s) => ({ sidebarOpen: !s.sidebarOpen })),

  toggleTrace: () => set((s) => ({ traceOpen: !s.traceOpen })),

  setSelectedModel: (model) => set({ selectedModel: model }),

  toggleWeb: () => set((s) => ({ webEnabled: !s.webEnabled })),

  addMode: (mode) =>
    set((s) => ({
      activeModes: s.activeModes.includes(mode)
        ? s.activeModes
        : [...s.activeModes, mode],
    })),

  removeMode: (mode) =>
    set((s) => ({
      activeModes: s.activeModes.filter((m) => m !== mode),
    })),

  clearModes: () => set({ activeModes: [] }),
}));
