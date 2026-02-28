"use client";

import { useCallback, useRef } from "react";
import { useChatStore } from "@/stores/chat-store";
import {
  sendChatMessage,
  listSessions,
  getSession,
  deleteSession,
  updateSession,
  uploadFile,
} from "@/lib/api";
import type { FileAttachment } from "@/lib/api";

function getStore() {
  return useChatStore.getState();
}

export function useChat() {
  const abortRef = useRef<AbortController | null>(null);

  const loadSessions = useCallback(async () => {
    const sessions = await listSessions();
    getStore().setSessions(sessions);
  }, []);

  const selectSession = useCallback(async (id: string) => {
    getStore().setActiveSession(id);
    getStore().clearTrace();
    getStore().clearStreamingTrace();
    const data = await getSession(id);
    getStore().setMessages(data.messages || []);
  }, []);

  const newChat = useCallback(() => {
    getStore().setActiveSession(null);
    getStore().setMessages([]);
    getStore().clearTrace();
    getStore().clearStreamingTrace();
    getStore().clearAttachments();
    getStore().clearModes();
  }, []);

  const removeSession = useCallback(async (id: string) => {
    await deleteSession(id);
    getStore().removeSession(id);
  }, []);

  const pinSession = useCallback(async (id: string, pinned: boolean) => {
    const updated = await updateSession(id, { pinned: pinned ? 1 : 0 });
    getStore().updateSessionInList(id, updated);
  }, []);

  const renameSession = useCallback(async (id: string, title: string) => {
    const updated = await updateSession(id, { title });
    getStore().updateSessionInList(id, updated);
  }, []);

  const handleFileUpload = useCallback(async (file: File) => {
    const result = await uploadFile(file);
    getStore().addAttachment({
      stored_name: result.stored_name,
      filename: result.filename,
    });
    return result;
  }, []);

  const sendMessage = useCallback(async (text: string) => {
    const s = getStore();
    if (!text.trim() || s.isStreaming) return;

    const sessionId = s.activeSessionId;
    const attachments: FileAttachment[] = [...s.pendingAttachments];
    const model = s.selectedModel;
    const webEnabled = s.webEnabled;
    const modes = [...s.activeModes];

    s.addMessage({
      id: `user-${Date.now()}`,
      session_id: sessionId || "",
      role: "user",
      content: text,
      created_at: new Date().toISOString(),
    });

    s.setIsStreaming(true);
    s.setStatusText("Connecting...");
    s.clearTrace();
    s.clearStreamingTrace();
    s.clearAttachments();
    s.clearModes();

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      await sendChatMessage(
        sessionId,
        text,
        attachments,
        {
          onSession: (sid) => {
            if (!sessionId) {
              getStore().setActiveSession(sid);
              const title = text.slice(0, 60) || "New Chat";
              getStore().addSession({
                id: sid,
                title,
                created_at: new Date().toISOString(),
                updated_at: new Date().toISOString(),
                pinned: 0,
              });
            }
          },
          onStatus: (text) => {
            getStore().setStatusText(text);
            if (text) {
              getStore().addTraceEvent({
                type: "status",
                name: "status",
                data: text,
                timestamp: Date.now(),
              });
            }
          },
          onToolCall: (name, args) => {
            const event = {
              type: "tool_call" as const,
              name,
              data: JSON.stringify(args, null, 2),
              timestamp: Date.now(),
            };
            getStore().addTraceEvent(event);
            getStore().addStreamingTraceEvent(event);
          },
          onToolResult: (name, output) => {
            const event = {
              type: "tool_result" as const,
              name,
              data: output,
              timestamp: Date.now(),
            };
            getStore().addTraceEvent(event);
            getStore().addStreamingTraceEvent(event);
          },
          onContent: (chunk) => {
            getStore().setStatusText("");
            getStore().appendStreamContent(chunk);
          },
          onFileReady: () => {},
          onDone: () => {
            const content = getStore().streamingContent;
            getStore().finalizeStream(content);
            getStore().setIsStreaming(false);
            getStore().setStatusText("");
          },
          onError: (msg) => {
            getStore().finalizeStream(`Error: ${msg}`);
            getStore().setIsStreaming(false);
            getStore().setStatusText("");
          },
        },
        controller.signal,
        { model, web_enabled: webEnabled, modes }
      );
    } catch (err) {
      if ((err as Error).name !== "AbortError") {
        getStore().finalizeStream(`Error: ${(err as Error).message}`);
      }
      getStore().setIsStreaming(false);
    }

    abortRef.current = null;
  }, []);

  const stopStreaming = useCallback(() => {
    abortRef.current?.abort();
    const content = getStore().streamingContent;
    if (content) {
      getStore().finalizeStream(content);
    }
    getStore().setIsStreaming(false);
  }, []);

  return {
    loadSessions,
    selectSession,
    newChat,
    removeSession,
    pinSession,
    renameSession,
    handleFileUpload,
    sendMessage,
    stopStreaming,
  };
}
