"use client";

import { useState, useRef, useCallback, type KeyboardEvent, type DragEvent } from "react";
import {
  Send,
  Square,
  Paperclip,
  X,
  PanelLeft,
  PanelRight,
  Presentation,
  Coffee,
  Users,
  Plus,
  Globe,
  ImageIcon,
  ChevronDown,
  Brain,
  Zap,
  Sparkles,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
  DropdownMenuSeparator,
  DropdownMenuLabel,
} from "@/components/ui/dropdown-menu";
import { useChatStore, type ModelOption } from "@/stores/chat-store";
import { useChat } from "@/lib/use-chat";
import { cn } from "@/lib/utils";

const MODEL_OPTIONS: { value: ModelOption; label: string; desc: string; icon: typeof Zap }[] = [
  { value: "default", label: "GPT-5.2", desc: "Fast & capable", icon: Zap },
  { value: "thinking", label: "GPT-5.2 Thinking", desc: "Deep reasoning", icon: Brain },
  { value: "pro", label: "GPT-5.2 Pro", desc: "Flagship (1.2M ctx)", icon: Sparkles },
];

export function ChatInput() {
  const [text, setText] = useState("");
  const [dragOver, setDragOver] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const isStreaming = useChatStore((s) => s.isStreaming);
  const pendingAttachments = useChatStore((s) => s.pendingAttachments);
  const sidebarOpen = useChatStore((s) => s.sidebarOpen);
  const traceOpen = useChatStore((s) => s.traceOpen);
  const toggleSidebar = useChatStore((s) => s.toggleSidebar);
  const toggleTrace = useChatStore((s) => s.toggleTrace);
  const removeAttachment = useChatStore((s) => s.removeAttachment);
  const selectedModel = useChatStore((s) => s.selectedModel);
  const setSelectedModel = useChatStore((s) => s.setSelectedModel);
  const webEnabled = useChatStore((s) => s.webEnabled);
  const toggleWeb = useChatStore((s) => s.toggleWeb);
  const activeModes = useChatStore((s) => s.activeModes);
  const addMode = useChatStore((s) => s.addMode);
  const removeMode = useChatStore((s) => s.removeMode);

  const { sendMessage, stopStreaming, handleFileUpload } = useChat();

  const currentModel = MODEL_OPTIONS.find((m) => m.value === selectedModel) ?? MODEL_OPTIONS[0];

  const handleSend = useCallback(() => {
    if (!text.trim() || isStreaming) return;
    sendMessage(text.trim());
    setText("");
    textareaRef.current?.focus();
  }, [text, isStreaming, sendMessage]);

  const handleKeyDown = useCallback(
    (e: KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    },
    [handleSend]
  );

  const handleFiles = useCallback(
    async (files: FileList | File[]) => {
      for (const file of Array.from(files)) {
        try {
          await handleFileUpload(file);
        } catch {
          // upload error handled in hook
        }
      }
    },
    [handleFileUpload]
  );

  const handleDrop = useCallback(
    (e: DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      if (e.dataTransfer.files.length > 0) {
        handleFiles(e.dataTransfer.files);
      }
    },
    [handleFiles]
  );

  const quickAction = useCallback(
    (prompt: string) => {
      setText(prompt);
      textareaRef.current?.focus();
    },
    []
  );

  return (
    <div className="border-t border-border bg-card">
      {/* Top row: + menu, model selector, web toggle */}
      <div className="flex items-center gap-2 px-4 pt-3">
        {/* "+" menu */}
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="outline" size="sm" className="h-8 w-8 p-0">
              <Plus className="h-4 w-4" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="start" className="w-48">
            <DropdownMenuLabel className="text-xs text-muted-foreground">Create</DropdownMenuLabel>
            <DropdownMenuItem
              onClick={() => addMode("image")}
              disabled={activeModes.includes("image")}
            >
              <ImageIcon className="mr-2 h-4 w-4" />
              Create Image
            </DropdownMenuItem>
            <DropdownMenuItem
              onClick={() => addMode("pptx")}
              disabled={activeModes.includes("pptx")}
            >
              <Presentation className="mr-2 h-4 w-4" />
              Create PowerPoint
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>

        {/* Model selector */}
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="outline" size="sm" className="h-8 gap-1.5 text-xs">
              <currentModel.icon className="h-3.5 w-3.5" />
              {currentModel.label}
              <ChevronDown className="h-3 w-3 opacity-50" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="start" className="w-56">
            <DropdownMenuLabel className="text-xs text-muted-foreground">Model</DropdownMenuLabel>
            {MODEL_OPTIONS.map((opt) => (
              <DropdownMenuItem
                key={opt.value}
                onClick={() => setSelectedModel(opt.value)}
                className={selectedModel === opt.value ? "bg-accent" : ""}
              >
                <opt.icon className="mr-2 h-4 w-4" />
                <div className="flex flex-col">
                  <span className="text-sm">{opt.label}</span>
                  <span className="text-xs text-muted-foreground">{opt.desc}</span>
                </div>
              </DropdownMenuItem>
            ))}
          </DropdownMenuContent>
        </DropdownMenu>

        <div className="h-5 w-px bg-border mx-1" />

        {/* Web search toggle */}
        <Button
          variant={webEnabled ? "outline" : "ghost"}
          size="sm"
          className={cn(
            "h-8 gap-1.5 text-xs transition-colors",
            webEnabled
              ? "border-primary/30 bg-primary/10 text-primary hover:bg-primary/20"
              : "text-muted-foreground hover:text-foreground"
          )}
          onClick={toggleWeb}
        >
          <Globe className="h-3.5 w-3.5" />
          {webEnabled ? "Web" : "Web off"}
        </Button>
      </div>

      {/* Mode chips + attachment chips */}
      {(activeModes.length > 0 || pendingAttachments.length > 0) && (
        <div className="flex flex-wrap gap-2 px-4 pt-2">
          {activeModes.map((mode) => (
            <div
              key={mode}
              className="flex items-center gap-1.5 rounded-full bg-primary/10 px-3 py-1 text-xs font-medium text-primary"
            >
              {mode === "image" ? (
                <ImageIcon className="h-3 w-3" />
              ) : (
                <Presentation className="h-3 w-3" />
              )}
              {mode === "image" ? "Create Image" : "Create PowerPoint"}
              <button
                onClick={() => removeMode(mode)}
                className="ml-0.5 rounded-full p-0.5 hover:bg-primary/20"
              >
                <X className="h-3 w-3" />
              </button>
            </div>
          ))}
          {pendingAttachments.map((att) => (
            <div
              key={att.stored_name}
              className="flex items-center gap-1.5 rounded-md bg-muted px-2.5 py-1 text-xs"
            >
              <Paperclip className="h-3 w-3" />
              <span className="max-w-32 truncate">{att.filename}</span>
              <button
                onClick={() => removeAttachment(att.stored_name)}
                className="ml-1 rounded-full p-0.5 hover:bg-background"
              >
                <X className="h-3 w-3" />
              </button>
            </div>
          ))}
        </div>
      )}

      {/* Input row */}
      <div
        className={`flex items-end gap-2 p-4 ${dragOver ? "ring-2 ring-primary ring-inset rounded-lg" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
      >
        <div className="flex gap-1">
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="ghost"
                size="icon"
                className="h-9 w-9"
                onClick={toggleSidebar}
              >
                <PanelLeft className={`h-4 w-4 ${sidebarOpen ? "text-primary" : ""}`} />
              </Button>
            </TooltipTrigger>
            <TooltipContent>Toggle sidebar</TooltipContent>
          </Tooltip>

          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="ghost"
                size="icon"
                className="h-9 w-9"
                onClick={() => fileInputRef.current?.click()}
              >
                <Paperclip className="h-4 w-4" />
              </Button>
            </TooltipTrigger>
            <TooltipContent>Attach file</TooltipContent>
          </Tooltip>
          <input
            ref={fileInputRef}
            type="file"
            className="hidden"
            multiple
            accept="*/*"
            onChange={(e) => {
              if (e.target.files) handleFiles(e.target.files);
              e.target.value = "";
            }}
          />
        </div>

        <Textarea
          ref={textareaRef}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask MemoryOS anything..."
          className="min-h-[40px] max-h-[200px] resize-none"
          rows={1}
          disabled={isStreaming}
        />

        <div className="flex gap-1">
          {isStreaming ? (
            <Button variant="destructive" size="icon" className="h-9 w-9" onClick={stopStreaming}>
              <Square className="h-4 w-4" />
            </Button>
          ) : (
            <Button
              size="icon"
              className="h-9 w-9"
              onClick={handleSend}
              disabled={!text.trim()}
            >
              <Send className="h-4 w-4" />
            </Button>
          )}

          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="ghost"
                size="icon"
                className="h-9 w-9"
                onClick={toggleTrace}
              >
                <PanelRight className={`h-4 w-4 ${traceOpen ? "text-primary" : ""}`} />
              </Button>
            </TooltipTrigger>
            <TooltipContent>Toggle trace panel</TooltipContent>
          </Tooltip>
        </div>
      </div>

      {/* Quick actions */}
      <div className="flex gap-2 px-4 pb-3">
        <Button
          variant="outline"
          size="sm"
          className="h-7 text-xs gap-1.5"
          onClick={() => quickAction("Build me a slide deck on ")}
        >
          <Presentation className="h-3 w-3" />
          Build Slides
        </Button>
        <Button
          variant="outline"
          size="sm"
          className="h-7 text-xs gap-1.5"
          onClick={() => quickAction("Run the morning-brief skill")}
        >
          <Coffee className="h-3 w-3" />
          Morning Brief
        </Button>
        <Button
          variant="outline"
          size="sm"
          className="h-7 text-xs gap-1.5"
          onClick={() => quickAction("Prep me for my next meeting")}
        >
          <Users className="h-3 w-3" />
          Meeting Prep
        </Button>
      </div>
    </div>
  );
}
