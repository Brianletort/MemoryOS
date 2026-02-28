"use client";

import { useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Dialog, DialogContent } from "@/components/ui/dialog";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8765";

interface SkillDetailDialogProps {
  skillName: string | null;
  onClose: () => void;
}

export function SkillDetailDialog({ skillName, onClose }: SkillDetailDialogProps) {
  const [content, setContent] = useState<string>("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!skillName) return;
    setLoading(true);
    fetch(`${API}/api/agents/skill-config/${skillName}`)
      .then((r) => r.json())
      .then((d) => setContent(d.skill_md || d.content || "No SKILL.md found"))
      .catch(() => setContent("Failed to load skill details"))
      .finally(() => setLoading(false));
  }, [skillName]);

  return (
    <Dialog open={!!skillName} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="max-w-2xl max-h-[80vh] overflow-y-auto">
        <div className="prose prose-sm dark:prose-invert max-w-none">
          {loading ? (
            <p className="text-muted-foreground">Loading...</p>
          ) : (
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}
