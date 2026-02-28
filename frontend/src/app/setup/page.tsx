"use client";

import { useEffect, useState, useCallback } from "react";
import { CheckCircle2, Circle, Loader2, AlertCircle, ArrowRight, Rocket, FolderOpen, Brain, Zap } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { GlassCard } from "@/components/ui/glass-card";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8765";

interface DepCheck {
  name: string;
  status: "ok" | "missing" | "warning";
  detail: string;
}

interface SetupStatus {
  dependencies: DepCheck[];
  config_exists: boolean;
  vault_path: string;
  agents_installed: boolean;
}

const STEPS = ["Dependencies", "Configuration", "AI Agents", "Activate", "Done"];

export default function SetupPage() {
  const [step, setStep] = useState(0);
  const [status, setStatus] = useState<SetupStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [acting, setActing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [vaultPath, setVaultPath] = useState("");
  const [emailSource, setEmailSource] = useState("mail_app");
  const [calendarSource, setCalendarSource] = useState("calendar_app");
  const [provider, setProvider] = useState("openai");
  const [model, setModel] = useState("gpt-5.2");
  const [apiKey, setApiKey] = useState("");

  const loadStatus = useCallback(async () => {
    try {
      const r = await fetch(`${API}/api/setup/status`);
      const d = await r.json();
      setStatus(d);
      if (d.vault_path) setVaultPath(d.vault_path);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { loadStatus(); }, [loadStatus]);

  const saveConfig = async () => {
    setActing(true);
    setError(null);
    try {
      await fetch(`${API}/api/setup/config`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          obsidian_vault: vaultPath,
          email_source: emailSource,
          calendar_source: calendarSource,
        }),
      });
      setStep(2);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setActing(false);
    }
  };

  const saveAgentConfig = async () => {
    setActing(true);
    setError(null);
    try {
      await fetch(`${API}/api/agents/config`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ provider, model }),
      });
      setStep(3);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setActing(false);
    }
  };

  const activate = async () => {
    setActing(true);
    setError(null);
    try {
      await fetch(`${API}/api/setup/install-agents`, { method: "POST" });
      await fetch(`${API}/api/setup/reindex`, { method: "POST" });
      setStep(4);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setActing(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
      </div>
    );
  }

  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto max-w-3xl px-6 py-8 space-y-6">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold mb-2">MemoryOS Setup</h1>
          <p className="text-muted-foreground">Configure your personal AI assistant in a few steps</p>
        </div>

        {/* Step indicators */}
        <div className="flex items-center justify-center gap-2 mb-8">
          {STEPS.map((label, i) => (
            <div key={label} className="flex items-center gap-2">
              <button
                onClick={() => i < step && setStep(i)}
                className={`flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-medium transition-colors ${
                  i === step ? "bg-primary text-primary-foreground" :
                  i < step ? "bg-green-400/15 text-green-400 cursor-pointer" :
                  "bg-muted text-muted-foreground"
                }`}
              >
                {i < step ? <CheckCircle2 className="h-3.5 w-3.5" /> : <Circle className="h-3.5 w-3.5" />}
                {label}
              </button>
              {i < STEPS.length - 1 && <ArrowRight className="h-3 w-3 text-muted-foreground" />}
            </div>
          ))}
        </div>

        {error && (
          <div className="rounded-xl border border-red-400/30 bg-red-400/10 px-4 py-3 text-sm text-red-300 flex items-center gap-2">
            <AlertCircle className="h-4 w-4 shrink-0" /> {error}
          </div>
        )}

        {/* Step 0: Dependencies */}
        {step === 0 && status && (
          <GlassCard label="System Dependencies">
            <div className="space-y-3">
              {status.dependencies.map((dep) => (
                <div key={dep.name} className="flex items-center gap-3">
                  {dep.status === "ok" ? (
                    <CheckCircle2 className="h-5 w-5 text-green-400 shrink-0" />
                  ) : dep.status === "warning" ? (
                    <AlertCircle className="h-5 w-5 text-yellow-400 shrink-0" />
                  ) : (
                    <Circle className="h-5 w-5 text-red-400 shrink-0" />
                  )}
                  <div className="flex-1">
                    <div className="font-medium text-sm">{dep.name}</div>
                    <div className="text-xs text-muted-foreground">{dep.detail}</div>
                  </div>
                </div>
              ))}
            </div>
            <div className="mt-4">
              <Button onClick={() => setStep(1)} className="gap-2">
                Continue <ArrowRight className="h-4 w-4" />
              </Button>
            </div>
          </GlassCard>
        )}

        {/* Step 1: Configuration */}
        {step === 1 && (
          <GlassCard label="Path Configuration">
            <div className="space-y-4">
              <div>
                <label className="text-sm font-medium flex items-center gap-2 mb-1.5">
                  <FolderOpen className="h-4 w-4 text-primary" /> Obsidian Vault Path
                </label>
                <Input value={vaultPath} onChange={(e) => setVaultPath(e.target.value)} placeholder="~/Documents/Obsidian/MyVault" className="text-sm" />
                <p className="text-xs text-muted-foreground mt-1">Where all Markdown output lands</p>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="text-sm font-medium mb-1.5 block">Email Source</label>
                  <select value={emailSource} onChange={(e) => setEmailSource(e.target.value)} className="w-full h-9 rounded-md border border-border bg-background px-3 text-sm">
                    <option value="mail_app">macOS Mail.app</option>
                    <option value="outlook">Outlook (local)</option>
                    <option value="graph">Microsoft Graph API</option>
                  </select>
                </div>
                <div>
                  <label className="text-sm font-medium mb-1.5 block">Calendar Source</label>
                  <select value={calendarSource} onChange={(e) => setCalendarSource(e.target.value)} className="w-full h-9 rounded-md border border-border bg-background px-3 text-sm">
                    <option value="calendar_app">macOS Calendar.app</option>
                    <option value="outlook">Outlook (local)</option>
                    <option value="graph">Microsoft Graph API</option>
                  </select>
                </div>
              </div>
            </div>
            <div className="mt-4 flex gap-2">
              <Button variant="outline" onClick={() => setStep(0)}>Back</Button>
              <Button onClick={saveConfig} disabled={acting || !vaultPath.trim()} className="gap-2">
                {acting && <Loader2 className="h-4 w-4 animate-spin" />} Continue <ArrowRight className="h-4 w-4" />
              </Button>
            </div>
          </GlassCard>
        )}

        {/* Step 2: AI Agents */}
        {step === 2 && (
          <GlassCard label="AI Agent Configuration">
            <div className="space-y-4">
              <div>
                <label className="text-sm font-medium flex items-center gap-2 mb-1.5">
                  <Brain className="h-4 w-4 text-primary" /> LLM Provider
                </label>
                <select value={provider} onChange={(e) => setProvider(e.target.value)} className="w-full h-9 rounded-md border border-border bg-background px-3 text-sm">
                  <option value="openai">OpenAI</option>
                  <option value="anthropic">Anthropic</option>
                  <option value="google">Google Gemini</option>
                  <option value="ollama">Ollama (local)</option>
                </select>
              </div>
              <div>
                <label className="text-sm font-medium mb-1.5 block">Model</label>
                <Input value={model} onChange={(e) => setModel(e.target.value)} placeholder="gpt-5.2" className="text-sm" />
              </div>
              <div>
                <label className="text-sm font-medium mb-1.5 block">API Key (set in .env.local)</label>
                <Input type="password" value={apiKey} onChange={(e) => setApiKey(e.target.value)} placeholder="sk-..." className="text-sm" />
                <p className="text-xs text-muted-foreground mt-1">Stored in .env.local, not in config.yaml</p>
              </div>
            </div>
            <div className="mt-4 flex gap-2">
              <Button variant="outline" onClick={() => setStep(1)}>Back</Button>
              <Button onClick={saveAgentConfig} disabled={acting} className="gap-2">
                {acting && <Loader2 className="h-4 w-4 animate-spin" />} Continue <ArrowRight className="h-4 w-4" />
              </Button>
            </div>
          </GlassCard>
        )}

        {/* Step 3: Activate */}
        {step === 3 && (
          <GlassCard label="Activate MemoryOS">
            <div className="text-center py-4">
              <Zap className="h-12 w-12 text-primary mx-auto mb-4" />
              <h2 className="text-xl font-semibold mb-2">Ready to Activate</h2>
              <p className="text-sm text-muted-foreground mb-6">
                This will install LaunchD agents, run initial extraction, and build the search index.
              </p>
              <div className="flex justify-center gap-2">
                <Button variant="outline" onClick={() => setStep(2)}>Back</Button>
                <Button onClick={activate} disabled={acting} className="gap-2" size="lg">
                  {acting ? <Loader2 className="h-4 w-4 animate-spin" /> : <Rocket className="h-4 w-4" />}
                  {acting ? "Activating..." : "Activate"}
                </Button>
              </div>
            </div>
          </GlassCard>
        )}

        {/* Step 4: Done */}
        {step === 4 && (
          <GlassCard>
            <div className="text-center py-8">
              <CheckCircle2 className="h-16 w-16 text-green-400 mx-auto mb-4" />
              <h2 className="text-2xl font-bold mb-2">MemoryOS is Active</h2>
              <p className="text-muted-foreground mb-6">
                Your personal AI assistant is now running. Extractors are collecting data and agents are scheduled.
              </p>
              <div className="flex justify-center gap-3">
                <a href="/dashboard">
                  <Button variant="outline">View Dashboard</Button>
                </a>
                <a href="/">
                  <Button className="gap-2">
                    <Rocket className="h-4 w-4" /> Start Chatting
                  </Button>
                </a>
              </div>
            </div>
          </GlassCard>
        )}
      </div>
    </div>
  );
}
