"use client";

import { useEffect, useState, useCallback } from "react";
import { Settings, ExternalLink, Loader2, Plus, X, Wifi, AppWindow, Mic, Brain, Mail, CheckCircle2, AlertCircle } from "lucide-react";
import { GlassCard } from "@/components/ui/glass-card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8765";

interface AgentConfig {
  provider: string;
  model: string;
  synthesis_model?: string;
  reasoning_effort?: string;
  api_base?: string;
  temperature?: number;
  skills_dir: string;
  reports_dir: string;
  email?: { enabled: boolean; delivery_method: string; from: string; to: string; smtp_host: string; smtp_port: number; smtp_user: string };
}

interface SettingsData {
  trusted_networks?: string[];
  work_apps?: string[];
  audio_filter?: { min_words: number; work_hours_only: boolean; work_hours_start: string; work_hours_end: string };
  [key: string]: unknown;
}

export default function SettingsPage() {
  const [settings, setSettings] = useState<SettingsData | null>(null);
  const [agentCfg, setAgentCfg] = useState<AgentConfig | null>(null);
  const [privacy, setPrivacy] = useState<{ paused: boolean } | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [testResult, setTestResult] = useState<{ type: string; ok: boolean; detail: string } | null>(null);
  const [newNetwork, setNewNetwork] = useState("");
  const [newApp, setNewApp] = useState("");

  const load = useCallback(() => {
    Promise.all([
      fetch(`${API}/api/settings`).then((r) => r.json()).catch(() => null),
      fetch(`${API}/api/agents/config`).then((r) => r.json()).catch(() => null),
      fetch(`${API}/api/privacy`).then((r) => r.json()).catch(() => null),
    ]).then(([s, a, p]) => {
      setSettings(s);
      setAgentCfg(a);
      setPrivacy(p);
    }).finally(() => setLoading(false));
  }, []);

  useEffect(() => { load(); }, [load]);

  const togglePrivacy = async () => {
    try {
      const res = await fetch(`${API}/api/privacy/toggle`, { method: "POST" });
      setPrivacy(await res.json());
    } catch {}
  };

  const saveSettings = async (updates: Partial<SettingsData>) => {
    setSaving(true);
    try {
      await fetch(`${API}/api/settings`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(updates),
      });
      load();
    } catch {}
    finally { setSaving(false); }
  };

  const saveAgentConfig = async (updates: Partial<AgentConfig>) => {
    setSaving(true);
    try {
      await fetch(`${API}/api/agents/config`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(updates),
      });
      load();
    } catch {}
    finally { setSaving(false); }
  };

  const testLLM = async () => {
    setTestResult(null);
    try {
      const r = await fetch(`${API}/api/agents/test-llm`, { method: "POST" });
      const d = await r.json();
      setTestResult({ type: "llm", ok: d.ok, detail: d.detail || d.error || "" });
    } catch (e) {
      setTestResult({ type: "llm", ok: false, detail: (e as Error).message });
    }
  };

  const testEmail = async () => {
    setTestResult(null);
    try {
      const r = await fetch(`${API}/api/agents/test-email`, { method: "POST" });
      const d = await r.json();
      setTestResult({ type: "email", ok: d.ok, detail: d.detail || d.error || "" });
    } catch (e) {
      setTestResult({ type: "email", ok: false, detail: (e as Error).message });
    }
  };

  const addNetwork = () => {
    if (!newNetwork.trim() || !settings) return;
    const nets = [...(settings.trusted_networks || []), newNetwork.trim()];
    saveSettings({ trusted_networks: nets });
    setNewNetwork("");
  };

  const removeNetwork = (n: string) => {
    if (!settings) return;
    saveSettings({ trusted_networks: (settings.trusted_networks || []).filter((x) => x !== n) });
  };

  const addApp = () => {
    if (!newApp.trim() || !settings) return;
    const apps = [...(settings.work_apps || []), newApp.trim()];
    saveSettings({ work_apps: apps });
    setNewApp("");
  };

  const removeApp = (a: string) => {
    if (!settings) return;
    saveSettings({ work_apps: (settings.work_apps || []).filter((x) => x !== a) });
  };

  if (loading) return <div className="flex items-center justify-center h-full"><Loader2 className="h-6 w-6 animate-spin text-muted-foreground" /></div>;

  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto max-w-4xl px-6 py-5 space-y-5">
        <div className="flex items-center gap-3">
          <Settings className="h-6 w-6 text-primary" />
          <h1 className="text-2xl font-semibold">Settings</h1>
          <a href={`${API}/#settings`} target="_blank" rel="noopener noreferrer" className="ml-auto inline-flex items-center gap-1.5 rounded-lg border border-border px-3 py-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors">
            <ExternalLink className="h-3.5 w-3.5" /> Legacy
          </a>
        </div>

        {testResult && (
          <div className={`rounded-xl border px-4 py-3 text-sm flex items-center gap-2 ${testResult.ok ? "border-green-400/30 bg-green-400/10 text-green-300" : "border-red-400/30 bg-red-400/10 text-red-300"}`}>
            {testResult.ok ? <CheckCircle2 className="h-4 w-4" /> : <AlertCircle className="h-4 w-4" />}
            {testResult.type === "llm" ? "LLM" : "Email"}: {testResult.detail}
          </div>
        )}

        {/* Privacy */}
        <GlassCard label="Privacy">
          <div className="flex items-center justify-between">
            <div>
              <div className="flex items-center gap-2">
                <span className={`w-2.5 h-2.5 rounded-full ${privacy?.paused ? "bg-red-400" : "bg-green-400"}`} />
                <span className="font-medium">{privacy?.paused ? "Audio Paused" : "Audio Recording"}</span>
              </div>
              <p className="text-xs text-muted-foreground mt-1">Controls Screenpipe audio transcriptions</p>
            </div>
            <Button variant={privacy?.paused ? "outline" : "destructive"} size="sm" onClick={togglePrivacy}>
              {privacy?.paused ? "Resume" : "Pause Audio"}
            </Button>
          </div>
        </GlassCard>

        {/* Trusted Networks */}
        <GlassCard label="Trusted WiFi Networks">
          <div className="flex items-center gap-2 mb-3">
            <Wifi className="h-4 w-4 text-primary" />
            <p className="text-xs text-muted-foreground">Privacy mode auto-enables on untrusted networks</p>
          </div>
          <div className="flex flex-wrap gap-2 mb-3">
            {(settings?.trusted_networks || []).map((n) => (
              <span key={n} className="inline-flex items-center gap-1.5 rounded-full bg-muted px-3 py-1 text-xs">
                {n}
                <button onClick={() => removeNetwork(n)} className="hover:text-red-400"><X className="h-3 w-3" /></button>
              </span>
            ))}
          </div>
          <div className="flex gap-2">
            <Input value={newNetwork} onChange={(e) => setNewNetwork(e.target.value)} placeholder="Network name" className="h-8 text-xs" onKeyDown={(e) => e.key === "Enter" && addNetwork()} />
            <Button variant="outline" size="sm" onClick={addNetwork}><Plus className="h-3.5 w-3.5" /></Button>
          </div>
        </GlassCard>

        {/* Work Apps */}
        <GlassCard label="Work Apps">
          <div className="flex items-center gap-2 mb-3">
            <AppWindow className="h-4 w-4 text-primary" />
            <p className="text-xs text-muted-foreground">Apps tracked for activity and focus analysis</p>
          </div>
          <div className="flex flex-wrap gap-2 mb-3">
            {(settings?.work_apps || []).map((a) => (
              <span key={a} className="inline-flex items-center gap-1.5 rounded-full bg-muted px-3 py-1 text-xs">
                {a}
                <button onClick={() => removeApp(a)} className="hover:text-red-400"><X className="h-3 w-3" /></button>
              </span>
            ))}
          </div>
          <div className="flex gap-2">
            <Input value={newApp} onChange={(e) => setNewApp(e.target.value)} placeholder="App name" className="h-8 text-xs" onKeyDown={(e) => e.key === "Enter" && addApp()} />
            <Button variant="outline" size="sm" onClick={addApp}><Plus className="h-3.5 w-3.5" /></Button>
          </div>
        </GlassCard>

        {/* Audio Filter */}
        {settings?.audio_filter && (
          <GlassCard label="Audio Filter">
            <div className="flex items-center gap-2 mb-3">
              <Mic className="h-4 w-4 text-primary" />
              <p className="text-xs text-muted-foreground">Filter short/noisy audio transcriptions</p>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="text-xs text-muted-foreground">Min Words</label>
                <Input type="number" defaultValue={settings.audio_filter.min_words} className="h-8 text-xs mt-1" onBlur={(e) => saveSettings({ audio_filter: { ...settings.audio_filter!, min_words: parseInt(e.target.value) || 5 } })} />
              </div>
              <div>
                <label className="text-xs text-muted-foreground">Work Hours Only</label>
                <div className="mt-1">
                  <Button variant={settings.audio_filter.work_hours_only ? "default" : "outline"} size="sm" className="text-xs" onClick={() => saveSettings({ audio_filter: { ...settings.audio_filter!, work_hours_only: !settings.audio_filter!.work_hours_only } })}>
                    {settings.audio_filter.work_hours_only ? "Enabled" : "Disabled"}
                  </Button>
                </div>
              </div>
              <div>
                <label className="text-xs text-muted-foreground">Start Time</label>
                <Input type="time" defaultValue={settings.audio_filter.work_hours_start} className="h-8 text-xs mt-1" onBlur={(e) => saveSettings({ audio_filter: { ...settings.audio_filter!, work_hours_start: e.target.value } })} />
              </div>
              <div>
                <label className="text-xs text-muted-foreground">End Time</label>
                <Input type="time" defaultValue={settings.audio_filter.work_hours_end} className="h-8 text-xs mt-1" onBlur={(e) => saveSettings({ audio_filter: { ...settings.audio_filter!, work_hours_end: e.target.value } })} />
              </div>
            </div>
          </GlassCard>
        )}

        {/* AI Provider */}
        {agentCfg && (
          <GlassCard label="AI Provider">
            <div className="flex items-center gap-2 mb-3">
              <Brain className="h-4 w-4 text-primary" />
              <p className="text-xs text-muted-foreground">LLM configuration for agent skills</p>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="text-xs text-muted-foreground">Provider</label>
                <select defaultValue={agentCfg.provider} className="mt-1 w-full h-8 rounded-md border border-border bg-background px-2 text-xs" onChange={(e) => saveAgentConfig({ provider: e.target.value })}>
                  <option value="openai">OpenAI</option>
                  <option value="anthropic">Anthropic</option>
                  <option value="google">Google</option>
                  <option value="azure">Azure</option>
                  <option value="ollama">Ollama</option>
                </select>
              </div>
              <div>
                <label className="text-xs text-muted-foreground">Model</label>
                <Input defaultValue={agentCfg.model} className="h-8 text-xs mt-1" onBlur={(e) => saveAgentConfig({ model: e.target.value })} />
              </div>
              <div>
                <label className="text-xs text-muted-foreground">Reasoning Effort</label>
                <select defaultValue={agentCfg.reasoning_effort || "high"} className="mt-1 w-full h-8 rounded-md border border-border bg-background px-2 text-xs" onChange={(e) => saveAgentConfig({ reasoning_effort: e.target.value } as Partial<AgentConfig>)}>
                  <option value="none">None</option>
                  <option value="low">Low</option>
                  <option value="medium">Medium</option>
                  <option value="high">High</option>
                </select>
              </div>
              <div>
                <label className="text-xs text-muted-foreground">Temperature</label>
                <Input type="number" step="0.1" min="0" max="2" defaultValue={agentCfg.temperature ?? 0.3} className="h-8 text-xs mt-1" onBlur={(e) => saveAgentConfig({ temperature: parseFloat(e.target.value) || 0.3 })} />
              </div>
              <div className="col-span-2">
                <label className="text-xs text-muted-foreground">API Base (optional, for Ollama/Azure)</label>
                <Input defaultValue={agentCfg.api_base || ""} placeholder="http://localhost:11434" className="h-8 text-xs mt-1" onBlur={(e) => saveAgentConfig({ api_base: e.target.value || undefined } as Partial<AgentConfig>)} />
              </div>
            </div>
            <div className="mt-3">
              <Button variant="outline" size="sm" className="text-xs gap-1.5" onClick={testLLM} disabled={saving}>
                <Brain className="h-3.5 w-3.5" /> Test LLM Connection
              </Button>
            </div>
          </GlassCard>
        )}

        {/* Email Delivery */}
        {agentCfg?.email && (
          <GlassCard label="Email Delivery">
            <div className="flex items-center gap-2 mb-3">
              <Mail className="h-4 w-4 text-primary" />
              <p className="text-xs text-muted-foreground">SMTP settings for skill report delivery</p>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="text-xs text-muted-foreground">SMTP Host</label>
                <Input defaultValue={agentCfg.email.smtp_host} className="h-8 text-xs mt-1" />
              </div>
              <div>
                <label className="text-xs text-muted-foreground">SMTP Port</label>
                <Input type="number" defaultValue={agentCfg.email.smtp_port} className="h-8 text-xs mt-1" />
              </div>
              <div>
                <label className="text-xs text-muted-foreground">From</label>
                <Input defaultValue={agentCfg.email.from} className="h-8 text-xs mt-1" />
              </div>
              <div>
                <label className="text-xs text-muted-foreground">To</label>
                <Input defaultValue={agentCfg.email.to} className="h-8 text-xs mt-1" />
              </div>
            </div>
            <div className="mt-3">
              <Button variant="outline" size="sm" className="text-xs gap-1.5" onClick={testEmail} disabled={saving}>
                <Mail className="h-3.5 w-3.5" /> Send Test Email
              </Button>
            </div>
          </GlassCard>
        )}

        {/* Connection Info */}
        <GlassCard label="Connection">
          <div className="space-y-2 text-sm">
            {[
              ["Backend URL", API],
              ["Frontend Port", "3000"],
              ["Chat Storage", "SQLite + Obsidian 95_chat/"],
            ].map(([label, value]) => (
              <div key={label} className="flex items-center justify-between py-1.5 border-b border-[rgba(255,255,255,.04)] last:border-0">
                <span className="text-muted-foreground">{label}</span>
                <code className="rounded bg-background px-2 py-0.5 text-xs">{value}</code>
              </div>
            ))}
          </div>
        </GlassCard>
      </div>
    </div>
  );
}
