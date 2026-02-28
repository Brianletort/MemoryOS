"use client";

import { useEffect, useState, useCallback } from "react";
import {
  ExternalLink, Cpu, HardDrive, MemoryStick, AlertTriangle,
  Wifi, Battery, BatteryCharging, Clock, ArrowUpDown, Monitor,
  Activity,
} from "lucide-react";
import { GlassCard, GradientValue } from "@/components/ui/glass-card";
import { PrivacyBar } from "@/components/dashboard/privacy-bar";
import { LaunchdTable } from "@/components/dashboard/launchd-table";
import { ExtractorCards } from "@/components/dashboard/extractor-cards";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8765";

interface SystemStatus {
  timestamp: string;
  extractors: Record<string, Record<string, unknown>>;
  vault: { total_markdown_files: number; size_mb: number };
  folders: Record<string, number>;
  launchd: Record<string, { pid: number | null; last_exit_code: number; loaded: boolean; healthy: boolean }>;
  privacy: { privacy_mode: boolean; screenpipe_audio_paused: boolean; wifi_ssid: string | null };
}

interface SyncStatus {
  [key: string]: { status: string; last_run?: string; files_today?: number };
}

interface SystemHealth {
  level?: string;
  system: { ram_total_gb: number; ram_used_gb: number; ram_percent: number; cpu_percent?: number };
  disk: { disk_total_gb: number; disk_free_gb: number; disk_percent: number };
  processes?: Record<string, { pid: number; rss_mb: number; cpu_percent: number }>;
  alerts?: { component: string; severity: string; message: string }[];
  growth_rates?: Record<string, number>;
  cpu_per_core?: number[];
  cpu_count?: number;
  cpu_freq_ghz?: number;
  load_avg?: number[];
  uptime_hours?: number;
  net_io?: { sent_mb: number; recv_mb: number };
  battery?: { percent: number; plugged: boolean } | null;
  gpu_name?: string;
  top_processes?: { pid: number; name: string; rss_mb: number; cpu: number }[];
}

interface WatchdogData { status: string; }

function ProgressBar({ value, max, color }: { value: number; max: number; color?: string }) {
  const pct = max > 0 ? Math.min((value / max) * 100, 100) : 0;
  const barColor = color || (pct > 90 ? "bg-red-400" : pct > 75 ? "bg-yellow-400" : "bg-primary");
  return (
    <div className="h-2 rounded-full bg-muted overflow-hidden">
      <div className={`h-full rounded-full transition-all duration-500 ${barColor}`} style={{ width: `${pct}%` }} />
    </div>
  );
}

function CoreBars({ cores }: { cores: number[] }) {
  return (
    <div className="flex items-end gap-[3px] h-8">
      {cores.map((pct, i) => (
        <div key={i} className="flex-1 min-w-[4px] max-w-[10px] rounded-t-sm bg-muted overflow-hidden flex flex-col justify-end h-full">
          <div
            className={`rounded-t-sm transition-all duration-500 ${pct > 80 ? "bg-red-400" : pct > 50 ? "bg-yellow-400" : "bg-primary"}`}
            style={{ height: `${Math.max(pct, 2)}%` }}
          />
        </div>
      ))}
    </div>
  );
}

export default function DashboardPage() {
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [sync, setSync] = useState<SyncStatus | null>(null);
  const [health, setHealth] = useState<SystemHealth | null>(null);
  const [watchdog, setWatchdog] = useState<WatchdogData | null>(null);
  const [error, setError] = useState<string | null>(null);

  const loadMain = useCallback(() => {
    fetch(`${API}/api/status`).then((r) => r.json()).then(setStatus).catch((e) => setError(e.message));
    fetch(`${API}/api/sync-status`).then((r) => r.json()).then(setSync).catch(() => {});
    fetch(`${API}/api/watchdog`).then((r) => r.json()).then(setWatchdog).catch(() => {});
  }, []);

  const loadHealth = useCallback(() => {
    fetch(`${API}/api/system-health`).then((r) => r.json()).then(setHealth).catch(() => {});
  }, []);

  useEffect(() => {
    loadMain();
    loadHealth();
    const mainIv = setInterval(loadMain, 15000);
    const healthIv = setInterval(loadHealth, 5000);
    return () => { clearInterval(mainIv); clearInterval(healthIv); };
  }, [loadMain, loadHealth]);

  const vaultFiles = status?.vault?.total_markdown_files || 0;
  const vaultSize = status?.vault?.size_mb || 0;
  const extractorCount = status ? Object.keys(status.extractors || {}).length : 0;
  const agentCount = status ? Object.values(status.launchd || {}).filter((a) => a.healthy).length : 0;
  const totalAgents = status ? Object.keys(status.launchd || {}).length : 0;
  const filesToday = status?.folders ? Object.values(status.folders).reduce((a, b) => a + b, 0) : 0;

  const h = health;
  const cpuAvg = h?.cpu_per_core ? Math.round(h.cpu_per_core.reduce((a, b) => a + b, 0) / h.cpu_per_core.length) : (h?.system?.cpu_percent ?? 0);

  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto max-w-6xl px-6 py-5 space-y-5">
        <div className="flex items-center justify-between">
          <h1 className="text-2xl font-semibold">MemoryOS Control Panel</h1>
          <div className="flex items-center gap-3">
            <span className="text-xs text-muted-foreground">Health: 5s | Status: 15s</span>
            <a href={API} target="_blank" rel="noopener noreferrer" className="inline-flex items-center gap-1.5 rounded-lg border border-border px-3 py-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors">
              <ExternalLink className="h-3.5 w-3.5" /> Legacy
            </a>
          </div>
        </div>

        {error && <div className="rounded-xl border border-red-400/30 bg-red-400/10 px-4 py-3 text-sm text-red-300">{error}</div>}

        {status?.privacy && (
          <PrivacyBar
            privacyMode={status.privacy.privacy_mode}
            audioPaused={status.privacy.screenpipe_audio_paused}
            wifiSsid={status.privacy.wifi_ssid}
            onRefresh={loadMain}
          />
        )}

        {/* Hero metrics */}
        <div className="grid grid-cols-2 lg:grid-cols-5 gap-4">
          <GlassCard label="Vault Files" className="text-center">
            <GradientValue className="text-3xl">{vaultFiles.toLocaleString()}</GradientValue>
            <div className="text-xs text-muted-foreground mt-1">markdown indexed</div>
          </GlassCard>
          <GlassCard label="Vault Size" className="text-center">
            <GradientValue className="text-3xl">{vaultSize.toFixed(0)}</GradientValue>
            <div className="text-xs text-muted-foreground mt-1">MB on disk</div>
          </GlassCard>
          <GlassCard label="Files Today" className="text-center">
            <GradientValue className="text-3xl">{filesToday.toLocaleString()}</GradientValue>
            <div className="text-xs text-muted-foreground mt-1">new/updated</div>
          </GlassCard>
          <GlassCard label="Extractors" className="text-center">
            <GradientValue className="text-3xl">{extractorCount}</GradientValue>
            <div className="text-xs text-muted-foreground mt-1">active</div>
          </GlassCard>
          <GlassCard label="Agents" className="text-center">
            <GradientValue className="text-3xl">{agentCount}/{totalAgents}</GradientValue>
            <div className="text-xs text-muted-foreground mt-1">healthy</div>
          </GlassCard>
        </div>

        {/* System Health - 6 cards */}
        {h && (
          <GlassCard label="System Health" className="!p-0">
            <div className="grid grid-cols-2 md:grid-cols-3 divide-x divide-y divide-border">
              {/* RAM */}
              <div className="p-4 space-y-2">
                <div className="flex items-center gap-2 text-sm font-medium">
                  <MemoryStick className="h-4 w-4 text-primary" /> RAM
                </div>
                <ProgressBar value={h.system.ram_used_gb} max={h.system.ram_total_gb} />
                <div className="text-xs text-muted-foreground">
                  {h.system.ram_used_gb.toFixed(1)} / {h.system.ram_total_gb.toFixed(0)} GB
                  <span className="ml-1 font-mono">({h.system.ram_percent}%)</span>
                </div>
              </div>

              {/* CPU */}
              <div className="p-4 space-y-2">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2 text-sm font-medium">
                    <Cpu className="h-4 w-4 text-primary" /> CPU
                  </div>
                  <span className="text-xs font-mono text-muted-foreground">
                    {cpuAvg}% avg
                    {h.cpu_freq_ghz ? ` @ ${h.cpu_freq_ghz} GHz` : ""}
                  </span>
                </div>
                {h.cpu_per_core ? (
                  <CoreBars cores={h.cpu_per_core} />
                ) : (
                  <ProgressBar value={cpuAvg} max={100} />
                )}
                <div className="text-xs text-muted-foreground">
                  {h.cpu_count || "—"} cores
                  {h.load_avg ? ` | Load: ${h.load_avg.join(" / ")}` : ""}
                </div>
              </div>

              {/* Disk */}
              <div className="p-4 space-y-2">
                <div className="flex items-center gap-2 text-sm font-medium">
                  <HardDrive className="h-4 w-4 text-primary" /> Disk
                </div>
                <ProgressBar value={h.disk.disk_percent} max={100} />
                <div className="text-xs text-muted-foreground">
                  {h.disk.disk_free_gb.toFixed(0)} GB free of {h.disk.disk_total_gb.toFixed(0)} GB
                  <span className="ml-1 font-mono">({h.disk.disk_percent}%)</span>
                </div>
              </div>

              {/* GPU / Battery */}
              <div className="p-4 space-y-2">
                <div className="flex items-center gap-2 text-sm font-medium">
                  <Monitor className="h-4 w-4 text-primary" /> GPU / Battery
                </div>
                {h.gpu_name && h.gpu_name !== "N/A" && (
                  <div className="text-xs font-medium">{h.gpu_name}</div>
                )}
                {h.battery ? (
                  <div className="flex items-center gap-2">
                    {h.battery.plugged ? (
                      <BatteryCharging className="h-4 w-4 text-green-400" />
                    ) : (
                      <Battery className="h-4 w-4 text-muted-foreground" />
                    )}
                    <ProgressBar value={h.battery.percent} max={100} color="bg-green-400" />
                    <span className="text-xs font-mono shrink-0">{h.battery.percent}%</span>
                  </div>
                ) : (
                  <div className="text-xs text-muted-foreground">Desktop (no battery)</div>
                )}
                {h.battery && (
                  <div className="text-xs text-muted-foreground">
                    {h.battery.plugged ? "Plugged in" : "On battery"}
                  </div>
                )}
              </div>

              {/* Network I/O */}
              <div className="p-4 space-y-2">
                <div className="flex items-center gap-2 text-sm font-medium">
                  <ArrowUpDown className="h-4 w-4 text-primary" /> Network
                </div>
                {h.net_io ? (
                  <div className="space-y-1">
                    <div className="flex items-center justify-between text-xs">
                      <span className="text-muted-foreground">Sent</span>
                      <span className="font-mono">
                        {h.net_io.sent_mb > 1024 ? `${(h.net_io.sent_mb / 1024).toFixed(1)} GB` : `${h.net_io.sent_mb.toFixed(0)} MB`}
                      </span>
                    </div>
                    <div className="flex items-center justify-between text-xs">
                      <span className="text-muted-foreground">Received</span>
                      <span className="font-mono">
                        {h.net_io.recv_mb > 1024 ? `${(h.net_io.recv_mb / 1024).toFixed(1)} GB` : `${h.net_io.recv_mb.toFixed(0)} MB`}
                      </span>
                    </div>
                  </div>
                ) : (
                  <div className="text-xs text-muted-foreground">No data</div>
                )}
              </div>

              {/* System / Uptime */}
              <div className="p-4 space-y-2">
                <div className="flex items-center gap-2 text-sm font-medium">
                  <Clock className="h-4 w-4 text-primary" /> System
                </div>
                <div className="space-y-1">
                  {h.uptime_hours != null && (
                    <div className="flex items-center justify-between text-xs">
                      <span className="text-muted-foreground">Uptime</span>
                      <span className="font-mono">
                        {h.uptime_hours >= 24
                          ? `${Math.floor(h.uptime_hours / 24)}d ${Math.round(h.uptime_hours % 24)}h`
                          : `${h.uptime_hours.toFixed(1)}h`}
                      </span>
                    </div>
                  )}
                  {h.load_avg && (
                    <div className="flex items-center justify-between text-xs">
                      <span className="text-muted-foreground">Load (1/5/15)</span>
                      <span className="font-mono">{h.load_avg.join(" / ")}</span>
                    </div>
                  )}
                  {h.level && (
                    <div className="flex items-center justify-between text-xs">
                      <span className="text-muted-foreground">Health Level</span>
                      <span className={`font-mono ${h.level === "ok" ? "text-green-400" : h.level === "warning" ? "text-yellow-400" : "text-red-400"}`}>
                        {h.level}
                      </span>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </GlassCard>
        )}

        {/* Top Processes */}
        {h?.top_processes && h.top_processes.length > 0 && (
          <GlassCard label={`Top Processes (${h.top_processes.length})`}>
            <div className="overflow-x-auto rounded-lg border border-border">
              <table className="w-full text-sm">
                <thead className="bg-muted/30">
                  <tr>
                    <th className="px-4 py-2 text-left text-xs font-semibold text-muted-foreground">Process</th>
                    <th className="px-4 py-2 text-right text-xs font-semibold text-muted-foreground">PID</th>
                    <th className="px-4 py-2 text-right text-xs font-semibold text-muted-foreground">RSS (MB)</th>
                    <th className="px-4 py-2 text-right text-xs font-semibold text-muted-foreground">CPU %</th>
                    <th className="px-4 py-2 text-right text-xs font-semibold text-muted-foreground">Memory</th>
                  </tr>
                </thead>
                <tbody>
                  {h.top_processes.map((p) => {
                    const ramPct = h.system.ram_total_gb > 0 ? ((p.rss_mb / 1024) / h.system.ram_total_gb * 100) : 0;
                    const growth = h.growth_rates?.[p.name.toLowerCase().replace(/[^a-z0-9]/g, "_")];
                    return (
                      <tr key={p.pid} className="border-t border-border hover:bg-muted/20 transition-colors">
                        <td className="px-4 py-1.5 text-xs truncate max-w-[200px]" title={p.name}>{p.name}</td>
                        <td className="px-4 py-1.5 text-xs text-right font-mono text-muted-foreground">{p.pid}</td>
                        <td className="px-4 py-1.5 text-xs text-right font-mono">
                          <span className={p.rss_mb > 2000 ? "text-red-400" : p.rss_mb > 500 ? "text-yellow-400" : ""}>
                            {p.rss_mb.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                          </span>
                        </td>
                        <td className="px-4 py-1.5 text-xs text-right font-mono">{p.cpu.toFixed(1)}</td>
                        <td className="px-4 py-1.5 text-right">
                          <div className="inline-flex items-center gap-2 w-24">
                            <div className="flex-1 h-1.5 rounded-full bg-muted overflow-hidden">
                              <div className="h-full rounded-full bg-primary transition-all" style={{ width: `${Math.min(ramPct, 100)}%` }} />
                            </div>
                            <span className="text-[10px] font-mono text-muted-foreground w-8 text-right">{ramPct.toFixed(1)}%</span>
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </GlassCard>
        )}

        {/* Alerts */}
        {h?.alerts && h.alerts.length > 0 && (
          <GlassCard label={`Alerts (${h.alerts.length})`}>
            <div className="space-y-2">
              {h.alerts.map((a, i) => (
                <div key={i} className={`flex items-start gap-2 rounded-lg border p-3 text-xs ${
                  a.severity === "critical" ? "border-red-400/30 bg-red-400/5 text-red-300" :
                  "border-yellow-400/30 bg-yellow-400/5 text-yellow-300"
                }`}>
                  <AlertTriangle className="h-4 w-4 shrink-0 mt-0.5" />
                  <div>
                    <span className="font-semibold uppercase text-[10px] mr-2">{a.severity}</span>
                    {a.message}
                  </div>
                </div>
              ))}
            </div>
          </GlassCard>
        )}

        {/* Extractors */}
        {status && (
          <GlassCard label="Extractors">
            <ExtractorCards extractors={status.extractors || {}} onRefresh={loadMain} />
          </GlassCard>
        )}

        {/* LaunchD Agents */}
        {status && Object.keys(status.launchd || {}).length > 0 && (
          <GlassCard label="LaunchD Agents">
            <LaunchdTable agents={status.launchd} onRefresh={loadMain} />
          </GlassCard>
        )}

        {/* Sync status */}
        {sync && Object.keys(sync).length > 0 && (
          <GlassCard label="Obsidian Sync">
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-2">
              {Object.entries(sync).map(([name, info]) => (
                <div key={name} className="flex items-center justify-between rounded-lg border border-border p-2.5">
                  <span className="text-xs capitalize truncate">{name}</span>
                  <span className={`rounded-full px-2 py-0.5 text-[10px] font-semibold ${
                    info.status === "healthy" ? "bg-green-400/15 text-green-400" :
                    info.status === "warning" ? "bg-yellow-400/15 text-yellow-400" :
                    "bg-muted text-muted-foreground"
                  }`}>{info.status || "—"}</span>
                </div>
              ))}
            </div>
          </GlassCard>
        )}

        {/* Watchdog */}
        {watchdog && (
          <GlassCard label="Watchdog">
            <div className="text-sm text-muted-foreground">
              Status: <span className={watchdog.status === "ok" ? "text-green-400" : "text-yellow-400"}>{watchdog.status}</span>
            </div>
          </GlassCard>
        )}
      </div>
    </div>
  );
}
