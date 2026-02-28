"use client";

import { useEffect, useState, useCallback, useMemo, useRef } from "react";
import { CheckCircle2, Loader2, Briefcase, ChevronDown, MapPin, Users, Clock, User } from "lucide-react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { GlassCard, GradientValue } from "@/components/ui/glass-card";
import { Tooltip, TooltipTrigger, TooltipContent, TooltipProvider } from "@/components/ui/tooltip";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog";
import { cn } from "@/lib/utils";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8765";

/* ── Types ── */
interface CalendarEvent { summary: string; start: string; end: string; is_all_day: boolean; location: string; attendees: string[]; organizer: string; }
interface AppUsage { name: string; minutes: number; percent: number; category: string; }
interface PersonEntry { name: string; meetings?: number; total_minutes?: number; }
interface WorkItem { task: string; priority: string; project: string; notes?: string; }
interface TaskBlock { start: string; end: string; title: string; apps: string[]; details?: string[]; }
interface ActivityData { total_active_hours: number; total_context_switches: number; app_breakdown: AppUsage[]; hourly: Record<string, Record<string, number>>; context_switches_per_hour: Record<string, number>; }
interface MyDayData { date: string; calendar: CalendarEvent[]; tasks: TaskBlock[]; activity: ActivityData; people: PersonEntry[]; work_completed: WorkItem[]; work_in_progress: WorkItem[]; has_data: boolean; }
interface TimelineData { hours: Record<string, Record<string, number>>; }

/* ── Helpers ── */
function fmt(d: Date) { return d.toISOString().split("T")[0]; }
function weekDates(center: string) { const d = new Date(center + "T12:00:00"); const mon = new Date(d); mon.setDate(d.getDate() - ((d.getDay() + 6) % 7)); return Array.from({ length: 7 }, (_, i) => { const dd = new Date(mon); dd.setDate(mon.getDate() + i); return dd; }); }
const DAYS = ["SUN", "MON", "TUE", "WED", "THU", "FRI", "SAT"];
function pColor(p: string) { return p === "P0" ? "text-red-400" : p === "P1" ? "text-yellow-400" : p === "P2" ? "text-blue-400" : "text-muted-foreground"; }
function timeToMin(t: string) { const [h, m] = t.split(":").map(Number); return h * 60 + (m || 0); }

const CAT_COLORS: Record<string, string> = {
  "Deep Work": "#4ade80", "Admin": "#94a3b8", "Communication": "#60a5fa",
  "Meeting/Communication": "#a78bfa", "Research": "#fb923c",
};
function catColor(cat: string) { return CAT_COLORS[cat] || "#6b7280"; }
function guessCat(app: string) {
  const a = app.toLowerCase();
  if (a.includes("cursor") || a.includes("terminal") || a.includes("code")) return "Deep Work";
  if (a.includes("mail") || a.includes("outlook") || a.includes("slack")) return "Communication";
  if (a.includes("teams")) return "Meeting/Communication";
  if (a.includes("chrome") || a.includes("safari") || a.includes("firefox")) return "Research";
  return "Admin";
}

/* ── SVG Focus Ring ── */
function FocusRing({ pct }: { pct: number }) {
  const r = 54, c = 2 * Math.PI * r, off = c - (c * Math.min(pct, 100)) / 100;
  return (
    <svg viewBox="0 0 128 128" className="w-28 h-28 mx-auto">
      <circle cx="64" cy="64" r={r} fill="none" stroke="rgba(108,124,255,.12)" strokeWidth="10" />
      <circle cx="64" cy="64" r={r} fill="none" stroke="url(#focusGrad)" strokeWidth="10" strokeLinecap="round" strokeDasharray={c} strokeDashoffset={off} transform="rotate(-90 64 64)" className="transition-all duration-700" />
      <defs><linearGradient id="focusGrad" x1="0" y1="0" x2="1" y2="1"><stop offset="0%" stopColor="#4ade80" /><stop offset="100%" stopColor="#22c55e" /></linearGradient></defs>
      <text x="64" y="60" textAnchor="middle" className="fill-foreground" style={{ fontSize: 28, fontWeight: 800 }}>{pct}%</text>
      <text x="64" y="78" textAnchor="middle" fill="#8b8fa3" style={{ fontSize: 10 }}>focus</text>
    </svg>
  );
}

/* ── Context Switches Area Sparkline ── */
function CsSpark({ csPerHour }: { csPerHour: Record<string, number> }) {
  const entries = Object.entries(csPerHour).sort(([a], [b]) => a.localeCompare(b));
  if (entries.length < 2) return null;
  const maxV = Math.max(...entries.map(([, v]) => v), 1);
  const w = 160, h = 48, px = 4;
  const stepX = (w - px * 2) / (entries.length - 1);
  const pts = entries.map(([, v], i) => ({ x: px + i * stepX, y: h - px - ((v / maxV) * (h - px * 2)) }));
  const line = pts.map((p, i) => `${i === 0 ? "M" : "L"}${p.x},${p.y}`).join(" ");
  const area = `${line} L${pts[pts.length - 1].x},${h - px} L${pts[0].x},${h - px} Z`;
  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="w-full h-12 mt-2">
      <defs>
        <linearGradient id="csGrad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="rgba(250,204,21,.3)" />
          <stop offset="100%" stopColor="rgba(250,204,21,.02)" />
        </linearGradient>
      </defs>
      <path d={area} fill="url(#csGrad)" />
      <path d={line} fill="none" stroke="rgba(250,204,21,.6)" strokeWidth="2" strokeLinejoin="round" />
    </svg>
  );
}

/* ── Format time for display ── */
function fmtTime(t: string) {
  const [h, m] = t.split(":").map(Number);
  const ampm = h >= 12 ? "PM" : "AM";
  const h12 = h === 0 ? 12 : h > 12 ? h - 12 : h;
  return `${h12}:${String(m || 0).padStart(2, "0")} ${ampm}`;
}

function fmtHour(h: number) {
  return h < 12 ? `${h}a` : h === 12 ? "12p" : `${h - 12}p`;
}

/* ── Interactive Timeline ── */
const SH = 6, EH = 21;
const LANE_ROW_H = 28;

function pctX(min: number) {
  return ((min - SH * 60) / ((EH - SH) * 60)) * 100;
}

function InteractiveTimeline({ calendar, tasks, csPerHour, isToday }: {
  calendar: CalendarEvent[]; tasks: TaskBlock[]; csPerHour: Record<string, number>; isToday: boolean;
}) {
  const [selectedMeeting, setSelectedMeeting] = useState<CalendarEvent | null>(null);
  const [expandedLane, setExpandedLane] = useState<string | null>(null);
  const [csHover, setCsHover] = useState<number | null>(null);
  const csLaneRef = useRef<HTMLDivElement>(null);

  const nonAllDay = useMemo(() => calendar.filter((e) => !e.is_all_day && e.start && e.end), [calendar]);
  const lanes = useMemo(() => {
    const result: { ev: CalendarEvent; s: number; e: number; lane: number }[] = [];
    const laneEnds: number[] = [];
    for (const ev of [...nonAllDay].sort((a, b) => timeToMin(a.start) - timeToMin(b.start))) {
      const s = timeToMin(ev.start), e = timeToMin(ev.end);
      let placed = -1;
      for (let l = 0; l < laneEnds.length; l++) { if (laneEnds[l] <= s) { placed = l; break; } }
      if (placed === -1) { placed = laneEnds.length; laneEnds.push(0); }
      laneEnds[placed] = e;
      result.push({ ev, s, e, lane: placed });
    }
    return { items: result, maxLanes: Math.max(laneEnds.length, 1) };
  }, [nonAllDay]);

  const hasTasks = tasks && tasks.length > 0;
  const hasCs = csPerHour && Object.values(csPerHour).some((v) => v > 0);

  const csEntries = useMemo(() => {
    if (!hasCs) return [];
    return Array.from({ length: EH - SH }, (_, i) => {
      const key = String(i + SH).padStart(2, "0") + ":00";
      return { hour: i + SH, val: csPerHour[key] || 0 };
    });
  }, [csPerHour, hasCs]);
  const csMax = Math.max(...csEntries.map((e) => e.val), 1);
  const csAvg = csEntries.length > 0 ? csEntries.reduce((s, e) => s + e.val, 0) / csEntries.length : 0;
  const csPeakHour = csEntries.reduce((best, e) => e.val > best.val ? e : best, { hour: 0, val: 0 }).hour;

  const nowMin = (() => { const n = new Date(); return n.getHours() * 60 + n.getMinutes(); })();
  const showNow = isToday && nowMin >= SH * 60 && nowMin <= EH * 60;

  const hours = Array.from({ length: EH - SH + 1 }, (_, i) => i + SH);

  const toggleLane = (lane: string) => setExpandedLane((prev) => prev === lane ? null : lane);

  const handleCsMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!csLaneRef.current || csEntries.length === 0) return;
    const rect = csLaneRef.current.getBoundingClientRect();
    const relX = (e.clientX - rect.left) / rect.width;
    const hourIdx = Math.round(relX * (csEntries.length - 1));
    const clamped = Math.max(0, Math.min(csEntries.length - 1, hourIdx));
    setCsHover(clamped);
  };

  return (
    <TooltipProvider delayDuration={100}>
      <div className="space-y-0">
        {/* Hour header */}
        <div className="relative ml-[72px] mr-[16px] h-5 mb-1">
          {hours.map((h) => (
            <span
              key={h}
              className="absolute text-[11px] text-[#6b7280] font-medium -translate-x-1/2"
              style={{ left: `${pctX(h * 60)}%` }}
            >
              {fmtHour(h)}
            </span>
          ))}
        </div>

        {/* Lane container */}
        <div className="space-y-1.5">
          {/* ── Meetings Lane ── */}
          <div className="flex items-stretch gap-0">
            <button
              onClick={() => toggleLane("meetings")}
              className="w-[72px] shrink-0 flex items-center justify-end pr-2.5 text-[10px] text-[#6b7280] font-medium hover:text-foreground transition-colors gap-1 cursor-pointer"
            >
              <span>Meetings</span>
              <ChevronDown className={cn("h-3 w-3 transition-transform duration-200", expandedLane === "meetings" && "rotate-180")} />
            </button>
            <div
              className="flex-1 relative rounded-md bg-[rgba(255,255,255,.02)] mr-4"
              style={{ minHeight: `${Math.max(40, lanes.maxLanes * LANE_ROW_H + 4)}px` }}
            >
              {/* Grid lines */}
              {hours.map((h) => (
                <div
                  key={h}
                  className="absolute top-0 bottom-0 border-l border-[rgba(255,255,255,.06)]"
                  style={{ left: `${pctX(h * 60)}%` }}
                />
              ))}
              {hours.filter((h) => h < EH && h % 2 === 0).map((h) => (
                <div
                  key={`bg-${h}`}
                  className="absolute top-0 bottom-0 bg-[rgba(255,255,255,.015)]"
                  style={{ left: `${pctX(h * 60)}%`, width: `${pctX((h + 1) * 60) - pctX(h * 60)}%` }}
                />
              ))}

              {/* Now marker */}
              {showNow && (
                <div className="absolute top-0 bottom-0 z-20 pointer-events-none" style={{ left: `${pctX(nowMin)}%` }}>
                  <div className="w-2 h-2 rounded-full bg-red-500 -translate-x-1/2 -translate-y-1" />
                  <div className="w-0.5 h-full -translate-x-[1px] opacity-80" style={{ backgroundImage: "repeating-linear-gradient(to bottom, #ef4444 0px, #ef4444 4px, transparent 4px, transparent 7px)" }} />
                </div>
              )}

              {/* Meeting bars */}
              {lanes.items.map((item, i) => {
                const leftPct = pctX(Math.max(item.s, SH * 60));
                const rightPct = pctX(Math.min(item.e, EH * 60));
                const widthPct = Math.max(rightPct - leftPct, 0.5);
                const top = item.lane * LANE_ROW_H + 2;
                const is1on1 = item.ev.attendees && item.ev.attendees.length <= 2;
                const col = is1on1 ? "#818cf8" : "#6c7cff";
                const durMin = item.e - item.s;

                return (
                  <Tooltip key={i}>
                    <TooltipTrigger asChild>
                      <button
                        onClick={() => setSelectedMeeting(item.ev)}
                        className="absolute rounded transition-all duration-150 hover:brightness-125 hover:z-10 hover:shadow-[0_0_12px_rgba(108,124,255,.3)] cursor-pointer overflow-hidden text-left"
                        style={{
                          left: `${leftPct}%`,
                          width: `${widthPct}%`,
                          top: `${top}px`,
                          height: `${LANE_ROW_H - 4}px`,
                          backgroundColor: col,
                          opacity: 0.85 - item.lane * 0.1,
                        }}
                      >
                        <span className="block px-1.5 text-[10px] font-semibold text-white truncate leading-[24px]">
                          {item.ev.summary}
                        </span>
                      </button>
                    </TooltipTrigger>
                    <TooltipContent side="top" className="max-w-xs bg-[rgba(20,22,30,.95)] border border-[rgba(108,124,255,.2)] text-foreground p-3 space-y-1.5">
                      <div className="font-semibold text-sm">{item.ev.summary}</div>
                      <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                        <Clock className="h-3 w-3" />
                        {fmtTime(item.ev.start)} – {fmtTime(item.ev.end)} ({durMin}m)
                      </div>
                      {item.ev.attendees.length > 0 && (
                        <div className="flex items-start gap-1.5 text-xs text-muted-foreground">
                          <Users className="h-3 w-3 mt-0.5 shrink-0" />
                          <span>{item.ev.attendees.slice(0, 4).join(", ")}{item.ev.attendees.length > 4 && ` +${item.ev.attendees.length - 4}`}</span>
                        </div>
                      )}
                      {item.ev.location && (
                        <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                          <MapPin className="h-3 w-3" />
                          <span className="truncate">{item.ev.location}</span>
                        </div>
                      )}
                      {is1on1 && <span className="inline-block text-[10px] bg-[#818cf8]/20 text-[#818cf8] px-1.5 py-0.5 rounded">1:1</span>}
                      <div className="text-[10px] text-muted-foreground/60 pt-0.5">Click for details</div>
                    </TooltipContent>
                  </Tooltip>
                );
              })}
            </div>
          </div>

          {/* ── Activity Lane ── */}
          {hasTasks && (
            <div className="flex items-stretch gap-0">
              <button
                onClick={() => toggleLane("activity")}
                className="w-[72px] shrink-0 flex items-center justify-end pr-2.5 text-[10px] text-[#6b7280] font-medium hover:text-foreground transition-colors gap-1 cursor-pointer"
              >
                <span>Activity</span>
                <ChevronDown className={cn("h-3 w-3 transition-transform duration-200", expandedLane === "activity" && "rotate-180")} />
              </button>
              <div className="flex-1 relative rounded-md bg-[rgba(255,255,255,.02)] mr-4" style={{ minHeight: "36px" }}>
                {hours.map((h) => (
                  <div key={h} className="absolute top-0 bottom-0 border-l border-[rgba(255,255,255,.06)]" style={{ left: `${pctX(h * 60)}%` }} />
                ))}
                {showNow && (
                  <div className="absolute top-0 bottom-0 z-20 pointer-events-none" style={{ left: `${pctX(nowMin)}%` }}>
                    <div className="w-0.5 h-full bg-red-500/70 -translate-x-[1px]" />
                  </div>
                )}

                {tasks.map((t, i) => {
                  const s = timeToMin(t.start), e = timeToMin(t.end);
                  if (e <= SH * 60 || s >= EH * 60) return null;
                  const leftPct = pctX(Math.max(s, SH * 60));
                  const rightPct = pctX(Math.min(e, EH * 60));
                  const widthPct = Math.max(rightPct - leftPct, 0.3);
                  const cat = t.apps?.length ? guessCat(t.apps[0]) : "Admin";
                  const col = CAT_COLORS[cat] || "#6b7280";

                  return (
                    <Tooltip key={i}>
                      <TooltipTrigger asChild>
                        <div
                          className="absolute top-[2px] bottom-[2px] rounded transition-all duration-150 hover:brightness-125 hover:z-10 overflow-hidden cursor-default"
                          style={{
                            left: `${leftPct}%`,
                            width: `${widthPct}%`,
                            backgroundColor: col,
                            opacity: 0.55,
                            borderLeft: `3px solid ${col}`,
                          }}
                        >
                          <span className="block px-1 text-[9px] font-medium text-white truncate leading-[32px]">
                            {t.title}
                          </span>
                        </div>
                      </TooltipTrigger>
                      <TooltipContent side="top" className="max-w-xs bg-[rgba(20,22,30,.95)] border border-[rgba(108,124,255,.2)] text-foreground p-3 space-y-1.5">
                        <div className="font-semibold text-sm">{t.title}</div>
                        <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                          <Clock className="h-3 w-3" />
                          {fmtTime(t.start)} – {fmtTime(t.end)}
                        </div>
                        {t.apps?.length > 0 && (
                          <div className="flex flex-wrap gap-1.5 mt-1">
                            {t.apps.map((app) => (
                              <span key={app} className="flex items-center gap-1 text-[10px] text-muted-foreground bg-[rgba(255,255,255,.06)] px-1.5 py-0.5 rounded">
                                <span className="w-1.5 h-1.5 rounded-full" style={{ background: catColor(guessCat(app)) }} />
                                {app}
                              </span>
                            ))}
                          </div>
                        )}
                        {t.details && t.details.length > 0 && (
                          <div className="text-xs text-muted-foreground/80 pt-1 space-y-0.5">
                            {t.details.slice(0, 3).map((d, j) => <div key={j}>· {d}</div>)}
                          </div>
                        )}
                      </TooltipContent>
                    </Tooltip>
                  );
                })}
              </div>
            </div>
          )}

          {/* ── Context Switches Lane ── */}
          {hasCs && (
            <div className="flex items-stretch gap-0">
              <button
                onClick={() => toggleLane("switches")}
                className="w-[72px] shrink-0 flex items-center justify-end pr-2.5 text-[10px] text-[#6b7280] font-medium hover:text-foreground transition-colors gap-1 cursor-pointer"
              >
                <span>Switches</span>
                <ChevronDown className={cn("h-3 w-3 transition-transform duration-200", expandedLane === "switches" && "rotate-180")} />
              </button>
              <div
                ref={csLaneRef}
                className="flex-1 relative rounded-md bg-[rgba(255,255,255,.02)] mr-4 cursor-crosshair"
                style={{ height: "40px" }}
                onMouseMove={handleCsMouseMove}
                onMouseLeave={() => setCsHover(null)}
              >
                {hours.map((h) => (
                  <div key={h} className="absolute top-0 bottom-0 border-l border-[rgba(255,255,255,.06)]" style={{ left: `${pctX(h * 60)}%` }} />
                ))}

                {/* SVG sparkline */}
                <svg className="absolute inset-0 w-full h-full" viewBox="0 0 1000 100" preserveAspectRatio="none">
                  <defs>
                    <linearGradient id="csTimelineGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="rgba(250,204,21,.25)" />
                      <stop offset="100%" stopColor="rgba(250,204,21,.02)" />
                    </linearGradient>
                  </defs>
                  {(() => {
                    if (csEntries.length < 2) return null;
                    const pts = csEntries.map((e) => ({
                      x: (pctX(e.hour * 60 + 30) / 100) * 1000,
                      y: (1 - (e.val / csMax) * 0.85) * 100,
                    }));
                    const linePath = `M${pts.map((p) => `${p.x},${p.y}`).join(" L")}`;
                    const areaPath = `${linePath} L${pts[pts.length - 1].x},100 L${pts[0].x},100 Z`;
                    return (
                      <g>
                        <path d={areaPath} fill="url(#csTimelineGrad)" />
                        <path d={linePath} fill="none" stroke="rgba(250,204,21,.5)" strokeWidth="2" vectorEffect="non-scaling-stroke" />
                        {pts.map((p, i) => csEntries[i].hour === csPeakHour && csEntries[i].val > 0 && (
                          <circle key={i} cx={p.x} cy={p.y} r="4" fill="#facc15" opacity="0.8" vectorEffect="non-scaling-size" />
                        ))}
                      </g>
                    );
                  })()}
                </svg>

                {/* Hover crosshair + tooltip */}
                {csHover !== null && csEntries[csHover] && (
                  <>
                    <div
                      className="absolute top-0 bottom-0 w-px bg-yellow-400/40 pointer-events-none z-10"
                      style={{ left: `${pctX(csEntries[csHover].hour * 60 + 30)}%` }}
                    />
                    <div
                      className="absolute z-20 pointer-events-none -translate-x-1/2 -top-10 bg-[rgba(20,22,30,.95)] border border-yellow-400/20 rounded px-2 py-1 text-[10px] text-yellow-300 whitespace-nowrap"
                      style={{ left: `${pctX(csEntries[csHover].hour * 60 + 30)}%` }}
                    >
                      {fmtHour(csEntries[csHover].hour)}: <b>{csEntries[csHover].val}</b> switches
                      {csEntries[csHover].val > csAvg * 1.5 ? " (High)" : csEntries[csHover].val < csAvg * 0.5 ? " (Low)" : ""}
                    </div>
                    <div
                      className="absolute z-10 w-2 h-2 rounded-full bg-yellow-400 -translate-x-1 pointer-events-none"
                      style={{
                        left: `${pctX(csEntries[csHover].hour * 60 + 30)}%`,
                        top: `${(1 - (csEntries[csHover].val / csMax) * 0.85) * 100}%`,
                        transform: "translate(-50%, -50%)",
                      }}
                    />
                  </>
                )}

                {showNow && (
                  <div className="absolute top-0 bottom-0 z-5 pointer-events-none" style={{ left: `${pctX(nowMin)}%` }}>
                    <div className="w-0.5 h-full bg-red-500/70 -translate-x-[1px]" />
                  </div>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Legend */}
        <div className="flex items-center gap-4 ml-[72px] mt-2">
          {[
            { label: "Meeting", color: "#6c7cff" },
            { label: "1:1", color: "#818cf8" },
            { label: "Deep Work", color: "#4ade80" },
            { label: "Comms", color: "#60a5fa" },
            { label: "Research", color: "#fb923c" },
            { label: "Switches", color: "#facc15" },
          ].map((it) => (
            <span key={it.label} className="flex items-center gap-1.5 text-[10px] text-[#6b7280]">
              <span className="w-2 h-2 rounded-sm" style={{ background: it.color, opacity: 0.7 }} />
              {it.label}
            </span>
          ))}
        </div>

        {/* ── Expanded Detail Panels ── */}
        {expandedLane === "meetings" && (
          <div className="mt-2 rounded-lg bg-[rgba(255,255,255,.03)] border border-[rgba(108,124,255,.08)] p-3 space-y-1 max-h-[280px] overflow-y-auto animate-in slide-in-from-top-2 duration-200">
            <div className="text-[10px] uppercase tracking-wider text-muted-foreground font-semibold mb-2">All Meetings ({calendar.length})</div>
            {calendar.map((evt, i) => {
              const durMin = !evt.is_all_day && evt.start && evt.end ? timeToMin(evt.end) - timeToMin(evt.start) : 0;
              const is1on1 = evt.attendees && evt.attendees.length <= 2 && !evt.is_all_day;
              return (
                <button
                  key={i}
                  onClick={() => setSelectedMeeting(evt)}
                  className="w-full flex items-start gap-3 py-2 px-2 rounded-md border-b border-[rgba(255,255,255,.04)] last:border-0 hover:bg-[rgba(108,124,255,.06)] transition-colors text-left cursor-pointer"
                >
                  <div className="min-w-[5rem] text-xs font-mono text-muted-foreground pt-0.5">
                    {evt.is_all_day ? "All day" : `${evt.start}–${evt.end}`}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="text-sm font-medium truncate flex items-center gap-2">
                      {evt.summary}
                      {is1on1 && <span className="text-[9px] bg-[#818cf8]/20 text-[#818cf8] px-1 py-0.5 rounded shrink-0">1:1</span>}
                    </div>
                    {evt.attendees.length > 0 && (
                      <div className="text-xs text-muted-foreground mt-0.5 truncate">
                        {evt.attendees.slice(0, 4).join(", ")}{evt.attendees.length > 4 && ` +${evt.attendees.length - 4}`}
                      </div>
                    )}
                  </div>
                  {durMin > 0 && <span className="text-xs text-muted-foreground shrink-0">{durMin}m</span>}
                </button>
              );
            })}
            {calendar.length === 0 && <div className="text-sm text-muted-foreground py-4 text-center">No meetings</div>}
          </div>
        )}

        {expandedLane === "activity" && (
          <div className="mt-2 rounded-lg bg-[rgba(255,255,255,.03)] border border-[rgba(108,124,255,.08)] p-3 space-y-1 max-h-[280px] overflow-y-auto animate-in slide-in-from-top-2 duration-200">
            <div className="text-[10px] uppercase tracking-wider text-muted-foreground font-semibold mb-2">Activity Blocks ({tasks.length})</div>
            {tasks.map((t, i) => {
              const cat = t.apps?.length ? guessCat(t.apps[0]) : "Admin";
              const col = CAT_COLORS[cat] || "#6b7280";
              return (
                <div key={i} className="flex items-start gap-3 py-2 px-2 border-b border-[rgba(255,255,255,.04)] last:border-0">
                  <div className="min-w-[5rem] text-xs font-mono text-muted-foreground">{t.start}–{t.end}</div>
                  <div className="w-1 self-stretch rounded-full shrink-0" style={{ background: col }} />
                  <div className="flex-1 min-w-0">
                    <div className="text-sm">{t.title}</div>
                    {t.apps?.length > 0 && (
                      <div className="flex flex-wrap gap-1.5 mt-1">
                        {t.apps.map((app) => (
                          <span key={app} className="flex items-center gap-1 text-[10px] text-muted-foreground bg-[rgba(255,255,255,.05)] px-1.5 py-0.5 rounded">
                            <span className="w-1.5 h-1.5 rounded-full" style={{ background: catColor(guessCat(app)) }} />
                            {app}
                          </span>
                        ))}
                      </div>
                    )}
                    {t.details && t.details.length > 0 && (
                      <div className="text-xs text-muted-foreground/70 mt-1 space-y-0.5">
                        {t.details.slice(0, 5).map((d, j) => <div key={j}>· {d}</div>)}
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {expandedLane === "switches" && (
          <div className="mt-2 rounded-lg bg-[rgba(255,255,255,.03)] border border-[rgba(108,124,255,.08)] p-3 animate-in slide-in-from-top-2 duration-200">
            <div className="text-[10px] uppercase tracking-wider text-muted-foreground font-semibold mb-2">
              Context Switches by Hour (avg {csAvg.toFixed(1)}/hr)
            </div>
            <div className="flex items-end gap-1 h-24">
              {csEntries.map((entry, i) => {
                const heightPct = csMax > 0 ? (entry.val / csMax) * 100 : 0;
                const isPeak = entry.hour === csPeakHour && entry.val > 0;
                const isHigh = entry.val > csAvg * 1.5;
                return (
                  <Tooltip key={i}>
                    <TooltipTrigger asChild>
                      <div className="flex-1 flex flex-col items-center gap-1 cursor-default">
                        <span className="text-[9px] text-muted-foreground">{entry.val}</span>
                        <div
                          className={cn(
                            "w-full rounded-t transition-all duration-200",
                            isPeak ? "bg-yellow-400" : isHigh ? "bg-yellow-400/70" : "bg-yellow-400/30"
                          )}
                          style={{ height: `${Math.max(heightPct, 2)}%` }}
                        />
                        <span className="text-[9px] text-muted-foreground">{fmtHour(entry.hour)}</span>
                      </div>
                    </TooltipTrigger>
                    <TooltipContent side="top" className="bg-[rgba(20,22,30,.95)] border border-yellow-400/20 text-foreground">
                      <div className="text-xs">
                        <b>{entry.val}</b> switches at {fmtHour(entry.hour)}
                        {isPeak && <span className="ml-1 text-yellow-400">(Peak)</span>}
                        {isHigh && !isPeak && <span className="ml-1 text-yellow-400/70">(High)</span>}
                      </div>
                    </TooltipContent>
                  </Tooltip>
                );
              })}
            </div>
          </div>
        )}

        {/* ── Meeting Detail Dialog ── */}
        <Dialog open={!!selectedMeeting} onOpenChange={(open) => !open && setSelectedMeeting(null)}>
          <DialogContent className="bg-[rgba(20,22,30,.97)] border-[rgba(108,124,255,.15)] max-w-md">
            {selectedMeeting && (
              <>
                <DialogHeader>
                  <DialogTitle className="text-lg">{selectedMeeting.summary}</DialogTitle>
                  <DialogDescription asChild>
                    <div className="space-y-3 pt-2">
                      <div className="flex items-center gap-2 text-sm text-muted-foreground">
                        <Clock className="h-4 w-4 shrink-0" />
                        {selectedMeeting.is_all_day ? (
                          <span>All day</span>
                        ) : (
                          <span>
                            {fmtTime(selectedMeeting.start)} – {fmtTime(selectedMeeting.end)}
                            <span className="ml-2 text-xs opacity-70">
                              ({timeToMin(selectedMeeting.end) - timeToMin(selectedMeeting.start)}m)
                            </span>
                          </span>
                        )}
                      </div>

                      {selectedMeeting.location && (
                        <div className="flex items-center gap-2 text-sm text-muted-foreground">
                          <MapPin className="h-4 w-4 shrink-0" />
                          <span className="break-all">{selectedMeeting.location}</span>
                        </div>
                      )}

                      {selectedMeeting.organizer && (
                        <div className="flex items-center gap-2 text-sm text-muted-foreground">
                          <User className="h-4 w-4 shrink-0" />
                          <span>Organized by {selectedMeeting.organizer}</span>
                        </div>
                      )}

                      {selectedMeeting.attendees.length > 0 && (
                        <div className="space-y-1.5">
                          <div className="flex items-center gap-2 text-sm text-muted-foreground">
                            <Users className="h-4 w-4 shrink-0" />
                            <span>{selectedMeeting.attendees.length} attendee{selectedMeeting.attendees.length !== 1 ? "s" : ""}</span>
                          </div>
                          <div className="ml-6 flex flex-wrap gap-1.5">
                            {selectedMeeting.attendees.map((a, i) => (
                              <span key={i} className="text-xs bg-[rgba(108,124,255,.1)] text-muted-foreground px-2 py-1 rounded-md">
                                {a}
                              </span>
                            ))}
                          </div>
                        </div>
                      )}

                      {selectedMeeting.attendees.length <= 2 && !selectedMeeting.is_all_day && (
                        <span className="inline-block text-xs bg-[#818cf8]/20 text-[#818cf8] px-2 py-1 rounded">1:1 Meeting</span>
                      )}
                    </div>
                  </DialogDescription>
                </DialogHeader>
              </>
            )}
          </DialogContent>
        </Dialog>
      </div>
    </TooltipProvider>
  );
}

/* ── Page ── */
export default function MyDayPage() {
  const [date, setDate] = useState(() => fmt(new Date()));
  const [data, setData] = useState<MyDayData | null>(null);
  const [, setTimeline] = useState<TimelineData | null>(null);
  const [loading, setLoading] = useState(false);

  const load = useCallback(async (d: string) => {
    setLoading(true);
    try {
      const [md, tl] = await Promise.all([
        fetch(`${API_BASE}/api/myday/${d}`).then((r) => r.json()),
        fetch(`${API_BASE}/api/timeline/${d}`).then((r) => r.json()),
      ]);
      setData(md);
      setTimeline(tl);
    } catch { setData(null); setTimeline(null); }
    finally { setLoading(false); }
  }, []);

  useEffect(() => { load(date); }, [date, load]);

  const week = useMemo(() => weekDates(date), [date]);
  const today = useMemo(() => fmt(new Date()), []);

  const mtgCount = data?.calendar?.length || 0;
  const mtgMins = data?.calendar?.reduce((s, e) => {
    if (e.is_all_day || !e.start || !e.end) return s;
    return s + timeToMin(e.end) - timeToMin(e.start);
  }, 0) || 0;
  const activeH = data?.activity?.total_active_hours || 0;
  const ctxSw = data?.activity?.total_context_switches || 0;
  const doneCount = data?.work_completed?.length || 0;
  const focus = activeH > 0 ? Math.round(100 - (ctxSw / activeH) * 8) : 0;
  const topPerson = data?.people?.[0]?.name || "—";
  const csPerHour = data?.activity?.context_switches_per_hour || {};

  const appGroups = useMemo(() => {
    if (!data?.activity?.app_breakdown) return [];
    const groups: Record<string, { apps: AppUsage[]; totalMin: number; totalPct: number }> = {};
    for (const app of data.activity.app_breakdown) {
      const cat = app.category || "Other";
      if (!groups[cat]) groups[cat] = { apps: [], totalMin: 0, totalPct: 0 };
      groups[cat].apps.push(app);
      groups[cat].totalMin += app.minutes;
      groups[cat].totalPct += app.percent;
    }
    return Object.entries(groups).sort(([, a], [, b]) => b.totalMin - a.totalMin);
  }, [data]);

  return (
    <ScrollArea className="h-full">
      <div className="mx-auto max-w-6xl px-6 py-5 space-y-4">
        {/* Date pills */}
        <div className="flex items-center gap-2 flex-wrap">
          {week.map((d) => {
            const ds = fmt(d); const isActive = ds === date; const isToday = ds === today;
            return (
              <button key={ds} onClick={() => setDate(ds)} className={cn(
                "flex flex-col items-center gap-0.5 rounded-2xl border px-4 py-2 text-sm transition-all duration-200",
                isActive ? "bg-gradient-to-br from-primary to-[#a78bfa] text-white border-transparent font-semibold shadow-[0_2px_12px_rgba(108,124,255,.35)]"
                  : "border-border bg-card text-muted-foreground hover:border-primary hover:text-foreground"
              )}>
                <span className="text-[0.68rem] uppercase opacity-70">{DAYS[d.getDay()]}</span>
                <span className="text-[0.95rem] font-bold">{d.getDate()}</span>
                {isToday && !isActive && <span className="w-1 h-1 rounded-full bg-green-400 mt-0.5" />}
              </button>
            );
          })}
        </div>

        {/* Stats bar */}
        {data && !loading && (
          <div className="flex items-center gap-3 flex-wrap">
            <span className="flex items-center gap-1.5 rounded-full bg-green-400/10 px-3 py-1 text-xs font-semibold text-green-400">
              <span className="w-2 h-2 rounded-full bg-green-400 shadow-[0_0_6px_#4ade80]" /> Live
            </span>
            <StatPill label="meetings" value={mtgCount} />
            <StatPill label="active" value={`${activeH.toFixed(1)}h`} />
            <StatPill label="focus" value={`${focus}%`} />
            <StatPill label="completed" value={doneCount} />
            <StatPill label="top" value={topPerson} />
          </div>
        )}

        {loading && <div className="flex justify-center py-20"><Loader2 className="h-6 w-6 animate-spin text-muted-foreground" /></div>}

        {data && !loading && (
          <>
            {/* Row 1: Interactive Timeline */}
            <GlassCard label="Timeline">
              <InteractiveTimeline
                calendar={data.calendar}
                tasks={data.tasks || []}
                csPerHour={csPerHour}
                isToday={date === today}
              />
            </GlassCard>

            {/* Row 2: Four metrics */}
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
              <GlassCard label="Focus Score" className="text-center flex flex-col items-center justify-center min-h-[180px]">
                <FocusRing pct={focus} />
                <div className="text-xs text-muted-foreground mt-2">{activeH.toFixed(1)}h active time</div>
              </GlassCard>
              <GlassCard label="Meeting Load" className="text-center min-h-[180px]">
                <GradientValue className="text-5xl">{mtgCount}</GradientValue>
                <div className="text-xs text-muted-foreground mt-1">{(mtgMins / 60).toFixed(1)}h in meetings</div>
                <div className="flex gap-px mt-3 justify-center">
                  {data.calendar.filter((e) => !e.is_all_day).slice(0, 15).map((e, i) => {
                    const dur = e.start && e.end ? timeToMin(e.end) - timeToMin(e.start) : 30;
                    return <div key={i} className="w-2 rounded-sm bg-primary/50" style={{ height: `${Math.max(4, dur / 4)}px` }} />;
                  })}
                </div>
              </GlassCard>
              <GlassCard label="Context Switches" className="text-center min-h-[180px]">
                <GradientValue className="text-5xl">{ctxSw}</GradientValue>
                <div className="text-xs text-muted-foreground mt-1">
                  {activeH > 0 ? `${(ctxSw / activeH).toFixed(1)}/hr` : "—"} total switches
                </div>
                <CsSpark csPerHour={csPerHour} />
              </GlassCard>
              <GlassCard label="Work Completed" className="text-center min-h-[180px]">
                <GradientValue className="text-5xl">{doneCount}</GradientValue>
                <div className="text-xs text-muted-foreground mt-1">tasks shipped</div>
                {data.work_completed?.slice(0, 2).map((w, i) => (
                  <div key={i} className="flex items-center gap-1.5 mt-2 text-xs text-green-400">
                    <CheckCircle2 className="h-3 w-3" /> <span className="truncate">{w.task}</span>
                  </div>
                ))}
              </GlassCard>
            </div>

            {/* Row 3: App Usage (grouped by category) + People */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <GlassCard label={`App Usage — ${data.activity?.app_breakdown?.reduce((s, a) => s + a.minutes, 0) || 0}m total tracked`}>
                {appGroups.length > 0 ? (
                  <div className="space-y-4">
                    {appGroups.map(([cat, group]) => (
                      <div key={cat}>
                        <div className="flex items-center gap-2 mb-2">
                          <span className="w-2.5 h-2.5 rounded-full" style={{ background: catColor(cat) }} />
                          <span className="text-sm font-semibold">{cat}</span>
                          <span className="text-xs text-muted-foreground ml-auto">{group.totalMin}m · {group.totalPct}%</span>
                        </div>
                        <div className="space-y-2 pl-5">
                          {group.apps.sort((a, b) => b.minutes - a.minutes).map((app) => (
                            <div key={app.name} className="space-y-0.5">
                              <div className="flex items-center justify-between text-sm">
                                <span className="flex items-center gap-2">
                                  <span className="w-1.5 h-1.5 rounded-full" style={{ background: catColor(cat) }} />
                                  {app.name}
                                </span>
                                <span className="text-muted-foreground text-xs">{app.minutes}m</span>
                              </div>
                              <div className="h-1.5 rounded-full bg-[rgba(108,124,255,.08)] overflow-hidden">
                                <div className="h-full rounded-full" style={{ width: `${Math.min(app.percent, 100)}%`, background: catColor(cat) }} />
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                ) : <div className="text-sm text-muted-foreground py-4 text-center">No activity data</div>}
              </GlassCard>

              <GlassCard label="People">
                {data.people?.length > 0 ? (
                  <div className="space-y-2.5">
                    {data.people.slice(0, 12).map((p, i) => {
                      const maxMtg = Math.max(...data.people.map((pp) => pp.meetings || 1));
                      const pct = ((p.meetings || 0) / maxMtg) * 100;
                      return (
                        <div key={i} className="space-y-0.5">
                          <div className="flex items-center justify-between text-sm">
                            <span className="truncate">{p.name}</span>
                            <span className="text-xs text-muted-foreground shrink-0 ml-2">
                              {p.meetings || 0} mtg · {Math.round(p.total_minutes || 0)}m
                            </span>
                          </div>
                          <div className="h-1 rounded-full bg-[rgba(108,124,255,.08)] overflow-hidden">
                            <div className="h-full rounded-full bg-gradient-to-r from-primary to-[#a78bfa]" style={{ width: `${pct}%` }} />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                ) : <div className="text-sm text-muted-foreground py-4 text-center">No people data</div>}
              </GlassCard>
            </div>

            {/* Row 4: Work Status + Schedule */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <GlassCard label="Work Status">
                <div className="space-y-1.5 max-h-[420px] overflow-y-auto pr-1">
                  {data.work_in_progress?.map((w, i) => (
                    <div key={`ip-${i}`} className="flex items-start gap-2 text-sm py-1.5 border-b border-[rgba(255,255,255,.04)] last:border-0">
                      <Briefcase className="h-3.5 w-3.5 mt-0.5 text-primary shrink-0" />
                      <div className="flex-1 min-w-0">
                        <div className="truncate">{w.task}</div>
                        <div className="text-xs text-muted-foreground"><span className={pColor(w.priority)}>{w.priority}</span> · {w.project}</div>
                      </div>
                    </div>
                  ))}
                  {data.work_completed?.map((w, i) => (
                    <div key={`wc-${i}`} className="flex items-start gap-2 text-sm py-1.5 border-b border-[rgba(255,255,255,.04)] last:border-0 opacity-60">
                      <CheckCircle2 className="h-3.5 w-3.5 mt-0.5 text-green-400 shrink-0" />
                      <div className="flex-1 min-w-0"><div className="truncate line-through">{w.task}</div><div className="text-xs text-muted-foreground">{w.project}</div></div>
                    </div>
                  ))}
                  {!data.work_in_progress?.length && !data.work_completed?.length && <div className="text-sm text-muted-foreground py-4 text-center">No work items</div>}
                </div>
              </GlassCard>

              <GlassCard label="Schedule">
                <div className="space-y-0.5 max-h-[420px] overflow-y-auto pr-1">
                  {data.calendar.map((evt, i) => (
                    <div key={i} className="flex items-start gap-3 py-2 border-b border-[rgba(255,255,255,.04)] last:border-0">
                      <div className="min-w-[4.5rem] text-xs font-mono text-muted-foreground pt-0.5">{evt.is_all_day ? "All day" : `${evt.start}–${evt.end}`}</div>
                      <div className="flex-1 min-w-0">
                        <div className="text-sm font-medium truncate">{evt.summary}</div>
                        {evt.attendees.length > 0 && <div className="text-xs text-muted-foreground mt-0.5 truncate">{evt.attendees.slice(0, 3).join(", ")}{evt.attendees.length > 3 && ` +${evt.attendees.length - 3}`}</div>}
                      </div>
                    </div>
                  ))}
                  {data.calendar.length === 0 && <div className="text-sm text-muted-foreground py-4 text-center">No events</div>}
                </div>
              </GlassCard>
            </div>

            {/* Row 5: Activity Timeline */}
            {data.tasks && data.tasks.length > 0 && (
              <GlassCard label="Activity Timeline">
                <div className="space-y-1.5 max-h-[350px] overflow-y-auto pr-1">
                  {data.tasks.map((t, i) => (
                    <div key={i} className="flex items-start gap-3 py-1.5 border-b border-[rgba(255,255,255,.04)] last:border-0">
                      <div className="min-w-[5.5rem] text-xs font-mono text-muted-foreground">{t.start}–{t.end}</div>
                      <div className="flex-1 min-w-0">
                        <div className="text-sm">{t.title}</div>
                        {t.apps?.length > 0 && (
                          <div className="flex flex-wrap gap-1.5 mt-0.5">
                            {t.apps.map((app) => (
                              <span key={app} className="flex items-center gap-1 text-[10px] text-muted-foreground">
                                <span className="w-1.5 h-1.5 rounded-full" style={{ background: catColor(guessCat(app)) }} />
                                {app}
                              </span>
                            ))}
                          </div>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </GlassCard>
            )}
          </>
        )}

        {!data && !loading && <div className="py-20 text-center text-muted-foreground">No data for {date}</div>}
      </div>
    </ScrollArea>
  );
}

function StatPill({ label, value }: { label: string; value: string | number }) {
  return (
    <span className="flex items-center gap-1.5 rounded-full bg-[rgba(108,124,255,.08)] px-3 py-1 text-xs text-muted-foreground">
      <b className="text-foreground font-semibold">{value}</b> {label}
    </span>
  );
}
