"use client";

import { useState, useEffect, useCallback } from "react";
import { Mic, MicOff, ChevronDown, Shield, ShieldOff } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8765";

interface PrivacyBarProps {
  privacyMode: boolean;
  audioPaused: boolean;
  wifiSsid: string | null;
  onRefresh: () => void;
}

interface AudioDevice {
  name: string;
  is_input: boolean;
}

export function PrivacyBar({ privacyMode, audioPaused, wifiSsid, onRefresh }: PrivacyBarProps) {
  const [devices, setDevices] = useState<AudioDevice[]>([]);
  const [preferred, setPreferred] = useState<string>("");
  const [toggling, setToggling] = useState(false);

  useEffect(() => {
    fetch(`${API}/api/audio/devices`)
      .then((r) => r.json())
      .then((d) => {
        setDevices(d.devices || []);
        setPreferred(d.preferred || "");
      })
      .catch(() => {});
  }, []);

  const togglePrivacy = useCallback(async () => {
    setToggling(true);
    try {
      await fetch(`${API}/api/privacy/toggle`, { method: "POST" });
      onRefresh();
    } catch {}
    finally { setToggling(false); }
  }, [onRefresh]);

  const switchDevice = useCallback(async (name: string) => {
    try {
      await fetch(`${API}/api/audio/device`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ device: name }),
      });
      setPreferred(name);
    } catch {}
  }, []);

  const isRecording = !privacyMode && !audioPaused;

  return (
    <div className="flex items-center gap-3 rounded-xl border border-border bg-card px-4 py-2.5">
      <Button
        variant={isRecording ? "outline" : "destructive"}
        size="sm"
        className="gap-2 text-xs"
        onClick={togglePrivacy}
        disabled={toggling}
      >
        {isRecording ? (
          <>
            <Mic className="h-3.5 w-3.5" />
            Recording
          </>
        ) : (
          <>
            <MicOff className="h-3.5 w-3.5" />
            Paused
          </>
        )}
      </Button>

      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button variant="ghost" size="sm" className="gap-1.5 text-xs text-muted-foreground max-w-[200px]">
            <span className="truncate">{preferred || "Select device"}</span>
            <ChevronDown className="h-3 w-3 shrink-0" />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="start" className="w-64">
          {devices.filter((d) => d.is_input).map((d) => (
            <DropdownMenuItem
              key={d.name}
              onClick={() => switchDevice(d.name)}
              className={d.name === preferred ? "bg-accent" : ""}
            >
              <Mic className="mr-2 h-3.5 w-3.5" />
              <span className="truncate text-xs">{d.name}</span>
            </DropdownMenuItem>
          ))}
        </DropdownMenuContent>
      </DropdownMenu>

      <div className="ml-auto flex items-center gap-2 text-xs text-muted-foreground">
        {wifiSsid && <span>WiFi: {wifiSsid}</span>}
        {privacyMode ? (
          <span className="flex items-center gap-1 text-yellow-400">
            <ShieldOff className="h-3.5 w-3.5" /> Privacy Mode
          </span>
        ) : (
          <span className="flex items-center gap-1 text-green-400">
            <Shield className="h-3.5 w-3.5" /> Active
          </span>
        )}
      </div>
    </div>
  );
}
