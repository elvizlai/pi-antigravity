import { randomUUID } from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { getAgentDir } from "@earendil-works/pi-coding-agent";
import type { AgyExecutionMode } from "../acp/models.ts";
import type { AgyToolMode } from "../bridge/server.ts";

export type { AgyExecutionMode, AgyToolMode };

export const DEFAULT_AGY_TOOL_MODE: AgyToolMode = "pi";

export interface AgySettings {
  toolMode?: AgyToolMode;
  firstResponseTimeoutMs?: number;
}

export type AgyToolModeCommandAction =
  | { kind: "status" }
  | { kind: "set"; mode: AgyToolMode }
  | { kind: "invalid"; raw: string };

export type AgyPlanCommandAction =
  | { kind: "status" }
  | { kind: "set"; enabled: boolean }
  | { kind: "invalid"; raw: string };

export function isAgyToolMode(value: unknown): value is AgyToolMode {
  return value === "pi" || value === "native";
}

export function resolveAgyToolMode(mode: AgyToolMode | undefined): AgyToolMode {
  return mode ?? DEFAULT_AGY_TOOL_MODE;
}

export function formatAgyToolModeStatus(mode: AgyToolMode): string {
  return mode === "pi" ? "acp:pi" : "acp:native";
}

export function parseAgyToolModeArg(raw: string | undefined): AgyToolModeCommandAction {
  const normalized = raw?.trim().toLowerCase() ?? "";
  if (!normalized || normalized === "status") return { kind: "status" };
  if (normalized === "pi" || normalized === "on") return { kind: "set", mode: "pi" };
  if (normalized === "native" || normalized === "off") return { kind: "set", mode: "native" };
  return { kind: "invalid", raw: raw?.trim() ?? "" };
}

export function resolveAgyExecutionMode(planEnabled: boolean): AgyExecutionMode {
  return planEnabled ? "plan" : "agent";
}

export function parseAgyPlanArg(raw: string | undefined): AgyPlanCommandAction {
  const normalized = raw?.trim().toLowerCase() ?? "";
  if (!normalized || normalized === "status") return { kind: "status" };
  if (normalized === "on" || normalized === "enable" || normalized === "true") {
    return { kind: "set", enabled: true };
  }
  if (normalized === "off" || normalized === "disable" || normalized === "false") {
    return { kind: "set", enabled: false };
  }
  return { kind: "invalid", raw: raw?.trim() ?? "" };
}

/** Extension-owned settings file under Pi's agent dir (`~/.pi/agent/antigravity.json`). */
export function agySettingsPath(agentDir: string = getAgentDir()): string {
  return path.join(agentDir, "antigravity.json");
}

function sanitizeSettings(raw: unknown): AgySettings {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) return {};
  const obj = raw as Record<string, unknown>;
  const settings: AgySettings = {};
  if (isAgyToolMode(obj.toolMode)) {
    settings.toolMode = obj.toolMode;
  }
  if (
    typeof obj.firstResponseTimeoutMs === "number" &&
    Number.isFinite(obj.firstResponseTimeoutMs) &&
    obj.firstResponseTimeoutMs > 0
  ) {
    settings.firstResponseTimeoutMs = Math.floor(obj.firstResponseTimeoutMs);
  }
  return settings;
}

/** Read extension settings from `antigravity.json`, returning `{}` when missing or invalid. */
export function loadAgySettings(agentDir: string = getAgentDir()): AgySettings {
  const file = agySettingsPath(agentDir);
  try {
    if (!fs.existsSync(file)) return {};
    return sanitizeSettings(JSON.parse(fs.readFileSync(file, "utf-8")));
  } catch {
    return {};
  }
}

/** Atomically merge and write extension settings to `antigravity.json` (temp file + rename). */
export function saveAgySettings(
  patch: Partial<AgySettings>,
  agentDir: string = getAgentDir(),
): AgySettings {
  const file = agySettingsPath(agentDir);
  const current = loadAgySettings(agentDir);
  const next = sanitizeSettings({ ...current, ...patch });

  fs.mkdirSync(path.dirname(file), { recursive: true });
  const tempFile = path.join(path.dirname(file), `.antigravity-${process.pid}-${randomUUID()}.tmp`);
  try {
    fs.writeFileSync(tempFile, `${JSON.stringify(next, null, 2)}\n`, {
      encoding: "utf-8",
      mode: 0o600,
    });
    fs.renameSync(tempFile, file);
  } catch (err) {
    try {
      fs.rmSync(tempFile, { force: true });
    } catch {
      // Ignore cleanup error on failed atomic write.
    }
    throw err;
  }
  return next;
}
