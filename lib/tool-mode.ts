/**
 * Controls how agy accesses tools:
 * - `"pi"` (default): bridge Pi's tools + MCP + skills into agy via a local HTTP MCP server.
 * - `"native"`: let agy run its own built-in tools directly without the Pi bridge.
 */
export type AgyToolMode = "pi" | "native";

export type AgyToolModeCommandAction =
  | { kind: "status" }
  | { kind: "set"; mode: AgyToolMode }
  | { kind: "invalid"; raw: string };

export const DEFAULT_AGY_TOOL_MODE: AgyToolMode = "pi";

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
