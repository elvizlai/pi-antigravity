/**
 * ACP execution modes for `session/set_config_option` (`configId: "mode"`).
 *
 * Verified live against `agy 1.5.13` ACP `session/new` `configOptions`:
 * `"default" | "accept-edits" | "auto" | "plan"`.
 */
export type AgyExecutionMode = "default" | "accept-edits" | "auto" | "plan";

export const DEFAULT_AGY_EXECUTION_MODE: AgyExecutionMode = "auto";

export function isAgyExecutionMode(value: unknown): value is AgyExecutionMode {
  return value === "default" || value === "accept-edits" || value === "auto" || value === "plan";
}

export function isAgyPlanMode(mode: AgyExecutionMode): boolean {
  return mode === "plan";
}

export function toggleAgyPlanMode(mode: AgyExecutionMode): AgyExecutionMode {
  return mode === "plan" ? "auto" : "plan";
}

/** Map ACP execution mode to the ACP `mode` config option value. */
export function resolveAcpModeValue(mode: AgyExecutionMode): AgyExecutionMode {
  return mode;
}

/** Plan mode label used in Pi status/notifications. */
export function formatAgyExecutionModeStatus(mode: AgyExecutionMode): string {
  return mode;
}
