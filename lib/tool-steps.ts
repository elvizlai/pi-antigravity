import type { AgyActivity } from "./acp-updates.ts";

/** Missing call IDs fall back to the tool name key. */
export function agyToolStepKey(activity: {
  callId?: string;
  stepId?: string | number;
  name: string;
}): string {
  if (activity.callId) return `call:${activity.callId}`;
  if (activity.stepId !== undefined) return `step:${activity.stepId}`;
  return `name:${activity.name}`;
}

/** True only when the step explicitly started an agy daemon (`IsDaemon: true`). */
export function isDaemonToolStep(activity: {
  name: string;
  args?: Record<string, unknown>;
}): boolean {
  const flag = activity.args?.IsDaemon ?? activity.args?.isDaemon;
  return (
    (activity.name === "run_command" || activity.name === "execute_command") &&
    (flag === true || flag === "true")
  );
}

/** Track active tool calls in the current turn; completions clear siblings by key only. */
export function trackActiveToolStep(
  active: Set<string>,
  activity: AgyActivity,
  daemons?: Set<string>,
): void {
  if (activity.type === "tool_start") {
    const key = agyToolStepKey(activity);
    active.add(key);
    if (daemons && isDaemonToolStep(activity)) daemons.add(key);
  } else if (activity.type === "tool_done" || activity.type === "tool_error") {
    const key = agyToolStepKey(activity);
    active.delete(key);
    daemons?.delete(key);
  }
}
