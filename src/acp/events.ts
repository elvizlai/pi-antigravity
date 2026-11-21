import type { JsonObject } from "@earendil-works/pi-ai";

export interface AgyUsage {
  input_tokens?: number;
  output_tokens?: number;
  cache_read_tokens?: number;
  cache_creation_tokens?: number;
  reasoning_tokens?: number;
  thinking_tokens?: number;
  total_tokens?: number;
}

export type AgyToolStatus = "pending" | "in_progress" | "completed" | "failed" | "cancelled";

export interface AgyToolView {
  toolCallId: string;
  name: string;
  title?: string;
  kind?: string;
  status: AgyToolStatus;
  args: JsonObject;
  output?: string;
  startedAtMs: number;
  startedEmitted: boolean;
  finishedEmitted: boolean;
}

export type AgyActivity =
  | { type: "thinking_delta"; text: string }
  | { type: "text_delta"; text: string }
  | { type: "context_usage"; usedTokens: number; sizeTokens: number }
  | {
      type: "tool_start";
      callId: string;
      name: string;
      title?: string;
      kind?: string;
      args: JsonObject;
    }
  | {
      type: "tool_update";
      callId: string;
      name: string;
      title?: string;
      kind?: string;
      args: JsonObject;
    }
  | {
      type: "tool_done";
      callId: string;
      name: string;
      title?: string;
      kind?: string;
      args: JsonObject;
      output?: string;
      durationSeconds?: number;
    }
  | {
      type: "tool_error";
      callId: string;
      name: string;
      title?: string;
      kind?: string;
      args: JsonObject;
      message: string;
      durationSeconds?: number;
    };

function asRecord(value: unknown): Record<string, unknown> | undefined {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : undefined;
}

function nonEmptyString(value: unknown): string | undefined {
  return typeof value === "string" && value.trim().length > 0 ? value.trim() : undefined;
}

function asJsonObject(value: unknown): JsonObject | undefined {
  const rec = asRecord(value);
  return rec ? (rec as JsonObject) : undefined;
}

function inferToolName(update: Record<string, unknown>, existing: AgyToolView | undefined): string {
  const explicit = nonEmptyString(update.toolName) ?? nonEmptyString(update.name);
  if (explicit) return explicit;
  if (existing && existing.name !== "unknown") return existing.name;

  const title = nonEmptyString(update.title) ?? existing?.title;
  if (title) {
    // When agy emits bare tool identifiers as title (e.g. `run_command`, `mcp_...`), keep them.
    if (/^[a-zA-Z0-9_.:-]+$/.test(title)) return title;
    return title.split(/\s+/)[0] || title;
  }

  return nonEmptyString(update.kind) ?? existing?.kind ?? "unknown";
}

function extractTextFromContentBlock(block: unknown): string | undefined {
  const rec = asRecord(block);
  if (!rec) return typeof block === "string" ? block : undefined;

  if (rec.type === "diff") {
    const filePath = nonEmptyString(rec.path);
    const oldText = typeof rec.oldText === "string" ? rec.oldText : undefined;
    const newText = typeof rec.newText === "string" ? rec.newText : undefined;
    const parts = [
      filePath ? `diff: ${filePath}` : "diff",
      oldText !== undefined ? `-${oldText.split("\n").length} lines` : undefined,
      newText !== undefined ? `+${newText.split("\n").length} lines` : undefined,
    ].filter(Boolean);
    return parts.join(" ");
  }

  if (typeof rec.text === "string" && rec.text.length > 0) return rec.text;

  const inner = asRecord(rec.content);
  if (inner && typeof inner.text === "string" && inner.text.length > 0) return inner.text;
  if (typeof rec.content === "string" && rec.content.length > 0) return rec.content;

  return undefined;
}

export function extractAcpToolOutput(rawOutput: unknown, content: unknown): string | undefined {
  const sections: string[] = [];

  if (typeof rawOutput === "string") {
    if (rawOutput.trim().length > 0) sections.push(rawOutput);
  } else {
    const out = asRecord(rawOutput);
    if (out) {
      for (const key of ["output", "stdout", "text", "content", "result", "error", "message"]) {
        const val = out[key];
        if (typeof val === "string" && val.trim().length > 0) sections.push(val);
      }
      const stderr = out.stderr;
      if (typeof stderr === "string" && stderr.trim().length > 0) {
        sections.push(stderr);
      }
      if (sections.length === 0) {
        const serialized = JSON.stringify(out);
        if (serialized && serialized !== "{}") sections.push(serialized);
      }
    }
  }

  if (sections.length === 0 && Array.isArray(content)) {
    for (const block of content) {
      const text = extractTextFromContentBlock(block);
      if (text) sections.push(text);
    }
  }

  if (sections.length === 0) return undefined;
  return sections.join("\n").trimEnd();
}

function normalizeStatus(raw: unknown, fallback: AgyToolStatus): AgyToolStatus {
  if (
    raw === "pending" ||
    raw === "in_progress" ||
    raw === "completed" ||
    raw === "failed" ||
    raw === "cancelled"
  ) {
    return raw;
  }
  return fallback;
}

function readDurationSeconds(startedAtMs: number, now: number): number | undefined {
  const elapsed = (now - startedAtMs) / 1000;
  return Number.isFinite(elapsed) && elapsed >= 0 ? elapsed : undefined;
}

/** Merge a `tool_call` or `tool_call_update` payload into the tracked map and return resulting activity. */
export function mergeAgyAcpTool(
  tools: Map<string, AgyToolView>,
  update: Record<string, unknown>,
  now = Date.now(),
): AgyActivity | undefined {
  const toolCallId = nonEmptyString(update.toolCallId);
  if (!toolCallId) return undefined;

  const existing = tools.get(toolCallId);
  const name = inferToolName(update, existing);
  const title = nonEmptyString(update.title) ?? existing?.title;
  const kind = nonEmptyString(update.kind) ?? existing?.kind;
  const status = normalizeStatus(update.status, existing?.status ?? "pending");
  const incomingArgs = asJsonObject(update.rawInput);
  const mergedArgs: JsonObject = incomingArgs
    ? { ...(existing?.args ?? {}), ...incomingArgs }
    : (existing?.args ?? {});

  // Add first location path if args has none so renderers / mappers still have a path hint.
  if (
    Object.keys(mergedArgs).length === 0 &&
    Array.isArray(update.locations) &&
    update.locations.length > 0
  ) {
    const firstLoc = asRecord(update.locations[0]);
    const locPath = nonEmptyString(firstLoc?.path);
    if (locPath) mergedArgs.path = locPath;
  }

  const output = extractAcpToolOutput(update.rawOutput, update.content) ?? existing?.output;

  const view: AgyToolView = {
    toolCallId,
    name,
    title,
    kind,
    status,
    args: mergedArgs,
    output,
    startedAtMs: existing?.startedAtMs ?? now,
    startedEmitted: existing?.startedEmitted ?? false,
    finishedEmitted: existing?.finishedEmitted ?? false,
  };
  tools.set(toolCallId, view);

  if (status === "completed") {
    if (view.finishedEmitted) return undefined;
    view.startedEmitted = true;
    view.finishedEmitted = true;
    return {
      type: "tool_done",
      callId: view.toolCallId,
      name: view.name,
      title: view.title,
      kind: view.kind,
      args: view.args,
      output: view.output,
      durationSeconds: readDurationSeconds(view.startedAtMs, now),
    };
  }

  if (status === "failed" || status === "cancelled") {
    if (view.finishedEmitted) return undefined;
    view.startedEmitted = true;
    view.finishedEmitted = true;
    return {
      type: "tool_error",
      callId: view.toolCallId,
      name: view.name,
      title: view.title,
      kind: view.kind,
      args: view.args,
      message:
        view.output || (status === "cancelled" ? "Tool call cancelled." : "Tool call failed."),
      durationSeconds: readDurationSeconds(view.startedAtMs, now),
    };
  }

  if (!view.startedEmitted) {
    view.startedEmitted = true;
    return {
      type: "tool_start",
      callId: view.toolCallId,
      name: view.name,
      title: view.title,
      kind: view.kind,
      args: view.args,
    };
  }

  return {
    type: "tool_update",
    callId: view.toolCallId,
    name: view.name,
    title: view.title,
    kind: view.kind,
    args: view.args,
  };
}

/** Convert an ACP `session/update` object into zero or one typed turn activities. */
export function acpUpdateToActivity(
  update: unknown,
  tools: Map<string, AgyToolView>,
  now = Date.now(),
): AgyActivity | undefined {
  const rec = asRecord(update);
  if (!rec) return undefined;

  switch (rec.sessionUpdate) {
    case "agent_message_chunk": {
      const content = asRecord(rec.content);
      const text = typeof content?.text === "string" ? content.text : undefined;
      return text && text.length > 0 ? { type: "text_delta", text } : undefined;
    }
    case "agent_thought_chunk": {
      const content = asRecord(rec.content);
      const text = typeof content?.text === "string" ? content.text : undefined;
      return text && text.length > 0 ? { type: "thinking_delta", text } : undefined;
    }
    case "usage_update": {
      const used = rec.used;
      const size = rec.size;
      if (
        typeof used === "number" &&
        Number.isFinite(used) &&
        typeof size === "number" &&
        Number.isFinite(size) &&
        size > 0
      ) {
        return { type: "context_usage", usedTokens: used, sizeTokens: size };
      }
      return undefined;
    }
    case "tool_call":
    case "tool_call_update":
      return mergeAgyAcpTool(tools, rec, now);
    default:
      return undefined;
  }
}

/** Normalize ACP `session/prompt` `usage` payload into `AgyUsage`. */
export function normalizeAcpPromptUsage(raw: unknown): AgyUsage | undefined {
  const rec = asRecord(raw);
  if (!rec) return undefined;
  const num = (key: string): number | undefined => {
    const val = rec[key];
    return typeof val === "number" && Number.isFinite(val) ? val : undefined;
  };
  const usage: AgyUsage = {
    input_tokens: num("inputTokens") ?? num("input_tokens"),
    output_tokens: num("outputTokens") ?? num("output_tokens"),
    cache_read_tokens: num("cachedReadTokens") ?? num("cache_read_tokens"),
    cache_creation_tokens: num("cachedWriteTokens") ?? num("cache_creation_tokens"),
    reasoning_tokens: num("thoughtTokens") ?? num("reasoning_tokens"),
    thinking_tokens: num("thoughtTokens") ?? num("thinking_tokens") ?? num("reasoning_tokens"),
    total_tokens: num("totalTokens") ?? num("total_tokens"),
  };
  if (
    usage.input_tokens === undefined &&
    usage.output_tokens === undefined &&
    usage.cache_read_tokens === undefined &&
    usage.cache_creation_tokens === undefined &&
    usage.reasoning_tokens === undefined &&
    usage.thinking_tokens === undefined &&
    usage.total_tokens === undefined
  ) {
    return undefined;
  }
  return usage;
}

function sanitizeMcpIdentifier(name: string): string {
  return name.replace(/[^a-zA-Z0-9_]/g, "_");
}

/**
 * True when an ACP tool call represents Pi's own MCP bridge tool (`pi__*` or
 * `call_mcp_tool` targeting the bridge server). User-configured external MCP
 * servers stay false so their calls still appear in Pi as antigravity cards.
 */
export function isBridgedMcpTool(
  activity: {
    name: string;
    args?: Record<string, unknown>;
  },
  bridgeServerName?: string,
): boolean {
  const { name, args } = activity;
  if (name.startsWith("pi__")) return true;

  if (bridgeServerName) {
    const sanitized = sanitizeMcpIdentifier(bridgeServerName);
    if (name.startsWith(`mcp_${sanitized}_`) || name.startsWith(`mcp_${bridgeServerName}_`)) {
      return true;
    }
  }

  if (name === "call_mcp_tool") {
    const serverName = nonEmptyString(args?.ServerName ?? args?.serverName);
    const toolName = nonEmptyString(args?.ToolName ?? args?.toolName);
    if (bridgeServerName && serverName === bridgeServerName) return true;
    if (toolName?.startsWith("pi__")) return true;
  }

  return false;
}

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
