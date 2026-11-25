import type { JsonObject } from "@earendil-works/pi-ai";
import { asRecord } from "./shared.ts";

export type AcpToolLocation = {
  path: string;
  line?: number;
};

export interface AgyUsage {
  inputTokens?: number;
  outputTokens?: number;
  cachedReadTokens?: number;
  cachedWriteTokens?: number;
  thoughtTokens?: number;
  totalTokens?: number;
}

export type AgyToolStatus = "pending" | "in_progress" | "completed" | "failed" | "cancelled";

export interface AgyToolView {
  toolCallId: string;
  title: string;
  name?: string;
  kind?: string;
  locations?: AcpToolLocation[];
  rawInput?: JsonObject;
  status: AgyToolStatus;
  output?: string;
  startedAtMs: number;
  startedEmitted: boolean;
  deferredEmitted: boolean;
  finishedEmitted: boolean;
}

export type AgyActivity =
  | { type: "thinking_delta"; text: string }
  | { type: "text_delta"; text: string }
  | { type: "context_usage"; usedTokens: number; sizeTokens: number }
  | {
      type: "tool_start";
      callId: string;
      title: string;
      name?: string;
      kind?: string;
      locations?: AcpToolLocation[];
      rawInput?: JsonObject;
    }
  | {
      type: "tool_update";
      callId: string;
      title: string;
      name?: string;
      kind?: string;
      locations?: AcpToolLocation[];
      rawInput?: JsonObject;
    }
  | {
      type: "tool_done";
      callId: string;
      title: string;
      name?: string;
      kind?: string;
      locations?: AcpToolLocation[];
      rawInput?: JsonObject;
      output?: string;
      durationSeconds?: number;
    }
  | {
      type: "tool_error";
      callId: string;
      title: string;
      name?: string;
      kind?: string;
      locations?: AcpToolLocation[];
      rawInput?: JsonObject;
      message: string;
      durationSeconds?: number;
    };

function nonEmptyString(value: unknown): string | undefined {
  return typeof value === "string" && value.trim().length > 0 ? value.trim() : undefined;
}

function asJsonObject(value: unknown): JsonObject | undefined {
  const rec = asRecord(value);
  return rec ? (rec as JsonObject) : undefined;
}

function parseAcpLocations(raw: unknown): AcpToolLocation[] | undefined {
  if (!Array.isArray(raw)) return undefined;
  const locations: AcpToolLocation[] = [];
  for (const item of raw) {
    const rec = asRecord(item);
    const filePath = nonEmptyString(rec?.path);
    if (!filePath) continue;
    const line = typeof rec?.line === "number" && Number.isFinite(rec.line) ? rec.line : undefined;
    locations.push(line !== undefined ? { path: filePath, line } : { path: filePath });
  }
  return locations.length > 0 ? locations : undefined;
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

/** Extract display text preferring standard ACP `content` blocks, falling back to `rawOutput`. */
export function extractAcpToolOutput(rawOutput: unknown, content: unknown): string | undefined {
  const sections: string[] = [];

  if (Array.isArray(content)) {
    for (const block of content) {
      const text = extractTextFromContentBlock(block);
      if (text) sections.push(text);
    }
  }

  if (sections.length === 0) {
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

/** Merge a standard ACP `tool_call` or `tool_call_update` payload into the tracked map. */
function mergeAgyAcpTool(
  tools: Map<string, AgyToolView>,
  update: Record<string, unknown>,
  now = Date.now(),
): AgyActivity | undefined {
  const toolCallId = nonEmptyString(update.toolCallId);
  if (!toolCallId) return undefined;

  const existing = tools.get(toolCallId);
  const kind = nonEmptyString(update.kind) ?? existing?.kind;
  const title = nonEmptyString(update.title) ?? existing?.title ?? kind ?? "tool";
  const name = nonEmptyString(update.name) ?? existing?.name;
  const status = normalizeStatus(update.status, existing?.status ?? "pending");
  const locations = parseAcpLocations(update.locations) ?? existing?.locations;
  const incomingRaw = asJsonObject(update.rawInput);
  const rawInput = incomingRaw
    ? { ...(existing?.rawInput ?? {}), ...incomingRaw }
    : existing?.rawInput;
  const output = extractAcpToolOutput(update.rawOutput, update.content) ?? existing?.output;

  const view: AgyToolView = {
    toolCallId,
    title,
    ...(name ? { name } : {}),
    ...(kind ? { kind } : {}),
    ...(locations ? { locations } : {}),
    ...(rawInput ? { rawInput } : {}),
    status,
    output,
    startedAtMs: existing?.startedAtMs ?? now,
    startedEmitted: existing?.startedEmitted ?? false,
    deferredEmitted: existing?.deferredEmitted ?? false,
    finishedEmitted: existing?.finishedEmitted ?? false,
  };
  tools.set(toolCallId, view);

  const shared = {
    callId: view.toolCallId,
    title: view.title,
    ...(view.name ? { name: view.name } : {}),
    ...(view.kind ? { kind: view.kind } : {}),
    ...(view.locations ? { locations: view.locations } : {}),
    ...(view.rawInput ? { rawInput: view.rawInput } : {}),
  };

  if (status === "completed") {
    if (view.finishedEmitted) return undefined;
    view.startedEmitted = true;
    view.finishedEmitted = true;
    return {
      type: "tool_done",
      ...shared,
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
      ...shared,
      message:
        view.output || (status === "cancelled" ? "Tool call cancelled." : "Tool call failed."),
      durationSeconds: readDurationSeconds(view.startedAtMs, now),
    };
  }

  if (!view.startedEmitted) {
    view.startedEmitted = true;
    return { type: "tool_start", ...shared };
  }

  return { type: "tool_update", ...shared };
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
    inputTokens: num("inputTokens"),
    outputTokens: num("outputTokens"),
    cachedReadTokens: num("cachedReadTokens"),
    cachedWriteTokens: num("cachedWriteTokens"),
    thoughtTokens: num("thoughtTokens"),
    totalTokens: num("totalTokens"),
  };
  if (
    usage.inputTokens === undefined &&
    usage.outputTokens === undefined &&
    usage.cachedReadTokens === undefined &&
    usage.cachedWriteTokens === undefined &&
    usage.thoughtTokens === undefined &&
    usage.totalTokens === undefined
  ) {
    return undefined;
  }
  return usage;
}

function sanitizeMcpIdentifier(name: string): string {
  return name.replace(/[^a-zA-Z0-9_]/g, "_");
}

const BRIDGED_TOOL_PREFIX_RE = /(?:^|[\s:_/])pi__[a-zA-Z0-9_-]+/;

/**
 * True when an ACP tool call belongs to Pi's own MCP bridge server (`pi__*` or
 * `mcp_<bridgeServerName>_*`). External MCP servers stay false so their calls
 * still render in Pi as wrapper cards.
 */
export function isBridgedMcpTool(
  activity: {
    title?: string;
    name?: string;
  },
  bridgeServerName?: string,
): boolean {
  const sanitized = bridgeServerName ? sanitizeMcpIdentifier(bridgeServerName) : undefined;
  for (const raw of [activity.name, activity.title]) {
    if (!raw) continue;
    if (BRIDGED_TOOL_PREFIX_RE.test(raw)) return true;
    if (
      bridgeServerName &&
      sanitized &&
      (raw.includes(`mcp_${sanitized}_`) ||
        raw.includes(`mcp_${bridgeServerName}_`) ||
        raw === bridgeServerName ||
        raw.startsWith(`${bridgeServerName}:`) ||
        raw.startsWith(`${bridgeServerName}/`))
    ) {
      return true;
    }
  }
  return false;
}
