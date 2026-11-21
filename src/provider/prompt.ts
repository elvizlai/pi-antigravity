import {
  contentText,
  getCurrentSystemMessage,
  type ImageContent,
  type TextContent,
  type TranscriptContext,
} from "@earendil-works/pi-ai";
import type { AcpContentBlock } from "../acp/client.ts";
import { WRAPPER_TOOL_NAME } from "../bridge/server.ts";

export { WRAPPER_TOOL_NAME };

/**
 * Display-only wrapper tool that replays recorded agy tool results.
 * Intentionally invisible to models: empty description and no prompt snippets.
 */
export const WRAPPER_TOOL_DESCRIPTION = "";

export const BRIDGE_PENDING_TOOL_MESSAGE =
  "Started and still running at tool handoff. Unfinished commands may be cancelled when " +
  "the agy turn ends.";

/**
 * Terminal error synthesized when the runtime closes a wave on still-running
 * `IsDaemon` work so Pi finishes the assistant turn without cutting off daemons.
 */
export const AGY_PARKED_TURN_ERROR =
  "timeout waiting for response — agy is still running work as a background task";

/** Maximum UTF-8 byte budget allocated to text blocks in a single `session/prompt`. */
export const MAX_AGY_PROMPT_BYTES = 185_000;

const OMITTED_SYSTEM_SECTIONS = new Set(["tools", "skills", "cwd"]);

/** Replay diagnostics must not promise that an unobserved task survived or stopped. */
export function agyIncompleteToolError(tool: string, resultError?: string): string {
  if (tool === "schedule") {
    return (
      "agy stopped reporting before the scheduled wait completed. The timer's final state " +
      "is unknown and it may still fire. Do not reschedule it blindly: confirm the work it " +
      "was waiting on instead, using pi's own tools."
    );
  }
  if (
    tool === "run_command" &&
    (!resultError || /timeout waiting for response/i.test(resultError))
  ) {
    return (
      "agy did not report this command completing. It may have become a background task " +
      '(headless agy can report "timeout waiting for response"). Confirm whether the command ' +
      "is still running before retrying; use pi's own bash for long-lived commands."
    );
  }
  return "agy tool call did not complete.";
}

export function omittedImagesPrompt(images: number): string {
  return `(${images} image(s) omitted — this agy request is text-only)`;
}

/**
 * Extract instructions from Pi's normalized transcript system message via
 * `@earendil-works/pi-ai` public helpers (`getCurrentSystemMessage` + `contentText`).
 * Strips Pi-specific `"tools"`, `"skills"`, and `"cwd"` sections structurally by
 * key instead of fragile prompt text regexes.
 */
export function extractSystemPromptAppend(
  messages: TranscriptContext["messages"],
): string | undefined {
  const sys = getCurrentSystemMessage(messages);
  if (!sys) return undefined;

  const parts: string[] = [];
  const seen = new Set<string>();
  const pushUnique = (raw: string | undefined) => {
    const trimmed = raw?.trim();
    if (!trimmed || seen.has(trimmed)) return;
    seen.add(trimmed);
    parts.push(trimmed);
  };

  pushUnique(contentText(sys.content));
  if (sys.sections) {
    for (const [key, value] of Object.entries(sys.sections)) {
      if (OMITTED_SYSTEM_SECTIONS.has(key) || typeof value !== "string") continue;
      pushUnique(value);
    }
  }

  return parts.length > 0 ? parts.join("\n\n") : undefined;
}

const TRUNCATION_NOTICE = "\n...[truncated]";

function truncateUtf8Tail(text: string, maxBytes: number): string {
  if (Buffer.byteLength(text, "utf8") <= maxBytes) return text;
  const noticeBytes = Buffer.byteLength(TRUNCATION_NOTICE, "utf8");
  if (maxBytes <= noticeBytes) {
    return Buffer.from(text, "utf8").subarray(0, maxBytes).toString("utf8");
  }
  const budget = maxBytes - noticeBytes;
  // Walk backwards from char boundary so surrogate pairs / multi-byte UTF-8 stay valid.
  const buf = Buffer.from(text, "utf8");
  let end = Math.max(0, buf.length);
  const start = Math.max(0, end - budget);
  let slice = buf.subarray(start, end).toString("utf8");
  // Trim any leading replacement char caused by slicing mid-codepoint.
  while (slice.length > 0 && slice.charCodeAt(0) === 0xfffd) {
    slice = slice.slice(1);
  }
  end = slice.length;
  return `${TRUNCATION_NOTICE.trimStart()}\n${slice.slice(0, end)}`;
}

function truncateUtf8Head(text: string, maxBytes: number): string {
  if (Buffer.byteLength(text, "utf8") <= maxBytes) return text;
  const noticeBytes = Buffer.byteLength(TRUNCATION_NOTICE, "utf8");
  if (maxBytes <= noticeBytes) {
    return Buffer.from(text, "utf8").subarray(0, maxBytes).toString("utf8");
  }
  const budget = maxBytes - noticeBytes;
  let end = Math.min(text.length, budget);
  while (end > 0 && Buffer.byteLength(text.slice(0, end), "utf8") > budget) {
    end -= Math.max(1, Math.floor((Buffer.byteLength(text.slice(0, end), "utf8") - budget) / 3));
  }
  return `${text.slice(0, end)}${TRUNCATION_NOTICE}`;
}

function clampPromptTextBlocks(
  blocks: readonly AcpContentBlock[],
  maxBytes: number,
): AcpContentBlock[] {
  const textIndexes: number[] = [];
  let totalBytes = 0;
  for (let i = 0; i < blocks.length; i += 1) {
    const block = blocks[i];
    if (block.type === "text") {
      textIndexes.push(i);
      totalBytes += Buffer.byteLength(block.text, "utf8");
    }
  }
  if (totalBytes <= maxBytes || textIndexes.length === 0) return [...blocks];

  // Keep the most recent user text intact first, capping older injected context.
  const kept = [...blocks];
  let remaining = maxBytes;
  const reverseResult: (AcpContentBlock | undefined)[] = new Array(blocks.length);

  for (let i = blocks.length - 1; i >= 0; i -= 1) {
    const block = kept[i];
    if (block.type === "image") {
      reverseResult[i] = block;
      continue;
    }
    const bytes = Buffer.byteLength(block.text, "utf8");
    if (remaining <= 0) {
      continue;
    }
    if (bytes <= remaining) {
      reverseResult[i] = block;
      remaining -= bytes;
      continue;
    }
    const truncated =
      i === textIndexes[textIndexes.length - 1]
        ? truncateUtf8Head(block.text, remaining)
        : truncateUtf8Tail(block.text, remaining);
    if (truncated.trim()) {
      reverseResult[i] = { type: "text", text: truncated };
    }
    remaining = 0;
  }

  return reverseResult.filter((b): b is AcpContentBlock => b !== undefined);
}

/**
 * Collect contiguous trailing `user` messages (skipping `system` entries) in
 * chronological order so extension-injected context + user prompt merge
 * seamlessly for `session/prompt`.
 */
export function extractTrailingUserPrompt(
  messages: TranscriptContext["messages"],
  options?: { maxBytes?: number; allowImages?: boolean },
): AcpContentBlock[] {
  const trailingUsers: Extract<TranscriptContext["messages"][number], { role: "user" }>[] = [];
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    const msg = messages[i];
    if (msg.role === "system") continue;
    if (msg.role !== "user") break;
    trailingUsers.unshift(msg);
  }

  if (trailingUsers.length === 0) return [];

  const blocks: AcpContentBlock[] = [];
  let omittedImages = 0;
  const allowImages = options?.allowImages ?? true;

  for (const msg of trailingUsers) {
    if (typeof msg.content === "string") {
      if (msg.content.trim()) {
        blocks.push({ type: "text", text: msg.content });
      }
      continue;
    }
    for (const part of msg.content) {
      if (part.type === "text") {
        if (part.text.trim()) blocks.push({ type: "text", text: part.text });
      } else if (part.type === "image") {
        if (allowImages) {
          blocks.push({ type: "image", data: part.data, mimeType: part.mimeType });
        } else {
          omittedImages += 1;
        }
      }
    }
  }

  if (omittedImages > 0) {
    blocks.push({ type: "text", text: omittedImagesPrompt(omittedImages) });
  }

  const maxBytes = options?.maxBytes ?? MAX_AGY_PROMPT_BYTES;
  return clampPromptTextBlocks(blocks, maxBytes);
}

export interface CollectedToolResult {
  callId: string;
  name: string;
  content: (TextContent | ImageContent)[];
  isError: boolean;
}

/**
 * Collect matching trailing `toolResult` messages (skipping `system` entries)
 * since the last assistant turn so pending bridged MCP calls can resolve cleanly.
 */
export function collectTurnToolResults(
  messages: TranscriptContext["messages"],
  pendingCallIds: ReadonlySet<string>,
): CollectedToolResult[] {
  if (pendingCallIds.size === 0) return [];
  const collected: CollectedToolResult[] = [];
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    const msg = messages[i];
    if (msg.role === "system") continue;
    if (msg.role !== "toolResult") break;
    if (!pendingCallIds.has(msg.toolCallId)) continue;
    collected.unshift({
      callId: msg.toolCallId,
      name: msg.toolName,
      content: msg.content,
      isError: Boolean(msg.isError),
    });
  }
  return collected;
}
