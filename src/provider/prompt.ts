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

/** Fallback error recorded when an ACP tool call has not reported completion at turn end. */
export const INCOMPLETE_ACP_TOOL_MESSAGE =
  "ACP tool call did not report completion before the turn ended.";

/** Maximum UTF-8 byte budget allocated to text blocks in a single `session/prompt`. */
export const MAX_AGY_PROMPT_BYTES = 185_000;

const OMITTED_SYSTEM_SECTIONS = new Set(["preamble", "tools", "rules", "docs", "skills", "cwd"]);

export function omittedImagesPrompt(images: number): string {
  return `(${images} image(s) omitted — this agy request is text-only)`;
}

/**
 * Extract custom (non-harness) instructions from Pi's normalized transcript
 * system message via `@earendil-works/pi-ai` (`getCurrentSystemMessage` + `contentText`).
 * Omits all built-in Pi sections (`preamble`, `tools`, `rules`, `docs`, `skills`, `cwd`)
 * so standard Pi harness framing is never forwarded to Antigravity.
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

/**
 * Collect contiguous trailing `user` messages (skipping `system` entries) in
 * chronological order and merge all text parts into a single `text` block
 * (followed by any `image` blocks) within the UTF-8 byte ceiling.
 */
export function extractTrailingUserPrompt(
  messages: TranscriptContext["messages"],
  options?: { maxBytes?: number; allowImages?: boolean; customSystemContext?: string },
): AcpContentBlock[] {
  const trailingUsers: Extract<TranscriptContext["messages"][number], { role: "user" }>[] = [];
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    const msg = messages[i];
    if (msg.role === "system") continue;
    if (msg.role !== "user") break;
    trailingUsers.unshift(msg);
  }

  if (trailingUsers.length === 0) return [];

  const textParts: string[] = [];
  const imageBlocks: AcpContentBlock[] = [];
  let omittedImages = 0;
  const allowImages = options?.allowImages ?? true;

  const customSys = options?.customSystemContext?.trim();
  if (customSys) {
    textParts.push(customSys);
  }

  for (const msg of trailingUsers) {
    if (typeof msg.content === "string") {
      if (msg.content.trim()) {
        textParts.push(msg.content);
      }
      continue;
    }
    for (const part of msg.content) {
      if (part.type === "text") {
        if (part.text.trim()) textParts.push(part.text);
      } else if (part.type === "image") {
        if (allowImages) {
          imageBlocks.push({ type: "image", data: part.data, mimeType: part.mimeType });
        } else {
          omittedImages += 1;
        }
      }
    }
  }

  if (omittedImages > 0) {
    textParts.push(omittedImagesPrompt(omittedImages));
  }

  const maxBytes = options?.maxBytes ?? MAX_AGY_PROMPT_BYTES;
  const joined = textParts.join("\n\n");
  const boundedText = joined ? truncateUtf8Tail(joined, maxBytes) : "";

  const result: AcpContentBlock[] = [];
  if (boundedText) {
    result.push({ type: "text", text: boundedText });
  }
  result.push(...imageBlocks);
  return result;
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
    if (msg.role === "system" || msg.role === "user") continue;
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
