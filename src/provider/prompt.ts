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

/** Status recorded when a native ACP tool continues running in the background after the turn ends. */
export const BACKGROUND_ACP_TOOL_MESSAGE = "Started and continuing in the background.";

/** Maximum UTF-8 byte budget allocated to text blocks in a single `session/prompt`. */
export const MAX_AGY_PROMPT_BYTES = 185_000;
const SEEDED_TOOL_RESULT_MAX_CHARS = 2_000;
const SEEDED_TOOL_ARGS_MAX_CHARS = 500;
const COMPACTION_CHECKPOINT_PREFIX =
  "The conversation history before this point was compacted into the following summary:";
const BRANCH_CHECKPOINT_PREFIX =
  "The following is a summary of a branch that this conversation came back from:";
const SEEDED_HISTORY_HEADER =
  "The following <conversation_history> is the prior context of this Pi session. Use it as background context and respond to the user message that follows it.\n\n<conversation_history>\n";
const SEEDED_HISTORY_FOOTER = "\n</conversation_history>";

const OMITTED_SYSTEM_SECTIONS = new Set(["preamble", "tools", "rules", "docs", "skills", "cwd"]);

function omittedImagesPrompt(images: number): string {
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

function truncateUtf8Head(text: string, maxBytes: number): string {
  if (Buffer.byteLength(text, "utf8") <= maxBytes) return text;
  const noticeBytes = Buffer.byteLength(TRUNCATION_NOTICE, "utf8");
  const buf = Buffer.from(text, "utf8");
  const sliceBytes = maxBytes <= noticeBytes ? Math.max(0, maxBytes) : maxBytes - noticeBytes;
  let slice = buf.subarray(0, sliceBytes).toString("utf8");
  // Trim any trailing replacement char caused by slicing mid-codepoint.
  while (slice.length > 0 && slice.charCodeAt(slice.length - 1) === 0xfffd) {
    slice = slice.slice(0, -1);
  }
  return maxBytes <= noticeBytes ? slice : `${slice}${TRUNCATION_NOTICE}`;
}

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
  return options?.customSystemContext
    ? prependSystemContext(result, options.customSystemContext, maxBytes)
    : result;
}

function truncateHeadChars(text: string, maxChars: number): string {
  if (text.length <= maxChars) return text;
  return `${text.slice(0, maxChars)}\n...[truncated]`;
}

function extractUserMessageTextAndImages(
  msg: Extract<TranscriptContext["messages"][number], { role: "user" }>,
): { text: string; imageCount: number } {
  if (typeof msg.content === "string") {
    return { text: msg.content.trim(), imageCount: 0 };
  }
  const textParts: string[] = [];
  let imageCount = 0;
  for (const part of msg.content) {
    if (part.type === "text") {
      if (part.text.trim()) textParts.push(part.text);
    } else if (part.type === "image") {
      imageCount += 1;
    }
  }
  return { text: textParts.join("\n\n").trim(), imageCount };
}

function serializePriorMessage(msg: TranscriptContext["messages"][number]): string[] {
  if (msg.role === "user") {
    const { text, imageCount } = extractUserMessageTextAndImages(msg);
    const parts = [text, imageCount > 0 ? `[${imageCount} image(s) in prior turn]` : ""]
      .filter(Boolean)
      .join("\n\n");
    return parts ? [`[User]:\n${parts}`] : [];
  }

  if (msg.role === "assistant") {
    const hasTextOrTool = msg.content.some(
      (b) => (b.type === "text" && b.text.trim().length > 0) || b.type === "toolCall",
    );
    if (msg.stopReason === "aborted" || (msg.stopReason === "error" && !hasTextOrTool)) {
      return [];
    }
    const items: string[] = [];
    const textParts: string[] = [];
    const toolCalls: string[] = [];
    for (const block of msg.content) {
      if (block.type === "text" && block.text.trim()) {
        textParts.push(block.text);
      } else if (block.type === "toolCall") {
        let serializedArgs = "{}";
        try {
          serializedArgs = JSON.stringify(block.arguments ?? {});
        } catch {
          serializedArgs = "{}";
        }
        toolCalls.push(
          `${block.name}(${truncateHeadChars(serializedArgs, SEEDED_TOOL_ARGS_MAX_CHARS)})`,
        );
      }
    }
    if (textParts.length > 0) {
      items.push(`[Assistant]:\n${textParts.join("\n\n")}`);
    }
    if (toolCalls.length > 0) {
      items.push(`[Assistant tool calls]: ${toolCalls.join("; ")}`);
    }
    return items;
  }

  if (msg.role === "toolResult") {
    const textParts: string[] = [];
    let imageCount = 0;
    for (const part of msg.content) {
      if (part.type === "text" && part.text.trim()) {
        textParts.push(part.text);
      } else if (part.type === "image") {
        imageCount += 1;
      }
    }
    const body = [
      textParts.join("\n"),
      imageCount > 0 ? `[${imageCount} image(s) in prior tool result]` : "",
    ]
      .filter(Boolean)
      .join("\n");
    if (!body) return [];
    const bounded = truncateHeadChars(body, SEEDED_TOOL_RESULT_MAX_CHARS);
    const status = msg.isError ? ", error" : "";
    return [`[Tool result (${msg.toolName}${status})]:\n${bounded}`];
  }

  return [];
}

/**
 * Attach optional custom system context while staying within the UTF-8 byte ceiling.
 * When the first text block starts with `/` (an `agy` slash command or skill), keep it
 * in block 0 untouched and append `customSystemContext` as a second text block so
 * `agy` still expands the leading command natively.
 */
export function prependSystemContext(
  blocks: readonly AcpContentBlock[],
  customSystemContext: string | undefined,
  maxBytes = MAX_AGY_PROMPT_BYTES,
): AcpContentBlock[] {
  const customSys = customSystemContext?.trim();
  if (!customSys) return [...blocks];
  const out = [...blocks];
  const firstTextIdx = out.findIndex((block) => block.type === "text");
  if (firstTextIdx >= 0 && out[firstTextIdx]?.type === "text") {
    const firstText = out[firstTextIdx].text;
    if (firstText.trimStart().startsWith("/")) {
      const firstBytes = Buffer.byteLength(firstText, "utf8");
      if (firstBytes >= maxBytes) {
        out[firstTextIdx] = { type: "text", text: truncateUtf8Head(firstText, maxBytes) };
      } else {
        const trailingText = truncateUtf8Tail(customSys, maxBytes - firstBytes);
        if (trailingText) {
          out.splice(firstTextIdx + 1, 0, { type: "text", text: trailingText });
        }
      }
    } else {
      out[firstTextIdx] = {
        type: "text",
        text: truncateUtf8Tail(`${customSys}\n\n${firstText}`, maxBytes),
      };
    }
  } else {
    out.unshift({ type: "text", text: truncateUtf8Tail(customSys, maxBytes) });
  }
  return out;
}

/**
 * Build an ACP `session/prompt` payload for a fresh ACP session that must inherit
 * prior projected transcript context (e.g., after `/compact`, `/fork`, `/tree`, or
 * cross-provider switch), keeping the live trailing user prompt and images intact.
 */
export function buildSeededSessionPrompt(
  messages: TranscriptContext["messages"],
  options?: {
    maxBytes?: number;
    allowImages?: boolean;
    customSystemContext?: string;
    trailingPrompt?: readonly AcpContentBlock[];
  },
): AcpContentBlock[] {
  let trailingStart = messages.length;
  const trailingUsers: Extract<TranscriptContext["messages"][number], { role: "user" }>[] = [];
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    const msg = messages[i];
    if (msg.role === "system") continue;
    if (msg.role !== "user") break;
    trailingUsers.unshift(msg);
    trailingStart = i;
  }

  const fallbackLive = () =>
    options?.trailingPrompt
      ? prependSystemContext(options.trailingPrompt, options.customSystemContext, options.maxBytes)
      : extractTrailingUserPrompt(messages, options);

  const priorMessages = messages.slice(0, trailingStart).filter((msg) => msg.role !== "system");
  if (priorMessages.length === 0) {
    return fallbackLive();
  }

  let checkpointSummary: string | undefined;
  let historyStart = 0;
  const firstPrior = priorMessages[0];
  if (firstPrior?.role === "user") {
    const { text } = extractUserMessageTextAndImages(firstPrior);
    if (
      text.startsWith(COMPACTION_CHECKPOINT_PREFIX) ||
      text.startsWith(BRANCH_CHECKPOINT_PREFIX)
    ) {
      checkpointSummary = text;
      historyStart = 1;
    }
  }

  const historyItems: string[] = [];
  for (let i = historyStart; i < priorMessages.length; i += 1) {
    historyItems.push(...serializePriorMessage(priorMessages[i]));
  }

  if (!checkpointSummary && historyItems.length === 0) {
    return fallbackLive();
  }

  const liveParts: string[] = [];
  const imageBlocks: AcpContentBlock[] = [];
  let omittedImages = 0;
  const allowImages = options?.allowImages ?? true;
  if (options?.trailingPrompt) {
    for (const block of options.trailingPrompt) {
      if (block.type === "text") {
        if (block.text.trim()) liveParts.push(block.text);
      } else if (block.type === "image") {
        if (allowImages) {
          imageBlocks.push({ type: "image", data: block.data, mimeType: block.mimeType });
        } else {
          omittedImages += 1;
        }
      }
    }
  } else {
    for (const msg of trailingUsers) {
      if (typeof msg.content === "string") {
        if (msg.content.trim()) liveParts.push(msg.content);
        continue;
      }
      for (const part of msg.content) {
        if (part.type === "text") {
          if (part.text.trim()) liveParts.push(part.text);
        } else if (part.type === "image") {
          if (allowImages) {
            imageBlocks.push({ type: "image", data: part.data, mimeType: part.mimeType });
          } else {
            omittedImages += 1;
          }
        }
      }
    }
  }
  if (omittedImages > 0) {
    liveParts.push(omittedImagesPrompt(omittedImages));
  }

  const customSys = options?.customSystemContext?.trim();
  const liveText = liveParts.join("\n\n");
  const maxBytes = options?.maxBytes ?? MAX_AGY_PROMPT_BYTES;
  const isSlashPrompt = liveText.trimStart().startsWith("/");

  if (isSlashPrompt) {
    const liveBytes = Buffer.byteLength(liveText, "utf8");
    if (liveBytes >= maxBytes) {
      return [{ type: "text", text: truncateUtf8Head(liveText, maxBytes) }, ...imageBlocks];
    }
  }

  const fixedSegments = [customSys, liveText].filter((s): s is string => Boolean(s));
  const fixedBytes =
    Buffer.byteLength(fixedSegments.join("\n\n"), "utf8") + (fixedSegments.length > 0 ? 4 : 0);
  const wrapperBytes = Buffer.byteLength(
    `${SEEDED_HISTORY_HEADER}${SEEDED_HISTORY_FOOTER}`,
    "utf8",
  );
  const historyBudget = maxBytes - fixedBytes - wrapperBytes;
  if (historyBudget <= 64) {
    return fallbackLive();
  }

  let remainingBudget = historyBudget;
  let boundedCheckpoint: string | undefined;
  if (checkpointSummary) {
    const checkpointLimit =
      historyItems.length > 0 ? Math.max(64, Math.floor(historyBudget * 0.6)) : historyBudget;
    boundedCheckpoint = truncateUtf8Tail(checkpointSummary, checkpointLimit);
    remainingBudget -=
      Buffer.byteLength(boundedCheckpoint, "utf8") + (historyItems.length > 0 ? 2 : 0);
  }

  const keptHistory: string[] = [];
  let omittedHistory = false;
  const omissionMarker = "[...earlier turns omitted...]";
  const omissionBytes = Buffer.byteLength(omissionMarker, "utf8") + 2;

  for (let i = historyItems.length - 1; i >= 0; i -= 1) {
    const item = historyItems[i];
    const itemBytes = Buffer.byteLength(item, "utf8") + (keptHistory.length > 0 ? 2 : 0);
    const reserveOmission = i > 0 ? omissionBytes : 0;
    if (itemBytes + reserveOmission <= remainingBudget) {
      keptHistory.unshift(item);
      remainingBudget -= itemBytes;
    } else {
      omittedHistory = true;
      break;
    }
  }

  if (
    omittedHistory &&
    keptHistory.length === 0 &&
    !boundedCheckpoint &&
    remainingBudget > omissionBytes + 32
  ) {
    const lastItem = historyItems.at(-1);
    if (lastItem) {
      keptHistory.push(truncateUtf8Tail(lastItem, remainingBudget - omissionBytes));
    }
  }

  const historySections: string[] = [];
  if (boundedCheckpoint) historySections.push(boundedCheckpoint);
  if (omittedHistory) historySections.push(omissionMarker);
  historySections.push(...keptHistory);

  if (historySections.length === 0) {
    return fallbackLive();
  }

  const historyBlock = `${SEEDED_HISTORY_HEADER}${historySections.join("\n\n")}${SEEDED_HISTORY_FOOTER}`;
  if (isSlashPrompt) {
    const contextBudget = maxBytes - Buffer.byteLength(liveText, "utf8");
    const contextText = truncateUtf8Tail(
      [customSys, historyBlock].filter(Boolean).join("\n\n"),
      contextBudget,
    );
    const result: AcpContentBlock[] = [{ type: "text", text: liveText }];
    if (contextText) {
      result.push({ type: "text", text: contextText });
    }
    result.push(...imageBlocks);
    return result;
  }

  const assembled = [customSys, historyBlock, liveText].filter(Boolean).join("\n\n");
  const boundedText = truncateUtf8Tail(assembled, maxBytes);

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

export interface BridgedToolImageBatch {
  callId: string;
  name: string;
  label?: string;
  images: Array<{ type: "image"; data: string; mimeType: string }>;
}

/**
 * Partition a bridged tool result into text-only MCP content and an optional
 * image batch for inline ACP `session/prompt` continuation.
 */
export function splitToolResultImages(
  item: CollectedToolResult,
  options?: { allowImages?: boolean; label?: string },
): { mcpContent: (TextContent | ImageContent)[]; imageBatch?: BridgedToolImageBatch } {
  const textBlocks: TextContent[] = [];
  const images: Array<{ type: "image"; data: string; mimeType: string }> = [];

  for (const part of item.content) {
    if (part.type === "text") {
      textBlocks.push(part);
    } else if (part.type === "image") {
      images.push({ type: "image", data: part.data, mimeType: part.mimeType });
    }
  }

  if (images.length === 0) {
    return { mcpContent: item.content };
  }

  if (options?.allowImages === false) {
    return {
      mcpContent: [...textBlocks, { type: "text", text: omittedImagesPrompt(images.length) }],
    };
  }

  const mimeSummary = [...new Set(images.map((img) => img.mimeType))].join(", ");
  return {
    mcpContent: [
      ...textBlocks,
      {
        type: "text",
        text: `[${images.length} image attachment(s) (${mimeSummary}) delivered inline in the immediate continuation message — inspect the attached image(s) directly without calling ${item.name} again.]`,
      },
    ],
    imageBatch: {
      callId: item.callId,
      name: item.name,
      ...(options?.label ? { label: options.label } : {}),
      images,
    },
  };
}

/**
 * Build an ACP `session/prompt` payload carrying bridged tool image results
 * in execution order, optionally merged with any trailing user steering prompt.
 */
export function buildToolImageContinuationPrompt(
  batches: readonly BridgedToolImageBatch[],
  trailingPrompt?: readonly AcpContentBlock[],
  options?: { maxBytes?: number },
): AcpContentBlock[] {
  if (batches.length === 0) {
    return trailingPrompt ? [...trailingPrompt] : [];
  }

  const lines: string[] = ["[Bridged tool image result(s) attached in order:"];
  const toolImageBlocks: AcpContentBlock[] = [];
  for (const batch of batches) {
    const target = batch.label ? ` (${JSON.stringify(batch.label)})` : "";
    for (const img of batch.images) {
      toolImageBlocks.push(img);
      lines.push(`${toolImageBlocks.length}. ${batch.name}${target} — ${img.mimeType}`);
    }
  }
  lines.push(
    "Continue the current turn using these attached image(s); do not re-call the tool for the same image(s).]",
  );
  const manifestText = lines.join("\n");

  const textParts: string[] = [manifestText];
  const trailingImages: AcpContentBlock[] = [];
  for (const block of trailingPrompt ?? []) {
    if (block.type === "text") {
      if (block.text.trim()) textParts.push(block.text);
    } else if (block.type === "image") {
      trailingImages.push(block);
    }
  }

  const maxBytes = options?.maxBytes ?? MAX_AGY_PROMPT_BYTES;
  const boundedText = truncateUtf8Tail(textParts.join("\n\n"), maxBytes);
  return [{ type: "text", text: boundedText }, ...toolImageBlocks, ...trailingImages];
}

function hasMeaningfulUserContent(
  msg: Extract<TranscriptContext["messages"][number], { role: "user" }>,
): boolean {
  if (typeof msg.content === "string") {
    return msg.content.trim().length > 0;
  }
  for (const part of msg.content) {
    if (part.type === "text" && part.text.trim().length > 0) return true;
    if (part.type === "image") return true;
  }
  return false;
}

/**
 * True only when the trailing non-system messages since the last assistant
 * turn are `toolResult`s with no subsequent meaningful `user` message (such as
 * a mid-turn steering prompt).
 */
export function isContinuationAfterToolUse(messages: TranscriptContext["messages"]): boolean {
  let sawToolResult = false;
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    const msg = messages[i];
    if (!msg || msg.role === "system") continue;
    if (msg.role === "user") {
      if (hasMeaningfulUserContent(msg)) return false;
      continue;
    }
    if (msg.role === "toolResult") {
      sawToolResult = true;
      continue;
    }
    if (msg.role === "assistant") return sawToolResult;
  }
  return sawToolResult;
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
