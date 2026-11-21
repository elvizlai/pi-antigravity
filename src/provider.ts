import { randomUUID } from "node:crypto";
import {
  type Api,
  type AssistantMessage,
  type AssistantMessageEventStream,
  calculateCost,
  createAssistantMessageEventStream,
  getCurrentSystemPrompt,
  type ImageContent,
  type JsonObject,
  type Message,
  type Model,
  type SimpleStreamOptions,
  type TextContent,
  type ThinkingContent,
  type ToolCall,
  type TranscriptContext,
  type Usage,
} from "@earendil-works/pi-ai";
import type { AcpContentBlock } from "../lib/acp-client.ts";
import {
  type AgyActivity,
  type AgyContextUsage,
  type AgyUsage,
  isBridgedMcpTool,
} from "../lib/acp-updates.ts";
import {
  type AgyPiBridge,
  type BridgeCall,
  resolveBridgeResultsFromContext,
} from "../lib/bridge.ts";
import type { BridgeLifecycleManager } from "../lib/bridge-lifecycle.ts";
import type { AgyAcpConfigOption } from "../lib/models.ts";
import type { AgyExecutionMode } from "../lib/profile.ts";
import {
  AGY_MAX_INPUT_BYTES,
  assembleBoundedAgyPrompt,
  boundRestoredPiTranscript,
  buildAgyRelayedInstructions,
  MAX_RESTORED_HISTORY_BYTES,
  omittedImagesPrompt,
  type PiDocsPaths,
  stallContinuationPrompt,
  stripPiToolInventory,
  WRAPPER_TOOL_NAME,
} from "../lib/prompt.ts";
import type { AgyReplayStore } from "../lib/replay.ts";
import type { AgyToolMode } from "../lib/tool-mode.ts";
import type { AgyAcpRuntime, AgyRuntimeTurnResult } from "./runtime.ts";

export interface AntigravityProviderDeps {
  runtime: AgyAcpRuntime;
  replayStore: AgyReplayStore;
  bridge?: AgyPiBridge;
  bridgeManager?: BridgeLifecycleManager;
  piDocs?: PiDocsPaths;
  getToolMode: () => AgyToolMode;
  getExecutionMode: () => AgyExecutionMode;
  getCwd?: () => string;
  getBridgeRevision: () => string;
  getConversationId: () => string | undefined;
  consumeMissingConversationNotice?: () => string | undefined;
  ensurePiToolsAgent?: (modelId: string) => Promise<string | undefined>;
  prepareNativeBridge?: () => Promise<void>;
  getExtraDirs?: (toolMode: AgyToolMode) => readonly string[];
  isSandbox?: () => boolean;
  onConversationResolved?: (info: {
    sessionId: string;
    resumed: boolean;
    configOptions?: readonly AgyAcpConfigOption[];
  }) => void;
  onMissingConversation?: (missingId: string) => void;
  onQuotaExhausted?: () => void;
  onContextUsage?: (usage: AgyContextUsage) => void;
  onUsage?: (usage: AgyUsage) => void;
  onActivity?: (activity: AgyActivity) => void;
}

function emptyUsage(): Usage {
  return {
    input: 0,
    output: 0,
    cacheRead: 0,
    cacheWrite: 0,
    totalTokens: 0,
    cost: {
      input: 0,
      output: 0,
      cacheRead: 0,
      cacheWrite: 0,
      total: 0,
    },
  };
}

function extractTextFromBlocks(content: string | readonly (TextContent | ImageContent)[]): {
  text: string;
  images: ImageContent[];
} {
  if (typeof content === "string") {
    return { text: content, images: [] };
  }
  const textParts: string[] = [];
  const images: ImageContent[] = [];
  for (const part of content) {
    if (part.type === "text" && part.text) {
      textParts.push(part.text);
    } else if (part.type === "image" && part.data && part.mimeType) {
      images.push(part);
    }
  }
  return { text: textParts.join("\n"), images };
}

export function extractTrailingUserPrompt(messages: readonly Message[]): {
  userIndex: number;
  text: string;
  images: ImageContent[];
} {
  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i];
    if (msg.role === "user") {
      const { text, images } = extractTextFromBlocks(msg.content);
      return {
        userIndex: i,
        text: text || "(see attached image)",
        images,
      };
    }
  }
  return {
    userIndex: -1,
    text: stallContinuationPrompt(),
    images: [],
  };
}

export function buildRestoredPiTranscript(
  messages: readonly Message[],
  upToIndex: number,
  maxBytes = MAX_RESTORED_HISTORY_BYTES,
): string | undefined {
  if (upToIndex <= 0) return undefined;

  let omittedImages = 0;
  const entries: string[] = [];

  for (let i = 0; i < upToIndex; i++) {
    const msg = messages[i];
    if (msg.role === "system") continue;
    if (msg.role === "user") {
      const { text, images } = extractTextFromBlocks(msg.content);
      omittedImages += images.length;
      if (text.trim()) entries.push(`User:\n${text.trim()}`);
      continue;
    }
    if (msg.role === "assistant") {
      const parts: string[] = [];
      for (const block of msg.content) {
        if (block.type === "text" && block.text.trim()) {
          parts.push(block.text.trim());
        } else if (block.type === "toolCall") {
          parts.push(`[Tool call: ${block.name}]`);
        }
      }
      if (parts.length > 0) entries.push(`Assistant:\n${parts.join("\n")}`);
      continue;
    }
    if (msg.role === "toolResult") {
      const { text, images } = extractTextFromBlocks(msg.content);
      omittedImages += images.length;
      const clipped = text.trim().slice(0, 1_000);
      if (clipped) {
        entries.push(`Tool result (${msg.toolName}${msg.isError ? ", error" : ""}):\n${clipped}`);
      }
    }
  }

  if (entries.length === 0 && omittedImages === 0) return undefined;
  const joined = entries.join("\n\n");
  const prefixed = [omittedImages > 0 ? omittedImagesPrompt(omittedImages) : undefined, joined]
    .filter((s): s is string => Boolean(s))
    .join("\n\n");

  return boundRestoredPiTranscript(prefixed, maxBytes);
}

export function buildAgySystemPromptAppend(options: {
  piSystemPrompt: string;
  piDocs?: PiDocsPaths;
  missingConversationNotice?: string;
}): string | undefined {
  const sections: string[] = [];
  if (options.missingConversationNotice) {
    sections.push(options.missingConversationNotice);
  }
  const relayedDocs = options.piDocs ? buildAgyRelayedInstructions(options.piDocs) : "";
  if (relayedDocs) {
    sections.push(relayedDocs);
  }
  const cleanedPiSystem = stripPiToolInventory(options.piSystemPrompt);
  if (cleanedPiSystem) {
    sections.push(
      [
        "## Additional Pi system instructions",
        "Use only the actual agy or Pi bridge tool schemas available to you, while respecting applicable project and user guidance.",
        cleanedPiSystem,
      ].join("\n\n"),
    );
  }
  const combined = sections.filter(Boolean).join("\n\n").trim();
  return combined.length > 0 ? combined : undefined;
}

function applyPromptUsage(
  output: AssistantMessage,
  usage: AgyUsage | undefined,
  contextUsage: AgyContextUsage | undefined,
  model: Model<Api>,
): void {
  const rawInput = usage?.input_tokens ?? 0;
  const out = usage?.output_tokens ?? 0;
  const cacheRead = usage?.cache_read_tokens ?? 0;
  const cacheWrite = usage?.cache_creation_tokens ?? 0;
  const reasoning = usage?.reasoning_tokens;
  const rawTotal = usage?.total_tokens ?? rawInput + out + cacheRead + cacheWrite;

  // When ACP returns `usage_update` context fill but omits prompt token usage, surface fill.
  const input =
    rawTotal === 0 && contextUsage && contextUsage.usedTokens > 0
      ? contextUsage.usedTokens
      : rawInput;
  const totalTokens = rawTotal > 0 ? rawTotal : input + out + cacheRead + cacheWrite;

  const baseUsage: Usage = {
    input,
    output: out,
    cacheRead,
    cacheWrite,
    ...(reasoning !== undefined ? { reasoning } : {}),
    totalTokens,
    cost: {
      input: 0,
      output: 0,
      cacheRead: 0,
      cacheWrite: 0,
      total: 0,
    },
  };
  baseUsage.cost = calculateCost(model, baseUsage);
  output.usage = baseUsage;
}

export function createAntigravityStreamSimple(deps: AntigravityProviderDeps) {
  return function streamAntigravity(
    model: Model<Api>,
    context: TranscriptContext,
    options?: SimpleStreamOptions,
  ): AssistantMessageEventStream {
    const stream = createAssistantMessageEventStream();

    void (async () => {
      const output: AssistantMessage = {
        role: "assistant",
        content: [],
        api: model.api,
        provider: model.provider,
        model: model.id,
        usage: emptyUsage(),
        stopReason: "stop",
        timestamp: Date.now(),
      };

      let openThinkingIndex: number | undefined;
      let openTextIndex: number | undefined;
      let hasReplayToolCalls = false;
      let started = false;

      const ensureStarted = () => {
        if (started) return;
        started = true;
        stream.push({ type: "start", partial: output });
      };

      const closeOpenThinking = () => {
        if (openThinkingIndex === undefined) return;
        const block = output.content[openThinkingIndex] as ThinkingContent;
        stream.push({
          type: "thinking_end",
          contentIndex: openThinkingIndex,
          content: block.thinking,
          partial: output,
        });
        openThinkingIndex = undefined;
      };

      const closeOpenText = () => {
        if (openTextIndex === undefined) return;
        const block = output.content[openTextIndex] as TextContent;
        stream.push({
          type: "text_end",
          contentIndex: openTextIndex,
          content: block.text,
          partial: output,
        });
        openTextIndex = undefined;
      };

      const emitToolCall = (toolCall: ToolCall) => {
        closeOpenThinking();
        closeOpenText();
        const contentIndex = output.content.length;
        output.content.push(toolCall);
        stream.push({ type: "toolcall_start", contentIndex, partial: output });
        stream.push({
          type: "toolcall_delta",
          contentIndex,
          delta: JSON.stringify(toolCall.arguments),
          partial: output,
        });
        stream.push({
          type: "toolcall_end",
          contentIndex,
          toolCall,
          partial: output,
        });
      };

      const onActivity = (activity: AgyActivity) => {
        deps.onActivity?.(activity);
        switch (activity.type) {
          case "thinking_delta": {
            closeOpenText();
            if (openThinkingIndex === undefined) {
              openThinkingIndex = output.content.length;
              output.content.push({ type: "thinking", thinking: "" });
              stream.push({
                type: "thinking_start",
                contentIndex: openThinkingIndex,
                partial: output,
              });
            }
            const block = output.content[openThinkingIndex] as ThinkingContent;
            block.thinking += activity.text;
            stream.push({
              type: "thinking_delta",
              contentIndex: openThinkingIndex,
              delta: activity.text,
              partial: output,
            });
            break;
          }
          case "text_delta": {
            closeOpenThinking();
            if (openTextIndex === undefined) {
              openTextIndex = output.content.length;
              output.content.push({ type: "text", text: "" });
              stream.push({
                type: "text_start",
                contentIndex: openTextIndex,
                partial: output,
              });
            }
            const block = output.content[openTextIndex] as TextContent;
            block.text += activity.text;
            stream.push({
              type: "text_delta",
              contentIndex: openTextIndex,
              delta: activity.text,
              partial: output,
            });
            break;
          }
          case "context_usage": {
            deps.onContextUsage?.({
              usedTokens: activity.usedTokens,
              sizeTokens: activity.sizeTokens,
            });
            break;
          }
          case "tool_done":
          case "tool_error": {
            // Pi already executes & renders bridged MCP calls directly from `bridgeCalls`.
            if (isBridgedMcpTool(activity, deps.bridge?.serverName)) break;
            const callId = `agy-${activity.callId || randomUUID()}`;
            deps.replayStore.record(callId, {
              agyTool: activity.name,
              output: activity.type === "tool_done" ? activity.output : undefined,
              error: activity.type === "tool_error" ? activity.message : undefined,
              durationSeconds: activity.durationSeconds,
            });
            emitToolCall({
              type: "toolCall",
              id: callId,
              name: WRAPPER_TOOL_NAME,
              arguments: {
                tool: activity.name,
                input: activity.args as JsonObject,
              },
            });
            hasReplayToolCalls = true;
            break;
          }
          case "tool_start":
          case "tool_update":
            break;
        }
      };

      try {
        ensureStarted();

        if (options?.signal?.aborted) {
          const abort = new Error("Aborted before starting Antigravity request.");
          abort.name = "AbortError";
          throw abort;
        }

        const resolvedBridgeCount = deps.bridge
          ? resolveBridgeResultsFromContext(deps.bridge, context.messages)
          : 0;
        const trailingIsToolResult =
          context.messages.length > 0 &&
          context.messages[context.messages.length - 1]?.role === "toolResult";

        let turnResult: AgyRuntimeTurnResult;
        if (deps.runtime.hasParkedTurn && (resolvedBridgeCount > 0 || trailingIsToolResult)) {
          turnResult = await deps.runtime.continueParkedTurn({
            signal: options?.signal,
            onActivity,
          });
        } else {
          const toolMode = deps.getToolMode();
          if (toolMode === "pi" && deps.bridgeManager) {
            await deps.bridgeManager.ensureRegistered();
          } else if (toolMode === "native") {
            await deps.prepareNativeBridge?.();
          }
          const agentName =
            toolMode === "pi" && deps.ensurePiToolsAgent
              ? await deps.ensurePiToolsAgent(model.id)
              : undefined;

          const extraDirs = deps.getExtraDirs?.(toolMode) ?? [];
          const { userIndex, text: userText, images } = extractTrailingUserPrompt(context.messages);
          const piSystemPrompt = getCurrentSystemPrompt(context.messages);
          const missingNotice = deps.consumeMissingConversationNotice?.();
          const systemPromptAppend = buildAgySystemPromptAppend({
            piSystemPrompt,
            piDocs: deps.piDocs,
            missingConversationNotice: missingNotice,
          });
          const effort = options?.reasoning;

          turnResult = await deps.runtime.runTurn({
            cwd: deps.getCwd?.() ?? process.cwd(),
            desiredConversationId: deps.getConversationId(),
            systemPromptAppend,
            modelId: model.id,
            effort,
            executionMode: deps.getExecutionMode(),
            agentName,
            extraDirs,
            sandbox: deps.isSandbox?.() ?? false,
            bridgeRevision: deps.getBridgeRevision(),
            signal: options?.signal,
            onConversationResolved: deps.onConversationResolved,
            onMissingConversation: deps.onMissingConversation,
            onQuotaExhausted: deps.onQuotaExhausted,
            onActivity,
            buildPrompt: ({ usedHistoryBootstrap }) => {
              const historyBootstrap = usedHistoryBootstrap
                ? buildRestoredPiTranscript(context.messages, userIndex)
                : undefined;
              const bounded = assembleBoundedAgyPrompt(
                {
                  historyBootstrap,
                  prompt: userText,
                },
                AGY_MAX_INPUT_BYTES,
              );
              const blocks: AcpContentBlock[] = [
                { type: "text", text: bounded },
                ...images.map((img) => ({
                  type: "image" as const,
                  data: img.data,
                  mimeType: img.mimeType,
                })),
              ];
              return { blocks };
            },
          });
        }

        closeOpenThinking();
        closeOpenText();

        if (turnResult.outcome.kind === "parked_tool_calls") {
          for (const call of turnResult.outcome.bridgeCalls as BridgeCall[]) {
            emitToolCall({
              type: "toolCall",
              id: call.id,
              name: call.tool,
              arguments: call.args,
            });
          }
          output.stopReason = "toolUse";
          stream.push({ type: "done", reason: "toolUse", message: output });
          stream.end();
          return;
        }

        applyPromptUsage(output, turnResult.outcome.usage, turnResult.lastContextUsage, model);
        if (turnResult.outcome.usage) {
          deps.onUsage?.(turnResult.outcome.usage);
        }

        output.endTurn = true;
        if (hasReplayToolCalls) {
          output.stopReason = "toolUse";
          stream.push({ type: "done", reason: "toolUse", message: output });
        } else if (turnResult.outcome.stopReason === "max_tokens") {
          output.stopReason = "length";
          stream.push({ type: "done", reason: "length", message: output });
        } else {
          output.stopReason = "stop";
          stream.push({ type: "done", reason: "stop", message: output });
        }
        stream.end();
      } catch (error) {
        ensureStarted();
        closeOpenThinking();
        closeOpenText();
        const err = error instanceof Error ? error : new Error(String(error));
        const aborted = err.name === "AbortError" || options?.signal?.aborted === true;
        output.stopReason = aborted ? "aborted" : "error";
        output.errorMessage = err.message;
        stream.push({
          type: "error",
          reason: aborted ? "aborted" : "error",
          error: output,
        });
        stream.end();
      }
    })();

    return stream;
  };
}
