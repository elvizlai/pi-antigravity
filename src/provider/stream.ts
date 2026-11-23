import { randomUUID } from "node:crypto";
import {
  type Api,
  type AssistantMessage,
  type AssistantMessageEventStream,
  createAssistantMessageEventStream,
  type JsonObject,
  type Model,
  type SimpleStreamOptions,
  type TextContent,
  type ThinkingContent,
  type ToolCall,
  type TranscriptContext,
  type Usage,
} from "@earendil-works/pi-ai";
import type { AcpContentBlock } from "../acp/client.ts";
import type { AgyUsage } from "../acp/events.ts";
import type { AgyExecutionMode } from "../acp/models.ts";
import { isAbortError } from "../acp/shared.ts";
import { WRAPPER_TOOL_NAME } from "../bridge/server.ts";
import { extractSystemPromptAppend, extractTrailingUserPrompt } from "./prompt.ts";
import type { AgyRuntime } from "./runtime.ts";
import type { SessionEntryLike } from "./state.ts";
import type { AgyReplayStore } from "./turn.ts";

export interface AgyStreamPayload {
  modelId: string;
  effort?: string;
  executionMode?: AgyExecutionMode;
  prompt: AcpContentBlock[];
  systemPromptAppend?: string;
}

export interface AgyStreamDeps {
  getRuntime: () => AgyRuntime;
  getSessionId: () => string;
  getBranchEntries: () => readonly SessionEntryLike[];
  getExecutionMode?: () => AgyExecutionMode;
  replayStore: AgyReplayStore;
  now?: () => number;
}

function zeroUsage(): Usage {
  return {
    input: 0,
    output: 0,
    cacheRead: 0,
    cacheWrite: 0,
    totalTokens: 0,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
  };
}

export function toPiUsage(
  usage: AgyUsage | undefined,
  model: Model<Api>,
  contextTokens?: number,
): Usage {
  const hasContextTokens =
    typeof contextTokens === "number" && Number.isFinite(contextTokens) && contextTokens > 0;
  if (!usage && !hasContextTokens) return zeroUsage();

  const rawInput = Math.max(0, usage?.input_tokens ?? 0);
  const rawOutput = Math.max(0, usage?.output_tokens ?? 0);
  const rawCacheRead = Math.max(0, usage?.cache_read_tokens ?? 0);
  const rawCacheWrite = Math.max(0, usage?.cache_creation_tokens ?? 0);
  const rawReasoning =
    usage?.thinking_tokens !== undefined
      ? Math.max(0, usage.thinking_tokens)
      : usage?.reasoning_tokens !== undefined
        ? Math.max(0, usage.reasoning_tokens)
        : undefined;
  const rawPrompt = rawInput + rawCacheRead + rawCacheWrite;
  const rawComponentSum = rawPrompt + rawOutput;
  const rawTotal = Math.max(usage?.total_tokens ?? 0, rawComponentSum);

  let input = rawInput;
  let output = rawOutput;
  let cacheRead = rawCacheRead;
  let cacheWrite = rawCacheWrite;
  let reasoning = rawReasoning;
  const totalTokens = hasContextTokens ? Math.round(contextTokens) : rawTotal;

  // ACP `PromptResponse.usage` accumulates across every internal LLM call in the turn,
  // whereas Pi expects `AssistantMessage.usage` to describe a single step (`isContextOverflow`
  // checks `input + cacheRead > contextWindow` when `stopReason === "stop"`). When the final
  // step's `usage_update.used` (`totalTokens`) is smaller than `rawTotal`, scale the cumulative
  // breakdown proportionally into `totalTokens`.
  if (hasContextTokens && rawTotal > totalTokens) {
    const scale = totalTokens / rawTotal;
    output = rawOutput > 0 ? Math.min(totalTokens, Math.max(1, Math.round(rawOutput * scale))) : 0;
    const promptBudget = Math.min(
      Math.max(0, totalTokens - output),
      rawPrompt > 0 ? Math.round(rawPrompt * scale) : 0,
    );
    if (rawPrompt > 0 && promptBudget > 0) {
      const promptScale = promptBudget / rawPrompt;
      cacheRead = Math.min(promptBudget, Math.round(rawCacheRead * promptScale));
      cacheWrite = Math.min(promptBudget - cacheRead, Math.round(rawCacheWrite * promptScale));
      input = Math.max(0, promptBudget - cacheRead - cacheWrite);
    } else {
      input = 0;
      cacheRead = 0;
      cacheWrite = 0;
    }
    if (rawReasoning !== undefined) {
      const scaledReasoning = rawReasoning > 0 ? Math.max(1, Math.round(rawReasoning * scale)) : 0;
      reasoning = rawReasoning <= rawOutput ? Math.min(output, scaledReasoning) : scaledReasoning;
    }
  }

  const costInput = (input / 1_000_000) * (model.cost?.input ?? 0);
  const costOutput = (output / 1_000_000) * (model.cost?.output ?? 0);
  const costCacheRead = (cacheRead / 1_000_000) * (model.cost?.cacheRead ?? 0);
  const costCacheWrite = (cacheWrite / 1_000_000) * (model.cost?.cacheWrite ?? 0);
  return {
    input,
    output,
    cacheRead,
    cacheWrite,
    ...(reasoning !== undefined ? { reasoning } : {}),
    totalTokens,
    cost: {
      input: costInput,
      output: costOutput,
      cacheRead: costCacheRead,
      cacheWrite: costCacheWrite,
      total: costInput + costOutput + costCacheRead + costCacheWrite,
    },
  };
}

/**
 * Create the `streamSimple` provider handler for Antigravity ACP.
 *
 * Emits strictly append-only Pi content block indices, invokes `options.onPayload`
 * and `options.onProviderStreamEvent`, forwards `options.env` and `options.timeoutMs`,
 * and records native `agy` tools in `AgyReplayStore` under the `antigravity` wrapper.
 */
export function createAntigravityStreamSimple(
  deps: AgyStreamDeps,
): (
  model: Model<Api>,
  context: TranscriptContext,
  options?: SimpleStreamOptions,
) => AssistantMessageEventStream {
  return (model, context, options) => {
    const stream = createAssistantMessageEventStream();
    const partial: AssistantMessage = {
      role: "assistant",
      content: [],
      api: model.api,
      provider: model.provider,
      model: model.id,
      usage: zeroUsage(),
      stopReason: "stop",
      timestamp: deps.now?.() ?? Date.now(),
    };

    void (async () => {
      let activeBlock:
        | { kind: "thinking"; index: number }
        | { kind: "text"; index: number }
        | undefined;
      let detachAbort: (() => void) | undefined;

      const closeActiveBlock = () => {
        if (!activeBlock) return;
        const { kind, index } = activeBlock;
        activeBlock = undefined;
        if (kind === "thinking") {
          const content = (partial.content[index] as ThinkingContent).thinking;
          stream.push({ type: "thinking_end", contentIndex: index, content, partial });
        } else {
          const content = (partial.content[index] as TextContent).text;
          stream.push({ type: "text_end", contentIndex: index, content, partial });
        }
      };

      const appendToolCall = (callId: string, name: string, args: JsonObject) => {
        closeActiveBlock();
        const contentIndex = partial.content.length;
        const toolCall: ToolCall = { type: "toolCall", id: callId, name, arguments: {} };
        partial.content.push(toolCall);
        stream.push({ type: "toolcall_start", contentIndex, partial });
        const serialized = JSON.stringify(args);
        if (serialized) {
          stream.push({ type: "toolcall_delta", contentIndex, delta: serialized, partial });
        }
        toolCall.arguments = args;
        stream.push({ type: "toolcall_end", contentIndex, toolCall, partial });
      };

      try {
        stream.push({ type: "start", partial });

        const allowImages = model.input.includes("image");
        const initialPayload: AgyStreamPayload = {
          modelId: model.id,
          effort: options?.reasoning,
          executionMode: deps.getExecutionMode?.(),
          prompt: extractTrailingUserPrompt(context.messages, { allowImages }),
          systemPromptAppend: extractSystemPromptAppend(context.messages),
        };

        let payload = initialPayload;
        if (options?.onPayload) {
          const replaced = await options.onPayload(initialPayload, model);
          if (replaced && typeof replaced === "object") {
            const candidate = replaced as Partial<AgyStreamPayload>;
            payload = {
              modelId: typeof candidate.modelId === "string" ? candidate.modelId : payload.modelId,
              effort: candidate.effort ?? payload.effort,
              executionMode: candidate.executionMode ?? payload.executionMode,
              prompt: Array.isArray(candidate.prompt) ? candidate.prompt : payload.prompt,
              systemPromptAppend:
                candidate.systemPromptAppend !== undefined
                  ? candidate.systemPromptAppend
                  : payload.systemPromptAppend,
            };
          }
        }

        const turn = await deps.getRuntime().startOrContinueTurn({
          modelId: payload.modelId,
          effort: payload.effort,
          executionMode: payload.executionMode,
          prompt: payload.prompt,
          allowImages,
          systemPromptAppend: payload.systemPromptAppend,
          messages: context.messages,
          sessionId: deps.getSessionId(),
          branchEntries: deps.getBranchEntries(),
          env: options?.env,
          timeoutMs: options?.timeoutMs,
          signal: options?.signal,
          replayStore: deps.replayStore,
        });

        if (options?.signal) {
          const onAbort = () => turn.signalTurnAbort();
          if (options.signal.aborted) {
            onAbort();
          } else {
            options.signal.addEventListener("abort", onAbort, { once: true });
            detachAbort = () => options.signal?.removeEventListener("abort", onAbort);
          }
        }

        let hasAnyToolCall = false;

        const waveResult = await turn.controller.drainWave(
          async (activity) => {
            if (options?.onProviderStreamEvent) {
              await options.onProviderStreamEvent(activity, model);
            }

            switch (activity.type) {
              case "thinking_delta": {
                if (!activity.text) break;
                if (activeBlock?.kind !== "thinking") {
                  closeActiveBlock();
                  const index = partial.content.length;
                  partial.content.push({ type: "thinking", thinking: "" });
                  activeBlock = { kind: "thinking", index };
                  stream.push({ type: "thinking_start", contentIndex: index, partial });
                }
                const block = partial.content[activeBlock.index] as ThinkingContent;
                block.thinking += activity.text;
                stream.push({
                  type: "thinking_delta",
                  contentIndex: activeBlock.index,
                  delta: activity.text,
                  partial,
                });
                break;
              }

              case "text_delta": {
                if (!activity.text) break;
                if (activeBlock?.kind !== "text") {
                  closeActiveBlock();
                  const index = partial.content.length;
                  partial.content.push({ type: "text", text: "" });
                  activeBlock = { kind: "text", index };
                  stream.push({ type: "text_start", contentIndex: index, partial });
                }
                const block = partial.content[activeBlock.index] as TextContent;
                block.text += activity.text;
                stream.push({
                  type: "text_delta",
                  contentIndex: activeBlock.index,
                  delta: activity.text,
                  partial,
                });
                break;
              }

              case "bridge_call": {
                hasAnyToolCall = true;
                appendToolCall(activity.callId, activity.name, activity.args);
                break;
              }

              case "tool_done": {
                const replayId = `agy_${randomUUID()}`;
                deps.replayStore.record(replayId, {
                  title: activity.title,
                  kind: activity.kind,
                  output: activity.output,
                  durationSeconds: activity.durationSeconds,
                });
                hasAnyToolCall = true;
                appendToolCall(replayId, WRAPPER_TOOL_NAME, {
                  tool: activity.title,
                  ...(activity.kind ? { kind: activity.kind } : {}),
                  ...(activity.locations && activity.locations.length > 0
                    ? { locations: activity.locations }
                    : {}),
                  input: activity.rawInput ?? {},
                });
                break;
              }

              case "tool_error": {
                const replayId = `agy_${randomUUID()}`;
                deps.replayStore.record(replayId, {
                  title: activity.title,
                  kind: activity.kind,
                  error: activity.message,
                  durationSeconds: activity.durationSeconds,
                });
                hasAnyToolCall = true;
                appendToolCall(replayId, WRAPPER_TOOL_NAME, {
                  tool: activity.title,
                  ...(activity.kind ? { kind: activity.kind } : {}),
                  ...(activity.locations && activity.locations.length > 0
                    ? { locations: activity.locations }
                    : {}),
                  input: activity.rawInput ?? {},
                });
                break;
              }

              default:
                break;
            }
          },
          { signal: options?.signal },
        );

        detachAbort?.();
        closeActiveBlock();

        if (waveResult.wave === "aborted") {
          partial.stopReason = "aborted";
          partial.errorMessage = "Request was aborted.";
          stream.push({ type: "error", reason: "aborted", error: partial });
          stream.end(partial);
          return;
        }

        if (waveResult.wave === "tool_wave") {
          if (waveResult.contextTokens && waveResult.contextTokens > 0) {
            partial.usage = { ...zeroUsage(), totalTokens: waveResult.contextTokens };
          }
          partial.stopReason = "toolUse";
          stream.push({ type: "done", reason: "toolUse", message: partial });
          stream.end(partial);
          return;
        }

        const settled = waveResult.result;
        if (settled.status === "aborted") {
          partial.stopReason = "aborted";
          partial.errorMessage = "Antigravity turn aborted.";
          stream.push({ type: "error", reason: "aborted", error: partial });
          stream.end(partial);
          return;
        }

        partial.usage = toPiUsage(settled.usage, model, settled.contextTokens);

        if (hasAnyToolCall) {
          partial.stopReason = "toolUse";
          stream.push({ type: "done", reason: "toolUse", message: partial });
          stream.end(partial);
          return;
        }

        const reason = settled.stopReason === "max_tokens" ? "length" : "stop";
        partial.stopReason = reason;
        stream.push({ type: "done", reason, message: partial });
        stream.end(partial);
      } catch (err) {
        detachAbort?.();
        closeActiveBlock();
        const isAbort = Boolean(options?.signal?.aborted) || isAbortError(err);
        partial.stopReason = isAbort ? "aborted" : "error";
        partial.errorMessage = err instanceof Error ? err.message : String(err);
        stream.push({
          type: "error",
          reason: isAbort ? "aborted" : "error",
          error: partial,
        });
        stream.end(partial);
      }
    })();

    return stream;
  };
}
