import { randomUUID } from "node:crypto";
import {
  type Api,
  type AssistantMessage,
  type AssistantMessageEventStream,
  createAssistantMessageEventStream,
  getCurrentTools,
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
import { type BridgeToolDef, WRAPPER_TOOL_NAME } from "../bridge/server.ts";
import {
  buildSeededSessionPrompt,
  extractSystemPromptAppend,
  extractTrailingUserPrompt,
  MAX_AGY_PROMPT_BYTES,
} from "./prompt.ts";
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
  getAllEntries?: () => readonly SessionEntryLike[];
  requireMatchingSessionId?: () => boolean;
  getExecutionMode?: () => AgyExecutionMode;
  getMainToolNames?: () => readonly string[];
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

  const rawInput = Math.max(0, usage?.inputTokens ?? 0);
  const rawOutput = Math.max(0, usage?.outputTokens ?? 0);
  const rawCacheRead = Math.max(0, usage?.cachedReadTokens ?? 0);
  const rawCacheWrite = Math.max(0, usage?.cachedWriteTokens ?? 0);
  const rawReasoning =
    usage?.thoughtTokens !== undefined ? Math.max(0, usage.thoughtTokens) : undefined;
  const rawPrompt = rawInput + rawCacheRead + rawCacheWrite;
  const rawComponentSum = rawPrompt + rawOutput;
  const rawTotal = Math.max(usage?.totalTokens ?? 0, rawComponentSum);

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
      let activeTurnHandle: { onConsumed?: () => void } | undefined;

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
        const mainSessionId = deps.getSessionId();
        const runtime = deps.getRuntime();
        const contextTools: BridgeToolDef[] = getCurrentTools(context.messages)
          .filter((tool) => tool.name !== WRAPPER_TOOL_NAME)
          .map((tool) => ({
            name: tool.name,
            description: tool.description ?? "",
            parameters: tool.parameters,
          }));
        const hasTranscriptToolDelta = context.messages.some(
          (msg) =>
            msg.role === "system" &&
            ((msg.toolsAdded?.length ?? 0) > 0 || (msg.toolsRemoved?.length ?? 0) > 0),
        );
        const isAuxContinuation = runtime.matchesAuxiliaryTurn(context.messages);
        const sessionMismatch =
          options?.sessionId !== undefined
            ? options.sessionId !== mainSessionId
            : Boolean(deps.requireMatchingSessionId?.());
        const forceNoTools = options?.toolChoice === "none";
        const forceIsolated =
          options?.disableContinuation === true ||
          options?.cacheRetention === "none" ||
          sessionMismatch;
        const mainToolNames = deps.getMainToolNames?.();
        const hasForeignTools =
          Boolean(mainToolNames && mainToolNames.length > 0 && contextTools.length > 0) &&
          (() => {
            const known = new Set(mainToolNames);
            return contextTools.some((tool) => !known.has(tool.name));
          })();
        const isAuxiliaryWithTools =
          isAuxContinuation ||
          (!forceNoTools && contextTools.length > 0 && (hasForeignTools || forceIsolated));
        const isEphemeral = !isAuxiliaryWithTools && (forceNoTools || forceIsolated);

        const systemPromptAppend = extractSystemPromptAppend(context.messages);
        const reservedSysBytes = systemPromptAppend
          ? Buffer.byteLength(systemPromptAppend, "utf8") + 2
          : 0;
        const initialPayload: AgyStreamPayload = {
          modelId: model.id,
          effort: options?.reasoning,
          executionMode:
            isEphemeral || isAuxiliaryWithTools ? "default" : deps.getExecutionMode?.(),
          prompt: isAuxContinuation
            ? []
            : isEphemeral || isAuxiliaryWithTools
              ? buildSeededSessionPrompt(context.messages, {
                  allowImages,
                  maxBytes: Math.max(1_024, MAX_AGY_PROMPT_BYTES - reservedSysBytes),
                })
              : extractTrailingUserPrompt(context.messages, { allowImages }),
          systemPromptAppend,
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
        const promptReplaced = payload.prompt !== initialPayload.prompt;

        const turn = isAuxiliaryWithTools
          ? await runtime.startOrContinueAuxiliaryTurn({
              modelId: payload.modelId,
              effort: payload.effort,
              prompt: payload.prompt,
              systemPromptAppend: payload.systemPromptAppend,
              tools: contextTools,
              messages: context.messages,
              env: options?.env,
              timeoutMs: options?.timeoutMs,
              signal: options?.signal,
            })
          : isEphemeral
            ? await runtime.startEphemeralTurn({
                modelId: payload.modelId,
                effort: payload.effort,
                prompt: payload.prompt,
                systemPromptAppend: payload.systemPromptAppend,
                env: options?.env,
                timeoutMs: options?.timeoutMs,
                signal: options?.signal,
              })
            : await runtime.startOrContinueTurn({
                modelId: payload.modelId,
                effort: payload.effort,
                executionMode: payload.executionMode,
                prompt: payload.prompt,
                promptReplaced,
                allowImages,
                systemPromptAppend: payload.systemPromptAppend,
                tools: hasTranscriptToolDelta ? contextTools : undefined,
                messages: context.messages,
                sessionId: mainSessionId,
                branchEntries: deps.getBranchEntries(),
                allEntries: deps.getAllEntries?.(),
                env: options?.env,
                timeoutMs: options?.timeoutMs,
                signal: options?.signal,
                replayStore: deps.replayStore,
              });
        activeTurnHandle = turn;

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
          activeTurnHandle?.onConsumed?.();
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
          activeTurnHandle?.onConsumed?.();
          partial.stopReason = "aborted";
          partial.errorMessage = "Antigravity turn aborted.";
          stream.push({ type: "error", reason: "aborted", error: partial });
          stream.end(partial);
          return;
        }

        partial.usage = toPiUsage(settled.usage, model, settled.contextTokens);

        if (hasAnyToolCall) {
          activeTurnHandle?.onConsumed?.();
          partial.stopReason = "toolUse";
          stream.push({ type: "done", reason: "toolUse", message: partial });
          stream.end(partial);
          return;
        }

        activeTurnHandle?.onConsumed?.();
        const reason = settled.stopReason === "max_tokens" ? "length" : "stop";
        partial.stopReason = reason;
        stream.push({ type: "done", reason, message: partial });
        stream.end(partial);
      } catch (err) {
        detachAbort?.();
        closeActiveBlock();
        activeTurnHandle?.onConsumed?.();
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
