import type { JsonObject } from "@earendil-works/pi-ai";
import type { AgyActivity, AgyUsage } from "../acp/events.ts";

export type AgyTurnActivity =
  | AgyActivity
  | { type: "bridge_call"; callId: string; name: string; args: JsonObject };

/** Recorded native agy tool execution replayed by the display-only `antigravity` tool. */
export interface RecordedAgyTool {
  title: string;
  kind?: string;
  output?: string;
  error?: string;
  durationSeconds?: number;
}

/** Per-session store mapping synthetic wrapper `toolCallId`s to recorded agy tool output. */
export class AgyReplayStore {
  #results = new Map<string, RecordedAgyTool>();

  record(toolCallId: string, result: RecordedAgyTool): void {
    this.#results.set(toolCallId, result);
  }

  take(toolCallId: string): RecordedAgyTool | undefined {
    const result = this.#results.get(toolCallId);
    this.#results.delete(toolCallId);
    return result;
  }

  clear(): void {
    this.#results.clear();
  }
}

export interface AgyTurnResult {
  status: "completed" | "aborted" | "error";
  stopReason: string;
  usage?: AgyUsage;
  contextTokens?: number;
  error?: Error;
}

export type AgyTurnWaveResult =
  | { wave: "tool_wave"; contextTokens?: number }
  | { wave: "settled"; result: AgyTurnResult }
  | { wave: "aborted" };

type QueuedTurnItem =
  | { kind: "activity"; activity: AgyTurnActivity }
  | { kind: "settled"; result: AgyTurnResult }
  | { kind: "error"; error: Error };

function isToolWaveActivity(activity: AgyTurnActivity): boolean {
  return (
    activity.type === "bridge_call" ||
    activity.type === "tool_done" ||
    activity.type === "tool_error"
  );
}

function breaksToolWave(item: QueuedTurnItem): boolean {
  if (item.kind === "settled" || item.kind === "error") return true;
  return item.activity.type === "thinking_delta" || item.activity.type === "text_delta";
}

/**
 * Single-turn async coordinator bridging push-based ACP updates + bridged MCP
 * requests to wave-based Pi `AssistantMessageEventStream` consumers.
 */
export class AgyTurnController {
  #queue: QueuedTurnItem[] = [];
  #notify?: () => void;
  #settled = false;
  #consumed = false;
  #error?: Error;
  #settledResult?: AgyTurnResult;
  #latestContextTokens: number | undefined;

  get isSettled(): boolean {
    return this.#settled;
  }

  get isConsumed(): boolean {
    return this.#consumed;
  }

  pushActivity(activity: AgyTurnActivity): void {
    if (this.#settled) return;
    if (activity.type === "context_usage" && activity.usedTokens > 0) {
      this.#latestContextTokens = activity.usedTokens;
    }
    this.#queue.push({ kind: "activity", activity });
    this.#wake();
  }

  finish(result: AgyTurnResult): void {
    if (this.#settled) return;
    this.#settled = true;
    this.#settledResult = result;
    this.#queue.push({ kind: "settled", result });
    this.#wake();
  }

  fail(error: Error): void {
    if (this.#settled) return;
    this.#settled = true;
    this.#error = error;
    this.#queue.push({ kind: "error", error });
    this.#wake();
  }

  /**
   * Consume events for the current Pi assistant wave until either:
   * - bridged MCP or native ACP tool calls suspend the turn into a `toolUse` wave,
   * - the underlying ACP turn settles (completed / aborted / error), or
   * - the per-wave `signal` aborts.
   */
  async drainWave(
    onActivity: (activity: AgyTurnActivity) => void | Promise<void>,
    options?: { signal?: AbortSignal },
  ): Promise<AgyTurnWaveResult> {
    const signal = options?.signal;
    const toolWave = (): AgyTurnWaveResult => ({
      wave: "tool_wave",
      ...(this.#latestContextTokens !== undefined
        ? { contextTokens: this.#latestContextTokens }
        : {}),
    });

    while (true) {
      if (signal?.aborted) {
        this.#consumed = true;
        return { wave: "aborted" };
      }

      if (this.#queue.length > 0) {
        let sawToolCall = false;
        while (this.#queue.length > 0) {
          const next = this.#queue[0];
          if (sawToolCall && breaksToolWave(next)) {
            return toolWave();
          }
          this.#queue.shift();
          if (next.kind === "error") {
            this.#consumed = true;
            throw next.error;
          }
          if (next.kind === "settled") {
            this.#consumed = true;
            return { wave: "settled", result: this.#enrichSettled(next.result) };
          }
          await onActivity(next.activity);
          if (isToolWaveActivity(next.activity)) {
            sawToolCall = true;
          }
        }
        if (sawToolCall) {
          return toolWave();
        }
        continue;
      }

      if (this.#error) {
        this.#consumed = true;
        throw this.#error;
      }
      if (this.#settledResult) {
        this.#consumed = true;
        return { wave: "settled", result: this.#enrichSettled(this.#settledResult) };
      }

      await this.#waitForEvent(signal);
    }
  }

  #enrichSettled(result: AgyTurnResult): AgyTurnResult {
    const contextTokens = result.contextTokens ?? this.#latestContextTokens;
    return contextTokens !== undefined ? { ...result, contextTokens } : result;
  }

  #wake(): void {
    const resolve = this.#notify;
    this.#notify = undefined;
    resolve?.();
  }

  #waitForEvent(signal?: AbortSignal): Promise<void> {
    if (signal?.aborted || this.#queue.length > 0 || this.#settled) {
      return Promise.resolve();
    }
    return new Promise<void>((resolve) => {
      const onAbort = () => {
        if (this.#notify === done) this.#notify = undefined;
        resolve();
      };
      const done = () => {
        signal?.removeEventListener("abort", onAbort);
        resolve();
      };
      this.#notify = done;
      signal?.addEventListener("abort", onAbort, { once: true });
    });
  }
}
