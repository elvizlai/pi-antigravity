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
  | { wave: "bridge_wave"; contextTokens?: number }
  | { wave: "settled"; result: AgyTurnResult }
  | { wave: "aborted" };

type QueuedTurnItem =
  | { kind: "activity"; activity: AgyTurnActivity }
  | { kind: "settled"; result: AgyTurnResult }
  | { kind: "error"; error: Error };

/**
 * Single-turn async coordinator bridging push-based ACP updates + bridged MCP
 * requests to wave-based Pi `AssistantMessageEventStream` consumers.
 */
export class AgyTurnController {
  #queue: QueuedTurnItem[] = [];
  #notify?: () => void;
  #settled = false;
  #error?: Error;
  #settledResult?: AgyTurnResult;
  #latestContextTokens: number | undefined;

  get isSettled(): boolean {
    return this.#settled;
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
   * - one or more `bridge_call`s suspend the turn for Pi tool execution,
   * - the underlying ACP turn settles (completed / aborted / error), or
   * - the per-wave `signal` aborts.
   */
  async drainWave(
    onActivity: (activity: AgyTurnActivity) => void | Promise<void>,
    options?: { signal?: AbortSignal },
  ): Promise<AgyTurnWaveResult> {
    const signal = options?.signal;
    const bridgeWave = (): AgyTurnWaveResult => ({
      wave: "bridge_wave",
      ...(this.#latestContextTokens !== undefined
        ? { contextTokens: this.#latestContextTokens }
        : {}),
    });

    while (true) {
      if (signal?.aborted) {
        return { wave: "aborted" };
      }

      if (this.#queue.length > 0) {
        let sawBridgeCall = false;
        while (this.#queue.length > 0) {
          const next = this.#queue[0];
          if (
            sawBridgeCall &&
            !(next.kind === "activity" && next.activity.type === "bridge_call")
          ) {
            return bridgeWave();
          }
          this.#queue.shift();
          if (next.kind === "error") {
            throw next.error;
          }
          if (next.kind === "settled") {
            if (sawBridgeCall) return bridgeWave();
            return { wave: "settled", result: this.#enrichSettled(next.result) };
          }
          await onActivity(next.activity);
          if (next.activity.type === "bridge_call") {
            sawBridgeCall = true;
          }
        }
        if (sawBridgeCall) {
          return bridgeWave();
        }
        continue;
      }

      if (this.#error) {
        throw this.#error;
      }
      if (this.#settledResult) {
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
