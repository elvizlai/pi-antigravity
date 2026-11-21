import type { AgyActivity, AgyUsage } from "../lib/acp-updates.ts";
import type { BridgeCall } from "../lib/bridge.ts";
import { agyToolStepKey } from "../lib/tool-steps.ts";

export interface AgyTurnCallbacks {
  onActivity?: (activity: AgyActivity) => void;
  onToolDelta?: (toolName: string) => void;
}

export interface AgyTurnFinalResult {
  stopReason: string;
  usage?: AgyUsage;
  activeToolsAtEnd?: readonly string[];
}

export type AgyTurnOutcome =
  | {
      kind: "complete";
      stopReason: string;
      usage?: AgyUsage;
      activeToolsAtEnd?: readonly string[];
    }
  | {
      kind: "parked_tool_calls";
      bridgeCalls: BridgeCall[];
    };

export interface ParkedTurnHandle {
  /** Wait for either the next parked bridge wave or the completion of the ACP turn. */
  waitNextWave(signal?: AbortSignal): Promise<AgyTurnOutcome>;
  /** True while the underlying `session/prompt` RPC has not yet settled. */
  readonly isActive: boolean;
}

const DEFAULT_COALESCE_MS = 25;

/**
 * Coordinates an in-flight `session/prompt` RPC against synchronous Pi tool-bridge
 * waves (`parked_tool_calls` -> Pi executes tools -> `bridge.resolveCall(...)` ->
 * ACP resumes -> `complete` or next `parked_tool_calls`).
 */
export class AgyTurnController {
  readonly #coalesceMs: number;
  #callbacks: AgyTurnCallbacks;
  #activeToolNames = new Map<string, string>();
  #bridgeQueue: BridgeCall[] = [];
  #coalesceTimer?: ReturnType<typeof setTimeout>;
  #settled = false;
  #finalResult?: AgyTurnFinalResult;
  #finalError?: Error;
  #waiter?: {
    resolve: (outcome: AgyTurnOutcome) => void;
    reject: (err: Error) => void;
    cleanupSignal?: () => void;
  };

  constructor(callbacks: AgyTurnCallbacks = {}, coalesceMs = DEFAULT_COALESCE_MS) {
    this.#callbacks = callbacks;
    this.#coalesceMs = coalesceMs;
  }

  get isActive(): boolean {
    return !this.#settled || this.#bridgeQueue.length > 0;
  }

  setCallbacks(callbacks: AgyTurnCallbacks): void {
    this.#callbacks = callbacks;
  }

  onActivity(activity: AgyActivity): void {
    if (activity.type === "tool_start") {
      const key = agyToolStepKey({ callId: activity.callId, name: activity.name });
      this.#activeToolNames.set(key, activity.name);
      this.#callbacks.onToolDelta?.(activity.name);
    } else if (activity.type === "tool_done" || activity.type === "tool_error") {
      const key = agyToolStepKey({ callId: activity.callId, name: activity.name });
      this.#activeToolNames.delete(key);
    }
    this.#callbacks.onActivity?.(activity);
  }

  onBridgeCall(call: BridgeCall): void {
    if (this.#settled) return;
    this.#bridgeQueue.push(call);
    if (!this.#coalesceTimer) {
      this.#coalesceTimer = setTimeout(() => {
        this.#coalesceTimer = undefined;
        this.#flushBridgeQueue();
      }, this.#coalesceMs);
      this.#coalesceTimer.unref?.();
    }
  }

  complete(result: AgyTurnFinalResult): void {
    if (this.#settled) return;
    this.#clearCoalesceTimer();
    this.#settled = true;
    this.#finalResult = {
      ...result,
      activeToolsAtEnd:
        result.activeToolsAtEnd ??
        (this.#activeToolNames.size > 0 ? [...this.#activeToolNames.values()] : undefined),
    };
    this.#tryWakeWaiter();
  }

  fail(error: Error): void {
    if (this.#settled) return;
    this.#clearCoalesceTimer();
    this.#settled = true;
    this.#finalError = error;
    this.#tryWakeWaiter();
  }

  waitNextWave(signal?: AbortSignal): Promise<AgyTurnOutcome> {
    if (this.#waiter) {
      return Promise.reject(new Error("AgyTurnController already has a pending wave waiter."));
    }
    if (signal?.aborted) {
      const abort = new Error("Aborted waiting for Antigravity turn wave.");
      abort.name = "AbortError";
      return Promise.reject(abort);
    }

    return new Promise<AgyTurnOutcome>((resolve, reject) => {
      const onAbort = () => {
        if (this.#waiter?.resolve !== resolve) return;
        this.#waiter = undefined;
        const abort = new Error("Aborted waiting for Antigravity turn wave.");
        abort.name = "AbortError";
        reject(abort);
      };
      signal?.addEventListener("abort", onAbort, { once: true });

      this.#waiter = {
        resolve,
        reject,
        cleanupSignal: () => signal?.removeEventListener("abort", onAbort),
      };
      this.#tryWakeWaiter();
    });
  }

  #clearCoalesceTimer(): void {
    if (this.#coalesceTimer) {
      clearTimeout(this.#coalesceTimer);
      this.#coalesceTimer = undefined;
    }
  }

  #flushBridgeQueue(): void {
    if (this.#bridgeQueue.length === 0) return;
    this.#tryWakeWaiter();
  }

  #tryWakeWaiter(): void {
    const waiter = this.#waiter;
    if (!waiter) return;

    if (this.#bridgeQueue.length > 0 && !this.#coalesceTimer && !this.#finalError) {
      const bridgeCalls = this.#bridgeQueue;
      this.#bridgeQueue = [];
      this.#waiter = undefined;
      waiter.cleanupSignal?.();
      waiter.resolve({ kind: "parked_tool_calls", bridgeCalls });
      return;
    }

    if (this.#finalError) {
      this.#waiter = undefined;
      waiter.cleanupSignal?.();
      waiter.reject(this.#finalError);
      return;
    }

    if (this.#settled && this.#finalResult) {
      this.#waiter = undefined;
      waiter.cleanupSignal?.();
      waiter.resolve({
        kind: "complete",
        stopReason: this.#finalResult.stopReason,
        usage: this.#finalResult.usage,
        activeToolsAtEnd: this.#finalResult.activeToolsAtEnd,
      });
    }
  }
}
