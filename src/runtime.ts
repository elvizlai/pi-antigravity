import { promises as fs } from "node:fs";
import path from "node:path";
import {
  type AcpContentBlock,
  AgyAcpClient,
  type AgyAcpClientOptions,
  MissingAgyConversationError,
} from "../lib/acp-client.ts";
import {
  type AgyActivity,
  type AgyContextUsage,
  type AgyToolView,
  acpUpdateToActivity,
  extractLatestTranscriptRetryError,
  normalizeAcpPromptUsage,
} from "../lib/acp-updates.ts";
import { agyBrainDir } from "../lib/agy-paths.ts";
import type { BridgeCall } from "../lib/bridge.ts";
import { type AgyBinaryInfo, checkAgyBinary } from "../lib/diagnostics.ts";
import {
  type AgyAcpConfigOption,
  extractAcpConfigOption,
  planAgyAcpSecondaryConfigWrites,
  resolveAcpModelCandidate,
} from "../lib/models.ts";
import type { AgyExecutionMode } from "../lib/profile.ts";
import { AgyQuotaTurnTracker, isAgyQuotaError } from "../lib/quota.ts";
import { trackActiveToolStep } from "../lib/tool-steps.ts";
import { AgyTurnController, type AgyTurnOutcome } from "./turn.ts";

export const DEFAULT_MAX_API_ATTEMPTS = 3;
const DEFAULT_TRANSCRIPT_POLL_MS = 500;

export class AgyQuotaError extends Error {
  readonly resetsIn?: string;

  constructor(message: string, resetsIn?: string) {
    // Include "quota exceeded" so Pi's `isRetryableAssistantError` does not trigger outer 429 retries.
    super(/quota exceeded/i.test(message) ? message : `Antigravity quota exceeded: ${message}`);
    this.name = "AgyQuotaError";
    this.resetsIn = resetsIn;
  }
}

export class AgyRetryExhaustedError extends Error {
  readonly attempts: number;
  readonly lastError: string;

  constructor(lastError: string, attempts: number) {
    // Match Pi's non-retryable pattern ("quota exceeded") so outer Pi auto-retry stops cleanly.
    super(
      `Antigravity API failed after ${attempts} attempts (quota exceeded / upstream capacity): ${lastError}`,
    );
    this.name = "AgyRetryExhaustedError";
    this.attempts = attempts;
    this.lastError = lastError;
  }
}

export interface AgyRunTurnRequest {
  cwd: string;
  desiredConversationId?: string;
  systemPromptAppend?: string;
  modelId: string;
  effort?: string;
  executionMode?: AgyExecutionMode;
  agentName?: string;
  extraDirs?: readonly string[];
  sandbox?: boolean;
  bridgeRevision?: string;
  signal?: AbortSignal;
  buildPrompt: (context: { usedHistoryBootstrap: boolean; sessionId: string }) => {
    blocks: AcpContentBlock[];
  };
  onConversationResolved?: (info: {
    sessionId: string;
    resumed: boolean;
    configOptions?: readonly AgyAcpConfigOption[];
  }) => void;
  onMissingConversation?: (missingId: string) => void;
  onQuotaExhausted?: () => void;
  onActivity?: (activity: AgyActivity) => void;
  onToolDelta?: (toolName: string) => void;
}

export interface AgyContinueWaveRequest {
  signal?: AbortSignal;
  onActivity?: (activity: AgyActivity) => void;
  onToolDelta?: (toolName: string) => void;
}

export interface AgyRuntimeTurnResult {
  sessionId: string;
  resumed: boolean;
  configOptions?: readonly AgyAcpConfigOption[];
  outcome: AgyTurnOutcome;
  lastContextUsage?: AgyContextUsage;
}

export interface AgySummaryTurnRequest {
  cwd: string;
  modelId: string;
  promptText: string;
  systemPromptAppend?: string;
  signal?: AbortSignal;
}

export interface AgyAcpRuntimeOptions {
  checkBinary?: () => Promise<AgyBinaryInfo>;
  createClient?: (options: AgyAcpClientOptions) => AgyAcpClient;
  cancelGraceMs?: number;
  maxApiAttempts?: number;
  transcriptPollIntervalMs?: number;
  brainDir?: string;
}

interface ParkedTurnState {
  sessionId: string;
  resumed: boolean;
  controller: AgyTurnController;
  turnAbortController: AbortController;
  getLastContextUsage: () => AgyContextUsage | undefined;
}

export class AgyAcpRuntime {
  readonly #checkBinary: () => Promise<AgyBinaryInfo>;
  readonly #createClient: (options: AgyAcpClientOptions) => AgyAcpClient;
  readonly #cancelGraceMs?: number;
  readonly #maxApiAttempts: number;
  readonly #transcriptPollIntervalMs: number;
  readonly #brainDir?: string;

  #client?: AgyAcpClient;
  #spawnKey?: string;
  #activeSessionId?: string;
  #systemPromptAppend?: string;
  #configOptions?: AgyAcpConfigOption[];
  #parkedTurn?: ParkedTurnState;

  constructor(options: AgyAcpRuntimeOptions = {}) {
    this.#checkBinary = options.checkBinary ?? (() => checkAgyBinary());
    this.#createClient = options.createClient ?? ((opts) => new AgyAcpClient(opts));
    this.#cancelGraceMs = options.cancelGraceMs;
    this.#maxApiAttempts = options.maxApiAttempts ?? DEFAULT_MAX_API_ATTEMPTS;
    this.#transcriptPollIntervalMs = options.transcriptPollIntervalMs ?? DEFAULT_TRANSCRIPT_POLL_MS;
    this.#brainDir = options.brainDir;
  }

  get isConnected(): boolean {
    return Boolean(this.#client?.isAlive);
  }

  get activeSessionId(): string | undefined {
    return this.#activeSessionId;
  }

  get activeConfigOptions(): readonly AgyAcpConfigOption[] | undefined {
    return this.#configOptions;
  }

  get hasParkedTurn(): boolean {
    return Boolean(this.#parkedTurn?.controller.isActive);
  }

  notifyBridgeCall(call: BridgeCall): boolean {
    if (!this.#parkedTurn) return false;
    this.#parkedTurn.controller.onBridgeCall(call);
    return true;
  }

  async resetConversation(reason = "Conversation reset"): Promise<void> {
    await this.#abortParkedTurn(reason);
    const prevId = this.#activeSessionId;
    this.#activeSessionId = undefined;
    this.#systemPromptAppend = undefined;
    this.#configOptions = undefined;
    if (prevId && this.#client?.isAlive) {
      await this.#client.closeSession(prevId).catch(() => {});
    }
  }

  async stop(reason = "Antigravity runtime stopped"): Promise<void> {
    await this.#abortParkedTurn(reason);
    this.#activeSessionId = undefined;
    this.#systemPromptAppend = undefined;
    this.#configOptions = undefined;
    this.#spawnKey = undefined;
    await this.#disposeClient();
  }

  async #abortParkedTurn(reason: string): Promise<void> {
    const parked = this.#parkedTurn;
    if (!parked) return;
    this.#parkedTurn = undefined;
    const err = new Error(reason);
    err.name = "AbortError";
    parked.turnAbortController.abort();
    parked.controller.fail(err);
    if (this.#client?.isAlive) {
      this.#client.cancelSession(parked.sessionId);
    }
    await this.#disposeClient();
  }

  async #disposeClient(): Promise<void> {
    const client = this.#client;
    this.#client = undefined;
    this.#spawnKey = undefined;
    this.#activeSessionId = undefined;
    this.#configOptions = undefined;
    if (client) {
      await client.dispose().catch(() => {});
    }
  }

  async #ensureClient(params: {
    binary: string;
    cwd: string;
    agentName?: string;
    extraDirs?: readonly string[];
    sandbox?: boolean;
    bridgeRevision?: string;
  }): Promise<AgyAcpClient> {
    const key = JSON.stringify({
      binary: params.binary,
      cwd: params.cwd,
      agentName: params.agentName ?? null,
      extraDirs: params.extraDirs ?? [],
      sandbox: Boolean(params.sandbox),
      bridgeRevision: params.bridgeRevision ?? null,
    });

    if (this.#client?.isAlive && this.#spawnKey === key) {
      return this.#client;
    }

    await this.#disposeClient();
    const client = this.#createClient({
      binary: params.binary,
      cwd: params.cwd,
      agentName: params.agentName,
      extraDirs: params.extraDirs,
      sandbox: params.sandbox,
      cancelGraceMs: this.#cancelGraceMs,
    });
    try {
      await client.initialize();
    } catch (error) {
      await client.dispose().catch(() => {});
      throw error;
    }
    this.#client = client;
    this.#spawnKey = key;
    return client;
  }

  async #applySessionConfig(
    client: AgyAcpClient,
    sessionId: string,
    desired: {
      modelId: string;
      effort?: string;
      executionMode?: AgyExecutionMode;
    },
  ): Promise<AgyAcpConfigOption[] | undefined> {
    let options = this.#configOptions;
    const modelOption = extractAcpConfigOption(options, "model");
    const candidate = resolveAcpModelCandidate(desired.modelId, modelOption);

    if (modelOption && candidate.modelId && modelOption.currentValue !== candidate.modelId) {
      const updated = await client.setConfigOption(sessionId, modelOption.id, candidate.modelId);
      options = updated.configOptions ?? options;
      this.#configOptions = options;
    }

    const writes = planAgyAcpSecondaryConfigWrites(options, {
      requestedEffort: desired.effort ?? candidate.impliedEffort,
      executionMode: desired.executionMode,
    });
    for (const write of writes) {
      const updated = await client.setConfigOption(sessionId, write.configId, write.value);
      options = updated.configOptions ?? options;
      this.#configOptions = options;
    }

    return options;
  }

  async runTurn(request: AgyRunTurnRequest): Promise<AgyRuntimeTurnResult> {
    if (this.#parkedTurn) {
      // Starting a brand-new user turn supersedes any unfinished parked wave.
      await this.#abortParkedTurn("Superseded by a new turn");
    }

    const { binary } = await this.#checkBinary();
    const client = await this.#ensureClient({
      binary,
      cwd: request.cwd,
      agentName: request.agentName,
      extraDirs: request.extraDirs,
      sandbox: request.sandbox,
      bridgeRevision: request.bridgeRevision,
    });

    let sessionId: string;
    let resumed = false;
    let usedHistoryBootstrap = false;

    const sameLoadedSession =
      this.#activeSessionId !== undefined &&
      this.#activeSessionId === request.desiredConversationId &&
      this.#systemPromptAppend === request.systemPromptAppend;

    if (sameLoadedSession) {
      sessionId = this.#activeSessionId as string;
      resumed = true;
      usedHistoryBootstrap = false;
    } else {
      if (this.#activeSessionId && this.#activeSessionId !== request.desiredConversationId) {
        await client.closeSession(this.#activeSessionId).catch(() => {});
        this.#activeSessionId = undefined;
        this.#configOptions = undefined;
      }

      if (request.desiredConversationId) {
        try {
          const res = await client.resumeSession({
            sessionId: request.desiredConversationId,
            cwd: request.cwd,
            systemPromptAppend: request.systemPromptAppend,
          });
          sessionId = res.sessionId;
          resumed = true;
          usedHistoryBootstrap = false;
          this.#activeSessionId = sessionId;
          this.#systemPromptAppend = request.systemPromptAppend;
          this.#configOptions = res.configOptions;
        } catch (error) {
          if (error instanceof MissingAgyConversationError) {
            request.onMissingConversation?.(request.desiredConversationId);
            const fresh = await client.newSession({
              cwd: request.cwd,
              systemPromptAppend: request.systemPromptAppend,
            });
            sessionId = fresh.sessionId;
            resumed = false;
            usedHistoryBootstrap = true;
            this.#activeSessionId = sessionId;
            this.#systemPromptAppend = request.systemPromptAppend;
            this.#configOptions = fresh.configOptions;
          } else {
            throw error;
          }
        }
      } else {
        const fresh = await client.newSession({
          cwd: request.cwd,
          systemPromptAppend: request.systemPromptAppend,
        });
        sessionId = fresh.sessionId;
        resumed = false;
        usedHistoryBootstrap = true;
        this.#activeSessionId = sessionId;
        this.#systemPromptAppend = request.systemPromptAppend;
        this.#configOptions = fresh.configOptions;
      }
    }

    const configOptions = await this.#applySessionConfig(client, sessionId, {
      modelId: request.modelId,
      effort: request.effort,
      executionMode: request.executionMode,
    });

    request.onConversationResolved?.({ sessionId, resumed, configOptions });

    const { blocks } = request.buildPrompt({ usedHistoryBootstrap, sessionId });

    const turnAbortController = new AbortController();
    const quotaTracker = new AgyQuotaTurnTracker();
    const toolsByCallId = new Map<string, AgyToolView>();
    const activeSteps = new Set<string>();
    const activeDaemons = new Set<string>();
    let lastContextUsage: AgyContextUsage | undefined;

    const controller = new AgyTurnController({
      onActivity: request.onActivity,
      onToolDelta: request.onToolDelta,
    });

    this.#parkedTurn = {
      sessionId,
      resumed,
      controller,
      turnAbortController,
      getLastContextUsage: () => lastContextUsage,
    };

    // Start the background ACP prompt loop; each Pi wave awaits `waitNextWave`.
    const transcriptPath = path.join(
      this.#brainDir ?? agyBrainDir(),
      sessionId,
      ".system_generated",
      "logs",
      "transcript.jsonl",
    );
    let lastRetryStep: number | undefined;
    let retryHintActive = false;
    let polling = false;
    let stopped = false;
    let promptSettled = false;

    const pollTranscriptRetry = async () => {
      if (polling || stopped) return;
      polling = true;
      let handle: Awaited<ReturnType<typeof fs.open>> | undefined;
      try {
        handle = await fs.open(transcriptPath, "r");
        const stat = await handle.stat();
        const readLen = Math.min(8_192, stat.size);
        if (readLen <= 0) return;
        const offset = Math.max(0, stat.size - readLen);
        const buf = Buffer.alloc(readLen);
        const { bytesRead } = await handle.read(buf, 0, readLen, offset);
        if (bytesRead <= 0 || stopped) return;
        const latest = extractLatestTranscriptRetryError(
          buf.subarray(0, bytesRead).toString("utf8"),
        );
        if (!latest) return;

        if (latest.attempt >= this.#maxApiAttempts) {
          clearRetryWatcher();
          if (isAgyQuotaError(latest.message) || /\bRESOURCE_EXHAUSTED\b/i.test(latest.message)) {
            request.onQuotaExhausted?.();
          }
          controller.fail(new AgyRetryExhaustedError(latest.message, latest.attempt));
          client.cancelSession(sessionId);
          const killTimer = setTimeout(() => {
            if (!promptSettled && this.#client === client) {
              void this.#disposeClient();
            }
          }, 1_000);
          killTimer.unref?.();
          return;
        }

        if (latest.stepIndex !== lastRetryStep) {
          lastRetryStep = latest.stepIndex;
          retryHintActive = true;
          controller.onActivity({
            type: "status_hint",
            text: `retry ${latest.attempt}/${this.#maxApiAttempts} — ${latest.message}`,
          });
        }
      } catch {
        // Missing or unreadable brain transcript while initializing is non-fatal.
      } finally {
        await handle?.close().catch(() => {});
        polling = false;
      }
    };

    const retryPollTimer = setInterval(() => {
      void pollTranscriptRetry();
    }, this.#transcriptPollIntervalMs);
    retryPollTimer.unref?.();

    const clearRetryWatcher = () => {
      stopped = true;
      clearInterval(retryPollTimer);
      if (retryHintActive) {
        retryHintActive = false;
        controller.onActivity({ type: "status_hint", text: "" });
      }
    };

    void client
      .prompt(sessionId, blocks, {
        signal: turnAbortController.signal,
        cancelGraceMs: this.#cancelGraceMs,
        onUpdate: (update) => {
          const activity = acpUpdateToActivity(update, toolsByCallId);
          if (!activity) return;
          if (
            retryHintActive &&
            (activity.type === "thinking_delta" ||
              activity.type === "text_delta" ||
              activity.type === "tool_start")
          ) {
            retryHintActive = false;
            controller.onActivity({ type: "status_hint", text: "" });
          }
          switch (activity.type) {
            case "text_delta":
              quotaTracker.noteText(activity.text);
              break;
            case "thinking_delta":
              quotaTracker.noteThought();
              break;
            case "context_usage":
              lastContextUsage = {
                usedTokens: activity.usedTokens,
                sizeTokens: activity.sizeTokens,
              };
              quotaTracker.noteUsageUpdate();
              break;
            case "tool_start":
            case "tool_done":
            case "tool_error":
              quotaTracker.noteToolActivity();
              trackActiveToolStep(activeSteps, activity, activeDaemons);
              break;
          }
          controller.onActivity(activity);
        },
      })
      .then((result) => {
        promptSettled = true;
        clearRetryWatcher();
        const usage = normalizeAcpPromptUsage(result.usage);
        const quotaNotice = quotaTracker.detectTurnQuotaExhaustion(result.usage);
        if (quotaNotice) {
          request.onQuotaExhausted?.();
          controller.fail(new AgyQuotaError(quotaNotice.message, quotaNotice.resetsIn));
          return;
        }
        controller.complete({
          stopReason: result.stopReason,
          usage,
        });
      })
      .catch((error) => {
        promptSettled = true;
        clearRetryWatcher();
        const err = error instanceof Error ? error : new Error(String(error));
        if (isAgyQuotaError(err.message)) {
          request.onQuotaExhausted?.();
        }
        if (!client.isAlive && this.#client === client) {
          this.#activeSessionId = undefined;
          this.#configOptions = undefined;
        }
        controller.fail(err);
      });

    return this.#awaitWave(this.#parkedTurn, request.signal);
  }

  async continueParkedTurn(request: AgyContinueWaveRequest): Promise<AgyRuntimeTurnResult> {
    const parked = this.#parkedTurn;
    if (!parked) {
      throw new Error("No parked Antigravity ACP turn to continue.");
    }
    parked.controller.setCallbacks({
      onActivity: request.onActivity,
      onToolDelta: request.onToolDelta,
    });
    return this.#awaitWave(parked, request.signal);
  }

  async #awaitWave(
    parked: ParkedTurnState,
    waveSignal?: AbortSignal,
  ): Promise<AgyRuntimeTurnResult> {
    const onWaveAbort = () => {
      parked.turnAbortController.abort();
    };
    if (waveSignal?.aborted) {
      onWaveAbort();
    } else {
      waveSignal?.addEventListener("abort", onWaveAbort, { once: true });
    }

    try {
      const outcome = await parked.controller.waitNextWave(waveSignal);
      if (outcome.kind === "complete" && this.#parkedTurn === parked) {
        this.#parkedTurn = undefined;
      }
      return {
        sessionId: parked.sessionId,
        resumed: parked.resumed,
        configOptions: this.#configOptions,
        outcome,
        lastContextUsage: parked.getLastContextUsage(),
      };
    } catch (error) {
      if (this.#parkedTurn === parked) {
        this.#parkedTurn = undefined;
      }
      if (waveSignal?.aborted) {
        parked.turnAbortController.abort();
      }
      throw error;
    } finally {
      waveSignal?.removeEventListener("abort", onWaveAbort);
    }
  }

  /**
   * Run a one-shot, tool-free `agy acp` session (e.g. for Pi compaction summaries)
   * without touching the main conversation session.
   */
  async runSummaryTurn(request: AgySummaryTurnRequest): Promise<{
    text: string;
    usage?: ReturnType<typeof normalizeAcpPromptUsage>;
  }> {
    const { binary } = await this.#checkBinary();
    const client = this.#createClient({
      binary,
      cwd: request.cwd,
      cancelGraceMs: this.#cancelGraceMs,
    });
    try {
      await client.initialize();
      const session = await client.newSession({
        cwd: request.cwd,
        systemPromptAppend: request.systemPromptAppend,
      });

      let options = session.configOptions;
      const modelOption = extractAcpConfigOption(options, "model");
      const candidate = resolveAcpModelCandidate(request.modelId, modelOption);
      if (modelOption && modelOption.currentValue !== candidate.modelId) {
        const updated = await client.setConfigOption(
          session.sessionId,
          modelOption.id,
          candidate.modelId,
        );
        options = updated.configOptions ?? options;
      }
      const modeOption = extractAcpConfigOption(options, "mode");
      if (modeOption && modeOption.currentValue !== "plan") {
        // Keep compaction/summary turns read-only where supported.
        const hasPlan = (modeOption.options ?? []).some((o) => o.value === "plan");
        if (hasPlan) {
          await client.setConfigOption(session.sessionId, modeOption.id, "plan");
        }
      }

      const toolsByCallId = new Map<string, AgyToolView>();
      let text = "";
      const result = await client.prompt(
        session.sessionId,
        [{ type: "text", text: request.promptText }],
        {
          signal: request.signal,
          onUpdate: (update) => {
            const activity = acpUpdateToActivity(update, toolsByCallId);
            if (activity?.type === "text_delta") {
              text += activity.text;
            }
          },
        },
      );
      await client.closeSession(session.sessionId).catch(() => {});
      return {
        text: text.trim(),
        usage: normalizeAcpPromptUsage(result.usage),
      };
    } finally {
      await client.dispose().catch(() => {});
    }
  }
}
