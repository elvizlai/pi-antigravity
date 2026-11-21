import type { TranscriptContext } from "@earendil-works/pi-ai";
import { checkAgyBinary } from "../acp/binary.ts";
import {
  type AcpContentBlock,
  type AcpInitializeResult,
  type AcpSessionResult,
  AgyAcpClient,
  type AgyAcpClientOptions,
} from "../acp/client.ts";
import {
  type AgyContextUsage,
  type AgyToolView,
  type AgyUsage,
  acpUpdateToActivity,
  isBridgedMcpTool,
  normalizeAcpPromptUsage,
  trackActiveToolStep,
} from "../acp/events.ts";
import {
  type AgyAcpConfigOption,
  type AgyExecutionMode,
  extractAcpConfigOption,
  isAgyOrchestratorModel,
  resolveAcpModelCandidate,
  resolveAcpThoughtLevel,
} from "../acp/models.ts";
import { type AgyQuotaNotice, AgyQuotaTurnTracker, isAgyQuotaError } from "../acp/quota.ts";
import { createPiToolsAgent, type PiToolsAgentHandle } from "../bridge/agent-files.ts";
import type { AgyPiBridge, BridgeCall } from "../bridge/server.ts";
import { AGY_PARKED_TURN_ERROR, agyIncompleteToolError, collectTurnToolResults } from "./prompt.ts";
import type { AgyToolMode } from "./settings.ts";
import {
  type AgyConversationState,
  findActiveAgyConversation,
  type SessionEntryLike,
} from "./state.ts";
import { type AgyReplayStore, AgyTurnController } from "./turn.ts";

export const DEFAULT_FIRST_RESPONSE_TIMEOUT_MS = 60_000;

/**
 * Quota exhaustion error (`"quota exceeded"` matches `@earendil-works/pi-ai`
 * `NON_RETRYABLE_PROVIDER_LIMIT_ERROR_PATTERN`, so Pi stops immediately with 0 retries).
 */
export class AgyQuotaError extends Error {
  readonly resetsIn?: string;
  readonly notice?: AgyQuotaNotice;

  constructor(message: string, options?: { resetsIn?: string; notice?: AgyQuotaNotice }) {
    const body = message.trim() || "Antigravity account quota has been reached.";
    const formatted = /quota exceeded/i.test(body) ? body : `Antigravity quota exceeded: ${body}`;
    super(formatted);
    this.name = "AgyQuotaError";
    this.resetsIn = options?.resetsIn ?? options?.notice?.resetsIn;
    this.notice = options?.notice;
  }
}

export function toAgyQuotaError(raw: unknown, resetsIn?: string): AgyQuotaError {
  const text = raw instanceof Error ? raw.message : String(raw ?? "");
  return new AgyQuotaError(text, { resetsIn });
}

/**
 * First-event watchdog error: avoids `"timed out"` / `"timeout"` so Pi does not
 * stack extra 60s retries on top of `agy`'s own internal backoff window. Fast
 * transient RPC/transport errors (e.g. `503` / `ECONNRESET`) still route to Pi's
 * native `autoRetry` (`maxRetries: 3`).
 */
export class AgyFirstResponseTimeoutError extends Error {
  readonly timeoutMs: number;

  constructor(timeoutMs: number) {
    const seconds = Math.max(1, Math.round(timeoutMs / 1000));
    super(
      `antigravity: no initial ACP response from agy within ${seconds}s — cancelled stalled turn`,
    );
    this.name = "AgyFirstResponseTimeoutError";
    this.timeoutMs = timeoutMs;
  }
}

export interface AgyAcpClientLike {
  readonly isAlive: boolean;
  initialize(): Promise<AcpInitializeResult>;
  newSession(options: { cwd: string; mcpServers?: unknown[] }): Promise<AcpSessionResult>;
  resumeSession(options: {
    sessionId: string;
    cwd: string;
    mcpServers?: unknown[];
  }): Promise<AcpSessionResult>;
  setConfigOption(
    sessionId: string,
    configId: string,
    value: string,
  ): Promise<{ configOptions?: AgyAcpConfigOption[] }>;
  cancelSession(sessionId: string): void;
  prompt(
    sessionId: string,
    blocks: readonly AcpContentBlock[],
    options?: {
      signal?: AbortSignal;
      cancelGraceMs?: number;
      onUpdate?: (update: Record<string, unknown>) => void;
    },
  ): Promise<{ stopReason: string; usage?: unknown }>;
  dispose(graceMs?: number): Promise<void>;
}

export interface AgyRuntimeDeps {
  cwd: string;
  bridge?: AgyPiBridge;
  bridgeToken?: string;
  getToolMode?: () => AgyToolMode;
  getExecutionMode?: () => AgyExecutionMode;
  sandbox?: boolean;
  extraDirs?: readonly string[];
  binaryPath?: string;
  agentsDir?: string;
  firstResponseTimeoutMs?: number;
  ensureBridge?: () => Promise<boolean>;
  checkBinary?: (
    binaryPath?: string,
    signal?: AbortSignal,
  ) => Promise<{ binary: string; version: string }>;
  createClient?: (options: AgyAcpClientOptions) => AgyAcpClientLike;
  onPersistedState?: (state: AgyConversationState) => void;
  onContextUsage?: (usage: AgyContextUsage) => void;
  onWarning?: (message: string) => void;
}

export interface StartTurnParams {
  modelId: string;
  effort?: string;
  executionMode?: AgyExecutionMode;
  prompt: AcpContentBlock[];
  systemPromptAppend?: string;
  messages: TranscriptContext["messages"];
  sessionId: string;
  branchEntries: readonly SessionEntryLike[];
  env?: Record<string, string>;
  timeoutMs?: number;
  signal?: AbortSignal;
  replayStore: AgyReplayStore;
}

export interface TurnSessionHandle {
  controller: AgyTurnController;
  attachExisting: boolean;
  bridgeServerName?: string;
  signalTurnAbort: () => void;
}

interface ActiveTurn {
  controller: AgyTurnController;
  sessionId: string;
  modelId: string;
  piSessionId: string;
  toolMode: AgyToolMode;
  bridgeServerName?: string;
  noteFirstResponse: () => void;
  requestAbort: () => void;
}

function stableEnvFingerprint(env: Record<string, string> | undefined): string {
  if (!env) return "";
  const entries = Object.entries(env).sort(([a], [b]) => a.localeCompare(b));
  return JSON.stringify(entries);
}

function sumUsage(prev: AgyUsage, next: AgyUsage | undefined): AgyUsage {
  if (!next) return { ...prev };
  const add = (a: number | undefined, b: number | undefined): number | undefined => {
    if (a === undefined && b === undefined) return undefined;
    return (a ?? 0) + (b ?? 0);
  };
  return {
    input_tokens: add(prev.input_tokens, next.input_tokens),
    output_tokens: add(prev.output_tokens, next.output_tokens),
    cache_read_tokens: add(prev.cache_read_tokens, next.cache_read_tokens),
    cache_creation_tokens: add(prev.cache_creation_tokens, next.cache_creation_tokens),
    reasoning_tokens: add(prev.reasoning_tokens, next.reasoning_tokens),
    thinking_tokens: add(prev.thinking_tokens, next.thinking_tokens),
    total_tokens: add(prev.total_tokens, next.total_tokens),
  };
}

/**
 * Owns the persistent `agy acp` child process, session/resume lifecycle, config
 * synchronization (`mode -> model -> thought_level`), bridged MCP wave loop,
 * quota detection, and single-attempt first-response stall timeout.
 *
 * Implements zero custom retry loops; transient failures surface cleanly as
 * `AgyFirstResponseTimeoutError` / transport errors for Pi's native retry loop.
 */
export class AgyRuntime {
  readonly #deps: AgyRuntimeDeps;
  #client: AgyAcpClientLike | undefined;
  #clientFingerprint: string | undefined;
  #pendingDispose: Promise<void> | undefined;
  #activeWarmup: Promise<void> | undefined;
  #warmupAbort: AbortController | undefined;
  #agentHandle: PiToolsAgentHandle | undefined;
  #agentKey: string | undefined;
  #conversationId: string | undefined;
  #piSessionId: string | undefined;
  #openedOnClient: string | undefined;
  #configOptions: AgyAcpConfigOption[] | undefined;
  #turns = 0;
  #usage: AgyUsage = {};
  #contextTokens: number | undefined;
  #lastQuotaError: string | undefined;
  #activeTurn: ActiveTurn | undefined;

  constructor(deps: AgyRuntimeDeps) {
    this.#deps = deps;
    this.#deps.bridge?.setOnCall((call) => this.#handleBridgeCall(call));
  }

  get conversationId(): string | undefined {
    return this.#conversationId;
  }

  get lastQuotaError(): string | undefined {
    return this.#lastQuotaError;
  }

  get isConnected(): boolean {
    return Boolean(this.#client?.isAlive);
  }

  async resetConversation(): Promise<void> {
    if (this.#activeTurn) {
      this.#activeTurn.requestAbort();
    }
    this.#conversationId = undefined;
    this.#piSessionId = undefined;
    this.#openedOnClient = undefined;
    this.#configOptions = undefined;
    this.#turns = 0;
    this.#usage = {};
    this.#contextTokens = undefined;
    this.#lastQuotaError = undefined;
    await this.dispose();
  }

  async dispose(): Promise<void> {
    this.#warmupAbort?.abort();
    this.#warmupAbort = undefined;
    if (this.#activeWarmup) {
      await this.#activeWarmup.catch(() => {});
    }
    const active = this.#activeTurn;
    this.#activeTurn = undefined;
    if (active) {
      active.requestAbort();
    }
    await this.#disposeClient();
    await this.#removeAgentFile();
  }

  async #runWarmup(
    externalSignal: AbortSignal | undefined,
    fn: (signal: AbortSignal) => Promise<void>,
  ): Promise<void> {
    const abort = new AbortController();
    this.#warmupAbort = abort;
    const onAbort = () => abort.abort();
    if (externalSignal) {
      if (externalSignal.aborted) {
        abort.abort();
      } else {
        externalSignal.addEventListener("abort", onAbort, { once: true });
      }
    }
    const task = fn(abort.signal).finally(() => {
      externalSignal?.removeEventListener("abort", onAbort);
      if (this.#warmupAbort === abort) {
        this.#warmupAbort = undefined;
      }
      if (this.#activeWarmup === task) {
        this.#activeWarmup = undefined;
      }
    });
    this.#activeWarmup = task;
    await task;
  }

  /** Pre-start and initialize `agy acp` + bridge agent without creating an empty ACP conversation. */
  async warmupClient(params: {
    modelId: string;
    env?: Record<string, string>;
    signal?: AbortSignal;
  }): Promise<void> {
    if (this.#activeTurn) return;
    if (this.#activeWarmup) {
      await this.#activeWarmup.catch(() => {});
    }
    if (params.signal?.aborted || this.#activeTurn) return;
    await this.#runWarmup(params.signal, async (signal) => {
      if (signal.aborted || this.#activeTurn) return;
      const toolMode = this.#deps.getToolMode?.() ?? "pi";
      this.#deps.bridge?.refreshTools();
      const { agentName } = await this.#ensurePiToolsAgent(toolMode, params.modelId);
      if (signal.aborted || this.#activeTurn) return;
      await this.#ensureClient({ agentName, env: params.env, signal });
    });
  }

  /** Pre-open/resume and configure the active ACP session right before `streamSimple` begins timing. */
  async warmupSession(params: {
    modelId: string;
    effort?: string;
    executionMode?: AgyExecutionMode;
    sessionId: string;
    branchEntries: readonly SessionEntryLike[];
    env?: Record<string, string>;
    signal?: AbortSignal;
  }): Promise<void> {
    if (this.#activeTurn) return;
    if (this.#activeWarmup) {
      await this.#activeWarmup.catch(() => {});
    }
    if (params.signal?.aborted || this.#activeTurn) return;
    await this.#runWarmup(params.signal, async (signal) => {
      if (signal.aborted || this.#activeTurn) return;
      const executionMode = params.executionMode ?? this.#deps.getExecutionMode?.() ?? "agent";
      await this.#prepareConfiguredSession({
        modelId: params.modelId,
        effort: params.effort,
        executionMode,
        sessionId: params.sessionId,
        branchEntries: params.branchEntries,
        env: params.env,
        signal,
      });
    });
  }

  #handleBridgeCall(call: BridgeCall): boolean {
    const active = this.#activeTurn;
    if (!active || active.controller.isSettled) return false;
    active.noteFirstResponse();
    active.controller.pushActivity({
      type: "bridge_call",
      callId: call.id,
      name: call.tool,
      args: call.args,
    });
    return true;
  }

  async #disposeClient(graceMs = 300): Promise<void> {
    const client = this.#client;
    this.#client = undefined;
    this.#clientFingerprint = undefined;
    this.#openedOnClient = undefined;
    this.#configOptions = undefined;
    if (this.#pendingDispose) {
      await this.#pendingDispose.catch(() => {});
    }
    if (!client) return;
    const task = client.dispose(graceMs).catch(() => {});
    this.#pendingDispose = task;
    try {
      await task;
    } finally {
      if (this.#pendingDispose === task) {
        this.#pendingDispose = undefined;
      }
    }
  }

  async #removeAgentFile(): Promise<void> {
    const handle = this.#agentHandle;
    this.#agentHandle = undefined;
    this.#agentKey = undefined;
    if (handle) {
      await handle.remove().catch(() => {});
    }
  }

  async #ensurePiToolsAgent(
    toolMode: AgyToolMode,
    modelId: string,
  ): Promise<{ agentName?: string; bridgeServerName?: string }> {
    const bridge = this.#deps.bridge;
    if (toolMode !== "pi" || !bridge) {
      await this.#removeAgentFile();
      return {};
    }

    const ready = this.#deps.ensureBridge ? await this.#deps.ensureBridge() : bridge.running;
    if (!ready || !bridge.url) {
      await this.#removeAgentFile();
      return {};
    }

    const orchestrator = isAgyOrchestratorModel(modelId);
    const desiredKey = `${bridge.serverName}:${bridge.url}:${orchestrator}`;
    if (this.#agentHandle && this.#agentKey === desiredKey) {
      return { agentName: this.#agentHandle.agentName, bridgeServerName: bridge.serverName };
    }

    await this.#removeAgentFile();
    const token = this.#deps.bridgeToken ?? "";
    const handle = createPiToolsAgent(
      {
        serverName: bridge.serverName,
        url: bridge.url,
        token,
      },
      {
        agentsDir: this.#deps.agentsDir,
        modelId,
        orchestrator,
      },
    );
    await handle.write();
    this.#agentHandle = handle;
    this.#agentKey = desiredKey;
    return { agentName: handle.agentName, bridgeServerName: bridge.serverName };
  }

  async #ensureClient(params: {
    agentName?: string;
    env?: Record<string, string>;
    signal?: AbortSignal;
  }): Promise<AgyAcpClientLike> {
    if (this.#pendingDispose) {
      await this.#pendingDispose.catch(() => {});
    }

    const fingerprint = JSON.stringify({
      agentName: params.agentName ?? "",
      bridgeCatalogRevision: params.agentName ? (this.#deps.bridge?.catalogRevision ?? 0) : 0,
      sandbox: Boolean(this.#deps.sandbox),
      extraDirs: this.#deps.extraDirs ?? [],
      env: stableEnvFingerprint(params.env),
    });

    if (this.#client?.isAlive && this.#clientFingerprint === fingerprint) {
      return this.#client;
    }

    await this.#disposeClient();

    const verified = this.#deps.checkBinary
      ? await this.#deps.checkBinary(this.#deps.binaryPath, params.signal)
      : await checkAgyBinary({ binaryPath: this.#deps.binaryPath, signal: params.signal });

    const clientOptions: AgyAcpClientOptions = {
      binary: verified.binary,
      cwd: this.#deps.cwd,
      agentName: params.agentName,
      extraDirs: this.#deps.extraDirs,
      sandbox: this.#deps.sandbox,
      env: params.env,
    };

    const client = this.#deps.createClient
      ? this.#deps.createClient(clientOptions)
      : new AgyAcpClient(clientOptions);
    this.#client = client;
    this.#clientFingerprint = fingerprint;
    try {
      await client.initialize();
      return client;
    } catch (err) {
      await this.#disposeClient();
      throw err;
    }
  }

  async #openOrResumeSession(
    client: AgyAcpClientLike,
    params: {
      piSessionId: string;
      branchEntries: readonly SessionEntryLike[];
      systemPromptAppend?: string;
    },
  ): Promise<{ sessionId: string; configOptions?: AgyAcpConfigOption[] }> {
    if (
      this.#openedOnClient &&
      this.#client === client &&
      this.#piSessionId === params.piSessionId
    ) {
      return { sessionId: this.#openedOnClient, configOptions: this.#configOptions };
    }

    const fromState = findActiveAgyConversation(
      params.piSessionId,
      this.#deps.cwd,
      params.branchEntries,
    );
    const candidateId =
      fromState?.conversationId ??
      (this.#piSessionId === params.piSessionId ? this.#conversationId : undefined);

    if (fromState && fromState.conversationId !== this.#conversationId) {
      this.#conversationId = fromState.conversationId;
      this.#piSessionId = params.piSessionId;
      this.#turns = fromState.turns;
      this.#usage = { ...fromState.usage };
      this.#contextTokens = fromState.contextTokens;
    }

    if (candidateId && this.#openedOnClient === candidateId && this.#client === client) {
      return { sessionId: candidateId, configOptions: this.#configOptions };
    }

    if (candidateId) {
      try {
        const resumed = await client.resumeSession({
          sessionId: candidateId,
          cwd: this.#deps.cwd,
        });
        this.#conversationId = resumed.sessionId;
        this.#piSessionId = params.piSessionId;
        this.#openedOnClient = resumed.sessionId;
        this.#configOptions = resumed.configOptions;
        return { sessionId: resumed.sessionId, configOptions: resumed.configOptions };
      } catch {
        // Fallback to a fresh session when the stored ACP conversation cannot resume.
        this.#conversationId = undefined;
        this.#turns = 0;
        this.#usage = {};
        this.#contextTokens = undefined;
      }
    }

    const created = await client.newSession({
      cwd: this.#deps.cwd,
    });
    this.#conversationId = created.sessionId;
    this.#piSessionId = params.piSessionId;
    this.#openedOnClient = created.sessionId;
    this.#configOptions = created.configOptions;
    this.#turns = 0;
    this.#usage = {};
    this.#contextTokens = undefined;
    return { sessionId: created.sessionId, configOptions: created.configOptions };
  }

  /**
   * Apply session config options in strict dependency order:
   * `mode -> model -> thought_level` (`model` can alter supported `thought_level` options).
   */
  async #syncSessionConfig(
    client: AgyAcpClientLike,
    sessionId: string,
    desired: { modelId: string; effort?: string; executionMode: AgyExecutionMode },
    initialOptions?: readonly AgyAcpConfigOption[],
  ): Promise<void> {
    let currentOptions = initialOptions;

    const modeOption = extractAcpConfigOption(currentOptions, "mode");
    if (modeOption && modeOption.currentValue !== desired.executionMode) {
      const supported = new Set((modeOption.options ?? []).map((o) => o.value));
      if (supported.size === 0 || supported.has(desired.executionMode)) {
        const updated = await client.setConfigOption(
          sessionId,
          modeOption.id,
          desired.executionMode,
        );
        if (updated.configOptions) currentOptions = updated.configOptions;
      }
    }

    const modelOption = extractAcpConfigOption(currentOptions, "model");
    const resolvedModel = resolveAcpModelCandidate(desired.modelId, modelOption);
    if (modelOption && modelOption.currentValue !== resolvedModel.modelId) {
      const updated = await client.setConfigOption(
        sessionId,
        modelOption.id,
        resolvedModel.modelId,
      );
      if (updated.configOptions) currentOptions = updated.configOptions;
    }

    const thoughtOption = extractAcpConfigOption(currentOptions, "thought_level");
    const resolvedThought = resolveAcpThoughtLevel(
      desired.effort ?? resolvedModel.impliedEffort,
      thoughtOption,
    );
    if (thoughtOption && resolvedThought && thoughtOption.currentValue !== resolvedThought) {
      const updated = await client.setConfigOption(sessionId, thoughtOption.id, resolvedThought);
      if (updated.configOptions) currentOptions = updated.configOptions;
    }

    this.#configOptions = currentOptions ? [...currentOptions] : undefined;
  }

  async #prepareConfiguredSession(params: {
    modelId: string;
    effort?: string;
    executionMode: AgyExecutionMode;
    sessionId: string;
    branchEntries: readonly SessionEntryLike[];
    env?: Record<string, string>;
    signal?: AbortSignal;
  }): Promise<{
    client: AgyAcpClientLike;
    sessionId: string;
    toolMode: AgyToolMode;
    bridgeServerName?: string;
  }> {
    const toolMode = this.#deps.getToolMode?.() ?? "pi";
    this.#deps.bridge?.refreshTools();
    const { agentName, bridgeServerName } = await this.#ensurePiToolsAgent(
      toolMode,
      params.modelId,
    );
    const client = await this.#ensureClient({
      agentName,
      env: params.env,
      signal: params.signal,
    });
    const session = await this.#openOrResumeSession(client, {
      piSessionId: params.sessionId,
      branchEntries: params.branchEntries,
    });
    await this.#syncSessionConfig(
      client,
      session.sessionId,
      { modelId: params.modelId, effort: params.effort, executionMode: params.executionMode },
      session.configOptions,
    );
    return { client, sessionId: session.sessionId, toolMode, bridgeServerName };
  }

  async startOrContinueTurn(params: StartTurnParams): Promise<TurnSessionHandle> {
    const active = this.#activeTurn;

    // 1. Continuation wave: resolve pending bridged Pi tool calls and re-attach to active turn.
    if (active && !active.controller.isSettled) {
      this.resolveBridgeResultsFromMessages(params.messages);
      return {
        controller: active.controller,
        attachExisting: true,
        bridgeServerName: active.bridgeServerName,
        signalTurnAbort: () => active.requestAbort(),
      };
    }

    if (params.signal?.aborted) {
      const abort = new Error("Antigravity turn aborted.");
      abort.name = "AbortError";
      throw abort;
    }

    if (this.#activeWarmup) {
      await this.#activeWarmup.catch(() => {});
    }

    // 2. Fresh ACP prompt turn.
    const executionMode = params.executionMode ?? this.#deps.getExecutionMode?.() ?? "agent";
    const { client, sessionId, toolMode, bridgeServerName } = await this.#prepareConfiguredSession({
      modelId: params.modelId,
      effort: params.effort,
      executionMode,
      sessionId: params.sessionId,
      branchEntries: params.branchEntries,
      env: params.env,
      signal: params.signal,
    });

    const controller = new AgyTurnController();
    const acpTools = new Map<string, AgyToolView>();
    const activeToolSteps = new Set<string>();
    const daemonToolSteps = new Set<string>();
    const quotaTracker = new AgyQuotaTurnTracker();
    const promptAbort = new AbortController();

    const baseTimeoutMs = this.#deps.firstResponseTimeoutMs ?? DEFAULT_FIRST_RESPONSE_TIMEOUT_MS;
    const effectiveTimeoutMs =
      typeof params.timeoutMs === "number" && params.timeoutMs > 0
        ? Math.min(params.timeoutMs, baseTimeoutMs)
        : baseTimeoutMs;

    let sawFirstResponse = false;
    let timedOut = false;
    let timeoutTimer: ReturnType<typeof setTimeout> | undefined;

    const clearTimeoutTimer = () => {
      if (timeoutTimer) {
        clearTimeout(timeoutTimer);
        timeoutTimer = undefined;
      }
    };

    const noteFirstResponse = () => {
      if (sawFirstResponse) return;
      sawFirstResponse = true;
      clearTimeoutTimer();
    };

    const activeTurn: ActiveTurn = {
      controller,
      sessionId,
      modelId: params.modelId,
      piSessionId: params.sessionId,
      toolMode,
      bridgeServerName,
      noteFirstResponse,
      requestAbort: () => {
        clearTimeoutTimer();
        promptAbort.abort();
      },
    };
    this.#activeTurn = activeTurn;

    const cleanupActive = () => {
      clearTimeoutTimer();
      if (this.#activeTurn === activeTurn) {
        this.#activeTurn = undefined;
      }
    };

    timeoutTimer = setTimeout(() => {
      if (sawFirstResponse || controller.isSettled || promptAbort.signal.aborted) return;
      timedOut = true;
      cleanupActive();
      try {
        client.cancelSession(sessionId);
      } catch {
        // Best-effort cancel before hard client disposal.
      }
      if (this.#turns === 0) {
        this.#conversationId = undefined;
      }
      void this.#disposeClient(50);
      controller.fail(new AgyFirstResponseTimeoutError(effectiveTimeoutMs));
    }, effectiveTimeoutMs);
    timeoutTimer.unref?.();

    const onExternalAbort = () => {
      activeTurn.requestAbort();
    };
    if (params.signal) {
      if (params.signal.aborted) {
        onExternalAbort();
      } else {
        params.signal.addEventListener("abort", onExternalAbort, { once: true });
      }
    }

    const outboundPrompt = [...params.prompt];
    const customSys = params.systemPromptAppend?.trim();
    if (this.#turns === 0 && customSys) {
      const firstTextIdx = outboundPrompt.findIndex((block) => block.type === "text");
      if (firstTextIdx >= 0 && outboundPrompt[firstTextIdx]?.type === "text") {
        outboundPrompt[firstTextIdx] = {
          type: "text",
          text: `${customSys}\n\n${outboundPrompt[firstTextIdx].text}`,
        };
      } else {
        outboundPrompt.unshift({ type: "text", text: customSys });
      }
    }

    // Fire `session/prompt` asynchronously while `streamSimple` consumes `controller.drainWave()`.
    void client
      .prompt(sessionId, outboundPrompt, {
        signal: promptAbort.signal,
        onUpdate: (update) => {
          if (timedOut || controller.isSettled) return;
          const activity = acpUpdateToActivity(update, acpTools);
          if (!activity || activity.type === "status_hint") return;

          if (activity.type === "context_usage") {
            this.#contextTokens = activity.usedTokens;
            // Pre-turn `usage_update` (usedTokens === 0) must not disarm the watchdog or quota guard.
            if (activity.usedTokens > 0) {
              noteFirstResponse();
              quotaTracker.noteUsageUpdate();
            }
            this.#deps.onContextUsage?.({
              usedTokens: activity.usedTokens,
              sizeTokens: activity.sizeTokens,
            });
            controller.pushActivity(activity);
            return;
          }

          noteFirstResponse();

          if (activity.type === "thinking_delta") {
            quotaTracker.noteThought();
            controller.pushActivity(activity);
            return;
          }

          if (activity.type === "text_delta") {
            quotaTracker.noteText(activity.text);
            controller.pushActivity(activity);
            return;
          }

          if (isBridgedMcpTool(activity, bridgeServerName)) {
            quotaTracker.noteToolActivity();
            return;
          }

          trackActiveToolStep(activeToolSteps, activity, daemonToolSteps);
          quotaTracker.noteToolActivity();
          controller.pushActivity(activity);
        },
      })
      .then((promptResult) => {
        params.signal?.removeEventListener("abort", onExternalAbort);
        if (timedOut || controller.isSettled) return;
        cleanupActive();

        const normalizedUsage = normalizeAcpPromptUsage(promptResult.usage);
        const quotaNotice = quotaTracker.detectTurnQuotaExhaustion({
          totalTokens: normalizedUsage?.total_tokens,
          outputTokens: normalizedUsage?.output_tokens,
        });
        if (quotaNotice) {
          this.#lastQuotaError = quotaNotice.message;
          controller.fail(toAgyQuotaError(quotaNotice.message, quotaNotice.resetsIn));
          return;
        }

        const isParkedDaemon =
          activeToolSteps.size > 0 && daemonToolSteps.size === activeToolSteps.size;

        if (!isParkedDaemon && promptResult.stopReason === "error") {
          controller.fail(new Error("Antigravity ACP turn failed with stopReason: error."));
          return;
        }

        // Materialize unfinished native tool steps before settling so Pi renders their final state.
        for (const [callId, tool] of acpTools.entries()) {
          if (tool.finishedEmitted || isBridgedMcpTool(tool, bridgeServerName)) continue;
          tool.finishedEmitted = true;
          const isDaemon = daemonToolSteps.has(`call:${callId}`);
          controller.pushActivity({
            type: "tool_error",
            callId,
            name: tool.name,
            title: tool.title,
            kind: tool.kind,
            args: tool.args,
            message: isDaemon ? AGY_PARKED_TURN_ERROR : agyIncompleteToolError(tool.name),
            meta: tool.meta,
          });
        }

        this.#turns += 1;
        this.#usage = sumUsage(this.#usage, normalizedUsage);
        this.#lastQuotaError = undefined;
        this.#deps.onPersistedState?.({
          version: 1,
          kind: "conversation",
          sessionId: params.sessionId,
          conversationId: sessionId,
          cwd: this.#deps.cwd,
          modelId: params.modelId,
          toolMode,
          ...(bridgeServerName ? { bridgeServer: bridgeServerName } : {}),
          turns: this.#turns,
          usage: this.#usage,
          ...(this.#contextTokens !== undefined ? { contextTokens: this.#contextTokens } : {}),
        });

        controller.finish({
          status: isParkedDaemon ? "parked_daemon" : "completed",
          stopReason: promptResult.stopReason,
          usage: normalizedUsage,
        });
      })
      .catch((err) => {
        params.signal?.removeEventListener("abort", onExternalAbort);
        if (timedOut || controller.isSettled) return;
        cleanupActive();

        if (promptAbort.signal.aborted || (err instanceof Error && err.name === "AbortError")) {
          controller.finish({ status: "aborted", stopReason: "aborted" });
          return;
        }

        const error = err instanceof Error ? err : new Error(String(err));
        if (isAgyQuotaError(error.message)) {
          this.#lastQuotaError = error.message;
          controller.fail(toAgyQuotaError(error.message));
          return;
        }

        if (!sawFirstResponse || !client.isAlive) {
          if (this.#turns === 0) this.#conversationId = undefined;
          void this.#disposeClient(50);
        }
        controller.fail(error);
      });

    return {
      controller,
      attachExisting: false,
      bridgeServerName,
      signalTurnAbort: () => activeTurn.requestAbort(),
    };
  }

  /** Resolve pending bridged MCP calls from trailing `toolResult` messages in context. */
  resolveBridgeResultsFromMessages(messages: Parameters<typeof collectTurnToolResults>[0]): number {
    const bridge = this.#deps.bridge;
    if (!bridge || bridge.pendingCount === 0) return 0;
    const pendingIds = new Set(bridge.pendingCallIds);
    const results = collectTurnToolResults(messages, pendingIds);
    let resolved = 0;
    for (const item of results) {
      if (
        bridge.resolveCall(item.callId, {
          content: item.content,
          isError: item.isError,
        })
      ) {
        resolved += 1;
      }
    }
    return resolved;
  }

  /** Expose parked-turn error marker for incomplete daemon rendering tests/stream consumers. */
  static readonly parkedTurnError = AGY_PARKED_TURN_ERROR;
}
