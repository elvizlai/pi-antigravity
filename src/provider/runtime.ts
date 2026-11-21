import type { TranscriptContext } from "@earendil-works/pi-ai";
import { checkAgyBinary } from "../acp/binary.ts";
import {
  type AcpContentBlock,
  type AcpHttpMcpServer,
  type AcpInitializeResult,
  type AcpSessionResult,
  AgyAcpClient,
  type AgyAcpClientOptions,
} from "../acp/client.ts";
import {
  type AgyToolView,
  acpUpdateToActivity,
  isBridgedMcpTool,
  normalizeAcpPromptUsage,
} from "../acp/events.ts";
import {
  type AgyAcpConfigOption,
  type AgyExecutionMode,
  extractAcpConfigOption,
  isAgyOrchestratorModel,
  resolveAcpExecutionMode,
  resolveAcpModelCandidate,
  resolveAcpThoughtLevel,
} from "../acp/models.ts";
import { type AgyQuotaNotice, AgyQuotaTurnTracker, isAgyQuotaError } from "../acp/quota.ts";
import { createPiToolsAgent, type PiToolsAgentHandle } from "../bridge/agent-files.ts";
import type { AgyPiBridge, BridgeCall } from "../bridge/server.ts";
import { collectTurnToolResults, INCOMPLETE_ACP_TOOL_MESSAGE } from "./prompt.ts";
import { AgyWorkingStatusTracker } from "./render.ts";
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
  newSession(options: {
    cwd: string;
    mcpServers?: readonly AcpHttpMcpServer[];
  }): Promise<AcpSessionResult>;
  resumeSession(options: {
    sessionId: string;
    cwd: string;
    mcpServers?: readonly AcpHttpMcpServer[];
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
  onWarning?: (message: string) => void;
  onWorkingStatus?: (message: string | undefined) => void;
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
  noteBridgeCall: () => void;
  requestAbort: () => void;
}

function stableEnvFingerprint(env: Record<string, string> | undefined): string {
  if (!env) return "";
  const entries = Object.entries(env).sort(([a], [b]) => a.localeCompare(b));
  return JSON.stringify(entries);
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
      this.#deps.onWorkingStatus?.(undefined);
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
      const { agentName, mcpServers } = await this.#ensurePiToolsAgent(toolMode, params.modelId);
      if (signal.aborted || this.#activeTurn) return;
      await this.#ensureClient({ agentName, mcpServers, env: params.env, signal });
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
      const executionMode = params.executionMode ?? this.#deps.getExecutionMode?.() ?? "default";
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
    active.noteBridgeCall();
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
  ): Promise<{ agentName?: string; bridgeServerName?: string; mcpServers: AcpHttpMcpServer[] }> {
    const bridge = this.#deps.bridge;
    if (toolMode !== "pi" || !bridge) {
      await this.#removeAgentFile();
      return { mcpServers: [] };
    }

    const ready = this.#deps.ensureBridge ? await this.#deps.ensureBridge() : bridge.running;
    if (!ready || !bridge.url) {
      await this.#removeAgentFile();
      return { mcpServers: [] };
    }

    const token = this.#deps.bridgeToken ?? "";
    const mcpServers: AcpHttpMcpServer[] = [
      {
        type: "http",
        name: bridge.serverName,
        url: bridge.url,
        headers: token ? [{ name: "x-pi-bridge-token", value: token }] : [],
      },
    ];

    const orchestrator = isAgyOrchestratorModel(modelId);
    const desiredKey = `${bridge.serverName}:${orchestrator}`;
    if (this.#agentHandle && this.#agentKey === desiredKey) {
      return {
        agentName: this.#agentHandle.agentName,
        bridgeServerName: bridge.serverName,
        mcpServers,
      };
    }

    await this.#removeAgentFile();
    const handle = createPiToolsAgent(
      { serverName: bridge.serverName },
      {
        agentsDir: this.#deps.agentsDir,
        modelId,
        orchestrator,
      },
    );
    await handle.write();
    this.#agentHandle = handle;
    this.#agentKey = desiredKey;
    return {
      agentName: handle.agentName,
      bridgeServerName: bridge.serverName,
      mcpServers,
    };
  }

  async #ensureClient(params: {
    agentName?: string;
    mcpServers?: readonly AcpHttpMcpServer[];
    env?: Record<string, string>;
    signal?: AbortSignal;
  }): Promise<AgyAcpClientLike> {
    if (this.#pendingDispose) {
      await this.#pendingDispose.catch(() => {});
    }

    const fingerprint = JSON.stringify({
      agentName: params.agentName ?? "",
      bridgeUrl: params.mcpServers?.[0]?.url ?? "",
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
      mcpServers: readonly AcpHttpMcpServer[];
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
    }

    if (candidateId && this.#openedOnClient === candidateId && this.#client === client) {
      return { sessionId: candidateId, configOptions: this.#configOptions };
    }

    if (candidateId) {
      try {
        const resumed = await client.resumeSession({
          sessionId: candidateId,
          cwd: this.#deps.cwd,
          mcpServers: params.mcpServers,
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
      }
    }

    const created = await client.newSession({
      cwd: this.#deps.cwd,
      mcpServers: params.mcpServers,
    });
    this.#conversationId = created.sessionId;
    this.#piSessionId = params.piSessionId;
    this.#openedOnClient = created.sessionId;
    this.#configOptions = created.configOptions;
    this.#turns = 0;
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
    const resolvedMode = resolveAcpExecutionMode(desired.executionMode, modeOption);
    if (modeOption && resolvedMode && modeOption.currentValue !== resolvedMode) {
      const updated = await client.setConfigOption(sessionId, modeOption.id, resolvedMode);
      if (updated.configOptions) currentOptions = updated.configOptions;
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
    const { agentName, bridgeServerName, mcpServers } = await this.#ensurePiToolsAgent(
      toolMode,
      params.modelId,
    );
    const client = await this.#ensureClient({
      agentName,
      mcpServers,
      env: params.env,
      signal: params.signal,
    });
    const session = await this.#openOrResumeSession(client, {
      piSessionId: params.sessionId,
      branchEntries: params.branchEntries,
      mcpServers,
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
    const executionMode = params.executionMode ?? this.#deps.getExecutionMode?.() ?? "default";
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
    const quotaTracker = new AgyQuotaTurnTracker();
    const workingTracker = new AgyWorkingStatusTracker();
    const promptAbort = new AbortController();
    let lastWorkingStatus: string | undefined;

    const setWorkingStatus = (next: string | undefined) => {
      if (next === lastWorkingStatus) return;
      lastWorkingStatus = next;
      this.#deps.onWorkingStatus?.(next);
    };

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
      noteBridgeCall: () => quotaTracker.noteToolActivity(),
      requestAbort: () => {
        clearTimeoutTimer();
        promptAbort.abort();
      },
    };
    this.#activeTurn = activeTurn;

    const cleanupActive = () => {
      clearTimeoutTimer();
      setWorkingStatus(undefined);
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
          if (!activity) return;

          if (activity.type === "context_usage") {
            // Pre-turn `usage_update` (usedTokens === 0) must not disarm the watchdog or quota guard.
            if (activity.usedTokens > 0) {
              noteFirstResponse();
              quotaTracker.noteUsageUpdate();
            }
            controller.pushActivity(activity);
            return;
          }

          noteFirstResponse();

          if (activity.type === "thinking_delta") {
            quotaTracker.noteThought();
            setWorkingStatus(workingTracker.onActivity(activity));
            controller.pushActivity(activity);
            return;
          }

          if (activity.type === "text_delta") {
            quotaTracker.noteText(activity.text);
            setWorkingStatus(workingTracker.onActivity(activity));
            controller.pushActivity(activity);
            return;
          }

          if (isBridgedMcpTool(activity, bridgeServerName)) {
            quotaTracker.noteToolActivity();
            return;
          }

          quotaTracker.noteToolActivity();
          setWorkingStatus(workingTracker.onActivity(activity));
          controller.pushActivity(activity);
        },
      })
      .then((promptResult) => {
        params.signal?.removeEventListener("abort", onExternalAbort);
        if (timedOut || controller.isSettled) return;
        cleanupActive();

        const normalizedUsage = normalizeAcpPromptUsage(promptResult.usage);
        const errorBanner = quotaTracker.detectTurnErrorBanner({
          totalTokens: normalizedUsage?.total_tokens,
          outputTokens: normalizedUsage?.output_tokens,
        });
        if (errorBanner?.kind === "quota") {
          this.#lastQuotaError = errorBanner.message;
          controller.fail(
            new AgyQuotaError(errorBanner.message, { resetsIn: errorBanner.resetsIn }),
          );
          return;
        }
        if (errorBanner?.kind === "api_error") {
          controller.fail(new Error(errorBanner.message));
          return;
        }

        if (promptResult.stopReason === "error") {
          controller.fail(new Error("Antigravity ACP turn failed with stopReason: error."));
          return;
        }

        // Materialize unfinished native tool steps before settling so Pi renders their final state.
        for (const [callId, tool] of acpTools.entries()) {
          if (tool.finishedEmitted || isBridgedMcpTool(tool, bridgeServerName)) continue;
          tool.finishedEmitted = true;
          controller.pushActivity({
            type: "tool_error",
            callId,
            title: tool.title,
            ...(tool.name ? { name: tool.name } : {}),
            ...(tool.kind ? { kind: tool.kind } : {}),
            ...(tool.locations ? { locations: tool.locations } : {}),
            ...(tool.rawInput ? { rawInput: tool.rawInput } : {}),
            message: INCOMPLETE_ACP_TOOL_MESSAGE,
          });
        }

        this.#turns += 1;
        this.#lastQuotaError = undefined;
        this.#deps.onPersistedState?.({
          version: 1,
          kind: "conversation",
          sessionId: params.sessionId,
          conversationId: sessionId,
          cwd: this.#deps.cwd,
          turns: this.#turns,
        });

        controller.finish({
          status: "completed",
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
          controller.fail(new AgyQuotaError(error.message));
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
}
