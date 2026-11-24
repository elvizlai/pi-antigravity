import { randomUUID } from "node:crypto";
import type { JsonObject, TranscriptContext } from "@earendil-works/pi-ai";
import { checkAgyBinary } from "../acp/binary.ts";
import {
  type AcpContentBlock,
  type AcpHttpMcpServer,
  type AcpInitializeResult,
  AcpRequestStallError,
  type AcpRpcCallOptions,
  type AcpSessionResult,
  AgyAcpClient,
  type AgyAcpClientOptions,
} from "../acp/client.ts";
import {
  type AgyActivity,
  type AgyToolView,
  acpUpdateToActivity,
  isBridgedMcpTool,
  normalizeAcpPromptUsage,
} from "../acp/events.ts";
import {
  type AgyAcpConfigOption,
  type AgyExecutionMode,
  extractAcpConfigOption,
  resolveAcpExecutionMode,
  resolveAcpModelCandidate,
  resolveAcpThoughtLevel,
} from "../acp/models.ts";
import { type AgyQuotaNotice, AgyQuotaTurnTracker, isAgyQuotaError } from "../acp/quota.ts";
import { isAbortError, makeAbortError } from "../acp/shared.ts";
import {
  createAuxiliaryToolsAgent,
  createPiToolsAgent,
  createToollessAgent,
  type PiToolsAgentHandle,
  resolveAgyChildEnv,
} from "../bridge/agent-files.ts";
import { AgyPiBridge, type BridgeCall, type BridgeToolDef } from "../bridge/server.ts";
import {
  BACKGROUND_ACP_TOOL_MESSAGE,
  type BridgedToolImageBatch,
  buildSeededSessionPrompt,
  buildToolImageContinuationPrompt,
  collectTurnToolResults,
  isContinuationAfterToolUse,
  prependSystemContext,
  splitToolResultImages,
} from "./prompt.ts";
import { AgyWorkingStatusTracker } from "./render.ts";
import type { AgyToolMode } from "./settings.ts";
import {
  type AgyConversationState,
  resolveAgyBranchState,
  resolveInTurnBranchState,
  type SessionEntryLike,
} from "./state.ts";
import { type AgyReplayStore, AgyTurnController } from "./turn.ts";

export const DEFAULT_FIRST_RESPONSE_TIMEOUT_MS = 120_000;
const DEFAULT_CANCEL_GRACE_MS = 5_000;
const DEFAULT_BRIDGED_TOOL_ACK_TIMEOUT_MS = 500;

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
 * stack extra 120s retries on top of `agy`'s own internal backoff window. Fast
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
  initialize(options?: AcpRpcCallOptions): Promise<AcpInitializeResult>;
  newSession(options: {
    cwd: string;
    mcpServers?: readonly AcpHttpMcpServer[];
    signal?: AbortSignal;
    timeoutMs?: number;
  }): Promise<AcpSessionResult>;
  resumeSession(options: {
    sessionId: string;
    cwd: string;
    mcpServers?: readonly AcpHttpMcpServer[];
    signal?: AbortSignal;
    timeoutMs?: number;
  }): Promise<AcpSessionResult>;
  setConfigOption(
    sessionId: string,
    configId: string,
    value: string,
    options?: AcpRpcCallOptions,
  ): Promise<{ configOptions?: AgyAcpConfigOption[] }>;
  cancelSession(sessionId: string): void;
  closeSession?(sessionId: string, options?: AcpRpcCallOptions): Promise<void>;
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
  isHostHindsightActive?: () => boolean;
  sandbox?: boolean;
  extraDirs?: readonly string[];
  binaryPath?: string;
  agentsDir?: string;
  firstResponseTimeoutMs?: number;
  cancelGraceMs?: number;
  bridgedToolAckTimeoutMs?: number;
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
  promptReplaced?: boolean;
  allowImages?: boolean;
  systemPromptAppend?: string;
  messages: TranscriptContext["messages"];
  sessionId: string;
  branchEntries: readonly SessionEntryLike[];
  allEntries?: readonly SessionEntryLike[];
  env?: Record<string, string>;
  timeoutMs?: number;
  signal?: AbortSignal;
  replayStore: AgyReplayStore;
}

export interface StartEphemeralTurnParams {
  modelId: string;
  effort?: string;
  prompt: AcpContentBlock[];
  systemPromptAppend?: string;
  env?: Record<string, string>;
  timeoutMs?: number;
  signal?: AbortSignal;
}

export interface StartAuxiliaryTurnParams {
  modelId: string;
  effort?: string;
  prompt: AcpContentBlock[];
  systemPromptAppend?: string;
  tools: readonly BridgeToolDef[];
  messages: TranscriptContext["messages"];
  env?: Record<string, string>;
  timeoutMs?: number;
  signal?: AbortSignal;
}

export interface TurnSessionHandle {
  controller: AgyTurnController;
  attachExisting: boolean;
  bridgeServerName?: string;
  signalTurnAbort: () => void;
  onConsumed?: () => void;
}

interface AuxiliaryTurn {
  controller: AgyTurnController;
  bridge: AgyPiBridge;
  requestAbort: () => void;
  cleanup: () => Promise<void>;
  promptDone: boolean;
}

interface ActiveTurn {
  controller: AgyTurnController;
  client: AgyAcpClientLike;
  sessionId: string;
  modelId: string;
  piSessionId: string;
  toolMode: AgyToolMode;
  bridgeServerName?: string;
  bridgeCallLabels: Map<string, string>;
  pendingToolImageBatches: BridgedToolImageBatch[];
  noteFirstResponse: () => void;
  isFirstResponseSeen?: () => boolean;
  noteBridgeCall: () => void;
  requestAbort: () => void;
  done: Promise<void>;
  waitForBridgedToolsAck: (timeoutMs: number, signal?: AbortSignal) => Promise<void>;
}

/**
 * Classify terminal ACP `stopReason` values that represent incomplete or failed
 * turns rather than normal completion. Keep messages free of retryable transport
 * keywords so deterministic protocol stops fail fast with 0 retries.
 */
function classifyAcpStopReasonFailure(stopReason: string): string | undefined {
  switch (stopReason) {
    case "end_turn":
    case "max_tokens":
    case "cancelled":
      return undefined;
    case "max_turn_requests":
      return "Antigravity ACP turn stopped early: agent reached max_turn_requests limit.";
    case "refusal":
      return "Antigravity ACP model refused the request (stopReason: refusal).";
    case "error":
      return "Antigravity ACP turn failed with stopReason: error.";
    default:
      return `Antigravity ACP turn ended with unexpected stopReason: ${stopReason}.`;
  }
}

function stableEnvFingerprint(env: Record<string, string> | undefined): string {
  if (!env) return "";
  const entries = Object.entries(env).sort(([a], [b]) => a.localeCompare(b));
  return JSON.stringify(entries);
}

function extractBridgeCallLabel(args: JsonObject): string | undefined {
  for (const key of ["path", "file_path", "filePath", "url", "uri"]) {
    const value = args[key];
    if (typeof value === "string" && value.trim().length > 0) {
      return value.trim();
    }
  }
  return undefined;
}

function isFatalSessionSetupError(
  err: unknown,
  signal: AbortSignal | undefined,
  client: AgyAcpClientLike,
): boolean {
  return (
    Boolean(signal?.aborted) ||
    isAbortError(err) ||
    err instanceof AcpRequestStallError ||
    !client.isAlive
  );
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
  #ephemeralAgentHandle: PiToolsAgentHandle | undefined;
  #agentKey: string | undefined;
  #conversationId: string | undefined;
  #piSessionId: string | undefined;
  #openedOnClient: string | undefined;
  #configOptions: AgyAcpConfigOption[] | undefined;
  #turns = 0;
  #failedResumeIds = new Set<string>();
  #skipHistorySeed = false;
  #lastQuotaError: string | undefined;
  #activeTurn: ActiveTurn | undefined;
  #auxTurns = new Set<AuxiliaryTurn>();
  #inTurnContinuation:
    | { piSessionId: string; conversationId: string; owner?: ActiveTurn }
    | undefined;

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
    this.#inTurnContinuation = undefined;
    this.#conversationId = undefined;
    this.#piSessionId = undefined;
    this.#openedOnClient = undefined;
    this.#configOptions = undefined;
    this.#turns = 0;
    this.#failedResumeIds.clear();
    this.#skipHistorySeed = false;
    this.#lastQuotaError = undefined;
    await this.dispose();
  }

  /**
   * Drop the active ACP conversation binding (e.g. after `/compact`) while
   * keeping the initialized `agy acp` worker process warm for the next turn.
   */
  async invalidateConversation(): Promise<void> {
    this.#inTurnContinuation = undefined;
    const active = this.#activeTurn;
    if (active) {
      await this.#interruptActiveTurn(active);
    }
    if (this.#openedOnClient && this.#client?.isAlive) {
      await this.#client.closeSession?.(this.#openedOnClient).catch(() => {});
    }
    this.#conversationId = undefined;
    this.#openedOnClient = undefined;
    this.#configOptions = undefined;
    this.#turns = 0;
    this.#failedResumeIds.clear();
    this.#skipHistorySeed = false;
  }

  async dispose(): Promise<void> {
    this.#inTurnContinuation = undefined;
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
    const auxList = [...this.#auxTurns];
    this.#auxTurns.clear();
    for (const aux of auxList) {
      aux.requestAbort();
    }
    await Promise.all([
      this.#disposeClient(),
      this.#removeAgentFile(),
      ...auxList.map((aux) => aux.cleanup().catch(() => {})),
    ]);
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
    const timeoutMs = this.#deps.firstResponseTimeoutMs ?? DEFAULT_FIRST_RESPONSE_TIMEOUT_MS;
    const timer =
      timeoutMs > 0
        ? setTimeout(() => {
            abort.abort();
          }, timeoutMs)
        : undefined;
    timer?.unref?.();
    const task = fn(abort.signal).finally(() => {
      if (timer) clearTimeout(timer);
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

  clearSettledTurn(): void {
    if (this.#activeTurn?.controller.isSettled) {
      this.#activeTurn = undefined;
    }
  }

  /** Pre-start and initialize `agy acp` + bridge agent without creating an empty ACP conversation. */
  async warmupClient(params?: {
    env?: Record<string, string>;
    signal?: AbortSignal;
  }): Promise<void> {
    this.clearSettledTurn();
    if (this.#activeTurn) return;
    if (this.#activeWarmup) {
      await this.#activeWarmup.catch(() => {});
    }
    if (params?.signal?.aborted || this.#activeTurn) return;
    await this.#runWarmup(params?.signal, async (signal) => {
      if (signal.aborted || this.#activeTurn) return;
      const toolMode = this.#deps.getToolMode?.() ?? "pi";
      this.#deps.bridge?.refreshTools();
      const { agentName, mcpServers } = await this.#ensurePiToolsAgent(toolMode);
      if (signal.aborted || this.#activeTurn) return;
      await this.#ensureClient({
        agentName,
        mcpServers,
        env: params?.env,
        signal,
        timeoutMs: this.#deps.firstResponseTimeoutMs ?? DEFAULT_FIRST_RESPONSE_TIMEOUT_MS,
      });
    });
  }

  /** Pre-open/resume and configure the active ACP session right before `streamSimple` begins timing. */
  async warmupSession(params: {
    modelId: string;
    effort?: string;
    executionMode?: AgyExecutionMode;
    sessionId: string;
    branchEntries: readonly SessionEntryLike[];
    allEntries?: readonly SessionEntryLike[];
    env?: Record<string, string>;
    signal?: AbortSignal;
  }): Promise<void> {
    this.clearSettledTurn();
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
        allEntries: params.allEntries,
        env: params.env,
        signal,
        timeoutMs: this.#deps.firstResponseTimeoutMs ?? DEFAULT_FIRST_RESPONSE_TIMEOUT_MS,
      });
    });
  }

  #handleBridgeCall(call: BridgeCall): boolean {
    const active = this.#activeTurn;
    if (!active || active.controller.isSettled) return false;
    active.noteFirstResponse();
    active.noteBridgeCall();
    const label = extractBridgeCallLabel(call.args);
    if (label) {
      active.bridgeCallLabels.set(call.id, label);
    }
    active.controller.pushActivity({
      type: "bridge_call",
      callId: call.id,
      name: call.tool,
      args: call.args,
    });
    return true;
  }

  async #disposeClient(graceMs = 400): Promise<void> {
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
    const ephemeralHandle = this.#ephemeralAgentHandle;
    this.#agentHandle = undefined;
    this.#ephemeralAgentHandle = undefined;
    this.#agentKey = undefined;
    await Promise.all([
      handle?.remove().catch(() => {}),
      ephemeralHandle?.remove().catch(() => {}),
    ]);
  }

  async #ensurePiToolsAgent(
    toolMode: AgyToolMode,
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

    const desiredKey = bridge.serverName;
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
      { agentsDir: this.#deps.agentsDir },
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
    timeoutMs?: number;
  }): Promise<AgyAcpClientLike> {
    if (this.#pendingDispose) {
      await this.#pendingDispose.catch(() => {});
    }

    const effectiveEnv = resolveAgyChildEnv({
      hostHindsightActive: this.#deps.isHostHindsightActive?.(),
      env: params.env,
    });

    const fingerprint = JSON.stringify({
      agentName: params.agentName ?? "",
      bridgeUrl: params.mcpServers?.[0]?.url ?? "",
      bridgeCatalogRevision: params.agentName ? (this.#deps.bridge?.catalogRevision ?? 0) : 0,
      sandbox: Boolean(this.#deps.sandbox),
      extraDirs: this.#deps.extraDirs ?? [],
      env: stableEnvFingerprint(effectiveEnv),
    });

    if (this.#client?.isAlive && this.#clientFingerprint === fingerprint) {
      return this.#client;
    }

    await this.#disposeClient();

    const verified = this.#deps.checkBinary
      ? await this.#deps.checkBinary(this.#deps.binaryPath, params.signal)
      : await checkAgyBinary({ binaryPath: this.#deps.binaryPath, signal: params.signal });

    if (params.signal?.aborted) {
      throw makeAbortError();
    }

    const clientOptions: AgyAcpClientOptions = {
      binary: verified.binary,
      cwd: this.#deps.cwd,
      agentName: params.agentName,
      extraDirs: this.#deps.extraDirs,
      sandbox: this.#deps.sandbox,
      env: effectiveEnv,
    };

    const client = this.#deps.createClient
      ? this.#deps.createClient(clientOptions)
      : new AgyAcpClient(clientOptions);
    this.#client = client;
    this.#clientFingerprint = fingerprint;
    try {
      await client.initialize({ signal: params.signal, timeoutMs: params.timeoutMs });
      return client;
    } catch (err) {
      await this.#disposeClient(50);
      throw err;
    }
  }

  async #openOrResumeSession(
    client: AgyAcpClientLike,
    params: {
      piSessionId: string;
      branchEntries: readonly SessionEntryLike[];
      allEntries?: readonly SessionEntryLike[];
      mcpServers: readonly AcpHttpMcpServer[];
      signal?: AbortSignal;
      timeoutMs?: number;
    },
  ): Promise<{ sessionId: string; configOptions?: AgyAcpConfigOption[] }> {
    const closeOpenedOnClient = () => {
      if (this.#openedOnClient && this.#client === client) {
        void client.closeSession?.(this.#openedOnClient)?.catch(() => {});
      }
      this.#openedOnClient = undefined;
      this.#configOptions = undefined;
    };

    const tryResume = async (
      candidateId: string,
      markerTurns?: number,
    ): Promise<{ sessionId: string; configOptions?: AgyAcpConfigOption[] } | undefined> => {
      try {
        const resumed = await client.resumeSession({
          sessionId: candidateId,
          cwd: this.#deps.cwd,
          mcpServers: params.mcpServers,
          signal: params.signal,
          timeoutMs: params.timeoutMs,
        });
        const samePrevious = this.#conversationId === candidateId;
        this.#conversationId = resumed.sessionId;
        this.#piSessionId = params.piSessionId;
        this.#openedOnClient = resumed.sessionId;
        this.#configOptions = resumed.configOptions;
        if (markerTurns !== undefined) {
          this.#turns = samePrevious ? Math.max(this.#turns, markerTurns) : markerTurns;
        }
        return { sessionId: resumed.sessionId, configOptions: resumed.configOptions };
      } catch (err) {
        if (isFatalSessionSetupError(err, params.signal, client)) {
          throw err;
        }
        return undefined;
      }
    };

    if (this.#piSessionId && this.#piSessionId !== params.piSessionId) {
      this.#inTurnContinuation = undefined;
      closeOpenedOnClient();
      this.#conversationId = undefined;
      this.#turns = 0;
      this.#skipHistorySeed = false;
      this.#failedResumeIds.clear();
    }

    const inTurn =
      this.#inTurnContinuation?.piSessionId === params.piSessionId
        ? this.#inTurnContinuation
        : undefined;
    if (inTurn) {
      const inTurnBranch = resolveInTurnBranchState(
        params.piSessionId,
        this.#deps.cwd,
        params.branchEntries,
        params.allEntries,
      );
      if (inTurnBranch === "ok") {
        this.#skipHistorySeed = false;
        const convId = inTurn.conversationId;
        if (
          this.#openedOnClient === convId &&
          this.#client === client &&
          this.#piSessionId === params.piSessionId
        ) {
          this.#turns = Math.max(1, this.#turns);
          return { sessionId: convId, configOptions: this.#configOptions };
        }
        if (!this.#failedResumeIds.has(convId)) {
          if (this.#openedOnClient && this.#openedOnClient !== convId) {
            closeOpenedOnClient();
          }
          const resumed = await tryResume(convId, Math.max(1, this.#turns));
          if (resumed) return resumed;
          this.#failedResumeIds.add(convId);
        }
        this.#inTurnContinuation = undefined;
        closeOpenedOnClient();
        this.#conversationId = undefined;
        this.#turns = 0;
        const created = await client.newSession({
          cwd: this.#deps.cwd,
          mcpServers: params.mcpServers,
          signal: params.signal,
          timeoutMs: params.timeoutMs,
        });
        this.#conversationId = created.sessionId;
        this.#piSessionId = params.piSessionId;
        this.#openedOnClient = created.sessionId;
        this.#configOptions = created.configOptions;
        this.#turns = 0;
        return { sessionId: created.sessionId, configOptions: created.configOptions };
      }
      this.#inTurnContinuation = undefined;
    }

    const branchState = resolveAgyBranchState(
      params.piSessionId,
      this.#deps.cwd,
      params.branchEntries,
      params.allEntries,
    );

    switch (branchState.status) {
      case "active": {
        this.#skipHistorySeed = false;
        const markerId = branchState.state.conversationId;
        if (this.#failedResumeIds.has(markerId)) {
          if (this.#openedOnClient && this.#client === client) {
            return { sessionId: this.#openedOnClient, configOptions: this.#configOptions };
          }
          if (this.#conversationId && this.#conversationId !== markerId) {
            const resumed = await tryResume(this.#conversationId);
            if (resumed) return resumed;
            this.#conversationId = undefined;
            this.#turns = 0;
          }
          break;
        }

        if (this.#openedOnClient === markerId && this.#client === client) {
          this.#turns = Math.max(this.#turns, branchState.state.turns);
          return { sessionId: markerId, configOptions: this.#configOptions };
        }
        if (this.#openedOnClient && this.#openedOnClient !== markerId) {
          closeOpenedOnClient();
        }
        const resumed = await tryResume(markerId, branchState.state.turns);
        if (resumed) return resumed;
        this.#failedResumeIds.add(markerId);
        this.#conversationId = undefined;
        this.#turns = 0;
        break;
      }

      case "invalidated":
      case "reset": {
        this.#skipHistorySeed = branchState.status === "reset";
        if (
          this.#openedOnClient &&
          this.#client === client &&
          this.#piSessionId === params.piSessionId &&
          this.#turns === 0
        ) {
          return { sessionId: this.#openedOnClient, configOptions: this.#configOptions };
        }
        closeOpenedOnClient();
        this.#conversationId = undefined;
        this.#turns = 0;
        break;
      }

      case "none": {
        this.#skipHistorySeed = false;
        if (
          this.#openedOnClient &&
          this.#client === client &&
          this.#piSessionId === params.piSessionId
        ) {
          return { sessionId: this.#openedOnClient, configOptions: this.#configOptions };
        }
        if (this.#conversationId && this.#piSessionId === params.piSessionId) {
          const resumed = await tryResume(this.#conversationId);
          if (resumed) return resumed;
          this.#conversationId = undefined;
          this.#turns = 0;
        }
        break;
      }
    }

    const created = await client.newSession({
      cwd: this.#deps.cwd,
      mcpServers: params.mcpServers,
      signal: params.signal,
      timeoutMs: params.timeoutMs,
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
  async #syncSessionConfigTo(
    client: AgyAcpClientLike,
    sessionId: string,
    desired: { modelId: string; effort?: string; executionMode: AgyExecutionMode },
    initialOptions?: readonly AgyAcpConfigOption[],
    rpcOptions?: AcpRpcCallOptions,
  ): Promise<AgyAcpConfigOption[] | undefined> {
    let currentOptions = initialOptions;

    const modeOption = extractAcpConfigOption(currentOptions, "mode");
    const resolvedMode = resolveAcpExecutionMode(desired.executionMode, modeOption);
    if (modeOption && resolvedMode && modeOption.currentValue !== resolvedMode) {
      const updated = await client.setConfigOption(
        sessionId,
        modeOption.id,
        resolvedMode,
        rpcOptions,
      );
      if (updated.configOptions) currentOptions = updated.configOptions;
    }

    const modelOption = extractAcpConfigOption(currentOptions, "model");
    const resolvedModel = resolveAcpModelCandidate(desired.modelId, modelOption);
    if (modelOption && modelOption.currentValue !== resolvedModel.modelId) {
      const updated = await client.setConfigOption(
        sessionId,
        modelOption.id,
        resolvedModel.modelId,
        rpcOptions,
      );
      if (updated.configOptions) currentOptions = updated.configOptions;
    }

    const thoughtOption = extractAcpConfigOption(currentOptions, "thought_level");
    const resolvedThought = resolveAcpThoughtLevel(
      desired.effort ?? resolvedModel.impliedEffort,
      thoughtOption,
    );
    if (thoughtOption && resolvedThought && thoughtOption.currentValue !== resolvedThought) {
      const updated = await client.setConfigOption(
        sessionId,
        thoughtOption.id,
        resolvedThought,
        rpcOptions,
      );
      if (updated.configOptions) currentOptions = updated.configOptions;
    }

    return currentOptions ? [...currentOptions] : undefined;
  }

  async #syncSessionConfig(
    client: AgyAcpClientLike,
    sessionId: string,
    desired: { modelId: string; effort?: string; executionMode: AgyExecutionMode },
    initialOptions?: readonly AgyAcpConfigOption[],
    rpcOptions?: AcpRpcCallOptions,
  ): Promise<void> {
    this.#configOptions = await this.#syncSessionConfigTo(
      client,
      sessionId,
      desired,
      initialOptions,
      rpcOptions,
    );
  }

  async #prepareConfiguredSession(params: {
    modelId: string;
    effort?: string;
    executionMode: AgyExecutionMode;
    sessionId: string;
    branchEntries: readonly SessionEntryLike[];
    allEntries?: readonly SessionEntryLike[];
    env?: Record<string, string>;
    signal?: AbortSignal;
    timeoutMs?: number;
  }): Promise<{
    client: AgyAcpClientLike;
    sessionId: string;
    toolMode: AgyToolMode;
    bridgeServerName?: string;
  }> {
    const deadlineMs =
      typeof params.timeoutMs === "number" && params.timeoutMs > 0
        ? Date.now() + params.timeoutMs
        : undefined;
    const remainingMs = () =>
      deadlineMs !== undefined ? Math.max(1, deadlineMs - Date.now()) : undefined;

    const toolMode = this.#deps.getToolMode?.() ?? "pi";
    this.#deps.bridge?.refreshTools();
    const { agentName, bridgeServerName, mcpServers } = await this.#ensurePiToolsAgent(toolMode);
    const client = await this.#ensureClient({
      agentName,
      mcpServers,
      env: params.env,
      signal: params.signal,
      timeoutMs: remainingMs(),
    });
    try {
      const session = await this.#openOrResumeSession(client, {
        piSessionId: params.sessionId,
        branchEntries: params.branchEntries,
        allEntries: params.allEntries,
        mcpServers,
        signal: params.signal,
        timeoutMs: remainingMs(),
      });
      await this.#syncSessionConfig(
        client,
        session.sessionId,
        { modelId: params.modelId, effort: params.effort, executionMode: params.executionMode },
        session.configOptions,
        { signal: params.signal, timeoutMs: remainingMs() },
      );
      return { client, sessionId: session.sessionId, toolMode, bridgeServerName };
    } catch (err) {
      if (isFatalSessionSetupError(err, params.signal, client)) {
        await this.#disposeClient(50);
      }
      throw err;
    }
  }

  async #awaitActiveWarmup(signal?: AbortSignal): Promise<void> {
    if (!this.#activeWarmup) return;
    const onWarmupAbort = () => this.#warmupAbort?.abort();
    signal?.addEventListener("abort", onWarmupAbort, { once: true });
    try {
      await this.#activeWarmup.catch(() => {});
    } finally {
      signal?.removeEventListener("abort", onWarmupAbort);
    }
    if (signal?.aborted) {
      throw makeAbortError("Antigravity turn aborted.");
    }
  }

  async #openTurnSession(
    params: StartTurnParams,
    effectiveTimeoutMs: number,
  ): Promise<{
    client: AgyAcpClientLike;
    sessionId: string;
    toolMode: AgyToolMode;
    bridgeServerName?: string;
    remainingWatchdogMs: number;
  }> {
    const executionMode = params.executionMode ?? this.#deps.getExecutionMode?.() ?? "default";
    const setupStartedAt = Date.now();
    try {
      const configured = await this.#prepareConfiguredSession({
        modelId: params.modelId,
        effort: params.effort,
        executionMode,
        sessionId: params.sessionId,
        branchEntries: params.branchEntries,
        allEntries: params.allEntries,
        env: params.env,
        signal: params.signal,
        timeoutMs: effectiveTimeoutMs,
      });
      return {
        ...configured,
        remainingWatchdogMs: Math.max(
          1,
          effectiveTimeoutMs - Math.max(0, Date.now() - setupStartedAt),
        ),
      };
    } catch (err) {
      if (params.signal?.aborted || isAbortError(err)) {
        throw makeAbortError("Antigravity turn aborted.");
      }
      if (err instanceof AcpRequestStallError) {
        if (this.#turns === 0) this.#conversationId = undefined;
        throw new AgyFirstResponseTimeoutError(effectiveTimeoutMs);
      }
      if (err instanceof Error && isAgyQuotaError(err.message)) {
        this.#lastQuotaError = err.message;
        throw new AgyQuotaError(err.message);
      }
      throw err;
    }
  }

  async #drivePromptTurn(ctx: {
    client: AgyAcpClientLike;
    sessionId: string;
    piSessionId: string;
    bridgeServerName?: string;
    outboundPrompt: AcpContentBlock[];
    activeTurn: ActiveTurn;
    controller: AgyTurnController;
    acpTools: Map<string, AgyToolView>;
    quotaTracker: AgyQuotaTurnTracker;
    workingTracker: AgyWorkingStatusTracker;
    promptAbort: AbortController;
    externalSignal?: AbortSignal;
    onExternalAbort: () => void;
    noteFirstResponse: () => void;
    isTimedOut: () => boolean;
    cleanupActive: (dropTurn?: boolean) => void;
    dropIfConsumed: () => void;
    setWorkingStatus: (next: string | undefined) => void;
    notifyAckWaiters: () => void;
  }): Promise<void> {
    const {
      client,
      sessionId,
      piSessionId,
      bridgeServerName,
      outboundPrompt,
      controller,
      acpTools,
      quotaTracker,
      workingTracker,
      promptAbort,
      externalSignal,
      onExternalAbort,
      noteFirstResponse,
      isTimedOut,
      cleanupActive,
      dropIfConsumed,
      setWorkingStatus,
      notifyAckWaiters,
    } = ctx;

    let turnRecorded = false;
    const recordTurnState = (requireFirstResponse: boolean) => {
      if (turnRecorded) return;
      if (requireFirstResponse && !ctx.activeTurn.isFirstResponseSeen?.()) return;
      turnRecorded = true;
      this.#turns += 1;
      this.#deps.onPersistedState?.({
        version: 1,
        kind: "conversation",
        sessionId: piSessionId,
        conversationId: sessionId,
        cwd: this.#deps.cwd,
        turns: this.#turns,
      });
    };

    const clearOwnedInTurnContinuation = () => {
      if (this.#inTurnContinuation?.owner === ctx.activeTurn) {
        this.#inTurnContinuation = undefined;
      }
    };

    const flushBackgroundNativeTools = () => {
      for (const [callId, tool] of acpTools.entries()) {
        if (
          tool.status !== "in_progress" ||
          tool.finishedEmitted ||
          tool.deferredEmitted ||
          isBridgedMcpTool(tool, bridgeServerName)
        ) {
          continue;
        }
        tool.deferredEmitted = true;
        const doneActivity: AgyActivity = {
          type: "tool_done",
          callId,
          title: tool.title,
          ...(tool.name ? { name: tool.name } : {}),
          ...(tool.kind ? { kind: tool.kind } : {}),
          ...(tool.locations ? { locations: tool.locations } : {}),
          ...(tool.rawInput ? { rawInput: tool.rawInput } : {}),
          output: tool.output ?? BACKGROUND_ACP_TOOL_MESSAGE,
        };
        setWorkingStatus(workingTracker.onActivity(doneActivity));
        controller.pushActivity(doneActivity);
      }
    };

    try {
      const promptResult = await client.prompt(sessionId, outboundPrompt, {
        signal: promptAbort.signal,
        onUpdate: (update) => {
          if (isTimedOut() || controller.isSettled) return;
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
            flushBackgroundNativeTools();
            quotaTracker.noteText(activity.text);
            setWorkingStatus(workingTracker.onActivity(activity));
            controller.pushActivity(activity);
            return;
          }

          if (isBridgedMcpTool(activity, bridgeServerName)) {
            quotaTracker.noteToolActivity();
            notifyAckWaiters();
            return;
          }

          quotaTracker.noteToolActivity();
          setWorkingStatus(workingTracker.onActivity(activity));
          controller.pushActivity(activity);
          notifyAckWaiters();
        },
      });

      externalSignal?.removeEventListener("abort", onExternalAbort);
      if (isTimedOut() || controller.isSettled) return;
      cleanupActive(false);

      const normalizedUsage = normalizeAcpPromptUsage(promptResult.usage);
      const errorBanner = quotaTracker.detectTurnErrorBanner({
        totalTokens: normalizedUsage?.total_tokens,
        outputTokens: normalizedUsage?.output_tokens,
      });
      if (errorBanner?.kind === "quota") {
        this.#lastQuotaError = errorBanner.message;
        controller.fail(new AgyQuotaError(errorBanner.message, { resetsIn: errorBanner.resetsIn }));
        dropIfConsumed();
        return;
      }
      if (errorBanner?.kind === "api_error") {
        controller.fail(new Error(errorBanner.message));
        dropIfConsumed();
        return;
      }

      if (promptResult.stopReason === "cancelled") {
        clearOwnedInTurnContinuation();
        recordTurnState(true);
        controller.finish({ status: "aborted", stopReason: "aborted" });
        dropIfConsumed();
        return;
      }

      const stopError = classifyAcpStopReasonFailure(promptResult.stopReason);
      if (stopError) {
        controller.fail(new Error(stopError));
        dropIfConsumed();
        return;
      }

      // Materialize background native tool steps before settling so Pi renders their non-error state.
      flushBackgroundNativeTools();

      this.#lastQuotaError = undefined;
      clearOwnedInTurnContinuation();
      recordTurnState(false);

      controller.finish({
        status: "completed",
        stopReason: promptResult.stopReason,
        usage: normalizedUsage,
      });
      dropIfConsumed();
    } catch (err) {
      externalSignal?.removeEventListener("abort", onExternalAbort);
      if (isTimedOut() || controller.isSettled) return;
      cleanupActive(false);

      if (promptAbort.signal.aborted || isAbortError(err)) {
        clearOwnedInTurnContinuation();
        recordTurnState(true);
        controller.finish({ status: "aborted", stopReason: "aborted" });
        dropIfConsumed();
        return;
      }

      const error = err instanceof Error ? err : new Error(String(err));
      if (isAgyQuotaError(error.message)) {
        this.#lastQuotaError = error.message;
        controller.fail(new AgyQuotaError(error.message));
        dropIfConsumed();
        return;
      }

      if (!ctx.activeTurn.isFirstResponseSeen?.() || !client.isAlive) {
        if (this.#turns === 0) this.#conversationId = undefined;
        void this.#disposeClient(50);
      }
      controller.fail(error);
      dropIfConsumed();
    } finally {
      notifyAckWaiters();
    }
  }

  async #interruptActiveTurn(
    active: ActiveTurn,
    options?: {
      messages?: TranscriptContext["messages"];
      samePiSession?: boolean;
      allowImages?: boolean;
      outImageBatches?: BridgedToolImageBatch[];
      signal?: AbortSignal;
    },
  ): Promise<void> {
    if (this.#activeTurn === active) {
      this.#activeTurn = undefined;
    }

    const bridge = this.#deps.bridge;
    const resolvedCount =
      options?.samePiSession && options.messages
        ? this.resolveBridgeResultsFromMessages(options.messages, {
            allowImages: options.allowImages,
            labels: active.bridgeCallLabels,
            outImageBatches: options.outImageBatches,
          })
        : 0;
    const cancelledCount =
      bridge?.cancelPendingCalls("antigravity: turn interrupted before tool call ran.") ?? 0;

    if ((resolvedCount > 0 || cancelledCount > 0) && bridge) {
      await bridge.flushResponses();
    }

    if (resolvedCount > 0 && !active.controller.isSettled) {
      const ackTimeoutMs =
        this.#deps.bridgedToolAckTimeoutMs ?? DEFAULT_BRIDGED_TOOL_ACK_TIMEOUT_MS;
      await active.waitForBridgedToolsAck(ackTimeoutMs, options?.signal);
    }

    if (!active.controller.isSettled) {
      active.requestAbort();
    }

    if (active.controller.isSettled && !active.client.isAlive) {
      return;
    }

    const graceMs = this.#deps.cancelGraceMs ?? DEFAULT_CANCEL_GRACE_MS;
    let graceTimer: ReturnType<typeof setTimeout> | undefined;
    let onAbort: (() => void) | undefined;
    let timedOutOrAborted = false;

    const racePromises: Array<Promise<"done" | "stop">> = [active.done.then(() => "done" as const)];
    if (graceMs > 0) {
      racePromises.push(
        new Promise<"stop">((resolve) => {
          graceTimer = setTimeout(() => {
            timedOutOrAborted = true;
            resolve("stop");
          }, graceMs);
          graceTimer.unref?.();
        }),
      );
    }
    if (options?.signal) {
      if (options.signal.aborted) {
        timedOutOrAborted = true;
      } else {
        racePromises.push(
          new Promise<"stop">((resolve) => {
            onAbort = () => {
              timedOutOrAborted = true;
              resolve("stop");
            };
            options.signal?.addEventListener("abort", onAbort, { once: true });
          }),
        );
      }
    }

    try {
      const outcome = timedOutOrAborted ? "stop" : await Promise.race(racePromises);
      if (outcome === "stop" || timedOutOrAborted) {
        await this.#disposeClient(50);
      }
    } finally {
      if (graceTimer) clearTimeout(graceTimer);
      if (onAbort) options?.signal?.removeEventListener("abort", onAbort);
    }
  }

  /**
   * Settle any active turn and pending bridge calls when Pi's agent loop finishes
   * (`agent_end`), e.g. when a tool batch terminates early or aborts mid-turn.
   */
  async settleAfterAgentEnd(): Promise<void> {
    this.#inTurnContinuation = undefined;
    const active = this.#activeTurn;
    if (!active) {
      const cancelled = this.#deps.bridge?.cancelPendingCalls() ?? 0;
      if (cancelled > 0) {
        await this.#deps.bridge?.flushResponses();
      }
      return;
    }
    if (active.controller.isSettled) {
      this.#activeTurn = undefined;
      const cancelled = this.#deps.bridge?.cancelPendingCalls() ?? 0;
      if (cancelled > 0) {
        await this.#deps.bridge?.flushResponses();
      }
      return;
    }
    await this.#interruptActiveTurn(active);
  }

  async startOrContinueTurn(params: StartTurnParams): Promise<TurnSessionHandle> {
    const active = this.#activeTurn;
    const allowImages = params.allowImages ?? true;
    let continuationImageBatches: BridgedToolImageBatch[] = [];

    // 1. Continuation wave: attach to an active or settled-but-unconsumed turn ONLY when trailing messages are tool results.
    if (active) {
      const samePiSession = active.piSessionId === params.sessionId;
      const canContinue =
        samePiSession &&
        !active.controller.isConsumed &&
        isContinuationAfterToolUse(params.messages);
      if (canContinue) {
        this.resolveBridgeResultsFromMessages(params.messages, {
          allowImages,
          labels: active.bridgeCallLabels,
          outImageBatches: active.pendingToolImageBatches,
        });
        if (active.pendingToolImageBatches.length === 0 || active.controller.isSettled) {
          return {
            controller: active.controller,
            attachExisting: true,
            bridgeServerName: active.bridgeServerName,
            signalTurnAbort: () => active.requestAbort(),
          };
        }

        const bridge = this.#deps.bridge;
        if ((bridge?.pendingCount ?? 0) > 0) {
          return {
            controller: active.controller,
            attachExisting: true,
            bridgeServerName: active.bridgeServerName,
            signalTurnAbort: () => active.requestAbort(),
          };
        }

        await bridge?.flushResponses();
        if (!active.controller.isSettled) {
          const ackTimeoutMs =
            this.#deps.bridgedToolAckTimeoutMs ?? DEFAULT_BRIDGED_TOOL_ACK_TIMEOUT_MS;
          await active.waitForBridgedToolsAck(ackTimeoutMs, params.signal);
        }

        if ((bridge?.pendingCount ?? 0) > 0 && !active.controller.isSettled) {
          return {
            controller: active.controller,
            attachExisting: true,
            bridgeServerName: active.bridgeServerName,
            signalTurnAbort: () => active.requestAbort(),
          };
        }

        continuationImageBatches = active.pendingToolImageBatches.splice(0);
        const continueConversationId =
          this.#turns > 0 || active.isFirstResponseSeen?.() ? active.sessionId : undefined;
        await this.#interruptActiveTurn(active, {
          samePiSession: true,
          signal: params.signal,
        });
        this.#inTurnContinuation = continueConversationId
          ? { piSessionId: params.sessionId, conversationId: continueConversationId }
          : undefined;
      } else {
        const steeredBatches = samePiSession ? [...active.pendingToolImageBatches] : [];
        active.pendingToolImageBatches.length = 0;
        const continueConversationId =
          samePiSession && (this.#turns > 0 || active.isFirstResponseSeen?.())
            ? active.sessionId
            : undefined;
        await this.#interruptActiveTurn(active, {
          messages: params.messages,
          samePiSession,
          allowImages,
          outImageBatches: steeredBatches,
          signal: params.signal,
        });
        this.#inTurnContinuation = continueConversationId
          ? { piSessionId: params.sessionId, conversationId: continueConversationId }
          : undefined;
        if (samePiSession && steeredBatches.length > 0) {
          continuationImageBatches = steeredBatches;
        }
      }
    }

    if (params.signal?.aborted) {
      throw makeAbortError("Antigravity turn aborted.");
    }

    await this.#awaitActiveWarmup(params.signal);

    const baseTimeoutMs = this.#deps.firstResponseTimeoutMs ?? DEFAULT_FIRST_RESPONSE_TIMEOUT_MS;
    const effectiveTimeoutMs =
      typeof params.timeoutMs === "number" && params.timeoutMs > 0
        ? Math.min(params.timeoutMs, baseTimeoutMs)
        : baseTimeoutMs;

    // 2. Fresh ACP prompt turn.
    const { client, sessionId, toolMode, bridgeServerName, remainingWatchdogMs } =
      await this.#openTurnSession(params, effectiveTimeoutMs);

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

    const ackWaiters = new Set<() => void>();
    const notifyAckWaiters = () => {
      for (const wake of ackWaiters) {
        wake();
      }
    };
    const hasInProgressBridgedTools = () => {
      for (const tool of acpTools.values()) {
        if (
          tool.status === "in_progress" &&
          !tool.finishedEmitted &&
          isBridgedMcpTool(tool, bridgeServerName)
        ) {
          return true;
        }
      }
      return false;
    };
    const waitForBridgedToolsAck = (timeoutMs: number, signal?: AbortSignal): Promise<void> => {
      if (
        controller.isSettled ||
        !hasInProgressBridgedTools() ||
        signal?.aborted ||
        timeoutMs <= 0
      ) {
        return Promise.resolve();
      }
      return new Promise<void>((resolve) => {
        let timer: ReturnType<typeof setTimeout> | undefined;
        const finish = () => {
          ackWaiters.delete(onUpdate);
          if (timer) clearTimeout(timer);
          signal?.removeEventListener("abort", finish);
          resolve();
        };
        const onUpdate = () => {
          if (controller.isSettled || !hasInProgressBridgedTools()) {
            finish();
          }
        };
        ackWaiters.add(onUpdate);
        timer = setTimeout(finish, timeoutMs);
        timer.unref?.();
        signal?.addEventListener("abort", finish, { once: true });
      });
    };

    const activeTurn: ActiveTurn = {
      controller,
      client,
      sessionId,
      modelId: params.modelId,
      piSessionId: params.sessionId,
      toolMode,
      bridgeServerName,
      bridgeCallLabels: new Map(),
      pendingToolImageBatches: [],
      noteFirstResponse,
      isFirstResponseSeen: () => sawFirstResponse,
      noteBridgeCall: () => quotaTracker.noteToolActivity(),
      requestAbort: () => {
        clearTimeoutTimer();
        promptAbort.abort();
      },
      done: Promise.resolve(),
      waitForBridgedToolsAck,
    };
    this.#activeTurn = activeTurn;
    if (this.#inTurnContinuation?.piSessionId === params.sessionId) {
      this.#inTurnContinuation = {
        piSessionId: params.sessionId,
        conversationId: sessionId,
        owner: activeTurn,
      };
    }

    const cleanupActive = (dropTurn = false) => {
      clearTimeoutTimer();
      setWorkingStatus(undefined);
      if (this.#activeTurn === activeTurn && (dropTurn || controller.isConsumed)) {
        this.#activeTurn = undefined;
      }
    };

    const dropIfConsumed = () => {
      if (this.#activeTurn === activeTurn && controller.isConsumed) {
        this.#activeTurn = undefined;
      }
    };

    timeoutTimer = setTimeout(() => {
      if (sawFirstResponse || controller.isSettled || promptAbort.signal.aborted) return;
      timedOut = true;
      cleanupActive(true);
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
    }, remainingWatchdogMs);
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

    const shouldSeedHistory = this.#turns === 0 && !this.#skipHistorySeed && !params.promptReplaced;
    this.#skipHistorySeed = false;

    const customSys = this.#turns === 0 ? params.systemPromptAppend?.trim() : undefined;
    const livePrompt =
      continuationImageBatches.length > 0
        ? buildToolImageContinuationPrompt(continuationImageBatches, params.prompt)
        : params.prompt;

    let outboundPrompt: AcpContentBlock[];
    if (shouldSeedHistory) {
      const seeded = buildSeededSessionPrompt(params.messages, {
        allowImages,
        customSystemContext: customSys,
        ...(continuationImageBatches.length > 0 ? { trailingPrompt: livePrompt } : {}),
      });
      outboundPrompt = seeded.length > 0 ? seeded : prependSystemContext(livePrompt, customSys);
    } else {
      outboundPrompt = prependSystemContext(livePrompt, customSys);
    }

    // Fire `session/prompt` asynchronously while `streamSimple` consumes `controller.drainWave()`.
    activeTurn.done = this.#drivePromptTurn({
      client,
      sessionId,
      piSessionId: params.sessionId,
      bridgeServerName,
      outboundPrompt,
      activeTurn,
      controller,
      acpTools,
      quotaTracker,
      workingTracker,
      promptAbort,
      externalSignal: params.signal,
      onExternalAbort,
      noteFirstResponse,
      isTimedOut: () => timedOut,
      cleanupActive,
      dropIfConsumed,
      setWorkingStatus,
      notifyAckWaiters,
    });

    return {
      controller,
      attachExisting: false,
      bridgeServerName,
      signalTurnAbort: () => activeTurn.requestAbort(),
    };
  }

  /**
   * Run a stateless, toolless one-off request (e.g. `/compact` summarization,
   * `/btw` side threads, cache warming) on an isolated ACP worker process so it
   * never mutates or pollutes the main conversation's context or state.
   */
  async startEphemeralTurn(params: StartEphemeralTurnParams): Promise<TurnSessionHandle> {
    if (params.signal?.aborted) {
      throw makeAbortError("Antigravity turn aborted.");
    }

    const baseTimeoutMs = this.#deps.firstResponseTimeoutMs ?? DEFAULT_FIRST_RESPONSE_TIMEOUT_MS;
    const effectiveTimeoutMs =
      typeof params.timeoutMs === "number" && params.timeoutMs > 0
        ? Math.min(params.timeoutMs, baseTimeoutMs)
        : baseTimeoutMs;
    const deadlineMs = effectiveTimeoutMs > 0 ? Date.now() + effectiveTimeoutMs : undefined;
    const remainingMs = () =>
      deadlineMs !== undefined ? Math.max(1, deadlineMs - Date.now()) : undefined;

    this.#ephemeralAgentHandle ??= createToollessAgent({ agentsDir: this.#deps.agentsDir });
    await this.#ephemeralAgentHandle.write();

    const verified = this.#deps.checkBinary
      ? await this.#deps.checkBinary(this.#deps.binaryPath, params.signal)
      : await checkAgyBinary({ binaryPath: this.#deps.binaryPath, signal: params.signal });

    if (params.signal?.aborted) {
      throw makeAbortError("Antigravity turn aborted.");
    }

    const effectiveEnv = resolveAgyChildEnv({
      hostHindsightActive: this.#deps.isHostHindsightActive?.(),
      env: params.env,
    });
    const clientOptions: AgyAcpClientOptions = {
      binary: verified.binary,
      cwd: this.#deps.cwd,
      agentName: this.#ephemeralAgentHandle.agentName,
      extraDirs: this.#deps.extraDirs,
      sandbox: this.#deps.sandbox,
      env: effectiveEnv,
    };
    const client = this.#deps.createClient
      ? this.#deps.createClient(clientOptions)
      : new AgyAcpClient(clientOptions);

    const setupStartedAt = Date.now();
    let sessionId: string;
    try {
      await client.initialize({ signal: params.signal, timeoutMs: remainingMs() });
      const created = await client.newSession({
        cwd: this.#deps.cwd,
        mcpServers: [],
        signal: params.signal,
        timeoutMs: remainingMs(),
      });
      sessionId = created.sessionId;
      await this.#syncSessionConfigTo(
        client,
        sessionId,
        { modelId: params.modelId, effort: params.effort, executionMode: "default" },
        created.configOptions,
        { signal: params.signal, timeoutMs: remainingMs() },
      );
    } catch (err) {
      await client.dispose(50).catch(() => {});
      if (params.signal?.aborted || isAbortError(err)) {
        throw makeAbortError("Antigravity turn aborted.");
      }
      if (err instanceof AcpRequestStallError) {
        throw new AgyFirstResponseTimeoutError(effectiveTimeoutMs);
      }
      if (err instanceof Error && isAgyQuotaError(err.message)) {
        throw new AgyQuotaError(err.message);
      }
      throw err;
    }

    const remainingWatchdogMs = Math.max(
      1,
      effectiveTimeoutMs - Math.max(0, Date.now() - setupStartedAt),
    );
    const controller = new AgyTurnController();
    const acpTools = new Map<string, AgyToolView>();
    const quotaTracker = new AgyQuotaTurnTracker();
    const promptAbort = new AbortController();

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

    timeoutTimer = setTimeout(() => {
      if (sawFirstResponse || controller.isSettled || promptAbort.signal.aborted) return;
      timedOut = true;
      clearTimeoutTimer();
      try {
        client.cancelSession(sessionId);
      } catch {
        // Best-effort cancel before hard client disposal.
      }
      void client.dispose(50).catch(() => {});
      controller.fail(new AgyFirstResponseTimeoutError(effectiveTimeoutMs));
    }, remainingWatchdogMs);
    timeoutTimer.unref?.();

    const onExternalAbort = () => {
      clearTimeoutTimer();
      promptAbort.abort();
    };
    if (params.signal) {
      if (params.signal.aborted) {
        onExternalAbort();
      } else {
        params.signal.addEventListener("abort", onExternalAbort, { once: true });
      }
    }

    const outboundPrompt = prependSystemContext(params.prompt, params.systemPromptAppend);

    void (async () => {
      try {
        const promptResult = await client.prompt(sessionId, outboundPrompt, {
          signal: promptAbort.signal,
          onUpdate: (update) => {
            if (timedOut || controller.isSettled) return;
            const activity = acpUpdateToActivity(update, acpTools);
            if (!activity) return;

            if (activity.type === "context_usage") {
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
              controller.pushActivity(activity);
              return;
            }

            if (activity.type === "text_delta") {
              quotaTracker.noteText(activity.text);
              controller.pushActivity(activity);
            }
          },
        });

        params.signal?.removeEventListener("abort", onExternalAbort);
        clearTimeoutTimer();
        if (timedOut || controller.isSettled) return;

        const normalizedUsage = normalizeAcpPromptUsage(promptResult.usage);
        const errorBanner = quotaTracker.detectTurnErrorBanner({
          totalTokens: normalizedUsage?.total_tokens,
          outputTokens: normalizedUsage?.output_tokens,
        });
        if (errorBanner?.kind === "quota") {
          controller.fail(
            new AgyQuotaError(errorBanner.message, { resetsIn: errorBanner.resetsIn }),
          );
          return;
        }
        if (errorBanner?.kind === "api_error") {
          controller.fail(new Error(errorBanner.message));
          return;
        }

        if (promptResult.stopReason === "cancelled") {
          controller.finish({ status: "aborted", stopReason: "aborted" });
          return;
        }

        const stopError = classifyAcpStopReasonFailure(promptResult.stopReason);
        if (stopError) {
          controller.fail(new Error(stopError));
          return;
        }

        controller.finish({
          status: "completed",
          stopReason: promptResult.stopReason,
          usage: normalizedUsage,
        });
      } catch (err) {
        params.signal?.removeEventListener("abort", onExternalAbort);
        clearTimeoutTimer();
        if (timedOut || controller.isSettled) return;

        if (promptAbort.signal.aborted || isAbortError(err)) {
          controller.finish({ status: "aborted", stopReason: "aborted" });
          return;
        }

        const error = err instanceof Error ? err : new Error(String(err));
        if (isAgyQuotaError(error.message)) {
          controller.fail(new AgyQuotaError(error.message));
          return;
        }
        controller.fail(error);
      } finally {
        await client.closeSession?.(sessionId).catch(() => {});
        await client.dispose(100).catch(() => {});
      }
    })();

    return {
      controller,
      attachExisting: false,
      signalTurnAbort: () => {
        clearTimeoutTimer();
        promptAbort.abort();
      },
    };
  }

  /** True when `messages` ends with `toolResult`s matching an active auxiliary turn's bridge calls. */
  matchesAuxiliaryTurn(messages: TranscriptContext["messages"]): boolean {
    return this.#findMatchingAuxiliaryTurn(messages) !== undefined;
  }

  #findMatchingAuxiliaryTurn(messages: TranscriptContext["messages"]): AuxiliaryTurn | undefined {
    for (const aux of this.#auxTurns) {
      if (aux.bridge.pendingCount === 0) continue;
      const pendingIds = new Set(aux.bridge.pendingCallIds);
      if (collectTurnToolResults(messages, pendingIds).length > 0) {
        return aux;
      }
    }
    return undefined;
  }

  /**
   * Run or continue an isolated, tool-bridged one-off turn (e.g. Watchdog
   * boundary reviews, permission arbiter, or foreground subagents) on its own
   * ACP worker and loopback MCP bridge, never touching main session state.
   */
  async startOrContinueAuxiliaryTurn(params: StartAuxiliaryTurnParams): Promise<TurnSessionHandle> {
    const existing = this.#findMatchingAuxiliaryTurn(params.messages);
    if (existing) {
      this.#resolveBridgeResultsFor(existing.bridge, params.messages, { allowImages: false });
      return {
        controller: existing.controller,
        attachExisting: true,
        bridgeServerName: existing.bridge.serverName,
        signalTurnAbort: () => existing.requestAbort(),
        onConsumed: () => {
          if (existing.promptDone) {
            this.#auxTurns.delete(existing);
          }
        },
      };
    }

    if (params.signal?.aborted) {
      throw makeAbortError("Antigravity turn aborted.");
    }

    const baseTimeoutMs = this.#deps.firstResponseTimeoutMs ?? DEFAULT_FIRST_RESPONSE_TIMEOUT_MS;
    const effectiveTimeoutMs =
      typeof params.timeoutMs === "number" && params.timeoutMs > 0
        ? Math.min(params.timeoutMs, baseTimeoutMs)
        : baseTimeoutMs;
    const deadlineMs = effectiveTimeoutMs > 0 ? Date.now() + effectiveTimeoutMs : undefined;
    const remainingMs = () =>
      deadlineMs !== undefined ? Math.max(1, deadlineMs - Date.now()) : undefined;

    const auxId = randomUUID().replaceAll("-", "").slice(0, 12);
    const auxServerName = `pi-aux-${auxId}`;
    const auxToken = randomUUID();
    const auxBridge = new AgyPiBridge(auxServerName, undefined, auxServerName);
    auxBridge.requireToken(auxToken);
    const toolSnapshot = params.tools.map((tool) => ({ ...tool }));
    auxBridge.setToolSource(() => toolSnapshot);
    auxBridge.refreshTools();

    const auxAgent = createAuxiliaryToolsAgent(
      { serverName: auxServerName },
      { agentsDir: this.#deps.agentsDir },
    );

    let client: AgyAcpClientLike | undefined;
    let sessionId = "";
    const setupStartedAt = Date.now();
    try {
      await auxBridge.start();
      await auxAgent.write();

      const verified = this.#deps.checkBinary
        ? await this.#deps.checkBinary(this.#deps.binaryPath, params.signal)
        : await checkAgyBinary({ binaryPath: this.#deps.binaryPath, signal: params.signal });

      if (params.signal?.aborted) {
        throw makeAbortError("Antigravity turn aborted.");
      }
      if (!auxBridge.url) {
        throw new Error("antigravity: auxiliary MCP bridge failed to expose a URL.");
      }

      const effectiveEnv = resolveAgyChildEnv({
        hostHindsightActive: this.#deps.isHostHindsightActive?.(),
        env: params.env,
      });
      const clientOptions: AgyAcpClientOptions = {
        binary: verified.binary,
        cwd: this.#deps.cwd,
        agentName: auxAgent.agentName,
        extraDirs: this.#deps.extraDirs,
        sandbox: this.#deps.sandbox,
        env: effectiveEnv,
      };
      client = this.#deps.createClient
        ? this.#deps.createClient(clientOptions)
        : new AgyAcpClient(clientOptions);

      const mcpServers: AcpHttpMcpServer[] = [
        {
          type: "http",
          name: auxServerName,
          url: auxBridge.url,
          headers: [{ name: "x-pi-bridge-token", value: auxToken }],
        },
      ];

      await client.initialize({ signal: params.signal, timeoutMs: remainingMs() });
      const created = await client.newSession({
        cwd: this.#deps.cwd,
        mcpServers,
        signal: params.signal,
        timeoutMs: remainingMs(),
      });
      sessionId = created.sessionId;
      await this.#syncSessionConfigTo(
        client,
        sessionId,
        { modelId: params.modelId, effort: params.effort, executionMode: "default" },
        created.configOptions,
        { signal: params.signal, timeoutMs: remainingMs() },
      );
    } catch (err) {
      await client?.dispose(50).catch(() => {});
      await auxBridge.close().catch(() => {});
      await auxAgent.remove().catch(() => {});
      if (params.signal?.aborted || isAbortError(err)) {
        throw makeAbortError("Antigravity turn aborted.");
      }
      if (err instanceof AcpRequestStallError) {
        throw new AgyFirstResponseTimeoutError(effectiveTimeoutMs);
      }
      if (err instanceof Error && isAgyQuotaError(err.message)) {
        throw new AgyQuotaError(err.message);
      }
      throw err;
    }

    const activeClient = client;
    const remainingWatchdogMs = Math.max(
      1,
      effectiveTimeoutMs - Math.max(0, Date.now() - setupStartedAt),
    );
    const controller = new AgyTurnController();
    const acpTools = new Map<string, AgyToolView>();
    const quotaTracker = new AgyQuotaTurnTracker();
    const promptAbort = new AbortController();

    let sawFirstResponse = false;
    let timedOut = false;
    let timeoutTimer: ReturnType<typeof setTimeout> | undefined;
    let cleanupPromise: Promise<void> | undefined;

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

    const onExternalAbort = () => {
      auxTurn.requestAbort();
    };

    const cleanup = (): Promise<void> => {
      cleanupPromise ??= (async () => {
        clearTimeoutTimer();
        params.signal?.removeEventListener("abort", onExternalAbort);
        const cancelled = auxBridge.cancelPendingCalls(
          "antigravity: auxiliary turn ended while tool call was pending.",
        );
        if (cancelled > 0) {
          await auxBridge.flushResponses().catch(() => {});
        }
        await activeClient.closeSession?.(sessionId).catch(() => {});
        await activeClient.dispose(100).catch(() => {});
        await auxBridge.close().catch(() => {});
        await auxAgent.remove().catch(() => {});
      })();
      return cleanupPromise;
    };

    const auxTurn: AuxiliaryTurn = {
      controller,
      bridge: auxBridge,
      requestAbort: () => {
        clearTimeoutTimer();
        promptAbort.abort();
        const cancelled = auxBridge.cancelPendingCalls(
          "antigravity: auxiliary turn aborted while tool call was pending.",
        );
        if (cancelled > 0) {
          void auxBridge.flushResponses().catch(() => {});
        }
      },
      cleanup,
      promptDone: false,
    };
    this.#auxTurns.add(auxTurn);

    auxBridge.setOnCall((call) => {
      if (timedOut || controller.isSettled || promptAbort.signal.aborted) return false;
      noteFirstResponse();
      quotaTracker.noteToolActivity();
      controller.pushActivity({
        type: "bridge_call",
        callId: call.id,
        name: call.tool,
        args: call.args,
      });
      return true;
    });

    timeoutTimer = setTimeout(() => {
      if (sawFirstResponse || controller.isSettled || promptAbort.signal.aborted) return;
      timedOut = true;
      clearTimeoutTimer();
      try {
        activeClient.cancelSession(sessionId);
      } catch {
        // Best-effort cancel before hard client disposal.
      }
      void cleanup();
      this.#auxTurns.delete(auxTurn);
      controller.fail(new AgyFirstResponseTimeoutError(effectiveTimeoutMs));
    }, remainingWatchdogMs);
    timeoutTimer.unref?.();

    if (params.signal) {
      if (params.signal.aborted) {
        onExternalAbort();
      } else {
        params.signal.addEventListener("abort", onExternalAbort, { once: true });
      }
    }

    const outboundPrompt = prependSystemContext(params.prompt, params.systemPromptAppend);

    void (async () => {
      try {
        const promptResult = await activeClient.prompt(sessionId, outboundPrompt, {
          signal: promptAbort.signal,
          onUpdate: (update) => {
            if (timedOut || controller.isSettled) return;
            const activity = acpUpdateToActivity(update, acpTools);
            if (!activity) return;

            if (activity.type === "context_usage") {
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
              controller.pushActivity(activity);
              return;
            }

            if (activity.type === "text_delta") {
              quotaTracker.noteText(activity.text);
              controller.pushActivity(activity);
              return;
            }

            if (isBridgedMcpTool(activity, auxServerName)) {
              quotaTracker.noteToolActivity();
            }
          },
        });

        params.signal?.removeEventListener("abort", onExternalAbort);
        clearTimeoutTimer();
        if (timedOut || controller.isSettled) return;

        const normalizedUsage = normalizeAcpPromptUsage(promptResult.usage);
        const errorBanner = quotaTracker.detectTurnErrorBanner({
          totalTokens: normalizedUsage?.total_tokens,
          outputTokens: normalizedUsage?.output_tokens,
        });
        if (errorBanner?.kind === "quota") {
          controller.fail(
            new AgyQuotaError(errorBanner.message, { resetsIn: errorBanner.resetsIn }),
          );
          return;
        }
        if (errorBanner?.kind === "api_error") {
          controller.fail(new Error(errorBanner.message));
          return;
        }

        if (promptResult.stopReason === "cancelled") {
          controller.finish({ status: "aborted", stopReason: "aborted" });
          return;
        }

        const stopError = classifyAcpStopReasonFailure(promptResult.stopReason);
        if (stopError) {
          controller.fail(new Error(stopError));
          return;
        }

        controller.finish({
          status: "completed",
          stopReason: promptResult.stopReason,
          usage: normalizedUsage,
        });
      } catch (err) {
        params.signal?.removeEventListener("abort", onExternalAbort);
        clearTimeoutTimer();
        if (timedOut || controller.isSettled) return;

        if (promptAbort.signal.aborted || isAbortError(err)) {
          controller.finish({ status: "aborted", stopReason: "aborted" });
          return;
        }

        const error = err instanceof Error ? err : new Error(String(err));
        if (isAgyQuotaError(error.message)) {
          controller.fail(new AgyQuotaError(error.message));
          return;
        }
        controller.fail(error);
      } finally {
        auxTurn.promptDone = true;
        await cleanup();
        if (controller.isConsumed || promptAbort.signal.aborted) {
          this.#auxTurns.delete(auxTurn);
        }
      }
    })();

    return {
      controller,
      attachExisting: false,
      bridgeServerName: auxServerName,
      signalTurnAbort: () => auxTurn.requestAbort(),
      onConsumed: () => {
        if (auxTurn.promptDone) {
          this.#auxTurns.delete(auxTurn);
        }
      },
    };
  }

  /** Resolve pending bridged MCP calls from trailing `toolResult` messages in context. */
  resolveBridgeResultsFromMessages(
    messages: Parameters<typeof collectTurnToolResults>[0],
    options?: {
      allowImages?: boolean;
      labels?: ReadonlyMap<string, string>;
      outImageBatches?: BridgedToolImageBatch[];
    },
  ): number {
    return this.#resolveBridgeResultsFor(this.#deps.bridge, messages, options);
  }

  #resolveBridgeResultsFor(
    bridge: AgyPiBridge | undefined,
    messages: Parameters<typeof collectTurnToolResults>[0],
    options?: {
      allowImages?: boolean;
      labels?: ReadonlyMap<string, string>;
      outImageBatches?: BridgedToolImageBatch[];
    },
  ): number {
    if (!bridge || bridge.pendingCount === 0) return 0;
    const pendingIds = new Set(bridge.pendingCallIds);
    const results = collectTurnToolResults(messages, pendingIds);
    let resolved = 0;
    for (const item of results) {
      const { mcpContent, imageBatch } = splitToolResultImages(item, {
        allowImages: options?.allowImages ?? true,
        label: options?.labels?.get(item.callId),
      });
      if (
        bridge.resolveCall(item.callId, {
          content: mcpContent,
          isError: item.isError,
        })
      ) {
        resolved += 1;
        if (imageBatch) {
          options?.outImageBatches?.push(imageBatch);
        }
      }
    }
    return resolved;
  }
}
