import { type ChildProcessWithoutNullStreams, spawn } from "node:child_process";
import { createInterface } from "node:readline";
import type { AgyAcpConfigOption } from "./models.ts";
import {
  killProcessTree,
  registerProcessGroupDeathHooks,
  terminateProcessTree,
  trackProcessGroup,
  untrackProcessGroup,
} from "./process-tree.ts";
import { asRecord, makeAbortError } from "./shared.ts";

export type AcpContentBlock =
  | { type: "text"; text: string }
  | { type: "image"; data: string; mimeType: string };

export interface AcpInitializeResult {
  protocolVersion?: number;
  agentInfo?: { name?: string; version?: string };
  agentCapabilities?: Record<string, unknown>;
}

export interface AcpHttpMcpServer {
  type: "http";
  name: string;
  url: string;
  headers: Array<{ name: string; value: string }>;
}

export interface AcpSessionResult {
  sessionId: string;
  configOptions?: AgyAcpConfigOption[];
  models?: unknown;
  modes?: unknown;
}

export interface AcpPromptResult {
  stopReason: string;
  usage?: {
    totalTokens?: number;
    inputTokens?: number;
    outputTokens?: number;
    cachedReadTokens?: number;
    cachedWriteTokens?: number;
    thoughtTokens?: number;
  };
}

export interface AcpPermissionOption {
  optionId?: string;
  id?: string;
  kind?: string;
  name?: string;
}

export interface AcpPermissionRequest {
  sessionId?: string;
  toolCall?: Record<string, unknown>;
  options?: AcpPermissionOption[];
}

export interface AcpRpcCallOptions {
  signal?: AbortSignal;
  timeoutMs?: number;
}

export class AcpRequestStallError extends Error {
  readonly method: string;
  readonly timeoutMs: number;

  constructor(method: string, timeoutMs: number) {
    const seconds = Math.max(1, Math.round(timeoutMs / 1000));
    super(`antigravity: ACP ${method} did not respond within ${seconds}s`);
    this.name = "AcpRequestStallError";
    this.method = method;
    this.timeoutMs = timeoutMs;
  }
}

export interface AgyAcpClientOptions {
  binary: string;
  cwd: string;
  agentName?: string;
  extraDirs?: readonly string[];
  sandbox?: boolean;
  env?: NodeJS.ProcessEnv;
  cancelGraceMs?: number;
  spawnProcess?: (
    command: string,
    args: readonly string[],
    options: {
      cwd: string;
      env: NodeJS.ProcessEnv;
      stdio: ["pipe", "pipe", "pipe"];
      detached: boolean;
      windowsHide?: boolean;
    },
  ) => ChildProcessWithoutNullStreams;
  onPermissionRequest?: (
    request: AcpPermissionRequest,
  ) => Promise<{ outcome: { outcome: "selected"; optionId: string } | { outcome: "cancelled" } }>;
  onStderr?: (chunk: string) => void;
}

interface PendingRpc {
  method: string;
  resolve: (value: unknown) => void;
  reject: (reason: Error) => void;
}

const MAX_STDERR_BYTES = 8_192;
const DEFAULT_CANCEL_GRACE_MS = 5_000;

function selectPermissionOptionId(
  options: readonly AcpPermissionOption[] | undefined,
): string | undefined {
  if (!options || options.length === 0) return undefined;
  const pick = (predicate: (opt: AcpPermissionOption) => boolean) => {
    const found = options.find(predicate);
    return found?.optionId ?? found?.id;
  };
  return (
    pick(
      (o) => o.kind === "allow_always" || o.optionId === "allow_always" || o.id === "allow_always",
    ) ??
    pick((o) => o.kind === "allow_once" || o.optionId === "allow_once" || o.id === "allow_once") ??
    pick((o) => (o.kind ?? o.optionId ?? "").startsWith("allow"))
  );
}

function formatRpcError(errObj: Record<string, unknown>, method: string): Error {
  const code = errObj.code;
  const msg = typeof errObj.message === "string" ? errObj.message : `JSON-RPC error in ${method}`;
  const err = new Error(code !== undefined ? `${msg} (code ${String(code)})` : msg);
  (err as Error & { code?: unknown }).code = code;
  return err;
}

export class AgyAcpClient {
  readonly #child: ChildProcessWithoutNullStreams;
  readonly #pending = new Map<number, PendingRpc>();
  readonly #sessionListeners = new Map<string, Set<(update: Record<string, unknown>) => void>>();
  readonly #onPermissionRequest?: AgyAcpClientOptions["onPermissionRequest"];
  readonly #onStderr?: AgyAcpClientOptions["onStderr"];
  readonly #cancelGraceMs: number;
  #nextId = 1;
  #stderrTail = "";
  #closed = false;
  #exited = false;
  #exitError?: Error;
  #exitPromise: Promise<void>;
  #forceTerminate?: (err: Error) => void;

  constructor(options: AgyAcpClientOptions) {
    this.#onPermissionRequest = options.onPermissionRequest;
    this.#onStderr = options.onStderr;
    this.#cancelGraceMs = options.cancelGraceMs ?? DEFAULT_CANCEL_GRACE_MS;

    const args = ["acp", "--dangerously-skip-permissions", "--process-per-session"];
    if (options.agentName) {
      args.push("--agent", options.agentName);
    }
    if (options.extraDirs) {
      for (const dir of options.extraDirs) {
        args.push("--add-dir", dir);
      }
    }
    if (options.sandbox) {
      args.push("--sandbox");
    }

    const spawner = options.spawnProcess ?? spawn;
    this.#child = spawner(options.binary, args, {
      cwd: options.cwd,
      env: options.env ? { ...process.env, ...options.env } : process.env,
      stdio: ["pipe", "pipe", "pipe"],
      detached: process.platform !== "win32",
      windowsHide: true,
    });
    registerProcessGroupDeathHooks();
    trackProcessGroup(this.#child.pid);

    // Guard against unhandled EPIPE if agy closes stdin right as we write.
    this.#child.stdin.on("error", () => {});

    this.#child.stderr.on("data", (chunk) => {
      const text = String(chunk);
      this.#stderrTail = (this.#stderrTail + text).slice(-MAX_STDERR_BYTES);
      this.#onStderr?.(text);
    });

    const rl = createInterface({ input: this.#child.stdout, crlfDelay: Number.POSITIVE_INFINITY });
    rl.on("line", (line) => {
      this.#handleStdoutLine(line);
    });

    this.#exitPromise = new Promise((resolve) => {
      const onTerminate = (err: Error) => {
        if (this.#exited) return;
        this.#exited = true;
        rl.close();
        untrackProcessGroup(this.#child.pid);
        this.#exitError = err;
        for (const [id, pending] of this.#pending.entries()) {
          this.#pending.delete(id);
          pending.reject(err);
        }
        resolve();
      };
      this.#forceTerminate = onTerminate;

      this.#child.once("error", (error) => {
        onTerminate(
          new Error(`agy acp process error: ${error.message}${this.#formatStderrSuffix()}`),
        );
      });
      this.#child.once("close", (code, signal) => {
        const reason =
          code !== null
            ? `agy acp exited with code ${code}`
            : `agy acp killed by signal ${signal ?? "unknown"}`;
        onTerminate(new Error(`${reason}${this.#formatStderrSuffix()}`));
      });
    });
  }

  get isAlive(): boolean {
    return !this.#closed && !this.#exited;
  }

  get stderrTail(): string {
    return this.#stderrTail.trim();
  }

  #formatStderrSuffix(): string {
    const tail = this.#stderrTail.trim();
    return tail ? `\nstderr: ${tail}` : "";
  }

  #writeMessage(payload: Record<string, unknown>): void {
    if (!this.isAlive || this.#child.stdin.destroyed || !this.#child.stdin.writable) {
      throw this.#exitError ?? new Error("agy acp stdio stream is closed.");
    }
    this.#child.stdin.write(`${JSON.stringify(payload)}\n`);
  }

  #request<T>(
    method: string,
    params: Record<string, unknown>,
    options?: AcpRpcCallOptions,
  ): Promise<T> {
    if (!this.isAlive) {
      return Promise.reject(this.#exitError ?? new Error(`agy acp is closed (${method}).`));
    }
    if (options?.signal?.aborted) {
      return Promise.reject(makeAbortError(`Aborted before ACP ${method}.`));
    }
    const id = this.#nextId++;
    return new Promise<T>((resolve, reject) => {
      let timer: ReturnType<typeof setTimeout> | undefined;
      const cleanup = () => {
        if (timer) {
          clearTimeout(timer);
          timer = undefined;
        }
        options?.signal?.removeEventListener("abort", onAbort);
      };
      const onAbort = () => {
        if (!this.#pending.delete(id)) return;
        cleanup();
        reject(makeAbortError(`ACP ${method} aborted.`));
      };

      if (typeof options?.timeoutMs === "number" && options.timeoutMs > 0) {
        const timeoutMs = options.timeoutMs;
        timer = setTimeout(() => {
          if (!this.#pending.delete(id)) return;
          cleanup();
          reject(new AcpRequestStallError(method, timeoutMs));
        }, timeoutMs);
        timer.unref?.();
      }
      options?.signal?.addEventListener("abort", onAbort, { once: true });

      this.#pending.set(id, {
        method,
        resolve: (value) => {
          cleanup();
          resolve(value as T);
        },
        reject: (err) => {
          cleanup();
          reject(err);
        },
      });
      try {
        this.#writeMessage({ jsonrpc: "2.0", id, method, params });
      } catch (err) {
        this.#pending.delete(id);
        cleanup();
        reject(err instanceof Error ? err : new Error(String(err)));
      }
    });
  }

  #notify(method: string, params: Record<string, unknown>): void {
    if (!this.isAlive) return;
    try {
      this.#writeMessage({ jsonrpc: "2.0", method, params });
    } catch {
      // Ignore best-effort notification failure on closing stream.
    }
  }

  #handleStdoutLine(rawLine: string): void {
    const line = rawLine.trim();
    if (!line) return;
    let parsed: unknown;
    try {
      parsed = JSON.parse(line);
    } catch {
      return;
    }
    const msg = asRecord(parsed);
    if (!msg) return;

    // 1. Response to a client request
    if (msg.id !== undefined && msg.method === undefined) {
      const id = typeof msg.id === "number" ? msg.id : Number(msg.id);
      const pending = this.#pending.get(id);
      if (!pending) return;
      this.#pending.delete(id);
      const errObj = asRecord(msg.error);
      if (errObj) {
        pending.reject(formatRpcError(errObj, pending.method));
      } else {
        pending.resolve(msg.result);
      }
      return;
    }

    // 2. Server -> client request (carries both id and method)
    if (msg.id !== undefined && typeof msg.method === "string") {
      void this.#handleServerRequest(msg.id, msg.method, asRecord(msg.params));
      return;
    }

    // 3. Server -> client notification
    if (typeof msg.method === "string") {
      this.#handleNotification(msg.method, asRecord(msg.params));
    }
  }

  async #handleServerRequest(
    id: unknown,
    method: string,
    params: Record<string, unknown> | undefined,
  ): Promise<void> {
    if (method === "session/request_permission") {
      try {
        const req: AcpPermissionRequest = {
          sessionId: typeof params?.sessionId === "string" ? params.sessionId : undefined,
          toolCall: asRecord(params?.toolCall),
          options: Array.isArray(params?.options)
            ? (params.options as AcpPermissionOption[])
            : undefined,
        };
        let result: {
          outcome: { outcome: "selected"; optionId: string } | { outcome: "cancelled" };
        };
        if (this.#onPermissionRequest) {
          result = await this.#onPermissionRequest(req);
        } else {
          const optionId = selectPermissionOptionId(req.options);
          result = optionId
            ? { outcome: { outcome: "selected", optionId } }
            : { outcome: { outcome: "cancelled" } };
        }
        this.#writeMessage({ jsonrpc: "2.0", id, result });
      } catch (err) {
        try {
          this.#writeMessage({
            jsonrpc: "2.0",
            id,
            error: {
              code: -32603,
              message: err instanceof Error ? err.message : String(err),
            },
          });
        } catch {
          // Ignore write failure on dying stream.
        }
      }
      return;
    }

    try {
      this.#writeMessage({
        jsonrpc: "2.0",
        id,
        error: { code: -32601, message: `Unsupported ACP client method: ${method}` },
      });
    } catch {
      // Ignore write failure on dying stream.
    }
  }

  #handleNotification(method: string, params: Record<string, unknown> | undefined): void {
    if (method !== "session/update" || !params) return;
    const sessionId = typeof params.sessionId === "string" ? params.sessionId : undefined;
    const update = asRecord(params.update);
    if (!sessionId || !update) return;
    const listeners = this.#sessionListeners.get(sessionId);
    if (!listeners) return;
    for (const listener of listeners) {
      listener(update);
    }
  }

  async initialize(options?: AcpRpcCallOptions): Promise<AcpInitializeResult> {
    return this.#request<AcpInitializeResult>(
      "initialize",
      {
        protocolVersion: 1,
        clientCapabilities: {
          fs: { readTextFile: false, writeTextFile: false },
          terminal: false,
        },
        clientInfo: {
          name: "@elvizlai/pi-antigravity",
          version: "0.1.0",
        },
      },
      options,
    );
  }

  async newSession(options: {
    cwd: string;
    mcpServers?: readonly AcpHttpMcpServer[];
    signal?: AbortSignal;
    timeoutMs?: number;
  }): Promise<AcpSessionResult> {
    return this.#request<AcpSessionResult>(
      "session/new",
      {
        cwd: options.cwd,
        mcpServers: options.mcpServers ?? [],
      },
      { signal: options.signal, timeoutMs: options.timeoutMs },
    );
  }

  async resumeSession(options: {
    sessionId: string;
    cwd: string;
    mcpServers?: readonly AcpHttpMcpServer[];
    signal?: AbortSignal;
    timeoutMs?: number;
  }): Promise<AcpSessionResult> {
    const result = await this.#request<AcpSessionResult>(
      "session/resume",
      {
        sessionId: options.sessionId,
        cwd: options.cwd,
        mcpServers: options.mcpServers ?? [],
      },
      { signal: options.signal, timeoutMs: options.timeoutMs },
    );
    if (!result.sessionId || result.sessionId !== options.sessionId) {
      if (result.sessionId) {
        await this.closeSession(result.sessionId).catch(() => {});
      }
      throw new Error(
        result.sessionId
          ? `agy session/resume returned different sessionId "${result.sessionId}" for requested "${options.sessionId}".`
          : `agy session/resume returned empty sessionId for requested "${options.sessionId}".`,
      );
    }
    return result;
  }

  async setConfigOption(
    sessionId: string,
    configId: string,
    value: string,
    options?: AcpRpcCallOptions,
  ): Promise<{ configOptions?: AgyAcpConfigOption[] }> {
    return this.#request<{ configOptions?: AgyAcpConfigOption[] }>(
      "session/set_config_option",
      {
        sessionId,
        configId,
        value,
      },
      options,
    );
  }

  cancelSession(sessionId: string): void {
    // ACP `session/cancel` is strictly a JSON-RPC notification (no id).
    this.#notify("session/cancel", { sessionId });
  }

  async closeSession(sessionId: string, options?: AcpRpcCallOptions): Promise<void> {
    if (!this.isAlive) return;
    await this.#request("session/close", { sessionId }, options);
  }

  async prompt(
    sessionId: string,
    blocks: readonly AcpContentBlock[],
    options?: {
      signal?: AbortSignal;
      cancelGraceMs?: number;
      onUpdate?: (update: Record<string, unknown>) => void;
    },
  ): Promise<AcpPromptResult> {
    if (options?.signal?.aborted) {
      throw makeAbortError("Aborted before sending ACP prompt.");
    }

    const listener = options?.onUpdate;
    if (listener) {
      let set = this.#sessionListeners.get(sessionId);
      if (!set) {
        set = new Set();
        this.#sessionListeners.set(sessionId, set);
      }
      set.add(listener);
    }

    let graceTimer: ReturnType<typeof setTimeout> | undefined;
    let aborted = false;

    const cleanup = () => {
      if (graceTimer) clearTimeout(graceTimer);
      options?.signal?.removeEventListener("abort", onAbort);
      if (listener) {
        const set = this.#sessionListeners.get(sessionId);
        set?.delete(listener);
        if (set && set.size === 0) this.#sessionListeners.delete(sessionId);
      }
    };

    const onAbort = () => {
      aborted = true;
      this.cancelSession(sessionId);
      const graceMs = options?.cancelGraceMs ?? this.#cancelGraceMs;
      graceTimer = setTimeout(() => {
        void this.dispose(100);
      }, graceMs);
      graceTimer.unref?.();
    };

    options?.signal?.addEventListener("abort", onAbort, { once: true });

    try {
      const result = await this.#request<AcpPromptResult>("session/prompt", {
        sessionId,
        prompt: blocks,
      });
      if (aborted || result.stopReason === "cancelled") {
        throw makeAbortError("Antigravity turn cancelled.");
      }
      return result;
    } catch (err) {
      if (aborted) {
        throw makeAbortError("Antigravity turn aborted.");
      }
      throw err;
    } finally {
      cleanup();
    }
  }

  #waitForExitWithin(ms: number): Promise<boolean> {
    if (this.#exited) return Promise.resolve(true);
    return Promise.race([
      this.#exitPromise.then(() => true),
      new Promise<boolean>((resolve) => {
        const t = setTimeout(() => resolve(false), ms);
        t.unref?.();
      }),
    ]);
  }

  async dispose(graceMs = 2_500): Promise<void> {
    if (this.#closed) {
      await this.#exitPromise;
      return;
    }
    this.#closed = true;
    try {
      this.#child.stdin.end();
    } catch {
      // Ignore already-ended stdin.
    }
    const exitedOnEof = await this.#waitForExitWithin(graceMs);
    if (!exitedOnEof && !this.#exited) {
      if (terminateProcessTree(this.#child.pid)) {
        await this.#waitForExitWithin(Math.max(20, Math.min(graceMs, 400)));
      }
    }
    if (!this.#exited) {
      killProcessTree(this.#child.pid);
      const exitedAfterKill = await this.#waitForExitWithin(200);
      if (!exitedAfterKill && !this.#exited) {
        this.#child.stdin.destroy();
        this.#child.stdout.destroy();
        this.#child.stderr.destroy();
        this.#forceTerminate?.(
          new Error(`agy acp forced termination after dispose${this.#formatStderrSuffix()}`),
        );
      }
      await this.#exitPromise;
    }
  }
}
