import { type ChildProcess, spawn, spawnSync } from "node:child_process";
import { makeAbortError } from "./shared.ts";

/**
 * Detached process-group lifecycle and cleanup for agy subprocesses.
 *
 * `agy` can spawn child tools and MCP servers in its own process tree. Spawning
 * each child detached (POSIX process group) lets one group kill reap the whole
 * tree even when Pi exits on SIGHUP or SIGTERM.
 */

const PROCESS_TREE_REGISTRY_SYMBOL = Symbol.for("pi-antigravity.process-tree");

interface ProcessTreeRegistry {
  pids: Set<number>;
  hooked: boolean;
  onExitHooks: Set<() => void>;
}

function getRegistry(): ProcessTreeRegistry {
  const g = globalThis as Record<symbol, ProcessTreeRegistry | undefined>;
  let registry = g[PROCESS_TREE_REGISTRY_SYMBOL];
  if (!registry) {
    registry = { pids: new Set<number>(), hooked: false, onExitHooks: new Set<() => void>() };
    g[PROCESS_TREE_REGISTRY_SYMBOL] = registry;
  }
  return registry;
}

/** Record a detached child leader pid so shutdown hooks reap it if Pi exits first. */
export function trackProcessGroup(pid: number | undefined): void {
  if (typeof pid !== "number" || !Number.isInteger(pid) || pid <= 0) return;
  getRegistry().pids.add(pid);
}

/** Stop tracking a child leader pid once it has exited or been reaped. */
export function untrackProcessGroup(pid: number | undefined): void {
  if (typeof pid !== "number" || !Number.isInteger(pid) || pid <= 0) return;
  getRegistry().pids.delete(pid);
}

/**
 * Send `SIGTERM` to a child process and every descendant in its detached POSIX process group.
 * Keeps the pid tracked so synchronous exit/SIGHUP hooks still `SIGKILL` it if the parent exits first.
 */
export function terminateProcessTree(pid: number | undefined): boolean {
  if (typeof pid !== "number" || !Number.isInteger(pid) || pid <= 0) return false;
  if (process.platform === "win32") return false;

  try {
    process.kill(-pid, "SIGTERM");
    return true;
  } catch {
    try {
      process.kill(pid, "SIGTERM");
      return true;
    } catch {
      return false;
    }
  }
}

/** Kill a child process and every descendant in its detached process group. */
export function killProcessTree(pid: number | undefined): void {
  if (typeof pid !== "number" || !Number.isInteger(pid) || pid <= 0) return;
  untrackProcessGroup(pid);

  if (process.platform === "win32") {
    try {
      spawnSync("taskkill", ["/F", "/T", "/PID", String(pid)], {
        stdio: "ignore",
        windowsHide: true,
        timeout: 3_000,
      });
    } catch {
      // Best-effort cleanup.
    }
    return;
  }

  try {
    process.kill(-pid, "SIGKILL");
  } catch {
    try {
      process.kill(pid, "SIGKILL");
    } catch {
      // Process or group already exited.
    }
  }
}

/** Sweep every tracked child process group synchronously. */
function sweepTrackedProcessGroups(): void {
  const { pids } = getRegistry();
  for (const pid of [...pids]) {
    killProcessTree(pid);
  }
}

/**
 * Install process-exit / SIGHUP cleanup hooks once per Node process (safe across
 * `/reload` via `globalThis` symbol registry).
 */
export function registerProcessGroupDeathHooks(onExit?: () => void): () => void {
  const registry = getRegistry();
  if (onExit) registry.onExitHooks.add(onExit);

  if (!registry.hooked) {
    registry.hooked = true;
    const sweep = () => {
      for (const hook of registry.onExitHooks) {
        try {
          hook();
        } catch {
          // Best-effort shutdown hook.
        }
      }
      sweepTrackedProcessGroups();
    };
    process.on("exit", sweep);
    if (process.platform !== "win32") {
      process.on("SIGHUP", () => {
        sweep();
        process.exit(129);
      });
    }
  }

  return () => {
    if (onExit) registry.onExitHooks.delete(onExit);
  };
}

export interface ExecAgyOptions {
  cwd?: string;
  env?: NodeJS.ProcessEnv;
  timeoutMs?: number;
  maxBufferBytes?: number;
  signal?: AbortSignal;
  abortMessage?: string;
  timeoutMessage?: string;
  maxBufferMessage?: string;
  spawnProcess?: typeof spawn;
}

export interface ExecAgyResult {
  code: number | null;
  signal: NodeJS.Signals | null;
  stdout: string;
  stderr: string;
}

const DEFAULT_EXEC_TIMEOUT_MS = 15_000;
const DEFAULT_MAX_BUFFER_BYTES = 1024 * 1024;
const MAX_EXEC_STDERR_CHARS = 8_192;

/**
 * Run a one-shot detached `agy` CLI command with process-group tracking,
 * timeout, abort propagation, and bounded stdio buffering.
 */
export function execAgyCommand(
  binary: string,
  args: readonly string[],
  options?: ExecAgyOptions,
): Promise<ExecAgyResult> {
  const abortMessage = options?.abortMessage ?? "Antigravity command aborted.";
  if (options?.signal?.aborted) {
    return Promise.reject(makeAbortError(abortMessage));
  }

  const timeoutMs = options?.timeoutMs ?? DEFAULT_EXEC_TIMEOUT_MS;
  const maxBufferBytes = options?.maxBufferBytes ?? DEFAULT_MAX_BUFFER_BYTES;
  const doSpawn = options?.spawnProcess ?? spawn;
  const env = options?.env ?? {
    ...process.env,
    HINDSIGHT_DISABLE_HOOKS: process.env.HINDSIGHT_DISABLE_HOOKS ?? "1",
  };

  return new Promise((resolve, reject) => {
    const child = doSpawn(binary, [...args], {
      cwd: options?.cwd,
      env,
      stdio: ["ignore", "pipe", "pipe"],
      detached: process.platform !== "win32",
      windowsHide: true,
    }) as ChildProcess;

    registerProcessGroupDeathHooks();
    trackProcessGroup(child.pid);

    let stdout = "";
    let stderr = "";
    let settled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;

    const cleanup = () => {
      if (timer) {
        clearTimeout(timer);
        timer = undefined;
      }
      options?.signal?.removeEventListener("abort", onAbort);
      untrackProcessGroup(child.pid);
    };

    const fail = (err: Error) => {
      if (settled) return;
      settled = true;
      cleanup();
      reject(err);
    };

    const onAbort = () => {
      killProcessTree(child.pid);
      fail(makeAbortError(abortMessage));
    };

    options?.signal?.addEventListener("abort", onAbort, { once: true });

    timer = setTimeout(() => {
      killProcessTree(child.pid);
      fail(new Error(options?.timeoutMessage ?? `Timed out after ${timeoutMs}ms.`));
    }, timeoutMs);
    timer.unref?.();

    child.stdout?.on("data", (chunk) => {
      stdout += String(chunk);
      if (Buffer.byteLength(stdout, "utf-8") > maxBufferBytes) {
        killProcessTree(child.pid);
        fail(new Error(options?.maxBufferMessage ?? "stdout maxBuffer exceeded"));
      }
    });

    child.stderr?.on("data", (chunk) => {
      stderr = (stderr + String(chunk)).slice(-MAX_EXEC_STDERR_CHARS);
    });

    child.once("error", (err) => fail(err));

    child.once("close", (code, exitSignal) => {
      if (settled) return;
      if (options?.signal?.aborted) {
        fail(makeAbortError(abortMessage));
        return;
      }
      settled = true;
      cleanup();
      resolve({ code, signal: exitSignal, stdout, stderr });
    });
  });
}
