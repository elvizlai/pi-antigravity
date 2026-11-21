import { spawnSync } from "node:child_process";

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
export function sweepTrackedProcessGroups(): void {
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
