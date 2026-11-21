import { spawn } from "node:child_process";
import { promises as fs, statSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { gte, valid } from "semver";
import which from "which";
import {
  killProcessTree,
  registerProcessGroupDeathHooks,
  trackProcessGroup,
  untrackProcessGroup,
} from "./process-tree.ts";

/** Minimum `agy` CLI version required for `agy acp` (`--process-per-session`, `_meta`, `session/resume`). */
export const MIN_AGY_VERSION = "1.5.0";

export const AGY_INSTALL_HINT =
  "Install Antigravity CLI >= 1.5.0 (`npm i -g @google/antigravity` or official installer) and ensure `agy` is on PATH.";

const AGY_PROBE_TIMEOUT_MS = 10_000;

export function withAgyAutoUpdateDisabled(env: NodeJS.ProcessEnv = process.env): NodeJS.ProcessEnv {
  return {
    ...env,
    // Verified against agy: `"1"` does not suppress auto-update checks; `"true"` does.
    AGY_CLI_DISABLE_AUTO_UPDATE: "true",
  };
}

export function isRegularFileSync(filePath: string): boolean {
  try {
    return statSync(filePath).isFile();
  } catch {
    return false;
  }
}

function candidateAgyPaths(home = os.homedir()): string[] {
  const candidates =
    process.platform === "win32"
      ? [path.join(home, ".local", "bin", "agy.exe"), path.join(home, ".local", "bin", "agy.cmd")]
      : ["/opt/homebrew/bin/agy", "/usr/local/bin/agy", path.join(home, ".local", "bin", "agy")];
  return candidates;
}

/** Locate the `agy` CLI binary across PATH and standard installation directories. */
export function getAgyBinary(overridePath?: string): string {
  const custom = overridePath ?? process.env.PI_ANTIGRAVITY_BINARY;
  if (custom) {
    if (isRegularFileSync(custom)) return custom;
    throw new AgyCompatibilityError(
      `Configured Antigravity binary not found at "${custom}". ${AGY_INSTALL_HINT}`,
    );
  }

  const fromPath = which.sync("agy", { nothrow: true });
  if (fromPath && isRegularFileSync(fromPath)) return fromPath;

  for (const candidate of candidateAgyPaths()) {
    if (isRegularFileSync(candidate)) return candidate;
  }

  throw new AgyCompatibilityError(`Antigravity CLI ("agy") not found. ${AGY_INSTALL_HINT}`);
}

export function parseAgyVersion(output: string): string | undefined {
  const cleaned = output.trim();
  if (!cleaned) return undefined;
  const tokens = cleaned.split(/\s+/);
  for (const token of tokens) {
    const normalized = valid(token.replace(/^v/i, ""));
    if (normalized) return normalized;
  }
  const match = /\bv?(\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?)/.exec(cleaned);
  return match ? (valid(match[1]) ?? undefined) : undefined;
}

export class AgyCompatibilityError extends Error {
  readonly detectedVersion?: string;
  readonly minimumVersion = MIN_AGY_VERSION;
  readonly binaryPath?: string;

  constructor(message: string, options?: { detectedVersion?: string; binaryPath?: string }) {
    super(message);
    this.name = "AgyCompatibilityError";
    this.detectedVersion = options?.detectedVersion;
    this.binaryPath = options?.binaryPath;
  }
}

export interface AgyBinaryInfo {
  binary: string;
  version: string;
}

type VersionRunner = (binary: string, signal?: AbortSignal) => Promise<string>;

const verifiedCache = new Map<string, { fingerprint: string; info: AgyBinaryInfo }>();

export function resetAgyBinaryCache(): void {
  verifiedCache.clear();
}

function defaultVersionRunner(binary: string, signal?: AbortSignal): Promise<string> {
  return new Promise((resolve, reject) => {
    const child = spawn(binary, ["--version"], {
      stdio: ["ignore", "pipe", "pipe"],
      detached: process.platform !== "win32",
      env: withAgyAutoUpdateDisabled(),
    });
    registerProcessGroupDeathHooks();
    trackProcessGroup(child.pid);

    let stdout = "";
    let stderr = "";
    let settled = false;

    const finish = (err?: Error) => {
      if (settled) return;
      settled = true;
      clearTimeout(timeout);
      signal?.removeEventListener("abort", onAbort);
      untrackProcessGroup(child.pid);
      if (err) reject(err);
      else resolve(`${stdout}\n${stderr}`.trim());
    };

    const onAbort = () => {
      killProcessTree(child.pid);
      const err = new Error("Antigravity version probe aborted.");
      err.name = "AbortError";
      finish(err);
    };

    if (signal?.aborted) {
      onAbort();
      return;
    }
    signal?.addEventListener("abort", onAbort, { once: true });

    const timeout = setTimeout(() => {
      killProcessTree(child.pid);
      finish(
        new Error(`Timed out waiting for \`${binary} --version\` after ${AGY_PROBE_TIMEOUT_MS}ms.`),
      );
    }, AGY_PROBE_TIMEOUT_MS);
    timeout.unref?.();

    child.stdout?.on("data", (chunk) => {
      stdout += String(chunk);
    });
    child.stderr?.on("data", (chunk) => {
      stderr += String(chunk);
    });
    child.once("error", (err) => finish(err));
    child.once("close", (code) => {
      if (code !== 0) {
        finish(new Error(stderr.trim() || stdout.trim() || `\`agy --version\` exited ${code}.`));
        return;
      }
      finish();
    });
  });
}

async function binaryFingerprint(binary: string): Promise<string> {
  try {
    const st = await fs.stat(binary);
    return `${st.size}:${Math.trunc(st.mtimeMs)}`;
  } catch {
    return "unknown";
  }
}

/** Verify that `agy` exists and meets `MIN_AGY_VERSION`. Cached per binary file stats. */
export async function checkAgyBinary(options?: {
  binaryPath?: string;
  signal?: AbortSignal;
  runner?: VersionRunner;
}): Promise<AgyBinaryInfo> {
  const binary = getAgyBinary(options?.binaryPath);
  const fingerprint = await binaryFingerprint(binary);
  const cached = verifiedCache.get(binary);
  if (cached && cached.fingerprint === fingerprint) {
    return cached.info;
  }

  const runVersion = options?.runner ?? defaultVersionRunner;
  const output = await runVersion(binary, options?.signal);
  const version = parseAgyVersion(output);
  if (!version) {
    throw new AgyCompatibilityError(
      `Could not determine Antigravity CLI version from \`${binary} --version\` output: ${JSON.stringify(output)}. ${AGY_INSTALL_HINT}`,
      { binaryPath: binary },
    );
  }
  if (!gte(version, MIN_AGY_VERSION)) {
    throw new AgyCompatibilityError(
      `Antigravity CLI v${version} is too old for \`agy acp\`. Upgrade to >= v${MIN_AGY_VERSION}. ${AGY_INSTALL_HINT}`,
      { detectedVersion: version, binaryPath: binary },
    );
  }

  const info: AgyBinaryInfo = { binary, version };
  verifiedCache.set(binary, { fingerprint, info });
  return info;
}
