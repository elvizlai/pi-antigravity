import type { spawn } from "node:child_process";
import { getAgyBinary } from "./binary.ts";
import { execAgyCommand } from "./process-tree.ts";
import { asRecord } from "./shared.ts";

/**
 * Neither ACP nor `agy` CLI subcommands expose a dedicated quota endpoint, so `/agy-usage`
 * invokes public CLI print mode (`agy --print /usage --output-format json --print-timeout <n>s`)
 * and reads the structured `command.data.groups` JSON envelope written to `stdout` (0 LLM tokens).
 */

const USAGE_TIMEOUT_MS = 30_000;

const USAGE_MAX_BUFFER_BYTES = 1024 * 1024;
const BAR_SEGMENTS = 20;
const LABEL_COLUMN = 18;
const WINDOW_HOURS = [5, 7 * 24];

export interface AgyUsageBucket {
  id: string;
  name: string;
  description?: string;
  window: string;
  remainingFraction: number;
  resetTime?: string;
}

export interface AgyUsageGroup {
  name: string;
  description?: string;
  buckets: AgyUsageBucket[];
}

export interface AgyUsageReport {
  description?: string;
  groups: AgyUsageGroup[];
}

export interface FetchAgyUsageOptions {
  binary?: string;
  timeoutMs?: number;
  signal?: AbortSignal;
  spawnOverride?: typeof spawn;
}

function asString(value: unknown): string | undefined {
  return typeof value === "string" && value.trim() ? value : undefined;
}

function asNumber(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function unwrapResult(parsed: unknown): Record<string, unknown> {
  const root = asRecord(parsed);
  if (!root) throw new Error("agy-usage: quota response was not an object.");
  if (root.event === "result") {
    const inner = asRecord(root.result);
    if (inner) return inner;
  }
  return root;
}

function parseBucket(value: unknown): AgyUsageBucket | undefined {
  const rec = asRecord(value);
  if (!rec) return undefined;
  const id = asString(rec.id) ?? asString(rec.name);
  const name = asString(rec.name) ?? asString(rec.id);
  const remaining =
    asNumber(rec.remaining_fraction) ?? asNumber(rec.remainingFraction) ?? asNumber(rec.remaining);
  if (!id || !name || remaining === undefined) return undefined;
  return {
    id,
    name,
    description: asString(rec.description),
    window: asString(rec.window) ?? id,
    remainingFraction: remaining,
    resetTime: asString(rec.reset_time) ?? asString(rec.resetTime),
  };
}

function parseGroup(value: unknown): AgyUsageGroup | undefined {
  const rec = asRecord(value);
  if (!rec) return undefined;
  const name = asString(rec.name);
  if (!name) return undefined;
  const buckets = Array.isArray(rec.buckets)
    ? rec.buckets
        .map(parseBucket)
        .filter((bucket): bucket is AgyUsageBucket => bucket !== undefined)
    : [];
  return { name, description: asString(rec.description), buckets };
}

/** Parse `agy --print /usage --output-format json` stdout (or a `stream-json` result envelope). */
export function parseAgyUsageJson(text: string): AgyUsageReport {
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    throw new Error("agy-usage: quota response was not valid JSON.");
  }
  const result = unwrapResult(parsed);
  const status = asString(result.status);
  if (status && status !== "SUCCESS" && status !== "OK") {
    throw new Error(asString(result.error) ?? `agy-usage: quota command failed (${status}).`);
  }
  const command = asRecord(result.command);
  if (command && asString(command.name) && command.name !== "usage") {
    throw new Error(`agy-usage: expected usage command, got "${asString(command.name)}".`);
  }
  const data = asRecord(command?.data) ?? asRecord(result.data);
  const groups = Array.isArray(data?.groups)
    ? data.groups.map(parseGroup).filter((group): group is AgyUsageGroup => group !== undefined)
    : [];
  if (groups.length === 0) {
    throw new Error("agy-usage: no quota groups returned.");
  }
  return { description: asString(data?.description), groups };
}

function formatCountdown(hours: number): string {
  if (hours * 60 < 59.5) return `${Math.max(1, Math.round(hours * 60))}m`;
  if (hours < 23.5) return `${Math.round(hours)}h`;
  return `${Math.round(hours / 24)}d`;
}

/** Remaining allowance as a clamped `0..100` integer percentage. */
export function remainingPercent(fraction: number): number {
  return Math.round(Math.min(1, Math.max(0, fraction)) * 100);
}

/** Progress bar visualization for a `0..1` remaining fraction. */
export function usageBar(fraction: number, segments = BAR_SEGMENTS): string {
  const filled = Math.round(Math.min(1, Math.max(0, fraction)) * segments);
  return `[${"█".repeat(filled)}${"░".repeat(segments - filled)}]`;
}

/** Normalized window label (`5h limit`, `Weekly limit`, or fallback bucket label). */
export function usageWindowLabel(bucket: AgyUsageBucket): string {
  const window = bucket.window.toLowerCase();
  if (window === "weekly" || window.includes("week")) return "Weekly limit";
  if (window === "5h" || window.includes("5") || window.includes("hour")) return "5h limit";
  const stripped = bucket.name.replace(/\s*limit remaining$/i, "").trim() || bucket.name;
  return /limit$/i.test(stripped) ? stripped : `${stripped} limit`;
}

function windowRank(bucket: AgyUsageBucket): number {
  const window = bucket.window.toLowerCase();
  if (window === "5h" || window.includes("hour") || /\b5h\b/.test(window)) return 0;
  if (window === "weekly" || window.includes("week")) return 1;
  return 2;
}

function padTime(value: number): string {
  return value.toString().padStart(2, "0");
}

/** Format an ISO reset timestamp (`HH:MM` today or `HH:MM on D Mon`), ignoring epoch sentinels. */
export function formatReset(iso: string, now: Date = new Date()): string | undefined {
  const reset = new Date(iso);
  if (Number.isNaN(reset.getTime()) || reset.getTime() <= 0) return undefined;
  const time = `${padTime(reset.getHours())}:${padTime(reset.getMinutes())}`;
  if (reset.toDateString() === now.toDateString()) return time;
  return `${time} on ${reset.getDate()} ${reset.toLocaleDateString("en-US", { month: "short" })}`;
}

/**
 * Compact one-line quota summary for the Pi status bar (e.g. `25m:88% · 7d:96%`).
 * Filters by active model group when `modelId` matches, otherwise takes worst per window.
 */
export function formatAgyUsageShort(
  report: AgyUsageReport,
  modelId?: string,
  now: Date = new Date(),
  paint?: (level: "warning" | "error", text: string) => string,
): string {
  const vendor = modelId?.split(/[-.]/)[0]?.toLowerCase();
  const matched = vendor
    ? report.groups.filter((group) =>
        `${group.name} ${group.description ?? ""}`.toLowerCase().includes(vendor),
      )
    : [];
  const groups = matched.length > 0 ? matched : report.groups;
  const byWindow = new Map<number, AgyUsageBucket>();
  for (const group of groups) {
    for (const bucket of group.buckets) {
      const rank = windowRank(bucket);
      const current = byWindow.get(rank);
      if (!current || bucket.remainingFraction < current.remainingFraction) {
        byWindow.set(rank, bucket);
      }
    }
  }
  return [...byWindow.entries()]
    .sort((a, b) => a[0] - b[0])
    .slice(0, 2)
    .map(([rank, bucket]) => {
      const fraction = bucket.remainingFraction;
      const percent = `${remainingPercent(fraction)}%`;
      const windowHours = WINDOW_HOURS[rank];
      const left = (new Date(bucket.resetTime ?? "").getTime() - now.getTime()) / 3_600_000;
      const timed = windowHours !== undefined && left > 0;
      const text = timed ? `${formatCountdown(left)}:${percent}` : `${bucket.window} ${percent}`;
      if (fraction < 0.1) return paint?.("error", text) ?? text;
      if (timed && fraction < left / windowHours) return paint?.("warning", text) ?? text;
      return text;
    })
    .join(" · ");
}

function formatBucket(bucket: AgyUsageBucket, now: Date): string {
  const label = `${usageWindowLabel(bucket)}:`.padEnd(LABEL_COLUMN);
  const percent = remainingPercent(bucket.remainingFraction);
  const parts = [`${usageBar(bucket.remainingFraction)} ${percent}% left`];
  const reset = bucket.resetTime ? formatReset(bucket.resetTime, now) : undefined;
  if (reset) parts.push(`resets ${reset}`);
  return `  ${label}${parts.join(" · ")}`;
}

/** Multi-line `/agy-usage` report showing progress bars, percentages, and reset times. */
export function formatAgyUsageReport(report: AgyUsageReport, now: Date = new Date()): string {
  return report.groups
    .map((group) => {
      const buckets = [...group.buckets].sort((a, b) => windowRank(a) - windowRank(b));
      const lines = [group.name];
      for (let i = 0; i < buckets.length; i++) {
        if (i > 0) lines.push("");
        lines.push(formatBucket(buckets[i], now));
      }
      return lines.join("\n");
    })
    .join("\n\n");
}

/** Build public `agy` CLI args to expand `/usage` non-interactively without disabling slash commands. */
export function buildAgyUsageArgs(timeoutMs = USAGE_TIMEOUT_MS): string[] {
  const timeout = Math.max(1, Math.ceil(timeoutMs / 1000));
  return ["--print", "/usage", "--output-format", "json", "--print-timeout", `${timeout}s`];
}

/** Run `agy --print /usage --output-format json` and parse the structured quota report. */
export async function fetchAgyUsage(options: FetchAgyUsageOptions = {}): Promise<AgyUsageReport> {
  const binary = options.binary ?? getAgyBinary();
  const timeoutMs = options.timeoutMs ?? USAGE_TIMEOUT_MS;
  const res = await execAgyCommand(binary, buildAgyUsageArgs(timeoutMs), {
    signal: options.signal,
    timeoutMs,
    maxBufferBytes: USAGE_MAX_BUFFER_BYTES,
    spawnProcess: options.spawnOverride,
    abortMessage: "Antigravity quota check aborted.",
    timeoutMessage: `agy-usage: timed out after ${timeoutMs}ms`,
    maxBufferMessage: "agy-usage: stdout maxBuffer exceeded",
  });
  if (res.code !== 0) {
    throw new Error(res.stderr.trim() || `${binary} exited with code ${res.code ?? "signal"}`);
  }
  return parseAgyUsageJson(res.stdout);
}
