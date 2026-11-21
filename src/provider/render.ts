/**
 * Standard-ACP rendering for wrapped native `agy` tool cards and status updates.
 *
 * Uses strictly public ACP metadata (`kind`, `title`, `locations`) and opaque
 * `rawInput` fallbacks — no internal tool-name tables or private parameter keys.
 */

import os from "node:os";
import {
  BorderedLoader,
  type ExtensionCommandContext,
  type Theme,
} from "@earendil-works/pi-coding-agent";
import type { AgyActivity } from "../acp/events.ts";
import { type AgyUsageReport, formatAgyUsageReport } from "../acp/usage.ts";

const STANDARD_ACP_KINDS = new Set([
  "read",
  "edit",
  "delete",
  "move",
  "search",
  "execute",
  "think",
  "fetch",
  "switch_mode",
]);

export interface AcpCallRenderInput {
  tool: string;
  kind?: string;
  locations?: ReadonlyArray<{ path: string; line?: number }>;
  input?: unknown;
}

function clipOneLine(value: string | undefined, max = 56): string | undefined {
  if (!value) return undefined;
  const collapsed = value.replace(/\s+/g, " ").trim();
  if (!collapsed) return undefined;
  return collapsed.length > max ? `${collapsed.slice(0, max - 1)}…` : collapsed;
}

/** Shorten an absolute path under the home directory to `~/...` like pi's own renderers. */
function shortenPath(filePath: string): string {
  // `os.homedir()` also resolves on Windows, where `$HOME` is unset.
  const home = os.homedir();
  if (home && filePath.startsWith(home)) return `~${filePath.slice(home.length)}`;
  return filePath;
}

function asRecord(input: unknown): Record<string, unknown> {
  return typeof input === "object" && input !== null && !Array.isArray(input)
    ? (input as Record<string, unknown>)
    : {};
}

function formatGenericArgs(input: Record<string, unknown>, theme: Theme): string {
  const entries = Object.entries(input).filter(
    ([, value]) => value !== undefined && value !== null && value !== "",
  );
  if (entries.length === 0) return "";
  const parts = entries.slice(0, 3).map(([key, value]) => {
    let rendered: string;
    if (typeof value === "string") {
      rendered = theme.fg("accent", `"${clipOneLine(value, 48) ?? ""}"`);
    } else if (typeof value === "number" || typeof value === "boolean") {
      rendered = theme.fg("accent", String(value));
    } else if (Array.isArray(value)) {
      rendered = theme.fg("muted", `[${value.length} item${value.length === 1 ? "" : "s"}]`);
    } else {
      rendered = theme.fg("muted", "{…}");
    }
    return `${theme.fg("toolOutput", key)} ${rendered}`;
  });
  let line = parts.join(theme.fg("toolOutput", ", "));
  if (entries.length > 3) line += theme.fg("muted", " …");
  return line;
}

function formatLocations(
  locations: ReadonlyArray<{ path: string; line?: number }> | undefined,
  title: string,
  theme: Theme,
): string {
  if (!locations || locations.length === 0) return "";
  const formatted: string[] = [];
  for (const loc of locations) {
    const short = shortenPath(loc.path);
    if (title.includes(loc.path) || title.includes(short)) continue;
    const withLine = loc.line !== undefined ? `${short}:${loc.line}` : short;
    if (!formatted.includes(withLine)) formatted.push(withLine);
  }
  if (formatted.length === 0) return "";
  const shown = formatted.slice(0, 2).map((text) => theme.fg("accent", text));
  const extra = formatted.length > 2 ? theme.fg("muted", ` (+${formatted.length - 2})`) : "";
  return `${shown.join(theme.fg("toolOutput", ", "))}${extra}`;
}

/** One-line ACP tool summary built strictly from `kind`, `title`, `locations`, and opaque `input`. */
export function formatAgyCall(call: AcpCallRenderInput, theme: Theme): string {
  const title = clipOneLine(call.tool, 72) ?? "tool";
  const normalizedKind = call.kind?.trim().toLowerCase();
  const hasKind = Boolean(normalizedKind && STANDARD_ACP_KINDS.has(normalizedKind));

  const parts: string[] = [];
  if (hasKind && normalizedKind) {
    parts.push(theme.fg("toolTitle", theme.bold(normalizedKind)));
    if (title.toLowerCase() !== normalizedKind) {
      parts.push(theme.fg("accent", title));
    }
  } else {
    parts.push(theme.fg("toolTitle", theme.bold(title)));
  }

  const locText = formatLocations(call.locations, title, theme);
  if (locText) {
    parts.push(locText);
  } else if (!call.locations || call.locations.length === 0) {
    const generic = formatGenericArgs(asRecord(call.input), theme);
    if (generic) parts.push(generic);
  }

  return parts.join(" ");
}

export interface AgyResultSummary {
  /** e.g. "10 lines". */
  counts?: string;
}

/** Derive a generic line-count summary for standard ACP `search` tool results. */
export function summarizeAgyResult(
  kind: string | undefined,
  output: string | undefined,
): AgyResultSummary {
  const lines = (output ?? "").split("\n").filter((line) => line.trim());
  if (kind?.trim().toLowerCase() === "search" && lines.length > 0) {
    const count = lines.length;
    return { counts: `${count} line${count === 1 ? "" : "s"}` };
  }
  return {};
}

/**
 * Compute the next active-tool list after a model switch, or `undefined` when
 * the wrapper's activation already matches the selected provider.
 */
export function wrapperToolActiveAfterModelSwitch(
  activeTools: readonly string[],
  wrapperTool: string,
  provider: string | undefined,
  wrapperProvider: string,
): readonly string[] | undefined {
  const want = provider === wrapperProvider;
  const has = activeTools.includes(wrapperTool);
  if (want === has) return undefined;
  return want ? [...activeTools, wrapperTool] : activeTools.filter((name) => name !== wrapperTool);
}

/** Format a human-readable `/agy-doctor` diagnostic summary. */
export function formatAgyDoctorReport(input: {
  binary?: string;
  version?: string;
  binaryError?: string;
  toolMode: string;
  executionMode: string;
  conversationId?: string;
  bridgeStatus: string;
  acpConnected: boolean;
  mcpServerCount: number;
  quotaError?: string;
}): string {
  const lines = ["Antigravity ACP Diagnostics (/agy-doctor)"];
  if (input.binaryError) {
    lines.push(`- Binary: ERROR — ${input.binaryError}`);
  } else {
    lines.push(`- Binary: ${input.binary ?? "unknown"} (v${input.version ?? "?"})`);
  }
  lines.push(`- Transport: agy acp (${input.acpConnected ? "connected" : "idle"})`);
  lines.push(`- Tool mode: ${input.toolMode}`);
  lines.push(`- Execution mode: ${input.executionMode}`);
  lines.push(`- Conversation: ${input.conversationId ?? "(new on next turn)"}`);
  lines.push(`- Pi tool bridge: ${input.bridgeStatus}`);
  lines.push(`- Active bridged Pi tools: ${input.mcpServerCount}`);
  if (input.quotaError) {
    lines.push(`- Quota warning: ${input.quotaError}`);
  }
  return lines.join("\n");
}

/** Track active ACP tool calls by standard `title` for Pi's working spinner. */
export class AgyWorkingStatusTracker {
  #inFlight = new Map<string, string>();

  onActivity(activity: AgyActivity): string | undefined {
    if (activity.type === "tool_start" || activity.type === "tool_update") {
      this.#inFlight.set(activity.callId, clipOneLine(activity.title, 56) ?? "Working");
    } else if (activity.type === "tool_done" || activity.type === "tool_error") {
      this.#inFlight.delete(activity.callId);
    }

    const latest = [...this.#inFlight.values()].at(-1);
    return latest ? `${latest}…` : undefined;
  }
}

const USAGE_REFRESH_ACTION = "Refresh";
const USAGE_CLOSE_ACTION = "Close";

type UsageLoaderResult = { ok: true; value: AgyUsageReport } | { ok: false; error: unknown } | null;

function isAbortError(error: unknown): boolean {
  return error instanceof Error && error.name === "AbortError";
}

async function loadQuotaWithOverlay(
  ctx: ExtensionCommandContext,
  label: string,
  parentSignal: AbortSignal,
  fetchReport: (signal?: AbortSignal) => Promise<AgyUsageReport>,
): Promise<AgyUsageReport | undefined> {
  if (ctx.mode !== "tui") return fetchReport(parentSignal);
  const result = await ctx.ui.custom<UsageLoaderResult>((tui, theme, _keybindings, done) => {
    const loader = new BorderedLoader(tui, theme, label, { cancellable: true });
    let finished = false;
    const finish = (value: UsageLoaderResult) => {
      if (finished) return;
      finished = true;
      done(value);
    };
    loader.onAbort = () => finish(null);
    const signal = AbortSignal.any([parentSignal, loader.signal]);
    void fetchReport(signal)
      .then((value) => finish({ ok: true, value }))
      .catch((error) => {
        if (isAbortError(error)) finish(null);
        else finish({ ok: false, error });
      });
    return loader;
  });
  if (!result) return undefined;
  if (!result.ok) throw result.error;
  return result.value;
}

/** Open the `/agy-usage` quota picker with a cancellable loader and Refresh / Close actions. */
export async function openAgyUsagePicker(
  ctx: ExtensionCommandContext,
  fetchReport: (signal?: AbortSignal) => Promise<AgyUsageReport>,
): Promise<void> {
  if (!ctx.hasUI) {
    try {
      ctx.ui.notify(formatAgyUsageReport(await fetchReport()), "info");
    } catch (error) {
      ctx.ui.notify(
        `agy-usage: ${error instanceof Error ? error.message : String(error)}`,
        "error",
      );
    }
    return;
  }

  const controller = new AbortController();
  try {
    let report = await loadQuotaWithOverlay(
      ctx,
      "Checking agy quota…",
      controller.signal,
      fetchReport,
    );
    if (!report) return;
    while (!controller.signal.aborted) {
      const action = await ctx.ui.select(
        formatAgyUsageReport(report),
        [USAGE_REFRESH_ACTION, USAGE_CLOSE_ACTION],
        { signal: controller.signal },
      );
      if (!action || action === USAGE_CLOSE_ACTION) return;
      if (action === USAGE_REFRESH_ACTION) {
        try {
          const refreshed = await loadQuotaWithOverlay(
            ctx,
            "Refreshing usage…",
            controller.signal,
            fetchReport,
          );
          if (refreshed) report = refreshed;
        } catch (error) {
          if (isAbortError(error)) return;
          ctx.ui.notify(
            `agy-usage: ${error instanceof Error ? error.message : String(error)}`,
            "error",
          );
        }
      }
    }
  } catch (error) {
    if (isAbortError(error)) return;
    ctx.ui.notify(`agy-usage: ${error instanceof Error ? error.message : String(error)}`, "error");
  } finally {
    controller.abort();
  }
}
