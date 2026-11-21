/**
 * Native-style rendering for agy tool cards.
 *
 * Call lines mirror pi's built-in tools (`find <pattern> in <path>`,
 * `grep "<query>" in <path>`, `search_web "<query>"`, …) instead of raw JSON,
 * and result lines summarize counts where the output allows it. All strings
 * are single-line; pi's Text component handles wrapping and width safety.
 */

import os from "node:os";
import {
  BorderedLoader,
  type ExtensionCommandContext,
  type Theme,
} from "@earendil-works/pi-coding-agent";
import type { AgyActivity } from "../acp/events.ts";
import { type AgyUsageReport, formatAgyUsageReport } from "../acp/usage.ts";

/** agy tool name → native-equivalent label shown on the card. */
const LABELS: Record<string, string> = {
  run_command: "bash",
  view_file: "read",
  read_file: "read",
  write_file: "write",
  write_to_file: "write",
  edit_file: "edit",
  replace_file_content: "edit",
  multi_replace_file_content: "edit",
  sed_file: "edit",
  list_dir: "ls",
  find_by_name: "find",
  grep_search: "grep",
  invoke_subagent: "subagent",
};

export function agyToolLabel(tool: string): string {
  return LABELS[tool] ?? tool;
}

/** First defined string among the given keys (agy mixes key casings). */
function pickArg(input: Record<string, unknown>, keys: string[]): string | undefined {
  for (const key of keys) {
    const value = input[key];
    if (typeof value === "string" && value.trim()) return value;
  }
  return undefined;
}

function clipOneLine(value: string | undefined, max = 56): string | undefined {
  if (!value) return undefined;
  const collapsed = value.replace(/\s+/g, " ").trim();
  if (!collapsed) return undefined;
  return collapsed.length > max ? `${collapsed.slice(0, max - 1)}…` : collapsed;
}

/** Shorten an absolute path under the home directory to `~/...` like pi's own renderers. */
function shortenPath(path: string): string {
  // `os.homedir()` also resolves on Windows, where `$HOME` is unset.
  const home = os.homedir();
  if (home && path.startsWith(home)) return `~${path.slice(home.length)}`;
  return path;
}

function asRecord(input: unknown): Record<string, unknown> {
  return typeof input === "object" && input !== null ? (input as Record<string, unknown>) : {};
}

/**
 * Generic bounded key/value summary for tools without a dedicated shape
 * (subagent orchestration, scheduling, media, …).
 */
function formatGenericArgs(input: Record<string, unknown>, theme: Theme): string {
  const entries = Object.entries(input).filter(
    ([, value]) => value !== undefined && value !== null && value !== "",
  );
  if (entries.length === 0) return "";
  const bound = (text: string, max = 48) =>
    text.length > max ? `${text.slice(0, max - 1)}…` : text;
  const parts = entries.slice(0, 3).map(([key, value]) => {
    const rendered =
      typeof value === "string"
        ? theme.fg("accent", `"${bound(value)}"`)
        : theme.fg("accent", bound(JSON.stringify(value) ?? "", 24));
    return `${theme.fg("toolOutput", key)} ${rendered}`;
  });
  let line = parts.join(theme.fg("toolOutput", ", "));
  if (entries.length > 3) line += theme.fg("muted", " …");
  return line;
}

/**
 * One-line call summary, aligned with pi's native tool cards:
 * bold label + accent arguments, no dot prefix, no JSON dump.
 */
export function formatAgyCall(tool: string, input: unknown, theme: Theme): string {
  const args = asRecord(input);
  const label = theme.fg("toolTitle", theme.bold(agyToolLabel(tool)));
  const invalid = theme.fg("error", "[invalid arg]");
  const path = (fallback: string) => {
    const raw = pickArg(args, [
      "path",
      "Path",
      "AbsolutePath",
      "TargetFile",
      "target_file",
      "FilePath",
      "file_path",
      "SearchPath",
      "SearchDirectory",
      "DirectoryPath",
      "directory",
      "Directory",
      "url",
      "Url",
      "URL",
    ]);
    if (raw === undefined) return fallback;
    return theme.fg("accent", shortenPath(raw));
  };

  switch (tool) {
    case "search_web": {
      const query = pickArg(args, ["query", "Query", "search_query", "q"]);
      return `${label} ${query === undefined ? invalid : theme.fg("accent", `"${query}"`)}`;
    }
    case "read_url_content": {
      const url = pickArg(args, ["url", "Url", "URL"]);
      return `${label} ${url === undefined ? invalid : theme.fg("accent", url)}`;
    }
    case "run_command": {
      const command = pickArg(args, ["command", "Command", "cmd", "CommandLine", "commandline"]);
      return `${label} ${command === undefined ? invalid : theme.fg("accent", command)}`;
    }
    case "view_file":
    case "read_file":
    case "write_file":
    case "write_to_file":
    case "edit_file":
    case "replace_file_content":
    case "multi_replace_file_content":
    case "sed_file":
    case "list_dir":
      return `${label} ${path(invalid)}`;
    case "find_by_name": {
      const pattern = pickArg(args, ["pattern", "Pattern", "glob", "name"]);
      const dir = pickArg(args, [
        "search_directory",
        "SearchDirectory",
        "path",
        "Path",
        "directory",
      ]);
      const patternText = pattern === undefined ? invalid : theme.fg("accent", pattern);
      const dirText = dir === undefined ? "" : theme.fg("toolOutput", ` in ${shortenPath(dir)}`);
      return `${label} ${patternText}${dirText}`;
    }
    case "grep_search": {
      const query = pickArg(args, ["query", "Query", "pattern", "Pattern"]);
      const searchPath = pickArg(args, ["search_path", "SearchPath", "path", "Path"]);
      const queryText = query === undefined ? invalid : theme.fg("accent", `"${query}"`);
      const pathText =
        searchPath === undefined ? "" : theme.fg("toolOutput", ` in ${shortenPath(searchPath)}`);
      return `${label} ${queryText}${pathText}`;
    }
    case "ask_question": {
      const question = pickArg(args, ["question", "Question", "prompt", "Prompt", "text"]);
      return `${label} ${question === undefined ? invalid : theme.fg("accent", `"${question}"`)}`;
    }
    case "generate_image": {
      const prompt = pickArg(args, ["prompt", "Prompt", "description", "Description"]);
      return `${label} ${prompt === undefined ? invalid : theme.fg("accent", `"${prompt}"`)}`;
    }
    case "invoke_subagent": {
      const typeName = pickArg(args, [
        "TypeName",
        "typeName",
        "name",
        "Name",
        "agent",
        "Agent",
        "type",
      ]);
      const role = clipOneLine(pickArg(args, ["Role", "role"]), 40);
      const model = pickArg(args, ["Model", "model"]);
      const prompt = clipOneLine(pickArg(args, ["Prompt", "prompt"]), 48);
      if (!typeName && !role && !prompt) return `${label} ${invalid}`;
      const parts = [label];
      if (typeName) parts.push(theme.fg("accent", typeName));
      if (role) parts.push(theme.fg("accent", `"${role}"`));
      else if (prompt) parts.push(theme.fg("toolOutput", `"${prompt}"`));
      if (model && model !== "inherit") parts.push(theme.fg("muted", `(${model})`));
      return parts.join(" ");
    }
    case "define_subagent": {
      const name = pickArg(args, ["Name", "name"]);
      const desc = clipOneLine(pickArg(args, ["Description", "description"]), 48);
      if (!name && !desc) return `${label} ${invalid}`;
      const parts = [label];
      if (name) parts.push(theme.fg("accent", name));
      if (desc) parts.push(theme.fg("toolOutput", `"${desc}"`));
      return parts.join(" ");
    }
    case "manage_subagents": {
      const action = pickArg(args, ["Action", "action"]);
      const ids = Array.isArray(args.ConversationIds)
        ? args.ConversationIds.filter((v): v is string => typeof v === "string")
        : [];
      if (!action) return `${label} ${invalid}`;
      const suffix = ids.length > 0 ? theme.fg("muted", ` (${ids.length})`) : "";
      return `${label} ${theme.fg("accent", action)}${suffix}`;
    }
    case "send_message": {
      const to = pickArg(args, [
        "Recipient",
        "recipient",
        "to",
        "To",
        "subagent",
        "Subagent",
        "target",
      ]);
      const message = clipOneLine(pickArg(args, ["Message", "message", "text", "Text"]), 56);
      if (!to && !message) return `${label} ${invalid}`;
      const parts = [label];
      if (to) parts.push(theme.fg("accent", to));
      if (message) parts.push(theme.fg("toolOutput", `"${message}"`));
      return parts.join(" ");
    }
    case "swarm": {
      const action = pickArg(args, ["action", "Action"]);
      const name = pickArg(args, ["name", "Name"]);
      const title = clipOneLine(pickArg(args, ["title", "Title"]), 48);
      const brief = clipOneLine(pickArg(args, ["brief", "Brief"]), 48);
      const model = pickArg(args, ["model", "Model"]);
      const count =
        typeof args.n === "number" && args.n > 1
          ? args.n
          : Array.isArray(args.tasks) && args.tasks.length > 1
            ? args.tasks.length
            : undefined;
      if (!action) return `${label} ${invalid}`;
      const parts = [label, theme.fg("accent", action)];
      if (name) parts.push(theme.fg("accent", `${name}${count ? `×${count}` : ""}`));
      if (action === "title" && title) {
        parts.push(theme.fg("accent", `"${title}"`));
      } else if (brief) {
        parts.push(theme.fg("toolOutput", `"${brief}"`));
      }
      if (model) parts.push(theme.fg("muted", `(${model})`));
      return parts.join(" ");
    }
    case "run_workflow": {
      const desc = clipOneLine(pickArg(args, ["Description", "description"]), 56);
      const scriptPath = pickArg(args, ["ScriptPath", "scriptPath"]);
      const summary = desc
        ? theme.fg("accent", `"${desc}"`)
        : scriptPath
          ? theme.fg("accent", shortenPath(scriptPath))
          : theme.fg("muted", "(inline script)");
      return `${label} ${summary}`;
    }
    case "schedule": {
      const dur = typeof args.DurationSeconds === "number" ? `${args.DurationSeconds}s` : undefined;
      const cron = pickArg(args, ["CronExpression", "cronExpression"]);
      const prompt = clipOneLine(pickArg(args, ["Prompt", "prompt"]), 48);
      const when = cron ?? dur;
      if (!when && !prompt) return `${label} ${invalid}`;
      const parts = [label];
      if (when) parts.push(theme.fg("accent", when));
      if (prompt) parts.push(theme.fg("toolOutput", `"${prompt}"`));
      return parts.join(" ");
    }
    case "manage_task": {
      const action = pickArg(args, ["action", "Action"]);
      const task = pickArg(args, ["task_id", "TaskId", "task", "Task", "id"]);
      if (action === undefined && task === undefined) return `${label} ${invalid}`;
      const actionText = action === undefined ? "" : theme.fg("accent", action);
      const taskText = task === undefined ? "" : theme.fg("accent", ` ${task}`);
      return `${label} ${actionText}${taskText}`;
    }
    case "exit_plan_mode":
    case "enter_plan_mode": {
      const summary = pickArg(args, ["reason", "Reason", "plan", "Plan"])
        ?.split("\n")
        .map((line) => line.replace(/^#+\s*/, "").trim())
        .find(Boolean);
      if (!summary) return label;
      const clipped = summary.length > 72 ? `${summary.slice(0, 71)}…` : summary;
      return `${label} ${theme.fg("accent", `"${clipped}"`)}`;
    }
    default: {
      const generic = formatGenericArgs(args, theme);
      return generic ? `${label} ${generic}` : label;
    }
  }
}

export interface AgyResultSummary {
  /** e.g. "10 matches in 4 files", "3 results", "no results". */
  counts?: string;
}

/** Derive a count summary from ripgrep/fd-style line output when possible. */
export function summarizeAgyResult(tool: string, output: string | undefined): AgyResultSummary {
  const lines = (output ?? "").split("\n").filter((line) => line.trim());
  if (tool === "grep_search" && lines.length > 0) {
    const files = new Set<string>();
    for (const line of lines) {
      const sep = line.indexOf(":");
      if (sep > 0) files.add(line.slice(0, sep));
    }
    const matches = lines.length;
    const fileCount = files.size;
    return {
      counts:
        `${matches} match${matches === 1 ? "" : "es"}` +
        (fileCount > 0 ? ` in ${fileCount} file${fileCount === 1 ? "" : "s"}` : ""),
    };
  }
  if (tool === "find_by_name") {
    const n = lines.length;
    return { counts: n > 0 ? `${n} result${n === 1 ? "" : "s"}` : "no results" };
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

/** Track in-flight native tools and background orchestration state for Pi's working spinner. */
export class AgyWorkingStatusTracker {
  #inFlight = new Map<string, { name: string; args: Record<string, unknown> }>();
  #subagents: string[] = [];
  #swarmWorkers = new Map<string, string>();

  onActivity(activity: AgyActivity): string | undefined {
    if (activity.type === "tool_start" || activity.type === "tool_update") {
      this.#inFlight.set(activity.callId, {
        name: activity.name,
        args: asRecord(activity.args),
      });
    } else if (activity.type === "tool_done") {
      this.#inFlight.delete(activity.callId);
      this.#recordOrchestrationDone(activity.name, asRecord(activity.args));
    } else if (activity.type === "tool_error") {
      this.#inFlight.delete(activity.callId);
    }

    return this.#formatCurrentStatus();
  }

  #recordOrchestrationDone(name: string, args: Record<string, unknown>): void {
    if (name === "invoke_subagent") {
      const label =
        clipOneLine(pickArg(args, ["Role", "role", "TypeName", "typeName", "name", "Name"]), 32) ??
        "subagent";
      this.#subagents.push(label);
      return;
    }

    if (name === "manage_subagents") {
      if (pickArg(args, ["Action", "action"]) === "kill_all") {
        this.#subagents.length = 0;
      }
      return;
    }

    if (name === "swarm") {
      const action = pickArg(args, ["action", "Action"]);
      const workerName = pickArg(args, ["name", "Name"]);
      if (!workerName) return;
      if (action === "add") {
        const count =
          typeof args.n === "number" && args.n > 1
            ? args.n
            : Array.isArray(args.tasks) && args.tasks.length > 1
              ? args.tasks.length
              : undefined;
        this.#swarmWorkers.set(workerName, `${workerName}${count ? `×${count}` : ""}`);
      } else if (action === "kill") {
        this.#swarmWorkers.delete(workerName);
      }
    }
  }

  #formatCurrentStatus(): string | undefined {
    const latest = [...this.#inFlight.values()].at(-1);
    if (latest) {
      return this.#formatInFlight(latest.name, latest.args);
    }

    if (this.#swarmWorkers.size > 0) {
      const workers = [...this.#swarmWorkers.values()];
      const list = workers.slice(0, 3).join(", ") + (workers.length > 3 ? ", …" : "");
      return `Swarm running: ${list}…`;
    }

    if (this.#subagents.length > 0) {
      const list = this.#subagents.slice(-2).join(", ");
      return `Subagent running: ${list}…`;
    }

    return undefined;
  }

  #formatInFlight(name: string, args: Record<string, unknown>): string {
    switch (name) {
      case "run_workflow": {
        const desc = clipOneLine(pickArg(args, ["Description", "description"]), 40);
        return `Running workflow${desc ? `: ${desc}` : ""}…`;
      }
      case "run_command": {
        const cmd = clipOneLine(
          pickArg(args, ["command", "Command", "cmd", "CommandLine", "commandline"]),
          40,
        );
        return `Running bash${cmd ? `: ${cmd}` : ""}…`;
      }
      case "invoke_subagent": {
        const roleOrType = clipOneLine(
          pickArg(args, ["Role", "role", "TypeName", "typeName", "name", "Name"]),
          36,
        );
        return `Starting subagent${roleOrType ? `: ${roleOrType}` : ""}…`;
      }
      case "swarm": {
        const action = pickArg(args, ["action", "Action"]);
        return `Updating swarm${action ? ` (${action})` : ""}…`;
      }
      case "search_web": {
        const query = clipOneLine(pickArg(args, ["query", "Query", "search_query", "q"]), 40);
        return `Searching web${query ? `: "${query}"` : ""}…`;
      }
      default:
        return `Running ${agyToolLabel(name)}…`;
    }
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
