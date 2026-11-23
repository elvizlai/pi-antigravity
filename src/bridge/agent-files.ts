import { randomUUID } from "node:crypto";
import { mkdir, readdir, rm, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import path from "node:path";
import type { PiToolInfo } from "./server.ts";

const PI_TOOLS_AGENT_DESCRIPTION =
  "Complete Pi requests using the current Pi session tools through its MCP bridge.";

export const MCP_ADAPTER_UNSUPPORTED_WARNING =
  "antigravity: pi-mcp-adapter is installed, but its MCP tools are not bridged to agy. " +
  "It also replaces Pi's built-in MCP support. Remove or disable pi-mcp-adapter, " +
  'move your servers to ~/.pi/agent/mcp.json (or .pi/mcp.json) with "exposure": "direct", ' +
  "then run /reload.";

function piToolsAgentPrompt(serverName: string): string {
  return [
    "# System Prompt",
    "You are the agent for the current Pi coding session. Complete the user request using Pi tools.",
    `Your tools are provided by the MCP server ${serverName}. Use the exact MCP tool names and schemas it exposes.`,
    "Relayed Pi instructions refer to tools by their Pi names; match them to the tools exposed by this bridge.",
    "Use the Pi bridge for file access, editing, commands, web access, skills, and other work. Pi handles tool execution, permissions, and tool results.",
    "Image outputs from bridged Pi tools (such as read on image files) are delivered inline as image attachments immediately after the tool call completes; inspect the attached images directly without re-calling the tool.",
    "Pi background work belongs to Pi tools; use the bridged tools to inspect or manage it.",
  ].join("\n\n");
}

export interface PiToolsAgentBridge {
  serverName: string;
}

export interface PiToolsAgentOptions {
  /** Override the agent file directory (defaults to `~/.gemini/config/agents`). */
  agentsDir?: string;
  /** Override the generated agent name (defaults to `pi-tools-<pid>-<uuid>`). */
  agentName?: string;
}

export interface PiToolsAgentHandle {
  agentName: string;
  agentDir: string;
  agentFile: string;
  write(): Promise<void>;
  remove(): Promise<void>;
}

const PI_TOOLS_AGENT_PATTERN = /^pi-tools-(\d+)-/i;
const MCP_ADAPTER_SOURCE = /(?:^|[:/\\])pi-mcp-adapter(?=$|[@/\\.])/;

function defaultIsPidDead(pid: number): boolean {
  if (!Number.isInteger(pid) || pid <= 0 || pid === process.pid) return false;
  try {
    process.kill(pid, 0);
    return false;
  } catch (err) {
    return (err as NodeJS.ErrnoException)?.code === "ESRCH";
  }
}

/**
 * Format an `agy --agent` definition (`~/.gemini/config/agents/<name>/agent.md`).
 *
 * Uses strictly documented agent frontmatter fields (`name`, `description`,
 * `mainAgent`, `subagent`, `tools`) to restrict built-in tools while the loopback
 * HTTP MCP server is attached via standard ACP `session/new` and `session/resume`.
 */
function formatPiToolsAgentFile(agentName: string, bridge: PiToolsAgentBridge): string {
  const definition = {
    name: agentName,
    description: PI_TOOLS_AGENT_DESCRIPTION,
    mainAgent: true,
    subagent: false,
    tools: [],
  };

  return `---\n${JSON.stringify(definition, null, 2)}\n---\n\n${piToolsAgentPrompt(bridge.serverName)}\n`;
}

/**
 * Create a static agy agent markdown definition (`~/.gemini/config/agents/<name>/agent.md`)
 * that scopes `agy` built-in tools and points prompts at this Pi process's ACP MCP bridge.
 */
export function createPiToolsAgent(
  bridge: PiToolsAgentBridge,
  options?: PiToolsAgentOptions,
): PiToolsAgentHandle {
  const agentsDir = options?.agentsDir ?? path.join(homedir(), ".gemini", "config", "agents");
  const agentName = options?.agentName ?? `pi-tools-${process.pid}-${randomUUID()}`;
  const agentDir = path.join(agentsDir, agentName);
  const agentFile = path.join(agentDir, "agent.md");

  return {
    agentName,
    agentDir,
    agentFile,
    async write(): Promise<void> {
      await mkdir(agentDir, { recursive: true, mode: 0o700 });
      await writeFile(agentFile, formatPiToolsAgentFile(agentName, bridge), {
        mode: 0o600,
      });
    },
    async remove(): Promise<void> {
      await rm(agentDir, { recursive: true, force: true });
    },
  };
}

/**
 * Best-effort cleanup of stale `pi-tools-<pid>-*` custom agent directories left
 * behind by terminated Pi processes under `~/.gemini/config/agents`.
 */
export async function sweepDeadPiToolsAgents(options?: {
  agentsRoot?: string;
  isPidDead?: (pid: number) => boolean;
}): Promise<void> {
  const agentsRoot = options?.agentsRoot ?? path.join(homedir(), ".gemini", "config", "agents");
  const isPidDead = options?.isPidDead ?? defaultIsPidDead;

  try {
    const entries = await readdir(agentsRoot);
    await Promise.all(
      entries.map(async (entry) => {
        const match = PI_TOOLS_AGENT_PATTERN.exec(entry);
        if (!match || !isPidDead(Number(match[1]))) return;
        await rm(path.join(agentsRoot, entry), { recursive: true, force: true }).catch(() => {});
      }),
    );
  } catch {
    // Missing or unreadable agents directory is non-fatal.
  }
}

/** True when any registered Pi tool originates from `pi-mcp-adapter`. */
export function hasUnsupportedMcpAdapter(
  resources: readonly Pick<PiToolInfo, "sourceInfo">[],
): boolean {
  return resources.some(({ sourceInfo }) =>
    [sourceInfo?.source, sourceInfo?.path].some((value) => MCP_ADAPTER_SOURCE.test(value ?? "")),
  );
}

export const HINDSIGHT_DISABLE_HOOKS_ENV = "HINDSIGHT_DISABLE_HOOKS";

const HOST_HINDSIGHT_TOOL_NAMES = new Set([
  "hindsight_search_pages",
  "hindsight_reflect",
  "hindsight_diagnose",
]);

/** True when the outer Pi host has already registered Hindsight memory tools. */
export function hasHostHindsightTools(tools: readonly Pick<PiToolInfo, "name">[]): boolean {
  return tools.some((tool) => HOST_HINDSIGHT_TOOL_NAMES.has(tool.name));
}

/**
 * Suppress `agy`'s own Hindsight `PreInvocation`/`Stop` hooks when outer Pi already
 * runs Hindsight, preventing duplicate prompt injection and double transcript retention.
 */
export function resolveAgyChildEnv(options: {
  hostHindsightActive?: boolean;
  env?: Record<string, string>;
  ambientEnv?: NodeJS.ProcessEnv;
}): Record<string, string> | undefined {
  if (!options.hostHindsightActive) return options.env;
  const ambient = options.ambientEnv ?? process.env;
  if (
    options.env?.[HINDSIGHT_DISABLE_HOOKS_ENV] !== undefined ||
    ambient[HINDSIGHT_DISABLE_HOOKS_ENV] !== undefined
  ) {
    return options.env;
  }
  return { [HINDSIGHT_DISABLE_HOOKS_ENV]: "1", ...options.env };
}
