import { randomUUID } from "node:crypto";
import { mkdir, readdir, rm, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import path from "node:path";
import { getAgentDir } from "@earendil-works/pi-coding-agent";
import { isAgyOrchestratorModel } from "../acp/models.ts";
import { type AgyPiBridge, BRIDGE_MCP_TIMEOUT_SECONDS, type PiToolInfo } from "./server.ts";
import type { SkillLite } from "./skills.ts";

export const PI_TOOLS_AGENT_DESCRIPTION =
  "Complete Pi requests using the current Pi session tools through its MCP bridge.";

export function piToolsAgentPrompt(serverName: string): string {
  return [
    "# System Prompt",
    "You are the agent for the current Pi coding session. Complete the user request using Pi tools (when <plan_execute_protocol> is active, bridged edits and mutating commands still go through exit_plan_mode).",
    `Your tools are provided by the MCP server ${serverName}. Use the exact MCP tool names and schemas it exposes.`,
    "Relayed Pi instructions refer to tools by their Pi names; match them to the tools exposed by this bridge.",
    "Use the Pi bridge for file access, editing, commands, web access, skills, and other work. Pi handles tool execution, permissions, and tool results.",
    "Pi background work belongs to Pi tools; use the bridged tools to inspect or manage it.",
  ].join("\n\n");
}

/**
 * Built-in phase-handoff tools required by agy's Orchestrator models
 * (`planexecute.checkPreconditions` disables the two-phase Planner -> Worker
 * pipeline unless both are allowed). Non-orchestrator models have no
 * `execution_model` configured and fail executor construction if either is
 * declared, so standard and Orchestrator models use separate static agents.
 */
export const PI_TOOLS_ORCHESTRATOR_TOOLS = ["enter_plan_mode", "exit_plan_mode"] as const;

export interface PiToolsAgentBridge {
  serverName: string;
  url: string;
  token: string;
}

export interface PiToolsAgentOptions {
  /** Override the agent file directory (defaults to `~/.gemini/config/agents`). */
  agentsDir?: string;
  /** Override the generated agent name (defaults to `pi-tools-<pid>-<uuid>`). */
  agentName?: string;
  /** Include orchestrator phase-handoff tools (`enter_plan_mode`, `exit_plan_mode`). */
  orchestrator?: boolean;
  /** Model ID used to auto-detect whether orchestrator phase-handoff tools are needed. */
  modelId?: string;
}

export interface PiToolsAgentHandle {
  agentName: string;
  agentFile: string;
  write(): Promise<void>;
  remove(): Promise<void>;
}

const BRIDGE_DIR_PATTERN =
  /^(\d+)-([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})$/i;
const PI_TOOLS_AGENT_PATTERN = /^pi-tools-(\d+)-[0-9a-f-]+(?:-orchestrator)?(?:\.md)?$/i;

function defaultBridgesRoot(): string {
  return path.join(getAgentDir(), "antigravity", "bridges");
}

function defaultIsPidDead(pid: number): boolean {
  if (!Number.isInteger(pid) || pid <= 0 || pid === process.pid) return false;
  try {
    process.kill(pid, 0);
    return false;
  } catch (err) {
    return (err as NodeJS.ErrnoException)?.code === "ESRCH";
  }
}

function yamlString(value: string): string {
  return JSON.stringify(value);
}

function formatPiToolsAgentFile(
  agentName: string,
  bridge: PiToolsAgentBridge,
  orchestrator: boolean,
): string {
  const mcpServerJson = JSON.stringify({
    url: bridge.url,
    headers: { "x-pi-bridge-token": bridge.token },
    timeoutSeconds: BRIDGE_MCP_TIMEOUT_SECONDS,
  });

  const lines = [
    "---",
    `name: ${yamlString(agentName)}`,
    `description: ${yamlString(PI_TOOLS_AGENT_DESCRIPTION)}`,
  ];

  if (orchestrator) {
    lines.push("tools:");
    for (const tool of PI_TOOLS_ORCHESTRATOR_TOOLS) {
      lines.push(`  - ${tool}`);
    }
  } else {
    lines.push("tools: []");
  }

  lines.push(
    "mcpServers:",
    `  ${yamlString(bridge.serverName)}: ${mcpServerJson}`,
    "---",
    piToolsAgentPrompt(bridge.serverName),
    "",
  );

  return lines.join("\n");
}

/**
 * Create a static agy agent markdown definition (`~/.gemini/config/agents/<name>.md`)
 * that wires only this Pi process's HTTP MCP bridge.
 */
export function createPiToolsAgent(
  bridge: PiToolsAgentBridge,
  options?: PiToolsAgentOptions,
): PiToolsAgentHandle {
  const orchestrator = options?.orchestrator ?? isAgyOrchestratorModel(options?.modelId);
  const agentsDir = options?.agentsDir ?? path.join(homedir(), ".gemini", "config", "agents");
  const agentName =
    options?.agentName ??
    `pi-tools-${process.pid}-${randomUUID()}${orchestrator ? "-orchestrator" : ""}`;
  const agentFile = path.join(agentsDir, `${agentName}.md`);

  return {
    agentName,
    agentFile,
    async write(): Promise<void> {
      await mkdir(agentsDir, { recursive: true, mode: 0o700 });
      await writeFile(agentFile, formatPiToolsAgentFile(agentName, bridge, orchestrator), {
        mode: 0o600,
      });
    },
    async remove(): Promise<void> {
      await rm(agentFile, { force: true });
    },
  };
}

/** One process's workspace MCP discovery directory, passed only via `cwd` / `--add-dir` if used. */
export function createPrivateMcpConfig(root = defaultBridgesRoot()) {
  const id = randomUUID();
  const serverName = `pi-${id.replaceAll("-", "").slice(0, 12)}`;
  const directory = path.join(root, `${process.pid}-${id}`);
  const agentsDirectory = path.join(directory, ".agents");

  return {
    serverName,
    directory,
    async write(url: string, token: string): Promise<void> {
      await mkdir(agentsDirectory, { recursive: true, mode: 0o700 });
      await writeFile(
        path.join(agentsDirectory, "mcp_config.json"),
        `${JSON.stringify(
          {
            mcpServers: {
              [serverName]: {
                serverUrl: url,
                headers: { "x-pi-bridge-token": token },
                timeoutSeconds: BRIDGE_MCP_TIMEOUT_SECONDS,
              },
            },
          },
          null,
          2,
        )}\n`,
        { mode: 0o600 },
      );
    },
    async remove(): Promise<void> {
      await rm(directory, { recursive: true, force: true });
    },
  };
}

/**
 * Best-effort cleanup of bridge workspaces and `pi-tools-*` agent files
 * left behind by terminated Pi processes. Uses public Pi and agent config dirs only.
 */
export async function sweepDeadBridgeArtifacts(options?: {
  bridgesRoot?: string;
  agentsRoot?: string;
  isPidDead?: (pid: number) => boolean;
}): Promise<void> {
  const bridgesRoot = options?.bridgesRoot ?? defaultBridgesRoot();
  const agentsRoot = options?.agentsRoot ?? path.join(homedir(), ".gemini", "config", "agents");
  const isPidDead = options?.isPidDead ?? defaultIsPidDead;

  try {
    const entries = await readdir(bridgesRoot);
    await Promise.all(
      entries.map(async (entry) => {
        const match = BRIDGE_DIR_PATTERN.exec(entry);
        if (!match || !isPidDead(Number(match[1]))) return;
        await rm(path.join(bridgesRoot, entry), { recursive: true, force: true }).catch(() => {});
      }),
    );
  } catch {
    // Missing or unreadable bridges directory is non-fatal.
  }

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

export interface BridgeLifecycleDeps {
  bridge: AgyPiBridge;
  bridgeToken: string;
  writeMcpConfig: (serverName: string, url: string, token: string) => Promise<void>;
  removeMcpConfig: (serverName: string) => Promise<void>;
  enabled?: boolean;
  notifyWarning?: (message: string) => void;
}

export interface BridgeLifecycleManager {
  readonly isRegistered: () => boolean;
  readonly isRunning: () => boolean;
  readonly ensureRegistered: (notify?: (message: string) => void) => Promise<boolean>;
  readonly teardown: () => Promise<void>;
  readonly registrationGeneration: () => number;
  readonly processRevision: () => string;
  readonly warnMcpAdapterUnsupported: (
    resources: readonly Pick<PiToolInfo, "sourceInfo">[],
    notify?: (message: string) => void,
  ) => void;
  readonly warnSkillsUnavailable: (skills: SkillLite[]) => void;
}

export function createBridgeLifecycleManager(deps: BridgeLifecycleDeps): BridgeLifecycleManager {
  const enabled = deps.enabled ?? true;
  let registered = false;
  let ownsConfig = false;
  let generation = 0;
  let preparing: Promise<boolean> | undefined;
  let tearingDown: Promise<void> | undefined;
  let warnedSkillsUnavailable = false;
  let warnedMcpAdapterUnsupported = false;

  const cleanup = async () => {
    if (!registered && !ownsConfig && !deps.bridge.running) return;
    registered = false;
    let removalError: unknown;
    try {
      await deps.removeMcpConfig(deps.bridge.serverName);
      ownsConfig = false;
    } catch (error) {
      removalError = error;
    }
    try {
      await deps.bridge.close();
    } finally {
      generation += 1;
    }
    if (removalError) throw removalError;
  };

  const prepare = async (notify?: (message: string) => void) => {
    try {
      await deps.bridge.start();
      const url = deps.bridge.url;
      if (!url) throw new Error("pi-tool bridge did not expose a URL after start.");
      ownsConfig = true;
      await deps.writeMcpConfig(deps.bridge.serverName, url, deps.bridgeToken);
      deps.bridge.refreshTools();
      registered = true;
      generation += 1;
      return true;
    } catch (error) {
      let message = error instanceof Error ? error.message : String(error);
      try {
        await cleanup();
      } catch (cleanupError) {
        message += `; cleanup failed: ${cleanupError instanceof Error ? cleanupError.message : String(cleanupError)}`;
      }
      const warning = `antigravity: pi-tool bridge unavailable (${message}).`;
      (notify ?? deps.notifyWarning)?.(warning);
      return false;
    }
  };

  return {
    isRegistered: () => registered,
    isRunning: () => deps.bridge.running,
    registrationGeneration: () => generation,
    processRevision: () => `${generation}:${deps.bridge.catalogRevision}`,

    ensureRegistered: async (notify) => {
      if (!enabled) return false;
      if (tearingDown) await tearingDown;
      if (registered) return true;
      preparing ??= prepare(notify).finally(() => {
        preparing = undefined;
      });
      return preparing;
    },

    teardown: async () => {
      tearingDown ??= (async () => {
        await preparing;
        await cleanup();
      })().finally(() => {
        tearingDown = undefined;
      });
      await tearingDown;
    },

    warnMcpAdapterUnsupported: (resources, notify) => {
      if (!enabled || warnedMcpAdapterUnsupported) return;
      const adapterSource = /(?:^|[:/\\])pi-mcp-adapter(?=$|[@/\\.])/;
      const installed = resources.some(({ sourceInfo }) =>
        [sourceInfo?.source, sourceInfo?.path].some((value) => adapterSource.test(value ?? "")),
      );
      const warn = notify ?? deps.notifyWarning;
      if (!installed || !warn) return;
      warnedMcpAdapterUnsupported = true;
      warn(
        "antigravity: pi-mcp-adapter is installed, but its MCP tools are not bridged to agy. " +
          "It also replaces Pi's built-in MCP support. Remove or disable pi-mcp-adapter, " +
          'move your servers to ~/.pi/agent/mcp.json (or .pi/mcp.json) with "exposure": "direct", ' +
          "then run /reload.",
      );
    },

    warnSkillsUnavailable: (skills) => {
      if (warnedSkillsUnavailable || skills.length === 0) return;
      if (enabled && registered) return;
      warnedSkillsUnavailable = true;
      deps.notifyWarning?.(
        enabled
          ? "antigravity: pi-tool bridge failed to prepare — pi-private skills are unavailable to agy this session."
          : "antigravity: pi-tool bridge is disabled — pi-private skills are unavailable to agy this session.",
      );
    },
  };
}
