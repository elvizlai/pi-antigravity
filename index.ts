import { randomUUID } from "node:crypto";
import type { Model } from "@earendil-works/pi-ai";
import {
  type ExtensionAPI,
  type ExtensionCommandContext,
  type ExtensionContext,
  getDocsPath,
  getExamplesPath,
  getReadmePath,
  type SessionEntry,
} from "@earendil-works/pi-coding-agent";
import { Text } from "@earendil-works/pi-tui";
import { Type } from "typebox";
import type { AgyUsage } from "./lib/acp-updates.ts";
import { listAgyArtifacts } from "./lib/artifacts.ts";
import { AgyPiBridge, selectBridgedTools } from "./lib/bridge.ts";
import { createBridgeLifecycleManager } from "./lib/bridge-lifecycle.ts";
import {
  AGY_COMPACTION_ENTRY,
  type AgyCompactionMarker,
  agyContextTokens,
  detectAgyCompaction,
  formatAgyContextTokens,
} from "./lib/compaction.ts";
import {
  AGY_CONVERSATION_STATE_ENTRY,
  agyConversationExists,
  type PersistedAgyConversation,
  type PersistedAgyReset,
  parsePersistedAgyState,
  restorableAgyConversation,
} from "./lib/conversation-state.ts";
import { checkAgyBinary, formatAgyDoctorReport } from "./lib/diagnostics.ts";
import { removeMcpCacheEntry } from "./lib/mcp-cache.ts";
import { createPrivateMcpConfig, sweepDeadBridgeArtifacts } from "./lib/mcp-config.ts";
import { getFallbackModels, mergeModelLists, probeAgyModels } from "./lib/models.ts";
import { createPiToolsAgent } from "./lib/pi-tools-agent.ts";
import {
  type AgyExecutionMode,
  DEFAULT_AGY_EXECUTION_MODE,
  toggleAgyPlanMode,
} from "./lib/profile.ts";
import { type PiDocsPaths, WRAPPER_TOOL_DESCRIPTION, WRAPPER_TOOL_NAME } from "./lib/prompt.ts";
import { formatAgyCall, summarizeAgyResult } from "./lib/render.ts";
import { AgyReplayStore } from "./lib/replay.ts";
import {
  ACTIVATE_SKILL_TOOL_NAME,
  activateSkillDescription,
  activateSkillParameters,
  handleActivateSkill,
  piPrivateSkills,
  type SkillLite,
} from "./lib/skills.ts";
import { type AgySubagentEntry, formatAgySubagents, trackAgySubagent } from "./lib/subagents.ts";
import { listAgyTasks } from "./lib/tasks.ts";
import {
  type AgyToolMode,
  formatAgyToolModeStatus,
  parseAgyToolModeArg,
  resolveAgyToolMode,
} from "./lib/tool-mode.ts";
import { loadSavedAgyToolMode, saveAgyToolMode } from "./lib/tool-mode-settings.ts";
import { type AgyUsageReport, fetchAgyUsage, formatAgyUsageShort } from "./lib/usage.ts";
import { wrapperToolActiveAfterModelSwitch } from "./lib/wrapper-activation.ts";
import { openAgyArtifactsPicker } from "./src/artifacts-ui.ts";
import { createAntigravityStreamSimple } from "./src/provider.ts";
import { AgyAcpRuntime } from "./src/runtime.ts";
import { openAgyTasksPicker } from "./src/tasks-ui.ts";
import { openAgyUsagePicker } from "./src/usage-ui.ts";

const PROVIDER_ID = "antigravity";
const USAGE_CACHE_TTL_MS = 5 * 60 * 1000;
const QUOTA_COOLDOWN_MS = 60 * 1000;

const WRAPPER_PARAMS = Type.Object({
  tool: Type.String(),
  input: Type.Optional(Type.Record(Type.String(), Type.Unknown())),
});

interface WrapperReplayDetails {
  agyTool: string;
  output?: string;
  error?: string;
  durationSeconds?: number;
}

function resolvePiDocsPaths(): PiDocsPaths {
  return {
    readme: getReadmePath(),
    docs: getDocsPath(),
    examples: getExamplesPath(),
  };
}

function parseExtraDirsFlag(raw: string | boolean | undefined): string[] {
  if (typeof raw !== "string" || !raw.trim()) return [];
  return raw
    .split(",")
    .map((part) => part.trim())
    .filter(Boolean);
}

function addUsage(acc: AgyUsage, delta: AgyUsage): AgyUsage {
  const sumOpt = (a?: number, b?: number) =>
    a === undefined && b === undefined ? undefined : (a ?? 0) + (b ?? 0);
  return {
    input_tokens: sumOpt(acc.input_tokens, delta.input_tokens),
    output_tokens: sumOpt(acc.output_tokens, delta.output_tokens),
    cache_read_tokens: sumOpt(acc.cache_read_tokens, delta.cache_read_tokens),
    cache_creation_tokens: sumOpt(acc.cache_creation_tokens, delta.cache_creation_tokens),
    reasoning_tokens: sumOpt(acc.reasoning_tokens, delta.reasoning_tokens),
    thinking_tokens: sumOpt(acc.thinking_tokens, delta.thinking_tokens),
    total_tokens: sumOpt(acc.total_tokens, delta.total_tokens),
  };
}

export default function antigravityExtension(pi: ExtensionAPI) {
  pi.registerFlag("agy-tools", {
    description: "Tool routing mode for Antigravity ACP: pi (default) or native",
    type: "string",
  });
  pi.registerFlag("agy-plan", {
    description: "Start Antigravity ACP in read-only plan mode",
    type: "boolean",
    default: false,
  });
  pi.registerFlag("agy-sandbox", {
    description: "Pass --sandbox to `agy acp`",
    type: "boolean",
    default: false,
  });
  pi.registerFlag("agy-add-dir", {
    description: "Comma-separated extra directories to pass via --add-dir to `agy acp`",
    type: "string",
  });

  const runtime = new AgyAcpRuntime();
  const replayStore = new AgyReplayStore();
  const mcpConfig = createPrivateMcpConfig();
  const piToolsAgent = createPiToolsAgent();
  const bridge = new AgyPiBridge(mcpConfig.serverName);
  const bridgeToken = randomUUID();
  bridge.requireToken(bridgeToken);

  let lastCtx: ExtensionContext | undefined;
  let models: Model<string>[] = getFallbackModels();
  let toolMode: AgyToolMode = resolveAgyToolMode(loadSavedAgyToolMode());
  let executionMode: AgyExecutionMode = DEFAULT_AGY_EXECUTION_MODE;
  let conversationId: string | undefined;
  let turnCount = 0;
  let accumulatedUsage: AgyUsage = {};
  let lastContextTokens: number | undefined;
  let missingConversationNotice: string | undefined;
  let privateSkills: SkillLite[] = [];
  let mcpWrittenForNative = false;
  let quotaExhaustedUntilMs = 0;
  let lastUsageReport: AgyUsageReport | undefined;
  let nextUsageFetchAt = 0;
  let activeModelId = models[0]?.id ?? "gemini-4-neon";
  const subagentsRoster = new Map<string, AgySubagentEntry>();

  bridge.setOnCall((call) => runtime.notifyBridgeCall(call));

  const getAllToolsSnapshot = () => {
    try {
      return pi.getAllTools();
    } catch {
      return [];
    }
  };

  const getActiveToolsSnapshot = () => {
    try {
      return pi.getActiveTools();
    } catch {
      return [];
    }
  };

  bridge.setToolSource(() =>
    selectBridgedTools(getAllToolsSnapshot(), getActiveToolsSnapshot(), toolMode),
  );

  const syncBridgeDynamicSkills = (skills: SkillLite[], cwd: string | undefined) => {
    privateSkills = piPrivateSkills(skills, cwd);
    if (privateSkills.length === 0) {
      bridge.setDynamicTools([]);
      return;
    }
    bridge.setDynamicTools([
      {
        name: ACTIVATE_SKILL_TOOL_NAME,
        description: activateSkillDescription(privateSkills),
        parameters: activateSkillParameters(privateSkills),
        handler: (args) => handleActivateSkill(privateSkills, args),
      },
    ]);
  };

  const bridgeManager = createBridgeLifecycleManager({
    bridge,
    bridgeToken,
    writeMcpConfig: async (_serverName, url, token) => {
      if (toolMode === "pi") {
        mcpWrittenForNative = false;
        await mcpConfig.remove().catch(() => {});
        await piToolsAgent.prepare({ serverName: bridge.serverName, url, token });
      } else {
        await piToolsAgent.remove().catch(() => {});
        await mcpConfig.write(url, token);
        mcpWrittenForNative = true;
      }
    },
    removeMcpConfig: async () => {
      mcpWrittenForNative = false;
      await Promise.allSettled([mcpConfig.remove(), piToolsAgent.remove()]);
    },
    evictMcpCache: (serverName) => removeMcpCacheEntry(serverName),
    notifyWarning: (message) => {
      lastCtx?.ui.notify(message, "warning");
    },
  });

  const updateFooterStatus = (ctx: ExtensionContext | undefined) => {
    if (!ctx) return;
    if (ctx.model?.provider !== PROVIDER_ID) {
      ctx.ui.setStatus(PROVIDER_ID, undefined);
      return;
    }
    const modeLabel = formatAgyToolModeStatus(toolMode);
    const planSuffix = executionMode === "plan" ? " · plan" : "";
    const usageSuffix =
      lastUsageReport && Date.now() < nextUsageFetchAt
        ? ` · ${formatAgyUsageShort(lastUsageReport, ctx.model?.id)}`
        : "";
    ctx.ui.setStatus(PROVIDER_ID, `${modeLabel}${planSuffix}${usageSuffix}`);
  };

  const syncWrapperTool = (ctx: ExtensionContext | undefined) => {
    if (!ctx) return;
    const nextTools = wrapperToolActiveAfterModelSwitch(
      getActiveToolsSnapshot(),
      WRAPPER_TOOL_NAME,
      ctx.model?.provider,
      PROVIDER_ID,
    );
    if (nextTools) {
      pi.setActiveTools([...nextTools]);
    }
  };

  const ensureNativeBridgePrepared = async (notify?: (message: string) => void) => {
    const hasNativeBridgedItems =
      selectBridgedTools(getAllToolsSnapshot(), getActiveToolsSnapshot(), "native").length > 0 ||
      privateSkills.length > 0;
    if (hasNativeBridgedItems) {
      await bridgeManager.ensureRegistered(notify);
    }
  };

  const registerProvider = () => {
    pi.registerProvider(PROVIDER_ID, {
      name: "Antigravity",
      baseUrl: "acp://agy",
      apiKey: "antigravity-acp",
      api: "antigravity-acp",
      models: models.map((m) => ({
        id: m.id,
        name: m.name,
        api: m.api,
        baseUrl: m.baseUrl,
        reasoning: m.reasoning,
        input: [...m.input],
        cost: m.cost,
        contextWindow: m.contextWindow,
        maxTokens: m.maxTokens,
      })),
      streamSimple: createAntigravityStreamSimple({
        runtime,
        replayStore,
        bridge,
        bridgeManager,
        piDocs: resolvePiDocsPaths(),
        getToolMode: () => toolMode,
        getExecutionMode: () => executionMode,
        getCwd: () => lastCtx?.cwd ?? process.cwd(),
        getBridgeRevision: () =>
          `${toolMode}:${bridgeManager.processRevision()}:${piToolsAgent.isPrepared()}`,
        getConversationId: () => conversationId,
        consumeMissingConversationNotice: () => {
          const msg = missingConversationNotice;
          missingConversationNotice = undefined;
          return msg;
        },
        ensurePiToolsAgent: async (modelId) => {
          const ok = await bridgeManager.ensureRegistered();
          if (!ok || !bridge.url) return undefined;
          await piToolsAgent.prepare({
            serverName: bridge.serverName,
            url: bridge.url,
            token: bridgeToken,
          });
          return piToolsAgent.nameForModel(modelId);
        },
        prepareNativeBridge: () =>
          ensureNativeBridgePrepared((msg) => lastCtx?.ui.notify(msg, "warning")),
        getExtraDirs: (mode) => {
          const userDirs = parseExtraDirsFlag(pi.getFlag("agy-add-dir"));
          if (mode === "pi") {
            // Never pass `--add-dir <mcpDirectory>` or `--add-dir <cwd>` in pi mode.
            return userDirs;
          }
          return mcpWrittenForNative ? [mcpConfig.directory, ...userDirs] : userDirs;
        },
        isSandbox: () => pi.getFlag("agy-sandbox") === true,
        onMissingConversation: (missingId) => {
          conversationId = undefined;
          lastContextTokens = undefined;
          missingConversationNotice =
            `[Notice] Previous Antigravity conversation "${missingId}" could not be resumed. ` +
            "Continue from the restored Pi transcript below.";
        },
        onConversationResolved: ({ sessionId, resumed }) => {
          const changed = sessionId !== conversationId;
          conversationId = sessionId;
          if (!resumed || changed) {
            if (!resumed) {
              turnCount = 0;
              accumulatedUsage = {};
            }
          }
          if (lastCtx) {
            const payload: PersistedAgyConversation = {
              version: 1,
              kind: "conversation",
              sessionId: lastCtx.sessionManager.getSessionId(),
              conversationId: sessionId,
              cwd: lastCtx.cwd,
              modelId: lastCtx.model?.id ?? activeModelId,
              toolMode,
              bridgeServer: bridge.serverName,
              turns: turnCount,
              usage: accumulatedUsage,
              contextTokens: lastContextTokens,
            };
            pi.appendEntry(AGY_CONVERSATION_STATE_ENTRY, payload);
          }
        },
        onQuotaExhausted: () => {
          quotaExhaustedUntilMs = Date.now() + QUOTA_COOLDOWN_MS;
          nextUsageFetchAt = 0;
        },
        onUsage: (usage) => {
          turnCount += 1;
          accumulatedUsage = addUsage(accumulatedUsage, usage);
          const nextContextTokens = agyContextTokens(usage);
          const compaction = detectAgyCompaction(lastContextTokens, nextContextTokens);
          if (compaction && nextContextTokens !== undefined) {
            const marker: AgyCompactionMarker = {
              version: 1,
              beforeTokens: compaction.beforeTokens,
              afterTokens: compaction.afterTokens,
              detectedAt: new Date().toISOString(),
            };
            pi.appendEntry(AGY_COMPACTION_ENTRY, marker);
          }
          if (nextContextTokens !== undefined) {
            lastContextTokens = nextContextTokens;
          }
        },
        onActivity: (activity) => {
          trackAgySubagent(subagentsRoster, activity);
        },
      }),
    });
  };

  registerProvider();

  pi.registerTool({
    name: WRAPPER_TOOL_NAME,
    label: WRAPPER_TOOL_NAME,
    description: WRAPPER_TOOL_DESCRIPTION,
    parameters: WRAPPER_PARAMS,
    async execute(toolCallId, params) {
      const recorded = replayStore.take(toolCallId);
      const agyTool = recorded?.agyTool ?? params.tool;
      const details: WrapperReplayDetails = {
        agyTool,
        output: recorded?.output,
        error: recorded?.error,
        durationSeconds: recorded?.durationSeconds,
      };
      const text = recorded?.error ?? recorded?.output ?? "(completed)";
      return {
        content: [{ type: "text", text }],
        isError: Boolean(recorded?.error),
        details,
        // Display-only replay cards must not trigger a follow-up ACP turn.
        terminate: true,
      };
    },
    renderCall(args, theme) {
      return new Text(formatAgyCall(args.tool, args.input, theme), 0, 0);
    },
    renderResult(result, _options, theme) {
      const details = result.details as WrapperReplayDetails | undefined;
      const agyTool = details?.agyTool ?? WRAPPER_TOOL_NAME;
      const summary = summarizeAgyResult(agyTool, details?.output);
      if (details?.error) {
        return new Text(theme.fg("error", details.error), 0, 0);
      }
      if (summary.counts) {
        return new Text(theme.fg("muted", summary.counts), 0, 0);
      }
      const firstText = result.content.find((c) => c.type === "text")?.text;
      if (firstText && firstText !== "(completed)") {
        const preview = firstText.split("\n")[0] ?? "";
        const clipped = preview.length > 100 ? `${preview.slice(0, 99)}…` : preview;
        return new Text(theme.fg("toolOutput", clipped), 0, 0);
      }
      return new Text("", 0, 0);
    },
  });

  pi.registerEntryRenderer<AgyCompactionMarker>(AGY_COMPACTION_ENTRY, (entry, _options, theme) => {
    const data = entry.data;
    if (!data) return undefined;
    const text = `antigravity compacted context: ${formatAgyContextTokens(data.beforeTokens)} -> ${formatAgyContextTokens(data.afterTokens)}`;
    return new Text(theme.fg("muted", text), 0, 0);
  });

  pi.registerEntryRenderer<PersistedAgyConversation | PersistedAgyReset>(
    AGY_CONVERSATION_STATE_ENTRY,
    (entry, _options, theme) => {
      const parsed = parsePersistedAgyState(entry.data);
      if (!parsed) return undefined;
      const text =
        parsed.kind === "reset"
          ? "antigravity conversation reset"
          : `antigravity conversation ${parsed.conversationId.slice(0, 8)} (${parsed.turns} turns)`;
      return new Text(theme.fg("dim", text), 0, 0);
    },
  );

  const restoreSessionState = async (ctx: ExtensionContext) => {
    lastCtx = ctx;
    if (ctx.model?.id) activeModelId = ctx.model.id;
    subagentsRoster.clear();

    const flagMode = pi.getFlag("agy-tools");
    if (typeof flagMode === "string") {
      const parsed = parseAgyToolModeArg(flagMode);
      if (parsed.kind === "set") {
        toolMode = parsed.mode;
      }
    }
    if (pi.getFlag("agy-plan") === true) {
      executionMode = "plan";
    }

    const entries: readonly SessionEntry[] = ctx.sessionManager.getEntries();
    const restored = restorableAgyConversation(
      entries,
      ctx.sessionManager.getSessionId(),
      ctx.cwd,
      toolMode,
      bridge.serverName,
    );
    if (restored && (await agyConversationExists(restored.conversationId))) {
      conversationId = restored.conversationId;
      turnCount = restored.turns;
      accumulatedUsage = restored.usage;
      lastContextTokens = restored.contextTokens;
    } else {
      conversationId = undefined;
      turnCount = 0;
      accumulatedUsage = {};
      lastContextTokens = undefined;
    }

    syncWrapperTool(ctx);
    updateFooterStatus(ctx);
  };

  pi.on("session_start", async (_event, ctx) => {
    void sweepDeadBridgeArtifacts();
    await restoreSessionState(ctx);
    void probeAgyModels()
      .then((probed) => {
        if (probed.length > 0) {
          models = mergeModelLists(probed, getFallbackModels());
          registerProvider();
        }
      })
      .catch(() => {});
  });

  pi.on("model_select", (event, ctx) => {
    lastCtx = ctx;
    activeModelId = event.model.id;
    syncWrapperTool(ctx);
    updateFooterStatus(ctx);
  });

  pi.on("session_tree", async (_event, ctx) => {
    await runtime.resetConversation("Session tree branch switched");
    await restoreSessionState(ctx);
  });

  pi.on("session_compact", async (_event, ctx) => {
    lastCtx = ctx;
    updateFooterStatus(ctx);
  });

  pi.on("before_agent_start", async (event, ctx) => {
    lastCtx = ctx;
    if (ctx.model?.provider !== PROVIDER_ID) return;

    const skills = (event.systemPromptOptions?.skills ?? []) as SkillLite[];
    syncBridgeDynamicSkills(skills, ctx.cwd);
    bridge.refreshTools();
    bridgeManager.warnMcpAdapterUnsupported(getAllToolsSnapshot(), (msg) =>
      ctx.ui.notify(msg, "warning"),
    );

    if (toolMode === "pi") {
      await bridgeManager.ensureRegistered((msg) => ctx.ui.notify(msg, "warning"));
      bridgeManager.warnSkillsUnavailable(privateSkills);
    } else {
      await ensureNativeBridgePrepared((msg) => ctx.ui.notify(msg, "warning"));
      bridgeManager.warnSkillsUnavailable(privateSkills);
    }

    if (Date.now() >= nextUsageFetchAt && Date.now() >= quotaExhaustedUntilMs) {
      void fetchAgyUsage()
        .then((report) => {
          lastUsageReport = report;
          nextUsageFetchAt = Date.now() + USAGE_CACHE_TTL_MS;
          updateFooterStatus(lastCtx);
        })
        .catch(() => {
          nextUsageFetchAt = Date.now() + 60_000;
        });
    }
  });

  pi.on("session_shutdown", async () => {
    await runtime.stop("Session shutdown");
    await bridgeManager.teardown();
  });

  pi.registerCommand("agy-tools", {
    description:
      "Switch Antigravity tool routing between `pi` (MCP bridge) and `native` (built-in tools)",
    handler: async (args, ctx: ExtensionCommandContext) => {
      lastCtx = ctx;
      const action = parseAgyToolModeArg(args);
      if (action.kind === "status") {
        ctx.ui.notify(
          `Antigravity tools mode: ${formatAgyToolModeStatus(toolMode)} (default: pi)`,
          "info",
        );
        return;
      }
      if (action.kind === "invalid") {
        ctx.ui.notify(
          `Invalid /agy-tools argument "${action.raw}". Use: pi | native | on | off | status`,
          "warning",
        );
        return;
      }
      if (action.mode === toolMode) {
        ctx.ui.notify(
          `Antigravity tools already set to ${formatAgyToolModeStatus(toolMode)}.`,
          "info",
        );
        return;
      }
      toolMode = action.mode;
      await saveAgyToolMode(toolMode);
      bridge.refreshTools();
      await runtime.stop("Tool mode changed");
      await bridgeManager.teardown();
      updateFooterStatus(ctx);
      ctx.ui.notify(`Switched Antigravity tools to ${formatAgyToolModeStatus(toolMode)}.`, "info");
    },
  });

  pi.registerCommand("agy-plan", {
    description: "Toggle Antigravity ACP read-only plan mode",
    handler: async (_args, ctx: ExtensionCommandContext) => {
      lastCtx = ctx;
      executionMode = toggleAgyPlanMode(executionMode);
      updateFooterStatus(ctx);
      ctx.ui.notify(`Antigravity ACP execution mode: ${executionMode}`, "info");
    },
  });

  pi.registerCommand("agy-reset", {
    description: "Reset the bound Antigravity ACP conversation and start fresh on next turn",
    handler: async (_args, ctx: ExtensionCommandContext) => {
      lastCtx = ctx;
      await runtime.resetConversation("User ran /agy-reset");
      conversationId = undefined;
      turnCount = 0;
      accumulatedUsage = {};
      lastContextTokens = undefined;
      subagentsRoster.clear();
      const resetEntry: PersistedAgyReset = {
        version: 1,
        kind: "reset",
        sessionId: ctx.sessionManager.getSessionId(),
        cwd: ctx.cwd,
      };
      pi.appendEntry(AGY_CONVERSATION_STATE_ENTRY, resetEntry);
      ctx.ui.notify("Antigravity ACP conversation reset.", "info");
    },
  });

  pi.registerCommand("agy-doctor", {
    description: "Check `agy acp` binary, bridge health, execution mode, and quota status",
    handler: async (_args, ctx: ExtensionCommandContext) => {
      lastCtx = ctx;
      let binary: string | undefined;
      let version: string | undefined;
      let binaryError: string | undefined;
      try {
        const info = await checkAgyBinary();
        binary = info.binary;
        version = info.version;
      } catch (err) {
        binaryError = err instanceof Error ? err.message : String(err);
      }
      const report = formatAgyDoctorReport({
        binary,
        version,
        binaryError,
        toolMode: formatAgyToolModeStatus(toolMode),
        executionMode,
        conversationId,
        bridgeStatus: bridge.running
          ? `listening (${bridge.url ?? "loopback"}, ${bridge.pendingCount} pending)`
          : "stopped",
        acpConnected: runtime.isConnected,
        mcpServerCount: getAllToolsSnapshot().filter((t) => t.sourceInfo?.path === "builtin:mcp")
          .length,
        quotaError:
          Date.now() < quotaExhaustedUntilMs ? "Quota exhausted recently; cooling down" : undefined,
      });
      ctx.ui.notify(report, binaryError ? "error" : "info");
    },
  });

  pi.registerCommand("agy-tasks", {
    description: "Inspect background tasks spawned by Antigravity for the current conversation",
    handler: async (_args, ctx: ExtensionCommandContext) => {
      lastCtx = ctx;
      if (!conversationId) {
        ctx.ui.notify("No active Antigravity conversation yet.", "info");
        return;
      }
      if (!ctx.hasUI) {
        const tasks = await listAgyTasks(conversationId);
        ctx.ui.notify(
          tasks.length > 0
            ? tasks
                .map((t) => `• ${t.id} (${t.pids.length > 0 ? t.pids.join(",") : "done"})`)
                .join("\n")
            : "No agy background tasks.",
          "info",
        );
        return;
      }
      await openAgyTasksPicker(ctx, () => listAgyTasks(conversationId ?? ""));
    },
  });

  pi.registerCommand("agy-artifacts", {
    description: "Browse generated artifacts in the current Antigravity conversation",
    handler: async (_args, ctx: ExtensionCommandContext) => {
      lastCtx = ctx;
      if (!conversationId) {
        ctx.ui.notify("No active Antigravity conversation yet.", "info");
        return;
      }
      if (!ctx.hasUI) {
        const artifacts = await listAgyArtifacts(conversationId);
        ctx.ui.notify(
          artifacts.length > 0
            ? artifacts.map((a) => `• ${a.name} (${a.absolutePath})`).join("\n")
            : "No agy artifacts.",
          "info",
        );
        return;
      }
      await openAgyArtifactsPicker(ctx, () => listAgyArtifacts(conversationId ?? ""));
    },
  });

  pi.registerCommand("agy-subagents", {
    description: "Show tracked Antigravity subagent calls from this session",
    handler: async (_args, ctx: ExtensionCommandContext) => {
      lastCtx = ctx;
      ctx.ui.notify(
        formatAgySubagents(subagentsRoster) ?? "No Antigravity subagents recorded yet.",
        "info",
      );
    },
  });

  pi.registerCommand("agy-usage", {
    description: "Check Antigravity quota buckets",
    handler: async (_args, ctx: ExtensionCommandContext) => {
      lastCtx = ctx;
      await openAgyUsagePicker(ctx, async (signal) => {
        const report = await fetchAgyUsage({ signal });
        lastUsageReport = report;
        nextUsageFetchAt = Date.now() + USAGE_CACHE_TTL_MS;
        updateFooterStatus(ctx);
        return report;
      });
    },
  });

  pi.registerCommand("agy-models", {
    description: "Refresh available Antigravity models via `agy models`",
    handler: async (_args, ctx: ExtensionCommandContext) => {
      lastCtx = ctx;
      try {
        const probed = await probeAgyModels();
        models = mergeModelLists(probed, getFallbackModels());
        registerProvider();
        ctx.ui.notify(`Loaded ${models.length} Antigravity ACP models.`, "info");
      } catch (err) {
        ctx.ui.notify(
          `Failed to refresh Antigravity models: ${err instanceof Error ? err.message : String(err)}`,
          "error",
        );
      }
    },
  });
}
