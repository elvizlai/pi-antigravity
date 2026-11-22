import { randomUUID } from "node:crypto";
import path from "node:path";
import type { Api, Model } from "@earendil-works/pi-ai";
import type {
  ExtensionAPI,
  ExtensionContext,
  ProviderModelConfig,
} from "@earendil-works/pi-coding-agent";
import { Text } from "@earendil-works/pi-tui";
import { Type } from "typebox";
import { checkAgyBinary } from "./src/acp/binary.ts";
import {
  getFallbackModels,
  mergeModelLists,
  modelCacheIsFresh,
  modelsFromCachedEntries,
  probeAgyModels,
} from "./src/acp/models.ts";
import { type AgyUsageReport, fetchAgyUsage, formatAgyUsageShort } from "./src/acp/usage.ts";
import {
  hasHostHindsightTools,
  hasUnsupportedMcpAdapter,
  MCP_ADAPTER_UNSUPPORTED_WARNING,
  sweepDeadPiToolsAgents,
} from "./src/bridge/agent-files.ts";
import { AgyPiBridge, selectBridgedTools } from "./src/bridge/server.ts";
import {
  ACTIVATE_SKILL_TOOL_NAME,
  activateSkillDescription,
  activateSkillParameters,
  handleActivateSkill,
  piPrivateSkills,
  type SkillLite,
} from "./src/bridge/skills.ts";
import {
  BRIDGE_PENDING_TOOL_MESSAGE,
  WRAPPER_TOOL_DESCRIPTION,
  WRAPPER_TOOL_NAME,
} from "./src/provider/prompt.ts";
import {
  formatAgyCall,
  formatAgyDoctorReport,
  openAgyUsagePicker,
  summarizeAgyResult,
  wrapperToolActiveAfterModelSwitch,
} from "./src/provider/render.ts";
import { AgyRuntime } from "./src/provider/runtime.ts";
import {
  type AgyExecutionMode,
  type AgyToolMode,
  formatAgyFooterStatus,
  formatAgyToolModeStatus,
  loadAgySettings,
  parseAgyPlanArg,
  parseAgyToolModeArg,
  parseAgyToolModeFlag,
  resolveAgyExecutionMode,
  resolveAgyToolMode,
  saveAgySettings,
} from "./src/provider/settings.ts";
import {
  AGY_STATE_ENTRY_TYPE,
  type AgyConversationState,
  type AgyPersistedState,
  parseAgyStateEntry,
} from "./src/provider/state.ts";
import { createAntigravityStreamSimple } from "./src/provider/stream.ts";
import { AgyReplayStore } from "./src/provider/turn.ts";

const PROVIDER_ID = "antigravity";

function toProviderModelConfigs(models: readonly Model<string>[]): ProviderModelConfig[] {
  return models.map((m) => ({
    id: m.id,
    name: m.name,
    api: m.api as Api,
    baseUrl: m.baseUrl,
    reasoning: m.reasoning,
    input: [...m.input],
    cost: m.cost,
    contextWindow: m.contextWindow,
    maxTokens: m.maxTokens,
  }));
}

function parseExtraDirs(raw: unknown): string[] {
  if (typeof raw !== "string") return [];
  return raw
    .split(",")
    .map((entry) => entry.trim())
    .filter(Boolean);
}

export default function antigravityExtension(pi: ExtensionAPI): void {
  pi.registerFlag("agy-tools", {
    type: "string",
    description: "Antigravity tool routing mode (pi | agy)",
  });
  pi.registerFlag("agy-plan", {
    type: "boolean",
    description: "Run Antigravity ACP sessions in plan execution mode",
    default: false,
  });
  pi.registerFlag("agy-sandbox", {
    type: "boolean",
    description: "Run agy acp with --sandbox",
    default: false,
  });
  pi.registerFlag("agy-add-dir", {
    type: "string",
    description: "Comma-separated extra directories passed via agy --add-dir",
  });

  const savedSettings = loadAgySettings();
  let toolMode: AgyToolMode = resolveAgyToolMode(savedSettings.toolMode);
  let executionMode: AgyExecutionMode = "default";
  let sandbox = false;
  let extraDirs: string[] = [];
  let cliFlagsApplied = false;

  const applyCliFlagsOnce = () => {
    if (cliFlagsApplied) return;
    cliFlagsApplied = true;
    const flagToolMode = parseAgyToolModeFlag(pi.getFlag("agy-tools"));
    if (flagToolMode) {
      toolMode = flagToolMode;
    }
    if (pi.getFlag("agy-plan") === true) {
      executionMode = "plan";
    }
    sandbox = Boolean(pi.getFlag("agy-sandbox"));
    extraDirs = parseExtraDirs(pi.getFlag("agy-add-dir"));
  };

  const bridgeToken = randomUUID();
  const bridgeServerName = `pi-${randomUUID().replaceAll("-", "").slice(0, 12)}`;
  const bridge = new AgyPiBridge(bridgeServerName);
  bridge.requireToken(bridgeToken);
  bridge.setToolSource(() => selectBridgedTools(pi.getAllTools(), pi.getActiveTools(), toolMode));

  const USAGE_SUMMARY_TTL_MS = 5 * 60_000;
  const USAGE_ERROR_BACKOFF_MS = 60_000;

  const replayStore = new AgyReplayStore();
  let activeCtx: ExtensionContext | undefined;
  let pendingState: AgyConversationState | undefined;
  let runtime: AgyRuntime | undefined;
  let usageReport: AgyUsageReport | undefined;
  let usageFetchInFlight = false;
  let nextUsageFetchAt = 0;
  let privateSkills: SkillLite[] = [];
  let bridgeStartPromise: Promise<boolean> | undefined;
  let warnedMcpAdapter = false;
  let warnedSkills = false;

  const notify = (
    ctx: ExtensionContext | undefined,
    message: string,
    level: "info" | "warning" | "error" = "info",
  ) => {
    if (ctx?.hasUI) ctx.ui.notify(message, level);
  };

  const ensureBridgeStarted = async (warn?: (message: string) => void): Promise<boolean> => {
    if (bridge.running) return true;
    bridgeStartPromise ??= bridge
      .start()
      .then(() => {
        bridge.refreshTools();
        return true;
      })
      .catch((error) => {
        const message = error instanceof Error ? error.message : String(error);
        warn?.(`antigravity: pi-tool bridge unavailable (${message}).`);
        return false;
      })
      .finally(() => {
        bridgeStartPromise = undefined;
      });
    return bridgeStartPromise;
  };

  const stopBridge = async () => {
    await bridgeStartPromise?.catch(() => {});
    await bridge.close();
  };

  const updateStatus = (ctx: ExtensionContext | undefined) => {
    if (!ctx?.hasUI) return;
    if (ctx.model?.provider !== PROVIDER_ID) {
      ctx.ui.setStatus(PROVIDER_ID, undefined);
      return;
    }
    const quotaSummary = usageReport
      ? formatAgyUsageShort(usageReport, ctx.model.id, new Date(), (level, text) =>
          ctx.ui.theme.fg(level, text),
        )
      : "";
    ctx.ui.setStatus(PROVIDER_ID, formatAgyFooterStatus(toolMode, executionMode, quotaSummary));
  };

  const refreshUsageSummary = async (ctx: ExtensionContext | undefined, force = false) => {
    if (!ctx?.hasUI || ctx.model?.provider !== PROVIDER_ID) return;
    if (usageFetchInFlight || (!force && Date.now() < nextUsageFetchAt)) return;
    usageFetchInFlight = true;
    try {
      usageReport = await fetchAgyUsage();
      nextUsageFetchAt = Date.now() + USAGE_SUMMARY_TTL_MS;
    } catch {
      nextUsageFetchAt = Date.now() + USAGE_ERROR_BACKOFF_MS;
    } finally {
      usageFetchInFlight = false;
    }
    updateStatus(activeCtx ?? ctx);
  };

  const syncWrapperTool = (ctx: ExtensionContext) => {
    const next = wrapperToolActiveAfterModelSwitch(
      pi.getActiveTools(),
      WRAPPER_TOOL_NAME,
      ctx.model?.provider,
      PROVIDER_ID,
    );
    if (next) {
      pi.setActiveTools([...next]);
    }
  };

  const applyPrivateSkills = (skills: readonly SkillLite[], cwd: string) => {
    privateSkills = piPrivateSkills([...skills], cwd);
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

  const seedSkillsFromCommands = (ctx: ExtensionContext) => {
    const skillCommands = pi
      .getCommands()
      .filter((cmd) => cmd.source === "skill" && Boolean(cmd.sourceInfo.path));
    applyPrivateSkills(
      skillCommands.map((cmd) => ({
        name: cmd.name.replace(/^skill:/, ""),
        description: cmd.description ?? "",
        filePath: cmd.sourceInfo.path,
        baseDir: path.dirname(cmd.sourceInfo.path),
      })),
      ctx.cwd,
    );
  };

  const createRuntimeForCwd = (cwd: string): AgyRuntime =>
    new AgyRuntime({
      cwd,
      bridge,
      bridgeToken,
      getToolMode: () => toolMode,
      getExecutionMode: () => executionMode,
      isHostHindsightActive: () => hasHostHindsightTools(pi.getAllTools()),
      sandbox,
      extraDirs,
      firstResponseTimeoutMs: savedSettings.firstResponseTimeoutMs,
      ensureBridge: () => ensureBridgeStarted((msg) => notify(activeCtx, msg, "warning")),
      onPersistedState: (state) => {
        pendingState = state;
      },
      onWarning: (msg) => notify(activeCtx, msg, "warning"),
      onWorkingStatus: (message) => {
        if (activeCtx?.hasUI) activeCtx.ui.setWorkingMessage(message);
      },
    });

  const getRuntime = (): AgyRuntime => {
    applyCliFlagsOnce();
    const cwd = activeCtx?.cwd ?? process.cwd();
    if (!runtime) {
      runtime = createRuntimeForCwd(cwd);
    }
    return runtime;
  };

  const cachedModels = savedSettings.cachedModels
    ? modelsFromCachedEntries(savedSettings.cachedModels)
    : [];
  let currentModels =
    cachedModels.length > 0
      ? mergeModelLists(cachedModels, getFallbackModels())
      : getFallbackModels();
  let modelsFetchedAt = savedSettings.modelsFetchedAt;

  pi.registerProvider(PROVIDER_ID, {
    name: "Antigravity (ACP)",
    api: "antigravity-acp",
    baseUrl: "acp://agy",
    apiKey: "antigravity-acp-local",
    models: toProviderModelConfigs(currentModels),
    refreshModels: async (context) => {
      if (context.signal.aborted || (!context.allowNetwork && !context.force)) {
        return toProviderModelConfigs(currentModels);
      }
      if (!context.force && modelCacheIsFresh({ fetchedAt: modelsFetchedAt, source: "live" })) {
        return toProviderModelConfigs(currentModels);
      }
      const probed = await probeAgyModels({ signal: context.signal, timeoutMs: 5_000 }).catch(
        () => [],
      );
      if (probed.length > 0) {
        currentModels = mergeModelLists(probed, getFallbackModels());
        modelsFetchedAt = Date.now();
        saveAgySettings({
          cachedModels: currentModels.map((m) => ({ id: m.id, name: m.name })),
          modelsFetchedAt,
        });
      }
      return toProviderModelConfigs(currentModels);
    },
    streamSimple: createAntigravityStreamSimple({
      getRuntime,
      getSessionId: () => activeCtx?.sessionManager.getSessionId() ?? "default",
      getBranchEntries: () => activeCtx?.sessionManager.getBranch() ?? [],
      getExecutionMode: () => executionMode,
      replayStore,
    }),
  });

  pi.registerTool({
    name: WRAPPER_TOOL_NAME,
    label: WRAPPER_TOOL_NAME,
    description: WRAPPER_TOOL_DESCRIPTION,
    parameters: Type.Object({
      tool: Type.String(),
      kind: Type.Optional(Type.String()),
      locations: Type.Optional(
        Type.Array(
          Type.Object({
            path: Type.String(),
            line: Type.Optional(Type.Number()),
          }),
        ),
      ),
      input: Type.Record(Type.String(), Type.Unknown()),
    }),
    async execute(toolCallId, params) {
      const recorded = replayStore.take(toolCallId);
      if (!recorded) {
        return {
          content: [{ type: "text", text: BRIDGE_PENDING_TOOL_MESSAGE }],
          details: {
            title: params.tool,
            ...(params.kind ? { kind: params.kind } : {}),
          },
        };
      }
      const text = recorded.error ?? recorded.output ?? "";
      return {
        content: [{ type: "text", text }],
        details: {
          title: recorded.title,
          ...(recorded.kind ? { kind: recorded.kind } : {}),
          ...(recorded.durationSeconds !== undefined
            ? { durationSeconds: recorded.durationSeconds }
            : {}),
        },
        ...(recorded.error ? { isError: true } : {}),
      };
    },
    renderCall(args, theme) {
      return new Text(
        formatAgyCall(
          {
            tool: args.tool,
            kind: args.kind,
            locations: args.locations,
            input: args.input,
          },
          theme,
        ),
        0,
        0,
      );
    },
    renderResult(result, options, theme) {
      const details = result.details as
        | { title?: string; kind?: string; durationSeconds?: number }
        | undefined;
      const rawText = result.content
        .filter((item): item is { type: "text"; text: string } => item.type === "text")
        .map((item) => item.text)
        .join("\n")
        .trim();

      if (result.isError) {
        const line = rawText.split("\n", 1)[0] ?? "tool error";
        return new Text(theme.fg("error", line), 0, 0);
      }

      const summary = summarizeAgyResult(details?.kind, rawText);
      const meta: string[] = [];
      if (summary.counts) meta.push(summary.counts);
      if (details?.durationSeconds !== undefined) {
        meta.push(`${details.durationSeconds.toFixed(1)}s`);
      }

      if (!options.expanded) {
        const preview = meta.length > 0 ? meta.join(" · ") : (rawText.split("\n", 1)[0] ?? "");
        return new Text(preview ? theme.fg("toolOutput", preview) : "", 0, 0);
      }
      const header = meta.length > 0 ? `${theme.fg("muted", meta.join(" · "))}\n` : "";
      return new Text(`${header}${theme.fg("toolOutput", rawText)}`, 0, 0);
    },
  });

  pi.registerEntryRenderer<AgyPersistedState>(AGY_STATE_ENTRY_TYPE, (entry, _options, theme) => {
    const parsed = parseAgyStateEntry(entry.data);
    if (parsed?.kind !== "reset") return undefined;
    return new Text(theme.fg("muted", "antigravity conversation reset"), 0, 0);
  });

  pi.registerCommand("agy-tools", {
    description: "Switch Antigravity tool routing mode (pi | agy)",
    getArgumentCompletions(prefix) {
      const query = prefix.trim().toLowerCase();
      const items = [
        { value: "pi", label: "pi", description: "Use Pi tools via MCP bridge (acp:pi)" },
        { value: "agy", label: "agy", description: "Use agy built-in tools (acp:agy)" },
      ].filter((item) => item.value.startsWith(query));
      return items.length > 0 ? items : null;
    },
    async handler(args, ctx) {
      applyCliFlagsOnce();
      const action = parseAgyToolModeArg(args);
      if (action.kind === "status") {
        notify(ctx, `Antigravity tool mode: ${formatAgyToolModeStatus(toolMode)}`);
        return;
      }
      if (action.kind === "invalid") {
        notify(ctx, `Invalid tool mode "${action.raw}". Use: pi | agy`, "warning");
        return;
      }
      toolMode = action.mode;
      saveAgySettings({ toolMode });
      bridge.refreshTools();
      if (toolMode !== "pi") {
        await stopBridge();
      }
      updateStatus(ctx);
      notify(
        ctx,
        `Antigravity tool mode set to ${formatAgyToolModeStatus(toolMode)} (saved to antigravity.json)`,
      );
    },
  });

  pi.registerCommand("agy-plan", {
    description: "Toggle Antigravity plan execution mode (on | off)",
    getArgumentCompletions(prefix) {
      const query = prefix.trim().toLowerCase();
      const items = [
        { value: "on", label: "on", description: "Enable ACP plan mode" },
        { value: "off", label: "off", description: "Use default ACP execution mode" },
      ].filter((item) => item.value.startsWith(query));
      return items.length > 0 ? items : null;
    },
    async handler(args, ctx) {
      applyCliFlagsOnce();
      const action = parseAgyPlanArg(args);
      if (action.kind === "status") {
        executionMode = executionMode === "plan" ? "default" : "plan";
      } else if (action.kind === "set") {
        executionMode = resolveAgyExecutionMode(action.enabled);
      } else {
        notify(ctx, `Invalid argument "${action.raw}". Use: on | off`, "warning");
        return;
      }
      updateStatus(ctx);
      notify(ctx, `Antigravity execution mode: ${executionMode}`);
    },
  });

  pi.registerCommand("agy-reset", {
    description: "Reset the active Antigravity ACP conversation",
    async handler(_args, ctx) {
      await runtime?.resetConversation();
      pendingState = undefined;
      pi.appendEntry(AGY_STATE_ENTRY_TYPE, {
        version: 1,
        kind: "reset",
        sessionId: ctx.sessionManager.getSessionId(),
        cwd: ctx.cwd,
      });
      updateStatus(ctx);
      notify(ctx, "Antigravity conversation reset; a new ACP session starts on the next prompt.");
    },
  });

  pi.registerCommand("agy-doctor", {
    description: "Verify agy CLI version, ACP transport, bridge state, and quota status",
    async handler(_args, ctx) {
      applyCliFlagsOnce();
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

      const bridgedCount = selectBridgedTools(
        pi.getAllTools(),
        pi.getActiveTools(),
        toolMode,
      ).length;
      const report = formatAgyDoctorReport({
        binary,
        version,
        binaryError,
        toolMode,
        executionMode,
        conversationId: runtime?.conversationId,
        bridgeStatus: bridge.running ? `running (${bridge.serverName})` : "idle",
        acpConnected: runtime?.isConnected ?? false,
        mcpServerCount: bridgedCount,
        quotaError: runtime?.lastQuotaError,
      });
      notify(ctx, report, binaryError ? "error" : "info");
    },
  });

  pi.registerCommand("agy-models", {
    description: "Refresh available Antigravity ACP models from agy CLI",
    async handler(_args, ctx) {
      const result = await ctx.modelRegistry.refresh({ providers: [PROVIDER_ID], force: true });
      const err = result.errors.get(PROVIDER_ID);
      if (err) {
        notify(ctx, `Failed to refresh Antigravity models: ${err.message}`, "error");
        return;
      }
      const ids = currentModels.map((m) => m.id).join(", ");
      notify(ctx, `Antigravity models (${currentModels.length}): ${ids}`);
    },
  });

  pi.registerCommand("agy-usage", {
    description: "Show Antigravity model quotas (5-hour and weekly limits)",
    async handler(args, ctx) {
      if (args.trim()) {
        notify(ctx, "Usage: /agy-usage", "warning");
        return;
      }
      await openAgyUsagePicker(ctx, async (signal) => {
        const report = await fetchAgyUsage({ signal });
        usageReport = report;
        nextUsageFetchAt = Date.now() + USAGE_SUMMARY_TTL_MS;
        updateStatus(ctx);
        return report;
      });
    },
  });

  const triggerBackgroundWarmup = (ctx: ExtensionContext) => {
    if (!ctx.hasUI || ctx.model?.provider !== PROVIDER_ID) return;
    void (async () => {
      // Yield a microtask so other extensions' synchronous session_start hooks
      // (e.g. Hindsight) finish registering tools before warmupClient reads them.
      await Promise.resolve();
      if (toolMode === "pi") {
        await ensureBridgeStarted();
      }
      await getRuntime().warmupClient();
    })().catch(() => {});
  };

  pi.on("session_start", async (_event, ctx) => {
    applyCliFlagsOnce();
    activeCtx = ctx;
    pendingState = undefined;
    void sweepDeadPiToolsAgents();
    if (runtime) {
      await runtime.dispose();
      runtime = undefined;
    }
    syncWrapperTool(ctx);
    seedSkillsFromCommands(ctx);
    bridge.refreshTools();
    updateStatus(ctx);
    if (ctx.model?.provider === PROVIDER_ID) {
      void refreshUsageSummary(ctx);
      triggerBackgroundWarmup(ctx);
    }
  });

  pi.on("model_select", (_event, ctx) => {
    activeCtx = ctx;
    syncWrapperTool(ctx);
    updateStatus(ctx);
    if (ctx.model?.provider === PROVIDER_ID) {
      void refreshUsageSummary(ctx);
      triggerBackgroundWarmup(ctx);
    } else if (runtime) {
      const prev = runtime;
      runtime = undefined;
      void prev.dispose();
    }
  });

  pi.on("session_tree", async (_event, ctx) => {
    activeCtx = ctx;
    pendingState = undefined;
    if (runtime) {
      await runtime.dispose();
      runtime = undefined;
    }
    updateStatus(ctx);
  });

  pi.on("before_agent_start", async (event, ctx) => {
    applyCliFlagsOnce();
    activeCtx = ctx;
    if (ctx.model?.provider !== PROVIDER_ID) {
      pendingState = undefined;
      if (runtime) {
        await runtime.dispose();
        runtime = undefined;
      }
      return;
    }
    const promptSkills = event.systemPromptOptions?.skills?.filter(
      (skill) => !skill.disableModelInvocation,
    );
    if (promptSkills) {
      applyPrivateSkills(promptSkills, ctx.cwd);
    } else {
      seedSkillsFromCommands(ctx);
    }
    bridge.refreshTools();
    if (toolMode === "pi") {
      if (!warnedMcpAdapter && hasUnsupportedMcpAdapter(pi.getAllTools())) {
        warnedMcpAdapter = true;
        notify(ctx, MCP_ADAPTER_UNSUPPORTED_WARNING, "warning");
      }
      const ready = await ensureBridgeStarted((msg) => notify(ctx, msg, "warning"));
      if (!ready && !warnedSkills && privateSkills.length > 0) {
        warnedSkills = true;
        notify(
          ctx,
          "antigravity: pi-tool bridge unavailable — pi-private skills are unavailable to agy this session.",
          "warning",
        );
      }
    }
    const effort = ctx.thinkingLevel === "off" ? undefined : ctx.thinkingLevel;
    await getRuntime()
      .warmupSession({
        modelId: ctx.model.id,
        effort,
        executionMode,
        sessionId: ctx.sessionManager.getSessionId(),
        branchEntries: ctx.sessionManager.getBranch(),
        signal: ctx.signal,
      })
      .catch(() => {
        // Defer binary/auth/config failures to streamSimple so Pi renders standard turn errors.
      });
  });

  pi.on("agent_end", async () => {
    if (activeCtx?.hasUI) {
      activeCtx.ui.setWorkingMessage(undefined);
    }
    await runtime?.settleAfterAgentEnd();
    if (pendingState) {
      pi.appendEntry(AGY_STATE_ENTRY_TYPE, pendingState);
      pendingState = undefined;
    }
    replayStore.clear();
    if (activeCtx?.model?.provider === PROVIDER_ID) {
      void refreshUsageSummary(activeCtx, Boolean(runtime?.lastQuotaError));
    }
  });

  pi.on("session_shutdown", async () => {
    await runtime?.dispose();
    runtime = undefined;
    await stopBridge();
  });
}
