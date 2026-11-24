import type { Model, ThinkingLevelMap } from "@earendil-works/pi-ai";
import { getAgyBinary } from "./binary.ts";
import { execAgyCommand } from "./process-tree.ts";

export type AgyExecutionMode = "default" | "plan";

export type AgyEffort = "low" | "medium" | "high" | "extra" | "xhigh" | "max";

const AGY_EFFORT_ORDER: readonly AgyEffort[] = ["low", "medium", "high", "extra", "xhigh", "max"];

export interface AgyAcpSelectOption {
  value: string;
  name: string;
  description?: string;
}

export interface AgyAcpConfigOption {
  id: string;
  category?: string;
  name?: string;
  type: string;
  currentValue?: string;
  options?: AgyAcpSelectOption[];
}

export interface AgyModelSpec {
  id: string;
  name: string;
  contextWindow: number;
  maxTokens: number;
  reasoning: boolean;
  defaultEffort?: AgyEffort;
  thinkingLevelMap?: ThinkingLevelMap;
}

/** Build a Pi `thinkingLevelMap` from a set of ACP `thought_level` / effort suffixes. */
export function buildThinkingLevelMap(
  efforts: Iterable<string> | undefined,
): ThinkingLevelMap | undefined {
  if (!efforts) return undefined;
  const set = new Set<string>();
  for (const raw of efforts) {
    const normalized = raw === "extra" ? "xhigh" : raw;
    if (
      normalized === "low" ||
      normalized === "medium" ||
      normalized === "high" ||
      normalized === "xhigh" ||
      normalized === "max"
    ) {
      set.add(normalized);
    }
  }
  if (set.size === 0) return undefined;
  const map: ThinkingLevelMap = {
    off: null,
    minimal: null,
    low: set.has("low") ? "low" : null,
    medium: set.has("medium") ? "medium" : null,
    high: set.has("high") ? "high" : null,
  };
  if (set.has("xhigh")) map.xhigh = "xhigh";
  if (set.has("max")) map.max = "max";
  return map;
}

const STANDARD_EFFORT_MAP = buildThinkingLevelMap(["low", "medium", "high"]);
const XHIGH_EFFORT_MAP = buildThinkingLevelMap(["low", "medium", "high", "xhigh"]);

const DEFAULT_FALLBACK_SPECS: readonly AgyModelSpec[] = [
  {
    id: "gemini-4-neon",
    name: "Gemini 4 Neon",
    contextWindow: 1_000_000,
    maxTokens: 65_536,
    reasoning: true,
    defaultEffort: "high",
    thinkingLevelMap: XHIGH_EFFORT_MAP,
  },
  {
    id: "gemini-4-carbon",
    name: "Gemini 4 Carbon",
    contextWindow: 1_000_000,
    maxTokens: 65_536,
    reasoning: true,
    defaultEffort: "high",
    thinkingLevelMap: STANDARD_EFFORT_MAP,
  },
  {
    id: "gemini-4-argon",
    name: "Gemini 4 Argon",
    contextWindow: 1_000_000,
    maxTokens: 65_536,
    reasoning: true,
    defaultEffort: "high",
    thinkingLevelMap: STANDARD_EFFORT_MAP,
  },
  {
    id: "gemini-4-hydrogen",
    name: "Gemini 4 Hydrogen",
    contextWindow: 1_000_000,
    maxTokens: 65_536,
    reasoning: true,
    defaultEffort: "high",
    thinkingLevelMap: XHIGH_EFFORT_MAP,
  },
  {
    id: "gemini-pro-orchestrator-neon",
    name: "Orchestrator (Neon)",
    contextWindow: 1_000_000,
    maxTokens: 65_536,
    reasoning: true,
  },
  {
    id: "gemini-pro-orchestrator",
    name: "Orchestrator (Carbon)",
    contextWindow: 1_000_000,
    maxTokens: 65_536,
    reasoning: true,
  },
  {
    id: "gemini-pro-orchestrator-argon",
    name: "Orchestrator (Argon)",
    contextWindow: 1_000_000,
    maxTokens: 65_536,
    reasoning: true,
  },
  {
    id: "gemini-3.8-flash",
    name: "Gemini 3.8 Flash",
    contextWindow: 1_000_000,
    maxTokens: 65_536,
    reasoning: true,
    defaultEffort: "medium",
    thinkingLevelMap: STANDARD_EFFORT_MAP,
  },
  {
    id: "claude-opus-5-5",
    name: "Claude Opus 5.5",
    contextWindow: 200_000,
    maxTokens: 64_000,
    reasoning: true,
    thinkingLevelMap: STANDARD_EFFORT_MAP,
  },
  {
    id: "claude-sonnet-5-5",
    name: "Claude Sonnet 5.5",
    contextWindow: 200_000,
    maxTokens: 64_000,
    reasoning: true,
    thinkingLevelMap: STANDARD_EFFORT_MAP,
  },
  {
    id: "gpt-oss-120b-medium",
    name: "GPT-OSS 120B (Medium)",
    contextWindow: 128_000,
    maxTokens: 32_768,
    reasoning: false,
  },
];

const ZERO_COST = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 };

function specToPiModel(spec: AgyModelSpec): Model<string> {
  return {
    id: spec.id,
    name: spec.name,
    api: "antigravity-acp",
    provider: "antigravity",
    baseUrl: "acp://agy",
    reasoning: spec.reasoning,
    ...(spec.thinkingLevelMap ? { thinkingLevelMap: { ...spec.thinkingLevelMap } } : {}),
    input: ["text", "image"],
    cost: ZERO_COST,
    contextWindow: spec.contextWindow,
    maxTokens: spec.maxTokens,
  };
}

/** Return built-in fallback models derived from live `agy 1.5.13` ACP capabilities. */
export function getFallbackModels(): Model<string>[] {
  return DEFAULT_FALLBACK_SPECS.map(specToPiModel);
}

/** Hydrate lightweight `{ id, name, thinkingLevelMap? }` cached model entries into Pi `Model` records synchronously. */
export function modelsFromCachedEntries(
  entries: ReadonlyArray<{ id: string; name?: string; thinkingLevelMap?: ThinkingLevelMap }>,
): Model<string>[] {
  if (entries.length === 0) return [];
  const byId = new Map<string, Model<string>>();
  for (const entry of entries) {
    const id = entry.id.trim();
    if (!id || byId.has(id)) continue;
    const name = entry.name?.trim() || id;
    const spec = inferModelSpec(id, name);
    byId.set(
      id,
      specToPiModel(
        entry.thinkingLevelMap
          ? { ...spec, thinkingLevelMap: { ...entry.thinkingLevelMap } }
          : spec,
      ),
    );
  }
  return [...byId.values()];
}

const KNOWN_SPECS = new Map(DEFAULT_FALLBACK_SPECS.map((spec) => [spec.id, spec]));
const SUFFIX_RE = /-(low|medium|high|extra|xhigh|max)$/;

function stripEffortSuffix(rawId: string): { base: string; suffix?: AgyEffort } {
  const match = SUFFIX_RE.exec(rawId);
  if (!match) return { base: rawId };
  return {
    base: rawId.slice(0, match.index),
    suffix: match[1] as AgyEffort,
  };
}

function inferModelSpec(id: string, displayName?: string): AgyModelSpec {
  const known = KNOWN_SPECS.get(id);
  if (known) {
    return displayName ? { ...known, name: displayName } : known;
  }
  const lower = id.toLowerCase();
  if (lower.startsWith("claude-")) {
    return {
      id,
      name: displayName || id,
      contextWindow: 200_000,
      maxTokens: 64_000,
      reasoning: true,
    };
  }
  if (lower.startsWith("gpt-oss")) {
    return {
      id,
      name: displayName || id,
      contextWindow: 128_000,
      maxTokens: 32_768,
      reasoning: false,
    };
  }
  return {
    id,
    name: displayName || id,
    contextWindow: 1_000_000,
    maxTokens: 65_536,
    reasoning: true,
  };
}

/**
 * Parse plain-text output from `agy models` into deduplicated base ACP models.
 * Handles spinner escape prefixes (e.g. `⠋ Fetching available models...\r`).
 */
export function parseAgyModelsOutput(stdout: string): Model<string>[] {
  const cleaned = stdout.replace(/\r/g, "\n").replace(/\x1b\[[0-9;]*[A-Za-z]/g, "");

  const rawEntries: Array<{ id: string; label: string }> = [];
  for (const rawLine of cleaned.split("\n")) {
    const line = rawLine.trim();
    if (!line || line.includes("Fetching available models")) continue;
    const match = /^([a-zA-Z0-9._-]+)\s+(.+)$/.exec(line);
    if (!match) continue;
    rawEntries.push({ id: match[1], label: match[2].trim() });
  }

  if (rawEntries.length === 0) return [];

  // Count how many distinct effort suffixes share a base id so solitary suffixed
  // names like `gpt-oss-120b-medium` keep their full id when no siblings exist.
  const siblingsByBase = new Map<string, Set<string>>();
  for (const entry of rawEntries) {
    const { base, suffix } = stripEffortSuffix(entry.id);
    const set = siblingsByBase.get(base) ?? new Set<string>();
    if (suffix) set.add(suffix);
    siblingsByBase.set(base, set);
  }

  const byId = new Map<string, AgyModelSpec>();
  for (const entry of rawEntries) {
    const { base, suffix } = stripEffortSuffix(entry.id);
    const siblings = siblingsByBase.get(base);
    const keepFullId =
      KNOWN_SPECS.has(entry.id) ||
      (!KNOWN_SPECS.has(base) && (!suffix || (siblings?.size ?? 0) <= 1));
    const canonicalId = keepFullId ? entry.id : base;
    if (byId.has(canonicalId)) continue;
    const cleanLabel = keepFullId
      ? entry.label
      : entry.label.replace(/\s*\((Low|Medium|High|Extra|XHigh|Max)\)\s*$/i, "").trim();
    const spec = inferModelSpec(canonicalId, cleanLabel);
    const thinkingLevelMap =
      !keepFullId && siblings && siblings.size > 0 ? buildThinkingLevelMap(siblings) : undefined;
    byId.set(canonicalId, thinkingLevelMap ? { ...spec, thinkingLevelMap } : spec);
  }

  return [...byId.values()].map(specToPiModel);
}

/** Merge freshly discovered models over fallback specs without losing order or defaults. */
export function mergeModelLists(
  discovered: readonly Model<string>[],
  fallback: readonly Model<string>[] = getFallbackModels(),
): Model<string>[] {
  if (discovered.length === 0) return [...fallback];
  const fallbackById = new Map(fallback.map((m) => [m.id, m]));
  const seen = new Set<string>();
  const out: Model<string>[] = [];

  for (const model of discovered) {
    if (seen.has(model.id)) continue;
    seen.add(model.id);
    const base = fallbackById.get(model.id);
    if (base) {
      const thinkingLevelMap = model.thinkingLevelMap ?? base.thinkingLevelMap;
      out.push({
        ...base,
        name: model.name || base.name,
        ...(thinkingLevelMap ? { thinkingLevelMap: { ...thinkingLevelMap } } : {}),
      });
    } else {
      out.push(model);
    }
  }
  return out;
}

export async function probeAgyModels(options?: {
  binaryPath?: string;
  timeoutMs?: number;
  signal?: AbortSignal;
}): Promise<Model<string>[]> {
  const binary = getAgyBinary(options?.binaryPath);
  const timeoutMs = options?.timeoutMs ?? 15_000;
  const res = await execAgyCommand(binary, ["models"], {
    signal: options?.signal,
    timeoutMs,
    abortMessage: "Model probe aborted.",
    timeoutMessage: `Timed out waiting for \`agy models\` after ${timeoutMs}ms.`,
  });
  if (res.code !== 0) {
    throw new Error(res.stderr.trim() || `\`agy models\` exited ${res.code}.`);
  }
  return parseAgyModelsOutput(res.stdout);
}

export function extractAcpConfigOption(
  configOptions: readonly AgyAcpConfigOption[] | undefined,
  idOrCategory: string,
): AgyAcpConfigOption | undefined {
  if (!configOptions) return undefined;
  return (
    configOptions.find((opt) => opt.id === idOrCategory) ??
    configOptions.find((opt) => opt.category === idOrCategory)
  );
}

/**
 * Resolve a requested model id (base id or legacy `-<effort>` id) against live
 * ACP `model` config options.
 */
export function resolveAcpModelCandidate(
  requestedModelId: string,
  modelOption: AgyAcpConfigOption | undefined,
): { modelId: string; impliedEffort?: AgyEffort } {
  const optionValues = new Set((modelOption?.options ?? []).map((o) => o.value));
  if (optionValues.size === 0 || optionValues.has(requestedModelId)) {
    return { modelId: requestedModelId };
  }
  const { base, suffix } = stripEffortSuffix(requestedModelId);
  if (suffix && optionValues.has(base)) {
    return { modelId: base, impliedEffort: suffix };
  }
  return { modelId: requestedModelId };
}

/**
 * Resolve Pi `reasoning` / effort into the closest supported ACP `thought_level`.
 * Returns `undefined` when `thought_level` is not offered for the active model.
 */
export function resolveAcpThoughtLevel(
  requested: string | undefined,
  thoughtOption: AgyAcpConfigOption | undefined,
): AgyEffort | undefined {
  if (!thoughtOption?.options || thoughtOption.options.length === 0) return undefined;
  const supported = new Set(thoughtOption.options.map((o) => o.value));
  if (!requested) return undefined;
  if (supported.has(requested)) return requested as AgyEffort;

  const mapped: AgyEffort =
    requested === "minimal"
      ? "low"
      : requested === "max"
        ? "xhigh"
        : requested === "low" ||
            requested === "medium" ||
            requested === "high" ||
            requested === "extra" ||
            requested === "xhigh"
          ? requested
          : "medium";

  if (supported.has(mapped)) return mapped;

  const targetIdx = AGY_EFFORT_ORDER.indexOf(mapped);
  for (let i = targetIdx - 1; i >= 0; i--) {
    if (supported.has(AGY_EFFORT_ORDER[i])) return AGY_EFFORT_ORDER[i];
  }
  for (let i = targetIdx + 1; i < AGY_EFFORT_ORDER.length; i++) {
    if (supported.has(AGY_EFFORT_ORDER[i])) return AGY_EFFORT_ORDER[i];
  }
  return undefined;
}

/** Resolve `default` / `plan` execution mode against advertised ACP `mode` options. */
export function resolveAcpExecutionMode(
  desired: AgyExecutionMode,
  modeOption: AgyAcpConfigOption | undefined,
): string | undefined {
  if (!modeOption) return undefined;
  const supported = new Set((modeOption.options ?? []).map((o) => o.value));
  if (supported.size === 0 || supported.has(desired)) return desired;
  if (desired === "default" && supported.has("agent")) return "agent";
  return undefined;
}

const LIVE_CACHE_TTL_MS = 60 * 60 * 1000;
const FALLBACK_CACHE_TTL_MS = 5 * 60 * 1000;

function modelCacheTtlMs(source: "live" | "fallback" | undefined): number {
  return source === "live" ? LIVE_CACHE_TTL_MS : FALLBACK_CACHE_TTL_MS;
}

export function modelCacheIsFresh(
  cache: { fetchedAt?: number; source?: "live" | "fallback" },
  now = Date.now(),
): boolean {
  return Boolean(cache.fetchedAt && now - cache.fetchedAt < modelCacheTtlMs(cache.source));
}
