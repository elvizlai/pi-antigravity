import type { AgyUsage } from "../acp/events.ts";
import type { AgyToolMode } from "../bridge/server.ts";

export const AGY_STATE_ENTRY_TYPE = "antigravity-state";

export interface AgyConversationState {
  version: 1;
  kind: "conversation";
  sessionId: string;
  conversationId: string;
  cwd: string;
  modelId: string;
  toolMode?: AgyToolMode;
  bridgeServer?: string;
  turns: number;
  usage: AgyUsage;
  contextTokens?: number;
}

export interface AgyResetState {
  version: 1;
  kind: "reset";
  sessionId: string;
  cwd: string;
}

export type AgyPersistedState = AgyConversationState | AgyResetState;

export interface SessionEntryLike {
  type: string;
  customType?: string;
  data?: unknown;
  message?: { role?: string };
}

/** Entries after a marker that do not alter what the agy conversation holds. */
const NON_CONTEXT_ENTRY_TYPES = new Set([
  "custom",
  "label",
  "session_info",
  "thinking_level_change",
  "model_change",
]);

function isBoundedString(value: unknown, maxLen: number): value is string {
  return typeof value === "string" && value.length > 0 && value.length <= maxLen;
}

function isNonNegativeNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value) && value >= 0;
}

function parseUsage(raw: unknown): AgyUsage | undefined {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) return undefined;
  const obj = raw as Record<string, unknown>;
  const keys: (keyof AgyUsage)[] = [
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_creation_tokens",
    "reasoning_tokens",
    "thinking_tokens",
    "total_tokens",
  ];
  const parsed: AgyUsage = {};
  for (const key of keys) {
    const val = obj[key];
    if (val === undefined) continue;
    if (!isNonNegativeNumber(val)) return undefined;
    parsed[key] = val;
  }
  return parsed;
}

/** Narrow unknown custom entry data to a valid state record, ignoring foreign or corrupt shapes. */
export function parseAgyStateEntry(data: unknown): AgyPersistedState | undefined {
  if (!data || typeof data !== "object" || Array.isArray(data)) return undefined;
  const record = data as Record<string, unknown>;
  if (record.version !== 1) return undefined;
  if (!isBoundedString(record.sessionId, 256) || !isBoundedString(record.cwd, 4096)) {
    return undefined;
  }

  if (record.kind === "reset") {
    return {
      version: 1,
      kind: "reset",
      sessionId: record.sessionId,
      cwd: record.cwd,
    };
  }
  if (record.kind !== "conversation") return undefined;
  if (!isBoundedString(record.conversationId, 256) || !isBoundedString(record.modelId, 256)) {
    return undefined;
  }
  if (record.toolMode !== undefined && record.toolMode !== "native" && record.toolMode !== "pi") {
    return undefined;
  }
  if (!Number.isSafeInteger(record.turns) || (record.turns as number) < 0) return undefined;
  if (record.bridgeServer !== undefined && !isBoundedString(record.bridgeServer, 64)) {
    return undefined;
  }
  const usage = parseUsage(record.usage);
  if (!usage) return undefined;
  if (record.contextTokens !== undefined && !isNonNegativeNumber(record.contextTokens)) {
    return undefined;
  }

  return {
    version: 1,
    kind: "conversation",
    sessionId: record.sessionId,
    conversationId: record.conversationId,
    cwd: record.cwd,
    modelId: record.modelId,
    ...(record.toolMode ? { toolMode: record.toolMode } : {}),
    ...(record.bridgeServer ? { bridgeServer: record.bridgeServer } : {}),
    turns: record.turns as number,
    usage,
    ...(record.contextTokens !== undefined ? { contextTokens: record.contextTokens } : {}),
  };
}

/**
 * Find the active agy conversation owned by the current Pi session branch.
 *
 * Forked sessions get a new `sessionId`, and tree rewinds/compactions add
 * context-mutating entries after our state marker; either invalidates resume
 * using pure session branch state (no internal agy DB inspection).
 */
export function findActiveAgyConversation(
  sessionId: string,
  cwd: string,
  branchEntries: readonly SessionEntryLike[],
): AgyConversationState | undefined {
  for (let i = branchEntries.length - 1; i >= 0; i -= 1) {
    const entry = branchEntries[i];
    if (entry.type === "custom" && entry.customType === AGY_STATE_ENTRY_TYPE) {
      const state = parseAgyStateEntry(entry.data);
      if (!state) continue;
      if (state.sessionId !== sessionId || state.cwd !== cwd) return undefined;
      return state.kind === "conversation" ? state : undefined;
    }

    if (entry.type === "message") {
      const role = entry.message?.role;
      if (role === "custom" || role === "system" || role === "user") continue;
      return undefined;
    }
    if (!NON_CONTEXT_ENTRY_TYPES.has(entry.type)) {
      return undefined;
    }
  }
  return undefined;
}
