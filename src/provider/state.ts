import { asRecord } from "../acp/shared.ts";

export const AGY_STATE_ENTRY_TYPE = "antigravity-state";

export interface AgyConversationState {
  version: 1;
  kind: "conversation";
  sessionId: string;
  conversationId: string;
  cwd: string;
  turns: number;
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

/** Narrow unknown custom entry data to a valid state record, ignoring foreign or corrupt shapes. */
export function parseAgyStateEntry(data: unknown): AgyPersistedState | undefined {
  const record = asRecord(data);
  if (!record) return undefined;
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
  if (!isBoundedString(record.conversationId, 256)) return undefined;
  if (!Number.isSafeInteger(record.turns) || (record.turns as number) < 0) return undefined;

  return {
    version: 1,
    kind: "conversation",
    sessionId: record.sessionId,
    conversationId: record.conversationId,
    cwd: record.cwd,
    turns: record.turns as number,
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
