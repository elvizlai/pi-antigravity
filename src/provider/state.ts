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

export type AgyBranchResolution =
  | { status: "active"; state: AgyConversationState }
  | { status: "reset" }
  | { status: "invalidated" }
  | { status: "none" };

/** Entries after a marker that do not alter what the agy conversation holds. */
const NON_CONTEXT_ENTRY_TYPES = new Set([
  "session",
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

function hasAdvancedConversationTurn(
  allEntries: readonly SessionEntryLike[],
  sessionId: string,
  conversationId: string,
  turns: number,
): boolean {
  for (const entry of allEntries) {
    if (entry.type !== "custom" || entry.customType !== AGY_STATE_ENTRY_TYPE) continue;
    const state = parseAgyStateEntry(entry.data);
    if (
      state?.kind === "conversation" &&
      state.sessionId === sessionId &&
      state.conversationId === conversationId &&
      state.turns > turns
    ) {
      return true;
    }
  }
  return false;
}

function evaluateStateMarker(
  state: AgyPersistedState,
  sessionId: string,
  cwd: string,
  allEntries?: readonly SessionEntryLike[],
): Exclude<AgyBranchResolution, { status: "none" }> {
  if (state.sessionId !== sessionId || state.cwd !== cwd) {
    return { status: "invalidated" };
  }
  if (state.kind === "reset") {
    return { status: "reset" };
  }
  if (
    allEntries &&
    hasAdvancedConversationTurn(allEntries, sessionId, state.conversationId, state.turns)
  ) {
    return { status: "invalidated" };
  }
  return { status: "active", state };
}

/**
 * Resolve the ACP conversation state for the current Pi session branch.
 *
 * Forked sessions get a new `sessionId`, tree rewinds/compactions add
 * context-mutating entries or rewind past an already-advanced ACP turn,
 * and `/agy-reset` appends an explicit `reset` marker.
 */
export function resolveAgyBranchState(
  sessionId: string,
  cwd: string,
  branchEntries: readonly SessionEntryLike[],
  allEntries?: readonly SessionEntryLike[],
): AgyBranchResolution {
  for (let i = branchEntries.length - 1; i >= 0; i -= 1) {
    const entry = branchEntries[i];
    if (entry.type === "custom" && entry.customType === AGY_STATE_ENTRY_TYPE) {
      const state = parseAgyStateEntry(entry.data);
      if (!state) continue;
      return evaluateStateMarker(state, sessionId, cwd, allEntries);
    }

    if (entry.type === "custom_message") continue;
    if (entry.type === "message") {
      const role = entry.message?.role;
      if (role === "custom" || role === "system" || role === "user" || role === "bashExecution") {
        continue;
      }
      return { status: "invalidated" };
    }
    if (!NON_CONTEXT_ENTRY_TYPES.has(entry.type)) {
      return { status: "invalidated" };
    }
  }
  return { status: "none" };
}

/**
 * Validate branch integrity for an active mid-turn continuation (bridged image
 * handoff or mid-turn steering). Because `antigravity-state` is only persisted at
 * `agent_end`, trailing `assistant` and `toolResult` messages from the live Pi
 * turn are expected and do not invalidate the in-flight ACP session.
 */
export function resolveInTurnBranchState(
  sessionId: string,
  cwd: string,
  branchEntries: readonly SessionEntryLike[],
  allEntries?: readonly SessionEntryLike[],
): "ok" | "reset" | "invalidated" {
  for (let i = branchEntries.length - 1; i >= 0; i -= 1) {
    const entry = branchEntries[i];
    if (entry.type === "custom" && entry.customType === AGY_STATE_ENTRY_TYPE) {
      const state = parseAgyStateEntry(entry.data);
      if (!state) continue;
      const resolution = evaluateStateMarker(state, sessionId, cwd, allEntries);
      return resolution.status === "active" ? "ok" : resolution.status;
    }

    if (entry.type === "custom_message" || entry.type === "message") continue;
    if (!NON_CONTEXT_ENTRY_TYPES.has(entry.type)) {
      return "invalidated";
    }
  }
  return "ok";
}

/**
 * Find the active agy conversation owned by the current Pi session branch.
 */
export function findActiveAgyConversation(
  sessionId: string,
  cwd: string,
  branchEntries: readonly SessionEntryLike[],
  allEntries?: readonly SessionEntryLike[],
): AgyConversationState | undefined {
  const resolved = resolveAgyBranchState(sessionId, cwd, branchEntries, allEntries);
  return resolved.status === "active" ? resolved.state : undefined;
}
