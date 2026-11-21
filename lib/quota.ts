/**
 * Personal quota-exhaustion detection for `agy acp`.
 *
 * OAR finding (verified on agy 1.5.12/1.5.13): when personal quota is exhausted,
 * `agy acp` sends a single `agent_message_chunk` containing the warning banner
 * and resolves `session/prompt` with `stopReason: "end_turn"` instead of a
 * JSON-RPC error, with no final-step `agent_thought_chunk`, no trailing
 * `usage_update`, and no `usage` tokens reported.
 */

export interface AgyQuotaNotice {
  message: string;
  resetsIn?: string;
}

const QUOTA_PREFIX = "⚠";
const QUOTA_CORE_REGEX =
  /(?:Individual quota reached|quota(?:\s+limit)?\s+reached|You have exhausted your capacity)/i;
const RESETS_IN_REGEX = /Resets in\s+([^\n.]+)/i;

/** Detect a standalone agy quota-exhaustion notice block. */
export function parseAgyQuotaNotice(text: string): AgyQuotaNotice | undefined {
  const trimmed = text.trim();
  if (!trimmed) return undefined;
  // Require the banner line itself rather than ordinary prose discussing quota.
  const firstLine = trimmed.split("\n", 1)[0]?.trim() ?? "";
  const looksLikeBanner =
    firstLine.startsWith(QUOTA_PREFIX) ||
    /^(?:Error:\s*)?Individual quota reached\b/i.test(firstLine);
  if (!looksLikeBanner || !QUOTA_CORE_REGEX.test(trimmed)) return undefined;
  if (trimmed.length > 400) return undefined;

  const resetsMatch = RESETS_IN_REGEX.exec(trimmed);
  return {
    message: trimmed.replace(/^⚠\s*/, ""),
    resetsIn: resetsMatch?.[1]?.trim(),
  };
}

/** Match RPC/process error messages caused by quota/capacity exhaustion. */
export function isAgyQuotaError(message: string): boolean {
  return QUOTA_CORE_REGEX.test(message) || /\bquota\s+exhausted\b/i.test(message);
}

/**
 * Per-turn signal tracker that guards against false positives if the model
 * quotes a short quota notice alongside real reasoning, usage, or context fills.
 */
export class AgyQuotaTurnTracker {
  #sawThoughtAfterLastTool = false;
  #sawUsageUpdateAfterLastTool = false;
  #accumulatedText = "";

  noteText(text: string): void {
    this.#accumulatedText += text;
  }

  noteThought(): void {
    this.#sawThoughtAfterLastTool = true;
  }

  noteUsageUpdate(): void {
    this.#sawUsageUpdateAfterLastTool = true;
  }

  noteToolActivity(): void {
    // Only signals in the final assistant step before end_turn count.
    this.#sawThoughtAfterLastTool = false;
    this.#sawUsageUpdateAfterLastTool = false;
    this.#accumulatedText = "";
  }

  /**
   * Returns the detected quota notice when `session/prompt` ends cleanly but
   * exhibits agy's fake-`end_turn` quota-exhaustion signature.
   */
  detectTurnQuotaExhaustion(promptUsage?: {
    totalTokens?: number;
    outputTokens?: number;
  }): AgyQuotaNotice | undefined {
    const notice = parseAgyQuotaNotice(this.#accumulatedText);
    if (!notice) return undefined;
    if (this.#sawThoughtAfterLastTool || this.#sawUsageUpdateAfterLastTool) return undefined;
    const totalTokens = promptUsage?.totalTokens ?? 0;
    const outputTokens = promptUsage?.outputTokens ?? 0;
    if (totalTokens > 0 || outputTokens > 0) return undefined;
    return notice;
  }
}
