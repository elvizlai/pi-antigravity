/**
 * Upstream banner error & quota-exhaustion detection for `agy acp`.
 *
 * When personal quota is exhausted or an upstream API/RPC call fails cleanly
 * after prior steps, `agy acp` emits a standalone `⚠ ...` `agent_message_chunk`
 * and resolves `session/prompt` with `stopReason: "end_turn"` instead of a
 * JSON-RPC error, with no final-step `agent_thought_chunk` and no trailing
 * `usage_update`.
 */

export interface AgyQuotaNotice {
  message: string;
  resetsIn?: string;
}

export type AgyTurnBanner =
  | { kind: "quota"; message: string; resetsIn?: string }
  | { kind: "api_error"; message: string };

const QUOTA_PREFIX = "⚠";
const QUOTA_CORE_REGEX =
  /(?:Individual quota reached|quota(?:\s+limit)?\s+reached|You have exhausted your capacity)/i;
const RESETS_IN_REGEX = /Resets in\s+([^\n.]+)/i;
const API_ERROR_BANNER_REGEX =
  /^(?:[A-Z][A-Z0-9_]{2,}(?:\s*\(code\s*\d+\))?:\s+\S+|(?:API|RPC|Authentication|Network|Internal)\s+error\b.*|Error(?:\s*\(code\s*\d+\))?:\s+\S+)/;

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

/** Detect either a quota banner or a standalone upstream `⚠ <ERROR>: ...` banner. */
export function parseAgyErrorBanner(text: string): AgyTurnBanner | undefined {
  const quota = parseAgyQuotaNotice(text);
  if (quota) {
    return { kind: "quota", message: quota.message, resetsIn: quota.resetsIn };
  }

  const trimmed = text.trim();
  if (!trimmed.startsWith(QUOTA_PREFIX) || trimmed.length > 600) return undefined;
  const nonEmptyLines = trimmed.split("\n").filter((line) => line.trim().length > 0);
  if (nonEmptyLines.length > 3) return undefined;

  const body = trimmed.replace(/^⚠\s*/, "");
  if (!API_ERROR_BANNER_REGEX.test(body)) return undefined;
  return { kind: "api_error", message: body };
}

/** Match RPC/process error messages caused by quota/capacity exhaustion. */
export function isAgyQuotaError(message: string): boolean {
  return QUOTA_CORE_REGEX.test(message) || /\bquota\s+exhausted\b/i.test(message);
}

/**
 * Per-turn signal tracker that guards against false positives if the model
 * quotes a short warning banner alongside real reasoning or usage updates.
 */
export class AgyQuotaTurnTracker {
  #sawThoughtAfterLastTool = false;
  #sawUsageUpdateAfterLastTool = false;
  #toolStepCount = 0;
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
    // Only signals in the final step before end_turn determine whether it failed.
    this.#toolStepCount += 1;
    this.#sawThoughtAfterLastTool = false;
    this.#sawUsageUpdateAfterLastTool = false;
    this.#accumulatedText = "";
  }

  /**
   * Returns a quota or API error banner when the final step after any tool calls
   * ends with `⚠ ...` and emitted no reasoning or usage update of its own.
   */
  detectTurnErrorBanner(promptUsage?: {
    totalTokens?: number;
    outputTokens?: number;
  }): AgyTurnBanner | undefined {
    const banner = parseAgyErrorBanner(this.#accumulatedText);
    if (!banner) return undefined;
    if (this.#sawThoughtAfterLastTool || this.#sawUsageUpdateAfterLastTool) return undefined;
    // Cumulative `promptUsage` can be non-zero from prior tool steps in the same turn.
    if (
      this.#toolStepCount === 0 &&
      ((promptUsage?.totalTokens ?? 0) > 0 || (promptUsage?.outputTokens ?? 0) > 0)
    ) {
      return undefined;
    }
    return banner;
  }
}
