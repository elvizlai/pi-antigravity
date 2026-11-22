import assert from "node:assert/strict";
import test from "node:test";
import {
  type AgyToolView,
  acpUpdateToActivity,
  extractAcpToolOutput,
  isBridgedMcpTool,
  normalizeAcpPromptUsage,
} from "../src/acp/events.ts";
import {
  AgyQuotaTurnTracker,
  isAgyQuotaError,
  parseAgyErrorBanner,
  parseAgyQuotaNotice,
} from "../src/acp/quota.ts";

test("acpUpdateToActivity normalizes thought, text, context usage, locations, and content-first tool updates", () => {
  const tools = new Map<string, AgyToolView>();

  assert.deepEqual(
    acpUpdateToActivity(
      {
        sessionUpdate: "agent_thought_chunk",
        content: { type: "text", text: "pondering" },
      },
      tools,
    ),
    { type: "thinking_delta", text: "pondering" },
  );

  assert.deepEqual(
    acpUpdateToActivity(
      {
        sessionUpdate: "agent_message_chunk",
        content: { type: "text", text: "answer" },
      },
      tools,
    ),
    { type: "text_delta", text: "answer" },
  );

  assert.deepEqual(
    acpUpdateToActivity(
      {
        sessionUpdate: "usage_update",
        used: 1200,
        size: 10000,
      },
      tools,
    ),
    { type: "context_usage", usedTokens: 1200, sizeTokens: 10000 },
  );

  const started = acpUpdateToActivity(
    {
      sessionUpdate: "tool_call",
      toolCallId: "tc-1",
      title: "Edit a.ts",
      kind: "edit",
      locations: [{ path: "a.ts", line: 12 }],
      status: "in_progress",
      rawInput: { path: "a.ts" },
    },
    tools,
    1000,
  );
  assert.equal(started?.type, "tool_start");
  if (started?.type === "tool_start") {
    assert.equal(started.title, "Edit a.ts");
    assert.equal(started.kind, "edit");
    assert.deepEqual(started.locations, [{ path: "a.ts", line: 12 }]);
  }

  const done = acpUpdateToActivity(
    {
      sessionUpdate: "tool_call_update",
      toolCallId: "tc-1",
      status: "completed",
      content: [{ type: "diff", path: "a.ts", oldText: "old", newText: "new" }],
    },
    tools,
    2500,
  );
  assert.equal(done?.type, "tool_done");
  if (done?.type === "tool_done") {
    assert.equal(done.durationSeconds, 1.5);
    assert.equal(done.output, "diff: a.ts -1 lines +1 lines");
  }

  assert.equal(
    extractAcpToolOutput({ output: "fallback raw" }, [
      { type: "content", content: { type: "text", text: "standard content" } },
    ]),
    "standard content",
  );
  assert.equal(extractAcpToolOutput({ output: "fallback raw" }, undefined), "fallback raw");
});

test("normalizeAcpPromptUsage and isBridgedMcpTool work across standard ACP shapes", () => {
  assert.deepEqual(
    normalizeAcpPromptUsage({
      inputTokens: 100,
      outputTokens: 50,
      cachedReadTokens: 20,
      thoughtTokens: 12,
      totalTokens: 170,
    }),
    {
      input_tokens: 100,
      output_tokens: 50,
      cache_read_tokens: 20,
      cache_creation_tokens: undefined,
      reasoning_tokens: 12,
      thinking_tokens: 12,
      total_tokens: 170,
    },
  );
  assert.equal(normalizeAcpPromptUsage({}), undefined);

  assert.equal(isBridgedMcpTool({ name: "pi__read" }, "pi-1234"), true);
  assert.equal(isBridgedMcpTool({ title: "Run pi__edit" }, "pi-1234"), true);
  assert.equal(isBridgedMcpTool({ name: "mcp_pi_1234_bash" }, "pi-1234"), true);
  assert.equal(isBridgedMcpTool({ title: "pi-1234/pi__read" }, "pi-1234"), true);
  assert.equal(isBridgedMcpTool({ title: "read pi-1234.txt" }, "pi-1234"), false);
  assert.equal(isBridgedMcpTool({ title: "Run command" }, "pi-1234"), false);
});

test("quota detection catches fake end_turn quota banners while avoiding prose false positives", () => {
  const banner = parseAgyQuotaNotice(
    "⚠ Individual quota reached for gemini-pro. Resets in 2h 15m.",
  );
  assert.deepEqual(banner, {
    message: "Individual quota reached for gemini-pro. Resets in 2h 15m.",
    resetsIn: "2h 15m",
  });
  assert.equal(isAgyQuotaError("You have exhausted your capacity for this model"), true);

  const tracker = new AgyQuotaTurnTracker();
  tracker.noteText("⚠ Individual quota reached. Resets in 10m.");
  const quotaNotice = tracker.detectTurnErrorBanner({ totalTokens: 0, outputTokens: 0 });
  assert.equal(quotaNotice?.kind, "quota");
  assert.equal(quotaNotice?.kind === "quota" ? quotaNotice.resetsIn : undefined, "10m");

  const realTurn = new AgyQuotaTurnTracker();
  realTurn.noteThought();
  realTurn.noteText("⚠ Individual quota reached is what agy prints.");
  assert.equal(realTurn.detectTurnErrorBanner({ totalTokens: 0, outputTokens: 0 }), undefined);

  assert.equal(
    parseAgyErrorBanner("⚠ FAILED_PRECONDITION (code 400): User location is not supported.")?.kind,
    "api_error",
  );

  // Cumulative promptUsage from earlier tool waves must not suppress final-step error banners.
  const failedAfterTools = new AgyQuotaTurnTracker();
  failedAfterTools.noteToolActivity();
  failedAfterTools.noteText(
    "\n⚠ FAILED_PRECONDITION (code 400): User location is not supported for the API use.\n",
  );
  assert.deepEqual(
    failedAfterTools.detectTurnErrorBanner({ totalTokens: 1_400_423, outputTokens: 25_705 }),
    {
      kind: "api_error",
      message: "FAILED_PRECONDITION (code 400): User location is not supported for the API use.",
    },
  );

  const quotaAfterTools = new AgyQuotaTurnTracker();
  quotaAfterTools.noteToolActivity();
  quotaAfterTools.noteText("⚠ Individual quota reached. Resets in 2h.");
  assert.deepEqual(
    quotaAfterTools.detectTurnErrorBanner({ totalTokens: 50_000, outputTokens: 500 }),
    {
      kind: "quota",
      message: "Individual quota reached. Resets in 2h.",
      resetsIn: "2h",
    },
  );

  // Final step with its own thought/usage update is legitimate prose, not an error banner.
  const validWithQuote = new AgyQuotaTurnTracker();
  validWithQuote.noteToolActivity();
  validWithQuote.noteUsageUpdate();
  validWithQuote.noteText("⚠ FAILED_PRECONDITION (code 400): User location is not supported.");
  assert.equal(
    validWithQuote.detectTurnErrorBanner({ totalTokens: 50_000, outputTokens: 500 }),
    undefined,
  );
});
