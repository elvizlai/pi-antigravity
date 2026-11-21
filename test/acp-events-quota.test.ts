import assert from "node:assert/strict";
import test from "node:test";
import {
  type AgyActivity,
  type AgyToolView,
  acpUpdateToActivity,
  agyToolStepKey,
  isBridgedMcpTool,
  isDaemonToolStep,
  normalizeAcpPromptUsage,
  trackActiveToolStep,
} from "../src/acp/events.ts";
import { AgyQuotaTurnTracker, isAgyQuotaError, parseAgyQuotaNotice } from "../src/acp/quota.ts";

test("acpUpdateToActivity normalizes thought, text, context usage, and diff tool updates", () => {
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
      title: "edit_file",
      status: "in_progress",
      rawInput: { path: "a.ts" },
    },
    tools,
    1000,
  );
  assert.equal(started?.type, "tool_start");

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
});

test("normalizeAcpPromptUsage and isBridgedMcpTool work across camelCase and snake_case", () => {
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
  assert.equal(isBridgedMcpTool({ name: "mcp_pi_1234_bash" }, "pi-1234"), true);
  assert.equal(
    isBridgedMcpTool({ name: "call_mcp_tool", args: { ServerName: "pi-1234" } }, "pi-1234"),
    true,
  );
  assert.equal(
    isBridgedMcpTool({ name: "call_mcp_tool", args: { ServerName: "other" } }, "pi-1234"),
    false,
  );
  assert.equal(isBridgedMcpTool({ name: "run_command" }, "pi-1234"), false);
});

test("tool step tracking identifies IsDaemon run_command steps", () => {
  const active = new Set<string>();
  const daemons = new Set<string>();

  const daemonStart: AgyActivity = {
    type: "tool_start",
    callId: "c1",
    name: "run_command",
    args: { CommandLine: "npm run dev", IsDaemon: true },
  };
  assert.equal(isDaemonToolStep(daemonStart), true);
  assert.equal(agyToolStepKey(daemonStart), "call:c1");

  trackActiveToolStep(active, daemonStart, daemons);
  assert.equal(active.has("call:c1"), true);
  assert.equal(daemons.has("call:c1"), true);

  trackActiveToolStep(
    active,
    { type: "tool_done", callId: "c1", name: "run_command", args: {} },
    daemons,
  );
  assert.equal(active.size, 0);
  assert.equal(daemons.size, 0);
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
  assert.equal(
    tracker.detectTurnQuotaExhaustion({ totalTokens: 0, outputTokens: 0 })?.resetsIn,
    "10m",
  );

  const realTurn = new AgyQuotaTurnTracker();
  realTurn.noteThought();
  realTurn.noteText("⚠ Individual quota reached is what agy prints.");
  assert.equal(realTurn.detectTurnQuotaExhaustion({ totalTokens: 0, outputTokens: 0 }), undefined);
});
