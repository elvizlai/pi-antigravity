import assert from "node:assert/strict";
import { test } from "node:test";
import {
  type AgyToolView,
  acpUpdateToActivity,
  extractAcpToolOutput,
  extractLatestTranscriptRetryError,
  isBridgedMcpTool,
  normalizeAcpPromptUsage,
} from "../lib/acp-updates.ts";

test("acpUpdateToActivity converts message chunks, thoughts, and usage_update", () => {
  const tools = new Map<string, AgyToolView>();

  assert.deepEqual(
    acpUpdateToActivity(
      {
        sessionUpdate: "agent_thought_chunk",
        content: { type: "text", text: "planning" },
      },
      tools,
    ),
    { type: "thinking_delta", text: "planning" },
  );

  assert.deepEqual(
    acpUpdateToActivity(
      {
        sessionUpdate: "agent_message_chunk",
        content: { type: "text", text: "hello" },
      },
      tools,
    ),
    { type: "text_delta", text: "hello" },
  );

  assert.deepEqual(
    acpUpdateToActivity(
      {
        sessionUpdate: "usage_update",
        used: 12_000,
        size: 1_000_000,
      },
      tools,
    ),
    { type: "context_usage", usedTokens: 12_000, sizeTokens: 1_000_000 },
  );
});

test("acpUpdateToActivity merges incremental tool_call lifecycle updates", () => {
  const tools = new Map<string, AgyToolView>();

  const started = acpUpdateToActivity(
    {
      sessionUpdate: "tool_call",
      toolCallId: "tc-1",
      name: "view_file",
      kind: "read",
      status: "pending",
      locations: [{ path: "/src/app.ts", line: 1 }],
    },
    tools,
    1_000,
  );
  assert.equal(started?.type, "tool_start");
  if (started?.type === "tool_start") {
    assert.equal(started.name, "view_file");
    assert.deepEqual(started.args, { path: "/src/app.ts" });
  }

  const updated = acpUpdateToActivity(
    {
      sessionUpdate: "tool_call_update",
      toolCallId: "tc-1",
      status: "in_progress",
      rawInput: { AbsolutePath: "/src/app.ts", StartLine: 10 },
    },
    tools,
    1_200,
  );
  assert.equal(updated?.type, "tool_update");

  const done = acpUpdateToActivity(
    {
      sessionUpdate: "tool_call_update",
      toolCallId: "tc-1",
      status: "completed",
      rawOutput: { stdout: "line 10\n" },
    },
    tools,
    2_500,
  );
  assert.equal(done?.type, "tool_done");
  if (done?.type === "tool_done") {
    assert.equal(done.name, "view_file");
    assert.equal(done.output, "line 10");
    assert.equal(done.durationSeconds, 1.5);
    assert.equal(done.args.StartLine, 10);
  }

  // Duplicate completed update is ignored idempotently.
  const duplicate = acpUpdateToActivity(
    {
      sessionUpdate: "tool_call_update",
      toolCallId: "tc-1",
      status: "completed",
    },
    tools,
    3_000,
  );
  assert.equal(duplicate, undefined);
});

test("extractAcpToolOutput handles diff blocks and stderr", () => {
  assert.equal(extractAcpToolOutput({ stdout: "ok", stderr: "warn" }, undefined), "ok\nwarn");
  assert.equal(
    extractAcpToolOutput(undefined, [
      { type: "diff", path: "a.ts", oldText: "1\n2", newText: "1\n2\n3" },
    ]),
    "diff: a.ts -2 lines +3 lines",
  );
});

test("normalizeAcpPromptUsage maps ACP token fields into AgyUsage", () => {
  assert.deepEqual(
    normalizeAcpPromptUsage({
      inputTokens: 100,
      outputTokens: 50,
      cachedReadTokens: 800,
      cachedWriteTokens: 20,
      thoughtTokens: 30,
      totalTokens: 1_000,
    }),
    {
      input_tokens: 100,
      output_tokens: 50,
      cache_read_tokens: 800,
      cache_creation_tokens: 20,
      reasoning_tokens: 30,
      thinking_tokens: 30,
      total_tokens: 1_000,
    },
  );
  assert.equal(normalizeAcpPromptUsage({}), undefined);
});

test("isBridgedMcpTool isolates Pi MCP bridge tools from external user MCP servers", () => {
  assert.equal(isBridgedMcpTool({ name: "pi__read" }, "pi-1234"), true);
  assert.equal(isBridgedMcpTool({ name: "mcp_pi_1234_bash" }, "pi-1234"), true);
  assert.equal(
    isBridgedMcpTool(
      {
        name: "call_mcp_tool",
        args: { ServerName: "pi-1234", ToolName: "pi__bash" },
      },
      "pi-1234",
    ),
    true,
  );
  assert.equal(
    isBridgedMcpTool(
      {
        name: "call_mcp_tool",
        args: { ServerName: "github", ToolName: "get_issue" },
      },
      "pi-1234",
    ),
    false,
  );
  assert.equal(isBridgedMcpTool({ name: "search_web" }, "pi-1234"), false);
});

test("extractLatestTranscriptRetryError surfaces trailing 429 retries and clears after model responds", () => {
  const retryingTail = [
    '...truncated leading partial line"}',
    JSON.stringify({
      step_index: 1,
      source: "SYSTEM",
      type: "ERROR_MESSAGE",
      status: "DONE",
      error: "API error (attempt 1): RESOURCE_EXHAUSTED (code 429): Resource has been exhausted.",
    }),
    JSON.stringify({
      step_index: 2,
      source: "SYSTEM",
      type: "ERROR_MESSAGE",
      status: "DONE",
      error: "API error (attempt 2): RESOURCE_EXHAUSTED (code 429): Resource has been exhausted.",
    }),
  ].join("\n");

  assert.deepEqual(extractLatestTranscriptRetryError(retryingTail), {
    stepIndex: 2,
    message: "API error (attempt 2): RESOURCE_EXHAUSTED (code 429): Resource has been exhausted.",
  });

  const recoveredTail = [
    retryingTail,
    JSON.stringify({
      step_index: 3,
      source: "MODEL",
      type: "PLANNER_RESPONSE",
      status: "DONE",
    }),
  ].join("\n");

  assert.equal(extractLatestTranscriptRetryError(recoveredTail), undefined);
});
