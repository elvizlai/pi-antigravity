import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import {
  type Api,
  isContextOverflow,
  isRetryableAssistantError,
  type Model,
  type TranscriptContext,
} from "@earendil-works/pi-ai";
import antigravityExtension from "../index.ts";
import { type AcpHttpMcpServer, AcpRequestStallError } from "../src/acp/client.ts";
import {
  HINDSIGHT_DISABLE_HOOKS_ENV,
  hasHostHindsightTools,
  resolveAgyChildEnv,
} from "../src/bridge/agent-files.ts";
import { AgyPiBridge } from "../src/bridge/server.ts";
import { BACKGROUND_ACP_TOOL_MESSAGE, MAX_AGY_PROMPT_BYTES } from "../src/provider/prompt.ts";
import {
  type AgyAcpClientLike,
  AgyFirstResponseTimeoutError,
  AgyQuotaError,
  AgyRuntime,
  DEFAULT_FIRST_RESPONSE_TIMEOUT_MS,
} from "../src/provider/runtime.ts";
import { AGY_STATE_ENTRY_TYPE, type AgyConversationState } from "../src/provider/state.ts";
import { createAntigravityStreamSimple, toPiUsage } from "../src/provider/stream.ts";
import { AgyReplayStore } from "../src/provider/turn.ts";
import { assertDeltasMatchPartial } from "./delta-replay.ts";

function testModel(): Model<Api> {
  return {
    id: "gemini-4-hydrogen",
    name: "Gemini 4 Hydrogen",
    api: "antigravity-acp",
    provider: "antigravity",
    baseUrl: "acp://agy",
    reasoning: true,
    input: ["text", "image"],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: 1_000_000,
    maxTokens: 65_536,
  };
}

function makeTranscript(messages: unknown[]): TranscriptContext {
  return { messages } as unknown as TranscriptContext;
}

test("AgyRuntime reuses session, falls back from missing resume to newSession, and orders mode -> model -> thought_level", async () => {
  const calls: string[] = [];
  const seenMcpServers: Array<{ method: string; mcpServers: readonly AcpHttpMcpServer[] }> = [];
  let resumeAttempts = 0;
  let currentMode = "default";
  let desiredExecutionMode: "default" | "plan" = "plan";
  let currentModel = "other";
  let currentThought = "low";

  const snapshotConfig = () => [
    {
      id: "mode",
      category: "mode",
      type: "select",
      currentValue: currentMode,
      options: [
        { value: "default", name: "Default" },
        { value: "plan", name: "Plan" },
      ],
    },
    { id: "model", category: "model", type: "select", currentValue: currentModel },
    {
      id: "thought_level",
      category: "thought_level",
      type: "select",
      currentValue: currentThought,
      options: [
        { value: "low", name: "Low" },
        { value: "high", name: "High" },
      ],
    },
  ];

  const client: AgyAcpClientLike = {
    isAlive: true,
    async initialize() {
      calls.push("initialize");
      return { protocolVersion: 1 };
    },
    async newSession(opts) {
      calls.push("newSession");
      seenMcpServers.push({ method: "newSession", mcpServers: opts.mcpServers ?? [] });
      return {
        sessionId: "acp-fresh",
        configOptions: snapshotConfig(),
      };
    },
    async resumeSession(opts) {
      resumeAttempts += 1;
      calls.push("resumeSession");
      seenMcpServers.push({ method: "resumeSession", mcpServers: opts.mcpServers ?? [] });
      throw new Error("missing session");
    },
    async setConfigOption(_sid, configId, value) {
      calls.push(`set:${configId}=${value}`);
      if (configId === "mode") currentMode = value;
      if (configId === "model") currentModel = value;
      if (configId === "thought_level") currentThought = value;
      return { configOptions: snapshotConfig() };
    },
    cancelSession() {},
    async prompt(_sid, _blocks, opts) {
      opts?.onUpdate?.({
        sessionUpdate: "agent_message_chunk",
        content: { type: "text", text: "ok" },
      });
      return {
        stopReason: "end_turn",
        usage: { inputTokens: 10, outputTokens: 2, totalTokens: 12 },
      };
    },
    async dispose() {},
  };

  const persisted: AgyConversationState[] = [];
  const runtime = new AgyRuntime({
    cwd: "/work",
    getToolMode: () => "native",
    getExecutionMode: () => "plan",
    checkBinary: async () => ({ binary: "agy", version: "1.5.13" }),
    createClient: () => client,
    onPersistedState: (s) => persisted.push(s),
  });

  const replayStore = new AgyReplayStore();
  const streamSimple = createAntigravityStreamSimple({
    getRuntime: () => runtime,
    getSessionId: () => "pi-s1",
    getBranchEntries: () => [
      {
        type: "custom",
        customType: AGY_STATE_ENTRY_TYPE,
        data: {
          version: 1,
          kind: "conversation",
          sessionId: "pi-s1",
          conversationId: "stale-acp",
          cwd: "/work",
          modelId: "gemini-4-hydrogen",
          turns: 1,
          usage: {},
        },
      },
    ],
    getExecutionMode: () => desiredExecutionMode,
    replayStore,
  });

  const s1 = streamSimple(
    testModel(),
    makeTranscript([{ role: "user", content: "hi", timestamp: 1 }]),
    { reasoning: "high" },
  );
  const events1 = [];
  for await (const ev of s1) events1.push(ev);
  assertDeltasMatchPartial(events1);
  assert.equal(resumeAttempts, 1);
  assert.deepEqual(seenMcpServers, [
    { method: "resumeSession", mcpServers: [] },
    { method: "newSession", mcpServers: [] },
  ]);
  assert.deepEqual(calls, [
    "initialize",
    "resumeSession",
    "newSession",
    "set:mode=plan",
    "set:model=gemini-4-hydrogen",
    "set:thought_level=high",
  ]);
  assert.equal(persisted[0]?.conversationId, "acp-fresh");

  // Second turn switches back to default (`/agy-plan off`) on the existing ACP session.
  desiredExecutionMode = "default";
  const s2 = streamSimple(
    testModel(),
    makeTranscript([{ role: "user", content: "next", timestamp: 2 }]),
    { reasoning: "high" },
  );
  const res2 = await s2.result();
  assert.equal(res2.stopReason, "stop");
  assert.equal(calls.filter((c) => c === "newSession").length, 1);
  assert.equal(currentMode, "default");
  assert.ok(calls.includes("set:mode=default"));
  await runtime.dispose();
});

test("first-response stall timeout cancels/disposes hung worker and lets Pi autoRetry recover on fresh worker", async () => {
  let spawnCount = 0;
  let cancelCount = 0;
  let disposeCount = 0;
  const seenEnvs: Array<NodeJS.ProcessEnv | undefined> = [];

  const runtime = new AgyRuntime({
    cwd: "/work",
    getToolMode: () => "native",
    firstResponseTimeoutMs: 40,
    checkBinary: async () => ({ binary: "agy", version: "1.5.13" }),
    createClient: (opts) => {
      spawnCount += 1;
      seenEnvs.push(opts.env);
      const attempt = spawnCount;
      let alive = true;
      return {
        get isAlive() {
          return alive;
        },
        async initialize() {
          return { protocolVersion: 1 };
        },
        async newSession() {
          return { sessionId: `sess-${attempt}` };
        },
        async resumeSession() {
          return { sessionId: `sess-${attempt}` };
        },
        async setConfigOption() {
          return {};
        },
        cancelSession() {
          cancelCount += 1;
        },
        async prompt(_sid, _blocks, promptOpts) {
          if (attempt === 1) {
            // Emit pre-turn metadata updates (including used: 0) — watchdog must NOT disarm!
            promptOpts?.onUpdate?.({ sessionUpdate: "config_option_update", configOptions: [] });
            promptOpts?.onUpdate?.({
              sessionUpdate: "available_commands_update",
              availableCommands: [],
            });
            promptOpts?.onUpdate?.({ sessionUpdate: "session_info_update", title: "Say Hi" });
            promptOpts?.onUpdate?.({
              sessionUpdate: "usage_update",
              used: 0,
              size: 1_048_576,
            });
            // Simulate agy stuck silently retrying upstream 429 before emitting any real response.
            return new Promise(() => {});
          }
          // Retry attempt emits first chunk before 40ms, then streams >40ms without false timeout!
          setTimeout(() => {
            promptOpts?.onUpdate?.({
              sessionUpdate: "agent_thought_chunk",
              content: { type: "text", text: "ok" },
            });
          }, 10);
          await new Promise((r) => setTimeout(r, 80));
          promptOpts?.onUpdate?.({
            sessionUpdate: "agent_message_chunk",
            content: { type: "text", text: "recovered" },
          });
          return { stopReason: "end_turn" };
        },
        async dispose() {
          alive = false;
          disposeCount += 1;
        },
      };
    },
  });

  const replayStore = new AgyReplayStore();
  const streamSimple = createAntigravityStreamSimple({
    getRuntime: () => runtime,
    getSessionId: () => "pi-s2",
    getBranchEntries: () => [],
    replayStore,
  });

  // Attempt 1 hits 40ms first-event watchdog -> non-stacking error after cancel & dispose.
  const firstStream = streamSimple(
    testModel(),
    makeTranscript([{ role: "user", content: "hello", timestamp: 1 }]),
    { env: { TEST_TOKEN: "abc" } },
  );
  const firstMsg = await firstStream.result();
  assert.equal(firstMsg.stopReason, "error");
  assert.match(firstMsg.errorMessage ?? "", /within .*s — cancelled stalled turn/);
  assert.equal(isRetryableAssistantError(firstMsg), false);
  assert.equal(cancelCount, 1);
  assert.equal(disposeCount, 1);

  // Subsequent turn spawns a fresh client and streams >40ms past first event cleanly.
  const secondStream = streamSimple(
    testModel(),
    makeTranscript([{ role: "user", content: "hello", timestamp: 1 }]),
    { env: { TEST_TOKEN: "abc" } },
  );
  const secondEvents = [];
  for await (const ev of secondStream) secondEvents.push(ev);
  const secondMsg = await secondStream.result();
  assertDeltasMatchPartial(secondEvents);
  assert.equal(secondMsg.stopReason, "stop");
  assert.equal(spawnCount, 2);
  assert.deepEqual(seenEnvs[0], { TEST_TOKEN: "abc" });
  await runtime.dispose();
});

test("fast transient RPC errors remain retryable for Pi built-in autoRetry", async () => {
  const runtime = new AgyRuntime({
    cwd: "/work",
    getToolMode: () => "native",
    checkBinary: async () => ({ binary: "agy", version: "1.5.13" }),
    createClient: () => ({
      isAlive: true,
      async initialize() {
        return { protocolVersion: 1 };
      },
      async newSession() {
        return { sessionId: "transient-sess" };
      },
      async resumeSession() {
        return { sessionId: "transient-sess" };
      },
      async setConfigOption() {
        return {};
      },
      cancelSession() {},
      async prompt() {
        throw new Error("503 Service Unavailable");
      },
      async dispose() {},
    }),
  });

  const streamSimple = createAntigravityStreamSimple({
    getRuntime: () => runtime,
    getSessionId: () => "pi-transient",
    getBranchEntries: () => [],
    replayStore: new AgyReplayStore(),
  });

  const msg = await streamSimple(
    testModel(),
    makeTranscript([{ role: "user", content: "hi", timestamp: 1 }]),
  ).result();
  assert.equal(msg.stopReason, "error");
  assert.equal(isRetryableAssistantError(msg), true);
  await runtime.dispose();
});

test("AgyQuotaError maps to non-retryable Pi error (0 retries) and hooks/env work", async () => {
  let spawnCount = 0;
  const observedStreamEvents: unknown[] = [];
  let payloadSawPrompt = false;

  const runtime = new AgyRuntime({
    cwd: "/work",
    getToolMode: () => "native",
    checkBinary: async () => ({ binary: "agy", version: "1.5.13" }),
    createClient: () => {
      spawnCount += 1;
      return {
        isAlive: true,
        async initialize() {
          return { protocolVersion: 1 };
        },
        async newSession() {
          return { sessionId: `q-${spawnCount}` };
        },
        async resumeSession() {
          return { sessionId: `q-${spawnCount}` };
        },
        async setConfigOption() {
          return {};
        },
        cancelSession() {},
        async prompt(_sid, _blocks, opts) {
          opts?.onUpdate?.({
            sessionUpdate: "agent_message_chunk",
            content: { type: "text", text: "⚠ Individual quota reached. Resets in 3h." },
          });
          return { stopReason: "end_turn" };
        },
        async dispose() {},
      };
    },
  });

  const streamSimple = createAntigravityStreamSimple({
    getRuntime: () => runtime,
    getSessionId: () => "pi-quota",
    getBranchEntries: () => [],
    replayStore: new AgyReplayStore(),
  });

  const s = streamSimple(
    testModel(),
    makeTranscript([{ role: "user", content: "ping", timestamp: 1 }]),
    {
      onPayload: (payload) => {
        const p = payload as { prompt: Array<{ type: string; text?: string }> };
        payloadSawPrompt = p.prompt[0]?.text === "ping";
        return { ...p, prompt: [{ type: "text", text: "rewritten" }] };
      },
      onProviderStreamEvent: (ev) => {
        observedStreamEvents.push(ev);
      },
    },
  );
  const msg = await s.result();
  assert.equal(payloadSawPrompt, true);
  assert.equal(observedStreamEvents.length > 0, true);
  assert.equal(msg.stopReason, "error");
  assert.match(msg.errorMessage ?? "", /quota exceeded/i);
  assert.equal(isRetryableAssistantError(msg), false);

  assert.equal(DEFAULT_FIRST_RESPONSE_TIMEOUT_MS, 60_000);
  assert.equal(new AgyFirstResponseTimeoutError(60_000).timeoutMs, 60_000);
  assert.match(new AgyQuotaError("Individual quota reached").message, /quota exceeded/i);
  await runtime.dispose();
});

test("bridged MCP tool call suspends wave and resumes seamlessly when Pi toolResult returns", async () => {
  const agentsDir = fs.mkdtempSync(path.join(os.tmpdir(), "pi-agy-rt-agents-"));
  const bridge = new AgyPiBridge("pi-wave");
  bridge.setToolSource(() => [{ name: "read", description: "read", parameters: {} }]);
  bridge.refreshTools();
  await bridge.start();
  let seenNewMcp: readonly AcpHttpMcpServer[] | undefined;

  const runtime = new AgyRuntime({
    cwd: "/work",
    agentsDir,
    bridge,
    bridgeToken: "tok-test",
    getToolMode: () => "pi",
    ensureBridge: async () => true,
    checkBinary: async () => ({ binary: "agy", version: "1.5.13" }),
    createClient: () => ({
      isAlive: true,
      async initialize() {
        return { protocolVersion: 1 };
      },
      async newSession(opts) {
        seenNewMcp = opts.mcpServers;
        return { sessionId: "bridge-sess" };
      },
      async resumeSession() {
        return { sessionId: "bridge-sess" };
      },
      async setConfigOption() {
        return {};
      },
      cancelSession() {},
      async prompt(_sid, _blocks, opts) {
        opts?.onUpdate?.({
          sessionUpdate: "usage_update",
          used: 2771,
          size: 1_048_576,
        });
        // Call the real loopback HTTP bridge like agy does!
        const res = await fetch(bridge.url!, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({
            jsonrpc: "2.0",
            id: 1,
            method: "tools/call",
            params: { name: "pi__read", arguments: { path: "note.txt" } },
          }),
        });
        const json = (await res.json()) as {
          result: { content: Array<{ text: string }> };
        };
        opts?.onUpdate?.({
          sessionUpdate: "usage_update",
          used: 3016,
          size: 1_048_576,
        });
        opts?.onUpdate?.({
          sessionUpdate: "agent_message_chunk",
          content: { type: "text", text: `read:${json.result.content[0]?.text}` },
        });
        return {
          stopReason: "end_turn",
          usage: {
            inputTokens: 7030,
            outputTokens: 250,
            cachedReadTokens: 1424,
            totalTokens: 8704,
          },
        };
      },
      async dispose() {},
    }),
  });

  const replayStore = new AgyReplayStore();
  let messages: unknown[] = [{ role: "user", content: "read note.txt", timestamp: 1 }];
  const streamSimple = createAntigravityStreamSimple({
    getRuntime: () => runtime,
    getSessionId: () => "pi-bridge-1",
    getBranchEntries: () => [],
    replayStore,
  });

  try {
    const wave1 = streamSimple(testModel(), makeTranscript(messages));
    const events1 = [];
    for await (const ev of wave1) events1.push(ev);
    assertDeltasMatchPartial(events1);
    const msg1 = await wave1.result();
    assert.equal(msg1.stopReason, "toolUse");
    assert.equal(msg1.usage.totalTokens, 2771);
    assert.equal(msg1.usage.output, 0);
    const toolCall = msg1.content.find((b) => b.type === "toolCall");
    assert.ok(toolCall && toolCall.type === "toolCall");
    assert.equal(toolCall.name, "read");

    // Simulate local Pi tool execution pause between waves (excluded from multi-wave LLM durationMs).
    await new Promise((r) => setTimeout(r, 80));

    messages = [
      ...messages,
      msg1,
      {
        role: "toolResult",
        toolCallId: toolCall.id,
        toolName: "read",
        content: [{ type: "text", text: "contents" }],
        isError: false,
        timestamp: 2,
      },
    ];

    const wave2 = streamSimple(testModel(), makeTranscript(messages));
    const events2 = [];
    for await (const ev of wave2) events2.push(ev);
    assertDeltasMatchPartial(events2);
    const msg2 = await wave2.result();
    assert.equal(msg2.stopReason, "stop");
    assert.equal(msg2.content[0]?.type === "text" ? msg2.content[0].text : "", "read:contents");
    assert.equal(msg2.usage.totalTokens, 3016);
    assert.equal(msg2.usage.input, 2436);
    assert.equal(msg2.usage.output, 87);
    assert.equal(msg2.usage.cacheRead, 493);
    assert.equal(msg2.usage.input + msg2.usage.cacheRead + msg2.usage.output, 3016);
    assert.equal(typeof msg2.durationMs, "number");
    assert.ok((msg2.durationMs ?? Number.POSITIVE_INFINITY) < 70);
    assert.deepEqual(seenNewMcp, [
      {
        type: "http",
        name: "pi-wave",
        url: bridge.url,
        headers: [{ name: "x-pi-bridge-token", value: "tok-test" }],
      },
    ]);
  } finally {
    await runtime.dispose();
    await bridge.close();
    fs.rmSync(agentsDir, { recursive: true, force: true });
  }
});

test("multi-step cumulative PromptResponse.usage scales to usage_update.used without triggering false silent overflow", async () => {
  const runtime = new AgyRuntime({
    cwd: "/work",
    getToolMode: () => "native",
    checkBinary: async () => ({ binary: "agy", version: "1.5.14" }),
    createClient: () => ({
      isAlive: true,
      async initialize() {
        return { protocolVersion: 1 };
      },
      async newSession() {
        return { sessionId: "scaled-usage-sess" };
      },
      async resumeSession() {
        return { sessionId: "scaled-usage-sess" };
      },
      async setConfigOption() {
        return {};
      },
      cancelSession() {},
      async prompt(_sid, _blocks, opts) {
        opts?.onUpdate?.({ sessionUpdate: "usage_update", used: 69_916, size: 1_048_576 });
        opts?.onUpdate?.({
          sessionUpdate: "agent_message_chunk",
          content: { type: "text", text: "done" },
        });
        return {
          stopReason: "end_turn",
          usage: {
            inputTokens: 101_108,
            outputTokens: 22_608,
            cachedReadTokens: 1_063_287,
            thoughtTokens: 6_619,
            totalTokens: 1_187_003,
          },
        };
      },
      async dispose() {},
    }),
  });

  const streamSimple = createAntigravityStreamSimple({
    getRuntime: () => runtime,
    getSessionId: () => "pi-scaled-usage",
    getBranchEntries: () => [],
    replayStore: new AgyReplayStore(),
  });

  try {
    const model = testModel();
    const s = streamSimple(
      model,
      makeTranscript([{ role: "user", content: "status", timestamp: 1 }]),
    );
    const msg = await s.result();
    assert.equal(msg.stopReason, "stop");
    assert.equal(msg.usage.totalTokens, 69_916);
    assert.equal(msg.usage.output, 1_332);
    assert.equal(msg.usage.reasoning, 390);
    assert.equal(msg.usage.cacheRead, 62_629);
    assert.equal(msg.usage.input, 5_955);
    assert.equal(msg.usage.input + msg.usage.cacheRead + msg.usage.output, 69_916);
    assert.equal(isContextOverflow(msg, 1_000_000), false);

    const singleStep = toPiUsage(
      {
        input_tokens: 3_113,
        cache_read_tokens: 4_662,
        output_tokens: 27,
        thinking_tokens: 10,
        total_tokens: 7_802,
      },
      model,
      7_802,
    );
    assert.equal(singleStep.input, 3_113);
    assert.equal(singleStep.cacheRead, 4_662);
    assert.equal(singleStep.output, 27);
    assert.equal(singleStep.reasoning, 10);
    assert.equal(singleStep.totalTokens, 7_802);

    const realOverflow = toPiUsage(
      {
        input_tokens: 150_000,
        cache_read_tokens: 1_900_000,
        output_tokens: 4_000,
        total_tokens: 2_054_000,
      },
      model,
      1_050_000,
    );
    assert.equal(isContextOverflow({ ...msg, usage: realOverflow }, 1_000_000), true);
  } finally {
    await runtime.dispose();
  }
});

test("upstream ⚠ error banners after tool waves fail with stopReason error (400 non-retryable, 503 retryable) and zero usage", async () => {
  const banners = [
    "\n⚠ FAILED_PRECONDITION (code 400): User location is not supported for the API use.\n",
    "\n⚠ UNAVAILABLE (code 503): Service temporarily unavailable.\n",
  ];

  for (const [idx, banner] of banners.entries()) {
    const agentsDir = fs.mkdtempSync(path.join(os.tmpdir(), "pi-agy-banner-agents-"));
    const bridge = new AgyPiBridge(`pi-banner-${idx}`);
    bridge.setToolSource(() => [{ name: "read", description: "read", parameters: {} }]);
    bridge.refreshTools();
    await bridge.start();

    const runtime = new AgyRuntime({
      cwd: "/work",
      agentsDir,
      bridge,
      getToolMode: () => "pi",
      ensureBridge: async () => true,
      checkBinary: async () => ({ binary: "agy", version: "1.5.14" }),
      createClient: () => ({
        isAlive: true,
        async initialize() {
          return { protocolVersion: 1 };
        },
        async newSession() {
          return { sessionId: `banner-sess-${idx}` };
        },
        async resumeSession() {
          return { sessionId: `banner-sess-${idx}` };
        },
        async setConfigOption() {
          return {};
        },
        cancelSession() {},
        async prompt(_sid, _blocks, opts) {
          opts?.onUpdate?.({ sessionUpdate: "usage_update", used: 25_000, size: 1_048_576 });
          await fetch(bridge.url!, {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify({
              jsonrpc: "2.0",
              id: 1,
              method: "tools/call",
              params: { name: "pi__read", arguments: { path: "note.txt" } },
            }),
          });
          // Final step fails without agent_thought_chunk or usage_update.
          opts?.onUpdate?.({
            sessionUpdate: "agent_message_chunk",
            content: { type: "text", text: banner },
          });
          return {
            stopReason: "end_turn",
            usage: {
              inputTokens: 125_102,
              outputTokens: 25_705,
              cachedReadTokens: 1_249_616,
              totalTokens: 1_400_423,
            },
          };
        },
        async dispose() {},
      }),
    });

    const streamSimple = createAntigravityStreamSimple({
      getRuntime: () => runtime,
      getSessionId: () => `pi-banner-${idx}`,
      getBranchEntries: () => [],
      replayStore: new AgyReplayStore(),
    });

    try {
      let messages: unknown[] = [{ role: "user", content: "check", timestamp: 1 }];
      const wave1 = streamSimple(testModel(), makeTranscript(messages));
      const msg1 = await wave1.result();
      assert.equal(msg1.stopReason, "toolUse");
      const toolCall = msg1.content.find((b) => b.type === "toolCall");
      assert.ok(toolCall && toolCall.type === "toolCall");

      messages = [
        ...messages,
        msg1,
        {
          role: "toolResult",
          toolCallId: toolCall.id,
          toolName: "read",
          content: [{ type: "text", text: "ok" }],
          isError: false,
          timestamp: 2,
        },
      ];

      const wave2 = streamSimple(testModel(), makeTranscript(messages));
      const msg2 = await wave2.result();
      assert.equal(msg2.stopReason, "error");
      assert.equal(msg2.usage.totalTokens, 0);
      assert.equal(isRetryableAssistantError(msg2), idx === 1);
      if (idx === 0) {
        assert.match(msg2.errorMessage ?? "", /FAILED_PRECONDITION \(code 400\)/);
      } else {
        assert.match(msg2.errorMessage ?? "", /UNAVAILABLE \(code 503\)/);
      }
    } finally {
      await runtime.dispose();
      await bridge.close();
      fs.rmSync(agentsDir, { recursive: true, force: true });
    }
  }
});

test("native wrapper tool replay store preserves output, explicit failures, and background tool status", async () => {
  const replayStore = new AgyReplayStore();
  const workingStatuses: Array<string | undefined> = [];
  let promptCalls = 0;
  const runtime = new AgyRuntime({
    cwd: "/work",
    getToolMode: () => "native",
    onWorkingStatus: (msg) => workingStatuses.push(msg),
    checkBinary: async () => ({ binary: "agy", version: "1.5.13" }),
    createClient: () => ({
      isAlive: true,
      async initialize() {
        return { protocolVersion: 1 };
      },
      async newSession() {
        return { sessionId: "native-sess" };
      },
      async resumeSession() {
        return { sessionId: "native-sess" };
      },
      async setConfigOption() {
        return {};
      },
      cancelSession() {},
      async prompt(_sid, _blocks, opts) {
        promptCalls += 1;
        opts?.onUpdate?.({
          sessionUpdate: "tool_call",
          toolCallId: "n1",
          title: "Read a.txt",
          kind: "read",
          locations: [{ path: "a.txt" }],
          status: "in_progress",
          rawInput: { path: "a.txt" },
        });
        opts?.onUpdate?.({
          sessionUpdate: "tool_call_update",
          toolCallId: "n1",
          status: "completed",
          content: [{ type: "content", content: { type: "text", text: "line1" } }],
        });
        opts?.onUpdate?.({
          sessionUpdate: "tool_call",
          toolCallId: "f1",
          title: "Run bad-cmd",
          kind: "execute",
          status: "in_progress",
          rawInput: { cmd: "bad-cmd" },
        });
        opts?.onUpdate?.({
          sessionUpdate: "tool_call_update",
          toolCallId: "f1",
          status: "failed",
          rawOutput: { error: "exit code 1" },
        });
        opts?.onUpdate?.({
          sessionUpdate: "tool_call",
          toolCallId: "d1",
          title: "Run sleep 10",
          kind: "execute",
          status: "in_progress",
          rawInput: { cmd: "sleep 10" },
        });
        return {
          stopReason: "end_turn",
          usage: { inputTokens: 100, outputTokens: 20, totalTokens: 120 },
        };
      },
      async dispose() {},
    }),
  });

  const streamSimple = createAntigravityStreamSimple({
    getRuntime: () => runtime,
    getSessionId: () => "pi-native-1",
    getBranchEntries: () => [],
    replayStore,
  });

  const messages: unknown[] = [{ role: "user", content: "go", timestamp: 1 }];
  const s = streamSimple(testModel(), makeTranscript(messages));
  const events = [];
  for await (const ev of s) events.push(ev);
  assertDeltasMatchPartial(events);
  const msg1 = await s.result();
  assert.equal(msg1.stopReason, "toolUse");
  assert.equal(
    msg1.content.some((b) => b.type === "text"),
    false,
  );
  const calls = msg1.content.filter((b) => b.type === "toolCall");
  assert.equal(calls.length, 3);
  assert.equal(replayStore.take(calls[0].id)?.output, "line1");
  assert.equal(replayStore.take(calls[1].id)?.error, "exit code 1");
  const bgRecorded = replayStore.take(calls[2].id);
  assert.equal(bgRecorded?.output, BACKGROUND_ACP_TOOL_MESSAGE);
  assert.equal(bgRecorded?.error, undefined);
  assert.equal(workingStatuses[0], "Read a.txt…");
  assert.ok(workingStatuses.includes("Run sleep 10…"));
  assert.equal(workingStatuses.at(-1), undefined);

  const followMessages = [
    ...messages,
    msg1,
    ...calls.map((call, idx) => ({
      role: "toolResult",
      toolCallId: call.id,
      toolName: "antigravity",
      content: [
        {
          type: "text",
          text: idx === 0 ? "line1" : idx === 1 ? "exit code 1" : BACKGROUND_ACP_TOOL_MESSAGE,
        },
      ],
      isError: idx === 1,
      timestamp: 2 + idx,
    })),
  ];
  const wave2 = streamSimple(testModel(), makeTranscript(followMessages));
  const events2 = [];
  for await (const ev of wave2) events2.push(ev);
  assertDeltasMatchPartial(events2);
  const msg2 = await wave2.result();
  assert.equal(promptCalls, 1);
  assert.equal(msg2.stopReason, "stop");
  assert.deepEqual(msg2.content, []);
  assert.equal(msg2.usage.totalTokens, 120);
  await runtime.dispose();
});

test("pending ACP tools settle naturally across turns while background tools flush before text and preserve same-turn completion", async () => {
  const replayStore = new AgyReplayStore();
  let promptCalls = 0;
  const runtime = new AgyRuntime({
    cwd: "/work",
    getToolMode: () => "native",
    checkBinary: async () => ({ binary: "agy", version: "1.5.14" }),
    createClient: () => ({
      isAlive: true,
      async initialize() {
        return { protocolVersion: 1 };
      },
      async newSession() {
        return { sessionId: "native-deferred-1" };
      },
      async resumeSession() {
        return { sessionId: "native-deferred-1" };
      },
      async setConfigOption() {
        return {};
      },
      cancelSession() {},
      async prompt(_sid, _blocks, opts) {
        promptCalls += 1;
        if (promptCalls === 1) {
          // Part A Turn 1: pending tool (ask_question) awaits follow-up user input.
          opts?.onUpdate?.({
            sessionUpdate: "tool_call",
            toolCallId: "q1",
            title: "Run ask_question",
            kind: "other",
            status: "in_progress",
            rawInput: { questions: [{ question: "Storage?" }] },
          });
          opts?.onUpdate?.({
            sessionUpdate: "tool_call_update",
            toolCallId: "q1",
            status: "pending",
          });
          opts?.onUpdate?.({
            sessionUpdate: "agent_message_chunk",
            content: { type: "text", text: "Question text?" },
          });
          return {
            stopReason: "end_turn",
            usage: { inputTokens: 100, outputTokens: 25, totalTokens: 125 },
          };
        }

        if (promptCalls === 2) {
          // Part A Turn 2: real completed update arrives for q1 when user answers.
          opts?.onUpdate?.({
            sessionUpdate: "tool_call_update",
            toolCallId: "q1",
            title: "Run ask_question",
            kind: "other",
            status: "completed",
          });
          opts?.onUpdate?.({
            sessionUpdate: "agent_message_chunk",
            content: { type: "text", text: "Agreed answer." },
          });
          return {
            stopReason: "end_turn",
            usage: { inputTokens: 120, outputTokens: 30, totalTokens: 150 },
          };
        }

        // Part B Turn 3: in_progress background tool flushes before interim text, then completes in the SAME turn.
        opts?.onUpdate?.({
          sessionUpdate: "tool_call",
          toolCallId: "bg1",
          title: "Run run_workflow",
          kind: "other",
          status: "in_progress",
          rawInput: { Description: "bg job" },
        });
        opts?.onUpdate?.({
          sessionUpdate: "agent_message_chunk",
          content: { type: "text", text: "Started in background." },
        });
        opts?.onUpdate?.({
          sessionUpdate: "tool_call_update",
          toolCallId: "bg1",
          title: "Run run_workflow",
          kind: "other",
          status: "completed",
          content: [{ type: "content", content: { type: "text", text: "workflow result: 42" } }],
        });
        opts?.onUpdate?.({
          sessionUpdate: "agent_message_chunk",
          content: { type: "text", text: "Final: 42." },
        });
        return {
          stopReason: "end_turn",
          usage: { inputTokens: 140, outputTokens: 35, totalTokens: 175 },
        };
      },
      async dispose() {},
    }),
  });

  const streamSimple = createAntigravityStreamSimple({
    getRuntime: () => runtime,
    getSessionId: () => "pi-deferred-1",
    getBranchEntries: () => [],
    replayStore,
  });

  // Part A Turn 1: pending ask_question is NOT materialized early; question text completes in 1 wave.
  const turn1Messages: unknown[] = [{ role: "user", content: "/grill-me", timestamp: 1 }];
  const t1 = streamSimple(testModel(), makeTranscript(turn1Messages));
  for await (const _ of t1) {
  }
  const msg1 = await t1.result();
  assert.equal(msg1.stopReason, "stop");
  assert.deepEqual(msg1.content, [{ type: "text", text: "Question text?" }]);
  assert.equal(msg1.usage.totalTokens, 125);

  // Part A Turn 2: real completed event for ask_question renders once in Wave 1, reply in Wave 2.
  const turn2Messages: unknown[] = [
    ...turn1Messages,
    msg1,
    { role: "user", content: "1", timestamp: 2 },
  ];
  const t2w1 = streamSimple(testModel(), makeTranscript(turn2Messages));
  for await (const _ of t2w1) {
  }
  const msg2w1 = await t2w1.result();
  assert.equal(msg2w1.stopReason, "toolUse");
  const t2Calls = msg2w1.content.filter((b) => b.type === "toolCall");
  assert.equal(t2Calls.length, 1);
  assert.equal(t2Calls[0].arguments.tool, "Run ask_question");

  const t2Follow = [
    ...turn2Messages,
    msg2w1,
    {
      role: "toolResult",
      toolCallId: t2Calls[0].id,
      toolName: "antigravity",
      content: [{ type: "text", text: "" }],
      isError: false,
      timestamp: 3,
    },
  ];
  const t2w2 = streamSimple(testModel(), makeTranscript(t2Follow));
  for await (const _ of t2w2) {
  }
  const msg2w2 = await t2w2.result();
  assert.equal(msg2w2.stopReason, "stop");
  assert.deepEqual(msg2w2.content, [{ type: "text", text: "Agreed answer." }]);

  // Part B Turn 3: background tool flushes before interim text, and real same-turn completion is NOT dropped.
  const turn3Messages: unknown[] = [
    ...t2Follow,
    msg2w2,
    { role: "user", content: "run bg", timestamp: 4 },
  ];
  const t3w1 = streamSimple(testModel(), makeTranscript(turn3Messages));
  for await (const _ of t3w1) {
  }
  const msg3w1 = await t3w1.result();
  assert.equal(msg3w1.stopReason, "toolUse");
  const t3w1Calls = msg3w1.content.filter((b) => b.type === "toolCall");
  assert.equal(t3w1Calls.length, 1);
  assert.equal(replayStore.take(t3w1Calls[0].id)?.output, BACKGROUND_ACP_TOOL_MESSAGE);

  const t3Follow1 = [
    ...turn3Messages,
    msg3w1,
    {
      role: "toolResult",
      toolCallId: t3w1Calls[0].id,
      toolName: "antigravity",
      content: [{ type: "text", text: BACKGROUND_ACP_TOOL_MESSAGE }],
      isError: false,
      timestamp: 5,
    },
  ];
  const t3w2 = streamSimple(testModel(), makeTranscript(t3Follow1));
  for await (const _ of t3w2) {
  }
  const msg3w2 = await t3w2.result();
  assert.equal(msg3w2.stopReason, "toolUse");
  assert.equal(
    msg3w2.content
      .filter((b) => b.type === "text")
      .map((b) => b.text)
      .join(""),
    "Started in background.",
  );
  const t3w2Calls = msg3w2.content.filter((b) => b.type === "toolCall");
  assert.equal(t3w2Calls.length, 1);
  assert.equal(replayStore.take(t3w2Calls[0].id)?.output, "workflow result: 42");

  const t3Follow2 = [
    ...t3Follow1,
    msg3w2,
    {
      role: "toolResult",
      toolCallId: t3w2Calls[0].id,
      toolName: "antigravity",
      content: [{ type: "text", text: "workflow result: 42" }],
      isError: false,
      timestamp: 6,
    },
  ];
  const t3w3 = streamSimple(testModel(), makeTranscript(t3Follow2));
  for await (const _ of t3w3) {
  }
  const msg3w3 = await t3w3.result();
  assert.equal(msg3w3.stopReason, "stop");
  assert.deepEqual(msg3w3.content, [{ type: "text", text: "Final: 42." }]);
  assert.equal(promptCalls, 3);
  await runtime.dispose();
});

test("native acp:agy multi-step tools split waves at thinking/text boundaries so final reply renders last", async () => {
  const replayStore = new AgyReplayStore();
  let promptCalls = 0;
  const runtime = new AgyRuntime({
    cwd: "/work",
    getToolMode: () => "native",
    checkBinary: async () => ({ binary: "agy", version: "1.5.14" }),
    createClient: () => ({
      isAlive: true,
      async initialize() {
        return { protocolVersion: 1 };
      },
      async newSession() {
        return { sessionId: "native-multi-wave" };
      },
      async resumeSession() {
        return { sessionId: "native-multi-wave" };
      },
      async setConfigOption() {
        return {};
      },
      cancelSession() {},
      async prompt(_sid, _blocks, opts) {
        promptCalls += 1;
        opts?.onUpdate?.({
          sessionUpdate: "agent_thought_chunk",
          content: { type: "text", text: "thinking_1" },
        });
        opts?.onUpdate?.({
          sessionUpdate: "tool_call",
          toolCallId: "t1",
          title: "Run check 1",
          kind: "execute",
          status: "in_progress",
        });
        opts?.onUpdate?.({
          sessionUpdate: "tool_call_update",
          toolCallId: "t1",
          status: "completed",
          content: [{ type: "content", content: { type: "text", text: "out1" } }],
        });
        opts?.onUpdate?.({
          sessionUpdate: "agent_thought_chunk",
          content: { type: "text", text: "thinking_2" },
        });
        opts?.onUpdate?.({
          sessionUpdate: "tool_call",
          toolCallId: "t2",
          title: "Read file 2",
          kind: "read",
          status: "in_progress",
        });
        opts?.onUpdate?.({
          sessionUpdate: "tool_call_update",
          toolCallId: "t2",
          status: "completed",
          content: [{ type: "content", content: { type: "text", text: "out2" } }],
        });
        opts?.onUpdate?.({
          sessionUpdate: "tool_call",
          toolCallId: "t3",
          title: "Read file 3",
          kind: "read",
          status: "in_progress",
        });
        opts?.onUpdate?.({
          sessionUpdate: "tool_call_update",
          toolCallId: "t3",
          status: "completed",
          content: [{ type: "content", content: { type: "text", text: "out3" } }],
        });
        opts?.onUpdate?.({
          sessionUpdate: "agent_thought_chunk",
          content: { type: "text", text: "thinking_3" },
        });
        opts?.onUpdate?.({
          sessionUpdate: "agent_message_chunk",
          content: { type: "text", text: "Final report" },
        });
        return {
          stopReason: "end_turn",
          usage: { inputTokens: 300, outputTokens: 50, totalTokens: 350 },
        };
      },
      async dispose() {},
    }),
  });

  const streamSimple = createAntigravityStreamSimple({
    getRuntime: () => runtime,
    getSessionId: () => "pi-native-multi",
    getBranchEntries: () => [],
    replayStore,
  });

  try {
    let messages: unknown[] = [{ role: "user", content: "audit repo", timestamp: 1 }];

    const wave1 = streamSimple(testModel(), makeTranscript(messages));
    const events1 = [];
    for await (const ev of wave1) events1.push(ev);
    assertDeltasMatchPartial(events1);
    const msg1 = await wave1.result();
    assert.equal(msg1.stopReason, "toolUse");
    assert.equal(msg1.content.length, 2);
    assert.deepEqual(msg1.content[0], { type: "thinking", thinking: "thinking_1" });
    assert.equal(msg1.content[1]?.type, "toolCall");
    const call1 = msg1.content[1];
    assert.ok(call1?.type === "toolCall");
    assert.equal(replayStore.take(call1.id)?.output, "out1");

    messages = [
      ...messages,
      msg1,
      {
        role: "toolResult",
        toolCallId: call1.id,
        toolName: "antigravity",
        content: [{ type: "text", text: "out1" }],
        isError: false,
        timestamp: 2,
      },
    ];

    const wave2 = streamSimple(testModel(), makeTranscript(messages));
    const events2 = [];
    for await (const ev of wave2) events2.push(ev);
    assertDeltasMatchPartial(events2);
    const msg2 = await wave2.result();
    assert.equal(msg2.stopReason, "toolUse");
    assert.equal(msg2.content.length, 3);
    assert.deepEqual(msg2.content[0], { type: "thinking", thinking: "thinking_2" });
    const calls2 = msg2.content.filter((b) => b.type === "toolCall");
    assert.equal(calls2.length, 2);
    assert.equal(replayStore.take(calls2[0].id)?.output, "out2");
    assert.equal(replayStore.take(calls2[1].id)?.output, "out3");

    messages = [
      ...messages,
      msg2,
      ...calls2.map((call, idx) => ({
        role: "toolResult",
        toolCallId: call.id,
        toolName: "antigravity",
        content: [{ type: "text", text: idx === 0 ? "out2" : "out3" }],
        isError: false,
        timestamp: 3 + idx,
      })),
    ];

    const wave3 = streamSimple(testModel(), makeTranscript(messages));
    const events3 = [];
    for await (const ev of wave3) events3.push(ev);
    assertDeltasMatchPartial(events3);
    const msg3 = await wave3.result();
    assert.equal(promptCalls, 1);
    assert.equal(msg3.stopReason, "stop");
    assert.deepEqual(msg3.content, [
      { type: "thinking", thinking: "thinking_3" },
      { type: "text", text: "Final report" },
    ]);
    assert.equal(msg3.usage.totalTokens, 350);
  } finally {
    await runtime.dispose();
  }
});

test("warmupSession completes setup before streamSimple and reloads client when bridge catalog changes", async () => {
  const agentsDir = fs.mkdtempSync(path.join(os.tmpdir(), "pi-agy-warmup-agents-"));
  let spawnCount = 0;
  const calls: string[] = [];
  const sessionMcpCalls: Array<{
    method: "newSession" | "resumeSession";
    mcpServers: readonly AcpHttpMcpServer[];
  }> = [];
  const bridge = new AgyPiBridge("pi-warmup-test");
  bridge.setToolSource(() => [{ name: "read", description: "read file", parameters: {} }]);
  bridge.refreshTools();
  await bridge.start();

  try {
    let aliveClient: { isAlive: boolean } | undefined;
    const runtime = new AgyRuntime({
      cwd: "/work",
      agentsDir,
      bridge,
      bridgeToken: "tok-warmup",
      getToolMode: () => "pi",
      checkBinary: async () => ({ binary: "agy", version: "1.5.13" }),
      createClient: () => {
        spawnCount += 1;
        const clientId = spawnCount;
        const client: AgyAcpClientLike & { isAlive: boolean } = {
          isAlive: true,
          async initialize() {
            calls.push(`init:${clientId}`);
            return { protocolVersion: 1 };
          },
          async newSession(opts) {
            calls.push(`new:${clientId}`);
            sessionMcpCalls.push({ method: "newSession", mcpServers: opts.mcpServers ?? [] });
            return {
              sessionId: `sess-${clientId}`,
              configOptions: [
                {
                  id: "mode",
                  category: "mode",
                  type: "select",
                  currentValue: "default",
                  options: [
                    { value: "default", name: "Default" },
                    { value: "plan", name: "Plan" },
                  ],
                },
                {
                  id: "model",
                  category: "model",
                  type: "select",
                  currentValue: "gemini-4-hydrogen",
                },
              ],
            };
          },
          async resumeSession(opts) {
            const { sessionId } = opts;
            calls.push(`resume:${clientId}:${sessionId}`);
            sessionMcpCalls.push({
              method: "resumeSession",
              mcpServers: opts.mcpServers ?? [],
            });
            return {
              sessionId,
              configOptions: [
                {
                  id: "mode",
                  category: "mode",
                  type: "select",
                  currentValue: "default",
                  options: [
                    { value: "default", name: "Default" },
                    { value: "plan", name: "Plan" },
                  ],
                },
                {
                  id: "model",
                  category: "model",
                  type: "select",
                  currentValue: "gemini-4-hydrogen",
                },
              ],
            };
          },
          async setConfigOption(_sid: string, configId: string, value: string) {
            calls.push(`set:${clientId}:${configId}=${value}`);
            return { configOptions: [] };
          },
          cancelSession() {},
          async prompt(
            _sid: string,
            _blocks: unknown,
            opts?: { onUpdate?: (u: Record<string, unknown>) => void },
          ) {
            calls.push(`prompt:${clientId}`);
            opts?.onUpdate?.({
              sessionUpdate: "agent_message_chunk",
              content: { type: "text", text: "hi" },
            });
            return {
              stopReason: "end_turn",
              usage: { inputTokens: 10, outputTokens: 2, totalTokens: 12 },
            };
          },
          async dispose() {
            client.isAlive = false;
            calls.push(`dispose:${clientId}`);
          },
        };
        aliveClient = client;
        return client;
      },
    });

    await runtime.warmupSession({
      modelId: "gemini-4-hydrogen",
      executionMode: "default",
      sessionId: "pi-warmup-1",
      branchEntries: [],
    });
    assert.deepEqual(calls, ["init:1", "new:1"]);

    calls.length = 0;
    const streamSimple = createAntigravityStreamSimple({
      getRuntime: () => runtime,
      getSessionId: () => "pi-warmup-1",
      getBranchEntries: () => [],
      replayStore: new AgyReplayStore(),
    });
    const wave1 = streamSimple(
      testModel(),
      makeTranscript([{ role: "user", content: "hi", timestamp: 1 }]),
    );
    for await (const _ev of wave1) {
      // Drain stream events.
    }
    const msg1 = await wave1.result();
    assert.equal(msg1.stopReason, "stop");
    assert.deepEqual(calls, ["prompt:1"]);

    // Changing dynamic bridge tools bumps catalogRevision and forces a fresh agy acp process.
    bridge.setDynamicTools([
      {
        name: "activate_skill",
        description: "load skill",
        parameters: { type: "object", properties: {} },
        handler: async () => ({ content: "ok", isError: false }),
      },
    ]);
    calls.length = 0;
    const wave2 = streamSimple(
      testModel(),
      makeTranscript([
        { role: "user", content: "hi", timestamp: 1 },
        { role: "assistant", content: [{ type: "text", text: "hi" }], timestamp: 2 },
        { role: "user", content: "again", timestamp: 3 },
      ]),
    );
    for await (const _ev of wave2) {
      // Drain stream events.
    }
    await wave2.result();
    assert.equal(spawnCount, 2);
    assert.ok(calls.includes("dispose:1"));
    assert.ok(calls.includes("init:2"));
    assert.ok(calls.includes("prompt:2"));
    assert.equal(aliveClient?.isAlive, true);
    assert.deepEqual(sessionMcpCalls, [
      {
        method: "newSession",
        mcpServers: [
          {
            type: "http",
            name: "pi-warmup-test",
            url: bridge.url!,
            headers: [{ name: "x-pi-bridge-token", value: "tok-warmup" }],
          },
        ],
      },
      {
        method: "resumeSession",
        mcpServers: [
          {
            type: "http",
            name: "pi-warmup-test",
            url: bridge.url!,
            headers: [{ name: "x-pi-bridge-token", value: "tok-warmup" }],
          },
        ],
      },
    ]);
    await runtime.dispose();
  } finally {
    await bridge.close();
    fs.rmSync(agentsDir, { recursive: true, force: true });
  }
});

test("antigravityExtension defers CLI flag reads until after Pi populates flag values", async () => {
  const flagValues = new Map<string, unknown>();
  const handlers = new Map<
    string,
    Array<(event: unknown, ctx: unknown) => Promise<unknown> | unknown>
  >();
  const commands = new Map<string, { handler: (args: string, ctx: unknown) => Promise<void> }>();
  let registeredTool:
    | {
        execute: (
          id: string,
          params: { tool: string; kind?: string; input: Record<string, unknown> },
        ) => Promise<{ terminate?: boolean }>;
      }
    | undefined;

  const mockPi = {
    registerFlag() {},
    getFlag(name: string) {
      return flagValues.get(name);
    },
    getAllTools: () => [],
    getActiveTools: () => [],
    setActiveTools() {},
    getCommands: () => [],
    registerProvider() {},
    registerTool(def: typeof registeredTool) {
      registeredTool = def;
    },
    registerEntryRenderer() {},
    registerCommand(name: string, def: { handler: (args: string, ctx: unknown) => Promise<void> }) {
      commands.set(name, def);
    },
    on(event: string, handler: (event: unknown, ctx: unknown) => Promise<unknown> | unknown) {
      const list = handlers.get(event) ?? [];
      list.push(handler);
      handlers.set(event, list);
    },
    appendEntry() {},
  };

  antigravityExtension(mockPi as unknown as Parameters<typeof antigravityExtension>[0]);

  assert.ok(registeredTool);
  const missingToolRes = await registeredTool.execute("missing-id", { tool: "Read a", input: {} });
  assert.equal(missingToolRes.terminate, undefined);

  // Pi populates CLI flag values after extension factory execution.
  flagValues.set("agy-tools", "agy");
  flagValues.set("agy-plan", true);

  const statuses: Array<string | undefined> = [];
  const notifications: string[] = [];
  const mockCtx = {
    hasUI: true,
    cwd: "/work",
    model: { provider: "antigravity", id: "gemini-4-hydrogen" },
    sessionManager: { getSessionId: () => "pi-flag-test", getBranch: () => [] },
    ui: {
      theme: { fg: (_level: string, text: string) => text },
      setStatus: (_id: string, text: string | undefined) => statuses.push(text),
      notify: (msg: string) => notifications.push(msg),
    },
  };

  try {
    for (const handler of handlers.get("session_start") ?? []) {
      await handler({}, { ...mockCtx, hasUI: false, model: { provider: "other", id: "m" } });
    }
    await commands.get("agy-doctor")?.handler("", mockCtx);
    const doctorReport = notifications[0] ?? "";
    assert.match(doctorReport, /- Tool mode: native/);
    assert.match(doctorReport, /- Execution mode: plan/);

    const statusHandler = commands.get("agy-tools")?.handler;
    notifications.length = 0;
    await statusHandler?.("", mockCtx);
    assert.match(notifications[0] ?? "", /acp:agy/);

    // Non-antigravity status update is cleared; antigravity renders acp:agy plan.
    for (const handler of handlers.get("model_select") ?? []) {
      await handler({}, { ...mockCtx, model: { provider: "other", id: "m" } });
    }
    assert.equal(statuses.at(-1), undefined);
  } finally {
    for (const handler of handlers.get("session_shutdown") ?? []) {
      await handler({}, mockCtx);
    }
  }
});

test("setup stall and abort during resumeSession dispose client without wiping stored conversationId", async () => {
  let disposeCount = 0;
  let newSessionCalls = 0;
  const resumedIds: string[] = [];
  let mode: "hang_init" | "hang_resume" | "ok" = "hang_init";

  const runtime = new AgyRuntime({
    cwd: "/work",
    getToolMode: () => "native",
    firstResponseTimeoutMs: 40,
    checkBinary: async () => ({ binary: "agy", version: "1.5.14" }),
    createClient: () => {
      let alive = true;
      return {
        get isAlive() {
          return alive;
        },
        async initialize(opts) {
          if (mode === "hang_init") {
            return new Promise((_, reject) => {
              const t = setTimeout(
                () => reject(new AcpRequestStallError("initialize", opts?.timeoutMs ?? 40)),
                opts?.timeoutMs ?? 40,
              );
              t.unref?.();
            });
          }
          return { protocolVersion: 1 };
        },
        async newSession() {
          newSessionCalls += 1;
          return { sessionId: "fresh-should-not-happen" };
        },
        async resumeSession(opts) {
          resumedIds.push(opts.sessionId);
          if (mode === "hang_resume") {
            return new Promise((_resolve, reject) => {
              opts.signal?.addEventListener(
                "abort",
                () => {
                  const err = new Error("aborted");
                  err.name = "AbortError";
                  reject(err);
                },
                { once: true },
              );
            });
          }
          return { sessionId: opts.sessionId };
        },
        async setConfigOption() {
          return {};
        },
        cancelSession() {},
        async prompt(_sid, _blocks, opts) {
          opts?.onUpdate?.({
            sessionUpdate: "agent_message_chunk",
            content: { type: "text", text: "resumed-ok" },
          });
          return { stopReason: "end_turn" };
        },
        async dispose() {
          alive = false;
          disposeCount += 1;
        },
      };
    },
  });

  // 1. Stall during initialize surfaces non-retryable AgyFirstResponseTimeoutError and disposes client.
  const branchEntries = [
    {
      type: "custom",
      customType: AGY_STATE_ENTRY_TYPE,
      data: {
        version: 1,
        kind: "conversation",
        sessionId: "pi-setup-1",
        conversationId: "stored-conv-1",
        cwd: "/work",
        turns: 2,
      },
    },
  ];
  const streamSimple = createAntigravityStreamSimple({
    getRuntime: () => runtime,
    getSessionId: () => "pi-setup-1",
    getBranchEntries: () => branchEntries,
    replayStore: new AgyReplayStore(),
  });

  const stalledMsg = await streamSimple(
    testModel(),
    makeTranscript([{ role: "user", content: "hi", timestamp: 0 }]),
  ).result();
  assert.equal(stalledMsg.stopReason, "error");
  assert.match(stalledMsg.errorMessage ?? "", /within .*s — cancelled stalled turn/);
  assert.equal(isRetryableAssistantError(stalledMsg), false);
  assert.equal(disposeCount, 1);

  // 2. Abort during resumeSession disposes client without wiping stored conversationId or calling newSession.
  mode = "hang_resume";

  const abort = new AbortController();
  setTimeout(() => abort.abort(), 20);
  const abortedMsg = await streamSimple(
    testModel(),
    makeTranscript([{ role: "user", content: "hi", timestamp: 1 }]),
    { signal: abort.signal },
  ).result();
  assert.equal(abortedMsg.stopReason, "aborted");
  assert.equal(newSessionCalls, 0);
  assert.equal(disposeCount, 2);
  assert.deepEqual(resumedIds, ["stored-conv-1"]);

  // Subsequent turn still resumes the same stored-conv-1 cleanly.
  mode = "ok";
  const okMsg = await streamSimple(
    testModel(),
    makeTranscript([{ role: "user", content: "hi again", timestamp: 2 }]),
  ).result();
  assert.equal(okMsg.stopReason, "stop");
  assert.equal(newSessionCalls, 0);
  assert.deepEqual(resumedIds, ["stored-conv-1", "stored-conv-1"]);
  await runtime.dispose();
});

test("stopReason max_turn_requests and refusal fail fast without retries while max_tokens maps to length", async () => {
  const stopReasons = ["max_turn_requests", "refusal", "max_tokens"] as const;
  let index = 0;

  const runtime = new AgyRuntime({
    cwd: "/work",
    getToolMode: () => "native",
    checkBinary: async () => ({ binary: "agy", version: "1.5.14" }),
    createClient: () => ({
      isAlive: true,
      async initialize() {
        return { protocolVersion: 1 };
      },
      async newSession() {
        return { sessionId: "stop-reason-sess" };
      },
      async resumeSession() {
        return { sessionId: "stop-reason-sess" };
      },
      async setConfigOption() {
        return {};
      },
      cancelSession() {},
      async prompt(_sid, _blocks, opts) {
        const stopReason = stopReasons[index++] ?? "end_turn";
        opts?.onUpdate?.({
          sessionUpdate: "agent_message_chunk",
          content: { type: "text", text: `partial:${stopReason}` },
        });
        return { stopReason };
      },
      async dispose() {},
    }),
  });

  const streamSimple = createAntigravityStreamSimple({
    getRuntime: () => runtime,
    getSessionId: () => "pi-stop-reasons",
    getBranchEntries: () => [],
    replayStore: new AgyReplayStore(),
  });

  try {
    const msg1 = await streamSimple(
      testModel(),
      makeTranscript([{ role: "user", content: "1", timestamp: 1 }]),
    ).result();
    assert.equal(msg1.stopReason, "error");
    assert.match(msg1.errorMessage ?? "", /max_turn_requests/);
    assert.equal(isRetryableAssistantError(msg1), false);
    assert.deepEqual(msg1.content, [{ type: "text", text: "partial:max_turn_requests" }]);

    const msg2 = await streamSimple(
      testModel(),
      makeTranscript([{ role: "user", content: "2", timestamp: 2 }]),
    ).result();
    assert.equal(msg2.stopReason, "error");
    assert.match(msg2.errorMessage ?? "", /refusal/);
    assert.equal(isRetryableAssistantError(msg2), false);
    assert.deepEqual(msg2.content, [{ type: "text", text: "partial:refusal" }]);

    const msg3 = await streamSimple(
      testModel(),
      makeTranscript([{ role: "user", content: "3", timestamp: 3 }]),
    ).result();
    assert.equal(msg3.stopReason, "length");
    assert.deepEqual(msg3.content, [{ type: "text", text: "partial:max_tokens" }]);
  } finally {
    await runtime.dispose();
  }
});

test("host Hindsight detection injects HINDSIGHT_DISABLE_HOOKS=1 into warmed and active agy acp env", async () => {
  assert.equal(hasHostHindsightTools([{ name: "read" }, { name: "bash" }]), false);
  assert.equal(hasHostHindsightTools([{ name: "read" }, { name: "hindsight_search_pages" }]), true);

  assert.equal(resolveAgyChildEnv({ hostHindsightActive: false }), undefined);
  assert.deepEqual(resolveAgyChildEnv({ hostHindsightActive: false, env: { CUSTOM: "1" } }), {
    CUSTOM: "1",
  });
  assert.deepEqual(resolveAgyChildEnv({ hostHindsightActive: true, ambientEnv: {} }), {
    [HINDSIGHT_DISABLE_HOOKS_ENV]: "1",
  });
  assert.deepEqual(
    resolveAgyChildEnv({
      hostHindsightActive: true,
      env: { [HINDSIGHT_DISABLE_HOOKS_ENV]: "" },
      ambientEnv: {},
    }),
    { [HINDSIGHT_DISABLE_HOOKS_ENV]: "" },
  );
  assert.deepEqual(
    resolveAgyChildEnv({
      hostHindsightActive: true,
      env: { CUSTOM: "1" },
      ambientEnv: { [HINDSIGHT_DISABLE_HOOKS_ENV]: "0" },
    }),
    { CUSTOM: "1" },
  );

  let spawnCount = 0;
  const seenEnvs: Array<NodeJS.ProcessEnv | undefined> = [];
  const runtime = new AgyRuntime({
    cwd: "/work",
    getToolMode: () => "native",
    isHostHindsightActive: () => true,
    checkBinary: async () => ({ binary: "agy", version: "1.5.14" }),
    createClient: (opts) => {
      spawnCount += 1;
      seenEnvs.push(opts.env);
      return {
        isAlive: true,
        async initialize() {
          return { protocolVersion: 1 };
        },
        async newSession() {
          return { sessionId: "hindsight-dedupe-sess" };
        },
        async resumeSession() {
          return { sessionId: "hindsight-dedupe-sess" };
        },
        async setConfigOption() {
          return {};
        },
        cancelSession() {},
        async prompt(_sid, _blocks, opts) {
          opts?.onUpdate?.({
            sessionUpdate: "agent_message_chunk",
            content: { type: "text", text: "ok" },
          });
          return { stopReason: "end_turn" };
        },
        async dispose() {},
      };
    },
  });

  try {
    await runtime.warmupClient();
    const streamSimple = createAntigravityStreamSimple({
      getRuntime: () => runtime,
      getSessionId: () => "pi-hs-1",
      getBranchEntries: () => [],
      replayStore: new AgyReplayStore(),
    });
    const msg = await streamSimple(
      testModel(),
      makeTranscript([{ role: "user", content: "hello", timestamp: 1 }]),
    ).result();
    assert.equal(msg.stopReason, "stop");
    assert.equal(spawnCount, 1);
    assert.deepEqual(seenEnvs, [{ [HINDSIGHT_DISABLE_HOOKS_ENV]: "1" }]);
  } finally {
    await runtime.dispose();
  }
});

test("mid-turn steering in acp:pi flushes executed tool results, cancels superseded prompt, and sends steered prompt", async () => {
  const agentsDir = fs.mkdtempSync(path.join(os.tmpdir(), "pi-agy-steer-pi-"));
  const bridge = new AgyPiBridge("pi-steer-test");
  bridge.setToolSource(() => [
    { name: "bash", description: "run bash", parameters: {} },
    { name: "edit", description: "edit file", parameters: {} },
  ]);
  bridge.refreshTools();
  await bridge.start();

  const events: string[] = [];
  const seenPrompts: Array<{ sessionId: string; text: string }> = [];
  let activePromptCalls = 0;
  let maxConcurrentPrompts = 0;
  let bashResult: { isError?: boolean; text?: string } | undefined;
  let editResult: { isError?: boolean; text?: string } | undefined;
  let cancelResolve: (() => void) | undefined;

  const persistedStates: AgyConversationState[] = [];
  const trackedRuntime = new AgyRuntime({
    cwd: "/work",
    agentsDir,
    bridge,
    bridgeToken: "tok-steer",
    getToolMode: () => "pi",
    ensureBridge: async () => true,
    checkBinary: async () => ({ binary: "agy", version: "1.5.14" }),
    onPersistedState: (s) => persistedStates.push(s),
    createClient: () => ({
      isAlive: true,
      async initialize() {
        return { protocolVersion: 1 };
      },
      async newSession() {
        return { sessionId: "steer-sess-1" };
      },
      async resumeSession() {
        return { sessionId: "steer-sess-1" };
      },
      async setConfigOption() {
        return {};
      },
      cancelSession() {
        events.push("cancelSession");
        cancelResolve?.();
      },
      async prompt(sid, blocks, opts) {
        activePromptCalls += 1;
        maxConcurrentPrompts = Math.max(maxConcurrentPrompts, activePromptCalls);
        const text = blocks.map((b) => (b.type === "text" ? b.text : "")).join("\n");
        seenPrompts.push({ sessionId: sid, text });
        const callNumber = seenPrompts.length;

        try {
          if (callNumber === 1) {
            opts?.signal?.addEventListener(
              "abort",
              () => {
                events.push("cancelSession");
                cancelResolve?.();
              },
              { once: true },
            );
            opts?.onUpdate?.({
              sessionUpdate: "tool_call",
              toolCallId: "tc-bash-1",
              name: "pi__bash",
              title: "pi__bash",
              status: "in_progress",
            });
            const bashPromise = fetch(bridge.url!, {
              method: "POST",
              headers: {
                "content-type": "application/json",
                "x-pi-bridge-token": "tok-steer",
              },
              body: JSON.stringify({
                jsonrpc: "2.0",
                id: 1,
                method: "tools/call",
                params: { name: "pi__bash", arguments: { command: "echo ok" } },
              }),
            }).then(async (res) => {
              const json = (await res.json()) as {
                result: { isError?: boolean; content: Array<{ text?: string }> };
              };
              bashResult = {
                isError: json.result.isError,
                text: json.result.content[0]?.text,
              };
              events.push("bash_http_resolved");
              opts?.onUpdate?.({
                sessionUpdate: "tool_call_update",
                toolCallId: "tc-bash-1",
                status: "completed",
              });
              events.push("bash_acp_ack");
            });

            const editPromise = fetch(bridge.url!, {
              method: "POST",
              headers: {
                "content-type": "application/json",
                "x-pi-bridge-token": "tok-steer",
              },
              body: JSON.stringify({
                jsonrpc: "2.0",
                id: 2,
                method: "tools/call",
                params: { name: "pi__edit", arguments: { path: "a.txt" } },
              }),
            }).then(async (res) => {
              const json = (await res.json()) as {
                result: { isError?: boolean; content: Array<{ text?: string }> };
              };
              editResult = {
                isError: json.result.isError,
                text: json.result.content[0]?.text,
              };
              events.push("edit_http_resolved");
            });

            await Promise.all([
              bashPromise,
              editPromise,
              new Promise<void>((resolve) => {
                cancelResolve = resolve;
              }),
            ]);
            events.push("prompt1_settled");
            return { stopReason: "cancelled" };
          }

          events.push("prompt2_started");
          opts?.onUpdate?.({
            sessionUpdate: "agent_message_chunk",
            content: { type: "text", text: "参数优化建议如下" },
          });
          return {
            stopReason: "end_turn",
            usage: { inputTokens: 50, outputTokens: 20, totalTokens: 70 },
          };
        } finally {
          activePromptCalls -= 1;
        }
      },
      async dispose() {},
    }),
  });

  const streamSimple = createAntigravityStreamSimple({
    getRuntime: () => trackedRuntime,
    getSessionId: () => "01a2019e-22eb-778f-8fc3-8c477c907c01",
    getBranchEntries: () => [],
    replayStore: new AgyReplayStore(),
  });

  try {
    const turn1Messages: unknown[] = [
      {
        role: "system",
        content: "",
        sections: { customNote: "CUSTOM_SYS_NOTE" },
        timestamp: 0,
      },
      { role: "user", content: "取消硬限，改用权重隔离", timestamp: 1 },
    ];
    const wave1 = streamSimple(testModel(), makeTranscript(turn1Messages));
    const msg1 = await wave1.result();
    assert.equal(msg1.stopReason, "toolUse");
    const toolCalls = msg1.content.filter((b) => b.type === "toolCall");
    const bashCall = toolCalls.find((c) => c.name === "bash");
    assert.ok(bashCall);
    while (bridge.pendingCount < 2) {
      await new Promise((r) => setTimeout(r, 5));
    }

    // Only bash was executed before user steered with a new message.
    const steeredMessages: unknown[] = [
      ...turn1Messages,
      msg1,
      {
        role: "toolResult",
        toolCallId: bashCall.id,
        toolName: "bash",
        content: [{ type: "text", text: "检查通过" }],
        isError: false,
        timestamp: 2,
      },
      { role: "user", content: "参数上还有什么值得优化的吗？", timestamp: 3 },
    ];

    const turn2 = streamSimple(testModel(), makeTranscript(steeredMessages));
    const msg2 = await turn2.result();
    assert.equal(msg2.stopReason, "stop");
    assert.deepEqual(msg2.content, [{ type: "text", text: "参数优化建议如下" }]);

    assert.equal(maxConcurrentPrompts, 1);
    assert.deepEqual(bashResult, { isError: false, text: "检查通过" });
    assert.equal(editResult?.isError, true);

    assert.ok(events.indexOf("bash_acp_ack") >= 0);
    assert.ok(events.indexOf("bash_acp_ack") < events.indexOf("cancelSession"));
    assert.ok(events.indexOf("prompt1_settled") < events.indexOf("prompt2_started"));

    assert.equal(seenPrompts.length, 2);
    assert.equal(seenPrompts[0]?.sessionId, "steer-sess-1");
    assert.match(seenPrompts[0]?.text ?? "", /CUSTOM_SYS_NOTE/);
    assert.match(seenPrompts[0]?.text ?? "", /取消硬限，改用权重隔离/);
    assert.equal(seenPrompts[1]?.sessionId, "steer-sess-1");
    assert.equal(seenPrompts[1]?.text, "参数上还有什么值得优化的吗？");
    assert.equal(seenPrompts[1]?.text.includes("CUSTOM_SYS_NOTE"), false);
    assert.equal(persistedStates.length, 2);
    assert.equal(persistedStates[0]?.turns, 1);
    assert.equal(persistedStates[1]?.turns, 2);
  } finally {
    await trackedRuntime.dispose();
    await bridge.close();
    fs.rmSync(agentsDir, { recursive: true, force: true });
  }
});

test("mid-turn steering in acp:agy (native) interrupts in-flight and settled waves and sends steered prompt", async () => {
  for (const mode of ["inflight", "settled"] as const) {
    let promptCount = 0;
    let cancelCount = 0;
    let releasePrompt1: (() => void) | undefined;
    const seenTexts: string[] = [];

    const runtime = new AgyRuntime({
      cwd: "/work",
      getToolMode: () => "native",
      cancelGraceMs: 100,
      checkBinary: async () => ({ binary: "agy", version: "1.5.14" }),
      createClient: () => ({
        isAlive: true,
        async initialize() {
          return { protocolVersion: 1 };
        },
        async newSession() {
          return { sessionId: `native-steer-${mode}` };
        },
        async resumeSession() {
          return { sessionId: `native-steer-${mode}` };
        },
        async setConfigOption() {
          return {};
        },
        cancelSession() {
          cancelCount += 1;
          releasePrompt1?.();
        },
        async prompt(_sid, blocks, opts) {
          promptCount += 1;
          const text = blocks.map((b) => (b.type === "text" ? b.text : "")).join("\n");
          seenTexts.push(text);
          if (promptCount === 1) {
            opts?.onUpdate?.({
              sessionUpdate: "tool_call",
              toolCallId: "n1",
              title: "Run check",
              kind: "execute",
              status: "in_progress",
            });
            opts?.onUpdate?.({
              sessionUpdate: "tool_call_update",
              toolCallId: "n1",
              status: "completed",
              content: [{ type: "content", content: { type: "text", text: "done" } }],
            });
            opts?.onUpdate?.({
              sessionUpdate: "agent_message_chunk",
              content: { type: "text", text: "stale-turn-1-reply" },
            });
            if (mode === "inflight") {
              await new Promise<void>((resolve) => {
                releasePrompt1 = resolve;
                opts?.signal?.addEventListener(
                  "abort",
                  () => {
                    cancelCount += 1;
                    resolve();
                  },
                  { once: true },
                );
              });
              return { stopReason: "cancelled" };
            }
            return { stopReason: "end_turn" };
          }

          opts?.onUpdate?.({
            sessionUpdate: "agent_message_chunk",
            content: { type: "text", text: "fresh-steered-reply" },
          });
          return { stopReason: "end_turn" };
        },
        async dispose() {},
      }),
    });

    const streamSimple = createAntigravityStreamSimple({
      getRuntime: () => runtime,
      getSessionId: () => `pi-native-steer-${mode}`,
      getBranchEntries: () => [],
      replayStore: new AgyReplayStore(),
    });

    try {
      const m0: unknown[] = [{ role: "user", content: "first", timestamp: 1 }];
      const wave1 = streamSimple(testModel(), makeTranscript(m0));
      const msg1 = await wave1.result();
      assert.equal(msg1.stopReason, "toolUse");
      const wrapperCall = msg1.content.find((b) => b.type === "toolCall");
      assert.ok(wrapperCall);

      if (mode === "settled") {
        // Give prompt #1 a microtask to settle before Wave 2 arrives with a steered user message.
        await new Promise((r) => setTimeout(r, 10));
      }

      const steered = [
        ...m0,
        msg1,
        {
          role: "toolResult",
          toolCallId: wrapperCall.id,
          toolName: "antigravity",
          content: [{ type: "text", text: "done" }],
          isError: false,
          timestamp: 2,
        },
        { role: "user", content: "参数上还有什么值得优化的吗？", timestamp: 3 },
      ];
      const wave2 = streamSimple(testModel(), makeTranscript(steered));
      const msg2 = await wave2.result();
      assert.equal(msg2.stopReason, "stop");
      assert.deepEqual(msg2.content, [{ type: "text", text: "fresh-steered-reply" }]);
      assert.equal(promptCount, 2);
      assert.deepEqual(seenTexts, ["first", "参数上还有什么值得优化的吗？"]);
      assert.equal(cancelCount, mode === "inflight" ? 1 : 0);
    } finally {
      await runtime.dispose();
    }
  }
});

test("settleAfterAgentEnd cancels unsettled turn and pending bridge calls, persists state, and unblocks next turn", async () => {
  const agentsDir = fs.mkdtempSync(path.join(os.tmpdir(), "pi-agy-agent-end-"));
  const bridge = new AgyPiBridge("pi-agent-end-test");
  bridge.setToolSource(() => [{ name: "bash", description: "run bash", parameters: {} }]);
  bridge.refreshTools();
  await bridge.start();

  let promptCount = 0;
  let cancelCount = 0;
  let releasePrompt1: (() => void) | undefined;
  let pendingCallResult: { isError?: boolean } | undefined;
  const persisted: AgyConversationState[] = [];

  const runtime = new AgyRuntime({
    cwd: "/work",
    agentsDir,
    bridge,
    bridgeToken: "tok-end",
    getToolMode: () => "pi",
    ensureBridge: async () => true,
    onPersistedState: (s) => persisted.push(s),
    checkBinary: async () => ({ binary: "agy", version: "1.5.14" }),
    createClient: () => ({
      isAlive: true,
      async initialize() {
        return { protocolVersion: 1 };
      },
      async newSession() {
        return { sessionId: "agent-end-sess" };
      },
      async resumeSession() {
        return { sessionId: "agent-end-sess" };
      },
      async setConfigOption() {
        return {};
      },
      cancelSession() {
        cancelCount += 1;
        releasePrompt1?.();
      },
      async prompt(_sid, blocks, opts) {
        promptCount += 1;
        if (promptCount === 1) {
          let aborted = false;
          opts?.signal?.addEventListener(
            "abort",
            () => {
              aborted = true;
              cancelCount += 1;
              releasePrompt1?.();
            },
            { once: true },
          );
          opts?.onUpdate?.({
            sessionUpdate: "agent_thought_chunk",
            content: { type: "text", text: "thinking" },
          });
          const callRes = await fetch(bridge.url!, {
            method: "POST",
            headers: {
              "content-type": "application/json",
              "x-pi-bridge-token": "tok-end",
            },
            body: JSON.stringify({
              jsonrpc: "2.0",
              id: 1,
              method: "tools/call",
              params: { name: "pi__bash", arguments: { command: "echo blocked" } },
            }),
          });
          const json = (await callRes.json()) as { result: { isError?: boolean } };
          pendingCallResult = json.result;
          if (!aborted) {
            await new Promise<void>((resolve) => {
              releasePrompt1 = resolve;
            });
          }
          return { stopReason: "cancelled" };
        }
        const text = blocks.map((b) => (b.type === "text" ? b.text : "")).join("");
        opts?.onUpdate?.({
          sessionUpdate: "agent_message_chunk",
          content: { type: "text", text: `reply:${text}` },
        });
        return { stopReason: "end_turn" };
      },
      async dispose() {},
    }),
  });

  const streamSimple = createAntigravityStreamSimple({
    getRuntime: () => runtime,
    getSessionId: () => "pi-agent-end",
    getBranchEntries: () => [],
    replayStore: new AgyReplayStore(),
  });

  try {
    const wave1 = streamSimple(
      testModel(),
      makeTranscript([{ role: "user", content: "run blocked", timestamp: 1 }]),
    );
    const msg1 = await wave1.result();
    assert.equal(msg1.stopReason, "toolUse");
    assert.equal(bridge.pendingCount, 1);

    // Simulate Pi terminating the agent run early (e.g. beforeToolCall terminate: true) -> agent_end fires.
    await runtime.settleAfterAgentEnd();
    assert.equal(bridge.pendingCount, 0);
    assert.equal(pendingCallResult?.isError, true);
    assert.equal(cancelCount, 1);
    assert.equal(persisted.length, 1);
    assert.equal(persisted[0]?.turns, 1);

    const wave2 = streamSimple(
      testModel(),
      makeTranscript([{ role: "user", content: "next turn", timestamp: 2 }]),
    );
    const msg2 = await wave2.result();
    assert.equal(msg2.stopReason, "stop");
    assert.deepEqual(msg2.content, [{ type: "text", text: "reply:next turn" }]);
  } finally {
    await runtime.dispose();
    await bridge.close();
    fs.rmSync(agentsDir, { recursive: true, force: true });
  }
});

test("bridged pi__read image result strips MCP image payload and continues same ACP session with inline image blocks", async () => {
  const agentsDir = fs.mkdtempSync(path.join(os.tmpdir(), "pi-agy-img-single-"));
  const bridge = new AgyPiBridge("pi-img-single");
  bridge.setToolSource(() => [{ name: "read", description: "read file", parameters: {} }]);
  bridge.refreshTools();
  await bridge.start();

  const events: string[] = [];
  const seenPrompts: Array<{
    sessionId: string;
    blocks: ReadonlyArray<{ type: string; text?: string; data?: string; mimeType?: string }>;
  }> = [];
  let mcpToolContent: Array<{ type: string; text?: string }> | undefined;
  let releasePrompt1: (() => void) | undefined;

  const runtime = new AgyRuntime({
    cwd: "/work",
    agentsDir,
    bridge,
    bridgeToken: "tok-img",
    getToolMode: () => "pi",
    ensureBridge: async () => true,
    checkBinary: async () => ({ binary: "agy", version: "1.5.15" }),
    createClient: () => ({
      isAlive: true,
      async initialize() {
        return { protocolVersion: 1 };
      },
      async newSession() {
        return { sessionId: "img-sess-1" };
      },
      async resumeSession() {
        return { sessionId: "img-sess-1" };
      },
      async setConfigOption() {
        return {};
      },
      cancelSession() {
        events.push("cancelSession");
        releasePrompt1?.();
      },
      async prompt(sid, blocks, opts) {
        seenPrompts.push({ sessionId: sid, blocks });
        if (seenPrompts.length === 1) {
          opts?.signal?.addEventListener(
            "abort",
            () => {
              events.push("cancelSession");
              releasePrompt1?.();
            },
            { once: true },
          );
          opts?.onUpdate?.({
            sessionUpdate: "tool_call",
            toolCallId: "tc-read-1",
            name: "pi__read",
            title: "pi__read",
            status: "in_progress",
          });
          const res = await fetch(bridge.url!, {
            method: "POST",
            headers: {
              "content-type": "application/json",
              "x-pi-bridge-token": "tok-img",
            },
            body: JSON.stringify({
              jsonrpc: "2.0",
              id: 1,
              method: "tools/call",
              params: { name: "pi__read", arguments: { path: "refs/face.jpg" } },
            }),
          });
          const json = (await res.json()) as {
            result: { content: Array<{ type: string; text?: string }> };
          };
          mcpToolContent = json.result.content;
          events.push("mcp_http_resolved");
          opts?.onUpdate?.({
            sessionUpdate: "tool_call_update",
            toolCallId: "tc-read-1",
            status: "completed",
          });
          events.push("acp_tool_ack");
          await new Promise<void>((resolve) => {
            releasePrompt1 = resolve;
          });
          events.push("prompt1_settled");
          return { stopReason: "cancelled" };
        }

        events.push("prompt2_started");
        opts?.onUpdate?.({
          sessionUpdate: "agent_message_chunk",
          content: { type: "text", text: "saw-image-ok" },
        });
        return {
          stopReason: "end_turn",
          usage: { inputTokens: 100, outputTokens: 10, totalTokens: 110 },
        };
      },
      async dispose() {},
    }),
  });

  const streamSimple = createAntigravityStreamSimple({
    getRuntime: () => runtime,
    getSessionId: () => "pi-img-1",
    getBranchEntries: () => [],
    replayStore: new AgyReplayStore(),
  });

  try {
    let messages: unknown[] = [
      {
        role: "system",
        content: "",
        sections: { customNote: "CUSTOM_SYS_ONCE" },
        timestamp: 0,
      },
      { role: "user", content: "inspect refs/face.jpg", timestamp: 1 },
    ];
    const wave1 = streamSimple(testModel(), makeTranscript(messages));
    const events1 = [];
    for await (const ev of wave1) events1.push(ev);
    assertDeltasMatchPartial(events1);
    const msg1 = await wave1.result();
    assert.equal(msg1.stopReason, "toolUse");
    const toolCall = msg1.content.find((b) => b.type === "toolCall");
    assert.ok(toolCall && toolCall.type === "toolCall");

    messages = [
      ...messages,
      msg1,
      {
        role: "toolResult",
        toolCallId: toolCall.id,
        toolName: "read",
        content: [
          { type: "text", text: "Read image file [image/jpeg]" },
          { type: "image", data: "aW1nMQ==", mimeType: "image/jpeg" },
        ],
        isError: false,
        timestamp: 2,
      },
    ];

    const wave2 = streamSimple(testModel(), makeTranscript(messages));
    const events2 = [];
    for await (const ev of wave2) events2.push(ev);
    assertDeltasMatchPartial(events2);
    const msg2 = await wave2.result();
    assert.equal(msg2.stopReason, "stop");
    assert.deepEqual(msg2.content, [{ type: "text", text: "saw-image-ok" }]);

    assert.ok(mcpToolContent);
    assert.equal(
      mcpToolContent.some((b) => b.type === "image"),
      false,
    );
    assert.match(
      mcpToolContent[1]?.text ?? "",
      /delivered inline in the immediate continuation message/,
    );

    assert.ok(events.indexOf("acp_tool_ack") >= 0);
    assert.ok(events.indexOf("acp_tool_ack") < events.indexOf("cancelSession"));
    assert.ok(events.indexOf("prompt1_settled") < events.indexOf("prompt2_started"));

    assert.equal(seenPrompts.length, 2);
    assert.equal(seenPrompts[0]?.sessionId, "img-sess-1");
    assert.equal(seenPrompts[1]?.sessionId, "img-sess-1");
    assert.match(seenPrompts[0]?.blocks[0]?.text ?? "", /CUSTOM_SYS_ONCE/);
    assert.equal((seenPrompts[1]?.blocks[0]?.text ?? "").includes("CUSTOM_SYS_ONCE"), false);
    assert.match(
      seenPrompts[1]?.blocks[0]?.text ?? "",
      /1\. read \("refs\/face\.jpg"\) — image\/jpeg/,
    );
    assert.deepEqual(seenPrompts[1]?.blocks[1], {
      type: "image",
      data: "aW1nMQ==",
      mimeType: "image/jpeg",
    });
  } finally {
    await runtime.dispose();
    await bridge.close();
    fs.rmSync(agentsDir, { recursive: true, force: true });
  }
});

test("parallel bridged calls (pi__read image + pi__bash text) wait for sibling completion before image handoff", async () => {
  const agentsDir = fs.mkdtempSync(path.join(os.tmpdir(), "pi-agy-img-parallel-"));
  const bridge = new AgyPiBridge("pi-img-parallel");
  bridge.setToolSource(() => [
    { name: "read", description: "read file", parameters: {} },
    { name: "bash", description: "run bash", parameters: {} },
  ]);
  bridge.refreshTools();
  await bridge.start();

  const events: string[] = [];
  const seenPrompts: Array<{
    blocks: ReadonlyArray<{ type: string; text?: string; data?: string; mimeType?: string }>;
  }> = [];
  let releasePrompt1: (() => void) | undefined;

  const runtime = new AgyRuntime({
    cwd: "/work",
    agentsDir,
    bridge,
    bridgeToken: "tok-par",
    getToolMode: () => "pi",
    ensureBridge: async () => true,
    checkBinary: async () => ({ binary: "agy", version: "1.5.15" }),
    createClient: () => ({
      isAlive: true,
      async initialize() {
        return { protocolVersion: 1 };
      },
      async newSession() {
        return { sessionId: "par-sess-1" };
      },
      async resumeSession() {
        return { sessionId: "par-sess-1" };
      },
      async setConfigOption() {
        return {};
      },
      cancelSession() {
        events.push("cancelSession");
        releasePrompt1?.();
      },
      async prompt(_sid, blocks, opts) {
        seenPrompts.push({ blocks });
        if (seenPrompts.length === 1) {
          opts?.signal?.addEventListener(
            "abort",
            () => {
              events.push("cancelSession");
              releasePrompt1?.();
            },
            { once: true },
          );
          opts?.onUpdate?.({
            sessionUpdate: "tool_call",
            toolCallId: "tc-read-par",
            name: "pi__read",
            title: "pi__read",
            status: "in_progress",
          });
          const readReq = fetch(bridge.url!, {
            method: "POST",
            headers: {
              "content-type": "application/json",
              "x-pi-bridge-token": "tok-par",
            },
            body: JSON.stringify({
              jsonrpc: "2.0",
              id: 1,
              method: "tools/call",
              params: { name: "pi__read", arguments: { path: "grid.png" } },
            }),
          }).then(async (res) => {
            await res.json();
            events.push("read_http_resolved");
            opts?.onUpdate?.({
              sessionUpdate: "tool_call_update",
              toolCallId: "tc-read-par",
              status: "completed",
            });
          });

          // Simulate sibling tool call arriving slightly after wave 1 starts.
          const bashReq = (async () => {
            await new Promise((r) => setTimeout(r, 15));
            opts?.onUpdate?.({
              sessionUpdate: "tool_call",
              toolCallId: "tc-bash-par",
              name: "pi__bash",
              title: "pi__bash",
              status: "in_progress",
            });
            const res = await fetch(bridge.url!, {
              method: "POST",
              headers: {
                "content-type": "application/json",
                "x-pi-bridge-token": "tok-par",
              },
              body: JSON.stringify({
                jsonrpc: "2.0",
                id: 2,
                method: "tools/call",
                params: { name: "pi__bash", arguments: { command: "ls" } },
              }),
            });
            await res.json();
            events.push("bash_http_resolved");
            opts?.onUpdate?.({
              sessionUpdate: "tool_call_update",
              toolCallId: "tc-bash-par",
              status: "completed",
            });
            events.push("bash_acp_ack");
          })();

          await Promise.all([
            readReq,
            bashReq,
            new Promise<void>((resolve) => {
              releasePrompt1 = resolve;
            }),
          ]);
          return { stopReason: "cancelled" };
        }

        opts?.onUpdate?.({
          sessionUpdate: "agent_message_chunk",
          content: { type: "text", text: "parallel-done" },
        });
        return { stopReason: "end_turn" };
      },
      async dispose() {},
    }),
  });

  const streamSimple = createAntigravityStreamSimple({
    getRuntime: () => runtime,
    getSessionId: () => "pi-par-1",
    getBranchEntries: () => [],
    replayStore: new AgyReplayStore(),
  });

  try {
    let messages: unknown[] = [{ role: "user", content: "check grid and ls", timestamp: 1 }];
    const wave1 = streamSimple(testModel(), makeTranscript(messages));
    const msg1 = await wave1.result();
    assert.equal(msg1.stopReason, "toolUse");
    const readCall = msg1.content.find((b) => b.type === "toolCall" && b.name === "read");
    assert.ok(readCall && readCall.type === "toolCall");

    // Wait until the sibling parallel `pi__bash` call has also arrived at the bridge.
    while (bridge.pendingCount < 2) {
      await new Promise((r) => setTimeout(r, 5));
    }

    // Wave 2 resolves only `read` (image) while `bash` is still pending on the bridge.
    messages = [
      ...messages,
      msg1,
      {
        role: "toolResult",
        toolCallId: readCall.id,
        toolName: "read",
        content: [
          { type: "text", text: "Read image file [image/png]" },
          { type: "image", data: "Z3JpZA==", mimeType: "image/png" },
        ],
        isError: false,
        timestamp: 2,
      },
    ];

    const wave2 = streamSimple(testModel(), makeTranscript(messages));
    const msg2 = await wave2.result();
    assert.equal(msg2.stopReason, "toolUse");
    const bashCall = msg2.content.find((b) => b.type === "toolCall" && b.name === "bash");
    assert.ok(bashCall && bashCall.type === "toolCall");
    assert.equal(seenPrompts.length, 1);

    // Wave 3 resolves `bash` -> now all pending bridge calls are done, triggering image handoff.
    messages = [
      ...messages,
      msg2,
      {
        role: "toolResult",
        toolCallId: bashCall.id,
        toolName: "bash",
        content: [{ type: "text", text: "file1.txt" }],
        isError: false,
        timestamp: 3,
      },
    ];

    const wave3 = streamSimple(testModel(), makeTranscript(messages));
    const msg3 = await wave3.result();
    assert.equal(msg3.stopReason, "stop");
    assert.deepEqual(msg3.content, [{ type: "text", text: "parallel-done" }]);
    assert.equal(seenPrompts.length, 2);
    assert.ok(events.indexOf("bash_acp_ack") >= 0);
    assert.ok(events.indexOf("bash_acp_ack") < events.indexOf("cancelSession"));
    assert.deepEqual(seenPrompts[1]?.blocks[1], {
      type: "image",
      data: "Z3JpZA==",
      mimeType: "image/png",
    });
  } finally {
    await runtime.dispose();
    await bridge.close();
    fs.rmSync(agentsDir, { recursive: true, force: true });
  }
});

test("text-only model omits bridged tool images over MCP without cancelling the active ACP turn", async () => {
  const agentsDir = fs.mkdtempSync(path.join(os.tmpdir(), "pi-agy-img-textonly-"));
  const bridge = new AgyPiBridge("pi-img-textonly");
  bridge.setToolSource(() => [{ name: "read", description: "read file", parameters: {} }]);
  bridge.refreshTools();
  await bridge.start();

  let promptCalls = 0;
  let cancelCalls = 0;
  let mcpToolContent: Array<{ type: string; text?: string }> | undefined;

  const runtime = new AgyRuntime({
    cwd: "/work",
    agentsDir,
    bridge,
    bridgeToken: "tok-txt",
    getToolMode: () => "pi",
    ensureBridge: async () => true,
    checkBinary: async () => ({ binary: "agy", version: "1.5.15" }),
    createClient: () => ({
      isAlive: true,
      async initialize() {
        return { protocolVersion: 1 };
      },
      async newSession() {
        return { sessionId: "txt-sess-1" };
      },
      async resumeSession() {
        return { sessionId: "txt-sess-1" };
      },
      async setConfigOption() {
        return {};
      },
      cancelSession() {
        cancelCalls += 1;
      },
      async prompt(_sid, _blocks, opts) {
        promptCalls += 1;
        const res = await fetch(bridge.url!, {
          method: "POST",
          headers: {
            "content-type": "application/json",
            "x-pi-bridge-token": "tok-txt",
          },
          body: JSON.stringify({
            jsonrpc: "2.0",
            id: 1,
            method: "tools/call",
            params: { name: "pi__read", arguments: { path: "pic.png" } },
          }),
        });
        const json = (await res.json()) as {
          result: { content: Array<{ type: string; text?: string }> };
        };
        mcpToolContent = json.result.content;
        opts?.onUpdate?.({
          sessionUpdate: "agent_message_chunk",
          content: { type: "text", text: "text-only-ok" },
        });
        return { stopReason: "end_turn" };
      },
      async dispose() {},
    }),
  });

  const textOnlyModel: Model<Api> = { ...testModel(), input: ["text"] };
  const streamSimple = createAntigravityStreamSimple({
    getRuntime: () => runtime,
    getSessionId: () => "pi-txt-1",
    getBranchEntries: () => [],
    replayStore: new AgyReplayStore(),
  });

  try {
    let messages: unknown[] = [{ role: "user", content: "read pic.png", timestamp: 1 }];
    const wave1 = streamSimple(textOnlyModel, makeTranscript(messages));
    const msg1 = await wave1.result();
    assert.equal(msg1.stopReason, "toolUse");
    const toolCall = msg1.content.find((b) => b.type === "toolCall");
    assert.ok(toolCall && toolCall.type === "toolCall");

    messages = [
      ...messages,
      msg1,
      {
        role: "toolResult",
        toolCallId: toolCall.id,
        toolName: "read",
        content: [
          { type: "text", text: "Read image file [image/png]" },
          { type: "image", data: "cGlj", mimeType: "image/png" },
        ],
        isError: false,
        timestamp: 2,
      },
    ];

    const wave2 = streamSimple(textOnlyModel, makeTranscript(messages));
    const msg2 = await wave2.result();
    assert.equal(msg2.stopReason, "stop");
    assert.deepEqual(msg2.content, [{ type: "text", text: "text-only-ok" }]);
    assert.equal(promptCalls, 1);
    assert.equal(cancelCalls, 0);
    assert.deepEqual(mcpToolContent, [
      { type: "text", text: "Read image file [image/png]" },
      { type: "text", text: "(1 image(s) omitted — this agy request is text-only)" },
    ]);
  } finally {
    await runtime.dispose();
    await bridge.close();
    fs.rmSync(agentsDir, { recursive: true, force: true });
  }
});

test("compaction isolates summarization on ephemeral clients and resets main ACP session with seeded history", async () => {
  const agentsDir = fs.mkdtempSync(path.join(os.tmpdir(), "pi-agy-compact-agents-"));
  const persisted: AgyConversationState[] = [];
  const mainSessionPrompts: Array<{ sessionId: string; text: string }> = [];
  const ephemeralClients: Array<{
    agentName?: string;
    mcpServersCount: number;
    closed: boolean;
    disposed: boolean;
    promptText: string;
  }> = [];
  const closedMainSessions: string[] = [];
  let mainSeq = 0;

  const runtime = new AgyRuntime({
    cwd: "/work",
    agentsDir,
    getToolMode: () => "native",
    checkBinary: async () => ({ binary: "agy", version: "1.5.14" }),
    onPersistedState: (s) => persisted.push(s),
    createClient: (opts) => {
      const isEphemeral = (opts.agentName ?? "").startsWith("pi-ephemeral-");
      if (isEphemeral) {
        const record = {
          agentName: opts.agentName,
          mcpServersCount: -1,
          closed: false,
          disposed: false,
          promptText: "",
        };
        ephemeralClients.push(record);
        let alive = true;
        return {
          get isAlive() {
            return alive;
          },
          async initialize() {
            return { protocolVersion: 1 };
          },
          async newSession(newOpts) {
            record.mcpServersCount = newOpts.mcpServers?.length ?? 0;
            return { sessionId: `eph-${ephemeralClients.length}` };
          },
          async resumeSession() {
            throw new Error("ephemeral client must not resume");
          },
          async setConfigOption() {
            return {};
          },
          cancelSession() {},
          async closeSession() {
            record.closed = true;
          },
          async prompt(_sid, blocks, promptOpts) {
            record.promptText = blocks.map((b) => (b.type === "text" ? b.text : "")).join("\n");
            promptOpts?.onUpdate?.({
              sessionUpdate: "agent_message_chunk",
              content: { type: "text", text: "## Goal\nCompacted summary" },
            });
            return {
              stopReason: "end_turn",
              usage: { inputTokens: 1200, outputTokens: 200, totalTokens: 1400 },
            };
          },
          async dispose() {
            alive = false;
            record.disposed = true;
          },
        };
      }

      let alive = true;
      return {
        get isAlive() {
          return alive;
        },
        async initialize() {
          return { protocolVersion: 1 };
        },
        async newSession() {
          mainSeq += 1;
          return { sessionId: `acp-main-${mainSeq}` };
        },
        async resumeSession(rOpts) {
          return { sessionId: rOpts.sessionId };
        },
        async setConfigOption() {
          return {};
        },
        cancelSession() {},
        async closeSession(sid) {
          closedMainSessions.push(sid);
        },
        async prompt(sid, blocks, promptOpts) {
          const text = blocks.map((b) => (b.type === "text" ? b.text : "")).join("\n");
          mainSessionPrompts.push({ sessionId: sid, text });
          const used = sid === "acp-main-1" ? 415_160 : 24_800;
          promptOpts?.onUpdate?.({
            sessionUpdate: "usage_update",
            used,
            size: 1_048_576,
          });
          promptOpts?.onUpdate?.({
            sessionUpdate: "agent_message_chunk",
            content: { type: "text", text: `reply from ${sid}` },
          });
          return {
            stopReason: "end_turn",
            usage: { inputTokens: used - 100, outputTokens: 100, totalTokens: used },
          };
        },
        async dispose() {
          alive = false;
        },
      };
    },
  });

  const branchEntries: Array<Record<string, unknown>> = [];
  const streamSimple = createAntigravityStreamSimple({
    getRuntime: () => runtime,
    getSessionId: () => "pi-01a2069c",
    getBranchEntries: () => branchEntries as never,
    getAllEntries: () => branchEntries as never,
    requireMatchingSessionId: () => true,
    replayStore: new AgyReplayStore(),
  });

  try {
    // Turn 1 on main session acp-main-1 -> 415_160 tokens.
    const t1 = await streamSimple(
      testModel(),
      makeTranscript([{ role: "user", content: "initial task", timestamp: 1 }]),
      { sessionId: "pi-01a2069c" },
    ).result();
    assert.equal(t1.usage.totalTokens, 415_160);
    assert.equal(persisted.length, 1);
    assert.equal(persisted[0]?.conversationId, "acp-main-1");
    branchEntries.push({
      type: "custom",
      customType: AGY_STATE_ENTRY_TYPE,
      data: persisted[0],
    });

    // Compaction summarization calls (cacheRetention: "none", fresh sessionId) must run isolated.
    const c1 = await streamSimple(
      testModel(),
      makeTranscript([{ role: "user", content: "summarize history", timestamp: 2 }]),
      { cacheRetention: "none", sessionId: "compact-1" },
    ).result();
    const c2 = await streamSimple(
      testModel(),
      makeTranscript([{ role: "user", content: "summarize prefix", timestamp: 3 }]),
      { cacheRetention: "none", sessionId: "compact-2" },
    ).result();
    assert.equal(c1.stopReason, "stop");
    assert.equal(c2.stopReason, "stop");
    assert.equal(ephemeralClients.length, 2);
    assert.deepEqual(
      ephemeralClients.map((c) => ({
        mcp: c.mcpServersCount,
        closed: c.closed,
        disposed: c.disposed,
      })),
      [
        { mcp: 0, closed: true, disposed: true },
        { mcp: 0, closed: true, disposed: true },
      ],
    );
    // Main session acp-main-1 was never polluted and persisted state stayed at 1.
    assert.equal(mainSessionPrompts.length, 1);
    assert.equal(persisted.length, 1);

    // Append compaction entry (without calling session_compact hook to verify runtime branch check).
    branchEntries.push({ type: "compaction", summary: "Compacted summary" });

    // Warmup + Turn 2 after compaction must close acp-main-1, open acp-main-2, seed summary+kept, and report 24_800!
    await runtime.warmupSession({
      modelId: "gemini-4-hydrogen",
      sessionId: "pi-01a2069c",
      branchEntries: branchEntries as never,
      allEntries: branchEntries as never,
    });
    assert.deepEqual(closedMainSessions, ["acp-main-1"]);

    const t2 = await streamSimple(
      testModel(),
      makeTranscript([
        { role: "system", content: "base", timestamp: 0 },
        {
          role: "user",
          content:
            "The conversation history before this point was compacted into the following summary:\n\n<summary>\nGoal: fix video pipeline\n</summary>",
          timestamp: 1,
        },
        {
          role: "assistant",
          content: [{ type: "text", text: "Kept reply before compact" }],
          api: "antigravity-acp",
          provider: "antigravity",
          model: "gemini-4-hydrogen",
          usage: {} as never,
          stopReason: "stop",
          timestamp: 2,
        },
        { role: "user", content: "continue after compact", timestamp: 3 },
      ]),
      { sessionId: "pi-01a2069c" },
    ).result();

    assert.equal(t2.usage.totalTokens, 24_800);
    assert.equal(mainSessionPrompts.length, 2);
    assert.equal(mainSessionPrompts[1]?.sessionId, "acp-main-2");
    assert.match(mainSessionPrompts[1]?.text ?? "", /<conversation_history>/);
    assert.match(mainSessionPrompts[1]?.text ?? "", /Goal: fix video pipeline/);
    assert.match(mainSessionPrompts[1]?.text ?? "", /Kept reply before compact/);
    assert.match(mainSessionPrompts[1]?.text ?? "", /continue after compact$/);
    assert.equal(persisted[1]?.conversationId, "acp-main-2");
    assert.equal(persisted[1]?.turns, 1);
  } finally {
    await runtime.dispose();
    fs.rmSync(agentsDir, { recursive: true, force: true });
  }
});

test("side-thread isolation, /agy-reset skip-seed, and /tree rewind invalidation behave cleanly", async () => {
  const agentsDir = fs.mkdtempSync(path.join(os.tmpdir(), "pi-agy-iso-agents-"));
  const mainPrompts: Array<{ sessionId: string; text: string }> = [];
  const ephemeralPrompts: string[] = [];
  const persisted: AgyConversationState[] = [];
  let mainSeq = 0;

  const runtime = new AgyRuntime({
    cwd: "/work",
    agentsDir,
    getToolMode: () => "native",
    checkBinary: async () => ({ binary: "agy", version: "1.5.14" }),
    onPersistedState: (s) => persisted.push(s),
    createClient: (opts) => {
      const isEphemeral = (opts.agentName ?? "").startsWith("pi-ephemeral-");
      let alive = true;
      return {
        get isAlive() {
          return alive;
        },
        async initialize() {
          return { protocolVersion: 1 };
        },
        async newSession() {
          if (isEphemeral) return { sessionId: `eph-${ephemeralPrompts.length + 1}` };
          mainSeq += 1;
          return { sessionId: `main-${mainSeq}` };
        },
        async resumeSession(rOpts) {
          return { sessionId: rOpts.sessionId };
        },
        async setConfigOption() {
          return {};
        },
        cancelSession() {},
        async closeSession() {},
        async prompt(sid, blocks, promptOpts) {
          const text = blocks.map((b) => (b.type === "text" ? b.text : "")).join("\n");
          if (isEphemeral) {
            ephemeralPrompts.push(text);
          } else {
            mainPrompts.push({ sessionId: sid, text });
          }
          promptOpts?.onUpdate?.({
            sessionUpdate: "agent_message_chunk",
            content: { type: "text", text: "ok" },
          });
          return { stopReason: "end_turn", usage: { totalTokens: 100 } };
        },
        async dispose() {
          alive = false;
        },
      };
    },
  });

  let branchEntries: Array<Record<string, unknown>> = [];
  let allEntries: Array<Record<string, unknown>> = [];
  const streamSimple = createAntigravityStreamSimple({
    getRuntime: () => runtime,
    getSessionId: () => "pi-main",
    getBranchEntries: () => branchEntries as never,
    getAllEntries: () => allEntries as never,
    requireMatchingSessionId: () => true,
    replayStore: new AgyReplayStore(),
  });

  try {
    // 1. Side threads (different sessionId, disableContinuation, or missing sessionId) route ephemerally.
    await streamSimple(
      testModel(),
      makeTranscript([{ role: "user", content: "btw question", timestamp: 1 }]),
      { sessionId: "pi-btw-1" },
    ).result();
    await streamSimple(
      testModel(),
      makeTranscript([{ role: "user", content: "cache warm", timestamp: 2 }]),
      { sessionId: "pi-main", disableContinuation: true },
    ).result();
    await streamSimple(
      testModel(),
      makeTranscript([{ role: "user", content: "missing sessionId", timestamp: 3 }]),
    ).result();
    assert.equal(ephemeralPrompts.length, 3);
    assert.equal(mainPrompts.length, 0);

    // 2. /agy-reset starts a fresh main session WITHOUT seeding prior history.
    branchEntries = [
      {
        type: "custom",
        customType: AGY_STATE_ENTRY_TYPE,
        data: {
          version: 1,
          kind: "reset",
          sessionId: "pi-main",
          cwd: "/work",
        },
      },
    ];
    allEntries = [...branchEntries];
    await streamSimple(
      testModel(),
      makeTranscript([
        { role: "user", content: "old turn", timestamp: 1 },
        {
          role: "assistant",
          content: [{ type: "text", text: "old reply" }],
          api: "antigravity-acp",
          provider: "antigravity",
          model: "gemini-4-hydrogen",
          usage: {} as never,
          stopReason: "stop",
          timestamp: 2,
        },
        { role: "user", content: "fresh after reset", timestamp: 3 },
      ]),
      { sessionId: "pi-main" },
    ).result();
    assert.equal(mainPrompts.length, 1);
    assert.equal(mainPrompts[0]?.text, "fresh after reset");

    // 3. /tree rewind: branch has turns=1 for main-1 while allEntries already advanced main-1 to turns=3.
    const markerT1 = {
      type: "custom",
      customType: AGY_STATE_ENTRY_TYPE,
      data: {
        version: 1,
        kind: "conversation",
        sessionId: "pi-main",
        conversationId: "main-1",
        cwd: "/work",
        turns: 1,
      },
    };
    const markerT3 = {
      type: "custom",
      customType: AGY_STATE_ENTRY_TYPE,
      data: {
        version: 1,
        kind: "conversation",
        sessionId: "pi-main",
        conversationId: "main-1",
        cwd: "/work",
        turns: 3,
      },
    };
    branchEntries = [markerT1];
    allEntries = [markerT1, markerT3];
    await streamSimple(
      testModel(),
      makeTranscript([
        { role: "user", content: "kept before rewind", timestamp: 1 },
        {
          role: "assistant",
          content: [{ type: "text", text: "reply at turn 1" }],
          api: "antigravity-acp",
          provider: "antigravity",
          model: "gemini-4-hydrogen",
          usage: {} as never,
          stopReason: "stop",
          timestamp: 2,
        },
        { role: "user", content: "new branch question", timestamp: 3 },
      ]),
      { sessionId: "pi-main" },
    ).result();
    assert.equal(mainPrompts.length, 2);
    assert.equal(mainPrompts[1]?.sessionId, "main-2");
    assert.match(mainPrompts[1]?.text ?? "", /reply at turn 1/);
    assert.match(mainPrompts[1]?.text ?? "", /new branch question$/);
  } finally {
    await runtime.dispose();
    fs.rmSync(agentsDir, { recursive: true, force: true });
  }
});

test("bridged image handoff on Turn 1 preserves ACP session when branchEntries already include live assistant/toolResult", async () => {
  const agentsDir = fs.mkdtempSync(path.join(os.tmpdir(), "pi-agy-img-livebranch-"));
  const bridge = new AgyPiBridge("pi-img-livebranch");
  bridge.setToolSource(() => [{ name: "read", description: "read file", parameters: {} }]);
  bridge.refreshTools();
  await bridge.start();

  const seenPrompts: Array<{
    sessionId: string;
    blocks: ReadonlyArray<{ type: string; text?: string; data?: string; mimeType?: string }>;
  }> = [];
  let newSessionCalls = 0;
  let closeSessionCalls = 0;
  let releasePrompt1: (() => void) | undefined;

  const runtime = new AgyRuntime({
    cwd: "/work",
    agentsDir,
    bridge,
    bridgeToken: "tok-live-img",
    getToolMode: () => "pi",
    ensureBridge: async () => true,
    checkBinary: async () => ({ binary: "agy", version: "1.5.15" }),
    createClient: () => ({
      isAlive: true,
      async initialize() {
        return { protocolVersion: 1 };
      },
      async newSession() {
        newSessionCalls += 1;
        return { sessionId: `live-img-sess-${newSessionCalls}` };
      },
      async resumeSession() {
        throw new Error("should reuse live session without resume");
      },
      async setConfigOption() {
        return {};
      },
      cancelSession() {
        releasePrompt1?.();
      },
      async closeSession() {
        closeSessionCalls += 1;
      },
      async prompt(sid, blocks, opts) {
        seenPrompts.push({ sessionId: sid, blocks });
        if (seenPrompts.length === 1) {
          opts?.signal?.addEventListener(
            "abort",
            () => {
              releasePrompt1?.();
            },
            { once: true },
          );
          opts?.onUpdate?.({
            sessionUpdate: "tool_call",
            toolCallId: "tc-img-live",
            name: "pi__read",
            title: "pi__read",
            status: "in_progress",
          });
          const res = await fetch(bridge.url!, {
            method: "POST",
            headers: {
              "content-type": "application/json",
              "x-pi-bridge-token": "tok-live-img",
            },
            body: JSON.stringify({
              jsonrpc: "2.0",
              id: 1,
              method: "tools/call",
              params: { name: "pi__read", arguments: { path: "01-new-app-dialog.png" } },
            }),
          });
          await res.json();
          opts?.onUpdate?.({
            sessionUpdate: "tool_call_update",
            toolCallId: "tc-img-live",
            status: "completed",
          });
          await new Promise<void>((resolve) => {
            releasePrompt1 = resolve;
          });
          return { stopReason: "cancelled" };
        }
        opts?.onUpdate?.({
          sessionUpdate: "agent_message_chunk",
          content: { type: "text", text: "kept-same-session" },
        });
        return { stopReason: "end_turn", usage: { totalTokens: 200 } };
      },
      async dispose() {},
    }),
  });

  // Simulate realistic Pi SessionManager.getBranch() during Turn 1 before agent_end persists antigravity-state.
  const branchEntries: Array<Record<string, unknown>> = [
    { type: "session" },
    { type: "model_change" },
    { type: "message", message: { role: "user" } },
  ];
  const streamSimple = createAntigravityStreamSimple({
    getRuntime: () => runtime,
    getSessionId: () => "pi-324c5bde",
    getBranchEntries: () => branchEntries as never,
    getAllEntries: () => branchEntries as never,
    replayStore: new AgyReplayStore(),
  });

  try {
    const initialMessages: unknown[] = [
      {
        role: "system",
        content: "",
        sections: { project_context: "CUSTOM_SYS_TURN1" },
        timestamp: 0,
      },
      { role: "user", content: "Implement managed PG and verify screenshots", timestamp: 1 },
    ];
    const msg1 = await streamSimple(testModel(), makeTranscript(initialMessages)).result();
    assert.equal(msg1.stopReason, "toolUse");
    const toolCall = msg1.content.find((b) => b.type === "toolCall");
    assert.ok(toolCall && toolCall.type === "toolCall");

    const toolResultMsg = {
      role: "toolResult",
      toolCallId: toolCall.id,
      toolName: "read",
      content: [
        { type: "text", text: "Read image file [image/png]" },
        { type: "image", data: "cG5n", mimeType: "image/png" },
      ],
      isError: false,
      timestamp: 2,
    };
    branchEntries.push(
      { type: "message", message: msg1 },
      { type: "message", message: toolResultMsg },
    );

    const msg2 = await streamSimple(
      testModel(),
      makeTranscript([...initialMessages, msg1, toolResultMsg]),
    ).result();
    assert.equal(msg2.stopReason, "stop");
    assert.deepEqual(msg2.content, [{ type: "text", text: "kept-same-session" }]);
    assert.equal(newSessionCalls, 1);
    assert.equal(closeSessionCalls, 0);
    assert.equal(seenPrompts.length, 2);
    assert.equal(seenPrompts[0]?.sessionId, "live-img-sess-1");
    assert.equal(seenPrompts[1]?.sessionId, "live-img-sess-1");
    const prompt2Text = seenPrompts[1]?.blocks[0]?.text ?? "";
    assert.equal(prompt2Text.includes("CUSTOM_SYS_TURN1"), false);
    assert.equal(prompt2Text.includes("<conversation_history>"), false);
    assert.match(prompt2Text, /01-new-app-dialog\.png/);
    assert.deepEqual(seenPrompts[1]?.blocks[1], {
      type: "image",
      data: "cG5n",
      mimeType: "image/png",
    });
  } finally {
    await runtime.dispose();
    await bridge.close();
    fs.rmSync(agentsDir, { recursive: true, force: true });
  }
});

test("mid-turn steering after subagent-notify keeps existing Turn N ACP session intact without re-seeding", async () => {
  const agentsDir = fs.mkdtempSync(path.join(os.tmpdir(), "pi-agy-steer-livebranch-"));
  const bridge = new AgyPiBridge("pi-steer-livebranch");
  bridge.setToolSource(() => [{ name: "bash", description: "run bash", parameters: {} }]);
  bridge.refreshTools();
  await bridge.start();

  const seenPrompts: Array<{ sessionId: string; text: string }> = [];
  let newSessionCalls = 0;
  let resumeCalls = 0;
  let closeSessionCalls = 0;
  let releasePrompt1: (() => void) | undefined;

  const runtime = new AgyRuntime({
    cwd: "/work",
    agentsDir,
    bridge,
    bridgeToken: "tok-steer-live",
    getToolMode: () => "pi",
    ensureBridge: async () => true,
    checkBinary: async () => ({ binary: "agy", version: "1.5.15" }),
    createClient: () => ({
      isAlive: true,
      async initialize() {
        return { protocolVersion: 1 };
      },
      async newSession() {
        newSessionCalls += 1;
        return { sessionId: `unexpected-new-${newSessionCalls}` };
      },
      async resumeSession(opts) {
        resumeCalls += 1;
        return { sessionId: opts.sessionId };
      },
      async setConfigOption() {
        return {};
      },
      cancelSession() {
        releasePrompt1?.();
      },
      async closeSession() {
        closeSessionCalls += 1;
      },
      async prompt(sid, blocks, opts) {
        const text = blocks.map((b) => (b.type === "text" ? b.text : "")).join("\n");
        seenPrompts.push({ sessionId: sid, text });
        if (seenPrompts.length === 1) {
          opts?.signal?.addEventListener(
            "abort",
            () => {
              releasePrompt1?.();
            },
            { once: true },
          );
          opts?.onUpdate?.({
            sessionUpdate: "tool_call",
            toolCallId: "tc-bash-live",
            name: "pi__bash",
            title: "pi__bash",
            status: "in_progress",
          });
          const res = await fetch(bridge.url!, {
            method: "POST",
            headers: {
              "content-type": "application/json",
              "x-pi-bridge-token": "tok-steer-live",
            },
            body: JSON.stringify({
              jsonrpc: "2.0",
              id: 1,
              method: "tools/call",
              params: { name: "pi__bash", arguments: { command: "git status" } },
            }),
          });
          await res.json();
          opts?.onUpdate?.({
            sessionUpdate: "tool_call_update",
            toolCallId: "tc-bash-live",
            status: "completed",
          });
          await new Promise<void>((resolve) => {
            releasePrompt1 = resolve;
          });
          return { stopReason: "cancelled" };
        }
        opts?.onUpdate?.({
          sessionUpdate: "agent_message_chunk",
          content: { type: "text", text: "steered-same-session" },
        });
        return { stopReason: "end_turn", usage: { totalTokens: 300 } };
      },
      async dispose() {},
    }),
  });

  const branchEntries: Array<Record<string, unknown>> = [
    {
      type: "custom",
      customType: AGY_STATE_ENTRY_TYPE,
      data: {
        version: 1,
        kind: "conversation",
        sessionId: "pi-01a20b48",
        conversationId: "0f683650-live",
        cwd: "/work",
        turns: 4,
      },
    },
    { type: "custom_message", customType: "subagent-notify" },
  ];
  const streamSimple = createAntigravityStreamSimple({
    getRuntime: () => runtime,
    getSessionId: () => "pi-01a20b48",
    getBranchEntries: () => branchEntries as never,
    getAllEntries: () => branchEntries as never,
    replayStore: new AgyReplayStore(),
  });

  try {
    const turnMessages: unknown[] = [
      {
        role: "system",
        content: "",
        sections: { project_context: "CUSTOM_SYS_SHOULD_NOT_REPEAT" },
        timestamp: 0,
      },
      { role: "user", content: "Background task completed: worker 1", timestamp: 1 },
    ];
    const msg1 = await streamSimple(testModel(), makeTranscript(turnMessages)).result();
    assert.equal(msg1.stopReason, "toolUse");
    const bashCall = msg1.content.find((b) => b.type === "toolCall");
    assert.ok(bashCall && bashCall.type === "toolCall");

    const toolResultMsg = {
      role: "toolResult",
      toolCallId: bashCall.id,
      toolName: "bash",
      content: [{ type: "text", text: "clean" }],
      isError: false,
      timestamp: 2,
    };
    const worker2Notice = {
      role: "user",
      content: "Background task completed: worker 2 (324c5bde)",
      timestamp: 3,
    };
    branchEntries.push(
      { type: "message", message: msg1 },
      { type: "message", message: toolResultMsg },
      { type: "custom_message", customType: "subagent-notify" },
    );

    const msg2 = await streamSimple(
      testModel(),
      makeTranscript([...turnMessages, msg1, toolResultMsg, worker2Notice]),
    ).result();
    assert.equal(msg2.stopReason, "stop");
    assert.deepEqual(msg2.content, [{ type: "text", text: "steered-same-session" }]);
    assert.equal(newSessionCalls, 0);
    assert.equal(resumeCalls, 1);
    assert.equal(closeSessionCalls, 0);
    assert.equal(seenPrompts.length, 2);
    assert.equal(seenPrompts[0]?.sessionId, "0f683650-live");
    assert.equal(seenPrompts[1]?.sessionId, "0f683650-live");
    assert.equal(seenPrompts[1]?.text, "Background task completed: worker 2 (324c5bde)");
    assert.equal(seenPrompts[1]?.text.includes("<conversation_history>"), false);
    assert.equal(seenPrompts[1]?.text.includes("CUSTOM_SYS_SHOULD_NOT_REPEAT"), false);
  } finally {
    await runtime.dispose();
    await bridge.close();
    fs.rmSync(agentsDir, { recursive: true, force: true });
  }
});

test("forced new ACP session during image handoff seeds history, customSys, manifest, and inline images", async () => {
  const agentsDir = fs.mkdtempSync(path.join(os.tmpdir(), "pi-agy-img-fallback-"));
  const bridge = new AgyPiBridge("pi-img-fallback");
  bridge.setToolSource(() => [{ name: "read", description: "read file", parameters: {} }]);
  bridge.refreshTools();
  await bridge.start();

  const seenPrompts: Array<{
    sessionId: string;
    blocks: ReadonlyArray<{ type: string; text?: string; data?: string; mimeType?: string }>;
  }> = [];
  let newSessionCalls = 0;

  const runtime = new AgyRuntime({
    cwd: "/work",
    agentsDir,
    bridge,
    bridgeToken: "tok-fallback",
    getToolMode: () => "pi",
    cancelGraceMs: 10,
    ensureBridge: async () => true,
    checkBinary: async () => ({ binary: "agy", version: "1.5.15" }),
    createClient: () => {
      let alive = true;
      return {
        get isAlive() {
          return alive;
        },
        async initialize() {
          return { protocolVersion: 1 };
        },
        async newSession() {
          newSessionCalls += 1;
          return { sessionId: `fallback-sess-${newSessionCalls}` };
        },
        async resumeSession() {
          throw new Error("cannot resume disposed session");
        },
        async setConfigOption() {
          return {};
        },
        cancelSession() {},
        async prompt(sid, blocks, opts) {
          seenPrompts.push({ sessionId: sid, blocks });
          if (seenPrompts.length === 1) {
            opts?.onUpdate?.({
              sessionUpdate: "tool_call",
              toolCallId: "tc-img-fb",
              name: "pi__read",
              title: "pi__read",
              status: "in_progress",
            });
            const res = await fetch(bridge.url!, {
              method: "POST",
              headers: {
                "content-type": "application/json",
                "x-pi-bridge-token": "tok-fallback",
              },
              body: JSON.stringify({
                jsonrpc: "2.0",
                id: 1,
                method: "tools/call",
                params: { name: "pi__read", arguments: { path: "shot.png" } },
              }),
            });
            await res.json();
            opts?.onUpdate?.({
              sessionUpdate: "tool_call_update",
              toolCallId: "tc-img-fb",
              status: "completed",
            });
            // Never resolves so cancelGraceMs triggers hard client disposal -> forces resume/newSession fallback.
            return new Promise(() => {});
          }
          opts?.onUpdate?.({
            sessionUpdate: "agent_message_chunk",
            content: { type: "text", text: "recovered-with-context" },
          });
          return { stopReason: "end_turn", usage: { totalTokens: 250 } };
        },
        async dispose() {
          alive = false;
        },
      };
    },
  });

  const branchEntries: Array<Record<string, unknown>> = [
    { type: "message", message: { role: "user" } },
  ];
  const streamSimple = createAntigravityStreamSimple({
    getRuntime: () => runtime,
    getSessionId: () => "pi-img-fb",
    getBranchEntries: () => branchEntries as never,
    getAllEntries: () => branchEntries as never,
    replayStore: new AgyReplayStore(),
  });

  try {
    const messages: unknown[] = [
      {
        role: "system",
        content: "",
        sections: { project_context: "PROJECT_RULES_KEEP" },
        timestamp: 0,
      },
      { role: "user", content: "Build feature Y and inspect shot.png", timestamp: 1 },
    ];
    const msg1 = await streamSimple(testModel(), makeTranscript(messages)).result();
    assert.equal(msg1.stopReason, "toolUse");
    const toolCall = msg1.content.find((b) => b.type === "toolCall");
    assert.ok(toolCall && toolCall.type === "toolCall");

    const toolResultMsg = {
      role: "toolResult",
      toolCallId: toolCall.id,
      toolName: "read",
      content: [
        { type: "text", text: "Read image file [image/png]" },
        { type: "image", data: "ZmFsbGJhY2s=", mimeType: "image/png" },
      ],
      isError: false,
      timestamp: 2,
    };
    branchEntries.push(
      { type: "message", message: msg1 },
      { type: "message", message: toolResultMsg },
    );

    const msg2 = await streamSimple(
      testModel(),
      makeTranscript([...messages, msg1, toolResultMsg]),
    ).result();
    assert.equal(msg2.stopReason, "stop");
    assert.deepEqual(msg2.content, [{ type: "text", text: "recovered-with-context" }]);
    assert.equal(newSessionCalls, 2);
    assert.equal(seenPrompts[1]?.sessionId, "fallback-sess-2");
    const fallbackText = seenPrompts[1]?.blocks[0]?.text ?? "";
    assert.match(fallbackText, /^PROJECT_RULES_KEEP/);
    assert.match(fallbackText, /<conversation_history>/);
    assert.match(fallbackText, /Build feature Y and inspect shot\.png/);
    assert.match(fallbackText, /<\/conversation_history>/);
    assert.match(fallbackText, /1\. read \("shot\.png"\) — image\/png/);
    assert.deepEqual(seenPrompts[1]?.blocks[1], {
      type: "image",
      data: "ZmFsbGJhY2s=",
      mimeType: "image/png",
    });
  } finally {
    await runtime.dispose();
    await bridge.close();
    fs.rmSync(agentsDir, { recursive: true, force: true });
  }
});

test("large systemPromptAppend plus >200KB history stays within MAX_AGY_PROMPT_BYTES on main and ephemeral turns", async () => {
  const agentsDir = fs.mkdtempSync(path.join(os.tmpdir(), "pi-agy-budget-agents-"));
  const seenMainTexts: string[] = [];
  const seenEphemeralTexts: string[] = [];

  const runtime = new AgyRuntime({
    cwd: "/work",
    agentsDir,
    getToolMode: () => "native",
    checkBinary: async () => ({ binary: "agy", version: "1.5.15" }),
    createClient: (opts) => {
      const isEphemeral = (opts.agentName ?? "").startsWith("pi-ephemeral-");
      let alive = true;
      return {
        get isAlive() {
          return alive;
        },
        async initialize() {
          return { protocolVersion: 1 };
        },
        async newSession() {
          return { sessionId: isEphemeral ? "eph-budget" : "main-budget" };
        },
        async resumeSession(rOpts) {
          return { sessionId: rOpts.sessionId };
        },
        async setConfigOption() {
          return {};
        },
        cancelSession() {},
        async closeSession() {},
        async prompt(_sid, blocks, promptOpts) {
          const text = blocks.map((b) => (b.type === "text" ? b.text : "")).join("\n");
          if (isEphemeral) {
            seenEphemeralTexts.push(text);
          } else {
            seenMainTexts.push(text);
          }
          promptOpts?.onUpdate?.({
            sessionUpdate: "agent_message_chunk",
            content: { type: "text", text: "ok" },
          });
          return { stopReason: "end_turn", usage: { totalTokens: 100 } };
        },
        async dispose() {
          alive = false;
        },
      };
    },
  });

  const streamSimple = createAntigravityStreamSimple({
    getRuntime: () => runtime,
    getSessionId: () => "pi-budget-main",
    getBranchEntries: () => [],
    getAllEntries: () => [],
    requireMatchingSessionId: () => true,
    replayStore: new AgyReplayStore(),
  });

  try {
    const bigSystem = `SYS_HEADER ${"A".repeat(20_000)} SYS_FOOTER`;
    const largeMessages: unknown[] = [
      {
        role: "system",
        content: "",
        sections: { project_context: bigSystem },
        timestamp: 0,
      },
      { role: "user", content: `OLD_HUGE_USER ${"U".repeat(110_000)}`, timestamp: 1 },
      {
        role: "assistant",
        content: [{ type: "text", text: `OLD_HUGE_ASSISTANT ${"R".repeat(110_000)}` }],
        api: "antigravity-acp",
        provider: "antigravity",
        model: "gemini-4-hydrogen",
        usage: {} as never,
        stopReason: "stop",
        timestamp: 2,
      },
      {
        role: "assistant",
        content: [{ type: "text", text: "RECENT_KEPT_TURN" }],
        api: "antigravity-acp",
        provider: "antigravity",
        model: "gemini-4-hydrogen",
        usage: {} as never,
        stopReason: "stop",
        timestamp: 3,
      },
      { role: "user", content: "FINAL_COMPLETE_LIVE_PROMPT", timestamp: 4 },
    ];

    const mainMsg = await streamSimple(testModel(), makeTranscript(largeMessages), {
      sessionId: "pi-budget-main",
    }).result();
    assert.equal(mainMsg.stopReason, "stop");
    assert.equal(seenMainTexts.length, 1);
    const mainText = seenMainTexts[0] ?? "";
    assert.ok(Buffer.byteLength(mainText, "utf8") <= MAX_AGY_PROMPT_BYTES);
    assert.match(mainText, /^SYS_HEADER/);
    assert.match(mainText, /SYS_FOOTER/);
    assert.match(mainText, /<\/conversation_history>/);
    assert.match(mainText, /RECENT_KEPT_TURN/);
    assert.match(mainText, /FINAL_COMPLETE_LIVE_PROMPT$/);

    const ephMsg = await streamSimple(testModel(), makeTranscript(largeMessages), {
      sessionId: "pi-budget-side",
    }).result();
    assert.equal(ephMsg.stopReason, "stop");
    assert.equal(seenEphemeralTexts.length, 1);
    const ephText = seenEphemeralTexts[0] ?? "";
    assert.ok(Buffer.byteLength(ephText, "utf8") <= MAX_AGY_PROMPT_BYTES);
    assert.match(ephText, /^SYS_HEADER/);
    assert.match(ephText, /SYS_FOOTER/);
    assert.match(ephText, /<\/conversation_history>/);
    assert.match(ephText, /RECENT_KEPT_TURN/);
    assert.match(ephText, /FINAL_COMPLETE_LIVE_PROMPT$/);
  } finally {
    await runtime.dispose();
    fs.rmSync(agentsDir, { recursive: true, force: true });
  }
});
