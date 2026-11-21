import assert from "node:assert/strict";
import test from "node:test";
import {
  type Api,
  isRetryableAssistantError,
  type Model,
  type TranscriptContext,
} from "@earendil-works/pi-ai";
import { AgyPiBridge } from "../src/bridge/server.ts";
import {
  type AgyAcpClientLike,
  AgyFirstResponseTimeoutError,
  AgyQuotaError,
  AgyRuntime,
  DEFAULT_FIRST_RESPONSE_TIMEOUT_MS,
} from "../src/provider/runtime.ts";
import { AGY_STATE_ENTRY_TYPE, type AgyConversationState } from "../src/provider/state.ts";
import { createAntigravityStreamSimple } from "../src/provider/stream.ts";
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
    async newSession() {
      calls.push("newSession");
      return {
        sessionId: "acp-fresh",
        configOptions: snapshotConfig(),
      };
    },
    async resumeSession() {
      resumeAttempts += 1;
      calls.push("resumeSession");
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
  const bridge = new AgyPiBridge("pi-wave");
  bridge.setToolSource(() => [{ name: "read", description: "read", parameters: {} }]);
  bridge.refreshTools();
  await bridge.start();

  const runtime = new AgyRuntime({
    cwd: "/work",
    bridge,
    getToolMode: () => "pi",
    ensureBridge: async () => true,
    checkBinary: async () => ({ binary: "agy", version: "1.5.13" }),
    createClient: () => ({
      isAlive: true,
      async initialize() {
        return { protocolVersion: 1 };
      },
      async newSession() {
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
          sessionUpdate: "agent_message_chunk",
          content: { type: "text", text: `read:${json.result.content[0]?.text}` },
        });
        return { stopReason: "end_turn" };
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
    const toolCall = msg1.content.find((b) => b.type === "toolCall");
    assert.ok(toolCall && toolCall.type === "toolCall");
    assert.equal(toolCall.name, "read");

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
  } finally {
    await runtime.dispose();
    await bridge.close();
  }
});

test("native wrapper tool replay store preserves output and daemon parked errors accurately", async () => {
  const replayStore = new AgyReplayStore();
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
        opts?.onUpdate?.({
          sessionUpdate: "tool_call",
          toolCallId: "n1",
          title: "view_file",
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
          toolCallId: "d1",
          title: "run_command",
          status: "in_progress",
          rawInput: { CommandLine: "sleep 10", IsDaemon: true },
        });
        return { stopReason: "end_turn" };
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

  const s = streamSimple(
    testModel(),
    makeTranscript([{ role: "user", content: "go", timestamp: 1 }]),
  );
  const events = [];
  for await (const ev of s) events.push(ev);
  assertDeltasMatchPartial(events);
  const msg = await s.result();
  assert.equal(msg.stopReason, "toolUse");
  const calls = msg.content.filter((b) => b.type === "toolCall");
  assert.equal(calls.length, 2);
  assert.equal(replayStore.take(calls[0].id)?.output, "line1");
  assert.match(replayStore.take(calls[1].id)?.error ?? "", /background task/);
  await runtime.dispose();
});

test("warmupSession completes setup before streamSimple and reloads client when bridge catalog changes", async () => {
  let spawnCount = 0;
  const calls: string[] = [];
  const bridge = new AgyPiBridge("pi-warmup-test");
  bridge.setToolSource(() => [{ name: "read", description: "read file", parameters: {} }]);
  bridge.refreshTools();
  await bridge.start();

  try {
    let aliveClient: { isAlive: boolean } | undefined;
    const runtime = new AgyRuntime({
      cwd: "/work",
      bridge,
      getToolMode: () => "pi",
      checkBinary: async () => ({ binary: "agy", version: "1.5.13" }),
      createClient: () => {
        spawnCount += 1;
        const clientId = spawnCount;
        const client = {
          isAlive: true,
          async initialize() {
            calls.push(`init:${clientId}`);
            return { protocolVersion: 1 };
          },
          async newSession() {
            calls.push(`new:${clientId}`);
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
          async resumeSession({ sessionId }: { sessionId: string }) {
            calls.push(`resume:${clientId}:${sessionId}`);
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
    await runtime.dispose();
  } finally {
    await bridge.close();
  }
});
