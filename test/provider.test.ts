import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import { createInterface } from "node:readline";
import { PassThrough } from "node:stream";
import { test } from "node:test";
import type { AssistantMessageEvent, Model, TranscriptContext } from "@earendil-works/pi-ai";
import { isRetryableAssistantError } from "@earendil-works/pi-ai/utils/retry";
import { AgyAcpClient } from "../lib/acp-client.ts";
import { AgyPiBridge, BRIDGE_CALL_TIMEOUT_MS } from "../lib/bridge.ts";
import { AgyReplayStore } from "../lib/replay.ts";
import {
  buildAgySystemPromptAppend,
  buildRestoredPiTranscript,
  createAntigravityStreamSimple,
  extractTrailingUserPrompt,
} from "../src/provider.ts";
import { AgyAcpRuntime, AgyQuotaError, AgyRetryExhaustedError } from "../src/runtime.ts";
import { assertDeltasMatchPartial } from "./delta-replay.ts";

const TEST_MODEL: Model<string> = {
  id: "gemini-4-neon",
  name: "Gemini 4 Neon",
  api: "antigravity-acp",
  provider: "antigravity",
  baseUrl: "acp://agy",
  reasoning: true,
  input: ["text", "image"],
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
  contextWindow: 1_000_000,
  maxTokens: 65_536,
};

function makeContext(messages: any[]): TranscriptContext {
  return { messages } as unknown as TranscriptContext;
}

test("buildAgySystemPromptAppend returns undefined for empty Pi system prompt without docs", () => {
  assert.equal(buildAgySystemPromptAppend({ piSystemPrompt: "" }), undefined);
  const built = buildAgySystemPromptAppend({
    piSystemPrompt: [
      "Available tools:",
      "- read: Read file contents",
      "In addition to the tools above, use skills when relevant.",
      "",
      "Guidelines:",
      "- Be concise.",
    ].join("\n"),
  });
  assert.equal(built?.includes("- Be concise."), true);
  assert.equal(built?.includes("Read file contents"), false);
});

test("createAntigravityStreamSimple streams thoughts, replay wrapper cards (excluding bridged MCP), and passes delta-replay invariance", async () => {
  let capturedBlocks: any[] = [];
  let capturedSystemPromptAppend: string | undefined;

  const runtime = new AgyAcpRuntime({
    checkBinary: async () => ({ binary: "/fake/agy", version: "1.5.13" }),
    createClient: (opts) =>
      new AgyAcpClient({
        ...opts,
        spawnProcess: () => {
          const child = new EventEmitter() as any;
          child.pid = 777_001;
          const stdin = new PassThrough();
          const stdout = new PassThrough();
          const stderr = new PassThrough();
          child.stdin = stdin;
          child.stdout = stdout;
          child.stderr = stderr;
          stdin.once("finish", () => {
            stdout.end();
            stderr.end();
            child.emit("close", 0, null);
          });

          const send = (payload: unknown) => stdout.write(`${JSON.stringify(payload)}\n`);
          const rl = createInterface({ input: stdin, crlfDelay: Number.POSITIVE_INFINITY });
          rl.on("line", (line) => {
            if (!line.trim()) return;
            const msg = JSON.parse(line) as any;
            if (msg.method === "initialize") {
              send({ jsonrpc: "2.0", id: msg.id, result: { protocolVersion: 1 } });
            } else if (msg.method === "session/new") {
              capturedSystemPromptAppend = msg.params?._meta?.antigravity?.systemPrompt?.append;
              send({ jsonrpc: "2.0", id: msg.id, result: { sessionId: "s-1", configOptions: [] } });
            } else if (msg.method === "session/prompt") {
              capturedBlocks = msg.params.prompt;
              const emitUpdate = (update: unknown) =>
                send({
                  jsonrpc: "2.0",
                  method: "session/update",
                  params: { sessionId: "s-1", update },
                });

              emitUpdate({
                sessionUpdate: "agent_thought_chunk",
                content: { type: "text", text: "Let's check" },
              });
              emitUpdate({
                sessionUpdate: "agent_message_chunk",
                content: { type: "text", text: "Searching..." },
              });
              // Non-bridged agy tool -> should become `antigravity` replay card
              emitUpdate({
                sessionUpdate: "tool_call_update",
                toolCallId: "t-web",
                name: "search_web",
                status: "completed",
                rawInput: { query: "pi agent" },
                rawOutput: "3 results found",
              });
              // Bridged MCP tool -> should be ignored in replay synthesis
              emitUpdate({
                sessionUpdate: "tool_call_update",
                toolCallId: "t-mcp",
                name: "mcp_pi_deadbeef_read",
                status: "completed",
                rawOutput: "file data",
              });
              emitUpdate({
                sessionUpdate: "usage_update",
                used: 4_500,
                size: 1_000_000,
              });
              emitUpdate({
                sessionUpdate: "agent_message_chunk",
                content: { type: "text", text: "Done!" },
              });
              send({
                jsonrpc: "2.0",
                id: msg.id,
                result: {
                  stopReason: "end_turn",
                  usage: { inputTokens: 100, outputTokens: 25, cachedReadTokens: 400 },
                },
              });
            }
          });
          return child;
        },
      }),
  });

  const bridge = new AgyPiBridge("pi-deadbeef", BRIDGE_CALL_TIMEOUT_MS);
  const replayStore = new AgyReplayStore();
  let recordedUsage: any;

  const streamSimple = createAntigravityStreamSimple({
    runtime,
    replayStore,
    bridge,
    getToolMode: () => "native",
    getExecutionMode: () => "auto",
    getBridgeRevision: () => "0:0",
    getConversationId: () => undefined,
    onUsage: (u) => {
      recordedUsage = u;
    },
  });

  const events: AssistantMessageEvent[] = [];
  const s = streamSimple(
    TEST_MODEL,
    makeContext([
      {
        role: "user",
        content: [
          { type: "text", text: "inspect screenshot" },
          { type: "image", data: "YWJj", mimeType: "image/png" },
        ],
        timestamp: 1,
      },
    ]),
  );
  for await (const ev of s) {
    events.push(ev);
  }
  const finalMsg = await s.result();

  assert.equal(capturedSystemPromptAppend, undefined);
  assert.equal(capturedBlocks[0].type, "text");
  assert.equal(capturedBlocks[0].text, "inspect screenshot");
  assert.deepEqual(capturedBlocks[1], { type: "image", data: "YWJj", mimeType: "image/png" });

  assertDeltasMatchPartial(events);
  assert.equal(finalMsg.stopReason, "toolUse");
  assert.equal(finalMsg.content.length, 4);
  assert.equal(finalMsg.content[0]!.type, "thinking");
  assert.equal(finalMsg.content[1]!.type, "text");
  assert.equal(finalMsg.content[2]!.type, "toolCall");
  assert.equal((finalMsg.content[2] as any).name, "antigravity");
  assert.equal(finalMsg.content[3]!.type, "text");
  assert.equal((finalMsg.content[3] as any).text, "Done!");

  assert.equal(replayStore.size, 1);
  const replayed = replayStore.take((finalMsg.content[2] as any).id);
  assert.equal(replayed?.agyTool, "search_web");
  assert.equal(replayed?.output, "3 results found");
  assert.equal(recordedUsage?.input_tokens, 100);
  await runtime.stop();
});

test("createAntigravityStreamSimple continues parked turn across toolResult wave", async () => {
  let promptRpcId: unknown;
  let sendRef: ((p: unknown) => void) | undefined;

  const runtime = new AgyAcpRuntime({
    checkBinary: async () => ({ binary: "/fake/agy", version: "1.5.13" }),
    createClient: (opts) =>
      new AgyAcpClient({
        ...opts,
        spawnProcess: () => {
          const child = new EventEmitter() as any;
          child.pid = 777_002;
          const stdin = new PassThrough();
          const stdout = new PassThrough();
          const stderr = new PassThrough();
          child.stdin = stdin;
          child.stdout = stdout;
          child.stderr = stderr;
          stdin.once("finish", () => {
            stdout.end();
            stderr.end();
            child.emit("close", 0, null);
          });

          sendRef = (payload: unknown) => stdout.write(`${JSON.stringify(payload)}\n`);
          const rl = createInterface({ input: stdin, crlfDelay: Number.POSITIVE_INFINITY });
          rl.on("line", (line) => {
            if (!line.trim()) return;
            const msg = JSON.parse(line) as any;
            if (msg.method === "initialize") {
              sendRef!({ jsonrpc: "2.0", id: msg.id, result: { protocolVersion: 1 } });
            } else if (msg.method === "session/new") {
              sendRef!({
                jsonrpc: "2.0",
                id: msg.id,
                result: { sessionId: "p-wave", configOptions: [] },
              });
            } else if (msg.method === "session/prompt") {
              promptRpcId = msg.id;
              setTimeout(() => {
                runtime.notifyBridgeCall({ id: "b-1", tool: "bash", args: { command: "pwd" } });
              }, 10);
            }
          });
          return child;
        },
      }),
  });

  const bridge = new AgyPiBridge("pi-wave", BRIDGE_CALL_TIMEOUT_MS);
  const replayStore = new AgyReplayStore();
  const streamSimple = createAntigravityStreamSimple({
    runtime,
    replayStore,
    bridge,
    getToolMode: () => "pi",
    getExecutionMode: () => "auto",
    getBridgeRevision: () => "0:0",
    getConversationId: () => undefined,
  });

  const wave1Events: AssistantMessageEvent[] = [];
  const s1 = streamSimple(
    TEST_MODEL,
    makeContext([{ role: "user", content: "where am I?", timestamp: 1 }]),
  );
  for await (const ev of s1) wave1Events.push(ev);
  const m1 = await s1.result();
  assertDeltasMatchPartial(wave1Events);
  assert.equal(m1.stopReason, "toolUse");
  assert.equal((m1.content[0] as any).id, "b-1");
  assert.equal((m1.content[0] as any).name, "bash");

  setTimeout(() => {
    sendRef!({
      jsonrpc: "2.0",
      method: "session/update",
      params: {
        sessionId: "p-wave",
        update: {
          sessionUpdate: "agent_message_chunk",
          content: { type: "text", text: "You are in /repo" },
        },
      },
    });
    sendRef!({
      jsonrpc: "2.0",
      id: promptRpcId,
      result: { stopReason: "end_turn", usage: { inputTokens: 10, outputTokens: 5 } },
    });
  }, 15);

  const wave2Events: AssistantMessageEvent[] = [];
  const s2 = streamSimple(
    TEST_MODEL,
    makeContext([
      { role: "user", content: "where am I?", timestamp: 1 },
      m1,
      {
        role: "toolResult",
        toolCallId: "b-1",
        toolName: "bash",
        content: [{ type: "text", text: "/repo" }],
        isError: false,
        timestamp: 2,
      },
    ]),
  );
  for await (const ev of s2) wave2Events.push(ev);
  const m2 = await s2.result();
  assertDeltasMatchPartial(wave2Events);
  assert.equal(m2.stopReason, "stop");
  assert.equal((m2.content[0] as any).text, "You are in /repo");
  await runtime.stop();
});

test("extractTrailingUserPrompt preserves user text followed by hindsight/custom user messages across turns", () => {
  const turn1Messages = [
    { role: "system", content: "system", timestamp: 0 },
    { role: "user", content: [{ type: "text", text: "hi" }], timestamp: 1 },
    { role: "user", content: "<hindsight_pages>memory</hindsight_pages>", timestamp: 2 },
  ] as any[];

  const t1 = extractTrailingUserPrompt(turn1Messages);
  assert.equal(t1.userIndex, 1);
  assert.equal(t1.text, "hi\n\n<hindsight_pages>memory</hindsight_pages>");
  assert.equal(buildRestoredPiTranscript(turn1Messages, t1.userIndex), undefined);

  const turn2Messages = [
    ...turn1Messages,
    { role: "assistant", content: [{ type: "text", text: "Hello!" }], timestamp: 3 },
    { role: "user", content: [{ type: "text", text: "turn2" }], timestamp: 4 },
    { role: "user", content: "<hindsight_pages>memory2</hindsight_pages>", timestamp: 5 },
  ] as any[];

  const t2 = extractTrailingUserPrompt(turn2Messages);
  assert.equal(t2.userIndex, 4);
  assert.equal(t2.text, "turn2\n\n<hindsight_pages>memory2</hindsight_pages>");
});

test("AgyQuotaError and AgyRetryExhaustedError prevent outer Pi 429 retry loops", () => {
  const retryExhausted = new AgyRetryExhaustedError(
    "API error (attempt 3): RESOURCE_EXHAUSTED (code 429): Resource has been exhausted.",
    3,
  );
  const quotaErr = new AgyQuotaError("Individual quota reached", "2h 10m");

  for (const err of [retryExhausted, quotaErr]) {
    const msg = {
      role: "assistant" as const,
      content: [],
      api: "antigravity-acp",
      provider: "antigravity",
      model: "gemini-4-neon",
      usage: {
        input: 0,
        output: 0,
        cacheRead: 0,
        cacheWrite: 0,
        totalTokens: 0,
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
      },
      stopReason: "error" as const,
      errorMessage: err.message,
      timestamp: Date.now(),
    };
    assert.equal(isRetryableAssistantError(msg), false);
  }
});
