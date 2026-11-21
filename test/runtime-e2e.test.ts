import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { createInterface } from "node:readline";
import { PassThrough } from "node:stream";
import { test } from "node:test";
import { AgyAcpClient } from "../lib/acp-client.ts";
import {
  AgyAcpRuntime,
  type AgyAcpRuntimeOptions,
  AgyQuotaError,
  AgyRetryExhaustedError,
} from "../src/runtime.ts";

function createFakeAcpRuntime(
  handler: (params: {
    msg: Record<string, unknown>;
    sendLine: (payload: unknown) => void;
    spawnIndex: number;
    exitWith: (code: number) => void;
  }) => void,
  extraOptions: Partial<AgyAcpRuntimeOptions> = {},
) {
  let spawnCount = 0;
  const runtime = new AgyAcpRuntime({
    checkBinary: async () => ({ binary: "/fake/agy", version: "1.5.13" }),
    cancelGraceMs: 200,
    ...extraOptions,
    createClient: (opts) => {
      const spawnIndex = spawnCount++;
      return new AgyAcpClient({
        ...opts,
        spawnProcess: () => {
          const child = new EventEmitter() as any;
          child.pid = 880_000 + spawnIndex;
          const stdin = new PassThrough();
          const stdout = new PassThrough();
          const stderr = new PassThrough();
          child.stdin = stdin;
          child.stdout = stdout;
          child.stderr = stderr;

          const exitWith = (code: number) => {
            stdout.end();
            stderr.end();
            child.emit("close", code, null);
          };
          stdin.once("finish", () => {
            stdout.end();
            stderr.end();
            child.emit("close", 0, null);
          });

          const rl = createInterface({ input: stdin, crlfDelay: Number.POSITIVE_INFINITY });
          rl.on("line", (line) => {
            if (!line.trim()) return;
            const msg = JSON.parse(line) as Record<string, unknown>;
            handler({
              msg,
              sendLine: (p) => stdout.write(`${JSON.stringify(p)}\n`),
              spawnIndex,
              exitWith,
            });
          });
          return child;
        },
      });
    },
  });

  return {
    runtime,
    getSpawnCount: () => spawnCount,
  };
}

function makeConfigOptions(currentModel: string) {
  const hasThought = currentModel === "gemini-4-neon";
  return [
    {
      id: "model",
      category: "model",
      type: "select",
      currentValue: currentModel,
      options: [
        { value: "gemini-4-neon", name: "Gemini 4 Neon" },
        { value: "claude-opus-5-5", name: "Claude Opus 5.5" },
      ],
    },
    ...(hasThought
      ? [
          {
            id: "thought_level",
            category: "thought_level",
            type: "select",
            currentValue: "medium",
            options: [
              { value: "low", name: "Low" },
              { value: "medium", name: "Medium" },
              { value: "high", name: "High" },
            ],
          },
        ]
      : []),
    {
      id: "mode",
      category: "mode",
      type: "select",
      currentValue: "default",
      options: [
        { value: "default", name: "Default" },
        { value: "auto", name: "Auto" },
        { value: "plan", name: "Plan" },
      ],
    },
  ];
}

test("AgyAcpRuntime reuses process & session across turns and applies model before thought_level/mode", async () => {
  const configCalls: Array<{ configId: string; value: string }> = [];
  let currentModel = "claude-opus-5-5";

  const { runtime, getSpawnCount } = createFakeAcpRuntime(({ msg, sendLine }) => {
    if (msg.method === "initialize") {
      sendLine({ jsonrpc: "2.0", id: msg.id, result: { protocolVersion: 1 } });
    } else if (msg.method === "session/new") {
      sendLine({
        jsonrpc: "2.0",
        id: msg.id,
        result: { sessionId: "conv-1", configOptions: makeConfigOptions(currentModel) },
      });
    } else if (msg.method === "session/set_config_option") {
      const p = msg.params as { configId: string; value: string };
      configCalls.push({ configId: p.configId, value: p.value });
      if (p.configId === "model") currentModel = p.value;
      sendLine({
        jsonrpc: "2.0",
        id: msg.id,
        result: { configOptions: makeConfigOptions(currentModel) },
      });
    } else if (msg.method === "session/prompt") {
      sendLine({
        jsonrpc: "2.0",
        method: "session/update",
        params: {
          sessionId: "conv-1",
          update: { sessionUpdate: "agent_message_chunk", content: { type: "text", text: "ok" } },
        },
      });
      sendLine({
        jsonrpc: "2.0",
        id: msg.id,
        result: { stopReason: "end_turn", usage: { inputTokens: 10, outputTokens: 5 } },
      });
    } else if (msg.method === "session/close") {
      sendLine({ jsonrpc: "2.0", id: msg.id, result: {} });
    }
  });

  const turn1 = await runtime.runTurn({
    cwd: "/repo",
    modelId: "gemini-4-neon",
    effort: "xhigh",
    executionMode: "auto",
    buildPrompt: ({ usedHistoryBootstrap }) => {
      assert.equal(usedHistoryBootstrap, true);
      return { blocks: [{ type: "text", text: "turn 1" }] };
    },
  });
  assert.equal(turn1.sessionId, "conv-1");
  assert.equal(turn1.resumed, false);
  // First write switches model from claude-opus-5-5 -> gemini-4-neon, exposing thought_level.
  assert.deepEqual(configCalls, [
    { configId: "model", value: "gemini-4-neon" },
    { configId: "thought_level", value: "high" },
    { configId: "mode", value: "auto" },
  ]);

  const turn2 = await runtime.runTurn({
    cwd: "/repo",
    desiredConversationId: "conv-1",
    modelId: "gemini-4-neon",
    effort: "xhigh",
    executionMode: "auto",
    buildPrompt: ({ usedHistoryBootstrap }) => {
      assert.equal(usedHistoryBootstrap, false);
      return { blocks: [{ type: "text", text: "turn 2" }] };
    },
  });
  assert.equal(turn2.resumed, true);
  assert.equal(getSpawnCount(), 1);
  await runtime.stop();
});

test("AgyAcpRuntime falls back to newSession + historyBootstrap when resume returns wrong sessionId", async () => {
  let missingCalledWith: string | undefined;
  let capturedPromptText = "";
  let currentModel = "gemini-4-neon";

  const { runtime } = createFakeAcpRuntime(({ msg, sendLine }) => {
    if (msg.method === "initialize") {
      sendLine({ jsonrpc: "2.0", id: msg.id, result: { protocolVersion: 1 } });
    } else if (msg.method === "session/resume") {
      // Mismatched id triggers MissingAgyConversationError + closeSession on rogue-id
      sendLine({ jsonrpc: "2.0", id: msg.id, result: { sessionId: "rogue-id" } });
    } else if (msg.method === "session/close") {
      sendLine({ jsonrpc: "2.0", id: msg.id, result: {} });
    } else if (msg.method === "session/new") {
      assert.equal((msg.params as any)._meta.antigravity.systemPrompt.append, "System rules");
      sendLine({
        jsonrpc: "2.0",
        id: msg.id,
        result: { sessionId: "fresh-conv", configOptions: makeConfigOptions(currentModel) },
      });
    } else if (msg.method === "session/set_config_option") {
      const p = msg.params as { configId: string; value: string };
      if (p.configId === "model") currentModel = p.value;
      sendLine({
        jsonrpc: "2.0",
        id: msg.id,
        result: { configOptions: makeConfigOptions(currentModel) },
      });
    } else if (msg.method === "session/prompt") {
      capturedPromptText = ((msg.params as any).prompt[0] as { text: string }).text;
      sendLine({ jsonrpc: "2.0", id: msg.id, result: { stopReason: "end_turn" } });
    }
  });

  const result = await runtime.runTurn({
    cwd: "/repo",
    desiredConversationId: "missing-old-id",
    systemPromptAppend: "System rules",
    modelId: "gemini-4-neon",
    onMissingConversation: (id) => {
      missingCalledWith = id;
    },
    buildPrompt: ({ usedHistoryBootstrap }) => ({
      blocks: [
        {
          type: "text",
          text: usedHistoryBootstrap ? "RESTORED HISTORY\n\nhi" : "hi",
        },
      ],
    }),
  });

  assert.equal(missingCalledWith, "missing-old-id");
  assert.equal(result.sessionId, "fresh-conv");
  assert.equal(result.resumed, false);
  assert.equal(capturedPromptText, "RESTORED HISTORY\n\nhi");
  await runtime.stop();
});

test("AgyAcpRuntime fails fake end_turn quota exhaustion with AgyQuotaError", async () => {
  let quotaCallbackFired = false;

  const { runtime } = createFakeAcpRuntime(({ msg, sendLine }) => {
    if (msg.method === "initialize") {
      sendLine({ jsonrpc: "2.0", id: msg.id, result: { protocolVersion: 1 } });
    } else if (msg.method === "session/new") {
      sendLine({ jsonrpc: "2.0", id: msg.id, result: { sessionId: "q-1", configOptions: [] } });
    } else if (msg.method === "session/prompt") {
      sendLine({
        jsonrpc: "2.0",
        method: "session/update",
        params: {
          sessionId: "q-1",
          update: {
            sessionUpdate: "agent_message_chunk",
            content: {
              type: "text",
              text: "\n⚠ Individual quota reached for this model. Resets in 2h 10m\n",
            },
          },
        },
      });
      sendLine({
        jsonrpc: "2.0",
        id: msg.id,
        result: { stopReason: "end_turn" },
      });
    }
  });

  await assert.rejects(
    runtime.runTurn({
      cwd: "/repo",
      modelId: "gemini-4-neon",
      onQuotaExhausted: () => {
        quotaCallbackFired = true;
      },
      buildPrompt: () => ({ blocks: [{ type: "text", text: "hi" }] }),
    }),
    (err: unknown) => {
      assert.ok(err instanceof AgyQuotaError);
      assert.equal(err.resetsIn, "2h 10m");
      return true;
    },
  );
  assert.equal(quotaCallbackFired, true);
  await runtime.stop();
});

test("AgyAcpRuntime parks bridge waves and resumes via continueParkedTurn", async () => {
  let promptRpcId: unknown;
  let sendLineRef: ((p: unknown) => void) | undefined;

  const { runtime } = createFakeAcpRuntime(({ msg, sendLine }) => {
    sendLineRef = sendLine;
    if (msg.method === "initialize") {
      sendLine({ jsonrpc: "2.0", id: msg.id, result: { protocolVersion: 1 } });
    } else if (msg.method === "session/new") {
      sendLine({ jsonrpc: "2.0", id: msg.id, result: { sessionId: "p-1", configOptions: [] } });
    } else if (msg.method === "session/prompt") {
      promptRpcId = msg.id;
      sendLine({
        jsonrpc: "2.0",
        method: "session/update",
        params: {
          sessionId: "p-1",
          update: {
            sessionUpdate: "agent_thought_chunk",
            content: { type: "text", text: "Need file" },
          },
        },
      });
      setTimeout(() => {
        runtime.notifyBridgeCall({ id: "call-1", tool: "read", args: { path: "README.md" } });
      }, 15);
    }
  });

  const wave1 = await runtime.runTurn({
    cwd: "/repo",
    modelId: "gemini-4-neon",
    buildPrompt: () => ({ blocks: [{ type: "text", text: "inspect readme" }] }),
  });

  assert.equal(wave1.outcome.kind, "parked_tool_calls");
  assert.equal(wave1.outcome.bridgeCalls[0]!.id, "call-1");
  assert.equal(runtime.hasParkedTurn, true);

  setTimeout(() => {
    sendLineRef!({
      jsonrpc: "2.0",
      method: "session/update",
      params: {
        sessionId: "p-1",
        update: {
          sessionUpdate: "agent_message_chunk",
          content: { type: "text", text: "Done reading." },
        },
      },
    });
    sendLineRef!({
      jsonrpc: "2.0",
      id: promptRpcId,
      result: { stopReason: "end_turn", usage: { inputTokens: 100, outputTokens: 20 } },
    });
  }, 15);

  const wave2 = await runtime.continueParkedTurn({});
  assert.equal(wave2.outcome.kind, "complete");
  if (wave2.outcome.kind === "complete") {
    assert.equal(wave2.outcome.usage?.output_tokens, 20);
  }
  assert.equal(runtime.hasParkedTurn, false);
  await runtime.stop();
});

test("AgyAcpRuntime caps retries at 3 attempts, cancels session, and fails with AgyRetryExhaustedError", async () => {
  const brainDir = mkdtempSync(path.join(os.tmpdir(), "agy-retry-"));
  const logsDir = path.join(brainDir, "r-1", ".system_generated", "logs");
  mkdirSync(logsDir, { recursive: true });
  const transcriptPath = path.join(logsDir, "transcript.jsonl");

  let cancelNotificationSent = false;
  let quotaCallbackFired = false;
  const hints: string[] = [];

  const { runtime } = createFakeAcpRuntime(
    ({ msg, sendLine }) => {
      if (msg.method === "initialize") {
        sendLine({ jsonrpc: "2.0", id: msg.id, result: { protocolVersion: 1 } });
      } else if (msg.method === "session/new") {
        sendLine({ jsonrpc: "2.0", id: msg.id, result: { sessionId: "r-1", configOptions: [] } });
      } else if (msg.method === "session/prompt") {
        writeFileSync(
          transcriptPath,
          `${JSON.stringify({
            step_index: 1,
            source: "SYSTEM",
            type: "ERROR_MESSAGE",
            error: "API error (attempt 1): RESOURCE_EXHAUSTED (code 429)",
          })}\n`,
        );
        setTimeout(() => {
          writeFileSync(
            transcriptPath,
            [
              JSON.stringify({
                step_index: 1,
                source: "SYSTEM",
                type: "ERROR_MESSAGE",
                error: "API error (attempt 1): RESOURCE_EXHAUSTED (code 429)",
              }),
              JSON.stringify({
                step_index: 2,
                source: "SYSTEM",
                type: "ERROR_MESSAGE",
                error: "API error (attempt 2): RESOURCE_EXHAUSTED (code 429)",
              }),
            ].join("\n") + "\n",
          );
        }, 35);
        setTimeout(() => {
          writeFileSync(
            transcriptPath,
            [
              JSON.stringify({
                step_index: 1,
                source: "SYSTEM",
                type: "ERROR_MESSAGE",
                error: "API error (attempt 1): RESOURCE_EXHAUSTED (code 429)",
              }),
              JSON.stringify({
                step_index: 2,
                source: "SYSTEM",
                type: "ERROR_MESSAGE",
                error: "API error (attempt 2): RESOURCE_EXHAUSTED (code 429)",
              }),
              JSON.stringify({
                step_index: 3,
                source: "SYSTEM",
                type: "ERROR_MESSAGE",
                error: "API error (attempt 3): RESOURCE_EXHAUSTED (code 429)",
              }),
            ].join("\n") + "\n",
          );
        }, 75);
      } else if (msg.method === "session/cancel") {
        cancelNotificationSent = true;
      }
    },
    { brainDir, transcriptPollIntervalMs: 20, maxApiAttempts: 3 },
  );

  try {
    await assert.rejects(
      runtime.runTurn({
        cwd: "/repo",
        modelId: "gemini-4-neon",
        onQuotaExhausted: () => {
          quotaCallbackFired = true;
        },
        onActivity: (act) => {
          if (act.type === "status_hint") hints.push(act.text);
        },
        buildPrompt: () => ({ blocks: [{ type: "text", text: "hi" }] }),
      }),
      (err: unknown) => {
        assert.ok(err instanceof AgyRetryExhaustedError);
        assert.equal(err.attempts, 3);
        assert.ok(err.message.includes("quota exceeded"));
        return true;
      },
    );

    assert.ok(hints.some((h) => h.startsWith("retry 1/3")));
    assert.ok(hints.some((h) => h.startsWith("retry 2/3")));
    assert.equal(cancelNotificationSent, true);
    assert.equal(quotaCallbackFired, true);
  } finally {
    await runtime.stop();
    rmSync(brainDir, { recursive: true, force: true });
  }
});
