import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import test from "node:test";
import { getAgyBinary } from "../src/acp/binary.ts";
import {
  AcpRequestStallError,
  AgyAcpClient,
  MissingAgyConversationError,
} from "../src/acp/client.ts";

type SpawnProcessFn = NonNullable<ConstructorParameters<typeof AgyAcpClient>[0]["spawnProcess"]>;

function spawnMockAcpProcess(
  script: string,
  _binary: string,
  _args: string[],
  options: Parameters<typeof spawn>[2],
) {
  return spawn(process.execPath, ["-e", script], options);
}

const MOCK_ACP_SERVER_SCRIPT = `
const readline = require("node:readline");
const rl = readline.createInterface({ input: process.stdin });
let promptTimer;
rl.on("line", (line) => {
  if (!line.trim()) return;
  const msg = JSON.parse(line);
  if (msg.method === "initialize") {
    process.stdout.write(JSON.stringify({
      jsonrpc: "2.0",
      id: msg.id,
      result: { protocolVersion: 1, agentInfo: { name: "mock-agy", version: "1.5.13" } }
    }) + "\\n");
    return;
  }
  if (msg.method === "session/new") {
    process.stdout.write(JSON.stringify({
      jsonrpc: "2.0",
      id: msg.id,
      result: {
        sessionId: "sess-1",
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
          { id: "model", category: "model", type: "select", currentValue: "gemini-pro" }
        ]
      }
    }) + "\\n");
    return;
  }
  if (msg.method === "session/resume") {
    process.stdout.write(JSON.stringify({
      jsonrpc: "2.0",
      id: msg.id,
      error: { code: -32004, message: "session not found" }
    }) + "\\n");
    return;
  }
  if (msg.method === "session/prompt") {
    const sid = msg.params.sessionId;
    const text = (msg.params.prompt && msg.params.prompt[0] && msg.params.prompt[0].text) || "";
    if (text === "hang-until-cancel") {
      // Wait silently for session/cancel notification.
      return;
    }
    if (text === "ask-permission") {
      process.stdout.write(JSON.stringify({
        jsonrpc: "2.0",
        id: "perm-1",
        method: "session/request_permission",
        params: {
          sessionId: sid,
          toolCall: { toolCallId: "tc-1", title: "bash" },
          options: [{ optionId: "allow_once", kind: "allow_once" }]
        }
      }) + "\\n");
    }
    process.stdout.write(JSON.stringify({
      jsonrpc: "2.0",
      method: "session/update",
      params: {
        sessionId: sid,
        update: { sessionUpdate: "agent_message_chunk", content: { type: "text", text: "hello:" + text } }
      }
    }) + "\\n");
    promptTimer = setTimeout(() => {
      process.stdout.write(JSON.stringify({
        jsonrpc: "2.0",
        id: msg.id,
        result: { stopReason: "end_turn", usage: { inputTokens: 10, outputTokens: 5, totalTokens: 15 } }
      }) + "\\n");
    }, 10);
    return;
  }
  if (msg.method === "session/cancel") {
    clearTimeout(promptTimer);
    // Cancel pending prompt request if needed cleanly via close or cancelled response.
    process.stdout.write(JSON.stringify({
      jsonrpc: "2.0",
      method: "session/update",
      params: { sessionId: msg.params.sessionId, update: { sessionUpdate: "agent_message_chunk", content: { type: "text", text: "" } } }
    }) + "\\n");
  }
});
`;

test("AgyAcpClient initialize, newSession, resumeSession missing error, prompt updates, and dispose", async () => {
  const client = new AgyAcpClient({
    binary: process.execPath,
    cwd: process.cwd(),
    env: { CUSTOM_ENV_FLAG: "yes" },
    spawnProcess: ((binary: string, args: string[], options: Parameters<typeof spawn>[2]) =>
      spawnMockAcpProcess(
        MOCK_ACP_SERVER_SCRIPT,
        binary,
        args,
        options,
      )) as unknown as SpawnProcessFn,
  });

  try {
    const init = await client.initialize();
    assert.equal(init.protocolVersion, 1);
    assert.equal(init.agentInfo?.name, "mock-agy");

    const session = await client.newSession({ cwd: process.cwd() });
    assert.equal(session.sessionId, "sess-1");

    await assert.rejects(
      () => client.resumeSession({ sessionId: "missing-id", cwd: process.cwd() }),
      (err: unknown) => err instanceof MissingAgyConversationError,
    );

    const chunks: string[] = [];
    const result = await client.prompt("sess-1", [{ type: "text", text: "ask-permission" }], {
      onUpdate: (u) => {
        const content = u.content as { text?: string } | undefined;
        if (content?.text) chunks.push(content.text);
      },
    });
    assert.equal(result.stopReason, "end_turn");
    assert.equal(result.usage?.totalTokens, 15);
    assert.deepEqual(chunks, ["hello:ask-permission"]);
  } finally {
    await client.dispose(100);
    assert.equal(client.isAlive, false);
  }
});

test("AgyAcpClient aborts prompt via AbortSignal and kills process tree on dispose", async () => {
  const client = new AgyAcpClient({
    binary: process.execPath,
    cwd: process.cwd(),
    cancelGraceMs: 50,
    spawnProcess: ((binary: string, args: string[], options: Parameters<typeof spawn>[2]) =>
      spawnMockAcpProcess(
        MOCK_ACP_SERVER_SCRIPT,
        binary,
        args,
        options,
      )) as unknown as SpawnProcessFn,
  });

  try {
    await client.initialize();
    await client.newSession({ cwd: process.cwd() });

    const abort = new AbortController();
    const timer = setTimeout(() => abort.abort(), 30);
    await assert.rejects(
      () =>
        client.prompt("sess-1", [{ type: "text", text: "hang-until-cancel" }], {
          signal: abort.signal,
          cancelGraceMs: 50,
        }),
      (err: unknown) => err instanceof Error && err.name === "AbortError",
    );
    clearTimeout(timer);
  } finally {
    await client.dispose(50);
  }
});

test("Live agy acp initialize handshake (non-LLM smoke test when agy CLI >= 1.5.0 is present)", async (t) => {
  let binary: string;
  try {
    binary = getAgyBinary();
  } catch {
    t.skip("agy CLI binary not installed");
    return;
  }

  const client = new AgyAcpClient({
    binary,
    cwd: process.cwd(),
  });
  try {
    const init = await client.initialize();
    assert.equal(init.protocolVersion, 1);
    assert.ok(init.agentInfo?.name, "expected ACP agentInfo.name from live agy acp");
  } finally {
    await client.dispose(300);
  }
});

test("AgyAcpClient initialize and newSession honor AbortSignal and timeoutMs", async () => {
  const hangInitScript = `
    const readline = require("node:readline");
    const rl = readline.createInterface({ input: process.stdin });
    rl.on("line", () => {});
  `;
  const client1 = new AgyAcpClient({
    binary: process.execPath,
    cwd: process.cwd(),
    spawnProcess: ((binary: string, args: string[], options: Parameters<typeof spawn>[2]) =>
      spawnMockAcpProcess(hangInitScript, binary, args, options)) as unknown as SpawnProcessFn,
  });
  try {
    const abort = new AbortController();
    const timer = setTimeout(() => abort.abort(), 20);
    await assert.rejects(
      () => client1.initialize({ signal: abort.signal }),
      (err: unknown) => err instanceof Error && err.name === "AbortError",
    );
    clearTimeout(timer);
  } finally {
    await client1.dispose(50);
  }

  const hangNewSessionScript = `
    const readline = require("node:readline");
    const rl = readline.createInterface({ input: process.stdin });
    rl.on("line", (line) => {
      if (!line.trim()) return;
      const msg = JSON.parse(line);
      if (msg.method === "initialize") {
        process.stdout.write(JSON.stringify({ jsonrpc: "2.0", id: msg.id, result: { protocolVersion: 1 } }) + "\\n");
      }
    });
  `;
  const client2 = new AgyAcpClient({
    binary: process.execPath,
    cwd: process.cwd(),
    spawnProcess: ((binary: string, args: string[], options: Parameters<typeof spawn>[2]) =>
      spawnMockAcpProcess(
        hangNewSessionScript,
        binary,
        args,
        options,
      )) as unknown as SpawnProcessFn,
  });
  try {
    await client2.initialize({ timeoutMs: 500 });
    await assert.rejects(
      () => client2.newSession({ cwd: process.cwd(), timeoutMs: 30 }),
      (err: unknown) =>
        err instanceof AcpRequestStallError &&
        err.method === "session/new" &&
        !/timed?\s*out/i.test(err.message),
    );
  } finally {
    await client2.dispose(50);
  }
});

test("AgyAcpClient.dispose sends SIGTERM after stdin EOF grace and escalates to SIGKILL when ignored", async (t) => {
  if (process.platform === "win32") {
    t.skip("POSIX SIGTERM/SIGKILL escalation test");
    return;
  }

  const trapSigtermScript = `
    const readline = require("node:readline");
    const rl = readline.createInterface({ input: process.stdin });
    rl.on("line", () => {});
    process.stdin.on("end", () => {
      // Ignore stdin EOF so dispose must escalate to SIGTERM.
    });
    process.on("SIGTERM", () => {
      process.stderr.write("saw-sigterm\\n");
      process.exit(0);
    });
    setInterval(() => {}, 1000);
  `;
  const client1 = new AgyAcpClient({
    binary: process.execPath,
    cwd: process.cwd(),
    spawnProcess: ((binary: string, args: string[], options: Parameters<typeof spawn>[2]) =>
      spawnMockAcpProcess(trapSigtermScript, binary, args, options)) as unknown as SpawnProcessFn,
  });
  await new Promise((r) => setTimeout(r, 30));
  await client1.dispose(30);
  assert.equal(client1.isAlive, false);
  assert.match(client1.stderrTail, /saw-sigterm/);

  const ignoreSigtermScript = `
    const readline = require("node:readline");
    const rl = readline.createInterface({ input: process.stdin });
    rl.on("line", () => {});
    process.on("SIGTERM", () => {});
    setInterval(() => {}, 1000);
  `;
  const client2 = new AgyAcpClient({
    binary: process.execPath,
    cwd: process.cwd(),
    spawnProcess: ((binary: string, args: string[], options: Parameters<typeof spawn>[2]) =>
      spawnMockAcpProcess(ignoreSigtermScript, binary, args, options)) as unknown as SpawnProcessFn,
  });
  await new Promise((r) => setTimeout(r, 30));
  await client2.dispose(30);
  assert.equal(client2.isAlive, false);
});
