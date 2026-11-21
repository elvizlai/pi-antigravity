import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import { createInterface } from "node:readline";
import { PassThrough } from "node:stream";
import { test } from "node:test";
import {
  AgyAcpClient,
  MissingAgyConversationError,
  selectPermissionOptionId,
} from "../lib/acp-client.ts";
import { checkAgyBinary } from "../lib/diagnostics.ts";

interface FakeChildHandle {
  child: any;
  captured: {
    command: string;
    args: readonly string[];
    options: { cwd: string; env: NodeJS.ProcessEnv };
  };
  requests: Array<Record<string, unknown>>;
  sendLine: (payload: unknown) => void;
  exitWith: (code: number) => void;
  setOnClientMessage: (fn: (msg: Record<string, unknown>) => void) => void;
}

function createFakeSpawner() {
  const handles: FakeChildHandle[] = [];

  const spawnProcess = (command: string, args: readonly string[], options: any) => {
    const child = new EventEmitter() as any;
    child.pid = 999_000 + handles.length;
    const stdin = new PassThrough();
    const stdout = new PassThrough();
    const stderr = new PassThrough();
    child.stdin = stdin;
    child.stdout = stdout;
    child.stderr = stderr;

    const requests: Array<Record<string, unknown>> = [];
    let onMessage: ((msg: Record<string, unknown>) => void) | undefined;

    const rl = createInterface({ input: stdin, crlfDelay: Number.POSITIVE_INFINITY });
    rl.on("line", (line) => {
      if (!line.trim()) return;
      const msg = JSON.parse(line) as Record<string, unknown>;
      requests.push(msg);
      onMessage?.(msg);
    });

    const handle: FakeChildHandle = {
      child,
      captured: { command, args, options },
      requests,
      sendLine: (payload) => stdout.write(`${JSON.stringify(payload)}\n`),
      exitWith: (code) => {
        stdout.end();
        stderr.end();
        child.emit("close", code, null);
      },
      setOnClientMessage: (fn) => {
        onMessage = fn;
      },
    };
    handles.push(handle);
    return child;
  };

  return { spawnProcess, handles };
}

test("AgyAcpClient spawns with expected flags, auto-update off, initialize, and session/new", async () => {
  const { spawnProcess, handles } = createFakeSpawner();
  const client = new AgyAcpClient({
    binary: "/usr/local/bin/agy",
    cwd: "/workspace/demo",
    agentName: "pi-tools-1",
    extraDirs: ["/extra/one"],
    sandbox: true,
    spawnProcess,
  });

  const handle = handles[0]!;
  assert.equal(handle.captured.command, "/usr/local/bin/agy");
  assert.deepEqual(handle.captured.args, [
    "acp",
    "--dangerously-skip-permissions",
    "--process-per-session",
    "--agent",
    "pi-tools-1",
    "--add-dir",
    "/extra/one",
    "--sandbox",
  ]);
  assert.equal(handle.captured.options.cwd, "/workspace/demo");
  assert.equal(handle.captured.options.env.AGY_CLI_DISABLE_AUTO_UPDATE, "true");

  handle.setOnClientMessage((msg) => {
    if (msg.method === "initialize") {
      assert.deepEqual((msg.params as any).clientCapabilities, {
        fs: { readTextFile: false, writeTextFile: false },
        terminal: false,
      });
      handle.sendLine({
        jsonrpc: "2.0",
        id: msg.id,
        result: { protocolVersion: 1, agentCapabilities: { loadSession: true } },
      });
    } else if (msg.method === "session/new") {
      assert.deepEqual(msg.params, {
        cwd: "/workspace/demo",
        mcpServers: [],
        _meta: { antigravity: { systemPrompt: { append: "Pi rules" } } },
      });
      handle.sendLine({
        jsonrpc: "2.0",
        id: msg.id,
        result: { sessionId: "conv-123", configOptions: [] },
      });
    }
  });

  const init = await client.initialize();
  assert.equal(init.protocolVersion, 1);
  const sess = await client.newSession({ cwd: "/workspace/demo", systemPromptAppend: "Pi rules" });
  assert.equal(sess.sessionId, "conv-123");

  handle.exitWith(0);
  await client.dispose();
  assert.equal(client.isAlive, false);
});

test("AgyAcpClient.resumeSession closes rogue sessionId and throws MissingAgyConversationError", async () => {
  const { spawnProcess, handles } = createFakeSpawner();
  const client = new AgyAcpClient({
    binary: "agy",
    cwd: "/workspace/demo",
    spawnProcess,
  });
  const handle = handles[0]!;
  const closedIds: string[] = [];

  handle.setOnClientMessage((msg) => {
    if (msg.method === "session/resume") {
      // agy pitfall: silently allocates a brand-new session when requested id is missing
      handle.sendLine({
        jsonrpc: "2.0",
        id: msg.id,
        result: { sessionId: "rogue-new-session", configOptions: [] },
      });
    } else if (msg.method === "session/close") {
      closedIds.push((msg.params as any).sessionId);
      handle.sendLine({ jsonrpc: "2.0", id: msg.id, result: {} });
    }
  });

  await assert.rejects(
    client.resumeSession({ sessionId: "wanted-old-id", cwd: "/workspace/demo" }),
    (err: unknown) => {
      assert.ok(err instanceof MissingAgyConversationError);
      assert.equal(err.requestedSessionId, "wanted-old-id");
      assert.equal(err.actualSessionId, "rogue-new-session");
      return true;
    },
  );
  assert.deepEqual(closedIds, ["rogue-new-session"]);

  handle.setOnClientMessage((msg) => {
    if (msg.method === "session/resume") {
      handle.sendLine({
        jsonrpc: "2.0",
        id: msg.id,
        error: { code: -32001, message: "Session not found" },
      });
    }
  });

  await assert.rejects(
    client.resumeSession({ sessionId: "missing-rpc-id", cwd: "/workspace/demo" }),
    MissingAgyConversationError,
  );

  handle.exitWith(0);
  await client.dispose();
});

test("selectPermissionOptionId and server requests handle allow_always, allow_once, and unknown methods", async () => {
  assert.equal(
    selectPermissionOptionId([
      { optionId: "allow_once", kind: "allow_once" },
      { optionId: "allow_always", kind: "allow_always" },
    ]),
    "allow_always",
  );
  assert.equal(selectPermissionOptionId([{ id: "allow_once", kind: "allow_once" }]), "allow_once");
  assert.equal(selectPermissionOptionId([{ optionId: "deny", kind: "reject" }]), undefined);

  const { spawnProcess, handles } = createFakeSpawner();
  const client = new AgyAcpClient({ binary: "agy", cwd: "/tmp", spawnProcess });
  const handle = handles[0]!;

  const responses: Array<Record<string, unknown>> = [];
  handle.setOnClientMessage((msg) => {
    if (msg.id !== undefined && msg.method === undefined) {
      responses.push(msg);
      if (msg.id === "perm-1") {
        // Complete an initialize after permission handling so we know stdout flush happened
      }
    } else if (msg.method === "initialize") {
      handle.sendLine({
        jsonrpc: "2.0",
        id: "perm-1",
        method: "session/request_permission",
        params: { options: [{ optionId: "allow_once", kind: "allow_once" }] },
      });
      handle.sendLine({
        jsonrpc: "2.0",
        id: "perm-2",
        method: "session/request_permission",
        params: { options: [] },
      });
      handle.sendLine({ jsonrpc: "2.0", id: "unk-3", method: "fs/read_text_file", params: {} });
      setTimeout(() => {
        handle.sendLine({ jsonrpc: "2.0", id: msg.id, result: { protocolVersion: 1 } });
      }, 20);
    }
  });

  await client.initialize();
  assert.deepEqual(responses, [
    {
      jsonrpc: "2.0",
      id: "perm-1",
      result: { outcome: { outcome: "selected", optionId: "allow_once" } },
    },
    { jsonrpc: "2.0", id: "perm-2", result: { outcome: { outcome: "cancelled" } } },
    {
      jsonrpc: "2.0",
      id: "unk-3",
      error: { code: -32601, message: "Unsupported ACP client method: fs/read_text_file" },
    },
  ]);

  handle.exitWith(0);
  await client.dispose();
});

test("AgyAcpClient.prompt cancels via notification (no id) and keeps client alive on clean cancel", async () => {
  const { spawnProcess, handles } = createFakeSpawner();
  const client = new AgyAcpClient({
    binary: "agy",
    cwd: "/tmp",
    cancelGraceMs: 250,
    spawnProcess,
  });
  const handle = handles[0]!;

  let cancelNotification: Record<string, unknown> | undefined;
  handle.setOnClientMessage((msg) => {
    if (msg.method === "session/cancel") {
      cancelNotification = msg;
      const promptReq = handle.requests.find((r) => r.method === "session/prompt");
      handle.sendLine({
        jsonrpc: "2.0",
        id: promptReq!.id,
        result: { stopReason: "cancelled" },
      });
    } else if (msg.method === "session/prompt") {
      handle.sendLine({
        jsonrpc: "2.0",
        method: "session/update",
        params: {
          sessionId: "sess-1",
          update: {
            sessionUpdate: "agent_message_chunk",
            content: { type: "text", text: "partial" },
          },
        },
      });
    }
  });

  const ac = new AbortController();
  const seenUpdates: string[] = [];
  const promptPromise = client.prompt("sess-1", [{ type: "text", text: "hello" }], {
    signal: ac.signal,
    onUpdate: (u) => seenUpdates.push(String(u.sessionUpdate)),
  });

  await new Promise((r) => setTimeout(r, 20));
  ac.abort();

  await assert.rejects(promptPromise, (err: any) => err?.name === "AbortError");
  assert.ok(cancelNotification);
  assert.equal("id" in cancelNotification, false);
  assert.deepEqual(cancelNotification.params, { sessionId: "sess-1" });
  assert.equal(seenUpdates[0], "agent_message_chunk");
  assert.equal(client.isAlive, true);

  handle.exitWith(0);
  await client.dispose();
});

test("Live non-LLM agy acp handshake smoke test (skipped if agy unavailable)", async (t) => {
  let binary: string;
  try {
    const info = await checkAgyBinary();
    binary = info.binary;
  } catch {
    t.skip("agy >= 1.5.0 binary not installed");
    return;
  }

  const client = new AgyAcpClient({
    binary,
    cwd: process.cwd(),
  });
  try {
    const init = await client.initialize();
    assert.equal(init.protocolVersion, 1);
    const session = await client.newSession({ cwd: process.cwd() });
    assert.ok(session.sessionId.length > 0);
    assert.ok(Array.isArray(session.configOptions));
    assert.ok(session.configOptions.some((o) => o.id === "model"));
    await client.closeSession(session.sessionId);
  } finally {
    await client.dispose();
  }
});
