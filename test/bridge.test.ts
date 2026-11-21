import assert from "node:assert/strict";
import { test } from "node:test";
import {
  AgyPiBridge,
  BRIDGE_MCP_TIMEOUT_SECONDS,
  type BridgeCall,
  resolveBridgeResultsFromContext,
  selectBridgedTools,
} from "../lib/bridge.ts";

async function postJsonRpc(
  url: string,
  body: unknown,
  headers: Record<string, string> = {},
): Promise<{ status: number; sessionId?: string; json?: any }> {
  const res = await fetch(url, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      ...headers,
    },
    body: JSON.stringify(body),
  });
  const text = await res.text();
  return {
    status: res.status,
    sessionId: res.headers.get("mcp-session-id") ?? undefined,
    json: text ? JSON.parse(text) : undefined,
  };
}

test("BRIDGE_MCP_TIMEOUT_SECONDS is -1 and selectBridgedTools separates pi vs native modes", () => {
  assert.equal(BRIDGE_MCP_TIMEOUT_SECONDS, -1);

  const tools = [
    { name: "bash", description: "Bash", parameters: {} },
    {
      name: "hindsight_recall",
      description: "Memory",
      parameters: {},
      sourceInfo: { path: "builtin:mcp" },
    },
    { name: "antigravity", description: "Wrapper", parameters: {} },
    { name: "secret", description: "Hidden", parameters: {}, exposure: "hidden" },
  ];

  const piMode = selectBridgedTools(
    tools,
    ["bash", "hindsight_recall", "antigravity", "secret"],
    "pi",
  );
  assert.deepEqual(
    piMode.map((t) => t.name),
    ["bash", "hindsight_recall"],
  );

  const nativeMode = selectBridgedTools(
    tools,
    ["bash", "hindsight_recall", "antigravity", "secret"],
    "native",
  );
  assert.deepEqual(
    nativeMode.map((t) => t.name),
    ["hindsight_recall"],
  );
});

test("AgyPiBridge enforces token auth, per-session mcp-session-id, and composite cancellation", async () => {
  const bridge = new AgyPiBridge("pi-test-server", 5_000);
  bridge.requireToken("tok-123");
  bridge.setToolSource(() => [
    { name: "echo", description: "Echo", parameters: { type: "object" } },
  ]);
  bridge.refreshTools();

  const receivedCalls: BridgeCall[] = [];
  bridge.setOnCall((call) => {
    receivedCalls.push(call);
    return true;
  });

  await bridge.start();
  try {
    const url = bridge.url!;

    // Reject unauthenticated POST
    const forbidden = await postJsonRpc(url, { jsonrpc: "2.0", id: 1, method: "initialize" });
    assert.equal(forbidden.status, 403);

    const authHeaders = { "x-pi-bridge-token": "tok-123" };

    // Initialize two independent MCP client sessions
    const initA = await postJsonRpc(
      url,
      { jsonrpc: "2.0", id: 1, method: "initialize" },
      authHeaders,
    );
    const initB = await postJsonRpc(
      url,
      { jsonrpc: "2.0", id: 1, method: "initialize" },
      authHeaders,
    );
    assert.equal(initA.status, 200);
    assert.equal(initB.status, 200);
    assert.ok(initA.sessionId && initB.sessionId && initA.sessionId !== initB.sessionId);

    const listRes = await postJsonRpc(
      url,
      { jsonrpc: "2.0", id: 2, method: "tools/list" },
      { ...authHeaders, "mcp-session-id": initA.sessionId },
    );
    assert.equal(listRes.json.result.tools[0].name, "pi__echo");

    // Start concurrent tools/call requests in session A and session B using identical rpcId = 7
    const callAPromise = postJsonRpc(
      url,
      {
        jsonrpc: "2.0",
        id: 7,
        method: "tools/call",
        params: { name: "pi__echo", arguments: { msg: "from-A" } },
      },
      { ...authHeaders, "mcp-session-id": initA.sessionId },
    );
    const callBPromise = postJsonRpc(
      url,
      {
        jsonrpc: "2.0",
        id: 7,
        method: "tools/call",
        params: { name: "pi__echo", arguments: { msg: "from-B" } },
      },
      { ...authHeaders, "mcp-session-id": initB.sessionId },
    );

    await new Promise((r) => setTimeout(r, 30));
    assert.equal(receivedCalls.length, 2);

    // Cancel rpcId 7 ONLY in session A
    const cancelRes = await postJsonRpc(
      url,
      {
        jsonrpc: "2.0",
        method: "notifications/cancelled",
        params: { requestId: 7 },
      },
      { ...authHeaders, "mcp-session-id": initA.sessionId },
    );
    assert.equal(cancelRes.status, 202);

    const cancelledA = await callAPromise;
    assert.equal(cancelledA.json.result.isError, true);

    // Session B's rpcId 7 must still be awaiting resolution
    const callB = receivedCalls.find((c) => (c.args as any).msg === "from-B")!;
    assert.equal(bridge.isAwaiting(callB.id), true);

    const resolvedCount = resolveBridgeResultsFromContext(bridge, [
      {
        role: "toolResult",
        toolCallId: callB.id,
        toolName: "echo",
        content: [{ type: "text", text: "ok-B" }],
        isError: false,
      },
    ]);
    assert.equal(resolvedCount, 1);

    const completedB = await callBPromise;
    assert.equal(completedB.json.result.isError, false);
    assert.equal(completedB.json.result.content[0].text, "ok-B");
  } finally {
    await bridge.close();
  }
});
