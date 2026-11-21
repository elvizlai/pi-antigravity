import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import {
  createPiToolsAgent,
  createPrivateMcpConfig,
  sweepDeadBridgeArtifacts,
} from "../src/bridge/agent-files.ts";
import { AgyPiBridge, selectBridgedTools } from "../src/bridge/server.ts";
import {
  activateSkillParameters,
  handleActivateSkill,
  piPrivateSkills,
  type SkillLite,
} from "../src/bridge/skills.ts";

test("selectBridgedTools separates pi vs native modes and strips hidden/wrapper tools", () => {
  const tools = [
    { name: "read", description: "Read file", parameters: {} },
    { name: "antigravity", description: "", parameters: {} },
    { name: "secret", description: "Hidden", parameters: {}, exposure: "hidden" },
    {
      name: "hindsight_recall",
      description: "MCP",
      parameters: {},
      sourceInfo: { path: "builtin:mcp" },
    },
  ];
  const active = ["read", "antigravity", "secret", "hindsight_recall"];

  assert.deepEqual(
    selectBridgedTools(tools, active, "pi").map((t) => t.name),
    ["read", "hindsight_recall"],
  );
  assert.deepEqual(
    selectBridgedTools(tools, active, "native").map((t) => t.name),
    ["hindsight_recall"],
  );
});

test("AgyPiBridge enforces token header, exposes tools/list + activate_skill, and resolves tools/call", async () => {
  const bridge = new AgyPiBridge("pi-test");
  bridge.requireToken("secret-token");
  bridge.setToolSource(() => [
    {
      name: "read",
      description: "Read file",
      parameters: { type: "object", properties: { path: { type: "string" } } },
    },
  ]);

  const skillDir = fs.mkdtempSync(path.join(os.tmpdir(), "pi-agy-skill-"));
  const skillFile = path.join(skillDir, "SKILL.md");
  fs.writeFileSync(skillFile, "# Demo Skill\nRun step 1.\n");
  const skills: SkillLite[] = [
    { name: "demo", description: "Demo workflow", filePath: skillFile, baseDir: skillDir },
  ];
  bridge.setDynamicTools([
    {
      name: "activate_skill",
      description: "Activate skill",
      parameters: activateSkillParameters(skills),
      handler: (args) => handleActivateSkill(skills, args),
    },
  ]);
  bridge.refreshTools();

  const seenCalls: Array<{ id: string; tool: string }> = [];
  bridge.setOnCall((call) => {
    seenCalls.push({ id: call.id, tool: call.tool });
    return true;
  });

  await bridge.start();
  try {
    const url = bridge.url;
    assert.ok(url);

    const unauth = await fetch(url, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ jsonrpc: "2.0", id: 1, method: "tools/list" }),
    });
    assert.equal(unauth.status, 403);

    const listRes = await fetch(url, {
      method: "POST",
      headers: { "content-type": "application/json", "x-pi-bridge-token": "secret-token" },
      body: JSON.stringify({ jsonrpc: "2.0", id: 2, method: "tools/list" }),
    });
    const listJson = (await listRes.json()) as { result: { tools: Array<{ name: string }> } };
    assert.deepEqual(listJson.result.tools.map((t) => t.name).sort(), [
      "pi__activate_skill",
      "pi__read",
    ]);

    const skillRes = await fetch(url, {
      method: "POST",
      headers: { "content-type": "application/json", "x-pi-bridge-token": "secret-token" },
      body: JSON.stringify({
        jsonrpc: "2.0",
        id: 3,
        method: "tools/call",
        params: { name: "pi__activate_skill", arguments: { name: "demo" } },
      }),
    });
    const skillJson = (await skillRes.json()) as {
      result: { content: Array<{ text: string }> };
    };
    assert.match(skillJson.result.content[0]?.text ?? "", /# Demo Skill/);

    const callPromise = fetch(url, {
      method: "POST",
      headers: { "content-type": "application/json", "x-pi-bridge-token": "secret-token" },
      body: JSON.stringify({
        jsonrpc: "2.0",
        id: 4,
        method: "tools/call",
        params: { name: "pi__read", arguments: { path: "foo.txt" } },
      }),
    });

    while (seenCalls.length === 0) {
      await new Promise((r) => setTimeout(r, 10));
    }
    assert.deepEqual(bridge.pendingCallIds, [seenCalls[0].id]);
    assert.equal(
      bridge.resolveCall(seenCalls[0].id, {
        content: [{ type: "text", text: "file body" }],
        isError: false,
      }),
      true,
    );

    const callRes = await callPromise;
    const callJson = (await callRes.json()) as { result: { content: Array<{ text: string }> } };
    assert.equal(callJson.result.content[0]?.text, "file body");
  } finally {
    await bridge.close();
    fs.rmSync(skillDir, { recursive: true, force: true });
  }
});

test("pi-tools agent files and dead bridge sweeping avoid private agy caches", async () => {
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "pi-agy-files-"));
  const bridgesRoot = path.join(tmpDir, "bridges");
  const agentsDir = path.join(tmpDir, "agents");

  try {
    const stdAgent = createPiToolsAgent(
      { serverName: "pi-abc", url: "http://127.0.0.1:9999/mcp", token: "tok" },
      { agentsDir, modelId: "gemini-4-hydrogen" },
    );
    await stdAgent.write();
    assert.match(fs.readFileSync(stdAgent.agentFile, "utf-8"), /tools: \[\]/);
    await stdAgent.remove();

    const orchAgent = createPiToolsAgent(
      { serverName: "pi-abc", url: "http://127.0.0.1:9999/mcp", token: "tok" },
      { agentsDir, modelId: "gemini-pro-orchestrator" },
    );
    await orchAgent.write();
    assert.match(fs.readFileSync(orchAgent.agentFile, "utf-8"), /enter_plan_mode/);

    const mcp = createPrivateMcpConfig(bridgesRoot);
    await mcp.write("http://127.0.0.1:9999/mcp", "tok");
    assert.equal(fs.existsSync(path.join(mcp.directory, ".agents", "mcp_config.json")), true);

    const deadBridge = path.join(bridgesRoot, "999999-11111111-1111-1111-1111-111111111111");
    fs.mkdirSync(deadBridge, { recursive: true });
    const deadAgent = path.join(agentsDir, "pi-tools-999999-11111111.md");
    fs.writeFileSync(deadAgent, "stale");

    await sweepDeadBridgeArtifacts({
      bridgesRoot,
      agentsRoot: agentsDir,
      isPidDead: (pid) => pid === 999999,
    });
    assert.equal(fs.existsSync(deadBridge), false);
    assert.equal(fs.existsSync(deadAgent), false);
    assert.equal(fs.existsSync(mcp.directory), true);

    const skills = piPrivateSkills(
      [
        {
          name: "shared",
          description: "shared",
          filePath: path.join(tmpDir, ".agents", "skills", "shared", "SKILL.md"),
          baseDir: path.join(tmpDir, ".agents", "skills", "shared"),
        },
        {
          name: "proj",
          description: "proj",
          filePath: path.join(tmpDir, ".pi", "skills", "proj", "SKILL.md"),
          baseDir: path.join(tmpDir, ".pi", "skills", "proj"),
        },
      ],
      tmpDir,
    );
    assert.deepEqual(
      skills.map((s) => s.name),
      ["proj"],
    );
  } finally {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  }
});
