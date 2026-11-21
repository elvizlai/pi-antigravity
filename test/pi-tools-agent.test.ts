import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, rmSync, statSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { test } from "node:test";
import { createPiToolsAgent } from "../lib/pi-tools-agent.ts";

test("createPiToolsAgent writes 0600 standard & orchestrator agent files and removes cleanly", async () => {
  const root = mkdtempSync(path.join(os.tmpdir(), "agy-agent-"));
  const agent = createPiToolsAgent(root);

  try {
    assert.equal(agent.isPrepared(), false);
    assert.equal(agent.nameForModel("gemini-4-neon"), agent.name);
    assert.equal(agent.nameForModel("gemini-pro-orchestrator-neon"), agent.orchestratorName);

    await agent.prepare({
      serverName: "pi-deadbeef",
      url: "http://127.0.0.1:43210/mcp",
      token: "secret-token",
    });
    assert.equal(agent.isPrepared(), true);

    const stdFile = path.join(agent.directory, "agent.md");
    const orchFile = path.join(agent.orchestratorDirectory, "agent.md");
    const stdMode = statSync(stdFile).mode & 0o777;
    assert.equal(stdMode, 0o600);

    const stdContent = readFileSync(stdFile, "utf8");
    const orchContent = readFileSync(orchFile, "utf8");
    const parseFrontmatter = (raw: string) => {
      const match = /^---\n([\s\S]*?)\n---\n/.exec(raw);
      assert.ok(match, "expected JSON/YAML frontmatter block");
      return JSON.parse(match[1]!);
    };
    const stdFrontmatter = parseFrontmatter(stdContent);
    const orchFrontmatter = parseFrontmatter(orchContent);

    assert.equal(stdFrontmatter.mcpServers[0].timeoutSeconds, -1);
    assert.equal(stdFrontmatter.mcpServers[0].serverUrl, "http://127.0.0.1:43210/mcp");
    assert.deepEqual(stdFrontmatter.tools, []);
    assert.ok(orchFrontmatter.tools.includes("enter_plan_mode"));
    assert.ok(orchFrontmatter.tools.includes("exit_plan_mode"));

    await agent.remove();
    assert.equal(agent.isPrepared(), false);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});
