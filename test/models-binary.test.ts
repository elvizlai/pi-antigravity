import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import {
  AgyCompatibilityError,
  checkAgyBinary,
  parseAgyVersion,
  resetAgyBinaryCache,
} from "../src/acp/binary.ts";
import {
  getFallbackModels,
  mergeModelLists,
  modelCacheIsFresh,
  modelsFromCachedEntries,
  parseAgyModelsOutput,
  planAgyAcpSecondaryConfigWrites,
  resolveAcpModelCandidate,
  resolveAcpThoughtLevel,
} from "../src/acp/models.ts";
import {
  agySettingsPath,
  formatAgyFooterStatus,
  formatAgyToolModeStatus,
  loadAgySettings,
  parseAgyToolModeArg,
  parseAgyToolModeFlag,
  saveAgySettings,
} from "../src/provider/settings.ts";

test("parseAgyVersion and checkAgyBinary enforce >= 1.5.0 and cache per binary fingerprint", async () => {
  resetAgyBinaryCache();
  assert.equal(parseAgyVersion("agy version v1.5.13"), "1.5.13");
  assert.equal(parseAgyVersion("not-a-version"), undefined);

  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "pi-agy-bin-"));
  const fakeBin = path.join(tmpDir, "agy");
  fs.writeFileSync(fakeBin, "#!/bin/sh\necho ok\n");

  try {
    await assert.rejects(
      () => checkAgyBinary({ binaryPath: fakeBin, runner: async () => "agy 1.4.9" }),
      (err: unknown) => err instanceof AgyCompatibilityError && err.detectedVersion === "1.4.9",
    );

    let runs = 0;
    const runner = async () => {
      runs += 1;
      return "agy 1.5.13";
    };
    const info1 = await checkAgyBinary({ binaryPath: fakeBin, runner });
    const info2 = await checkAgyBinary({ binaryPath: fakeBin, runner });
    assert.equal(info1.version, "1.5.13");
    assert.equal(info2.version, "1.5.13");
    assert.equal(runs, 1);
  } finally {
    fs.rmSync(tmpDir, { recursive: true, force: true });
    resetAgyBinaryCache();
  }
});

test("model parsing, fallback merge, effort stripping, and config planning", () => {
  const parsed = parseAgyModelsOutput(
    "⠋ Fetching available models...\r\ngemini-4-hydrogen  Gemini 4 Hydrogen\nclaude-opus-5-5  Claude Opus 5.5\n",
  );
  assert.equal(parsed.length, 2);

  const merged = mergeModelLists(parsed, getFallbackModels());
  assert.ok(merged.some((m) => m.id === "gemini-4-hydrogen"));
  assert.ok(merged.some((m) => m.id === "claude-opus-5-5"));
  assert.ok(mergeModelLists([], getFallbackModels()).some((m) => m.id === "gpt-oss-120b-medium"));

  const candidate = resolveAcpModelCandidate("gemini-3.1-pro-high", {
    id: "model",
    type: "select",
    options: [{ value: "gemini-3.1-pro", name: "Gemini 3.1 Pro" }],
  });
  assert.deepEqual(candidate, { modelId: "gemini-3.1-pro", impliedEffort: "high" });

  const resolvedEffort = resolveAcpThoughtLevel("max", {
    id: "thought_level",
    type: "select",
    options: [
      { value: "low", name: "Low" },
      { value: "high", name: "High" },
    ],
  });
  assert.equal(resolvedEffort, "high");

  const writes = planAgyAcpSecondaryConfigWrites(
    [
      {
        id: "thought_level",
        category: "thought_level",
        type: "select",
        currentValue: "low",
        options: [
          { value: "low", name: "Low" },
          { value: "high", name: "High" },
        ],
      },
      {
        id: "mode",
        category: "mode",
        type: "select",
        currentValue: "agent",
        options: [
          { value: "agent", name: "Agent" },
          { value: "plan", name: "Plan" },
        ],
      },
    ],
    { requestedEffort: "high", executionMode: "plan" },
  );
  assert.deepEqual(writes, [
    { configId: "thought_level", value: "high" },
    { configId: "mode", value: "plan" },
  ]);

  assert.equal(modelCacheIsFresh({ fetchedAt: Date.now(), source: "live" }), true);
  assert.equal(modelCacheIsFresh({ fetchedAt: 0, source: "live" }), false);
});

test("extension settings persist atomically to antigravity.json without touching Pi settings.json", () => {
  const agentDir = fs.mkdtempSync(path.join(os.tmpdir(), "pi-agy-settings-"));
  const piSettings = path.join(agentDir, "settings.json");
  fs.writeFileSync(piSettings, JSON.stringify({ theme: "dark" }));

  try {
    assert.deepEqual(loadAgySettings(agentDir), {});
    const saved = saveAgySettings(
      {
        toolMode: "native",
        firstResponseTimeoutMs: 25_000,
        cachedModels: [{ id: "gemini-4-hydrogen", name: "Gemini 4 Hydrogen" }],
        modelsFetchedAt: 1_700_000_000_000,
      },
      agentDir,
    );
    assert.deepEqual(saved, {
      toolMode: "native",
      firstResponseTimeoutMs: 25_000,
      cachedModels: [{ id: "gemini-4-hydrogen", name: "Gemini 4 Hydrogen" }],
      modelsFetchedAt: 1_700_000_000_000,
    });
    assert.deepEqual(loadAgySettings(agentDir), saved);
    assert.equal(modelsFromCachedEntries(saved.cachedModels ?? []).length, 1);
    assert.equal(fs.existsSync(agySettingsPath(agentDir)), true);
    assert.equal(fs.readFileSync(piSettings, "utf-8"), JSON.stringify({ theme: "dark" }));
    assert.equal(formatAgyToolModeStatus("pi"), "acp:pi");
    assert.equal(formatAgyToolModeStatus("native"), "acp:agy");
    assert.equal(formatAgyFooterStatus("pi"), "acp:pi");
    assert.equal(formatAgyFooterStatus("native"), "acp:agy");
    assert.equal(
      formatAgyFooterStatus("pi", "plan", { usedTokens: 120, sizeTokens: 1000 }),
      "acp:pi plan 12%",
    );
    assert.deepEqual(parseAgyToolModeArg("agy"), { kind: "set", mode: "native" });
    assert.equal(parseAgyToolModeFlag("agy"), "native");
  } finally {
    fs.rmSync(agentDir, { recursive: true, force: true });
  }
});
