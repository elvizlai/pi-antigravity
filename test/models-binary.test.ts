import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { clampThinkingLevel, getSupportedThinkingLevels } from "@earendil-works/pi-ai";
import {
  AgyCompatibilityError,
  checkAgyBinary,
  parseAgyVersion,
  resetAgyBinaryCache,
} from "../src/acp/binary.ts";
import {
  buildThinkingLevelMap,
  getFallbackModels,
  mergeModelLists,
  modelCacheIsFresh,
  modelsFromCachedEntries,
  parseAgyModelsOutput,
  resolveAcpExecutionMode,
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

test("parseAgyVersion and checkAgyBinary enforce >= 1.5.16 and cache per binary fingerprint", async () => {
  resetAgyBinaryCache();
  assert.equal(parseAgyVersion("agy version v1.5.16"), "1.5.16");
  assert.equal(parseAgyVersion("not-a-version"), undefined);

  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "pi-agy-bin-"));
  const fakeBin = path.join(tmpDir, "agy");
  fs.writeFileSync(fakeBin, "#!/bin/sh\necho ok\n");

  try {
    await assert.rejects(
      () => checkAgyBinary({ binaryPath: fakeBin, runner: async () => "agy 1.5.15" }),
      (err: unknown) => err instanceof AgyCompatibilityError && err.detectedVersion === "1.5.15",
    );

    let runs = 0;
    const runner = async () => {
      runs += 1;
      return "agy 1.5.16";
    };
    const info1 = await checkAgyBinary({ binaryPath: fakeBin, runner });
    const info2 = await checkAgyBinary({ binaryPath: fakeBin, runner });
    assert.equal(info1.version, "1.5.16");
    assert.equal(info2.version, "1.5.16");
    assert.equal(runs, 1);
  } finally {
    fs.rmSync(tmpDir, { recursive: true, force: true });
    resetAgyBinaryCache();
  }
});

test("model parsing, fallback merge, effort stripping, and config planning", () => {
  const parsed = parseAgyModelsOutput(
    "⠋ Fetching available models...\r\ngemini-4-hydrogen  Gemini 4 Hydrogen\nclaude-opus-5-5  Claude Opus 5.5\ngemini-pro-orchestrator-next  Orchestrator (Next)\n",
  );
  assert.equal(parsed.length, 3);
  const parsedById = new Map(parsed.map((m) => [m.id, m]));
  assert.equal(parsedById.get("gemini-4-hydrogen")?.contextWindow, 1_048_576);
  assert.equal(parsedById.get("claude-opus-5-5")?.contextWindow, 1_000_000);
  assert.equal(parsedById.get("gemini-pro-orchestrator-next")?.contextWindow, 1_048_576);
  assert.equal(parsedById.get("gemini-pro-orchestrator-next")?.reasoning, false);
  assert.deepEqual(getSupportedThinkingLevels(parsedById.get("gemini-pro-orchestrator-next")!), [
    "off",
  ]);

  const merged = mergeModelLists(parsed, getFallbackModels());
  assert.ok(merged.some((m) => m.id === "gemini-4-hydrogen"));
  assert.ok(merged.some((m) => m.id === "claude-opus-5-5"));
  const fallbackById = new Map(getFallbackModels().map((m) => [m.id, m]));
  assert.equal(fallbackById.get("gemini-4-neon")?.contextWindow, 1_048_576);
  assert.equal(fallbackById.get("claude-opus-5-5")?.contextWindow, 1_000_000);
  assert.equal(fallbackById.get("claude-sonnet-5-5")?.contextWindow, 1_000_000);
  assert.equal(fallbackById.get("gpt-oss-120b-medium")?.contextWindow, 131_072);
  for (const orchId of [
    "gemini-pro-orchestrator-neon",
    "gemini-pro-orchestrator",
    "gemini-pro-orchestrator-argon",
  ]) {
    const orch = fallbackById.get(orchId);
    assert.ok(orch);
    assert.equal(orch.contextWindow, 1_048_576);
    assert.equal(orch.reasoning, false);
    assert.deepEqual(getSupportedThinkingLevels(orch), ["off"]);
  }

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

  assert.equal(
    resolveAcpThoughtLevel("xhigh", {
      id: "thought_level",
      type: "select",
      options: [
        { value: "low", name: "Low" },
        { value: "xhigh", name: "XHigh" },
        { value: "max", name: "Max" },
      ],
    }),
    "xhigh",
  );
  assert.equal(
    resolveAcpThoughtLevel("max", {
      id: "thought_level",
      type: "select",
      options: [
        { value: "low", name: "Low" },
        { value: "xhigh", name: "XHigh" },
        { value: "max", name: "Max" },
      ],
    }),
    "max",
  );

  const modeOption = {
    id: "mode",
    category: "mode",
    type: "select",
    currentValue: "default",
    options: [
      { value: "default", name: "Default" },
      { value: "plan", name: "Plan" },
    ],
  };
  assert.equal(resolveAcpExecutionMode("plan", modeOption), "plan");
  assert.equal(resolveAcpExecutionMode("default", modeOption), "default");

  assert.equal(modelCacheIsFresh({ fetchedAt: Date.now() }), true);
  assert.equal(modelCacheIsFresh({ fetchedAt: 0 }), false);
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
      formatAgyFooterStatus("pi", "plan", "25m:88% · 7d:96%"),
      "acp:pi plan · 25m:88% · 7d:96%",
    );
    assert.deepEqual(parseAgyToolModeArg("agy"), { kind: "set", mode: "native" });
    assert.equal(parseAgyToolModeFlag("agy"), "native");
  } finally {
    fs.rmSync(agentDir, { recursive: true, force: true });
  }
});

test("thinkingLevelMap unlocks xhigh for neon/hydrogen and preserves live discovered efforts", () => {
  const fallback = getFallbackModels();
  const byId = new Map(fallback.map((m) => [m.id, m]));

  for (const id of ["gemini-4-neon", "gemini-4-hydrogen"]) {
    const model = byId.get(id);
    assert.ok(model);
    assert.deepEqual(getSupportedThinkingLevels(model), ["low", "medium", "high", "xhigh"]);
    assert.equal(clampThinkingLevel(model, "xhigh"), "xhigh");
  }

  for (const id of [
    "gemini-4-carbon",
    "gemini-4-argon",
    "gemini-3.8-flash",
    "claude-opus-5-5",
    "claude-sonnet-5-5",
  ]) {
    const model = byId.get(id);
    assert.ok(model);
    assert.deepEqual(getSupportedThinkingLevels(model), ["low", "medium", "high"]);
    assert.equal(clampThinkingLevel(model, "xhigh"), "high");
  }

  const gptOss = byId.get("gpt-oss-120b-medium");
  assert.ok(gptOss);
  assert.deepEqual(getSupportedThinkingLevels(gptOss), ["off"]);

  assert.equal(buildThinkingLevelMap([]), undefined);
  assert.deepEqual(buildThinkingLevelMap(["low", "xhigh", "max"]), {
    off: null,
    minimal: null,
    low: "low",
    medium: null,
    high: null,
    xhigh: "xhigh",
    max: "max",
  });

  const liveStdout = [
    "gemini-4-neon-low       Gemini 4 Neon (Low)",
    "gemini-4-neon-medium    Gemini 4 Neon (Medium)",
    "gemini-4-neon-high      Gemini 4 Neon (High)",
    "gemini-4-neon-xhigh     Gemini 4 Neon (XHigh)",
    "gemini-4-carbon-low     Gemini 4 Carbon (Low)",
    "gemini-4-carbon-medium  Gemini 4 Carbon (Medium)",
    "gemini-4-carbon-high    Gemini 4 Carbon (High)",
    "gemini-4-carbon-xhigh   Gemini 4 Carbon (XHigh)",
    "gemini-5-nova-low       Gemini 5 Nova (Low)",
    "gemini-5-nova-medium    Gemini 5 Nova (Medium)",
    "gemini-5-nova-high      Gemini 5 Nova (High)",
    "gemini-5-nova-xhigh     Gemini 5 Nova (XHigh)",
    "gpt-oss-120b-medium     GPT-OSS 120B (Medium)",
  ].join("\n");

  const parsed = parseAgyModelsOutput(liveStdout);
  const merged = mergeModelLists(parsed, fallback);
  const mergedById = new Map(merged.map((m) => [m.id, m]));

  assert.deepEqual(getSupportedThinkingLevels(mergedById.get("gemini-4-neon")!), [
    "low",
    "medium",
    "high",
    "xhigh",
  ]);
  assert.deepEqual(getSupportedThinkingLevels(mergedById.get("gemini-4-carbon")!), [
    "low",
    "medium",
    "high",
    "xhigh",
  ]);
  assert.equal(mergedById.get("gemini-5-nova")?.name, "Gemini 5 Nova");
  assert.deepEqual(getSupportedThinkingLevels(mergedById.get("gemini-5-nova")!), [
    "low",
    "medium",
    "high",
    "xhigh",
  ]);
  assert.deepEqual(getSupportedThinkingLevels(mergedById.get("gpt-oss-120b-medium")!), ["off"]);

  const agentDir = fs.mkdtempSync(path.join(os.tmpdir(), "pi-agy-thinking-cache-"));
  try {
    const saved = saveAgySettings(
      {
        cachedModels: [
          {
            id: "gemini-5-nova",
            name: "Gemini 5 Nova",
            thinkingLevelMap: mergedById.get("gemini-5-nova")?.thinkingLevelMap,
          },
          {
            id: "gemini-4-neon",
            name: "Gemini 4 Neon",
          },
        ],
      },
      agentDir,
    );
    const loaded = loadAgySettings(agentDir);
    assert.deepEqual(loaded.cachedModels, saved.cachedModels);

    const hydrated = modelsFromCachedEntries(loaded.cachedModels ?? []);
    assert.deepEqual(getSupportedThinkingLevels(hydrated[0]!), ["low", "medium", "high", "xhigh"]);
    assert.deepEqual(getSupportedThinkingLevels(hydrated[1]!), ["low", "medium", "high", "xhigh"]);
  } finally {
    fs.rmSync(agentDir, { recursive: true, force: true });
  }
});
