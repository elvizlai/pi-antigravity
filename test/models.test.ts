import assert from "node:assert/strict";
import { test } from "node:test";
import {
  extractAcpConfigOption,
  parseAgyModelsOutput,
  planAgyAcpSecondaryConfigWrites,
  resolveAcpModelCandidate,
  resolveAcpThoughtLevel,
} from "../lib/models.ts";

test("parseAgyModelsOutput deduplicates spinner lines and effort variants cleanly", () => {
  const sample = [
    "\r⠋ Fetching available models...\r⠙ Fetching available models...",
    "gemini-4-neon-high             Gemini 4 Neon (High)",
    "gemini-4-neon-low              Gemini 4 Neon (Low)",
    "gemini-pro-orchestrator-neon   Orchestrator (Neon)",
    "claude-opus-5-5-medium         Claude Opus 5.5 (Medium)",
    "claude-opus-5-5-high           Claude Opus 5.5 (High)",
    "gpt-oss-120b-medium            GPT-OSS 120B (Medium)",
  ].join("\n");

  const parsed = parseAgyModelsOutput(sample);
  assert.deepEqual(
    parsed.map((m) => [m.id, m.name]),
    [
      ["gemini-4-neon", "Gemini 4 Neon"],
      ["gemini-pro-orchestrator-neon", "Orchestrator (Neon)"],
      ["claude-opus-5-5", "Claude Opus 5.5"],
      ["gpt-oss-120b-medium", "GPT-OSS 120B (Medium)"],
    ],
  );
  assert.deepEqual(parsed[0]!.input, ["text", "image"]);
});

test("resolveAcpModelCandidate accepts both base ACP ids and legacy suffixed ids", () => {
  const modelOption = {
    id: "model",
    type: "select",
    options: [
      { value: "gemini-4-neon", name: "Gemini 4 Neon" },
      { value: "gpt-oss-120b-medium", name: "GPT-OSS 120B (Medium)" },
    ],
  };

  assert.deepEqual(resolveAcpModelCandidate("gemini-4-neon", modelOption), {
    modelId: "gemini-4-neon",
  });
  assert.deepEqual(resolveAcpModelCandidate("gemini-4-neon-xhigh", modelOption), {
    modelId: "gemini-4-neon",
    impliedEffort: "xhigh",
  });
  assert.deepEqual(resolveAcpModelCandidate("gpt-oss-120b-medium", modelOption), {
    modelId: "gpt-oss-120b-medium",
  });
});

test("resolveAcpThoughtLevel & planAgyAcpSecondaryConfigWrites respect model capabilities", () => {
  const fullThoughtOption = {
    id: "thought_level",
    type: "select",
    currentValue: "medium",
    options: [
      { value: "low", name: "Low" },
      { value: "medium", name: "Medium" },
      { value: "high", name: "High" },
    ],
  };

  assert.equal(resolveAcpThoughtLevel("minimal", fullThoughtOption), "low");
  assert.equal(resolveAcpThoughtLevel("xhigh", fullThoughtOption), "high");
  assert.equal(resolveAcpThoughtLevel("max", fullThoughtOption), "high");
  assert.equal(resolveAcpThoughtLevel("high", undefined), undefined);

  const configOptions = [
    { id: "model", category: "model", type: "select", currentValue: "gemini-4-neon" },
    fullThoughtOption,
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
  assert.ok(extractAcpConfigOption(configOptions, "thought_level"));
  assert.deepEqual(
    planAgyAcpSecondaryConfigWrites(configOptions, {
      requestedEffort: "max",
      executionMode: "auto",
    }),
    [
      { configId: "thought_level", value: "high" },
      { configId: "mode", value: "auto" },
    ],
  );
});
