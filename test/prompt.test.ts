import assert from "node:assert/strict";
import { test } from "node:test";
import {
  AGY_MAX_INPUT_BYTES,
  assembleBoundedAgyPrompt,
  boundRestoredPiTranscript,
  restoredPiContextPrompt,
  stripPiToolInventory,
  utf8Tail,
} from "../lib/prompt.ts";

test("utf8Tail keeps valid UTF-8 codepoint boundaries", () => {
  const emoji = "🚀".repeat(10); // 4 bytes each = 40 bytes
  const tail = utf8Tail(emoji, 10);
  assert.ok(Buffer.byteLength(tail, "utf8") <= 10);
  assert.equal(tail, "🚀🚀");
});

test("assembleBoundedAgyPrompt preserves user request while trimming historyBootstrap first", () => {
  const hugeHistory = restoredPiContextPrompt("Line of prior chat history.\n".repeat(12_000));
  assert.ok(Buffer.byteLength(hugeHistory, "utf8") > AGY_MAX_INPUT_BYTES);

  const bounded = assembleBoundedAgyPrompt(
    {
      historyBootstrap: hugeHistory,
      prompt: "CRITICAL USER REQUEST: do not drop this",
    },
    AGY_MAX_INPUT_BYTES,
  );

  assert.ok(Buffer.byteLength(bounded, "utf8") <= AGY_MAX_INPUT_BYTES);
  assert.ok(bounded.endsWith("CRITICAL USER REQUEST: do not drop this"));
  assert.ok(bounded.includes("Restored pi conversation context"));
});

test("boundRestoredPiTranscript omits history when budget is too small", () => {
  assert.equal(boundRestoredPiTranscript("hello world", 5), undefined);
});

test("stripPiToolInventory removes Pi tool list boilerplate", () => {
  const raw = [
    "Available tools:",
    "- read: Read file contents",
    "- bash: Execute bash commands",
    "",
    "In addition to the tools above, you have access to:",
    "- skills",
    "",
    "Guidelines:",
    "- Keep code clean.",
  ].join("\n");

  const stripped = stripPiToolInventory(raw);
  assert.equal(stripped.includes("Read file contents"), false);
  assert.ok(stripped.includes("Keep code clean."));
});
