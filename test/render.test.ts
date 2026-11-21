import assert from "node:assert/strict";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import type { Theme } from "@earendil-works/pi-coding-agent";
import {
  AgyWorkingStatusTracker,
  formatAgyCall,
  summarizeAgyResult,
} from "../src/provider/render.ts";

const mockTheme = {
  fg: (role: string, text: string) => (role === "error" ? `ERR:${text}` : text),
  bold: (text: string) => text,
} as unknown as Theme;

test("formatAgyCall renders standard ACP kind, title, locations, and opaque rawInput cleanly", () => {
  const homeFile = path.join(os.homedir(), "work", "app.ts");
  assert.equal(
    formatAgyCall(
      {
        tool: "Read configuration",
        kind: "read",
        locations: [{ path: homeFile, line: 42 }],
      },
      mockTheme,
    ),
    "read Read configuration ~/work/app.ts:42",
  );

  // Skip locations already present in title and suppress redundant generic args when locations exist.
  assert.equal(
    formatAgyCall(
      {
        tool: `Edit ${homeFile}`,
        kind: "edit",
        locations: [{ path: homeFile }],
        input: { path: homeFile },
      },
      mockTheme,
    ),
    `edit Edit ${homeFile}`,
  );

  // When title equals standard kind, do not repeat the label.
  assert.equal(
    formatAgyCall(
      {
        tool: "execute",
        kind: "execute",
        input: { cmd: "npm test", cwd: "/tmp/repo" },
      },
      mockTheme,
    ),
    'execute cmd "npm test", cwd "/tmp/repo"',
  );

  // Arbitrary unknown/opaque inputs never emit invalid-arg errors.
  const other = formatAgyCall(
    {
      tool: "Parallel\n  Review",
      kind: "other",
      input: {
        role: "Auditor",
        count: 2,
        nested: { deep: true },
        items: ["a", "b"],
      },
    },
    mockTheme,
  );
  assert.doesNotMatch(other, /ERR:/);
  assert.equal(other, 'Parallel Review role "Auditor", count 2, nested {…} …');

  // Search results summarize line count without inspecting internal tool names.
  assert.deepEqual(summarizeAgyResult("search", "a.ts:1:foo\nb.ts:2:bar"), { counts: "2 lines" });
  assert.deepEqual(summarizeAgyResult("read", "a.ts\nb.ts"), {});
});

test("AgyWorkingStatusTracker tracks in-flight standard ACP titles until completion or error", () => {
  const tracker = new AgyWorkingStatusTracker();

  assert.equal(
    tracker.onActivity({
      type: "tool_start",
      callId: "c1",
      title: "Run\n  npm test",
      kind: "execute",
    }),
    "Run npm test…",
  );

  assert.equal(
    tracker.onActivity({
      type: "tool_update",
      callId: "c2",
      title: "Inspect src/index.ts",
      kind: "read",
    }),
    "Inspect src/index.ts…",
  );

  assert.equal(
    tracker.onActivity({
      type: "tool_error",
      callId: "c2",
      title: "Inspect src/index.ts",
      message: "failed",
    }),
    "Run npm test…",
  );

  assert.equal(
    tracker.onActivity({
      type: "tool_done",
      callId: "c1",
      title: "Run npm test",
    }),
    undefined,
  );
});
