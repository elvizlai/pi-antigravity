import assert from "node:assert/strict";
import test from "node:test";
import type { Theme } from "@earendil-works/pi-coding-agent";
import { AgyWorkingStatusTracker, formatAgyCall } from "../src/provider/render.ts";

const mockTheme = {
  fg: (role: string, text: string) => (role === "error" ? `ERR:${text}` : text),
  bold: (text: string) => text,
} as unknown as Theme;

test("formatAgyCall formats multi-agent orchestration and file tools cleanly without invalid-arg errors", () => {
  const invoke = formatAgyCall(
    "invoke_subagent",
    {
      TypeName: "research",
      Role: "Quick Probe",
      Model: "flash_lite",
      Prompt: "Reply with OK",
    },
    mockTheme,
  );
  assert.doesNotMatch(invoke, /ERR:/);
  assert.equal(invoke, 'subagent research "Quick Probe" (flash_lite)');

  const invokeInherit = formatAgyCall(
    "invoke_subagent",
    {
      TypeName: "self",
      Model: "inherit",
      Prompt: "Inspect\nmultiple\nlines",
    },
    mockTheme,
  );
  assert.equal(invokeInherit, 'subagent self "Inspect multiple lines"');

  assert.equal(
    formatAgyCall(
      "define_subagent",
      { Name: "auditor", Description: "Audit security\nboundaries" },
      mockTheme,
    ),
    'define_subagent auditor "Audit security boundaries"',
  );

  assert.equal(
    formatAgyCall("manage_subagents", { Action: "kill", ConversationIds: ["c1", "c2"] }, mockTheme),
    "manage_subagents kill (2)",
  );
  assert.equal(
    formatAgyCall("manage_subagents", { Action: "list" }, mockTheme),
    "manage_subagents list",
  );

  const msg = formatAgyCall(
    "send_message",
    { Recipient: "conv-123", Message: "Line 1\nLine 2" },
    mockTheme,
  );
  assert.doesNotMatch(msg, /ERR:/);
  assert.equal(msg, 'send_message conv-123 "Line 1 Line 2"');

  assert.equal(formatAgyCall("swarm", { action: "status" }, mockTheme), "swarm status");
  assert.equal(
    formatAgyCall("swarm", { action: "title", title: "Review Changes" }, mockTheme),
    'swarm title "Review Changes"',
  );
  assert.equal(
    formatAgyCall(
      "swarm",
      { action: "add", name: "reviewer", n: 2, model: "gemini-3.8-flash", brief: "Check\nPR diff" },
      mockTheme,
    ),
    'swarm add reviewer×2 "Check PR diff" (gemini-3.8-flash)',
  );
  assert.equal(
    formatAgyCall("swarm", { action: "add", name: "shard", tasks: ["a", "b", "c"] }, mockTheme),
    "swarm add shard×3",
  );
  assert.equal(
    formatAgyCall("swarm", { action: "kill", name: "reviewer" }, mockTheme),
    "swarm kill reviewer",
  );

  assert.equal(
    formatAgyCall("run_workflow", { Description: "Fan out\nmodule checks" }, mockTheme),
    'run_workflow "Fan out module checks"',
  );
  assert.equal(
    formatAgyCall("run_workflow", { ScriptPath: "/tmp/scan.py" }, mockTheme),
    "run_workflow /tmp/scan.py",
  );
  assert.equal(
    formatAgyCall("run_workflow", { Script: "print(1)" }, mockTheme),
    "run_workflow (inline script)",
  );

  assert.equal(
    formatAgyCall("schedule", { DurationSeconds: 30, Prompt: "Check status" }, mockTheme),
    'schedule 30s "Check status"',
  );
  assert.equal(
    formatAgyCall("schedule", { CronExpression: "*/5 * * * *", Prompt: "Poll health" }, mockTheme),
    'schedule */5 * * * * "Poll health"',
  );

  assert.equal(
    formatAgyCall("write_file", { Path: "/tmp/out.txt" }, mockTheme),
    "write /tmp/out.txt",
  );
  assert.equal(formatAgyCall("edit_file", { Path: "/tmp/in.txt" }, mockTheme), "edit /tmp/in.txt");
});

test("AgyWorkingStatusTracker tracks foreground tools and persistent subagent/swarm wait phases", () => {
  const tracker = new AgyWorkingStatusTracker();

  assert.equal(
    tracker.onActivity({
      type: "tool_start",
      callId: "c1",
      name: "run_command",
      args: { CommandLine: "npm test" },
    }),
    "Running bash: npm test…",
  );
  assert.equal(
    tracker.onActivity({
      type: "tool_done",
      callId: "c1",
      name: "run_command",
      args: { CommandLine: "npm test" },
    }),
    undefined,
  );

  assert.equal(
    tracker.onActivity({
      type: "tool_start",
      callId: "c2",
      name: "invoke_subagent",
      args: { TypeName: "research", Role: "Quick Probe" },
    }),
    "Starting subagent: Quick Probe…",
  );
  assert.equal(
    tracker.onActivity({
      type: "tool_done",
      callId: "c2",
      name: "invoke_subagent",
      args: { TypeName: "research", Role: "Quick Probe" },
    }),
    "Subagent running: Quick Probe…",
  );

  assert.equal(
    tracker.onActivity({
      type: "tool_start",
      callId: "c3",
      name: "swarm",
      args: { action: "add", name: "reviewer", n: 2 },
    }),
    "Updating swarm (add)…",
  );
  assert.equal(
    tracker.onActivity({
      type: "tool_done",
      callId: "c3",
      name: "swarm",
      args: { action: "add", name: "reviewer", n: 2 },
    }),
    "Swarm running: reviewer×2…",
  );

  assert.equal(
    tracker.onActivity({
      type: "tool_done",
      callId: "c4",
      name: "swarm",
      args: { action: "kill", name: "reviewer" },
    }),
    "Subagent running: Quick Probe…",
  );

  assert.equal(
    tracker.onActivity({
      type: "tool_done",
      callId: "c5",
      name: "manage_subagents",
      args: { Action: "kill_all" },
    }),
    undefined,
  );
});
