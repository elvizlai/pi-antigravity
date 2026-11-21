import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import { PassThrough } from "node:stream";
import test from "node:test";
import {
  buildAgyUsageArgs,
  fetchAgyUsage,
  formatAgyUsageReport,
  formatAgyUsageShort,
  formatReset,
  parseAgyUsageJson,
  remainingPercent,
  usageBar,
  usageWindowLabel,
} from "../src/acp/usage.ts";

const USAGE_JSON = `{
  "conversation_id": "",
  "status": "SUCCESS",
  "response": "Gemini Models\\tWeekly Limit Remaining\\t96%\\t2026-11-28T02:49:01Z\\n",
  "duration_seconds": 0,
  "num_turns": 0,
  "usage": { "input_tokens": 0, "output_tokens": 0, "total_tokens": 0 },
  "command": {
    "name": "usage",
    "data": {
      "description": "Within each group, models share a weekly limit and a 5-hour limit.",
      "groups": [
        {
          "name": "Gemini Models",
          "description": "Models within this group: Gemini 4 Neon, Gemini 4 Hydrogen, Orchestrator (Neon)",
          "buckets": [
            {
              "id": "gemini-weekly",
              "name": "Weekly Limit Remaining",
              "window": "weekly",
              "remaining_fraction": 0.9623777270317078,
              "reset_time": "2026-11-28T02:49:01Z"
            },
            {
              "id": "gemini-5h",
              "name": "Five Hour Limit Remaining",
              "window": "5h",
              "remaining_fraction": 0.8795106410980225,
              "reset_time": "2026-11-21T07:49:01Z"
            }
          ]
        },
        {
          "name": "Claude and GPT models",
          "description": "Models within this group: Claude Opus 5.5, Claude Sonnet 5.5, GPT-OSS 120B",
          "buckets": [
            {
              "id": "3p-weekly",
              "name": "Weekly Limit Remaining",
              "window": "weekly",
              "remaining_fraction": 1,
              "reset_time": "1970-01-01T00:00:00Z"
            },
            {
              "id": "3p-5h",
              "name": "Five Hour Limit Remaining",
              "window": "5h",
              "remaining_fraction": 1,
              "reset_time": "1970-01-01T00:00:00Z"
            }
          ]
        }
      ]
    }
  }
}`;

test("buildAgyUsageArgs expands /usage in print mode without disabling slash commands", () => {
  const args = buildAgyUsageArgs(30_000);
  assert.deepEqual(args, [
    "--print",
    "/usage",
    "--output-format",
    "json",
    "--print-timeout",
    "30s",
  ]);
  assert.ok(!args.includes("--disable-slash-commands"));
});

test("parseAgyUsageJson parses groups and unwraps stream-json envelopes", () => {
  const report = parseAgyUsageJson(USAGE_JSON);
  assert.equal(report.groups.length, 2);
  assert.equal(report.groups[0]?.name, "Gemini Models");
  assert.equal(report.groups[0]?.buckets[0]?.id, "gemini-weekly");
  assert.equal(report.groups[0]?.buckets[0]?.remainingFraction, 0.9623777270317078);

  const wrapped = parseAgyUsageJson(
    JSON.stringify({ event: "result", result: JSON.parse(USAGE_JSON) }),
  );
  assert.equal(wrapped.groups.length, 2);

  assert.throws(() => parseAgyUsageJson("not-json"), /not valid JSON/);
  assert.throws(
    () => parseAgyUsageJson(JSON.stringify({ status: "ERROR", error: "not logged in" })),
    /not logged in/,
  );
  assert.throws(
    () => parseAgyUsageJson(JSON.stringify({ status: "SUCCESS", command: { name: "tasks" } })),
    /expected usage command/,
  );
  assert.throws(
    () =>
      parseAgyUsageJson(
        JSON.stringify({ status: "SUCCESS", command: { name: "usage", data: { groups: [] } } }),
      ),
    /no quota groups/,
  );
});

test("remainingPercent, usageBar, usageWindowLabel, and formatReset handle epoch sentinels", () => {
  assert.equal(remainingPercent(0.8795106410980225), 88);
  assert.equal(remainingPercent(1.5), 100);
  assert.equal(remainingPercent(-0.2), 0);
  assert.equal(usageBar(0.5, 4), "[██░░]");

  assert.equal(
    usageWindowLabel({
      id: "gemini-5h",
      name: "Five Hour Limit Remaining",
      window: "5h",
      remainingFraction: 1,
    }),
    "5h limit",
  );
  assert.equal(
    usageWindowLabel({
      id: "gemini-weekly",
      name: "Weekly Limit Remaining",
      window: "weekly",
      remainingFraction: 1,
    }),
    "Weekly limit",
  );

  const now = new Date(2026, 10, 21, 12, 0, 0);
  const laterToday = new Date(2026, 10, 21, 15, 49, 0);
  const nextWeek = new Date(2026, 10, 28, 10, 49, 0);
  assert.equal(formatReset(laterToday.toISOString(), now), "15:49");
  assert.equal(formatReset(nextWeek.toISOString(), now), "10:49 on 28 Nov");
  assert.equal(formatReset("not-a-date", now), undefined);
  assert.equal(formatReset("1970-01-01T00:00:00Z", now), undefined);
});

test("formatAgyUsageReport renders bars, percentages, and omits epoch reset times", () => {
  const report = parseAgyUsageJson(USAGE_JSON);
  const now = new Date("2026-11-21T07:24:01Z");
  const lines = formatAgyUsageReport(report, now).split("\n");
  assert.equal(lines[0], "Gemini Models");
  assert.match(lines[1] ?? "", /^ {2}5h limit:\s+\[█+░*\] 88% left · resets /);
  assert.match(lines[3] ?? "", /^ {2}Weekly limit:\s+\[█+░*\] 96% left · resets /);
  assert.equal(lines[5], "Claude and GPT models");
  assert.match(lines[6] ?? "", /^ {2}5h limit:\s+\[█{20}\] 100% left$/);
  assert.match(lines[8] ?? "", /^ {2}Weekly limit:\s+\[█{20}\] 100% left$/);
});

test("formatAgyUsageShort selects model group and paints warning/error pace", () => {
  const report = parseAgyUsageJson(USAGE_JSON);
  const now = new Date("2026-11-21T07:24:01Z");
  assert.equal(formatAgyUsageShort(report, "gemini-4-hydrogen", now), "25m:88% · 7d:96%");
  assert.equal(formatAgyUsageShort(report, "orchestrator-neon", now), "25m:88% · 7d:96%");
  assert.equal(formatAgyUsageShort(report, "claude-opus-5-5", now), "5h 100% · weekly 100%");
  assert.equal(formatAgyUsageShort(report, "gpt-oss-120b-medium", now), "5h 100% · weekly 100%");

  const paint = (level: string, text: string) => `<${level}>${text}</${level}>`;
  const paced = formatAgyUsageShort(
    {
      groups: [
        {
          name: "Gemini Models",
          buckets: [
            {
              id: "5h",
              name: "5h",
              window: "5h",
              remainingFraction: 0.05,
              resetTime: "2026-11-21T00:05:00Z",
            },
            {
              id: "weekly",
              name: "Weekly",
              window: "weekly",
              remainingFraction: 0.38,
              resetTime: "2026-11-26T00:00:00Z",
            },
          ],
        },
      ],
    },
    "gemini-4-hydrogen",
    new Date("2026-11-21T00:00:00Z"),
    paint,
  );
  assert.equal(paced, "<error>5m:5%</error> · <warning>5d:38%</warning>");
});

test("fetchAgyUsage sets AGY_CLI_DISABLE_AUTO_UPDATE=true, surfaces errors, and honors AbortSignal", async () => {
  let capturedEnv: NodeJS.ProcessEnv | undefined;
  const report = await fetchAgyUsage({
    binary: "agy-test",
    spawnOverride: ((
      _file: string,
      _args: readonly string[],
      options: { env?: NodeJS.ProcessEnv },
    ) => {
      capturedEnv = options.env;
      const child = new EventEmitter() as EventEmitter & {
        stdout: PassThrough;
        stderr: PassThrough;
      };
      child.stdout = new PassThrough();
      child.stderr = new PassThrough();
      queueMicrotask(() => {
        child.stdout.write(USAGE_JSON);
        child.emit("close", 0);
      });
      return child;
    }) as never,
  });
  assert.equal(capturedEnv?.AGY_CLI_DISABLE_AUTO_UPDATE, "true");
  assert.equal(report.groups.length, 2);

  await assert.rejects(
    () =>
      fetchAgyUsage({
        binary: "agy-test",
        spawnOverride: (() => {
          const child = new EventEmitter() as EventEmitter & {
            stdout: PassThrough;
            stderr: PassThrough;
          };
          child.stdout = new PassThrough();
          child.stderr = new PassThrough();
          queueMicrotask(() => {
            child.stderr.write("not logged in\n");
            child.emit("close", 1);
          });
          return child;
        }) as never,
      }),
    /not logged in/,
  );

  await assert.rejects(
    () => fetchAgyUsage({ binary: "agy-test", signal: AbortSignal.abort() }),
    (err: unknown) => err instanceof Error && err.name === "AbortError",
  );
});
