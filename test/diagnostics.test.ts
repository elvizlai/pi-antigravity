import assert from "node:assert/strict";
import { chmodSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { test } from "node:test";
import {
  AgyCompatibilityError,
  checkAgyBinary,
  MIN_AGY_VERSION,
  parseAgyVersion,
  resetAgyBinaryCache,
  withAgyAutoUpdateDisabled,
} from "../lib/diagnostics.ts";
import {
  DEFAULT_AGY_TOOL_MODE,
  formatAgyToolModeStatus,
  parseAgyToolModeArg,
} from "../lib/tool-mode.ts";

test("parseAgyVersion extracts semver across CLI banners", () => {
  assert.equal(parseAgyVersion("antigravity 1.5.13"), "1.5.13");
  assert.equal(parseAgyVersion("v1.5.0-rc.1"), "1.5.0-rc.1");
  assert.equal(parseAgyVersion("unknown"), undefined);
});

test("checkAgyBinary validates minimum version >= 1.5.0", async () => {
  const dir = mkdtempSync(path.join(os.tmpdir(), "agy-diag-"));
  const fakeBin = path.join(dir, "agy");
  writeFileSync(fakeBin, "#!/bin/sh\necho 1.5.13\n");
  chmodSync(fakeBin, 0o755);

  try {
    resetAgyBinaryCache();
    const ok = await checkAgyBinary({
      binaryPath: fakeBin,
      runner: async () => "antigravity 1.5.13",
    });
    assert.equal(ok.version, "1.5.13");
    assert.equal(MIN_AGY_VERSION, "1.5.0");

    resetAgyBinaryCache();
    await assert.rejects(
      checkAgyBinary({
        binaryPath: fakeBin,
        runner: async () => "antigravity 1.4.9",
      }),
      AgyCompatibilityError,
    );
  } finally {
    resetAgyBinaryCache();
    rmSync(dir, { recursive: true, force: true });
  }
});

test("withAgyAutoUpdateDisabled sets AGY_CLI_DISABLE_AUTO_UPDATE=true", () => {
  const env = withAgyAutoUpdateDisabled({ PATH: "/bin" });
  assert.equal(env.AGY_CLI_DISABLE_AUTO_UPDATE, "true");
});

test("tool-mode defaults to pi and formats acp status labels cleanly", () => {
  assert.equal(DEFAULT_AGY_TOOL_MODE, "pi");
  assert.equal(formatAgyToolModeStatus("pi"), "acp:pi");
  assert.equal(formatAgyToolModeStatus("native"), "acp:native");
  assert.deepEqual(parseAgyToolModeArg(""), { kind: "status" });
  assert.deepEqual(parseAgyToolModeArg("on"), { kind: "set", mode: "pi" });
  assert.deepEqual(parseAgyToolModeArg("off"), { kind: "set", mode: "native" });
  assert.deepEqual(parseAgyToolModeArg("native"), { kind: "set", mode: "native" });
  assert.deepEqual(parseAgyToolModeArg("bogus"), { kind: "invalid", raw: "bogus" });
});
