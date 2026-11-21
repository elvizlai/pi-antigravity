import assert from "node:assert/strict";
import { test } from "node:test";
import { AgyQuotaTurnTracker, isAgyQuotaError, parseAgyQuotaNotice } from "../lib/quota.ts";

test("parseAgyQuotaNotice parses standalone quota banners and ignores normal prose", () => {
  const banner = parseAgyQuotaNotice(
    "\n⚠ Individual quota reached for this model. Resets in 3h 20m\n",
  );
  assert.deepEqual(banner, {
    message: "Individual quota reached for this model. Resets in 3h 20m",
    resetsIn: "3h 20m",
  });

  assert.equal(
    parseAgyQuotaNotice("The error message says Individual quota reached when capacity runs out."),
    undefined,
  );
});

test("isAgyQuotaError matches quota and capacity exhaustion errors", () => {
  assert.equal(isAgyQuotaError("You have exhausted your capacity for Gemini"), true);
  assert.equal(isAgyQuotaError("Individual quota reached"), true);
  assert.equal(isAgyQuotaError("ENOENT: no such file"), false);
});

test("AgyQuotaTurnTracker detects fake end_turn quota failure and rejects false positives", () => {
  const exhausted = new AgyQuotaTurnTracker();
  exhausted.noteText("\n⚠ Individual quota reached. Resets in 45m\n");
  assert.deepEqual(exhausted.detectTurnQuotaExhaustion({ totalTokens: 0, outputTokens: 0 }), {
    message: "Individual quota reached. Resets in 45m",
    resetsIn: "45m",
  });

  const withThought = new AgyQuotaTurnTracker();
  withThought.noteThought();
  withThought.noteText("⚠ Individual quota reached. Resets in 45m");
  assert.equal(withThought.detectTurnQuotaExhaustion(), undefined);

  const withUsage = new AgyQuotaTurnTracker();
  withUsage.noteText("⚠ Individual quota reached. Resets in 45m");
  assert.equal(withUsage.detectTurnQuotaExhaustion({ outputTokens: 12 }), undefined);
});
