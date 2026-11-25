import assert from "node:assert/strict";
import test from "node:test";
import type { TranscriptContext } from "@earendil-works/pi-ai";
import {
  buildSeededSessionPrompt,
  buildToolImageContinuationPrompt,
  collectTurnToolResults,
  extractSystemPromptAppend,
  extractTrailingUserPrompt,
  isContinuationAfterToolUse,
  MAX_AGY_PROMPT_BYTES,
  prependSystemContext,
  splitToolResultImages,
} from "../src/provider/prompt.ts";
import {
  AGY_STATE_ENTRY_TYPE,
  findActiveAgyConversation,
  parseAgyStateEntry,
  resolveAgyBranchState,
  resolveInTurnBranchState,
} from "../src/provider/state.ts";

function asTranscript(messages: TranscriptContext["messages"]): TranscriptContext["messages"] {
  return messages;
}

test("extractSystemPromptAppend strips all built-in Pi sections while preserving custom extensions", () => {
  const messages = asTranscript([
    {
      role: "system",
      content: "Base custom note.",
      sections: {
        preamble: "You are a coding assistant running inside pi, a coding agent harness.",
        tools: "Available tools:\n- bash\n- read",
        rules: "<rules>\n- Use bash\n</rules>",
        docs: "<docs>\nPi documentation\n</docs>",
        skills: "<available_skills>foo</available_skills>",
        cwd: "/tmp/work",
        customSection: "# Custom Rule\nFollow strict types.",
      },
      timestamp: 1,
    },
    {
      role: "system",
      content: "",
      sections: {
        hindsight: "Recall enabled.",
      },
      timestamp: 2,
    },
  ]);

  const append = extractSystemPromptAppend(messages);
  assert.equal(
    append,
    "Base custom note.\n\n# Custom Rule\nFollow strict types.\n\nRecall enabled.",
  );
  assert.equal(append?.includes("running inside pi"), false);
  assert.equal(append?.includes("Available tools"), false);
  assert.equal(append?.includes("Pi documentation"), false);
  assert.equal(append?.includes("available_skills"), false);
  assert.equal(append?.includes("/tmp/work"), false);
});

test("extractTrailingUserPrompt merges contiguous user turns, strips system messages, and caps UTF-8 bytes", () => {
  const messages = asTranscript([
    {
      role: "assistant",
      content: [],
      api: "a",
      provider: "antigravity",
      model: "m",
      usage: {} as never,
      stopReason: "stop",
      timestamp: 1,
    },
    { role: "user", content: "<memory>recalled context</memory>", timestamp: 2 },
    { role: "system", content: "mid notification", timestamp: 3 },
    {
      role: "user",
      content: [
        { type: "text", text: "actual user prompt" },
        { type: "image", data: "Zm9v", mimeType: "image/png" },
      ],
      timestamp: 4,
    },
  ]);

  const blocks = extractTrailingUserPrompt(messages, { allowImages: true });
  assert.deepEqual(blocks, [
    { type: "text", text: "<memory>recalled context</memory>\n\nactual user prompt" },
    { type: "image", data: "Zm9v", mimeType: "image/png" },
  ]);

  const textOnly = extractTrailingUserPrompt(messages, { allowImages: false });
  assert.equal(textOnly.length, 1);
  assert.equal(textOnly[0]?.type, "text");

  const huge = asTranscript([
    { role: "user", content: "prefix ".repeat(500), timestamp: 1 },
    { role: "user", content: "final question", timestamp: 2 },
  ]);
  const capped = extractTrailingUserPrompt(huge, { maxBytes: 200 });
  const totalBytes = capped.reduce(
    (sum, b) => sum + (b.type === "text" ? Buffer.byteLength(b.text, "utf8") : 0),
    0,
  );
  assert.ok(totalBytes <= 200);
  const last = capped.at(-1);
  assert.equal(last?.type === "text" && last.text.endsWith("final question"), true);
});

test("collectTurnToolResults gathers trailing matching tool results", () => {
  const results = collectTurnToolResults(
    asTranscript([
      {
        role: "assistant",
        content: [{ type: "toolCall", id: "c1", name: "read", arguments: {} }],
        api: "a",
        provider: "antigravity",
        model: "m",
        usage: {} as never,
        stopReason: "toolUse",
        timestamp: 1,
      },
      {
        role: "toolResult",
        toolCallId: "c1",
        toolName: "read",
        content: [{ type: "text", text: "ok" }],
        isError: false,
        timestamp: 2,
      },
      { role: "system", content: "ignored", timestamp: 3 },
    ]),
    new Set(["c1"]),
  );
  assert.equal(results.length, 1);
  assert.equal(results[0]?.callId, "c1");
});

test("findActiveAgyConversation enforces session ownership and invalidation markers", () => {
  const validMarker = {
    type: "custom",
    customType: AGY_STATE_ENTRY_TYPE,
    data: {
      version: 1,
      kind: "conversation",
      sessionId: "pi-1",
      conversationId: "acp-1",
      cwd: "/work",
      turns: 2,
    },
  };
  const legacyMarker = {
    type: "custom",
    customType: AGY_STATE_ENTRY_TYPE,
    data: {
      version: 1,
      kind: "conversation",
      sessionId: "pi-1",
      conversationId: "acp-legacy",
      cwd: "/work",
      modelId: "gemini-4-hydrogen",
      turns: 3,
      usage: { input_tokens: 10 },
    },
  };

  assert.equal(parseAgyStateEntry({ version: 2 }), undefined);
  assert.equal(findActiveAgyConversation("pi-1", "/work", [validMarker])?.conversationId, "acp-1");
  assert.equal(
    findActiveAgyConversation("pi-1", "/work", [legacyMarker])?.conversationId,
    "acp-legacy",
  );
  assert.equal(
    findActiveAgyConversation("pi-1", "/work", [validMarker, { type: "model_change" }])
      ?.conversationId,
    "acp-1",
  );
  assert.equal(findActiveAgyConversation("pi-fork", "/work", [validMarker]), undefined);
  assert.equal(findActiveAgyConversation("pi-1", "/other", [validMarker]), undefined);
  assert.equal(
    findActiveAgyConversation("pi-1", "/work", [
      validMarker,
      { type: "message", message: { role: "user" } },
    ])?.conversationId,
    "acp-1",
  );
  assert.equal(
    findActiveAgyConversation("pi-1", "/work", [
      validMarker,
      { type: "message", message: { role: "assistant" } },
      { type: "message", message: { role: "user" } },
    ]),
    undefined,
  );
  assert.equal(
    findActiveAgyConversation("pi-1", "/work", [
      validMarker,
      {
        type: "custom",
        customType: AGY_STATE_ENTRY_TYPE,
        data: { version: 1, kind: "reset", sessionId: "pi-1", cwd: "/work" },
      },
    ]),
    undefined,
  );
});

test("isContinuationAfterToolUse distinguishes pure tool continuations from steered user turns", () => {
  const assistantTurn = {
    role: "assistant" as const,
    content: [
      { type: "toolCall" as const, id: "c1", name: "bash", arguments: {} },
      { type: "toolCall" as const, id: "c2", name: "edit", arguments: {} },
    ],
    api: "a",
    provider: "antigravity",
    model: "m",
    usage: {} as never,
    stopReason: "toolUse" as const,
    timestamp: 1,
  };
  const toolResult1 = {
    role: "toolResult" as const,
    toolCallId: "c1",
    toolName: "bash",
    content: [{ type: "text" as const, text: "ok1" }],
    isError: false,
    timestamp: 2,
  };
  const toolResult2 = {
    role: "toolResult" as const,
    toolCallId: "c2",
    toolName: "edit",
    content: [{ type: "text" as const, text: "ok2" }],
    isError: false,
    timestamp: 3,
  };

  assert.equal(
    isContinuationAfterToolUse(asTranscript([{ role: "user", content: "hi", timestamp: 0 }])),
    false,
  );
  assert.equal(isContinuationAfterToolUse(asTranscript([assistantTurn, toolResult1])), true);
  assert.equal(
    isContinuationAfterToolUse(
      asTranscript([assistantTurn, toolResult1, { role: "system", content: "note", timestamp: 4 }]),
    ),
    true,
  );
  assert.equal(
    isContinuationAfterToolUse(
      asTranscript([
        assistantTurn,
        toolResult1,
        { role: "user", content: "参数上还有什么值得优化的吗？", timestamp: 4 },
      ]),
    ),
    false,
  );
  assert.equal(
    isContinuationAfterToolUse(
      asTranscript([
        assistantTurn,
        toolResult1,
        { role: "user", content: "参数上还有什么值得优化的吗？", timestamp: 4 },
        { role: "system", content: "trailing system", timestamp: 5 },
      ]),
    ),
    false,
  );
  assert.equal(
    isContinuationAfterToolUse(
      asTranscript([
        assistantTurn,
        toolResult1,
        {
          role: "user",
          content: [{ type: "image", data: "Zm9v", mimeType: "image/png" }],
          timestamp: 4,
        },
      ]),
    ),
    false,
  );
  assert.equal(
    isContinuationAfterToolUse(
      asTranscript([assistantTurn, toolResult1, { role: "user", content: "   \n ", timestamp: 4 }]),
    ),
    true,
  );

  const steeredTranscript = asTranscript([
    assistantTurn,
    toolResult1,
    toolResult2,
    { role: "system", content: "ignored", timestamp: 4 },
    { role: "user", content: "参数上还有什么值得优化的吗？", timestamp: 5 },
  ]);
  assert.equal(isContinuationAfterToolUse(steeredTranscript), false);
  assert.deepEqual(
    collectTurnToolResults(steeredTranscript, new Set(["c1", "c2"])).map((r) => r.callId),
    ["c1", "c2"],
  );
  assert.deepEqual(extractTrailingUserPrompt(steeredTranscript), [
    { type: "text", text: "参数上还有什么值得优化的吗？" },
  ]);
});

test("splitToolResultImages and buildToolImageContinuationPrompt handle text-only, vision, and steered prompts", () => {
  const textOnly = splitToolResultImages({
    callId: "c-text",
    name: "read",
    content: [{ type: "text", text: "plain" }],
    isError: false,
  });
  assert.equal(textOnly.imageBatch, undefined);
  assert.deepEqual(textOnly.mcpContent, [{ type: "text", text: "plain" }]);

  const vision = splitToolResultImages(
    {
      callId: "c-img",
      name: "read",
      content: [
        { type: "text", text: "Read image file [image/jpeg]" },
        { type: "image", data: "Zm9v", mimeType: "image/jpeg" },
      ],
      isError: false,
    },
    { allowImages: true, label: "s07/ep001/refs/suanian_face2.jpg" },
  );
  assert.equal(
    vision.mcpContent.some((b) => b.type === "image"),
    false,
  );
  assert.equal(vision.mcpContent.length, 2);
  assert.match(
    vision.mcpContent[1]?.type === "text" ? vision.mcpContent[1].text : "",
    /delivered inline in the immediate continuation message/,
  );
  assert.deepEqual(vision.imageBatch, {
    callId: "c-img",
    name: "read",
    label: "s07/ep001/refs/suanian_face2.jpg",
    images: [{ type: "image", data: "Zm9v", mimeType: "image/jpeg" }],
  });

  const omitted = splitToolResultImages(
    {
      callId: "c-img",
      name: "read",
      content: [
        { type: "text", text: "Read image file [image/jpeg]" },
        { type: "image", data: "Zm9v", mimeType: "image/jpeg" },
      ],
      isError: false,
    },
    { allowImages: false },
  );
  assert.equal(omitted.imageBatch, undefined);
  assert.deepEqual(omitted.mcpContent, [
    { type: "text", text: "Read image file [image/jpeg]" },
    { type: "text", text: "(1 image(s) omitted — this agy request is text-only)" },
  ]);

  const continuation = buildToolImageContinuationPrompt(
    [
      vision.imageBatch!,
      {
        callId: "c-img-2",
        name: "read",
        label: "grid.png",
        images: [{ type: "image", data: "YmFy", mimeType: "image/png" }],
      },
    ],
    [
      { type: "text", text: "also check lighting" },
      { type: "image", data: "dXNlcg==", mimeType: "image/webp" },
    ],
  );
  assert.equal(continuation[0]?.type, "text");
  assert.match(
    continuation[0]?.type === "text" ? continuation[0].text : "",
    /1\. read \("s07\/ep001\/refs\/suanian_face2\.jpg"\) — image\/jpeg/,
  );
  assert.match(
    continuation[0]?.type === "text" ? continuation[0].text : "",
    /2\. read \("grid\.png"\) — image\/png/,
  );
  assert.match(continuation[0]?.type === "text" ? continuation[0].text : "", /also check lighting/);
  assert.deepEqual(continuation.slice(1), [
    { type: "image", data: "Zm9v", mimeType: "image/jpeg" },
    { type: "image", data: "YmFy", mimeType: "image/png" },
    { type: "image", data: "dXNlcg==", mimeType: "image/webp" },
  ]);
});

test("resolveAgyBranchState distinguishes none, active, reset, compaction, and tree rewind", () => {
  const markerT1 = {
    type: "custom",
    customType: AGY_STATE_ENTRY_TYPE,
    data: {
      version: 1,
      kind: "conversation",
      sessionId: "pi-1",
      conversationId: "acp-1",
      cwd: "/work",
      turns: 1,
    },
  };
  const markerT3 = {
    type: "custom",
    customType: AGY_STATE_ENTRY_TYPE,
    data: {
      version: 1,
      kind: "conversation",
      sessionId: "pi-1",
      conversationId: "acp-1",
      cwd: "/work",
      turns: 3,
    },
  };

  assert.equal(
    resolveAgyBranchState("pi-1", "/work", [
      { type: "session" },
      { type: "model_change" },
      { type: "message", message: { role: "user" } },
    ]).status,
    "none",
  );
  assert.deepEqual(
    resolveAgyBranchState("pi-1", "/work", [
      markerT1,
      { type: "custom_message" },
      { type: "message", message: { role: "bashExecution" } },
      { type: "message", message: { role: "user" } },
    ]),
    {
      status: "active",
      state: markerT1.data,
    },
  );
  assert.equal(
    resolveAgyBranchState("pi-1", "/work", [
      markerT1,
      {
        type: "custom",
        customType: AGY_STATE_ENTRY_TYPE,
        data: { version: 1, kind: "reset", sessionId: "pi-1", cwd: "/work" },
      },
    ]).status,
    "reset",
  );
  for (const invalidType of ["compaction", "branch_summary", "context_edit"]) {
    assert.equal(
      resolveAgyBranchState("pi-1", "/work", [markerT1, { type: invalidType }]).status,
      "invalidated",
    );
  }
  // Tree rewind: branch ends at turns=1 while allEntries already advanced acp-1 to turns=3.
  assert.equal(
    resolveAgyBranchState("pi-1", "/work", [markerT1], [markerT1, markerT3]).status,
    "invalidated",
  );
  assert.equal(
    findActiveAgyConversation("pi-1", "/work", [markerT1], [markerT1, markerT3]),
    undefined,
  );
});

test("resolveInTurnBranchState allows trailing live-turn messages while enforcing reset, compaction, ownership, and tree rewind", () => {
  const markerT4 = {
    type: "custom",
    customType: AGY_STATE_ENTRY_TYPE,
    data: {
      version: 1,
      kind: "conversation",
      sessionId: "pi-1",
      conversationId: "acp-1",
      cwd: "/work",
      turns: 4,
    },
  };
  const markerT5 = {
    type: "custom",
    customType: AGY_STATE_ENTRY_TYPE,
    data: {
      version: 1,
      kind: "conversation",
      sessionId: "pi-1",
      conversationId: "acp-1",
      cwd: "/work",
      turns: 5,
    },
  };

  // Turn 1 before antigravity-state is persisted at agent_end.
  assert.equal(
    resolveInTurnBranchState("pi-1", "/work", [
      { type: "session" },
      { type: "model_change" },
      { type: "message", message: { role: "user" } },
      { type: "message", message: { role: "assistant" } },
      { type: "message", message: { role: "toolResult" } },
    ]),
    "ok",
  );

  // Turn N mid-turn steering / subagent-notify after antigravity-state(turns=4).
  assert.equal(
    resolveInTurnBranchState("pi-1", "/work", [
      markerT4,
      { type: "custom_message" },
      { type: "message", message: { role: "assistant" } },
      { type: "message", message: { role: "toolResult" } },
      { type: "custom_message" },
    ]),
    "ok",
  );

  // Explicit reset marker.
  assert.equal(
    resolveInTurnBranchState("pi-1", "/work", [
      markerT4,
      {
        type: "custom",
        customType: AGY_STATE_ENTRY_TYPE,
        data: { version: 1, kind: "reset", sessionId: "pi-1", cwd: "/work" },
      },
      { type: "message", message: { role: "user" } },
    ]),
    "reset",
  );

  // Compaction / branch_summary / context_edit, wrong session/cwd, and /tree rewind still invalidate.
  for (const invalidType of ["compaction", "branch_summary", "context_edit"]) {
    assert.equal(
      resolveInTurnBranchState("pi-1", "/work", [
        markerT4,
        { type: invalidType },
        { type: "message", message: { role: "assistant" } },
      ]),
      "invalidated",
    );
  }
  assert.equal(resolveInTurnBranchState("pi-fork", "/work", [markerT4]), "invalidated");
  assert.equal(resolveInTurnBranchState("pi-1", "/other", [markerT4]), "invalidated");
  assert.equal(
    resolveInTurnBranchState(
      "pi-1",
      "/work",
      [markerT4, { type: "message", message: { role: "assistant" } }],
      [markerT4, markerT5],
    ),
    "invalidated",
  );
});

test("prependSystemContext and buildSeededSessionPrompt budget customSystemContext and trailingPrompt within maxBytes", () => {
  assert.equal(MAX_AGY_PROMPT_BYTES, 185_000);

  const prepended = prependSystemContext(
    [
      { type: "text", text: "LIVE_TAIL" },
      { type: "image", data: "Zm9v", mimeType: "image/png" },
    ],
    "SYS_PREFIX",
    64,
  );
  assert.equal(prepended.length, 2);
  assert.equal(prepended[0]?.type, "text");
  assert.equal(prepended[0]?.type === "text" ? prepended[0].text : "", "SYS_PREFIX\n\nLIVE_TAIL");

  const cappedPrepend = prependSystemContext(
    [{ type: "text", text: `${"a".repeat(200)} KEEP_END` }],
    "SYS_TOO_BIG",
    80,
  );
  const cappedPrependText = cappedPrepend[0]?.type === "text" ? cappedPrepend[0].text : "";
  assert.ok(Buffer.byteLength(cappedPrependText, "utf8") <= 80);
  assert.match(cappedPrependText, /KEEP_END$/);

  // Seeded prompt with customSystemContext keeps total bytes <= maxBytes and preserves </conversation_history> + live tail.
  const largeCustomSys = `CUSTOM_SYS_START ${"S".repeat(600)} CUSTOM_SYS_END`;
  const transcriptWithBigHistory = asTranscript([
    { role: "user", content: `OLD_TURN ${"H".repeat(2_000)}`, timestamp: 1 },
    {
      role: "assistant",
      content: [{ type: "text", text: "RECENT_ASSISTANT_CONTEXT" }],
      api: "a",
      provider: "antigravity",
      model: "m",
      usage: {} as never,
      stopReason: "stop",
      timestamp: 2,
    },
    { role: "user", content: "LIVE_STEERING_NOTICE_COMPLETE", timestamp: 3 },
  ]);
  const seededBudgeted = buildSeededSessionPrompt(transcriptWithBigHistory, {
    maxBytes: 1_200,
    customSystemContext: largeCustomSys,
  });
  const seededBudgetedText = seededBudgeted[0]?.type === "text" ? seededBudgeted[0].text : "";
  assert.ok(Buffer.byteLength(seededBudgetedText, "utf8") <= 1_200);
  assert.match(seededBudgetedText, /^CUSTOM_SYS_START/);
  assert.match(seededBudgetedText, /CUSTOM_SYS_END/);
  assert.match(seededBudgetedText, /<\/conversation_history>/);
  assert.match(seededBudgetedText, /RECENT_ASSISTANT_CONTEXT/);
  assert.match(seededBudgetedText, /LIVE_STEERING_NOTICE_COMPLETE$/);
  assert.equal(seededBudgetedText.includes("OLD_TURN"), false);

  // Pure image continuation (no trailing user message) with trailingPrompt + customSystemContext.
  const imageContinuation = buildToolImageContinuationPrompt([
    {
      callId: "c1",
      name: "read",
      label: "shot.png",
      images: [{ type: "image", data: "aW1n", mimeType: "image/png" }],
    },
  ]);
  const imageTranscript = asTranscript([
    { role: "user", content: "Implement feature X and inspect shot.png", timestamp: 1 },
    {
      role: "assistant",
      content: [{ type: "toolCall", id: "c1", name: "read", arguments: { path: "shot.png" } }],
      api: "a",
      provider: "antigravity",
      model: "m",
      usage: {} as never,
      stopReason: "toolUse",
      timestamp: 2,
    },
    {
      role: "toolResult",
      toolCallId: "c1",
      toolName: "read",
      content: [{ type: "text", text: "Read image file [image/png]" }],
      isError: false,
      timestamp: 3,
    },
  ]);
  const seededImg = buildSeededSessionPrompt(imageTranscript, {
    allowImages: true,
    customSystemContext: "SYS_CTX",
    trailingPrompt: imageContinuation,
  });
  assert.equal(seededImg.length, 2);
  const seededImgText = seededImg[0]?.type === "text" ? seededImg[0].text : "";
  assert.match(seededImgText, /^SYS_CTX/);
  assert.match(seededImgText, /Implement feature X and inspect shot\.png/);
  assert.match(seededImgText, /<\/conversation_history>/);
  assert.match(seededImgText, /1\. read \("shot\.png"\) — image\/png$/m);
  assert.deepEqual(seededImg[1], { type: "image", data: "aW1n", mimeType: "image/png" });

  // Tight maxBytes fallback (historyBudget <= 64) still preserves trailingPrompt + image + customSystemContext.
  const tightImg = buildSeededSessionPrompt(imageTranscript, {
    allowImages: true,
    maxBytes: 220,
    customSystemContext: "SYS",
    trailingPrompt: imageContinuation,
  });
  assert.equal(tightImg.length, 2);
  const tightImgText = tightImg[0]?.type === "text" ? tightImg[0].text : "";
  assert.ok(Buffer.byteLength(tightImgText, "utf8") <= 220);
  assert.match(tightImgText, /shot\.png/);
  assert.deepEqual(tightImg[1], { type: "image", data: "aW1n", mimeType: "image/png" });
});

test("buildSeededSessionPrompt preserves single-turn fast path and seeds post-compaction history within byte budget", () => {
  // 1. Single-turn [system, user] matches extractTrailingUserPrompt identically (no wrapper).
  const singleTurn = asTranscript([
    { role: "system", content: "sys", timestamp: 0 },
    { role: "user", content: "Summarize this", timestamp: 1 },
  ]);
  assert.deepEqual(buildSeededSessionPrompt(singleTurn), extractTrailingUserPrompt(singleTurn));

  // 2. Post-compaction [system, compactionSummary, kept turns, new user with image].
  const postCompaction = asTranscript([
    { role: "system", content: "sys", timestamp: 0 },
    {
      role: "user",
      content:
        "The conversation history before this point was compacted into the following summary:\n\n<summary>\nGoal: fix auth\n</summary>",
      timestamp: 1,
    },
    { role: "user", content: "Check login.ts", timestamp: 2 },
    {
      role: "assistant",
      content: [
        { type: "thinking", thinking: "secret internal thought" },
        { type: "text", text: "Reading login.ts now." },
        { type: "toolCall", id: "c1", name: "read", arguments: { path: "login.ts" } },
      ],
      api: "antigravity-acp",
      provider: "antigravity",
      model: "m",
      usage: {} as never,
      stopReason: "toolUse",
      timestamp: 3,
    },
    {
      role: "toolResult",
      toolCallId: "c1",
      toolName: "read",
      content: [{ type: "text", text: "export const ok = true;" }],
      isError: false,
      timestamp: 4,
    },
    {
      role: "user",
      content: [
        { type: "text", text: "Now inspect this screenshot" },
        { type: "image", data: "aW1n", mimeType: "image/png" },
      ],
      timestamp: 5,
    },
  ]);

  const seeded = buildSeededSessionPrompt(postCompaction, { allowImages: true });
  assert.equal(seeded.length, 2);
  const seededText = seeded[0]?.type === "text" ? seeded[0].text : "";
  assert.match(seededText, /<conversation_history>/);
  assert.match(seededText, /Goal: fix auth/);
  assert.match(seededText, /\[User\]:\nCheck login\.ts/);
  assert.match(seededText, /\[Assistant\]:\nReading login\.ts now\./);
  assert.match(seededText, /\[Assistant tool calls\]: read\(\{"path":"login\.ts"\}\)/);
  assert.match(seededText, /\[Tool result \(read\)\]:\nexport const ok = true;/);
  assert.equal(seededText.includes("secret internal thought"), false);
  assert.match(seededText, /Now inspect this screenshot$/);
  assert.deepEqual(seeded[1], { type: "image", data: "aW1n", mimeType: "image/png" });

  // 3. Byte-capped transcript preserves compaction <summary> and live prompt while dropping oldest middle turns.
  const huge = asTranscript([
    {
      role: "user",
      content:
        "The conversation history before this point was compacted into the following summary:\n\n<summary>\nCHECKPOINT_KEEP\n</summary>",
      timestamp: 1,
    },
    { role: "user", content: `OLD_TURN_DROP ${"x".repeat(400)}`, timestamp: 2 },
    { role: "user", content: "NEWEST_KEPT_TURN", timestamp: 3 },
    {
      role: "assistant",
      content: [{ type: "text", text: "LATEST_REPLY" }],
      api: "a",
      provider: "antigravity",
      model: "m",
      usage: {} as never,
      stopReason: "stop",
      timestamp: 4,
    },
    { role: "user", content: "LIVE_USER_QUESTION", timestamp: 5 },
  ]);
  const capped = buildSeededSessionPrompt(huge, { maxBytes: 700 });
  const cappedText = capped[0]?.type === "text" ? capped[0].text : "";
  assert.ok(Buffer.byteLength(cappedText, "utf8") <= 700);
  assert.match(cappedText, /CHECKPOINT_KEEP/);
  assert.match(cappedText, /\[\.\.\.earlier turns omitted\.\.\.\]/);
  assert.match(cappedText, /LATEST_REPLY/);
  assert.match(cappedText, /LIVE_USER_QUESTION/);
  assert.equal(cappedText.includes("OLD_TURN_DROP"), false);
});

test("leading slash commands keep block 0 untouched and move custom context/history to block 1", () => {
  // 1. prependSystemContext keeps /plan and /swarm in block 0, puts customSys in block 1.
  const planBlocks = prependSystemContext(
    [
      { type: "text", text: "  /plan build auth" },
      { type: "image", data: "Zm9v", mimeType: "image/png" },
    ],
    "SYS_CONTEXT",
  );
  assert.deepEqual(planBlocks, [
    { type: "text", text: "  /plan build auth" },
    { type: "text", text: "SYS_CONTEXT" },
    { type: "image", data: "Zm9v", mimeType: "image/png" },
  ]);

  // 2. extractTrailingUserPrompt (used by Turn 1 fallbackLive) applies the same two-block rule.
  const turn1Swarm = extractTrailingUserPrompt(
    asTranscript([{ role: "user", content: "/swarm implement feature X", timestamp: 1 }]),
    { customSystemContext: "HINDSIGHT_MEMORY" },
  );
  assert.deepEqual(turn1Swarm, [
    { type: "text", text: "/swarm implement feature X" },
    { type: "text", text: "HINDSIGHT_MEMORY" },
  ]);

  // 3. Tight maxBytes keeps total UTF-8 bytes <= maxBytes when splitting slash + customSys.
  const tightSlash = prependSystemContext(
    [{ type: "text", text: "/swarm go" }],
    `HEAD_${"x".repeat(200)}_TAIL`,
    48,
  );
  assert.equal(tightSlash.length, 2);
  assert.equal(tightSlash[0]?.type === "text" ? tightSlash[0].text : "", "/swarm go");
  const totalBytes = tightSlash.reduce(
    (sum, b) => sum + (b.type === "text" ? Buffer.byteLength(b.text, "utf8") : 0),
    0,
  );
  assert.ok(totalBytes <= 48);
  assert.match(tightSlash[1]?.type === "text" ? tightSlash[1].text : "", /_TAIL$/);

  // Oversized command text exceeding maxBytes head-truncates cleanly and omits customSys.
  const hugeCmd = prependSystemContext(
    [{ type: "text", text: `/plan ${"你".repeat(40)}` }],
    "SYS",
    40,
  );
  assert.equal(hugeCmd.length, 1);
  const hugeCmdText = hugeCmd[0]?.type === "text" ? hugeCmd[0].text : "";
  assert.ok(Buffer.byteLength(hugeCmdText, "utf8") <= 40);
  assert.match(hugeCmdText, /^\/plan /);

  // 4. buildSeededSessionPrompt with prior history emits [command, customSys + history] for slash prompts.
  const seededSlash = buildSeededSessionPrompt(
    asTranscript([
      { role: "user", content: "earlier question", timestamp: 1 },
      {
        role: "assistant",
        content: [{ type: "text", text: "earlier reply" }],
        api: "a",
        provider: "antigravity",
        model: "m",
        usage: {} as never,
        stopReason: "stop",
        timestamp: 2,
      },
      { role: "user", content: "/plan next phase", timestamp: 3 },
    ]),
    { customSystemContext: "SYS_AFTER_COMPACT" },
  );
  assert.equal(seededSlash.length, 2);
  assert.equal(seededSlash[0]?.type === "text" ? seededSlash[0].text : "", "/plan next phase");
  const ctxText = seededSlash[1]?.type === "text" ? seededSlash[1].text : "";
  assert.match(ctxText, /^SYS_AFTER_COMPACT/);
  assert.match(ctxText, /<conversation_history>/);
  assert.match(ctxText, /earlier reply/);

  // 5. Non-slash prompts remain single-block with prepended customSys.
  const normal = extractTrailingUserPrompt(
    asTranscript([{ role: "user", content: "normal question", timestamp: 1 }]),
    { customSystemContext: "SYS_PREFIX" },
  );
  assert.deepEqual(normal, [{ type: "text", text: "SYS_PREFIX\n\nnormal question" }]);
});
