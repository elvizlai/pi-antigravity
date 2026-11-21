# @sdrzlyz/pi-antigravity

Clean 4-layer ACP [Pi](https://github.com/earendil-works/pi) provider extension powered directly by `agy acp` (`agy >= 1.5.0`).

Communicates over Antigravity CLI's public stdio JSON-RPC Agent Client Protocol (`agy acp --dangerously-skip-permissions --process-per-session`) and delegates all retries to Pi's native `autoRetry` loop (`settings.retry`, default 3 attempts).

## Architecture

- `src/acp/` — Process-group cleanup, binary verification (`agy >= 1.5.0`), ACP JSON-RPC 2.0 client transport, protocol event normalization, model discovery & config planning, and quota error classification.
- `src/bridge/` — Localhost HTTP MCP server bridging Pi active tools & Pi-private skills (`pi__activate_skill`), plus `--agent` / workspace MCP lifecycle helpers.
- `src/provider/` — Extension settings (`~/.pi/agent/antigravity.json`), session-branch persistence, transcript prompt/system extraction (`getCurrentSystemMessage`), wave turn controller + replay store, runtime & first-response stall timeout, Pi `streamSimple` adapter, and TUI card rendering.
- `index.ts` — Thin Pi extension entrypoint registering provider `"antigravity"` (with `refreshModels` & `streamSimple`), flags, display-only `antigravity` wrapper tool (`terminate: true`), commands, and session hooks.

## Requirements

- Node.js `>= 22.18.0`
- Antigravity CLI (`agy`) `>= 1.5.0`

## Installation

Add the package to `~/.pi/agent/settings.json`, then restart Pi or run `/reload`:

```json
{
  "packages": ["git:github.com/elvizlai/pi-antigravity"]
}
```

Or install from a local checkout:

```json
{
  "packages": ["/Volumes/Data/workspace/elvizlai/pi-antigravity"]
}
```

## Tool Modes & Settings

Extension settings are stored independently in `~/.pi/agent/antigravity.json` (never modifying Pi's `settings.json`):

- **`pi` (default)** — Status: `acp:pi`
  Bridges Pi's active tools (`read`, `bash`, `edit`, `write`, `codemode`, MCP tools) and Pi-private skills (`pi__activate_skill`) into `agy acp` via `--agent pi-tools-<pid>` and a loopback HTTP MCP server.
- **`agy`** — Status: `acp:agy`
  Lets `agy` execute its own built-in tools directly.

## Retries & Timeouts

- **Transient RPC/transport errors**: surface directly to Pi's built-in `autoRetry` (`settings.retry`, default `maxRetries: 3`, exponential backoff, cancellable with `Esc`).
- **First-event watchdog (`60s` default)**: applies **only before the first ACP event** (`session/update` or bridged tool call) and clears immediately once any thinking, text, or tool/subagent activity begins — so subagents and long-running tasks have no time limit. If `agy acp` produces zero initial events for 60 seconds, the runtime sends `session/cancel`, disposes the stalled worker, and stops cleanly without stacking extra 60s retries.
- **Explicit quota exhaustion**: fails fast with `Antigravity quota exceeded` (`0` retries).

## Commands & Flags

- `/agy-tools [pi|agy]` — Switch or inspect tool routing (persists to `~/.pi/agent/antigravity.json`)
- `/agy-plan [on|off]` — Toggle ACP `plan` execution mode vs `agent`
- `/agy-reset` — Reset the bound ACP conversation and start fresh on the next turn
- `/agy-doctor` — Verify `agy >= 1.5.0`, bridge server state, ACP connection, and quota status
- `/agy-models` — Refresh available models from `agy models`

CLI flags: `--agy-tools pi|agy`, `--agy-plan`, `--agy-sandbox`, `--agy-add-dir dir1,dir2`.

## Development

```bash
npm install
npm run check
npm test
```
