# @sdrzlyz/pi-antigravity

Standalone pure-ACP [Pi](https://github.com/earendil-works/pi) provider extension powered directly by `agy acp` (`agy >= 1.5.0`).

Unlike legacy `stream-json` wrappers, this extension communicates strictly over Antigravity CLI's stdio JSON-RPC Agent Client Protocol (`agy acp --dangerously-skip-permissions --process-per-session`), keeping persistent session reuse, native `thought` streaming, multimodal image input, config option negotiation (`model`, `thought_level`, `mode`), and `session/cancel` interruption while preserving full support for bridging Pi tools & skills over HTTP MCP.

## Requirements

- Node.js `>= 22.18.0`
- Antigravity CLI (`agy`) `>= 1.5.0`

## Installation

Point Pi's `~/.pi/agent/settings.json` `packages` array at the GitHub repository (or a local checkout path for development), then restart Pi or run `/reload`:

```json
{
  "packages": ["git:github.com/elvizlai/pi-antigravity"]
}
```

Or use a local checkout:

```json
{
  "packages": ["/Volumes/Data/workspace/elvizlai/pi-antigravity"]
}
```

## Tool Modes

- **`pi` (default)** — Status: `acp:pi`
  Bridges Pi's active tools (`read`, `bash`, `edit`, `write`, `codemode`, user MCP servers like `hindsight`) and Pi-private skills (`pi__activate_skill`) into `agy acp` via `--agent pi-tools-<pid>` and a loopback HTTP MCP bridge.
- **`native`** — Status: `acp:native`
  Lets `agy` execute its own built-in tools directly while optionally exposing Pi's built-in MCP servers & skills via `--add-dir`.

Switch live at any time with:

```text
/agy-tools pi
/agy-tools native
/agy-tools status
```

## Commands & Flags

- `/agy-tools [pi|native|on|off|status]` — Switch or inspect tool routing mode (persists to Pi settings)
- `/agy-plan` — Toggle ACP read-only `plan` execution mode vs `auto`
- `/agy-reset` — Reset the bound ACP conversation and start fresh on the next turn
- `/agy-doctor` — Verify `agy >= 1.5.0`, bridge server state, ACP connection, and quota state
- `/agy-models` — Refresh available models from `agy models`
- `/agy-usage` — Inspect Antigravity quota windows (`/usage` panel)
- `/agy-tasks` — Browse background processes started in the active conversation
- `/agy-artifacts` — Browse generated artifacts in `~/.gemini/antigravity-cli/brain/<id>`
- `/agy-subagents` — List tracked subagent spawns from the ACP stream

CLI flags: `--agy-tools pi|native`, `--agy-plan`, `--agy-sandbox`, `--agy-add-dir dir1,dir2`.

## Development

```bash
npm install
npm run check
npm test
```
