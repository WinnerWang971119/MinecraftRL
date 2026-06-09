# Bridge Message Schema

Human-readable description of every JSON-lines message type exchanged over the
Node↔Python TCP bridge.  The machine-readable version is `schema.json`.

**Owner:** T3 (Environment/bridge track)

> TODO(T3): schema documented by task T3

## Message types (indicative — finalized by T3)

### Python → Node

- `action` — discrete action integer to execute on the learner bot
- `reset` — request a new episode (stops current bots, teleports, regears)

### Node → Python

- `step` — aggregated event bundle for one ACTION_REPEAT window
- `reset_ack` — confirmation that post-reset readback passed all gates

All messages are UTF-8 JSON objects terminated by `\n` (newline-delimited JSON).
