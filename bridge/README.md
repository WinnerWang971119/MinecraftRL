# bridge/

Node.js + Mineflayer TCP bridge that connects the Paper Minecraft server to the
Python training stack. Handles bot lifecycle, action macro execution, event
aggregation over `ACTION_REPEAT` ticks, and the `reset` RPC with read-back gate.
Emits and receives newline-delimited JSON over a raw TCP socket.

**Owner workstream:** Environment/bridge track
