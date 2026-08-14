# arena:load — runs automatically when the datapack loads (server boot / /reload).
#
# Wired via the minecraft:load function tag
# (data/minecraft/tags/function/load.json). It performs the once-per-boot GLOBAL
# scaffolding plus pad 0, so the single-arena world is ready before the bridge
# connects its bots.
#
# Additional pads (i > 0) are NOT built here — the launcher/bridge calls
#     /function arena:setup_pad {x:<int>,z:<int>}
# once per pad with anchors from T10's padAnchor(i).

tellraw @a {"text":"[arena] datapack loaded. Running arena:setup ...","color":"green"}
function arena:setup
