# arena:load — runs automatically when the datapack loads (server boot / /reload).
#
# Wired via the minecraft:load function tag
# (data/minecraft/tags/function/load.json). It performs the once-per-boot arena
# scaffolding so the world is ready before the bridge connects its bots.

tellraw @a {"text":"[arena] datapack loaded (T8). Running arena:setup ...","color":"green"}
function arena:setup
