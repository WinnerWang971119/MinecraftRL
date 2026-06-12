# arena:spawn_dummy — reset the stationary dummy bot to its fixed spawn + state.
#
# Called by the bridge's reset RPC (bridge/bot.js handleReset / T7a) every
# episode, alongside arena:spawn_learner. Invoke as one opped command:
#     /function arena:spawn_dummy
#
# The dummy is the Stage-0 target (project spec §8): it must stay PUT and be a
# stable, fully-observed target so the reward/bridge can be validated.
#
# Reset template (MUST stay consistent with bridge/bot.js):
#   position : [3.5, 64, 0.5]   (learner spawn X + 3; matches handleReset's
#              `spawn.x + 3` /tp for the dummy)
#   facing   : toward the learner (-X, i.e. yaw -90 / 270)
#   health   : full (20)
#   inventory: empty (a passive target; no weapon)
#   effects  : none active (resistance applied as an ATTRIBUTE, see below)
#
# Two things keep the dummy a clean MDP target:
#   1. knockback_resistance = 1.0 (attribute) -> the learner's hits never shove
#      it off its spawn, so its position stays observable and constant. This is
#      an attribute base value, NOT an effect, so it does NOT trip the bridge's
#      requireNoEffects read-back check.
#   2. The bedrock under-floor (arena:setup) + no fall damage gamerule give it
#      void/fall immunity without needing a stored effect.

# --- Clear inventory: the dummy carries nothing ---
clear dummy_bot

# --- Teleport to the fixed spawn, facing the learner (-X => yaw -90) ---
tp dummy_bot 3.5 64 0.5 -90 0

# --- Full health + clear leftover effects ---
effect clear dummy_bot
effect give dummy_bot minecraft:instant_health 1 9 true
effect clear dummy_bot

# --- Knockback immunity so the dummy never moves when hit (attribute, 1.21+) ---
# 1.21 FLATTENED the attribute IDs: the "generic." infix was removed.
# Correct 1.21.1 id: minecraft:knockback_resistance  (NOT minecraft:generic.knockback_resistance)
# base 1.0 == full resistance. Applied every reset to be robust to a respawn that re-rolls it.
attribute dummy_bot minecraft:knockback_resistance base set 1.0

# Belt-and-suspenders: also pin movement-speed to 0 so even a server-side AI
# nudge (there is none for a player, but harmless) cannot drift it. Players are
# driven by the bridge, which holds the dummy idle, so this is purely defensive.
# 1.21 flattened: minecraft:movement_speed  (NOT minecraft:generic.movement_speed)
attribute dummy_bot minecraft:movement_speed base set 0.0

tellraw @a[tag=arena_debug] {"text":"[arena] dummy reset @ 3.5 64 0.5 (kb_resist=1.0, idle).","color":"gold"}
