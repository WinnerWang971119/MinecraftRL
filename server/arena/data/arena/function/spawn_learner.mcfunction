# arena:spawn_learner — reset the learner bot to its fixed spawn + gear.
#
# Called by the bridge's reset RPC (bridge/bot.js handleReset / T7a) at the start
# of every episode. The bridge ALSO issues its own /tp + /effect clear + regear
# (see handleReset) and then runs the read-back gate; this function is the
# server-side, opped equivalent that the bridge can invoke as a single command:
#     /function arena:spawn_learner
#
# Reset template (MUST match bridge/bot.js resetTemplate):
#   position : [0.5, 64, 0.5]   (block center; matches DEFAULT resetTemplate)
#   facing   : toward the dummy (+X)
#   health   : full (20)
#   inventory: exactly { iron_sword }   (no leftovers -> read-back gate passes)
#   effects  : none active
#
# Targets the learner by name so it works whether or not the bridge passes an
# @s context. learner_bot is opped in server/ops.json.

# --- Clear inventory so the read-back gate sees EXACTLY the template gear ---
clear learner_bot

# --- Teleport to the fixed spawn, facing the dummy (+X, i.e. yaw 90) ---
tp learner_bot 0.5 64 0.5 90 0

# --- Full health + clear any leftover effects from the previous episode ---
effect clear learner_bot
# Reset health by re-applying instant_health at high amplifier, then clear it so
# no effect lingers (instant_health is applied immediately, not stored).
effect give learner_bot minecraft:instant_health 1 9 true
effect clear learner_bot

# --- Re-gear: exactly one iron sword (matches inventory:['iron_sword']) ---
give learner_bot minecraft:iron_sword 1

# --- Anti-void safety: brief resistance so a 1-tick settle never kills it.
#     Short duration so it is gone before the read-back gate samples effects
#     (the gate requires NO active effects). 1s @ 20 TPS = 20 ticks.
# NOTE: bridge requireNoEffects=true means the gate waits for this to expire; the
#       3s gate timeout (DEFAULT_READBACK.timeoutMs) comfortably covers a 1s buff.
#       If you prefer zero settle effects, delete the next line — the bedrock
#       under-floor from arena:setup already prevents void death.
# effect give learner_bot minecraft:resistance 1 4 true

tellraw @a[tag=arena_debug] {"text":"[arena] learner reset @ 0.5 64 0.5 (iron_sword).","color":"aqua"}
