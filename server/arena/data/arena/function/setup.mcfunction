# arena:setup — one-time arena preparation (datapack namespace "arena").
#
# Run ONCE per server boot (or whenever the arena needs re-flattening). It:
#   - sets the world rules that keep the MDP fully-observed and deterministic,
#   - clears any ambient entities,
#   - lays a clean flat play surface around spawn so both bots stand on solid
#     ground and never fall into the void.
#
# The bridge's reset RPC (bridge/bot.js handleReset / T7a) does NOT call this
# every episode — it calls arena:spawn_learner and arena:spawn_dummy. This file
# is the once-per-boot scaffolding. Invoke from the console or from arena:load:
#     /function arena:setup
#
# Coordinates: spawn template is learner [0.5,64,0.5], dummy [3.5,64,0.5]
# (matches bridge/bot.js resetTemplate.position and the +3 X offset for dummy).
# The floor is y=63 (one block below feet at y=64).

# --- World rules: fully-observed, deterministic, no surprises ---
gamerule doMobSpawning false
gamerule doDaylightCycle false
gamerule doWeatherCycle false
gamerule doFireTick false
gamerule mobGriefing false
gamerule keepInventory true
gamerule doImmediateRespawn true
gamerule announceAdvancements false
gamerule showDeathMessages false
gamerule naturalRegeneration true
gamerule randomTickSpeed 0
gamerule spawnRadius 0
gamerule fallDamage false
gamerule drowningDamage false
gamerule fireDamage false
gamerule freezeDamage false

# --- Fixed time + clear weather so observations never vary by daylight ---
time set day
weather clear 1000000

# --- Clear stray entities (anything that is not a player) near the arena ---
# Radius 64 around the learner spawn covers the whole play area.
kill @e[type=!minecraft:player,x=0,y=64,z=0,distance=..64]

# --- Lay a clean flat floor + clear the air above it around both spawns ---
# Floor at y=63, a 24x24 pad centered between the two spawns; air for 8 blocks up.
fill -8 63 -12 16 63 12 minecraft:smooth_stone replace
fill -8 64 -12 16 71 12 minecraft:air replace

# --- Optional bedrock under-floor so nothing can dig/fall through to the void ---
fill -8 62 -12 16 62 12 minecraft:bedrock replace

# --- Set the world spawn to the learner spawn so fresh joins land in-arena ---
setworldspawn 0 64 0

# Audit line in the server console / ops chat.
tellraw @a {"text":"[arena] setup complete: floor laid, gamerules set, entities cleared.","color":"green"}
