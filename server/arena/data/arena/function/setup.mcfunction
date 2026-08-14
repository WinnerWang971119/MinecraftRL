# arena:setup — one-shot GLOBAL world preparation (datapack namespace "arena").
#
# Takes NO arguments. Run ONCE per server boot. It sets only WORLD-WIDE state:
# gamerules, time, weather, and the world spawn. Per-pad geometry lives in the
# arena:setup_pad macro and is NOT world-wide.
#
# For backward compatibility with the single-arena path (plan AC11) this also
# builds pad 0 at anchor (0, 0), which is exactly what this file did before the
# pad topology existed and is exactly padAnchor(0).
#
# Fleet boot sequence (N pads):
#     /function arena:setup                     # once: gamerules + pad 0
#     /function arena:setup_pad {x:..,z:..}     # once per pad i > 0
#
# Every anchor passed above comes from T10's padAnchor(i), which is the SOLE
# coordinate source. No anchor value and no coordinate formula is reproduced
# anywhere in this datapack, deliberately — they would be a copy of a number
# T10 owns, free to drift.
#
# arena:load invokes this automatically on datapack load / /reload.

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
# naturalRegeneration MUST stay false. With it on, the dummy heals between
# cooled hits, which (a) makes a dummy that cannot die net-positive to farm and
# (b) turns the combat probe's exact per-hit deltas into false negatives on a
# CORRECT implementation. See plan AC8/AC18.
gamerule naturalRegeneration false
gamerule randomTickSpeed 0
gamerule spawnRadius 0
gamerule fallDamage false
gamerule drowningDamage false
gamerule fireDamage false
gamerule freezeDamage false

# --- Fixed time + clear weather so observations never vary by daylight ---
time set day
weather clear 1000000

# --- World spawn = pad 0's anchor, so a fresh join lands in a real arena. ---
#     Per-bot spawnpoints are set every reset by arena:reset_pad; this is only
#     the fallback for a bot that joins before its first reset.
setworldspawn 0 64 0

# --- Build pad 0 (anchor 0,0). Idempotent; safe to re-run. ---
function arena:setup_pad {x:0,z:0}

tellraw @a {"text":"[arena] setup complete: gamerules set (naturalRegeneration OFF), pad 0 built and enclosed.","color":"green"}
