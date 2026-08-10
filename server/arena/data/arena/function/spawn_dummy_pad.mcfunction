# arena:spawn_dummy_pad — MACRO FUNCTION. Reset ONE dummy bot on ONE pad.
#
# ============================================================================
# MACRO ARGUMENT CONTRACT
# ============================================================================
#   /function arena:spawn_dummy_pad {x:<int>,z:<int>,dummy:"<name>",nonce:<int>}
#
#   x, z  : pad ANCHOR. NON-NEGATIVE PLAIN INTEGERS, no NBT type suffix.
#           (See arena:setup_pad's header for the rationale.)
#   dummy : the dummy bot's Minecraft username, e.g. "dummy_bot" (pad 0) or
#           "dummy_3". Must be opped (server/ops.json).
#   nonce : NON-NEGATIVE PLAIN INTEGER, unique per reset. Stamped into the
#           causality beacon on the last line so a beacon that arrives after its
#           own reset gave up cannot satisfy the next one. REQUIRED.
#
#   Normally invoked via arena:reset_pad, which forwards its own arguments here.
#   arena:spawn_dummy is the pad-0 / "dummy_bot" convenience wrapper.
#
# ============================================================================
# RESET TEMPLATE (MUST stay consistent with bridge/bot.js: dummy = spawn.x + 3)
# ============================================================================
#   position : (x+3.5, 64, z+0.5)
#   facing   : -X toward the learner (yaw 90, pitch 0)  <- 90, see the teleport
#   health   : full (20)
#   food     : full (20) + full saturation
#   inventory: empty (a passive target, no weapon)
#   effects  : none active. The instant_health/saturation instances below last
#              ONE GAMETICK (~50 ms) — /effect give's duration is in gameticks,
#              not seconds, for instant_health, instant_damage and saturation
#              (minecraft.wiki, Commands/effect; matches bridge/bot.js:858-860).
#              Knockback immunity is an ATTRIBUTE, not an effect, so it never
#              appears as an active effect at all.
#   spawnpoint: this pad
#
# Two things keep the dummy a clean, stationary MDP target:
#   1. knockback_resistance = 1.0 so hits never shove it off its spawn.
#   2. movement_speed = 0.0 as a belt-and-suspenders anti-drift measure.
#
# ATTRIBUTE IDS: THIS STACK REQUIRES THE `generic.` INFIX. Use
# `minecraft:generic.knockback_resistance` and `minecraft:generic.movement_speed`.
#
# The attribute-id flattening that DROPPED the `generic.` infix landed in
# MINECRAFT 1.21.2. This project is pinned to Paper 1.21.1 build 133
# (server/setup/setup.sh PAPER_VERSION), which is BEFORE that change, so the
# un-prefixed ids do not exist here. VERIFIED TWO WAYS, do not re-flip this from
# memory:
#   1. Live console round-trip against the booted server
#      (server/logs/latest.log, 03:12:20-03:12:24: the un-prefixed probes were
#      followed immediately by `generic.`-prefixed retries, and only the
#      prefixed form was used thereafter at 03:13:02).
#   2. This repo's own boot logs, which failed on it for three consecutive
#      boots (server/logs/2026-08-08-{1,2,3}.log.gz):
#        [ServerMain/ERROR]: Failed to load function arena:spawn_dummy
#        IllegalArgumentException: Whilst parsing command on line 41:
#          Can't find element 'minecraft:knockback_resistance'
#          of type 'minecraft:attribute'
#
# The failure mode is WORSE in a macro function than it was in the old plain
# function. A plain function is parsed at LOAD, so a bad id shows up as the boot
# error above. A macro line is parsed at INSTANTIATION, and a parse failure
# aborts instantiation of the WHOLE function — so a bad id here means NOT ONE
# command in this file runs: no teleport, no heal, no food, no effect clear, no
# knockback immunity, no spawnpoint. It is silent at boot and total at runtime.
# If you ever bump the pinned Paper version to 1.21.2+, these two ids must be
# flattened in the same commit.

# --- Clear inventory: the dummy carries nothing ---
$clear $(dummy)

# --- Teleport to the fixed spawn, facing the learner (-X, yaw 90) -----------
#     THE YAW IS 90, NOT -90. Minecraft's look vector is
#         look.xz = (-sin(yaw), cos(yaw))     [env/perception_filter.py:59-65]
#     so yaw -90 -> (+1, 0) = +X and yaw 90 -> (-1, 0) = -X. The dummy stands at
#     anchor+3.5 and the learner at anchor+0.5 on the SAME z, so dummy ->
#     learner is -X, and the yaw that looks -X solves -sin(yaw) = -1, i.e.
#     yaw = 90 (cos(90) = 0, so the z component is 0 as the shared z requires).
#
#     MEASURED, NOT INFERRED (T22). This line used to read `-90 0` and was
#     annotated "facing the learner (-X, yaw -90)". A server-authoritative read
#     after a reset, both bots at rest, showed:
#         learner_bot Rotation: [90.0f, 0.0f]   Pos: [1024.5d, 64.0d, 0.5d]
#         dummy_bot   Rotation: [-90.0f, 0.0f]  Pos: [1027.5d, 64.0d, 0.5d]
#     Both bots were pointing directly AWAY from each other. The dummy is the
#     worse half: it is stationary and passive, so it never turns, and it faced
#     away for the ENTIRE episode, every episode. Corroborated by the wire yaws
#     (1.570796 rad and 4.712389 rad) and a live walk whose forward/APPROACH leg
#     moved -X. No test catches this — bot.attack(entity) does not require
#     facing, so AC8's combat probe passed throughout.
#     spawn_learner_pad.mcfunction carries the mirrored fix; the two must stay
#     opposite (learner -90, dummy +90) or the bots point the same way.
#
#     Two steps, because the dummy sits at anchor + 3.5 and `$(x)` is a purely
#     textual substitution: there is no macro form that yields "3.5" from "0".
#       step 1: absolute park on the learner cell, using the same safe
#               `$(x).5` concatenation as the learner.
#       step 2: a purely RELATIVE +3 X hop, run from an execution position that
#               is pinned to the SAME point step 1 just parked the dummy on.
#               Whether Java resolves a /teleport `<location>` relative against
#               the TARGET or against the EXECUTION POSITION is genuinely
#               contested, and there is no live server in which to settle it, so
#               the line is deliberately written to be correct under BOTH
#               readings: both origins are (x+0.5, 64, z+0.5), so `~3 ~ ~` lands
#               on (x+3.5, 64, z+0.5) either way. The explicit `.5` literals in
#               `positioned` also sidestep the integer-Vec3 block-centring rule.
#               Rotation is set on this final command so the end state is
#               unambiguous. LIVE CHECK (TC6/TC7): confirm the dummy sits at
#               anchor+3.5 on EVERY pad, not just pad 0.
$tp $(dummy) $(x).5 64 $(z).5
$execute positioned $(x).5 64 $(z).5 run tp $(dummy) ~3 ~ ~ 90 0

# --- Full health, full hunger, no leftover effects --------------------------
#     CLEAR FIRST, THEN GIVE. See spawn_learner_pad for the full rationale: a
#     trailing `effect clear` in the same tick may strip an instant effect
#     before its first tick applies it, which would silently void the heal and
#     the food restore. This ordering is correct by construction.
#     Restoring food/saturation is what makes the dummy's health stationary
#     across episodes (plan AC18): naturalRegeneration is off so health cannot
#     creep, and hunger no longer drifts downward run over run. The dummy has no
#     bridge-side heal backstop, so this file is the ONLY thing standing behind
#     AC18 — the ordering matters more here than anywhere else.
#     The trailing `1` on both gives is ONE GAMETICK, not one second: /effect
#     give's duration is in gameticks for instant_health, instant_damage and
#     saturation (minecraft.wiki, Commands/effect). Nothing lingers for the
#     read-back gate to wait on.
$effect clear $(dummy)
$effect give $(dummy) minecraft:instant_health 1 9 true
$effect give $(dummy) minecraft:saturation 1 19 true

# --- Knockback immunity (attribute) ---
#     Re-applied every reset so a respawn that re-rolls base values cannot
#     silently un-pin the dummy.
#     `generic.` infix REQUIRED on Paper 1.21.1 — see the header block. These
#     are the only two /attribute lines in the whole datapack.
$attribute $(dummy) minecraft:generic.knockback_resistance base set 1.0
$attribute $(dummy) minecraft:generic.movement_speed base set 0.0

# --- Per-bot spawnpoint on THIS pad ----------------------------------------
#     The dummy dies every episode; with doImmediateRespawn and a shared world
#     spawn it would otherwise respawn inside pad 0.
#     `execute as <bot> at @s run spawnpoint @s ~ ~ ~` is deliberately chosen
#     over a composed absolute position: the execution position and the target
#     are the SAME entity, so the command is correct under either reading of how
#     a BlockPos relative resolves. The bot was pinned to its pad two lines
#     above, so this records (x+3, 64, z) — inside this pad, never pad 0.
$execute as $(dummy) at @s run spawnpoint @s ~ ~ ~

$tellraw @a[tag=arena_debug] {"text":"[arena] dummy $(dummy) reset @ anchor $(x),$(z) +3.5X (kb_resist=1.0, idle, spawnpoint pinned).","color":"gold"}

# --- RESET CAUSALITY BEACON. MUST STAY THE LAST LINE OF THIS FILE. ----------
#     See the twin block at the end of spawn_learner_pad.mcfunction for the full
#     rationale. It matters MOST here: the dummy dies every episode, so its
#     post-respawn state (full health, empty inventory, effects cleared by
#     death, at the previously-pinned spawnpoint) is indistinguishable from a
#     correct reset — and the `generic.` attribute ids above are the single
#     likeliest line in this datapack to abort instantiation of this whole
#     function. Addressed to $(dummy) by name; the bridge listens on the DUMMY's
#     own connection for it and matches the text exactly.
$tellraw $(dummy) {"text":"[arena] reset_ok dummy $(x) $(z) $(dummy) $(nonce)"}
