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
#   inventory: one iron_sword HELD, plus a full iron set WORN in the four
#              armor slots. NO LONGER EMPTY — the M4 iron loadout (issue #33)
#              arms this bot for the first time. The re-gear block below
#              explains why armor without a sword would have been strictly
#              worse than neither.
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
# PER-OPPONENT MOBILITY OVERRIDE (T11c). BOTH lines above are UNCONDITIONAL
# here on purpose — this file has exactly one behavior, the M2 stationary
# dummy, and stays that way. A scripted opponent
# (OpponentConfig.knockback_immune=False) needs the OPPOSITE of both: an
# opponent that can never be knocked back makes the fight unreal, and one
# pinned to zero movement speed cannot chase, strafe or retreat at all. That
# can't be expressed as a macro key here without breaking BOTH of this
# function's callers (arena:reset_pad and the pad-0 arena:spawn_dummy wrapper),
# which forward only {x,z,dummy,nonce} and would abort — silently,
# whole-function — the moment either omits a new required key. So the toggle is
# NOT in this file: bridge/bot.js's handleReset issues two separate, non-macro
# `/attribute ... base set` overrides (knockback_resistance 0.0, movement_speed
# 0.1), addressed to the dummy's own connection, ONLY when
# dummyKnockbackImmune is false, and ONLY once THIS function's own causality
# beacon (the last line below) confirms it already ran — see bridge/bot.js's
# "PER-OPPONENT MOBILITY OVERRIDE" section. When dummyKnockbackImmune is true
# (the default), the bridge sends nothing at all and these lines are the only
# things that ever set the attributes — byte-identical to before this toggle
# existed.
#
# DO NOT conclude that the movement_speed pin is inert and can be left alone.
# The reasoning that gets there ("Mineflayer's physics ignores the server
# attribute") is FALSE. prismarine-physics/index.js:546-570 DOES consult it and
# scales acceleration by it; it merely MISSES on 1.21.1, because physics looks
# the attribute up under `minecraft:generic.movement_speed` while mineflayer
# stores the wire key verbatim and 1.21.1 decodes that key as
# `generic.movement_speed`, with no namespace. Nor does the server rubber-band
# a client that moves faster than its attribute allows: ServerGamePacketListenerImpl
# in server/versions/1.21.1/paper-1.21.1.jar has ZERO references to Attributes,
# and its "moved too quickly" gate is a fixed 300.0f/100.0f constant pair. So a
# speed-0 opponent walks BY ACCIDENT OF A VERSION STRING — 1.20.6 matched the
# keys and would freeze the bot; 1.21.4 renames the attribute to
# `minecraft:movement_speed` and misses again. A minecraft-data bump in either
# direction silently freezes APPROACH/STRAFE/RETREAT with clean logs, which is
# why the bridge overrides the value instead of relying on the miss.
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

# --- Clear inventory, then re-gear below ------------------------------------
#     The blanket `$clear` stays exactly as it was: it is what makes the
#     re-gear below exact rather than cumulative. It is safe here because
#     this is a bot whose whole inventory this function owns — a human
#     challenger gets a clear SCOPED to the gear instead (deploy/exhibition.py
#     deliberately does not copy this line), because emptying a person's
#     inventory is not "heal and reposition".
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

# --- Re-gear: one iron sword HELD, a full iron set WORN ---------------------
#     THE SWORD IS THE MOST CONSEQUENTIAL LINE IN THIS FILE (M4, issue #33).
#     Until M4 this function handed the opponent a `$clear`, two one-gametick
#     effects, two attributes and a spawnpoint — AND NO WEAPON. It has been
#     punching for 1 damage for the life of the project, which was survivable
#     only because it was also unarmored.
#
#     Armor without a sword is not a milder version of that; it is the run
#     thrown away. The numbers come from the pinned jar, but the conclusion is
#     ARITHMETIC, not a live measurement:
#       - a full iron set is 15 armor points at 0 toughness
#         (net/minecraft/world/item/ArmorMaterials in
#         server/versions/1.21.1/paper-1.21.1.jar: boots 2, leggings 5,
#         chestplate 6, helmet 2; both float arguments to register("iron", ...)
#         are 0.0f, so iron adds neither toughness NOR knockback resistance);
#       - net/minecraft/world/damagesource/CombatRules.getDamageAfterAbsorb is
#         damage * (1 - clamp(armor - damage/(2 + toughness/4), armor/5, 20)/25);
#       - a bare hand does 1, so 1 * (1 - 14.5/25) = 0.42 HP lands per punch:
#         about 48 connected hits to take a fighter from 20 HP to 0, inside an
#         episode capped at 600 steps. Nothing would ever terminate. Every
#         episode a draw, the learner never shown a loss, PFSP flat on a
#         win-rate of 1.0, Elo standing still — a whole training window spent
#         on a fight nobody can lose;
#       - an iron sword does 6 (the figure this repo has used throughout; see
#         deploy/exhibition.py, "a barehanded human is 1 damage against 6"),
#         so 6 * (1 - 12/25) = 3.12 HP a hit: about 7 hits. That is what puts
#         a death back inside the cap.
#
#     ARMOR IS `item replace`, NOT `give`. `/give` fills an INVENTORY slot and
#     equips nothing — four `$give`s would produce a bot CARRYING iron armor
#     at zero armor points, which reads as "armored" in every log line and is
#     naked in the fight. spawn_learner_pad.mcfunction's re-gear note carries
#     the long version, including the itemised check of all four slot names
#     and all five item ids against the pinned jar. Repeat that check before
#     touching any id here: one bad id aborts this ENTIRE function at
#     instantiation, silently, which is the exact hazard the header block
#     above documents at length. `item replace` also overwrites rather than
#     appends, so this block is idempotent across resets and hands out fresh
#     pieces whose durability never accumulates.
#
#     BOTH FIGHTERS MUST CARRY THE SAME LOADOUT. This is a self-play opponent
#     now, not a punching bag: the learner's kit and this one are the same
#     matchup seen from two seats, and any asymmetry here is an asymmetry the
#     policy would learn to exploit and then lose to on demo day.
#
#     PLACED BEFORE THE TWO `$attribute base set` LINES on purpose, so the
#     attribute pins are written last. Iron contributes a 0.0f
#     knockback-resistance modifier (verified above), so `base set 1.0`
#     survives the armor either way — but there is no reason to make that
#     ordering question load-bearing. Nothing in this block is an effect, so
#     it cannot disturb the clear-then-give ordering above either.
$give $(dummy) minecraft:iron_sword 1
$item replace entity $(dummy) armor.head with minecraft:iron_helmet
$item replace entity $(dummy) armor.chest with minecraft:iron_chestplate
$item replace entity $(dummy) armor.legs with minecraft:iron_leggings
$item replace entity $(dummy) armor.feet with minecraft:iron_boots

# --- Knockback immunity (attribute) ---
#     Re-applied every reset so a respawn that re-rolls base values cannot
#     silently un-pin the dummy.
#     `generic.` infix REQUIRED on Paper 1.21.1 — see the header block. These
#     are the only two /attribute lines in the whole datapack.
#     UNCONDITIONAL, always 1.0 / always 0.0, always run — see the
#     "PER-OPPONENT MOBILITY OVERRIDE (T11c)" note near the top of this file
#     for why a scripted opponent's knockback_immune=False is NOT handled here,
#     and why the movement_speed line is NOT inert.
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

# DEBUG LINE — DO NOT READ `kb_resist=1.0` AS THE FINAL VALUE (T11c), AND DO
# NOT READ THE LOADOUT AS CONFIRMATION.
#     Everything after the anchor is a hard-coded literal, printed
#     unconditionally. It is accurate about what THIS function just did, and
#     misleading about what the opponent ends up with: on a scripted-opponent
#     run the bridge's override lands moments later and sets
#     knockback_resistance to 0.0 and movement_speed to 0.1. The same caveat
#     applies to the word "idle".
#     The gear half is the same species of claim: it says the `give` and the
#     four `item replace`s were ISSUED, not that the armor is on the bot. The
#     reset read-back gate cannot see armor either — mineflayer's
#     inventory.items() spans slots 9-44 and the armor slots are 5-8 — so the
#     only thing that proves this loadout is the fail-closed
#     server-authoritative read (T3, AC9). Verify knockback by HITTING the
#     opponent and watching it move, and armor by that read; never from here.
$tellraw @a[tag=arena_debug] {"text":"[arena] dummy $(dummy) reset @ anchor $(x),$(z) +3.5X (iron_sword + full iron armor, kb_resist=1.0, idle, spawnpoint pinned).","color":"gold"}

# --- RESET CAUSALITY BEACON. MUST STAY THE LAST LINE OF THIS FILE. ----------
#     See the twin block at the end of spawn_learner_pad.mcfunction for the full
#     rationale. It matters MOST here: the dummy dies every episode, so its
#     post-respawn state (full health, effects cleared by death, gear intact
#     because keepInventory is on — worn armor included, at the
#     previously-pinned spawnpoint) is indistinguishable from a correct reset — and the `generic.` attribute ids above are the single
#     likeliest line in this datapack to abort instantiation of this whole
#     function. Addressed to $(dummy) by name; the bridge listens on the DUMMY's
#     own connection for it and matches the text exactly.
$tellraw $(dummy) {"text":"[arena] reset_ok dummy $(x) $(z) $(dummy) $(nonce)"}
