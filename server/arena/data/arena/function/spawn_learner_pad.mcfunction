# arena:spawn_learner_pad — MACRO FUNCTION. Reset ONE learner bot on ONE pad.
#
# ============================================================================
# MACRO ARGUMENT CONTRACT
# ============================================================================
#   /function arena:spawn_learner_pad {x:<int>,z:<int>,learner:"<name>"}
#
#   x, z    : pad ANCHOR. NON-NEGATIVE PLAIN INTEGERS, no NBT type suffix.
#             (See arena:setup_pad's header for the full rationale — `$(x)` is a
#             textual substitution and this file builds `$(x).5`, so `x:-340`
#             would silently yield -340.5, i.e. anchor MINUS half a block, and
#             `x:340L` would yield the non-coordinate `340L.5`. These numbers
#             illustrate literal syntax only; they are not anchors.)
#   learner : the learner bot's Minecraft username, e.g. "learner_bot" (pad 0)
#             or "learner_3". Must be opped (server/ops.json).
#
#   Normally invoked via arena:reset_pad, which forwards its own arguments here.
#   arena:spawn_learner is the pad-0 / "learner_bot" convenience wrapper.
#
# ============================================================================
# RESET TEMPLATE (MUST match bridge/bot.js resetTemplate, anchored)
# ============================================================================
#   position : (x+0.5, 64, z+0.5)   block centre, one block above the y=63 floor
#   facing   : +X toward the dummy  (yaw 90, pitch 0)
#   health   : full (20)
#   food     : full (20) + full saturation
#   inventory: exactly { iron_sword }
#   effects  : none active. The instant_health/saturation instances granted
#              below last ONE GAMETICK (~50 ms) — see the ordering note further
#              down for why the `1` is ticks and not seconds — so they are gone
#              long before a requireNoEffects read-back gate can sample them.
#              Do not "fix" the ordering by appending a trailing `effect clear`
#              — that is the bug the ordering removes.
#   spawnpoint: this pad
#
# At anchor (0,0) with learner="learner_bot" the /tp line expands to the
# byte-identical text of the pre-macro arena:spawn_learner:
#   `tp learner_bot 0.5 64 0.5 90 0`  ->  AC11 (same coordinates and usernames).
# ONLY that one line is byte-identical. Do NOT read this as a claim about the
# whole sequence — it differs from the pre-macro arena:spawn_learner in five
# ways, all deliberate:
#   1. no trailing `effect clear` (see the ordering note below);
#   2. an added saturation restore (food/hunger stationarity, plan AC18);
#   3. an added per-bot /spawnpoint (cross-pad respawn contamination);
#   4. a different tellraw payload — HEAD emitted
#      "[arena] learner reset @ 0.5 64 0.5 (iron_sword)."; this file emits the
#      macro form carrying the username and "spawnpoint pinned";
#   5. every line here is `$`-prefixed, so it is the EXPANSION that matches
#      HEAD's text, never the source line as written in this file.

# --- Clear inventory so the read-back gate sees EXACTLY the template gear ---
$clear $(learner)

# --- Teleport to the fixed spawn, facing the dummy (+X, yaw 90) ---
#     `$(x).5` concatenates the anchor with a half-block offset: x=0 -> "0.5",
#     x=340 -> "340.5". This is deliberately an ABSOLUTE coordinate. Whether a
#     /teleport `<location>` relative is measured from the TARGET or from the
#     EXECUTION POSITION is contested on Java, so no relative form is trusted to
#     place a named player from scratch; an absolute coordinate has exactly one
#     reading. It is also byte-identical to the pre-macro command at anchor 0.
$tp $(learner) $(x).5 64 $(z).5 90 0

# --- Full health, full hunger, no leftover effects --------------------------
#     ORDER IS LOAD-BEARING: `effect clear` FIRST, then give. Do not append a
#     trailing clear "to be tidy" — that is the bug this ordering removes.
#
#     An instant effect given with a duration is NOT applied synchronously at
#     addEffect; it is applied on the effect instance's first TICK. A trailing
#     `effect clear` in the SAME tick can therefore strip the instance before it
#     ever applies, silently restoring nothing. Clearing first removes that
#     ambiguity by construction: the clear does the only job it was ever there
#     for (stripping leftovers from the previous episode), and nothing can strip
#     what we just granted.
#
#     Why the old trailing-clear idiom looked fine: bridge/bot.js handleReset
#     issues its OWN heal, so a dead function-side instant_health would have
#     been masked. saturation has no such backstop and AC18 rides on it alone,
#     so "health resets have always worked" was never evidence this idiom did.
#
#     saturation amplifier 19 == level 20 == +20 food and +40 saturation per
#     tick, both capped at full. Restoring food matters: naturalRegeneration is
#     off, but hunger still drives exhaustion state, and letting it drift makes
#     the episode-start state non-stationary (plan AC18).
#
#     THE `1` IS GAMETICKS, NOT SECONDS. /effect give's duration argument is
#     "the effect's duration in seconds (or in gameticks for instant_damage,
#     instant_health, and saturation)" — minecraft.wiki, Commands/effect. All
#     three effects used in this datapack are in that instant list, so `1` means
#     ONE GAMETICK (~50 ms), and omitting the argument would default to 1 tick
#     rather than 30 seconds. This matches what bridge/bot.js:858-860 already
#     says of the same command: "applied within a tick and never lingering in
#     active effects, so the gate's no-effects check is unaffected."
#
#     CONSEQUENCE FOR THE READ-BACK GATE: effectively none. The instances are
#     gone after one tick, so a requireNoEffects gate has nothing to wait for and
#     the 3 s DEFAULT_READBACK.timeoutMs is untouched.
#
#     RETRACTED TEXT — an earlier revision of this file asserted the following.
#     It is FALSE; it read /effect give's duration as seconds. Quoted verbatim so
#     a grep for the original words lands here, one marker per line so a
#     single-line hit cannot be mistaken for an assertion:
#       FALSE, RETRACTED: "both effects now stay ACTIVE for their 1-second
#       FALSE, RETRACTED:  duration, so a gate with requireNoEffects waits up to
#       FALSE, RETRACTED:  ~1 s before passing"
#       FALSE, RETRACTED: "it does consume about a third of the gate budget on
#       FALSE, RETRACTED:  every reset"
#       FALSE, RETRACTED: "none active ONCE the ~1s instant-effect window
#       FALSE, RETRACTED:  expires"
#     Cite the minecraft.wiki line above before changing this again.
#
#     One tick is also exactly WHY the ordering above is load-bearing: the
#     window in which a trailing `effect clear` could strip the instance before
#     its first tick applies is the same single tick the effect lives for.
$effect clear $(learner)
$effect give $(learner) minecraft:instant_health 1 9 true
$effect give $(learner) minecraft:saturation 1 19 true

# --- Re-gear: exactly one iron sword (matches inventory:['iron_sword']) ---
$give $(learner) minecraft:iron_sword 1

# --- Per-bot spawnpoint on THIS pad ----------------------------------------
#     doImmediateRespawn is true and the world spawn is a single shared point.
#     Without this, any death teleports the bot to pad 0 and contaminates that
#     pad's episode.
#     `execute as <bot> at @s run spawnpoint @s ~ ~ ~` is deliberately chosen
#     over a composed absolute position: the execution position and the target
#     are the SAME entity, so the command is correct under either reading of how
#     a BlockPos relative resolves. The bot was pinned to its pad on the /tp
#     above, so this records (x, 64, z) — inside this pad, never pad 0.
$execute as $(learner) at @s run spawnpoint @s ~ ~ ~

$tellraw @a[tag=arena_debug] {"text":"[arena] learner $(learner) reset @ $(x).5 64 $(z).5 (iron_sword, spawnpoint pinned).","color":"aqua"}
