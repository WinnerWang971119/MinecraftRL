# arena:spawn_learner_pad — MACRO FUNCTION. Reset ONE learner bot on ONE pad.
#
# ============================================================================
# MACRO ARGUMENT CONTRACT
# ============================================================================
#   /function arena:spawn_learner_pad {x:<int>,z:<int>,learner:"<name>",nonce:<int>}
#
#   x, z    : pad ANCHOR. NON-NEGATIVE PLAIN INTEGERS, no NBT type suffix.
#             (See arena:setup_pad's header for the full rationale — `$(x)` is a
#             textual substitution and this file builds `$(x).5`, so `x:-340`
#             would silently yield -340.5, i.e. anchor MINUS half a block, and
#             `x:340L` would yield the non-coordinate `340L.5`. These numbers
#             illustrate literal syntax only; they are not anchors.)
#   learner : the learner bot's Minecraft username, e.g. "learner_bot" (pad 0)
#             or "learner_3". Must be opped (server/ops.json).
#   nonce   : NON-NEGATIVE PLAIN INTEGER, unique per reset. Stamped into the
#             causality beacon on the last line so a beacon that arrives after
#             its own reset gave up cannot satisfy the next one. REQUIRED.
#
#   Normally invoked via arena:reset_pad, which forwards its own arguments here.
#   arena:spawn_learner is the pad-0 / "learner_bot" convenience wrapper.
#
# ============================================================================
# RESET TEMPLATE (MUST match bridge/bot.js resetTemplate, anchored)
# ============================================================================
#   position : (x+0.5, 64, z+0.5)   block centre, one block above the y=63 floor
#   facing   : +X toward the dummy  (yaw -90, pitch 0)  <- -90, see the teleport
#   health   : full (20)
#   food     : full (20) + full saturation
#   inventory: exactly { iron_sword } in the slots inventory.items() can see,
#              PLUS a full iron set WORN in the four armor slots, which it
#              cannot. The re-gear block below owns that distinction; it is
#              the reason a green reset ack is not evidence of armor.
#   effects  : none active. The instant_health/saturation instances granted
#              below last ONE GAMETICK (~50 ms) — see the ordering note further
#              down for why the `1` is ticks and not seconds — so they are gone
#              long before a requireNoEffects read-back gate can sample them.
#              Do not "fix" the ordering by appending a trailing `effect clear`
#              — that is the bug the ordering removes.
#   spawnpoint: this pad
#
# At anchor (0,0) with learner="learner_bot" the /tp line expands to
#   `tp learner_bot 0.5 64 0.5 -90 0`
# — the same POSITION and USERNAME the pre-macro arena:spawn_learner used, which
# is exactly what AC11 pins (same ports, usernames and coordinates at N=1).
#
# THE YAW DELIBERATELY NO LONGER MATCHES THE PRE-MACRO TEXT. That text ended
# `90 0`, and 90 was measured live to point the learner AWAY from the dummy (the
# teleport comment below carries the reading). AC11 says nothing about rotation,
# and the offline proof of AC11 — bridge/bot.test.js "the default-anchor reset
# command is byte-identical to the committed arena:reset wrapper" — compares the
# `function arena:reset_pad {...}` invocation line in reset.mcfunction, which
# carries no yaw at all. So this correction cannot move that test.
#
# Do NOT read the expansion above as a claim about the whole sequence — it
# differs from the pre-macro arena:spawn_learner in seven ways, all deliberate:
#   1. no trailing `effect clear` (see the ordering note below);
#   2. an added saturation restore (food/hunger stationarity, plan AC18);
#   3. an added per-bot /spawnpoint (cross-pad respawn contamination);
#   4. a different tellraw payload — HEAD emitted
#      "[arena] learner reset @ 0.5 64 0.5 (iron_sword)."; this file emits the
#      macro form carrying the username, the loadout and "spawnpoint pinned";
#   5. every line here is `$`-prefixed, so it is the EXPANSION that matches
#      HEAD's text, never the source line as written in this file;
#   6. the spawn yaw is -90; HEAD's 90 was inverted (T22);
#   7. a full set of iron armor is EQUIPPED (M4 iron loadout, issue #33).
#      HEAD armed the learner with a sword and dressed it in nothing.

# --- Clear inventory so the read-back gate sees EXACTLY the template gear ---
$clear $(learner)

# --- Teleport to the fixed spawn, facing the dummy (+X, yaw -90) ------------
#     THE YAW IS -90, NOT 90. Minecraft's look vector is
#         look.xz = (-sin(yaw), cos(yaw))     [env/perception_filter.py:59-65]
#     so yaw 0 -> (0, +1) = +Z, yaw 90 -> (-1, 0) = -X, yaw -90 -> (+1, 0) = +X.
#     The dummy sits at anchor+3.5 on the SAME z, so learner -> dummy is +X, and
#     the yaw that looks +X is the solution of -sin(yaw) = 1, i.e. yaw = -90
#     (cos(-90) = 0, so the z component is 0 as the shared z requires).
#
#     MEASURED, NOT INFERRED (T22). This line used to read `90 0` and was
#     annotated "facing the dummy (+X, yaw 90)". A server-authoritative read
#     after a reset, both bots at rest, showed:
#         learner_bot Rotation: [90.0f, 0.0f]   Pos: [1024.5d, 64.0d, 0.5d]
#         dummy_bot   Rotation: [-90.0f, 0.0f]  Pos: [1027.5d, 64.0d, 0.5d]
#     The learner looked -X while the dummy stood +X of it: dot product -1,
#     facing exactly away. Corroborated by the wire yaws (1.570796 rad and
#     4.712389 rad) and by a live walk whose forward/APPROACH leg moved -X.
#
#     Nothing in the automated suite catches a bad yaw: bot.attack(entity) does
#     not require the attacker to be facing its target, which is why AC8's
#     combat probe passed throughout and why this survived so long. The cost was
#     paid by the reward instead — r_aim is hard-gated on the opponent being
#     visible AND in the crosshair (env/reward.py), so every episode opened with
#     the one dense shaping term unearnable until the agent turned ~180 degrees.
#     spawn_dummy_pad.mcfunction carries the mirrored fix; the two must stay
#     opposite (learner -90, dummy +90) or the bots point the same way.
#
#     `$(x).5` concatenates the anchor with a half-block offset: x=0 -> "0.5",
#     x=340 -> "340.5". This is deliberately an ABSOLUTE coordinate. Whether a
#     /teleport `<location>` relative is measured from the TARGET or from the
#     EXECUTION POSITION is contested on Java, so no relative form is trusted to
#     place a named player from scratch; an absolute coordinate has exactly one
#     reading. Its expansion at anchor 0 also reproduces the pre-macro command's
#     coordinate text character for character; only the yaw differs (header).
$tp $(learner) $(x).5 64 $(z).5 -90 0

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

# --- Re-gear: one iron sword HELD, a full iron set WORN ---------------------
#     ARMOR IS `item replace`, NOT `give`, AND THAT IS THE WHOLE POINT.
#     `/give` pushes a stack into the first free slot of the INVENTORY and
#     stops there; nothing in Minecraft moves a piece from a player's
#     inventory onto their body. Four `$give`s would leave the learner
#     CARRYING a full iron set at zero armor points — armored in the chat
#     log, in the hotbar screenshot and in the run notes, naked in the fight.
#     `item replace entity <target> armor.<slot> with <item>` writes the
#     equipment slot itself, so the piece is worn the instant the command
#     returns. It also OVERWRITES rather than appends, which makes these FOUR
#     ARMOR LINES idempotent across resets whatever the `$clear` above left
#     behind (the `$give` sword above is NOT independently idempotent — it
#     relies on the `$clear`, and would stack a spare sword per reset without
#     it), and
#     hands out a FRESH piece every episode so armor durability never
#     accumulates across a run. Four `give`s would instead pile a spare set
#     into the inventory every reset.
#
#     ALL FIVE ITEM IDS AND ALL FOUR SLOT NAMES WERE READ OUT OF THE PINNED
#     JAR, NOT RECALLED. Every line in this file is a macro line, parsed at
#     INSTANTIATION, so ONE bad id aborts the WHOLE function — no teleport,
#     no heal, no saturation, no spawnpoint, no gear — silently, with nothing
#     in the boot log. spawn_dummy_pad.mcfunction's header carries the full
#     writeup and the three consecutive boot failures that taught it. What
#     was checked, in server/versions/1.21.1/paper-1.21.1.jar:
#       - net/minecraft/world/inventory/SlotRanges carries the literals
#         `armor.head`, `armor.chest`, `armor.legs`, `armor.feet` (alongside
#         `armor.body`, `weapon.mainhand`, `weapon.offhand`), and
#         net/minecraft/server/commands/ItemCommands carries `item`,
#         `replace`, `entity` and `with` — so the grammar below is this
#         build's, not a later one's.
#       - net/minecraft/world/item/Items carries iron_helmet,
#         iron_chestplate, iron_leggings, iron_boots and iron_sword, and
#         minecraft-data's 1.21.1 registry agrees (864, 865, 866, 867, 833).
#
#     THE RESET READ-BACK GATE IS BLIND TO ARMOR. mineflayer's
#     `inventory.items()` spans slots 9-44 (prismarine-windows/index.js:11,
#     `minecraft:inventory` -> inventory {start: 9, end: 44}); the four armor
#     slots are 5-8, outside that window entirely. A template check on
#     `inventory` therefore reads ['iron_sword'] and PASSES on a fighter
#     wearing nothing at all — which is why armor is proved by a
#     server-authoritative read instead, and why that read is fail-closed
#     (T3, AC9). Never take a green reset ack as evidence the armor is on.
$give $(learner) minecraft:iron_sword 1
$item replace entity $(learner) armor.head with minecraft:iron_helmet
$item replace entity $(learner) armor.chest with minecraft:iron_chestplate
$item replace entity $(learner) armor.legs with minecraft:iron_leggings
$item replace entity $(learner) armor.feet with minecraft:iron_boots

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

# DEBUG LINE — IT REPORTS WHAT WAS ASKED FOR, NOT WHAT LANDED.
#     The loadout text is a hard-coded literal printed unconditionally, the
#     same species of claim as the dummy's `kb_resist=1.0` (see that file's
#     DEBUG LINE note). It is emitted from inside the very function that
#     issues the gear commands, so at most it says they were SENT. The four
#     armor slots are invisible to the reset gate on top of that. Confirm the
#     loadout with the server-authoritative read (T3), never from this line.
$tellraw @a[tag=arena_debug] {"text":"[arena] learner $(learner) reset @ $(x).5 64 $(z).5 (iron_sword + full iron armor, spawnpoint pinned).","color":"aqua"}

# --- RESET CAUSALITY BEACON. MUST STAY THE LAST LINE OF THIS FILE. ----------
#     The bridge's read-back gate verifies TEMPLATE MATCH, not causality, and
#     after a kill the natural post-respawn state IS the template state (full
#     health at the pinned spawnpoint, effects cleared by death, keepInventory
#     preserving the gear, worn armor included). So if this function ever aborts at INSTANTIATION —
#     the silent-at-boot, total-at-runtime macro hazard documented in
#     spawn_dummy_pad's header, whose likeliest real trigger is a Paper 1.21.2
#     bump or someone "fixing" the `generic.` attribute prefix from memory —
#     the gate would pass and the bridge would ack a reset that never happened.
#     No saturation restore (AC18 drifts), no armor, no attribute re-apply,
#     invisibly.
#
#     A bare respawn cannot produce this line, so observing it is proof the
#     function ran. It is addressed to $(learner) BY NAME rather than @a so a
#     25-pad fleet does not broadcast 2N confirmations to 2N clients, and it
#     carries the anchor and the username so one pad's beacon can never confirm
#     another's. bridge/bot.js matches this text EXACTLY (formatResetConfirmation)
#     — change one and you must change the other.
$tellraw $(learner) {"text":"[arena] reset_ok learner $(x) $(z) $(learner) $(nonce)"}
