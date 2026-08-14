# arena:setup_pad — MACRO FUNCTION. Builds one fully-enclosed arena pad.
#
# ============================================================================
# MACRO ARGUMENT CONTRACT  (read this before calling — T9/T10 depend on it)
# ============================================================================
#   /function arena:setup_pad {x:<int>,z:<int>}
#
#   x : pad ANCHOR X. A NON-NEGATIVE PLAIN INTEGER. No NBT type suffix
#       (`0`, `340`, `1200` — never `0b`, `0d`, `340L`, `"340"`, `340.0`).
#       The numbers here illustrate LITERAL SYNTAX only; they are not anchors.
#   z : pad ANCHOR Z. Same rules.
#
#   WHY non-negative plain integers: `$(x)` is a TEXTUAL substitution. This file
#   only ever uses `$(x)` in forms that concatenate safely — bare (`$(x) 64 $(z)`)
#   or with a `.5` suffix in the sibling reset_pad file. A negative anchor would
#   turn `$(x).5` into `-340.5` (anchor MINUS a half block), and an NBT suffix
#   would turn `$(x)` into `340L`, which is not a coordinate. Anchors come from
#   T10's padAnchor(i), the SOLE coordinate source, which only ever produces
#   non-negative integers — so the constraint is free, but it IS a constraint
#   and it is not checked at runtime. The formula is deliberately NOT reproduced
#   here: this datapack must not carry a copy of a number T10 owns.
#
#   Idempotent: safe to re-run on an existing pad.
#   Called ONCE PER PAD PER BOOT. Global, world-wide state lives in arena:setup.
#
# ============================================================================
# GEOMETRY — everything is expressed relative to the ANCHOR (x, 64, z)
# ============================================================================
#   The anchor is the LEARNER SPAWN CELL, not the floor origin.
#   Learner feet sit at (x+0.5, 64, z+0.5); the dummy at (x+3.5, 64, z+0.5).
#
#   Let A = (x, z). Offsets below are relative to A.
#
#     sub-floor  y=62       bedrock       X -8..+16, Z -12..+12   (25 x 25 = 625)
#     floor      y=63       smooth_stone  X -8..+16, Z -12..+12   (25 x 25 = 625)
#     interior   y=64..71   air           X -8..+16, Z -12..+12   (25*25*8 = 5000)
#     walls      y=64..71   bedrock       the perimeter RING of that footprint
#
#   The air fill runs BEFORE the wall fills, so the four wall slabs overwrite the
#   outermost ring of the freshly-cleared column. Net walkable interior is
#   X -7..+15, Z -11..+11 (23 x 23), standing on solid floor at y=63.
#
#   WALL CLOSURE (this is the classic four-fills-with-corner-holes bug — it is
#   avoided here by giving BOTH pairs of slabs their FULL span, so each corner
#   column is covered twice rather than zero times):
#
#     west  slab: X = -8            , Z = -12..+12   -> 1 * 25 * 8 = 200 blocks
#     east  slab: X = +16           , Z = -12..+12   -> 1 * 25 * 8 = 200 blocks
#     north slab: X = -8..+16       , Z = -12        -> 25 * 1 * 8 = 200 blocks
#     south slab: X = -8..+16       , Z = +12        -> 25 * 1 * 8 = 200 blocks
#                                                      ------------------------
#                                        sum with overlap   800
#     The four corner columns (-8,-12) (-8,+12) (+16,-12) (+16,+12) are each
#     filled twice: 4 corners * 8 layers = 32 double-covered blocks.
#     Distinct wall blocks = 800 - 32 = 768.
#     Cross-check against the ring identity: (25*25 - 23*23) * 8
#                                          = (625 - 529) * 8 = 96 * 8 = 768.  OK
#
#   Wall height is 8 blocks above the floor (y=64 feet level through y=71). A
#   player jumps ~1.25 blocks and has no blocks to place, so the pad is closed.
#   There is deliberately no ceiling.
#
#   EXACT BOUNDS for the AC7 / TC6 live assertions:
#     occupiable block volume : x in [A.x-7, A.x+15], z in [A.z-11, A.z+11]
#     player-center bounds    : x in [A.x-6.7, A.x+15.7]   (0.6-wide hitbox)
#                               z in [A.z-10.7, A.z+11.7]
#
#   Largest single fill is the 5000-block air column — well under the 32768
#   block /fill limit.
#
# ============================================================================
# WHY `execute positioned` + `~` HERE, BUT NOT FOR /tp
# ============================================================================
#   Block-position arguments (`fill`, `setblock`) and the `distance=` selector
#   predicate resolve `~` against the COMMAND'S EXECUTION POSITION, so
#   `execute positioned $(x) 64 $(z) run ...` relocates them correctly and needs
#   no string arithmetic at all. Integer Vec3 arguments get block-centered
#   (`positioned 0 64 0` -> 0.5), which is harmless here because block coords
#   floor: floor(0.5) - 8 == -8 either way.
#
#   /teleport is the exception and is handled in the spawn_*_pad files instead:
#   whether its `<location>` relatives are measured from the TARGET or from the
#   execution position is contested on Java, so those files use absolute
#   coordinates to place a bot and only ever use a relative offset when both
#   candidate origins are provably the same point.
#
#   Entity-selector `x=/y=/z=` arguments cannot be relative at all, hence
#   `distance=` everywhere below.
#
# ============================================================================
# CHUNK RESIDENCY — WHY THIS FUNCTION FORCE-LOADS BEFORE IT BUILDS (issue #27)
# ============================================================================
#   `/fill` resolves BOTH corner arguments with the LOADED-ONLY form of the
#   block-position argument: it asks whether that chunk is ALREADY resident and
#   throws "that position is not loaded" when it is not. The throw happens
#   during ARGUMENT RESOLUTION, so the fill never reaches the block writer — and
#   the writer is the only part of `/fill` that would have loaded a chunk on
#   demand. Inside a function such a per-command failure is swallowed whole: the
#   caller sees "Running function", the log stays clean, and the pad is air.
#
#   Not hypothetical. Every caller issues this from wherever the bots happen to
#   be standing, which on a fresh bridge start is pad 0. A pad 512+ blocks away
#   is outside every loaded chunk at that instant, so all seven fills no-opped,
#   both bots teleported onto air, fell 124 blocks to the flat-world floor and
#   fought mid-fall — crits and reduced hits instead of a clean 6.
#
#   `/forceload` is the asymmetric partner and the reason this is fixable here
#   rather than at the call site: its arguments are CHUNK-CONTAINING BLOCK
#   COLUMNS (X and Z only, no Y, `~` allowed) with NO loaded-only gate, and
#   marking a chunk forced loads/generates it to full status right there. So the
#   add below makes the footprint resident and the fills that follow resolve —
#   for every caller, at any anchor, with no ordering dependency on where a bot
#   is standing.
#
#   ASSUMPTION — NOT PROVEN ON A LIVE SERVER. It is ONE HOP, not the whole
#   mechanism, and it is worth stating narrowly so a live result is readable
#   either way. Settled: the add blocks until the chunk reaches full chunk
#   STATUS. Open: the loaded-only gate the fills trip does not read chunk
#   status — it reads the server's map of full chunks, and that map is filled in
#   by a SEPARATE full-status transition. Whether that transition has been
#   processed by the time the add returns is the only unclosed link in the
#   chain. It very likely has, because the main thread drains chunk-system work
#   while it blocks and that is precisely what the wait exists for; but that is
#   not provable from the bytecode, so treat it as open.
#
#   If the transition turns out NOT to be processed in time, a pad at a FRESH
#   anchor still comes up missing and the fix becomes two-phase: park the anchor
#   in storage NBT, `schedule` a reader a tick later, and re-inject the anchor
#   with `function ... with storage <path>`. It has to travel through storage
#   because `schedule function` cannot carry macro arguments — the command has
#   no `with` form at all. Two traps for whoever writes that version: `schedule`
#   REPLACES a pending entry for the same function id unless `append` is used,
#   and a single storage slot would be overwritten by the next pad's anchor. A
#   fleet boots many pads in the same tick, so that form needs
#   `schedule ... append` plus a per-pad LIST in storage, never a scalar.
#
#   The footprint spans at most 3 chunks per axis (a 25-block span can straddle
#   three), so 9 chunks at worst — far under the 256-per-add limit.
#
#   `/forceload` needs permission level 2, which is exactly what a function body
#   runs at under this server's `function-permission-level=2`. Same level every
#   other command in this datapack already uses; nothing to raise.
#
#   SELF-HEALING LEAK: a crash between the add and the remove leaves the chunks
#   marked forced in the world's saved chunk list. The next boot's add then
#   fails with "no chunks marked" — a per-command RUNTIME failure, which unlike
#   a bad macro value does NOT abort the function. The mechanism differs on that
#   branch and the difference is worth knowing: the add short-circuits on the
#   already-present entry and skips the synchronous load ENTIRELY, so residency
#   there rests on the boot-time restore re-ticketing saved forced chunks, not
#   on anything this function does. Same outcome by a different route. The
#   remove at the end clears the leak either way.
#
#   This runs FIRST, ahead of the entity sweep, so the sweep is not reading an
#   empty world. Only partly, though, and the comment should not overclaim: the
#   add blocks on BLOCK data reaching full status, while entities ride a later
#   asynchronous transition that the forced ticket requests AFTER the add
#   returns. Entities saved in a chunk that was unloaded may therefore still be
#   outside the selector's lookup. Benign — worst case one stray item survives a
#   boot, never missing geometry — and the placement is still strictly better
#   than sweeping before anything is resident at all.
$execute positioned $(x) 64 $(z) run forceload add ~-8 ~-12 ~16 ~12

# --- Clear stray entities around this pad (items, XP orbs, anything non-player).
#     Radius 64 covers the pad (max corner-to-anchor distance ~21.4).
#     INVARIANT: this radius must stay below HALF of T10's PAD_SPACING, or a
#     sweep would reach into a neighbouring pad. T10 owns that constant and this
#     file deliberately does not restate its value, not even in derived form —
#     a restated number goes stale silently. If PAD_SPACING is ever reduced,
#     re-check this radius and the one in reset_pad against it.
$execute positioned $(x) 64 $(z) run kill @e[type=!minecraft:player,distance=..64]

# --- Bedrock sub-floor at y=62: nothing can dig or fall through to the void.
$execute positioned $(x) 64 $(z) run fill ~-8 62 ~-12 ~16 62 ~12 minecraft:bedrock replace

# --- Play surface at y=63 (feet stand at y=64).
$execute positioned $(x) 64 $(z) run fill ~-8 63 ~-12 ~16 63 ~12 minecraft:smooth_stone replace

# --- Clear the whole 8-block-tall column ABOVE the floor, then wall its ring.
#     Order matters: air first, bedrock ring second.
$execute positioned $(x) 64 $(z) run fill ~-8 64 ~-12 ~16 71 ~12 minecraft:air replace

# --- CLOSED bedrock perimeter, y=64..71. Both pairs span the full side so every
#     corner column is covered twice; there is no gap at any corner.
#     west  (-X face)
$execute positioned $(x) 64 $(z) run fill ~-8 64 ~-12 ~-8 71 ~12 minecraft:bedrock replace
#     east  (+X face)
$execute positioned $(x) 64 $(z) run fill ~16 64 ~-12 ~16 71 ~12 minecraft:bedrock replace
#     north (-Z face)
$execute positioned $(x) 64 $(z) run fill ~-8 64 ~-12 ~16 71 ~-12 minecraft:bedrock replace
#     south (+Z face)
$execute positioned $(x) 64 $(z) run fill ~-8 64 ~12 ~16 71 ~12 minecraft:bedrock replace

# --- PROVE the pad before claiming it, while the footprint is still held.
#     This line used to fire unconditionally, which meant it reported "pad
#     built" in EXACTLY the failure this fix exists to prevent: seven no-opped
#     fills, no geometry, and a cheerful green line saying otherwise.
#
#     Three probes, chosen so that between them all seven fills must have
#     resolved their arguments:
#       ~ 63 ~     floor at the learner spawn cell — the block the bots were
#                  missing when they fell; written only by the y=63 fill.
#       ~-8 71 ~   west wall at MID-span — written only by the west fill.
#       ~16 71 ~   east wall at MID-span — written only by the east fill.
#     Mid-span and not a corner, deliberately: a corner column is written by TWO
#     wall fills, so a corner probe proves only that ONE of them ran. The west
#     and east probes each pin their fill's two corner chunks, and between them
#     that is all four corners of the footprint — which is also where the north
#     and south fills take their corners from. So if these three pass, every
#     fill in the file resolved.
#
#     HONEST LIMIT, and it is a real one: this detects "the chunks were resident
#     but the fills did not land". It CANNOT detect "the chunks never loaded",
#     because `if block` resolves its position through the very same loaded-only
#     gate and would throw during its own argument resolution, printing nothing.
#     In that case the discriminator degrades from a failure line to SILENCE.
#     Still strictly better than a false success, and silence where a line is
#     expected is itself a usable signal — but read it as "no verdict", not as
#     "no failure".
#
#     Failure goes to @a while success keeps the debug tag it always had: a pad
#     that did not build must not be invisible to whoever is watching. Neither
#     text can collide with the reset beacons the bridge matches on.
$execute positioned $(x) 64 $(z) if block ~ 63 ~ minecraft:smooth_stone if block ~-8 71 ~ minecraft:bedrock if block ~16 71 ~ minecraft:bedrock run tellraw @a[tag=arena_debug] {"text":"[arena] pad built @ anchor $(x),64,$(z): floor 25x25, closed bedrock ring y=64..71 (768 blocks), verified.","color":"green"}
$execute positioned $(x) 64 $(z) unless block ~ 63 ~ minecraft:smooth_stone run tellraw @a {"text":"[arena] PAD NOT BUILT @ anchor $(x),64,$(z): no floor at the spawn cell — bots teleported here will fall.","color":"red"}
$execute positioned $(x) 64 $(z) unless block ~-8 71 ~ minecraft:bedrock run tellraw @a {"text":"[arena] PAD NOT BUILT @ anchor $(x),64,$(z): west wall missing — the pad is not enclosed.","color":"red"}
$execute positioned $(x) 64 $(z) unless block ~16 71 ~ minecraft:bedrock run tellraw @a {"text":"[arena] PAD NOT BUILT @ anchor $(x),64,$(z): east wall missing — the pad is not enclosed.","color":"red"}

# --- RELEASE the footprint. The geometry is durable now, and a pad only needs
#     to be resident while it is occupied — which the bots' own chunk tickets
#     already guarantee. The run that found issue #27 is the proof: with nothing
#     force-loaded, the bots still arrived in fully generated terrain and fell
#     through it. Holding every pad forever would keep a whole fleet's chunks
#     ticking for no one, against the TPS budget this topology is measured on.
#     Must stay paired with the add above — see CHUNK RESIDENCY in the header.
#     LAST, after the probes: they read blocks, and reading is gated on the same
#     residency the add is holding open.
$execute positioned $(x) 64 $(z) run forceload remove ~-8 ~-12 ~16 ~12
