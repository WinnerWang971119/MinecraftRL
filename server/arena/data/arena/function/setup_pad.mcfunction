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

$tellraw @a[tag=arena_debug] {"text":"[arena] pad built @ anchor $(x),64,$(z): floor 25x25, closed bedrock ring y=64..71 (768 blocks).","color":"green"}
