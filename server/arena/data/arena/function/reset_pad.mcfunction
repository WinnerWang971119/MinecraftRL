# arena:reset_pad — MACRO FUNCTION. Per-episode reset of BOTH bots on ONE pad.
#
# ============================================================================
# MACRO ARGUMENT CONTRACT  (this is the entry point the bridge calls every reset)
# ============================================================================
#   /function arena:reset_pad {x:<int>,z:<int>,learner:"<name>",dummy:"<name>",nonce:<int>}
#
#   x, z    : pad ANCHOR. NON-NEGATIVE PLAIN INTEGERS, no NBT type suffix
#             (`0`, `340`, `1200` — never `0b`, `0d`, `340L`, `"340"`, `340.0`;
#             these illustrate LITERAL SYNTAX only, they are not anchors).
#             The anchor is the LEARNER SPAWN CELL: learner feet land at
#             (x+0.5, 64, z+0.5), the dummy at (x+3.5, 64, z+0.5).
#             `$(x)` is a TEXTUAL substitution and this call chain builds
#             `$(x).5`, so a negative anchor would yield anchor-0.5 and an NBT
#             suffix would yield a non-coordinate. Neither is checked at runtime.
#   learner : learner bot username, e.g. "learner_bot" (pad 0) or "learner_3".
#   dummy   : dummy   bot username, e.g. "dummy_bot"   (pad 0) or "dummy_3".
#             Both must be opped (server/ops.json).
#   nonce   : NON-NEGATIVE PLAIN INTEGER, unique per reset (the bridge passes its
#             monotonic reset epoch). Forwarded verbatim to both spawn functions
#             and stamped into their causality beacons, so a beacon delayed past
#             its own reset cannot satisfy the NEXT one. Pad-0 wrappers pass 0.
#             REQUIRED, like every macro key: a macro function errors if a
#             referenced key is absent.
#
#   Example, pad 0 (the anchor is (0,0) by definition, not a copied constant):
#     /function arena:reset_pad {x:0,z:0,learner:"learner_bot",dummy:"dummy_bot"}
#   Example, pad i > 0 — the anchor comes from T10's padAnchor(i), which is the
#   SOLE coordinate source; no literal anchor is reproduced here on purpose:
#     /function arena:reset_pad {x:<padAnchor(3).x>,z:<padAnchor(3).z>,learner:"learner_3",dummy:"dummy_3"}
#
#   This does NOT rebuild geometry — arena:setup_pad is once-per-pad-per-boot.
#   The commands here are async/unacked like any others, so the bridge MUST
#   still run its read-back gate before replying reset_ack.

# --- Sweep loose entities (dropped items, XP orbs) inside this pad ----------
#     keepInventory is on so deaths drop nothing, but a sweep keeps the pad
#     provably empty episode to episode. Radius 64 is the SAME radius used by
#     arena:setup_pad, deliberately kept identical so there is no band of space
#     that one sweep reaches and the other misses. It comfortably covers the pad
#     (max corner-to-anchor distance ~21.4) and must stay below HALF of T10's
#     PAD_SPACING so it can never reach a neighbouring pad — see the invariant
#     note in setup_pad.mcfunction, which owns the reasoning for both radii.
#     `distance=` is used because selector x=/y=/z= arguments cannot be relative.
$execute positioned $(x) 64 $(z) run kill @e[type=!minecraft:player,distance=..64]

# --- Re-place and re-gear both bots ----------------------------------------
#     Arguments are forwarded verbatim; the two callees hold the single source
#     of truth for each bot's reset template.
$function arena:spawn_learner_pad {x:$(x),z:$(z),learner:"$(learner)",nonce:$(nonce)}
$function arena:spawn_dummy_pad {x:$(x),z:$(z),dummy:"$(dummy)",nonce:$(nonce)}
