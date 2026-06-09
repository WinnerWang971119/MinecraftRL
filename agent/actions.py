"""agent/actions — Discrete action macro enum (frozen day-1 contract artifact AC1).

Defines the eight named macros that form the agent's action space.  Every
component that issues or interprets actions (env, DQN agent, bridge) must
import exclusively from here.  The enum is frozen once T4 merges; any change
requires a PR visible to all tracks.

These macros map directly to Mineflayer plugin/control calls on the Node.js
side.  This module is **descriptive only** — execution lives in bridge/actions.js
(task T7b), which reads MACRO_SEMANTICS as documentation and implements each
macro against the Mineflayer API.

Owner: T4 (DQN core track)
"""

from enum import IntEnum

__all__ = ["Macro", "N_ACTIONS", "MACRO_SEMANTICS"]


class Macro(IntEnum):
    """Discrete action macros — integer values 0..7, stable across all tracks."""

    IDLE = 0
    APPROACH = 1
    RETREAT = 2
    STRAFE_L = 3
    STRAFE_R = 4
    ATTACK = 5
    JUMP = 6
    TURN_TO_LAST_SEEN = 7


# Derived so it can never drift from the enum definition.
N_ACTIONS: int = len(Macro)  # == 8


# ---------------------------------------------------------------------------
# Semantic mapping table
# ---------------------------------------------------------------------------
# Documents what each macro means and how bridge/actions.js executes it.
# The bridge reads this at startup for self-consistency assertions and logs it
# for debugging; the agent uses it only as documentation.
#
# Execution model (T7b notes):
#   - Movement macros (APPROACH, RETREAT, STRAFE_L, STRAFE_R) use
#     bot.setControlState(<direction>, true/false) held for ACTION_REPEAT
#     ticks, then cleared.  They do NOT use pathfinder goals.
#   - ATTACK calls bot.attack(entity) for a single swing and is manually
#     cooldown-gated by the bridge (NOT bot.pvp.attack).
#   - JUMP sets the "jump" control state for one tick.
#   - TURN_TO_LAST_SEEN calls bot.lookAt(last_seen_position, force=true)
#     using the stored memory position of the opponent (memory-driven look).
#   - IDLE clears all control states and does not call bot.attack.

MACRO_SEMANTICS: dict[Macro, str] = {
    Macro.IDLE: (
        "No-op.  All control states are cleared (forward/back/left/right/jump)"
        " and no attack is issued.  The bot stands still for this tick."
    ),
    Macro.APPROACH: (
        "Move toward the opponent.  Calls bot.setControlState('forward', true)"
        " held for ACTION_REPEAT ticks, then cleared.  Uses raw control state,"
        " not a pathfinder goal."
    ),
    Macro.RETREAT: (
        "Move away from the opponent.  Calls bot.setControlState('back', true)"
        " held for ACTION_REPEAT ticks, then cleared.  Uses raw control state,"
        " not a pathfinder goal."
    ),
    Macro.STRAFE_L: (
        "Strafe left relative to the bot's current facing direction.  Calls"
        " bot.setControlState('left', true) held for ACTION_REPEAT ticks, then"
        " cleared.  Uses raw control state, not a pathfinder goal."
    ),
    Macro.STRAFE_R: (
        "Strafe right relative to the bot's current facing direction.  Calls"
        " bot.setControlState('right', true) held for ACTION_REPEAT ticks, then"
        " cleared.  Uses raw control state, not a pathfinder goal."
    ),
    Macro.ATTACK: (
        "Single melee swing.  Calls bot.attack(entity) directly.  The bridge"
        " enforces the weapon's attack-cooldown window before issuing the call"
        " (manually gated — does NOT use bot.pvp.attack or any pvp plugin)."
    ),
    Macro.JUMP: (
        "Jump.  Calls bot.setControlState('jump', true) for one tick, then"
        " clears it.  Used for traversal and to break predictable movement."
    ),
    Macro.TURN_TO_LAST_SEEN: (
        "Rotate to face the last-known opponent position stored in bridge memory."
        "  Calls bot.lookAt(last_seen_position, true) (force=true bypasses"
        " interpolation).  Executes even when the opponent is not currently"
        " visible so the agent can re-acquire line-of-sight."
    ),
}

# Sanity-check: every member must have a semantics entry (caught at import time).
_missing = [m for m in Macro if m not in MACRO_SEMANTICS]
if _missing:
    raise RuntimeError(
        f"MACRO_SEMANTICS is missing entries for: {[m.name for m in _missing]}"
    )
