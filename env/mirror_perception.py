"""mirror_perception — the OPPONENT-seat observation vector (M4 self-play).

WHY THIS MODULE EXISTS
----------------------
In M4 self-play the SECOND bot in each arena is driven by a **frozen copy of the
agent's own past policy**. That snapshot was trained on the 23-dim vector of
``env.observation_spec`` as computed from the **LEARNER's** seat. To act, it
needs those same 23 numbers computed from the **OPPONENT's** seat.

Nothing else in the repo can produce one. ``MCPvPEnv._state_to_obs`` is hard
wired to the seats of the wire message: ``state.self_state`` is always the SELF
block and ``state.opponent`` is always the gated target. This module is exactly
that function with the two seats exchanged — and nothing else. Keeping it a
separate object (rather than a flag on the env) is what lets the two seats own
**separate** ``PerceptionFilter`` memories, which they must: each fighter
remembers where it last *saw* the other, and those two memories are different
facts.

    wire block                     -> mirrored observation block
    ------------------------------------------------------------------
    state.opponent                 -> SELF   (indices 0..10)
    state.self_state               -> TARGET, perception-gated (11..20)
    (derived from the gated pair)  -> DERIVED (21..22)

WAYS THIS HAS GONE SILENTLY WRONG SO FAR (each pinned by a test)
----------------------------------------------------------------
1. **Rotating about the wrong yaw.** ``vel_local`` in the SELF block is the
   opponent's world-frame velocity rotated into the **OPPONENT's** yaw-aligned
   local frame. Rotating about the learner's yaw instead produces a perfectly
   well-formed, in-range, validate()-passing vector that describes a fighter
   moving in a direction it is not moving in. Nothing downstream can detect it;
   the frozen policy simply plays badly forever. ``TC7`` pins it.

2. **Skipping the perception gate.** The TARGET block MUST go through a
   ``PerceptionFilter`` with the seats reversed. The snapshot was TRAINED under
   FOV + line-of-sight + memory-timeout gating; feeding it ungated,
   always-visible input is off-distribution — it stops resembling the policy
   that earned its snapshot, so the Elo it is measured at stops meaning
   anything. It is also the project's hard rule (README rule 2): fairness gating
   lives in Python and applies to whichever seat is observing. The filter needs
   NO modification for this: ``PerceptionFilter.filter`` is role-generic over
   its ordered ``(viewer, target)`` arguments — it rotates by the VIEWER's yaw
   and tests the VIEWER's look vector — so calling it as
   ``filter(opponent, learner)`` makes the opponent the viewer.

3. **Substituting constants for absent wire fields — partially caught.**
   ``on_ground`` and ``held_item`` reach the opponent block only because T1
   added them across FIVE agreeing surfaces, and those surfaces do not all fail
   the same way. Missing surface 2 (``OpponentState`` built without the
   fields) is exactly what :func:`require_opponent_wire_fields` catches, and it
   catches it for any hand-rolled or replay state object too. Missing surfaces
   1, 4, or 5 fails loudly in a JS/Python validator before the mirror is ever
   reached. Missing surface 3 (``_snapshotOpponent``) is caught by NEITHER:
   ``assembleStateMsg`` rebuilds the block with ``Boolean(opponent.on_ground)``
   and a ``typeof``-guarded ``held_item``, so a producer that never emits these
   fields still hands Python a present, non-``None`` ``False`` / ``""`` —
   the exact "two features pinned at a constant the whole match" outcome this
   check exists to prevent, and invisible to it. ``TC11`` pins what the guard
   does cover; it is still worth having for that.

4. **Leaking memory across episodes.** ``PerceptionFilter`` memory is per-episode
   state; :meth:`OpponentMirror.reset` must clear it (and the cached vector) at
   every episode boundary, or the frozen net starts a fresh episode already
   "remembering" a position from the previous one.

WHERE ``attack_cooldown`` COMES FROM
------------------------------------
The opponent has **no** ``attack_cooldown`` channel on the wire:
``state.self_state.attack_cooldown`` is the learner's, and ``state.opponent``
carries every self-block field EXCEPT the cooldown (see the ``StateMsg``
docstring in ``bridge/messages.py``). The env shadow-tracks the opponent's swing
meter in Python and hands the value to :meth:`OpponentMirror.observe` as
``opp_attack_cooldown``. This module never computes it — a second, independent
estimate here would drift from the one ``raw_opponent_view()`` already feeds the
scripted bot, and the two seats would disagree about the same swing.

Opponent health in the SELF block is **not** a leak: from the mirrored seat it
IS self-health, which the frozen contract permits at index 0. The reverse never
happens — the observation layout has no opponent-health slot at all, so the
learner's health cannot reach the frozen net through this vector.

WHY THE ``_world_to_local_yaw`` IMPORT IS DEFERRED
--------------------------------------------------
The rotation helper is imported from ``env.mc_pvp_env`` **inside** the call, not
at module import. The env imports this module (it owns an ``OpponentMirror``),
so a module-level ``from env.mc_pvp_env import ...`` here closes an import cycle:
importing the env first would re-enter this module and ask a half-initialized
``env.mc_pvp_env`` for a name defined hundreds of lines below its own import
block — an ``ImportError`` at startup. The deferred import runs at call time,
when both modules are fully loaded either way.

The four-float helper in ``env.mc_pvp_env`` is the right one deliberately.
``env.perception_filter`` exports a same-named ``_world_to_local_yaw`` with a
DIFFERENT signature (array + yaw); it is a different function and is not
interchangeable here.

Owner: T5 (mirror track). Consumer: T6 (env wiring / ``opponent_observation()``).
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple, TYPE_CHECKING

import numpy as np

from env.observation_spec import (
    OBS_DTYPE,
    SelfState,
    build_observation,
    validate,
)
from env.perception_filter import PerceptionFilter

if TYPE_CHECKING:  # pragma: no cover - typing only
    from bridge.messages import StateMsg

__all__ = [
    "REQUIRED_OPPONENT_WIRE_FIELDS",
    "WIRE_SURFACES",
    "MirrorWireError",
    "require_opponent_wire_fields",
    "OpponentMirror",
]


# ---------------------------------------------------------------------------
# Wire-field contract (T1). These two keys are what make an opponent-seat SELF
# block possible at all; every other field the block needs (pos/yaw/pitch/
# velocity/health) already arrives on the wire. "On the wire" is not the same
# claim as "a correct measurement": the opponent's `velocity` was found to be
# broken provenance — read through the LEARNER's mineflayer connection, which
# never updates a walking entity's velocity (`entity_velocity` fires only on
# knockback/explosions; `rel_entity_move` translates position only;
# `sync_entity_position` is a 1.21.3 packet, dead on the pinned 1.21.1 server).
# T20 is fixing the provenance in the bridge now; this module has no way to
# tell a good wire value from a bad one and reads whatever state.opponent
# carries.
# ---------------------------------------------------------------------------

#: Opponent-block wire fields this module reads and CANNOT synthesize.
REQUIRED_OPPONENT_WIRE_FIELDS: Tuple[str, ...] = ("on_ground", "held_item")

#: The five surfaces that must agree for those fields to survive the trip from
#: the opponent's mineflayer connection to ``bridge.messages.OpponentState``.
#: They do not all fail the same way. Missing surface 1, 4, or 5 is caught
#: earlier, by a JS/Python validator, before a mirrored observation is ever
#: attempted. Missing surface 2 is what :func:`require_opponent_wire_fields`
#: actually catches. Missing surface 3 is caught by NEITHER: ``assembleStateMsg``
#: coerces an absent upstream value to ``False``/``""`` rather than omitting
#: the key, so the field arrives present and the guard passes it through. The
#: refusal message still names all five surfaces, so whichever one turns out to
#: be the cause is found in seconds rather than guessed at.
WIRE_SURFACES: Tuple[str, ...] = (
    "1. bridge/schema.json   — opponent properties AND the `required` list",
    "2. bridge/messages.py   — OpponentState fields + from_dict/to_dict + "
    "_validate_opponent's key tuple",
    "3. bridge/bot.js        — _snapshotOpponent() must EMIT both (on_ground read "
    "from the OPPONENT'S OWN bot connection, never the learner's view of the entity)",
    "4. bridge/bot.js        — assembleStateMsg() rebuilds the opponent block "
    "field-by-field; omitting them there STRIPS both before validation",
    "5. bridge/transport.js  — validateOpponent's requireExactKeys list",
)


class MirrorWireError(ValueError):
    """Raised when the opponent wire block cannot support a mirrored observation.

    A ``ValueError`` subclass so the env (T6) can let it propagate out of
    :meth:`OpponentMirror.observe` — which T6 calls from ``reset()``/``step()``,
    not from construction — as an ordinary config-level failure.
    """


def _surfaces_block() -> str:
    """Render :data:`WIRE_SURFACES` as an indented, one-per-line block."""
    return "\n".join(f"    {surface}" for surface in WIRE_SURFACES)


def require_opponent_wire_fields(opponent: Any) -> None:
    """Assert the opponent wire block carries every field the mirror needs.

    Checks each name in :data:`REQUIRED_OPPONENT_WIRE_FIELDS` is both PRESENT
    and non-``None``. ``None`` counts as absent on purpose: ``bool(None)`` is
    ``False`` and ``held_item_id(None)`` is ``0``, so a ``None`` that slipped
    through would be indistinguishable from a real "airborne, empty-handed"
    reading — the exact silent constant this check exists to prevent. An empty
    string ``held_item`` is NOT absent: ``bridge/messages.py`` documents ``""``
    as meaning an empty hand OR no connection to read from (a human
    challenger); either way it resolves to the ``"empty"`` vocabulary id, and
    self-play never hits the second case since both seats are bots.

    This only catches a field that is outright missing or ``None`` — surface 2
    (``OpponentState`` built without the fields) or a hand-rolled/replay
    object. It cannot catch surface 3 (see :data:`WIRE_SURFACES`):
    ``assembleStateMsg`` coerces an absent upstream value to ``False``/``""``
    rather than omitting the key, so that failure arrives here looking like a
    legitimate reading.

    Args:
        opponent: The ``state.opponent`` block (``bridge.messages.OpponentState``
            or any object exposing the same attributes).

    Raises:
        MirrorWireError: naming the missing field(s) and all five wire surfaces.
    """
    missing = [
        name
        for name in REQUIRED_OPPONENT_WIRE_FIELDS
        if getattr(opponent, name, None) is None
    ]
    if not missing:
        return

    raise MirrorWireError(
        "cannot mirror the opponent seat: state.opponent is missing "
        f"{missing} (got fields for a "
        f"{type(opponent).__name__}). The opponent-seat observation needs the "
        "opponent's OWN on_ground/held_item; substituting constants would pin "
        "two features for the whole match. FIVE surfaces must all carry these "
        "fields — check every one:\n" + _surfaces_block()
    )


def _seats(state: Any) -> Tuple[Any, Any]:
    """Return ``(learner_raw, opponent_raw)`` from a ``state`` message.

    Duck-typed rather than an ``isinstance`` check so tests (and any future
    replay/offline source) can supply a lightweight stand-in for ``StateMsg``.

    Raises:
        MirrorWireError: if ``state`` exposes neither seat.
    """
    if not hasattr(state, "self_state") or not hasattr(state, "opponent"):
        raise MirrorWireError(
            "state must expose `self_state` (learner) and `opponent` blocks, "
            f"like bridge.messages.StateMsg; got {type(state).__name__}"
        )
    return state.self_state, state.opponent


# ---------------------------------------------------------------------------
# Deferred rotation helper (see the module docstring: this import closes an
# import cycle if it is done at module level).
# ---------------------------------------------------------------------------

_ROTATE: Optional[Callable[[float, float, float, float], Tuple[float, float, float]]] = None


def _world_to_local_yaw(vx: float, vy: float, vz: float, yaw: float) -> Tuple[float, float, float]:
    """Rotate a world-frame vector into ``yaw``'s local frame (env's helper).

    Thin, memoized forwarder to ``env.mc_pvp_env._world_to_local_yaw`` so both
    seats rotate through the SAME code. A local re-implementation would be one
    sign flip away from mirroring the forward axis front-to-back, which no
    downstream check can see.
    """
    global _ROTATE
    if _ROTATE is None:
        from env.mc_pvp_env import _world_to_local_yaw as _impl

        _ROTATE = _impl
    return _ROTATE(vx, vy, vz, yaw)


# ---------------------------------------------------------------------------
# The mirror.
# ---------------------------------------------------------------------------


class OpponentMirror:
    """Builds the OPPONENT-seat observation from a learner-seat wire message.

    One instance per arena (per env). Call :meth:`reset` at every episode
    boundary and :meth:`observe` exactly once per decision window; the consumer
    (T6) caches the result and serves it from ``MCPvPEnv.opponent_observation()``.

    Args:
        perception_filter: The gate applied to the LEARNER as seen from the
            opponent's seat. Defaults to a fresh :class:`PerceptionFilter` with
            the frozen knobs. Pass one only to inject a custom ``los_clear`` or
            tuned FOV — and never the env's own filter instance: the two seats
            hold genuinely different memories, and sharing one object would let
            each seat refresh the other's "last seen" age.

    Attributes:
        perception_filter: The instance in use (exposed for the leak battery).
    """

    def __init__(self, perception_filter: Optional[PerceptionFilter] = None) -> None:
        self.perception_filter: PerceptionFilter = (
            perception_filter if perception_filter is not None else PerceptionFilter()
        )
        self._latest: Optional[np.ndarray] = None

    # -- lifecycle ---------------------------------------------------------

    def reset(self) -> None:
        """Clear per-episode state: the filter's memory and the cached vector.

        Both halves matter. A stale filter memory makes the frozen net start an
        episode already "remembering" the previous one's geometry; a stale
        cached vector would let a consumer read the last episode's observation
        before the new one's first state has arrived.
        """
        self.perception_filter.reset()
        self._latest = None

    # -- per-window observation -------------------------------------------

    def observe(
        self, state: "StateMsg", dt: float, opp_attack_cooldown: float
    ) -> np.ndarray:
        """Build (and cache) the opponent seat's observation for one window.

        Mirrors ``MCPvPEnv._state_to_obs`` with the seats exchanged: the SELF
        block is ``state.opponent``, and the gated block is the LEARNER as seen
        from the opponent's eyes.

        Not idempotent, by design: it advances the perception filter's memory
        age by ``dt``, exactly as the learner-seat path does. Call it once per
        decision window, in lockstep with ``_state_to_obs``; the caller owns any
        caching (:attr:`latest` serves repeat reads).

        Args:
            state: The raw ``state`` message for this window (learner seat).
            dt: Seconds elapsed this window. Advances the mirrored memory age;
                pass the env's decision interval, the same value the learner's
                filter receives.
            opp_attack_cooldown: The opponent's swing progress in ``[0, 1]``,
                from the env's shadow meter. There is no wire channel for it.

        Returns:
            ``np.ndarray`` of shape ``(OBS_DIM,)``, dtype float32, already
            passed through :func:`env.observation_spec.validate`.

        Raises:
            MirrorWireError: if ``state`` lacks a seat, or the opponent block
                lacks ``on_ground``/``held_item``.
            ValueError: if ``dt`` is negative, a raw vector is malformed, or the
                assembled vector fails validation (e.g. an out-of-range
                cooldown) — failing loudly beats feeding the frozen net garbage.
        """
        learner_raw, opponent_raw = _seats(state)
        require_opponent_wire_fields(opponent_raw)

        # GATE — role-generic filter, seats reversed: the opponent is the VIEWER
        # and the learner is the TARGET, so it rotates by the opponent's yaw and
        # tests the opponent's look vector/FOV. The target's health (the
        # learner's, here) is ignored by the filter (privileged), as in the
        # learner-seat path.
        gated_learner, derived = self.perception_filter.filter(
            opponent_raw, learner_raw, dt=dt
        )

        # SELF block — the opponent's own state, straight off the wire. The
        # velocity rotates about the OPPONENT's yaw (see the module docstring:
        # the learner's yaw here is the silent poison case).
        vel_local = _world_to_local_yaw(
            float(opponent_raw.velocity[0]),
            float(opponent_raw.velocity[1]),
            float(opponent_raw.velocity[2]),
            float(opponent_raw.yaw),
        )
        self_state = SelfState(
            health=float(opponent_raw.health),
            yaw=float(opponent_raw.yaw),
            pitch=float(opponent_raw.pitch),
            vel_local=vel_local,
            on_ground=bool(opponent_raw.on_ground),
            held_item=opponent_raw.held_item,
            attack_cooldown=float(opp_attack_cooldown),
        )

        obs = build_observation(self_state, gated_learner, derived)
        # Fail loudly on a malformed vector rather than feeding the frozen net
        # garbage — the learner-seat path makes the identical check.
        validate(obs)
        obs = obs.astype(OBS_DTYPE, copy=False)
        self._latest = obs
        return obs

    # -- cache -------------------------------------------------------------

    @property
    def latest(self) -> Optional[np.ndarray]:
        """The vector from the most recent :meth:`observe`, or ``None``.

        ``None`` before the first :meth:`observe` of an episode — including
        immediately after :meth:`reset`, so a consumer can tell "no state yet"
        from "state, but nothing visible". Returned by reference (repeat reads
        are byte-identical and allocate nothing); treat it as read-only.
        """
        return self._latest
