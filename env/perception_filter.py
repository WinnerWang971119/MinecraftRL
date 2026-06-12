"""perception_filter — FOV cone + raycast LoS + memory gating for opponent features.

This is the **soul module** (project-spec §2.2 / §5): it enforces *perceptual
parity* between the learned agent and a human. A protocol-reading bot is
omniscient by default — it knows the exact enemy position through walls. This
filter strips that omniscience (not the numbers) by exposing opponent
position/facing/velocity **only** when the opponent is inside the agent's
field-of-view cone AND a raycast line-of-sight is clear. When the opponent is
not currently visible it degrades the POSITION to last-seen *memory* with a
growing ``time_since_seen``, and after ``MEMORY_TTL`` to *absent*.

It produces the **gated** ``OpponentState`` + ``DerivedState`` that
``observation_spec.build_observation`` packs into the frozen observation vector;
this module is the *only* place gating happens (``observation_spec`` never
gates). The leak-detection battery (T13) hammers the single most important
fairness invariant implemented here: ``in_range`` / ``in_crosshair`` are nonzero
**only** when ``visible == 1``, so derived features never leak the opponent's
live position.

------------------------------------------------------------------------------
Visibility regimes (project-spec §5 "GATED -> MEMORY")
------------------------------------------------------------------------------
Each step the filter classifies the opponent into exactly one of three regimes:

  * **VISIBLE NOW** — opponent in the FOV cone AND LoS clear.
      ``visible = 1``; pos/facing/vel are the REAL values transformed into the
      agent's LOCAL frame (clamped to ±POS_SCALE on position); ``time_since_seen
      = 0``. Memory is refreshed (store last-seen local pos, reset the age).

  * **MEMORY** — not visible, but ``age <= MEMORY_TTL`` since the last sighting.
      ``visible = 0``; ``pos_local`` = the HELD last-seen value; facing and
      velocity are zeroed; ``time_since_seen = age`` (growing each step). The
      opponent's CURRENT real position is never exposed.

  * **ABSENT** — never seen, or ``age > MEMORY_TTL``.
      ``visible = 0``; pos/facing/vel all zero; ``time_since_seen`` at max
      (normalizes to ~1.0 in the obs).

------------------------------------------------------------------------------
Reconciliation: "MEMORY holds last-seen pos" vs "current position withheld"
------------------------------------------------------------------------------
These two rules are NOT in tension once you see *which* position each refers to.
MEMORY holds the **last-seen** local position — a stale snapshot captured at the
last frame the opponent was actually visible. It deliberately does NOT track the
opponent as it moves while unseen, so it reveals nothing about the opponent's
*current* whereabouts. The "withheld" invariant is about that current position:
while not visible, the live world position is never transformed, never exposed,
and the derived features (which are computed from the live values) are forced to
zero. So memory exposes a fading echo of where the opponent *was*, while the
current truth stays hidden — exactly the human experience of "I saw them duck
behind that wall a moment ago."

------------------------------------------------------------------------------
Coordinate / angle convention (documented contract for T9)
------------------------------------------------------------------------------
Inputs are the RAW bridge ``state`` world-frame values (see ``bridge.messages``):
positions ``[x, y, z]``, ``yaw`` / ``pitch`` in radians, velocities ``[x, y, z]``.

Minecraft world axes: ``+x`` east, ``+y`` up, ``+z`` south. Yaw is measured so
that yaw ``0`` looks toward ``+z`` and increases turning clockwise (toward
``-x``); pitch is positive looking down. The agent's unit LOOK vector is therefore::

    look = ( -cos(pitch) * sin(yaw),   # x
             -sin(pitch),              # y
              cos(pitch) * cos(yaw) )  # z

The agent LOCAL frame used for the observation is right-handed with:
  * ``+Z_local`` = the agent's forward look direction,
  * ``+X_local`` = the agent's right,
  * ``+Y_local`` = world up (the local frame only yaws with the agent; pitch is
    a separate sin/cos self-feature, so the local frame is a pure yaw rotation
    about world up — vertical offset is preserved on the local Y axis).

A world delta ``d = opp_pos - eye_pos`` maps to the local frame by the inverse
yaw rotation; with the yaw convention above this is::

    x_local =  d.x * cos(yaw) + d.z * sin(yaw)   # right
    y_local =  d.y                               # up (unchanged)
    z_local = -d.x * sin(yaw) + d.z * cos(yaw)   # forward

so an opponent dead ahead at distance ``r`` lands at ``(0, 0, r)``. Opponent yaw
is transformed to the agent-relative facing ``opp_yaw - self_yaw`` before being
handed to ``build_observation`` (which encodes it as sin/cos). Velocities are
rotated the same way as position deltas (yaw-only, no translation).

Owner: T12 (PerceptionFilter track)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from env.observation_spec import (
    MEMORY_TTL_SECONDS,
    POS_SCALE,
    DerivedState,
    OpponentState,
)

__all__ = [
    "FOV_DEGREES",
    "ATTACK_RANGE",
    "CROSSHAIR_DEGREES",
    "DEFAULT_DT",
    "RawState",
    "LosClear",
    "default_los_clear",
    "PerceptionFilter",
]


# ---------------------------------------------------------------------------
# Tunable module constants (project-spec §5 / open question Q4 — all ``TUNE``).
#
# These are the fairness knobs: FOV width and MEMORY_TTL double as the
# difficulty curriculum ("train first, blind down second"). MEMORY_TTL is NOT
# redefined here — it is imported from observation_spec so the filter and the
# obs normalization can never drift (the constant lives in the frozen contract).
# ---------------------------------------------------------------------------

#: Full field-of-view cone angle in degrees (project-spec §5 ≈ 70°, ``TUNE``).
#: The opponent is "in view" only if the angle between the agent's look vector
#: and the eye->opponent direction is <= FOV_DEGREES / 2.
FOV_DEGREES: float = 70.0

#: Melee attack range in blocks (project-spec ``in_range``, ≈ 3.0, ``TUNE``).
#: ``in_range`` is true iff the (currently visible) opponent is within this
#: eye->opponent distance. Vanilla 1.9+ reach is ~3 blocks.
ATTACK_RANGE: float = 3.0

#: Half-angle window (degrees) for ``in_crosshair`` (``TUNE``). The opponent is
#: "under the crosshair" iff the angle between the look vector and the
#: eye->opponent direction is <= CROSSHAIR_DEGREES. Tighter than the FOV cone.
CROSSHAIR_DEGREES: float = 10.0

#: Default per-step time advance (seconds). ACTION_REPEAT (≈ 4 ticks) / 20 ticks
#: per second = 0.2 s. The env (T9) passes its real decision interval; this is
#: only the convenience default.
DEFAULT_DT: float = 0.2

#: A "tick" of eye height above the feet position (blocks). Minecraft players
#: sense from eye level, not from their feet; both eyes are raised equally so the
#: vertical component of the look/LoS geometry is realistic. ``TUNE``.
_EYE_HEIGHT: float = 1.62


# ---------------------------------------------------------------------------
# Input contract.
#
# The filter consumes RAW world-frame state for the agent ("self") and the
# opponent. To keep T9 (the env) decoupled from a particular dataclass, the
# public ``filter()`` accepts EITHER:
#   * a ``RawState`` (the small dataclass below), or
#   * any object exposing ``.pos`` / ``.yaw`` / ``.pitch`` / ``.velocity``
#     (e.g. ``bridge.messages.SelfState`` / ``OpponentState`` straight off the
#     wire — note bridge OpponentState carries privileged health, which this
#     filter NEVER reads), or
#   * a plain mapping with ``"pos"`` / ``"yaw"`` / ``"pitch"`` / ``"velocity"``.
# All three are normalized through ``RawState.coerce`` below.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RawState:
    """RAW world-frame state for one entity (agent or opponent).

    Attributes:
        pos: ``[x, y, z]`` world-frame FEET position (blocks).
        yaw: Yaw in radians (Minecraft convention; see module docstring).
        pitch: Pitch in radians (positive looking down).
        velocity: ``[x, y, z]`` world-frame velocity (blocks/tick).
    """

    pos: Tuple[float, float, float]
    yaw: float
    pitch: float
    velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    @staticmethod
    def coerce(
        value: Union["RawState", Mapping[str, object], object]
    ) -> "RawState":
        """Normalize a RawState / wire dataclass / mapping into a ``RawState``.

        Accepts a ``RawState`` (returned as-is), a mapping with ``pos`` / ``yaw``
        / ``pitch`` / (optional) ``velocity`` keys, or any object exposing those
        as attributes (e.g. the bridge ``SelfState`` / ``OpponentState``).

        Raises:
            TypeError: if ``value`` exposes none of the supported shapes.
            ValueError: if ``pos`` or ``velocity`` is not a length-3 vector.
        """
        if isinstance(value, RawState):
            return value

        if isinstance(value, Mapping):
            pos = value["pos"]
            yaw = value["yaw"]
            pitch = value["pitch"]
            velocity = value.get("velocity", (0.0, 0.0, 0.0))
        elif (
            hasattr(value, "pos")
            and hasattr(value, "yaw")
            and hasattr(value, "pitch")
        ):
            pos = value.pos
            yaw = value.yaw
            pitch = value.pitch
            velocity = getattr(value, "velocity", (0.0, 0.0, 0.0))
        else:
            raise TypeError(
                "raw state must be a RawState, a mapping with pos/yaw/pitch/"
                "velocity, or an object exposing those attributes; got "
                f"{type(value).__name__}"
            )

        return RawState(
            pos=_as_vec3(pos, "pos"),
            yaw=float(yaw),
            pitch=float(pitch),
            velocity=_as_vec3(velocity, "velocity"),
        )


#: Signature of a pluggable line-of-sight test. Returns ``True`` when the
#: straight segment from ``eye_world_pos`` to ``target_world_pos`` is clear of
#: solid blocks (i.e. the target is visible). Both arguments are length-3
#: world-frame coordinate sequences.
LosClear = Callable[[Sequence[float], Sequence[float]], bool]


def default_los_clear(
    eye_world_pos: Sequence[float], target_world_pos: Sequence[float]
) -> bool:
    """Default LoS test for the open flat arena: nothing ever occludes.

    The kickoff arena is a flat, empty plane, so the line of sight is always
    clear. Tests (and, later, the real bridge raycast) inject their own
    ``los_clear`` to model obstacles.
    """
    return True


# ---------------------------------------------------------------------------
# Small vector helpers (float64 internally; the obs is float32 downstream).
# ---------------------------------------------------------------------------


def _as_vec3(values: Sequence[float], field_name: str) -> Tuple[float, float, float]:
    """Coerce a length-3 sequence to a tuple of 3 floats (raises on wrong length)."""
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.shape[0] != 3:
        raise ValueError(
            f"{field_name} must have exactly 3 components, got {arr.shape[0]}"
        )
    return (float(arr[0]), float(arr[1]), float(arr[2]))


def _look_vector(yaw: float, pitch: float) -> np.ndarray:
    """Unit look direction for the given yaw/pitch (Minecraft convention)."""
    cp = math.cos(pitch)
    return np.array(
        [-cp * math.sin(yaw), -math.sin(pitch), cp * math.cos(yaw)],
        dtype=np.float64,
    )


def _world_to_local_yaw(delta_world: np.ndarray, yaw: float) -> np.ndarray:
    """Rotate a world-frame delta into the agent's yaw-aligned LOCAL frame.

    Pure yaw rotation about world up (see module docstring): ``+Z_local`` is the
    agent's forward, ``+X_local`` its right, ``+Y_local`` world up.
    """
    s = math.sin(yaw)
    c = math.cos(yaw)
    dx, dy, dz = float(delta_world[0]), float(delta_world[1]), float(delta_world[2])
    return np.array(
        [
            dx * c + dz * s,   # right  (+X_local)
            dy,                # up     (+Y_local), preserved
            -dx * s + dz * c,  # forward(+Z_local)
        ],
        dtype=np.float64,
    )


def _eye_pos(state: RawState) -> np.ndarray:
    """World-frame eye position = feet position raised by the eye height."""
    return np.array(
        [state.pos[0], state.pos[1] + _EYE_HEIGHT, state.pos[2]], dtype=np.float64
    )


def _angle_between(u: np.ndarray, v: np.ndarray) -> float:
    """Angle in radians between two vectors. Returns ``pi`` for a zero vector.

    A zero direction (opponent exactly at the eye) is treated as maximally
    off-axis so it never spuriously counts as "in view"/"in crosshair".
    """
    nu = float(np.linalg.norm(u))
    nv = float(np.linalg.norm(v))
    if nu == 0.0 or nv == 0.0:
        return math.pi
    cos_theta = float(np.dot(u, v)) / (nu * nv)
    # Guard against fp drift pushing |cos| slightly past 1.0.
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return math.acos(cos_theta)


# ---------------------------------------------------------------------------
# The filter.
# ---------------------------------------------------------------------------


class PerceptionFilter:
    """Stateful FOV + line-of-sight + memory gate over raw opponent state.

    Construct one filter per arena/episode-stream and call :meth:`reset` at the
    start of every episode to clear memory. Each decision step, call
    :meth:`filter` with the raw self + raw opponent world-state and the step
    ``dt``; it returns the gated ``(OpponentState, DerivedState)`` pair that
    ``observation_spec.build_observation`` then packs.

    The filter holds exactly one piece of cross-step state: the last-seen
    opponent LOCAL position and its age. Everything else is recomputed per step.

    Args:
        fov_degrees: Full FOV cone angle (the opponent must be within half this
            of the look vector). Defaults to :data:`FOV_DEGREES`.
        memory_ttl: Seconds a last-seen position is held before the opponent
            drops to ABSENT. Defaults to the frozen :data:`MEMORY_TTL_SECONDS`.
        attack_range: ``in_range`` distance threshold in blocks. Defaults to
            :data:`ATTACK_RANGE`.
        crosshair_degrees: ``in_crosshair`` half-angle window in degrees.
            Defaults to :data:`CROSSHAIR_DEGREES`.
        los_clear: Pluggable line-of-sight test (see :data:`LosClear`). Defaults
            to :func:`default_los_clear` (always clear, for the open arena).

    Raises:
        ValueError: if any threshold is non-positive.
    """

    def __init__(
        self,
        fov_degrees: float = FOV_DEGREES,
        memory_ttl: float = MEMORY_TTL_SECONDS,
        attack_range: float = ATTACK_RANGE,
        crosshair_degrees: float = CROSSHAIR_DEGREES,
        los_clear: Optional[LosClear] = None,
    ) -> None:
        if fov_degrees <= 0.0:
            raise ValueError(f"fov_degrees must be > 0, got {fov_degrees}")
        if memory_ttl <= 0.0:
            raise ValueError(f"memory_ttl must be > 0, got {memory_ttl}")
        if attack_range <= 0.0:
            raise ValueError(f"attack_range must be > 0, got {attack_range}")
        if crosshair_degrees <= 0.0:
            raise ValueError(
                f"crosshair_degrees must be > 0, got {crosshair_degrees}"
            )

        self.fov_degrees = float(fov_degrees)
        self.memory_ttl = float(memory_ttl)
        self.attack_range = float(attack_range)
        self.crosshair_degrees = float(crosshair_degrees)
        self.los_clear: LosClear = los_clear if los_clear is not None else default_los_clear

        # Precompute the cone half-angle (radians) once.
        self._fov_half_rad = math.radians(self.fov_degrees) / 2.0
        self._crosshair_rad = math.radians(self.crosshair_degrees)

        # Cross-step memory (initialized by reset()).
        self._last_seen_local: Optional[np.ndarray] = None
        self._time_since_seen: float = float("inf")
        self.reset()

    # -- lifecycle ---------------------------------------------------------

    def reset(self) -> None:
        """Clear all memory at the start of an episode (project-spec §1 reset).

        After this the opponent is in the ABSENT regime until first sighted: no
        last-seen position is held and ``time_since_seen`` is effectively
        infinite (normalizes to the clamped max in the obs).
        """
        self._last_seen_local = None
        self._time_since_seen = float("inf")

    # -- per-step gate -----------------------------------------------------

    def filter(
        self,
        self_raw: Union[RawState, Mapping[str, object], object],
        opponent_raw: Union[RawState, Mapping[str, object], object],
        dt: float = DEFAULT_DT,
    ) -> Tuple[OpponentState, DerivedState]:
        """Gate one step of raw state into ``(OpponentState, DerivedState)``.

        Classifies the opponent into VISIBLE NOW / MEMORY / ABSENT (see module
        docstring), advances the memory age by ``dt``, and computes the derived
        features post-gating so they are nonzero only when ``visible == 1``.

        Args:
            self_raw: RAW world-frame agent state (RawState / wire dataclass /
                mapping; see :class:`RawState`).
            opponent_raw: RAW world-frame opponent state (same accepted shapes).
                Any opponent ``health`` on the input is IGNORED — privileged.
            dt: Seconds elapsed this step. Advances ``time_since_seen`` and the
                memory age. Must be >= 0.

        Returns:
            ``(OpponentState, DerivedState)`` already gated, ready for
            ``observation_spec.build_observation``.

        Raises:
            ValueError: if ``dt`` is negative or an input vector is malformed.
        """
        if dt < 0.0:
            raise ValueError(f"dt must be >= 0, got {dt}")

        me = RawState.coerce(self_raw)
        opp = RawState.coerce(opponent_raw)

        # Age existing memory first: time always advances by dt this step. If we
        # see the opponent below we reset it to 0; otherwise this is the age used.
        self._advance_age(dt)

        visible = self._is_visible(me, opp)

        if visible:
            return self._visible_now(me, opp)

        # Not visible this step: decide MEMORY vs ABSENT by the (already aged)
        # time since the last sighting.
        if self._last_seen_local is not None and self._time_since_seen <= self.memory_ttl:
            return self._memory()

        return self._absent()

    # -- regime builders ---------------------------------------------------

    def _visible_now(
        self, me: RawState, opp: RawState
    ) -> Tuple[OpponentState, DerivedState]:
        """VISIBLE-NOW regime: real local-frame values + refreshed memory."""
        eye = _eye_pos(me)
        opp_eye = _eye_pos(opp)
        delta_world = opp_eye - eye

        pos_local = _world_to_local_yaw(delta_world, me.yaw)
        # Clamp position components to ±POS_SCALE so the normalized obs (divided
        # by POS_SCALE) never exceeds [-1, 1] — a far opponent must not crash
        # observation_spec.validate (the documented fail-loud fix).
        pos_local_clamped = np.clip(pos_local, -POS_SCALE, POS_SCALE)

        vel_local = _world_to_local_yaw(
            np.asarray(opp.velocity, dtype=np.float64), me.yaw
        )
        facing_rel = opp.yaw - me.yaw

        # Refresh memory: store the (unclamped) last-seen local position and
        # reset the age. The unclamped value is kept so a re-sighting and a fresh
        # memory hold agree; it is clamped again on output if ever surfaced.
        self._last_seen_local = pos_local.copy()
        self._time_since_seen = 0.0

        opponent_state = OpponentState(
            pos_local=(
                float(pos_local_clamped[0]),
                float(pos_local_clamped[1]),
                float(pos_local_clamped[2]),
            ),
            facing_yaw=float(facing_rel),
            vel_local=(
                float(vel_local[0]),
                float(vel_local[1]),
                float(vel_local[2]),
            ),
            visible=True,
            time_since_seen=0.0,
        )

        # Derived features are computed from the REAL currently-visible geometry
        # (eye->opponent distance and look-vector angle) — and ONLY here, where
        # visible == True. In every other regime they are forced to 0.
        distance = float(np.linalg.norm(_eye_pos(opp) - eye))
        look = _look_vector(me.yaw, me.pitch)
        angle = _angle_between(look, delta_world)

        derived_state = DerivedState(
            in_range=distance <= self.attack_range,
            in_crosshair=angle <= self._crosshair_rad,
        )
        return opponent_state, derived_state

    def _memory(self) -> Tuple[OpponentState, DerivedState]:
        """MEMORY regime: held last-seen position; facing/vel zeroed; derived 0.

        The held position is re-clamped to ±POS_SCALE on output so a sighting
        near the arena edge that aged into memory still packs into [-1, 1].
        """
        assert self._last_seen_local is not None  # guarded by caller
        held = np.clip(self._last_seen_local, -POS_SCALE, POS_SCALE)
        opponent_state = OpponentState(
            pos_local=(float(held[0]), float(held[1]), float(held[2])),
            facing_yaw=0.0,            # spec degrades POSITION to memory, not facing
            vel_local=(0.0, 0.0, 0.0),  # ...nor velocity
            visible=False,
            time_since_seen=float(self._time_since_seen),
        )
        # visible == False -> derived features reveal nothing (fairness AC5).
        return opponent_state, DerivedState(in_range=False, in_crosshair=False)

    def _absent(self) -> Tuple[OpponentState, DerivedState]:
        """ABSENT regime: everything zeroed; time_since_seen at/above max."""
        opponent_state = OpponentState(
            pos_local=(0.0, 0.0, 0.0),
            facing_yaw=0.0,
            vel_local=(0.0, 0.0, 0.0),
            visible=False,
            # Report the TTL as the max so the obs normalizes to ~1.0 even when
            # the true age is infinite (never seen). build_observation clamps to
            # [0, 1], so any value >= TTL is equivalent here.
            time_since_seen=self.memory_ttl,
        )
        return opponent_state, DerivedState(in_range=False, in_crosshair=False)

    # -- gating primitives -------------------------------------------------

    def _is_visible(self, me: RawState, opp: RawState) -> bool:
        """True iff the opponent is inside the FOV cone AND the LoS is clear."""
        eye = _eye_pos(me)
        opp_eye = _eye_pos(opp)
        delta_world = opp_eye - eye

        # FOV cone: angle between the look vector and the eye->opponent direction
        # must be within the half-cone.
        look = _look_vector(me.yaw, me.pitch)
        if _angle_between(look, delta_world) > self._fov_half_rad:
            return False

        # Line of sight: pluggable raycast against solid blocks. Computed
        # eye-to-eye so the segment matches the FOV geometry.
        return bool(self.los_clear(eye.tolist(), opp_eye.tolist()))

    def _advance_age(self, dt: float) -> None:
        """Advance the memory age by ``dt`` (no-op once already infinite)."""
        if math.isinf(self._time_since_seen):
            return
        self._time_since_seen += dt
