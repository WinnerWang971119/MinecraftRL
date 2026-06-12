"""observation_spec — Frozen observation vector contract (single source of truth).

This module is **one of the four FROZEN CONTRACT artifacts** (AC1). It defines
the fixed-length ``float32`` observation vector, its frozen index map, the
normalization constants, and the helpers that pack/validate observations. Every
component that reads or writes an observation MUST import from here rather than
hardcode indices or normalization scales:

    - the Dueling-DRQN net (T14)        -> input dim asserts against ``OBS_DIM``
    - the Gym env (T9)                  -> assembles vectors via ``build_observation``
    - the PerceptionFilter (T12)        -> produces the *gated* opponent inputs
    - every future distributed actor    -> so train/deploy can never drift

------------------------------------------------------------------------------
Frozen layout (OBS_DIM = 23)
------------------------------------------------------------------------------
The exact indices below are **frozen**. ``tests/test_observation_spec.py``
asserts the index map has no gaps/overlaps, totals ``OBS_DIM``, and pins the
index of key fields, so any reordering breaks the test (the freeze guard).

    SELF block (FULL — always real, never gated)        indices  0..10
      health            [0]      scalar  health / MAX_HEALTH               -> [0, 1]
      yaw_sin           [1]      scalar  sin(yaw)                          -> [-1, 1]
      yaw_cos           [2]      scalar  cos(yaw)                          -> [-1, 1]
      pitch_sin         [3]      scalar  sin(pitch)                        -> [-1, 1]
      pitch_cos         [4]      scalar  cos(pitch)                        -> [-1, 1]
      vel_local         [5:8]    vec3    self velocity (local frame) / MAX_SPEED
      on_ground         [8]      scalar  0.0 / 1.0
      held_item         [9]      scalar  held_item_id / HELD_ITEM_VOCAB    -> [0, 1]
      attack_cooldown   [10]     scalar  swing progress 0..1 (BRIDGE-computed)

    OPPONENT block (GATED — real only when visible & LoS clear,            indices 11..20
                    else MEMORY, then absent; gating is done UPSTREAM by
                    the PerceptionFilter — this module never gates)
      opp_pos_local     [11:14]  vec3    opponent position (local frame) / POS_SCALE
      opp_facing_sin    [14]     scalar  sin(opponent yaw)                 -> [-1, 1]
      opp_facing_cos    [15]     scalar  cos(opponent yaw)                 -> [-1, 1]
      opp_vel_local     [16:19]  vec3    opponent velocity (local frame) / MAX_SPEED
      visible           [19]     scalar  1.0 if in FOV cone + clear raycast this step
      time_since_seen   [20]     scalar  seconds-since-seen / MEMORY_TTL   -> [0, 1] (clamped)

    DERIVED block (computed AFTER gating, from gated values only)          indices 21..22
      in_range          [21]     scalar  0.0 / 1.0
      in_crosshair      [22]     scalar  0.0 / 1.0

------------------------------------------------------------------------------
Frozen design choices (documented contract decisions)
------------------------------------------------------------------------------
- ``held_item`` is encoded as a **single normalized categorical id**
  (``held_item_id / HELD_ITEM_VOCAB_SIZE``), NOT one-hot. The plan's data model
  budgets one slot for it; a one-hot would change ``OBS_DIM``. Item ids are
  assigned by ``HELD_ITEM_VOCAB`` below (id 0 == empty hand / unknown).

- ``attack_cooldown`` is **BRIDGE-computed**, not derived in Python. The bridge
  measures swing timing as ``(current_tick - last_swing_tick) /
  weapon_attack_speed_ticks`` and reports a 0..1 progress (1.9+ combat). ATTACK
  uses raw ``bot.attack`` so this feature is genuinely observable. This module
  only normalizes/clamps the value it is handed.

- **Opponent health is NEVER in the observation.** It is privileged information:
  the reward (T5/T17) may read the raw true health, but it must never reach the
  agent's input. There is intentionally no slot for it here.

- **This module does NOT perform gating.** The PerceptionFilter (T12) decides
  visibility/memory and produces the opponent dataclass already gated. When the
  opponent is absent, callers pass an opponent state with ``visible == 0.0`` and
  zeroed position/facing/velocity (and ``time_since_seen`` at/near 1.0). The
  ``visible`` flag is what lets the net distinguish "unseen" from "value 0".

Owner: T2 (contract — PerceptionFilter/contract track)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, Mapping, Sequence, Tuple, Union

import numpy as np

__all__ = [
    "OBS_DIM",
    "OBS_DTYPE",
    "Obs",
    "FIELD_SLICES",
    "field_slice",
    "field_index",
    "MAX_HEALTH",
    "MAX_SPEED",
    "POS_SCALE",
    "MEMORY_TTL_SECONDS",
    "HELD_ITEM_VOCAB",
    "HELD_ITEM_VOCAB_SIZE",
    "held_item_id",
    "SelfState",
    "OpponentState",
    "DerivedState",
    "build_observation",
    "validate",
    "ObservationError",
]


# ---------------------------------------------------------------------------
# Numpy dtype of the observation vector. FROZEN: float32.
# ---------------------------------------------------------------------------
OBS_DTYPE = np.float32


# ---------------------------------------------------------------------------
# Normalization constants — the SINGLE source of truth.
#
# Every module that normalizes raw bridge values (env, perception filter,
# actors) MUST import these constants so the scales can never drift between
# training and deployment. Tuning any of these is a CONTRACT CHANGE: it must be
# done here and re-frozen, never re-derived locally.
# ---------------------------------------------------------------------------

#: Maximum survivable health in vanilla Minecraft (20 = 10 hearts). ``health``
#: is normalized as ``health / MAX_HEALTH`` -> [0, 1].
MAX_HEALTH: float = 20.0

#: Velocity normalization scale (blocks/tick). Sprint-jump peaks well under this;
#: per-axis velocity is divided by MAX_SPEED and may be clamped by ``validate``'s
#: tolerance. ``TUNE`` — re-freeze here if the action model changes.
MAX_SPEED: float = 1.0

#: Position normalization scale (blocks). Opponent local-frame position is
#: divided by POS_SCALE so a typical arena maps to ≈[-1, 1]. ``TUNE``.
POS_SCALE: float = 16.0

#: Memory time-to-live (seconds). ``time_since_seen`` is normalized as
#: ``seconds / MEMORY_TTL_SECONDS`` and clamped to [0, 1]; at 1.0 the opponent
#: has aged out of memory. Mirrors the PerceptionFilter ``MEMORY_TTL`` knob
#: (≈5 s, ``TUNE``) — keep the two in sync (filter imports this constant).
MEMORY_TTL_SECONDS: float = 5.0


# ---------------------------------------------------------------------------
# Held-item vocabulary. FROZEN ordering: id == index in this tuple. Id 0 is the
# empty hand / unknown sentinel so an absent item is a valid, in-range value.
# Extending the vocabulary appends to the END (never reorders) to keep prior
# ids stable across model checkpoints.
# ---------------------------------------------------------------------------
HELD_ITEM_VOCAB: Tuple[str, ...] = (
    "empty",          # 0 — empty hand / unknown
    "wooden_sword",   # 1
    "stone_sword",    # 2
    "iron_sword",     # 3
    "golden_sword",   # 4
    "diamond_sword",  # 5
    "netherite_sword",  # 6
    "wooden_axe",     # 7
    "stone_axe",      # 8
    "iron_axe",       # 9
    "diamond_axe",    # 10
    "netherite_axe",  # 11
    "shield",         # 12
    "bow",            # 13
    "crossbow",       # 14
    "fishing_rod",    # 15
)

#: Vocabulary size used to normalize the held-item id into [0, 1].
HELD_ITEM_VOCAB_SIZE: int = len(HELD_ITEM_VOCAB)

# Fast reverse lookup name -> id (built once at import).
_HELD_ITEM_TO_ID: Dict[str, int] = {name: i for i, name in enumerate(HELD_ITEM_VOCAB)}


def held_item_id(item: Union[str, int, None]) -> int:
    """Resolve a held-item identifier to its frozen integer id.

    Accepts a vocabulary name (e.g. ``"iron_sword"``), an already-resolved
    integer id, or ``None``/empty for an empty hand. Unknown names and
    out-of-range ids both map to id 0 (the ``"empty"`` / unknown sentinel) so a
    surprising item never produces an out-of-range observation value.
    """
    if item is None:
        return 0
    if isinstance(item, (int, np.integer)):
        idx = int(item)
        return idx if 0 <= idx < HELD_ITEM_VOCAB_SIZE else 0
    # String: normalize minecraft-style ``minecraft:iron_sword`` namespaces.
    name = str(item).split(":")[-1].strip().lower()
    return _HELD_ITEM_TO_ID.get(name, 0)


# ---------------------------------------------------------------------------
# Frozen index map.
#
# ``Obs`` is an IntEnum of the *start index* of every field (scalar fields are
# their own index; vector fields point at their first element). ``FIELD_SLICES``
# maps each logical field name to its ``slice`` in the vector so callers slice by
# name and never use magic numbers. Both are derived from a single ordered
# layout table below, which is the authoritative definition.
# ---------------------------------------------------------------------------

# (field_name, width). Order is the freeze; widths sum to OBS_DIM.
_LAYOUT: Tuple[Tuple[str, int], ...] = (
    # --- SELF (FULL) ---
    ("health", 1),
    ("yaw_sin", 1),
    ("yaw_cos", 1),
    ("pitch_sin", 1),
    ("pitch_cos", 1),
    ("vel_local", 3),
    ("on_ground", 1),
    ("held_item", 1),
    ("attack_cooldown", 1),
    # --- OPPONENT (GATED upstream) ---
    ("opp_pos_local", 3),
    ("opp_facing_sin", 1),
    ("opp_facing_cos", 1),
    ("opp_vel_local", 3),
    ("visible", 1),
    ("time_since_seen", 1),
    # --- DERIVED (post-gating) ---
    ("in_range", 1),
    ("in_crosshair", 1),
)


def _build_slices(layout: Tuple[Tuple[str, int], ...]) -> Dict[str, slice]:
    slices: Dict[str, slice] = {}
    cursor = 0
    for name, width in layout:
        slices[name] = slice(cursor, cursor + width)
        cursor += width
    return slices


#: Frozen mapping ``field name -> slice`` into the observation vector. Use this
#: (or ``field_slice`` / ``field_index``) to read/write fields by name.
FIELD_SLICES: Mapping[str, slice] = _build_slices(_LAYOUT)


#: Frozen total length of the observation vector. The net's input dimension
#: asserts against this. Computed from the layout so it can never disagree.
OBS_DIM: int = sum(width for _, width in _LAYOUT)


class Obs(IntEnum):
    """Frozen start-index enum for each observation field.

    For scalar fields the value is the field's index; for vector fields it is
    the index of the first element (use ``FIELD_SLICES`` / ``field_slice`` for
    the full span). Reference fields through this enum, never via literals.
    """

    HEALTH = FIELD_SLICES["health"].start
    YAW_SIN = FIELD_SLICES["yaw_sin"].start
    YAW_COS = FIELD_SLICES["yaw_cos"].start
    PITCH_SIN = FIELD_SLICES["pitch_sin"].start
    PITCH_COS = FIELD_SLICES["pitch_cos"].start
    VEL_LOCAL = FIELD_SLICES["vel_local"].start  # first of 3
    ON_GROUND = FIELD_SLICES["on_ground"].start
    HELD_ITEM = FIELD_SLICES["held_item"].start
    ATTACK_COOLDOWN = FIELD_SLICES["attack_cooldown"].start
    OPP_POS_LOCAL = FIELD_SLICES["opp_pos_local"].start  # first of 3
    OPP_FACING_SIN = FIELD_SLICES["opp_facing_sin"].start
    OPP_FACING_COS = FIELD_SLICES["opp_facing_cos"].start
    OPP_VEL_LOCAL = FIELD_SLICES["opp_vel_local"].start  # first of 3
    VISIBLE = FIELD_SLICES["visible"].start
    TIME_SINCE_SEEN = FIELD_SLICES["time_since_seen"].start
    IN_RANGE = FIELD_SLICES["in_range"].start
    IN_CROSSHAIR = FIELD_SLICES["in_crosshair"].start


def field_slice(name: str) -> slice:
    """Return the frozen ``slice`` for ``name`` (raises ``KeyError`` if unknown)."""
    return FIELD_SLICES[name]


def field_index(name: str) -> int:
    """Return the frozen start index for ``name`` (raises ``KeyError`` if unknown)."""
    return FIELD_SLICES[name].start


# ---------------------------------------------------------------------------
# Validation bounds.
#
# All features are normalized to ≈[-1, 1]. ``validate`` enforces this with a
# small tolerance to absorb floating-point error and slight clamp overshoot.
# ---------------------------------------------------------------------------
_VALID_LOW: float = -1.0
_VALID_HIGH: float = 1.0
#: Absolute tolerance applied to the [-1, 1] bound check (absorbs fp/clamp noise).
_VALID_TOL: float = 1e-4


class ObservationError(ValueError):
    """Raised by :func:`validate` when an observation vector is malformed."""


# ---------------------------------------------------------------------------
# Input dataclasses for build_observation.
#
# These mirror the three logical blocks. The caller (the env, via the
# PerceptionFilter for the opponent block) populates them. Vector fields are
# length-3 sequences in the agent's LOCAL frame; angles are passed as raw
# radians and encoded to (sin, cos) here.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SelfState:
    """FULL self-state inputs (always real, never gated).

    Attributes:
        health: Raw current health in [0, MAX_HEALTH]. Normalized by MAX_HEALTH.
        yaw: Yaw angle in radians. Encoded as (sin, cos).
        pitch: Pitch angle in radians. Encoded as (sin, cos).
        vel_local: (vx, vy, vz) velocity in the agent's LOCAL frame (blocks/tick).
            Normalized by MAX_SPEED.
        on_ground: Whether the agent is standing on ground.
        held_item: Vocabulary name, integer id, or None (see :func:`held_item_id`).
        attack_cooldown: BRIDGE-computed swing progress in [0, 1] (1.0 == ready).
    """

    health: float
    yaw: float
    pitch: float
    vel_local: Sequence[float]
    on_ground: bool
    held_item: Union[str, int, None]
    attack_cooldown: float


@dataclass(frozen=True)
class OpponentState:
    """GATED opponent-state inputs.

    The PerceptionFilter (T12) produces this **already gated**: when the opponent
    is not visible and has aged out of memory, the caller passes zeroed
    pos/facing/velocity with ``visible=False`` and ``time_since_seen`` near
    MEMORY_TTL_SECONDS. This module never decides visibility.

    Attributes:
        pos_local: (x, y, z) opponent position in the agent's LOCAL frame
            (blocks). Normalized by POS_SCALE. Zero when absent.
        facing_yaw: Opponent yaw in radians. Encoded as (sin, cos). Zero when absent.
        vel_local: (vx, vy, vz) opponent velocity in LOCAL frame (blocks/tick).
            Normalized by MAX_SPEED. Zero when absent.
        visible: True iff in FOV cone AND clear raycast this step.
        time_since_seen: Seconds since the opponent was last seen. Normalized by
            MEMORY_TTL_SECONDS and clamped to [0, 1].
    """

    pos_local: Sequence[float] = field(default_factory=lambda: (0.0, 0.0, 0.0))
    facing_yaw: float = 0.0
    vel_local: Sequence[float] = field(default_factory=lambda: (0.0, 0.0, 0.0))
    visible: bool = False
    time_since_seen: float = MEMORY_TTL_SECONDS


@dataclass(frozen=True)
class DerivedState:
    """DERIVED features, computed AFTER gating from gated values only.

    Leaking opponent position through a derived feature is the most common
    fairness bug (AC5); the PerceptionFilter computes these post-gating and the
    leak battery (T13) guards them. This module just packs the booleans.

    Attributes:
        in_range: Opponent within attack range (gated).
        in_crosshair: Opponent under the crosshair (gated).
    """

    in_range: bool = False
    in_crosshair: bool = False


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _as_vec3(values: Sequence[float], field_name: str) -> np.ndarray:
    """Coerce a length-3 sequence to a float64 ndarray (raises on wrong length)."""
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.shape[0] != 3:
        raise ValueError(
            f"{field_name} must have exactly 3 components, got {arr.shape[0]}"
        )
    return arr


def build_observation(
    self_state: SelfState,
    opponent_state: OpponentState,
    derived_state: DerivedState,
) -> np.ndarray:
    """Assemble the frozen ``float32`` observation vector of length ``OBS_DIM``.

    Packs already-gated inputs into the frozen layout. Responsibilities:
      - normalize raw magnitudes by the module-level constants,
      - encode angles as ``(sin, cos)``,
      - clamp ``time_since_seen`` into [0, 1],
      - resolve ``held_item`` to its normalized vocabulary id.

    This function does **not** gate opponent features (the PerceptionFilter does,
    upstream) and never reads opponent health (privileged → reward only).

    Args:
        self_state: FULL self-state (see :class:`SelfState`).
        opponent_state: GATED opponent-state (see :class:`OpponentState`).
        derived_state: Post-gating derived features (see :class:`DerivedState`).

    Returns:
        ``np.ndarray`` of shape ``(OBS_DIM,)`` and dtype ``float32``.

    Raises:
        ValueError: if a vector field does not have exactly 3 components.
    """
    vec = np.zeros(OBS_DIM, dtype=OBS_DTYPE)

    # --- SELF -------------------------------------------------------------
    vec[FIELD_SLICES["health"]] = float(self_state.health) / MAX_HEALTH

    yaw = float(self_state.yaw)
    pitch = float(self_state.pitch)
    vec[FIELD_SLICES["yaw_sin"]] = np.sin(yaw)
    vec[FIELD_SLICES["yaw_cos"]] = np.cos(yaw)
    vec[FIELD_SLICES["pitch_sin"]] = np.sin(pitch)
    vec[FIELD_SLICES["pitch_cos"]] = np.cos(pitch)

    vec[FIELD_SLICES["vel_local"]] = (
        _as_vec3(self_state.vel_local, "self_state.vel_local") / MAX_SPEED
    )
    vec[FIELD_SLICES["on_ground"]] = 1.0 if self_state.on_ground else 0.0
    vec[FIELD_SLICES["held_item"]] = (
        held_item_id(self_state.held_item) / HELD_ITEM_VOCAB_SIZE
    )
    vec[FIELD_SLICES["attack_cooldown"]] = float(self_state.attack_cooldown)

    # --- OPPONENT (already gated upstream) --------------------------------
    vec[FIELD_SLICES["opp_pos_local"]] = (
        _as_vec3(opponent_state.pos_local, "opponent_state.pos_local") / POS_SCALE
    )
    opp_yaw = float(opponent_state.facing_yaw)
    vec[FIELD_SLICES["opp_facing_sin"]] = np.sin(opp_yaw)
    vec[FIELD_SLICES["opp_facing_cos"]] = np.cos(opp_yaw)
    vec[FIELD_SLICES["opp_vel_local"]] = (
        _as_vec3(opponent_state.vel_local, "opponent_state.vel_local") / MAX_SPEED
    )
    vec[FIELD_SLICES["visible"]] = 1.0 if opponent_state.visible else 0.0

    # time_since_seen normalized and clamped to [0, 1].
    tss = float(opponent_state.time_since_seen) / MEMORY_TTL_SECONDS
    vec[FIELD_SLICES["time_since_seen"]] = min(max(tss, 0.0), 1.0)

    # --- DERIVED ----------------------------------------------------------
    vec[FIELD_SLICES["in_range"]] = 1.0 if derived_state.in_range else 0.0
    vec[FIELD_SLICES["in_crosshair"]] = 1.0 if derived_state.in_crosshair else 0.0

    return vec


def validate(vec: np.ndarray) -> None:
    """Validate an observation vector in place. **Raises** on any violation.

    This is the strict variant: it returns ``None`` on success and raises
    :class:`ObservationError` (a ``ValueError`` subclass) otherwise. Checks:
      - the vector is 1-D with exactly ``OBS_DIM`` elements,
      - the dtype is real-valued floating point,
      - no element is NaN or infinite,
      - every element lies within [-1, 1] (± a small tolerance).

    Args:
        vec: The candidate observation vector.

    Raises:
        ObservationError: if any of the above checks fail.
    """
    arr = np.asarray(vec)

    if arr.ndim != 1:
        raise ObservationError(
            f"observation must be 1-D, got ndim={arr.ndim} (shape={arr.shape})"
        )
    if arr.shape[0] != OBS_DIM:
        raise ObservationError(
            f"observation length {arr.shape[0]} != OBS_DIM ({OBS_DIM})"
        )
    if not np.issubdtype(arr.dtype, np.floating):
        raise ObservationError(
            f"observation dtype must be floating point, got {arr.dtype}"
        )
    if not np.all(np.isfinite(arr)):
        bad = np.where(~np.isfinite(arr))[0].tolist()
        raise ObservationError(f"observation has non-finite values at indices {bad}")

    lo = _VALID_LOW - _VALID_TOL
    hi = _VALID_HIGH + _VALID_TOL
    if np.any(arr < lo) or np.any(arr > hi):
        bad = np.where((arr < lo) | (arr > hi))[0].tolist()
        raise ObservationError(
            f"observation values out of range [{_VALID_LOW}, {_VALID_HIGH}] "
            f"at indices {bad}: {arr[bad].tolist()}"
        )
