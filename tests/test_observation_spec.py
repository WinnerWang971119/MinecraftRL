"""Tests for the frozen observation contract (T2 / AC1 / TC1).

These tests are the **freeze guard**. They assert:
  - the module imports and round-trips build -> validate (AC1, TC1),
  - the index map has no gaps/overlaps and totals OBS_DIM (TC1),
  - the exact frozen indices of key fields (so any reordering breaks a test),
  - validate() rejects wrong length, out-of-range, and NaN/inf vectors (TC1),
  - normalization and (sin, cos) encoding behave as documented.
"""

import math

import numpy as np
import pytest

from env import observation_spec as obs
from env.observation_spec import (
    OBS_DIM,
    OBS_DTYPE,
    FIELD_SLICES,
    Obs,
    ObservationError,
    SelfState,
    OpponentState,
    DerivedState,
    build_observation,
    validate,
    held_item_id,
    MAX_HEALTH,
    MAX_SPEED,
    POS_SCALE,
    MEMORY_TTL_SECONDS,
    HELD_ITEM_VOCAB,
    HELD_ITEM_VOCAB_SIZE,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers.
# ---------------------------------------------------------------------------


def _make_self(**overrides):
    base = dict(
        health=20.0,
        yaw=0.0,
        pitch=0.0,
        vel_local=(0.0, 0.0, 0.0),
        on_ground=True,
        held_item="iron_sword",
        attack_cooldown=1.0,
    )
    base.update(overrides)
    return SelfState(**base)


def _make_opp(**overrides):
    base = dict(
        pos_local=(1.0, 0.0, 2.0),
        facing_yaw=0.0,
        vel_local=(0.0, 0.0, 0.0),
        visible=True,
        time_since_seen=0.0,
    )
    base.update(overrides)
    return OpponentState(**base)


def _make_derived(**overrides):
    base = dict(in_range=False, in_crosshair=False)
    base.update(overrides)
    return DerivedState(**base)


def _build(**kw):
    return build_observation(
        kw.get("self_state", _make_self()),
        kw.get("opponent_state", _make_opp()),
        kw.get("derived_state", _make_derived()),
    )


# ---------------------------------------------------------------------------
# AC1: importable, frozen layout, build->validate round-trip.
# ---------------------------------------------------------------------------


def test_module_importable_and_roundtrips():
    """AC1: the module imports and a built vector passes validation."""
    vec = _build()
    assert isinstance(vec, np.ndarray)
    assert vec.dtype == OBS_DTYPE == np.float32
    validate(vec)  # must not raise


def test_obs_dim_is_frozen_constant():
    """OBS_DIM is the frozen integer 23 (the indicative layout sum)."""
    assert isinstance(OBS_DIM, int)
    assert OBS_DIM == 23


# ---------------------------------------------------------------------------
# TC1: index map has no gaps/overlaps and totals OBS_DIM.
# ---------------------------------------------------------------------------


def test_index_map_total_equals_obs_dim():
    total = sum(s.stop - s.start for s in FIELD_SLICES.values())
    assert total == OBS_DIM


def test_index_map_has_no_gaps_or_overlaps():
    """Slices must tile [0, OBS_DIM) exactly once with no gaps or overlaps."""
    covered = np.zeros(OBS_DIM, dtype=np.int32)
    for name, s in FIELD_SLICES.items():
        assert s.start >= 0, f"{name} starts below 0"
        assert s.stop <= OBS_DIM, f"{name} runs past OBS_DIM"
        assert s.stop > s.start, f"{name} is empty"
        covered[s] += 1
    # Every index covered exactly once -> no gaps (0) and no overlaps (>1).
    assert np.all(covered == 1), (
        f"index map not a clean partition: {covered.tolist()}"
    )


def test_obs_dim_matches_vector_length():
    assert len(_build()) == OBS_DIM


# ---------------------------------------------------------------------------
# Freeze guard: exact frozen indices of key fields.
# Reordering the layout MUST break these.
# ---------------------------------------------------------------------------


def test_frozen_key_field_indices():
    assert Obs.HEALTH == 0
    assert FIELD_SLICES["health"] == slice(0, 1)

    # SELF block boundaries.
    assert Obs.YAW_SIN == 1
    assert FIELD_SLICES["vel_local"] == slice(5, 8)
    assert Obs.ON_GROUND == 8
    assert Obs.HELD_ITEM == 9
    assert Obs.ATTACK_COOLDOWN == 10

    # OPPONENT block.
    assert FIELD_SLICES["opp_pos_local"] == slice(11, 14)
    assert FIELD_SLICES["opp_vel_local"] == slice(16, 19)
    assert Obs.VISIBLE == 19
    assert Obs.TIME_SINCE_SEEN == 20

    # DERIVED block (last two).
    assert Obs.IN_RANGE == 21
    assert Obs.IN_CROSSHAIR == 22


def test_no_opponent_health_field():
    """Opponent health is privileged and must never appear in the index map."""
    for name in FIELD_SLICES:
        assert "health" not in name or name == "health", (
            f"unexpected health-like opponent field: {name}"
        )
    # Only the SELF health field exists.
    health_fields = [n for n in FIELD_SLICES if n.endswith("health")]
    assert health_fields == ["health"]


# ---------------------------------------------------------------------------
# TC1: validate rejects malformed vectors.
# ---------------------------------------------------------------------------


def test_validate_rejects_wrong_length():
    with pytest.raises(ObservationError):
        validate(np.zeros(OBS_DIM - 1, dtype=np.float32))
    with pytest.raises(ObservationError):
        validate(np.zeros(OBS_DIM + 1, dtype=np.float32))


def test_validate_rejects_wrong_ndim():
    with pytest.raises(ObservationError):
        validate(np.zeros((OBS_DIM, 1), dtype=np.float32))


def test_validate_rejects_nan():
    vec = _build()
    vec[Obs.HEALTH] = np.nan
    with pytest.raises(ObservationError):
        validate(vec)


def test_validate_rejects_inf():
    vec = _build()
    vec[Obs.YAW_SIN] = np.inf
    with pytest.raises(ObservationError):
        validate(vec)


def test_validate_rejects_out_of_range():
    vec = _build()
    vec[Obs.ATTACK_COOLDOWN] = 2.5  # well above 1.0
    with pytest.raises(ObservationError):
        validate(vec)

    vec = _build()
    vec[Obs.YAW_SIN] = -3.0  # well below -1.0
    with pytest.raises(ObservationError):
        validate(vec)


def test_validate_rejects_non_float_dtype():
    with pytest.raises(ObservationError):
        validate(np.zeros(OBS_DIM, dtype=np.int32))


def test_validate_accepts_boundary_values():
    """Exactly ±1 (within tolerance) is accepted."""
    vec = np.zeros(OBS_DIM, dtype=np.float32)
    vec[:] = 1.0
    validate(vec)
    vec[:] = -1.0
    validate(vec)


# ---------------------------------------------------------------------------
# Normalization and encoding behavior.
# ---------------------------------------------------------------------------


def test_health_normalized_by_max_health():
    vec = _build(self_state=_make_self(health=MAX_HEALTH))
    assert vec[Obs.HEALTH] == pytest.approx(1.0)
    vec = _build(self_state=_make_self(health=MAX_HEALTH / 2.0))
    assert vec[Obs.HEALTH] == pytest.approx(0.5)
    vec = _build(self_state=_make_self(health=0.0))
    assert vec[Obs.HEALTH] == pytest.approx(0.0)


def test_yaw_pitch_encoded_as_sin_cos():
    vec = _build(self_state=_make_self(yaw=math.pi / 2.0, pitch=0.0))
    assert vec[Obs.YAW_SIN] == pytest.approx(1.0, abs=1e-6)
    assert vec[Obs.YAW_COS] == pytest.approx(0.0, abs=1e-6)
    assert vec[Obs.PITCH_SIN] == pytest.approx(0.0, abs=1e-6)
    assert vec[Obs.PITCH_COS] == pytest.approx(1.0, abs=1e-6)


def test_velocity_normalized_by_max_speed():
    vel = (MAX_SPEED, 0.0, -MAX_SPEED)
    vec = _build(self_state=_make_self(vel_local=vel))
    s = FIELD_SLICES["vel_local"]
    np.testing.assert_allclose(vec[s], [1.0, 0.0, -1.0], atol=1e-6)


def test_velocity_clamped_when_exceeding_max_speed():
    # Falling velocity routinely exceeds MAX_SPEED on the vertical axis. It must
    # saturate at the [-1, 1] bound instead of overshooting and tripping validate
    # (the live multi-arena crash: vel_local[1] == -1.19 raised ObservationError).
    vel = (3.0 * MAX_SPEED, -1.19 * MAX_SPEED, -3.0 * MAX_SPEED)
    self_state = _make_self(vel_local=vel)
    opp_state = _make_opp(vel_local=vel, visible=True)
    vec = _build(self_state=self_state, opponent_state=opp_state)
    np.testing.assert_allclose(vec[FIELD_SLICES["vel_local"]], [1.0, -1.0, -1.0])
    np.testing.assert_allclose(vec[FIELD_SLICES["opp_vel_local"]], [1.0, -1.0, -1.0])
    validate(vec)  # must not raise


def test_opponent_position_normalized_by_pos_scale():
    pos = (POS_SCALE, 0.0, -POS_SCALE)
    vec = _build(opponent_state=_make_opp(pos_local=pos))
    s = FIELD_SLICES["opp_pos_local"]
    np.testing.assert_allclose(vec[s], [1.0, 0.0, -1.0], atol=1e-6)


def test_on_ground_and_visible_flags():
    vec = _build(
        self_state=_make_self(on_ground=False),
        opponent_state=_make_opp(visible=False),
    )
    assert vec[Obs.ON_GROUND] == 0.0
    assert vec[Obs.VISIBLE] == 0.0

    vec = _build(
        self_state=_make_self(on_ground=True),
        opponent_state=_make_opp(visible=True),
    )
    assert vec[Obs.ON_GROUND] == 1.0
    assert vec[Obs.VISIBLE] == 1.0


def test_time_since_seen_normalized_and_clamped():
    # Half the TTL -> 0.5.
    vec = _build(opponent_state=_make_opp(time_since_seen=MEMORY_TTL_SECONDS / 2.0))
    assert vec[Obs.TIME_SINCE_SEEN] == pytest.approx(0.5)
    # Beyond TTL -> clamped to 1.0 (stays in range so validate passes).
    vec = _build(opponent_state=_make_opp(time_since_seen=MEMORY_TTL_SECONDS * 10.0))
    assert vec[Obs.TIME_SINCE_SEEN] == pytest.approx(1.0)
    validate(vec)


def test_derived_flags_packed():
    vec = _build(derived_state=_make_derived(in_range=True, in_crosshair=True))
    assert vec[Obs.IN_RANGE] == 1.0
    assert vec[Obs.IN_CROSSHAIR] == 1.0


# ---------------------------------------------------------------------------
# held_item id resolution.
# ---------------------------------------------------------------------------


def test_held_item_id_resolution():
    assert held_item_id("empty") == 0
    assert held_item_id(None) == 0
    assert held_item_id("iron_sword") == HELD_ITEM_VOCAB.index("iron_sword")
    # Namespaced + case-insensitive.
    assert held_item_id("minecraft:IRON_SWORD") == HELD_ITEM_VOCAB.index("iron_sword")
    # Unknown name and out-of-range id both fall back to the sentinel 0.
    assert held_item_id("totally_unknown_item") == 0
    assert held_item_id(9999) == 0
    # Pass-through of a valid integer id.
    assert held_item_id(3) == 3


def test_held_item_normalized_into_range():
    last_id = HELD_ITEM_VOCAB_SIZE - 1
    vec = _build(self_state=_make_self(held_item=last_id))
    assert 0.0 <= vec[Obs.HELD_ITEM] <= 1.0
    assert vec[Obs.HELD_ITEM] == pytest.approx(last_id / HELD_ITEM_VOCAB_SIZE)
    validate(vec)


def test_held_item_vocab_first_is_empty_sentinel():
    assert HELD_ITEM_VOCAB[0] == "empty"


# ---------------------------------------------------------------------------
# Vector-field length guards.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_vel", [(1.0, 2.0), (1.0, 2.0, 3.0, 4.0)])
def test_build_rejects_wrong_vec3_length(bad_vel):
    with pytest.raises(ValueError):
        _build(self_state=_make_self(vel_local=bad_vel))


# ---------------------------------------------------------------------------
# Full round-trip over a randomized but in-range battery.
# ---------------------------------------------------------------------------


def test_randomized_roundtrip_stays_valid():
    rng = np.random.default_rng(1234)
    for _ in range(200):
        s = SelfState(
            health=float(rng.uniform(0.0, MAX_HEALTH)),
            yaw=float(rng.uniform(-math.pi, math.pi)),
            pitch=float(rng.uniform(-math.pi / 2, math.pi / 2)),
            vel_local=tuple(rng.uniform(-MAX_SPEED, MAX_SPEED, size=3)),
            on_ground=bool(rng.integers(0, 2)),
            held_item=int(rng.integers(0, HELD_ITEM_VOCAB_SIZE)),
            attack_cooldown=float(rng.uniform(0.0, 1.0)),
        )
        o = OpponentState(
            pos_local=tuple(rng.uniform(-POS_SCALE, POS_SCALE, size=3)),
            facing_yaw=float(rng.uniform(-math.pi, math.pi)),
            vel_local=tuple(rng.uniform(-MAX_SPEED, MAX_SPEED, size=3)),
            visible=bool(rng.integers(0, 2)),
            time_since_seen=float(rng.uniform(0.0, MEMORY_TTL_SECONDS)),
        )
        d = DerivedState(
            in_range=bool(rng.integers(0, 2)),
            in_crosshair=bool(rng.integers(0, 2)),
        )
        vec = build_observation(s, o, d)
        validate(vec)


def test_default_absent_opponent_is_valid():
    """The PerceptionFilter's 'absent' opponent (all defaults) packs and validates."""
    vec = build_observation(_make_self(), OpponentState(), DerivedState())
    validate(vec)
    assert vec[Obs.VISIBLE] == 0.0
    assert vec[Obs.TIME_SINCE_SEEN] == pytest.approx(1.0)  # TTL/TTL, clamped
    s = FIELD_SLICES["opp_pos_local"]
    np.testing.assert_allclose(vec[s], [0.0, 0.0, 0.0])
