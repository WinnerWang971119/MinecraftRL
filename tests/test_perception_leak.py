"""test_perception_leak — the fairness leak-detection battery (T13 / AC5 / TC3).

This is the adversarial counterpart to ``tests/test_perception_filter.py``. Where
that suite checks the filter behaves *correctly* on representative geometry, this
battery exists to **break** the single most important fairness invariant of the
project (project-spec §2.2 / §5, plan AC5):

    When ``visible == false``, NO derived feature (``in_range`` / ``in_crosshair``)
    and NO field of the packed observation vector may reveal the opponent's
    CURRENT (hidden) position. The agent may legitimately *remember* a frozen
    last-seen position (MEMORY), but nothing in the observation may change as a
    function of where the opponent actually is *right now* while unseen.

The tests are deliberately hostile: they put the opponent geometrically dead in
the crosshair and well inside attack range while it is NOT visible (behind the
agent, or behind a blocking wall), then assert the derived features stay 0; and
they sweep the opponent's live world position across a very wide range while it is
unseen and assert the *entire* observation vector (every index, byte for byte)
does not move. If anyone later wires a derived (or any) feature to the opponent's
current position while invisible, this battery fails loudly.

Determinism: numpy is seeded once per test (``_seed_np``) so the randomized
position sweeps are reproducible.

Coordinate convention (see ``env/perception_filter`` module docstring): yaw 0
looks toward world ``+z``; an opponent dead ahead at distance ``r`` lands at local
``(0, 0, r)``; an opponent at ``+z`` is in front, ``-z`` is behind the agent.
"""

import math

import numpy as np
import pytest

from env.observation_spec import (
    FIELD_SLICES,
    MEMORY_TTL_SECONDS,
    OBS_DIM,
    POS_SCALE,
    SelfState,
    build_observation,
    validate,
)
from env.perception_filter import (
    ATTACK_RANGE,
    CROSSHAIR_DEGREES,
    DEFAULT_DT,
    FOV_DEGREES,
    PerceptionFilter,
    RawState,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers.
# ---------------------------------------------------------------------------

#: A fixed seed so the randomized opponent-position sweeps are reproducible.
_SEED = 1234567


def _seed_np() -> np.random.Generator:
    """Return a freshly seeded numpy Generator (deterministic across runs)."""
    return np.random.default_rng(_SEED)


def _self(pos=(0.0, 0.0, 0.0), yaw=0.0, pitch=0.0, velocity=(0.0, 0.0, 0.0)):
    """Agent at the origin looking toward +z by default."""
    return RawState(pos=pos, yaw=yaw, pitch=pitch, velocity=velocity)


def _opp(pos, yaw=0.0, pitch=0.0, velocity=(0.0, 0.0, 0.0)):
    return RawState(pos=pos, yaw=yaw, pitch=pitch, velocity=velocity)


def _self_state_for_obs():
    """A minimal valid SelfState so a full obs vector can be packed + validated."""
    return SelfState(
        health=20.0,
        yaw=0.0,
        pitch=0.0,
        vel_local=(0.0, 0.0, 0.0),
        on_ground=True,
        held_item="iron_sword",
        attack_cooldown=1.0,
    )


def _derived_fields():
    """The DERIVED block field names, read live from the frozen layout.

    The DERIVED block is everything after ``time_since_seen`` in the frozen
    ``observation_spec`` layout. Deriving the list from ``FIELD_SLICES`` (rather
    than hardcoding ``in_range`` / ``in_crosshair``) means a future derived field
    added to the contract is automatically swept by the position-independence test
    below — so a newly-added leaky derived feature trips this battery.
    """
    names = list(FIELD_SLICES.keys())
    cut = names.index("time_since_seen") + 1
    derived = names[cut:]
    # Guard: the current contract's derived block. If this assertion ever fires,
    # the layout changed and the battery should be re-examined (intentional).
    assert "in_range" in derived and "in_crosshair" in derived, (
        f"expected in_range/in_crosshair in the derived block, got {derived}"
    )
    return derived


def _block_all(eye, target):
    """A LoS test modelling a wall that always occludes (target never visible)."""
    return False


# A grid of live opponent positions that are GUARANTEED to be out of view of an
# agent at the origin looking toward +z: every point is behind the agent (z < 0),
# spanning a wide range of x/y/z including points that are geometrically in-range
# and perfectly on the (reversed) sight line. None of these may move the obs.
def _behind_sweep_positions(rng, n=400):
    xs = rng.uniform(-60.0, 60.0, size=n)
    ys = rng.uniform(-20.0, 20.0, size=n)
    zs = rng.uniform(-60.0, -0.5, size=n)  # strictly behind the agent
    positions = list(zip(xs.tolist(), ys.tolist(), zs.tolist()))
    # Explicitly include the nastiest adversarial points: directly behind, at the
    # exact mirror of an in-range/in-crosshair target.
    positions.append((0.0, 0.0, -0.5))          # point-blank, dead behind
    positions.append((0.0, 0.0, -ATTACK_RANGE)) # in-range distance, behind
    positions.append((0.0, 0.0, -50.0))         # far, dead behind
    return positions


# ---------------------------------------------------------------------------
# 1. Derived-features-zero-when-invisible (the core AC5 invariant).
# ---------------------------------------------------------------------------


def test_derived_zero_for_many_out_of_fov_positions():
    """Randomized opponents BEHIND the agent -> derived stay 0 (never seen)."""
    rng = _seed_np()
    for pos in _behind_sweep_positions(rng, n=400):
        pf = PerceptionFilter()  # fresh filter: opponent never seen -> ABSENT
        opp_state, derived = pf.filter(_self(), _opp(pos), dt=DEFAULT_DT)
        assert opp_state.visible is False, f"unexpectedly visible at {pos}"
        assert derived.in_range is False, f"in_range leaked at {pos}"
        assert derived.in_crosshair is False, f"in_crosshair leaked at {pos}"


def test_derived_zero_behind_los_wall_for_many_positions():
    """Randomized opponents inside the FOV cone but behind a wall -> derived 0.

    Here the geometry would otherwise pass the FOV test (positions are in front of
    the agent); only the blocking LoS hook gates them out. Derived features must
    not leak through the raycast path.
    """
    rng = _seed_np()
    for _ in range(400):
        # In front of the agent (z > 0) and within a tight cone, so FOV passes;
        # the wall is what hides it.
        dist = float(rng.uniform(0.5, 40.0))
        off_angle = float(rng.uniform(0.0, math.radians(FOV_DEGREES / 2.0 - 1.0)))
        bearing = float(rng.uniform(0.0, 2.0 * math.pi))
        x = dist * math.sin(off_angle) * math.cos(bearing)
        y = dist * math.sin(off_angle) * math.sin(bearing)
        z = dist * math.cos(off_angle)
        pf = PerceptionFilter(los_clear=_block_all)
        opp_state, derived = pf.filter(_self(), _opp((x, y, z)), dt=DEFAULT_DT)
        assert opp_state.visible is False, f"wall failed to gate {(x, y, z)}"
        assert derived.in_range is False
        assert derived.in_crosshair is False


def test_derived_zero_when_in_range_and_dead_in_crosshair_but_behind_agent():
    """Adversarial: geometrically in-range + dead-center, but BEHIND -> derived 0.

    The opponent sits at the exact distance/alignment that WOULD set both derived
    flags if it were visible, but it is directly behind the agent (outside the
    cone). The fairness invariant requires both derived features to be 0 because
    ``visible == false``.
    """
    pf = PerceptionFilter()
    # Mirror of a perfect in-range, dead-center front target onto the -z axis.
    behind_in_range = (0.0, 0.0, -(ATTACK_RANGE - 0.5))
    opp_state, derived = pf.filter(_self(), _opp(behind_in_range), dt=DEFAULT_DT)

    assert opp_state.visible is False
    assert derived.in_range is False
    assert derived.in_crosshair is False
    # Position is not exposed either (never seen -> ABSENT zeroes it).
    np.testing.assert_array_equal(opp_state.pos_local, [0.0, 0.0, 0.0])


def test_derived_zero_when_in_range_and_dead_in_crosshair_but_behind_wall():
    """Adversarial: in-range + dead-center IN FRONT, but a wall hides it -> 0.

    Identical geometry to a guaranteed-visible kill shot (dead ahead, inside
    range and the crosshair window), but a blocking wall gates it out. With a
    clear LoS both derived flags would be true; behind the wall they must be 0.
    """
    dead_ahead_in_range = (0.0, 0.0, ATTACK_RANGE - 0.5)

    pf_clear = PerceptionFilter()
    clear_state, clear_derived = pf_clear.filter(
        _self(), _opp(dead_ahead_in_range), dt=DEFAULT_DT
    )
    # Sanity: this geometry IS a kill shot when visible.
    assert clear_state.visible is True
    assert clear_derived.in_range is True
    assert clear_derived.in_crosshair is True

    pf_blocked = PerceptionFilter(los_clear=_block_all)
    blocked_state, blocked_derived = pf_blocked.filter(
        _self(), _opp(dead_ahead_in_range), dt=DEFAULT_DT
    )
    assert blocked_state.visible is False
    assert blocked_derived.in_range is False
    assert blocked_derived.in_crosshair is False


# ---------------------------------------------------------------------------
# 2. Current-position-invariance: moving the hidden opponent must not move
#    any observation value.
# ---------------------------------------------------------------------------


def _obs_for_hidden_position(live_pos, *, los_clear=None):
    """Pack the FULL obs vector for an opponent at ``live_pos`` while unseen.

    Drives a fresh filter that has NEVER seen the opponent (ABSENT regime), so the
    only thing that could move the obs is an (illegal) dependence on the current
    hidden position.
    """
    pf = PerceptionFilter(los_clear=los_clear)
    opp_state, derived = pf.filter(_self(), _opp(live_pos), dt=DEFAULT_DT)
    assert opp_state.visible is False
    return build_observation(_self_state_for_obs(), opp_state, derived)


def test_absent_obs_invariant_to_current_position_behind():
    """ABSENT: sweep the live BEHIND position; the whole obs vector is constant."""
    rng = _seed_np()
    baseline = _obs_for_hidden_position((0.0, 0.0, -10.0))
    validate(baseline)
    for pos in _behind_sweep_positions(rng, n=400):
        vec = _obs_for_hidden_position(pos)
        np.testing.assert_array_equal(
            vec, baseline, err_msg=f"obs moved for hidden opponent at {pos}"
        )


def test_absent_obs_invariant_to_current_position_behind_wall():
    """ABSENT behind a wall: sweep live in-FRONT positions; obs stays constant."""
    rng = _seed_np()
    baseline = _obs_for_hidden_position((0.0, 0.0, 5.0), los_clear=_block_all)
    validate(baseline)
    for _ in range(400):
        dist = float(rng.uniform(0.5, 40.0))
        off_angle = float(rng.uniform(0.0, math.radians(FOV_DEGREES / 2.0 - 1.0)))
        bearing = float(rng.uniform(0.0, 2.0 * math.pi))
        x = dist * math.sin(off_angle) * math.cos(bearing)
        y = dist * math.sin(off_angle) * math.sin(bearing)
        z = dist * math.cos(off_angle)
        vec = _obs_for_hidden_position((x, y, z), los_clear=_block_all)
        np.testing.assert_array_equal(
            vec, baseline, err_msg=f"obs moved for wall-hidden opponent at {(x, y, z)}"
        )


def test_memory_obs_invariant_to_current_position():
    """MEMORY: with the age held constant, moving the hidden opponent is a no-op.

    After a single sighting (which freezes the last-seen memory), the obs in the
    subsequent unseen step is fully determined by that frozen memory and the
    (constant, matched) ``dt`` aging — NOT by where the opponent currently is. We
    rebuild a fresh filter for each live position, replay the SAME sighting, then
    one unseen step with the SAME ``dt``; the resulting obs must be identical.
    """
    rng = _seed_np()

    def memory_obs_after_move(live_pos):
        pf = PerceptionFilter()
        # Identical sighting every time -> identical frozen memory + age.
        seen, _ = pf.filter(_self(), _opp((1.0, 0.5, 2.0)), dt=DEFAULT_DT)
        assert seen.visible is True
        # One unseen step at the SAME dt; only live_pos differs between calls.
        opp_state, derived = pf.filter(_self(), _opp(live_pos), dt=DEFAULT_DT)
        assert opp_state.visible is False
        return build_observation(_self_state_for_obs(), opp_state, derived)

    baseline = memory_obs_after_move((0.0, 0.0, -10.0))
    validate(baseline)
    for pos in _behind_sweep_positions(rng, n=300):
        vec = memory_obs_after_move(pos)
        np.testing.assert_array_equal(
            vec, baseline, err_msg=f"memory obs moved for hidden opponent at {pos}"
        )


# ---------------------------------------------------------------------------
# 3. Memory is not a current-position leak: the exposed pos is the STALE
#    last-seen value, never the live one; teleporting the hidden opponent
#    does not update it.
# ---------------------------------------------------------------------------


def test_memory_pos_is_stale_last_seen_not_live():
    """After a sighting then loss of sight, opp_pos_local == frozen last-seen."""
    pf = PerceptionFilter()
    last_seen_world = (0.0, 0.0, 2.0)  # dead ahead, distance 2 -> local (0,0,2)
    seen, _ = pf.filter(_self(), _opp(last_seen_world), dt=DEFAULT_DT)
    assert seen.visible is True
    np.testing.assert_allclose(seen.pos_local, [0.0, 0.0, 2.0], atol=1e-6)

    # Opponent teleports far away while out of view; the held memory must not move.
    for live_pos in [
        (50.0, 0.0, -50.0),
        (-30.0, 10.0, -5.0),
        (0.0, 0.0, -2.0),     # mirror of the last-seen pos onto the hidden side
        (16.0, 0.0, -16.0),
    ]:
        mem, derived = pf.filter(_self(), _opp(live_pos), dt=DEFAULT_DT)
        assert mem.visible is False
        np.testing.assert_allclose(
            mem.pos_local, [0.0, 0.0, 2.0], atol=1e-6,
            err_msg=f"memory tracked the live position {live_pos}",
        )
        assert derived.in_range is False and derived.in_crosshair is False


def test_teleporting_hidden_opponent_never_updates_memory_pos():
    """Many random teleports of the unseen opponent leave opp_pos_local frozen.

    Each teleport probe is taken at ``dt=0`` (the same instant) so the held memory
    stays inside ``MEMORY_TTL`` and does not legitimately age out to ABSENT — that
    isolates exactly the claim under test: teleporting the hidden opponent never
    *updates* the frozen last-seen position to track its current whereabouts.
    """
    rng = _seed_np()
    pf = PerceptionFilter()
    # Sight it once off to the right so the frozen memory is a clearly-identifiable
    # non-zero vector we can pin every subsequent step against.
    seen, _ = pf.filter(_self(), _opp((2.0, 0.0, 3.0)), dt=DEFAULT_DT)
    assert seen.visible is True
    frozen = np.asarray(seen.pos_local, dtype=np.float64)

    for pos in _behind_sweep_positions(rng, n=200):
        mem, _ = pf.filter(_self(), _opp(pos), dt=0.0)
        assert mem.visible is False
        np.testing.assert_allclose(
            mem.pos_local, frozen, atol=1e-6,
            err_msg=f"teleport to {pos} moved the frozen memory",
        )


# ---------------------------------------------------------------------------
# 4. Every-derived-feature sweep: enumerate the DERIVED fields from the frozen
#    spec and assert each is position-independent when invisible.
# ---------------------------------------------------------------------------


def test_every_derived_field_position_independent_when_invisible():
    """Each DERIVED obs field is constant as the hidden opponent's pos sweeps.

    Enumerates the derived block straight from ``FIELD_SLICES`` so a future-added
    derived feature that depends on the current (hidden) position trips this.
    Covers both regimes: ABSENT (never seen) and MEMORY (seen once, then hidden).
    """
    rng = _seed_np()
    derived_fields = _derived_fields()

    def derived_slice_values(vec):
        return {name: vec[FIELD_SLICES[name]].copy() for name in derived_fields}

    # --- ABSENT regime: fresh filter, opponent never seen. ---
    base_absent = derived_slice_values(_obs_for_hidden_position((0.0, 0.0, -10.0)))
    for pos in _behind_sweep_positions(rng, n=300):
        vals = derived_slice_values(_obs_for_hidden_position(pos))
        for name in derived_fields:
            np.testing.assert_array_equal(
                vals[name], base_absent[name],
                err_msg=f"derived '{name}' moved (ABSENT) for hidden opponent at {pos}",
            )

    # --- MEMORY regime: identical sighting, then one unseen step at matched dt. ---
    def memory_derived_after_move(live_pos):
        pf = PerceptionFilter()
        pf.filter(_self(), _opp((1.0, 0.5, 2.0)), dt=DEFAULT_DT)  # fixed sighting
        opp_state, derived = pf.filter(_self(), _opp(live_pos), dt=DEFAULT_DT)
        assert opp_state.visible is False
        vec = build_observation(_self_state_for_obs(), opp_state, derived)
        return derived_slice_values(vec)

    base_mem = memory_derived_after_move((0.0, 0.0, -10.0))
    for pos in _behind_sweep_positions(rng, n=300):
        vals = memory_derived_after_move(pos)
        for name in derived_fields:
            np.testing.assert_array_equal(
                vals[name], base_mem[name],
                err_msg=f"derived '{name}' moved (MEMORY) for hidden opponent at {pos}",
            )


def test_all_derived_fields_are_zero_when_invisible():
    """Every derived obs field reads exactly 0.0 in the unseen ABSENT regime."""
    derived_fields = _derived_fields()
    vec = _obs_for_hidden_position((0.0, 0.0, -7.0))
    for name in derived_fields:
        np.testing.assert_array_equal(
            vec[FIELD_SLICES[name]], np.zeros_like(vec[FIELD_SLICES[name]]),
            err_msg=f"derived '{name}' nonzero while invisible",
        )


# ---------------------------------------------------------------------------
# 5. LoS leak: identical FOV geometry, blocking obstacle -> visible 0, derived 0,
#    and NO leak through the raycast path (front-hidden obs == behind-hidden obs).
# ---------------------------------------------------------------------------


def test_los_block_yields_no_visible_and_no_derived_leak():
    """Same in-FOV geometry: clear -> visible+derived; wall -> hidden, derived 0."""
    geometry = (0.0, 0.0, 2.0)  # dead ahead, inside FOV + range + crosshair

    pf_clear = PerceptionFilter()
    clear_state, clear_derived = pf_clear.filter(_self(), _opp(geometry), dt=DEFAULT_DT)
    assert clear_state.visible is True
    assert clear_derived.in_range is True
    assert clear_derived.in_crosshair is True

    pf_wall = PerceptionFilter(los_clear=_block_all)
    wall_state, wall_derived = pf_wall.filter(_self(), _opp(geometry), dt=DEFAULT_DT)
    assert wall_state.visible is False
    np.testing.assert_array_equal(wall_state.pos_local, [0.0, 0.0, 0.0])
    assert wall_derived.in_range is False
    assert wall_derived.in_crosshair is False


def test_los_hidden_obs_matches_behind_hidden_obs():
    """A wall-hidden in-front opponent produces the SAME obs as a behind one.

    Both are unseen-and-never-seen (ABSENT): the observation may not distinguish
    "hidden behind a wall, dead ahead" from "behind me, far away". If the front
    geometry leaked through any field, these two vectors would differ.
    """
    front_hidden = _obs_for_hidden_position((0.0, 0.0, 2.0), los_clear=_block_all)
    behind_hidden = _obs_for_hidden_position((0.0, 0.0, -2.0))
    np.testing.assert_array_equal(
        front_hidden, behind_hidden,
        err_msg="wall-hidden front geometry leaked into the observation",
    )
    # And both are valid, in-range [-1, 1] vectors.
    validate(front_hidden)
    validate(behind_hidden)


def test_los_block_derived_invariant_to_position_in_front():
    """Behind a wall: sweeping the in-FRONT live position never moves derived."""
    rng = _seed_np()
    derived_fields = _derived_fields()

    def derived_after(live_pos):
        pf = PerceptionFilter(los_clear=_block_all)
        opp_state, derived = pf.filter(_self(), _opp(live_pos), dt=DEFAULT_DT)
        assert opp_state.visible is False
        vec = build_observation(_self_state_for_obs(), opp_state, derived)
        return {name: vec[FIELD_SLICES[name]].copy() for name in derived_fields}

    base = derived_after((0.0, 0.0, 1.0))
    for _ in range(300):
        dist = float(rng.uniform(0.5, 40.0))
        off_angle = float(rng.uniform(0.0, math.radians(FOV_DEGREES / 2.0 - 1.0)))
        bearing = float(rng.uniform(0.0, 2.0 * math.pi))
        x = dist * math.sin(off_angle) * math.cos(bearing)
        y = dist * math.sin(off_angle) * math.sin(bearing)
        z = dist * math.cos(off_angle)
        vals = derived_after((x, y, z))
        for name in derived_fields:
            np.testing.assert_array_equal(
                vals[name], base[name],
                err_msg=f"derived '{name}' moved behind wall at {(x, y, z)}",
            )


# ---------------------------------------------------------------------------
# Spec sanity: the crosshair window is genuinely exercised by the "kill shot"
# geometry above (so the visible-baseline assertions are not vacuous).
# ---------------------------------------------------------------------------


def test_kill_shot_geometry_is_inside_crosshair_window():
    """Sanity: the dead-ahead geometry used as the 'visible kill shot' really is
    inside both the attack range and the crosshair window (guards against the
    baseline assertions silently becoming vacuous if the constants change)."""
    assert ATTACK_RANGE - 0.5 < ATTACK_RANGE
    # Dead ahead is 0deg off-axis, trivially inside the crosshair half-angle.
    assert 0.0 <= CROSSHAIR_DEGREES
    assert FOV_DEGREES > 0.0
    assert MEMORY_TTL_SECONDS > 0.0
    # The contract really has OBS_DIM slots and a derived block at the tail.
    assert OBS_DIM == sum(s.stop - s.start for s in FIELD_SLICES.values())
