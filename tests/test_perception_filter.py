"""Tests for the PerceptionFilter — the fairness "soul module" (T12 / AC5).

These tests exercise the filter against SYNTHETIC geometry (no live server) and
satisfy the kickoff test cases:

  * TC2 — opponent OUTSIDE the FOV cone (or behind a blocking LoS obstacle) is
    gated out (``visible=0``); the CURRENT real position is never exposed,
    ``time_since_seen`` grows across steps, and derived ``in_range`` /
    ``in_crosshair`` are 0. The complementary INSIDE-FOV + LoS-clear case yields
    ``visible=1`` with the correct local-frame position and derived flags set.

  * TC4 — memory expiry: after seeing the opponent the last-seen position is
    HELD for ``MEMORY_TTL`` (``visible=0``, ``time_since_seen`` growing), then
    drops to ABSENT (position zeroed, ``time_since_seen`` maxed).

Plus the LoS-hook test (a blocking obstacle gates the opponent out with
otherwise-identical geometry) and the far-opponent clamp test (a visible
opponent beyond POS_SCALE still packs through build_observation + validate).

Coordinate convention (see env/perception_filter module docstring): yaw 0 looks
toward world +z; an opponent dead ahead at distance r lands at local (0, 0, r);
an opponent to the east (+x) lands at local +x (the agent's right).
"""

import math

import numpy as np
import pytest

from env.observation_spec import (
    MEMORY_TTL_SECONDS,
    POS_SCALE,
    SelfState,
    build_observation,
    validate,
)
from env.perception_filter import (
    ATTACK_RANGE,
    DEFAULT_DT,
    FOV_DEGREES,
    PerceptionFilter,
    RawState,
    default_los_clear,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures.
# ---------------------------------------------------------------------------

# A self placed at the world origin (feet), looking toward +z (yaw 0, level
# pitch). With this look direction an opponent on the +z axis is dead ahead.
def _self(pos=(0.0, 0.0, 0.0), yaw=0.0, pitch=0.0, velocity=(0.0, 0.0, 0.0)):
    return RawState(pos=pos, yaw=yaw, pitch=pitch, velocity=velocity)


def _opp(pos, yaw=0.0, pitch=0.0, velocity=(0.0, 0.0, 0.0)):
    return RawState(pos=pos, yaw=yaw, pitch=pitch, velocity=velocity)


def _self_state_for_obs():
    """A minimal valid SelfState so we can pack a full obs and validate it."""
    return SelfState(
        health=20.0,
        yaw=0.0,
        pitch=0.0,
        vel_local=(0.0, 0.0, 0.0),
        on_ground=True,
        held_item="iron_sword",
        attack_cooldown=1.0,
    )


# An opponent dead ahead on the +z axis, 2 blocks away (inside FOV, inside
# attack range). Eye height cancels (both raised equally) so the local pos is
# (0, 0, 2) on the forward axis.
_OPP_AHEAD = (0.0, 0.0, 2.0)

# An opponent directly behind the agent (negative z), well outside a 70° cone.
_OPP_BEHIND = (0.0, 0.0, -5.0)


# ---------------------------------------------------------------------------
# TC2 — FOV gating + no current-position leak + derived flags.
# ---------------------------------------------------------------------------


def test_visible_in_fov_sets_real_local_pos_and_derived():
    """INSIDE FOV + LoS clear -> visible=1 with correct local pos + derived."""
    pf = PerceptionFilter()
    opp_state, derived = pf.filter(_self(), _opp(_OPP_AHEAD), dt=DEFAULT_DT)

    assert opp_state.visible is True
    assert opp_state.time_since_seen == 0.0
    # Dead ahead at distance 2 -> local (0, 0, 2): forward axis only.
    np.testing.assert_allclose(opp_state.pos_local, [0.0, 0.0, 2.0], atol=1e-6)

    # Within ATTACK_RANGE (2 < 3) and dead-center -> both derived flags true.
    assert derived.in_range is True
    assert derived.in_crosshair is True


def test_in_range_false_when_visible_but_far():
    """Derived in_range respects the attack-range threshold (still visible)."""
    pf = PerceptionFilter()
    far = (0.0, 0.0, ATTACK_RANGE + 1.0)  # ahead but beyond reach
    opp_state, derived = pf.filter(_self(), _opp(far))

    assert opp_state.visible is True
    assert derived.in_range is False
    # Dead-center, so crosshair is still true even though out of range.
    assert derived.in_crosshair is True


def test_in_crosshair_false_when_off_center_but_in_fov():
    """A target inside the wide FOV but outside the tight crosshair window."""
    pf = PerceptionFilter(fov_degrees=70.0, crosshair_degrees=5.0)
    # ~25° off the +z axis (within 35° half-FOV, outside 5° crosshair).
    angle = math.radians(25.0)
    dist = 2.0
    opp = (dist * math.sin(angle), 0.0, dist * math.cos(angle))
    # Note: +x maps to local right; the sign of x doesn't matter for the angle.
    opp_state, derived = pf.filter(_self(), _opp(opp))

    assert opp_state.visible is True
    assert derived.in_range is True
    assert derived.in_crosshair is False


def test_outside_fov_gates_opponent_out():
    """TC2: opponent OUTSIDE the FOV cone -> visible=0 and current pos withheld."""
    pf = PerceptionFilter()
    opp_state, derived = pf.filter(_self(), _opp(_OPP_BEHIND))

    assert opp_state.visible is False
    # Never seen yet -> ABSENT: position zeroed (the live behind-pos is NOT
    # exposed in any component).
    np.testing.assert_allclose(opp_state.pos_local, [0.0, 0.0, 0.0])
    # Derived features reveal nothing.
    assert derived.in_range is False
    assert derived.in_crosshair is False


def test_current_position_never_exposed_while_unseen():
    """TC2: even with memory set, the unseen-opponent's CURRENT pos is withheld.

    See it once (memory holds the last-seen pos), then it moves to a brand-new
    place while out of view. The gated pos must equal the stale last-seen value,
    NOT the live position.
    """
    pf = PerceptionFilter()
    # Step 1: visible dead ahead at +z=2 -> memory stores local (0,0,2).
    seen_state, _ = pf.filter(_self(), _opp((0.0, 0.0, 2.0)))
    assert seen_state.visible is True
    np.testing.assert_allclose(seen_state.pos_local, [0.0, 0.0, 2.0], atol=1e-6)

    # Step 2: opponent teleports far to the side AND behind (out of the cone).
    live_pos = (50.0, 0.0, -50.0)
    mem_state, derived = pf.filter(_self(), _opp(live_pos))
    assert mem_state.visible is False
    # Held last-seen pos, NOT the live (50, .., -50) location.
    np.testing.assert_allclose(mem_state.pos_local, [0.0, 0.0, 2.0], atol=1e-6)
    # And nothing leaks through derived.
    assert derived.in_range is False
    assert derived.in_crosshair is False


def test_time_since_seen_grows_across_steps_while_unseen():
    """TC2: time_since_seen increases each step the opponent stays unseen."""
    pf = PerceptionFilter()
    # Establish a sighting so we are in MEMORY (not ABSENT) afterward.
    pf.filter(_self(), _opp(_OPP_AHEAD), dt=0.2)

    s1, _ = pf.filter(_self(), _opp(_OPP_BEHIND), dt=0.2)
    s2, _ = pf.filter(_self(), _opp(_OPP_BEHIND), dt=0.2)
    s3, _ = pf.filter(_self(), _opp(_OPP_BEHIND), dt=0.2)

    assert s1.visible is False and s2.visible is False and s3.visible is False
    assert s1.time_since_seen == pytest.approx(0.2)
    assert s2.time_since_seen == pytest.approx(0.4)
    assert s3.time_since_seen == pytest.approx(0.6)
    # Strictly growing.
    assert s1.time_since_seen < s2.time_since_seen < s3.time_since_seen


def test_seeing_again_resets_time_since_seen():
    """Re-acquiring sight resets the age to 0 and refreshes memory."""
    pf = PerceptionFilter()
    pf.filter(_self(), _opp(_OPP_AHEAD), dt=0.2)
    aged, _ = pf.filter(_self(), _opp(_OPP_BEHIND), dt=0.2)
    assert aged.time_since_seen == pytest.approx(0.2)

    reseen, derived = pf.filter(_self(), _opp((0.0, 0.0, 1.0)), dt=0.2)
    assert reseen.visible is True
    assert reseen.time_since_seen == 0.0
    np.testing.assert_allclose(reseen.pos_local, [0.0, 0.0, 1.0], atol=1e-6)
    assert derived.in_range is True


# ---------------------------------------------------------------------------
# Line-of-sight hook — identical geometry, blocking obstacle gates it out.
# ---------------------------------------------------------------------------


def test_los_block_gates_opponent_out():
    """A blocking LoS test hides an opponent that is geometrically in the FOV."""
    def block_all(eye, target):
        return False  # a wall always occludes

    pf_clear = PerceptionFilter()  # default open arena
    pf_blocked = PerceptionFilter(los_clear=block_all)

    # Same geometry: dead ahead, inside FOV and range.
    clear_state, clear_derived = pf_clear.filter(_self(), _opp(_OPP_AHEAD))
    blocked_state, blocked_derived = pf_blocked.filter(_self(), _opp(_OPP_AHEAD))

    # With a clear LoS the opponent is visible; with the wall it is gated out.
    assert clear_state.visible is True
    assert clear_derived.in_range is True

    assert blocked_state.visible is False
    np.testing.assert_allclose(blocked_state.pos_local, [0.0, 0.0, 0.0])
    assert blocked_derived.in_range is False
    assert blocked_derived.in_crosshair is False


def test_los_clear_receives_world_eye_positions():
    """The LoS callable is invoked with world-frame eye coords of both bots."""
    captured = {}

    def spy(eye, target):
        captured["eye"] = list(eye)
        captured["target"] = list(target)
        return True

    pf = PerceptionFilter(los_clear=spy)
    pf.filter(_self(pos=(10.0, 64.0, 10.0)), _opp((10.0, 64.0, 12.0)))

    # Eye is feet + eye-height on Y; x/z unchanged.
    assert captured["eye"][0] == pytest.approx(10.0)
    assert captured["eye"][2] == pytest.approx(10.0)
    assert captured["eye"][1] > 64.0  # raised to eye level
    assert captured["target"][0] == pytest.approx(10.0)
    assert captured["target"][2] == pytest.approx(12.0)


def test_default_los_is_always_clear():
    assert default_los_clear([0, 0, 0], [100, 0, 100]) is True


# ---------------------------------------------------------------------------
# TC4 — memory hold then expiry.
# ---------------------------------------------------------------------------


def test_memory_holds_last_seen_then_drops_to_absent():
    """TC4: last-seen pos is HELD within TTL, then zeroed once TTL elapses."""
    dt = 1.0
    pf = PerceptionFilter(memory_ttl=MEMORY_TTL_SECONDS, los_clear=default_los_clear)

    # See the opponent once at local (0, 0, 2).
    seen, _ = pf.filter(_self(), _opp((0.0, 0.0, 2.0)), dt=dt)
    assert seen.visible is True
    np.testing.assert_allclose(seen.pos_local, [0.0, 0.0, 2.0], atol=1e-6)

    # Now keep it out of view. Steps at dt=1s: ages 1,2,3,4,5 are <= TTL (5s)
    # -> MEMORY (held). Age 6 > TTL -> ABSENT (zeroed).
    held_positions = []
    for expected_age in (1.0, 2.0, 3.0, 4.0, 5.0):
        st, derived = pf.filter(_self(), _opp(_OPP_BEHIND), dt=dt)
        assert st.visible is False, f"should still be in memory at age {expected_age}"
        assert st.time_since_seen == pytest.approx(expected_age)
        held_positions.append(tuple(st.pos_local))
        # Memory never leaks through derived features.
        assert derived.in_range is False and derived.in_crosshair is False

    # The held position was the last-seen value at every memory step.
    for pos in held_positions:
        np.testing.assert_allclose(pos, [0.0, 0.0, 2.0], atol=1e-6)

    # One more step pushes age to 6s > TTL -> ABSENT.
    absent, derived = pf.filter(_self(), _opp(_OPP_BEHIND), dt=dt)
    assert absent.visible is False
    np.testing.assert_allclose(absent.pos_local, [0.0, 0.0, 0.0])
    assert absent.time_since_seen == pytest.approx(MEMORY_TTL_SECONDS)
    assert derived.in_range is False and derived.in_crosshair is False


def test_memory_zeros_facing_and_velocity_but_holds_position():
    """Spec: MEMORY degrades POSITION to last-seen; facing/velocity go to 0."""
    pf = PerceptionFilter()
    # See it once with a real facing + velocity.
    pf.filter(
        _self(),
        _opp((0.0, 0.0, 2.0), yaw=math.pi / 2, velocity=(0.3, 0.0, 0.1)),
        dt=0.2,
    )
    mem, _ = pf.filter(_self(), _opp(_OPP_BEHIND), dt=0.2)

    assert mem.visible is False
    # Position held...
    np.testing.assert_allclose(mem.pos_local, [0.0, 0.0, 2.0], atol=1e-6)
    # ...but facing and velocity are zeroed.
    assert mem.facing_yaw == 0.0
    np.testing.assert_allclose(mem.vel_local, [0.0, 0.0, 0.0])


def test_reset_clears_memory_to_absent():
    """reset() drops any held memory back to the ABSENT regime."""
    pf = PerceptionFilter()
    pf.filter(_self(), _opp(_OPP_AHEAD), dt=0.2)
    mem, _ = pf.filter(_self(), _opp(_OPP_BEHIND), dt=0.2)
    assert mem.time_since_seen == pytest.approx(0.2)  # in memory

    pf.reset()
    absent, derived = pf.filter(_self(), _opp(_OPP_BEHIND), dt=0.2)
    assert absent.visible is False
    np.testing.assert_allclose(absent.pos_local, [0.0, 0.0, 0.0])
    # Straight to ABSENT (max age), not memory.
    assert absent.time_since_seen == pytest.approx(MEMORY_TTL_SECONDS)
    assert derived.in_range is False and derived.in_crosshair is False


def test_absent_when_never_seen():
    """Never-seen opponent out of view is ABSENT (age maxed), not memory."""
    pf = PerceptionFilter()
    st, _ = pf.filter(_self(), _opp(_OPP_BEHIND), dt=0.2)
    assert st.visible is False
    assert st.time_since_seen == pytest.approx(MEMORY_TTL_SECONDS)
    np.testing.assert_allclose(st.pos_local, [0.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# Far-opponent clamp — a visible opponent beyond POS_SCALE still validates.
# ---------------------------------------------------------------------------


def test_far_visible_opponent_clamped_and_obs_validates():
    """A visible opponent far beyond POS_SCALE packs into a valid [-1,1] obs."""
    pf = PerceptionFilter(attack_range=1000.0)  # keep it "visible" + in range
    # 100 blocks dead ahead — well beyond POS_SCALE (16).
    far = (0.0, 0.0, 100.0)
    opp_state, derived = pf.filter(_self(), _opp(far))

    assert opp_state.visible is True
    # Forward component is clamped to +POS_SCALE (not 100).
    assert opp_state.pos_local[2] == pytest.approx(POS_SCALE)
    assert all(abs(c) <= POS_SCALE + 1e-9 for c in opp_state.pos_local)

    # The whole observation must pack and validate without raising.
    vec = build_observation(_self_state_for_obs(), opp_state, derived)
    validate(vec)  # must not raise


def test_far_memory_position_clamped_and_validates():
    """A far last-seen position, once in memory, still packs into a valid obs."""
    pf = PerceptionFilter(attack_range=1000.0)
    pf.filter(_self(), _opp((0.0, 0.0, 100.0)), dt=0.2)  # sighting at 100 ahead
    mem, derived = pf.filter(_self(), _opp(_OPP_BEHIND), dt=0.2)

    assert mem.visible is False
    assert mem.pos_local[2] == pytest.approx(POS_SCALE)  # held + clamped
    vec = build_observation(_self_state_for_obs(), mem, derived)
    validate(vec)


# ---------------------------------------------------------------------------
# Input-shape flexibility (documented contract for T9).
# ---------------------------------------------------------------------------


def test_filter_accepts_mapping_and_attr_objects():
    """filter() coerces RawState, plain dicts, and attr-objects identically."""
    pf = PerceptionFilter()
    as_dataclass = pf.filter(_self(), _opp(_OPP_AHEAD))

    pf.reset()
    as_dict = pf.filter(
        {"pos": [0.0, 0.0, 0.0], "yaw": 0.0, "pitch": 0.0, "velocity": [0.0, 0.0, 0.0]},
        {"pos": [0.0, 0.0, 2.0], "yaw": 0.0, "pitch": 0.0, "velocity": [0.0, 0.0, 0.0]},
    )

    assert as_dataclass[0].visible == as_dict[0].visible is True
    np.testing.assert_allclose(as_dataclass[0].pos_local, as_dict[0].pos_local, atol=1e-6)


def test_velocity_optional_in_mapping():
    """A mapping without velocity defaults to zero velocity (no KeyError)."""
    pf = PerceptionFilter()
    st, _ = pf.filter(
        {"pos": [0.0, 0.0, 0.0], "yaw": 0.0, "pitch": 0.0},
        {"pos": [0.0, 0.0, 2.0], "yaw": 0.0, "pitch": 0.0},
    )
    assert st.visible is True
    np.testing.assert_allclose(st.vel_local, [0.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# Constructor validation.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fov_degrees": 0.0},
        {"fov_degrees": -10.0},
        {"memory_ttl": 0.0},
        {"attack_range": -1.0},
        {"crosshair_degrees": 0.0},
    ],
)
def test_constructor_rejects_nonpositive_thresholds(kwargs):
    with pytest.raises(ValueError):
        PerceptionFilter(**kwargs)


def test_filter_rejects_negative_dt():
    pf = PerceptionFilter()
    with pytest.raises(ValueError):
        pf.filter(_self(), _opp(_OPP_AHEAD), dt=-0.1)


def test_yaw_rotated_local_frame():
    """With yaw=pi/2 (looking -x), an opponent due west is dead ahead (+z_local)."""
    pf = PerceptionFilter()
    me = _self(yaw=math.pi / 2)
    # Opponent 3 blocks west (-x) of the agent.
    opp_state, derived = pf.filter(me, _opp((-3.0, 0.0, 0.0)))
    assert opp_state.visible is True
    np.testing.assert_allclose(opp_state.pos_local, [0.0, 0.0, 3.0], atol=1e-6)
    assert derived.in_crosshair is True


def test_fov_constant_default_matches_spec():
    """Sanity: the FOV default is the ~70° spec value (TUNE)."""
    assert FOV_DEGREES == pytest.approx(70.0)
    pf = PerceptionFilter()
    assert pf.fov_degrees == pytest.approx(70.0)
