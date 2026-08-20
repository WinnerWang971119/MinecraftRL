"""test_perception_leak_mirror — the fairness leak battery for the OPPONENT seat (T7).

This is the mirror image of ``tests/test_perception_leak.py``. That battery
hammers the fairness invariant on the LEARNER's observation; this one hammers
the identical invariant on the observation ``env/mirror_perception.py`` builds
for the **opponent** seat (README rule 2: *fairness lives in Python, never in
the bridge* — and it applies to whichever seat is observing).

    When ``visible == false`` from the OPPONENT's eyes, NO derived feature
    (``in_range`` / ``in_crosshair``) and NO field of the mirrored observation
    vector may reveal the LEARNER's CURRENT (hidden) position. The opponent may
    legitimately *remember* a frozen last-seen position (MEMORY), but nothing in
    the mirrored vector may change as a function of where the learner actually
    is *right now* while unseen.

WHY A SECOND BATTERY, AND WHY THE STAKES ARE HIGHER HERE
--------------------------------------------------------
The opponent seat is driven by a FROZEN snapshot of the agent's own past
policy. That snapshot was trained under FOV + line-of-sight + memory gating. A
leak here does not merely make the opponent strong — it makes it play a game
its training never showed it, so it stops resembling the policy that earned its
snapshot, and the Elo curve measured against it silently stops meaning
anything. Nothing downstream can see this: the vector still has the right
length, the right dtype and in-range values, so ``validate()`` passes either
way.

WHAT EACH SECTION PINS, AND THE FAILURE MODE IT GUARDS
------------------------------------------------------
1. **Derived-zero-when-invisible.** ``in_range`` / ``in_crosshair`` must be 0
   whenever the learner is unseen *from the opponent's seat* — including the
   adversarial case where the learner is geometrically point-blank and dead in
   the crosshair but behind the opponent, or behind a wall. Guards: derived
   features wired to the live geometry instead of the gated one.
2. **Whole-vector invariance.** Sweeping the hidden learner's live position,
   velocity and facing must not move a single byte. Guards: ANY field (present
   or future) taking a dependence on the hidden learner's live state.
3. **Role-swapped geometry.** The gate must key off the OPPONENT's yaw and look
   vector. Guards the module's documented poison case: a mirror that reuses the
   LEARNER's yaw produces a well-formed vector describing a fighter that can see
   things it cannot see (and, in reverse, is blind to what is in front of it).
4. **Memory regime.** ``time_since_seen`` ages from the opponent's seat and the
   exposed position is the LAST SEEN one, never the live one. Guards: a memory
   that quietly tracks the hidden learner — the subtlest leak of all, because
   the vector keeps reporting ``visible == 0`` while carrying live coordinates.
5. **Reset isolation.** ``OpponentMirror.reset`` clears the per-episode memory,
   and so does ``MCPvPEnv.reset`` at the production seam. Guards: a frozen net
   starting a fresh episode already "remembering" the previous one's geometry.

DELIBERATE OVERLAP WITH ``tests/test_mirror_perception.py`` (T5)
----------------------------------------------------------------
T5's ``test_mirrored_observation_hides_the_learners_live_position`` proves the
invariance property once, over 50 probes of one benign geometry. That is the
*characterization* test. This file is the *adversarial* one: it aims every
probe at the gate — point-blank behind, the exact mirror of a kill shot, inside
the cone but walled off, teleports mid-memory, and a role-swap where the two
seats disagree about who can see whom. Where a claim is restated here it is
restated on hostile geometry, and the comment says so.

SELF-DIAGNOSING FAILURES
------------------------
Every invariance assertion is preceded by its PRECONDITION — that the learner
really is unseen from the opponent's seat. A visible probe SHOULD move the
vector (a visible position is not a leak), so without the precondition check a
test that drifted off its geometry would fail with a byte mismatch that reads
exactly like a leak. Asserting ``visible == 0`` first makes such a test fail
with "probe was VISIBLE" instead of a false leak alarm.

Determinism: numpy is seeded once per test (``_seed_np``) so every sweep is
reproducible; no sockets, no live server, hand-authored fixtures only.

Coordinate convention (see ``env/perception_filter``): yaw 0 looks toward world
``+z``; a target dead ahead at distance ``r`` lands at local ``(0, 0, r)``. In
THIS file the viewer is the OPPONENT, so "in front" / "behind" are always
relative to the opponent's look vector, not the learner's.
"""

import math
from dataclasses import dataclass, replace
from typing import Iterator, List, Tuple

import numpy as np
import pytest

from bridge.messages import ResetAckMsg, StateMsg
from env.mc_pvp_env import MCPvPEnv
from env.mirror_perception import OpponentMirror
from env.observation_spec import (
    FIELD_SLICES,
    MEMORY_TTL_SECONDS,
    OBS_DIM,
    POS_SCALE,
    validate,
)
from env.perception_filter import (
    ATTACK_RANGE,
    CROSSHAIR_DEGREES,
    FOV_DEGREES,
    PerceptionFilter,
    RawState,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers.
# ---------------------------------------------------------------------------

#: A fixed seed so the randomized learner-position sweeps are reproducible.
#: Same value as ``tests/test_perception_leak.py`` so the two files can be
#: diffed line for line.
_SEED = 1234567

#: Decision interval (seconds). Every call uses the same dt, so the mirrored
#: memory age is identical across probes and cannot itself explain a byte
#: difference.
_DT = 0.2

#: Arena floor height. Both fighters stand on it, so the eye-height offset
#: (+1.62 on each) cancels out of the eye-to-eye delta and the geometry below
#: is exactly the flat 2-D reasoning it looks like.
_ARENA_Y = 64.0

#: The opponent's swing meter, shadow-tracked in Python by the env. Held
#: constant so it can never be the thing that moves a byte.
_OPP_COOLDOWN = 0.375
#: The learner's swing meter, which rides the wire. Never reaches the mirrored
#: vector (T5 pins that); fixed here so this file's claims stay about geometry.
_LEARNER_COOLDOWN = 1.0


def _seed_np() -> np.random.Generator:
    """Return a freshly seeded numpy Generator (deterministic across runs)."""
    return np.random.default_rng(_SEED)


@dataclass(frozen=True)
class _Fighter:
    """One fighter's RAW world-frame state, seat-agnostic.

    Same shape as the fixture in ``tests/test_mirror_perception.py``: the wire
    blocks are built from it, so a probe is one ``replace()`` away.
    """

    pos: Tuple[float, float, float]
    yaw: float
    pitch: float
    vel: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    health: float = 20.0
    on_ground: bool = True
    held_item: str = "iron_sword"


def _self_block(fighter: _Fighter, attack_cooldown: float) -> dict:
    """Wire ``self`` block — the only block carrying ``attack_cooldown``."""
    return {
        "pos": list(fighter.pos),
        "yaw": fighter.yaw,
        "pitch": fighter.pitch,
        "velocity": list(fighter.vel),
        "on_ground": fighter.on_ground,
        "health": fighter.health,
        "held_item": fighter.held_item,
        "attack_cooldown": attack_cooldown,
    }


def _opponent_block(fighter: _Fighter) -> dict:
    """Wire ``opponent`` block — every self field EXCEPT ``attack_cooldown``."""
    return {
        "pos": list(fighter.pos),
        "yaw": fighter.yaw,
        "pitch": fighter.pitch,
        "velocity": list(fighter.vel),
        "on_ground": fighter.on_ground,
        "health": fighter.health,
        "held_item": fighter.held_item,
    }


def _state(learner: _Fighter, opponent: _Fighter) -> StateMsg:
    """Build a valid ``StateMsg`` (learner in the self seat, as on the wire)."""
    return StateMsg.from_dict(
        {
            "type": "state",
            "self": _self_block(learner, _LEARNER_COOLDOWN),
            "opponent": _opponent_block(opponent),
            "events": {
                "damage_dealt": 0.0,
                "damage_taken": 0.0,
                "i_died": False,
                "opponent_died": False,
            },
            "arena": {"wall_distances": [8.0, 8.0, 8.0, 8.0]},
            "tick": 1,
            "code_version": "test",
        }
    )


def _scalar(obs: np.ndarray, name: str) -> float:
    """Read a scalar observation field by its frozen name."""
    return float(obs[FIELD_SLICES[name]][0])


def _vec(obs: np.ndarray, name: str) -> np.ndarray:
    """Read a vector observation field by its frozen name."""
    return np.asarray(obs[FIELD_SLICES[name]], dtype=np.float64)


def _raw(fighter: _Fighter) -> RawState:
    """The fighter as the perception filter's raw input (learner-seat reference)."""
    return RawState(
        pos=fighter.pos, yaw=fighter.yaw, pitch=fighter.pitch, velocity=fighter.vel
    )


def _block_all(eye, target):
    """A LoS test modelling a wall that always occludes (target never visible)."""
    return False


# The VIEWER of every mirrored observation: the opponent, at the origin, looking
# toward +z. Its pitch is 0 on purpose and the sweeps below depend on it — a
# tilted look vector would swing the FOV cone up or down and let a high or low
# "behind" probe re-enter it, which is exactly the kind of drift the per-probe
# ``visible == 0`` precondition is there to report honestly.
#
# Its own SELF-block fields are deliberately non-default (wounded, airborne,
# axe in hand, moving): a mirrored vector that is mostly zeros would make the
# byte-equality assertions below far too easy to satisfy.
_OPPONENT = _Fighter(
    pos=(0.0, _ARENA_Y, 0.0),
    yaw=0.0,
    pitch=0.0,
    vel=(0.12, 0.0, -0.05),
    health=11.0,
    on_ground=False,
    held_item="stone_axe",
)

#: The learner, in its "hidden" default: behind the opponent, off-axis yaw/pitch
#: so an accidental symmetry cannot hide a wrong rotation.
_LEARNER = _Fighter(
    pos=(0.0, _ARENA_Y, -4.0),
    yaw=0.10,
    pitch=0.05,
    vel=(0.10, 0.0, 0.05),
    health=17.0,
    on_ground=True,
    held_item="iron_sword",
)

#: The learner where the opponent CAN see it: dead ahead at 2.0 blocks, which is
#: inside the FOV cone, inside the crosshair window and inside attack range.
_SEEN_LEARNER = replace(_LEARNER, pos=(0.0, _ARENA_Y, 2.0))

#: That sighting's frozen local position, in raw blocks (dead ahead at 2.0).
_SEEN_POS_LOCAL = (0.0, 0.0, 2.0)


def _derived_fields() -> List[str]:
    """The DERIVED block field names, read live from the frozen layout.

    Derived from ``FIELD_SLICES`` (rather than hardcoding ``in_range`` /
    ``in_crosshair``) so a future derived field added to the contract is swept
    automatically — a newly-added leaky derived feature trips this battery on
    the opponent's seat as well as the learner's.
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


# A grid of live LEARNER positions GUARANTEED to be out of the opponent's view:
# every point is behind the opponent (z < 0 while it looks toward +z), spanning
# a wide range of x/y/z including points that are geometrically in-range and
# perfectly on the (reversed) sight line. None of these may move the mirrored
# vector.
def _behind_sweep_positions(rng, n=400) -> List[Tuple[float, float, float]]:
    xs = rng.uniform(-60.0, 60.0, size=n)
    ys = _ARENA_Y + rng.uniform(-20.0, 20.0, size=n)
    zs = rng.uniform(-60.0, -0.5, size=n)  # strictly behind the opponent
    positions = list(zip(xs.tolist(), ys.tolist(), zs.tolist()))
    # Explicitly include the nastiest adversarial points: directly behind the
    # opponent, at the exact mirror of an in-range/in-crosshair target.
    positions.append((0.0, _ARENA_Y, -0.5))            # point-blank, dead behind
    positions.append((0.0, _ARENA_Y, -ATTACK_RANGE))   # in-range distance, behind
    positions.append((0.0, _ARENA_Y, -50.0))           # far, dead behind
    return positions


def _in_cone_positions(rng, n=400) -> List[Tuple[float, float, float]]:
    """Live learner positions strictly INSIDE the opponent's FOV cone.

    These pass the FOV test outright; only a blocking ``los_clear`` gates them
    out, so they exercise the raycast path rather than the cone path.
    """
    positions = []
    for _ in range(n):
        dist = float(rng.uniform(0.5, 40.0))
        off_angle = float(rng.uniform(0.0, math.radians(FOV_DEGREES / 2.0 - 1.0)))
        bearing = float(rng.uniform(0.0, 2.0 * math.pi))
        positions.append(
            (
                dist * math.sin(off_angle) * math.cos(bearing),
                _ARENA_Y + dist * math.sin(off_angle) * math.sin(bearing),
                dist * math.cos(off_angle),
            )
        )
    return positions


def _hidden_probes(rng, positions) -> Iterator[_Fighter]:
    """Hidden-learner probes: swept position AND velocity AND facing.

    Position is the leak everyone thinks of; velocity and facing are gated by
    the SAME regime and are just as much of a leak, so they are swept together.
    None of the three may move a byte while the learner is unseen.
    """
    for pos in positions:
        yield replace(
            _LEARNER,
            pos=pos,
            vel=tuple(float(v) for v in rng.uniform(-2.0, 2.0, size=3)),
            yaw=float(rng.uniform(-math.pi, math.pi)),
            pitch=float(rng.uniform(-1.0, 1.0)),
        )


def _mirrored_obs(learner: _Fighter, opponent: _Fighter = _OPPONENT, *, los_clear=None):
    """Mirror ONE window through a fresh ``OpponentMirror`` and return the obs.

    A fresh mirror per call means the perception memory is in the ABSENT regime
    (the learner has never been seen), so the ONLY thing that could move the
    returned vector is an illegal dependence on the learner's live state.
    """
    mirror = OpponentMirror(perception_filter=PerceptionFilter(los_clear=los_clear))
    return mirror.observe(
        _state(learner, opponent), dt=_DT, opp_attack_cooldown=_OPP_COOLDOWN
    )


def _hidden_mirrored_obs(learner: _Fighter, **kwargs) -> np.ndarray:
    """:func:`_mirrored_obs`, with the "the learner really is unseen" precondition.

    The precondition is asserted BEFORE any caller compares bytes: a visible
    probe legitimately moves the vector, so without this check a test that
    drifted off its geometry would fail with a byte mismatch that reads exactly
    like a perception leak.
    """
    obs = _mirrored_obs(learner, **kwargs)
    assert _scalar(obs, "visible") == 0.0, (
        f"precondition failed: the learner at {learner.pos} (yaw={learner.yaw:.3f}) "
        "was VISIBLE from the opponent's seat, so this probe tests nothing"
    )
    return obs


# ---------------------------------------------------------------------------
# 1. Derived-features-zero-when-invisible, from the OPPONENT's seat.
# ---------------------------------------------------------------------------


def test_mirrored_derived_zero_for_many_out_of_fov_learner_positions():
    """Randomized learners BEHIND the opponent -> derived stay 0 (never seen)."""
    rng = _seed_np()
    for learner in _hidden_probes(rng, _behind_sweep_positions(rng, n=400)):
        obs = _mirrored_obs(learner)
        assert _scalar(obs, "visible") == 0.0, f"unexpectedly visible at {learner.pos}"
        assert _scalar(obs, "in_range") == 0.0, f"in_range leaked at {learner.pos}"
        assert _scalar(obs, "in_crosshair") == 0.0, (
            f"in_crosshair leaked at {learner.pos}"
        )


def test_mirrored_derived_zero_behind_los_wall_for_many_positions():
    """Randomized learners inside the opponent's cone but behind a wall -> 0.

    Here the geometry would otherwise pass the FOV test (every point is in
    front of the opponent); only the blocking LoS hook gates them out. Derived
    features must not leak through the raycast path of the mirrored seat.
    """
    rng = _seed_np()
    for learner in _hidden_probes(rng, _in_cone_positions(rng, n=400)):
        obs = _mirrored_obs(learner, los_clear=_block_all)
        assert _scalar(obs, "visible") == 0.0, f"wall failed to gate {learner.pos}"
        assert _scalar(obs, "in_range") == 0.0
        assert _scalar(obs, "in_crosshair") == 0.0


def test_mirrored_derived_zero_when_in_range_and_dead_in_crosshair_but_behind():
    """Adversarial: in-range + dead-center, but BEHIND the opponent -> derived 0.

    The learner sits at the exact distance/alignment that WOULD set both derived
    flags if the opponent could see it, mirrored onto the -z axis so it is
    outside the cone. ``visible == false`` forces both flags to 0.
    """
    behind_in_range = replace(
        _LEARNER, pos=(0.0, _ARENA_Y, -(ATTACK_RANGE - 0.5)), yaw=0.0, pitch=0.0
    )
    obs = _mirrored_obs(behind_in_range)

    assert _scalar(obs, "visible") == 0.0
    assert _scalar(obs, "in_range") == 0.0
    assert _scalar(obs, "in_crosshair") == 0.0
    # Position is not exposed either (never seen -> ABSENT zeroes it).
    np.testing.assert_array_equal(_vec(obs, "opp_pos_local"), np.zeros(3))


def test_mirrored_derived_zero_when_in_range_and_dead_in_crosshair_but_behind_wall():
    """Adversarial: in-range + dead-center IN FRONT, but a wall hides it -> 0.

    Identical geometry to a guaranteed-visible kill shot from the opponent's
    seat, but a blocking wall gates it out. The clear-LoS half of this test is
    also the positive control for the whole file: it proves the "kill shot"
    geometry really does light up both flags, so every zero asserted elsewhere
    is a gate doing its job rather than a gate that can never fire.
    """
    dead_ahead_in_range = replace(
        _LEARNER, pos=(0.0, _ARENA_Y, ATTACK_RANGE - 0.5), yaw=0.0, pitch=0.0
    )

    clear = _mirrored_obs(dead_ahead_in_range)
    assert _scalar(clear, "visible") == 1.0
    assert _scalar(clear, "in_range") == 1.0
    assert _scalar(clear, "in_crosshair") == 1.0

    blocked = _mirrored_obs(dead_ahead_in_range, los_clear=_block_all)
    assert _scalar(blocked, "visible") == 0.0
    assert _scalar(blocked, "in_range") == 0.0
    assert _scalar(blocked, "in_crosshair") == 0.0


# ---------------------------------------------------------------------------
# 2. Current-position-invariance: moving the hidden LEARNER must not move any
#    value of the mirrored observation.
#
#    Deliberate overlap with T5's 50-probe sweep: same claim, hostile probes.
#    T5 sweeps one benign box around the arena; this sweeps 403 points that
#    include the point-blank mirror of a kill shot, 60-block extremes, and a
#    swept velocity and facing on every probe.
# ---------------------------------------------------------------------------


def test_mirrored_absent_obs_invariant_to_the_learners_current_position_behind():
    """ABSENT: sweep the live BEHIND learner; the whole vector is constant."""
    rng = _seed_np()
    baseline = _hidden_mirrored_obs(replace(_LEARNER, pos=(0.0, _ARENA_Y, -10.0)))
    validate(baseline)
    for learner in _hidden_probes(rng, _behind_sweep_positions(rng, n=400)):
        vec = _hidden_mirrored_obs(learner)
        np.testing.assert_array_equal(
            vec,
            baseline,
            err_msg=f"mirrored obs moved for a hidden learner at {learner.pos}",
        )


def test_mirrored_absent_obs_invariant_to_the_learners_position_behind_wall():
    """ABSENT behind a wall: sweep live IN-CONE positions; the vector holds."""
    rng = _seed_np()
    baseline = _hidden_mirrored_obs(
        replace(_LEARNER, pos=(0.0, _ARENA_Y, 5.0)), los_clear=_block_all
    )
    validate(baseline)
    for learner in _hidden_probes(rng, _in_cone_positions(rng, n=400)):
        vec = _hidden_mirrored_obs(learner, los_clear=_block_all)
        np.testing.assert_array_equal(
            vec,
            baseline,
            err_msg=f"mirrored obs moved for a wall-hidden learner at {learner.pos}",
        )


def test_mirrored_memory_obs_invariant_to_the_learners_current_position():
    """MEMORY: with the age held constant, moving the hidden learner is a no-op.

    After a single sighting (which freezes the last-seen memory), the mirrored
    obs in the next unseen window is fully determined by that frozen memory and
    the matched ``dt`` aging — NOT by where the learner currently is. Each probe
    rebuilds the mirror, replays the SAME sighting, then takes one unseen window
    at the SAME dt, so the only thing that differs is the hidden live state.
    """
    rng = _seed_np()

    def memory_obs_after_move(learner: _Fighter) -> np.ndarray:
        mirror = OpponentMirror()
        # Identical sighting every time -> identical frozen memory + age.
        seen = mirror.observe(
            _state(_SEEN_LEARNER, _OPPONENT), dt=_DT, opp_attack_cooldown=_OPP_COOLDOWN
        )
        assert _scalar(seen, "visible") == 1.0, "the sighting window was not visible"
        # One unseen window at the SAME dt; only the learner's live state differs.
        obs = mirror.observe(
            _state(learner, _OPPONENT), dt=_DT, opp_attack_cooldown=_OPP_COOLDOWN
        )
        assert _scalar(obs, "visible") == 0.0, (
            f"precondition failed: the learner at {learner.pos} was VISIBLE"
        )
        return obs

    baseline = memory_obs_after_move(replace(_LEARNER, pos=(0.0, _ARENA_Y, -10.0)))
    validate(baseline)
    assert _scalar(baseline, "time_since_seen") == pytest.approx(
        _DT / MEMORY_TTL_SECONDS
    ), "the baseline is not in the MEMORY regime, so this test proves nothing"

    for learner in _hidden_probes(rng, _behind_sweep_positions(rng, n=300)):
        vec = memory_obs_after_move(learner)
        np.testing.assert_array_equal(
            vec,
            baseline,
            err_msg=f"mirrored memory obs moved for a hidden learner at {learner.pos}",
        )


# ---------------------------------------------------------------------------
# 3. Memory is not a current-position leak: the position the opponent seat
#    exposes is the STALE last-seen value, never the live one.
# ---------------------------------------------------------------------------


def test_mirrored_memory_pos_is_stale_last_seen_not_live():
    """After a sighting then loss of sight, the held pos is the frozen one."""
    mirror = OpponentMirror()
    seen = mirror.observe(
        _state(_SEEN_LEARNER, _OPPONENT), dt=_DT, opp_attack_cooldown=_OPP_COOLDOWN
    )
    assert _scalar(seen, "visible") == 1.0
    np.testing.assert_allclose(
        _vec(seen, "opp_pos_local"),
        np.asarray(_SEEN_POS_LOCAL) / POS_SCALE,
        atol=1e-6,
    )

    # The learner teleports around while out of the opponent's view; the held
    # memory must not follow it.
    for live_pos in [
        (50.0, _ARENA_Y, -50.0),
        (-30.0, _ARENA_Y + 10.0, -5.0),
        (0.0, _ARENA_Y, -2.0),  # the mirror of the last-seen pos, on the blind side
        (16.0, _ARENA_Y, -16.0),
    ]:
        obs = mirror.observe(
            _state(replace(_LEARNER, pos=live_pos), _OPPONENT),
            dt=_DT,
            opp_attack_cooldown=_OPP_COOLDOWN,
        )
        assert _scalar(obs, "visible") == 0.0, f"precondition: visible at {live_pos}"
        np.testing.assert_allclose(
            _vec(obs, "opp_pos_local"),
            np.asarray(_SEEN_POS_LOCAL) / POS_SCALE,
            atol=1e-6,
            err_msg=f"the mirrored memory tracked the live position {live_pos}",
        )
        assert _scalar(obs, "in_range") == 0.0
        assert _scalar(obs, "in_crosshair") == 0.0


def test_teleporting_the_hidden_learner_never_updates_the_mirrored_memory_pos():
    """Many random teleports of the unseen learner leave the held pos frozen.

    Each teleport probe is taken at ``dt=0`` (the same instant) so the memory
    stays inside ``MEMORY_TTL`` and does not legitimately age out to ABSENT —
    that isolates exactly the claim under test: teleporting the hidden learner
    never *updates* the frozen last-seen position to track its whereabouts.
    """
    rng = _seed_np()
    mirror = OpponentMirror()
    # Sight it once off to the right so the frozen memory is a clearly
    # identifiable non-zero vector every later window can be pinned against.
    seen = mirror.observe(
        _state(replace(_LEARNER, pos=(2.0, _ARENA_Y, 3.0)), _OPPONENT),
        dt=_DT,
        opp_attack_cooldown=_OPP_COOLDOWN,
    )
    assert _scalar(seen, "visible") == 1.0
    frozen = _vec(seen, "opp_pos_local")
    assert np.any(frozen != 0.0), "the sighting froze an all-zero memory"

    for learner in _hidden_probes(rng, _behind_sweep_positions(rng, n=200)):
        obs = mirror.observe(
            _state(learner, _OPPONENT), dt=0.0, opp_attack_cooldown=_OPP_COOLDOWN
        )
        assert _scalar(obs, "visible") == 0.0, f"precondition: visible at {learner.pos}"
        np.testing.assert_allclose(
            _vec(obs, "opp_pos_local"),
            frozen,
            atol=1e-6,
            err_msg=f"a teleport to {learner.pos} moved the frozen mirrored memory",
        )


def test_mirrored_time_since_seen_ages_from_the_opponents_seat():
    """The age ticks by dt per window while the learner jumps around unseen.

    T5 ages the memory with a stationary learner; this ages it while the hidden
    learner teleports every window, which is what makes it a leak test: the age
    must track elapsed seconds and the position must stay frozen, no matter how
    far the live learner moves in between.
    """
    mirror = OpponentMirror()
    seen = mirror.observe(
        _state(_SEEN_LEARNER, _OPPONENT), dt=_DT, opp_attack_cooldown=_OPP_COOLDOWN
    )
    assert _scalar(seen, "visible") == 1.0
    assert _scalar(seen, "time_since_seen") == 0.0

    teleports = [
        (0.0, _ARENA_Y, -1.0),
        (30.0, _ARENA_Y + 5.0, -30.0),
        (-12.0, _ARENA_Y - 3.0, -0.75),
    ]
    for step, live_pos in enumerate(teleports, start=1):
        obs = mirror.observe(
            _state(replace(_LEARNER, pos=live_pos), _OPPONENT),
            dt=_DT,
            opp_attack_cooldown=_OPP_COOLDOWN,
        )
        assert _scalar(obs, "visible") == 0.0, f"precondition: visible at {live_pos}"
        assert _scalar(obs, "time_since_seen") == pytest.approx(
            step * _DT / MEMORY_TTL_SECONDS
        ), f"the mirrored age did not advance by dt at window {step}"
        np.testing.assert_allclose(
            _vec(obs, "opp_pos_local"),
            np.asarray(_SEEN_POS_LOCAL) / POS_SCALE,
            atol=1e-6,
            err_msg=f"the held position followed the learner to {live_pos}",
        )

    # Keep it hidden until the memory expires: ABSENT zeroes the position and
    # pins the age at its normalized maximum.
    for _ in range(int(math.ceil(MEMORY_TTL_SECONDS / _DT)) + 1):
        obs = mirror.observe(
            _state(_LEARNER, _OPPONENT), dt=_DT, opp_attack_cooldown=_OPP_COOLDOWN
        )
    assert _scalar(obs, "visible") == 0.0
    np.testing.assert_array_equal(_vec(obs, "opp_pos_local"), np.zeros(3))
    assert _scalar(obs, "time_since_seen") == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 4. Every-derived-feature sweep: enumerate the DERIVED fields from the frozen
#    spec and assert each is position-independent on the mirrored seat.
# ---------------------------------------------------------------------------


def test_every_mirrored_derived_field_is_position_independent_when_invisible():
    """Each DERIVED field is constant as the hidden learner's position sweeps.

    Enumerates the derived block straight from ``FIELD_SLICES`` so a
    future-added derived feature that depends on the hidden learner's live
    position trips this. Covers both regimes: ABSENT (never seen) and MEMORY
    (seen once, then hidden).
    """
    rng = _seed_np()
    derived_fields = _derived_fields()

    def derived_values(vec: np.ndarray) -> dict:
        return {name: vec[FIELD_SLICES[name]].copy() for name in derived_fields}

    # --- ABSENT regime: fresh mirror, learner never seen. ---
    base_absent = derived_values(
        _hidden_mirrored_obs(replace(_LEARNER, pos=(0.0, _ARENA_Y, -10.0)))
    )
    for learner in _hidden_probes(rng, _behind_sweep_positions(rng, n=300)):
        values = derived_values(_hidden_mirrored_obs(learner))
        for name in derived_fields:
            np.testing.assert_array_equal(
                values[name],
                base_absent[name],
                err_msg=(
                    f"derived '{name}' moved (ABSENT) for a hidden learner "
                    f"at {learner.pos}"
                ),
            )

    # --- MEMORY regime: identical sighting, then one unseen window at dt. ---
    def memory_derived(learner: _Fighter) -> dict:
        mirror = OpponentMirror()
        mirror.observe(
            _state(_SEEN_LEARNER, _OPPONENT), dt=_DT, opp_attack_cooldown=_OPP_COOLDOWN
        )
        obs = mirror.observe(
            _state(learner, _OPPONENT), dt=_DT, opp_attack_cooldown=_OPP_COOLDOWN
        )
        assert _scalar(obs, "visible") == 0.0, (
            f"precondition failed: the learner at {learner.pos} was VISIBLE"
        )
        return derived_values(obs)

    base_memory = memory_derived(replace(_LEARNER, pos=(0.0, _ARENA_Y, -10.0)))
    for learner in _hidden_probes(rng, _behind_sweep_positions(rng, n=300)):
        values = memory_derived(learner)
        for name in derived_fields:
            np.testing.assert_array_equal(
                values[name],
                base_memory[name],
                err_msg=(
                    f"derived '{name}' moved (MEMORY) for a hidden learner "
                    f"at {learner.pos}"
                ),
            )


def test_all_mirrored_derived_fields_are_zero_when_invisible():
    """Every derived field of the mirrored vector reads exactly 0.0 when unseen."""
    obs = _hidden_mirrored_obs(replace(_LEARNER, pos=(0.0, _ARENA_Y, -7.0)))
    for name in _derived_fields():
        np.testing.assert_array_equal(
            obs[FIELD_SLICES[name]],
            np.zeros_like(obs[FIELD_SLICES[name]]),
            err_msg=f"derived '{name}' nonzero while the learner is invisible",
        )


# ---------------------------------------------------------------------------
# 5. LoS leak on the mirrored seat: identical FOV geometry, blocking obstacle
#    -> visible 0, derived 0, and NO leak through the raycast path.
# ---------------------------------------------------------------------------


def test_mirrored_los_block_yields_no_visible_and_no_derived_leak():
    """Same in-cone geometry: clear -> visible + derived; wall -> hidden, 0."""
    geometry = replace(_LEARNER, pos=(0.0, _ARENA_Y, 2.0))  # dead ahead of the opponent

    clear = _mirrored_obs(geometry)
    assert _scalar(clear, "visible") == 1.0
    assert _scalar(clear, "in_range") == 1.0
    assert _scalar(clear, "in_crosshair") == 1.0

    walled = _mirrored_obs(geometry, los_clear=_block_all)
    assert _scalar(walled, "visible") == 0.0
    np.testing.assert_array_equal(_vec(walled, "opp_pos_local"), np.zeros(3))
    assert _scalar(walled, "in_range") == 0.0
    assert _scalar(walled, "in_crosshair") == 0.0


def test_mirrored_wall_hidden_obs_matches_behind_hidden_obs():
    """A wall-hidden learner in front produces the SAME vector as one behind.

    Both are unseen-and-never-seen (ABSENT): the mirrored observation may not
    distinguish "hidden behind a wall, dead ahead" from "behind me, far away".
    If the front geometry leaked through any field, these two would differ.
    """
    front_hidden = _hidden_mirrored_obs(
        replace(_LEARNER, pos=(0.0, _ARENA_Y, 2.0)), los_clear=_block_all
    )
    behind_hidden = _hidden_mirrored_obs(replace(_LEARNER, pos=(0.0, _ARENA_Y, -2.0)))
    np.testing.assert_array_equal(
        front_hidden,
        behind_hidden,
        err_msg="wall-hidden front geometry leaked into the mirrored observation",
    )
    validate(front_hidden)
    validate(behind_hidden)


def test_mirrored_los_block_derived_invariant_to_position_in_front():
    """Behind a wall: sweeping the in-cone live position never moves derived."""
    rng = _seed_np()
    derived_fields = _derived_fields()

    def derived_values(learner: _Fighter) -> dict:
        obs = _hidden_mirrored_obs(learner, los_clear=_block_all)
        return {name: obs[FIELD_SLICES[name]].copy() for name in derived_fields}

    base = derived_values(replace(_LEARNER, pos=(0.0, _ARENA_Y, 1.0)))
    for learner in _hidden_probes(rng, _in_cone_positions(rng, n=300)):
        values = derived_values(learner)
        for name in derived_fields:
            np.testing.assert_array_equal(
                values[name],
                base[name],
                err_msg=f"derived '{name}' moved behind the wall at {learner.pos}",
            )


# ---------------------------------------------------------------------------
# 6. Role-swapped geometry — the half of this battery that has no counterpart in
#    tests/test_perception_leak.py. The gate must key off the OPPONENT's yaw and
#    look vector; a mirror that reuses the LEARNER's produces a well-formed
#    vector whose blindness belongs to the wrong fighter.
# ---------------------------------------------------------------------------

#: A role-swap probe distance that is comfortably inside attack range (3.0) so
#: the boundary is never what decides a flag.
_SWAP_RANGE = 2.5


@pytest.mark.parametrize(
    "opponent_yaw, learner_yaw, expect_visible",
    [
        (0.0, math.pi, False),
        (math.pi, 0.0, True),
    ],
    ids=["learners_yaw_would_see_it", "learners_yaw_would_miss_it"],
)
def test_mirrored_gate_keys_off_the_opponents_yaw_not_the_learners(
    opponent_yaw, learner_yaw, expect_visible
):
    """The two yaws disagree about visibility; the OPPONENT's must win.

    The learner sits behind the opponent at ``-z``. The two cases put the
    learner's yaw exactly opposite the opponent's, so a gate keyed off the
    learner's yaw flips the verdict in BOTH directions: case 1 would turn an
    unseen learner visible (an outright fairness leak), case 2 would blind the
    opponent to a learner standing in its crosshair.
    """
    opponent = replace(_OPPONENT, yaw=opponent_yaw, pitch=0.0)
    learner = replace(
        _LEARNER, pos=(0.0, _ARENA_Y, -_SWAP_RANGE), yaw=learner_yaw, pitch=0.0
    )

    obs = _mirrored_obs(learner, opponent)

    assert _scalar(obs, "visible") == (1.0 if expect_visible else 0.0), (
        "the mirrored gate followed the learner's yaw instead of the opponent's"
    )
    if expect_visible:
        # Rotated about the OPPONENT's yaw, the learner is dead ahead at 2.5.
        np.testing.assert_allclose(
            _vec(obs, "opp_pos_local"), (0.0, 0.0, _SWAP_RANGE / POS_SCALE), atol=1e-6
        )
        assert _scalar(obs, "in_range") == 1.0
        assert _scalar(obs, "in_crosshair") == 1.0
    else:
        np.testing.assert_array_equal(_vec(obs, "opp_pos_local"), np.zeros(3))
        assert _scalar(obs, "in_range") == 0.0
        assert _scalar(obs, "in_crosshair") == 0.0


def test_a_learner_that_sees_the_opponent_is_still_unseen_from_the_opponents_seat():
    """Visibility is NOT symmetric, and the mirrored vector must know it.

    The learner is staring straight at the opponent's back from inside attack
    range: from the LEARNER's seat this is a kill shot, with ``visible``,
    ``in_range`` and ``in_crosshair`` all set. From the OPPONENT's seat the same
    instant is total blindness. A mirror that copied the learner's gate would
    hand the frozen snapshot a perfect firing solution on an enemy it cannot
    see.
    """
    opponent = replace(_OPPONENT, yaw=0.0, pitch=0.0)  # looking +z, back turned
    learner = replace(
        _LEARNER, pos=(0.0, _ARENA_Y, -_SWAP_RANGE), yaw=0.0, pitch=0.0
    )  # looking +z, straight at the opponent

    # The learner's own seat: the same role-generic filter, seats in wire order.
    learner_seat, learner_derived = PerceptionFilter().filter(
        _raw(learner), _raw(opponent), dt=_DT
    )
    assert learner_seat.visible is True, "the learner-seat premise is not a kill shot"
    assert learner_derived.in_range is True
    assert learner_derived.in_crosshair is True

    # The opponent's seat, same instant: nothing.
    obs = _mirrored_obs(learner, opponent)
    assert _scalar(obs, "visible") == 0.0
    np.testing.assert_array_equal(_vec(obs, "opp_pos_local"), np.zeros(3))
    np.testing.assert_array_equal(_vec(obs, "opp_vel_local"), np.zeros(3))
    assert _scalar(obs, "in_range") == 0.0
    assert _scalar(obs, "in_crosshair") == 0.0
    assert _scalar(obs, "time_since_seen") == pytest.approx(1.0)


def test_mirrored_crosshair_uses_the_opponents_look_vector():
    """``in_crosshair`` is measured off the OPPONENT's look axis, not the learner's.

    The learner is inside the opponent's FOV cone and inside attack range — so
    ``visible`` and ``in_range`` are both set — but ~20 deg off the opponent's
    look axis, well outside the 10 deg crosshair window, while pointing dead at
    the opponent itself. The mirrored ``in_crosshair`` must be 0 even though the
    learner's own is 1: distance is symmetric between the seats, but aim is not.
    """
    offset_x = 0.9
    opponent = replace(_OPPONENT, yaw=math.pi, pitch=0.0)  # looking -z, at the learner
    learner = replace(
        _LEARNER,
        pos=(offset_x, _ARENA_Y, -_SWAP_RANGE),
        # Aim straight back at the opponent: yaw such that the look vector
        # (-sin yaw, 0, cos yaw) points along (-offset_x, 0, +range).
        yaw=math.atan2(offset_x, _SWAP_RANGE),
        pitch=0.0,
    )

    off_axis_deg = math.degrees(math.atan2(offset_x, _SWAP_RANGE))
    assert CROSSHAIR_DEGREES < off_axis_deg < FOV_DEGREES / 2.0, (
        "the probe must sit inside the FOV cone but outside the crosshair window "
        f"for this test to mean anything (got {off_axis_deg:.1f} deg)"
    )

    learner_seat, learner_derived = PerceptionFilter().filter(
        _raw(learner), _raw(opponent), dt=_DT
    )
    assert learner_seat.visible is True
    assert learner_derived.in_crosshair is True, "the learner is not actually aiming"

    obs = _mirrored_obs(learner, opponent)
    assert _scalar(obs, "visible") == 1.0
    assert _scalar(obs, "in_range") == 1.0, "distance is symmetric; this should be set"
    assert _scalar(obs, "in_crosshair") == 0.0, (
        "the mirrored crosshair was measured off the learner's look vector"
    )


# ---------------------------------------------------------------------------
# 7. Reset isolation: per-episode memory must not cross an episode boundary,
#    on the mirror itself and at the env seam the self-play driver actually
#    reads.
# ---------------------------------------------------------------------------


def test_reset_clears_the_mirrored_per_episode_memory():
    """A sighting in episode N must not be remembered in episode N+1."""
    mirror = OpponentMirror()
    seen = mirror.observe(
        _state(_SEEN_LEARNER, _OPPONENT), dt=_DT, opp_attack_cooldown=_OPP_COOLDOWN
    )
    assert _scalar(seen, "visible") == 1.0

    mirror.reset()
    assert mirror.latest is None

    obs = mirror.observe(
        _state(_LEARNER, _OPPONENT), dt=_DT, opp_attack_cooldown=_OPP_COOLDOWN
    )
    assert _scalar(obs, "visible") == 0.0
    # ABSENT, not MEMORY: no last-seen position survived the episode boundary.
    np.testing.assert_array_equal(_vec(obs, "opp_pos_local"), np.zeros(3))
    assert _scalar(obs, "time_since_seen") == pytest.approx(1.0)


def test_after_reset_a_primed_mirror_is_byte_identical_to_a_fresh_one():
    """The strong form: a reset mirror carries NOTHING of the last episode.

    Field-by-field assertions can only catch the leaks someone thought to check.
    Comparing a reset mirror's whole vector against a never-used mirror's, on
    the same hidden window, catches any of them — the previous episode's frozen
    position, its memory age, or a stale cached vector.
    """
    primed = OpponentMirror()
    primed.observe(
        _state(_SEEN_LEARNER, _OPPONENT), dt=_DT, opp_attack_cooldown=_OPP_COOLDOWN
    )
    primed.reset()

    hidden = replace(_LEARNER, pos=(0.0, _ARENA_Y, -6.0))
    after_reset = primed.observe(
        _state(hidden, _OPPONENT), dt=_DT, opp_attack_cooldown=_OPP_COOLDOWN
    )
    fresh = OpponentMirror().observe(
        _state(hidden, _OPPONENT), dt=_DT, opp_attack_cooldown=_OPP_COOLDOWN
    )

    assert _scalar(after_reset, "visible") == 0.0, "precondition: the probe is hidden"
    assert after_reset.tobytes() == fresh.tobytes(), (
        "the reset mirror still carried the previous episode's perception memory"
    )


# --- the production seam: MCPvPEnv(mirror_opponent=True) -------------------


class _ScriptedBridge:
    """Minimal ``BridgeTransport`` stand-in: a FIFO of scripted inbound messages.

    The env's reset exchange is send(reset) -> recv(reset_ack) -> recv(state),
    which is all this battery needs. An empty queue is a test bug and says so
    rather than hanging or inventing a state.
    """

    def __init__(self) -> None:
        self.inbound: List[object] = []
        self.sent: List[object] = []

    def connect(self) -> None:
        pass

    def send(self, obj) -> None:
        self.sent.append(obj)

    def recv(self):
        if not self.inbound:  # pragma: no cover - defensive
            raise AssertionError("_ScriptedBridge: recv() with an empty queue")
        return self.inbound.pop(0)

    def close(self) -> None:
        pass


def _reset_ack() -> ResetAckMsg:
    """A passing read-back gate, so ``reset()`` proceeds to the first state."""
    return ResetAckMsg.from_dict(
        {
            "type": "reset_ack",
            "ok": True,
            "readback": {"self_hp": 20.0, "opp_hp": 20.0},
        }
    )


def test_env_opponent_observation_hides_the_learners_live_position():
    """AC4 at the seam the self-play driver actually reads.

    Everything above drives ``OpponentMirror`` directly; this drives the whole
    production path — ``MCPvPEnv(mirror_opponent=True).reset()`` ingesting a
    wire state and serving the cached vector from ``opponent_observation()``.
    Each iteration is a fresh episode, but this does NOT prove env-level reset
    isolation, and an earlier revision of this docstring wrongly claimed it did.
    Every probe here comes from the behind sweep, so the learner is never
    sighted, ``_last_seen_local`` stays ``None``, and there is no memory for a
    reset to carry. Commenting out ``MCPvPEnv``'s ``self._mirror.reset()``
    leaves this whole file green — verified by mutation.

    That invariant is pinned by
    ``tests/test_mc_pvp_env.py::test_mirror_memory_does_not_leak_into_the_next_episode``
    and ``::test_a_dead_reset_leaves_opponent_observation_raising_not_stale``,
    the only two tests in the repo that die to it. Do not delete those believing
    this file covers them.
    """
    rng = _seed_np()
    bridge = _ScriptedBridge()
    env = MCPvPEnv(
        transport=bridge,
        perception_filter=PerceptionFilter(),
        dt=_DT,
        mirror_opponent=True,
        auto_connect=False,
    )

    baseline = None
    for learner in _hidden_probes(rng, _behind_sweep_positions(rng, n=40)):
        bridge.inbound.extend([_reset_ack(), _state(learner, _OPPONENT)])
        env.reset(seed=0)
        obs = env.opponent_observation()

        assert _scalar(obs, "visible") == 0.0, (
            f"precondition failed: the learner at {learner.pos} was VISIBLE "
            "from the opponent's seat, so this probe tests nothing"
        )
        if baseline is None:
            validate(obs)
            baseline = obs.copy()
            continue
        np.testing.assert_array_equal(
            obs,
            baseline,
            err_msg=(
                "opponent_observation() moved for a hidden learner at "
                f"{learner.pos}"
            ),
        )

    assert not bridge.inbound, "the scripted queue was not fully consumed"


# ---------------------------------------------------------------------------
# Spec sanity: the geometry this battery calls a "kill shot" really is one, so
# the visible-baseline assertions above cannot silently become vacuous.
# ---------------------------------------------------------------------------


def test_mirrored_kill_shot_geometry_is_inside_the_crosshair_window():
    """The fixtures really do exercise the visible branch and both flags."""
    assert ATTACK_RANGE - 0.5 < ATTACK_RANGE
    assert _SWAP_RANGE < ATTACK_RANGE
    # Dead ahead is 0 deg off-axis, trivially inside the crosshair half-angle.
    assert 0.0 <= CROSSHAIR_DEGREES < FOV_DEGREES / 2.0
    assert MEMORY_TTL_SECONDS > 0.0
    # The contract really has OBS_DIM slots and a derived block at the tail.
    assert OBS_DIM == sum(s.stop - s.start for s in FIELD_SLICES.values())

    # ...and the standing sighting fixture is genuinely a visible kill shot.
    obs = _mirrored_obs(_SEEN_LEARNER)
    assert _scalar(obs, "visible") == 1.0
    assert _scalar(obs, "in_range") == 1.0
    assert _scalar(obs, "in_crosshair") == 1.0
    assert _scalar(obs, "time_since_seen") == 0.0
