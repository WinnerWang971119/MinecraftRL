"""test_mirror_perception — the opponent-seat observation builder (T5).

``OpponentMirror`` feeds a FROZEN snapshot of the agent's own past policy. Every
bug it can have is silent: the vector always has the right length, the right
dtype, and values inside ``validate``'s bounds, so nothing downstream ever
complains — the frozen net simply stops resembling the policy that earned its
snapshot, and the Elo measured against it stops meaning anything. This suite is
therefore built around the failure modes rather than around the API surface:

  * ``TC6`` — **seat-swap symmetry**, the load-bearing test. Mirroring a state
    must be byte-for-byte identical to running the ordinary learner-seat
    ``MCPvPEnv._state_to_obs`` over the SAME state with the two fighters' raw
    blocks exchanged. Provenance caveat that makes the comparison well-defined:
    the learner's ``attack_cooldown`` rides the wire while the mirrored seat's
    is supplied by the caller (no wire channel exists for it), so the swapped
    state is built with the mirrored cooldown in its self block and the same
    number is handed to ``observe()``.
  * ``TC7`` — the velocity rotates about the **OPPONENT's** yaw. A fighter
    moving along its own forward axis must read the same ``vel_local`` from
    either seat. Rotating about the learner's yaw instead produces a
    well-formed vector describing motion in a direction nobody is moving.
  * ``TC11`` — a wire block missing ``on_ground``/``held_item`` is REFUSED,
    naming all five surfaces that must agree. Defaulting them would pin two
    features at a constant for the whole match.
  * Gating — the learner block goes through a real ``PerceptionFilter`` with
    the seats reversed (FOV, memory, aging, per-episode reset), and reveals
    nothing about the learner's live position while it is unseen.

House conventions: no sockets, no live server, hand-authored fixtures, numpy
only (no torch). Geometry follows ``env/perception_filter``'s convention — yaw 0
looks toward world ``+z``, an opponent dead ahead at range ``r`` lands at local
``(0, 0, r)``.
"""

import math
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Tuple

import numpy as np
import pytest

from bridge.messages import StateMsg
from env.mc_pvp_env import MCPvPEnv
from env.mirror_perception import (
    REQUIRED_OPPONENT_WIRE_FIELDS,
    WIRE_SURFACES,
    MirrorWireError,
    OpponentMirror,
)
from env.observation_spec import (
    FIELD_SLICES,
    HELD_ITEM_VOCAB_SIZE,
    MAX_HEALTH,
    MEMORY_TTL_SECONDS,
    OBS_DIM,
    POS_SCALE,
    ObservationError,
    held_item_id,
    validate,
)
from env.perception_filter import ATTACK_RANGE, PerceptionFilter

# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------

#: Decision interval used by every test (seconds). Both seats must be advanced
#: by the same dt or their memory ages diverge and TC6 stops being meaningful.
_DT = 0.2


@dataclass(frozen=True)
class _Fighter:
    """One fighter's RAW world-frame state, seat-agnostic.

    Deliberately seat-agnostic: the same object is packed into the ``self`` wire
    block in one state and into the ``opponent`` block in the swapped one, which
    is what makes the TC6 comparison an exact mirror rather than an approximate
    one.
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


def _state(
    learner: _Fighter, opponent: _Fighter, *, learner_cooldown: float = 1.0
) -> StateMsg:
    """Build a valid ``StateMsg`` from two fighters (learner in the self seat)."""
    return StateMsg.from_dict(
        {
            "type": "state",
            "self": _self_block(learner, learner_cooldown),
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


class _NullTransport:
    """Fake ``BridgeTransport``: the env is built only to call ``_state_to_obs``.

    That method touches no transport at all, so any I/O here is a test bug and
    says so loudly rather than hanging or inventing a state.
    """

    def connect(self) -> None:
        pass

    def send(self, obj) -> None:  # pragma: no cover - defensive
        raise AssertionError("_state_to_obs must not send on the transport")

    def recv(self):  # pragma: no cover - defensive
        raise AssertionError("_state_to_obs must not read from the transport")

    def close(self) -> None:
        pass


def _learner_env() -> MCPvPEnv:
    """An env with a FRESH filter, used only as the learner-seat reference."""
    return MCPvPEnv(
        transport=_NullTransport(),
        perception_filter=PerceptionFilter(),
        dt=_DT,
        auto_connect=False,
    )


def _scalar(obs: np.ndarray, name: str) -> float:
    """Read a scalar observation field by its frozen name."""
    return float(obs[FIELD_SLICES[name]][0])


def _vec(obs: np.ndarray, name: str) -> np.ndarray:
    """Read a vector observation field by its frozen name."""
    return np.asarray(obs[FIELD_SLICES[name]], dtype=np.float64)


# Duelling geometry: the learner at the origin, the opponent 2.5 blocks ahead on
# +z and turned around to face it. Both yaws/pitches are deliberately off-axis
# so an accidental symmetry cannot hide a wrong rotation, yet the learner stays
# inside the opponent's FOV cone (~8.3 deg off its look vector), inside the
# crosshair window (10 deg) and inside attack range (2.5 <= 3.0) — so the
# mirrored vector exercises the VISIBLE-NOW branch and both derived flags.
_LEARNER = _Fighter(
    pos=(0.0, 64.0, 0.0),
    yaw=0.10,
    pitch=0.05,
    vel=(0.10, 0.00, 0.05),
    health=17.0,
    on_ground=True,
    held_item="iron_sword",
)
_OPPONENT = _Fighter(
    pos=(0.0, 64.0, 2.5),
    yaw=math.pi - 0.12,
    pitch=-0.08,
    vel=(-0.20, 0.03, 0.07),
    health=11.0,
    on_ground=False,
    held_item="stone_axe",
)

#: The opponent's swing meter, shadow-tracked in Python by the env (there is no
#: ``attack_cooldown`` channel on the opponent wire block).
_OPP_COOLDOWN = 0.375
#: The learner's swing meter, which DOES ride the wire.
_LEARNER_COOLDOWN = 1.0


# ---------------------------------------------------------------------------
# TC6 — seat-swap symmetry (AC3). The load-bearing test.
# ---------------------------------------------------------------------------


def test_tc6_seat_swap_symmetry_is_byte_for_byte():
    """Mirroring a state == learner-seat obs of the same state, seats exchanged.

    Both sides describe the SAME seat (the opponent's), reached two different
    ways, so every byte must agree: health, angles, the locally-rotated
    velocity, the wire-sourced on_ground/held_item, the cooldown, the gated
    learner block and the derived flags.
    """
    state = _state(_LEARNER, _OPPONENT, learner_cooldown=_LEARNER_COOLDOWN)
    # Same cooldown provenance on both sides: the swapped state's SELF block
    # carries the number that is handed to observe() as the shadow value.
    swapped = _state(_OPPONENT, _LEARNER, learner_cooldown=_OPP_COOLDOWN)

    mirrored = OpponentMirror().observe(
        state, dt=_DT, opp_attack_cooldown=_OPP_COOLDOWN
    )
    reference = _learner_env()._state_to_obs(swapped)

    assert mirrored.dtype == np.float32
    assert reference.dtype == np.float32
    assert mirrored.tobytes() == reference.tobytes()

    # Guard against the vacuous pass: two all-zero ABSENT vectors would also be
    # byte-equal. This geometry must light up the whole opponent block.
    assert _scalar(mirrored, "visible") == 1.0
    assert _scalar(mirrored, "in_range") == 1.0
    assert _scalar(mirrored, "in_crosshair") == 1.0
    assert _scalar(mirrored, "health") == pytest.approx(_OPPONENT.health / MAX_HEALTH)


def test_tc6_symmetry_holds_when_neither_fighter_can_see_the_other():
    """Seat-swap symmetry also holds in the ABSENT regime (back to back)."""
    learner = replace(_LEARNER, yaw=0.0)          # looking +z, away from -z
    opponent = replace(_OPPONENT, pos=(0.0, 64.0, -4.0), yaw=math.pi)  # behind, facing -z

    state = _state(learner, opponent, learner_cooldown=_LEARNER_COOLDOWN)
    swapped = _state(opponent, learner, learner_cooldown=_OPP_COOLDOWN)

    mirrored = OpponentMirror().observe(
        state, dt=_DT, opp_attack_cooldown=_OPP_COOLDOWN
    )
    reference = _learner_env()._state_to_obs(swapped)

    assert mirrored.tobytes() == reference.tobytes()
    assert _scalar(mirrored, "visible") == 0.0


def test_tc6_symmetry_survives_a_multi_step_episode():
    """Symmetry must hold step after step, not just on a fresh filter.

    The two paths each carry their own perception memory; if the mirror aged its
    filter differently (skipped dt, or reset mid-episode) the vectors would only
    diverge from the second window onward.
    """
    mirror = OpponentMirror()
    env = _learner_env()

    # Window 1 visible, windows 2-3 hidden (the opponent turns away), window 4
    # visible again: exercises VISIBLE -> MEMORY -> MEMORY -> re-sighting.
    opponent_yaws = [math.pi - 0.12, 0.0, 0.0, math.pi - 0.12]
    for step, opp_yaw in enumerate(opponent_yaws):
        opponent = replace(_OPPONENT, yaw=opp_yaw, pos=(0.0, 64.0, 2.5 + 0.1 * step))
        learner = replace(_LEARNER, pos=(0.0, 64.0, 0.05 * step))

        mirrored = mirror.observe(
            _state(learner, opponent, learner_cooldown=_LEARNER_COOLDOWN),
            dt=_DT,
            opp_attack_cooldown=_OPP_COOLDOWN,
        )
        reference = env._state_to_obs(
            _state(opponent, learner, learner_cooldown=_OPP_COOLDOWN)
        )
        assert mirrored.tobytes() == reference.tobytes(), f"diverged at window {step}"


# ---------------------------------------------------------------------------
# TC7 — the velocity rotates about the OPPONENT's yaw.
# ---------------------------------------------------------------------------


def _forward_velocity(yaw: float, speed: float) -> Tuple[float, float, float]:
    """World-frame velocity of a fighter sprinting along its OWN forward axis."""
    return (-speed * math.sin(yaw), 0.0, speed * math.cos(yaw))


@pytest.mark.parametrize(
    "opponent_yaw",
    [0.0, 0.7, math.pi / 2.0, math.pi, -2.4],
    ids=["north", "off_axis", "east", "south", "negative"],
)
def test_tc7_velocity_rotates_about_the_opponents_own_yaw(opponent_yaw):
    """Forward motion reads as ``+Z_local`` from either seat, at any yaw.

    The learner sits at yaw 0 on purpose: a rotation about the LEARNER's yaw is
    then the identity, so a mirror that used the wrong yaw would emit the raw
    world velocity — which this asserts it does not.
    """
    speed = 0.42
    learner = replace(_LEARNER, yaw=0.0, pitch=0.0, vel=_forward_velocity(0.0, speed))
    opponent = replace(
        _OPPONENT,
        yaw=opponent_yaw,
        pitch=0.0,
        vel=_forward_velocity(opponent_yaw, speed),
    )

    mirrored = OpponentMirror().observe(
        _state(learner, opponent, learner_cooldown=_LEARNER_COOLDOWN),
        dt=_DT,
        opp_attack_cooldown=_OPP_COOLDOWN,
    )
    learner_seat = _learner_env()._state_to_obs(
        _state(learner, opponent, learner_cooldown=_LEARNER_COOLDOWN)
    )

    mirrored_vel = _vec(mirrored, "vel_local")
    # Both fighters move along their own forward axis at the same speed, so both
    # seats must report the identical local-frame velocity.
    np.testing.assert_allclose(mirrored_vel, (0.0, 0.0, speed), atol=1e-6)
    np.testing.assert_allclose(mirrored_vel, _vec(learner_seat, "vel_local"), atol=1e-6)

    if abs(opponent_yaw) > 1e-9:
        # The wrong-yaw bug (rotating about the learner's yaw == identity here)
        # would have emitted the world velocity unchanged.
        assert not np.allclose(mirrored_vel, opponent.vel, atol=1e-3)


def test_tc7_vertical_velocity_is_preserved_by_the_rotation():
    """The yaw rotation is about world up: the ``y`` component passes through."""
    opponent = replace(_OPPONENT, yaw=1.3, pitch=0.0, vel=(0.0, -0.35, 0.0))
    mirrored = OpponentMirror().observe(
        _state(_LEARNER, opponent, learner_cooldown=_LEARNER_COOLDOWN),
        dt=_DT,
        opp_attack_cooldown=_OPP_COOLDOWN,
    )
    np.testing.assert_allclose(_vec(mirrored, "vel_local"), (0.0, -0.35, 0.0), atol=1e-6)


# ---------------------------------------------------------------------------
# TC11 — refuse when the T1 wire fields are absent.
# ---------------------------------------------------------------------------


def _stub_opponent(**overrides) -> SimpleNamespace:
    """An opponent block as a loose stand-in, so a field can be truly ABSENT.

    ``bridge.messages.OpponentState`` is a frozen dataclass whose ``from_dict``
    already raises on a missing key, so the shape this guard has to catch is a
    block that reached Python some other way — e.g. a pre-T1 bridge, or one of
    the five surfaces silently stripping the fields back out.
    """
    fields = {
        "pos": [0.0, 64.0, 2.5],
        "yaw": 0.0,
        "pitch": 0.0,
        "velocity": [0.0, 0.0, 0.0],
        "health": 20.0,
        "on_ground": True,
        "held_item": "iron_sword",
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _stub_state(opponent: SimpleNamespace) -> SimpleNamespace:
    """A ``state``-shaped stand-in carrying a real learner block."""
    return SimpleNamespace(
        self_state=_state(_LEARNER, _OPPONENT).self_state, opponent=opponent
    )


@pytest.mark.parametrize("missing", list(REQUIRED_OPPONENT_WIRE_FIELDS))
@pytest.mark.parametrize("how", ["absent", "none"])
def test_tc11_mirror_refuses_when_a_wire_field_is_missing(missing, how):
    """Refuse loudly — never substitute a constant for an absent wire field."""
    opponent = _stub_opponent()
    if how == "absent":
        delattr(opponent, missing)
    else:
        setattr(opponent, missing, None)

    mirror = OpponentMirror()
    with pytest.raises(MirrorWireError) as excinfo:
        mirror.observe(_stub_state(opponent), dt=_DT, opp_attack_cooldown=_OPP_COOLDOWN)

    message = str(excinfo.value)
    assert missing in message
    # The refusal must name every surface that can strip the field, because the
    # message is the only clue to which one was missed.
    for surface in ("bridge/schema.json", "bridge/messages.py", "bridge/transport.js"):
        assert surface in message
    assert message.count("bridge/bot.js") == 2  # _snapshotOpponent AND assembleStateMsg
    assert "_snapshotOpponent" in message
    assert "assembleStateMsg" in message
    assert len(WIRE_SURFACES) == 5
    for surface in WIRE_SURFACES:
        assert surface in message

    # A refused window must not leave a half-built vector behind.
    assert mirror.latest is None


def test_tc11_refusal_leaves_the_previous_vector_untouched():
    """A bad window must not corrupt the cache built by a good one."""
    mirror = OpponentMirror()
    good = mirror.observe(
        _state(_LEARNER, _OPPONENT), dt=_DT, opp_attack_cooldown=_OPP_COOLDOWN
    )

    broken = _stub_opponent()
    delattr(broken, "held_item")
    with pytest.raises(MirrorWireError):
        mirror.observe(_stub_state(broken), dt=_DT, opp_attack_cooldown=_OPP_COOLDOWN)

    assert mirror.latest is not None
    assert mirror.latest.tobytes() == good.tobytes()


def test_tc11_empty_hand_is_a_legitimate_held_item():
    """``""`` is the documented empty-hand encoding, NOT an absent field."""
    opponent = replace(_OPPONENT, held_item="")
    obs = OpponentMirror().observe(
        _state(_LEARNER, opponent), dt=_DT, opp_attack_cooldown=_OPP_COOLDOWN
    )
    assert _scalar(obs, "held_item") == pytest.approx(0.0)


def test_state_without_both_seats_is_refused():
    """A message missing a seat is a wiring error, not an empty observation."""
    mirror = OpponentMirror()
    with pytest.raises(MirrorWireError):
        mirror.observe(
            SimpleNamespace(opponent=_stub_opponent()),
            dt=_DT,
            opp_attack_cooldown=_OPP_COOLDOWN,
        )


# ---------------------------------------------------------------------------
# The SELF block comes off the OPPONENT's wire block.
# ---------------------------------------------------------------------------


def test_self_block_is_the_opponents_own_wire_state():
    """Every SELF feature must trace to ``state.opponent``, not to the learner."""
    obs = OpponentMirror().observe(
        _state(_LEARNER, _OPPONENT, learner_cooldown=_LEARNER_COOLDOWN),
        dt=_DT,
        opp_attack_cooldown=_OPP_COOLDOWN,
    )

    assert _scalar(obs, "health") == pytest.approx(_OPPONENT.health / MAX_HEALTH)
    assert _scalar(obs, "yaw_sin") == pytest.approx(math.sin(_OPPONENT.yaw), abs=1e-6)
    assert _scalar(obs, "yaw_cos") == pytest.approx(math.cos(_OPPONENT.yaw), abs=1e-6)
    assert _scalar(obs, "pitch_sin") == pytest.approx(math.sin(_OPPONENT.pitch), abs=1e-6)
    assert _scalar(obs, "pitch_cos") == pytest.approx(math.cos(_OPPONENT.pitch), abs=1e-6)
    # T1's two fields, straight off the wire — the opponent is airborne and
    # holding a stone axe while the learner is grounded with an iron sword.
    assert _scalar(obs, "on_ground") == 0.0
    assert _scalar(obs, "held_item") == pytest.approx(
        held_item_id("stone_axe") / HELD_ITEM_VOCAB_SIZE
    )
    # ...supplied by the caller: no wire channel carries the opponent's meter.
    assert _scalar(obs, "attack_cooldown") == pytest.approx(_OPP_COOLDOWN)


def test_on_ground_and_held_item_track_the_wire_rather_than_a_constant():
    """Flipping the wire values must move indices 8 and 9."""
    mirror = OpponentMirror()
    grounded = mirror.observe(
        _state(_LEARNER, replace(_OPPONENT, on_ground=True, held_item="diamond_sword")),
        dt=_DT,
        opp_attack_cooldown=_OPP_COOLDOWN,
    ).copy()
    mirror.reset()
    airborne = mirror.observe(
        _state(_LEARNER, replace(_OPPONENT, on_ground=False, held_item="bow")),
        dt=_DT,
        opp_attack_cooldown=_OPP_COOLDOWN,
    )

    assert _scalar(grounded, "on_ground") == 1.0
    assert _scalar(airborne, "on_ground") == 0.0
    assert _scalar(grounded, "held_item") == pytest.approx(
        held_item_id("diamond_sword") / HELD_ITEM_VOCAB_SIZE
    )
    assert _scalar(airborne, "held_item") == pytest.approx(
        held_item_id("bow") / HELD_ITEM_VOCAB_SIZE
    )


def test_learner_self_only_fields_never_reach_the_mirrored_vector():
    """The learner's health/cooldown/on_ground/held_item are invisible here.

    The layout has no opponent-health slot at all, and the learner's own SELF
    features belong to its seat. Changing every one of them must not move a
    single byte of the opponent's observation.
    """
    mirror_a = OpponentMirror()
    baseline = mirror_a.observe(
        _state(_LEARNER, _OPPONENT, learner_cooldown=1.0),
        dt=_DT,
        opp_attack_cooldown=_OPP_COOLDOWN,
    )

    altered_learner = replace(
        _LEARNER, health=2.0, on_ground=False, held_item="netherite_axe"
    )
    mirror_b = OpponentMirror()
    altered = mirror_b.observe(
        _state(altered_learner, _OPPONENT, learner_cooldown=0.0),
        dt=_DT,
        opp_attack_cooldown=_OPP_COOLDOWN,
    )

    assert altered.tobytes() == baseline.tobytes()


# ---------------------------------------------------------------------------
# Gating: the learner block is perception-filtered from the OPPONENT's eyes.
# ---------------------------------------------------------------------------


def test_learner_outside_the_opponents_fov_is_not_visible():
    """Turn the opponent's back on the learner: nothing about it may show."""
    opponent = replace(_OPPONENT, yaw=0.0, pitch=0.0)  # looking +z; learner is at -z
    obs = OpponentMirror().observe(
        _state(_LEARNER, opponent), dt=_DT, opp_attack_cooldown=_OPP_COOLDOWN
    )

    assert _scalar(obs, "visible") == 0.0
    np.testing.assert_array_equal(_vec(obs, "opp_pos_local"), np.zeros(3))
    np.testing.assert_array_equal(_vec(obs, "opp_vel_local"), np.zeros(3))
    assert _scalar(obs, "time_since_seen") == pytest.approx(1.0)
    assert _scalar(obs, "in_range") == 0.0
    assert _scalar(obs, "in_crosshair") == 0.0


def test_visible_learner_lands_in_front_of_the_opponent():
    """A learner dead ahead of the opponent reads as ``+Z_local`` at its range."""
    opponent = replace(_OPPONENT, pos=(0.0, 64.0, 2.5), yaw=math.pi, pitch=0.0)
    learner = replace(_LEARNER, pos=(0.0, 64.0, 0.0))

    obs = OpponentMirror().observe(
        _state(learner, opponent), dt=_DT, opp_attack_cooldown=_OPP_COOLDOWN
    )

    assert _scalar(obs, "visible") == 1.0
    np.testing.assert_allclose(
        _vec(obs, "opp_pos_local"), (0.0, 0.0, 2.5 / POS_SCALE), atol=1e-6
    )
    assert _scalar(obs, "time_since_seen") == 0.0
    assert 2.5 <= ATTACK_RANGE
    assert _scalar(obs, "in_range") == 1.0


def test_memory_holds_the_last_seen_position_then_ages_out():
    """VISIBLE -> MEMORY -> ABSENT, driven by the dt the caller passes."""
    mirror = OpponentMirror()
    seen = replace(_OPPONENT, yaw=math.pi, pitch=0.0)
    turned_away = replace(seen, yaw=0.0)

    visible = mirror.observe(
        _state(_LEARNER, seen), dt=_DT, opp_attack_cooldown=_OPP_COOLDOWN
    ).copy()
    assert _scalar(visible, "visible") == 1.0

    remembered = mirror.observe(
        _state(_LEARNER, turned_away), dt=_DT, opp_attack_cooldown=_OPP_COOLDOWN
    ).copy()
    assert _scalar(remembered, "visible") == 0.0
    # The position is HELD at its last-seen value, and the age starts ticking.
    np.testing.assert_allclose(
        _vec(remembered, "opp_pos_local"), _vec(visible, "opp_pos_local"), atol=0.0
    )
    assert _scalar(remembered, "time_since_seen") == pytest.approx(
        _DT / MEMORY_TTL_SECONDS
    )

    # Keep looking away until the memory expires.
    steps_to_expiry = int(math.ceil(MEMORY_TTL_SECONDS / _DT)) + 1
    for _ in range(steps_to_expiry):
        obs = mirror.observe(
            _state(_LEARNER, turned_away), dt=_DT, opp_attack_cooldown=_OPP_COOLDOWN
        )

    assert _scalar(obs, "visible") == 0.0
    np.testing.assert_array_equal(_vec(obs, "opp_pos_local"), np.zeros(3))
    assert _scalar(obs, "time_since_seen") == pytest.approx(1.0)


def test_mirrored_observation_hides_the_learners_live_position():
    """While unseen, the learner's real position must move NO byte of the vector.

    The mirrored counterpart of the fairness leak battery: the frozen policy is
    entitled to the same blindness the learner trained under.
    """
    mirror = OpponentMirror()
    turned_away = replace(_OPPONENT, yaw=0.0, pitch=0.0)

    baseline = mirror.observe(
        _state(_LEARNER, turned_away), dt=_DT, opp_attack_cooldown=_OPP_COOLDOWN
    ).copy()

    rng = np.random.default_rng(20260819)
    for _ in range(50):
        hidden = replace(
            _LEARNER,
            pos=tuple(
                float(v) for v in rng.uniform(-12.0, 12.0, size=3) + (0.0, 64.0, 0.0)
            ),
            vel=tuple(float(v) for v in rng.uniform(-0.5, 0.5, size=3)),
        )
        # A fresh mirror per probe: only the hidden learner's live state varies,
        # so any change in the bytes is a leak, not accumulated memory age.
        probe = OpponentMirror().observe(
            _state(hidden, turned_away), dt=_DT, opp_attack_cooldown=_OPP_COOLDOWN
        )
        # Guard the premise before trusting the byte comparison below: if a
        # probe ever lands inside the opponent's FOV cone, the vector SHOULD
        # change (visible position is not a leak), and a byte mismatch there
        # would misread as a perception leak. Checking visibility first makes
        # that failure say "probe was visible" instead of "vector changed".
        assert _scalar(probe, "visible") == 0.0, "probe was visible"
        assert probe.tobytes() == baseline.tobytes()


def test_reset_clears_the_per_episode_perception_memory():
    """Memory must not leak across episodes (TC10's mirror-side invariant)."""
    mirror = OpponentMirror()
    seen = replace(_OPPONENT, yaw=math.pi, pitch=0.0)
    turned_away = replace(seen, yaw=0.0)

    mirror.observe(_state(_LEARNER, seen), dt=_DT, opp_attack_cooldown=_OPP_COOLDOWN)
    mirror.reset()

    assert mirror.latest is None

    after_reset = mirror.observe(
        _state(_LEARNER, turned_away), dt=_DT, opp_attack_cooldown=_OPP_COOLDOWN
    )
    # ABSENT, not MEMORY: no last-seen position survived the episode boundary.
    np.testing.assert_array_equal(_vec(after_reset, "opp_pos_local"), np.zeros(3))
    assert _scalar(after_reset, "time_since_seen") == pytest.approx(1.0)


def test_reset_restarts_time_since_seen_at_zero_on_re_sighting():
    """After a reset the first sighting is fresh, not aged by the last episode."""
    mirror = OpponentMirror()
    seen = replace(_OPPONENT, yaw=math.pi, pitch=0.0)
    turned_away = replace(seen, yaw=0.0)

    mirror.observe(_state(_LEARNER, seen), dt=_DT, opp_attack_cooldown=_OPP_COOLDOWN)
    for _ in range(3):
        mirror.observe(
            _state(_LEARNER, turned_away), dt=_DT, opp_attack_cooldown=_OPP_COOLDOWN
        )
    mirror.reset()

    obs = mirror.observe(
        _state(_LEARNER, seen), dt=_DT, opp_attack_cooldown=_OPP_COOLDOWN
    )
    assert _scalar(obs, "visible") == 1.0
    assert _scalar(obs, "time_since_seen") == 0.0


def test_each_mirror_owns_an_independent_filter():
    """Two arenas must not share perception memory through a class attribute."""
    seen = replace(_OPPONENT, yaw=math.pi, pitch=0.0)
    turned_away = replace(seen, yaw=0.0)

    primed = OpponentMirror()
    primed.observe(_state(_LEARNER, seen), dt=_DT, opp_attack_cooldown=_OPP_COOLDOWN)

    fresh = OpponentMirror()
    obs = fresh.observe(
        _state(_LEARNER, turned_away), dt=_DT, opp_attack_cooldown=_OPP_COOLDOWN
    )
    np.testing.assert_array_equal(_vec(obs, "opp_pos_local"), np.zeros(3))
    assert _scalar(obs, "time_since_seen") == pytest.approx(1.0)


def test_injected_line_of_sight_test_gates_the_learner():
    """A blocked raycast hides a learner that is otherwise dead in the cone."""
    blocked = OpponentMirror(
        perception_filter=PerceptionFilter(los_clear=lambda eye, target: False)
    )
    seen = replace(_OPPONENT, yaw=math.pi, pitch=0.0)

    obs = blocked.observe(
        _state(_LEARNER, seen), dt=_DT, opp_attack_cooldown=_OPP_COOLDOWN
    )
    assert _scalar(obs, "visible") == 0.0
    assert _scalar(obs, "in_range") == 0.0
    assert _scalar(obs, "in_crosshair") == 0.0


# ---------------------------------------------------------------------------
# Output contract (AC2) and the ``latest`` cache.
# ---------------------------------------------------------------------------


def test_observe_returns_a_validated_obs_dim_float32_vector():
    obs = OpponentMirror().observe(
        _state(_LEARNER, _OPPONENT), dt=_DT, opp_attack_cooldown=_OPP_COOLDOWN
    )
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32
    validate(obs)  # raises on any violation


def test_latest_is_none_before_the_first_observation():
    assert OpponentMirror().latest is None


def test_latest_returns_the_most_recent_vector():
    mirror = OpponentMirror()
    first = mirror.observe(
        _state(_LEARNER, _OPPONENT), dt=_DT, opp_attack_cooldown=_OPP_COOLDOWN
    )
    assert mirror.latest is first
    assert mirror.latest.tobytes() == first.tobytes()

    second = mirror.observe(
        _state(_LEARNER, replace(_OPPONENT, health=4.0)),
        dt=_DT,
        opp_attack_cooldown=0.5,
    )
    assert mirror.latest is second
    assert _scalar(mirror.latest, "health") == pytest.approx(4.0 / MAX_HEALTH)


def test_repeated_reads_of_latest_are_byte_identical():
    """AC2's idempotency half: reading the cache never re-runs the gate."""
    mirror = OpponentMirror()
    mirror.observe(_state(_LEARNER, _OPPONENT), dt=_DT, opp_attack_cooldown=_OPP_COOLDOWN)

    first_read = mirror.latest.tobytes()
    for _ in range(5):
        assert mirror.latest.tobytes() == first_read


def test_negative_dt_is_rejected_before_the_memory_advances():
    """A bad dt must fail loudly rather than silently rewinding memory age."""
    mirror = OpponentMirror()
    seen = replace(_OPPONENT, yaw=math.pi, pitch=0.0)
    turned_away = replace(seen, yaw=0.0)
    mirror.observe(_state(_LEARNER, seen), dt=_DT, opp_attack_cooldown=_OPP_COOLDOWN)

    with pytest.raises(ValueError, match="dt must be >= 0"):
        mirror.observe(
            _state(_LEARNER, seen), dt=-1.0, opp_attack_cooldown=_OPP_COOLDOWN
        )

    # The rejected window advanced nothing: the next real window is the FIRST dt
    # since the sighting. Had the -1.0 landed, the age would have gone negative
    # and this would read 0.0 (clamped) instead.
    obs = mirror.observe(
        _state(_LEARNER, turned_away), dt=_DT, opp_attack_cooldown=_OPP_COOLDOWN
    )
    assert _scalar(obs, "time_since_seen") == pytest.approx(_DT / MEMORY_TTL_SECONDS)


def test_out_of_range_cooldown_fails_validation():
    """The cooldown is packed unclamped, exactly as the learner seat packs it."""
    with pytest.raises(ObservationError):
        OpponentMirror().observe(
            _state(_LEARNER, _OPPONENT), dt=_DT, opp_attack_cooldown=7.5
        )
