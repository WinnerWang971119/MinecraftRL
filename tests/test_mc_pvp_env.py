"""Tests for MCPvPEnv — the Gym-style env over the bridge (T9).

No live Minecraft server is touched: every test injects a FAKE bridge transport
that returns scripted ``StateMsg`` / ``ResetAckMsg`` dataclasses and records what
the env sent. This file also doubles as the reference implementation of the
fake-bridge contract documented at the bottom of ``env/mc_pvp_env.py`` (T10/T20
reuse ``ScriptedBridge``).

Covered behaviors:
  * reset() returns a valid OBS_DIM vector (passes ``observation_spec.validate``).
  * reset() retries ONCE on ``reset_ack.ok == False`` then RAISES on a second
    failure (read-back gate / AC7 protection).
  * step() returns ``(obs, reward, done, info)``; obs validates; a ``damage_dealt``
    event yields a positive reward contribution; ``i_died`` ends the episode as a
    terminal loss; reaching MAX_EPISODE_STEPS ends it via timeout.
  * Perception gating is actually applied end-to-end: an opponent OUTSIDE the FOV
    yields ``visible == 0`` with derived ``in_range``/``in_crosshair == 0`` (no
    leak), while an in-FOV opponent yields ``visible == 1``.
  * ``info`` exposes the per-reward-component breakdown.
  * A simulated bridge disconnect raises ``BridgeError`` (and triggers a
    reconnect attempt).
"""

import math

import numpy as np
import pytest

from agent.actions import Macro, N_ACTIONS
from agent.contract_config import MAX_EPISODE_STEPS
from agent.reward_config import RewardConfig
from bridge.messages import ResetAckMsg, StateMsg
from env.mc_pvp_env import (
    REWARD_COMPONENT_KEYS,
    BridgeError,
    ExhibitionConfig,
    MCPvPEnv,
)
from env.observation_spec import OBS_DIM, Obs, validate
from env.perception_filter import PerceptionFilter


# ---------------------------------------------------------------------------
# Fake bridge transport — the reference implementation of the fake-bridge
# contract (see env/mc_pvp_env.py "FAKE-BRIDGE CONTRACT").
# ---------------------------------------------------------------------------


class ScriptedBridge:
    """A fake :class:`~env.mc_pvp_env.BridgeTransport` driven by a scripted queue.

    ``inbound`` is the ordered list of messages ``recv()`` will return (one per
    call). ``sent`` records every wire dict the env sent. ``connects`` /
    ``closes`` count lifecycle calls so reconnect behavior can be asserted.

    A queued item may be the sentinel :class:`Disconnect` to make ``recv()`` raise
    a :class:`BridgeError`, simulating a dropped connection.
    """

    class Disconnect:
        """Sentinel: when ``recv()`` reaches this, it raises ``BridgeError``."""

    def __init__(self, inbound=None):
        self.inbound = list(inbound) if inbound is not None else []
        self.sent = []
        self.connects = 0
        self.closes = 0
        self.is_open = False

    # -- queue management (test helpers) ----------------------------------

    def push(self, *messages):
        """Append more scripted inbound messages to the recv queue."""
        self.inbound.extend(messages)

    # -- BridgeTransport protocol -----------------------------------------

    def connect(self):
        self.connects += 1
        self.is_open = True

    def send(self, obj):
        self.sent.append(dict(obj))

    def recv(self):
        if not self.inbound:
            raise BridgeError("ScriptedBridge: recv() with an empty queue")
        item = self.inbound.pop(0)
        if item is ScriptedBridge.Disconnect or isinstance(
            item, ScriptedBridge.Disconnect
        ):
            raise BridgeError("ScriptedBridge: simulated disconnect")
        return item

    def close(self):
        self.closes += 1
        self.is_open = False


# ---------------------------------------------------------------------------
# Message builders (canonical valid wire shapes -> dataclasses).
# ---------------------------------------------------------------------------


def _reset_ack(ok=True, readback=None):
    return ResetAckMsg.from_dict(
        {
            "type": "reset_ack",
            "ok": ok,
            "readback": readback if readback is not None else {"self_hp": 20.0, "opp_hp": 20.0},
        }
    )


def _state(
    *,
    self_pos=(0.0, 64.0, 0.0),
    self_yaw=0.0,
    self_pitch=0.0,
    self_vel=(0.0, 0.0, 0.0),
    self_health=20.0,
    held_item="iron_sword",
    attack_cooldown=1.0,
    opp_pos=(0.0, 64.0, 2.0),  # dead ahead on +z, in FOV + range by default
    opp_yaw=0.0,
    opp_pitch=0.0,
    opp_vel=(0.0, 0.0, 0.0),
    opp_health=20.0,
    damage_dealt=0.0,
    damage_taken=0.0,
    i_died=False,
    opponent_died=False,
    tick=1,
    code_version="test",
):
    """Build a valid ``StateMsg`` from keyword overrides (sane combat defaults)."""
    return StateMsg.from_dict(
        {
            "type": "state",
            "self": {
                "pos": list(self_pos),
                "yaw": self_yaw,
                "pitch": self_pitch,
                "velocity": list(self_vel),
                "on_ground": True,
                "health": self_health,
                "held_item": held_item,
                "attack_cooldown": attack_cooldown,
            },
            "opponent": {
                "pos": list(opp_pos),
                "yaw": opp_yaw,
                "pitch": opp_pitch,
                "velocity": list(opp_vel),
                "health": opp_health,
            },
            "events": {
                "damage_dealt": damage_dealt,
                "damage_taken": damage_taken,
                "i_died": i_died,
                "opponent_died": opponent_died,
            },
            "arena": {"wall_distances": [8.0, 8.0, 8.0, 8.0]},
            "tick": tick,
            "code_version": code_version,
        }
    )


def _env(bridge, **kwargs):
    """Construct an env over a fake bridge (no socket)."""
    return MCPvPEnv(transport=bridge, **kwargs)


# ---------------------------------------------------------------------------
# reset().
# ---------------------------------------------------------------------------


def test_reset_returns_valid_observation():
    """reset() returns a validated OBS_DIM float32 vector and consumes ack+state."""
    bridge = ScriptedBridge([_reset_ack(ok=True), _state()])
    env = _env(bridge)

    obs = env.reset(seed=123)

    # Shape / dtype / validity.
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32
    validate(obs)  # raises on any violation

    # Sent exactly one reset with the right episode/seed.
    resets = [m for m in bridge.sent if m["type"] == "reset"]
    assert len(resets) == 1
    assert resets[0]["episode"] == 0
    assert resets[0]["seed"] == 123

    # Connected once at construction; queue fully drained (ack + state).
    assert bridge.connects == 1
    assert bridge.inbound == []


def test_reset_seed_none_becomes_zero_on_wire():
    """A None seed serializes as integer 0 (schema requires an int seed)."""
    bridge = ScriptedBridge([_reset_ack(ok=True), _state()])
    env = _env(bridge)
    env.reset(seed=None)
    reset = next(m for m in bridge.sent if m["type"] == "reset")
    assert reset["seed"] == 0


def test_reset_retries_once_on_ok_false_then_succeeds():
    """ok==False triggers exactly one retry; a following ok==True starts the episode."""
    bridge = ScriptedBridge(
        [
            _reset_ack(ok=False),  # first gate fails
            _reset_ack(ok=True),   # retry succeeds
            _state(),              # post-reset initial observation
        ]
    )
    env = _env(bridge)

    obs = env.reset(seed=7)
    validate(obs)

    # Two reset commands were sent (original + one retry).
    resets = [m for m in bridge.sent if m["type"] == "reset"]
    assert len(resets) == 2
    assert bridge.inbound == []  # ack, ack, state all consumed


def test_reset_raises_on_second_ok_false():
    """Two consecutive ok==False -> BridgeError (never start from unverified state)."""
    bridge = ScriptedBridge([_reset_ack(ok=False), _reset_ack(ok=False)])
    env = _env(bridge)

    with pytest.raises(BridgeError, match="read-back gate failed twice"):
        env.reset(seed=1)

    # Exactly two resets attempted, and NO state was requested/consumed.
    resets = [m for m in bridge.sent if m["type"] == "reset"]
    assert len(resets) == 2


def test_reset_raises_if_state_precedes_ack():
    """An out-of-order reply (state before reset_ack) is a loud BridgeError."""
    bridge = ScriptedBridge([_state()])  # wrong: should be a reset_ack first
    env = _env(bridge)
    with pytest.raises(BridgeError, match="expected a reset_ack"):
        env.reset()


# ---------------------------------------------------------------------------
# step() — basic transition, reward sign, info breakdown.
# ---------------------------------------------------------------------------


def _reset_env(bridge):
    """Helper: build an env and drive a successful reset, returning (env, obs0)."""
    env = _env(bridge)
    obs0 = env.reset(seed=0)
    return env, obs0


def test_step_returns_tuple_with_valid_obs():
    """step() returns (obs, reward, done, info) with a validated obs."""
    bridge = ScriptedBridge([_reset_ack(ok=True), _state()])
    env, _ = _reset_env(bridge)
    bridge.push(_state(tick=2))

    obs, reward, done, info = env.step(Macro.APPROACH)

    assert isinstance(obs, np.ndarray) and obs.shape == (OBS_DIM,)
    validate(obs)
    assert isinstance(reward, float)
    assert isinstance(done, bool) and done is False
    assert isinstance(info, dict)

    # The action reached the wire as the integer macro value.
    steps = [m for m in bridge.sent if m["type"] == "step"]
    assert steps[-1]["action"] == int(Macro.APPROACH)


def test_damage_dealt_gives_positive_reward_contribution():
    """A damage_dealt event contributes positively to the reward."""
    bridge = ScriptedBridge([_reset_ack(ok=True), _state()])
    env, _ = _reset_env(bridge)
    # Same geometry, but the agent dealt 6 damage this interval.
    bridge.push(_state(tick=2, damage_dealt=6.0))

    _, reward, done, info = env.step(Macro.ATTACK)

    assert not done
    # The damage component is exactly c_dmg_out * 6.
    cfg = RewardConfig()
    assert info["r_damage_dealt"] == pytest.approx(cfg.c_dmg_out * 6.0)
    assert info["r_damage_dealt"] > 0.0
    # And it dominates the tiny per-step penalty -> net reward positive.
    assert reward > 0.0


def test_damage_taken_gives_negative_reward_contribution():
    """A damage_taken event contributes negatively to the reward."""
    bridge = ScriptedBridge([_reset_ack(ok=True), _state()])
    env, _ = _reset_env(bridge)
    bridge.push(_state(tick=2, damage_taken=4.0))

    _, reward, _, info = env.step(Macro.RETREAT)

    cfg = RewardConfig()
    assert info["r_damage_taken"] == pytest.approx(-cfg.c_dmg_in * 4.0)
    assert info["r_damage_taken"] < 0.0


def test_info_exposes_per_reward_component_breakdown():
    """info carries every named reward component plus raw events/outcome flags."""
    bridge = ScriptedBridge([_reset_ack(ok=True), _state()])
    env, _ = _reset_env(bridge)
    bridge.push(_state(tick=2, damage_dealt=2.0, damage_taken=1.0))

    _, reward, _, info = env.step(Macro.ATTACK)

    # All component keys present.
    for key in REWARD_COMPONENT_KEYS:
        assert key in info, f"missing reward component {key!r}"

    # The components sum to the scalar reward (within fp tolerance).
    component_sum = sum(info[key] for key in REWARD_COMPONENT_KEYS)
    assert component_sum == pytest.approx(reward, abs=1e-6)

    # Raw events and outcome flags are exposed for logging.
    assert info["events"]["damage_dealt"] == pytest.approx(2.0)
    assert info["events"]["damage_taken"] == pytest.approx(1.0)
    assert info["won"] is False and info["lost"] is False and info["timeout"] is False


# ---------------------------------------------------------------------------
# step() — termination paths.
# ---------------------------------------------------------------------------


def test_i_died_ends_episode_as_terminal_loss():
    """events.i_died -> done with a terminal loss penalty in the reward."""
    bridge = ScriptedBridge([_reset_ack(ok=True), _state()])
    env, _ = _reset_env(bridge)
    bridge.push(_state(tick=2, i_died=True, damage_taken=20.0))

    _, reward, done, info = env.step(Macro.IDLE)

    assert done is True
    assert info["lost"] is True
    assert info["won"] is False
    assert info["timeout"] is False
    # Terminal loss subtracts R_terminal_loss.
    cfg = RewardConfig()
    assert info["r_terminal"] == pytest.approx(-cfg.R_terminal_loss)
    assert reward < 0.0


def test_opponent_died_ends_episode_as_terminal_win():
    """events.opponent_died -> done with a terminal win bonus."""
    bridge = ScriptedBridge([_reset_ack(ok=True), _state()])
    env, _ = _reset_env(bridge)
    bridge.push(_state(tick=2, opponent_died=True, damage_dealt=20.0))

    _, reward, done, info = env.step(Macro.ATTACK)

    assert done is True
    assert info["won"] is True
    assert info["lost"] is False
    cfg = RewardConfig()
    assert info["r_terminal"] == pytest.approx(cfg.R_terminal_win)
    assert reward > 0.0


def test_double_death_resolves_as_loss():
    """A same-step i_died AND opponent_died resolves to a loss, never a win."""
    bridge = ScriptedBridge([_reset_ack(ok=True), _state()])
    env, _ = _reset_env(bridge)
    bridge.push(_state(tick=2, i_died=True, opponent_died=True))

    _, _, done, info = env.step(Macro.ATTACK)

    assert done is True
    assert info["lost"] is True
    assert info["won"] is False


def test_timeout_ends_episode_at_max_steps():
    """Reaching max_episode_steps ends the episode via timeout (penalized, not a draw)."""
    # Use a tiny horizon so the test is fast but exercises the real timeout path.
    bridge = ScriptedBridge([_reset_ack(ok=True), _state()])
    env = _env(bridge, max_episode_steps=3)
    env.reset(seed=0)

    done = False
    info = {}
    for i in range(3):
        bridge.push(_state(tick=2 + i))
        _, _, done, info = env.step(Macro.IDLE)

    assert done is True
    assert info["timeout"] is True
    assert info["won"] is False and info["lost"] is False
    assert info["step"] == 3
    # Timeout terminal reward is the configured timeout penalty (anti-kiting).
    assert info["r_terminal"] == pytest.approx(RewardConfig().R_terminal_timeout)


# ---------------------------------------------------------------------------
# EXHIBITION MODE (T3, AC4) — no episode timeout against a human, and no
# auto-restart after a death.
#
# "Disabled" is expressed as ``max_episode_steps=None`` meaning no truncation.
# NOT a sentinel and NOT a large integer: a large integer is a timeout that has
# merely been moved somewhere less convenient to notice, and it would fire in
# the middle of a live match with an audience watching. Three consumers (this
# task, the launcher, and TC16) depend on exactly this form.
# ---------------------------------------------------------------------------


def test_tc16_no_timeout_never_truncates_past_the_frozen_horizon():
    """``max_episode_steps=None`` runs past MAX_EPISODE_STEPS without ending (AC4)."""
    bridge = ScriptedBridge([_reset_ack(ok=True), _state()])
    env = _env(bridge, max_episode_steps=None)
    env.reset(seed=0)

    assert env.max_episode_steps is None

    # Step PAST the frozen horizon, not merely past a small test cap: the
    # regression this guards is somebody re-introducing a hidden ceiling (or
    # "None means the default"), and only crossing the real number catches it.
    for i in range(MAX_EPISODE_STEPS + 1):
        bridge.push(_state(tick=2 + i))
        _, _, done, info = env.step(Macro.IDLE)
        assert done is False, f"truncated at step {i + 1} with no horizon set"
        assert info["timeout"] is False

    assert env.step_count == MAX_EPISODE_STEPS + 1

    # NO HIDDEN CEILING ANYWHERE IN THE COMPARISON — the property the loop above
    # cannot bind. A mutant that reads `self._max_steps if ... is not None else
    # 10_000_000` survives any loop-based test, because no test can afford to
    # outrun a large sentinel. Jumping the counter costs O(1) and catches every
    # finite ceiling, however large.
    env._step_count = 10**9
    bridge.push(_state(tick=1234))
    _, _, done, info = env.step(Macro.IDLE)
    assert done is False, "a horizon of None truncated at a hidden ceiling"
    assert info["timeout"] is False
    assert env.max_episode_steps is None

    # A death still ends it. Disabling the horizon must not disable termination —
    # otherwise the exhibition never reports a winner at all.
    bridge.push(_state(tick=9999, opponent_died=True))
    _, _, done, info = env.step(Macro.ATTACK)
    assert done is True
    assert info["won"] is True
    assert info["timeout"] is False


def test_no_timeout_does_not_change_the_default_horizon():
    """Omitting ``max_episode_steps`` still truncates at the frozen constant."""
    bridge = ScriptedBridge([_reset_ack(ok=True), _state()])
    env = _env(bridge)

    assert env.max_episode_steps == MAX_EPISODE_STEPS


@pytest.mark.parametrize("bad", [0, -1, -400])
def test_max_episode_steps_still_rejects_non_positive_integers(bad):
    """``None`` is the disabled form; 0 and negatives remain errors."""
    bridge = ScriptedBridge([])
    with pytest.raises(ValueError, match="max_episode_steps must be > 0 or None"):
        _env(bridge, max_episode_steps=bad)


def test_a_death_does_not_auto_restart_the_match():
    """After a death the env stays finished until reset() is called (AC4)."""
    bridge = ScriptedBridge([_reset_ack(ok=True), _state()])
    env = _env(bridge, max_episode_steps=None)
    env.reset(seed=0)

    bridge.push(_state(tick=2, opponent_died=True))
    _, _, done, info = env.step(Macro.ATTACK)
    assert done is True and info["won"] is True

    # NOTHING here restarts. The match is over, the result is reported, and the
    # operator arms the next challenger with the separate reset command; an env
    # that resumed on its own would put the agent back in the pad against a
    # dead opponent while the operator is still talking to the audience.
    with pytest.raises(ValueError, match="finished/unstarted episode"):
        env.step(Macro.IDLE)
    assert bridge.sent[-1] == {"type": "step", "action": int(Macro.ATTACK)}

    # ...and the operator's reset is what starts the next one.
    bridge.push(_reset_ack(ok=True), _state(tick=3))
    env.reset(seed=1)
    assert env.step_count == 0
    bridge.push(_state(tick=4))
    _, _, done, _ = env.step(Macro.IDLE)
    assert done is False


# ---------------------------------------------------------------------------
# ExhibitionConfig (T3) — the settings T5/T6/T7 consume.
# ---------------------------------------------------------------------------


def test_exhibition_config_defaults_match_the_contract():
    """The four documented fields, with the documented defaults."""
    cfg = ExhibitionConfig()

    assert cfg.challenger_username is None  # first claimant in the pad
    assert cfg.no_timeout is True
    assert cfg.auto_reset is False
    assert cfg.reflex_blind_steps == 8  # ~1.6 s at the frozen 200 ms interval
    # The ONE form "no timeout" takes, handed straight to MCPvPEnv.
    assert cfg.env_max_episode_steps is None
    assert ExhibitionConfig(no_timeout=False).env_max_episode_steps == MAX_EPISODE_STEPS


def test_exhibition_config_builds_an_env_that_never_truncates():
    """The config's horizon reaches the env intact — the seam T5 uses."""
    bridge = ScriptedBridge([_reset_ack(ok=True), _state()])
    env = _env(bridge, max_episode_steps=ExhibitionConfig().env_max_episode_steps)

    assert env.max_episode_steps is None


def test_exhibition_config_refuses_an_auto_restart():
    """``auto_reset=True`` is refused loudly rather than silently ignored (AC4)."""
    # Nothing implements an auto-restart, so accepting the flag would be a
    # config that promises a behavior no code provides — and the failure would
    # only show up as a match that does not restart, in front of a classroom.
    with pytest.raises(ValueError, match="auto_reset must be exactly False"):
        ExhibitionConfig(auto_reset=True)


@pytest.mark.parametrize("bad", [0, 1, None, "", "False", "false"])
def test_exhibition_config_refuses_a_non_bool_auto_reset(bad):
    """The type strictness covers FALSY non-bools, and says so in the message."""
    # `0`, `None` and `""` are all rejected by `is not False`, and the old
    # message told the operator "auto_reset=True is not implemented" — which
    # describes none of them and sends them looking for a True they never passed.
    with pytest.raises(ValueError, match="auto_reset must be exactly False"):
        ExhibitionConfig(auto_reset=bad)


@pytest.mark.parametrize("bad", [0, 1, None, "", "false", "True", 1.0])
def test_exhibition_config_refuses_a_non_bool_no_timeout(bad):
    """``no_timeout`` is held to the same strictness as the rest of the class."""
    # The dangerous value is a truthy stand-in: `no_timeout="false"` would
    # DISABLE the horizon — the opposite of what it reads as — and a horizon
    # that is silently off is indistinguishable from a match still in progress.
    with pytest.raises(ValueError, match="no_timeout must be exactly True or False"):
        ExhibitionConfig(no_timeout=bad)


@pytest.mark.parametrize("good", [True, False])
def test_exhibition_config_accepts_both_real_booleans_for_no_timeout(good):
    assert ExhibitionConfig(no_timeout=good).no_timeout is good


@pytest.mark.parametrize(
    "bad", ["", "bad name", "seventeen_chars_x", "quote\"name", 42, "classmate 1"]
)
def test_exhibition_config_rejects_an_unmatchable_challenger_name(bad):
    """A pin that can never equal a real username is refused at construction."""
    # It would otherwise produce an exhibition in which nobody is ever the
    # opponent — indistinguishable, from the operator's side, from an empty pad.
    with pytest.raises(ValueError, match="challenger_username must be"):
        ExhibitionConfig(challenger_username=bad)


@pytest.mark.parametrize("good", ["classmate_1", "a", "sixteen_chars_ok"])
def test_exhibition_config_accepts_real_usernames(good):
    assert ExhibitionConfig(challenger_username=good).challenger_username == good


@pytest.mark.parametrize("bad", [-1, 2.5, True, "8"])
def test_exhibition_config_rejects_a_bad_reflex_window(bad):
    with pytest.raises(ValueError, match="reflex_blind_steps must be"):
        ExhibitionConfig(reflex_blind_steps=bad)


def test_step_after_done_raises():
    """Calling step() after the episode finished is a programming error."""
    bridge = ScriptedBridge([_reset_ack(ok=True), _state()])
    env, _ = _reset_env(bridge)
    bridge.push(_state(tick=2, i_died=True))
    env.step(Macro.IDLE)  # ends the episode

    with pytest.raises(ValueError, match="finished/unstarted episode"):
        env.step(Macro.IDLE)


def test_step_rejects_out_of_range_action():
    """An action outside [0, N_ACTIONS) is rejected before hitting the wire."""
    bridge = ScriptedBridge([_reset_ack(ok=True), _state()])
    env, _ = _reset_env(bridge)

    with pytest.raises(ValueError, match="action must be in"):
        env.step(N_ACTIONS)  # 8 is out of range (valid are 0..7)


# ---------------------------------------------------------------------------
# Perception gating applied end-to-end (no opponent-position leak).
# ---------------------------------------------------------------------------


def test_opponent_in_fov_sets_visible():
    """An opponent dead ahead (in FOV, LoS clear) -> visible==1 in the obs."""
    bridge = ScriptedBridge([_reset_ack(ok=True), _state(opp_pos=(0.0, 64.0, 2.0))])
    env, _ = _reset_env(bridge)
    bridge.push(_state(tick=2, opp_pos=(0.0, 64.0, 2.0)))

    obs, _, _, _ = env.step(Macro.IDLE)

    assert obs[Obs.VISIBLE] == pytest.approx(1.0)
    # Dead ahead within range/crosshair -> derived flags set.
    assert obs[Obs.IN_RANGE] == pytest.approx(1.0)
    assert obs[Obs.IN_CROSSHAIR] == pytest.approx(1.0)


def test_opponent_outside_fov_is_gated_out_no_leak():
    """An opponent BEHIND the agent -> visible==0 and derived flags 0 (no leak).

    The current opponent position must not surface through any opponent or
    derived slot when it is outside the FOV (fairness AC5).
    """
    # Opponent directly behind on -z, never previously seen -> ABSENT regime.
    behind = _state(opp_pos=(0.0, 64.0, -5.0))
    bridge = ScriptedBridge([_reset_ack(ok=True), behind])
    env, _ = _reset_env(bridge)
    bridge.push(_state(tick=2, opp_pos=(0.0, 64.0, -5.0)))

    obs, _, _, _ = env.step(Macro.IDLE)

    assert obs[Obs.VISIBLE] == pytest.approx(0.0)
    assert obs[Obs.IN_RANGE] == pytest.approx(0.0)
    assert obs[Obs.IN_CROSSHAIR] == pytest.approx(0.0)
    # The live behind-position never leaks into the opponent position block.
    opp_pos = obs[Obs.OPP_POS_LOCAL : Obs.OPP_POS_LOCAL + 3]
    np.testing.assert_allclose(opp_pos, [0.0, 0.0, 0.0], atol=1e-6)


def test_gating_uses_injected_filter_los():
    """A custom PerceptionFilter (blocking LoS) gates out an in-FOV opponent."""

    def block_all(eye, target):
        return False  # a wall always occludes

    pf = PerceptionFilter(los_clear=block_all)
    # Opponent dead ahead and in range, but LoS is blocked -> gated out.
    bridge = ScriptedBridge([_reset_ack(ok=True), _state(opp_pos=(0.0, 64.0, 2.0))])
    env = _env(bridge, perception_filter=pf)
    env.reset(seed=0)
    bridge.push(_state(tick=2, opp_pos=(0.0, 64.0, 2.0)))

    obs, _, _, _ = env.step(Macro.IDLE)
    assert obs[Obs.VISIBLE] == pytest.approx(0.0)


def test_aim_bonus_only_when_visible():
    """The aim reward component is nonzero only when visible AND in crosshair."""
    cfg = RewardConfig()

    # Visible + crosshair -> aim bonus present.
    bridge_v = ScriptedBridge([_reset_ack(ok=True), _state(opp_pos=(0.0, 64.0, 2.0))])
    env_v, _ = _reset_env(bridge_v)
    bridge_v.push(_state(tick=2, opp_pos=(0.0, 64.0, 2.0)))
    _, _, _, info_v = env_v.step(Macro.IDLE)
    assert info_v["r_aim"] == pytest.approx(cfg.c_aim)

    # Not visible (behind) -> aim bonus exactly 0 (anti-spin-farm).
    bridge_h = ScriptedBridge([_reset_ack(ok=True), _state(opp_pos=(0.0, 64.0, -5.0))])
    env_h, _ = _reset_env(bridge_h)
    bridge_h.push(_state(tick=2, opp_pos=(0.0, 64.0, -5.0)))
    _, _, _, info_h = env_h.step(Macro.IDLE)
    assert info_h["r_aim"] == 0.0


def test_filter_memory_reset_between_episodes():
    """reset() clears PerceptionFilter memory so a stale sighting never carries over.

    See the opponent in episode 1, then in episode 2 it is behind from step 1 —
    with memory cleared the opponent must read ABSENT (visible 0), not a held
    last-seen position.
    """
    bridge = ScriptedBridge([_reset_ack(ok=True), _state(opp_pos=(0.0, 64.0, 2.0))])
    env, _ = _reset_env(bridge)
    bridge.push(_state(tick=2, opp_pos=(0.0, 64.0, 2.0)))
    obs1, _, _, _ = env.step(Macro.IDLE)
    assert obs1[Obs.VISIBLE] == pytest.approx(1.0)  # sighted in episode 1

    # Episode 2: reset (clears memory), opponent behind from the first state.
    bridge.push(_reset_ack(ok=True), _state(opp_pos=(0.0, 64.0, -5.0)))
    obs0_ep2 = env.reset(seed=1)
    # The initial obs of episode 2 must not carry episode-1 memory.
    assert obs0_ep2[Obs.VISIBLE] == pytest.approx(0.0)
    opp_pos = obs0_ep2[Obs.OPP_POS_LOCAL : Obs.OPP_POS_LOCAL + 3]
    np.testing.assert_allclose(opp_pos, [0.0, 0.0, 0.0], atol=1e-6)


# ---------------------------------------------------------------------------
# Self velocity world->local rotation.
# ---------------------------------------------------------------------------


def test_self_velocity_rotated_into_local_frame():
    """World-frame self velocity is rotated into the agent's local frame in the obs.

    Facing +z (yaw 0), a world velocity of +x is the agent's RIGHT (+X_local) and
    a world velocity of +z is FORWARD (+Z_local). With MAX_SPEED == 1.0 the
    normalized obs equals the rotated value.
    """
    # Yaw 0, moving east (+x) at 0.5 blocks/tick.
    bridge = ScriptedBridge(
        [_reset_ack(ok=True), _state(self_yaw=0.0, self_vel=(0.5, 0.0, 0.0))]
    )
    env = _env(bridge)
    obs = env.reset(seed=0)

    vx, vy, vz = (
        obs[Obs.VEL_LOCAL],
        obs[Obs.VEL_LOCAL + 1],
        obs[Obs.VEL_LOCAL + 2],
    )
    # +x world at yaw 0 -> +x local (right), zero forward.
    assert vx == pytest.approx(0.5, abs=1e-6)
    assert vy == pytest.approx(0.0, abs=1e-6)
    assert vz == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Bridge disconnect / error handling.
# ---------------------------------------------------------------------------


def test_disconnect_during_step_raises_bridge_error_and_reconnects():
    """A dropped connection during step() raises BridgeError after one reconnect."""
    bridge = ScriptedBridge([_reset_ack(ok=True), _state()])
    env, _ = _reset_env(bridge)
    connects_before = bridge.connects

    # Next recv (the step's state reply) simulates a disconnect.
    bridge.push(ScriptedBridge.Disconnect())

    with pytest.raises(BridgeError):
        env.step(Macro.IDLE)

    # Exactly one reconnect was attempted (close + connect).
    assert bridge.connects == connects_before + 1
    assert bridge.closes >= 1


def test_disconnect_during_reset_raises_bridge_error():
    """A dropped connection while awaiting reset_ack surfaces as a BridgeError."""
    bridge = ScriptedBridge([ScriptedBridge.Disconnect()])
    env = _env(bridge)

    with pytest.raises(BridgeError):
        env.reset(seed=0)

    # A reconnect was attempted.
    assert bridge.connects >= 2  # initial connect + one reconnect


def test_reconnect_failure_aborts_loudly():
    """If the single reconnect also fails, the env aborts with a clear message."""

    class FlakyBridge(ScriptedBridge):
        def __init__(self):
            super().__init__([ScriptedBridge.Disconnect()])
            self._connected_once = False

        def connect(self):
            if self._connected_once:
                # The reconnect attempt fails outright.
                raise BridgeError("reconnect refused")
            self._connected_once = True
            self.connects += 1
            self.is_open = True

    env = _env(FlakyBridge())
    with pytest.raises(BridgeError, match="aborting the run"):
        env.reset(seed=0)


# ---------------------------------------------------------------------------
# Post-eval dead-socket recovery (Approach A): reset() tolerates a reconnect.
#
# Regression guard for the release-blocker where a periodic eval opens a SECOND
# connection to the single-connection bridge; the bridge adopts the eval socket
# and destroys the training socket. When training resumes, the next reset()'s
# wire I/O hits the dead socket. reset() is idempotent (no in-flight episode), so
# it must reconnect and re-send the WHOLE exchange from scratch — whether the dead
# socket surfaces on the first send OR the first recv — and only step() keeps its
# strict re-raise-on-lost-reply behavior.
# ---------------------------------------------------------------------------


class DeadSocketOnceBridge(ScriptedBridge):
    """Fake bridge whose FIRST wire op fails once, then a reconnect heals it.

    Models the post-eval dead training socket: the very next ``send`` (or ``recv``,
    per ``fail_on``) raises a transport :class:`BridgeError` exactly once. A
    :meth:`connect` call (the env's reconnect) clears the armed failure, so the
    re-sent exchange on the "fresh socket" goes through against the scripted queue.

    This is the faithful seam for the bug: the bridge dropped the old connection,
    and a reconnect + redo-from-scratch recovers because reset is idempotent.
    """

    def __init__(self, inbound=None, *, fail_on="recv"):
        super().__init__(inbound)
        assert fail_on in ("send", "recv")
        self._fail_on = fail_on
        self._armed = True  # the dead old socket; cleared by the reconnect

    def connect(self):
        # The reconnect hands us a fresh, healthy socket.
        self._armed = False
        super().connect()

    def send(self, obj):
        if self._armed and self._fail_on == "send":
            # Record nothing — the write hit the dead socket and was lost.
            raise BridgeError("DeadSocketOnceBridge: send on dead socket")
        super().send(obj)

    def recv(self):
        if self._armed and self._fail_on == "recv":
            # EOF / orderly peer shutdown mid-stream, exactly like recv() b''.
            raise BridgeError("DeadSocketOnceBridge: recv on dead socket (EOF)")
        return super().recv()


def test_reset_recovers_when_send_hits_dead_socket():
    """(1) A dead socket surfacing on reset()'s SEND -> reconnect + redo -> valid obs."""
    bridge = DeadSocketOnceBridge(
        [_reset_ack(ok=True), _state()], fail_on="send"
    )
    # auto_connect=False so construction does not disarm the (already-dead) socket;
    # this models the eval stealing the connection just before this reset, with the
    # first write therefore hitting the dead old socket.
    connects_before = bridge.connects
    env = _env(bridge, auto_connect=False)

    obs = env.reset(seed=11)

    # Recovered: a valid initial observation, queue fully drained on the redo.
    assert isinstance(obs, np.ndarray) and obs.shape == (OBS_DIM,)
    validate(obs)
    assert bridge.inbound == []  # ack + state consumed by the successful retry
    # Exactly one reconnect happened to heal the dead socket.
    assert bridge.connects == connects_before + 1
    assert bridge.closes >= 1
    # The reset was re-sent on the fresh socket (the lost first send recorded
    # nothing; the retry's send is the only reset on the wire).
    resets = [m for m in bridge.sent if m["type"] == "reset"]
    assert len(resets) == 1
    assert resets[0]["seed"] == 11


def test_reset_recovers_when_recv_hits_dead_socket():
    """(2) The currently-aborting case: dead socket on reset()'s RECV -> recovers.

    This is the exact path that aborts the live run today (recv returns EOF ->
    BridgeError -> re-raised by design). With Approach A, reset() reconnects and
    re-runs the whole exchange, returning a valid initial observation.
    """
    bridge = DeadSocketOnceBridge(
        [_reset_ack(ok=True), _state()], fail_on="recv"
    )
    # See test_reset_recovers_when_send_hits_dead_socket: keep the socket "dead"
    # through construction so reset()'s first recv hits it.
    connects_before = bridge.connects
    env = _env(bridge, auto_connect=False)

    obs = env.reset(seed=22)

    assert isinstance(obs, np.ndarray) and obs.shape == (OBS_DIM,)
    validate(obs)
    assert bridge.inbound == []
    assert bridge.connects == connects_before + 1
    assert bridge.closes >= 1
    # The reset was sent once on the dead socket (recorded), then re-sent on the
    # fresh socket after the reconnect -> two reset writes on this fake.
    resets = [m for m in bridge.sent if m["type"] == "reset"]
    assert len(resets) == 2


def test_reset_aborts_loudly_if_bridge_stays_down():
    """A bridge that NEVER heals exhausts the bounded retries and aborts loudly."""

    class AlwaysDeadBridge(ScriptedBridge):
        def __init__(self):
            super().__init__([_reset_ack(ok=True), _state()])

        def recv(self):
            # Every recv hits a dead socket; the reconnect never heals it.
            raise BridgeError("AlwaysDeadBridge: still down")

    bridge = AlwaysDeadBridge()
    env = _env(bridge)

    with pytest.raises(BridgeError, match="bridge reset failed"):
        env.reset(seed=0)

    # It tried the full bounded budget of transport attempts, not a single shot.
    assert bridge.connects >= 2  # initial connect + at least one reconnect


def test_step_still_aborts_on_dead_socket_recv():
    """(3) The SAME failure during step() STILL raises — mid-episode desync is fatal.

    step() must NOT inherit reset()'s reconnect-and-retry tolerance: a lost reply
    mid-episode genuinely desyncs the request/reply stream and is unrecoverable.
    """
    bridge = ScriptedBridge([_reset_ack(ok=True), _state()])
    env, _ = _reset_env(bridge)
    connects_before = bridge.connects

    # The step's state reply hits a dead socket (EOF), exactly like the reset case.
    bridge.push(ScriptedBridge.Disconnect())

    with pytest.raises(BridgeError, match="in-flight reply is lost"):
        env.step(Macro.IDLE)

    # step() reconnects exactly ONCE (so the next episode can proceed) but does NOT
    # silently retry the step — strict abort preserved.
    assert bridge.connects == connects_before + 1
    assert bridge.closes >= 1


def test_step_send_failure_keeps_single_reconnect_retry_semantics():
    """A dead socket on step()'s SEND keeps the existing one-reconnect send retry.

    Distinct from reset(): step()'s _send self-recovers a failed WRITE with one
    reconnect (the request hadn't been answered yet, so re-sending is safe), but
    the subsequent recv is still strict. Here the fresh socket serves the reply,
    so the step completes — proving Approach A left step()'s send path untouched.
    """

    class StepSendDeadOnceBridge(ScriptedBridge):
        def __init__(self, inbound):
            super().__init__(inbound)
            self._send_armed = False

        def arm_send_failure(self):
            self._send_armed = True

        def connect(self):
            self._send_armed = False
            super().connect()

        def send(self, obj):
            if self._send_armed:
                raise BridgeError("StepSendDeadOnceBridge: send on dead socket")
            super().send(obj)

    bridge = StepSendDeadOnceBridge([_reset_ack(ok=True), _state()])
    env, _ = _reset_env(bridge)
    connects_before = bridge.connects

    # The step's reply is ready on the (post-reconnect) fresh socket.
    bridge.push(_state(tick=2))
    bridge.arm_send_failure()

    obs, reward, done, info = env.step(Macro.IDLE)

    validate(obs)
    assert done is False
    # Exactly one reconnect to recover the failed send.
    assert bridge.connects == connects_before + 1


# ---------------------------------------------------------------------------
# close().
# ---------------------------------------------------------------------------


def test_close_sends_close_and_tears_down():
    """close() sends a close message and closes the transport (idempotently)."""
    bridge = ScriptedBridge([_reset_ack(ok=True), _state()])
    env, _ = _reset_env(bridge)

    env.close()
    assert any(m["type"] == "close" for m in bridge.sent)
    assert bridge.closes >= 1

    # Idempotent: a second close does not raise.
    env.close()


def test_context_manager_closes():
    """Using the env as a context manager closes the transport on exit."""
    bridge = ScriptedBridge([_reset_ack(ok=True), _state()])
    with _env(bridge) as env:
        env.reset(seed=0)
    assert bridge.closes >= 1
    assert any(m["type"] == "close" for m in bridge.sent)
