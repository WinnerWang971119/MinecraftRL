"""Tests for the damage-anchored reward and the anti-hacking battery (T17 / TC5, TC6).

These tests pin the finalized reward (spec §7 / training-spec §3):

  TC5 — reward matches the formula on hand-authored event dicts + obs vectors,
        and the aim bonus is added ONLY when visible AND in_crosshair (all four
        combinations of visible × in_crosshair are checked).

  TC6 — anti-hacking battery:
        * a spinning agent with the opponent UNSEEN accrues NO aim reward across
          a multi-step sequence (visible=false the whole time → aim == 0 exactly),
        * damage terms scale linearly with events,
        * the step penalty is always applied,
        * the terminal reward fires only at done (and respects loss > win),
        * the potential-based positional shaping is a no-op at the default
          coefficient, telescopes to 0 over a closed loop, and equals
          γ·Φ(s') − Φ(s) when enabled (provably policy-invariant).

  Plus: ``compute_reward == sum(compute_reward_components)`` for many random
        inputs (the single-source-of-truth invariant the env/eval rely on).
"""

import dataclasses

import numpy as np
import pytest

from agent.reward_config import RewardConfig
from bridge.messages import Events
from env.observation_spec import OBS_DIM, Obs, field_slice
from env.reward import (
    REWARD_COMPONENT_KEYS,
    TermInfo,
    compute_reward,
    compute_reward_components,
)


# ---------------------------------------------------------------------------
# Builders for hand-authored events / observation vectors.
# ---------------------------------------------------------------------------


def _events(damage_dealt=0.0, damage_taken=0.0, i_died=False, opponent_died=False):
    """An :class:`Events` from a hand-authored event dict (privileged-but-fair anchors)."""
    return Events.from_dict(
        {
            "damage_dealt": float(damage_dealt),
            "damage_taken": float(damage_taken),
            "i_died": bool(i_died),
            "opponent_died": bool(opponent_died),
        }
    )


def _obs(visible=False, in_crosshair=False, opp_pos=(0.0, 0.0, 0.0)):
    """A length-OBS_DIM observation vector with only the reward-relevant fields set.

    The reward reads exactly three fields: ``visible`` (the aim/shaping gate),
    ``in_crosshair`` (the aim trigger), and ``opp_pos_local`` (the shaping Φ).
    Everything else is irrelevant to the reward, so a zero vector with these
    fields populated is a faithful hand-authored input.
    """
    vec = np.zeros(OBS_DIM, dtype=np.float32)
    vec[Obs.VISIBLE] = 1.0 if visible else 0.0
    vec[Obs.IN_CROSSHAIR] = 1.0 if in_crosshair else 0.0
    vec[field_slice("opp_pos_local")] = np.asarray(opp_pos, dtype=np.float32)
    return vec


def _nonterminal():
    return TermInfo()


# ===========================================================================
# TC5 — reward matches the formula; aim bonus only when visible AND in_crosshair.
# ===========================================================================


def test_default_config_reproduces_combat_reward_shape():
    """The frozen defaults are the tuned combat-reward shape (see RewardConfig).

    Win dominates (+50), damage dealt is weighted 2x damage taken so combat is
    net-positive (1.0 vs 0.5 per HP = +2 vs -1 per heart), and a timeout is the
    worst outcome (-30) to punish kiting.
    """
    cfg = RewardConfig()
    assert cfg.c_dmg_out == 1.0
    assert cfg.c_dmg_in == 0.5
    assert cfg.c_dmg_out == 2.0 * cfg.c_dmg_in  # dealt weighted 2x taken
    assert cfg.c_step == 0.005  # finalized step penalty (T17)
    assert cfg.c_aim == 0.01
    assert cfg.R_terminal_win == 50.0
    assert cfg.R_terminal_loss == 8.0
    assert cfg.R_terminal_timeout == -30.0
    assert cfg.R_terminal_win > cfg.R_terminal_loss  # winning beats fear of losing
    assert cfg.R_terminal_timeout < -cfg.R_terminal_loss  # timeout worse than a loss
    assert cfg.gamma == 0.99
    assert cfg.c_approach == 0.0  # shaping is a no-op by default


def test_reward_matches_formula_on_hand_authored_inputs():
    """r = c_dmg_out·dealt − c_dmg_in·taken − c_step + c_aim·1[vis&xhair] + R_term + F."""
    cfg = RewardConfig()
    events = _events(damage_dealt=4.0, damage_taken=1.5)
    # Opponent visible AND under crosshair → the aim term is active. No terminal,
    # default shaping (c_approach == 0) → F == 0.
    obs = _obs(visible=True, in_crosshair=True, opp_pos=(0.3, 0.0, 0.4))
    prev = _obs(visible=True, in_crosshair=True, opp_pos=(0.5, 0.0, 0.5))

    expected = (
        cfg.c_dmg_out * 4.0
        - cfg.c_dmg_in * 1.5
        - cfg.c_step
        + cfg.c_aim  # visible AND in_crosshair
        + 0.0  # F: c_approach == 0
        + 0.0  # not done
    )
    r = compute_reward(events, obs, prev, _nonterminal(), cfg)
    assert r == pytest.approx(expected)


@pytest.mark.parametrize(
    "visible,in_crosshair,aim_active",
    [
        (False, False, False),
        (False, True, False),  # NOT visible → 0 even though under crosshair
        (True, False, False),  # visible but not under crosshair → 0
        (True, True, True),  # the ONLY case that earns the aim bonus
    ],
)
def test_aim_bonus_only_when_visible_and_in_crosshair(visible, in_crosshair, aim_active):
    """All four visible × in_crosshair combinations: aim added iff BOTH are true."""
    cfg = RewardConfig()
    events = _events()  # no damage, isolate the aim term
    obs = _obs(visible=visible, in_crosshair=in_crosshair)
    prev = _obs(visible=visible, in_crosshair=in_crosshair)

    comps = compute_reward_components(events, obs, prev, _nonterminal(), cfg)
    assert comps["r_aim"] == pytest.approx(cfg.c_aim if aim_active else 0.0)

    # And the scalar reward differs from the no-aim baseline by exactly the bonus.
    r = compute_reward(events, obs, prev, _nonterminal(), cfg)
    baseline = -cfg.c_step  # only the always-on step penalty otherwise
    assert r == pytest.approx(baseline + (cfg.c_aim if aim_active else 0.0))


def test_aim_term_is_exactly_c_aim_no_partial_credit():
    """The aim bonus is exactly c_aim or exactly 0 — never a fractional in-between."""
    cfg = RewardConfig()
    obs = _obs(visible=True, in_crosshair=True)
    comps = compute_reward_components(_events(), obs, obs, _nonterminal(), cfg)
    assert comps["r_aim"] == cfg.c_aim


# ===========================================================================
# TC6 — anti-hacking battery.
# ===========================================================================


def test_spinning_unseen_opponent_accrues_no_aim_reward():
    """AC6/TC6: visible=false for a whole multi-step spin → aim contribution is EXACTLY 0.

    Model a spinning agent: ``in_crosshair`` flickers true as the crosshair
    sweeps past the (remembered) opponent direction, but ``visible`` is false
    the entire time. The aim term must be exactly 0 on every step, so the agent
    cannot farm aim reward by spinning while the opponent is unseen.
    """
    cfg = RewardConfig()
    rng = np.random.default_rng(0)

    total_aim = 0.0
    for _ in range(64):
        # visible always FALSE; in_crosshair flickers; position is stale memory.
        obs = _obs(
            visible=False,
            in_crosshair=bool(rng.integers(0, 2)),
            opp_pos=tuple(rng.uniform(-0.5, 0.5, size=3)),
        )
        prev = _obs(
            visible=False,
            in_crosshair=bool(rng.integers(0, 2)),
            opp_pos=tuple(rng.uniform(-0.5, 0.5, size=3)),
        )
        comps = compute_reward_components(_events(), obs, prev, _nonterminal(), cfg)
        assert comps["r_aim"] == 0.0
        total_aim += comps["r_aim"]

    assert total_aim == 0.0


def test_aim_gate_is_zero_even_with_shaping_enabled_and_in_crosshair():
    """The visibility hard-gate holds regardless of in_crosshair or positional value."""
    cfg = dataclasses.replace(RewardConfig(), c_approach=0.5)
    # Not visible, but under crosshair and with a non-zero remembered position.
    obs = _obs(visible=False, in_crosshair=True, opp_pos=(0.2, 0.0, 0.2))
    comps = compute_reward_components(_events(), obs, obs, _nonterminal(), cfg)
    assert comps["r_aim"] == 0.0
    # Shaping is also gated off when not visible → no phantom potential.
    assert comps["r_shaping"] == 0.0


@pytest.mark.parametrize("dealt,taken", [(0.0, 0.0), (1.0, 0.0), (0.0, 3.0), (7.5, 2.25)])
def test_damage_terms_scale_with_events(dealt, taken):
    """Damage components scale linearly with the event magnitudes (correct signs)."""
    cfg = RewardConfig()
    obs = _obs()
    comps = compute_reward_components(_events(dealt, taken), obs, obs, _nonterminal(), cfg)
    assert comps["r_damage_dealt"] == pytest.approx(cfg.c_dmg_out * dealt)
    assert comps["r_damage_taken"] == pytest.approx(-cfg.c_dmg_in * taken)


def test_damage_dealt_is_rewarded_damage_taken_is_penalized():
    """Dealing raises reward, taking lowers it — and dealing is weighted more (aggression)."""
    cfg = RewardConfig()
    obs = _obs()
    r_deal = compute_reward(_events(damage_dealt=5.0), obs, obs, _nonterminal(), cfg)
    r_take = compute_reward(_events(damage_taken=5.0), obs, obs, _nonterminal(), cfg)
    baseline = compute_reward(_events(), obs, obs, _nonterminal(), cfg)
    assert r_deal > baseline
    assert r_take < baseline
    # Each delta tracks its own coefficient.
    assert (r_deal - baseline) == pytest.approx(cfg.c_dmg_out * 5.0)
    assert (r_take - baseline) == pytest.approx(-cfg.c_dmg_in * 5.0)
    # Asymmetric shape: dealing the same HP is rewarded more than taking it is
    # penalized, so an even trade is net-positive → engaging beats avoiding.
    assert (r_deal - baseline) > -(r_take - baseline)


def test_step_penalty_always_applied():
    """The −c_step penalty is present on every step regardless of any other term."""
    cfg = RewardConfig()
    for events in (_events(), _events(damage_dealt=10.0), _events(damage_taken=2.0)):
        for obs in (_obs(), _obs(visible=True, in_crosshair=True)):
            comps = compute_reward_components(events, obs, obs, _nonterminal(), cfg)
            assert comps["r_step"] == pytest.approx(-cfg.c_step)


def test_step_penalty_magnitude_does_not_incentivize_suicide_rush():
    """A full-episode step bleed stays small vs. a single hit and vs. R_terminal_win.

    This is the documented anti-hacking property of the finalized c_step (0.005):
    dying to stop the step penalty early can never out-earn engaging, because the
    whole-episode penalty is dwarfed by one landed hit and by the win bonus.
    """
    cfg = RewardConfig()
    horizon = 200  # generous episode length in decision steps
    episode_step_cost = cfg.c_step * horizon
    # A single landed melee hit is worth several HP of damage; the whole-episode
    # step bleed must stay well under one such hit, so ending the episode early
    # (by dying) to stop the bleed is never worth giving up even one good trade.
    typical_hit_hp = 6.0  # ~ a sword hit; far above the per-episode step cost
    assert episode_step_cost < cfg.c_dmg_out * typical_hit_hp
    assert episode_step_cost < cfg.R_terminal_win  # << the win bonus


def test_terminal_reward_only_at_done():
    """Terminal reward is 0 on non-terminal steps and ±R_terminal exactly at done."""
    cfg = RewardConfig()
    obs = _obs()

    # Non-terminal → no terminal contribution.
    comps = compute_reward_components(_events(), obs, obs, TermInfo(done=False), cfg)
    assert comps["r_terminal"] == 0.0

    win = compute_reward_components(
        _events(), obs, obs, TermInfo(done=True, won=True), cfg
    )
    assert win["r_terminal"] == pytest.approx(cfg.R_terminal_win)

    loss = compute_reward_components(
        _events(), obs, obs, TermInfo(done=True, lost=True), cfg
    )
    assert loss["r_terminal"] == pytest.approx(-cfg.R_terminal_loss)

    draw = compute_reward_components(
        _events(), obs, obs, TermInfo(done=True, timeout=True), cfg
    )
    assert draw["r_terminal"] == pytest.approx(cfg.R_terminal_timeout)


def test_termination_flags_must_be_consistent():
    """Contradictory TermInfo fails loud (T5-review note a): won AND lost is rejected."""
    with pytest.raises(ValueError):
        TermInfo(done=True, won=True, lost=True)
    # An outcome flag without done is also rejected (an upstream producer bug).
    with pytest.raises(ValueError):
        TermInfo(done=False, won=True)


def test_terminal_precedence_matches_env_double_death_is_loss():
    """Loss takes precedence over win — the env resolves a double death to a loss.

    The env never constructs ``won and lost`` (``__post_init__`` would reject
    it), but the reward's precedence is asserted here to lock the documented
    loss > win ordering.
    """
    cfg = RewardConfig()
    obs = _obs()
    # A bare loss yields the negative terminal; mirror the env's resolution
    # (double death → lost=True, won=False).
    loss = compute_reward_components(
        _events(i_died=True, opponent_died=True),
        obs,
        obs,
        TermInfo(done=True, lost=True, won=False),
        cfg,
    )
    assert loss["r_terminal"] == pytest.approx(-cfg.R_terminal_loss)


# ---------------------------------------------------------------------------
# Potential-based positional shaping: no-op at default, telescopes, policy-invariant.
# ---------------------------------------------------------------------------


def test_shaping_is_noop_at_default_coefficient():
    """With c_approach == 0 the shaping term is exactly 0 for any pair of states."""
    cfg = RewardConfig()
    assert cfg.c_approach == 0.0
    rng = np.random.default_rng(1)
    for _ in range(32):
        obs = _obs(visible=True, in_crosshair=True, opp_pos=tuple(rng.uniform(-1, 1, 3)))
        prev = _obs(visible=True, in_crosshair=True, opp_pos=tuple(rng.uniform(-1, 1, 3)))
        comps = compute_reward_components(_events(), obs, prev, _nonterminal(), cfg)
        assert comps["r_shaping"] == 0.0


def test_shaping_equals_gamma_phi_next_minus_phi_when_enabled():
    """With c_approach > 0, r_shaping == γ·Φ(s') − Φ(s) where Φ = −c_approach·‖pos‖."""
    cfg = dataclasses.replace(RewardConfig(), c_approach=0.7, gamma=0.95)

    prev_pos = (0.5, 0.0, 0.5)  # farther
    cur_pos = (0.2, 0.0, 0.1)  # closer
    obs = _obs(visible=True, in_crosshair=False, opp_pos=cur_pos)
    prev = _obs(visible=True, in_crosshair=False, opp_pos=prev_pos)

    phi_prev = -cfg.c_approach * float(np.linalg.norm(prev_pos))
    phi_cur = -cfg.c_approach * float(np.linalg.norm(cur_pos))
    expected_F = cfg.gamma * phi_cur - phi_prev

    comps = compute_reward_components(_events(), obs, prev, _nonterminal(), cfg)
    assert comps["r_shaping"] == pytest.approx(expected_F)


def test_shaping_uses_gamma_from_config_only():
    """The shaping discount is read from cfg.gamma (one source of truth), not hardcoded."""
    pos_prev = (0.6, 0.0, 0.0)
    pos_cur = (0.1, 0.0, 0.0)
    obs = _obs(visible=True, opp_pos=pos_cur)
    prev = _obs(visible=True, opp_pos=pos_prev)

    for gamma in (0.90, 0.99, 1.0):
        cfg = dataclasses.replace(RewardConfig(), c_approach=1.0, gamma=gamma)
        phi_prev = -1.0 * float(np.linalg.norm(pos_prev))
        phi_cur = -1.0 * float(np.linalg.norm(pos_cur))
        expected = gamma * phi_cur - phi_prev
        comps = compute_reward_components(_events(), obs, prev, _nonterminal(), cfg)
        assert comps["r_shaping"] == pytest.approx(expected)


def test_shaping_telescopes_to_zero_over_a_closed_loop_at_gamma_one():
    """Potential-based shaping over a closed loop sums to 0 at γ=1 (policy-invariant).

    For a trajectory s0 → s1 → … → s0 that returns to its start, the shaping
    contributions Σ (γΦ(s_{t+1}) − Φ(s_t)) telescope to Φ(s_end) − Φ(s_start) at
    γ=1; closing the loop (s_end == s_start) makes that exactly 0. This is the
    mechanical reason the shaping cannot change the return ranking of policies.
    """
    cfg = dataclasses.replace(RewardConfig(), c_approach=0.4, gamma=1.0)

    positions = [
        (0.5, 0.0, 0.5),
        (0.3, 0.0, 0.2),
        (0.1, 0.0, 0.1),
        (0.4, 0.0, 0.3),
        (0.5, 0.0, 0.5),  # back to the start → closed loop
    ]
    states = [_obs(visible=True, opp_pos=p) for p in positions]

    total_shaping = 0.0
    for prev, cur in zip(states[:-1], states[1:]):
        comps = compute_reward_components(_events(), cur, prev, _nonterminal(), cfg)
        total_shaping += comps["r_shaping"]

    assert total_shaping == pytest.approx(0.0, abs=1e-9)


def test_shaping_guarded_against_phantom_potential_when_unseen():
    """An unseen (or zeroed) opponent injects no phantom potential into the shaping."""
    cfg = dataclasses.replace(RewardConfig(), c_approach=1.0)
    # prev: opponent was visible and far; cur: opponent now UNSEEN (Φ(cur) == 0).
    prev = _obs(visible=True, opp_pos=(0.9, 0.0, 0.0))
    cur = _obs(visible=False, opp_pos=(0.0, 0.0, 0.0))

    phi_prev = -cfg.c_approach * 0.9
    expected = cfg.gamma * 0.0 - phi_prev  # Φ(cur) is gated to 0
    comps = compute_reward_components(_events(), cur, prev, _nonterminal(), cfg)
    assert comps["r_shaping"] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Single source of truth: compute_reward == sum(components) for random inputs.
# ---------------------------------------------------------------------------


def test_components_keys_are_frozen_set():
    """The component dict has exactly the frozen, env-matching keys."""
    cfg = RewardConfig()
    obs = _obs()
    comps = compute_reward_components(_events(), obs, obs, _nonterminal(), cfg)
    assert tuple(comps.keys()) == REWARD_COMPONENT_KEYS


def test_compute_reward_equals_sum_of_components_random():
    """compute_reward(...) == sum(compute_reward_components(...)) over many random inputs."""
    rng = np.random.default_rng(2025)

    for _ in range(500):
        cfg = dataclasses.replace(
            RewardConfig(),
            c_dmg_out=float(rng.uniform(0.5, 2.0)),
            c_dmg_in=float(rng.uniform(0.5, 2.0)),
            c_step=float(rng.uniform(0.0, 0.02)),
            c_aim=float(rng.uniform(0.0, 0.05)),
            R_terminal_win=float(rng.uniform(5.0, 10.0)),
            R_terminal_loss=float(rng.uniform(5.0, 10.0)),
            R_terminal_timeout=0.0,
            gamma=float(rng.uniform(0.9, 1.0)),
            c_approach=float(rng.choice([0.0, rng.uniform(0.1, 1.0)])),
        )
        events = _events(
            damage_dealt=float(rng.uniform(0.0, 10.0)),
            damage_taken=float(rng.uniform(0.0, 10.0)),
        )
        obs = _obs(
            visible=bool(rng.integers(0, 2)),
            in_crosshair=bool(rng.integers(0, 2)),
            opp_pos=tuple(rng.uniform(-1.0, 1.0, size=3)),
        )
        prev = _obs(
            visible=bool(rng.integers(0, 2)),
            in_crosshair=bool(rng.integers(0, 2)),
            opp_pos=tuple(rng.uniform(-1.0, 1.0, size=3)),
        )

        # Pick a valid (non-contradictory) terminal at random.
        roll = rng.integers(0, 4)
        if roll == 0:
            terminal = TermInfo()  # non-terminal
        elif roll == 1:
            terminal = TermInfo(done=True, won=True)
        elif roll == 2:
            terminal = TermInfo(done=True, lost=True)
        else:
            terminal = TermInfo(done=True, timeout=True)

        scalar = compute_reward(events, obs, prev, terminal, cfg)
        comps = compute_reward_components(events, obs, prev, terminal, cfg)
        assert scalar == pytest.approx(sum(comps.values()))


# ---------------------------------------------------------------------------
# Input-validation contract (T5-review note b).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_len", [OBS_DIM - 1, OBS_DIM + 1, 0])
def test_wrong_shaped_obs_fails_loud(bad_len):
    """A wrong-length obs vector raises rather than silently reading a stale index."""
    cfg = RewardConfig()
    good = _obs()
    bad = np.zeros(bad_len, dtype=np.float32)
    with pytest.raises(ValueError):
        compute_reward(_events(), bad, good, _nonterminal(), cfg)
    with pytest.raises(ValueError):
        compute_reward(_events(), good, bad, _nonterminal(), cfg)


def test_nonfinite_position_does_not_poison_shaping():
    """A non-finite opp_pos is treated as 'no signal' so shaping never returns NaN."""
    cfg = dataclasses.replace(RewardConfig(), c_approach=1.0)
    obs = _obs(visible=True, opp_pos=(np.nan, 0.0, 0.0))
    prev = _obs(visible=True, opp_pos=(0.3, 0.0, 0.0))
    comps = compute_reward_components(_events(), obs, prev, _nonterminal(), cfg)
    # Φ(obs) is guarded to 0 (non-finite) → shaping = γ·0 − Φ(prev), finite.
    assert np.isfinite(comps["r_shaping"])
    assert np.isfinite(compute_reward(_events(), obs, prev, _nonterminal(), cfg))
