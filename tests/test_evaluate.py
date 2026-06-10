"""Tests for the M2 eval harness (T19) — offline, fake-bridge only.

No live Minecraft server, Node bridge, or socket is touched. ``evaluate`` is
driven against the REAL :class:`~env.mc_pvp_env.MCPvPEnv` over a FAKE scripted
bridge (the same four-method ``BridgeTransport`` contract documented at the
bottom of ``env/mc_pvp_env.py`` and reference-implemented by
``tests/test_mc_pvp_env.py``), with a tiny torch-free scripted policy. The
:class:`~eval.logging.MetricsLogger` calls are captured by an in-memory
:class:`FakeLogger` so the per-reward-component logging is asserted directly.

What is proved here (the OFFLINE half of AC6 / TC13):
  * win rate is computed correctly from the scripted win/loss/timeout outcomes;
  * every reward component is accumulated AND logged SEPARATELY (per episode and
    in the run summary);
  * ``aim_while_invisible`` is exactly 0 when the script keeps the opponent unseen
    (the spin-farming guard), AND the guard WOULD catch a leak if a step granted
    aim while invisible (proved with a small fake env that injects the leak);
  * the mean-episode-length / run-away guard works;
  * ``passed_m2`` reflects the win-rate + guard thresholds.

What is NOT proved here (the LIVE human follow-up): AC6 / TC13 proper — the
greedy trained DRQN vs the LIVE stationary dummy over 100 episodes hitting
>= 95% win rate. That needs a trained checkpoint, the live Paper server, and the
Node bridge, and runs as part of T20 via ``python -m eval.evaluate``. See the
module docstring of ``eval/evaluate.py``.
"""

import numpy as np
import pytest

from agent.actions import Macro, N_ACTIONS
from agent.reward_config import RewardConfig
from bridge.messages import ResetAckMsg, StateMsg
from env.mc_pvp_env import REWARD_COMPONENT_KEYS, BridgeError, MCPvPEnv
from env.observation_spec import OBS_DIM, Obs
from eval.evaluate import (
    M2_WIN_RATE_THRESHOLD,
    EpisodeOutcome,
    EvalReport,
    evaluate,
)


# ===========================================================================
# Fake bridge transport (mirrors tests/test_mc_pvp_env.ScriptedBridge).
# ===========================================================================


class ScriptedBridge:
    """A fake ``BridgeTransport`` driven by a scripted inbound queue.

    ``recv()`` pops the next scripted dataclass; ``send()`` records the wire dict.
    A queued :class:`Disconnect` sentinel makes ``recv()`` raise ``BridgeError``.
    This is the eval-test copy of the contract reference in
    ``tests/test_mc_pvp_env.py``, kept self-contained.
    """

    class Disconnect:
        """Sentinel: when ``recv()`` reaches this, it raises ``BridgeError``."""

    def __init__(self, inbound=None):
        self.inbound = list(inbound) if inbound is not None else []
        self.sent = []
        self.connects = 0
        self.closes = 0
        self.is_open = False

    def push(self, *messages):
        self.inbound.extend(messages)

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


def _reset_ack(ok=True, readback=None):
    return ResetAckMsg.from_dict(
        {
            "type": "reset_ack",
            "ok": ok,
            "readback": readback if readback is not None else {"self_hp": 20.0},
        }
    )


def _state(
    *,
    opp_pos=(0.0, 64.0, 2.0),  # dead ahead, in FOV/range/crosshair by default
    damage_dealt=0.0,
    damage_taken=0.0,
    i_died=False,
    opponent_died=False,
    tick=1,
):
    """A canonical valid ``state`` dataclass with stationary-dummy defaults."""
    return StateMsg.from_dict(
        {
            "type": "state",
            "self": {
                "pos": [0.0, 64.0, 0.0],
                "yaw": 0.0,
                "pitch": 0.0,
                "velocity": [0.0, 0.0, 0.0],
                "on_ground": True,
                "health": 20.0,
                "held_item": "iron_sword",
                "attack_cooldown": 1.0,
            },
            "opponent": {
                "pos": list(opp_pos),
                "yaw": 0.0,
                "pitch": 0.0,
                "velocity": [0.0, 0.0, 0.0],
                "health": 20.0,
            },
            "events": {
                "damage_dealt": damage_dealt,
                "damage_taken": damage_taken,
                "i_died": i_died,
                "opponent_died": opponent_died,
            },
            "arena": {"wall_distances": [8.0, 8.0, 8.0, 8.0]},
            "tick": tick,
            "code_version": "test",
        }
    )


# ===========================================================================
# In-memory fake logger — captures every MetricsLogger call.
# ===========================================================================


class FakeLogger:
    """Records ``log`` / ``log_scalar`` / ``summary`` / ``close`` calls in memory.

    Mirrors the read-only public API of :class:`~eval.logging.MetricsLogger`
    (``log(metrics, step=None)``, ``log_scalar(name, value, step=None)``,
    ``summary(values)``, ``close()``) so the eval harness logs through it
    unchanged, and the test can assert WHICH metrics were logged WHEN.
    """

    def __init__(self):
        self.logged = []  # list of (metrics_dict, step)
        self.scalars = []  # list of (name, value, step)
        self.summaries = []  # list of values dicts
        self.closed = False

    def log(self, metrics, step=None):
        self.logged.append((dict(metrics), step))

    def log_scalar(self, name, value, step=None):
        self.scalars.append((name, value, step))

    def summary(self, values):
        self.summaries.append(dict(values))

    def close(self):
        self.closed = True


# ===========================================================================
# Scripted, torch-free greedy policy.
# ===========================================================================


class ScriptedPolicy:
    """A torch-free :class:`~eval.evaluate.GreedyPolicy` that returns fixed actions.

    Returns ``action`` every step (default ``Macro.ATTACK``) and counts
    :meth:`reset` calls so the test can assert the evaluator resets per-episode
    state at each episode boundary (the recurrent-memory contract).
    """

    def __init__(self, action=int(Macro.ATTACK)):
        self._action = int(action)
        self.reset_calls = 0
        self.act_calls = 0

    def reset(self):
        self.reset_calls += 1

    def act(self, obs):
        self.act_calls += 1
        return self._action


# ===========================================================================
# Scripted-episode bridge builders.
# ===========================================================================


def _win_episode_states():
    """Inbound queue for one WIN episode: ack, reset-state, then a killing step."""
    return [
        _reset_ack(ok=True),
        _state(opp_pos=(0.0, 64.0, 2.0)),  # initial obs: opponent visible ahead
        _state(tick=2, opponent_died=True, damage_dealt=20.0),  # step -> win
    ]


def _loss_episode_states():
    """Inbound queue for one LOSS episode."""
    return [
        _reset_ack(ok=True),
        _state(opp_pos=(0.0, 64.0, 2.0)),
        _state(tick=2, i_died=True, damage_taken=20.0),  # step -> loss
    ]


def _timeout_episode_states(n_steps):
    """Inbound queue for one TIMEOUT episode of exactly ``n_steps`` steps."""
    queue = [_reset_ack(ok=True), _state(opp_pos=(0.0, 64.0, 2.0))]
    for i in range(n_steps):
        queue.append(_state(tick=2 + i, opp_pos=(0.0, 64.0, 2.0)))
    return queue


def _build_bridge(outcomes, *, timeout_steps=2):
    """Build one ScriptedBridge whose queue scripts ``outcomes`` in order.

    ``outcomes`` is a sequence of ``"win"`` / ``"loss"`` / ``"timeout"``. A single
    bridge is reused across all episodes (the env is reused too), so each episode's
    ack/state messages are concatenated into one queue in episode order.
    """
    queue = []
    for outcome in outcomes:
        if outcome == "win":
            queue.extend(_win_episode_states())
        elif outcome == "loss":
            queue.extend(_loss_episode_states())
        elif outcome == "timeout":
            queue.extend(_timeout_episode_states(timeout_steps))
        else:  # pragma: no cover - guarded by the test author
            raise ValueError(f"unknown scripted outcome {outcome!r}")
    return ScriptedBridge(queue)


def _make_env(bridge, *, max_episode_steps):
    """Construct a real MCPvPEnv over the fake bridge with a tiny horizon."""
    return MCPvPEnv(transport=bridge, max_episode_steps=max_episode_steps)


# ===========================================================================
# Win-rate / outcome counting.
# ===========================================================================


def test_win_rate_computed_from_scripted_outcomes():
    """win_rate and outcome counts match the scripted win/loss/timeout mix."""
    outcomes = ["win", "win", "win", "loss", "timeout"]
    bridge = _build_bridge(outcomes, timeout_steps=2)
    env = _make_env(bridge, max_episode_steps=2)
    policy = ScriptedPolicy()

    report = evaluate(env, policy, n_episodes=len(outcomes), timeout_cap=2, log=None)

    assert isinstance(report, EvalReport)
    assert report.n_episodes == 5
    assert report.n_wins == 3
    assert report.n_losses == 1
    assert report.n_timeouts == 1
    assert report.win_rate == pytest.approx(3 / 5)
    # Counts partition the run exactly.
    assert report.n_wins + report.n_losses + report.n_timeouts == report.n_episodes


def test_all_wins_gives_unit_win_rate():
    """A clean sweep of wins yields win_rate == 1.0."""
    outcomes = ["win"] * 4
    bridge = _build_bridge(outcomes)
    env = _make_env(bridge, max_episode_steps=8)
    report = evaluate(env, ScriptedPolicy(), n_episodes=4, timeout_cap=8)
    assert report.win_rate == pytest.approx(1.0)
    assert report.n_wins == 4


def test_policy_reset_once_per_episode():
    """The evaluator resets the policy's per-episode state at each episode start."""
    outcomes = ["win", "loss", "win"]
    bridge = _build_bridge(outcomes)
    env = _make_env(bridge, max_episode_steps=8)
    policy = ScriptedPolicy()
    evaluate(env, policy, n_episodes=3, timeout_cap=8)
    assert policy.reset_calls == 3


# ===========================================================================
# Per-reward-component accumulation AND separate logging.
# ===========================================================================


def test_reward_components_accumulated_in_report():
    """Each reward component is summed per episode and over the run."""
    bridge = _build_bridge(["win"])
    env = _make_env(bridge, max_episode_steps=8)
    report = evaluate(env, ScriptedPolicy(), n_episodes=1, timeout_cap=8)

    # Every component key is present in both the sums and the means.
    for key in REWARD_COMPONENT_KEYS:
        assert key in report.reward_component_sums
        assert key in report.reward_component_means

    cfg = RewardConfig()
    # The win episode dealt 20 damage and ended on a win, so those two components
    # are exactly the reward formula's contribution (single-step episode).
    assert report.reward_component_sums["r_damage_dealt"] == pytest.approx(
        cfg.c_dmg_out * 20.0
    )
    assert report.reward_component_sums["r_terminal"] == pytest.approx(
        cfg.R_terminal_win
    )
    # With one episode, sum == mean.
    for key in REWARD_COMPONENT_KEYS:
        assert report.reward_component_means[key] == pytest.approx(
            report.reward_component_sums[key]
        )


def test_each_component_logged_separately_per_episode():
    """Every reward component is logged as its OWN key in each per-episode record."""
    outcomes = ["win", "loss"]
    bridge = _build_bridge(outcomes)
    env = _make_env(bridge, max_episode_steps=8)
    logger = FakeLogger()

    evaluate(env, ScriptedPolicy(), n_episodes=2, logger=logger, timeout_cap=8)

    # One per-episode log() call per episode, at the episode step index.
    assert len(logger.logged) == 2
    for ep_index, (metrics, step) in enumerate(logger.logged):
        assert step == ep_index
        # Each component appears under its OWN distinct key (separate logging — the
        # whole point of the breakdown is that no component is folded into another).
        for key in REWARD_COMPONENT_KEYS:
            assert key in metrics, f"component {key!r} not logged separately"
        # Plus the episode-level series fields.
        assert "episode_length" in metrics
        assert "episode_reward" in metrics
        assert "win" in metrics
        assert "aim_while_invisible" in metrics

    # The win-flag series matches the outcomes.
    assert logger.logged[0][0]["win"] == pytest.approx(1.0)  # episode 0 won
    assert logger.logged[1][0]["win"] == pytest.approx(0.0)  # episode 1 lost


def test_run_summary_logs_component_breakdown():
    """The run summary logs each component sum AND mean under namespaced keys."""
    bridge = _build_bridge(["win", "win"])
    env = _make_env(bridge, max_episode_steps=8)
    logger = FakeLogger()

    report = evaluate(env, ScriptedPolicy(), n_episodes=2, logger=logger, timeout_cap=8)

    assert len(logger.summaries) == 1
    summary = logger.summaries[0]
    # Headline gate numbers present.
    assert summary["win_rate"] == pytest.approx(report.win_rate)
    assert summary["n_episodes"] == 2
    assert summary["passed_m2"] == report.passed_m2
    # Every component is in the summary under BOTH a sum.* and a mean.* key.
    for key in REWARD_COMPONENT_KEYS:
        assert f"sum.{key}" in summary
        assert f"mean.{key}" in summary
        assert summary[f"sum.{key}"] == pytest.approx(
            report.reward_component_sums[key]
        )
        assert summary[f"mean.{key}"] == pytest.approx(
            report.reward_component_means[key]
        )


def test_no_logger_still_computes_breakdown():
    """With no logger the component breakdown is still computed into the report."""
    bridge = _build_bridge(["win"])
    env = _make_env(bridge, max_episode_steps=8)
    report = evaluate(env, ScriptedPolicy(), n_episodes=1, logger=None, timeout_cap=8)
    assert set(report.reward_component_sums.keys()) == set(REWARD_COMPONENT_KEYS)
    assert report.reward_component_sums["r_damage_dealt"] > 0.0


# ===========================================================================
# Aim-while-invisible guard (spin-farming).
# ===========================================================================


def test_aim_while_invisible_is_zero_when_opponent_unseen():
    """A run where the opponent is never visible accrues exactly 0 aim-while-invisible.

    The opponent is scripted BEHIND the agent (outside the FOV) for the whole
    episode, so ``visible`` is false at every step. The real env's aim bonus is
    visibility-gated, so r_aim is 0 — and the guard's accumulator stays exactly 0.
    """
    # A timeout episode with the opponent always behind -> never visible.
    queue = [_reset_ack(ok=True), _state(opp_pos=(0.0, 64.0, -5.0))]
    for i in range(3):
        queue.append(_state(tick=2 + i, opp_pos=(0.0, 64.0, -5.0)))
    bridge = ScriptedBridge(queue)
    env = _make_env(bridge, max_episode_steps=3)

    report = evaluate(env, ScriptedPolicy(action=int(Macro.IDLE)), n_episodes=1, timeout_cap=3)

    assert report.aim_while_invisible == 0.0
    # And r_aim itself never accrued (visibility-gated in the env).
    assert report.reward_component_sums["r_aim"] == pytest.approx(0.0)


def test_aim_bonus_present_when_visible_does_not_count_as_invisible():
    """When the opponent IS visible, the legitimate aim bonus is NOT counted as a leak."""
    # Opponent dead ahead and in crosshair every step -> visible, aim bonus granted.
    queue = [_reset_ack(ok=True), _state(opp_pos=(0.0, 64.0, 2.0))]
    for i in range(3):
        queue.append(_state(tick=2 + i, opp_pos=(0.0, 64.0, 2.0)))
    bridge = ScriptedBridge(queue)
    env = _make_env(bridge, max_episode_steps=3)

    report = evaluate(env, ScriptedPolicy(action=int(Macro.IDLE)), n_episodes=1, timeout_cap=3)

    cfg = RewardConfig()
    # The aim bonus DID accrue (visible + crosshair every step).
    assert report.reward_component_sums["r_aim"] == pytest.approx(cfg.c_aim * 3)
    # But none of it is charged to the invisible guard.
    assert report.aim_while_invisible == 0.0


class _AimLeakEnv:
    """A minimal fake env that LEAKS an aim bonus on a step where visible is false.

    Proves the guard would CATCH a regression: a healthy env never grants r_aim
    while invisible, so to exercise the guard we need an env that does. ``reset``
    returns an invisible obs; ``step`` returns an invisible obs together with a
    nonzero ``r_aim`` in info, exactly the spin-farm signature the guard must catch.
    """

    def __init__(self, leak_aim=0.01, n_steps=2):
        self._leak_aim = float(leak_aim)
        self._n_steps = int(n_steps)
        self._step = 0

    def _invisible_obs(self):
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        obs[int(Obs.VISIBLE)] = 0.0  # explicitly NOT visible
        return obs

    def reset(self, seed=None):
        self._step = 0
        return self._invisible_obs()

    def step(self, action):
        self._step += 1
        done = self._step >= self._n_steps
        info = {key: 0.0 for key in REWARD_COMPONENT_KEYS}
        # The leak: aim bonus granted while the opponent is NOT visible.
        info["r_aim"] = self._leak_aim
        info["won"] = False
        info["lost"] = False
        info["timeout"] = done
        return self._invisible_obs(), self._leak_aim, done, info


def test_guard_catches_aim_while_invisible_leak():
    """If a step grants r_aim while visible is false, the guard accumulates it."""
    env = _AimLeakEnv(leak_aim=0.01, n_steps=2)
    report = evaluate(env, ScriptedPolicy(), n_episodes=1, timeout_cap=10)

    # Two steps each leaked 0.01 of aim while invisible -> caught.
    assert report.aim_while_invisible == pytest.approx(0.02)
    # And that makes the M2 gate FAIL even though the policy did not lose.
    assert report.passed_m2 is False


# ===========================================================================
# Mean-episode-length / run-away guard.
# ===========================================================================


def test_mean_and_median_episode_length_computed():
    """mean/median episode length are computed from the per-episode lengths."""
    # Two single-step wins and one 3-step timeout -> lengths [1, 1, 3].
    bridge = _build_bridge(["win", "win", "timeout"], timeout_steps=3)
    env = _make_env(bridge, max_episode_steps=3)
    report = evaluate(env, ScriptedPolicy(action=int(Macro.IDLE)), n_episodes=3, timeout_cap=3)

    assert report.mean_episode_length == pytest.approx((1 + 1 + 3) / 3)
    assert report.median_episode_length == pytest.approx(1.0)


def test_run_away_guard_fails_when_mean_length_hits_cap():
    """If episodes run to the timeout cap, mean length >= cap fails the gate."""
    # Every episode times out at exactly the cap -> mean length == cap -> NOT < cap.
    cap = 3
    bridge = _build_bridge(["timeout", "timeout"], timeout_steps=cap)
    env = _make_env(bridge, max_episode_steps=cap)
    report = evaluate(env, ScriptedPolicy(action=int(Macro.IDLE)), n_episodes=2, timeout_cap=cap)

    assert report.mean_episode_length == pytest.approx(float(cap))
    # mean_len (==cap) is NOT strictly below the cap -> run-away guard trips.
    assert report.passed_m2 is False


def test_mean_length_below_cap_satisfies_run_away_guard():
    """Short winning episodes keep mean length well below the cap (guard passes)."""
    bridge = _build_bridge(["win"] * 20)
    env = _make_env(bridge, max_episode_steps=400)
    report = evaluate(env, ScriptedPolicy(), n_episodes=20, timeout_cap=400)
    assert report.mean_episode_length < report.timeout_cap
    # All wins + short + no aim leak -> the whole M2 gate passes.
    assert report.passed_m2 is True


# ===========================================================================
# passed_m2 threshold logic.
# ===========================================================================


def test_passed_m2_true_at_threshold():
    """win_rate exactly at 95% (with the guards clean) passes the gate."""
    # 19 wins + 1 loss == 0.95 win rate, all short, no aim leak.
    outcomes = ["win"] * 19 + ["loss"]
    bridge = _build_bridge(outcomes)
    env = _make_env(bridge, max_episode_steps=8)
    report = evaluate(env, ScriptedPolicy(), n_episodes=20, timeout_cap=8)

    assert report.win_rate == pytest.approx(M2_WIN_RATE_THRESHOLD)
    assert report.aim_while_invisible == 0.0
    assert report.mean_episode_length < report.timeout_cap
    assert report.passed_m2 is True


def test_passed_m2_false_below_threshold():
    """A win rate just below 95% fails the gate even with clean guards."""
    # 18 wins + 2 losses == 0.90 win rate.
    outcomes = ["win"] * 18 + ["loss"] * 2
    bridge = _build_bridge(outcomes)
    env = _make_env(bridge, max_episode_steps=8)
    report = evaluate(env, ScriptedPolicy(), n_episodes=20, timeout_cap=8)

    assert report.win_rate == pytest.approx(0.90)
    assert report.win_rate < M2_WIN_RATE_THRESHOLD
    assert report.passed_m2 is False


# ===========================================================================
# Determinism + input validation.
# ===========================================================================


def test_evaluate_is_deterministic():
    """Two identical eval runs produce identical reports (no hidden RNG)."""
    reports = []
    for _ in range(2):
        bridge = _build_bridge(["win", "loss", "win"])
        env = _make_env(bridge, max_episode_steps=8)
        reports.append(
            evaluate(env, ScriptedPolicy(), n_episodes=3, timeout_cap=8).to_dict()
        )
    assert reports[0] == reports[1]


def test_evaluate_rejects_non_positive_episodes():
    """n_episodes < 1 is a loud ValueError."""
    bridge = _build_bridge(["win"])
    env = _make_env(bridge, max_episode_steps=8)
    with pytest.raises(ValueError, match="n_episodes must be >= 1"):
        evaluate(env, ScriptedPolicy(), n_episodes=0)


def test_evaluate_rejects_non_positive_timeout_cap():
    """timeout_cap < 1 is a loud ValueError."""
    bridge = _build_bridge(["win"])
    env = _make_env(bridge, max_episode_steps=8)
    with pytest.raises(ValueError, match="timeout_cap must be >= 1"):
        evaluate(env, ScriptedPolicy(), n_episodes=1, timeout_cap=0)


# ===========================================================================
# Per-episode outcome records.
# ===========================================================================


def test_episode_outcomes_recorded_in_order():
    """The report carries an EpisodeOutcome per episode, in order, with results."""
    outcomes = ["win", "loss", "timeout"]
    bridge = _build_bridge(outcomes, timeout_steps=2)
    env = _make_env(bridge, max_episode_steps=2)
    report = evaluate(env, ScriptedPolicy(action=int(Macro.IDLE)), n_episodes=3, timeout_cap=2)

    assert len(report.episodes) == 3
    assert [e.result for e in report.episodes] == outcomes
    for i, ep in enumerate(report.episodes):
        assert isinstance(ep, EpisodeOutcome)
        assert ep.index == i
        assert ep.length >= 1
        # Each per-episode record carries the full component breakdown.
        assert set(ep.components.keys()) == set(REWARD_COMPONENT_KEYS)


def test_action_space_is_respected_by_scripted_policy():
    """Sanity: the scripted action stays within the frozen action space."""
    # A guard that the fixtures never feed an out-of-range action to env.step.
    policy = ScriptedPolicy(action=int(Macro.ATTACK))
    assert 0 <= policy.act(None) < N_ACTIONS


# ===========================================================================
# Real DRQNGreedyPolicy integration (torch-guarded: SKIPs when torch absent).
#
# Proves the PRODUCTION policy path — DRQNGreedyPolicy wrapping a real
# DuelingDRQN, driven greedy (eps=0) over the real env / fake bridge — works end
# to end, is deterministic, and resets the LSTM hidden state per episode. The
# offline win-rate/component/guard logic above uses the torch-free ScriptedPolicy
# so the suite stays green without torch; this single test exercises the adapter.
# ===========================================================================


def test_drqn_greedy_policy_drives_eval_end_to_end():
    """A real DuelingDRQN wrapped greedily evaluates against the env deterministically."""
    torch = pytest.importorskip("torch", exc_type=ImportError)
    from agent.dqn import DuelingDRQN
    from eval.evaluate import DRQNGreedyPolicy

    torch.manual_seed(0)
    net = DuelingDRQN()

    # The same scripted run, evaluated twice with the greedy DRQN, must match
    # exactly (greedy + eps=0 is the no-RNG path -> fully deterministic).
    def run_once():
        bridge = _build_bridge(["win", "loss", "win"])
        env = _make_env(bridge, max_episode_steps=8)
        policy = DRQNGreedyPolicy(net)
        return evaluate(env, policy, n_episodes=3, timeout_cap=8).to_dict()

    first = run_once()
    second = run_once()
    assert first == second
    # The scripted outcomes still come through regardless of the net's actions
    # (the bridge scripts the terminal events), so the win rate is the scripted mix.
    assert first["n_wins"] == 2
    assert first["n_losses"] == 1
    assert first["win_rate"] == pytest.approx(2 / 3)


def test_drqn_greedy_policy_resets_hidden_between_episodes():
    """reset() clears the wrapped net's hidden state so memory never leaks forward."""
    torch = pytest.importorskip("torch", exc_type=ImportError)
    from agent.dqn import DuelingDRQN
    from eval.evaluate import DRQNGreedyPolicy

    torch.manual_seed(0)
    policy = DRQNGreedyPolicy(DuelingDRQN())

    # Drive a couple of steps so a hidden state exists, then reset.
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    policy.act(obs)
    assert policy._hidden is not None  # an LSTM state was carried
    policy.reset()
    assert policy._hidden is None  # cleared -> next episode starts from zeros
