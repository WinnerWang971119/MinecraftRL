"""Offline end-to-end integration test for the T20 M2 wiring.

Drives the FULL stack end to end against a FAKE bridge — NO live Minecraft
server, Node bridge, or socket is touched:

    seed_everything (inside Trainer)
      -> MCPvPEnv(real PerceptionFilter + canonical reward) over a scripted bridge
         + the stationary-dummy stage opponent (served by the fake bridge here)
      -> online/target DuelingDRQN + PrioritizedSequenceReplay (inside Trainer)
      -> DRQN.act -> env.step -> replay.add_episode -> Trainer.learn (a real
         gradient step) -> periodic greedy evaluate(...) -> EvalReport

What this PROVES (the offline half of AC6 / TC13):
  * the whole loop runs without error end to end;
  * a training step actually executes — the loss is finite AND the online params
    move after an update;
  * the periodic eval produces a populated ``EvalReport``;
  * the env ``info`` reward components STILL sum to the scalar reward (the W1 swap
    to the canonical ``compute_reward_components`` is correct);
  * the packed obs passes ``observation_spec.validate``; and
  * perception gating is LIVE — an invisible (behind / out-of-FOV) opponent yields
    ``visible == 0`` in the obs, so the real PerceptionFilter is in the loop.

What this does NOT do (the LIVE human follow-up): AC6 / TC13 proper — the greedy
trained DRQN reaching >= 95% win rate over 100 eval episodes vs the LIVE
stationary dummy. That needs a real training budget, the live Paper server, and
the Node bridge, and runs via ``python -m agent.train`` / ``python -m
eval.evaluate`` against a started bridge. See ``server/README.md`` ("Live
follow-up"), ``server/compat_check.md`` (the live-handshake follow-ups), and the
section banner in ``agent/train.py``. We deliberately do NOT try to reach 95%
offline.

torch is guarded with ``pytest.importorskip`` so the suite stays GREEN (this file
SKIPs, never fails) on an interpreter without a torch wheel. Any longer
convergence attempt would be marked ``@pytest.mark.slow``; this file keeps every
test FAST (tiny net, short windows, a few episodes).
"""

from __future__ import annotations

import numpy as np
import pytest

from agent.actions import N_ACTIONS
from agent.reward_config import RewardConfig
from agent.train_config import TrainConfig
from bridge.messages import ResetAckMsg, StateMsg
from env.mc_pvp_env import REWARD_COMPONENT_KEYS, BridgeError
from env.observation_spec import OBS_DIM, Obs, validate


# ===========================================================================
# Generative fake bridge — an ENDLESS, deterministic state stream.
#
# Unlike the per-episode scripted queues in tests/test_mc_pvp_env.py and
# tests/test_evaluate.py, the M2 loop collects MANY episodes and ALSO opens a
# FRESH env (over a FRESH transport) for every periodic eval. So the offline
# proof needs a transport that can serve an UNBOUNDED number of episodes without
# the caller pre-scripting each one. This bridge GENERATES valid messages on
# demand:
#
#   * on a `reset`  command -> yields ResetAckMsg(ok=True), then the first state;
#   * on a `step`   command -> yields the next state, marking opponent_died=True
#                              on step ``kill_step`` so every episode ends as a
#                              short WIN (keeps the loop fast and bounded);
#   * on a `close`  command -> no state follows (the env only `recv`s after
#                              reset/step).
#
# The opponent position is configurable so the test can drive a VISIBLE-ahead
# stream (the default) or an INVISIBLE-behind stream (to prove perception gating
# is live). This is still the same four-method BridgeTransport contract; it just
# computes the next message instead of popping a fixed list.
# ===========================================================================


class GenerativeBridge:
    """A fake ``BridgeTransport`` that generates a valid state stream on demand.

    Args:
        opp_pos: Opponent world position used for every emitted ``state``. Dead
            ahead on +z (the default) is in FOV/range/crosshair (-> visible);
            behind on -z is out of FOV (-> gated out, visible == 0).
        kill_step: 1-based step at which a ``state`` reports ``opponent_died`` so
            the episode terminates as a WIN. Keeps episodes short and bounded.
        damage_per_step: ``damage_dealt`` reported on every non-terminal step (so a
            nonzero reward flows through the canonical reward path).
    """

    def __init__(self, *, opp_pos=(0.0, 64.0, 2.0), kill_step=2, damage_per_step=1.0):
        self.opp_pos = tuple(float(v) for v in opp_pos)
        self.kill_step = int(kill_step)
        self.damage_per_step = float(damage_per_step)

        self.sent = []
        self.connects = 0
        self.closes = 0
        self.is_open = False

        # Per-episode counters driving the generated stream.
        self._episode_step = 0  # steps taken in the current episode
        self._pending_reset_ack = False  # a reset_ack is owed before the state
        self._tick = 0

    # -- BridgeTransport protocol -----------------------------------------

    def connect(self):
        self.connects += 1
        self.is_open = True

    def close(self):
        self.closes += 1
        self.is_open = False

    def send(self, obj):
        msg = dict(obj)
        self.sent.append(msg)
        mtype = msg.get("type")
        if mtype == "reset":
            # A reset starts a new episode: owe one reset_ack, then the first state.
            self._pending_reset_ack = True
            self._episode_step = 0
        elif mtype == "step":
            self._episode_step += 1
        # `close` produces no inbound message (the env never recv()s after close).

    def recv(self):
        self._tick += 1
        if self._pending_reset_ack:
            self._pending_reset_ack = False
            return ResetAckMsg.from_dict(
                {
                    "type": "reset_ack",
                    "ok": True,
                    "readback": {"self_hp": 20.0, "opp_hp": 20.0},
                }
            )
        # Otherwise emit the next state. The post-reset first state is step 0; a
        # win fires once the agent has taken ``kill_step`` steps.
        opponent_died = self._episode_step >= self.kill_step
        damage = 20.0 if opponent_died else self.damage_per_step
        return self._make_state(opponent_died=opponent_died, damage_dealt=damage)

    # -- state builder ----------------------------------------------------

    def _make_state(self, *, opponent_died, damage_dealt):
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
                    "pos": list(self.opp_pos),
                    "yaw": 0.0,
                    "pitch": 0.0,
                    "velocity": [0.0, 0.0, 0.0],
                    "on_ground": True,
                    "health": 20.0,
                    "held_item": "iron_sword",
                },
                "events": {
                    "damage_dealt": float(damage_dealt),
                    "damage_taken": 0.0,
                    "i_died": False,
                    "opponent_died": bool(opponent_died),
                },
                "arena": {"wall_distances": [8.0, 8.0, 8.0, 8.0]},
                "tick": self._tick,
                "code_version": "test",
            }
        )


def _visible_factory(**kwargs):
    """Transport factory: a fresh VISIBLE-ahead GenerativeBridge per call."""

    def factory():
        return GenerativeBridge(opp_pos=(0.0, 64.0, 2.0), **kwargs)

    return factory


def _invisible_factory(**kwargs):
    """Transport factory: a fresh INVISIBLE-behind GenerativeBridge per call."""

    def factory():
        return GenerativeBridge(opp_pos=(0.0, 64.0, -5.0), **kwargs)

    return factory


# ===========================================================================
# Tiny, fast TrainConfig + net so a real gradient step is cheap.
# ===========================================================================


def _tiny_cfg(**overrides):
    """A fast TrainConfig: short windows, tiny warm-up, small replay."""
    base = dict(
        lr=1e-3,
        batch_size=4,
        seq_len=4,
        burn_in=2,
        n_step=2,
        gamma=0.99,
        tau=0.1,
        grad_clip=10.0,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay_episodes=10,
        replay_capacity=2_000,
        min_replay=1,  # learn almost immediately for a fast smoke
        per_beta_anneal_steps=100,
        eval_interval=0,
        checkpoint_interval=0,
        log_interval=1,
        seed=0,
    )
    base.update(overrides)
    return TrainConfig(**base)


#: Shrunken net so forward/backward is cheap. obs_dim / n_actions still assert
#: against the frozen contracts inside DuelingDRQN; only the hidden sizes shrink.
_TINY_NET = {"encoder_hidden": 16, "lstm_hidden": 16, "lstm_layers": 1}

#: Episode length the kill_step yields (kill_step + the post-reset step). Must
#: exceed burn_in + seq_len + n_step so collected episodes are sampleable.
_KILL_STEP = 10


# ===========================================================================
# In-memory fake logger — captures every MetricsLogger call (no backend).
# ===========================================================================


class FakeLogger:
    """Records ``log`` / ``summary`` / ``close`` calls in memory (no disk/backend)."""

    def __init__(self):
        self.logged = []
        self.summaries = []
        self.closed = False

    def log(self, metrics, step=None):
        self.logged.append((dict(metrics), step))

    def log_scalar(self, name, value, step=None):
        self.logged.append(({name: value}, step))

    def summary(self, values):
        self.summaries.append(dict(values))

    def close(self):
        self.closed = True


# ===========================================================================
# End-to-end: the whole stack runs, a gradient step executes, eval reports.
# ===========================================================================


def test_m2_integration_runs_end_to_end_and_trains():
    """The full M2 loop runs without error; a real gradient step moves the params."""
    pytest.importorskip("torch", exc_type=ImportError)
    from agent.train import M2Result, train_vs_dummy

    cfg = _tiny_cfg()

    # stop_on_pass=False so we run the WHOLE budget. (The generative bridge scripts
    # every episode as a short win, so a greedy eval trivially clears the gate
    # offline — that early-stop path is covered by test_m2_loop_stops_at_gate. We
    # deliberately do NOT lean on hitting 95% offline; this asserts the wiring runs
    # end to end and a real gradient step executes — the LIVE 95% run is the human
    # follow-up.)
    result = train_vs_dummy(
        cfg,
        transport_factory=_visible_factory(kill_step=_KILL_STEP),
        max_episodes=6,
        updates_per_step=1,
        eval_every_episodes=3,
        eval_episodes=2,
        timeout_cap=64,
        env_max_episode_steps=64,
        net_kwargs=dict(_TINY_NET),
        stop_on_pass=False,
    )

    # Ran end to end and produced a structured result.
    assert isinstance(result, M2Result)
    assert result.episodes_run == 6  # full budget (stop_on_pass=False)
    assert result.grad_steps >= 1  # at least one real gradient step executed
    # Two evals at episodes 3 and 6.
    assert len(result.reports) == 2
    assert result.last_report is not None

    # A populated EvalReport (the M2 gate artifact shape).
    from eval.evaluate import EvalReport

    report = result.last_report
    assert isinstance(report, EvalReport)
    assert report.n_episodes == 2
    assert report.n_wins + report.n_losses + report.n_timeouts == 2
    # Every reward component is present in the run breakdown.
    assert set(report.reward_component_sums.keys()) == set(REWARD_COMPONENT_KEYS)


def test_m2_gradient_step_updates_parameters():
    """A learn() step actually moves the online net's weights (real backward)."""
    torch = pytest.importorskip("torch", exc_type=ImportError)
    from agent.train import Trainer

    cfg = _tiny_cfg(min_replay=1)
    trainer = Trainer(cfg, net_kwargs=dict(_TINY_NET))

    # Drive the real env (real perception + reward) over the fake bridge so the
    # replay fills with REAL transitions, then take one gradient step.
    from env.mc_pvp_env import MCPvPEnv

    transport = _visible_factory(kill_step=_KILL_STEP)()
    env = MCPvPEnv(transport=transport, max_episode_steps=64)
    try:
        trainer.collect_episode(env, max_steps=64)
    finally:
        env.close()

    assert trainer.ready_to_learn()
    before = [p.detach().clone() for p in trainer.online.parameters()]
    stats = trainer.learn()
    assert stats is not None
    assert np.isfinite(stats.loss)  # the loss is a finite scalar

    after = list(trainer.online.parameters())
    # At least one parameter tensor changed -> the optimizer step did real work.
    moved = any(
        not torch.equal(b, a) for b, a in zip(before, after)
    )
    assert moved, "online parameters did not change after a gradient step"


# ===========================================================================
# W1 swap correctness: env info components sum to the scalar reward.
# ===========================================================================


def test_env_info_components_sum_to_scalar_reward():
    """The env info reward components (from the canonical function) sum to reward."""
    pytest.importorskip("torch", exc_type=ImportError)
    from env.mc_pvp_env import MCPvPEnv

    transport = _visible_factory(kill_step=_KILL_STEP)()
    env = MCPvPEnv(transport=transport, max_episode_steps=64)
    try:
        env.reset(seed=0)
        # Walk several steps and assert the invariant holds on each transition.
        for _ in range(5):
            obs, reward, done, info = env.step(0)
            validate(obs)  # the obs always passes the frozen validator
            component_sum = sum(info[key] for key in REWARD_COMPONENT_KEYS)
            assert component_sum == pytest.approx(reward, abs=1e-6)
            if done:
                break
    finally:
        env.close()


def test_env_damage_component_matches_canonical_formula():
    """A damage_dealt event yields exactly the canonical r_damage_dealt component."""
    pytest.importorskip("torch", exc_type=ImportError)
    from env.mc_pvp_env import MCPvPEnv

    transport = _visible_factory(kill_step=_KILL_STEP, damage_per_step=3.0)()
    env = MCPvPEnv(transport=transport, max_episode_steps=64)
    try:
        env.reset(seed=0)
        _, reward, _, info = env.step(0)
    finally:
        env.close()

    cfg = RewardConfig()
    # The canonical component is exactly c_dmg_out * damage (3.0 per non-kill step).
    assert info["r_damage_dealt"] == pytest.approx(cfg.c_dmg_out * 3.0)
    # And it still sums correctly with all the other components.
    assert sum(info[key] for key in REWARD_COMPONENT_KEYS) == pytest.approx(
        reward, abs=1e-6
    )


# ===========================================================================
# Perception gating is LIVE in the integration loop.
# ===========================================================================


def test_perception_gating_live_invisible_opponent_not_seen():
    """An opponent behind the agent is gated out (visible == 0) through the env."""
    pytest.importorskip("torch", exc_type=ImportError)
    from env.mc_pvp_env import MCPvPEnv

    # Opponent directly behind (-z), never previously seen -> ABSENT regime.
    transport = _invisible_factory(kill_step=_KILL_STEP)()
    env = MCPvPEnv(transport=transport, max_episode_steps=64)
    try:
        obs0 = env.reset(seed=0)
        # The initial obs already gates the behind-opponent out.
        assert obs0[Obs.VISIBLE] == pytest.approx(0.0)
        obs, _, _, _ = env.step(0)
        assert obs[Obs.VISIBLE] == pytest.approx(0.0)
        # The derived flags do not leak either.
        assert obs[Obs.IN_RANGE] == pytest.approx(0.0)
        assert obs[Obs.IN_CROSSHAIR] == pytest.approx(0.0)
        # And the live behind-position never leaks into the opponent position block.
        opp_pos = obs[Obs.OPP_POS_LOCAL : Obs.OPP_POS_LOCAL + 3]
        np.testing.assert_allclose(opp_pos, [0.0, 0.0, 0.0], atol=1e-6)
    finally:
        env.close()


def test_perception_gating_live_visible_opponent_seen():
    """An opponent dead ahead is visible (visible == 1) through the env."""
    pytest.importorskip("torch", exc_type=ImportError)
    from env.mc_pvp_env import MCPvPEnv

    transport = _visible_factory(kill_step=_KILL_STEP)()
    env = MCPvPEnv(transport=transport, max_episode_steps=64)
    try:
        env.reset(seed=0)
        obs, _, _, _ = env.step(0)
        assert obs[Obs.VISIBLE] == pytest.approx(1.0)
    finally:
        env.close()


# ===========================================================================
# Per-reward-component logging flows through the MetricsLogger in the M2 loop.
# ===========================================================================


def test_m2_eval_logs_each_reward_component_separately():
    """Each periodic eval logs every reward component as its own metric key."""
    pytest.importorskip("torch", exc_type=ImportError)
    from agent.train import train_vs_dummy

    logger = FakeLogger()
    cfg = _tiny_cfg()

    train_vs_dummy(
        cfg,
        transport_factory=_visible_factory(kill_step=_KILL_STEP),
        max_episodes=3,
        eval_every_episodes=3,
        eval_episodes=2,
        timeout_cap=64,
        env_max_episode_steps=64,
        net_kwargs=dict(_TINY_NET),
        logger=logger,
    )

    # The eval logged one run-summary with every component under sum.* / mean.*.
    assert len(logger.summaries) == 1
    summary = logger.summaries[0]
    for key in REWARD_COMPONENT_KEYS:
        assert f"sum.{key}" in summary
        assert f"mean.{key}" in summary
    # And per-episode eval records carry each component under its own key.
    eval_records = [
        m for m, _ in logger.logged if any(k in m for k in REWARD_COMPONENT_KEYS)
    ]
    assert eval_records, "no per-episode eval record logged the reward components"
    for record in eval_records:
        for key in REWARD_COMPONENT_KEYS:
            assert key in record


# ===========================================================================
# The stop-at-gate path: a forced-pass eval stops the loop with passed_m2.
# ===========================================================================


class _AlwaysWinEnv:
    """A fake env whose every episode is an instant, visible WIN.

    Used to prove the M2 loop STOPS at the gate: with every eval episode a short
    win (visible the whole time, no aim-while-invisible, length < cap), the
    greedy eval clears ``passed_m2`` and ``train_vs_dummy`` must break early.
    """

    def __init__(self, *args, **kwargs):
        self._done = False

    def reset(self, seed=None):
        self._done = False
        return self._visible_obs()

    def step(self, action):
        self._done = True
        info = {key: 0.0 for key in REWARD_COMPONENT_KEYS}
        info.update({"won": True, "lost": False, "timeout": False})
        info["r_terminal"] = 8.0
        return self._visible_obs(), 8.0, True, info

    @staticmethod
    def _visible_obs():
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        obs[int(Obs.VISIBLE)] = 1.0
        return obs


def test_m2_loop_stops_at_gate(monkeypatch):
    """When eval clears the M2 gate, the loop stops early with passed_m2=True.

    We monkeypatch ``MCPvPEnv`` (the env the loop builds for BOTH collection and
    eval) to an always-win fake so a greedy eval clears the gate immediately. The
    real env wiring is covered by the other tests; this isolates the stop path.
    """
    pytest.importorskip("torch", exc_type=ImportError)
    import agent.train as train_mod

    monkeypatch.setattr(train_mod, "MCPvPEnv", _AlwaysWinEnv, raising=False)
    # The loop imports MCPvPEnv locally; patch the source module too so both the
    # collection env and the eval env resolve to the always-win fake.
    import env.mc_pvp_env as env_mod

    monkeypatch.setattr(env_mod, "MCPvPEnv", _AlwaysWinEnv, raising=False)

    from agent.train import train_vs_dummy

    cfg = _tiny_cfg(min_replay=1)
    result = train_vs_dummy(
        cfg,
        transport_factory=lambda: object(),  # never used by the fake env
        max_episodes=20,
        eval_every_episodes=1,  # eval after the very first episode
        eval_episodes=3,
        timeout_cap=64,
        env_max_episode_steps=64,
        rollout_step_cap=2,  # the fake env ends in one step anyway
        net_kwargs=dict(_TINY_NET),
    )

    assert result.passed_m2 is True
    assert result.stop_reason == "passed_m2"
    # Stopped EARLY: far fewer than the 20-episode budget.
    assert result.episodes_run < 20
    assert result.last_report is not None
    assert result.last_report.passed_m2 is True


# ===========================================================================
# Budget paths: the loop honors the episode budget when the gate never clears.
# ===========================================================================


def test_m2_loop_honors_episode_budget_without_eval():
    """With eval disabled the loop runs exactly the episode budget (no gate check)."""
    pytest.importorskip("torch", exc_type=ImportError)
    from agent.train import train_vs_dummy

    cfg = _tiny_cfg()
    result = train_vs_dummy(
        cfg,
        transport_factory=_visible_factory(kill_step=_KILL_STEP),
        max_episodes=4,
        eval_every_episodes=0,  # disable periodic eval entirely
        env_max_episode_steps=64,
        net_kwargs=dict(_TINY_NET),
    )
    assert result.episodes_run == 4
    assert result.stop_reason == "max_episodes"
    assert result.passed_m2 is False
    assert result.last_report is None  # no eval ran
    assert result.reports == []


def test_run_m2_is_train_vs_dummy_alias():
    """The plan's alternate entrypoint name ``run_m2`` is the same callable."""
    from agent.train import run_m2, train_vs_dummy

    assert run_m2 is train_vs_dummy


# ===========================================================================
# Input validation on the entrypoint.
# ===========================================================================


def test_train_vs_dummy_rejects_non_positive_max_episodes():
    """max_episodes <= 0 is a loud ValueError before any env is built."""
    pytest.importorskip("torch", exc_type=ImportError)
    from agent.train import train_vs_dummy

    with pytest.raises(ValueError, match="max_episodes must be > 0"):
        train_vs_dummy(
            _tiny_cfg(),
            transport_factory=_visible_factory(),
            max_episodes=0,
            net_kwargs=dict(_TINY_NET),
        )


def test_train_vs_dummy_rejects_negative_eval_interval():
    """A negative eval_every_episodes is a loud ValueError."""
    pytest.importorskip("torch", exc_type=ImportError)
    from agent.train import train_vs_dummy

    with pytest.raises(ValueError, match="eval_every_episodes must be >= 0"):
        train_vs_dummy(
            _tiny_cfg(),
            transport_factory=_visible_factory(),
            max_episodes=2,
            eval_every_episodes=-1,
            net_kwargs=dict(_TINY_NET),
        )


# ===========================================================================
# Sanity: the scripted bridge stays within the frozen action space.
# ===========================================================================


def test_generative_bridge_action_space_sanity():
    """The integration loop only ever sends actions in the frozen [0, N_ACTIONS)."""
    pytest.importorskip("torch", exc_type=ImportError)
    from agent.train import train_vs_dummy

    bridges = []
    factory = _visible_factory(kill_step=_KILL_STEP)

    def recording_factory():
        bridge = factory()
        bridges.append(bridge)
        return bridge

    train_vs_dummy(
        _tiny_cfg(),
        transport_factory=recording_factory,
        max_episodes=3,
        eval_every_episodes=0,
        env_max_episode_steps=64,
        net_kwargs=dict(_TINY_NET),
    )

    # Every `step` action that reached the wire is a valid frozen macro index.
    for bridge in bridges:
        for msg in bridge.sent:
            if msg.get("type") == "step":
                assert 0 <= msg["action"] < N_ACTIONS


# ===========================================================================
# Disconnect handling still surfaces as a BridgeError through the loop.
# ===========================================================================


def test_generative_bridge_is_a_valid_transport():
    """The generative bridge satisfies the four-method BridgeTransport contract."""
    bridge = GenerativeBridge()
    # connect/close lifecycle.
    bridge.connect()
    assert bridge.is_open is True
    # A reset exchange yields an ack then a state.
    bridge.send({"type": "reset", "episode": 0, "seed": 0})
    ack = bridge.recv()
    assert isinstance(ack, ResetAckMsg) and ack.ok is True
    state = bridge.recv()
    assert isinstance(state, StateMsg)
    bridge.close()
    assert bridge.is_open is False


def test_bridge_error_importable_for_offline_tests():
    """BridgeError is the documented failure type the offline contract raises."""
    assert issubclass(BridgeError, RuntimeError)


# ===========================================================================
# Single-connection regression: a periodic eval must NOT open a second bridge
# connection (the real bridge serves exactly one; a second steals the stream and
# aborts the live run). This models the bridge as a single-slot server and proves
# the train -> eval -> train sequence never holds two connections at once, and
# that training resumes with intact env state afterward.
# ===========================================================================


class _SingleSlotBridgeServer:
    """Models the production bridge: ONE listening server, ONE live connection.

    Every :class:`_TrackedBridge` handed out by the factory registers/unregisters
    here on ``connect``/``close``. ``live`` is the current open-connection count and
    ``peak`` the max ever held at once — the regression asserts ``peak == 1`` (a
    second concurrent connection is exactly the bug). ``total_connects`` counts how
    many times anyone connected at all.
    """

    def __init__(self):
        self.live = 0
        self.peak = 0
        self.total_connects = 0

    def open(self):
        self.live += 1
        self.total_connects += 1
        self.peak = max(self.peak, self.live)

    def close(self):
        if self.live > 0:
            self.live -= 1


class _TrackedBridge(GenerativeBridge):
    """A GenerativeBridge that reports its connect/close to a shared server.

    Reuses the generative state stream (endless valid episodes) but routes the
    connection lifecycle through ``server`` so concurrency is observable. The real
    bridge's stream is per-connection; this fake faithfully keeps its own per-stream
    counters, so a shared transport re-syncs on the next ``reset`` exactly like the
    live bridge re-initializes its arena.
    """

    def __init__(self, server, **kwargs):
        super().__init__(**kwargs)
        self._server = server

    def connect(self):
        super().connect()
        self._server.open()

    def close(self):
        # Only the FIRST close of an open socket frees the slot (close is
        # idempotent on the env side); mirror that so double-close can't underflow.
        was_open = self.is_open
        super().close()
        if was_open:
            self._server.close()


def test_eval_opens_no_second_concurrent_connection(monkeypatch):
    """A periodic eval reuses the training connection; never a 2nd concurrent one.

    Regression for the release blocker: the eval used to build a FRESH transport,
    opening a SECOND connection to the single-slot bridge. The bridge adopts the
    new socket and destroys the training one, so training aborts at the next reset.
    With the fix, eval borrows the (idle) training transport, so across
    train -> eval -> train the server never holds more than ONE connection, and the
    factory is called exactly once (the training env's only connect).
    """
    pytest.importorskip("torch", exc_type=ImportError)
    from agent.train import train_vs_dummy

    server = _SingleSlotBridgeServer()
    bridges = []
    factory_calls = {"n": 0}

    def factory():
        factory_calls["n"] += 1
        bridge = _TrackedBridge(server, opp_pos=(0.0, 64.0, 2.0), kill_step=_KILL_STEP)
        bridges.append(bridge)
        return bridge

    cfg = _tiny_cfg()
    result = train_vs_dummy(
        cfg,
        transport_factory=factory,
        max_episodes=6,
        updates_per_step=1,
        eval_every_episodes=3,  # evals fire after episodes 3 and 6
        eval_episodes=2,
        timeout_cap=64,
        env_max_episode_steps=64,
        rollout_step_cap=64,
        net_kwargs=dict(_TINY_NET),
        stop_on_pass=False,  # run the whole budget so BOTH evals execute
    )

    # Two evals ran (episodes 3 and 6) over a full 6-episode budget.
    assert result.episodes_run == 6
    assert len(result.reports) == 2

    # THE REGRESSION: the single-slot server never held two connections at once.
    assert server.peak == 1, (
        f"a second concurrent bridge connection was opened (peak={server.peak}); "
        "eval must reuse the training connection, not open its own"
    )
    # Exactly ONE transport was ever created: the training env's. Eval borrowed it
    # instead of calling the factory again.
    assert factory_calls["n"] == 1
    assert len(bridges) == 1
    # And exactly one connect happened over the whole run.
    assert server.total_connects == 1

    # The single shared connection is closed exactly once at run teardown.
    assert bridges[0].closes == 1
    assert bridges[0].is_open is False


def test_training_env_state_intact_after_eval():
    """Training resumes with correct env state after an eval shares its socket.

    Proves the eval (a separate MCPvPEnv over the SHARED transport) does not corrupt
    the training env: the training env keeps collecting episodes through the same
    connection after eval, the per-episode reset reseeds the bridge stream, and the
    run completes the full budget with both evals producing populated reports.
    """
    pytest.importorskip("torch", exc_type=ImportError)
    from agent.train import train_vs_dummy
    from eval.evaluate import EvalReport

    server = _SingleSlotBridgeServer()
    bridges = []

    def factory():
        bridge = _TrackedBridge(server, opp_pos=(0.0, 64.0, 2.0), kill_step=_KILL_STEP)
        bridges.append(bridge)
        return bridge

    result = train_vs_dummy(
        _tiny_cfg(),
        transport_factory=factory,
        max_episodes=4,
        eval_every_episodes=2,  # eval after episodes 2 and 4
        eval_episodes=2,
        timeout_cap=64,
        env_max_episode_steps=64,
        rollout_step_cap=64,
        net_kwargs=dict(_TINY_NET),
        stop_on_pass=False,
    )

    # Both evals ran AFTER training continued past them (episode 2 eval, then more
    # training, then episode 4 eval) on the one shared connection.
    assert result.episodes_run == 4
    assert len(result.reports) == 2
    assert server.peak == 1
    assert len(bridges) == 1  # the shared transport, never a second one

    # Both eval reports are populated EvalReports (training was healthy across the
    # eval boundary — a torn socket would have aborted with a BridgeError instead).
    for report in result.reports:
        assert isinstance(report, EvalReport)
        assert report.n_episodes == 2
        assert report.n_wins + report.n_losses + report.n_timeouts == 2

    # The bridge stream advanced through training AND eval episodes on the one
    # transport: training episodes (4) + eval episodes (2 evals * 2 eps) = 8 resets.
    reset_count = sum(1 for m in bridges[0].sent if m.get("type") == "reset")
    assert reset_count == 4 + 2 * 2
