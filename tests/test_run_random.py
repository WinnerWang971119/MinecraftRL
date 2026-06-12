"""Tests for the M1 tracer bullet runner (T10) — offline, fake-bridge only.

No live Minecraft server, Node bridge, or socket is touched. Every test injects a
FAKE bridge transport (a reusable :class:`ScriptedTracerBridge`, the same
four-method contract documented at the bottom of ``env/mc_pvp_env.py`` and
reference-implemented by ``tests/test_mc_pvp_env.py``) whose ``recv()`` returns
canned ``reset_ack`` / ``state`` dataclasses. A per-episode factory hands the
runner a fresh scripted bridge so it can drive >= 100 short episodes fast.

What is proved here (the OFFLINE half of TC11):
  * the runner completes >= 100 episodes with NO crash and NO hang — every
    episode terminates on a scripted death or on the env timeout, never loops;
  * a win-rate is computed and matches the scripted win/loss/timeout mix;
  * the toy replay buffer accumulates transitions and the no-op grad step runs
    once the buffer is warm;
  * the RSS sampler's cadence bookkeeping runs every 10 episodes without error.

What is NOT proved here (the LIVE human follow-up): TC11 / AC3 proper — a random
policy completing >= 100 episodes vs the idle dummy through the REAL Paper server
+ REAL bridge with zero crashes and combined Node + Python + JVM RSS growth
< ~200 MB. That needs a live server and is run via ``python -m eval.run_random``.
See the module docstring of ``eval/run_random.py``.
"""

import numpy as np
import pytest

from agent.actions import N_ACTIONS
from agent.random_policy import RandomPolicy
from bridge.messages import ResetAckMsg, StateMsg
from env.mc_pvp_env import BridgeError
from env.observation_spec import OBS_DIM
from eval.run_random import (
    RSS_SAMPLE_EVERY,
    EpisodeRecord,
    RssSampler,
    RunResult,
    ToyReplayBuffer,
    noop_grad_step,
    run_random,
)


# ===========================================================================
# Reusable fake bridge + scripted-episode helpers.
# ===========================================================================


def _reset_ack(ok=True, readback=None):
    """A canonical valid ``reset_ack`` dataclass."""
    return ResetAckMsg.from_dict(
        {
            "type": "reset_ack",
            "ok": ok,
            "readback": readback if readback is not None else {"self_hp": 20.0},
        }
    )


def _state(
    *,
    opp_pos=(0.0, 64.0, 2.0),  # dead ahead, in FOV/range by default
    damage_dealt=0.0,
    damage_taken=0.0,
    i_died=False,
    opponent_died=False,
    tick=1,
):
    """A canonical valid ``state`` dataclass with idle-dummy combat defaults.

    The opponent is stationary and immune (the idle dummy), so every scripted
    state keeps it at a fixed position; only the terminal state flips a death
    event.
    """
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


class ScriptedTracerBridge:
    """A fake ``BridgeTransport`` that replays a queued list of inbound messages.

    Implements exactly the four-method contract the env depends on (connect/send/
    recv/close). ``recv()`` pops the next scripted dataclass; ``send()`` records
    the wire dict. This is the tracer-test copy of ``test_mc_pvp_env.ScriptedBridge``
    (the contract reference), kept self-contained so the tracer test owns its
    fixtures.
    """

    def __init__(self, inbound):
        self.inbound = list(inbound)
        self.sent = []
        self.connects = 0
        self.closes = 0
        self.is_open = False

    def connect(self):
        self.connects += 1
        self.is_open = True

    def send(self, obj):
        self.sent.append(dict(obj))

    def recv(self):
        if not self.inbound:
            raise BridgeError("ScriptedTracerBridge: recv() with an empty queue")
        return self.inbound.pop(0)

    def close(self):
        self.closes += 1
        self.is_open = False


def _episode_script(outcome, *, steps_until_terminal=2, max_episode_steps=5):
    """Build the inbound message queue for ONE scripted episode.

    Layout consumed by the env:
      reset(): reset_ack(ok=True), then the post-reset initial state
      step() x N: one state per step; the terminal one carries the death event
                  (or, for a timeout, no death event ever fires and the env's
                  ``max_episode_steps`` truncates it).

    Args:
        outcome: ``"win"`` (opponent_died), ``"loss"`` (i_died), or ``"timeout"``.
        steps_until_terminal: For win/loss, how many steps before the death event.
        max_episode_steps: The env horizon (a timeout episode supplies exactly
            this many non-terminal step states).

    Returns:
        An ordered list of ``ResetAckMsg`` / ``StateMsg`` for one episode.
    """
    queue = [_reset_ack(ok=True), _state(tick=1)]

    if outcome == "timeout":
        # Never flip a death event; the env truncates at max_episode_steps. Supply
        # exactly that many step states (the env requests one per step()).
        for i in range(max_episode_steps):
            queue.append(_state(tick=2 + i))
        return queue

    # win / loss: non-terminal states, then the terminal one.
    for i in range(steps_until_terminal - 1):
        queue.append(_state(tick=2 + i))
    terminal_tick = 2 + max(0, steps_until_terminal - 1)
    if outcome == "win":
        queue.append(
            _state(tick=terminal_tick, opponent_died=True, damage_dealt=20.0)
        )
    elif outcome == "loss":
        queue.append(_state(tick=terminal_tick, i_died=True, damage_taken=20.0))
    else:  # pragma: no cover - guarded by the caller
        raise ValueError(f"unknown outcome {outcome!r}")
    return queue


def _make_factory(outcomes, *, max_episode_steps=5, steps_until_terminal=2):
    """Return a ``transport_factory`` that yields one scripted bridge per episode.

    ``outcomes`` is the ordered list of per-episode outcomes; the factory hands
    out one fully-scripted :class:`ScriptedTracerBridge` per call (per episode),
    plus a ``.bridges`` list so a test can inspect what was sent/received.
    """
    bridges = []
    it = iter(outcomes)

    def factory():
        outcome = next(it)
        queue = _episode_script(
            outcome,
            steps_until_terminal=steps_until_terminal,
            max_episode_steps=max_episode_steps,
        )
        bridge = ScriptedTracerBridge(queue)
        bridges.append(bridge)
        return bridge

    factory.bridges = bridges  # type: ignore[attr-defined]
    return factory


# ===========================================================================
# RandomPolicy.
# ===========================================================================


def test_random_policy_in_range_and_deterministic():
    """act() returns valid macro indices and is reproducible under the same seed."""
    p1 = RandomPolicy(seed=42)
    p2 = RandomPolicy(seed=42)
    seq1 = [p1.act(None) for _ in range(200)]
    seq2 = [p2.act(np.zeros(OBS_DIM, dtype=np.float32)) for _ in range(200)]

    assert seq1 == seq2  # same seed -> identical stream
    assert all(0 <= a < N_ACTIONS for a in seq1)
    assert all(isinstance(a, int) for a in seq1)
    # A uniform sampler over 200 draws should touch more than one action.
    assert len(set(seq1)) > 1


def test_random_policy_reseed_and_reset():
    """seed()/reset(seed) restart the same stream; reset(None) keeps going."""
    p = RandomPolicy(seed=1)
    first = [p.act() for _ in range(10)]

    p.seed(1)
    assert [p.act() for _ in range(10)] == first

    p.reset(seed=1)
    assert [p.act() for _ in range(10)] == first

    # reset(None) does not reseed: the stream continues (does not repeat `first`).
    p.reset(seed=None)
    cont = [p.act() for _ in range(10)]
    assert cont != first


def test_random_policy_rejects_bad_n_actions():
    with pytest.raises(ValueError):
        RandomPolicy(n_actions=0)


# ===========================================================================
# ToyReplayBuffer.
# ===========================================================================


def _rand_obs(rng):
    return rng.standard_normal(OBS_DIM).astype(np.float32)


def test_replay_buffer_add_sample_and_capacity():
    """The toy buffer accumulates, caps at capacity, and samples well-formed batches."""
    rng = np.random.default_rng(0)
    buf = ToyReplayBuffer(capacity=10, rng=np.random.default_rng(7))

    assert len(buf) == 0
    assert not buf.is_ready(1)

    for i in range(25):
        buf.add(_rand_obs(rng), i % N_ACTIONS, float(i), _rand_obs(rng), i == 24)

    # Capped at capacity but lifetime count keeps climbing.
    assert len(buf) == 10
    assert buf.total_added == 25
    assert buf.is_ready(10)

    obs, actions, rewards, next_obs, dones = buf.sample(8)
    assert obs.shape == (8, OBS_DIM)
    assert next_obs.shape == (8, OBS_DIM)
    assert actions.shape == (8,) and actions.dtype == np.int64
    assert rewards.shape == (8,) and rewards.dtype == np.float32
    assert dones.shape == (8,) and dones.dtype == bool
    assert np.all((0 <= actions) & (actions < N_ACTIONS))


def test_replay_buffer_empty_sample_raises():
    buf = ToyReplayBuffer(capacity=4)
    with pytest.raises(ValueError):
        buf.sample(2)


def test_replay_buffer_rejects_bad_capacity():
    with pytest.raises(ValueError):
        ToyReplayBuffer(capacity=0)


def test_noop_grad_step_runs_and_returns_finite_loss():
    """The no-op update consumes a batch and returns a finite scalar loss."""
    rng = np.random.default_rng(0)
    buf = ToyReplayBuffer(capacity=64, rng=np.random.default_rng(1))
    for i in range(64):
        buf.add(_rand_obs(rng), i % N_ACTIONS, float(i % 3), _rand_obs(rng), False)
    loss = noop_grad_step(buf.sample(32))
    assert isinstance(loss, float)
    assert np.isfinite(loss)


def test_noop_grad_step_rejects_empty_batch():
    empty = (
        np.zeros((0, OBS_DIM), dtype=np.float32),
        np.zeros((0,), dtype=np.int64),
        np.zeros((0,), dtype=np.float32),
        np.zeros((0, OBS_DIM), dtype=np.float32),
        np.zeros((0,), dtype=bool),
    )
    with pytest.raises(ValueError):
        noop_grad_step(empty)


# ===========================================================================
# RssSampler bookkeeping.
# ===========================================================================


def test_rss_sampler_bookkeeping():
    """Sampling records values and computes non-negative growth without error."""
    rss = RssSampler()
    # Two samples are enough to define a baseline + growth; the call must never
    # raise regardless of platform/backend.
    rss.sample()
    rss.sample()
    assert isinstance(rss.growth_mb, float)
    assert rss.growth_mb >= 0.0
    # within_budget is True when measurement is unavailable or only a baseline
    # exists; with a sane budget it must hold for an idle test process.
    assert rss.within_budget(200.0) is True
    if rss.available:
        assert len(rss.samples_mb) == 2
        assert rss.baseline_mb is not None


# ===========================================================================
# The runner end to end (the OFFLINE TC11 proof).
# ===========================================================================


def _mixed_outcomes(n):
    """A deterministic win/loss/timeout mix of length ``n`` (every 3rd a timeout)."""
    pattern = ["win", "loss", "timeout"]
    return [pattern[i % 3] for i in range(n)]


def test_runner_completes_100_episodes_no_crash_and_terminates():
    """>= 100 short episodes: no crash, all terminate, win-rate matches the mix."""
    n = 102  # > 100, and divisible by 3 so the win/loss/timeout mix is exact
    factory = _make_factory(_mixed_outcomes(n), max_episode_steps=5)

    result = run_random(
        factory,
        episodes=n,
        seed=123,
        batch_size=8,
        min_replay=16,
        max_episode_steps=5,
        log=None,  # silent
    )

    # No crash, no hang: every episode terminated and was recorded.
    assert result.crashes == 0
    assert result.completed == n
    assert len(result.episodes) == n

    # Outcome counts match the scripted 1/3 : 1/3 : 1/3 mix exactly.
    assert result.wins == n // 3
    assert result.losses == n // 3
    assert result.timeouts == n // 3

    # A win-rate is computed and equals wins / completed.
    assert result.win_rate == pytest.approx(result.wins / n)
    assert 0.0 <= result.win_rate <= 1.0

    # Every recorded episode has a valid outcome and a bounded length (proof that
    # nothing ran away — the env timeout caps a non-dying episode at 5 steps).
    for rec in result.episodes:
        assert rec.outcome in {"win", "loss", "timeout"}
        assert 1 <= rec.length <= 5


def test_runner_exercises_replay_and_noop_grad_step():
    """The full path runs: transitions accumulate and no-op grad steps fire."""
    n = 100
    factory = _make_factory(_mixed_outcomes(n), max_episode_steps=5)

    result = run_random(
        factory,
        episodes=n,
        seed=7,
        batch_size=4,
        min_replay=8,  # warm quickly so updates run within the first episodes
        max_episode_steps=5,
        log=None,
    )

    # Transitions were stored (rollout -> store) ...
    assert result.replay_total_added > 0
    assert result.replay_size > 0
    # ... and the update path ran (sample -> no-op update) once warm.
    assert result.grad_steps > 0
    assert result.last_loss is not None
    assert np.isfinite(result.last_loss)


def test_runner_samples_rss_every_10_episodes():
    """RSS sampling cadence fires once per ``RSS_SAMPLE_EVERY`` episodes."""
    assert RSS_SAMPLE_EVERY == 10
    n = 30  # -> samples at episodes 10, 20, 30
    factory = _make_factory(_mixed_outcomes(n), max_episode_steps=4)

    result = run_random(
        factory,
        episodes=n,
        seed=0,
        batch_size=4,
        min_replay=8,
        rss_sample_every=10,
        max_episode_steps=4,
        log=None,
    )

    assert result.rss is not None
    if result.rss.available:
        # 30 episodes / 10 == 3 samples; bookkeeping ran without error.
        assert len(result.rss.samples_mb) == n // 10
        assert result.rss.growth_mb >= 0.0
    # Offline (only the local Python process), growth must not be flagged as over
    # the combined-process budget.
    assert result.rss.within_budget(200.0) is True


def test_runner_is_deterministic_under_same_seed():
    """Same seed + same scripted outcomes -> identical episode records and counts."""
    outcomes = _mixed_outcomes(30)
    r1 = run_random(
        _make_factory(list(outcomes), max_episode_steps=5),
        episodes=30,
        seed=99,
        batch_size=4,
        min_replay=8,
        max_episode_steps=5,
        log=None,
    )
    r2 = run_random(
        _make_factory(list(outcomes), max_episode_steps=5),
        episodes=30,
        seed=99,
        batch_size=4,
        min_replay=8,
        max_episode_steps=5,
        log=None,
    )

    assert (r1.wins, r1.losses, r1.timeouts) == (r2.wins, r2.losses, r2.timeouts)
    assert r1.grad_steps == r2.grad_steps
    assert [(e.outcome, e.length) for e in r1.episodes] == [
        (e.outcome, e.length) for e in r2.episodes
    ]
    assert r1.last_loss == pytest.approx(r2.last_loss)


def test_runner_counts_bridge_crash_and_continues():
    """A BridgeError in one episode is counted as a crash; the run continues."""

    good = _episode_script("win", max_episode_steps=5)

    class _CrashOnRecvBridge(ScriptedTracerBridge):
        # recv() with an empty queue raises BridgeError (the reset_ack is missing),
        # simulating a transport failure on this episode only.
        def __init__(self):
            super().__init__([])

    bridges = []
    calls = {"n": 0}

    def factory():
        calls["n"] += 1
        if calls["n"] == 2:
            b = _CrashOnRecvBridge()
        else:
            b = ScriptedTracerBridge(list(good))
        bridges.append(b)
        return b

    result = run_random(
        factory,
        episodes=3,
        seed=0,
        batch_size=4,
        min_replay=8,
        max_episode_steps=5,
        log=None,
    )

    # Exactly one crash; the other two episodes completed as wins.
    assert result.crashes == 1
    assert result.completed == 2
    assert result.wins == 2


def test_run_result_win_rate_zero_when_nothing_completed():
    """A RunResult with no completed episodes reports a 0.0 win-rate (no div-by-0)."""
    empty = RunResult()
    assert empty.completed == 0
    assert empty.win_rate == 0.0


def test_episode_record_fields():
    """EpisodeRecord carries the documented fields."""
    rec = EpisodeRecord(index=3, outcome="win", length=4, total_reward=1.5)
    assert rec.index == 3
    assert rec.outcome == "win"
    assert rec.length == 4
    assert rec.total_reward == pytest.approx(1.5)


# ===========================================================================
# Regression: reset seed must equal episode_seed (seed + ep), not -1 (T10 fix).
# ===========================================================================


def test_reset_seed_matches_per_episode_seed():
    """Each episode's reset message must carry seed == base_seed + ep (AC7/TC14).

    Before the T10 fix, _run_one_episode called env.reset(seed=env.episode).
    MCPvPEnv.reset increments _episode BEFORE using the seed, so env.episode
    was still -1 at the call site — every reset hit the wire with seed=-1,
    defeating per-episode spawn/gear variation.

    This test drives 5 episodes, captures the wire dicts that ScriptedTracerBridge
    recorded in .sent, and asserts that each episode's reset message has
    seed == base_seed + episode_index.
    """
    BASE_SEED = 42
    N_EPS = 5
    factory = _make_factory(
        _mixed_outcomes(N_EPS),
        max_episode_steps=5,
        steps_until_terminal=2,
    )

    run_random(
        factory,
        episodes=N_EPS,
        seed=BASE_SEED,
        batch_size=4,
        min_replay=8,
        max_episode_steps=5,
        log=None,
    )

    assert len(factory.bridges) == N_EPS, (
        f"expected {N_EPS} bridges, got {len(factory.bridges)}"
    )
    for ep, bridge in enumerate(factory.bridges):
        expected_seed = BASE_SEED + ep
        # The env always sends the reset dict first (before any step dicts).
        reset_dicts = [d for d in bridge.sent if d.get("type") == "reset"]
        assert len(reset_dicts) >= 1, (
            f"ep {ep}: no 'reset' message found in bridge.sent={bridge.sent!r}"
        )
        actual_seed = reset_dicts[0]["seed"]
        assert actual_seed == expected_seed, (
            f"ep {ep}: reset seed={actual_seed!r}, expected {expected_seed!r} "
            f"(base_seed={BASE_SEED}, ep={ep}). "
            "If this fails, _run_one_episode is not threading episode_seed into "
            "env.reset() — the original T10 bug (seed=-1 on every episode)."
        )
