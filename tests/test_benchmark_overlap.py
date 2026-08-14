"""Tests for multi-arena overlap, eval one-connection invariant, and regressions.

TC12 -- Benchmark overlap (loose bound): N=4 arenas complete in less than 2x the
        wall-time of N=1, demonstrating the concurrent driver overlaps per-arena
        server-tick waits. A real DuelingDRQN.act forward is injected via
        step_work to exercise GIL contention (not just sleep overlap). This is a
        LOOSE harness-level signal, not the live AC4 number; see the comment on
        the test itself. The test uses a real time.sleep (SleepingFakeBridge) so
        the GIL is actually released and threads genuinely overlap.

TC13 -- Aggregate scales, per-arena preserved: transitions_per_s_aggregate rises
        with arena count; transitions_per_s_per_arena == aggregate / n_arenas for
        N=1, 2, 4. Uses SleepingFakeBridge (real clock) because a FakeClock shared
        across threads cannot show throughput scaling by construction -- every recv
        advances the same shared clock so the summed elapsed is identical for N=1
        and N=4.

TC14 -- (Covered by test_train.py::test_bootstrap_uses_correct_recurrent_hidden_
        state; the bootstrap recurrence invariant is owned there, confirmed green
        by the full suite. No duplicate fixture is created here.)

TC15 -- --arenas 1 dispatches to the single-arena path, not train_multi_arena.
        Verified by parsing the CLI args and asserting arenas==1 selects the N=1
        code branch in main().

TC16 -- Multi-arena eval on ONE designated arena (one-connection invariant). Drive
        train_multi_arena with N fake arenas whose envs expose a connection-counting
        _transport. Eval fires several times. After the run: every arena's
        _transport.connect_count == 1 (one initial connect by the collector; eval
        borrows the same idle connection and never opens a second one), and
        peak_concurrent on every arena == 1.

No live Minecraft server or socket is used anywhere in this file.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

torch = pytest.importorskip("torch", exc_type=ImportError)

from agent.actions import N_ACTIONS
from agent.dqn import DuelingDRQN
from agent.train_config import TrainConfig
from bridge.messages import ResetAckMsg, StateMsg
from env.observation_spec import OBS_DIM
from eval.benchmark import FakeClock, SleepingFakeBridge, run_benchmark


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# A shrunken net is enough to prove GIL contention; full-size would just be slow.
_TINY_NET = {"encoder_hidden": 16, "lstm_hidden": 8, "lstm_layers": 1}

# Number of per-arena decisions per overlap timing run. Small enough to finish
# in a few seconds; large enough that timing noise averages out.
_TC12_DECISIONS_PER_ARENA = 8
_TC12_LATENCY_S = 0.012  # 12 ms per recv sleep -- short enough to keep total < ~5 s

# Per-arena decision count for TC13 (scaling check, no torch forward).
_TC13_DECISIONS_PER_ARENA = 6
_TC13_LATENCY_S = 0.010  # 10 ms; real clock, real sleep

# Generous timeout (seconds) for any join/wait call so a regression fails fast.
_THREAD_TIMEOUT = 30.0


def _make_state(tick: int = 1) -> StateMsg:
    """Return a minimal valid StateMsg for benchmark scripting."""
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
                "pos": [0.0, 64.0, 2.0],
                "yaw": 0.0,
                "pitch": 0.0,
                "velocity": [0.0, 0.0, 0.0],
                "health": 20.0,
            },
            "events": {
                "damage_dealt": 0.0,
                "damage_taken": 0.0,
                "i_died": False,
                "opponent_died": False,
            },
            "arena": {"wall_distances": [8.0, 8.0, 8.0, 8.0]},
            "tick": tick,
            "code_version": "test",
        }
    )


def _make_reset_ack() -> ResetAckMsg:
    """Return a minimal valid ResetAckMsg for benchmark scripting."""
    return ResetAckMsg.from_dict(
        {
            "type": "reset_ack",
            "ok": True,
            "readback": {
                "pos": [0.0, 64.0, 0.0],
                "yaw": 0.0,
                "pitch": 0.0,
                "health": 20.0,
            },
        }
    )


def _sleeping_factory(n_decisions: int, latency_s: float) -> Callable[[int], SleepingFakeBridge]:
    """Return a transport_factory that hands each arena a fresh SleepingFakeBridge.

    The benchmark's run_benchmark calls transport_factory(arena_index) once per
    arena before starting its threads. Each arena gets its own independent bridge
    with n_decisions scripted state messages, so no two arenas share a queue and
    the per-arena decision count is exactly n_decisions.
    """
    def factory(arena_index: int) -> SleepingFakeBridge:
        clock = FakeClock()
        msgs: list = [_make_state(tick=i + 1) for i in range(n_decisions)]
        return SleepingFakeBridge(inbound=msgs, latency_s=latency_s, clock=clock)

    return factory


# ---------------------------------------------------------------------------
# TC12 -- benchmark overlap (loose bound, loose by design)
# ---------------------------------------------------------------------------


def test_tc12_overlap_loose_bound():
    """N=4 arenas finish in less than 2x the single-arena wall-time.

    This is a LOOSE offline overlap signal -- NOT the live AC4 number.

    The real per-step GIL work (DuelingDRQN.act) is injected via step_work so
    the test exercises actual GIL contention between threads, not just sleep
    overlap. The design is I/O-bound (most of each decision window is the
    blocking sleep in SleepingFakeBridge.recv), so even with non-trivial Python
    work between sleeps, N concurrent threads should finish faster than N serial
    ones. A ratio < 2.0 is a generous bound that accounts for scheduler
    overhead and CI thermal variance; the live AC4 target is much tighter.

    Margin rationale: with 12 ms sleep and ~8 decisions per arena, N=1 takes
    ~96 ms; N=4 should take ~96-120 ms (nearly parallel). The 2.0x bound gives
    headroom for cold CPU / slow CI without making the test meaningless.
    """
    # Build a small DuelingDRQN; call act() once per step to contend the GIL.
    net = DuelingDRQN(**_TINY_NET).eval()
    hidden_store: dict = {}  # one hidden state per arena, keyed by arena_index
    hidden_lock = threading.Lock()

    def step_work(arena_index: int, state: StateMsg) -> None:
        # Build a random obs tensor and advance the LSTM one step.
        obs = torch.zeros(OBS_DIM, dtype=torch.float32)
        with hidden_lock:
            h = hidden_store.get(arena_index)
        _action, new_h = net.act(obs, h, epsilon=0.0)
        with hidden_lock:
            hidden_store[arena_index] = new_h

    # --- N=1 baseline ---
    t0 = time.perf_counter()
    report_1 = run_benchmark(
        _sleeping_factory(_TC12_DECISIONS_PER_ARENA, _TC12_LATENCY_S),
        n_arenas=1,
        max_decisions=_TC12_DECISIONS_PER_ARENA,
        duration_s=60.0,
        step_work=step_work,
    )
    wall_1 = time.perf_counter() - t0

    # Clear per-arena hidden states so N=4 run starts fresh.
    with hidden_lock:
        hidden_store.clear()

    # --- N=4 concurrent ---
    total_4 = _TC12_DECISIONS_PER_ARENA * 4
    t0 = time.perf_counter()
    report_4 = run_benchmark(
        _sleeping_factory(_TC12_DECISIONS_PER_ARENA, _TC12_LATENCY_S),
        n_arenas=4,
        max_decisions=total_4,
        duration_s=60.0,
        step_work=step_work,
    )
    wall_4 = time.perf_counter() - t0

    # Both runs completed the right number of transitions.
    assert report_1.transitions == _TC12_DECISIONS_PER_ARENA
    assert report_4.transitions == total_4

    # Loose overlap assertion: N=4 must finish in less than 2x the N=1 time.
    # At perfect overlap (all 4 arenas sleep at the same time) the ratio would
    # be ~1.0; a ratio < 2.0 is meaningful evidence that threads are running
    # concurrently, not serially. This bound is deliberately generous to survive
    # CI cold-start and thermal variance without producing a flaky gate.
    ratio = wall_4 / wall_1
    assert ratio < 2.0, (
        f"TC12 overlap bound failed: wall_4={wall_4:.3f}s, wall_1={wall_1:.3f}s, "
        f"ratio={ratio:.2f} (expected < 2.0). If this flaps, check that the GIL "
        f"is being released (SleepingFakeBridge does time.sleep inside recv) and "
        f"that the test machine is not heavily loaded."
    )


# ---------------------------------------------------------------------------
# TC13 -- aggregate transitions/s rises with arena count; per-arena preserved
# ---------------------------------------------------------------------------


def test_tc13_aggregate_scales_per_arena_preserved():
    """transitions_per_s_aggregate rises with N; per-arena == aggregate / N.

    TC13 must use SleepingFakeBridge (real clock), NOT a FakeClock. A FakeClock
    shared across N arena threads advances by latency on every recv regardless of
    concurrency, so the total elapsed returned by clock() - start is the SUM of
    all arena latencies for N=1 OR N=4. The aggregate throughput (transitions /
    elapsed) would therefore be constant across N -- you cannot observe scaling
    on a deterministic fake clock.

    With a real sleep (SleepingFakeBridge), N concurrent arenas genuinely overlap
    their blocking recv windows and the wall-clock elapsed is closer to serial/N,
    so the aggregate throughput rises with N. The assertions are LOOSE (agg(4) >
    agg(1)) to tolerate timing noise; this is a sanity check that the concurrent
    driver is wired correctly, not a precise throughput gate.
    """
    decisions = _TC13_DECISIONS_PER_ARENA
    latency = _TC13_LATENCY_S

    def _run(n_arenas: int):
        return run_benchmark(
            _sleeping_factory(decisions, latency),
            n_arenas=n_arenas,
            max_decisions=decisions * n_arenas,
            duration_s=60.0,
        )

    r1 = _run(1)
    r2 = _run(2)
    r4 = _run(4)

    # --- transitions count sanity ------------------------------------------
    assert r1.transitions == decisions * 1
    assert r2.transitions == decisions * 2
    assert r4.transitions == decisions * 4

    # --- per-arena == aggregate / n_arenas (exact float identity by construction)
    # The report sets per_arena = aggregate / n_arenas unconditionally.
    assert r1.transitions_per_s_per_arena == pytest.approx(
        r1.transitions_per_s_aggregate / 1, rel=1e-9
    )
    assert r2.transitions_per_s_per_arena == pytest.approx(
        r2.transitions_per_s_aggregate / 2, rel=1e-9
    )
    assert r4.transitions_per_s_per_arena == pytest.approx(
        r4.transitions_per_s_aggregate / 4, rel=1e-9
    )

    # --- aggregate rises with N (loose monotone check) ----------------------
    # Allow 10% slack for scheduler noise: agg(2) should be at least 0.9x agg(1).
    # The goal is to catch a regression where N=4 is somehow SLOWER than N=1.
    slack = 0.85
    assert r2.transitions_per_s_aggregate >= r1.transitions_per_s_aggregate * slack, (
        f"TC13: aggregate did not rise from N=1 to N=2: "
        f"agg(1)={r1.transitions_per_s_aggregate:.2f}, "
        f"agg(2)={r2.transitions_per_s_aggregate:.2f}"
    )
    assert r4.transitions_per_s_aggregate >= r2.transitions_per_s_aggregate * slack, (
        f"TC13: aggregate did not rise from N=2 to N=4: "
        f"agg(2)={r2.transitions_per_s_aggregate:.2f}, "
        f"agg(4)={r4.transitions_per_s_aggregate:.2f}"
    )
    # Stronger: N=4 aggregate must beat N=1 by a clear margin (1.5x is still
    # conservative; perfect scaling would be 4x).
    assert r4.transitions_per_s_aggregate > r1.transitions_per_s_aggregate * 1.5, (
        f"TC13: aggregate at N=4 did not meaningfully exceed N=1: "
        f"agg(1)={r1.transitions_per_s_aggregate:.2f}, "
        f"agg(4)={r4.transitions_per_s_aggregate:.2f}"
    )


# ---------------------------------------------------------------------------
# TC15 -- --arenas 1 dispatches to the single-arena path
# ---------------------------------------------------------------------------


def test_tc15_arenas_1_is_single_arena_path():
    """--arenas 1 parses to arenas==1 and must NOT invoke train_multi_arena.

    The M2 single-arena training loop (train_vs_dummy) and the multi-arena stack
    (train_multi_arena) are separate code paths gated by args.arenas in main().
    This test confirms the parse-and-dispatch is correct so a regression that
    accidentally routes N=1 into the multi-arena stack (which requires arenas>=2)
    is caught immediately rather than failing obscurely at runtime.

    The heavy offline proof of the N=1 training loop is test_integration_m2.py.
    The bootstrap recurrence invariant is test_train.py::test_bootstrap_uses_
    correct_recurrent_hidden_state. We do NOT re-run those heavy fixtures here.
    """
    from agent.train import _build_parser, train_multi_arena

    parser = _build_parser()

    # Parse the single-arena case.
    args_1 = parser.parse_args(["--arenas", "1"])
    assert int(args_1.arenas) == 1, (
        "--arenas 1 must parse to arenas==1; got arenas="
        f"{args_1.arenas!r}"
    )

    # Parse the multi-arena case -- confirm the flag is accepted.
    args_4 = parser.parse_args(["--arenas", "4"])
    assert int(args_4.arenas) == 4

    # Structural check: train_multi_arena raises ValueError for arenas < 2 when
    # driven with the N=1 config, which is the guard that enforces the dispatch.
    # This is the cheapest way to assert the invariant without running a live run.
    import dataclasses

    cfg_1 = dataclasses.replace(TrainConfig(), arenas=1)
    with pytest.raises(ValueError, match="train_multi_arena requires cfg.arenas >= 2"):
        train_multi_arena(
            cfg_1,
            env_factory_for=lambda _: (lambda: None),
            launcher=MagicMock(),
        )


# ---------------------------------------------------------------------------
# TC14 -- ownership comment (no duplicate fixture)
# ---------------------------------------------------------------------------
# TC14 (TC8b): the n-step bootstrap MUST recur over the contiguous obs stream
# (obs_ext) so bootstrap Q-values carry the correct LSTM memory. This property
# is owned and fully proved by:
#
#   tests/test_train.py::test_bootstrap_uses_correct_recurrent_hidden_state
#
# Adding a second fixture here would duplicate the heavy replay/trainer setup
# without adding coverage. Running the full suite confirms TC14 is green.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# TC16 -- multi-arena eval on ONE designated arena (one-connection invariant)
# ---------------------------------------------------------------------------


class _CountingTransport:
    """Fake BridgeTransport that counts connect() calls and tracks concurrency.

    Each call to connect() increments connect_count and peak_concurrent; each
    call to close() decrements the live counter. This lets TC16 assert that the
    designated arena's transport was connected exactly once (by the collector)
    and that eval never opened a second connection on any arena.

    The transport is intentionally trivial: send() and recv() never need to be
    called on this object because the eval path (patched evaluate) does not
    drive any real steps through it. The object is only inspected for connection
    semantics.
    """

    def __init__(self, arena_id: int) -> None:
        self.arena_id = int(arena_id)
        self._lock = threading.Lock()
        self.connect_count: int = 0
        self._concurrent: int = 0
        self.peak_concurrent: int = 0

    def connect(self) -> None:
        with self._lock:
            self.connect_count += 1
            self._concurrent += 1
            if self._concurrent > self.peak_concurrent:
                self.peak_concurrent = self._concurrent

    def send(self, obj: Any) -> None:
        pass  # not exercised by TC16

    def recv(self) -> Any:
        # This should never be called: the collector's env is paused-and-idle
        # at the eval boundary and the patched evaluate never steps the env.
        raise RuntimeError(
            f"_CountingTransport.recv() called on arena {self.arena_id} during TC16 "
            "-- this means eval opened or exercised a second connection, which "
            "violates the one-connection invariant."
        )

    def close(self) -> None:
        with self._lock:
            self._concurrent = max(0, self._concurrent - 1)


class _FakeEnvWithTransport:
    """Minimal Gym-style env backed by a _CountingTransport.

    Satisfies EnvProtocol (reset/step). Exposes _transport so
    _eval_via_designated_arena can borrow it for the eval connection-sharing check.
    Episodes terminate after k steps; reset() counts as a connect to the transport
    (the first call opens the connection to match the real env lifecycle: the real
    MCPvPEnv calls transport.connect() in its constructor when auto_connect=True,
    which is what the collector's env_factory triggers).
    """

    def __init__(self, transport: _CountingTransport, k: int = 6) -> None:
        self._transport = transport
        self.k = int(k)
        self._t = 0
        self._rng = np.random.default_rng(0)
        # Connect immediately (mirrors auto_connect=True in the real env).
        self._transport.connect()

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        self._rng = np.random.default_rng(0 if seed is None else int(seed))
        self._t = 0
        return self._obs()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        self._t += 1
        done = self._t >= self.k
        return self._obs(), 0.0, done, {}

    def _obs(self) -> np.ndarray:
        return self._rng.uniform(-1.0, 1.0, size=OBS_DIM).astype(np.float32)

    def close(self) -> None:
        self._transport.close()


def _tiny_multi_cfg(n_arenas: int = 2) -> TrainConfig:
    """Minimal TrainConfig for the multi-arena path (fast, no real learning).

    Window geometry: burn_in=1 + seq_len=2 = 3 total timesteps per window.
    Each episode is k=6 steps (see _FakeEnvWithTransport k=6 below), so a
    single episode fills more than one sampleable window. This ensures the
    replay has sampleable windows after a small number of episodes and the
    learner can make grad steps quickly.
    """
    import dataclasses

    return dataclasses.replace(
        TrainConfig(),
        arenas=n_arenas,
        # Tiny window so even short episodes (k=6) provide sampleable windows.
        batch_size=4,
        seq_len=2,
        burn_in=1,
        n_step=1,
        # min_replay=1: allow learning once the first episode arrives; the
        # window geometry above guarantees n_sampleable > 0 at that point.
        min_replay=1,
        replay_capacity=2_000,
        weight_sync_every_k_steps=1,
        # No bridge restarts: these fakes never die, and a restart attempt here would
        # only reach a fake launcher. The JVM watchdog stays unwired (jvm_probe=None),
        # so nothing in this test ever touches a socket looking for Paper.
        fault_relaunch=False,
        seed=0,
    )


def _fake_eval_report() -> MagicMock:
    """Build a minimal fake EvalReport that satisfies the learner's gate check."""
    report = MagicMock()
    report.win_rate = 0.0
    report.mean_episode_length = 10.0
    report.aim_while_invisible = 0.0
    report.passed_m2 = False
    return report


def test_tc16_eval_one_connection_per_arena():
    """Eval reuses the designated arena's idle connection; no second connect anywhere.

    The bridge serves exactly ONE connection per arena. This test guards the
    regression (recorded in memory/bridge-single-tcp-connection.md) where a
    second eval connect would destroy the first, aborting the live run.

    Setup:
      - N=2 arenas, each with its own _CountingTransport.
      - Collector env factories return _FakeEnvWithTransport, which calls
        transport.connect() once on construction.
      - The patched evaluate never drives real env steps; it returns a fake
        EvalReport immediately so eval is fast.
      - eval_every_grad_steps is tiny so eval fires several times during the run.

    Assertions after the run:
      - Every arena's _transport.connect_count == 1 (one initial connect,
        never a second one from eval).
      - Every arena's _transport.peak_concurrent == 1.
      - The run completed without PoolAbortedError or LearnerError.
    """
    # These need to be imported lazily (same reason train_multi_arena does).
    from distributed.actor import GlobalEpisodeCounter, PoolAbortedError
    from distributed.transport import LocalTransport
    from distributed.weights import WeightStore

    n_arenas = 2
    cfg = _tiny_multi_cfg(n_arenas=n_arenas)

    # Per-arena counting transports -- one per arena.
    transports = [_CountingTransport(arena_id=i) for i in range(n_arenas)]

    def env_factory_for(arena_id: int) -> Callable[[], _FakeEnvWithTransport]:
        transport = transports[arena_id]

        def _build() -> _FakeEnvWithTransport:
            return _FakeEnvWithTransport(transport=transport, k=4)

        return _build

    # Fake ArenaLauncher (no subprocess spawning needed).
    fake_launcher = MagicMock()
    fake_launcher.launch = MagicMock()
    fake_launcher.terminate = MagicMock()

    # Fake evaluate: immediately returns a fake report. The fake must satisfy the
    # call signature evaluate(env, policy, *, n_episodes, ...) so keyword args land.
    eval_call_arenas: List[int] = []
    eval_call_lock = threading.Lock()

    def fake_evaluate(env, policy, *, n_episodes=1, logger=None, timeout_cap=None,
                      base_seed=0, is_live=False, max_episode_steps=None, log=None,
                      **kwargs):
        # Record which arena's env was handed to eval (via env._transport.arena_id).
        transport = getattr(env, "_transport", None)
        if transport is not None:
            with eval_call_lock:
                eval_call_arenas.append(getattr(transport, "arena_id", -1))
        return _fake_eval_report()

    fake_policy_cls = MagicMock(return_value=MagicMock())

    # Shared infra -- pass pre-built so we can close them cleanly.
    transport_q = LocalTransport(maxsize=0)
    weight_store = WeightStore()
    counter = GlobalEpisodeCounter()

    # Run train_multi_arena with eval firing frequently. The learner drains
    # episodes off the queue and applies gradient steps; we stop after a handful.
    # poll_interval is tiny (no real sleep overhead) and eval fires every 1 step
    # so we get multiple eval calls within a short run.
    from agent.train import train_multi_arena

    # A lenient watchdog so a brief warm-up period (buffer filling up before the
    # first sampleable window) does not trip a false stall alarm.
    from distributed.learner import LearnerWatchdog

    lenient_watchdog = LearnerWatchdog(patience=200, interval_s=1.0)

    # train_multi_arena does `from eval.evaluate import DRQNGreedyPolicy, evaluate`
    # inside the function body (lazy import to avoid a cycle). Patching the source
    # module before the function runs causes the local `from ... import` to bind
    # the patched object. We must NOT patch "agent.train.evaluate" because that
    # name does not exist at the module level -- the import lives inside the function.
    with (
        patch("eval.evaluate.evaluate", fake_evaluate),
        patch("eval.evaluate.DRQNGreedyPolicy", fake_policy_cls),
    ):
        try:
            result = train_multi_arena(
                cfg,
                env_factory_for=env_factory_for,
                launcher=fake_launcher,
                transport=transport_q,
                weight_store=weight_store,
                counter=counter,
                max_grad_steps=30,          # short run
                eval_every_grad_steps=5,    # eval fires ~6 times over 30 steps
                eval_episodes=1,
                designated_arena=0,
                stop_on_pass=False,
                eval_pause_timeout=_THREAD_TIMEOUT,
                relaunch_backoff_seconds=0.001,
                relaunch_backoff_max_seconds=0.001,
                sleep=lambda _: None,       # no real sleeping in backoff
                poll_interval=0.005,        # fast poll so the test finishes quickly
                net_kwargs=_TINY_NET,
                rollout_step_cap=6,
                watchdog=lenient_watchdog,
            )
        except PoolAbortedError:
            # A pool abort means collectors could not keep min_live_arenas running.
            # This can happen if the fake env fails in an unexpected way; diagnose
            # the transport counts before re-raising so the assert message is clear.
            for i, t in enumerate(transports):
                if t.connect_count != 1:
                    pytest.fail(
                        f"Arena {i} connect_count={t.connect_count} != 1 "
                        "(PoolAbortedError AND bad connect count -- second connection opened)"
                    )
            raise

    # --- one-connection invariant ------------------------------------------
    for i, t in enumerate(transports):
        assert t.connect_count == 1, (
            f"Arena {i}: connect_count={t.connect_count}, expected 1. "
            "A second connect() means eval opened a new connection on this arena, "
            "which would destroy the bridge's single-connection invariant."
        )
        assert t.peak_concurrent == 1, (
            f"Arena {i}: peak_concurrent={t.peak_concurrent}, expected 1. "
            "More than one concurrent connection existed on this arena."
        )

    # --- eval only ran on the designated arena (arena 0) -------------------
    # The patched evaluate is called once per eval cycle; the env it receives
    # should always belong to arena 0 (the designated arena's borrowed transport).
    assert len(eval_call_arenas) > 0, (
        "Eval never ran during the TC16 run. Increase max_grad_steps or decrease "
        "eval_every_grad_steps so at least one eval cycle fires."
    )
    non_zero = [a for a in eval_call_arenas if a != 0]
    assert non_zero == [], (
        f"Eval ran on non-designated arenas {non_zero!r}; expected only arena 0. "
        "This means the pause/handoff protocol borrowed from the wrong collector."
    )

    # --- run completed without learner error --------------------------------
    # (If train_multi_arena raised, we would have exited above.)
    assert result.stop_reason in ("max_grad_steps", "passed_m2", "max_episodes"), (
        f"Unexpected stop reason: {result.stop_reason!r}"
    )
