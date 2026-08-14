"""Tests for LearnerLoop drain/learn/publish behaviour and the LearnerWatchdog.

TC7  -- LearnerLoop drains episodes -> add_episode + learn() + publishes every K.
TC8  -- LearnerLoop is the sole replay mutator (Collector holds no replay ref).
TC17 -- Watchdog: (a) WARM stall fires WatchdogError, (b) cold buffer never fires,
         (c) healthy grad advance never fires.

No socket / live server; all tests use a tiny in-process fake env and
LocalTransport. Every torch-dependent body guards with pytest.importorskip.
Threading discipline: every join/wait uses an explicit timeout and asserts on
is_alive() so a regression fails fast instead of hanging the suite.
"""

from __future__ import annotations

import threading
import time
from typing import List, Optional, Tuple

import numpy as np
import pytest

from agent.train_config import TrainConfig
from distributed.serialization import Episode
from distributed.transport import LocalTransport, TransportError
from distributed.weights import WeightStore
from distributed.learner import LearnerLoop, LearnerWatchdog, WatchdogError

torch = pytest.importorskip("torch", exc_type=ImportError)


# ---------------------------------------------------------------------------
# Shared tiny config / net constants
# ---------------------------------------------------------------------------

from env.observation_spec import OBS_DIM
from agent.actions import N_ACTIONS

_TINY_NET = {"encoder_hidden": 16, "lstm_hidden": 8, "lstm_layers": 1}
_EPISODE_K = 14  # steps per fake episode; must be > burn_in + seq_len + n_step


def _tiny_cfg(**overrides) -> TrainConfig:
    """Fast TrainConfig: small windows, low warm-up threshold, tiny replay."""
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
        min_replay=1,
        per_beta_anneal_steps=100,
        eval_interval=0,
        checkpoint_interval=0,
        log_interval=0,
        seed=0,
        # multi-arena fields -- required by TrainConfig validation
        weight_sync_every_k_steps=5,
    )
    base.update(overrides)
    return TrainConfig(**base)


# ---------------------------------------------------------------------------
# Minimal Gym-style fake env (no socket)
# ---------------------------------------------------------------------------


class _FakeEnv:
    """Terminates after k steps, deterministic from seed."""

    def __init__(self, k: int = _EPISODE_K) -> None:
        self.k = k
        self._rng = np.random.default_rng(0)
        self._t = 0

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        self._rng = np.random.default_rng(0 if seed is None else int(seed))
        self._t = 0
        return self._obs()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        self._t += 1
        obs = self._obs()
        reward = float(self._rng.uniform(-1.0, 1.0))
        done = self._t >= self.k
        return obs, reward, done, {"t": self._t}

    def _obs(self) -> np.ndarray:
        return self._rng.uniform(-1.0, 1.0, size=OBS_DIM).astype(np.float32)


# ---------------------------------------------------------------------------
# Episode factory shared by TC7 and TC8
# ---------------------------------------------------------------------------

_H_SIZE = 8  # matches _TINY_NET lstm_hidden


def _make_hidden_arr(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((2, 1, _H_SIZE)).astype(np.float32)


def _make_episode(arena_id: int = 0, n_transitions: int = _EPISODE_K) -> Episode:
    """Build a small realistic Episode (pure-NumPy, no torch needed)."""
    transitions = []
    hidden_states = []
    for i in range(n_transitions):
        rng = np.random.default_rng(arena_id * 1000 + i)
        obs = rng.uniform(-1.0, 1.0, size=OBS_DIM).astype(np.float32)
        next_obs = rng.uniform(-1.0, 1.0, size=OBS_DIM).astype(np.float32)
        transitions.append(
            (obs, i % N_ACTIONS, float(i) * 0.1, next_obs, i == n_transitions - 1)
        )
        hidden_states.append(_make_hidden_arr(arena_id * 1000 + i))
    return Episode(
        transitions=transitions,
        hidden_states=hidden_states,
        arena_id=arena_id,
        policy_version=0,
        code_version="test",
        total_reward=sum(float(i) * 0.1 for i in range(n_transitions)),
    )


# ---------------------------------------------------------------------------
# Trainer factory
# ---------------------------------------------------------------------------


def _make_trainer(cfg: Optional[TrainConfig] = None):
    from agent.train import Trainer

    return Trainer(cfg or _tiny_cfg(), net_kwargs=dict(_TINY_NET), seed_global=False)


# ---------------------------------------------------------------------------
# TC7 -- drain -> add_episode -> learn -> publish every K
# ---------------------------------------------------------------------------


class TestTC7LearnerLoopDrainLearnPublish:
    """TC7: end-to-end LearnerLoop with a real Trainer and LocalTransport."""

    def _run_loop_on_thread(
        self,
        loop: LearnerLoop,
        *,
        timeout: float = 10.0,
    ) -> threading.Thread:
        """Start ``loop.run()`` on a daemon thread and return it."""
        t = threading.Thread(target=loop.run, daemon=True)
        t.start()
        return t

    def test_replay_grows_and_grad_steps_advance(self):
        """After feeding N episodes the replay has content and grad_step advanced."""
        cfg = _tiny_cfg(
            min_replay=1,
            weight_sync_every_k_steps=3,
            batch_size=4,
        )
        trainer = _make_trainer(cfg)
        transport = LocalTransport()
        store = WeightStore()

        loop = LearnerLoop(trainer, transport, store, cfg, drain_batch=8)

        # Push several episodes before starting the loop so it has work immediately.
        n_episodes = 8
        for i in range(n_episodes):
            ep = _make_episode(arena_id=i % 3, n_transitions=_EPISODE_K)
            transport.send(ep)

        # Start the loop on a daemon thread.
        t = self._run_loop_on_thread(loop)

        # Give the loop time to drain the pre-loaded episodes and take some steps.
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            if trainer.grad_step >= 1:
                break
            time.sleep(0.05)

        # Stop cleanly by closing the transport then stopping the loop.
        loop.stop()
        transport.close()
        t.join(timeout=5.0)
        assert not t.is_alive(), "learner thread did not exit after stop+close"

        assert loop.received >= 1, (
            f"loop.received={loop.received}: no episodes were drained from the transport"
        )
        assert len(trainer.replay) >= 1, "replay must have grown after add_episode"
        assert trainer.grad_step >= 1, "grad_step must advance once buffer is warm"
        assert loop.error is None, f"loop exited with error: {loop.error}"

    def test_weight_store_version_bumps_every_k_steps(self):
        """WeightStore version advances past the initial v0 publish on K-step cadence."""
        K = 4
        cfg = _tiny_cfg(
            min_replay=1,
            weight_sync_every_k_steps=K,
            batch_size=4,
        )
        trainer = _make_trainer(cfg)
        transport = LocalTransport()
        store = WeightStore()

        loop = LearnerLoop(trainer, transport, store, cfg, drain_batch=8)

        # Pre-load enough episodes for several learn passes.
        for i in range(20):
            transport.send(_make_episode(arena_id=i % 4, n_transitions=_EPISODE_K))

        t = self._run_loop_on_thread(loop)

        # Wait for at least 2 publishes beyond the initial v0 publish.
        target_store_version = 2
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            _, v = store.latest()
            if v >= target_store_version:
                break
            time.sleep(0.05)

        loop.stop()
        transport.close()
        t.join(timeout=5.0)
        assert not t.is_alive(), "learner thread did not exit after stop+close"

        _, final_version = store.latest()
        assert final_version >= 1, (
            "store version should have advanced past the initial v0 publish"
        )
        # The publish cadence: version 0 is published before the loop.
        # Each time grad_step crosses a K boundary, one more publish fires.
        # The loop may stop mid-boundary so the final version is:
        #   1 (initial) + number of K-boundaries crossed = 1 + grad_step // K
        # However the loop stop and the version read are not perfectly atomic,
        # so allow off-by-one in either direction (the key property is monotone
        # advance, not the exact count at the exact stop instant).
        gs = trainer.grad_step
        expected_min = 1  # at least one publish beyond v0
        expected_max = 1 + (gs // K) + 1  # at most one publish ahead
        assert expected_min <= final_version <= expected_max, (
            f"store version {final_version} out of expected range "
            f"[{expected_min}, {expected_max}] for grad_step={gs}, K={K}"
        )

    def test_initial_publish_at_version_zero_before_episodes(self):
        """The loop publishes an initial snapshot at version 0 before draining."""
        cfg = _tiny_cfg(weight_sync_every_k_steps=50)
        trainer = _make_trainer(cfg)
        transport = LocalTransport()
        store = WeightStore()

        loop = LearnerLoop(trainer, transport, store, cfg)

        # Start without any episodes queued -- the loop will block on recv().
        t = self._run_loop_on_thread(loop)

        # The initial publish must happen before the first recv() blocks.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            _, v = store.latest()
            if v == 0:
                break
            time.sleep(0.01)

        _, initial_version = store.latest()

        # Clean up: close transport so the blocked recv unblocks.
        transport.close()
        t.join(timeout=5.0)
        assert not t.is_alive(), "learner thread did not exit after transport.close()"

        assert initial_version == 0, (
            f"initial publish version should be 0, got {initial_version}"
        )

    def test_loop_exits_cleanly_on_transport_close(self):
        """Closing the transport while the loop is blocked on recv() exits cleanly."""
        cfg = _tiny_cfg()
        trainer = _make_trainer(cfg)
        transport = LocalTransport()
        store = WeightStore()

        loop = LearnerLoop(trainer, transport, store, cfg)
        t = self._run_loop_on_thread(loop)

        # Let the initial publish happen.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            _, v = store.latest()
            if v >= 0:
                break
            time.sleep(0.01)

        transport.close()
        t.join(timeout=5.0)
        assert not t.is_alive(), "learner thread hung after transport close"
        assert loop.error is None, f"unexpected error on clean close: {loop.error}"
        assert loop.stopped


# ---------------------------------------------------------------------------
# TC8 -- sole replay mutator: Collector holds no replay reference
# ---------------------------------------------------------------------------


class TestTC8SoleReplayMutator:
    """TC8: structural and observational checks that only the LearnerLoop
    calls add_episode and that Collector exposes no replay reference."""

    def test_collector_has_no_replay_attribute(self):
        """Collector must not carry a .replay attribute (design-level check)."""
        from distributed.actor import Collector, GlobalEpisodeCounter

        # Build a minimal Collector with stub dependencies; never start it.
        cfg = _tiny_cfg(arenas=1)
        transport = LocalTransport()
        store = WeightStore()
        counter = GlobalEpisodeCounter()

        class _MinimalLauncher:
            def launch(self, arena_id: int) -> None:
                pass

            def terminate(self, arena_id: int) -> None:
                pass

        # Fake policy satisfying RolloutPolicy.
        from distributed.weights import SnapshotPolicy

        def _net_factory():
            from agent.dqn import DuelingDRQN
            return DuelingDRQN(**_TINY_NET)

        policy = SnapshotPolicy(_net_factory, generator_seed=0, arena_id=0)

        # Fake env factory -- never called because we do not start the thread.
        def _env_factory():
            return _FakeEnv()

        collector = Collector(
            arena_id=0,
            env_factory=_env_factory,
            policy=policy,
            transport=transport,
            weight_store=store,
            cfg=cfg,
            launcher=_MinimalLauncher(),
            counter=counter,
            sleep=lambda _s: None,
        )

        assert not hasattr(collector, "replay"), (
            "Collector must not have a .replay attribute -- "
            "only LearnerLoop may mutate the replay buffer"
        )

    def test_only_learner_thread_calls_add_episode(self):
        """Spy on add_episode: only the learner thread (not others) must call it."""
        import threading

        cfg = _tiny_cfg(min_replay=1, weight_sync_every_k_steps=10)
        trainer = _make_trainer(cfg)
        transport = LocalTransport()
        store = WeightStore()

        # Spy: record the thread id of every add_episode caller.
        caller_thread_ids: List[int] = []
        real_add_episode = trainer.replay.add_episode

        def _spy_add_episode(transitions, *, hidden_states):
            caller_thread_ids.append(threading.current_thread().ident)
            return real_add_episode(transitions, hidden_states=hidden_states)

        trainer.replay.add_episode = _spy_add_episode

        loop = LearnerLoop(trainer, transport, store, cfg, drain_batch=8)

        # Pre-load episodes.
        for i in range(6):
            transport.send(_make_episode(arena_id=i, n_transitions=_EPISODE_K))

        learner_thread = threading.Thread(target=loop.run, daemon=True)
        learner_thread.start()
        learner_thread_id = learner_thread.ident

        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            if loop.received >= 6:
                break
            time.sleep(0.05)

        loop.stop()
        transport.close()
        learner_thread.join(timeout=5.0)
        assert not learner_thread.is_alive(), "learner thread did not exit"

        assert len(caller_thread_ids) > 0, "add_episode was never called"
        for tid in caller_thread_ids:
            assert tid == learner_thread_id, (
                f"add_episode called from thread {tid}, "
                f"expected learner thread {learner_thread_id}"
            )


# ---------------------------------------------------------------------------
# TC17 -- LearnerWatchdog: stall detection, cold-start exclusion, healthy no-trip
# ---------------------------------------------------------------------------


class TestTC17Watchdog:
    """TC17: deterministic unit-level checks driven by calling check() directly,
    plus one end-to-end abort test."""

    # --- (b) Cold buffer: must NOT trip even with backlog growing and grad stalled ---

    def test_cold_buffer_never_trips(self):
        """With warm=False the stall streak cannot grow; patience never reached."""
        wd = LearnerWatchdog(patience=2, interval_s=0.1)

        # First call: establishes baseline (cannot trip).
        assert not wd.check(received=0, grad_step=0, warm=False)

        # Subsequent calls: received grows, grad stalls, but warm=False so no trip.
        for received in range(1, 10):
            result = wd.check(received=received, grad_step=0, warm=False)
            assert not result, (
                f"watchdog tripped on cold buffer at received={received}"
            )

        assert wd.error is None
        assert not wd.abort_event.is_set()

    # --- (c) Healthy: grad advancing prevents trip even with warm=True ---

    def test_grad_advancing_never_trips(self):
        """With grad_step increasing each sample, the watchdog must not trip."""
        wd = LearnerWatchdog(patience=2, interval_s=0.1)

        # Baseline.
        assert not wd.check(received=1, grad_step=0, warm=True)

        for i in range(1, 10):
            result = wd.check(received=i + 1, grad_step=i, warm=True)
            assert not result, (
                f"watchdog tripped while grad_step was advancing at i={i}"
            )

        assert wd.error is None

    # --- (a) WARM stall: must trip after patience consecutive bad samples ---

    def test_warm_stall_trips_after_patience(self):
        """With warm=True, backlog growing, grad stalled: trips after patience samples."""
        patience = 3
        wd = LearnerWatchdog(patience=patience, interval_s=0.1)

        # First call: baseline only.
        assert not wd.check(received=1, grad_step=0, warm=True)

        # Each subsequent call: received grows, grad stays 0, buffer warm.
        for i in range(1, patience):
            result = wd.check(received=i + 1, grad_step=0, warm=True)
            assert not result, f"watchdog tripped early at streak sample {i}"

        # The patience-th stall sample should trip.
        tripped = wd.check(received=patience + 1, grad_step=0, warm=True)
        assert tripped, "watchdog did not trip after patience consecutive stall samples"
        assert wd.error is not None
        assert isinstance(wd.error, WatchdogError)
        assert wd.abort_event.is_set()

    def test_watchdog_does_not_trip_one_sample_before_patience(self):
        """No trip until exactly patience consecutive stall samples are observed."""
        patience = 4
        wd = LearnerWatchdog(patience=patience, interval_s=0.1)

        wd.check(received=0, grad_step=0, warm=True)  # baseline

        # patience-1 stall samples after the baseline: should NOT trip.
        for i in range(1, patience):
            wd.check(received=i, grad_step=0, warm=True)
            assert not wd.abort_event.is_set(), (
                f"watchdog tripped early at streak sample {i}"
            )

    def test_non_warm_sample_resets_streak(self):
        """A warm=False sample in the middle resets the stall streak to zero."""
        patience = 2
        wd = LearnerWatchdog(patience=patience, interval_s=0.1)

        wd.check(received=0, grad_step=0, warm=True)   # baseline

        # One WARM stall.
        wd.check(received=1, grad_step=0, warm=True)
        assert not wd.abort_event.is_set()

        # COLD sample: resets streak.
        wd.check(received=2, grad_step=0, warm=False)
        assert not wd.abort_event.is_set()

        # Another warm stall: streak is back to 1 -- patience=2 requires one more.
        wd.check(received=3, grad_step=0, warm=True)
        assert not wd.abort_event.is_set(), (
            "watchdog tripped on streak reset -- cold sample did not clear the streak"
        )

    def test_watchdog_is_idempotent_after_trip(self):
        """Once tripped, every subsequent check() returns True without re-evaluating."""
        wd = LearnerWatchdog(patience=1, interval_s=0.1)

        wd.check(received=0, grad_step=0, warm=True)   # baseline

        # One warm stall trips with patience=1.
        tripped = wd.check(received=1, grad_step=0, warm=True)
        assert tripped

        # Even with 'healthy' inputs the watchdog stays tripped.
        assert wd.check(received=2, grad_step=100, warm=False)
        assert wd.check(received=3, grad_step=200, warm=True)

    def test_watchdog_reset_clears_state(self):
        """reset() clears the trip flag, error, and streak so the watchdog is reusable."""
        wd = LearnerWatchdog(patience=1, interval_s=0.1)

        wd.check(received=0, grad_step=0, warm=True)
        wd.check(received=1, grad_step=0, warm=True)
        assert wd.abort_event.is_set()

        wd.reset()
        assert not wd.abort_event.is_set()
        assert wd.error is None

        # Must behave as fresh after reset.
        assert not wd.check(received=0, grad_step=0, warm=True)

    # --- End-to-end: LearnerLoop run() aborts with WatchdogError on warm stall ---

    def test_loop_aborts_loudly_on_warm_stall(self):
        """With a no-op learn() and warm buffer, the loop raises WatchdogError."""
        cfg = _tiny_cfg(
            min_replay=1,
            weight_sync_every_k_steps=100,
            batch_size=4,
        )
        trainer = _make_trainer(cfg)
        transport = LocalTransport()
        store = WeightStore()

        # Make the buffer warm immediately: add enough transitions by pre-warming.
        # We override ready_to_learn to always return True and freeze grad_step.
        trainer.ready_to_learn = lambda: True
        original_learn = trainer.learn
        trainer.learn = lambda: None  # no-op: grad_step never advances

        # Short patience so the test finishes quickly.
        watchdog = LearnerWatchdog(patience=2, interval_s=0.001)

        errors_on_thread: List[BaseException] = []
        loop = LearnerLoop(trainer, transport, store, cfg, watchdog=watchdog)

        def _run():
            try:
                loop.run()
            except BaseException as exc:
                errors_on_thread.append(exc)

        t = threading.Thread(target=_run, daemon=True)
        t.start()

        # Keep feeding episodes so received keeps growing (backlog grows).
        feed_stop = threading.Event()

        def _feed():
            i = 0
            while not feed_stop.is_set():
                try:
                    transport.send(_make_episode(arena_id=i % 4, n_transitions=_EPISODE_K))
                except TransportError:
                    break
                i += 1
                time.sleep(0.01)

        feeder = threading.Thread(target=_feed, daemon=True)
        feeder.start()

        # Wait for the loop to abort (watchdog trips).
        t.join(timeout=15.0)
        feed_stop.set()
        feeder.join(timeout=2.0)
        transport.close()

        assert not t.is_alive(), (
            "learner loop did not abort after watchdog trip within timeout"
        )
        assert loop.error is not None, "loop.error must be set on abort"
        assert isinstance(loop.error, WatchdogError), (
            f"expected WatchdogError, got {type(loop.error).__name__}: {loop.error}"
        )
        assert len(errors_on_thread) == 1
        assert isinstance(errors_on_thread[0], WatchdogError), (
            "WatchdogError must propagate out of run() (loud abort)"
        )

    def test_loop_does_not_abort_during_cold_warmup(self):
        """Watchdog must NOT trip while the buffer is cold (warm=False blocks trips)."""
        cfg = _tiny_cfg(
            # High min_replay so the buffer STAYS cold throughout the test.
            min_replay=999_999,
            weight_sync_every_k_steps=100,
        )
        trainer = _make_trainer(cfg)
        transport = LocalTransport()
        store = WeightStore()

        watchdog = LearnerWatchdog(patience=2, interval_s=0.001)
        loop = LearnerLoop(trainer, transport, store, cfg, watchdog=watchdog)

        errors_on_thread: List[BaseException] = []

        def _run():
            try:
                loop.run()
            except BaseException as exc:
                errors_on_thread.append(exc)

        t = threading.Thread(target=_run, daemon=True)
        t.start()

        # Feed several episodes; grad_step stays 0 (buffer cold); watchdog must NOT trip.
        for i in range(12):
            transport.send(_make_episode(arena_id=i % 3, n_transitions=_EPISODE_K))
            time.sleep(0.02)

        # Give the loop time to drain all of them.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if loop.received >= 12:
                break
            time.sleep(0.05)

        # Stop cleanly.
        loop.stop()
        transport.close()
        t.join(timeout=5.0)
        assert not t.is_alive(), "learner loop hung after stop+close"

        assert not watchdog.abort_event.is_set(), (
            "watchdog tripped during cold warm-up (must not trip when warm=False)"
        )
        # No WatchdogError on the thread.
        watchdog_errors = [e for e in errors_on_thread if isinstance(e, WatchdogError)]
        assert not watchdog_errors, (
            f"WatchdogError raised during cold warm-up: {watchdog_errors}"
        )
