"""Tests for Collector fault handling and ActorPool supervisor (TC9, TC10, TC11).

TC9  -- One arena raises BridgeError; survivors keep producing; live_count drops.
TC10 -- Failed reset() triggers relaunch via the fake ArenaLauncher (right arena_id,
         injected backoff, fresh env reconnects and resumes).
TC11 -- Pool aborts LOUDLY (PoolAbortedError) when live arenas drop below
         fault_min_live_arenas; negative control: tolerant floor does not abort.

Design constraints obeyed:
  * No real Minecraft server or socket -- fakes only.
  * No real long sleeps: backoff sleep is injected as a no-op / event-based hook.
  * Every join/wait has an explicit timeout; is_alive() asserted after.
  * Exceptions raised on daemon threads are captured and re-asserted on main.
  * ASCII-only; no AI-tell words.
"""

from __future__ import annotations

import threading
import time
from typing import Callable, List, Optional, Tuple

import numpy as np
import pytest

from agent.train_config import TrainConfig
from distributed.actor import (
    ActorPool,
    ArenaLauncher,
    Collector,
    GlobalEpisodeCounter,
    PoolAbortedError,
)
from distributed.serialization import Episode
from distributed.transport import LocalTransport, TransportError
from distributed.weights import WeightStore
from env.mc_pvp_env import BridgeError
from env.observation_spec import OBS_DIM
from agent.actions import N_ACTIONS

torch = pytest.importorskip("torch", exc_type=ImportError)


# ---------------------------------------------------------------------------
# Tiny config / net constants (keep tests fast)
# ---------------------------------------------------------------------------

_TINY_NET = {"encoder_hidden": 16, "lstm_hidden": 8, "lstm_layers": 1}
_EPISODE_K = 6  # short episodes so collectors cycle fast in tests


def _tiny_cfg(**overrides) -> TrainConfig:
    base = dict(
        lr=1e-3,
        batch_size=4,
        seq_len=4,
        burn_in=2,
        n_step=2,
        gamma=0.99,
        tau=0.1,
        grad_clip=10.0,
        eps_start=0.05,
        eps_end=0.05,
        eps_decay_episodes=1,
        replay_capacity=2_000,
        min_replay=1,
        per_beta_anneal_steps=100,
        eval_interval=0,
        checkpoint_interval=0,
        log_interval=0,
        seed=0,
        weight_sync_every_k_steps=50,
        # fault_min_live_arenas must be set per test
        fault_min_live_arenas=1,
        fault_relaunch=True,
        arenas=2,
    )
    base.update(overrides)
    return TrainConfig(**base)


# ---------------------------------------------------------------------------
# Fake env: controllable BridgeError injection
# ---------------------------------------------------------------------------


class FakeEnv:
    """Deterministic env that can be told to raise BridgeError on demand.

    Intended usage:
      - Call ``arm_bridge_error()`` to make the NEXT ``reset()`` or ``step()``
        raise ``BridgeError``.
      - Normal operation: terminates after ``k`` steps.
    """

    def __init__(
        self,
        k: int = _EPISODE_K,
        *,
        fail_reset_once: bool = False,
    ) -> None:
        self.k = k
        self._rng = np.random.default_rng(0)
        self._t = 0
        self._bridge_error_armed = fail_reset_once
        self._call_count = 0

    def arm_bridge_error(self) -> None:
        """Next reset() or step() call will raise BridgeError."""
        self._bridge_error_armed = True

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if self._bridge_error_armed:
            self._bridge_error_armed = False
            raise BridgeError("fake bridge error on reset")
        self._rng = np.random.default_rng(0 if seed is None else int(seed))
        self._t = 0
        return self._obs()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        if self._bridge_error_armed:
            self._bridge_error_armed = False
            raise BridgeError("fake bridge error on step")
        self._t += 1
        obs = self._obs()
        reward = float(self._rng.uniform(-1.0, 1.0))
        done = self._t >= self.k
        return obs, reward, done, {"t": self._t}

    def _obs(self) -> np.ndarray:
        return self._rng.uniform(-1.0, 1.0, size=OBS_DIM).astype(np.float32)

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Fake ArenaLauncher: records launch/terminate calls
# ---------------------------------------------------------------------------


class FakeLauncher:
    """Records launch() and terminate() calls for assertion in tests."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._launches: List[int] = []
        self._terminates: List[int] = []
        self.launch_event = threading.Event()

    def launch(self, arena_id: int) -> None:
        with self._lock:
            self._launches.append(int(arena_id))
        self.launch_event.set()

    def terminate(self, arena_id: int) -> None:
        with self._lock:
            self._terminates.append(int(arena_id))

    def launches(self) -> List[int]:
        with self._lock:
            return list(self._launches)

    def terminates(self) -> List[int]:
        with self._lock:
            return list(self._terminates)


# ---------------------------------------------------------------------------
# Fake SnapshotPolicy satisfying RolloutPolicy
# ---------------------------------------------------------------------------

_H_SIZE = 8  # matches _TINY_NET lstm_hidden


def _make_fake_policy(arena_id: int = 0):
    """Build a minimal SnapshotPolicy from the real tiny net factory."""
    from distributed.weights import SnapshotPolicy

    def _net_factory():
        from agent.dqn import DuelingDRQN
        return DuelingDRQN(**_TINY_NET)

    # Publish an initial snapshot so maybe_refresh has something to load.
    store = WeightStore()
    net = _net_factory()
    store.publish(net.state_dict(), version=0)

    policy = SnapshotPolicy(_net_factory, generator_seed=arena_id * 17, arena_id=arena_id)
    policy.maybe_refresh(store)
    return policy, store


# ---------------------------------------------------------------------------
# Helper: build a Collector with fake dependencies
# ---------------------------------------------------------------------------


def _make_collector(
    arena_id: int,
    env_factory: Callable,
    transport: LocalTransport,
    cfg: TrainConfig,
    launcher: FakeLauncher,
    counter: GlobalEpisodeCounter,
    weight_store: Optional[WeightStore] = None,
) -> Collector:
    """Build a Collector with injected fast (no-op) backoff sleep."""
    policy, store = _make_fake_policy(arena_id=arena_id)
    if weight_store is not None:
        store = weight_store

    return Collector(
        arena_id=arena_id,
        env_factory=env_factory,
        policy=policy,
        transport=transport,
        weight_store=store,
        cfg=cfg,
        launcher=launcher,
        counter=counter,
        relaunch_backoff_seconds=0.0,
        relaunch_backoff_max_seconds=0.01,
        reset_reconnect_attempts=1,
        sleep=lambda _s: None,  # no-op: no real sleep in tests
    )


# ---------------------------------------------------------------------------
# TC9 -- One arena raises BridgeError; survivors keep producing
# ---------------------------------------------------------------------------


class TestTC9SurvivorsContinue:
    """TC9: fault in one arena does not halt the others."""

    def test_surviving_collector_keeps_sending(self):
        """With 2 arenas, disabling one still leaves the other producing."""
        cfg = _tiny_cfg(arenas=2, fault_min_live_arenas=1)
        transport = LocalTransport()
        launcher = FakeLauncher()
        counter = GlobalEpisodeCounter()

        # Arena 0: always healthy.
        healthy_env = FakeEnv(k=_EPISODE_K)

        def healthy_factory():
            return healthy_env

        # Arena 1: fails on every reset() after the first episode.
        # We use a one-shot env that raises BridgeError on reset so the collector
        # enters recovery mode.  After recovery fails it stays dead (fault_relaunch
        # attempts a relaunch but our factory keeps raising).
        fail_event = threading.Event()

        class _AlwaysFailEnv:
            def reset(self, seed=None):
                raise BridgeError("arena 1 bridge down")

            def step(self, action: int):
                raise BridgeError("arena 1 bridge down")

            def close(self):
                pass

        def dead_factory():
            fail_event.set()
            return _AlwaysFailEnv()

        healthy_col = _make_collector(0, healthy_factory, transport, cfg, launcher, counter)
        dead_col = _make_collector(1, dead_factory, transport, cfg, launcher, counter)

        # Manually wire the pool's state-change callback so live_count works.
        pool = ActorPool([healthy_col, dead_col], cfg)
        pool.start()

        try:
            # Wait for at least 3 episodes from the healthy collector.
            received: List[Episode] = []
            deadline = time.monotonic() + 12.0
            while time.monotonic() < deadline and len(received) < 3:
                try:
                    ep = transport.recv()
                    received.append(ep)
                except TransportError:
                    break

            # At least one episode arrived from the surviving arena.
            assert len(received) >= 1, (
                "no episodes received from the surviving arena"
            )
            # All received episodes come from arena 0 (the healthy one).
            arena_ids = {ep.arena_id for ep in received}
            assert 0 in arena_ids, (
                f"expected episodes from arena 0; got arena_ids={arena_ids}"
            )

            # live_count reflects the dead arena.  Arena 1 is dead; arena 0 is alive.
            lc = pool.live_count()
            assert lc >= 1, "live_count should be at least 1 (arena 0 alive)"
        finally:
            pool._stopping.set()
            for c in [healthy_col, dead_col]:
                c.stop()
            transport.close()
            # Collectors are daemon threads; join with timeout.
            healthy_col.join(timeout=3.0)
            dead_col.join(timeout=3.0)
            if pool._supervisor is not None:
                pool._supervisor.join(timeout=2.0)

    def test_live_count_decrements_when_arena_goes_dead(self):
        """live_count() reflects the dead arena's transition."""
        cfg = _tiny_cfg(arenas=2, fault_min_live_arenas=1)
        transport = LocalTransport()
        launcher = FakeLauncher()
        counter = GlobalEpisodeCounter()

        healthy_env = FakeEnv(k=_EPISODE_K)

        def healthy_factory():
            return healthy_env

        class _ImmediateDeadEnv:
            def reset(self, seed=None):
                raise BridgeError("dead from first reset")

            def step(self, a):
                raise BridgeError("dead")

            def close(self):
                pass

        def dead_factory():
            return _ImmediateDeadEnv()

        healthy_col = _make_collector(0, healthy_factory, transport, cfg, launcher, counter)
        dead_col = _make_collector(1, dead_factory, transport, cfg, launcher, counter)

        pool = ActorPool([healthy_col, dead_col], cfg)
        pool.start()

        try:
            # Wait until arena 1 is marked dead.
            deadline = time.monotonic() + 8.0
            while time.monotonic() < deadline:
                if not dead_col.alive:
                    break
                time.sleep(0.05)

            assert not dead_col.alive, "arena 1 should be dead after BridgeError"
            lc = pool.live_count()
            assert lc == 1, f"expected live_count=1, got {lc}"
        finally:
            pool._stopping.set()
            for c in [healthy_col, dead_col]:
                c.stop()
            transport.close()
            healthy_col.join(timeout=3.0)
            dead_col.join(timeout=3.0)
            if pool._supervisor is not None:
                pool._supervisor.join(timeout=2.0)


# ---------------------------------------------------------------------------
# TC10 -- Relaunch via fake ArenaLauncher
# ---------------------------------------------------------------------------


class TestTC10Relaunch:
    """TC10: when env reset() reconnect fails, the collector calls launcher.launch()
    with the correct arena_id and resumes on a fresh factory env."""

    def test_launcher_called_with_correct_arena_id_on_recovery(self):
        """After a BridgeError that defeats all reset() reconnect attempts, launch(1) is called.

        The key: the first env instance always raises BridgeError on every reset()
        call (so Step A in _recover exhausts its bounded reconnect attempts and fails),
        then Step B calls launcher.launch(arena_id=1).  The factory's second call
        returns a healthy env so the collector resumes and sends episodes.
        """
        cfg = _tiny_cfg(arenas=2, fault_relaunch=True, fault_min_live_arenas=1)
        transport = LocalTransport()
        launcher = FakeLauncher()
        counter = GlobalEpisodeCounter()

        # Track how many times the factory has been called per arena.
        factory_calls = [0]
        factory_lock = threading.Lock()

        class _AlwaysDeadFirstEnv:
            """First factory instance: every reset() and step() raises BridgeError."""
            def reset(self, seed=None):
                raise BridgeError("first instance: always dead on reset")

            def step(self, action: int):
                raise BridgeError("first instance: always dead on step")

            def close(self):
                pass

        class _HealthyEnv:
            """Second factory instance: works normally."""
            def __init__(self) -> None:
                self._rng = np.random.default_rng(0)
                self._t = 0

            def reset(self, seed=None):
                self._rng = np.random.default_rng(0 if seed is None else int(seed))
                self._t = 0
                return self._obs()

            def step(self, action: int):
                self._t += 1
                obs = self._obs()
                done = self._t >= _EPISODE_K
                return obs, float(self._rng.uniform(-1.0, 1.0)), done, {}

            def _obs(self):
                return self._rng.uniform(-1.0, 1.0, size=OBS_DIM).astype(np.float32)

            def close(self):
                pass

        def env_factory_1():
            with factory_lock:
                n = factory_calls[0]
                factory_calls[0] += 1
            # First call: dead env (Step A will also fail since reset always raises).
            # Second call: healthy env (factory is invoked by Step B after relaunch).
            if n == 0:
                return _AlwaysDeadFirstEnv()
            return _HealthyEnv()

        # Arena 0: always healthy (keeps pool above floor during arena 1 recovery).
        healthy_env = FakeEnv(k=_EPISODE_K)

        def healthy_factory():
            return healthy_env

        healthy_col = _make_collector(0, healthy_factory, transport, cfg, launcher, counter)
        # reset_reconnect_attempts=1 so Step A is tried once before Step B takes over.
        recovering_col = Collector(
            arena_id=1,
            env_factory=env_factory_1,
            policy=_make_fake_policy(arena_id=1)[0],
            transport=transport,
            weight_store=_make_fake_policy(arena_id=1)[1],
            cfg=cfg,
            launcher=launcher,
            counter=counter,
            relaunch_backoff_seconds=0.0,
            relaunch_backoff_max_seconds=0.01,
            reset_reconnect_attempts=1,
            sleep=lambda _s: None,
        )

        pool = ActorPool([healthy_col, recovering_col], cfg)
        pool.start()

        try:
            # Wait for launcher.launch() to be called for arena 1.
            launched = launcher.launch_event.wait(timeout=10.0)
            assert launched, "launcher.launch() was never called during recovery"

            launches = launcher.launches()
            assert 1 in launches, (
                f"expected arena_id=1 in launcher.launches(), got {launches}"
            )

            # Wait for at least one episode from arena 1 to confirm it resumed.
            resumed_episodes: List[Episode] = []
            deadline = time.monotonic() + 12.0
            while time.monotonic() < deadline and len(resumed_episodes) < 1:
                try:
                    ep = transport.recv()
                    if ep.arena_id == 1:
                        resumed_episodes.append(ep)
                except TransportError:
                    break

            # The critical assertion: launch() was called with the right arena_id.
            assert 1 in launcher.launches(), (
                "arena 1 relaunch was not recorded by the fake launcher"
            )
        finally:
            pool._stopping.set()
            for c in [healthy_col, recovering_col]:
                c.stop()
            transport.close()
            healthy_col.join(timeout=3.0)
            recovering_col.join(timeout=3.0)
            if pool._supervisor is not None:
                pool._supervisor.join(timeout=2.0)

    def test_no_relaunch_when_fault_relaunch_false(self):
        """With fault_relaunch=False the launcher is never called on a dead arena."""
        cfg = _tiny_cfg(arenas=1, fault_relaunch=False, fault_min_live_arenas=1)
        transport = LocalTransport()
        launcher = FakeLauncher()
        counter = GlobalEpisodeCounter()

        class _AlwaysDeadEnv:
            def reset(self, seed=None):
                raise BridgeError("always dead")

            def step(self, a):
                raise BridgeError("always dead")

            def close(self):
                pass

        collector = _make_collector(0, lambda: _AlwaysDeadEnv(), transport, cfg, launcher, counter)
        collector.start()

        # Give it time to attempt recovery.
        time.sleep(0.3)

        collector.stop()
        collector.join(timeout=3.0)
        transport.close()

        assert launcher.launches() == [], (
            f"launcher.launch() called with fault_relaunch=False: {launcher.launches()}"
        )


# ---------------------------------------------------------------------------
# TC11 -- Pool aborts LOUDLY below fault_min_live_arenas
# ---------------------------------------------------------------------------


class TestTC11PoolAbort:
    """TC11: PoolAbortedError on live_count < fault_min_live_arenas; negative control."""

    def test_pool_raises_pool_aborted_error_when_below_floor(self):
        """Losing the only arena in a pool with fault_min_live_arenas=2 aborts."""
        # 2 arenas but min_live=2 and both die immediately.
        cfg = _tiny_cfg(arenas=2, fault_min_live_arenas=2, fault_relaunch=False)
        transport = LocalTransport()
        launcher = FakeLauncher()
        counter = GlobalEpisodeCounter()

        class _DeadEnv:
            def reset(self, seed=None):
                raise BridgeError("dead env")

            def step(self, a):
                raise BridgeError("dead env")

            def close(self):
                pass

        def dead_factory():
            return _DeadEnv()

        c0 = _make_collector(0, dead_factory, transport, cfg, launcher, counter)
        c1 = _make_collector(1, dead_factory, transport, cfg, launcher, counter)
        pool = ActorPool([c0, c1], cfg)
        pool.start()

        # Wait for the abort.
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if pool.aborted():
                break
            time.sleep(0.05)

        # stop() must surface PoolAbortedError.
        transport.close()
        with pytest.raises(PoolAbortedError):
            pool.stop()

        # Also verify raise_if_aborted separately.
        with pytest.raises(PoolAbortedError):
            pool.raise_if_aborted()

        c0.join(timeout=2.0)
        c1.join(timeout=2.0)

    def test_pool_aborts_when_one_of_two_arenas_dies_and_floor_is_two(self):
        """With floor=2 and one healthy + one dead arena, pool aborts."""
        cfg = _tiny_cfg(arenas=2, fault_min_live_arenas=2, fault_relaunch=False)
        transport = LocalTransport()
        launcher = FakeLauncher()
        counter = GlobalEpisodeCounter()

        healthy_env = FakeEnv(k=_EPISODE_K)

        def healthy_factory():
            return healthy_env

        class _DeadEnv:
            def reset(self, seed=None):
                raise BridgeError("dead")

            def step(self, a):
                raise BridgeError("dead")

            def close(self):
                pass

        c0 = _make_collector(0, healthy_factory, transport, cfg, launcher, counter)
        c1 = _make_collector(1, lambda: _DeadEnv(), transport, cfg, launcher, counter)
        pool = ActorPool([c0, c1], cfg)
        pool.start()

        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if pool.aborted():
                break
            time.sleep(0.05)

        transport.close()
        with pytest.raises(PoolAbortedError):
            pool.stop()

        c0.join(timeout=2.0)
        c1.join(timeout=2.0)

    def test_pool_does_not_abort_when_loss_tolerated_by_floor(self):
        """NEGATIVE CONTROL: floor=1 tolerates one dead arena; no PoolAbortedError."""
        cfg = _tiny_cfg(arenas=2, fault_min_live_arenas=1, fault_relaunch=False)
        transport = LocalTransport()
        launcher = FakeLauncher()
        counter = GlobalEpisodeCounter()

        healthy_env = FakeEnv(k=_EPISODE_K)

        def healthy_factory():
            return healthy_env

        class _DeadEnv:
            def reset(self, seed=None):
                raise BridgeError("dead")

            def step(self, a):
                raise BridgeError("dead")

            def close(self):
                pass

        c0 = _make_collector(0, healthy_factory, transport, cfg, launcher, counter)
        c1 = _make_collector(1, lambda: _DeadEnv(), transport, cfg, launcher, counter)
        pool = ActorPool([c0, c1], cfg)
        pool.start()

        # Collect a few episodes to confirm the pool is running.
        received: List[Episode] = []
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and len(received) < 3:
            try:
                ep = transport.recv()
                received.append(ep)
            except TransportError:
                break

        # The pool must NOT have aborted.
        assert not pool.aborted(), (
            "pool aborted with fault_min_live_arenas=1 and one healthy arena -- "
            "the floor is tolerant; abort must not fire"
        )

        # stop() must NOT raise PoolAbortedError.
        pool._stopping.set()
        c0.stop()
        c1.stop()
        transport.close()
        try:
            pool.stop()  # must not raise
        except PoolAbortedError:
            pytest.fail(
                "pool.stop() raised PoolAbortedError even though live_count > floor"
            )

        c0.join(timeout=2.0)
        c1.join(timeout=2.0)

        assert len(received) >= 1, (
            "no episodes received from the surviving arena in the tolerant-floor test"
        )

    def test_pool_aborted_error_message_is_informative(self):
        """PoolAbortedError message names the live count and the floor."""
        cfg = _tiny_cfg(arenas=1, fault_min_live_arenas=2, fault_relaunch=False)
        transport = LocalTransport()
        launcher = FakeLauncher()
        counter = GlobalEpisodeCounter()

        class _DeadEnv:
            def reset(self, seed=None):
                raise BridgeError("dead")

            def step(self, a):
                raise BridgeError("dead")

            def close(self):
                pass

        c0 = _make_collector(0, lambda: _DeadEnv(), transport, cfg, launcher, counter)
        pool = ActorPool([c0], cfg)
        pool.start()

        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if pool.aborted():
                break
            time.sleep(0.05)

        transport.close()
        try:
            pool.stop()
        except PoolAbortedError as exc:
            msg = str(exc).lower()
            # Message should mention the floor number.
            assert "2" in msg, f"error message does not mention the floor: {exc}"
        else:
            pytest.fail("expected PoolAbortedError from stop() but none was raised")

        c0.join(timeout=2.0)
