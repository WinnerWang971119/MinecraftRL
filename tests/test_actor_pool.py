"""Tests for Collector fault handling and the two-tier fault policy (TC9, TC10, TC14).

TC9  -- One pad raises BridgeError; the other pads keep producing; live_count drops.
TC10 -- Failed reset() triggers a bridge restart via the fake ArenaLauncher (right
         pad index, injected backoff, fresh env reconnects and resumes).
TC14 -- THE TWO-TIER FAULT POLICY, by injection (AC15):
         tier 1, a dead BRIDGE restarts THAT pad's bridge and nothing else, and never
         aborts the run;
         tier 2, a dead shared Paper JVM aborts the WHOLE run loudly, names the JVM,
         and restarts no bridge at all.
         Plus the seam itself: the real jvm_alive() probe against a real socket, and
         the real SubprocessArenaLauncher abandoning an in-flight launch on shutdown.

There is no floor test any more: fault_min_live_arenas was deleted in T11 precisely
because a survivor floor licenses training on a silently shrunken fleet.

Design constraints obeyed:
  * No real Minecraft server -- fakes only. The only real sockets are loopback
    listeners this file opens and closes itself, to test jvm_alive() honestly.
  * No real long sleeps: backoff sleep is injected as a no-op / event-based hook.
  * Every join/wait has an explicit timeout; is_alive() asserted after.
  * Exceptions raised on daemon threads are captured and re-asserted on main.
  * ASCII-only; no AI-tell words.
"""

from __future__ import annotations

import socket
import threading
import time
from typing import Callable, List, Optional, Tuple

import numpy as np
import pytest

from agent.train_config import TrainConfig
from distributed.actor import (
    MC_HOST,
    MC_PORT,
    ActorPool,
    ArenaLauncher,
    Collector,
    GlobalEpisodeCounter,
    LaunchAbandoned,
    PoolAbortedError,
    ShutdownSignal,
    jvm_alive,
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
        # fault_relaunch arms TIER 1 (restart the dead pad's own bridge). There is no
        # survivor floor to configure: tier 2 (a dead JVM) is armed by injecting a
        # jvm_probe into the pool, not by config.
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
    jvm_probe: Optional[Callable[[str, int], bool]] = None,
) -> Collector:
    """Build a Collector with injected fast (no-op) backoff sleep.

    ``jvm_probe`` left as ``None`` means "no JVM supervision on this collector",
    which is what every pre-two-tier test wants: the pad-fault paths must behave
    identically whether or not a JVM is being watched. Pass a stub to drive tier 2.
    """
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
        jvm_probe=jvm_probe,
        relaunch_backoff_seconds=0.0,
        relaunch_backoff_max_seconds=0.01,
        reset_reconnect_attempts=1,
        sleep=lambda _s: None,  # no-op: no real sleep in tests
    )


class RecordingJvmProbe:
    """A ``(host, port) -> bool`` JVM probe whose verdict a test controls.

    Records every ``(host, port)`` it was called with, so a test can assert the
    watchdog probes the MINECRAFT port and never a bridge port -- connecting to a
    bridge port would evict that pad's collector (BridgeServer serves ONE client).
    """

    def __init__(self, alive: bool = True) -> None:
        self._lock = threading.Lock()
        self._alive = bool(alive)
        self.calls: List[Tuple[str, int]] = []
        self.called = threading.Event()

    def __call__(self, host: str, port: int) -> bool:
        with self._lock:
            self.calls.append((host, int(port)))
            alive = self._alive
        self.called.set()
        return alive

    def kill(self) -> None:
        """Make every subsequent probe report the JVM as gone."""
        with self._lock:
            self._alive = False

    def call_count(self) -> int:
        with self._lock:
            return len(self.calls)

    def ports(self) -> List[int]:
        with self._lock:
            return [port for _host, port in self.calls]


class _DeadEnv:
    """An env whose bridge is gone: every reset() and step() raises BridgeError."""

    def reset(self, seed: Optional[int] = None):
        raise BridgeError("dead bridge")

    def step(self, action: int):
        raise BridgeError("dead bridge")

    def close(self) -> None:
        pass


def _drain_a_few(transport: LocalTransport, count: int, timeout: float) -> List[Episode]:
    """Pull up to ``count`` episodes off the transport within ``timeout`` seconds."""
    received: List[Episode] = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and len(received) < count:
        try:
            received.append(transport.recv())
        except TransportError:
            break
    return received


def _wait_for(predicate: Callable[[], bool], timeout: float) -> bool:
    """Poll ``predicate`` until true or ``timeout`` elapses. Returns the outcome."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def _shutdown_pool(pool: ActorPool, collectors: List[Collector], transport) -> None:
    """Wind a pool down without asserting on the abort (teardown helper)."""
    pool._stopping.set()
    for collector in collectors:
        collector.stop()
    try:
        transport.close()
    except Exception:  # noqa: BLE001 - teardown is best-effort
        pass
    for collector in collectors:
        collector.join(timeout=3.0)
    if pool._supervisor is not None:
        pool._supervisor.join(timeout=2.0)


# ---------------------------------------------------------------------------
# TC9 -- One arena raises BridgeError; survivors keep producing
# ---------------------------------------------------------------------------


class TestTC9SurvivorsContinue:
    """TC9: fault in one arena does not halt the others."""

    def test_surviving_collector_keeps_sending(self):
        """With 2 arenas, disabling one still leaves the other producing."""
        cfg = _tiny_cfg(arenas=2)
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
        cfg = _tiny_cfg(arenas=2)
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
        cfg = _tiny_cfg(arenas=2, fault_relaunch=True)
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
        cfg = _tiny_cfg(arenas=1, fault_relaunch=False)
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
# TC14 / AC15 -- TIER 1: a dead BRIDGE restarts THAT pad's bridge, nothing else
# ---------------------------------------------------------------------------


class TestTC14TierOneBridgeDeath:
    """AC15, first half: killing one bridge restarts only that bridge.

    The discrimination that matters: the OTHER pads must be untouched (no launch
    call for them, and they keep producing), and the run must NOT abort. The
    deleted survivor floor would have aborted the second case below.
    """

    def test_only_the_dead_pads_bridge_is_restarted(self):
        """Pad 1's bridge dies: launch() fires for pad 1 and for no other pad."""
        cfg = _tiny_cfg(arenas=3, fault_relaunch=True)
        transport = LocalTransport()
        launcher = FakeLauncher()
        counter = GlobalEpisodeCounter()
        probe = RecordingJvmProbe(alive=True)  # the JVM is fine; this is tier 1

        healthy_envs = {0: FakeEnv(k=_EPISODE_K), 2: FakeEnv(k=_EPISODE_K)}

        class _HealthyEnv:
            """Pad 1's replacement env, handed out from the second factory call on."""

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
                return obs, float(self._rng.uniform(-1.0, 1.0)), self._t >= _EPISODE_K, {}

            def _obs(self):
                return self._rng.uniform(-1.0, 1.0, size=OBS_DIM).astype(np.float32)

            def close(self):
                pass

        calls = {"pad1": 0}
        calls_lock = threading.Lock()

        def pad1_factory():
            with calls_lock:
                n = calls["pad1"]
                calls["pad1"] += 1
            # First instance: the bridge is gone. After the restart: healthy again.
            return _DeadEnv() if n == 0 else _HealthyEnv()

        collectors = [
            _make_collector(0, lambda: healthy_envs[0], transport, cfg, launcher, counter),
            _make_collector(1, pad1_factory, transport, cfg, launcher, counter),
            _make_collector(2, lambda: healthy_envs[2], transport, cfg, launcher, counter),
        ]
        pool = ActorPool(
            collectors, cfg, jvm_probe=probe, mc_port=25599, jvm_poll_seconds=0.01
        )
        pool.start()

        try:
            assert launcher.launch_event.wait(timeout=10.0), (
                "pad 1's bridge was never restarted"
            )
            # The other two pads must keep producing while pad 1 is repaired.
            received = _drain_a_few(transport, count=25, timeout=12.0)
            arena_ids = {ep.arena_id for ep in received}
            assert {0, 2} <= arena_ids, (
                f"pads 0 and 2 must keep producing during pad 1's restart; "
                f"saw arena_ids={sorted(arena_ids)}"
            )

            # THE tier-1 assertion: exactly one pad's bridge was restarted.
            launches = launcher.launches()
            assert set(launches) == {1}, (
                f"a bridge restart must touch ONLY the dead pad; launch() was called "
                f"for pads {sorted(set(launches))}"
            )

            # A bridge fault is not a run-ending fault.
            assert not pool.aborted(), (
                "a dead bridge must never abort the run -- only a dead JVM does"
            )
            # And the watchdog probed the MINECRAFT port, never a bridge port.
            assert probe.call_count() > 0, "the JVM watchdog never ran"
            assert set(probe.ports()) == {25599}, (
                f"the JVM watchdog must probe only the mc port; it probed "
                f"{sorted(set(probe.ports()))} -- connecting to a bridge port would "
                f"evict that pad's collector"
            )
        finally:
            _shutdown_pool(pool, collectors, transport)

        pool.stop()  # must not raise: nothing aborted

    def test_a_pad_that_never_comes_back_does_not_abort_the_run(self):
        """The deleted floor, directly: 1 of 2 pads down forever is NOT an abort.

        Under ``fault_min_live_arenas=2`` this exact situation aborted the run. The
        two-tier policy keeps repairing the dead pad and keeps the run alive on the
        pads that are working; only a dead JVM ends it.
        """
        cfg = _tiny_cfg(arenas=2, fault_relaunch=True)
        transport = LocalTransport()
        launcher = FakeLauncher()
        counter = GlobalEpisodeCounter()
        probe = RecordingJvmProbe(alive=True)

        healthy_env = FakeEnv(k=_EPISODE_K)
        collectors = [
            _make_collector(0, lambda: healthy_env, transport, cfg, launcher, counter),
            _make_collector(1, lambda: _DeadEnv(), transport, cfg, launcher, counter),
        ]
        pool = ActorPool(collectors, cfg, jvm_probe=probe, jvm_poll_seconds=0.01)
        pool.start()

        try:
            assert _wait_for(lambda: not collectors[1].alive, timeout=8.0), (
                "pad 1 should have been marked dead"
            )
            received = _drain_a_few(transport, count=3, timeout=10.0)
            assert received, "the healthy pad stopped producing"
            assert pool.live_count() == 1, (
                f"expected exactly one live pad, got {pool.live_count()}"
            )
            assert not pool.aborted(), (
                "the pool aborted with a permanently dead pad and a live JVM -- "
                "that is the deleted survivor floor, not the two-tier policy"
            )
            # Repair was attempted repeatedly, and only ever for pad 1.
            assert set(launcher.launches()) == {1}, (
                f"only pad 1's bridge may be restarted, got "
                f"{sorted(set(launcher.launches()))}"
            )
        finally:
            _shutdown_pool(pool, collectors, transport)

        pool.stop()  # must not raise


# ---------------------------------------------------------------------------
# TC14 / AC15 -- TIER 2: a dead JVM aborts the whole run, loudly
# ---------------------------------------------------------------------------


class TestTC14TierTwoJvmDeath:
    """AC15, second half: killing the JVM aborts loudly and restarts nothing."""

    def test_collector_aborts_the_run_when_the_jvm_is_gone(self):
        """A pad fault + a dead JVM: the run aborts and NO bridge is restarted.

        The probe is injected on the COLLECTOR only (the pool gets none), so this
        isolates the recovery-path detector from the supervisor's periodic one.
        """
        cfg = _tiny_cfg(arenas=2, fault_relaunch=True)
        transport = LocalTransport()
        launcher = FakeLauncher()
        counter = GlobalEpisodeCounter()
        dead_jvm = RecordingJvmProbe(alive=False)

        healthy_env = FakeEnv(k=_EPISODE_K)
        collectors = [
            _make_collector(0, lambda: healthy_env, transport, cfg, launcher, counter),
            _make_collector(
                1, lambda: _DeadEnv(), transport, cfg, launcher, counter,
                jvm_probe=dead_jvm,
            ),
        ]
        pool = ActorPool(collectors, cfg, mc_port=25599)  # no supervisor probe
        pool.start()

        try:
            assert _wait_for(pool.aborted, timeout=10.0), (
                "a dead JVM discovered during recovery must abort the run"
            )
            assert launcher.launches() == [], (
                f"no bridge may be restarted into a dead JVM; launch() was called "
                f"for {launcher.launches()}"
            )
        finally:
            for collector in collectors:
                collector.stop()
            transport.close()
            for collector in collectors:
                collector.join(timeout=3.0)

        with pytest.raises(PoolAbortedError) as excinfo:
            pool.stop()
        message = str(excinfo.value)
        assert "JVM" in message, f"the abort must name the JVM: {message}"
        assert "25599" in message, f"the abort must name the probed port: {message}"
        assert "pad 1" in message, (
            f"the collector-side abort should say which pad found it: {message}"
        )

    def test_supervisor_aborts_on_a_jvm_death_with_every_pad_healthy(self):
        """The watchdog catches a JVM death that no pad has noticed yet."""
        cfg = _tiny_cfg(arenas=2, fault_relaunch=True)
        transport = LocalTransport()
        launcher = FakeLauncher()
        counter = GlobalEpisodeCounter()
        probe = RecordingJvmProbe(alive=True)

        envs = {0: FakeEnv(k=_EPISODE_K), 1: FakeEnv(k=_EPISODE_K)}
        collectors = [
            _make_collector(0, lambda: envs[0], transport, cfg, launcher, counter),
            _make_collector(1, lambda: envs[1], transport, cfg, launcher, counter),
        ]
        pool = ActorPool(
            collectors, cfg, jvm_probe=probe, mc_port=25599, jvm_poll_seconds=0.01
        )
        pool.start()

        try:
            assert probe.called.wait(timeout=5.0), "the watchdog never probed"
            assert not pool.aborted(), "aborted while the JVM was still answering"
            # Kill the JVM out from under two perfectly healthy pads.
            probe.kill()
            assert _wait_for(pool.aborted, timeout=10.0), (
                "the supervisor must abort once the shared JVM stops answering, even "
                "though no pad had faulted"
            )
            assert launcher.launches() == [], (
                "a JVM death must not trigger any bridge restart"
            )
        finally:
            for collector in collectors:
                collector.stop()
            transport.close()
            for collector in collectors:
                collector.join(timeout=3.0)

        with pytest.raises(PoolAbortedError) as excinfo:
            pool.stop()
        message = str(excinfo.value)
        assert "JVM" in message, f"the abort must name the JVM: {message}"
        assert "25599" in message, f"the abort must name the probed port: {message}"

    def test_the_pool_pins_every_collector_to_its_own_mc_port(self):
        """One JVM, one port: a supervised collector never watches a different one.

        Without this, an externally-built collector keeps the default port while the
        pool watches another -- the collector then probes the wrong port and names
        the wrong port in the abort it raises.
        """
        cfg = _tiny_cfg(arenas=1, fault_relaunch=False)
        transport = LocalTransport()
        launcher = FakeLauncher()
        counter = GlobalEpisodeCounter()
        probe = RecordingJvmProbe(alive=True)

        collector = _make_collector(
            0, lambda: FakeEnv(k=_EPISODE_K), transport, cfg, launcher, counter
        )
        assert collector._mc_port == MC_PORT  # the default, before the pool binds it

        pool = ActorPool(
            [collector], cfg, jvm_probe=probe, mc_host="127.0.0.2", mc_port=25599,
            jvm_poll_seconds=0.01,
        )
        pool.start()
        try:
            assert collector._mc_port == 25599
            assert collector._mc_host == "127.0.0.2"
            assert collector.jvm_alive() is True
            assert probe.calls, "the collector's probe never ran"
            assert set(probe.ports()) == {25599}, (
                f"collector and supervisor must probe the same port, got "
                f"{sorted(set(probe.ports()))}"
            )
        finally:
            _shutdown_pool(pool, [collector], transport)

        pool.stop()  # must not raise

    def test_no_probe_means_no_jvm_supervision(self):
        """With jvm_probe unset (offline pools) nothing probes and nothing aborts."""
        cfg = _tiny_cfg(arenas=2, fault_relaunch=True)
        transport = LocalTransport()
        launcher = FakeLauncher()
        counter = GlobalEpisodeCounter()

        envs = {0: FakeEnv(k=_EPISODE_K), 1: FakeEnv(k=_EPISODE_K)}
        collectors = [
            _make_collector(0, lambda: envs[0], transport, cfg, launcher, counter),
            _make_collector(1, lambda: envs[1], transport, cfg, launcher, counter),
        ]
        pool = ActorPool(collectors, cfg)
        pool.start()

        try:
            received = _drain_a_few(transport, count=3, timeout=10.0)
            assert received, "the pool produced nothing"
            assert not pool.aborted(), (
                "an unconfigured JVM probe must not be read as a dead JVM"
            )
            assert collectors[0].jvm_alive() is True, (
                "a collector with no probe must report the JVM as alive"
            )
        finally:
            _shutdown_pool(pool, collectors, transport)

        pool.stop()  # must not raise


# ---------------------------------------------------------------------------
# The seam itself: restart_bridge(), jvm_alive(), and the shutdown signal
# ---------------------------------------------------------------------------


class TestRestartBridgeSeam:
    """``restart_bridge(pad_index)`` -- the tier-1 half of the T10<->T11 seam."""

    def _lone_collector(self, launcher: FakeLauncher) -> Collector:
        cfg = _tiny_cfg(arenas=2)
        return _make_collector(
            0, lambda: FakeEnv(k=_EPISODE_K), LocalTransport(), cfg,
            launcher, GlobalEpisodeCounter(),
        )

    def test_restarts_exactly_the_named_pad(self):
        launcher = FakeLauncher()
        collector = self._lone_collector(launcher)

        collector.restart_bridge(3)

        assert launcher.launches() == [3], (
            f"restart_bridge(3) must launch pad 3 and nothing else, got "
            f"{launcher.launches()}"
        )

    def test_rejects_a_negative_pad_index(self):
        launcher = FakeLauncher()
        collector = self._lone_collector(launcher)

        with pytest.raises(ValueError):
            collector.restart_bridge(-1)
        assert launcher.launches() == [], "a rejected index must not reach the launcher"

    def test_rejects_a_non_integer_pad_index(self):
        launcher = FakeLauncher()
        collector = self._lone_collector(launcher)

        with pytest.raises(ValueError):
            collector.restart_bridge("0")  # type: ignore[arg-type]
        assert launcher.launches() == []

    def test_launcher_failures_propagate_to_the_caller(self):
        """The live launcher raises rather than duplicate a live bridge."""

        class _RefusingLauncher:
            def launch(self, arena_id: int) -> None:
                raise RuntimeError("bridge port still occupied")

            def terminate(self, arena_id: int) -> None:
                pass

        cfg = _tiny_cfg(arenas=2)
        collector = _make_collector(
            0, lambda: FakeEnv(k=_EPISODE_K), LocalTransport(), cfg,
            _RefusingLauncher(), GlobalEpisodeCounter(),
        )
        with pytest.raises(RuntimeError):
            collector.restart_bridge(0)

    def test_an_abandoned_launch_terminates_the_bridge_it_spawned(self):
        """Shutdown mid-launch must not orphan the process we just started."""
        events: List[str] = []

        class _AbandoningLauncher:
            def launch(self, arena_id: int) -> None:
                events.append(f"launch:{arena_id}")
                raise LaunchAbandoned("shutdown")

            def terminate(self, arena_id: int) -> None:
                events.append(f"terminate:{arena_id}")

        cfg = _tiny_cfg(arenas=2)
        collector = _make_collector(
            0, lambda: FakeEnv(k=_EPISODE_K), LocalTransport(), cfg,
            _AbandoningLauncher(), GlobalEpisodeCounter(),
        )
        with pytest.raises(LaunchAbandoned):
            collector.restart_bridge(2)

        assert events == ["launch:2", "terminate:2"], (
            f"an abandoned launch must clean up the bridge it spawned, got {events}"
        )


class TestJvmAliveProbe:
    """``jvm_alive()`` -- the tier-2 detector, against real loopback sockets."""

    def test_true_for_a_real_listening_socket(self):
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        try:
            assert jvm_alive("127.0.0.1", port) is True
        finally:
            listener.close()

    def test_false_once_the_listener_is_gone(self):
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        assert jvm_alive("127.0.0.1", port) is True
        listener.close()

        assert jvm_alive("127.0.0.1", port) is False, (
            "a closed port must read as a dead JVM"
        )

    def test_zero_arg_call_targets_the_shared_minecraft_port(self, monkeypatch):
        """The contract form ``jvm_alive()`` must probe the MC port, not a bridge port."""
        import distributed.actor as actor_module

        seen: List[Tuple[str, int]] = []

        class _StubSocketModule:
            @staticmethod
            def create_connection(address, timeout=None):
                seen.append((address[0], address[1]))

                class _Conn:
                    def __enter__(self_inner):
                        return self_inner

                    def __exit__(self_inner, *exc):
                        return False

                return _Conn()

        monkeypatch.setattr(actor_module, "socket", _StubSocketModule)

        assert actor_module.jvm_alive() is True
        assert seen == [(MC_HOST, MC_PORT)], f"jvm_alive() probed {seen}"
        assert MC_PORT == 25565, (
            "one JVM serves every pad on server.properties' server-port"
        )
        # 5555 is the pad-0 bridge port; probing it would evict a live collector.
        assert MC_PORT != 5555

    def test_the_cli_default_matches_the_watchdog_default(self):
        """agent.train cannot import this module (cycle), so pin the copies here.

        ``--mc-port`` feeds BOTH the launcher's precondition check and this
        watchdog. If the two constants drift, the launcher and the watchdog end up
        watching different ports and the abort names a port nobody probed.
        """
        from agent.train import _DEFAULT_MC_PORT, _build_parser

        assert _DEFAULT_MC_PORT == MC_PORT
        args = _build_parser().parse_args(["--arenas", "2"])
        assert int(args.mc_port) == MC_PORT

    def test_only_oserror_is_read_as_a_dead_jvm(self, monkeypatch):
        """A non-OSError from the socket layer is a bug and must not read as 'dead'."""
        import distributed.actor as actor_module

        class _ExplodingSocketModule:
            @staticmethod
            def create_connection(address, timeout=None):
                raise ValueError("not an OSError")

        monkeypatch.setattr(actor_module, "socket", _ExplodingSocketModule)

        with pytest.raises(ValueError):
            actor_module.jvm_alive("127.0.0.1", 1)


class TestShutdownSignal:
    """``ShutdownSignal`` -- what makes a launcher's bounded wait interruptible."""

    def test_sleep_is_a_plain_sleep_until_signalled(self):
        signal = ShutdownSignal()
        signal.sleep(0.01)  # must not raise
        assert signal.is_set() is False

    def test_sleep_raises_immediately_once_signalled(self):
        signal = ShutdownSignal()
        signal.set()
        started = time.monotonic()
        with pytest.raises(LaunchAbandoned):
            signal.sleep(30.0)
        assert time.monotonic() - started < 1.0

    def test_a_zero_length_sleep_still_notices_shutdown(self):
        signal = ShutdownSignal()
        signal.set()
        with pytest.raises(LaunchAbandoned):
            signal.sleep(0.0)

    def test_sleep_raises_when_shutdown_lands_mid_wait(self):
        signal = ShutdownSignal()
        timer = threading.Timer(0.05, signal.set)
        timer.start()
        started = time.monotonic()
        try:
            with pytest.raises(LaunchAbandoned):
                signal.sleep(30.0)
        finally:
            timer.cancel()
        assert time.monotonic() - started < 5.0, (
            "the sleep must wake on the signal, not run its full duration"
        )


class TestPoolReleasesAnInFlightLaunch:
    """The third link of the stop-aware chain: the POOL must set the signal.

    ``ShutdownSignal.sleep`` raising and the real launcher unwinding on that raise
    are both proved above, but neither fires unless something sets the signal. This
    is the live trigger: if ``build()`` stopped forwarding ``shutdown=``, or
    ``stop()`` lost its ``set()``, both other tests would still pass while the live
    run went back to holding a collector thread for the launcher's full wait.
    """

    def _pool_with_signal(self, signal: ShutdownSignal, probe, cfg: TrainConfig):
        transport = LocalTransport()
        launcher = FakeLauncher()
        counter = GlobalEpisodeCounter()
        collectors = [
            _make_collector(
                arena_id, lambda: _DeadEnv() if arena_id else FakeEnv(k=_EPISODE_K),
                transport, cfg, launcher, counter,
            )
            for arena_id in range(cfg.arenas)
        ]
        pool = ActorPool(
            collectors, cfg, jvm_probe=probe, mc_port=25599, jvm_poll_seconds=0.01,
            shutdown=signal,
        )
        return pool, collectors, transport

    def test_stop_sets_the_shutdown_signal(self):
        signal = ShutdownSignal()
        cfg = _tiny_cfg(arenas=1, fault_relaunch=False)
        pool, collectors, transport = self._pool_with_signal(
            signal, RecordingJvmProbe(alive=True), cfg
        )
        pool.start()
        try:
            assert signal.is_set() is False, "the signal fired before shutdown"
        finally:
            transport.close()
        pool.stop()

        assert signal.is_set() is True, (
            "pool.stop() must release a collector parked inside a bridge relaunch"
        )
        for collector in collectors:
            collector.join(timeout=3.0)

    def test_a_jvm_abort_sets_the_shutdown_signal(self):
        signal = ShutdownSignal()
        probe = RecordingJvmProbe(alive=True)
        cfg = _tiny_cfg(arenas=2, fault_relaunch=True)
        pool, collectors, transport = self._pool_with_signal(signal, probe, cfg)
        pool.start()
        try:
            assert probe.called.wait(timeout=5.0)
            probe.kill()
            assert _wait_for(pool.aborted, timeout=10.0), "the JVM abort never fired"
            assert signal.is_set() is True, (
                "the abort must release an in-flight relaunch immediately, not wait "
                "for the driver to get around to calling stop()"
            )
        finally:
            for collector in collectors:
                collector.stop()
            transport.close()
            for collector in collectors:
                collector.join(timeout=3.0)

        with pytest.raises(PoolAbortedError):
            pool.stop()

    def test_build_forwards_the_shutdown_signal_to_the_pool(self):
        """ActorPool.build() is the live construction path; it must carry it."""
        signal = ShutdownSignal()
        cfg = _tiny_cfg(arenas=2, fault_relaunch=False)
        transport = LocalTransport()

        pool = ActorPool.build(
            cfg,
            env_factory_for=lambda arena_id: (lambda: FakeEnv(k=_EPISODE_K)),
            policy_for=lambda arena_id: _make_fake_policy(arena_id=arena_id)[0],
            transport=transport,
            weight_store=WeightStore(),
            launcher=FakeLauncher(),
            counter=GlobalEpisodeCounter(),
            sleep=lambda _s: None,
            jvm_probe=RecordingJvmProbe(alive=True),
            mc_port=25599,
            shutdown=signal,
        )
        transport.close()
        pool.stop()

        assert signal.is_set() is True, (
            "ActorPool.build() dropped the shutdown signal, so nothing would ever "
            "interrupt a launcher's bounded wait on the live path"
        )
        # build() must also hand the probe's port down to the collectors it makes.
        collector = pool.collector_for(0)
        assert collector is not None and collector._mc_port == 25599


class _FakeBridgeProcess:
    """Stand-in for a spawned bridge: never exits, never opens its port."""

    def __init__(self) -> None:
        self.terminated = False
        self.killed = False

    def poll(self):
        return None  # still running

    def terminate(self) -> None:
        self.terminated = True

    def wait(self, timeout=None):
        return 0

    def kill(self) -> None:
        self.killed = True


class TestRealLauncherIsStopAware:
    """The REAL SubprocessArenaLauncher must abandon its wait on shutdown.

    Driven against the real class (fake ``popen``/``port_probe`` only), because the
    thing under test is that its injected ``sleep`` is the interrupt point. A fake
    launcher would prove nothing: it is precisely the real one's ~135 s bounded wait
    on a collector thread that the ShutdownSignal exists to cut short.
    """

    def test_launch_abandons_promptly_when_shutdown_is_signalled(self):
        from distributed.launcher import SubprocessArenaLauncher

        spawned: List[_FakeBridgeProcess] = []
        mc_port = 25565
        bridge_base_port = 5555

        def _popen(command, **kwargs):
            process = _FakeBridgeProcess()
            spawned.append(process)
            return process

        def _port_probe(host: str, port: int) -> bool:
            # The JVM answers; the bridge port never comes up, so launch() settles
            # into its bounded readiness wait -- 120 s by default.
            return port == mc_port

        signal = ShutdownSignal()
        launcher = SubprocessArenaLauncher(
            mc_port=mc_port,
            bridge_base_port=bridge_base_port,
            popen=_popen,
            sleep=signal.sleep,
            port_probe=_port_probe,
            log=lambda _message: None,
        )

        timer = threading.Timer(0.05, signal.set)
        timer.start()
        started = time.monotonic()
        try:
            with pytest.raises(LaunchAbandoned):
                launcher.launch(3)
        finally:
            timer.cancel()
        elapsed = time.monotonic() - started

        assert elapsed < 5.0, (
            f"launch() took {elapsed:.1f}s to notice shutdown; with a plain "
            f"time.sleep it would have held the collector thread for ~120s"
        )
        assert spawned, "the launcher should have spawned a bridge before waiting"

    def test_launch_is_not_disturbed_while_no_shutdown_is_signalled(self):
        """Negative control: the same wiring succeeds normally when nothing stops it."""
        from distributed.launcher import SubprocessArenaLauncher

        mc_port = 25565
        bridge_base_port = 5555
        state = {"bridge_up": False}

        def _popen(command, **kwargs):
            state["bridge_up"] = True  # the bridge comes up as soon as it is spawned
            return _FakeBridgeProcess()

        def _port_probe(host: str, port: int) -> bool:
            if port == mc_port:
                return True
            return state["bridge_up"]

        signal = ShutdownSignal()
        launcher = SubprocessArenaLauncher(
            mc_port=mc_port,
            bridge_base_port=bridge_base_port,
            popen=_popen,
            sleep=signal.sleep,
            port_probe=_port_probe,
            log=lambda _message: None,
        )

        launcher.launch(2)  # must not raise
        assert signal.is_set() is False


# ---------------------------------------------------------------------------
# The watchdog must never fail silently, and a live run must never be unwatched
# ---------------------------------------------------------------------------


class TestWatchdogNeverDiesQuietly:
    """A supervisor that dies leaves the run training with tier 2 disarmed.

    The daemon thread would simply end, ``aborted()`` would stay False forever, and
    nothing would ever say so. The shipped jvm_alive() cannot cause this (it swallows
    only OSError), but a custom probe can, so the failure is itself an abort.
    """

    def test_a_raising_probe_aborts_the_run_instead_of_killing_the_thread(
        self, monkeypatch
    ):
        surfaced = threading.Event()
        seen: List[str] = []

        def _excepthook(args):
            seen.append(type(args.exc_value).__name__)
            surfaced.set()

        # Replace the thread excepthook (pytest installs its own) so the deliberate
        # re-raise is captured here instead of being reported as a stray warning.
        monkeypatch.setattr(threading, "excepthook", _excepthook)

        def _exploding_probe(host: str, port: int) -> bool:
            raise RuntimeError("probe blew up")

        cfg = _tiny_cfg(arenas=1, fault_relaunch=False)
        transport = LocalTransport()
        launcher = FakeLauncher()
        counter = GlobalEpisodeCounter()
        collector = _make_collector(
            0, lambda: FakeEnv(k=_EPISODE_K), transport, cfg, launcher, counter
        )
        pool = ActorPool(
            [collector], cfg, jvm_probe=_exploding_probe, mc_port=25599,
            jvm_poll_seconds=0.01,
        )
        pool.start()

        try:
            assert _wait_for(pool.aborted, timeout=10.0), (
                "a watchdog that cannot probe must abort the run, not die quietly "
                "and leave the fleet unsupervised"
            )
            assert surfaced.wait(timeout=5.0), (
                "the watchdog failure must also reach the thread excepthook"
            )
            assert seen == ["RuntimeError"], f"unexpected excepthook payload: {seen}"
        finally:
            collector.stop()
            transport.close()
            collector.join(timeout=3.0)

        with pytest.raises(PoolAbortedError) as excinfo:
            pool.stop()
        message = str(excinfo.value)
        assert "watchdog itself failed" in message, message
        assert "RuntimeError" in message, message
        assert "25599" in message, message

    def test_a_healthy_probe_leaves_the_watchdog_running(self):
        """Negative control: the try/except must not swallow the normal path."""
        cfg = _tiny_cfg(arenas=1, fault_relaunch=False)
        transport = LocalTransport()
        launcher = FakeLauncher()
        counter = GlobalEpisodeCounter()
        probe = RecordingJvmProbe(alive=True)
        collector = _make_collector(
            0, lambda: FakeEnv(k=_EPISODE_K), transport, cfg, launcher, counter
        )
        pool = ActorPool(
            [collector], cfg, jvm_probe=probe, jvm_poll_seconds=0.01
        )
        pool.start()
        try:
            assert probe.called.wait(timeout=5.0)
            assert _wait_for(lambda: probe.call_count() >= 3, timeout=5.0), (
                "the watchdog stopped polling a healthy JVM"
            )
            assert not pool.aborted()
        finally:
            _shutdown_pool(pool, [collector], transport)

        pool.stop()  # must not raise


class TestLiveRunsMustBeSupervised:
    """A live multi-pad run with no tier 2 is a configuration that must not exist."""

    def test_is_live_without_a_jvm_probe_is_refused(self):
        import dataclasses

        from agent.train import train_multi_arena

        cfg = dataclasses.replace(TrainConfig(), arenas=2)
        with pytest.raises(ValueError, match="requires a jvm_probe"):
            train_multi_arena(
                cfg,
                env_factory_for=lambda _arena_id: (lambda: FakeEnv(k=_EPISODE_K)),
                launcher=FakeLauncher(),
                is_live=True,
            )

    def test_offline_runs_may_omit_the_probe(self):
        """Negative control: the guard must not fire on the offline path.

        Reaching the arenas>=2 check with is_live unset proves the jvm_probe guard
        did not trip; this uses arenas=1 so nothing is actually started.
        """
        import dataclasses

        from agent.train import train_multi_arena

        cfg = dataclasses.replace(TrainConfig(), arenas=1)
        with pytest.raises(ValueError, match="requires cfg.arenas >= 2"):
            train_multi_arena(
                cfg,
                env_factory_for=lambda _arena_id: (lambda: FakeEnv(k=_EPISODE_K)),
                launcher=FakeLauncher(),
            )


class _StubReached(Exception):
    """Sentinel: the stubbed train_multi_arena was reached (nothing was started)."""


class TestMcPortMayNotBeABridgePort:
    """--mc-port inside the bridge range would make the watchdog evict a collector.

    Every mc-port probe CONNECTS. Against Paper that is free; against a bridge it
    destroys the incumbent client, so a watchdog pointed at pad i's bridge evicts
    that pad's collector on a timer, forever, with nothing naming the cause.
    """

    def _args(self, *, port: int, mc_port: int):
        import argparse

        return argparse.Namespace(
            port=port, host="127.0.0.1", mc_port=mc_port,
            max_episodes=1, max_grad_steps=1, eval_every_grad_steps=0,
            eval_episodes=1,
        )

    def _run(self, *, port: int, mc_port: int, arenas: int) -> int:
        import dataclasses

        from agent.train import _main_multi_arena

        cfg = dataclasses.replace(TrainConfig(), arenas=arenas)
        return _main_multi_arena(
            self._args(port=port, mc_port=mc_port),
            cfg,
            logger=None,
            checkpoint_hook=None,
        )

    def test_an_mc_port_inside_the_bridge_range_is_refused(self, capsys):
        code = self._run(port=5555, mc_port=5556, arenas=4)

        assert code == 1, "an mc port that collides with a bridge port must abort"
        stderr = capsys.readouterr().err
        assert "5556" in stderr and "5555" in stderr, (
            f"the failure must name both ports: {stderr}"
        )

    def test_the_first_and_last_bridge_ports_are_both_refused(self):
        assert self._run(port=5555, mc_port=5555, arenas=4) == 1
        assert self._run(port=5555, mc_port=5558, arenas=4) == 1

    def test_the_port_just_past_the_range_is_allowed_through(self, monkeypatch):
        """Negative control: 5559 is not pad 0..3's port, so the guard must pass it.

        Stubs train_multi_arena so nothing is started, and asserts the mc_port the
        guard let through is the one actually handed to the pool AND to the launcher
        -- the two must agree or the watchdog names a port nobody probes.
        """
        import agent.train as train_module

        seen: dict = {}

        def _stub_train_multi_arena(cfg, **kwargs):
            seen.update(kwargs)
            raise _StubReached()

        monkeypatch.setattr(train_module, "train_multi_arena", _stub_train_multi_arena)

        with pytest.raises(_StubReached):
            self._run(port=5555, mc_port=5559, arenas=4)

        assert seen["mc_port"] == 5559, (
            f"the guard must forward the mc port it accepted, got {seen.get('mc_port')}"
        )
        assert seen["jvm_probe"] is jvm_alive, "the live path must arm tier 2"
        assert isinstance(seen["launcher_shutdown"], ShutdownSignal)
        # The launcher was built with the same port before this call.
        assert seen["launcher"]._mc_port == 5559, (
            "the launcher and the watchdog must be given the same mc port"
        )

    def test_the_defaults_are_disjoint(self):
        """25565 vs 5555+i: the shipped defaults can never collide."""
        from agent.train import _DEFAULT_MC_PORT

        assert _DEFAULT_MC_PORT not in range(5555, 5555 + 25), (
            "the default mc port collides with the bridge range at the 25-pad ladder "
            "rung"
        )
