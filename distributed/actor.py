"""actor — thread-per-arena collectors + the ActorPool supervisor (T7).

This is the actor side of the Ape-X-lite seam (plan §Decisions). One
:class:`Collector` runs as a daemon thread per arena: it rolls episodes against
its OWN :class:`~env.mc_pvp_env.MCPvPEnv` using a :class:`~distributed.weights.SnapshotPolicy`
(a periodically-synced weight SNAPSHOT, never the live learner net — that would be
a torch read-during-write race) and pushes each whole
:class:`~distributed.serialization.Episode` onto the shared
:class:`~distributed.transport.ExperienceTransport`. The :class:`ActorPool`
supervises N collectors and decides, at the WHOLE-RUN level, when too many arenas
are down to keep going.

Why these specific shapes:

  * **Snapshot at the EPISODE BOUNDARY.** Each loop calls
    ``policy.maybe_refresh(weight_store)`` once, before rolling. Refreshing only
    between episodes (never mid-episode) keeps a single episode's LSTM trajectory
    on one coherent weight set, which is what the learner's R2D2 recurrence gate
    (TC8b) relies on.

  * **GLOBAL ε counter, LOCAL seed.** ε is computed from a GLOBAL atomic episode
    counter shared across all collectors, so the ε schedule advances monotonically
    over the combined episode stream regardless of which arena produced which
    episode (plan §Decisions: schedule progression is global; PER β is the
    learner's concern, indexed off its grad step). The per-episode SEED, by
    contrast, is LOCAL — ``arena_episode_seed(cfg, arena_id, local_ep)`` — so each
    arena draws an independent, reproducible stream and two arenas never collide.

  * **Fault handling follows the bridge resilience contract EXACTLY.** A
    :class:`~env.mc_pvp_env.BridgeError` during a rollout aborts THAT episode
    (``step()`` desync is unrecoverable — the env already refuses to silent-retry a
    mid-episode reply loss, and we must NOT add one). The collector then FIRST tries
    the env's idempotent ``reset()`` reconnect (bounded): a transient drop — e.g. a
    periodic eval transiently stole the bridge's single connection — recovers there
    and the collector resumes. Only if that fails does the collector treat the arena
    as genuinely DEAD: it marks itself down, asks the injectable
    :class:`ArenaLauncher` to relaunch the OS processes, backs off on a
    seconds-to-tens-of-seconds scale (relaunching a Paper server is slow — 30-60s+),
    and reconnects with a FRESH env/client to the SAME arena's bridge. A bridge
    serves exactly ONE connection, so a reconnect always opens a fresh client to the
    same port; we never multiplex. The learner and the OTHER collectors are
    unaffected throughout — survivors keep producing.

  * **The pool aborts on the AGGREGATE, not per-arena.** Because a relaunched arena
    may legitimately be down for tens of seconds, "any arena down" is the wrong abort
    trigger. The :class:`ActorPool` aborts the run LOUDLY only when the number of
    LIVE arenas drops below ``cfg.fault_min_live_arenas``; survivors keep feeding the
    learner meanwhile.

Injectability (T14): every collaborator is injected — an ``env_factory`` (so a
relaunch rebuilds a fresh client to the same arena), the ``policy``, the
``transport``, the ``weight_store``, the ``cfg``, and the :class:`ArenaLauncher`
(a fake in tests records ``launch``/``terminate`` calls). Backoff durations are
constructor parameters so unit tests pass tiny values and never sleep for real
seconds. A fake env can raise :class:`~env.mc_pvp_env.BridgeError` on demand to
drive the fault paths.

Owner: T7 (multi-arena throughput track, issue #4)
"""

from __future__ import annotations

import threading
from typing import Callable, Dict, List, Optional, Protocol

from agent.train import (
    EnvProtocol,
    RolloutPolicy,
    arena_episode_seed,
    collect_episode,
    epsilon_for_episode,
)
from agent.train_config import TrainConfig
from distributed.transport import ExperienceTransport, TransportError
from distributed.weights import WeightStore
from env.mc_pvp_env import BridgeError

__all__ = [
    "ArenaLauncher",
    "PoolAbortedError",
    "GlobalEpisodeCounter",
    "Collector",
    "ActorPool",
]


# A zero-arg callable that builds (and connects) a FRESH env bound to one arena's
# bridge. A relaunch rebuilds the env through this so the new connection is a fresh
# client to the SAME arena's single-connection bridge — never a reused dead socket.
EnvFactory = Callable[[], EnvProtocol]


# Default fault timing (seconds). Relaunching a Paper server is slow (world-gen,
# plugin load, bot re-op/re-teleport take 30-60s+), so the backoff between a failed
# reconnect and the next attempt is on the order of seconds and grows toward a cap.
# Both are constructor-overridable so unit tests inject tiny values and never sleep
# for real seconds.
_DEFAULT_RELAUNCH_BACKOFF_SECONDS: float = 5.0
_DEFAULT_RELAUNCH_BACKOFF_MAX_SECONDS: float = 60.0
# Bounded attempts at the env's own idempotent reconnect before declaring the arena
# dead. reset() already retries the transport internally; a couple of outer attempts
# absorbs a transient drop without masking a genuinely-down bridge.
_DEFAULT_RESET_RECONNECT_ATTEMPTS: int = 2


class PoolAbortedError(RuntimeError):
    """Raised when the :class:`ActorPool` aborts the run LOUDLY.

    The pool raises this from :meth:`ActorPool.stop` / :meth:`ActorPool.raise_if_aborted`
    when the number of live arenas has dropped below ``cfg.fault_min_live_arenas``
    after relaunch attempts could not keep enough arenas alive. It is a loud,
    explicit failure: the run must stop rather than silently train on a degraded
    pool that can no longer feed the learner at the required rate.
    """


class ArenaLauncher(Protocol):
    """Starts/stops the OS processes (Paper server + bridge) for one arena.

    Injectable so all supervisor/fault logic is testable offline against a fake that
    merely records ``launch``/``terminate`` calls; the real subprocess shim (T11) is
    the thin untested layer. A relaunch is REQUESTED here (start the processes) but
    the collector reconnects to the bridge itself once the arena is back — the
    launcher owns process lifecycle, the collector owns the connection.
    """

    def launch(self, arena_id: int) -> None:
        """Start (or restart) the Paper server + bridge for ``arena_id``.

        Slow in production (30-60s+). The collector backs off and then reconnects on
        a fresh client; this call need only kick the processes off.
        """
        ...

    def terminate(self, arena_id: int) -> None:
        """Stop the Paper server + bridge for ``arena_id``. Best-effort, idempotent."""
        ...


class GlobalEpisodeCounter:
    """A lock-guarded monotone episode counter shared across ALL collectors.

    ε is computed per collector from this GLOBAL index so the ε schedule advances
    monotonically over the combined episode stream no matter which arena produced
    which episode (plan §Decisions). Each collector calls :meth:`next_index` once per
    episode to claim a unique, strictly-increasing index; the lock makes the
    read-increment atomic across the N collector threads.
    """

    def __init__(self, start: int = 0) -> None:
        self._lock = threading.Lock()
        self._value = int(start)

    def next_index(self) -> int:
        """Claim and return the next global episode index (strictly increasing)."""
        with self._lock:
            index = self._value
            self._value += 1
            return index

    @property
    def value(self) -> int:
        """The next index that would be handed out (the count claimed so far)."""
        with self._lock:
            return self._value


class Collector:
    """One arena's daemon collector: refresh -> roll one Episode -> send, with faults.

    Runs :meth:`run` on its own thread. Each iteration:

      1. ``policy.maybe_refresh(weight_store)`` — pull the latest weight SNAPSHOT at
         the EPISODE BOUNDARY (coherent within-episode LSTM trajectory; protects TC8b).
      2. claim a GLOBAL episode index and compute ε from it
         (``epsilon_for_episode``), so the ε schedule advances across all arenas.
      3. compute this arena's LOCAL per-episode seed
         (``arena_episode_seed(cfg, arena_id, local_ep)``) and ``policy.reseed`` it.
      4. ``collect_episode(env, policy, ...)`` — roll one episode.
      5. ``transport.send(episode)`` — hand it to the learner.
      6. advance ``local_ep`` (the global counter advanced in step 2).

    Fault path (the bridge resilience contract): a :class:`BridgeError` from any of
    the above aborts THAT episode. The collector FIRST tries the env's idempotent
    ``reset()`` reconnect (bounded); if that recovers, it resumes. If not, it marks
    the arena DEAD, asks the :class:`ArenaLauncher` to relaunch it, backs off, and
    rebuilds the env via ``env_factory`` (a FRESH client to the same arena's bridge),
    then resumes. ``step()`` is NEVER silently retried (that corrupts the episode) —
    only ``reset()`` may reconnect-and-retry, and the env enforces that internally.

    Args:
        arena_id: 0-based index of this collector's arena.
        env_factory: Zero-arg callable building (and connecting) a FRESH env for this
            arena. Called once at start and again on every relaunch so a reconnect is
            always a fresh client to the same single-connection bridge.
        policy: The acting surface (a :class:`~distributed.weights.SnapshotPolicy`),
            satisfying ``RolloutPolicy``. Its ``arena_id`` should match ``arena_id``.
        transport: The shared actor->learner channel (episodes flow up).
        weight_store: The shared snapshot store the learner publishes to.
        cfg: Training config (ε schedule, per-arena seed scheme, fault knobs).
        launcher: The :class:`ArenaLauncher` used to relaunch a dead arena.
        on_state_change: Optional callback ``(arena_id, alive: bool) -> None`` the
            collector fires when it transitions live<->dead, so the pool tracks the
            aggregate live count. Called OUTSIDE any collector-held lock.
        max_episode_steps: Per-episode decision cap forwarded to ``collect_episode``.
        relaunch_backoff_seconds: Initial backoff after a failed reconnect.
        relaunch_backoff_max_seconds: Cap the (doubling) backoff grows toward.
        reset_reconnect_attempts: Bounded env ``reset()`` reconnect attempts before
            declaring the arena dead.
        sleep: Injectable sleep (defaults to ``time.sleep``); tests pass a no-op /
            recorder so unit tests never block on real seconds.
    """

    def __init__(
        self,
        arena_id: int,
        env_factory: EnvFactory,
        policy: RolloutPolicy,
        transport: ExperienceTransport,
        weight_store: WeightStore,
        cfg: TrainConfig,
        launcher: ArenaLauncher,
        *,
        counter: GlobalEpisodeCounter,
        on_state_change: Optional[Callable[[int, bool], None]] = None,
        max_episode_steps: Optional[int] = None,
        relaunch_backoff_seconds: float = _DEFAULT_RELAUNCH_BACKOFF_SECONDS,
        relaunch_backoff_max_seconds: float = _DEFAULT_RELAUNCH_BACKOFF_MAX_SECONDS,
        reset_reconnect_attempts: int = _DEFAULT_RESET_RECONNECT_ATTEMPTS,
        sleep: Optional[Callable[[float], None]] = None,
    ) -> None:
        self.arena_id = int(arena_id)
        self._env_factory = env_factory
        self._policy = policy
        self._transport = transport
        self._weight_store = weight_store
        self._cfg = cfg
        self._launcher = launcher
        self._counter = counter
        self._on_state_change = on_state_change
        self._max_episode_steps = max_episode_steps
        self._relaunch_backoff_seconds = float(relaunch_backoff_seconds)
        self._relaunch_backoff_max_seconds = float(relaunch_backoff_max_seconds)
        self._reset_reconnect_attempts = int(reset_reconnect_attempts)

        # Backoff sleep. When no sleep is injected we use the stop Event's timed wait
        # (interruptible, so a long relaunch backoff never delays a clean shutdown).
        # Tests inject a no-op / recorder so a unit test never blocks on real seconds.
        self._injected_sleep = sleep

        # The live env. Built lazily in run() (or after a relaunch) so constructing a
        # Collector never opens a connection — important for tests and for symmetry
        # with a slow real launch.
        self._env: Optional[EnvProtocol] = None

        # Liveness + stop signalling. `_alive` mirrors the connection state the pool
        # watches; `_stop` is latched by the pool to wind the loop down cleanly. Both
        # are simple flags read across threads; assignment to a bool is atomic under
        # CPython, and the pool only ever READS `_alive`.
        self._alive: bool = True
        self._stop = threading.Event()

        # This arena's own episode index. LOCAL (not global): it drives the per-arena
        # deterministic seed so each arena's stream is independent and reproducible.
        self._local_ep: int = 0

        self._thread: Optional[threading.Thread] = None

    # -- liveness (read by the pool's supervisor) --------------------------

    @property
    def alive(self) -> bool:
        """True while this arena holds a working connection (False while down)."""
        return self._alive

    def _set_alive(self, alive: bool) -> None:
        """Flip liveness and notify the pool, only on an actual transition."""
        if self._alive == alive:
            return
        self._alive = alive
        if self._on_state_change is not None:
            # Fire OUTSIDE any lock: the callback updates the pool's aggregate count
            # and must not be able to deadlock against a collector-held lock.
            self._on_state_change(self.arena_id, alive)

    # -- thread lifecycle --------------------------------------------------

    def start(self) -> threading.Thread:
        """Spawn the daemon collector thread and return it (idempotent-ish).

        Daemon so a crash of the main process never hangs on a blocked collector. The
        pool owns join/stop ordering via :meth:`stop`.
        """
        if self._thread is not None and self._thread.is_alive():
            return self._thread
        self._thread = threading.Thread(
            target=self.run,
            name=f"collector-arena-{self.arena_id}",
            daemon=True,
        )
        self._thread.start()
        return self._thread

    def stop(self) -> None:
        """Signal the loop to wind down after the current episode (non-blocking)."""
        self._stop.set()

    def join(self, timeout: Optional[float] = None) -> None:
        """Join the collector thread (if it was started)."""
        if self._thread is not None:
            self._thread.join(timeout)

    # -- the rollout loop --------------------------------------------------

    def run(self) -> None:
        """Drive the refresh -> collect -> send loop until stopped, recovering faults.

        Builds the env on first entry. A :class:`BridgeError` anywhere in an episode
        is funneled into :meth:`_recover`, which honors the resilience contract
        (idempotent ``reset()`` reconnect first, then relaunch + backoff + fresh
        client). The loop exits cleanly when :meth:`stop` is signalled or when a
        transport close ends the run; any other exception propagates (a real bug
        should surface, not be swallowed).
        """
        try:
            self._ensure_env()
        except BridgeError:
            # The very first connect failed: treat the arena as dead and try to
            # recover before entering the steady-state loop.
            self._set_alive(False)
            if not self._recover():
                return

        while not self._stop.is_set():
            try:
                self._collect_one()
            except BridgeError:
                # The episode aborted on a desync/drop. Per the contract this episode
                # is lost; try to get the arena back. If recovery is abandoned
                # (stop signalled), leave the loop.
                self._set_alive(False)
                if not self._recover():
                    return
            except TransportError:
                # The learner closed the channel: the run is ending. Stop cleanly
                # rather than treating the closed queue as an arena fault.
                return

    def _collect_one(self) -> None:
        """Run ONE episode and send it. May raise :class:`BridgeError`/``TransportError``."""
        env = self._env
        if env is None:  # pragma: no cover - _ensure_env guarantees this
            raise BridgeError(f"arena {self.arena_id}: no env to collect against")

        # (1) Snapshot refresh at the EPISODE BOUNDARY — coherent within-episode
        # weights for the whole LSTM trajectory (TC8b).
        self._policy.maybe_refresh(self._weight_store)

        # (2) GLOBAL episode index -> ε. The schedule advances across all arenas.
        global_index = self._counter.next_index()
        epsilon = epsilon_for_episode(global_index, self._cfg)

        # (3) LOCAL per-arena seed -> reproducible, independent per-arena stream.
        episode_seed = arena_episode_seed(self._cfg, self.arena_id, self._local_ep)
        self._policy.reseed(episode_seed)

        # (4) Roll one episode. collect_episode re-reseeds and resets the env from
        # episode_seed; it raises BridgeError straight through on a mid-episode desync
        # (env.step never silent-retries), which our caller funnels into recovery.
        episode = collect_episode(
            env,
            self._policy,
            max_steps=self._max_episode_steps,
            episode_index=global_index,
            epsilon=epsilon,
            episode_seed=episode_seed,
        )

        # (5) Hand the whole Episode to the learner (the only replay mutator).
        self._transport.send(episode)

        # (6) Advance this arena's LOCAL counter. The global counter already advanced
        # in step (2). A successful send means we are (still) live.
        self._local_ep += 1
        self._set_alive(True)

    # -- fault recovery ----------------------------------------------------

    def _ensure_env(self) -> None:
        """Build (and connect) the env if we do not currently hold one.

        The factory connects on construction (the env's ``auto_connect``), so a
        successful return means a working connection. A failure raises
        :class:`BridgeError` for the caller to route into recovery.
        """
        if self._env is not None:
            return
        self._env = self._env_factory()
        self._set_alive(True)

    def _recover(self) -> bool:
        """Bring the arena back, honoring the resilience contract. Returns liveness.

        Step A — idempotent reconnect: try the env's own ``reset()`` (bounded). It is
        idempotent and carries no in-flight episode state, so a transient drop (e.g. a
        periodic eval briefly stole the single connection) recovers here without
        touching the OS processes. A successful reset means the arena is live again.

        Step B — relaunch: if the reconnect cannot recover, the arena is genuinely
        dead. Ask the launcher to relaunch the processes, back off (seconds-scale,
        doubling toward the cap — a Paper relaunch is slow), drop the old env, rebuild
        a FRESH env/client via the factory, and confirm with a reset. Repeat until the
        arena is back or :meth:`stop` is signalled.

        Returns:
            ``True`` if the arena is live again; ``False`` if recovery was abandoned
            because the pool signalled stop (the loop should then exit).
        """
        # --- Step A: bounded idempotent reset() reconnect on the SAME env. ---
        env = self._env
        if env is not None:
            for _ in range(max(0, self._reset_reconnect_attempts)):
                if self._stop.is_set():
                    return False
                try:
                    # reset() reconnect-and-retries internally; a clean return means
                    # the bridge is back. Seed deterministically off this arena's next
                    # episode so the resumed stream stays reproducible.
                    seed = arena_episode_seed(self._cfg, self.arena_id, self._local_ep)
                    env.reset(seed=seed)
                except BridgeError:
                    continue
                else:
                    # Recovered without a relaunch. NOTE: collect_episode will reset()
                    # again at the top of the next episode (idempotent), so this probe
                    # reset does not desync the stream — it only proves the link.
                    self._set_alive(True)
                    return True

        # --- Step B: relaunch the OS processes, then reconnect on a fresh client. ---
        backoff = self._relaunch_backoff_seconds
        while not self._stop.is_set():
            if self._cfg.fault_relaunch:
                # Best-effort: a launcher failure must not crash the collector — we
                # back off and try again. (The pool's aggregate-liveness watchdog is
                # what ultimately aborts the run if relaunches never succeed.)
                try:
                    self._launcher.launch(self.arena_id)
                except Exception:  # noqa: BLE001 - launcher faults must not kill us
                    pass

            # Back off on a seconds-to-tens-of-seconds scale: a Paper server takes
            # 30-60s+ to come up, so hammering reconnect immediately is pointless.
            self._interruptible_sleep(backoff)
            if self._stop.is_set():
                return False
            backoff = min(backoff * 2.0, self._relaunch_backoff_max_seconds)

            # Drop the old (dead) env and rebuild a FRESH client to the SAME arena's
            # single-connection bridge. Never reuse or multiplex a connection.
            self._close_env_quietly()
            try:
                self._env = self._env_factory()
            except BridgeError:
                # Arena not back yet (factory connects eagerly); keep backing off.
                continue

            # Confirm the fresh connection with an idempotent reset probe.
            try:
                seed = arena_episode_seed(self._cfg, self.arena_id, self._local_ep)
                self._env.reset(seed=seed)
            except BridgeError:
                continue

            self._set_alive(True)
            return True

        return False

    def _interruptible_sleep(self, seconds: float) -> None:
        """Sleep ``seconds`` but wake immediately if stop is signalled.

        With an injected sleep (tests) use it verbatim — a recorder/no-op so the test
        never blocks on real seconds. Otherwise use the stop Event's timed wait so a
        long relaunch backoff wakes instantly on :meth:`stop` and never delays a clean
        shutdown.
        """
        if seconds <= 0.0:
            return
        if self._injected_sleep is not None:
            self._injected_sleep(seconds)
            return
        self._stop.wait(timeout=seconds)

    def _close_env_quietly(self) -> None:
        """Close the current env, swallowing teardown errors (it is going away)."""
        env = self._env
        self._env = None
        if env is None:
            return
        close = getattr(env, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # noqa: BLE001 - teardown of a dead arena is best-effort
                pass


class ActorPool:
    """Supervises N :class:`Collector` daemons and aborts the run if too many die.

    ``start()`` builds and starts one collector per arena (each owns its own env,
    policy, and recovery loop) plus tracks the aggregate live count. A single arena
    going down does NOT stop the pool: that collector relaunches its arena out-of-band
    (slow — 30-60s+) while the survivors keep feeding the learner. The pool aborts the
    WHOLE run LOUDLY only when :meth:`live_count` drops below
    ``cfg.fault_min_live_arenas`` — using the aggregate threshold (not "any arena
    down") precisely because a relaunching arena is expected to be absent a while.

    The abort is surfaced two ways: :meth:`raise_if_aborted` lets a driver poll, and
    :meth:`stop` raises :class:`PoolAbortedError` on its way out if the run aborted —
    so a run can never quietly continue on a sub-floor pool.

    Args:
        collectors: The collectors to supervise (one per arena). Build them with
            :meth:`build` for the common case, or inject pre-built/fake collectors.
        cfg: Training config (reads ``fault_min_live_arenas``).
    """

    def __init__(self, collectors: List[Collector], cfg: TrainConfig) -> None:
        self._collectors = list(collectors)
        self._cfg = cfg

        # Aggregate liveness. Seeded from each collector's current state; mutated only
        # via the collectors' on_state_change callback (fired off the collector
        # threads) and read by live_count()/the supervisor — all under this lock.
        self._lock = threading.Lock()
        self._live: Dict[int, bool] = {
            c.arena_id: c.alive for c in self._collectors
        }

        # Abort state. Latched once; the supervisor sets it, stop()/raise_if_aborted
        # surface it.
        self._aborted = threading.Event()
        self._abort_error: Optional[PoolAbortedError] = None

        self._stopping = threading.Event()
        self._supervisor: Optional[threading.Thread] = None

    # -- construction helper ----------------------------------------------

    @classmethod
    def build(
        cls,
        cfg: TrainConfig,
        *,
        env_factory_for: Callable[[int], EnvFactory],
        policy_for: Callable[[int], RolloutPolicy],
        transport: ExperienceTransport,
        weight_store: WeightStore,
        launcher: ArenaLauncher,
        counter: Optional[GlobalEpisodeCounter] = None,
        max_episode_steps: Optional[int] = None,
        relaunch_backoff_seconds: float = _DEFAULT_RELAUNCH_BACKOFF_SECONDS,
        relaunch_backoff_max_seconds: float = _DEFAULT_RELAUNCH_BACKOFF_MAX_SECONDS,
        reset_reconnect_attempts: int = _DEFAULT_RESET_RECONNECT_ATTEMPTS,
        sleep: Optional[Callable[[float], None]] = None,
    ) -> "ActorPool":
        """Build a pool of ``cfg.arenas`` collectors sharing one transport/store/counter.

        ``env_factory_for(arena_id)`` returns that arena's zero-arg env factory (so a
        relaunch rebuilds a fresh client to the same bridge); ``policy_for(arena_id)``
        returns that arena's :class:`~distributed.weights.SnapshotPolicy`. The GLOBAL
        ε counter is shared across all collectors (created here if not injected) so the
        ε schedule advances over the combined stream.

        Args:
            cfg: Training config (``arenas`` count + fault knobs).
            env_factory_for: ``arena_id -> EnvFactory`` (fresh-client builder).
            policy_for: ``arena_id -> RolloutPolicy`` (per-arena snapshot policy).
            transport: Shared actor->learner channel.
            weight_store: Shared snapshot store.
            launcher: Shared :class:`ArenaLauncher`.
            counter: Optional shared :class:`GlobalEpisodeCounter`; created if ``None``.
            max_episode_steps / relaunch_backoff_* / reset_reconnect_attempts / sleep:
                Forwarded to every :class:`Collector` (see its docstring).

        Returns:
            A constructed :class:`ActorPool` (not yet started).
        """
        shared_counter = counter if counter is not None else GlobalEpisodeCounter()

        # on_state_change is left None here and bound in start(): the pool wires its
        # own _mark callback onto every collector (built or injected) when it starts,
        # so there is no need to reference a half-constructed pool from a closure.
        collectors = [
            Collector(
                arena_id=arena_id,
                env_factory=env_factory_for(arena_id),
                policy=policy_for(arena_id),
                transport=transport,
                weight_store=weight_store,
                cfg=cfg,
                launcher=launcher,
                counter=shared_counter,
                max_episode_steps=max_episode_steps,
                relaunch_backoff_seconds=relaunch_backoff_seconds,
                relaunch_backoff_max_seconds=relaunch_backoff_max_seconds,
                reset_reconnect_attempts=reset_reconnect_attempts,
                sleep=sleep,
            )
            for arena_id in range(cfg.arenas)
        ]
        return cls(collectors, cfg)

    # -- liveness bookkeeping ----------------------------------------------

    def _mark(self, arena_id: int, alive: bool) -> None:
        """Record an arena's live/dead transition (collector-thread callback)."""
        with self._lock:
            self._live[arena_id] = alive

    def live_count(self) -> int:
        """Number of arenas currently holding a working connection."""
        with self._lock:
            return sum(1 for alive in self._live.values() if alive)

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Start every collector daemon and the supervisor that watches liveness."""
        # Bind the pool's callback onto any collectors that were injected without one
        # (so externally-built/fake collectors still update the aggregate count).
        for collector in self._collectors:
            if collector._on_state_change is None:
                collector._on_state_change = self._mark
            # Re-seed the live map from the (possibly updated) current state.
            with self._lock:
                self._live[collector.arena_id] = collector.alive

        for collector in self._collectors:
            collector.start()

        self._supervisor = threading.Thread(
            target=self._supervise,
            name="actor-pool-supervisor",
            daemon=True,
        )
        self._supervisor.start()

    def _supervise(self) -> None:
        """Watch the aggregate live count; abort the run if it falls below the floor.

        Polls liveness on a short cadence. The abort threshold is the AGGREGATE
        ``fault_min_live_arenas`` — NOT "any arena down" — so a single arena that is
        slowly relaunching (30-60s+) does not abort a pool that still has enough live
        arenas feeding the learner.
        """
        floor = self._cfg.fault_min_live_arenas
        while not self._stopping.is_set():
            if self.live_count() < floor:
                self._abort(
                    f"actor pool aborting: only {self.live_count()} live arena(s) "
                    f"remain, below the required floor of {floor} "
                    f"(fault_min_live_arenas); the surviving arenas cannot sustain "
                    f"the run. Relaunch attempts did not restore enough arenas."
                )
                return
            # Short poll so an abort is detected promptly without busy-spinning.
            if self._stopping.wait(timeout=0.05):
                return

    def _abort(self, message: str) -> None:
        """Latch the abort error and signal shutdown (idempotent)."""
        if self._aborted.is_set():
            return
        self._abort_error = PoolAbortedError(message)
        self._aborted.set()
        # Tell every collector to wind down — the run is over.
        self._stopping.set()
        for collector in self._collectors:
            collector.stop()

    def raise_if_aborted(self) -> None:
        """Raise :class:`PoolAbortedError` if the pool has aborted (else no-op).

        Lets a driver poll the pool's health between learner steps and stop the run
        loudly the moment the live floor is breached.
        """
        if self._aborted.is_set() and self._abort_error is not None:
            raise self._abort_error

    def aborted(self) -> bool:
        """True once the pool has aborted below the live floor."""
        return self._aborted.is_set()

    def stop(self) -> None:
        """Stop all collectors and the supervisor cleanly, then surface any abort.

        Idempotent. Signals every collector to wind down after its current episode,
        joins the supervisor, and — if the pool aborted below the live floor — raises
        :class:`PoolAbortedError` so a degraded run can never end silently.

        Raises:
            PoolAbortedError: if the pool aborted below ``fault_min_live_arenas``.
        """
        self._stopping.set()
        for collector in self._collectors:
            collector.stop()
        # Join the supervisor first (it no longer needs to poll), then the collectors.
        if self._supervisor is not None:
            self._supervisor.join(timeout=1.0)
        for collector in self._collectors:
            collector.join(timeout=1.0)

        self.raise_if_aborted()
