"""actor — thread-per-pad collectors + the two-tier fault policy (T7, T11).

This is the actor side of the Ape-X-lite seam (plan §Decisions). One
:class:`Collector` runs as a daemon thread per pad: it rolls episodes against
its OWN :class:`~env.mc_pvp_env.MCPvPEnv` using a :class:`~distributed.weights.SnapshotPolicy`
(a periodically-synced weight SNAPSHOT, never the live learner net — that would be
a torch read-during-write race) and pushes each whole
:class:`~distributed.serialization.Episode` onto the shared
:class:`~distributed.transport.ExperienceTransport`. The :class:`ActorPool`
supervises N collectors and owns the WHOLE-RUN abort.

THE TWO-TIER FAULT POLICY (T11). Under the one-JVM/N-pad topology there are
exactly two kinds of fault and they get opposite responses:

  * **Tier 1 — a pad's BRIDGE dies.** Restart THAT pad's bridge and nothing else
    (:meth:`Collector.restart_bridge`, which goes through the injected
    :class:`ArenaLauncher`). Every other pad keeps collecting; the run continues.
  * **Tier 2 — the shared Paper JVM dies.** ABORT the whole run, loudly, naming
    the JVM (detected by :func:`jvm_alive`, a plain TCP connect probe of the
    Minecraft port). Every pad in the fleet lives in that one JVM, so there are no
    survivors to continue on — and restarting a bridge into a dead JVM only
    produces a process that exits immediately.

There is deliberately NO survivor floor. ``fault_min_live_arenas`` was DELETED in
T11: a floor means the run quietly keeps training on a shrunken fleet, which is
exactly what the plan forbids ("abort rather than silently train on fewer
arenas"). Tier 1 does not need a floor because a dead pad is REPAIRED rather than
written off, and tier 2 does not need one because a dead JVM leaves nothing alive.
A single ``False`` from :func:`jvm_alive` is the abort trigger — no grace period,
no retry budget, no partial-fleet mode.

Why probe the Minecraft port and not a bridge port: Paper is an ordinary
multi-client server, so a connect-and-close costs it nothing and evicts nobody.
``BridgeServer`` is the opposite — it accepts exactly ONE TCP client and a second
connection DESTROYS the incumbent (``bridge/transport.js``), so a connect probe
against a live bridge would silently kill the very collector it was checking on.
Never point :func:`jvm_alive` at a bridge port.

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
    and the collector resumes. Only if that fails does the collector treat the pad
    as genuinely DEAD: it marks itself down, checks the SHARED JVM (tier 2 first —
    a bridge restart into a dead JVM is pointless), then asks the injectable
    :class:`ArenaLauncher` to restart THAT PAD'S BRIDGE, backs off, and reconnects
    with a FRESH env/client to the SAME pad's bridge. A bridge serves exactly ONE
    connection, so a reconnect always opens a fresh client to the same port; we
    never multiplex. The learner and the OTHER collectors are unaffected
    throughout — the other pads keep producing.

  * **A restarted bridge re-runs ``arena:reset_pad`` by itself.** The bridge is the
    SOLE reset authority: ``handleReset`` issues ``/function arena:reset_pad`` with
    this pad's anchor on EVERY reset, and the relaunched process is spawned with
    its ``--pad-origin`` anchor. Recovery already ends in ``env.reset()``, so the
    pad is re-placed by the ordinary path. There is deliberately no separate
    "re-run the reset macro" step here: it would have to open a SECOND client to a
    single-client bridge, evicting the collector that just reconnected.

Injectability (T14): every collaborator is injected — an ``env_factory`` (so a
relaunch rebuilds a fresh client to the same pad), the ``policy``, the
``transport``, the ``weight_store``, the ``cfg``, the :class:`ArenaLauncher`
(a fake in tests records ``launch``/``terminate`` calls), and the JVM probe
(``jvm_probe``, a ``(host, port) -> bool``; :func:`jvm_alive` is the live one).
Backoff durations are constructor parameters so unit tests pass tiny values and
never sleep for real seconds. A fake env can raise
:class:`~env.mc_pvp_env.BridgeError` on demand to drive the fault paths.

Owner: T7 (multi-arena throughput track, issue #4); T11 (two-tier fault policy)
"""

from __future__ import annotations

import socket
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
    "JvmProbe",
    "LaunchAbandoned",
    "PoolAbortedError",
    "ShutdownSignal",
    "GlobalEpisodeCounter",
    "Collector",
    "ActorPool",
    "jvm_alive",
    "MC_HOST",
    "MC_PORT",
]


# A zero-arg callable that builds (and connects) a FRESH env bound to one pad's
# bridge. A relaunch rebuilds the env through this so the new connection is a fresh
# client to the SAME pad's single-connection bridge — never a reused dead socket.
EnvFactory = Callable[[], EnvProtocol]

#: ``(host, port) -> bool`` liveness check for the SHARED Paper JVM.
#: :func:`jvm_alive` is the live implementation; a test injects a stub so no unit
#: test ever touches the network. Called positionally, so any two-arg callable works.
JvmProbe = Callable[[str, int], bool]


# Default fault timing (seconds). A bridge relaunch is slow (the process reconnects
# BOTH bots to Paper and waits for their spawns before it listens), so the backoff
# between a failed reconnect and the next attempt is on the order of seconds and
# grows toward a cap. Both are constructor-overridable so unit tests inject tiny
# values and never sleep for real seconds.
_DEFAULT_RELAUNCH_BACKOFF_SECONDS: float = 5.0
_DEFAULT_RELAUNCH_BACKOFF_MAX_SECONDS: float = 60.0
# Bounded attempts at the env's own idempotent reconnect before declaring the pad
# dead. reset() already retries the transport internally; a couple of outer attempts
# absorbs a transient drop without masking a genuinely-down bridge.
_DEFAULT_RESET_RECONNECT_ATTEMPTS: int = 2

#: Where the SHARED Paper JVM listens. ONE JVM serves every pad, so this is a single
#: port and not a base (mirrors ``server.properties`` ``server-port`` and
#: ``distributed.launcher``'s default). Callers that can be pointed elsewhere pass
#: BOTH the launcher and the pool the same value — see ``agent.train._main_multi_arena``.
MC_HOST: str = "127.0.0.1"
MC_PORT: int = 25565

#: Seconds between the pool supervisor's JVM probes. Long enough that a live server
#: is not connect-hammered for hours, short enough that a dead JVM is caught within
#: a couple of episode lengths. The stop wait uses this same interval, so shutdown
#: latency is bounded by it.
_DEFAULT_JVM_POLL_SECONDS: float = 5.0

#: Connect timeout for one JVM probe. Loopback: a live server accepts instantly, and
#: a dead one refuses instantly. The timeout only bounds the pathological case.
_DEFAULT_JVM_PROBE_TIMEOUT_SECONDS: float = 1.0


def jvm_alive(
    host: str = MC_HOST,
    port: int = MC_PORT,
    *,
    timeout: float = _DEFAULT_JVM_PROBE_TIMEOUT_SECONDS,
) -> bool:
    """True while the SHARED Paper JVM accepts TCP connections on its MC port.

    The tier-2 detector of the two-tier fault policy: a ``False`` here is the abort
    trigger for the WHOLE run (see :class:`ActorPool`). Defaults make it callable as
    the contract's ``jvm_alive() -> bool``, and its ``(host, port)`` positional shape
    makes it directly usable as the injectable :data:`JvmProbe`.

    A plain connect-and-close — NOT a Minecraft handshake, NOT a status ping. That is
    enough: the failure this guards against is the JVM being gone (crash, OOM kill,
    operator ``stop``), and a listening socket that is not Paper is an operator error
    the preflight in ``server/setup/start-pads.sh`` already refuses to start on.

    SAFE ONLY ON THE MC PORT. Paper is an ordinary multi-client server, so this costs
    it nothing. ``BridgeServer`` accepts exactly ONE client and DESTROYS the incumbent
    when a second arrives (``bridge/transport.js``), so pointing this at a bridge port
    would kill the collector attached to it. Never do that.

    Args:
        host: Host the JVM listens on. Loopback by default.
        port: The shared Minecraft port (one JVM, so one port).
        timeout: Per-probe connect timeout in seconds.

    Returns:
        ``True`` if the connection was accepted, ``False`` on any ``OSError``
        (refused, unreachable, timed out). Only ``OSError`` is swallowed: anything
        else is a bug in this probe and must surface rather than be read as "dead".
    """
    try:
        with socket.create_connection((host, int(port)), timeout=float(timeout)):
            return True
    except OSError:
        return False


class LaunchAbandoned(RuntimeError):
    """Raised out of :meth:`ShutdownSignal.sleep` to abandon an in-flight launch.

    Not a fault: it means the run is shutting down while an
    :class:`ArenaLauncher` happened to be inside its bounded wait. The collector's
    recovery loop treats it like any other launcher failure and then exits, because
    its own stop flag is set by the same shutdown.
    """


class ShutdownSignal:
    """Shutdown flag shared between the :class:`ActorPool` and an :class:`ArenaLauncher`.

    :meth:`distributed.launcher.SubprocessArenaLauncher.launch` waits — bounded, but
    for up to ~135 s — for a pad's bridge port to free and then to come back up, and
    it does so by polling with an INJECTED ``sleep``. That sleep runs on a collector
    thread. With the stock ``time.sleep`` a shutdown landing inside a relaunch is
    ignored until the whole wait elapses.

    This class is that injection point: pass :meth:`sleep` as the launcher's ``sleep``
    and hand the same object to the pool. The pool sets it in :meth:`ActorPool.stop`
    and :meth:`ActorPool._abort`, so the next poll raises :class:`LaunchAbandoned`
    and ``launch()`` unwinds at once instead of blocking the collector for minutes.
    Raising (rather than returning early) is required: the launcher's loops re-check
    a deadline right after sleeping, so a sleep that merely returned would turn a
    two-minute wait into a two-minute hot spin.

    Thread-safe: it is a :class:`threading.Event` behind two methods.
    """

    def __init__(self, event: Optional[threading.Event] = None) -> None:
        self._event = event if event is not None else threading.Event()

    @property
    def event(self) -> threading.Event:
        """The underlying event (exposed for callers that must wait on it)."""
        return self._event

    def set(self) -> None:
        """Signal shutdown. Idempotent."""
        self._event.set()

    def is_set(self) -> bool:
        """True once shutdown has been signalled."""
        return self._event.is_set()

    def sleep(self, seconds: float) -> None:
        """Sleep ``seconds``, or raise :class:`LaunchAbandoned` if shutdown lands.

        Args:
            seconds: Requested sleep. Non-positive values still check the flag, so a
                zero-length poll cannot slip past a pending shutdown.

        Raises:
            LaunchAbandoned: if shutdown is already signalled, or becomes signalled
                before the sleep elapses.
        """
        if self._event.is_set():
            raise LaunchAbandoned("shutdown signalled: abandoning the bridge launch")
        if self._event.wait(timeout=max(0.0, float(seconds))):
            raise LaunchAbandoned(
                "shutdown signalled while waiting on a bridge launch: abandoning it"
            )


class PoolAbortedError(RuntimeError):
    """Raised when the :class:`ActorPool` aborts the run LOUDLY.

    Under the two-tier fault policy there is exactly ONE cause: the shared Paper JVM
    stopped answering on its Minecraft port. Every pad in the fleet lives inside that
    one JVM, so there are no survivors to continue on — the run must stop rather than
    quietly grind on against a world that no longer exists. A dead pad BRIDGE never
    raises this; it is repaired in place by :meth:`Collector.restart_bridge`.

    Surfaced from :meth:`ActorPool.stop` and :meth:`ActorPool.raise_if_aborted`, with
    a message naming the JVM and the port that was probed.
    """


class ArenaLauncher(Protocol):
    """Starts/stops the BRIDGE process for one pad. It never touches the JVM.

    Under the one-JVM/N-pad topology a pad's only private process is its Node bridge;
    the Paper JVM is shared by the whole fleet and is owned by
    ``server/setup/start-pads.sh``. Killing or restarting it to recover a single pad
    would take every other pad down with it, so this Protocol is deliberately
    bridge-scoped — a dead JVM is the run-aborting tier-2 fault, handled by
    :class:`ActorPool`, not by a relaunch.

    Injectable so all supervisor/fault logic is testable offline against a fake that
    merely records ``launch``/``terminate`` calls;
    :class:`~distributed.launcher.SubprocessArenaLauncher` is the live implementation.
    A restart is REQUESTED here (start the process) but the collector reconnects to
    the bridge itself once the pad is back — the launcher owns process lifecycle, the
    collector owns the single TCP connection.
    """

    def launch(self, arena_id: int) -> None:
        """Start (or restart) the BRIDGE for pad ``arena_id``. The JVM is untouched.

        Slow in production: the live launcher waits for the dying bridge's port to
        free, spawns a replacement, and waits for it to accept a connection (which
        only happens once BOTH of that pad's bots have joined and spawned). The
        collector backs off and then reconnects on a fresh client.

        May raise — the live launcher refuses to spawn a duplicate onto a port some
        other bridge still holds, and refuses to spawn against an unreachable JVM.
        The collector treats any exception as a failed attempt and backs off.
        """
        ...

    def terminate(self, arena_id: int) -> None:
        """Stop pad ``arena_id``'s bridge. Best-effort, idempotent, JVM untouched."""
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

    Fault path (the bridge resilience contract + the two-tier fault policy): a
    :class:`BridgeError` from any of the above aborts THAT episode. The collector
    FIRST tries the env's idempotent ``reset()`` reconnect (bounded); if that
    recovers, it resumes. If not, it marks the pad DEAD and consults the SHARED JVM
    before doing anything else:

      * JVM alive  -> TIER 1: :meth:`restart_bridge` for THIS pad only, back off,
        rebuild the env via ``env_factory`` (a FRESH client to the same pad's
        bridge), confirm with a reset, resume. No other pad is touched.
      * JVM dead   -> TIER 2: fire ``on_jvm_down`` so the pool aborts the WHOLE run
        loudly, and abandon recovery. A bridge restarted into a dead JVM would exit
        immediately, and there is no fleet left to restart it for.

    ``step()`` is NEVER silently retried (that corrupts the episode) — only
    ``reset()`` may reconnect-and-retry, and the env enforces that internally.

    Args:
        arena_id: 0-based index of this collector's pad.
        env_factory: Zero-arg callable building (and connecting) a FRESH env for this
            pad. Called once at start and again on every restart so a reconnect is
            always a fresh client to the same single-connection bridge.
        policy: The acting surface (a :class:`~distributed.weights.SnapshotPolicy`),
            satisfying ``RolloutPolicy``. Its ``arena_id`` should match ``arena_id``.
        transport: The shared actor->learner channel (episodes flow up).
        weight_store: The shared snapshot store the learner publishes to.
        cfg: Training config (ε schedule, per-pad seed scheme, ``fault_relaunch``).
        launcher: The :class:`ArenaLauncher` used to restart this pad's bridge.
        on_state_change: Optional callback ``(arena_id, alive: bool) -> None`` the
            collector fires when it transitions live<->dead, so the pool tracks the
            aggregate live count. Called OUTSIDE any collector-held lock.
        on_jvm_down: Optional callback ``(message: str) -> None`` fired ONCE when the
            JVM probe says the shared Paper JVM is gone. :meth:`ActorPool.start`
            binds the pool's abort to it. With no callback the collector still
            abandons recovery — it must never sit restarting bridges into a dead JVM.
        jvm_probe: Optional :data:`JvmProbe` called as ``probe(mc_host, mc_port)``.
            ``None`` means "no JVM supervision configured" and the collector behaves
            as if the JVM were alive — the right default for offline pools driving
            fake envs, where there is no JVM at all. The live path injects
            :func:`jvm_alive` (see ``agent.train._main_multi_arena``).
        mc_host / mc_port: Where the shared JVM listens. Passed to ``jvm_probe`` AND
            named in the abort message, so the two can never disagree.
        max_episode_steps: Per-episode decision cap forwarded to ``collect_episode``.
        relaunch_backoff_seconds: Initial backoff after a failed reconnect.
        relaunch_backoff_max_seconds: Cap the (doubling) backoff grows toward.
        reset_reconnect_attempts: Bounded env ``reset()`` reconnect attempts before
            declaring the pad dead.
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
        on_jvm_down: Optional[Callable[[str], None]] = None,
        jvm_probe: Optional[JvmProbe] = None,
        mc_host: str = MC_HOST,
        mc_port: int = MC_PORT,
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
        self._on_jvm_down = on_jvm_down
        self._jvm_probe = jvm_probe
        self._mc_host = str(mc_host)
        self._mc_port = int(mc_port)
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

        # Eval pause/handoff (T8). The DESIGNATED-arena collector parks at an EPISODE
        # BOUNDARY when `_pause` is set so a periodic eval can BORROW its idle env /
        # connection (the bridge serves exactly ONE connection — eval must never open
        # a second one). `_paused_idle` is set by the collector ONLY once it has
        # finished its current episode and is parked at the boundary (no reply in
        # flight on the shared connection), so the eval routine can wait on it and
        # then safely reuse `current_env()`. Clearing `_pause` (via :meth:`resume`)
        # un-parks the loop. The pause is checked at the boundary, NEVER mid-episode,
        # so an in-flight episode is never abandoned. All additive: a collector whose
        # pause flag is never set behaves exactly as before.
        self._pause = threading.Event()
        self._paused_idle = threading.Event()

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

    # -- eval pause / handoff (T8) -----------------------------------------
    #
    # A periodic eval in multi-arena mode runs on exactly ONE designated arena and
    # must never open a second connection on ANY arena (the bridge serves one). The
    # protocol: the eval driver calls :meth:`pause`, then :meth:`wait_until_idle` to
    # block until this collector has finished its current episode and parked at the
    # boundary (``paused_idle`` set, so nothing is in flight on the shared socket),
    # then BORROWS :meth:`current_env`'s transport (an ``MCPvPEnv`` built with
    # ``auto_connect=False`` over it — the same idle-connection borrow the single-
    # arena ``_eval_against_dummy`` uses), runs the greedy eval, and finally calls
    # :meth:`resume`. Other arenas keep collecting throughout.

    def pause(self) -> None:
        """Request this collector park at the next EPISODE BOUNDARY (idempotent).

        Does NOT interrupt an in-flight episode: the loop checks the flag only
        between episodes, so a paused-and-idle collector has no reply in flight on
        its single bridge connection and the eval can safely borrow it.
        """
        self._paused_idle.clear()
        self._pause.set()

    def resume(self) -> None:
        """Clear the pause so the collector resumes its rollout loop (idempotent)."""
        self._paused_idle.clear()
        self._pause.clear()

    def wait_until_idle(self, timeout: Optional[float] = None) -> bool:
        """Block until the collector confirms it is paused-and-idle at a boundary.

        Returns ``True`` once the collector has parked between episodes with its
        connection idle (so :meth:`current_env` is safe to borrow), or ``False`` if
        ``timeout`` elapsed first. Call only after :meth:`pause`.
        """
        return self._paused_idle.wait(timeout=timeout)

    @property
    def paused_idle(self) -> bool:
        """True while the collector is parked at a boundary with an idle connection."""
        return self._paused_idle.is_set()

    def current_env(self) -> Optional[EnvProtocol]:
        """Return the collector's live env (or ``None`` if it holds none yet).

        Used by the eval driver to BORROW the idle env's shared transport while this
        collector is paused-and-idle. The borrower must NOT close it or send
        ``close`` (the collector still owns the connection and resumes on it).
        """
        return self._env

    def _wait_while_paused(self) -> None:
        """Park at the episode boundary while pause is requested (stop-interruptible).

        Marks the collector paused-and-idle (so :meth:`wait_until_idle` unblocks the
        eval driver), then waits on the pause flag clearing or a stop. The connection
        is genuinely idle here: the previous episode's last ``step`` already received
        its ``state`` reply and no new wire I/O happens while parked, so the eval can
        borrow the transport without racing a reply. Clears the idle flag on the way
        out so a later borrow can never reuse a stale confirmation.
        """
        if not self._pause.is_set():
            return
        # Confirm parked-and-idle for the eval driver, then wait for resume()/stop.
        self._paused_idle.set()
        try:
            while self._pause.is_set() and not self._stop.is_set():
                # Short, interruptible wait so resume()/stop() wake us promptly without
                # busy-spinning.
                self._stop.wait(timeout=0.01)
        finally:
            # Once unpaused (or stopping) we are about to do wire I/O again, so we are
            # no longer idle-for-borrow. Clear before the next episode.
            self._paused_idle.clear()

    # -- the rollout loop --------------------------------------------------

    def run(self) -> None:
        """Drive the refresh -> collect -> send loop until stopped, recovering faults.

        Builds the env on first entry. A :class:`BridgeError` anywhere in an episode
        is funneled into :meth:`_recover`, which honors the resilience contract
        (idempotent ``reset()`` reconnect first, then the two-tier fault policy:
        bridge restart + backoff + fresh client, or a loud whole-run abort if the
        shared JVM is gone). The loop exits cleanly when :meth:`stop` is signalled or
        when a transport close ends the run; any other exception propagates (a real
        bug should surface, not be swallowed).
        """
        try:
            self._ensure_env()
        except BridgeError:
            # The very first connect failed: treat the pad as dead and try to
            # recover before entering the steady-state loop.
            self._set_alive(False)
            if not self._recover():
                return

        while not self._stop.is_set():
            # Eval pause/handoff boundary (T8): if a periodic eval has requested this
            # (designated) arena pause, park HERE — between episodes — so the eval can
            # borrow the idle env/connection. Checked only at the boundary, so an
            # in-flight episode is never abandoned. A no-op when pause is never set.
            self._wait_while_paused()
            if self._stop.is_set():
                return
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

    # -- the two-tier fault seam (T11) -------------------------------------

    def jvm_alive(self) -> bool:
        """True while the SHARED Paper JVM is reachable (tier-2 detector).

        Delegates to the injected :data:`JvmProbe` with this collector's configured
        ``(mc_host, mc_port)``. With NO probe injected this returns ``True``: an
        offline pool driving fake envs has no JVM to lose, and treating "unconfigured"
        as "dead" would abort every such run on the first bridge fault.

        Returns:
            ``True`` if the JVM answers (or no probe is configured), ``False`` if the
            probe says the port is closed. A single ``False`` is the abort trigger —
            there is deliberately no confirmation retry and no grace period.
        """
        probe = self._jvm_probe
        if probe is None:
            return True
        return bool(probe(self._mc_host, self._mc_port))

    def restart_bridge(self, pad_index: int) -> None:
        """TIER 1: restart the BRIDGE of pad ``pad_index`` and nothing else.

        The narrow half of the two-tier policy. It asks the injected
        :class:`ArenaLauncher` to relaunch exactly one pad's Node bridge process; the
        shared Paper JVM, every other pad's bridge, and this collector's own thread
        are all untouched. The reconnect is NOT done here — the collector rebuilds a
        fresh client itself after backing off, because the bridge serves exactly one
        TCP client and the launcher must not hold that slot.

        The restarted bridge re-runs ``arena:reset_pad`` for its own anchor without
        any help from here: it is spawned with its ``--pad-origin``, it is the sole
        reset authority, and it issues the macro on every ``reset`` — including the
        confirmation reset :meth:`_recover` performs right after reconnecting.

        CALL ONLY FOR A PAD WHOSE CLIENT IS ALREADY GONE. The live launcher's
        ``_wait_for_port_free`` / ``_wait_for_bridge`` CONNECT to the pad's bridge
        port, and ``BridgeServer`` answers a second client by destroying the
        incumbent. The one call site is safe by TIMING, not by construction:
        :meth:`_recover` only reaches here after Step A's reconnect attempts have
        already failed on that connection. Shortening Step A to zero attempts, or
        calling this for a pad other than ``self.arena_id``, would turn the
        launcher's readiness probe into an eviction of a live collector.

        Args:
            pad_index: The pad whose bridge to restart. Non-negative, and in practice
                always this collector's own ``arena_id`` (see the note above).

        Raises:
            ValueError: if ``pad_index`` is negative or not an int. Restarting "pad
                -1" would be a supervisor bug, and the live launcher would reject it
                much later with a confusing message about anchors.
            Exception: whatever the launcher raises (it refuses to duplicate a live
                bridge, and refuses to spawn against an unreachable JVM). The caller
                treats that as a failed attempt and backs off.
        """
        if isinstance(pad_index, bool) or not isinstance(pad_index, int):
            raise ValueError(f"pad_index must be an int, got {pad_index!r}")
        if pad_index < 0:
            raise ValueError(f"pad_index must be >= 0, got {pad_index}")
        try:
            self._launcher.launch(pad_index)
        except LaunchAbandoned:
            # Shutdown landed inside the launcher's bounded wait. We spawned a bridge
            # and then walked away from its readiness gate, so do not orphan it —
            # terminate() only reaches processes this launcher itself owns.
            try:
                self._launcher.terminate(pad_index)
            except Exception:  # noqa: BLE001 - teardown during shutdown is best-effort
                pass
            raise

    def _jvm_down_message(self) -> str:
        """The loud tier-2 abort text, naming the JVM and the port that was probed."""
        return (
            f"actor pool aborting: the SHARED Paper JVM stopped answering on "
            f"minecraft port {self._mc_host}:{self._mc_port} (jvm_alive() probe), "
            f"detected while pad {self.arena_id} was trying to recover. Every pad in "
            f"the fleet lives in that one JVM, so there are no survivors to continue "
            f"on and restarting a pad's bridge would only spawn a process that exits "
            f"immediately. The run stops here rather than training against a world "
            f"that no longer exists."
        )

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
        """Bring the pad back, honoring the resilience contract. Returns liveness.

        Step A — idempotent reconnect: try the env's own ``reset()`` (bounded). It is
        idempotent and carries no in-flight episode state, so a transient drop (e.g. a
        periodic eval briefly stole the single connection) recovers here without
        touching any OS process. A successful reset means the pad is live again.

        Step B — the TWO-TIER fault policy. The reconnect failed, so the pad is
        genuinely dead and we must decide which fault this is. The JVM is checked
        FIRST, on every attempt:

          * JVM dead -> fire ``on_jvm_down`` (the pool aborts the whole run loudly)
            and return ``False``. No bridge is restarted: a bridge spawned against a
            dead JVM exits immediately, and the fleet it belongs to is gone anyway.
          * JVM alive -> :meth:`restart_bridge` for THIS pad only, back off
            (seconds-scale, doubling toward the cap — a bridge relaunch waits for
            both bots to rejoin), drop the old env, rebuild a FRESH env/client via the
            factory, and confirm with a reset. Repeat until the pad is back or
            :meth:`stop` is signalled.

        That confirmation reset is also what re-runs ``arena:reset_pad`` for this
        pad's anchor: the restarted bridge is the sole reset authority and issues the
        macro itself. Nothing here opens a second connection to do it.

        Returns:
            ``True`` if the pad is live again; ``False`` if recovery was abandoned —
            either because the pool signalled stop, or because the JVM is gone (the
            loop should then exit).
        """
        # --- Step A: bounded idempotent reset() reconnect on the SAME env. ---
        env = self._env
        if env is not None:
            for _ in range(max(0, self._reset_reconnect_attempts)):
                if self._stop.is_set():
                    return False
                try:
                    # reset() reconnect-and-retries internally; a clean return means
                    # the bridge is back. Seed deterministically off this pad's next
                    # episode so the resumed stream stays reproducible.
                    seed = arena_episode_seed(self._cfg, self.arena_id, self._local_ep)
                    env.reset(seed=seed)
                except BridgeError:
                    continue
                else:
                    # Recovered without a restart. NOTE: collect_episode will reset()
                    # again at the top of the next episode (idempotent), so this probe
                    # reset does not desync the stream — it only proves the link.
                    self._set_alive(True)
                    return True

        # --- Step B: two-tier fault policy, then reconnect on a fresh client. ---
        backoff = self._relaunch_backoff_seconds
        while not self._stop.is_set():
            # TIER 2 FIRST. A dead JVM is not a pad fault and must never be answered
            # with a bridge restart; it ends the run. Checked regardless of
            # `fault_relaunch`, which gates repair, not the abort.
            if not self.jvm_alive():
                self._report_jvm_down()
                return False

            if self._cfg.fault_relaunch:
                # TIER 1: this pad's bridge only. Best-effort — a launcher failure
                # must not crash the collector (the live launcher raises rather than
                # duplicate a bridge that is still alive on the port), so we back off
                # and try again. A shutdown landing inside the launcher's bounded wait
                # arrives here as LaunchAbandoned and is handled the same way: the
                # stop check right below ends the loop.
                try:
                    self.restart_bridge(self.arena_id)
                except Exception:  # noqa: BLE001 - launcher faults must not kill us
                    pass

            # Back off on a seconds-to-tens-of-seconds scale: a restarted bridge only
            # accepts a client once BOTH of its bots have rejoined and spawned, so
            # hammering reconnect immediately is pointless.
            self._interruptible_sleep(backoff)
            if self._stop.is_set():
                return False
            backoff = min(backoff * 2.0, self._relaunch_backoff_max_seconds)

            # Drop the old (dead) env and rebuild a FRESH client to the SAME pad's
            # single-connection bridge. Never reuse or multiplex a connection.
            self._close_env_quietly()
            try:
                self._env = self._env_factory()
            except BridgeError:
                # Pad not back yet (factory connects eagerly); keep backing off.
                continue

            # Confirm the fresh connection with an idempotent reset probe. This is the
            # reset that re-runs the pad's arena:reset_pad macro after a restart.
            try:
                seed = arena_episode_seed(self._cfg, self.arena_id, self._local_ep)
                self._env.reset(seed=seed)
            except BridgeError:
                continue

            self._set_alive(True)
            return True

        return False

    def _report_jvm_down(self) -> None:
        """Escalate a dead JVM to the pool (tier 2), then let recovery abandon.

        Fires ``on_jvm_down`` when one is bound. With none bound — a standalone
        collector in a test, or one built outside a pool — the collector still stops
        itself, because the alternative is an endless loop restarting bridges into a
        JVM that is not there. Either way the caller returns ``False`` and the rollout
        loop exits.
        """
        callback = self._on_jvm_down
        if callback is not None:
            callback(self._jvm_down_message())
        # Independent of the callback: this collector is done either way.
        self._stop.set()

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
    """Supervises N :class:`Collector` daemons and owns the WHOLE-RUN abort (tier 2).

    ``start()`` starts one collector per pad (each owns its own env, policy, and
    recovery loop), tracks the aggregate live count, and runs a supervisor thread
    whose single job is watching the SHARED Paper JVM.

    A pad going down does NOT stop the pool: that collector restarts its OWN bridge
    out-of-band while every other pad keeps feeding the learner. There is NO survivor
    floor — ``fault_min_live_arenas`` was deleted in T11 — because a floor is a licence
    to keep training on a shrunken fleet, which the plan forbids. The pool aborts for
    exactly one reason: :func:`jvm_alive` reports the shared JVM gone, either from the
    supervisor's periodic probe or from a collector that hit it first during recovery.
    A single ``False`` aborts; there is no grace period and no retry budget.

    ``fault_relaunch=False`` disables tier 1 (no bridge is restarted) as a diagnostic
    mode. It does NOT abort — a pad simply stays down. Use it only when you are
    watching the run.

    The abort is surfaced two ways: :meth:`raise_if_aborted` lets a driver poll, and
    :meth:`stop` raises :class:`PoolAbortedError` on its way out if the run aborted —
    so a run can never quietly end after its world disappeared.

    Args:
        collectors: The collectors to supervise (one per pad). Build them with
            :meth:`build` for the common case, or inject pre-built/fake collectors.
        cfg: Training config (reads ``fault_relaunch``; the ε/seed knobs are the
            collectors' concern).
        jvm_probe: Optional :data:`JvmProbe` called as ``probe(mc_host, mc_port)``.
            ``None`` disables JVM supervision entirely — the correct default for a
            pool of fake envs, which has no JVM. The live path injects
            :func:`jvm_alive`. The same probe is pushed onto every supervised
            collector that has none, so both detectors agree by construction.
        mc_host / mc_port: Where the shared JVM listens; probed and named in the
            abort message.
        jvm_poll_seconds: Supervisor probe interval. Also the shutdown-latency bound
            for the supervisor thread.
        shutdown: Optional :class:`ShutdownSignal` shared with the
            :class:`ArenaLauncher`'s injected ``sleep``. Set by :meth:`stop` and
            :meth:`_abort`, so a collector parked inside a bridge relaunch unwinds
            immediately instead of finishing a two-minute wait nobody is waiting for.
    """

    def __init__(
        self,
        collectors: List[Collector],
        cfg: TrainConfig,
        *,
        jvm_probe: Optional[JvmProbe] = None,
        mc_host: str = MC_HOST,
        mc_port: int = MC_PORT,
        jvm_poll_seconds: float = _DEFAULT_JVM_POLL_SECONDS,
        shutdown: Optional[ShutdownSignal] = None,
    ) -> None:
        self._collectors = list(collectors)
        self._cfg = cfg
        self._jvm_probe = jvm_probe
        self._mc_host = str(mc_host)
        self._mc_port = int(mc_port)
        self._jvm_poll_seconds = float(jvm_poll_seconds)
        self._shutdown = shutdown

        # Aggregate liveness. Seeded from each collector's current state; mutated only
        # via the collectors' on_state_change callback (fired off the collector
        # threads) and read by live_count() — all under this lock. Telemetry now, not
        # an abort trigger: no live count, however low, ends the run by itself.
        self._lock = threading.Lock()
        self._live: Dict[int, bool] = {
            c.arena_id: c.alive for c in self._collectors
        }

        # Abort state. Latched once; the supervisor or a collector sets it,
        # stop()/raise_if_aborted surface it.
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
        jvm_probe: Optional[JvmProbe] = None,
        mc_host: str = MC_HOST,
        mc_port: int = MC_PORT,
        jvm_poll_seconds: float = _DEFAULT_JVM_POLL_SECONDS,
        shutdown: Optional[ShutdownSignal] = None,
    ) -> "ActorPool":
        """Build a pool of ``cfg.arenas`` collectors sharing one transport/store/counter.

        ``env_factory_for(arena_id)`` returns that pad's zero-arg env factory (so a
        restart rebuilds a fresh client to the same bridge); ``policy_for(arena_id)``
        returns that pad's :class:`~distributed.weights.SnapshotPolicy`. The GLOBAL
        ε counter is shared across all collectors (created here if not injected) so the
        ε schedule advances over the combined stream.

        Args:
            cfg: Training config (``arenas`` count + ``fault_relaunch``).
            env_factory_for: ``arena_id -> EnvFactory`` (fresh-client builder).
            policy_for: ``arena_id -> RolloutPolicy`` (per-pad snapshot policy).
            transport: Shared actor->learner channel.
            weight_store: Shared snapshot store.
            launcher: Shared :class:`ArenaLauncher` (bridge lifecycle only).
            counter: Optional shared :class:`GlobalEpisodeCounter`; created if ``None``.
            max_episode_steps / relaunch_backoff_* / reset_reconnect_attempts / sleep:
                Forwarded to every :class:`Collector` (see its docstring).
            jvm_probe / mc_host / mc_port: The tier-2 detector, given to the pool AND
                to every collector so the two agree on what "the JVM" means.
            jvm_poll_seconds / shutdown: See :class:`ActorPool`.

        Returns:
            A constructed :class:`ActorPool` (not yet started).
        """
        shared_counter = counter if counter is not None else GlobalEpisodeCounter()

        # on_state_change / on_jvm_down are left None here and bound in start(): the
        # pool wires its own callbacks onto every collector (built or injected) when it
        # starts, so there is no need to reference a half-constructed pool from a
        # closure. The JVM probe IS passed here (it is plain data, not a back-reference).
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
                jvm_probe=jvm_probe,
                mc_host=mc_host,
                mc_port=mc_port,
                max_episode_steps=max_episode_steps,
                relaunch_backoff_seconds=relaunch_backoff_seconds,
                relaunch_backoff_max_seconds=relaunch_backoff_max_seconds,
                reset_reconnect_attempts=reset_reconnect_attempts,
                sleep=sleep,
            )
            for arena_id in range(cfg.arenas)
        ]
        return cls(
            collectors,
            cfg,
            jvm_probe=jvm_probe,
            mc_host=mc_host,
            mc_port=mc_port,
            jvm_poll_seconds=jvm_poll_seconds,
            shutdown=shutdown,
        )

    # -- liveness bookkeeping ----------------------------------------------

    def _mark(self, arena_id: int, alive: bool) -> None:
        """Record a pad's live/dead transition (collector-thread callback)."""
        with self._lock:
            self._live[arena_id] = alive

    def live_count(self) -> int:
        """Number of pads currently holding a working connection.

        TELEMETRY, not a fault trigger. Under the two-tier policy a low live count
        means "some pads are being repaired", which is a normal transient state; only
        a dead JVM ends the run.
        """
        with self._lock:
            return sum(1 for alive in self._live.values() if alive)

    def collector_for(self, arena_id: int) -> Optional[Collector]:
        """Return the supervised :class:`Collector` for ``arena_id`` (or ``None``).

        Used by the eval pause/handoff driver (T8) to pause exactly ONE designated
        arena and borrow its idle env/transport for a greedy eval. Read-only access
        to a supervised collector; does not alter pool state.
        """
        for collector in self._collectors:
            if collector.arena_id == arena_id:
                return collector
        return None

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Start every collector daemon and the supervisor that watches the JVM."""
        # Bind the pool's callbacks onto any collectors that were injected without
        # them (so externally-built/fake collectors still update the aggregate count
        # and can still escalate a dead JVM), and push the pool's JVM probe onto any
        # collector that has none.
        for collector in self._collectors:
            if collector._on_state_change is None:
                collector._on_state_change = self._mark
            if collector._on_jvm_down is None:
                collector._on_jvm_down = self._abort
            if collector._jvm_probe is None and self._jvm_probe is not None:
                collector._jvm_probe = self._jvm_probe
            # THE POOL OWNS WHAT "THE JVM" MEANS. One JVM serves the whole fleet, so
            # a supervised collector never has a different one to watch — and a
            # collector left on the default port while the pool watched another would
            # probe the wrong port and then name the wrong port in its abort. Aligning
            # here covers externally-built collectors; build() already passes these.
            collector._mc_host = self._mc_host
            collector._mc_port = self._mc_port
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
        """Watch the SHARED Paper JVM; abort the WHOLE run the moment it is gone.

        This is the tier-2 watchdog, and the pool's only reason to abort. It exists
        alongside the collectors' own pre-restart probe because the two catch
        different timings: a collector notices only when ITS pad faults, while a JVM
        that dies between episodes would otherwise go unnoticed until every pad had
        independently failed.

        The loop is a no-op when no probe is configured (an offline pool of fake envs
        has no JVM to watch), and a single ``False`` aborts — no confirmation retry,
        because a survivor policy is exactly what this task deleted.

        THE WATCHDOG MAY NOT DIE QUIETLY. This runs on a daemon thread, so an
        unexpected exception out of the probe would otherwise end the thread, leave
        ``aborted()`` False forever, and let the run continue completely unsupervised
        — the one failure mode this design exists to prevent, arrived at silently.
        The shipped :func:`jvm_alive` swallows only ``OSError`` and so cannot trigger
        this, but a custom probe can, so a watchdog failure is itself an abort AND is
        re-raised so the thread excepthook prints it. Loud twice, silent never.
        """
        probe = self._jvm_probe
        if probe is None:
            return
        try:
            while not self._stopping.is_set():
                if not probe(self._mc_host, self._mc_port):
                    self._abort(
                        f"actor pool aborting: the SHARED Paper JVM stopped answering "
                        f"on minecraft port {self._mc_host}:{self._mc_port} "
                        f"(jvm_alive() probe). Every pad in the fleet lives in that "
                        f"one JVM, so there are no survivors to continue on. The run "
                        f"stops here rather than training against a world that no "
                        f"longer exists."
                    )
                    return
                # Poll on the JVM cadence; the wait returns at once on stop().
                if self._stopping.wait(timeout=self._jvm_poll_seconds):
                    return
        except BaseException as exc:  # noqa: BLE001 - see the docstring: never silent
            self._abort(
                f"actor pool aborting: the JVM watchdog itself failed while probing "
                f"minecraft port {self._mc_host}:{self._mc_port} "
                f"({type(exc).__name__}: {exc}). Tier 2 of the fault policy is no "
                f"longer being enforced, and a run that is not watching its JVM must "
                f"not keep training. Fix the probe, not this abort."
            )
            raise

    def _abort(self, message: str) -> None:
        """Latch the abort error and signal shutdown (idempotent).

        Also the ``on_jvm_down`` callback bound onto every collector, so a pad that
        discovers the dead JVM during recovery aborts the run through the same path
        as the supervisor. Safe to call from any thread.
        """
        if self._aborted.is_set():
            return
        self._abort_error = PoolAbortedError(message)
        self._aborted.set()
        # Tell every collector to wind down — the run is over.
        self._stopping.set()
        # Release anything parked inside a launcher's bounded relaunch wait, so the
        # abort is not held up by a bridge that is never coming back.
        if self._shutdown is not None:
            self._shutdown.set()
        for collector in self._collectors:
            collector.stop()

    def raise_if_aborted(self) -> None:
        """Raise :class:`PoolAbortedError` if the pool has aborted (else no-op).

        Lets a driver poll the pool's health between learner steps and stop the run
        loudly the moment the shared JVM is gone.
        """
        if self._aborted.is_set() and self._abort_error is not None:
            raise self._abort_error

    def aborted(self) -> bool:
        """True once the pool has aborted (i.e. the shared Paper JVM died)."""
        return self._aborted.is_set()

    def stop(self) -> None:
        """Stop all collectors and the supervisor cleanly, then surface any abort.

        Idempotent. Signals every collector to wind down after its current episode,
        releases any in-flight bridge relaunch, joins the supervisor, and — if the
        pool aborted on a dead JVM — raises :class:`PoolAbortedError` so a run whose
        world disappeared can never end silently.

        Raises:
            PoolAbortedError: if the shared Paper JVM died during the run.
        """
        self._stopping.set()
        # Before joining anything: unblock a collector parked inside a launcher's
        # bounded wait (up to ~135 s with a plain time.sleep), so shutdown is not
        # hostage to a bridge relaunch nobody is waiting for any more.
        if self._shutdown is not None:
            self._shutdown.set()
        for collector in self._collectors:
            collector.stop()
        # Join the supervisor first (it no longer needs to poll), then the collectors.
        if self._supervisor is not None:
            self._supervisor.join(timeout=1.0)
        for collector in self._collectors:
            collector.join(timeout=1.0)

        self.raise_if_aborted()
