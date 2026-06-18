"""learner - the decoupled learner thread; the SOLE mutator of the replay buffer.

Route-1 (single-process) learner side of the Ape-X-lite seam (plan SS Decisions).
N collector threads roll whole :class:`~distributed.serialization.Episode` objects
and push them onto an :class:`~distributed.transport.ExperienceTransport`; this
one learner thread drains that channel, feeds replay, runs gradient steps at its
OWN rate, and republishes weight snapshots to the collectors.

Why a decoupled (IMPALA-shaped) learner instead of a synchronous
drain-then-learn loop: collectors produce continuously and at a different rate
than the learner can train. Coupling the two would either stall collection while
the learner steps or stall the learner waiting on collection. The learner instead
drains whatever episodes are ready, adds them to replay, and steps ``learn()`` on
its own clock; weights flow back to collectors every ``K`` grad steps via the
:class:`~distributed.weights.WeightStore`.

Why this class is the ONLY thing allowed to touch ``trainer.replay``:
:class:`~agent.replay.PrioritizedSequenceReplay` is pure-NumPy with NO internal
locks. ``add_episode`` (here), ``sample_sequences`` and ``update_priorities``
(inside ``trainer.learn()``) all mutate the sum-tree in place. A second concurrent
mutator (e.g. a collector that kept a replay reference) would corrupt the sum-tree
SILENTLY -- no crash, just wrong priorities and a poisoned sampler. So every replay
mutation is funnelled onto this single thread; collectors hold no replay reference
and only ever produce Episodes.

Why a watchdog: the learner is the run's single point of liveness. If it wedges
between passes (an exception a caller swallowed, a WARM buffer whose grad steps
stop advancing) while collectors keep producing, episodes pile up in the transport
forever and the run silently makes no progress. The :class:`LearnerWatchdog` below
samples (episodes received, grad steps taken) over consecutive drain passes and
trips LOUDLY when the WARM-buffer backlog keeps growing while gradient progress is
static -- it would rather abort the whole run than let it collect into a buffer no
one drains. It does NOT catch a hang INSIDE a single ``trainer.learn()`` call:
sampling is cooperative (once per pass), so a pass that never returns is never
sampled. Cold-start warm-up (buffer below ``min_replay``, grad_step pinned at 0) is
healthy by definition and is explicitly excluded from the stall condition.

Owner: T6 (multi-arena throughput track, issue #4)
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Optional

from distributed.serialization import Episode
from distributed.transport import ExperienceTransport, TransportError
from distributed.weights import WeightStore

__all__ = ["LearnerError", "WatchdogError", "LearnerWatchdog", "LearnerLoop"]


# How many episodes the learner pulls off the transport per drain pass before it
# runs a gradient step, when more than one is already waiting. Bounding the drain
# keeps the learn() cadence interleaved with collection (so a sudden flood of
# episodes cannot starve gradient steps) while still emptying a healthy backlog.
_DEFAULT_DRAIN_BATCH = 16

# Default watchdog timing. The watchdog samples the (received, grad_step) pair
# every interval and trips after this many consecutive "backlog grew, grad stalled"
# samples. Both are injectable so tests drive the trip deterministically (by calling
# check() directly) with no real sleeps in the hot path.
_DEFAULT_WATCHDOG_INTERVAL_S = 5.0
_DEFAULT_WATCHDOG_PATIENCE = 3


class LearnerError(RuntimeError):
    """Raised (or stored) when the learner thread aborts the run.

    Carried back to the supervising thread via :attr:`LearnerLoop.error` so the
    caller can fail the whole run loudly rather than discover a dead learner only
    by noticing the transport backed up.
    """


class WatchdogError(LearnerError):
    """Raised when the :class:`LearnerWatchdog` trips on a stalled drain.

    The learner is wedged between passes: with the replay buffer WARM (>=
    ``min_replay``, so ``learn()`` should be making progress), the transport backlog
    kept growing (episodes are still being received / queued) while
    ``trainer.grad_step`` did not advance for the configured patience window. This is
    unrecoverable from inside the loop, so the watchdog aborts the run loudly rather
    than let it collect forever. Note this is a BETWEEN-pass stall: a hang inside a
    single ``learn()`` call is never sampled and so cannot trip the watchdog.
    """


class LearnerWatchdog:
    """Trips loudly on a between-pass stall: WARM backlog grows while grad steps stall.

    Liveness guard for the single learner thread (plan Error Handling: "the learner
    thread dies -> a watchdog detects the stalled drain (queue growing, no grad
    progress) and aborts the run loudly -- never silently collect into a buffer no
    one drains"). It detects only the BETWEEN-pass stall it actually samples: a hang
    INSIDE a single ``trainer.learn()`` call is never sampled (sampling is
    cooperative, once per drain pass) and so cannot trip it.

    Trip condition (sampled per :meth:`check`): across ``patience`` CONSECUTIVE
    samples, the replay buffer was WARM (``warm`` True -- ready to learn, so
    ``learn()`` SHOULD be advancing) AND the count of episodes the learner has
    received off the transport STRICTLY INCREASED (new work keeps arriving) AND the
    learner's ``grad_step`` did NOT advance at all (no training progress). One sample
    that breaks any part -- buffer not warm, grad_step advanced, or no new episodes
    arrived -- resets the streak, so a healthy learner never trips: in particular a
    COLD start (buffer below ``min_replay``, so ``learn()`` is a no-op and grad_step
    stays 0 while collectors keep pushing) is excluded by the ``warm`` gate and is
    NOT a stall. When the streak reaches ``patience`` the watchdog records a
    :class:`WatchdogError` and sets its abort :class:`threading.Event`.

    The watchdog is deliberately decoupled from wall-clock timing in its logic: it
    counts SAMPLES, not seconds. A production monitor drives :meth:`check` every
    ``interval_s`` seconds (so a real wedge trips after ~``patience * interval_s``);
    a test drives :meth:`check` directly and deterministically with no sleeps.

    Args:
        patience: Consecutive "backlog grew, grad stalled" samples required to trip
            (>= 1). Larger tolerates longer healthy idle/warm-up gaps before firing.
        interval_s: Seconds a wall-clock monitor should wait between :meth:`check`
            calls. Stored for the monitor's use; the trip LOGIC never reads it, so a
            test can ignore it entirely. Must be > 0.

    Attributes:
        abort_event: Set the moment the watchdog trips, so the learner loop (and any
            other waiter) can observe the abort and stop.
        error: The :class:`WatchdogError` recorded on trip, else ``None``.
    """

    def __init__(
        self,
        *,
        patience: int = _DEFAULT_WATCHDOG_PATIENCE,
        interval_s: float = _DEFAULT_WATCHDOG_INTERVAL_S,
    ) -> None:
        if patience < 1:
            raise ValueError(f"patience must be >= 1, got {patience}")
        if interval_s <= 0.0:
            raise ValueError(f"interval_s must be > 0, got {interval_s}")

        self.patience = int(patience)
        self.interval_s = float(interval_s)

        self.abort_event = threading.Event()
        self.error: Optional[WatchdogError] = None

        # Last observed (received episodes, grad steps). None until the first sample
        # establishes a baseline; the first check only records, it cannot trip.
        self._last_received: Optional[int] = None
        self._last_grad_step: Optional[int] = None
        # Consecutive samples that satisfied the trip condition.
        self._stall_streak = 0

    def reset(self) -> None:
        """Clear all state so the watchdog can be reused for a fresh run.

        Drops the abort flag, the recorded error, the sampled baseline, and the
        stall streak. Call before reusing one watchdog instance across runs.
        """
        self.abort_event.clear()
        self.error = None
        self._last_received = None
        self._last_grad_step = None
        self._stall_streak = 0

    def check(self, received: int, grad_step: int, warm: bool) -> bool:
        """Take one liveness sample; return ``True`` once the watchdog has tripped.

        The first call only records a baseline (it cannot trip with nothing to
        compare against). Each later call compares against the previous sample:

          * warm           := the buffer is ready to learn (``len(replay) >=
            min_replay``), so ``learn()`` SHOULD be advancing ``grad_step``
          * backlog grew  := ``received > last_received`` (new episodes arrived)
          * grad stalled   := ``grad_step <= last_grad_step`` (no training progress)

        When ALL THREE hold the stall streak increments; otherwise it resets to zero.
        The ``warm`` gate is what keeps a cold start healthy: while the buffer is
        below ``min_replay`` the learner pins ``grad_step`` at 0 BY DESIGN even as
        collectors keep pushing, so a cold pass is excluded from the stall condition
        (it resets the streak) rather than counted as a wedge. Reaching ``patience``
        records a :class:`WatchdogError` and sets :attr:`abort_event`. Once tripped
        the watchdog stays tripped (idempotent): every later call returns ``True``
        without re-evaluating.

        Args:
            received: Total episodes the learner has pulled off the transport so far
                (monotonically non-decreasing).
            grad_step: The learner's ``trainer.grad_step`` (monotonically
                non-decreasing; advances once per successful ``learn()``).
            warm: Whether the replay buffer is ready to learn at this sample
                (``trainer.ready_to_learn()``). A not-warm sample can never accrue
                the stall streak -- a cold buffer filling up normally is healthy.

        Returns:
            ``True`` if the watchdog has tripped (now or earlier), else ``False``.
        """
        if self.abort_event.is_set():
            return True

        if self._last_received is None or self._last_grad_step is None:
            # First sample: establish the baseline, cannot trip yet.
            self._last_received = int(received)
            self._last_grad_step = int(grad_step)
            return False

        backlog_grew = received > self._last_received
        grad_stalled = grad_step <= self._last_grad_step

        if warm and backlog_grew and grad_stalled:
            self._stall_streak += 1
        else:
            # Cold buffer (warm-up), training advanced, or no new work arrived ->
            # healthy; reset.
            self._stall_streak = 0

        self._last_received = int(received)
        self._last_grad_step = int(grad_step)

        if self._stall_streak >= self.patience:
            self.error = WatchdogError(
                "learner watchdog tripped: transport backlog grew while grad_step "
                f"stalled for {self.patience} consecutive samples "
                f"(received={received}, grad_step={grad_step}). The learner is "
                "wedged and the run is aborting rather than collecting into a "
                "buffer no one drains."
            )
            self.abort_event.set()
            return True

        return False


class LearnerLoop:
    """Decoupled learner thread: drains episodes, feeds replay, steps, publishes.

    The ONLY mutator of ``trainer.replay`` in the whole system (see the module
    docstring). It owns three responsibilities on one thread:

      1. Drain episodes from the ``transport`` and ``add_episode`` them to replay.
      2. Step ``trainer.learn()`` at the learner's own rate (decoupled from
         collection); ``learn()`` is a no-op until the buffer is warm.
      3. Publish ``trainer.online.state_dict()`` to the ``weight_store`` every
         ``cfg.weight_sync_every_k_steps`` grad steps, so collectors act on a fresh
         snapshot. An INITIAL snapshot is published at version 0 BEFORE the loop so
         collectors have weights to act with immediately.

    Threading model: this class is thread-agnostic. :meth:`run` is a plain blocking
    method; production runs it on a ``threading.Thread`` (T8 wires that), tests call
    it inline or on a thread. Everything it touches (``trainer``, ``transport``,
    ``weight_store``, ``cfg``) is injected so it is fully fakeable.

    Shutdown / error surfacing: :meth:`run` returns cleanly when the transport is
    closed and drained (``recv`` raises :class:`~distributed.transport.TransportError`)
    OR when :meth:`stop` is called. On any other exception it stores the error in
    :attr:`error`, sets :attr:`stopped`, and re-raises so the thread does not die
    silently; the supervising thread reads :attr:`error` to abort the whole run. The
    :class:`LearnerWatchdog`, when attached, is the liveness guard for a wedge that
    raises no exception of its own.

    Args:
        trainer: The :class:`~agent.train.Trainer` whose replay/nets/optimizer this
            loop drives. Only its PUBLIC surface is used: ``replay.add_episode``,
            ``learn()``, ``grad_step``, ``online.state_dict()``, ``cfg``.
        transport: The actor->learner channel (:class:`ExperienceTransport`). Its
            ``recv()`` blocks for an episode and raises ``TransportError`` once the
            channel is closed and drained.
        weight_store: The publish/subscribe handoff collectors read from.
        cfg: The training config; ``weight_sync_every_k_steps`` sets the publish
            cadence. (PER beta annealing is handled INSIDE ``trainer.learn()`` off the
            global ``grad_step`` -- this loop does NOT re-anneal.)
        watchdog: Optional liveness guard. When given, :meth:`run` samples it after
            each drain+learn pass and aborts loudly if it trips. ``None`` disables
            the watchdog (e.g. a test that only exercises the drain/publish path).
        drain_batch: Max episodes pulled per drain pass before a gradient step, when
            the transport offers a non-blocking peek (>= 1). Bounds how far a flood
            of episodes can delay a ``learn()`` step.
        log: Optional ASCII-only ``str -> None`` sink for lifecycle/abort lines
            (Windows cp1252-safe; no unicode glyphs). ``None`` silences logging.

    Attributes:
        version: The next :class:`WeightStore` version to publish (0-based; equals
            the number of publishes performed, so it bumps once per publish).
        received: Total episodes pulled off the transport (read by the watchdog).
        error: The exception that aborted the loop, else ``None``.
        stopped: Set once :meth:`run` exits (cleanly or by error) or :meth:`stop`
            is requested.
    """

    def __init__(
        self,
        trainer: Any,
        transport: ExperienceTransport,
        weight_store: WeightStore,
        cfg: Any,
        *,
        watchdog: Optional[LearnerWatchdog] = None,
        drain_batch: int = _DEFAULT_DRAIN_BATCH,
        log: Optional[Callable[[str], None]] = None,
    ) -> None:
        if drain_batch < 1:
            raise ValueError(f"drain_batch must be >= 1, got {drain_batch}")

        self._trainer = trainer
        self._transport = transport
        self._weight_store = weight_store
        self._cfg = cfg
        self._watchdog = watchdog
        self._drain_batch = int(drain_batch)
        self._log = log

        # Publish cadence (learner grad steps between weight pushes). Validated by
        # TrainConfig (>= 1); guard anyway so a fake cfg cannot divide-by-zero.
        k = int(getattr(cfg, "weight_sync_every_k_steps"))
        if k < 1:
            raise ValueError(f"weight_sync_every_k_steps must be >= 1, got {k}")
        self._k = k

        # _stop is the cooperative shutdown signal; _stopped latches once run() has
        # exited so a caller can poll completion without joining the thread.
        self._stop = threading.Event()
        self._stopped = threading.Event()

        # Monotone publish version: starts at 0 (the initial pre-loop snapshot), then
        # bumps once per publish so collectors' strictly-greater reload gate advances.
        self.version = 0
        # Total episodes received off the transport (the watchdog's backlog signal).
        self.received = 0
        # The grad_step at which we last published, so we publish once per K-boundary
        # crossing even when several grad steps happen between drains.
        self._last_publish_grad_step = 0
        self.error: Optional[BaseException] = None

    # ------------------------------------------------------------------
    # Lifecycle control
    # ------------------------------------------------------------------
    @property
    def stopped(self) -> bool:
        """True once :meth:`run` has exited or a stop was requested."""
        return self._stopped.is_set()

    def stop(self) -> None:
        """Request a cooperative shutdown of the loop. Idempotent and thread-safe.

        Sets the stop flag; the loop checks it at the top of every iteration and
        between drained episodes, then returns cleanly. Does NOT close the transport
        (the caller owns the transport lifecycle) -- closing it is the OTHER clean
        way to end the loop. Safe to call from any thread.
        """
        self._stop.set()

    def _emit(self, message: str) -> None:
        """Forward one ASCII-only line to the injected log sink, if any."""
        if self._log is not None:
            self._log(message)

    def _should_stop(self) -> bool:
        """True if a stop was requested or the watchdog has tripped."""
        if self._stop.is_set():
            return True
        if self._watchdog is not None and self._watchdog.abort_event.is_set():
            return True
        return False

    # ------------------------------------------------------------------
    # The loop
    # ------------------------------------------------------------------
    def run(self) -> None:
        """Run the drain -> add_episode -> learn -> publish loop until shutdown.

        Publishes the initial snapshot (version 0) FIRST so collectors can act
        immediately, then loops: drain ready episodes into replay, run gradient
        steps, publish weights every ``K`` steps, and sample the watchdog. Returns
        cleanly when the transport is closed and drained or :meth:`stop` is called.

        Raises:
            WatchdogError: If the attached watchdog trips on a stalled drain.
            BaseException: Any error from the trainer/transport is stored in
                :attr:`error` and re-raised (the learner never dies silently).
        """
        self._stopped.clear()
        self.error = None
        try:
            # Publish BEFORE the loop so collectors have a real snapshot to act with
            # from their very first episode (rather than acting on a fresh-init clone
            # until the first K-boundary). This is version 0; the next publish bumps.
            self._publish_initial_snapshot()

            while not self._should_stop():
                # 1) Drain whatever episodes are ready (at least one, blocking, when
                # the queue is empty) into replay. Returns False on a clean close.
                if not self._drain_into_replay():
                    break  # transport closed and drained -> clean shutdown.

                if self._should_stop():
                    break

                # 2) Step the learner at its OWN rate. learn() is a no-op until the
                # buffer is warm; once warm we run one gradient step per drained
                # pass. We do not over-train on a stale buffer here -- a single step
                # per pass keeps the update-to-data ratio bounded and interleaves
                # cleanly with collection.
                self._learn_once()

                # 3) Publish a fresh snapshot every K grad steps (after stepping, so
                # the snapshot reflects the latest weights).
                self._maybe_publish()

                # 4) Liveness: sample the watchdog with the latest (received,
                # grad_step). It trips loudly if the backlog grows while grad stalls.
                self._sample_watchdog()

            # If the watchdog tripped, surface it as the loop's error before exit so
            # the caller aborts the run rather than treating this as a clean stop.
            if self._watchdog is not None and self._watchdog.error is not None:
                raise self._watchdog.error
        except BaseException as exc:  # noqa: BLE001 -- surface ANY learner failure.
            # Store and re-raise: a learner that dies must not do so silently. The
            # supervising thread reads ``self.error`` to abort the whole run.
            self.error = exc
            self._emit(f"[learner] aborting: {type(exc).__name__}: {exc}")
            raise
        finally:
            self._stopped.set()

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------
    def _publish_initial_snapshot(self) -> None:
        """Publish version 0 so collectors have weights before the loop starts."""
        self._weight_store.publish(self._trainer.online.state_dict(), self.version)
        self._emit(f"[learner] published initial weights version {self.version}")
        self.version += 1
        self._last_publish_grad_step = int(self._trainer.grad_step)

    def _drain_into_replay(self) -> bool:
        """Pull ready episodes off the transport and add them to replay.

        Always consumes AT LEAST one episode, blocking on ``transport.recv()`` until
        one arrives (so an empty queue parks the thread instead of busy-spinning).
        If the transport additionally exposes a non-blocking peek (a ``pending()`` or
        ``qsize()`` accessor), drains up to ``drain_batch - 1`` more already-queued
        episodes in the same pass via ``try_recv()`` -- emptying a healthy backlog
        without one ``learn()`` step per episode. Falls back to exactly one episode
        per pass when the transport offers only the blocking ``recv`` from the ABC.

        Returns:
            ``True`` after adding one or more episodes; ``False`` if the transport
            was closed and drained (a clean end-of-stream).
        """
        # Blocking receive for the first episode. recv() itself parks the thread, so
        # we re-check the stop flag after it returns rather than busy-waiting.
        try:
            episode = self._transport.recv()
        except TransportError:
            self._emit("[learner] transport closed; draining complete")
            return False

        self._add_episode(episode)

        # Opportunistically drain more episodes that are ALREADY queued, without
        # blocking, when the transport supports a non-blocking peek + pull. This is
        # duck-typed so it works with the production LocalTransport (whose queue has
        # qsize) and a fake test transport, and degrades to one-per-pass otherwise.
        pending = self._pending()
        if pending is not None:
            budget = self._drain_batch - 1
            while budget > 0 and pending > 0 and not self._should_stop():
                extra = self._try_recv()
                if extra is None:
                    break  # nothing ready (or unsupported) -> stop opportunistic drain.
                self._add_episode(extra)
                budget -= 1
                pending = self._pending()
        return True

    def _add_episode(self, episode: Episode) -> None:
        """Add one episode to replay (the single, thread-confined replay mutation).

        Skips an empty episode (no transitions) for the same reason
        ``Trainer.collect_episode`` does: ``add_episode`` has nothing to store and
        the buffer guards against zero-length sequences upstream.
        """
        self.received += 1
        if episode.transitions:
            self._trainer.replay.add_episode(
                episode.transitions, hidden_states=episode.hidden_states
            )

    def _learn_once(self) -> None:
        """Run one gradient step. No-op (returns None) until the buffer is warm.

        ``trainer.learn()`` owns sampling, the loss, backward, the optimizer step,
        priority updates, the soft target update, PER beta annealing, and advancing
        ``trainer.grad_step``. This loop neither samples nor anneals itself -- it
        only PACES the calls (sole-mutator discipline: all replay reads/writes stay
        on this thread).
        """
        self._trainer.learn()

    def _maybe_publish(self) -> None:
        """Publish a fresh weight snapshot when grad_step crossed a K boundary.

        Publishes once per K-boundary crossing using the CURRENT grad_step floor:
        as long as ``grad_step // K`` advanced past the last publish point, push a
        new snapshot and bump :attr:`version`. Driving off ``grad_step // K`` (rather
        than "+= 1 each call") makes the cadence exact even if several grad steps
        landed between drains, and keeps the published version monotone (TC7).
        """
        grad_step = int(self._trainer.grad_step)
        # The number of K-boundaries we have crossed since the initial publish.
        if grad_step // self._k > self._last_publish_grad_step // self._k:
            self._weight_store.publish(
                self._trainer.online.state_dict(), self.version
            )
            self._emit(
                f"[learner] published weights version {self.version} "
                f"at grad_step {grad_step}"
            )
            self.version += 1
            self._last_publish_grad_step = grad_step

    def _sample_watchdog(self) -> None:
        """Feed one liveness sample to the watchdog (no-op when none attached).

        Exactly ONE sample per drain pass, regardless of how many episodes the pass
        drained. ``warm`` (``trainer.ready_to_learn()``) gates the stall streak so a
        cold buffer filling up (grad_step pinned at 0 by design) is never a wedge.
        """
        if self._watchdog is None:
            return
        warm = bool(self._trainer.ready_to_learn())
        self._watchdog.check(self.received, int(self._trainer.grad_step), warm)

    # ------------------------------------------------------------------
    # Optional non-blocking transport surface (duck-typed)
    # ------------------------------------------------------------------
    def _pending(self) -> Optional[int]:
        """Best-effort count of episodes already queued on the transport.

        Prefers an explicit ``pending()`` accessor; falls back to ``qsize()`` (the
        production ``LocalTransport`` wraps a ``queue.Queue``, whose ``qsize`` is an
        approximate-but-fine hint for an opportunistic drain). Returns ``None`` when
        the transport exposes neither -- then the loop drains exactly one episode
        per pass via the blocking ``recv`` and never relies on a peek.
        """
        for attr in ("pending", "qsize"):
            accessor = getattr(self._transport, attr, None)
            if callable(accessor):
                try:
                    return int(accessor())
                except Exception:  # noqa: BLE001 -- a peek must never break the loop.
                    return None
        return None

    def _try_recv(self) -> Optional[Episode]:
        """Non-blocking receive of one already-queued episode, or ``None``.

        Uses the transport's ``try_recv()`` when present (a fake test transport may
        offer it). The production ``LocalTransport`` exposes only the blocking
        ``recv`` from the ABC, so this returns ``None`` there and the loop stays on
        the one-episode-per-pass path -- correct, just less eager about draining a
        backlog in a single pass.
        """
        try_recv = getattr(self._transport, "try_recv", None)
        if callable(try_recv):
            try:
                return try_recv()
            except TransportError:
                return None
            except Exception:  # noqa: BLE001 -- treat any peek failure as "nothing".
                return None
        return None
