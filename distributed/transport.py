"""transport — the actor->learner channel for thread-per-arena collection.

The transfer unit is one whole :class:`~distributed.serialization.Episode`, not a
sampled batch dict. Replay and PER stay centralized on the single learner thread:
collectors only produce episodes, and the learner is the sole replay mutator, so
the channel never carries a sampled batch. A collector calls :meth:`ExperienceTransport.send`
once per rolled episode; the learner calls :meth:`ExperienceTransport.recv` in a
drain loop and feeds replay itself.

:class:`LocalTransport` is the Route-1 (single-process) implementation over
``queue.Queue``. It passes the :class:`Episode` BY REFERENCE — it does NOT call
``Episode.to_dict`` / ``from_dict``. Those lower every numpy array to nested lists
and rebuild them on the far side; in-process that round-trip is a numpy->list->numpy
copy per episode for nothing. ``to_dict`` / ``from_dict`` stay dormant until the
Route-2 networked transport (a separate implementation) needs a wire form.

Owner: T2 (multi-arena throughput track, issue #4)
"""

from __future__ import annotations

import abc
import queue

from distributed.serialization import Episode

__all__ = ["ExperienceTransport", "LocalTransport", "TransportError"]


class TransportError(RuntimeError):
    """Raised when a transport operation fails (channel closed, etc.)."""


class ExperienceTransport(abc.ABC):
    """Abstract actor->learner channel; the transfer unit is one :class:`Episode`.

    Replay and PER live on the single learner thread, so the channel moves whole
    collected episodes, never a sampled batch dict. Implementations carry one
    :class:`Episode` per :meth:`send`, hand it back in FIFO order from :meth:`recv`,
    and must be safe for many producer threads (the collectors) and one consumer
    thread (the learner).
    """

    @abc.abstractmethod
    def send(self, episode: Episode) -> None:
        """Publish one collected episode from a collector to the learner.

        Args:
            episode: The :class:`Episode` a collector rolled against its arena.

        Raises:
            TransportError: If the transport has been closed.
        """

    @abc.abstractmethod
    def recv(self) -> Episode:
        """Receive the next episode on the learner side, blocking until one arrives.

        Returns:
            The next :class:`Episode` in send order.

        Raises:
            TransportError: If the transport has been closed and the stream is drained.
        """

    @abc.abstractmethod
    def close(self) -> None:
        """Shut the transport down and unblock any waiting :meth:`recv`. Idempotent."""


# Module-level sentinel placed on the queue by close() to wake a blocked recv().
# A unique object() is never a valid Episode, so the consumer cannot confuse it
# with real data.
_CLOSED = object()

# How long recv() blocks on a single get() before re-checking the _closed flag.
# This bounded wait lets close() stay non-blocking on a full bounded queue: even
# if the sentinel is dropped (no free slot), the consumer still wakes to see the
# latched flag within this interval rather than hanging forever.
_RECV_POLL_SECONDS = 0.1


class LocalTransport(ExperienceTransport):
    """Single-process actor->learner channel over ``queue.Queue``.

    The episode is passed by reference through the queue, so the object the learner
    receives is the identical object a collector sent — no ``to_dict`` / ``from_dict``
    round-trip on the in-process hot path.

    Backpressure: ``maxsize > 0`` makes :meth:`send` block when the queue is full and
    unblock as the learner drains it (``queue.Queue.put``'s native behavior), so a
    slow learner throttles fast collectors instead of letting the queue grow without
    bound. The default ``maxsize=0`` is unbounded: :meth:`send` never blocks.

    Threading: ``queue.Queue`` is already safe for many producers and one consumer.
    There is exactly one consumer (the learner thread). :meth:`close` latches a
    ``_closed`` flag and best-effort enqueues a single sentinel to wake a blocked
    :meth:`recv`; :meth:`recv` polls with a short timeout so it also wakes to the
    latched flag, which keeps :meth:`close` non-blocking even on a full bounded queue
    (where the sentinel can be dropped) and stops a second :meth:`recv` from hanging.
    """

    def __init__(self, maxsize: int = 0) -> None:
        """Build the transport.

        Args:
            maxsize: Queue capacity. ``> 0`` bounds the queue for backpressure
                (send blocks when full); ``0`` (the default) is unbounded.
        """
        self._queue: queue.Queue = queue.Queue(maxsize=maxsize)
        self._closed = False

    def send(self, episode: Episode) -> None:
        """Enqueue one episode for the learner (collector side).

        Blocks if the queue is bounded and currently full, returning once the learner
        drains a slot. The episode is stored by reference (no copy, no serialization).

        Args:
            episode: The :class:`Episode` to hand to the learner.

        Raises:
            TransportError: If the transport has been closed.
        """
        if self._closed:
            raise TransportError("send() on a closed transport")
        # Blocking put: on a bounded queue this is the backpressure point. The closed
        # check above is best-effort (a concurrent close() can race it); the closed
        # flag still latches so recv() reports end-of-stream cleanly.
        self._queue.put(episode)

    def recv(self) -> Episode:
        """Dequeue the next episode in FIFO order (learner side), blocking if empty.

        Polls the queue with a short ``_RECV_POLL_SECONDS`` timeout rather than blocking
        indefinitely so it can also wake to the latched ``_closed`` flag; that timeout
        is what lets :meth:`close` stay non-blocking on a full bounded queue, where the
        wake-up sentinel may have been dropped. Queued episodes still drain FIFO ahead
        of any close, and the timeout only fires when the queue is empty, so there is
        no busy-spin.

        Returns:
            The next :class:`Episode`, the identical object that was sent.

        Raises:
            TransportError: Once the stream is closed and drained. Re-raises on every
                later call (the flag is latched and the sentinel best-effort put back)
                so the single consumer never hangs.
        """
        while True:
            try:
                item = self._queue.get(timeout=_RECV_POLL_SECONDS)
            except queue.Empty:
                # No item arrived this interval; if close() latched the flag (and any
                # wake-up sentinel was dropped on a full queue), report end-of-stream.
                if self._closed:
                    raise TransportError("recv() on a closed transport")
                continue
            if item is _CLOSED:
                # Best-effort re-put so any subsequent recv() also sees end-of-stream
                # rather than blocking; if the queue is full the latched flag still
                # makes the next poll raise, so dropping the re-put is harmless.
                try:
                    self._queue.put_nowait(_CLOSED)
                except queue.Full:
                    pass
                raise TransportError("recv() on a closed transport")
            return item

    def close(self) -> None:
        """Close the channel and wake a blocked :meth:`recv`. Safe to call twice.

        Never blocks: it latches ``_closed`` first, then best-effort enqueues a single
        sentinel with a non-blocking put. On a full bounded queue the sentinel is
        dropped, but :meth:`recv` polls with ``_RECV_POLL_SECONDS`` and wakes to the
        latched flag, so the consumer still reports end-of-stream — this avoids the
        deadlock where the sole consumer blocks inside :meth:`close` waiting for a
        drain only it could perform. Already queued episodes stay ahead of the sentinel
        and drain in order before :meth:`recv` reports end-of-stream.
        """
        if self._closed:
            return
        self._closed = True
        try:
            self._queue.put_nowait(_CLOSED)
        except queue.Full:
            pass
