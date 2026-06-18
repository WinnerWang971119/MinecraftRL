"""Tests for LocalTransport FIFO ordering, close semantics, and backpressure (TC2, TC18).

Also covers the Episode serialization round-trip (TC1) since Episode is the
transfer unit and this file already imports distributed.serialization.

TC1 -- Episode.to_dict / from_dict round-trip + JSON-friendliness.
TC2 -- LocalTransport FIFO ordering; by-reference pass; close unblocks recv.
TC18 -- Bounded-queue backpressure: sender blocks until consumer drains; unbounded
        never blocks; close on a full queue does not deadlock.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from typing import List

import numpy as np
import pytest

from distributed.serialization import Episode
from distributed.transport import LocalTransport, TransportError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_H = 8  # small LSTM hidden size for test hidden states


def _make_obs(seed: int) -> np.ndarray:
    """Return a reproducible float32 obs of shape (OBS_DIM,)."""
    from env.observation_spec import OBS_DIM

    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=OBS_DIM).astype(np.float32)


def _make_hidden(seed: int) -> np.ndarray:
    """Return a float32 array of shape (2, 1, _H) for one LSTM layer."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((2, 1, _H)).astype(np.float32)


def _make_episode(arena_id: int = 0, n_transitions: int = 3) -> Episode:
    """Build a small but realistic Episode with ``n_transitions`` steps."""
    transitions = []
    hidden_states = []
    for i in range(n_transitions):
        obs = _make_obs(seed=arena_id * 100 + i)
        next_obs = _make_obs(seed=arena_id * 100 + i + 1)
        transitions.append((obs, i % 8, float(i) * 0.1, next_obs, i == n_transitions - 1))
        hidden_states.append(_make_hidden(seed=arena_id * 100 + i))
    return Episode(
        transitions=transitions,
        hidden_states=hidden_states,
        arena_id=arena_id,
        policy_version=7,
        code_version="v0.1",
        total_reward=sum(float(i) * 0.1 for i in range(n_transitions)),
    )


# ---------------------------------------------------------------------------
# TC1 -- Episode serialization round-trip
# ---------------------------------------------------------------------------


def test_tc1_episode_round_trip_arrays_equal():
    """from_dict(to_dict(ep)) reconstructs all arrays with float32 dtype and correct shape."""
    ep = _make_episode(arena_id=1, n_transitions=4)
    d = ep.to_dict()
    ep2 = Episode.from_dict(d)

    assert len(ep2.transitions) == len(ep.transitions)
    for (obs, action, reward, next_obs, done), (obs2, action2, reward2, next_obs2, done2) in zip(
        ep.transitions, ep2.transitions
    ):
        # Arrays match in value and dtype.
        assert obs2.dtype == np.float32
        assert next_obs2.dtype == np.float32
        assert obs2.shape == obs.shape
        assert next_obs2.shape == next_obs.shape
        assert np.allclose(obs2, obs)
        assert np.allclose(next_obs2, next_obs)
        # Scalars have the right types and values.
        assert isinstance(action2, int)
        assert action2 == action
        assert isinstance(reward2, float)
        assert reward2 == pytest.approx(reward)
        assert isinstance(done2, bool)
        assert done2 == done

    assert len(ep2.hidden_states) == len(ep.hidden_states)
    for h, h2 in zip(ep.hidden_states, ep2.hidden_states):
        assert h2.dtype == np.float32
        assert h2.shape == h.shape
        assert np.allclose(h2, h)

    assert ep2.arena_id == ep.arena_id
    assert ep2.policy_version == ep.policy_version
    assert ep2.code_version == ep.code_version
    assert ep2.total_reward == pytest.approx(ep.total_reward)


def test_tc1_episode_to_dict_is_json_friendly():
    """to_dict() produces a dict that json.dumps() can serialize without error."""
    ep = _make_episode(arena_id=2, n_transitions=2)
    d = ep.to_dict()
    # This must not raise; it proves no numpy scalars/arrays leaked through.
    serialized = json.dumps(d)
    assert isinstance(serialized, str)
    assert len(serialized) > 0


def test_tc1_episode_scalar_types_in_dict():
    """to_dict() lowers every scalar to a plain Python int/float/bool/str."""
    ep = _make_episode(arena_id=0, n_transitions=2)
    d = ep.to_dict()

    assert isinstance(d["arena_id"], int)
    assert isinstance(d["policy_version"], int)
    assert isinstance(d["code_version"], str)
    assert isinstance(d["total_reward"], float)

    for tr in d["transitions"]:
        _obs_list, action, reward, _next_obs_list, done = tr
        assert isinstance(action, int)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        # Lists of Python floats — no numpy array inside.
        assert isinstance(_obs_list, list)
        assert isinstance(_next_obs_list, list)
        assert all(isinstance(x, float) for x in _obs_list)


# ---------------------------------------------------------------------------
# TC2 -- LocalTransport FIFO ordering, by-reference, close semantics
# ---------------------------------------------------------------------------


def test_tc2_fifo_ordering_and_by_reference():
    """Episodes are received in the same order they were sent and by reference."""
    transport = LocalTransport()
    episodes: List[Episode] = [_make_episode(arena_id=i) for i in range(5)]

    for ep in episodes:
        transport.send(ep)

    received: List[Episode] = []
    for _ in episodes:
        received.append(transport.recv())

    transport.close()

    assert len(received) == len(episodes)
    for sent, got in zip(episodes, received):
        # By reference: the exact same object, no copy.
        assert got is sent


def test_tc2_close_unblocks_blocked_recv():
    """close() unblocks a recv() that is blocked waiting on an empty queue."""
    transport = LocalTransport()

    error_holder: List[Exception] = []
    finished = threading.Event()

    def blocked_recv():
        try:
            transport.recv()
            # If recv somehow returned a value, that is also wrong — we expect
            # the transport to be closed before any episode arrives.
            error_holder.append(AssertionError("recv returned instead of raising"))
        except TransportError:
            pass  # expected
        except Exception as exc:
            error_holder.append(exc)
        finally:
            finished.set()

    t = threading.Thread(target=blocked_recv, daemon=True)
    t.start()

    # Give the thread time to block inside recv().
    time.sleep(0.05)

    transport.close()

    # The thread must terminate promptly after close.
    finished.wait(timeout=2.0)
    t.join(timeout=2.0)

    assert not t.is_alive(), "recv thread did not terminate after close()"
    assert not error_holder, f"recv thread raised unexpected error: {error_holder}"


def test_tc2_send_after_close_raises_transport_error():
    """send() on a closed transport raises TransportError."""
    transport = LocalTransport()
    transport.close()
    ep = _make_episode()
    with pytest.raises(TransportError):
        transport.send(ep)


def test_tc2_recv_after_close_raises_transport_error_and_does_not_hang():
    """recv() after close raises TransportError and does not hang (second call too)."""
    transport = LocalTransport()
    transport.close()

    # First call must raise promptly.
    with pytest.raises(TransportError):
        transport.recv()

    # Second call must also raise and must not hang.
    error_raised = threading.Event()

    def second_recv():
        try:
            transport.recv()
        except TransportError:
            error_raised.set()

    t = threading.Thread(target=second_recv, daemon=True)
    t.start()
    t.join(timeout=2.0)

    assert not t.is_alive(), "second recv() call hung after close()"
    assert error_raised.is_set(), "second recv() did not raise TransportError"


def test_tc2_close_is_idempotent():
    """Calling close() twice does not raise."""
    transport = LocalTransport()
    transport.close()
    transport.close()  # must not raise


# ---------------------------------------------------------------------------
# TC18 -- Backpressure and bounded-queue semantics
# ---------------------------------------------------------------------------


def test_tc18_bounded_queue_blocks_sender_until_consumer_drains():
    """With maxsize=1, the second send blocks until the consumer drains one slot."""
    transport = LocalTransport(maxsize=1)
    ep0 = _make_episode(arena_id=0)
    ep1 = _make_episode(arena_id=1)

    # Fill the queue.
    transport.send(ep0)

    # Now the queue is full (maxsize=1). A second send must block.
    sender_started = threading.Event()
    sender_done = threading.Event()

    def slow_sender():
        sender_started.set()
        transport.send(ep1)  # should block until consumer recvs ep0
        sender_done.set()

    t = threading.Thread(target=slow_sender, daemon=True)
    t.start()

    # Wait for the sender to actually start and attempt the blocking put.
    sender_started.wait(timeout=2.0)

    # Sender should NOT have completed yet; the queue is still full.
    time.sleep(0.05)
    assert not sender_done.is_set(), "sender completed before consumer drained (no backpressure)"

    # Consumer drains one slot; sender must now unblock.
    got = transport.recv()
    assert got is ep0

    sender_done.wait(timeout=2.0)
    t.join(timeout=2.0)
    assert not t.is_alive(), "sender thread did not unblock after consumer drained"

    # Drain the second episode.
    got2 = transport.recv()
    assert got2 is ep1

    transport.close()


def test_tc18_unbounded_queue_never_blocks_sender():
    """With maxsize=0 (unbounded), many sends never block the caller."""
    transport = LocalTransport(maxsize=0)
    n = 200
    episodes = [_make_episode(arena_id=i) for i in range(n)]

    # All sends must complete immediately (no consumer running).
    for ep in episodes:
        transport.send(ep)

    # All received in FIFO order.
    for expected in episodes:
        got = transport.recv()
        assert got is expected

    transport.close()


def test_tc18_close_on_full_bounded_queue_does_not_deadlock():
    """close() on a full bounded queue returns promptly without deadlocking."""
    transport = LocalTransport(maxsize=1)
    ep = _make_episode(arena_id=0)
    transport.send(ep)  # queue is now full

    # close() must return quickly even though the queue is full (the sentinel
    # may be dropped, but the recv-poll still sees the closed flag).
    close_done = threading.Event()

    def do_close():
        transport.close()
        close_done.set()

    t = threading.Thread(target=do_close, daemon=True)
    t.start()
    close_done.wait(timeout=2.0)
    t.join(timeout=2.0)

    assert not t.is_alive(), "close() deadlocked on a full bounded queue"
    assert close_done.is_set()

    # A recv on the now-closed transport must either return the queued episode
    # or raise TransportError -- it must NOT hang.
    result_holder: List[object] = []
    error_holder: List[Exception] = []
    recv_done = threading.Event()

    def do_recv():
        try:
            result_holder.append(transport.recv())
        except TransportError:
            error_holder.append(TransportError("closed"))
        finally:
            recv_done.set()

    t2 = threading.Thread(target=do_recv, daemon=True)
    t2.start()
    recv_done.wait(timeout=2.0)
    t2.join(timeout=2.0)

    assert not t2.is_alive(), "recv after close on full queue hung"
    # Either the queued episode was drained or TransportError was raised; both are fine.
    assert result_holder or error_holder, "recv after close neither returned nor raised"
