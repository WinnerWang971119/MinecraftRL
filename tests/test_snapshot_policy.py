"""Tests for WeightStore isolation and SnapshotPolicy version-gated refresh (TC3, TC4).

TC3 -- WeightStore.publish() stores a detached clone; mutating the source net
       afterwards does not change the stored snapshot.
TC4 -- SnapshotPolicy.maybe_refresh() loads only on a strict version advance;
       act() returns an int and runs without building gradients; two policies
       seeded identically produce the same action stream.
"""

from __future__ import annotations

import pytest

from env.observation_spec import OBS_DIM
from agent.actions import N_ACTIONS

torch = pytest.importorskip("torch", exc_type=ImportError)

import torch as _torch  # noqa: E402 -- only reached when torch is present
import numpy as np      # noqa: E402

from distributed.weights import WeightStore, SnapshotPolicy  # noqa: E402


# ---------------------------------------------------------------------------
# Architecture kwargs -- small but valid (obs_dim / n_actions frozen to
# OBS_DIM / N_ACTIONS as required by DuelingDRQN's freeze guard).
# ---------------------------------------------------------------------------
_SMALL_NET_KWARGS = dict(
    obs_dim=OBS_DIM,
    n_actions=N_ACTIONS,
    encoder_hidden=16,
    lstm_hidden=8,
    lstm_layers=1,
)


def _net_factory():
    from agent.dqn import DuelingDRQN

    return DuelingDRQN(**_SMALL_NET_KWARGS)


def _make_obs() -> np.ndarray:
    """Return a valid float32 obs vector."""
    rng = np.random.default_rng(42)
    return rng.uniform(-1.0, 1.0, size=OBS_DIM).astype(np.float32)


# ---------------------------------------------------------------------------
# TC3 -- WeightStore stores a detached clone, not a view
# ---------------------------------------------------------------------------


def test_tc3_publish_stores_detached_clone():
    """Mutating the source net after publish does not change the stored snapshot."""
    net = _net_factory()
    store = WeightStore()

    store.publish(net.state_dict(), version=0)
    snapshot_before, _ = store.latest()
    assert snapshot_before is not None

    # Capture the value of the first parameter tensor in the snapshot NOW.
    first_key = next(iter(snapshot_before))
    value_before = snapshot_before[first_key].clone()

    # Mutate the live net in place (simulate an optimizer step).
    with _torch.no_grad():
        first_param = next(net.parameters())
        first_param.add_(100.0)

    # The stored snapshot must be unchanged.
    snapshot_after, _ = store.latest()
    assert snapshot_after is not None
    value_after = snapshot_after[first_key]

    assert _torch.allclose(value_after, value_before), (
        "WeightStore snapshot changed after in-place mutation of the source net "
        "(snapshot is a view, not a clone)"
    )


def test_tc3_publish_clone_is_not_a_view_of_source():
    """snapshot tensors do not alias the net's parameter storage."""
    net = _net_factory()
    store = WeightStore()
    store.publish(net.state_dict(), version=0)

    snapshot, _ = store.latest()
    assert snapshot is not None

    for key, snap_tensor in snapshot.items():
        net_tensor = net.state_dict()[key]
        # If they share storage, data_ptr() would match for the same byte.
        # A genuine clone always has a distinct data pointer.
        if snap_tensor.is_contiguous() and net_tensor.is_contiguous():
            assert snap_tensor.data_ptr() != net_tensor.data_ptr(), (
                f"snapshot[{key!r}] shares storage with the live net parameter"
            )


def test_tc3_publish_raises_on_negative_version():
    """publish() rejects a negative version (reserved sentinel is -1)."""
    net = _net_factory()
    store = WeightStore()
    with pytest.raises(ValueError):
        store.publish(net.state_dict(), version=-1)


def test_tc3_latest_returns_none_before_first_publish():
    """latest() on an empty store returns (None, -1)."""
    store = WeightStore()
    sd, version = store.latest()
    assert sd is None
    assert version == -1


# ---------------------------------------------------------------------------
# TC4 -- SnapshotPolicy version-gated refresh and no_grad act
# ---------------------------------------------------------------------------


def test_tc4_maybe_refresh_no_op_on_empty_store():
    """maybe_refresh with nothing published does not crash and version stays -1."""
    store = WeightStore()
    policy = SnapshotPolicy(_net_factory, generator_seed=0)

    assert policy.version == -1
    policy.maybe_refresh(store)  # must not raise
    assert policy.version == -1


def test_tc4_maybe_refresh_loads_on_first_publish():
    """maybe_refresh loads the snapshot after the first publish."""
    store = WeightStore()
    net = _net_factory()

    # Give the net distinctly non-zero weights so the load is detectable.
    with _torch.no_grad():
        for p in net.parameters():
            p.fill_(1.23)

    store.publish(net.state_dict(), version=0)

    policy = SnapshotPolicy(_net_factory, generator_seed=0)
    assert policy.version == -1

    policy.maybe_refresh(store)
    assert policy.version == 0

    # The policy net's first parameter must now equal 1.23.
    first_policy_param = next(policy.net.parameters()).detach()
    assert _torch.allclose(first_policy_param, _torch.full_like(first_policy_param, 1.23)), (
        "policy net was not updated after maybe_refresh with v0"
    )


def test_tc4_maybe_refresh_does_not_reload_same_version():
    """Republishing at the same version must not trigger a reload."""
    store = WeightStore()

    net_v0 = _net_factory()
    with _torch.no_grad():
        for p in net_v0.parameters():
            p.fill_(1.0)
    store.publish(net_v0.state_dict(), version=0)

    policy = SnapshotPolicy(_net_factory, generator_seed=0)
    policy.maybe_refresh(store)
    assert policy.version == 0

    # Capture the policy's current first-param value (1.0).
    first_key = next(iter(net_v0.state_dict()))
    val_after_v0 = next(policy.net.parameters()).detach().clone()

    # Publish a DIFFERENT state_dict at the SAME version 0.
    net_v0b = _net_factory()
    with _torch.no_grad():
        for p in net_v0b.parameters():
            p.fill_(99.0)
    store.publish(net_v0b.state_dict(), version=0)

    policy.maybe_refresh(store)

    # Policy must NOT have reloaded; version still 0 and weights still 1.0.
    assert policy.version == 0
    val_after_v0b = next(policy.net.parameters()).detach()
    assert _torch.allclose(val_after_v0b, val_after_v0), (
        "policy reloaded on same version (should only reload on STRICT advance)"
    )


def test_tc4_maybe_refresh_reloads_on_version_advance():
    """Advancing the version from 0 to 1 triggers a reload."""
    store = WeightStore()

    net_v0 = _net_factory()
    with _torch.no_grad():
        for p in net_v0.parameters():
            p.fill_(1.0)
    store.publish(net_v0.state_dict(), version=0)

    policy = SnapshotPolicy(_net_factory, generator_seed=0)
    policy.maybe_refresh(store)
    assert policy.version == 0

    net_v1 = _net_factory()
    with _torch.no_grad():
        for p in net_v1.parameters():
            p.fill_(2.0)
    store.publish(net_v1.state_dict(), version=1)

    policy.maybe_refresh(store)
    assert policy.version == 1

    first_policy_param = next(policy.net.parameters()).detach()
    assert _torch.allclose(first_policy_param, _torch.full_like(first_policy_param, 2.0)), (
        "policy net was not updated after maybe_refresh with v1"
    )


def test_tc4_act_returns_int_action_in_valid_range():
    """act() returns a Python int in [0, N_ACTIONS)."""
    store = WeightStore()
    net = _net_factory()
    store.publish(net.state_dict(), version=0)

    policy = SnapshotPolicy(_net_factory, generator_seed=1)
    policy.maybe_refresh(store)

    obs = _make_obs()
    action, hidden = policy.act(obs, hidden=None, epsilon=0.5)

    assert isinstance(action, int), f"action is {type(action).__name__}, expected int"
    assert 0 <= action < N_ACTIONS


def test_tc4_act_does_not_build_grad():
    """act() runs under no_grad: net params have no grad after the call."""
    store = WeightStore()
    net = _net_factory()
    store.publish(net.state_dict(), version=0)

    policy = SnapshotPolicy(_net_factory, generator_seed=2)
    policy.maybe_refresh(store)

    obs = _make_obs()
    _action, new_hidden = policy.act(obs, hidden=None, epsilon=0.0)

    # No parameter should have accumulated a gradient.
    for param in policy.net.parameters():
        assert param.grad is None, "policy net parameter has .grad after act() call"

    # Hidden tensors must not require grad.
    h, c = new_hidden
    assert not h.requires_grad, "returned hidden h requires grad"
    assert not c.requires_grad, "returned hidden c requires grad"


def test_tc4_same_seed_produces_same_action_stream():
    """Two SnapshotPolicies with identical seeds and weights produce the same actions."""
    store = WeightStore()
    net = _net_factory()
    store.publish(net.state_dict(), version=0)

    policy_a = SnapshotPolicy(_net_factory, generator_seed=42)
    policy_b = SnapshotPolicy(_net_factory, generator_seed=42)
    policy_a.maybe_refresh(store)
    policy_b.maybe_refresh(store)

    hidden_a = None
    hidden_b = None
    rng = np.random.default_rng(0)

    for _ in range(10):
        obs = rng.uniform(-1.0, 1.0, size=OBS_DIM).astype(np.float32)
        action_a, hidden_a = policy_a.act(obs, hidden_a, epsilon=0.5)
        action_b, hidden_b = policy_b.act(obs, hidden_b, epsilon=0.5)
        assert action_a == action_b, (
            "policies with the same seed diverged on the same input"
        )


def test_tc4_different_seed_diverges():
    """Two SnapshotPolicies with different seeds produce different exploration streams."""
    store = WeightStore()
    net = _net_factory()
    store.publish(net.state_dict(), version=0)

    policy_a = SnapshotPolicy(_net_factory, generator_seed=0)
    policy_b = SnapshotPolicy(_net_factory, generator_seed=999)
    policy_a.maybe_refresh(store)
    policy_b.maybe_refresh(store)

    # Use a high epsilon to maximize random-action draws, then check at least
    # one step differs over a moderate number of trials.
    hidden_a = None
    hidden_b = None
    rng = np.random.default_rng(7)
    actions_a = []
    actions_b = []

    for _ in range(30):
        obs = rng.uniform(-1.0, 1.0, size=OBS_DIM).astype(np.float32)
        a, hidden_a = policy_a.act(obs, hidden_a, epsilon=1.0)
        b, hidden_b = policy_b.act(obs, hidden_b, epsilon=1.0)
        actions_a.append(a)
        actions_b.append(b)

    assert actions_a != actions_b, (
        "policies with different seeds produced identical action streams over 30 steps"
    )
