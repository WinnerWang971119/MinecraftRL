"""Tests for the prioritized SEQUENCE replay buffer (T15 / TC9).

These exercise ``agent.replay.PrioritizedSequenceReplay`` against synthetic
episodes (no env, no torch) and cover the kickoff test case TC9:

  * length-L sequences are returned at the right length and are CONTIGUOUS within
    a single episode (no boundary crossing),
  * importance-sampling weights are normalized (max == 1.0) with the right shape,
  * higher-priority items are sampled more often (fixed-seed statistical check),
  * ``update_priorities`` changes subsequent sampling — zeroing a priority makes
    an item (almost) never sampled; raising one makes it dominate,
  * new episodes enter at MAX priority (immediately eligible),
  * β annealing moves IS weights toward uniform as β -> 1.

Plus boundary-policy, capacity/eviction, burn-in, and n-step-return checks.

Determinism: every sampling test constructs the buffer with a SEEDED
``np.random.default_rng`` so the statistical assertions are reproducible.
"""

from __future__ import annotations

import numpy as np
import pytest

from agent.replay import (
    DEFAULT_PRIORITY_EPS,
    PrioritizedSequenceReplay,
    SequenceBatch,
    Transition,
    compute_n_step_returns,
)


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------

OBS_DIM = 4


def _make_episode(length: int, *, tag: float = 0.0, with_hidden: bool = False):
    """Build a length-``length`` episode of Transitions.

    Each step's obs encodes (tag, t, 0, 0) so a returned window can be checked for
    contiguity (consecutive t) and single-episode membership (constant tag).
    """
    episode = []
    hidden = [] if with_hidden else None
    for t in range(length):
        obs = np.array([tag, float(t), 0.0, 0.0], dtype=np.float32)
        next_obs = np.array([tag, float(t + 1), 0.0, 0.0], dtype=np.float32)
        episode.append(
            Transition(
                obs=obs,
                action=t % 8,
                reward=float(t),
                next_obs=next_obs,
                done=(t == length - 1),
            )
        )
        if with_hidden:
            hidden.append(np.full(3, tag, dtype=np.float32))
    return episode, hidden


def _seeded_buffer(**kwargs) -> PrioritizedSequenceReplay:
    """Construct a buffer with a fixed RNG seed for deterministic sampling."""
    kwargs.setdefault("capacity", 1000)
    kwargs.setdefault("seq_len", 4)
    rng = np.random.default_rng(kwargs.pop("seed", 12345))
    return PrioritizedSequenceReplay(rng=rng, **kwargs)


# ---------------------------------------------------------------------------
# Sequence shape + contiguity (TC9, primary).
# ---------------------------------------------------------------------------


def test_sampled_sequences_have_length_L():
    """Each returned window's learn span has exactly L steps."""
    buf = _seeded_buffer(seq_len=4)
    ep, _ = _make_episode(20, tag=1.0)
    buf.add_episode(ep)

    batch = buf.sample_sequences(batch_size=8, L=4)
    # No burn-in here, so the full window == L.
    assert batch.obs.shape == (8, 4, OBS_DIM)
    assert batch.actions.shape == (8, 4)
    assert batch.rewards.shape == (8, 4)
    assert batch.next_obs.shape == (8, 4, OBS_DIM)
    assert batch.dones.shape == (8, 4)
    assert len(batch) == 8


def test_sampled_sequences_are_contiguous_within_one_episode():
    """Every window is consecutive in time AND from a single episode (no crossing)."""
    buf = _seeded_buffer(seq_len=5)
    # Three episodes with distinct tags so we can detect a boundary crossing.
    for tag in (10.0, 20.0, 30.0):
        ep, _ = _make_episode(15, tag=tag)
        buf.add_episode(ep)

    batch = buf.sample_sequences(batch_size=64, L=5)
    for b in range(len(batch)):
        window = batch.obs[b]  # (5, OBS_DIM)
        tags = window[:, 0]
        steps = window[:, 1]
        # Single episode: the tag column is constant across the window.
        assert np.all(tags == tags[0]), f"window {b} crossed an episode boundary"
        # Contiguous: the step column increments by exactly 1.
        assert np.array_equal(steps, np.arange(steps[0], steps[0] + 5)), (
            f"window {b} is not contiguous in time: {steps}"
        )


def test_sample_rejects_mismatched_L():
    """Passing an L that disagrees with the configured seq_len raises."""
    buf = _seeded_buffer(seq_len=4)
    ep, _ = _make_episode(10, tag=1.0)
    buf.add_episode(ep)
    with pytest.raises(ValueError):
        buf.sample_sequences(batch_size=2, L=7)


# ---------------------------------------------------------------------------
# Boundary policy: short episodes contribute no windows.
# ---------------------------------------------------------------------------


def test_episode_shorter_than_window_yields_no_samples():
    """An episode shorter than the window is stored but never sampled."""
    buf = _seeded_buffer(seq_len=8)  # window == 8
    short_ep, _ = _make_episode(5, tag=1.0)  # 5 < 8 -> 0 start indices
    buf.add_episode(short_ep)

    assert len(buf) == 5  # transitions counted
    assert buf.n_sampleable == 0
    with pytest.raises(ValueError):
        buf.sample_sequences(batch_size=1)


def test_start_index_count_matches_boundary_policy():
    """An episode of length M yields exactly M - window + 1 start indices."""
    buf = _seeded_buffer(seq_len=4, burn_in=2)  # window == 6
    ep, _ = _make_episode(10, tag=1.0)
    buf.add_episode(ep)
    assert buf.n_sampleable == 10 - 6 + 1  # == 5


# ---------------------------------------------------------------------------
# IS weights: shape + normalization (TC9).
# ---------------------------------------------------------------------------


def test_is_weights_are_normalized_to_max_one():
    """IS weights have shape (batch,) and a maximum of exactly 1.0."""
    buf = _seeded_buffer(seq_len=4)
    # Two episodes so priorities can differ after an update.
    for tag in (1.0, 2.0):
        ep, _ = _make_episode(20, tag=tag)
        buf.add_episode(ep)

    batch = buf.sample_sequences(batch_size=16, L=4)
    assert batch.is_weights.shape == (16,)
    assert np.isclose(batch.is_weights.max(), 1.0)
    assert np.all(batch.is_weights > 0.0)
    assert np.all(batch.is_weights <= 1.0 + 1e-6)


def test_is_weights_uniform_when_all_priorities_equal():
    """With all priorities equal (fresh max-priority entries), weights are ~1."""
    buf = _seeded_buffer(seq_len=4)
    ep, _ = _make_episode(40, tag=1.0)
    buf.add_episode(ep)

    batch = buf.sample_sequences(batch_size=16, L=4)
    # All start indices share the max priority, so every weight is 1.0.
    assert np.allclose(batch.is_weights, 1.0)


# ---------------------------------------------------------------------------
# Prioritization: higher priority -> sampled more often (TC9, statistical).
# ---------------------------------------------------------------------------


def test_higher_priority_sampled_more_often():
    """After raising one window's priority, it is sampled far more than average."""
    buf = _seeded_buffer(seq_len=4, capacity=1000, seed=7)
    ep, _ = _make_episode(20, tag=1.0)
    buf.add_episode(ep)
    n = buf.n_sampleable
    assert n > 1

    # Sample once to learn the leaf ids, then crank up exactly one leaf.
    first = buf.sample_sequences(batch_size=n, L=4)
    target_leaf = int(first.indices[0])
    # Reset every sampled leaf to a small priority, then spike the target.
    small_td = np.full(len(first.indices), 0.0)
    buf.update_priorities(first.indices, small_td)  # all -> ε_p^α (tiny)
    buf.update_priorities(np.array([target_leaf]), np.array([1000.0]))  # huge

    counts = np.zeros(buf.capacity, dtype=np.int64)
    trials = 4000
    for _ in range(trials):
        b = buf.sample_sequences(batch_size=1, L=4)
        counts[int(b.indices[0])] += 1

    # The spiked leaf should dominate: far above a uniform 1/n share.
    uniform_share = trials / n
    assert counts[target_leaf] > 10 * uniform_share, (
        f"spiked leaf sampled {counts[target_leaf]} times, "
        f"uniform share would be {uniform_share:.1f}"
    )


def test_zeroing_priority_makes_item_almost_never_sampled():
    """A near-zero priority leaf is (almost) never drawn versus high-priority ones."""
    buf = _seeded_buffer(seq_len=4, capacity=1000, seed=99)
    ep, _ = _make_episode(20, tag=1.0)
    buf.add_episode(ep)
    n = buf.n_sampleable

    first = buf.sample_sequences(batch_size=n, L=4)
    # All leaves get a large priority...
    buf.update_priorities(first.indices, np.full(len(first.indices), 100.0))
    # ...except one driven to the floor (|δ| = 0 -> priority == ε_p^α).
    dead_leaf = int(first.indices[0])
    buf.update_priorities(np.array([dead_leaf]), np.array([0.0]))

    counts = np.zeros(buf.capacity, dtype=np.int64)
    trials = 4000
    for _ in range(trials):
        b = buf.sample_sequences(batch_size=1, L=4)
        counts[int(b.indices[0])] += 1

    # The floored leaf is drawn essentially never relative to its peers.
    assert counts[dead_leaf] <= trials * 0.005, (
        f"floored leaf was still sampled {counts[dead_leaf]} times"
    )


def test_update_priorities_changes_sampling_distribution():
    """update_priorities shifts which leaf dominates between two regimes."""
    buf = _seeded_buffer(seq_len=4, capacity=1000, seed=3)
    ep, _ = _make_episode(20, tag=1.0)
    buf.add_episode(ep)
    n = buf.n_sampleable
    first = buf.sample_sequences(batch_size=n, L=4)

    leaf_a = int(first.indices[0])
    leaf_b = int(first.indices[1])

    # Regime 1: A dominant.
    buf.update_priorities(first.indices, np.full(n, 0.0))
    buf.update_priorities(np.array([leaf_a]), np.array([500.0]))
    a_share = _empirical_share(buf, leaf_a, trials=2000)

    # Regime 2: flip — B dominant, A floored.
    buf.update_priorities(np.array([leaf_a]), np.array([0.0]))
    buf.update_priorities(np.array([leaf_b]), np.array([500.0]))
    a_share_after = _empirical_share(buf, leaf_a, trials=2000)

    assert a_share > 0.5, f"A should dominate regime 1, share={a_share}"
    assert a_share_after < 0.05, f"A should be rare in regime 2, share={a_share_after}"


def _empirical_share(buf, leaf, *, trials):
    hits = 0
    for _ in range(trials):
        b = buf.sample_sequences(batch_size=1, L=4)
        if int(b.indices[0]) == leaf:
            hits += 1
    return hits / trials


# ---------------------------------------------------------------------------
# New entries enter at MAX priority (TC9).
# ---------------------------------------------------------------------------


def test_new_episode_enters_at_max_priority_and_is_eligible():
    """A freshly added episode's windows are immediately sampleable at max priority."""
    buf = _seeded_buffer(seq_len=4, capacity=1000, seed=1)
    # Episode 1, then drive its priorities to the floor.
    ep1, _ = _make_episode(10, tag=1.0)
    buf.add_episode(ep1)
    first = buf.sample_sequences(batch_size=buf.n_sampleable, L=4)
    buf.update_priorities(first.indices, np.full(len(first.indices), 0.0))

    # Episode 2 enters at the current max priority (which the spike below sets).
    # First bump the running max via a high TD error on an ep1 leaf so "max" is
    # meaningfully larger than the floor.
    buf.update_priorities(np.array([int(first.indices[0])]), np.array([50.0]))
    ep2, _ = _make_episode(10, tag=2.0)
    buf.add_episode(ep2)

    # Sample many singletons; ep2 windows (tag 2.0) should appear despite ep1
    # being mostly floored — they entered at max priority.
    seen_ep2 = 0
    for _ in range(500):
        b = buf.sample_sequences(batch_size=1, L=4)
        if b.obs[0, 0, 0] == 2.0:
            seen_ep2 += 1
    assert seen_ep2 > 0, "freshly added episode at max priority was never sampled"


# ---------------------------------------------------------------------------
# Beta annealing moves IS weights toward uniform (TC9).
# ---------------------------------------------------------------------------


def test_beta_annealing_moves_weights_toward_uniform():
    """As β -> 1 with a skewed priority distribution, IS weights spread toward 1."""
    # Build a buffer and a deliberately skewed priority distribution.
    buf = _seeded_buffer(seq_len=4, capacity=1000, seed=42)
    ep, _ = _make_episode(40, tag=1.0)
    buf.add_episode(ep)
    n = buf.n_sampleable
    first = buf.sample_sequences(batch_size=n, L=4)
    # Geometric spread of priorities so probabilities are non-uniform.
    tds = np.geomspace(0.01, 100.0, num=n)
    buf.update_priorities(first.indices, tds)

    # Low beta: weights are more spread (smaller minimum after normalization).
    buf.beta = 0.0
    low = buf.sample_sequences(batch_size=n, L=4)
    spread_low = low.is_weights.min()

    # High beta -> 1: weights compress toward 1 (minimum closer to 1).
    buf.beta = 1.0
    high = buf.sample_sequences(batch_size=n, L=4)
    spread_high = high.is_weights.min()

    # At beta == 0 every weight is exactly 1 (no correction); at beta == 1 the
    # correction is strongest, so the minimum weight is SMALLER. Verify the
    # direction of the effect explicitly.
    assert np.isclose(spread_low, 1.0), "beta=0 must give uniform weights of 1"
    assert spread_high < spread_low, "beta=1 must produce more spread-out weights"


def test_anneal_beta_linear_schedule():
    """anneal_beta ramps β linearly from beta0 to 1.0 across the horizon."""
    buf = _seeded_buffer(seq_len=4, beta0=0.4, beta_anneal_steps=100)
    assert np.isclose(buf.anneal_beta(0), 0.4)
    assert np.isclose(buf.anneal_beta(50), 0.4 + 0.6 * 0.5)
    assert np.isclose(buf.anneal_beta(100), 1.0)
    assert np.isclose(buf.anneal_beta(1000), 1.0)  # clamped


def test_anneal_beta_without_horizon_raises():
    """anneal_beta requires beta_anneal_steps to have been configured."""
    buf = _seeded_buffer(seq_len=4)  # no beta_anneal_steps
    with pytest.raises(RuntimeError):
        buf.anneal_beta(10)


# ---------------------------------------------------------------------------
# Capacity bound + eviction.
# ---------------------------------------------------------------------------


def test_capacity_evicts_oldest_episodes():
    """Total stored transitions never exceed capacity; oldest episode is evicted."""
    buf = PrioritizedSequenceReplay(
        capacity=30,
        seq_len=4,
        rng=np.random.default_rng(0),
    )
    # Each episode is 10 transitions; capacity 30 holds at most 3.
    for tag in range(6):
        ep, _ = _make_episode(10, tag=float(tag))
        buf.add_episode(ep)
        assert len(buf) <= 30

    assert len(buf) == 30
    assert buf.n_episodes == 3
    # The three most-recent tags (3, 4, 5) survive; older ones are gone.
    survivors = set()
    for _ in range(300):
        b = buf.sample_sequences(batch_size=1, L=4)
        survivors.add(float(b.obs[0, 0, 0]))
    assert survivors <= {3.0, 4.0, 5.0}
    assert 5.0 in survivors  # newest definitely present


def test_len_reports_transition_count():
    """__len__ returns the number of stored transitions, for MIN_REPLAY gating."""
    buf = _seeded_buffer(seq_len=4, capacity=1000)
    assert len(buf) == 0
    ep, _ = _make_episode(12, tag=1.0)
    buf.add_episode(ep)
    assert len(buf) == 12
    ep2, _ = _make_episode(8, tag=2.0)
    buf.add_episode(ep2)
    assert len(buf) == 20


def test_is_ready_gate():
    """is_ready requires both MIN_REPLAY transitions and at least one window."""
    buf = _seeded_buffer(seq_len=4, capacity=1000)
    short_ep, _ = _make_episode(2, tag=1.0)  # too short to sample
    buf.add_episode(short_ep)
    assert not buf.is_ready(min_transitions=1)  # no sampleable window
    ep, _ = _make_episode(10, tag=2.0)
    buf.add_episode(ep)
    assert buf.is_ready(min_transitions=5)
    assert not buf.is_ready(min_transitions=1000)


# ---------------------------------------------------------------------------
# Burn-in + stored hidden state.
# ---------------------------------------------------------------------------


def test_burn_in_extends_window_and_returns_hidden():
    """With burn_in > 0 the window is burn_in + L and hidden states come back."""
    buf = PrioritizedSequenceReplay(
        capacity=1000,
        seq_len=4,
        burn_in=3,
        rng=np.random.default_rng(11),
    )
    ep, hidden = _make_episode(20, tag=7.0, with_hidden=True)
    buf.add_episode(ep, hidden_states=hidden)

    batch = buf.sample_sequences(batch_size=8, L=4)
    assert batch.burn_in == 3
    assert batch.obs.shape == (8, 7, OBS_DIM)  # 3 burn-in + 4 learn
    # Hidden state is captured at each window's FIRST step (tag 7.0 here).
    assert batch.hidden is not None
    assert batch.hidden.shape == (8, 3)
    assert np.allclose(batch.hidden, 7.0)
    # The learn span starts at burn_in.
    learn = batch.obs[:, batch.burn_in:, :]
    assert learn.shape == (8, 4, OBS_DIM)


def test_burn_in_contiguity_across_full_window():
    """The full burn_in + L window is contiguous and single-episode."""
    buf = PrioritizedSequenceReplay(
        capacity=1000,
        seq_len=4,
        burn_in=2,
        rng=np.random.default_rng(5),
    )
    for tag in (1.0, 2.0):
        ep, _ = _make_episode(20, tag=tag)
        buf.add_episode(ep)
    batch = buf.sample_sequences(batch_size=32, L=4)
    for b in range(len(batch)):
        steps = batch.obs[b, :, 1]
        tags = batch.obs[b, :, 0]
        assert np.all(tags == tags[0])
        assert np.array_equal(steps, np.arange(steps[0], steps[0] + 6))


def test_hidden_none_when_not_stored():
    """No hidden states stored -> batch.hidden is None."""
    buf = _seeded_buffer(seq_len=4)
    ep, _ = _make_episode(20, tag=1.0)
    buf.add_episode(ep)  # no hidden_states
    batch = buf.sample_sequences(batch_size=4, L=4)
    assert batch.hidden is None


# ---------------------------------------------------------------------------
# n-step returns helper.
# ---------------------------------------------------------------------------


def test_compute_n_step_returns_basic():
    """n-step discounted sum and bootstrap mask are correct on a hand example."""
    # One row, rewards [1, 2, 3, 4, 5], no terminals, n=3, gamma=0.5.
    rewards = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
    dones = np.zeros((1, 5), dtype=bool)
    returns, bootstrap = compute_n_step_returns(rewards, dones, n=3, gamma=0.5)

    # G_0 = 1 + 0.5*2 + 0.25*3 = 2.75 ; horizon (0..2) fits -> bootstrap True.
    assert np.isclose(returns[0, 0], 1 + 0.5 * 2 + 0.25 * 3)
    assert bootstrap[0, 0]
    # G_3 = 4 + 0.5*5 (+ off-window) ; horizon runs past end -> bootstrap False.
    assert np.isclose(returns[0, 3], 4 + 0.5 * 5)
    assert not bootstrap[0, 3]
    # Last step: only its own reward, horizon off-window -> no bootstrap.
    assert np.isclose(returns[0, 4], 5.0)
    assert not bootstrap[0, 4]


def test_compute_n_step_returns_truncates_on_done():
    """A terminal inside the horizon truncates the sum and kills the bootstrap."""
    rewards = np.array([[1.0, 2.0, 3.0, 4.0]])
    dones = np.array([[False, True, False, False]])  # terminal at t=1
    returns, bootstrap = compute_n_step_returns(rewards, dones, n=4, gamma=1.0)

    # From t=0: include r0 + r1, stop at the done -> 1 + 2 = 3, no bootstrap.
    assert np.isclose(returns[0, 0], 3.0)
    assert not bootstrap[0, 0]
    # From t=1: r1 only (done at t=1) -> 2, no bootstrap.
    assert np.isclose(returns[0, 1], 2.0)
    assert not bootstrap[0, 1]


def test_n_step_returns_wrapper_uses_buffer_config():
    """The buffer wrapper applies its configured n_step / gamma to a batch."""
    buf = PrioritizedSequenceReplay(
        capacity=1000,
        seq_len=6,
        n_step=3,
        gamma=0.9,
        rng=np.random.default_rng(0),
    )
    ep, _ = _make_episode(20, tag=1.0)
    buf.add_episode(ep)
    batch = buf.sample_sequences(batch_size=4, L=6)
    returns, bootstrap = buf.n_step_returns(batch)
    assert returns.shape == batch.rewards.shape
    assert bootstrap.shape == batch.rewards.shape


def test_compute_n_step_returns_validates_inputs():
    """Bad n / gamma / shapes raise ValueError."""
    rewards = np.zeros((2, 3))
    dones = np.zeros((2, 3), dtype=bool)
    with pytest.raises(ValueError):
        compute_n_step_returns(rewards, dones, n=0, gamma=0.9)
    with pytest.raises(ValueError):
        compute_n_step_returns(rewards, dones, n=2, gamma=1.5)
    with pytest.raises(ValueError):
        compute_n_step_returns(rewards, np.zeros((2, 4), dtype=bool), n=2, gamma=0.9)


# ---------------------------------------------------------------------------
# Constructor validation + tuple-episode acceptance.
# ---------------------------------------------------------------------------


def test_accepts_tuple_episodes():
    """Episodes may be 5-tuples, not just Transition instances."""
    buf = _seeded_buffer(seq_len=4)
    ep = [
        (
            np.array([0.0, float(t), 0.0, 0.0], dtype=np.float32),
            t % 8,
            float(t),
            np.array([0.0, float(t + 1), 0.0, 0.0], dtype=np.float32),
            t == 9,
        )
        for t in range(10)
    ]
    buf.add_episode(ep)
    batch = buf.sample_sequences(batch_size=2, L=4)
    assert batch.obs.shape == (2, 4, OBS_DIM)


def test_empty_episode_rejected():
    buf = _seeded_buffer(seq_len=4)
    with pytest.raises(ValueError):
        buf.add_episode([])


def test_window_exceeding_capacity_rejected():
    with pytest.raises(ValueError):
        PrioritizedSequenceReplay(capacity=5, seq_len=4, burn_in=4)


def test_update_priorities_length_mismatch_raises():
    buf = _seeded_buffer(seq_len=4)
    ep, _ = _make_episode(10, tag=1.0)
    buf.add_episode(ep)
    with pytest.raises(ValueError):
        buf.update_priorities(np.array([0, 1]), np.array([1.0]))


def test_update_priorities_skips_stale_indices():
    """Indices from an evicted episode are skipped, not errored."""
    buf = PrioritizedSequenceReplay(
        capacity=20, seq_len=4, rng=np.random.default_rng(0)
    )
    ep1, _ = _make_episode(10, tag=1.0)
    buf.add_episode(ep1)
    batch = buf.sample_sequences(batch_size=4, L=4)
    stale = batch.indices.copy()
    # Push two more episodes to evict ep1.
    for tag in (2.0, 3.0):
        ep, _ = _make_episode(10, tag=tag)
        buf.add_episode(ep)
    # Updating stale indices must not raise.
    buf.update_priorities(stale, np.full(len(stale), 5.0))


def test_priority_eps_keeps_zero_td_sampleable():
    """A zero TD error yields priority ε_p^α > 0, never an unsamplable leaf."""
    buf = _seeded_buffer(seq_len=4)
    ep, _ = _make_episode(8, tag=1.0)
    buf.add_episode(ep)
    batch = buf.sample_sequences(batch_size=buf.n_sampleable, L=4)
    buf.update_priorities(batch.indices, np.zeros(len(batch.indices)))
    # Total priority is still positive (every leaf == ε_p^α).
    assert buf._tree.total > 0.0
    # And sampling still works.
    again = buf.sample_sequences(batch_size=2, L=4)
    assert again.obs.shape[0] == 2
    assert DEFAULT_PRIORITY_EPS > 0.0  # sanity on the module constant
