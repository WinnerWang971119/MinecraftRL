"""replay — Prioritized SEQUENCE replay buffer for the Dueling-DRQN (R2D2-style).

Stores whole episodes as ordered chunks of transitions (NOT shuffled singletons)
and samples fixed-length, contiguous sub-sequences with proportional priority —
the storage half of spec §5.4 (PER) + §5.5 (DRQN sequence storage).

Pure NumPy, **no torch**: the dev machine runs Python 3.14, which has no torch
wheel yet, and this buffer must import and run (and be tested) without it. The
training loop (T16) converts the sampled numpy batches into tensors at the
boundary; nothing here touches autograd.

------------------------------------------------------------------------------
What a "sample" is — the sampleable-unit / boundary policy
------------------------------------------------------------------------------
The atomic sampleable unit is a **start index** ``(episode_id, t)`` such that the
contiguous window ``[t, t + L)`` lies ENTIRELY inside that one episode. We never
cross an episode boundary and we never pad: an episode of length ``M`` yields
exactly ``M - L + 1`` start indices when ``M >= L``, and ZERO when ``M < L``
(too short to fill a window). This is the "restrict start indices" option from
the task spec, chosen over zero-padding because:

  * padding injects fake transitions into the LSTM rollout and the TD targets,
    which the agent then has to learn to ignore — a contiguous real window has
    none of that ambiguity, and
  * with a burn-in prefix (R2D2) the window is already a warm-up + learn split;
    padding the warm-up would defeat the point of seeding the hidden state from
    real history.

Each start index carries its own priority in the sum-tree, so a long episode
contributes many independently-prioritized windows (overlapping windows share
transitions but are distinct samples — standard in R2D2).

Optional **burn-in** (``burn_in > 0``): a sampled window is split into a burn-in
prefix of ``burn_in`` steps (used only to warm the LSTM hidden state, gradients
detached by T16) followed by ``L`` learn steps. The full returned window is then
``burn_in + L`` steps and a start index is valid only when
``t + burn_in + L <= M``. ``stored_hidden`` (the LSTM state captured at
collection time) is returned alongside so T16 can seed the very first burn-in
cell instead of zeroing it.

------------------------------------------------------------------------------
n-step returns
------------------------------------------------------------------------------
We store enough to form n-step targets and ALSO provide a vectorized helper
(:func:`compute_n_step_returns`) the training loop can call on a sampled batch.
n-step target formation itself (bootstrapping off the target net at ``t + n``)
lives in T16, because it needs the network; this module supplies the discounted
reward sums and the validity mask so T16 only has to add the bootstrap term.
``n`` and ``gamma`` are config on the buffer (n TUNE 3–5, gamma per train_config)
but the buffer does not *require* them to sample — they default to a 1-step,
``gamma`` discounted setup and can be overridden per call.

------------------------------------------------------------------------------
Priority structure
------------------------------------------------------------------------------
A **sum-tree** (a complete binary tree backed by a flat array) over the
per-start-index priorities. ``add`` / ``update`` / ``sample`` are all
O(log capacity). New entries enter at the current MAX priority so every freshly
collected window is sampled at least once before its priority is corrected by
the first :func:`update_priorities`. Capacity is bounded in **transitions**;
the oldest episode is evicted whole when a new episode would overflow, which
keeps stored windows contiguous (we never split an episode across eviction).

Owner: T15 (DQN core track)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "Transition",
    "SequenceBatch",
    "PrioritizedSequenceReplay",
    "compute_n_step_returns",
    "DEFAULT_ALPHA",
    "DEFAULT_BETA0",
    "DEFAULT_PRIORITY_EPS",
]


# ---------------------------------------------------------------------------
# PER defaults (spec §5.4). All TUNE-able via the constructor; these mirror the
# values the plan calls out so the buffer is usable out of the box.
# ---------------------------------------------------------------------------

#: Priority exponent. α=0 -> uniform sampling, α=1 -> full prioritization.
DEFAULT_ALPHA: float = 0.6

#: Initial importance-sampling exponent. Annealed 0.4 -> 1.0 over training.
DEFAULT_BETA0: float = 0.4

#: Small constant added to |TD error| so no transition ever has zero priority
#: (which would make it unsamplable forever). ε_p in the spec.
DEFAULT_PRIORITY_EPS: float = 1e-6


# ---------------------------------------------------------------------------
# Transition / batch containers.
# ---------------------------------------------------------------------------


@dataclass
class Transition:
    """A single environment transition (the unit episodes are built from).

    ``obs``/``next_obs`` are observation vectors (see ``env.observation_spec``);
    nothing here asserts their shape so the buffer stays decoupled from OBS_DIM,
    but every transition in a given buffer is expected to share one shape.

    Attributes:
        obs: Observation at time ``t``.
        action: Discrete action index taken at ``t`` (see ``agent.actions.Macro``).
        reward: Scalar reward received for the transition.
        next_obs: Observation at time ``t + 1``.
        done: Whether ``next_obs`` is terminal (episode ended at this step).
    """

    obs: np.ndarray
    action: int
    reward: float
    next_obs: np.ndarray
    done: bool


@dataclass
class SequenceBatch:
    """A sampled batch of fixed-length sequences plus everything T16 needs.

    Every array has its leading axis of size ``batch_size`` and a time axis of
    size ``T = burn_in + L`` (the full rolled-out window). Arrays are contiguous
    ``float32``/``int64``/``bool`` so they convert to tensors with zero copies.

    Attributes:
        obs: ``(batch, T, obs_dim)`` observations across the window.
        actions: ``(batch, T)`` actions (int64).
        rewards: ``(batch, T)`` rewards (float32).
        next_obs: ``(batch, T, obs_dim)`` next observations across the window.
        dones: ``(batch, T)`` terminal flags (bool).
        hidden: Optional ``(batch, hidden_dim)`` or ``(batch, n_layers, hidden_dim)``
            LSTM hidden state captured at each window's FIRST step, for burn-in
            seeding. ``None`` if no hidden states were stored.
        indices: ``(batch,)`` opaque sum-tree leaf ids — pass straight back to
            :meth:`PrioritizedSequenceReplay.update_priorities`.
        is_weights: ``(batch,)`` importance-sampling weights, normalized so the
            max weight in the batch is exactly 1.0 (float32).
        burn_in: Number of leading steps reserved for hidden-state warm-up
            (gradients detached by T16). The learn span is ``obs[:, burn_in:]``.
    """

    obs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_obs: np.ndarray
    dones: np.ndarray
    hidden: Optional[np.ndarray]
    indices: np.ndarray
    is_weights: np.ndarray
    burn_in: int

    def __len__(self) -> int:
        return int(self.obs.shape[0])


# ---------------------------------------------------------------------------
# Sum-tree — O(log n) prioritized sampling / point update.
#
# A complete binary tree stored in a flat array of size ``2 * capacity``. Leaves
# occupy ``[capacity, 2*capacity)``; internal node ``i`` holds the sum of its two
# children. The root (index 1; index 0 is unused) holds the total priority. We
# expose only what the buffer needs: set a leaf, query the total, and find the
# leaf whose cumulative-sum bucket contains a value in ``[0, total)``.
# ---------------------------------------------------------------------------


class _SumTree:
    """Fixed-capacity sum-tree over non-negative priorities.

    Leaf ``i`` (``0 <= i < capacity``) maps to tree index ``capacity + i``.
    All operations are O(log capacity).
    """

    __slots__ = ("_capacity", "_tree")

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError(f"sum-tree capacity must be positive, got {capacity}")
        self._capacity = capacity
        # float64 for the running sums: many small ε_p priorities summed over a
        # 1e6 buffer lose too much precision in float32 and skew sampling.
        self._tree = np.zeros(2 * capacity, dtype=np.float64)

    @property
    def total(self) -> float:
        """Sum of all leaf priorities (root of the tree)."""
        return float(self._tree[1])

    def leaf_value(self, leaf: int) -> float:
        """Return the stored priority of leaf ``leaf``."""
        return float(self._tree[self._capacity + leaf])

    def set(self, leaf: int, priority: float) -> None:
        """Set leaf ``leaf`` to ``priority`` and propagate the delta to the root.

        ``priority`` must be finite and non-negative. The delta is added up the
        ancestor chain so internal sums stay exact in O(log capacity).
        """
        if not np.isfinite(priority) or priority < 0.0:
            raise ValueError(f"priority must be finite and >= 0, got {priority}")
        idx = self._capacity + leaf
        delta = priority - self._tree[idx]
        self._tree[idx] = priority
        idx >>= 1
        while idx >= 1:
            self._tree[idx] += delta
            idx >>= 1

    def find(self, prefix: float) -> int:
        """Return the leaf id whose cumulative-sum bucket contains ``prefix``.

        Walks from the root choosing the left child while ``prefix`` fits inside
        its sum, otherwise descending right with ``prefix`` reduced by the left
        sum. ``prefix`` is clamped into ``[0, total)`` first so floating-point
        slop at the top of the range can never index past the last live leaf.
        """
        total = self._tree[1]
        if total <= 0.0:
            raise ValueError("sum-tree is empty (total priority is zero)")
        # Clamp strictly below total: a prefix == total would walk off the right
        # edge into an unfilled/zero leaf.
        if prefix < 0.0:
            prefix = 0.0
        elif prefix >= total:
            prefix = np.nextafter(total, 0.0)

        idx = 1
        while idx < self._capacity:  # while not yet at a leaf row
            left = idx << 1
            left_sum = self._tree[left]
            if prefix < left_sum:
                idx = left
            else:
                prefix -= left_sum
                idx = left + 1
        return idx - self._capacity

    def clear_leaf(self, leaf: int) -> None:
        """Zero a leaf (used on eviction). Equivalent to ``set(leaf, 0.0)``."""
        self.set(leaf, 0.0)


# ---------------------------------------------------------------------------
# Internal episode record.
# ---------------------------------------------------------------------------


@dataclass
class _Episode:
    """One stored episode as parallel, contiguous arrays.

    Storing transposed (struct-of-arrays) lets ``sample_sequences`` slice a window
    with three contiguous reads instead of reconstructing per-step dataclasses.

    Attributes:
        obs: ``(M, obs_dim)`` observations.
        actions: ``(M,)`` int64 actions.
        rewards: ``(M,)`` float32 rewards.
        next_obs: ``(M, obs_dim)`` next observations.
        dones: ``(M,)`` bool terminal flags.
        hidden: ``(M, ...)`` per-step LSTM hidden states, or ``None``.
        start_leaf: First sum-tree leaf id owned by this episode (its valid start
            indices map to a contiguous leaf block).
        n_starts: Number of valid window-start indices in this episode.
    """

    obs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_obs: np.ndarray
    dones: np.ndarray
    hidden: Optional[np.ndarray]
    start_leaf: int
    n_starts: int

    @property
    def length(self) -> int:
        return int(self.obs.shape[0])


# ---------------------------------------------------------------------------
# n-step return helper (vectorized, network-free).
# ---------------------------------------------------------------------------


def compute_n_step_returns(
    rewards: np.ndarray,
    dones: np.ndarray,
    *,
    n: int,
    gamma: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute truncated n-step discounted reward sums for a sampled batch.

    For each step ``t`` returns ``G_t = Σ_{k=0}^{n-1} γ^k r_{t+k}`` truncated at
    the first terminal inside the window, plus a ``bootstrap`` mask that is False
    once a ``done`` has been hit (so T16 drops the bootstrap term there) and at
    steps whose full n-step horizon runs past the end of the window.

    This is the network-free half of an n-step target: T16 adds
    ``bootstrap * γ^n * max_a Q_target(s_{t+n}, a)`` to ``G_t``. Keeping it here
    means the reward arithmetic is unit-tested independently of the net.

    Args:
        rewards: ``(batch, T)`` float rewards.
        dones: ``(batch, T)`` bool terminal flags.
        n: n-step horizon (>= 1; TUNE 3–5).
        gamma: Discount factor in [0, 1].

    Returns:
        ``(returns, bootstrap)`` each shaped ``(batch, T)``: ``returns`` is the
        discounted reward sum, ``bootstrap`` (bool) marks where a γ^n bootstrap
        term is valid.

    Raises:
        ValueError: if ``n < 1``, ``gamma`` is out of [0, 1], or shapes mismatch.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if not (0.0 <= gamma <= 1.0):
        raise ValueError(f"gamma must be in [0, 1], got {gamma}")
    rewards = np.asarray(rewards, dtype=np.float64)
    dones = np.asarray(dones, dtype=bool)
    if rewards.shape != dones.shape:
        raise ValueError(
            f"rewards shape {rewards.shape} != dones shape {dones.shape}"
        )
    if rewards.ndim != 2:
        raise ValueError(f"rewards must be 2-D (batch, T), got ndim={rewards.ndim}")

    batch, length = rewards.shape
    returns = np.zeros((batch, length), dtype=np.float64)
    bootstrap = np.ones((batch, length), dtype=bool)

    for t in range(length):
        discount = 1.0
        acc = np.zeros(batch, dtype=np.float64)
        # ``alive`` tracks, per row, whether the horizon is still open (no done
        # consumed yet) AND still inside the window.
        alive = np.ones(batch, dtype=bool)
        horizon_fits = np.ones(batch, dtype=bool)
        for k in range(n):
            j = t + k
            if j >= length:
                # The n-step horizon runs off the end of the sampled window for
                # every row at this offset: no valid bootstrap there.
                horizon_fits[:] = False
                break
            acc += alive * (discount * rewards[:, j])
            # A done at step j ends the return AFTER including r_j; subsequent
            # rewards and the bootstrap are dropped for that row.
            newly_done = alive & dones[:, j]
            alive = alive & ~dones[:, j]
            # Once done, no bootstrap for that row.
            bootstrap[newly_done, t] = False
            discount *= gamma
        returns[:, t] = acc
        # Bootstrap valid only if the row never terminated AND the full horizon
        # fit inside the window.
        bootstrap[:, t] &= alive & horizon_fits

    return returns.astype(np.float32), bootstrap


# ---------------------------------------------------------------------------
# The buffer.
# ---------------------------------------------------------------------------


class PrioritizedSequenceReplay:
    """Prioritized, sequence-structured replay buffer (R2D2-style) for the DRQN.

    Stores whole episodes; samples fixed-length contiguous windows with
    proportional priority and importance-sampling correction. See the module
    docstring for the boundary policy, burn-in handling, and n-step contract.

    The buffer fixes ``L`` (the learn-span length) and ``burn_in`` at
    construction because they determine which start indices are valid — and thus
    how many sum-tree leaves an episode claims. ``sample_sequences`` accepts an
    ``L`` argument only to assert it matches (so the training loop can pass it
    explicitly and fail loudly on a mismatch rather than silently sampling a
    different length).
    """

    def __init__(
        self,
        capacity: int,
        *,
        seq_len: int,
        burn_in: int = 0,
        alpha: float = DEFAULT_ALPHA,
        beta0: float = DEFAULT_BETA0,
        priority_eps: float = DEFAULT_PRIORITY_EPS,
        n_step: int = 1,
        gamma: float = 0.997,
        beta_anneal_steps: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        """Create an empty prioritized sequence buffer.

        Args:
            capacity: Max stored TRANSITIONS (1e5–1e6). Oldest episodes are
                evicted whole when a new episode would overflow.
            seq_len: ``L``, the learn-span length sampled per window (TUNE 8–16).
            burn_in: Leading warm-up steps prepended to each window for LSTM
                hidden-state seeding (R2D2). 0 disables burn-in. The full window
                returned is ``burn_in + seq_len`` steps.
            alpha: PER priority exponent (≈0.6). 0 == uniform.
            beta0: Initial IS exponent (0.4); annealed toward 1.0.
            priority_eps: ε_p added to |TD error| so priorities stay positive.
            n_step: n-step horizon stored config (TUNE 3–5). Used as the default
                by :meth:`n_step_returns`; sampling itself does not require it.
            gamma: Discount factor for n-step returns.
            beta_anneal_steps: If given, :meth:`anneal_beta` ramps β linearly from
                ``beta0`` to 1.0 over this many calls/steps. If ``None``, β only
                changes when you set :attr:`beta` directly.
            rng: NumPy ``Generator`` for sampling. Defaults to a fresh default_rng;
                pass a seeded one (or set the global seed and rely on default) for
                deterministic tests.

        Raises:
            ValueError: on non-positive capacity / seq_len, negative burn_in,
                out-of-range alpha/beta/eps, or a window longer than capacity.
        """
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}")
        if burn_in < 0:
            raise ValueError(f"burn_in must be non-negative, got {burn_in}")
        if alpha < 0.0:
            raise ValueError(f"alpha must be >= 0, got {alpha}")
        if not (0.0 <= beta0 <= 1.0):
            raise ValueError(f"beta0 must be in [0, 1], got {beta0}")
        if priority_eps < 0.0:
            raise ValueError(f"priority_eps must be >= 0, got {priority_eps}")
        if n_step < 1:
            raise ValueError(f"n_step must be >= 1, got {n_step}")
        if not (0.0 <= gamma <= 1.0):
            raise ValueError(f"gamma must be in [0, 1], got {gamma}")
        if beta_anneal_steps is not None and beta_anneal_steps <= 0:
            raise ValueError(
                f"beta_anneal_steps must be positive or None, got {beta_anneal_steps}"
            )

        self._capacity = capacity
        self._seq_len = seq_len
        self._burn_in = burn_in
        self._window = burn_in + seq_len  # full rolled-out window length
        if self._window > capacity:
            raise ValueError(
                f"window (burn_in + seq_len = {self._window}) exceeds capacity "
                f"{capacity}: no episode could ever be stored."
            )

        self._alpha = alpha
        self._beta0 = beta0
        self._beta = beta0
        self._priority_eps = priority_eps
        self._n_step = n_step
        self._gamma = gamma
        self._beta_anneal_steps = beta_anneal_steps

        self._rng = rng if rng is not None else np.random.default_rng()

        # Leaf bookkeeping. The sum-tree is sized to ``capacity`` leaves: each
        # transition can be the start of at most one window, so the number of
        # valid start indices never exceeds the number of stored transitions,
        # which is bounded by ``capacity``. We allocate leaves in a ring and map
        # each live leaf back to its (episode, offset) so sampling resolves to a
        # concrete window.
        self._tree = _SumTree(capacity)
        # Per-leaf reverse map; -1 marks a free/dead leaf.
        self._leaf_episode = np.full(capacity, -1, dtype=np.int64)
        self._leaf_offset = np.full(capacity, -1, dtype=np.int64)
        self._next_leaf = 0  # ring cursor for leaf allocation

        # Episodes are held in a dict keyed by a monotonic id so eviction is O(1)
        # and leaf->episode lookups never go stale on eviction.
        self._episodes: dict[int, _Episode] = {}
        self._episode_order: List[int] = []  # FIFO of live episode ids (oldest first)
        self._next_episode_id = 0

        self._n_transitions = 0  # total stored transitions (for capacity + __len__)
        self._max_priority = 1.0  # max priority seen (new entries enter here)

    # -- configuration / annealing ----------------------------------------

    @property
    def beta(self) -> float:
        """Current importance-sampling exponent β (0.4 -> 1.0 over training)."""
        return self._beta

    @beta.setter
    def beta(self, value: float) -> None:
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"beta must be in [0, 1], got {value}")
        self._beta = value

    def anneal_beta(self, step: int) -> float:
        """Set β by linear annealing from ``beta0`` to 1.0 over ``beta_anneal_steps``.

        ``β = beta0 + (1 - beta0) * min(step / beta_anneal_steps, 1)``. Requires
        ``beta_anneal_steps`` to have been set at construction. Returns the new β.

        Args:
            step: Current training step (>= 0). Clamped to the anneal horizon.

        Raises:
            RuntimeError: if no ``beta_anneal_steps`` was configured.
            ValueError: if ``step`` is negative.
        """
        if self._beta_anneal_steps is None:
            raise RuntimeError(
                "anneal_beta requires beta_anneal_steps to be set at construction; "
                "set buffer.beta directly instead."
            )
        if step < 0:
            raise ValueError(f"step must be non-negative, got {step}")
        frac = min(step / self._beta_anneal_steps, 1.0)
        self._beta = self._beta0 + (1.0 - self._beta0) * frac
        return self._beta

    # -- sizing -----------------------------------------------------------

    def __len__(self) -> int:
        """Number of stored TRANSITIONS (so the loop can gate on MIN_REPLAY)."""
        return self._n_transitions

    @property
    def n_episodes(self) -> int:
        """Number of episodes currently stored."""
        return len(self._episodes)

    @property
    def n_sampleable(self) -> int:
        """Number of valid window start indices currently stored (sum-tree leaves)."""
        return int(np.count_nonzero(self._leaf_episode >= 0))

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def seq_len(self) -> int:
        return self._seq_len

    @property
    def burn_in(self) -> int:
        return self._burn_in

    def is_ready(self, min_transitions: int) -> bool:
        """True iff at least ``min_transitions`` are stored AND a window is sampleable."""
        return self._n_transitions >= min_transitions and self.n_sampleable > 0

    # -- insertion --------------------------------------------------------

    def add_episode(
        self,
        episode: Sequence[Transition] | Sequence[Tuple],
        hidden_states: Optional[Sequence[np.ndarray]] = None,
    ) -> None:
        """Store one episode as an ordered, contiguous chunk of transitions.

        The episode is stored struct-of-arrays. Every valid window-start index
        (those whose ``burn_in + seq_len`` window stays inside the episode) gets
        a fresh sum-tree leaf at the current MAX priority, so each new window is
        guaranteed to be eligible and sampled at least once before its priority
        is corrected.

        Episodes SHORTER than the full window contribute no start indices: they
        are stored (they still count toward ``__len__`` / capacity, and a longer
        following episode is unaffected) but never sampled. This is the
        documented "restrict start indices, no padding" boundary policy.

        Args:
            episode: An ordered sequence of :class:`Transition`, or of
                ``(obs, action, reward, next_obs, done)`` tuples.
            hidden_states: Optional per-step LSTM hidden states captured at
                collection time (one per transition), used to seed burn-in. Must
                match the episode length when provided.

        Raises:
            ValueError: if the episode is empty, longer than capacity, or
                ``hidden_states`` length disagrees with the episode length.
        """
        transitions = list(episode)
        m = len(transitions)
        if m == 0:
            raise ValueError("cannot add an empty episode")
        if m > self._capacity:
            raise ValueError(
                f"episode length {m} exceeds buffer capacity {self._capacity}"
            )

        # --- coerce to struct-of-arrays --------------------------------
        obs_list: List[np.ndarray] = []
        next_obs_list: List[np.ndarray] = []
        actions = np.empty(m, dtype=np.int64)
        rewards = np.empty(m, dtype=np.float32)
        dones = np.empty(m, dtype=bool)

        for i, tr in enumerate(transitions):
            o, a, r, no, d = _unpack_transition(tr)
            obs_list.append(np.asarray(o, dtype=np.float32))
            next_obs_list.append(np.asarray(no, dtype=np.float32))
            actions[i] = int(a)
            rewards[i] = float(r)
            dones[i] = bool(d)

        obs = np.stack(obs_list, axis=0)
        next_obs = np.stack(next_obs_list, axis=0)

        hidden_arr: Optional[np.ndarray] = None
        if hidden_states is not None:
            hidden_list = list(hidden_states)
            if len(hidden_list) != m:
                raise ValueError(
                    f"hidden_states length {len(hidden_list)} != episode length {m}"
                )
            hidden_arr = np.stack(
                [np.asarray(h, dtype=np.float32) for h in hidden_list], axis=0
            )

        # --- make room (evict oldest whole episodes) -------------------
        # Evict until the incoming episode fits. Guaranteed to terminate because
        # m <= capacity, so emptying the buffer always makes room.
        while self._n_transitions + m > self._capacity and self._episode_order:
            self._evict_oldest()

        # --- allocate leaves for this episode's valid start indices ----
        n_starts = max(0, m - self._window + 1)
        start_leaf = self._allocate_leaves(n_starts)

        episode_id = self._next_episode_id
        self._next_episode_id += 1
        ep = _Episode(
            obs=obs,
            actions=actions,
            rewards=rewards,
            next_obs=next_obs,
            dones=dones,
            hidden=hidden_arr,
            start_leaf=start_leaf,
            n_starts=n_starts,
        )
        self._episodes[episode_id] = ep
        self._episode_order.append(episode_id)
        self._n_transitions += m

        # Register each start index in the reverse map and seed it at max priority.
        for offset in range(n_starts):
            leaf = start_leaf + offset
            self._leaf_episode[leaf] = episode_id
            self._leaf_offset[leaf] = offset
            self._tree.set(leaf, self._priority_for(self._max_priority))

    def _allocate_leaves(self, count: int) -> int:
        """Reserve ``count`` consecutive ring leaves, evicting any episodes whose
        leaves they collide with, and return the first leaf id.

        Leaves are handed out from a ring cursor. Because the number of live
        start indices never exceeds ``capacity`` (one per transition at most),
        a contiguous block of ``count`` leaves always fits once we have evicted
        enough transitions above — but the ring cursor may still land on leaves
        belonging to an episode not yet evicted by the transition-count check
        (start indices < transitions). We defensively evict any episode that owns
        a leaf in the target block before claiming it.
        """
        if count == 0:
            # No samplable windows; do not advance the cursor.
            return self._next_leaf

        start = self._next_leaf
        # The block may wrap; allocate without wrapping by resetting to 0 if it
        # would run past the end (leaves are contiguous per episode for cheap
        # offset math, so we never split a block across the ring boundary).
        if start + count > self._capacity:
            start = 0

        # Evict any live episode owning a leaf in [start, start+count).
        block = set(range(start, start + count))
        # Collect colliding episode ids first (avoid mutating while iterating).
        colliding = {
            int(self._leaf_episode[leaf])
            for leaf in block
            if self._leaf_episode[leaf] >= 0
        }
        for ep_id in list(self._episode_order):
            if ep_id in colliding:
                self._evict_episode(ep_id)

        self._next_leaf = (start + count) % self._capacity
        return start

    def _evict_oldest(self) -> None:
        """Evict the oldest live episode (FIFO)."""
        if not self._episode_order:
            return
        self._evict_episode(self._episode_order[0])

    def _evict_episode(self, episode_id: int) -> None:
        """Remove an episode and free all of its sum-tree leaves."""
        ep = self._episodes.pop(episode_id, None)
        if ep is None:
            return
        # Order list may contain the id once; remove it.
        try:
            self._episode_order.remove(episode_id)
        except ValueError:
            pass
        for offset in range(ep.n_starts):
            leaf = ep.start_leaf + offset
            self._tree.clear_leaf(leaf)
            self._leaf_episode[leaf] = -1
            self._leaf_offset[leaf] = -1
        self._n_transitions -= ep.length

    def _priority_for(self, abs_td: float) -> float:
        """Map an absolute TD error to a stored priority: ``(|δ| + ε_p)^α``."""
        return float((abs_td + self._priority_eps) ** self._alpha)

    # -- sampling ---------------------------------------------------------

    def sample_sequences(
        self,
        batch_size: int,
        L: Optional[int] = None,
    ) -> SequenceBatch:
        """Sample ``batch_size`` prioritized, contiguous length-``L`` windows.

        Sampling is proportional to ``p_i^α`` via stratified sampling over the
        sum-tree (the total priority is split into ``batch_size`` equal strata and
        one uniform draw is taken per stratum — the standard PER scheme that
        lowers variance versus independent draws). Each returned window is
        contiguous within ONE episode and never crosses a boundary.

        Importance-sampling weights are ``w_i = (1 / (N · P(i)))^β`` then divided
        by ``max_i w_i`` so the largest weight in the batch is exactly 1.0 (weights
        only scale the update down, never up). ``N`` is the number of sampleable
        windows currently stored.

        Args:
            batch_size: Number of windows to sample (> 0).
            L: Optional assertion of the learn-span length. If given it must equal
                the buffer's configured ``seq_len`` (guards against a caller
                expecting a different length).

        Returns:
            A :class:`SequenceBatch`; see its docstring for array shapes. The time
            axis is ``burn_in + seq_len``; the learn span is ``[burn_in:]``.

        Raises:
            ValueError: if ``batch_size <= 0``, ``L`` disagrees with ``seq_len``,
                or no sampleable windows exist yet.
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if L is not None and L != self._seq_len:
            raise ValueError(
                f"requested L={L} but buffer was built with seq_len={self._seq_len}"
            )

        n_sampleable = self.n_sampleable
        if n_sampleable == 0:
            raise ValueError(
                "no sampleable windows: add episodes of length >= burn_in + seq_len "
                f"({self._window}) first."
            )

        total = self._tree.total
        if total <= 0.0:
            raise ValueError("total priority is zero; cannot sample.")

        window = self._window
        # Output buffers. obs_dim is taken from the first sampled window so the
        # buffer never needs to know OBS_DIM up front.
        leaves = np.empty(batch_size, dtype=np.int64)
        priorities = np.empty(batch_size, dtype=np.float64)

        obs_batch: List[np.ndarray] = []
        next_obs_batch: List[np.ndarray] = []
        actions_batch = np.empty((batch_size, window), dtype=np.int64)
        rewards_batch = np.empty((batch_size, window), dtype=np.float32)
        dones_batch = np.empty((batch_size, window), dtype=bool)
        hidden_batch: List[Optional[np.ndarray]] = []
        any_hidden = False

        segment = total / batch_size
        for b in range(batch_size):
            # Stratified draw within stratum b.
            lo = segment * b
            hi = segment * (b + 1)
            prefix = self._rng.uniform(lo, hi)
            leaf = self._tree.find(prefix)

            # A clamped/edge prefix could in principle land on a freed leaf if the
            # tree mutated mid-batch (it does not here), so guard defensively.
            episode_id = int(self._leaf_episode[leaf])
            if episode_id < 0:
                # Fall back to a uniform re-draw over the whole range.
                leaf = self._tree.find(self._rng.uniform(0.0, total))
                episode_id = int(self._leaf_episode[leaf])

            offset = int(self._leaf_offset[leaf])
            ep = self._episodes[episode_id]

            sl = slice(offset, offset + window)
            obs_batch.append(ep.obs[sl])
            next_obs_batch.append(ep.next_obs[sl])
            actions_batch[b] = ep.actions[sl]
            rewards_batch[b] = ep.rewards[sl]
            dones_batch[b] = ep.dones[sl]

            if ep.hidden is not None:
                hidden_batch.append(ep.hidden[offset])  # state at window start
                any_hidden = True
            else:
                hidden_batch.append(None)

            leaves[b] = leaf
            priorities[b] = self._tree.leaf_value(leaf)

        # --- importance-sampling weights -------------------------------
        # P(i) = p_i / total  (p_i already raised to α when stored).
        probs = priorities / total
        # w_i = (1 / (N * P_i))^β ; guard against any zero prob (shouldn't happen
        # because ε_p keeps priorities positive).
        probs = np.maximum(probs, np.finfo(np.float64).tiny)
        is_weights = (1.0 / (n_sampleable * probs)) ** self._beta
        max_w = is_weights.max()
        if max_w > 0.0:
            is_weights = is_weights / max_w
        is_weights = is_weights.astype(np.float32)

        hidden_out: Optional[np.ndarray] = None
        if any_hidden:
            # Any window whose episode lacked hidden states gets zeros of the
            # right shape so the batch is a single dense array.
            ref = next(h for h in hidden_batch if h is not None)
            hidden_out = np.zeros((batch_size, *ref.shape), dtype=np.float32)
            for b, h in enumerate(hidden_batch):
                if h is not None:
                    hidden_out[b] = h

        return SequenceBatch(
            obs=np.stack(obs_batch, axis=0),
            actions=actions_batch,
            rewards=rewards_batch,
            next_obs=np.stack(next_obs_batch, axis=0),
            dones=dones_batch,
            hidden=hidden_out,
            indices=leaves,
            is_weights=is_weights,
            burn_in=self._burn_in,
        )

    # -- priority update --------------------------------------------------

    def update_priorities(
        self,
        indices: Sequence[int] | np.ndarray,
        td_errors: Sequence[float] | np.ndarray,
    ) -> None:
        """Update sampled leaves with fresh |TD error|: ``p_i = (|δ_i| + ε_p)^α``.

        ``indices`` are the opaque leaf ids returned by :meth:`sample_sequences`.
        Stale indices (whose episode was evicted since sampling) are skipped
        silently — by the time you compute TD errors the buffer may have rolled.
        The running max priority is bumped so subsequent new entries enter at the
        true current max.

        Args:
            indices: Leaf ids from a prior :meth:`sample_sequences` call.
            td_errors: Matching TD errors (signed or absolute; ``abs`` is taken).

        Raises:
            ValueError: if the two arrays differ in length or a TD error is
                non-finite.
        """
        idx = np.asarray(indices, dtype=np.int64).reshape(-1)
        deltas = np.asarray(td_errors, dtype=np.float64).reshape(-1)
        if idx.shape[0] != deltas.shape[0]:
            raise ValueError(
                f"indices ({idx.shape[0]}) and td_errors ({deltas.shape[0]}) "
                "must have the same length"
            )
        if not np.all(np.isfinite(deltas)):
            raise ValueError("td_errors contains non-finite values")

        abs_deltas = np.abs(deltas)
        for leaf, abs_td in zip(idx.tolist(), abs_deltas.tolist()):
            if leaf < 0 or leaf >= self._capacity:
                continue
            if self._leaf_episode[leaf] < 0:
                # Leaf was evicted after sampling; nothing to update.
                continue
            priority = self._priority_for(abs_td)
            self._tree.set(leaf, priority)
            # Track the max in PRE-α space so new entries match the largest |δ|.
            self._max_priority = max(self._max_priority, abs_td)

    # -- n-step convenience ----------------------------------------------

    def n_step_returns(
        self,
        batch: SequenceBatch,
        *,
        n: Optional[int] = None,
        gamma: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute n-step discounted returns + bootstrap mask for a sampled batch.

        Thin wrapper over :func:`compute_n_step_returns` using the buffer's
        configured ``n_step`` / ``gamma`` unless overridden. T16 adds the
        ``γ^n · max_a Q_target`` bootstrap term where the mask is True.
        """
        return compute_n_step_returns(
            batch.rewards,
            batch.dones,
            n=self._n_step if n is None else n,
            gamma=self._gamma if gamma is None else gamma,
        )


# ---------------------------------------------------------------------------
# Module-private helpers.
# ---------------------------------------------------------------------------


def _unpack_transition(tr: object) -> Tuple[np.ndarray, int, float, np.ndarray, bool]:
    """Coerce a Transition or a 5-tuple into ``(obs, action, reward, next_obs, done)``."""
    if isinstance(tr, Transition):
        return tr.obs, tr.action, tr.reward, tr.next_obs, tr.done
    seq = tuple(tr)  # type: ignore[arg-type]
    if len(seq) != 5:
        raise ValueError(
            "each transition must be a Transition or a 5-tuple "
            f"(obs, action, reward, next_obs, done); got length {len(seq)}"
        )
    return seq  # type: ignore[return-value]
