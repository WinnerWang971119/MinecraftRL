"""weights — WeightStore + SnapshotPolicy for thread-per-arena collection (T3).

The Ape-X-lite seam (plan §Decisions): the learner is the single writer and the
collectors are readers, on different threads. They must NOT share the live net,
because ``act()`` on a collector thread would read tensors that the learner is
concurrently mutating with ``optimizer.step()`` (a torch read-during-write race).

Two pieces close that gap:

  - :class:`WeightStore` is the publish/subscribe handoff. The learner calls
    :meth:`WeightStore.publish` every K grad steps with ``net.state_dict()`` and a
    monotone version. ``state_dict()`` returns tensor VIEWS into the live net, so
    ``publish`` ``.detach().clone()``s every tensor (onto CPU) before storing.
    After that, the learner can keep stepping the optimizer and the stored
    snapshot will not change (TC3). The store is guarded by a ``threading.Lock``
    so publish (learner thread) and :meth:`latest` (collector threads) never race.

  - :class:`SnapshotPolicy` is one-per-collector. It owns its OWN net clone (built
    from a zero-arg ``net_factory``) and its OWN ``torch.Generator`` seeded from a
    per-collector seed, so the hot path (``act``) takes no lock and no shared RNG.
    :meth:`maybe_refresh` reloads the clone from the store ONLY when the published
    version strictly advances (TC4), and the caller refreshes at episode
    boundaries so the within-episode LSTM trajectory uses one coherent weight set
    (protects TC8b). :meth:`act` runs under ``no_grad`` on the clone, mirroring how
    :class:`agent.train.Trainer` drives ``DuelingDRQN.act``.

Device: CPU. The dev box runs a CPU-only torch wheel and clones live on CPU, so a
collector never touches GPU memory the learner owns.

Owner: T3 (multi-arena throughput track, issue #4)
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn

__all__ = ["WeightStore", "SnapshotPolicy"]


# A torch ``state_dict``: parameter/buffer name -> tensor.
StateDict = Dict[str, torch.Tensor]
# A zero-arg callable returning a fresh net, e.g. ``lambda: DuelingDRQN(**kwargs)``.
NetFactory = Callable[[], nn.Module]


class WeightStore:
    """Lock-guarded weight handoff: the learner publishes, collectors read.

    The single learner thread calls :meth:`publish` every K grad steps; the N
    collector threads call :meth:`latest`. A :class:`threading.Lock` guards the
    stored ``(state_dict, version)`` so a publish never tears against a read.

    The whole point is isolation: :meth:`publish` stores a DETACHED CPU CLONE of
    every tensor, so the learner can keep mutating its live net (``optimizer.step``)
    without changing the stored snapshot, and a collector can ``load_state_dict``
    from it on another thread with no read-during-write race (TC3).

    Contract: the dict returned by :meth:`latest` is the stored snapshot itself
    (not a fresh copy). Callers MUST treat it as read-only — ``load_state_dict``
    copies out of it, which is fine; do NOT mutate it in place. Each ``publish``
    stores a brand-new cloned dict, so a held reference stays valid and unchanged.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state_dict: Optional[StateDict] = None
        # -1 means "nothing published yet"; the first publish must use a version
        # >= 0 so a collector's initial ``self.version == -1`` always advances.
        self._version: int = -1

    def publish(self, state_dict: StateDict, version: int) -> None:
        """Store a detached CPU clone of ``state_dict`` under ``version``.

        Called by the learner thread. ``state_dict`` is expected to be a live
        ``net.state_dict()`` whose tensors are VIEWS into the net's parameters and
        buffers; we ``.detach().clone()`` each one onto CPU so the snapshot is fully
        decoupled from the live net (the learner may mutate it the instant this
        returns). Non-tensor entries (none in this net today, but ``state_dict``
        may carry them) are copied through by reference.

        Args:
            state_dict: The learner net's ``state_dict()`` (tensor views).
            version: A monotone snapshot version. Collectors reload only when this
                strictly exceeds the version they last loaded; pass a value that
                never decreases across publishes (e.g. the learner grad step).

        Raises:
            ValueError: if ``version`` is negative (``-1`` is reserved for the
                empty store, so a published version must be ``>= 0``).
        """
        if version < 0:
            raise ValueError(f"version must be >= 0, got {version}")

        # Clone OUTSIDE the lock: cloning is the expensive part and touches only
        # the caller's tensors, so we do not hold the lock across it. Only the
        # pointer swap below is critical-section work.
        cloned: StateDict = {}
        for key, value in state_dict.items():
            if torch.is_tensor(value):
                # detach() drops autograd history; clone() copies storage so the
                # snapshot does not alias the live parameter; .cpu() pins it on CPU
                # (a no-op when already CPU). Order: detach before clone so the
                # clone carries no grad graph.
                cloned[key] = value.detach().clone().cpu()
            else:
                # Pass non-tensor state through unchanged (rare; future-proofing).
                cloned[key] = value

        with self._lock:
            self._state_dict = cloned
            self._version = version

    def latest(self) -> Tuple[Optional[StateDict], int]:
        """Return the most recently published ``(state_dict, version)``.

        Called by collector threads. The returned dict is the stored cloned
        snapshot; per the class contract the caller must not mutate it (treat it as
        read-only — ``load_state_dict`` reads out of it safely). When nothing has
        been published yet, returns ``(None, -1)``.

        Returns:
            ``(state_dict, version)`` — ``state_dict`` is ``None`` until the first
            publish; ``version`` is ``-1`` until then.
        """
        with self._lock:
            return self._state_dict, self._version


class SnapshotPolicy:
    """A per-collector net clone + private RNG that acts on a synced snapshot.

    Each collector thread owns one of these. It builds its OWN net from
    ``net_factory`` (a zero-arg callable, e.g. ``lambda: DuelingDRQN(**net_kwargs)``)
    and its OWN ``torch.Generator`` seeded from ``generator_seed`` — no shared net,
    no shared generator, so the rollout hot path takes no lock.

    The clone is set to ``.eval()`` after construction and acts under ``no_grad``
    (via :meth:`DuelingDRQN.act`, which is itself ``@torch.no_grad()``), so a
    collector never builds an autograd graph.

    :meth:`maybe_refresh` is the only place the clone's weights change; the caller
    invokes it at EPISODE BOUNDARIES so a whole episode's LSTM trajectory comes from
    one coherent weight set (protects the learner's R2D2 recurrence gate, TC8b).

    Args:
        net_factory: Zero-arg callable returning a fresh net (the clone the policy
            owns). Must return a net with the ``DuelingDRQN`` inference surface:
            ``act(obs_tensor, hidden, *, epsilon, generator) -> (action, hidden)``
            and ``init_hidden(batch, device)``.
        generator_seed: Seed for this policy's private ``torch.Generator`` (the
            per-arena ε-greedy stream). The caller picks a per-collector seed so
            arenas explore differently yet reproducibly.

    Attributes:
        net: The owned net clone (CPU, ``.eval()``).
        version: The :class:`WeightStore` version currently loaded into the clone;
            ``-1`` until the first successful :meth:`maybe_refresh`.
    """

    def __init__(self, net_factory: NetFactory, generator_seed: int) -> None:
        # CPU only: the dev box has a CPU-only torch wheel and the learner publishes
        # CPU clones, so the clone, its generator, and every obs tensor stay on CPU.
        self._device = torch.device("cpu")

        self.net = net_factory().to(self._device)
        # Inference-only clone: no dropout/batchnorm drift, no grad graph.
        self.net.eval()

        # Private ε-greedy RNG. Seeded per collector so the exploration stream is
        # reproducible AND distinct across arenas, with no shared-generator lock on
        # the hot path.
        self._generator = torch.Generator(device=self._device)
        self._generator.manual_seed(int(generator_seed))

        # No snapshot loaded yet; the first store version (>= 0) always advances.
        self.version: int = -1

    def maybe_refresh(self, store: WeightStore) -> None:
        """Reload the clone from ``store`` iff a newer version was published.

        Reads ``store.latest()`` and ``load_state_dict``s ONLY when the published
        version is STRICTLY greater than the version already loaded (so the same
        version is never reloaded — TC4). With nothing published yet (or no version
        advance), this is a no-op and the clone keeps its current weights. Call this
        at episode boundaries so within-episode weights stay coherent.

        Args:
            store: The shared :class:`WeightStore` the learner publishes to.
        """
        state_dict, version = store.latest()
        if state_dict is None:
            # Nothing published yet — keep the freshly-initialized clone.
            return
        if version <= self.version:
            # Same (or older) snapshot already loaded; no reload (TC4).
            return

        # The stored dict is read-only per the WeightStore contract; load_state_dict
        # copies tensors INTO the clone's parameters, leaving the store's snapshot
        # untouched. Strict=True catches any architecture mismatch loudly.
        self.net.load_state_dict(state_dict)
        self.version = version

    @torch.no_grad()
    def act(
        self, obs: Any, hidden: Any, epsilon: float
    ) -> Tuple[int, Any]:
        """ε-greedy single-step action on the snapshot clone (no gradient).

        Mirrors how :class:`agent.train.Trainer` drives the net during rollout:
        coerce ``obs`` to a ``float32`` tensor on CPU, advance the clone's LSTM by
        one step with the policy's own generator for ε-greedy sampling, and return
        the chosen macro plus the advanced hidden state.

        Args:
            obs: A single observation — a numpy array, sequence, or tensor of width
                ``OBS_DIM`` (the net's ``act`` accepts ``(OBS_DIM,)``,
                ``(1, OBS_DIM)`` or ``(1, 1, OBS_DIM)``).
            hidden: The carried LSTM hidden state, or ``None`` to zero-init at the
                start of an episode.
            epsilon: Exploration rate in ``[0, 1]`` for this episode.

        Returns:
            ``(action, new_hidden)`` — ``action`` is a Python ``int`` in
            ``[0, N_ACTIONS)``; ``new_hidden`` is the advanced LSTM state to carry
            into the next step.
        """
        # Convert to a float32 CPU tensor unless it already is one. as_tensor avoids
        # a copy when obs is already a matching tensor; the net's act() does its own
        # single-step shape coercion (OBS_DIM,) / (1, OBS_DIM) / (1, 1, OBS_DIM).
        if torch.is_tensor(obs):
            obs_tensor = obs.to(dtype=torch.float32, device=self._device)
        else:
            obs_tensor = torch.as_tensor(
                obs, dtype=torch.float32, device=self._device
            )

        action, new_hidden = self.net.act(
            obs_tensor,
            hidden,
            epsilon=epsilon,
            generator=self._generator,
        )
        return int(action), new_hidden
