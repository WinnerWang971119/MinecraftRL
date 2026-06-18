"""serialization — the actor->learner transfer unit (``Episode``) + its wire form.

A collector rolls one whole episode against its arena and hands the learner an
:class:`Episode`: the ordered 5-tuple transitions ``collect_episode`` already
builds, the parallel per-step LSTM hidden snapshots (for R2D2 burn-in seeding),
and a little metadata (which arena, which weight-snapshot version, the
train/serve ``code_version`` skew guard, and the episode's total reward for
logging). In-process this object is passed BY REFERENCE through the
``LocalTransport`` queue, so ``to_dict`` / ``from_dict`` are never called on the
Route-1 hot path.

:meth:`Episode.to_dict` / :meth:`Episode.from_dict` are the dormant Route-2
(networked transport / Redis) serialization boundary. ``to_dict`` lowers every
numpy array (the per-step hidden states AND the obs/next_obs inside each
transition tuple) to plain nested lists and every scalar to a plain
int/float/bool/str, so the result is JSON-friendly with no numpy values leaking
through. ``from_dict`` is the exact inverse: it rebuilds the obs/next_obs and
hidden states as ``float32`` arrays and re-coerces the scalars, so
``from_dict(to_dict(ep))`` reconstructs an equal Episode (arrays ``allclose``
with the same float32 dtype and shape, scalars equal, tuple structure preserved).

Owner: T1 (distributed track)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

__all__ = ["Episode"]


@dataclass(frozen=True)
class Episode:
    """One collected episode — the unit of transfer from a collector to the learner.

    Frozen so a collected episode is an immutable value object: once a collector
    publishes it, neither the learner draining the queue nor the (dormant)
    serialization boundary can mutate it in place.

    Attributes:
        transitions: Ordered list of ``(obs, action, reward, next_obs, done)``
            5-tuples, exactly as ``Trainer.collect_episode`` builds them — ``obs``
            and ``next_obs`` are ``float32`` :class:`numpy.ndarray`, ``action`` is
            an ``int``, ``reward`` a ``float``, ``done`` a ``bool``.
        hidden_states: Parallel list (one per transition) of the LSTM ``(h, c)``
            snapshot captured at collection time, each a ``float32`` array of shape
            ``(2, num_layers, lstm_hidden)``. The learner seeds burn-in from the
            window-start snapshot (R2D2 "stored state").
        arena_id: Which arena produced this episode (0-based).
        policy_version: The :class:`~distributed.weights.WeightStore` version of
            the weight snapshot the collector acted under.
        code_version: The train/serve ``code_version`` stamp (from the bridge
            ``state`` messages); a learner can reject actors whose build differs.
        total_reward: Sum of per-step rewards over the episode, kept for logging.
    """

    transitions: List[Tuple[np.ndarray, int, float, np.ndarray, bool]]
    hidden_states: List[np.ndarray]
    arena_id: int
    policy_version: int
    code_version: str
    total_reward: float

    def to_dict(self) -> Dict[str, Any]:
        """Lower this Episode to a JSON-friendly dict (the Route-2 wire form).

        Every numpy array is converted to plain nested Python lists and every
        scalar to a plain ``int`` / ``float`` / ``bool`` / ``str``, so the result
        contains only dicts / lists / ints / floats / bools / strs — no numpy
        scalars or arrays leak through. This is the inverse of :meth:`from_dict`.

        The obs/next_obs arrays INSIDE each transition tuple are lowered too (not
        just ``hidden_states``); a transition becomes a 5-element list
        ``[obs_list, action, reward, next_obs_list, done]``.

        Returns:
            A dict with keys ``transitions`` / ``hidden_states`` / ``arena_id`` /
            ``policy_version`` / ``code_version`` / ``total_reward``.
        """
        transitions_out: List[list] = []
        for obs, action, reward, next_obs, done in self.transitions:
            transitions_out.append(
                [
                    np.asarray(obs, dtype=np.float32).tolist(),
                    int(action),
                    float(reward),
                    np.asarray(next_obs, dtype=np.float32).tolist(),
                    bool(done),
                ]
            )

        hidden_out: List[list] = [
            np.asarray(h, dtype=np.float32).tolist() for h in self.hidden_states
        ]

        return {
            "transitions": transitions_out,
            "hidden_states": hidden_out,
            "arena_id": int(self.arena_id),
            "policy_version": int(self.policy_version),
            "code_version": str(self.code_version),
            "total_reward": float(self.total_reward),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Episode":
        """Rebuild an Episode from its :meth:`to_dict` form (the Route-2 inverse).

        The obs/next_obs inside each transition and every hidden-state entry are
        reconstructed as ``float32`` numpy arrays; ``action`` / ``reward`` /
        ``done`` are re-coerced to ``int`` / ``float`` / ``bool``. This accepts
        the dict ``to_dict`` produces as well as one that has round-tripped
        through JSON (where the nested values are already plain Python types).

        Args:
            d: A mapping with the keys :meth:`to_dict` writes.

        Returns:
            An :class:`Episode` equal to the one ``to_dict`` was called on.

        Raises:
            KeyError: if a required key is missing.
            ValueError: if a transition is not a 5-element sequence.
        """
        transitions: List[Tuple[np.ndarray, int, float, np.ndarray, bool]] = []
        for tr in d["transitions"]:
            tr = tuple(tr)
            if len(tr) != 5:
                raise ValueError(
                    "each serialized transition must have 5 elements "
                    f"(obs, action, reward, next_obs, done); got length {len(tr)}"
                )
            obs, action, reward, next_obs, done = tr
            transitions.append(
                (
                    np.asarray(obs, dtype=np.float32),
                    int(action),
                    float(reward),
                    np.asarray(next_obs, dtype=np.float32),
                    bool(done),
                )
            )

        hidden_states: List[np.ndarray] = [
            np.asarray(h, dtype=np.float32) for h in d["hidden_states"]
        ]

        return cls(
            transitions=transitions,
            hidden_states=hidden_states,
            arena_id=int(d["arena_id"]),
            policy_version=int(d["policy_version"]),
            code_version=str(d["code_version"]),
            total_reward=float(d["total_reward"]),
        )
