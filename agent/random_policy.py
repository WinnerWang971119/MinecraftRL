"""random_policy — Uniform random action policy.

Returns a randomly sampled action from the discrete action space on each call.
Used by the M1 tracer bullet (``eval/run_random.py``) to drive the end-to-end
vertical slice without any learned agent. Must be importable with no heavy
dependencies (no torch required at import time) — it uses NumPy only.

------------------------------------------------------------------------------
Seeding (why this matters even for "just random")
------------------------------------------------------------------------------
The action stream MUST be seedable for the M1 tracer to be reproducible.
``agent/seeding.py`` documents the well-known gotcha: seeding NumPy's *global*
RNG does NOT make ``action_space.sample()`` reproducible, because a Gym space (or
any policy) that samples actions carries its OWN generator. This policy follows
that contract: it owns a private :class:`numpy.random.Generator` and draws every
action from it, so the only thing that determines the action sequence is the seed
handed to this object — never the global RNG.

It also exposes a Gym-``Space``-style :meth:`seed` method so the shared
:func:`agent.seeding.seed_action_space` helper can reseed it per episode exactly
as it would a real action space.

Owner: T10 (DQN core track)
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from agent.actions import N_ACTIONS

__all__ = ["RandomPolicy"]


class RandomPolicy:
    """Samples a uniform random macro index in ``[0, n_actions)`` on every call.

    The policy is the M1 stand-in for a learned agent: it ignores the observation
    entirely and returns a uniformly random discrete action. It owns its own
    :class:`numpy.random.Generator` so the action stream is deterministic given a
    seed and is unaffected by the global NumPy RNG (see the module docstring for
    the seeding rationale).

    Args:
        seed: Initial seed for the policy's private generator. ``None`` draws a
            fresh, OS-entropy-seeded generator (non-reproducible — pass an int for
            a reproducible run).
        n_actions: Size of the discrete action space. Defaults to the frozen
            :data:`agent.actions.N_ACTIONS` (== 8); overridable only for tests.

    Raises:
        ValueError: if ``n_actions`` is not a positive integer.
    """

    def __init__(self, seed: Optional[int] = None, n_actions: int = N_ACTIONS) -> None:
        n = int(n_actions)
        if n <= 0:
            raise ValueError(f"n_actions must be a positive integer, got {n_actions!r}")
        self._n_actions = n
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    # -- properties --------------------------------------------------------

    @property
    def n_actions(self) -> int:
        """Number of discrete actions this policy samples over."""
        return self._n_actions

    # -- Gym-Space-style seeding seam --------------------------------------

    def seed(self, seed: Optional[int]) -> None:
        """Reseed the policy's private generator (Gym ``Space.seed`` signature).

        Provided so :func:`agent.seeding.seed_action_space` can reseed this policy
        per episode exactly as it would a real Gym action space. After this call
        the action stream restarts deterministically from ``seed``.

        Args:
            seed: New seed for the private generator (``None`` re-draws from OS
                entropy).
        """
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset the policy at an episode boundary, optionally reseeding.

        The random policy is stateless across steps, so a reset only matters when
        a per-episode ``seed`` is supplied (reproducible exploration). If ``seed``
        is ``None`` the generator is left untouched so the action stream simply
        continues.

        Args:
            seed: Optional per-episode seed. When given, the generator is reseeded
                so the episode's action sequence is reproducible.
        """
        if seed is not None:
            self.seed(seed)

    # -- the policy --------------------------------------------------------

    def act(self, obs: Any = None) -> int:
        """Return a uniformly random action index in ``[0, n_actions)``.

        The observation is ignored (accepted as any type, including ``None``) so
        the runner can hand the policy its standard obs without special-casing.

        Returns:
            A Python ``int`` action index, drawn from the policy's private RNG.
        """
        # ``integers(low, high)`` draws from the half-open interval [low, high),
        # i.e. a valid macro index 0..n_actions-1. Cast to a plain int so callers
        # (env.step, the wire) never see a numpy scalar.
        return int(self._rng.integers(0, self._n_actions))
