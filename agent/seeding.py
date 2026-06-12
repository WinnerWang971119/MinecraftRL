"""seeding — Determinism helpers: seed propagation across Python, NumPy, PyTorch, Gym.

Provides the seeding primitives that make a run reproducible:

  - :func:`seed_everything` seeds the Python ``random`` module, NumPy, and (if it
    is installed) PyTorch — CPU and CUDA — plus cuDNN's deterministic flags, in a
    single call. Call it once at process startup.
  - :func:`seed_action_space` seeds a Gym-style action space's own RNG. Call it
    **per episode** (with a per-episode seed) so exploration is reproducible.

------------------------------------------------------------------------------
Why PyTorch is imported lazily
------------------------------------------------------------------------------
``torch`` is imported inside a ``try``/``except`` rather than at module top level.
The dev machine runs Python 3.14, which may not yet have a matching ``torch``
wheel; this module must still import and seed Python + NumPy without it. When
torch is absent we simply skip the torch-specific seeding — no hard failure — and
report it via the return value so callers can log the gap.

------------------------------------------------------------------------------
Gotcha (note for T16): ε decays per EPISODE, not per step
------------------------------------------------------------------------------
This module only supplies seeding primitives; it does not own the ε-greedy
schedule. But it is the natural place to record the gotcha that bites here: the
exploration rate ε is decayed **once per episode**, NOT once per environment
step. Decaying per step collapses ε far too fast (hundreds of steps per episode)
and silently kills exploration. T16's training loop owns that schedule — keep the
decay on the episode boundary, and reseed the action space there (see
:func:`seed_action_space`).

Owner: T6 (DQN core track / shared contract)
"""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np

__all__ = ["seed_everything", "seed_action_space"]


def seed_everything(seed: int, *, set_pythonhashseed: bool = True) -> bool:
    """Seed every global RNG we can reach for a reproducible run.

    Seeds, in order:
      - the ``PYTHONHASHSEED`` env var (process-wide hash randomization), when
        ``set_pythonhashseed`` is true — note this only affects child processes
        and interpreters started *after* this call, not the current one,
      - the Python standard-library ``random`` module,
      - NumPy's legacy global RNG (``np.random.seed``),
      - PyTorch CPU and all-CUDA RNGs, and cuDNN's determinism flags
        (``deterministic=True``, ``benchmark=False``) — **only if torch is
        installed**.

    torch is imported lazily; if it is not available the torch step is skipped
    cleanly (no exception). This keeps the module usable on interpreters that
    lack a torch wheel (e.g. a bleeding-edge Python).

    Args:
        seed: The base seed. Must be a non-negative integer.
        set_pythonhashseed: Also set ``PYTHONHASHSEED`` for reproducible hashing
            in subprocesses. Defaults to True.

    Returns:
        ``True`` if PyTorch was found and seeded, ``False`` if torch was absent
        (so callers can log "torch not installed — torch RNG not seeded").

    Raises:
        ValueError: if ``seed`` is negative.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")

    if set_pythonhashseed:
        # Affects only interpreters launched after this point; harmless otherwise.
        os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    return _seed_torch(seed)


def _seed_torch(seed: int) -> bool:
    """Seed PyTorch (CPU + CUDA) and force deterministic cuDNN, if torch exists.

    Imported lazily and guarded so a missing torch install is a no-op, not an
    error. Returns whether torch was actually seeded.
    """
    try:
        import torch
    except ImportError:
        return False

    torch.manual_seed(seed)
    # Seeds every visible CUDA device; safe to call even with no GPU present.
    torch.cuda.manual_seed_all(seed)

    # Force deterministic convolution algorithms. ``benchmark=False`` stops cuDNN
    # from auto-tuning (which picks nondeterministic kernels); ``deterministic``
    # pins the deterministic ones. Both are needed for reproducible CUDA runs.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return True


def seed_action_space(action_space: Any, seed: int) -> None:
    """Seed a Gym-style action space's internal RNG.

    This is the well-known gotcha that ``seed_everything`` alone does NOT cover:
    a Gym ``Space`` carries its **own** ``np.random.Generator``, and
    ``action_space.sample()`` draws from that generator — *not* from the global
    NumPy RNG. So seeding NumPy globally does nothing for ε-greedy exploration
    that samples random actions via ``action_space.sample()``; the random-action
    stream stays unseeded and runs diverge.

    Seeding the action space makes the exploration (ε-greedy random actions)
    reproducible. Call this **per episode** with a per-episode seed (e.g.
    ``base_seed + episode_index``) so each episode's random-action sequence is
    deterministic and replayable.

    Args:
        action_space: A Gym/Gymnasium ``Space`` (or anything exposing a
            ``seed(int)`` method).
        seed: The seed for this action space / episode.

    Raises:
        AttributeError: if ``action_space`` has no ``seed`` method (surfaced
            rather than swallowed — a spaceless object here is a caller bug).
    """
    action_space.seed(seed)
