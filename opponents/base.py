"""opponents/base — Abstract interface for all opponent policies.

Defines ``Opponent``, the seam through which the env and training loop interact
with any opposing bot.  Concrete implementations follow a three-stage roadmap:

  Stage 0 (M1/M2, this file):  ``StationaryDummy`` — idle, immunity-declared.
  Stage 1 (M3, deferred):      ``ScriptedBot``     — heuristic chase/attack.
  Stage 2 (M4, deferred):      snapshot opponents drawn from ``SnapshotPool``.

The interface is intentionally minimal: ``reset`` + ``act`` + an ``OpponentConfig``
dataclass that the bridge/server (T7a, T8) reads to enforce physical immunity
constraints server-side.  Nothing here touches Mineflayer or network I/O.

Owner: T18 (Reward/opponent track)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from agent.actions import Macro

__all__ = ["OpponentConfig", "Opponent"]


@dataclass(frozen=True)
class OpponentConfig:
    """Bridge-consumed flags that enforce the opponent's physical constraints.

    These values are read by the bridge/server (tasks T7a and T8) to configure
    the Mineflayer bot before each episode.  They are *not* enforced in Python —
    they declare the intent and let the server-side code act on them:

    knockback_immune
        When ``True`` the bridge sets the knockback-resistance attribute to ≈ 1.0
        (or teleports the bot back to its spawn point every reset) so that hits
        from the learner agent do not displace the bot.  Required for Stage 0 to
        remain a fully-observed, degenerate MDP where the opponent's state never
        changes.

    fall_immune
        When ``True`` the bridge suppresses fall damage (e.g. via game-rule or
        attribute override) so the bot cannot die from environmental hazards.

    void_immune
        When ``True`` the bridge teleports the bot to its spawn point if it ever
        reaches the void (y < 0), preventing episode-ending falls out of the arena.

    fixed_spawn
        When ``True`` the bridge teleports the bot to a deterministic spawn
        coordinate at the start of every episode rather than using the last
        known position.  Required for reproducible, fully-observed M2 episodes.
    """

    knockback_immune: bool
    fall_immune: bool
    void_immune: bool
    fixed_spawn: bool


class Opponent(ABC):
    """Abstract policy interface for all opponent bots.

    Subclasses must implement ``act``.  The ``reset`` hook is optional but
    should be overridden by stateful opponents (scripted bots, snapshot pools).

    The bridge calls ``config`` once during setup to read immunity flags; it
    calls ``reset`` at the start of each episode and ``act`` once per step.

    Observation format
    ------------------
    The ``observation`` argument passed to ``act`` is whatever the env hands to
    the opponent — currently an arbitrary Python object (dict, numpy array, or
    ``None``).  Stage 0 opponents ignore it entirely.  The type is left unbound
    here so that the interface does not pull in the obs-spec dependency before
    the obs contract is frozen.
    """

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Short, stable identifier used in logs and checkpoint paths."""

    # ------------------------------------------------------------------
    # Configuration (bridge-consumed)
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def config(self) -> OpponentConfig:
        """Physical-immunity flags consumed by the bridge/server at setup time.

        See ``OpponentConfig`` for the semantics of each flag.  The property
        must return the same object on every call (frozen dataclass recommended).
        """

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None) -> None:
        """Reset per-episode internal state.

        Called by the env at the start of each episode before the first ``act``
        call.  The default implementation is a no-op; stateful subclasses
        (scripted bots, memory-based policies) should override it.

        Parameters
        ----------
        seed:
            Optional RNG seed for reproducible episode rollouts.  Stateless
            opponents (e.g. ``StationaryDummy``) may ignore it.
        """

    @abstractmethod
    def act(self, observation) -> Macro:
        """Choose a macro action given the opponent's current observation.

        Parameters
        ----------
        observation:
            The opponent's view of the world, as supplied by the env.  The
            exact type depends on the observation spec (T5/T6) and may change
            across milestones.  Stage 0 implementations should accept and ignore
            any value, including ``None``.

        Returns
        -------
        Macro
            One of the eight discrete action macros defined in ``agent.actions``.
        """
