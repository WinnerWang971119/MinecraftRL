"""opponents/dummy — Stationary idle bot for Stage 0 (M1/M2).

``StationaryDummy`` is the simplest possible opponent: it never moves, never
attacks, and always returns ``Macro.IDLE``.  Its ``OpponentConfig`` declares
all four immunity flags ``True`` so the bridge/server enforces the physical
constraints that make the M2 environment a fully-observed, degenerate MDP:

  - knockback_immune=True  →  bridge sets knockback-resistance ≈ 1.0 (or
                              teleports back to spawn on reset) so hits from
                              the learner agent never displace the dummy.
  - fall_immune=True       →  bridge suppresses fall damage; the dummy cannot
                              die from environmental hazards.
  - void_immune=True       →  bridge teleports the dummy to spawn if it reaches
                              the void (y < 0), preventing accidental episode
                              termination.
  - fixed_spawn=True       →  bridge teleports the dummy to a deterministic
                              coordinate at the start of every episode so the
                              opponent state is always fully known to the learner.

*This class only declares intent and supplies the idle policy.*  None of the
immunity logic runs in Python; it is all enforced server-side by the bridge
(T7a, T8) reading ``self.config``.

Owner: T18 (Reward/opponent track)
"""

from __future__ import annotations

from agent.actions import Macro
from opponents.base import Opponent, OpponentConfig

__all__ = ["StationaryDummy"]

# Single shared config instance — frozen dataclass, safe to reuse.
_DUMMY_CONFIG = OpponentConfig(
    knockback_immune=True,
    fall_immune=True,
    void_immune=True,
    fixed_spawn=True,
)


class StationaryDummy(Opponent):
    """A bot that stands completely still for the entire episode.

    Intended for M1 smoke-testing and M2 baseline evaluation.  Because the
    dummy never moves or attacks, the learner faces a target-practice scenario
    that isolates the reward signal and confirms the training loop is healthy
    before a reactive opponent is introduced.

    The ``reset`` hook is a no-op (the dummy carries no episode-scoped state);
    ``act`` unconditionally returns ``Macro.IDLE`` regardless of the observation.
    """

    @property
    def name(self) -> str:
        return "stationary_dummy"

    @property
    def config(self) -> OpponentConfig:
        """All immunity flags enabled — enforced by the bridge, not Python."""
        return _DUMMY_CONFIG

    # reset() inherits the no-op default from Opponent — no override needed.

    def act(self, observation) -> Macro:
        """Always return IDLE — the dummy never moves or attacks.

        Parameters
        ----------
        observation:
            Ignored.  Accepted as any type (including ``None``) so the env can
            pass its standard obs without special-casing the dummy.

        Returns
        -------
        Macro.IDLE
        """
        return Macro.IDLE
