"""reward_config — Reward shaping weights and hyperparameters (OWNED by T5).

Centralizes every coefficient of the damage-anchored reward (spec §7 /
training-spec §3) so tuning the reward signal touches only this file.
:func:`env.reward.compute_reward` and the training config both import
:class:`RewardConfig`; no coefficient is hardcoded anywhere else.

All defaults are **starting points** and are marked ``TUNE``. Damage is measured
in HP (full health = 20 HP = 10 hearts, so 1 heart = 2 HP); the per-HP damage
coefficients below are quoted in hearts in the comments because that is how the
signal was designed. Current table:

    | coeff               | role                                   | start  |
    |---------------------|----------------------------------------|--------|
    | c_dmg_out           | reward per HP dealt  (= +2 / heart)     | 1.0    |
    | c_dmg_in            | penalty per HP taken (= −1 / heart)     | 0.5    |
    | c_step              | per-step penalty (decisiveness)         | 0.005  |
    | c_aim               | aim shaping, visibility-gated, tiny     | 0.01   |
    | R_terminal_win      | +W on win                               | 50.0   |
    | R_terminal_loss     | −L on loss (stored positive; negated)   | 30.0   |
    | R_terminal_timeout  | timeout penalty (mild, anti-kiting)     | -15.0  |
    | gamma               | discount (also the potential-shaping γ) | 0.99   |
    | c_approach          | potential-shaping weight for Φ          | 0.0    |

Reward-shape rationale (why these values)
------------------------------------------
An earlier symmetric shape (dealt and taken both 1.0/HP, win +8, timeout a 0.0
draw) taught the agent to AVOID combat: a single bad exchange (~19 HP taken)
booked −19, dwarfing the +8 win, so kiting to a 0-reward timeout was the
safest policy. A later shape (win +50, loss −8, timeout −30) fixed that but
introduced a new failure: ``timeout(−30) < loss(−8)`` made deliberate death
strictly better than running out the clock. That was harmless against a dummy
opponent that cannot attack, but becomes actively harmful the moment the
opponent (M3's scripted bot and beyond) can kill the agent — it would learn to
suicide rather than survive to the clock. The current shape fixes both
failures at once:

  - **Winning dominates** — ``R_terminal_win = 50`` makes a kill worth far more
    than any damage-trade bookkeeping, so winning stays "the most important
    thing" and damage accumulation never rivals it.
  - **Combat is net-positive** — dealing is rewarded twice as much per heart as
    taking is penalized (``+2/heart`` vs ``−1/heart``); an even trade nets
    positive, so engaging beats disengaging.
  - **Timeout is strictly the mildest terminal penalty** — ``R_terminal_timeout
    = −15`` sits strictly between ``−R_terminal_loss`` (−30) and
    ``R_terminal_win`` (50), so running out the clock is worse than winning but
    always strictly better than a scored loss. Dying is never the safer option;
    the agent has no incentive to suicide to end an episode early.

Anti-hacking notes for the tuner (T17):
  - ``c_step`` is the single most important knob. Too large → suicide-rushing;
    too small → the agent runs away forever. Re-check it whenever the action
    set or episode length changes. **Finalized at 0.005** (T17): at the spec's
    episode horizon the accumulated step penalty over a full episode stays an
    order of magnitude below a single landed hit (``c_dmg_out`` per HP) and far
    below ``R_terminal_win``, so it nudges decisiveness without ever making
    death (which stops the bleed early) look attractive — i.e. it is too small
    to motivate a suicide-rush, yet large enough that endlessly running away
    is strictly worse than engaging. Tune up only if the agent stalls/kites;
    tune down if it trades recklessly to end episodes.
  - ``c_aim`` is deliberately tiny and **visibility-gated** (see ``compute_reward``)
    so the agent cannot spin in place to farm an always-on aim bonus.
  - ``R_terminal_win`` (50) is kept well above the loss penalty (30) on purpose:
    the agent should chase the win, not play it safe to avoid the loss. Raising
    the loss penalty back toward the win would re-teach the timid, avoid-combat
    behavior this shape was built to fix. ``R_terminal_timeout`` (−15) is kept
    strictly BETWEEN ``−R_terminal_loss`` and ``R_terminal_win`` — a timeout is
    a real penalty (worse than winning) but is never worse than a scored loss,
    so the agent has no incentive to suicide to dodge the clock. Make the
    timeout penalty larger in magnitude (more negative, but never at or below
    ``−R_terminal_loss``) only if the agent starts kiting; make it smaller if it
    starts trading recklessly to end episodes fast.
  - ``c_approach`` defaults to 0.0 so the potential-based positional shaping is a
    no-op until T17 tunes Φ — it can never change the optimal policy regardless
    of its value (that is the point of potential-based shaping), but starting at
    0 keeps the day-1 reward exactly the spec formula.

Owner: T5 (Reward/opponent track)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["RewardConfig"]


@dataclass(frozen=True)
class RewardConfig:
    """Frozen reward coefficients and shaping hyperparameters (all ``TUNE``).

    The defaults reproduce the spec §7 starting reward exactly. Construct a
    modified config with :func:`dataclasses.replace` (the instance is frozen, so
    fields cannot be mutated in place) when sweeping coefficients.

    Attributes:
        c_dmg_out: Reward per HP of damage dealt to the opponent. ``TUNE`` (1.0
            = +2 per heart).
        c_dmg_in: Penalty per HP of damage taken (subtracted). ``TUNE`` (0.5 =
            −1 per heart). Deliberately HALF of ``c_dmg_out`` so an even trade
            nets positive and the agent is rewarded for engaging, not for
            avoiding every hit.
        c_step: Per-step penalty subtracted every step for decisiveness. ``TUNE``
            (0.005). The make-or-break knob: too big → suicide-rush, too small →
            run-away.
        c_aim: Aim-shaping bonus, added only when the opponent is **visible AND**
            under the crosshair. Deliberately tiny and visibility-gated to block
            spin-to-farm. ``TUNE`` (0.01).
        R_terminal_win: Terminal reward added on a win. ``TUNE`` (50.0). Kept
            large so winning dominates every other term. Must be strictly
            positive and strictly greater than ``R_terminal_timeout``.
        R_terminal_loss: Magnitude of the terminal penalty on a loss, stored
            **positive** and subtracted by ``compute_reward`` (i.e. applied as
            ``-R_terminal_loss``). ``TUNE`` (30.0). Must be strictly positive.
            ``-R_terminal_loss`` is the worst achievable terminal value, strictly
            below ``R_terminal_timeout`` — a scored loss must always be worse
            than running out the clock.
        R_terminal_timeout: Terminal penalty on a timeout, applied as-is (already
            signed). ``TUNE`` (−15.0). Must be strictly negative, and strictly
            between ``-R_terminal_loss`` and ``R_terminal_win`` — a timeout is a
            real penalty (worse than winning) but never the worst outcome (a
            scored loss is always worse), so the agent has no incentive to
            suicide rather than survive to the clock.
        gamma: Discount factor γ. Also the γ used in the potential-based shaping
            term ``F(s, s') = gamma·Φ(s') − Φ(s)`` so the discount lives in ONE
            place (the learner reads the same value). ``TUNE`` (0.99).
        c_approach: Weight of the approach potential Φ used by the potential-based
            positional shaping hook. Defaults to 0.0 (shaping is a no-op) until
            T17 tunes Φ. ``TUNE``.
    """

    # --- damage anchors (privileged + fair; from the bridge events block) ---
    # HP units: 1 heart = 2 HP. c_dmg_out = +2/heart, c_dmg_in = -1/heart
    # (dealt weighted 2x taken so engaging beats avoiding).
    c_dmg_out: float = 1.0
    c_dmg_in: float = 0.5

    # --- shaping ---
    c_step: float = 0.005
    c_aim: float = 0.01

    # --- terminal (win > timeout > loss; a scored loss is always the worst
    # outcome so the agent never prefers dying to running out the clock) ---
    R_terminal_win: float = 50.0
    R_terminal_loss: float = 30.0
    R_terminal_timeout: float = -15.0

    # --- discount / potential-based positional shaping ---
    gamma: float = 0.99
    c_approach: float = 0.0

    def __post_init__(self) -> None:
        """Enforce the terminal-reward invariant: finiteness, signs, ordering.

        The three terminal fields interact (``R_terminal_loss`` is stored
        positive and negated at the call site in :func:`env.reward._terminal_reward`)
        so their validity cannot be checked field-by-field — a config that
        passes each field's own sanity check in isolation can still encode a
        nonsense ordering (e.g. a timeout worse than a loss). This guards the
        combined invariant at construction time so a bad config fails loudly
        instead of silently teaching the wrong policy:

          - all three finite (no inf/-inf/nan — they poison every downstream sum),
          - ``R_terminal_win > 0`` and ``R_terminal_loss > 0`` (sign semantics:
            both are stored as positive magnitudes; ``R_terminal_loss`` is
            negated at the point of use),
          - ``R_terminal_timeout < 0`` (stored and applied pre-signed, unlike
            loss),
          - ``-R_terminal_loss < R_terminal_timeout < R_terminal_win`` (a scored
            loss is always the worst terminal outcome, strictly worse than
            timing out; winning is always the best).
        """
        for name in ("R_terminal_win", "R_terminal_loss", "R_terminal_timeout"):
            value = getattr(self, name)
            if not math.isfinite(value):
                raise ValueError(
                    f"RewardConfig.{name} must be finite, got {value!r}"
                )

        if self.R_terminal_win <= 0.0:
            raise ValueError(
                "RewardConfig.R_terminal_win must be strictly positive "
                f"(a win must always be rewarded), got {self.R_terminal_win!r}"
            )
        if self.R_terminal_loss <= 0.0:
            raise ValueError(
                "RewardConfig.R_terminal_loss must be strictly positive "
                "(it is stored as a positive magnitude and negated at the "
                f"call site), got {self.R_terminal_loss!r}"
            )
        if self.R_terminal_timeout >= 0.0:
            raise ValueError(
                "RewardConfig.R_terminal_timeout must be strictly negative "
                f"(it is applied pre-signed), got {self.R_terminal_timeout!r}"
            )
        if not (-self.R_terminal_loss < self.R_terminal_timeout < self.R_terminal_win):
            raise ValueError(
                "RewardConfig terminal ordering violated: require "
                "-R_terminal_loss < R_terminal_timeout < R_terminal_win, got "
                f"-R_terminal_loss={-self.R_terminal_loss!r}, "
                f"R_terminal_timeout={self.R_terminal_timeout!r}, "
                f"R_terminal_win={self.R_terminal_win!r} "
                "(a scored loss must remain the worst outcome, and winning "
                "must remain the best)"
            )
