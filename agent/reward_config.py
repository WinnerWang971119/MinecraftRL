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
    | c_aim               | aim shaping, visibility-gated, tiny     | 0.002  |
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
    so the agent cannot spin in place to farm an always-on aim bonus. It must also
    stay **strictly below** ``c_step`` (enforced in ``__post_init__``; equality
    is REJECTED, and the invariant therefore also implies ``c_step > 0``): the aim
    bonus is paid whenever the opponent is visible and in the crosshair, while
    the step penalty is paid on every step regardless, so a stationary agent
    that simply stares at a visible opponent collects ``+c_aim − c_step`` per
    step forever. At ``c_aim >= c_step`` that nets to zero or positive — a flat
    or uphill degenerate equilibrium with no gradient pushing the agent back
    toward engaging — which is exactly issue #25 (a mutually-staring pair of
    self-play agents converges to an infinite draw). Keeping ``c_aim`` strictly
    less than ``c_step`` makes staring net-negative **at the default
    ``c_approach == 0``**, so standing still and aiming is never a rest state;
    the agent still has to act to stop bleeding reward.

    NOTE FOR THE T17 TUNER, measured not assumed: that guarantee is conditional
    on the potential term being off. In a self-loop state the shaping term pays
    ``γΦ − Φ = (1−γ)·c_approach·d`` per step, so enabling ``c_approach`` reopens
    the stare bonus and reintroduces issue #25 even with ``c_aim < c_step``.
    Measured against this code at normalized ``d = 1``: ``c_approach`` 0.0 →
    −0.003/step, 0.3 → 0.000/step, 1.0 → +0.007/step. The break-even is
    ``c_approach > (c_step − c_aim)/((1−γ)·d)``. The validator deliberately does
    NOT couple ``c_approach`` into the ordering check — that is a new invariant,
    not this one — so if you turn shaping on, re-derive this bound yourself.
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
            spin-to-farm. ``TUNE`` (0.002). Must be strictly less than
            ``c_step`` (enforced in ``__post_init__``, equality included): the
            aim bonus and the step penalty are both paid every step a
            stationary agent stares at a visible, in-crosshair opponent, so if
            ``c_aim >= c_step`` that behavior nets to zero-or-positive reward
            forever — a degenerate stable equilibrium with no gradient out of
            it (issue #25). ``c_aim < c_step`` keeps staring strictly
            net-negative.
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
    c_aim: float = 0.002

    # --- terminal (win > timeout > loss; a scored loss is always the worst
    # outcome so the agent never prefers dying to running out the clock) ---
    R_terminal_win: float = 50.0
    R_terminal_loss: float = 30.0
    R_terminal_timeout: float = -15.0

    # --- discount / potential-based positional shaping ---
    gamma: float = 0.99
    c_approach: float = 0.0

    def __post_init__(self) -> None:
        """Enforce the terminal-reward invariant and the aim/step ordering.

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

        ``c_aim`` and ``c_step`` interact the same way (issue #25): both are
        paid on the same step when the opponent is visible and in the
        crosshair, so a config that passes each coefficient's own
        finite/non-negative check in isolation can still make a stationary,
        staring agent net zero-or-positive reward forever if ``c_aim`` is not
        held strictly below ``c_step``. This guards that combined invariant
        too:

          - both finite (no inf/-inf/nan),
          - both non-negative (a negative shaping coefficient would flip the
            sign of the intended incentive),
          - ``c_aim < c_step`` **strictly** — equality is rejected because at
            ``c_aim == c_step`` staring is exactly free (net 0 per step): a
            flat local optimum with no gradient pushing the agent back toward
            engaging is just as much a degenerate equilibrium as a profitable
            one.
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

        for name in ("c_aim", "c_step"):
            value = getattr(self, name)
            if not math.isfinite(value):
                raise ValueError(
                    f"RewardConfig.{name} must be finite, got {value!r}"
                )
            # `not 0.0 <= value` (rather than `value < 0.0`) also rejects NaN,
            # matching the idiom used elsewhere in this repo (train_config.py,
            # dqn.py, scripted_bot.py) — belt-and-braces
            # with the isfinite check above, since a coefficient that is
            # merely negative would silently flip the sign of the intended
            # incentive.
            if not 0.0 <= value:
                raise ValueError(
                    f"RewardConfig.{name} must be non-negative, got {value!r}"
                )
        if not self.c_aim < self.c_step:
            raise ValueError(
                "RewardConfig aim/step ordering violated: require "
                f"c_aim < c_step (strictly), got c_aim={self.c_aim!r}, "
                f"c_step={self.c_step!r} (a stationary agent that stares at a "
                "visible, in-crosshair opponent is paid c_aim every step it "
                "does so and c_step every step regardless; at c_aim >= c_step "
                "that nets to zero-or-positive reward forever — issue #25 — "
                "so staring must remain strictly net-negative)"
            )
