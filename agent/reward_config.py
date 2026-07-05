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
    | R_terminal_loss     | −L on loss (stored positive; negated)   | 8.0    |
    | R_terminal_timeout  | timeout penalty (anti-kiting)           | -30.0  |
    | gamma               | discount (also the potential-shaping γ) | 0.99   |
    | c_approach          | potential-shaping weight for Φ          | 0.0    |

Reward-shape rationale (why these values)
------------------------------------------
An earlier symmetric shape (dealt and taken both 1.0/HP, win +8, timeout a 0.0
draw) taught the agent to AVOID combat: a single bad exchange (~19 HP taken)
booked −19, dwarfing the +8 win, so kiting to a 0-reward timeout was the
safest policy. The current shape flips that:

  - **Winning dominates** — ``R_terminal_win = 50`` makes a kill worth far more
    than any damage-trade bookkeeping, so winning stays "the most important
    thing" and damage accumulation never rivals it.
  - **Combat is net-positive** — dealing is rewarded twice as much per heart as
    taking is penalized (``+2/heart`` vs ``−1/heart``); an even trade nets
    positive, so engaging beats disengaging.
  - **Kiting is the worst outcome** — ``R_terminal_timeout = -30`` (below even a
    loss) directly punishes running out the clock. Dying while fighting (−8) is
    strictly better than timing out, so the agent has no incentive to stall.

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
  - ``R_terminal_win`` (50) is kept well above the loss penalty (8) on purpose:
    the agent should chase the win, not play it safe to avoid the loss. Raising
    the loss penalty back toward the win would re-teach the timid, avoid-combat
    behavior this shape was built to fix. ``R_terminal_timeout`` (−30) is set
    below the loss so a timeout is strictly the worst outcome — the anti-kiting
    lever; make it less negative only if the agent starts trading recklessly to
    end episodes fast.
  - ``c_approach`` defaults to 0.0 so the potential-based positional shaping is a
    no-op until T17 tunes Φ — it can never change the optimal policy regardless
    of its value (that is the point of potential-based shaping), but starting at
    0 keeps the day-1 reward exactly the spec formula.

Owner: T5 (Reward/opponent track)
"""

from __future__ import annotations

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
            large so winning dominates every other term.
        R_terminal_loss: Magnitude of the terminal penalty on a loss, stored
            **positive** and subtracted by ``compute_reward``. ``TUNE`` (8.0).
            Deliberately small vs. the win so fear of losing does not deter the
            agent from engaging.
        R_terminal_timeout: Terminal penalty on a timeout. No longer a 0.0 draw:
            timing out means the agent kited / avoided combat, so it is the worst
            outcome (−30.0, below even a loss) to kill that behavior. ``TUNE``.
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

    # --- terminal (win >> loss; timeout is the worst outcome to punish kiting) ---
    R_terminal_win: float = 50.0
    R_terminal_loss: float = 8.0
    R_terminal_timeout: float = -30.0

    # --- discount / potential-based positional shaping ---
    gamma: float = 0.99
    c_approach: float = 0.0
