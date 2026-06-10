"""reward_config — Reward shaping weights and hyperparameters (OWNED by T5).

Centralizes every coefficient of the damage-anchored reward (spec §7 /
training-spec §3) so tuning the reward signal touches only this file.
:func:`env.reward.compute_reward` and the training config both import
:class:`RewardConfig`; no coefficient is hardcoded anywhere else.

All defaults are **starting points** and are marked ``TUNE``. They follow the
spec's start table:

    | coeff               | role                                   | start  |
    |---------------------|----------------------------------------|--------|
    | c_dmg_out           | reward per HP dealt                     | 1.0    |
    | c_dmg_in            | penalty per HP taken                    | 1.0    |
    | c_step              | per-step penalty (decisiveness)         | 0.005  |
    | c_aim               | aim shaping, visibility-gated, tiny     | 0.01   |
    | R_terminal_win      | +W on win                               | 8.0    |
    | R_terminal_loss     | −L on loss (stored positive; negated)   | 8.0    |
    | R_terminal_timeout  | timeout is a draw                       | 0.0    |
    | gamma               | discount (also the potential-shaping γ) | 0.99   |
    | c_approach          | potential-shaping weight for Φ          | 0.0    |

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
  - ``R_terminal_win``/``R_terminal_loss`` start equal (symmetric) inside the
    spec's 5..10 band; raising the loss penalty discourages reckless trades.
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
        c_dmg_out: Reward per HP of damage dealt to the opponent. ``TUNE`` (1.0).
        c_dmg_in: Penalty per HP of damage taken (subtracted). ``TUNE`` (1.0,
            symmetric with ``c_dmg_out`` or slightly smaller).
        c_step: Per-step penalty subtracted every step for decisiveness. ``TUNE``
            (0.005). The make-or-break knob: too big → suicide-rush, too small →
            run-away.
        c_aim: Aim-shaping bonus, added only when the opponent is **visible AND**
            under the crosshair. Deliberately tiny and visibility-gated to block
            spin-to-farm. ``TUNE`` (0.01).
        R_terminal_win: Terminal reward added on a win. ``TUNE`` (8.0, in 5..10).
        R_terminal_loss: Magnitude of the terminal penalty on a loss, stored
            **positive** and subtracted by ``compute_reward``. ``TUNE`` (8.0).
        R_terminal_timeout: Terminal reward on a timeout. A timeout is a draw, so
            this is 0.0 and should stay 0.0 (changing it breaks the draw
            semantics in spec §1.1 / §3).
        gamma: Discount factor γ. Also the γ used in the potential-based shaping
            term ``F(s, s') = gamma·Φ(s') − Φ(s)`` so the discount lives in ONE
            place (the learner reads the same value). ``TUNE`` (0.99).
        c_approach: Weight of the approach potential Φ used by the potential-based
            positional shaping hook. Defaults to 0.0 (shaping is a no-op) until
            T17 tunes Φ. ``TUNE``.
    """

    # --- damage anchors (privileged + fair; from the bridge events block) ---
    c_dmg_out: float = 1.0
    c_dmg_in: float = 1.0

    # --- shaping ---
    c_step: float = 0.005
    c_aim: float = 0.01

    # --- terminal (win = +W, loss = −L, timeout = draw = 0) ---
    R_terminal_win: float = 8.0
    R_terminal_loss: float = 8.0
    R_terminal_timeout: float = 0.0

    # --- discount / potential-based positional shaping ---
    gamma: float = 0.99
    c_approach: float = 0.0
