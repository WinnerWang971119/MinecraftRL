"""reward — Damage-anchored reward (frozen signature, AC1) + initial implementation.

This module owns the **frozen** ``compute_reward`` signature that both the Gym
env (T9) and the training loop import, plus a correct initial implementation of
the spec §7 / training-spec §3 reward. T17 later finalizes this file (tunes
``c_step`` and Φ) and adds the full anti-hacking test battery (TC5, TC6); the
signature below must stay stable so T9 and T17 build on it without churn.

Reward (per decision step)
--------------------------
    r = c_dmg_out·damage_dealt − c_dmg_in·damage_taken − c_step
        + c_aim·1[visible AND in_crosshair]          # gated on visibility
        + R_terminal                                  # episode end only
        + F(s, s')                                    # potential-based shaping

Coefficients live in :class:`agent.reward_config.RewardConfig` (all ``TUNE``).

Privileged-data boundary (spec §5, §2.5)
----------------------------------------
``damage_dealt`` / ``damage_taken`` and the death flags come from the bridge
``events`` block (:class:`bridge.messages.Events`) — privileged but **fair**:
they are the agent's own damage outcomes, not the opponent's hidden state. The
aim bonus reads ``visible`` and ``in_crosshair`` from the **GATED** observation
vector only, never from raw opponent state. Opponent raw health is read (if at
all) only here in the reward, never in the observation.

Anti-hacking guardrails (context for T17's tuning, training-spec §3 / §8)
-------------------------------------------------------------------------
  - ``c_step`` is the make-or-break knob: too large → the agent suicide-rushes
    to end episodes early; too small → it runs away forever. Plot the step term
    separately when tuning.
  - An always-on aim bonus → spin-to-farm. The aim term is therefore (a) tiny
    and (b) **hard-gated on visibility**: it is EXACTLY 0 whenever ``visible``
    is false, so spinning while the opponent is unseen accrues nothing. This is
    the invariant behind AC6 (aim-bonus-while-invisible == 0) and TC6 (no
    spin-to-farm). It is enforced unconditionally below, independent of
    ``in_crosshair`` and of any positional value in the vector.
  - Positional terms use **potential-based shaping** (``F = γ·Φ(s') − Φ(s)``) so
    they provably cannot change the optimal policy. The Φ hook is provided here;
    ``c_approach`` defaults to 0 so shaping is a no-op until T17 tunes it.

Owner: T5 (signature + initial impl) / T17 (finalize + TC5/TC6) — Reward/opponent track
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from agent.reward_config import RewardConfig
from bridge.messages import Events
from env.observation_spec import OBS_DIM, Obs, field_slice

__all__ = [
    "Events",
    "TermInfo",
    "compute_reward",
]


# ---------------------------------------------------------------------------
# Chosen types for the signature (DOCUMENTED so T9 and T17 match).
#
# ``Events``    — reused directly from ``bridge.messages`` (re-exported above).
#                 Fields: damage_dealt, damage_taken, i_died, opponent_died.
#                 These are the privileged-but-fair damage anchors.
#
# ``gated_obs`` / ``prev_obs`` — the packed, frozen observation VECTOR
#                 (``np.ndarray`` of shape ``(OBS_DIM,)``, dtype float32) as
#                 produced by ``observation_spec.build_observation`` AFTER the
#                 PerceptionFilter has gated the opponent block. The aim term and
#                 the positional potential read fields by their frozen index via
#                 the ``Obs`` enum / ``field_slice``, never by magic numbers and
#                 never from raw opponent state. ``prev_obs`` is the vector from
#                 the previous step (``s``); ``gated_obs`` is the current one
#                 (``s'``). Using the same vector both tracks pass around keeps
#                 the reward perfectly aligned with what the agent observed.
#
# ``TermInfo``  — defined below: episode-termination flags from the env.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TermInfo:
    """Episode-termination flags for the current step (produced by the env).

    A step is either non-terminal (``done`` False, all others False) or terminal
    in exactly one way. The three outcome flags are mutually exclusive; the
    terminal reward is applied only when ``done`` is True.

    Attributes:
        done: True iff this step ends the episode (any reason).
        won: True iff the episode ended with the learner winning (opponent died
            / lost all health). Adds ``+R_terminal_win``.
        lost: True iff the episode ended with the learner losing (learner died).
            Adds ``−R_terminal_loss``.
        timeout: True iff the episode ended on the step/time cap. A draw → adds
            ``R_terminal_timeout`` (0.0 by default; no win/loss reward).
    """

    done: bool = False
    won: bool = False
    lost: bool = False
    timeout: bool = False


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _aim_bonus(gated_obs: np.ndarray, cfg: RewardConfig) -> float:
    """Visibility-gated aim shaping bonus.

    Returns ``cfg.c_aim`` iff the opponent is BOTH ``visible`` AND
    ``in_crosshair`` in the gated observation, else exactly ``0.0``.

    The visibility gate is the anti-spin-farm invariant (AC6 / TC6): when
    ``visible`` is false the bonus is 0 regardless of ``in_crosshair`` or any
    positional value, so the agent cannot earn aim reward by spinning while the
    opponent is unseen. Both flags are read from the GATED vector by their frozen
    index, never from raw opponent state.
    """
    visible = gated_obs[Obs.VISIBLE] > 0.5
    if not visible:
        return 0.0
    in_crosshair = gated_obs[Obs.IN_CROSSHAIR] > 0.5
    if not in_crosshair:
        return 0.0
    return float(cfg.c_aim)


def _approach_potential(obs: np.ndarray, cfg: RewardConfig) -> float:
    """Potential ``Φ(s)`` for the positional (approach) shaping term.

    Potential-based shaping (Ng et al., 1999) adds ``F(s, s') = γ·Φ(s') − Φ(s)``
    to the reward; because it is a potential difference it provably leaves the
    optimal policy unchanged for any choice of Φ. This is the **hook** T17 tunes;
    it must remain a pure function of a single (gated) observation so the shaping
    stays potential-based.

    The starter Φ rewards being close to the opponent (closing distance), read
    from the gated ``opp_pos_local`` block — but **only when the opponent is
    visible**, so an unseen/zeroed position never contributes a phantom
    potential (which would leak through the same channel the aim gate protects).
    Φ is scaled by ``cfg.c_approach``, which defaults to 0.0 → the whole shaping
    term is a no-op until T17 enables it, keeping the day-1 reward exactly the
    spec formula.

    Args:
        obs: A gated observation vector (current or previous step).
        cfg: Reward config (provides ``c_approach``).

    Returns:
        The scalar potential ``Φ(obs)``.
    """
    if cfg.c_approach == 0.0:
        return 0.0
    if obs[Obs.VISIBLE] <= 0.5:
        # Opponent not visible this step → no positional signal to shape on.
        return 0.0
    # Local-frame opponent position is normalized by POS_SCALE in the vector;
    # its magnitude is the (normalized) distance. Closer ⇒ higher potential, so
    # negate the distance. c_approach scales the closing incentive.
    pos = np.asarray(obs[field_slice("opp_pos_local")], dtype=np.float64)
    distance = float(np.linalg.norm(pos))
    return -float(cfg.c_approach) * distance


def _terminal_reward(terminal: TermInfo, cfg: RewardConfig) -> float:
    """Terminal reward applied once, only on the step where ``done`` is True.

    win → ``+R_terminal_win``; loss → ``−R_terminal_loss``; timeout (draw) →
    ``R_terminal_timeout`` (0.0). If ``done`` is True but no outcome flag is set,
    falls back to the timeout/draw value (the safe, no-bias default).
    """
    if not terminal.done:
        return 0.0
    if terminal.won:
        return float(cfg.R_terminal_win)
    if terminal.lost:
        return -float(cfg.R_terminal_loss)
    # timeout, or an unspecified terminal — treat as a draw (no win/loss reward).
    return float(cfg.R_terminal_timeout)


# ---------------------------------------------------------------------------
# Frozen public signature (AC1). Do not change without a contract PR.
# ---------------------------------------------------------------------------


def compute_reward(
    events: Events,
    gated_obs: np.ndarray,
    prev_obs: np.ndarray,
    terminal: TermInfo,
    cfg: RewardConfig,
) -> float:
    """Compute the scalar reward for one decision step (spec §7 / training-spec §3).

        r = c_dmg_out·dealt − c_dmg_in·taken − c_step
            + c_aim·1[visible AND in_crosshair] + R_terminal + F(s, s')

    Component breakdown:
      - **Damage** (from ``events``, privileged-but-fair): ``+c_dmg_out·damage_dealt``
        and ``−c_dmg_in·damage_taken``.
      - **Step penalty**: ``−c_step`` every step (decisiveness).
      - **Aim bonus**: ``+c_aim`` iff the opponent is ``visible`` AND
        ``in_crosshair`` in ``gated_obs`` — EXACTLY 0 when not visible (guards
        AC6 / TC6 spin-farming).
      - **Terminal**: applied only when ``terminal.done``; win ``+R_terminal_win``,
        loss ``−R_terminal_loss``, timeout/draw ``R_terminal_timeout`` (0.0).
      - **Positional shaping**: potential-based ``F = γ·Φ(gated_obs) − Φ(prev_obs)``
        with ``γ = cfg.gamma``; a no-op while ``cfg.c_approach == 0.0``.

    Args:
        events: Bridge damage/death events aggregated over this decision interval.
        gated_obs: The current-step gated observation vector (``s'``), shape
            ``(OBS_DIM,)``. The aim term and Φ read it by frozen index.
        prev_obs: The previous-step gated observation vector (``s``), same shape.
            Used only by the potential-based positional shaping term.
        terminal: Episode-termination flags for this step.
        cfg: Reward coefficients (all ``TUNE``; see :class:`RewardConfig`).

    Returns:
        The scalar step reward as a Python ``float``.

    Raises:
        ValueError: if ``gated_obs`` or ``prev_obs`` is not a length-``OBS_DIM``
            vector (a wrong-shaped vector would silently corrupt the reward).
    """
    # Shape guard — a malformed vector must fail loudly, never read a stale index.
    gated = np.asarray(gated_obs)
    prev = np.asarray(prev_obs)
    if gated.shape != (OBS_DIM,):
        raise ValueError(
            f"gated_obs must have shape ({OBS_DIM},), got {gated.shape}"
        )
    if prev.shape != (OBS_DIM,):
        raise ValueError(
            f"prev_obs must have shape ({OBS_DIM},), got {prev.shape}"
        )

    # --- damage anchors (privileged + fair) -------------------------------
    reward = cfg.c_dmg_out * float(events.damage_dealt)
    reward -= cfg.c_dmg_in * float(events.damage_taken)

    # --- per-step penalty -------------------------------------------------
    reward -= cfg.c_step

    # --- visibility-gated aim bonus (exactly 0 when not visible) ----------
    reward += _aim_bonus(gated, cfg)

    # --- potential-based positional shaping: F = γ·Φ(s') − Φ(s) -----------
    # gamma is read once from cfg here (not hardcoded; the learner uses the same
    # cfg.gamma), keeping a single source of truth for the discount.
    shaping = cfg.gamma * _approach_potential(gated, cfg) - _approach_potential(
        prev, cfg
    )
    reward += shaping

    # --- terminal reward (episode end only) -------------------------------
    reward += _terminal_reward(terminal, cfg)

    return float(reward)
