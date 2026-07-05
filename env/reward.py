"""reward — Damage-anchored reward (FROZEN signature, AC1) + finalized impl (T17).

This module owns the **frozen** ``compute_reward`` signature that both the Gym
env (T9) and the training loop import, plus the finalized spec §7 /
training-spec §3 reward, its single-source-of-truth component decomposition, the
potential-based positional shaping Φ, and the anti-hacking invariants the TC5 /
TC6 battery pins. The signature must stay stable so T9 and T19 build on it
without churn.

Reward (per decision step)
--------------------------
    r = c_dmg_out·damage_dealt − c_dmg_in·damage_taken − c_step
        + c_aim·1[visible AND in_crosshair]          # HARD-gated on visibility
        + R_terminal                                  # episode end only
        + F(s, s')                                    # potential-based shaping

Coefficients live in :class:`agent.reward_config.RewardConfig` (all ``TUNE``).

Single source of truth for components (T9 / T19 / T20)
------------------------------------------------------
:func:`compute_reward_components` returns the per-term additive breakdown and
:func:`compute_reward` returns **exactly the sum** of those components. The env
(T9, ``mc_pvp_env.py``) currently re-derives the same breakdown in its private
``_reward_components`` helper; **T20 should swap that helper for
``compute_reward_components`` so the scalar reward and the logged components can
never drift**. The component keys and signs here are chosen to match the env's
existing ``REWARD_COMPONENT_KEYS`` so the swap is a drop-in:

    r_damage_dealt   = +c_dmg_out·damage_dealt        (>= 0)
    r_damage_taken   = −c_dmg_in·damage_taken         (<= 0, sign baked in)
    r_step           = −c_step                        (<= 0, sign baked in)
    r_aim            = +c_aim if (visible & crosshair) else 0
    r_shaping        = γ·Φ(s') − Φ(s)
    r_terminal       = ±R_terminal at done, else 0

Every component already carries its own sign, so the scalar reward is the plain
``sum(...)`` of the dict values — no caller needs to know which terms subtract.

Privileged-data boundary (spec §5, §2.5)
----------------------------------------
``damage_dealt`` / ``damage_taken`` and the death flags come from the bridge
``events`` block (:class:`bridge.messages.Events`) — privileged but **fair**:
they are the agent's own damage outcomes, not the opponent's hidden state. The
aim bonus and the positional potential read ``visible`` / ``in_crosshair`` /
``opp_pos_local`` from the **GATED** observation vector only, never from raw
opponent state. Opponent raw health is read (if at all) only here in the reward,
never in the observation.

Input-validation contract (T5-review note b)
--------------------------------------------
``compute_reward`` assumes an **already-validated** observation vector — the env
(T9) builds and validates every vector through
:func:`env.observation_spec.build_observation` / ``validate`` before it reaches
the reward, so the reward does not re-run the full range/dtype validation on the
hot path. It DOES enforce two cheap guards that protect the scalar from silent
corruption: a shape guard (a wrong-length vector would read a stale index) and a
finiteness guard on the fields the shaping Φ reads (a NaN/inf opponent position
would otherwise poison the potential and, via bootstrapping, the value target).

Anti-hacking guardrails (training-spec §3 / §8)
-----------------------------------------------
  - ``c_step`` is the make-or-break knob: too large → the agent suicide-rushes
    to end episodes early; too small → it runs away forever. Kept at the spec
    default ``0.005`` (rationale in :class:`RewardConfig`). Plot ``r_step``
    separately when tuning.
  - An always-on aim bonus → spin-to-farm. The aim term is (a) tiny and (b)
    **HARD-gated on visibility**: it is EXACTLY 0 the instant ``visible`` is
    false, regardless of ``in_crosshair`` or any positional value. This is the
    invariant behind AC6 (aim-bonus-while-invisible == 0) and TC6 (no
    spin-to-farm); do not regress it.
  - Positional terms use **potential-based shaping** (``F = γ·Φ(s') − Φ(s)``,
    Ng et al. 1999) so they provably cannot change the optimal policy for ANY
    Φ. Φ is itself visibility-guarded so an unseen/zeroed opponent injects no
    phantom potential. ``c_approach`` defaults to 0 so shaping is a no-op until
    it is tuned; γ is read from ``cfg`` only (one source of truth with the
    learner's discount).

Owner: T5 (signature + initial impl) / T17 (finalize + TC5/TC6) — Reward/opponent track
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from agent.reward_config import RewardConfig
from bridge.messages import Events
from env.observation_spec import OBS_DIM, Obs, field_slice

__all__ = [
    "Events",
    "TermInfo",
    "REWARD_COMPONENT_KEYS",
    "compute_reward",
    "compute_reward_components",
]


# ---------------------------------------------------------------------------
# Chosen types for the signature (DOCUMENTED so T9 and T19 match).
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

    Outcome precedence (T5-review note a)
    -------------------------------------
    The env (the producer, ``mc_pvp_env.py``) resolves a same-step double death
    to a **loss** — ``lost = i_died``; ``won = opponent_died and not i_died`` —
    so it never constructs a contradictory flag set. To keep the reward in
    lock-step with that producer and to catch any future producer bug loudly
    rather than silently mis-scoring an episode, this dataclass:

      - validates in ``__post_init__`` that the flags are self-consistent
        (``won`` and ``lost`` are never both set; an outcome flag is never set
        without ``done``), raising ``ValueError`` if they are; and
      - documents **loss > win > timeout** as the precedence the terminal reward
        applies — matching the env's "double death is a loss" rule. The
        validation makes the contradictory case unreachable, so the precedence
        is a belt-and-braces statement of intent, not a silent tie-break.

    Attributes:
        done: True iff this step ends the episode (any reason).
        won: True iff the episode ended with the learner winning (opponent died
            / lost all health). Adds ``+R_terminal_win``.
        lost: True iff the episode ended with the learner losing (learner died).
            Adds ``−R_terminal_loss``. Takes precedence over ``won``.
        timeout: True iff the episode ended on the step/time cap. Adds
            ``R_terminal_timeout`` — a penalty by default (kiting/avoiding combat
            is the worst outcome), not a neutral draw. See ``RewardConfig``.
    """

    done: bool = False
    won: bool = False
    lost: bool = False
    timeout: bool = False

    def __post_init__(self) -> None:
        # Fail loud on contradictory outcome flags rather than silently scoring
        # an episode the wrong way (T5-review note a). The env never produces
        # these, so a raise here means an upstream producer bug.
        if self.won and self.lost:
            raise ValueError(
                "TermInfo is contradictory: won and lost are both True "
                "(a double death must resolve to lost; see env precedence)."
            )
        if not self.done and (self.won or self.lost or self.timeout):
            raise ValueError(
                "TermInfo has an outcome flag set without done=True "
                f"(won={self.won}, lost={self.lost}, timeout={self.timeout})."
            )


#: Ordered keys of the per-component reward breakdown returned by
#: :func:`compute_reward_components`. Mirrors the env's ``REWARD_COMPONENT_KEYS``
#: (``mc_pvp_env.py``) so T20 can swap the env's private helper for this function
#: without changing any logging/plotting code.
REWARD_COMPONENT_KEYS = (
    "r_damage_dealt",
    "r_damage_taken",
    "r_step",
    "r_aim",
    "r_shaping",
    "r_terminal",
)


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
    optimal policy unchanged for any choice of Φ. Φ here is a pure function of a
    single (gated) observation, which is what keeps the shaping potential-based
    (a term that depended on the *action* or on both states jointly would not).

    The shaping Φ rewards being close to the opponent (closing distance), read
    from the gated ``opp_pos_local`` block — but **only when the opponent is
    visible**, so an unseen/zeroed position never contributes a phantom
    potential (which would leak through the same channel the aim gate protects,
    and would let the agent farm shaping by losing sight of the opponent). Φ is
    scaled by ``cfg.c_approach``, which defaults to 0.0 → the whole shaping term
    is a no-op until it is enabled, keeping the day-1 reward exactly the spec
    formula.

    Args:
        obs: A gated observation vector (current or previous step). Assumed
            already validated by the env; this function additionally treats a
            non-finite ``opp_pos_local`` as "no signal" (returns 0.0) so a stray
            NaN/inf can never poison the potential.
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
    if not np.all(np.isfinite(pos)):
        # Defensive: a non-finite position would make the potential (and hence
        # the bootstrapped value target) NaN. Treat it as no signal.
        return 0.0
    distance = float(np.linalg.norm(pos))
    return -float(cfg.c_approach) * distance


def _shaping(gated: np.ndarray, prev: np.ndarray, cfg: RewardConfig) -> float:
    """Potential-based positional shaping ``F = γ·Φ(s') − Φ(s)``.

    γ is read once from ``cfg.gamma`` (the same discount the learner uses, so the
    shaping stays policy-invariant under that discount). With ``c_approach == 0``
    both potentials are 0 and ``F == 0`` exactly.
    """
    return cfg.gamma * _approach_potential(gated, cfg) - _approach_potential(prev, cfg)


def _terminal_reward(terminal: TermInfo, cfg: RewardConfig) -> float:
    """Terminal reward applied once, only on the step where ``done`` is True.

    Precedence **loss > win > timeout** (matches the env's double-death rule;
    ``TermInfo.__post_init__`` already guarantees ``won`` and ``lost`` are never
    both set, so this ordering is unambiguous):

      - loss → ``−R_terminal_loss``,
      - win  → ``+R_terminal_win``,
      - timeout / unspecified terminal → ``R_terminal_timeout`` (a penalty by
        default: timing out means the agent kited instead of fighting).
    """
    if not terminal.done:
        return 0.0
    if terminal.lost:
        return -float(cfg.R_terminal_loss)
    if terminal.won:
        return float(cfg.R_terminal_win)
    # timeout, or an unspecified terminal — the configured timeout reward
    # (a penalty by default; see RewardConfig.R_terminal_timeout).
    return float(cfg.R_terminal_timeout)


def _check_obs_shape(gated: np.ndarray, prev: np.ndarray) -> None:
    """Cheap shape guard — a malformed vector must fail loudly, never read a stale index.

    The full range/dtype/finiteness validation is the env's job
    (:func:`env.observation_spec.validate`, run before the vector reaches the
    reward); this only guards the shape the reward indexes into.
    """
    if gated.shape != (OBS_DIM,):
        raise ValueError(f"gated_obs must have shape ({OBS_DIM},), got {gated.shape}")
    if prev.shape != (OBS_DIM,):
        raise ValueError(f"prev_obs must have shape ({OBS_DIM},), got {prev.shape}")


# ---------------------------------------------------------------------------
# Component decomposition — the SINGLE source of truth (T9 / T19 / T20).
# ---------------------------------------------------------------------------


def compute_reward_components(
    events: Events,
    gated_obs: np.ndarray,
    prev_obs: np.ndarray,
    terminal: TermInfo,
    cfg: RewardConfig,
) -> Dict[str, float]:
    """Per-term additive breakdown of the step reward (single source of truth).

    Returns a dict keyed by :data:`REWARD_COMPONENT_KEYS`. **Every value already
    carries its own sign**, so the scalar reward is the plain ``sum`` of the
    values — :func:`compute_reward` is implemented as exactly that sum, which is
    why the env (T9) and eval (T19) can log these components and never drift from
    the scalar. T20 should replace the env's private ``_reward_components`` with
    a call to this function.

    Components (each pre-signed):
      - ``r_damage_dealt``  ``= +c_dmg_out·damage_dealt``     (>= 0)
      - ``r_damage_taken``  ``= −c_dmg_in·damage_taken``      (<= 0)
      - ``r_step``          ``= −c_step``                     (<= 0, every step)
      - ``r_aim``           ``= +c_aim`` iff visible & in_crosshair, else 0
      - ``r_shaping``       ``= γ·Φ(gated_obs) − Φ(prev_obs)`` (0 at default coeff)
      - ``r_terminal``      ``= ±R_terminal`` at ``done``, else 0

    Args:
        events: Bridge damage/death events aggregated over this decision interval.
        gated_obs: The current-step gated observation vector (``s'``), shape
            ``(OBS_DIM,)``. Assumed already validated by the env.
        prev_obs: The previous-step gated observation vector (``s``), same shape.
        terminal: Episode-termination flags for this step.
        cfg: Reward coefficients (all ``TUNE``; see :class:`RewardConfig`).

    Returns:
        ``Dict[str, float]`` whose values sum to the scalar ``compute_reward``.

    Raises:
        ValueError: if ``gated_obs`` or ``prev_obs`` is not a length-``OBS_DIM``
            vector, or if ``terminal`` is internally contradictory.
    """
    gated = np.asarray(gated_obs)
    prev = np.asarray(prev_obs)
    _check_obs_shape(gated, prev)

    return {
        "r_damage_dealt": float(cfg.c_dmg_out) * float(events.damage_dealt),
        "r_damage_taken": -float(cfg.c_dmg_in) * float(events.damage_taken),
        "r_step": -float(cfg.c_step),
        "r_aim": _aim_bonus(gated, cfg),
        "r_shaping": float(_shaping(gated, prev, cfg)),
        "r_terminal": _terminal_reward(terminal, cfg),
    }


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

    The return value is **exactly** ``sum(compute_reward_components(...).values())``
    — the components are the single source of truth, so the env (T9) and eval
    (T19) can log the breakdown without it ever disagreeing with this scalar.

    Component breakdown (see :func:`compute_reward_components`):
      - **Damage** (from ``events``, privileged-but-fair): ``+c_dmg_out·damage_dealt``
        and ``−c_dmg_in·damage_taken``.
      - **Step penalty**: ``−c_step`` every step (decisiveness).
      - **Aim bonus**: ``+c_aim`` iff the opponent is ``visible`` AND
        ``in_crosshair`` in ``gated_obs`` — EXACTLY 0 the instant not visible
        (guards AC6 / TC6 spin-farming).
      - **Terminal**: applied only when ``terminal.done``; loss ``−R_terminal_loss``
        (precedence over win), win ``+R_terminal_win``, timeout
        ``R_terminal_timeout`` (a penalty by default — kiting is the worst outcome).
      - **Positional shaping**: potential-based ``F = γ·Φ(gated_obs) − Φ(prev_obs)``
        with ``γ = cfg.gamma``; a no-op while ``cfg.c_approach == 0.0`` and
        provably policy-invariant for any Φ.

    Args:
        events: Bridge damage/death events aggregated over this decision interval.
        gated_obs: The current-step gated observation vector (``s'``), shape
            ``(OBS_DIM,)``. ASSUMED already validated by the env
            (``observation_spec.validate``); the aim term and Φ read it by frozen
            index. A shape guard and a finiteness guard on the shaping fields are
            still enforced cheaply here.
        prev_obs: The previous-step gated observation vector (``s``), same shape.
            Used only by the potential-based positional shaping term.
        terminal: Episode-termination flags for this step.
        cfg: Reward coefficients (all ``TUNE``; see :class:`RewardConfig`).

    Returns:
        The scalar step reward as a Python ``float``.

    Raises:
        ValueError: if ``gated_obs`` or ``prev_obs`` is not a length-``OBS_DIM``
            vector, or if ``terminal`` is internally contradictory.
    """
    components = compute_reward_components(events, gated_obs, prev_obs, terminal, cfg)
    return float(sum(components.values()))
