"""opponents/scripted_bot — Stage-1 heuristic opponent (M3 target).

``ScriptedBot`` replaces the M1/M2 ``StationaryDummy`` with a reactive rival:
it flees at low health, attacks when in range and its swing is charged
(strafing and jumping between swings while the meter recharges, rather than
standing still), otherwise closes distance on the target with that same
occasional strafing and jumping, and falls back to turning toward the
target's last-known position when it cannot currently see it.  Two presets — ``EASY`` and ``HARD`` — tune how
often it strafes, jumps, and flees; explicit constructor arguments override
individual preset values without requiring a whole new preset.

The bot is **omniscient**: ``act`` reads ``OpponentView``, a hand-authored
snapshot of raw ground truth with no FOV/line-of-sight/memory gating applied.
This mirrors the plan's decision that the scripted opponent is a training
target, not a fair rival — ``env/perception_filter.py`` (the gating the
*learner* is subject to) is deliberately never imported here.  ``can_see_target``
is always ``True`` for this bot today; the field exists on ``OpponentView``
for a future filtered mode that does not exist yet.

Determinism is the headline requirement (AC7): every probabilistic decision
(strafe, jump, flee) is drawn from an RNG instance the bot owns exclusively
(``random.Random``, never the module-level ``random`` functions), seeded in
``__init__`` and re-seeded by ``reset(seed)`` when an explicit ``seed`` is
given.  ``reset()`` / ``reset(None)`` deliberately leaves the stream alone
(gym convention), so the constructor's seed governs a whole run.  The same
seed replayed against the same fixture sequence therefore reproduces an
identical ``Macro`` sequence, and per-arena instances (T12) never share RNG
state.

Owner: T9 (Demo-day scripted opponent track)
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum

from agent.actions import Macro
from opponents.base import Opponent, OpponentConfig

__all__ = ["OpponentView", "ScriptedPreset", "ScriptedBot"]


@dataclass(frozen=True)
class OpponentView:
    """Omniscient snapshot of combat state handed to ``ScriptedBot.act``.

    Deliberately plain — tuples of floats and bools, no numpy — so a unit
    test can hand-author one without touching the env or the bridge.  Field
    semantics are pinned by the plan's Contracts section:

    self_pos / self_yaw / self_health
        The bot's own position, facing (degrees), and remaining health.
    target_pos / target_yaw / target_health
        The opponent's (learner's) position, facing, and remaining health,
        read as raw ground truth regardless of visibility.
    distance
        Horizontal (XZ-plane) distance from ``self_pos`` to ``target_pos``.
    in_attack_range
        Whether the target is currently within melee range.
    attack_cooldown
        0.0..1.0 gauge on the bot's own weapon swing; 1.0 means fully
        charged and ready to swing again.  **The producer MUST clamp this to
        exactly 1.0** (T11a's shadow tracker, ``raw_opponent_view()``): the
        ready test below is a deliberately tight ``>= 1.0 - 1e-6``, so a
        tracked value a hair under 1.0 (say ``1.0 - 1e-5``) means the bot
        never attacks at all — which surfaces as a mysteriously passive
        opponent rather than as an error.  Values above 1.0 are safe.
    can_see_target
        Always ``True`` for this bot in the current, unfiltered stage.
        Reserved on the type for a future mode that gates it by FOV/LoS —
        that gating is out of scope here and must not be implemented by
        importing ``env/perception_filter.py``.
    last_known_target_pos
        The target's most recently observed position, or ``None`` if it has
        never been seen this episode.  Consulted only when ``can_see_target``
        is ``False``.
    """

    self_pos: tuple[float, float, float]
    self_yaw: float
    self_health: float
    target_pos: tuple[float, float, float]
    target_yaw: float
    target_health: float
    distance: float
    in_attack_range: bool
    attack_cooldown: float
    can_see_target: bool
    last_known_target_pos: tuple[float, float, float] | None


class ScriptedPreset(Enum):
    """Named difficulty tiers for ``ScriptedBot`` (spec-7.2 presets)."""

    EASY = "easy"
    HARD = "hard"


@dataclass(frozen=True)
class _PresetParams:
    """Default probabilities for one ``ScriptedPreset``."""

    p_strafe: float
    p_jump: float
    c_flee: float
    flee_health: float


# EASY never flees (c_flee=0.0), so its flee_health is unreachable by
# construction — the spec table lists it as "—" for that reason.  HARD flees
# unconditionally (c_flee=1.0) once self_health drops to/below 6.0.
_PRESET_PARAMS: dict[ScriptedPreset, _PresetParams] = {
    ScriptedPreset.EASY: _PresetParams(p_strafe=0.15, p_jump=0.05, c_flee=0.0, flee_health=6.0),
    ScriptedPreset.HARD: _PresetParams(p_strafe=0.40, p_jump=0.20, c_flee=1.0, flee_health=6.0),
}

# attack_cooldown is treated as "ready" only within this epsilon of 1.0
# (fully charged).  A tight epsilon — rather than a lenient threshold like
# 0.9 — is deliberate: the hard requirement calls out attacking on an
# uncharged meter as the classic "flailing" failure, so this bot only ever
# swings when the gauge is (for all practical purposes) full, while still
# tolerating float accumulation error in the shadow-tracked cooldown
# (see the plan's OpponentView contract note on attack_cooldown's source).
_ATTACK_COOLDOWN_EPSILON = 1e-6
_ATTACK_READY_THRESHOLD = 1.0 - _ATTACK_COOLDOWN_EPSILON

# Single shared config instance — frozen dataclass, safe to reuse across both
# presets, which declare identical immunity flags.  Unlike the dummy,
# knockback_immune is False: a scripted opponent that cannot be knocked back
# makes the fight unreal (see plan Contracts).
_SCRIPTED_CONFIG = OpponentConfig(
    knockback_immune=False,
    fall_immune=True,
    void_immune=True,
    fixed_spawn=True,
)


def _check_probability(name: str, value: float) -> None:
    """Raise ``ValueError`` unless ``value`` is a probability in ``[0.0, 1.0]``.

    Written as ``not 0.0 <= value <= 1.0`` rather than two ``or``-ed
    comparisons so that ``NaN`` — which fails every ordered comparison — is
    rejected too instead of slipping through as an always-false predicate.
    """
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0.0, 1.0], got {value!r}")


class ScriptedBot(Opponent):
    """Reactive heuristic opponent implementing the spec-7.2 behavior ladder.

    Precedence, evaluated in exactly this order on every ``act`` call::

        if low_health and c_flee:  RETREAT
        elif in_attack_range:      ATTACK if charged, else the movement draw
                                    (strafe/jump), falling back to IDLE
        elif can_see_target:       the movement draw — with prob p_strafe
                                    strafe (L/R), with prob p_jump jump —
                                    falling back to APPROACH
        else:                      turn toward last-known position / search

    ``low_health`` means ``self_health <= flee_health``; whether that
    actually triggers a retreat is then gated by a Bernoulli draw against
    ``c_flee`` (0.0 == never, 1.0 == always, values between give partial
    probability) so the same knob works for both listed presets and for a
    hand-tuned intermediate difficulty.

    Two assumptions not pinned by the spec, made explicit here:

    - When ``in_attack_range`` is ``True`` but the swing is not yet charged,
      the bot never flails (``Macro.ATTACK`` is unreachable on an uncharged
      meter) but neither does it stand still: it takes the same strafe/jump
      movement draw as the approach branch, because a realistic cooldown
      spans three to four decision intervals and a motionless bot in melee
      is barely distinguishable from the dummy it exists to replace — easy
      to simply out-trade, where spec-7.2 asks for "competent but beatable".
      Strafing between hits is also what real PvP players do.
      ``Macro.APPROACH`` is deliberately excluded — closing further when
      already in range buys nothing — so ``Macro.IDLE`` stays the fallback
      for when neither strafe nor jump fires.
    - When ``can_see_target`` is ``False`` and no last-known position has
      ever been recorded (``last_known_target_pos is None``), there is
      nothing to search toward, so the bot returns ``Macro.IDLE``.
    """

    def __init__(
        self,
        preset: ScriptedPreset = ScriptedPreset.EASY,
        *,
        p_strafe: float | None = None,
        p_jump: float | None = None,
        c_flee: float | None = None,
        seed: int | None = None,
    ) -> None:
        """Build a scripted opponent from a preset, with optional overrides.

        Parameters
        ----------
        preset:
            Base parameter set (``ScriptedPreset.EASY`` or ``.HARD``).
        p_strafe, p_jump, c_flee:
            When given, override the preset's corresponding probability.
            ``None`` (the default) keeps the preset's value.
        seed:
            Seed for this bot's private ``random.Random`` instance.  ``None``
            seeds from OS entropy (Python's own ``random.Random`` convention)
            and is not expected to be reproducible; pass an explicit seed for
            deterministic rollouts (AC7).

        Raises
        ------
        ValueError
            If any resolved probability falls outside ``[0.0, 1.0]``, or if
            ``p_strafe + p_jump`` exceeds ``1.0``.  Both are caught here, at
            construction, rather than presenting as inexplicable behavior
            thousands of episodes into a curriculum run (T12).
        """
        defaults = _PRESET_PARAMS[preset]
        self._preset = preset
        self._p_strafe = defaults.p_strafe if p_strafe is None else p_strafe
        self._p_jump = defaults.p_jump if p_jump is None else p_jump
        self._c_flee = defaults.c_flee if c_flee is None else c_flee
        self._flee_health = defaults.flee_health
        # Validate the *resolved* values: an override silently outside [0, 1]
        # degenerates the behavior ladder (p_strafe=1.9 == always strafe) with
        # no other symptom.  Both presets sit well inside the bounds.
        _check_probability("p_strafe", self._p_strafe)
        _check_probability("p_jump", self._p_jump)
        _check_probability("c_flee", self._c_flee)
        # The single-draw movement ladder in `act` partitions [0, 1) into
        # strafe / jump / neither, so the two probabilities may not overshoot.
        if self._p_strafe + self._p_jump > 1.0:
            raise ValueError(
                "p_strafe + p_jump must not exceed 1.0, got "
                f"p_strafe={self._p_strafe!r} + p_jump={self._p_jump!r} "
                f"= {self._p_strafe + self._p_jump!r}"
            )
        # Owned exclusively by this instance — never the global `random`
        # module — so per-arena bots (T12) never share RNG state.
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """``"scripted_easy"`` or ``"scripted_hard"``, per the active preset."""
        return f"scripted_{self._preset.value}"

    # ------------------------------------------------------------------
    # Configuration (bridge-consumed)
    # ------------------------------------------------------------------

    @property
    def config(self) -> OpponentConfig:
        """Immunity flags shared by both presets.

        ``knockback_immune=False`` deliberately differs from the dummy — a
        scripted opponent that cannot be knocked back makes the fight unreal.
        """
        return _SCRIPTED_CONFIG

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None) -> None:
        """Start a new episode, re-seeding the private RNG only if asked.

        ``reset(<int>)`` mirrors ``__init__``'s seeding exactly
        (``random.Random(seed)``), so seeding via ``reset(seed)`` is
        indistinguishable from seeding a fresh instance via
        ``__init__(seed=seed)`` (TC8): the same seed followed by the same
        fixture sequence reproduces the original ``Macro`` sequence.

        ``reset()`` / ``reset(None)`` is a **no-op on the RNG** — the gym /
        gymnasium convention that ``seed=None`` means "do not reseed".  The
        existing stream simply runs on, so the constructor's seed governs the
        whole run while consecutive episodes stay naturally decorrelated.
        This matters because T12 builds one bot per arena and calls the plain
        ``bot.reset()`` every episode: the old ``random.Random(None)`` path
        reseeded from OS entropy there and silently destroyed run
        reproducibility.  Nor is ``None`` mapped to ``0`` (as
        ``env/mc_pvp_env.py`` does for the wire) — that would replay one
        identical stream every episode and correlate all same-seeded arena
        bots with each other.
        """
        if seed is None:
            return
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Policy
    # ------------------------------------------------------------------

    def act(self, observation: OpponentView) -> Macro:
        """Choose a macro per the spec-7.2 behavior ladder.

        Parameters
        ----------
        observation:
            Omniscient ``OpponentView`` snapshot for the current step.

        Returns
        -------
        Macro
            Always a member of ``agent.actions.Macro`` (0..7); this bot never
            returns a bare int or a value outside the frozen action space.
        """
        low_health = observation.self_health <= self._flee_health
        if low_health and self._rng.random() < self._c_flee:
            return Macro.RETREAT

        if observation.in_attack_range:
            if observation.attack_cooldown >= _ATTACK_READY_THRESHOLD:
                return Macro.ATTACK
            # Swing not charged yet — never flail, but never freeze either:
            # juke while the meter refills.  APPROACH is not an option in
            # range, so IDLE is the fallback when no movement fires.
            # (`is not None`, not truthiness: Macro is an IntEnum whose
            # IDLE member is 0 and therefore falsy.)
            movement = self._draw_movement()
            return Macro.IDLE if movement is None else movement

        if observation.can_see_target:
            movement = self._draw_movement()
            return Macro.APPROACH if movement is None else movement

        # Not currently visible: fall back to memory-driven search.
        if observation.last_known_target_pos is not None:
            return Macro.TURN_TO_LAST_SEEN
        # Never seen this episode — nothing to search toward.
        return Macro.IDLE

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _draw_movement(self) -> Macro | None:
        """Draw the shared strafe/jump juke, or ``None`` for "neither fired".

        A **single** uniform draw laddered across the two probabilities, not
        two sequential Bernoulli draws.  The sequential form only reached the
        jump test when the strafe test had already failed, so the realized
        jump rate was ``(1 - p_strafe) * p_jump`` — 0.118 measured against
        HARD's pinned 0.20, a 41% shortfall.  The preset table is a contract
        artifact, so each knob has to mean what it says; the ladder makes
        both marginals exact and costs one draw per step instead of two.
        The constructor's ``p_strafe + p_jump <= 1.0`` check is what keeps
        the jump band from running off the end of the interval.

        Caller-owned fallback: the branch decides what "neither fired" means
        (``APPROACH`` while closing, ``IDLE`` while waiting out a swing).
        """
        draw = self._rng.random()
        if draw < self._p_strafe:
            # Left/right is an independent, deliberately even coin flip.
            return self._rng.choice((Macro.STRAFE_L, Macro.STRAFE_R))
        if draw < self._p_strafe + self._p_jump:
            return Macro.JUMP
        return None
