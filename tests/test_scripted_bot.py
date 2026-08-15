"""Tests for opponents/scripted_bot.py -- ScriptedBot, OpponentView, ScriptedPreset (T10).

Coverage (spec Testing Strategy TC1-TC10, hand-authored ``OpponentView``
fixtures -- no server, no bridge, no network, no numpy):

  - TC1  Low health + ``c_flee=1.0`` -> ``Macro.RETREAT``.
  - TC2  In attack range, cooldown charged -> ``Macro.ATTACK`` (including the
         exact ``1.0 - 1e-6`` ready threshold and values above ``1.0``).
  - TC3  In attack range, cooldown NOT charged -> never ``ATTACK``, never
         ``APPROACH`` (falls through to the strafe/jump/idle movement draw).
  - TC4  Visible, far, ``p_strafe=1.0, p_jump=0.0`` -> ``STRAFE_L`` or
         ``STRAFE_R``; the ``p_strafe=1.0``-alone ``ValueError`` trap (sums to
         1.05 against EASY's default ``p_jump=0.05``).
  - TC5  Visible, far, ``p_strafe=0, p_jump=0`` -> ``Macro.APPROACH``.
  - TC6  Not visible with a last-known position -> ``TURN_TO_LAST_SEEN``
         (see the test's own comment for why the spec's "or APPROACH"
         latitude is not actually exercisable by this bot); not visible
         with no last-known position -> ``IDLE``.
  - TC7  Same seed, same fixture sequence, two instances -> identical
         ``Macro`` sequences (AC7).
  - TC8  ``reset(seed)`` re-seeds and repeats the original sequence;
         ``reset()`` / ``reset(None)`` is a no-op that continues the existing
         RNG stream rather than replaying or reseeding from OS entropy.
  - TC9  EASY vs HARD, N=10000 samples, seed=0 -> HARD strafes/jumps/flees
         strictly more often (fixed N and seed -- deterministic, not
         statistical).
  - TC10 ``act()`` always returns a valid ``Macro`` member, across every
         branch of the ladder.

  Also covered: construction validation (``ValueError`` for out-of-range and
  ``NaN`` probabilities, and ``p_strafe + p_jump > 1.0``); the in-melee
  recharging branch (never ``ATTACK``, never ``APPROACH``); ``config()``'s
  shared-object identity and ``knockback_immune=False`` contract (unlike
  ``StationaryDummy``); ``name``; the ``flee_health`` boundary (HARD
  flees at exactly 6.0 -- inclusive ``<=`` -- EASY never flees even at 1 HP);
  the flee-vs-attack precedence at the joint where both guards fire on the
  same fixture (RETREAT must win the ladder, gated on ``c_flee`` and not on
  health alone); and ``OpponentView``'s frozen-dataclass contract.

Note: ``Macro`` is an ``IntEnum`` with ``IDLE = 0``, which is falsy in
Python.  Every assertion below uses ``is``/``==``/``in``, never bare
truthiness (``assert result`` or ``if macro:``).
"""

from __future__ import annotations

import math

import pytest

from agent.actions import Macro
from opponents.base import Opponent, OpponentConfig
from opponents.scripted_bot import OpponentView, ScriptedBot, ScriptedPreset


# ---------------------------------------------------------------------------
# Fixture helper
# ---------------------------------------------------------------------------

def _view(
    *,
    self_pos: tuple[float, float, float] = (0.0, 64.0, 0.0),
    self_yaw: float = 0.0,
    self_health: float = 20.0,
    target_pos: tuple[float, float, float] = (5.0, 64.0, 0.0),
    target_yaw: float = 180.0,
    target_health: float = 20.0,
    distance: float = 5.0,
    in_attack_range: bool = False,
    attack_cooldown: float = 1.0,
    can_see_target: bool = True,
    last_known_target_pos: tuple[float, float, float] | None = None,
) -> OpponentView:
    """Build an OpponentView with sane defaults, overriding only what a test cares about."""
    return OpponentView(
        self_pos=self_pos,
        self_yaw=self_yaw,
        self_health=self_health,
        target_pos=target_pos,
        target_yaw=target_yaw,
        target_health=target_health,
        distance=distance,
        in_attack_range=in_attack_range,
        attack_cooldown=attack_cooldown,
        can_see_target=can_see_target,
        last_known_target_pos=last_known_target_pos,
    )


def _mixed_sequence() -> list[OpponentView]:
    """A fixture sequence exercising every branch of the ladder at least
    once, including the strafe branch's second (L/R) draw -- repeated so
    TC7/TC8's sequence comparisons are not vacuously trivial (single-item
    sequences would pass even with a badly broken RNG stream)."""
    base = [
        _view(self_health=20.0, can_see_target=True, in_attack_range=False),
        _view(self_health=20.0, can_see_target=True, in_attack_range=True, attack_cooldown=1.0),
        _view(self_health=20.0, can_see_target=True, in_attack_range=True, attack_cooldown=0.2),
        _view(self_health=4.0, can_see_target=True, in_attack_range=False),
        _view(self_health=20.0, can_see_target=False, last_known_target_pos=(1.0, 64.0, 1.0)),
        _view(self_health=20.0, can_see_target=False, last_known_target_pos=None),
        _view(self_health=20.0, can_see_target=True, in_attack_range=False, distance=30.0),
        _view(self_health=6.0, can_see_target=True, in_attack_range=False),
    ]
    return base * 5


# ---------------------------------------------------------------------------
# TC1 -- low health + c_flee=1.0 -> RETREAT
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("health", [6.0, 5.0, 1.0, 0.0])
def test_tc1_low_health_and_full_flee_chance_retreats(health):
    """c_flee=1.0 makes the Bernoulli draw ``random() < 1.0`` unconditionally
    True (random() never returns exactly 1.0), so RETREAT is guaranteed for
    every seed once health drops to/below flee_health (6.0)."""
    bot = ScriptedBot(ScriptedPreset.EASY, c_flee=1.0, seed=0)
    result = bot.act(_view(self_health=health, can_see_target=True, in_attack_range=False))
    assert result is Macro.RETREAT


# ---------------------------------------------------------------------------
# Extra -- flee-vs-attack precedence at the joint where both guards fire
# ---------------------------------------------------------------------------

def test_flee_beats_attack_when_both_guards_are_satisfied():
    """``ScriptedBot`` docs the ladder as "evaluated in exactly this order":
    low-health-flee is checked BEFORE in_attack_range/ATTACK.  Every other
    fixture in this file that sets ``in_attack_range=True`` also leaves
    ``self_health=20.0``, so nothing else in the suite ever puts both
    triggering guards (low health AND in melee range) on the same view --
    swapping the branch order (ATTACK checked before RETREAT) would leave
    every other test green while a HARD bot at 6 HP in melee traded blows
    instead of fleeing.

    HARD's c_flee=1.0 makes the flee draw unconditional
    (``random() < 1.0`` is always True), so this holds for every seed --
    verified across 20 seeds -- but the seed is still pinned per the
    project's no-unseeded-randomness rule."""
    view = _view(self_health=6.0, in_attack_range=True, attack_cooldown=1.0, can_see_target=True)

    hard_bot = ScriptedBot(ScriptedPreset.HARD, c_flee=1.0, seed=0)
    assert hard_bot.act(view) is Macro.RETREAT


def test_attack_fires_on_the_same_fixture_when_c_flee_is_zero():
    """Same joint fixture as above, but EASY's c_flee=0.0 makes the flee
    draw unconditionally False (``random() < 0.0`` is never True).  This
    pins that the flee branch is genuinely gated on ``c_flee`` -- not on
    health alone -- so a HARD-only fix that special-cased health would not
    quietly break EASY's never-flee guarantee."""
    view = _view(self_health=6.0, in_attack_range=True, attack_cooldown=1.0, can_see_target=True)

    easy_bot = ScriptedBot(ScriptedPreset.EASY, c_flee=0.0, seed=0)
    assert easy_bot.act(view) is Macro.ATTACK


# ---------------------------------------------------------------------------
# TC2 -- in range, cooldown charged -> ATTACK
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cooldown", [1.0, 1.0 - 1e-6, 1.5, 2.0])
def test_tc2_in_range_charged_attacks(cooldown):
    """attack_cooldown at/above the 1.0 - 1e-6 ready threshold always
    attacks, regardless of preset or seed."""
    bot = ScriptedBot(ScriptedPreset.EASY, seed=0)
    result = bot.act(_view(in_attack_range=True, attack_cooldown=cooldown, self_health=20.0))
    assert result is Macro.ATTACK


# ---------------------------------------------------------------------------
# TC3 -- in range, cooldown NOT charged -> never ATTACK, never APPROACH
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cooldown", [0.0, 0.5, 0.99, 1.0 - 1e-5])
def test_tc3_in_range_uncharged_never_attacks(cooldown):
    """Below the ready threshold -- including the exact 1.0 - 1e-5 value the
    plan calls out as the mysteriously-passive failure mode if the epsilon
    were ever loosened -- the bot must never flail an uncharged swing, and
    must never APPROACH (closing further while already in range is wrong)."""
    bot = ScriptedBot(ScriptedPreset.HARD, seed=0)
    view = _view(in_attack_range=True, attack_cooldown=cooldown, self_health=20.0)
    results = {bot.act(view) for _ in range(200)}
    assert Macro.ATTACK not in results
    assert Macro.APPROACH not in results
    assert results <= {Macro.STRAFE_L, Macro.STRAFE_R, Macro.JUMP, Macro.IDLE}


def test_tc3_uncharged_in_range_still_moves_sometimes():
    """The uncharged-in-range branch does not just freeze into IDLE forever
    -- it takes the same strafe/jump draw as the approach branch, so over
    many calls at HARD's high strafe/jump rates it must produce more than
    just IDLE."""
    bot = ScriptedBot(ScriptedPreset.HARD, seed=0)
    view = _view(in_attack_range=True, attack_cooldown=0.0, self_health=20.0)
    results = {bot.act(view) for _ in range(200)}
    assert len(results) > 1


# ---------------------------------------------------------------------------
# TC4 -- visible, far, p_strafe=1.0, p_jump=0.0 -> STRAFE_L or STRAFE_R
# ---------------------------------------------------------------------------

def test_tc4_full_strafe_probability_always_strafes():
    """p_jump=0.0 MUST be passed explicitly -- see the companion ValueError
    test below for what happens if it is omitted."""
    bot = ScriptedBot(ScriptedPreset.EASY, p_strafe=1.0, p_jump=0.0, seed=0)
    results = set()
    for _ in range(50):
        result = bot.act(_view(can_see_target=True, in_attack_range=False, self_health=20.0))
        assert result in (Macro.STRAFE_L, Macro.STRAFE_R)
        results.add(result)
    # ``_draw_movement`` documents the L/R pick as "an independent,
    # deliberately even coin flip" (``self._rng.choice((STRAFE_L, STRAFE_R))``).
    # Replacing that choice with a bare ``return Macro.STRAFE_L`` satisfies
    # every assertion above -- membership in {STRAFE_L, STRAFE_R} -- while
    # the bot circles one direction forever, which is obvious to a human on
    # demo day and invisible to a membership-only check.  At the pinned
    # seed=0 this 50-draw sequence realizes 22 STRAFE_L / 28 STRAFE_R, so
    # both members are guaranteed present deterministically, no tolerance
    # needed.
    assert Macro.STRAFE_L in results
    assert Macro.STRAFE_R in results


def test_tc4_p_strafe_alone_inherits_easy_default_jump_and_raises():
    """ScriptedBot(EASY, p_strafe=1.0) alone inherits EASY's default
    p_jump=0.05, summing to 1.05 > 1.0.  This validation is deliberate and
    must not be weakened to make a test pass."""
    with pytest.raises(ValueError):
        ScriptedBot(ScriptedPreset.EASY, p_strafe=1.0)


# ---------------------------------------------------------------------------
# TC5 -- visible, far, p_strafe=0, p_jump=0 -> APPROACH
# ---------------------------------------------------------------------------

def test_tc5_zero_movement_probability_always_approaches():
    bot = ScriptedBot(ScriptedPreset.EASY, p_strafe=0.0, p_jump=0.0, seed=0)
    for _ in range(50):
        result = bot.act(_view(can_see_target=True, in_attack_range=False, self_health=20.0))
        assert result is Macro.APPROACH


# ---------------------------------------------------------------------------
# TC6 -- not visible, last-known set -> TURN_TO_LAST_SEEN
# ---------------------------------------------------------------------------

def test_tc6_not_visible_with_memory_searches():
    """Pinned to TURN_TO_LAST_SEEN alone, not the spec's ``{TURN_TO_LAST_SEEN,
    APPROACH}`` pair -- mutating the search branch to ``return
    Macro.APPROACH`` would satisfy a membership check against that pair
    while leaving the bot unable to ever turn toward memory.

    The spec's "or APPROACH" latitude describes a turn-then-approach
    *sequence* playing out across multiple steps, not a single stateless
    call.  Per ``MACRO_SEMANTICS[Macro.APPROACH]`` in ``agent/actions.py``,
    APPROACH moves along the bot's *current facing* via
    ``bot.setControlState('forward', true)`` -- it is not a pathfinder
    goal aimed at a point.  So "APPROACH toward memory" is not expressible
    in one stateless ``act()`` call: a lone APPROACH on this branch is a
    blind forward walk that never turns to face the last-known position at
    all."""
    bot = ScriptedBot(ScriptedPreset.EASY, seed=0)
    view = _view(can_see_target=False, last_known_target_pos=(3.0, 64.0, 3.0), self_health=20.0)
    result = bot.act(view)
    assert result is Macro.TURN_TO_LAST_SEEN


def test_tc6_not_visible_without_memory_idles():
    """Nothing has ever been seen this episode -- no last-known position to
    search toward, so the bot idles rather than wandering blind."""
    bot = ScriptedBot(ScriptedPreset.EASY, seed=0)
    view = _view(can_see_target=False, last_known_target_pos=None, self_health=20.0)
    assert bot.act(view) is Macro.IDLE


# ---------------------------------------------------------------------------
# TC7 -- same seed, same fixtures, two instances -> identical Macro sequences (AC7)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("preset", [ScriptedPreset.EASY, ScriptedPreset.HARD])
def test_tc7_same_seed_same_fixtures_identical_sequences(preset):
    sequence = _mixed_sequence()
    bot_a = ScriptedBot(preset, seed=12345)
    bot_b = ScriptedBot(preset, seed=12345)
    result_a = [bot_a.act(v) for v in sequence]
    result_b = [bot_b.act(v) for v in sequence]
    assert result_a == result_b
    # Sanity: the mixed sequence isn't degenerate -- more than one macro fires.
    assert len(set(result_a)) > 1


# ---------------------------------------------------------------------------
# TC8 -- reset(seed) re-seeds; reset()/reset(None) does not (a deliberate
# fix per T9's docstring -- nothing currently guards it)
# ---------------------------------------------------------------------------

def test_tc8_reset_with_seed_repeats_original_sequence():
    sequence = _mixed_sequence()
    bot = ScriptedBot(ScriptedPreset.HARD, seed=99)
    first_run = [bot.act(v) for v in sequence]
    bot.reset(99)
    second_run = [bot.act(v) for v in sequence]
    assert first_run == second_run


def test_tc8_reset_with_seed_matches_a_fresh_instance():
    """reset(seed) must be indistinguishable from constructing a brand-new
    instance with the same seed."""
    sequence = _mixed_sequence()
    seeded_bot = ScriptedBot(ScriptedPreset.HARD, seed=99)
    run_before_reset = [seeded_bot.act(v) for v in sequence]
    seeded_bot.reset(99)
    run_after_reset = [seeded_bot.act(v) for v in sequence]

    fresh_bot = ScriptedBot(ScriptedPreset.HARD, seed=99)
    run_from_fresh = [fresh_bot.act(v) for v in sequence]

    assert run_before_reset == run_from_fresh
    assert run_after_reset == run_from_fresh


def _reset_no_arg(bot: ScriptedBot) -> None:
    bot.reset()


def _reset_explicit_none(bot: ScriptedBot) -> None:
    bot.reset(None)


@pytest.mark.parametrize(
    "call_reset", [_reset_no_arg, _reset_explicit_none], ids=["reset_no_arg", "reset_explicit_none"]
)
def test_tc8_reset_without_seed_continues_the_stream(call_reset):
    """reset() / reset(None) must NOT replay from the constructor seed and
    must NOT reseed from OS entropy -- it is a no-op on the RNG (gym
    convention).  A bot that runs the sequence, calls reset() with no
    argument, and runs the sequence again must produce the exact same
    combined output as a fresh, same-seeded bot running the sequence twice
    back-to-back with no reset call in between."""
    sequence = _mixed_sequence()
    seed = 2024

    bot_a = ScriptedBot(ScriptedPreset.HARD, seed=seed)
    first_half = [bot_a.act(v) for v in sequence]
    call_reset(bot_a)
    second_half = [bot_a.act(v) for v in sequence]
    combined_a = first_half + second_half

    bot_b = ScriptedBot(ScriptedPreset.HARD, seed=seed)
    combined_b = [bot_b.act(v) for v in sequence + sequence]

    assert combined_a == combined_b


# ---------------------------------------------------------------------------
# TC9 -- EASY vs HARD, N=10000 samples, seed=0 -> HARD strafes/jumps/flees
# strictly more often.  Fixed N and seed make this a deterministic
# inequality, not a statistical one that could flake.
#
# N was raised from the spec's minimum of 2000 to 10000: at p=0.40 the
# binomial std dev is only ~11 samples-worth at N=2000 (sigma~0.011), so the
# pytest.approx(abs=0.02) band was only ~1.8 sigma wide -- tight enough that
# an unrelated change to how many draws act() spends per call could turn it
# red spuriously.  5x the samples tightens the realized values around the
# nominal preset probabilities instead of loosening the tolerance, while
# still running in well under a second.
# ---------------------------------------------------------------------------

_TC9_N = 10000


def _count_macros(bot: ScriptedBot, view: OpponentView, n: int) -> dict[Macro, int]:
    counts: dict[Macro, int] = {}
    for _ in range(n):
        result = bot.act(view)
        counts[result] = counts.get(result, 0) + 1
    return counts


def test_tc9_hard_strafes_and_jumps_more_than_easy():
    """Movement fixture: full health, visible, out of attack range, so every
    draw goes through the shared strafe/jump ladder and never the flee
    branch."""
    view = _view(self_health=20.0, can_see_target=True, in_attack_range=False)
    easy_counts = _count_macros(ScriptedBot(ScriptedPreset.EASY, seed=0), view, _TC9_N)
    hard_counts = _count_macros(ScriptedBot(ScriptedPreset.HARD, seed=0), view, _TC9_N)

    easy_strafe = easy_counts.get(Macro.STRAFE_L, 0) + easy_counts.get(Macro.STRAFE_R, 0)
    hard_strafe = hard_counts.get(Macro.STRAFE_L, 0) + hard_counts.get(Macro.STRAFE_R, 0)
    easy_jump = easy_counts.get(Macro.JUMP, 0)
    hard_jump = hard_counts.get(Macro.JUMP, 0)

    assert hard_strafe > easy_strafe
    assert hard_jump > easy_jump

    # Sanity band around the preset table's own probabilities (0.15/0.05 vs
    # 0.40/0.20) -- catches a regression to the old sequential-draw ladder
    # (two Bernoulli draws, jump only reached when the strafe draw already
    # failed), which historically measured HARD's realized jump rate at
    # ~0.107-0.118 against its pinned 0.20 (a 41-46% shortfall) -- the
    # strafe marginal is unaffected by draw order (it's drawn first either
    # way), so only the jump band catches this regression.  Measured at
    # this exact N=10000, seed=0: EASY strafe~0.1461 / jump~0.0521, HARD
    # strafe~0.4020 / jump~0.1935.
    assert easy_strafe / _TC9_N == pytest.approx(0.15, abs=0.02)
    assert easy_jump / _TC9_N == pytest.approx(0.05, abs=0.02)
    assert hard_strafe / _TC9_N == pytest.approx(0.40, abs=0.02)
    assert hard_jump / _TC9_N == pytest.approx(0.20, abs=0.02)


def test_tc9_hard_flees_more_than_easy():
    """Flee fixture: health at the flee threshold, visible, out of range.
    EASY's c_flee=0.0 makes flee impossible by construction
    (random() < 0.0 is always False); HARD's c_flee=1.0 makes it
    unconditional (random() < 1.0 is always True).  Deterministic for any
    seed, not just seed=0."""
    view = _view(self_health=6.0, can_see_target=True, in_attack_range=False)
    easy_counts = _count_macros(ScriptedBot(ScriptedPreset.EASY, seed=0), view, _TC9_N)
    hard_counts = _count_macros(ScriptedBot(ScriptedPreset.HARD, seed=0), view, _TC9_N)

    easy_flee = easy_counts.get(Macro.RETREAT, 0)
    hard_flee = hard_counts.get(Macro.RETREAT, 0)

    assert hard_flee > easy_flee
    assert easy_flee == 0
    assert hard_flee == _TC9_N


# ---------------------------------------------------------------------------
# TC10 -- act() output domain: always a valid Macro member
# ---------------------------------------------------------------------------

_TC10_VIEWS = [
    _view(self_health=1.0, can_see_target=True, in_attack_range=False),  # low health (HARD flees)
    _view(in_attack_range=True, attack_cooldown=1.0, self_health=20.0),  # attack ready
    _view(in_attack_range=True, attack_cooldown=0.0, self_health=20.0),  # attack charging
    _view(can_see_target=True, in_attack_range=False, self_health=20.0),  # approach/strafe/jump
    _view(can_see_target=False, last_known_target_pos=(2.0, 64.0, 2.0), self_health=20.0),
    _view(can_see_target=False, last_known_target_pos=None, self_health=20.0),
    _view(self_health=6.0, in_attack_range=False, can_see_target=True),  # flee boundary
]


@pytest.mark.parametrize("preset", [ScriptedPreset.EASY, ScriptedPreset.HARD])
def test_tc10_act_output_is_always_a_macro(preset):
    bot = ScriptedBot(preset, seed=1)
    for view in _TC10_VIEWS:
        for _ in range(20):  # repeat to sample every branch of the RNG-gated ladder
            result = bot.act(view)
            assert isinstance(result, Macro)
            assert result in Macro


# ---------------------------------------------------------------------------
# Extra -- construction validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "kwargs",
    [
        {"p_strafe": -0.01},
        {"p_strafe": 1.01},
        {"p_jump": -0.01},
        {"p_jump": 1.01},
        {"c_flee": -0.01},
        {"c_flee": 1.01},
        {"p_strafe": math.nan},
        {"p_jump": math.nan},
        {"c_flee": math.nan},
        {"p_strafe": 0.6, "p_jump": 0.5},  # sums to 1.1 > 1.0
    ],
    ids=[
        "p_strafe_below_zero",
        "p_strafe_above_one",
        "p_jump_below_zero",
        "p_jump_above_one",
        "c_flee_below_zero",
        "c_flee_above_one",
        "p_strafe_nan",
        "p_jump_nan",
        "c_flee_nan",
        "p_strafe_plus_p_jump_over_one",
    ],
)
def test_construction_validation_raises_value_error(kwargs):
    with pytest.raises(ValueError):
        ScriptedBot(ScriptedPreset.EASY, **kwargs)


# ---------------------------------------------------------------------------
# Extra -- config() contract
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("preset", [ScriptedPreset.EASY, ScriptedPreset.HARD])
def test_config_returns_same_object_on_every_call(preset):
    bot = ScriptedBot(preset)
    assert bot.config is bot.config


@pytest.mark.parametrize("preset", [ScriptedPreset.EASY, ScriptedPreset.HARD])
def test_config_knockback_immune_is_false(preset):
    """Unlike StationaryDummy (all four flags True), a scripted opponent
    that cannot be knocked back makes the fight unreal."""
    cfg = ScriptedBot(preset).config
    assert cfg.knockback_immune is False
    assert cfg.fall_immune is True
    assert cfg.void_immune is True
    assert cfg.fixed_spawn is True


def test_config_is_opponent_config_instance():
    assert isinstance(ScriptedBot().config, OpponentConfig)


# ---------------------------------------------------------------------------
# Extra -- OpponentView frozen-ness
# ---------------------------------------------------------------------------

def test_opponent_view_is_frozen():
    """OpponentView is declared @dataclass(frozen=True) and the spec's
    Contracts section pins it as frozen -- mirrors
    tests/test_opponents.py::test_dummy_config_is_frozen's idiom for the
    analogous OpponentConfig check.  Dropping frozen=True currently passes
    every other test in this file, since nothing else attempts a mutation."""
    view = _view()
    with pytest.raises((AttributeError, TypeError)):
        view.self_health = 10.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Extra -- name and Opponent interface
# ---------------------------------------------------------------------------

def test_name_easy():
    assert ScriptedBot(ScriptedPreset.EASY).name == "scripted_easy"


def test_name_hard():
    assert ScriptedBot(ScriptedPreset.HARD).name == "scripted_hard"


def test_default_preset_is_easy():
    assert ScriptedBot().name == "scripted_easy"


def test_scripted_bot_is_an_opponent():
    assert isinstance(ScriptedBot(), Opponent)


# ---------------------------------------------------------------------------
# Extra -- flee_health boundary (inclusive <=)
# ---------------------------------------------------------------------------

def test_hard_flees_at_exactly_flee_health():
    bot = ScriptedBot(ScriptedPreset.HARD, seed=0)
    result = bot.act(_view(self_health=6.0, can_see_target=True, in_attack_range=False))
    assert result is Macro.RETREAT


def test_hard_does_not_flee_just_above_flee_health():
    """6.01 is just above the 6.0 threshold, so low_health must be False."""
    bot = ScriptedBot(ScriptedPreset.HARD, seed=0)
    result = bot.act(_view(self_health=6.01, can_see_target=True, in_attack_range=False))
    assert result is not Macro.RETREAT


def test_easy_never_flees_even_at_one_hp():
    """EASY's c_flee=0.0 makes RETREAT unreachable regardless of health."""
    bot = ScriptedBot(ScriptedPreset.EASY, seed=0)
    for _ in range(50):
        result = bot.act(_view(self_health=1.0, can_see_target=True, in_attack_range=False))
        assert result is not Macro.RETREAT
