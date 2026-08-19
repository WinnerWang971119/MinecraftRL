"""Unit tests for the pure analysis half of eval/combat_probe.py (T8, T23).

The probe is the AC8 go/no-go gate, so the thing that must be provably correct
OFFLINE is its judgement: a probe that passes wrongly is worse than one that
fails. These tests drive ``analyze_cycle`` / ``reconcile_against_wire`` /
``check_anchor`` with scripted window records — the healthy kill in several
legal timing variants, plus every failure class the probe exists to catch
(wrong sequence, phantom hit, unrecorded hit, regeneration, double death, dirty
baseline, wrong pad). No socket, no server.

T23 ADDS THE HALF THAT USED TO BE A LITERAL. The expected sequence is now
DERIVED from the target's loadout, so the derivation itself is under test here:
the jar's absorption formula, the armor table, the identity that
``--target-armor none`` reproduces the historical ``6, 6, 6, 2``, the fact that
a bare-handed run must FAIL against the armored expectation (that is the whole
point of the change), and both ends of the float tolerance — a float32 replay of
what the server actually puts on the wire on one side, a one-armor-point defect
on the other.

BOTH ENDS OF THE CLAMP, AND THE TERM NO TIER USES. ``getDamageAfterAbsorb``
clamps between ``MIN_ARMOR_RATIO`` (``armor * 0.2``, the LOWER bound) and
``MAX_ARMOR`` (``20.0``, the UPPER one), and only the upper bound delivers the
">= 20% of every swing lands" guarantee ``expected_hit_sequence`` leans on — so
each bound gets its own test rather than one test named after the other one's
number. The ``2.0 + toughness / 4.0`` term is pinned directly for the same
reason: every tier in ``ARMOR_SETS`` carries toughness 0.0, so nothing else in
this suite would notice if that line changed, and the module invites the next
contributor to add a tier that does depend on it.
"""

from __future__ import annotations

import math
import struct
from typing import Dict, List, Sequence

import pytest

from eval.combat_probe import (
    ARMOR_SETS,
    CHAINMAIL_ARMOR,
    EXPECTED_HITS,
    EXPECTED_TOTAL,
    FULL_HEALTH,
    IRON_ARMOR,
    IRON_SWORD_ATTACK_SPEED_TICKS,
    IRON_SWORD_DAMAGE,
    LEATHER_ARMOR,
    NO_ARMOR,
    TARGET_ARMOR,
    WINDOWS_PER_SWING,
    ArmorSet,
    CycleRecord,
    StepRecord,
    analyze_cycle,
    check_anchor,
    damage_after_absorb,
    damage_for_swing_charge,
    expected_hit_sequence,
    extract_hits,
    reconcile_against_wire,
    run_probe,
)
from eval.combat_probe import _DAMAGE_TOL, _TOL

#: The bare-handed sequence this probe shipped with, kept as a literal so the
#: "did the derivation change the old physics?" test cannot drift with the code.
HISTORICAL_BARE_HITS = (6.0, 6.0, 6.0, 2.0)


def _step(
    damage: float = 0.0,
    health: float = 20.0,
    died: bool = False,
    action: int = 0,
) -> StepRecord:
    return StepRecord(
        action=action,
        damage_dealt=damage,
        opponent_died=died,
        wire_health=health,
        attack_cooldown=1.0,
        tick=0,
    )


def _record(
    steps: Sequence[StepRecord],
    outcome: str = "win",
    start_health: float = 20.0,
    anchor: tuple = (0, 0),
) -> CycleRecord:
    ax, az = anchor
    return CycleRecord(
        index=0,
        reset_ms=100.0,
        start_health=start_health,
        start_self_pos=(ax + 0.5, 64.0, az + 0.5),
        start_opp_pos=(ax + 3.5, 64.0, az + 0.5),
        outcome=outcome,
        steps=list(steps),
    )


def _hit_windows(hits: Sequence[float]) -> Dict[int, float]:
    """Windows the live probe lands its hits in: the first at 1, then every 4.

    The spacing mirrors what the live driver observes (a 12.5-tick swing
    cooldown over 4-tick decision windows), but nothing in the analysis under
    test depends on it — it only has to be consistent within a fixture.
    """
    return {1 + i * WINDOWS_PER_SWING: float(a) for i, a in enumerate(hits)}


def _healthy_steps(
    *,
    skew: int = 0,
    respawn_masks_final_drop: bool = True,
    hits: Sequence[float] = EXPECTED_HITS,
    start_health: float = FULL_HEALTH,
) -> List[StepRecord]:
    """The canonical clean kill for ``hits``, as the wire would report it.

    ``skew=0`` puts each wire-health drop in the same window as its damage
    event; ``skew=+1`` delays every drop one window (the second-connection
    ``update_health`` skew AC8 explicitly permits), ``skew=-1`` advances it.
    With ``respawn_masks_final_drop`` the killing blow's drop to 0 is invisible
    because ``doImmediateRespawn`` snaps the window-end snapshot back to full.

    Defaults to :data:`EXPECTED_HITS` — the ARMORED sequence (3.12 x6 then
    1.28), not the bare-handed 6,6,6,2 this fixture used before T23.
    """
    hit_windows = _hit_windows(hits)
    death_window = max(hit_windows)

    # Wire health at each window END, without skew. Health is clamped at 0 on
    # the fatal blow (LivingEntity.setHealth), so the last entry is exact.
    health_after: Dict[int, float] = {}
    health = float(start_health)
    ordered = sorted(hit_windows)
    for i, w in enumerate(ordered):
        health -= hit_windows[w]
        health_after[w] = 0.0 if i == len(ordered) - 1 else health
    if respawn_masks_final_drop:
        health_after[death_window] = float(start_health)

    steps: List[StepRecord] = []
    health = float(start_health)
    for w in range(death_window + 1):
        drop_window = w - skew
        if drop_window in health_after:
            health = health_after[drop_window]
        steps.append(
            _step(
                damage=hit_windows.get(w, 0.0),
                health=health,
                died=(w == death_window),
                action=5 if w in hit_windows else 0,
            )
        )
    return steps


def _last_hit_window(hits: Sequence[float] = EXPECTED_HITS) -> int:
    return max(_hit_windows(hits))


# ---------------------------------------------------------------------------
# The derivation itself (T23). Every number here is checked against the pinned
# jar in the module's own comments; these tests pin the ARITHMETIC.
# ---------------------------------------------------------------------------


def test_armor_table_matches_the_pinned_jar() -> None:
    # ArmorMaterials, boots + leggings + chestplate + helmet, toughness 0 for
    # every tier below diamond.
    assert (NO_ARMOR.points, NO_ARMOR.toughness) == (0, 0.0)
    assert (LEATHER_ARMOR.points, LEATHER_ARMOR.toughness) == (1 + 2 + 3 + 1, 0.0)
    assert (CHAINMAIL_ARMOR.points, CHAINMAIL_ARMOR.toughness) == (1 + 4 + 5 + 2, 0.0)
    assert (IRON_ARMOR.points, IRON_ARMOR.toughness) == (2 + 5 + 6 + 2, 0.0)
    assert IRON_ARMOR.points == 15
    # The datapack equips iron today (spawn_dummy_pad.mcfunction's re-gear).
    assert TARGET_ARMOR is IRON_ARMOR
    assert set(ARMOR_SETS) == {"none", "leather", "chainmail", "iron"}
    assert all(name == s.name for name, s in ARMOR_SETS.items())


def test_damage_after_absorb_is_the_combat_rules_formula() -> None:
    # damage * (1 - clamp(armor - damage/(2 + toughness/4), armor*0.2, 20)/25)
    for armor, expected in (
        (NO_ARMOR, 6.0),
        (LEATHER_ARMOR, 5.04),  # g = clamp(7-3, 1.4, 20) = 4  -> 6*(1-0.16)
        (CHAINMAIL_ARMOR, 3.84),  # g = clamp(12-3, 2.4, 20) = 9  -> 6*(1-0.36)
        (IRON_ARMOR, 3.12),  # g = clamp(15-3, 3.0, 20) = 12 -> 6*(1-0.48)
    ):
        landed = damage_after_absorb(
            IRON_SWORD_DAMAGE, armor.points, armor.toughness
        )
        assert landed == pytest.approx(expected, abs=1e-12), armor.name


def test_absorption_is_not_a_flat_percent_per_point() -> None:
    # The `armor - damage/f` term makes the absorbed FRACTION depend on the
    # incoming damage. A flat-percentage model (15 points -> 60%) would predict
    # 2.4 through, which is the mistake this file exists to stop.
    assert damage_after_absorb(6.0, 15, 0.0) == pytest.approx(3.12)
    assert damage_after_absorb(6.0, 15, 0.0) != pytest.approx(6.0 * 0.4)
    weak = damage_after_absorb(1.0, 15, 0.0)  # a bare fist
    assert weak == pytest.approx(0.42, abs=1e-12)
    # The weak hit loses 58% while the sword hit loses only 48%.
    assert (1.0 - weak / 1.0) > (1.0 - 3.12 / 6.0)


def test_min_armor_ratio_clamp_bounds_absorption_from_below() -> None:
    # `armor * 0.2` is MIN_ARMOR_RATIO, the clamp's LOWER bound: it stops `g`
    # from being driven below `armor * 0.2` by a big hit, so the absorbed
    # FRACTION can never fall below `armor * 0.2 / 25 == armor / 125`.
    #
    # A 1000-damage hit sends `armor - damage/f` far negative, so the floor
    # binds and absorption sits at its MINIMUM -- 10 points absorb exactly 8%,
    # which is the floor's behaviour and not the 80% ceiling (that is
    # MAX_ARMOR's job; see the test below).
    landed = damage_after_absorb(1000.0, 10, 0.0)
    assert landed == pytest.approx(1000.0 * (1.0 - (10 * 0.2) / 25.0))
    assert landed == pytest.approx(1000.0 * (1.0 - 10 / 125.0))
    assert landed == pytest.approx(920.0)  # 8% absorbed, not 80%


def test_max_armor_clamp_caps_absorption_at_eighty_percent() -> None:
    # `20.0` is MAX_ARMOR, the clamp's UPPER bound, and it -- not
    # MIN_ARMOR_RATIO -- is what guarantees ">= 20% of every swing lands":
    # g <= 20 implies g/25 <= 0.8.
    #
    # A weak hit against heavy armor saturates it: armor - damage/f
    # = 30 - 1.0/2.0 = 29.5, well past 20, so g == 20 and exactly 80% is
    # absorbed. Raising the 20.0 ceiling breaks this assertion, which is the
    # point: before this test, mutating it to 25.0 survived the whole suite.
    assert 30 - 1.0 / 2.0 > 20.0  # the ceiling really is the binding bound
    assert damage_after_absorb(1.0, 30, 0.0) == pytest.approx(0.2, abs=1e-12)

    # The same guarantee as the property `expected_hit_sequence` relies on to
    # know its `while remaining` loop terminates for ANY loadout.
    for armor_points in (0, 7, 15, 30, 1000):
        for damage in (0.5, 1.0, 6.0, 100.0):
            landed = damage_after_absorb(damage, armor_points, 0.0)
            assert landed >= 0.2 * damage - 1e-12, (armor_points, damage)


def test_toughness_term_scales_the_absorption_denominator() -> None:
    # Every tier in ARMOR_SETS carries toughness 0.0, which makes
    # `f = 2.0 + toughness / 4.0` dead arithmetic for the whole suite: deleting
    # the term or changing the divisor used to leave all tests green. The
    # module advertises "adding a tier ... add a row here" and notes the absent
    # tiers carry non-zero toughness, so pin the term directly.
    #
    # Hand-derivation for 20 armor points and 2.0 TOTAL toughness:
    #   f      = 2.0 + 2.0 / 4.0                     == 2.5
    #   g      = clamp(20 - 6.0 / 2.5, 20 * 0.2, 20.0)
    #          = clamp(17.6, 4.0, 20.0)              == 17.6
    #   landed = 6.0 * (1.0 - 17.6 / 25.0) = 6.0 * 0.296
    #                                                == 1.776
    assert damage_after_absorb(6.0, 20, 2.0) == pytest.approx(1.776, abs=1e-12)

    # Direction, so an inverted term cannot pass: toughness shrinks the
    # `damage / f` erosion of `g`, so the SAME points absorb MORE with it.
    #   toughness 0: g = clamp(20 - 3.0, 4.0, 20.0) == 17.0 -> 6 * 0.32 == 1.92
    assert damage_after_absorb(6.0, 20, 0.0) == pytest.approx(1.92, abs=1e-12)
    assert damage_after_absorb(6.0, 20, 2.0) < damage_after_absorb(6.0, 20, 0.0)

    # DELIBERATELY SYNTHETIC, NOT "diamond". ArmorSet.toughness is the TOTAL of
    # the four worn slots, while the jar's ArmorMaterials register(...)
    # toughness argument is PER PIECE (ArmorItem adds one ADD_VALUE
    # ARMOR_TOUGHNESS modifier per item), so a real tier's total is 4x its
    # material value. Nobody here has read diamond's row off the pinned jar,
    # so no row is added to ARMOR_SETS on the strength of this test.
    assert "diamond" not in ARMOR_SETS


def test_expected_hits_is_the_iron_arithmetic_seven_hits() -> None:
    assert len(EXPECTED_HITS) == 7
    assert EXPECTED_HITS[:6] == pytest.approx([3.12] * 6, abs=1e-12)
    assert EXPECTED_HITS[-1] == pytest.approx(1.28, abs=1e-12)
    assert math.fsum(EXPECTED_HITS) == pytest.approx(FULL_HEALTH, abs=1e-12)
    assert EXPECTED_TOTAL == FULL_HEALTH == 20.0


def test_bare_handed_derivation_reproduces_the_historical_constant() -> None:
    # THE REGRESSION ANCHOR. If the derivation is right, feeding it the
    # unarmored target it was written against must give back the literal the
    # probe shipped with, exactly. This is also what makes `--target-armor none`
    # a valid revert / A/B path with no code change.
    # `==`, not approx: with 0 armor the clamp yields g == 0 and every step of
    # the derivation is exact in binary, so "exactly" means exactly.
    assert expected_hit_sequence(armor=NO_ARMOR) == HISTORICAL_BARE_HITS
    assert len(expected_hit_sequence(armor=NO_ARMOR)) == 4


def test_expected_hit_sequence_tracks_the_tier() -> None:
    for armor, count in (
        (NO_ARMOR, 4),
        (LEATHER_ARMOR, 4),
        (CHAINMAIL_ARMOR, 6),
        (IRON_ARMOR, 7),
    ):
        seq = expected_hit_sequence(armor=armor)
        assert len(seq) == count, armor.name
        assert math.fsum(seq) == pytest.approx(FULL_HEALTH, abs=1e-9), armor.name
        # Every blow but the last is the same full-strength hit...
        assert all(v == pytest.approx(seq[0], abs=1e-12) for v in seq[:-1])
        # ...and the last is short, because health clamps at 0 (or exactly
        # equal, if the tier happens to divide 20 evenly).
        assert seq[-1] <= seq[0] + _DAMAGE_TOL


def test_trailing_hit_is_the_remaining_health_clamp() -> None:
    # The `2` in the old 6,6,6,2 and the `1.28` in the new sequence are the same
    # thing: the fatal blow reports the health that was left, because
    # damage_dealt is a health DROP and setHealth clamps at 0.
    per_hit = damage_after_absorb(IRON_SWORD_DAMAGE, IRON_ARMOR.points, 0.0)
    assert EXPECTED_HITS[-1] == pytest.approx(
        FULL_HEALTH - 6 * per_hit, abs=1e-12
    )
    assert EXPECTED_HITS[-1] < per_hit


def test_damage_for_swing_charge_prices_a_partial_swing() -> None:
    # Player.attack scales RAW damage by 0.2 + f*f*0.8 BEFORE absorption, so a
    # full-charge swing is exactly the per-hit expectation...
    assert damage_for_swing_charge(1.0) == pytest.approx(EXPECTED_HITS[0], abs=1e-12)
    # ...and these two are REACHABLE charges the module docstring quotes:
    # f = 0.92 is ticker 11 (one tick short) and f = 0.44 is ticker 5. Pinned
    # here so the prose cannot drift back to a continuous band -- neither the
    # old 0.999 nor the old 0.5 is a value `(ticker + 0.5) / 12.5` can take.
    assert round(damage_for_swing_charge(0.92), 4) == 2.659
    assert round(damage_for_swing_charge(0.44), 4) == 0.9424
    # Monotone in the charge.
    values = [damage_for_swing_charge(f / 10.0) for f in range(11)]
    assert values == sorted(values)


def test_the_reachable_swing_ladder_is_quantized() -> None:
    # THE TPS-FLOOR DIAGNOSTIC, PINNED. `LivingEntity.attackStrengthTicker` is
    # an int and `Player.getCurrentItemAttackStrengthDelay()` is 20/1.6 == 12.5,
    # so `getAttackStrengthScale(0.5f)` can only return
    # Mth.clamp((ticker + 0.5) / 12.5, 0, 1) -- steps of 0.08. The docstring's
    # ladder is pinned here so the prose cannot drift back to a continuous band
    # that would send an operator hunting the bridge's meter on demo day.
    def scale(ticker: int) -> float:
        return min((ticker + 0.5) / IRON_SWORD_ATTACK_SPEED_TICKS, 1.0)

    assert scale(12) == 1.0
    assert scale(13) == 1.0  # Mth.clamp(..., 0.0f, 1.0f)
    assert scale(11) == pytest.approx(0.92, abs=1e-12)
    assert scale(10) == pytest.approx(0.84, abs=1e-12)
    assert scale(0) == pytest.approx(0.04, abs=1e-12)
    assert scale(11) - scale(10) == pytest.approx(0.08, abs=1e-12)

    ladder = {t: round(damage_for_swing_charge(scale(t)), 4) for t in (12, 11, 10, 0)}
    assert ladder == {12: 3.12, 11: 2.659, 10: 2.2555, 0: 0.5122}

    # The shortfall jumps 0 -> ~0.46 HP (~15% of the expected hit). "A few
    # thousandths under 3.12" is not a reading the server can produce.
    assert ladder[12] == pytest.approx(EXPECTED_HITS[0], abs=1e-12)
    assert EXPECTED_HITS[0] - ladder[11] == pytest.approx(0.461, abs=5e-4)
    assert ladder[11] / EXPECTED_HITS[0] == pytest.approx(0.852, abs=5e-4)
    assert ladder[10] / EXPECTED_HITS[0] == pytest.approx(0.723, abs=5e-4)

    # ...and "at or below ~0.6" really does mean the meter never ramped: the
    # threshold is crossed between ticker 2 and 3, and ticker 0 (a meter at
    # dead zero) is the floor of the entire ladder.
    assert damage_for_swing_charge(scale(2)) < 0.6
    assert damage_for_swing_charge(scale(3)) > 0.6
    assert ladder[0] == min(
        round(damage_for_swing_charge(scale(t)), 4) for t in range(13)
    )


def test_damage_for_swing_charge_rejects_an_impossible_charge() -> None:
    for bad in (-0.1, 1.1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            damage_for_swing_charge(bad)


def test_derivation_rejects_impossible_inputs() -> None:
    with pytest.raises(ValueError):
        expected_hit_sequence(weapon_damage=0.0)
    with pytest.raises(ValueError):
        expected_hit_sequence(weapon_damage=float("nan"))
    with pytest.raises(ValueError):
        expected_hit_sequence(target_health=0.0)
    with pytest.raises(ValueError):
        damage_after_absorb(-1.0, 15, 0.0)
    with pytest.raises(ValueError):
        ArmorSet("bogus", -1, 0.0)
    with pytest.raises(ValueError):
        ArmorSet("bogus", 15, -1.0)


def test_attack_speed_constants_match_the_bridge() -> None:
    # bridge/actions.js: IRON_SWORD_ATTACK_SPEED_TICKS = 20 / 1.6 == 12.5, and
    # bot.js's attack-meter note: ceil(12.5 / ACTION_REPEAT) == 4 windows.
    assert IRON_SWORD_ATTACK_SPEED_TICKS == 12.5
    assert WINDOWS_PER_SWING == 4


# ---------------------------------------------------------------------------
# The tolerance, from both ends.
# ---------------------------------------------------------------------------


def _f32(x: float) -> float:
    return struct.unpack("<f", struct.pack("<f", x))[0]


def _f32_damage_after_absorb(damage: float, armor: float, toughness: float) -> float:
    """CombatRules.getDamageAfterAbsorb in the server's own float32 precision."""
    d, a, t = _f32(damage), _f32(armor), _f32(toughness)
    f = _f32(2.0 + _f32(t / 4.0))
    g = _f32(a - _f32(d / f))
    g = _f32(min(max(g, _f32(a * _f32(0.2))), _f32(20.0)))
    return _f32(d * _f32(1.0 - _f32(g / _f32(25.0))))


def _f32_wire_cascade(armor: ArmorSet) -> List[float]:
    """The per-hit health DROPS the server would actually put on the wire.

    Health is a float32 server-side and mineflayer reads that float32 verbatim
    off ``update_health``, so this is the exact value sequence the probe sees —
    including the rounding the float64 expectation does not have.
    """
    per_hit = _f32_damage_after_absorb(
        IRON_SWORD_DAMAGE, float(armor.points), armor.toughness
    )
    assert per_hit > 0.0
    health = _f32(FULL_HEALTH)
    drops: List[float] = []
    while health > 0.0:
        nxt = _f32(health - per_hit)
        if nxt < 0.0:
            nxt = 0.0
        drops.append(health - nxt)  # doubles, as the bridge subtracts them
        health = nxt
    return drops


@pytest.mark.parametrize(
    "armor", [NO_ARMOR, LEATHER_ARMOR, CHAINMAIL_ARMOR, IRON_ARMOR]
)
def test_float32_wire_values_land_inside_the_tolerance(armor: ArmorSet) -> None:
    # LOWER BOUND of _DAMAGE_TOL: the server's float32 arithmetic must not be
    # able to red-fail a healthy run.
    expected = expected_hit_sequence(armor=armor)
    drops = _f32_wire_cascade(armor)
    assert len(drops) == len(expected), armor.name
    divergence = max(abs(d - e) for d, e in zip(drops, expected))
    assert divergence < _DAMAGE_TOL / 10.0, (armor.name, divergence)


def test_float32_iron_run_passes_analyze_cycle_end_to_end() -> None:
    # The same cascade, driven through the real judgement: a healthy armored
    # live run must come back green.
    drops = _f32_wire_cascade(IRON_ARMOR)
    steps = _healthy_steps(hits=drops)
    assert analyze_cycle(_record(steps)) == []


def test_one_armor_point_is_far_outside_the_tolerance() -> None:
    # UPPER BOUND of _DAMAGE_TOL: the quietest real defect that can exist (one
    # iron piece silently swapped for leather, 15 -> 14 points) must still be
    # caught with enormous margin.
    full = damage_after_absorb(IRON_SWORD_DAMAGE, 15, 0.0)
    one_short = damage_after_absorb(IRON_SWORD_DAMAGE, 14, 0.0)
    assert abs(one_short - full) == pytest.approx(0.24, abs=1e-12)
    assert abs(one_short - full) > 1000 * _DAMAGE_TOL
    # And the tolerance is still tighter than any wire-vs-wire comparison needs
    # to be loosened by.
    assert _TOL < _DAMAGE_TOL < 1e-3


def test_a_single_missing_armor_piece_red_fails_the_probe() -> None:
    # Concretely: the helmet's `$item replace` silently no-ops (15 -> 13).
    missing_helmet = ArmorSet("iron_minus_helmet", 13, 0.0)
    steps = _healthy_steps(hits=expected_hit_sequence(armor=missing_helmet))
    errors = analyze_cycle(_record(steps))
    assert any("per-hit sequence" in e for e in errors), errors


# ---------------------------------------------------------------------------
# Healthy cycles must pass, in every legal timing variant.
# ---------------------------------------------------------------------------


def test_healthy_cycle_passes() -> None:
    assert analyze_cycle(_record(_healthy_steps())) == []


def test_healthy_cycle_with_one_window_skew_passes() -> None:
    # update_health arrives on a second connection: drops one window late.
    assert analyze_cycle(_record(_healthy_steps(skew=1))) == []


def test_healthy_cycle_with_visible_final_drop_passes() -> None:
    # The snapshot catches the fatal drop to 0 before the respawn.
    steps = _healthy_steps(respawn_masks_final_drop=False)
    assert analyze_cycle(_record(steps)) == []


def test_extract_hits_orders_and_filters() -> None:
    steps = _healthy_steps()
    assert extract_hits(steps) == [
        (1 + i * WINDOWS_PER_SWING, amount)
        for i, amount in enumerate(EXPECTED_HITS)
    ]


# ---------------------------------------------------------------------------
# The loadout has to be part of the judgement, not an assumption.
# ---------------------------------------------------------------------------


def test_bare_handed_run_fails_against_the_armored_expectation() -> None:
    # THE T23 SCENARIO, INVERTED. If the four `$item replace` lines silently do
    # nothing, the dummy fights naked and the wire reports 6,6,6,2. That is a
    # real defect now and the probe must say so.
    steps = _healthy_steps(hits=HISTORICAL_BARE_HITS)
    errors = analyze_cycle(_record(steps))
    assert any("per-hit sequence" in e for e in errors), errors
    # ...and the cumulative check must NOT fire: 20 HP is still 20 HP. Only the
    # split moved, which is exactly why the per-hit assertion carries the signal.
    assert not any("cumulative" in e for e in errors), errors


def test_bare_handed_run_passes_when_told_the_target_is_unarmored() -> None:
    # The revert / A/B path: `--target-armor none`.
    steps = _healthy_steps(hits=HISTORICAL_BARE_HITS)
    assert analyze_cycle(_record(steps), expected_hit_sequence(armor=NO_ARMOR)) == []


def test_armored_run_fails_against_the_bare_handed_expectation() -> None:
    # The pre-T23 probe against today's fleet: this is the false alarm the
    # change exists to remove, reproduced so it stays reproduced.
    steps = _healthy_steps()
    errors = analyze_cycle(_record(steps), HISTORICAL_BARE_HITS)
    assert any("per-hit sequence" in e for e in errors), errors


def test_wrong_tier_is_rejected() -> None:
    steps = _healthy_steps(hits=expected_hit_sequence(armor=CHAINMAIL_ARMOR))
    errors = analyze_cycle(_record(steps))
    assert any("per-hit sequence" in e for e in errors), errors


# ---------------------------------------------------------------------------
# Every failure class must be caught.
# ---------------------------------------------------------------------------


def _assert_fails(record: CycleRecord, fragment: str) -> List[str]:
    errors = analyze_cycle(record)
    assert errors, "expected the cycle to fail but it passed"
    assert any(fragment in e for e in errors), (
        f"expected a failure mentioning {fragment!r}, got {errors}"
    )
    return errors


def test_wrong_hit_sequence_fails() -> None:
    # Cooldown-violating early hit in place of the second full swing, at a
    # charge the meter can actually be at: ticker 5, f = 0.44.
    steps = _healthy_steps()
    weak = damage_for_swing_charge(0.44)
    hit_window = 1 + WINDOWS_PER_SWING
    steps[hit_window].damage_dealt = weak
    for w in range(hit_window, len(steps)):
        steps[w].wire_health += EXPECTED_HITS[1] - weak
    _assert_fails(_record(steps), "per-hit sequence")


def test_phantom_hit_without_wire_drop_fails() -> None:
    # A recorded damage event the wire never saw (over-counting).
    steps = _healthy_steps()
    steps[3].damage_dealt = EXPECTED_HITS[0]
    _assert_fails(_record(steps), "no matching wire-health drop")


def test_unrecorded_wire_drop_fails() -> None:
    # The wire lost health but no damage event was recorded (under-counting —
    # the original bug class).
    steps = _healthy_steps()
    steps[1].damage_dealt = 0.0
    errors = analyze_cycle(_record(steps))
    assert any("unrecorded hit" in e for e in errors)


def test_regeneration_heal_fails() -> None:
    # Health creeps back up mid-episode: regeneration is supposed to be off.
    steps = _healthy_steps()
    for w in (3, 4):
        steps[w].wire_health += 1.0  # far from the death window
    errors = reconcile_against_wire(FULL_HEALTH, steps)
    assert any("INCREASED" in e for e in errors)


def test_cumulative_over_20_is_tc16_defect() -> None:
    steps = _healthy_steps()
    steps[_last_hit_window()].damage_dealt = EXPECTED_HITS[0] + 5.0
    _assert_fails(_record(steps), "TC16")


def test_double_death_fails() -> None:
    steps = _healthy_steps()
    steps[_last_hit_window() - 1].opponent_died = True
    _assert_fails(_record(steps), "opponent_died fired in 2")


def test_zero_deaths_fails() -> None:
    steps = _healthy_steps()
    steps[_last_hit_window()].opponent_died = False
    _assert_fails(_record(steps), "opponent_died fired in 0")


def test_dirty_start_baseline_fails() -> None:
    record = _record(_healthy_steps(), start_health=14.0)
    _assert_fails(record, "clean")


def test_non_win_outcome_fails() -> None:
    _assert_fails(_record(_healthy_steps(), outcome="timeout"), "expected a win")


def test_masked_killing_blow_with_wrong_entering_health_fails() -> None:
    # W1: the killing-blow carve-out (death nearby + drop masked by the
    # immediate respawn) must ONLY accept a hit whose amount equals the health
    # ENTERING its window. Here the channel recorded something else — a masked
    # killing blow that does NOT reconcile.
    steps = _healthy_steps(respawn_masks_final_drop=True)
    death = _last_hit_window()
    steps[death].damage_dealt = EXPECTED_HITS[-1] + 1.0
    errors = reconcile_against_wire(FULL_HEALTH, steps)
    assert any("no matching wire-health drop" in e for e in errors), errors


def test_cumulative_total_off_by_under_two_fails() -> None:
    # W2: a small negative damage_dealt slipping through leaves the positive
    # hit sequence intact (extract_hits filters it) and only the exact
    # cumulative check can catch it. Off by 0.5 must still FAIL.
    steps = _healthy_steps()
    steps[2].damage_dealt = -0.5  # window 2 carries no hit
    errors = analyze_cycle(_record(steps))
    assert any("cumulative" in e for e in errors), errors
    assert not any("per-hit sequence" in e for e in errors), errors


def test_cumulative_check_is_independent_of_the_derived_sequence() -> None:
    # S1: the total is held against EXPECTED_TOTAL (the target's full health),
    # NOT against fsum(expected_hits) -- otherwise a wrong derivation moves
    # both sides of the comparison together and the check can never fire.
    # Here the live run is healthy (it really did deal exactly 20) while the
    # caller supplied a sequence inflated 20%: only the per-hit check may fire.
    inflated = tuple(v * 1.2 for v in EXPECTED_HITS)
    assert math.fsum(inflated) > EXPECTED_TOTAL + 1.0  # the derivation is wrong
    errors = analyze_cycle(_record(_healthy_steps()), inflated)
    assert any("per-hit sequence" in e for e in errors), errors
    assert not any("cumulative" in e for e in errors), errors
    assert not any("TC16" in e for e in errors), errors


def test_cumulative_check_uses_the_tight_wire_against_wire_tolerance() -> None:
    # S1: every damage_dealt is a health DROP, so the total telescopes to
    # h_0 - h_n == 20 exactly for EVERY loadout -- both sides of this
    # comparison are wire values, so it is held to _TOL and not to the 100x
    # looser _DAMAGE_TOL the per-hit check needs for its float32-vs-float64
    # crossing. A leak between the two tolerances must still be caught.
    drift = 1e-5
    assert _TOL < drift < _DAMAGE_TOL
    steps = _healthy_steps()
    steps[2].damage_dealt = -drift  # window 2 carries no hit, and -x is no hit
    errors = analyze_cycle(_record(steps))
    assert any("cumulative" in e for e in errors), errors
    assert not any("per-hit sequence" in e for e in errors), errors


def test_healthy_cycle_with_early_drop_passes() -> None:
    # W3: skew = -1 — the wire-health drop lands one window BEFORE the recorded
    # damage event. Still within the +/-1 contract, so it must reconcile.
    steps = _healthy_steps(skew=-1, respawn_masks_final_drop=False)
    assert analyze_cycle(_record(steps)) == []


def test_two_window_skew_is_rejected() -> None:
    # +/-1 window is the contract; a 2-window skew must NOT reconcile.
    errors = analyze_cycle(_record(_healthy_steps(skew=2)))
    assert any("no matching wire-health drop" in e for e in errors)


def test_drop_cannot_satisfy_two_hits() -> None:
    # Two recorded hits but only one wire drop: the second hit is phantom.
    amount = EXPECTED_HITS[0]
    steps = [_step() for _ in range(6)]
    steps[1].damage_dealt = amount
    steps[2].damage_dealt = amount
    for w in range(1, 6):
        steps[w].wire_health = FULL_HEALTH - amount  # a single drop at w=1
    errors = reconcile_against_wire(FULL_HEALTH, steps)
    assert any("no matching wire-health drop" in e for e in errors)


# ---------------------------------------------------------------------------
# Step budget: the armored kill is ~2x longer, so a stale --max-steps has to be
# a message, not a run of mysterious timeouts.
# ---------------------------------------------------------------------------


def test_run_probe_rejects_a_step_cap_too_small_for_the_derived_kill() -> None:
    with pytest.raises(ValueError, match="cannot fit the expected"):
        run_probe(
            host="127.0.0.1",
            port=5555,
            cycles=1,
            seed=0,
            max_steps=WINDOWS_PER_SWING * len(EXPECTED_HITS),  # one short
            anchor=(0, 0),
            log=lambda *_: None,
        )


def test_run_probe_rejects_a_non_positive_cycle_count() -> None:
    with pytest.raises(ValueError, match="cycles must be"):
        run_probe(
            host="127.0.0.1",
            port=5555,
            cycles=0,
            seed=0,
            max_steps=80,
            anchor=(0, 0),
            log=lambda *_: None,
        )


def test_default_step_cap_fits_every_tier_in_the_table() -> None:
    # The CLI default is 80; no tier the probe can be pointed at may exceed it.
    for armor in ARMOR_SETS.values():
        needed = WINDOWS_PER_SWING * len(expected_hit_sequence(armor=armor)) + 1
        assert needed <= 80, armor.name


# ---------------------------------------------------------------------------
# Pad-anchor assertion (the non-zero-pad ride-along).
# ---------------------------------------------------------------------------


def test_anchor_check_passes_on_matching_pad() -> None:
    record = _record(_healthy_steps(), anchor=(512, 0))
    assert check_anchor(record, (512, 0)) == []


def test_anchor_check_fails_on_wrong_pad() -> None:
    # Bots that spawned on pad 0 while the probe expects the 512,0 pad — the
    # exact silent-relative-teleport failure the live check exists to catch.
    record = _record(_healthy_steps(), anchor=(0, 0))
    errors = check_anchor(record, (512, 0))
    assert len(errors) == 2  # both learner and dummy are misplaced


def test_anchor_check_fails_on_half_block_offset() -> None:
    # The negative-anchor "$(x).5" hazard: half a block off, no error anywhere.
    record = _record(_healthy_steps())
    record.start_opp_pos = (3.0, 64.0, 0.5)  # dummy at +3.0 instead of +3.5
    errors = check_anchor(record, (0, 0))
    assert len(errors) == 1 and "dummy" in errors[0]
