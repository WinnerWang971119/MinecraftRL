"""Unit tests for the pure analysis half of eval/combat_probe.py (T8).

The probe is the AC8 go/no-go gate, so the thing that must be provably correct
OFFLINE is its judgement: a probe that passes wrongly is worse than one that
fails. These tests drive ``analyze_cycle`` / ``reconcile_against_wire`` /
``check_anchor`` with scripted window records — the healthy 6,6,6,2 kill in
several legal timing variants, plus every failure class the probe exists to
catch (wrong sequence, phantom hit, unrecorded hit, regeneration, double death,
dirty baseline, wrong pad). No socket, no server.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import pytest

from eval.combat_probe import (
    EXPECTED_HITS,
    CycleRecord,
    StepRecord,
    analyze_cycle,
    check_anchor,
    extract_hits,
    reconcile_against_wire,
)


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


def _healthy_steps(
    *, skew: int = 0, respawn_masks_final_drop: bool = True
) -> List[StepRecord]:
    """The canonical 6,6,6,2 kill: hits in windows 1, 5, 9, 13.

    ``skew=0`` puts each wire-health drop in the same window as its damage
    event; ``skew=+1`` delays every drop one window (the second-connection
    ``update_health`` skew AC8 explicitly permits). With
    ``respawn_masks_final_drop`` the killing blow's 2 -> 0 is invisible because
    ``doImmediateRespawn`` snaps the window-end snapshot back to 20.
    """
    hit_windows = {1: 6.0, 5: 6.0, 9: 6.0, 13: 2.0}
    # Wire health at each window END, without skew.
    health_after = {1: 14.0, 5: 8.0, 9: 2.0, 13: 20.0 if respawn_masks_final_drop else 0.0}
    steps: List[StepRecord] = []
    health = 20.0
    for w in range(14):
        drop_window = w - skew
        if drop_window in health_after:
            health = health_after[drop_window]
        steps.append(
            _step(
                damage=hit_windows.get(w, 0.0),
                health=health,
                died=(w == 13),
                action=5 if w in hit_windows else 0,
            )
        )
    return steps


# ---------------------------------------------------------------------------
# Healthy cycles must pass, in every legal timing variant.
# ---------------------------------------------------------------------------


def test_healthy_cycle_passes() -> None:
    assert analyze_cycle(_record(_healthy_steps())) == []


def test_healthy_cycle_with_one_window_skew_passes() -> None:
    # update_health arrives on a second connection: drops one window late.
    assert analyze_cycle(_record(_healthy_steps(skew=1))) == []


def test_healthy_cycle_with_visible_final_drop_passes() -> None:
    # The snapshot catches 2 -> 0 before the respawn.
    steps = _healthy_steps(respawn_masks_final_drop=False)
    assert analyze_cycle(_record(steps)) == []


def test_extract_hits_orders_and_filters() -> None:
    steps = _healthy_steps()
    assert extract_hits(steps) == [(1, 6.0), (5, 6.0), (9, 6.0), (13, 2.0)]


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
    # Cooldown-violating half-power hit: 6, 3.38, ...
    steps = _healthy_steps()
    steps[5].damage_dealt = 3.38
    steps[5].wire_health = 10.62
    for w in range(6, 9):
        steps[w].wire_health = 10.62
    _assert_fails(_record(steps), "per-hit sequence")


def test_phantom_hit_without_wire_drop_fails() -> None:
    # A recorded damage event the wire never saw (over-counting).
    steps = _healthy_steps()
    steps[3].damage_dealt = 6.0
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
    for w in range(3, 5):
        steps[w].wire_health = 15.0  # 14 -> 15 heal, far from the death window
    errors = reconcile_against_wire(20.0, steps)
    assert any("INCREASED" in e for e in errors)


def test_cumulative_over_20_is_tc16_defect() -> None:
    steps = _healthy_steps()
    steps[13].damage_dealt = 8.0  # 6+6+6+8 = 26 > 20
    _assert_fails(_record(steps), "TC16")


def test_double_death_fails() -> None:
    steps = _healthy_steps()
    steps[12].opponent_died = True
    _assert_fails(_record(steps), "opponent_died fired in 2")


def test_zero_deaths_fails() -> None:
    steps = _healthy_steps()
    steps[13].opponent_died = False
    _assert_fails(_record(steps), "opponent_died fired in 0")


def test_dirty_start_baseline_fails() -> None:
    record = _record(_healthy_steps(), start_health=14.0)
    _assert_fails(record, "clean")


def test_non_win_outcome_fails() -> None:
    _assert_fails(_record(_healthy_steps(), outcome="timeout"), "expected a win")


def test_masked_killing_blow_with_wrong_entering_health_fails() -> None:
    # W1: the killing-blow carve-out (death nearby + drop masked by the
    # immediate respawn) must ONLY accept a hit whose amount equals the health
    # ENTERING its window. Here the wire entered the death window at 8, but the
    # channel recorded a 2 — a masked killing blow that does NOT reconcile.
    steps = [_step() for _ in range(14)]
    steps[1].damage_dealt = 6.0
    steps[5].damage_dealt = 6.0
    steps[13].damage_dealt = 2.0
    steps[13].opponent_died = True
    health = 20.0
    for w in range(14):
        if w == 1:
            health = 14.0
        elif w == 5:
            health = 8.0
        elif w == 13:
            health = 20.0  # respawn masks the (supposed) fatal drop
        steps[w].wire_health = health
    errors = reconcile_against_wire(20.0, steps)
    assert any("no matching wire-health drop" in e for e in errors), errors


def test_cumulative_total_off_by_under_two_fails() -> None:
    # W2: a small negative damage_dealt slipping through leaves the positive
    # hit sequence intact (extract_hits filters it) and only the exact
    # cumulative-20 check can catch it. Off by 0.5 must still FAIL.
    steps = _healthy_steps()
    steps[2].damage_dealt = -0.5  # total = 19.5, sequence still 6,6,6,2
    _assert_fails(_record(steps), "cumulative")


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
    # Two recorded 6s but only one wire drop of 6: the second hit is phantom.
    steps = [_step() for _ in range(6)]
    steps[1].damage_dealt = 6.0
    steps[2].damage_dealt = 6.0
    for w in range(1, 6):
        steps[w].wire_health = 14.0  # a single 20 -> 14 drop at w=1
    errors = reconcile_against_wire(20.0, steps)
    assert any("no matching wire-health drop" in e for e in errors)


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


def test_expected_hits_constant_is_the_iron_sword_arithmetic() -> None:
    assert EXPECTED_HITS == (6.0, 6.0, 6.0, 2.0)
    assert sum(EXPECTED_HITS) == 20.0
