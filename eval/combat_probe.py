"""combat_probe — T8 deterministic combat gate (AC8, the go/no-go gate).

Drives ``--cycles`` reset/kill cycles against the LIVE stack (Paper -> bridge ->
this driver): face the dummy from the spawn posture, issue an ATTACK whenever
the wire reports the swing fully cooled (``state.self.attack_cooldown == 1.0``),
IDLE otherwise, and record every decision window. Per cycle it asserts:

  * the recorded per-hit ``events.damage_dealt`` sequence is exactly the one
    :func:`expected_hit_sequence` DERIVES from the target's loadout (iron sword
    vs a 20 HP dummy, regeneration off). Against the M4 iron set that is
    ``3.12`` six times then ``1.28`` — NOT the bare-handed ``6, 6, 6, 2`` this
    probe shipped with; see "THE SEQUENCE IS DERIVED" below;
  * cumulative dealt damage is exactly 20. That total is loadout-INDEPENDENT
    (it is the target's full health); only the per-hit split moves;
  * ``events.opponent_died`` fires in exactly one window and the episode ends
    as a win;
  * the episode starts from a clean baseline (wire opponent health == 20 at the
    first post-reset state — which also proves the first post-respawn hit of
    the NEXT cycle measures from a clean baseline);
  * every recorded damage value RECONCILES against the wire's privileged
    ``state.opponent.health`` at +/-1 window (the dummy's ``update_health``
    arrives on a second connection, so a one-window skew is legal);
  * no unexplained wire-health drop (an unrecorded hit) and no wire-health
    increase outside the death/respawn window (regeneration is off, so any
    heal is a defect);
  * TC16: cumulative dealt damage > 20 in any episode is a defect, not noise.

THE WIRE HEALTH IS THE ORACLE, NOT THE PRODUCTION PATH. ``state.opponent.health``
is used here strictly as the free, independent cross-check of the repaired
``events`` channel (see the plan's Decisions). Deriving ``damage_dealt`` from it
in production was considered and rejected — this probe must never be read as a
template for that.

THE SEQUENCE IS DERIVED, NOT HARDCODED (T23). This probe shipped with the three
literals ``6, 6, 6, 2`` / total 20, correct for an UNARMORED dummy. M4 (issue
#33) put a full iron set on it — ``spawn_dummy_pad.mcfunction``'s re-gear block
ends with four ``$item replace entity $(dummy) armor.<slot> ...`` lines — and a
full iron set absorbs 48% of an incoming iron-sword hit. A probe still asserting
``6, 6, 6, 2`` FAILS against a healthy fleet, and it fails looking exactly like
the dead damage channel it was built to detect: under-counted per-hit damage.
That is the worst false positive this file can produce, so the expectation is
now COMPUTED from the target's loadout (:data:`ARMOR_SETS`,
:func:`damage_after_absorb`, :func:`expected_hit_sequence`) and a future armor
tier is a table row plus ``--target-armor``, not a rewrite. Run the probe with
``--target-armor none`` and the derivation reproduces ``6, 6, 6, 2`` exactly, so
a revert or a bare-handed A/B needs no code change either — that identity is
pinned by a unit test.

WHAT THE ASSERTION STILL IS. Not "some damage happened": the derived numbers are
EXACT, computed independently of the wire, and every one of them is still
reconciled hit-by-hit against ``state.opponent.health``. The only thing that
loosened is the PER-HIT float comparison — see ``_DAMAGE_TOL`` below, which is
pinned between the server's float32 rounding (~2e-6) and the quietest real
defect the probe must catch (one armor point == 0.24 HP). The cumulative check
did not loosen: it still holds the total to the target's full health at the
tight ``_TOL``.

SINGLE-CONNECTION DISCIPLINE. ``BridgeServer`` accepts exactly ONE TCP client
and a second connection silently destroys the first. The probe therefore opens
ONE :class:`~env.mc_pvp_env.TcpBridgeClient` (wrapped in a recording shim so the
raw ``state`` messages stay inspectable), constructs ONE
:class:`~env.mc_pvp_env.MCPvPEnv` over it with ``auto_connect=False``, and runs
every cycle through that single env/connection — the same borrow pattern as
``agent.train._eval_against_dummy``.

FALSE-FAIL MODE (D2, issue #28) — HISTORICAL, NOW FIXED. Read this before
assuming a red run means the damage channel is broken; it doesn't anymore, but
the history is kept because a future regression here will look exactly like it
did the first time. One fresh-boot run recorded a weak w0 swing (observed live:
``1.269, 6, 6, 6, 0.731`` — still reconciling to exactly 20) and failed the
sequence assertion ON A CORRECT DAMAGE CHANNEL (1 of 3 fresh-boot runs
observed; the sequence was exact across the other 48/48 live cycles, all of
them past cycle 0). THOSE FIVE NUMBERS ARE FROM THE BARE-HANDED ERA and are
kept verbatim as the recorded evidence — the shape to recognise is "a weak
first swing, the rest full, the total still exact", not the literal values,
which through today's iron armor would be smaller (see
:func:`damage_for_swing_charge`). That weak swing is itself a rung of the
quantized ladder in the TPS FLOOR caveat below, which is part of why the
recorded evidence is worth keeping: bare-handed, ``ticker = 1`` gives
``f = (1 + 0.5) / 12.5 == 0.12`` and ``6 * (0.2 + 0.12**2 * 0.8) == 1.26912``,
and the run's trailing ``0.731`` is the ``20 - 3*6 - 1.26912 == 0.73088`` the
remaining-health clamp left.

The mechanism first written down here was WRONG. It blamed the reset's
``/clear`` + ``/give`` re-equip for zeroing the server's attack-strength meter.
Decompiling the pinned ``server/versions/1.21.1/paper-1.21.1.jar``
(Mojang-mapped, readable with ``javap -p -c``) shows ``Player.tick()`` only
calls ``resetAttackStrengthTicker()`` when the main-hand item TYPE changes
tick-over-tick; a same-tick ``/clear`` + ``/give`` of the SAME item is invisible
to it — both ends of the tick still read ``iron_sword``. The real sources of a
zeroed meter at episode start are (a) the learner joining EMPTY-HANDED, where
``air -> iron_sword`` genuinely is a type change (a playerdata property, not
stochastic — see the "Playerdata sampling" caveat below), and (b) the previous
episode's final kill swing, which zeroes the meter server-side while
``handleReset`` clears the bridge's ``lastSwingTick`` so the bridge forgets it
happened.

That reporting gap is now fixed (commit ``8ab2634``, T18): ``bridge/bot.js``'s
``attackCooldown()`` also ramps from the reset boundary (``_meterResetTick``,
which stands in for both sources above) and reports the MINIMUM of that ramp
and the swing-tracker ramp, so the wire no longer claims a charged swing at
episode start when the server isn't actually ready. This probe already drives
off that value (see ``_COOLDOWN_READY`` below) — it IDLEs the early windows and
opens its first ATTACK only once the wire reports ready, with no special-casing
of cycle 0. DO NOT weaken the per-hit sequence assertion to tolerate a weak
first hit: a red run here now is a regression, not expected noise. T23 changed
WHICH numbers that assertion expects; it did not loosen the assertion, and
``_DAMAGE_TOL`` is nowhere near wide enough to swallow a partial swing.

TWO LIVE CAVEATS remain, since a T13 operator will gate on this probe's exit
code unattended:

  * TPS FLOOR. The first ATTACK lands at bridge tick 16, ~800ms of wall clock
    into the episode. At exactly 15 TPS that is server ticker 12, giving
    ``f = (12 + 0.5) / 12.5 == 1.0`` with zero margin — so below 15 TPS the
    first swing is not fully charged, the first hit lands SHORT, and the exact
    per-hit assertion fails WITH THE IMPLEMENTATION CORRECT.

    THE SHORTFALL IS QUANTIZED — THERE IS NO NARROW BAND UNDER A FULL HIT.
    ``LivingEntity.attackStrengthTicker`` is an ``int`` and
    ``Player.getCurrentItemAttackStrengthDelay()`` is ``20 / 1.6 == 12.5``, so
    the ``f = getAttackStrengthScale(0.5f)`` that ``Player.attack`` reads can
    only take the values ``(ticker + 0.5) / 12.5`` — steps of 0.08, and one
    whole tick is the smallest miss there is. Armor attenuates that miss
    further, because ``Player.attack`` scales the RAW weapon damage by
    ``0.2 + f*f*0.8`` BEFORE ``LivingEntity.getDamageAfterArmorAbsorb`` runs
    and absorption is not linear in the incoming damage. Against the default
    iron target the top of the reachable ladder is::

        ticker 12   f = 1.00   lands 3.1200   (100%, the expectation)
        ticker 11   f = 0.92   lands 2.6590   ( 85%, one tick short)
        ticker 10   f = 0.84   lands 2.2555   ( 72%, two ticks short)
        ticker  0   f = 0.04   lands 0.5122   ( 16%, meter still at zero)

    DIAGNOSTIC. The shortfall jumps straight from 0 to ~0.46 HP, so a first hit
    of 2.659 (or 2.256) together with measured world-age TPS below 15 IS the
    cooldown floor and not a damage-channel fault — do NOT reject it as "far
    more than a timing miss could explain", because a reading a few thousandths
    under 3.12 is one the server cannot produce. A first hit at or below ~0.6
    is the other diagnosis: the meter had not ramped at all (ticker <= 2) when
    the swing went out, which is a genuinely wrong cooldown anchor rather than
    a slow server. Price any other suspect reading with
    :func:`damage_for_swing_charge` instead of eyeballing a band; every number
    above is pinned by unit test. This applies to the FIRST swing of a cycle
    only — later swings are gated on ``lastSwingTick``, ride the same bridge
    clock, and inherit the same 16-tick spacing.
  * PLAYERDATA SAMPLING. The empty-hand branch above only fires at cycle 0 if
    the learner's playerdata has an empty main hand at join. Against a
    persisted ``server/world`` the learner rejoins already holding last
    session's sword, cycle 0 degenerates to steady state, and a run against
    that world cannot reproduce the original regression even if it returned.

Usage (Paper and the bridge already running, in that order):

    python -m eval.combat_probe --cycles 10
    python -m eval.combat_probe --cycles 5 --expect-anchor 512,0   # non-zero pad
    python -m eval.combat_probe --cycles 5 --target-armor none     # bare-handed A/B

The run prints the loadout it derived its expectation from before the first
cycle, so a mismatch between the probe's assumption and what the datapack
actually equips is visible in the log rather than inferred from a red run.

Exit codes: 0 = AC8 PASS across all cycles, 1 = FAIL (any assertion, any
non-win outcome), 2 = no verdict. A 2 is either a :class:`BridgeError` — the
probe could not connect at all, or a mid-episode transport abort ended the run
early (step() never silently retries; a lost in-flight reply is an
unrecoverable desync) — or a configuration the probe refused before opening the
socket (a step cap too small for the derived hit sequence). A 2 therefore means
"no verdict", never "pass".

Owner: T8 (Eval/infra track)
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from agent.actions import Macro
from agent.contract_config import ACTION_REPEAT, SERVER_TPS
from bridge.messages import ResetAckMsg, StateMsg
from env.mc_pvp_env import BridgeError, MCPvPEnv, TcpBridgeClient

__all__ = [
    "RecordingTransport",
    "StepRecord",
    "CycleRecord",
    "ArmorSet",
    "damage_after_absorb",
    "damage_for_swing_charge",
    "expected_hit_sequence",
    "extract_hits",
    "reconcile_against_wire",
    "analyze_cycle",
    "run_probe",
    "main",
    "ARMOR_SETS",
    "NO_ARMOR",
    "LEATHER_ARMOR",
    "CHAINMAIL_ARMOR",
    "IRON_ARMOR",
    "TARGET_ARMOR",
    "IRON_SWORD_DAMAGE",
    "IRON_SWORD_ATTACK_SPEED_TICKS",
    "WINDOWS_PER_SWING",
    "EXPECTED_HITS",
    "EXPECTED_TOTAL",
    "FULL_HEALTH",
]

# ---------------------------------------------------------------------------
# Expected combat arithmetic (AC8) — DERIVED FROM THE LOADOUT, NOT HARDCODED.
#
# EVERY CONSTANT BELOW IS READ OUT OF THE PINNED JAR
# (``server/versions/1.21.1/paper-1.21.1.jar``, Mojang-mapped, ``javap -p -c``).
# Do not "correct" one from memory; re-read the class named beside it.
# ---------------------------------------------------------------------------

#: The dummy's max health. ``spawn_dummy_pad.mcfunction`` heals it to full every
#: reset and ``naturalRegeneration`` is off, so this is both the starting health
#: and the exact cumulative damage a clean kill must deal.
FULL_HEALTH: float = 20.0


@dataclass(frozen=True)
class ArmorSet:
    """A worn armor loadout, reduced to the only two things absorption reads.

    ``LivingEntity.getDamageAfterArmorAbsorb`` passes exactly two numbers into
    the formula: ``getArmorValue()`` (the ARMOR attribute, floored to an int)
    and the ARMOR_TOUGHNESS attribute. Which four items produced them does not
    enter the arithmetic, so this carries the TOTAL of the four worn slots
    rather than a per-piece map.

    Attributes:
        name: CLI/log name for this tier (``--target-armor``).
        points: Sum of the four worn pieces' defense values.
        toughness: The tier's armor-toughness contribution (0.0 below netherite
            and diamond).
    """

    name: str
    points: int
    toughness: float

    def __post_init__(self) -> None:
        if self.points < 0:
            raise ValueError(f"armor points must be >= 0, got {self.points}")
        if self.toughness < 0:
            raise ValueError(f"armor toughness must be >= 0, got {self.toughness}")


# ---------------------------------------------------------------------------
# THE ARMOR TABLE. Per-slot defense values come from
# ``net/minecraft/world/item/ArmorMaterials``: ``register(<name>, <EnumMap>,
# <enchantmentValue>, <equipSound>, <toughness>, <knockbackResistance>, ...)``,
# with the EnumMap filled by that tier's ``lambda$static$N``. Read for this
# task, boots/leggings/chestplate/helmet (the ``Type.BODY`` entry in the same
# map is the animal-armor slot and is NOT worn by a player):
#
#     leather    1 + 2 + 3 + 1 ==  7   toughness 0.0   (lambda$static$0)
#     chainmail  1 + 4 + 5 + 2 == 12   toughness 0.0   (lambda$static$2)
#     iron       2 + 5 + 6 + 2 == 15   toughness 0.0   (lambda$static$4)
#
# Both float arguments to each of those three ``register`` calls are
# ``fconst_0``, so none of them adds toughness OR knockback resistance.
#
# ADDING A TIER (issue #33 floats leather and chainmail as alternatives): add a
# row here with its own jar reading and it is reachable from
# ``--target-armor``. Diamond and netherite are deliberately ABSENT rather than
# guessed — they carry non-zero toughness, which changes the ``f`` term, and
# nothing in this repo has read their numbers off the jar yet.
# ---------------------------------------------------------------------------

NO_ARMOR: ArmorSet = ArmorSet("none", 0, 0.0)
LEATHER_ARMOR: ArmorSet = ArmorSet("leather", 7, 0.0)
CHAINMAIL_ARMOR: ArmorSet = ArmorSet("chainmail", 12, 0.0)
IRON_ARMOR: ArmorSet = ArmorSet("iron", 15, 0.0)

#: Name -> tier, for ``--target-armor``.
ARMOR_SETS: Dict[str, ArmorSet] = {
    s.name: s for s in (NO_ARMOR, LEATHER_ARMOR, CHAINMAIL_ARMOR, IRON_ARMOR)
}

#: What the datapack ACTUALLY equips the dummy with today — the four
#: ``$item replace entity $(dummy) armor.<slot> with minecraft:iron_*`` lines at
#: the end of ``server/arena/data/arena/function/spawn_dummy_pad.mcfunction``.
#: Change this only in the same commit as that file.
TARGET_ARMOR: ArmorSet = IRON_ARMOR

#: Damage of one fully-charged iron-sword hit from a player, before absorption.
#: ``Items.IRON_SWORD`` is ``new SwordItem(Tiers.IRON, ...attributes(
#: SwordItem.createAttributes(Tiers.IRON, 3, -2.4f)))``;
#: ``SwordItem.createAttributes`` adds ``3 + Tier.getAttackDamageBonus()`` to
#: ATTACK_DAMAGE, and ``Tiers.IRON``'s bonus is ``2.0f``, so the item adds 5.0
#: on top of the player's ATTACK_DAMAGE base of ``1.0`` (``Player
#: .createAttributes``: ``dconst_1``). 5 + 1 == 6.
IRON_SWORD_DAMAGE: float = 6.0

#: Attacks per second with that sword: the ATTACK_SPEED attribute's base is
#: ``4.0`` (``Attributes.ATTACK_SPEED``, a RangedAttribute registered with
#: ``4.0d``) and ``createAttributes`` above adds ``-2.4f``.
IRON_SWORD_ATTACK_SPEED: float = 1.6

#: Ticks for a full swing recharge. Mirrors ``bridge/actions.js``'s
#: ``IRON_SWORD_ATTACK_SPEED_TICKS = 20 / 1.6``, derived here from the same two
#: numbers so the two sides cannot drift silently. == 12.5.
IRON_SWORD_ATTACK_SPEED_TICKS: float = SERVER_TPS / IRON_SWORD_ATTACK_SPEED

#: Decision windows between two fully-charged swings. 12.5 cooldown ticks over
#: a 4-tick decision window rounds UP to 4 windows — the same
#: ``ceil(12.5 / ACTION_REPEAT) == 4`` the bridge documents in
#: ``bridge/bot.js``'s attack-meter section. Used only to size the per-cycle
#: step cap; the probe never assumes this spacing, it reads readiness off the
#: wire.
WINDOWS_PER_SWING: int = math.ceil(IRON_SWORD_ATTACK_SPEED_TICKS / ACTION_REPEAT)

#: Float comparison tolerance for WIRE-AGAINST-WIRE arithmetic: recorded
#: ``damage_dealt`` against the ``opponent.health`` drop that produced it, the
#: clean-baseline check, and the nonzero-damage threshold. Both sides of every
#: such comparison are the SAME float32 values the server put on the wire,
#: subtracted the same way on both sides, so they agree bit-for-bit and 1e-6 is
#: generous. ``eval/benchmark.py`` mirrors this value in ``_RECONCILE_TOL``.
_TOL: float = 1e-6

#: Float comparison tolerance for a RECORDED value against the expectation
#: DERIVED below. This is the only comparison that crosses precisions, so it is
#: the only one that needs to be looser than ``_TOL`` — and it must not be much
#: looser. Both bounds are pinned, and both are pinned by unit tests:
#:
#:   LOWER BOUND (float32). The server does this arithmetic in float32 and the
#:   wire carries float32 healths; the expectation below is Python float64.
#:   Replaying the iron cascade in float32 (20 -> 16.880001 -> 13.760001 ->
#:   ... -> 1.2800016 -> 0) puts the largest float32-vs-float64 divergence at
#:   1.6e-6, and the drift is bounded by one ``ulp(20.0f) == 1.9e-6`` per hit,
#:   so even a 20-hit tier stays under 4e-5.
#:
#:   UPPER BOUND (the quietest real defect). The smallest armor fault that can
#:   exist is ONE armor point — one iron piece silently swapped for leather
#:   takes the target from 15 points to 14 and the landed hit from 3.12 to
#:   3.36. That is 0.24 HP: 2400x this tolerance. A missing piece, a wrong
#:   tier, a half-charged swing and a dead damage channel are all larger still.
#:
#: 1e-4 therefore sits ~60x above the float noise and ~2400x below the quietest
#: failure the probe exists to catch. Loosening it past ~0.1 would start
#: tolerating real defects, which is worse than a red run.
_DAMAGE_TOL: float = 1e-4


def damage_after_absorb(damage: float, armor_points: float, toughness: float) -> float:
    """Damage that lands after armor absorption — the jar's formula, verbatim.

    ``net/minecraft/world/damagesource/CombatRules.getDamageAfterAbsorb``
    (pinned jar, read with ``javap -p -c``) is, in bytecode order::

        f = 2.0 + toughness / 4.0
        g = Mth.clamp(armor - damage / f, armor * 0.2, 20.0)
        return damage * (1.0 - g / 25.0)

    ARMOR IS NOT A FLAT PERCENT PER POINT and has not been since 1.9: ``g``
    depends on the incoming ``damage``, so the same set absorbs a different
    FRACTION of a weak hit than of a strong one. That is exactly why a probe
    cannot patch a percentage over its old constants.

    THE ONE TERM NOT MODELLED HERE. Between the clamp and the return, 1.21.1
    routes the ratio through ``EnchantmentHelper.modifyArmorEffectiveness``
    when the damage source carries a weapon and the level is a ``ServerLevel``.
    That call walks the WEAPON's enchantments and returns its input unchanged
    when there are none (it is a ``MutableFloat`` seeded with the ratio and
    handed to ``runIterationOnItem``). Both fighters are geared by ``$give`` /
    ``$item replace`` with plain items and this repo has no enchanting path at
    all, so the term is identity here. It stops being identity the moment
    anything enchants a weapon — that would need modelling, not a wider
    tolerance.

    Args:
        damage: Incoming damage BEFORE absorption (already scaled by the
            attack-strength charge, if it was a partial swing).
        armor_points: The target's ARMOR attribute.
        toughness: The target's ARMOR_TOUGHNESS attribute.

    Returns:
        The damage that actually lands.

    Raises:
        ValueError: If any argument is negative or non-finite.
    """
    for label, value in (
        ("damage", damage),
        ("armor_points", armor_points),
        ("toughness", toughness),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{label} must be finite and >= 0, got {value!r}")
    f = 2.0 + toughness / 4.0
    g = min(max(armor_points - damage / f, armor_points * 0.2), 20.0)
    return damage * (1.0 - g / 25.0)


def damage_for_swing_charge(
    charge: float,
    *,
    weapon_damage: float = IRON_SWORD_DAMAGE,
    armor: ArmorSet = TARGET_ARMOR,
) -> float:
    """Damage a swing at attack-strength scale ``charge`` lands through ``armor``.

    ``Player.attack`` in the pinned jar multiplies the raw weapon damage by
    ``0.2f + f * f * 0.8f``, where ``f = getAttackStrengthScale(0.5f)``, and
    does it BEFORE ``LivingEntity.hurt`` reaches armor absorption. So a partial
    swing is attenuated twice, and not proportionally: absorption depends on
    the incoming damage (see :func:`damage_after_absorb`).

    The probe never swings partially on purpose — it IDLEs until the wire
    reports ``attack_cooldown == 1.0``. This exists to price a suspect reading
    during triage: it turns "the first hit was 2.659, is that the TPS floor or
    a broken cooldown anchor?" into an answer, instead of a band memorised from
    the bare-handed era that armor has since moved.

    ONLY SOME CHARGES ARE REACHABLE LIVE. ``getAttackStrengthScale(0.5f)``
    divides an ``int`` ticker by the 12.5-tick delay, so a real swing always
    sits on ``(ticker + 0.5) / 12.5`` — 0.04, 0.12, ..., 0.92, 1.0, in steps of
    0.08. ``charge`` is deliberately NOT restricted to those, since pricing a
    hypothetical is useful; but quote a LADDER value when documenting what an
    operator can actually SEE, or the prose describes a reading the server
    cannot produce.

    Args:
        charge: Attack-strength scale in ``[0, 1]``; 1.0 is a fully-cooled swing.
        weapon_damage: Raw weapon damage at full charge.
        armor: The target's worn loadout.

    Returns:
        The damage such a swing would land.

    Raises:
        ValueError: If ``charge`` is outside ``[0, 1]`` or ``weapon_damage`` is
            negative / non-finite.
    """
    if not math.isfinite(charge) or not (0.0 <= charge <= 1.0):
        raise ValueError(f"charge must be a finite value in [0, 1], got {charge!r}")
    raw = weapon_damage * (0.2 + charge * charge * 0.8)
    return damage_after_absorb(raw, armor.points, armor.toughness)


def expected_hit_sequence(
    *,
    weapon_damage: float = IRON_SWORD_DAMAGE,
    armor: ArmorSet = TARGET_ARMOR,
    target_health: float = FULL_HEALTH,
) -> Tuple[float, ...]:
    """The per-hit ``damage_dealt`` sequence a clean kill must produce.

    Every blow but the last is a fully-charged hit through ``armor``; the last
    is whatever health remains.

    WHY THE TRAILING ELEMENT IS SHORT — the reasoning the old ``6, 6, 6, 2``
    encoded, restated because it survives the armor change unaltered.
    ``damage_dealt`` is not the server's damage roll: ``bridge/bot.js`` derives
    it from the DROP in the dummy's own ``health`` channel, and
    ``LivingEntity.setHealth`` clamps health at 0. A killing blow therefore
    reports the remaining health, never the full swing — bare-handed that was
    ``20 - 3 * 6 == 2``, and through iron it is ``20 - 6 * 3.12 == 1.28``. The
    trailing element is the remaining-health clamp on the FATAL blow, NOT a
    partially-charged swing.

    THE COOLDOWN'S ROLE IS THE OTHER ONE, and it is why the leading elements
    are all equal: a swing inside the cooldown is scaled by
    ``0.2 + f*f*0.8`` before absorption (:func:`damage_for_swing_charge`), so
    ANY early swing would break this sequence by construction. That is what the
    driver's IDLE-until-``attack_cooldown == 1.0`` gate buys, and why the gate
    must not be relaxed to speed the probe up.

    Args:
        weapon_damage: Raw damage of one fully-charged hit.
        armor: The target's worn loadout.
        target_health: The target's health at the start of the episode.

    Returns:
        The exact per-hit sequence, summing to ``target_health``.

    Raises:
        ValueError: If ``weapon_damage`` or ``target_health`` is not positive
            and finite, or if the loadout leaves no damage getting through.
    """
    if not math.isfinite(weapon_damage) or weapon_damage <= 0.0:
        raise ValueError(f"weapon_damage must be finite and > 0, got {weapon_damage!r}")
    if not math.isfinite(target_health) or target_health <= 0.0:
        raise ValueError(f"target_health must be finite and > 0, got {target_health!r}")

    per_hit = damage_after_absorb(weapon_damage, armor.points, armor.toughness)
    if per_hit <= _DAMAGE_TOL:
        # Unreachable for any tier in ARMOR_SETS. What guarantees that is the
        # clamp's UPPER bound, `CombatRules.MAX_ARMOR == 20.0f`: `g <= 20` so
        # `g / 25 <= 0.8`, and >= 20% of every swing lands however much armor
        # the target wears. (NOT the `armor * 0.2` term — that is
        # `MIN_ARMOR_RATIO`, the clamp's LOWER bound; it stops heavy armor from
        # being erased by a huge hit and guarantees nothing about the ceiling.
        # Both names and both clamp positions are read off the pinned jar.)
        # A caller-supplied loadout must still not be able to spin this loop
        # forever.
        raise ValueError(
            f"{armor.name} armor absorbs a {weapon_damage:g} hit down to "
            f"{per_hit:g}, which cannot kill a {target_health:g} HP target"
        )

    hits: List[float] = []
    remaining = float(target_health)
    while remaining > _DAMAGE_TOL:
        blow = per_hit if per_hit < remaining else remaining
        hits.append(blow)
        remaining -= blow
    return tuple(hits)


#: The sequence the LIVE probe asserts, for the loadout the datapack equips
#: today. Against ``TARGET_ARMOR`` (iron) that is ``3.12`` six times then
#: ``1.28`` — seven hits, not four.
EXPECTED_HITS: Tuple[float, ...] = expected_hit_sequence()

#: Cumulative damage a clean kill deals. Loadout-INDEPENDENT: the target starts
#: at full health, regeneration is off, and the fatal blow is clamped to what is
#: left, so the total is the target's health no matter how it is split. This is
#: what :func:`analyze_cycle` holds the live total to — an anchor OUTSIDE the
#: derivation, which ``math.fsum(EXPECTED_HITS)`` would not be.
EXPECTED_TOTAL: float = FULL_HEALTH

#: Position tolerance for the spawn-anchor assertion — the same posEpsilon the
#: bridge's own read-back gate uses (bridge/bot.js DEFAULT_READBACK).
_POS_EPS: float = 0.25

#: Swing readiness threshold on the wire's [0,1] attack_cooldown.
_COOLDOWN_READY: float = 1.0 - _TOL


# ---------------------------------------------------------------------------
# Recording transport shim.
# ---------------------------------------------------------------------------


class RecordingTransport:
    """Wraps a :class:`BridgeTransport` and records the raw inbound messages.

    The env consumes parsed ``StateMsg``/``ResetAckMsg`` dataclasses through the
    transport seam but only exposes gated observations; the probe needs the RAW
    wire state (privileged ``opponent.health``, ``self.attack_cooldown``,
    positions). This shim is transparent to the env and keeps the last seen
    message of each type inspectable.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.last_state: Optional[StateMsg] = None
        self.last_reset_ack: Optional[ResetAckMsg] = None

    def connect(self) -> None:
        self._inner.connect()

    def send(self, obj: Mapping[str, Any]) -> None:
        self._inner.send(obj)

    def recv(self) -> Union[StateMsg, ResetAckMsg]:
        msg = self._inner.recv()
        if isinstance(msg, StateMsg):
            self.last_state = msg
        elif isinstance(msg, ResetAckMsg):
            self.last_reset_ack = msg
        return msg

    def close(self) -> None:
        self._inner.close()


# ---------------------------------------------------------------------------
# Per-cycle records (plain data so the analysis below stays pure/unit-testable).
# ---------------------------------------------------------------------------


@dataclass
class StepRecord:
    """One decision window as the probe observed it.

    Attributes:
        action: The action index the probe issued for this window.
        damage_dealt: ``events.damage_dealt`` reported for this window.
        opponent_died: ``events.opponent_died`` reported for this window.
        wire_health: Raw ``state.opponent.health`` at the END of this window.
        attack_cooldown: Raw ``state.self.attack_cooldown`` at window end.
        tick: ``state.tick`` at window end.
    """

    action: int
    damage_dealt: float
    opponent_died: bool
    wire_health: float
    attack_cooldown: float
    tick: int


@dataclass
class CycleRecord:
    """Everything recorded for one reset/kill cycle."""

    index: int
    reset_ms: float
    start_health: float
    start_self_pos: Tuple[float, float, float]
    start_opp_pos: Tuple[float, float, float]
    outcome: str  # "win" | "loss" | "timeout"
    steps: List[StepRecord] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pure analysis (unit-tested in tests/test_combat_probe.py).
# ---------------------------------------------------------------------------


def extract_hits(steps: Sequence[StepRecord]) -> List[Tuple[int, float]]:
    """Return ``(window_index, amount)`` for every window with nonzero damage."""
    return [
        (i, float(s.damage_dealt)) for i, s in enumerate(steps) if s.damage_dealt > _TOL
    ]


def reconcile_against_wire(
    start_health: float, steps: Sequence[StepRecord]
) -> List[str]:
    """Cross-check recorded damage events against the wire's opponent health.

    Rules (all violations are returned as human-readable error strings):

      * Every recorded hit must match a wire-health DROP of the same amount in
        its own window or an adjacent one (+/-1 — the dummy's ``update_health``
        arrives on a second connection). Each drop can satisfy only one hit.
      * The killing blow may be invisible as a drop: with ``doImmediateRespawn``
        the window-end snapshot can already read the respawned 20, so a hit
        whose windows +/-1 contain ``opponent_died`` reconciles instead against
        the health ENTERING its window (the blow equals remaining health).
      * A wire drop that no recorded hit explains is an UNRECORDED hit — the
        exact under-counting failure the repair must not have.
      * A wire-health INCREASE outside the death/respawn neighbourhood is a
        heal; with regeneration off it is a defect.
    """
    errors: List[str] = []
    n = len(steps)
    # health_entering[w] is the wire health at the START of window w;
    # steps[w].wire_health is the health at its end.
    health_entering = [float(start_health)] + [float(s.wire_health) for s in steps[:-1]]

    drops: Dict[int, float] = {}
    increases: Dict[int, float] = {}
    for w in range(n):
        delta = health_entering[w] - float(steps[w].wire_health)
        if delta > _TOL:
            drops[w] = delta
        elif delta < -_TOL:
            increases[w] = -delta

    death_windows = {w for w in range(n) if steps[w].opponent_died}

    def _near_death(w: int) -> bool:
        return any(u in death_windows for u in (w - 1, w, w + 1))

    hits = extract_hits(steps)
    used_drops: set = set()
    for w, amount in hits:
        matched = False
        for u in (w, w - 1, w + 1):  # exact window first, then the skew cases
            if u in drops and u not in used_drops and abs(drops[u] - amount) <= _TOL:
                used_drops.add(u)
                matched = True
                break
        if matched:
            continue
        # Death path: the fatal drop can be masked by the immediate respawn.
        if _near_death(w) and abs(health_entering[w] - amount) <= _TOL:
            continue
        errors.append(
            f"recorded hit of {amount:g} in window {w} has no matching wire-health "
            f"drop within +/-1 window (wire drops: "
            f"{ {k: round(v, 3) for k, v in sorted(drops.items())} })"
        )

    for u in sorted(drops):
        if u not in used_drops:
            errors.append(
                f"wire health dropped {drops[u]:g} in window {u} with no recorded "
                f"damage event to explain it (unrecorded hit)"
            )

    for u in sorted(increases):
        if not _near_death(u):
            errors.append(
                f"wire health INCREASED {increases[u]:g} in window {u} outside the "
                f"death/respawn neighbourhood — a heal with regeneration off is a "
                f"defect"
            )

    return errors


def analyze_cycle(
    record: CycleRecord, expected_hits: Sequence[float] = EXPECTED_HITS
) -> List[str]:
    """All AC8 assertions for one cycle. Returns a list of failures (empty == pass).

    ``expected_hits`` defaults to :data:`EXPECTED_HITS` — the sequence derived
    for the loadout the datapack equips today. Pass an
    :func:`expected_hit_sequence` computed for a different ``ArmorSet`` to judge
    a run against a different (or reverted, or bare-handed) loadout; nothing
    here assumes armor is present.

    Only the PER-HIT check reads ``expected_hits``. The cumulative check is
    deliberately anchored on :data:`EXPECTED_TOTAL` instead, so it stays a real
    check even when the derivation handed in is the thing that is wrong.
    """
    errors: List[str] = []

    if record.outcome != "win":
        errors.append(f"episode ended as {record.outcome!r}, expected a win")

    if abs(record.start_health - FULL_HEALTH) > _TOL:
        errors.append(
            f"episode started from wire opponent health {record.start_health:g}, "
            f"expected a clean {FULL_HEALTH:g} baseline"
        )

    hits = extract_hits(record.steps)
    amounts = [amount for _, amount in hits]
    expected = [float(v) for v in expected_hits]
    # _DAMAGE_TOL, not _TOL: this is the one comparison that crosses precisions
    # (a float32 wire value against a float64 expectation computed here). See
    # the tolerance's own note for both bounds.
    if len(amounts) != len(expected) or any(
        abs(a - e) > _DAMAGE_TOL for a, e in zip(amounts, expected)
    ):
        errors.append(
            f"per-hit sequence {[round(a, 4) for a in amounts]} != expected "
            f"{[round(e, 4) for e in expected]} (tolerance {_DAMAGE_TOL:g})"
        )

    # THE LOADOUT-INDEPENDENT ANCHOR. Every damage_dealt is a health DROP
    # (h_i - h_{i+1}), so the total telescopes to h_0 - h_n == FULL_HEALTH - 0
    # for EVERY loadout. This is therefore compared against EXPECTED_TOTAL and
    # NOT against fsum(expected_hits): if the derivation above is wrong, an
    # fsum(expected) comparison moves both of its own sides together and can
    # never fire, while the target's full health is fixed by the clean-baseline
    # check a few lines up. Both sides being wire values, it is also held to
    # the tight _TOL rather than the per-hit check's _DAMAGE_TOL.
    # fsum, not sum, so the total carries no avoidable accumulation error.
    total = math.fsum(s.damage_dealt for s in record.steps)
    if abs(total - EXPECTED_TOTAL) > _TOL:
        errors.append(f"cumulative dealt damage {total:g} != {EXPECTED_TOTAL:g}")
    if total > EXPECTED_TOTAL + _TOL:
        errors.append(
            f"TC16: cumulative dealt damage {total:g} > {EXPECTED_TOTAL:g} with "
            f"regeneration off is a defect, not noise"
        )

    deaths = sum(1 for s in record.steps if s.opponent_died)
    if deaths != 1:
        errors.append(f"opponent_died fired in {deaths} windows, expected exactly 1")

    errors.extend(reconcile_against_wire(record.start_health, record.steps))
    return errors


def check_anchor(
    record: CycleRecord, anchor: Tuple[int, int], eps: float = _POS_EPS
) -> List[str]:
    """Assert both bots start on THIS pad's anchor (learner +0.5, dummy +3.5)."""
    ax, az = anchor
    errors: List[str] = []
    expectations = (
        ("learner", record.start_self_pos, (ax + 0.5, 64.0, az + 0.5)),
        ("dummy", record.start_opp_pos, (ax + 3.5, 64.0, az + 0.5)),
    )
    for name, actual, expected in expectations:
        if any(abs(a - e) > eps for a, e in zip(actual, expected)):
            errors.append(
                f"{name} start pos {tuple(round(v, 3) for v in actual)} is not at "
                f"the pad anchor expectation {expected} (eps {eps})"
            )
    return errors


# ---------------------------------------------------------------------------
# The live driver.
# ---------------------------------------------------------------------------


def _classify(info: Mapping[str, Any]) -> str:
    if info.get("lost"):
        return "loss"
    if info.get("won"):
        return "win"
    return "timeout"


def _run_cycle(
    env: MCPvPEnv,
    transport: RecordingTransport,
    cycle: int,
    seed: int,
    max_steps: int,
) -> CycleRecord:
    """Run one reset/kill cycle and record every window from the raw wire."""
    t0 = time.monotonic()
    env.reset(seed=seed)
    reset_ms = (time.monotonic() - t0) * 1000.0

    initial = transport.last_state
    if initial is None:
        raise BridgeError("no post-reset state message was recorded")

    record = CycleRecord(
        index=cycle,
        reset_ms=reset_ms,
        start_health=float(initial.opponent.health),
        start_self_pos=tuple(float(v) for v in initial.self_state.pos),
        start_opp_pos=tuple(float(v) for v in initial.opponent.pos),
        outcome="timeout",
    )

    cooldown = float(initial.self_state.attack_cooldown)
    info: Dict[str, Any] = {}
    for _ in range(max_steps):
        # Fully-cooled swings ONLY: a swing inside the cooldown deals reduced
        # damage and would break the 6,6,6,2 arithmetic by construction.
        action = Macro.ATTACK if cooldown >= _COOLDOWN_READY else Macro.IDLE
        _, _, done, info = env.step(int(action))

        state = transport.last_state
        if state is None:  # pragma: no cover - recv() always records on success
            raise BridgeError("step returned but no state message was recorded")
        events = info["events"]
        record.steps.append(
            StepRecord(
                action=int(action),
                damage_dealt=float(events["damage_dealt"]),
                opponent_died=bool(events["opponent_died"]),
                wire_health=float(state.opponent.health),
                attack_cooldown=float(state.self_state.attack_cooldown),
                tick=int(state.tick),
            )
        )
        cooldown = float(state.self_state.attack_cooldown)
        if done:
            record.outcome = _classify(info)
            break

    return record


def _format_cycle_line(record: CycleRecord, errors: Sequence[str]) -> str:
    hits = extract_hits(record.steps)
    hit_text = ", ".join(f"{amount:g}@w{w}" for w, amount in hits) or "none"
    total = sum(s.damage_dealt for s in record.steps)
    deaths = sum(1 for s in record.steps if s.opponent_died)
    return (
        f"[cycle {record.index:>2}] reset={record.reset_ms:6.0f}ms "
        f"start_hp={record.start_health:g} hits=[{hit_text}] total={total:g} "
        f"deaths={deaths} steps={len(record.steps)} outcome={record.outcome} "
        f"{'OK' if not errors else 'FAIL'}"
    )


def _minimum_steps_for(expected_hits: Sequence[float]) -> int:
    """Conservative per-cycle window budget a kill of this length needs.

    One full swing period for the FIRST hit (the attack meter can start
    mid-ramp — ``bridge/bot.js``'s ``_meterResetTick``), one more for each
    subsequent hit, and one window for the death to be reported. The observed
    healthy run is a little shorter than this; it is a floor that turns "why
    did every cycle time out?" into a message before the socket is opened, not
    a prediction of the window count.
    """
    return WINDOWS_PER_SWING * len(expected_hits) + 1


def run_probe(
    *,
    host: str,
    port: int,
    cycles: int,
    seed: int,
    max_steps: int,
    anchor: Tuple[int, int],
    armor: ArmorSet = TARGET_ARMOR,
    weapon_damage: float = IRON_SWORD_DAMAGE,
    log=print,
) -> bool:
    """Run the full probe. Returns True iff every cycle passed (AC8 PASS).

    ``armor`` / ``weapon_damage`` describe the TARGET's loadout and the
    learner's weapon; the asserted per-hit sequence is derived from them by
    :func:`expected_hit_sequence` rather than assumed. Defaults match what
    ``spawn_dummy_pad.mcfunction`` equips today.
    """
    if cycles < 1:
        raise ValueError(f"cycles must be >= 1, got {cycles}")

    expected_hits = expected_hit_sequence(
        weapon_damage=weapon_damage, armor=armor, target_health=FULL_HEALTH
    )
    minimum_steps = _minimum_steps_for(expected_hits)
    if max_steps < minimum_steps:
        raise ValueError(
            f"max_steps={max_steps} cannot fit the expected "
            f"{len(expected_hits)}-hit kill against {armor.name} armor: swings "
            f"are gated on a {IRON_SWORD_ATTACK_SPEED_TICKS:g}-tick cooldown "
            f"== {WINDOWS_PER_SWING} decision windows apart, so at least "
            f"{minimum_steps} windows are needed. Raise --max-steps."
        )

    # State the expectation BEFORE the first cycle, so a mismatch between the
    # probe's assumed loadout and what the datapack actually equips is visible
    # in the log rather than inferred from a red sequence assertion.
    log(
        f"[combat_probe] target loadout: {weapon_damage:g} raw weapon damage vs "
        f"{armor.name} armor ({armor.points} pts, toughness {armor.toughness:g})"
        f" -> {damage_after_absorb(weapon_damage, armor.points, armor.toughness):g}"
        f" per fully-cooled hit"
    )
    log(
        f"[combat_probe] expecting {len(expected_hits)} hits "
        f"{[round(v, 4) for v in expected_hits]} totalling "
        f"{math.fsum(expected_hits):g} (tolerance {_DAMAGE_TOL:g})"
    )

    transport = RecordingTransport(TcpBridgeClient(host=host, port=port))
    # ONE connection for the whole probe (the bridge serves exactly one client);
    # connect explicitly, then hand the transport to the env with
    # auto_connect=False so the ownership is unambiguous.
    transport.connect()

    all_ok = True
    reset_times: List[float] = []
    env = MCPvPEnv(transport=transport, auto_connect=False, max_episode_steps=max_steps)
    try:
        for cycle in range(cycles):
            record = _run_cycle(env, transport, cycle, seed + cycle, max_steps)
            errors = analyze_cycle(record, expected_hits) + check_anchor(
                record, anchor
            )
            reset_times.append(record.reset_ms)
            log(_format_cycle_line(record, errors))
            if errors:
                all_ok = False
                for err in errors:
                    log(f"    FAIL: {err}")
            # Full per-window trace so the evidence is verbatim, not summarized.
            log(
                "    windows: "
                + " ".join(
                    f"(w{i} a={s.action} d={s.damage_dealt:g} hp={s.wire_health:g}"
                    f"{' DIED' if s.opponent_died else ''})"
                    for i, s in enumerate(record.steps)
                )
            )
    finally:
        env.close()

    if reset_times:
        log(
            f"[resets] n={len(reset_times)} min={min(reset_times):.0f}ms "
            f"median={statistics.median(reset_times):.0f}ms "
            f"max={max(reset_times):.0f}ms"
        )
    log(f"[combat_probe] AC8 {'PASS' if all_ok else 'FAIL'} over {cycles} cycle(s)")
    return all_ok


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def _parse_anchor(raw: str) -> Tuple[int, int]:
    # Negative anchors are rejected DELIBERATELY: T10's padAnchor(i) only ever
    # produces non-negative anchors, and the datapack's textual "$(x).5" macro
    # concatenation is unsafe for negatives (it would yield anchor MINUS half a
    # block). Relax this only if pads ever legitimately go negative — together
    # with the macro plumbing, never alone.
    parts = raw.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f'--expect-anchor must be "<x>,<z>", got {raw!r}'
        )
    try:
        x, z = (int(p.strip()) for p in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f'--expect-anchor must be two integers "<x>,<z>", got {raw!r}'
        ) from exc
    if x < 0 or z < 0:
        raise argparse.ArgumentTypeError(
            f"--expect-anchor coordinates must be non-negative, got {raw!r}"
        )
    return (x, z)


def _positive_float(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected a number, got {raw!r}") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError(f"must be finite and > 0, got {raw!r}")
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="combat_probe",
        description=(
            "T8 deterministic combat gate (AC8): fully-cooled ATTACKs vs the "
            "stationary dummy, asserting the EXACT per-hit sequence derived "
            "from the target's loadout and reconciling every recorded value "
            "against the wire's privileged opponent health. Against the iron "
            "set the datapack equips today that sequence is 3.12 x6 then 1.28, "
            "not the bare-handed 6,6,6,2. Requires a LIVE Paper server + bridge."
        ),
    )
    parser.add_argument(
        "--cycles", type=int, default=10, help="reset/kill cycles to run (default: 10)"
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="bridge host")
    parser.add_argument("--port", type=int, default=5555, help="bridge TCP port")
    parser.add_argument("--seed", type=int, default=0, help="base reset seed")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=80,
        help=(
            "per-cycle decision-step cap (default: 80; the healthy armored kill "
            "takes ~26 windows -- 7 hits, 4 windows apart -- so hitting this cap "
            "is itself a failure). Rejected up front if it cannot fit the "
            "derived hit sequence."
        ),
    )
    parser.add_argument(
        "--target-armor",
        choices=sorted(ARMOR_SETS),
        default=TARGET_ARMOR.name,
        help=(
            "worn armor the DUMMY is expected to have, which sets the per-hit "
            f"expectation (default: {TARGET_ARMOR.name}, matching "
            "spawn_dummy_pad.mcfunction). Use 'none' for a bare-handed A/B or "
            "after a revert -- the derivation then reproduces the historical "
            "6,6,6,2 exactly."
        ),
    )
    parser.add_argument(
        "--weapon-damage",
        type=_positive_float,
        default=IRON_SWORD_DAMAGE,
        help=(
            "raw damage of one fully-charged hit from the LEARNER's weapon, "
            f"before the target's armor absorbs any of it (default: "
            f"{IRON_SWORD_DAMAGE:g}, an iron sword)"
        ),
    )
    parser.add_argument(
        "--expect-anchor",
        type=_parse_anchor,
        default=(0, 0),
        metavar="X,Z",
        help=(
            'pad anchor the bots must spawn on, as "<x>,<z>" (default 0,0). '
            "Must match the bridge's --pad-origin."
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        ok = run_probe(
            host=args.host,
            port=args.port,
            cycles=args.cycles,
            seed=args.seed,
            max_steps=args.max_steps,
            anchor=args.expect_anchor,
            armor=ARMOR_SETS[args.target_armor],
            weapon_damage=args.weapon_damage,
        )
    except ValueError as exc:
        # Loadout/step-budget misconfiguration, raised before the socket opens.
        # A 2 is right: the probe reached no verdict, which is never a pass.
        print(f"[combat_probe] ABORT (bad configuration): {exc}", file=sys.stderr)
        return 2
    except BridgeError as exc:
        print(f"[combat_probe] ABORT (bridge error): {exc}", file=sys.stderr)
        return 2
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
