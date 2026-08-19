"""Tests for the T17 self-play launch canary (AC12, TC38).

`scripts/canary_selfplay.sh` is the LAST gate before a 24-hour, 25-pad run. Its
value is entirely in its refusals, and a refusal that does not fire is worse
than no gate at all: it manufactures confidence. So every named refusal
condition is driven here over synthetic metrics, offline — no server, no
sockets, no bridge, no torch on the decision path.

HOW THIS REACHES THE CODE UNDER TEST. The decision logic is the `canary_verdict`
module the shell script writes to disk before it uses it. These tests extract
that module's source VERBATIM from between the script's `CANARY_VERDICT_PY`
heredoc sentinels and exec it, so what is tested is byte-identical to what the
operator runs — there is no second copy to drift.

WHAT IS PINNED, AND WHY EACH PIN EXISTS:

* **The healthy baseline is GREEN.** Every mutation test is measured against it.
  Without this anchor a check that refuses unconditionally would pass every
  refusal test and block every real run.
* **Each refusal fires on its own mutation.** One test per code, table-driven
  where the mutation is naturally isolated so an over-broad check (one that
  fires on unrelated evidence) is caught too.
* **Fail-closed on absent evidence.** A check whose input is missing REFUSES
  under its own code. The canary must never read a missing field as a healthy
  zero — that is exactly how the naive 15-minute canary would have passed.
* **The exit code is not the health signal.** `agent.train._main_multi_arena`
  returns `0 if passed_m2 else 1`, and `passed_m2` is the M2 gate against the
  STATIONARY dummy. A healthy self-play canary exits 1. A gate keyed on the exit
  code would refuse every good run, so `check_driver` reads the `[multi done]`
  teardown line instead and this file pins that.
* **The calibration constants are pinned to their sources, not to comments.**
  `ARMORED_HIT_DAMAGE` is asserted equal to
  `eval.combat_probe.damage_after_absorb(6.0, 15, 0.0)`, `ATTACK_MACRO` to
  `agent.actions.Macro.ATTACK`, `MAX_EPISODE_STEPS` to the contract constant.
  The armored number is 3.12 per fully-cooled hit (~7 hits to a 20 HP kill), NOT
  the 2.4/~9 an earlier revision of the plan carried.
* **Only fully-charged swings carry the loadout signature.** The charge scalar
  `0.2 + f^2 * 0.8` multiplies the RAW weapon damage BEFORE armor absorption,
  and absorption is not linear in the incoming damage, so it can never be
  applied over the post-armor 3.12. A 90%-charged iron hit through full iron
  lands `damage_after_absorb(6.0 * 0.848, 15, 0) == 2.553`, not the
  `3.12 * 0.848 == 2.646` a flat percentage predicts. `summarize_probe` must
  drop those from the armor evidence, and both the drop and the derivation are
  pinned against `eval.combat_probe` rather than restated.
* **A second attempt cannot inherit the first one's verdict.**
  `build_snapshot_opponents` LOAD-EXTENDS a pool it finds on disk and
  `SnapshotPool.load` restores both Elo series and the match counters, so
  re-running into the same `--run-name` would pre-satisfy `NO_NEW_SNAPSHOT`,
  `SNAPSHOT_UNCHANGED`, `RATED_ELO_EMPTY` and `DRAW_MAJORITY_TRAINING` with
  artifacts this attempt never produced. `STALE_POOL` refuses that, and the
  shell preflight refuses to start on it at all. Both are pinned.
* **A cooldown "drop" is never counted across an episode boundary**, where the
  meter legitimately resets to 1.0.
* **The script starts nothing.** It verifies the fleet and refuses with
  instructions; the operator owns Paper -> bridges. Pinned by asserting the
  script never invokes the fleet/JVM launchers.
* **Bridge ports are never connect-probed.** `BridgeServer` accepts exactly ONE
  TCP client and a second connection silently destroys the first — four outages
  in this project came from that. Only the Minecraft port (Paper is
  multi-client) may be connected to, and that is pinned by name.
"""

from __future__ import annotations

import copy
import json
import math
import os
import re
import shlex
import subprocess
import sys
import types
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT_PATH = os.path.join(REPO_ROOT, "scripts", "canary_selfplay.sh")


# ---------------------------------------------------------------------------
# Extract the decision logic from the shell script it ships inside.
# ---------------------------------------------------------------------------


def _script_lines() -> List[str]:
    with open(SCRIPT_PATH, "r", encoding="utf-8") as handle:
        return handle.read().splitlines()


def extract_heredoc(name: str) -> str:
    """Return the body of the ``<<'<name>'`` heredoc, verbatim.

    Asserts there is EXACTLY one opener and at least one terminator: two copies
    of the verdict module in one script would mean the tests and the operator
    could be running different code, which is the whole failure this extraction
    exists to prevent.
    """
    lines = _script_lines()
    opener = f"<<'{name}'"
    starts = [i for i, line in enumerate(lines) if line.rstrip().endswith(opener)]
    assert len(starts) == 1, f"expected exactly one {opener} in the script, got {starts}"
    start = starts[0]
    ends = [i for i in range(start + 1, len(lines)) if lines[i].strip() == name]
    assert ends, f"no {name} terminator after line {start + 1}"
    return "\n".join(lines[start + 1 : ends[0]]) + "\n"


def executable_shell() -> str:
    """The script with heredoc bodies, comments and quoted strings removed.

    What remains is the shell that actually RUNS. The distinction matters for
    :func:`test_the_script_starts_no_server`: the script names
    ``server/setup/start-pads.sh`` several times, but every one of those is
    instruction text printed for the operator. An occurrence in executable
    position would be the canary booting its own fleet — which it must not do,
    both because the operator owns the Paper -> bridges half of the boot order
    and because a self-booting canary is free to boot the fleet WRONG
    (knockback-immune), the exact failure it exists to detect.
    """
    lines = _script_lines()
    kept: List[str] = []
    terminator: Optional[str] = None
    for line in lines:
        if terminator is not None:
            if line.strip() == terminator:
                terminator = None
            continue
        match = re.search(r"<<-?'?([A-Za-z_][A-Za-z0-9_]*)'?\s*$", line)
        if match:
            terminator = match.group(1)
            line = line[: match.start()]
        kept.append(line)
    text = "\n".join(kept)

    # A character scanner, not a regex: several `die`/`warn` messages are
    # MULTI-LINE double-quoted strings, so quoting state has to carry across
    # newlines. A per-line regex leaves a dangling quote and a DOTALL one
    # swallows the rest of the file - which would make every "no launcher here"
    # assertion below vacuously true.
    out: List[str] = []
    quote: Optional[str] = None
    index = 0
    at_word_start = True
    while index < len(text):
        char = text[index]
        if quote is None:
            if char == "\\":
                index += 2
                at_word_start = False
                continue
            if char == "#" and at_word_start:
                newline = text.find("\n", index)
                index = len(text) if newline < 0 else newline
                continue
            if char in "'\"":
                quote = char
                out.append(" ")
                index += 1
                at_word_start = False
                continue
            out.append(char)
            at_word_start = char.isspace() or char in ";&|(){}"
            index += 1
            continue
        if quote == '"' and char == "\\":
            index += 2
            continue
        if char == quote:
            quote = None
        index += 1
    return "".join(out)


def _load_verdict_module() -> types.ModuleType:
    source = extract_heredoc("CANARY_VERDICT_PY")
    module = types.ModuleType("canary_verdict_under_test")
    module.__file__ = SCRIPT_PATH
    exec(compile(source, SCRIPT_PATH, "exec"), module.__dict__)
    return module


verdict_module = _load_verdict_module()

IDLE, APPROACH, STRAFE_L, ATTACK, JUMP = 0, 1, 3, 5, 6


# ---------------------------------------------------------------------------
# Synthetic evidence.
#
# The healthy fixtures are deliberately REALISTIC, so they double as a
# description of what a good canary run looks like: an armored kill is 20 HP,
# split 6 x 3.12 + 1.28, and the fighters miss between landed hits.
# ---------------------------------------------------------------------------

#: The 7-hit armored kill: six fully-cooled iron-sword hits through full iron
#: plus the remaining-health clamp. 6 * 3.12 + 1.28 == 20.0 == a fighter's HP.
ARMORED_KILL_SEQUENCE = (3.12, 3.12, 3.12, 3.12, 3.12, 3.12, 1.28)

#: When the synthetic run's driver started, in epoch seconds, and when its
#: newest snapshot was archived. Named rather than buried because `STALE_POOL`
#: compares exactly these two: the snapshot lands 1500 s into an 1800 s run, so
#: the healthy fixture PROVES this attempt produced it. A snapshot older than
#: the driver is the re-run trap, and it has its own test below.
DRIVER_STARTED_AT = 1_755_600_000.0
NEWEST_SNAPSHOT_MTIME = DRIVER_STARTED_AT + 1500.0


def make_episode(
    length: int = 200,
    outcome: str = "win",
    *,
    hit_damage: Sequence[float] = ARMORED_KILL_SEQUENCE,
    taken_damage: Optional[Sequence[float]] = None,
    learner_macros: Sequence[int] = (ATTACK, APPROACH, STRAFE_L, JUMP),
    opponent_macros: Sequence[int] = (ATTACK, APPROACH, STRAFE_L, IDLE),
    learner_cooldowns: Sequence[float] = (1.0, 0.0, 0.4, 0.8),
    opponent_cooldowns: Sequence[float] = (1.0, 0.0, 0.4, 0.8),
    opponent_speed: float = 0.18,
    opponent_moves: bool = True,
) -> Dict[str, Any]:
    """One synthetic probe episode.

    The four-phase cycle mirrors the real swing cadence: an iron sword recharges
    in 12.5 ticks and a decision window is 4 ticks (ACTION_REPEAT), so a fighter
    that swings the instant it is cooled attacks roughly every fourth window and
    its meter climbs 0.0 -> 0.4 -> 0.8 -> 1.0 in between.
    """
    taken = list(taken_damage) if taken_damage is not None else list(hit_damage)
    dealt = list(hit_damage)
    windows: List[Dict[str, Any]] = []
    attack_index = 0
    x, z = 100.0, 200.0
    for step in range(length):
        phase = step % 4
        record: Dict[str, Any] = {
            "a": int(learner_macros[phase % len(learner_macros)]),
            "oa": int(opponent_macros[phase % len(opponent_macros)]),
            "cd": float(learner_cooldowns[phase % len(learner_cooldowns)]),
            "ocd": float(opponent_cooldowns[phase % len(opponent_cooldowns)]),
            "osp": float(opponent_speed if phase % 2 else 0.0),
            "opx": x,
            "opz": z,
            "dd": 0.0,
            "dt": 0.0,
        }
        if phase == 0:
            if attack_index < len(dealt):
                record["dd"] = float(dealt[attack_index])
            if attack_index < len(taken):
                record["dt"] = float(taken[attack_index])
            attack_index += 1
        if opponent_moves:
            x += 0.05
            z -= 0.03
        windows.append(record)
    return {"length": length, "outcome": outcome, "windows": windows}


def make_raw_probe(
    episodes: Optional[Sequence[Mapping[str, Any]]] = None,
    *,
    ok: bool = True,
    error: Optional[str] = None,
    max_episode_steps: int = 600,
    learner_epsilon: float = 0.05,
    opponent_epsilon: float = 0.02,
) -> Dict[str, Any]:
    """A raw probe document as the live probe writes it."""
    if episodes is None:
        episodes = [
            make_episode(200, "win"),
            make_episode(180, "loss"),
            make_episode(220, "win"),
        ]
    return {
        "ok": ok,
        "error": error,
        "checkpoint": "runs/m4_selfplay_canary.pt",
        "learner_epsilon": learner_epsilon,
        "opponent_epsilon": opponent_epsilon,
        "max_episode_steps": max_episode_steps,
        "episodes": [dict(e) for e in episodes],
    }


def make_evidence(
    probe_post: Optional[Mapping[str, Any]] = None, **overrides: Any
) -> Dict[str, Any]:
    """A healthy evidence document — the anchor every mutation is measured against."""
    summary = (
        dict(probe_post)
        if probe_post is not None
        else verdict_module.summarize_probe(make_raw_probe())
    )
    evidence: Dict[str, Any] = {
        "evidence_version": verdict_module.EVIDENCE_VERSION,
        "evidence_path": "runs/m4_selfplay_canary/canary/evidence.json",
        "run_name": "m4_selfplay_canary",
        "arenas": 4,
        "min_replay": 2000,
        "min_replay_production": 25000,
        "max_grad_steps": 1200,
        "warm_start": "/abs/runs/m4.best.pt",
        "warm_start_sha256": "0" * 64,
        "driver_wall_seconds": 1800.0,
        "driver_started_at": DRIVER_STARTED_AT,
        "deadline_seconds": 5400,
        "driver": {
            "completed": True,
            # 1, not 0: a healthy self-play canary exits nonzero because
            # passed_m2 is the M2 gate against the stationary dummy.
            "exit_code": 1,
            "deadline_hit": False,
            "stop_reason": "max_grad_steps",
            "episodes": 214,
            "grad_steps": 1204,
            "checkpoints_saved": 3,
            "log_path": "runs/m4_selfplay_canary/canary/driver.log",
        },
        "pool": {
            "ok": True,
            "error": None,
            "directory": "runs/m4_selfplay_canary/snapshots",
            "sampling": "pfsp",
            "size": 5,
            "snapshot_ids": [0, 1, 2, 3, 4],
            "pinned_ids": [0, 2, 3],
            "grad_steps": {"0": 0, "1": 300, "2": 600, "3": 900, "4": 1200},
            "matches_scored": 214,
            "draws_scored": 12,
            "rated_matches": 12,
            "learner_elo_rated": 1014.2,
            "learner_elo_online": 1003.5,
            "pfsp_weights": {"0": 0.2, "1": 0.2, "2": 0.2, "3": 0.2, "4": 0.2},
            "newest_snapshot_id": 4,
            "newest_snapshot_mtime": NEWEST_SNAPSHOT_MTIME,
            "newest_vs_snapshot0_max_abs_delta": 0.0143,
            "newest_vs_snapshot0_error": None,
            # The newest snapshot must also differ from the one BEFORE it: a
            # pool of snapshots 1..N that are clones of each other but differ
            # from the seed is the same "one policy under many ids" failure,
            # one archive cycle later.
            "newest_vs_second_newest_max_abs_delta": 0.0071,
            "newest_vs_second_newest_error": None,
        },
        # Read out of driver.log, NOT summary.json: the multi-arena path's only
        # logger.summary() call writes selfplay_log_row(pool), which carries no
        # win_rate and no mean_episode_length at all. `opponent` is the FIXED
        # SCRIPTED yardstick EvalOpponentDriver.name reports on a self-play run
        # ("scripted_" + cfg.eval_opponent_preset) - NOT an eps=0 self-play
        # match, which is the rated gauntlet and reports Elo, not lengths.
        "eval": {
            "ran": True,
            "source": "runs/m4_selfplay_canary/canary/driver.log",
            "win_rate": 0.5,
            "mean_episode_length": 201.3,
            "aim_while_invisible": 0.012,
            "opponent": "scripted_mixed",
            "grad_step": 1200,
            "episodes_per_cycle": 6,
        },
        "epsilon": {"schedule": 0.05, "mean": 0.021},
        "checkpoint": {
            "final": {
                "path": "runs/m4_selfplay_canary.pt",
                "loadable": True,
                "error": None,
            },
            "snapshot": {
                "path": "runs/m4_selfplay_canary/snapshots/snap_4.pt",
                "loadable": True,
                "error": None,
            },
        },
        "probe_pre": verdict_module.summarize_probe(
            make_raw_probe(episodes=[make_episode(150, "win")])
        ),
        "probe_post": summary,
    }
    evidence.update(overrides)
    return evidence


def refusal_codes(evidence: Mapping[str, Any]) -> Set[str]:
    """The set of codes blocking this evidence document."""
    verdict = verdict_module.evaluate_canary(evidence)
    return {check.code for check in verdict.refusals}


def mutate(evidence: Mapping[str, Any], path: str, value: Any) -> Dict[str, Any]:
    """Deep-copy ``evidence`` and set one dotted path to ``value``."""
    out = copy.deepcopy(dict(evidence))
    keys = path.split(".")
    cursor: Any = out
    for key in keys[:-1]:
        cursor = cursor[key]
    cursor[keys[-1]] = value
    return out


# ===========================================================================
# The anchor: a healthy run is GREEN.
# ===========================================================================


def test_healthy_evidence_is_green() -> None:
    """No refusal fires on a healthy run.

    Every mutation test below is meaningless without this: a check that refuses
    unconditionally would satisfy its own refusal test AND block every real
    launch, and only this assertion tells the two apart.
    """
    verdict = verdict_module.evaluate_canary(make_evidence())
    assert verdict.refusals == [], [
        (c.code, c.why) for c in verdict.refusals
    ]
    assert verdict.ok is True
    # Every check ran; none was silently skipped into existence-by-omission.
    assert len(verdict.checks) == len({c.code for c in verdict.checks})


def test_every_named_refusal_code_is_reachable() -> None:
    """The gate emits a check for each code the report can print.

    Guards against a check being deleted (or renamed) while its test still
    passes because the mutation happens to trip a different code.
    """
    codes = {c.code for c in verdict_module.evaluate_canary(make_evidence()).checks}
    assert codes == {
        "ACTION_COLLAPSE_LEARNER",
        "ACTION_COLLAPSE_OPPONENT",
        "CHECKPOINT_UNLOADABLE",
        "COOLDOWN_DISAGREEMENT",
        "COOLDOWN_METER_STUCK",
        "DRAW_MAJORITY_PROBE",
        "DRAW_MAJORITY_TRAINING",
        "DRIVER_FAILED",
        "MISSING_ARMOR",
        "NO_DAMAGE_DEALT",
        "NO_DAMAGE_TAKEN",
        "NO_NEW_SNAPSHOT",
        "OPPONENT_FROZEN",
        "PFSP_INVALID",
        "POOL_UNREADABLE",
        "PROBE_FAILED",
        "RATED_ELO_EMPTY",
        "SNAPSHOT_UNCHANGED",
        "STALE_POOL",
        "ZERO_GRAD_STEPS",
    }


# ===========================================================================
# Calibration constants, pinned to their sources rather than to comments.
# ===========================================================================


def test_armored_hit_damage_matches_the_combat_rules_derivation() -> None:
    """3.12 per fully-cooled hit, ~7 hits to a kill — derived, not remembered.

    Paper's ``CombatRules.getDamageAfterAbsorb`` is the source of truth and
    ``eval.combat_probe`` implements it. An earlier revision of the plan said
    2.4/~9, which is the flat 4%-per-point model and is wrong.
    """
    from eval.combat_probe import IRON_ARMOR, IRON_SWORD_DAMAGE, damage_after_absorb

    derived = damage_after_absorb(
        IRON_SWORD_DAMAGE, IRON_ARMOR.points, IRON_ARMOR.toughness
    )
    assert verdict_module.ARMORED_HIT_DAMAGE == pytest.approx(derived)
    assert verdict_module.ARMORED_HIT_DAMAGE == pytest.approx(3.12)
    assert verdict_module.BARE_HIT_DAMAGE == pytest.approx(IRON_SWORD_DAMAGE)
    # ~7 hits, not ~9: 20 / 3.12 == 6.41.
    assert math.ceil(20.0 / verdict_module.ARMORED_HIT_DAMAGE) == 7
    # A fist through the same armor — why arming the dummy was blocking.
    assert verdict_module.UNARMED_THROUGH_IRON_DAMAGE == pytest.approx(
        damage_after_absorb(1.0, IRON_ARMOR.points, IRON_ARMOR.toughness), abs=5e-3
    )


def test_action_and_horizon_constants_match_the_repo() -> None:
    """ATTACK's index and the 600-step cap are pinned to their real definitions."""
    from agent.actions import Macro
    from agent.contract_config import MAX_EPISODE_STEPS

    assert verdict_module.ATTACK_MACRO == int(Macro.ATTACK)
    assert verdict_module.MAX_EPISODE_STEPS == MAX_EPISODE_STEPS == 600


def test_the_armored_kill_sequence_sums_to_a_full_health_bar() -> None:
    """The synthetic fixture's damage profile is the real one, at 3.12."""
    assert sum(ARMORED_KILL_SEQUENCE) == pytest.approx(20.0)
    assert ARMORED_KILL_SEQUENCE[0] == pytest.approx(verdict_module.ARMORED_HIT_DAMAGE)


# ===========================================================================
# summarize_probe — all the arithmetic the live probe deliberately does not do.
# ===========================================================================


def test_summarize_probe_aggregates_a_healthy_run() -> None:
    summary = verdict_module.summarize_probe(make_raw_probe())
    assert summary["ok"] is True
    assert summary["episodes"] == 3
    assert summary["windows"] == 200 + 180 + 220
    assert summary["mean_episode_length"] == pytest.approx(200.0)
    assert summary["cap_hits"] == 0
    assert summary["cap_hit_rate"] == pytest.approx(0.0)
    # One full 20 HP kill's worth of damage per episode, both directions.
    assert summary["damage_dealt_per_episode"] == pytest.approx(20.0)
    assert summary["damage_taken_per_episode"] == pytest.approx(20.0)
    # Six of the seven landed hits carry the iron signature; the seventh is the
    # remaining-health clamp (1.28) and is reported as "other", not forced.
    assert summary["hits_dealt"]["count"] == 21
    assert summary["hits_dealt"]["iron_like"] == 18
    assert summary["hits_dealt"]["bare_like"] == 0
    assert summary["hits_dealt"]["other"] == 3
    assert summary["learner"]["distinct_macros"] == 4
    assert summary["learner"]["top_macro_share"] == pytest.approx(0.25, abs=0.01)
    assert summary["learner"]["cooldown_drop_rate"] == pytest.approx(1.0)
    assert summary["opponent"]["cooldown_drop_rate"] == pytest.approx(1.0)
    assert summary["opponent"]["max_speed"] == pytest.approx(0.18)
    assert summary["opponent"]["path_length"] > 1.0


def test_summarize_probe_reports_a_failed_probe_rather_than_inventing_zeros() -> None:
    """Fail-closed: a probe that raised is not a probe that measured nothing."""
    for probe in (
        None,
        {},
        {"ok": False, "error": "BridgeError: connection reset"},
        {"ok": True, "episodes": []},
    ):
        summary = verdict_module.summarize_probe(probe)
        assert summary["ok"] is False
        assert summary["error"]


def test_a_partial_swing_is_priced_before_armor_not_after() -> None:
    """The charge scalar multiplies RAW damage, then armor absorbs the result.

    ``0.2 + f^2 * 0.8`` is applied by ``Player.attack`` BEFORE
    ``getDamageAfterArmorAbsorb`` runs, and absorption is not linear in the
    incoming damage — so the scalar can never be applied over the post-armor
    3.12. Doing that is the flat-percentage error ``damage_after_absorb``'s own
    docstring calls out, and it is worth ~0.09 HP here.

    Derived from the repo's implementation, not restated: at f = 0.90 a swing
    lands 2.553 through full iron, NOT 3.12 * 0.848 == 2.646.
    """
    from eval.combat_probe import (
        IRON_ARMOR,
        IRON_SWORD_DAMAGE,
        damage_after_absorb,
        damage_for_swing_charge,
    )

    landed = damage_for_swing_charge(0.9)
    assert landed == pytest.approx(
        damage_after_absorb(
            IRON_SWORD_DAMAGE * (0.2 + 0.9 * 0.9 * 0.8),
            IRON_ARMOR.points,
            IRON_ARMOR.toughness,
        )
    )
    assert landed == pytest.approx(2.553, abs=5e-4)
    # The number a flat percentage over the post-armor figure would predict,
    # and which four sites of this gate used to carry.
    assert verdict_module.ARMORED_HIT_DAMAGE * 0.848 == pytest.approx(2.646, abs=5e-4)
    assert landed != pytest.approx(verdict_module.ARMORED_HIT_DAMAGE * 0.848, abs=1e-3)


def test_only_fully_charged_swings_carry_the_loadout_signature() -> None:
    """A partial swing is NOT evidence about armor.

    At 90% charge an iron hit through full iron lands 2.553 (see
    :func:`test_a_partial_swing_is_priced_before_armor_not_after`), which is
    0.567 below 3.12 — well outside the +/-0.30 iron band. Counting it would
    read as "not 3.12" and refuse a perfectly geared fleet.
    """
    from eval.combat_probe import damage_for_swing_charge

    assert abs(damage_for_swing_charge(0.9) - verdict_module.ARMORED_HIT_DAMAGE) > 0.30
    episode = make_episode(40, "win")
    for record in episode["windows"]:
        if record["a"] == ATTACK:
            record["cd"] = 0.9  # cooled, but not fully
    summary = verdict_module.summarize_probe(make_raw_probe(episodes=[episode]))
    assert summary["hits_dealt"]["count"] == 0
    assert summary["damage_dealt_per_episode"] > 0  # the damage still counts


def test_a_cooldown_drop_is_never_counted_across_an_episode_boundary() -> None:
    """The meter resets to 1.0 at every reset; that reset is not a swing response.

    Two one-window episodes whose only window is an ATTACK must contribute ZERO
    ATTACK windows to the drop-rate denominator — comparing across the boundary
    would compare the end of one episode with the start of the next.
    """
    episodes = [
        {
            "length": 1,
            "outcome": "timeout",
            "windows": [
                {
                    "a": ATTACK,
                    "oa": ATTACK,
                    "cd": 1.0,
                    "ocd": 1.0,
                    "dd": 0.0,
                    "dt": 0.0,
                    "osp": 0.2,
                    "opx": 0.0,
                    "opz": 0.0,
                }
            ],
        }
        for _ in range(2)
    ]
    summary = verdict_module.summarize_probe(make_raw_probe(episodes=episodes))
    assert summary["learner"]["attack_windows"] == 0
    assert summary["learner"]["cooldown_drop_rate"] is None


def test_cap_length_episodes_are_counted_as_draws() -> None:
    raw = make_raw_probe(
        episodes=[make_episode(600, "timeout"), make_episode(200, "win")]
    )
    summary = verdict_module.summarize_probe(raw)
    assert summary["cap_hits"] == 1
    assert summary["cap_hit_rate"] == pytest.approx(0.5)


# ===========================================================================
# One test per refusal condition. Each proves the condition BLOCKS.
# ===========================================================================


def test_zero_grad_steps_blocks() -> None:
    """THE headline check: a canary that never learned proved nothing.

    This is exactly the hole in the naive canary — 2 arenas for 15 minutes
    collects ~8,781 transitions against a 25,000 warm-up floor, so `learn()` is
    a no-op the entire time and the run exits green on unchanged warm-start
    weights.
    """
    codes = refusal_codes(mutate(make_evidence(), "driver.grad_steps", 0))
    assert codes == {"ZERO_GRAD_STEPS"}


def test_missing_grad_step_count_blocks() -> None:
    """Fail-closed: an unreadable count is not a zero and not a pass."""
    assert "ZERO_GRAD_STEPS" in refusal_codes(
        mutate(make_evidence(), "driver.grad_steps", None)
    )


def test_driver_that_never_finished_blocks() -> None:
    codes = refusal_codes(mutate(make_evidence(), "driver.completed", False))
    assert codes == {"DRIVER_FAILED"}


def test_wall_clock_deadline_blocks() -> None:
    codes = refusal_codes(mutate(make_evidence(), "driver.deadline_hit", True))
    assert codes == {"DRIVER_FAILED"}


def test_a_nonzero_driver_exit_code_alone_does_not_block() -> None:
    """A healthy self-play canary EXITS 1 and that must not refuse.

    ``_main_multi_arena`` returns ``0 if passed_m2 else 1`` and ``passed_m2`` is
    the M2 gate (win_rate >= 0.95 vs the STATIONARY dummy), which a run fighting
    a past self does not clear. Gating on the exit code would refuse every good
    run; this pins that it does not.
    """
    for exit_code in (0, 1, 2, 130):
        evidence = mutate(make_evidence(), "driver.exit_code", exit_code)
        assert refusal_codes(evidence) == set(), exit_code


def test_unreadable_pool_blocks() -> None:
    evidence = mutate(
        make_evidence(), "pool", {"ok": False, "error": "FileNotFoundError: pool.json"}
    )
    assert "POOL_UNREADABLE" in refusal_codes(evidence)


def test_pool_that_never_grew_blocks() -> None:
    """The T18 failure: the archive cadence never fired, and nothing raises.

    A 24-hour run would then fight the frozen warm start every single episode
    while ``selfplay/pool_size`` reads 1 all night.
    """
    evidence = make_evidence()
    evidence["pool"].update(
        size=1,
        snapshot_ids=[0],
        pinned_ids=[0],
        pfsp_weights={"0": 1.0},
        newest_snapshot_id=0,
        grad_steps={"0": 0},
        newest_vs_snapshot0_max_abs_delta=None,
        newest_vs_snapshot0_error="the pool holds only snapshot 0",
        newest_vs_second_newest_max_abs_delta=None,
        newest_vs_second_newest_error="the pool holds only snapshot 0",
    )
    assert "NO_NEW_SNAPSHOT" in refusal_codes(evidence)


def test_snapshot_identical_to_snapshot_zero_blocks() -> None:
    """The archive hook fired from the wrong source: two ids, one policy."""
    codes = refusal_codes(
        mutate(make_evidence(), "pool.newest_vs_snapshot0_max_abs_delta", 0.0)
    )
    assert codes == {"SNAPSHOT_UNCHANGED"}


def test_undiffable_snapshot_blocks() -> None:
    """Fail-closed: a snapshot that cannot be diffed is not a proven new version."""
    evidence = mutate(
        make_evidence(),
        "pool.newest_vs_snapshot0_error",
        "snapshot 0 and the newest snapshot have different parameter sets",
    )
    assert "SNAPSHOT_UNCHANGED" in refusal_codes(evidence)


@pytest.mark.parametrize(
    "weights",
    [
        pytest.param({}, id="empty"),
        pytest.param({"0": float("nan"), "1": 1.0}, id="nan"),
        pytest.param({"0": float("inf"), "1": 1.0}, id="inf"),
        pytest.param({"0": -0.5, "1": 1.5}, id="negative"),
        pytest.param({"0": 0.3, "1": 0.3}, id="not-normalized"),
        pytest.param({"0": 1.0}, id="missing-a-live-snapshot"),
    ],
)
def test_invalid_pfsp_probabilities_block(weights: Dict[str, float]) -> None:
    """``SnapshotPool.pfsp_weights`` is contracted finite, non-negative, normalized.

    This canary is that contract's only live check; a violation means opponent
    sampling is undefined for the whole night.
    """
    codes = refusal_codes(mutate(make_evidence(), "pool.pfsp_weights", weights))
    assert codes == {"PFSP_INVALID"}


def test_empty_rated_elo_blocks() -> None:
    """AC7's series with no data looks exactly like AC7's series gone flat.

    ``elo/learner_rated`` only moves on matches where BOTH sides sat at eps=0,
    which happens inside an eval cycle. Zero rated matches means the rated eval
    is not wired and AC7 would produce nothing overnight.
    """
    codes = refusal_codes(mutate(make_evidence(), "pool.rated_matches", 0))
    assert codes == {"RATED_ELO_EMPTY"}


def test_draw_majority_in_training_blocks() -> None:
    """The degenerate mutual-stalling equilibrium: PFSP flat, Elo pinned."""
    evidence = mutate(make_evidence(), "pool.draws_scored", 130)
    evidence["pool"]["matches_scored"] = 200
    assert refusal_codes(evidence) == {"DRAW_MAJORITY_TRAINING"}


def test_no_scored_training_match_blocks() -> None:
    """Zero scored matches means the self-play driver never reached the collectors."""
    evidence = mutate(make_evidence(), "pool.matches_scored", 0)
    assert "DRAW_MAJORITY_TRAINING" in refusal_codes(evidence)


@pytest.mark.parametrize("entry", ["final", "snapshot"])
def test_unloadable_checkpoint_blocks(entry: str) -> None:
    """Both the shipped file and the newest snapshot must load through the
    SHARED ``eval.evaluate._load_drqn`` — otherwise it is found on demo day."""
    evidence = make_evidence()
    evidence["checkpoint"][entry] = {
        "path": "runs/whatever.pt",
        "loadable": False,
        "error": "RuntimeError: Missing key(s) in state_dict",
    }
    assert refusal_codes(evidence) == {"CHECKPOINT_UNLOADABLE"}


def test_failed_probe_blocks() -> None:
    """Absent first-hand evidence is a refusal, never a pass."""
    evidence = mutate(
        make_evidence(),
        "probe_post",
        verdict_module.summarize_probe({"ok": False, "error": "BridgeError"}),
    )
    assert refusal_codes(evidence) == {"PROBE_FAILED"}


def test_a_probe_too_short_to_mean_anything_blocks() -> None:
    summary = verdict_module.summarize_probe(
        make_raw_probe(episodes=[make_episode(20, "win")])
    )
    assert refusal_codes(make_evidence(summary)) == {"PROBE_FAILED"}


def test_missing_armor_blocks() -> None:
    """Full-charge hits landing at 6.0 mean a fighter is wearing nothing.

    Armor is invisible to the reset gate's inventory read (``items()`` covers
    slots 9-44; armor sits in 5-8), so measured damage is the honest check.
    """
    episodes = [
        make_episode(200, "win", hit_damage=(6.0, 6.0, 6.0, 2.0)) for _ in range(3)
    ]
    summary = verdict_module.summarize_probe(make_raw_probe(episodes=episodes))
    assert summary["hits_dealt"]["bare_like"] > summary["hits_dealt"]["iron_like"]
    assert "MISSING_ARMOR" in refusal_codes(make_evidence(summary))


def test_an_unarmed_fighter_blocks() -> None:
    """0.42 per fully-cooled swing is a FIST through full iron — ~48 hits a kill."""
    fist = tuple([verdict_module.UNARMED_THROUGH_IRON_DAMAGE] * 20)
    episodes = [make_episode(200, "timeout", hit_damage=fist) for _ in range(3)]
    summary = verdict_module.summarize_probe(make_raw_probe(episodes=episodes))
    assert "MISSING_ARMOR" in refusal_codes(make_evidence(summary))


def test_too_few_full_charge_hits_blocks_in_strict_mode() -> None:
    """Fail-closed: a canary that cannot SEE the loadout must not clear a launch."""
    episodes = [make_episode(200, "win", hit_damage=(3.12,)) for _ in range(2)]
    summary = verdict_module.summarize_probe(make_raw_probe(episodes=episodes))
    assert summary["hits_dealt"]["count"] == 2
    assert "MISSING_ARMOR" in refusal_codes(make_evidence(summary))


def test_no_damage_dealt_blocks() -> None:
    """A dead damage channel, or an unarmed learner. This project has already
    shipped a silently dead ``damage_dealt`` for its entire life."""
    episodes = [make_episode(200, "timeout", hit_damage=()) for _ in range(3)]
    summary = verdict_module.summarize_probe(make_raw_probe(episodes=episodes))
    assert summary["damage_dealt_per_episode"] == pytest.approx(0.0)
    assert "NO_DAMAGE_DEALT" in refusal_codes(make_evidence(summary))


def test_no_damage_taken_blocks() -> None:
    """The opponent lands nothing: it is unarmed, or the channel is one-sided."""
    episodes = [
        make_episode(200, "win", taken_damage=()) for _ in range(3)
    ]
    summary = verdict_module.summarize_probe(make_raw_probe(episodes=episodes))
    assert summary["damage_taken_per_episode"] == pytest.approx(0.0)
    assert "NO_DAMAGE_TAKEN" in refusal_codes(make_evidence(summary))


def test_near_zero_damage_blocks_even_when_nonzero() -> None:
    """The floor is one clean armored hit per episode, not "any damage at all"."""
    episodes = [make_episode(200, "timeout", hit_damage=(0.5,)) for _ in range(3)]
    summary = verdict_module.summarize_probe(make_raw_probe(episodes=episodes))
    assert 0.0 < summary["damage_dealt_per_episode"] < verdict_module.ARMORED_HIT_DAMAGE
    codes = refusal_codes(make_evidence(summary))
    assert {"NO_DAMAGE_DEALT", "NO_DAMAGE_TAKEN"} <= codes


def test_draw_majority_in_the_probe_blocks() -> None:
    """Most episodes ending at the 600-step cap is stalling, not fighting."""
    episodes = [
        make_episode(600, "timeout"),
        make_episode(600, "timeout"),
        make_episode(200, "win"),
    ]
    summary = verdict_module.summarize_probe(make_raw_probe(episodes=episodes))
    assert refusal_codes(make_evidence(summary)) == {"DRAW_MAJORITY_PROBE"}


@pytest.mark.parametrize(
    "seat,kwargs,code",
    [
        ("learner", {"learner_macros": (ATTACK,)}, "ACTION_COLLAPSE_LEARNER"),
        ("opponent", {"opponent_macros": (ATTACK,)}, "ACTION_COLLAPSE_OPPONENT"),
    ],
)
def test_collapsed_action_diversity_blocks(
    seat: str, kwargs: Dict[str, Any], code: str
) -> None:
    """One macro dominating makes every win rate and Elo number meaningless."""
    episodes = [make_episode(200, "win", **kwargs) for _ in range(3)]
    summary = verdict_module.summarize_probe(make_raw_probe(episodes=episodes))
    assert summary[seat]["top_macro_share"] == pytest.approx(1.0)
    assert code in refusal_codes(make_evidence(summary))


def test_a_pinned_cooldown_meter_blocks() -> None:
    """A meter that never moves means the swing gate is mis-modelled."""
    episodes = [
        make_episode(200, "win", opponent_cooldowns=(1.0, 1.0, 1.0, 1.0))
        for _ in range(3)
    ]
    summary = verdict_module.summarize_probe(make_raw_probe(episodes=episodes))
    assert summary["opponent"]["cooldown_range"] == pytest.approx(0.0)
    assert "COOLDOWN_METER_STUCK" in refusal_codes(make_evidence(summary))


def test_an_out_of_range_cooldown_reading_blocks() -> None:
    episodes = [
        make_episode(200, "win", learner_cooldowns=(1.0, 0.0, 0.4, 1.7))
        for _ in range(3)
    ]
    summary = verdict_module.summarize_probe(make_raw_probe(episodes=episodes))
    assert summary["learner"]["cooldown_out_of_range"] > 0
    assert "COOLDOWN_METER_STUCK" in refusal_codes(make_evidence(summary))


def test_cooldown_disagreement_between_the_two_seats_blocks() -> None:
    """Both fighters hold an iron sword, so their meters must respond alike.

    The opponent's meter here still MOVES (so it is not "stuck") and still reads
    fully charged at the swing (so the armor evidence is intact) — it simply
    never drops afterwards, which is exactly what a shadow meter that has
    stopped tracking the bridge's swing gate looks like.
    """
    episodes = [
        make_episode(200, "win", opponent_cooldowns=(1.0, 1.0, 0.6, 0.9))
        for _ in range(3)
    ]
    summary = verdict_module.summarize_probe(make_raw_probe(episodes=episodes))
    assert summary["learner"]["cooldown_drop_rate"] == pytest.approx(1.0)
    assert summary["opponent"]["cooldown_drop_rate"] == pytest.approx(0.0)
    assert refusal_codes(make_evidence(summary)) == {"COOLDOWN_DISAGREEMENT"}


def test_a_seat_that_never_swings_blocks_in_strict_mode() -> None:
    """Fail-closed: the seats must be compared on real swings, not assumed equal."""
    episodes = [
        make_episode(200, "timeout", opponent_macros=(APPROACH,), taken_damage=())
        for _ in range(3)
    ]
    summary = verdict_module.summarize_probe(make_raw_probe(episodes=episodes))
    assert summary["opponent"]["attack_windows"] == 0
    codes = refusal_codes(make_evidence(summary))
    assert "COOLDOWN_DISAGREEMENT" in codes
    assert "COOLDOWN_METER_STUCK" in codes


def test_a_frozen_opponent_blocks() -> None:
    """DUMMY_KNOCKBACK_IMMUNE=false was not set, or movement_speed is pinned again."""
    episodes = [
        make_episode(200, "timeout", opponent_speed=0.0, opponent_moves=False)
        for _ in range(3)
    ]
    summary = verdict_module.summarize_probe(make_raw_probe(episodes=episodes))
    assert summary["opponent"]["max_speed"] == pytest.approx(0.0)
    assert summary["opponent"]["path_length"] == pytest.approx(0.0)
    assert "OPPONENT_FROZEN" in refusal_codes(make_evidence(summary))


def test_a_broken_velocity_channel_alone_does_not_refuse() -> None:
    """OPPONENT_FROZEN needs BOTH signals dead, so it cannot false-refuse.

    mineflayer's non-self entity velocity is unreliable on pinned 1.21.1 — which
    is exactly why the opponent's velocity is read from its OWN connection. A
    velocity channel that reports zero while the bot visibly walks must not
    block the run; the walked path is the corroboration.
    """
    episodes = [
        make_episode(200, "win", opponent_speed=0.0, opponent_moves=True)
        for _ in range(3)
    ]
    summary = verdict_module.summarize_probe(make_raw_probe(episodes=episodes))
    assert summary["opponent"]["max_speed"] == pytest.approx(0.0)
    assert summary["opponent"]["path_length"] > 1.0
    assert "OPPONENT_FROZEN" not in refusal_codes(make_evidence(summary))


def test_an_unrecognized_evidence_version_blocks() -> None:
    """Reading an unknown layout would let absent fields pass as healthy zeros."""
    assert refusal_codes(mutate(make_evidence(), "evidence_version", 99)) == {
        "DRIVER_FAILED"
    }


# ===========================================================================
# The fail-fast pre-probe: defers what one episode cannot support, and NOTHING
# else.
# ===========================================================================


def test_the_pre_probe_defers_the_volume_dependent_checks() -> None:
    """One episode cannot support a draw rate or an action histogram.

    Refusing on them would make the fail-fast pre-probe reject healthy fleets;
    passing them silently would be the manufactured confidence this whole file
    exists to prevent. They are DEFERRED, and the report says so.
    """
    summary = verdict_module.summarize_probe(
        make_raw_probe(episodes=[make_episode(600, "timeout", learner_macros=(ATTACK,))])
    )
    deferred = {
        c.code: c
        for c in verdict_module.evaluate_gear(summary, strict=False)
        if c.passed
    }
    for code in (
        "DRAW_MAJORITY_PROBE",
        "ACTION_COLLAPSE_LEARNER",
        "ACTION_COLLAPSE_OPPONENT",
    ):
        assert code in deferred
        assert "deferred" in deferred[code].detail
    # ... and the same evidence in STRICT mode does refuse on all three.
    strict = {c.code for c in verdict_module.evaluate_gear(summary) if not c.passed}
    assert {"DRAW_MAJORITY_PROBE", "ACTION_COLLAPSE_LEARNER"} <= strict


def test_the_pre_probe_still_catches_a_mis_geared_fleet() -> None:
    """Fail-fast has to be worth running: gear defects block before the budget."""
    summary = verdict_module.summarize_probe(
        make_raw_probe(
            episodes=[make_episode(200, "win", hit_damage=(6.0,) * 8)]
        )
    )
    codes = {c.code for c in verdict_module.evaluate_gear(summary, strict=False) if not c.passed}
    assert "MISSING_ARMOR" in codes


def test_the_pre_probe_still_catches_a_dead_damage_channel() -> None:
    summary = verdict_module.summarize_probe(
        make_raw_probe(episodes=[make_episode(200, "timeout", hit_damage=())])
    )
    codes = {c.code for c in verdict_module.evaluate_gear(summary, strict=False) if not c.passed}
    assert "NO_DAMAGE_DEALT" in codes


def test_the_pre_probe_still_catches_a_frozen_opponent() -> None:
    summary = verdict_module.summarize_probe(
        make_raw_probe(
            episodes=[
                make_episode(200, "timeout", opponent_speed=0.0, opponent_moves=False)
            ]
        )
    )
    codes = {c.code for c in verdict_module.evaluate_gear(summary, strict=False) if not c.passed}
    assert "OPPONENT_FROZEN" in codes


def test_a_probe_that_did_not_run_blocks_the_pre_probe_too() -> None:
    checks = verdict_module.evaluate_gear({"ok": False, "error": "boom"}, strict=False)
    assert [c.code for c in checks if not c.passed] == ["PROBE_FAILED"]


# ===========================================================================
# STALE_POOL — the gate must not stop gating on the second attempt.
#
# `agent.train.build_snapshot_opponents` load-extends a pool it finds:
#
#     if os.path.isfile(os.path.join(directory, INDEX_FILENAME)):
#         pool = SnapshotPool.load(directory, sampling=..., log=log)
#
# and `SnapshotPool.load` restores the registry, BOTH Elo series and the match
# counters. Re-running the canary into the same --run-name therefore hands
# attempt 2 attempt 1's artifacts, and NO_NEW_SNAPSHOT, SNAPSHOT_UNCHANGED,
# RATED_ELO_EMPTY and DRAW_MAJORITY_TRAINING are all satisfied before this
# attempt has proved anything. That is the "manufactures confidence" failure in
# its purest form: a green verdict earned by a run that no longer happened.
# ===========================================================================


def test_a_newest_snapshot_older_than_this_run_blocks() -> None:
    """THE re-run trap, and the case no grad-step comparison can catch.

    Attempt 2's archive hook adds NOTHING, so the pool still holds attempt 1's
    snapshots. Every pool-derived check reads healthy — the pool has 5 members,
    the newest differs from snapshot 0 AND from the second-newest, rated matches
    are nonzero, the draw rate is fine — and the run itself completed with 1204
    gradient steps. Only the snapshot FILE's age gives it away.
    """
    evidence = mutate(
        make_evidence(), "pool.newest_snapshot_mtime", DRIVER_STARTED_AT - 3600.0
    )
    assert refusal_codes(evidence) == {"STALE_POOL"}

    # THE POINT: every check that attempt 1's artifacts pre-satisfy DOES pass on
    # this document. Without STALE_POOL the gate would report GREEN for a run
    # that archived nothing, which is the whole failure being closed.
    checks = {c.code: c for c in verdict_module.evaluate_canary(evidence).checks}
    for code in (
        "DRIVER_FAILED",
        "ZERO_GRAD_STEPS",
        "NO_NEW_SNAPSHOT",
        "SNAPSHOT_UNCHANGED",
        "RATED_ELO_EMPTY",
        "DRAW_MAJORITY_TRAINING",
        "CHECKPOINT_UNLOADABLE",
    ):
        assert checks[code].passed, code
    refusal = [
        c
        for c in verdict_module.evaluate_canary(evidence).refusals
        if c.code == "STALE_POOL"
    ][0]
    # The operator must be told what to DO, not just that something is stale.
    assert "--run-name" in refusal.check or "snapshots" in refusal.check


def test_a_pool_holding_two_runs_of_history_blocks() -> None:
    """A reload seam: grad steps that DROP as snapshot ids rise.

    Attempt 1 archived at 300/600/900/1200; attempt 2 reloaded the pool and
    archived at its own 300/600/900/1200, appending ids 5..8. One run cannot
    produce that series, and the pool's Elo and match counters are now a
    mixture of two runs — so this attempt's draw rate and rated-match count are
    diluted by an earlier one's.
    """
    evidence = make_evidence()
    evidence["pool"].update(
        size=9,
        snapshot_ids=list(range(9)),
        grad_steps={
            "0": 0, "1": 300, "2": 600, "3": 900, "4": 1200,
            "5": 300, "6": 600, "7": 900, "8": 1200,
        },
        pfsp_weights={str(i): 1.0 / 9.0 for i in range(9)},
        newest_snapshot_id=8,
    )
    assert refusal_codes(evidence) == {"STALE_POOL"}


def test_a_snapshot_from_a_grad_step_this_run_never_reached_blocks() -> None:
    """The pool's newest member outran the run that supposedly produced it."""
    evidence = mutate(make_evidence(), "driver.grad_steps", 400)
    # 400 is a perfectly healthy gradient-step count on its own ...
    assert "ZERO_GRAD_STEPS" not in refusal_codes(evidence)
    # ... but the newest snapshot claims grad step 1200, which this run never saw.
    assert refusal_codes(evidence) == {"STALE_POOL"}


@pytest.mark.parametrize(
    "path,value",
    [
        pytest.param("pool.newest_snapshot_mtime", None, id="no-file-mtime"),
        pytest.param("driver_started_at", None, id="no-run-start"),
        pytest.param("pool.grad_steps", {}, id="no-grad-step-record"),
        pytest.param("pool.newest_snapshot_id", None, id="no-newest-id"),
    ],
)
def test_unprovable_pool_provenance_blocks(path: str, value: Any) -> None:
    """Fail-closed: "cannot tell" is a refusal, never a pass.

    Absent provenance is exactly what a pre-existing pool looks like to a
    collector that could not stamp it, and reading it as healthy is how the
    second attempt would inherit the first one's verdict.
    """
    assert "STALE_POOL" in refusal_codes(mutate(make_evidence(), path, value))


def test_a_zero_grad_step_run_is_not_also_reported_as_a_stale_pool() -> None:
    """One missing field, one code.

    ``ZERO_GRAD_STEPS`` owns an absent or zero gradient-step count; STALE_POOL
    asks its grad-step question only when the run reported a POSITIVE one, so
    the headline refusal is not buried under a second code saying the same
    thing. The snapshot's own timestamp still clears it here, which is why this
    stays a single refusal rather than becoming two.
    """
    for value in (0, None):
        assert refusal_codes(mutate(make_evidence(), "driver.grad_steps", value)) == {
            "ZERO_GRAD_STEPS"
        }


def test_a_pool_of_clones_blocks_even_when_it_differs_from_the_seed() -> None:
    """The archive hook froze after its first successful write.

    Snapshots 1..N are bit-identical to each other but differ from snapshot 0,
    so a newest-vs-snapshot-0 diff alone passes. The pool is still one policy
    under many ids, and PFSP, Elo and the reference eval all measure nothing.
    """
    evidence = mutate(
        make_evidence(), "pool.newest_vs_second_newest_max_abs_delta", 0.0
    )
    assert evidence["pool"]["newest_vs_snapshot0_max_abs_delta"] > 0.0
    assert refusal_codes(evidence) == {"SNAPSHOT_UNCHANGED"}


def test_an_undiffable_second_newest_snapshot_blocks() -> None:
    """Fail-closed on the second diff too, not only the first."""
    evidence = mutate(
        make_evidence(),
        "pool.newest_vs_second_newest_error",
        "snapshots 3 and 4 have different parameter sets",
    )
    assert "SNAPSHOT_UNCHANGED" in refusal_codes(evidence)


# ===========================================================================
# The probe-window floor is a SAMPLE-SIZE floor, and it is sized from the fight.
# ===========================================================================


def test_the_probe_window_floor_is_derived_from_the_shortest_armored_kill() -> None:
    """28 windows per episode, and every term of that comes from the repo.

    ``ceil(20 HP / 3.12) == 7`` landed hits, spaced
    ``WINDOWS_PER_SWING == ceil(12.5 / ACTION_REPEAT) == 4`` decision windows
    apart. The flat 100 this replaces worked out at ~34 windows per episode at
    the default ``--probe-episodes 3`` — ABOVE the best case — so a healthy
    fleet whose armored episodes turned out short would have been refused on
    statistics rather than on a defect, by a threshold calibrated against the
    very number this canary exists to measure for the first time.
    """
    from eval.combat_probe import WINDOWS_PER_SWING

    hits = math.ceil(20.0 / verdict_module.ARMORED_HIT_DAMAGE)
    assert hits == 7
    assert WINDOWS_PER_SWING == 4
    assert (
        verdict_module.DEFAULT_THRESHOLDS["min_probe_windows_per_episode"]
        == WINDOWS_PER_SWING * hits
        == 28
    )
    assert "min_probe_windows" not in verdict_module.DEFAULT_THRESHOLDS


def test_a_short_but_healthy_armored_probe_is_not_refused() -> None:
    """Three 30-window episodes: 90 windows, under the old flat floor of 100.

    Each one is a complete, healthy armored kill — 7 landed hits, both meters
    moving, four macros per seat, the opponent walking. Refusing it would send
    an operator hunting a defect that is not there.
    """
    summary = verdict_module.summarize_probe(
        make_raw_probe(episodes=[make_episode(30, "win") for _ in range(3)])
    )
    assert summary["windows"] == 90
    assert summary["hits_dealt"]["iron_like"] == 18
    assert refusal_codes(make_evidence(summary)) == set()


def test_the_sample_size_refusal_says_it_is_a_sample_size_refusal() -> None:
    """"PROBE_FAILED" on volume must not read as "the fleet is broken"."""
    summary = verdict_module.summarize_probe(
        make_raw_probe(episodes=[make_episode(10, "win") for _ in range(3)])
    )
    checks = verdict_module.evaluate_gear(summary, strict=True)
    refusal = [c for c in checks if not c.passed][0]
    assert refusal.code == "PROBE_FAILED"
    assert refusal.why.startswith("THIS IS A SAMPLE-SIZE FLOOR")
    assert "--probe-episodes" in refusal.check


# ===========================================================================
# The eval figure T19 reads: where it comes from, and who it was measured against.
# ===========================================================================


def _collector_regexes() -> Dict[str, Any]:
    """The ``*_RE`` patterns the collector heredoc compiles, extracted by ast."""
    import ast

    patterns: Dict[str, Any] = {}
    for node in ast.walk(ast.parse(extract_heredoc("CANARY_COLLECT_PY"))):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        call = node.value
        if (
            isinstance(target, ast.Name)
            and target.id.endswith("_RE")
            and isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "compile"
            and call.args
            and isinstance(call.args[0], ast.Constant)
        ):
            patterns[target.id] = re.compile(call.args[0].value)
    return patterns


def test_the_eval_figure_is_read_from_the_driver_log_not_summary_json() -> None:
    """``summary.json`` never carries a ``win_rate`` on the multi-arena path.

    The only ``logger.summary()`` call in `agent/train.py` is
    ``logger.summary(selfplay_log_row(snapshot_pool))``, whose keys are
    ``elo/*``, ``selfplay/pool_size``, ``selfplay/matches_scored``,
    ``selfplay/rated_matches``, ``selfplay/draw_rate`` and
    ``selfplay/win_rate_vs_ref_*`` — no ``win_rate``, no
    ``mean_episode_length``. A collector gated on ``"win_rate" in summary``
    therefore recorded ``{"ran": False}`` on EVERY run and T19's eval figure was
    always ``None``: fail-closed, but silently absent.
    """
    # Comments in the collector EXPLAIN why summary.json is unusable here, so
    # the assertion is against executable code only.
    collect = "\n".join(
        line
        for line in extract_heredoc("CANARY_COLLECT_PY").splitlines()
        if not line.lstrip().startswith("#")
    )
    assert "read_summary" not in collect
    assert "summary.json" not in collect
    assert "[multi grad_step " in collect

    train = open(
        os.path.join(REPO_ROOT, "agent", "train.py"), "r", encoding="utf-8"
    ).read()
    assert "logger.summary(selfplay_log_row(snapshot_pool))" in train
    # ... and the driver still prints the two lines the collector parses.
    for fragment in (
        'f"[multi grad_step {eval_grad_step}] "',
        'f"opponent={eval_opponent_name}"',
        'f"  last eval: win_rate={report.win_rate:.3f} "',
        'f"mean_len={report.mean_episode_length:.1f} "',
    ):
        assert fragment in train, fragment


def test_the_collectors_eval_patterns_match_the_lines_the_driver_prints() -> None:
    """Two heredocs and one f-string in a third file, with no shared type.

    A format drift would silently return the eval figure to ``None``, which is
    exactly the failure being fixed — so the patterns are matched against the
    real line shapes here, including the one that must NOT match.
    """
    patterns = _collector_regexes()
    assert {"_EVAL_CYCLE_RE", "_LAST_EVAL_RE"} <= set(patterns)

    cycle = patterns["_EVAL_CYCLE_RE"].search(
        "[multi grad_step 1200] win_rate=0.500 mean_len=201.3 "
        "aim_invisible=0.012 passed_m2=False opponent=scripted_mixed"
    )
    assert cycle is not None
    assert cycle.group("opponent") == "scripted_mixed"
    assert int(cycle.group("grad")) == 1200
    assert float(cycle.group("mean")) == pytest.approx(201.3)

    final = patterns["_LAST_EVAL_RE"].search(
        "  last eval: win_rate=0.500 mean_len=201.3 aim_invisible=0.012"
    )
    assert final is not None
    assert float(final.group("mean")) == pytest.approx(201.3)

    # The self-play row shares the "[multi grad_step N]" prefix and carries no
    # episode length at all; matching it would report an Elo as a mean length.
    assert not patterns["_EVAL_CYCLE_RE"].search(
        "[multi grad_step 1200] selfplay: elo_rated=1014.2 (12 rated match(es)) "
        "elo_online=1003.5 pool=5 matches=214 draw_rate=0.056"
    )


def test_the_eval_figure_names_the_opponent_it_was_measured_against() -> None:
    """It is the SCRIPTED yardstick, not "eps=0 both sides".

    ``build_eval_opponent`` returns the same scripted ``EvalOpponentDriver`` for
    ``cfg.opponent == "selfplay"`` — deliberately, because an Elo ladder
    measured only against past selves can climb while the whole pool drifts. The
    eps=0 half of a self-play eval cycle is the rated reference gauntlet, and it
    reports Elo, never an episode length.
    """
    measurements = verdict_module.build_measurements(make_evidence())
    assert measurements["armored_mean_episode_length_eval_vs_scripted"] is not None
    assert measurements["eval_opponent"] == "scripted_mixed"
    report = verdict_module.format_report(
        verdict_module.evaluate_canary(make_evidence()), make_evidence()
    )
    assert "armored_mean_episode_length_eval_vs_scripted" in report
    assert "eval_opponent" in report
    assert "eval_greedy" not in report


def test_an_eval_that_never_ran_is_none_rather_than_zero() -> None:
    """A run whose eval cadence never fired must not report a length of 0."""
    evidence = mutate(
        make_evidence(), "eval", {"ran": False, "error": "no eval line in the log"}
    )
    measurements = verdict_module.build_measurements(evidence)
    assert measurements["armored_mean_episode_length_eval_vs_scripted"] is None
    assert measurements["eval_opponent"] is None
    # The probe's own figure is unaffected: it is measured, not logged.
    assert measurements["armored_mean_episode_length_probe"] == pytest.approx(200.0)


# ===========================================================================
# Reporting and the T19 measurements.
# ===========================================================================


def test_every_refusal_prints_why_and_what_to_check() -> None:
    """"Exit non-zero" is not a diagnosis. Each refusal must name the failure."""
    evidence = mutate(make_evidence(), "driver.grad_steps", 0)
    verdict = verdict_module.evaluate_canary(evidence)
    assert verdict.refusals
    for check in verdict.refusals:
        assert check.why.strip(), check.code
        assert check.check.strip(), check.code
        assert check.code not in check.why  # a code is not an explanation
    report = verdict_module.format_report(verdict, evidence)
    assert "VERDICT: REFUSED" in report
    assert "ZERO_GRAD_STEPS" in report
    assert "WHY:" in report and "CHECK:" in report
    # The lowered replay floor is stated, so nobody mistakes it for production.
    assert "--min-replay 2000" in report
    assert "25000" in report


def test_a_green_report_says_so_and_still_prints_the_measurements() -> None:
    evidence = make_evidence()
    verdict = verdict_module.evaluate_canary(evidence)
    report = verdict_module.format_report(verdict, evidence)
    assert "VERDICT: GREEN" in report
    assert "MEASURED FOR T19" in report
    assert "armored_mean_episode_length_probe" in report


def test_measurements_carry_the_armored_numbers_t19_needs() -> None:
    """T19 sizes ``--eps-decay-episodes`` from these, and they are MEASURED.

    Nothing here may fall back to the bare-handed 95-step figure or the 285-step
    one that predates it; both describe a different game.
    """
    evidence = make_evidence()
    measurements = verdict_module.build_measurements(evidence)
    assert measurements["armored_mean_episode_length_probe"] == pytest.approx(200.0)
    assert measurements["armored_median_episode_length_probe"] == pytest.approx(200.0)
    assert measurements["probe_learner_epsilon"] == pytest.approx(0.05)
    assert measurements["probe_opponent_epsilon"] == pytest.approx(0.02)
    # NOT "eval_greedy": the periodic eval fights the FIXED SCRIPTED yardstick
    # on a self-play run, and the figure carries the driver's own name so it
    # cannot be relabelled by accident.
    assert measurements[
        "armored_mean_episode_length_eval_vs_scripted"
    ] == pytest.approx(201.3)
    assert measurements["eval_opponent"] == "scripted_mixed"
    assert measurements["eval_episodes_per_cycle"] == 6
    assert "armored_mean_episode_length_eval_greedy" not in measurements
    assert measurements["training_episodes"] == 214
    assert measurements["training_grad_steps"] == 1204
    # 214 episodes / (1800 s / 3600) / 4 arenas == 107.0 episodes per arena-hour.
    assert measurements["measured_episodes_per_arena_hour"] == pytest.approx(107.0)
    assert measurements["projected_episodes_per_hour_at_25_pads"] == pytest.approx(
        107.0 * 25
    )
    assert measurements["armored_damage_dealt_per_episode"] == pytest.approx(20.0)
    assert measurements["max_episode_steps"] == 600
    assert measurements["notes"]


def test_measurements_are_none_rather_than_wrong_when_inputs_are_missing() -> None:
    """A measurement T19 would size a 24-hour run from must never be fabricated."""
    evidence = mutate(make_evidence(), "driver_wall_seconds", 0.0)
    measurements = verdict_module.build_measurements(evidence)
    assert measurements["measured_episodes_per_arena_hour"] is None
    assert measurements["projected_episodes_per_hour_at_25_pads"] is None


def test_the_measurements_record_which_warm_start_was_proved() -> None:
    """T19's cross-check reads the MEASUREMENTS file, so the digest must be in it.

    evidence.json has always recorded `warm_start_sha256`, but the launch gate
    never opens evidence.json — it reads canary_measurements.json. Without this
    pass-through, a canary GREEN earned on runs/m4.best.pt says nothing about a
    launch tab-completed onto runs/m4.pt.
    """
    measurements = verdict_module.build_measurements(make_evidence())
    assert measurements["warm_start"] == "/abs/runs/m4.best.pt"
    assert measurements["warm_start_sha256"] == "0" * 64


def test_an_unrecorded_warm_start_digest_is_none_not_invented() -> None:
    """Absent provenance stays absent; T19's gate fails closed on the None."""
    measurements = verdict_module.build_measurements({})
    assert measurements["warm_start"] is None
    assert measurements["warm_start_sha256"] is None


def test_measurements_are_json_serializable(tmp_path: Any) -> None:
    """The measurement file is T19's input, so it has to survive a round trip."""
    measurements = verdict_module.build_measurements(make_evidence())
    path = tmp_path / "canary_measurements.json"
    path.write_text(json.dumps(measurements, indent=2, sort_keys=True))
    assert json.loads(path.read_text())["training_grad_steps"] == 1204


def test_main_exits_zero_on_green_and_one_on_a_refusal(tmp_path: Any) -> None:
    green = tmp_path / "green.json"
    green.write_text(json.dumps(make_evidence()))
    measurements = tmp_path / "measurements.json"
    assert verdict_module.main([str(green), str(measurements)]) == 0
    assert json.loads(measurements.read_text())["training_grad_steps"] == 1204

    refused = tmp_path / "refused.json"
    refused.write_text(json.dumps(mutate(make_evidence(), "driver.grad_steps", 0)))
    assert verdict_module.main([str(refused)]) == 1


# ===========================================================================
# The shell script's own contract.
# ===========================================================================


def test_the_script_exists_and_is_executable() -> None:
    assert os.path.isfile(SCRIPT_PATH)
    assert os.access(SCRIPT_PATH, os.X_OK), "the operator runs this directly"


def test_the_script_parses() -> None:
    """`bash -n` parses without executing — a broken quote would only show up
    live, halfway through the last gate before a 24-hour run."""
    result = subprocess.run(
        ["bash", "-n", SCRIPT_PATH], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "name",
    [
        "CANARY_ROOT_PY",
        "CANARY_VERDICT_PY",
        "CANARY_PROBE_PY",
        "CANARY_GATE_PY",
        "CANARY_COLLECT_PY",
    ],
)
def test_every_embedded_python_block_compiles(name: str) -> None:
    """A syntax error in a heredoc is invisible until the phase that runs it."""
    compile(extract_heredoc(name), f"{SCRIPT_PATH}:{name}", "exec")


def test_the_script_starts_no_server() -> None:
    """The boot order is Paper -> bridges -> driver, and the operator owns the
    first two. A canary that boots its own fleet would also be free to boot it
    WRONG (knockback-immune), which is the failure it is meant to detect."""
    code = executable_shell()
    # The stripper must not have eaten the file: if it had, "no launcher found"
    # would be vacuously true and this test would guarantee nothing.
    assert "agent.train" in code
    assert "lsof" in code
    for launcher in (
        "start-pads",
        "start.sh",
        "setup.sh",
        ".jar",
        "java",
        "distributed.launcher",
        "npm",
        "node ",
    ):
        assert launcher not in code, f"the canary must not invoke {launcher}"
    # It must still TELL the operator how to boot it, in the help text.
    text = open(SCRIPT_PATH, "r", encoding="utf-8").read()
    assert "DUMMY_KNOCKBACK_IMMUNE=false server/setup/start-pads.sh" in text


def test_bridge_ports_are_inspected_without_connecting() -> None:
    """BridgeServer accepts ONE client; a second connection destroys the first.

    Occupancy must come from `lsof ... -sTCP:LISTEN`, never from a connect
    probe, and the only sanctioned connect target is the Minecraft port (Paper
    is multi-client).
    """
    text = open(SCRIPT_PATH, "r", encoding="utf-8").read()
    assert "lsof -nP -iTCP:" in text
    assert "-sTCP:LISTEN" in text
    # The one connect probe in the file is named for what it may touch, and is
    # only ever called with the Minecraft port.
    assert "mc_connect_probe" in text
    calls = [
        line.strip()
        for line in text.splitlines()
        if "mc_connect_probe " in line and not line.lstrip().startswith("#")
    ]
    assert calls, "the Minecraft port must still be checked"
    for call in calls:
        assert "MC_PORT" in call or "mc_connect_probe()" in call, call


def test_the_driver_invocation_matches_the_plan() -> None:
    """The canary launches the run the plan specifies, not a lookalike."""
    text = open(SCRIPT_PATH, "r", encoding="utf-8").read()
    for flag in (
        "--opponent selfplay",
        "--warm-start ",
        "--warm-start-sha256 ",
        "--checkpoint ",
        "--min-replay ",
        "--max-grad-steps ",
        "--snapshot-every-grad-steps ",
        "--snapshot-sampling pfsp",
        "--reference-promote-grad-steps ",
        "--eval-every-grad-steps ",
        "--eval-episodes ",
        "--eps-decay-episodes ",
    ):
        assert flag in text, flag
    # "--checkpoint, never --best-checkpoint alone."
    assert '--best-checkpoint "${BEST_CHECKPOINT_PATH}"' in text
    assert '--checkpoint "${CHECKPOINT_PATH}"' in text
    assert "${CHECKPOINT_PATH}" != "${BEST_CHECKPOINT_PATH}"


def test_the_canary_lowers_min_replay_and_says_so() -> None:
    """The plan's whole point: at min_replay=25_000 a short run takes ZERO
    gradient steps. The choice must be visible in the script's own output."""
    text = open(SCRIPT_PATH, "r", encoding="utf-8").read()
    assert "MIN_REPLAY=2000" in text
    assert "MIN_REPLAY_PRODUCTION=25000" in text
    assert "4.8782" in text  # the measured rate the sizing is derived from
    # ... and the banner the operator sees names both numbers.
    report = verdict_module.format_report(
        verdict_module.evaluate_canary(make_evidence()), make_evidence()
    )
    assert "LOWERED for the" in report


def test_the_preflight_refuses_a_pre_existing_pool_or_checkpoint() -> None:
    """A second attempt must not inherit the first one's artifacts.

    ``build_snapshot_opponents`` LOAD-EXTENDS a pool.json it finds and
    ``SnapshotPool.load`` restores both Elo series and the match counters, so
    re-running into the same ``--run-name`` pre-satisfies NO_NEW_SNAPSHOT,
    SNAPSHOT_UNCHANGED, RATED_ELO_EMPTY and DRAW_MAJORITY_TRAINING with work
    this attempt never did. ``runs/<run>.pt`` does the same for
    CHECKPOINT_UNLOADABLE, and ``metrics.jsonl`` is APPENDED to.

    This must BLOCK, not warn: the failure it prevents is silent by
    construction — attempt 2 exits GREEN and nothing in the report says which
    attempt earned it.
    """
    text = open(SCRIPT_PATH, "r", encoding="utf-8").read()
    code = executable_shell()
    # The check is real shell, not a comment.
    for name in ("STALE_ARTIFACTS", "POOL_INDEX", "METRICS_JSONL"):
        assert name in code, name

    marker = 'if [[ -n "${STALE_ARTIFACTS}" ]]; then'
    assert text.count(marker) == 1
    block = text[text.index(marker) :]
    block = block[: block.index("\nfi\n")]
    # Blocking, and it names both ways out.
    assert "die " in block
    assert "warn " not in block
    assert "--run-name" in block
    assert "rm -rf" in block

    # All four artifacts are actually inspected, and every one of them is a
    # thing the driver reads back rather than overwrites.
    loop = text[text.index("STALE_ARTIFACTS=\"\"") : text.index(marker)]
    for artifact in (
        "${POOL_INDEX}",
        "${CHECKPOINT_PATH}",
        "${BEST_CHECKPOINT_PATH}",
        "${METRICS_JSONL}",
    ):
        assert artifact in loop, artifact
    assert "snapshots/pool.json" in text
    assert "metrics.jsonl" in text

    # ... and it runs BEFORE the budget is spent.
    assert text.index(marker) < text.index("-m agent.train")


def test_analyze_only_is_handled_before_every_preflight_check() -> None:
    """``--analyze-only`` must need no fleet, no warm start and no clean run dir.

    It re-runs the verdict over an evidence directory and connects to nothing,
    so it is the one entry point that works on a machine with no Paper, no
    bridges and a run directory full of the artifacts the preflight now refuses.
    A future edit sliding that block below phase 0 would silently make it
    require ``--warm-start`` and a live server — with no test failing — so the
    ORDER is pinned here rather than the behaviour being assumed.

    Raw source lines, not :func:`executable_shell`: the quote stripper replaces
    ``"${ANALYZE_ONLY}"`` with a space, which erases the very anchor this test
    needs. Each anchor below is asserted unique first, so the ordering claim
    cannot be satisfied by a lookalike elsewhere in the file.
    """
    text = open(SCRIPT_PATH, "r", encoding="utf-8").read()
    anchor = 'if [[ -n "${ANALYZE_ONLY}" ]]; then'
    assert text.count(anchor) == 1
    analyze_at = text.index(anchor)

    for later in (
        'log "phase 0: preflight"',
        '[[ -n "${WARM_START}" ]] ||',
        'if [[ -n "${STALE_ARTIFACTS}" ]]; then',
        'if ! mc_connect_probe "${HOST}" "${MC_PORT}"; then',
        "-m agent.train",
    ):
        assert text.count(later) == 1, later
        assert analyze_at < text.index(later), later

    # The block must EXIT, or "first" would not mean "instead of".
    block = text[analyze_at : text.index('log "phase 0: preflight"')]
    assert 'exit "${VERDICT_EXIT}"' in block
    # It judges an existing document; it must not launch or connect to anything.
    for forbidden in ("run_probe ", "mc_connect_probe ", "agent.train"):
        assert forbidden not in block, forbidden


def test_a_collector_failure_does_not_claim_nothing_was_run() -> None:
    """Exit 2 promises "nothing was run"; a collector failure breaks that.

    By the time the collector runs, the budget has been spent and the fleet
    driven for half an hour. What failed is the judging, not the setup, so it
    gets its own code — and an operator reading "nothing was run" would
    otherwise re-run the whole gate looking for a config error.
    """
    text = open(SCRIPT_PATH, "r", encoding="utf-8").read()
    # The four codes are documented where the operator will look for them.
    for described in (
        "0 = GREEN",
        "1 = REFUSED",
        "2 = usage/preflight error",
        "3 = the run happened but could not be JUDGED",
    ):
        assert described in text, described
    assert "  3  the run happened but could not be JUDGED" in text  # usage text

    # The collector's failure branch exits 3, and NOT through die() (which is 2).
    assert "CANARY_COLLECT_PY\n) || {" in text
    branch = text[text.index("CANARY_COLLECT_PY\n) || {") :]
    branch = branch[: branch.index("\n}\n")]
    assert "exit 3" in branch
    assert "die " not in branch


def test_the_canary_produces_a_measurement_file_for_t19() -> None:
    text = open(SCRIPT_PATH, "r", encoding="utf-8").read()
    assert "canary_measurements.json" in text
    assert "MEASUREMENTS_JSON" in text


def test_the_operator_facing_strings_are_ascii_only() -> None:
    """Every string the gate PRINTS is ASCII.

    The recorded gotcha this respects: a non-ASCII byte on a cp1252 console
    crashes the encode, and `agent.train._log` already escapes for exactly that
    reason. A gate that raises while printing its own refusal is a gate that
    silently green-lights nothing at all. Docstrings are exempt - they are never
    written to the operator's terminal.
    """
    import ast

    source = extract_heredoc("CANARY_VERDICT_PY")
    tree = ast.parse(source)
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)):
            body = getattr(node, "body", None)
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                docstrings.add(id(body[0].value))
    offenders = [
        (node.lineno, node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
        and any(ord(ch) > 127 for ch in node.value)
    ]
    assert offenders == []
    # And the rendered report, end to end.
    evidence = mutate(make_evidence(), "driver.grad_steps", 0)
    report = verdict_module.format_report(
        verdict_module.evaluate_canary(evidence), evidence
    )
    report.encode("ascii")


def test_the_probe_records_every_field_the_verdict_reads() -> None:
    """The two heredocs must agree on the raw probe's field names.

    They are separate blocks in one shell script with no shared type, so a
    renamed key would silently produce an all-zero summary: no damage, no hits,
    no cooldown drops - which reads as a catastrophic fleet failure rather than
    as a typo.
    """
    import ast

    probe_source = extract_heredoc("CANARY_PROBE_PY")
    literals = {
        node.value
        for node in ast.walk(ast.parse(probe_source))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    required = {
        # per-window record
        "a",
        "oa",
        "cd",
        "ocd",
        "dd",
        "dt",
        "osp",
        "opx",
        "opz",
        # per-episode record
        "length",
        "outcome",
        "windows",
        # document level
        "ok",
        "error",
        "checkpoint",
        "episodes",
        "learner_epsilon",
        "opponent_epsilon",
        "max_episode_steps",
    }
    assert required <= literals, sorted(required - literals)
    # The probe is the ONE place that reads the two seats' cooldowns and the
    # opponent's raw position; if those accessors go, the evidence goes silently.
    for accessor in (
        "opponent_observation()",
        "raw_opponent_view()",
        "Obs.ATTACK_COOLDOWN",
        "Obs.VEL_LOCAL",
        "mirror_opponent=True",
        "opp_action=",
    ):
        assert accessor in probe_source, accessor


def test_the_canary_refuses_a_foreign_checkout() -> None:
    """The M4 work is a WORKTREE beside the main checkout, sharing its venv.

    A PYTHONPATH or editable install that resolves `agent`/`env` to the other
    tree would have the canary green-light code the fleet never runs - the plan's
    "or the run trains the old game" failure, one level up.
    """
    code = executable_shell()
    assert "RESOLVED_ROOTS" in code
    text = open(SCRIPT_PATH, "r", encoding="utf-8").read()
    assert "CANARY_ROOT_PY" in text
    root_probe = extract_heredoc("CANARY_ROOT_PY")
    assert "import agent" in root_probe and "import env" in root_probe


def test_the_probe_never_connects_when_a_client_is_already_attached() -> None:
    """A probe that "just tries it" would evict whatever is attached.

    `run_probe` must record the refusal as a failed probe document (which the
    verdict turns into PROBE_FAILED) instead of connecting anyway.
    """
    text = open(SCRIPT_PATH, "r", encoding="utf-8").read()
    assert "established_peers" in text
    start = text.index("run_probe() {")
    body = text[start : text.index("\n}", start)]
    assert "established_peers" in body
    assert "refused to connect" in body
    # ... by WRITING a failed probe document, which the verdict turns into a
    # PROBE_FAILED refusal, rather than connecting and evicting the incumbent.
    assert '{"ok": false, "error": "%s"}' in body
    assert "return 0" in body


# ===========================================================================
# The AC14 recorded-digest gate (--expect-sha256). The function under test is
# extracted VERBATIM from the script — the same no-second-copy rule the heredoc
# extraction enforces — and run under bash with the script's own log/warn/die,
# so the exit codes observed here are the exit codes the operator gets.
# Offline: nothing in this section opens a socket.
# ===========================================================================


def _shell_function(name: str) -> str:
    """One top-level shell function's text, verbatim, definition to close."""
    text = "\n".join(_script_lines())
    marker = f"{name}() {{"
    assert text.count(marker) == 1, marker
    start = text.index(marker)
    return text[start : text.index("\n}", start) + 2]


def _shell_logger_defs() -> str:
    """The script's OWN log/warn/die one-liners, so exit codes cannot drift."""
    lines = [
        line for line in _script_lines() if re.match(r"^(log|warn|die)\(\)\s+\{", line)
    ]
    assert len(lines) == 3, lines
    return "\n".join(lines)


def run_digest_gate(
    warm_start: str,
    computed: str,
    expect: Optional[str],
    cwd: Optional[str] = None,
) -> "subprocess.CompletedProcess[str]":
    """Drive the real `check_expected_sha256` exactly as the preflight calls it:
    after WARM_START_SHA256 has been computed from the --warm-start file."""
    harness = "\n".join(
        [
            "set -euo pipefail",
            f"WARM_START={shlex.quote(warm_start)}",
            f"WARM_START_SHA256={shlex.quote(computed)}",
            f"EXPECT_SHA256={shlex.quote(expect or '')}",
            _shell_logger_defs(),
            _shell_function("check_expected_sha256"),
            "check_expected_sha256",
        ]
    )
    return subprocess.run(
        ["bash", "-c", harness], capture_output=True, text=True, check=False, cwd=cwd
    )


def test_a_matching_expected_digest_clears_the_preflight() -> None:
    result = run_digest_gate("/abs/runs/m4.best.pt", "2" * 64, "2" * 64)
    assert result.returncode == 0, result.stderr
    assert "matches --expect-sha256" in result.stdout
    assert "WARNING" not in result.stderr


def test_a_mismatched_expected_digest_refuses_naming_both(tmp_path: Any) -> None:
    """The refusal carries BOTH digests and the RESOLVED absolute path.

    The warm start is passed RELATIVE here, because that is when the resolved
    path earns its place: "runs/m4.pt" in the refusal would leave the operator
    guessing which checkout's runs/ it meant.
    """
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "m4.pt").write_bytes(b"the rejected 30k-step net")
    result = run_digest_gate("runs/m4.pt", "2" * 64, "1" * 64, cwd=str(tmp_path))
    assert result.returncode == 2
    assert "WARM_START_DIGEST_MISMATCH" in result.stderr
    assert "2" * 64 in result.stderr
    assert "1" * 64 in result.stderr
    assert str(tmp_path / "runs" / "m4.pt") in result.stderr


def test_omitting_the_flag_changes_nothing_but_names_the_hole() -> None:
    """No flag, no new refusal — but the unverified digest is announced."""
    result = run_digest_gate("/abs/runs/m4.best.pt", "2" * 64, None)
    assert result.returncode == 0, result.stderr
    assert "--expect-sha256" in result.stderr
    assert "NOTHING" in result.stderr


@pytest.mark.parametrize(
    "malformed",
    ["xyz", "A" * 64, "a" * 63, "a" * 65, "1d3d0c60"],
    ids=["not_hex", "uppercase", "too_short", "too_long", "truncated"],
)
def test_a_malformed_expected_digest_is_refused_up_front(malformed: str) -> None:
    """A value that can never match must refuse at parse time, not report a
    baffling "mismatch" after half the preflight has run. This drives the REAL
    script: the validation sits with the argument checks, before the
    interpreter probe, so the run dies before it writes or connects to
    anything."""
    result = subprocess.run(
        ["bash", SCRIPT_PATH, "--expect-sha256", malformed],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "64 lowercase hex" in result.stderr


def test_the_digest_gate_sits_on_the_preflight_path() -> None:
    """A function nobody calls gates nothing. The call is pinned directly after
    the digest is computed and logged, before any budget is spent."""
    text = "\n".join(_script_lines())
    assert 'log "  sha256 ${WARM_START_SHA256}"\ncheck_expected_sha256' in text
