"""tests/test_pad_isolation.py — T12's cross-pad isolation instrument (AC13).

Two offline-testable halves, both exercised here with NO socket and NO live
server:

  1. The bridge-log PARSER (``parse_pad_log_lines`` / ``parse_pad_log_file`` /
     ``verify_pad_log``) against realistic fixtures: a boot anchor line, a
     foreign-player line, bridge lifecycle noise, a malformed line, and a
     truncated/decode-mangled file. The verbatim line formats are copied from
     their source, not guessed:
       * anchor line   -- bridge/run.js:343
       * foreign line  -- bridge/bot.js:1264 (``_scanForeignPlayers``)
     Both are stderr-only (never the frozen wire).

  2. The RECONCILIATION logic (``reconcile_pad_damage`` / ``_wire_health_loss``)
     against synthetic per-pad ``StateMsg`` sequences built with the real
     dataclass (``StateMsg.from_dict``, mirroring tests/test_benchmark.py's own
     ``_state`` helper) — never a hand-rolled mock with extra fields, which is
     exactly the mistake that shipped the original damage-channel bug (see the
     plan's Background section). Covers a clean multi-kill run (both the
     doImmediateRespawn-masked and the visible fatal-drop shapes), +/-1 window
     skew, every defect class (unrecorded hit, phantom hit / cross-pad
     mis-attribution signature, an off-death heal, a cumulative mismatch), the
     trailing-edge allowance, and an unprimed/dirty baseline.

Neither half touches eval/combat_probe.py — reconcile_against_wire is imported
and used verbatim (see eval/benchmark.py's T12 section docstring for why).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Tuple

import pytest

from bridge.messages import StateMsg
from eval.benchmark import (
    BenchmarkReport,
    PadForeignSighting,
    PadIsolationRecorder,
    PadLogAnchor,
    check_pad_isolation,
    default_pad_log_path,
    format_isolation_line,
    main,
    parse_pad_log_file,
    parse_pad_log_lines,
    reconcile_pad_damage,
    verify_pad_log,
)


# ===========================================================================
# Scripted StateMsg helper (mirrors tests/test_benchmark.py's ``_state``).
# ===========================================================================


def _state(
    *,
    damage_dealt: float = 0.0,
    opponent_health: float = 20.0,
    opponent_died: bool = False,
    attack_cooldown: float = 1.0,
    tick: int = 1,
) -> StateMsg:
    """A canonical valid ``state`` dataclass; only combat fields vary.

    Built through ``StateMsg.from_dict`` (the real schema parser), never a
    hand-built stand-in, so a fixture that drifts from the real wire shape
    fails loudly here instead of silently testing nothing.
    """
    return StateMsg.from_dict(
        {
            "type": "state",
            "self": {
                "pos": [0.0, 64.0, 0.0],
                "yaw": 0.0,
                "pitch": 0.0,
                "velocity": [0.0, 0.0, 0.0],
                "on_ground": True,
                "health": 20.0,
                "held_item": "iron_sword",
                "attack_cooldown": attack_cooldown,
            },
            "opponent": {
                "pos": [0.0, 64.0, 2.0],
                "yaw": 0.0,
                "pitch": 0.0,
                "velocity": [0.0, 0.0, 0.0],
                "on_ground": True,
                "health": opponent_health,
                "held_item": "iron_sword",
            },
            "events": {
                "damage_dealt": damage_dealt,
                "damage_taken": 0.0,
                "i_died": False,
                "opponent_died": opponent_died,
            },
            "arena": {"wall_distances": [8.0, 8.0, 8.0, 8.0]},
            "tick": tick,
            "code_version": "test",
        }
    )


def _states(rows: List[Tuple[float, float, bool]]) -> List[StateMsg]:
    return [
        _state(damage_dealt=d, opponent_health=h, opponent_died=died, tick=i)
        for i, (d, h, died) in enumerate(rows)
    ]


def _single_kill_rows(
    *, skew: int = 0, masked_final_drop: bool = True
) -> List[Tuple[float, float, bool]]:
    """The canonical 6,6,6,2 kill (mirrors test_combat_probe.py's helper).

    ``skew`` delays the wire-health drop by that many windows relative to its
    damage event (the ``update_health`` second-connection skew AC8 permits).
    ``masked_final_drop=True`` snaps health back to 20 in the SAME window the
    kill fires (doImmediateRespawn); ``False`` leaves the visible 0 in that
    window (the respawn shows up as its own later transition, if any).
    """
    hit_windows = {1: 6.0, 5: 6.0, 9: 6.0, 13: 2.0}
    health_after = {1: 14.0, 5: 8.0, 9: 2.0, 13: 20.0 if masked_final_drop else 0.0}
    rows: List[Tuple[float, float, bool]] = []
    health = 20.0
    for w in range(14):
        drop_window = w - skew
        if drop_window in health_after:
            health = health_after[drop_window]
        rows.append((hit_windows.get(w, 0.0), health, w == 13))
    return rows


def _two_kill_rows() -> List[Tuple[float, float, bool]]:
    """Two back-to-back kills, no episode boundary: kill 1's fatal drop is
    MASKED (doImmediateRespawn, same-window snap-back to 20); kill 2's is
    VISIBLE (0 shows before the run ends). Exercises both carve-out paths
    ``_wire_health_loss`` must handle without double-counting, in one
    continuous sequence — matching how run_benchmark's reset-less driver
    actually produces states.
    """
    return _single_kill_rows(masked_final_drop=True) + _single_kill_rows(
        masked_final_drop=False
    )


# ===========================================================================
# Reconciliation: the clean cases.
# ===========================================================================


def test_two_kill_run_reconciles_cleanly() -> None:
    report = reconcile_pad_damage(0, _states(_two_kill_rows()))
    assert report.errors == []
    assert report.ok
    assert report.n_windows == 28
    assert report.cumulative_damage_dealt == pytest.approx(40.0)
    assert report.cumulative_wire_health_loss == pytest.approx(40.0)


def test_single_kill_with_one_window_skew_reconciles() -> None:
    report = reconcile_pad_damage(0, _states(_single_kill_rows(skew=1)))
    assert report.errors == []
    assert report.cumulative_damage_dealt == pytest.approx(20.0)
    assert report.cumulative_wire_health_loss == pytest.approx(20.0)


def test_single_kill_with_early_skew_reconciles() -> None:
    report = reconcile_pad_damage(
        0, _states(_single_kill_rows(skew=-1, masked_final_drop=False))
    )
    assert report.errors == []


def test_start_health_is_overridable_for_a_non_clean_baseline() -> None:
    # The run's actual first sample is 14.0 (mid-episode), not the FULL_HEALTH
    # default. Passing the true start_health must reconcile cleanly.
    rows = [(0.0, 14.0, False), (6.0, 8.0, False)]
    report = reconcile_pad_damage(0, _states(rows), start_health=14.0)
    assert report.errors == []


# ===========================================================================
# Reconciliation: defect classes (the offline half of AC13's evidence).
# ===========================================================================


def test_empty_run_is_flagged_not_silently_ok() -> None:
    report = reconcile_pad_damage(3, [])
    assert not report.ok
    assert report.n_windows == 0
    assert any("no decision windows recorded" in e for e in report.errors)


def test_unprimed_dirty_baseline_fails_loudly() -> None:
    # The default start_health (20.0, the primed-fleet assumption) does not
    # match this pad's actual first sample (14.0, an unprimed/mid-episode
    # pad) -- must fail rather than silently assume a clean start.
    rows = [(0.0, 14.0, False)]
    report = reconcile_pad_damage(0, _states(rows))
    assert not report.ok
    assert any("unrecorded hit" in e for e in report.errors)


def test_unrecorded_wire_drop_is_a_defect() -> None:
    # The wire lost health but no damage_dealt event explains it -- the
    # original under-counting bug class the whole plan exists to fix.
    rows = _single_kill_rows()
    rows[1] = (0.0, rows[1][1], rows[1][2])  # drop the first hit's recording
    report = reconcile_pad_damage(0, _states(rows))
    assert not report.ok
    assert any("unrecorded hit" in e for e in report.errors)


def test_phantom_hit_is_the_cross_pad_mis_attribution_signature() -> None:
    # A recorded damage_dealt event with NO corresponding wire-health loss on
    # THIS pad's own dummy is exactly the shape a mis-attributed cross-pad hit
    # would take (see eval/benchmark.py's T12 section docstring: the
    # reconciliation channel-check is necessary but not sufficient on its
    # own -- this is the failure mode it CAN catch).
    rows = _single_kill_rows()
    w, h, died = rows[3]
    rows[3] = (6.0, h, died)  # extra event, wire never dropped here
    report = reconcile_pad_damage(0, _states(rows))
    assert not report.ok
    assert any("no matching wire-health drop" in e for e in report.errors)


def test_heal_outside_death_window_is_a_defect() -> None:
    # naturalRegeneration is off; a health INCREASE far from any death/respawn
    # is a defect, not noise.
    rows = _single_kill_rows()
    rows[3] = (0.0, 15.0, False)  # 14 -> 15 heal, nowhere near window 13's death
    report = reconcile_pad_damage(0, _states(rows))
    assert not report.ok
    assert any("INCREASED" in e for e in report.errors)


def test_cumulative_mismatch_beyond_trailing_allowance_is_flagged() -> None:
    rows = _two_kill_rows()
    d, h, died = rows[5]
    rows[5] = (d + 5.0, h, died)  # +5 phantom, nowhere near the run's last window
    report = reconcile_pad_damage(0, _states(rows))
    assert not report.ok
    assert any("cumulative damage_dealt" in e for e in report.errors)


def test_negative_residual_is_never_covered_by_the_trailing_allowance() -> None:
    # S1: the trailing allowance is ONE-SIDED. Shrink window 1's recorded hit
    # by exactly 1.0 (6.0 -> 5.0) on the plain two-kill run, whose trailing
    # hit is 2.0 (kill 2's own last hit, untouched). The resulting residual is
    # -1.0 -- SMALLER in magnitude than the 2.0 trailing allowance, so the
    # OLD symmetric check (`abs(residual) > trailing_hit`) would have SILENTLY
    # TOLERATED it (1.0 <= 2.0). That would be a real hole: a small
    # under-recorded hit anywhere in the run, hidden behind an allowance that
    # was only ever meant to excuse the boundary. The one-sided check must
    # flag any negative residual regardless of its magnitude relative to the
    # trailing hit.
    rows = _two_kill_rows()
    d, h, died = rows[1]
    rows[1] = (d - 1.0, h, died)  # 6.0 -> 5.0; wire still shows the full drop
    report = reconcile_pad_damage(0, _states(rows))
    assert not report.ok
    assert report.trailing_residual_allowance == pytest.approx(2.0)
    assert any("cumulative damage_dealt" in e for e in report.errors)


def test_phantom_in_second_to_last_window_is_still_flagged() -> None:
    # S5: the trailing allowance is scoped to "at most one hit's worth of
    # residual", not "anything near the tail". A legitimate trailing hit sits
    # in the run's TRUE last window (would earn the allowance on its own,
    # exactly as in test_trailing_window_hit_gets_the_boundary_allowance
    # below); an UNRELATED phantom hit sits in the window just before it. That
    # phantom pushes the aggregate residual past what the true trailing hit
    # alone can explain, so the run must still fail -- tested here by
    # CONSEQUENCE (the aggregate mismatch), not merely by asserting the
    # recorded trailing_residual_allowance value in isolation.
    rows = _two_kill_rows() + [(5.0, 20.0, False), (6.0, 20.0, False)]
    report = reconcile_pad_damage(0, _states(rows))
    assert not report.ok
    assert report.trailing_residual_allowance == pytest.approx(6.0)
    assert any("cumulative damage_dealt" in e for e in report.errors)


def test_trailing_window_hit_gets_the_boundary_allowance() -> None:
    # A hit recorded in the VERY LAST window with no wire-drop confirmation
    # yet (there is no window+1 in a truncated recording to catch a
    # +/-1-skewed drop). eval.combat_probe.reconcile_against_wire still (and
    # correctly, by its own unmodified contract) flags this window as
    # unreconciled -- T12 does not suppress that. What T12 must NOT do is
    # pile its OWN redundant cumulative-mismatch error on top when the only
    # discrepancy is exactly this trailing hit.
    rows = _two_kill_rows() + [(6.0, 20.0, False)]  # health not yet observed to drop
    report = reconcile_pad_damage(0, _states(rows))
    assert any("no matching wire-health drop" in e for e in report.errors)
    assert not any("cumulative damage_dealt" in e for e in report.errors)
    assert report.trailing_residual_allowance == pytest.approx(6.0)


# ===========================================================================
# Log parser: realistic fixtures (anchor line, foreign line, malformed,
# truncated / decode error).
# ===========================================================================

# Verbatim strings, copied from source (never guessed):
_ANCHOR_LINE_0 = "[bridge] pad 0 @ anchor 0,0 (learner_bot / dummy_bot)"  # run.js:343
_ANCHOR_LINE_3 = "[bridge] pad 3 @ anchor 1536,0 (learner_3 / dummy_3)"
_FOREIGN_LINE_3 = "[bridge] pad 3 foreign_players learner_5,dummy_7"  # bot.js:1264
_LIFECYCLE_NOISE = [
    "[bridge] listening on 127.0.0.1:5558, both bots spawned",
    "[bridge] env connected",
    "[bridge] learner bot disconnected: socketClosed",
]


def test_parser_recognizes_anchor_and_foreign_lines() -> None:
    lines = [_ANCHOR_LINE_3, *_LIFECYCLE_NOISE, _FOREIGN_LINE_3]
    summary = parse_pad_log_lines(lines)
    assert summary.lines_scanned == len(lines)
    assert summary.anchors == [
        PadLogAnchor(
            pad_index=3, anchor_x=1536, anchor_z=0, learner="learner_3", dummy="dummy_3"
        )
    ]
    assert summary.foreign_sightings == [
        PadForeignSighting(pad_index=3, names=("learner_5", "dummy_7"))
    ]


def test_parser_ignores_bridge_lifecycle_noise() -> None:
    summary = parse_pad_log_lines(_LIFECYCLE_NOISE)
    assert summary.anchors == []
    assert summary.foreign_sightings == []
    assert summary.lines_scanned == len(_LIFECYCLE_NOISE)


def test_parser_tolerates_malformed_lines_without_crashing() -> None:
    malformed = [
        "[bridge] pad foreign_players",  # missing pad index
        "[bridge] pad three @ anchor 0,0 (learner_bot / dummy_bot)",  # non-numeric pad
        "[bridge] pad 1 @ anchor notanumber,0 (learner_1 / dummy_1)",  # bad coord
        "garbage\x00binary\xffnoise not even a bridge line",
        "",
    ]
    summary = parse_pad_log_lines(malformed)
    assert summary.anchors == []
    assert summary.foreign_sightings == []
    assert summary.lines_scanned == len(malformed)


def test_parser_tolerates_a_truncated_final_line() -> None:
    # A mid-write crash: the last line has no trailing newline and is cut off
    # mid-token. Must not raise and must not spuriously match.
    lines = [_ANCHOR_LINE_3, "[bridge] pad 3 foreign_pla"]
    summary = parse_pad_log_lines(lines)
    assert summary.anchors == [
        PadLogAnchor(3, 1536, 0, "learner_3", "dummy_3")
    ]
    assert summary.foreign_sightings == []


def test_parse_pad_log_file_tolerates_decode_errors_at_eof(tmp_path: Path) -> None:
    # Simulates a log file caught mid-write in the middle of a multi-byte
    # UTF-8 sequence (a killed bridge process). errors="replace" must degrade
    # that trailing byte to unmatched noise, not raise UnicodeDecodeError.
    path = tmp_path / "pad-3.log"
    payload = (_ANCHOR_LINE_3 + "\n" + _FOREIGN_LINE_3 + "\n").encode("utf-8")
    payload += b"[bridge] pad 3 foreign_players learner_9,\xff\xfe"  # truncated multibyte
    path.write_bytes(payload)

    summary = parse_pad_log_file(path)
    assert summary.anchors == [PadLogAnchor(3, 1536, 0, "learner_3", "dummy_3")]
    # The clean foreign line still parses; the mangled trailing one does not
    # crash the parse (it may or may not match -- what matters is no raise).
    assert any(s.pad_index == 3 for s in summary.foreign_sightings)


def test_parse_pad_log_file_missing_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        parse_pad_log_file(tmp_path / "does-not-exist.log")


# ===========================================================================
# verify_pad_log: a log must be PROVEN to belong to the expected pad.
# ===========================================================================


def test_verify_pad_log_passes_when_anchor_matches() -> None:
    summary = parse_pad_log_lines([_ANCHOR_LINE_3, *_LIFECYCLE_NOISE])
    assert verify_pad_log(summary, 3) == []


def test_verify_pad_log_fails_with_no_anchor_line() -> None:
    # A pad-log-dir mistake (wrong dir, rotated-away file): absence of a
    # foreign_players line here must NOT read as a clean pass.
    summary = parse_pad_log_lines(_LIFECYCLE_NOISE)
    errors = verify_pad_log(summary, 3)
    assert errors and "no matching '@ anchor'" in errors[0]


def test_verify_pad_log_fails_when_anchor_is_for_a_different_pad() -> None:
    # This file is pad 0's own log, mistakenly pointed at as if it were pad 3's.
    summary = parse_pad_log_lines([_ANCHOR_LINE_0])
    errors = verify_pad_log(summary, 3)
    assert any("no matching '@ anchor'" in e for e in errors)
    assert any("also contains a boot line for pad 0" in e for e in errors)


# ===========================================================================
# check_pad_isolation / PadIsolationRecorder — the combined verdict.
# ===========================================================================


def _write_log(tmp_path: Path, pad_index: int, lines: List[str]) -> Path:
    path = tmp_path / f"pad-{pad_index}.log"
    path.write_text("\n".join(lines) + "\n")
    return path


def test_check_pad_isolation_clean_run_is_ok(tmp_path: Path) -> None:
    log_path = _write_log(tmp_path, 3, [_ANCHOR_LINE_3, *_LIFECYCLE_NOISE])
    report = check_pad_isolation(3, _states(_two_kill_rows()), log_path)
    assert report.ok
    assert report.foreign_sightings == []
    assert report.violations() == []


def test_check_pad_isolation_flags_injected_cross_pad_contamination(
    tmp_path: Path,
) -> None:
    # The reconciliation is perfectly clean (the damage CHANNEL is fine); the
    # foreign-player scan is the signal that actually proves contamination.
    # This is the AC13 scenario: reconciliation alone would pass, and the
    # scan is what catches it.
    log_path = _write_log(tmp_path, 3, [_ANCHOR_LINE_3, _FOREIGN_LINE_3])
    report = check_pad_isolation(3, _states(_two_kill_rows()), log_path)
    assert report.reconciliation.ok  # the channel itself is clean
    assert not report.ok  # but isolation is violated
    assert len(report.foreign_sightings) == 1
    assert report.foreign_sightings[0].names == ("learner_5", "dummy_7")
    assert any("foreign player" in v for v in report.violations())


def test_check_pad_isolation_ignores_another_pads_foreign_sighting(
    tmp_path: Path,
) -> None:
    # A shared/misrouted log containing pad 0's foreign_players line must not
    # be blamed on pad 3.
    log_path = _write_log(tmp_path, 3, [_ANCHOR_LINE_3, "[bridge] pad 0 foreign_players x"])
    report = check_pad_isolation(3, _states(_two_kill_rows()), log_path)
    assert report.foreign_sightings == []


def test_check_pad_isolation_propagates_missing_log(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        check_pad_isolation(3, _states(_two_kill_rows()), tmp_path / "missing.log")


# ===========================================================================
# W2: min_mtime -- verify_pad_log proves identity, not freshness. A leftover
# log from a previous, smaller/different boot can carry a perfectly valid
# anchor line and no foreign_players lines and would otherwise report a
# silent clean pass. min_mtime closes that gap.
# ===========================================================================


def test_check_pad_isolation_flags_a_stale_log(tmp_path: Path) -> None:
    log_path = _write_log(tmp_path, 3, [_ANCHOR_LINE_3])
    # This run supposedly started an hour AFTER the log was last touched --
    # far beyond the freshness slack, so this is unambiguously stale.
    run_start = log_path.stat().st_mtime + 3600.0
    report = check_pad_isolation(
        3, _states(_single_kill_rows()), log_path, min_mtime=run_start
    )
    assert not report.ok
    assert any("STALE" in e for e in report.log_errors)


def test_check_pad_isolation_accepts_a_fresh_log(tmp_path: Path) -> None:
    log_path = _write_log(tmp_path, 3, [_ANCHOR_LINE_3])
    # The log was touched (mtime) an hour AFTER this "run start" -- exactly
    # what connecting to a genuinely live, correctly-addressed bridge produces
    # (bridge/run.js logs "env connected" on every new connection), so this
    # must not be flagged.
    run_start = log_path.stat().st_mtime - 3600.0
    report = check_pad_isolation(
        3, _states(_single_kill_rows()), log_path, min_mtime=run_start
    )
    assert not any("STALE" in e for e in report.log_errors)


def test_check_pad_isolation_min_mtime_none_skips_the_freshness_check(
    tmp_path: Path,
) -> None:
    log_path = _write_log(tmp_path, 3, [_ANCHOR_LINE_3])
    report = check_pad_isolation(3, _states(_single_kill_rows()), log_path)  # no min_mtime
    assert report.ok


def test_check_pad_isolation_freshness_slack_absorbs_small_gaps(tmp_path: Path) -> None:
    # A gap smaller than _MTIME_FRESHNESS_SLACK_S (filesystem mtime
    # granularity / clock skew) must not be flagged; a gap well past it must.
    log_path = _write_log(tmp_path, 3, [_ANCHOR_LINE_3])
    mtime = log_path.stat().st_mtime

    within_slack = check_pad_isolation(
        3, _states(_single_kill_rows()), log_path, min_mtime=mtime + 0.5
    )
    assert not any("STALE" in e for e in within_slack.log_errors)

    beyond_slack = check_pad_isolation(
        3, _states(_single_kill_rows()), log_path, min_mtime=mtime + 30.0
    )
    assert any("STALE" in e for e in beyond_slack.log_errors)


def test_pad_isolation_recorder_check_all_forwards_min_mtime(tmp_path: Path) -> None:
    log_path = _write_log(tmp_path, 0, [_ANCHOR_LINE_0])
    stale_run_start = log_path.stat().st_mtime + 3600.0

    recorder = PadIsolationRecorder(1)
    for state in _states(_single_kill_rows()):
        recorder.record(0, state)

    reports = recorder.check_all({0: log_path}, min_mtime=stale_run_start)
    assert not reports[0].ok
    assert any("STALE" in e for e in reports[0].log_errors)


def test_pad_isolation_recorder_tracks_each_arena_independently(tmp_path: Path) -> None:
    recorder = PadIsolationRecorder(2)
    for state in _states(_single_kill_rows()):
        recorder.record(0, state)
    for state in _states(_two_kill_rows()):
        recorder.record(1, state)

    assert len(recorder.states_for(0)) == 14
    assert len(recorder.states_for(1)) == 28

    log_paths = {
        0: _write_log(tmp_path, 0, [_ANCHOR_LINE_0]),
        1: _write_log(tmp_path, 1, ["[bridge] pad 1 @ anchor 512,0 (learner_1 / dummy_1)"]),
    }
    reports = recorder.check_all(log_paths)
    assert set(reports) == {0, 1}
    assert reports[0].reconciliation.n_windows == 14
    assert reports[1].reconciliation.n_windows == 28
    assert reports[0].ok and reports[1].ok


def test_pad_isolation_recorder_rejects_a_missing_log_path(tmp_path: Path) -> None:
    recorder = PadIsolationRecorder(1)
    recorder.record(0, _state())
    with pytest.raises(KeyError):
        recorder.check_all({})  # pad 0 has no entry


def test_pad_isolation_recorder_rejects_bad_arena_count() -> None:
    with pytest.raises(ValueError):
        PadIsolationRecorder(0)


# ===========================================================================
# W3: pad_index_for_arena -- arena index must not be silently assumed to be
# the TRUE pad index. A caller connecting arena i to a port offset from the
# fleet's own base must resolve i to the pad it's ACTUALLY talking to, both
# for the log path AND for verify_pad_log's expected-pad-index check.
# ===========================================================================


def test_pad_isolation_recorder_check_all_uses_pad_index_for_arena_resolver(
    tmp_path: Path,
) -> None:
    # Arena 0 in THIS run is actually pad 2 (e.g. --port 5557 against a fleet
    # whose base is 5555). log_paths is keyed by ARENA index (0), but the
    # file on disk is genuinely pad 2's log (anchor line says "pad 2").
    recorder = PadIsolationRecorder(1)
    for state in _states(_single_kill_rows()):
        recorder.record(0, state)

    log_paths = {0: _write_log(tmp_path, 2, ["[bridge] pad 2 @ anchor 512,0 (learner_2 / dummy_2)"])}

    # Without the resolver (defaults to identity, arena 0 == pad 0): the
    # anchor line says pad 2, expected is pad 0 -> mismatch, correctly caught.
    unresolved = recorder.check_all(log_paths)
    assert not unresolved[0].ok
    assert any("no matching '@ anchor'" in e for e in unresolved[0].log_errors)

    # With the resolver telling check_all arena 0 IS pad 2: verifies cleanly,
    # and the report's own pad_index reflects the TRUE pad, not the arena.
    resolved = recorder.check_all(log_paths, pad_index_for_arena=lambda i: i + 2)
    assert resolved[0].ok
    assert resolved[0].pad_index == 2
    assert resolved[0].reconciliation.pad_index == 2


# ===========================================================================
# Small formatting / convention helpers.
# ===========================================================================


def test_default_pad_log_path_matches_start_pads_convention() -> None:
    assert default_pad_log_path("/srv/logs/pads", 7) == Path("/srv/logs/pads/pad-7.log")


def test_format_isolation_line_reports_ok_and_fail(tmp_path: Path) -> None:
    log_path = _write_log(tmp_path, 0, [_ANCHOR_LINE_0])

    ok_report = check_pad_isolation(0, _states(_single_kill_rows()), log_path)
    line = format_isolation_line(ok_report)
    assert line.startswith("[isolation] pad 0:")
    assert "OK" in line

    fail_report = check_pad_isolation(0, [], log_path)
    assert "FAIL" in format_isolation_line(fail_report)


def test_benchmark_report_pad_isolation_defaults_empty_and_serializes() -> None:
    report = BenchmarkReport()
    assert report.pad_isolation == {}
    assert report.to_dict()["pad_isolation"] == {}

    report.pad_isolation = {"0": {"ok": True}}
    assert report.to_dict()["pad_isolation"] == {"0": {"ok": True}}


# ===========================================================================
# main() CLI wiring — --pad-log-dir, offline (mirrors test_benchmark.py's
# _patch_main_for_live pattern: a scripted transport + a shared FakeClock, no
# socket, no real wall-clock wait).
# ===========================================================================


class _ScriptedIsolationBridge:
    """A minimal four-method transport: healthy, damage-free states forever.

    Isolation wiring is what this test proves, not combat arithmetic, so
    every state is a clean, unchanging 20 HP with no damage event. ``recv``
    does a tiny REAL ``time.sleep`` (mirroring eval.benchmark's own
    ``SleepingFakeBridge``) so the GIL yields between calls -- with a plain,
    non-blocking scripted ``recv`` a single arena thread can race through the
    whole decision budget before the OS ever schedules the other arena's
    thread, starving it of any recorded windows. That would be a test-fixture
    flakiness bug, not evidence about the production wiring under test.
    """

    def __init__(self, clock, *, wall_per_step: float) -> None:
        self._clock = clock
        self._wall_per_step = float(wall_per_step)
        self._tick = 0

    def connect(self) -> None:
        pass

    def send(self, obj) -> None:
        pass

    def recv(self) -> StateMsg:
        self._clock.advance(self._wall_per_step)
        self._tick += 1
        time.sleep(0.005)  # force a GIL yield so both arenas get scheduled
        return _state(tick=self._tick)

    def close(self) -> None:
        pass


class _NullLogger:
    def __init__(self, *a, **k) -> None:
        pass

    def log(self, *a, **k) -> None:
        pass

    def summary(self, *a, **k) -> None:
        pass

    def close(self) -> None:
        pass


def test_main_wires_pad_log_dir_end_to_end(monkeypatch, tmp_path: Path, capsys) -> None:
    import json

    import eval.benchmark as bench

    clock = bench.FakeClock()
    monkeypatch.setattr(
        bench,
        "TcpBridgeClient",
        lambda host, port: _ScriptedIsolationBridge(clock, wall_per_step=0.05),
    )
    monkeypatch.setattr(bench, "MetricsLogger", _NullLogger)
    monkeypatch.setattr(bench.time, "perf_counter", clock)

    log_dir = tmp_path / "pads"
    log_dir.mkdir()
    for i in range(2):
        (log_dir / f"pad-{i}.log").write_text(
            f"[bridge] pad {i} @ anchor {i * 512},0 (learner_{i} / dummy_{i})\n"
        )

    main(["--duration", "0.18", "--arenas", "2", "--pad-log-dir", str(log_dir)])
    out, err = capsys.readouterr()

    assert "foreign-player scan fires ONLY on reset" in err
    assert "[isolation] pad 0:" in err
    assert "[isolation] pad 1:" in err

    report = json.loads(out)
    assert set(report["pad_isolation"]) == {"0", "1"}
    for entry in report["pad_isolation"].values():
        assert entry["ok"] is True
        assert entry["foreign_events"] == 0
        assert entry["windows"] >= 1


def test_main_pad_log_dir_reports_missing_log_as_a_failure(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    import eval.benchmark as bench

    clock = bench.FakeClock()
    monkeypatch.setattr(
        bench,
        "TcpBridgeClient",
        lambda host, port: _ScriptedIsolationBridge(clock, wall_per_step=0.05),
    )
    monkeypatch.setattr(bench, "MetricsLogger", _NullLogger)
    monkeypatch.setattr(bench.time, "perf_counter", clock)

    log_dir = tmp_path / "pads"
    log_dir.mkdir()  # no pad-0.log written -- a missing-log-dir mistake

    rc = main(["--duration", "0.18", "--arenas", "1", "--pad-log-dir", str(log_dir)])
    err = capsys.readouterr().err

    assert rc == 1
    assert "[isolation] ABORT" in err
    assert "cross-pad isolation check failed" in err


# ===========================================================================
# W3, CLI level: --bridge-base-port derives the TRUE pad index, and a --port
# below --bridge-base-port is refused loudly before anything starts.
# ===========================================================================


def test_main_bridge_base_port_offset_derives_correct_pad_index(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    import json

    import eval.benchmark as bench

    clock = bench.FakeClock()
    monkeypatch.setattr(
        bench,
        "TcpBridgeClient",
        lambda host, port: _ScriptedIsolationBridge(clock, wall_per_step=0.05),
    )
    monkeypatch.setattr(bench, "MetricsLogger", _NullLogger)
    monkeypatch.setattr(bench.time, "perf_counter", clock)

    # Fleet base is 5555 (pads 0..); this run's --port 5557 with --arenas 2
    # connects to pads 2 and 3, NOT pads 0 and 1.
    log_dir = tmp_path / "pads"
    log_dir.mkdir()
    (log_dir / "pad-2.log").write_text(
        "[bridge] pad 2 @ anchor 1024,0 (learner_2 / dummy_2)\n"
    )
    (log_dir / "pad-3.log").write_text(
        "[bridge] pad 3 @ anchor 1536,0 (learner_3 / dummy_3)\n"
    )

    main(
        [
            "--duration", "0.18",
            "--arenas", "2",
            "--port", "5557",
            "--bridge-base-port", "5555",
            "--pad-log-dir", str(log_dir),
        ]
    )
    out, err = capsys.readouterr()

    assert "[isolation] pad 2:" in err
    assert "[isolation] pad 3:" in err
    assert "[isolation] pad 0:" not in err  # never misread as pad 0/1

    report = json.loads(out)
    assert set(report["pad_isolation"]) == {"2", "3"}
    for entry in report["pad_isolation"].values():
        assert entry["ok"] is True


def test_main_refuses_port_below_bridge_base_port(monkeypatch, tmp_path: Path, capsys) -> None:
    import eval.benchmark as bench

    # No transport/logger patched: this must fail BEFORE anything connects.
    logger_calls = []
    monkeypatch.setattr(
        bench,
        "MetricsLogger",
        lambda *a, **k: logger_calls.append((a, k)) or _NullLogger(),
    )

    rc = main(
        [
            "--arenas", "1",
            "--port", "5000",
            "--bridge-base-port", "5555",
            "--pad-log-dir", str(tmp_path),
        ]
    )
    err = capsys.readouterr().err

    assert rc == 1
    assert "[isolation] ABORT" in err
    assert "bridge-base-port" in err
    assert logger_calls == []  # refused before the logger (or anything else) was built


# ===========================================================================
# S4: PadIsolationRecorder construction must not leak the MetricsLogger on a
# bad --arenas.
# ===========================================================================


def test_main_bad_arenas_with_pad_log_dir_fails_before_constructing_logger(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    import eval.benchmark as bench

    logger_calls = []
    monkeypatch.setattr(
        bench,
        "MetricsLogger",
        lambda *a, **k: logger_calls.append((a, k)) or _NullLogger(),
    )

    rc = main(["--arenas", "0", "--pad-log-dir", str(tmp_path)])
    err = capsys.readouterr().err

    assert rc == 1
    assert "[isolation] ABORT" in err
    assert logger_calls == []  # PadIsolationRecorder(0) raised before the logger existed
