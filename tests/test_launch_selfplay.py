"""Tests for the T19 launch gate (`scripts/launch_selfplay.sh`).

This is the script that COMMITS THE NIGHT: it sizes `--eps-decay-episodes` from
a measurement, judges a 25-pad smoke, and starts a 24-hour run detached. Its
value is entirely in its refusals and in one number, so both are driven here,
offline — no server, no sockets, no bridge, no torch on the decision path.

HOW THIS REACHES THE CODE UNDER TEST. All arithmetic and every refusal live in
the `launch_sizing` module the shell script writes to disk before it uses it.
These tests extract that module's source VERBATIM from between the script's
`LAUNCH_SIZING_PY` heredoc sentinels and exec it, so what is tested is
byte-identical to what the operator runs — there is no second copy to drift.

WHAT IS PINNED, AND WHY EACH PIN EXISTS:

* **The healthy baseline CLEARS.** Every mutation test is measured against it.
  Without this anchor a check that refused unconditionally would pass every
  refusal test and block every real run.
* **Each refusal fires on its own mutation.** Where two measured quantities are
  genuinely coupled (a collapsed collection rate also changes the memory
  projection's horizon), the test asserts the expected code is PRESENT rather
  than pretending the mutation was isolated.
* **Fail-closed on absent evidence.** A check whose input is missing REFUSES
  under its own code. Absence is never read as a healthy zero — that is exactly
  how a missing measurement becomes a stale constant.
* **The sizing NEVER falls back to a constant.** Emptying the canary
  measurement must refuse, not silently produce 2773 (the derived default for
  25 pads, built from the bare-handed 285-step figure).
* **The arithmetic is pinned to numbers measured elsewhere in the project.** The
  eval-cost model reproduces the measured "100 episodes == 97 minutes" figure,
  and the episodes/hour model reproduces the M3 retry's measured ~4,640
  episodes/hour from its measured ~95-step episodes. A sizing model that cannot
  reproduce a measurement it was not fitted to is not a model.
* **Every calibration constant is pinned to its SOURCE, not to a comment**:
  `MAX_EPISODE_STEPS` to `agent.contract_config`, the collection rate and the
  decay fraction to `agent.train_config`, the drain batch to
  `distributed.learner`, the episode-budget default to `agent/train.py`'s own
  argparse.
* **The script starts no fleet.** The boot order is Paper -> bridges -> driver
  and the operator owns the first two; a self-booting launcher is free to boot
  the fleet WRONG (knockback-immune), which is the one thing nothing can check
  afterwards.
* **Bridge ports are never connect-probed.** `BridgeServer` accepts exactly ONE
  TCP client and a second connection silently destroys the first. Only the
  Minecraft port may be connected to, and that is pinned by name.
* **A refused plan leaves no argv behind.** The shell reads the argv the GATE
  wrote; if a refusal left a stale file there, a later invocation could start a
  run nothing cleared.
* **The morning table does not rank.** The two most attractive numbers in this
  project's history are both misleading, and the table's job is to say so.
"""

from __future__ import annotations

import json
import os
import re
import types
from typing import Any, Dict, List, Mapping, Optional

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT_PATH = os.path.join(REPO_ROOT, "scripts", "launch_selfplay.sh")
#: T17's script, read for ONE thing only: the key list its `build_measurements`
#: returns, which is the document this gate's whole sizing chain consumes.
CANARY_SCRIPT_PATH = os.path.join(REPO_ROOT, "scripts", "canary_selfplay.sh")


# ---------------------------------------------------------------------------
# Extract the decision logic from the shell script it ships inside.
# ---------------------------------------------------------------------------


def _script_lines() -> List[str]:
    with open(SCRIPT_PATH, "r", encoding="utf-8") as handle:
        return handle.read().splitlines()


def extract_heredoc(name: str) -> str:
    """Return the body of the ``<<'<name>'`` heredoc, verbatim.

    Asserts there is EXACTLY one opener: two copies of the sizing module in one
    script would mean the tests and the operator could be running different
    code, which is the whole failure this extraction exists to prevent.
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

    What remains is the shell that actually RUNS. The distinction matters: the
    script names ``server/setup/start-pads.sh`` many times, but every one is
    instruction text printed for the operator. An occurrence in executable
    position would be the launcher booting its own fleet.
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

    # A character scanner, not a regex: several `die` messages are MULTI-LINE
    # double-quoted strings, so quoting state has to carry across newlines.
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


def _load_sizing_module() -> types.ModuleType:
    source = extract_heredoc("LAUNCH_SIZING_PY")
    module = types.ModuleType("launch_sizing_under_test")
    module.__file__ = SCRIPT_PATH
    exec(compile(source, SCRIPT_PATH, "exec"), module.__dict__)
    return module


sizing_module = _load_sizing_module()


# ---------------------------------------------------------------------------
# Synthetic documents.
#
# The healthy fixtures are deliberately REALISTIC, so they double as a
# description of what a good night's inputs look like: an armored fight of ~132
# steps, a 25-pad fleet an hour older than the canary that probed it, and a
# smoke that held at ~120 transitions/s.
# ---------------------------------------------------------------------------

NOW = 1_755_000_000.0


def make_canary_measurements(**overrides: Any) -> Dict[str, Any]:
    """A canary measurement file, key-for-key as T17's `build_measurements` writes it.

    Copied from that function's OWN key list rather than written from memory,
    and `test_the_canary_fixture_carries_exactly_the_producers_keys` re-derives
    that list from `scripts/canary_selfplay.sh` on every run so the next rename
    fails here instead of at 20:00.

    The drift this replaced was load-bearing: the fixture carried
    `armored_mean_episode_length_eval_greedy`, a key T17 REMOVED (its own
    `notes` record the rename), so three tests drove an input production can
    never supply — and one of them was the only coverage of the eval squeeze.

    The numbers are internally consistent: 885 episodes over 1080 s at 25 pads
    is the 118.0 episodes/arena/hour recorded below, and 118 x 25 is its 25-pad
    projection. The probe's 132-step fight and the eval's 118-step one are
    DIFFERENT regimes, which is the whole reason T17 writes both.
    """
    document: Dict[str, Any] = {
        "measured_at_run": "m4_selfplay_canary",
        "max_episode_steps": 600,
        # -- the probe: both seats at the run's terminal epsilons -------------
        "armored_mean_episode_length_probe": 132.0,
        "armored_median_episode_length_probe": 128.0,
        "armored_episode_lengths_probe": [140, 128, 128],
        "probe_learner_epsilon": 0.05,
        "probe_opponent_epsilon": 0.02,
        # -- the periodic eval: one seat is the FIXED scripted yardstick ------
        "armored_mean_episode_length_eval_vs_scripted": 118.0,
        "eval_opponent": "scripted_mixed",
        "eval_grad_step": 1000,
        "eval_episodes_per_cycle": 6,
        # -- the collection rate a window converts through --------------------
        "training_episodes": 885,
        "training_grad_steps": 1200,
        "training_wall_seconds": 1080.0,
        "training_arenas": 25,
        "training_epsilon_schedule_at_end": 0.05,
        "training_epsilon_mean_at_end": 0.0072,
        "measured_episodes_per_arena_hour": 118.0,
        "projected_episodes_per_hour_at_25_pads": 2950.0,
        # -- the armored damage regime ----------------------------------------
        "armored_damage_dealt_per_episode": 9.36,
        "armored_damage_taken_per_episode": 8.1,
        "armored_full_charge_hits_dealt": [3.12, 3.12, 3.12],
        "armored_full_charge_hits_taken": [3.12, 3.12],
        "armored_cap_hit_rate": 0.08,
        # T17 writes four notes here; the gate reads none of them, so the
        # fixture carries one stand-in rather than a second copy of T17's prose
        # that could drift into a false claim of its own.
        "notes": ["the two lengths and the rate are NOT interchangeable"],
    }
    document.update(overrides)
    return document


def make_smoke_measurements(**overrides: Any) -> Dict[str, Any]:
    """A smoke measurement file as `build_smoke_measurements` writes it."""
    document: Dict[str, Any] = {
        "arenas": 25,
        "grad_steps_per_hour": 4300.0,
        "transitions_per_s": 119.0,
        "episodes_per_arena_hour": 128.0,
        "episodes_per_grad_step": 1.1,
        "pool_size": 6,
    }
    document.update(overrides)
    return document


def make_plan(**overrides: Any) -> Dict[str, Any]:
    """A launch plan input document whose every check CLEARS."""
    plan: Dict[str, Any] = {
        "plan_version": sizing_module.PLAN_VERSION,
        "now_epoch": NOW,
        "window_hours": 12.0,
        "arenas": 25,
        "run_name": "m4_selfplay",
        "opponent": "selfplay",
        "host": "127.0.0.1",
        "bridge_base_port": 5555,
        "mc_port": 25565,
        "seed": 0,
        "python_bin": "/repo/.venv/bin/python",
        "repo_root": "/repo",
        "episode_length_source": "probe",
        "warm_start": {
            "path": "/Users/diego/Documents/MinecraftRL/runs/m4.best.pt",
            "exists": True,
            "is_file": True,
            "sha256": "a" * 64,
            "bytes": 2406463,
        },
        "checkpoint": "/repo/runs/m4_selfplay.pt",
        "best_checkpoint": "/repo/runs/m4_selfplay.best.pt",
        "log_path": "/repo/runs/m4_selfplay.log",
        "pid_path": "/repo/runs/m4_selfplay.pid",
        "existing_outputs": [],
        "canary": {
            "directory": "/repo/runs/m4_selfplay_canary/canary",
            "measurements_path": "/repo/runs/m4_selfplay_canary/canary/canary_measurements.json",
            "exists": True,
            "mtime": NOW - 3600.0,
            "measurements": make_canary_measurements(),
            "analyze_exit": 0,
        },
        "smoke": {
            "measurements_path": "/repo/runs/m4_selfplay/launch/smoke_measurements.json",
            "exists": True,
            "mtime": NOW - 1800.0,
            "measurements": make_smoke_measurements(),
            "verdict_ok": True,
        },
        "fleet": {
            "host": "127.0.0.1",
            "mc_port": 25565,
            "mc_reachable": True,
            "missing_ports": [],
            "busy_ports": [],
            "listener_count": 25,
            # Every bridge is OLDER than the canary measurement, so nothing was
            # restarted since it proved the fleet takes knockback.
            "youngest_listener_age_seconds": 7200.0,
            "oldest_listener_age_seconds": 7300.0,
        },
        "overrides": {},
    }
    plan.update(overrides)
    return plan


def make_smoke_evidence(**overrides: Any) -> Dict[str, Any]:
    """A smoke evidence document whose every check CLEARS.

    2,400 s at 25 pads: 4,300 grad steps/hour and ~119 transitions/s derived
    from 2,150 episodes of 132 steps.
    """
    samples = [[float(t), 3.0e9 + 1.2e6 * t] for t in range(0, 2400, 15)]
    document: Dict[str, Any] = {
        "smoke_version": sizing_module.SMOKE_EVIDENCE_VERSION,
        "evidence_path": "/repo/runs/m4_selfplay/launch/smoke_evidence.json",
        "run_name": "m4_selfplay_smoke",
        "arenas": 25,
        "min_replay": 25000,
        "max_grad_steps": 2500,
        "wall_seconds": 2400.0,
        "window_hours": 12.0,
        "warm_start": "/abs/runs/m4.best.pt",
        "warm_start_sha256": "a" * 64,
        "driver": {
            "completed": True,
            "exit_code": 1,
            "deadline_hit": False,
            "stop_reason": "max_grad_steps",
            "episodes": 2150,
            "grad_steps": 2500,
            "checkpoints_saved": 5,
            "log_path": "/repo/runs/m4_selfplay/launch/smoke_driver.log",
        },
        "watchdog_tripped": False,
        "rss": {
            "samples": samples,
            "first_bytes": samples[0][1],
            "peak_bytes": samples[-1][1],
            "jvm_peak_bytes": 6.0e9,
        },
        "physical_memory_bytes": 68719476736.0,
        "replay_capacity": 1_000_000,
        "pool": {"ok": True, "size": 6, "snapshot_ids": [0, 1, 2, 3, 4, 5], "error": None},
        "snapshot_load": {"ok": True, "seconds": [0.04, 0.05, 0.06, 0.07], "error": None},
        "episode_length_steps": 132.0,
        "episode_length_source": "canary probe (learner eps=0.05 / opponent eps=0.02)",
    }
    document.update(overrides)
    return document


def canary_measurement_keys() -> List[str]:
    """The keys T17's `build_measurements` ACTUALLY returns, called for real.

    Extracted and exec'd the same way this file loads the sizing module, then
    invoked on an empty evidence document: every key is emitted unconditionally,
    so the returned dict IS the schema. `CANARY_VERDICT_PY` imports only the
    standard library, so this costs nothing and reaches no server.
    """
    with open(CANARY_SCRIPT_PATH, "r", encoding="utf-8") as handle:
        lines = handle.read().splitlines()
    opener = "<<'CANARY_VERDICT_PY'"
    starts = [i for i, line in enumerate(lines) if line.rstrip().endswith(opener)]
    assert len(starts) == 1, f"expected exactly one {opener}, got {starts}"
    ends = [
        i
        for i in range(starts[0] + 1, len(lines))
        if lines[i].strip() == "CANARY_VERDICT_PY"
    ]
    assert ends, f"no CANARY_VERDICT_PY terminator after line {starts[0] + 1}"
    source = "\n".join(lines[starts[0] + 1 : ends[0]]) + "\n"
    module = types.ModuleType("t17_canary_verdict")
    module.__dict__["__file__"] = CANARY_SCRIPT_PATH
    exec(  # noqa: S102 - the point is to run T17's own code, not a copy of it
        compile(source, f"{CANARY_SCRIPT_PATH}:CANARY_VERDICT_PY", "exec"),
        module.__dict__,
    )
    return sorted(module.build_measurements({}))


def refusal_codes(verdict: Any) -> List[str]:
    return [check.code for check in verdict.refusals]


def evaluate(plan: Mapping[str, Any]) -> Any:
    return sizing_module.evaluate_launch(plan)


# ---------------------------------------------------------------------------
# Extraction and shape.
# ---------------------------------------------------------------------------


def test_the_canary_fixture_carries_exactly_the_producers_keys() -> None:
    """`make_canary_measurements` must be T17's schema — all of it, nothing else.

    A fixture that invents a key lets a test drive an input production can never
    supply, and a fixture that omits one hides a field the gate should be
    reading. Both happened at once: the fixture carried
    `armored_mean_episode_length_eval_greedy` (removed by T17) while omitting
    the `armored_mean_episode_length_eval_vs_scripted` / `eval_opponent` pair
    that replaced it, so the gate read a permanently-absent field and told every
    operator, truthfully-looking, that a measurement was missing.

    Derived from `build_measurements` itself rather than typed out here, so a
    rename fails in this file instead of at 20:00 on the launch night.
    """
    assert sorted(make_canary_measurements()) == canary_measurement_keys()


def test_the_eval_sizing_reads_the_key_t17_writes() -> None:
    """The eval cost comes from the periodic-eval length, labelled by its opponent.

    The regression this pins: T19 read `..._eval_greedy` after T17 renamed it,
    so on EVERY healthy run the eval fell back to the probe's length and the
    report announced a missing measurement. The sizing stayed sane — the
    fallback is the measured probe, never a constant — which is exactly why the
    false report could have survived the night.
    """
    sizing = sizing_module.derive_sizing(make_canary_measurements(), window_hours=12.0)
    assert sizing["eval_episode_length_steps"] == 118.0
    assert sizing["eval_opponent"] == "scripted_mixed"
    assert "scripted_mixed" in sizing["eval_length_source"]
    assert "scripted yardstick" in sizing["eval_length_source"]
    # The label must not resurrect the claim T17 removed.
    assert "greedy" not in sizing["eval_length_source"]
    assert "eps=0" not in sizing["eval_length_source"]
    # ... and it travels into the report the operator reads at 20:00.
    verdict = evaluate(make_plan())
    assert "scripted_mixed" in sizing_module.format_launch_report(verdict, make_plan())


def test_a_cycle_with_no_eval_length_falls_back_and_says_so_truthfully() -> None:
    """Absence is reported as THIS CYCLE's, not as a missing field.

    T17 writes the eval length only when the cycle it probed produced one, so
    its absence is an ordinary outcome. Saying "no greedy-eval length in the
    canary measurement" instead named a field that no longer exists and sent the
    operator looking for a bug in the producer.
    """
    sizing = sizing_module.derive_sizing(
        make_canary_measurements(
            armored_mean_episode_length_eval_vs_scripted=None, eval_opponent=None
        ),
        window_hours=12.0,
    )
    # The fallback is the SAME measured armored length, never a constant.
    assert sizing["eval_episode_length_steps"] == sizing["episode_length_steps"]
    assert sizing["eval_opponent"] is None
    assert "for this cycle" in sizing["eval_length_source"]
    assert "probe" in sizing["eval_length_source"]
    assert "greedy" not in sizing["eval_length_source"]


def test_the_named_episode_length_sources_all_exist_in_the_measurement() -> None:
    """Every source key the sizing offers must be one T17 actually writes."""
    produced = set(canary_measurement_keys())
    for source, (key, label) in sizing_module.EPISODE_LENGTH_SOURCES.items():
        assert key in produced, f"{source} points at {key}, which T17 never writes"
        assert label.strip()


def test_sizing_module_is_stdlib_only() -> None:
    """No torch, no numpy, no project imports on the decision path.

    The gate must be able to judge on a machine where the training stack is
    broken — which is exactly the machine where a launch is most likely to be
    attempted anyway.
    """
    source = extract_heredoc("LAUNCH_SIZING_PY")
    imports = set(re.findall(r"^\s*(?:import|from)\s+([A-Za-z_][A-Za-z0-9_.]*)", source, re.M))
    allowed = {"__future__", "json", "math", "os", "shlex", "sys", "typing"}
    assert imports <= allowed, f"unexpected imports on the decision path: {imports - allowed}"


def test_the_script_starts_no_fleet() -> None:
    """`launch_selfplay.sh` never boots Paper or the bridges.

    The boot order is Paper -> bridges -> driver, and the operator owns the
    first two. A self-booting launcher is free to boot the fleet WITHOUT
    DUMMY_KNOCKBACK_IMMUNE=false, which is precisely the condition nothing can
    verify from a running process on macOS.
    """
    shell = executable_shell()
    for launcher in ("start-pads.sh", "start.sh", "setup.sh", "distributed.launcher"):
        assert launcher not in shell, f"{launcher} appears in executable position"


def test_only_the_minecraft_port_is_connect_probed() -> None:
    """Bridges are inspected with `lsof`; only Paper is ever connected to.

    BridgeServer accepts exactly ONE TCP client and resolves a second connection
    by destroying the incumbent. Four outages in this project came from that.
    """
    raw = _script_lines()
    assert "\n".join(raw).count("socket.create_connection") == 1
    # Scanned on the RAW text: executable_shell() strips quoted strings, which
    # is where the port argument lives.
    invocations = [
        line
        for line in raw
        if "mc_connect_probe" in line
        and "mc_connect_probe()" not in line
        and not line.lstrip().startswith("#")
    ]
    assert invocations, "expected at least one mc_connect_probe call"
    for line in invocations:
        assert "MC_PORT" in line, f"connect probe against a non-Minecraft port: {line}"
    shell = executable_shell()
    for helper in ("listener_pids", "established_peers"):
        assert helper in shell, f"{helper} must remain the bridge-port inspector"


def test_the_launch_runs_detached() -> None:
    """The driver survives the session that starts it, and its pid is recorded."""
    shell = executable_shell()
    assert "nohup" in shell
    assert "RUN_PID_FILE" in shell


# ---------------------------------------------------------------------------
# Calibration constants, pinned to their SOURCES rather than to comments.
# ---------------------------------------------------------------------------


def test_max_episode_steps_matches_the_contract_constant() -> None:
    from agent.contract_config import MAX_EPISODE_STEPS

    assert sizing_module.MAX_EPISODE_STEPS == MAX_EPISODE_STEPS == 600


def test_collection_rate_and_decay_fraction_match_train_config() -> None:
    from agent import train_config

    assert (
        sizing_module.MEASURED_PER_ARENA_TRANSITIONS_PER_S
        == train_config.MEASURED_PER_ARENA_TRANSITIONS_PER_S
    )
    assert (
        sizing_module.EPS_DECAY_FRACTION_OF_RUN == train_config.EPS_DECAY_FRACTION_OF_RUN
    )
    # The stale figure is kept by NAME so a refusal can say what it is refusing
    # to use. It must stay equal to the constant it is warning about.
    assert (
        sizing_module.STALE_EPISODE_STEPS_SCRIPTED_BARE
        == train_config.MEASURED_MEAN_EPISODE_STEPS
    )


def test_aggregate_rate_is_the_measured_sweep_figure() -> None:
    """25 x 4.8782 == 121.955 transitions/s, the 600 s confirm at N=25."""
    assert sizing_module.MEASURED_AGGREGATE_TRANSITIONS_PER_S == pytest.approx(
        121.955, abs=1e-3
    )


def test_drain_batch_matches_the_learner() -> None:
    """The queue-depth proxy is only meaningful against the real drain batch."""
    from distributed import learner

    assert sizing_module.LEARNER_DRAIN_BATCH == learner._DEFAULT_DRAIN_BATCH


def test_episode_budget_default_matches_the_cli() -> None:
    """The trap this gate exists to catch is argparse's own default."""
    pytest.importorskip("torch")
    from agent.train import _build_parser

    defaults = {action.dest: action.default for action in _build_parser()._actions}
    assert sizing_module.DEFAULT_MAX_EPISODES == defaults["max_episodes"]
    assert (
        sizing_module.DEFAULT_REFERENCE_EVAL_EPISODES
        == defaults["reference_eval_episodes"]
    )


def test_max_pinned_references_matches_the_promotion_schedule() -> None:
    """Snapshot 0 at seed plus one per promotion step."""
    from agent.train_config import TrainConfig

    assert sizing_module.MAX_PINNED_REFERENCES == 1 + len(
        TrainConfig().reference_promote_grad_steps
    )


def test_m3_retry_grad_step_rate_matches_its_own_numbers() -> None:
    """30,000 gradient steps in 6h34m == 23,640 s."""
    measured = 30_000 / (6 * 3600 + 34 * 60) * 3600
    assert sizing_module.M3_RETRY_GRAD_STEPS_PER_HOUR == pytest.approx(measured, rel=0.01)


def test_expected_run_name_is_not_the_completed_runs_name() -> None:
    assert sizing_module.EXPECTED_RUN_NAME == "m4_selfplay"
    assert "m4" in sizing_module.FORBIDDEN_RUN_NAMES
    assert "m4" in sizing_module.BARE_HANDED_RUNS


# ---------------------------------------------------------------------------
# parse_etime — macOS `ps` has no `etimes` keyword, so the age of a bridge
# process arrives as a string.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("05:09", 309.0),
        ("01:02:03", 3723.0),
        ("2-01:02:03", 176523.0),
        ("00:00", 0.0),
        ("  01:00  ", 60.0),
        ("", None),
        ("nonsense", None),
        ("1:2:3:4", None),
        ("x-01:02:03", None),
        (None, None),
        (12345, None),
    ],
)
def test_parse_etime(text: Any, expected: Optional[float]) -> None:
    assert sizing_module.parse_etime(text) == expected


# ---------------------------------------------------------------------------
# PART 2 — the sizing arithmetic.
# ---------------------------------------------------------------------------


def test_the_eval_cost_model_reproduces_the_measured_97_minutes() -> None:
    """100 serial eval episodes at the bare-handed 285-step length == 97 min.

    That figure was MEASURED (`docs/plans/STATUS-2026-08-16.md`: "One eval at
    --eval-episodes 100 | 'about 30 min' | 97 minutes"). A cost model that
    cannot reproduce a number it was not fitted to is not a model, and this one
    is the whole basis for sizing `--eval-episodes`.
    """
    sizing = sizing_module.derive_sizing(
        make_canary_measurements(
            armored_mean_episode_length_eval_vs_scripted=(
                sizing_module.STALE_EPISODE_STEPS_SCRIPTED_BARE
            )
        ),
        window_hours=12.0,
    )
    minutes = 100 * sizing["eval_episode_seconds"] / 60.0
    assert minutes == pytest.approx(97.0, abs=1.0)


def test_the_episode_rate_model_reproduces_the_m3_retry() -> None:
    """~95-step episodes at 25 pads must give the measured ~4,640 episodes/hour.

    The M3 retry collected 30,503 episodes in 6h34m. Feeding this model the
    episode length that run measured has to return the episode rate that run
    measured — otherwise the conversion is fitted to nothing.
    """
    sizing = sizing_module.derive_sizing(
        make_canary_measurements(
            armored_mean_episode_length_probe=sizing_module.STALE_EPISODE_STEPS_M3_RETRY
        ),
        window_hours=12.0,
    )
    measured_rate = 30_503 / (6 * 3600 + 34 * 60) * 3600
    assert sizing["episodes_per_hour"] == pytest.approx(measured_rate, rel=0.02)


def test_the_decay_window_is_exactly_the_target_fraction() -> None:
    sizing = sizing_module.derive_sizing(make_canary_measurements(), window_hours=12.0)
    assert sizing["eps_decay_fraction"] == pytest.approx(0.15, abs=0.001)
    assert sizing["eps_decay_source"] == "derived from the canary measurement"
    # The chain, spelled out: 25 x 4.8782 x 3600 / 132 steps x 12 h x 15%.
    expected = round(
        0.15
        * (25 * sizing_module.MEASURED_PER_ARENA_TRANSITIONS_PER_S * 3600.0 / 132.0)
        * 12.0
    )
    assert sizing["eps_decay_episodes"] == expected


def test_a_longer_window_scales_the_decay_window_linearly() -> None:
    twelve = sizing_module.derive_sizing(make_canary_measurements(), window_hours=12.0)
    twentyfour = sizing_module.derive_sizing(make_canary_measurements(), window_hours=24.0)
    assert twentyfour["eps_decay_episodes"] == pytest.approx(
        2 * twelve["eps_decay_episodes"], rel=0.001
    )
    # ... and the FRACTION is unchanged, which is the point of sizing by it.
    assert twentyfour["eps_decay_fraction"] == pytest.approx(
        twelve["eps_decay_fraction"], abs=1e-6
    )


def test_the_sizing_never_falls_back_to_a_constant() -> None:
    """An empty measurement produces NO number, and the gate refuses.

    The two available constants are 285 (bare-handed, unarmed opponent) and 95
    (also bare-handed). `eps_decay_episodes_for(25)` would happily return 2773
    from the first of them; this must not be reachable through absence.
    """
    sizing = sizing_module.derive_sizing({}, window_hours=12.0)
    assert sizing["episode_length_steps"] is None
    assert sizing["eps_decay_episodes"] is None

    plan = make_plan()
    plan["canary"]["measurements"] = {}
    codes = refusal_codes(evaluate(plan))
    assert "EPISODE_LENGTH_UNMEASURED" in codes
    assert "EPS_DECAY_ABSURD" in codes


def test_the_derived_window_differs_from_the_stale_default() -> None:
    """The measurement-driven number must not coincide with the stale one.

    If it did, nothing about this gate would be observable: the run would look
    identical whether it read a measurement or a constant.
    """
    from agent.train_config import eps_decay_episodes_for

    sizing = sizing_module.derive_sizing(make_canary_measurements(), window_hours=12.0)
    assert sizing["eps_decay_episodes"] != eps_decay_episodes_for(25)
    assert sizing["eps_decay_episodes"] > 2 * eps_decay_episodes_for(25)


def test_the_driver_startup_line_disagreement_is_reported() -> None:
    """The run's own epsilon line divides by the stale 285-step projection.

    Reporting what it WILL say is not decoration: an operator who sees 32% at
    20:00 and expects 15% will otherwise assume the sizing is broken and
    restart the night by hand.
    """
    sizing = sizing_module.derive_sizing(make_canary_measurements(), window_hours=12.0)
    from agent.train_config import projected_episodes

    assert sizing["stale_projected_episodes"] == pytest.approx(
        projected_episodes(25), rel=1e-9
    )
    assert sizing["stale_reported_fraction"] != pytest.approx(
        sizing["eps_decay_fraction"], abs=0.01
    )


def test_the_eval_cycle_lands_inside_the_target_band() -> None:
    sizing = sizing_module.derive_sizing(make_canary_measurements(), window_hours=12.0)
    assert 30.0 <= sizing["eval_cycle_minutes_worst"] <= 45.0
    # The gauntlet is sized first and keeps T13's per-reference default.
    assert sizing["reference_eval_episodes"] == sizing_module.DEFAULT_REFERENCE_EVAL_EPISODES
    assert sizing["eval_episodes"] < 100


def test_expensive_episodes_squeeze_the_scripted_track_not_the_gauntlet() -> None:
    """A 550-step eval against the scripted yardstick shrinks the yardstick only.

    The gauntlet is the checkpoint-SELECTION input. Losing precision there
    changes which net ships; losing it on the yardstick only makes one logged
    curve noisier.

    The squeeze is driven through `armored_mean_episode_length_eval_vs_scripted`
    — the key T17 actually writes — so the path exercised here is one a real
    measurement can reach. A near-cap eval is not in tension with the probe's
    8% cap-hit rate below it: the probe is both seats at eps=0.05/0.02 and this
    is one seat against the fixed scripted bot, which is exactly why T17 keeps
    the two lengths apart.
    """
    sizing = sizing_module.derive_sizing(
        make_canary_measurements(armored_mean_episode_length_eval_vs_scripted=550.0),
        window_hours=12.0,
    )
    assert sizing["eval_episodes"] == 10
    assert sizing["reference_eval_episodes"] >= 5


def test_the_eval_cadence_targets_the_duty_ceiling() -> None:
    sizing = sizing_module.derive_sizing(
        make_canary_measurements(), window_hours=12.0, grad_steps_per_hour=4300.0
    )
    assert sizing["eval_duty"] <= sizing_module.DEFAULT_THRESHOLDS["max_eval_duty"]
    assert sizing["eval_every_grad_steps"] % 500 == 0
    assert sizing["eval_every_grad_steps"] >= 1000


def test_budgets_exceed_the_projection_by_the_margin() -> None:
    sizing = sizing_module.derive_sizing(
        make_canary_measurements(), window_hours=12.0, grad_steps_per_hour=4300.0
    )
    margin = sizing_module.DEFAULT_THRESHOLDS["budget_margin"]
    assert sizing["max_episodes"] >= sizing["projected_episodes"] * margin - 1
    assert sizing["max_grad_steps"] >= sizing["projected_grad_steps"] * margin - 1
    # ... and they are far above the argparse default that ended M3 early.
    assert sizing["max_episodes"] > sizing_module.DEFAULT_MAX_EPISODES


def test_the_smoke_rate_is_preferred_over_the_bare_handed_one() -> None:
    """An ARMORED learner rate must displace the M3 retry's bare-handed figure."""
    plan = make_plan()
    plan["smoke"]["measurements"] = make_smoke_measurements(grad_steps_per_hour=3300.0)
    sizing = evaluate(plan).facts["sizing"]
    assert sizing["grad_steps_per_hour"] == 3300.0
    assert "smoke" in sizing["grad_steps_per_hour_source"]

    plan["smoke"]["measurements"] = make_smoke_measurements(grad_steps_per_hour=None)
    sizing = evaluate(plan).facts["sizing"]
    assert sizing["grad_steps_per_hour"] == sizing_module.M3_RETRY_GRAD_STEPS_PER_HOUR
    assert "bare-handed" in sizing["grad_steps_per_hour_source"]


def test_the_arithmetic_is_shown_not_just_computed() -> None:
    """Every step of the chain appears in the report the operator reads."""
    sizing = sizing_module.derive_sizing(make_canary_measurements(), window_hours=12.0)
    text = "\n".join(sizing_module.sizing_arithmetic_lines(sizing))
    assert "measured armored mean episode length" in text
    assert "episodes/hour at 25 pads" in text
    assert "episodes in the 12 h window" in text
    assert "--eps-decay-episodes" in text
    assert "142%" in text
    assert str(sizing["eps_decay_episodes"]) in text


# ---------------------------------------------------------------------------
# PART 3 — the launch gate. The anchor first, then one mutation per refusal.
# ---------------------------------------------------------------------------


def test_the_healthy_plan_clears() -> None:
    verdict = evaluate(make_plan())
    assert verdict.ok, refusal_codes(verdict)


def test_every_check_reports_a_reason_and_a_next_step() -> None:
    """A refusal that only prints its code is a refusal nobody can act on."""
    plan = make_plan()
    plan["canary"]["exists"] = False
    plan["canary"]["measurements"] = None
    plan["warm_start"]["path"] = ""
    plan["checkpoint"] = ""
    plan["opponent"] = "scripted"
    verdict = evaluate(plan)
    assert verdict.refusals
    for check in verdict.refusals:
        assert check.why.strip(), check.code
        assert check.check.strip(), check.code


def test_an_unknown_plan_version_refuses_wholesale() -> None:
    plan = make_plan(plan_version=999)
    verdict = evaluate(plan)
    assert refusal_codes(verdict) == ["PLAN_VERSION"]


def test_missing_canary_measurement_refuses() -> None:
    plan = make_plan()
    plan["canary"]["exists"] = False
    plan["canary"]["measurements"] = None
    codes = refusal_codes(evaluate(plan))
    assert "CANARY_MEASUREMENTS_MISSING" in codes


@pytest.mark.parametrize(
    "canary_mutation, fleet_mutation, expected_detail",
    [
        pytest.param(
            {"mtime": NOW - 13 * 3600.0},
            # The fleet is OLDER than the measurement, so the bridge-restart
            # clause CANNOT fire and only the age limit can produce this code.
            # The fixture's 2 h-old bridges trip that clause too, under the SAME
            # code, so without this the age branch could be deleted outright and
            # the case would still go green.
            {
                "youngest_listener_age_seconds": 20 * 3600.0,
                "oldest_listener_age_seconds": 21 * 3600.0,
            },
            "measurement is 13.0 h old (limit 12 h)",
            id="older_than_the_12_h_allowance",
        ),
        pytest.param(
            {"mtime": NOW + 600.0},
            {},
            "dated 0.17 h in the FUTURE",
            id="dated_in_the_future",
        ),
        pytest.param(
            {"mtime": None},
            {},
            "mtime=None",
            id="age_unknowable",
        ),
    ],
)
def test_a_stale_canary_measurement_refuses(
    canary_mutation: Dict[str, Any],
    fleet_mutation: Dict[str, Any],
    expected_detail: str,
) -> None:
    """Each way freshness can fail must refuse ON ITS OWN CLAUSE.

    All three share one code, so asserting the code alone is not coverage: the
    age case used to pass because the fixture's 2 h-old bridges tripped the
    bridge-restart clause underneath it. The detail is asserted for that reason
    — it is the only thing that distinguishes which clause spoke, and it carries
    the 12 h threshold itself, so widening the limit fails here too.

    The age clause is the one that catches "canary from Monday, fleet booted
    Sunday and never restarted since" — precisely the case the bridge-age clause
    is blind to.
    """
    plan = make_plan()
    plan["canary"].update(canary_mutation)
    plan["fleet"].update(fleet_mutation)
    verdict = evaluate(plan)
    assert "CANARY_MEASUREMENTS_STALE" in refusal_codes(verdict)
    stale = next(c for c in verdict.refusals if c.code == "CANARY_MEASUREMENTS_STALE")
    assert expected_detail in stale.detail


def test_a_bridge_restarted_since_the_canary_refuses() -> None:
    """The knockback proof covers only the processes the canary probed.

    DUMMY_KNOCKBACK_IMMUNE cannot be read back from a running process on macOS
    and start-pads.sh passes it as an env var rather than argv, so a pad started
    after the canary is a pad nothing has verified.
    """
    plan = make_plan()
    # The canary measurement is an hour old; make one bridge 10 minutes old.
    plan["fleet"]["youngest_listener_age_seconds"] = 600.0
    verdict = evaluate(plan)
    assert "CANARY_MEASUREMENTS_STALE" in refusal_codes(verdict)
    stale = next(c for c in verdict.refusals if c.code == "CANARY_MEASUREMENTS_STALE")
    assert "DUMMY_KNOCKBACK_IMMUNE" in stale.why


@pytest.mark.parametrize("exit_code", [1, 2, None])
def test_a_canary_that_did_not_clear_refuses(exit_code: Optional[int]) -> None:
    plan = make_plan()
    plan["canary"]["analyze_exit"] = exit_code
    codes = refusal_codes(evaluate(plan))
    assert "CANARY_NOT_GREEN" in codes


@pytest.mark.parametrize(
    "length", [None, 0.0, -5.0, float("nan"), "132", float("inf")]
)
def test_an_unusable_episode_length_refuses(length: Any) -> None:
    plan = make_plan()
    plan["canary"]["measurements"]["armored_mean_episode_length_probe"] = length
    codes = refusal_codes(evaluate(plan))
    assert "EPISODE_LENGTH_UNMEASURED" in codes


@pytest.mark.parametrize("length", [1.0, 4.9, 545.0, 600.0])
def test_an_implausible_episode_length_refuses(length: float) -> None:
    """Too short is a reset loop; too long is a cap-hit draw wearing a costume."""
    plan = make_plan()
    measurements = plan["canary"]["measurements"]
    measurements["armored_mean_episode_length_probe"] = length
    # Keep the cross-check consistent so this test isolates ONE refusal.
    measurements["measured_episodes_per_arena_hour"] = (
        sizing_module.MEASURED_PER_ARENA_TRANSITIONS_PER_S * 3600.0 / length
    )
    codes = refusal_codes(evaluate(plan))
    assert "EPISODE_LENGTH_IMPLAUSIBLE" in codes


def test_disagreeing_episode_length_measurements_refuse() -> None:
    """Two measurements of one stream that disagree mean one of them is wrong."""
    plan = make_plan()
    # 132 steps from the probe, but the collection rate implies ~440.
    plan["canary"]["measurements"]["measured_episodes_per_arena_hour"] = 40.0
    codes = refusal_codes(evaluate(plan))
    assert "EPISODE_LENGTH_DISAGREEMENT" in codes


def test_a_missing_cross_check_refuses_rather_than_passing() -> None:
    plan = make_plan()
    plan["canary"]["measurements"]["measured_episodes_per_arena_hour"] = None
    codes = refusal_codes(evaluate(plan))
    assert "EPISODE_LENGTH_DISAGREEMENT" in codes


def test_the_142_percent_decay_window_refuses() -> None:
    """The exact historical failure: a window spanning more than the run.

    A previous run shipped `eps_decay_episodes` covering 142% of a 12 h run, so
    epsilon never finished decaying and sat near 0.25 all night.
    """
    plan = make_plan()
    sizing = evaluate(plan).facts["sizing"]
    plan["overrides"] = {"eps_decay_episodes": int(sizing["projected_episodes"] * 1.42)}
    verdict = evaluate(plan)
    assert "EPS_DECAY_ABSURD" in refusal_codes(verdict)
    absurd = next(c for c in verdict.refusals if c.code == "EPS_DECAY_ABSURD")
    assert "142%" in absurd.why


def test_a_single_arena_sized_decay_window_refuses() -> None:
    """The other half of the same bug: every pad claims from ONE counter."""
    from agent.train_config import eps_decay_episodes_for

    plan = make_plan(overrides={"eps_decay_episodes": eps_decay_episodes_for(1)})
    codes = refusal_codes(evaluate(plan))
    assert "EPS_DECAY_ABSURD" in codes


@pytest.mark.parametrize(
    "mutation",
    [
        {"exists": False, "measurements": None},
        {"verdict_ok": False},
        {"measurements": {"arenas": 4}},
    ],
)
def test_an_absent_or_narrow_or_refused_smoke_refuses(mutation: Dict[str, Any]) -> None:
    plan = make_plan()
    plan["smoke"].update(mutation)
    codes = refusal_codes(evaluate(plan))
    assert "SMOKE_NOT_CLEARED" in codes


@pytest.mark.parametrize(
    "mutation",
    [
        {"mc_reachable": False},
        {"missing_ports": [5570, 5571]},
        {"busy_ports": [5555]},
        {"listener_count": 24},
    ],
)
def test_an_unready_fleet_refuses(mutation: Dict[str, Any]) -> None:
    plan = make_plan()
    plan["fleet"].update(mutation)
    codes = refusal_codes(evaluate(plan))
    assert "FLEET_NOT_READY" in codes


def test_an_occupied_bridge_port_says_why_it_matters() -> None:
    plan = make_plan()
    plan["fleet"]["busy_ports"] = [5560]
    verdict = evaluate(plan)
    busy = next(c for c in verdict.refusals if c.code == "FLEET_NOT_READY")
    assert "DESTROYS" in busy.why


@pytest.mark.parametrize(
    "mutation",
    [
        {"path": ""},
        {"path": "runs/m4.best.pt"},  # relative: the detached driver has another cwd
        {"is_file": False},
        {"sha256": ""},
        {"sha256": "A" * 64},  # not lowercase
        {"sha256": "abc"},
    ],
)
def test_an_unusable_warm_start_refuses(mutation: Dict[str, Any]) -> None:
    plan = make_plan()
    plan["warm_start"].update(mutation)
    codes = refusal_codes(evaluate(plan))
    assert "WARM_START_UNUSABLE" in codes


@pytest.mark.parametrize("run_name", ["m4", "m3", "m2_multi", ""])
def test_a_colliding_run_name_refuses(run_name: str) -> None:
    """`runs/m4.*` is a completed run AND this run's warm-start source."""
    plan = make_plan(run_name=run_name)
    codes = refusal_codes(evaluate(plan))
    assert "RUN_NAME_COLLISION" in codes


def test_existing_outputs_refuse() -> None:
    plan = make_plan(existing_outputs=["/repo/runs/m4_selfplay.pt"])
    codes = refusal_codes(evaluate(plan))
    assert "RUN_NAME_COLLISION" in codes


def test_a_warm_start_inside_the_runs_namespace_refuses() -> None:
    plan = make_plan(run_name="m4_selfplay")
    plan["warm_start"]["path"] = "/repo/runs/m4_selfplay.best.pt"
    codes = refusal_codes(evaluate(plan))
    assert "RUN_NAME_COLLISION" in codes


def test_best_checkpoint_alone_refuses() -> None:
    """The combination that disables BOTH the periodic and the final save."""
    plan = make_plan(checkpoint="")
    verdict = evaluate(plan)
    assert "CHECKPOINT_UNSAFE" in refusal_codes(verdict)
    unsafe = next(c for c in verdict.refusals if c.code == "CHECKPOINT_UNSAFE")
    assert "final save is a no-op" in unsafe.why


def test_one_path_for_both_checkpoints_refuses() -> None:
    plan = make_plan(best_checkpoint="/repo/runs/m4_selfplay.pt")
    codes = refusal_codes(evaluate(plan))
    assert "CHECKPOINT_UNSAFE" in codes


def test_an_hour_long_eval_cycle_refuses() -> None:
    """100 serial episodes at the measured cost is not a 30-45 minute cycle."""
    plan = make_plan(overrides={"eval_episodes": 100, "reference_eval_episodes": 20})
    codes = refusal_codes(evaluate(plan))
    assert "EVAL_CYCLE_TOO_LONG" in codes


def test_an_uncomputable_eval_cycle_refuses() -> None:
    plan = make_plan()
    measurements = plan["canary"]["measurements"]
    measurements["armored_mean_episode_length_probe"] = None
    measurements["armored_mean_episode_length_eval_vs_scripted"] = None
    codes = refusal_codes(evaluate(plan))
    assert "EVAL_CYCLE_TOO_LONG" in codes


def test_a_too_tight_eval_cadence_refuses() -> None:
    """Evals every 1000 grad steps against a 40-minute cycle is an eval harness."""
    plan = make_plan(overrides={"eval_every_grad_steps": 1000})
    codes = refusal_codes(evaluate(plan))
    assert "EVAL_CADENCE_TOO_TIGHT" in codes


def test_the_argparse_default_episode_budget_refuses() -> None:
    """`--max-episodes` omitted ends a 25-pad armored night in a couple of hours."""
    plan = make_plan(overrides={"max_episodes": sizing_module.DEFAULT_MAX_EPISODES})
    verdict = evaluate(plan)
    assert "BUDGET_ENDS_EARLY" in refusal_codes(verdict)
    early = next(c for c in verdict.refusals if c.code == "BUDGET_ENDS_EARLY")
    assert "DEFAULT" in early.detail


def test_a_short_grad_step_budget_refuses() -> None:
    plan = make_plan(overrides={"max_grad_steps": 5000})
    codes = refusal_codes(evaluate(plan))
    assert "BUDGET_ENDS_EARLY" in codes


@pytest.mark.parametrize("opponent", ["dummy", "scripted", None])
def test_a_non_selfplay_opponent_refuses(opponent: Optional[str]) -> None:
    plan = make_plan(opponent=opponent)
    codes = refusal_codes(evaluate(plan))
    assert "OPPONENT_NOT_SELFPLAY" in codes


# ---------------------------------------------------------------------------
# The argv the gate produces. A flag that goes missing between the check and
# the command line is a gate that checked something the run did not do.
# ---------------------------------------------------------------------------


def cleared_argv(**plan_overrides: Any) -> List[str]:
    plan = make_plan(**plan_overrides)
    verdict = evaluate(plan)
    assert verdict.ok, refusal_codes(verdict)
    return list(verdict.facts["argv"])


def flag_value(argv: List[str], flag: str) -> Optional[str]:
    return argv[argv.index(flag) + 1] if flag in argv else None


def test_the_argv_carries_every_mandatory_flag() -> None:
    argv = cleared_argv()
    assert flag_value(argv, "--opponent") == "selfplay"
    assert flag_value(argv, "--run-name") == "m4_selfplay"
    assert flag_value(argv, "--warm-start").startswith("/")
    assert len(flag_value(argv, "--warm-start-sha256")) == 64
    assert flag_value(argv, "--checkpoint") == "/repo/runs/m4_selfplay.pt"
    assert flag_value(argv, "--best-checkpoint") == "/repo/runs/m4_selfplay.best.pt"
    assert flag_value(argv, "--snapshot-sampling") == "pfsp"
    assert flag_value(argv, "--arenas") == "25"
    for flag in (
        "--eps-decay-episodes",
        "--eval-episodes",
        "--reference-eval-episodes",
        "--eval-every-grad-steps",
        "--checkpoint-every-grad-steps",
        "--max-episodes",
        "--max-grad-steps",
    ):
        assert flag in argv, flag
        assert int(flag_value(argv, flag)) > 0, flag


def test_the_argv_never_carries_best_checkpoint_without_checkpoint() -> None:
    argv = cleared_argv()
    assert ("--best-checkpoint" in argv) <= ("--checkpoint" in argv)
    assert flag_value(argv, "--checkpoint") != flag_value(argv, "--best-checkpoint")


def test_the_argv_uses_the_derived_decay_window() -> None:
    plan = make_plan()
    verdict = evaluate(plan)
    assert flag_value(list(verdict.facts["argv"]), "--eps-decay-episodes") == str(
        verdict.facts["sizing"]["eps_decay_episodes"]
    )


def test_the_argv_omits_flags_whose_config_default_is_the_decision() -> None:
    """T11a owns these; restating them here would be a second place to drift."""
    argv = cleared_argv()
    for flag in (
        "--opponent-epsilon",
        "--elo-k",
        "--elo-initial",
        "--reference-promote-grad-steps",
        "--warm-start-eps-start",
        "--snapshot-every-grad-steps",
    ):
        assert flag not in argv, flag


def test_an_operator_snapshot_cadence_reaches_the_argv() -> None:
    argv = cleared_argv(overrides={"snapshot_every_grad_steps": 2000})
    assert flag_value(argv, "--snapshot-every-grad-steps") == "2000"


def test_the_argv_parses_against_the_real_cli() -> None:
    """Every flag and value is one `agent.train` actually accepts.

    This is the check that catches a typo'd flag name: argparse would exit(2)
    in the first second of the night, after the fleet was booted and the
    operator had gone to bed.
    """
    pytest.importorskip("torch")
    from agent.train import _build_parser

    args = _build_parser().parse_args(cleared_argv())
    assert args.opponent == "selfplay"
    assert args.run_name == "m4_selfplay"
    assert args.arenas == 25
    assert args.checkpoint and args.best_checkpoint
    assert args.eps_decay_episodes and args.eps_decay_episodes > 0


def test_the_argv_builds_a_config_the_dataclass_accepts() -> None:
    """The values survive TrainConfig's own `__post_init__` validation.

    AC14's "selfplay requires warm_start" and the epsilon-ordering rules live
    there, so a plan that clears this gate but not the dataclass would still
    die at startup.
    """
    pytest.importorskip("torch")
    from agent.train import _build_parser, _config_from_args

    args = _build_parser().parse_args(cleared_argv())
    cfg = _config_from_args(args)
    assert cfg.opponent == "selfplay"
    assert cfg.warm_start and cfg.warm_start_sha256
    assert cfg.eps_decay_episodes == int(
        flag_value(cleared_argv(), "--eps-decay-episodes")
    )


def test_the_rendered_command_is_detached_and_records_its_pid() -> None:
    plan = make_plan()
    command = evaluate(plan).facts["command"]
    assert "nohup" in command
    assert command.rstrip().endswith("/repo/runs/m4_selfplay.pid")
    assert "2>&1 &" in command
    assert "/repo/runs/m4_selfplay.log" in command
    assert "-m agent.train" in command


def test_the_fleet_boot_command_carries_the_knockback_flag() -> None:
    """The one place DUMMY_KNOCKBACK_IMMUNE=false belongs.

    The driver derives its own launcher setting from `cfg.opponent == "dummy"`,
    so only the pads booted BEFORE the run are at risk — which are exactly the
    ones this command starts.
    """
    plan = make_plan()
    command = evaluate(plan).facts["fleet_boot_command"]
    assert command.startswith("DUMMY_KNOCKBACK_IMMUNE=false ")
    assert "--pads 25" in command


def test_the_report_shows_the_command_only_when_cleared() -> None:
    plan = make_plan()
    assert "THE COMMAND" in sizing_module.format_launch_report(evaluate(plan), plan)
    plan["opponent"] = "scripted"
    refused = sizing_module.format_launch_report(evaluate(plan), plan)
    assert "THE COMMAND" not in refused
    assert "REFUSED" in refused


# ---------------------------------------------------------------------------
# PART 1 — the 25-pad smoke. Anchor first, then one mutation per refusal.
# ---------------------------------------------------------------------------


def smoke_codes(**mutations: Any) -> List[str]:
    evidence = make_smoke_evidence()
    for key, value in mutations.items():
        if isinstance(value, dict) and isinstance(evidence.get(key), dict):
            evidence[key].update(value)
        else:
            evidence[key] = value
    return refusal_codes(sizing_module.evaluate_smoke(evidence))


def test_the_healthy_smoke_is_green() -> None:
    verdict = sizing_module.evaluate_smoke(make_smoke_evidence())
    assert verdict.ok, refusal_codes(verdict)


def test_the_smoke_measurements_are_arithmetic_over_the_evidence() -> None:
    measured = sizing_module.build_smoke_measurements(make_smoke_evidence())
    assert measured["grad_steps_per_hour"] == pytest.approx(2500 / (2400 / 3600))
    assert measured["transitions_per_s"] == pytest.approx(2150 * 132.0 / 2400.0)
    assert measured["episodes_per_grad_step"] == pytest.approx(2150 / 2500)
    assert measured["learner_drain_batch"] == sizing_module.LEARNER_DRAIN_BATCH
    # RSS growth is a least-squares slope, so it recovers the fixture's exact
    # 1.2 MB/s ramp rather than whichever two samples happen to bracket it.
    assert measured["rss_growth_bytes_per_s"] == pytest.approx(1.2e6, rel=1e-6)
    # The projection stops at a FULL replay buffer, not at the end of the night.
    assert measured["rss_growth_horizon_seconds"] < 12 * 3600 - 2400


def test_the_smoke_records_that_transitions_are_derived() -> None:
    """The one number here that is not measured directly must say so."""
    measured = sizing_module.build_smoke_measurements(make_smoke_evidence())
    notes = " ".join(measured["notes"])
    assert "transitions_per_s is DERIVED" in notes
    assert measured["episode_length_source"]


def test_an_unknown_smoke_version_refuses_wholesale() -> None:
    assert smoke_codes(smoke_version=999) == ["SMOKE_EVIDENCE_VERSION"]


def test_a_driver_that_never_finished_refuses() -> None:
    assert "SMOKE_DRIVER_FAILED" in smoke_codes(driver={"completed": False})


def test_the_driver_exit_code_alone_does_not_refuse() -> None:
    """A healthy self-play smoke exits 1: passed_m2 is the M2 DUMMY gate.

    A gate keyed on the exit code would refuse every good run.
    """
    for exit_code in (0, 1, 2, 130):
        verdict = sizing_module.evaluate_smoke(
            make_smoke_evidence(
                driver=dict(make_smoke_evidence()["driver"], exit_code=exit_code)
            )
        )
        assert verdict.ok, (exit_code, refusal_codes(verdict))


@pytest.mark.parametrize("arenas", [None, 4, 16, 24])
def test_a_narrow_smoke_refuses(arenas: Any) -> None:
    assert "SMOKE_NOT_FULL_WIDTH" in smoke_codes(arenas=arenas)


@pytest.mark.parametrize("wall", [None, 0.0, 599.0])
def test_a_short_smoke_refuses(wall: Any) -> None:
    """Below the 25,000-transition warm-up there is no steady state to measure."""
    assert "SMOKE_TOO_SHORT" in smoke_codes(wall_seconds=wall)


def test_a_smoke_that_never_learned_refuses() -> None:
    codes = smoke_codes(driver={"grad_steps": 0})
    assert "SMOKE_ZERO_GRAD_STEPS" in codes


def test_a_slow_learner_refuses() -> None:
    # 1000 steps in 2400 s == 1500/hour, below 60% of the M3 retry's 4570.
    assert "SMOKE_GRAD_STEPS_LOW" in smoke_codes(driver={"grad_steps": 1000})


def test_a_missing_learner_rate_refuses() -> None:
    assert "SMOKE_GRAD_STEPS_LOW" in smoke_codes(
        driver={"grad_steps": None}, wall_seconds=2400.0
    )


def test_a_collapsed_collection_rate_refuses() -> None:
    assert "SMOKE_TRANSITIONS_LOW" in smoke_codes(driver={"episodes": 300})


def test_a_missing_episode_length_makes_the_rate_unknowable_and_refuses() -> None:
    assert "SMOKE_TRANSITIONS_LOW" in smoke_codes(episode_length_steps=None)


def test_a_tripped_watchdog_refuses() -> None:
    """The loudest queue-depth signal: a wedged learner with a growing backlog."""
    assert "SMOKE_QUEUE_BACKLOG" in smoke_codes(watchdog_tripped=True)


def test_a_saturated_drain_batch_refuses() -> None:
    """One gradient step per drain pass, 16 episodes per pass, is a backlog."""
    assert "SMOKE_QUEUE_BACKLOG" in smoke_codes(driver={"episodes": 40000})


def test_an_uncomputable_backlog_ratio_refuses() -> None:
    assert "SMOKE_QUEUE_BACKLOG" in smoke_codes(driver={"episodes": None})


def test_a_memory_projection_over_the_ceiling_refuses() -> None:
    """An OOM kill at 4am leaves no [multi done] line and no final save."""
    assert "SMOKE_RSS_PROJECTION" in smoke_codes(physical_memory_bytes=8.0e9)


@pytest.mark.parametrize(
    "mutation",
    [
        {"rss": {"samples": [], "peak_bytes": None, "first_bytes": None}},
        {"rss": {"samples": [[0.0, 1.0], [1.0, 2.0]]}},  # too few for a slope
        {"physical_memory_bytes": None},
    ],
)
def test_unmeasured_memory_refuses(mutation: Dict[str, Any]) -> None:
    assert "SMOKE_RSS_PROJECTION" in smoke_codes(**mutation)


def test_a_slow_snapshot_load_refuses() -> None:
    """The per-episode read runs on the collector thread: it is stolen collection."""
    codes = smoke_codes(snapshot_load={"ok": True, "seconds": [2.0, 2.1, 2.2, 2.3]})
    assert "SMOKE_SNAPSHOT_LOAD_SLOW" in codes


@pytest.mark.parametrize(
    "load",
    [
        {"ok": False, "seconds": [], "error": "pool unreadable"},
        {"ok": True, "seconds": []},
    ],
)
def test_an_untimed_snapshot_load_refuses(load: Dict[str, Any]) -> None:
    assert "SMOKE_SNAPSHOT_LOAD_SLOW" in smoke_codes(snapshot_load=load)


@pytest.mark.parametrize("pool", [{"size": 1}, {"size": None}, {"size": 0}])
def test_a_pool_that_never_grew_refuses(pool: Dict[str, Any]) -> None:
    """Without a second snapshot the per-episode load path read ONE file."""
    assert "SMOKE_POOL_NOT_GROWING" in smoke_codes(pool=pool)


def test_the_smoke_report_names_the_production_replay_floor() -> None:
    """The canary lowers --min-replay; the smoke must not, and must say so."""
    evidence = make_smoke_evidence()
    report = sizing_module.format_smoke_report(
        sizing_module.evaluate_smoke(evidence), evidence
    )
    assert "PRODUCTION value" in report
    assert "25000" in report


# ---------------------------------------------------------------------------
# PART 4 — the morning comparison. Its job is to stop a bare number being read
# as a ranking.
# ---------------------------------------------------------------------------

#: The real payload of `runs/m4.best.pt`, read from the file the M3 retry wrote.
#: Kept here as a fixture so the caution logic is exercised against the exact
#: shape it has to handle on demo morning.
M4_BEST = {
    "label": "runs/m4.best.pt",
    "path": "/Users/diego/Documents/MinecraftRL/runs/m4.best.pt",
    "exists": True,
    "run": "m4",
    "kind": "legacy",
    "grad_step": 8307,
    "win_rate": 1.0,
    "scripted_win_rate": 1.0,
    "eval_opponent": "scripted_mixed",
    "source": "stamped into the checkpoint by the save hook over 20 episodes",
}

M4_LATEST = {
    "label": "runs/m4.pt",
    "path": "/Users/diego/Documents/MinecraftRL/runs/m4.pt",
    "exists": True,
    "run": "m4",
    "kind": "legacy",
    "grad_step": 30000,
    "win_rate": 0.85,
    "scripted_win_rate": 0.85,
    "eval_opponent": "scripted_mixed",
    "source": "m4/summary.json - the RUN's LAST eval, not this file's own score",
}

SELFPLAY_BEST = {
    "label": "runs/m4_selfplay.best.pt",
    "path": "/repo/runs/m4_selfplay.best.pt",
    "exists": True,
    "run": "m4_selfplay",
    "kind": "best",
    "grad_step": 42000,
    "win_rate": 0.62,
    "reference_aggregate": 0.62,
    "reference_worst": 0.55,
    "references_evaluated": 3,
    "scripted_win_rate": 0.91,
    "eval_opponent": "the pinned reference gauntlet (aggregate)",
    "rated_elo": 1104.0,
}

SNAPSHOT = {
    "label": "snapshot 12 (PINNED)",
    "path": "/repo/runs/m4_selfplay/snapshots/snap_12.pt",
    "exists": True,
    "run": "m4_selfplay",
    "kind": "snapshot",
    "grad_step": 12000,
    "elo": 1042.0,
}


def compare_document(*candidates: Mapping[str, Any]) -> Dict[str, Any]:
    return {"compare_version": 1, "candidates": [dict(c) for c in candidates]}


def test_the_weaponless_perfect_score_carries_both_of_its_cautions() -> None:
    """1.000 at grad_step 8307, against an opponent holding NOTHING.

    Two independent traps sit on this one file: the opponent could not fight
    back, AND the selector keeps the FIRST net to reach a score, so later,
    better nets at the same rate were never saved.
    """
    notes = sizing_module.cautions_for(M4_BEST)
    assert sizing_module.CAUTION_BARE_REGIME in notes
    assert sizing_module.CAUTION_SELECTOR_FIRST in notes
    perfect = [n for n in notes if n.startswith(sizing_module.CAUTION_PERFECT_VS_UNARMED)]
    assert perfect and "8307" in perfect[0]


def test_the_selector_caution_names_the_strictly_greater_rule() -> None:
    """The claim must be checkable against the code it describes."""
    assert "STRICTLY higher" in sizing_module.CAUTION_SELECTOR_FIRST
    assert "_BestCheckpointSelector.consider" in sizing_module.CAUTION_SELECTOR_FIRST


def test_an_armored_best_checkpoint_gets_only_the_selector_caution() -> None:
    notes = sizing_module.cautions_for(SELFPLAY_BEST)
    assert sizing_module.CAUTION_SELECTOR_FIRST in notes
    assert sizing_module.CAUTION_BARE_REGIME not in notes
    assert not any(
        n.startswith(sizing_module.CAUTION_PERFECT_VS_UNARMED) for n in notes
    )


def test_a_bare_handed_latest_checkpoint_is_flagged_as_a_different_regime() -> None:
    notes = sizing_module.cautions_for(M4_LATEST)
    assert sizing_module.CAUTION_BARE_REGIME in notes
    # It is NOT a `.best.pt`, so the selector caution does not apply to it.
    assert sizing_module.CAUTION_SELECTOR_FIRST not in notes


def test_a_snapshot_carries_the_elo_scope_caution() -> None:
    notes = sizing_module.cautions_for(SNAPSHOT)
    assert sizing_module.CAUTION_ELO_SCOPE in notes


def test_an_unscored_candidate_is_marked_unevaluated_not_zero() -> None:
    notes = sizing_module.cautions_for(
        {"label": "runs/m4_selfplay.pt", "path": "/repo/runs/m4_selfplay.pt", "run": "m4_selfplay"}
    )
    assert sizing_module.CAUTION_UNEVALUATED in notes


def test_a_win_rate_without_a_named_opponent_is_called_out() -> None:
    notes = sizing_module.cautions_for(
        {"label": "mystery.pt", "path": "/x/mystery.pt", "run": "x", "win_rate": 0.9}
    )
    assert sizing_module.CAUTION_NO_OPPONENT_NAMED in notes


def test_the_table_does_not_rank_the_weaponless_score_first() -> None:
    """Sorting by win rate would put 1.000-vs-nothing at the top of a table
    whose entire purpose is to explain why that number is not a ranking."""
    text = sizing_module.compare_candidates(
        compare_document(M4_BEST, M4_LATEST, SELFPLAY_BEST, SNAPSHOT)
    )
    armored = text.index("ARMORED SELF-PLAY")
    bare = text.index("BARE-HANDED")
    assert armored < bare
    assert text.index("snapshot 12") < text.index("runs/m4_selfplay.best.pt")
    assert "THIS TABLE DOES NOT RANK" in text


def test_the_table_prints_every_candidate_and_its_provenance() -> None:
    text = sizing_module.compare_candidates(
        compare_document(M4_BEST, M4_LATEST, SELFPLAY_BEST, SNAPSHOT)
    )
    for label in ("runs/m4.best.pt", "runs/m4.pt", "runs/m4_selfplay.best.pt", "snapshot 12"):
        assert label in text
    assert "scored against: scripted_mixed" in text
    assert "numbers from:" in text
    assert "1.000" in text and "0.850" in text and "0.620" in text
    assert "1104" in text  # the rated Elo, where known


def test_a_missing_candidate_is_shown_as_missing() -> None:
    text = sizing_module.compare_candidates(
        compare_document(dict(SELFPLAY_BEST, exists=False))
    )
    assert "[MISSING ON DISK]" in text


def test_an_empty_comparison_says_so() -> None:
    text = sizing_module.compare_candidates(compare_document())
    assert "no candidates found" in text


def test_unknown_numbers_render_as_absent_not_zero() -> None:
    text = sizing_module.compare_candidates(
        compare_document(
            {"label": "runs/m4_selfplay.pt", "path": "/x.pt", "exists": True,
             "run": "m4_selfplay", "grad_step": 50000}
        )
    )
    assert "0.000" not in text


# ---------------------------------------------------------------------------
# The CLI the shell drives. Exit codes and the argv handoff.
# ---------------------------------------------------------------------------


def write_json(path: str, payload: Any) -> str:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    return path


def test_plan_cli_writes_the_argv_only_when_cleared(tmp_path: Any) -> None:
    """A refused plan must leave NO argv behind.

    The shell starts the run from the file the gate wrote. A stale argv from an
    earlier, cleared plan would let a later refused invocation start a run
    nothing had cleared.
    """
    plan_path = write_json(str(tmp_path / "plan.json"), make_plan())
    out_path = str(tmp_path / "plan_out.json")
    argv_path = str(tmp_path / "argv.txt")

    assert sizing_module.main(["plan", plan_path, out_path, argv_path]) == 0
    assert os.path.exists(argv_path)
    with open(argv_path, "r", encoding="utf-8") as handle:
        written = [line.strip() for line in handle if line.strip()]
    assert "--opponent" in written and "selfplay" in written

    refused = make_plan(opponent="dummy")
    write_json(plan_path, refused)
    assert sizing_module.main(["plan", plan_path, out_path, argv_path]) == 1
    assert not os.path.exists(argv_path)
    with open(out_path, "r", encoding="utf-8") as handle:
        recorded = json.load(handle)
    assert recorded["ok"] is False
    assert "OPPONENT_NOT_SELFPLAY" in recorded["refusals"]


def test_smoke_verdict_cli_writes_measurements_either_way(tmp_path: Any) -> None:
    """Measurements are written whether the smoke cleared or refused.

    That is why the launch gate re-derives the smoke's verdict instead of
    treating the file's existence as a pass.
    """
    evidence_path = write_json(str(tmp_path / "smoke.json"), make_smoke_evidence())
    measurements_path = str(tmp_path / "smoke_measurements.json")
    assert sizing_module.main(["smoke-verdict", evidence_path, measurements_path]) == 0
    assert os.path.exists(measurements_path)

    write_json(evidence_path, make_smoke_evidence(watchdog_tripped=True))
    os.remove(measurements_path)
    assert sizing_module.main(["smoke-verdict", evidence_path, measurements_path]) == 1
    assert os.path.exists(measurements_path)


def test_compare_cli_rejects_an_unknown_document_version(tmp_path: Any) -> None:
    path = write_json(str(tmp_path / "compare.json"), {"compare_version": 99})
    assert sizing_module.main(["compare", path]) == 2


def test_cli_usage_is_exit_two(tmp_path: Any) -> None:
    assert sizing_module.main([]) == 2
    assert sizing_module.main(["nonsense", str(tmp_path / "x.json")]) == 2
