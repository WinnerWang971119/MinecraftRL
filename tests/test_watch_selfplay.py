"""Tests for the overnight watcher (`scripts/watch_selfplay.sh`).

This is the script the operator runs against the LIVE 12-hour self-play run to
answer "is this still healthy?". It exists because a collector thread that
raises anything other than `BridgeError`/`TransportError` just dies
(`distributed/actor.py`'s collect loop), `ActorPool.aborted()` stays False (its
only abort trigger is the tier-2 JVM watchdog), and so the fleet can dwindle
from 25 arenas toward zero while training "continues" and nothing fails.

HOW THIS REACHES THE CODE UNDER TEST. Every verdict lives in the `watch_verdict`
module the script pipes to Python on stdin. These tests extract that module's
source VERBATIM from between the `WATCH_VERDICT_PY` heredoc sentinels and exec
it, so what is tested is byte-identical to what the operator runs. (The script
may not WRITE a module to disk the way `launch_selfplay.sh` does — read-only is
its whole contract — which is why the module travels on stdin.)

WHAT IS PINNED, AND WHY EACH PIN EXISTS:

* **Read-only is enforced by a test, not by intent.** The executable shell is
  scanned for every mutating verb and for every redirect that is not
  `/dev/null`, and the embedded Python's imports are scanned for `socket`. A
  watcher that connects to a bridge port DESTROYS the incumbent client
  (`BridgeServer` accepts exactly one), so "it doesn't connect" has to be a
  proven property.
* **Every log line and metric key is derived from its PRODUCER.** The patterns
  are checked against strings RENDERED from the very f-strings `agent/train.py`
  holds (via `ast`), with the rule that every interpolated expression must be
  one the test knows — so a rename or a reformat upstream fails here, at 09:00,
  instead of silently emptying a signal at 03:00. The metrics record shape is
  driven through the real `MetricsLogger`, and the throughput baseline's keys
  are re-derived from the canary's own `build_measurements`.
* **Each alarm is proved by MUTATION.** A dwindled fleet, a frozen grad step and
  a stale eval each get a fixture that differs from the healthy one in exactly
  that respect, and each must produce its intended verdict AND exit code.
* **UNKNOWN is never OK.** Every signal has a test for its "input missing"
  path, and the exit-code map is pinned: a run whose every signal is
  undetermined must exit nonzero, because a watcher that exits 0 on "I checked
  nothing" is the manufactured-confidence failure this project keeps hitting.
* **The healthy baseline CLEARS.** Without that anchor a check that fired
  unconditionally would pass every mutation test and cry wolf all night.
"""

from __future__ import annotations

import ast
import functools
import json
import os
import re
import subprocess
import sys
import types
from typing import Any, Dict, List, Mapping, Optional, Tuple

import pytest

# T19's tests own the canary-measurement fixture and the helper that re-derives
# the canary's key list from its own heredoc. Reused rather than re-typed: the
# throughput baseline this watcher reads IS that document, and a second copy of
# its schema would be free to drift away from the producer.
import test_launch_selfplay
from test_launch_selfplay import canary_measurement_keys, make_canary_measurements

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT_PATH = os.path.join(REPO_ROOT, "scripts", "watch_selfplay.sh")
TRAIN_PATH = os.path.join(REPO_ROOT, "agent", "train.py")


# ---------------------------------------------------------------------------
# Extract the decision logic from the shell script it ships inside.
# ---------------------------------------------------------------------------


def _script_lines() -> List[str]:
    with open(SCRIPT_PATH, "r", encoding="utf-8") as handle:
        return handle.read().splitlines()


def extract_heredoc(name: str) -> str:
    """Return the body of the ``<<'<name>'`` heredoc, verbatim.

    Asserts there is EXACTLY one opener: two copies of the verdict module in one
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

    What remains is the shell that actually RUNS. The distinction is
    load-bearing here: the script's header comment discusses `kill -0`, `nohup`
    and `agent.train` at length, and the read-only tests below would be
    vacuously satisfied — or vacuously broken — by prose. Same scanner as
    `tests/test_canary_selfplay.py` and `tests/test_launch_selfplay.py`, because
    quoting state has to carry across newlines in multi-line `die` messages.
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


def shell_without_comments() -> str:
    """The script with heredoc bodies and comments removed, quotes KEPT.

    A strictly stronger instrument than :func:`executable_shell` for "does this
    script run a forbidden command", because most of this script's fact
    gathering lives inside ``VAR="$(cmd ...)"`` — and the quote stripper, which
    knows nothing about command substitution, erases all of it. What remains
    here is real shell plus the `die` message strings, none of which name a
    mutating verb.
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
        if line.lstrip().startswith("#"):
            continue
        kept.append(line)
    return "\n".join(kept)


def _load_watch_module() -> types.ModuleType:
    source = extract_heredoc("WATCH_VERDICT_PY")
    module = types.ModuleType("watch_verdict_under_test")
    module.__dict__["__file__"] = SCRIPT_PATH
    exec(  # noqa: S102 - the point is to run the operator's code, not a copy
        compile(source, f"{SCRIPT_PATH}:WATCH_VERDICT_PY", "exec"), module.__dict__
    )
    return module


watch = _load_watch_module()

OK = watch.OK
WARN = watch.WARN
ALARM = watch.ALARM
UNKNOWN = watch.UNKNOWN


# ---------------------------------------------------------------------------
# Producer introspection: render the driver's own log lines from its f-strings.
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _train_tree() -> ast.Module:
    with open(TRAIN_PATH, "r", encoding="utf-8") as handle:
        return ast.parse(handle.read())


@functools.lru_cache(maxsize=1)
def _train_source() -> str:
    with open(TRAIN_PATH, "r", encoding="utf-8") as handle:
        return handle.read()


def _skeleton(node: ast.JoinedStr) -> str:
    """The f-string with every interpolation removed — a stable identifier."""
    return "".join(
        part.value
        for part in node.values
        if isinstance(part, ast.Constant) and isinstance(part.value, str)
    )


def find_fstrings(skeleton: str) -> List[ast.JoinedStr]:
    """Every f-string in `agent/train.py` whose literal skeleton is exactly this."""
    return [
        node
        for node in ast.walk(_train_tree())
        if isinstance(node, ast.JoinedStr) and _skeleton(node) == skeleton
    ]


def render_fstring(node: ast.JoinedStr, values: Mapping[str, Any]) -> str:
    """Render one of the driver's f-strings using known values for its fields.

    Every interpolated expression MUST appear in ``values``. That rule is the
    pin: if the producer starts interpolating something new, or renames what it
    interpolates, this raises here rather than quietly producing a line the
    watcher's regex still happens to match.
    """
    out: List[str] = []
    for part in node.values:
        if isinstance(part, ast.Constant):
            out.append(str(part.value))
            continue
        expression = ast.unparse(part.value)
        assert expression in values, (
            f"agent/train.py interpolates {expression!r} into a line "
            f"scripts/watch_selfplay.sh parses, and this test has no value for "
            f"it. The producer changed; update the watcher's pattern too."
        )
        spec = ""
        if part.format_spec is not None:
            spec = "".join(
                item.value
                for item in part.format_spec.values
                if isinstance(item, ast.Constant)
            )
        out.append(format(values[expression], spec))
    return "".join(out)


# ---------------------------------------------------------------------------
# Synthetic observations.
#
# The healthy document is deliberately realistic and doubles as a description of
# what a good night looks like an hour in: 25 pads all listening with a client,
# the learner at ~4,500 grad steps/hour, an eval cycle every 2,000 steps, and a
# rated Elo series with data in it.
# ---------------------------------------------------------------------------

NOW = 1_755_000_000.0
RUN_NAME = "m4_selfplay"
BASE_PORT = 5555
ARENAS = 25
CADENCE = 2000
CURRENT_STEP = 30_000

#: The canary fixture's baseline works out to 1200 grad steps / 1080 s / 25
#: arenas = 160 grad steps per arena-hour. Written down because every throughput
#: boundary below is expressed against it.
BASELINE_PER_ARENA_HOUR = 160.0

#: 160/arena-hour across 25 arenas.
EXPECTED_GRAD_STEPS_PER_HOUR = BASELINE_PER_ARENA_HOUR * ARENAS

#: The gate pins TARGET_PERIODIC_CHECKPOINTS = 20 across the window, so a 48,000
#: grad-step night checkpoints every 2,400. The tighter of the two cadences wins,
#: and here that is the 2,000-step eval cadence.
CHECKPOINT_EVERY = 2400

#: 2,000 grad steps at 4,000/hour = 30 minutes between training rows, which is
#: the shape of the real run: the launcher's own numbers put it near 39 minutes
#: at the M3 retry's measured rate. The stall window is 2x that and the rate
#: window 4x, both derived rather than constant.
DERIVED_ROW_INTERVAL_SECONDS = CADENCE / EXPECTED_GRAD_STEPS_PER_HOUR * 3600.0
DERIVED_STALL_SECONDS = 2.0 * DERIVED_ROW_INTERVAL_SECONDS
DERIVED_WINDOW_SECONDS = 4.0 * DERIVED_ROW_INTERVAL_SECONDS

HEALTHY_COMMAND = (
    "/repo/.venv/bin/python -m agent.train --arenas 25 --opponent selfplay "
    f"--run-name {RUN_NAME} --checkpoint /repo/runs/{RUN_NAME}.pt "
    "--log-backend jsonl"
)


def metrics_rows(
    pairs: List[Tuple[int, float]],
    *,
    now: float = NOW,
    rated_matches: Optional[float] = 12.0,
    trailing_eval_start: bool = False,
) -> List[Dict[str, Any]]:
    """TRAINING rows as the driver writes them: ``(step, seconds_ago)``, oldest first.

    Every row carries the self-play series, because that is what the loop logs at
    a checkpoint boundary. ``trailing_eval_start`` appends the row the eval cycle
    writes on ENTRY — which covers only the gap before the cycle's first episode
    finishes; after that, eval-EPISODE rows take over (see :func:`eval_episode_rows`).
    """
    rows: List[Dict[str, Any]] = []
    for index, (step, ago) in enumerate(pairs):
        row: Dict[str, Any] = {
            "step": step,
            "wall_time": now - ago,
            "elo/learner_rated": 1180.0 + index,
            "elo/learner_online": 1195.0 + index,
            "selfplay/pool_size": 6.0,
        }
        if rated_matches is not None:
            row["selfplay/rated_matches"] = rated_matches
        rows.append(row)
    if trailing_eval_start:
        rows.append(
            {
                "step": pairs[-1][0],
                "wall_time": now,
                "train/epsilon_mean": 0.0072,
                "train/epsilon_schedule": 0.05,
            }
        )
    return rows


def eval_episode_rows(
    count: int, *, oldest_ago: float, spacing: float, now: float = NOW
) -> List[Dict[str, Any]]:
    """Rows exactly as `eval/evaluate.py` writes them, into THIS SAME FILE.

    `agent/train.py` hands the run's logger to the MAIN eval track, and
    `evaluate` logs one row per episode at ``step=episode_index``. So a live run
    at grad step 300,000 starts emitting rows stepped 0, 1, 2, ... the moment a
    cycle begins. The keys are asserted against the producer in
    `test_the_eval_episode_row_keys_are_the_ones_evaluate_writes`.
    """
    return [
        {
            "step": index,
            "wall_time": now - oldest_ago + index * spacing,
            "episode_length": 118.0 + index,
            "episode_reward": -1.5 + index * 0.1,
            "win": 1.0 if index % 2 else 0.0,
            "aim_while_invisible": 0.0,
            "r_damage_dealt": 9.4,
            "r_damage_taken": -8.1,
            "r_step": -0.6,
            "r_aim": 0.2,
            "r_shaping": 0.0,
            "r_terminal": 1.0,
        }
        for index in range(count)
    ]


def metrics_section(rows: List[Dict[str, Any]], **overrides: Any) -> Dict[str, Any]:
    """A metrics section built by the REAL parser over a real JSONL rendering.

    Not a hand-built dict: the training/eval split is the thing under test in
    half this file, so every fixture goes through `parse_metrics_jsonl` rather
    than asserting the classification it is supposed to prove.
    """
    text = "".join(json.dumps(row) + "\n" for row in rows)
    section: Dict[str, Any] = {"read": True, "error": None}
    section.update(watch.parse_metrics_jsonl(text))
    section.update(overrides)
    return section


def with_metrics(
    observation: Mapping[str, Any], rows: List[Dict[str, Any]], **overrides: Any
) -> Dict[str, Any]:
    """``observation`` with its metrics section rebuilt from ``rows``."""
    updated = dict(observation)
    updated["metrics"] = metrics_section(rows, **overrides)
    return updated


def healthy_pairs(step_per_row: int = 750) -> List[Tuple[int, float]]:
    """Seven rows, ten minutes apart, ending at ``CURRENT_STEP`` right now.

    Six intervals over 3,600 s means the measured rate is exactly
    ``6 * step_per_row`` grad steps/hour — 4,500 at the default.
    """
    return [
        (CURRENT_STEP - (6 - index) * step_per_row, (6 - index) * 600.0)
        for index in range(7)
    ]


def make_log_facts(**overrides: Any) -> Dict[str, Any]:
    facts: Dict[str, Any] = {
        "read": True,
        "error": None,
        "truncated": False,
        "max_grad_step": CURRENT_STEP,
        "grad_step_lines": 46,
        "evals": [
            {
                "step": 26_000,
                "win_rate": 0.42,
                "mean_len": 121.0,
                "aim_invisible": 0.0,
                "passed_m2": "False",
                "opponent": "scripted_mixed",
            },
            {
                "step": 28_000,
                "win_rate": 0.47,
                "mean_len": 118.0,
                "aim_invisible": 0.0,
                "passed_m2": "False",
                "opponent": "scripted_mixed",
            },
        ],
        "skips": [],
        "selfplay": [
            {
                "step": 28_000,
                "elo_rated": 1186.0,
                "rated_matches": 12,
                "elo_online": 1201.0,
                "pool_size": 6,
                "matches_scored": 48,
            }
        ],
        "done": None,
        "last_eval": None,
    }
    facts.update(overrides)
    return facts


def make_document(**overrides: Any) -> Dict[str, Any]:
    """An observation of a healthy run. Every signal CLEARS."""
    document: Dict[str, Any] = {
        "now": NOW,
        "run_name": RUN_NAME,
        "paths": {
            "log": f"/repo/runs/{RUN_NAME}.log",
            "pid": f"/repo/runs/{RUN_NAME}.pid",
            "metrics": f"/repo/runs/{RUN_NAME}/metrics.jsonl",
            "argv": f"/repo/runs/{RUN_NAME}/launch/launch_argv.txt",
            "measurements": "/repo/runs/m4_selfplay_canary/canary/canary_measurements.json",
        },
        # None means DERIVE, which is what the shell passes unless the operator
        # named a number. The derived values are asserted in
        # `test_the_stall_window_is_derived_from_the_runs_own_cadence`.
        "thresholds": {"stall_minutes": None, "window_minutes": None},
        "eval_every_grad_steps": CADENCE,
        "process": {
            "pid_file_exists": True,
            "pid_raw": "43117",
            "command": HEALTHY_COMMAND,
            "etime": "07:41:22",
        },
        "log": make_log_facts(),
        "metrics": metrics_section(metrics_rows(healthy_pairs())),
        "launch_argv": {
            "read": True,
            "error": None,
            "flags": {
                "--arenas": str(ARENAS),
                "--port": str(BASE_PORT),
                "--eval-every-grad-steps": str(CADENCE),
                "--checkpoint-every-grad-steps": str(CHECKPOINT_EVERY),
                "--opponent": "selfplay",
            },
        },
        "measurements": {
            "read": True,
            "error": None,
            "document": make_canary_measurements(),
        },
        "fleet": {
            "lsof_available": True,
            "base_port": BASE_PORT,
            "base_port_source": "launch-argv",
            "expected_arenas": ARENAS,
            "expected_arenas_source": "launch-argv",
            "scan_low": BASE_PORT,
            "scan_high": BASE_PORT + ARENAS - 1,
            "listening_ports": list(range(BASE_PORT, BASE_PORT + ARENAS)),
            "attached_ports": list(range(BASE_PORT, BASE_PORT + ARENAS)),
        },
    }
    document.update(overrides)
    return document


def with_section(observation: Mapping[str, Any], name: str, **changes: Any) -> Dict[str, Any]:
    """A copy of ``observation`` with one section's keys replaced.

    The first parameter is deliberately NOT called ``document``: the
    measurements section carries a key of that name, and a collision would make
    every baseline mutation below unwritable.
    """
    updated = dict(observation)
    section = dict(updated[name])
    section.update(changes)
    updated[name] = section
    return updated


def verdict_of(document: Mapping[str, Any], key: str) -> str:
    return watch.evaluate_watch(document).signal(key).verdict


# ===========================================================================
# The script's own contract.
# ===========================================================================


def test_the_script_exists_and_is_executable() -> None:
    assert os.path.isfile(SCRIPT_PATH)
    assert os.access(SCRIPT_PATH, os.X_OK), "the operator runs this directly"


def test_the_script_parses() -> None:
    """`bash -n` parses without executing. A broken quote in a watcher would
    only show up at 3am, on the one run it exists to watch."""
    result = subprocess.run(
        ["bash", "-n", SCRIPT_PATH], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_the_embedded_python_compiles() -> None:
    compile(
        extract_heredoc("WATCH_VERDICT_PY"),
        f"{SCRIPT_PATH}:WATCH_VERDICT_PY",
        "exec",
    )


# ===========================================================================
# READ-ONLY, proved rather than promised.
# ===========================================================================


def test_the_script_mutates_nothing() -> None:
    """No verb that writes, deletes, signals or starts anything.

    This is the script's entire contract. It runs against a live 12-hour run
    that must not be disturbed, and a `watch` loop would repeat any mistake
    every few minutes all night.
    """
    code = shell_without_comments()
    # Positive control: if the stripper had eaten the file, every assertion
    # below would be vacuously true and this test would guarantee nothing.
    assert "lsof -nP" in code
    assert "ps -ww -o command=" in code

    for verb in (
        "mkdir",
        "rm ",
        "rmdir",
        "touch ",
        "tee ",
        "kill ",
        "kill -",
        "pkill",
        "nohup",
        "truncate",
        "mv ",
        "cp ",
        "chmod",
        "git ",
    ):
        assert verb not in code, f"the watcher must not run {verb!r}"


def test_the_script_redirects_only_to_dev_null() -> None:
    """Every redirect target is /dev/null or a file descriptor.

    A redirect to a path is a write, and the one thing this script may not do is
    write. Checked over the executable shell, so `>` inside prose or inside the
    Python heredoc cannot satisfy or break it.
    """
    code = executable_shell()
    targets = re.findall(r">>?\s*([^\s;&|)]+)", code)
    assert targets, "the stripper produced no redirects at all; it ate the file"
    for target in targets:
        assert target == "/dev/null" or target.startswith(
            "&"
        ), f"redirect to {target!r} would be a write"


def test_the_script_opens_no_socket() -> None:
    """No connect probe anywhere — not even to the multi-client Minecraft port.

    `BridgeServer` accepts exactly ONE TCP client and `_onConnection` resolves a
    second by DESTROYING the incumbent, so a connect against a bridge port takes
    a pad down. The canary and the launch gate may connect to Paper because they
    run once; a watcher meant for a `watch` loop adds a socket per poll, so it
    connects to nothing at all.
    """
    code = shell_without_comments()
    for connector in ("/dev/tcp", "nc ", "netcat", "curl", "telnet", "wget"):
        assert connector not in code, f"the watcher must not use {connector!r}"
    # The non-connecting inspection it uses INSTEAD is present.
    assert "-sTCP:LISTEN" in code
    assert "-sTCP:ESTABLISHED" in code

    # And the embedded Python imports nothing that could open one.
    tree = ast.parse(extract_heredoc("WATCH_VERDICT_PY"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported <= {"__future__", "json", "os", "re", "sys", "time", "typing"}, (
        f"the verdict module imports {sorted(imported)}; it must stay "
        "stdlib-only and must not be able to open a socket or spawn a process"
    )


def test_the_script_starts_nothing() -> None:
    """It runs no driver, no bridge, no JVM. It only ever observes."""
    code = shell_without_comments()
    for launcher in (
        "agent.train",
        "start-pads",
        "start.sh",
        "distributed.launcher",
        "npm",
        "node ",
        ".jar",
        "java",
    ):
        assert launcher not in code, f"the watcher must not invoke {launcher}"


# ===========================================================================
# The producer pins: every line and key this watcher parses, derived from the
# code that emits it.
# ===========================================================================


def test_the_eval_completion_line_is_the_one_train_py_emits() -> None:
    """`[multi grad_step N] win_rate=... opponent=...` is the ONLY mid-run
    evidence that an eval cycle finished, and the whole EVAL signal rests on it."""
    nodes = find_fstrings(
        "[multi grad_step ] win_rate= mean_len= aim_invisible= passed_m2= opponent="
    )
    assert len(nodes) == 1, f"expected one producer, found {len(nodes)}"
    line = render_fstring(
        nodes[0],
        {
            "eval_grad_step": 28_000,
            "report.win_rate": 0.472,
            "report.mean_episode_length": 118.0,
            "report.aim_while_invisible": 0.0,
            "report.passed_m2": False,
            "eval_opponent_name": "scripted_mixed",
        },
    )
    match = watch.EVAL_DONE_RE.search(line)
    assert match is not None, line
    assert int(match.group("step")) == 28_000
    assert float(match.group("win_rate")) == pytest.approx(0.472)
    assert float(match.group("mean_len")) == pytest.approx(118.0)
    assert match.group("passed_m2") == "False"
    assert match.group("opponent") == "scripted_mixed"

    parsed = watch.parse_run_log(line)
    assert parsed["evals"] == [
        {
            "step": 28_000,
            "win_rate": 0.472,
            "mean_len": 118.0,
            "aim_invisible": 0.0,
            "passed_m2": "False",
            "opponent": "scripted_mixed",
        }
    ]


def test_the_skipped_eval_line_is_the_one_train_py_emits() -> None:
    """The pinned-reference failure mode's fingerprint. A cycle that raises is
    swallowed so it cannot end the night; this line is all that is left of it."""
    nodes = find_fstrings(
        "[multi grad_step ] eval cycle SKIPPED: it raised : . Training continues "
        "and the next cycle is due at grad_step ; this cycle selected no "
        "checkpoint and rated no match, so both series simply have a gap here."
    )
    assert len(nodes) == 1
    line = render_fstring(
        nodes[0],
        {
            "grad_step": 31_000,
            "type(exc).__name__": "BridgeError",
            "exc": "arena 7: reply lost",
            "next_eval_at": 33_000,
        },
    )
    parsed = watch.parse_run_log(line)
    assert parsed["skips"] == [{"step": 31_000, "error": "BridgeError"}]


def test_the_selfplay_line_is_the_one_train_py_emits_on_both_branches() -> None:
    """The rated-Elo line, including its "(0 rated matches ... EMPTY)" branch.

    Those two branches are a DIFFERENT claim from each other — "the learner
    stopped improving" versus "the series has no data at all" — and both render
    as a flat line in any plot, so the watcher must tell them apart.
    """
    head = find_fstrings("[multi grad_step ] selfplay: elo_rated= ")
    assert len(head) == 1
    tail = find_fstrings(" elo_online= pool= matches= ")
    assert len(tail) == 1
    rated_clause = find_fstrings("( rated match(es))")
    assert len(rated_clause) == 1

    head_text = render_fstring(
        head[0], {"step": 28_000, "row['elo/learner_rated']": 1186.0}
    )
    tail_text = render_fstring(
        tail[0],
        {
            "row['elo/learner_online']": 1201.0,
            "int(row['selfplay/pool_size'])": 6,
            "int(row['selfplay/matches_scored'])": 48,
        },
    )
    # The label `elo_rated=` is pinned to the metric key `elo/learner_rated` by
    # the render above: the watcher reads the label and reports it under the key.
    assert watch.ELO_RATED_METRIC in ast.unparse(head[0])

    populated = head_text + render_fstring(rated_clause[0], {"rated": 12}) + tail_text
    parsed = watch.parse_run_log(populated + " draw_rate=0.120")
    assert parsed["selfplay"] == [
        {
            "step": 28_000,
            "elo_rated": 1186.0,
            "rated_matches": 12,
            "elo_online": 1201.0,
            "pool_size": 6,
            "matches_scored": 48,
        }
    ]

    empty_clause = "(0 rated matches - elo/learner_rated is EMPTY)"
    assert empty_clause in _train_source(), "the EMPTY branch was reworded"
    empty = head_text + empty_clause + tail_text
    parsed_empty = watch.parse_run_log(empty + " draw_rate=n/a")
    assert parsed_empty["selfplay"][0]["rated_matches"] == 0
    assert parsed_empty["selfplay"][0]["elo_rated"] == pytest.approx(1186.0)


def test_the_teardown_line_is_the_one_train_py_emits() -> None:
    """`[multi done]` is the only reliable completion signal: the driver's exit
    code is `0 if passed_m2 else 1` and a self-play run never clears that gate,
    so "ended cleanly" and "died" are told apart by this line alone."""
    nodes = find_fstrings(
        "[multi done] reason= episodes= grad_steps= passed_m2= checkpoints_saved="
    )
    assert len(nodes) == 1
    line = render_fstring(
        nodes[0],
        {
            "result.stop_reason": "max_grad_steps",
            "result.episodes_received": 41_233,
            "result.grad_steps": 54_000,
            "result.passed_m2": False,
            "result.checkpoints_saved": 108,
        },
    )
    parsed = watch.parse_run_log(line)
    assert parsed["done"] == {
        "reason": "max_grad_steps",
        "episodes": 41_233,
        "grad_steps": 54_000,
        "passed_m2": "False",
        "checkpoints_saved": 108,
    }


def test_the_last_eval_line_is_the_one_train_py_emits() -> None:
    """Printed only at teardown, so it is the morning read, never the mid-run one."""
    nodes = find_fstrings("  last eval: win_rate= mean_len= aim_invisible=")
    assert nodes, "agent/train.py no longer prints a 'last eval:' summary"
    for node in nodes:
        line = render_fstring(
            node,
            {
                "report.win_rate": 0.512,
                "report.mean_episode_length": 118.0,
                "report.aim_while_invisible": 0.0,
            },
        )
        parsed = watch.parse_run_log(line)
        assert parsed["last_eval"] == {
            "win_rate": 0.512,
            "mean_len": 118.0,
            "aim_invisible": 0.0,
        }


def _function_source(relative_path: str, function_name: str) -> str:
    """The source of one function in any repo file, for pinning a producer."""
    path = os.path.join(REPO_ROOT, relative_path)
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return ast.get_source_segment(text, node) or ""
    raise AssertionError(f"{relative_path} has no {function_name}")


def _string_literals_in(function_name: str) -> List[str]:
    for node in ast.walk(_train_tree()):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return [
                item.value
                for item in ast.walk(node)
                if isinstance(item, ast.Constant) and isinstance(item.value, str)
            ]
    raise AssertionError(f"agent/train.py has no {function_name}")


def test_the_metric_keys_are_the_ones_train_py_logs() -> None:
    """Three keys carry three signals; each is read from its producer's own row."""
    selfplay_keys = _string_literals_in("selfplay_log_row")
    assert watch.ELO_RATED_METRIC in selfplay_keys
    assert watch.RATED_MATCHES_METRIC in selfplay_keys
    assert watch.EVAL_START_METRIC in _string_literals_in("epsilon_log_row")


def test_the_eval_in_flight_inference_rests_on_one_call_site() -> None:
    """`train/epsilon_*` in the NEWEST row means an eval cycle is in flight.

    That inference is only sound because `_maybe_log_mean_epsilon` has exactly
    one call site and it is inside the eval block, ahead of the long part. If a
    second call site were ever added — at a checkpoint boundary, say — a healthy
    run would look permanently mid-eval and every stall ALARM would soften to a
    WARN. The order is pinned here rather than assumed.
    """
    source = _train_source()
    call = "_maybe_log_mean_epsilon(grad_step)"
    assert source.count(call) == 1, "a second call site would break the inference"

    for anchor in (
        "if do_eval and grad_step >= next_eval_at:",
        "outcome = _eval_via_designated_arena(",
    ):
        assert source.count(anchor) == 1, anchor
    assert source.index("if do_eval and grad_step >= next_eval_at:") < source.index(call)
    assert source.index(call) < source.index("outcome = _eval_via_designated_arena(")


def test_the_metrics_row_shape_comes_from_the_real_logger(tmp_path: Any) -> None:
    """Driven through `MetricsLogger` itself, not an imitation of its output.

    `step` and `wall_time` are the two fields every time-based verdict in this
    watcher depends on, and the driver's stderr carries no timestamp at all — so
    if this file's shape changed, signals 2 and 4 would go dark.
    """
    from eval.logging import MetricsLogger

    logger = MetricsLogger(
        run_name="watch_probe", backend="jsonl", log_dir=str(tmp_path)
    )
    try:
        assert logger.backend == "jsonl"
        logger.log({watch.ELO_RATED_METRIC: 1186.0, watch.RATED_MATCHES_METRIC: 12.0}, step=28_000)
        logger.log({watch.EVAL_START_METRIC: 0.05, "train/epsilon_mean": 0.0072}, step=30_000)
    finally:
        logger.close()

    with open(logger.metrics_path, "r", encoding="utf-8") as handle:
        parsed = watch.parse_metrics_jsonl(handle.read())

    assert parsed["torn_lines"] == 0
    rows = parsed["rows"]
    assert [row["step"] for row in rows] == [28_000, 30_000]
    assert all(isinstance(row["wall_time"], float) for row in rows)
    assert rows[0][watch.ELO_RATED_METRIC] == pytest.approx(1186.0)
    assert watch.EVAL_START_METRIC in rows[-1]


def test_the_throughput_baseline_keys_are_written_by_the_canary() -> None:
    """Re-derived from T17's own `build_measurements`, called for real.

    A rename there must fail here, not at 03:00 as a silent UNKNOWN on the one
    signal that detects a dwindled collector fleet.
    """
    produced = set(canary_measurement_keys())
    for key in watch.BASELINE_KEYS + watch.BASELINE_EPISODE_KEYS:
        assert key in produced, f"the canary no longer writes {key!r}"


def test_the_fixture_is_the_canarys_own_document() -> None:
    """The baseline fixture below is T19's, which is pinned to the producer by
    `tests/test_launch_selfplay.py`. Asserted so a future edit cannot quietly
    swap it for a hand-written stand-in."""
    assert make_canary_measurements is test_launch_selfplay.make_canary_measurements
    assert set(make_canary_measurements()) == set(canary_measurement_keys())


# ===========================================================================
# Pure parsers.
# ===========================================================================


def test_parse_etime_reads_every_bsd_shape() -> None:
    assert watch.parse_etime("14:03") == 843.0
    assert watch.parse_etime("07:41:22") == 27_682.0
    assert watch.parse_etime("2-03:00:00") == 183_600.0
    assert watch.parse_etime("") is None
    assert watch.parse_etime("not-a-time") is None


def test_parse_lsof_ports_reads_listeners_and_both_ends_of_a_pair() -> None:
    listen = "p900\nf10\nn*:5555\nf11\nn127.0.0.1:5556\nf12\nn*:9999\n"
    assert watch.parse_lsof_ports(listen, 5555, 5579) == [5555, 5556]

    # Loopback traffic puts BOTH ends on this machine, so the client's row names
    # the bridge port too. Only ports inside the scanned range count.
    established = (
        "p900\nf20\nn127.0.0.1:5555->127.0.0.1:54321\n"
        "p901\nf21\nn127.0.0.1:54321->127.0.0.1:5555\n"
    )
    assert watch.parse_lsof_ports(established, 5555, 5579) == [5555]


def test_parse_lsof_ports_cannot_manufacture_a_pad_from_an_ipv6_address() -> None:
    """Only the text after an endpoint's LAST colon is a port. An IPv6 hextet
    that happens to be decimal must not read as a listening bridge."""
    ipv6 = "p900\nf10\nn[fe80:5560::1]:993\n"
    assert watch.parse_lsof_ports(ipv6, 5555, 5579) == []


def test_parse_launch_argv_reads_the_flags_the_gate_wrote() -> None:
    text = "--arenas\n25\n--opponent\nselfplay\n--eval-every-grad-steps\n2000\n"
    flags = watch.parse_launch_argv(text)
    assert flags["--arenas"] == "25"
    assert flags["--eval-every-grad-steps"] == "2000"
    assert watch.parse_launch_argv("") == {}


def test_parse_metrics_jsonl_survives_a_torn_final_line() -> None:
    """The file is APPENDED TO while it is read. A half-written last record must
    cost that record and nothing else — a watcher that crashes on a live file is
    dead on arrival."""
    good = json.dumps({"step": 100, "wall_time": NOW - 60.0, "elo/learner_rated": 1.0})
    torn = '{"step": 200, "wall_ti'
    parsed = watch.parse_metrics_jsonl(good + "\n" + torn)
    assert parsed["torn_lines"] == 1
    assert [row["step"] for row in parsed["rows"]] == [100]


def test_parse_metrics_jsonl_orders_the_two_rows_one_step_can_carry() -> None:
    """The eval boundary writes two rows at ONE grad step; the later of the two
    must sort later, because the newest row is what decides `eval_in_flight`."""
    first = json.dumps({"step": 500, "wall_time": NOW - 10.0, "elo/learner_rated": 1.0})
    second = json.dumps({"step": 500, "wall_time": NOW, "train/epsilon_schedule": 0.05})
    parsed = watch.parse_metrics_jsonl("\n".join([second, first]))
    assert watch.EVAL_START_METRIC in parsed["rows"][-1]


# ===========================================================================
# The healthy baseline. Every mutation below is measured against it.
# ===========================================================================


def test_a_healthy_run_reports_every_signal_ok_and_exits_zero() -> None:
    verdict = watch.evaluate_watch(make_document())
    assert [signal.verdict for signal in verdict.signals] == [OK] * 5, [
        (signal.key, signal.verdict, signal.headline) for signal in verdict.signals
    ]
    assert verdict.worst == OK
    assert verdict.exit_code == 0
    assert verdict.notes == ()


def test_the_healthy_baselines_arithmetic_is_what_it_claims() -> None:
    """Seven rows ten minutes apart at +750 steps each is 4,500 grad steps/hour,
    and the canary's document works out to 160 per arena-hour."""
    facts = watch.derive_facts(make_document())
    assert facts["current_grad_step"] == CURRENT_STEP
    assert facts["grad_steps_per_hour"] == pytest.approx(4500.0)
    assert facts["rate_span_seconds"] == pytest.approx(3600.0)
    assert facts["alive_arenas"] == ARENAS
    assert facts["frozen_seconds"] == pytest.approx(0.0)
    assert facts["eval_in_flight"] is False

    baseline = make_canary_measurements()
    per_arena = (
        baseline["training_grad_steps"]
        / (baseline["training_wall_seconds"] / 3600.0)
        / baseline["training_arenas"]
    )
    assert per_arena == pytest.approx(BASELINE_PER_ARENA_HOUR)


# ===========================================================================
# 1 LIVENESS.
# ===========================================================================


def test_liveness_alarms_when_the_driver_is_gone_with_no_teardown_line() -> None:
    """The mutation: `ps` reports nothing and the log has no `[multi done]`.
    That is a crash, and it is the loudest thing this script can find."""
    document = with_section(make_document(), "process", command="", etime="")
    signal = watch.evaluate_watch(document).signal("liveness")
    assert signal.verdict == ALARM
    assert "GONE" in signal.headline
    assert watch.evaluate_watch(document).exit_code == 1


def test_liveness_warns_rather_than_alarms_when_the_run_finished() -> None:
    """A completed run is not an emergency, and at 07:00 it is the expected
    reading. The teardown line is what separates it from a crash."""
    document = with_section(make_document(), "process", command="", etime="")
    document = with_section(
        document,
        "log",
        done={
            "reason": "max_grad_steps",
            "episodes": 41_233,
            "grad_steps": 54_000,
            "passed_m2": "False",
            "checkpoints_saved": 108,
        },
    )
    signal = watch.evaluate_watch(document).signal("liveness")
    assert signal.verdict == WARN
    assert "max_grad_steps" in signal.headline
    assert watch.evaluate_watch(document).exit_code == 0


def test_liveness_alarms_on_a_recycled_pid() -> None:
    """A pid that is alive but is not `agent.train` means the pidfile is stale.
    Reporting that as OK is exactly the defect class this watcher exists to
    prevent: something IS running, just not the run."""
    document = with_section(
        make_document(), "process", command="/usr/sbin/cupsd -l -f", etime="10:00"
    )
    signal = watch.evaluate_watch(document).signal("liveness")
    assert signal.verdict == ALARM
    assert "NOT the driver" in signal.headline


def test_liveness_alarms_when_the_pid_is_another_runs_driver() -> None:
    document = with_section(
        make_document(),
        "process",
        command=HEALTHY_COMMAND.replace(RUN_NAME, "m4_selfplay_smoke"),
    )
    assert verdict_of(document, "liveness") == ALARM


def test_liveness_is_unknown_without_a_pidfile() -> None:
    document = with_section(
        make_document(), "process", pid_file_exists=False, pid_raw="", command=""
    )
    signal = watch.evaluate_watch(document).signal("liveness")
    assert signal.verdict == UNKNOWN
    assert "no pidfile" in signal.headline


def test_liveness_is_unknown_when_the_pidfile_is_not_a_pid() -> None:
    document = with_section(make_document(), "process", pid_raw="garbage", command="")
    assert verdict_of(document, "liveness") == UNKNOWN


# ===========================================================================
# 2 GRAD STEP.
# ===========================================================================


def test_grad_step_alarms_when_the_step_has_not_moved_in_the_stall_window() -> None:
    """The mutation: rows keep arriving but every one carries the same step.

    "No new rows" would be the wrong question — the eval boundary writes two rows
    at one step — so the watcher measures how long the NUMBER has been frozen.
    """
    frozen = [
        (20_000, 3.0 * 3600.0),
        (CURRENT_STEP, 95 * 60.0),
        (CURRENT_STEP, 40 * 60.0),
        (CURRENT_STEP, 30.0),
    ]
    document = with_metrics(make_document(), metrics_rows(frozen))
    facts = watch.derive_facts(document)
    assert facts["frozen_seconds"] == pytest.approx(95 * 60.0)
    assert facts["frozen_seconds"] > facts["stall_seconds"]

    signal = watch.evaluate_watch(document).signal("grad_step")
    assert signal.verdict == ALARM
    assert "has NOT moved" in signal.headline
    assert watch.evaluate_watch(document).exit_code == 1


def test_a_restarted_run_cannot_date_the_freeze_to_the_previous_attempt() -> None:
    """`metrics.jsonl` is opened in APPEND mode, so a run restarted into the
    same name leaves the previous attempt's series in front of this one. An old
    row that happens to share the current step number must not be read as the
    moment this step was reached — that would ALARM a perfectly healthy run on a
    number from yesterday.
    """
    stale_attempt = [(29_000, 90_000.0), (CURRENT_STEP, 86_400.0)]
    this_attempt = [(1_000, 1_200.0), (CURRENT_STEP, 60.0)]
    document = with_metrics(
        make_document(), metrics_rows(stale_attempt + this_attempt)
    )
    facts = watch.derive_facts(document)
    assert facts["frozen_seconds"] == pytest.approx(60.0)
    assert verdict_of(document, "grad_step") == OK


def test_grad_step_is_ok_just_inside_the_derived_stall_window() -> None:
    """The boundary, on the DERIVED window rather than a constant.

    The healthy fixture's cadence and baseline put the expected interval between
    training rows at 30 minutes, so the stall window is 60. One minute either
    side of that is pinned, so a change to the multiple has to be deliberate.
    """
    window_minutes = DERIVED_STALL_SECONDS / 60.0
    inside = [
        (20_000, 5.0 * 3600.0),
        (CURRENT_STEP, (window_minutes - 1) * 60.0),
        (CURRENT_STEP, 60.0),
    ]
    outside = [
        (20_000, 5.0 * 3600.0),
        (CURRENT_STEP, (window_minutes + 1) * 60.0),
        (CURRENT_STEP, 60.0),
    ]
    assert verdict_of(with_metrics(make_document(), metrics_rows(inside)), "grad_step") == OK
    assert verdict_of(with_metrics(make_document(), metrics_rows(outside)), "grad_step") == ALARM


def test_grad_step_downgrades_to_warn_while_an_eval_cycle_is_in_flight() -> None:
    """An eval cycle is synchronous inside the supervising loop and can run tens
    of minutes, writing one TRAINING row on entry and no further one until it
    ends. That row is the evidence for the first minutes of a cycle (the
    eval-EPISODE rows take over from there), so a frozen step with it on top is
    a WARN, not a page."""
    closed = [
        (CURRENT_STEP - 6000, 160 * 60.0),
        (CURRENT_STEP - 4000, 130 * 60.0),
        (CURRENT_STEP - 2000, 100 * 60.0),
        (CURRENT_STEP, 70 * 60.0),
    ]
    document = with_metrics(
        make_document(), metrics_rows(closed, trailing_eval_start=True)
    )
    verdict = watch.evaluate_watch(document)
    signal = verdict.signal("grad_step")
    assert signal.verdict == WARN
    assert "in flight" in signal.headline
    # The rate still comes from the CLOSED boundaries, so THROUGHPUT stays a
    # measurement rather than collapsing along with the frozen step.
    assert verdict.signal("throughput").verdict == OK
    assert verdict.exit_code == 0


def test_the_rate_is_measured_to_the_last_closed_boundary_during_an_eval() -> None:
    """The learner keeps stepping through an eval cycle — the cycle pauses ONE
    designated arena, not the learner — but the loop writes no TRAINING row
    until the cycle ends. Counting that flat stretch as elapsed time would
    report a collapsed rate every time a cycle ran long, and would fire the
    THROUGHPUT alarm on a perfectly healthy run.
    """
    pairs = [(20_000, 3000.0), (CURRENT_STEP, 0.0)]
    rows = metrics_rows(pairs, trailing_eval_start=True)
    # Push the in-flight entry row out to 40 minutes after the last closed one.
    rows[-1]["wall_time"] = NOW
    document = with_metrics(make_document(), rows)

    facts = watch.derive_facts(document)
    assert facts["eval_in_flight"] is True
    assert facts["rate_excludes_in_flight_rows"] is True
    # 10,000 steps over the 3,000 s between the two CLOSED rows.
    assert facts["rate_span_seconds"] == pytest.approx(3000.0)
    assert facts["grad_steps_per_hour"] == pytest.approx(12_000.0)

    verdict = watch.evaluate_watch(document)
    assert verdict.signal("throughput").verdict == OK
    assert any(
        "last CLOSED boundary" in line
        for line in verdict.signal("grad_step").detail
    )


def test_a_skipped_cycle_cancels_the_in_flight_excuse() -> None:
    """A cycle that RAISED writes no closing row, so its entry row would look
    like a cycle still running — forever. The SKIPPED line at that same step is
    what closes it, and without this the pinned-reference failure would soften
    every stall ALARM to a WARN all night."""
    closed = [
        (CURRENT_STEP - 6000, 160 * 60.0),
        (CURRENT_STEP - 4000, 130 * 60.0),
        (CURRENT_STEP - 2000, 100 * 60.0),
        (CURRENT_STEP, 70 * 60.0),
    ]
    document = with_metrics(
        make_document(), metrics_rows(closed, trailing_eval_start=True)
    )
    document = with_section(
        document, "log", skips=[{"step": CURRENT_STEP, "error": "BridgeError"}]
    )
    assert watch.derive_facts(document)["eval_in_flight"] is False
    assert verdict_of(document, "grad_step") == ALARM


def test_grad_step_warns_rather_than_alarms_during_the_replay_warm_up() -> None:
    """No row yet, minutes into the run, is expected: the first row lands at the
    first checkpoint or eval boundary, which is after `min_replay` fills."""
    document = with_metrics(make_document(), [])
    document = with_section(document, "process", etime="12:00")
    document = with_section(document, "log", max_grad_step=None, evals=[], selfplay=[])
    signal = watch.evaluate_watch(document).signal("grad_step")
    assert signal.verdict == WARN
    assert "no training row yet" in signal.headline


def test_grad_step_alarms_when_there_is_still_no_row_hours_in() -> None:
    document = with_metrics(make_document(), [])
    document = with_section(document, "process", etime="07:41:22")
    signal = watch.evaluate_watch(document).signal("grad_step")
    assert signal.verdict == ALARM
    assert "NO training row" in signal.headline


def test_grad_step_is_unknown_when_the_metrics_file_cannot_be_read() -> None:
    """The driver's stderr carries no timestamps, so without this file neither
    the rate nor the stall can be measured — and saying OK would be a lie."""
    document = with_metrics(
        make_document(), [], read=False, error="FileNotFoundError: no such file"
    )
    signal = watch.evaluate_watch(document).signal("grad_step")
    assert signal.verdict == UNKNOWN
    assert "no metrics" in signal.headline
    # The coarse step from the log is still reported; it just cannot be timed.
    assert any("30,000" in line for line in signal.detail)


def test_grad_step_reports_a_torn_tail_without_letting_it_change_the_verdict() -> None:
    document = with_metrics(
        make_document(), metrics_rows(healthy_pairs()), torn_lines=1
    )
    signal = watch.evaluate_watch(document).signal("grad_step")
    assert signal.verdict == OK
    assert any("unparseable" in line for line in signal.detail)


# ===========================================================================
# 3 FLEET.
# ===========================================================================


def test_fleet_alarms_on_any_shortfall_and_names_the_missing_ports() -> None:
    """The mutation: four bridges are gone. This is the dwindling detector, so
    it ALARMs on ONE missing pad, not on a fraction."""
    alive = list(range(BASE_PORT, BASE_PORT + ARENAS - 4))
    document = with_section(
        make_document(), "fleet", listening_ports=alive, attached_ports=alive
    )
    signal = watch.evaluate_watch(document).signal("fleet")
    assert signal.verdict == ALARM
    assert "4 of 25" in signal.headline
    assert any("5576 5577 5578 5579" in line for line in signal.detail)
    assert watch.evaluate_watch(document).exit_code == 1


def test_fleet_alarms_on_a_single_missing_bridge() -> None:
    alive = list(range(BASE_PORT, BASE_PORT + ARENAS - 1))
    document = with_section(
        make_document(), "fleet", listening_ports=alive, attached_ports=alive
    )
    assert verdict_of(document, "fleet") == ALARM


def test_fleet_warns_when_a_bridge_listens_with_no_client_attached() -> None:
    """A pad nothing is collecting from. Transient during a relaunch, which is
    why it is a WARN and not a page."""
    attached = list(range(BASE_PORT, BASE_PORT + ARENAS - 2))
    document = with_section(make_document(), "fleet", attached_ports=attached)
    signal = watch.evaluate_watch(document).signal("fleet")
    assert signal.verdict == WARN
    assert "NO client attached" in signal.headline


def test_fleet_is_unknown_without_lsof() -> None:
    """There is no safe fallback: counting the fleet by CONNECTING would destroy
    the incumbent client on every pad it probed."""
    document = with_section(
        make_document(),
        "fleet",
        lsof_available=False,
        listening_ports=[],
        attached_ports=[],
    )
    signal = watch.evaluate_watch(document).signal("fleet")
    assert signal.verdict == UNKNOWN
    assert "lsof" in signal.headline


def test_fleet_is_unknown_when_the_launched_arena_count_was_never_read() -> None:
    """A count with nothing to compare it against is an observation, not a
    verdict. Defaulting the expectation to 25 would ALARM every run launched
    with fewer pads."""
    document = with_section(
        make_document(),
        "fleet",
        expected_arenas=None,
        expected_arenas_source="unknown",
    )
    signal = watch.evaluate_watch(document).signal("fleet")
    assert signal.verdict == UNKNOWN
    assert "25 bridge listener(s) found" in signal.headline


def test_fleet_always_says_what_a_listening_bridge_does_not_prove() -> None:
    """A collector that dies inside the driver leaves its bridge and its socket
    standing, so a full fleet is not evidence of a full fleet of COLLECTORS.
    Every fleet verdict has to carry that, or the OK is misread."""
    for document in (
        make_document(),
        with_section(make_document(), "fleet", listening_ports=[], attached_ports=[]),
    ):
        signal = watch.evaluate_watch(document).signal("fleet")
        assert any("collector thread is alive" in line for line in signal.detail)


# ===========================================================================
# 4 THROUGHPUT.
# ===========================================================================


def _with_baseline_grad_steps(document: Mapping[str, Any], grad_steps: float) -> Dict[str, Any]:
    """Move the canary's baseline so the ratio lands exactly where a test wants.

    Cleaner than nudging the observed rate: the observed side stays the healthy
    4,500 grad steps/hour over 25 live arenas (180 per arena-hour) in every case,
    so the only thing under test is the fraction.
    """
    baseline = make_canary_measurements(training_grad_steps=grad_steps)
    return with_section(document, "measurements", document=baseline)


def test_throughput_is_ok_exactly_at_the_warn_fraction() -> None:
    """180 observed against a 240 baseline is exactly 0.75x, and 0.75 is the
    launch gate's own `min_transitions_per_s_fraction`. At the line it clears."""
    document = _with_baseline_grad_steps(make_document(), 1800.0)
    signal = watch.evaluate_watch(document).signal("throughput")
    assert signal.verdict == OK
    assert signal.headline.startswith("0.75x")


def test_throughput_warns_just_below_the_warn_fraction() -> None:
    document = _with_baseline_grad_steps(make_document(), 1801.0)
    signal = watch.evaluate_watch(document).signal("throughput")
    assert signal.verdict == WARN
    assert watch.evaluate_watch(document).exit_code == 0


def test_throughput_warns_exactly_at_the_alarm_fraction_and_alarms_below_it() -> None:
    """Half the measured rate turns a 12-hour window into a 6-hour run, which
    changes what the night can deliver. At the line it is still a WARN."""
    assert verdict_of(_with_baseline_grad_steps(make_document(), 2700.0), "throughput") == WARN
    document = _with_baseline_grad_steps(make_document(), 2701.0)
    assert verdict_of(document, "throughput") == ALARM
    assert watch.evaluate_watch(document).exit_code == 1


def test_throughput_scales_the_expectation_by_the_arenas_actually_alive() -> None:
    """The mutation: 20 of 25 pads are gone, and the surviving 5 are producing
    their full share. Scaling by the arenas REQUESTED would report that as a
    throughput collapse and send the operator hunting the wrong failure."""
    alive = list(range(BASE_PORT, BASE_PORT + 5))
    document = with_section(
        make_document(), "fleet", listening_ports=alive, attached_ports=alive
    )
    # Five pads at the healthy per-arena rate: 5 * 180 = 900 grad steps/hour.
    slowed = [
        (CURRENT_STEP - (6 - index) * 150, (6 - index) * 600.0) for index in range(7)
    ]
    document = with_metrics(document, metrics_rows(slowed))
    assert watch.derive_facts(document)["grad_steps_per_hour"] == pytest.approx(900.0)

    verdict = watch.evaluate_watch(document)
    assert verdict.signal("throughput").verdict == OK
    assert verdict.signal("fleet").verdict == ALARM
    assert verdict.exit_code == 1


def test_throughput_reports_the_implied_episode_rate_as_implied() -> None:
    """Episodes are NOT observable mid-run: `learner.received` is never logged
    and no metrics row carries an episode counter, so the only honest episode
    figure is the grad-step rate times the canary's own episodes-per-grad-step.
    It must never be presented as a measurement."""
    signal = watch.evaluate_watch(make_document()).signal("throughput")
    implied = [line for line in signal.detail if "episodes/hour" in line]
    assert implied, signal.detail
    # 4,500 grad steps/hour * (885 episodes / 1200 grad steps) = 3,319/hour,
    # against 118.0 per arena-hour * 25 live arenas = 2,950 expected.
    assert "3,319" in implied[0]
    assert "2,950" in implied[0]
    assert "IMPLIED, not measured" in implied[0]


def test_throughput_says_the_baseline_is_a_floor() -> None:
    """The canary's wall clock includes its replay warm-up, during which it took
    no gradient steps. Reading its rate as a target rather than a floor would
    make a merely-adequate night look healthy."""
    signal = watch.evaluate_watch(make_document()).signal("throughput")
    assert any("FLOOR" in line for line in signal.detail)


def test_throughput_is_unknown_without_the_canary_measurement() -> None:
    document = with_section(
        make_document(),
        "measurements",
        read=False,
        error="FileNotFoundError: no such file",
        document={},
    )
    signal = watch.evaluate_watch(document).signal("throughput")
    assert signal.verdict == UNKNOWN
    assert "no baseline" in signal.headline


@pytest.mark.parametrize("key", watch.BASELINE_KEYS)
def test_throughput_is_unknown_when_a_baseline_key_is_absent(key: str) -> None:
    """Absence is never read as a healthy zero: that is how a missing
    measurement becomes a stale constant."""
    baseline = make_canary_measurements(**{key: None})
    document = with_section(make_document(), "measurements", document=baseline)
    signal = watch.evaluate_watch(document).signal("throughput")
    assert signal.verdict == UNKNOWN
    assert key in signal.headline


def test_throughput_is_unknown_when_the_rate_could_not_be_measured() -> None:
    document = with_metrics(
        make_document(), [], read=False, error="unreadable"
    )
    assert verdict_of(document, "throughput") == UNKNOWN


def test_throughput_is_unknown_when_no_arena_is_alive_to_scale_by() -> None:
    document = with_section(
        make_document(), "fleet", listening_ports=[], attached_ports=[]
    )
    signal = watch.evaluate_watch(document).signal("throughput")
    assert signal.verdict == UNKNOWN
    assert "scaled" in signal.headline


# ===========================================================================
# 5 EVAL.
# ===========================================================================


def test_eval_warns_past_two_cadences_and_alarms_past_four() -> None:
    """The mutation: the last completed cycle recedes while the learner runs on.

    This is the shape of the pinned-reference failure — the run keeps training,
    the checkpoint keeps saving, and every eval cycle is silently skipped.
    """
    base = make_document()
    last_completed = 28_000

    fresh = with_metrics(base, metrics_rows(healthy_pairs()))
    assert verdict_of(fresh, "eval") == OK  # gap 2,000 == one cadence

    def at_step(step: int) -> Dict[str, Any]:
        pairs = [(step - 4500, 3600.0), (step, 0.0)]
        return with_metrics(base, metrics_rows(pairs))

    assert verdict_of(at_step(last_completed + 2 * CADENCE), "eval") == OK
    assert verdict_of(at_step(last_completed + 2 * CADENCE + 1), "eval") == WARN
    assert verdict_of(at_step(last_completed + 4 * CADENCE), "eval") == WARN

    stale = at_step(last_completed + 4 * CADENCE + 1)
    signal = watch.evaluate_watch(stale).signal("eval")
    assert signal.verdict == ALARM
    assert watch.evaluate_watch(stale).exit_code == 1


def test_eval_escalates_to_alarm_when_the_gap_is_explained_by_skipped_cycles() -> None:
    """A stale eval plus SKIPPED lines is not a slow cycle, it is the failure
    itself: those cycles selected no checkpoint and rated no match."""
    pairs = [(28_000, 3600.0), (33_000, 0.0)]
    document = with_metrics(make_document(), metrics_rows(pairs))
    assert verdict_of(document, "eval") == WARN

    with_skips = with_section(
        document,
        "log",
        skips=[
            {"step": 30_000, "error": "BridgeError"},
            {"step": 32_000, "error": "BridgeError"},
        ],
        evals=make_log_facts()["evals"],
    )
    signal = watch.evaluate_watch(with_skips).signal("eval")
    assert signal.verdict == ALARM
    assert any("SKIPPED" in line for line in signal.detail)


def test_a_skipped_cycle_alone_is_a_warn() -> None:
    """One skip inside the cadence is a bad cycle, not a broken run."""
    document = with_section(
        make_document(),
        "log",
        evals=make_log_facts()["evals"],
        skips=[{"step": 29_000, "error": "BridgeError"}],
    )
    assert verdict_of(document, "eval") == WARN


def test_eval_warns_when_the_rated_series_is_still_empty() -> None:
    """`elo/learner_rated` flat and `selfplay/rated_matches` at 0 are different
    claims that plot identically. Only the denominator tells them apart, and a
    run with no rated match has no AC7 curve at all in the morning."""
    document = with_metrics(
        make_document(), metrics_rows(healthy_pairs(), rated_matches=0.0)
    )
    signal = watch.evaluate_watch(document).signal("eval")
    assert signal.verdict == WARN
    assert any("EMPTY, not flat" in line for line in signal.detail)


def test_eval_reports_the_rated_elo_with_the_timestamp_from_the_metrics_row() -> None:
    """The driver's stderr carries no timestamp, so the age of the rated series
    can only come from metrics.jsonl."""
    signal = watch.evaluate_watch(make_document()).signal("eval")
    line = [item for item in signal.detail if watch.ELO_RATED_METRIC + " = " in item]
    assert line, signal.detail
    assert "1186.0" in line[0]
    assert "0s ago" in line[0]


def test_eval_falls_back_to_the_log_and_says_the_timestamp_is_missing() -> None:
    """Before any metrics row carries the series, the log still has the number —
    and the report must not imply it knows when it was written."""
    document = with_metrics(
        make_document(),
        [
            {
                "step": CURRENT_STEP - 4500,
                "wall_time": NOW - 3600.0,
                "selfplay/pool_size": 6.0,
            },
            {"step": CURRENT_STEP, "wall_time": NOW, "selfplay/pool_size": 6.0},
        ],
    )
    signal = watch.evaluate_watch(document).signal("eval")
    assert any("carries no timestamp" in line for line in signal.detail)
    assert any("1186.0" in line for line in signal.detail)


def test_eval_is_unknown_without_a_cadence_to_be_stale_against() -> None:
    document = dict(make_document())
    document["eval_every_grad_steps"] = None
    document = with_section(
        document, "launch_argv", read=False, error="FileNotFoundError", flags={}
    )
    signal = watch.evaluate_watch(document).signal("eval")
    assert signal.verdict == UNKNOWN
    assert "no eval cadence" in signal.headline


def test_eval_is_unknown_rather_than_ok_when_eval_was_switched_off() -> None:
    """`--eval-every-grad-steps 0` is a deliberate configuration, but nothing was
    checked, so it may not report OK."""
    document = dict(make_document())
    document["eval_every_grad_steps"] = 0
    signal = watch.evaluate_watch(document).signal("eval")
    assert signal.verdict == UNKNOWN
    assert watch.evaluate_watch(document).exit_code == 3


def test_eval_is_unknown_when_the_log_cannot_be_read() -> None:
    document = with_section(
        make_document(),
        "log",
        read=False,
        error="FileNotFoundError: no such file",
        evals=[],
        skips=[],
        selfplay=[],
        max_grad_step=None,
    )
    assert verdict_of(document, "eval") == UNKNOWN


def test_eval_says_so_plainly_when_no_cycle_has_ever_completed() -> None:
    document = with_section(make_document(), "log", evals=[], selfplay=[])
    document = with_metrics(document, metrics_rows([(1000, 3600.0), (1500, 0.0)])
    )
    signal = watch.evaluate_watch(document).signal("eval")
    assert signal.verdict == OK
    assert "no eval has completed yet" in signal.headline


# ===========================================================================
# Exit codes and the cross-signal reading.
# ===========================================================================


def test_the_exit_code_carries_the_worst_verdict() -> None:
    assert watch.EXIT_FOR[OK] == 0
    assert watch.EXIT_FOR[WARN] == 0
    assert watch.EXIT_FOR[ALARM] == 1
    assert watch.EXIT_FOR[UNKNOWN] == 3


def test_an_undetermined_run_exits_nonzero() -> None:
    """A mistyped --run-name makes every signal UNKNOWN. Exiting 0 there would
    let a cron watch nothing all night and report success every time."""
    document = make_document()
    document = with_section(document, "process", pid_file_exists=False, command="")
    document = with_metrics(document, [], read=False, error="gone")
    document = with_section(document, "fleet", lsof_available=False)
    document = with_section(document, "measurements", read=False, error="gone", document={})
    document = with_section(document, "log", read=False, error="gone", evals=[], skips=[], selfplay=[], max_grad_step=None)
    verdict = watch.evaluate_watch(document)
    assert [signal.verdict for signal in verdict.signals] == [UNKNOWN] * 5
    assert verdict.worst == UNKNOWN
    assert verdict.exit_code == 3


def test_an_alarm_outranks_an_unknown() -> None:
    """A failure you can see is worse news than a signal you could not read, and
    the exit code should name the loudest thing that is actually KNOWN."""
    document = with_section(make_document(), "measurements", read=False, error="gone", document={})
    alive = list(range(BASE_PORT, BASE_PORT + ARENAS - 3))
    document = with_section(document, "fleet", listening_ports=alive, attached_ports=alive)
    verdict = watch.evaluate_watch(document)
    assert verdict.signal("throughput").verdict == UNKNOWN
    assert verdict.signal("fleet").verdict == ALARM
    assert verdict.worst == ALARM
    assert verdict.exit_code == 1


def test_a_full_fleet_with_a_collapsed_rate_names_the_defect_this_watcher_exists_for() -> None:
    """The one reading neither signal can give alone.

    Every bridge listening with a client attached, and the rate on the floor, is
    what a collector thread dying INSIDE the driver looks like from outside it:
    the bridge process and its socket both survive, so the fleet count cannot
    see it and only the rate can.
    """
    document = _with_baseline_grad_steps(make_document(), 20_000.0)
    verdict = watch.evaluate_watch(document)
    assert verdict.signal("fleet").verdict == OK
    assert verdict.signal("throughput").verdict == ALARM
    assert any("FULL FLEET + COLLAPSED RATE" in note for note in verdict.notes)
    assert verdict.exit_code == 1


def test_a_dwindled_fleet_does_not_explain_away_a_per_arena_shortfall() -> None:
    alive = list(range(BASE_PORT, BASE_PORT + 10))
    document = with_section(
        make_document(), "fleet", listening_ports=alive, attached_ports=alive
    )
    # Ten pads producing a tenth of what ten pads should: both facts are real.
    crawling = [
        (CURRENT_STEP - (6 - index) * 12, (6 - index) * 600.0) for index in range(7)
    ]
    document = with_metrics(document, metrics_rows(crawling))
    verdict = watch.evaluate_watch(document)
    assert verdict.signal("fleet").verdict == ALARM
    assert verdict.signal("throughput").verdict == ALARM
    assert any("PER LIVE ARENA" in note for note in verdict.notes)


# ===========================================================================
# The report itself.
# ===========================================================================


def test_the_report_puts_every_signal_and_its_verdict_on_one_screen() -> None:
    document = make_document()
    text = watch.format_report(watch.evaluate_watch(document), document)
    for title in ("LIVENESS", "GRAD STEP", "FLEET", "THROUGHPUT", "EVAL"):
        assert title in text
    assert "VERDICT: OK" in text
    assert "READ-ONLY" in text
    assert len(text.splitlines()) < 60, "one screen, not a monitoring platform"


def test_the_report_never_badges_an_unchecked_signal_as_ok() -> None:
    """The defect class this whole script exists to prevent, applied to itself."""
    document = with_section(
        make_document(), "measurements", read=False, error="gone", document={}
    )
    verdict = watch.evaluate_watch(document)
    text = watch.format_report(verdict, document)
    headline = [line for line in text.splitlines() if line.startswith("  [4]")]
    assert len(headline) == 1
    assert "THROUGHPUT" in headline[0]
    assert watch._BADGE[UNKNOWN] in headline[0]
    assert " OK " not in headline[0]


# ===========================================================================
# End to end: the real script, over paths that do not exist. Starts nothing,
# connects to nothing, and is the only test that exercises the shell wiring.
# ===========================================================================


def _run_script(tmp_path: Any, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            SCRIPT_PATH,
            "--python",
            sys.executable,
            "--log",
            str(tmp_path / "run.log"),
            "--pid-file",
            str(tmp_path / "run.pid"),
            "--metrics",
            str(tmp_path / "metrics.jsonl"),
            "--launch-argv",
            str(tmp_path / "launch_argv.txt"),
            "--measurements",
            str(tmp_path / "canary_measurements.json"),
            *args,
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def test_the_script_exits_three_when_it_can_determine_nothing(tmp_path: Any) -> None:
    """Every input absent. The operator must see UNKNOWN and a nonzero exit,
    never a clean bill of health for a run the watcher never found."""
    result = _run_script(tmp_path, "--port", "65000")
    assert result.returncode == 3, result.stdout + result.stderr
    assert "VERDICT: UNKNOWN" in result.stdout
    assert "no pidfile" in result.stdout


def test_the_script_exits_one_on_an_alarm(tmp_path: Any) -> None:
    """A pidfile whose process does not exist and a log with no teardown line:
    the crash reading, end to end through the shell."""
    (tmp_path / "run.pid").write_text("999999\n", encoding="utf-8")
    (tmp_path / "run.log").write_text(
        "[multi grad_step 100] selfplay: elo_rated=1180.0 (0 rated matches - "
        "elo/learner_rated is EMPTY) elo_online=1180.0 pool=1 matches=0 "
        "draw_rate=n/a\n",
        encoding="utf-8",
    )
    result = _run_script(tmp_path, "--port", "65000")
    assert result.returncode == 1, result.stdout + result.stderr
    assert "VERDICT: ALARM" in result.stdout
    assert "no [multi done] line" in result.stdout
    # The ALARM is LIVENESS alone: with no arena count read, the fleet signal is
    # UNKNOWN, so nothing else could have produced the exit code.
    assert "bridge listener(s) found" in result.stdout


def test_the_script_leaves_no_file_behind(tmp_path: Any) -> None:
    """Read-only, measured rather than asserted: the directory it was pointed at
    holds exactly what was put there."""
    (tmp_path / "run.pid").write_text("999999\n", encoding="utf-8")
    before = sorted(os.listdir(tmp_path))
    _run_script(tmp_path, "--arenas", "1", "--port", "65000")
    assert sorted(os.listdir(tmp_path)) == before


def test_the_script_runs_with_no_python_flag_at_all(tmp_path: Any) -> None:
    """The DEFAULT invocation must work in a checkout with no venv.

    The canary and the launch gate hard-default to `<repo>/.venv/bin/python`
    because their Python imports torch and this package. This one's does not —
    it is stdlib-only — and this worktree has no venv, so inheriting that
    default would have made `scripts/watch_selfplay.sh` exit 2 before reading a
    single file, with a message about a package it never imports.
    """
    environment = {key: value for key, value in os.environ.items() if key != "PYTHON"}
    result = subprocess.run(
        [
            SCRIPT_PATH,
            "--log",
            str(tmp_path / "run.log"),
            "--pid-file",
            str(tmp_path / "run.pid"),
            "--metrics",
            str(tmp_path / "metrics.jsonl"),
            "--launch-argv",
            str(tmp_path / "launch_argv.txt"),
            "--measurements",
            str(tmp_path / "canary_measurements.json"),
            "--port",
            "65000",
        ],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    assert result.returncode != 2, result.stdout + result.stderr
    assert result.returncode == 3, result.stdout + result.stderr
    assert "VERDICT: UNKNOWN" in result.stdout


def test_the_process_table_is_read_at_full_width() -> None:
    """`ps -ww`, because LIVENESS matches `--run-name` inside the command line.

    The driver's argv carries a 64-character sha256 and three absolute paths
    before `--run-name` appears. A truncated command would read as "this pid
    belongs to a different run" and ALARM all night on a perfectly healthy
    fleet — the one failure that would make the operator stop trusting this
    screen.
    """
    code = shell_without_comments()
    reads = re.findall(r"ps\s+(-\S+\s+)*-o\s+\w+=", code)
    assert code.count("ps -ww -o ") == 2, "both ps reads must be full-width"
    assert "ps -o " not in code, "a width-limited ps read would truncate the argv"
    assert reads, "the stripper ate the ps reads"


def test_the_script_refuses_a_non_numeric_threshold(tmp_path: Any) -> None:
    result = _run_script(tmp_path, "--stall-minutes", "soon")
    assert result.returncode == 2
    assert "takes a number" in result.stderr


# ===========================================================================
# CRITICAL 1 — two series share metrics.jsonl.
#
# `agent/train.py` hands the run's logger to the MAIN eval track and
# `eval/evaluate.py` logs one row PER EVAL EPISODE at `step=episode_index`.
# Reading `step` as a grad step made a healthy run at 300,000 report `grad_step
# 9`, drove the rate to zero, and fired the loudest line on the screen at a
# healthy fleet for the ~97 minutes a cycle lasts, three times a night.
# ===========================================================================

#: The reviewer's scenario, reproduced exactly: a healthy run deep into the
#: night, 40 minutes into an eval cycle, ten episodes logged so far.
REAL_GRAD_STEP = 300_000


def mid_eval_cycle_document() -> Dict[str, Any]:
    """A healthy run at grad_step 300,000 with an eval cycle in flight.

    Training rows at the run's real cadence, then the cycle's entry row, then
    ten of `eval/evaluate.py`'s per-episode rows stepped 0..9 — which is what
    the file genuinely holds ~40 minutes into a cycle.
    """
    closed = [
        (REAL_GRAD_STEP - 6000, 160 * 60.0),
        (REAL_GRAD_STEP - 4000, 130 * 60.0),
        (REAL_GRAD_STEP - 2000, 100 * 60.0),
        (REAL_GRAD_STEP, 70 * 60.0),
    ]
    rows = metrics_rows(closed, trailing_eval_start=True)
    rows[-1]["wall_time"] = NOW - 40 * 60.0  # the cycle entered 40 minutes ago
    rows += eval_episode_rows(10, oldest_ago=38 * 60.0, spacing=4 * 60.0)
    document = with_metrics(make_document(), rows)
    return with_section(
        document,
        "log",
        max_grad_step=REAL_GRAD_STEP,
        evals=[
            {
                "step": REAL_GRAD_STEP - 2000,
                "win_rate": 0.47,
                "mean_len": 118.0,
                "aim_invisible": 0.0,
                "passed_m2": "False",
                "opponent": "scripted_mixed",
            }
        ],
    )


def test_the_eval_tracks_episode_rows_do_not_become_the_grad_step() -> None:
    """The reviewer's exact failing scenario, and what it must report instead.

    BEFORE: `OK GRAD STEP grad_step 9` (the newest eval EPISODE index),
    `ALARM THROUGHPUT 0.00x`, and the FULL FLEET + COLLAPSED RATE note firing on
    a healthy fleet. All four wrong at once, for over an hour, three times a
    night.
    """
    document = mid_eval_cycle_document()
    facts = watch.derive_facts(document)

    # The file really does hold both series, and the newest row really is an
    # episode-numbered one — otherwise this test proves nothing.
    assert facts["eval_episode_rows"] == 10
    assert document["metrics"]["rows"][-1]["step"] == 9

    assert facts["current_grad_step"] == REAL_GRAD_STEP
    assert facts["grad_steps_per_hour"] is not None
    assert facts["grad_steps_per_hour"] > 0.0

    verdict = watch.evaluate_watch(document)
    assert verdict.signal("grad_step").verdict == WARN  # frozen, but mid-cycle
    assert "300,000" in verdict.signal("grad_step").headline
    assert verdict.signal("throughput").verdict == OK
    assert verdict.notes == ()
    assert verdict.exit_code == 0


def test_an_eval_episode_row_is_positive_evidence_of_a_cycle_in_flight() -> None:
    """Strictly better evidence than the entry row, which the episode rows
    displace within minutes of a ~97-minute cycle. Without this the in-flight
    downgrade was dead code in production — it only ever fired in tests."""
    document = mid_eval_cycle_document()
    facts = watch.derive_facts(document)
    assert facts["eval_in_flight"] is True
    assert "eval-episode row" in facts["eval_in_flight_evidence"]


def test_a_skipped_cycle_still_cancels_the_episode_row_evidence() -> None:
    """A cycle that raised leaves its episode rows as the newest rows forever.
    The SKIPPED line at that grad step is what closes it — otherwise a
    permanently failing eval would soften every stall for the rest of the night."""
    document = mid_eval_cycle_document()
    document = with_section(
        document, "log", skips=[{"step": REAL_GRAD_STEP, "error": "BridgeError"}]
    )
    assert watch.derive_facts(document)["eval_in_flight"] is False
    assert verdict_of(document, "grad_step") == ALARM


def test_the_eval_episode_row_keys_are_the_ones_evaluate_writes() -> None:
    """Derived from `eval/evaluate.py`, not from this file's fixture.

    The classifier is a claim about two producers at once: every key the
    TRAINING rows carry must be inside `TRAINING_ROW_PREFIXES`, and no key the
    EVAL rows carry may be. Both halves are checked against the real code.
    """
    from env.mc_pvp_env import REWARD_COMPONENT_KEYS

    evaluate_source = _function_source("eval/evaluate.py", "evaluate")
    for key in ("episode_length", "episode_reward", "win", "aim_while_invisible"):
        assert f'"{key}"' in evaluate_source, f"evaluate no longer logs {key!r}"
        assert key in watch.EVAL_EPISODE_KEYS
    written = set(watch.EVAL_EPISODE_KEYS) | set(REWARD_COMPONENT_KEYS)
    for key in written:
        assert not key.startswith(
            watch.TRAINING_ROW_PREFIXES
        ), f"{key!r} would be misread as a training row"

    # And every key the training rows carry IS inside the prefixes.
    for producer in ("selfplay_log_row", "epsilon_log_row", "selfplay_eval_cycle_row"):
        keys = [
            literal
            for literal in _string_literals_in(producer)
            if "/" in literal and " " not in literal
        ]
        assert keys, producer
        for key in keys:
            assert key.startswith(
                watch.TRAINING_ROW_PREFIXES
            ), f"{producer} writes {key!r}, which the classifier would drop"


def test_the_main_eval_track_really_is_handed_the_runs_logger() -> None:
    """The premise of the whole workaround, pinned to the source.

    `agent/train.py` passes `logger=logger` to the MAIN track and `logger=None`
    to the REFERENCE tracks, with a comment naming this exact collision. If the
    main track is ever given the same treatment, this test fails and the
    classifier can be simplified — that is the point of pinning it.
    """
    source = _train_source()
    assert source.count("logger=None,") >= 1, "the reference tracks' guard is gone"
    evaluate_source = _function_source("eval/evaluate.py", "evaluate")
    assert "logger.log(metrics, step=ep)" in evaluate_source, (
        "eval/evaluate.py no longer steps its rows by EPISODE INDEX; re-check "
        "whether the training/eval row split is still needed"
    )


def test_a_file_of_only_eval_rows_says_so_instead_of_no_rows() -> None:
    """"NO training row" is a different claim from "no rows", and a file full of
    episode rows must not be reported as empty."""
    document = with_metrics(
        make_document(), eval_episode_rows(6, oldest_ago=1800.0, spacing=300.0)
    )
    signal = watch.evaluate_watch(document).signal("grad_step")
    assert signal.verdict == ALARM
    assert "NO training row" in signal.headline
    assert any("6 are eval-episode rows" in line for line in signal.detail)


# ===========================================================================
# CRITICAL 2 — the stall window is a property of the RUN, not a constant.
# ===========================================================================


def test_the_stall_window_is_derived_from_the_runs_own_cadence() -> None:
    """2x the interval the run should go between training rows.

    The 35-minute constant this replaced was BELOW the healthy interval: the
    gate pins TARGET_PERIODIC_CHECKPOINTS = 20 across a 12-hour window, which
    puts the interval near 39 minutes at the M3 retry's measured rate. A healthy
    run therefore ALARMed for a slice of every interval, and the operator's
    first check after launch read a hard ALARM.
    """
    facts = watch.derive_facts(make_document())
    assert facts["row_interval_seconds"] == pytest.approx(DERIVED_ROW_INTERVAL_SECONDS)
    assert facts["stall_seconds"] == pytest.approx(DERIVED_STALL_SECONDS)
    assert facts["window_seconds"] == pytest.approx(DERIVED_WINDOW_SECONDS)
    assert facts["stall_source"] == "derived"
    # The tighter of the two cadences wins, because either one writes a row.
    assert facts["row_interval"]["cadence_grad_steps"] == min(CADENCE, CHECKPOINT_EVERY)
    assert facts["row_interval"]["expected_grad_steps_per_hour"] == pytest.approx(
        EXPECTED_GRAD_STEPS_PER_HOUR
    )


def test_a_healthy_run_at_the_launchers_own_cadence_does_not_alarm() -> None:
    """The reviewer's second scenario: rows arriving at the derived interval,
    checked at the worst moment — just before the next one is due.

    Under the old 35-minute constant this read `ALARM GRAD STEP` on a run that
    was doing exactly what the launcher sized it to do.
    """
    interval = DERIVED_ROW_INTERVAL_SECONDS
    healthy = [
        (CURRENT_STEP - 3 * CADENCE, 3 * interval + interval * 0.98),
        (CURRENT_STEP - 2 * CADENCE, 2 * interval + interval * 0.98),
        (CURRENT_STEP - CADENCE, interval + interval * 0.98),
        (CURRENT_STEP, interval * 0.98),
    ]
    document = with_metrics(make_document(), metrics_rows(healthy))
    verdict = watch.evaluate_watch(document)
    assert verdict.signal("grad_step").verdict == OK
    assert verdict.exit_code == 0
    # The number is under the operator's nose, with its derivation.
    assert any(
        "stall window" in line and "the canary's" in line
        for line in verdict.signal("grad_step").detail
    )


def test_the_stall_window_falls_back_loudly_when_it_cannot_be_derived() -> None:
    """Without the cadence there is nothing to derive from. The fallback is
    generous and SAYS it is a fallback, naming what was missing."""
    document = with_section(make_document(), "launch_argv", flags={"--arenas": "25"})
    facts = watch.derive_facts(document)
    assert facts["stall_source"] == "fallback"
    assert facts["stall_seconds"] == pytest.approx(watch.FALLBACK_STALL_MINUTES * 60.0)
    assert facts["stall_seconds"] > 35 * 60.0, "the constant this replaced was too small"
    signal = watch.evaluate_watch(document).signal("grad_step")
    assert any("FALLBACK constant" in line for line in signal.detail)


def test_an_operator_override_still_wins() -> None:
    document = dict(make_document())
    document["thresholds"] = {"stall_minutes": 12.0, "window_minutes": 240.0}
    facts = watch.derive_facts(document)
    assert facts["stall_seconds"] == pytest.approx(720.0)
    assert facts["stall_source"] == "--stall-minutes"


# ===========================================================================
# The warnings that shipped alongside.
# ===========================================================================


def test_a_stale_rate_is_unknown_rather_than_a_confident_ok() -> None:
    """WARNING 1. Falling back to "the last two rows, whenever they were" let
    THROUGHPUT print a confident 1.12x from rows three hours old while GRAD STEP
    alarmed — which SUPPRESSED the cross-signal note in exactly the
    total-collector-death case this script exists for."""
    ancient = [
        (CURRENT_STEP - 4500, 5.0 * 3600.0),
        (CURRENT_STEP, 4.0 * 3600.0),
    ]
    document = with_metrics(make_document(), metrics_rows(ancient))
    facts = watch.derive_facts(document)
    assert facts["grad_steps_per_hour"] is None

    verdict = watch.evaluate_watch(document)
    assert verdict.signal("grad_step").verdict == ALARM
    assert verdict.signal("throughput").verdict == UNKNOWN
    assert verdict.exit_code == 1


def test_a_restart_into_the_same_run_name_is_a_discontinuity_not_a_rate() -> None:
    """WARNING 2. Two series inside one window gave `ALARM THROUGHPUT -189.48x`
    off a rate of -757,914/hour. A negative delta is not a slow run."""
    straddling = [
        (CURRENT_STEP, 40 * 60.0),
        (900, 20 * 60.0),
        (1_800, 60.0),
    ]
    document = with_metrics(make_document(), metrics_rows(straddling))
    facts = watch.derive_facts(document)
    assert facts["rate_discontinuity"] is True
    assert facts["grad_steps_per_hour"] is None

    signal = watch.evaluate_watch(document).signal("throughput")
    assert signal.verdict == UNKNOWN
    assert "BACKWARDS" in signal.headline
    assert any("append mode" in line for line in signal.detail)


def test_eval_is_unknown_when_the_two_files_disagree_about_the_run() -> None:
    """WARNING 4. Clamping `current - last_step` at zero turned contradictory
    evidence into a clean OK with exit 0 — which is what a --log pointed at the
    WRONG run looks like."""
    document = with_section(
        make_document(),
        "log",
        evals=[
            {
                "step": 90_000,
                "win_rate": 0.5,
                "mean_len": 118.0,
                "aim_invisible": 0.0,
                "passed_m2": "False",
                "opponent": "scripted_mixed",
            }
        ],
    )
    signal = watch.evaluate_watch(document).signal("eval")
    assert signal.verdict == UNKNOWN
    assert "ahead of the metrics" in signal.headline
    assert watch.evaluate_watch(document).exit_code == 3


def test_eval_will_not_claim_no_cycle_ever_ran_from_a_truncated_log() -> None:
    """WARNING 5. With the 8 MiB tail cut, "none has EVER completed" is a claim
    about the tail, not about the run. LIVENESS already refuses the same
    inference for a missing [multi done]."""
    document = with_section(make_document(), "log", truncated=True, evals=[])
    signal = watch.evaluate_watch(document).signal("eval")
    assert signal.verdict == UNKNOWN
    assert "TRUNCATED" in signal.headline

    # A cycle that IS visible in the tail is still the newest one, so truncation
    # does not poison a positive finding.
    intact = with_section(make_document(), "log", truncated=True)
    assert verdict_of(intact, "eval") == OK


def test_rows_without_a_step_are_counted_and_reported() -> None:
    """WARNING 6. Dropping them silently was safe but untested, and the
    resulting message read as "no rows" for a file that plainly had some."""
    text = (
        json.dumps({"step": None, "wall_time": NOW - 60.0, "selfplay/pool_size": 6.0})
        + "\n"
        + json.dumps({"wall_time": NOW - 30.0, "elo/learner_rated": 1180.0})
        + "\n"
        + json.dumps({"step": 42, "selfplay/pool_size": 6.0})
        + "\n"
    )
    parsed = watch.parse_metrics_jsonl(text)
    assert parsed["rows"] == []
    assert parsed["stepless_rows"] == 3
    assert parsed["torn_lines"] == 0

    document = with_metrics(make_document(), [], stepless_rows=3)
    signal = watch.evaluate_watch(document).signal("grad_step")
    assert any("carried no step" in line for line in signal.detail)


#: The real run's numbers, not the fixture's: the M3 retry measured ~4,570 grad
#: steps/hour at 25 pads, and the gate's TARGET_PERIODIC_CHECKPOINTS = 20 over a
#: 12-hour window rounds the checkpoint cadence to 3,000 grad steps. That is
#: 39.4 minutes between training rows — ABOVE the 35-minute constant this
#: watcher first shipped with.
M3_GRAD_STEPS_PER_HOUR = 4570.0
M3_CHECKPOINT_EVERY = 3000


def launcher_cadence_document() -> Dict[str, Any]:
    """A healthy 25-pad run sized exactly as `launch_selfplay.sh` sizes one."""
    baseline = make_canary_measurements(
        # A 1,080 s canary at 25 arenas sustaining the M3 retry's whole-fleet
        # 4,570 grad steps/hour takes 1,371 of them - i.e. 182.8 per arena-hour.
        training_grad_steps=M3_GRAD_STEPS_PER_HOUR * (1080.0 / 3600.0),
        training_wall_seconds=1080.0,
        training_arenas=25,
    )
    document = with_section(make_document(), "measurements", document=baseline)
    document = with_section(
        document,
        "launch_argv",
        flags={
            "--arenas": str(ARENAS),
            "--port": str(BASE_PORT),
            "--checkpoint-every-grad-steps": str(M3_CHECKPOINT_EVERY),
            "--eval-every-grad-steps": "6000",
        },
    )
    interval = M3_CHECKPOINT_EVERY / M3_GRAD_STEPS_PER_HOUR * 3600.0
    rows = [
        (CURRENT_STEP - 3 * M3_CHECKPOINT_EVERY, 3 * interval + 39 * 60.0),
        (CURRENT_STEP - 2 * M3_CHECKPOINT_EVERY, 2 * interval + 39 * 60.0),
        (CURRENT_STEP - M3_CHECKPOINT_EVERY, interval + 39 * 60.0),
        # Checked 39 minutes after the last row — a completely ordinary moment,
        # a few seconds before the next checkpoint boundary is due.
        (CURRENT_STEP, 39 * 60.0),
    ]
    return with_metrics(document, metrics_rows(rows))


def test_a_run_at_the_launchers_real_cadence_alarmed_under_the_old_constant() -> None:
    """CRITICAL 2, reproduced on the launcher's own arithmetic.

    39.4 minutes between training rows is what `TARGET_PERIODIC_CHECKPOINTS = 20`
    over a 12-hour window produces at the M3 retry's measured 4,570 grad
    steps/hour. The 35-minute constant this replaced sat BELOW that, so a
    perfectly healthy run ALARMed for roughly 11% of every interval — and the
    operator's first check after launch read a hard ALARM.
    """
    document = launcher_cadence_document()
    facts = watch.derive_facts(document)
    assert facts["row_interval_seconds"] == pytest.approx(2363.2, abs=1.0)  # 39.4 min
    assert facts["stall_seconds"] == pytest.approx(4726.5, abs=2.0)  # 78.8 min

    # BEFORE: the constant, applied as an override, is below the healthy interval.
    before = dict(document)
    before["thresholds"] = {"stall_minutes": 35.0, "window_minutes": 60.0}
    verdict_before = watch.evaluate_watch(before)
    assert verdict_before.signal("grad_step").verdict == ALARM
    assert verdict_before.exit_code == 1

    # AFTER: derived from the same two files the run was launched from.
    verdict_after = watch.evaluate_watch(document)
    assert verdict_after.signal("grad_step").verdict == OK
    assert verdict_after.exit_code == 0


def test_the_derived_window_still_alarms_on_a_real_stall_at_that_cadence() -> None:
    """The window widened, but it did not stop working: two missed boundaries at
    the launcher's own cadence is still an ALARM."""
    document = launcher_cadence_document()
    stall_seconds = watch.derive_facts(document)["stall_seconds"]
    rows = metrics_rows(
        [
            (CURRENT_STEP - M3_CHECKPOINT_EVERY, stall_seconds + 3600.0),
            (CURRENT_STEP, stall_seconds + 60.0),
        ]
    )
    stalled = with_metrics(document, rows)
    signal = watch.evaluate_watch(stalled).signal("grad_step")
    assert signal.verdict == ALARM
    assert watch.evaluate_watch(stalled).exit_code == 1


def test_the_base_port_comes_from_the_launch_argv(tmp_path: Any) -> None:
    """WARNING 3. The gate writes `--port` into launch_argv.txt, so a run on any
    other base port is a fact already sitting in a file this script opens.
    Defaulting to 5555 without looking produced `FLEET ALARM: 25 of 25 bridges
    are GONE` — the most alarming false positive available."""
    (tmp_path / "launch_argv.txt").write_text(
        "--arenas\n3\n--port\n61234\n--eval-every-grad-steps\n2000\n", encoding="utf-8"
    )
    result = _run_script(tmp_path)
    assert "61234-61236" in result.stdout, result.stdout
    assert "base port from: launch-argv" in result.stdout


def test_the_training_row_classifier_is_what_keeps_the_grad_step_honest() -> None:
    """CRITICAL 1's BEFORE state, pinned by mutation so it cannot come back.

    Neutralise the classifier — which is exactly what reading `step`
    unconditionally amounted to — and the same healthy run at grad_step 300,000
    reports `grad_step 9`, the newest eval EPISODE index. This is the defect
    the reviewer reproduced, held in place by a test so a later "simplification"
    of the classifier fails here instead of at 3am.
    """
    original = watch.is_training_row
    try:
        watch.is_training_row = lambda row: True
        # Rebuilt AFTER the patch: metrics_section classifies at build time.
        broken = mid_eval_cycle_document()
        assert watch.derive_facts(broken)["current_grad_step"] == 9
    finally:
        watch.is_training_row = original

    assert watch.derive_facts(mid_eval_cycle_document())["current_grad_step"] == (
        REAL_GRAD_STEP
    )
