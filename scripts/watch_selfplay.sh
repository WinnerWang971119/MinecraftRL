#!/usr/bin/env bash
# watch_selfplay.sh — the 3am read on a live self-play run. READ-ONLY.
#
# WHY THIS EXISTS. `distributed/actor.py`'s collector loop (`Collector.run`,
# actor.py:620-659) funnels `BridgeError` into `_recover` and treats
# `TransportError` as a clean end. Anything ELSE propagates out of the thread and
# the thread simply dies. `ActorPool` notices nothing: its only abort trigger is
# the tier-2 JVM watchdog (`_supervise`, actor.py:1192-1233), which watches the
# shared Paper port and nothing else. So over twelve unattended hours the fleet
# can dwindle from 25 collectors toward zero while the learner keeps stepping on
# a thinning stream: training "continues", `elo/learner_rated` freezes,
# throughput collapses, and NOTHING fails. Loud stderr at 3am is not a gate.
#
# The decision was NOT to patch the actor pool hours before the run. This script
# is the other half of that decision: a checker the operator can run at any
# moment against the live run, that answers "is this still healthy?" in one
# screen and says so in its exit code.
#
# READ-ONLY IS THE WHOLE CONTRACT. This script writes no file, creates no
# directory, sends no signal, starts nothing, and opens NO socket at all. It
# reads files, and it inspects the process and socket tables with `ps` and
# `lsof`. In particular:
#
#   * It never connects to a bridge port. `BridgeServer` accepts exactly ONE TCP
#     client and `_onConnection` resolves a second one by DESTROYING the
#     incumbent (bridge/transport.js) — a connect probe here would take a pad
#     down. `lsof` listener/peer inspection opens no connection.
#   * It does not connect to the Minecraft port either, even though Paper is
#     multi-client and `canary_selfplay.sh` / `launch_selfplay.sh` both do. A
#     watcher meant to be dropped in a `watch` loop should not add a socket per
#     poll to a JVM the run depends on.
#   * It never signals the driver, not even `kill -0`. Liveness comes from `ps`.
#
# WHAT IT CHECKS — five signals, each with its own verdict and the number behind
# it. See the embedded `watch_verdict` module for the thresholds and their
# reasoning; the module is pure, stdlib-only, and `tests/test_watch_selfplay.py`
# extracts it verbatim from between the WATCH_VERDICT_PY sentinels and drives
# every verdict boundary over synthetic documents, so what is tested is
# byte-identical to what the operator runs.
#
#   1 LIVENESS    the driver process, from the pidfile via `ps` (with a
#                 recycled-pid guard and a completed-vs-crashed distinction).
#   2 GRAD STEP   the current grad step and how long it has been frozen, from
#                 the TRAINING rows of runs/<run>/metrics.jsonl (the only source
#                 with timestamps on it). That file carries a second, unrelated
#                 series — `eval/evaluate.py` logs one row per eval EPISODE at
#                 step=episode_index into it — and the two are told apart by key
#                 namespace. See TRAINING_ROW_PREFIXES for what happens if they
#                 are not.
#   3 FLEET       bridge listeners on the run's port range vs the arena count it
#                 was launched with.
#   4 THROUGHPUT  the measured grad-step rate per LIVE arena against the canary's
#                 own measurement, plus the episodes/hour that implies.
#   5 EVAL        how many grad steps since an eval cycle last COMPLETED, the
#                 SKIPPED cycles since, and the latest `elo/learner_rated`.
#
# EXIT CODES
#   0  worst verdict is OK or WARN  (WARN is printed loudly but does not page)
#   1  at least one ALARM
#   2  usage error (raised by this shell before anything is read)
#   3  no ALARM, but at least one signal could not be DETERMINED
#
# 3 is not decoration. A mistyped --run-name makes every signal UNKNOWN, and a
# watcher that exits 0 on "I checked nothing" is the manufactured-confidence
# failure this project keeps paying for. UNKNOWN is never OK.
#
# Owner: the M4 self-play night. Written to be read at 3am, not extended.

set -euo pipefail

# --- Defaults ---------------------------------------------------------------
RUN_NAME="m4_selfplay"
CANARY_RUN_NAME="m4_selfplay_canary"
PYTHON_BIN=""
LOG_PATH=""
PID_PATH=""
METRICS_PATH=""
MEASUREMENTS=""
ARGV_PATH=""
ARENAS=""
EVAL_EVERY=""
BRIDGE_BASE_PORT=""

# EMPTY means DERIVE. Both windows are properties of the run, not constants:
# this driver writes a metrics row only at its own boundaries, so the healthy
# interval between rows is `min(--checkpoint-every-grad-steps,
# --eval-every-grad-steps)` divided by the rate the fleet actually sustains. The
# launch gate writes both cadences into launch_argv.txt and the canary measured
# the rate, so the watcher reads them rather than guessing.
#
# A CONSTANT HERE WAS A BUG, not a simplification. The first version shipped 35
# minutes. The gate pins TARGET_PERIODIC_CHECKPOINTS = 20 across a 12-hour
# window, which puts the healthy interval at ~39 minutes at the M3 retry's
# measured 4,570 grad steps/hour — so a perfectly healthy run sat above the
# threshold for a good fraction of every interval, and the operator's FIRST
# check after launch read a hard ALARM. A watcher that cries wolf on a healthy
# fleet is worse than no watcher.
STALL_MINUTES=""
WINDOW_MINUTES=""

# --- Resolve paths ----------------------------------------------------------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)}"

log()  { echo "[watch] $*"; }
warn() { echo "[watch] WARNING: $*" >&2; }
die()  { echo "[watch] FATAL: $*" >&2; exit 2; }

usage() {
    cat <<USAGE
usage: scripts/watch_selfplay.sh [options]

Read-only health check for a LIVE self-play run. Writes nothing, signals
nothing, connects to nothing. Safe to run in a loop:

    watch -n 300 scripts/watch_selfplay.sh

options:
  --run-name NAME        the run to watch (default: ${RUN_NAME}). Every path
                         below defaults from it, exactly as
                         scripts/launch_selfplay.sh names things.
  --log PATH             driver log (default: <repo>/runs/<run>.log)
  --pid-file PATH        pidfile   (default: <repo>/runs/<run>.pid)
  --metrics PATH         metrics   (default: <repo>/runs/<run>/metrics.jsonl)
  --launch-argv PATH     the argv the launch gate wrote
                         (default: <repo>/runs/<run>/launch/launch_argv.txt).
                         This is where --arenas and --eval-every-grad-steps are
                         read from, so the watcher checks what the run ACTUALLY
                         started with instead of a default.
  --measurements PATH    the canary's throughput baseline
                         (default: <repo>/runs/${CANARY_RUN_NAME}/canary/canary_measurements.json)
  --arenas N             override the launched arena count.
  --eval-every-grad-steps N
                         override the launched eval cadence.
  --port P               bridge BASE port (default: the --port the launch gate
                         wrote into launch_argv.txt, else 5555).
  --stall-minutes M      grad step frozen this long is an ALARM. DEFAULT:
                         DERIVED as 2x this run's own expected interval between
                         metrics rows (its checkpoint/eval cadence over the
                         canary's measured rate). Override with a number.
  --window-minutes M     the rate window. DEFAULT: DERIVED as 4x that same
                         interval, so the window always holds enough rows to
                         state a rate.
  --python PATH          interpreter (default: \$PYTHON, else <repo>/.venv/bin/python
                         when it exists, else python3 on PATH). The judgement is
                         stdlib-only, so any python3 works - no venv, no torch.
  -h, --help             this text.

exit codes:
  0  OK (or WARN — printed loudly, but does not page)
  1  ALARM
  2  usage error
  3  no ALARM, but a signal could not be DETERMINED. Never treated as OK.
USAGE
}

need_value() {
    if [[ "$2" -lt 2 ]]; then
        echo "[watch] $1 requires a value." >&2
        exit 2
    fi
}

# --- Parse ------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-name)       need_value "$1" $#; RUN_NAME="$2"; shift 2 ;;
        --log)            need_value "$1" $#; LOG_PATH="$2"; shift 2 ;;
        --pid-file)       need_value "$1" $#; PID_PATH="$2"; shift 2 ;;
        --metrics)        need_value "$1" $#; METRICS_PATH="$2"; shift 2 ;;
        --launch-argv)    need_value "$1" $#; ARGV_PATH="$2"; shift 2 ;;
        --measurements)   need_value "$1" $#; MEASUREMENTS="$2"; shift 2 ;;
        --arenas)         need_value "$1" $#; ARENAS="$2"; shift 2 ;;
        --eval-every-grad-steps) need_value "$1" $#; EVAL_EVERY="$2"; shift 2 ;;
        --port)           need_value "$1" $#; BRIDGE_BASE_PORT="$2"; shift 2 ;;
        --stall-minutes)  need_value "$1" $#; STALL_MINUTES="$2"; shift 2 ;;
        --window-minutes) need_value "$1" $#; WINDOW_MINUTES="$2"; shift 2 ;;
        --python)         need_value "$1" $#; PYTHON_BIN="$2"; shift 2 ;;
        -h|--help)        usage; exit 0 ;;
        *) usage >&2; die "unknown argument: $1" ;;
    esac
done

for pair in "--port:${BRIDGE_BASE_PORT}" "--arenas:${ARENAS}" \
            "--eval-every-grad-steps:${EVAL_EVERY}"; do
    value="${pair#*:}"
    [[ -z "${value}" || "${value}" =~ ^[0-9]+$ ]] || die "${pair%%:*} takes a whole number, got: ${value}"
done
for pair in "--stall-minutes:${STALL_MINUTES}" "--window-minutes:${WINDOW_MINUTES}"; do
    value="${pair#*:}"
    [[ -z "${value}" || "${value}" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "${pair%%:*} takes a number, got: ${value}"
done

LOG_PATH="${LOG_PATH:-${REPO_ROOT}/runs/${RUN_NAME}.log}"
PID_PATH="${PID_PATH:-${REPO_ROOT}/runs/${RUN_NAME}.pid}"
# `agent/train.py` builds its MetricsLogger with the default log_dir "runs" and
# the run name, and the launch gate pins --log-backend jsonl, so the run's rows
# land here. This file is the ONLY source in the run with timestamps on it: the
# driver's own stderr lines carry none.
METRICS_PATH="${METRICS_PATH:-${REPO_ROOT}/runs/${RUN_NAME}/metrics.jsonl}"
ARGV_PATH="${ARGV_PATH:-${REPO_ROOT}/runs/${RUN_NAME}/launch/launch_argv.txt}"
MEASUREMENTS="${MEASUREMENTS:-${REPO_ROOT}/runs/${CANARY_RUN_NAME}/canary/canary_measurements.json}"

# --- Interpreter ------------------------------------------------------------
# ANY python3 will do, and that is deliberate. The canary and the launch gate
# both hard-default to the repo's `.venv` because their Python imports torch and
# this package; the judgement below imports nothing but the standard library, so
# it must not refuse to run on a checkout that has no venv — which is exactly
# the case for the worktree this was written in. A watcher that cannot start is
# worth less than no watcher, because the operator finds out at 3am.
if [[ -z "${PYTHON_BIN}" ]]; then
    if [[ -n "${PYTHON:-}" ]]; then
        PYTHON_BIN="${PYTHON}"
    elif [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
        PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
    else
        PYTHON_BIN="python3"
    fi
fi
if [[ ! -x "${PYTHON_BIN}" ]] && ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    die "python interpreter not found: ${PYTHON_BIN}
      This script needs only a stdlib python3 - not this project's venv, and
      not torch. Install one, or pass --python /path/to/python."
fi

# ===========================================================================
# Fact gathering. Every command below is an OBSERVATION: `ps` and `lsof` read
# kernel tables, `date` reads the clock, and the file reads happen inside the
# Python block. Nothing here mutates anything, and nothing opens a socket.
# ===========================================================================

# argv_value <flag> — the value following <flag> in the launch gate's argv file,
# which launch_selfplay.sh writes one element per line. Empty when the file or
# the flag is absent; the Python block re-reads the same file for the flags IT
# needs, so this is only for sizing the port scan below.
argv_value() {
    local flag="$1" line previous=""
    [[ -f "${ARGV_PATH}" ]] || return 0
    while IFS= read -r line; do
        if [[ "${previous}" == "${flag}" ]]; then
            printf '%s\n' "${line}"
            return 0
        fi
        previous="${line}"
    done <"${ARGV_PATH}"
}

ARENAS_SOURCE="option"
if [[ -z "${ARENAS}" ]]; then
    ARENAS="$(argv_value --arenas || true)"
    ARENAS_SOURCE="launch-argv"
fi

# The BASE PORT comes from the same file, for the same reason: the gate writes
# `--port` into the argv it starts the driver with, so a run on any other base
# port is a FACT already sitting in a file this script opens. Defaulting to 5555
# without looking made a healthy fleet read as "25 of 25 bridges are GONE" - the
# most alarming false positive this script can produce, from data it had.
PORT_SOURCE="option"
if [[ -z "${BRIDGE_BASE_PORT}" ]]; then
    BRIDGE_BASE_PORT="$(argv_value --port || true)"
    PORT_SOURCE="launch-argv"
fi
if [[ ! "${BRIDGE_BASE_PORT}" =~ ^[0-9]+$ ]]; then
    BRIDGE_BASE_PORT=5555
    PORT_SOURCE="default"
fi
if [[ ! "${ARENAS}" =~ ^[0-9]+$ ]] || [[ "${ARENAS}" -lt 1 ]]; then
    # The expectation is UNKNOWN, so the fleet signal will refuse to render a
    # verdict. The scan still runs over the launcher's documented 25-pad range so
    # the operator sees a COUNT — an observation, never scored against a number
    # nobody read from this run.
    ARENAS=""
    ARENAS_SOURCE="unknown"
fi

SCAN_LOW="${BRIDGE_BASE_PORT}"
SCAN_HIGH=$(( BRIDGE_BASE_PORT + ${ARENAS:-25} - 1 ))

# The pidfile, read as text. Judged in Python: whether it is missing, empty or
# not a number is part of the verdict, not something to swallow here.
PID_FILE_EXISTS="false"
PID_RAW=""
if [[ -f "${PID_PATH}" ]]; then
    PID_FILE_EXISTS="true"
    PID_RAW="$(head -c 64 "${PID_PATH}" 2>/dev/null | tr -d '[:space:]' || true)"
fi

# `ps`, never `kill -0`. Signal 0 delivers no signal, but this script promises it
# sends none at all, and a promise with an asterisk on it is not one.
#
# `-ww` because the run-name check below reads the WHOLE command line: the
# driver's argv carries a 64-char sha256 and three absolute paths before
# `--run-name` appears, and a truncated command would read as "this pid belongs
# to another run" and ALARM all night on a healthy fleet. (Piped `ps -o command=`
# was measured unlimited on this Mac at 654 bytes; `-ww` removes the question.)
PS_COMMAND=""
PS_ETIME=""
if [[ "${PID_RAW}" =~ ^[0-9]+$ ]]; then
    PS_COMMAND="$(ps -ww -o command= -p "${PID_RAW}" 2>/dev/null | head -n 1 || true)"
    PS_ETIME="$(ps -ww -o etime= -p "${PID_RAW}" 2>/dev/null | tr -d '[:space:]' || true)"
fi

# Two `lsof` calls for the WHOLE port range rather than two per pad: at 25 pads
# the per-port form costs ~50 invocations, and a watcher that takes ten seconds
# will not be run often enough to matter. `-Fn` prints one `n<name>` field per
# socket; the ports are pulled out of it by `parse_lsof_ports` below, which is a
# pure function and is tested as one.
LSOF_AVAILABLE="false"
LSOF_LISTEN=""
LSOF_ESTABLISHED=""
if command -v lsof >/dev/null 2>&1; then
    LSOF_AVAILABLE="true"
    LSOF_LISTEN="$(lsof -nP -iTCP:"${SCAN_LOW}"-"${SCAN_HIGH}" -sTCP:LISTEN -Fn 2>/dev/null || true)"
    LSOF_ESTABLISHED="$(lsof -nP -iTCP:"${SCAN_LOW}"-"${SCAN_HIGH}" -sTCP:ESTABLISHED -Fn 2>/dev/null || true)"
fi

# ===========================================================================
# The judgement. Everything below the sentinel is pure, stdlib-only Python, fed
# entirely through the environment and through the files it names. It is piped
# on stdin rather than written to disk because this script may not write files —
# which is also why it cannot follow launch_selfplay.sh's emit-a-module pattern.
# ===========================================================================
set +e
WATCH_NOW="$(date +%s)" \
WATCH_RUN_NAME="${RUN_NAME}" \
WATCH_LOG_PATH="${LOG_PATH}" \
WATCH_PID_PATH="${PID_PATH}" \
WATCH_METRICS_PATH="${METRICS_PATH}" \
WATCH_ARGV_PATH="${ARGV_PATH}" \
WATCH_MEASUREMENTS_PATH="${MEASUREMENTS}" \
WATCH_PID_FILE_EXISTS="${PID_FILE_EXISTS}" \
WATCH_PID_RAW="${PID_RAW}" \
WATCH_PS_COMMAND="${PS_COMMAND}" \
WATCH_PS_ETIME="${PS_ETIME}" \
WATCH_ARENAS="${ARENAS}" \
WATCH_ARENAS_SOURCE="${ARENAS_SOURCE}" \
WATCH_EVAL_EVERY="${EVAL_EVERY}" \
WATCH_BASE_PORT="${BRIDGE_BASE_PORT}" \
WATCH_PORT_SOURCE="${PORT_SOURCE}" \
WATCH_SCAN_LOW="${SCAN_LOW}" \
WATCH_SCAN_HIGH="${SCAN_HIGH}" \
WATCH_LSOF_AVAILABLE="${LSOF_AVAILABLE}" \
WATCH_LSOF_LISTEN="${LSOF_LISTEN}" \
WATCH_LSOF_ESTABLISHED="${LSOF_ESTABLISHED}" \
WATCH_STALL_MINUTES="${STALL_MINUTES}" \
WATCH_WINDOW_MINUTES="${WINDOW_MINUTES}" \
"${PYTHON_BIN}" - <<'WATCH_VERDICT_PY'
"""watch_verdict — is this self-play run still healthy, and how do you know?

Five signals, four verdicts (OK / WARN / ALARM / UNKNOWN), one exit code. The
whole module is pure and stdlib-only apart from :func:`collect_document` and
:func:`main`, so ``tests/test_watch_selfplay.py`` drives every boundary over
synthetic documents without a server, a socket or a run.

THE RULE THIS MODULE IS BUILT AROUND: never report OK for something that was
not actually checked. A signal whose input is missing, unreadable or absent
returns UNKNOWN and says which file and which key, and UNKNOWN carries its own
exit code so a mistyped path cannot read as a clean night.

WHAT IS AND IS NOT OBSERVABLE FROM OUTSIDE THE DRIVER. This matters more than
any threshold below, because two of the five signals are shaped by it:

  * The driver's stderr lines carry NO timestamps (``_emit`` is a bare
    ``print``). Anything time-based therefore comes from
    ``runs/<run>/metrics.jsonl``, whose every record is
    ``{"step": ..., "wall_time": ..., ...}`` (``eval/logging.py``'s
    ``MetricsLogger.log``, flushed per row).
  * EPISODES are not observable mid-run at all. ``learner.received`` is never
    logged, no metrics row carries an episode counter, and the only episode
    total the driver ever prints is on its ``[multi done]`` teardown line. So
    "episodes/hour" cannot be MEASURED against a running fleet. Signal 4 judges
    the grad-step rate per live arena instead — which, because the baseline
    document supplies both an episode count and a grad-step count over the same
    wall clock, produces exactly the same ratio the episode comparison would —
    and RENDERS it in episodes/hour so it can be read against the canary's
    headline ``measured_episodes_per_arena_hour``. The rendered figure is
    labelled "implied" wherever it appears.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Tuple

# ---------------------------------------------------------------------------
# Verdict vocabulary.
# ---------------------------------------------------------------------------

OK = "OK"
WARN = "WARN"
ALARM = "ALARM"
UNKNOWN = "UNKNOWN"

#: Worst-wins ordering. ALARM outranks UNKNOWN deliberately: a failure you can
#: see is worse news than a signal you could not read, and the exit code should
#: name the loudest thing that is actually known.
SEVERITY: Dict[str, int] = {OK: 0, WARN: 1, UNKNOWN: 2, ALARM: 3}

EXIT_OK = 0
EXIT_ALARM = 1
EXIT_USAGE = 2
EXIT_UNKNOWN = 3

#: WARN exits 0 on purpose: this is meant to live in a `watch` loop, and a
#: signal that pages on every eval cycle stops being read. ALARM pages.
EXIT_FOR: Dict[str, int] = {
    OK: EXIT_OK,
    WARN: EXIT_OK,
    UNKNOWN: EXIT_UNKNOWN,
    ALARM: EXIT_ALARM,
}

# ---------------------------------------------------------------------------
# Thresholds. Every one of them is a judgement call; each is written down with
# the reason it was picked, so a future edit argues with the reason rather than
# with the number.
# ---------------------------------------------------------------------------

#: Both time windows are DERIVED from the run, because the healthy interval
#: between metrics rows is a property of the run and not a constant. The driver
#: writes a row only at its own boundaries, so that interval is
#: ``min(--checkpoint-every-grad-steps, --eval-every-grad-steps)`` divided by
#: the rate the fleet sustains. The launch gate writes both cadences into
#: launch_argv.txt and the canary measured the rate.
#:
#: A CONSTANT HERE WAS A BUG. The first version shipped 35 minutes. The gate
#: pins TARGET_PERIODIC_CHECKPOINTS = 20 across a 12-hour window, which puts the
#: healthy interval near 39 minutes at the M3 retry's measured 4,570 grad
#: steps/hour — so a healthy run ALARMed for a slice of every single interval,
#: and the operator's first check after launch read a hard ALARM.
STALL_INTERVAL_MULTIPLE = 2.0
WINDOW_INTERVAL_MULTIPLE = 4.0

#: Used ONLY when the cadence or the rate cannot be read. Generous on purpose:
#: an over-long stall window costs lateness on a real stall, while a short one
#: costs the operator's trust in every reading the script makes.
FALLBACK_STALL_MINUTES = 90.0
FALLBACK_WINDOW_MINUTES = 180.0

#: Clamps, so a garbage measurement can neither disable the stall check nor
#: revive the false-ALARM bug above.
MIN_STALL_MINUTES = 30.0
MAX_STALL_MINUTES = 240.0

#: Throughput fractions of the measured baseline. 0.75 matches the launch gate's
#: own `min_transitions_per_s_fraction` for the 25-pad smoke, so a WARN here and
#: a refusal there mean the same amount of shortfall. 0.50 is not a tuned
#: number: at half the rate a 12-hour window buys a 6-hour run, which changes
#: what the night can deliver rather than merely disappointing.
THROUGHPUT_WARN_FRACTION = 0.75
THROUGHPUT_ALARM_FRACTION = 0.50

#: Eval staleness, in multiples of the run's own configured cadence. One missed
#: boundary is a slow cycle; two is a pattern; four is the pinned-reference
#: failure mode where every cycle raises and is silently skipped.
EVAL_WARN_CADENCES = 2.0
EVAL_ALARM_CADENCES = 4.0

#: Never read more than this much of the driver log. The multi-arena path emits
#: only at boundaries so the file stays small, but a watcher that can be wedged
#: by an unexpectedly large file is not a watcher.
LOG_TAIL_BYTES = 8 * 1024 * 1024

#: The canary keys signal 4's baseline is built from. Each is written by T17's
#: `build_measurements` (scripts/canary_selfplay.sh); the test suite re-derives
#: that key list from the canary's own heredoc, so a rename upstream fails there
#: rather than here at 3am.
BASELINE_KEYS = (
    "training_grad_steps",
    "training_wall_seconds",
    "training_arenas",
)

#: Additional canary keys used only to RENDER the baseline in episodes/hour.
#: Their absence costs the rendering, never the verdict.
BASELINE_EPISODE_KEYS = (
    "training_episodes",
    "measured_episodes_per_arena_hour",
)

# ---------------------------------------------------------------------------
# The driver's own log lines, pinned to their producers in agent/train.py.
# `tests/test_watch_selfplay.py` RENDERS each of these from the f-string the
# producer actually holds (via `ast`) and asserts these patterns match it, so a
# reformat upstream fails a test instead of silently emptying a signal.
# ---------------------------------------------------------------------------

#: Any `[multi grad_step N]` line. The coarse "where is this run" reading, used
#: only as a fallback for the current step when metrics.jsonl cannot be read.
GRAD_STEP_RE = re.compile(r"\[multi grad_step (\d+)\]")

#: An eval cycle that COMPLETED (agent/train.py, the `_emit` after the
#: best-checkpoint block). `eval_grad_step` is the step the evaluated weights
#: were frozen at, not the learner's step now.
EVAL_DONE_RE = re.compile(
    r"\[multi grad_step (?P<step>\d+)\] "
    r"win_rate=(?P<win_rate>-?[\d.]+) "
    r"mean_len=(?P<mean_len>-?[\d.]+) "
    r"aim_invisible=(?P<aim_invisible>-?[\d.]+) "
    r"passed_m2=(?P<passed_m2>\S+) "
    r"opponent=(?P<opponent>\S+)"
)

#: An eval cycle that RAISED and was swallowed. This is the pinned-reference
#: failure mode's fingerprint: the run continues, the cycle "selected no
#: checkpoint and rated no match", and both series simply gap.
EVAL_SKIPPED_RE = re.compile(
    r"\[multi grad_step (?P<step>\d+)\] eval cycle SKIPPED: it raised "
    r"(?P<error>[^:\s]+)"
)

#: The self-play summary line. Both branches of its rated-match clause are
#: matched: the populated one and the "(0 rated matches - elo/learner_rated is
#: EMPTY)" one, which is a DIFFERENT claim from a flat rating and the only place
#: the two can be told apart.
SELFPLAY_RE = re.compile(
    r"\[multi grad_step (?P<step>\d+)\] selfplay: "
    r"elo_rated=(?P<elo_rated>-?[\d.]+) "
    r"\((?:(?P<rated>\d+) rated match\(es\)"
    r"|0 rated matches - elo/learner_rated is EMPTY)\) "
    r"elo_online=(?P<elo_online>-?[\d.]+) "
    r"pool=(?P<pool>\d+) "
    r"matches=(?P<matches>\d+)"
)

#: The driver's teardown line — the ONLY reliable completion signal, because the
#: process exit code is `0 if passed_m2 else 1` and a self-play run never clears
#: the M2 dummy gate. Same pattern the canary reads.
DRIVER_DONE_RE = re.compile(
    r"\[multi done\] reason=(?P<reason>\S+) episodes=(?P<episodes>\d+) "
    r"grad_steps=(?P<grad_steps>\d+) passed_m2=(?P<passed_m2>\S+) "
    r"checkpoints_saved=(?P<checkpoints_saved>\d+)"
)

#: The end-of-run eval summary. Present only AFTER the driver has finished, so
#: it is reported when it exists and is never the mid-run source.
LAST_EVAL_RE = re.compile(
    r"last eval: win_rate=(?P<win_rate>-?[\d.]+) "
    r"mean_len=(?P<mean_len>-?[\d.]+) "
    r"aim_invisible=(?P<aim_invisible>-?[\d.]+)"
)

#: The row the eval cycle writes on ENTRY. `_maybe_log_mean_epsilon` has exactly
#: one call site and it is the first statement of the eval block, so this key in
#: the newest row means a cycle has started and has not yet written its closing
#: self-play row.
#:
#: It is the WEAKER of the two in-flight signals and covers only the first
#: minutes of a cycle: once the cycle's first episode finishes, the eval track's
#: per-episode rows (see EVAL_EPISODE_KEYS) become the newest rows and take over
#: as positive evidence for the remaining ~90. A cycle that RAISED writes no
#: closing row at all — its failure leaves an `eval cycle SKIPPED` line in the
#: driver's LOG, not a row in this file — so that log line is what closes the
#: cycle for both signals. Pinned by test.
EVAL_START_METRIC = "train/epsilon_schedule"

#: The rated-Elo series (AC7) and its denominator, as `selfplay_log_row` writes
#: them. `rated_matches` at 0 is what separates "the learner stopped improving"
#: from "the series has no data at all"; both render as a flat line.
ELO_RATED_METRIC = "elo/learner_rated"
RATED_MATCHES_METRIC = "selfplay/rated_matches"

#: A metrics row belongs to the TRAINING series iff it carries a key under one
#: of these namespaces (`epsilon_log_row`, `selfplay_log_row` and
#: `selfplay_eval_cycle_row` write nothing outside them).
#:
#: THIS IS NOT A STYLISTIC CHOICE, IT IS FORCED. `agent/train.py` hands the run's
#: logger to the MAIN eval track (`evaluate(..., logger=logger)`) and
#: `eval/evaluate.py` logs one row PER EVAL EPISODE at ``step=episode_index``.
#: Those rows land in this same file with ``step`` running 0..n_episodes-1. A
#: watcher that reads ``step`` as a grad step therefore reports `grad_step 9` on
#: a run sitting at 300,000, drives the rate to zero or below, and fires its
#: loudest cross-signal note on a perfectly healthy fleet for the ~97 minutes an
#: eval cycle lasts, three times a night. (The REFERENCE tracks pass
#: ``logger=None`` with a comment naming exactly this collision; the main track
#: never got the same treatment. That is the RUN's quirk to fix another day —
#: this script only works around it.)
TRAINING_ROW_PREFIXES = ("train/", "elo/", "selfplay/")

#: The keys `eval/evaluate.py` writes for one eval EPISODE. A row carrying one
#: of these, newer than the newest training row, is POSITIVE evidence that an
#: eval cycle is running right now — strictly better evidence than anything the
#: training series can offer, because it arrives every episode instead of once.
EVAL_EPISODE_KEYS = (
    "episode_length",
    "episode_reward",
    "win",
    "aim_while_invisible",
)


def is_training_row(row: Mapping[str, Any]) -> bool:
    """Does this row carry the LEARNER's grad step, rather than an episode index?"""
    return any(
        str(key).startswith(TRAINING_ROW_PREFIXES) for key in row
    )


def is_eval_episode_row(row: Mapping[str, Any]) -> bool:
    """Is this one of `eval/evaluate.py`'s per-episode rows?"""
    if is_training_row(row):
        return False
    return any(key in row for key in EVAL_EPISODE_KEYS)


# ---------------------------------------------------------------------------
# Pure parsers.
# ---------------------------------------------------------------------------


def parse_etime(text: str) -> Optional[float]:
    """Seconds from BSD ``ps -o etime=`` (``[[dd-]hh:]mm:ss``), or ``None``.

    macOS `ps` has no `etimes` keyword (verified: "ps: etimes: keyword not
    found"), so the string is parsed rather than read as a number.
    """
    match = re.fullmatch(r"(?:(?:(\d+)-)?(\d+):)?(\d+):(\d+)", (text or "").strip())
    if match is None:
        return None
    days, hours, minutes, seconds = (int(g or 0) for g in match.groups())
    return float(((days * 24 + hours) * 60 + minutes) * 60 + seconds)


def format_duration(seconds: Optional[float]) -> str:
    """``4712`` -> ``1h18m``. ``None`` -> ``?``."""
    if seconds is None:
        return "?"
    total = int(max(0.0, float(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def format_clock(epoch: Optional[float]) -> str:
    """Local wall-clock for an epoch, or ``?``."""
    if epoch is None:
        return "?"
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(epoch)))


def _num(value: Any) -> Optional[float]:
    """A finite float, or ``None``. Booleans are NOT numbers here."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value == int(value):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def parse_lsof_ports(text: str, low: int, high: int) -> List[int]:
    """Ports inside ``[low, high]`` named by an ``lsof -Fn`` field dump.

    An `n` field is either ``*:5555`` / ``127.0.0.1:5555`` (a listener) or
    ``local->remote`` (an established pair, of which BOTH endpoints appear
    because loopback traffic has both ends on this machine). Only the text after
    each endpoint's LAST colon is read as a port, so an IPv6 address whose
    hextets happen to be decimal cannot manufacture a pad.
    """
    found = set()
    for line in (text or "").splitlines():
        if not line.startswith("n"):
            continue
        for endpoint in line[1:].split("->"):
            tail = endpoint.rsplit(":", 1)[-1].strip()
            if not tail.isdigit():
                continue
            port = int(tail)
            if low <= port <= high:
                found.add(port)
    return sorted(found)


def parse_launch_argv(text: str) -> Dict[str, str]:
    """``{flag: value}`` from the launch gate's argv file (one element per line).

    A flag whose next element is another flag is recorded with an empty value
    rather than swallowing it: `agent.train` has no valueless flags in this argv,
    so that shape means the file is malformed, and losing the following flag
    would hide it.
    """
    elements = [line.strip() for line in (text or "").splitlines()]
    elements = [element for element in elements if element]
    flags: Dict[str, str] = {}
    index = 0
    while index < len(elements):
        element = elements[index]
        if not element.startswith("--"):
            index += 1
            continue
        following = elements[index + 1] if index + 1 < len(elements) else ""
        if following.startswith("--"):
            flags[element] = ""
            index += 1
            continue
        flags[element] = following
        index += 2
    return flags


def parse_metrics_jsonl(text: str) -> Dict[str, Any]:
    """Rows from a metrics.jsonl that is being APPENDED TO while it is read.

    The last line of a live file can be half-written, so a record that does not
    decode is counted and dropped — never raised, and never allowed to empty the
    signal. Rows are ordered by ``wall_time`` and then ``step``: the loop writes
    two rows at one step at every eval boundary, and the later of the two must
    sort later.

    TWO SERIES SHARE THIS FILE and they are separated here, once. The training
    rows are stepped by the learner's grad step; `eval/evaluate.py`'s per-episode
    rows are stepped by an EPISODE INDEX starting at 0. See
    :data:`TRAINING_ROW_PREFIXES` for why, and for what reading them as one
    series does to a healthy run.
    """
    rows: List[Dict[str, Any]] = []
    torn = 0
    stepless = 0
    for line in (text or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            record = json.loads(stripped)
        except ValueError:
            torn += 1
            continue
        if not isinstance(record, dict):
            torn += 1
            continue
        step = _int(record.get("step"))
        wall = _num(record.get("wall_time"))
        if step is None or wall is None:
            # A row with no step or no timestamp carries no progress
            # information. Not torn — just not evidence — so it is counted
            # separately rather than inflating the torn tally, and the count is
            # reported so "no usable row" never reads as "no rows".
            stepless += 1
            continue
        row = dict(record)
        row["step"] = step
        row["wall_time"] = wall
        rows.append(row)
    rows.sort(key=lambda item: (item["wall_time"], item["step"]))
    return {
        "rows": rows,
        "torn_lines": torn,
        "stepless_rows": stepless,
        # The two series that share this file, separated once, here.
        "training_rows": [row for row in rows if is_training_row(row)],
        "eval_episode_rows": [row for row in rows if is_eval_episode_row(row)],
    }


def parse_run_log(text: str) -> Dict[str, Any]:
    """Everything the watcher reads out of the driver's stderr log.

    Line-at-a-time regex over text decoded with ``errors="replace"``: this file
    is also being appended to, and a torn final line must cost that line and
    nothing else.
    """
    body = text or ""
    grad_steps = [int(match.group(1)) for match in GRAD_STEP_RE.finditer(body)]

    evals = [
        {
            "step": int(match.group("step")),
            "win_rate": float(match.group("win_rate")),
            "mean_len": float(match.group("mean_len")),
            "aim_invisible": float(match.group("aim_invisible")),
            "passed_m2": match.group("passed_m2"),
            "opponent": match.group("opponent"),
        }
        for match in EVAL_DONE_RE.finditer(body)
    ]
    skips = [
        {"step": int(match.group("step")), "error": match.group("error")}
        for match in EVAL_SKIPPED_RE.finditer(body)
    ]
    selfplay = [
        {
            "step": int(match.group("step")),
            "elo_rated": float(match.group("elo_rated")),
            "rated_matches": int(match.group("rated") or 0),
            "elo_online": float(match.group("elo_online")),
            "pool_size": int(match.group("pool")),
            "matches_scored": int(match.group("matches")),
        }
        for match in SELFPLAY_RE.finditer(body)
    ]

    done: Optional[Dict[str, Any]] = None
    for match in DRIVER_DONE_RE.finditer(body):
        done = {
            "reason": match.group("reason"),
            "episodes": int(match.group("episodes")),
            "grad_steps": int(match.group("grad_steps")),
            "passed_m2": match.group("passed_m2"),
            "checkpoints_saved": int(match.group("checkpoints_saved")),
        }
    last_eval: Optional[Dict[str, Any]] = None
    for match in LAST_EVAL_RE.finditer(body):
        last_eval = {
            "win_rate": float(match.group("win_rate")),
            "mean_len": float(match.group("mean_len")),
            "aim_invisible": float(match.group("aim_invisible")),
        }

    return {
        "max_grad_step": max(grad_steps) if grad_steps else None,
        "grad_step_lines": len(grad_steps),
        "evals": evals,
        "skips": skips,
        "selfplay": selfplay,
        "done": done,
        "last_eval": last_eval,
    }


# ---------------------------------------------------------------------------
# Derived facts. Computed once so signals 2 and 4 cannot disagree about the
# rate, and exposed on the verdict so the tests can assert the arithmetic.
# ---------------------------------------------------------------------------


def baseline_per_arena_hour(baseline: Mapping[str, Any]) -> Optional[float]:
    """Grad steps per arena-hour from the canary's measurement, or ``None``.

    One helper rather than two call sites, so the stall window and the
    throughput verdict can never disagree about what the baseline says.
    """
    grad_steps = _num(baseline.get("training_grad_steps"))
    wall = _num(baseline.get("training_wall_seconds"))
    arenas = _num(baseline.get("training_arenas"))
    if grad_steps is None or not wall or wall <= 0.0 or not arenas or arenas <= 0.0:
        return None
    return grad_steps / (wall / 3600.0) / arenas


def expected_row_interval(document: Mapping[str, Any]) -> Dict[str, Any]:
    """How long a HEALTHY run should go between TRAINING rows, and from what.

    The driver writes a training row only at a boundary it already has: the
    periodic checkpoint (`_maybe_log_selfplay`) and the eval cycle (the epsilon
    row on entry, the self-play and cycle rows on exit). So the healthy interval
    is the TIGHTER of the two cadences, in grad steps, over the rate the fleet
    sustains. The launch gate writes both cadences into launch_argv.txt and the
    canary measured the rate, so both inputs are already on disk.

    Returns the interval AND every input to it, so the report can show its
    working instead of asserting a number the operator cannot check.
    """
    argv = (document.get("launch_argv") or {}).get("flags") or {}
    baseline = ((document.get("measurements") or {}).get("document")) or {}
    arenas = _int((document.get("fleet") or {}).get("expected_arenas"))

    out: Dict[str, Any] = {
        "interval_seconds": None,
        "cadence_flag": None,
        "cadence_grad_steps": None,
        "expected_grad_steps_per_hour": None,
        "reason": None,
    }

    cadences = []
    for flag in ("--checkpoint-every-grad-steps", "--eval-every-grad-steps"):
        value = _int(argv.get(flag))
        if value is not None and value > 0:
            cadences.append((value, flag))
    if not cadences:
        out["reason"] = "no checkpoint/eval cadence in the launch argv"
        return out
    cadence, flag = min(cadences)
    out["cadence_grad_steps"] = cadence
    out["cadence_flag"] = flag

    per_arena = baseline_per_arena_hour(baseline)
    if per_arena is None or per_arena <= 0.0:
        out["reason"] = "no measured grad-step rate in the canary baseline"
        return out
    if arenas is None or arenas < 1:
        out["reason"] = "the launched arena count is unknown"
        return out

    rate = per_arena * float(arenas)
    out["expected_grad_steps_per_hour"] = rate
    out["interval_seconds"] = float(cadence) / rate * 3600.0
    return out


def derive_facts(document: Mapping[str, Any]) -> Dict[str, Any]:
    """The numbers every signal shares, derived once."""
    now = _num(document.get("now")) or 0.0
    thresholds = document.get("thresholds") or {}
    interval = expected_row_interval(document)
    row_interval = interval["interval_seconds"]

    # --- the two time windows, DERIVED unless the operator overrode them -----
    stall_override = _num(thresholds.get("stall_minutes"))
    if stall_override:
        stall_seconds, stall_source = stall_override * 60.0, "--stall-minutes"
    elif row_interval:
        stall_seconds = min(
            MAX_STALL_MINUTES * 60.0,
            max(MIN_STALL_MINUTES * 60.0, STALL_INTERVAL_MULTIPLE * row_interval),
        )
        stall_source = "derived"
    else:
        stall_seconds, stall_source = FALLBACK_STALL_MINUTES * 60.0, "fallback"

    window_override = _num(thresholds.get("window_minutes"))
    if window_override:
        window_seconds, window_source = window_override * 60.0, "--window-minutes"
    elif row_interval:
        window_seconds = min(
            12.0 * 3600.0,
            max(MIN_STALL_MINUTES * 60.0, WINDOW_INTERVAL_MULTIPLE * row_interval),
        )
        window_source = "derived"
    else:
        window_seconds, window_source = FALLBACK_WINDOW_MINUTES * 60.0, "fallback"

    metrics = document.get("metrics") or {}
    rows: List[Dict[str, Any]] = list(metrics.get("rows") or [])
    training: List[Dict[str, Any]] = list(metrics.get("training_rows") or [])
    eval_rows: List[Dict[str, Any]] = list(metrics.get("eval_episode_rows") or [])
    log = document.get("log") or {}

    facts: Dict[str, Any] = {
        "now": now,
        "stall_seconds": stall_seconds,
        "stall_source": stall_source,
        "window_seconds": window_seconds,
        "window_source": window_source,
        "row_interval": interval,
        "row_interval_seconds": row_interval,
        "metrics_rows": len(rows),
        "training_rows": len(training),
        "eval_episode_rows": len(eval_rows),
        "current_grad_step": None,
        "grad_step_source": None,
        "frozen_seconds": None,
        "grad_steps_per_hour": None,
        "rate_span_seconds": None,
        "rate_span_steps": None,
        "rate_anchor": None,
        "rate_discontinuity": False,
        "eval_in_flight": False,
        "eval_in_flight_evidence": None,
        "rate_excludes_in_flight_rows": False,
        "alive_arenas": None,
        "expected_arenas": _int((document.get("fleet") or {}).get("expected_arenas")),
    }

    # --- the grad step, from the TRAINING series only ------------------------
    # Never from `rows`: the eval track's per-episode rows share this file and
    # are stepped 0..n_episodes-1 (see TRAINING_ROW_PREFIXES).
    current: Optional[int] = None
    if training:
        current = _int(training[-1].get("step"))
        facts["current_grad_step"] = current
        facts["grad_step_source"] = "metrics"
        # When the step LAST CHANGED: the earliest row in the CONTIGUOUS trailing
        # run of rows already carrying the current step. Rows keep flowing while
        # the step sits still (the eval boundary writes two of them), so "no new
        # rows" is the wrong question and "how long has this number been frozen"
        # is the right one. Walked from the END rather than the start because
        # metrics.jsonl is opened in APPEND mode: a run restarted into the same
        # name leaves an older series in front of this one, and an old row that
        # happens to share the current step number would otherwise date the
        # freeze to yesterday.
        changed_at = float(training[-1]["wall_time"])
        for row in reversed(training):
            if _int(row.get("step")) != current:
                break
            changed_at = float(row["wall_time"])
        facts["frozen_seconds"] = max(0.0, now - changed_at)
    elif log.get("max_grad_step") is not None:
        facts["current_grad_step"] = _int(log.get("max_grad_step"))
        facts["grad_step_source"] = "log"

    # --- is an eval cycle running RIGHT NOW? --------------------------------
    # Two kinds of evidence, strongest first. An eval-episode row newer than the
    # newest training row is POSITIVE proof: those arrive once per episode, all
    # cycle long. The epsilon row only covers the gap before the first episode
    # finishes. A cycle that RAISED writes no closing row, so its entry evidence
    # would otherwise persist forever and soften every stall from here on — the
    # SKIPPED line at that grad step is what closes it.
    if rows:
        newest = rows[-1]
        if is_eval_episode_row(newest):
            facts["eval_in_flight_evidence"] = "an eval-episode row is the newest row"
        elif EVAL_START_METRIC in newest:
            facts["eval_in_flight_evidence"] = (
                "the cycle's entry row is the newest row"
            )
        if facts["eval_in_flight_evidence"] is not None:
            skipped_at_or_after = [
                skip
                for skip in (log.get("skips") or [])
                if _int(skip.get("step")) is not None
                and _int(skip.get("step")) >= (current or 0)
            ]
            if skipped_at_or_after:
                facts["eval_in_flight_evidence"] = None
            else:
                facts["eval_in_flight"] = True

    # --- the rate, over CLOSED training boundaries only ---------------------
    if training:
        # While a cycle is in flight the learner keeps stepping — the eval pauses
        # one designated arena, not the learner — but no training row is written
        # until the cycle ends, so the flat stretch after the entry row measures
        # nothing. Drop it and anchor the window at the last CLOSED boundary.
        rate_rows = list(training)
        while (
            facts["eval_in_flight"] and rate_rows and EVAL_START_METRIC in rate_rows[-1]
        ):
            rate_rows.pop()
        facts["rate_excludes_in_flight_rows"] = len(rate_rows) != len(training)

        if rate_rows:
            anchor = (
                float(rate_rows[-1]["wall_time"]) if facts["eval_in_flight"] else now
            )
            facts["rate_anchor"] = anchor
            window = [
                row
                for row in rate_rows
                if anchor - window_seconds <= float(row["wall_time"]) <= anchor
            ]
            # NO fallback to "the last two rows whenever they were". A rate built
            # from rows three hours old reads OK while GRAD STEP alarms, which
            # suppresses the cross-signal note in exactly the total-collector-
            # death case this script exists for. Outside the window there is no
            # current rate, and the honest answer is that we do not know it.
            if len(window) >= 2:
                for earlier, later in zip(window, window[1:]):
                    if (_int(later.get("step")) or 0) < (_int(earlier.get("step")) or 0):
                        # A restart into the same run name appends a SECOND
                        # series to this file. Two series in one window give a
                        # negative delta and a nonsense rate; that is a
                        # discontinuity to report, never a number to publish.
                        facts["rate_discontinuity"] = True
                        break
                if not facts["rate_discontinuity"]:
                    span = float(window[-1]["wall_time"]) - float(window[0]["wall_time"])
                    steps = (_int(window[-1].get("step")) or 0) - (
                        _int(window[0].get("step")) or 0
                    )
                    facts["rate_span_seconds"] = span
                    facts["rate_span_steps"] = steps
                    if span > 0.0:
                        facts["grad_steps_per_hour"] = steps / (span / 3600.0)

    fleet = document.get("fleet") or {}
    if fleet.get("lsof_available"):
        facts["alive_arenas"] = len(list(fleet.get("listening_ports") or []))

    return facts


# ---------------------------------------------------------------------------
# Signals.
# ---------------------------------------------------------------------------


class Signal(NamedTuple):
    """One check: what it is, what it decided, and the number behind it."""

    key: str
    title: str
    verdict: str
    headline: str
    detail: Tuple[str, ...]


class Verdict(NamedTuple):
    """Every signal, the shared arithmetic, and the cross-signal reading."""

    signals: Tuple[Signal, ...]
    facts: Dict[str, Any]
    notes: Tuple[str, ...]

    @property
    def worst(self) -> str:
        return max((s.verdict for s in self.signals), key=lambda v: SEVERITY[v])

    @property
    def exit_code(self) -> int:
        return EXIT_FOR[self.worst]

    def signal(self, key: str) -> Signal:
        for item in self.signals:
            if item.key == key:
                return item
        raise KeyError(key)


def _worse(left: str, right: str) -> str:
    return left if SEVERITY[left] >= SEVERITY[right] else right


def _signal_liveness(document: Mapping[str, Any]) -> Signal:
    """Is the driver process still there, and is it still THIS run's?"""
    title = "LIVENESS"
    process = document.get("process") or {}
    log = document.get("log") or {}
    pid_path = (document.get("paths") or {}).get("pid", "?")
    run_name = str(document.get("run_name") or "")

    if not process.get("pid_file_exists"):
        return Signal(
            "liveness",
            title,
            UNKNOWN,
            f"no pidfile at {pid_path}",
            (
                "launch_selfplay.sh launch writes the pid there. If the run was "
                "started another way, pass --pid-file, or --run-name if it only "
                "used a different name.",
            ),
        )

    raw = str(process.get("pid_raw") or "")
    pid = _int(raw)
    if pid is None or pid <= 0:
        return Signal(
            "liveness",
            title,
            UNKNOWN,
            f"pidfile {pid_path} holds {raw!r}, which is not a pid",
            ("Nothing can be said about the driver until that file is readable.",),
        )

    command = str(process.get("command") or "")
    etime = str(process.get("etime") or "")
    if not command:
        done = log.get("done")
        if not log.get("read"):
            return Signal(
                "liveness",
                title,
                ALARM,
                f"pid {pid} is GONE and the log could not be read",
                (
                    f"log: {log.get('error') or 'unreadable'}",
                    "Whether the run finished or died cannot be told apart from "
                    "here. Read the log by hand before restarting anything.",
                ),
            )
        if done:
            return Signal(
                "liveness",
                title,
                WARN,
                f"pid {pid} is gone; the driver logged [multi done] "
                f"reason={done.get('reason')}",
                (
                    f"episodes={done.get('episodes')} "
                    f"grad_steps={done.get('grad_steps')} "
                    f"checkpoints_saved={done.get('checkpoints_saved')}",
                    "The run ENDED on its own terms. Exit code 1 from the driver "
                    "is not failure here: it means no eval cleared the M2 gate, "
                    "which is defined against the stationary dummy.",
                ),
            )
        crashed = [
            "The driver did not finish: it was killed or it crashed. The "
            "teardown line is the only reliable completion signal, because the "
            "process exit code is `0 if passed_m2 else 1` and a self-play run "
            "never clears the M2 dummy gate.",
        ]
        if log.get("truncated"):
            crashed.append(
                "NOTE: only the tail of the log was read, so an earlier "
                "[multi done] line would have been missed."
            )
        return Signal(
            "liveness",
            title,
            ALARM,
            f"pid {pid} is GONE and the log has no [multi done] line",
            tuple(crashed),
        )

    if "agent.train" not in command:
        return Signal(
            "liveness",
            title,
            ALARM,
            f"pid {pid} is alive but is NOT the driver",
            (
                f"command: {command}",
                "The pidfile is stale and the pid has been recycled. The run "
                "itself is gone.",
            ),
        )
    if run_name and f" --run-name {run_name} " not in f" {command} ":
        return Signal(
            "liveness",
            title,
            ALARM,
            f"pid {pid} runs agent.train, but not for {run_name}",
            (
                f"command: {command}",
                "This pidfile does not belong to the run being watched.",
            ),
        )

    return Signal(
        "liveness",
        title,
        OK,
        f"pid {pid} alive, up {format_duration(parse_etime(etime))} (etime {etime})",
        (f"command: {command}",),
    )


def _signal_grad_step(document: Mapping[str, Any], facts: Mapping[str, Any]) -> Signal:
    """Is the learner still taking gradient steps, and how fast?"""
    title = "GRAD STEP"
    metrics = document.get("metrics") or {}
    metrics_path = (document.get("paths") or {}).get("metrics", "?")
    stall_seconds = float(facts["stall_seconds"])
    process = document.get("process") or {}
    up_seconds = parse_etime(str(process.get("etime") or ""))

    if not metrics.get("read"):
        fallback = (
            f"the log's newest is grad_step {facts['current_grad_step']:,}"
            if facts.get("current_grad_step") is not None
            else "the log carries no [multi grad_step] line either"
        )
        return Signal(
            "grad_step",
            title,
            UNKNOWN,
            f"no metrics at {metrics_path}",
            (
                f"reason: {metrics.get('error') or 'unreadable'}",
                f"{fallback}. The driver's stderr carries no timestamps, so "
                "without this file neither the rate nor the stall can be "
                "measured at all.",
                "The launch gate pins --log-backend jsonl; a run started without "
                "it writes nothing here.",
            ),
        )

    training = int(facts["training_rows"])
    detail: List[str] = []
    if metrics.get("torn_lines"):
        detail.append(
            f"{metrics['torn_lines']} unparseable line(s) skipped (the file is "
            "being appended to while it is read; a torn tail costs that line "
            "only)."
        )
    if metrics.get("stepless_rows"):
        detail.append(
            f"{metrics['stepless_rows']} row(s) carried no step or no "
            "wall_time and were ignored."
        )

    if not training:
        # NOT "no rows": the eval track's per-episode rows share this file, so a
        # file can be full of rows and still carry no grad step at all.
        warming = up_seconds is not None and up_seconds < stall_seconds
        detail.append(
            f"{facts['metrics_rows']} row(s) in the file, of which "
            f"{facts['eval_episode_rows']} are eval-episode rows (stepped by "
            "EPISODE, not by grad step) and 0 are training rows."
        )
        detail.append(
            "The first training row lands at the first checkpoint or eval "
            "boundary, which is after the min_replay warm-up — silence before "
            "that is expected."
        )
        if facts.get("current_grad_step") is not None:
            detail.append(
                f"the log's newest is grad_step {facts['current_grad_step']:,} "
                f"(from {facts.get('grad_step_source')})"
            )
        return Signal(
            "grad_step",
            title,
            WARN if warming else ALARM,
            (
                f"no training row yet, {format_duration(up_seconds)} into the run"
                if warming
                else f"NO training row in {metrics_path}"
            ),
            tuple(detail),
        )

    current = facts["current_grad_step"]
    frozen = facts["frozen_seconds"]
    rate = facts.get("grad_steps_per_hour")
    span = facts.get("rate_span_seconds")

    if facts.get("rate_discontinuity"):
        detail.append(
            "rate: NOT measurable — the grad step goes BACKWARDS inside the "
            f"window. {metrics_path} is opened in append mode, so this is a run "
            "restarted into the same name and two series are interleaved here."
        )
    elif rate is None:
        detail.append(
            f"rate: not measurable — {training} training row(s), and fewer than "
            f"two of them fall inside the {format_duration(facts['window_seconds'])} "
            "window."
        )
    else:
        detail.append(
            f"rate: {rate:,.0f} grad steps/hour over the last "
            f"{format_duration(span)} ({facts['rate_span_steps']:,} steps)"
            + (
                ", measured to the last CLOSED boundary (the in-flight eval "
                "cycle has written no row to end the interval)."
                if facts.get("rate_excludes_in_flight_rows")
                else "."
            )
        )
    detail.append(
        f"{training} training row(s) + {facts['eval_episode_rows']} eval-episode "
        f"row(s) in {metrics_path}."
    )
    detail.append(_stall_window_note(facts))

    if frozen is not None and frozen > stall_seconds:
        if facts.get("eval_in_flight"):
            return Signal(
                "grad_step",
                title,
                WARN,
                f"grad_step {current:,} frozen for {format_duration(frozen)}, but "
                "an eval cycle is in flight",
                tuple(
                    detail
                    + [
                        f"evidence: {facts.get('eval_in_flight_evidence')}, and "
                        "no SKIPPED line has closed that cycle. The cycle fights "
                        "the scripted yardstick plus one leg per pinned "
                        "reference and writes no training row until it ends.",
                    ]
                ),
            )
        return Signal(
            "grad_step",
            title,
            ALARM,
            f"grad_step {current:,} has NOT moved in {format_duration(frozen)} "
            f"(stall window {format_duration(stall_seconds)})",
            tuple(
                detail
                + [
                    "No eval cycle is in flight to explain it. The learner is "
                    "wedged, starved of episodes, or gone.",
                ]
            ),
        )

    return Signal(
        "grad_step",
        title,
        OK,
        f"grad_step {current:,}, last moved {format_duration(frozen)} ago",
        tuple(detail),
    )


def _stall_window_note(facts: Mapping[str, Any]) -> str:
    """One line showing where the stall window came from — never just a number."""
    interval = facts.get("row_interval") or {}
    stall = format_duration(facts.get("stall_seconds"))
    source = facts.get("stall_source")
    if source == "--stall-minutes":
        return f"stall window {stall} (operator override)."
    if source == "derived":
        return (
            f"stall window {stall} = {STALL_INTERVAL_MULTIPLE:g}x the "
            f"{format_duration(interval.get('interval_seconds'))} this run should "
            f"go between training rows ({interval.get('cadence_flag')} "
            f"{interval.get('cadence_grad_steps'):,} at the canary's "
            f"{interval.get('expected_grad_steps_per_hour'):,.0f} grad steps/hour "
            f"for {facts.get('expected_arenas')} arenas)."
        )
    return (
        f"stall window {stall} (FALLBACK constant: {interval.get('reason')}). "
        "The run's own cadence would give a better number."
    )


def _signal_fleet(document: Mapping[str, Any], facts: Mapping[str, Any]) -> Signal:
    """Are all the bridges the run was launched with still listening?"""
    title = "FLEET"
    fleet = document.get("fleet") or {}
    base = _int(fleet.get("base_port"))
    expected = _int(fleet.get("expected_arenas"))
    listening = sorted(_int(p) for p in (fleet.get("listening_ports") or []))
    attached = {_int(p) for p in (fleet.get("attached_ports") or [])}

    caveat = (
        "A listening bridge is not proof its collector thread is alive: a "
        "collector that raises a non-BridgeError dies inside the driver and "
        "leaves both the bridge process and its socket standing. THROUGHPUT is "
        "what catches that."
    )

    if not fleet.get("lsof_available"):
        return Signal(
            "fleet",
            title,
            UNKNOWN,
            "lsof is not on PATH, so no bridge can be counted",
            (
                "The fleet is counted by LISTENER inspection because a connect "
                "probe would DESTROY the incumbent client (BridgeServer accepts "
                "exactly one). There is no safe fallback.",
            ),
        )

    if expected is None or expected < 1:
        return Signal(
            "fleet",
            title,
            UNKNOWN,
            f"{len(listening)} bridge listener(s) found, but the launched arena "
            "count is unknown",
            (
                f"scanned ports {fleet.get('scan_low')}-{fleet.get('scan_high')}; "
                f"found: {' '.join(str(p) for p in listening) or 'none'}",
                f"no usable --arenas in "
                f"{(document.get('paths') or {}).get('argv', '?')}. Pass "
                "--arenas N to give this count something to be scored against.",
                caveat,
            ),
        )

    wanted = list(range(base or 0, (base or 0) + expected))
    present = set(listening)
    missing = [port for port in wanted if port not in present]
    unattached = [port for port in wanted if port in present and port not in attached]
    source = fleet.get("expected_arenas_source") or "?"
    port_source = fleet.get("base_port_source") or "?"
    detail = [
        f"expected {expected} arena(s) on ports {wanted[0]}-{wanted[-1]} "
        f"(arenas from: {source}; base port from: {port_source}).",
        f"{expected - len(missing)} listening, "
        f"{expected - len(missing) - len(unattached)} of those with a client "
        "attached.",
        caveat,
    ]

    if missing:
        return Signal(
            "fleet",
            title,
            ALARM,
            f"{len(missing)} of {expected} bridges are GONE "
            f"({expected - len(missing)} listening)",
            tuple(
                [f"missing ports: {' '.join(str(p) for p in missing)}"]
                + detail
                + [
                    "The launcher restarts a pad's bridge only from inside a "
                    "collector's recovery path, so a bridge that stays down is "
                    "a pad nothing is collecting from.",
                ]
            ),
        )

    if unattached:
        return Signal(
            "fleet",
            title,
            WARN,
            f"all {expected} bridges listening, but {len(unattached)} have NO "
            "client attached",
            tuple(
                [f"unattached ports: {' '.join(str(p) for p in unattached)}"]
                + detail
                + [
                    "Transient during a bridge relaunch. Persisting across two "
                    "checks means those pads are idle.",
                ]
            ),
        )

    return Signal(
        "fleet",
        title,
        OK,
        f"{expected}/{expected} bridges listening, all with a client attached",
        tuple(detail),
    )


def _signal_throughput(
    document: Mapping[str, Any], facts: Mapping[str, Any]
) -> Signal:
    """Is the measured rate near what the canary measured, per LIVE arena?"""
    title = "THROUGHPUT"
    measurements = document.get("measurements") or {}
    path = (document.get("paths") or {}).get("measurements", "?")

    if not measurements.get("read"):
        return Signal(
            "throughput",
            title,
            UNKNOWN,
            f"no baseline at {path}",
            (
                f"reason: {measurements.get('error') or 'unreadable'}",
                "The canary writes it; without it there is no measured number "
                "to compare against, and this check refuses to substitute a "
                "constant.",
            ),
        )

    baseline_doc = measurements.get("document") or {}
    missing_keys = [key for key in BASELINE_KEYS if _num(baseline_doc.get(key)) is None]
    if missing_keys:
        return Signal(
            "throughput",
            title,
            UNKNOWN,
            f"the baseline at {path} is missing {', '.join(missing_keys)}",
            (
                "Those keys are how the canary records what it measured. A "
                "baseline that does not carry them describes no run.",
            ),
        )

    rate = facts.get("grad_steps_per_hour")
    alive = facts.get("alive_arenas")
    if facts.get("rate_discontinuity"):
        return Signal(
            "throughput",
            title,
            UNKNOWN,
            "the grad step goes BACKWARDS inside the rate window",
            (
                f"{(document.get('paths') or {}).get('metrics', '?')} is opened "
                "in append mode, so a run RESTARTED into the same name leaves "
                "two series interleaved in it. A rate across that boundary is "
                "arithmetic on two different runs, not a measurement.",
                "Point --metrics at a clean file, or watch a fresh --run-name.",
            ),
        )
    if rate is None:
        return Signal(
            "throughput",
            title,
            UNKNOWN,
            "the current rate could not be measured, so nothing can be compared",
            ("See GRAD STEP for why.",),
        )
    if alive is None or alive < 1:
        return Signal(
            "throughput",
            title,
            UNKNOWN,
            "no live arena count, so the expectation cannot be scaled",
            (
                "Scaling by the arenas REQUESTED rather than the arenas alive "
                "would report a dwindled fleet as a throughput problem. See "
                "FLEET.",
            ),
        )

    grad_steps = float(baseline_doc["training_grad_steps"])
    wall_seconds = float(baseline_doc["training_wall_seconds"])
    arenas = float(baseline_doc["training_arenas"])
    if wall_seconds <= 0.0 or arenas <= 0.0:
        return Signal(
            "throughput",
            title,
            UNKNOWN,
            f"the baseline at {path} is not usable "
            f"(wall_seconds={wall_seconds:g}, arenas={arenas:g})",
            ("A rate cannot be built from a zero denominator.",),
        )

    # Per ARENA on both sides, which is what makes a 4-pad canary comparable to a
    # 25-pad night. The project's own arena sweep recorded that scaling as linear
    # with no knee out to 25 pads (600 s confirm at N=25: 121.95 transitions/s
    # aggregate, 4.8782 per arena), and the launch gate projects the canary the
    # same way. The baseline's own arena count is printed below so the reader can
    # see how far it was scaled.
    baseline_per_arena = baseline_per_arena_hour(baseline_doc) or 0.0
    observed_per_arena = rate / float(alive)
    ratio = observed_per_arena / baseline_per_arena if baseline_per_arena > 0 else None
    if ratio is None:
        return Signal(
            "throughput",
            title,
            UNKNOWN,
            f"the baseline at {path} computes to a zero rate",
            ("Nothing can be a fraction of zero.",),
        )

    detail = [
        f"observed {rate:,.0f} grad steps/hour over "
        f"{alive} live arena(s) = {observed_per_arena:,.1f}/arena/hour.",
        f"baseline {baseline_per_arena:,.1f}/arena/hour "
        f"({grad_steps:,.0f} grad steps / {wall_seconds:,.0f}s / "
        f"{arenas:g} arenas, from {path}).",
    ]

    episodes = _num(baseline_doc.get("training_episodes"))
    per_arena_hour = _num(baseline_doc.get("measured_episodes_per_arena_hour"))
    if episodes is not None and grad_steps > 0 and per_arena_hour is not None:
        implied = rate * (episodes / grad_steps)
        expected = per_arena_hour * float(alive)
        detail.append(
            f"implied {implied:,.0f} episodes/hour vs {expected:,.0f} expected "
            f"at {alive} arena(s). IMPLIED, not measured: the driver logs no "
            "episode counter before its [multi done] line, so the episode "
            "figure is the grad-step rate times the canary's own "
            "episodes-per-grad-step. It moves the ratio not at all."
        )
    detail.append(
        "The baseline is a FLOOR: the canary's wall clock includes its replay "
        "warm-up, during which it took no gradient steps. A healthy long run "
        "should beat it, not merely match it."
    )

    if ratio < THROUGHPUT_ALARM_FRACTION:
        verdict = ALARM
    elif ratio < THROUGHPUT_WARN_FRACTION:
        verdict = WARN
    else:
        verdict = OK
    return Signal(
        "throughput",
        title,
        verdict,
        f"{ratio:.2f}x the measured baseline, per live arena",
        tuple(detail),
    )


def _signal_eval(document: Mapping[str, Any], facts: Mapping[str, Any]) -> Signal:
    """Has an eval cycle COMPLETED recently, and does the rated series have data?"""
    title = "EVAL"
    log = document.get("log") or {}
    argv = document.get("launch_argv") or {}
    argv_path = (document.get("paths") or {}).get("argv", "?")

    cadence = _int(document.get("eval_every_grad_steps"))
    if cadence is None:
        if not argv.get("read"):
            return Signal(
                "eval",
                title,
                UNKNOWN,
                f"no eval cadence: {argv_path} could not be read",
                (
                    f"reason: {argv.get('error') or 'unreadable'}",
                    "Staleness is measured in multiples of the run's OWN "
                    "--eval-every-grad-steps. Pass --eval-every-grad-steps N to "
                    "check it anyway.",
                ),
            )
        return Signal(
            "eval",
            title,
            UNKNOWN,
            f"no --eval-every-grad-steps in {argv_path}",
            ("Pass --eval-every-grad-steps N to check staleness anyway.",),
        )
    if cadence < 1:
        return Signal(
            "eval",
            title,
            UNKNOWN,
            f"the run was launched with --eval-every-grad-steps {cadence}",
            (
                "There is no cadence for an eval to be stale against, so this "
                "signal has nothing to check. It is reported UNKNOWN rather "
                "than OK because nothing was verified.",
            ),
        )

    current = _int(facts.get("current_grad_step"))
    if current is None:
        return Signal(
            "eval",
            title,
            UNKNOWN,
            "the current grad step is unknown, so staleness cannot be measured",
            ("See GRAD STEP for why.",),
        )

    evals = list(log.get("evals") or [])
    skips = list(log.get("skips") or [])
    selfplay = list(log.get("selfplay") or [])
    if not log.get("read"):
        return Signal(
            "eval",
            title,
            UNKNOWN,
            "the driver log could not be read, so no eval cycle can be found",
            (f"reason: {log.get('error') or 'unreadable'}",),
        )

    last_eval = max(evals, key=lambda item: item["step"]) if evals else None
    # No completed cycle yet is measured from 0, which is where the learner
    # started: "how far past due" needs no special case for the first cycle.
    last_step = int(last_eval["step"]) if last_eval else 0
    if last_step > current:
        # Contradictory evidence, and clamping it to zero would render a clean OK
        # with exit 0 — which is what a --log pointed at the WRONG run looks
        # like. Two files that disagree about which run they describe is not a
        # healthy night, it is an unanswerable question.
        return Signal(
            "eval",
            title,
            UNKNOWN,
            f"the log reports an eval at grad_step {last_step:,}, ahead of the "
            f"metrics' current {current:,}",
            (
                f"log:     {(document.get('paths') or {}).get('log', '?')}",
                f"metrics: {(document.get('paths') or {}).get('metrics', '?')}",
                "These two files do not describe the same run. Check --run-name, "
                "--log and --metrics before reading anything else on this screen.",
            ),
        )
    if log.get("truncated") and not evals:
        # The 8 MB tail cut means "none has EVER completed" is a claim about the
        # tail, not about the run. LIVENESS already refuses to call a missing
        # [multi done] a crash under the same circumstance.
        return Signal(
            "eval",
            title,
            UNKNOWN,
            "no completed eval in the part of the log that was read, and the log "
            "was TRUNCATED",
            (
                f"only the last {LOG_TAIL_BYTES // (1024 * 1024)} MiB of "
                f"{(document.get('paths') or {}).get('log', '?')} was read; an "
                "earlier cycle would have been missed.",
                "Nothing here says a cycle has not completed — only that this "
                "read cannot see one.",
            ),
        )
    gap = max(0, current - last_step)
    skips_after = [skip for skip in skips if int(skip["step"]) > last_step]

    detail: List[str] = []
    if last_eval is None:
        detail.append(
            f"no eval cycle has COMPLETED yet; the first was due at grad_step "
            f"{cadence:,}."
        )
    else:
        detail.append(
            f"last completed eval at grad_step {last_step:,}: "
            f"win_rate={last_eval['win_rate']:.3f} "
            f"mean_len={last_eval['mean_len']:.1f} "
            f"opponent={last_eval['opponent']}."
        )
    detail.append(f"cadence {cadence:,} grad steps; gap now {gap:,}.")

    # The rated series and its denominator. The metrics row carries a timestamp,
    # which the driver's stderr does not, so it is preferred when present.
    rated_matches: Optional[int] = None
    elo_rated: Optional[float] = None
    elo_at: Optional[float] = None
    for row in document.get("metrics", {}).get("rows") or []:
        if ELO_RATED_METRIC in row:
            elo_rated = _num(row.get(ELO_RATED_METRIC))
            elo_at = _num(row.get("wall_time"))
            candidate = _int(row.get(RATED_MATCHES_METRIC))
            if candidate is not None:
                rated_matches = candidate
    if elo_rated is None and selfplay:
        newest = selfplay[-1]
        elo_rated = float(newest["elo_rated"])
        rated_matches = int(newest["rated_matches"])
        detail.append(
            "the rated Elo below comes from the driver log, which carries no "
            "timestamp: no metrics row holds it yet."
        )
    if elo_rated is None:
        detail.append(
            f"{ELO_RATED_METRIC}: no value anywhere yet (no eval cycle has "
            "logged one)."
        )
    else:
        age = (
            format_duration(float(facts["now"]) - elo_at) + " ago"
            if elo_at is not None
            else "timestamp unavailable"
        )
        detail.append(
            f"{ELO_RATED_METRIC} = {elo_rated:.1f} at {format_clock(elo_at)} "
            f"({age}), over "
            + (
                f"{rated_matches} rated match(es)."
                if rated_matches is not None
                else "an unknown number of rated matches."
            )
        )

    if gap > EVAL_ALARM_CADENCES * cadence:
        verdict = ALARM
    elif gap > EVAL_WARN_CADENCES * cadence:
        verdict = ALARM if skips_after else WARN
    elif skips_after:
        verdict = WARN
    else:
        verdict = OK

    if skips_after:
        detail.append(
            f"{len(skips_after)} eval cycle(s) SKIPPED since the last completed "
            "one: "
            + ", ".join(
                f"grad_step {skip['step']:,} raised {skip['error']}"
                for skip in skips_after[-3:]
            )
            + ". A skipped cycle selects no checkpoint and rates no match."
        )
    if rated_matches == 0 and current > EVAL_WARN_CADENCES * cadence:
        verdict = _worse(verdict, WARN)
        detail.append(
            f"{RATED_MATCHES_METRIC} is still 0 this far in: "
            f"{ELO_RATED_METRIC} is EMPTY, not flat. Nothing has been rated."
        )

    behind = (
        f"{gap:,} grad steps since one last completed"
        if last_eval is not None
        else f"none has EVER completed ({gap:,} grad steps in)"
    )
    if verdict == ALARM:
        headline = f"eval is {gap / cadence:.1f} cadences stale: {behind}"
    elif verdict == WARN:
        headline = f"eval is behind: {behind}"
    elif last_eval is None:
        headline = f"no eval has completed yet; the first is due at grad_step {cadence:,}"
    else:
        headline = f"last completed eval {gap:,} grad steps ago"
    last_line = log.get("last_eval")
    if last_line is not None:
        detail.append(
            f"the driver's teardown 'last eval:' line reads win_rate="
            f"{last_line['win_rate']:.3f} mean_len={last_line['mean_len']:.1f} "
            f"aim_invisible={last_line['aim_invisible']:.3f} (printed only after "
            "the run ends)."
        )
    return Signal("eval", title, verdict, headline, tuple(detail))


def evaluate_watch(document: Mapping[str, Any]) -> Verdict:
    """Judge one observation of a live run. Pure: reads nothing, writes nothing."""
    facts = derive_facts(document)
    signals = (
        _signal_liveness(document),
        _signal_grad_step(document, facts),
        _signal_fleet(document, facts),
        _signal_throughput(document, facts),
        _signal_eval(document, facts),
    )
    by_key = {signal.key: signal for signal in signals}

    notes: List[str] = []
    # The cross-signal reading this whole script exists for. Neither signal can
    # say it alone: a full fleet says the bridges are up, a collapsed rate says
    # the episodes are not arriving, and together they say the collectors inside
    # the driver are dead while everything outside it looks fine.
    if by_key["fleet"].verdict == OK and by_key["throughput"].verdict == ALARM:
        notes.append(
            "FULL FLEET + COLLAPSED RATE: every bridge is listening with a "
            "client attached, yet the rate has collapsed. That is what a "
            "collector thread dying INSIDE the driver looks like from out here "
            "(distributed/actor.py: a non-BridgeError out of the collect loop "
            "kills the thread and ActorPool.aborted() stays False). The bridges "
            "cannot show it; only the rate can."
        )
    if by_key["fleet"].verdict == ALARM and by_key["throughput"].verdict in (
        WARN,
        ALARM,
    ):
        notes.append(
            "The rate shortfall is measured PER LIVE ARENA, so it is not "
            "explained away by the missing bridges. Both are real."
        )
    return Verdict(signals, facts, tuple(notes))


# ---------------------------------------------------------------------------
# Rendering.
# ---------------------------------------------------------------------------

_BADGE = {OK: "OK   ", WARN: "WARN ", ALARM: "ALARM", UNKNOWN: "?????"}


def format_report(verdict: Verdict, document: Mapping[str, Any]) -> str:
    """The one screen. Verdict first on every line, number second, prose third."""
    paths = document.get("paths") or {}
    lines: List[str] = []
    lines.append(
        f"[watch] {document.get('run_name')}  {format_clock(document.get('now'))}"
    )
    lines.append(f"        log {paths.get('log', '?')}")
    lines.append("")
    for index, signal in enumerate(verdict.signals, start=1):
        lines.append(
            f"  [{index}] {_BADGE[signal.verdict]}  "
            f"{signal.title:<11} {signal.headline}"
        )
        for item in signal.detail:
            lines.append(f"            {item}")
        lines.append("")
    for note in verdict.notes:
        lines.append(f"  >> {note}")
    if verdict.notes:
        lines.append("")
    lines.append(
        f"VERDICT: {verdict.worst}   (exit {verdict.exit_code}; "
        "0=OK/WARN 1=ALARM 3=UNDETERMINED)"
    )
    lines.append(
        "This check is READ-ONLY: it wrote nothing, signalled nothing and opened "
        "no socket."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Collection. The only impure code in this module.
# ---------------------------------------------------------------------------


def _env_float(environ: Mapping[str, str], name: str) -> Optional[float]:
    """A float from the environment, or ``None``. The shell validates these, so
    a bad value here means someone invoked the module directly."""
    try:
        return _num(float(environ.get(name, "") or "nan"))
    except ValueError:
        return None


def _read_text(path: str, tail_bytes: Optional[int] = None) -> Dict[str, Any]:
    """Read a file that may be growing underneath us. Never raises."""
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as handle:
            truncated = False
            if tail_bytes is not None and size > tail_bytes:
                handle.seek(size - tail_bytes)
                truncated = True
            data = handle.read()
    except OSError as exc:
        return {"read": False, "error": f"{type(exc).__name__}: {exc}", "text": ""}
    return {
        "read": True,
        "error": None,
        "text": data.decode("utf-8", errors="replace"),
        "truncated": truncated,
    }


def collect_document(environ: Mapping[str, str]) -> Dict[str, Any]:
    """Assemble the observation the judge runs on. Reads files; decides nothing."""

    def path_of(name: str) -> str:
        return environ.get(name, "")

    log_path = path_of("WATCH_LOG_PATH")
    metrics_path = path_of("WATCH_METRICS_PATH")
    argv_path = path_of("WATCH_ARGV_PATH")
    measurements_path = path_of("WATCH_MEASUREMENTS_PATH")

    log_read = _read_text(log_path, tail_bytes=LOG_TAIL_BYTES)
    log: Dict[str, Any] = {
        "read": log_read["read"],
        "error": log_read["error"],
        "truncated": log_read.get("truncated", False),
        "max_grad_step": None,
        "grad_step_lines": 0,
        "evals": [],
        "skips": [],
        "selfplay": [],
        "done": None,
        "last_eval": None,
    }
    if log_read["read"]:
        log.update(parse_run_log(log_read["text"]))

    metrics_read = _read_text(metrics_path)
    metrics: Dict[str, Any] = {
        "read": metrics_read["read"],
        "error": metrics_read["error"],
        "rows": [],
        "torn_lines": 0,
    }
    if metrics_read["read"]:
        metrics.update(parse_metrics_jsonl(metrics_read["text"]))

    argv_read = _read_text(argv_path)
    launch_argv: Dict[str, Any] = {
        "read": argv_read["read"],
        "error": argv_read["error"],
        "flags": {},
    }
    if argv_read["read"]:
        launch_argv["flags"] = parse_launch_argv(argv_read["text"])

    measurements_read = _read_text(measurements_path)
    measurements: Dict[str, Any] = {
        "read": measurements_read["read"],
        "error": measurements_read["error"],
        "document": {},
    }
    if measurements_read["read"]:
        try:
            parsed = json.loads(measurements_read["text"])
        except ValueError as exc:
            measurements["read"] = False
            measurements["error"] = f"not JSON: {exc}"
        else:
            if isinstance(parsed, dict):
                measurements["document"] = parsed
            else:
                measurements["read"] = False
                measurements["error"] = "not a JSON object"

    eval_every = _int(environ.get("WATCH_EVAL_EVERY", ""))
    if eval_every is None:
        eval_every = _int(launch_argv["flags"].get("--eval-every-grad-steps"))

    scan_low = _int(environ.get("WATCH_SCAN_LOW", "")) or 0
    scan_high = _int(environ.get("WATCH_SCAN_HIGH", "")) or 0
    lsof_available = environ.get("WATCH_LSOF_AVAILABLE", "") == "true"

    return {
        "now": _num(_int(environ.get("WATCH_NOW", ""))) or time.time(),
        "run_name": environ.get("WATCH_RUN_NAME", ""),
        "paths": {
            "log": log_path,
            "pid": path_of("WATCH_PID_PATH"),
            "metrics": metrics_path,
            "argv": argv_path,
            "measurements": measurements_path,
        },
        "thresholds": {
            # None means DERIVE. The shell passes these empty unless the
            # operator named one, because the right window is a property of the
            # run's own cadence, not a constant this file can hold.
            "stall_minutes": _env_float(environ, "WATCH_STALL_MINUTES"),
            "window_minutes": _env_float(environ, "WATCH_WINDOW_MINUTES"),
        },
        "eval_every_grad_steps": eval_every,
        "process": {
            "pid_file_exists": environ.get("WATCH_PID_FILE_EXISTS", "") == "true",
            "pid_raw": environ.get("WATCH_PID_RAW", ""),
            "command": environ.get("WATCH_PS_COMMAND", ""),
            "etime": environ.get("WATCH_PS_ETIME", ""),
        },
        "log": log,
        "metrics": metrics,
        "launch_argv": launch_argv,
        "measurements": measurements,
        "fleet": {
            "lsof_available": lsof_available,
            "base_port": _int(environ.get("WATCH_BASE_PORT", "")),
            "base_port_source": environ.get("WATCH_PORT_SOURCE", ""),
            "expected_arenas": _int(environ.get("WATCH_ARENAS", "")),
            "expected_arenas_source": environ.get("WATCH_ARENAS_SOURCE", ""),
            "scan_low": scan_low,
            "scan_high": scan_high,
            "listening_ports": parse_lsof_ports(
                environ.get("WATCH_LSOF_LISTEN", ""), scan_low, scan_high
            ),
            "attached_ports": parse_lsof_ports(
                environ.get("WATCH_LSOF_ESTABLISHED", ""), scan_low, scan_high
            ),
        },
    }


def main() -> int:
    document = collect_document(os.environ)
    verdict = evaluate_watch(document)
    print(format_report(verdict, document))
    return verdict.exit_code


if __name__ == "__main__":
    sys.exit(main())
WATCH_VERDICT_PY
WATCH_EXIT=$?
set -e
exit "${WATCH_EXIT}"
