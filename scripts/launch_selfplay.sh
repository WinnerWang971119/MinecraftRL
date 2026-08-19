#!/usr/bin/env bash
# launch_selfplay.sh — T19: the 25-pad smoke, the epsilon sizing, and the command
# that commits the night.
#
# T17's canary (scripts/canary_selfplay.sh) answers "is the self-play WIRING
# alive?" at 4 pads. This file answers the two questions left:
#
#   1. Does the fleet HOLD at full width — 25 pads, a second frozen DRQN per
#      collector, and a snapshot read at the head of every episode? Nothing has
#      measured that. The 121.95 transitions/s ceiling was measured at 25 pads
#      with ONE net per collector and no per-episode weight load.
#   2. What is `--eps-decay-episodes` for the ARMORED regime? This has been
#        wrong twice already, in both directions:
#          * `ASSUMED_MEAN_EPISODE_STEPS = 30` produced a decay window spanning
#            142% of a 12 h run, so epsilon never finished decaying and sat near
#            0.25 all night;
#          * the 285-step replacement is a BARE-handed, scripted-opponent,
#            400-cap measurement, and the M3 retry then measured ~95 steps in a
#            regime that number was supposed to describe.
#      Neither describes an armored fight. T17's canary measures that for the
#      first time and writes it to
#      `runs/<run>/canary/canary_measurements.json`; this file reads it and
#      REFUSES to size from any constant.
#
# WHAT THIS SCRIPT STARTS. `smoke` and `launch` start the PYTHON DRIVER only.
# The boot order for anything live is Paper -> bridges -> driver, and the
# operator owns the first two steps:
#
#     DUMMY_KNOCKBACK_IMMUNE=false server/setup/start-pads.sh --pads 25
#
# That env var is not decoration and it cannot be set here: the bridges read it
# at their own startup, and `agent/train.py` derives its own launcher setting
# from `cfg.opponent == "dummy"` (so a self-play run's RELAUNCHED bridges are
# already non-immune — only the ones booted before the run are at risk). macOS
# exposes no other process's environment, and `start-pads.sh` passes the setting
# via the env var rather than argv, so the flag cannot be read back. The only
# proof is empirical, and it is the canary's OPPONENT_FROZEN check. This script
# therefore requires a GREEN canary AND refuses if any bridge in the fleet
# started AFTER that canary ran — a rebooted pad is an unproven pad.
#
# NEVER CONNECT TWICE TO A BRIDGE PORT. `BridgeServer` accepts exactly ONE TCP
# client and `_onConnection` resolves a second one by DESTROYING the incumbent
# (bridge/transport.js). Every bridge-port check below is `lsof` listener
# inspection, which opens no connection. Only the Minecraft port is ever
# connected to, because Paper is multi-client.
#
# WHERE THE JUDGING LIVES. Every threshold, every projection and every refusal
# is in the embedded `launch_sizing` module below, which is pure and
# stdlib-only. `tests/test_launch_selfplay.py` extracts it verbatim from between
# the LAUNCH_SIZING_PY sentinels and drives every refusal over synthetic
# measurement files, so what is tested is byte-identical to what the operator
# runs. The shell does I/O and process control; it decides nothing.
#
# SUBCOMMANDS
#   smoke     bounded 25-pad run at production settings; asserts transitions/s,
#             grad steps/hour, queue backlog, RSS and snapshot-load latency.
#   plan      read the canary (and smoke) measurements, derive the sizing, print
#             the arithmetic and the exact launch command. Starts nothing,
#             connects to nothing.
#   launch    everything `plan` does, plus the fleet preflight, then starts the
#             driver DETACHED under nohup and prints the PID and log path.
#   compare   the morning checkpoint table. Offline; reads run directories.
#
# EXIT CODES: 0 = cleared / done; 1 = REFUSED (something blocks); 2 = usage or
# preflight error (nothing was run).
#
# Owner: T19.

set -euo pipefail

# --- Defaults ---------------------------------------------------------------
RUN_NAME="m4_selfplay"
SMOKE_RUN_NAME="m4_selfplay_smoke"
CANARY_RUN_NAME="m4_selfplay_canary"
ARENAS=25
WINDOW_HOURS=12
WARM_START=""
HOST="127.0.0.1"
BRIDGE_BASE_PORT=5555
MC_PORT=25565
PYTHON_BIN=""
CANARY_DIR=""
OUT_DIR=""
SEED=0

# -- smoke sizing -----------------------------------------------------------
# The smoke is a DRESS REHEARSAL, so it runs production `min_replay`: the whole
# point is to measure the fleet under the settings the night will use. (The
# canary is the opposite and lowers it to 2000 — it is testing wiring, and a
# 21-minute warm-up would eat its budget.)
SMOKE_MIN_REPLAY=25000          # agent/train_config.py TrainConfig.min_replay
SMOKE_GRAD_STEPS=2500           # ~33 min at the M3 retry's ~4570 grad steps/hour
SMOKE_MINUTES=45                # wall-clock deadline; the driver is SIGINTed at it
SMOKE_SNAPSHOT_EVERY=500        # -> snapshots at 500/1000/1500/2000/2500
SMOKE_PROMOTE_FIRST=800         # pinned reference #2
SMOKE_PROMOTE_SECOND=1600       # pinned reference #3
SMOKE_CHECKPOINT_EVERY=500
# Pin epsilon at the run's TERMINAL value for the whole smoke. The smoke
# measures CAPACITY, not learning, and `eps_decay_episodes=1` collapses the
# whole decay into GLOBAL episode 0: that one episode runs at
# `effective_eps_start`=0.25 (the warm start's, since the smoke passes
# --warm-start) and every episode after it at `eps_end`=0.05 — one episode in
# the ~2,000 a smoke of this length collects. That is the regime the canary's
# probe measured the armored episode length in (learner 0.05 / opponent 0.02)
# and the one the long run spends >85% of its episodes in. Measuring
# throughput at 0.25 throughout would describe a fight nobody is going to
# have all night.
SMOKE_EPS_DECAY_EPISODES=1
SMOKE_RSS_SAMPLE_SECONDS=15
SMOKE_SNAPSHOT_LOAD_REPEATS=5

# -- launch overrides (empty == DERIVE from the measurements) ---------------
OV_EPS_DECAY=""
OV_EVAL_EPISODES=""
OV_REFERENCE_EVAL_EPISODES=""
OV_EVAL_EVERY=""
OV_MAX_EPISODES=""
OV_MAX_GRAD_STEPS=""
OV_CHECKPOINT_EVERY=""
OV_SNAPSHOT_EVERY=""

# -- compare ----------------------------------------------------------------
EXTRA_RUNS_DIRS=()

# --- Resolve paths ----------------------------------------------------------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)}"

log()  { echo "[launch] $*"; }
warn() { echo "[launch] WARNING: $*" >&2; }
die()  { echo "[launch] FATAL: $*" >&2; exit 2; }

usage() {
    cat <<USAGE
usage: scripts/launch_selfplay.sh <command> [options]

commands:
  smoke     bounded ${ARENAS}-pad run at production settings; refuses on
            transitions/s, grad steps/hour, queue backlog, RSS or snapshot-load
            latency. Requires the fleet to be UP and the canary to be GREEN.
  plan      derive the sizing from the canary/smoke measurements and print the
            arithmetic plus the exact launch command. Starts nothing.
  launch    plan, preflight, then start the driver detached (nohup) and print
            the PID and log path.
  compare   the morning checkpoint table (offline).

common options:
  --warm-start PATH        ABSOLUTE path to the frozen M3 checkpoint. REQUIRED
                           for smoke/plan/launch (TrainConfig refuses
                           --opponent selfplay without one, AC14).
  --run-name NAME          long-run name (default: ${RUN_NAME}). NOT "m4": the
                           completed bare-handed run owns runs/m4.*, which is
                           also this run's warm-start source.
  --window-hours H         the night's intended length (default: ${WINDOW_HOURS}).
                           This is the ONE input no measurement can supply.
  --arenas N               pads (default: ${ARENAS}).
  --host H / --port P      bridge host / BASE port (default: ${HOST} / ${BRIDGE_BASE_PORT}).
  --mc-port P              shared Minecraft port (default: ${MC_PORT}).
  --canary-dir DIR         where T17 wrote canary_measurements.json
                           (default: <repo>/runs/${CANARY_RUN_NAME}/canary).
  --out-dir DIR            evidence directory (default: <repo>/runs/<run>/launch).
  --python PATH            interpreter (default: \$PYTHON, else <repo>/.venv/bin/python).
  --seed N                 base RNG seed (default: ${SEED}).

smoke options:
  --smoke-run-name NAME    (default: ${SMOKE_RUN_NAME}; must differ from --run-name)
  --smoke-grad-steps N     (default: ${SMOKE_GRAD_STEPS})
  --smoke-minutes M        wall-clock deadline (default: ${SMOKE_MINUTES})
  --smoke-min-replay N     (default: ${SMOKE_MIN_REPLAY}, the PRODUCTION value)
  --analyze-only DIR       re-judge an existing smoke evidence directory and
                           connect to nothing.

launch sizing overrides (each defaults to DERIVED; passing one is still checked):
  --eps-decay-episodes N   --eval-episodes N        --reference-eval-episodes N
  --eval-every-grad-steps N                         --max-episodes N
  --max-grad-steps N       --checkpoint-every-grad-steps N
  --snapshot-every-grad-steps N

compare options:
  --extra-runs DIR         another runs/ directory to search for candidates
                           (repeatable). The bare-handed M3 retry's checkpoints
                           live in the MAIN checkout, not this worktree.

Boot order, which this script does not do for you:
  DUMMY_KNOCKBACK_IMMUNE=false server/setup/start-pads.sh --pads ${ARENAS}
USAGE
}

need_value() {
    if [[ "$2" -lt 2 ]]; then
        echo "[launch] $1 requires a value." >&2
        exit 2
    fi
}

# --- Parse ------------------------------------------------------------------
COMMAND="${1:-}"
case "${COMMAND}" in
    smoke|plan|launch|compare) shift ;;
    -h|--help|help) usage; exit 0 ;;
    "") usage >&2; die "no command given." ;;
    *) usage >&2; die "unknown command: ${COMMAND}" ;;
esac

ANALYZE_ONLY=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --warm-start)     need_value "$1" $#; WARM_START="$2"; shift 2 ;;
        --run-name)       need_value "$1" $#; RUN_NAME="$2"; shift 2 ;;
        --smoke-run-name) need_value "$1" $#; SMOKE_RUN_NAME="$2"; shift 2 ;;
        --window-hours)   need_value "$1" $#; WINDOW_HOURS="$2"; shift 2 ;;
        --arenas)         need_value "$1" $#; ARENAS="$2"; shift 2 ;;
        --host)           need_value "$1" $#; HOST="$2"; shift 2 ;;
        --port)           need_value "$1" $#; BRIDGE_BASE_PORT="$2"; shift 2 ;;
        --mc-port)        need_value "$1" $#; MC_PORT="$2"; shift 2 ;;
        --canary-dir)     need_value "$1" $#; CANARY_DIR="$2"; shift 2 ;;
        --out-dir)        need_value "$1" $#; OUT_DIR="$2"; shift 2 ;;
        --python)         need_value "$1" $#; PYTHON_BIN="$2"; shift 2 ;;
        --seed)           need_value "$1" $#; SEED="$2"; shift 2 ;;
        --smoke-grad-steps) need_value "$1" $#; SMOKE_GRAD_STEPS="$2"; shift 2 ;;
        --smoke-minutes)  need_value "$1" $#; SMOKE_MINUTES="$2"; shift 2 ;;
        --smoke-min-replay) need_value "$1" $#; SMOKE_MIN_REPLAY="$2"; shift 2 ;;
        --analyze-only)   need_value "$1" $#; ANALYZE_ONLY="$2"; shift 2 ;;
        --eps-decay-episodes) need_value "$1" $#; OV_EPS_DECAY="$2"; shift 2 ;;
        --eval-episodes)  need_value "$1" $#; OV_EVAL_EPISODES="$2"; shift 2 ;;
        --reference-eval-episodes) need_value "$1" $#; OV_REFERENCE_EVAL_EPISODES="$2"; shift 2 ;;
        --eval-every-grad-steps) need_value "$1" $#; OV_EVAL_EVERY="$2"; shift 2 ;;
        --max-episodes)   need_value "$1" $#; OV_MAX_EPISODES="$2"; shift 2 ;;
        --max-grad-steps) need_value "$1" $#; OV_MAX_GRAD_STEPS="$2"; shift 2 ;;
        --checkpoint-every-grad-steps) need_value "$1" $#; OV_CHECKPOINT_EVERY="$2"; shift 2 ;;
        --snapshot-every-grad-steps) need_value "$1" $#; OV_SNAPSHOT_EVERY="$2"; shift 2 ;;
        --extra-runs)     need_value "$1" $#; EXTRA_RUNS_DIRS+=("$2"); shift 2 ;;
        -h|--help)        usage; exit 0 ;;
        *) usage >&2; die "unknown argument: $1" ;;
    esac
done

# --- Interpreter ------------------------------------------------------------
if [[ -z "${PYTHON_BIN}" ]]; then
    PYTHON_BIN="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"
fi
if [[ ! -x "${PYTHON_BIN}" ]] && ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    die "python interpreter not found: ${PYTHON_BIN}
      System python 3.9 cannot import this package. Create the venv:
        python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
      Or pass --python /path/to/python."
fi

CANARY_DIR="${CANARY_DIR:-${REPO_ROOT}/runs/${CANARY_RUN_NAME}/canary}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/runs/${RUN_NAME}/launch}"
SIZING_MODULE="${OUT_DIR}/launch_sizing.py"
SMOKE_EVIDENCE="${OUT_DIR}/smoke_evidence.json"
SMOKE_MEASUREMENTS="${OUT_DIR}/smoke_measurements.json"
SMOKE_DRIVER_LOG="${OUT_DIR}/smoke_driver.log"
SMOKE_RSS_LOG="${OUT_DIR}/smoke_rss.tsv"
SMOKE_LOAD_JSON="${OUT_DIR}/smoke_snapshot_load.json"
PLAN_INPUT="${OUT_DIR}/launch_plan_input.json"
PLAN_JSON="${OUT_DIR}/launch_plan.json"
PLAN_ARGV="${OUT_DIR}/launch_argv.txt"
COMPARE_INPUT="${OUT_DIR}/compare_input.json"
CANARY_MEASUREMENTS="${CANARY_DIR}/canary_measurements.json"
CANARY_EVIDENCE="${CANARY_DIR}/evidence.json"
CANARY_SCRIPT="${SCRIPT_DIR}/canary_selfplay.sh"
RUN_LOG="${REPO_ROOT}/runs/${RUN_NAME}.log"
RUN_PID_FILE="${REPO_ROOT}/runs/${RUN_NAME}.pid"
CHECKPOINT_PATH="${REPO_ROOT}/runs/${RUN_NAME}.pt"
BEST_CHECKPOINT_PATH="${REPO_ROOT}/runs/${RUN_NAME}.best.pt"
SMOKE_CHECKPOINT_PATH="${REPO_ROOT}/runs/${SMOKE_RUN_NAME}.pt"
SMOKE_BEST_CHECKPOINT_PATH="${REPO_ROOT}/runs/${SMOKE_RUN_NAME}.best.pt"

# ===========================================================================
# Non-connecting port inspection. Mirrors scripts/canary_selfplay.sh's helpers,
# and for the same reason: a connect probe against a bridge is a MUTATION —
# BridgeServer destroys the incumbent client.
# ===========================================================================

# listener_pids <port> — pids LISTENing on that TCP port. Opens no connection.
listener_pids() {
    if command -v lsof >/dev/null 2>&1; then
        lsof -nP -iTCP:"$1" -sTCP:LISTEN -t 2>/dev/null | tr '\n' ' '
    fi
}

# established_peers <port> — count of ESTABLISHED sockets on that port.
established_peers() {
    if command -v lsof >/dev/null 2>&1; then
        lsof -nP -iTCP:"$1" -sTCP:ESTABLISHED 2>/dev/null | grep -c . || true
    else
        echo 0
    fi
}

# proc_etime <pid> — BSD elapsed time ([[dd-]hh:]mm:ss). macOS ps has no
# `etimes` keyword (verified: "ps: etimes: keyword not found"), so the string is
# parsed by launch_sizing.parse_etime rather than read as a number.
proc_etime() {
    ps -o etime= -p "$1" 2>/dev/null | tr -d ' '
}

# proc_rss_kb <pid> — resident set size in KiB, or empty when the pid is gone.
proc_rss_kb() {
    ps -o rss= -p "$1" 2>/dev/null | tr -d ' '
}

# mc_connect_probe <host> <port> — the ONLY connect in this file. Safe because
# Paper is multi-client; a bridge is not.
mc_connect_probe() {
    "${PYTHON_BIN}" - "$1" "$2" <<'LAUNCH_MC_PROBE_PY'
import socket
import sys

host, port = sys.argv[1], int(sys.argv[2])
try:
    with socket.create_connection((host, port), timeout=5.0):
        pass
except OSError:
    raise SystemExit(1)
raise SystemExit(0)
LAUNCH_MC_PROBE_PY
}

# physical_memory_bytes — hw.memsize, or empty when sysctl is unavailable. The
# RSS projection compares against this rather than an absolute ceiling, so the
# gate says something true on a machine other than this one.
physical_memory_bytes() {
    sysctl -n hw.memsize 2>/dev/null || true
}

# ===========================================================================
# The sizing + verdict module. Written to disk, then run by the phases below.
#
# It is a SEPARATE FILE, not an inline heredoc, for one reason:
# tests/test_launch_selfplay.py extracts the text between the two
# LAUNCH_SIZING_PY sentinels verbatim and drives every refusal over synthetic
# measurement documents. Editing a threshold here changes what the tests assert.
# Keep it stdlib-only — it must be importable with nothing but an interpreter,
# and none of its decisions may depend on torch being installed.
# ===========================================================================
emit_sizing_module() {
    cat >"$1" <<'LAUNCH_SIZING_PY'
"""launch_sizing — T19's arithmetic and every refusal that guards the night.

This module decides three things and computes one number:

  * **the number** — ``--eps-decay-episodes`` for the ARMORED self-play regime,
    derived from T17's canary MEASUREMENT and never from a constant;
  * may the 25-pad smoke's numbers be called healthy (:func:`evaluate_smoke`);
  * may the 24-hour run start, and with exactly which argv
    (:func:`evaluate_launch`, :func:`build_launch_argv`);
  * which checkpoints are even candidates in the morning, and what each number
    is NOT evidence of (:func:`compare_candidates`).

It is pure: it reads plain dicts and returns verdicts. No sockets, no torch, no
subprocesses, and no filesystem beyond reading the input document in
:func:`main`. That is what makes every refusal below testable offline, and
testing them is the point — a gate whose refusals do not fire is worse than no
gate, because it manufactures confidence.

WHY EACH REFUSAL EXISTS — the failure, not the symptom.

Sizing (:func:`evaluate_launch`):

* ``CANARY_MEASUREMENTS_MISSING`` — there is no measured armored episode length,
  so the only remaining inputs are the two stale constants
  (:data:`STALE_EPISODE_STEPS_SCRIPTED_BARE`,
  :data:`STALE_EPISODE_STEPS_M3_RETRY`), and sizing from either is the mistake
  this whole file exists to prevent. Fail closed: absence is a refusal.
* ``CANARY_MEASUREMENTS_STALE`` — the measurement is older than
  ``max_measurement_age_hours``, or it is dated in the future (an unusable
  clock), or a bridge in the fleet started AFTER it was written. The last case
  is the load-bearing one: the canary's ``OPPONENT_FROZEN`` check is the only
  empirical proof that ``DUMMY_KNOCKBACK_IMMUNE=false`` reached the bridges, and
  that proof covers only the processes it probed.
* ``CANARY_NOT_GREEN`` — T17's gate refused (or was never re-run over its own
  evidence). Launching over a red canary is exactly the mistake the canary was
  built to make impossible.
* ``EPISODE_LENGTH_UNMEASURED`` / ``EPISODE_LENGTH_IMPLAUSIBLE`` — the figure is
  absent, non-finite, non-positive, or so close to
  :data:`MAX_EPISODE_STEPS` that the "episodes" being measured are cap-hit
  draws. Sizing off a draw length would stretch the decay window over a run that
  never happens.
* ``EPISODE_LENGTH_DISAGREEMENT`` — the canary's probe length and the length
  implied by its own collection rate differ by more than
  ``max_episode_length_ratio``. Two measurements of one stream that disagree
  mean one of them is wrong, and this module cannot tell which.
* ``EPS_DECAY_ABSURD`` — the resolved window is not a sane FRACTION of the
  projected run. A previous run shipped a window spanning 142% of the night, so
  epsilon never finished decaying and sat near 0.25 until morning; the opposite
  error floors epsilon a few percent in. Neither direction is safe, so both ends
  of the band refuse.
* ``SMOKE_NOT_CLEARED`` — no smoke measurement, or one that refused. The canary
  ran at 4 pads with a lowered replay floor; it says nothing about 25 pads under
  per-episode snapshot loads.
* ``FLEET_NOT_READY`` — Paper unreachable, a missing bridge listener, or a port
  that already has an ESTABLISHED client. The last one is not a warning: a
  second connection DESTROYS the incumbent.
* ``WARM_START_UNUSABLE`` — missing, relative, or unhashable. ``TrainConfig``
  refuses ``--opponent selfplay`` without a warm start (AC14), and the pool's
  snapshot 0 — a PINNED reference, never dropped, never evicted — is seeded
  entirely from this file.
* ``RUN_NAME_COLLISION`` — the run would overwrite an existing run's outputs, or
  its own warm start. ``runs/m4.*`` belongs to the completed bare-handed run AND
  is this run's warm-start source: reusing the name overwrites the warm start
  with this run's own output, mid-run.
* ``CHECKPOINT_UNSAFE`` — no ``--checkpoint``, or ``--best-checkpoint`` alone, or
  both pointing at one path. ``--best-checkpoint`` alone disables the periodic
  save AND makes the final save a no-op (``agent/train.py`` warns about exactly
  this at startup), so a run that never wins an eval leaves an empty ``runs/``.
* ``EVAL_CYCLE_TOO_LONG`` / ``EVAL_CADENCE_TOO_TIGHT`` — an eval cycle costs
  ``episodes x episode_seconds`` on ONE borrowed arena, serially. 100 episodes at
  the bare-handed 285-step length cost 97 minutes, measured. Too long a cycle
  eats the night; too tight a cadence leaves the designated arena in eval
  instead of training.
* ``BUDGET_ENDS_EARLY`` — ``--max-episodes`` defaults to
  :data:`DEFAULT_MAX_EPISODES`, which at 25 armored pads ends the run in a
  couple of hours. The M3 run hit this and had to raise it deliberately.
* ``OPPONENT_NOT_SELFPLAY`` — a self-play launch that is not fighting past
  selves is a differently-named scripted run.

Smoke (:func:`evaluate_smoke`): ``SMOKE_DRIVER_FAILED``,
``SMOKE_NOT_FULL_WIDTH``, ``SMOKE_TOO_SHORT``, ``SMOKE_ZERO_GRAD_STEPS``,
``SMOKE_GRAD_STEPS_LOW``, ``SMOKE_TRANSITIONS_LOW``, ``SMOKE_QUEUE_BACKLOG``,
``SMOKE_RSS_PROJECTION``, ``SMOKE_SNAPSHOT_LOAD_SLOW``,
``SMOKE_POOL_NOT_GROWING`` — each documented on its own check function.

Owner: T19. Extracted verbatim from ``scripts/launch_selfplay.sh`` by
``tests/test_launch_selfplay.py``.
"""

from __future__ import annotations

import json
import math
import os
import shlex
import sys
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple

__all__ = [
    "CheckResult",
    "DEFAULT_MAX_EPISODES",
    "DEFAULT_THRESHOLDS",
    "EPS_DECAY_FRACTION_OF_RUN",
    "LEARNER_DRAIN_BATCH",
    "M3_RETRY_GRAD_STEPS_PER_HOUR",
    "MAX_EPISODE_STEPS",
    "MEASURED_AGGREGATE_TRANSITIONS_PER_S",
    "MEASURED_PER_ARENA_TRANSITIONS_PER_S",
    "PLAN_VERSION",
    "PRODUCTION_PADS",
    "SMOKE_EVIDENCE_VERSION",
    "STALE_EPISODE_STEPS_M3_RETRY",
    "STALE_EPISODE_STEPS_SCRIPTED_BARE",
    "Verdict",
    "build_launch_argv",
    "cautions_for",
    "compare_candidates",
    "derive_sizing",
    "evaluate_launch",
    "evaluate_smoke",
    "format_launch_command",
    "format_launch_report",
    "format_smoke_report",
    "parse_etime",
    "sizing_arithmetic_lines",
]

#: Bumped when the plan/smoke document shapes change incompatibly. The collector
#: stamps it and the evaluators refuse a document they do not understand, rather
#: than reading absent fields as healthy zeros.
PLAN_VERSION = 1
SMOKE_EVIDENCE_VERSION = 1

#: Episode horizon in decision steps — ``agent.contract_config.MAX_EPISODE_STEPS``
#: after T4 raised it from 400. An episode of exactly this length is a cap-hit
#: draw, not a fight. Pinned against the real constant by the test suite.
MAX_EPISODE_STEPS = 600

#: The pad count the long run uses, and the measured scaling ceiling: the arena
#: sweep found the per-arena rate flat from 16 to 25 pads with no knee, and
#: ``max-players`` binds before the machine does above it.
PRODUCTION_PADS = 25

#: MEASURED (2026-08-16, 600 s confirm at N=25) — per-arena collection rate in
#: transitions/second. ``agent.train_config.MEASURED_PER_ARENA_TRANSITIONS_PER_S``.
#: Pinned against that constant by the test suite.
MEASURED_PER_ARENA_TRANSITIONS_PER_S = 4.8782

#: 25 x 4.8782 == 121.955 transitions/s aggregate, the number the sweep reported.
MEASURED_AGGREGATE_TRANSITIONS_PER_S = (
    PRODUCTION_PADS * MEASURED_PER_ARENA_TRANSITIONS_PER_S
)

#: MEASURED (M3 retry, finished 2026-08-19 10:18) — 30,000 gradient steps in
#: 6h34m (23,640 s) == 1.269/s == ~4,568/hour, at 25 pads against the scripted
#: opponent. The yardstick the smoke's learner rate is judged against. It is a
#: BARE-handed figure; the armored regime's is what the smoke measures.
M3_RETRY_GRAD_STEPS_PER_HOUR = 4570.0

#: ``agent.train_config.EPS_DECAY_FRACTION_OF_RUN`` — the share of a run's
#: projected episodes the epsilon decay spans. ~15% is the DQN-lineage guidance:
#: rich exploration early, mostly exploitative for the bulk of training.
EPS_DECAY_FRACTION_OF_RUN = 0.15

#: ``distributed.learner._DEFAULT_DRAIN_BATCH`` — the most episodes the learner
#: pulls off the transport in ONE drain pass, and it takes exactly one gradient
#: step per pass. So ``episodes_received / grad_steps`` approaching this number
#: means every pass is hitting the cap, i.e. the backlog is not draining. That
#: ratio is the only queue-depth signal observable from OUTSIDE the driver
#: process: the multi-arena path logs no replay size and no queue length, and
#: adding one would mean editing ``agent/train.py``.
LEARNER_DRAIN_BATCH = 16

#: ``agent/train.py``'s ``--max-episodes`` argparse default. At 25 armored pads
#: this is a couple of hours, not a night. The M3 run raised it deliberately and
#: recorded why ("the default 10000 ends the run at ~6.5 hours").
DEFAULT_MAX_EPISODES = 10_000

#: ``agent.train.DEFAULT_REFERENCE_EVAL_EPISODES`` — episodes against EACH pinned
#: reference in a self-play eval cycle.
DEFAULT_REFERENCE_EVAL_EPISODES = 10

#: The most pinned references a cycle ever fights: snapshot 0 at seed, plus the
#: two promotions in ``TrainConfig.reference_promote_grad_steps``. Cycle cost is
#: sized against THIS, not against the 1 reference the first hour has, because
#: the 3-reference cycle is what runs for most of the night.
MAX_PINNED_REFERENCES = 3

#: The two episode lengths this module must never fall back to, kept by name so
#: a refusal can say what the operator is probably about to reach for.
#: 285.0 is ``agent.train_config.MEASURED_MEAN_EPISODE_STEPS``: bare-handed, an
#: UNARMED opponent, scripted HARD tier, against a 400-step cap.
STALE_EPISODE_STEPS_SCRIPTED_BARE = 285.0
#: 95.0 is the M3 retry's: 30,503 episodes at 122 transitions/s. Also
#: bare-handed, also against an unarmed opponent.
STALE_EPISODE_STEPS_M3_RETRY = 95.0

#: The run name the plan fixes, and the one it must never be. ``runs/m4.*``
#: belongs to the completed bare-handed run and is this run's warm-start source.
EXPECTED_RUN_NAME = "m4_selfplay"
FORBIDDEN_RUN_NAMES = frozenset({"m4", "m3", "m2_multi", "m2_train"})

#: Runs whose checkpoints were trained and scored BARE-handed — no armor on
#: either side, and the opponent holding nothing. Their win rates are not
#: comparable to an armored number.
BARE_HANDED_RUNS = frozenset({"m2_multi", "m3", "m4"})

DEFAULT_THRESHOLDS: Dict[str, float] = {
    # -- measurement freshness ------------------------------------------------
    # A canary older than this describes a fleet and a codebase that have had a
    # working day to drift. 12 h covers "canary before dinner, launch after".
    "max_measurement_age_hours": 12.0,
    # A measurement dated in the future by more than a few seconds means the
    # clock cannot be used to reason about freshness at all.
    "max_clock_skew_seconds": 60.0,
    # -- episode length -------------------------------------------------------
    # Below this an "episode" is not a fight; above the cap fraction it is a
    # cap-hit draw wearing an episode's clothes.
    "min_episode_length_steps": 5.0,
    "max_episode_length_cap_fraction": 0.90,
    # Two measurements of one stream may differ by warm-up and eval overhead;
    # beyond 2x one of them is wrong.
    "max_episode_length_ratio": 2.0,
    # -- the epsilon window ---------------------------------------------------
    # The band the decay window must land in as a fraction of the projected run.
    # 0.15 is the target; 1.42 is what shipped once.
    "min_eps_decay_fraction": 0.05,
    "max_eps_decay_fraction": 0.40,
    # -- eval sizing ----------------------------------------------------------
    "eval_cycle_target_min_minutes": 30.0,
    "eval_cycle_target_max_minutes": 45.0,
    "eval_cycle_hard_max_minutes": 60.0,
    # Share of wall-clock the DESIGNATED arena spends in eval rather than
    # collecting. The derivation targets half; the hard line is 0.60, above
    # which that arena spends MORE of the night evaluating than collecting and
    # has stopped being a collector at all.
    "max_eval_duty": 0.50,
    "eval_duty_hard_max": 0.60,
    # -- budgets --------------------------------------------------------------
    # Headroom over the projection. Erring LARGE is the cheap direction: an
    # over-budgeted run is still going in the morning and its periodic
    # checkpoint is right there, while an under-budgeted one ends at 2am.
    "budget_margin": 1.25,
    # -- the smoke ------------------------------------------------------------
    # A smoke at fewer pads proves nothing about the width the night runs at.
    "smoke_min_pads": float(PRODUCTION_PADS),
    # Long enough to clear the 25,000-transition warm-up (~205 s at 122/s) and
    # still leave a measurable steady state.
    "min_smoke_wall_seconds": 600.0,
    "min_smoke_grad_steps": 200.0,
    # Fractions of the measured references. The second DRQN per collector and
    # the per-episode snapshot read are real costs; the gate allows for them and
    # refuses a collapse.
    "min_transitions_per_s_fraction": 0.75,
    "min_grad_steps_per_hour_fraction": 0.60,
    # Fraction of LEARNER_DRAIN_BATCH the episodes-per-grad-step ratio may reach
    # before the backlog is considered undrained. A healthy run sits near 1.
    "max_episodes_per_grad_step_fraction": 0.75,
    # Projected peak RSS (driver + JVM) as a share of physical memory.
    "max_rss_fraction_of_ram": 0.60,
    # Share of ONE arena's mean episode wall time that the head-of-episode
    # snapshot read may consume. It runs on the collector thread, so it is time
    # stolen directly from collection.
    "max_snapshot_load_duty": 0.05,
    # The pool must have grown past snapshot 0, or the per-episode load path the
    # smoke exists to measure never ran at width.
    "min_smoke_pool_size": 2.0,
}


class CheckResult(NamedTuple):
    """One gate check's outcome.

    Attributes:
        code: The stable refusal code (also the name of the check).
        passed: True iff this check clears the launch.
        detail: What was measured, for the "here is why it passed" line.
        why: On a refusal, WHAT is wrong — never just the code.
        check: On a refusal, what the operator should look at next.
    """

    code: str
    passed: bool
    detail: str
    why: str = ""
    check: str = ""


class Verdict(NamedTuple):
    """The gate's answer: every check, plus whatever it derived on the way."""

    checks: List[CheckResult]
    facts: Dict[str, Any]

    @property
    def refusals(self) -> List[CheckResult]:
        """Only the failing checks, in evaluation order."""
        return [c for c in self.checks if not c.passed]

    @property
    def ok(self) -> bool:
        """True iff nothing blocks."""
        return not self.refusals


# ---------------------------------------------------------------------------
# Small helpers. Everything that touches an input document goes through these,
# so a missing or wrong-typed field can never be read as a healthy zero.
# ---------------------------------------------------------------------------


def _thresholds(overrides: Optional[Mapping[str, Any]]) -> Dict[str, float]:
    merged = dict(DEFAULT_THRESHOLDS)
    if overrides:
        for key, value in overrides.items():
            if key in merged:
                coerced = _num(value)
                if coerced is not None:
                    merged[key] = coerced
    return merged


def _num(value: Any) -> Optional[float]:
    """Coerce to a FINITE float, or None. NaN and Inf are None, not values."""
    if isinstance(value, bool) or value is None:
        return None
    if not isinstance(value, (int, float)):
        return None
    out = float(value)
    return out if math.isfinite(out) else None


def _int(value: Any) -> Optional[int]:
    """Coerce to an int, or None. Bools are not ints here."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float) and math.isfinite(value):
        return int(value)
    return None


def _section(document: Mapping[str, Any], name: str) -> Dict[str, Any]:
    """Return a mapping section, or an empty dict — never a non-mapping."""
    value = document.get(name)
    return dict(value) if isinstance(value, Mapping) else {}


def parse_etime(text: Any) -> Optional[float]:
    """Seconds from a BSD ``ps -o etime=`` string, or ``None`` if unparseable.

    macOS ``ps`` has no ``etimes`` keyword (verified on this machine: "ps:
    etimes: keyword not found"), so a process's age arrives as
    ``[[dd-]hh:]mm:ss`` and must be parsed rather than read. Used to answer one
    question: did any bridge in the fleet start AFTER the canary that proved the
    fleet is not knockback-immune?
    """
    if not isinstance(text, str):
        return None
    raw = text.strip()
    if not raw:
        return None
    days = 0
    if "-" in raw:
        head, _, raw = raw.partition("-")
        if not head.isdigit():
            return None
        days = int(head)
    parts = raw.split(":")
    if len(parts) not in (2, 3):
        return None
    try:
        values = [int(part) for part in parts]
    except ValueError:
        return None
    if any(value < 0 for value in values):
        return None
    if len(parts) == 2:
        hours, (minutes, seconds) = 0, values
    else:
        hours, minutes, seconds = values
    return float(days * 86400 + hours * 3600 + minutes * 60 + seconds)


# ---------------------------------------------------------------------------
# PART 2 — sizing --eps-decay-episodes from MEASUREMENT.
#
# The chain, in the order the report prints it:
#
#   measured armored episode length  ->  episodes/hour at 25 pads
#     ->  episodes in the intended window  ->  the decay window (15% of them)
#
# Every step is arithmetic over ONE measured input and ONE measured constant.
# Nothing here reaches for STALE_EPISODE_STEPS_* — those exist only so a
# refusal can name what it is refusing to use.
# ---------------------------------------------------------------------------

#: Which of the canary's armored episode-length figures sizes a TRAINING run,
#: and why. The probe's is measured at learner eps=0.05 / opponent eps=0.02 —
#: the run's TERMINAL epsilons — and a run whose decay spans 15% of its episodes
#: fights the other 85% at (or near) exactly those values. The periodic eval's
#: figure is a DIFFERENT regime — one seat is the fixed scripted yardstick that
#: ``agent.train.build_eval_opponent`` returns even on a self-play run, not the
#: trained net — and belongs to eval sizing, which is where it is used below.
#:
#: There is no eps=0-both-sides episode length anywhere in the canary: the rated
#: reference gauntlet is the only greedy-vs-greedy regime and it reports Elo,
#: never a length. A key by that name was removed from T17's measurement.
EPISODE_LENGTH_SOURCES: Dict[str, Tuple[str, str]] = {
    "probe": (
        "armored_mean_episode_length_probe",
        "canary probe (learner eps=0.05 / opponent eps=0.02 — the run's terminal "
        "epsilons, and the demo's regime)",
    ),
    "eval_vs_scripted": (
        "armored_mean_episode_length_eval_vs_scripted",
        "canary training-run periodic eval, fought against the FIXED SCRIPTED "
        "yardstick named in eval_opponent",
    ),
}

#: Eval-cycle sizing constants. A cycle is ``--eval-episodes`` on the scripted
#: yardstick track plus ``--reference-eval-episodes`` against EACH pinned
#: reference, run SERIALLY on one borrowed arena.
#:
#: The gauntlet is sized FIRST and capped at
#: :data:`DEFAULT_REFERENCE_EVAL_EPISODES`, which is T13's own choice ("10 eps
#: each - 3 references x 10 = 30 eps/cycle keeps a cycle near 30-45 min");
#: whatever the time budget has left then goes to the scripted track, up to its
#: ceiling. Order matters more than the split: the gauntlet is the checkpoint
#: SELECTION input, so when episodes are expensive it is the scripted yardstick
#: that gets squeezed, never the gauntlet.
#:
#: :data:`REFERENCE_SHARE_OF_EVAL_BUDGET` is therefore a CAP on the gauntlet's
#: share of a tight budget, not a target split: with cheap episodes the cap is
#: never reached and the per-reference count simply lands on its default.
REFERENCE_SHARE_OF_EVAL_BUDGET = 0.75
MIN_REFERENCE_EVAL_EPISODES = 5
MIN_SCRIPTED_EVAL_EPISODES = 10
MAX_SCRIPTED_EVAL_EPISODES = 100

#: Eval cadence is rounded UP to a multiple of this, and never set below
#: ``MIN_EVAL_EVERY_GRAD_STEPS``: a cadence finer than the checkpoint interval
#: produces eval rows faster than the run produces distinguishable nets.
EVAL_CADENCE_ROUNDING = 500
MIN_EVAL_EVERY_GRAD_STEPS = 1000

#: Periodic ``--checkpoint`` saves to aim for across the whole window, and the
#: band the derived cadence is clamped to. ~20 saves is "lose at most 5% of the
#: night to a 4am crash" at 2.4 MB a save.
TARGET_PERIODIC_CHECKPOINTS = 20
MIN_CHECKPOINT_EVERY_GRAD_STEPS = 1000
MAX_CHECKPOINT_EVERY_GRAD_STEPS = 5000


def _round_up_to(value: float, multiple: int) -> int:
    """Smallest multiple of ``multiple`` that is >= ``value`` (>= multiple)."""
    if multiple < 1:
        raise ValueError(f"multiple must be >= 1, got {multiple}")
    steps = int(math.ceil(float(value) / float(multiple)))
    return max(1, steps) * int(multiple)


def select_episode_length(
    measurements: Mapping[str, Any], source: str = "probe"
) -> Tuple[Optional[float], str]:
    """Return ``(steps, human label)`` for the armored episode length to size on.

    Returns ``(None, label)`` when the canary recorded no usable figure — which
    is a REFUSAL upstream, never a cue to substitute a constant.
    """
    key, label = EPISODE_LENGTH_SOURCES.get(source, EPISODE_LENGTH_SOURCES["probe"])
    return _num(measurements.get(key)), label


def implied_episode_length(measurements: Mapping[str, Any]) -> Optional[float]:
    """The episode length implied by the canary's OWN collection rate.

    ``4.8782 transitions/s x 3600 / episodes-per-arena-hour``. This is the same
    stream measured a second way, and the two are compared because a mean
    episode length is easy to mis-derive — the project has done it twice.

    It is expected to run somewhat HIGH: the canary's rate includes replay
    warm-up and a whole eval cycle, during which episodes are collected slowly
    or not at all, so dividing by it over-states the length. That is why the
    disagreement threshold is a ratio band and not an equality.
    """
    per_arena_hour = _num(measurements.get("measured_episodes_per_arena_hour"))
    if per_arena_hour is None or per_arena_hour <= 0.0:
        return None
    return MEASURED_PER_ARENA_TRANSITIONS_PER_S * 3600.0 / per_arena_hour


def derive_sizing(
    measurements: Mapping[str, Any],
    *,
    window_hours: float,
    pads: int = PRODUCTION_PADS,
    fraction: float = EPS_DECAY_FRACTION_OF_RUN,
    episode_length_source: str = "probe",
    grad_steps_per_hour: Optional[float] = None,
    grad_steps_per_hour_source: str = "M3 retry (bare-handed, 25 pads)",
    overrides: Optional[Mapping[str, Any]] = None,
    thresholds: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Derive every launch number from the canary's measurement.

    Pure arithmetic: no field here is validated, and any input this cannot use
    comes back as ``None`` for :func:`evaluate_launch` to refuse on. Splitting
    derivation from judgement keeps the report able to print a half-derived
    chain next to the refusal that stopped it, which is what an operator at
    20:00 actually needs.

    Args:
        measurements: T17's ``canary_measurements.json``, as a dict.
        window_hours: The night's intended length. The ONE input no measurement
            supplies — a run length is a plan, not an observation.
        pads: Arena count for the long run.
        fraction: Share of the projected episodes the decay spans.
        episode_length_source: Key into :data:`EPISODE_LENGTH_SOURCES`.
        grad_steps_per_hour: The learner rate to budget from; ``None`` falls back
            to :data:`M3_RETRY_GRAD_STEPS_PER_HOUR`. The smoke supplies the
            armored measurement when it has one.
        grad_steps_per_hour_source: Human label for the above.
        overrides: Operator-supplied values that REPLACE a derived one. Each is
            still checked; an override is a different input, not an exemption.
        thresholds: Overrides for :data:`DEFAULT_THRESHOLDS`.

    Returns:
        A dict of every derived quantity, with ``None`` where the inputs did not
        support one.
    """
    limits = _thresholds(thresholds)
    over = dict(overrides or {})
    out: Dict[str, Any] = {}

    length, length_label = select_episode_length(measurements, episode_length_source)
    implied = implied_episode_length(measurements)
    out["episode_length_steps"] = length
    out["episode_length_source"] = episode_length_source
    out["episode_length_label"] = length_label
    out["episode_length_median_steps"] = _num(
        measurements.get("armored_median_episode_length_probe")
    )
    out["cross_check_episode_length_steps"] = implied
    out["cross_check_ratio"] = (
        max(length, implied) / min(length, implied)
        if length and implied and length > 0.0 and implied > 0.0
        else None
    )
    out["measured_at_run"] = measurements.get("measured_at_run")
    out["cap_hit_rate"] = _num(measurements.get("armored_cap_hit_rate"))
    out["max_episode_steps"] = _int(measurements.get("max_episode_steps")) or (
        MAX_EPISODE_STEPS
    )

    # -- measured length -> episodes/hour at N pads --------------------------
    # transitions/hour is measured and linear in the pad count (the sweep found
    # the per-arena rate flat from 16 to 25 with no knee); dividing by the
    # measured steps/episode is the whole conversion.
    out["pads"] = int(pads)
    out["transitions_per_s"] = float(pads) * MEASURED_PER_ARENA_TRANSITIONS_PER_S
    out["transitions_per_hour"] = out["transitions_per_s"] * 3600.0
    out["episodes_per_hour"] = (
        out["transitions_per_hour"] / length if length and length > 0.0 else None
    )
    # The canary's own rate scaled to the same width — a second opinion on the
    # line above, from a different measurement of the same fleet.
    out["episodes_per_hour_canary_projection"] = _num(
        measurements.get("projected_episodes_per_hour_at_25_pads")
    )

    # -- episodes in the window -> the decay window --------------------------
    out["window_hours"] = float(window_hours)
    out["projected_episodes"] = (
        out["episodes_per_hour"] * float(window_hours)
        if out["episodes_per_hour"] is not None
        else None
    )
    out["eps_decay_fraction_target"] = float(fraction)
    derived_decay = (
        max(1, int(round(float(fraction) * out["projected_episodes"])))
        if out["projected_episodes"] is not None
        else None
    )
    out["eps_decay_episodes_derived"] = derived_decay
    override_decay = _int(over.get("eps_decay_episodes"))
    out["eps_decay_episodes"] = (
        override_decay if override_decay is not None else derived_decay
    )
    out["eps_decay_source"] = (
        "operator override (--eps-decay-episodes)"
        if override_decay is not None
        else "derived from the canary measurement"
    )
    out["eps_decay_fraction"] = (
        float(out["eps_decay_episodes"]) / out["projected_episodes"]
        if out["eps_decay_episodes"] and out["projected_episodes"]
        else None
    )

    # -- what the DRIVER's own startup line will claim -----------------------
    # `agent.train.epsilon_schedule_report` prints the window as a percentage of
    # `train_config.projected_episodes(arenas)`, which is still built from the
    # bare-handed 285-step constant. So the run's opening lines will disagree
    # with the block above, by exactly this ratio, and an operator who does not
    # expect that will think the sizing is wrong at 20:00.
    stale_projected = (
        float(pads)
        * MEASURED_PER_ARENA_TRANSITIONS_PER_S
        * 3600.0
        * 12.0
        / STALE_EPISODE_STEPS_SCRIPTED_BARE
    )
    out["stale_projected_episodes"] = stale_projected
    out["stale_reported_fraction"] = (
        float(out["eps_decay_episodes"]) / stale_projected
        if out["eps_decay_episodes"]
        else None
    )

    # -- eval sizing ---------------------------------------------------------
    # An eval episode is played on ONE borrowed arena at the per-arena rate, so
    # its wall cost is steps / 4.8782 s. That model reproduces the measured
    # "100 episodes == 97 minutes" figure exactly: 100 x 285 / 4.8782 == 5842 s.
    #
    # The length used here is the canary's PERIODIC-EVAL figure, whose opponent
    # is the fixed scripted yardstick — ``eval_opponent`` names the driver, and
    # is carried through to the report so the regime can never be read off the
    # number alone. The canary writes this figure only when the cycle it probed
    # produced one, so its absence is an ordinary outcome, not a missing field.
    eval_length = _num(
        measurements.get("armored_mean_episode_length_eval_vs_scripted")
    )
    eval_opponent = measurements.get("eval_opponent")
    out["eval_opponent"] = eval_opponent
    if eval_length is None or eval_length <= 0.0:
        # Fall back to the SAME measured armored length, never to a constant.
        eval_length = length
        out["eval_opponent"] = None
        out["eval_length_source"] = (
            "the canary recorded no scripted-eval length for this cycle; using "
            "the probe's armored length instead"
        )
    else:
        out["eval_length_source"] = (
            "canary periodic eval vs the fixed scripted yardstick"
            + (f", {eval_opponent}" if eval_opponent else "")
        )
    out["eval_episode_length_steps"] = eval_length
    out["eval_episode_seconds"] = (
        eval_length / MEASURED_PER_ARENA_TRANSITIONS_PER_S
        if eval_length and eval_length > 0.0
        else None
    )

    episode_seconds = out["eval_episode_seconds"]
    if episode_seconds and episode_seconds > 0.0:
        # Aim at the MIDDLE of the 30-45 min band, not its ceiling. The eval
        # cadence below is derived from the cycle length, so a longer cycle buys
        # tighter win rates at the cost of FEWER eval points across the night —
        # and `elo/learner_rated` (AC7's rising-trend series) and the
        # best-checkpoint selector both read points, not precision.
        target_minutes = (
            limits["eval_cycle_target_min_minutes"]
            + limits["eval_cycle_target_max_minutes"]
        ) / 2.0
        budget_episodes = (target_minutes * 60.0) / episode_seconds
        reference_eps = min(
            DEFAULT_REFERENCE_EVAL_EPISODES,
            max(
                MIN_REFERENCE_EVAL_EPISODES,
                int(
                    budget_episodes
                    * REFERENCE_SHARE_OF_EVAL_BUDGET
                    / MAX_PINNED_REFERENCES
                ),
            ),
        )
        scripted_eps = int(budget_episodes - MAX_PINNED_REFERENCES * reference_eps)
        scripted_eps = max(
            MIN_SCRIPTED_EVAL_EPISODES, min(MAX_SCRIPTED_EVAL_EPISODES, scripted_eps)
        )
    else:
        reference_eps = None
        scripted_eps = None
    override_ref = _int(over.get("reference_eval_episodes"))
    override_scripted = _int(over.get("eval_episodes"))
    out["reference_eval_episodes_derived"] = reference_eps
    out["eval_episodes_derived"] = scripted_eps
    out["reference_eval_episodes"] = (
        override_ref if override_ref is not None else reference_eps
    )
    out["eval_episodes"] = (
        override_scripted if override_scripted is not None else scripted_eps
    )
    out["eval_episodes_source"] = (
        "operator override"
        if (override_ref is not None or override_scripted is not None)
        else "derived to land a 3-reference cycle inside the target band"
    )

    def _cycle_seconds(references: int) -> Optional[float]:
        if (
            episode_seconds is None
            or out["eval_episodes"] is None
            or out["reference_eval_episodes"] is None
        ):
            return None
        episodes = int(out["eval_episodes"]) + references * int(
            out["reference_eval_episodes"]
        )
        return episodes * episode_seconds

    out["eval_cycle_seconds_first"] = _cycle_seconds(1)
    out["eval_cycle_seconds_worst"] = _cycle_seconds(MAX_PINNED_REFERENCES)
    out["eval_cycle_minutes_first"] = (
        out["eval_cycle_seconds_first"] / 60.0
        if out["eval_cycle_seconds_first"] is not None
        else None
    )
    out["eval_cycle_minutes_worst"] = (
        out["eval_cycle_seconds_worst"] / 60.0
        if out["eval_cycle_seconds_worst"] is not None
        else None
    )
    # The number the plan text warns about, recomputed rather than quoted, so
    # the report can show what the DEFAULT would have cost tonight.
    out["default_eval_cycle_minutes"] = (
        (100 + MAX_PINNED_REFERENCES * DEFAULT_REFERENCE_EVAL_EPISODES)
        * episode_seconds
        / 60.0
        if episode_seconds
        else None
    )

    # -- learner rate, budgets, cadences -------------------------------------
    rate = _num(grad_steps_per_hour)
    if rate is None or rate <= 0.0:
        rate = M3_RETRY_GRAD_STEPS_PER_HOUR
        grad_steps_per_hour_source = (
            "M3 retry, bare-handed (no armored measurement available)"
        )
    out["grad_steps_per_hour"] = rate
    out["grad_steps_per_hour_source"] = grad_steps_per_hour_source
    out["projected_grad_steps"] = rate * float(window_hours)

    # An eval borrows the designated arena for a whole cycle; the cadence has to
    # be long enough that it is training in between. `next_eval_at` is re-armed
    # from `trainer.grad_step` AFTER the cycle, so this is a duty cycle, never a
    # back-to-back loop — but a duty near 1 still means that arena never trains.
    if out["eval_cycle_seconds_worst"]:
        min_gap = (
            out["eval_cycle_seconds_worst"] / 3600.0 * rate / limits["max_eval_duty"]
        )
        derived_eval_every = max(
            MIN_EVAL_EVERY_GRAD_STEPS, _round_up_to(min_gap, EVAL_CADENCE_ROUNDING)
        )
    else:
        derived_eval_every = None
    override_eval_every = _int(over.get("eval_every_grad_steps"))
    out["eval_every_grad_steps_derived"] = derived_eval_every
    out["eval_every_grad_steps"] = (
        override_eval_every if override_eval_every is not None else derived_eval_every
    )
    if out["eval_every_grad_steps"] and out["eval_cycle_seconds_worst"]:
        gap_seconds = float(out["eval_every_grad_steps"]) / rate * 3600.0
        out["eval_duty"] = out["eval_cycle_seconds_worst"] / (
            gap_seconds + out["eval_cycle_seconds_worst"]
        )
        out["eval_cycles_in_window"] = out["projected_grad_steps"] / float(
            out["eval_every_grad_steps"]
        )
    else:
        out["eval_duty"] = None
        out["eval_cycles_in_window"] = None

    margin = limits["budget_margin"]
    derived_max_episodes = (
        int(math.ceil(out["projected_episodes"] * margin))
        if out["projected_episodes"]
        else None
    )
    derived_max_grad_steps = int(math.ceil(out["projected_grad_steps"] * margin))
    override_max_episodes = _int(over.get("max_episodes"))
    override_max_grad_steps = _int(over.get("max_grad_steps"))
    out["budget_margin"] = margin
    out["max_episodes_derived"] = derived_max_episodes
    out["max_grad_steps_derived"] = derived_max_grad_steps
    out["max_episodes"] = (
        override_max_episodes if override_max_episodes is not None else derived_max_episodes
    )
    out["max_grad_steps"] = (
        override_max_grad_steps
        if override_max_grad_steps is not None
        else derived_max_grad_steps
    )

    derived_checkpoint_every = int(
        min(
            MAX_CHECKPOINT_EVERY_GRAD_STEPS,
            max(
                MIN_CHECKPOINT_EVERY_GRAD_STEPS,
                _round_up_to(
                    out["projected_grad_steps"] / TARGET_PERIODIC_CHECKPOINTS,
                    EVAL_CADENCE_ROUNDING,
                ),
            ),
        )
    )
    override_checkpoint_every = _int(over.get("checkpoint_every_grad_steps"))
    out["checkpoint_every_grad_steps_derived"] = derived_checkpoint_every
    out["checkpoint_every_grad_steps"] = (
        override_checkpoint_every
        if override_checkpoint_every is not None
        else derived_checkpoint_every
    )
    out["periodic_checkpoints_in_window"] = (
        out["projected_grad_steps"] / float(out["checkpoint_every_grad_steps"])
        if out["checkpoint_every_grad_steps"]
        else None
    )

    # Snapshot cadence is NOT derived: TrainConfig's 1000 and the promotion
    # steps are tuned decisions from T11a, and changing the archive rate changes
    # the pool PFSP samples from. Only the resulting pool size is reported, so
    # an absurd one is visible before the night rather than at 8am.
    override_snapshot_every = _int(over.get("snapshot_every_grad_steps"))
    out["snapshot_every_grad_steps"] = override_snapshot_every
    effective_snapshot_every = override_snapshot_every or 1000
    out["projected_pool_size"] = (
        1 + out["projected_grad_steps"] / float(effective_snapshot_every)
    )
    # 2.4 MB a snapshot, the plan's own figure.
    out["projected_pool_megabytes"] = out["projected_pool_size"] * 2.4
    return out


def sizing_arithmetic_lines(sizing: Mapping[str, Any]) -> List[str]:
    """The derivation, shown. Every number the launch depends on, in order.

    Printed rather than merely computed because the two previous times this was
    got wrong, the wrong value was plausible on its own and only obviously wrong
    next to the chain that produced it.
    """

    def fmt(value: Any, spec: str = ".4g", suffix: str = "") -> str:
        number = _num(value)
        if number is None:
            return "UNMEASURED"
        return f"{number:{spec}}{suffix}"

    pads = sizing.get("pads", PRODUCTION_PADS)
    lines = [
        "  measured armored mean episode length   "
        + fmt(sizing.get("episode_length_steps"), ".1f", " steps"),
        f"    source                               {sizing.get('episode_length_label')}",
        "    median (probe)                       "
        + fmt(sizing.get("episode_length_median_steps"), ".1f", " steps"),
        "    cap-hit rate                         "
        + fmt(sizing.get("cap_hit_rate"), ".3f")
        + f"   (an episode of {sizing.get('max_episode_steps')} steps is a draw, not a fight)",
        "  cross-check, same stream measured twice "
        + fmt(sizing.get("cross_check_episode_length_steps"), ".1f", " steps"),
        f"    = {MEASURED_PER_ARENA_TRANSITIONS_PER_S} transitions/s x 3600 / the canary's own episodes/arena/hour",
        "    ratio                                "
        + fmt(sizing.get("cross_check_ratio"), ".2f")
        + "   (expected slightly >1: the canary's rate includes warm-up and an eval cycle)",
        "",
        f"  episodes/hour at {pads} pads",
        f"    {pads} x {MEASURED_PER_ARENA_TRANSITIONS_PER_S} transitions/s x 3600      "
        + fmt(sizing.get("transitions_per_hour"), ".6g", " transitions/hour"),
        "    / the measured steps/episode         "
        + fmt(sizing.get("episodes_per_hour"), ".6g", " episodes/hour"),
        "    canary's own rate, scaled to 25 pads "
        + fmt(sizing.get("episodes_per_hour_canary_projection"), ".6g", " episodes/hour"),
        "",
        (
            "  episodes in the " + fmt(sizing.get("window_hours"), ".4g") + " h window"
        ).ljust(41)
        + fmt(sizing.get("projected_episodes"), ".7g", " episodes"),
        (
            "  eps decay window at "
            + fmt(sizing.get("eps_decay_fraction_target"), ".0%")
            + " of the run"
        ).ljust(41)
        + f"{sizing.get('eps_decay_episodes')}  <- --eps-decay-episodes",
        f"    how it was chosen                    {sizing.get('eps_decay_source')}",
        "    decay fraction of THIS run           "
        + fmt(sizing.get("eps_decay_fraction"), ".1%")
        + "   (a previous run shipped 142%: epsilon never finished decaying)",
        "    what the driver's startup line says  "
        + fmt(sizing.get("stale_reported_fraction"), ".1%")
        + "   (it divides by a 285-step, 12 h projection - IGNORE it)",
    ]
    return lines


# ---------------------------------------------------------------------------
# PART 3 — the launch gate. Every condition that must hold before a 24-hour run
# may start, one check function each, all fail-closed.
# ---------------------------------------------------------------------------


def _ok(code: str, detail: str) -> CheckResult:
    return CheckResult(code=code, passed=True, detail=detail)


def _refuse(code: str, detail: str, why: str, check: str) -> CheckResult:
    return CheckResult(code=code, passed=False, detail=detail, why=why, check=check)


def check_canary_present(plan: Mapping[str, Any]) -> CheckResult:
    """The armored episode length must EXIST. Absence is never a constant."""
    canary = _section(plan, "canary")
    path = canary.get("measurements_path")
    if not canary.get("exists") or not isinstance(canary.get("measurements"), Mapping):
        return _refuse(
            "CANARY_MEASUREMENTS_MISSING",
            f"no readable measurement at {path}",
            "T17's canary never wrote a measurement here, so the armored mean "
            "episode length is UNMEASURED. The only other numbers available are "
            f"{STALE_EPISODE_STEPS_SCRIPTED_BARE:g} (bare-handed, unarmed "
            f"opponent, 400-step cap) and {STALE_EPISODE_STEPS_M3_RETRY:g} (also "
            "bare-handed), and sizing epsilon from either is the mistake this "
            "gate exists to prevent.",
            "run scripts/canary_selfplay.sh --warm-start <abs path> --arenas 25 "
            "first; it writes canary_measurements.json on every run, refused or "
            "not.",
        )
    return _ok(
        "CANARY_MEASUREMENTS_MISSING",
        f"measurement present at {path}",
    )


def check_canary_fresh(plan: Mapping[str, Any]) -> CheckResult:
    """The measurement must describe THIS fleet and a recent codebase.

    Three ways it can fail, and the third is the load-bearing one:

      * older than ``max_measurement_age_hours``;
      * dated in the future beyond the clock-skew allowance, which means
        freshness cannot be reasoned about at all;
      * older than the youngest bridge process in the fleet. The canary's
        ``OPPONENT_FROZEN`` check is the ONLY empirical evidence that
        ``DUMMY_KNOCKBACK_IMMUNE=false`` reached the bridges — macOS exposes no
        other process's environment and start-pads.sh passes it as an env var,
        not argv — and that evidence covers only the processes it probed. A pad
        restarted since is an unproven pad.
    """
    limits = _thresholds(plan.get("thresholds"))
    canary = _section(plan, "canary")
    fleet = _section(plan, "fleet")
    now = _num(plan.get("now_epoch"))
    mtime = _num(canary.get("mtime"))
    if now is None or mtime is None:
        return _refuse(
            "CANARY_MEASUREMENTS_STALE",
            f"now={plan.get('now_epoch')!r} mtime={canary.get('mtime')!r}",
            "the measurement's age could not be computed, so its freshness is "
            "unknown. Absent evidence is a refusal here, never a pass.",
            "check that the canary directory is readable and the system clock is "
            "sane.",
        )
    age_hours = (now - mtime) / 3600.0
    if age_hours < -(limits["max_clock_skew_seconds"] / 3600.0):
        return _refuse(
            "CANARY_MEASUREMENTS_STALE",
            f"measurement is dated {-age_hours:.2f} h in the FUTURE",
            "the measurement file is newer than the current time, so the clock "
            "cannot be used to decide whether anything here is fresh.",
            "fix the system clock, then re-run the canary.",
        )
    if age_hours > limits["max_measurement_age_hours"]:
        return _refuse(
            "CANARY_MEASUREMENTS_STALE",
            f"measurement is {age_hours:.1f} h old "
            f"(limit {limits['max_measurement_age_hours']:.0f} h)",
            "the armored episode length was measured against a fleet and a "
            "codebase that have had a working day to change. Sizing a whole "
            "night on it is sizing on a guess with a timestamp.",
            "re-run scripts/canary_selfplay.sh against the fleet you are about "
            "to launch on.",
        )
    youngest_age = _num(fleet.get("youngest_listener_age_seconds"))
    if youngest_age is not None:
        started_at = now - youngest_age
        if started_at > mtime:
            return _refuse(
                "CANARY_MEASUREMENTS_STALE",
                f"a bridge started {(started_at - mtime) / 60.0:.1f} min AFTER the "
                "measurement was written",
                "at least one pad in this fleet was booted after the canary ran, "
                "so nothing has verified that IT was started with "
                "DUMMY_KNOCKBACK_IMMUNE=false. That flag cannot be read back from "
                "a running process on macOS; the canary's OPPONENT_FROZEN check is "
                "the only proof, and it only covers the processes it probed. An "
                "immune, speed-pinned opponent produces clean logs and a wasted "
                "night.",
                "re-run the canary against the CURRENT fleet, or reboot the whole "
                "fleet with DUMMY_KNOCKBACK_IMMUNE=false "
                "server/setup/start-pads.sh --pads 25 and re-run it.",
            )
    return _ok(
        "CANARY_MEASUREMENTS_STALE",
        f"measurement is {age_hours:.1f} h old and predates no bridge restart",
    )


def check_canary_green(plan: Mapping[str, Any]) -> CheckResult:
    """T17's own verdict must be GREEN, re-read from its evidence.

    The exit code comes from ``canary_selfplay.sh --analyze-only``, which
    re-judges an existing evidence directory and connects to nothing. ``None``
    (never run) refuses: a canary whose verdict was not read is a canary that
    did not gate anything.
    """
    canary = _section(plan, "canary")
    exit_code = _int(canary.get("analyze_exit"))
    if exit_code is None:
        return _refuse(
            "CANARY_NOT_GREEN",
            "the canary verdict was never re-read",
            "nothing confirmed that T17's gate PASSED. Its measurements are "
            "written on every run, refused or not, so their presence proves the "
            "canary ran — not that it cleared.",
            "run scripts/canary_selfplay.sh --analyze-only "
            f"{canary.get('directory') or '<canary dir>'} and read its verdict.",
        )
    if exit_code != 0:
        return _refuse(
            "CANARY_NOT_GREEN",
            f"canary --analyze-only exited {exit_code}",
            "T17's gate REFUSED this run. Every condition it checks — a snapshot "
            "that actually changed, valid PFSP probabilities, a non-empty rated "
            "Elo series, armor on both fighters, an opponent that moves — is a "
            "precondition for the night being worth anything.",
            "read the canary's REFUSALS block; each names what is wrong and what "
            "to check. Fix it and re-run the canary before coming back here.",
        )
    return _ok("CANARY_NOT_GREEN", "canary --analyze-only exited 0 (GREEN)")


def check_episode_length(plan: Mapping[str, Any], sizing: Mapping[str, Any]) -> List[CheckResult]:
    """The measured length must exist, be plausible, and agree with itself."""
    limits = _thresholds(plan.get("thresholds"))
    length = _num(sizing.get("episode_length_steps"))
    results: List[CheckResult] = []
    if length is None or length <= 0.0:
        results.append(
            _refuse(
                "EPISODE_LENGTH_UNMEASURED",
                f"{sizing.get('episode_length_source')} length is "
                f"{sizing.get('episode_length_steps')!r}",
                "the canary wrote a measurement file but no usable armored "
                "episode length in it, so there is nothing to size from. The "
                "fallbacks are the two stale constants, and both describe a "
                "bare-handed fight against an unarmed opponent.",
                "check the canary's probe section: a refused or aborted probe "
                "leaves these fields null.",
            )
        )
        return results
    results.append(
        _ok("EPISODE_LENGTH_UNMEASURED", f"measured {length:.1f} steps/episode")
    )

    cap = float(sizing.get("max_episode_steps") or MAX_EPISODE_STEPS)
    cap_limit = cap * limits["max_episode_length_cap_fraction"]
    if length < limits["min_episode_length_steps"]:
        results.append(
            _refuse(
                "EPISODE_LENGTH_IMPLAUSIBLE",
                f"{length:.1f} steps is below the {limits['min_episode_length_steps']:.0f}-step floor",
                "an episode this short is not a fight — it is a reset loop, a "
                "spawn kill, or an episode that aborted. Dividing the hour's "
                "transitions by it inflates the episode count and stretches the "
                "decay window over a run that will not happen.",
                "read the canary's episode-length list; a handful of 1-2 step "
                "episodes means the arena is not producing fights.",
            )
        )
    elif length > cap_limit:
        results.append(
            _refuse(
                "EPISODE_LENGTH_IMPLAUSIBLE",
                f"{length:.1f} steps is above {cap_limit:.0f} "
                f"({limits['max_episode_length_cap_fraction']:.0%} of the {cap:.0f}-step cap)",
                "the 'episodes' being measured are cap-hit DRAWS, not fights. "
                "Two armored self-play agents converging on mutual stalling is "
                "the degenerate equilibrium this project is most likely to hit, "
                "and sizing epsilon off draw lengths bakes it in.",
                "read the canary's DRAW_MAJORITY checks and cap-hit rate; if the "
                "fleet really is drawing, epsilon sizing is not the problem.",
            )
        )
    else:
        results.append(
            _ok(
                "EPISODE_LENGTH_IMPLAUSIBLE",
                f"{length:.1f} steps sits between {limits['min_episode_length_steps']:.0f} "
                f"and {cap_limit:.0f}",
            )
        )

    ratio = _num(sizing.get("cross_check_ratio"))
    if ratio is None:
        results.append(
            _refuse(
                "EPISODE_LENGTH_DISAGREEMENT",
                "the canary recorded no collection rate to cross-check against",
                "one measurement of an episode length with nothing to check it "
                "against is how the 30-step figure survived long enough to size a "
                "run at 142% of itself.",
                "check measured_episodes_per_arena_hour in the canary "
                "measurement; a driver that never completed leaves it null.",
            )
        )
    elif ratio > limits["max_episode_length_ratio"]:
        results.append(
            _refuse(
                "EPISODE_LENGTH_DISAGREEMENT",
                f"probe {length:.1f} steps vs rate-implied "
                f"{_num(sizing.get('cross_check_episode_length_steps')):.1f} steps "
                f"(ratio {ratio:.2f} > {limits['max_episode_length_ratio']:.2f})",
                "the same stream measured two ways disagrees by more than "
                "warm-up and eval overhead can explain, so one of the two numbers "
                "is wrong and nothing here can tell which.",
                "compare the canary's probe episode lengths against its "
                "training episodes/arena/hour; a huge gap usually means the probe "
                "and the training run fought different opponents or different "
                "gear.",
            )
        )
    else:
        results.append(
            _ok(
                "EPISODE_LENGTH_DISAGREEMENT",
                f"probe and rate-implied lengths agree within {ratio:.2f}x",
            )
        )
    return results


def check_eps_decay(plan: Mapping[str, Any], sizing: Mapping[str, Any]) -> CheckResult:
    """The decay window must be a sane FRACTION of the projected run.

    This is the check the historical failure walks into: a window spanning 142%
    of the run leaves epsilon near its start value all night, and one spanning
    0.1% floors it before the agent has seen anything. Neither direction is
    safe, so the band refuses at both ends — including when the operator passes
    the value by hand, which is the only way an absurd one gets in now.
    """
    limits = _thresholds(plan.get("thresholds"))
    window = _int(sizing.get("eps_decay_episodes"))
    fraction = _num(sizing.get("eps_decay_fraction"))
    if window is None or window < 1 or fraction is None:
        return _refuse(
            "EPS_DECAY_ABSURD",
            f"eps_decay_episodes={window!r} fraction={fraction!r}",
            "the decay window could not be derived, so the run would fall back "
            "to TrainConfig's default — which is sized from the bare-handed "
            f"{STALE_EPISODE_STEPS_SCRIPTED_BARE:g}-step constant and is wrong "
            "for an armored 25-pad night by a factor of about three.",
            "fix whatever blocked the episode-length chain above; do not pass "
            "--eps-decay-episodes to paper over it.",
        )
    if fraction > limits["max_eps_decay_fraction"]:
        return _refuse(
            "EPS_DECAY_ABSURD",
            f"{window} episodes is {fraction:.1%} of the projected run "
            f"(ceiling {limits['max_eps_decay_fraction']:.0%})",
            "epsilon would still be well above its floor when the night ends. "
            "This is the exact shape of the failure that shipped once: a window "
            "spanning 142% of the run, epsilon parked near 0.25 until morning, "
            "and a night of mostly-random play with a clean log.",
            "either shorten the window or lengthen --window-hours to match the "
            "run you actually intend.",
        )
    if fraction < limits["min_eps_decay_fraction"]:
        return _refuse(
            "EPS_DECAY_ABSURD",
            f"{window} episodes is {fraction:.1%} of the projected run "
            f"(floor {limits['min_eps_decay_fraction']:.0%})",
            "epsilon reaches its floor almost immediately, so the run explores "
            "for a few percent of the night and exploits a barely-trained policy "
            "for the rest. This is the other half of the same bug — every pad "
            "claims from ONE shared episode counter, so a single-arena number is "
            "spent 25 times too fast.",
            "let the value be derived rather than passing a single-arena figure.",
        )
    return _ok(
        "EPS_DECAY_ABSURD",
        f"{window} episodes == {fraction:.1%} of the projected run "
        f"(target {sizing.get('eps_decay_fraction_target'):.0%})",
    )


def check_smoke_cleared(plan: Mapping[str, Any]) -> CheckResult:
    """A GREEN 25-pad smoke is a precondition, not a nicety.

    The canary runs at 4 pads with a lowered replay floor. It says nothing about
    whether 25 collectors, each holding a second frozen DRQN and reading a
    snapshot off disk at the head of every episode, still collect at the
    measured rate — which is the only reason the sizing above is arithmetic
    rather than hope.
    """
    limits = _thresholds(plan.get("thresholds"))
    smoke = _section(plan, "smoke")
    if not smoke.get("exists") or not isinstance(smoke.get("measurements"), Mapping):
        return _refuse(
            "SMOKE_NOT_CLEARED",
            f"no smoke measurement at {smoke.get('measurements_path')}",
            "nothing has run this configuration at full width. Every throughput "
            "number the sizing above uses is a 4-pad canary reading scaled by a "
            "constant.",
            "run scripts/launch_selfplay.sh smoke --warm-start <abs path>.",
        )
    if not smoke.get("verdict_ok"):
        return _refuse(
            "SMOKE_NOT_CLEARED",
            f"the smoke verdict was {smoke.get('verdict_ok')!r}",
            "the 25-pad smoke REFUSED. Its refusals are throughput, backlog, "
            "memory and snapshot-load latency — the things that decide whether a "
            "24-hour run finishes or wedges at 3am.",
            "read the smoke report's REFUSALS block and fix the named condition.",
        )
    measured = dict(smoke.get("measurements") or {})
    pads = _num(measured.get("arenas"))
    if pads is None or pads < limits["smoke_min_pads"]:
        return _refuse(
            "SMOKE_NOT_CLEARED",
            f"the smoke ran at {measured.get('arenas')!r} pads, not "
            f"{limits['smoke_min_pads']:.0f}",
            "a narrower smoke measures a different machine load. The whole "
            "question is whether the fleet holds at FULL width.",
            "re-run the smoke with --arenas 25.",
        )
    return _ok(
        "SMOKE_NOT_CLEARED",
        f"smoke cleared at {pads:.0f} pads",
    )


def check_fleet(plan: Mapping[str, Any]) -> CheckResult:
    """Paper answering, every bridge listening, and NO bridge already claimed.

    The last clause is not a warning. ``BridgeServer`` accepts exactly ONE TCP
    client and resolves a second connection by destroying the incumbent, so
    launching onto an occupied port takes down whatever is already there — four
    outages in this project came from exactly that.
    """
    fleet = _section(plan, "fleet")
    arenas = _int(plan.get("arenas")) or 0
    if not fleet.get("mc_reachable"):
        return _refuse(
            "FLEET_NOT_READY",
            f"nothing answering on {fleet.get('host')}:{fleet.get('mc_port')}",
            "the Paper server is not up, so there is no world to fight in. The "
            "boot order is Paper -> bridges -> driver and this script owns only "
            "the last step.",
            "DUMMY_KNOCKBACK_IMMUNE=false server/setup/start-pads.sh --pads "
            f"{arenas}, and wait for the FLEET READY line.",
        )
    missing = list(fleet.get("missing_ports") or [])
    if missing:
        return _refuse(
            "FLEET_NOT_READY",
            f"no bridge listening on {missing}",
            "the fleet is narrower than the run. Arena i connects to port "
            "base+i, so a missing listener is an arena that will never collect.",
            "DUMMY_KNOCKBACK_IMMUNE=false server/setup/start-pads.sh --pads "
            f"{arenas}, and wait for FLEET READY.",
        )
    busy = list(fleet.get("busy_ports") or [])
    if busy:
        return _refuse(
            "FLEET_NOT_READY",
            f"bridge port(s) already have an established client: {busy}",
            "BridgeServer accepts exactly ONE TCP client and a second connection "
            "silently DESTROYS the first. Launching now would take down whatever "
            "is attached — a canary, a probe, or a run already training.",
            "find the other client (lsof -nP -iTCP:<port>) and stop it. Never "
            "connect-probe a bridge to find out.",
        )
    listeners = _int(fleet.get("listener_count"))
    if listeners is None or listeners < arenas:
        return _refuse(
            "FLEET_NOT_READY",
            f"{listeners!r} bridge listeners for {arenas} arenas",
            "fewer bridges are listening than the run will connect to.",
            "check server/logs/pads/ for a pad that failed to boot.",
        )
    return _ok(
        "FLEET_NOT_READY",
        f"Paper answering and {listeners} bridge listener(s) free",
    )


def check_warm_start(plan: Mapping[str, Any]) -> CheckResult:
    """An absolute, existing, hashed warm start.

    ``TrainConfig`` refuses ``--opponent selfplay`` without one (AC14), and the
    pool's snapshot 0 — the first PINNED member, never dropped and never evicted
    — is seeded entirely from this file. A stale path or a half-copied file
    loads perfectly cleanly into the same architecture and becomes a permanent
    reference opponent, which is why T11b's ``--warm-start-sha256`` gate exists
    and why this refuses without a digest.
    """
    warm = _section(plan, "warm_start")
    path = warm.get("path")
    if not path:
        return _refuse(
            "WARM_START_UNUSABLE",
            "no --warm-start given",
            "a self-play run has no past selves to fight without one: snapshot 0 "
            "IS the warm start. TrainConfig raises rather than starting.",
            "pass --warm-start with an ABSOLUTE path to the frozen M3 "
            "checkpoint.",
        )
    if not str(path).startswith("/"):
        return _refuse(
            "WARM_START_UNUSABLE",
            f"warm start {path!r} is not an absolute path",
            "the driver is started detached with a different working directory "
            "than the shell that launched it, and a relative warm start would "
            "resolve somewhere else — or, worse, resolve to a DIFFERENT "
            "checkpoint in another checkout.",
            "pass the full path, e.g. "
            "/Users/diego/Documents/MinecraftRL/runs/m4.best.pt.",
        )
    if not warm.get("is_file"):
        return _refuse(
            "WARM_START_UNUSABLE",
            f"warm start {path} does not exist or is not a file",
            "the run would raise at startup, in the first second — but only "
            "after the fleet has been booted and the operator has gone to bed.",
            "check the path; the bare-handed run's checkpoints live in the MAIN "
            "checkout's runs/, not in this worktree.",
        )
    digest = warm.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64 or digest != digest.lower():
        return _refuse(
            "WARM_START_UNUSABLE",
            f"unusable sha256 {digest!r}",
            "without a digest the run cannot verify it loaded the checkpoint the "
            "operator meant. T11b's --warm-start-sha256 gate compares the file's "
            "actual hash and refuses on a mismatch, naming both digests.",
            "check that shasum ran; the launch records the digest it computed.",
        )
    return _ok(
        "WARM_START_UNUSABLE",
        f"{path} ({warm.get('bytes')} bytes, sha256 {digest[:12]}...)",
    )


def check_run_name(plan: Mapping[str, Any]) -> CheckResult:
    """The run must not overwrite another run — least of all its own warm start.

    ``runs/m4.pt`` / ``runs/m4.best.pt`` belong to the COMPLETED bare-handed run
    and are this run's warm-start source. Launching under ``--run-name m4``
    would point the periodic checkpoint hook at the very file the pool seeds
    snapshot 0 from, and the warm start would be overwritten by this run's own
    output while the run was still using it.
    """
    run_name = str(plan.get("run_name") or "").strip()
    warm = _section(plan, "warm_start")
    warm_path = str(warm.get("path") or "")
    if not run_name:
        return _refuse(
            "RUN_NAME_COLLISION",
            "empty --run-name",
            "an empty run name collapses the snapshot pool path to "
            "runs/snapshots and merges two runs' pools "
            "(agent.train.snapshot_pool_directory raises on it).",
            f"pass --run-name {EXPECTED_RUN_NAME}.",
        )
    if run_name in FORBIDDEN_RUN_NAMES:
        return _refuse(
            "RUN_NAME_COLLISION",
            f"--run-name {run_name} names a completed run",
            f"runs/{run_name}.* already belongs to a finished run, and "
            f"runs/{run_name}.best.pt is this run's warm-start source. The "
            "periodic checkpoint hook would rewrite it mid-run, replacing the "
            "warm start with this run's own output while snapshot 0 still points "
            "at it.",
            f"use --run-name {EXPECTED_RUN_NAME}.",
        )
    existing = [str(p) for p in (plan.get("existing_outputs") or [])]
    if existing:
        return _refuse(
            "RUN_NAME_COLLISION",
            f"{len(existing)} output(s) already exist: {existing[:4]}",
            "this run's own output paths are already occupied, so starting would "
            "overwrite whatever produced them — most likely an earlier attempt "
            "whose checkpoints are the only record of it.",
            "move or delete them deliberately, or pick a different --run-name.",
        )
    if warm_path and os.path.basename(warm_path).split(".")[0] == run_name:
        return _refuse(
            "RUN_NAME_COLLISION",
            f"the warm start {warm_path} lives in --run-name {run_name}'s namespace",
            "the run would write its checkpoints over its own warm start.",
            "rename the run, or copy the warm start somewhere this run does not "
            "write.",
        )
    return _ok(
        "RUN_NAME_COLLISION",
        f"--run-name {run_name}: no existing output, warm start is elsewhere",
    )


def check_checkpoints(plan: Mapping[str, Any]) -> CheckResult:
    """``--checkpoint`` is mandatory, and never ``--best-checkpoint`` alone.

    Only the ``--checkpoint`` hook drives ``_save_latest``, which is BOTH the
    periodic and the final save. With it unset the sole remaining write sits
    behind an eval that must STRICTLY improve a win rate above zero — so a run
    whose agent never wins an eval episode leaves an empty ``runs/`` and a clean
    log. ``agent/train.py`` prints this warning at startup and then trains for
    twelve hours anyway; here it is a refusal.
    """
    checkpoint = plan.get("checkpoint")
    best = plan.get("best_checkpoint")
    if not checkpoint:
        return _refuse(
            "CHECKPOINT_UNSAFE",
            f"--checkpoint={checkpoint!r} --best-checkpoint={best!r}",
            "with no --checkpoint the periodic save is DISABLED and the final "
            "save is a no-op. If some eval does not strictly improve the win "
            "rate above zero, the whole night writes nothing at all.",
            "pass --checkpoint runs/<run>.pt. --best-checkpoint is fine "
            "ALONGSIDE it and is derived automatically when omitted.",
        )
    if best and str(best) == str(checkpoint):
        return _refuse(
            "CHECKPOINT_UNSAFE",
            f"--checkpoint and --best-checkpoint are both {checkpoint}",
            "one path for both means the next periodic save overwrites the best "
            "net with a more recent, worse one — selection by recency with extra "
            "steps.",
            "leave --best-checkpoint off and let _best_checkpoint_path derive "
            "<name>.best.pt from --checkpoint.",
        )
    return _ok(
        "CHECKPOINT_UNSAFE",
        f"latest -> {checkpoint}, best -> {best}",
    )


def check_eval_sizing(plan: Mapping[str, Any], sizing: Mapping[str, Any]) -> List[CheckResult]:
    """The eval cycle must fit the night, and the cadence must leave room to train."""
    limits = _thresholds(plan.get("thresholds"))
    results: List[CheckResult] = []
    minutes = _num(sizing.get("eval_cycle_minutes_worst"))
    if minutes is None:
        results.append(
            _refuse(
                "EVAL_CYCLE_TOO_LONG",
                "the eval cycle cost could not be computed",
                "without an armored eval-episode length there is no way to know "
                "whether a cycle costs half an hour or two.",
                "check the canary's armored_mean_episode_length_eval_vs_scripted "
                "AND its probe length: the eval figure falls back to the probe's, "
                "so reaching here means BOTH are absent.",
            )
        )
    elif minutes > limits["eval_cycle_hard_max_minutes"]:
        results.append(
            _refuse(
                "EVAL_CYCLE_TOO_LONG",
                f"a 3-reference cycle costs {minutes:.0f} min "
                f"(hard limit {limits['eval_cycle_hard_max_minutes']:.0f} min)",
                "eval episodes are played SERIALLY on one borrowed arena. The "
                "measured cost of 100 episodes at the bare-handed length was 97 "
                "minutes, and a cycle that long turns the run into an eval "
                "harness that occasionally trains.",
                "lower --eval-episodes and/or --reference-eval-episodes; the "
                "derived values land the cycle inside the target band.",
            )
        )
    else:
        band = (
            "inside"
            if limits["eval_cycle_target_min_minutes"]
            <= minutes
            <= limits["eval_cycle_target_max_minutes"]
            else "outside"
        )
        results.append(
            _ok(
                "EVAL_CYCLE_TOO_LONG",
                f"3-reference cycle {minutes:.0f} min ({band} the "
                f"{limits['eval_cycle_target_min_minutes']:.0f}-"
                f"{limits['eval_cycle_target_max_minutes']:.0f} min target; the "
                f"default 100+30 episodes would cost "
                f"{_num(sizing.get('default_eval_cycle_minutes')) or float('nan'):.0f} min)",
            )
        )

    duty = _num(sizing.get("eval_duty"))
    if duty is None:
        results.append(
            _refuse(
                "EVAL_CADENCE_TOO_TIGHT",
                "the eval duty cycle could not be computed",
                "without it, nothing bounds how much of the night the designated "
                "arena spends in eval instead of collecting.",
                "check --eval-every-grad-steps and the cycle cost above.",
            )
        )
    elif duty > limits["eval_duty_hard_max"]:
        results.append(
            _refuse(
                "EVAL_CADENCE_TOO_TIGHT",
                f"the designated arena would spend {duty:.0%} of the night in eval "
                f"(hard limit {limits['eval_duty_hard_max']:.0%})",
                "evals re-arm from the CURRENT grad step after each cycle, so "
                "they never run back-to-back — but at this cadence the borrowed "
                "arena is in eval far more than it is training, and every cycle "
                "is scored on a net barely different from the last.",
                "raise --eval-every-grad-steps; the derived value targets "
                f"{limits['max_eval_duty']:.0%}.",
            )
        )
    else:
        results.append(
            _ok(
                "EVAL_CADENCE_TOO_TIGHT",
                f"eval duty {duty:.0%} on the designated arena, "
                f"~{_num(sizing.get('eval_cycles_in_window')) or 0:.0f} cycles in the window",
            )
        )
    return results


def check_budgets(plan: Mapping[str, Any], sizing: Mapping[str, Any]) -> CheckResult:
    """The episode and grad-step budgets must outlast the window.

    ``--max-episodes`` defaults to 10,000, which at 25 armored pads is a couple
    of hours: the run stops with ``stop_reason=max_episodes``, saves, and sits
    idle until morning. The M3 run hit this and raised it deliberately.
    """
    projected_episodes = _num(sizing.get("projected_episodes"))
    projected_steps = _num(sizing.get("projected_grad_steps"))
    max_episodes = _int(sizing.get("max_episodes"))
    max_grad_steps = _int(sizing.get("max_grad_steps"))
    if (
        projected_episodes is None
        or projected_steps is None
        or max_episodes is None
        or max_grad_steps is None
    ):
        return _refuse(
            "BUDGET_ENDS_EARLY",
            f"episodes={max_episodes!r}/{projected_episodes!r} "
            f"grad_steps={max_grad_steps!r}/{projected_steps!r}",
            "the budgets could not be checked against the window, so nothing "
            "rules out a run that stops at 2am.",
            "fix the sizing chain above.",
        )
    if max_episodes < projected_episodes:
        hours = projected_episodes and (
            max_episodes / projected_episodes * float(sizing.get("window_hours") or 0.0)
        )
        note = (
            " (this is argparse's DEFAULT — it is not a choice, it is what "
            "happens when the flag is omitted)"
            if max_episodes == DEFAULT_MAX_EPISODES
            else ""
        )
        return _refuse(
            "BUDGET_ENDS_EARLY",
            f"--max-episodes {max_episodes} < the {projected_episodes:.0f} episodes "
            f"projected for the window{note}",
            f"the run would stop after about {hours:.1f} h with "
            "stop_reason=max_episodes and then sit idle until morning, wasting "
            "the rest of the night.",
            f"pass --max-episodes {sizing.get('max_episodes_derived')} or more.",
        )
    if max_grad_steps < projected_steps:
        return _refuse(
            "BUDGET_ENDS_EARLY",
            f"--max-grad-steps {max_grad_steps} < the {projected_steps:.0f} steps "
            "projected for the window",
            "the learner is the single clock: it stops at this budget and takes "
            "the collectors down with it, however much window is left.",
            f"pass --max-grad-steps {sizing.get('max_grad_steps_derived')} or more.",
        )
    return _ok(
        "BUDGET_ENDS_EARLY",
        f"budgets {max_episodes} episodes / {max_grad_steps} grad steps cover the "
        f"projected {projected_episodes:.0f} / {projected_steps:.0f}",
    )


def check_opponent(plan: Mapping[str, Any]) -> CheckResult:
    """``--opponent selfplay``, or this is a differently-named scripted run."""
    opponent = plan.get("opponent")
    if opponent != "selfplay":
        return _refuse(
            "OPPONENT_NOT_SELFPLAY",
            f"--opponent {opponent!r}",
            "M4 is the self-play run. Any other value trains against the "
            "scripted curriculum or an idle dummy while every self-play artifact "
            "— the pool, PFSP, both Elo series, the reference gauntlet — stays "
            "empty, and nothing raises.",
            "pass --opponent selfplay.",
        )
    return _ok("OPPONENT_NOT_SELFPLAY", "--opponent selfplay")


def evaluate_launch(plan: Mapping[str, Any]) -> Verdict:
    """Run every launch check over ``plan`` and return the verdict.

    Order matters only for readability — every check runs regardless, so one
    invocation reports everything that is wrong rather than the first thing.
    """
    version = _int(plan.get("plan_version"))
    if version != PLAN_VERSION:
        return Verdict(
            checks=[
                _refuse(
                    "PLAN_VERSION",
                    f"plan_version={plan.get('plan_version')!r}, expected {PLAN_VERSION}",
                    "this document was written by a different revision of the "
                    "collector, so fields this gate reads may mean something "
                    "else or be absent. Reading them anyway is how a missing "
                    "field becomes a healthy zero.",
                    "re-run the launch script so the collector and the gate come "
                    "from the same file.",
                )
            ],
            facts={},
        )

    canary = _section(plan, "canary")
    smoke = _section(plan, "smoke")
    smoke_measurements = (
        dict(smoke.get("measurements") or {}) if isinstance(smoke.get("measurements"), Mapping) else {}
    )
    # The armored learner rate, when the smoke measured one. Falling back to the
    # M3 retry's bare-handed 4570/hour is explicit and labelled, never silent.
    smoke_rate = _num(smoke_measurements.get("grad_steps_per_hour"))
    sizing = derive_sizing(
        dict(canary.get("measurements") or {})
        if isinstance(canary.get("measurements"), Mapping)
        else {},
        window_hours=_num(plan.get("window_hours")) or 0.0,
        pads=_int(plan.get("arenas")) or PRODUCTION_PADS,
        episode_length_source=str(plan.get("episode_length_source") or "probe"),
        grad_steps_per_hour=smoke_rate,
        grad_steps_per_hour_source=(
            f"the {smoke_measurements.get('arenas')}-pad smoke (ARMORED self-play)"
            if smoke_rate
            else "M3 retry, bare-handed (no armored measurement available)"
        ),
        overrides=_section(plan, "overrides"),
        thresholds=plan.get("thresholds"),
    )

    checks: List[CheckResult] = [
        check_canary_present(plan),
        check_canary_fresh(plan),
        check_canary_green(plan),
    ]
    checks.extend(check_episode_length(plan, sizing))
    checks.append(check_eps_decay(plan, sizing))
    checks.append(check_smoke_cleared(plan))
    checks.append(check_fleet(plan))
    checks.append(check_warm_start(plan))
    checks.append(check_run_name(plan))
    checks.append(check_checkpoints(plan))
    checks.extend(check_eval_sizing(plan, sizing))
    checks.append(check_budgets(plan, sizing))
    checks.append(check_opponent(plan))

    facts: Dict[str, Any] = {"sizing": sizing}
    facts["argv"] = build_launch_argv(plan, sizing)
    facts["command"] = format_launch_command(plan, facts["argv"])
    facts["fleet_boot_command"] = fleet_boot_command(plan)
    return Verdict(checks=checks, facts=facts)


def fleet_boot_command(plan: Mapping[str, Any]) -> str:
    """The operator's step 1+2, printed verbatim so it cannot be mistyped.

    ``DUMMY_KNOCKBACK_IMMUNE=false`` belongs HERE and nowhere else: the bridges
    read it at their own startup. The driver derives its own launcher setting
    from ``cfg.opponent == "dummy"``, so a self-play run's RELAUNCHED bridges are
    already non-immune — the flag only matters for the pads booted before the
    run, which are exactly the ones this command starts.
    """
    arenas = _int(plan.get("arenas")) or PRODUCTION_PADS
    return (
        "DUMMY_KNOCKBACK_IMMUNE=false server/setup/start-pads.sh "
        f"--pads {arenas}"
    )


def build_launch_argv(
    plan: Mapping[str, Any], sizing: Mapping[str, Any]
) -> List[str]:
    """The exact ``agent.train`` argv this plan launches.

    Built here rather than in the shell so the tests can assert on it directly:
    a flag that goes missing between the gate and the command line is a gate
    that checked something the run did not do.

    Every value is either measured, derived above, or an operator override that
    the checks have already seen. Flags whose TrainConfig default is the right
    answer are OMITTED unless overridden — ``--snapshot-every-grad-steps``,
    ``--opponent-epsilon``, ``--reference-promote-grad-steps``, ``--elo-k``,
    ``--elo-initial`` and ``--warm-start-eps-start`` are tuned decisions from
    T11a, and restating them here would create a second place for them to drift.
    Only the first of the six has an override at all: an explicit
    ``--snapshot-every-grad-steps`` reaches ``sizing`` and IS emitted below. The
    other five have no operator path and never appear.
    """
    warm = _section(plan, "warm_start")
    argv: List[str] = [
        "--arenas", str(_int(plan.get("arenas")) or PRODUCTION_PADS),
        # The whole point of the run. Also the flag that makes --warm-start
        # mandatory and turns on the opponent-seat observation mirror at BOTH
        # env construction sites.
        "--opponent", "selfplay",
        "--host", str(plan.get("host") or "127.0.0.1"),
        "--port", str(_int(plan.get("bridge_base_port")) or 5555),
        "--mc-port", str(_int(plan.get("mc_port")) or 25565),
        # Absolute: the detached driver does not share this shell's cwd.
        "--warm-start", str(warm.get("path") or ""),
        # T11b's gate. Without it a stale path or a half-copied file loads
        # cleanly and becomes snapshot 0 — a PINNED reference for the whole run.
        "--warm-start-sha256", str(warm.get("sha256") or ""),
        "--run-name", str(plan.get("run_name") or EXPECTED_RUN_NAME),
        # NEVER --best-checkpoint alone: only this hook drives the periodic AND
        # the final save.
        "--checkpoint", str(plan.get("checkpoint") or ""),
        "--best-checkpoint", str(plan.get("best_checkpoint") or ""),
        # The number this whole file exists to compute.
        "--eps-decay-episodes", str(_int(sizing.get("eps_decay_episodes")) or 1),
        "--eval-every-grad-steps", str(_int(sizing.get("eval_every_grad_steps")) or 0),
        "--eval-episodes", str(_int(sizing.get("eval_episodes")) or 0),
        "--reference-eval-episodes",
        str(_int(sizing.get("reference_eval_episodes")) or 0),
        "--checkpoint-every-grad-steps",
        str(_int(sizing.get("checkpoint_every_grad_steps")) or 0),
        # Both budgets, both sized to the window: the learner stops on whichever
        # binds first and takes the collectors with it.
        "--max-episodes", str(_int(sizing.get("max_episodes")) or 0),
        "--max-grad-steps", str(_int(sizing.get("max_grad_steps")) or 0),
        "--snapshot-sampling", "pfsp",
        "--seed", str(_int(plan.get("seed")) or 0),
        # Explicit, not `auto`: wandb/tensorboard are not installed here, and the
        # morning comparison reads runs/<run>/metrics.jsonl and summary.json.
        "--log-backend", "jsonl",
    ]
    snapshot_every = _int(sizing.get("snapshot_every_grad_steps"))
    if snapshot_every is not None:
        argv.extend(["--snapshot-every-grad-steps", str(snapshot_every)])
    return argv


def format_launch_command(plan: Mapping[str, Any], argv: Sequence[str]) -> str:
    """Render the detached launch exactly as the operator would type it.

    ``nohup`` + ``&`` + a pid file is this project's overnight convention (the
    M3 run recorded it: "PID in runs/m3.pid, log runs/m3.log. Launched detached
    with nohup so it survives the session that started it").
    """
    python_bin = str(plan.get("python_bin") or "python")
    log_path = str(plan.get("log_path") or "")
    pid_path = str(plan.get("pid_path") or "")
    repo_root = str(plan.get("repo_root") or ".")
    parts = [f"cd {shlex.quote(repo_root)} && nohup {shlex.quote(python_bin)} -m agent.train \\"]
    index = 0
    while index < len(argv):
        flag = argv[index]
        if index + 1 < len(argv) and not str(argv[index + 1]).startswith("--"):
            parts.append(f"  {flag} {shlex.quote(str(argv[index + 1]))} \\")
            index += 2
        else:
            parts.append(f"  {flag} \\")
            index += 1
    parts.append(f"  > {shlex.quote(log_path)} 2>&1 &")
    parts.append(f"echo $! > {shlex.quote(pid_path)}")
    return "\n".join(parts)


def format_launch_report(verdict: Verdict, plan: Mapping[str, Any]) -> str:
    """The operator-facing report: the arithmetic, every check, then the command."""
    rule = "=" * 78
    sizing = dict(verdict.facts.get("sizing") or {})
    canary = _section(plan, "canary")
    smoke = _section(plan, "smoke")
    lines = [rule, " T19 SELF-PLAY LAUNCH PLAN", rule]
    lines.append(f" run                {plan.get('run_name')}  ({plan.get('arenas')} pads)")
    lines.append(f" window             {plan.get('window_hours')} h  (an ASSUMPTION: a run length is a plan, not a measurement)")
    lines.append(f" canary measurement {canary.get('measurements_path')}")
    lines.append(f" smoke measurement  {smoke.get('measurements_path')}")
    lines.append("")
    lines.append(" SIZING --eps-decay-episodes FROM MEASUREMENT (never from a constant)")
    lines.extend(sizing_arithmetic_lines(sizing))
    lines.append("")
    lines.append(" DERIVED RUN SHAPE")
    lines.append(
        f"   learner rate                         {_num(sizing.get('grad_steps_per_hour')) or 0:.0f} grad steps/hour"
        f"   ({sizing.get('grad_steps_per_hour_source')})"
    )
    lines.append(
        f"   eval episode                         {_num(sizing.get('eval_episode_seconds')) or 0:.1f} s"
        f"   ({sizing.get('eval_length_source')})"
    )
    lines.append(
        f"   eval cycle, 1 reference              {_num(sizing.get('eval_cycle_minutes_first')) or 0:.0f} min"
    )
    lines.append(
        f"   eval cycle, 3 references             {_num(sizing.get('eval_cycle_minutes_worst')) or 0:.0f} min"
        f"   <- the one that runs most of the night"
    )
    lines.append(
        f"   snapshot pool by morning             ~{_num(sizing.get('projected_pool_size')) or 0:.0f} snapshots"
        f" (~{_num(sizing.get('projected_pool_megabytes')) or 0:.0f} MB)"
    )
    lines.append(
        f"   periodic checkpoints                 ~{_num(sizing.get('periodic_checkpoints_in_window')) or 0:.0f}"
    )
    lines.append("")
    lines.append(" CHECKS")
    for check in verdict.checks:
        marker = "ok    " if check.passed else "REFUSE"
        lines.append(f"   [{marker}] {check.code:<28} {check.detail}")

    if verdict.refusals:
        lines.append("")
        lines.append(" REFUSALS - the 24-hour run must NOT start")
        for check in verdict.refusals:
            lines.append("")
            lines.append(f"   {check.code}")
            lines.append(f"     WHY:   {check.why}")
            lines.append(f"     CHECK: {check.check}")
        lines.append("")
        lines.append(f" VERDICT: REFUSED - {len(verdict.refusals)} condition(s) block the launch.")
        lines.append(rule)
        return "\n".join(lines)

    lines.append("")
    lines.append(" BOOT ORDER - Paper -> bridges -> driver. Steps 1+2 are the operator's:")
    lines.append(f"   {verdict.facts.get('fleet_boot_command')}")
    lines.append("")
    lines.append(" THE COMMAND (this is what `launch` runs):")
    for line in str(verdict.facts.get("command") or "").splitlines():
        lines.append(f"   {line}")
    lines.append("")
    lines.append(" VERDICT: CLEARED - every condition holds.")
    lines.append(rule)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# PART 1 — the bounded 25-pad smoke.
#
# The canary proved the self-play WIRING at 4 pads with a lowered replay floor.
# This proves the fleet HOLDS at the width and the settings the night will use,
# with a second frozen DRQN in every collector and a snapshot read off disk at
# the head of every episode. Five quantities, five refusals.
# ---------------------------------------------------------------------------


def _percentile(values: Sequence[float], q: float) -> Optional[float]:
    """Linear-interpolated percentile of a small sample, or None when empty.

    ``statistics.quantiles`` needs at least two points and cuts at fixed
    fractions; a snapshot-load probe legitimately produces one measurement per
    snapshot, so this handles n == 1.
    """
    clean = sorted(float(v) for v in values if _num(v) is not None)
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    position = (len(clean) - 1) * max(0.0, min(1.0, q))
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return clean[lower]
    return clean[lower] + (clean[upper] - clean[lower]) * (position - lower)


def _rss_growth_bytes_per_second(samples: Sequence[Sequence[float]]) -> Optional[float]:
    """Least-squares slope of RSS against time, in bytes/second.

    Least squares rather than (last - first) / elapsed because the replay buffer
    fills roughly linearly while a single late sample can land on a garbage
    collection and halve or double a two-point estimate.
    """
    points = [
        (float(t), float(rss))
        for t, rss in (
            (pair[0], pair[1]) for pair in samples if isinstance(pair, (list, tuple)) and len(pair) >= 2
        )
        if _num(t) is not None and _num(rss) is not None
    ]
    if len(points) < 3:
        return None
    n = float(len(points))
    mean_t = sum(t for t, _ in points) / n
    mean_r = sum(r for _, r in points) / n
    denominator = sum((t - mean_t) ** 2 for t, _ in points)
    if denominator <= 0.0:
        return None
    numerator = sum((t - mean_t) * (r - mean_r) for t, r in points)
    return numerator / denominator


def build_smoke_measurements(evidence: Mapping[str, Any]) -> Dict[str, Any]:
    """Every number the smoke measured, plus how each one was obtained.

    HONEST DERIVATION NOTE, because one of these five is not measured directly:
    the multi-arena path logs no replay size and no queue length, so a
    transitions counter is not observable from outside the driver process
    (adding one would mean editing ``agent/train.py``, which T19 does not own).
    ``transitions_per_s`` is therefore ``episodes/s x the canary's measured
    armored episode length``, and the episode length it used is recorded
    alongside it. ``episodes_per_arena_hour`` and ``grad_steps_per_hour`` ARE
    direct: both come from the driver's own ``[multi done]`` line over a
    wall-clock the shell measured.
    """
    driver = _section(evidence, "driver")
    pool = _section(evidence, "pool")
    load = _section(evidence, "snapshot_load")
    rss = _section(evidence, "rss")

    episodes = _int(driver.get("episodes"))
    grad_steps = _int(driver.get("grad_steps"))
    wall = _num(evidence.get("wall_seconds"))
    arenas = _int(evidence.get("arenas"))
    length = _num(evidence.get("episode_length_steps"))

    out: Dict[str, Any] = {
        "arenas": arenas,
        "run_name": evidence.get("run_name"),
        "min_replay": _int(evidence.get("min_replay")),
        "wall_seconds": wall,
        "episodes": episodes,
        "grad_steps": grad_steps,
        "stop_reason": driver.get("stop_reason"),
        "episode_length_steps": length,
        "episode_length_source": evidence.get("episode_length_source"),
    }
    hours = wall / 3600.0 if wall and wall > 0.0 else None
    out["grad_steps_per_hour"] = (
        grad_steps / hours if grad_steps is not None and hours else None
    )
    out["episodes_per_hour"] = (
        episodes / hours if episodes is not None and hours else None
    )
    out["episodes_per_arena_hour"] = (
        out["episodes_per_hour"] / arenas
        if out["episodes_per_hour"] is not None and arenas
        else None
    )
    out["transitions_per_s"] = (
        episodes * length / wall
        if episodes is not None and length and wall and wall > 0.0
        else None
    )
    out["transitions_per_arena_s"] = (
        out["transitions_per_s"] / arenas
        if out["transitions_per_s"] is not None and arenas
        else None
    )
    out["transitions_collected"] = (
        episodes * length if episodes is not None and length else None
    )
    # The queue-depth proxy. The learner takes exactly ONE gradient step per
    # drain pass and pulls at most LEARNER_DRAIN_BATCH episodes per pass, so a
    # ratio approaching the batch size means every pass is hitting the cap and
    # the backlog is not draining. A healthy run sits near 1.
    out["episodes_per_grad_step"] = (
        float(episodes) / float(grad_steps)
        if episodes is not None and grad_steps
        else None
    )
    out["learner_drain_batch"] = LEARNER_DRAIN_BATCH
    out["watchdog_tripped"] = bool(evidence.get("watchdog_tripped"))

    out["rss_peak_bytes"] = _num(rss.get("peak_bytes"))
    out["rss_first_bytes"] = _num(rss.get("first_bytes"))
    out["jvm_peak_bytes"] = _num(rss.get("jvm_peak_bytes"))
    out["rss_samples"] = len(list(rss.get("samples") or []))
    slope = _rss_growth_bytes_per_second(list(rss.get("samples") or []))
    out["rss_growth_bytes_per_s"] = slope
    out["physical_memory_bytes"] = _num(evidence.get("physical_memory_bytes"))

    # Project the peak forward over whatever remains of the night, but only
    # until the replay buffer is FULL: past capacity it stops growing, so
    # extrapolating a 12-hour straight line would refuse every healthy run.
    capacity = _num(evidence.get("replay_capacity"))
    window_seconds = (_num(evidence.get("window_hours")) or 0.0) * 3600.0
    remaining = max(0.0, window_seconds - (wall or 0.0))
    if capacity and out["transitions_per_s"] and out["transitions_collected"] is not None:
        seconds_to_full = max(
            0.0, (capacity - out["transitions_collected"]) / out["transitions_per_s"]
        )
    else:
        seconds_to_full = None
    horizon = (
        min(remaining, seconds_to_full) if seconds_to_full is not None else remaining
    )
    out["rss_growth_horizon_seconds"] = horizon
    out["projected_rss_bytes"] = (
        out["rss_peak_bytes"] + max(0.0, slope) * horizon + (out["jvm_peak_bytes"] or 0.0)
        if out["rss_peak_bytes"] is not None and slope is not None
        else None
    )
    out["projected_rss_fraction_of_ram"] = (
        out["projected_rss_bytes"] / out["physical_memory_bytes"]
        if out["projected_rss_bytes"] and out["physical_memory_bytes"]
        else None
    )

    seconds = [s for s in (load.get("seconds") or []) if _num(s) is not None]
    out["snapshot_load_ok"] = bool(load.get("ok"))
    out["snapshot_load_samples"] = len(seconds)
    out["snapshot_load_p95_seconds"] = _percentile(seconds, 0.95)
    out["snapshot_load_max_seconds"] = max(seconds) if seconds else None
    # One arena's mean episode WALL time. The snapshot read happens once per
    # episode on the collector's own thread, so this ratio is the share of
    # collection time it costs.
    out["episode_wall_seconds_per_arena"] = (
        length / MEASURED_PER_ARENA_TRANSITIONS_PER_S if length else None
    )
    out["snapshot_load_duty"] = (
        out["snapshot_load_p95_seconds"] / out["episode_wall_seconds_per_arena"]
        if out["snapshot_load_p95_seconds"] is not None
        and out["episode_wall_seconds_per_arena"]
        else None
    )
    out["pool_size"] = _int(pool.get("size"))
    out["pool_snapshot_ids"] = list(pool.get("snapshot_ids") or [])
    out["notes"] = [
        "transitions_per_s is DERIVED (episodes/s x the armored episode length "
        "named in episode_length_source): the multi-arena path logs no replay "
        "size and no queue length, so a transitions counter is not observable "
        "from outside the driver process.",
        "grad_steps_per_hour and episodes_per_arena_hour are direct, from the "
        "driver's own [multi done] line over a shell-measured wall clock.",
        "episodes_per_grad_step is the queue-depth proxy: the learner takes one "
        f"gradient step per drain pass and drains at most {LEARNER_DRAIN_BATCH} "
        "episodes per pass, so a ratio near that number means the backlog is not "
        "draining.",
        "projected_rss_bytes extrapolates the measured growth only until the "
        "replay buffer is full; past capacity it stops growing.",
    ]
    return out


def evaluate_smoke(
    evidence: Mapping[str, Any], thresholds: Optional[Mapping[str, Any]] = None
) -> Verdict:
    """Judge the 25-pad smoke. Every check fail-closed on absent evidence."""
    version = _int(evidence.get("smoke_version"))
    if version != SMOKE_EVIDENCE_VERSION:
        return Verdict(
            checks=[
                _refuse(
                    "SMOKE_EVIDENCE_VERSION",
                    f"smoke_version={evidence.get('smoke_version')!r}, expected "
                    f"{SMOKE_EVIDENCE_VERSION}",
                    "this evidence was written by a different revision of the "
                    "collector, so the fields below may mean something else.",
                    "re-run the smoke so the collector and the gate come from the "
                    "same file.",
                )
            ],
            facts={},
        )
    limits = _thresholds(thresholds)
    measured = build_smoke_measurements(evidence)
    driver = _section(evidence, "driver")
    checks: List[CheckResult] = []

    # -- the driver actually finished ----------------------------------------
    # The EXIT CODE is not the signal: `_main_multi_arena` returns
    # `0 if passed_m2 else 1`, and passed_m2 is the M2 gate against the
    # STATIONARY dummy, which a self-play run never clears. A healthy smoke
    # exits 1. The `[multi done]` teardown line is the real completion signal.
    if not driver.get("completed"):
        checks.append(
            _refuse(
                "SMOKE_DRIVER_FAILED",
                f"no [multi done] line (exit code {driver.get('exit_code')!r})",
                "the driver crashed, was killed, or was cut off before its "
                "teardown. Every number below would then describe a partial run, "
                "and a partial run's throughput is not the fleet's throughput.",
                f"read the tail of {driver.get('log_path')}; a config ValueError "
                "lands in the first second, a bridge fault later.",
            )
        )
    else:
        checks.append(
            _ok(
                "SMOKE_DRIVER_FAILED",
                f"[multi done] reason={driver.get('stop_reason')} "
                f"episodes={driver.get('episodes')} grad_steps={driver.get('grad_steps')}",
            )
        )

    # -- width ---------------------------------------------------------------
    arenas = _num(measured.get("arenas"))
    if arenas is None or arenas < limits["smoke_min_pads"]:
        checks.append(
            _refuse(
                "SMOKE_NOT_FULL_WIDTH",
                f"ran at {measured.get('arenas')!r} pads, needs "
                f"{limits['smoke_min_pads']:.0f}",
                "a narrower smoke measures a different machine load. 25 pads is "
                "where the CPU, the JVM's entity count and the per-collector "
                "second DRQN all land together, and it is the only width the "
                "night will run at.",
                "re-run with --arenas 25 on a 25-pad fleet.",
            )
        )
    else:
        checks.append(_ok("SMOKE_NOT_FULL_WIDTH", f"{arenas:.0f} pads"))

    # -- long enough to mean anything ----------------------------------------
    wall = _num(measured.get("wall_seconds"))
    if wall is None or wall < limits["min_smoke_wall_seconds"]:
        checks.append(
            _refuse(
                "SMOKE_TOO_SHORT",
                f"{wall!r} s of wall clock, needs {limits['min_smoke_wall_seconds']:.0f} s",
                "the 25,000-transition replay warm-up alone takes about 205 s at "
                "the measured rate, during which the learner takes NO gradient "
                "step. A smoke shorter than that measures the warm-up, not the "
                "steady state.",
                "raise --smoke-grad-steps or --smoke-minutes.",
            )
        )
    else:
        checks.append(_ok("SMOKE_TOO_SHORT", f"{wall:.0f} s of wall clock"))

    # -- it learned ----------------------------------------------------------
    grad_steps = _int(measured.get("grad_steps"))
    if grad_steps is None or grad_steps < limits["min_smoke_grad_steps"]:
        checks.append(
            _refuse(
                "SMOKE_ZERO_GRAD_STEPS",
                f"{grad_steps!r} gradient steps, needs "
                f"{limits['min_smoke_grad_steps']:.0f}",
                "the run collected and never learned, so no weights were "
                "published, no snapshot was archived from a moving net, and the "
                "per-episode snapshot load this smoke exists to measure was "
                "reading one unchanging file.",
                "check --min-replay against the wall clock: below the floor, "
                "Trainer.learn() is a no-op for every second of the run.",
            )
        )
    else:
        checks.append(_ok("SMOKE_ZERO_GRAD_STEPS", f"{grad_steps} gradient steps"))

    # -- grad steps/hour -----------------------------------------------------
    rate = _num(measured.get("grad_steps_per_hour"))
    floor_rate = M3_RETRY_GRAD_STEPS_PER_HOUR * limits["min_grad_steps_per_hour_fraction"]
    if rate is None:
        checks.append(
            _refuse(
                "SMOKE_GRAD_STEPS_LOW",
                "the learner rate could not be computed",
                "without it the night's grad-step budget is a guess and the eval "
                "cadence cannot be sized.",
                "check that the driver printed [multi done] and that the wall "
                "clock was recorded.",
            )
        )
    elif rate < floor_rate:
        checks.append(
            _refuse(
                "SMOKE_GRAD_STEPS_LOW",
                f"{rate:.0f} grad steps/hour is below {floor_rate:.0f} "
                f"({limits['min_grad_steps_per_hour_fraction']:.0%} of the M3 "
                f"retry's {M3_RETRY_GRAD_STEPS_PER_HOUR:.0f})",
                "the learner is starved or contended. Self-play puts a second "
                "frozen DRQN in every collector, all on CPU, competing with the "
                "learner for the same cores — if that is the cause, the night "
                "produces a fraction of the gradient steps the budget assumes.",
                "check the queue-backlog and snapshot-load checks below; if both "
                "are clean the contention is CPU, and fewer pads buys more "
                "learning.",
            )
        )
    else:
        checks.append(
            _ok(
                "SMOKE_GRAD_STEPS_LOW",
                f"{rate:.0f} grad steps/hour (M3 retry: "
                f"{M3_RETRY_GRAD_STEPS_PER_HOUR:.0f}, bare-handed)",
            )
        )

    # -- transitions/s -------------------------------------------------------
    transitions = _num(measured.get("transitions_per_s"))
    floor_transitions = (
        MEASURED_AGGREGATE_TRANSITIONS_PER_S * limits["min_transitions_per_s_fraction"]
    )
    if transitions is None:
        checks.append(
            _refuse(
                "SMOKE_TRANSITIONS_LOW",
                "collection rate could not be derived",
                "the episode count, the wall clock or the armored episode length "
                "was missing, so there is no throughput figure — and every "
                "episode-count projection for the night runs through it.",
                "check that the canary measurement supplied an episode length "
                "and that the driver completed.",
            )
        )
    elif transitions < floor_transitions:
        checks.append(
            _refuse(
                "SMOKE_TRANSITIONS_LOW",
                f"{transitions:.1f} transitions/s is below {floor_transitions:.1f} "
                f"({limits['min_transitions_per_s_fraction']:.0%} of the measured "
                f"{MEASURED_AGGREGATE_TRANSITIONS_PER_S:.2f})",
                "the fleet does not hold at full width under self-play. Every "
                "episode count in the sizing above is linear in this number, so "
                "a shortfall here means the epsilon window, the eval cadence and "
                "both budgets are all wrong in the same direction.",
                "compare transitions_per_arena_s against 4.8782: a per-arena drop "
                "is CPU contention from the second DRQN, while a full-fleet drop "
                "with a healthy per-arena rate means pads are missing.",
            )
        )
    else:
        checks.append(
            _ok(
                "SMOKE_TRANSITIONS_LOW",
                f"{transitions:.1f} transitions/s "
                f"({_num(measured.get('transitions_per_arena_s')) or 0:.2f}/arena vs the "
                f"measured {MEASURED_PER_ARENA_TRANSITIONS_PER_S})",
            )
        )

    # -- queue depth ---------------------------------------------------------
    ratio = _num(measured.get("episodes_per_grad_step"))
    ratio_ceiling = LEARNER_DRAIN_BATCH * limits["max_episodes_per_grad_step_fraction"]
    if measured.get("watchdog_tripped"):
        checks.append(
            _refuse(
                "SMOKE_QUEUE_BACKLOG",
                "the learner watchdog tripped",
                "the transport backlog kept growing while grad_step did not "
                "advance, with the replay buffer WARM. That is a wedged learner, "
                "and it aborts the run rather than collecting into a buffer "
                "nobody drains.",
                "search the driver log for WatchdogError and '[learner] "
                "aborting'.",
            )
        )
    elif ratio is None:
        checks.append(
            _refuse(
                "SMOKE_QUEUE_BACKLOG",
                "episodes-per-grad-step could not be computed",
                "this ratio is the only queue-depth signal observable from "
                "outside the driver process, so without it nothing rules out a "
                "backlog that grows all night into an unbounded queue.",
                "check the [multi done] line for both episodes and grad_steps.",
            )
        )
    elif ratio >= ratio_ceiling:
        checks.append(
            _refuse(
                "SMOKE_QUEUE_BACKLOG",
                f"{ratio:.1f} episodes per gradient step, at or above "
                f"{ratio_ceiling:.1f} ({limits['max_episodes_per_grad_step_fraction']:.0%} "
                f"of the {LEARNER_DRAIN_BATCH}-episode drain batch)",
                "the learner is draining a full batch on every pass, which means "
                "it is behind the collectors. `collector_queue_max` defaults to 0 "
                "— the queue is UNBOUNDED, so there is no backpressure: the "
                "backlog and its per-step LSTM hidden states grow in RAM until "
                "the night ends or the machine does.",
                "compare grad_steps_per_hour above; a slow learner with a fast "
                "fleet is the shape of this. Fewer pads or a smaller batch.",
            )
        )
    else:
        checks.append(
            _ok(
                "SMOKE_QUEUE_BACKLOG",
                f"{ratio:.2f} episodes per gradient step (drain batch is "
                f"{LEARNER_DRAIN_BATCH}; a keeping-up learner sits near 1)",
            )
        )

    # -- RSS -----------------------------------------------------------------
    projected_fraction = _num(measured.get("projected_rss_fraction_of_ram"))
    if projected_fraction is None:
        checks.append(
            _refuse(
                "SMOKE_RSS_PROJECTION",
                f"{measured.get('rss_samples')} RSS sample(s), "
                f"ram={measured.get('physical_memory_bytes')!r}",
                "the driver's memory growth was not measured, so nothing rules "
                "out an OOM kill at 4am — which leaves no [multi done] line, no "
                "final save, and only whatever the periodic hook already wrote.",
                "check that ps sampled the driver pid and that sysctl reported "
                "hw.memsize.",
            )
        )
    elif projected_fraction > limits["max_rss_fraction_of_ram"]:
        checks.append(
            _refuse(
                "SMOKE_RSS_PROJECTION",
                f"projected peak {(_num(measured.get('projected_rss_bytes')) or 0) / 1e9:.1f} GB "
                f"== {projected_fraction:.0%} of RAM "
                f"(ceiling {limits['max_rss_fraction_of_ram']:.0%})",
                "extrapolating the measured growth to a full replay buffer puts "
                "the driver plus the JVM past a safe share of physical memory. "
                "The per-step LSTM hidden snapshot is ~90% of a stored "
                "transition, so the replay buffer is the term that grows.",
                "lower --replay-capacity, or bound the collector queue; the "
                "growth measured here was "
                f"{(_num(measured.get('rss_growth_bytes_per_s')) or 0) / 1e6:.1f} MB/s.",
            )
        )
    else:
        checks.append(
            _ok(
                "SMOKE_RSS_PROJECTION",
                f"peak {(_num(measured.get('rss_peak_bytes')) or 0) / 1e9:.1f} GB, "
                f"projected {(_num(measured.get('projected_rss_bytes')) or 0) / 1e9:.1f} GB "
                f"== {projected_fraction:.0%} of RAM",
            )
        )

    # -- snapshot-load latency -----------------------------------------------
    duty = _num(measured.get("snapshot_load_duty"))
    if not measured.get("snapshot_load_ok") or duty is None:
        checks.append(
            _refuse(
                "SMOKE_SNAPSHOT_LOAD_SLOW",
                f"snapshot-load probe ok={measured.get('snapshot_load_ok')!r} "
                f"samples={measured.get('snapshot_load_samples')!r}",
                "the per-episode snapshot read was never timed. It happens on "
                "the collector's own thread at the head of every episode, so if "
                "it is slow it is time taken directly out of collection — and "
                "this smoke is the only thing that measures it.",
                "check the snapshot_load section of the evidence for the probe's "
                "error.",
            )
        )
    elif duty > limits["max_snapshot_load_duty"]:
        checks.append(
            _refuse(
                "SMOKE_SNAPSHOT_LOAD_SLOW",
                f"p95 load {(_num(measured.get('snapshot_load_p95_seconds')) or 0):.3f} s "
                f"is {duty:.1%} of a {(_num(measured.get('episode_wall_seconds_per_arena')) or 0):.1f} s "
                f"episode (ceiling {limits['max_snapshot_load_duty']:.0%})",
                "every collector pays this once per episode, before its first "
                "decision. At this share it is a standing tax on collection that "
                "grows with the pool: 25 arenas reading a ~2.4 MB snapshot each "
                "time an episode starts.",
                "check disk contention and the pool size; a pool that grew far "
                "past the projection makes each read no slower but every eval "
                "cycle longer.",
            )
        )
    else:
        checks.append(
            _ok(
                "SMOKE_SNAPSHOT_LOAD_SLOW",
                f"p95 {(_num(measured.get('snapshot_load_p95_seconds')) or 0):.3f} s "
                f"== {duty:.2%} of one episode's wall time",
            )
        )

    # -- the pool grew -------------------------------------------------------
    pool_size = _num(measured.get("pool_size"))
    if pool_size is None or pool_size < limits["min_smoke_pool_size"]:
        checks.append(
            _refuse(
                "SMOKE_POOL_NOT_GROWING",
                f"pool holds {measured.get('pool_size')!r} snapshot(s)",
                "the pool never grew past the seeded snapshot 0, so every "
                "collector spent the smoke reloading ONE file and the "
                "per-episode load latency measured above is the best case, not "
                "the real one. It also means the archive cadence is not firing "
                "at this width.",
                "check --snapshot-every-grad-steps against the grad steps the "
                "smoke actually took.",
            )
        )
    else:
        checks.append(
            _ok(
                "SMOKE_POOL_NOT_GROWING",
                f"pool holds {pool_size:.0f} snapshots {measured.get('pool_snapshot_ids')}",
            )
        )

    return Verdict(checks=checks, facts={"measurements": measured})


def format_smoke_report(verdict: Verdict, evidence: Mapping[str, Any]) -> str:
    """The operator-facing smoke report."""
    rule = "=" * 78
    measured = dict(verdict.facts.get("measurements") or {})
    lines = [rule, " T19 25-PAD SMOKE - VERDICT", rule]
    lines.append(
        f" run                {evidence.get('run_name')}  "
        f"({evidence.get('arenas')} pads, --min-replay {evidence.get('min_replay')} "
        "== the PRODUCTION value)"
    )
    lines.append(f" evidence           {evidence.get('evidence_path')}")
    lines.append("")
    lines.append(" MEASURED")
    for key in (
        "wall_seconds",
        "episodes",
        "grad_steps",
        "grad_steps_per_hour",
        "transitions_per_s",
        "transitions_per_arena_s",
        "episodes_per_arena_hour",
        "episodes_per_grad_step",
        "rss_peak_bytes",
        "rss_growth_bytes_per_s",
        "projected_rss_bytes",
        "projected_rss_fraction_of_ram",
        "snapshot_load_p95_seconds",
        "snapshot_load_duty",
        "pool_size",
        "episode_length_steps",
        "episode_length_source",
    ):
        value = measured.get(key)
        rendered = f"{value:.4g}" if isinstance(value, float) else str(value)
        lines.append(f"    {key:<34} {rendered}")
    lines.append("")
    lines.append(" CHECKS")
    for check in verdict.checks:
        marker = "ok    " if check.passed else "REFUSE"
        lines.append(f"   [{marker}] {check.code:<28} {check.detail}")
    if verdict.refusals:
        lines.append("")
        lines.append(" REFUSALS - the 24-hour run must NOT start")
        for check in verdict.refusals:
            lines.append("")
            lines.append(f"   {check.code}")
            lines.append(f"     WHY:   {check.why}")
            lines.append(f"     CHECK: {check.check}")
    lines.append("")
    if verdict.ok:
        lines.append(" VERDICT: GREEN - the fleet holds at full width.")
    else:
        lines.append(
            f" VERDICT: REFUSED - {len(verdict.refusals)} condition(s) block the launch."
        )
    lines.append(rule)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# PART 4 — the morning checkpoint comparison.
#
# The operator picks the demo net from this table. Its job is NOT to rank: it is
# to put every candidate's number next to WHAT that number was earned against,
# because the two most attractive numbers in this project's history are both
# misleading, in different ways.
# ---------------------------------------------------------------------------

COMPARE_VERSION = 1

#: Every checkpoint from a bare-handed run was trained and scored with no armor
#: on either fighter AND the opponent holding nothing — `dummyResetTemplate`
#: carried `inventory: []` until T3. An unarmed opponent punches for ~0.42
#: through full iron, ~48 hits to a kill.
CAUTION_BARE_REGIME = (
    "BARE-HANDED REGIME: trained and scored with no armor on either fighter and "
    "an UNARMED opponent. Not comparable to an armored number, and mis-calibrated "
    "for an armored demo."
)

#: `_BestCheckpointSelector.consider` keeps a candidate only when it scores
#: STRICTLY higher than the incumbent, so ties keep the EARLIER net. A run that
#: reaches its ceiling early never revisits it, however much better the policy
#: gets afterwards.
CAUTION_SELECTOR_FIRST = (
    "SELECTOR KEEPS THE FIRST, NOT THE BEST: _BestCheckpointSelector.consider "
    "requires a STRICTLY higher score, so ties keep the earlier net. This file "
    "holds the first net to reach its win rate, not the strongest net the run "
    "produced."
)

#: The specific trap named in the plan: `runs/m4.best.pt` scored win_rate=1.000
#: at grad_step 8307 while the fully-trained net at 30,000 scored 0.850.
CAUTION_PERFECT_VS_UNARMED = (
    "A PERFECT SCORE AGAINST A WEAPONLESS OPPONENT is not evidence of skill "
    "against an armed one. It says the opponent could not fight back."
)

CAUTION_NO_OPPONENT_NAMED = (
    "NO OPPONENT RECORDED for this win rate. A rate without an opponent is not a "
    "ranking and must not be read as one."
)

CAUTION_ELO_SCOPE = (
    "ELO IS POOL-LOCAL: a rating is only comparable to other ratings from the "
    "SAME run's pool, because only the learner's rating moves and snapshots are "
    "frozen at the learner's rating when they were archived."
)

CAUTION_UNEVALUATED = (
    "NEVER EVALUATED under this run's gauntlet: it has no armored number at all. "
    "Absence of a number is not a low number, and it is not a high one either."
)


def cautions_for(candidate: Mapping[str, Any]) -> List[str]:
    """Every caution that applies to one candidate, in reading order.

    Pure and table-driven so the tests can assert that the two traps the plan
    names — the weaponless perfect score and the first-not-best selector — are
    attached to the exact files that carry them.
    """
    notes: List[str] = []
    run = str(candidate.get("run") or "")
    path = str(candidate.get("path") or "")
    bare = run in BARE_HANDED_RUNS
    win_rate = _num(candidate.get("win_rate"))
    scripted = _num(candidate.get("scripted_win_rate"))
    headline = win_rate if win_rate is not None else scripted

    if bare:
        notes.append(CAUTION_BARE_REGIME)
    if path.endswith(".best.pt"):
        notes.append(CAUTION_SELECTOR_FIRST)
        if bare and headline is not None and headline >= 0.999:
            grad_step = candidate.get("grad_step")
            notes.append(
                CAUTION_PERFECT_VS_UNARMED
                + f" (this one: {headline:.3f} at grad_step {grad_step})"
            )
    if headline is not None and not candidate.get("eval_opponent"):
        notes.append(CAUTION_NO_OPPONENT_NAMED)
    if _num(candidate.get("rated_elo")) is not None or _num(candidate.get("elo")) is not None:
        notes.append(CAUTION_ELO_SCOPE)
    if headline is None and _num(candidate.get("reference_aggregate")) is None:
        notes.append(CAUTION_UNEVALUATED)
    return notes


def compare_rows(document: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Normalize every candidate into one row, with its cautions attached."""
    rows: List[Dict[str, Any]] = []
    for candidate in document.get("candidates") or []:
        if not isinstance(candidate, Mapping):
            continue
        run = str(candidate.get("run") or "")
        row: Dict[str, Any] = {
            "label": str(candidate.get("label") or candidate.get("path") or "?"),
            "path": candidate.get("path"),
            "exists": bool(candidate.get("exists")),
            "run": run,
            "kind": candidate.get("kind"),
            "grad_step": _int(candidate.get("grad_step")),
            "scripted_win_rate": _num(candidate.get("scripted_win_rate")),
            "reference_aggregate": _num(candidate.get("reference_aggregate")),
            "reference_worst": _num(candidate.get("reference_worst")),
            "references_evaluated": _int(candidate.get("references_evaluated")),
            "rated_elo": _num(candidate.get("rated_elo")),
            "snapshot_elo": _num(candidate.get("elo")),
            "eval_opponent": candidate.get("eval_opponent"),
            "regime": (
                "bare-handed" if run in BARE_HANDED_RUNS else "armored self-play"
            ),
            # Where each number came from, verbatim from the collector. A win
            # rate attributed to a checkpoint by proximity is not the same
            # evidence as one stamped into that checkpoint by the save hook,
            # and the morning read has to be able to tell them apart.
            "source": candidate.get("source"),
            "error": candidate.get("error"),
        }
        row["cautions"] = cautions_for(candidate)
        rows.append(row)
    return rows


def compare_candidates(document: Mapping[str, Any]) -> str:
    """Render the morning table. Deliberately produces NO overall ranking.

    Sorting by a win rate would put the weaponless 1.000 at the top of a table
    whose whole purpose is to explain why that number is not a ranking. Rows are
    grouped by regime and ordered by grad step within each group, which is the
    only ordering that means the same thing for every row.
    """
    rows = compare_rows(document)
    rule = "=" * 100
    lines = [rule, " M4 MORNING CHECKPOINT COMPARISON", rule]
    if not rows:
        lines.append(" no candidates found.")
        lines.append(rule)
        return "\n".join(lines)

    header = (
        f" {'CANDIDATE':<38} {'GRAD STEP':>9} {'SCRIPTED':>9} {'REF AGG':>8} "
        f"{'REF WORST':>9} {'RATED ELO':>9}"
    )

    def render(value: Optional[float], spec: str = ".3f") -> str:
        return "-" if value is None else f"{value:{spec}}"

    for regime in ("armored self-play", "bare-handed"):
        group = [row for row in rows if row["regime"] == regime]
        if not group:
            continue
        group.sort(key=lambda r: (r["grad_step"] is None, r["grad_step"] or 0))
        lines.append("")
        lines.append(f" {regime.upper()}")
        lines.append(header)
        lines.append(f" {'-' * 96}")
        for row in group:
            elo = row["rated_elo"] if row["rated_elo"] is not None else row["snapshot_elo"]
            missing = "" if row["exists"] else "   [MISSING ON DISK]"
            lines.append(
                f" {row['label'][:38]:<38} {row['grad_step'] if row['grad_step'] is not None else '-':>9} "
                f"{render(row['scripted_win_rate']):>9} {render(row['reference_aggregate']):>8} "
                f"{render(row['reference_worst']):>9} {render(elo, '.0f'):>9}{missing}"
            )
            if row["error"]:
                lines.append(f"   ! {row['error']}")

    lines.append("")
    lines.append(" WHAT EACH NUMBER IS NOT")
    for row in rows:
        # A row with no caution still gets its provenance printed: "where this
        # number came from" is part of what the number is not.
        if not (row["cautions"] or row["source"] or row["eval_opponent"]):
            continue
        lines.append("")
        lines.append(f"   {row['label']}")
        if row["eval_opponent"]:
            lines.append(f"     scored against: {row['eval_opponent']}")
        if row["source"]:
            lines.append(f"     numbers from:   {row['source']}")
        for note in row["cautions"]:
            lines.append(f"     - {note}")
    lines.append("")
    lines.append(
        " THIS TABLE DOES NOT RANK. The two regimes are not comparable: an "
        "armored fight is ~7 hits"
    )
    lines.append(
        " to a kill and a bare one ~4, and every bare-handed number was earned "
        "against an opponent"
    )
    lines.append(
        " holding NOTHING. Pick with the cautions in hand, and prefer a net "
        "evaluated in the regime"
    )
    lines.append(" the demo is fought in.")
    lines.append(rule)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI. One entry point per phase; the shell supplies documents and reads exit
# codes, and decides nothing itself.
# ---------------------------------------------------------------------------


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: str, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``launch_sizing.py <smoke-verdict|plan|compare> <document> [outputs...]``.

    Exits 0 when nothing blocks, 1 on any refusal, 2 on usage.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) < 2:
        print(
            "usage: launch_sizing.py smoke-verdict <evidence.json> [measurements.json]\n"
            "       launch_sizing.py plan <plan_input.json> [plan.json] [argv.txt]\n"
            "       launch_sizing.py compare <compare_input.json>",
            file=sys.stderr,
        )
        return 2
    command, path = args[0], args[1]

    if command == "smoke-verdict":
        evidence = dict(_read_json(path))
        evidence.setdefault("evidence_path", path)
        verdict = evaluate_smoke(evidence, evidence.get("thresholds"))
        if len(args) > 2:
            _write_json(args[2], verdict.facts.get("measurements") or {})
        print(format_smoke_report(verdict, evidence))
        return 0 if verdict.ok else 1

    if command == "plan":
        plan = dict(_read_json(path))
        verdict = evaluate_launch(plan)
        if len(args) > 2:
            _write_json(
                args[2],
                {
                    "ok": verdict.ok,
                    "refusals": [c.code for c in verdict.refusals],
                    "checks": [c._asdict() for c in verdict.checks],
                    "sizing": verdict.facts.get("sizing"),
                    "argv": verdict.facts.get("argv"),
                    "command": verdict.facts.get("command"),
                },
            )
        # The argv file is written ONLY on a clear verdict: a refused plan must
        # leave nothing behind that a later shell could pick up and run.
        if len(args) > 3:
            if verdict.ok:
                with open(args[3], "w", encoding="utf-8") as handle:
                    for item in verdict.facts.get("argv") or []:
                        handle.write(f"{item}\n")
            elif os.path.exists(args[3]):
                os.remove(args[3])
        print(format_launch_report(verdict, plan))
        return 0 if verdict.ok else 1

    if command == "compare":
        document = dict(_read_json(path))
        version = _int(document.get("compare_version"))
        if version != COMPARE_VERSION:
            print(
                f"compare_version={document.get('compare_version')!r}, expected "
                f"{COMPARE_VERSION}",
                file=sys.stderr,
            )
            return 2
        print(compare_candidates(document))
        return 0 if document.get("candidates") else 1

    print(f"unknown command: {command}", file=sys.stderr)
    return 2


if __name__ == "__main__":  # pragma: no cover - exercised by the shell script
    raise SystemExit(main())
LAUNCH_SIZING_PY
}

# ===========================================================================
# Shared preflight. Starts nothing; connects only to Paper.
# ===========================================================================

WARM_START_SHA256=""

sha256_file() {
    if command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | awk '{print $1}'
    else
        "${PYTHON_BIN}" - "$1" <<'LAUNCH_SHA_PY'
import hashlib
import sys

digest = hashlib.sha256()
with open(sys.argv[1], "rb") as handle:
    for chunk in iter(lambda: handle.read(1 << 20), b""):
        digest.update(chunk)
print(digest.hexdigest())
LAUNCH_SHA_PY
    fi
}

require_warm_start() {
    [[ -n "${WARM_START}" ]] || { usage >&2; die "--warm-start is REQUIRED.
      agent/train_config.py refuses --opponent selfplay without one (AC14): the
      snapshot pool's snapshot 0 IS that checkpoint."; }
    [[ -f "${WARM_START}" ]] || die "warm start not found: ${WARM_START}
      The bare-handed run's checkpoints live in the MAIN checkout's runs/, not
      in this worktree."
    case "${WARM_START}" in
        /*) : ;;
        *) die "--warm-start must be an ABSOLUTE path, got ${WARM_START}
      The driver is started detached and does not share this shell's working
      directory." ;;
    esac
    WARM_START_SHA256="$(sha256_file "${WARM_START}")"
    log "warm start ${WARM_START}"
    log "  sha256 ${WARM_START_SHA256}"
}

# inspect_fleet — fills FLEET_* with what `lsof` and one Paper connect can see.
# Sets no exit status of its own: the Python gate judges, this only reports.
FLEET_MC_REACHABLE="false"
FLEET_MISSING_PORTS=""
FLEET_BUSY_PORTS=""
FLEET_LISTENER_COUNT=0
FLEET_ETIMES=""
inspect_fleet() {
    FLEET_MISSING_PORTS=""
    FLEET_BUSY_PORTS=""
    FLEET_LISTENER_COUNT=0
    FLEET_ETIMES=""
    if mc_connect_probe "${HOST}" "${MC_PORT}"; then
        FLEET_MC_REACHABLE="true"
    else
        FLEET_MC_REACHABLE="false"
    fi
    local i port pids pid
    for (( i = 0; i < ARENAS; i++ )); do
        port=$(( BRIDGE_BASE_PORT + i ))
        pids="$(listener_pids "${port}")"
        if [[ -z "${pids// }" ]]; then
            FLEET_MISSING_PORTS="${FLEET_MISSING_PORTS} ${port}"
            continue
        fi
        FLEET_LISTENER_COUNT=$(( FLEET_LISTENER_COUNT + 1 ))
        if [[ "$(established_peers "${port}")" -gt 0 ]]; then
            FLEET_BUSY_PORTS="${FLEET_BUSY_PORTS} ${port}"
        fi
        for pid in ${pids}; do
            FLEET_ETIMES="${FLEET_ETIMES} $(proc_etime "${pid}")"
        done
    done
}

# run_canary_analyze — re-read T17's verdict over its own evidence. This is the
# canary's `--analyze-only` mode, which connects to nothing and starts nothing;
# reusing it is deliberate, because a second copy of that decision logic here
# could disagree with the gate the operator actually ran.
CANARY_ANALYZE_EXIT=""
run_canary_analyze() {
    CANARY_ANALYZE_EXIT=""
    if [[ ! -f "${CANARY_EVIDENCE}" ]]; then
        warn "no canary evidence at ${CANARY_EVIDENCE}; the verdict cannot be re-read."
        return 0
    fi
    if [[ ! -x "${CANARY_SCRIPT}" ]]; then
        warn "canary script not executable at ${CANARY_SCRIPT}."
        return 0
    fi
    log "re-reading the canary verdict (--analyze-only; connects to nothing)"
    set +e
    "${CANARY_SCRIPT}" --analyze-only "${CANARY_DIR}" --python "${PYTHON_BIN}" >"${OUT_DIR}/canary_verdict.txt" 2>&1
    CANARY_ANALYZE_EXIT=$?
    set -e
    log "canary --analyze-only exited ${CANARY_ANALYZE_EXIT} (verdict text in ${OUT_DIR}/canary_verdict.txt)"
}

mkdir -p "${OUT_DIR}" "${REPO_ROOT}/runs"
emit_sizing_module "${SIZING_MODULE}"

# ===========================================================================
# COMMAND: smoke — the bounded 25-pad run.
# ===========================================================================
cmd_smoke() {
    if [[ -n "${ANALYZE_ONLY}" ]]; then
        local evidence="${ANALYZE_ONLY}/smoke_evidence.json"
        [[ -f "${evidence}" ]] || die "no smoke_evidence.json in ${ANALYZE_ONLY}"
        set +e
        "${PYTHON_BIN}" "${SIZING_MODULE}" smoke-verdict "${evidence}" \
            "${ANALYZE_ONLY}/smoke_measurements.json"
        local exit_code=$?
        set -e
        exit "${exit_code}"
    fi

    log "phase 0: preflight"
    require_warm_start
    [[ "${SMOKE_RUN_NAME}" != "${RUN_NAME}" ]] || die \
        "--smoke-run-name must differ from --run-name (${RUN_NAME}).
      A smoke that writes into the long run's namespace seeds its snapshot pool
      with smoke snapshots and leaves a checkpoint the launch gate will then
      refuse as a pre-existing output."
    [[ -f "${CANARY_MEASUREMENTS}" ]] || die \
        "no canary measurement at ${CANARY_MEASUREMENTS}
      The smoke derives its transitions/s from the ARMORED episode length the
      canary measures, and refuses to substitute a constant. Run:
        scripts/canary_selfplay.sh --warm-start ${WARM_START} --arenas ${ARENAS}"
    if [[ -d "${REPO_ROOT}/runs/${SMOKE_RUN_NAME}/snapshots" ]]; then
        die "a snapshot pool already exists at runs/${SMOKE_RUN_NAME}/snapshots
      The smoke's pool-growth check would be measuring the PREVIOUS smoke's
      pool. Move or delete it first:
        rm -rf runs/${SMOKE_RUN_NAME} runs/${SMOKE_RUN_NAME}.pt runs/${SMOKE_RUN_NAME}.best.pt"
    fi

    inspect_fleet
    [[ "${FLEET_MC_REACHABLE}" == "true" ]] || die \
        "no Minecraft server on ${HOST}:${MC_PORT}.
      Boot order is Paper -> bridges -> driver and this script owns only the
      last step:
        DUMMY_KNOCKBACK_IMMUNE=false server/setup/start-pads.sh --pads ${ARENAS}"
    [[ -z "${FLEET_MISSING_PORTS// }" ]] || die \
        "no bridge listening on:${FLEET_MISSING_PORTS}
      Boot the fleet at the SAME width:
        DUMMY_KNOCKBACK_IMMUNE=false server/setup/start-pads.sh --pads ${ARENAS}"
    [[ -z "${FLEET_BUSY_PORTS// }" ]] || die \
        "bridge port(s) already have an established client:${FLEET_BUSY_PORTS}
      BridgeServer accepts exactly ONE TCP client and a second connection
      silently destroys the first. Stop the other driver before running this."
    log "Paper answering, ${FLEET_LISTENER_COUNT} bridge listener(s) free"

    # ---------------------------------------------------------------------
    # Phase 1 — the run. PRODUCTION settings: this is a dress rehearsal, not a
    # wiring test. The only deliberate departures are the budget (bounded), the
    # eval (OFF) and the epsilon schedule (pinned at the terminal value).
    #
    # Eval is off because an eval cycle PAUSES the designated arena for 30-45
    # minutes: leaving it on would put an arena-sized hole in the throughput
    # measurement this phase exists to take. The eval path is the canary's to
    # prove, and it did.
    # ---------------------------------------------------------------------
    log "phase 1: ${ARENAS}-pad smoke (${SMOKE_GRAD_STEPS} grad steps, --min-replay ${SMOKE_MIN_REPLAY})"
    local driver_pid=""
    cleanup_driver() {
        if [[ -n "${driver_pid}" ]] && kill -0 "${driver_pid}" 2>/dev/null; then
            kill -INT "${driver_pid}" 2>/dev/null || true
        fi
    }
    trap cleanup_driver EXIT INT TERM

    local driver_start
    driver_start=$(date +%s)
    (
        cd "${REPO_ROOT}" && exec "${PYTHON_BIN}" -m agent.train \
            --arenas "${ARENAS}" \
            --opponent selfplay \
            --host "${HOST}" \
            --port "${BRIDGE_BASE_PORT}" \
            --mc-port "${MC_PORT}" \
            --warm-start "${WARM_START}" \
            --warm-start-sha256 "${WARM_START_SHA256}" \
            --run-name "${SMOKE_RUN_NAME}" \
            --checkpoint "${SMOKE_CHECKPOINT_PATH}" \
            --best-checkpoint "${SMOKE_BEST_CHECKPOINT_PATH}" \
            --max-grad-steps "${SMOKE_GRAD_STEPS}" \
            --max-episodes 10000000 \
            --min-replay "${SMOKE_MIN_REPLAY}" \
            --snapshot-every-grad-steps "${SMOKE_SNAPSHOT_EVERY}" \
            --snapshot-sampling pfsp \
            --reference-promote-grad-steps "${SMOKE_PROMOTE_FIRST}" "${SMOKE_PROMOTE_SECOND}" \
            --eval-every-grad-steps 0 \
            --checkpoint-every-grad-steps "${SMOKE_CHECKPOINT_EVERY}" \
            --eps-decay-episodes "${SMOKE_EPS_DECAY_EPISODES}" \
            --seed "${SEED}" \
            --log-backend jsonl
    ) >"${SMOKE_DRIVER_LOG}" 2>&1 &
    driver_pid=$!
    log "driver pid ${driver_pid}, log ${SMOKE_DRIVER_LOG}"

    local jvm_pid
    jvm_pid="$(listener_pids "${MC_PORT}" | awk '{print $1}')"
    : >"${SMOKE_RSS_LOG}"
    local deadline_at=$(( driver_start + SMOKE_MINUTES * 60 ))
    local deadline_hit="false"
    local now rss jvm_rss
    while kill -0 "${driver_pid}" 2>/dev/null; do
        now=$(date +%s)
        rss="$(proc_rss_kb "${driver_pid}")"
        jvm_rss="$(proc_rss_kb "${jvm_pid:-0}")"
        if [[ -n "${rss}" ]]; then
            printf '%s\t%s\t%s\n' "${now}" "${rss}" "${jvm_rss:-0}" >>"${SMOKE_RSS_LOG}"
        fi
        if [[ "${now}" -ge "${deadline_at}" ]]; then
            deadline_hit="true"
            warn "wall-clock deadline (${SMOKE_MINUTES} min) reached; interrupting the driver."
            # SIGINT, not SIGTERM: train_multi_arena's teardown runs in a
            # `finally`, so an interrupt still joins the learner, flushes the
            # metrics summary and persists the pool. SIGTERM would take the
            # process out with no teardown and leave nothing to judge.
            kill -INT "${driver_pid}" 2>/dev/null || true
            local grace
            for (( grace = 0; grace < 180; grace++ )); do
                kill -0 "${driver_pid}" 2>/dev/null || break
                sleep 1
            done
            kill -KILL "${driver_pid}" 2>/dev/null || true
            break
        fi
        sleep "${SMOKE_RSS_SAMPLE_SECONDS}"
    done
    set +e
    wait "${driver_pid}"
    local driver_exit=$?
    set -e
    driver_pid=""
    trap - EXIT INT TERM
    local driver_wall=$(( $(date +%s) - driver_start ))
    # The exit code is NOT a health signal: `_main_multi_arena` returns
    # `0 if passed_m2 else 1`, and passed_m2 is the M2 gate against the
    # STATIONARY dummy, which a self-play run never clears. The verdict reads
    # the `[multi done]` line instead.
    log "driver exited ${driver_exit} after ${driver_wall}s (exit 1 is normal: passed_m2 is the M2 dummy gate)"
    tail -n 12 "${SMOKE_DRIVER_LOG}" || true

    # ---------------------------------------------------------------------
    # Phase 2 — the snapshot-load probe. Times exactly what
    # SnapshotOpponentDriver.begin_episode does: pool.load_state_dict(record)
    # then net.load_state_dict(state_dict). Connects to no bridge.
    # ---------------------------------------------------------------------
    log "phase 2: timing the per-episode snapshot load"
    (
        cd "${REPO_ROOT}" && \
        SMOKE_RUN_NAME="${SMOKE_RUN_NAME}" \
        SMOKE_LOAD_JSON="${SMOKE_LOAD_JSON}" \
        SMOKE_LOAD_REPEATS="${SMOKE_SNAPSHOT_LOAD_REPEATS}" \
        "${PYTHON_BIN}" - <<'LAUNCH_LOAD_PY'
"""Time the head-of-episode snapshot read. No sockets, no bridge."""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List

out: Dict[str, Any] = {"ok": False, "seconds": [], "error": None, "snapshots": 0}
try:
    import torch

    from agent.train import snapshot_pool_directory
    from eval.evaluate import _load_drqn
    from opponents.snapshot_pool import SnapshotPool

    directory = snapshot_pool_directory(os.environ["SMOKE_RUN_NAME"])
    pool = SnapshotPool.load(directory, sampling="pfsp")
    records = pool.records()
    out["snapshots"] = len(records)
    if not records:
        raise RuntimeError(f"no snapshots in {directory}")
    # Build the net ONCE, exactly as a collector does: the driver holds a
    # private CPU clone for the whole run and only ever reloads weights into it.
    net = _load_drqn(records[0].path, torch.device("cpu"))
    net.eval()
    repeats = int(os.environ.get("SMOKE_LOAD_REPEATS", "5"))
    seconds: List[float] = []
    for _ in range(repeats):
        for record in records:
            start = time.perf_counter()
            state_dict = pool.load_state_dict(record)
            net.load_state_dict(state_dict)
            seconds.append(time.perf_counter() - start)
    out["seconds"] = seconds
    out["ok"] = True
except Exception as exc:  # noqa: BLE001 - a failed probe is a REFUSAL upstream
    out["error"] = f"{type(exc).__name__}: {exc}"

with open(os.environ["SMOKE_LOAD_JSON"], "w", encoding="utf-8") as handle:
    json.dump(out, handle, indent=2, sort_keys=True)
    handle.write("\n")
LAUNCH_LOAD_PY
    ) || warn "the snapshot-load probe failed; the gate refuses on that."

    # ---------------------------------------------------------------------
    # Phase 3 — assemble the evidence, then judge. All arithmetic lives in
    # launch_sizing; this only reads files.
    # ---------------------------------------------------------------------
    log "phase 3: collecting evidence"
    (
        cd "${REPO_ROOT}" && \
        SMOKE_EVIDENCE="${SMOKE_EVIDENCE}" \
        SMOKE_RUN_NAME="${SMOKE_RUN_NAME}" \
        SMOKE_ARENAS="${ARENAS}" \
        SMOKE_MIN_REPLAY="${SMOKE_MIN_REPLAY}" \
        SMOKE_MAX_GRAD_STEPS="${SMOKE_GRAD_STEPS}" \
        SMOKE_WALL="${driver_wall}" \
        SMOKE_EXIT="${driver_exit}" \
        SMOKE_DEADLINE_HIT="${deadline_hit}" \
        SMOKE_DRIVER_LOG="${SMOKE_DRIVER_LOG}" \
        SMOKE_RSS_LOG="${SMOKE_RSS_LOG}" \
        SMOKE_LOAD_JSON="${SMOKE_LOAD_JSON}" \
        SMOKE_CANARY_MEASUREMENTS="${CANARY_MEASUREMENTS}" \
        SMOKE_WARM_START="${WARM_START}" \
        SMOKE_WARM_START_SHA256="${WARM_START_SHA256}" \
        SMOKE_WINDOW_HOURS="${WINDOW_HOURS}" \
        SMOKE_PHYSICAL_MEMORY="$(physical_memory_bytes)" \
        "${PYTHON_BIN}" - <<'LAUNCH_SMOKE_COLLECT_PY'
"""Assemble the smoke evidence document. Reads files; judges nothing."""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional


def read_json(path: str) -> Optional[Any]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


# --- the driver's own teardown line ---------------------------------------
# `[multi done] reason=X episodes=N grad_steps=M passed_m2=B checkpoints_saved=K`
# is the ONLY reliable completion signal; the process exit code is
# `0 if passed_m2 else 1` and a self-play run never clears the M2 dummy gate.
driver: Dict[str, Any] = {
    "completed": False,
    "exit_code": int(os.environ["SMOKE_EXIT"]),
    "deadline_hit": os.environ["SMOKE_DEADLINE_HIT"] == "true",
    "stop_reason": None,
    "episodes": None,
    "grad_steps": None,
    "checkpoints_saved": None,
    "log_path": os.environ["SMOKE_DRIVER_LOG"],
}
try:
    with open(
        os.environ["SMOKE_DRIVER_LOG"], "r", encoding="utf-8", errors="replace"
    ) as handle:
        log_text = handle.read()
except OSError:
    log_text = ""
match = None
for match in re.finditer(
    r"\[multi done\] reason=(?P<reason>\S+) episodes=(?P<episodes>\d+) "
    r"grad_steps=(?P<grad>\d+) passed_m2=(?P<passed>\S+) "
    r"checkpoints_saved=(?P<saved>\d+)",
    log_text,
):
    pass
if match is not None:
    driver.update(
        completed=True,
        stop_reason=match.group("reason"),
        episodes=int(match.group("episodes")),
        grad_steps=int(match.group("grad")),
        checkpoints_saved=int(match.group("saved")),
    )
# The learner watchdog trips when the backlog grows while grad_step stalls with
# the buffer WARM. It aborts the run, so it is also the loudest queue-depth
# signal the log carries.
watchdog_tripped = ("WatchdogError" in log_text) or ("[learner] aborting" in log_text)

# --- RSS samples ----------------------------------------------------------
samples: List[List[float]] = []
jvm_peak = 0.0
try:
    with open(os.environ["SMOKE_RSS_LOG"], "r", encoding="utf-8") as handle:
        for line in handle:
            fields = line.split()
            if len(fields) < 2:
                continue
            try:
                stamp = float(fields[0])
                # `ps -o rss=` reports KiB on macOS.
                rss_bytes = float(fields[1]) * 1024.0
                jvm_bytes = float(fields[2]) * 1024.0 if len(fields) > 2 else 0.0
            except ValueError:
                continue
            samples.append([stamp, rss_bytes])
            jvm_peak = max(jvm_peak, jvm_bytes)
except OSError:
    pass
rss = {
    "samples": samples,
    "first_bytes": samples[0][1] if samples else None,
    "peak_bytes": max((s[1] for s in samples), default=None),
    "jvm_peak_bytes": jvm_peak or None,
}

# --- the pool the collectors were reading ---------------------------------
pool: Dict[str, Any] = {"ok": False, "size": None, "snapshot_ids": [], "error": None}
try:
    from agent.train import snapshot_pool_directory
    from opponents.snapshot_pool import SnapshotPool

    loaded = SnapshotPool.load(
        snapshot_pool_directory(os.environ["SMOKE_RUN_NAME"]), sampling="pfsp"
    )
    pool.update(
        ok=True,
        size=len(loaded),
        snapshot_ids=[record.snapshot_id for record in loaded.records()],
    )
except Exception as exc:  # noqa: BLE001
    pool["error"] = f"{type(exc).__name__}: {exc}"

# --- the armored episode length this smoke's throughput is derived with ----
canary = read_json(os.environ["SMOKE_CANARY_MEASUREMENTS"]) or {}
episode_length = canary.get("armored_mean_episode_length_probe")
replay_capacity = None
try:
    from agent.train_config import TrainConfig

    replay_capacity = TrainConfig().replay_capacity
except Exception:  # noqa: BLE001
    replay_capacity = None

physical = os.environ.get("SMOKE_PHYSICAL_MEMORY", "").strip()
evidence = {
    "smoke_version": 1,
    "evidence_path": os.environ["SMOKE_EVIDENCE"],
    "run_name": os.environ["SMOKE_RUN_NAME"],
    "arenas": int(os.environ["SMOKE_ARENAS"]),
    "min_replay": int(os.environ["SMOKE_MIN_REPLAY"]),
    "max_grad_steps": int(os.environ["SMOKE_MAX_GRAD_STEPS"]),
    "wall_seconds": float(os.environ["SMOKE_WALL"]),
    "window_hours": float(os.environ["SMOKE_WINDOW_HOURS"]),
    "warm_start": os.environ["SMOKE_WARM_START"],
    "warm_start_sha256": os.environ["SMOKE_WARM_START_SHA256"],
    "driver": driver,
    "watchdog_tripped": watchdog_tripped,
    "rss": rss,
    "physical_memory_bytes": float(physical) if physical else None,
    "replay_capacity": replay_capacity,
    "pool": pool,
    "snapshot_load": read_json(os.environ["SMOKE_LOAD_JSON"])
    or {"ok": False, "seconds": [], "error": "probe wrote nothing"},
    "episode_length_steps": episode_length,
    "episode_length_source": (
        "canary probe (learner eps=0.05 / opponent eps=0.02), "
        f"{os.environ['SMOKE_CANARY_MEASUREMENTS']}"
    ),
}

with open(os.environ["SMOKE_EVIDENCE"], "w", encoding="utf-8") as handle:
    json.dump(evidence, handle, indent=2, sort_keys=True)
    handle.write("\n")
LAUNCH_SMOKE_COLLECT_PY
    ) || die "smoke evidence collection failed (traceback above). Nothing was
      judged, so NOTHING is cleared: treat this exactly as a refusal."

    log "evidence written to ${SMOKE_EVIDENCE}"
    echo ""
    set +e
    "${PYTHON_BIN}" "${SIZING_MODULE}" smoke-verdict "${SMOKE_EVIDENCE}" "${SMOKE_MEASUREMENTS}"
    local verdict_exit=$?
    set -e
    echo ""
    log "smoke measurements written to ${SMOKE_MEASUREMENTS}"
    exit "${verdict_exit}"
}

# ===========================================================================
# COMMAND: plan / launch — derive the sizing, gate, and (for launch) start the
# driver detached.
# ===========================================================================

build_plan_input() {
    (
        cd "${REPO_ROOT}" && \
        PLAN_OUT="${PLAN_INPUT}" \
        PLAN_RUN_NAME="${RUN_NAME}" \
        PLAN_ARENAS="${ARENAS}" \
        PLAN_WINDOW_HOURS="${WINDOW_HOURS}" \
        PLAN_HOST="${HOST}" \
        PLAN_BRIDGE_BASE_PORT="${BRIDGE_BASE_PORT}" \
        PLAN_MC_PORT="${MC_PORT}" \
        PLAN_SEED="${SEED}" \
        PLAN_PYTHON="${PYTHON_BIN}" \
        PLAN_REPO_ROOT="${REPO_ROOT}" \
        PLAN_WARM_START="${WARM_START}" \
        PLAN_WARM_START_SHA256="${WARM_START_SHA256}" \
        PLAN_CHECKPOINT="${CHECKPOINT_PATH}" \
        PLAN_BEST_CHECKPOINT="${BEST_CHECKPOINT_PATH}" \
        PLAN_LOG_PATH="${RUN_LOG}" \
        PLAN_PID_PATH="${RUN_PID_FILE}" \
        PLAN_CANARY_DIR="${CANARY_DIR}" \
        PLAN_CANARY_MEASUREMENTS="${CANARY_MEASUREMENTS}" \
        PLAN_CANARY_ANALYZE_EXIT="${CANARY_ANALYZE_EXIT}" \
        PLAN_SMOKE_MEASUREMENTS="${SMOKE_MEASUREMENTS}" \
        PLAN_SMOKE_EVIDENCE="${SMOKE_EVIDENCE}" \
        PLAN_MC_REACHABLE="${FLEET_MC_REACHABLE}" \
        PLAN_MISSING_PORTS="${FLEET_MISSING_PORTS}" \
        PLAN_BUSY_PORTS="${FLEET_BUSY_PORTS}" \
        PLAN_LISTENER_COUNT="${FLEET_LISTENER_COUNT}" \
        PLAN_ETIMES="${FLEET_ETIMES}" \
        PLAN_OV_EPS_DECAY="${OV_EPS_DECAY}" \
        PLAN_OV_EVAL_EPISODES="${OV_EVAL_EPISODES}" \
        PLAN_OV_REFERENCE_EVAL_EPISODES="${OV_REFERENCE_EVAL_EPISODES}" \
        PLAN_OV_EVAL_EVERY="${OV_EVAL_EVERY}" \
        PLAN_OV_MAX_EPISODES="${OV_MAX_EPISODES}" \
        PLAN_OV_MAX_GRAD_STEPS="${OV_MAX_GRAD_STEPS}" \
        PLAN_OV_CHECKPOINT_EVERY="${OV_CHECKPOINT_EVERY}" \
        PLAN_OV_SNAPSHOT_EVERY="${OV_SNAPSHOT_EVERY}" \
        PLAN_SIZING_MODULE="${SIZING_MODULE}" \
        "${PYTHON_BIN}" - <<'LAUNCH_PLAN_COLLECT_PY'
"""Assemble the launch plan's INPUT document. Reads files; judges nothing."""
from __future__ import annotations

import importlib.util
import json
import os
import time
from typing import Any, Dict, List, Optional

spec = importlib.util.spec_from_file_location(
    "launch_sizing", os.environ["PLAN_SIZING_MODULE"]
)
sizing_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sizing_module)


def read_json(path: str) -> Optional[Any]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def env_int(name: str) -> Optional[int]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def ports(name: str) -> List[int]:
    return [int(token) for token in os.environ.get(name, "").split() if token.isdigit()]


def mtime(path: str) -> Optional[float]:
    try:
        return os.path.getmtime(path)
    except OSError:
        return None


canary_path = os.environ["PLAN_CANARY_MEASUREMENTS"]
canary_measurements = read_json(canary_path)
smoke_path = os.environ["PLAN_SMOKE_MEASUREMENTS"]
smoke_measurements = read_json(smoke_path)

# The smoke's verdict, re-derived from its own evidence rather than remembered:
# a measurements file is written whether the smoke passed or refused, exactly
# like the canary's, so its presence is not a pass.
smoke_ok = False
smoke_evidence = read_json(os.environ["PLAN_SMOKE_EVIDENCE"])
if isinstance(smoke_evidence, dict):
    smoke_ok = sizing_module.evaluate_smoke(
        smoke_evidence, smoke_evidence.get("thresholds")
    ).ok

warm_start = os.environ.get("PLAN_WARM_START", "")
warm = {
    "path": warm_start,
    "exists": os.path.exists(warm_start) if warm_start else False,
    "is_file": os.path.isfile(warm_start) if warm_start else False,
    "sha256": os.environ.get("PLAN_WARM_START_SHA256", ""),
    "bytes": os.path.getsize(warm_start) if warm_start and os.path.isfile(warm_start) else None,
}

# What already occupies this run's OUTPUT namespace. Deliberately NOT
# `runs/<run>/` itself: this script's own evidence directory lives there, so a
# `plan` run would otherwise refuse the launch it just planned.
run_name = os.environ["PLAN_RUN_NAME"]
run_dir = os.path.join("runs", run_name)
existing = [
    path
    for path in (
        os.environ["PLAN_CHECKPOINT"],
        os.environ["PLAN_BEST_CHECKPOINT"],
        os.path.join(run_dir, "metrics.jsonl"),
        os.path.join(run_dir, "summary.json"),
        os.path.join(run_dir, "snapshots"),
    )
    if os.path.exists(path)
]

# The YOUNGEST bridge process decides freshness: one pad restarted after the
# canary means one pad nothing has proved is knockback-enabled.
ages = [
    parsed
    for parsed in (
        sizing_module.parse_etime(token) for token in os.environ.get("PLAN_ETIMES", "").split()
    )
    if parsed is not None
]

plan: Dict[str, Any] = {
    "plan_version": sizing_module.PLAN_VERSION,
    "now_epoch": time.time(),
    "window_hours": float(os.environ["PLAN_WINDOW_HOURS"]),
    "arenas": int(os.environ["PLAN_ARENAS"]),
    "run_name": run_name,
    "opponent": "selfplay",
    "host": os.environ["PLAN_HOST"],
    "bridge_base_port": int(os.environ["PLAN_BRIDGE_BASE_PORT"]),
    "mc_port": int(os.environ["PLAN_MC_PORT"]),
    "seed": int(os.environ["PLAN_SEED"]),
    "python_bin": os.environ["PLAN_PYTHON"],
    "repo_root": os.environ["PLAN_REPO_ROOT"],
    "episode_length_source": "probe",
    "warm_start": warm,
    "checkpoint": os.environ["PLAN_CHECKPOINT"],
    "best_checkpoint": os.environ["PLAN_BEST_CHECKPOINT"],
    "log_path": os.environ["PLAN_LOG_PATH"],
    "pid_path": os.environ["PLAN_PID_PATH"],
    "existing_outputs": existing,
    "canary": {
        "directory": os.environ["PLAN_CANARY_DIR"],
        "measurements_path": canary_path,
        "exists": canary_measurements is not None,
        "mtime": mtime(canary_path),
        "measurements": canary_measurements,
        "analyze_exit": env_int("PLAN_CANARY_ANALYZE_EXIT"),
    },
    "smoke": {
        "measurements_path": smoke_path,
        "exists": smoke_measurements is not None,
        "mtime": mtime(smoke_path),
        "measurements": smoke_measurements,
        "verdict_ok": smoke_ok,
    },
    "fleet": {
        "host": os.environ["PLAN_HOST"],
        "mc_port": int(os.environ["PLAN_MC_PORT"]),
        "mc_reachable": os.environ["PLAN_MC_REACHABLE"] == "true",
        "missing_ports": ports("PLAN_MISSING_PORTS"),
        "busy_ports": ports("PLAN_BUSY_PORTS"),
        "listener_count": int(os.environ["PLAN_LISTENER_COUNT"]),
        "youngest_listener_age_seconds": min(ages) if ages else None,
        "oldest_listener_age_seconds": max(ages) if ages else None,
    },
    "overrides": {
        "eps_decay_episodes": env_int("PLAN_OV_EPS_DECAY"),
        "eval_episodes": env_int("PLAN_OV_EVAL_EPISODES"),
        "reference_eval_episodes": env_int("PLAN_OV_REFERENCE_EVAL_EPISODES"),
        "eval_every_grad_steps": env_int("PLAN_OV_EVAL_EVERY"),
        "max_episodes": env_int("PLAN_OV_MAX_EPISODES"),
        "max_grad_steps": env_int("PLAN_OV_MAX_GRAD_STEPS"),
        "checkpoint_every_grad_steps": env_int("PLAN_OV_CHECKPOINT_EVERY"),
        "snapshot_every_grad_steps": env_int("PLAN_OV_SNAPSHOT_EVERY"),
    },
}

with open(os.environ["PLAN_OUT"], "w", encoding="utf-8") as handle:
    json.dump(plan, handle, indent=2, sort_keys=True)
    handle.write("\n")
LAUNCH_PLAN_COLLECT_PY
    ) || die "could not assemble the launch plan (traceback above)."
}

cmd_plan() {
    require_warm_start
    inspect_fleet
    run_canary_analyze
    build_plan_input
    set +e
    "${PYTHON_BIN}" "${SIZING_MODULE}" plan "${PLAN_INPUT}" "${PLAN_JSON}" "${PLAN_ARGV}"
    local exit_code=$?
    set -e
    echo ""
    log "plan written to ${PLAN_JSON}"
    return "${exit_code}"
}

cmd_launch() {
    set +e
    cmd_plan
    local plan_exit=$?
    set -e
    if [[ "${plan_exit}" -ne 0 ]]; then
        echo "" >&2
        echo "[launch] REFUSED. Nothing was started." >&2
        exit 1
    fi
    [[ -f "${PLAN_ARGV}" ]] || die "the plan cleared but wrote no argv file; refusing to guess."

    # Read the argv the GATE produced, one element per line. Never rebuilt here:
    # a flag that differs between the checked plan and the started process is a
    # gate that checked something the run did not do.
    local -a driver_args=()
    while IFS= read -r line; do
        [[ -n "${line}" ]] && driver_args+=("${line}")
    done <"${PLAN_ARGV}"
    [[ "${#driver_args[@]}" -gt 0 ]] || die "empty argv file at ${PLAN_ARGV}"

    # Last look at the bridges. The plan was gated seconds ago, but a probe or a
    # stray driver attaching in between would be destroyed by this connection.
    inspect_fleet
    [[ -z "${FLEET_BUSY_PORTS// }" ]] || die \
        "bridge port(s) acquired a client since the plan was gated:${FLEET_BUSY_PORTS}
      BridgeServer destroys the incumbent on a second connection. Not starting."

    log "starting the driver DETACHED (nohup): it must survive this session."
    ( cd "${REPO_ROOT}" && exec nohup "${PYTHON_BIN}" -m agent.train "${driver_args[@]}" ) \
        >"${RUN_LOG}" 2>&1 &
    local run_pid=$!
    echo "${run_pid}" >"${RUN_PID_FILE}"

    # A config refusal (a bad warm-start digest, an impossible flag combination)
    # lands in the first second and would otherwise leave the operator believing
    # the night had started.
    sleep 20
    if ! kill -0 "${run_pid}" 2>/dev/null; then
        echo "" >&2
        echo "[launch] the driver EXITED within 20 s. It did not start." >&2
        tail -n 30 "${RUN_LOG}" >&2 || true
        exit 1
    fi

    echo ""
    log "RUNNING"
    log "  pid        ${run_pid}   (also in ${RUN_PID_FILE})"
    log "  log        ${RUN_LOG}"
    log "  checkpoint ${CHECKPOINT_PATH}"
    log "  best       ${BEST_CHECKPOINT_PATH}"
    log "  pool       runs/${RUN_NAME}/snapshots"
    log ""
    log "In the morning:"
    log "  scripts/launch_selfplay.sh compare --run-name ${RUN_NAME} \\"
    log "      --extra-runs /Users/diego/Documents/MinecraftRL/runs"
    log ""
    log "Exit code 1 at the end is NOT failure: it means no eval cleared the M2"
    log "gate, which is defined against the STATIONARY dummy and which a"
    log "self-play run does not clear. Judge by the 'best checkpoint:' line."
    exit 0
}

# ===========================================================================
# COMMAND: compare — the morning checkpoint table. Offline; starts nothing and
# connects to nothing.
# ===========================================================================
cmd_compare() {
    local extra_joined=""
    local dir
    for dir in "${EXTRA_RUNS_DIRS[@]+"${EXTRA_RUNS_DIRS[@]}"}"; do
        extra_joined="${extra_joined}${dir}:"
    done
    (
        cd "${REPO_ROOT}" && \
        COMPARE_OUT="${COMPARE_INPUT}" \
        COMPARE_RUN_NAME="${RUN_NAME}" \
        COMPARE_RUNS_DIR="${REPO_ROOT}/runs" \
        COMPARE_EXTRA_DIRS="${extra_joined}" \
        "${PYTHON_BIN}" - <<'LAUNCH_COMPARE_COLLECT_PY'
"""Collect every demo candidate and whatever is KNOWN about each one.

Reads only: checkpoint payload metadata, the snapshot pool index, and the run's
own metrics.jsonl / summary.json. It runs no evaluation — a number that was
never measured is reported as absent, because inventing one is how a bare
figure becomes a ranking.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional


def read_json(path: str) -> Optional[Any]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
    except OSError:
        return []
    return rows


def payload_meta(path: str) -> Dict[str, Any]:
    """Checkpoint metadata WITHOUT the weights, or an error string."""
    if not os.path.isfile(path):
        return {"error": None}
    try:
        import torch

        payload = torch.load(path, map_location="cpu")
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}
    if not isinstance(payload, dict):
        return {"error": "checkpoint is a bare state_dict with no metadata"}
    return {k: v for k, v in payload.items() if k != "model"}


def nearest_row(rows: List[Dict[str, Any]], key: str, at_step: Optional[int]):
    """The last row carrying ``key`` at or before ``at_step`` — value and step."""
    best = (None, None)
    for row in rows:
        step = row.get("step")
        if key not in row or not isinstance(step, int):
            continue
        if at_step is not None and step > at_step:
            continue
        best = (row.get(key), step)
    return best


run_name = os.environ["COMPARE_RUN_NAME"]
runs_dir = os.environ["COMPARE_RUNS_DIR"]
extra_dirs = [d for d in os.environ.get("COMPARE_EXTRA_DIRS", "").split(":") if d]
candidates: List[Dict[str, Any]] = []

# --- the self-play run's own checkpoints -----------------------------------
metrics = read_jsonl(os.path.join("runs", run_name, "metrics.jsonl"))
for suffix, kind in ((".best.pt", "best"), (".pt", "latest")):
    path = os.path.join(runs_dir, f"{run_name}{suffix}")
    meta = payload_meta(path)
    grad_step = meta.get("grad_step")
    rated, rated_step = nearest_row(metrics, "elo/learner_rated", grad_step)
    references = meta.get("references_evaluated")
    entry: Dict[str, Any] = {
        "label": f"runs/{run_name}{suffix}",
        "path": path,
        "exists": os.path.isfile(path),
        "run": run_name,
        "kind": kind,
        "grad_step": grad_step,
        "rated_elo": rated,
        "error": meta.get("error"),
    }
    if references is not None:
        # A self-play best checkpoint is SELECTED on the reference aggregate,
        # while `scripted_win_rate` in the same payload is the yardstick track.
        entry["reference_aggregate"] = meta.get("win_rate")
        entry["reference_worst"] = meta.get("worst_reference_win_rate")
        entry["references_evaluated"] = references
        entry["scripted_win_rate"] = meta.get("scripted_win_rate")
        entry["win_rate"] = meta.get("win_rate")
        entry["eval_opponent"] = meta.get("eval_opponent")
        entry["source"] = (
            f"stamped into the checkpoint by the save hook at grad_step {grad_step}"
            + (
                f"; rated Elo from the metrics row at step {rated_step}"
                if rated_step is not None
                else "; no rated Elo row at or before that step"
            )
            + (
                ""
                if meta.get("candidate_frozen")
                else "; candidate_frozen=False - the cycle scored the LIVE net, "
                "so this rate describes approximately-these-bytes"
            )
        )
    elif meta.get("win_rate") is not None:
        entry["scripted_win_rate"] = meta.get("win_rate")
        entry["win_rate"] = meta.get("win_rate")
        entry["eval_opponent"] = meta.get("eval_opponent")
        entry["source"] = (
            f"stamped into the checkpoint by the save hook at grad_step {grad_step}"
        )
    elif not entry["exists"]:
        entry["source"] = "no such file - this run never wrote it"
    else:
        entry["source"] = (
            "the periodic/final save carries no eval metadata: it is the LATEST "
            "net, selected by recency, never by a score"
        )
    candidates.append(entry)

# --- every snapshot in the pool --------------------------------------------
try:
    from agent.train import snapshot_pool_directory
    from opponents.snapshot_pool import SnapshotPool

    pool = SnapshotPool.load(snapshot_pool_directory(run_name), sampling="pfsp")
    for record in pool.records():
        learner_rate, rate_step = nearest_row(
            metrics, f"selfplay/win_rate_vs_ref_{record.snapshot_id}", None
        )
        source = (
            "Elo frozen at archive time from the learner's rating then; a "
            "snapshot is never evaluated on its own"
        )
        if learner_rate is not None:
            source += (
                f". The LEARNER scored {learner_rate:.3f} against it at step "
                f"{rate_step} - that is the learner's number, not this "
                "snapshot's"
            )
        candidates.append(
            {
                "label": f"snapshot {record.snapshot_id}"
                + (" (PINNED)" if record.pinned else ""),
                "path": record.path,
                "exists": os.path.isfile(record.path),
                "run": run_name,
                "kind": "snapshot",
                "grad_step": record.grad_step,
                "elo": record.elo,
                "source": source,
            }
        )
except Exception as exc:  # noqa: BLE001
    candidates.append(
        {
            "label": f"runs/{run_name}/snapshots",
            "path": os.path.join("runs", run_name, "snapshots"),
            "exists": False,
            "run": run_name,
            "kind": "snapshot",
            "error": f"pool unreadable: {type(exc).__name__}: {exc}",
        }
    )

# --- the fallbacks, which live in whatever tree the operator names ----------
# The bare-handed run's checkpoints are in the MAIN checkout, not this worktree.
for directory in [runs_dir] + extra_dirs:
    for legacy in ("m4.best.pt", "m4.pt", "m2_multi.pt"):
        path = os.path.join(directory, legacy)
        if not os.path.isfile(path):
            continue
        legacy_run = legacy.split(".")[0]
        meta = payload_meta(path)
        summary = read_json(os.path.join(directory, legacy_run, "summary.json")) or {}
        entry = {
            "label": os.path.join(os.path.basename(directory), legacy),
            "path": path,
            "exists": True,
            "run": legacy_run,
            "kind": "legacy",
            "grad_step": meta.get("grad_step"),
            "error": meta.get("error"),
        }
        if meta.get("win_rate") is not None:
            entry["win_rate"] = meta.get("win_rate")
            entry["scripted_win_rate"] = meta.get("win_rate")
            entry["eval_opponent"] = meta.get("eval_opponent")
            entry["source"] = (
                "stamped into the checkpoint by the save hook at grad_step "
                f"{meta.get('grad_step')} over {meta.get('eval_episodes')} episodes"
            )
        elif summary.get("win_rate") is not None:
            entry["scripted_win_rate"] = summary.get("win_rate")
            entry["win_rate"] = summary.get("win_rate")
            entry["eval_opponent"] = summary.get("opponent")
            entry["source"] = (
                f"{legacy_run}/summary.json - the RUN's LAST eval over "
                f"{summary.get('n_episodes')} episodes, not this file's own "
                "score. The final checkpoint and the last eval are close in "
                "time, not identical."
            )
        else:
            entry["source"] = "no eval metadata anywhere for this file"
        candidates.append(entry)

with open(os.environ["COMPARE_OUT"], "w", encoding="utf-8") as handle:
    json.dump(
        {"compare_version": 1, "candidates": candidates},
        handle,
        indent=2,
        sort_keys=True,
        default=str,
    )
    handle.write("\n")
LAUNCH_COMPARE_COLLECT_PY
    ) || die "could not collect the checkpoint candidates (traceback above)."

    set +e
    "${PYTHON_BIN}" "${SIZING_MODULE}" compare "${COMPARE_INPUT}"
    local exit_code=$?
    set -e
    exit "${exit_code}"
}

# ===========================================================================
# Dispatch.
# ===========================================================================
case "${COMMAND}" in
    smoke)   cmd_smoke ;;
    plan)
        set +e
        cmd_plan
        PLAN_EXIT=$?
        set -e
        exit "${PLAN_EXIT}"
        ;;
    launch)  cmd_launch ;;
    compare) cmd_compare ;;
    *) die "unreachable: ${COMMAND}" ;;
esac
