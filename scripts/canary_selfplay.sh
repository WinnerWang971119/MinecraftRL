#!/usr/bin/env bash
# canary_selfplay.sh — T17: the LAST gate before the 24-hour, 25-pad self-play run.
#
# WHY THE OBVIOUS CANARY IS WORTHLESS. At the measured 4.8782 transitions/s per
# arena (docs/plans/2026-08-16-demo-scripted-opponent-exhibition.md, the 600 s
# confirm at N=25: "121.95/s aggregate, 4.8782 per-arena") and the production
# `min_replay = 25_000`, a 2-arena 15-minute run collects
#
#     2 * 4.8782 * 900 = 8780.8  ->  ~8,781 transitions
#
# which is 35% of the warm-up floor. `Trainer.learn()` is a no-op below that
# floor, so such a run takes ZERO gradient steps. It would exit green having
# published no weights, archived no snapshot, drawn no PFSP sample, moved no Elo
# and run no eval — a full pass on unchanged M3 warm-start weights, proving
# nothing. That is the failure this file exists to make impossible.
#
# THE CHOICE THIS CANARY MAKES: LOWER `--min-replay`, DO NOT RUN LONGER.
# The plan allows either "run >= ~45 minutes" or "lower --min-replay for the
# canary". This script lowers it, to 2000 (production is 25000), because:
#
#   * `min_replay` decides only WHEN learning starts, never WHETHER the
#     snapshot / PFSP / Elo / eval wiring works - and wiring is all a canary can
#     test. Sample efficiency is the long run's business.
#   * At the default --arenas 4 the warm-up is 2000 / (4 * 4.8782) = ~103 s
#     instead of 25000 / (4 * 4.8782) = ~21.4 min (~42.7 min at 2 pads). The
#     whole 1200-step budget then lands inside a ~30-minute window, so a second
#     attempt after a fix costs half an hour rather than three quarters of one.
#     A SECOND ATTEMPT IS NOT FREE OF STATE - see the next block.
#   * The measured learner rate is ~4,570 grad steps/hour (the M3 retry:
#     30,000 steps in 6h34m), i.e. ~1.27/s, so 1200 steps is ~16 minutes of
#     actual learning.
#
# Those four numbers are this script's DEFAULTS (MIN_REPLAY / ARENAS /
# MAX_GRAD_STEPS below); the flags override them and the verdict banner reports
# whatever was actually used.
#
# The lowered floor is PRINTED in the verdict banner and recorded in
# evidence.json, so nobody can mistake this run's replay regime for the long
# run's.
#
# EVERY ATTEMPT STARTS CLEAN, AND THE GATE ENFORCES IT.
# `agent/train.py`'s `build_snapshot_opponents` LOAD-EXTENDS a pool it finds:
#
#     if os.path.isfile(os.path.join(directory, INDEX_FILENAME)):
#         pool = SnapshotPool.load(directory, sampling=..., log=...)
#
# and `SnapshotPool.load` restores the registry, BOTH Elo series and the match
# counters. So a second attempt into the same --run-name inherits attempt 1's
# artifacts, and four refusals are pre-satisfied by them even if the fix under
# test broke the very thing being gated: NO_NEW_SNAPSHOT (the pool is already
# >= 2), SNAPSHOT_UNCHANGED (the newest snapshot already differs from snapshot
# 0), RATED_ELO_EMPTY (rated_matches is already nonzero) and
# DRAW_MAJORITY_TRAINING (this attempt's draws diluted by attempt 1's matches).
# CHECKPOINT_UNLOADABLE can likewise pass on the stale runs/<run>.pt. Only
# ZERO_GRAD_STEPS, DRIVER_FAILED and the probe checks are immune, because they
# read this attempt's truncated driver.log and this attempt's live episodes.
#
# A gate that stops gating on attempt 2 is the "manufactures confidence"
# failure this file exists to prevent, so it is closed twice over:
#
#   1. PREFLIGHT (phase 0, below) REFUSES to start when runs/<run>/snapshots/
#      pool.json, runs/<run>.pt, runs/<run>.best.pt or runs/<run>/metrics.jsonl
#      already exists, naming each file and telling the operator to delete them
#      or pass a fresh --run-name.
#   2. STALE_POOL (a verdict refusal, so it cannot be sidestepped by choosing a
#      new name or by editing the preflight out) refuses unless the newest
#      snapshot is PROVEN to be this run's: its file must have been written at
#      or after the driver started, and the pool's per-snapshot grad steps must
#      read as one run's history rather than two concatenated.
#
# WHAT THIS SCRIPT DOES *NOT* DO: it starts no Minecraft server, no Paper JVM
# and no bridge. The boot order for anything live is Paper -> bridges -> Python
# driver, and the operator owns the first two steps:
#
#     DUMMY_KNOCKBACK_IMMUNE=false server/setup/start-pads.sh --pads N
#
# This script is the third step only. It VERIFIES the fleet is up (Paper by TCP
# connect, which is safe because Paper is multi-client; bridges by `lsof`
# listener inspection, which never opens a connection) and refuses with
# instructions if it is not.
#
# NEVER CONNECT TWICE TO A BRIDGE PORT. `BridgeServer` accepts exactly ONE TCP
# client and `_onConnection` resolves a second one by DESTROYING the incumbent
# (bridge/transport.js). Four outages in this project came from that. Every
# bridge-port check below is non-connecting (`lsof`), and the two phases that DO
# connect (the probe, then the training driver) are strictly sequential and are
# gated on `lsof` reporting no ESTABLISHED peer on the port first.
#
# PHASES
#   0. Preflight — paths, interpreter, warm start, Paper, bridge listeners, no
#      pre-existing bridge clients. Starts nothing.
#   1. Pre-probe — one live episode (default) on pad 0 with the WARM-START net
#      driving BOTH seats through the mirrored observation. Fail-fast on a
#      mis-geared or frozen fleet BEFORE spending the training budget.
#      Non-strict: the volume-dependent checks are deferred to phase 3.
#   2. Training — `agent.train --opponent selfplay`, bounded by
#      --max-grad-steps and a wall-clock deadline.
#   3. Post-probe — three live episodes (default) with the run's FINAL
#      checkpoint. This is the strict gear/diversity evidence AND the first
#      measurement of armored episode length anyone has ever taken (T19 reads
#      it).
#   4. Collect + verdict — one evidence document, one pure decision function,
#      one exit code.
#
# EXIT CODES: 0 = GREEN (launch cleared); 1 = REFUSED (at least one condition
# blocks); 2 = usage/preflight error, raised BEFORE the budget is spent, so
# nothing was run; 3 = the run happened but could not be JUDGED (the evidence
# collector raised). 3 exists because 2 promises "nothing was run" and a
# collector failure breaks that promise: the fleet has been driven for half an
# hour and there is simply no verdict. Treat 3 exactly as a refusal.
#
# Owner: T17. The decision logic is the embedded `canary_verdict` module below,
# extracted verbatim and driven over synthetic metrics by
# tests/test_canary_selfplay.py — a canary whose refusals do not fire is worse
# than no canary, because it manufactures confidence.

set -euo pipefail

# --- Defaults --------------------------------------------------------------
RUN_NAME="m4_selfplay_canary"
ARENAS=4
MAX_GRAD_STEPS=1200
MIN_REPLAY=2000                 # LOWERED for the canary; see the header.
MIN_REPLAY_PRODUCTION=25000     # agent/train_config.py TrainConfig.min_replay.
SNAPSHOT_EVERY=300              # EXPECTS snapshots at 300/600/900/1200;
                                # NO_NEW_SNAPSHOT refuses if none appear.
PROMOTE_FIRST=400               # pinned reference #2.
PROMOTE_SECOND=800              # pinned reference #3.
EVAL_EVERY=400
EVAL_EPISODES=6
CHECKPOINT_EVERY=400
EPS_DECAY_EPISODES=50           # canary-only; T19 sizes the real value.
PROBE_EPISODES=3
PREFLIGHT_PROBE_EPISODES=1
PROBE_LEARNER_EPSILON=0.05      # TrainConfig.eps_end — the run's terminal eps.
PROBE_OPPONENT_EPSILON=0.02     # TrainConfig.opponent_epsilon.
DEADLINE_MINUTES=90
SEED=0
HOST="127.0.0.1"
BRIDGE_BASE_PORT=5555
MC_PORT=25565
WARM_START=""
OUT_DIR=""
PYTHON_BIN=""
ANALYZE_ONLY=""
DO_PREFLIGHT_PROBE=1

# --- Resolve paths ---------------------------------------------------------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)}"

usage() {
    cat <<USAGE
usage: scripts/canary_selfplay.sh --warm-start PATH [options]

The T17 learning canary: a SHORT self-play run that actually takes gradient
steps, then a live probe, then a pass/refuse verdict on the 24-hour run.

REQUIRED
  --warm-start PATH     Absolute path to the frozen M3 checkpoint the self-play
                        run starts from (e.g. .../MinecraftRL/runs/m4.best.pt).
                        Its sha256 is computed and passed to the driver so the
                        AC14 checksum gate actually executes.

FLEET (this script starts NOTHING — boot it yourself, Paper -> bridges first)
  --arenas N            Pads to use. Default ${ARENAS}. Ports ${BRIDGE_BASE_PORT}..${BRIDGE_BASE_PORT}+N-1.
  --bridge-base-port P  Default ${BRIDGE_BASE_PORT}.
  --mc-port P           Default ${MC_PORT}.
  --host H              Bridge host. Default ${HOST}.

LEARNING BUDGET
  --max-grad-steps N    Gradient-step budget. Default ${MAX_GRAD_STEPS}.
  --min-replay N        Canary warm-up floor. Default ${MIN_REPLAY} (production ${MIN_REPLAY_PRODUCTION}).
  --snapshot-every N    Snapshot cadence in grad steps. Default ${SNAPSHOT_EVERY}.
  --promote A B         Pinned-reference promotion steps. Default ${PROMOTE_FIRST} ${PROMOTE_SECOND}.
  --eval-every N        Eval cadence in grad steps. Default ${EVAL_EVERY}.
  --eval-episodes N     Episodes per eval cycle. Default ${EVAL_EPISODES}.
  --eps-decay-episodes N  Default ${EPS_DECAY_EPISODES} (canary-only; T19 sizes the real one).
  --deadline-minutes M  Hard wall-clock ceiling. Default ${DEADLINE_MINUTES}.

PROBE
  --probe-episodes N    Post-run probe episodes. Default ${PROBE_EPISODES}.
  --no-preflight-probe  Skip the fail-fast pre-probe (NOT recommended).

MISC
  --run-name NAME       Default ${RUN_NAME}.
  --out-dir DIR         Evidence directory. Default runs/<run-name>/canary.
  --seed N              Default ${SEED}.
  --python PATH         Interpreter. Default \$PYTHON, else <repo>/.venv/bin/python.
  --analyze-only DIR    Re-run the verdict over an existing evidence directory.
                        Connects to nothing and runs nothing. Handled BEFORE
                        every preflight check, so it needs no fleet and no
                        --warm-start.
  -h, --help            This message.

EXIT CODES
  0  GREEN - the 24-hour run is cleared to launch.
  1  REFUSED - at least one condition blocks it.
  2  usage or preflight error, raised before the budget is spent (nothing ran).
  3  the run happened but could not be JUDGED (the evidence collector raised).

EVERY ATTEMPT STARTS CLEAN. The preflight REFUSES when runs/<run>/snapshots/
pool.json, runs/<run>.pt, runs/<run>.best.pt or runs/<run>/metrics.jsonl already
exists: build_snapshot_opponents load-extends an existing pool (and
SnapshotPool.load restores both Elo series and the match counters), so a second
attempt into the same --run-name would inherit the first one's evidence and pass
four refusals it never earned. Delete them, or pass a fresh --run-name.

Boot the fleet first, and boot it with knockback ON for the opponent:

  DUMMY_KNOCKBACK_IMMUNE=false server/setup/start-pads.sh --pads ${ARENAS}
USAGE
}

need_value() {
    if [[ "$2" -lt 2 ]]; then
        echo "[canary] $1 requires a value." >&2
        exit 2
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --warm-start)         need_value "$1" $#; WARM_START="$2"; shift 2 ;;
        --arenas)             need_value "$1" $#; ARENAS="$2"; shift 2 ;;
        --bridge-base-port)   need_value "$1" $#; BRIDGE_BASE_PORT="$2"; shift 2 ;;
        --mc-port)            need_value "$1" $#; MC_PORT="$2"; shift 2 ;;
        --host)               need_value "$1" $#; HOST="$2"; shift 2 ;;
        --max-grad-steps)     need_value "$1" $#; MAX_GRAD_STEPS="$2"; shift 2 ;;
        --min-replay)         need_value "$1" $#; MIN_REPLAY="$2"; shift 2 ;;
        --snapshot-every)     need_value "$1" $#; SNAPSHOT_EVERY="$2"; shift 2 ;;
        --promote)
            if [[ $# -lt 3 ]]; then
                echo "[canary] --promote requires two values." >&2
                exit 2
            fi
            PROMOTE_FIRST="$2"; PROMOTE_SECOND="$3"; shift 3 ;;
        --eval-every)         need_value "$1" $#; EVAL_EVERY="$2"; shift 2 ;;
        --eval-episodes)      need_value "$1" $#; EVAL_EPISODES="$2"; shift 2 ;;
        --eps-decay-episodes) need_value "$1" $#; EPS_DECAY_EPISODES="$2"; shift 2 ;;
        --deadline-minutes)   need_value "$1" $#; DEADLINE_MINUTES="$2"; shift 2 ;;
        --probe-episodes)     need_value "$1" $#; PROBE_EPISODES="$2"; shift 2 ;;
        --no-preflight-probe) DO_PREFLIGHT_PROBE=0; shift ;;
        --run-name)           need_value "$1" $#; RUN_NAME="$2"; shift 2 ;;
        --out-dir)            need_value "$1" $#; OUT_DIR="$2"; shift 2 ;;
        --seed)               need_value "$1" $#; SEED="$2"; shift 2 ;;
        --python)             need_value "$1" $#; PYTHON_BIN="$2"; shift 2 ;;
        --analyze-only)       need_value "$1" $#; ANALYZE_ONLY="$2"; shift 2 ;;
        -h|--help)            usage; exit 0 ;;
        *)
            echo "[canary] unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

log()  { echo "[canary] $*"; }
warn() { echo "[canary] WARNING: $*" >&2; }
die()  { echo "[canary] FATAL: $*" >&2; exit 2; }

for pair in "ARENAS ${ARENAS}" "MAX_GRAD_STEPS ${MAX_GRAD_STEPS}" \
            "MIN_REPLAY ${MIN_REPLAY}" "SNAPSHOT_EVERY ${SNAPSHOT_EVERY}" \
            "PROMOTE_FIRST ${PROMOTE_FIRST}" "PROMOTE_SECOND ${PROMOTE_SECOND}" \
            "EVAL_EVERY ${EVAL_EVERY}" "EVAL_EPISODES ${EVAL_EPISODES}" \
            "EPS_DECAY_EPISODES ${EPS_DECAY_EPISODES}" \
            "PROBE_EPISODES ${PROBE_EPISODES}" \
            "DEADLINE_MINUTES ${DEADLINE_MINUTES}" \
            "BRIDGE_BASE_PORT ${BRIDGE_BASE_PORT}" "MC_PORT ${MC_PORT}"; do
    name="${pair%% *}"
    value="${pair#* }"
    if ! [[ "${value}" =~ ^[0-9]+$ ]] || [[ "${value}" -lt 1 ]]; then
        die "${name} must be a positive integer, got '${value}'."
    fi
done
if ! [[ "${SEED}" =~ ^[0-9]+$ ]]; then
    die "SEED must be a non-negative integer, got '${SEED}'."
fi
# The existing `--arenas < 2` refusal in agent/train.py covers `selfplay` (the
# single-arena loop steps no opponent policy at all), so a 1-pad canary would
# die inside the driver with a config error after the probe had already run.
if [[ "${ARENAS}" -lt 2 ]]; then
    die "--arenas must be >= 2 for --opponent selfplay (the single-arena loop
      steps no opponent policy; agent/train.py refuses it at config time)."
fi

# --- Resolve the interpreter ----------------------------------------------
if [[ -z "${PYTHON_BIN}" ]]; then
    PYTHON_BIN="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"
fi
if [[ ! -x "${PYTHON_BIN}" ]] && ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    die "python interpreter not found: ${PYTHON_BIN}
      System python 3.9 cannot import this package. Create the venv:
        python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
      Or pass --python /path/to/python."
fi

# --- Refuse to canary somebody else's checkout -----------------------------
# WHICH TREE'S CODE IS THIS ACTUALLY TESTING? The M4 work lives in a WORKTREE
# beside the main checkout, and the venv belongs to the main checkout. Every
# Python phase below runs `python -` with the repo root as the cwd, so `''`
# leads sys.path and the worktree wins - but a PYTHONPATH or an editable install
# silently flips that, and then the canary green-lights code that is not the
# code the fleet will run. The plan says it plainly: reboot from this branch
# "or the run trains the old game". Verified here, not assumed.
RESOLVED_ROOTS="$(cd "${REPO_ROOT}" && "${PYTHON_BIN}" - <<'CANARY_ROOT_PY'
import os

import agent
import env

for module in (agent, env):
    print(os.path.dirname(os.path.dirname(os.path.abspath(module.__file__))))
CANARY_ROOT_PY
)"
while IFS= read -r resolved; do
    [[ -n "${resolved}" ]] || continue
    if [[ "${resolved}" != "${REPO_ROOT}" ]]; then
        die "the interpreter imports this project from ${resolved}, not ${REPO_ROOT}.
      The canary would green-light a DIFFERENT checkout's code than the one the
      fleet runs. Unset PYTHONPATH, or remove the editable install shadowing
      this worktree."
    fi
done <<<"${RESOLVED_ROOTS}"
log "python resolves this project from ${REPO_ROOT}"

OUT_DIR="${OUT_DIR:-${REPO_ROOT}/runs/${RUN_NAME}/canary}"
VERDICT_MODULE="${OUT_DIR}/canary_verdict.py"
EVIDENCE_JSON="${OUT_DIR}/evidence.json"
MEASUREMENTS_JSON="${OUT_DIR}/canary_measurements.json"
PROBE_PRE_JSON="${OUT_DIR}/probe_pre.json"
PROBE_POST_JSON="${OUT_DIR}/probe_post.json"
DRIVER_LOG="${OUT_DIR}/driver.log"
CHECKPOINT_PATH="${REPO_ROOT}/runs/${RUN_NAME}.pt"
BEST_CHECKPOINT_PATH="${REPO_ROOT}/runs/${RUN_NAME}.best.pt"
PROBE_PORT=$(( BRIDGE_BASE_PORT ))

# ===========================================================================
# The verdict module. Written to disk, then imported by the collector below.
#
# It is a SEPARATE FILE, not an inline heredoc, for one reason:
# tests/test_canary_selfplay.py extracts the text between the two
# CANARY_VERDICT_PY sentinels verbatim and drives every refusal condition over
# synthetic metrics. Editing a threshold here changes what the tests assert.
# Keep it stdlib-only — it must be executable and importable with nothing but a
# Python interpreter.
# ===========================================================================
emit_verdict_module() {
    cat >"$1" <<'CANARY_VERDICT_PY'
"""canary_verdict — the T17 self-play launch gate's decision logic (AC12, TC38).

This module decides ONE thing: may the 24-hour, 25-pad self-play run start? It
is pure — it reads an evidence document (a plain dict, normally
``evidence.json``) and returns a verdict. No sockets, no torch, no filesystem
beyond reading the evidence file in :func:`main`. That is what makes every
refusal below testable offline, and testing them is the point: a canary whose
refusals do not fire is worse than no canary, because it manufactures
confidence.

WHAT EACH REFUSAL PREVENTS — the failure, not the symptom:

* ``DRIVER_FAILED`` — the training driver never printed its ``[multi done]``
  line, so it crashed or was cut off by the wall-clock deadline. Everything
  downstream would then be measured on a partial run.
  **Read the exit code note in :func:`check_driver` before touching this.**
* ``ZERO_GRAD_STEPS`` — THE headline check. The run collected data and never
  learned from it, so nothing this canary claims to have exercised was
  exercised. This is the exact hole in the naive 15-minute canary.
* ``POOL_UNREADABLE`` — ``runs/<run>/snapshots/pool.json`` could not be loaded.
  The pool IS the self-play run; without it there is no PFSP, no Elo, no
  reference eval.
* ``NO_NEW_SNAPSHOT`` — the pool never grew past the seeded snapshot 0. The
  archive cadence (T18) is not firing, so a 24-hour run would fight the frozen
  warm start every single episode, PFSP would have one candidate, and Elo could
  not move. Nothing raises when this happens; ``selfplay/pool_size`` just reads
  1 all night.
* ``SNAPSHOT_UNCHANGED`` — the pool grew but the newest snapshot's weights are
  bit-identical to snapshot 0's. That is the archive hook firing from the wrong
  source (an un-published net, or the warm-start clone) and it is precisely the
  failure T18 exists to prevent — the pool would fill with copies of one policy.
* ``STALE_POOL`` — the pool on disk is not this run's work.
  ``build_snapshot_opponents`` RELOADS a pool it finds (``SnapshotPool.load``
  restores the registry, both Elo series and the match counters), so a second
  attempt into the same ``--run-name`` inherits the first attempt's artifacts
  and ``NO_NEW_SNAPSHOT``, ``SNAPSHOT_UNCHANGED``, ``RATED_ELO_EMPTY`` and
  ``DRAW_MAJORITY_TRAINING`` all read as satisfied before this attempt has
  proved anything. A gate that stops gating on the second attempt is exactly
  the manufactured confidence this file exists to prevent.
* ``PFSP_INVALID`` — a NaN, a negative, a non-normalized or an incomplete
  probability vector. ``SnapshotPool.pfsp_weights`` is contracted to be finite
  and normalized over EVERY live snapshot; a violation means opponent sampling
  is undefined for the whole night.
* ``RATED_ELO_EMPTY`` — no match was ever scored with both sides at eps=0, so
  ``elo/learner_rated`` has no data. That series is the AC7 rising-trend
  evidence AND the checkpoint-selection input. An empty one and a flat one both
  render as a flat line; only the rated-match count tells them apart, and by
  morning it is too late.
* ``DRAW_MAJORITY_TRAINING`` / ``DRAW_MAJORITY_PROBE`` — more than half the
  matches ended at the 600-step cap. That is the degenerate mutual-stalling
  equilibrium two self-play agents converge to; PFSP goes flat, Elo pins, and
  the window is wasted.
* ``CHECKPOINT_UNLOADABLE`` — the file the demo would ship, or the newest
  snapshot the opponents would play, does not load through the shared
  ``eval.evaluate._load_drqn``. Discovered on demo day otherwise.
* ``PROBE_FAILED`` — the live seat probe did not complete, so there is no
  first-hand evidence about gear, damage, cooldowns or action diversity. Fail
  closed: absent evidence is a refusal, never a pass.
* ``MISSING_ARMOR`` — the FULLY-CHARGED hit damage does not look like an iron
  sword through full iron. See :data:`ARMORED_HIT_DAMAGE`. Armor is invisible to
  the reset gate's inventory read (``items()`` covers slots 9-44; armor is 5-8),
  so measured damage is the honest check, and it covers BOTH fighters: hits the
  learner DEALS prove the opponent's armor, hits it TAKES prove its own.
* ``NO_DAMAGE_DEALT`` / ``NO_DAMAGE_TAKEN`` — the damage channel is dead in that
  direction, or that fighter is unarmed. This project has already shipped a dead
  ``damage_dealt`` channel for its entire life, silently.
* ``ACTION_COLLAPSE_LEARNER`` / ``ACTION_COLLAPSE_OPPONENT`` — one macro
  dominates. A policy that only ever emits one action is not playing; it also
  makes every win rate and Elo number meaningless.
* ``COOLDOWN_METER_STUCK`` — a seat's attack-cooldown reading never moves, or
  leaves [0, 1]. The learner's comes from the wire; the opponent's is the env's
  shadow meter. Either being pinned means the swing gate is mis-modelled and the
  agent is optimizing against a fiction.
* ``COOLDOWN_DISAGREEMENT`` — the two seats respond differently to their OWN
  swings. Both fighters hold an iron sword, so their meters must behave the
  same; a gap means the shadow meter does not track the gate it mirrors.
* ``OPPONENT_FROZEN`` — the opponent never moved. Either the fleet was booted
  without ``DUMMY_KNOCKBACK_IMMUNE=false`` (leaving the datapack's
  knockback-resistance and the pinned ``movement_speed`` in place) or the
  ``movement_speed`` attribute-key mismatch is back. Both produce clean logs and
  a stationary punching bag. Requires BOTH the reported speed AND the walked
  path to be dead, so a broken velocity channel alone cannot false-refuse.

Every check is FAIL-CLOSED: when the evidence a check needs is missing, that
check refuses under its own code rather than being skipped. The single exception
is :func:`evaluate_gear` with ``strict=False``, used only by the fail-fast
pre-probe, where the volume-dependent checks are explicitly deferred (and say
so) because one episode cannot support them.

Owner: T17. Extracted verbatim from ``scripts/canary_selfplay.sh`` by
``tests/test_canary_selfplay.py``.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Sequence

__all__ = [
    "ARMORED_HIT_DAMAGE",
    "ATTACK_MACRO",
    "BARE_HIT_DAMAGE",
    "CheckResult",
    "DEFAULT_THRESHOLDS",
    "EVIDENCE_VERSION",
    "MAX_EPISODE_STEPS",
    "UNARMED_THROUGH_IRON_DAMAGE",
    "Verdict",
    "build_measurements",
    "evaluate_canary",
    "evaluate_gear",
    "format_report",
    "summarize_probe",
]

#: Bumped when the evidence document's shape changes incompatibly. The collector
#: stamps it; :func:`evaluate_canary` refuses a document it does not understand
#: rather than silently reading absent fields as healthy zeros.
#:
#: 2: added ``driver_started_at``, ``pool.newest_snapshot_mtime`` and
#: ``pool.newest_vs_second_newest_*`` (STALE_POOL and the second snapshot diff
#: FAIL CLOSED without them, so a version-1 document must be refused rather than
#: read), and rebuilt the ``eval`` section from the driver log because the
#: multi-arena path never writes a ``win_rate`` into ``summary.json``.
EVIDENCE_VERSION = 2

#: Episode horizon in decision steps - ``agent.contract_config.MAX_EPISODE_STEPS``
#: after T4 raised it from 400. An episode of exactly this length is a cap-hit
#: draw, not a fight. Pinned against the real constant by the test suite.
MAX_EPISODE_STEPS = 600

#: The ATTACK macro's index - ``agent.actions.Macro.ATTACK``. Pinned by the test
#: suite so this module can stay stdlib-only without the constant drifting.
ATTACK_MACRO = 5

#: Damage one FULLY-CHARGED iron sword hit does through a full iron set.
#: Derived, not guessed: Paper's ``CombatRules.getDamageAfterAbsorb`` is
#: ``f = 2 + toughness/4; g = clamp(armor - damage/f, armor*0.2, 20);
#: damage * (1 - g/25)``. Iron is 15 points at toughness 0, an iron sword is 6.0,
#: so ``g = clamp(15 - 3, 3, 20) = 12`` and the hit lands
#: ``6.0 * (1 - 12/25) = 3.12`` - a 48% reduction, ~7 hits to a 20 HP kill.
#: NOT the 2.4/~9 an earlier revision of the plan carried.
#: ``eval.combat_probe.damage_after_absorb(6.0, 15, 0.0)`` computes it, and the
#: test suite asserts this literal equals that call.
ARMORED_HIT_DAMAGE = 3.12

#: The same swing against an UNARMORED target: 6.0 straight through. If measured
#: full-charge hits cluster here, a fighter is missing its armor.
BARE_HIT_DAMAGE = 6.0

#: A bare FIST through full iron: ``1.0 * (1 - clamp(15 - 0.5, 3, 20)/25) = 0.42``
#: - ~48 hits per kill, which is why arming the dummy was blocking. Recorded so
#: the MISSING_ARMOR message can name what the operator is probably looking at.
UNARMED_THROUGH_IRON_DAMAGE = 0.42

#: A wire ``attack_cooldown`` at or above this counts as fully charged. The
#: combat probe gates on exact 1.0; this leaves room for float32 round-tripping
#: through JSON without admitting a genuinely partial swing.
#:
#: THE RUNG BELOW A FULL SWING, both ways round, because "0.85 of full damage"
#: is ambiguous and the two readings differ. ``attackStrengthTicker`` is an int
#: over a 12.5-tick recharge, so one gate tick short is
#: ``f = (11 + 0.5) / 12.5 == 0.92``. That scales the RAW weapon damage by
#: ``0.2 + 0.92^2 * 0.8 == 0.877``, and the result lands
#: ``damage_after_absorb(6.0 * 0.877, 15, 0) == 2.659`` through full iron, which
#: is ``2.659 / 3.12 == 0.852`` of what a cooled hit lands. So: ~0.88 of raw
#: damage, ~0.85 of landed damage. Both are ``eval.combat_probe``'s own pinned
#: ladder ("ticker 11   f = 0.92   lands 2.6590   ( 85%, one tick short)").
FULL_CHARGE_COOLDOWN = 0.999

#: Floats that came through JSON are compared with this slack, not ==.
_EPS = 1e-9

DEFAULT_THRESHOLDS: Dict[str, float] = {
    # A canary that took no gradient step proved nothing at all.
    "min_grad_steps": 1,
    # The pool must hold snapshot 0 plus at least one archived successor.
    "min_pool_size": 2,
    # Bit-identical weights mean the archive hook read the wrong source. Any
    # real gradient step moves parameters far above float32 noise.
    "min_snapshot_weight_delta": 1e-6,
    # elo/learner_rated needs at least one both-sides-greedy match to exist.
    "min_rated_matches": 1,
    # Above half is the mutual-stalling equilibrium, not variance.
    "max_draw_rate": 0.5,
    # One clean armored hit per episode is the "this channel is alive" bar. It
    # is deliberately NOT a skill bar - the canary detects a dead channel or an
    # unarmed fighter, not a weak policy.
    "min_damage_per_episode": ARMORED_HIT_DAMAGE,
    # A greedy policy at eps=0.05 spreads ~5% of its windows over the other
    # seven macros by exploration alone, so a fully collapsed net still reads
    # ~0.956 here. 0.90 fires on collapse and clears a healthy aggressive net.
    "max_top_macro_share": 0.90,
    "min_distinct_macros": 3,
    # A meter that never moves by this much across a whole probe is pinned.
    "min_cooldown_range": 0.10,
    # Both fighters hold an iron sword: their post-swing drop rates must agree.
    "max_cooldown_drop_rate_gap": 0.50,
    # Below these, a check has no evidence and refuses rather than guessing.
    "min_attack_windows_per_seat": 5,
    "min_full_charge_hits": 3,
    # A SAMPLE-SIZE floor, per episode rather than flat, and derived from the
    # shortest armored kill this fight can produce: ceil(20 HP / 3.12) == 7
    # landed hits, spaced eval.combat_probe's
    # WINDOWS_PER_SWING == ceil(12.5 / ACTION_REPEAT) == ceil(12.5 / 4) == 4
    # decision windows apart, so 7 * 4 == 28 windows is the floor of a HEALTHY
    # armored episode. The flat 100 this replaces worked out at 34 windows per
    # episode at the default --probe-episodes 3 - above that best case, so a
    # healthy fleet whose armored episodes turned out short would have been
    # refused on a statistics floor rather than on a defect. That is the one
    # threshold whose calibration depended on the number this canary exists to
    # measure for the first time, which is precisely how a gate refuses good
    # fleets.
    "min_probe_windows_per_episode": 28,
    # Walking is ~0.216 blocks/tick (MAX_SPEED normalizes by 1.0), knockback far
    # more. 0.05 is clear of numerical noise and far below a walk.
    "min_opponent_speed": 0.05,
    # ... and the independent corroboration: blocks of horizontal path walked
    # over the whole probe.
    "min_opponent_path": 1.0,
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
    """The gate's answer: every check, the refusals, and the T19 measurements."""

    checks: List[CheckResult]
    measurements: Dict[str, Any]

    @property
    def refusals(self) -> List[CheckResult]:
        """Only the failing checks, in evaluation order."""
        return [c for c in self.checks if not c.passed]

    @property
    def ok(self) -> bool:
        """True iff nothing blocks the long run."""
        return not self.refusals


# ---------------------------------------------------------------------------
# Small helpers. Everything that touches the evidence document goes through
# these, so a missing or wrong-typed field can never be read as a healthy zero.
# ---------------------------------------------------------------------------


def _thresholds(overrides: Optional[Mapping[str, Any]]) -> Dict[str, float]:
    merged = dict(DEFAULT_THRESHOLDS)
    if overrides:
        for key, value in overrides.items():
            if key in merged:
                merged[key] = float(value)
    return merged


def _num(value: Any) -> Optional[float]:
    """Coerce to a FINITE float, or None. NaN and Inf are None, not values."""
    if isinstance(value, bool) or value is None:
        return None
    if not isinstance(value, (int, float)):
        return None
    out = float(value)
    return out if math.isfinite(out) else None


def _section(evidence: Mapping[str, Any], name: str) -> Dict[str, Any]:
    """Return a mapping section, or an empty dict — never a non-mapping."""
    value = evidence.get(name)
    return dict(value) if isinstance(value, Mapping) else {}


# ---------------------------------------------------------------------------
# Probe summarization.
#
# The live probe (phase 1/3 of the shell script) is deliberately dumb I/O: it
# records one raw record per decision window and does no arithmetic. ALL of the
# aggregation lives here so that every number the gate reasons about is produced
# by tested code rather than by a heredoc nobody can run offline.
#
# Raw probe document:
#   {"ok": bool, "error": str|None,
#    "checkpoint": str, "learner_epsilon": f, "opponent_epsilon": f,
#    "max_episode_steps": int,
#    "episodes": [{"length": int, "outcome": "win"|"loss"|"timeout",
#                  "windows": [{"a": int,    learner macro
#                               "oa": int,   opponent macro
#                               "cd": f,     learner attack_cooldown, pre-action
#                               "ocd": f,    opponent shadow meter, pre-action
#                               "dd": f,     damage dealt in this transition
#                               "dt": f,     damage taken in this transition
#                               "osp": f,    opponent horizontal speed
#                               "opx"/"opz": f  opponent world position
#                              }, ...]}]}
# ---------------------------------------------------------------------------


def _seat_summary(
    windows: Sequence[Mapping[str, Any]],
    episode_bounds: Sequence[Sequence[int]],
    action_key: str,
    cooldown_key: str,
) -> Dict[str, Any]:
    """Aggregate one seat's macros and cooldown behaviour over the whole probe.

    ``episode_bounds`` is the list of ``[start, stop)`` index ranges into
    ``windows``, so the "did the meter drop on the NEXT window" comparison never
    straddles an episode boundary (where the meter legitimately resets to 1.0
    and a reset would otherwise be counted as a swing response).
    """
    counts: Dict[int, int] = {}
    values: List[float] = []
    out_of_range = 0
    attack_windows = 0
    drops = 0

    for record in windows:
        action = record.get(action_key)
        if isinstance(action, bool) or not isinstance(action, int):
            continue
        counts[action] = counts.get(action, 0) + 1

    for start, stop in episode_bounds:
        for index in range(start, stop):
            cooldown = _num(windows[index].get(cooldown_key))
            if cooldown is None:
                continue
            values.append(cooldown)
            if cooldown < -_EPS or cooldown > 1.0 + _EPS:
                out_of_range += 1
        for index in range(start, stop - 1):
            if windows[index].get(action_key) != ATTACK_MACRO:
                continue
            here = _num(windows[index].get(cooldown_key))
            nxt = _num(windows[index + 1].get(cooldown_key))
            if here is None or nxt is None:
                continue
            attack_windows += 1
            if nxt < here - 1e-6:
                drops += 1

    total = sum(counts.values())
    top_action, top_count = (None, 0)
    for action, count in sorted(counts.items()):
        if count > top_count:
            top_action, top_count = action, count

    return {
        "action_counts": {str(k): v for k, v in sorted(counts.items())},
        "windows": total,
        "top_macro": top_action,
        "top_macro_share": (top_count / total) if total else None,
        "distinct_macros": len(counts),
        "attack_windows": attack_windows,
        "cooldown_drops": drops,
        "cooldown_drop_rate": (drops / attack_windows) if attack_windows else None,
        "cooldown_min": min(values) if values else None,
        "cooldown_max": max(values) if values else None,
        "cooldown_range": (max(values) - min(values)) if values else None,
        "cooldown_out_of_range": out_of_range,
    }


def _classify_hits(values: Sequence[float]) -> Dict[str, Any]:
    """Bucket fully-charged hit damages against the three loadout hypotheses.

    ``iron_like`` is a sword through full iron (3.12), ``bare_like`` is the same
    sword through nothing (6.0). Anything else — the clamped killing blow, a
    critical (an iron-sword crit through iron lands ~5.22), a fist (0.42) — falls
    in ``other`` and is reported rather than forced into a bucket.
    """
    tolerance = 0.30
    iron_like = sum(1 for v in values if abs(v - ARMORED_HIT_DAMAGE) <= tolerance)
    bare_like = sum(1 for v in values if abs(v - BARE_HIT_DAMAGE) <= tolerance)
    return {
        "count": len(values),
        "iron_like": iron_like,
        "bare_like": bare_like,
        "other": len(values) - iron_like - bare_like,
        "median": float(statistics.median(values)) if values else None,
        "values": [float(v) for v in values],
    }


def summarize_probe(probe: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Reduce a raw probe document to the numbers the gate reasons about.

    Returns a dict that always carries ``"ok"``. A probe that did not run, or
    that raised, yields ``{"ok": False, "error": ...}`` and every gear check
    then refuses under ``PROBE_FAILED`` — absent evidence never passes.
    """
    if not isinstance(probe, Mapping):
        return {"ok": False, "error": "no probe document"}
    if probe.get("ok") is not True:
        return {
            "ok": False,
            "error": str(probe.get("error") or "probe reported ok=false"),
        }

    raw_episodes = probe.get("episodes")
    if not isinstance(raw_episodes, Sequence) or not raw_episodes:
        return {"ok": False, "error": "probe recorded no episodes"}

    cap = int(probe.get("max_episode_steps") or MAX_EPISODE_STEPS)

    windows: List[Mapping[str, Any]] = []
    bounds: List[List[int]] = []
    lengths: List[int] = []
    outcomes: Dict[str, int] = {"win": 0, "loss": 0, "timeout": 0}
    cap_hits = 0
    dealt_total = 0.0
    taken_total = 0.0
    dealt_hits: List[float] = []
    taken_hits: List[float] = []
    max_speed = 0.0
    path_length = 0.0

    for episode in raw_episodes:
        if not isinstance(episode, Mapping):
            return {"ok": False, "error": "probe episode is not an object"}
        episode_windows = episode.get("windows")
        if not isinstance(episode_windows, Sequence):
            return {"ok": False, "error": "probe episode carries no windows"}
        start = len(windows)
        prev_xz: Optional[Sequence[float]] = None
        for record in episode_windows:
            if not isinstance(record, Mapping):
                return {"ok": False, "error": "probe window is not an object"}
            windows.append(record)

            dealt = _num(record.get("dd")) or 0.0
            taken = _num(record.get("dt")) or 0.0
            dealt_total += dealt
            taken_total += taken

            # Only FULLY-CHARGED swings carry the loadout signature. The
            # charge scalar (0.2 + f^2 * 0.8) multiplies the RAW weapon damage
            # BEFORE absorption runs, and absorption is not linear in the
            # incoming damage, so the scalar can NEVER be applied over the
            # post-armor 3.12. At f = 0.90 the swing lands
            #   damage_after_absorb(6.0 * 0.848, 15, 0)
            #     = 5.088 * (1 - 12.456/25) = 2.553
            # through full iron - not the 3.12 * 0.848 = 2.646 a flat
            # percentage predicts, which is the exact error
            # eval.combat_probe.damage_after_absorb's docstring calls out.
            # Either way it sits outside the +/-0.30 iron band, so counting it
            # would read as "not iron" and refuse a perfectly geared fleet.
            cooldown = _num(record.get("cd"))
            if (
                record.get("a") == ATTACK_MACRO
                and cooldown is not None
                and cooldown >= FULL_CHARGE_COOLDOWN
                and dealt > 0.0
            ):
                dealt_hits.append(dealt)
            opp_cooldown = _num(record.get("ocd"))
            if (
                record.get("oa") == ATTACK_MACRO
                and opp_cooldown is not None
                and opp_cooldown >= FULL_CHARGE_COOLDOWN
                and taken > 0.0
            ):
                taken_hits.append(taken)

            speed = _num(record.get("osp"))
            if speed is not None:
                max_speed = max(max_speed, abs(speed))
            pos_x, pos_z = _num(record.get("opx")), _num(record.get("opz"))
            if pos_x is not None and pos_z is not None:
                if prev_xz is not None:
                    path_length += math.hypot(pos_x - prev_xz[0], pos_z - prev_xz[1])
                prev_xz = (pos_x, pos_z)
        bounds.append([start, len(windows)])

        length = int(episode.get("length") or (len(windows) - start))
        lengths.append(length)
        if length >= cap:
            cap_hits += 1
        outcome = str(episode.get("outcome") or "timeout")
        outcomes[outcome] = outcomes.get(outcome, 0) + 1

    n_episodes = len(lengths)
    return {
        "ok": True,
        "error": None,
        "checkpoint": probe.get("checkpoint"),
        "learner_epsilon": _num(probe.get("learner_epsilon")),
        "opponent_epsilon": _num(probe.get("opponent_epsilon")),
        "max_episode_steps": cap,
        "episodes": n_episodes,
        "windows": len(windows),
        "episode_lengths": lengths,
        "mean_episode_length": statistics.fmean(lengths) if lengths else None,
        "median_episode_length": (
            float(statistics.median(lengths)) if lengths else None
        ),
        "outcomes": outcomes,
        "cap_hits": cap_hits,
        "cap_hit_rate": (cap_hits / n_episodes) if n_episodes else None,
        "damage_dealt_total": dealt_total,
        "damage_taken_total": taken_total,
        "damage_dealt_per_episode": (dealt_total / n_episodes) if n_episodes else None,
        "damage_taken_per_episode": (taken_total / n_episodes) if n_episodes else None,
        "learner": _seat_summary(windows, bounds, "a", "cd"),
        "opponent": dict(
            _seat_summary(windows, bounds, "oa", "ocd"),
            max_speed=max_speed,
            path_length=path_length,
        ),
        "hits_dealt": _classify_hits(dealt_hits),
        "hits_taken": _classify_hits(taken_hits),
    }


# ---------------------------------------------------------------------------
# The gear/fleet checks - everything the live probe can see.
# ---------------------------------------------------------------------------


def _armor_check(
    summary: Mapping[str, Any], limits: Mapping[str, float], strict: bool
) -> CheckResult:
    """MISSING_ARMOR, from the measured FULL-CHARGE hit damage in BOTH directions.

    Hits the learner DEALS prove the OPPONENT's armor; hits it TAKES prove its
    own. That two-sided read is why this check is worth more than the reset
    gate's inventory probe, which cannot see armor at all (``items()`` covers
    slots 9-44; armor sits in 5-8).
    """
    minimum = int(limits["min_full_charge_hits"])
    verdicts: List[str] = []
    thin: List[str] = []
    for label, key in (("dealt", "hits_dealt"), ("taken", "hits_taken")):
        hits = summary.get(key) or {}
        count = int(hits.get("count") or 0)
        if count < minimum:
            thin.append(f"{label}: only {count} fully-charged hit(s)")
            continue
        iron_like = int(hits.get("iron_like") or 0)
        bare_like = int(hits.get("bare_like") or 0)
        median = hits.get("median")
        if bare_like >= iron_like or iron_like == 0:
            verdicts.append(
                f"{label}: {count} fully-charged hits, median "
                f"{median!r} HP ({iron_like} look like {ARMORED_HIT_DAMAGE} "
                f"through full iron, {bare_like} like {BARE_HIT_DAMAGE} through "
                f"nothing)"
            )
    if verdicts:
        return CheckResult(
            "MISSING_ARMOR",
            False,
            "; ".join(verdicts),
            "fully-charged hit damage does not look like an iron sword through "
            "full iron: " + "; ".join(verdicts),
            f"expected {ARMORED_HIT_DAMAGE} HP per fully-cooled hit (~7 hits to "
            f"a 20 HP kill). {BARE_HIT_DAMAGE} means that fighter has NO armor; "
            f"~{UNARMED_THROUGH_IRON_DAMAGE} means the attacker has no sword. "
            "Check spawn_learner_pad/spawn_dummy_pad.mcfunction actually ran "
            "(a bad item id voids a $-macro function ENTIRELY, at invocation, "
            "with nothing in the boot log), then bot.js's dummyResetTemplate "
            "and the server-authoritative armor read-back.",
        )
    if thin:
        detail = "; ".join(thin)
        if strict:
            return CheckResult(
                "MISSING_ARMOR",
                False,
                detail,
                f"not enough fully-charged hits to prove either fighter's gear "
                f"({detail}); a canary that cannot see the loadout must not "
                "clear a 24-hour run",
                "raise --probe-episodes, or find out why the fighters are not "
                "landing cooled swings at all (a stuck cooldown meter, or a "
                "policy that never attacks).",
            )
        return CheckResult(
            "MISSING_ARMOR", True, f"deferred to the post-run probe ({detail})"
        )
    return CheckResult(
        "MISSING_ARMOR",
        True,
        f"dealt {summary.get('hits_dealt', {}).get('iron_like')} / taken "
        f"{summary.get('hits_taken', {}).get('iron_like')} fully-charged hits "
        f"consistent with {ARMORED_HIT_DAMAGE} HP through full iron",
    )


def _damage_check(
    summary: Mapping[str, Any], limits: Mapping[str, float], strict: bool, key: str
) -> CheckResult:
    """NO_DAMAGE_DEALT / NO_DAMAGE_TAKEN — is the channel alive in this direction?

    Deliberately a "channel is alive" bar, not a skill bar: one clean armored hit
    per episode. The failure it catches is a dead events channel or an unarmed
    fighter, both of which read as a perfectly healthy training curve.
    """
    code = "NO_DAMAGE_DEALT" if key == "dealt" else "NO_DAMAGE_TAKEN"
    direction = "the learner dealt" if key == "dealt" else "the learner took"
    value = _num(summary.get(f"damage_{key}_per_episode"))
    # The pre-probe cannot support a per-episode threshold off one episode, so it
    # only asks whether ANY damage moved in this direction at all.
    floor = float(limits["min_damage_per_episode"]) if strict else 0.0
    if value is None:
        return CheckResult(
            code,
            False,
            "no damage figure recorded",
            f"the probe recorded no damage_{key} at all, so the channel cannot "
            "be shown to work",
            "check env.MCPvPEnv.step's info['events'] and the bridge's damage "
            "reporting; this project has already shipped a silently dead "
            "damage_dealt channel for its entire life.",
        )
    if value <= floor + _EPS:
        return CheckResult(
            code,
            False,
            f"{value:.3f} HP/episode",
            f"{direction} {value:.3f} HP per episode, at or below the "
            f"{floor:.3f} floor - the damage channel is dead in that direction, "
            "or that fighter is unarmed",
            f"one fully-cooled iron-sword hit through full iron is "
            f"{ARMORED_HIT_DAMAGE} HP. Verify the sword is EQUIPPED (the "
            "datapack's $give arms the dummy; 'give' alone does not equip "
            "armor), then that events.damage_* is populated from the right "
            "connection.",
        )
    return CheckResult(code, True, f"{value:.3f} HP/episode {direction}")


def _draw_check(summary: Mapping[str, Any], limits: Mapping[str, float]) -> CheckResult:
    """DRAW_MAJORITY_PROBE — the mutual-stalling equilibrium, seen live."""
    rate = _num(summary.get("cap_hit_rate"))
    cap = summary.get("max_episode_steps")
    if rate is None:
        return CheckResult(
            "DRAW_MAJORITY_PROBE",
            False,
            "no episode outcomes recorded",
            "the probe recorded no episodes, so the draw rate is unknown",
            "re-run the probe; an empty probe cannot clear a 24-hour run.",
        )
    if rate > float(limits["max_draw_rate"]) + _EPS:
        return CheckResult(
            "DRAW_MAJORITY_PROBE",
            False,
            f"{rate:.0%} of probe episodes hit the {cap}-step cap",
            f"{rate:.0%} of probe episodes ended at the {cap}-step cap instead "
            "of a death - the degenerate mutual-stalling equilibrium two "
            "self-play agents converge to",
            "check reward_config c_aim < c_step (issue #25), that both "
            "fighters are actually armed, and whether the armored regime needs "
            "a larger MAX_EPISODE_STEPS or a stronger terminal signal. A night "
            "of draws leaves PFSP flat and Elo pinned.",
        )
    return CheckResult(
        "DRAW_MAJORITY_PROBE", True, f"{rate:.0%} cap-hit draws (limit 50%)"
    )


def _diversity_check(
    summary: Mapping[str, Any], limits: Mapping[str, float], seat: str
) -> CheckResult:
    """ACTION_COLLAPSE_* — one macro dominating is not a policy."""
    code = f"ACTION_COLLAPSE_{seat.upper()}"
    seat_summary = summary.get(seat) or {}
    share = _num(seat_summary.get("top_macro_share"))
    distinct = seat_summary.get("distinct_macros")
    if share is None or not isinstance(distinct, int):
        return CheckResult(
            code,
            False,
            "no action histogram recorded",
            f"the probe recorded no macros for the {seat} seat, so action "
            "diversity cannot be checked",
            "re-run the probe; an unmeasured policy cannot clear a launch.",
        )
    if share > float(limits["max_top_macro_share"]) + _EPS:
        return CheckResult(
            code,
            False,
            f"macro {seat_summary.get('top_macro')} is {share:.0%} of windows",
            f"the {seat} seat emitted macro {seat_summary.get('top_macro')} in "
            f"{share:.0%} of its decision windows - the policy has collapsed "
            "onto one action",
            "a collapsed policy makes every win rate and Elo number "
            "meaningless. Check the warm start, the epsilon schedule, and "
            "whether the Q-values have saturated.",
        )
    if distinct < int(limits["min_distinct_macros"]):
        return CheckResult(
            code,
            False,
            f"only {distinct} distinct macro(s)",
            f"the {seat} seat used only {distinct} distinct macro(s) across the "
            "whole probe",
            "with epsilon > 0 even a collapsed net samples several macros, so "
            "this low a count points at the action plumbing, not the policy.",
        )
    return CheckResult(
        code,
        True,
        f"top macro {seat_summary.get('top_macro')} at {share:.0%}, "
        f"{distinct} distinct macros",
    )


def _cooldown_checks(
    summary: Mapping[str, Any], limits: Mapping[str, float], strict: bool
) -> List[CheckResult]:
    """COOLDOWN_METER_STUCK and COOLDOWN_DISAGREEMENT, over the two seats.

    The learner's ``attack_cooldown`` comes off the wire; the opponent's is the
    env's SHADOW meter, reconstructed from observed swings. Both fighters carry
    an iron sword, so a swing must visibly discharge both meters at the same
    rate. A gap means the shadow meter is not tracking the gate it mirrors, and
    the frozen opponent is then swinging against a fiction.
    """
    results: List[CheckResult] = []
    seats = {
        "learner": summary.get("learner") or {},
        "opponent": summary.get("opponent") or {},
    }

    stuck: List[str] = []
    thin: List[str] = []
    for name, seat in seats.items():
        spread = _num(seat.get("cooldown_range"))
        out_of_range = int(seat.get("cooldown_out_of_range") or 0)
        attacks = int(seat.get("attack_windows") or 0)
        if out_of_range:
            stuck.append(f"{name}: {out_of_range} reading(s) outside [0, 1]")
            continue
        if attacks < int(limits["min_attack_windows_per_seat"]):
            thin.append(f"{name}: only {attacks} ATTACK window(s)")
            continue
        if spread is None or spread < float(limits["min_cooldown_range"]):
            stuck.append(f"{name}: meter spans only {spread!r} across the probe")

    if stuck:
        results.append(
            CheckResult(
                "COOLDOWN_METER_STUCK",
                False,
                "; ".join(stuck),
                "an attack-cooldown meter never moved, or left [0, 1]: "
                + "; ".join(stuck),
                "the learner's meter is state.self.attack_cooldown off the "
                "wire; the opponent's is MCPvPEnv._opponent_attack_cooldown, a "
                "shadow reconstructed from observed swings. A pinned meter "
                "means the agent is optimizing against a swing gate that does "
                "not exist.",
            )
        )
    elif thin and strict:
        results.append(
            CheckResult(
                "COOLDOWN_METER_STUCK",
                False,
                "; ".join(thin),
                "not enough ATTACK windows to exercise the cooldown meters ("
                + "; ".join(thin)
                + ")",
                "raise --probe-episodes, or find out why a fighter never "
                "swings.",
            )
        )
    elif thin:
        results.append(
            CheckResult(
                "COOLDOWN_METER_STUCK", True, "deferred (" + "; ".join(thin) + ")"
            )
        )
    else:
        results.append(
            CheckResult(
                "COOLDOWN_METER_STUCK",
                True,
                "learner meter spans "
                f"{seats['learner'].get('cooldown_range')!r}, opponent "
                f"{seats['opponent'].get('cooldown_range')!r}",
            )
        )

    learner_rate = _num(seats["learner"].get("cooldown_drop_rate"))
    opponent_rate = _num(seats["opponent"].get("cooldown_drop_rate"))
    if learner_rate is None or opponent_rate is None:
        message = (
            f"learner={learner_rate!r} opponent={opponent_rate!r} post-swing "
            "drop rate"
        )
        if strict:
            results.append(
                CheckResult(
                    "COOLDOWN_DISAGREEMENT",
                    False,
                    message,
                    "one or both seats never swung, so the two cooldown meters "
                    f"could not be compared ({message})",
                    "raise --probe-episodes; the seats must be compared on real "
                    "swings, not assumed equal.",
                )
            )
        else:
            results.append(
                CheckResult("COOLDOWN_DISAGREEMENT", True, f"deferred ({message})")
            )
        return results

    gap = abs(learner_rate - opponent_rate)
    if gap > float(limits["max_cooldown_drop_rate_gap"]) + _EPS:
        results.append(
            CheckResult(
                "COOLDOWN_DISAGREEMENT",
                False,
                f"learner {learner_rate:.2f} vs opponent {opponent_rate:.2f} "
                f"(gap {gap:.2f})",
                "the two seats respond differently to their own swings: the "
                f"learner's meter dropped after {learner_rate:.0%} of its "
                f"ATTACK windows, the opponent's after {opponent_rate:.0%} "
                f"(gap {gap:.2f})",
                "both fighters hold an iron sword, so the rates must match. "
                "Re-audit MCPvPEnv._note_opponent_swing / "
                "_opponent_attack_cooldown against the bridge's "
                "MacroExecutor.canSwing gate - the shadow rides decision "
                "windows, the gate rides ticks, and they are only equal by "
                "construction.",
            )
        )
    else:
        results.append(
            CheckResult(
                "COOLDOWN_DISAGREEMENT",
                True,
                f"post-swing drop rate learner {learner_rate:.2f} vs opponent "
                f"{opponent_rate:.2f}",
            )
        )
    return results


def _frozen_opponent_check(
    summary: Mapping[str, Any], limits: Mapping[str, float]
) -> CheckResult:
    """OPPONENT_FROZEN — the recorded gotcha that produces clean logs and no fight.

    Refuses only when BOTH the reported speed AND the walked path are dead. One
    signal alone would false-refuse on a broken velocity channel (a real risk
    here: mineflayer's non-self entity velocity is unreliable on pinned 1.21.1,
    which is why the opponent's velocity is read from its OWN connection).
    """
    seat = summary.get("opponent") or {}
    speed = _num(seat.get("max_speed"))
    path = _num(seat.get("path_length"))
    if speed is None or path is None:
        return CheckResult(
            "OPPONENT_FROZEN",
            False,
            f"speed={speed!r} path={path!r}",
            "the probe recorded no opponent motion data, so a frozen opponent "
            "cannot be ruled out",
            "the probe reads the opponent's own velocity from the mirrored "
            "observation and its world position from raw_opponent_view(); both "
            "must be present.",
        )
    if speed < float(limits["min_opponent_speed"]) and path < float(
        limits["min_opponent_path"]
    ):
        return CheckResult(
            "OPPONENT_FROZEN",
            False,
            f"max horizontal speed {speed:.4f}, path walked {path:.2f} blocks",
            f"the opponent never moved (peak horizontal speed {speed:.4f} "
            f"blocks/tick, {path:.2f} blocks walked across the whole probe)",
            "boot the fleet with DUMMY_KNOCKBACK_IMMUNE=false - without it the "
            "datapack's knockback-resistance and pinned movement_speed stay in "
            "place and the opponent is a punching bag. If the flag WAS set, "
            "re-check the 1.21.1 movement_speed attribute-key mismatch that an "
            "npm update reintroduces silently.",
        )
    return CheckResult(
        "OPPONENT_FROZEN",
        True,
        f"peak horizontal speed {speed:.3f} blocks/tick, {path:.1f} blocks walked",
    )


def evaluate_gear(
    summary: Optional[Mapping[str, Any]],
    thresholds: Optional[Mapping[str, Any]] = None,
    *,
    strict: bool = True,
) -> List[CheckResult]:
    """Run every check the live probe can answer.

    Args:
        summary: A :func:`summarize_probe` result.
        thresholds: Optional overrides of :data:`DEFAULT_THRESHOLDS`.
        strict: ``True`` for the post-run probe — every check must be answerable
            or it refuses. ``False`` for the fail-fast pre-probe, where the
            volume-dependent checks (draw rate, action diversity, and the
            thin-evidence arms of the gear/cooldown checks) are explicitly
            DEFERRED and say so, because one episode cannot support them.

    Returns:
        Every check's :class:`CheckResult`, in evaluation order.
    """
    limits = _thresholds(thresholds)
    if not isinstance(summary, Mapping) or summary.get("ok") is not True:
        error = "probe did not run"
        if isinstance(summary, Mapping):
            error = str(summary.get("error") or error)
        return [
            CheckResult(
                "PROBE_FAILED",
                False,
                error,
                f"the live seat probe did not complete: {error}",
                "without it there is no first-hand evidence about gear, "
                "damage, cooldowns, action diversity or opponent motion. Check "
                "that the fleet is up, that nothing else is attached to the "
                "bridge port (BridgeServer accepts exactly ONE client and a "
                "second connection destroys the first), and re-run.",
            )
        ]

    windows = int(summary.get("windows") or 0)
    episodes = int(summary.get("episodes") or 0)
    per_episode = int(limits["min_probe_windows_per_episode"])
    floor = per_episode * max(1, episodes)
    if strict and windows < floor:
        return [
            CheckResult(
                "PROBE_FAILED",
                False,
                f"{windows} decision window(s) over {episodes} episode(s), "
                f"below the {floor}-window floor",
                "THIS IS A SAMPLE-SIZE FLOOR, NOT A DEFECT REPORT: the probe "
                f"recorded {windows} decision windows over {episodes} "
                f"episode(s), under the {floor} its statistics need",
                "raise --probe-episodes and re-run FIRST. The floor is "
                f"{per_episode} windows per episode - the SHORTEST armored kill "
                "there is (7 landed hits at 4 windows' spacing), so a genuinely "
                "short but healthy fight lands here too. Only if longer probes "
                "stay short is something wrong, and then the question is why "
                "episodes end immediately.",
            )
        ]

    results = [CheckResult("PROBE_FAILED", True, f"{windows} decision windows recorded")]
    results.append(_armor_check(summary, limits, strict))
    results.append(_damage_check(summary, limits, strict, "dealt"))
    results.append(_damage_check(summary, limits, strict, "taken"))
    results.append(_frozen_opponent_check(summary, limits))
    results.extend(_cooldown_checks(summary, limits, strict))
    if strict:
        results.append(_draw_check(summary, limits))
        results.append(_diversity_check(summary, limits, "learner"))
        results.append(_diversity_check(summary, limits, "opponent"))
    else:
        results.append(
            CheckResult("DRAW_MAJORITY_PROBE", True, "deferred to the post-run probe")
        )
        results.append(
            CheckResult(
                "ACTION_COLLAPSE_LEARNER", True, "deferred to the post-run probe"
            )
        )
        results.append(
            CheckResult(
                "ACTION_COLLAPSE_OPPONENT", True, "deferred to the post-run probe"
            )
        )
    return results


# ---------------------------------------------------------------------------
# The run-level checks - everything the training run's artifacts can answer.
# ---------------------------------------------------------------------------


def check_driver(evidence: Mapping[str, Any]) -> CheckResult:
    """DRIVER_FAILED — did the training driver finish its own loop?

    !! DO NOT GATE THIS ON THE PROCESS EXIT CODE. !!
    ``agent.train._main_multi_arena`` returns ``0 if result.passed_m2 else 1``,
    and ``passed_m2`` is the M2 gate (win_rate >= 0.95 vs the STATIONARY dummy).
    A self-play run fighting a past self will essentially never clear it, so a
    healthy canary exits 1. Gating on the exit code would refuse every good run.
    The real signal is the ``[multi done]`` line the driver prints from its own
    teardown, plus the wall-clock deadline.
    """
    driver = _section(evidence, "driver")
    if driver.get("deadline_hit") is True:
        return CheckResult(
            "DRIVER_FAILED",
            False,
            "wall-clock deadline reached",
            "the training driver was still running when the canary's wall-clock "
            "deadline expired and had to be interrupted",
            "raise --deadline-minutes, lower --max-grad-steps, or find out why "
            "the learner is slower than the measured ~4,570 grad steps/hour.",
        )
    if driver.get("completed") is not True:
        return CheckResult(
            "DRIVER_FAILED",
            False,
            f"no [multi done] line (exit code {driver.get('exit_code')!r})",
            "the training driver never printed its [multi done] teardown line, "
            "so it crashed or was killed rather than finishing its loop",
            f"read {driver.get('log_path')!r}. NOTE: a nonzero exit code alone "
            "is NORMAL here - _main_multi_arena returns 1 whenever passed_m2 is "
            "False, and a self-play run does not clear the M2 dummy gate.",
        )
    return CheckResult(
        "DRIVER_FAILED",
        True,
        f"completed: reason={driver.get('stop_reason')!r} "
        f"episodes={driver.get('episodes')} exit={driver.get('exit_code')} "
        "(exit 1 is normal - passed_m2 is the M2 dummy gate)",
    )


def check_grad_steps(
    evidence: Mapping[str, Any], limits: Mapping[str, float]
) -> CheckResult:
    """ZERO_GRAD_STEPS — the headline check this whole file exists for."""
    driver = _section(evidence, "driver")
    steps = driver.get("grad_steps")
    if not isinstance(steps, int) or isinstance(steps, bool):
        return CheckResult(
            "ZERO_GRAD_STEPS",
            False,
            f"grad_steps={steps!r}",
            "the run's gradient-step count could not be read, so it cannot be "
            "shown to have learned anything",
            "read the [multi done] line in the driver log.",
        )
    floor = int(limits["min_grad_steps"])
    if steps < floor:
        return CheckResult(
            "ZERO_GRAD_STEPS",
            False,
            f"{steps} gradient steps",
            f"the run took {steps} gradient steps (floor {floor}) - it collected "
            "data and never learned from it, so no snapshot, PFSP draw, Elo "
            "update or eval was exercised on anything but the warm start",
            "the replay warm-up is the usual cause: Trainer.learn() is a no-op "
            f"below --min-replay ({evidence.get('min_replay')!r} here, "
            f"{evidence.get('min_replay_production')!r} in production). At "
            "4.8782 transitions/s/arena, filling 25,000 takes 21.4 minutes at 4 "
            "pads. Lower --min-replay or run longer.",
        )
    return CheckResult("ZERO_GRAD_STEPS", True, f"{steps} gradient steps")


def check_pool(
    evidence: Mapping[str, Any], limits: Mapping[str, float]
) -> List[CheckResult]:
    """POOL_UNREADABLE / NO_NEW_SNAPSHOT / SNAPSHOT_UNCHANGED / PFSP_INVALID /
    RATED_ELO_EMPTY / DRAW_MAJORITY_TRAINING — everything the pool index says."""
    pool = _section(evidence, "pool")
    if pool.get("ok") is not True:
        error = str(pool.get("error") or "pool.json could not be read")
        blocked = CheckResult(
            "POOL_UNREADABLE",
            False,
            error,
            f"the snapshot pool could not be loaded: {error}",
            "the pool IS the self-play run - without it there is no PFSP, no "
            "Elo and no reference eval. Check "
            "runs/<run>/snapshots/pool.json and that "
            "SnapshotPool.load(sampling=...) was given the run's sampling mode.",
        )
        return [blocked]

    results = [
        CheckResult(
            "POOL_UNREADABLE",
            True,
            f"loaded {pool.get('size')} snapshot(s), sampling="
            f"{pool.get('sampling')!r}",
        )
    ]

    size = pool.get("size")
    if not isinstance(size, int) or size < int(limits["min_pool_size"]):
        results.append(
            CheckResult(
                "NO_NEW_SNAPSHOT",
                False,
                f"pool holds {size!r} snapshot(s)",
                f"the pool holds {size!r} snapshot(s): the archive cadence never "
                "produced a successor to the seeded snapshot 0",
                "this is the T18 failure. Without SnapshotPool.add being called "
                "on cadence from the PUBLISHED weights, a 24-hour run fights the "
                "frozen warm start every episode, PFSP has one candidate and Elo "
                "cannot move - and nothing raises. Check "
                "SnapshotArchivist.maybe_archive and "
                "--snapshot-every-grad-steps against the run's grad-step count.",
            )
        )
    else:
        results.append(
            CheckResult("NO_NEW_SNAPSHOT", True, f"pool grew to {size} snapshots")
        )

    # TWO diffs, not one. Against snapshot 0 alone, a pool whose snapshots
    # 1..N are bit-identical clones of EACH OTHER still passes as long as they
    # differ from the seed - the same "the pool fills with copies of one policy"
    # failure, just one archive cycle later. The newest-vs-second-newest diff is
    # what catches an archive hook that froze after its first successful write.
    unchanged: List[str] = []
    undiffable: List[str] = []
    cleared: List[str] = []
    for label, delta_key, error_key in (
        ("snapshot 0", "newest_vs_snapshot0_max_abs_delta",
         "newest_vs_snapshot0_error"),
        ("the second-newest snapshot", "newest_vs_second_newest_max_abs_delta",
         "newest_vs_second_newest_error"),
    ):
        delta = _num(pool.get(delta_key))
        delta_error = pool.get(error_key)
        if delta_error or delta is None:
            undiffable.append(
                f"vs {label}: delta={delta!r} "
                f"error={delta_error or 'no delta recorded'!r}"
            )
        elif delta < float(limits["min_snapshot_weight_delta"]):
            unchanged.append(f"vs {label}: max |delta| {delta:.3e}")
        else:
            cleared.append(f"vs {label}: max |delta| {delta:.3e}")

    if unchanged:
        results.append(
            CheckResult(
                "SNAPSHOT_UNCHANGED",
                False,
                "; ".join(unchanged),
                "the newest snapshot's weights are indistinguishable from an "
                "earlier snapshot's (" + "; ".join(unchanged) + ") - they are "
                "the same policy under two ids",
                "the archive hook is reading the wrong source. It must freeze "
                "the PUBLISHED weights (the stamped weight store), never the "
                "warm-start clone and never trainer.online by reference. A pool "
                "of identical snapshots makes PFSP, Elo and the reference eval "
                "all measure nothing. A delta that is dead against the "
                "second-newest but alive against snapshot 0 means the hook "
                "stopped after its first write.",
            )
        )
    elif undiffable:
        results.append(
            CheckResult(
                "SNAPSHOT_UNCHANGED",
                False,
                "; ".join(undiffable),
                "the newest snapshot's weights could not be compared against an "
                "earlier one (" + "; ".join(undiffable) + ")",
                "a snapshot that cannot be diffed cannot be shown to be a NEW "
                "policy version. Check both files load and share a parameter "
                "set.",
            )
        )
    else:
        results.append(
            CheckResult("SNAPSHOT_UNCHANGED", True, "; ".join(cleared))
        )

    results.append(_stale_pool_check(evidence, pool))
    results.append(_pfsp_check(pool))

    rated = pool.get("rated_matches")
    rated_ok = (
        isinstance(rated, int)
        and not isinstance(rated, bool)
        and rated >= int(limits["min_rated_matches"])
    )
    if not rated_ok:
        results.append(
            CheckResult(
                "RATED_ELO_EMPTY",
                False,
                f"rated_matches={rated!r}",
                f"only {rated!r} match(es) were scored with BOTH sides at "
                "eps=0, so elo/learner_rated has no data",
                "that series is the AC7 rising-trend evidence AND the "
                "checkpoint-selection input, and an empty one looks exactly "
                "like a flat one on the chart. The rated driver is built by "
                "build_rated_eval_opponent (rated=True, both epsilons 0.0) and "
                "only runs inside an eval cycle - check --eval-every-grad-steps "
                "fired at least once and that the eval opponent is the rated "
                "one.",
            )
        )
    else:
        results.append(
            CheckResult(
                "RATED_ELO_EMPTY",
                True,
                f"{rated} rated match(es); elo/learner_rated="
                f"{pool.get('learner_elo_rated')!r}",
            )
        )

    matches = pool.get("matches_scored")
    draws = pool.get("draws_scored")
    if not isinstance(matches, int) or isinstance(matches, bool) or matches <= 0:
        results.append(
            CheckResult(
                "DRAW_MAJORITY_TRAINING",
                False,
                f"matches_scored={matches!r}",
                "no training match was ever scored into the pool, so the draw "
                "rate is unknown",
                "an unscored run means the self-play driver never reached the "
                "collectors: check the cfg.opponent == 'selfplay' branch in "
                "train_multi_arena actually sets opponent_for.",
            )
        )
    elif not isinstance(draws, int) or isinstance(draws, bool):
        results.append(
            CheckResult(
                "DRAW_MAJORITY_TRAINING",
                False,
                f"draws_scored={draws!r}",
                "the pool's draw count could not be read",
                "check pool.json's draws_scored field.",
            )
        )
    else:
        rate = draws / matches
        if rate > float(limits["max_draw_rate"]) + _EPS:
            results.append(
                CheckResult(
                    "DRAW_MAJORITY_TRAINING",
                    False,
                    f"{draws}/{matches} = {rate:.0%} draws",
                    f"{rate:.0%} of the {matches} scored training matches were "
                    "draws - the mutual-stalling equilibrium, not a fight",
                    "an armored episode that never terminates burns the whole "
                    "night: PFSP goes flat (every win rate sits at 0.5), Elo "
                    "pins, and the reference eval measures nothing. Check "
                    "c_aim < c_step, both loadouts, and whether the 600-step "
                    "cap is short of what an armored kill needs.",
                )
            )
        else:
            results.append(
                CheckResult(
                    "DRAW_MAJORITY_TRAINING",
                    True,
                    f"{draws}/{matches} = {rate:.0%} draws (limit 50%)",
                )
            )
    return results


def _stale_pool_check(
    evidence: Mapping[str, Any], pool: Mapping[str, Any]
) -> CheckResult:
    """STALE_POOL — is this pool THIS run's work, or an earlier attempt's?

    THE HOLE THIS CLOSES. ``agent.train.build_snapshot_opponents`` RELOADS a
    pool it finds on disk::

        if os.path.isfile(os.path.join(directory, INDEX_FILENAME)):
            pool = SnapshotPool.load(directory, sampling=..., log=log)

    and ``SnapshotPool.load`` restores the registry, BOTH Elo series and the
    match counters. So a second canary attempt into the same ``--run-name``
    inherits the first attempt's artifacts, and ``NO_NEW_SNAPSHOT``,
    ``SNAPSHOT_UNCHANGED``, ``RATED_ELO_EMPTY`` and ``DRAW_MAJORITY_TRAINING``
    all read as satisfied before this attempt has proved a thing — even if the
    fix under test broke the very wiring being gated. The shell preflight
    refuses that setup outright; THIS half is the one that cannot be sidestepped
    by picking a fresh ``--run-name`` or by editing the preflight out, because
    it lives in the tested decision module and judges the evidence itself.

    Three independent signals, all fail-closed:

    * The newest snapshot's FILE must have been written at or after the driver
      started. That is the direct measurement of "this run produced it", and it
      is the only one that catches the nastiest case: an archive hook that adds
      NOTHING this attempt, leaving the previous attempt's snapshot as
      ``newest`` with a healthy grad step and a healthy weight delta.
    * The per-snapshot grad steps must not REGRESS as snapshot ids rise. One run
      archives at a monotonically rising grad step; a reloaded pool concatenates
      two such runs and the seam shows up as a drop.
    * The newest snapshot must not carry a grad step this run never reached.

    That last one is asked only when the run reported a POSITIVE grad-step
    count: a zero or unreadable one belongs to :func:`check_grad_steps`, and two
    codes for one absent field would say the same thing twice.
    """
    driver = _section(evidence, "driver")
    grad_steps = pool.get("grad_steps")
    ids = pool.get("snapshot_ids")
    newest_id = pool.get("newest_snapshot_id")

    if (
        not isinstance(grad_steps, Mapping)
        or not isinstance(ids, Sequence)
        or isinstance(ids, (str, bytes))
        or not ids
        or newest_id is None
    ):
        return CheckResult(
            "STALE_POOL",
            False,
            f"snapshot_ids={ids!r} newest={newest_id!r} "
            f"grad_steps={grad_steps!r}",
            "the pool index carries no usable per-snapshot grad-step record, so "
            "this pool cannot be shown to be THIS run's work rather than an "
            "earlier attempt's",
            "check the collector still writes pool.snapshot_ids, "
            "pool.grad_steps and pool.newest_snapshot_id. An unprovenanced pool "
            "pre-satisfies NO_NEW_SNAPSHOT, SNAPSHOT_UNCHANGED, RATED_ELO_EMPTY "
            "and DRAW_MAJORITY_TRAINING with artifacts this run never produced.",
        )

    series: List[List[float]] = []
    for raw_id in ids:
        key = _num(raw_id)
        step = _num(grad_steps.get(str(raw_id)))
        if key is None or step is None:
            return CheckResult(
                "STALE_POOL",
                False,
                f"snapshot {raw_id!r}: grad_step={grad_steps.get(str(raw_id))!r}",
                f"snapshot {raw_id!r} has no readable grad step, so the pool's "
                "history cannot be checked for a reload seam",
                "every live snapshot must appear in pool.grad_steps; a missing "
                "entry means the index and the registry disagree.",
            )
        series.append([key, step])
    series.sort(key=lambda pair: pair[0])

    regressions = [
        f"snapshot {int(series[i][0])} archived at grad step "
        f"{series[i][1]:.0f}, after snapshot {int(series[i - 1][0])} at "
        f"{series[i - 1][1]:.0f}"
        for i in range(1, len(series))
        if series[i][1] < series[i - 1][1] - _EPS
    ]
    if regressions:
        return CheckResult(
            "STALE_POOL",
            False,
            "; ".join(regressions),
            "the pool's grad-step series DROPS as snapshot ids rise, which one "
            "run cannot produce: this pool holds more than one run's history ("
            + "; ".join(regressions)
            + ")",
            "build_snapshot_opponents load-extends any pool.json it finds, and "
            "SnapshotPool.load also restores both Elo series and the match "
            "counters - so this run's draw rate, rated-match count and pool "
            "size are all diluted by an earlier attempt's. Delete "
            "runs/<run>/snapshots (and runs/<run>.pt) or re-run with a fresh "
            "--run-name, then judge the run on its own evidence.",
        )

    newest_grad = _num(grad_steps.get(str(newest_id)))
    run_grad_steps = driver.get("grad_steps")
    if (
        newest_grad is not None
        and isinstance(run_grad_steps, int)
        and not isinstance(run_grad_steps, bool)
        and run_grad_steps > 0
        and newest_grad > float(run_grad_steps) + _EPS
    ):
        return CheckResult(
            "STALE_POOL",
            False,
            f"newest snapshot {newest_id} is from grad step {newest_grad:.0f}, "
            f"but this run took {run_grad_steps}",
            f"the newest snapshot was archived at grad step {newest_grad:.0f}, "
            f"a step THIS run never reached (it took {run_grad_steps}), so it "
            "was produced by an earlier attempt",
            "the pool was reloaded from a previous run. Delete "
            "runs/<run>/snapshots or pass a fresh --run-name; every pool-derived "
            "refusal below is otherwise reading that earlier attempt's work.",
        )

    written_at = _num(pool.get("newest_snapshot_mtime"))
    started_at = _num(evidence.get("driver_started_at"))
    if written_at is None or started_at is None:
        return CheckResult(
            "STALE_POOL",
            False,
            f"newest_snapshot_mtime={written_at!r} "
            f"driver_started_at={started_at!r}",
            "the newest snapshot's write time could not be compared against "
            "this run's start, so there is no proof this run produced it",
            "the collector stamps pool.newest_snapshot_mtime from the snapshot "
            "file and driver_started_at from the shell. Without both, an "
            "attempt that archived NOTHING is indistinguishable from one that "
            "archived correctly.",
        )
    if written_at < started_at - _EPS:
        return CheckResult(
            "STALE_POOL",
            False,
            f"newest snapshot written {started_at - written_at:.0f}s BEFORE the "
            "driver started",
            f"the newest snapshot's file predates this run's driver by "
            f"{started_at - written_at:.0f}s, so THIS attempt archived nothing "
            "and every pool-derived check above is reading an earlier "
            "attempt's artifacts",
            "this is the re-run trap: build_snapshot_opponents reloads an "
            "existing pool.json and SnapshotPool.load restores the Elo series "
            "and the match counters with it. Find out why "
            "SnapshotArchivist.maybe_archive never fired this run, then delete "
            "runs/<run>/snapshots (or pass a fresh --run-name) and re-run so "
            "the gate judges this attempt alone.",
        )

    return CheckResult(
        "STALE_POOL",
        True,
        f"newest snapshot {newest_id} written {written_at - started_at:.0f}s "
        f"into this run at grad step "
        + ("unknown" if newest_grad is None else f"{newest_grad:.0f}")
        + f"; {len(series)} snapshot(s) on a single rising grad-step series",
    )


def _pfsp_check(pool: Mapping[str, Any]) -> CheckResult:
    """PFSP_INVALID — finite, non-negative, normalized, and covering every member.

    ``SnapshotPool.pfsp_weights`` is contracted to return exactly this (the
    Beta(1, 1) prior is what makes ``p(1-p)`` defined at zero plays, and the
    ``0.05/N`` floor is what keeps a pinned reference sampleable at win rate 0.0
    or 1.0). Reading it as-is here is that contract's only live check.
    """
    weights = pool.get("pfsp_weights")
    if not isinstance(weights, Mapping) or not weights:
        return CheckResult(
            "PFSP_INVALID",
            False,
            f"pfsp_weights={weights!r}",
            "PFSP returned no probabilities at all",
            "an empty weight vector means sample() has nothing to draw from; "
            "check the pool is non-empty and pfsp_weights() ran.",
        )
    values: List[float] = []
    for key, raw in weights.items():
        value = _num(raw)
        if value is None:
            return CheckResult(
                "PFSP_INVALID",
                False,
                f"snapshot {key}: {raw!r}",
                f"PFSP probability for snapshot {key} is {raw!r} - not a finite "
                "number",
                "a NaN or Inf here makes opponent sampling undefined for the "
                "whole run. The Beta(1, 1) prior exists precisely so f(p)=p(1-p) "
                "is defined at zero plays; check win_rate() and the floor term.",
            )
        if value < -_EPS:
            return CheckResult(
                "PFSP_INVALID",
                False,
                f"snapshot {key}: {value}",
                f"PFSP probability for snapshot {key} is negative ({value})",
                "weights are f(p) + floor, both non-negative by construction; a "
                "negative one means the normalization is wrong.",
            )
        values.append(value)
    total = math.fsum(values)
    if abs(total - 1.0) > 1e-6:
        return CheckResult(
            "PFSP_INVALID",
            False,
            f"probabilities sum to {total!r}",
            f"PFSP probabilities sum to {total!r}, not 1.0",
            "an un-normalized vector means the sampler's effective "
            "distribution is not the one the metrics describe.",
        )
    ids = pool.get("snapshot_ids")
    if isinstance(ids, Sequence) and not isinstance(ids, (str, bytes)):
        missing = [i for i in ids if str(i) not in weights]
        if missing:
            return CheckResult(
                "PFSP_INVALID",
                False,
                f"no weight for snapshot(s) {missing}",
                f"PFSP produced no probability for live snapshot(s) {missing}",
                "a live snapshot with no weight can never be sampled and can "
                "never earn the results that would raise its weight - that is "
                "how a self-play run forgets how to beat its ancestors.",
            )
    return CheckResult(
        "PFSP_INVALID",
        True,
        f"{len(values)} finite probabilities summing to {total:.6f}",
    )


def check_checkpoints(evidence: Mapping[str, Any]) -> CheckResult:
    """CHECKPOINT_UNLOADABLE — the shipped file and the newest snapshot both load.

    Both go through ``eval.evaluate._load_drqn``, the SHARED loader the eval and
    deploy paths use. A payload only the training loop can read is a checkpoint
    that fails on demo day.
    """
    checkpoint = _section(evidence, "checkpoint")
    problems: List[str] = []
    for label, key in (("final checkpoint", "final"), ("newest snapshot", "snapshot")):
        entry = checkpoint.get(key)
        entry = dict(entry) if isinstance(entry, Mapping) else {}
        if entry.get("loadable") is not True:
            problems.append(
                f"{label} {entry.get('path')!r}: "
                f"{entry.get('error') or 'not loadable'}"
            )
    if problems:
        return CheckResult(
            "CHECKPOINT_UNLOADABLE",
            False,
            "; ".join(problems),
            "a checkpoint this run produced does not load: " + "; ".join(problems),
            "both must load through eval.evaluate._load_drqn, which builds "
            "DuelingDRQN() with DEFAULT kwargs. A payload the shared loader "
            "cannot read is discovered on demo day otherwise.",
        )
    return CheckResult(
        "CHECKPOINT_UNLOADABLE",
        True,
        "final checkpoint and newest snapshot both load through _load_drqn",
    )


def evaluate_canary(
    evidence: Mapping[str, Any], thresholds: Optional[Mapping[str, Any]] = None
) -> Verdict:
    """Decide whether the 24-hour self-play run may start.

    Args:
        evidence: The evidence document (normally ``evidence.json``).
        thresholds: Optional overrides of :data:`DEFAULT_THRESHOLDS`.

    Returns:
        A :class:`Verdict`. ``verdict.ok`` is the go/no-go; ``verdict.refusals``
        carries one entry per blocking condition, each with WHY and what to
        check.
    """
    limits = _thresholds(thresholds)
    version = evidence.get("evidence_version")
    if version != EVIDENCE_VERSION:
        return Verdict(
            [
                CheckResult(
                    "DRIVER_FAILED",
                    False,
                    f"evidence_version={version!r}",
                    f"the evidence document is version {version!r}, not "
                    f"{EVIDENCE_VERSION} - it cannot be read safely",
                    "re-run the canary with the matching script; reading an "
                    "unknown layout would let absent fields pass as healthy "
                    "zeros.",
                )
            ],
            {},
        )

    checks: List[CheckResult] = [check_driver(evidence)]
    checks.append(check_grad_steps(evidence, limits))
    checks.extend(check_pool(evidence, limits))
    checks.append(check_checkpoints(evidence))
    checks.extend(
        evaluate_gear(_section(evidence, "probe_post"), limits, strict=True)
    )
    return Verdict(checks, build_measurements(evidence))


# ---------------------------------------------------------------------------
# The measurement T19 needs. The armored regime's mean episode length has never
# been measured - the 95-step figure is from the BARE-handed M3 retry and the
# 285-step figure predates even that. This canary is the first thing that ever
# produces one, so it is written out as a first-class artifact rather than left
# in a log line.
# ---------------------------------------------------------------------------


def build_measurements(evidence: Mapping[str, Any]) -> Dict[str, Any]:
    """Extract the numbers T19 sizes ``--eps-decay-episodes`` from.

    TWO episode-length figures and one RATE, each labelled with the regime that
    produced it, because they are NOT interchangeable and T19 must pick
    deliberately:

      * ``armored_mean_episode_length_probe`` — the live probe's, at the run's
        terminal epsilons (learner 0.05 / frozen opponent 0.02): the regime the
        demo is actually fought in, and the only one where BOTH seats are the
        trained net.
      * ``armored_mean_episode_length_eval_vs_scripted`` — the training run's
        periodic eval, fought against the FIXED SCRIPTED yardstick.
        ``agent.train.build_eval_opponent`` returns that same scripted driver on
        a self-play run, deliberately: an Elo ladder measured only against past
        selves can climb while the whole pool drifts, so the cycle also needs
        one number that cannot inflate. It is therefore NOT "eps=0 on both
        sides" — that is the rated reference gauntlet, which reports Elo and
        never an episode length. ``eval_opponent`` carries the driver's own
        name so the figure can never be relabelled by accident.
      * ``measured_episodes_per_arena_hour`` — the collection RATE actually
        observed, which includes reset overhead and the whole epsilon schedule,
        and is what converts a 24-hour budget into an episode count.

    WHERE THE EVAL FIGURE COMES FROM. The driver log, not ``summary.json``. On
    the multi-arena path the only ``logger.summary()`` call in ``agent/train.py``
    is ``logger.summary(selfplay_log_row(snapshot_pool))``, whose keys are
    ``elo/*``, ``selfplay/pool_size``, ``selfplay/matches_scored``,
    ``selfplay/rated_matches``, ``selfplay/draw_rate`` and
    ``selfplay/win_rate_vs_ref_*``. There is no ``win_rate`` and no
    ``mean_episode_length`` in that row at all, so a collector gated on
    ``"win_rate" in summary`` reported ``{"ran": False}`` on EVERY run and this
    figure was always ``None``.

    Nothing here is hardcoded from the bare-handed era.
    """
    probe = _section(evidence, "probe_post")
    eval_section = _section(evidence, "eval")
    driver = _section(evidence, "driver")

    episodes = driver.get("episodes")
    wall_seconds = _num(evidence.get("driver_wall_seconds"))
    arenas = evidence.get("arenas")

    per_arena_hour: Optional[float] = None
    if (
        isinstance(episodes, int)
        and not isinstance(episodes, bool)
        and wall_seconds
        and wall_seconds > 0
        and isinstance(arenas, int)
        and arenas > 0
    ):
        per_arena_hour = episodes / (wall_seconds / 3600.0) / arenas

    return {
        "measured_at_run": evidence.get("run_name"),
        "max_episode_steps": probe.get("max_episode_steps", MAX_EPISODE_STEPS),
        # -- armored episode length, three regimes ---------------------------
        "armored_mean_episode_length_probe": probe.get("mean_episode_length"),
        "armored_median_episode_length_probe": probe.get("median_episode_length"),
        "armored_episode_lengths_probe": probe.get("episode_lengths"),
        "probe_learner_epsilon": probe.get("learner_epsilon"),
        "probe_opponent_epsilon": probe.get("opponent_epsilon"),
        "armored_mean_episode_length_eval_vs_scripted": eval_section.get(
            "mean_episode_length"
        ),
        "eval_opponent": eval_section.get("opponent"),
        "eval_grad_step": eval_section.get("grad_step"),
        "eval_episodes_per_cycle": eval_section.get("episodes_per_cycle"),
        # -- collection rate, which is what a 24-hour budget converts through -
        "training_episodes": episodes,
        "training_grad_steps": driver.get("grad_steps"),
        "training_wall_seconds": wall_seconds,
        "training_arenas": arenas,
        "training_epsilon_schedule_at_end": _section(evidence, "epsilon").get(
            "schedule"
        ),
        "training_epsilon_mean_at_end": _section(evidence, "epsilon").get("mean"),
        "measured_episodes_per_arena_hour": per_arena_hour,
        "projected_episodes_per_hour_at_25_pads": (
            per_arena_hour * 25.0 if per_arena_hour is not None else None
        ),
        # -- the armored damage regime ---------------------------------------
        "armored_damage_dealt_per_episode": probe.get("damage_dealt_per_episode"),
        "armored_damage_taken_per_episode": probe.get("damage_taken_per_episode"),
        "armored_full_charge_hits_dealt": (probe.get("hits_dealt") or {}).get("values"),
        "armored_full_charge_hits_taken": (probe.get("hits_taken") or {}).get("values"),
        "armored_cap_hit_rate": probe.get("cap_hit_rate"),
        "notes": [
            "TWO episode-length figures plus one rate, and they are NOT "
            "interchangeable. armored_mean_episode_length_probe is both seats "
            "at the run's terminal epsilons (the demo's regime); "
            "armored_mean_episode_length_eval_vs_scripted is the periodic eval "
            "against the FIXED SCRIPTED yardstick named in eval_opponent, not "
            "an eps=0 self-play match; measured_episodes_per_arena_hour is a "
            "rate, not a length.",
            "projected_episodes_per_hour_at_25_pads scales the measured "
            "per-arena rate linearly to 25 pads, which the arena sweep recorded "
            "as linear with no knee (600 s confirm at N=25: 121.95 "
            "transitions/s aggregate, 4.8782 per arena).",
            "measured_episodes_per_arena_hour INCLUDES this run's replay "
            "warm-up and every eval cycle, so it is a floor on the long run's "
            "rate, not an estimate of its steady state.",
            "RENAMED: armored_mean_episode_length_eval_greedy is gone. It was "
            "always null - the multi-arena path writes no win_rate into "
            "summary.json, so the collector's gate never fired - and its label "
            "was wrong even once populated, because the periodic eval fights "
            "the fixed scripted yardstick, not eps=0 on both sides. Read "
            "armored_mean_episode_length_eval_vs_scripted together with "
            "eval_opponent instead.",
        ],
    }


# ---------------------------------------------------------------------------
# Reporting.
# ---------------------------------------------------------------------------


def _format_measurement(key: str, value: Any) -> str:
    if isinstance(value, float):
        return f"    {key:<44} {value:.4g}"
    return f"    {key:<44} {value}"


def format_report(verdict: Verdict, evidence: Mapping[str, Any]) -> str:
    """Render the operator-facing report: every check, then the verdict."""
    rule = "=" * 78
    lines = [rule, " T17 SELF-PLAY LAUNCH CANARY - VERDICT", rule]

    lines.append(
        f" run                {evidence.get('run_name')}  "
        f"({evidence.get('arenas')} pads)"
    )
    lines.append(
        f" replay regime      --min-replay {evidence.get('min_replay')} "
        f"(production {evidence.get('min_replay_production')} - LOWERED for the "
        "canary so it reaches real gradient steps)"
    )
    lines.append(f" evidence           {evidence.get('evidence_path')}")
    lines.append("")
    lines.append(" CHECKS")
    for check in verdict.checks:
        marker = "ok    " if check.passed else "REFUSE"
        lines.append(f"   [{marker}] {check.code:<26} {check.detail}")

    if verdict.refusals:
        lines.append("")
        lines.append(" REFUSALS - the 24-hour run must NOT start")
        for check in verdict.refusals:
            lines.append("")
            lines.append(f"   {check.code}")
            lines.append(f"     WHY:   {check.why}")
            lines.append(f"     CHECK: {check.check}")

    lines.append("")
    lines.append(" MEASURED FOR T19 (the first armored numbers this project has)")
    for key in (
        "armored_mean_episode_length_probe",
        "armored_median_episode_length_probe",
        "probe_learner_epsilon",
        "probe_opponent_epsilon",
        "armored_mean_episode_length_eval_vs_scripted",
        "eval_opponent",
        "armored_damage_dealt_per_episode",
        "armored_damage_taken_per_episode",
        "armored_cap_hit_rate",
        "training_episodes",
        "training_grad_steps",
        "measured_episodes_per_arena_hour",
        "projected_episodes_per_hour_at_25_pads",
    ):
        lines.append(_format_measurement(key, verdict.measurements.get(key)))

    lines.append("")
    if verdict.ok:
        lines.append(" VERDICT: GREEN - the 25-pad self-play run is cleared to launch.")
    else:
        lines.append(
            f" VERDICT: REFUSED - {len(verdict.refusals)} condition(s) block the "
            "launch."
        )
    lines.append(rule)
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI: ``canary_verdict.py <evidence.json> [measurements.json]``.

    Exits 0 on GREEN, 1 on any refusal. Also writes the T19 measurements when a
    second path is given.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print("usage: canary_verdict.py <evidence.json> [measurements.json]", file=sys.stderr)
        return 2
    with open(args[0], "r", encoding="utf-8") as handle:
        evidence = json.load(handle)
    evidence = dict(evidence)
    evidence.setdefault("evidence_path", args[0])
    verdict = evaluate_canary(evidence, evidence.get("thresholds"))
    if len(args) > 1:
        with open(args[1], "w", encoding="utf-8") as handle:
            json.dump(verdict.measurements, handle, indent=2, sort_keys=True)
            handle.write("\n")
    print(format_report(verdict, evidence))
    return 0 if verdict.ok else 1


if __name__ == "__main__":  # pragma: no cover - exercised by the shell script
    raise SystemExit(main())
CANARY_VERDICT_PY
}

# ===========================================================================
# Non-connecting port inspection. Mirrors server/setup/start-pads.sh's helpers,
# and for the same reason: a connect probe against a bridge is a MUTATION.
# ===========================================================================

# listener_pids <port> — pids LISTENing on that TCP port. Opens no connection.
listener_pids() {
    if command -v lsof >/dev/null 2>&1; then
        lsof -nP -iTCP:"$1" -sTCP:LISTEN -t 2>/dev/null || true
    else
        pgrep -f "run\\.js .*--bridge-port[= ]$1([^0-9]|\$)" 2>/dev/null || true
    fi
}

# established_peers <port> — count of ESTABLISHED connections on that TCP port.
#
# This is the guard against the outage that has hit this project four times:
# BridgeServer accepts exactly ONE client and destroys the incumbent when a
# second arrives. Before this script connects anything, it asks lsof whether
# somebody is already attached — WITHOUT connecting, which a probe would.
established_peers() {
    if command -v lsof >/dev/null 2>&1; then
        lsof -nP -iTCP:"$1" 2>/dev/null | grep -c "ESTABLISHED" || true
    else
        echo "0"
    fi
}

# mc_connect_probe <host> <port> — 0 when a TCP connect succeeds within 2s.
#
# Legal ONLY against the Minecraft port: Paper is a normal multi-client server,
# so connecting to it is free. Never point this at a bridge port.
mc_connect_probe() {
    "${PYTHON_BIN}" -c '
import socket, sys
try:
    socket.create_connection((sys.argv[1], int(sys.argv[2])), timeout=2.0).close()
except OSError:
    sys.exit(1)
' "$1" "$2" >/dev/null 2>&1
}

# ===========================================================================
# The live seat probe. Deliberately DUMB I/O: it records one raw record per
# decision window and does no arithmetic — every number the gate reasons about
# is produced by canary_verdict.summarize_probe, which the test suite drives.
#
# ONE connection, ONE env, both seats driven from the SAME net through the
# mirrored observation. It does not import the self-play driver: everything here
# is the stable, landed surface (TcpBridgeClient, MCPvPEnv(mirror_opponent=True),
# DuelingDRQN, eval.evaluate._load_drqn), so the probe cannot break when a
# sibling task edits agent/train.py.
# ===========================================================================
run_probe() {
    # run_probe <checkpoint> <episodes> <out.json> <label>
    local checkpoint="$1" episodes="$2" out="$3" label="$4"
    local peers
    peers="$(established_peers "${PROBE_PORT}")"
    if [[ "${peers}" -gt 0 ]]; then
        # Record the refusal instead of connecting anyway. BridgeServer accepts
        # exactly ONE client and resolves a second by destroying the incumbent,
        # so "probe it and see" would take down whatever is attached. The
        # verdict turns this document into a PROBE_FAILED refusal.
        local why
        why="bridge port ${PROBE_PORT} already has ${peers} established peer(s);"
        why="${why} BridgeServer accepts exactly ONE client and a second"
        why="${why} connection destroys the first, so the probe refused to connect"
        warn "${why}"
        printf '{"ok": false, "error": "%s"}\n' "${why}" >"${out}"
        return 0
    fi
    log "probe (${label}): ${episodes} episode(s) on port ${PROBE_PORT} using ${checkpoint}"
    (
        cd "${REPO_ROOT}" && \
        CANARY_CHECKPOINT="${checkpoint}" \
        CANARY_EPISODES="${episodes}" \
        CANARY_OUT="${out}" \
        CANARY_HOST="${HOST}" \
        CANARY_PORT="${PROBE_PORT}" \
        CANARY_LEARNER_EPS="${PROBE_LEARNER_EPSILON}" \
        CANARY_OPPONENT_EPS="${PROBE_OPPONENT_EPSILON}" \
        CANARY_SEED="${SEED}" \
        "${PYTHON_BIN}" - <<'CANARY_PROBE_PY'
"""The T17 live seat probe — raw recording only, no arithmetic."""
from __future__ import annotations

import json
import math
import os
import sys
import traceback

OUT = os.environ["CANARY_OUT"]
CHECKPOINT = os.environ["CANARY_CHECKPOINT"]
EPISODES = int(os.environ["CANARY_EPISODES"])
HOST = os.environ["CANARY_HOST"]
PORT = int(os.environ["CANARY_PORT"])
LEARNER_EPS = float(os.environ["CANARY_LEARNER_EPS"])
OPPONENT_EPS = float(os.environ["CANARY_OPPONENT_EPS"])
SEED = int(os.environ["CANARY_SEED"])

document = {
    "ok": False,
    "error": None,
    "checkpoint": CHECKPOINT,
    "learner_epsilon": LEARNER_EPS,
    "opponent_epsilon": OPPONENT_EPS,
    "max_episode_steps": None,
    "episodes": [],
}

arena_env = None
try:
    import numpy as np
    import torch

    from env.mc_pvp_env import MCPvPEnv, TcpBridgeClient
    from env.observation_spec import Obs
    from eval.evaluate import _load_drqn

    net = _load_drqn(CHECKPOINT, torch.device("cpu"))
    # Two independent generators so the learner's exploration coin flips cannot
    # correlate with the frozen opponent's — the two seats must be independent
    # players, not one player mirrored.
    learner_rng = torch.Generator().manual_seed(SEED * 2 + 1)
    opponent_rng = torch.Generator().manual_seed(SEED * 2 + 2)

    arena_env = MCPvPEnv(TcpBridgeClient(host=HOST, port=PORT), mirror_opponent=True)
    cap = int(arena_env.max_episode_steps or 600)
    document["max_episode_steps"] = cap

    for index in range(EPISODES):
        obs = arena_env.reset(seed=SEED + index)
        mirrored = arena_env.opponent_observation()
        learner_hidden = None
        opponent_hidden = None
        windows = []
        info = {}
        for _ in range(cap):
            action, learner_hidden = net.act(
                torch.from_numpy(np.asarray(obs, dtype=np.float32)),
                learner_hidden,
                LEARNER_EPS,
                generator=learner_rng,
            )
            opp_action, opponent_hidden = net.act(
                torch.from_numpy(np.asarray(mirrored, dtype=np.float32)),
                opponent_hidden,
                OPPONENT_EPS,
                generator=opponent_rng,
            )
            # RAW, ungated opponent position. Recorded for the frozen-opponent
            # diagnostic ONLY and never routed into either net's observation —
            # both seats are fed obs/mirrored above and nothing else.
            view = arena_env.raw_opponent_view()
            record = {
                "a": int(action),
                "oa": int(opp_action),
                "cd": float(obs[Obs.ATTACK_COOLDOWN]),
                "ocd": float(mirrored[Obs.ATTACK_COOLDOWN]),
                # Horizontal speed: the yaw rotation that builds vel_local turns
                # x/z about y, so the xz magnitude survives it. y is dropped so
                # gravity cannot masquerade as walking.
                "osp": float(
                    math.hypot(
                        float(mirrored[Obs.VEL_LOCAL]),
                        float(mirrored[Obs.VEL_LOCAL + 2]),
                    )
                ),
                "opx": float(view.self_pos[0]),
                "opz": float(view.self_pos[2]),
            }
            obs, _reward, done, info = arena_env.step(
                int(action), opp_action=int(opp_action)
            )
            mirrored = arena_env.opponent_observation()
            events = info.get("events", {})
            record["dd"] = float(events.get("damage_dealt", 0.0))
            record["dt"] = float(events.get("damage_taken", 0.0))
            windows.append(record)
            if done:
                break
        outcome = "timeout"
        if info.get("lost"):
            outcome = "loss"
        elif info.get("won"):
            outcome = "win"
        document["episodes"].append(
            {"length": len(windows), "outcome": outcome, "windows": windows}
        )
        print(
            f"[probe] episode {index}: {outcome} in {len(windows)} windows",
            file=sys.stderr,
        )
    document["ok"] = True
except BaseException as exc:  # noqa: BLE001 - the probe must always leave evidence
    document["ok"] = False
    document["error"] = f"{type(exc).__name__}: {exc}"
    traceback.print_exc()
finally:
    if arena_env is not None:
        try:
            arena_env.close()
        except BaseException:  # noqa: BLE001 - the connection is going away anyway
            pass
    with open(OUT, "w", encoding="utf-8") as handle:
        json.dump(document, handle)

raise SystemExit(0 if document["ok"] else 1)
CANARY_PROBE_PY
    ) || true
}

# gate_probe <probe.json> <strict> — run the gear checks and refuse early.
gate_probe() {
    ( cd "${REPO_ROOT}" && \
      CANARY_PROBE_JSON="$1" \
      CANARY_STRICT="$2" \
      CANARY_VERDICT_MODULE="${VERDICT_MODULE}" \
      "${PYTHON_BIN}" - <<'CANARY_GATE_PY'
import importlib.util
import json
import os
import sys

spec = importlib.util.spec_from_file_location(
    "canary_verdict", os.environ["CANARY_VERDICT_MODULE"]
)
verdict_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verdict_module)

try:
    with open(os.environ["CANARY_PROBE_JSON"], "r", encoding="utf-8") as handle:
        probe = json.load(handle)
except (OSError, ValueError) as exc:
    probe = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

summary = verdict_module.summarize_probe(probe)
strict = os.environ["CANARY_STRICT"] == "1"
checks = verdict_module.evaluate_gear(summary, strict=strict)
refusals = [c for c in checks if not c.passed]
for check in checks:
    marker = "ok    " if check.passed else "REFUSE"
    print(f"   [{marker}] {check.code:<26} {check.detail}")
for check in refusals:
    print(f"\n   {check.code}\n     WHY:   {check.why}\n     CHECK: {check.check}")
sys.exit(1 if refusals else 0)
CANARY_GATE_PY
    )
}

# ===========================================================================
# --analyze-only: re-run the verdict over an existing evidence directory.
# Connects to nothing, starts nothing.
# ===========================================================================
if [[ -n "${ANALYZE_ONLY}" ]]; then
    OUT_DIR="${ANALYZE_ONLY}"
    VERDICT_MODULE="${OUT_DIR}/canary_verdict.py"
    EVIDENCE_JSON="${OUT_DIR}/evidence.json"
    MEASUREMENTS_JSON="${OUT_DIR}/canary_measurements.json"
    [[ -f "${EVIDENCE_JSON}" ]] || die "no evidence.json in ${OUT_DIR}"
    emit_verdict_module "${VERDICT_MODULE}"
    set +e
    "${PYTHON_BIN}" "${VERDICT_MODULE}" "${EVIDENCE_JSON}" "${MEASUREMENTS_JSON}"
    VERDICT_EXIT=$?
    set -e
    exit "${VERDICT_EXIT}"
fi

# ===========================================================================
# Phase 0 — preflight. Starts nothing, connects to nothing but Paper.
# ===========================================================================
log "phase 0: preflight"

[[ -n "${WARM_START}" ]] || { usage >&2; die "--warm-start is REQUIRED.
      agent/train_config.py refuses --opponent selfplay without one (AC14), and
      a canary that starts from a fresh net measures a different game."; }
[[ -f "${WARM_START}" ]] || die "warm start not found: ${WARM_START}"

# EVERY ATTEMPT STARTS CLEAN. agent/train.py's build_snapshot_opponents does
#   if os.path.isfile(os.path.join(directory, INDEX_FILENAME)):
#       pool = SnapshotPool.load(directory, sampling=..., log=log)
# and SnapshotPool.load restores the registry, BOTH Elo series and the match
# counters. A second attempt into the same --run-name therefore inherits the
# first attempt's artifacts, and NO_NEW_SNAPSHOT, SNAPSHOT_UNCHANGED,
# RATED_ELO_EMPTY and DRAW_MAJORITY_TRAINING are all pre-satisfied by them even
# when the fix under test broke the wiring being gated. runs/<run>.pt likewise
# lets CHECKPOINT_UNLOADABLE pass on a stale file, and metrics.jsonl is APPENDED
# to, so an epsilon measurement could come from the earlier attempt.
#
# This refusal is LOUD AND BLOCKING rather than a warning, because the failure
# it prevents is silent by construction: attempt 2 exits GREEN and nothing in
# the report says which attempt earned it. STALE_POOL is the verdict-level half
# that survives an operator picking a fresh --run-name.
POOL_INDEX="${REPO_ROOT}/runs/${RUN_NAME}/snapshots/pool.json"
METRICS_JSONL="${REPO_ROOT}/runs/${RUN_NAME}/metrics.jsonl"
STALE_ARTIFACTS=""
for artifact in "${POOL_INDEX}" "${CHECKPOINT_PATH}" "${BEST_CHECKPOINT_PATH}" \
                "${METRICS_JSONL}"; do
    if [[ -e "${artifact}" ]]; then
        STALE_ARTIFACTS="${STALE_ARTIFACTS}
        ${artifact}"
    fi
done
if [[ -n "${STALE_ARTIFACTS}" ]]; then
    die "run '${RUN_NAME}' already has artifacts from an earlier attempt:${STALE_ARTIFACTS}

      The snapshot pool is LOAD-EXTENDED, not replaced: build_snapshot_opponents
      reloads any pool.json it finds and SnapshotPool.load restores the Elo
      series and the match counters with it. Re-running on top of these would
      pre-satisfy NO_NEW_SNAPSHOT, SNAPSHOT_UNCHANGED, RATED_ELO_EMPTY and
      DRAW_MAJORITY_TRAINING with the EARLIER attempt's work, and this gate
      would report GREEN without having tested the fix.

      Start clean - either delete this run's artifacts:
        rm -rf ${REPO_ROOT}/runs/${RUN_NAME} ${CHECKPOINT_PATH} ${BEST_CHECKPOINT_PATH}
      or give this attempt its own name:
        --run-name ${RUN_NAME}_2"
fi

mkdir -p "${OUT_DIR}"
mkdir -p "${REPO_ROOT}/runs"

if command -v shasum >/dev/null 2>&1; then
    WARM_START_SHA256="$(shasum -a 256 "${WARM_START}" | awk '{print $1}')"
else
    WARM_START_SHA256="$("${PYTHON_BIN}" -c '
import hashlib, sys
h = hashlib.sha256()
with open(sys.argv[1], "rb") as fh:
    for chunk in iter(lambda: fh.read(1 << 20), b""):
        h.update(chunk)
print(h.hexdigest())
' "${WARM_START}")"
fi
log "warm start ${WARM_START}"
log "  sha256 ${WARM_START_SHA256}"

# Paper is multi-client, so this connect is free. It is also the ONLY connect
# probe in this file.
if ! mc_connect_probe "${HOST}" "${MC_PORT}"; then
    die "no Minecraft server on ${HOST}:${MC_PORT}.
      The boot order for anything live is Paper -> bridges -> this script, and
      this script starts NEITHER. Boot the fleet first:
        DUMMY_KNOCKBACK_IMMUNE=false server/setup/start-pads.sh --pads ${ARENAS}"
fi
log "Minecraft server answering on ${HOST}:${MC_PORT}"

MISSING_PORTS=""
BUSY_PORTS=""
for (( i = 0; i < ARENAS; i++ )); do
    port=$(( BRIDGE_BASE_PORT + i ))
    if [[ -z "$(listener_pids "${port}")" ]]; then
        MISSING_PORTS="${MISSING_PORTS} ${port}"
        continue
    fi
    if [[ "$(established_peers "${port}")" -gt 0 ]]; then
        BUSY_PORTS="${BUSY_PORTS} ${port}"
    fi
done
if [[ -n "${MISSING_PORTS}" ]]; then
    die "no bridge listening on:${MISSING_PORTS}
      Boot the fleet with the SAME pad count (and knockback ON):
        DUMMY_KNOCKBACK_IMMUNE=false server/setup/start-pads.sh --pads ${ARENAS}"
fi
if [[ -n "${BUSY_PORTS}" ]]; then
    die "bridge port(s) already have an established client:${BUSY_PORTS}
      BridgeServer accepts exactly ONE TCP client and a second connection
      silently destroys the first. Stop the other driver before running this."
fi
log "${ARENAS} bridge listener(s) on ${BRIDGE_BASE_PORT}..$(( BRIDGE_BASE_PORT + ARENAS - 1 )), none attached"

# DUMMY_KNOCKBACK_IMMUNE cannot be read back from a running process on macOS
# (`ps -E` / `ps eww` do not expose another process's environment), so this is
# an advisory check only: distributed.launcher passes the FLAG on argv, while
# server/setup/start-pads.sh relies on the env var and shows nothing. The
# authoritative check is empirical — the probe's OPPONENT_FROZEN condition,
# which measures whether the opponent actually moves.
if pgrep -f -- "--dummy-knockback-immune[= ]false" >/dev/null 2>&1; then
    log "a bridge argv carries --dummy-knockback-immune false"
else
    warn "could not confirm DUMMY_KNOCKBACK_IMMUNE=false from process state."
    warn "  macOS does not expose another process's environment, and"
    warn "  start-pads.sh passes the setting via the env var, not argv."
    warn "  The probe's OPPONENT_FROZEN check verifies it empirically instead."
fi

emit_verdict_module "${VERDICT_MODULE}"
log "verdict module written to ${VERDICT_MODULE}"

# ===========================================================================
# Phase 1 — fail-fast pre-probe on the WARM-START net.
#
# Runs before the training budget is spent so a mis-geared or frozen fleet costs
# one episode, not half an hour. Non-strict: the volume-dependent checks are
# deferred to the post-run probe and say so.
# ===========================================================================
if [[ "${DO_PREFLIGHT_PROBE}" -eq 1 ]]; then
    log "phase 1: fail-fast pre-probe (warm-start net, non-strict)"
    run_probe "${WARM_START}" "${PREFLIGHT_PROBE_EPISODES}" "${PROBE_PRE_JSON}" "pre"
    set +e
    gate_probe "${PROBE_PRE_JSON}" 0
    PRE_GATE_EXIT=$?
    set -e
    if [[ "${PRE_GATE_EXIT}" -ne 0 ]]; then
        echo "" >&2
        echo "[canary] REFUSED at the pre-probe. The training budget was NOT spent." >&2
        exit 1
    fi
    log "pre-probe clear; spending the training budget"
else
    warn "pre-probe skipped (--no-preflight-probe): a mis-geared fleet will not"
    warn "  be caught until after the training budget has been spent."
fi

# ===========================================================================
# Phase 2 — the short run that ACTUALLY LEARNS.
# ===========================================================================
log "phase 2: training (${ARENAS} pads, ${MAX_GRAD_STEPS} grad steps, --min-replay ${MIN_REPLAY})"

DRIVER_PID=""
cleanup() {
    if [[ -n "${DRIVER_PID}" ]] && kill -0 "${DRIVER_PID}" 2>/dev/null; then
        kill -INT "${DRIVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

DRIVER_START=$(date +%s)
(
    cd "${REPO_ROOT}" && exec "${PYTHON_BIN}" -m agent.train \
        --arenas "${ARENAS}" \
        --opponent selfplay \
        --host "${HOST}" \
        --port "${BRIDGE_BASE_PORT}" \
        --mc-port "${MC_PORT}" \
        --warm-start "${WARM_START}" \
        --warm-start-sha256 "${WARM_START_SHA256}" \
        --run-name "${RUN_NAME}" \
        --checkpoint "${CHECKPOINT_PATH}" \
        --best-checkpoint "${BEST_CHECKPOINT_PATH}" \
        --max-grad-steps "${MAX_GRAD_STEPS}" \
        --min-replay "${MIN_REPLAY}" \
        --snapshot-every-grad-steps "${SNAPSHOT_EVERY}" \
        --snapshot-sampling pfsp \
        --reference-promote-grad-steps "${PROMOTE_FIRST}" "${PROMOTE_SECOND}" \
        --eval-every-grad-steps "${EVAL_EVERY}" \
        --eval-episodes "${EVAL_EPISODES}" \
        --checkpoint-every-grad-steps "${CHECKPOINT_EVERY}" \
        --eps-decay-episodes "${EPS_DECAY_EPISODES}" \
        --seed "${SEED}" \
        --log-backend jsonl \
        --no-progress
) >"${DRIVER_LOG}" 2>&1 &
DRIVER_PID=$!
log "driver pid ${DRIVER_PID}, log ${DRIVER_LOG}"

DEADLINE_SECONDS=$(( DEADLINE_MINUTES * 60 ))
DEADLINE_AT=$(( DRIVER_START + DEADLINE_SECONDS ))
DEADLINE_HIT="false"
while kill -0 "${DRIVER_PID}" 2>/dev/null; do
    if [[ "$(date +%s)" -ge "${DEADLINE_AT}" ]]; then
        DEADLINE_HIT="true"
        warn "wall-clock deadline (${DEADLINE_MINUTES} min) reached; interrupting the driver."
        # SIGINT, not SIGTERM: train_multi_arena's teardown runs in a `finally`,
        # so an interrupt still joins the learner, flushes the metrics summary
        # and persists the pool. SIGTERM would take the process out with no
        # teardown and leave no evidence to judge.
        kill -INT "${DRIVER_PID}" 2>/dev/null || true
        for (( grace = 0; grace < 120; grace++ )); do
            kill -0 "${DRIVER_PID}" 2>/dev/null || break
            sleep 1
        done
        kill -KILL "${DRIVER_PID}" 2>/dev/null || true
        break
    fi
    sleep 5
done
set +e
wait "${DRIVER_PID}"
DRIVER_EXIT=$?
set -e
DRIVER_PID=""
trap - EXIT INT TERM
DRIVER_WALL=$(( $(date +%s) - DRIVER_START ))
# The exit code is NOT a health signal here: _main_multi_arena returns
# `0 if passed_m2 else 1`, and passed_m2 is the M2 gate against the STATIONARY
# dummy, which a self-play run does not clear. The verdict reads the
# `[multi done]` line instead.
log "driver exited ${DRIVER_EXIT} after ${DRIVER_WALL}s (exit 1 is normal: passed_m2 is the M2 dummy gate)"
tail -n 12 "${DRIVER_LOG}" || true

# ===========================================================================
# Phase 3 — the post-run probe, on the run's FINAL checkpoint.
# ===========================================================================
log "phase 3: post-run probe (final checkpoint, strict)"
PROBE_CHECKPOINT="${CHECKPOINT_PATH}"
if [[ ! -f "${PROBE_CHECKPOINT}" ]]; then
    warn "no checkpoint at ${CHECKPOINT_PATH}; probing the warm start instead."
    warn "  (a missing checkpoint is itself reported by CHECKPOINT_UNLOADABLE)"
    PROBE_CHECKPOINT="${WARM_START}"
fi
# The driver has exited, so its collectors have dropped their connections; the
# probe is once again the only client. established_peers re-checks that inside
# run_probe rather than assuming it.
run_probe "${PROBE_CHECKPOINT}" "${PROBE_EPISODES}" "${PROBE_POST_JSON}" "post"

# ===========================================================================
# Phase 4 — collect the evidence, then decide.
# ===========================================================================
log "phase 4: collecting evidence"
(
    cd "${REPO_ROOT}" && \
    CANARY_VERDICT_MODULE="${VERDICT_MODULE}" \
    CANARY_EVIDENCE="${EVIDENCE_JSON}" \
    CANARY_PROBE_PRE="${PROBE_PRE_JSON}" \
    CANARY_PROBE_POST="${PROBE_POST_JSON}" \
    CANARY_DRIVER_LOG="${DRIVER_LOG}" \
    CANARY_DRIVER_EXIT="${DRIVER_EXIT}" \
    CANARY_DRIVER_WALL="${DRIVER_WALL}" \
    CANARY_DRIVER_STARTED_AT="${DRIVER_START}" \
    CANARY_EVAL_EPISODES="${EVAL_EPISODES}" \
    CANARY_DEADLINE_HIT="${DEADLINE_HIT}" \
    CANARY_DEADLINE_SECONDS="${DEADLINE_SECONDS}" \
    CANARY_RUN_NAME="${RUN_NAME}" \
    CANARY_ARENAS="${ARENAS}" \
    CANARY_MIN_REPLAY="${MIN_REPLAY}" \
    CANARY_MIN_REPLAY_PRODUCTION="${MIN_REPLAY_PRODUCTION}" \
    CANARY_MAX_GRAD_STEPS="${MAX_GRAD_STEPS}" \
    CANARY_CHECKPOINT="${CHECKPOINT_PATH}" \
    CANARY_WARM_START="${WARM_START}" \
    CANARY_WARM_START_SHA256="${WARM_START_SHA256}" \
    "${PYTHON_BIN}" - <<'CANARY_COLLECT_PY'
"""Assemble the evidence document. All arithmetic lives in canary_verdict."""
from __future__ import annotations

import importlib.util
import json
import os
import re
from typing import Any, Dict, Optional

spec = importlib.util.spec_from_file_location(
    "canary_verdict", os.environ["CANARY_VERDICT_MODULE"]
)
verdict_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verdict_module)


def read_json(path: str) -> Optional[Any]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


# --- the driver's own teardown line ---------------------------------------
# `[multi done] reason=X episodes=N grad_steps=M passed_m2=B checkpoints_saved=K`
# is the ONLY reliable completion signal: the process exit code is
# `0 if passed_m2 else 1`, and a self-play run never clears the M2 dummy gate.
driver: Dict[str, Any] = {
    "completed": False,
    "exit_code": int(os.environ["CANARY_DRIVER_EXIT"]),
    "deadline_hit": os.environ["CANARY_DEADLINE_HIT"] == "true",
    "stop_reason": None,
    "episodes": None,
    "grad_steps": None,
    "checkpoints_saved": None,
    "log_path": os.environ["CANARY_DRIVER_LOG"],
}
try:
    with open(
        os.environ["CANARY_DRIVER_LOG"], "r", encoding="utf-8", errors="replace"
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

# --- the snapshot pool -----------------------------------------------------
pool: Dict[str, Any] = {"ok": False, "error": "not loaded"}
snapshot_entry: Dict[str, Any] = {"path": None, "loadable": False, "error": "no snapshot"}
try:
    import torch

    from agent.train import snapshot_pool_directory
    from eval.evaluate import _load_drqn
    from opponents.snapshot_pool import SnapshotPool

    directory = snapshot_pool_directory(os.environ["CANARY_RUN_NAME"])
    loaded = SnapshotPool.load(directory, sampling="pfsp")
    records = loaded.records()
    pool = {
        "ok": True,
        "error": None,
        "directory": directory,
        "sampling": loaded.sampling,
        "size": len(loaded),
        "snapshot_ids": [r.snapshot_id for r in records],
        "pinned_ids": [r.snapshot_id for r in records if r.pinned],
        "grad_steps": {str(r.snapshot_id): r.grad_step for r in records},
        "matches_scored": int(loaded.matches_scored),
        "draws_scored": int(loaded.draws_scored),
        "rated_matches": int(loaded.rated_matches),
        "learner_elo_rated": float(loaded.learner_elo_rated),
        "learner_elo_online": float(loaded.learner_elo_online),
        # Read AS-IS: SnapshotPool.pfsp_weights is contracted to be finite and
        # normalized, and this canary is its only live check.
        "pfsp_weights": {str(k): float(v) for k, v in loaded.pfsp_weights().items()},
        "newest_snapshot_id": None,
        "newest_snapshot_mtime": None,
        "newest_vs_snapshot0_max_abs_delta": None,
        "newest_vs_snapshot0_error": None,
        "newest_vs_second_newest_max_abs_delta": None,
        "newest_vs_second_newest_error": None,
    }

    baseline = loaded.get(0)
    newest = max(records, key=lambda r: r.snapshot_id) if records else None
    older = (
        [r for r in records if r.snapshot_id < newest.snapshot_id]
        if newest is not None
        else []
    )
    second_newest = max(older, key=lambda r: r.snapshot_id) if older else None
    if newest is not None:
        pool["newest_snapshot_id"] = newest.snapshot_id
        snapshot_entry = {"path": newest.path, "loadable": False, "error": None}
        try:
            _load_drqn(newest.path, torch.device("cpu"))
            snapshot_entry["loadable"] = True
        except Exception as exc:  # noqa: BLE001
            snapshot_entry["error"] = f"{type(exc).__name__}: {exc}"
        # PROVENANCE. The only DIRECT evidence that THIS run produced the newest
        # snapshot, and the only signal that catches an attempt which archived
        # NOTHING: build_snapshot_opponents reloads an existing pool, so the
        # previous attempt's snapshot then sits here as `newest` with a healthy
        # grad step and a healthy weight delta. STALE_POOL compares this against
        # driver_started_at.
        try:
            pool["newest_snapshot_mtime"] = float(os.path.getmtime(newest.path))
        except OSError as exc:
            pool["newest_snapshot_mtime"] = None
            pool["newest_snapshot_mtime_error"] = f"{type(exc).__name__}: {exc}"

    def max_abs_delta(left: Any, right: Any) -> Any:
        """(delta, error) between two snapshot records' weights."""
        try:
            first = loaded.load_state_dict(left)
            last = loaded.load_state_dict(right)
        except Exception as exc:  # noqa: BLE001
            return None, f"{type(exc).__name__}: {exc}"
        if set(first) != set(last):
            return None, (
                f"snapshots {left.snapshot_id} and {right.snapshot_id} have "
                "different parameter sets"
            )
        try:
            delta = 0.0
            for key in first:
                delta = max(
                    delta,
                    float(
                        torch.max(
                            torch.abs(
                                last[key].to(torch.float64)
                                - first[key].to(torch.float64)
                            )
                        ).item()
                    ),
                )
        except Exception as exc:  # noqa: BLE001
            return None, f"{type(exc).__name__}: {exc}"
        return delta, None

    # TWO diffs: against the seed AND against the snapshot immediately before
    # this one. A pool whose snapshots 1..N are clones of each other but differ
    # from snapshot 0 passes the first comparison and fails the second, and that
    # is an archive hook which froze after its first successful write.
    for key_prefix, other in (
        ("newest_vs_snapshot0", baseline),
        ("newest_vs_second_newest", second_newest),
    ):
        if newest is None:
            pool[f"{key_prefix}_error"] = "the pool holds no snapshots at all"
            continue
        if other is None or other.snapshot_id == newest.snapshot_id:
            pool[f"{key_prefix}_error"] = (
                f"the pool holds only snapshot {newest.snapshot_id}; there is "
                "nothing to diff"
            )
            continue
        delta, error = max_abs_delta(other, newest)
        pool[f"{key_prefix}_max_abs_delta"] = delta
        pool[f"{key_prefix}_error"] = error
except Exception as exc:  # noqa: BLE001
    pool = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

# --- the shipped checkpoint -----------------------------------------------
final_entry: Dict[str, Any] = {
    "path": os.environ["CANARY_CHECKPOINT"],
    "loadable": False,
    "error": None,
}
try:
    import torch

    from eval.evaluate import _load_drqn

    _load_drqn(final_entry["path"], torch.device("cpu"))
    final_entry["loadable"] = True
except Exception as exc:  # noqa: BLE001
    final_entry["error"] = f"{type(exc).__name__}: {exc}"

# --- the eval's numbers, read from the DRIVER LOG --------------------------
# NOT from runs/<run>/summary.json. On the multi-arena path the ONLY
# `logger.summary()` call in agent/train.py is
#     logger.summary(selfplay_log_row(snapshot_pool))       (_summarize_selfplay)
# and that row's keys are elo/*, selfplay/pool_size, selfplay/matches_scored,
# selfplay/rated_matches, selfplay/draw_rate and selfplay/win_rate_vs_ref_*.
# There is no `win_rate` and no `mean_episode_length` in it at all, so a
# collector gated on `"win_rate" in summary` recorded {"ran": False} on EVERY
# run and T19's eval episode-length figure was always None. Fail-closed rather
# than wrong - but silently absent all the same.
#
# The eval's mean episode length DOES reach this log, in two lines agent/train.py
# prints from the SAME `report` object:
#   [multi grad_step N] win_rate=.. mean_len=.. aim_invisible=.. passed_m2=.. opponent=NAME
#       (train_multi_arena, once per eval cycle - the only line naming the opponent)
#   "  last eval: win_rate=.. mean_len=.. aim_invisible=.."
#       (_main_multi_arena teardown, printing result.last_report)
# The teardown line is authoritative for the final numbers; the cycle line is
# where the opponent's NAME comes from. On a self-play run that opponent is the
# fixed SCRIPTED yardstick - build_eval_opponent returns the same scripted
# driver for cfg.opponent == "selfplay" - so this figure must never be labelled
# "eps=0 both sides". The rated reference gauntlet is the eps=0 half, and it
# reports Elo, never an episode length.
_EVAL_CYCLE_RE = re.compile(
    r"\[multi grad_step (?P<grad>\d+)\] win_rate=(?P<win>[-+0-9.eE]+) "
    r"mean_len=(?P<mean>[-+0-9.eE]+) aim_invisible=(?P<aim>[-+0-9.eE]+) "
    r"passed_m2=(?P<passed>\S+) opponent=(?P<opponent>\S+)"
)
_LAST_EVAL_RE = re.compile(
    r"last eval: win_rate=(?P<win>[-+0-9.eE]+) "
    r"mean_len=(?P<mean>[-+0-9.eE]+) aim_invisible=(?P<aim>[-+0-9.eE]+)"
)


def last_match(pattern: Any, text: str) -> Any:
    """The LAST match, or None. Later eval cycles supersede earlier ones."""
    found = None
    for found in pattern.finditer(text):
        pass
    return found


cycle_match = last_match(_EVAL_CYCLE_RE, log_text)
final_match = last_match(_LAST_EVAL_RE, log_text)
numbers = final_match or cycle_match
if numbers is None:
    eval_section: Dict[str, Any] = {
        "ran": False,
        "error": "no eval line in the driver log - no eval cycle completed",
    }
else:
    eval_section = {
        "ran": True,
        "source": os.environ["CANARY_DRIVER_LOG"],
        "win_rate": float(numbers.group("win")),
        "mean_episode_length": float(numbers.group("mean")),
        "aim_while_invisible": float(numbers.group("aim")),
        # Named by the driver itself, so the figure carries WHO it was measured
        # against. None only when the per-cycle line is missing entirely.
        "opponent": cycle_match.group("opponent") if cycle_match else None,
        "grad_step": int(cycle_match.group("grad")) if cycle_match else None,
        # CONFIGURED, not counted: --eval-episodes is what each cycle runs, and
        # the driver log never prints how many it completed. Named so.
        "episodes_per_cycle": int(os.environ["CANARY_EVAL_EPISODES"]),
    }

# --- the epsilon schedule, from the jsonl metrics sink ---------------------
# Its own try: an unreadable metrics.jsonl must not wipe the eval numbers above.
epsilon_section: Dict[str, Any] = {"schedule": None, "mean": None}
try:
    from eval.logging import read_jsonl

    run_dir = os.path.join("runs", os.environ["CANARY_RUN_NAME"])
    for row in read_jsonl(os.path.join(run_dir, "metrics.jsonl")):
        if "train/epsilon_schedule" in row:
            epsilon_section["schedule"] = row.get("train/epsilon_schedule")
        if "train/epsilon_mean" in row:
            epsilon_section["mean"] = row.get("train/epsilon_mean")
except Exception as exc:  # noqa: BLE001
    epsilon_section["error"] = f"{type(exc).__name__}: {exc}"

evidence = {
    "evidence_version": verdict_module.EVIDENCE_VERSION,
    "evidence_path": os.environ["CANARY_EVIDENCE"],
    "run_name": os.environ["CANARY_RUN_NAME"],
    "arenas": int(os.environ["CANARY_ARENAS"]),
    "min_replay": int(os.environ["CANARY_MIN_REPLAY"]),
    "min_replay_production": int(os.environ["CANARY_MIN_REPLAY_PRODUCTION"]),
    "max_grad_steps": int(os.environ["CANARY_MAX_GRAD_STEPS"]),
    "warm_start": os.environ["CANARY_WARM_START"],
    "warm_start_sha256": os.environ["CANARY_WARM_START_SHA256"],
    "driver_wall_seconds": float(os.environ["CANARY_DRIVER_WALL"]),
    # Epoch seconds, captured by the shell immediately before the driver was
    # launched. STALE_POOL compares the newest snapshot's mtime against it.
    "driver_started_at": float(os.environ["CANARY_DRIVER_STARTED_AT"]),
    "deadline_seconds": int(os.environ["CANARY_DEADLINE_SECONDS"]),
    "driver": driver,
    "pool": pool,
    "eval": eval_section,
    "epsilon": epsilon_section,
    "checkpoint": {"final": final_entry, "snapshot": snapshot_entry},
    "probe_pre": verdict_module.summarize_probe(read_json(os.environ["CANARY_PROBE_PRE"])),
    "probe_post": verdict_module.summarize_probe(read_json(os.environ["CANARY_PROBE_POST"])),
}

with open(os.environ["CANARY_EVIDENCE"], "w", encoding="utf-8") as handle:
    json.dump(evidence, handle, indent=2, sort_keys=True)
    handle.write("\n")
CANARY_COLLECT_PY
) || {
    # Exit 3, NOT 2. Exit 2 promises "nothing was run", and by this point the
    # budget has been spent and the fleet driven for half an hour - what failed
    # is the judging, not the setup. Same operational meaning as a refusal.
    echo "[canary] FATAL: evidence collection failed (traceback above)." >&2
    echo "[canary]   The run HAPPENED; nothing was JUDGED, so NOTHING is" >&2
    echo "[canary]   cleared. Treat this exactly as a refusal and fix the" >&2
    echo "[canary]   collector before the long run." >&2
    exit 3
}

log "evidence written to ${EVIDENCE_JSON}"
echo ""
set +e
"${PYTHON_BIN}" "${VERDICT_MODULE}" "${EVIDENCE_JSON}" "${MEASUREMENTS_JSON}"
VERDICT_EXIT=$?
set -e
echo ""
log "T19 measurements written to ${MEASUREMENTS_JSON}"
exit "${VERDICT_EXIT}"
