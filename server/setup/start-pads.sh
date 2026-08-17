#!/usr/bin/env bash
# start-pads.sh — boot the N-pad training fleet: ONE Paper JVM, N Node bridges (T10).
#
# Replaces the N-JVM PowerShell orchestrator (start-arenas.ps1, deleted). The world
# now holds N enclosed bedrock pads inside ONE flat world served by ONE Paper JVM on
# port 25565. Pad i gets its own bridge process on TCP port 5555+i, its own anchor,
# and its own bot pair (learner_bot/dummy_bot at i==0, learner_<i>/dummy_<i> above).
#
# WHERE THE NUMBERS COME FROM. This script computes NO coordinates. It asks
# `python -m distributed.launcher --emit-plan` for the per-pad anchors, ports,
# usernames and the exact bridge argv, because padAnchor(i) has exactly one
# implementation in this repo (distributed/launcher.py) and a bash mirror of it is
# the failure mode the plan explicitly forbids.
#
# BOOT SEQUENCE (the run order for anything live is Paper -> bridges -> driver):
#   1. Preflight gates (--check runs only this and exits).
#   2. Write ops.json for all 2N bots — Paper reads the op list at BOOT.
#   3. Start Paper via start.sh (which pins Java 21 and verifies the jar), wait for
#      the Minecraft port.
#   4. Start the N bridges one at a time, each gated on its own TCP port opening.
#      run.js connects BOTH bots and waits for both spawns BEFORE it calls listen(),
#      so "port 5555+i is open" is exactly "pad i's two bots joined and spawned" —
#      that is the join gate, and serializing on it IS the stagger.
#   5. Prime: reset every pad once, descending, before any pad may step.
#   6. Print FLEET READY and supervise until Ctrl-C or the JVM dies.
#
# FAULT POLICY (two tiers, and the tier matters). Losing the shared JVM aborts the
# whole run loudly -- every pad went with it. Losing ONE pad's bridge affects that
# pad only: it is reported with its log tail and supervision continues, because the
# training driver's ActorPool restarts exactly that bridge through
# SubprocessArenaLauncher.launch(i). Tearing the fleet down for one dead bridge
# would be the wrong tier AND would make that recovery path unreachable. Before
# FLEET READY the rule is inverted -- a pad that will not boot is fatal, since a
# half-built fleet has every bot still stacked at the shared world spawn.
#
# WHY STEP 5 IS NOT OPTIONAL. `arena:setup` sets ONE world spawn at 0 64 0, which is
# pad 0's anchor, so at fleet boot all 2N bots join STACKED inside pad 0 and only
# leave when their own first arena:reset_pad runs. PvP is on (the damage channel
# needs it) and the bedrock walls do not help while everyone is inside the same
# walls: a pad that starts stepping early lands real hits on idle foreign bots and
# registers damage_taken on THEIR bridges. So every pad is reset before any pad may
# step, and the driver must not be started until FLEET READY is printed — a driver
# connecting mid-prime also steals the bridge's single TCP client slot.
#
# WHAT FLEET READY DOES NOT MEAN. A reset ack, even a gated one, does not prove a
# pad's geometry exists (issue #27). Ready here means "2N bots placed at their
# anchors", never "arena verified".
#
# Prerequisites: server/setup/setup.sh has been run with the same PADS (it sizes
# max-players and installs the datapack), node on PATH, and a Python >=3.11 venv
# with requirements.txt installed (system python 3.9 cannot import the env package).

set -euo pipefail

# --- Pinned constants (mirror setup.sh / start.sh) -------------------------
PAPER_VERSION="1.21.1"
PAPER_BUILD="133"
JAR_NAME="paper-${PAPER_VERSION}-${PAPER_BUILD}.jar"

# --- Defaults --------------------------------------------------------------
PADS=1
MC_PORT=25565            # ONE JVM: a single port, not a base.
BRIDGE_BASE_PORT=5555    # pad i listens on BRIDGE_BASE_PORT + i.
STAGGER_SECONDS=2        # extra pause after a pad's port opens, before the next.
BRIDGE_TIMEOUT=120       # bounded wait for one bridge's port (both bots joining).
SERVER_TIMEOUT=300       # bounded wait for the Minecraft port (cold boot + world-gen).
BRIDGE_ATTEMPTS=3        # per-pad start attempts before the fleet fails.
BRIDGE_BACKOFF=5         # initial seconds between pad attempts; doubles per retry.
DO_CHECK=0
DO_DRY_RUN=0
START_SERVER=1
DO_PRIME=1
LOG_TAIL_LINES=40
PYTHON_BIN=""
LOG_DIR=""

# --- Resolve paths ---------------------------------------------------------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
SERVER_DIR="${SERVER_DIR:-$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)}"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd)}"

usage() {
    cat <<USAGE
usage: server/setup/start-pads.sh [--pads N] [options]

  --pads N            Number of pads (one bridge + two bots each). Default 1.
  --check             Run the preflight gates only, print every result, then exit.
  --dry-run           Print the launch plan and exit. Starts nothing.
  --no-server         Do NOT start Paper; attach to an already-running JVM.
  --no-prime          Skip the reset-before-step barrier. NOT recommended.
  --stagger SECONDS   Pause between pads after each one's port opens. Default ${STAGGER_SECONDS}.
  --bridge-timeout S  Bounded wait for one bridge's port. Default ${BRIDGE_TIMEOUT}.
  --server-timeout S  Bounded wait for the Minecraft port. Default ${SERVER_TIMEOUT}.
  --attempts K        Start attempts per pad before failing. Default ${BRIDGE_ATTEMPTS}.
  --mc-port P         The shared Minecraft port. Default ${MC_PORT}.
  --bridge-base-port P  Pad i uses P+i. Default ${BRIDGE_BASE_PORT}.
  --python PATH       Interpreter. Default \$PYTHON, else <repo>/.venv/bin/python.
  --log-dir DIR       Per-pad bridge logs. Default <server>/logs/pads.
  --xms VALUE         JVM initial heap passed to start.sh (e.g. 4G).
  --xmx VALUE         JVM max heap passed to start.sh (e.g. 4G).
  -h, --help          This message.

Run server/setup/setup.sh with the SAME pad count first (PADS=N setup.sh): it
sizes max-players, which this script verifies and refuses to launch without.
USAGE
}

# --- Parse arguments -------------------------------------------------------
need_value() {
    # need_value <flag> <count-remaining>
    if [[ "$2" -lt 2 ]]; then
        echo "[start-pads] $1 requires a value." >&2
        exit 1
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pads)              need_value "$1" $#; PADS="$2"; shift 2 ;;
        --pads=*)            PADS="${1#--pads=}"; shift ;;
        --check)             DO_CHECK=1; shift ;;
        --dry-run)           DO_DRY_RUN=1; shift ;;
        --no-server)         START_SERVER=0; shift ;;
        --no-prime)          DO_PRIME=0; shift ;;
        --stagger)           need_value "$1" $#; STAGGER_SECONDS="$2"; shift 2 ;;
        --bridge-timeout)    need_value "$1" $#; BRIDGE_TIMEOUT="$2"; shift 2 ;;
        --server-timeout)    need_value "$1" $#; SERVER_TIMEOUT="$2"; shift 2 ;;
        --attempts)          need_value "$1" $#; BRIDGE_ATTEMPTS="$2"; shift 2 ;;
        --mc-port)           need_value "$1" $#; MC_PORT="$2"; shift 2 ;;
        --bridge-base-port)  need_value "$1" $#; BRIDGE_BASE_PORT="$2"; shift 2 ;;
        --python)            need_value "$1" $#; PYTHON_BIN="$2"; shift 2 ;;
        --log-dir)           need_value "$1" $#; LOG_DIR="$2"; shift 2 ;;
        --xms)               need_value "$1" $#; export XMS="$2"; shift 2 ;;
        --xmx)               need_value "$1" $#; export XMX="$2"; shift 2 ;;
        -h|--help)           usage; exit 0 ;;
        *)
            echo "[start-pads] unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

for pair in "PADS ${PADS}" "MC_PORT ${MC_PORT}" "BRIDGE_BASE_PORT ${BRIDGE_BASE_PORT}" \
            "BRIDGE_ATTEMPTS ${BRIDGE_ATTEMPTS}"; do
    name="${pair%% *}"
    value="${pair#* }"
    if ! [[ "${value}" =~ ^[0-9]+$ ]] || [[ "${value}" -lt 1 ]]; then
        echo "[start-pads] ${name} must be a positive integer, got '${value}'." >&2
        exit 1
    fi
done
# Durations may be 0 (no stagger / no wait) but must still be plain integers: they
# are compared arithmetically, and a stray word would abort mid-boot under set -u.
for pair in "STAGGER_SECONDS ${STAGGER_SECONDS}" "BRIDGE_TIMEOUT ${BRIDGE_TIMEOUT}" \
            "SERVER_TIMEOUT ${SERVER_TIMEOUT}" "BRIDGE_BACKOFF ${BRIDGE_BACKOFF}"; do
    name="${pair%% *}"
    value="${pair#* }"
    if ! [[ "${value}" =~ ^[0-9]+$ ]]; then
        echo "[start-pads] ${name} must be a non-negative integer, got '${value}'." >&2
        exit 1
    fi
done

REQUIRED_PLAYERS=$(( 2 * PADS + 10 ))
LOG_DIR="${LOG_DIR:-${SERVER_DIR}/logs/pads}"
PROPS_FILE="${SERVER_DIR}/server.properties"
OPS_FILE="${SERVER_DIR}/ops.json"
RUN_JS="${REPO_ROOT}/bridge/run.js"
JAR_PATH="${SERVER_DIR}/${JAR_NAME}"
DATAPACK_MARKER="${SERVER_DIR}/world/datapacks/arena/pack.mcmeta"

log()  { echo "[start-pads] $*"; }
warn() { echo "[start-pads] WARNING: $*" >&2; }
die()  { echo "[start-pads] FATAL: $*" >&2; exit 1; }

# --- Resolve the Python interpreter ----------------------------------------
# Every mode needs it: the plan, ops.json and the prime barrier all come from
# distributed/launcher.py, which is the single source of truth for pad geometry.
if [[ -z "${PYTHON_BIN}" ]]; then
    PYTHON_BIN="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"
fi
if [[ ! -x "${PYTHON_BIN}" ]] && ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    die "python interpreter not found: ${PYTHON_BIN}
      Create the venv first (system python 3.9 is too old):
        python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
      Or pass --python /path/to/python."
fi

# launcher_py <args...> — run the launcher CLI from the repo root so the package
# imports whether or not it was pip-installed into the venv.
launcher_py() {
    ( cd "${REPO_ROOT}" && "${PYTHON_BIN}" -m distributed.launcher "$@" )
}

# connect_probe <host> <port> — 0 when a TCP connect succeeds within one second.
#
# !! THIS IS NOT A READ-ONLY CHECK ON A BRIDGE PORT. !!
# BridgeServer accepts exactly ONE TCP client, and its _onConnection resolves a
# second one by DESTROYING the incumbent (bridge/transport.js). So a connect probe
# against a bridge that a training driver is attached to silently kills that
# driver's connection. It is named "connect_probe", not "port_open", so every call
# site reads as the mutation it is.
#
# Legal call sites (safe by TIMING, not by construction — re-check if you move one):
#   * the Minecraft port, always: Paper is a normal multi-client server.
#   * a bridge port during BOOT (wait_for_port), before any driver exists.
# For anything else use listener_pids / bridge_pids_on_port below, which detect a
# listener WITHOUT opening a connection.
connect_probe() {
    "${PYTHON_BIN}" -c '
import socket, sys
try:
    socket.create_connection((sys.argv[1], int(sys.argv[2])), timeout=1.0).close()
except OSError:
    sys.exit(1)
' "$1" "$2" >/dev/null 2>&1
}

# bridge_pids_on_port <port> — pids of BRIDGE processes serving that pad, or empty.
#
# Non-connecting, so it is safe against a bridge a driver is attached to. Matching
# the argv (rather than any listener) is also what makes "the driver restarted this
# pad" distinguishable from "something unrelated bound this port": only a bridge
# has run.js plus this --bridge-port on its command line.
#
# The pattern is load-bearing in three places:
#   `run\.js `      the trailing space holds because run.js is never the LAST argv
#                   element — every bridge we or the driver spawn passes flags.
#   `[= ]`          run.js accepts both `--bridge-port 5555` and `--bridge-port=5555`.
#   `([^0-9]|$)`    stops --bridge-port 5555 from matching 55550, and still matches
#                   when the port IS the final element.
# BLIND SPOT: run.js also reads the BRIDGE_PORT env var, which never appears in the
# command line, so an env-configured bridge is invisible to this. Everything this
# script and distributed/launcher.py spawn uses the flag.
bridge_pids_on_port() {
    pgrep -f "run\\.js .*--bridge-port[= ]$1([^0-9]|\$)" 2>/dev/null || true
}

# listener_pids <port> — pids listening on that TCP port, or empty. Non-connecting.
#
# Broader than bridge_pids_on_port: answers "is this port free?", so ANY listener
# counts. lsof is the accurate answer.
#
# The argv fallback is strictly weaker and can report a genuinely occupied port as
# free: it misses a non-bridge squatter, and it misses a real bridge configured
# through the BRIDGE_PORT env var (see bridge_pids_on_port's blind spot). A false
# "free" is NOT harmless — it lets the boot spawn a second bridge onto a live port,
# which dies on EADDRINUSE while the incumbent keeps answering. wait_for_port
# re-checks that our own child is still alive before calling a pad up, so the boot
# FAILS LOUDLY and names the port conflict.
#
# Be precise about what that does and does not buy, because the difference is the
# point: by then the probe has already CONNECTED, so the incumbent's single client
# is evicted once regardless. The check does NOT prevent that eviction. It prevents
# the worse second half — a pad reported "up (both bots joined)" on a pid that is
# already dead, then supervised forever against someone else's listener. One loud
# failure instead of a silent phantom.
#
# The 2>/dev/null also swallows lsof's macOS permission warnings for sockets owned
# by another user; that path reports "free" for a port we could not inspect, and
# the same wait_for_port liveness check is the backstop.
listener_pids() {
    if command -v lsof >/dev/null 2>&1; then
        lsof -nP -iTCP:"$1" -sTCP:LISTEN -t 2>/dev/null || true
    else
        bridge_pids_on_port "$1"
    fi
}

# prop_value <file> <key> — the last uncommented "key=value" value, or empty.
prop_value() {
    [[ -f "$1" ]] || return 0
    awk -F'=' -v key="$2" '
        $0 ~ /^[[:space:]]*#/ { next }
        $1 == key { line = $0; sub(/^[^=]*=/, "", line); v = line }
        END { print v }
    ' "$1"
}

# dump_tail <file> — echo the tail of a log so a live failure is diagnosable here.
dump_tail() {
    local path="$1"
    if [[ -f "${path}" ]]; then
        echo "----- last ${LOG_TAIL_LINES} lines of ${path} -----" >&2
        tail -n "${LOG_TAIL_LINES}" "${path}" >&2 || true
        echo "----- end ${path} -----" >&2
    else
        echo "(no log at ${path})" >&2
    fi
}

# --- Dry run: print the plan and stop --------------------------------------
if [[ "${DO_DRY_RUN}" -eq 1 ]]; then
    log "repo root   : ${REPO_ROOT}"
    log "server dir  : ${SERVER_DIR}"
    log "pads        : ${PADS} (needs max-players >= ${REQUIRED_PLAYERS})"
    log "python      : ${PYTHON_BIN}"
    echo ""
    launcher_py --pads "${PADS}" --mc-port "${MC_PORT}" \
        --bridge-base-port "${BRIDGE_BASE_PORT}" --dry-run
    echo ""
    log "dry run: nothing was started."
    exit 0
fi

# --- Preflight -------------------------------------------------------------
# Every gate runs and reports, then the script exits once. A single failure that
# hides five others wastes a live debugging round-trip.
PREFLIGHT_FAILURES=0
pass() { echo "[start-pads]   ok   $*"; }
fail() { echo "[start-pads]   FAIL $*" >&2; PREFLIGHT_FAILURES=$(( PREFLIGHT_FAILURES + 1 )); }

log "preflight for ${PADS} pad(s) (${REQUIRED_PLAYERS} player slots needed)"

if command -v node >/dev/null 2>&1; then
    pass "node on PATH ($(command -v node))"
else
    fail "node not found on PATH. The bridge runs on Node; install it first."
fi

if [[ -f "${RUN_JS}" ]]; then
    pass "bridge entry present (${RUN_JS})"
else
    fail "bridge/run.js not found at ${RUN_JS}"
fi

if ( cd "${REPO_ROOT}" && "${PYTHON_BIN}" -c "import distributed.launcher" ) >/dev/null 2>&1; then
    pass "python can import distributed.launcher (${PYTHON_BIN})"
else
    fail "${PYTHON_BIN} cannot import distributed.launcher (run from ${REPO_ROOT})"
fi

if [[ "${DO_PRIME}" -eq 1 ]]; then
    if ( cd "${REPO_ROOT}" && "${PYTHON_BIN}" -c "import env.mc_pvp_env" ) >/dev/null 2>&1; then
        pass "python can import env.mc_pvp_env (needed by the prime barrier)"
    else
        fail "${PYTHON_BIN} cannot import env.mc_pvp_env; the prime barrier needs it.
       python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt"
    fi
fi

if [[ "${START_SERVER}" -eq 1 ]]; then
    if [[ -f "${JAR_PATH}" ]]; then
        pass "Paper jar present (${JAR_NAME})"
    else
        fail "${JAR_NAME} not found in ${SERVER_DIR}. Run server/setup/setup.sh first."
    fi
    if [[ -f "${SERVER_DIR}/eula.txt" ]]; then
        pass "eula.txt present"
    else
        fail "eula.txt missing. Run server/setup/setup.sh first."
    fi
fi

# Datapack CURRENCY, not just presence. Paper loads the copy under world/datapacks,
# never server/arena, and setup.sh refreshes that copy by rm -rf + re-copy. A world
# carrying a pre-pad-topology pack still passes a presence check, still boots, and
# still acks resets — and a missing macro function is exactly the failure FLEET
# READY cannot see. It is also the difference between getting setup_pad's
# "PAD NOT BUILT" diagnostic and getting silence.
if [[ ! -f "${DATAPACK_MARKER}" ]]; then
    fail "arena datapack missing (${DATAPACK_MARKER}). Run server/setup/setup.sh."
else
    DATAPACK_FN_DIR="${SERVER_DIR}/world/datapacks/arena/data/arena/function"
    MISSING_FUNCTIONS=""
    for fn in setup_pad reset_pad spawn_learner_pad spawn_dummy_pad; do
        [[ -f "${DATAPACK_FN_DIR}/${fn}.mcfunction" ]] || MISSING_FUNCTIONS="${MISSING_FUNCTIONS} arena:${fn}"
    done
    if [[ -n "${MISSING_FUNCTIONS}" ]]; then
        fail "the installed datapack has no${MISSING_FUNCTIONS}. The world copy predates the pad topology.
       Fix: PADS=${PADS} server/setup/setup.sh (it re-copies server/arena into the world)."
    # Compare only what setup.sh actually copies (pack.mcmeta + data/); server/arena
    # also holds a README that is deliberately not installed.
    elif ! diff -r "${SERVER_DIR}/arena/data" "${SERVER_DIR}/world/datapacks/arena/data" >/dev/null 2>&1 \
         || ! diff "${SERVER_DIR}/arena/pack.mcmeta" "${SERVER_DIR}/world/datapacks/arena/pack.mcmeta" >/dev/null 2>&1; then
        fail "the installed datapack differs from server/arena; the world is running stale macro functions.
       $(diff -rq "${SERVER_DIR}/arena/data" "${SERVER_DIR}/world/datapacks/arena/data" 2>&1 | head -n 5)
       Fix: PADS=${PADS} server/setup/setup.sh"
    else
        pass "arena datapack installed and matches server/arena"
    fi
fi

# max-players is the gate the plan calls out by name: 2N bots cannot join a server
# that will not seat them, and the failure looks like a bridge fault, not a config
# one. server.properties is REGENERATED by setup.sh, so the fix is a setup re-run.
if [[ -f "${PROPS_FILE}" ]]; then
    MAX_PLAYERS_VALUE="$(prop_value "${PROPS_FILE}" "max-players")"
    if [[ -z "${MAX_PLAYERS_VALUE}" ]]; then
        fail "no max-players in ${PROPS_FILE}; need at least ${REQUIRED_PLAYERS} (2*${PADS}+10).
       Fix: PADS=${PADS} server/setup/setup.sh"
    elif ! [[ "${MAX_PLAYERS_VALUE}" =~ ^[0-9]+$ ]]; then
        fail "max-players='${MAX_PLAYERS_VALUE}' in ${PROPS_FILE} is not a number; need at least ${REQUIRED_PLAYERS} (2*${PADS}+10).
       Fix: PADS=${PADS} server/setup/setup.sh"
    elif [[ "${MAX_PLAYERS_VALUE}" -lt "${REQUIRED_PLAYERS}" ]]; then
        fail "max-players=${MAX_PLAYERS_VALUE} in ${PROPS_FILE} is too small for ${PADS} pad(s): need at least ${REQUIRED_PLAYERS} (2*${PADS}+10 = 2 bots per pad + 10 headroom).
       Fix: PADS=${PADS} server/setup/setup.sh"
    else
        pass "max-players=${MAX_PLAYERS_VALUE} >= ${REQUIRED_PLAYERS}"
    fi

    SERVER_PORT_VALUE="$(prop_value "${PROPS_FILE}" "server-port")"
    if [[ -n "${SERVER_PORT_VALUE}" && "${SERVER_PORT_VALUE}" != "${MC_PORT}" ]]; then
        fail "server-port=${SERVER_PORT_VALUE} in ${PROPS_FILE} but the bridges are told to use ${MC_PORT}.
       Fix: pass --mc-port ${SERVER_PORT_VALUE}, or re-run server/setup/setup.sh."
    else
        pass "server-port matches the bridge target (${MC_PORT})"
    fi
else
    fail "no ${PROPS_FILE}. Run server/setup/setup.sh (PADS=${PADS}) first."
fi

# Port occupancy. In managed mode nothing may be listening yet; in attach mode the
# JVM must already be up. Either way a bridge port already open means a fleet is
# still running, and a second one would fight it for bots and ports.
# The MINECRAFT port, unlike a bridge port, is a normal multi-client server socket:
# connecting to it and hanging up is what a server-list ping does. Safe to probe.
if [[ "${START_SERVER}" -eq 1 ]]; then
    if connect_probe 127.0.0.1 "${MC_PORT}"; then
        fail "something is already listening on mc port ${MC_PORT}. Stop the running server, or pass --no-server to attach to it."
    else
        pass "mc port ${MC_PORT} is free"
    fi
else
    if connect_probe 127.0.0.1 "${MC_PORT}"; then
        pass "mc port ${MC_PORT} is up (attaching to the running JVM)"
    else
        fail "--no-server was given but nothing answers on mc port ${MC_PORT}."
    fi
    # A running Paper will not re-read ops.json and rewrites it on save, so in
    # attach mode the op list must ALREADY cover every bot.
    if launcher_py --pads "${PADS}" --check-ops --ops-path "${OPS_FILE}" >/dev/null 2>&1; then
        pass "ops.json opps all $(( 2 * PADS )) bot(s)"
    else
        fail "$(launcher_py --pads "${PADS}" --check-ops --ops-path "${OPS_FILE}" 2>&1 || true)
       A running server will not re-read ops.json: write it with
         ${PYTHON_BIN} -m distributed.launcher --pads ${PADS} --write-ops
       and RESTART Paper (drop --no-server)."
    fi
fi

# Say out loud how good the occupancy answer below actually is. Silent degradation
# to "every port is free" is worse than no check: it is the one input that decides
# whether we spawn a bridge onto a live port.
if command -v lsof >/dev/null 2>&1; then
    :
elif command -v pgrep >/dev/null 2>&1; then
    warn "lsof not found: bridge-port occupancy falls back to matching bridge argv."
    warn "  A non-bridge process, or a bridge configured via the BRIDGE_PORT env var,"
    warn "  will not be seen and its port will be reported free."
else
    warn "neither lsof nor pgrep found: bridge-port occupancy CANNOT be checked, and"
    warn "  every port below will be reported free regardless of what is running."
    warn "  Supervision also cannot notice a pad's bridge coming back after a restart."
    warn "  Install either tool before running a fleet on a shared machine."
fi

# Detected WITHOUT connecting. --check is advertised as the safe "gates only" mode,
# and a connect probe here would evict the collector of every pad of a fleet that is
# already running -- turning the safe mode into the destructive one.
OCCUPIED_BRIDGE_PORTS=""
pad_index=0
while [[ "${pad_index}" -lt "${PADS}" ]]; do
    bridge_port=$(( BRIDGE_BASE_PORT + pad_index ))
    if [[ -n "$(listener_pids "${bridge_port}")" ]]; then
        OCCUPIED_BRIDGE_PORTS="${OCCUPIED_BRIDGE_PORTS} ${bridge_port}"
    fi
    pad_index=$(( pad_index + 1 ))
done
if [[ -n "${OCCUPIED_BRIDGE_PORTS}" ]]; then
    fail "bridge port(s) already in use:${OCCUPIED_BRIDGE_PORTS}. A fleet is still running; stop it first."
else
    pass "bridge ports ${BRIDGE_BASE_PORT}..$(( BRIDGE_BASE_PORT + PADS - 1 )) are free"
fi

if [[ "${DO_CHECK}" -eq 1 ]]; then
    if [[ "${PREFLIGHT_FAILURES}" -gt 0 ]]; then
        echo "[start-pads] preflight: ${PREFLIGHT_FAILURES} failure(s)." >&2
        exit 1
    fi
    log "preflight: all gates passed. Nothing was started (--check)."
    exit 0
fi

if [[ "${PREFLIGHT_FAILURES}" -gt 0 ]]; then
    die "preflight: ${PREFLIGHT_FAILURES} failure(s); refusing to launch."
fi

# ===========================================================================
# LIVE path below (UNVERIFIED in-session: spawns a real JVM + real Node bridges).
# ===========================================================================

mkdir -p "${LOG_DIR}"
PAPER_LOG="${LOG_DIR}/paper.log"
PAPER_PID=""
BRIDGE_PIDS=()
# Per-pad supervision state, parallel to BRIDGE_PIDS:
#   child   -- we spawned it and hold its pid
#   down    -- it died after boot; the driver's launcher may replace it
#   adopted -- its port came back on a process that is NOT our child
BRIDGE_STATE=()
TEARDOWN_DONE=0

# stop_pid <pid> <label> — SIGTERM, wait out a grace period, then SIGKILL.
stop_pid() {
    local pid="$1" label="$2" waited=0
    [[ -n "${pid}" ]] || return 0
    kill -0 "${pid}" 2>/dev/null || return 0
    log "stopping ${label} (pid ${pid})"
    kill -TERM "${pid}" 2>/dev/null || true
    while [[ "${waited}" -lt 20 ]]; do
        kill -0 "${pid}" 2>/dev/null || return 0
        sleep 1
        waited=$(( waited + 1 ))
    done
    warn "${label} (pid ${pid}) did not exit; killing."
    kill -KILL "${pid}" 2>/dev/null || true
}

# teardown — bridges first (so they stop talking to a server that is going away),
# then Paper. Idempotent: the EXIT trap and an explicit call must not double-stop.
#
# Only bridges we still hold a pid for are stopped. A pad whose bridge was
# restarted by the training driver belongs to that process tree, so it is named
# here rather than killed — the operator needs to know a stray bridge survived.
teardown() {
    [[ "${TEARDOWN_DONE}" -eq 0 ]] || return 0
    TEARDOWN_DONE=1
    log "tearing the fleet down."
    if [[ "${#BRIDGE_PIDS[@]}" -gt 0 ]]; then
        local i=$(( ${#BRIDGE_PIDS[@]} - 1 ))
        while [[ "${i}" -ge 0 ]]; do
            stop_pid "${BRIDGE_PIDS[$i]}" "bridge for pad ${i}"
            if [[ "${BRIDGE_STATE[$i]:-}" == "adopted" ]]; then
                warn "pad ${i}'s bridge was restarted by the driver and is NOT this"
                warn "  script's child; it is still running on port $(( BRIDGE_BASE_PORT + i ))."
            fi
            i=$(( i - 1 ))
        done
    fi
    if [[ -n "${PAPER_PID}" ]]; then
        stop_pid "${PAPER_PID}" "Paper"
    fi
}
trap teardown EXIT
trap 'log "interrupted."; exit 130' INT TERM

# wait_for_port <port> <timeout> <label> <pid-or-empty> — poll until the port
# accepts, the owning process dies, or the timeout elapses.
wait_for_port() {
    local port="$1" timeout="$2" label="$3" pid="${4:-}" waited=0
    while true; do
        # connect_probe is legal here: this runs at BOOT, against the Minecraft
        # port or against a bridge we just spawned, so no driver is attached yet
        # and there is no incumbent client to evict. It must ALSO stay a real
        # connect for a bridge: run.js listens only after BOTH bots have spawned,
        # so accepting a connection is precisely the join gate. A non-connecting
        # check would pass the moment the process existed and prove nothing.
        if connect_probe 127.0.0.1 "${port}"; then
            # A successful connect proves SOMETHING is listening, not that it is
            # OURS. If the process we spawned is already gone, the answer came from
            # a pre-existing listener: our bridge lost the bind race (EADDRINUSE)
            # and the probe just connected to — and, on a bridge, evicted the single
            # client of — someone else's server. Reporting "up" here would strand a
            # phantom-ready pad that this script would supervise by a pid that has
            # already exited. Fail instead; the caller dumps the log, where the
            # EADDRINUSE is waiting.
            if [[ -n "${pid}" ]] && ! kill -0 "${pid}" 2>/dev/null; then
                warn "${label}: port ${port} is answering, but the process we started"
                warn "  (pid ${pid}) is already gone — that listener is NOT ours."
                warn "  Something else owns port ${port}; stop it before launching."
                return 1
            fi
            return 0
        fi
        if [[ -n "${pid}" ]] && ! kill -0 "${pid}" 2>/dev/null; then
            warn "${label}: process ${pid} exited before port ${port} opened."
            return 1
        fi
        if [[ "${waited}" -ge "${timeout}" ]]; then
            warn "${label}: port ${port} did not open within ${timeout}s."
            return 1
        fi
        sleep 2
        waited=$(( waited + 2 ))
    done
}

log "repo root  : ${REPO_ROOT}"
log "server dir : ${SERVER_DIR}"
log "pads       : ${PADS}  bots: $(( 2 * PADS ))  logs: ${LOG_DIR}"
echo ""
# "FLEET READY" is printed EXACTLY once, at the end, so it stays a greppable
# token: this warning deliberately spells the condition instead of the token.
log "DO NOT start the Python driver until this script says the fleet is ready."
log "  Until every pad has been reset, all 2N bots are stacked in pad 0 and a"
log "  stepping pad will hit foreign bots, crediting damage to the wrong policy."
echo ""

# --- 1. ops.json (BEFORE Paper boots; Paper reads the op list at startup) ---
if [[ "${START_SERVER}" -eq 1 ]]; then
    # server/ops.json is git-IGNORED, like server.properties and bukkit.yml: it is
    # generated per fleet size (2N entries) and Paper rewrites it on shutdown, so
    # tracking it dirtied the tree on every cycle (issue #29). Rewriting it here is
    # the normal path, not something to clean up afterwards.
    launcher_py --pads "${PADS}" --write-ops --ops-path "${OPS_FILE}"
fi

# --- 2. Paper ---------------------------------------------------------------
if [[ "${START_SERVER}" -eq 1 ]]; then
    # Invoked through `bash` rather than as an executable: start.sh is committed
    # without the exec bit, and it is the piece that pins Java 21 (Paper 1.21.1
    # boots on Java 26 and then SIGSEGVs in spark's native profiler) and checks the
    # jar. Never bypass it by calling java directly here.
    log "starting Paper (log: ${PAPER_LOG}); start.sh pins Java 21."
    SERVER_DIR="${SERVER_DIR}" bash "${SCRIPT_DIR}/start.sh" >"${PAPER_LOG}" 2>&1 </dev/null &
    PAPER_PID=$!
    if ! wait_for_port "${MC_PORT}" "${SERVER_TIMEOUT}" "Paper" "${PAPER_PID}"; then
        dump_tail "${PAPER_LOG}"
        die "Paper did not come up on port ${MC_PORT}."
    fi
    log "Paper is up on port ${MC_PORT} (pid ${PAPER_PID})."
else
    log "attaching to the Paper JVM already running on port ${MC_PORT}."
fi

# --- 3. Bridges, one pad at a time, gated on the join ------------------------
# The plan comes from distributed/launcher.py: fields 1..6 are pad_index,
# bridge_port, anchor_x, anchor_z, learner, dummy; every field after that is one
# element of the bridge argv, in order.
PLAN_FILE="${LOG_DIR}/plan.tsv"
launcher_py --pads "${PADS}" --mc-port "${MC_PORT}" \
    --bridge-base-port "${BRIDGE_BASE_PORT}" --emit-plan >"${PLAN_FILE}"

while IFS= read -r plan_line; do
    [[ -n "${plan_line}" ]] || continue
    # Split on tabs only. `set -f` because the split is unquoted on purpose and a
    # path element containing a glob character must not be expanded into filenames.
    OLD_IFS="${IFS}"
    IFS=$'\t'
    set -f
    # shellcheck disable=SC2206  # deliberate word split on tab; no field is empty.
    fields=( ${plan_line} )
    set +f
    IFS="${OLD_IFS}"

    pad="${fields[0]}"
    bridge_port="${fields[1]}"
    anchor_x="${fields[2]}"
    anchor_z="${fields[3]}"
    learner="${fields[4]}"
    dummy="${fields[5]}"
    bridge_argv=( "${fields[@]:6}" )
    pad_log="${LOG_DIR}/pad-${pad}.log"

    attempt=1
    backoff="${BRIDGE_BACKOFF}"
    started=0
    while [[ "${attempt}" -le "${BRIDGE_ATTEMPTS}" ]]; do
        log "pad ${pad}: starting bridge on ${bridge_port} @ anchor ${anchor_x},${anchor_z} (${learner} / ${dummy}) [attempt ${attempt}/${BRIDGE_ATTEMPTS}]"
        ( cd "${REPO_ROOT}" && exec "${bridge_argv[@]}" ) >"${pad_log}" 2>&1 </dev/null &
        bridge_pid=$!

        # run.js connects BOTH bots and awaits both spawns before listen(), so the
        # port opening IS the join gate for this pad's two bots.
        if wait_for_port "${bridge_port}" "${BRIDGE_TIMEOUT}" "pad ${pad} bridge" "${bridge_pid}"; then
            BRIDGE_PIDS[${pad}]="${bridge_pid}"
            BRIDGE_STATE[${pad}]="child"
            log "pad ${pad}: bridge up (pid ${bridge_pid}); both bots joined."
            started=1
            break
        fi

        dump_tail "${pad_log}"
        stop_pid "${bridge_pid}" "pad ${pad} bridge (failed attempt ${attempt})"
        attempt=$(( attempt + 1 ))
        if [[ "${attempt}" -le "${BRIDGE_ATTEMPTS}" ]]; then
            log "pad ${pad}: retrying in ${backoff}s"
            sleep "${backoff}"
            backoff=$(( backoff * 2 ))
        fi
    done

    if [[ "${started}" -ne 1 ]]; then
        die "pad ${pad}: bridge failed ${BRIDGE_ATTEMPTS} time(s) (port ${bridge_port}, anchor ${anchor_x},${anchor_z}, bots ${learner}/${dummy}). See ${pad_log}."
    fi

    if [[ "${STAGGER_SECONDS}" -gt 0 ]]; then
        sleep "${STAGGER_SECONDS}"
    fi
done < "${PLAN_FILE}"

# --- 4. Prime: every pad reset before any pad may step -----------------------
if [[ "${DO_PRIME}" -eq 1 ]]; then
    log "priming ${PADS} pad(s) (descending) — the reset-before-step barrier."
    if ! launcher_py --pads "${PADS}" --prime \
            --bridge-base-port "${BRIDGE_BASE_PORT}" --log-dir "${LOG_DIR}"; then
        die "the prime barrier failed; the fleet is NOT safe to train on. See ${LOG_DIR}."
    fi
else
    warn "--no-prime: pads were NOT reset. All $(( 2 * PADS )) bots are still stacked"
    warn "  in pad 0, and the first pad to step will hit foreign bots. Reset every pad"
    warn "  before stepping any, or run without --no-prime."
fi

# --- 5. Ready ---------------------------------------------------------------
# The banner states exactly what happened and nothing more. "primed" is a claim
# about resets that actually ran, so --no-prime must not borrow it, and neither
# state claims the pads' GEOMETRY was verified — a reset ack cannot show that.
echo ""
# The unprimed state gets a DIFFERENT token, not a qualified "FLEET READY": the
# RUNBOOK tells operators to wait for FLEET READY, and a substring match on an
# unprimed fleet is precisely the mistake this barrier exists to prevent.
if [[ "${DO_PRIME}" -eq 1 ]]; then
    log "FLEET READY: ${PADS} pad(s) primed, $(( 2 * PADS )) bots placed at their anchors."
    log "  This does NOT prove pad geometry exists — a reset ack is not a wall check."
    log "  Bridge ports: ${BRIDGE_BASE_PORT}..$(( BRIDGE_BASE_PORT + PADS - 1 ))   logs: ${LOG_DIR}"
    echo ""
    log "Now, in another terminal:"
    log "  python -m agent.train --arenas ${PADS} --port ${BRIDGE_BASE_PORT} \\"
    log "      --max-episodes 10000 --checkpoint runs/m2_multi.pt --run-name m2_multi"
else
    log "FLEET NOT PRIMED: ${PADS} bridge(s) up, $(( 2 * PADS )) bots joined but NOT placed."
    log "  --no-prime was given: every bot is still stacked in pad 0. Reset EVERY pad"
    log "  before stepping ANY pad, or the damage channel will cross-credit."
    log "  Bridge ports: ${BRIDGE_BASE_PORT}..$(( BRIDGE_BASE_PORT + PADS - 1 ))   logs: ${LOG_DIR}"
    echo ""
    log "No driver command is printed for an unprimed fleet on purpose. Prime it"
    log "yourself first:"
    log "  ${PYTHON_BIN} -m distributed.launcher --pads ${PADS} \\"
    log "      --prime --bridge-base-port ${BRIDGE_BASE_PORT} --log-dir ${LOG_DIR}"
fi
echo ""
log "Ctrl-C here tears down every bridge and the Paper JVM."
log "Fault policy while this runs: one pad's bridge dying is reported and that pad"
log "  alone is affected (the driver's launcher restarts it). The JVM dying aborts."

# --- 6. Supervise ------------------------------------------------------------
# The two-tier fault policy, from the supervisor's side:
#
#   Paper JVM dies -> ABORT loudly. Every pad is gone with it; there is nothing
#     left to supervise and a survivor policy would mean training on nothing.
#
#   One pad's bridge dies -> that pad ONLY. Report it, dump its log, and KEEP
#     SUPERVISING the rest. Killing the fleet here would be the wrong tier, and
#     it would also make the collector's own recovery path unreachable: the
#     ActorPool asks SubprocessArenaLauncher.launch(i) to restart exactly that
#     bridge, which cannot happen if this script has already torn everything down.
#
# A replacement bridge started by the driver is NOT our child, so once a pad dies
# we stop watching its pid and watch its PORT instead. That also lets us log the
# recovery, and it keeps teardown honest: an adopted bridge is not ours to kill.
#
# Note the asymmetry with BOOT: before FLEET READY a pad that will not start IS
# fatal (see the retry loop above), because a fleet that never came up whole has
# no primed pads to protect and every bot is still stacked at the world spawn.
while true; do
    sleep 5

    # Tier 1: the shared JVM. Fatal in both managed and attach mode.
    if [[ "${START_SERVER}" -eq 1 ]] && ! kill -0 "${PAPER_PID}" 2>/dev/null; then
        dump_tail "${PAPER_LOG}"
        die "the Paper JVM (pid ${PAPER_PID}) exited. Every pad is gone with it."
    fi
    # Minecraft port again: multi-client, safe to connect to on every poll.
    if [[ "${START_SERVER}" -eq 0 ]] && ! connect_probe 127.0.0.1 "${MC_PORT}"; then
        die "the Paper JVM stopped answering on port ${MC_PORT}. Every pad is gone with it."
    fi

    # Tier 2: individual bridges. Never fatal after boot.
    pad_index=0
    while [[ "${pad_index}" -lt "${PADS}" ]]; do
        bridge_port=$(( BRIDGE_BASE_PORT + pad_index ))
        # ":-" for the same reason teardown uses it: an unset entry must degrade to
        # one reported pad, not abort the whole supervisor under `set -u`. Every pad
        # is populated before this loop is reachable (a pad that will not start
        # calls die), so this is defence against a future short plan, not a live gap.
        case "${BRIDGE_STATE[${pad_index}]:-}" in
            child)
                if ! kill -0 "${BRIDGE_PIDS[${pad_index}]}" 2>/dev/null; then
                    dump_tail "${LOG_DIR}/pad-${pad_index}.log"
                    warn "pad ${pad_index}'s bridge (pid ${BRIDGE_PIDS[${pad_index}]}) exited."
                    warn "  Its two bots are gone from the world; the other $(( PADS - 1 )) pad(s)"
                    warn "  and the JVM keep running. The training driver's launcher restarts"
                    warn "  this pad's bridge on port ${bridge_port}; watching that port now."
                    BRIDGE_PIDS[${pad_index}]=""
                    BRIDGE_STATE[${pad_index}]="down"
                fi
                ;;
            down|adopted)
                # NEVER connect_probe here. The training driver owns this pad now,
                # and its collector is the bridge's single TCP client: a probe
                # every 5s would destroy that connection every 5s, forever, while
                # the port stayed open and this loop reported everything fine.
                # Matching the bridge's argv is both non-destructive and more
                # specific -- an unrelated process that merely binds the port is
                # not this pad coming back.
                if [[ -n "$(bridge_pids_on_port "${bridge_port}")" ]]; then
                    if [[ "${BRIDGE_STATE[${pad_index}]:-}" == "down" ]]; then
                        log "pad ${pad_index}: a bridge is serving port ${bridge_port} again (not our child; not ours to stop)."
                        BRIDGE_STATE[${pad_index}]="adopted"
                    fi
                elif [[ "${BRIDGE_STATE[${pad_index}]:-}" == "adopted" ]]; then
                    warn "pad ${pad_index}: the replacement bridge on port ${bridge_port} went away again."
                    BRIDGE_STATE[${pad_index}]="down"
                fi
                ;;
            *)
                # No recorded state: this pad was never started (a plan shorter than
                # --pads). Say so ONCE, then let the port-watching states handle it.
                warn "pad ${pad_index}: no supervision state recorded; it was never started."
                warn "  Watching port ${bridge_port} only from here on."
                BRIDGE_STATE[${pad_index}]="down"
                ;;
        esac
        pad_index=$(( pad_index + 1 ))
    done
done
