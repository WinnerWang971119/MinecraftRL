#!/usr/bin/env bash
# start.sh — launch the Paper training server (task T8), Linux/macOS.
#
# Counterpart of start.ps1. Run setup.sh first. Heap defaults to -Xms2G -Xmx2G
# (fixed Xms==Xmx avoids GC resize pauses on a tiny flat arena). Override with
# XMS / XMX env vars. Uses --nogui so it stays a console process.
#
# Java: this script PINS Java 21 (see the JAVA_HOME block below). Paper 1.21.1
# does not run on Java 26 — it boots, then the JVM dies with SIGSEGV inside the
# bundled spark profiler's native library. Set JAVA_HOME yourself to override.
#
# Ops: this script also refuses to launch unless server/ops.json opps learner_bot
# and dummy_bot at level 4. That file is generated, never committed (issue #29);
# setup.sh does not write it because the op list depends on the pad count.

set -euo pipefail

PAPER_VERSION="1.21.1"
PAPER_BUILD="133"
JAR_NAME="paper-${PAPER_VERSION}-${PAPER_BUILD}.jar"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
SERVER_DIR="${SERVER_DIR:-$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)}"
# server/setup/ -> server/ -> repo root. Derived from the SCRIPT location, not from
# SERVER_DIR, so it stays correct when SERVER_DIR points somewhere else.
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd)"
JAR_PATH="${SERVER_DIR}/${JAR_NAME}"
OPS_PATH="${SERVER_DIR}/ops.json"
XMS="${XMS:-2G}"
XMX="${XMX:-2G}"

# --- Pin Java 21 -----------------------------------------------------------
# Paper 1.21.1 targets Java 21. On Java 26 it reaches "Done (6.4s)!" and then
# the JVM aborts:
#     SIGSEGV ... [libasyncProfiler.so+0x10b80]
#     Lookup::fillJavaMethodInfo(MethodInfo*, _jmethodID*, bool)+0x3c
# spark (bundled in Paper, started automatically as the background profiler)
# ships a native async-profiler that reads JVM-internal structures; Java 26
# moved them. Observed on macOS 26.5 arm64 with Temurin 26.0.1+8. Java 26 also
# emits restricted-method and sun.misc.Unsafe deprecation warnings, so the boot
# is not warning-free there either.
#
# An explicit JAVA_HOME always wins. Otherwise, on macOS ask java_home for 21;
# elsewhere fall back to whatever `java` is on PATH.
JAVA_PIN_VERSION="21"

# Platform-appropriate install advice. `brew install --cask temurin@21` is
# macOS-only guidance and a Linux operator would hit it on the cloud VMs these
# POSIX scripts exist to serve.
java_install_hint() {
    if [[ "$(uname -s)" == "Darwin" ]]; then
        echo "    brew install --cask temurin@${JAVA_PIN_VERSION}"
    else
        echo "    Debian/Ubuntu: apt-get install temurin-${JAVA_PIN_VERSION}-jdk"
        echo "    or download:   https://adoptium.net/temurin/releases/?version=${JAVA_PIN_VERSION}"
        echo "    then re-run with JAVA_HOME=/path/to/jdk-${JAVA_PIN_VERSION}"
    fi
}

# java_major <java-binary> — prints the major version (e.g. "21"), or fails.
# Handles both the modern format  openjdk version "21.0.11"  -> 21
# and the legacy one              java version "1.8.0_402"   -> 8
java_major() {
    local raw ver major
    # Deliberately NOT `head -n1`. When JAVA_TOOL_OPTIONS or _JAVA_OPTIONS is
    # set, the JVM prints "Picked up JAVA_TOOL_OPTIONS: ..." BEFORE the version
    # banner — routine on CI runners and cloud images, i.e. exactly the hosts
    # these POSIX scripts exist for. Taking line 1 there would fail to parse and
    # refuse to launch on a perfectly good Java 21. Select the version line.
    raw="$("$1" -version 2>&1 | grep -m1 'version "' || true)"
    ver="$(printf '%s\n' "${raw}" | sed -n 's/.*version "\([^"]*\)".*/\1/p')"
    [[ -z "${ver}" ]] && return 1
    major="${ver%%.*}"
    if [[ "${major}" == "1" ]]; then
        major="$(printf '%s' "${ver}" | cut -d. -f2)"
    fi
    # A non-numeric major (unusual vendor banner) must never reach the `-gt`
    # test at the call site: [[ ]] evaluates that arithmetically, so under
    # `set -u` a word like "unknown" aborts the script with a raw
    # "unbound variable" error instead of printing our refusal message.
    # Failing here routes it into the unparseable branch instead.
    [[ "${major}" =~ ^[0-9]+$ ]] || return 1
    printf '%s\n' "${major}"
}

# macOS can select the pinned JDK directly. /usr/libexec/java_home does not
# exist on Linux, so this block is simply skipped there — the version assertion
# below is what actually protects every platform.
if [[ -z "${JAVA_HOME:-}" && -x /usr/libexec/java_home ]]; then
    if JAVA_HOME="$(/usr/libexec/java_home -v "${JAVA_PIN_VERSION}" 2>/dev/null)"; then
        export JAVA_HOME
    else
        unset JAVA_HOME
        echo "[start] WARNING: no Java ${JAVA_PIN_VERSION} JDK registered with java_home." >&2
        java_install_hint >&2
    fi
fi

if [[ -n "${JAVA_HOME:-}" ]]; then
    JAVA_BIN="${JAVA_HOME}/bin/java"
    if [[ ! -x "${JAVA_BIN}" ]]; then
        echo "JAVA_HOME=${JAVA_HOME} has no executable bin/java." >&2
        exit 1
    fi
elif command -v java >/dev/null 2>&1; then
    JAVA_BIN="java"
else
    echo "java not found on PATH and JAVA_HOME is unset. Install Java ${JAVA_PIN_VERSION}:" >&2
    java_install_hint >&2
    exit 1
fi

# --- Assert the resolved JVM is actually the pinned major version ----------
# This single check covers all three ways a wrong JVM gets here: Linux (where
# the java_home block above never ran and `java` on PATH could be anything), an
# explicit JAVA_HOME pointing at the wrong JDK, and the macOS fallback path.
#
# It ABORTS rather than warns. The Java 26 failure mode is a native SIGSEGV
# several seconds AFTER "Done (…)! For help" — the server looks like it started,
# so the crash reads as a bridge fault, not a JVM fault. Printing the version in
# the banner is not enough; in a launcher it scrolls past. Set
# ALLOW_JAVA_MISMATCH=1 to override deliberately.
JAVA_MAJOR="$(java_major "${JAVA_BIN}" || true)"
if [[ -z "${JAVA_MAJOR}" ]]; then
    echo "[start] Could not parse a version from: ${JAVA_BIN} -version" >&2
    echo "[start] Set ALLOW_JAVA_MISMATCH=1 to launch anyway." >&2
    [[ "${ALLOW_JAVA_MISMATCH:-0}" == "1" ]] || exit 1
elif [[ "${JAVA_MAJOR}" != "${JAVA_PIN_VERSION}" ]]; then
    echo "[start] REFUSING TO LAUNCH: Java ${JAVA_MAJOR} is not the pinned Java ${JAVA_PIN_VERSION}." >&2
    echo "[start]   java   : ${JAVA_BIN}" >&2
    echo "[start]   home   : ${JAVA_HOME:-<from PATH>}" >&2
    echo "[start]   banner : $("${JAVA_BIN}" -version 2>&1 | head -n1)" >&2
    if [[ "${JAVA_MAJOR}" -gt "${JAVA_PIN_VERSION}" ]]; then
        echo "[start] Paper ${PAPER_VERSION} boots on newer JDKs and THEN dies with a native" >&2
        echo "[start] SIGSEGV in the bundled spark profiler, seconds after it reports Done." >&2
        echo "[start] That looks like a bridge failure, which is why this is fatal here." >&2
    fi
    java_install_hint >&2
    echo "[start] Or set ALLOW_JAVA_MISMATCH=1 to launch anyway." >&2
    [[ "${ALLOW_JAVA_MISMATCH:-0}" == "1" ]] || exit 1
    echo "[start] ALLOW_JAVA_MISMATCH=1 set — launching on Java ${JAVA_MAJOR} anyway." >&2
fi

if [[ ! -f "${JAR_PATH}" ]]; then
    echo "${JAR_NAME} not found in ${SERVER_DIR}. Run server/setup/setup.sh first." >&2
    exit 1
fi
if [[ ! -f "${SERVER_DIR}/eula.txt" ]]; then
    echo "eula.txt missing. Run server/setup/setup.sh first." >&2
    exit 1
fi

# --- Assert the arena bots are opped ---------------------------------------
# ops.json is GENERATED, not committed (issue #29): Paper reads the op list at boot
# and rewrites it on shutdown, and its contents depend on the pad count, so tracking
# it dirtied the tree on every cycle. setup.sh deliberately does not write it either
# (the op list depends on N). start-pads.sh and deploy/exhibition.py both write it
# before they reach this script; a bare start.sh on a fresh clone does not.
#
# This ABORTS rather than warns, for the same reason the Java check does. An unopped
# bot cannot run /function or /attribute at all, so the datapack builds no arena and
# every reset fails SILENTLY — the bridge still acks, the bots still join, and the
# fault surfaces much later as nonsense observations.
#
# Deliberately grep-level: this script must validate without the venv on PATH, so no
# JSON parser. All whitespace is stripped first, which makes the check independent of
# the two writers' formatting (Paper rewrites the file without the trailing newline
# our generator emits, and that one byte is what started issue #29).
OPS_WRITE_CMD=".venv/bin/python -m distributed.launcher --pads 1 --write-ops"
if [[ "${OPS_PATH}" != "${REPO_ROOT}/server/ops.json" ]]; then
    OPS_WRITE_CMD="${OPS_WRITE_CMD} --ops-path ${OPS_PATH}"
fi

ops_refusal() {
    echo "[start] REFUSING TO LAUNCH: ${OPS_PATH} does not op the arena bots." >&2
    echo "[start]   reason : $1" >&2
    echo "[start] Paper reads the op list at BOOT. A bot that is not opped at level 4" >&2
    echo "[start] cannot run /function or /attribute, so the arena is never built and" >&2
    echo "[start] every reset fails SILENTLY. That is why this is fatal here." >&2
    echo "[start] This file is generated, not committed (issue #29). Write it with:" >&2
    echo "[start]   cd ${REPO_ROOT} && ${OPS_WRITE_CMD}" >&2
    echo "[start] For a multi-pad fleet use server/setup/start-pads.sh --pads N, which" >&2
    echo "[start] writes the op list for all 2N bots itself." >&2
    exit 1
}

if [[ ! -f "${OPS_PATH}" ]]; then
    ops_refusal "the file does not exist"
fi

# Strip every space/tab/newline so "level": 4 and "level":4 compare the same. The
# 2>/dev/null comes FIRST so an unreadable file reports through our message, not the
# shell's redirection error.
OPS_FLAT="$(tr -d '[:space:]' 2>/dev/null <"${OPS_PATH}" || true)"
UNOPPED=""
for bot in learner_bot dummy_bot; do
    if [[ "${OPS_FLAT}" != *"\"name\":\"${bot}\",\"level\":4"* ]]; then
        UNOPPED="${UNOPPED:+${UNOPPED}, }${bot}"
    fi
done
if [[ -n "${UNOPPED}" ]]; then
    ops_refusal "not opped at level 4: ${UNOPPED}"
fi

echo "[start] Java : $("${JAVA_BIN}" -version 2>&1 | head -n1)"
echo "[start] Home : ${JAVA_HOME:-<PATH>}"
echo "[start] Jar  : ${JAR_PATH}"
echo "[start] Heap : -Xms${XMS} -Xmx${XMX}"
echo "[start] Launching Paper (--nogui). Use 'stop' in the console to shut down."

cd "${SERVER_DIR}"
exec "${JAVA_BIN}" "-Xms${XMS}" "-Xmx${XMX}" -jar "${JAR_NAME}" --nogui
