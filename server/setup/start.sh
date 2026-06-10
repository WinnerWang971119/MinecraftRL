#!/usr/bin/env bash
# start.sh — launch the Paper training server (task T8), Linux/macOS.
#
# Counterpart of start.ps1. Run setup.sh first. Heap defaults to -Xms2G -Xmx2G
# (fixed Xms==Xmx avoids GC resize pauses on a tiny flat arena). Override with
# XMS / XMX env vars. Uses --nogui so it stays a console process.
#
# Requires Java 21+ on PATH (this machine has Java 25 — see server/compat_check.md).

set -euo pipefail

PAPER_VERSION="1.21.1"
PAPER_BUILD="133"
JAR_NAME="paper-${PAPER_VERSION}-${PAPER_BUILD}.jar"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
SERVER_DIR="${SERVER_DIR:-$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)}"
JAR_PATH="${SERVER_DIR}/${JAR_NAME}"
XMS="${XMS:-2G}"
XMX="${XMX:-2G}"

if ! command -v java >/dev/null 2>&1; then
    echo "java not found on PATH. Install Java 21+ (this machine has Java 25)." >&2
    exit 1
fi
if [[ ! -f "${JAR_PATH}" ]]; then
    echo "${JAR_NAME} not found in ${SERVER_DIR}. Run server/setup/setup.sh first." >&2
    exit 1
fi
if [[ ! -f "${SERVER_DIR}/eula.txt" ]]; then
    echo "eula.txt missing. Run server/setup/setup.sh first." >&2
    exit 1
fi

echo "[start] Java : $(java -version 2>&1 | head -n1)"
echo "[start] Jar  : ${JAR_PATH}"
echo "[start] Heap : -Xms${XMS} -Xmx${XMX}"
echo "[start] Launching Paper (--nogui). Use 'stop' in the console to shut down."

cd "${SERVER_DIR}"
exec java "-Xms${XMS}" "-Xmx${XMX}" -jar "${JAR_NAME}" --nogui
