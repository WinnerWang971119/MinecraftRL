#!/usr/bin/env bash
# setup.sh — POSIX/bash setup for the Paper training server (task T8).
#
# Linux/macOS counterpart of setup.ps1, for the CPU-heavy cloud VMs that host
# parallel arenas (project spec §9). Idempotent: re-running only rewrites the
# config files we own and skips the jar download if it already exists.
#
# It:
#   1. Downloads paper-1.21.1-133.jar from PaperMC and verifies its SHA-256
#      (skip if present; FORCE=1 to refresh).
#   2. Writes eula.txt (eula=true — accepting the Mojang EULA).
#   3. Writes a training-tuned server.properties (offline, flat, no mobs,
#      view/sim distance 2, fixed seed, pvp on, survival, normal difficulty,
#      anti-spam friendly).
#   4. Writes bukkit.yml (connection-throttle: -1 so a cross-host bridge fleet's
#      2N bots are not throttle-kicked; loopback is exempt regardless).
#   5. Installs the arena datapack into world/datapacks/.
#
# It does NOT launch the server — use start.sh.
#
# Prerequisites: curl, and Java 21 for start.sh — NOT a newer JDK. Paper 1.21.1
#                boots on Java 26 and then dies with a native SIGSEGV in the
#                bundled spark profiler, so start.sh pins Java 21 explicitly.
#                Pinned by Tv: Paper 1.21.1 build 133, channel STABLE.

set -euo pipefail

# --- Pinned constants (from server/compat_check.md / Tv) -------------------
PAPER_VERSION="1.21.1"
PAPER_BUILD="133"
JAR_NAME="paper-${PAPER_VERSION}-${PAPER_BUILD}.jar"

# PaperMC retired the v2 download API (it now answers 410 Gone) in favour of the
# v3 "fill" API, which serves content-addressed artifacts. The pin is unchanged —
# same project, same version, same build 133, same STABLE channel, same jar name.
# The URL and checksum below are exactly what this endpoint reports:
#   https://fill.papermc.io/v3/projects/paper/versions/1.21.1/builds/133
# Pinning the digest is a stronger guarantee than the old build-number URL was:
# the download either hashes to this value or setup fails.
PAPER_SHA256="39bd8c00b9e18de91dcabd3cc3dcfa5328685a53b7187a2f63280c22e2d287b9"
DOWNLOAD_URL="https://fill-data.papermc.io/v1/objects/${PAPER_SHA256}/${JAR_NAME}"
LEVEL_SEED="8675309"   # fixed seed -> reproducible flat world (logged per spec).

# --- Resolve paths ---------------------------------------------------------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
SERVER_DIR="${SERVER_DIR:-$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)}"
JAR_PATH="${SERVER_DIR}/${JAR_NAME}"
FORCE="${FORCE:-0}"

echo "[setup] Server directory : ${SERVER_DIR}"
echo "[setup] Paper jar        : ${JAR_NAME}"
echo "[setup] Download URL     : ${DOWNLOAD_URL}"

# --- 1. Download the Paper jar (idempotent) --------------------------------
# sha256_of <file> — prints the hex digest, using whichever tool this OS ships
# (macOS has shasum, most Linux distros have sha256sum).
sha256_of() {
    if command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | cut -d' ' -f1
    elif command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | cut -d' ' -f1
    else
        echo "[setup] Neither shasum nor sha256sum found; cannot verify the jar." >&2
        return 1
    fi
}

# A pin is only worth anything if it is checked on EVERY run, not just the run
# that happens to download. An already-present jar can be corrupt, a leftover
# from the old (now dead) v2 URL, hand-copied, or tampered with — and start.sh
# would execute it either way. Hashing 47 MB costs about a second.
if [[ -f "${JAR_PATH}" && "${FORCE}" != "1" ]]; then
    EXISTING_SHA256="$(sha256_of "${JAR_PATH}")"
    if [[ "${EXISTING_SHA256}" != "${PAPER_SHA256}" ]]; then
        echo "[setup] SHA-256 MISMATCH on the existing ${JAR_NAME} — refusing to use it." >&2
        echo "[setup]   expected ${PAPER_SHA256}" >&2
        echo "[setup]   actual   ${EXISTING_SHA256}" >&2
        echo "[setup]   path     ${JAR_PATH}" >&2
        echo "[setup] This jar is not Paper ${PAPER_VERSION} build ${PAPER_BUILD}. Delete it, or" >&2
        echo "[setup] re-run with FORCE=1 to replace it with the pinned build." >&2
        exit 1
    fi
    echo "[setup] Jar already present, sha256 verified (FORCE=1 to refresh)."
else
    # FORCE=1 skips the check above, so a bad jar could still be sitting at the
    # final path. If curl then fails, `set -e` aborts before the replacement is
    # moved into place and that unverified jar survives — and start.sh only
    # tests -f. Remove a known-bad jar BEFORE the download so that after any
    # setup.sh run, successful or not, the jar at JAR_PATH is either verified
    # or absent. A jar that already matches the pin is left alone.
    if [[ -f "${JAR_PATH}" ]]; then
        if [[ "$(sha256_of "${JAR_PATH}")" != "${PAPER_SHA256}" ]]; then
            echo "[setup] Existing ${JAR_NAME} does not match the pin; removing it first."
            rm -f "${JAR_PATH}"
        fi
    fi
    echo "[setup] Downloading Paper jar ..."
    curl -fL --retry 3 -o "${JAR_PATH}.partial" "${DOWNLOAD_URL}"
    ACTUAL_SHA256="$(sha256_of "${JAR_PATH}.partial")"
    if [[ "${ACTUAL_SHA256}" != "${PAPER_SHA256}" ]]; then
        rm -f "${JAR_PATH}.partial"
        echo "[setup] SHA-256 mismatch for ${JAR_NAME}." >&2
        echo "[setup]   expected ${PAPER_SHA256}" >&2
        echo "[setup]   actual   ${ACTUAL_SHA256}" >&2
        exit 1
    fi
    mv -f "${JAR_PATH}.partial" "${JAR_PATH}"
    echo "[setup] Saved ${JAR_NAME} (sha256 verified)."
fi

# --- 2. eula.txt -----------------------------------------------------------
cat > "${SERVER_DIR}/eula.txt" <<'EOF'
# By setting eula=true you agree to the Minecraft EULA:
# https://aka.ms/MinecraftEULA
# Written by server/setup/setup.sh (task T8).
eula=true
EOF
echo "[setup] Wrote eula.txt (eula=true)."

# --- 3. server.properties (training-tuned) ---------------------------------
cat > "${SERVER_DIR}/server.properties" <<EOF
# Minecraft server properties — training arena (task T8).
# Generated by server/setup/setup.sh. Idempotent: re-running rewrites this file.
# Rationale for the non-default values lives in server/README.md.

# --- Identity / networking ---
online-mode=false
server-port=25565
server-ip=
network-compression-threshold=256
prevent-proxy-connections=false
enforce-secure-profile=false

# --- World generation (flat, deterministic) ---
# DO NOT "FIX" generator-settings={}. It does not parse; Paper logs
#   ERROR: No key layers in MapLike[{}]
# at world creation and falls back to the DEFAULT flat preset. That fallback is
# the intended, empirically verified world: it places grass_block y=-61,
# dirt y=-62/-63, bedrock y=-64 everywhere, including outside the arena pad.
# Combined with `gamerule fallDamage false`, that is why walking off the y=63
# platform STRANDS the agent alive at y=-60 instead of killing it — the void is
# unreachable. Column scan and consequences: server/compat_check.md.
# Supplying real layers here would change world topology and invalidate that
# analysis. Leave it exactly as it is.
level-name=world
level-type=minecraft:flat
level-seed=${LEVEL_SEED}
generate-structures=false
generator-settings={}
allow-nether=false
allow-flight=true

# --- Gameplay ---
gamemode=survival
force-gamemode=true
difficulty=normal
pvp=true
hardcore=false
spawn-monsters=false
spawn-animals=false
spawn-npcs=false
spawn-protection=0

# --- Performance (bots stay at spawn; load nothing extra) ---
view-distance=2
simulation-distance=2
entity-broadcast-range-percentage=100
max-players=20
player-idle-timeout=0
sync-chunk-writes=true

# --- Chat / anti-spam friendliness for the bridge bots ---
max-chat-message-length=2048

# --- Misc ---
white-list=false
enable-command-block=true
function-permission-level=2
op-permission-level=4
broadcast-console-to-ops=true
broadcast-rcon-to-ops=true
enable-rcon=false
enable-query=false
motd=Minecraft PvP RL training arena (T8)
EOF
echo "[setup] Wrote server.properties (offline, flat, view/sim=2, pvp=true)."

# --- 4. bukkit.yml (join-storm mitigation) ---------------------------------
# Every value below is the Bukkit default EXCEPT settings.connection-throttle.
#
# Measured on this stack (Paper 1.21.1, four back-to-back joins from one IP):
#
#   throttle   source           result
#   -1         127.0.0.1        4/4 joined
#   -1         LAN address      4/4 joined
#   4000       127.0.0.1        4/4 joined   <- loopback is EXEMPT
#   4000       LAN address      1 joined, 3 kicked "Connection throttled!"
#
# So the honest framing: CraftBukkit exempts 127.0.0.1 from the throttle, which
# means a single-host fleet (all bridges on the Paper box) was never actually
# exposed. -1 is defense in depth, and it becomes load-bearing the moment the
# bridges run on a different host than the JVM — then 2N bots arrive from one
# non-loopback address and the 4000 ms default kicks all but the first.
# Disabling it is safe here: offline-mode training server, not public.
#
# The full document is written rather than a fragment so Paper has no missing
# keys to re-expand from defaults. Verified: Paper leaves this file
# byte-identical across a boot (no rewrite, no key normalization).
cat > "${SERVER_DIR}/bukkit.yml" <<'EOF'
# Bukkit configuration — training arena.
# Generated by server/setup/setup.sh. Idempotent: re-running rewrites this file.
# Only settings.connection-throttle departs from the Bukkit defaults. It matters
# for cross-host bridge fleets, not for loopback (which Bukkit exempts anyway).
# The measured evidence is in the comment above this heredoc in setup.sh.
settings:
  allow-end: true
  warn-on-overload: true
  permissions-file: permissions.yml
  update-folder: update
  plugin-profiling: false
  connection-throttle: -1
  query-plugins: true
  deprecated-verbose: default
  shutdown-message: Server closed
  minimum-api: none
  use-map-color-cache: true
spawn-limits:
  monsters: 70
  animals: 10
  water-animals: 15
  water-ambient: 20
  water-underground-creature: 5
  axolotls: 5
  ambient: 15
chunk-gc:
  period-in-ticks: 600
ticks-per:
  animal-spawns: 400
  monster-spawns: 1
  water-spawns: 1
  water-ambient-spawns: 1
  water-underground-creature-spawns: 1
  axolotl-spawns: 1
  ambient-spawns: 1
  autosave: 6000
aliases: now-in-commands.yml
EOF
echo "[setup] Wrote bukkit.yml (connection-throttle=-1; matters for cross-host bridges)."

# --- 5. Install the arena datapack into the world --------------------------
DATAPACK_SRC="${SERVER_DIR}/arena"
WORLD_DATAPACKS="${SERVER_DIR}/world/datapacks"
DATAPACK_DST="${WORLD_DATAPACKS}/arena"
if [[ -d "${DATAPACK_SRC}" ]]; then
    mkdir -p "${WORLD_DATAPACKS}"
    rm -rf "${DATAPACK_DST}"
    mkdir -p "${DATAPACK_DST}"
    cp "${DATAPACK_SRC}/pack.mcmeta" "${DATAPACK_DST}/pack.mcmeta"
    cp -r "${DATAPACK_SRC}/data" "${DATAPACK_DST}/data"
    echo "[setup] Installed arena datapack -> ${DATAPACK_DST}"
else
    echo "[setup] WARNING: server/arena not found; skipped datapack install." >&2
fi

echo ""
echo "[setup] Done. Next: server/setup/start.sh"
