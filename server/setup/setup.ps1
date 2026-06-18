<#
.SYNOPSIS
  One-shot, idempotent setup for the Paper training server (task T8).

.DESCRIPTION
  Prepares server/ to run a Paper 1.21.1 build 133 instance tuned for the PvP
  RL training arena:

    1. Downloads paper-1.21.1-133.jar from the PaperMC v2 API (skipped if the
       jar already exists, unless -Force is passed).
    2. Writes eula.txt with eula=true (accepting Mojang's EULA — this is a
       deliberate, recorded action; running the server implies acceptance).
    3. Writes server.properties tuned for headless, deterministic, low-overhead
       training: offline-mode, flat world, no mobs/animals, view/sim distance 2,
       fixed seed, pvp on, gamemode survival, normal difficulty, plus chat
       anti-spam friendliness so the bridge's rapid /tp + /effect commands are
       not throttled or kicked.
    4. Installs the arena datapack into the world's datapacks/ folder so the
       arena/* functions are available on first boot.

  This script does NOT launch the server. Use start.ps1 for that. It is safe to
  re-run: existing files are only overwritten when they are config files we own
  (eula.txt, server.properties) or when -Force is given (the jar).

.PARAMETER Force
  Re-download the Paper jar even if it already exists.

.PARAMETER ServerDir
  Server root. Defaults to the parent of this script's directory (server/).

.PARAMETER McPort
  Minecraft server-port written into server.properties. Defaults to 25565 (the
  single-arena default). Multi-arena callers pass 25565+i (see start-arenas.ps1).

.PARAMETER WorldName
  level-name / world directory written into server.properties. Defaults to
  'world' (the single-arena default). Multi-arena callers pass a distinct name
  per arena so the N JVMs never share one world dir.

.PARAMETER LearnerUsername
  Learner bot username opped in this root's ops.json. Defaults to 'learner_bot'.

.PARAMETER DummyUsername
  Dummy bot username opped in this root's ops.json. Defaults to 'dummy_bot'.

.PARAMETER ArenaId
  0-based arena index, recorded only in the operator-facing summary. Optional;
  defaults to unset (single-arena). Does not affect any written file by itself.

.EXAMPLE
  pwsh -NoProfile -File server/setup/setup.ps1

.EXAMPLE
  # Materialize a second arena root (port 25566, distinct world + bot names):
  pwsh -NoProfile -File server/setup/setup.ps1 -ServerDir server/arenas/arena-1 `
      -McPort 25566 -WorldName world-1 `
      -LearnerUsername learner_1 -DummyUsername dummy_1 -ArenaId 1

.NOTES
  Prerequisites: Java 21+ (this machine has Java 25 — see server/compat_check.md).
  Pinned by Tv: Paper 1.21.1 build 133, channel STABLE.

  Backward compatibility: with NO new args this script behaves exactly as
  before (single arena, port 25565, world 'world', usernames learner_bot /
  dummy_bot, plus the jar download, eula, and datapack install). The new
  parameters and the ops.json write are ADDITIVE; start-arenas.ps1 reuses this
  script to stamp each per-arena root.
#>

[CmdletBinding()]
param(
    [switch]$Force,
    [string]$ServerDir,
    [int]$McPort = 25565,
    [string]$WorldName = 'world',
    [string]$LearnerUsername = 'learner_bot',
    [string]$DummyUsername = 'dummy_bot',
    [Nullable[int]]$ArenaId = $null
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# --- Pinned constants (from server/compat_check.md / Tv) -------------------
$PaperVersion = '1.21.1'
$PaperBuild   = '133'
$JarName      = "paper-$PaperVersion-$PaperBuild.jar"
$DownloadUrl  = "https://api.papermc.io/v2/projects/paper/versions/$PaperVersion/builds/$PaperBuild/downloads/$JarName"

# Bot accounts that must be opped (mirror bridge/bot.js DEFAULT_BOT_CONFIG and
# server/ops.json). Recorded here only for the operator-facing summary.
$LevelSeed = '8675309'   # fixed seed -> reproducible flat world (logged per spec).

# --- Offline-mode UUID helper ----------------------------------------------
# An offline (cracked) server derives a player's UUID deterministically from the
# username: it is a version-3 (name-based, MD5) UUID over the bytes of
# "OfflinePlayer:<username>", with the version nibble forced to 3 and the IETF
# variant bits set. ops.json must list exactly this UUID or the op is silently
# ignored (the name alone is not enough). Verified to reproduce the existing
# server/ops.json UUIDs for learner_bot / dummy_bot.
function Get-OfflineUuid {
    [CmdletBinding()]
    param([Parameter(Mandatory)][string]$Username)

    $bytes = [System.Text.Encoding]::UTF8.GetBytes("OfflinePlayer:$Username")
    $md5 = [System.Security.Cryptography.MD5]::Create()
    try {
        $hash = $md5.ComputeHash($bytes)
    }
    finally {
        $md5.Dispose()
    }
    # Force version 3 (name-based MD5) and the IETF variant (10xx) bits.
    $hash[6] = [byte](($hash[6] -band 0x0F) -bor 0x30)
    $hash[8] = [byte](($hash[8] -band 0x3F) -bor 0x80)
    $hex = ([System.BitConverter]::ToString($hash)).Replace('-', '').ToLowerInvariant()
    return ('{0}-{1}-{2}-{3}-{4}' -f `
        $hex.Substring(0, 8), $hex.Substring(8, 4), $hex.Substring(12, 4), `
        $hex.Substring(16, 4), $hex.Substring(20, 12))
}

# --- ops.json writer (one root) --------------------------------------------
# Writes an ops.json into $RootDir opping exactly the two given usernames at
# level 4 with their offline UUIDs. Mirrors the shape of the existing
# server/ops.json. Called for every root (single-arena writes learner_bot /
# dummy_bot, matching today's committed file byte-for-byte in shape).
function Write-OpsJson {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$RootDir,
        [Parameter(Mandatory)][string]$Learner,
        [Parameter(Mandatory)][string]$Dummy
    )
    $ops = @(
        [ordered]@{
            uuid                 = (Get-OfflineUuid -Username $Learner)
            name                 = $Learner
            level                = 4
            bypassesPlayerLimit  = $false
        },
        [ordered]@{
            uuid                 = (Get-OfflineUuid -Username $Dummy)
            name                 = $Dummy
            level                = 4
            bypassesPlayerLimit  = $false
        }
    )
    $opsPath = Join-Path $RootDir 'ops.json'
    Set-Content -LiteralPath $opsPath -Value ($ops | ConvertTo-Json -Depth 4) -Encoding ASCII
    Write-Host "[setup] Wrote ops.json (opped $Learner, $Dummy @ level 4, offline UUIDs)."
}

# --- Resolve paths ---------------------------------------------------------
if ([string]::IsNullOrWhiteSpace($ServerDir)) {
    # server/setup/setup.ps1 -> server/ is one level up.
    $ServerDir = Split-Path -Parent $PSScriptRoot
}
$ServerDir = (Resolve-Path -LiteralPath $ServerDir).Path
$JarPath = Join-Path $ServerDir $JarName

Write-Host "[setup] Server directory : $ServerDir"
Write-Host "[setup] Paper jar        : $JarName"
Write-Host "[setup] Download URL     : $DownloadUrl"

# --- 1. Download the Paper jar (idempotent) --------------------------------
if ((Test-Path -LiteralPath $JarPath) -and (-not $Force)) {
    Write-Host "[setup] Jar already present, skipping download (use -Force to refresh)."
}
else {
    Write-Host "[setup] Downloading Paper jar ..."
    # TLS 1.2 for older Windows PowerShell hosts; pwsh 7 already defaults higher.
    try { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 } catch {}
    $tmp = "$JarPath.partial"
    Invoke-WebRequest -Uri $DownloadUrl -OutFile $tmp -UseBasicParsing
    Move-Item -LiteralPath $tmp -Destination $JarPath -Force
    Write-Host "[setup] Saved $JarName ($([math]::Round((Get-Item -LiteralPath $JarPath).Length / 1MB, 1)) MB)."
}

# --- 2. eula.txt -----------------------------------------------------------
$eulaPath = Join-Path $ServerDir 'eula.txt'
$eulaBody = @"
# By setting eula=true you agree to the Minecraft EULA:
# https://aka.ms/MinecraftEULA
# Written by server/setup/setup.ps1 (task T8).
eula=true
"@
Set-Content -LiteralPath $eulaPath -Value $eulaBody -Encoding ASCII -NoNewline
Write-Host "[setup] Wrote eula.txt (eula=true)."

# --- 3. server.properties (training-tuned) ---------------------------------
# Tuned for: offline join, flat deterministic world, no ambient entities, tiny
# view/sim distance (the bots and the dummy are always near spawn, so loading
# more chunks is pure CPU/RAM waste — the throughput limit per the project spec),
# pvp enabled, survival, normal difficulty. Anti-spam: max-chat-message-length
# raised and spawn-protection disabled so opped bots can act at spawn.
$serverProps = @"
# Minecraft server properties — training arena (task T8).
# Generated by server/setup/setup.ps1. Idempotent: re-running rewrites this file.
# Rationale for the non-default values lives in server/README.md.

# --- Identity / networking ---
online-mode=false
server-port=$McPort
server-ip=
network-compression-threshold=256
prevent-proxy-connections=false
enforce-secure-profile=false

# --- World generation (flat, deterministic) ---
level-name=$WorldName
level-type=minecraft:flat
level-seed=$LevelSeed
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
# The bridge fires rapid /tp + /effect clear + regear bursts on every reset.
# Keep message length generous; never rate-limit or kick for "spam".
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
"@
$propsPath = Join-Path $ServerDir 'server.properties'
Set-Content -LiteralPath $propsPath -Value $serverProps -Encoding ASCII
Write-Host "[setup] Wrote server.properties (offline, flat, view/sim=2, pvp=true)."

# --- 4. Install the arena datapack into the world --------------------------
# Paper reads datapacks from <level-name>/datapacks/. The pack source of truth
# lives in server/arena/; we copy it in so it is enabled on first world gen.
# Source: this root's own arena/ if present, else fall back to the canonical
# server/ root (so a per-arena root does not need its own copy of the pack).
$datapackSrc = Join-Path $ServerDir 'arena'
if (-not (Test-Path -LiteralPath $datapackSrc)) {
    $canonicalArena = Join-Path (Split-Path -Parent $PSScriptRoot) 'arena'
    if (Test-Path -LiteralPath $canonicalArena) {
        $datapackSrc = $canonicalArena
    }
}
$worldDatapacks = Join-Path (Join-Path $ServerDir $WorldName) 'datapacks'
$datapackDst = Join-Path $worldDatapacks 'arena'
if (Test-Path -LiteralPath $datapackSrc) {
    New-Item -ItemType Directory -Force -Path $worldDatapacks | Out-Null
    # Copy only the datapack payload (pack.mcmeta + data/), not the README.
    Copy-Item -LiteralPath (Join-Path $datapackSrc 'pack.mcmeta') -Destination (Join-Path $worldDatapacks 'arena') -Force -ErrorAction SilentlyContinue
    if (Test-Path -LiteralPath $datapackDst) { Remove-Item -LiteralPath $datapackDst -Recurse -Force }
    New-Item -ItemType Directory -Force -Path $datapackDst | Out-Null
    Copy-Item -LiteralPath (Join-Path $datapackSrc 'pack.mcmeta') -Destination $datapackDst -Force
    Copy-Item -LiteralPath (Join-Path $datapackSrc 'data') -Destination $datapackDst -Recurse -Force
    Write-Host "[setup] Installed arena datapack -> $datapackDst"
    Write-Host "[setup]   (run /reload or /datapack enable after first boot if added live)"
}
else {
    Write-Warning "[setup] server/arena not found; skipped datapack install."
}

# --- 5. ops.json (this root opps exactly its two bot usernames) ------------
# Idempotent: rewrites this root's ops.json with the two usernames' offline
# UUIDs. With the defaults this reproduces the committed server/ops.json
# (learner_bot / dummy_bot); multi-arena roots get learner_<i> / dummy_<i>.
Write-OpsJson -RootDir $ServerDir -Learner $LearnerUsername -Dummy $DummyUsername

# --- Operator-facing summary -----------------------------------------------
if ($null -ne $ArenaId) {
    Write-Host "[setup] Arena id        : $ArenaId"
}
Write-Host "[setup] MC port         : $McPort"
Write-Host "[setup] World (level)    : $WorldName"
Write-Host "[setup] Bots opped       : $LearnerUsername, $DummyUsername"

Write-Host ""
Write-Host "[setup] Done. Next steps:"
Write-Host "[setup]   1) pwsh -NoProfile -File server/setup/start.ps1   # launch the server"
Write-Host "[setup]   2) In-game/console: /datapack list   (expect 'file/arena' enabled)"
Write-Host "[setup]   3) Bots opped in $ServerDir/ops.json:"
Write-Host "[setup]        $LearnerUsername, $DummyUsername @ level 4 (offline UUIDs)."
