<#
.SYNOPSIS
  Launch the Paper training server (task T8).

.DESCRIPTION
  Starts paper-1.21.1-133.jar with fixed heap flags suited to a single training
  arena. Run setup.ps1 first (it downloads the jar, writes eula.txt and
  server.properties, and installs the arena datapack).

  Heap: -Xms2G -Xmx2G. A flat arena with view/sim distance 2 and a handful of
  entities is tiny; 2 GB is comfortable headroom and a fixed Xms==Xmx avoids GC
  resize pauses. Bump for many parallel arenas on one host (see project spec §9).

.PARAMETER ServerDir
  Server root. Defaults to the parent of this script's directory (server/).

.PARAMETER Xms
  Initial heap (default 2G).

.PARAMETER Xmx
  Max heap (default 2G).

.EXAMPLE
  pwsh -NoProfile -File server/setup/start.ps1

.NOTES
  Requires Java 21+ on PATH (this machine has Java 25). Runs with --nogui so it
  stays a console process (the bridge talks to it over the Minecraft protocol,
  not the GUI). Ctrl+C / the `stop` console command shuts it down cleanly.
#>

[CmdletBinding()]
param(
    [string]$ServerDir,
    [string]$Xms = '2G',
    [string]$Xmx = '2G'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$PaperVersion = '1.21.1'
$PaperBuild   = '133'
$JarName      = "paper-$PaperVersion-$PaperBuild.jar"

if ([string]::IsNullOrWhiteSpace($ServerDir)) {
    $ServerDir = Split-Path -Parent $PSScriptRoot
}
$ServerDir = (Resolve-Path -LiteralPath $ServerDir).Path
$JarPath = Join-Path $ServerDir $JarName

# --- Preflight ------------------------------------------------------------
$java = Get-Command java -ErrorAction SilentlyContinue
if ($null -eq $java) {
    throw "java not found on PATH. Install Java 21+ (this machine has Java 25). See server/compat_check.md."
}
if (-not (Test-Path -LiteralPath $JarPath)) {
    throw "$JarName not found in $ServerDir. Run server/setup/setup.ps1 first."
}
$eulaPath = Join-Path $ServerDir 'eula.txt'
if (-not (Test-Path -LiteralPath $eulaPath)) {
    throw "eula.txt missing. Run server/setup/setup.ps1 first."
}

Write-Host "[start] Java : $((& java -version 2>&1 | Select-Object -First 1))"
Write-Host "[start] Jar  : $JarPath"
Write-Host "[start] Heap : -Xms$Xms -Xmx$Xmx"
Write-Host "[start] Launching Paper (--nogui). Use 'stop' in the console to shut down."

# Run from the server directory so world/, logs/, ops.json etc. land there.
Push-Location -LiteralPath $ServerDir
try {
    # Aikar-style flags are deliberately omitted for a single tiny arena; plain
    # fixed heap is fine and keeps the launch reproducible. Add G1 tuning if you
    # later host many arenas per host.
    & java "-Xms$Xms" "-Xmx$Xmx" -jar $JarName --nogui
}
finally {
    Pop-Location
}
