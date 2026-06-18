<#
.SYNOPSIS
  Materialize and launch N independent (Paper server, bridge) arena pairs (task T11).

.DESCRIPTION
  The PowerShell orchestrator for multi-arena training (Topology A, issue #4):
  N SEPARATE Paper servers + N Node bridges, one TCP connection each, the frozen
  wire unchanged. Arena i (i = 0..N-1) gets:

    * Minecraft server port 25565+i.
    * Bridge TCP port 5555+i.
    * Its OWN server ROOT directory (own world/, server.properties with the
      matching server-port and a distinct level-name, logs/, ops.json) so the N
      JVMs never fight over one world directory.
    * Distinct bot usernames (learner_<i> / dummy_<i>) and an ops.json in that
      root opping exactly those two usernames at level 4 with their offline UUIDs.

  For each arena this script:
    1. Reuses setup.ps1 to stamp the per-arena root (jar present, eula.txt,
       server.properties on port 25565+i with the distinct world, the arena
       datapack, and ops.json opping that arena's two usernames).
    2. Starts that arena's Paper server (mirrors start.ps1's java invocation,
       run from the arena root so world/ and logs/ land there).
    3. Waits a short stagger, then starts that arena's Node bridge
       (node bridge/run.js with --port/--bridge-port/--learner-username/
       --dummy-username for that arena), so Paper is up before the bridge tries
       to connect (run.js exits 1 if Paper is down).

  Use -DryRun to print the full plan (roots, ports, usernames, commands) and
  start NOTHING. That is the only path exercised offline; the real launch is a
  human follow-up documented in the RUNBOOK (T16).

.PARAMETER Arenas
  Number of arenas to materialize + launch. Default 1.

.PARAMETER ArenasRoot
  Directory under which per-arena roots live. Each arena's root is
  <ArenasRoot>/arena-<i>. Defaults to server/arenas (a sibling of this script's
  server root).

.PARAMETER DryRun
  Print the full launch plan and start nothing. Materializes no roots, spawns no
  processes. This is the only way to exercise the orchestrator without a live
  Paper + Node stack.

.PARAMETER SkipSetup
  Skip the per-arena setup.ps1 step (assume the roots already exist). Useful for
  a fast relaunch after the first (slow) boot has already materialized worlds.

.PARAMETER Xms
  JVM initial heap per server (default 2G; mirrors start.ps1).

.PARAMETER Xmx
  JVM max heap per server (default 2G; mirrors start.ps1).

.PARAMETER ServerReadyStaggerSeconds
  Seconds to wait after starting a Paper server before starting its bridge, and
  between arenas. Default 20. A cold Paper boot is slow; this is a coarse stagger,
  not a real readiness gate (the bridge's connect-before-listen + the collector's
  reset() retry are the real backstop).

.EXAMPLE
  pwsh -NoProfile -File server/setup/start-arenas.ps1 -Arenas 3 -DryRun

.EXAMPLE
  pwsh -NoProfile -File server/setup/start-arenas.ps1 -Arenas 3

.NOTES
  Prerequisites: Java 21+ (this machine has Java 25) and node on PATH; the Paper
  jar present (setup.ps1 downloads it for the canonical root, then this script
  copies it into each arena root).

  CAVEATS (read before a long unattended run):
    * First boot is SLOW. The FIRST launch of N fresh worlds generates N world
      dirs and loads plugins N times; expect minutes, not seconds, before all
      bridges connect. Subsequent boots reuse the worlds and are far faster.
    * Windows Update auto-reboots this box overnight (recorded gotcha). PAUSE
      Windows Update before any multi-hour run or it dies mid-run with no
      checkpoint. See the RUNBOOK.
    * This script keeps running and holds the spawned processes; Ctrl-C stops it
      and tears every arena down. Each arena server is also stoppable from its own
      console with the `stop` command.
#>

[CmdletBinding()]
param(
    [int]$Arenas = 1,
    [string]$ArenasRoot,
    [switch]$DryRun,
    [switch]$SkipSetup,
    [string]$Xms = '2G',
    [string]$Xmx = '2G',
    [int]$ServerReadyStaggerSeconds = 20
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if ($Arenas -lt 1) {
    throw "Arenas must be >= 1, got $Arenas."
}

# --- Pinned constants (mirror setup.ps1 / start.ps1) -----------------------
$PaperVersion = '1.21.1'
$PaperBuild   = '133'
$JarName      = "paper-$PaperVersion-$PaperBuild.jar"

# Port + username bases. Arena i derives base+i. These match the single-arena
# defaults at i == 0 so arena 0 of a multi launch uses the same ports as today.
$McBasePort     = 25565
$BridgeBasePort = 5555

# --- Resolve paths ---------------------------------------------------------
# server/setup/start-arenas.ps1 -> server/ is one level up from $PSScriptRoot.
$ServerDir = (Resolve-Path -LiteralPath (Split-Path -Parent $PSScriptRoot)).Path
$RepoRoot  = (Resolve-Path -LiteralPath (Split-Path -Parent $ServerDir)).Path
$SetupScript = Join-Path $PSScriptRoot 'setup.ps1'
$RunJs       = Join-Path (Join-Path $RepoRoot 'bridge') 'run.js'
$CanonicalJar = Join-Path $ServerDir $JarName

if ([string]::IsNullOrWhiteSpace($ArenasRoot)) {
    $ArenasRoot = Join-Path $ServerDir 'arenas'
}

# --- Build the per-arena plan (pure: no side effects) ----------------------
# One object per arena holding everything both the plan print and the live
# launch need, so the dry-run prints exactly what would run.
$plan = for ($i = 0; $i -lt $Arenas; $i++) {
    $root = Join-Path $ArenasRoot ("arena-$i")
    [pscustomobject]@{
        ArenaId         = $i
        McPort          = $McBasePort + $i
        BridgePort      = $BridgeBasePort + $i
        WorldName       = "world-$i"
        LearnerUsername = "learner_$i"
        DummyUsername   = "dummy_$i"
        Root            = $root
        ServerCommand   = "java -Xms$Xms -Xmx$Xmx -jar $JarName --nogui   (cwd=$root)"
        BridgeCommand   = "node `"$RunJs`" --port $($McBasePort + $i) --bridge-port $($BridgeBasePort + $i) --learner-username learner_$i --dummy-username dummy_$i"
    }
}

# --- Print the plan (always; the dry-run does only this) -------------------
Write-Host "[start-arenas] Repo root      : $RepoRoot"
Write-Host "[start-arenas] Arenas root    : $ArenasRoot"
Write-Host "[start-arenas] Arenas         : $Arenas"
Write-Host "[start-arenas] Paper jar      : $JarName"
Write-Host "[start-arenas] Heap per arena : -Xms$Xms -Xmx$Xmx"
Write-Host ""
foreach ($a in $plan) {
    Write-Host "[start-arenas] arena $($a.ArenaId):"
    Write-Host "[start-arenas]   mc_port          : $($a.McPort)"
    Write-Host "[start-arenas]   bridge_port      : $($a.BridgePort)"
    Write-Host "[start-arenas]   world            : $($a.WorldName)"
    Write-Host "[start-arenas]   learner_username : $($a.LearnerUsername)"
    Write-Host "[start-arenas]   dummy_username   : $($a.DummyUsername)"
    Write-Host "[start-arenas]   server_root      : $($a.Root)"
    Write-Host "[start-arenas]   server_command   : $($a.ServerCommand)"
    Write-Host "[start-arenas]   bridge_command   : $($a.BridgeCommand)"
    Write-Host ""
}

if ($DryRun) {
    Write-Host "[start-arenas] Dry run: no roots materialized, no processes started."
    Write-Host "[start-arenas] CAVEATS: first boot of N worlds is slow; pause Windows"
    Write-Host "[start-arenas]   Update before a long run (this box auto-reboots overnight)."
    return
}

# ===========================================================================
# LIVE path below (UNVERIFIED in-session: spawns real JVMs + Node bridges).
# ===========================================================================

# --- Preflight -------------------------------------------------------------
$java = Get-Command java -ErrorAction SilentlyContinue
if ($null -eq $java) {
    throw "java not found on PATH. Install Java 21+ (this machine has Java 25). See server/compat_check.md."
}
$node = Get-Command node -ErrorAction SilentlyContinue
if ($null -eq $node) {
    throw "node not found on PATH. The bridge runs on Node; install it before launching arenas."
}
if (-not (Test-Path -LiteralPath $RunJs)) {
    throw "bridge/run.js not found at $RunJs."
}
if (-not (Test-Path -LiteralPath $CanonicalJar)) {
    throw "$JarName not found in $ServerDir. Run server/setup/setup.ps1 first (it downloads the jar)."
}

Write-Host "[start-arenas] CAVEAT: first boot of N fresh worlds is SLOW (minutes). Pause"
Write-Host "[start-arenas]   Windows Update before a long run (this box auto-reboots overnight)."
Write-Host ""

# Track spawned processes so Ctrl-C / a fault tears every arena down.
$procs = New-Object System.Collections.Generic.List[System.Diagnostics.Process]

try {
    foreach ($a in $plan) {
        # --- 1. Materialize this arena's root (jar, eula, props, datapack, ops). ---
        New-Item -ItemType Directory -Force -Path $a.Root | Out-Null

        # Each arena root needs its own copy of the Paper jar (the launcher invokes
        # it by name from inside the root). Copy from the canonical root once.
        $arenaJar = Join-Path $a.Root $JarName
        if (-not (Test-Path -LiteralPath $arenaJar)) {
            Copy-Item -LiteralPath $CanonicalJar -Destination $arenaJar -Force
        }

        if (-not $SkipSetup) {
            Write-Host "[start-arenas] arena $($a.ArenaId): running setup for $($a.Root)"
            & $SetupScript `
                -ServerDir $a.Root `
                -McPort $a.McPort `
                -WorldName $a.WorldName `
                -LearnerUsername $a.LearnerUsername `
                -DummyUsername $a.DummyUsername `
                -ArenaId $a.ArenaId
        }

        # --- 2. Start this arena's Paper server (cwd = arena root). ---
        Write-Host "[start-arenas] arena $($a.ArenaId): starting Paper on mc port $($a.McPort)"
        $serverArgs = @("-Xms$Xms", "-Xmx$Xmx", '-jar', $JarName, '--nogui')
        $serverProc = Start-Process -FilePath 'java' -ArgumentList $serverArgs `
            -WorkingDirectory $a.Root -PassThru
        $procs.Add($serverProc)

        # --- Stagger: give Paper a head start before the bridge connects. ---
        # Coarse, not a real readiness gate (see header NOTES). The bridge's own
        # connect-before-listen + the collector's reset() retry are the backstop.
        if ($ServerReadyStaggerSeconds -gt 0) {
            Write-Host "[start-arenas] arena $($a.ArenaId): waiting ${ServerReadyStaggerSeconds}s for Paper before starting bridge"
            Start-Sleep -Seconds $ServerReadyStaggerSeconds
        }

        # --- 3. Start this arena's Node bridge (cwd = repo root). ---
        Write-Host "[start-arenas] arena $($a.ArenaId): starting bridge on bridge port $($a.BridgePort)"
        $bridgeArgs = @(
            $RunJs,
            '--port', "$($a.McPort)",
            '--bridge-port', "$($a.BridgePort)",
            '--learner-username', $a.LearnerUsername,
            '--dummy-username', $a.DummyUsername
        )
        $bridgeProc = Start-Process -FilePath 'node' -ArgumentList $bridgeArgs `
            -WorkingDirectory $RepoRoot -PassThru
        $procs.Add($bridgeProc)
    }

    Write-Host ""
    Write-Host "[start-arenas] Launched $Arenas arena(s) ($($procs.Count) processes). Ctrl-C to stop all."

    # Idle until interrupted; the training collectors connect to the bridges
    # themselves. Poll so a process that dies is noticed and surfaced.
    while ($true) {
        Start-Sleep -Seconds 5
        $dead = $procs | Where-Object { $_.HasExited }
        if ($null -ne $dead -and @($dead).Count -gt 0) {
            foreach ($d in @($dead)) {
                Write-Warning "[start-arenas] process PID $($d.Id) exited (code $($d.ExitCode))."
            }
            break
        }
    }
}
finally {
    Write-Host "[start-arenas] Tearing down all arena processes."
    foreach ($p in $procs) {
        try {
            if (-not $p.HasExited) {
                $p.CloseMainWindow() | Out-Null
                Start-Sleep -Milliseconds 200
                if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue }
            }
        }
        catch {
            Write-Warning "[start-arenas] error stopping PID $($p.Id) (ignored): $($_.Exception.Message)"
        }
    }
}
