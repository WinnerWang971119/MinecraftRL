"""launcher -- the REAL subprocess ArenaLauncher + a per-arena launch planner (T11).

This is the live, process-spawning implementation of the
:class:`~distributed.actor.ArenaLauncher` Protocol that the
:class:`~distributed.actor.ActorPool` calls when it relaunches a dead arena.
Every supervisor/fault decision in :mod:`distributed.actor` is already tested
offline against a FAKE launcher (T7/T14); THIS module is the thin layer that
those tests cannot exercise, because it spawns real Paper JVMs and Node bridge
processes. It is therefore kept deliberately small, and its only verifiable
surface in-session is the pure :func:`plan` planner plus the ``--dry-run`` CLI.

Topology A (plan, Decisions section): N arenas == N INDEPENDENT (Paper server, bridge)
pairs, never one shared server with an arena id on the wire. Arena ``i`` gets:

  * Minecraft server port ``mc_base_port + i`` (default base 25565).
  * Bridge TCP port ``bridge_base_port + i`` (default base 5555).
  * Its OWN server ROOT directory (own ``world/``, ``server.properties`` with the
    matching ``server-port`` and a distinct ``level-name``, ``logs/``, ...), so the
    N JVMs never fight over one world directory.
  * Distinct bot usernames (default ``learner_<i>`` / ``dummy_<i>``) so two arenas
    never collide on one offline-mode account, and an ``ops.json`` in that root
    opping exactly those two usernames.

The PowerShell orchestrator ``server/setup/start-arenas.ps1`` is the production
path for materializing those N roots (it computes the offline-mode ``ops.json``
UUIDs and reuses ``setup.ps1`` for the jar/eula/datapack); this launcher assumes
each arena root already exists (start-arenas.ps1 created it) and only SPAWNS the
two processes per arena and tracks their handles for ``terminate``.

UNVERIFIED LIVE CONCERNS (honest about what this session cannot test):
  * Process startup ORDERING / READINESS. ``bridge/run.js`` connects to Paper
    BEFORE it opens its TCP port and EXITS 1 if Paper is down (see run.js header),
    so this launcher starts Paper FIRST, waits for the MC port to accept a TCP
    connection (bounded), and only then starts the bridge. The readiness probe is a
    plain TCP-connect check, not a Minecraft handshake, so a server that is bound
    but not yet done with world-gen can still bounce the first bridge connect; the
    bridge's own connect-before-listen + the collector's reset() retry are the
    backstop. None of this is exercised here.
  * That Java 25 / ``node`` resolve on PATH at spawn time.
  * Real world-gen timing (first boot of N worlds is slow; see the RUNBOOK).

ASCII-ONLY: all log lines are ASCII (the Windows console is cp1252 by default and
unguarded unicode crashes a run -- a recorded project gotcha). No unicode glyphs.

Owner: T11 (multi-arena throughput track, issue #4)
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

__all__ = [
    "ArenaSpec",
    "ArenaProcesses",
    "SubprocessArenaLauncher",
    "plan",
    "main",
]


# --- Defaults (mirror bridge/bot.js DEFAULT_BOT_CONFIG + server/setup) --------
# Base ports: arena i derives its ports as base + i. These match the
# single-arena defaults (25565 / 5555) at i == 0, so a one-arena multi launch is
# byte-for-byte the same ports the existing scripts use.
_DEFAULT_MC_BASE_PORT: int = 25565
_DEFAULT_BRIDGE_BASE_PORT: int = 5555

# Username patterns. ``{i}`` is the 0-based arena id. Distinct per arena so two
# arenas never share one offline-mode account (which would let the servers steal
# each other's bot). The single-arena scripts keep using learner_bot/dummy_bot;
# the multi-arena launcher uses indexed names so ops.json per root is unambiguous.
_DEFAULT_LEARNER_PATTERN: str = "learner_{i}"
_DEFAULT_DUMMY_PATTERN: str = "dummy_{i}"

# Per-arena server-root directory pattern, relative to the server roots base dir.
# Each arena gets its own root so the JVMs do not share a world/.
_DEFAULT_ROOT_PATTERN: str = "arena-{i}"

# The pinned Paper jar (server/compat_check.md). The jar lives in each arena root
# (start-arenas.ps1 copies/links it per root); the launcher invokes it by name
# from within that root so world/, logs/, ops.json land there.
_DEFAULT_PAPER_JAR: str = "paper-1.21.1-133.jar"

# Java + Node executables. Injectable so the dry-run can print a deterministic
# plan and a test can point at a fake binary; defaults rely on PATH (Java 25 /
# node are documented prerequisites).
_DEFAULT_JAVA: str = "java"
_DEFAULT_NODE: str = "node"

# Heap flags mirror server/setup/start.ps1 (-Xms2G -Xmx2G): fixed heap avoids GC
# resize pauses on a tiny flat arena.
_DEFAULT_XMS: str = "2G"
_DEFAULT_XMX: str = "2G"

# Readiness wait: after starting Paper, wait up to this long for the MC port to
# accept a TCP connection before starting the bridge. A Paper cold boot is slow
# (30-60s+, world-gen + plugin load), so the cap is generous. UNTESTED live.
_DEFAULT_SERVER_READY_TIMEOUT_SECONDS: float = 120.0
_DEFAULT_SERVER_READY_POLL_SECONDS: float = 1.0

# Grace period given to a process to exit on terminate() before we kill() it.
_DEFAULT_TERMINATE_GRACE_SECONDS: float = 10.0


def _ascii_log(message: str) -> None:
    """Print one ASCII launcher log line to stderr (cp1252-safe; never unicode)."""
    # Encode->decode through ASCII with backslash-escape so a stray non-ASCII
    # character can never raise on the cp1252 Windows console (recorded gotcha).
    safe = message.encode("ascii", "backslashreplace").decode("ascii")
    print(f"[launcher] {safe}", file=sys.stderr, flush=True)


@dataclass(frozen=True)
class ArenaSpec:
    """The fully-resolved launch parameters for ONE arena.

    Pure data: produced by :meth:`SubprocessArenaLauncher.spec_for` and by
    :func:`plan`, with no side effects, so the ``--dry-run`` plan and the live
    launch share exactly one source of truth for ports/usernames/commands.

    Attributes:
        arena_id: 0-based arena index.
        mc_port: This arena's Minecraft server port (``mc_base_port + arena_id``).
        bridge_port: This arena's bridge TCP port (``bridge_base_port + arena_id``).
        learner_username: This arena's learner bot username.
        dummy_username: This arena's dummy bot username.
        server_root: Absolute path to this arena's server root directory.
        server_command: The exact argv used to launch Paper (run from server_root).
        bridge_command: The exact argv used to launch the Node bridge.
    """

    arena_id: int
    mc_port: int
    bridge_port: int
    learner_username: str
    dummy_username: str
    server_root: str
    server_command: List[str]
    bridge_command: List[str]

    def to_dict(self) -> Dict[str, object]:
        """Return a plain-dict view of this spec (the unit the planner returns)."""
        return {
            "arena_id": self.arena_id,
            "mc_port": self.mc_port,
            "bridge_port": self.bridge_port,
            "learner_username": self.learner_username,
            "dummy_username": self.dummy_username,
            "server_root": self.server_root,
            "server_command": list(self.server_command),
            "bridge_command": list(self.bridge_command),
        }


@dataclass
class ArenaProcesses:
    """The live ``Popen`` handles for one arena (server + bridge), for terminate()."""

    server: Optional[subprocess.Popen] = None
    bridge: Optional[subprocess.Popen] = None


class SubprocessArenaLauncher:
    """Real :class:`~distributed.actor.ArenaLauncher`: spawns Paper + bridge per arena.

    Implements the injected Protocol the :class:`~distributed.actor.ActorPool`
    relaunches through. Each :meth:`launch` starts that arena's Paper server, waits
    (bounded) for its MC port to come up, then starts its Node bridge; each
    :meth:`terminate` stops both, best-effort and idempotent. Process handles are
    tracked per arena under a lock because the pool may relaunch arenas from
    different collector threads concurrently.

    Every collaborator that the ``--dry-run`` must print or a test must redirect is
    injectable: base ports, the server-roots directory, the username/root patterns,
    the paper jar name, and the java/node executables. Defaults reproduce the
    single-arena ports/jar so arena 0 of a multi launch matches today's scripts.

    Args:
        repo_root: Repository root. Used to resolve ``bridge/run.js`` for the bridge
            command. Defaults to this file's repo root (two levels up).
        server_roots_dir: Directory under which per-arena roots live. Each arena's
            root is ``server_roots_dir / root_pattern.format(i=arena_id)``. Defaults
            to ``<repo_root>/server/arenas``. start-arenas.ps1 materializes these.
        mc_base_port / bridge_base_port: Port bases; arena ``i`` uses base + i.
        learner_pattern / dummy_pattern: Username patterns; ``{i}`` is the arena id.
        root_pattern: Per-arena root directory name pattern; ``{i}`` is the arena id.
        paper_jar: The Paper jar filename, invoked from inside each arena root.
        java / node: Executables for the JVM and the bridge (resolved on PATH).
        xms / xmx: JVM heap flags (mirror start.ps1's -Xms2G -Xmx2G).
        server_ready_timeout_seconds / server_ready_poll_seconds: Bounded wait for
            the MC port to accept a TCP connection before the bridge is started.
        terminate_grace_seconds: Grace before a SIGTERM/terminate escalates to kill.
        popen: Injectable process spawner (defaults to ``subprocess.Popen``); a test
            passes a fake so no real JVM/Node is started.
        sleep: Injectable sleep (defaults to ``time.sleep``); a test passes a no-op.
        ready_probe: Injectable ``(host, port) -> bool`` readiness check; defaults to
            a TCP-connect probe. A test passes a stub so it never touches the network.
    """

    def __init__(
        self,
        *,
        repo_root: Optional[str] = None,
        server_roots_dir: Optional[str] = None,
        mc_base_port: int = _DEFAULT_MC_BASE_PORT,
        bridge_base_port: int = _DEFAULT_BRIDGE_BASE_PORT,
        learner_pattern: str = _DEFAULT_LEARNER_PATTERN,
        dummy_pattern: str = _DEFAULT_DUMMY_PATTERN,
        root_pattern: str = _DEFAULT_ROOT_PATTERN,
        paper_jar: str = _DEFAULT_PAPER_JAR,
        java: str = _DEFAULT_JAVA,
        node: str = _DEFAULT_NODE,
        xms: str = _DEFAULT_XMS,
        xmx: str = _DEFAULT_XMX,
        server_ready_timeout_seconds: float = _DEFAULT_SERVER_READY_TIMEOUT_SECONDS,
        server_ready_poll_seconds: float = _DEFAULT_SERVER_READY_POLL_SECONDS,
        terminate_grace_seconds: float = _DEFAULT_TERMINATE_GRACE_SECONDS,
        popen=subprocess.Popen,
        sleep=time.sleep,
        ready_probe=None,
    ) -> None:
        resolved_repo = Path(repo_root) if repo_root is not None else _default_repo_root()
        self._repo_root = resolved_repo.resolve()
        self._server_roots_dir = (
            Path(server_roots_dir).resolve()
            if server_roots_dir is not None
            else (self._repo_root / "server" / "arenas")
        )
        self._mc_base_port = int(mc_base_port)
        self._bridge_base_port = int(bridge_base_port)
        self._learner_pattern = learner_pattern
        self._dummy_pattern = dummy_pattern
        self._root_pattern = root_pattern
        self._paper_jar = paper_jar
        self._java = java
        self._node = node
        self._xms = xms
        self._xmx = xmx
        self._server_ready_timeout_seconds = float(server_ready_timeout_seconds)
        self._server_ready_poll_seconds = float(server_ready_poll_seconds)
        self._terminate_grace_seconds = float(terminate_grace_seconds)
        self._popen = popen
        self._sleep = sleep
        self._ready_probe = ready_probe if ready_probe is not None else _tcp_port_open

        # run.js path (the bridge entry). Resolved once; the live spawn uses it.
        self._run_js = self._repo_root / "bridge" / "run.js"

        # Per-arena live handles, guarded because the pool may relaunch from
        # different collector threads. Read-modify-write of the dict happens only
        # under this lock; the spawned Popen objects themselves are thread-safe to
        # signal.
        self._lock = threading.Lock()
        self._procs: Dict[int, ArenaProcesses] = {}

    # -- pure spec derivation (shared by dry-run + live) -------------------

    def _server_root_for(self, arena_id: int) -> Path:
        """Absolute server-root directory for ``arena_id`` (own world/, ops.json)."""
        return self._server_roots_dir / self._root_pattern.format(i=arena_id)

    def spec_for(self, arena_id: int) -> ArenaSpec:
        """Resolve every launch parameter for ``arena_id`` (pure: no side effects).

        Ports are ``base + arena_id``; usernames come from the patterns; the server
        and bridge commands are the EXACT argv the live path would spawn, so the
        dry-run prints precisely what would run.
        """
        if arena_id < 0:
            raise ValueError(f"arena_id must be >= 0, got {arena_id}")

        mc_port = self._mc_base_port + arena_id
        bridge_port = self._bridge_base_port + arena_id
        learner_username = self._learner_pattern.format(i=arena_id)
        dummy_username = self._dummy_pattern.format(i=arena_id)
        server_root = self._server_root_for(arena_id)

        # Paper is invoked by jar NAME from inside the root (cwd=server_root on
        # spawn), mirroring start.ps1, so world/ and logs/ land in the right root.
        server_command = [
            self._java,
            f"-Xms{self._xms}",
            f"-Xmx{self._xmx}",
            "-jar",
            self._paper_jar,
            "--nogui",
        ]

        # The bridge reads per-arena config from these flags (run.js / T10). We pass
        # them EXPLICITLY (not via env) so the plan is self-describing and two
        # bridges in the same shell never inherit a stale MC_PORT/BRIDGE_PORT.
        bridge_command = [
            self._node,
            str(self._run_js),
            "--port",
            str(mc_port),
            "--bridge-port",
            str(bridge_port),
            "--learner-username",
            learner_username,
            "--dummy-username",
            dummy_username,
        ]

        return ArenaSpec(
            arena_id=arena_id,
            mc_port=mc_port,
            bridge_port=bridge_port,
            learner_username=learner_username,
            dummy_username=dummy_username,
            server_root=str(server_root),
            server_command=server_command,
            bridge_command=bridge_command,
        )

    def plan(self, n_arenas: int) -> List[Dict[str, object]]:
        """Return the per-arena launch plan (one dict per arena), spawning NOTHING."""
        if n_arenas < 1:
            raise ValueError(f"n_arenas must be >= 1, got {n_arenas}")
        return [self.spec_for(i).to_dict() for i in range(n_arenas)]

    # -- ArenaLauncher Protocol (the live, untested path) ------------------

    def launch(self, arena_id: int) -> None:
        """Start (or restart) the Paper server then the bridge for ``arena_id``.

        Ordering is Paper-first because ``bridge/run.js`` connects to Paper before
        it opens its TCP port and exits 1 if Paper is down. After spawning Paper we
        wait (bounded) for the MC port to accept a TCP connection, then spawn the
        bridge. Any previous handles for this arena are terminated first so a
        relaunch never leaks a process (idempotent restart).

        UNVERIFIED LIVE: the readiness probe is a TCP-connect check, not a Minecraft
        handshake, so a bound-but-not-ready server can still bounce the first bridge
        connect; run.js's own connect-before-listen + the collector's reset() retry
        are the backstop. Real world-gen timing is not exercised here.
        """
        spec = self.spec_for(arena_id)

        # Idempotent restart: clear any prior handles for this arena first.
        self.terminate(arena_id)

        server_root = Path(spec.server_root)
        if not server_root.is_dir():
            # The root must already exist (start-arenas.ps1 materializes it with a
            # distinct world + ops.json). Fail loudly rather than spawn Paper into a
            # missing/empty root that would silently regenerate the wrong world.
            raise FileNotFoundError(
                f"arena {arena_id}: server root does not exist: {server_root}. "
                f"Run server/setup/start-arenas.ps1 to materialize the arena roots "
                f"first."
            )

        _ascii_log(
            f"arena {arena_id}: starting Paper on mc port {spec.mc_port} "
            f"(root={spec.server_root})"
        )
        server_proc = self._popen(spec.server_command, cwd=str(server_root))

        # Record the server handle immediately so a failure between here and the
        # bridge spawn still lets terminate() clean it up.
        with self._lock:
            self._procs[arena_id] = ArenaProcesses(server=server_proc, bridge=None)

        self._wait_for_server(arena_id, spec, server_proc)

        _ascii_log(
            f"arena {arena_id}: starting bridge on bridge port {spec.bridge_port} "
            f"(learner={spec.learner_username}, dummy={spec.dummy_username})"
        )
        bridge_proc = self._popen(spec.bridge_command, cwd=str(self._repo_root))

        with self._lock:
            handles = self._procs.get(arena_id)
            if handles is None:
                handles = ArenaProcesses(server=server_proc)
                self._procs[arena_id] = handles
            handles.bridge = bridge_proc

    def terminate(self, arena_id: int) -> None:
        """Stop this arena's bridge then server. Best-effort and idempotent.

        Bridge first so it stops talking to a server that is about to disappear.
        Each handle gets a graceful ``terminate()`` and a bounded wait, escalating
        to ``kill()`` if it does not exit. Missing handles are a no-op (idempotent).
        """
        with self._lock:
            handles = self._procs.pop(arena_id, None)
        if handles is None:
            return

        # Bridge first, then server (reverse of launch order).
        self._stop_process(arena_id, "bridge", handles.bridge)
        self._stop_process(arena_id, "server", handles.server)

    # -- internals ---------------------------------------------------------

    def _wait_for_server(
        self, arena_id: int, spec: ArenaSpec, server_proc: subprocess.Popen
    ) -> None:
        """Block until the arena's MC port accepts a TCP connection, bounded.

        Returns on first successful probe. Raises if the Paper process exits before
        the port comes up, or if the timeout elapses, so a wedged boot surfaces
        rather than spawning a bridge that will immediately exit 1.
        """
        deadline = time.monotonic() + self._server_ready_timeout_seconds
        while time.monotonic() < deadline:
            # If Paper died during boot, stop waiting and fail loudly.
            if server_proc.poll() is not None:
                raise RuntimeError(
                    f"arena {arena_id}: Paper exited (code "
                    f"{server_proc.returncode}) before mc port {spec.mc_port} "
                    f"was reachable"
                )
            if self._ready_probe("127.0.0.1", spec.mc_port):
                _ascii_log(f"arena {arena_id}: mc port {spec.mc_port} is up")
                return
            self._sleep(self._server_ready_poll_seconds)

        raise TimeoutError(
            f"arena {arena_id}: mc port {spec.mc_port} did not come up within "
            f"{self._server_ready_timeout_seconds:.0f}s"
        )

    def _stop_process(
        self, arena_id: int, label: str, proc: Optional[subprocess.Popen]
    ) -> None:
        """Terminate one process gracefully, escalating to kill, swallowing errors."""
        if proc is None:
            return
        try:
            if proc.poll() is not None:
                return  # already exited
            _ascii_log(f"arena {arena_id}: terminating {label}")
            proc.terminate()
            try:
                proc.wait(timeout=self._terminate_grace_seconds)
            except subprocess.TimeoutExpired:
                _ascii_log(f"arena {arena_id}: {label} did not exit, killing")
                proc.kill()
        except Exception as exc:  # noqa: BLE001 - teardown is best-effort/idempotent
            _ascii_log(f"arena {arena_id}: error stopping {label} (ignored): {exc}")


def _default_repo_root() -> Path:
    """Repo root for the default launcher: two levels up from this file.

    ``distributed/launcher.py`` -> ``distributed/`` -> repo root.
    """
    return Path(__file__).resolve().parent.parent


def _tcp_port_open(host: str, port: int, *, timeout: float = 1.0) -> bool:
    """True if a TCP connection to ``host:port`` succeeds within ``timeout``.

    A plain connect probe (NOT a Minecraft handshake), used only to decide that
    Paper has bound its port before the bridge is started.
    """
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def plan(
    n_arenas: int,
    *,
    mc_base_port: int = _DEFAULT_MC_BASE_PORT,
    bridge_base_port: int = _DEFAULT_BRIDGE_BASE_PORT,
    learner_pattern: str = _DEFAULT_LEARNER_PATTERN,
    dummy_pattern: str = _DEFAULT_DUMMY_PATTERN,
    root_pattern: str = _DEFAULT_ROOT_PATTERN,
    paper_jar: str = _DEFAULT_PAPER_JAR,
    repo_root: Optional[str] = None,
    server_roots_dir: Optional[str] = None,
    java: str = _DEFAULT_JAVA,
    node: str = _DEFAULT_NODE,
) -> List[Dict[str, object]]:
    """Build the per-arena launch plan WITHOUT constructing any live state.

    Importable so a test (and the ``--dry-run`` CLI) can assert the derived ports /
    usernames / roots / commands for ``n_arenas`` arenas without spawning anything.
    Delegates to :meth:`SubprocessArenaLauncher.spec_for` so the dry-run and the
    live launch can never diverge.

    Args:
        n_arenas: Number of arenas (must be >= 1).
        (all others): Forwarded to :class:`SubprocessArenaLauncher` so the plan
            reflects any non-default ports/patterns/roots.

    Returns:
        A list of per-arena dicts: ``arena_id``, ``mc_port``, ``bridge_port``,
        ``learner_username``, ``dummy_username``, ``server_root``,
        ``server_command``, ``bridge_command``.
    """
    launcher = SubprocessArenaLauncher(
        repo_root=repo_root,
        server_roots_dir=server_roots_dir,
        mc_base_port=mc_base_port,
        bridge_base_port=bridge_base_port,
        learner_pattern=learner_pattern,
        dummy_pattern=dummy_pattern,
        root_pattern=root_pattern,
        paper_jar=paper_jar,
        java=java,
        node=node,
    )
    return launcher.plan(n_arenas)


def _format_plan(arena_plan: Sequence[Dict[str, object]]) -> str:
    """Render the launch plan as an ASCII, human-readable block (no unicode)."""
    lines: List[str] = []
    lines.append(f"launch plan for {len(arena_plan)} arena(s):")
    for entry in arena_plan:
        server_cmd = " ".join(str(part) for part in entry["server_command"])
        bridge_cmd = " ".join(str(part) for part in entry["bridge_command"])
        lines.append("")
        lines.append(f"  arena {entry['arena_id']}:")
        lines.append(f"    mc_port          : {entry['mc_port']}")
        lines.append(f"    bridge_port      : {entry['bridge_port']}")
        lines.append(f"    learner_username : {entry['learner_username']}")
        lines.append(f"    dummy_username   : {entry['dummy_username']}")
        lines.append(f"    server_root      : {entry['server_root']}")
        lines.append(f"    server_command   : {server_cmd}")
        lines.append(f"    bridge_command   : {bridge_cmd}")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry: ``python -m distributed.launcher --arenas N [--dry-run]``.

    ``--dry-run`` prints the per-arena plan and spawns NOTHING (the only path
    exercisable in this session). Without ``--dry-run`` it actually launches every
    arena (the live, untested path) and then waits for Ctrl-C to terminate them all.
    """
    parser = argparse.ArgumentParser(
        prog="python -m distributed.launcher",
        description=(
            "Launch N independent (Paper server, bridge) arena pairs for "
            "multi-arena training (Topology A). Use --dry-run to print the "
            "launch plan without spawning anything."
        ),
    )
    parser.add_argument(
        "--arenas",
        type=int,
        default=1,
        help="Number of arenas to launch (each its own server+bridge). Default 1.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the per-arena launch plan and spawn NOTHING.",
    )
    parser.add_argument(
        "--mc-base-port",
        type=int,
        default=_DEFAULT_MC_BASE_PORT,
        help=f"Base Minecraft port; arena i uses base+i. Default {_DEFAULT_MC_BASE_PORT}.",
    )
    parser.add_argument(
        "--bridge-base-port",
        type=int,
        default=_DEFAULT_BRIDGE_BASE_PORT,
        help=f"Base bridge port; arena i uses base+i. Default {_DEFAULT_BRIDGE_BASE_PORT}.",
    )
    parser.add_argument(
        "--server-roots-dir",
        type=str,
        default=None,
        help="Directory holding per-arena server roots. Default <repo>/server/arenas.",
    )
    args = parser.parse_args(argv)

    if args.arenas < 1:
        parser.error(f"--arenas must be >= 1, got {args.arenas}")

    launcher = SubprocessArenaLauncher(
        server_roots_dir=args.server_roots_dir,
        mc_base_port=args.mc_base_port,
        bridge_base_port=args.bridge_base_port,
    )
    arena_plan = launcher.plan(args.arenas)

    if args.dry_run:
        # Dry-run: print and exit. Nothing is spawned, no roots are touched.
        print(_format_plan(arena_plan))
        _ascii_log("dry-run: no processes were started.")
        return 0

    # --- LIVE path (UNVERIFIED in-session) --------------------------------
    # Spawns real JVMs + Node bridges. Each arena root must already exist
    # (server/setup/start-arenas.ps1 materializes them). We launch every arena,
    # then block until interrupted, then terminate everything best-effort.
    _ascii_log(_format_plan(arena_plan))
    launched: List[int] = []
    try:
        for entry in arena_plan:
            arena_id = int(entry["arena_id"])
            launcher.launch(arena_id)
            launched.append(arena_id)
        _ascii_log(
            f"launched {len(launched)} arena(s). Press Ctrl-C to stop them all."
        )
        # Idle until interrupted; the collectors connect to the bridges themselves.
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        _ascii_log("interrupted; terminating all arenas.")
    finally:
        for arena_id in launched:
            launcher.terminate(arena_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
