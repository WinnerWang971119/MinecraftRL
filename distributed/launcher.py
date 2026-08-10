"""launcher -- pad geometry, the fleet launch plan, and the live bridge launcher (T10).

ONE JVM, N PADS. This module replaces the N-JVM "Topology A" launcher that shipped
with PR #21 (N server roots, N Paper processes, one bridge each). The world now holds
N enclosed bedrock pads inside a SINGLE flat world served by a SINGLE Paper JVM on
port 25565; pad ``i`` gets:

  * its ANCHOR ``pad_anchor(i)`` -- ``PAD_SPACING`` blocks apart on a
    ``PAD_GRID_COLS``-wide row-major grid,
  * its own Node bridge process on TCP port ``bridge_base_port + i``,
  * bot usernames ``learner_<i>`` / ``dummy_<i>``, except that ``i == 0`` keeps
    ``learner_bot`` / ``dummy_bot`` so the manual single-arena path is byte-identical.

:func:`pad_anchor` is the SOLE implementation of the anchor formula in this repo.
``bridge/run.js`` only *parses* the ``--pad-origin "<x>,<z>"`` value it is handed and
deliberately refuses to derive one from ``--pad-index``; ``server/setup/start-pads.sh``
asks this module for the plan (``--emit-plan``) rather than recomputing coordinates in
bash. There is no mirror, and adding one would be a defect.

WHAT LIVES WHERE (the T10 split):

  * ``server/setup/start-pads.sh`` is the OPERATOR entry point. It runs the preflight
    gates, writes ``ops.json``, starts Paper via ``start.sh``, starts the N bridges
    with staggered joins, runs the prime barrier, and supervises the fleet.
  * THIS module owns every derived VALUE (anchors, ports, usernames, offline UUIDs,
    the ``max-players`` requirement, the exact bridge argv) plus two runtime pieces
    bash cannot do: :func:`prime_pads` (the reset-before-step barrier) and
    :class:`SubprocessArenaLauncher` (the mid-run, single-pad bridge relaunch that
    :class:`~distributed.actor.ActorPool` calls through the ``ArenaLauncher``
    Protocol).

THE RESET-BEFORE-STEP BARRIER (:func:`prime_pads`). ``arena:setup`` puts ONE world
spawn at ``0 64 0``, which is pad 0's anchor, so at fleet boot all 2N bots join
STACKED inside pad 0 and only leave when their own first ``arena:reset_pad`` runs.
PvP is necessarily on (the damage channel needs it) and bedrock walls do not help
while everyone is inside the same walls, so a pad that starts stepping while other
pads are still stacked can land real hits on idle foreign bots and register
``damage_taken`` on THEIR bridges. :func:`prime_pads` therefore resets every pad
once before any driver is allowed to step. The ORDER is descending for a separate
reason -- nothing can attack during the prime itself; see :func:`prime_pads` for why
pad 0, the stack site, must be the one reset last. It is run
ONCE at fleet boot and NEVER from :meth:`SubprocessArenaLauncher.launch`: a primer
connection mid-run would steal the bridge's single TCP client slot from the live
collector. A mid-run bridge restart does not re-stack anything -- ``arena:reset_pad``
pins a per-bot ``/spawnpoint``, which persists in player NBT across reconnects.

WHAT A RESET ACK DOES NOT PROVE: geometry. A gated ack says the bots read back at the
expected position/health; it does not say the pad's walls exist (issue #27). Fleet
readiness here means "2N bots placed at their anchors", never "arena verified".

UNVERIFIED LIVE CONCERNS (honest about what this session cannot test):
  * Real Paper boot / world-gen timing for a fleet-sized world.
  * That ``node`` resolves on PATH at spawn time.
  * Whether a single Paper main thread holds N pads at >=19 TPS (that is T13's
    measured ladder, not an assumption made here).

ASCII-ONLY: all log lines are ASCII. No unicode glyphs.

Owner: T10 (damage-channel repair + one-JVM pad topology)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "PAD_SPACING",
    "PAD_GRID_COLS",
    "PadAnchor",
    "pad_anchor",
    "pad_usernames",
    "offline_uuid",
    "ops_entries",
    "ops_json",
    "write_ops_json",
    "missing_ops",
    "required_max_players",
    "PadSpec",
    "PadProcess",
    "SubprocessArenaLauncher",
    "plan",
    "PadPrimeError",
    "prime_pads",
    "main",
]


# --- Pad geometry (the SOLE implementation; see the module docstring) --------
#: Blocks between adjacent pad anchors on both axes. A bot at ~4.3 m/s covers
#: ~344 blocks in an 80 s episode, so 512 clears that even without walls; walls
#: make it moot and the spacing is free (flat world, nothing is generated between
#: pads because view/simulation distance is 2).
PAD_SPACING: int = 512

#: Pads per grid row. Row-major: index i sits at column ``i % PAD_GRID_COLS`` and
#: row ``i // PAD_GRID_COLS``, so 25 pads form a compact 5x5 block rather than one
#: 12800-block line.
PAD_GRID_COLS: int = 5


@dataclass(frozen=True)
class PadAnchor:
    """A pad's ANCHOR: the learner SPAWN CELL, not the floor origin.

    The learner's feet land at ``(x + 0.5, 64, z + 0.5)`` and the dummy's at
    ``(x + 3.5, 64, z + 0.5)``; the floor spans ``x-8 .. x+16`` by ``z-12 .. z+12``
    at ``y = 63``. Conflating the anchor with the floor origin is a documented
    first-draft error of this plan -- all pad math is expressed against the anchor.
    """

    x: int
    z: int

    def as_flag(self) -> str:
        """Render as the ``--pad-origin`` value ``"<x>,<z>"`` run.js accepts.

        Deliberately plain non-negative integers with no spaces: ``bridge/run.js``
        validates the string with ``/^\\d+$/`` per component because the value is
        pasted TEXTUALLY into the ``arena:setup_pad`` macro arguments.
        """
        return f"{self.x},{self.z}"

    def to_dict(self) -> Dict[str, int]:
        """Plain-dict view (the unit the launch plan carries)."""
        return {"x": self.x, "z": self.z}


def pad_anchor(index: int) -> PadAnchor:
    """Return the anchor of pad ``index``. THE sole coordinate source in this repo.

    Formula (plan, Contracts/Signatures)::

        x = (i %  PAD_GRID_COLS) * PAD_SPACING
        z = (i // PAD_GRID_COLS) * PAD_SPACING

    So pad 0 is ``(0, 0)`` -- byte-identical to today's single arena -- pad 4 is
    ``(2048, 0)`` and pad 5 wraps to the next row at ``(0, 512)``.

    Args:
        index: 0-based pad index.

    Returns:
        The :class:`PadAnchor` for that pad.

    Raises:
        ValueError: if ``index`` is negative or not an integer. Both are launcher
            bugs, and a negative anchor would be silently rejected much later by
            run.js's non-negative-integer gate.
    """
    if isinstance(index, bool) or not isinstance(index, int):
        raise ValueError(f"pad index must be an int, got {index!r}")
    if index < 0:
        raise ValueError(f"pad index must be >= 0, got {index}")
    return PadAnchor(
        x=(index % PAD_GRID_COLS) * PAD_SPACING,
        z=(index // PAD_GRID_COLS) * PAD_SPACING,
    )


def pad_usernames(index: int) -> Tuple[str, str]:
    """Return ``(learner_username, dummy_username)`` for pad ``index``.

    ``i == 0`` is DELIBERATELY ``learner_bot`` / ``dummy_bot`` and not
    ``learner_0``, so the committed ``server/ops.json``, the datapack's single-arena
    wrapper and every existing runbook step stay byte-identical. This is the same
    policy as ``usernamesForPad`` in ``bridge/run.js``; the two must agree, and
    ``tests/test_pad_launcher.py`` pins both literals.
    """
    if isinstance(index, bool) or not isinstance(index, int):
        raise ValueError(f"pad index must be an int, got {index!r}")
    if index < 0:
        raise ValueError(f"pad index must be >= 0, got {index}")
    if index == 0:
        return ("learner_bot", "dummy_bot")
    return (f"learner_{index}", f"dummy_{index}")


# --- ops.json (offline-mode op list) ----------------------------------------
#: Prefix Mojang hashes to derive an offline-mode player UUID.
_OFFLINE_UUID_PREFIX = "OfflinePlayer:"

#: Op level written for every bot. 4 is required: the bridge issues
#: ``/function arena:setup_pad`` and ``/function arena:reset_pad`` as the opped
#: learner, and a non-op cannot run /function at all.
_OPS_LEVEL: int = 4


def offline_uuid(username: str) -> str:
    """Return the offline-mode UUID Paper assigns to ``username``.

    Mirrors Java's ``UUID.nameUUIDFromBytes(("OfflinePlayer:" + name).getBytes(UTF_8))``:
    an MD5 digest with the version nibble forced to 3 (name-based) and the IETF
    variant bits forced to ``10``. Verified against the two UUIDs already committed
    in ``server/ops.json``.
    """
    if not isinstance(username, str) or username.strip() == "":
        raise ValueError(f"username must be a non-empty string, got {username!r}")
    digest = bytearray(
        hashlib.md5((_OFFLINE_UUID_PREFIX + username).encode("utf-8")).digest()
    )
    digest[6] = (digest[6] & 0x0F) | 0x30  # version 3 (name-based, MD5)
    digest[8] = (digest[8] & 0x3F) | 0x80  # IETF variant (10xx)
    return str(uuid.UUID(bytes=bytes(digest)))


def ops_entries(n_pads: int) -> List[Dict[str, object]]:
    """Return the ``ops.json`` entries for every bot of an ``n_pads`` fleet.

    2N entries, learner then dummy per pad in ascending pad order, each at level 4.
    The key order and types match what the live Paper server writes back, so a
    server rewrite is a no-op diff rather than a reshuffle.
    """
    if n_pads < 1:
        raise ValueError(f"n_pads must be >= 1, got {n_pads}")
    entries: List[Dict[str, object]] = []
    for index in range(n_pads):
        for username in pad_usernames(index):
            entries.append(
                {
                    "uuid": offline_uuid(username),
                    "name": username,
                    "level": _OPS_LEVEL,
                    "bypassesPlayerLimit": False,
                }
            )
    return entries


def ops_json(n_pads: int) -> str:
    """Render :func:`ops_entries` as the exact ``ops.json`` file text.

    Two-space indent plus a trailing newline: at ``n_pads == 1`` this is
    byte-identical to the committed ``server/ops.json`` (pinned by a test).
    """
    return json.dumps(ops_entries(n_pads), indent=2) + "\n"


def write_ops_json(n_pads: int, path: str) -> str:
    """Write ``ops.json`` for an ``n_pads`` fleet and return the path written.

    MUST run BEFORE Paper boots: the server reads the op list at startup and
    rewrites the file itself afterwards, so a write into a running server is both
    ignored and likely to be clobbered.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(ops_json(n_pads), encoding="utf-8")
    return str(target)


def missing_ops(n_pads: int, path: str) -> List[str]:
    """Return the bot usernames NOT opped at level 4 in the ``ops.json`` at ``path``.

    Used by ``start-pads.sh`` in ``--no-server`` (attach) mode, where the file
    cannot be rewritten usefully because Paper is already running. An unreadable or
    malformed file reports EVERY username as missing rather than passing silently.
    """
    required = [name for index in range(n_pads) for name in pad_usernames(index)]
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return required
    if not isinstance(raw, list):
        return required
    opped = set()
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        try:
            level = int(entry.get("level", 0))
        except (TypeError, ValueError):
            continue
        if isinstance(name, str) and level >= _OPS_LEVEL:
            opped.add(name)
    return [name for name in required if name not in opped]


def required_max_players(n_pads: int) -> int:
    """Minimum ``server.properties`` ``max-players`` for an ``n_pads`` fleet: ``2N + 10``.

    2N bots plus 10 slots of headroom for a human joining to look at the world, a
    lingering ghost session after a bridge restart, and the reconnect overlap when a
    single pad's bridge is relaunched mid-run.
    """
    if n_pads < 1:
        raise ValueError(f"n_pads must be >= 1, got {n_pads}")
    return 2 * n_pads + 10


# --- Defaults (mirror bridge/bot.js DEFAULT_BOT_CONFIG + server/setup) --------
#: ONE JVM serves every pad, so this is a single port, not a base. It matches
#: ``server.properties`` ``server-port`` written by ``server/setup/setup.sh``.
_DEFAULT_MC_PORT: int = 25565

#: Bridge TCP port base; pad ``i`` listens on ``base + i``. Pad 0 -> 5555, the
#: single-arena default, so an N=1 fleet uses exactly today's port.
_DEFAULT_BRIDGE_BASE_PORT: int = 5555

#: Executables, injectable so a dry-run prints a deterministic plan and a test can
#: point at a fake binary. ``node`` is a documented prerequisite on PATH.
_DEFAULT_NODE: str = "node"

#: Bounded wait for a bridge's TCP port to accept a connection after its process is
#: spawned. run.js connects BOTH bots to Paper before it calls listen(), so "the
#: port is open" is exactly "both bots joined and spawned" -- that is the join gate.
_DEFAULT_BRIDGE_READY_TIMEOUT_SECONDS: float = 120.0
_DEFAULT_BRIDGE_READY_POLL_SECONDS: float = 0.5

#: Separate, much SHORTER bound on waiting for a dying bridge to release its port
#: before a relaunch. A genuinely dead process frees the port at once; this only
#: absorbs the moment between SIGTERM and the socket closing. It must not reuse the
#: ready timeout: launch() runs on a collector thread, and blocking it for two
#: minutes to discover that the pad's bridge was alive all along is a stall, not a
#: diagnosis.
_DEFAULT_BRIDGE_PORT_FREE_TIMEOUT_SECONDS: float = 15.0

#: Grace period given to a process to exit on terminate() before we kill() it.
_DEFAULT_TERMINATE_GRACE_SECONDS: float = 10.0

#: Prime-barrier retry budget per pad, and the initial (doubling) backoff.
_DEFAULT_PRIME_ATTEMPTS: int = 3
_DEFAULT_PRIME_BACKOFF_SECONDS: float = 3.0

#: Lines of a failing pad's bridge log echoed when the prime barrier gives up. The
#: fleet is debugged live by a human, so the diagnosis must be in the terminal.
_DEFAULT_LOG_TAIL_LINES: int = 40


def _ascii_log(message: str) -> None:
    """Print one ASCII launcher log line to stderr (never unicode)."""
    safe = message.encode("ascii", "backslashreplace").decode("ascii")
    print(f"[launcher] {safe}", file=sys.stderr, flush=True)


@dataclass(frozen=True)
class PadSpec:
    """The fully-resolved launch parameters for ONE pad.

    Pure data: produced by :meth:`SubprocessArenaLauncher.spec_for` and by
    :func:`plan`, with no side effects, so the printed plan, the argv bash spawns
    and the argv a mid-run relaunch spawns are all one source of truth.

    Attributes:
        pad_index: 0-based pad index.
        mc_port: The SHARED Minecraft port (one JVM; identical for every pad).
        bridge_port: This pad's bridge TCP port (``bridge_base_port + pad_index``).
        anchor: This pad's :class:`PadAnchor`.
        learner_username / dummy_username: This pad's bot usernames.
        bridge_command: The exact argv used to launch the Node bridge, run from the
            repo root.
    """

    pad_index: int
    mc_port: int
    bridge_port: int
    anchor: PadAnchor
    learner_username: str
    dummy_username: str
    bridge_command: List[str]

    def to_dict(self) -> Dict[str, object]:
        """Return a plain-dict view of this spec (the unit the planner returns)."""
        return {
            "pad_index": self.pad_index,
            "mc_port": self.mc_port,
            "bridge_port": self.bridge_port,
            "anchor_x": self.anchor.x,
            "anchor_z": self.anchor.z,
            "pad_origin": self.anchor.as_flag(),
            "learner_username": self.learner_username,
            "dummy_username": self.dummy_username,
            "bridge_command": list(self.bridge_command),
        }


@dataclass
class PadProcess:
    """The live ``Popen`` handle for one pad's bridge (there is no per-pad JVM)."""

    bridge: Optional[subprocess.Popen] = None


class SubprocessArenaLauncher:
    """Real :class:`~distributed.actor.ArenaLauncher`: restarts ONE pad's bridge.

    In the one-JVM topology a "relaunch" is a BRIDGE relaunch. The Paper JVM is
    shared by every pad and is owned by ``server/setup/start-pads.sh``, so this class
    never starts or stops a JVM: killing it to recover one pad would take the whole
    fleet down. A dead JVM is the run-aborting fault, and that policy is T11's.

    The class name and the ``bridge_base_port`` keyword are preserved from the
    N-JVM version because ``agent/train.py`` constructs it exactly that way.

    Args:
        repo_root: Repository root, used to resolve ``bridge/run.js`` and as the
            bridge process cwd. Defaults to this file's repo root (two levels up).
        mc_port: The shared Minecraft port every bridge connects to. Default 25565.
        bridge_base_port: Bridge port base; pad ``i`` uses ``base + i``. Default 5555.
        node: The Node executable (resolved on PATH).
        bridge_ready_timeout_seconds / bridge_ready_poll_seconds: Bounded wait for a
            freshly spawned bridge's port to accept a connection (the join gate).
        bridge_port_free_timeout_seconds: Shorter bound on waiting for a dying
            bridge to RELEASE its port before a relaunch spawns a replacement.
        terminate_grace_seconds: Grace before terminate() escalates to kill().
        popen: Injectable process spawner (defaults to ``subprocess.Popen``); a test
            passes a fake so no real Node process is started.
        sleep: Injectable sleep (defaults to ``time.sleep``).
        port_probe: Injectable ``(host, port) -> bool`` TCP-connect check; a test
            passes a stub so it never touches the network.
        log: Injectable ASCII log sink.
    """

    def __init__(
        self,
        *,
        repo_root: Optional[str] = None,
        mc_port: int = _DEFAULT_MC_PORT,
        bridge_base_port: int = _DEFAULT_BRIDGE_BASE_PORT,
        node: str = _DEFAULT_NODE,
        bridge_ready_timeout_seconds: float = _DEFAULT_BRIDGE_READY_TIMEOUT_SECONDS,
        bridge_ready_poll_seconds: float = _DEFAULT_BRIDGE_READY_POLL_SECONDS,
        bridge_port_free_timeout_seconds: float = _DEFAULT_BRIDGE_PORT_FREE_TIMEOUT_SECONDS,
        terminate_grace_seconds: float = _DEFAULT_TERMINATE_GRACE_SECONDS,
        popen=subprocess.Popen,
        sleep=time.sleep,
        port_probe=None,
        log: Callable[[str], None] = _ascii_log,
    ) -> None:
        resolved_repo = Path(repo_root) if repo_root is not None else _default_repo_root()
        self._repo_root = resolved_repo.resolve()
        self._mc_port = int(mc_port)
        self._bridge_base_port = int(bridge_base_port)
        self._node = node
        self._bridge_ready_timeout_seconds = float(bridge_ready_timeout_seconds)
        self._bridge_ready_poll_seconds = float(bridge_ready_poll_seconds)
        self._bridge_port_free_timeout_seconds = float(bridge_port_free_timeout_seconds)
        self._terminate_grace_seconds = float(terminate_grace_seconds)
        self._popen = popen
        self._sleep = sleep
        self._port_probe = port_probe if port_probe is not None else _tcp_port_open
        self._log = log

        # run.js path (the bridge entry). Resolved once; the live spawn uses it.
        self._run_js = self._repo_root / "bridge" / "run.js"

        # Per-pad live handles, guarded because the pool may relaunch from different
        # collector threads. Read-modify-write of the dict happens only under this
        # lock; the spawned Popen objects themselves are safe to signal.
        self._lock = threading.Lock()
        self._procs: Dict[int, PadProcess] = {}

    # -- pure spec derivation (shared by the plan + the live relaunch) ------

    def spec_for(self, pad_index: int) -> PadSpec:
        """Resolve every launch parameter for ``pad_index`` (pure: no side effects).

        The bridge port is ``base + pad_index``; the anchor comes from
        :func:`pad_anchor`; the usernames from :func:`pad_usernames`. ``--pad-origin``
        is passed EXPLICITLY because run.js refuses to derive an anchor from an
        index -- defaulting it would stack pad i on pad 0. Usernames are passed
        explicitly too so the argv is self-describing in a process listing.
        """
        anchor = pad_anchor(pad_index)
        bridge_port = self._bridge_base_port + pad_index
        learner_username, dummy_username = pad_usernames(pad_index)

        bridge_command = [
            self._node,
            str(self._run_js),
            "--port",
            str(self._mc_port),
            "--bridge-port",
            str(bridge_port),
            "--pad-index",
            str(pad_index),
            "--pad-origin",
            anchor.as_flag(),
            "--learner-username",
            learner_username,
            "--dummy-username",
            dummy_username,
        ]

        return PadSpec(
            pad_index=pad_index,
            mc_port=self._mc_port,
            bridge_port=bridge_port,
            anchor=anchor,
            learner_username=learner_username,
            dummy_username=dummy_username,
            bridge_command=bridge_command,
        )

    def plan(self, n_pads: int) -> List[Dict[str, object]]:
        """Return the per-pad launch plan (one dict per pad), spawning NOTHING."""
        if n_pads < 1:
            raise ValueError(f"n_pads must be >= 1, got {n_pads}")
        return [self.spec_for(i).to_dict() for i in range(n_pads)]

    # -- ArenaLauncher Protocol (the live, mid-run relaunch path) ----------

    def launch(self, arena_id: int) -> None:
        """Start (or restart) the BRIDGE for pad ``arena_id``. The JVM is untouched.

        Called by :class:`~distributed.actor.ActorPool` when a collector cannot
        recover its connection. Sequence: drop any handle we own for this pad, wait
        (bounded) for its port to become FREE, spawn a fresh bridge, wait (bounded)
        for the port to come back up.

        The wait-for-free step is the important one. Bridges are normally started by
        ``start-pads.sh``, which is a DIFFERENT process tree, so this object usually
        has no handle to kill. If the port is still occupied after the wait, some
        bridge is alive on it and this raises rather than spawning a duplicate that
        would fail to bind -- or, worse, race the survivor for the single TCP client
        slot. ``ActorPool`` treats a launcher exception as a failed relaunch attempt
        and backs off, which is the correct response either way.

        NEVER primes. A prime connection here would steal the single client slot
        from the collector that is trying to reconnect. Bots do not re-stack at pad 0
        on a bridge restart: ``arena:reset_pad`` pins a per-bot ``/spawnpoint`` and
        that persists in player NBT across reconnects.

        Raises:
            ValueError: if ``arena_id`` is negative.
            RuntimeError: if Paper is unreachable, if the port stays occupied, or if
                the fresh bridge does not come up within the bounded wait.
        """
        spec = self.spec_for(arena_id)

        # Idempotent restart: clear any prior handle for this pad first.
        self.terminate(arena_id)

        # Precondition, not a fault policy: a bridge spawned against a dead JVM
        # exits 1 immediately and the collector would spin on it. Naming the JVM
        # here turns that into one clear line. (The public ``jvm_alive()`` seam and
        # the abort policy that consumes it are T11's, not this check.)
        if not self._port_probe("127.0.0.1", spec.mc_port):
            raise RuntimeError(
                f"pad {arena_id}: the Paper JVM is not accepting connections on mc "
                f"port {spec.mc_port}; a bridge started now would exit immediately. "
                f"The whole fleet shares this one JVM."
            )

        self._wait_for_port_free(arena_id, spec)

        self._log(
            f"pad {arena_id}: starting bridge on port {spec.bridge_port} "
            f"@ anchor {spec.anchor.as_flag()} "
            f"({spec.learner_username} / {spec.dummy_username})"
        )
        bridge_proc = self._popen(spec.bridge_command, cwd=str(self._repo_root))

        with self._lock:
            self._procs[arena_id] = PadProcess(bridge=bridge_proc)

        self._wait_for_bridge(arena_id, spec, bridge_proc)

    def terminate(self, arena_id: int) -> None:
        """Stop this pad's bridge. Best-effort and idempotent; the JVM is untouched."""
        with self._lock:
            handles = self._procs.pop(arena_id, None)
        if handles is None:
            return
        self._stop_process(arena_id, "bridge", handles.bridge)

    # -- internals ---------------------------------------------------------

    def _wait_for_port_free(self, pad_index: int, spec: PadSpec) -> None:
        """Block until this pad's bridge port stops accepting connections, bounded.

        A still-open port means a live bridge owns the pad. Raise instead of
        spawning a second process for it.

        The probe CONNECTS, and BridgeServer evicts an incumbent client when a
        second one arrives, so this is only safe because of WHEN it runs: the pool
        calls launch() after its collector has already lost the connection, so there
        is no live client of this pad's bridge left to evict. Do not reuse this
        probe on a pad that a driver is still talking to.
        """
        deadline = time.monotonic() + self._bridge_port_free_timeout_seconds
        while True:
            if not self._port_probe("127.0.0.1", spec.bridge_port):
                return
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"pad {pad_index}: bridge port {spec.bridge_port} is still "
                    f"accepting connections after "
                    f"{self._bridge_port_free_timeout_seconds:.0f}s, so a bridge is "
                    f"still alive on it (probably one started by "
                    f"server/setup/start-pads.sh). Refusing to spawn a duplicate."
                )
            self._sleep(self._bridge_ready_poll_seconds)

    def _wait_for_bridge(
        self, pad_index: int, spec: PadSpec, bridge_proc: subprocess.Popen
    ) -> None:
        """Block until this pad's bridge port accepts a connection, bounded.

        run.js connects BOTH bots and waits for both spawns before it calls
        ``listen()``, so a successful probe means both bots are in the world. If the
        process exits first, or the wait elapses, this raises so a wedged bridge
        surfaces instead of a collector spinning on a port that never opens.

        Connecting is both safe and NECESSARY here: safe because the bridge was
        spawned moments ago and has no client yet, and necessary because only an
        accepted connection proves the two bots joined -- a non-connecting check
        would succeed as soon as the process existed and prove nothing.
        """
        deadline = time.monotonic() + self._bridge_ready_timeout_seconds
        while True:
            exit_code = bridge_proc.poll()
            if exit_code is not None:
                raise RuntimeError(
                    f"pad {pad_index}: bridge exited (code {exit_code}) before port "
                    f"{spec.bridge_port} was reachable"
                )
            if self._port_probe("127.0.0.1", spec.bridge_port):
                self._log(
                    f"pad {pad_index}: bridge port {spec.bridge_port} is up "
                    f"(both bots joined)"
                )
                return
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"pad {pad_index}: bridge port {spec.bridge_port} did not come "
                    f"up within {self._bridge_ready_timeout_seconds:.0f}s"
                )
            self._sleep(self._bridge_ready_poll_seconds)

    def _stop_process(
        self, pad_index: int, label: str, proc: Optional[subprocess.Popen]
    ) -> None:
        """Terminate one process gracefully, escalating to kill, swallowing errors."""
        if proc is None:
            return
        try:
            if proc.poll() is not None:
                return  # already exited
            self._log(f"pad {pad_index}: terminating {label}")
            proc.terminate()
            try:
                proc.wait(timeout=self._terminate_grace_seconds)
            except subprocess.TimeoutExpired:
                self._log(f"pad {pad_index}: {label} did not exit, killing")
                proc.kill()
        except Exception as exc:  # noqa: BLE001 - teardown is best-effort/idempotent
            self._log(f"pad {pad_index}: error stopping {label} (ignored): {exc}")


def _default_repo_root() -> Path:
    """Repo root for the default launcher: two levels up from this file.

    ``distributed/launcher.py`` -> ``distributed/`` -> repo root.
    """
    return Path(__file__).resolve().parent.parent


def _tcp_port_open(host: str, port: int, *, timeout: float = 1.0) -> bool:
    """True if a TCP connection to ``host:port`` succeeds within ``timeout``.

    A plain connect probe (NOT a Minecraft handshake, NOT a bridge handshake).
    """
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def plan(
    n_pads: int,
    *,
    mc_port: int = _DEFAULT_MC_PORT,
    bridge_base_port: int = _DEFAULT_BRIDGE_BASE_PORT,
    repo_root: Optional[str] = None,
    node: str = _DEFAULT_NODE,
) -> List[Dict[str, object]]:
    """Build the per-pad launch plan WITHOUT constructing any live state.

    Importable so a test, the ``--dry-run`` CLI and ``start-pads.sh`` (via
    ``--emit-plan``) all read the same anchors/ports/usernames/argv. Delegates to
    :meth:`SubprocessArenaLauncher.spec_for` so the printed plan, the bash spawn and
    a mid-run relaunch can never diverge.

    Args:
        n_pads: Number of pads (must be >= 1).
        (all others): Forwarded to :class:`SubprocessArenaLauncher`.

    Returns:
        A list of per-pad dicts: ``pad_index``, ``mc_port``, ``bridge_port``,
        ``anchor_x``, ``anchor_z``, ``pad_origin``, ``learner_username``,
        ``dummy_username``, ``bridge_command``.
    """
    launcher = SubprocessArenaLauncher(
        repo_root=repo_root,
        mc_port=mc_port,
        bridge_base_port=bridge_base_port,
        node=node,
    )
    return launcher.plan(n_pads)


# --- The reset-before-step barrier ------------------------------------------


class PadPrimeError(RuntimeError):
    """Raised when a pad could not be primed (reset once) at fleet boot.

    Loud by construction: the message names the pad, its bridge port and its anchor.
    A fleet with an unprimed pad must NOT start training -- that pad's two bots are
    still stacked in pad 0 and will be hit by pad 0's learner.
    """


def _log_tail(path: Path, lines: int) -> str:
    """Return the last ``lines`` lines of ``path``, or a short note if unreadable."""
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"(could not read {path}: {exc})"
    tail = content.splitlines()[-lines:]
    if not tail:
        return f"({path} is empty)"
    return "\n".join(f"  | {line}" for line in tail)


def _default_prime_env_factory(pad_index: int, host: str, bridge_port: int):
    """Build a live :class:`~env.mc_pvp_env.MCPvPEnv` for one pad's bridge.

    Imported lazily so ``--dry-run`` / ``--write-ops`` / ``--emit-plan`` stay free of
    the numpy/env dependency chain (``start-pads.sh`` calls those before it has any
    reason to require a fully provisioned venv).
    """
    from env.mc_pvp_env import MCPvPEnv, TcpBridgeClient

    transport = TcpBridgeClient(host=host, port=bridge_port)
    return MCPvPEnv(transport=transport)


def prime_pads(
    n_pads: int,
    *,
    host: str = "127.0.0.1",
    bridge_base_port: int = _DEFAULT_BRIDGE_BASE_PORT,
    attempts: int = _DEFAULT_PRIME_ATTEMPTS,
    backoff_seconds: float = _DEFAULT_PRIME_BACKOFF_SECONDS,
    log_dir: Optional[str] = None,
    log_tail_lines: int = _DEFAULT_LOG_TAIL_LINES,
    env_factory: Optional[Callable[[int, str, int], object]] = None,
    sleep=time.sleep,
    log: Callable[[str], None] = _ascii_log,
) -> List[int]:
    """Reset every pad ONCE, so no pad can step while another is still stacked.

    Connects to each pad's bridge in turn, runs one full ``reset`` (which is the
    bridge's ``arena:reset_pad`` macro plus its read-back gate), then CLOSES the
    connection so the training driver can take the bridge's single client slot.

    Order is DESCENDING (pad N-1 first, pad 0 last), and the reason is COLLISION,
    not damage. Nothing can hit anything during the prime: the driver is not running
    and this only calls ``reset()``, so no ATTACK is ever issued. What descending
    buys is that pad 0 -- the shared world spawn, and therefore the stack site where
    all 2N bots are standing on top of each other -- is reset LAST, once every other
    pad's pair has been teleported away. Player-player collision shove is live
    regardless of PvP, so resetting pad 0 into a crowd risks its learner being pushed
    off ``(anchor+0.5, 64, anchor+0.5)`` between the teleport and the read-back, which
    fails the position gate and turns a healthy pad into a spurious prime failure.
    Resetting outside-in makes pad 0's reset land in an empty pad, and incidentally
    gives its ``_scanForeignPlayers`` a clean result to report.

    A prime failure dumps the tail of that pad's bridge log (when ``log_dir`` is
    given) before raising, because this runs unattended inside ``start-pads.sh`` and
    the diagnosis has to reach the operator's terminal.

    Args:
        n_pads: Number of pads to prime (must be >= 1).
        host: Bridge host. Loopback by default.
        bridge_base_port: Pad ``i``'s bridge listens on ``base + i``.
        attempts: Per-pad attempts before the pad is declared failed (>= 1).
        backoff_seconds: Initial backoff between attempts; doubles each retry.
        log_dir: Directory holding ``pad-<i>.log`` bridge logs, for the failure dump.
        log_tail_lines: How many trailing log lines to echo on failure.
        env_factory: ``(pad_index, host, bridge_port) -> env`` with ``reset()`` and
            ``close()``. Defaults to a live :class:`~env.mc_pvp_env.MCPvPEnv`; tests
            inject a fake so no socket is opened.
        sleep: Injectable sleep.
        log: Injectable ASCII log sink.

    Returns:
        The pad indices primed, in the order they were primed (descending).

    Raises:
        ValueError: on a bad ``n_pads`` / ``attempts``.
        PadPrimeError: if any pad could not be reset within its attempts.
    """
    if n_pads < 1:
        raise ValueError(f"n_pads must be >= 1, got {n_pads}")
    if attempts < 1:
        raise ValueError(f"attempts must be >= 1, got {attempts}")

    factory = env_factory if env_factory is not None else _default_prime_env_factory
    primed: List[int] = []

    log(
        f"priming {n_pads} pad(s) in descending order -- every pad is reset before "
        f"any pad may step"
    )
    for pad_index in range(n_pads - 1, -1, -1):
        bridge_port = bridge_base_port + pad_index
        anchor = pad_anchor(pad_index)
        last_exc: Optional[BaseException] = None

        for attempt in range(1, attempts + 1):
            env = None
            try:
                env = factory(pad_index, host, bridge_port)
                env.reset()
                primed.append(pad_index)
                log(
                    f"pad {pad_index}: primed (port {bridge_port}, anchor "
                    f"{anchor.as_flag()})"
                )
                last_exc = None
                break
            except Exception as exc:  # noqa: BLE001 - any failure is a pad failure
                last_exc = exc
                log(
                    f"pad {pad_index}: prime attempt {attempt}/{attempts} failed: "
                    f"{type(exc).__name__}: {exc}"
                )
                if attempt < attempts:
                    delay = backoff_seconds * (2 ** (attempt - 1))
                    log(f"pad {pad_index}: retrying in {delay:.1f}s")
                    sleep(delay)
            finally:
                # Always hand the single client slot back, success or failure.
                if env is not None:
                    try:
                        env.close()
                    except Exception as close_exc:  # noqa: BLE001 - best effort
                        log(f"pad {pad_index}: error closing prime env: {close_exc}")

        if last_exc is not None:
            detail = ""
            if log_dir is not None:
                log_path = Path(log_dir) / f"pad-{pad_index}.log"
                detail = (
                    f"\nlast {log_tail_lines} lines of {log_path}:\n"
                    f"{_log_tail(log_path, log_tail_lines)}"
                )
            raise PadPrimeError(
                f"pad {pad_index} could not be primed after {attempts} attempt(s) "
                f"(bridge 127.0.0.1:{bridge_port}, anchor {anchor.as_flag()}, "
                f"bots {'/'.join(pad_usernames(pad_index))}). The fleet must NOT "
                f"start training: this pad's bots are still stacked in pad 0. "
                f"Last error: {type(last_exc).__name__}: {last_exc}{detail}"
            ) from last_exc

    log(f"primed {len(primed)} pad(s): {','.join(str(i) for i in primed)}")
    return primed


# --- CLI ---------------------------------------------------------------------

#: Field separator for --emit-plan. Bridge argv elements never contain a tab, so a
#: tab-separated line survives bash word-splitting with IFS=$'\t' intact -- no
#: quoting, no eval, no JSON parser in bash.
_PLAN_FIELD_SEP = "\t"


def _format_plan(pad_plan: Sequence[Dict[str, object]]) -> str:
    """Render the launch plan as an ASCII, human-readable block (no unicode)."""
    lines: List[str] = []
    lines.append(f"launch plan for {len(pad_plan)} pad(s) in ONE Paper JVM:")
    for entry in pad_plan:
        bridge_cmd = " ".join(str(part) for part in entry["bridge_command"])
        lines.append("")
        lines.append(f"  pad {entry['pad_index']}:")
        lines.append(f"    mc_port          : {entry['mc_port']} (shared)")
        lines.append(f"    bridge_port      : {entry['bridge_port']}")
        lines.append(f"    anchor           : {entry['pad_origin']}")
        lines.append(f"    learner_username : {entry['learner_username']}")
        lines.append(f"    dummy_username   : {entry['dummy_username']}")
        lines.append(f"    bridge_command   : {bridge_cmd}")
    return "\n".join(lines)


def _emit_plan(pad_plan: Sequence[Dict[str, object]]) -> str:
    """Render the plan as machine-readable, tab-separated lines for ``start-pads.sh``.

    One line per pad::

        <pad_index> <bridge_port> <anchor_x> <anchor_z> <learner> <dummy> <argv...>

    Fields 1..6 are fixed; every remaining field is one element of the bridge argv,
    in order. Bash reads it with ``IFS=$'\\t' read -r -a fields`` and spawns
    ``"${fields[@]:6}"``, so the anchors AND the flag names come from this module --
    nothing about the topology is recomputed in bash.
    """
    lines: List[str] = []
    for entry in pad_plan:
        fields = [
            str(entry["pad_index"]),
            str(entry["bridge_port"]),
            str(entry["anchor_x"]),
            str(entry["anchor_z"]),
            str(entry["learner_username"]),
            str(entry["dummy_username"]),
        ]
        fields.extend(str(part) for part in entry["bridge_command"])
        lines.append(_PLAN_FIELD_SEP.join(fields))
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry: ``python -m distributed.launcher --pads N <mode>``.

    Modes are mutually exclusive and every one of them is safe to run by hand:

      ``--dry-run``    print the human-readable plan (the default; spawns nothing).
      ``--emit-plan``  print the tab-separated plan ``start-pads.sh`` consumes.
      ``--write-ops``  write ``server/ops.json`` for all 2N bots (BEFORE Paper boots).
      ``--check-ops``  verify an existing ``ops.json`` covers all 2N bots.
      ``--prime``      run the reset-before-step barrier against running bridges.

    This CLI never starts Paper and never starts a bridge -- ``start-pads.sh`` owns
    the fleet boot, and :meth:`SubprocessArenaLauncher.launch` owns the mid-run,
    single-pad relaunch.
    """
    parser = argparse.ArgumentParser(
        prog="python -m distributed.launcher",
        description=(
            "Pad-fleet planning helpers for the one-JVM/N-pad topology: print the "
            "launch plan, write ops.json, or run the reset-before-step prime "
            "barrier. server/setup/start-pads.sh is the operator entry point."
        ),
    )
    parser.add_argument(
        "--pads",
        type=int,
        default=1,
        help="Number of pads (one bridge each, all in one Paper JVM). Default 1.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the per-pad launch plan and spawn NOTHING (the default).",
    )
    mode.add_argument(
        "--emit-plan",
        action="store_true",
        help="Print the tab-separated machine-readable plan for start-pads.sh.",
    )
    mode.add_argument(
        "--write-ops",
        action="store_true",
        help="Write ops.json opping all 2N bots at level 4. Run BEFORE Paper boots.",
    )
    mode.add_argument(
        "--check-ops",
        action="store_true",
        help="Verify an existing ops.json opps all 2N bots; exit 1 listing any gaps.",
    )
    mode.add_argument(
        "--prime",
        action="store_true",
        help=(
            "Reset every pad once, descending, so no pad can step while another is "
            "still stacked at the shared world spawn. Bridges must already be up."
        ),
    )
    parser.add_argument(
        "--mc-port",
        type=int,
        default=_DEFAULT_MC_PORT,
        help=f"The shared Minecraft port (one JVM). Default {_DEFAULT_MC_PORT}.",
    )
    parser.add_argument(
        "--bridge-base-port",
        type=int,
        default=_DEFAULT_BRIDGE_BASE_PORT,
        help=f"Bridge port base; pad i uses base+i. Default {_DEFAULT_BRIDGE_BASE_PORT}.",
    )
    parser.add_argument(
        "--ops-path",
        type=str,
        default=None,
        help="ops.json path for --write-ops/--check-ops. Default <repo>/server/ops.json.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Bridge host for --prime. Default 127.0.0.1.",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Directory of pad-<i>.log bridge logs, echoed on a --prime failure.",
    )
    args = parser.parse_args(argv)

    if args.pads < 1:
        parser.error(f"--pads must be >= 1, got {args.pads}")

    ops_path = args.ops_path or str(_default_repo_root() / "server" / "ops.json")

    if args.write_ops:
        written = write_ops_json(args.pads, ops_path)
        _ascii_log(
            f"wrote {written} opping {2 * args.pads} bot(s) at level 4 for "
            f"{args.pads} pad(s). Paper reads this at BOOT."
        )
        return 0

    if args.check_ops:
        gaps = missing_ops(args.pads, ops_path)
        if gaps:
            _ascii_log(
                f"FAIL: {ops_path} does not op {len(gaps)} of the "
                f"{2 * args.pads} bot(s) needed for {args.pads} pad(s) at level 4: "
                f"{','.join(gaps)}"
            )
            return 1
        _ascii_log(f"ok: {ops_path} opps all {2 * args.pads} bot(s) at level 4.")
        return 0

    if args.prime:
        try:
            prime_pads(
                args.pads,
                host=args.host,
                bridge_base_port=args.bridge_base_port,
                log_dir=args.log_dir,
            )
        except PadPrimeError as exc:
            _ascii_log(f"FAIL: {exc}")
            return 1
        return 0

    pad_plan = plan(
        args.pads,
        mc_port=args.mc_port,
        bridge_base_port=args.bridge_base_port,
    )

    if args.emit_plan:
        # stdout, unadorned: this is parsed by start-pads.sh.
        print(_emit_plan(pad_plan))
        return 0

    # Default mode: the human-readable dry run.
    print(_format_plan(pad_plan))
    _ascii_log("dry-run: no processes were started.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
