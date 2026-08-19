"""deploy.exhibition — one-command human-exhibition launcher (T5) + reset (T6).

``python -m deploy.exhibition`` starts Paper, waits for it to actually accept
connections, starts the bridge in HUMAN opponent mode (``--opponent-mode
human``, T3), connects the trained agent playing GREEDILY (epsilon=0, no
learning) from a checkpoint, and prints the LAN join address plus the pad
coordinates so a classmate can join and find the arena without being talked
through it.

``python -m deploy.exhibition --reset`` is the SEPARATE between-challengers
command (T6): it heals, repositions and re-gears both sides (an iron sword
AND a full iron set each, since M4), hands the slot to the next challenger and
re-drives play — once per invocation, never by itself. See "THE RESET COMMAND"
below.

A PLAYER CAN DO THE SAME THING WITHOUT A TERMINAL by typing ``reset`` (or
``!reset``) in Minecraft chat. That is not a second reset implementation: the
bridge writes the very same request file this module's ``--reset`` writes, and
the launcher's poll below cannot tell the two apart. See "THE IN-GAME
TRIGGER".

This module deliberately does the ORDERING ``server/setup/start-pads.sh``
already gets right — preflight gates before anything is spawned, write
``ops.json`` before Paper boots, wait for the Minecraft port, THEN start the
bridge and wait for its port — rather than inventing a second boot sequence.
It does not shell out to ``start-pads.sh`` itself: that script's bridge argv
comes from ``distributed.launcher --emit-plan``, which has no notion of
"human opponent" (T3's ``--opponent-mode``/``--challenger-username`` did not
exist when it was written), so a training-fleet bridge would connect the
dummy bot instead of waiting for a person. ``pad_anchor(0)`` / ``pad_usernames(0)``
/ ``write_ops_json`` are still reused from :mod:`distributed.launcher` — the
sole implementations of pad geometry and the op list — only the bridge argv
itself is exhibition-specific (see :func:`build_bridge_argv`).

Before Paper boots, ``run()`` writes ``server/ops.json`` for this one pad
(``distributed.launcher.write_ops_json(1, ...)`` — Paper reads the op list at
startup, so this must happen first). That file is GIT-TRACKED, and writing it
here is the same rewrite ``server/setup/start-pads.sh`` already does on every
fleet boot (issue #29): at ``n_pads == 1`` the write is byte-identical to the
committed file, so an exhibition run leaves the tree clean, but a stray dirty
diff there after a multi-pad training sweep is that pre-existing issue, not a
regression introduced by this module.

REFUSAL PATHS ARE ACCEPTANCE CRITERIA, not polish (AC5). Every gate below runs
BEFORE anything is spawned and before ``ops.json`` is written, in this order:

  1. The checkpoint must exist AND load into a :class:`~agent.dqn.DuelingDRQN`.
     Missing or unloadable -> exit non-zero, print the expected path and list
     whatever checkpoints DO exist. NEVER fall back to a randomly-initialized
     agent — a random agent in front of a classroom is worse than a clear
     error.
  2. The bridge TCP port must be free. The bridge accepts exactly ONE client
     and a second connect DESTROYS the first (``bridge/transport.js``), so a
     second launcher must never half-start: it refuses before touching Paper,
     ``ops.json`` or the bridge process. The Minecraft port is gated the same
     way, for the same "nothing started" guarantee.
  3. The bridge toolchain must resolve: the ``node`` executable, ``bridge/
     run.js`` and ``server/setup/start.sh``. AC5 does not enumerate this one,
     but leaving it out buys exactly the failure AC5 exists to forbid — a
     missing ``node`` would otherwise surface only after Paper had spent
     30-60s booting and generating a world, i.e. a launcher that starts
     something it cannot finish. It deliberately does NOT re-check the Paper
     jar or ``eula.txt``: ``start.sh`` gates both already, and a failure there
     comes back through :func:`_dump_log_tail`.

LIFECYCLE. This process is a FOREGROUND supervisor, like ``start-pads.sh``: it
plays one match, reports the result, and then WAITS — holding the agent's
bridge connection open — for either a reset request or Ctrl-C, which is what
tears Paper and the bridge down. A match ending (``done=True``) never starts
another one by itself (AC4).

THE RESET COMMAND (T6). ``python -m deploy.exhibition --reset`` does not talk to
the bridge and starts nothing. It files a one-shot REQUEST FILE
(``<log-dir>/reset.request``) that the running launcher above is polling for,
and the launcher does the work in-process:

  1. heal + reposition the HUMAN through Paper's console (see "THE HUMAN SIDE"),
  2. re-drive :func:`play_one_match`, whose ``env.reset()`` is what heals and
     repositions the LEARNER, sweeps the pad and releases the bridge's
     challenger slot for the next person.

WHY A REQUEST FILE, and not a second process that speaks the wire: the bridge
accepts exactly ONE TCP client and a second connect DESTROYS the first, so a
``--reset`` that connected would evict the live agent mid-exhibition. Among the
in-process triggers, a file beats the alternatives on this machine: a signal
needs a pidfile, and a stale pidfile whose pid has been REUSED aims SIGUSR1 at
an unrelated process (macOS has no ``/proc`` to validate it against); a signal
handler is also asynchronous, and the only async deliveries this module accepts
are the two that mean STOP — Ctrl-C and SIGTERM (see TEARDOWN below) — each of
which unwinds straight out through the teardown chain and ends the process. A
reset has to RESUME afterwards, so routing one through a signal would need the
chain to survive an interruption it then continues past, which is a different
and much weaker guarantee than the one that chain actually makes. stdin would
tie the launcher to an interactive terminal it does not otherwise need. A file also
gives the exact semantics the user asked for — "one command, one trigger" —
because a duplicate ``--reset`` collapses onto the same single file rather than
queueing a second match.

THE IN-GAME TRIGGER. Everything above is why a FILE, and the file is exactly
what makes the in-game keyword cheap: any process that can write one path can
arm a match. So the bridge — which is already in game, already reading chat, and
already holding the arena's only command channel — writes it when a player types
``reset``. :func:`build_bridge_argv` hands it ``--reset-request-path``, the same
:func:`reset_request_path` value this launcher polls, so the two cannot name
different files. The bridge connects to nothing to do this (it must not: one TCP
client, and a second connect evicts the live agent) and everything downstream —
the discard rules below, the heal, the re-arm, the one-trigger-one-match rule —
is untouched and cannot tell which trigger fired. The keyword is gated on
``--opponent-mode human``, so a training fleet reading the same chat does
nothing; the bridge-side details (case-insensitive whole-message match, own-bot
filter, debounce, the in-chat confirmation) live in ``bridge/bot.js`` under
"IN-GAME CHAT RESET".

ONE TRIGGER, ONE MATCH, AND NEVER A DEATH. The request is honored ONLY while
the launcher is idle between matches. A request filed while a match is still
running is DISCARDED (loudly) when that match ends, because honoring it would
make the death itself the proximate cause of the restart — which is the thing
AC4 forbids and the thing the user was explicit about ("after one death, no
restart auto for human"). Requests left over from an earlier launch are
discarded at startup for the same reason.

THE HUMAN SIDE. ``formatHumanResetCommands`` (``bridge/bot.js``) resets the
LEARNER only; the datapack has no template for a challenger, and the wire has no
slot for one. Two things go wrong without a human-side reset. Health is the
quiet one: it is invisible in the common case — a human who DIES respawns at
full health — but a match ending in the AGENT's death leaves the human carrying
partial health into the next round, i.e. it goes wrong exactly when somebody has
just beaten the AI in front of a room. GEAR is the loud one: the datapack re-gears
the learner on EVERY reset — since M4 (T2, issue #33) with an iron sword AND a
full iron set — and nothing in the stack ever armed the challenger. The old gap
was a bare fist (1 damage) against an iron sword (6); the M4 gap is the same
gap widened, because an unarmored human takes 6 a hit while the agent takes
3.12 through 15 armor points. The only command channel this process can reach
is Paper's own console, so ``run()`` starts Paper with a stdin PIPE and writes
the heal, reposition and re-gear lines into it (:func:`human_reset_commands`).
ARMOR IS ``item replace``, NOT ``give``: ``give`` fills an INVENTORY slot and
equips nothing, so four ``give``s would hand the challenger a full iron set to
CARRY at zero armor points — armored in the log, naked in the fight.
``item replace`` also overwrites rather than appends, which makes the four
armor lines idempotent on their own; only the sword needs its paired
sword-scoped ``clear`` to stay so.
AND THE GEAR IS READ BACK (AC9). A human has no bot connection, so the bridge's
fail-closed reset gate — which proves the BOTS' loadout off their own
connections (T3) — cannot see the challenger at all, and worn armor would be
invisible to an inventory check even if it could (slots 5-8 sit outside
mineflayer's 9-44 window). So the reset follows the gear lines with five
``data get entity`` reads on the same console and scans Paper's own log for the
server's answers (:func:`confirm_human_loadout`). That is a server-authoritative
read that needs no bot; it is BEST-EFFORT where the bots' gate is fail-closed,
because a demo that refuses to play over an unconfirmed helmet is worse than
one that plays and says so loudly.
Best-effort by construction: every command is echoed to the exhibition log, a
write failure is reported with the commands that did not run, and
``--no-paper-console`` turns the channel off. It needs a PINNED
``--challenger-username``: nothing on the wire tells this process who claimed
the slot, so an unpinned exhibition gets a warning at reset time instead of a
heal.

TEARDOWN IS ``BaseException``-PROOF, not ``Exception``-proof, AND IT IS WHERE
BOTH STOP SIGNALS LAND. Ctrl-C is the normal way an exhibition ends, so the
teardown chain runs with an operator's finger still on the key: a SECOND Ctrl-C
lands inside the first one's grace wait, and ``KeyboardInterrupt`` derives from
``BaseException``. Every helper the ``finally`` chain calls absorbs it, and the
chain itself is nested so each link still runs when the one before it raises —
otherwise a double-tap orphans the Paper JVM on port 25565 and the NEXT launch
refuses on its own mc-port gate.
``kill <pid>`` is the OTHER way an exhibition ends, and it is the one that had
to be taught: SIGTERM's default disposition tears this process down instantly
with no ``finally`` chain at all, so both children outlive their supervisor and
the orphaned JVM keeps the Minecraft port bound — the next launch then refuses
on that gate with nothing visibly running. :func:`install_sigterm_handler`
gives SIGTERM a handler that does nothing but raise
:class:`ExhibitionTerminated` (a ``BaseException``, for exactly the reason
``KeyboardInterrupt`` is one), so a ``kill`` unwinds through the same chain and
returns 143 (128 + 15) rather than Ctrl-C's 130. The handler stays installed
for the whole teardown, so a second ``kill`` is absorbed by the same helpers
that absorb a second Ctrl-C.

Every side-effecting step (spawning a process, probing a port, sleeping) is
behind an injectable seam so :mod:`tests.test_exhibition` can drive the real
decision path — :func:`run`, not a re-implementation of it — without booting
a real Paper server or Node bridge.

REFLEX SHIELD (T7). The shipped checkpoint was trained against a stationary
dummy that never left the frame, so it may never have learned to press
``Macro.TURN_TO_LAST_SEEN`` when it goes blind. ``play_one_match`` tracks how
many CONSECUTIVE decision steps the observation's ``visible`` flag (read via
the frozen ``env.observation_spec.Obs.VISIBLE`` accessor) has been ``0``; once
that streak reaches ``ExhibitionConfig.reflex_blind_steps``, this module
overrides the policy's chosen macro with ``TURN_TO_LAST_SEEN`` instead of
sending it — one ``env.step()`` call either way. The streak resets the instant
the opponent is visible again, and the override is BOUNDED: it fires for at
most ``REFLEX_MAX_CONSECUTIVE_OVERRIDES`` consecutive steps, then yields
``REFLEX_COOLOFF_STEPS`` steps back to the policy before it may fire again, so
it cannot latch on and hold the agent's yaw frozen for the rest of a match.
**This is a demo crutch, not a policy change**,
and it is structurally confined to exhibitions: ``play_one_match`` defaults
``reflex_blind_steps`` to ``0`` (the shield never even looks at ``obs`` in
that case), only ``run()`` here passes the nonzero value from
``ExhibitionConfig``, and no training entry point (``agent/train.py``,
``distributed/``) imports this module at all — the enum, the observation
layout and the training loops are all outside this file's reach.
"""

from __future__ import annotations

import argparse
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from agent.actions import Macro
from distributed.launcher import PadAnchor, pad_anchor, pad_usernames, write_ops_json
from env.mc_pvp_env import ExhibitionConfig, MCPvPEnv, TcpBridgeClient
from env.observation_spec import Obs

__all__ = [
    "ExhibitionLaunchError",
    "ExhibitionTerminated",
    "CheckpointError",
    "build_bridge_argv",
    "confirm_human_loadout",
    "drain_reset_request",
    "find_toolchain_problems",
    "human_gear_commands",
    "human_loadout_read_commands",
    "human_reset_commands",
    "install_sigterm_handler",
    "is_port_free",
    "load_greedy_policy",
    "play_one_match",
    "request_reset",
    "reset_command_hint",
    "reset_mode_conflicts",
    "reset_request_path",
    "send_paper_console_commands",
    "take_reset_request",
    "wait_for_reset_request",
    "main",
    "run",
]

# ---------------------------------------------------------------------------
# Paths (fixed, repo-internal — not user-configurable).
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
SERVER_DIR = REPO_ROOT / "server"
START_SH = SERVER_DIR / "setup" / "start.sh"
RUN_JS = REPO_ROOT / "bridge" / "run.js"
OPS_JSON_PATH = SERVER_DIR / "ops.json"

# ---------------------------------------------------------------------------
# Defaults. Mirror distributed.launcher / bridge/bot.js DEFAULT_BOT_CONFIG so
# an exhibition with no flags targets exactly the single-arena training setup
# (pad 0, mc port 25565, bridge port 5555, learner_bot/dummy_bot).
# ---------------------------------------------------------------------------

DEFAULT_MC_HOST = "127.0.0.1"
DEFAULT_MC_PORT = 25565
DEFAULT_BRIDGE_HOST = "127.0.0.1"
DEFAULT_BRIDGE_PORT = 5555
#: Exhibitions are single-pad; T5 does not expose a --pad-index flag.
PAD_INDEX = 0
#: The only checkpoint this branch has actually trained (README/RUNBOOK call
#: it "the existing checkpoint"). Overridable with --checkpoint.
DEFAULT_CHECKPOINT = "runs/m2_multi.pt"
DEFAULT_CHECKPOINTS_DIR = "runs"
DEFAULT_NODE = "node"
DEFAULT_LOG_DIR = SERVER_DIR / "logs" / "exhibition"

PAPER_READY_TIMEOUT_SECONDS = 300.0
BRIDGE_READY_TIMEOUT_SECONDS = 120.0
PORT_POLL_SECONDS = 2.0
STOP_GRACE_SECONDS = 20.0
LOG_TAIL_LINES = 40

# ---------------------------------------------------------------------------
# The reset trigger (T6).
# ---------------------------------------------------------------------------

#: The one-shot request file ``--reset`` creates and the launcher consumes. It
#: lives in the log dir, which is already this launcher's own runtime directory
#: and is git-ignored (``server/logs/``), so a stray request can never dirty the
#: tree. Every runtime file this launcher writes is ignored, ``server/ops.json``
#: included (issue #29 untracked it).
RESET_REQUEST_FILENAME = "reset.request"

#: How often the idle launcher looks for a request. Deliberately shorter than a
#: human's patience: an operator who typed the reset command in front of a class
#: and saw nothing happen for five seconds types it again, and the second one
#: would be pure noise (it collapses onto the same file, but they do not know
#: that while they are waiting).
RESET_POLL_SECONDS = 1.0

# --- The challenger's spawn, mirroring the DUMMY's datapack template --------
# ``server/arena/data/arena/function/spawn_dummy_pad.mcfunction`` places the
# opponent at the pad anchor + 3 on x, feet at y=64, facing -X (yaw 90) toward
# the learner:
#     $tp $(dummy) $(x).5 64 $(z).5
#     $execute positioned $(x).5 64 $(z).5 run tp $(dummy) ~3 ~ ~ 90 0
# and heals it with `effect clear` + instant_health/saturation. A human
# challenger stands in the same slot and gets the same treatment, so these
# constants mirror that file rather than inventing a second opponent spawn.
# tests/test_exhibition.py reads the committed datapack and fails if it moves.
CHALLENGER_SPAWN_DX = 3
PAD_SPAWN_Y = 64
#: Yaw looking -X, i.e. back at the learner (yaw 90 -> (-1, 0); see
#: ``env/perception_filter.py`` for the look-vector convention). The learner's
#: own template is the mirror image, -90. The two must stay opposite or both
#: fighters spawn pointing the same way.
CHALLENGER_SPAWN_YAW = 90

#: The challenger's WEAPON -- half their gear; the four worn pieces below are
#: the other half. Mirrors the LEARNER's line -- ``$give $(learner)
#: minecraft:iron_sword 1`` in ``spawn_learner_pad.mcfunction`` -- because the
#: fight has to be symmetric: the agent is re-geared on every reset, so a
#: barehanded human is 1 damage against 6.
CHALLENGER_WEAPON = "minecraft:iron_sword"

#: The four WORN pieces, head->feet. Mirrors the ``$item replace entity
#: $(learner) armor.<slot> with minecraft:iron_*`` lines that close the re-gear
#: block of BOTH ``spawn_learner_pad.mcfunction`` and
#: ``spawn_dummy_pad.mcfunction`` (M4/T2, issue #33), so the human fights the
#: same matchup the agent trained on. tests/test_exhibition.py reads the
#: committed datapack and fails if these four drift from it.
#:
#: THE OLD RATIONALE HERE WAS "NO ARMOR", AND IT IS RETRACTED. This file used to
#: say the checkpoint had never trained against an armored opponent, so armor
#: was out of scope. That was true when it was written and is now exactly
#: backwards: M4 arms BOTH bots, so it is the UNARMORED human who is the
#: mismatch -- they would take 6 a hit where the agent takes 3.12 through 15
#: armor points, in a fight the policy has never played.
#:
#: ORDER IS head -> feet, matching the datapack's line order and ``ARMOR_PIECES``
#: in ``bridge/bot.js``, so a missing piece is reported the same way everywhere.
CHALLENGER_ARMOR: Tuple[Tuple[str, str], ...] = (
    ("head", "minecraft:iron_helmet"),
    ("chest", "minecraft:iron_chestplate"),
    ("legs", "minecraft:iron_leggings"),
    ("feet", "minecraft:iron_boots"),
)


def _ascii_log(message: str) -> None:
    """Print one or more ASCII exhibition log lines to stderr (never unicode,
    matches the ``[launcher]``/``[start-pads]`` convention elsewhere in this
    repo). ``message`` may itself contain embedded newlines (e.g. a refusal
    message built from several lines) — each gets its own ``[exhibition]``
    prefix rather than only the first, so a multi-line refusal reads the same
    as several single-line calls would."""
    safe = message.encode("ascii", "backslashreplace").decode("ascii")
    for line in safe.split("\n"):
        print(f"[exhibition] {line}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Errors. Every instance carries an already fully-formatted, actionable
# message — the exact text `run()` prints to stderr before returning non-zero.
# ---------------------------------------------------------------------------


class ExhibitionLaunchError(RuntimeError):
    """A preflight or boot gate refused to continue."""


class CheckpointError(ExhibitionLaunchError):
    """Checkpoint missing or unloadable. `run()` never catches this to retry —
    the whole point is to refuse loudly instead of falling back to a
    randomly-initialized agent."""


class ExhibitionTerminated(BaseException):
    """SIGTERM was delivered; :func:`run` must tear down and exit.

    Derives from ``BaseException``, not ``Exception``, for the same reason
    ``KeyboardInterrupt`` does: it is raised ASYNCHRONOUSLY out of a signal
    handler, at whatever line the main thread happens to be on. An ``Exception``
    subclass would be caught by the blanket ``except Exception`` :func:`run`
    uses to keep a mid-match fault off the projector — and by any other blanket
    handler between the raise and there — so a ``kill`` would print "fatal
    error" and, worse, could be swallowed entirely by library code that catches
    broadly around its own I/O. Being a ``BaseException`` also puts it in the
    SAME teardown the Ctrl-C path uses: every helper that chain calls already
    absorbs ``BaseException`` precisely so an async interrupt cannot strand a
    child (module docstring, TEARDOWN).

    Not an :class:`ExhibitionLaunchError`: nothing refused and nothing failed —
    the operator asked this process to stop.
    """


#: Exit code for a launcher stopped by SIGTERM: 128 + 15, the shell convention,
#: and deliberately NOT the 130 (128 + 2) that Ctrl-C returns. They are two
#: different events — a person at the console versus `kill` from a script or
#: whatever supervises this process — and an operator reading a wrapper's log
#: after a demo must be able to tell them apart.
SIGTERM_EXIT_CODE = 143


# ---------------------------------------------------------------------------
# Pure decision functions — the ones tests/test_exhibition.py drives directly.
# ---------------------------------------------------------------------------


def _tcp_port_open(host: str, port: int, *, timeout: float = 1.0) -> bool:
    """True if a TCP connect to ``host:port`` succeeds within ``timeout``.

    A plain connect-and-close probe — not a Minecraft handshake, not a bridge
    handshake. Mirrors ``distributed.launcher._tcp_port_open`` /
    ``start-pads.sh``'s ``connect_probe``. Safe to call before anything of
    ours is running: there is no client yet for a stray connect to evict.
    """
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def is_port_free(
    host: str, port: int, *, port_probe: Callable[[str, int], bool] = _tcp_port_open
) -> bool:
    """True iff nothing currently answers a TCP connect to ``host:port``.

    ``port_probe`` is injectable so a test can assert the refusal path without
    opening a socket (``TC20``).
    """
    return not port_probe(host, port)


def bridge_port_in_use_message(host: str, port: int) -> str:
    """The exact refusal text for TC20 (AC5): actionable, and explicit that
    nothing has been started yet. Plain text, no ``[exhibition]`` prefix —
    that is ``log()``'s job, applied once per line."""
    return (
        f"bridge port {port} on {host} is already in use.\n"
        "the bridge accepts exactly ONE TCP client, and a second connect "
        "DESTROYS the first (bridge/transport.js) -- so a second launcher "
        "must never half-start. Stop whatever is holding that port (another "
        "exhibition, a training run, eval tooling) first, or pass "
        "--bridge-port to use a different one.\n"
        "nothing has been started."
    )


def mc_port_in_use_message(host: str, port: int) -> str:
    """Refusal text when the Minecraft port is already bound (not AC5-required,
    but the same "nothing started" guarantee should hold for it too).

    Carries the recovery COMMAND, not just the instruction to stop the server:
    the most likely way to reach this state is a previous launcher leaving a
    Paper JVM behind, and the operator hitting it is mid-demo with a room
    waiting, not at a keyboard with time to look up how to find a pid.
    """
    return (
        f"something is already listening on Minecraft port {port} on {host}. "
        "Stop the running server first, or pass --mc-port to use a different "
        "one.\n"
        "if that is a leftover Paper JVM (say, an earlier launcher that was "
        "interrupted during teardown), stop it with:\n"
        f"  lsof -ti:{port} | xargs kill\n"
        "nothing has been started."
    )


def find_toolchain_problems(
    node: str, *, which: Callable[[str], Optional[str]] = shutil.which
) -> List[str]:
    """Every unmet bridge-toolchain precondition, one actionable line each.

    Returns ``[]`` when the ``node`` executable resolves and both entry-point
    scripts exist. Reports ALL problems at once rather than the first: an
    operator repairing a demo machine should not rediscover the next missing
    piece one 30-second Paper boot at a time.

    ``RUN_JS``/``START_SH`` are read from the module globals at CALL time (not
    captured as defaults) so a test can point them at a path that does not
    exist. ``which`` is the PATH-lookup seam — it is environment-dependent I/O
    exactly like a port probe, and gating on the real PATH would make the test
    suite depend on a Node install.

    Deliberately does NOT re-check the Paper jar or ``eula.txt``: ``start.sh``
    already gates both, and a failure there surfaces via ``_dump_log_tail``.
    """
    problems: List[str] = []
    if which(node) is None:
        problems.append(
            f"the node executable {node!r} did not resolve. Install Node.js, or "
            "pass --node with an explicit path to the binary."
        )
    for label, path in (("bridge entry point", RUN_JS), ("Paper start script", START_SH)):
        if not path.is_file():
            problems.append(f"the {label} is missing: {path}")
    return problems


def toolchain_missing_message(problems: Sequence[str]) -> str:
    """Refusal text for :func:`find_toolchain_problems`' findings.

    Same "nothing has been started" framing as the port and checkpoint gates:
    this runs in the same preflight block, before Paper is spawned.
    """
    lines = ["cannot start: the bridge toolchain does not resolve."]
    lines.extend(f"  {problem}" for problem in problems)
    lines.append("nothing has been started.")
    return "\n".join(lines)


def find_checkpoints(checkpoints_dir: Path) -> List[str]:
    """Sorted ``*.pt`` file names directly under ``checkpoints_dir``.

    Returns ``[]`` for a missing/unreadable/empty directory — never raises, so
    it is safe to call purely to build a refusal message.
    """
    try:
        return sorted(p.name for p in Path(checkpoints_dir).glob("*.pt") if p.is_file())
    except OSError:
        return []


def checkpoint_missing_message(checkpoint_path: Path, checkpoints_dir: Path) -> str:
    """TC21's required text: the expected path AND whatever checkpoints exist.

    Plain text, no ``[exhibition]`` prefix -- that is ``log()``'s job, applied
    once per line so a multi-line refusal reads like several single-line
    calls instead of one.
    """
    lines = [
        f"checkpoint not found: {checkpoint_path}",
        "never falling back to a randomly-initialized agent for a demo -- "
        "pass --checkpoint with a real path.",
    ]
    existing = find_checkpoints(checkpoints_dir)
    if existing:
        lines.append(f"checkpoints found in {checkpoints_dir}:")
        lines.extend(f"  {name}" for name in existing)
    else:
        lines.append(f"no checkpoints found in {checkpoints_dir} either.")
    return "\n".join(lines)


def checkpoint_unloadable_message(
    checkpoint_path: Path, checkpoints_dir: Path, error: BaseException
) -> str:
    """Same "expected path + what exists" framing as missing, for a checkpoint
    that IS present but failed to load (bad pickle, shape mismatch, ...).

    Only the FIRST line of ``error`` goes on the headline. Torch's
    ``weights_only`` failure is a six-line essay about pickle security, and
    burying "fix or replace the checkpoint" underneath it is how an operator
    ends up reading release notes instead of picking a different file. The rest
    of the exception text is not discarded — it goes at the BOTTOM, past the
    actionable lines, where it costs nothing to skip.
    """
    detail_lines = str(error).splitlines()
    headline = detail_lines[0].strip() if detail_lines else ""
    if not headline:
        headline = f"<{type(error).__name__} with no message>"
    lines = [
        f"checkpoint at {checkpoint_path} could not be loaded: {headline}",
        "never falling back to a randomly-initialized agent for a demo -- "
        "fix or replace the checkpoint.",
    ]
    existing = [n for n in find_checkpoints(checkpoints_dir) if n != checkpoint_path.name]
    if existing:
        lines.append(f"other checkpoints found in {checkpoints_dir}:")
        lines.extend(f"  {name}" for name in existing)
    remainder = [line for line in detail_lines[1:] if line.strip()]
    if remainder:
        lines.append(f"full {type(error).__name__} text follows:")
        lines.extend(f"  {line}" for line in remainder)
    return "\n".join(lines)


def load_greedy_policy(checkpoint_path: Path, checkpoints_dir: Path) -> Any:
    """Load ``checkpoint_path`` into a greedy (epsilon=0) policy, or refuse.

    Raises :class:`CheckpointError` — never returns a randomly-initialized
    network — when the file is missing OR when it exists but fails to load.
    Reuses :class:`eval.evaluate.DRQNGreedyPolicy` and its checkpoint-shape
    unwrapping (``eval.evaluate._load_drqn``) rather than re-implementing the
    "state_dict vs {'model': state_dict}" logic a second time; torch and that
    module are imported lazily so this file stays importable (and TC21 stays
    fast) without touching either when the checkpoint does not even exist.
    """
    if not checkpoint_path.is_file():
        raise CheckpointError(checkpoint_missing_message(checkpoint_path, checkpoints_dir))
    try:
        import torch  # lazy: only the live/loadable path needs it

        from eval.evaluate import DRQNGreedyPolicy, _load_drqn

        device = torch.device("cpu")
        net = _load_drqn(str(checkpoint_path), device)
        return DRQNGreedyPolicy(net, device=device)
    except Exception as exc:  # noqa: BLE001 — ANY load failure must refuse, never fall back.
        raise CheckpointError(
            checkpoint_unloadable_message(checkpoint_path, checkpoints_dir, exc)
        ) from exc


def build_bridge_argv(
    *,
    node: str = DEFAULT_NODE,
    run_js: Path = RUN_JS,
    mc_port: int,
    bridge_port: int,
    anchor: PadAnchor,
    learner_username: str,
    dummy_username: str,
    challenger_username: Optional[str],
    reset_request_path: Path,
    pad_index: int = PAD_INDEX,
) -> List[str]:
    """The exact argv that launches this pad's bridge in HUMAN opponent mode.

    Mirrors ``distributed.launcher.SubprocessArenaLauncher.spec_for`` (the
    training bridge argv — same flags, same order for the shared subset) plus
    the three exhibition-only flags: ``--opponent-mode human`` always,
    ``--challenger-username`` when pinned, and ``--reset-request-path`` so a
    player can arm the next match by typing ``reset`` in Minecraft chat instead
    of the operator alt-tabbing to a terminal.

    ``reset_request_path`` is a PARAMETER and not something derived here, and
    that is the whole guarantee: :func:`reset_request_path` is the one
    derivation, and :func:`run` hands this function the SAME value it goes on to
    poll. Deriving it a second time inside here would be a second
    implementation, free to drift from the first. (It also shadows that
    function's name for the body below, which nothing here needs.)

    Pure: no subprocess, no filesystem, no network, so tests/test_exhibition.py
    exercises it directly.
    """
    argv = [
        node,
        str(run_js),
        "--port",
        str(mc_port),
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
        "--opponent-mode",
        "human",
        "--reset-request-path",
        str(reset_request_path),
    ]
    if challenger_username is not None:
        argv.extend(["--challenger-username", challenger_username])
    return argv


# ---------------------------------------------------------------------------
# The reset trigger (T6): the request file, the human-side commands, and the
# ``--reset`` side of the CLI. Pure functions first, I/O behind seams after.
# ---------------------------------------------------------------------------


def reset_request_path(log_dir: Path) -> Path:
    """Where the reset request lives for a launcher using ``log_dir``.

    The ONE place this path is derived, so the ``--reset`` process and the
    launcher process cannot disagree about it — they are separate invocations
    of this module and have nothing else in common but their ``--log-dir``.

    ALWAYS ABSOLUTE, because there is now a THIRD process reading it: the bridge
    files the same request when a player types the keyword in game (``bridge/
    bot.js``), and :func:`run` spawns that bridge with ``cwd=REPO_ROOT`` while
    keeping whatever directory the operator launched from. A relative
    ``--log-dir`` would therefore have the launcher poll one file and the bridge
    write another while both agreed character-for-character about the string —
    an in-game reset that confirms in chat and never happens. Anchored to the
    caller's cwd rather than ``resolve()``-d, so a symlinked log dir (``/tmp`` is
    one on macOS) keeps the name the operator typed.
    """
    path = Path(log_dir) / RESET_REQUEST_FILENAME
    return path if path.is_absolute() else Path.cwd() / path


def reset_command_hint(log_dir: Path) -> str:
    """The exact command line the operator types to arm the next challenger.

    Carries ``--log-dir`` only when it is not the default: the reset process
    finds the request file by that flag alone, so a hint that omitted a
    non-default one would send the operator's request somewhere the launcher is
    not looking — a failure with no error message at either end.
    """
    hint = "python -m deploy.exhibition --reset"
    if Path(log_dir) != DEFAULT_LOG_DIR:
        hint = f"{hint} --log-dir {log_dir}"
    return hint


def _validate_challenger_username(challenger_username: str) -> None:
    """Refuse anything that is not a Minecraft username, loudly.

    Shared by every builder below because all of them interpolate the name into
    text a LEVEL-4 console executes: a name carrying a newline would be a second
    command of the attacker's choosing (``op <them>``), and a name carrying a
    space would silently retarget the command at a different selector.
    :class:`ExhibitionConfig` owns the username rule and is reused rather than
    re-implemented.
    """
    if not isinstance(challenger_username, str) or not challenger_username:
        raise ValueError(
            "challenger_username must be a non-empty Minecraft username, got "
            f"{challenger_username!r}"
        )
    ExhibitionConfig(challenger_username=challenger_username)  # raises on a bad name


def human_gear_commands(challenger_username: str) -> List[str]:
    """The five Paper CONSOLE lines that re-gear the human: one held iron sword,
    four WORN iron pieces.

    Mirrors the LEARNER's re-gear block line for line
    (``spawn_learner_pad.mcfunction``: ``$give $(learner) minecraft:iron_sword
    1`` followed by four ``$item replace entity $(learner) armor.<slot> with
    minecraft:iron_*``), because since M4 (T2, issue #33) BOTH bots are re-geared
    with a sword AND a full iron set on every reset, and the human has to fight
    the same fight the agent trained on. The dummy's template used to be the
    wrong model for gear -- it armed its opponent with nothing -- and since M4 it
    carries the identical kit; the one thing still NOT copied from it is its
    blanket ``$clear``.

    ARMOR IS ``item replace``, NEVER ``give``, AND THAT IS THE POINT OF THIS
    FUNCTION. ``give`` pushes a stack into the first free INVENTORY slot and
    equips nothing; nothing in Minecraft moves a piece from a player's bag onto
    their body. Four ``give``s would leave the challenger standing there HOLDING
    a full iron set at zero armor points, taking 6 a hit while the agent takes
    3.12 -- armored in the log, naked in the fight. ``item replace entity <name>
    armor.<slot> with <item>`` writes the equipment slot itself, so the piece is
    worn the instant the command returns.

    THE TWO HALVES ARE IDEMPOTENT FOR DIFFERENT REASONS, so do not "simplify"
    either into the other:

    * The four armor lines are idempotent BY CONSTRUCTION -- ``item replace``
      OVERWRITES the slot rather than appending to a bag, so an evening of
      resets leaves exactly one set, always at full durability, with no paired
      ``clear`` anywhere.
    * The sword is NOT independently idempotent: a bare ``give`` really would
      pile a sword per reset into the challenger's hotbar, so ``clear <name>
      minecraft:iron_sword`` immediately BEFORE it is what makes the pair
      idempotent and what lets a broken sword fix itself at the next reset. Keep
      the clear narrow: widening it to a blanket ``clear <name>`` (which the
      dummy's template does use) would empty a person's unrelated items, which
      is no part of heal-and-reposition.

    WHAT THIS CANNOT DO is make the challenger HOLD the sword. ``give`` chooses
    a free slot, not the selected one, and a person's hand is theirs; the sword
    is confirmed as OWNED by :func:`confirm_human_loadout`, never as held. Do not
    "fix" that by switching the sword to ``item replace ... weapon.mainhand``:
    that would break the line-for-line mirror of the datapack that the drift
    tests pin, and the datapack is what the agent trained against.
    """
    _validate_challenger_username(challenger_username)
    return [
        # Scoped clear THEN give -- see the docstring. The clear is what keeps
        # the SWORD idempotent across an evening of resets; the give is what
        # makes the fight symmetric.
        f"clear {challenger_username} {CHALLENGER_WEAPON}",
        f"give {challenger_username} {CHALLENGER_WEAPON} 1",
        # ...and `item replace` for the armor, which needs no clear because it
        # overwrites the equipment slot outright.
        *(
            f"item replace entity {challenger_username} armor.{slot} with {item}"
            for slot, item in CHALLENGER_ARMOR
        ),
    ]


def human_reset_commands(anchor: PadAnchor, challenger_username: str) -> List[str]:
    """Paper CONSOLE lines that heal, reposition and re-gear the human
    challenger.

    Two datapack templates, not one, because the human is the dummy's
    understudy in position and the learner's equal in gear:

    * HEAL AND REPOSITION mirror the dummy's template
      (``spawn_dummy_pad.mcfunction``) applied to a player instead of a bot:
      teleport to the opponent slot facing the learner, clear leftover effects,
      then restore health and food. Same amplifiers (``instant_health 1 9``,
      ``saturation 1 19``), same clear-then-give ORDER — an instant effect is
      applied on its first tick, so a clear issued afterwards can strip it
      before it ever lands, which is the bug that ordering exists to remove.
    * GEAR mirrors the LEARNER's re-gear block instead, and lives in
      :func:`human_gear_commands` — an iron sword by ``give``, four worn iron
      pieces by ``item replace``. The ONE thing not copied from the dummy is its
      blanket ``$clear``: taking a person's unrelated items is not part of
      heal-and-reposition.

    RETRACTED, AND IT WAS TRUE WHEN IT WAS WRITTEN: this docstring used to end
    "one sword and no armor — the checkpoint never trained against an armored
    opponent, so armor is out of scope". M4 arms BOTH bots (T2, issue #33), so
    the reasoning now runs the other way — the unarmored human is the mismatch,
    taking 6 a hit where the agent takes 3.12 through 15 armor points, in a
    matchup the policy has never played. Armor is in scope precisely BECAUSE of
    what the checkpoint trained against.

    NO LEADING SLASH. These go to the server console, whose commands are
    slash-free. ``formatHumanResetCommands`` (``bridge/bot.js``) is NOT a mirror
    of this list — it resets the LEARNER, and the learner's own gear arrives
    through the ``arena:spawn_learner_pad`` call inside it — but its output does
    keep a ``/`` because the bridge feeds it to ``bot.chat()``.

    Raises:
        ValueError: on a username that is not a Minecraft username, or a pad
            anchor that is not a pair of non-negative ints. Validation is not
            decoration here: this text is executed by a level-4 console, so a
            name containing a newline would be a second command of the
            attacker's choosing (``op <them>``). :class:`ExhibitionConfig` owns
            the username rule and is reused rather than re-implemented.
    """
    _validate_challenger_username(challenger_username)
    for label, value in (("x", anchor.x), ("z", anchor.z)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(
                f"pad anchor {label} must be a non-negative int, got {value!r}"
            )
    # `<n>.5` is written as a textual concatenation for the same reason the
    # datapack builds `$(x).5`: it is a block CENTRE, and the anchor is always a
    # non-negative int (asserted above), so the two forms agree character for
    # character.
    spawn_x = f"{anchor.x + CHALLENGER_SPAWN_DX}.5"
    spawn_z = f"{anchor.z}.5"
    return [
        f"tp {challenger_username} {spawn_x} {PAD_SPAWN_Y} {spawn_z} "
        f"{CHALLENGER_SPAWN_YAW} 0",
        f"effect clear {challenger_username}",
        f"effect give {challenger_username} minecraft:instant_health 1 9 true",
        f"effect give {challenger_username} minecraft:saturation 1 19 true",
        *human_gear_commands(challenger_username),
    ]


# ---------------------------------------------------------------------------
# READING THE CHALLENGER'S GEAR BACK OFF THE SERVER (AC9).
#
# The bots' loadout is proved by the bridge's fail-closed reset gate, which
# reads each bot's OWN connection (T3). A HUMAN HAS NO BOT CONNECTION, so that
# gate cannot see them at all — and worn armor would be invisible to an
# inventory check even if it could, because mineflayer's `inventory.items()`
# spans slots 9-44 and the armor slots are 5-8. A `$`-macro or a mistyped
# console line that silently did nothing would therefore leave a challenger
# fighting an armored agent in their own clothes, with a clean log.
#
# So the reset ASKS THE SERVER, on the only channel this process has: five
# `data get entity` reads down the same console pipe the gear went down, and a
# scan of Paper's own log for the answers. Paper runs console input in order, so
# the reads observe the world AFTER the re-gear rather than racing it.
#
# WHAT THIS IS, HONESTLY. It is a real server-authoritative read: the values
# come from the server's own serialization of the player, not from anything this
# process believes. It is NOT a structured reply channel — there is none; the
# launcher's console pipe is write-only (`send_paper_console_commands` confirms
# only that BYTES WERE WRITTEN) and the answers come back the same way an
# operator would see them, as text in `paper.log`. Three consequences, all of
# them stated in the log rather than papered over:
#
#   * a reply that is late, absent, or worded differently by some future Paper
#     build reads as COULD NOT CONFIRM, never as confirmed;
#   * "could not confirm" therefore does not prove the piece is missing, and the
#     failure message says exactly that;
#   * unlike the bots' gate this never refuses to play. A demo that will not
#     start because a helmet is unconfirmed is worse than one that starts and
#     says so loudly.
# ---------------------------------------------------------------------------

#: Slot number of each WORN piece inside a player's ``Inventory`` NBT list, read
#: out of the pinned jar rather than recalled: ``Inventory.save`` (
#: ``net/minecraft/world/entity/player/Inventory`` in
#: ``server/versions/1.21.1/paper-1.21.1.jar``) writes the main items at their
#: own index and then the armor list at ``index + 100``, and
#: ``net/minecraft/world/entity/EquipmentSlot`` gives that armor list the order
#: FEET=0, LEGS=1, CHEST=2, HEAD=3. Hence 100..103 = feet, legs, chest, head.
#: NOT ``ArmorItems``: that tag belongs to Mob, and a Player writes ``Inventory``
#: instead — a read against ``ArmorItems`` would silently find nothing on a
#: perfectly armored person.
ARMOR_NBT_SLOT: Dict[str, int] = {"feet": 100, "legs": 101, "chest": 102, "head": 103}

#: The console's rendering of ``commands.data.entity.query``, which the pinned
#: jar's ``assets/minecraft/lang/en_us.json`` gives as
#: ``"%s has the following entity data: %s"``. Matched as a SUBSTRING and
#: without pinning how the value is quoted, so the check survives cosmetic
#: differences and fails closed (as "could not confirm") on real ones.
DATA_QUERY_PHRASE = "has the following entity data:"

#: How long to wait for Paper to answer the five reads. The JVM runs console
#: input on the next tick and log4j flushes asynchronously, so an immediate read
#: of the log is expected to come up empty; five seconds is far past both and
#: still far short of an operator noticing a pause between matches.
HUMAN_LOADOUT_READ_TIMEOUT_SECONDS = 5.0
HUMAN_LOADOUT_READ_POLL_SECONDS = 0.25


def _human_loadout_reads(challenger_username: str) -> List[Tuple[str, str, str]]:
    """``(label, console command, item id the server must report)`` per piece.

    Every expected value is a DISTINCT item id, which is what lets one scan of
    the log resolve all five reads without depending on the order the replies
    arrive in.
    """
    _validate_challenger_username(challenger_username)
    reads = [
        (
            f"armor.{slot}",
            f"data get entity {challenger_username} "
            f"Inventory[{{Slot:{ARMOR_NBT_SLOT[slot]}b}}].id",
            item,
        )
        for slot, item in CHALLENGER_ARMOR
    ]
    # The sword is read by CONTENT, not by slot, and that asymmetry is
    # deliberate: `give` drops it in the first FREE inventory slot, which for a
    # person carrying their own things is not a slot this process can predict.
    # So this asks "does the server say they own an iron sword", which is the
    # strongest true claim available -- it is NOT a proof they are holding it,
    # and nothing here or in the log may say that it is.
    reads.append(
        (
            "the sword",
            f"data get entity {challenger_username} "
            f'Inventory[{{id:"{CHALLENGER_WEAPON}"}}].id',
            CHALLENGER_WEAPON,
        )
    )
    return reads


def human_loadout_read_commands(challenger_username: str) -> List[str]:
    """The five server-authoritative read commands, in gear order.

    Console lines, so no leading slash — same channel and same rules as
    :func:`human_reset_commands`. Split out from the confirmation itself so the
    ``--no-paper-console`` path can print exactly what an opped operator should
    type by hand.
    """
    return [command for _label, command, _item in _human_loadout_reads(challenger_username)]


def take_reset_request(path: Path) -> bool:
    """Consume a pending reset request; True iff there was one.

    The take is the ``unlink`` itself rather than an ``exists()`` followed by
    one, so the request is consumed exactly once even if it is filed at the
    instant it is read. An unexpected ``OSError`` (a permission problem on the
    log dir, say) deliberately PROPAGATES: returning False there would spin the
    idle loop forever on a reset that can never arrive, silently, which is the
    project's signature failure mode.
    """
    try:
        Path(path).unlink()
    except FileNotFoundError:
        return False
    return True


def drain_reset_request(path: Path, reason: str, *, log: Callable[[str], None]) -> bool:
    """Discard a request that must not arm a match, and say so. True if one was
    discarded.

    Two callers, one rule — a match only ever starts from a request the operator
    filed while the launcher was IDLE:

      * at startup, a file left over from an earlier launch;
      * at the end of every match, a file filed while that match was still
        running. Honoring that one would make the death the proximate cause of
        the restart, which is exactly what AC4 forbids.

    Never silent: a discarded request is a command the operator typed and is
    waiting on.
    """
    if not take_reset_request(path):
        return False
    log(f"discarded a reset request {reason}.")
    return True


def wait_for_reset_request(
    path: Path,
    *,
    sleep: Callable[[float], None] = time.sleep,
    poll_seconds: float = RESET_POLL_SECONDS,
) -> None:
    """Block until a reset request appears, then consume it and return.

    The only other way out is the ``KeyboardInterrupt`` that ends the
    exhibition, which unwinds into :func:`run`'s handler exactly as it did when
    this was an unconditional idle loop.
    """
    while not take_reset_request(path):
        sleep(poll_seconds)


def reset_dir_missing_message(log_dir: Path) -> str:
    """Refusal text for ``--reset`` with no launcher log dir to file into."""
    return (
        f"no exhibition launcher is using {log_dir}: that directory does not "
        "exist, so there is nothing here to reset.\n"
        "the reset command does NOT start anything and never connects to the "
        "bridge (the bridge accepts exactly ONE TCP client, and a second "
        "connect would evict the running agent) -- it only hands a request to "
        "an already-running launcher.\n"
        "start the exhibition first with `python -m deploy.exhibition`, or pass "
        "the same --log-dir that launcher was given."
    )


def reset_already_armed_message(request_path: Path) -> str:
    """Refusal text for a request that is still sitting unconsumed."""
    return (
        f"a reset is already armed and has not been consumed: {request_path}\n"
        "an idle launcher picks one up within about a second, so either it is "
        "still playing the current match (the request will be DISCARDED when "
        "that match ends -- a death must never restart the match by itself; "
        "run this again once it has), or no launcher is running at all.\n"
        "nothing was changed."
    )


def reset_armed_message(request_path: Path) -> str:
    """The confirmation ``--reset`` prints once the request is filed."""
    return (
        f"reset armed: {request_path}\n"
        "the running launcher will heal and reposition both fighters, release "
        "the challenger slot for the next person, and play ONE more match. It "
        "does not restart by itself after that -- one reset command, one "
        "match.\n"
        "if the launcher is mid-match right now, this request is discarded when "
        "that match ends; run it again once it has."
    )


def reset_mode_conflicts(args: Any, defaults: Any) -> List[str]:
    """Option flags passed alongside ``--reset`` that ``--reset`` cannot honor.

    Everything except ``--log-dir`` is decided when the LAUNCHER starts: the
    checkpoint is already loaded, the ports are already bound, the challenger is
    already pinned into the bridge's argv. A ``--reset`` that quietly ignored
    ``--challenger-username`` would look to the operator like it had swapped
    the challenger, which is precisely the mistake it is worth an error to
    prevent. Compares the parsed namespace against the parser's own defaults, so
    a new flag is covered the day it is added rather than the day someone
    remembers to list it here.
    """
    conflicts = []
    for name, default in sorted(vars(defaults).items()):
        if name in ("reset", "log_dir"):
            continue
        if getattr(args, name, default) != default:
            conflicts.append(_flag_for(name))
    return conflicts


def _flag_for(dest: str) -> str:
    """The command-line spelling of an argparse ``dest``.

    ``--no-paper-console`` stores into ``paper_console``, so the mechanical
    ``dest.replace("_", "-")`` would name a flag that does not exist and send an
    operator looking for it.
    """
    if dest == "paper_console":
        return "--no-paper-console"
    return "--" + dest.replace("_", "-")


def reset_conflicts_message(conflicts: Sequence[str]) -> str:
    """Refusal text for :func:`reset_mode_conflicts`' findings."""
    lines = [
        "--reset takes only --log-dir. These were also passed and it cannot "
        "honor them:"
    ]
    lines.extend(f"  {flag}" for flag in conflicts)
    lines.append(
        "they are all decided when the LAUNCHER starts (the checkpoint is "
        "loaded, the ports are bound, the challenger is pinned into the "
        "bridge's argv), so changing one means restarting the exhibition."
    )
    lines.append("nothing was changed.")
    return "\n".join(lines)


def request_reset(
    log_dir: Path,
    *,
    log: Callable[[str], None] = _ascii_log,
    now: Callable[[], str] = lambda: time.strftime("%Y-%m-%d %H:%M:%S"),
) -> int:
    """File one reset request for a running launcher. Returns an exit code.

    Starts nothing, connects to nothing, loads no checkpoint and does not create
    the log dir — a missing dir is the evidence that no launcher is using it,
    and creating one would destroy that evidence.
    """
    request_path = reset_request_path(log_dir)
    if not Path(log_dir).is_dir():
        log(reset_dir_missing_message(log_dir))
        return 1
    try:
        # "x": exclusive-create. The refusal for an already-armed reset is the
        # SAME operation as the arming itself, so two operators racing cannot
        # both believe they armed it.
        with open(request_path, "x", encoding="ascii") as handle:
            handle.write(f"reset requested {now()}\n")
    except FileExistsError:
        log(reset_already_armed_message(request_path))
        return 1
    except OSError as exc:
        log(f"could not file the reset request at {request_path}: {exc}")
        return 1
    log(reset_armed_message(request_path))
    return 0


def send_paper_console_commands(
    proc: Any,
    commands: Sequence[str],
    *,
    log: Callable[[str], None] = _ascii_log,
    what: str = "heal/reposition the human",
) -> bool:
    """Write ``commands`` into Paper's console. True iff all of them were sent.

    ``server/setup/start.sh`` ``exec``s the JVM, so the pipe :func:`run` opens
    on the start script IS the server console's stdin. Console lines carry no
    leading slash.

    WRITE-ONLY, AND THAT IS THE WHOLE OF WHAT A ``True`` HERE MEANS: the bytes
    reached the pipe. It says nothing about whether Paper parsed them, whether
    the commands succeeded, or whether the world changed. Anything that needs to
    know THAT has to read the server's answer back out of ``paper.log`` — see
    :func:`confirm_human_loadout`, which is the only caller that does.

    ``what`` names the job for the failure message, because "the commands did
    not run" is only actionable if the operator is told which commands.

    Best-effort and non-fatal: Paper dying mid-exhibition surfaces here as a
    ``BrokenPipeError`` (an ``OSError``) and a launcher that aborted a match
    over a failed heal would be worse than one that plays it unhealed. Whatever
    happens, the commands are echoed to the exhibition log — an operator who is
    opped can run them by hand, and a channel that quietly did nothing is the
    failure mode this repo keeps paying for.
    """
    stream = getattr(proc, "stdin", None)
    if stream is None:
        _log_unrun_commands(
            commands,
            "no console pipe to Paper (--no-paper-console, or a server this "
            "launcher did not start)",
            log=log,
            what=what,
        )
        return False
    try:
        for command in commands:
            stream.write(f"{command}\n".encode("ascii"))
        stream.flush()
    except (OSError, ValueError) as exc:  # broken pipe, closed stream
        _log_unrun_commands(
            commands,
            f"the Paper console rejected the write ({exc!r})",
            log=log,
            what=what,
        )
        return False
    for command in commands:
        log(f"  console> {command}")
    return True


def _log_unrun_commands(
    commands: Sequence[str],
    why: str,
    *,
    log: Callable[[str], None],
    what: str = "heal/reposition the human",
) -> None:
    log(f"could not {what}: {why}.")
    log("these console commands did NOT run:")
    for command in commands:
        log(f"  {command}")


def _close_paper_console(proc: Any) -> None:
    """Close Paper's console pipe. Silent, and cannot raise.

    Last action of the teardown chain, after the JVM has already been stopped,
    so there is nothing left to say to it. Explicit rather than left to garbage
    collection because a buffered writer finalized against a dead child prints
    "Exception ignored in ..." at interpreter exit — noise on the operator's
    terminal that reads like a crash in a launcher that shut down correctly.
    """
    try:
        stream = getattr(proc, "stdin", None)
        if stream is not None:
            stream.close()
    except BaseException:  # noqa: BLE001 — teardown's last link must not raise.
        pass


# ---------------------------------------------------------------------------
# SIGTERM — the other way an exhibition is stopped (see the module docstring's
# TEARDOWN section). Ctrl-C already reaches the teardown chain because CPython
# installs its own SIGINT handler; SIGTERM has to be given one.
# ---------------------------------------------------------------------------


def _sigterm_handler(signum: int, _frame: Any) -> None:
    """Turn SIGTERM into :class:`ExhibitionTerminated`, and do NOTHING else.

    A signal handler runs in the main thread between two bytecodes, with the
    interpreter left in whatever state the interrupted line was mid-way
    through — possibly a ``write`` on the very console pipe the teardown is
    about to close, possibly ``subprocess``'s own bookkeeping for a child it is
    about to stop. So this stops the children by RAISING, which unwinds into
    :func:`run`'s existing ``finally`` chain and lets the normal path do the
    work: no process control, no I/O and no logging happen in here.
    """
    raise ExhibitionTerminated(f"terminated by SIGTERM (signal {signum})")


def install_sigterm_handler(
    *, log: Callable[[str], None] = _ascii_log
) -> Optional[Callable[[], None]]:
    """Route SIGTERM into :func:`run`'s teardown. Returns a callable that puts
    the inherited disposition back, or ``None`` when no handler was installed.

    Reuses the teardown rather than duplicating it — the handler only raises,
    and the ``finally`` chain that Ctrl-C already unwinds through stops both
    children. A second teardown written for this path could drift out of step
    with that one, and the failure mode of the drift is a stranded child.

    Two dispositions are deliberately left ALONE:

    * ``SIG_IGN``. A process that inherited "SIGTERM ignored" got it from a
      parent that meant it (POSIX: ``nohup``, a job-control shell backgrounding
      a child, a supervisor that manages its tree some other way), and that is
      the same convention under which CPython declines to install a SIGINT
      handler over an inherited ``SIG_IGN`` — the reason a ``nohup``-ed
      launcher looks like it ignores Ctrl-C. Overriding it would make this
      process killable by a signal something above it masked on purpose.
    * Any interpreter that will not take a handler: ``signal.signal`` raises
      ``ValueError`` outside the main thread, and a platform with no SIGTERM
      raises ``AttributeError`` on the constant. A launcher running in a worker
      thread is otherwise completely functional, and crashing it over a
      teardown nicety would be the worse trade.

    Neither is silent. A machine where ``kill`` will orphan the Paper JVM on
    the Minecraft port is exactly what the operator needs told BEFORE the
    exhibition, not discovered at the next launch's port gate.
    """
    try:
        previous = signal.getsignal(signal.SIGTERM)
    except (AttributeError, ValueError, OSError) as exc:  # no usable SIGTERM here
        log(
            f"this platform has no usable SIGTERM ({exc!r}); end the exhibition "
            "with Ctrl-C."
        )
        return None
    if previous is signal.SIG_IGN:
        log(
            "SIGTERM is inherited as IGNORED, so this launcher leaves it that "
            "way -- something above it (nohup, a job-control shell, a "
            "supervisor) masked it on purpose. `kill <pid>` will NOT stop this "
            "process; end the exhibition with Ctrl-C, and if you must `kill -9` "
            "it, stop the Paper JVM and the bridge by hand afterwards."
        )
        return None
    try:
        signal.signal(signal.SIGTERM, _sigterm_handler)
    except (ValueError, OSError, RuntimeError) as exc:
        log(
            f"could not install a SIGTERM handler ({exc!r}), so `kill <pid>` "
            "would orphan the Paper JVM on the Minecraft port. End the "
            "exhibition with Ctrl-C instead."
        )
        return None

    def restore() -> None:
        """Put the inherited disposition back. Cannot raise — it is the last
        link of a chain whose entire purpose is that nothing in it strands a
        child."""
        try:
            # `getsignal` reports None for a handler installed from C, which
            # `signal.signal` will not accept back; SIG_DFL is the honest
            # approximation, and this process is on its way out regardless.
            signal.signal(
                signal.SIGTERM, signal.SIG_DFL if previous is None else previous
            )
        except BaseException:  # noqa: BLE001 — teardown's last link must not raise.
            pass

    return restore


# ---------------------------------------------------------------------------
# Small side-effecting helpers, each taking its I/O behind a keyword seam.
# ---------------------------------------------------------------------------


def _detect_lan_ip() -> Optional[str]:
    """Best-effort LOCAL network IP for the join message.

    Non-fatal by construction: any failure (no network, sandboxed CI, no
    default route) returns ``None`` and the caller falls back to a manual-
    lookup hint rather than aborting the whole launch over a cosmetic detail.
    Connecting a UDP socket sends no packet — this only asks the OS which
    local interface WOULD carry traffic to that address.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.settimeout(1.0)
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except OSError:
        return None


def _print_join_instructions(
    mc_port: int, anchor: PadAnchor, *, log: Callable[[str], None]
) -> None:
    """AC5: print the LAN IP/port and the pad coordinates so a classmate can
    join and find the arena without being talked through it."""
    lan_ip = _detect_lan_ip()
    log("=" * 60)
    if lan_ip is not None:
        log(f"JOIN AT: {lan_ip}:{mc_port}  (same LAN, offline-mode -- any username)")
    else:
        log(
            f"JOIN AT: <this machine's LAN IP>:{mc_port}  (could not auto-detect "
            "it -- try `ipconfig getifaddr en0` or System Settings > Wi-Fi)"
        )
    log(
        f"ARENA: pad {PAD_INDEX}, anchor ({anchor.x}, {anchor.z}) -- the world "
        "spawn, so a fresh join should land right there."
    )
    log("=" * 60)


def wait_for_port(
    host: str,
    port: int,
    *,
    timeout: float,
    label: str,
    process: subprocess.Popen,
    port_probe: Callable[[str, int], bool] = _tcp_port_open,
    sleep: Callable[[float], None] = time.sleep,
    poll_seconds: float = PORT_POLL_SECONDS,
    log: Callable[[str], None] = _ascii_log,
) -> bool:
    """Poll until ``host:port`` accepts a connection, ``process`` exits, or
    ``timeout`` elapses.

    Mirrors ``start-pads.sh``'s ``wait_for_port``: a successful connect is
    only trusted while the process we spawned is still alive — a listener
    answering after our child already exited belongs to someone else, not to
    us, and reporting it "up" would supervise a phantom.
    """
    waited = 0.0
    while True:
        if port_probe(host, port):
            exit_code = process.poll()
            if exit_code is not None:
                log(
                    f"{label}: port {port} is answering, but the process we "
                    f"started already exited (code {exit_code}). That listener "
                    "is not ours."
                )
                return False
            return True
        exit_code = process.poll()
        if exit_code is not None:
            log(f"{label}: process exited (code {exit_code}) before port {port} opened.")
            return False
        if waited >= timeout:
            log(f"{label}: port {port} did not open within {timeout:.0f}s.")
            return False
        sleep(poll_seconds)
        waited += poll_seconds


def _dump_log_tail(
    path: Path, *, lines: int = LOG_TAIL_LINES, log: Callable[[str], None] = _ascii_log
) -> None:
    if not path.is_file():
        log(f"(no log at {path})")
        return
    try:
        text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        log(f"(could not read {path}: {exc})")
        return
    tail = text[-lines:]
    log(f"----- last {len(tail)} line(s) of {path} -----")
    for line in tail:
        log(line)
    log(f"----- end {path} -----")


def _force_kill(
    proc: subprocess.Popen, label: str, *, log: Callable[[str], None] = _ascii_log
) -> None:
    """Last-resort SIGKILL that can neither raise nor block.

    Reached only when :func:`_stop_process`'s graceful path was cut short by a
    ``BaseException`` — in practice a second Ctrl-C landing inside the grace
    wait. There is deliberately no ``wait()`` afterwards: waiting is precisely
    the interruptible call that just got interrupted, and the child is reaped
    by the OS once this supervisor exits.
    """
    try:
        if proc.poll() is None:
            log(f"{label} (pid {proc.pid}): interrupted mid-stop; killing outright.")
            proc.kill()
    except BaseException:  # noqa: BLE001 — nothing left to try; must never re-raise.
        pass


def _stop_process(
    proc: subprocess.Popen,
    label: str,
    *,
    grace_seconds: float = STOP_GRACE_SECONDS,
    log: Callable[[str], None] = _ascii_log,
) -> None:
    """Terminate one process gracefully, escalating to kill. Best-effort and
    safe to call on an already-exited process.

    Catches ``BaseException``, not ``Exception``. ``KeyboardInterrupt`` derives
    from ``BaseException``, and the ``wait(timeout=grace_seconds)`` below is
    exactly where an operator's SECOND Ctrl-C lands — the first one is what
    started this teardown, and double-tapping through a 20-second silence is
    the normal human response, not an edge case. Letting it propagate would
    abort the rest of the caller's teardown chain and strand the Paper JVM on
    the Minecraft port, which then makes the next launch refuse on its own
    mc-port gate. So the interrupt is absorbed here and the child is killed
    outright instead of waited on.
    """
    try:
        if proc.poll() is not None:
            return
        log(f"stopping {label} (pid {proc.pid}).")
        proc.terminate()
        try:
            proc.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired:
            log(f"{label} (pid {proc.pid}) did not exit; killing.")
            proc.kill()
            proc.wait()
    except BaseException as exc:  # noqa: BLE001 — teardown must not raise past this point.
        # `!r`, not `!s`: a bare KeyboardInterrupt stringifies to "" and would
        # otherwise print as an error message with nothing after the colon.
        log(f"error stopping {label} (ignored): {exc!r}")
        _force_kill(proc, label, log=log)


def _close_env(env: Any, *, log: Callable[[str], None] = _ascii_log) -> None:
    """Close the agent's bridge connection, absorbing anything it raises.

    ``BaseException``-proof for the same reason as :func:`_stop_process`, and
    it matters more here: this is the FIRST link of the teardown chain, so an
    escaping ``KeyboardInterrupt`` would strand BOTH child processes rather
    than only Paper.
    """
    try:
        env.close()
    except BaseException as exc:  # noqa: BLE001 — teardown must not raise.
        log(f"error closing the env (continuing teardown): {exc!r}")


def _paper_env(xms: Optional[str], xmx: Optional[str]) -> Optional[Dict[str, str]]:
    """Env dict for start.sh's heap override, or None to inherit unchanged."""
    if xms is None and xmx is None:
        return None
    import os

    env = dict(os.environ)
    if xms is not None:
        env["XMS"] = xms
    if xmx is not None:
        env["XMX"] = xmx
    return env


#: THE SHIELD'S BOUND (see :func:`_reflex_shield_action`). The shield may fire
#: on at most this many CONSECUTIVE decision steps before it must hand control
#: back to the policy. Without a bound the override latches: once the streak
#: passes the threshold every later blind step is also past it, so the policy's
#: choice is discarded forever and the agent's yaw freezes. That is not
#: hypothetical -- it was measured firing on 190 of 196 consecutive steps, yaw
#: frozen for 90 seconds.
REFLEX_MAX_CONSECUTIVE_OVERRIDES = 3

#: ...and this many consecutive steps of pure policy control follow each capped
#: burst, before the shield is allowed to fire again. A COOL-OFF rather than a
#: one-shot cap on purpose: a hard cap would disarm the shield for the rest of
#: an arbitrarily long blind stretch, which is the opposite failure. Cycling
#: keeps the crutch available while guaranteeing the policy drives the majority
#: of steps (8 of every 11) and is never locked out for more than
#: ``REFLEX_MAX_CONSECUTIVE_OVERRIDES`` steps in a row.
REFLEX_COOLOFF_STEPS = 8


def _reflex_shield_action(
    obs: Any,
    action: int,
    blind_streak: int,
    *,
    reflex_blind_steps: int,
) -> Tuple[int, int, bool]:
    """T7: decide the macro actually sent for one decision step.

    Pure and stateless across calls — :func:`play_one_match` threads
    ``blind_streak`` through the loop itself, so this function holds nothing
    that could leak between matches, episodes, or callers.

    ``reflex_blind_steps <= 0`` disables the shield outright and — this is the
    point, not an optimization — the function does not even LOOK at ``obs`` in
    that case. That is what lets :func:`play_one_match` default the shield off
    for every caller that does not explicitly opt in (every training loop; the
    unit tests that hand it a bare placeholder instead of a real observation
    vector), with no dependency on ``obs`` being observation-shaped at all
    unless the shield is actually armed.

    When armed, ``visible`` is read through the FROZEN
    ``env.observation_spec.Obs.VISIBLE`` accessor — never a hard-coded index —
    so a future observation-layout change breaks loudly here instead of
    silently reading the wrong float.

    Args:
        obs: This step's ``(OBS_DIM,)`` observation — the SAME one just handed
            to the policy, so "blind" means exactly what the policy saw.
        action: The macro the policy chose for ``obs``.
        blind_streak: Consecutive PRIOR decision steps (not counting this one)
            whose observation had ``visible == 0``.
        reflex_blind_steps: ``ExhibitionConfig.reflex_blind_steps`` — the
            number of consecutive blind steps that must elapse before the
            override fires.

    THE BOUND. Once armed, the shield runs a duty cycle rather than latching:
    it fires for at most ``REFLEX_MAX_CONSECUTIVE_OVERRIDES`` consecutive steps,
    then hands the policy ``REFLEX_COOLOFF_STEPS`` steps of uncontested control,
    then repeats for as long as the agent stays blind. So the policy can never
    be locked out for an unbounded run of steps, which is what previously froze
    the agent's yaw for 90 seconds. The FIRST fire is unchanged — still exactly
    at ``reflex_blind_steps`` consecutive blind steps — so the shield's armed
    behaviour is the same one T7 shipped; only its tail is bounded.

    Returns:
        ``(actual_action, new_blind_streak, fired)``. ``new_blind_streak`` is
        always the correct ``blind_streak`` to pass on the NEXT call, whether
        or not the shield fired this time. ``fired`` is True only when this
        call itself substituted ``TURN_TO_LAST_SEEN`` for the policy's choice
        — never merely because the policy happened to choose it on its own.
    """
    if reflex_blind_steps <= 0:
        return action, 0, False
    blind_this_step = float(obs[Obs.VISIBLE]) == 0.0
    if not blind_this_step:
        # The moment the opponent is visible again, the streak resets to zero
        # — including on the very step visibility returns, per spec.
        return action, 0, False
    new_streak = blind_streak + 1
    if new_streak < reflex_blind_steps:
        return action, new_streak, False
    # ARMED. Which of the two phases of the duty cycle this step falls in is a
    # pure function of the streak -- no extra state to thread, so this function
    # stays stateless across calls exactly as documented above.
    #
    # `fire_index` counts armed steps from 0 (the step that first reaches the
    # threshold). The cycle is REFLEX_MAX_CONSECUTIVE_OVERRIDES firing steps
    # followed by REFLEX_COOLOFF_STEPS steps the policy owns outright, repeating
    # for as long as the agent stays blind.
    fire_index = new_streak - reflex_blind_steps
    period = REFLEX_MAX_CONSECUTIVE_OVERRIDES + REFLEX_COOLOFF_STEPS
    if fire_index % period < REFLEX_MAX_CONSECUTIVE_OVERRIDES:
        return int(Macro.TURN_TO_LAST_SEEN), new_streak, True
    # Cool-off: still blind, still counting, but the policy's macro goes through
    # untouched. This is the branch that makes the override impossible to latch.
    return action, new_streak, False


def play_one_match(
    env: Any,
    policy: Any,
    *,
    reflex_blind_steps: int = 0,
    log: Callable[[str], None] = _ascii_log,
) -> str:
    """Play exactly one match to completion, greedily, with no learning.

    Resets once, then steps the greedy policy until the env reports ``done`` —
    which under ``ExhibitionConfig.no_timeout`` only happens on an actual
    death (agent or human, AC4), never a step-count timeout. While no
    challenger has claimed the pad, the opponent block of the observation is
    zeroed (the bridge holds IDLE and keeps ``state`` flowing rather than
    carrying a status string — there is no wire slot for one); the greedy
    policy simply acts on that, same as it would on any other observation.
    There is nothing exhibition-specific to special-case here.

    Args:
        env: The env to play (real ``MCPvPEnv`` in production).
        policy: The greedy policy (``reset()``/``act(obs)``).
        reflex_blind_steps: T7's reflex shield (``ExhibitionConfig
            .reflex_blind_steps``). After this many CONSECUTIVE decision steps
            whose observation has ``visible == 0``, the policy's chosen macro
            is overridden with ``Macro.TURN_TO_LAST_SEEN`` — still exactly one
            ``env.step()`` call for the decision, just with a different macro
            in it. The policy is asked for its action on every step regardless
            (so any recurrent state it carries keeps advancing); only the
            macro actually sent may change. The streak resets to zero the
            instant the opponent is visible again, and while it is armed the
            override is BOUNDED — at most
            ``REFLEX_MAX_CONSECUTIVE_OVERRIDES`` consecutive steps before
            ``REFLEX_COOLOFF_STEPS`` steps of pure policy control — so it can
            never latch on for the rest of a match. Defaults to ``0``, which
            disables the shield: only :func:`run` (below) passes the nonzero
            value from ``ExhibitionConfig``, so a caller must opt in
            explicitly — see the module docstring's REFLEX SHIELD section for
            why that makes it structurally impossible for training to enable
            this by accident.
        log: Where match/shield summary lines go.

    Returns a short human-readable outcome string.
    """
    if hasattr(policy, "reset"):
        policy.reset()
    obs = env.reset()
    done = False
    info: Dict[str, Any] = {}
    steps = 0
    blind_streak = 0
    reflex_overrides = 0
    while not done:
        action = policy.act(obs)
        action, blind_streak, fired = _reflex_shield_action(
            obs, action, blind_streak, reflex_blind_steps=reflex_blind_steps
        )
        if fired:
            reflex_overrides += 1
        obs, _reward, done, info = env.step(action)
        steps += 1
    if reflex_overrides:
        log(
            f"reflex shield overrode the action on {reflex_overrides}/{steps} "
            f"decision step(s) (reflex_blind_steps={reflex_blind_steps})."
        )
    if info.get("lost"):
        result = "AGENT LOSS (agent died)"
    elif info.get("won"):
        result = "AGENT WIN (opponent died)"
    else:
        # Exhibition mode never truncates (max_episode_steps=None), so this is
        # unreachable in a real run; kept as an honest label instead of a
        # crash if `env` ever disagrees.
        result = f"episode ended without a recorded death: {info!r}"
    log(f"match finished after {steps} decision step(s): {result}")
    return result


def _log_size(path: Path) -> int:
    """Byte length of ``path``, or 0 if it cannot be stat-ed.

    Used as the START of the region :func:`confirm_human_loadout` scans, so a
    missing or unreadable log degrades to "scan from the beginning and confirm
    nothing" rather than raising inside a best-effort reset.
    """
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _log_text_since(path: Path, offset: int) -> str:
    """Whatever has been appended to ``path`` since byte ``offset``.

    Binary read plus a replacing decode, deliberately: this is a live server log
    being appended to by another process, so a read can land mid-way through a
    multi-byte character, and a UnicodeDecodeError inside a heal would take the
    reset down over a cosmetic problem.
    """
    try:
        with open(path, "rb") as handle:
            handle.seek(offset)
            return handle.read().decode("utf-8", "replace")
    except OSError:
        return ""


def _confirmed_loadout_ids(
    text: str, challenger_username: str, expected: Sequence[str]
) -> set:
    """The item ids ``text`` shows the SERVER reporting for ``challenger_username``.

    Only lines that carry both the challenger's name and the console's
    ``data get`` reply phrase count, and the id has to appear on the VALUE side
    of that phrase. That is what keeps a command echo, a chat message, or a
    reply about some other player from confirming a piece nobody is wearing —
    the failure mode this whole check exists to prevent, arriving through the
    check itself.
    """
    seen = set()
    for line in text.splitlines():
        head, phrase, value = line.partition(DATA_QUERY_PHRASE)
        if not phrase or challenger_username not in head:
            continue
        for item in expected:
            if item in value:
                seen.add(item)
    return seen


def confirm_human_loadout(
    paper_proc: Any,
    paper_log: Optional[Path],
    challenger_username: str,
    *,
    log: Callable[[str], None] = _ascii_log,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
    timeout: Optional[float] = None,
) -> bool:
    """Ask the server what the challenger is actually wearing, and check the
    answer. True iff all five pieces came back confirmed.

    See "READING THE CHALLENGER'S GEAR BACK OFF THE SERVER" above this module's
    read-command builders for why this exists (a human has no bot connection,
    and the bots' fail-closed gate can therefore not see them) and for exactly
    how much it does and does not prove.

    THE ORDER IS THE CORRECTNESS ARGUMENT. Paper runs console input in the order
    it arrives, so reads written after the gear lines observe the world AFTER the
    re-gear rather than racing it — and the scan window opens at the log's
    CURRENT length, so a reply from an earlier reset can never be mistaken for
    this one's.

    NEVER FATAL, AND NEVER SILENT. An unconfirmed piece is logged loudly with
    the commands to check it by hand, and the match is played anyway; the
    message says "could not confirm", not "is missing", because a lost or
    late reply looks exactly the same from here.

    Args:
        paper_proc: the Paper child, whose ``stdin`` is the console pipe.
        paper_log: where Paper's stdout is being captured. ``None`` (or a path
            that cannot be read) means there is nowhere for an answer to arrive,
            which is reported rather than treated as a pass.
        timeout: seconds to wait for the replies; defaults to
            :data:`HUMAN_LOADOUT_READ_TIMEOUT_SECONDS`. Zero makes exactly one
            attempt, which is what an offline test wants.
    """
    reads = _human_loadout_reads(challenger_username)
    budget = HUMAN_LOADOUT_READ_TIMEOUT_SECONDS if timeout is None else timeout
    if paper_log is None:
        log(
            f"CANNOT CONFIRM {challenger_username}'s gear: this launcher is not "
            "capturing Paper's log, so there is nowhere for the server's answer "
            "to arrive. The gear commands were still sent."
        )
        return False
    paper_log = Path(paper_log)
    # Opened BEFORE the reads are written: everything the server says from here
    # on is an answer to THIS reset, never a leftover from the last one.
    offset = _log_size(paper_log)
    if not send_paper_console_commands(
        paper_proc,
        [command for _label, command, _item in reads],
        log=log,
        what=f"read {challenger_username}'s gear back off the server",
    ):
        return False
    expected = [item for _label, _command, item in reads]
    deadline = monotonic() + max(0.0, budget)
    while True:
        confirmed = _confirmed_loadout_ids(
            _log_text_since(paper_log, offset), challenger_username, expected
        )
        if len(confirmed) == len(expected) or monotonic() >= deadline:
            break
        sleep(HUMAN_LOADOUT_READ_POLL_SECONDS)
    unconfirmed = [
        (label, command, item) for label, command, item in reads if item not in confirmed
    ]
    if not unconfirmed:
        log(
            f"server-authoritative read: {challenger_username} has all four iron "
            "armor pieces WORN and an iron sword in their inventory. (It does "
            "NOT prove the sword is in their HAND -- `give` picks a free slot, "
            "and which slot a person selects is theirs to choose.)"
        )
        return True
    log(
        f"COULD NOT CONFIRM {len(unconfirmed)} of {len(expected)} gear slot(s) "
        f"for {challenger_username}: "
        + ", ".join(f"{label} ({item})" for label, _command, item in unconfirmed)
        + "."
    )
    log(
        f"Nothing in {paper_log} confirmed those within {budget:g}s. That is NOT "
        "proof the gear is missing: a piece that silently failed to equip and a "
        "reply that was late, lost or worded differently look identical from "
        "here, and this does not claim to tell them apart. It IS the case where "
        "a challenger fights an armored agent unarmored, so check it -- from an "
        "opped account, or the server console:"
    )
    for _label, command, _item in unconfirmed:
        log(f"  {command}")
    log(
        "Then re-run the reset (type `reset` in Minecraft chat) to re-issue the "
        "gear commands -- `item replace` overwrites, so repeating it is safe."
    )
    return False


def _reset_human_side(
    paper_proc: Any,
    anchor: PadAnchor,
    challenger_username: Optional[str],
    *,
    paper_console: bool,
    paper_log: Optional[Path] = None,
    log: Callable[[str], None] = _ascii_log,
) -> bool:
    """Heal, reposition and re-gear the human before the next match, then read
    the gear back off the server. True iff the commands were sent.

    The learner's half of the reset is the bridge's job and happens inside
    :func:`play_one_match`'s ``env.reset()`` — sword, armor and all, via the
    datapack, and proved by that side's fail-closed gate. This is the half
    nothing else covers. Never fatal — an unhealed, unarmored challenger is a
    worse match, a crashed launcher is no match at all — but never silent
    either: every path that cannot run these says so and prints what would have
    run.

    The return value tracks the SEND, not the confirmation: a reset that got its
    commands onto the console did its job even when the read-back cannot see the
    answer, and :func:`confirm_human_loadout` does its own loud reporting.
    """
    if challenger_username is None:
        log(
            "NOT healing or repositioning the human, and NOT re-gearing them: "
            "no --challenger-username was pinned, and nothing on the wire tells "
            "this process who claimed the challenger slot (the state message "
            "has no field for it). Your challenger gets no iron sword and no "
            "armor, so they fight an armored agent bare-fisted: 6 damage a hit "
            "taken, and a bare hand's 1 arrives as 0.42 through the agent's "
            "iron, which is roughly 48 connected punches to win; and if the "
            "last match ended in the AGENT's death, they start this one on "
            "whatever health they had left. "
            "Restart the exhibition with --challenger-username <name> to fix it "
            "for the rest of the demo."
        )
        return False
    commands = human_reset_commands(anchor, challenger_username)
    if not paper_console:
        log(
            "--no-paper-console: not healing, repositioning or re-gearing "
            f"{challenger_username}. Run these from an opped account before "
            "the next match -- the gear lines matter every time, the heal "
            "only if the last match ended in the AGENT's death:"
        )
        for command in commands:
            log(f"  {command}")
        log(
            "Then confirm the armor actually went ON (a `give` would have put "
            "it in the bag instead, which looks identical in chat) -- these "
            "print the server's own answer:"
        )
        for command in human_loadout_read_commands(challenger_username):
            log(f"  {command}")
        return False
    log(
        f"healing, repositioning and re-gearing {challenger_username} via the "
        "Paper console."
    )
    if not send_paper_console_commands(paper_proc, commands, log=log):
        return False
    # Same pipe, written after the gear lines, and Paper runs console input in
    # order -- so this reads the world the re-gear left behind, not a race
    # against it. Best-effort and self-reporting: the match is played either way.
    confirm_human_loadout(paper_proc, paper_log, challenger_username, log=log)
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m deploy.exhibition",
        description=(
            "One-command exhibition launcher (T5): starts Paper, waits for it "
            "to be ready, starts the bridge in human-opponent mode, and "
            "connects the trained agent playing GREEDILY (no exploration, no "
            "learning) from a checkpoint. Runs in the foreground for the "
            "whole exhibition; Ctrl-C tears Paper and the bridge down. It "
            "plays ONE match and then waits: nothing restarts a match by "
            "itself. To arm the next challenger, anyone can type `reset` in "
            "Minecraft chat; the separate --reset command, run from another "
            "terminal, is the same trigger for an operator who has one."
        ),
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help=(
            "ARM THE NEXT CHALLENGER in an exhibition that is already running, "
            "then exit. Heals, repositions and re-gears both fighters (an iron "
            "sword AND a full iron set each), releases the challenger slot and "
            "plays exactly ONE more match -- one reset "
            "command, one match, never automatic. Starts nothing and never "
            "connects to the bridge (a second TCP client would evict the "
            "running agent); it hands a request file to the live launcher, "
            "which must be using the same --log-dir. Takes no other flags."
        ),
    )
    parser.add_argument(
        "--no-paper-console",
        dest="paper_console",
        action="store_false",
        help=(
            "do NOT open a console pipe to Paper. The pipe is how a reset "
            "gears the HUMAN and reads that gear back off the server (the "
            "datapack resets the learner only), so turning it off means an "
            "unarmed, UNARMORED challenger against an agent with an iron sword "
            "and a full iron set, and one who beat the agent carries their "
            "leftover health into the next match; the commands are printed "
            "instead. An escape hatch, not a default."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT,
        help=(
            f"trained DuelingDRQN checkpoint to play from (default: "
            f"{DEFAULT_CHECKPOINT}). Missing or unloadable => refuses to "
            "start and lists what checkpoints DO exist; NEVER falls back to "
            "a randomly-initialized agent."
        ),
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=str,
        default=DEFAULT_CHECKPOINTS_DIR,
        help=(
            "directory listed in the refusal message when --checkpoint is "
            f"missing or unloadable (default: {DEFAULT_CHECKPOINTS_DIR})."
        ),
    )
    parser.add_argument(
        "--challenger-username",
        type=str,
        default=None,
        help=(
            "PIN the human opponent's Minecraft username. Strongly "
            "recommended. Leave this unset and the bridge credits the FIRST "
            "non-agent player who steps into the pad as the opponent, so a "
            "bystander who dies to anything else gets reported as the "
            "agent's win. Pass this before a real exhibition."
        ),
    )
    parser.add_argument("--mc-host", type=str, default=DEFAULT_MC_HOST)
    parser.add_argument("--mc-port", type=int, default=DEFAULT_MC_PORT)
    parser.add_argument("--bridge-host", type=str, default=DEFAULT_BRIDGE_HOST)
    parser.add_argument("--bridge-port", type=int, default=DEFAULT_BRIDGE_PORT)
    parser.add_argument(
        "--node", type=str, default=DEFAULT_NODE, help="node executable (default: on PATH)."
    )
    parser.add_argument("--log-dir", type=str, default=str(DEFAULT_LOG_DIR))
    parser.add_argument(
        "--server-timeout", type=float, default=PAPER_READY_TIMEOUT_SECONDS,
        help="bounded wait for the Minecraft port (cold boot + world-gen).",
    )
    parser.add_argument(
        "--bridge-timeout", type=float, default=BRIDGE_READY_TIMEOUT_SECONDS,
        help="bounded wait for the bridge port (the learner bot joining).",
    )
    parser.add_argument("--xms", type=str, default=None, help="JVM initial heap, e.g. 4G.")
    parser.add_argument("--xmx", type=str, default=None, help="JVM max heap, e.g. 4G.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="run every preflight gate and print the resolved plan; start nothing.",
    )
    return parser


def run(
    argv: Optional[Sequence[str]] = None,
    *,
    popen: Callable[..., subprocess.Popen] = subprocess.Popen,
    port_probe: Callable[[str, int], bool] = _tcp_port_open,
    sleep: Callable[[float], None] = time.sleep,
    load_policy: Callable[[Path, Path], Any] = load_greedy_policy,
    transport_factory: Optional[Callable[[str, int], Any]] = None,
    write_ops: Callable[[int, str], str] = write_ops_json,
    which: Callable[[str], Optional[str]] = shutil.which,
    log: Callable[[str], None] = _ascii_log,
) -> int:
    """The real decision path behind ``main()``.

    Every side effect (spawning Paper/the bridge, probing a port, sleeping,
    loading the checkpoint, connecting the transport, writing ``ops.json``,
    resolving ``node`` on PATH) is behind a keyword seam, so tests drive this
    exact function — not a re-implementation of it — to prove the refusal
    paths without a live server AND without writing the git-tracked
    ``server/ops.json`` (TC20/TC21).

    Returns the process exit code: 0 for a clean ``--dry-run`` or an armed
    ``--reset``, 130 for a Ctrl-C (SIGINT) — the NORMAL way to end an
    exhibition, whether it interrupts the live boot or the wait between
    matches — ``SIGTERM_EXIT_CODE`` (143, i.e. 128 + 15) for a ``kill <pid>``,
    which runs exactly the same teardown as the Ctrl-C path but is a distinct
    event and so gets a distinct code, and non-zero for any refusal or boot
    failure.

    ``--reset`` (T6) is handled first and returns before every gate below: it
    is a different program that happens to share this entry point, and it must
    start nothing at all (see :func:`request_reset`).
    """
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    checkpoint_path = Path(args.checkpoint)
    checkpoints_dir = Path(args.checkpoints_dir)
    log_dir = Path(args.log_dir)

    # --- `--reset` is a different program sharing one entry point (T6). It
    # returns BEFORE every gate below: it loads no checkpoint, probes no port,
    # spawns nothing, writes no ops.json and does not even create the log dir.
    # All it does is hand a request to the launcher that is already running. ---
    if args.reset:
        conflicts = reset_mode_conflicts(args, parser.parse_args([]))
        if conflicts:
            log(reset_conflicts_message(conflicts))
            return 1
        return request_reset(log_dir, log=log)

    # --- Preflight gates. Nothing is spawned, and no file is written, until
    # every one of these passes (AC5: refusal paths start nothing). ----------

    try:
        exhibition_cfg = ExhibitionConfig(challenger_username=args.challenger_username)
    except ValueError as exc:
        log(f"invalid exhibition config: {exc}")
        return 1

    if args.challenger_username is None:
        log(
            "WARNING: no --challenger-username pinned. The bridge will credit "
            "the FIRST non-agent player who steps into the pad as the "
            "opponent -- a bystander who dies to anything else is reported "
            "as the agent's win. Pass --challenger-username before a real "
            "exhibition."
        )

    try:
        policy = load_policy(checkpoint_path, checkpoints_dir)
    except CheckpointError as exc:
        log(str(exc))
        return 1

    if not is_port_free(args.bridge_host, args.bridge_port, port_probe=port_probe):
        log(bridge_port_in_use_message(args.bridge_host, args.bridge_port))
        return 1

    if not is_port_free(args.mc_host, args.mc_port, port_probe=port_probe):
        log(mc_port_in_use_message(args.mc_host, args.mc_port))
        return 1

    # Last gate, and still before anything is spawned: without it a missing
    # `node` is discovered by the bridge popen BELOW, i.e. after Paper has
    # already spent 30-60s booting and generating a world. Deliberately after
    # the port gates so the AC5 refusals (TC20/TC21) keep their exact ordering.
    toolchain_problems = find_toolchain_problems(args.node, which=which)
    if toolchain_problems:
        log(toolchain_missing_message(toolchain_problems))
        return 1

    anchor = pad_anchor(PAD_INDEX)
    learner_username, dummy_username = pad_usernames(PAD_INDEX)
    # ONE derivation, THREE readers: the bridge argv below (a player typing
    # `reset` in chat writes this file), the drain at startup, and the idle poll
    # between matches. Computed here, before the argv that carries it, so the
    # launcher cannot end up polling a path it did not hand the bridge --
    # deriving it twice would be two implementations free to drift.
    request_path = reset_request_path(log_dir)
    bridge_argv = build_bridge_argv(
        node=args.node,
        mc_port=args.mc_port,
        bridge_port=args.bridge_port,
        anchor=anchor,
        learner_username=learner_username,
        dummy_username=dummy_username,
        challenger_username=args.challenger_username,
        reset_request_path=request_path,
    )

    if args.dry_run:
        log("dry run -- every preflight gate passed. Nothing was started.")
        log(f"  checkpoint  : {checkpoint_path}")
        log(f"  paper       : bash {START_SH}  (mc port {args.mc_port})")
        log(f"  bridge argv : {' '.join(bridge_argv)}")
        log(f"  pad anchor  : {anchor.x},{anchor.z}")
        return 0

    log_dir.mkdir(parents=True, exist_ok=True)

    # --- A request left over from an earlier launch must never arm a match
    # nobody asked for. Drained here, before anything is running, so the first
    # thing the idle loop sees below is genuinely this operator's request.
    # `request_path` is the value already baked into bridge_argv above. --------
    drain_reset_request(request_path, "left over from an earlier launch", log=log)

    # --- ops.json BEFORE Paper boots: Paper reads the op list at startup and
    # will not re-read a file written into an already-running server. The file is
    # generated, never committed (issue #29), so this write is also what makes a
    # fresh clone bootable at all -- and it narrows a fleet-sized op list a
    # training sweep left behind back to this one pad's two bots. It must happen
    # before start.sh, which refuses to launch on an op list without them. ------
    write_ops(1, str(OPS_JSON_PATH))

    paper_log = log_dir / "paper.log"
    bridge_log = log_dir / "bridge.log"
    paper_proc: Optional[subprocess.Popen] = None
    bridge_proc: Optional[subprocess.Popen] = None
    env: Optional[MCPvPEnv] = None

    # --- From here on this process owns children, so SIGTERM has to reach the
    # teardown below instead of killing the supervisor outright and leaving them
    # behind (module docstring, TEARDOWN). Installed HERE, after every branch
    # that returns early -- `--reset`, `--dry-run` and every refusal -- so those
    # modes leave this process's inherited signal behavior exactly as they found
    # it, and their exit codes cannot change. ---------------------------------
    restore_sigterm = install_sigterm_handler(log=log)

    try:
        # --- Paper ----------------------------------------------------------
        log(f"starting Paper (log: {paper_log}); start.sh pins Java 21.")
        # stdin is a PIPE, not DEVNULL: `start.sh` execs the JVM, so this pipe
        # is the SERVER CONSOLE, and it is the only command channel this process
        # can reach (RCON is off and the wire has no command slot). A reset uses
        # it to heal and reposition the human, whom the datapack's reset does
        # not touch. --no-paper-console falls back to DEVNULL, exactly the T5
        # behavior, and prints the commands instead of running them.
        with open(paper_log, "wb") as handle:
            paper_proc = popen(
                ["bash", str(START_SH)],
                cwd=str(SERVER_DIR),
                stdout=handle,
                stderr=subprocess.STDOUT,
                stdin=subprocess.PIPE if args.paper_console else subprocess.DEVNULL,
                env=_paper_env(args.xms, args.xmx),
            )
        if not wait_for_port(
            args.mc_host,
            args.mc_port,
            timeout=args.server_timeout,
            label="Paper",
            process=paper_proc,
            port_probe=port_probe,
            sleep=sleep,
            log=log,
        ):
            _dump_log_tail(paper_log, log=log)
            raise ExhibitionLaunchError("Paper did not come up in time.")
        log(f"Paper is up on {args.mc_host}:{args.mc_port}.")

        # --- Bridge (human opponent mode) ------------------------------------
        log(f"starting the bridge (log: {bridge_log}): {' '.join(bridge_argv)}")
        with open(bridge_log, "wb") as handle:
            bridge_proc = popen(
                bridge_argv,
                cwd=str(REPO_ROOT),
                stdout=handle,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
            )
        if not wait_for_port(
            args.bridge_host,
            args.bridge_port,
            timeout=args.bridge_timeout,
            label="bridge",
            process=bridge_proc,
            port_probe=port_probe,
            sleep=sleep,
            log=log,
        ):
            _dump_log_tail(bridge_log, log=log)
            raise ExhibitionLaunchError("the bridge did not come up in time.")
        log(f"bridge is up on {args.bridge_host}:{args.bridge_port}; learner spawned.")

        _print_join_instructions(args.mc_port, anchor, log=log)

        # --- Connect the agent, greedily, no learning. -----------------------
        if transport_factory is not None:
            transport = transport_factory(args.bridge_host, args.bridge_port)
        else:
            transport = TcpBridgeClient(host=args.bridge_host, port=args.bridge_port)
        env = MCPvPEnv(
            transport=transport, max_episode_steps=exhibition_cfg.env_max_episode_steps
        )

        # --- One match per trigger, for as long as the operator keeps asking.
        # The launch itself is the first trigger; every later match needs a
        # reset command. Nothing here can start a match on its own, which is
        # AC4 and the user's own rule ("after one death, no restart auto"). ----
        hint = reset_command_hint(log_dir)
        match_number = 1
        while True:
            log(f"--- match {match_number} ---")
            play_one_match(
                env, policy, reflex_blind_steps=exhibition_cfg.reflex_blind_steps, log=log
            )
            log(
                "no auto-restart: a death ends the match and nothing starts "
                "another one by itself. To arm the next challenger, anyone can "
                "type `reset` in Minecraft chat -- no terminal needed -- or run "
                f"`{hint}` in another terminal. Ctrl-C here ends the exhibition."
            )
            # A request filed while that match was still running is discarded:
            # honoring it would make the death the proximate cause of the
            # restart. Loud, because the operator is waiting on it.
            drain_reset_request(
                request_path,
                "that was filed while the match was still running -- a death "
                "must never restart the match by itself, so trigger it again "
                "now that this one has ended (type `reset` in Minecraft chat, "
                f"or run `{hint}`)",
                log=log,
            )
            wait_for_reset_request(request_path, sleep=sleep)
            match_number += 1
            log(f"reset requested -- arming match {match_number}.")
            _reset_human_side(
                paper_proc,
                anchor,
                exhibition_cfg.challenger_username,
                paper_console=args.paper_console,
                paper_log=paper_log,
                log=log,
            )
    except ExhibitionLaunchError as exc:
        log(str(exc))
        return 1
    except KeyboardInterrupt:
        log("interrupted.")
        return 130
    except ExhibitionTerminated as exc:
        # `kill <pid>`, or whatever supervises this process stopping it. The
        # teardown below is the same one Ctrl-C gets -- only the code differs,
        # because reporting 130 would tell whoever reads the log that somebody
        # pressed Ctrl-C at a console nobody was sitting at.
        log(f"{exc}; shutting down.")
        return SIGTERM_EXIT_CODE
    except Exception as exc:  # noqa: BLE001 — a mid-match fault (e.g. a dropped
        # bridge connection) must not crash with a raw traceback in front of a
        # live audience; the step contract forbids a silent retry, so tear
        # down and report instead.
        log(f"fatal error: {exc}")
        return 1
    finally:
        # NESTED, not sequential. Each helper already absorbs BaseException, so
        # this is belt-and-braces — but the braces are what guarantee that a
        # link raising something no handler anticipated still cannot skip the
        # links after it. Losing the last link means an orphaned Paper JVM
        # squatting on the Minecraft port; losing the first means both.
        #
        # The order is fixed: the agent's connection, then the bridge, then
        # Paper. Stopping Paper first would leave the bridge's mineflayer bots
        # thrashing against a dead server while it shuts down.
        #
        # Restoring SIGTERM is the OUTERMOST link, so the handler stays live for
        # the whole chain: a second `kill` arriving during the grace wait then
        # raises into the helpers that already absorb a second Ctrl-C, instead
        # of hitting SIG_DFL and killing this supervisor mid-teardown with a
        # child still up -- the exact orphan this handler exists to prevent.
        try:
            try:
                if env is not None:
                    _close_env(env, log=log)
            finally:
                try:
                    if bridge_proc is not None:
                        _stop_process(bridge_proc, "bridge", log=log)
                finally:
                    try:
                        if paper_proc is not None:
                            _stop_process(paper_proc, "Paper", log=log)
                    finally:
                        # Last of the child-facing links, and deliberately after
                        # Paper is already down: there is nothing left to say to
                        # the console, and an unclosed pipe finalized at
                        # interpreter exit prints an "Exception ignored in ..."
                        # that reads like a crash.
                        if paper_proc is not None:
                            _close_paper_console(paper_proc)
        finally:
            if restore_sigterm is not None:
                restore_sigterm()


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point (``python -m deploy.exhibition``). Thin wrapper over
    :func:`run` — keep decision logic in `run` so tests exercise it directly."""
    return run(argv)


if __name__ == "__main__":
    sys.exit(main())
