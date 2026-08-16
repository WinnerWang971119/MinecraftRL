"""deploy.exhibition — one-command human-exhibition launcher (T5).

``python -m deploy.exhibition`` starts Paper, waits for it to actually accept
connections, starts the bridge in HUMAN opponent mode (``--opponent-mode
human``, T3), connects the trained agent playing GREEDILY (epsilon=0, no
learning) from a checkpoint, and prints the LAN join address plus the pad
coordinates so a classmate can join and find the arena without being talked
through it.

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
plays EXACTLY ONE match (:func:`play_one_match` is called once), then holds the
agent's bridge connection open and idles until Ctrl-C, which is what tears
Paper and the bridge down. A match ending (``done=True``, AC4: no auto-restart)
neither exits the process nor starts a second match — it reports the result and
idles, still connected. Driving another match is T6's job, and it has to happen
INSIDE this process: the bridge accepts exactly one TCP client, so a separate
reset process that connected would evict this one's live agent connection.
:func:`_idle_until_interrupt` is the seam where that plugs in. Whether T6
reuses this live connection or this process should instead disconnect while
idling is T6's call, not decided here — left as a one-line TODO at that
function.

TEARDOWN IS ``BaseException``-PROOF, not ``Exception``-proof. Ctrl-C is the
normal way an exhibition ends, so the teardown chain runs with an operator's
finger still on the key: a SECOND Ctrl-C lands inside the first one's grace
wait, and ``KeyboardInterrupt`` derives from ``BaseException``. Every helper
the ``finally`` chain calls absorbs it, and the chain itself is nested so each
link still runs when the one before it raises — otherwise a double-tap orphans
the Paper JVM on port 25565 and the NEXT launch refuses on its own mc-port
gate.

Every side-effecting step (spawning a process, probing a port, sleeping) is
behind an injectable seam so :mod:`tests.test_exhibition` can drive the real
decision path — :func:`run`, not a re-implementation of it — without booting
a real Paper server or Node bridge.
"""

from __future__ import annotations

import argparse
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, NoReturn, Optional, Sequence

from distributed.launcher import PadAnchor, pad_anchor, pad_usernames, write_ops_json
from env.mc_pvp_env import ExhibitionConfig, MCPvPEnv, TcpBridgeClient

__all__ = [
    "ExhibitionLaunchError",
    "CheckpointError",
    "build_bridge_argv",
    "find_toolchain_problems",
    "is_port_free",
    "load_greedy_policy",
    "play_one_match",
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
IDLE_POLL_SECONDS = 5.0


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
    pad_index: int = PAD_INDEX,
) -> List[str]:
    """The exact argv that launches this pad's bridge in HUMAN opponent mode.

    Mirrors ``distributed.launcher.SubprocessArenaLauncher.spec_for`` (the
    training bridge argv — same flags, same order for the shared subset) plus
    the two exhibition-only flags: ``--opponent-mode human`` always, and
    ``--challenger-username`` when pinned. Pure: no subprocess, no filesystem,
    no network, so tests/test_exhibition.py exercises it directly.
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
    ]
    if challenger_username is not None:
        argv.extend(["--challenger-username", challenger_username])
    return argv


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


def play_one_match(env: Any, policy: Any, *, log: Callable[[str], None] = _ascii_log) -> str:
    """Play exactly one match to completion, greedily, with no learning.

    Resets once, then steps the greedy policy until the env reports ``done`` —
    which under ``ExhibitionConfig.no_timeout`` only happens on an actual
    death (agent or human, AC4), never a step-count timeout. While no
    challenger has claimed the pad, the opponent block of the observation is
    zeroed (the bridge holds IDLE and keeps ``state`` flowing rather than
    carrying a status string — there is no wire slot for one); the greedy
    policy simply acts on that, same as it would on any other observation.
    There is nothing exhibition-specific to special-case here.

    Returns a short human-readable outcome string.
    """
    if hasattr(policy, "reset"):
        policy.reset()
    obs = env.reset()
    done = False
    info: Dict[str, Any] = {}
    steps = 0
    while not done:
        action = policy.act(obs)
        obs, _reward, done, info = env.step(action)
        steps += 1
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


def _idle_until_interrupt(
    *, sleep: Callable[[float], None] = time.sleep, poll_seconds: float = IDLE_POLL_SECONDS
) -> NoReturn:
    """Block, holding the agent's connection open, until Ctrl-C.

    Annotated ``NoReturn`` because it is: the loop has no exit condition, so
    the only ways out are the ``KeyboardInterrupt`` this exists to wait for or
    an exception from ``sleep``. Both unwind into :func:`run`'s handlers.

    TODO(T6): the separate reset command needs a way to arm the next
    challenger while this process still holds the bridge's single TCP client
    slot. This is the seam where that plugs in (e.g. watching a reset-request
    file or a signal) instead of a second process reconnecting to the bridge
    and evicting this one. Left as a plain wait — T6 decides the mechanism.
    """
    while True:
        sleep(poll_seconds)


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
            "whole exhibition; Ctrl-C tears Paper and the bridge down. Arming "
            "the next challenger between matches is the separate --reset "
            "command (T6)."
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

    Returns the process exit code: 0 only for a clean ``--dry-run``, 130 for
    a Ctrl-C (SIGINT) — the NORMAL way to end an exhibition, whether it
    interrupts the live boot or the post-match idle wait — and non-zero for
    any refusal or boot failure.
    """
    args = _build_arg_parser().parse_args(argv)

    checkpoint_path = Path(args.checkpoint)
    checkpoints_dir = Path(args.checkpoints_dir)
    log_dir = Path(args.log_dir)

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
    bridge_argv = build_bridge_argv(
        node=args.node,
        mc_port=args.mc_port,
        bridge_port=args.bridge_port,
        anchor=anchor,
        learner_username=learner_username,
        dummy_username=dummy_username,
        challenger_username=args.challenger_username,
    )

    if args.dry_run:
        log("dry run -- every preflight gate passed. Nothing was started.")
        log(f"  checkpoint  : {checkpoint_path}")
        log(f"  paper       : bash {START_SH}  (mc port {args.mc_port})")
        log(f"  bridge argv : {' '.join(bridge_argv)}")
        log(f"  pad anchor  : {anchor.x},{anchor.z}")
        return 0

    log_dir.mkdir(parents=True, exist_ok=True)

    # --- ops.json BEFORE Paper boots: Paper reads the op list at startup and
    # will not re-read a file written into an already-running server. At N=1
    # this is byte-identical to the committed server/ops.json, so it both ops
    # this pad's bots and self-heals a tree a training sweep left dirty
    # (issue #29 -- see server/setup/start-pads.sh's docstring; this is the
    # same rewrite, not a new one). -------------------------------------------
    write_ops(1, str(OPS_JSON_PATH))

    paper_log = log_dir / "paper.log"
    bridge_log = log_dir / "bridge.log"
    paper_proc: Optional[subprocess.Popen] = None
    bridge_proc: Optional[subprocess.Popen] = None
    env: Optional[MCPvPEnv] = None

    try:
        # --- Paper ----------------------------------------------------------
        log(f"starting Paper (log: {paper_log}); start.sh pins Java 21.")
        with open(paper_log, "wb") as handle:
            paper_proc = popen(
                ["bash", str(START_SH)],
                cwd=str(SERVER_DIR),
                stdout=handle,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
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

        play_one_match(env, policy, log=log)
        log(
            "holding Paper and the bridge up. Run the reset command before the "
            "next challenger (T6); Ctrl-C here ends the exhibition."
        )
        _idle_until_interrupt(sleep=sleep)
        # Unreachable: _idle_until_interrupt is NoReturn. Kept so every path
        # out of this block is an explicit exit code rather than an implicit
        # None, if that function ever grows a way to finish normally.
        return 0
    except ExhibitionLaunchError as exc:
        log(str(exc))
        return 1
    except KeyboardInterrupt:
        log("interrupted.")
        return 130
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
        try:
            if env is not None:
                _close_env(env, log=log)
        finally:
            try:
                if bridge_proc is not None:
                    _stop_process(bridge_proc, "bridge", log=log)
            finally:
                if paper_proc is not None:
                    _stop_process(paper_proc, "Paper", log=log)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point (``python -m deploy.exhibition``). Thin wrapper over
    :func:`run` — keep decision logic in `run` so tests exercise it directly."""
    return run(argv)


if __name__ == "__main__":
    sys.exit(main())
