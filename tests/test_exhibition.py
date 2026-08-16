"""tests/test_exhibition.py — T5 launcher + T6 reset command.

Offline only: no socket is opened, no subprocess is spawned, no live Paper
server or Node bridge is touched, and the git-tracked ``server/ops.json`` is
never written to (every test that reaches the file-mutation point in
``run()`` mocks ``write_ops_json`` or never reaches that far).

Required by the plan (docs/plans/2026-08-16-demo-scripted-opponent-exhibition.md):

  * TC20 — bridge port already taken -> ``run()`` exits non-zero with an
    actionable message and starts NOTHING (no ``popen`` call, no ops.json
    write). Driven through the real ``run()`` orchestration, not a
    re-implementation of the port-refusal logic.
  * TC21 — checkpoint missing -> ``run()`` exits non-zero, the message names
    the expected path AND lists the checkpoints that DO exist. Also driven
    through the real ``run()`` / ``load_greedy_policy`` path (a real tmp-dir
    filesystem, no mocking of the decision itself).

  * The HAPPY path — both children boot, the agent plays exactly one match
    over a scripted bridge, Ctrl-C ends the exhibition (exit 130) and teardown
    runs in order. This is the only section that exercises anything past "Paper
    never comes up": without it, deleting the ``play_one_match()`` call from
    ``run()`` entirely leaves every other test in this file green.

  * T6 (AC5) — the SEPARATE reset command. ``--reset`` files a request and
    starts nothing; the running launcher consumes it, heals, repositions and
    re-arms BOTH sides and plays exactly one more match. The AC4 half is tested as
    hard as the AC5 half: a match never restarts itself, a request left over
    from an earlier launch is discarded, and a request filed while a match is
    still running is discarded too (honoring it would make the death the
    proximate cause of the restart).

Everything else here is supporting coverage for the pure helpers `run()` is
built from (``is_port_free``, ``find_checkpoints``, ``build_bridge_argv``,
``load_greedy_policy``, ``wait_for_port``, ``play_one_match``,
``find_toolchain_problems``, ``human_reset_commands``, the request-file
helpers) plus the --checkpoint-missing/unloadable "never random-init"
guarantee, the --challenger-username help-text requirement from the spec, and
the ``BaseException``-proof teardown a second Ctrl-C depends on.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from agent.actions import Macro
from bridge.messages import ResetAckMsg, StateMsg
from deploy.exhibition import (
    DEFAULT_BRIDGE_HOST,
    DEFAULT_BRIDGE_PORT,
    DEFAULT_LOG_DIR,
    DEFAULT_MC_PORT,
    RESET_REQUEST_FILENAME,
    CheckpointError,
    build_bridge_argv,
    checkpoint_missing_message,
    checkpoint_unloadable_message,
    drain_reset_request,
    find_checkpoints,
    find_toolchain_problems,
    human_reset_commands,
    is_port_free,
    load_greedy_policy,
    main,
    play_one_match,
    request_reset,
    reset_command_hint,
    reset_mode_conflicts,
    reset_request_path,
    run,
    send_paper_console_commands,
    take_reset_request,
    wait_for_port,
    wait_for_reset_request,
)
from distributed.launcher import pad_anchor, pad_usernames
from env.mc_pvp_env import BridgeError, MCPvPEnv
from env.observation_spec import OBS_DIM, Obs

REPO_ROOT = Path(__file__).resolve().parent.parent
OPS_JSON = REPO_ROOT / "server" / "ops.json"
DUMMY_PAD_MCFUNCTION = (
    REPO_ROOT / "server" / "arena" / "data" / "arena" / "function" / "spawn_dummy_pad.mcfunction"
)
LEARNER_PAD_MCFUNCTION = (
    REPO_ROOT / "server" / "arena" / "data" / "arena" / "function" / "spawn_learner_pad.mcfunction"
)


# ---------------------------------------------------------------------------
# Shared fakes (mirror the FakeProc / ScriptedProbe style in
# tests/test_pad_launcher.py — no real OS process or socket).
# ---------------------------------------------------------------------------


class FakeConsole:
    """Stand-in for Paper's stdin PIPE — the server console (T6).

    Records the lines written to it. ``fail=True`` is Paper having died
    mid-exhibition: the real pipe raises ``BrokenPipeError`` (an ``OSError``),
    which a reset must survive rather than take the exhibition down with it.

    ``write`` asserts BYTES. ``subprocess.PIPE`` without an encoding is a binary
    stream, so a launcher that wrote ``str`` would raise ``TypeError`` on the
    first real reset and heal nobody; a fake that accepted both would hide it.
    """

    def __init__(self, *, fail=False, record=None):
        self.lines = []
        self.flushes = 0
        self.closed = False
        self._fail = fail
        self._record = record

    def write(self, data):
        if self._fail:
            raise BrokenPipeError("paper is gone")
        assert isinstance(data, bytes), f"the console pipe takes bytes, got {type(data)}"
        line = data.decode("ascii").rstrip("\n")
        self.lines.append(line)
        if self._record is not None:
            self._record(("console", line))
        return len(data)

    def flush(self):
        self.flushes += 1

    def close(self):
        self.closed = True


class FakeProc:
    """Minimal Popen stand-in: poll only, no OS process.

    ``label``/``record`` are optional: pass both and every lifecycle call
    appends ``(what, label)`` to the shared event list, which is how the
    happy-path test pins teardown ORDER (bridge before Paper) rather than
    merely asserting both were stopped.

    ``stdin`` mirrors ``Popen.stdin``: a stream when the caller asked for
    ``subprocess.PIPE``, ``None`` otherwise (``DEVNULL`` opens no pipe).
    """

    def __init__(self, pid=4242, exit_code=None, *, label=None, record=None, stdin=None):
        self.pid = pid
        self._exit_code = exit_code
        self.label = label
        self._record = record
        self.stdin = stdin
        self.terminated = False
        self.killed = False
        self.waits = 0

    def _emit(self, what):
        if self._record is not None:
            self._record((what, self.label))

    def poll(self):
        return self._exit_code

    def terminate(self):
        self.terminated = True
        self._exit_code = -15
        self._emit("terminate")

    def kill(self):
        self.killed = True
        self._exit_code = -9
        self._emit("kill")

    def wait(self, timeout=None):
        self.waits += 1
        return self._exit_code


class SlowStoppingProc(FakeProc):
    """A child that does NOT die on SIGTERM, and whose grace wait is cut short
    by a SECOND Ctrl-C.

    This is the demo-day sequence W1 is about: first Ctrl-C -> ``run()``
    returns 130 -> ``finally`` -> ``_stop_process(bridge)`` -> ``terminate()``
    -> ``wait(timeout=20)``, and the operator, watching twenty seconds of
    silence, hits Ctrl-C again. ``KeyboardInterrupt`` derives from
    ``BaseException``, so an ``except Exception`` teardown lets it escape and
    Paper is never stopped at all.
    """

    def terminate(self):
        # SIGTERM delivered, but the process is still alive -- which is the
        # whole reason _stop_process waits instead of returning immediately.
        self.terminated = True
        self._emit("terminate")

    def wait(self, timeout=None):
        self.waits += 1
        if self.waits == 1:
            raise KeyboardInterrupt
        return self._exit_code


class RefusingPopen:
    """A ``popen`` stand-in that fails the test if it is ever called.

    TC20/TC21 assert "nothing has been started" by asserting THIS never
    fires, not by inspecting real OS processes.
    """

    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        raise AssertionError(
            "run() must not spawn any process once a preflight gate refuses"
        )


def collector():
    """A list-backed ``log`` callable a test can assert against."""
    messages = []
    messages_fn = messages.append
    return messages, messages_fn


def recording_write_ops(sink):
    """A ``write_ops`` seam that RECORDS instead of writing.

    The refusal tests assert "nothing was written" by asserting this recorded
    nothing. Byte-comparing ``server/ops.json`` cannot make that assertion on
    its own: at ``n_pads == 1`` the write is byte-identical to the committed
    file BY DESIGN, so a refusing run that wrongly rewrote it would still
    compare equal. The byte-compare stays as a cheap belt-and-braces check that
    the real file was not touched by some other route.
    """

    def write_ops(n_pads, path):
        sink.append((n_pads, path))
        return path

    return write_ops


def call_capturing_escape(fn, *args, **kwargs):
    """Call ``fn`` and return whatever it let escape (``None`` if it returned).

    The teardown helpers must absorb ``BaseException``. Asserting that with a
    plain call would let a ``KeyboardInterrupt`` reach pytest, which ABORTS the
    session -- "56 passed", no failures -- instead of failing the test that
    just caught the regression.
    """
    try:
        fn(*args, **kwargs)
    except BaseException as exc:  # noqa: BLE001 — that is the thing under test.
        return exc
    return None


def stub_which(name):
    """A ``which`` seam that resolves everything.

    Every test that reaches the toolchain gate injects this, so the suite never
    depends on whether Node happens to be installed on the machine running it.
    """
    return f"/usr/bin/{name}"


# ---------------------------------------------------------------------------
# TC20 — bridge port already in use: exits non-zero, starts nothing (AC5).
# ---------------------------------------------------------------------------


class TestTC20BridgePortInUse:
    def test_refuses_and_spawns_nothing(self):
        messages, log = collector()
        popen = RefusingPopen()
        probed = []

        def port_probe(host, port):
            probed.append((host, port))
            # Only the bridge port looks occupied; the mc port is free so the
            # bridge-port gate is unambiguously what triggers the refusal.
            return port == DEFAULT_BRIDGE_PORT

        # The checkpoint gate runs first in run()'s ordering; stub it so this
        # test is only exercising the port-refusal decision, per TC20's scope.
        def fake_load_policy(_checkpoint_path, _checkpoints_dir):
            return object()

        ops_writes = []
        ops_json_before = OPS_JSON.read_bytes()

        code = run(
            ["--challenger-username", "Steve"],
            popen=popen,
            port_probe=port_probe,
            load_policy=fake_load_policy,
            write_ops=recording_write_ops(ops_writes),
            log=log,
        )

        assert code != 0
        assert popen.calls == []
        assert (DEFAULT_BRIDGE_HOST, DEFAULT_BRIDGE_PORT) in probed
        # The binding half of "nothing has been started": the write was never
        # ATTEMPTED. See recording_write_ops -- an N=1 write is byte-identical
        # to the committed file, so the byte-compare below cannot detect one.
        assert ops_writes == []
        assert OPS_JSON.read_bytes() == ops_json_before  # not touched either

        text = "\n".join(messages)
        assert "already in use" in text
        assert "nothing has been started" in text
        assert str(DEFAULT_BRIDGE_PORT) in text

    def test_bridge_port_in_use_message_names_the_port_and_says_nothing_started(self):
        from deploy.exhibition import bridge_port_in_use_message

        text = bridge_port_in_use_message("127.0.0.1", 5555)
        assert "5555" in text
        assert "127.0.0.1" in text
        assert "nothing has been started" in text
        # ONE TCP client is the reason a second launcher is refused, not just
        # a generic "port busy" -- the operator needs to know WHY.
        assert "ONE TCP client" in text

    def test_mc_port_in_use_also_refuses_with_nothing_started(self):
        messages, log = collector()
        popen = RefusingPopen()
        ops_writes = []

        def port_probe(host, port):
            # Bridge port free, mc port occupied.
            return port != DEFAULT_BRIDGE_PORT

        def fake_load_policy(_checkpoint_path, _checkpoints_dir):
            return object()

        code = run(
            ["--challenger-username", "Steve"],
            popen=popen,
            port_probe=port_probe,
            load_policy=fake_load_policy,
            write_ops=recording_write_ops(ops_writes),
            log=log,
        )

        assert code != 0
        assert popen.calls == []
        assert ops_writes == []
        assert "already listening on Minecraft port" in "\n".join(messages)

    def test_mc_port_message_carries_the_command_that_frees_the_port(self):
        from deploy.exhibition import mc_port_in_use_message

        text = mc_port_in_use_message("127.0.0.1", 25565)
        # "Stop the running server first" is not actionable at demo time if the
        # server is an orphaned JVM with no terminal attached. The most likely
        # way to reach this state is exactly that, so ship the recovery command.
        assert "lsof -ti:25565 | xargs kill" in text
        assert "nothing has been started" in text


# ---------------------------------------------------------------------------
# TC21 — checkpoint missing: exits non-zero, lists what exists (AC5).
# ---------------------------------------------------------------------------


class TestTC21CheckpointMissing:
    def test_refuses_lists_existing_checkpoints_and_spawns_nothing(self, tmp_path):
        checkpoints_dir = tmp_path / "runs"
        checkpoints_dir.mkdir()
        (checkpoints_dir / "alpha.pt").write_bytes(b"not a real checkpoint")
        (checkpoints_dir / "beta.pt").write_bytes(b"not a real checkpoint")
        missing_path = checkpoints_dir / "does_not_exist.pt"

        messages, log = collector()
        popen = RefusingPopen()
        ops_writes = []

        # This test drives the REAL load_greedy_policy (the default `run()`
        # uses) -- not an injected stand-in -- because TC21 is specifically
        # about that decision path. port_probe is irrelevant here (the
        # checkpoint gate runs first and refuses before any port is probed).
        code = run(
            [
                "--checkpoint",
                str(missing_path),
                "--checkpoints-dir",
                str(checkpoints_dir),
                "--challenger-username",
                "Steve",
            ],
            popen=popen,
            port_probe=lambda host, port: False,
            write_ops=recording_write_ops(ops_writes),
            log=log,
        )

        assert code != 0
        assert popen.calls == []
        assert ops_writes == []  # see recording_write_ops

        text = "\n".join(messages)
        assert str(missing_path) in text
        assert "alpha.pt" in text
        assert "beta.pt" in text
        assert "randomly-initialized" in text  # the "never fall back" promise

    def test_no_checkpoints_at_all_says_so_plainly(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        missing = empty_dir / "ghost.pt"

        text = checkpoint_missing_message(missing, empty_dir)
        assert str(missing) in text
        assert "no checkpoints found" in text

    def test_load_greedy_policy_raises_checkpoint_error_for_a_missing_file(self, tmp_path):
        checkpoints_dir = tmp_path
        (checkpoints_dir / "one.pt").write_bytes(b"x")
        missing = checkpoints_dir / "two.pt"

        with pytest.raises(CheckpointError) as excinfo:
            load_greedy_policy(missing, checkpoints_dir)
        assert "one.pt" in str(excinfo.value)
        assert str(missing) in str(excinfo.value)

    def test_load_greedy_policy_never_falls_back_on_a_corrupt_checkpoint(self, tmp_path):
        # Present, but not a real torch checkpoint: this is the "unloadable"
        # half of the refusal contract -- exists on disk, still refuses, and
        # still never returns a randomly-initialized network.
        bad = tmp_path / "bad.pt"
        bad.write_bytes(b"this is not a torch checkpoint")
        (tmp_path / "good_looking.pt").write_bytes(b"also not real, just listed")

        with pytest.raises(CheckpointError) as excinfo:
            load_greedy_policy(bad, tmp_path)
        text = str(excinfo.value)
        assert str(bad) in text
        assert "could not be loaded" in text
        assert "randomly-initialized" in text
        assert "good_looking.pt" in text

    def test_checkpoint_unloadable_message_excludes_the_bad_file_itself(self, tmp_path):
        bad = tmp_path / "bad.pt"
        bad.write_bytes(b"junk")
        (tmp_path / "other.pt").write_bytes(b"junk")

        text = checkpoint_unloadable_message(bad, tmp_path, RuntimeError("boom"))
        assert "boom" in text
        assert "other.pt" in text
        # The checkpoint that just failed to load must not appear in its own
        # "other checkpoints found" list -- that would be nonsensical advice.
        assert text.count("bad.pt") == 1  # only in the "could not be loaded" line

    def test_unloadable_message_leads_with_the_actionable_line(self, tmp_path):
        bad = tmp_path / "bad.pt"
        bad.write_bytes(b"junk")
        # Shaped like torch's real weights_only failure: one useful sentence
        # followed by several paragraphs of pickle-security background.
        essay = RuntimeError(
            "Weights only load failed.\n"
            "In PyTorch 2.6 the default for weights_only was flipped.\n"
            "Please read the release notes.\n"
            "And this appendix."
        )
        lines = checkpoint_unloadable_message(bad, tmp_path, essay).splitlines()

        # The headline carries only the exception's FIRST line, and the thing
        # the operator should actually do is the very next line -- not the
        # seventh, under an essay nobody reads with a classroom waiting.
        assert lines[0].endswith("Weights only load failed.")
        assert "fix or replace the checkpoint" in lines[1]
        # The remaining detail is not discarded, only demoted.
        assert any("And this appendix." in line for line in lines[2:])

    def test_unloadable_message_survives_an_exception_with_no_text(self, tmp_path):
        bad = tmp_path / "bad.pt"
        bad.write_bytes(b"junk")
        # str(RuntimeError()) is "", so splitlines() is [] -- taking [0] of it
        # would raise IndexError while building a refusal message.
        text = checkpoint_unloadable_message(bad, tmp_path, RuntimeError())
        assert "could not be loaded" in text
        assert "RuntimeError" in text


# ---------------------------------------------------------------------------
# is_port_free / find_checkpoints — the pure per-gate building blocks.
# ---------------------------------------------------------------------------


class TestIsPortFree:
    def test_true_when_the_probe_reports_nothing_listening(self):
        assert is_port_free("127.0.0.1", 12345, port_probe=lambda h, p: False) is True

    def test_false_when_the_probe_reports_something_listening(self):
        assert is_port_free("127.0.0.1", 12345, port_probe=lambda h, p: True) is False

    def test_passes_host_and_port_through_unchanged(self):
        seen = []
        is_port_free("10.0.0.5", 9999, port_probe=lambda h, p: seen.append((h, p)) or False)
        assert seen == [("10.0.0.5", 9999)]


class TestFindCheckpoints:
    def test_lists_pt_files_sorted(self, tmp_path):
        (tmp_path / "z.pt").write_bytes(b"")
        (tmp_path / "a.pt").write_bytes(b"")
        (tmp_path / "notes.txt").write_bytes(b"")  # must be ignored
        assert find_checkpoints(tmp_path) == ["a.pt", "z.pt"]

    def test_missing_directory_returns_empty_list_not_an_error(self, tmp_path):
        assert find_checkpoints(tmp_path / "does_not_exist") == []

    def test_empty_directory_returns_empty_list(self, tmp_path):
        assert find_checkpoints(tmp_path) == []


# ---------------------------------------------------------------------------
# build_bridge_argv — the "assemble the argv" pure function.
# ---------------------------------------------------------------------------


class TestBuildBridgeArgv:
    def _base_kwargs(self, **overrides):
        anchor = pad_anchor(0)
        learner, dummy = pad_usernames(0)
        kwargs = dict(
            node="node",
            run_js=Path("/repo/bridge/run.js"),
            mc_port=25565,
            bridge_port=5555,
            anchor=anchor,
            learner_username=learner,
            dummy_username=dummy,
            challenger_username=None,
        )
        kwargs.update(overrides)
        return kwargs

    def test_always_carries_opponent_mode_human(self):
        argv = build_bridge_argv(**self._base_kwargs())
        assert "--opponent-mode" in argv
        assert argv[argv.index("--opponent-mode") + 1] == "human"

    def test_omits_challenger_username_when_unset(self):
        argv = build_bridge_argv(**self._base_kwargs(challenger_username=None))
        assert "--challenger-username" not in argv

    def test_pins_challenger_username_when_set(self):
        argv = build_bridge_argv(**self._base_kwargs(challenger_username="Alice"))
        assert "--challenger-username" in argv
        assert argv[argv.index("--challenger-username") + 1] == "Alice"

    def test_pad_origin_is_the_reused_pad_anchor_flag(self):
        anchor = pad_anchor(0)
        argv = build_bridge_argv(**self._base_kwargs(anchor=anchor))
        assert "--pad-origin" in argv
        assert argv[argv.index("--pad-origin") + 1] == anchor.as_flag() == "0,0"

    def test_node_and_run_js_lead_the_argv(self):
        argv = build_bridge_argv(**self._base_kwargs(node="/usr/bin/node"))
        assert argv[0] == "/usr/bin/node"
        assert argv[1] == "/repo/bridge/run.js"

    def test_usernames_are_pad_zeros_default_bot_names(self):
        argv = build_bridge_argv(**self._base_kwargs())
        assert argv[argv.index("--learner-username") + 1] == "learner_bot"
        assert argv[argv.index("--dummy-username") + 1] == "dummy_bot"

    def test_pure_no_side_effects_and_deterministic(self):
        kwargs = self._base_kwargs(challenger_username="Bob")
        assert build_bridge_argv(**kwargs) == build_bridge_argv(**kwargs)


# ---------------------------------------------------------------------------
# wait_for_port — the boot-gate poll loop (FakeProc, no real process/socket).
# ---------------------------------------------------------------------------


class TestWaitForPort:
    def test_true_as_soon_as_the_port_answers(self):
        proc = FakeProc()
        sleeps = []
        assert wait_for_port(
            "127.0.0.1", 5555, timeout=10, label="x", process=proc,
            port_probe=lambda h, p: True, sleep=sleeps.append,
        ) is True
        assert sleeps == []  # no polling needed at all

    def test_false_when_the_process_exits_before_the_port_opens(self):
        proc = FakeProc(exit_code=1)
        assert wait_for_port(
            "127.0.0.1", 5555, timeout=10, label="x", process=proc,
            port_probe=lambda h, p: False, sleep=lambda s: None,
        ) is False

    def test_false_when_the_timeout_elapses(self):
        proc = FakeProc()
        elapsed = {"t": 0.0}

        def fake_sleep(seconds):
            elapsed["t"] += seconds

        assert wait_for_port(
            "127.0.0.1", 5555, timeout=3, label="x", process=proc,
            port_probe=lambda h, p: False, sleep=fake_sleep, poll_seconds=1,
        ) is False
        assert elapsed["t"] >= 3

    def test_a_listener_answering_after_our_process_died_is_not_trusted(self):
        # The process we spawned already exited, but SOMETHING answers the
        # port (e.g. an unrelated survivor from a previous run). This must
        # not be reported as "up" -- see start-pads.sh's wait_for_port for
        # the same distinction.
        proc = FakeProc(exit_code=0)
        assert wait_for_port(
            "127.0.0.1", 5555, timeout=10, label="x", process=proc,
            port_probe=lambda h, p: True, sleep=lambda s: None,
        ) is False

    def test_polls_at_the_given_interval_until_the_port_opens(self):
        proc = FakeProc()
        calls = {"n": 0}

        def port_probe(host, port):
            calls["n"] += 1
            return calls["n"] >= 3  # opens on the third probe

        sleeps = []
        assert wait_for_port(
            "127.0.0.1", 5555, timeout=10, label="x", process=proc,
            port_probe=port_probe, sleep=sleeps.append, poll_seconds=1,
        ) is True
        assert len(sleeps) == 2  # slept between probes 1->2 and 2->3, not after


# ---------------------------------------------------------------------------
# play_one_match — the greedy single-episode loop (fake env/policy).
# ---------------------------------------------------------------------------


class FakePolicy:
    def __init__(self, actions):
        self._actions = list(actions)
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1

    def act(self, obs):
        return self._actions.pop(0)


class FakeEnv:
    """Scripted env: reset() once, then step() replays a fixed outcome list."""

    def __init__(self, steps):
        # steps: list of (obs, reward, done, info)
        self._steps = list(steps)
        self.reset_calls = 0

    def reset(self, seed=None):
        self.reset_calls += 1
        return "obs0"

    def step(self, action):
        return self._steps.pop(0)


class TestPlayOneMatch:
    def test_resets_the_policy_and_the_env_exactly_once(self):
        env = FakeEnv([("obs1", 0.0, True, {"won": True})])
        policy = FakePolicy([0])
        play_one_match(env, policy, log=lambda m: None)
        assert env.reset_calls == 1
        assert policy.reset_calls == 1

    def test_loops_until_done_then_reports_a_win(self):
        env = FakeEnv(
            [
                ("obs1", 0.0, False, {}),
                ("obs2", 0.0, False, {}),
                ("obs3", 1.0, True, {"won": True, "lost": False}),
            ]
        )
        policy = FakePolicy([0, 0, 0])
        result = play_one_match(env, policy, log=lambda m: None)
        assert "WIN" in result

    def test_reports_a_loss(self):
        env = FakeEnv([("obs1", -1.0, True, {"won": False, "lost": True})])
        policy = FakePolicy([0])
        result = play_one_match(env, policy, log=lambda m: None)
        assert "LOSS" in result

    def test_a_simultaneous_double_death_is_reported_as_a_loss(self):
        # Mirrors MCPvPEnv.step()'s own tie-break: the learner dying can never
        # count as a win, even if the opponent also died the same step.
        env = FakeEnv([("obs1", 0.0, True, {"won": True, "lost": True})])
        policy = FakePolicy([0])
        result = play_one_match(env, policy, log=lambda m: None)
        assert "LOSS" in result

    def test_tolerates_a_policy_with_no_reset_method(self):
        class NoResetPolicy:
            def act(self, obs):
                return 0

        env = FakeEnv([("obs1", 0.0, True, {"won": True})])
        # Must not raise even though NoResetPolicy has no reset().
        play_one_match(env, NoResetPolicy(), log=lambda m: None)


# ---------------------------------------------------------------------------
# --dry-run: every gate runs, nothing is spawned, exits 0.
# ---------------------------------------------------------------------------


class TestDryRun:
    def test_dry_run_passes_all_gates_and_starts_nothing(self, tmp_path):
        checkpoints_dir = tmp_path
        checkpoint = checkpoints_dir / "ckpt.pt"
        checkpoint.write_bytes(b"unused, load_policy is stubbed")

        messages, log = collector()
        popen = RefusingPopen()

        code = run(
            [
                "--dry-run",
                "--checkpoint",
                str(checkpoint),
                "--checkpoints-dir",
                str(checkpoints_dir),
                "--challenger-username",
                "Steve",
            ],
            popen=popen,
            port_probe=lambda host, port: False,
            load_policy=lambda p, d: object(),
            which=stub_which,
            log=log,
        )

        assert code == 0
        assert popen.calls == []
        text = "\n".join(messages)
        assert "dry run" in text
        assert "--opponent-mode human" in text
        assert "--challenger-username Steve" in text


# ---------------------------------------------------------------------------
# --challenger-username: help text + runtime warning (spec: "make it easy to
# pass, and say so in --help").
# ---------------------------------------------------------------------------


class TestChallengerUsernameGuidance:
    def test_help_text_recommends_pinning_the_name_not_leaving_it_unset(self):
        from deploy.exhibition import _build_arg_parser

        # argparse re-wraps help to the terminal width, so assert against a
        # whitespace-normalized copy rather than the wrapped lines.
        help_text = " ".join(_build_arg_parser().format_help().split())

        assert "--challenger-username" in help_text
        assert "bystander" in help_text

        # WHAT the text recommends, not merely that it mentions the risk. With
        # a colon -- "Strongly recommended: leave this unset and the bridge
        # credits the FIRST non-agent player..." -- the clause after the colon
        # reads as the recommendation, which is the exact inverse of the
        # mitigation this flag exists to provide. A substring check for
        # "bystander" passes either way, which is how that shipped.
        assert (
            "PIN the human opponent's Minecraft username. Strongly recommended."
            in help_text
        )
        assert "recommended: leave this unset" not in help_text
        assert "Pass this before a real exhibition." in help_text

    def test_warns_at_runtime_when_unset(self, tmp_path):
        checkpoint = tmp_path / "ckpt.pt"
        checkpoint.write_bytes(b"unused")
        messages, log = collector()

        run(
            ["--checkpoint", str(checkpoint), "--checkpoints-dir", str(tmp_path), "--dry-run"],
            port_probe=lambda h, p: False,
            load_policy=lambda p, d: object(),
            which=stub_which,
            log=log,
        )

        assert any("no --challenger-username pinned" in m for m in messages)

    def test_no_warning_when_pinned(self, tmp_path):
        checkpoint = tmp_path / "ckpt.pt"
        checkpoint.write_bytes(b"unused")
        messages, log = collector()

        run(
            [
                "--checkpoint", str(checkpoint), "--checkpoints-dir", str(tmp_path),
                "--dry-run", "--challenger-username", "Steve",
            ],
            port_probe=lambda h, p: False,
            load_policy=lambda p, d: object(),
            which=stub_which,
            log=log,
        )

        assert not any("no --challenger-username pinned" in m for m in messages)

    def test_invalid_username_refuses_before_anything_is_spawned(self, tmp_path):
        checkpoint = tmp_path / "ckpt.pt"
        checkpoint.write_bytes(b"unused")
        popen = RefusingPopen()
        messages, log = collector()

        code = run(
            [
                "--checkpoint", str(checkpoint), "--checkpoints-dir", str(tmp_path),
                "--challenger-username", "not a valid mc username!!",
            ],
            popen=popen,
            port_probe=lambda h, p: False,
            load_policy=lambda p, d: object(),
            log=log,
        )

        assert code != 0
        assert popen.calls == []


# ---------------------------------------------------------------------------
# main() — thin wrapper over run().
# ---------------------------------------------------------------------------


class TestMain:
    def test_main_delegates_to_run_and_returns_its_code(self, monkeypatch):
        seen = {}

        def fake_run(argv=None, **kwargs):
            seen["argv"] = argv
            return 7

        monkeypatch.setattr("deploy.exhibition.run", fake_run)
        assert main(["--dry-run"]) == 7
        assert seen["argv"] == ["--dry-run"]


# ---------------------------------------------------------------------------
# Teardown ordering on a boot failure: bridge before Paper, best-effort.
# ---------------------------------------------------------------------------


class TestBootFailureTeardown:
    def test_paper_never_ready_tears_down_and_writes_nothing_extra(self, tmp_path):
        checkpoint = tmp_path / "ckpt.pt"
        checkpoint.write_bytes(b"unused")
        messages, log = collector()
        spawned = []
        ops_writes = []

        def fake_popen(cmd, **kwargs):
            proc = FakeProc(exit_code=1)  # Paper "exits" immediately
            spawned.append(cmd)
            return proc

        def fake_write_ops(n_pads, path):
            # The git-tracked server/ops.json must never be touched by a test;
            # this seam is exactly what keeps this test from writing to it.
            ops_writes.append((n_pads, path))
            return path

        code = run(
            [
                "--checkpoint", str(checkpoint), "--checkpoints-dir", str(tmp_path),
                "--challenger-username", "Steve", "--server-timeout", "0",
                "--log-dir", str(tmp_path / "logs"),
            ],
            popen=fake_popen,
            port_probe=lambda h, p: False,  # Paper's port never opens
            sleep=lambda s: None,
            load_policy=lambda p, d: object(),
            write_ops=fake_write_ops,
            which=stub_which,
            log=log,
        )

        assert code == 1
        assert ops_writes == [(1, str(OPS_JSON))]
        # Only Paper was attempted -- the bridge must never be spawned once
        # Paper itself failed to come up.
        assert len(spawned) == 1
        assert "bash" in spawned[0][0] or spawned[0][0].endswith("bash")
        assert "did not come up in time" in "\n".join(messages)


# ---------------------------------------------------------------------------
# Toolchain preflight — node / bridge/run.js / server/setup/start.sh.
#
# Not enumerated by AC5, but the failure it prevents is the one AC5 exists to
# forbid: without this gate a missing `node` surfaces from the BRIDGE popen,
# which only runs after Paper has already spent 30-60s booting and generating
# a world. "A launcher that boots Paper and then discovers the checkpoint is
# missing has failed the requirement" -- same class, different precondition.
# ---------------------------------------------------------------------------


class TestToolchainPreflight:
    def _drive(self, tmp_path, *, which, port_probe=lambda h, p: False):
        """Run the real run() with everything else passing, and report what it
        refused with plus proof that nothing was started or written."""
        checkpoint = tmp_path / "ckpt.pt"
        checkpoint.write_bytes(b"unused, load_policy is stubbed")
        messages, log = collector()
        popen = RefusingPopen()
        ops_writes = []

        code = run(
            [
                "--checkpoint", str(checkpoint),
                "--checkpoints-dir", str(tmp_path),
                "--challenger-username", "Steve",
                "--log-dir", str(tmp_path / "logs"),
            ],
            popen=popen,
            port_probe=port_probe,
            sleep=lambda s: None,
            load_policy=lambda p, d: object(),
            write_ops=recording_write_ops(ops_writes),
            which=which,
            log=log,
        )
        return code, "\n".join(messages), popen, ops_writes

    def test_missing_node_refuses_before_paper_is_ever_started(self, tmp_path):
        code, text, popen, ops_writes = self._drive(tmp_path, which=lambda name: None)

        assert code != 0
        assert popen.calls == []  # Paper is NOT booted first and asked later
        assert ops_writes == []
        assert "node" in text
        assert "--node" in text  # how to fix it, not just what is wrong
        assert "nothing has been started" in text

    def test_missing_run_js_refuses(self, tmp_path, monkeypatch):
        monkeypatch.setattr("deploy.exhibition.RUN_JS", tmp_path / "gone" / "run.js")

        code, text, popen, ops_writes = self._drive(tmp_path, which=stub_which)

        assert code != 0
        assert popen.calls == []
        assert ops_writes == []
        assert "run.js" in text
        assert "nothing has been started" in text

    def test_missing_start_sh_refuses(self, tmp_path, monkeypatch):
        monkeypatch.setattr("deploy.exhibition.START_SH", tmp_path / "gone" / "start.sh")

        code, text, popen, ops_writes = self._drive(tmp_path, which=stub_which)

        assert code != 0
        assert popen.calls == []
        assert ops_writes == []
        assert "start.sh" in text
        assert "nothing has been started" in text

    def test_every_problem_is_reported_at_once(self, tmp_path, monkeypatch):
        # An operator repairing a demo machine should not rediscover the next
        # missing piece one 30-second Paper boot at a time.
        monkeypatch.setattr("deploy.exhibition.RUN_JS", tmp_path / "gone" / "run.js")
        monkeypatch.setattr("deploy.exhibition.START_SH", tmp_path / "gone" / "start.sh")

        _code, text, _popen, _ops = self._drive(tmp_path, which=lambda name: None)

        assert "node" in text
        assert "run.js" in text
        assert "start.sh" in text

    def test_the_port_gates_still_refuse_first(self, tmp_path):
        # Ordering guard for TC20: the toolchain gate is LAST, so a machine
        # with no Node still reports the port conflict -- the thing the
        # operator must fix before a second launcher can start at all.
        _code, text, _popen, _ops = self._drive(
            tmp_path,
            which=lambda name: None,
            port_probe=lambda h, p: p == DEFAULT_BRIDGE_PORT,
        )

        assert "already in use" in text
        assert "the bridge toolchain does not resolve" not in text

    def test_no_problems_when_everything_resolves(self):
        # Also asserts the two repo-internal paths really are where the module
        # thinks they are -- a rename of either would surface here.
        assert find_toolchain_problems("node", which=stub_which) == []

    def test_reports_the_node_argument_it_was_given(self):
        problems = find_toolchain_problems("/opt/weird/node", which=lambda name: None)
        assert len(problems) == 1
        assert "/opt/weird/node" in problems[0]


# ---------------------------------------------------------------------------
# The happy path — both children boot, the agent plays ONE match over a
# scripted bridge, Ctrl-C ends the exhibition, teardown runs in order.
#
# Every section above this one only ever proved what run() REFUSES to do.
# Nothing past "Paper never comes up" was exercised at all, which left three
# silent mutations available: deleting the play_one_match() call, replacing the
# no-timeout horizon with a large integer, and swapping the teardown order.
# ---------------------------------------------------------------------------


def _reset_ack_msg():
    """A passing read-back gate (``ok=True``), which is what the env waits for
    before it will start an episode."""
    return ResetAckMsg.from_dict(
        {"type": "reset_ack", "ok": True, "readback": {"self_hp": 20.0, "opp_hp": 20.0}}
    )


def _state_msg(*, tick, opp_health=20.0, opponent_died=False, visible=True):
    """One valid ``state`` line: both fighters at full health, no events unless
    a death is being scripted.

    Only the OPPONENT's death is ever scripted here. Under
    ``ExhibitionConfig.no_timeout`` a death is the one and only thing that ends
    an exhibition match (AC4), so it is also the only thing worth scripting.

    ``visible`` places the opponent two blocks in FRONT of the learner
    (``visible=True``, the historical default -- inside the frozen 70-degree
    FOV cone, see ``env/perception_filter.py``'s ``FOV_DEGREES``) or two blocks
    directly BEHIND it (``visible=False``, outside the cone). At self yaw 0.0
    the look vector is +Z, so +2 on z is dead ahead and -2 is dead behind --
    genuinely gated out by the REAL ``PerceptionFilter`` the env runs, not a
    hand-set ``visible`` bit. This is what a human "circling behind" the agent
    (the T7 reflex shield's whole reason to exist) looks like on the wire.
    """
    opp_z = 2.0 if visible else -2.0
    return StateMsg.from_dict(
        {
            "type": "state",
            "self": {
                "pos": [0.0, 64.0, 0.0],
                "yaw": 0.0,
                "pitch": 0.0,
                "velocity": [0.0, 0.0, 0.0],
                "on_ground": True,
                "health": 20.0,
                "held_item": "iron_sword",
                "attack_cooldown": 1.0,
            },
            "opponent": {
                "pos": [0.0, 64.0, opp_z],
                "yaw": 0.0,
                "pitch": 0.0,
                "velocity": [0.0, 0.0, 0.0],
                "health": opp_health,
            },
            "events": {
                "damage_dealt": 0.0,
                "damage_taken": 0.0,
                "i_died": False,
                "opponent_died": opponent_died,
            },
            "arena": {"wall_distances": [8.0, 8.0, 8.0, 8.0]},
            "tick": tick,
            "code_version": "test",
        }
    )


class ScriptedBridgeTransport:
    """A fake ``BridgeTransport`` that answers the real wire protocol.

    Injected through run()'s ``transport_factory`` seam, so the launcher builds
    a REAL :class:`~env.mc_pvp_env.MCPvPEnv` over it. That is what makes the
    horizon assertion meaningful: it is made against the env the launcher
    actually constructed, not against a stand-in that could disagree with it.

    Replies are produced inside ``send()`` rather than queued up front, so the
    inbound queue can never drift out of step with what the env asked for.
    ``close`` gets no reply -- ``MCPvPEnv.close()`` sends it and never reads.

    ``steps_sent`` is PER EPISODE and is re-zeroed by ``reset`` (T6). Left
    cumulative, the second match's very FIRST step would satisfy
    ``steps_sent >= steps_to_win`` and end instantly, so every re-drive test
    would pass while proving nothing about the match actually being played.

    ``max_resets`` is a HANG GUARD, and it earned its place: a launcher that
    consumed a reset request without removing it replays a match forever, and
    a runaway loop makes the whole pytest session hang instead of failing one
    test -- the same "green for the wrong reason" class as an escaping
    KeyboardInterrupt. Exceeding it raises, which unwinds into run()'s fatal
    handler and fails the assertions loudly.

    ``opponent_visible`` (T7) is an optional ``obs_index -> bool`` schedule,
    1-indexed to match DECISION numbering within the current match: index 1 is
    what the post-reset ``state`` carries (the observation ``play_one_match``'s
    first decision acts on), index 2 is what the FIRST ``step`` reply carries
    (the observation the SECOND decision acts on), and so on. Defaults to
    "always visible", i.e. byte-identical to every test that predates T7.
    """

    def __init__(
        self,
        *,
        steps_to_win,
        record,
        interrupt_on_close=False,
        on_step=None,
        max_resets=8,
        opponent_visible=None,
    ):
        self._steps_to_win = steps_to_win
        self._record = record
        self._interrupt_on_close = interrupt_on_close
        self._on_step = on_step
        self._max_resets = max_resets
        self._opponent_visible = opponent_visible or (lambda obs_index: True)
        self._inbound = []
        self._tick = 0
        self.steps_sent = 0
        self.total_steps = 0
        self.resets = 0
        self.connects = 0
        self.closes = 0
        #: The `action` field of every "step" message sent, in order, across
        #: every match this transport has played (T7: what the reflex shield
        #: actually put on the wire, as opposed to what the policy chose).
        self.actions_sent = []

    def connect(self):
        self.connects += 1
        self._record(("transport.connect",))

    def send(self, obj):
        kind = obj["type"]
        self._record(("bridge.send", kind))
        self._tick += 1
        if kind == "reset":
            self.resets += 1
            if self.resets > self._max_resets:
                raise AssertionError(
                    f"the launcher started {self.resets} matches; this test "
                    f"scripted at most {self._max_resets}. Something is "
                    "re-driving play without a reset request."
                )
            self.steps_sent = 0
            self._inbound.append(_reset_ack_msg())
            self._inbound.append(
                _state_msg(tick=self._tick, visible=self._opponent_visible(1))
            )
        elif kind == "step":
            self.actions_sent.append(obj.get("action"))
            self.steps_sent += 1
            self.total_steps += 1
            if self._on_step is not None:
                # The hook is how a test simulates something happening WHILE a
                # match is in flight (an operator filing a reset request mid-
                # match, say) -- the launcher is inside play_one_match there and
                # is not polling anything.
                self._on_step(self)
            killed = self.steps_sent >= self._steps_to_win
            # This reply feeds the NEXT decision, i.e. obs index `steps_sent + 1`
            # (index 1 was the post-reset state, consumed by decision 1).
            self._inbound.append(
                _state_msg(
                    tick=self._tick,
                    opp_health=0.0 if killed else 20.0,
                    opponent_died=killed,
                    visible=self._opponent_visible(self.steps_sent + 1),
                )
            )
        elif kind == "close" and self._interrupt_on_close:
            # A Ctrl-C landing on the FIRST link of the teardown chain.
            raise KeyboardInterrupt

    def recv(self):
        if not self._inbound:
            raise BridgeError("scripted bridge: recv() on an empty queue")
        return self._inbound.pop(0)

    def close(self):
        self.closes += 1
        self._record(("transport.close",))


class RecordingGreedyPolicy:
    """Stands in for ``eval.evaluate.DRQNGreedyPolicy``: the same
    ``reset()``/``act(obs)`` surface, no torch. Always returns macro 0, a valid
    index that the real env will accept."""

    def __init__(self, record):
        self._record = record
        self.reset_calls = 0
        self.act_calls = 0

    def reset(self):
        self.reset_calls += 1
        self._record(("policy.reset",))

    def act(self, obs):
        self.act_calls += 1
        return 0


class ExhibitionRun:
    """Everything run()'s seams recorded during one launch, plus the exit code.

    ``events`` is the single ordered list every seam appends to -- which is
    what lets one assertion pin write-before-boot, spawn order, the match
    actually being played, and teardown order all at once.

    ``escaped`` holds whatever ``run()`` let propagate instead of returning an
    exit code. It is always a bug -- ``run()`` is the supervisor -- and it is
    recorded rather than propagated for a practical reason: a KeyboardInterrupt
    escaping into pytest ABORTS the session ("56 passed", no failures) instead
    of failing the test that caught the regression.
    """

    def __init__(self):
        self.code = None
        self.escaped = None
        self.events = []
        self.messages = []
        #: What the SEPARATE ``--reset`` invocations printed. A different
        #: process in real life, so a different sink here.
        self.reset_messages = []
        self.reset_codes = []
        self.ops_writes = []
        self.procs = {}
        self.spawn_kwargs = {}
        self.envs = []
        self.transports = []
        self.policy = None
        self.request_path = None

    @property
    def console_lines(self):
        """Every line written to Paper's console pipe, in order."""
        paper = self.procs.get("paper")
        console = getattr(paper, "stdin", None) if paper is not None else None
        return list(console.lines) if console is not None else []

    def text(self):
        return "\n".join(self.messages)


def drive_exhibition(
    tmp_path,
    monkeypatch,
    *,
    steps_to_win=3,
    proc_classes=None,
    interrupt_on_close=False,
    log_bomb=None,
    resets=0,
    idle_wakeups=0,
    challenger="Steve",
    extra_argv=(),
    console_fails=False,
    on_step=None,
    opponent_visible=None,
):
    """Drive the REAL ``run()`` through a complete exhibition and record it.

    Spawning a child "opens" its port, so both boot on their first probe and
    ``wait_for_port`` never sleeps. That leaves the between-matches wait as the
    only sleeper.

    ``resets`` is how many times the OPERATOR runs the reset command. Each one
    is delivered from inside a ``sleep`` — the launcher is blocked there,
    exactly as it is in production — and it goes through the REAL ``--reset``
    branch of ``run()``, not a hand-written file touch, so both halves of T6
    are the code under test. Once they are used up, the next ``sleep`` raises
    ``KeyboardInterrupt``: the normal way an exhibition ends.

    ``MCPvPEnv`` is WRAPPED, not replaced: the launcher builds the real class
    over the scripted transport and the wrapper only keeps a reference, so a
    test can assert what the env was built with.

    ``log_bomb`` makes the ``log`` seam raise ``KeyboardInterrupt`` on the
    first message containing that substring — an interrupt landing somewhere a
    teardown helper's own handler cannot absorb it, which is what the nested
    ``finally`` chain (rather than a flat one) exists to survive.

    ``opponent_visible`` (T7) forwards straight to
    :class:`ScriptedBridgeTransport`'s schedule of the same name, so a test can
    script the REAL ``MCPvPEnv``/``PerceptionFilter`` into gating the opponent
    out of view for specific decisions, exactly as a human circling behind the
    agent would.
    """
    proc_classes = dict(proc_classes or {})
    result = ExhibitionRun()
    record = result.events.append
    ports = {"mc": False, "bridge": False}
    log_dir = tmp_path / "logs"
    result.request_path = reset_request_path(log_dir)

    def port_probe(host, port):
        if port == DEFAULT_MC_PORT:
            return ports["mc"]
        if port == DEFAULT_BRIDGE_PORT:
            return ports["bridge"]
        raise AssertionError(f"unexpected port probe: {host}:{port}")

    def popen(cmd, **kwargs):
        # start.sh is invoked as `bash <path>`; anything else is the bridge.
        label = "paper" if cmd[0] == "bash" else "bridge"
        ports["mc" if label == "paper" else "bridge"] = True
        record(("spawn", label))
        result.spawn_kwargs[label] = kwargs
        # Popen only exposes a .stdin stream when the caller asked for a PIPE.
        console = (
            FakeConsole(fail=console_fails, record=record)
            if kwargs.get("stdin") is subprocess.PIPE
            else None
        )
        proc = proc_classes.get(label, FakeProc)(
            pid=101 if label == "paper" else 202,
            label=label,
            record=record,
            stdin=console,
        )
        result.procs[label] = proc
        return proc

    def transport_factory(host, port):
        record(("transport_factory", host, port))
        transport = ScriptedBridgeTransport(
            steps_to_win=steps_to_win,
            record=record,
            interrupt_on_close=interrupt_on_close,
            on_step=on_step,
            # One match at launch plus one per reset command is the MOST this
            # launcher may play. Anything beyond it is a runaway loop.
            max_resets=1 + resets,
            opponent_visible=opponent_visible,
        )
        result.transports.append(transport)
        return transport

    def write_ops(n_pads, path):
        record(("write_ops", n_pads))
        result.ops_writes.append((n_pads, path))
        return path

    pending = {"resets": resets, "wakeups": idle_wakeups}

    def interrupting_sleep(seconds):
        if pending["wakeups"] > 0:
            # A poll that found nothing: the launcher woke up, nobody had run
            # the reset command, and it must go straight back to waiting.
            pending["wakeups"] -= 1
            record(("idle_poll",))
            return
        if pending["resets"] > 0:
            pending["resets"] -= 1
            # The operator, in ANOTHER terminal, runs the real reset command.
            code = run(["--reset", "--log-dir", str(log_dir)], log=result.reset_messages.append)
            result.reset_codes.append(code)
            record(("reset_command", code))
            return
        raise KeyboardInterrupt

    real_env_cls = MCPvPEnv

    def capturing_env(**kwargs):
        env = real_env_cls(**kwargs)
        result.envs.append(env)
        return env

    monkeypatch.setattr("deploy.exhibition.MCPvPEnv", capturing_env)
    # The join banner calls _detect_lan_ip(), which opens a real UDP socket.
    # This file opens no sockets at all.
    monkeypatch.setattr("deploy.exhibition._detect_lan_ip", lambda: "192.168.1.50")

    armed = {"bomb": log_bomb is not None}

    def log(message):
        result.messages.append(message)
        if armed["bomb"] and log_bomb in message:
            armed["bomb"] = False
            raise KeyboardInterrupt

    checkpoint = tmp_path / "ckpt.pt"
    checkpoint.write_bytes(b"unused, load_policy is stubbed")
    result.policy = RecordingGreedyPolicy(record)

    launch_argv = [
        "--checkpoint", str(checkpoint),
        "--checkpoints-dir", str(tmp_path),
        "--log-dir", str(log_dir),
    ]
    if challenger is not None:
        launch_argv.extend(["--challenger-username", challenger])
    launch_argv.extend(extra_argv)

    try:
        result.code = run(
            launch_argv,
            popen=popen,
            port_probe=port_probe,
            sleep=interrupting_sleep,
            load_policy=lambda p, d: result.policy,
            transport_factory=transport_factory,
            write_ops=write_ops,
            which=stub_which,
            log=log,
        )
    except BaseException as exc:  # noqa: BLE001 — see ExhibitionRun.escaped.
        result.escaped = exc
    return result


class TestHappyPath:
    def test_the_whole_launch_sequence_happens_in_order(self, tmp_path, monkeypatch):
        result = drive_exhibition(tmp_path, monkeypatch, steps_to_win=3)

        # ops.json before Paper (it reads the op list at startup), Paper before
        # the bridge (the bots need a server to join), the transport only once
        # both are up, one match, then teardown inside-out.
        assert result.events == [
            ("write_ops", 1),
            ("spawn", "paper"),
            ("spawn", "bridge"),
            ("transport_factory", DEFAULT_BRIDGE_HOST, DEFAULT_BRIDGE_PORT),
            ("transport.connect",),
            ("policy.reset",),
            ("bridge.send", "reset"),
            ("bridge.send", "step"),
            ("bridge.send", "step"),
            ("bridge.send", "step"),
            ("bridge.send", "close"),
            ("transport.close",),
            ("terminate", "bridge"),
            ("terminate", "paper"),
        ]

    def test_ctrl_c_at_the_idle_wait_exits_130(self, tmp_path, monkeypatch):
        result = drive_exhibition(tmp_path, monkeypatch)

        # 130 is the NORMAL end of an exhibition, not a failure: the operator
        # ends it from the keyboard once the last challenger is done.
        assert result.escaped is None
        assert result.code == 130
        assert any("interrupted" in m for m in result.messages)

    def test_the_env_is_built_with_no_episode_horizon(self, tmp_path, monkeypatch):
        result = drive_exhibition(tmp_path, monkeypatch)

        (env,) = result.envs
        # None is the ONE form "disabled" takes (plan Contracts / AC4). Not a
        # sentinel and emphatically not a large integer -- a large integer is a
        # timeout that has merely been moved somewhere less convenient to
        # notice, and it fires mid-match in front of an audience.
        assert env.max_episode_steps is None

    def test_the_match_is_actually_played_to_a_death(self, tmp_path, monkeypatch):
        result = drive_exhibition(tmp_path, monkeypatch, steps_to_win=4)

        (transport,) = result.transports
        assert result.policy.reset_calls == 1
        assert result.policy.act_calls == 4
        assert transport.steps_sent == 4

        text = "\n".join(result.messages)
        assert "match finished after 4 decision step(s)" in text
        assert "AGENT WIN" in text

    def test_exactly_one_match_is_played_with_no_auto_restart(self, tmp_path, monkeypatch):
        result = drive_exhibition(tmp_path, monkeypatch)

        # AC4: the match ending reports a result and idles. A second `reset` on
        # the wire would mean the launcher restarted the match by itself.
        assert result.events.count(("bridge.send", "reset")) == 1
        assert len(result.envs) == 1

    def test_the_join_instructions_are_printed_before_the_match(self, tmp_path, monkeypatch):
        result = drive_exhibition(tmp_path, monkeypatch)

        anchor = pad_anchor(0)
        join_idx = next(
            i for i, m in enumerate(result.messages) if "192.168.1.50:25565" in m
        )
        arena_idx = next(
            i for i, m in enumerate(result.messages) if f"({anchor.x}, {anchor.z})" in m
        )
        match_idx = next(
            i for i, m in enumerate(result.messages) if "match finished" in m
        )

        # AC5: the LAN address a classmate types and where the arena is -- and
        # BEFORE the match, not after it. Printed afterwards they are a
        # transcript of an exhibition nobody could join.
        assert join_idx < match_idx
        assert arena_idx < match_idx

    def test_teardown_stops_the_bridge_before_paper(self, tmp_path, monkeypatch):
        result = drive_exhibition(tmp_path, monkeypatch)

        # Order, not just "both were stopped": killing Paper first leaves the
        # bridge's mineflayer bots thrashing against a dead server.
        assert [e for e in result.events if e[0] == "terminate"] == [
            ("terminate", "bridge"),
            ("terminate", "paper"),
        ]

    def test_ops_json_is_written_once_for_this_single_pad(self, tmp_path, monkeypatch):
        result = drive_exhibition(tmp_path, monkeypatch)

        assert result.ops_writes == [(1, str(OPS_JSON))]


class TestTeardownSurvivesASecondCtrlC:
    """W1: ``KeyboardInterrupt`` derives from ``BaseException``, so an
    ``except Exception`` teardown lets a second Ctrl-C escape mid-chain and
    orphan whatever had not been stopped yet."""

    def test_a_second_ctrl_c_in_the_grace_wait_still_stops_paper(
        self, tmp_path, monkeypatch
    ):
        result = drive_exhibition(
            tmp_path, monkeypatch, proc_classes={"bridge": SlowStoppingProc}
        )

        paper = result.procs["paper"]
        bridge = result.procs["bridge"]

        # THE assertion: without a BaseException-proof chain this is False and
        # a Paper JVM is left squatting on 25565, which then makes the next
        # launch refuse on its own mc-port gate.
        assert paper.terminated is True
        # The interrupted child is not left half-stopped either: the grace wait
        # is what got interrupted, so it goes straight to SIGKILL.
        assert bridge.terminated is True
        assert bridge.killed is True
        # The second interrupt is absorbed, not merely survived: it neither
        # escapes the supervisor nor clobbers the pending exit code.
        assert result.escaped is None
        assert result.code == 130

    def test_a_ctrl_c_while_closing_the_env_still_stops_both_children(
        self, tmp_path, monkeypatch
    ):
        # env.close() is the FIRST link of the chain, so an escape here strands
        # both children rather than only Paper.
        result = drive_exhibition(tmp_path, monkeypatch, interrupt_on_close=True)

        assert result.procs["bridge"].terminated is True
        assert result.procs["paper"].terminated is True
        assert result.escaped is None
        assert result.code == 130
        assert any("continuing teardown" in m for m in result.messages)

    def test_an_interrupt_no_helper_can_absorb_still_reaches_every_link(
        self, tmp_path, monkeypatch
    ):
        # The chain is NESTED, not sequential. Here the interrupt lands on the
        # log call inside _close_env's own handler, so no helper can absorb it
        # and it unwinds through the whole finally block. Nested, each later
        # link still runs; flat, this one raise strands BOTH children.
        result = drive_exhibition(
            tmp_path,
            monkeypatch,
            interrupt_on_close=True,
            log_bomb="continuing teardown",
        )

        assert result.procs["bridge"].terminated is True
        assert result.procs["paper"].terminated is True
        # Nothing could have caught this one, so it does escape -- the point is
        # only that it escaped from the END of the chain, not the middle.
        assert isinstance(result.escaped, KeyboardInterrupt)


class TestStopProcessInterruptProofing:
    """Unit-level counterpart to the end-to-end tests above."""

    def test_a_keyboard_interrupt_in_the_grace_wait_is_absorbed(self):
        from deploy.exhibition import _stop_process

        messages, log = collector()
        proc = SlowStoppingProc(pid=7, label="bridge")

        escaped = call_capturing_escape(_stop_process, proc, "bridge", log=log)

        assert escaped is None
        # The wait is what got interrupted, so waiting again is not the answer.
        assert proc.killed is True
        assert any("KeyboardInterrupt" in m for m in messages)

    def test_an_already_exited_process_is_left_alone(self):
        from deploy.exhibition import _stop_process

        messages, log = collector()
        proc = FakeProc(exit_code=0)

        _stop_process(proc, "bridge", log=log)

        assert proc.terminated is False
        assert proc.killed is False

    def test_close_env_absorbs_a_keyboard_interrupt(self):
        from deploy.exhibition import _close_env

        class InterruptingEnv:
            def close(self):
                raise KeyboardInterrupt

        messages, log = collector()

        escaped = call_capturing_escape(_close_env, InterruptingEnv(), log=log)

        assert escaped is None
        # `!r`, not `!s`: a bare KeyboardInterrupt stringifies to "" and would
        # otherwise log as an error message with nothing after the colon.
        assert any("KeyboardInterrupt" in m for m in messages)

    def test_close_env_absorbs_an_ordinary_exception_too(self):
        from deploy.exhibition import _close_env

        class BrokenEnv:
            def close(self):
                raise RuntimeError("socket already gone")

        messages, log = collector()

        _close_env(BrokenEnv(), log=log)

        assert any("socket already gone" in m for m in messages)


# ===========================================================================
# T6 — the separate reset command (AC5), and the AC4 guarantees around it.
# ===========================================================================


# ---------------------------------------------------------------------------
# The request file: the whole trigger mechanism, as pure helpers.
# ---------------------------------------------------------------------------


class TestResetRequestFile:
    def test_the_path_is_the_shared_filename_inside_the_log_dir(self, tmp_path):
        # The launcher and the --reset process are separate invocations with
        # nothing in common but --log-dir, so this derivation is the only thing
        # that makes them agree on where the request lives.
        assert reset_request_path(tmp_path) == tmp_path / RESET_REQUEST_FILENAME

    def test_taking_a_missing_request_is_false_not_an_error(self, tmp_path):
        assert take_reset_request(tmp_path / RESET_REQUEST_FILENAME) is False

    def test_a_request_is_taken_exactly_once_and_the_file_is_gone(self, tmp_path):
        path = tmp_path / RESET_REQUEST_FILENAME
        path.write_text("reset requested\n")

        assert take_reset_request(path) is True
        # Consumed, not merely observed: a request that survived being taken
        # would re-fire at the end of the very match it just started, which is
        # an auto-restart with extra steps.
        assert not path.exists()
        assert take_reset_request(path) is False

    def test_draining_reports_and_removes_a_pending_request(self, tmp_path):
        path = tmp_path / RESET_REQUEST_FILENAME
        path.write_text("reset requested\n")
        messages, log = collector()

        assert drain_reset_request(path, "for a reason", log=log) is True
        assert not path.exists()
        # Never silent: the operator typed that command and is waiting on it.
        assert any("discarded a reset request for a reason" in m for m in messages)

    def test_draining_nothing_says_nothing(self, tmp_path):
        messages, log = collector()

        assert drain_reset_request(tmp_path / "nope", "for a reason", log=log) is False
        assert messages == []

    def test_waiting_polls_until_the_request_appears_then_consumes_it(self, tmp_path):
        path = tmp_path / RESET_REQUEST_FILENAME
        sleeps = []

        def sleep(seconds):
            sleeps.append(seconds)
            if len(sleeps) == 3:  # the operator finally runs the reset command
                path.write_text("reset requested\n")

        wait_for_reset_request(path, sleep=sleep, poll_seconds=0.5)

        assert sleeps == [0.5, 0.5, 0.5]
        assert not path.exists()

    def test_the_hint_names_a_non_default_log_dir(self, tmp_path):
        # The reset process finds the request file by --log-dir alone. A hint
        # that dropped a non-default one would send the operator's request
        # somewhere no launcher is looking, with no error at either end.
        hint = reset_command_hint(tmp_path)
        assert hint == f"python -m deploy.exhibition --reset --log-dir {tmp_path}"

    def test_the_hint_stays_short_for_the_default_log_dir(self):
        assert reset_command_hint(DEFAULT_LOG_DIR) == "python -m deploy.exhibition --reset"


# ---------------------------------------------------------------------------
# `--reset`, driven through the real run() — it files a request and starts
# NOTHING. Not a checkpoint load, not a port probe, not a process, not
# ops.json, and not even the log dir.
# ---------------------------------------------------------------------------


def exploding_load_policy(_checkpoint_path, _checkpoints_dir):
    raise AssertionError("--reset must not load a checkpoint")


def exploding_probe(host, port):
    raise AssertionError("--reset must not probe any port")


class TestResetCommand:
    def _drive_reset(self, log_dir, *extra):
        messages, log = collector()
        popen = RefusingPopen()
        ops_writes = []
        code = run(
            ["--reset", "--log-dir", str(log_dir), *extra],
            popen=popen,
            port_probe=exploding_probe,
            load_policy=exploding_load_policy,
            write_ops=recording_write_ops(ops_writes),
            which=lambda name: None,
            log=log,
        )
        return code, "\n".join(messages), popen, ops_writes

    def test_files_a_request_and_starts_nothing(self, tmp_path):
        log_dir = tmp_path / "logs"
        log_dir.mkdir()

        code, text, popen, ops_writes = self._drive_reset(log_dir)

        assert code == 0
        assert (log_dir / RESET_REQUEST_FILENAME).is_file()
        # The refusal-proof trio: the checkpoint gate, the port gates and the
        # spawn are all AFTER this branch returns. exploding_load_policy and
        # exploding_probe would have raised; RefusingPopen would have raised.
        assert popen.calls == []
        assert ops_writes == []
        assert str(log_dir / RESET_REQUEST_FILENAME) in text
        # It must say plainly that ONE reset means ONE match.
        assert "one reset command, one match" in text

    def test_never_connects_to_the_bridge(self, tmp_path):
        # THE structural constraint: the bridge accepts exactly one TCP client
        # and a second connect destroys the first, so a --reset that spoke the
        # wire would evict the live agent mid-exhibition. exploding_probe
        # covers the port; this pins the reasoning in the refusal text an
        # operator reads when it cannot find a launcher.
        code, text, _popen, _ops = self._drive_reset(tmp_path / "not-a-dir")

        assert code != 0
        assert "ONE TCP client" in text

    def test_refuses_when_no_launcher_is_using_that_log_dir(self, tmp_path):
        missing = tmp_path / "logs"

        code, text, popen, ops_writes = self._drive_reset(missing)

        assert code != 0
        assert popen.calls == []
        assert ops_writes == []
        # It must NOT create the directory: its absence is the evidence that no
        # launcher is using it, and creating one destroys that evidence (and
        # leaves a request nobody will ever consume).
        assert not missing.exists()
        assert "does not exist" in text
        assert "python -m deploy.exhibition" in text

    def test_refuses_a_second_request_without_disturbing_the_first(self, tmp_path):
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        request = log_dir / RESET_REQUEST_FILENAME
        request.write_text("the first request\n")

        code, text, _popen, _ops = self._drive_reset(log_dir)

        assert code != 0
        assert request.read_text() == "the first request\n"  # not overwritten
        assert "already armed" in text
        # And it explains the mid-match case rather than leaving the operator to
        # guess why nothing consumed it.
        assert "DISCARDED" in text

    def test_refuses_flags_it_cannot_honor_instead_of_ignoring_them(self, tmp_path):
        log_dir = tmp_path / "logs"
        log_dir.mkdir()

        code, text, _popen, _ops = self._drive_reset(log_dir, "--challenger-username", "Bob")

        assert code != 0
        # Silently ignoring it would look, to the operator, like the challenger
        # had been swapped for the next match. It cannot be: the name is pinned
        # into the bridge's argv at LAUNCH.
        assert "--challenger-username" in text
        assert not (log_dir / RESET_REQUEST_FILENAME).exists()

    def test_request_reset_writes_a_timestamped_marker(self, tmp_path):
        messages, log = collector()

        assert request_reset(tmp_path, log=log, now=lambda: "2026-08-16 12:00:00") == 0

        assert (tmp_path / RESET_REQUEST_FILENAME).read_text() == (
            "reset requested 2026-08-16 12:00:00\n"
        )


class TestResetModeConflicts:
    def _args(self, argv):
        from deploy.exhibition import _build_arg_parser

        return _build_arg_parser().parse_args(argv)

    def test_log_dir_alone_is_no_conflict(self, tmp_path):
        defaults = self._args([])
        assert reset_mode_conflicts(self._args(["--reset", "--log-dir", str(tmp_path)]), defaults) == []

    def test_every_launch_only_flag_is_reported(self):
        defaults = self._args([])
        conflicts = reset_mode_conflicts(
            self._args(["--reset", "--checkpoint", "x.pt", "--mc-port", "1234"]), defaults
        )
        assert conflicts == ["--checkpoint", "--mc-port"]

    def test_the_store_false_flag_is_named_the_way_it_is_typed(self):
        # dest is `paper_console`, so the mechanical dest->flag rendering would
        # print "--paper-console", a flag that does not exist.
        defaults = self._args([])
        conflicts = reset_mode_conflicts(self._args(["--reset", "--no-paper-console"]), defaults)
        assert conflicts == ["--no-paper-console"]


# ---------------------------------------------------------------------------
# human_reset_commands — the half of the reset the datapack does not do.
# ---------------------------------------------------------------------------


class TestHumanResetCommands:
    def test_the_exact_commands_for_pad_zero(self):
        assert human_reset_commands(pad_anchor(0), "Steve") == [
            "tp Steve 3.5 64 0.5 90 0",
            "effect clear Steve",
            "effect give Steve minecraft:instant_health 1 9 true",
            "effect give Steve minecraft:saturation 1 19 true",
            "clear Steve minecraft:iron_sword",
            "give Steve minecraft:iron_sword 1",
        ]

    def test_the_challenger_is_armed_with_a_sword(self):
        # The bug this pair exists to kill: the datapack re-arms the LEARNER on
        # every reset (spawn_learner_pad.mcfunction) and nothing ever armed the
        # human, so a demo-day classmate punched for 1 damage into an iron
        # sword's 6. A reset that heals but does not arm is the old bug back.
        commands = human_reset_commands(pad_anchor(0), "Steve")
        assert "give Steve minecraft:iron_sword 1" in commands

    def test_the_gear_clear_is_scoped_to_the_sword_and_comes_before_the_give(self):
        commands = human_reset_commands(pad_anchor(0), "Steve")
        clear_idx = commands.index("clear Steve minecraft:iron_sword")
        give_idx = commands.index("give Steve minecraft:iron_sword 1")
        # ORDER: clear THEN give, or the clear eats the sword it was supposed to
        # make room for and the challenger fights bare-fisted anyway.
        assert clear_idx < give_idx
        # SCOPED, both ways. Widening it to a blanket `clear Steve` would empty
        # a person's inventory, which is not "heal and reposition"; dropping it
        # for a lone `give` would pile a sword per reset into their hotbar. The
        # narrow clear is the whole reason the give is idempotent.
        assert "clear Steve" not in commands
        assert not any(c.startswith("clear ") and "iron_sword" not in c for c in commands)

    def test_no_command_carries_a_leading_slash(self):
        # Console commands are slash-free; formatHumanResetCommands' output is
        # NOT, because the bridge feeds it to bot.chat(). Copying that form here
        # is the obvious "consistency" fix and it would send `/tp ...` to a
        # console that has no idea what that is.
        for command in human_reset_commands(pad_anchor(0), "Steve"):
            assert not command.startswith("/")

    def test_the_clear_comes_before_the_gives(self):
        commands = human_reset_commands(pad_anchor(0), "Steve")
        clear_idx = next(i for i, c in enumerate(commands) if c.startswith("effect clear"))
        give_idx = next(i for i, c in enumerate(commands) if c.startswith("effect give"))
        # An instant effect applies on its first tick, so a clear issued after
        # the give can strip it before it ever lands -- the datapack's ordering
        # note owns this reasoning and this is the same trap.
        assert clear_idx < give_idx

    def test_a_non_zero_pad_anchor_still_lands_on_the_opponent_slot(self):
        anchor = pad_anchor(3)
        commands = human_reset_commands(anchor, "Steve")
        assert commands[0] == (
            f"tp Steve {anchor.x + 3}.5 64 {anchor.z}.5 90 0"
        )

    def test_a_username_that_could_inject_a_second_command_is_refused(self):
        # This text is executed by a LEVEL-4 console. A newline in the name is a
        # free command of the attacker's choosing.
        with pytest.raises(ValueError):
            human_reset_commands(pad_anchor(0), "a\nop b")
        with pytest.raises(ValueError):
            human_reset_commands(pad_anchor(0), "Steve Jobs")

    def test_an_empty_or_missing_username_is_refused(self):
        with pytest.raises(ValueError):
            human_reset_commands(pad_anchor(0), "")
        with pytest.raises(ValueError):
            human_reset_commands(pad_anchor(0), None)

    def test_a_negative_anchor_is_refused(self):
        from distributed.launcher import PadAnchor

        # `<n>.5` is a textual concatenation exactly like the datapack's
        # `$(x).5`, so a negative anchor would silently yield anchor MINUS half
        # a block -- the same trap arena:setup_pad's header documents.
        with pytest.raises(ValueError):
            human_reset_commands(PadAnchor(x=-1, z=0), "Steve")

    def test_they_still_mirror_the_committed_dummy_datapack(self):
        """Drift pin. These commands duplicate the dummy's reset template for a
        player instead of a bot; if that template moves, this fails here rather
        than in a live exhibition where the human quietly spawns in the wrong
        place or at the wrong health."""
        datapack = DUMMY_PAD_MCFUNCTION.read_text(encoding="utf-8").splitlines()

        # Position: anchor + 3 on x, y=64, z centre, yaw 90 (facing the
        # learner). The datapack expresses the +3 as a relative hop from the
        # learner cell; CHALLENGER_SPAWN_DX is the same 3.
        assert "$execute positioned $(x).5 64 $(z).5 run tp $(dummy) ~3 ~ ~ 90 0" in datapack
        # Health/food: same effects, same amplifiers, same clear-then-give.
        assert "$effect clear $(dummy)" in datapack
        assert "$effect give $(dummy) minecraft:instant_health 1 9 true" in datapack
        assert "$effect give $(dummy) minecraft:saturation 1 19 true" in datapack

        ours = human_reset_commands(pad_anchor(0), "Steve")
        for line in datapack:
            if line.startswith("$effect "):
                assert line[1:].replace("$(dummy)", "Steve") in ours

        # Gear is the ONE place the dummy is the wrong model: it is given a
        # blanket `$clear` and no weapon at all, which is right for a passive
        # training target and wrong for a person facing an armed agent.
        assert "$clear $(dummy)" in datapack
        assert not any(line.startswith("$give ") for line in datapack)
        assert "clear Steve" not in ours

    def test_the_gear_matches_what_the_committed_datapack_hands_the_learner(self):
        """Drift pin for the symmetry claim. The challenger's sword exists only
        because the learner gets one on every reset; if that line is re-geared
        (a diamond sword, armor, nothing at all) the fight silently stops being
        symmetric, so it fails here instead of in front of a room."""
        datapack = LEARNER_PAD_MCFUNCTION.read_text(encoding="utf-8").splitlines()

        give_lines = [line for line in datapack if line.startswith("$give ")]
        assert give_lines == ["$give $(learner) minecraft:iron_sword 1"]

        ours = human_reset_commands(pad_anchor(0), "Steve")
        assert give_lines[0][1:].replace("$(learner)", "Steve") in ours
        # And no armor on either side: the checkpoint never trained against an
        # armored opponent, so armor is out of scope by decision, not oversight.
        # Command lines only (`$...`); a comment may discuss armor freely.
        commands_in_datapack = [line for line in datapack if line.startswith("$")]
        assert not any("armor" in line for line in commands_in_datapack)
        assert not any("armor" in command for command in ours)


# ---------------------------------------------------------------------------
# The Paper console channel — the only command path this process can reach.
# ---------------------------------------------------------------------------


class TestPaperConsole:
    def test_commands_are_written_as_bytes_lines_and_flushed(self):
        console = FakeConsole()
        proc = FakeProc(stdin=console)
        messages, log = collector()

        assert send_paper_console_commands(proc, ["say one", "say two"], log=log) is True

        assert console.lines == ["say one", "say two"]
        assert console.flushes == 1
        # Echoed, always: a channel that quietly did nothing is this project's
        # signature failure mode, and an opped operator can re-run these by hand.
        assert any("console> say one" in m for m in messages)

    def test_no_pipe_is_reported_with_the_commands_that_did_not_run(self):
        proc = FakeProc(stdin=None)  # DEVNULL opens no pipe
        messages, log = collector()

        assert send_paper_console_commands(proc, ["say one"], log=log) is False

        text = "\n".join(messages)
        assert "did NOT run" in text
        assert "say one" in text

    def test_a_broken_pipe_is_survivable_and_reported(self):
        # Paper died mid-exhibition. A launcher that took the whole exhibition
        # down over a failed heal would be worse than one that plays unhealed.
        proc = FakeProc(stdin=FakeConsole(fail=True))
        messages, log = collector()

        assert send_paper_console_commands(proc, ["say one"], log=log) is False

        text = "\n".join(messages)
        assert "BrokenPipeError" in text
        assert "did NOT run" in text


# ---------------------------------------------------------------------------
# End to end: one reset command -> one more match, in the SAME process.
# ---------------------------------------------------------------------------


class TestResetRedrivesPlay:
    def test_one_reset_command_plays_exactly_one_more_match(self, tmp_path, monkeypatch):
        result = drive_exhibition(tmp_path, monkeypatch, steps_to_win=3, resets=1)

        assert result.escaped is None
        assert result.code == 130
        assert result.reset_codes == [0]
        # THE point of T6: resetting the arena is not enough -- play has to be
        # re-driven from inside this process, because the bridge's single TCP
        # client slot is held here. Two resets and two full step sequences on
        # the wire, from one launcher and one connection.
        (transport,) = result.transports
        assert transport.resets == 2
        assert transport.total_steps == 6
        assert result.policy.reset_calls == 2
        assert len(result.envs) == 1  # the SAME connection, not a second one
        assert transport.connects == 1
        # The request was CONSUMED by the wait, not merely observed by it.
        # Asserted here and not only on the helper: a launcher that read the
        # file some other way would leave it behind for the NEXT match's drain
        # to remove, and the operator would be told their honored reset had
        # been "discarded" -- a message that must only ever appear for a
        # request that really was filed mid-match.
        assert not result.request_path.exists()
        assert "discarded a reset request" not in result.text()

    def test_two_reset_commands_play_two_more_matches(self, tmp_path, monkeypatch):
        result = drive_exhibition(tmp_path, monkeypatch, steps_to_win=2, resets=2)

        (transport,) = result.transports
        assert transport.resets == 3
        assert transport.total_steps == 6
        assert result.text().count("match finished") == 3

    def test_the_human_is_healed_and_repositioned_before_the_next_match(
        self, tmp_path, monkeypatch
    ):
        result = drive_exhibition(tmp_path, monkeypatch, steps_to_win=2, resets=1)

        # The gap T3's review found: formatHumanResetCommands resets the LEARNER
        # only, so a match the AGENT lost leaves the human on partial health.
        assert result.console_lines == human_reset_commands(pad_anchor(0), "Steve")

        # And they land BEFORE the second match's reset -- a heal that arrives
        # after the fight has started is a heal mid-fight.
        console_idx = next(i for i, e in enumerate(result.events) if e[0] == "console")
        reset_indices = [
            i for i, e in enumerate(result.events) if e == ("bridge.send", "reset")
        ]
        assert reset_indices[0] < console_idx < reset_indices[1]

    def test_no_human_commands_are_sent_before_the_first_match(self, tmp_path, monkeypatch):
        # Nobody has joined yet at launch, so healing "the challenger" would be
        # a pile of `No player was found` lines in the console an operator is
        # watching -- the exact noise T3 worked to remove.
        result = drive_exhibition(tmp_path, monkeypatch, resets=0)

        assert result.console_lines == []

    def test_an_unpinned_challenger_still_plays_but_says_what_it_cannot_do(
        self, tmp_path, monkeypatch
    ):
        result = drive_exhibition(tmp_path, monkeypatch, steps_to_win=2, resets=1, challenger=None)

        (transport,) = result.transports
        assert transport.resets == 2  # the match is still re-driven
        assert result.console_lines == []
        text = result.text()
        # Nothing on the wire tells this process who claimed the slot, so this
        # is a real limitation -- stated, with the fix, not silently skipped.
        assert "NOT healing or repositioning the human" in text
        assert "--challenger-username" in text

    def test_no_paper_console_prints_the_commands_instead_of_running_them(
        self, tmp_path, monkeypatch
    ):
        result = drive_exhibition(
            tmp_path, monkeypatch, steps_to_win=2, resets=1, extra_argv=["--no-paper-console"]
        )

        assert result.spawn_kwargs["paper"]["stdin"] is subprocess.DEVNULL
        assert result.console_lines == []
        text = result.text()
        assert "--no-paper-console" in text
        for command in human_reset_commands(pad_anchor(0), "Steve"):
            assert command in text
        # The match is still played: the console is a fairness aid, not a gate.
        assert result.transports[0].resets == 2

    def test_a_dead_console_does_not_take_the_exhibition_down(self, tmp_path, monkeypatch):
        result = drive_exhibition(
            tmp_path, monkeypatch, steps_to_win=2, resets=1, console_fails=True
        )

        assert result.escaped is None
        assert result.code == 130
        assert result.transports[0].resets == 2
        assert "did NOT run" in result.text()

    def test_paper_gets_a_console_pipe_and_the_bridge_does_not(self, tmp_path, monkeypatch):
        result = drive_exhibition(tmp_path, monkeypatch)

        # start.sh execs the JVM, so this pipe IS the server console -- the only
        # command channel this process can reach (RCON is off, and the wire has
        # no slot for a command).
        assert result.spawn_kwargs["paper"]["stdin"] is subprocess.PIPE
        assert result.spawn_kwargs["bridge"]["stdin"] is subprocess.DEVNULL

    def test_the_console_pipe_is_closed_after_paper_is_stopped(self, tmp_path, monkeypatch):
        result = drive_exhibition(tmp_path, monkeypatch)

        # Last link of the teardown chain. Left open, the buffered writer is
        # finalized at interpreter exit and prints "Exception ignored in ...",
        # which reads like a crash in a launcher that shut down correctly.
        assert result.procs["paper"].stdin.closed is True

    def test_the_reset_hint_names_this_launchers_log_dir(self, tmp_path, monkeypatch):
        result = drive_exhibition(tmp_path, monkeypatch)

        assert f"--reset --log-dir {tmp_path / 'logs'}" in result.text()


class TestNothingRestartsAMatchByItself:
    """AC4, and the user's own rule: "after one death, no restart auto for
    human". Every match after the first must trace back to a command the
    operator ran while the launcher was IDLE."""

    def test_a_death_alone_never_starts_a_second_match(self, tmp_path, monkeypatch):
        result = drive_exhibition(tmp_path, monkeypatch, resets=0)

        (transport,) = result.transports
        assert transport.resets == 1
        assert result.text().count("match finished") == 1
        assert "no auto-restart" in result.text()

    def test_idle_polls_that_find_no_request_never_start_a_match(
        self, tmp_path, monkeypatch
    ):
        # The wait must key on the REQUEST, not merely on having waited. A loop
        # that replayed a match every time its sleep returned would look right
        # in every test where a request happens to be filed.
        result = drive_exhibition(tmp_path, monkeypatch, resets=0, idle_wakeups=5)

        assert result.events.count(("idle_poll",)) == 5
        assert result.transports[0].resets == 1
        assert result.text().count("match finished") == 1

    def test_a_request_left_over_from_an_earlier_launch_is_discarded(
        self, tmp_path, monkeypatch
    ):
        # A previous exhibition died before consuming its request. Honoring it
        # would start a second match nobody asked for, in front of a room.
        log_dir = tmp_path / "logs"
        log_dir.mkdir(parents=True)
        (log_dir / RESET_REQUEST_FILENAME).write_text("stale\n")

        result = drive_exhibition(tmp_path, monkeypatch, resets=0)

        assert result.transports[0].resets == 1
        assert "left over from an earlier launch" in result.text()

    def test_a_request_filed_mid_match_is_discarded_when_that_match_ends(
        self, tmp_path, monkeypatch
    ):
        # Honoring it would make the DEATH the proximate cause of the restart,
        # which is the thing AC4 exists to prevent. The launcher is inside
        # play_one_match here and is polling nothing, so the request is filed
        # from the transport's step hook -- the same instant it would arrive in
        # production.
        log_dir = tmp_path / "logs"

        def file_a_request_mid_match(transport):
            path = log_dir / RESET_REQUEST_FILENAME
            if transport.total_steps == 1 and not path.exists():
                path.write_text("filed while the match was running\n")

        result = drive_exhibition(
            tmp_path, monkeypatch, steps_to_win=3, resets=0, on_step=file_a_request_mid_match
        )

        (transport,) = result.transports
        assert transport.resets == 1  # NOT re-driven
        text = result.text()
        assert "filed while the match was still running" in text
        # And the operator is told exactly what to do about it.
        assert "run again" in text or "again now that this one has ended" in text
        assert not (log_dir / RESET_REQUEST_FILENAME).exists()

    def test_the_help_text_says_one_command_one_match(self):
        from deploy.exhibition import _build_arg_parser

        help_text = " ".join(_build_arg_parser().format_help().split())

        assert "--reset" in help_text
        assert "one reset command, one match, never automatic" in help_text


# ===========================================================================
# T7 — the exhibition-only reflex shield.
#
# After `reflex_blind_steps` CONSECUTIVE decision steps whose observation has
# `visible == 0`, the policy's chosen macro is overridden with
# `Macro.TURN_TO_LAST_SEEN`. Three layers, each driven through the REAL
# function under test rather than a re-implementation of its logic:
#
#   1. `_reflex_shield_action` directly (TestReflexShieldPureDecision) --
#      the exact boundary arithmetic `play_one_match` calls.
#   2. `play_one_match()` itself (TestPlayOneMatchReflexShield) -- a fake env
#      that hands back real, controllable observation vectors, so the shield
#      logic runs unmodified inside the real decision loop.
#   3. `run()` end to end (TestReflexShieldEndToEnd) -- the REAL MCPvPEnv and
#      PerceptionFilter, with the opponent placed genuinely behind the
#      learner on the wire, so even a bug in the Obs.VISIBLE accessor itself
#      would surface here.
#
# Plus TestReflexShieldNeverReachesTraining, which pins the "impossible to
# leak into training by accident" requirement rather than just asserting it in
# a docstring.
# ===========================================================================


def _obs_vec(*, visible: bool) -> np.ndarray:
    """A minimal, real ``(OBS_DIM,)`` observation with a controllable
    ``visible`` bit at the frozen ``Obs.VISIBLE`` index. Every other field is
    zero -- the reflex shield reads nothing else."""
    vec = np.zeros(OBS_DIM, dtype=np.float32)
    vec[Obs.VISIBLE] = 1.0 if visible else 0.0
    return vec


class TestReflexShieldPureDecision:
    """Unit-level coverage of ``_reflex_shield_action`` -- the exact function
    ``play_one_match`` calls every decision step, not a re-implementation of
    its arithmetic."""

    def _shield(self, *args, **kwargs):
        from deploy.exhibition import _reflex_shield_action

        return _reflex_shield_action(*args, **kwargs)

    def test_disabled_never_even_looks_at_obs(self):
        # A sentinel with no __getitem__/__float__ -- if the shield touched it
        # at all with the threshold at 0, this raises before the assertion
        # gets a chance to run.
        sentinel = object()
        assert self._shield(sentinel, 3, 5, reflex_blind_steps=0) == (3, 0, False)

    def test_a_negative_threshold_is_also_disabled(self):
        sentinel = object()
        assert self._shield(sentinel, 3, 5, reflex_blind_steps=-1) == (3, 0, False)

    def test_visible_resets_the_streak_and_never_overrides(self):
        obs = _obs_vec(visible=True)
        assert self._shield(obs, 4, 7, reflex_blind_steps=3) == (4, 0, False)

    def test_below_threshold_leaves_the_chosen_action_untouched(self):
        obs = _obs_vec(visible=False)
        assert self._shield(obs, 4, 1, reflex_blind_steps=3) == (4, 2, False)

    def test_reaching_the_threshold_overrides_to_turn_to_last_seen(self):
        obs = _obs_vec(visible=False)
        assert self._shield(obs, 4, 2, reflex_blind_steps=3) == (
            int(Macro.TURN_TO_LAST_SEEN),
            3,
            True,
        )

    def test_stays_overridden_for_every_further_blind_step(self):
        obs = _obs_vec(visible=False)
        assert self._shield(obs, 4, 10, reflex_blind_steps=3) == (
            int(Macro.TURN_TO_LAST_SEEN),
            11,
            True,
        )

    def test_fired_is_false_even_when_the_policy_already_chose_turn_to_last_seen(self):
        # `fired` must mean "the shield substituted this", not merely "the
        # action equals TURN_TO_LAST_SEEN" -- the policy is allowed to choose
        # macro 7 on its own without that counting as an override.
        obs = _obs_vec(visible=False)
        action, streak, fired = self._shield(
            obs, int(Macro.TURN_TO_LAST_SEEN), 0, reflex_blind_steps=3
        )
        assert (action, streak, fired) == (int(Macro.TURN_TO_LAST_SEEN), 1, False)

    def test_reads_visibility_through_the_frozen_accessor(self):
        # Pin against a magic-number regression: builds the vector by hand
        # (not via _obs_vec) so this fails if Obs.VISIBLE's index ever drifts
        # out of step with a literal index someone hard-codes here by mistake.
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        obs[Obs.VISIBLE] = 0.0
        _, streak, fired = self._shield(obs, 0, 2, reflex_blind_steps=3)
        assert (streak, fired) == (3, True)


class FakePolicyCountingActs(FakePolicy):
    """FakePolicy that also counts ``act()`` calls, for the "policy is asked
    every decision, even overridden ones" assertion."""

    def __init__(self, actions):
        super().__init__(actions)
        self.act_calls = 0

    def act(self, obs):
        self.act_calls += 1
        return super().act(obs)


class VisibilityScriptedEnv:
    """Minimal env stand-in whose observations carry a real, controllable
    ``visible`` bit -- what lets ``play_one_match`` be driven directly, not
    re-implemented, without a live ``MCPvPEnv``.

    ``visible_schedule`` is a list of bools, one per decision (1-indexed in
    spirit, 0-indexed in the list): entry 0 is what ``reset()`` returns and is
    what decision 1 acts on; entry ``d`` (for ``d >= 1``) is what the ``d``-th
    ``step()`` call returns and is what decision ``d + 1`` acts on. The match
    ends exactly when ``len(visible_schedule)`` decisions have been taken.
    """

    def __init__(self, visible_schedule):
        self._schedule = list(visible_schedule)
        assert self._schedule, "need at least one decision to play"
        self.reset_calls = 0
        self.actions_sent = []

    def reset(self, seed=None):
        self.reset_calls += 1
        self.actions_sent = []
        return _obs_vec(visible=self._schedule[0])

    def step(self, action):
        self.actions_sent.append(action)
        d = len(self.actions_sent)  # this call is decision `d`
        done = d >= len(self._schedule)
        # The obs is irrelevant once done -- the loop will not consult it.
        next_visible = self._schedule[d] if not done else True
        obs = _obs_vec(visible=next_visible)
        info = {"won": True} if done else {}
        return obs, 0.0, done, info


class TestPlayOneMatchReflexShield:
    """``play_one_match()`` itself, driven directly -- not a re-implementation
    of the override decision -- with observations carrying a real,
    controllable ``visible`` bit."""

    def test_default_reflex_blind_steps_never_overrides(self):
        # The DEFAULT every pre-T7 caller gets (including every OTHER test in
        # this file that calls play_one_match without the keyword) must stay
        # "off". Pinned explicitly here rather than left to be an accident of
        # every other test never going blind for long enough.
        env = VisibilityScriptedEnv([False] * 10)
        policy = FakePolicy([1] * 10)

        play_one_match(env, policy, log=lambda m: None)

        assert env.actions_sent == [1] * 10

    def test_overrides_after_the_configured_consecutive_blind_streak(self):
        env = VisibilityScriptedEnv([False] * 10)
        policy = FakePolicy([1] * 10)

        play_one_match(env, policy, reflex_blind_steps=3, log=lambda m: None)

        ttls = int(Macro.TURN_TO_LAST_SEEN)
        assert env.actions_sent == [1, 1] + [ttls] * 8

    def test_unchanged_for_the_whole_match_while_visible(self):
        env = VisibilityScriptedEnv([True] * 10)
        policy = FakePolicy([2] * 10)

        play_one_match(env, policy, reflex_blind_steps=3, log=lambda m: None)

        assert env.actions_sent == [2] * 10

    def test_the_streak_resets_on_visibility_and_re_accumulates(self):
        # blind, blind, blind(fires), blind(fires), VISIBLE, blind, blind, blind(fires)
        schedule = [False, False, False, False, True, False, False, False]
        env = VisibilityScriptedEnv(schedule)
        policy = FakePolicy([1] * 8)

        play_one_match(env, policy, reflex_blind_steps=3, log=lambda m: None)

        ttls = int(Macro.TURN_TO_LAST_SEEN)
        assert env.actions_sent == [1, 1, ttls, ttls, 1, 1, 1, ttls]

    def test_the_policy_is_still_asked_every_decision_even_when_overridden(self):
        env = VisibilityScriptedEnv([False] * 5)
        policy = FakePolicyCountingActs([1] * 5)

        play_one_match(env, policy, reflex_blind_steps=2, log=lambda m: None)

        # One act() per decision, whether or not the shield substituted the
        # macro afterward -- any recurrent state the policy carries must keep
        # advancing even on overridden steps.
        assert policy.act_calls == 5
        assert len(env.actions_sent) == 5

    def test_exactly_one_env_step_per_decision_even_with_the_shield_active(self):
        env = VisibilityScriptedEnv([False] * 6)
        policy = FakePolicy([0] * 6)

        play_one_match(env, policy, reflex_blind_steps=2, log=lambda m: None)

        # Not two -- the override must not cost a second env.step() call.
        assert len(env.actions_sent) == 6

    def test_the_summary_log_reports_the_override_count_and_threshold(self):
        messages = []
        env = VisibilityScriptedEnv([False] * 5)
        policy = FakePolicy([1] * 5)

        play_one_match(env, policy, reflex_blind_steps=2, log=messages.append)

        text = "\n".join(messages)
        assert "reflex shield overrode the action on 4/5 decision step(s)" in text
        assert "reflex_blind_steps=2" in text

    def test_no_summary_log_line_when_the_shield_never_fires(self):
        messages = []
        env = VisibilityScriptedEnv([True] * 3)
        policy = FakePolicy([0] * 3)

        play_one_match(env, policy, reflex_blind_steps=2, log=messages.append)

        assert not any("reflex shield" in m for m in messages)

    def test_the_override_count_ignores_the_policys_own_choice_of_the_same_macro(self):
        # The policy independently choosing TURN_TO_LAST_SEEN while blind, but
        # before the shield's own threshold, must not inflate the reported
        # override count -- "fired" means the shield substituted the macro,
        # not merely that the macro sent equals TURN_TO_LAST_SEEN.
        messages = []
        ttls = int(Macro.TURN_TO_LAST_SEEN)
        env = VisibilityScriptedEnv([False] * 5)
        policy = FakePolicy([ttls, ttls, 1, 1, 1])

        play_one_match(env, policy, reflex_blind_steps=3, log=messages.append)

        # Every action sent happens to be TTLS (2 by the policy's own choice,
        # 3 by the shield once the streak reaches the threshold) -- only the
        # LATTER 3 should be counted as overrides.
        assert env.actions_sent == [ttls] * 5
        text = "\n".join(messages)
        assert "reflex shield overrode the action on 3/5 decision step(s)" in text

    def test_no_summary_log_line_when_the_shield_is_disabled(self):
        messages = []
        env = VisibilityScriptedEnv([False] * 10)
        policy = FakePolicy([1] * 10)

        play_one_match(env, policy, log=messages.append)  # reflex_blind_steps=0

        assert not any("reflex shield" in m for m in messages)


class TestReflexShieldEndToEnd:
    """T7 driven through the REAL ``run()`` -> real ``MCPvPEnv`` -> real
    ``PerceptionFilter`` pipeline, with the opponent placed genuinely BEHIND
    the learner on the wire (outside the frozen FOV cone -- see
    ``_state_msg``'s ``visible`` parameter) rather than a hand-set observation
    bit. ``ExhibitionConfig.reflex_blind_steps`` defaults to 8, and none of
    these tests override it -- there is no CLI flag for it (T7 does not add
    one), which is itself exactly what proves ``run()`` wires the config
    value through rather than a hard-coded constant.
    """

    def test_overrides_to_turn_to_last_seen_after_the_default_blind_streak(
        self, tmp_path, monkeypatch
    ):
        # Blind for all 10 decisions of a 10-decision match: the override
        # should fire on decisions 8, 9, 10 (reflex_blind_steps defaults to 8)
        # and leave the policy's own IDLE (macro 0) alone before that.
        result = drive_exhibition(
            tmp_path, monkeypatch, steps_to_win=10, opponent_visible=lambda idx: False
        )

        (transport,) = result.transports
        ttls = int(Macro.TURN_TO_LAST_SEEN)
        assert transport.actions_sent == [0] * 7 + [ttls] * 3
        text = result.text()
        assert "reflex shield overrode the action on 3/10 decision step(s)" in text
        assert "reflex_blind_steps=8" in text
        # The policy is still asked for its action on every decision.
        assert result.policy.act_calls == 10

    def test_never_overrides_while_the_opponent_stays_in_view(self, tmp_path, monkeypatch):
        result = drive_exhibition(tmp_path, monkeypatch, steps_to_win=10)

        (transport,) = result.transports
        assert transport.actions_sent == [0] * 10
        assert "reflex shield" not in result.text()

    def test_the_streak_resets_the_moment_the_opponent_is_visible_again(
        self, tmp_path, monkeypatch
    ):
        # Blind for decisions 1-9 and 11-12, visible only on decision 10. If the
        # streak survived that one visible decision, decision 12 would already
        # be overridden (9 + 1 + 2 = 12 >= 8); it must not be -- the streak has
        # only reached 2 by then.
        result = drive_exhibition(
            tmp_path,
            monkeypatch,
            steps_to_win=12,
            opponent_visible=lambda idx: idx == 10,
        )

        (transport,) = result.transports
        ttls = int(Macro.TURN_TO_LAST_SEEN)
        assert transport.actions_sent == [0] * 7 + [ttls, ttls] + [0] * 3

    def test_reflex_overrides_are_still_exactly_one_step_per_decision(
        self, tmp_path, monkeypatch
    ):
        result = drive_exhibition(
            tmp_path, monkeypatch, steps_to_win=10, opponent_visible=lambda idx: False
        )

        (transport,) = result.transports
        # One "step" message per decision -- the override must not cost a
        # second window on the wire.
        assert transport.total_steps == 10
        assert len(transport.actions_sent) == 10


class TestReflexShieldNeverReachesTraining:
    """T7 spec: "It must NEVER be active during training... Structure it so
    that is impossible to do by accident, and pin it with a test."

    Two independent guarantees are pinned, matching the module docstring's
    REFLEX SHIELD section:

      1. ``play_one_match``'s ``reflex_blind_steps`` defaults to 0 (off) --
         pinned above by ``test_default_reflex_blind_steps_never_overrides``
         and ``test_no_summary_log_line_when_the_shield_is_disabled``.
      2. No training entry point even IMPORTS this module, so the shield is
         not reachable from a training loop at all regardless of what
         parameters anyone might one day pass -- pinned here, by reading the
         committed source of every training entry point directly (a stronger
         guarantee than importing them in-process, which cannot prove
         anything: this very test file already imports deploy.exhibition at
         module scope, so it would already be in sys.modules by the time any
         in-process check ran).
    """

    TRAINING_ENTRY_POINTS = (
        REPO_ROOT / "agent" / "train.py",
        REPO_ROOT / "agent" / "train_config.py",
        REPO_ROOT / "distributed" / "actor.py",
        REPO_ROOT / "distributed" / "learner.py",
        REPO_ROOT / "distributed" / "launcher.py",
    )

    def test_no_training_entry_point_references_this_module(self):
        needles = ("deploy.exhibition", "deploy import exhibition", "play_one_match")
        for path in self.TRAINING_ENTRY_POINTS:
            assert path.is_file(), f"expected training entry point missing: {path}"
            text = path.read_text(encoding="utf-8")
            for needle in needles:
                assert needle not in text, (
                    f"{path} references {needle!r} -- the T7 reflex shield must "
                    "stay reachable only from an exhibition launch, never from "
                    "a training entry point."
                )
