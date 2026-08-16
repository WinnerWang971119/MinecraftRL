"""tests/test_exhibition.py — T5: the one-command exhibition launcher.

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

Everything else here is supporting coverage for the pure helpers `run()` is
built from (``is_port_free``, ``find_checkpoints``, ``build_bridge_argv``,
``load_greedy_policy``, ``wait_for_port``, ``play_one_match``,
``find_toolchain_problems``) plus the --checkpoint-missing/unloadable "never
random-init" guarantee, the --challenger-username help-text requirement from
the spec, and the ``BaseException``-proof teardown a second Ctrl-C depends on.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bridge.messages import ResetAckMsg, StateMsg
from deploy.exhibition import (
    DEFAULT_BRIDGE_HOST,
    DEFAULT_BRIDGE_PORT,
    DEFAULT_MC_PORT,
    CheckpointError,
    build_bridge_argv,
    checkpoint_missing_message,
    checkpoint_unloadable_message,
    find_checkpoints,
    find_toolchain_problems,
    is_port_free,
    load_greedy_policy,
    main,
    play_one_match,
    run,
    wait_for_port,
)
from distributed.launcher import pad_anchor, pad_usernames
from env.mc_pvp_env import BridgeError, MCPvPEnv

REPO_ROOT = Path(__file__).resolve().parent.parent
OPS_JSON = REPO_ROOT / "server" / "ops.json"


# ---------------------------------------------------------------------------
# Shared fakes (mirror the FakeProc / ScriptedProbe style in
# tests/test_pad_launcher.py — no real OS process or socket).
# ---------------------------------------------------------------------------


class FakeProc:
    """Minimal Popen stand-in: poll only, no OS process.

    ``label``/``record`` are optional: pass both and every lifecycle call
    appends ``(what, label)`` to the shared event list, which is how the
    happy-path test pins teardown ORDER (bridge before Paper) rather than
    merely asserting both were stopped.
    """

    def __init__(self, pid=4242, exit_code=None, *, label=None, record=None):
        self.pid = pid
        self._exit_code = exit_code
        self.label = label
        self._record = record
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


def _state_msg(*, tick, opp_health=20.0, opponent_died=False):
    """One valid ``state`` line: both fighters at full health, two blocks apart
    and facing each other, no events unless a death is being scripted.

    Only the OPPONENT's death is ever scripted here. Under
    ``ExhibitionConfig.no_timeout`` a death is the one and only thing that ends
    an exhibition match (AC4), so it is also the only thing worth scripting.
    """
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
                "pos": [0.0, 64.0, 2.0],
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
    """

    def __init__(self, *, steps_to_win, record, interrupt_on_close=False):
        self._steps_to_win = steps_to_win
        self._record = record
        self._interrupt_on_close = interrupt_on_close
        self._inbound = []
        self._tick = 0
        self.steps_sent = 0
        self.connects = 0
        self.closes = 0

    def connect(self):
        self.connects += 1
        self._record(("transport.connect",))

    def send(self, obj):
        kind = obj["type"]
        self._record(("bridge.send", kind))
        self._tick += 1
        if kind == "reset":
            self._inbound.append(_reset_ack_msg())
            self._inbound.append(_state_msg(tick=self._tick))
        elif kind == "step":
            self.steps_sent += 1
            killed = self.steps_sent >= self._steps_to_win
            self._inbound.append(
                _state_msg(
                    tick=self._tick,
                    opp_health=0.0 if killed else 20.0,
                    opponent_died=killed,
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
        self.ops_writes = []
        self.procs = {}
        self.envs = []
        self.transports = []
        self.policy = None


def drive_exhibition(
    tmp_path,
    monkeypatch,
    *,
    steps_to_win=3,
    proc_classes=None,
    interrupt_on_close=False,
    log_bomb=None,
):
    """Drive the REAL ``run()`` through a complete exhibition and record it.

    Spawning a child "opens" its port, so both boot on their first probe and
    ``wait_for_port`` never sleeps. That leaves the idle wait as the only
    sleeper, and its first call raises ``KeyboardInterrupt`` -- the normal way
    an exhibition ends.

    ``MCPvPEnv`` is WRAPPED, not replaced: the launcher builds the real class
    over the scripted transport and the wrapper only keeps a reference, so a
    test can assert what the env was built with.

    ``log_bomb`` makes the ``log`` seam raise ``KeyboardInterrupt`` on the
    first message containing that substring — an interrupt landing somewhere a
    teardown helper's own handler cannot absorb it, which is what the nested
    ``finally`` chain (rather than a flat one) exists to survive.
    """
    proc_classes = dict(proc_classes or {})
    result = ExhibitionRun()
    record = result.events.append
    ports = {"mc": False, "bridge": False}

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
        proc = proc_classes.get(label, FakeProc)(
            pid=101 if label == "paper" else 202, label=label, record=record
        )
        result.procs[label] = proc
        return proc

    def transport_factory(host, port):
        record(("transport_factory", host, port))
        transport = ScriptedBridgeTransport(
            steps_to_win=steps_to_win,
            record=record,
            interrupt_on_close=interrupt_on_close,
        )
        result.transports.append(transport)
        return transport

    def write_ops(n_pads, path):
        record(("write_ops", n_pads))
        result.ops_writes.append((n_pads, path))
        return path

    def interrupting_sleep(seconds):
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

    try:
        result.code = run(
            [
                "--checkpoint", str(checkpoint),
                "--checkpoints-dir", str(tmp_path),
                "--challenger-username", "Steve",
                "--log-dir", str(tmp_path / "logs"),
            ],
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
