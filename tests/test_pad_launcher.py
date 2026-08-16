"""tests/test_pad_launcher.py — pad geometry, ops.json, the prime barrier, T10 gates.

Covers the offline-testable half of T10 (the one-JVM / N-pad launcher):

  * ``pad_anchor`` -- THE sole coordinate source in the repo. Pinned at the row
    wrap so the row-major / 5-column convention cannot drift silently.
  * ``pad_usernames`` -- the ``i == 0`` special case (``learner_bot``, NOT
    ``learner_0``) that keeps the manual single-arena path byte-identical, and that
    must agree with ``usernamesForPad`` in ``bridge/run.js``.
  * ``ops.json`` generation -- offline-mode UUIDs, and byte-identity with the
    committed ``server/ops.json`` at N=1.
  * The launch plan -- ports, anchors and the exact bridge argv, including a
    round-trip through run.js's REAL argv parser when node is available (the T9/T10
    seam: launcher.py writes the flags, run.js validates them).
  * ``prime_pads`` -- the reset-before-step barrier: descending order, bounded
    retries, always-close, and a loud failure naming the pad.
  * ``SubprocessArenaLauncher`` -- bridge-only relaunch, and the constructor
    keyword ``agent/train.py`` calls it with.
  * TC19 -- ``start-pads.sh --check`` fails loudly when ``max-players`` cannot seat
    ``2N + 10``, and ``setup.sh`` writes a sufficient value for a given ``PADS``.

Nothing here opens a socket, spawns a JVM or spawns a bridge.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from distributed.launcher import (
    PAD_GRID_COLS,
    PAD_SPACING,
    PadAnchor,
    PadPrimeError,
    SubprocessArenaLauncher,
    missing_ops,
    offline_uuid,
    ops_entries,
    ops_json,
    pad_anchor,
    pad_usernames,
    plan,
    prime_pads,
    required_max_players,
    write_ops_json,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
START_PADS_SH = REPO_ROOT / "server" / "setup" / "start-pads.sh"
SETUP_SH = REPO_ROOT / "server" / "setup" / "setup.sh"
COMMITTED_OPS = REPO_ROOT / "server" / "ops.json"
PAPER_JAR = REPO_ROOT / "server" / "paper-1.21.1-133.jar"


# ---------------------------------------------------------------------------
# pad_anchor -- the sole coordinate source
# ---------------------------------------------------------------------------


class TestPadAnchor:
    """The anchor formula, pinned. A change here relocates every pad in the world."""

    def test_constants_are_the_named_values(self):
        assert PAD_SPACING == 512
        assert PAD_GRID_COLS == 5

    @pytest.mark.parametrize(
        "index,expected",
        [
            (0, (0, 0)),        # byte-identical to today's single arena
            (1, (512, 0)),
            (4, (2048, 0)),     # last column of row 0
            (5, (0, 512)),      # row wrap: the case a column-major bug would break
            (12, (1024, 1024)),
            (24, (2048, 2048)),  # last pad of a 5x5 fleet
        ],
    )
    def test_anchor_is_row_major_on_a_five_wide_grid(self, index, expected):
        anchor = pad_anchor(index)
        assert (anchor.x, anchor.z) == expected

    def test_anchor_matches_the_formula_for_a_whole_fleet(self):
        for index in range(25):
            anchor = pad_anchor(index)
            assert anchor.x == (index % PAD_GRID_COLS) * PAD_SPACING
            assert anchor.z == (index // PAD_GRID_COLS) * PAD_SPACING

    def test_anchors_are_all_distinct(self):
        anchors = {(a.x, a.z) for a in (pad_anchor(i) for i in range(25))}
        assert len(anchors) == 25

    def test_as_flag_is_the_plain_form_run_js_accepts(self):
        # run.js validates each component with /^\d+$/: no sign, no decimal point,
        # no NBT suffix, no spaces. A space would also break the tab-split plan.
        for index in range(25):
            flag = pad_anchor(index).as_flag()
            x, _, z = flag.partition(",")
            assert x.isdigit() and z.isdigit(), flag
            assert " " not in flag

    def test_negative_index_raises(self):
        with pytest.raises(ValueError, match=">= 0"):
            pad_anchor(-1)

    def test_non_integer_index_raises(self):
        with pytest.raises(ValueError, match="must be an int"):
            pad_anchor(1.5)  # type: ignore[arg-type]

    def test_anchor_is_frozen(self):
        with pytest.raises(Exception):
            pad_anchor(0).x = 7  # type: ignore[misc]


class TestPadUsernames:
    """Pad 0 keeps the historical names; every other pad is indexed."""

    def test_pad_zero_keeps_the_single_arena_names(self):
        # NOT learner_0: this is the deliberate change from PR #21's launcher.
        assert pad_usernames(0) == ("learner_bot", "dummy_bot")

    @pytest.mark.parametrize("index", [1, 2, 7, 24])
    def test_other_pads_are_indexed(self, index):
        assert pad_usernames(index) == (f"learner_{index}", f"dummy_{index}")

    def test_all_usernames_in_a_fleet_are_distinct(self):
        names = [name for i in range(25) for name in pad_usernames(i)]
        assert len(set(names)) == len(names)

    def test_negative_index_raises(self):
        with pytest.raises(ValueError):
            pad_usernames(-1)


# ---------------------------------------------------------------------------
# ops.json
# ---------------------------------------------------------------------------


class TestOpsJson:
    """Offline UUIDs and the exact file shape Paper reads at boot."""

    def test_offline_uuid_matches_the_committed_ops_file(self):
        committed = json.loads(COMMITTED_OPS.read_text(encoding="utf-8"))
        by_name = {entry["name"]: entry["uuid"] for entry in committed}
        assert by_name["learner_bot"] == offline_uuid("learner_bot")
        assert by_name["dummy_bot"] == offline_uuid("dummy_bot")

    def test_offline_uuid_is_version_3_with_the_ietf_variant(self):
        raw = offline_uuid("learner_7").replace("-", "")
        assert raw[12] == "3"                     # version nibble
        assert raw[16] in "89ab"                  # variant 10xx

    def test_offline_uuid_rejects_an_empty_name(self):
        with pytest.raises(ValueError):
            offline_uuid("   ")

    def test_one_pad_is_byte_identical_to_the_committed_file(self):
        # AC11's ops analog: an N=1 fleet must not rewrite server/ops.json at all.
        assert ops_json(1) == COMMITTED_OPS.read_text(encoding="utf-8")

    def test_entries_cover_every_bot_at_level_four(self):
        entries = ops_entries(3)
        assert len(entries) == 6
        assert [e["name"] for e in entries] == [
            "learner_bot", "dummy_bot",
            "learner_1", "dummy_1",
            "learner_2", "dummy_2",
        ]
        for entry in entries:
            # Level 4 is required: a non-op cannot run /function at all, and the
            # bridge is the sole command channel (RCON is disabled).
            assert entry["level"] == 4
            assert entry["bypassesPlayerLimit"] is False
            assert set(entry) == {"uuid", "name", "level", "bypassesPlayerLimit"}

    def test_uuids_are_unique_across_a_fleet(self):
        uuids = [e["uuid"] for e in ops_entries(25)]
        assert len(set(uuids)) == len(uuids)

    def test_zero_pads_raises(self):
        with pytest.raises(ValueError):
            ops_entries(0)

    def test_write_then_check_round_trips(self, tmp_path):
        target = tmp_path / "nested" / "ops.json"
        written = write_ops_json(4, str(target))
        assert Path(written) == target
        assert missing_ops(4, str(target)) == []

    def test_missing_ops_names_every_uncovered_bot(self, tmp_path):
        target = tmp_path / "ops.json"
        write_ops_json(2, str(target))
        # The file covers 2 pads; a 4-pad fleet needs 4 more bots.
        assert missing_ops(4, str(target)) == [
            "learner_2", "dummy_2", "learner_3", "dummy_3",
        ]

    def test_missing_ops_rejects_an_under_levelled_op(self, tmp_path):
        target = tmp_path / "ops.json"
        entries = ops_entries(1)
        entries[1]["level"] = 2  # dummy_bot demoted: cannot run /function
        target.write_text(json.dumps(entries), encoding="utf-8")
        assert missing_ops(1, str(target)) == ["dummy_bot"]

    def test_missing_ops_treats_an_unreadable_file_as_covering_nothing(self, tmp_path):
        absent = tmp_path / "nope.json"
        assert missing_ops(2, str(absent)) == [
            "learner_bot", "dummy_bot", "learner_1", "dummy_1",
        ]
        garbage = tmp_path / "garbage.json"
        garbage.write_text("not json at all", encoding="utf-8")
        assert len(missing_ops(2, str(garbage))) == 4


class TestRequiredMaxPlayers:
    @pytest.mark.parametrize("pads,expected", [(1, 12), (2, 14), (8, 26), (25, 60)])
    def test_two_n_plus_ten(self, pads, expected):
        assert required_max_players(pads) == expected

    def test_zero_pads_raises(self):
        with pytest.raises(ValueError):
            required_max_players(0)


# ---------------------------------------------------------------------------
# The launch plan
# ---------------------------------------------------------------------------


class TestPlan:
    """Ports, anchors, usernames and the exact bridge argv."""

    def test_single_pad_reproduces_todays_manual_path(self):
        (entry,) = plan(1)
        assert entry["mc_port"] == 25565
        assert entry["bridge_port"] == 5555
        assert entry["pad_origin"] == "0,0"
        assert entry["learner_username"] == "learner_bot"
        assert entry["dummy_username"] == "dummy_bot"

    def test_one_shared_mc_port_and_stepped_bridge_ports(self):
        entries = plan(6)
        assert {e["mc_port"] for e in entries} == {25565}, "one JVM, one port"
        assert [e["bridge_port"] for e in entries] == [5555, 5556, 5557, 5558, 5559, 5560]

    def test_anchors_come_from_pad_anchor(self):
        for entry in plan(7):
            anchor = pad_anchor(int(entry["pad_index"]))
            assert entry["anchor_x"] == anchor.x
            assert entry["anchor_z"] == anchor.z
            assert entry["pad_origin"] == anchor.as_flag()

    def test_bridge_argv_is_exact(self):
        entries = plan(2, repo_root="/repo")
        assert entries[1]["bridge_command"] == [
            "node",
            "/repo/bridge/run.js",
            "--port", "25565",
            "--bridge-port", "5556",
            "--pad-index", "1",
            "--pad-origin", "512,0",
            "--learner-username", "learner_1",
            "--dummy-username", "dummy_1",
        ]

    def test_the_default_argv_carries_no_knockback_flag_at_all(self):
        """T11c: the toggle must be INVISIBLE at its default.

        Every other launch parameter is passed explicitly, but this one is
        appended only when it is False. Emitting ``--dummy-knockback-immune true``
        by default would change the argv every existing run was built against for
        no behavioral gain.
        """
        for entry in plan(4):
            assert "--dummy-knockback-immune" not in entry["bridge_command"]

    def test_constructor_default_carries_no_knockback_flag(self):
        """T11c follow-up: pin the CONSTRUCTOR's default, not just plan()'s.

        The module-level ``plan()`` helper declares its own
        ``dummy_knockback_immune=True`` default (see its signature above) and
        forwards it to the constructor explicitly, so every test that calls
        the bare ``plan(...)`` function -- including the one right above --
        binds THAT default, never ``SubprocessArenaLauncher.__init__``'s own.
        Production (``agent/train.py``) constructs ``SubprocessArenaLauncher``
        directly with no ``dummy_knockback_immune`` argument, so only a test
        that goes through the bare constructor actually pins what ships.
        """
        launcher = SubprocessArenaLauncher(repo_root="/repo")
        entries = launcher.plan(2)
        for entry in entries:
            assert "--dummy-knockback-immune" not in entry["bridge_command"]

    def test_a_non_immune_fleet_appends_the_flag_to_every_pad(self):
        """T11c/AC18: the ONLY way a scripted opponent can take knockback.

        The exhibition path cannot stand in for this -- it runs with a human
        opponent, where the bridge's override branch never fires -- so if the
        flag does not reach the bridge argv here, ``knockback_immune=False``
        reaches nothing at all and the retrain trains against an immune,
        immobile target.
        """
        entries = plan(3, repo_root="/repo", dummy_knockback_immune=False)
        for entry in entries:
            argv = entry["bridge_command"]
            assert argv[-2:] == ["--dummy-knockback-immune", "false"]
        # The rest of the argv is untouched -- the flag is appended, not woven in.
        assert entries[1]["bridge_command"][:-2] == [
            "node",
            "/repo/bridge/run.js",
            "--port", "25565",
            "--bridge-port", "5556",
            "--pad-index", "1",
            "--pad-origin", "512,0",
            "--learner-username", "learner_1",
            "--dummy-username", "dummy_1",
        ]

    def test_the_flag_must_be_a_real_bool(self):
        # `bool("false")` is True, so a string forwarded from a config layer
        # would silently keep every dummy immune while the caller believed it had
        # turned the toggle off.
        for bad in ["false", "true", 0, 1, None]:
            with pytest.raises(TypeError):
                SubprocessArenaLauncher(dummy_knockback_immune=bad)

    def test_every_pad_passes_an_explicit_anchor(self):
        # run.js makes `--pad-index i>0` without `--pad-origin` a hard startup
        # failure, because defaulting would stack pad i on pad 0.
        for entry in plan(5):
            argv = entry["bridge_command"]
            assert "--pad-origin" in argv
            assert argv[argv.index("--pad-origin") + 1] == entry["pad_origin"]

    def test_ports_are_overridable(self):
        entries = plan(2, mc_port=25600, bridge_base_port=6000)
        assert [e["mc_port"] for e in entries] == [25600, 25600]
        assert [e["bridge_port"] for e in entries] == [6000, 6001]

    def test_zero_pads_raises(self):
        with pytest.raises(ValueError):
            plan(0)


@pytest.mark.skipif(
    shutil.which("node") is None or not (REPO_ROOT / "bridge" / "node_modules").is_dir(),
    reason="node and bridge/node_modules are required to drive run.js's real parser",
)
def test_emitted_argv_round_trips_through_run_js(tmp_path):
    """The T9/T10 seam: run.js's REAL parser must accept every argv we emit.

    launcher.py writes the flags; run.js validates them (strictly -- a malformed
    anchor is a hard startup failure). Asserting the parse here catches a flag-name
    or anchor-format divergence offline instead of at fleet boot.
    """
    pads = 6
    entries = plan(pads, repo_root=str(REPO_ROOT))
    argvs = [entry["bridge_command"][2:] for entry in entries]  # drop node + run.js
    script = tmp_path / "check.js"
    script.write_text(
        "const { parseBridgeConfig } = require(process.argv[2]);\n"
        "const argvs = JSON.parse(process.argv[3]);\n"
        "const out = argvs.map((a) => parseBridgeConfig(a, {}));\n"
        "process.stdout.write(JSON.stringify(out));\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            "node",
            str(script),
            str(REPO_ROOT / "bridge" / "run.js"),
            json.dumps(argvs),
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    parsed = json.loads(result.stdout)
    assert len(parsed) == pads
    for index, config in enumerate(parsed):
        anchor = pad_anchor(index)
        learner, dummy = pad_usernames(index)
        assert config["padIndex"] == index
        assert config["padOriginX"] == anchor.x
        assert config["padOriginZ"] == anchor.z
        assert config["bridgePort"] == 5555 + index
        assert config["port"] == 25565
        assert config["learnerUsername"] == learner
        assert config["dummyUsername"] == dummy


@pytest.mark.skipif(
    shutil.which("node") is None or not (REPO_ROOT / "bridge" / "node_modules").is_dir(),
    reason="node and bridge/node_modules are required to drive run.js's real parser",
)
def test_non_immune_argv_round_trips_through_run_js(tmp_path):
    """T11c: the knockback flag must survive the same seam as every other flag.

    Two halves that can drift silently: launcher.py writes the flag name and
    value, run.js's CONFIG_SPECS decides which names exist and what values they
    accept (strictly -- ``0``/``yes`` are hard startup failures). If they ever
    disagree the bridge would refuse to start on a scripted-opponent run, or --
    worse, if the name were merely unknown to a laxer parser -- start immune.
    """
    pads = 3
    entries = plan(pads, repo_root=str(REPO_ROOT), dummy_knockback_immune=False)
    argvs = [entry["bridge_command"][2:] for entry in entries]  # drop node + run.js
    script = tmp_path / "check.js"
    script.write_text(
        "const { parseBridgeConfig } = require(process.argv[2]);\n"
        "const argvs = JSON.parse(process.argv[3]);\n"
        "const out = argvs.map((a) => parseBridgeConfig(a, {}));\n"
        "process.stdout.write(JSON.stringify(out));\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            "node",
            str(script),
            str(REPO_ROOT / "bridge" / "run.js"),
            json.dumps(argvs),
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    parsed = json.loads(result.stdout)
    assert len(parsed) == pads
    for config in parsed:
        # A real JSON false, not the string "false": run.js coerced it, so the
        # ArenaBots constructor's boolean check will pass rather than throw.
        assert config["dummyKnockbackImmune"] is False


# ---------------------------------------------------------------------------
# SubprocessArenaLauncher -- bridge-only relaunch
# ---------------------------------------------------------------------------


class FakeProc:
    """Minimal Popen stand-in: poll/terminate/wait/kill, no OS process."""

    def __init__(self, exit_code=None):
        self._exit_code = exit_code
        self.terminated = False
        self.killed = False

    def poll(self):
        return self._exit_code

    def terminate(self):
        self.terminated = True
        self._exit_code = -15

    def wait(self, timeout=None):
        return self._exit_code

    def kill(self):
        self.killed = True
        self._exit_code = -9


class ScriptedProbe:
    """A ``(host, port) -> bool`` probe driven by a mutable {port: bool} map."""

    def __init__(self, open_ports):
        self.open_ports = dict(open_ports)
        self.calls = []

    def __call__(self, host, port):
        self.calls.append(port)
        return bool(self.open_ports.get(port, False))


class TestSubprocessArenaLauncher:
    def test_accepts_the_keyword_agent_train_constructs_it_with(self):
        # agent/train.py: SubprocessArenaLauncher(bridge_base_port=base_port).
        launcher = SubprocessArenaLauncher(bridge_base_port=5600)
        assert launcher.spec_for(2).bridge_port == 5602

    def test_launch_spawns_exactly_one_bridge_with_the_planned_argv(self):
        probe = ScriptedProbe({25565: True, 5556: False})
        spawned = []

        def fake_popen(cmd, cwd=None):
            spawned.append((list(cmd), cwd))
            probe.open_ports[5556] = True  # the bridge comes up
            return FakeProc()

        launcher = SubprocessArenaLauncher(
            repo_root="/repo", popen=fake_popen, port_probe=probe,
            sleep=lambda _s: None, log=lambda _m: None,
        )
        launcher.launch(1)

        assert len(spawned) == 1
        cmd, cwd = spawned[0]
        assert cwd == "/repo"
        assert cmd == launcher.spec_for(1).bridge_command
        # No JVM is ever started: one Paper serves every pad.
        assert not any("java" in part or ".jar" in part for part in cmd)

    def test_launch_refuses_when_the_jvm_is_down(self):
        probe = ScriptedProbe({25565: False})

        def fake_popen(cmd, cwd=None):
            raise AssertionError("must not spawn a bridge against a dead JVM")

        launcher = SubprocessArenaLauncher(
            popen=fake_popen, port_probe=probe, sleep=lambda _s: None,
            log=lambda _m: None,
        )
        with pytest.raises(RuntimeError, match="Paper JVM"):
            launcher.launch(0)

    def test_launch_refuses_to_duplicate_a_live_bridge(self):
        # The fleet's bridges are started by start-pads.sh, a different process
        # tree, so this launcher usually has no handle to kill. A port that stays
        # open means that bridge is alive; spawning a second one would fight it.
        probe = ScriptedProbe({25565: True, 5555: True})

        def fake_popen(cmd, cwd=None):
            raise AssertionError("must not spawn a duplicate bridge")

        launcher = SubprocessArenaLauncher(
            popen=fake_popen, port_probe=probe, sleep=lambda _s: None,
            bridge_port_free_timeout_seconds=0.0, log=lambda _m: None,
        )
        with pytest.raises(RuntimeError, match="still alive"):
            launcher.launch(0)

    def test_the_port_free_wait_is_bounded_separately_from_the_ready_wait(self):
        # launch() runs on a collector thread: waiting the full 120s ready timeout
        # just to discover the pad's bridge was alive all along is a stall.
        launcher = SubprocessArenaLauncher()
        assert launcher._bridge_port_free_timeout_seconds < launcher._bridge_ready_timeout_seconds

    def test_launch_raises_when_the_bridge_exits_before_listening(self):
        probe = ScriptedProbe({25565: True, 5555: False})

        def fake_popen(cmd, cwd=None):
            return FakeProc(exit_code=1)

        launcher = SubprocessArenaLauncher(
            popen=fake_popen, port_probe=probe, sleep=lambda _s: None,
            log=lambda _m: None,
        )
        with pytest.raises(RuntimeError, match="exited"):
            launcher.launch(0)

    def test_terminate_stops_the_bridge_and_is_idempotent(self):
        probe = ScriptedProbe({25565: True, 5555: False})
        procs = []

        def fake_popen(cmd, cwd=None):
            proc = FakeProc()
            procs.append(proc)
            probe.open_ports[5555] = True
            return proc

        launcher = SubprocessArenaLauncher(
            popen=fake_popen, port_probe=probe, sleep=lambda _s: None,
            log=lambda _m: None,
        )
        launcher.launch(0)
        launcher.terminate(0)
        assert procs[0].terminated is True
        launcher.terminate(0)  # no handle left: must not raise
        launcher.terminate(99)  # never launched: must not raise

    def test_launch_never_primes(self):
        """A prime here would steal the collector's single TCP client slot."""
        probe = ScriptedProbe({25565: True, 5555: False})

        def fake_popen(cmd, cwd=None):
            probe.open_ports[5555] = True
            return FakeProc()

        launcher = SubprocessArenaLauncher(
            popen=fake_popen, port_probe=probe, sleep=lambda _s: None,
            log=lambda _m: None,
        )
        import distributed.launcher as launcher_module

        called = []
        original = launcher_module.prime_pads
        launcher_module.prime_pads = lambda *a, **k: called.append(a)
        try:
            launcher.launch(0)
        finally:
            launcher_module.prime_pads = original
        assert called == []


# ---------------------------------------------------------------------------
# prime_pads -- the reset-before-step barrier
# ---------------------------------------------------------------------------


class FakeEnv:
    """An env stand-in: records reset/close, optionally fails a bounded number of times."""

    def __init__(self, pad_index, record, failures=0):
        self.pad_index = pad_index
        self._record = record
        self._failures = failures
        self.closed = False

    def reset(self, seed=None):
        if self._failures > 0:
            self._failures -= 1
            raise RuntimeError(f"pad {self.pad_index} not ready")
        self._record.append(("reset", self.pad_index))
        return None

    def close(self):
        self.closed = True
        self._record.append(("close", self.pad_index))


class TestPrimePads:
    def test_resets_every_pad_in_descending_order(self):
        record = []
        envs = []

        def factory(pad_index, host, bridge_port):
            assert host == "127.0.0.1"
            assert bridge_port == 5555 + pad_index
            env = FakeEnv(pad_index, record)
            envs.append(env)
            return env

        primed = prime_pads(4, env_factory=factory, sleep=lambda _s: None,
                            log=lambda _m: None)

        assert primed == [3, 2, 1, 0]
        assert [pad for kind, pad in record if kind == "reset"] == [3, 2, 1, 0]
        # Every connection is handed back so the driver can take the single slot.
        assert all(env.closed for env in envs)

    def test_pad_zero_is_primed_last(self):
        # Pad 0's anchor IS the shared world spawn, so it is the stack site. Primed
        # last, its own reset runs against an already-evacuated pad.
        record = []
        prime_pads(
            5,
            env_factory=lambda i, h, p: FakeEnv(i, record),
            sleep=lambda _s: None,
            log=lambda _m: None,
        )
        resets = [pad for kind, pad in record if kind == "reset"]
        assert resets[-1] == 0

    def test_retries_a_transient_failure_then_succeeds(self):
        record = []
        slept = []
        # A fresh env is built per attempt (a fresh socket to the same bridge), so
        # the "refuse once" budget has to live outside the env.
        budget = {1: 1}

        def factory(pad_index, host, bridge_port):
            failures = budget.get(pad_index, 0)
            budget[pad_index] = 0
            return FakeEnv(pad_index, record, failures=failures)

        primed = prime_pads(
            2, env_factory=factory, attempts=3, backoff_seconds=2.0,
            sleep=slept.append, log=lambda _m: None,
        )
        assert primed == [1, 0]
        assert slept == [2.0]

    def test_backoff_doubles(self):
        slept = []

        def factory(pad_index, host, bridge_port):
            return FakeEnv(pad_index, [], failures=99)

        with pytest.raises(PadPrimeError):
            prime_pads(1, env_factory=factory, attempts=4, backoff_seconds=1.0,
                       sleep=slept.append, log=lambda _m: None)
        assert slept == [1.0, 2.0, 4.0]

    def test_a_dead_pad_raises_loudly_and_names_it(self):
        def factory(pad_index, host, bridge_port):
            if pad_index == 1:
                raise OSError("connection refused")
            return FakeEnv(pad_index, [])

        with pytest.raises(PadPrimeError) as excinfo:
            prime_pads(3, env_factory=factory, attempts=2, sleep=lambda _s: None,
                       log=lambda _m: None)
        message = str(excinfo.value)
        assert "pad 1" in message
        assert "5556" in message            # its bridge port
        assert "512,0" in message           # its anchor
        assert "learner_1/dummy_1" in message
        assert "still stacked in pad 0" in message
        assert "connection refused" in message

    def test_the_env_is_closed_even_when_reset_fails(self):
        envs = []

        def factory(pad_index, host, bridge_port):
            env = FakeEnv(pad_index, [], failures=99)
            envs.append(env)
            return env

        with pytest.raises(PadPrimeError):
            prime_pads(1, env_factory=factory, attempts=2, sleep=lambda _s: None,
                       log=lambda _m: None)
        assert len(envs) == 2
        assert all(env.closed for env in envs)

    def test_failure_dumps_the_pads_bridge_log_tail(self, tmp_path):
        (tmp_path / "pad-0.log").write_text(
            "[bridge] boot\n[bridge] fatal: Error: ECONNREFUSED 127.0.0.1:25565\n",
            encoding="utf-8",
        )

        def factory(pad_index, host, bridge_port):
            raise OSError("connection refused")

        with pytest.raises(PadPrimeError) as excinfo:
            prime_pads(1, env_factory=factory, attempts=1, log_dir=str(tmp_path),
                       sleep=lambda _s: None, log=lambda _m: None)
        assert "ECONNREFUSED" in str(excinfo.value)

    def test_bad_arguments_raise(self):
        with pytest.raises(ValueError):
            prime_pads(0, env_factory=lambda i, h, p: FakeEnv(i, []))
        with pytest.raises(ValueError):
            prime_pads(1, attempts=0, env_factory=lambda i, h, p: FakeEnv(i, []))


# ---------------------------------------------------------------------------
# The CLI
# ---------------------------------------------------------------------------


def _run_launcher_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "distributed.launcher", *args],
        capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=120,
    )


class TestLauncherCli:
    def test_dry_run_is_the_default_mode_and_spawns_nothing(self):
        result = _run_launcher_cli("--pads", "2")
        assert result.returncode == 0
        assert "launch plan for 2 pad(s)" in result.stdout
        assert "no processes were started" in result.stderr

    def test_emit_plan_is_tab_separated_and_parses_back(self):
        result = _run_launcher_cli("--pads", "3", "--emit-plan")
        assert result.returncode == 0
        lines = result.stdout.strip().split("\n")
        assert len(lines) == 3
        for index, line in enumerate(lines):
            fields = line.split("\t")
            assert fields[0] == str(index)
            assert fields[1] == str(5555 + index)
            anchor = pad_anchor(index)
            assert fields[2] == str(anchor.x)
            assert fields[3] == str(anchor.z)
            assert (fields[4], fields[5]) == pad_usernames(index)
            assert fields[6:] == plan(3, repo_root=str(REPO_ROOT))[index]["bridge_command"]
            # No field may be empty: start-pads.sh splits on IFS=$'\t', which
            # collapses runs of tabs.
            assert all(field != "" for field in fields)

    def test_write_ops_then_check_ops(self, tmp_path):
        ops_path = str(tmp_path / "ops.json")
        written = _run_launcher_cli("--pads", "3", "--write-ops", "--ops-path", ops_path)
        assert written.returncode == 0
        ok = _run_launcher_cli("--pads", "3", "--check-ops", "--ops-path", ops_path)
        assert ok.returncode == 0
        short = _run_launcher_cli("--pads", "5", "--check-ops", "--ops-path", ops_path)
        assert short.returncode == 1
        assert "learner_3" in short.stderr

    def test_zero_pads_is_rejected(self):
        result = _run_launcher_cli("--pads", "0")
        assert result.returncode != 0

    def test_modes_are_mutually_exclusive(self):
        result = _run_launcher_cli("--pads", "1", "--dry-run", "--write-ops")
        assert result.returncode != 0


# ---------------------------------------------------------------------------
# TC19 -- the max-players guard, live in the shell scripts
# ---------------------------------------------------------------------------


REAL_DATAPACK = REPO_ROOT / "server" / "arena"


def _fake_server_root(tmp_path, max_players):
    """A scratch SERVER_DIR complete enough for start-pads.sh's preflight.

    The datapack is copied into BOTH the source location and the world copy,
    because the preflight checks the world copy is CURRENT (identical to
    server/arena), not merely present.
    """
    root = tmp_path / "server"
    root.mkdir(parents=True)
    shutil.copytree(REAL_DATAPACK, root / "arena")
    (root / "world" / "datapacks").mkdir(parents=True)
    shutil.copytree(REAL_DATAPACK, root / "world" / "datapacks" / "arena")
    (root / "eula.txt").write_text("eula=true\n", encoding="utf-8")
    (root / "paper-1.21.1-133.jar").write_text("not really a jar", encoding="utf-8")
    (root / "server.properties").write_text(
        "# scratch\nserver-port=25565\nmax-players={}\n".format(max_players),
        encoding="utf-8",
    )
    write_ops_json(25, str(root / "ops.json"))
    return root


def _run_start_pads(server_root, *args):
    env = dict(os.environ)
    env["SERVER_DIR"] = str(server_root)
    return subprocess.run(
        ["bash", str(START_PADS_SH), "--python", sys.executable, *args],
        capture_output=True, text=True, cwd=str(REPO_ROOT), env=env, timeout=300,
    )


class TestStartPadsPreflight:
    def test_help_exits_zero(self):
        result = subprocess.run(
            ["bash", str(START_PADS_SH), "--help"],
            capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=60,
        )
        assert result.returncode == 0
        assert "--pads N" in result.stdout

    def test_unknown_flag_is_rejected(self):
        result = subprocess.run(
            ["bash", str(START_PADS_SH), "--nope"],
            capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=60,
        )
        assert result.returncode != 0
        assert "unknown argument" in result.stderr

    def test_zero_pads_is_rejected(self):
        result = subprocess.run(
            ["bash", str(START_PADS_SH), "--pads", "0"],
            capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=60,
        )
        assert result.returncode != 0
        assert "positive integer" in result.stderr

    def test_max_players_guard_fails_loudly_with_the_requirement(self, tmp_path):
        """TC19: 8 pads need 26 slots; a 20-slot server must refuse to launch."""
        root = _fake_server_root(tmp_path, max_players=20)
        result = _run_start_pads(root, "--pads", "8", "--check")
        combined = result.stdout + result.stderr
        assert result.returncode != 0
        assert "max-players=20" in combined
        assert "need at least 26" in combined
        assert "2*8+10" in combined
        assert "PADS=8 server/setup/setup.sh" in combined

    def test_a_sufficient_max_players_passes_the_guard(self, tmp_path):
        root = _fake_server_root(tmp_path, max_players=26)
        result = _run_start_pads(root, "--pads", "8", "--check")
        combined = result.stdout + result.stderr
        assert "ok   max-players=26 >= 26" in combined
        assert "need at least" not in combined

    def test_server_port_mismatch_is_caught(self, tmp_path):
        root = _fake_server_root(tmp_path, max_players=26)
        result = _run_start_pads(root, "--pads", "2", "--check", "--mc-port", "25599")
        combined = result.stdout + result.stderr
        assert result.returncode != 0
        assert "server-port=25565" in combined

    def test_a_current_datapack_passes(self, tmp_path):
        root = _fake_server_root(tmp_path, max_players=26)
        result = _run_start_pads(root, "--pads", "2", "--check")
        assert "ok   arena datapack installed and matches server/arena" in (
            result.stdout + result.stderr
        )

    def test_a_world_copy_predating_the_pad_topology_is_caught(self, tmp_path):
        """Presence is not currency: Paper loads the WORLD copy, not server/arena."""
        root = _fake_server_root(tmp_path, max_players=26)
        (root / "world" / "datapacks" / "arena" / "data" / "arena" / "function"
         / "setup_pad.mcfunction").unlink()
        result = _run_start_pads(root, "--pads", "2", "--check")
        combined = result.stdout + result.stderr
        assert result.returncode != 0
        assert "arena:setup_pad" in combined
        assert "predates the pad topology" in combined

    def test_a_stale_world_copy_is_caught(self, tmp_path):
        """A macro function that exists but no longer matches the source."""
        root = _fake_server_root(tmp_path, max_players=26)
        stale = (root / "world" / "datapacks" / "arena" / "data" / "arena"
                 / "function" / "reset_pad.mcfunction")
        stale.write_text(stale.read_text(encoding="utf-8") + "\n# drifted\n",
                         encoding="utf-8")
        result = _run_start_pads(root, "--pads", "2", "--check")
        combined = result.stdout + result.stderr
        assert result.returncode != 0
        assert "stale macro functions" in combined

    @staticmethod
    def _code_only(text):
        """Drop comment-only lines so an assertion sees CALLS, not prose about them."""
        return "\n".join(
            line for line in text.splitlines() if not line.lstrip().startswith("#")
        )

    def test_preflight_never_connects_to_a_bridge_port(self):
        """The CRITICAL: --check must not evict a live fleet's collectors.

        BridgeServer accepts ONE TCP client and resolves a second by destroying the
        incumbent, so a connect-based occupancy scan silently kills every attached
        driver. Assert at the source level that the bridge-port scan uses the
        non-connecting detector and that the connect probe is named for what it does.
        """
        text = START_PADS_SH.read_text(encoding="utf-8")
        assert "port_open()" not in text, "the connect probe must not read as read-only"
        scan = self._code_only(
            text.split('OCCUPIED_BRIDGE_PORTS=""', 1)[1].split("done", 1)[0]
        )
        assert "listener_pids" in scan
        assert "connect_probe" not in scan

    def test_supervision_never_connects_to_a_bridge_port(self):
        """Same hazard, worse: the supervise loop polls forever, every 5s."""
        text = START_PADS_SH.read_text(encoding="utf-8")
        supervise = self._code_only(text.split("# Tier 2: individual bridges", 1)[1])
        assert "bridge_pids_on_port" in supervise
        assert "connect_probe" not in supervise, (
            "the supervise loop must never open a connection to a bridge a driver owns"
        )

    def test_the_unprimed_banner_does_not_collide_with_the_ready_token(self):
        """RUNBOOK tells operators to wait for FLEET READY; an unprimed fleet must
        not match that substring, and must not print a driver command."""
        text = START_PADS_SH.read_text(encoding="utf-8")
        printed = [
            line for line in text.splitlines()
            if line.strip().startswith("log \"FLEET")
        ]
        assert len(printed) == 2
        ready = [line for line in printed if "FLEET READY" in line]
        assert len(ready) == 1, "FLEET READY must identify exactly one state"
        assert any("FLEET NOT PRIMED" in line for line in printed)

    def test_dry_run_prints_the_plan_and_starts_nothing(self, tmp_path):
        root = _fake_server_root(tmp_path, max_players=26)
        result = _run_start_pads(root, "--pads", "3", "--dry-run")
        assert result.returncode == 0, result.stderr
        assert "launch plan for 3 pad(s)" in result.stdout
        assert "--pad-origin 1024,0" in result.stdout
        assert "nothing was started" in result.stdout


@pytest.mark.skipif(
    not PAPER_JAR.is_file(),
    reason="setup.sh verifies the pinned Paper jar's sha256; it is not in git",
)
class TestSetupShMaxPlayers:
    """setup.sh REGENERATES server.properties, so max-players must live there."""

    @staticmethod
    def _run_setup(tmp_path, pads=None):
        root = tmp_path / "server"
        root.mkdir()
        # Symlink the already-verified jar so setup.sh skips the download and its
        # sha256 gate still passes (the gate itself is not bypassed).
        (root / PAPER_JAR.name).symlink_to(PAPER_JAR)
        env = dict(os.environ)
        env["SERVER_DIR"] = str(root)
        if pads is not None:
            env["PADS"] = str(pads)
        result = subprocess.run(
            ["bash", str(SETUP_SH)],
            capture_output=True, text=True, cwd=str(REPO_ROOT), env=env, timeout=300,
        )
        assert result.returncode == 0, result.stderr
        props = (root / "server.properties").read_text(encoding="utf-8")
        value = next(
            line.split("=", 1)[1]
            for line in props.splitlines()
            if line.startswith("max-players=")
        )
        return int(value), props, result

    def test_default_run_is_unchanged_at_twenty(self, tmp_path):
        # The historical value: a default setup must not shrink today's server.
        value, _, _ = self._run_setup(tmp_path)
        assert value == 20

    @pytest.mark.parametrize("pads", [8, 25])
    def test_large_fleets_get_at_least_two_n_plus_ten(self, tmp_path, pads):
        value, _, _ = self._run_setup(tmp_path, pads=pads)
        assert value >= required_max_players(pads)

    def test_small_fleets_keep_the_floor_of_twenty(self, tmp_path):
        value, _, _ = self._run_setup(tmp_path, pads=4)
        assert value == 20 >= required_max_players(4)

    def test_generated_properties_have_no_command_substitution_damage(self, tmp_path):
        # The properties heredoc is unquoted (it interpolates the seed and
        # max-players), so an unescaped backtick would run as a command and blank
        # the line. Assert the comment survived intact.
        _, props, result = self._run_setup(tmp_path, pads=2)
        assert "`gamerule fallDamage false`" in props
        assert "command not found" not in result.stderr

    def test_a_bad_pad_count_is_rejected(self, tmp_path):
        root = tmp_path / "server"
        root.mkdir()
        env = dict(os.environ)
        env["SERVER_DIR"] = str(root)
        env["PADS"] = "0"
        result = subprocess.run(
            ["bash", str(SETUP_SH)],
            capture_output=True, text=True, cwd=str(REPO_ROOT), env=env, timeout=60,
        )
        assert result.returncode != 0
        assert "positive integer" in result.stderr
