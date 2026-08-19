"""test_selfplay_driver — the frozen past-self opponent and its wiring (T10/T11b, M4).

Every pin here exists because the corresponding failure is SILENT: the run keeps
going, every log line still says ``selfplay``, and the only symptom is an Elo
curve that means nothing when somebody reads it the next morning.

* **``--opponent selfplay`` never reaching the collectors** (AC13). The
  multi-arena loop builds an opponent factory per opponent choice; a choice with
  no branch leaves ``opponent_for`` at ``None`` and the whole night trains
  against the stationary bridge-served dummy while the logger's config record
  says otherwise. ``TestSelfPlayReachesTheWire`` is the test that bites: it runs
  ``train_multi_arena`` end to end over fake pads and reads ``opp_action`` off
  the wire.
* **The mirror missing at either env construction site.**
  ``opponent_observation()`` raises without ``mirror_opponent=True``, so the
  TRAINING factory's mistake is loud on episode 1 — but the EVAL env is built
  somewhere else entirely and its mistake surfaces only at the first eval cycle,
  potentially an hour in. Both sites are pinned.
* **A carried-over LSTM hidden state.** A DRQN whose memory survives the episode
  boundary still returns legal macros; it just plays the END of the previous
  fight. Nothing anywhere reports it.
* **Two arenas sharing one driver.** ``ActorPool.build`` calls the factory once
  per arena; one shared object would put N threads on one LSTM state and one
  ε-greedy generator, correlating the very arenas the per-arena seed bands exist
  to keep independent.
* **The scripted and dummy paths drifting** (AC10). A scripted driver must still
  receive an ``OpponentView``, and a dummy step line must still carry no
  ``opp_action`` at all.

T11b adds the CLI half of the same story, where the failures are quieter still:

* **``--reference-promote-grad-steps`` arriving as a LIST.** ``argparse``
  ``nargs=2`` yields one, ``TrainConfig`` rejects one, and the reason it rejects
  one is that a list field makes ``hash(cfg)`` raise somewhere unrelated.
* **A warm start that is not the checkpoint the run was sized against** (AC14).
  It loads cleanly into the same architecture, and snapshot 0 — the pinned
  reference both Elo series are measured against — is seeded from it.
* **Two runs sharing one pool of past selves.** The pool directory is derived
  from ``--run-name``; unset, every run resolves to ``runs/selfplay/snapshots``
  and a second run silently resumes the first one's snapshots and Elo series.
* **The middle hop of the mirror chain.** ``_eval_via_designated_arena`` forwards
  ``mirror_opponent`` to ``_eval_against_opponent``; every OTHER test of the eval
  path monkeypatches that function out, so without the pin here the forwarding
  line can be deleted with a fully green suite.

House conventions: no sockets, no live server, no Minecraft — every bridge and
every pad here is a fake, and torch is optional via ``pytest.importorskip``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import threading
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock

import numpy as np
import pytest

from agent.actions import Macro, N_ACTIONS
from agent.train_config import TrainConfig
from env.observation_spec import OBS_DIM
from opponents.scripted_bot import OpponentView


# Seconds a worker thread gets before a test calls it a stall. Generous enough
# to absorb a loaded box, short enough that a genuine block fails the run
# instead of wedging pytest.
_THREAD_TIMEOUT = 30.0

#: Shrunken net so constructing ~4 of them per test is cheap. ``obs_dim`` /
#: ``n_actions`` are NOT overridable — ``DuelingDRQN`` asserts them against the
#: frozen contracts.
_TINY_NET = {"encoder_hidden": 16, "lstm_hidden": 16, "lstm_layers": 1}

#: Every seed role one arena's opponent drivers use, scripted and self-play
#: together. ``"eval"`` is excluded because the eval driver is seeded from arena
#: band ``cfg.arenas`` — one past the last collector — not from a collector's.
_OPPONENT_ROLES = (
    "mixture",
    "easy",
    "hard",
    "snapshot_sample",
    "snapshot_epsilon",
)


def _completes_within(fn, timeout: float = _THREAD_TIMEOUT):
    """Run ``fn`` on a daemon thread and FAIL (never hang) if it does not finish."""
    box: Dict[str, Any] = {}

    def _run() -> None:
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 - re-raised on the main thread
            box["error"] = exc

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        pytest.fail(f"call did not finish within {timeout}s (it blocked)")
    if "error" in box:
        raise box["error"]
    return box.get("value")


# ===========================================================================
# Fixtures: checkpoints with a KNOWN greedy action, fake envs, fake policies.
# ===========================================================================


def _net_factory():
    """Zero-arg builder for the tiny net every driver in this file owns."""
    from agent.dqn import DuelingDRQN

    return DuelingDRQN(**_TINY_NET)


def _write_constant_action_checkpoint(path, action: Macro):
    """Save a checkpoint whose greedy action is ``action`` for ANY observation.

    The advantage head's WEIGHT is zeroed and its BIAS set to a one-hot, so
    ``A(s, ·) == bias`` regardless of the encoder/LSTM features. The dueling
    aggregation adds the same scalar ``V(s)`` to every entry and subtracts the
    same mean advantage, neither of which can reorder the row — so
    ``argmax_a Q(s, a) == action`` for every state and every hidden state.

    That is what lets the end-to-end test say something exact: an ``opp_action``
    equal to ``action`` on every step can only have come from THESE weights.
    A stationary dummy sends no ``opp_action`` at all, and any other net would
    have to hit one specific macro on every step of every episode.

    Args:
        path: Destination file (a ``tmp_path`` child).
        action: The macro the saved policy plays.

    Returns:
        The path as a ``str``, ready for ``TrainConfig.warm_start``.
    """
    import torch

    net = _net_factory()
    with torch.no_grad():
        net.advantage_head.weight.zero_()
        bias = torch.full((N_ACTIONS,), -1.0)
        bias[int(action)] = 5.0
        net.advantage_head.bias.copy_(bias)
    torch.save(
        {
            "model": {k: v.clone() for k, v in net.state_dict().items()},
            "grad_step": 0,
            "code_version": "test",
        },
        path,
    )
    return str(path)


def _write_default_checkpoint(path):
    """Save a freshly initialized tiny net — a policy whose actions DO vary."""
    import torch

    net = _net_factory()
    torch.save(
        {
            "model": {k: v.clone() for k, v in net.state_dict().items()},
            "grad_step": 0,
            "code_version": "test",
        },
        path,
    )
    return str(path)


def _obs(seed: int) -> np.ndarray:
    """A deterministic, well-shaped ``(OBS_DIM,)`` observation."""
    rng = np.random.default_rng(int(seed))
    return rng.uniform(-1.0, 1.0, size=OBS_DIM).astype(np.float32)


def _view(**overrides: Any) -> OpponentView:
    """A hand-authored ``OpponentView``: target ahead, in range, swing ready."""
    base: Dict[str, Any] = dict(
        self_pos=(0.0, 64.0, 0.0),
        self_yaw=0.0,
        self_health=20.0,
        target_pos=(0.0, 64.0, 2.0),
        target_yaw=180.0,
        target_health=20.0,
        distance=2.0,
        in_attack_range=True,
        # Exactly 1.0: the producer clamps it there and ScriptedBot tests
        # readiness at >= 1.0 - 1e-6.
        attack_cooldown=1.0,
        can_see_target=True,
        last_known_target_pos=(0.0, 64.0, 2.0),
    )
    base.update(overrides)
    return OpponentView(**base)


class _MirrorPadEnv:
    """Fake pad env exposing the SELF-PLAY seam: ``opponent_observation()``.

    Serves a MIRRORED observation that is deliberately different from the
    learner's, so a test can prove which one a driver was handed. Both
    accessors are counted: the point of the routing branch is that exactly one
    of them is ever called for a given driver, and a count of zero on the other
    is what fails if the branch inverts.
    """

    #: Constant offset separating the mirrored vector from the learner's, so an
    #: assertion can name which seat an array came from.
    MIRROR_OFFSET = 0.25

    def __init__(self, *, k: int = 4, won: bool = False) -> None:
        self.k = int(k)
        self._won = bool(won)
        self._t = 0
        self._obs_index = 0
        self.step_calls: List[Tuple[int, Optional[int]]] = []
        self.opp_actions: List[Optional[int]] = []
        self.observation_calls = 0
        self.view_calls = 0
        self.mirrored: List[np.ndarray] = []
        self.closed = False

    # -- EnvProtocol + the opponent seam ----------------------------------

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        self._t = 0
        self._obs_index = 0 if seed is None else int(seed)
        return self._learner_obs()

    def opponent_observation(self) -> np.ndarray:
        self.observation_calls += 1
        vector = (self._learner_obs() + self.MIRROR_OFFSET).astype(np.float32)
        self.mirrored.append(vector)
        return vector

    def raw_opponent_view(self) -> OpponentView:
        # Present but counted, NOT raising: a raise here would surface on a
        # collector thread inside the actor pool, where it becomes a relaunch
        # storm rather than a test failure. The zero-count assertion is the
        # check; this just makes the wrong branch survivable enough to report.
        self.view_calls += 1
        return _view()

    def step(
        self, action: int, opp_action: Optional[int] = None
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        self.step_calls.append((int(action), opp_action))
        self.opp_actions.append(opp_action)
        self._t += 1
        self._obs_index += 1
        done = self._t >= self.k
        info = {
            "step": self._t,
            "won": bool(done and self._won),
            "lost": bool(done and not self._won),
            "timeout": False,
        }
        return self._learner_obs(), 0.0, done, info

    def close(self) -> None:
        self.closed = True

    # -- helpers -----------------------------------------------------------

    def _learner_obs(self) -> np.ndarray:
        return _obs(self._obs_index)


class _FixedPolicy:
    """A ``RolloutPolicy`` that always picks one action and carries a 1x1x2 hidden."""

    arena_id = 0
    policy_version = 0
    code_version = ""

    def __init__(self, torch_mod, action: int = int(Macro.IDLE)) -> None:
        self._torch = torch_mod
        self._action = int(action)
        self.seeds: List[int] = []

    def reseed(self, episode_seed: int) -> None:
        self.seeds.append(int(episode_seed))

    def init_hidden(self):
        zeros = self._torch.zeros(1, 1, 2)
        return (zeros, zeros.clone())

    def act(self, obs, hidden, epsilon):
        return self._action, hidden


class _RecordingObservationOpponent:
    """An ``ObservationOpponent`` recording every array it was handed."""

    needs_observation = True

    def __init__(self, macro: Macro = Macro.JUMP) -> None:
        self._macro = macro
        self.calls: List[str] = []
        self.observations: List[np.ndarray] = []
        self.outcomes: List[Dict[str, Any]] = []
        self.noted_epsilons: List[float] = []

    def note_learner_epsilon(self, epsilon: float) -> None:
        self.noted_epsilons.append(float(epsilon))

    def begin_episode(self) -> None:
        self.calls.append("begin")

    def act(self, obs) -> int:
        self.calls.append("act")
        self.observations.append(np.asarray(obs))
        return int(self._macro)

    def observe_outcome(self, info) -> None:
        self.calls.append("outcome")
        self.outcomes.append(dict(info))


class _RecordingViewOpponent:
    """An ``EpisodeOpponent`` recording every view it was handed.

    Deliberately has NO ``needs_observation`` attribute: that absence is what
    routes it to the historical view branch, and adding one here would stop the
    scripted-path pins from testing the real discriminator.
    """

    def __init__(self, macro: Macro = Macro.APPROACH) -> None:
        self._macro = macro
        self.calls: List[str] = []
        self.views: List[Any] = []

    def begin_episode(self) -> None:
        self.calls.append("begin")

    def act(self, view) -> int:
        self.calls.append("act")
        self.views.append(view)
        return int(self._macro)

    def observe_outcome(self, info) -> None:
        self.calls.append("outcome")


def _selfplay_cfg(warm_start: str, **overrides: Any) -> TrainConfig:
    """Minimal multi-arena SELF-PLAY config: tiny windows, instant warm-up."""
    base: Dict[str, Any] = dict(
        arenas=2,
        batch_size=4,
        seq_len=2,
        burn_in=1,
        n_step=1,
        min_replay=1,
        replay_capacity=2_000,
        weight_sync_every_k_steps=1,
        fault_relaunch=False,
        seed=0,
        opponent="selfplay",
        warm_start=str(warm_start),
        # Exactly 0.0 so the frozen net is purely greedy and the wire assertion
        # can be an equality rather than a distribution.
        opponent_epsilon=0.0,
    )
    base.update(overrides)
    return dataclasses.replace(TrainConfig(), **base)


def _build_opponents(cfg: TrainConfig, snapshot_dir):
    """Build the pool + per-arena driver factory over a throwaway directory."""
    from agent.train import build_snapshot_opponents

    return build_snapshot_opponents(
        cfg, snapshot_dir=str(snapshot_dir), net_factory=_net_factory
    )


def _run_multi_arena(cfg: TrainConfig, pads: List[Any], **kwargs: Any):
    """Drive ``train_multi_arena`` over fake pads to a small grad-step budget."""
    from distributed.actor import GlobalEpisodeCounter
    from distributed.learner import LearnerWatchdog
    from distributed.transport import LocalTransport
    from distributed.weights import WeightStore

    from agent.train import train_multi_arena

    def env_factory_for(arena_id: int):
        return lambda: pads[arena_id]

    call: Dict[str, Any] = dict(
        env_factory_for=env_factory_for,
        launcher=MagicMock(),
        transport=LocalTransport(maxsize=0),
        weight_store=WeightStore(),
        counter=GlobalEpisodeCounter(),
        max_grad_steps=20,
        eval_every_grad_steps=0,
        designated_arena=0,
        stop_on_pass=False,
        relaunch_backoff_seconds=0.001,
        relaunch_backoff_max_seconds=0.001,
        sleep=lambda _s: None,
        poll_interval=0.005,
        net_kwargs=dict(_TINY_NET),
        rollout_step_cap=6,
        eval_episodes=1,
        eval_pause_timeout=1.0,
        watchdog=LearnerWatchdog(patience=500, interval_s=1.0),
    )
    call.update(kwargs)
    return _completes_within(lambda: train_multi_arena(cfg, **call))


# ===========================================================================
# The protocol and its discriminator.
# ===========================================================================


class TestTheObservationOpponentProtocol:
    """``needs_observation`` is the branch key, and only the new kind carries it."""

    def test_the_snapshot_driver_declares_it_at_class_level(self):
        pytest.importorskip("torch")

        from agent.train import SnapshotOpponentDriver

        # Class level, not instance level: collect_episode resolves the branch
        # from the driver object without instantiating anything.
        assert SnapshotOpponentDriver.needs_observation is True

    def test_the_scripted_driver_does_not_declare_it(self):
        pytest.importorskip("torch")

        from agent.train import (
            OpponentCurriculum,
            ScriptedOpponentDriver,
            _needs_observation,
        )

        cfg = TrainConfig()
        driver = ScriptedOpponentDriver(cfg, OpponentCurriculum(cfg), 0)

        assert not hasattr(driver, "needs_observation")
        assert _needs_observation(driver) is False

    def test_none_and_unknown_objects_route_to_the_view_path(self):
        from agent.train import _needs_observation

        assert _needs_observation(None) is False
        assert _needs_observation(object()) is False
        assert _needs_observation(_RecordingViewOpponent()) is False
        assert _needs_observation(_RecordingObservationOpponent()) is True


# ===========================================================================
# build_snapshot_opponents: the pool seed + the per-arena factory.
# ===========================================================================


class TestBuildSnapshotOpponents:
    """The pool is seeded from the warm start and every arena gets its own driver."""

    def test_snapshot_zero_is_seeded_from_the_warm_start_and_pinned(self, tmp_path):
        pytest.importorskip("torch")
        import torch

        warm = _write_constant_action_checkpoint(tmp_path / "warm.pt", Macro.JUMP)
        pool, _ = _build_opponents(_selfplay_cfg(warm), tmp_path / "pool")

        assert len(pool) == 1
        record = pool.get(0)
        assert record is not None
        assert record.pinned is True, (
            "snapshot 0 must be PINNED: it is the run's floor opponent and its "
            "fixed Elo yardstick, and an unpinned member can be dropped"
        )
        assert record.grad_step == 0
        assert record.elo == pytest.approx(TrainConfig().elo_initial)

        # The bytes on disk are the warm start's, not a fresh init.
        from agent.train import load_checkpoint_state_dict

        expected = load_checkpoint_state_dict(warm)
        stored = pool.load_state_dict(record)
        assert set(stored) == set(expected)
        assert all(torch.equal(stored[k], expected[k]) for k in expected)

    def test_a_missing_warm_start_is_refused(self, tmp_path):
        pytest.importorskip("torch")

        from agent.train import build_snapshot_opponents

        cfg = dataclasses.replace(TrainConfig(), opponent="dummy", warm_start=None)
        with pytest.raises(ValueError, match="warm_start"):
            build_snapshot_opponents(cfg, snapshot_dir=str(tmp_path / "pool"))

    def test_an_existing_pool_is_reloaded_not_reseeded(self, tmp_path):
        pytest.importorskip("torch")

        warm = _write_constant_action_checkpoint(tmp_path / "warm.pt", Macro.JUMP)
        cfg = _selfplay_cfg(warm)
        directory = tmp_path / "pool"

        first, _ = _build_opponents(cfg, directory)
        assert len(first) == 1

        second, _ = _build_opponents(cfg, directory)
        # A second seed would allocate snapshot id 1 for the SAME weights, and a
        # fresh pool over a populated directory would restart the id counter and
        # let the next add overwrite snap_0.pt — the pinned reference.
        assert len(second) == 1
        assert second.get(0) is not None

    def test_the_default_directory_is_derived_from_the_run_name(self):
        from agent.train import snapshot_pool_directory

        assert snapshot_pool_directory("m4_selfplay").replace("\\", "/") == (
            "runs/m4_selfplay/snapshots"
        )

    @pytest.mark.parametrize("bad", ["", "   "])
    def test_an_empty_run_name_is_refused(self, bad):
        from agent.train import snapshot_pool_directory

        # "runs/snapshots" would silently merge two runs' pools of past selves.
        with pytest.raises(ValueError, match="run_name"):
            snapshot_pool_directory(bad)


# ===========================================================================
# TC20 — per-arena driver isolation.
# ===========================================================================


class TestPerArenaDriverIsolation:
    """TC20: distinct objects, independent RNG streams, independent LSTM state.

    ``ActorPool.build`` calls the factory once per arena and hands the result
    straight to that arena's collector thread. One shared driver would mean N
    threads mutating one LSTM hidden state mid-fight and drawing from one
    ε-greedy generator — the exact cross-arena correlation the per-arena seed
    bands exist to prevent, and invisible in every metric.
    """

    def test_each_arena_gets_its_own_driver_and_its_own_net(self, tmp_path):
        pytest.importorskip("torch")

        warm = _write_default_checkpoint(tmp_path / "warm.pt")
        cfg = _selfplay_cfg(warm, arenas=3)
        _, opponent_for = _build_opponents(cfg, tmp_path / "pool")

        drivers = [opponent_for(i) for i in range(3)]

        assert len({id(d) for d in drivers}) == 3
        assert len({id(d.net) for d in drivers}) == 3
        assert [d.arena_id for d in drivers] == [0, 1, 2]

    def test_the_same_arena_is_memoized(self, tmp_path):
        pytest.importorskip("torch")

        warm = _write_default_checkpoint(tmp_path / "warm.pt")
        _, opponent_for = _build_opponents(_selfplay_cfg(warm), tmp_path / "pool")

        assert opponent_for(0) is opponent_for(0)
        assert opponent_for(1) is not opponent_for(0)

    def test_the_memoizing_factory_is_thread_safe(self, tmp_path):
        pytest.importorskip("torch")

        warm = _write_default_checkpoint(tmp_path / "warm.pt")
        cfg = _selfplay_cfg(warm, arenas=8)
        _, opponent_for = _build_opponents(cfg, tmp_path / "pool")

        seen: List[Any] = []
        seen_lock = threading.Lock()
        start = threading.Event()

        def _grab(arena_id: int) -> None:
            start.wait(timeout=_THREAD_TIMEOUT)
            driver = opponent_for(arena_id)
            with seen_lock:
                seen.append((arena_id, id(driver)))

        # Every arena requested twice, concurrently: an unguarded dict can hand
        # two collectors two different objects for the same arena.
        threads = [
            threading.Thread(target=_grab, args=(arena,), daemon=True)
            for arena in list(range(8)) * 2
        ]
        for thread in threads:
            thread.start()
        start.set()
        for thread in threads:
            thread.join(timeout=_THREAD_TIMEOUT)
            assert not thread.is_alive()

        by_arena: Dict[int, set] = {}
        for arena_id, driver_id in seen:
            by_arena.setdefault(arena_id, set()).add(driver_id)
        assert len(seen) == 16
        assert all(len(ids) == 1 for ids in by_arena.values())
        assert len({next(iter(ids)) for ids in by_arena.values()}) == 8

    def test_the_epsilon_streams_are_independent(self, tmp_path):
        pytest.importorskip("torch")

        warm = _write_default_checkpoint(tmp_path / "warm.pt")
        # epsilon 1.0 makes every action a draw from the driver's OWN generator,
        # so the recorded stream IS the RNG stream.
        cfg = _selfplay_cfg(warm, opponent_epsilon=1.0)
        observations = [_obs(i) for i in range(24)]

        def _streams(interleaved: bool) -> Tuple[List[int], List[int]]:
            _, opponent_for = _build_opponents(cfg, tmp_path / f"pool{interleaved}")
            zero = opponent_for(0)
            one = opponent_for(1)
            zero.begin_episode()
            if interleaved:
                one.begin_episode()
            zero_actions: List[int] = []
            one_actions: List[int] = []
            for obs in observations:
                zero_actions.append(zero.act(obs))
                if interleaved:
                    one_actions.append(one.act(obs))
            return zero_actions, one_actions

        solo, _ = _streams(False)
        interleaved_zero, interleaved_one = _streams(True)

        # The behavioral form of "no shared generator": arena 0's stream is
        # identical whether or not arena 1 drew in between. A shared generator
        # makes the interleaved run consume every other draw.
        assert solo == interleaved_zero
        # And the two arenas are not replaying ONE stream: an arena-blind seed
        # would make every pad in the fleet explore identically, which the
        # equality above cannot see.
        assert interleaved_zero != interleaved_one

    def test_stepping_one_arena_does_not_touch_anothers_hidden_state(self, tmp_path):
        pytest.importorskip("torch")

        warm = _write_default_checkpoint(tmp_path / "warm.pt")
        _, opponent_for = _build_opponents(_selfplay_cfg(warm), tmp_path / "pool")

        zero = opponent_for(0)
        one = opponent_for(1)
        zero.begin_episode()
        one.begin_episode()

        for i in range(4):
            zero.act(_obs(i))

        assert zero.hidden is not None
        assert one.hidden is None, (
            "arena 1's LSTM advanced while only arena 0 acted: the two arenas "
            "are sharing one hidden state"
        )


# ===========================================================================
# TC21 — the LSTM hidden state is reset every episode.
# ===========================================================================


class TestBeginEpisodeResetsTheHiddenState:
    """TC21: a hidden state must never survive an episode boundary.

    A DRQN carrying the last episode's memory still returns legal macros, so
    nothing downstream complains — the opponent simply plays the END of a
    different fight for the first several decisions of this one.
    """

    def test_the_hidden_state_is_none_at_the_first_step_of_each_episode(
        self, tmp_path
    ):
        pytest.importorskip("torch")

        warm = _write_default_checkpoint(tmp_path / "warm.pt")
        _, opponent_for = _build_opponents(_selfplay_cfg(warm), tmp_path / "pool")
        driver = opponent_for(0)

        for _episode in range(3):
            driver.begin_episode()
            assert driver.hidden is None
            for i in range(3):
                driver.act(_obs(i))
            assert driver.hidden is not None

    def test_two_episodes_over_the_same_inputs_reach_the_same_state(self, tmp_path):
        pytest.importorskip("torch")
        import torch

        warm = _write_default_checkpoint(tmp_path / "warm.pt")
        # epsilon 0.0 removes the RNG entirely, so any difference between the
        # two episodes can only come from carried recurrent memory.
        cfg = _selfplay_cfg(warm, opponent_epsilon=0.0)
        _, opponent_for = _build_opponents(cfg, tmp_path / "pool")
        driver = opponent_for(0)
        inputs = [_obs(i) for i in range(5)]

        def _episode() -> Tuple[List[int], Tuple[Any, Any]]:
            driver.begin_episode()
            actions = [driver.act(obs) for obs in inputs]
            hidden = driver.hidden
            return actions, (hidden[0].clone(), hidden[1].clone())

        first_actions, first_hidden = _episode()
        second_actions, second_hidden = _episode()

        assert first_actions == second_actions
        assert torch.equal(first_hidden[0], second_hidden[0])
        assert torch.equal(first_hidden[1], second_hidden[1])


# ===========================================================================
# The driver's own contract: snapshot loading, scoring, epsilons.
# ===========================================================================


class TestSnapshotOpponentDriverContract:
    """The frozen net, the match record, and how an episode is scored."""

    def test_the_net_is_a_cpu_clone_in_eval_mode(self, tmp_path):
        pytest.importorskip("torch")
        import torch

        warm = _write_default_checkpoint(tmp_path / "warm.pt")
        _, opponent_for = _build_opponents(_selfplay_cfg(warm), tmp_path / "pool")
        driver = opponent_for(0)

        assert driver.net.training is False
        assert all(
            param.device == torch.device("cpu") for param in driver.net.parameters()
        )

    def test_act_builds_no_autograd_graph(self, tmp_path):
        pytest.importorskip("torch")
        import torch

        warm = _write_default_checkpoint(tmp_path / "warm.pt")
        _, opponent_for = _build_opponents(_selfplay_cfg(warm), tmp_path / "pool")
        driver = opponent_for(0)
        driver.begin_episode()
        driver.act(_obs(0))

        h, c = driver.hidden
        assert not h.requires_grad and h.grad_fn is None
        assert not c.requires_grad and c.grad_fn is None
        assert torch.is_grad_enabled(), "act must not leak a disabled-grad context"

    def test_begin_episode_loads_the_sampled_snapshots_weights(self, tmp_path):
        pytest.importorskip("torch")
        import torch

        warm = _write_constant_action_checkpoint(tmp_path / "warm.pt", Macro.RETREAT)
        _, opponent_for = _build_opponents(_selfplay_cfg(warm), tmp_path / "pool")
        driver = opponent_for(0)

        assert driver.snapshot_id is None
        driver.begin_episode()
        assert driver.snapshot_id == 0
        assert driver.name == "snapshot_0"

        from agent.train import load_checkpoint_state_dict

        expected = load_checkpoint_state_dict(warm)
        live = driver.net.state_dict()
        assert all(torch.equal(live[k].cpu(), expected[k]) for k in expected)

    def test_the_frozen_policys_action_is_what_act_returns(self, tmp_path):
        pytest.importorskip("torch")

        warm = _write_constant_action_checkpoint(tmp_path / "warm.pt", Macro.STRAFE_R)
        _, opponent_for = _build_opponents(_selfplay_cfg(warm), tmp_path / "pool")
        driver = opponent_for(0)
        driver.begin_episode()

        actions = [driver.act(_obs(i)) for i in range(6)]

        assert actions == [int(Macro.STRAFE_R)] * 6
        assert all(0 <= a < N_ACTIONS for a in actions)

    @pytest.mark.parametrize(
        "info,expected_score",
        [
            ({"won": True, "lost": False, "timeout": False}, 1.0),
            ({"won": False, "lost": True, "timeout": False}, 0.0),
            ({"won": False, "lost": False, "timeout": True}, 0.5),
            # An episode stopped by the rollout's max_steps: nobody died, so it
            # is a draw. Reading it as a loss would depress the learner's Elo
            # for a cap the opponent had nothing to do with.
            ({"won": False, "lost": False, "timeout": False}, 0.5),
            ({}, 0.5),
            # A malformed pair (MCPvPEnv never emits one) resolves to a LOSS,
            # keeping the env's own rule that a double death cannot be a win.
            ({"won": True, "lost": True, "timeout": False}, 0.0),
        ],
    )
    def test_the_outcome_is_scored_from_the_learners_perspective(
        self, tmp_path, info, expected_score
    ):
        pytest.importorskip("torch")

        warm = _write_default_checkpoint(tmp_path / "warm.pt")
        pool, opponent_for = _build_opponents(_selfplay_cfg(warm), tmp_path / "pool")
        driver = opponent_for(0)
        driver.begin_episode()
        driver.observe_outcome(info)

        stats = pool.stats_for(0)
        assert stats.plays == 1
        assert stats.learner_wins == pytest.approx(expected_score)
        assert driver.current_match is not None
        assert driver.current_match.score == pytest.approx(expected_score)
        assert driver.current_match.snapshot_id == 0

    def test_an_outcome_without_a_begun_episode_scores_nothing(self, tmp_path):
        pytest.importorskip("torch")

        warm = _write_default_checkpoint(tmp_path / "warm.pt")
        pool, opponent_for = _build_opponents(_selfplay_cfg(warm), tmp_path / "pool")
        driver = opponent_for(0)

        driver.observe_outcome({"won": True})

        assert pool.stats_for(0).plays == 0
        assert driver.current_match is None

    def test_the_match_record_carries_both_epsilons(self, tmp_path):
        pytest.importorskip("torch")

        warm = _write_default_checkpoint(tmp_path / "warm.pt")
        cfg = _selfplay_cfg(warm, opponent_epsilon=0.02)
        _, opponent_for = _build_opponents(cfg, tmp_path / "pool")
        driver = opponent_for(0)

        driver.note_learner_epsilon(0.1)
        driver.begin_episode()
        driver.observe_outcome({"won": True})

        match = driver.current_match
        assert match.learner_epsilon == pytest.approx(0.1)
        assert match.opponent_epsilon == pytest.approx(0.02)
        assert match.rated_eligible is False

    def test_a_doubly_greedy_match_is_rated_eligible(self, tmp_path):
        pytest.importorskip("torch")

        warm = _write_default_checkpoint(tmp_path / "warm.pt")
        # Both epsilons EXACTLY 0.0 — the eligibility test is float equality, so
        # an epsilon-adjacent constant would empty the rated Elo series in
        # silence.
        cfg = _selfplay_cfg(warm, opponent_epsilon=0.0)
        _, opponent_for = _build_opponents(cfg, tmp_path / "pool")
        driver = opponent_for(0)

        driver.note_learner_epsilon(0.0)
        driver.begin_episode()
        driver.observe_outcome({"won": True})

        assert driver.current_match.rated_eligible is True

    def test_the_default_learner_epsilon_is_not_greedy(self, tmp_path):
        pytest.importorskip("torch")

        warm = _write_default_checkpoint(tmp_path / "warm.pt")
        cfg = _selfplay_cfg(warm, opponent_epsilon=0.0)
        _, opponent_for = _build_opponents(cfg, tmp_path / "pool")
        driver = opponent_for(0)

        # Never noted. The seeded default is the learner's epsilon FLOOR, so an
        # unreported match cannot pass itself off as a greedy one and slip into
        # the rated Elo series.
        assert driver.learner_epsilon == pytest.approx(cfg.eps_end)
        driver.begin_episode()
        driver.observe_outcome({"won": True})
        assert driver.current_match.rated_eligible is False

    @pytest.mark.parametrize("bad", [-0.1, 1.5, float("nan")])
    def test_an_out_of_range_learner_epsilon_is_refused(self, tmp_path, bad):
        pytest.importorskip("torch")

        warm = _write_default_checkpoint(tmp_path / "warm.pt")
        _, opponent_for = _build_opponents(_selfplay_cfg(warm), tmp_path / "pool")
        driver = opponent_for(0)

        with pytest.raises(ValueError, match="epsilon"):
            driver.note_learner_epsilon(bad)


# ===========================================================================
# TC22 — collect_episode routes the mirrored observation.
# ===========================================================================


class TestCollectEpisodeRoutesTheObservation:
    """TC22: a ``needs_observation`` driver is fed the 23-dim vector, not a view."""

    def _collect(self, opponent, env, epsilon: float = 0.3):
        import torch

        from agent.train import collect_episode

        return collect_episode(
            env,
            _FixedPolicy(torch, action=int(Macro.ATTACK)),
            max_steps=16,
            episode_index=0,
            epsilon=epsilon,
            episode_seed=7,
            opponent=opponent,
        )

    def test_the_driver_receives_the_mirrored_observation(self):
        pytest.importorskip("torch")

        env = _MirrorPadEnv(k=4)
        opponent = _RecordingObservationOpponent(Macro.JUMP)

        episode = self._collect(opponent, env)

        assert len(episode.transitions) == 4
        assert env.observation_calls == 4
        assert env.view_calls == 0, (
            "a needs_observation driver was routed to raw_opponent_view(); the "
            "frozen net would be handed an OpponentView it cannot read"
        )
        assert len(opponent.observations) == 4
        for recorded, served in zip(opponent.observations, env.mirrored):
            assert recorded.shape == (OBS_DIM,)
            assert recorded.dtype == np.float32
            np.testing.assert_array_equal(recorded, served)

    def test_the_mirrored_vector_is_not_the_learners(self):
        pytest.importorskip("torch")

        env = _MirrorPadEnv(k=3)
        opponent = _RecordingObservationOpponent()

        episode = self._collect(opponent, env)

        learner_obs = [t[0] for t in episode.transitions]
        for mirrored, learner in zip(opponent.observations, learner_obs):
            # Same shape, different seat. Handing the snapshot the LEARNER's
            # observation would be a well-formed vector describing the wrong
            # fighter — exactly the class of silent bug the mirror exists for.
            assert mirrored.shape == learner.shape
            assert not np.array_equal(mirrored, learner)

    def test_one_observation_one_macro_one_step_per_decision(self):
        pytest.importorskip("torch")

        env = _MirrorPadEnv(k=5)
        opponent = _RecordingObservationOpponent(Macro.RETREAT)

        self._collect(opponent, env)

        # The env shadow-tracks the opponent's attack meter by COUNTING decision
        # windows, so a skipped or doubled step desynchronizes it silently.
        assert env.observation_calls == len(env.step_calls) == 5
        assert opponent.calls == ["begin"] + ["act"] * 5 + ["outcome"]
        assert [opp for _a, opp in env.step_calls] == [int(Macro.RETREAT)] * 5

    def test_the_learners_epsilon_is_reported_before_the_episode(self):
        pytest.importorskip("torch")

        env = _MirrorPadEnv(k=2)
        opponent = _RecordingObservationOpponent()

        self._collect(opponent, env, epsilon=0.17)

        assert opponent.noted_epsilons == [0.17]
        # Reported BEFORE begin_episode, so the match record built for this
        # episode's snapshot already carries the right learner epsilon.
        assert opponent.calls[0] == "begin"

    def test_an_opponent_without_the_hook_still_works(self):
        pytest.importorskip("torch")

        class _NoHook(_RecordingObservationOpponent):
            note_learner_epsilon = None  # not callable

        env = _MirrorPadEnv(k=2)
        opponent = _NoHook()

        self._collect(opponent, env)

        assert env.observation_calls == 2

    def test_the_final_info_is_handed_back_for_scoring(self):
        pytest.importorskip("torch")

        env = _MirrorPadEnv(k=3, won=True)
        opponent = _RecordingObservationOpponent()

        self._collect(opponent, env)

        assert len(opponent.outcomes) == 1
        assert opponent.outcomes[0]["won"] is True


# ===========================================================================
# TC25 / TC26 — the scripted and dummy paths are unchanged (AC10).
# ===========================================================================


class TestTheScriptedAndDummyPathsAreUnchanged:
    """AC10: adding the self-play branch must not move the other two."""

    def _collect(self, opponent, env):
        import torch

        from agent.train import collect_episode

        return collect_episode(
            env,
            _FixedPolicy(torch, action=int(Macro.ATTACK)),
            max_steps=16,
            episode_index=0,
            epsilon=0.2,
            episode_seed=11,
            opponent=opponent,
        )

    def test_a_view_opponent_still_receives_an_opponent_view(self):
        pytest.importorskip("torch")

        env = _MirrorPadEnv(k=4)
        opponent = _RecordingViewOpponent(Macro.APPROACH)

        self._collect(opponent, env)

        assert env.view_calls == 4
        assert env.observation_calls == 0, (
            "the scripted path was routed to opponent_observation(); a "
            "ScriptedBot cannot read an observation vector"
        )
        assert len(opponent.views) == 4
        assert all(isinstance(v, OpponentView) for v in opponent.views)
        assert [opp for _a, opp in env.step_calls] == [int(Macro.APPROACH)] * 4

    def test_a_real_scripted_driver_still_receives_an_opponent_view(self):
        pytest.importorskip("torch")

        from agent.train import OpponentCurriculum, ScriptedOpponentDriver

        cfg = TrainConfig()
        env = _MirrorPadEnv(k=3)
        driver = ScriptedOpponentDriver(cfg, OpponentCurriculum(cfg), 0)

        self._collect(driver, env)

        assert env.view_calls == 3
        assert env.observation_calls == 0
        assert all(opp is not None for _a, opp in env.step_calls)

    def test_the_dummy_path_puts_nothing_extra_on_the_wire(self):
        pytest.importorskip("torch")

        env = _MirrorPadEnv(k=4)

        self._collect(None, env)

        assert env.view_calls == 0
        assert env.observation_calls == 0
        assert [opp for _a, opp in env.step_calls] == [None] * 4


# ===========================================================================
# TC24 — mirror_opponent is enabled at BOTH env construction sites (AC13).
# ===========================================================================


class _RecordingEnv:
    """Stands in for ``MCPvPEnv`` and records the kwargs it was constructed with."""

    constructed: List[Dict[str, Any]] = []

    def __init__(self, **kwargs: Any) -> None:
        type(self).constructed.append(dict(kwargs))
        self._transport = kwargs.get("transport")

    def close(self) -> None:
        pass


@pytest.fixture
def recording_env(monkeypatch):
    """Replace ``env.mc_pvp_env.MCPvPEnv`` with a constructor recorder."""
    import env.mc_pvp_env as env_module

    class _Env(_RecordingEnv):
        constructed: List[Dict[str, Any]] = []

    monkeypatch.setattr(env_module, "MCPvPEnv", _Env)
    return _Env


class TestTheMirrorReachesBothEnvConstructionSites:
    """TC24 / AC13. Two sites, two very different failure timings.

    The TRAINING factory's omission raises on the run's first episode —
    ``opponent_observation()`` refuses rather than inventing a zeroed world. The
    EVAL env's omission is invisible until the first eval cycle, which on the
    real cadence is roughly an hour in, and it takes the checkpoint-selection
    signal down with it. Both are pinned, separately.
    """

    def test_the_training_factory_mirrors_under_selfplay(
        self, tmp_path, monkeypatch, recording_env
    ):
        pytest.importorskip("torch")
        import env.mc_pvp_env as env_module

        from agent.train import build_live_env_factory_for

        monkeypatch.setattr(
            env_module, "TcpBridgeClient", lambda host, port: ("tcp", host, port)
        )
        warm = _write_default_checkpoint(tmp_path / "warm.pt")
        factory = build_live_env_factory_for(
            _selfplay_cfg(warm), host="127.0.0.1", base_port=5555
        )

        factory(0)()
        factory(3)()

        assert len(recording_env.constructed) == 2
        assert all(k["mirror_opponent"] is True for k in recording_env.constructed)
        # And still one client per pad, on that pad's own port.
        assert [k["transport"][2] for k in recording_env.constructed] == [5555, 5558]

    @pytest.mark.parametrize("opponent", ["dummy", "scripted"])
    def test_the_training_factory_does_not_mirror_otherwise(
        self, monkeypatch, recording_env, opponent
    ):
        pytest.importorskip("torch")
        import env.mc_pvp_env as env_module

        from agent.train import build_live_env_factory_for

        monkeypatch.setattr(
            env_module, "TcpBridgeClient", lambda host, port: ("tcp", host, port)
        )
        cfg = dataclasses.replace(TrainConfig(), opponent=opponent, arenas=2)
        factory = build_live_env_factory_for(cfg, host="h", base_port=5555)

        factory(0)()

        assert recording_env.constructed[0]["mirror_opponent"] is False

    @pytest.mark.parametrize("mirror", [True, False])
    def test_the_eval_env_is_built_with_the_flag_it_was_given(
        self, recording_env, mirror
    ):
        pytest.importorskip("torch")

        from agent.train import Trainer, _eval_against_opponent

        report = MagicMock()
        report.win_rate = 0.5
        trainer = Trainer(
            dataclasses.replace(TrainConfig(), arenas=2, batch_size=4, seq_len=2,
                                burn_in=1, n_step=1, min_replay=1),
            net_kwargs=dict(_TINY_NET),
        )

        _eval_against_opponent(
            trainer=trainer,
            evaluate=lambda *_a, **_k: report,
            policy_cls=lambda _net, device=None: MagicMock(),
            shared_transport=object(),
            n_episodes=1,
            timeout_cap=64,
            env_max_episode_steps=64,
            eval_step_cap=4,
            logger=None,
            is_live=False,
            base_seed=0,
            log=None,
            mirror_opponent=mirror,
        )

        assert len(recording_env.constructed) == 1
        assert recording_env.constructed[0]["mirror_opponent"] is mirror
        # Still the borrowed connection, never a second one.
        assert recording_env.constructed[0]["auto_connect"] is False

    def test_the_multi_arena_loop_hands_the_eval_the_flag(self, tmp_path, monkeypatch):
        pytest.importorskip("torch")

        import agent.train as train_module

        seen: List[Dict[str, Any]] = []

        def _stub(**kwargs):
            seen.append(dict(kwargs))
            return None  # eval skipped; the loop carries on

        monkeypatch.setattr(train_module, "_eval_via_designated_arena", _stub)

        warm = _write_constant_action_checkpoint(tmp_path / "warm.pt", Macro.JUMP)
        cfg = _selfplay_cfg(warm)
        pads = [_MirrorPadEnv() for _ in range(cfg.arenas)]
        _run_multi_arena(
            cfg,
            pads,
            snapshot_dir=str(tmp_path / "pool"),
            eval_every_grad_steps=1,
            max_grad_steps=4,
        )

        assert seen, "no eval cycle ran; this test cannot observe the flag"
        assert all(call["mirror_opponent"] is True for call in seen)

    def test_a_scripted_run_hands_the_eval_a_false_flag(self, tmp_path, monkeypatch):
        pytest.importorskip("torch")

        import agent.train as train_module

        seen: List[Dict[str, Any]] = []
        monkeypatch.setattr(
            train_module,
            "_eval_via_designated_arena",
            lambda **kwargs: seen.append(dict(kwargs)),
        )

        cfg = dataclasses.replace(
            _selfplay_cfg(_write_default_checkpoint(tmp_path / "warm.pt")),
            opponent="scripted",
        )
        pads = [_MirrorPadEnv() for _ in range(cfg.arenas)]
        _run_multi_arena(cfg, pads, eval_every_grad_steps=1, max_grad_steps=4)

        assert seen
        assert all(call["mirror_opponent"] is False for call in seen)


# ===========================================================================
# TC23 — AC13: `--opponent selfplay` reaches the collectors.
# ===========================================================================


class TestSelfPlayReachesTheWire:
    """AC13/TC23: a snapshot's action must actually arrive as ``opp_action``.

    This is the wiring test. Every piece below it can be perfect and the run
    still trains against a stationary dummy, because the branch that builds the
    opponent factory lives in ``train_multi_arena`` and a missing branch leaves
    ``opponent_for`` at ``None`` — no exception, no warning, and every log line
    and metric label still reading ``selfplay``.

    The frozen snapshot is engineered to play ONE macro for every observation
    (see :func:`_write_constant_action_checkpoint`), so the assertion is an
    equality on the wire rather than a statistical argument: a dummy run sends
    ``None``, and no other component in the stack emits that macro on every step
    of every episode.
    """

    def test_a_snapshot_action_arrives_as_opp_action(self, tmp_path):
        pytest.importorskip("torch")

        warm = _write_constant_action_checkpoint(tmp_path / "warm.pt", Macro.JUMP)
        cfg = _selfplay_cfg(warm)
        pads = [_MirrorPadEnv() for _ in range(cfg.arenas)]

        _run_multi_arena(cfg, pads, snapshot_dir=str(tmp_path / "pool"))

        on_the_wire = [opp for pad in pads for opp in pad.opp_actions]
        assert on_the_wire, (
            "no step reached a pad at all; this run collected nothing and the "
            "test cannot observe the wire"
        )
        assert all(opp is not None for opp in on_the_wire), (
            "--opponent selfplay produced step lines with NO opp_action: the "
            "opponent factory never reached the collectors and this run trained "
            "against the stationary dummy while every log line said selfplay"
        )
        assert set(on_the_wire) == {int(Macro.JUMP)}, (
            f"opp_action did not come from the frozen snapshot: "
            f"{sorted(set(on_the_wire))}"
        )

    def test_the_collectors_read_the_mirrored_observation(self, tmp_path):
        pytest.importorskip("torch")

        warm = _write_constant_action_checkpoint(tmp_path / "warm.pt", Macro.JUMP)
        cfg = _selfplay_cfg(warm)
        pads = [_MirrorPadEnv() for _ in range(cfg.arenas)]

        _run_multi_arena(cfg, pads, snapshot_dir=str(tmp_path / "pool"))

        assert sum(pad.observation_calls for pad in pads) > 0
        assert sum(pad.view_calls for pad in pads) == 0, (
            "a self-play collector called raw_opponent_view(); the routing "
            "branch in collect_episode is inverted"
        )
        # One mirrored read per decision window, on every pad that collected.
        for pad in pads:
            assert pad.observation_calls == len(pad.step_calls)

    def test_every_arena_collected_against_a_snapshot(self, tmp_path):
        pytest.importorskip("torch")

        warm = _write_constant_action_checkpoint(tmp_path / "warm.pt", Macro.STRAFE_L)
        cfg = _selfplay_cfg(warm, arenas=3)
        pads = [_MirrorPadEnv() for _ in range(cfg.arenas)]

        _run_multi_arena(cfg, pads, snapshot_dir=str(tmp_path / "pool"))

        # Not "some arena worked": a factory that is not memoized per arena, or
        # one wired only to arena 0, still passes a fleet-wide count.
        for arena_id, pad in enumerate(pads):
            assert pad.opp_actions, f"arena {arena_id} never stepped"
            assert set(pad.opp_actions) == {int(Macro.STRAFE_L)}

    def test_the_pool_is_created_on_disk(self, tmp_path):
        pytest.importorskip("torch")

        warm = _write_constant_action_checkpoint(tmp_path / "warm.pt", Macro.JUMP)
        cfg = _selfplay_cfg(warm)
        pool_dir = tmp_path / "pool"
        pads = [_MirrorPadEnv() for _ in range(cfg.arenas)]

        _run_multi_arena(cfg, pads, snapshot_dir=str(pool_dir))

        assert (pool_dir / "pool.json").is_file()
        assert (pool_dir / "snap_0.pt").is_file()

    def test_a_dummy_run_over_the_same_pads_sends_no_opp_action(self, tmp_path):
        pytest.importorskip("torch")

        # TC26: the control. Same pads, same loop, opponent="dummy" — every step
        # line must carry opp_action=None and neither accessor may be touched.
        cfg = dataclasses.replace(
            _selfplay_cfg(_write_default_checkpoint(tmp_path / "warm.pt")),
            opponent="dummy",
        )
        pads = [_MirrorPadEnv() for _ in range(cfg.arenas)]

        _run_multi_arena(cfg, pads)

        on_the_wire = [opp for pad in pads for opp in pad.opp_actions]
        assert on_the_wire, "no step reached a pad; the control proves nothing"
        assert set(on_the_wire) == {None}
        assert sum(pad.observation_calls for pad in pads) == 0
        assert sum(pad.view_calls for pad in pads) == 0

    def test_a_scripted_run_over_the_same_pads_reads_the_view(self, tmp_path):
        pytest.importorskip("torch")

        # TC25 at the wiring level: the scripted branch must still resolve to
        # the view accessor after the self-play branch was added beside it.
        cfg = dataclasses.replace(
            _selfplay_cfg(_write_default_checkpoint(tmp_path / "warm.pt")),
            opponent="scripted",
        )
        pads = [_MirrorPadEnv() for _ in range(cfg.arenas)]

        _run_multi_arena(cfg, pads)

        assert sum(pad.view_calls for pad in pads) > 0
        assert sum(pad.observation_calls for pad in pads) == 0
        on_the_wire = [opp for pad in pads for opp in pad.opp_actions]
        assert on_the_wire and all(opp is not None for opp in on_the_wire)


# ===========================================================================
# The seed scheme gained two roles; the four it already had must not move.
# ===========================================================================


class TestTheSnapshotSeedRoles:
    """New roles are APPENDED — a role's seed is its index in the tuple."""

    def test_the_existing_roles_keep_their_offsets(self):
        from agent.train import _OPPONENT_SEED_ROLES

        assert _OPPONENT_SEED_ROLES[:4] == ("mixture", "easy", "hard", "eval")

    def test_the_snapshot_roles_exist_and_are_distinct(self):
        from agent.train import opponent_seed

        cfg = dataclasses.replace(TrainConfig(), arenas=4)
        seeds = [
            opponent_seed(cfg, arena, role)
            for arena in range(4)
            for role in _OPPONENT_ROLES
        ]
        assert len(set(seeds)) == len(seeds)


# ===========================================================================
# T11b — the CLI: choices, casts, the checksum gate, the run's own record.
# ===========================================================================

#: A ``--warm-start`` value for tests that never open the file. ``TrainConfig``
#: validates only that the path is a non-empty string, and nothing on the
#: config-building path reads the bytes — the checksum gate below is the one
#: place that does, and it is a no-op without ``--warm-start-sha256``.
_UNREAD_WARM_START = "runs/never-opened-by-this-test.pt"


def _parse(argv: List[str]):
    """Parse ``argv`` with the REAL training parser."""
    from agent.train import _build_parser

    return _build_parser().parse_args(argv)


def _cli_cfg(argv: List[str]) -> TrainConfig:
    """Build the run's config exactly as ``main()`` does, from the real parser."""
    from agent.train import _config_from_args

    return _config_from_args(_parse(argv))


def _selfplay_argv(warm_start: str, *extra: str) -> List[str]:
    """The smallest argv that parses into a valid self-play config."""
    return [
        "--arenas", "2",
        "--opponent", "selfplay",
        "--warm-start", str(warm_start),
        *extra,
    ]


def _sha256_of(path) -> str:
    """The digest ``--warm-start-sha256`` is compared against."""
    digest = hashlib.sha256()
    with open(str(path), "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TestTheOpponentFlagAcceptsSelfPlay:
    """``--opponent selfplay`` has to be spellable before anything else matters."""

    def test_every_config_choice_is_reachable_from_the_cli(self):
        pytest.importorskip("torch")
        from agent.train_config import _OPPONENT_CHOICES

        # A choice TrainConfig accepts but argparse rejects is a mode nobody can
        # run; the reverse (argparse accepts, TrainConfig rejects) fails in
        # `_config_from_args` below. Both directions are covered here.
        for choice in sorted(_OPPONENT_CHOICES):
            assert _parse(["--opponent", choice]).opponent == choice

    def test_an_unknown_opponent_is_still_refused(self):
        pytest.importorskip("torch")

        with pytest.raises(SystemExit):
            _parse(["--opponent", "shadowboxing"])

    def test_the_default_is_still_the_dummy(self):
        pytest.importorskip("torch")

        assert _parse([]).opponent == "dummy"

    def test_the_choice_reaches_the_config(self):
        pytest.importorskip("torch")

        assert _cli_cfg(_selfplay_argv(_UNREAD_WARM_START)).opponent == "selfplay"


class TestTheSelfPlayFlagsReachTheConfig:
    """Every knob defaults to ``None`` so ``TrainConfig`` owns the defaults."""

    def test_omitted_flags_keep_the_dataclass_defaults(self):
        pytest.importorskip("torch")

        cfg = _cli_cfg(_selfplay_argv(_UNREAD_WARM_START))
        default = TrainConfig()

        assert cfg.snapshot_every_grad_steps == default.snapshot_every_grad_steps
        assert cfg.snapshot_sampling == default.snapshot_sampling
        assert cfg.opponent_epsilon == default.opponent_epsilon
        assert cfg.reference_promote_grad_steps == (
            default.reference_promote_grad_steps
        )
        assert cfg.elo_k == default.elo_k
        assert cfg.elo_initial == default.elo_initial
        assert cfg.warm_start_sha256 is None

    def test_every_flag_overrides_its_field(self):
        pytest.importorskip("torch")

        digest = "a" * 64
        cfg = _cli_cfg(
            _selfplay_argv(
                _UNREAD_WARM_START,
                "--snapshot-every-grad-steps", "250",
                "--snapshot-sampling", "uniform",
                "--opponent-epsilon", "0.05",
                "--reference-promote-grad-steps", "300", "900",
                "--elo-k", "16",
                "--elo-initial", "1200",
                "--warm-start-sha256", digest,
            )
        )

        assert cfg.snapshot_every_grad_steps == 250
        assert cfg.snapshot_sampling == "uniform"
        assert cfg.opponent_epsilon == pytest.approx(0.05)
        assert cfg.reference_promote_grad_steps == (300, 900)
        assert cfg.elo_k == pytest.approx(16.0)
        assert cfg.elo_initial == pytest.approx(1200.0)
        assert cfg.warm_start_sha256 == digest

    def test_the_promote_steps_arrive_as_a_tuple_not_a_list(self):
        """THE trap on this path, and the reason the cast is ``tuple``.

        ``argparse`` ``nargs=2`` hands over a LIST. ``TrainConfig`` rejects a
        non-tuple outright — because a list field on a frozen dataclass makes
        ``hash(cfg)`` raise at whatever unrelated call site hashes the config
        first, arbitrarily far from the CLI that produced it.
        """
        pytest.importorskip("torch")
        from agent.train import _config_from_args

        args = _parse(
            _selfplay_argv(
                _UNREAD_WARM_START, "--reference-promote-grad-steps", "300", "900"
            )
        )
        assert isinstance(args.reference_promote_grad_steps, list), (
            "argparse no longer yields a list for nargs=2; this test exists to "
            "pin the list -> tuple cast that fact requires"
        )

        cfg = _config_from_args(args)

        assert isinstance(cfg.reference_promote_grad_steps, tuple)
        assert cfg.reference_promote_grad_steps == (300, 900)
        assert all(
            isinstance(step, int) for step in cfg.reference_promote_grad_steps
        ), "T18 promotes on `grad_step == step`, an equality against an int counter"
        # The consequence the tuple is FOR: the frozen config stays hashable.
        assert isinstance(hash(cfg), int)

    def test_an_out_of_order_pair_is_still_refused(self):
        pytest.importorskip("torch")

        with pytest.raises(ValueError, match="strictly increasing"):
            _cli_cfg(
                _selfplay_argv(
                    _UNREAD_WARM_START,
                    "--reference-promote-grad-steps", "900", "300",
                )
            )

    def test_an_unknown_sampling_mode_is_refused_by_argparse(self):
        pytest.importorskip("torch")

        with pytest.raises(SystemExit):
            _parse(_selfplay_argv(_UNREAD_WARM_START, "--snapshot-sampling", "elo"))

    def test_a_selfplay_run_does_not_stop_on_the_m2_gate(self):
        """The M2 gate scores the STATIONARY dummy; it is not self-play's finish.

        A warm-started agent can clear it in its first eval, which would end a
        24-hour self-play run minutes in on a verdict about a different opponent.
        """
        pytest.importorskip("torch")
        from agent.train import _resolve_stop_on_pass

        cfg = _cli_cfg(_selfplay_argv(_UNREAD_WARM_START))

        assert _resolve_stop_on_pass(None, cfg) is False
        # An explicit flag still wins, on this path as on every other.
        assert _resolve_stop_on_pass(True, cfg) is True


class TestTheWarmStartChecksumGate:
    """AC14's second half: the bytes on disk, not just the shape of the digest.

    ``TrainConfig`` validates that ``warm_start_sha256`` LOOKS like a SHA-256 and
    never opens the file. A self-play run seeds snapshot 0 — the pool's first
    PINNED member, never dropped and never evicted — entirely from that file, and
    a stale path or the wrong run's checkpoint loads perfectly cleanly into the
    same architecture. The night then runs against a permanent reference opponent
    nobody can identify afterwards.
    """

    def test_a_matching_checksum_is_accepted(self, tmp_path):
        pytest.importorskip("torch")
        from agent.train import _verify_warm_start_checksum

        warm = _write_default_checkpoint(tmp_path / "warm.pt")
        cfg = _cli_cfg(
            _selfplay_argv(warm, "--warm-start-sha256", _sha256_of(warm))
        )

        assert _verify_warm_start_checksum(cfg) is None

    def test_a_mismatch_names_both_digests(self, tmp_path):
        pytest.importorskip("torch")
        from agent.train import _verify_warm_start_checksum

        warm = _write_default_checkpoint(tmp_path / "warm.pt")
        actual = _sha256_of(warm)
        expected = "b" * 64
        cfg = _cli_cfg(_selfplay_argv(warm, "--warm-start-sha256", expected))

        with pytest.raises(ValueError) as excinfo:
            _verify_warm_start_checksum(cfg)

        message = str(excinfo.value)
        # BOTH, because "checksum mismatch" alone cannot tell an operator whether
        # they pasted the wrong checksum or aimed at the wrong checkpoint.
        assert actual in message
        assert expected in message
        assert str(warm) in message

    def test_a_changed_file_stops_matching(self, tmp_path):
        """The gate reads the FILE, not a cached or config-level value."""
        pytest.importorskip("torch")
        from agent.train import _verify_warm_start_checksum

        warm = _write_default_checkpoint(tmp_path / "warm.pt")
        cfg = _cli_cfg(
            _selfplay_argv(warm, "--warm-start-sha256", _sha256_of(warm))
        )
        _verify_warm_start_checksum(cfg)

        # Same path, different bytes: exactly the swapped-checkpoint case.
        _write_constant_action_checkpoint(tmp_path / "warm.pt", Macro.JUMP)

        with pytest.raises(ValueError, match="checksum MISMATCH"):
            _verify_warm_start_checksum(cfg)

    def test_no_checksum_verifies_nothing(self):
        """Ordinary warm-started runs are untouched — the file is never opened."""
        pytest.importorskip("torch")
        from agent.train import _verify_warm_start_checksum

        cfg = _cli_cfg(_selfplay_argv(_UNREAD_WARM_START))

        assert cfg.warm_start_sha256 is None
        assert _verify_warm_start_checksum(cfg) is None

    def test_a_missing_checkpoint_is_named(self, tmp_path):
        pytest.importorskip("torch")
        from agent.train import _verify_warm_start_checksum

        missing = str(tmp_path / "gone.pt")
        cfg = _cli_cfg(_selfplay_argv(missing, "--warm-start-sha256", "c" * 64))

        with pytest.raises(FileNotFoundError, match="gone.pt"):
            _verify_warm_start_checksum(cfg)

    def test_main_refuses_a_mismatch_before_it_builds_anything(
        self, tmp_path, monkeypatch
    ):
        """The gate runs before the logger and before the fleet is touched."""
        pytest.importorskip("torch")

        import eval.logging as logging_module

        import agent.train as train_module

        monkeypatch.setattr(
            logging_module,
            "MetricsLogger",
            lambda **_k: pytest.fail(
                "a run directory was created before the warm start was verified"
            ),
        )
        monkeypatch.setattr(
            train_module,
            "_main_multi_arena",
            lambda *_a, **_k: pytest.fail(
                "the run started against an unverified warm start"
            ),
        )
        warm = _write_default_checkpoint(tmp_path / "warm.pt")

        with pytest.raises(ValueError, match="checksum MISMATCH"):
            train_module.main(
                _selfplay_argv(warm, "--warm-start-sha256", "d" * 64)
            )


class TestTheSingleArenaRefusalCoversSelfPlay:
    """TC27. The existing ``cfg.opponent != "dummy"`` check already covers it.

    Pinned rather than re-implemented: the single-arena loop steps no opponent
    policy at all, so a self-play run there would fight the bridge's stationary
    dummy while every log line and the logger's config record said ``selfplay``.
    """

    def test_one_arena_selfplay_is_refused(self, capsys):
        pytest.importorskip("torch")
        from agent.train import main

        code = main(
            ["--arenas", "1", "--opponent", "selfplay",
             "--warm-start", _UNREAD_WARM_START]
        )

        assert code == 1
        stderr = capsys.readouterr().err
        assert "needs --arenas >1" in stderr
        assert "selfplay" in stderr

    def test_two_arenas_selfplay_passes_the_refusal(self, tmp_path, monkeypatch):
        """The complement: the gate refuses ONE arena, not the mode itself."""
        pytest.importorskip("torch")

        import eval.logging as logging_module

        import agent.train as train_module

        monkeypatch.setattr(logging_module, "MetricsLogger", _NullLogger)
        monkeypatch.setattr(train_module, "_main_multi_arena", lambda *_a, **_k: 0)

        warm = _write_default_checkpoint(tmp_path / "warm.pt")

        assert train_module.main(_selfplay_argv(warm)) == 0


class _NullLogger:
    """A ``MetricsLogger`` stand-in that records the config it was handed."""

    #: Last config dict any instance was constructed with.
    last_config: Dict[str, Any] = {}

    def __init__(self, **kwargs: Any) -> None:
        type(self).last_config = dict(kwargs.get("config") or {})

    def log(self, *_a: Any, **_k: Any) -> None:
        pass

    def summary(self, *_a: Any, **_k: Any) -> None:
        pass

    def close(self) -> None:
        pass


class TestTheRunRecordsWhatItFought:
    """``summary.json`` has to name the opponent AND the knobs that shaped it.

    ``opponent: selfplay`` alone proves only that the flag parsed. Which past
    self was drawn, how often a new one was archived and what its rating meant
    are the difference between a readable Elo curve and a number nobody can
    reconstruct — the same provenance gap the reward coefficients are recorded
    for.
    """

    def _recorded_config(self, monkeypatch, argv: List[str]) -> Dict[str, Any]:
        import eval.logging as logging_module

        import agent.train as train_module

        _NullLogger.last_config = {}
        monkeypatch.setattr(logging_module, "MetricsLogger", _NullLogger)
        monkeypatch.setattr(train_module, "_main_multi_arena", lambda *_a, **_k: 0)
        assert train_module.main(argv) == 0
        return _NullLogger.last_config

    def test_the_selfplay_knobs_are_recorded(self, tmp_path, monkeypatch):
        pytest.importorskip("torch")

        warm = _write_default_checkpoint(tmp_path / "warm.pt")
        digest = _sha256_of(warm)
        config = self._recorded_config(
            monkeypatch,
            _selfplay_argv(
                warm,
                "--run-name", "m4_selfplay",
                "--warm-start-sha256", digest,
                "--snapshot-every-grad-steps", "250",
                "--snapshot-sampling", "uniform",
                "--opponent-epsilon", "0.05",
                "--reference-promote-grad-steps", "300", "900",
                "--elo-k", "16",
                "--elo-initial", "1200",
            ),
        )

        assert config["opponent"] == "selfplay"
        assert config["warm_start"] == str(warm)
        assert config["warm_start_sha256"] == digest
        assert config["snapshot_every_grad_steps"] == 250
        assert config["snapshot_sampling"] == "uniform"
        assert config["opponent_epsilon"] == pytest.approx(0.05)
        assert config["reference_promote_grad_steps"] == [300, 900]
        assert config["elo_k"] == pytest.approx(16.0)
        assert config["elo_initial"] == pytest.approx(1200.0)
        assert config["snapshot_pool_dir"] == os.path.join(
            "runs", "m4_selfplay", "snapshots"
        )
        # summary.json is written with json.dump, so anything unserializable
        # here would take the whole run record down at close() time.
        json.dumps(config)

    def test_a_dummy_run_records_no_pool(self, monkeypatch):
        pytest.importorskip("torch")

        config = self._recorded_config(monkeypatch, ["--arenas", "2"])

        assert config["opponent"] == "dummy"
        assert config["snapshot_pool_dir"] is None


# ===========================================================================
# W2 — the snapshot pool directory follows --run-name (T11b).
# ===========================================================================


class TestThePoolDirectoryFollowsTheRunName:
    """A pool nobody passes is a pool two runs SHARE.

    ``train_multi_arena``'s ``snapshot_dir`` default resolves to
    ``runs/selfplay/snapshots`` for every run alike, and
    ``build_snapshot_opponents`` RELOADS a directory that already holds a
    ``pool.json`` rather than reseeding it. So a second self-play run under a
    different ``--run-name`` would silently inherit the first run's past selves,
    its head-to-head statistics and its Elo series — with no error, and a
    ``pool_size`` line at startup that looks perfectly healthy.
    """

    def _snapshot_dir_the_cli_passes(
        self, monkeypatch, argv: List[str]
    ) -> Optional[str]:
        """Run ``_main_multi_arena`` with the loop stubbed; return its kwarg."""
        import distributed.launcher as launcher_module

        import agent.train as train_module

        monkeypatch.setattr(
            launcher_module, "SubprocessArenaLauncher", lambda **_k: MagicMock()
        )
        captured: Dict[str, Any] = {}

        def _stub_loop(_cfg, **kwargs):
            captured.update(kwargs)
            return train_module.MultiArenaResult(
                trainer=MagicMock(),
                passed_m2=False,
                grad_steps=0,
                episodes_received=0,
            )

        monkeypatch.setattr(train_module, "train_multi_arena", _stub_loop)

        args = _parse(argv)
        train_module._main_multi_arena(
            args,
            _cli_cfg(argv),
            logger=None,
            checkpoint_hook=None,
        )
        assert "snapshot_dir" in captured, (
            "_main_multi_arena passed no snapshot_dir at all, so the loop falls "
            "back to runs/selfplay/snapshots for every run"
        )
        return captured["snapshot_dir"]

    def test_the_pool_lives_under_the_run_name(self, monkeypatch):
        pytest.importorskip("torch")
        from agent.train import snapshot_pool_directory

        directory = self._snapshot_dir_the_cli_passes(
            monkeypatch,
            _selfplay_argv(_UNREAD_WARM_START, "--run-name", "m4_selfplay"),
        )

        assert directory == snapshot_pool_directory("m4_selfplay")
        assert directory == os.path.join("runs", "m4_selfplay", "snapshots")

    def test_two_run_names_get_two_pools(self, monkeypatch):
        """The failure this exists for: one pool of past selves, two learners."""
        pytest.importorskip("torch")
        from agent.train import snapshot_pool_directory

        first = self._snapshot_dir_the_cli_passes(
            monkeypatch, _selfplay_argv(_UNREAD_WARM_START, "--run-name", "run_a")
        )
        second = self._snapshot_dir_the_cli_passes(
            monkeypatch, _selfplay_argv(_UNREAD_WARM_START, "--run-name", "run_b")
        )

        assert first != second
        # Neither may land on the loop's own name-blind fallback.
        assert snapshot_pool_directory("selfplay") not in (first, second)

    @pytest.mark.parametrize("opponent", ["dummy", "scripted"])
    def test_a_non_selfplay_run_passes_no_directory(self, monkeypatch, opponent):
        """No pool is built off the self-play path, so none is named."""
        pytest.importorskip("torch")

        directory = self._snapshot_dir_the_cli_passes(
            monkeypatch,
            ["--arenas", "2", "--opponent", opponent, "--run-name", "m4_selfplay"],
        )

        assert directory is None


# ===========================================================================
# W1 — the MIDDLE hop of the mirror chain (T11b).
# ===========================================================================


class _IdleEnv:
    """The designated arena's parked env: all the eval handoff reads is this."""

    def __init__(self) -> None:
        self._transport = object()


class _IdleCollector:
    """A collector that parks on demand, recording the pause/resume protocol."""

    def __init__(self, env: Optional[_IdleEnv]) -> None:
        self._env = env
        self.calls: List[str] = []

    def pause(self) -> None:
        self.calls.append("pause")

    def wait_until_idle(self, timeout: float) -> bool:
        return True

    def current_env(self) -> Optional[_IdleEnv]:
        return self._env

    def resume(self) -> None:
        self.calls.append("resume")


class _OneCollectorPool:
    """The slice of ``ActorPool`` the eval handoff touches."""

    def __init__(self, collector: Optional[_IdleCollector]) -> None:
        self._collector = collector

    def collector_for(self, arena_id: int) -> Optional[_IdleCollector]:
        return self._collector if arena_id == 0 else None


class TestTheEvalHandoffForwardsTheMirror:
    """The one hop of the ``mirror_opponent`` chain nothing else pins.

    ``TestTheMirrorReachesBothEnvConstructionSites`` covers the two ENDS: the
    multi-arena loop hands ``_eval_via_designated_arena`` the flag, and
    ``_eval_against_opponent`` builds its env with whatever flag it is given.
    The hop BETWEEN them is invisible to both, because every test that names
    ``_eval_via_designated_arena`` monkeypatches it out — delete its
    ``mirror_opponent=mirror_opponent`` line and the whole suite still passes.

    What that costs: the eval env is built without the mirror, so
    ``opponent_observation()`` raises inside the eval at the FIRST eval cycle —
    roughly an hour into a 24-hour run.
    """

    def _drive(self, monkeypatch, *, mirror: bool, opponent: Any, collector):
        """Run the REAL handoff with only the INNER eval stubbed out."""
        import agent.train as train_module

        recorded: Dict[str, Any] = {}

        def _record(**kwargs: Any) -> str:
            recorded.update(kwargs)
            return "outcome"

        monkeypatch.setattr(train_module, "_eval_against_opponent", _record)

        returned = train_module._eval_via_designated_arena(
            trainer=MagicMock(),
            pool=_OneCollectorPool(collector),
            designated_arena=0,
            evaluate=lambda *_a, **_k: None,
            policy_cls=lambda *_a, **_k: None,
            n_episodes=1,
            timeout_cap=64,
            env_max_episode_steps=64,
            eval_step_cap=4,
            logger=None,
            is_live=False,
            base_seed=0,
            log=None,
            pause_timeout=1.0,
            opponent=opponent,
            mirror_opponent=mirror,
        )
        return returned, recorded

    @pytest.mark.parametrize("mirror", [True, False])
    def test_the_flag_reaches_the_inner_eval(self, monkeypatch, mirror):
        pytest.importorskip("torch")

        env = _IdleEnv()
        collector = _IdleCollector(env)
        opponent = object()

        returned, recorded = self._drive(
            monkeypatch, mirror=mirror, opponent=opponent, collector=collector
        )

        assert returned == "outcome", "the inner eval's outcome must reach the caller"
        assert "mirror_opponent" in recorded, (
            "_eval_via_designated_arena dropped mirror_opponent: the eval env is "
            "then built without the observation mirror and "
            "opponent_observation() raises at the first eval cycle"
        )
        assert recorded["mirror_opponent"] is mirror
        # The same hop carries the eval's own opponent and the BORROWED
        # connection; a drop of either is the same class of silent defect.
        assert recorded["opponent"] is opponent
        assert recorded["shared_transport"] is env._transport
        assert collector.calls == ["pause", "resume"]

    def test_a_skipped_eval_still_resumes_the_collector(self, monkeypatch):
        """No env to borrow: eval is skipped, and the arena is not left parked."""
        pytest.importorskip("torch")

        collector = _IdleCollector(None)

        returned, recorded = self._drive(
            monkeypatch, mirror=True, opponent=None, collector=collector
        )

        assert returned is None
        assert recorded == {}, "no env means the inner eval must not run at all"
        assert collector.calls == ["pause", "resume"]
