"""test_selfplay_reference_eval — the rated eval cycle that fills AC7 (T13, M4).

Everything here guards ONE morning: a 24-hour self-play run whose
``elo/learner_rated`` curve is empty, whose ``selfplay/win_rate_vs_ref_<id>``
series never appeared, and whose logs said nothing about either. Every failure
in this area is silent by construction — the run keeps training, the scripted
eval keeps reporting a win rate, and the two series a checkpoint is selected on
simply have no data behind them.

The specific silences pinned below:

* **The eval fighting the STATIONARY DUMMY.** ``build_eval_opponent`` returned
  ``None`` for ``selfplay``, so the periodic eval sent no ``opp_action`` at all
  and the win rate that selects the demo checkpoint was earned against a target
  that never moves.
* **Nothing building a RATED driver.** ``elo/learner_rated`` moves only on
  matches where BOTH epsilons are exactly 0.0, which no training match can ever
  be. Until something builds the ε=0 gauntlet the series is empty BY
  CONSTRUCTION, and an empty series plots exactly like a flat one.
* **``evaluate`` having no ``needs_observation`` branch and never calling
  ``observe_outcome``.** The first feeds a frozen ``DuelingDRQN`` an
  ``OpponentView``; the second means no ``MatchResult`` is ever built, so even a
  perfectly wired gauntlet would rate nothing.
* **``note_learner_epsilon`` being called from the eval.** It RAISES on a
  nonzero ε against a rated driver — deliberately, because silently accepting
  ``cfg.eps_end`` would un-rate the whole cycle invisibly. An eval path that
  reports the schedule's ε kills the run at the first eval cycle, tens of
  minutes in.
* **A MOVING eval target.** The greedy policy holds ``trainer.online`` by
  reference while the learner keeps stepping, so a four-track cycle would score
  reference 0 and reference 2 on different networks.
* **A checkpoint selected on the aggregate alone.** A policy that grows decisive
  against the two recent references while collapsing against the oldest still
  improves the mean — that is specialization, and it is the checkpoint an
  unfamiliar human beats on demo day.

Two coverage holes left by T12 are closed here too: the ``(0 rated matches ...
EMPTY)`` warning and the three-mode opponent banner had NO test (every fixture
left ``train_multi_arena``'s ``log`` at ``None``, so every ``_emit`` in that
function was dead in the suite), and the dedup latch was never driven with both
log cadences enabled.

House conventions: no sockets, no live server, no Minecraft — every bridge, pad
and env here is a fake, and torch is optional via ``pytest.importorskip``.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock

import numpy as np
import pytest

from agent.actions import Macro
from agent.train_config import TrainConfig
from env.observation_spec import OBS_DIM
from opponents.scripted_bot import OpponentView


#: Seconds a worker thread gets before a test calls it a stall.
_THREAD_TIMEOUT = 30.0

#: Shrunken net so building several per test is cheap. ``obs_dim`` /
#: ``n_actions`` are NOT overridable — ``DuelingDRQN`` asserts them against the
#: frozen contracts.
_TINY_NET = {"encoder_hidden": 16, "lstm_hidden": 16, "lstm_layers": 1}


# ===========================================================================
# Fixtures: nets, checkpoints, pools, fake envs.
# ===========================================================================


def _completes_within(fn, timeout: float = _THREAD_TIMEOUT):
    """Run ``fn`` on a daemon thread and FAIL (never hang) if it does not finish."""
    import threading

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


def _net_factory():
    """Zero-arg builder for the tiny net every driver in this file owns."""
    from agent.dqn import DuelingDRQN

    return DuelingDRQN(**_TINY_NET)


def _write_checkpoint(path, *, bias: float = 0.0):
    """Save a tiny-net checkpoint, optionally shifted so two differ detectably."""
    import torch

    net = _net_factory()
    if bias:
        with torch.no_grad():
            for param in net.parameters():
                param.add_(float(bias))
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
        attack_cooldown=1.0,
        can_see_target=True,
        last_known_target_pos=(0.0, 64.0, 2.0),
    )
    base.update(overrides)
    return OpponentView(**base)


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
        opponent_epsilon=0.0,
    )
    base.update(overrides)
    return dataclasses.replace(TrainConfig(), **base)


def _pool(tmp_path, *, references: int = 1, unpinned: int = 0):
    """Build a pool with ``references`` PINNED snapshots and ``unpinned`` others.

    Snapshot ids are allocated in order, so the pinned ids are ``0..references-1``
    and the unpinned ones follow. Every snapshot carries genuinely different
    weights so a test can tell which one a driver loaded.
    """
    from opponents.snapshot_pool import SnapshotPool

    pool = SnapshotPool(str(tmp_path / "pool"), sampling="uniform")
    for index in range(references):
        net = _net_factory()
        import torch

        with torch.no_grad():
            for param in net.parameters():
                param.add_(float(index + 1))
        pool.add(net.state_dict(), grad_step=index, elo=1000.0, pinned=True)
    for index in range(unpinned):
        net = _net_factory()
        import torch

        with torch.no_grad():
            for param in net.parameters():
                param.mul_(-1.0).add_(float(100 + index))
        pool.add(net.state_dict(), grad_step=100 + index, elo=1000.0, pinned=False)
    return pool


class _FakeReport:
    """The slice of ``EvalReport`` the cycle summary and the loop actually read."""

    def __init__(
        self,
        win_rate: float,
        *,
        n_episodes: int = 10,
        passed_m2: bool = False,
        opponent: str = "scripted_mixed",
    ) -> None:
        self.win_rate = float(win_rate)
        self.n_episodes = int(n_episodes)
        self.mean_episode_length = 12.0
        self.aim_while_invisible = 0.0
        self.passed_m2 = bool(passed_m2)
        self.opponent = str(opponent)


class _MirrorPadEnv:
    """Fake pad env exposing BOTH opponent accessors, each counted separately."""

    MIRROR_OFFSET = 0.25

    def __init__(self, *, k: int = 4, won: bool = False) -> None:
        self.k = int(k)
        self._won = bool(won)
        self._t = 0
        self._obs_index = 0
        self.step_calls: List[Tuple[int, Optional[int]]] = []
        self.observation_calls = 0
        self.view_calls = 0
        self.outcomes: List[Dict[str, Any]] = []
        self.closed = False

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        self._t = 0
        self._obs_index = 0 if seed is None else int(seed)
        return _obs(self._obs_index)

    def opponent_observation(self) -> np.ndarray:
        self.observation_calls += 1
        return (_obs(self._obs_index) + self.MIRROR_OFFSET).astype(np.float32)

    def raw_opponent_view(self) -> OpponentView:
        self.view_calls += 1
        return _view()

    def step(self, action: int, opp_action: Optional[int] = None):
        self.step_calls.append((int(action), opp_action))
        self._t += 1
        self._obs_index += 1
        done = self._t >= self.k
        info = {
            "step": self._t,
            "won": bool(done and self._won),
            "lost": bool(done and not self._won),
            "timeout": False,
        }
        self.outcomes.append(info)
        return _obs(self._obs_index), 0.0, done, info

    def close(self) -> None:
        self.closed = True


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


class _RecordingLogger:
    """A ``MetricsLogger`` stand-in that keeps every row and the final summary."""

    def __init__(self, **_kwargs: Any) -> None:
        self.rows: List[Tuple[Optional[int], Dict[str, Any]]] = []
        self.summaries: List[Dict[str, Any]] = []

    def log(self, metrics: Any, step: Optional[int] = None) -> None:
        self.rows.append((step, dict(metrics)))

    def summary(self, values: Any) -> None:
        self.summaries.append(dict(values))

    def close(self) -> None:
        pass

    def values_of(self, key: str) -> List[Any]:
        return [row[key] for _step, row in self.rows if key in row]

    def steps_of(self, key: str) -> List[Optional[int]]:
        return [step for step, row in self.rows if key in row]


# ===========================================================================
# The eval opponent: a self-play run must fight SOMETHING that moves.
# ===========================================================================


class TestTheSelfPlayEvalGetsTheScriptedYardstick:
    """``build_eval_opponent`` must not leave a self-play run on the dummy.

    Before T13 it returned ``None`` for ``selfplay``, and ``None`` is the
    stationary-dummy path: the eval sent no ``opp_action`` at all, so the win
    rate that selects the demo checkpoint was earned against a target that never
    moves. Nothing raised, and the report even recorded ``opponent="dummy"`` —
    on a run whose every other log line said ``selfplay``.
    """

    def test_a_selfplay_run_gets_a_scripted_eval_opponent(self, tmp_path):
        from agent.train import build_eval_opponent

        factory = build_eval_opponent(_selfplay_cfg(_write_checkpoint(tmp_path / "w.pt")))

        assert factory is not None, (
            "a self-play run's periodic eval fights the STATIONARY dummy: its "
            "win rate is not a win rate against anything that moves, and it is "
            "what selects the demo checkpoint"
        )
        assert factory().name == "scripted_mixed"

    def test_the_yardstick_is_fresh_per_eval(self, tmp_path):
        # Two cycles must face an identical opponent or their win rates are not
        # comparable and "select the best" compares two different fights.
        from agent.train import build_eval_opponent

        factory = build_eval_opponent(_selfplay_cfg(_write_checkpoint(tmp_path / "w.pt")))

        first, second = factory(), factory()

        assert first is not second
        first.begin_episode()
        second.begin_episode()
        assert first.preset is second.preset

    def test_the_dummy_path_still_builds_nothing(self):
        # The regression guard: AC10 keeps the M2 wire line byte-identical.
        from agent.train import build_eval_opponent

        cfg = dataclasses.replace(TrainConfig(), opponent="dummy", arenas=2)

        assert build_eval_opponent(cfg) is None

    def test_the_scripted_path_is_unchanged(self):
        from agent.train import build_eval_opponent

        cfg = dataclasses.replace(TrainConfig(), opponent="scripted", arenas=2)
        factory = build_eval_opponent(cfg)

        assert factory is not None
        assert factory().name == "scripted_mixed"


# ===========================================================================
# The pinned rated driver: one named past self, for a whole track.
# ===========================================================================


class TestAPinnedDriverFightsExactlyOneSnapshot:
    """``selfplay/win_rate_vs_ref_<id>`` names a SPECIFIC snapshot.

    A driver that resampled would spread one track's ten episodes across
    whatever PFSP happened to draw, and the series would be labelled with a
    snapshot the agent mostly did not fight. Nothing about that is observable
    from the metric.
    """

    def test_the_pin_replaces_the_pool_draw_entirely(self, tmp_path):
        pytest.importorskip("torch")

        from agent.train import SnapshotOpponentDriver

        pool = _pool(tmp_path, references=2, unpinned=3)
        reference = pool.pinned_references()[1]

        def _refuse(*_args, **_kwargs):
            raise AssertionError(
                "a pinned reference driver drew from the pool: the track's "
                "episodes are spread over other snapshots and the "
                "selfplay/win_rate_vs_ref_<id> series names the wrong opponent"
            )

        pool.sample_state_dict = _refuse  # type: ignore[assignment]
        driver = SnapshotOpponentDriver(
            _selfplay_cfg(_write_checkpoint(tmp_path / "w.pt")),
            pool,
            0,
            net_factory=_net_factory,
            rated=True,
            reference=reference,
        )

        for _ in range(3):
            driver.begin_episode()
            assert driver.snapshot_id == reference.snapshot_id

        assert driver.reference is reference

    def test_the_pinned_weights_are_the_reference_s_own(self, tmp_path):
        pytest.importorskip("torch")
        import torch

        from agent.train import SnapshotOpponentDriver

        pool = _pool(tmp_path, references=2)
        reference = pool.pinned_references()[1]
        expected = pool.load_state_dict(reference)

        driver = SnapshotOpponentDriver(
            _selfplay_cfg(_write_checkpoint(tmp_path / "w.pt")),
            pool,
            0,
            net_factory=_net_factory,
            rated=True,
            reference=reference,
        )
        driver.begin_episode()

        loaded = driver.net.state_dict()
        assert all(
            torch.equal(loaded[key].cpu(), expected[key].cpu()) for key in expected
        ), "the pinned driver is not playing the reference's weights"

    def test_an_unpinned_driver_still_samples(self, tmp_path):
        # The control: omitting `reference` must leave every training driver's
        # behavior exactly as it was.
        pytest.importorskip("torch")

        from agent.train import SnapshotOpponentDriver

        pool = _pool(tmp_path, references=1, unpinned=4)
        driver = SnapshotOpponentDriver(
            _selfplay_cfg(_write_checkpoint(tmp_path / "w.pt")),
            pool,
            0,
            net_factory=_net_factory,
        )

        assert driver.reference is None
        drawn = set()
        for _ in range(24):
            driver.begin_episode()
            drawn.add(driver.snapshot_id)
        assert len(drawn) > 1, "the sampling driver stopped sampling"

    def test_a_pinned_rated_match_is_rated_against_that_id(self, tmp_path):
        pytest.importorskip("torch")

        from agent.train import build_rated_eval_opponent

        pool = _pool(tmp_path, references=2)
        reference = pool.pinned_references()[1]
        driver = build_rated_eval_opponent(
            _selfplay_cfg(_write_checkpoint(tmp_path / "w.pt")),
            pool,
            net_factory=_net_factory,
            reference=reference,
        )()

        driver.begin_episode()
        driver.observe_outcome({"won": True, "lost": False})

        match = driver.current_match
        assert match is not None
        assert match.snapshot_id == reference.snapshot_id
        assert match.learner_epsilon == 0.0
        assert match.opponent_epsilon == 0.0
        assert match.rated_eligible is True
        assert pool.rated_matches == 1
        assert pool.stats_for(reference.snapshot_id).plays == 1
        assert pool.stats_for(pool.pinned_references()[0].snapshot_id).plays == 0


# ===========================================================================
# The gauntlet: however many references exist.
# ===========================================================================


class TestTheGauntletDegradesGracefully:
    """1, 2 or 3 references must all work (AC8).

    The plan pins snapshot 0 at seed and promotes two more at
    ``reference_promote_grad_steps``, so the FIRST eval cycles of a live run see
    one reference and only the late run sees three. Code that assumed three
    would either crash on the first cycle or, worse, skip the gauntlet entirely
    and leave ``elo/learner_rated`` empty for the hours before the promotion.
    """

    @pytest.mark.parametrize("references", [1, 2, 3])
    def test_one_track_per_pinned_reference(self, tmp_path, references):
        pytest.importorskip("torch")

        from agent.train import build_reference_tracks

        pool = _pool(tmp_path, references=references, unpinned=2)
        tracks = build_reference_tracks(
            _selfplay_cfg(_write_checkpoint(tmp_path / "w.pt")),
            pool,
            n_episodes=10,
            net_factory=_net_factory,
        )

        assert [track.snapshot_id for track in tracks] == list(range(references))
        assert [track.name for track in tracks] == [
            f"snapshot_{i}" for i in range(references)
        ]
        assert all(track.n_episodes == 10 for track in tracks)

    def test_unpinned_snapshots_are_not_reference_tracks(self, tmp_path):
        # An unpinned snapshot is dropped on corruption and can vanish mid-run,
        # so a series named after one would simply stop with no explanation.
        pytest.importorskip("torch")

        from agent.train import build_reference_tracks

        pool = _pool(tmp_path, references=1, unpinned=5)
        tracks = build_reference_tracks(
            _selfplay_cfg(_write_checkpoint(tmp_path / "w.pt")),
            pool,
            net_factory=_net_factory,
        )

        assert [track.snapshot_id for track in tracks] == [0]

    def test_each_track_builds_its_own_pinned_rated_driver(self, tmp_path):
        pytest.importorskip("torch")

        from agent.train import build_reference_tracks

        pool = _pool(tmp_path, references=3)
        tracks = build_reference_tracks(
            _selfplay_cfg(_write_checkpoint(tmp_path / "w.pt")),
            pool,
            net_factory=_net_factory,
        )

        drivers = [track.opponent_factory() for track in tracks]

        assert [d.reference.snapshot_id for d in drivers] == [0, 1, 2]
        assert all(d.rated for d in drivers)
        assert all(d.epsilon == 0.0 and d.learner_epsilon == 0.0 for d in drivers)
        # Distinct objects: one shared driver would carry reference 0's LSTM
        # memory into reference 1's opening episodes.
        assert len({id(d) for d in drivers}) == 3
        # And a second call to the same factory is a FRESH driver, so cycle #1
        # and cycle #40 start from the same state.
        assert tracks[0].opponent_factory() is not drivers[0]

    def test_a_zero_episode_track_is_refused(self, tmp_path):
        pytest.importorskip("torch")

        from agent.train import build_reference_tracks

        pool = _pool(tmp_path, references=1)
        with pytest.raises(ValueError, match="reference eval episodes"):
            build_reference_tracks(
                _selfplay_cfg(_write_checkpoint(tmp_path / "w.pt")),
                pool,
                n_episodes=0,
                net_factory=_net_factory,
            )

    def test_an_empty_pool_yields_no_tracks_rather_than_raising(self, tmp_path):
        pytest.importorskip("torch")

        from agent.train import build_reference_tracks
        from opponents.snapshot_pool import SnapshotPool

        empty = SnapshotPool(str(tmp_path / "empty"), sampling="uniform")

        assert (
            build_reference_tracks(
                _selfplay_cfg(_write_checkpoint(tmp_path / "w.pt")),
                empty,
                net_factory=_net_factory,
            )
            == ()
        )


# ===========================================================================
# The evaluator: the two things it did not do.
# ===========================================================================


class _RecordingObservationOpponent:
    """An eval opponent that wants the MIRRORED observation and scores outcomes."""

    needs_observation = True
    name = "snapshot_0"

    def __init__(self) -> None:
        self.observations: List[np.ndarray] = []
        self.outcomes: List[Dict[str, Any]] = []
        self.begin_calls = 0
        self.noted_epsilons: List[float] = []

    def note_learner_epsilon(self, epsilon: float) -> None:
        self.noted_epsilons.append(float(epsilon))

    def begin_episode(self) -> None:
        self.begin_calls += 1

    def act(self, obs) -> int:
        self.observations.append(np.asarray(obs))
        return int(Macro.JUMP)

    def observe_outcome(self, info) -> None:
        self.outcomes.append(dict(info))


class _RecordingViewOpponent:
    """An eval opponent with NEITHER optional member — the scripted shape."""

    name = "scripted_mixed"

    def __init__(self) -> None:
        self.views: List[Any] = []
        self.begin_calls = 0

    def begin_episode(self) -> None:
        self.begin_calls += 1

    def act(self, view) -> int:
        self.views.append(view)
        return int(Macro.APPROACH)


class _ScriptedGreedyPolicy:
    """A torch-free ``GreedyPolicy`` returning a fixed action every step."""

    def __init__(self, action: int = int(Macro.ATTACK)) -> None:
        self._action = int(action)
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1

    def act(self, obs) -> int:
        return self._action


class TestTheEvaluatorRoutesAndScores:
    """``evaluate`` had no ``needs_observation`` branch and never scored outcomes.

    Both gaps are silent. The first hands a frozen ``DuelingDRQN`` an
    ``OpponentView`` — a wrong TYPE, so it surfaces as an exception minutes into
    a cycle rather than at wiring time. The second builds no
    :class:`~opponents.snapshot_pool.MatchResult` at all, so a perfectly wired
    gauntlet still leaves ``elo/learner_rated`` and every
    ``selfplay/win_rate_vs_ref_<id>`` series empty.
    """

    def test_an_observation_opponent_is_fed_the_mirror(self):
        from eval.evaluate import evaluate

        env = _MirrorPadEnv(k=3)
        opponent = _RecordingObservationOpponent()

        evaluate(
            env,
            _ScriptedGreedyPolicy(),
            n_episodes=2,
            timeout_cap=64,
            max_episode_steps=8,
            opponent=opponent,
        )

        assert env.observation_calls == 6, "the mirrored accessor was not read"
        assert env.view_calls == 0, (
            "the eval fed a frozen DRQN an OpponentView: a wrong TYPE, and the "
            "failure lands minutes into an eval cycle rather than at wiring time"
        )
        # ONE read, ONE macro, ONE step per decision window.
        assert len(opponent.observations) == len(env.step_calls) == 6
        assert all(vec.shape == (OBS_DIM,) for vec in opponent.observations)
        assert all(action is not None for _agent, action in env.step_calls)

    def test_the_final_info_reaches_observe_outcome(self):
        from eval.evaluate import evaluate

        env = _MirrorPadEnv(k=3, won=True)
        opponent = _RecordingObservationOpponent()

        evaluate(
            env,
            _ScriptedGreedyPolicy(),
            n_episodes=2,
            timeout_cap=64,
            max_episode_steps=8,
            opponent=opponent,
        )

        assert len(opponent.outcomes) == 2, (
            "no episode was scored back: no MatchResult is built, so "
            "elo/learner_rated never moves and nothing reports why"
        )
        assert all(info["won"] is True for info in opponent.outcomes)
        assert all(info["step"] == 3 for info in opponent.outcomes), (
            "observe_outcome got a mid-episode info, not the TERMINAL one"
        )

    def test_the_learner_epsilon_is_never_reported(self):
        # THE TRIPWIRE. `note_learner_epsilon` RAISES on a nonzero epsilon
        # against a rated driver, deliberately: silently accepting cfg.eps_end
        # would un-rate the whole cycle invisibly. The evaluator is greedy by
        # construction and has no epsilon to report, so it must not call it at
        # all — a path that does kills the run at the first eval cycle.
        from eval.evaluate import evaluate

        opponent = _RecordingObservationOpponent()

        evaluate(
            _MirrorPadEnv(k=2),
            _ScriptedGreedyPolicy(),
            n_episodes=1,
            timeout_cap=64,
            max_episode_steps=8,
            opponent=opponent,
        )

        assert opponent.noted_epsilons == [], (
            "the eval reported an epsilon to its opponent; against a RATED "
            "driver that raises and ends the run tens of minutes into a cycle"
        )

    def test_a_real_rated_driver_survives_a_whole_eval(self, tmp_path):
        # The end of the same story, with the real driver rather than a
        # recorder: a full eval must leave the driver still rated and its
        # matches still eligible.
        pytest.importorskip("torch")

        from agent.train import build_rated_eval_opponent
        from eval.evaluate import evaluate

        pool = _pool(tmp_path, references=1)
        driver = build_rated_eval_opponent(
            _selfplay_cfg(_write_checkpoint(tmp_path / "w.pt")),
            pool,
            net_factory=_net_factory,
            reference=pool.pinned_references()[0],
        )()

        report = evaluate(
            _MirrorPadEnv(k=3, won=True),
            _ScriptedGreedyPolicy(),
            n_episodes=4,
            timeout_cap=64,
            max_episode_steps=8,
            opponent=driver,
            opponent_name="snapshot_0",
        )

        assert report.opponent == "snapshot_0"
        assert report.win_rate == pytest.approx(1.0)
        assert pool.rated_matches == 4
        assert pool.learner_elo_rated > pool.elo_initial

    def test_a_view_opponent_is_untouched(self):
        # AC10: the scripted eval driver has NEITHER optional member, and its
        # routing must be exactly what it always was.
        from eval.evaluate import evaluate

        env = _MirrorPadEnv(k=3)
        opponent = _RecordingViewOpponent()

        evaluate(
            env,
            _ScriptedGreedyPolicy(),
            n_episodes=2,
            timeout_cap=64,
            max_episode_steps=8,
            opponent=opponent,
        )

        assert env.view_calls == 6
        assert env.observation_calls == 0
        assert all(isinstance(view, OpponentView) for view in opponent.views)

    def test_the_dummy_path_reads_neither_accessor(self):
        # No opponent at all: the byte-identical M1/M2 wire line.
        from eval.evaluate import evaluate

        env = _MirrorPadEnv(k=3)

        evaluate(
            env,
            _ScriptedGreedyPolicy(),
            n_episodes=2,
            timeout_cap=64,
            max_episode_steps=8,
        )

        assert env.view_calls == 0
        assert env.observation_calls == 0
        assert all(opp is None for _agent, opp in env.step_calls)


# ===========================================================================
# The frozen candidate.
# ===========================================================================


class TestTheCandidateIsFrozenOnDisk:
    """One immutable net, staged on disk, for a WHOLE eval cycle.

    The greedy policy holds ``trainer.online`` by reference and the learner
    thread never stops, so an un-frozen four-track cycle scores reference 0 and
    reference 2 on different networks — and the three win rates the checkpoint
    is selected on then describe three different agents.
    """

    def _trainer(self):
        from agent.train import Trainer

        return Trainer(
            dataclasses.replace(
                TrainConfig(),
                arenas=2,
                batch_size=4,
                seq_len=2,
                burn_in=1,
                n_step=1,
                min_replay=1,
            ),
            net_kwargs=dict(_TINY_NET),
        )

    def test_the_candidate_does_not_track_the_live_net(self, tmp_path):
        pytest.importorskip("torch")
        import torch

        from agent.train import EVAL_CANDIDATE_FILENAME, _freeze_eval_candidate
        from eval.evaluate import DRQNGreedyPolicy

        trainer = self._trainer()
        candidate = _freeze_eval_candidate(
            trainer=trainer,
            policy_cls=DRQNGreedyPolicy,
            net_factory=_net_factory,
            directory=str(tmp_path),
            log=None,
        )
        assert candidate is not None
        assert candidate.path.endswith(EVAL_CANDIDATE_FILENAME)
        frozen = {
            key: value.clone()
            for key, value in candidate.policy._net.state_dict().items()
        }

        # What the learner thread does for the rest of the cycle.
        with torch.no_grad():
            for param in trainer.online.parameters():
                param.add_(1.0)

        after = candidate.policy._net.state_dict()
        assert all(torch.equal(after[key], frozen[key]) for key in frozen), (
            "the eval policy still holds the LIVE net, so the gauntlet's "
            "per-reference win rates describe different networks"
        )
        assert all(
            torch.equal(candidate.weights[key], frozen[key]) for key in frozen
        ), "the weights handed to the save-best hook are not the evaluated ones"
        assert candidate.grad_step == int(trainer.grad_step)

    def test_the_staged_file_is_the_evaluated_net(self, tmp_path):
        pytest.importorskip("torch")
        import torch

        from agent.train import _freeze_eval_candidate, load_checkpoint_state_dict
        from eval.evaluate import DRQNGreedyPolicy

        candidate = _freeze_eval_candidate(
            trainer=self._trainer(),
            policy_cls=DRQNGreedyPolicy,
            net_factory=_net_factory,
            directory=str(tmp_path),
            log=None,
        )
        assert candidate is not None

        on_disk = load_checkpoint_state_dict(candidate.path)
        in_memory = candidate.policy._net.state_dict()
        assert all(
            torch.equal(on_disk[key].cpu(), in_memory[key].cpu()) for key in on_disk
        ), "what the gauntlet scored is not what is on disk"

    def test_a_staging_failure_degrades_loudly_instead_of_raising(self, tmp_path):
        # A full disk at 4am must not end a 24-hour run; it must say so and let
        # the cycle fall back to the live net.
        pytest.importorskip("torch")

        import agent.train as train_module
        from eval.evaluate import DRQNGreedyPolicy

        lines: List[str] = []

        def _boom(*_args, **_kwargs):
            raise OSError("no space left on device")

        original = train_module._atomic_torch_save
        train_module._atomic_torch_save = _boom
        try:
            candidate = train_module._freeze_eval_candidate(
                trainer=self._trainer(),
                policy_cls=DRQNGreedyPolicy,
                net_factory=_net_factory,
                directory=str(tmp_path),
                log=lines.append,
            )
        finally:
            train_module._atomic_torch_save = original

        assert candidate is None
        assert any("could NOT be frozen" in line for line in lines)

    def test_an_unreadable_staged_file_degrades_instead_of_scoring_it(
        self, tmp_path, monkeypatch
    ):
        # THE READ-BACK's only functional contribution. Replacing
        # `net.load_state_dict(load_checkpoint_state_dict(path, ...))` with
        # `net.load_state_dict(weights)` is observationally identical for every
        # other test in this file — the two agree unless the write was corrupt —
        # so nothing proved the round trip catches a staged file that does not
        # read back. Here the write SUCCEEDS and only the read fails, which is
        # the shape of a torn/truncated save at 4am: the cycle must fall back to
        # the live net and say so, not score a net nobody can reload.
        pytest.importorskip("torch")

        import agent.train as train_module
        from agent.train import EVAL_CANDIDATE_FILENAME
        from eval.evaluate import DRQNGreedyPolicy

        lines: List[str] = []

        def _unreadable(*_args, **_kwargs):
            raise RuntimeError("PytorchStreamReader failed reading zip archive")

        monkeypatch.setattr(train_module, "load_checkpoint_state_dict", _unreadable)
        candidate = train_module._freeze_eval_candidate(
            trainer=self._trainer(),
            policy_cls=DRQNGreedyPolicy,
            net_factory=_net_factory,
            directory=str(tmp_path),
            log=lines.append,
        )

        assert candidate is None, (
            "the candidate survived a staged file that cannot be read back, so "
            "the gauntlet would score a net the save-best hook cannot reproduce"
        )
        assert any("could NOT be frozen" in line for line in lines)
        assert (tmp_path / EVAL_CANDIDATE_FILENAME).exists(), (
            "the write never happened, so this test failed for the wrong reason "
            "and says nothing about the read-back"
        )

    def test_the_grad_step_read_cannot_end_the_run(self, tmp_path):
        # S1. `grad_step` and the weight clone used to sit OUTSIDE the try, so
        # the "torch failure" the docstring promises to survive would propagate
        # out of the very function whose contract is to never raise.
        pytest.importorskip("torch")

        import agent.train as train_module
        from eval.evaluate import DRQNGreedyPolicy

        class _UnreadableTrainer:
            """A learner whose live net refuses to be read, as torch can."""

            device = "cpu"

            @property
            def grad_step(self) -> int:
                raise RuntimeError("CUDA error: an illegal memory access")

        lines: List[str] = []
        candidate = train_module._freeze_eval_candidate(
            trainer=_UnreadableTrainer(),
            policy_cls=DRQNGreedyPolicy,
            net_factory=_net_factory,
            directory=str(tmp_path),
            log=lines.append,
        )

        assert candidate is None
        assert any("could NOT be frozen" in line for line in lines), (
            "the failure was not reported, so a degraded cycle would look "
            "identical to a frozen one in the log"
        )


# ===========================================================================
# The cycle: one pause, one connection, every track.
# ===========================================================================


class _RecordingEnv:
    """Records how the eval env was constructed; steps nothing."""

    constructed: List[Dict[str, Any]] = []

    def __init__(self, **kwargs: Any) -> None:
        type(self).constructed.append(dict(kwargs))

    def close(self) -> None:  # pragma: no cover - never called by the eval
        raise AssertionError("the eval closed the BORROWED transport")


@pytest.fixture
def recording_env(monkeypatch):
    """Replace ``env.mc_pvp_env.MCPvPEnv`` with a constructor recorder."""
    import env.mc_pvp_env as env_module

    class _Env(_RecordingEnv):
        constructed: List[Dict[str, Any]] = []

    monkeypatch.setattr(env_module, "MCPvPEnv", _Env)
    return _Env


class TestTheWholeCycleRunsOverOneBorrowedConnection:
    """Every track shares ONE env, ONE transport and ONE frozen policy.

    The bridge accepts exactly one TCP client per pad and resolves a second by
    DESTROYING the incumbent — the failure that has taken this project down four
    times. A gauntlet that built its own env per reference would be four
    borrows where one is safe.
    """

    def _trainer(self):
        from agent.train import Trainer

        return Trainer(
            dataclasses.replace(
                TrainConfig(),
                arenas=2,
                batch_size=4,
                seq_len=2,
                burn_in=1,
                n_step=1,
                min_replay=1,
            ),
            net_kwargs=dict(_TINY_NET),
        )

    def _tracks(self, tmp_path, references: int):
        from agent.train import build_reference_tracks

        pool = _pool(tmp_path, references=references)
        return build_reference_tracks(
            _selfplay_cfg(_write_checkpoint(tmp_path / "w.pt")),
            pool,
            n_episodes=2,
            net_factory=_net_factory,
        )

    def test_every_track_shares_one_env_and_one_policy(
        self, tmp_path, recording_env
    ):
        pytest.importorskip("torch")

        from agent.train import _eval_against_opponent, _freeze_eval_candidate
        from eval.evaluate import DRQNGreedyPolicy

        calls: List[Dict[str, Any]] = []

        def _fake_evaluate(env, policy, **kwargs):
            calls.append({"env": env, "policy": policy, **kwargs})
            return _FakeReport(0.5, n_episodes=int(kwargs["n_episodes"]))

        trainer = self._trainer()
        candidate = _freeze_eval_candidate(
            trainer=trainer,
            policy_cls=DRQNGreedyPolicy,
            net_factory=_net_factory,
            directory=str(tmp_path),
            log=None,
        )
        transport = object()

        outcome = _eval_against_opponent(
            trainer=trainer,
            evaluate=_fake_evaluate,
            policy_cls=DRQNGreedyPolicy,
            shared_transport=transport,
            n_episodes=3,
            timeout_cap=64,
            env_max_episode_steps=64,
            eval_step_cap=4,
            logger=_RecordingLogger(),
            is_live=False,
            base_seed=7,
            log=None,
            opponent=_RecordingViewOpponent(),
            mirror_opponent=True,
            candidate=candidate,
            reference_tracks=self._tracks(tmp_path, 3),
        )

        # ONE env, over the BORROWED transport, never a second connection.
        assert len(recording_env.constructed) == 1
        assert recording_env.constructed[0]["transport"] is transport
        assert recording_env.constructed[0]["auto_connect"] is False
        assert recording_env.constructed[0]["mirror_opponent"] is True
        # Main track + one leg per reference, all on the SAME env object and the
        # SAME frozen policy.
        assert len(calls) == 4
        assert len({id(call["env"]) for call in calls}) == 1
        assert all(call["policy"] is candidate.policy for call in calls)
        # The gauntlet's own shape.
        assert [o.snapshot_id for o in outcome.reference_outcomes] == [0, 1, 2]
        assert [call["opponent_name"] for call in calls[1:]] == [
            "snapshot_0",
            "snapshot_1",
            "snapshot_2",
        ]
        assert all(call["n_episodes"] == 2 for call in calls[1:])
        # The reference legs must NOT share the run logger: `evaluate` writes a
        # per-episode series at step=episode_index and a run summary, so three
        # extra tracks would overwrite the main track's rows at steps 0..1 and
        # end the cycle with the LAST reference's numbers in the run summary.
        assert calls[0]["logger"] is not None
        assert all(call["logger"] is None for call in calls[1:])
        # The outcome names the frozen candidate, not a later net.
        assert outcome.weights is candidate.weights
        assert outcome.grad_step == candidate.grad_step

    def test_no_candidate_keeps_the_historical_live_net_path(
        self, tmp_path, recording_env
    ):
        pytest.importorskip("torch")
        import torch

        from agent.train import _eval_against_opponent
        from eval.evaluate import DRQNGreedyPolicy

        trainer = self._trainer()
        seen: List[Any] = []

        def _fake_evaluate(env, policy, **kwargs):
            seen.append(policy)
            return _FakeReport(0.5, n_episodes=int(kwargs["n_episodes"]))

        outcome = _eval_against_opponent(
            trainer=trainer,
            evaluate=_fake_evaluate,
            policy_cls=DRQNGreedyPolicy,
            shared_transport=object(),
            n_episodes=1,
            timeout_cap=64,
            env_max_episode_steps=64,
            eval_step_cap=4,
            logger=None,
            is_live=False,
            base_seed=0,
            log=None,
        )

        assert seen[0]._net is trainer.online
        assert outcome.reference_outcomes == ()
        # Still a detached clone, per the pre-T13 contract.
        taken = {k: v.clone() for k, v in outcome.weights.items()}
        with torch.no_grad():
            for param in trainer.online.parameters():
                param.add_(1.0)
        assert all(torch.equal(outcome.weights[k], taken[k]) for k in taken)

    def test_the_handoff_forwards_the_candidate_and_the_tracks(
        self, tmp_path, monkeypatch
    ):
        # The middle hop: every OTHER test of the eval path stubs
        # `_eval_against_opponent` out, so without this pin the two forwarding
        # lines in `_eval_via_designated_arena` can be deleted with a green
        # suite — and the gauntlet silently stops running.
        pytest.importorskip("torch")

        import agent.train as train_module

        recorded: Dict[str, Any] = {}

        class _Env:
            _transport = object()

        class _Collector:
            def __init__(self) -> None:
                self.calls: List[str] = []

            def pause(self) -> None:
                self.calls.append("pause")

            def wait_until_idle(self, timeout: float) -> bool:
                return True

            def current_env(self):
                return _Env()

            def resume(self) -> None:
                self.calls.append("resume")

        class _Pool:
            def __init__(self, collector) -> None:
                self._collector = collector

            def collector_for(self, arena_id: int):
                return self._collector if arena_id == 0 else None

        monkeypatch.setattr(
            train_module,
            "_eval_against_opponent",
            lambda **kwargs: recorded.update(kwargs) or "outcome",
        )
        collector = _Collector()
        tracks = self._tracks(tmp_path, 2)
        sentinel = object()

        train_module._eval_via_designated_arena(
            trainer=MagicMock(),
            pool=_Pool(collector),
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
            candidate=sentinel,
            reference_tracks=tracks,
        )

        assert recorded["candidate"] is sentinel
        assert recorded["reference_tracks"] is tracks
        assert collector.calls == ["pause", "resume"]


# ===========================================================================
# Selection: aggregate AND worst reference.
# ===========================================================================


class TestTheCycleSummary:
    """``_summarize_reference_outcomes`` and the metrics row it feeds."""

    def _outcome(self, snapshot_id: int, win_rate: float, episodes: int = 10):
        from agent.train import _ReferenceOutcome

        return _ReferenceOutcome(
            snapshot_id=snapshot_id,
            report=_FakeReport(win_rate, n_episodes=episodes),
        )

    def test_the_aggregate_is_episode_weighted(self):
        from agent.train import _summarize_reference_outcomes

        verdict = _summarize_reference_outcomes(
            [self._outcome(0, 1.0, 10), self._outcome(1, 0.0, 30)]
        )

        assert verdict is not None
        # 10 wins over 40 episodes, NOT the 0.5 an unweighted mean would give.
        assert verdict.aggregate == pytest.approx(0.25)
        assert verdict.worst == pytest.approx(0.0)
        assert verdict.references == 2
        assert verdict.episodes == 40

    def test_no_reference_is_none_not_a_zero(self):
        # A zeroed verdict would read as "lost every reference episode" and push
        # the selector's floor to a number no candidate earned, blocking every
        # later cycle from ever shipping.
        from agent.train import _summarize_reference_outcomes

        assert _summarize_reference_outcomes([]) is None
        assert _summarize_reference_outcomes([self._outcome(0, 0.0, 0)]) is None

    def test_the_row_carries_the_yardstick_and_both_selection_inputs(self):
        from agent.train import (
            _summarize_reference_outcomes,
            selfplay_eval_cycle_row,
        )

        verdict = _summarize_reference_outcomes(
            [self._outcome(0, 0.4), self._outcome(1, 0.8)]
        )
        row = selfplay_eval_cycle_row(_FakeReport(0.9), verdict)

        assert row == {
            "selfplay/scripted_win_rate": pytest.approx(0.9),
            "selfplay/reference_win_rate": pytest.approx(0.6),
            "selfplay/worst_reference_win_rate": pytest.approx(0.4),
            "selfplay/references_evaluated": pytest.approx(2.0),
        }

    def test_a_missing_measurement_is_omitted_not_zeroed(self):
        # A logged 0.0 reads as "lost everything", which is a different claim
        # from "did not play".
        from agent.train import selfplay_eval_cycle_row

        assert selfplay_eval_cycle_row(None, None) == {}
        assert selfplay_eval_cycle_row(_FakeReport(0.3), None) == {
            "selfplay/scripted_win_rate": pytest.approx(0.3)
        }


class TestSelectionNeedsBothCriteria:
    """The demo checkpoint must dominate the incumbent on BOTH numbers."""

    def test_the_legacy_single_criterion_path_is_unchanged(self):
        from agent.train import _BestCheckpointSelector

        selector = _BestCheckpointSelector()

        assert selector.consider(0.0, 1) is False  # must beat zero
        assert selector.best_win_rate == pytest.approx(0.0)
        assert selector.consider(0.5, 2) is True
        assert selector.consider(0.5, 3) is False  # ties keep the earlier net
        assert selector.consider(0.6, 4) is True
        assert selector.best_grad_step == 4

    def test_a_gain_on_the_aggregate_alone_does_not_ship(self):
        from agent.train import _BestCheckpointSelector

        selector = _BestCheckpointSelector()

        assert selector.consider(0.60, 1, worst_reference=0.40) is True
        # Better on the mean, WORSE against its weakest reference: the
        # specialization failure a single aggregate hides.
        assert selector.consider(0.75, 2, worst_reference=0.30) is False
        assert selector.best_grad_step == 1

    def test_a_gain_on_the_worst_alone_does_not_ship(self):
        from agent.train import _BestCheckpointSelector

        selector = _BestCheckpointSelector()

        assert selector.consider(0.60, 1, worst_reference=0.40) is True
        assert selector.consider(0.55, 2, worst_reference=0.50) is False
        assert selector.best_grad_step == 1

    def test_both_improving_ships(self):
        from agent.train import _BestCheckpointSelector

        selector = _BestCheckpointSelector()

        assert selector.consider(0.60, 1, worst_reference=0.40) is True
        assert selector.consider(0.61, 2, worst_reference=0.41) is True
        assert selector.best_grad_step == 2
        assert selector.best_worst_reference == pytest.approx(0.41)

    def test_a_rejected_candidate_does_not_move_the_bar(self):
        # If a rejected candidate raised either bar, the bar would describe a
        # net nobody is holding and a genuinely better candidate could be
        # refused for failing to beat it.
        from agent.train import _BestCheckpointSelector

        selector = _BestCheckpointSelector()

        assert selector.consider(0.60, 1, worst_reference=0.40) is True
        assert selector.consider(0.90, 2, worst_reference=0.10) is False
        assert selector.best_win_rate == pytest.approx(0.60)
        assert selector.consider(0.65, 3, worst_reference=0.45) is True

    def test_a_cycle_that_wins_nothing_is_never_shipped(self):
        from agent.train import _BestCheckpointSelector

        selector = _BestCheckpointSelector()

        assert selector.consider(0.0, 1, worst_reference=0.0) is False
        assert selector.best_grad_step == -1

    def test_the_first_ship_of_a_run_faces_the_worst_reference_bar_too(self):
        # W1. `best_worst_reference` starts at the -1.0 "no incumbent" sentinel,
        # so on a FRESH selector the worst reference faced NO bar at all and only
        # the aggregate was gated: a cycle averaging 0.90 while being SWEPT by one
        # reference shipped — and then BECAME the incumbent every later candidate
        # is measured against. The test above cannot see this; it fails the
        # aggregate gate first (rate 0.0) and never reaches the floor.
        #
        # A fresh run hides it too — the first eval predates the first promotion,
        # so there is one reference and aggregate == worst. A RESUMED run does
        # not: this selector is rebuilt per `train_multi_arena` call, so a restart
        # against a reloaded pool of three pinned references re-opens the window
        # with the whole night's training already behind it.
        from agent.train import _BestCheckpointSelector

        selector = _BestCheckpointSelector()

        assert selector.consider(0.90, 1, worst_reference=0.0) is False
        assert selector.best_grad_step == -1
        # And a rejection here moves NEITHER bar, so the next honest candidate is
        # not measured against a net nobody is holding.
        assert selector.best_win_rate == pytest.approx(-1.0)
        assert selector.best_worst_reference == pytest.approx(-1.0)
        # The bar is `min_win_rate`, not "anything nonzero": one won episode
        # against the weakest reference is enough to be selectable.
        assert selector.consider(0.90, 2, worst_reference=0.10) is True
        assert selector.best_grad_step == 2


# ===========================================================================
# End to end: does `elo/learner_rated` actually fill?
# ===========================================================================


def _playing_eval_stub(monkeypatch, *, scripted_win_rate: float = 0.5, won=True):
    """Stub the eval HANDOFF with one that PLAYS the reference tracks it is given.

    Not a canned report: it builds each track's real rated driver, runs its
    episodes' begin/observe cycle against the real pool, and returns the
    resulting :class:`_EvalOutcome`. That is what makes the assertions below
    statements about the production wiring — the tracks the loop actually
    passes, the drivers they actually build, and the pool they actually rate
    into — with the socket the only thing faked away.

    Returns the list the stub appends each cycle's kwargs to.
    """
    import agent.train as train_module
    from distributed.weights import clone_state_dict

    seen: List[Dict[str, Any]] = []

    def _stub(**kwargs):
        seen.append(dict(kwargs))
        outcomes = []
        for track in kwargs["reference_tracks"]:
            driver = track.opponent_factory()
            for _ in range(track.n_episodes):
                driver.begin_episode()
                driver.observe_outcome({"won": bool(won), "lost": not bool(won)})
            outcomes.append(
                train_module._ReferenceOutcome(
                    snapshot_id=track.snapshot_id,
                    report=_FakeReport(
                        1.0 if won else 0.0,
                        n_episodes=track.n_episodes,
                        opponent=track.name,
                    ),
                )
            )
        trainer = kwargs["trainer"]
        return train_module._EvalOutcome(
            report=_FakeReport(scripted_win_rate, n_episodes=1),
            weights=clone_state_dict(trainer.online.state_dict()),
            grad_step=int(trainer.grad_step),
            reference_outcomes=tuple(outcomes),
        )

    monkeypatch.setattr(train_module, "_eval_via_designated_arena", _stub)
    return seen


def _scheduled_gauntlet_stub(monkeypatch, cycles):
    """Stub the eval handoff with a per-cycle schedule of per-reference rates.

    ``cycles`` is a list of ``[rate_per_reference, ...]``; the last entry is
    held once the schedule runs out, so a run of unknown length still follows a
    known script. The reference ids are fabricated because what is under test
    here is the SELECTION wiring — which two numbers the loop feeds the selector
    — not the track building, which has its own tests above.

    Returns the list of cycle indices actually driven.
    """
    import agent.train as train_module
    from distributed.weights import clone_state_dict

    driven: List[int] = []

    def _stub(**kwargs):
        rates = cycles[min(len(driven), len(cycles) - 1)]
        driven.append(len(driven))
        trainer = kwargs["trainer"]
        return train_module._EvalOutcome(
            report=_FakeReport(0.0, n_episodes=1),
            weights=clone_state_dict(trainer.online.state_dict()),
            grad_step=int(trainer.grad_step),
            reference_outcomes=tuple(
                train_module._ReferenceOutcome(
                    snapshot_id=index,
                    report=_FakeReport(rate, n_episodes=10),
                )
                for index, rate in enumerate(rates)
            ),
        )

    monkeypatch.setattr(train_module, "_eval_via_designated_arena", _stub)
    return driven


class TestTheRatedSeriesFillsInARun:
    """AC7's whole point: after T13, ``elo/learner_rated`` has data.

    Before it, every self-play cycle printed
    ``(0 rated matches - elo/learner_rated is EMPTY)`` and the series sat at its
    initial rating all night, looking exactly like a flat curve.
    """

    def _run(self, tmp_path, monkeypatch, **overrides):
        warm = _write_checkpoint(tmp_path / "warm.pt")
        cfg = _selfplay_cfg(warm)
        pads = [_MirrorPadEnv() for _ in range(cfg.arenas)]
        logger = _RecordingLogger()
        lines: List[str] = []
        call: Dict[str, Any] = dict(
            snapshot_dir=str(tmp_path / "pool"),
            logger=logger,
            log=lines.append,
            eval_every_grad_steps=1,
            checkpoint_every_grad_steps=0,
            reference_eval_episodes=2,
            max_grad_steps=6,
        )
        call.update(overrides)
        _run_multi_arena(cfg, pads, **call)
        return logger, lines

    def test_the_loop_hands_the_eval_a_reference_track(self, tmp_path, monkeypatch):
        pytest.importorskip("torch")

        seen = _playing_eval_stub(monkeypatch)
        self._run(tmp_path, monkeypatch)

        assert seen, "no eval cycle ran; this test can prove nothing"
        tracks = seen[0]["reference_tracks"]
        assert tracks, (
            "the eval cycle got NO reference track: no rated match is ever "
            "played and elo/learner_rated stays empty by construction"
        )
        assert [track.snapshot_id for track in tracks] == [0]
        assert tracks[0].n_episodes == 2
        # And a frozen candidate, not the live net.
        assert seen[0]["candidate"] is not None

    def test_the_rated_series_moves_and_the_reference_series_appears(
        self, tmp_path, monkeypatch
    ):
        pytest.importorskip("torch")

        _playing_eval_stub(monkeypatch)
        logger, _lines = self._run(tmp_path, monkeypatch)

        rated = logger.values_of("elo/learner_rated")
        assert rated, "elo/learner_rated was never logged"
        assert rated[-1] > 1000.0, (
            "elo/learner_rated never moved off its initial rating: the eval "
            "cycle scored no rated match and AC7's curve is empty"
        )
        assert logger.values_of("selfplay/rated_matches")[-1] > 0
        assert logger.values_of("selfplay/win_rate_vs_ref_0"), (
            "AC8's per-reference series never appeared, so nothing reports "
            "whether the agent still beats the reference it started from"
        )
        assert logger.values_of("selfplay/scripted_win_rate"), (
            "the absolute yardstick was not logged; only relative Elo remains, "
            "and Elo can inflate"
        )
        assert logger.values_of("selfplay/reference_win_rate")
        assert logger.values_of("selfplay/worst_reference_win_rate")

    def test_the_empty_warning_stops_firing_once_the_gauntlet_runs(
        self, tmp_path, monkeypatch
    ):
        pytest.importorskip("torch")

        _playing_eval_stub(monkeypatch)
        _logger, lines = self._run(tmp_path, monkeypatch)

        selfplay_lines = [line for line in lines if "selfplay:" in line]
        assert selfplay_lines, "the self-play stderr line never fired"
        assert not any(
            "elo/learner_rated is EMPTY" in line for line in selfplay_lines
        ), (
            "the run still reports an EMPTY rated series after a rated eval "
            "cycle: the row is being logged BEFORE the gauntlet scores it"
        )
        assert any("rated match(es)" in line for line in selfplay_lines)
        assert any("reference gauntlet:" in line for line in lines)

    def test_the_warning_DOES_fire_when_no_rated_eval_runs(
        self, tmp_path, monkeypatch
    ):
        # The control, and the guard-against-silence T12 built: with eval off,
        # nothing rates anything, and the run must SAY so rather than leave a
        # flat curve to be discovered in the morning.
        pytest.importorskip("torch")

        _logger, lines = self._run(
            tmp_path,
            monkeypatch,
            eval_every_grad_steps=0,
            checkpoint_every_grad_steps=1,
            checkpoint_hook=lambda _trainer, _step: None,
        )

        assert any(
            "0 rated matches - elo/learner_rated is EMPTY" in line for line in lines
        ), (
            "a run that rated nothing said nothing about it; an empty "
            "elo/learner_rated plots exactly like a flat one"
        )

    def test_the_checkpoint_is_selected_on_the_reference_aggregate(
        self, tmp_path, monkeypatch
    ):
        pytest.importorskip("torch")

        _playing_eval_stub(monkeypatch, scripted_win_rate=0.0, won=True)
        saved: List[Tuple[int, Dict[str, Any]]] = []
        warm = _write_checkpoint(tmp_path / "warm.pt")
        cfg = _selfplay_cfg(warm)
        pads = [_MirrorPadEnv() for _ in range(cfg.arenas)]
        result = _run_multi_arena(
            cfg,
            pads,
            snapshot_dir=str(tmp_path / "pool"),
            eval_every_grad_steps=1,
            reference_eval_episodes=2,
            max_grad_steps=6,
            best_checkpoint_hook=lambda _t, step, meta, _w: saved.append(
                (int(step), dict(meta))
            ),
        )

        assert saved, (
            "nothing was ever selected: the scripted track scored 0.0 and the "
            "reference aggregate — the actual selection input — was ignored"
        )
        _step, meta = saved[0]
        assert meta["win_rate"] == pytest.approx(1.0)
        assert meta["scripted_win_rate"] == pytest.approx(0.0)
        assert "pinned reference" in meta["eval_opponent"]
        assert meta["worst_reference_win_rate"] == pytest.approx(1.0)
        assert meta["references_evaluated"] == 1
        # And the result must not label a reference aggregate as a scripted
        # win rate — that line is what freeze day picks a checkpoint from.
        assert result.eval_opponent == "scripted_mixed"
        assert "pinned reference" in result.selection_opponent
        assert result.best_win_rate == pytest.approx(1.0)

    def test_a_specializing_candidate_is_not_shipped(self, tmp_path, monkeypatch):
        # THE WIRING of the second criterion. `_BestCheckpointSelector` has its
        # own unit tests, but nothing else proves the LOOP hands it the cycle's
        # worst reference: pass `worst_reference=None` there and every test but
        # this one stays green while the run happily ships a policy that has
        # stopped beating the reference it started from.
        pytest.importorskip("torch")

        driven = _scheduled_gauntlet_stub(
            monkeypatch,
            [
                # Cycle 0: aggregate 0.50, worst 0.40 - ships.
                [0.60, 0.40],
                # Every later cycle: aggregate 0.625 (BETTER) but worst 0.30
                # (WORSE). That is specialization, not improvement, and it must
                # not replace the incumbent.
                [0.95, 0.30],
            ],
        )
        saved: List[int] = []
        warm = _write_checkpoint(tmp_path / "warm.pt")
        cfg = _selfplay_cfg(warm)
        pads = [_MirrorPadEnv() for _ in range(cfg.arenas)]
        result = _run_multi_arena(
            cfg,
            pads,
            snapshot_dir=str(tmp_path / "pool"),
            eval_every_grad_steps=1,
            reference_eval_episodes=2,
            max_grad_steps=12,
            best_checkpoint_hook=lambda _t, step, _m, _w: saved.append(int(step)),
        )

        assert len(driven) >= 2, (
            "only one eval cycle ran, so the second (specializing) candidate "
            "was never offered and this test proves nothing"
        )
        assert len(saved) == 1, (
            "a candidate that gained on the reference AGGREGATE while "
            "collapsing against its weakest reference was shipped: the worst-"
            "reference criterion is not reaching the selector"
        )
        assert result.best_win_rate == pytest.approx(0.5)


# ===========================================================================
# The cycle is the risky part of the night — and it must not be the fatal one.
# ===========================================================================


class _BorrowablePadEnv(_MirrorPadEnv):
    """A pad env the eval handoff can actually BORROW a transport from.

    ``_eval_via_designated_arena`` returns ``None`` — quietly, no exception —
    when the paused env exposes no ``_transport``, which is precisely the path a
    test of the FAILURE guard must not take. The attribute is never dereferenced:
    the gauntlet is stubbed out before it can build an env over it.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._transport = object()


class TestAFailedEvalCycleCostsOneCycleNotTheNight:
    """An eval that raises must SKIP the cycle, not end the run (W2).

    Nothing wrapped the eval block. A ``BridgeError`` on the wire — or an
    ``opponent_observation()`` refusal from a mis-built env — propagated out of
    the driver loop into teardown and ended the run, and T13 multiplied the
    exposure: a cycle went from one ``evaluate`` call to four, so a failure in
    the last leg now also discards the legs already fought. Both ``finally``
    blocks underneath (resume the collector, restore train mode) run either way,
    so the only thing the loop has to add is to survive and say so — which is
    the policy ``_save_latest`` and the best-checkpoint hook already follow.
    """

    def test_a_raising_gauntlet_skips_the_cycle_and_the_run_survives(
        self, tmp_path, monkeypatch
    ):
        pytest.importorskip("torch")

        import agent.train as train_module
        from env.mc_pvp_env import BridgeError

        raised: List[int] = []
        attempts: List[int] = []
        #: Was the designated collector resumed, as seen the instant the eval's
        #: own ``finally`` has run and BEFORE the loop's handler is entered?
        resumed: List[bool] = []

        def _boom(**_kwargs: Any):
            # Raised from inside the borrow: by now the REAL handoff has paused
            # the designated collector and taken its connection, which is the
            # state a skipped cycle has to leave clean.
            raised.append(len(raised))
            raise BridgeError("pad lost its connection during the reference leg")

        real_handoff = train_module._eval_via_designated_arena

        def _watched_handoff(**kwargs: Any):
            collector = kwargs["pool"].collector_for(kwargs["designated_arena"])
            attempts.append(len(attempts))
            try:
                return real_handoff(**kwargs)
            finally:
                resumed.append(
                    collector is not None
                    and not collector._pause.is_set()
                    and not collector.paused_idle
                )

        # Only the gauntlet is faked; the pause/borrow/resume handoff is REAL.
        monkeypatch.setattr(train_module, "_eval_against_opponent", _boom)
        monkeypatch.setattr(
            train_module, "_eval_via_designated_arena", _watched_handoff
        )

        warm = _write_checkpoint(tmp_path / "warm.pt")
        cfg = _selfplay_cfg(warm)
        pads = [_BorrowablePadEnv() for _ in range(cfg.arenas)]
        lines: List[str] = []
        result = _run_multi_arena(
            cfg,
            pads,
            snapshot_dir=str(tmp_path / "pool"),
            log=lines.append,
            eval_every_grad_steps=1,
            reference_eval_episodes=2,
            max_grad_steps=12,
            eval_pause_timeout=5.0,
        )

        assert raised, (
            "the gauntlet never raised: the handoff bailed out before reaching "
            "it, so this test exercises none of the guard it exists for"
        )
        assert all(resumed), (
            "the designated collector was left PARKED after a failed eval; the "
            "run would keep training on N-1 arenas forever with nothing saying "
            "why, which is worse than the crash this guard replaces"
        )
        assert result.stop_reason == "max_grad_steps", (
            "a failed eval cycle ended the run instead of skipping it "
            f"(stop_reason={result.stop_reason!r})"
        )
        skipped = [line for line in lines if "eval cycle SKIPPED" in line]
        assert skipped, "the cycle was skipped SILENTLY, which is the worse half"
        assert "BridgeError" in skipped[0], (
            f"the skip line does not name what failed: {skipped[0]!r}"
        )
        assert len(attempts) >= 2, (
            "the loop attempted exactly one eval, so nothing here shows it kept "
            "evaluating after the failure rather than merely surviving it"
        )
        # The cycle scored NOTHING: no report, so nothing was selected on it.
        assert result.reports == []
        assert result.last_report is None

        # And the boundary was re-armed past the failure. Without this the loop
        # re-fires the same failing cycle every `poll_interval` for the rest of
        # the night, because the success path's assignment sits AFTER the call
        # that raised.
        failed_at = int(skipped[0].split("]")[0].rsplit(" ", 1)[-1])
        next_due = int(skipped[0].split("due at grad_step ")[1].split(";")[0])
        assert next_due > failed_at, (
            f"the next eval is due at {next_due}, at or before the step that "
            f"just failed ({failed_at}): the cycle retries in a hot loop"
        )

    def test_a_ctrl_c_during_an_eval_still_ends_the_run(self, tmp_path, monkeypatch):
        # The guard catches `Exception`, never `BaseException`. A 4am Ctrl-C
        # swallowed here would leave the operator holding a run that refuses to
        # die, and the final checkpoint in the loop's `finally` never written.
        pytest.importorskip("torch")

        import agent.train as train_module

        def _interrupt(**_kwargs: Any):
            raise KeyboardInterrupt()

        monkeypatch.setattr(train_module, "_eval_via_designated_arena", _interrupt)

        warm = _write_checkpoint(tmp_path / "warm.pt")
        cfg = _selfplay_cfg(warm)
        pads = [_MirrorPadEnv() for _ in range(cfg.arenas)]
        saved: List[int] = []
        with pytest.raises(KeyboardInterrupt):
            _run_multi_arena(
                cfg,
                pads,
                snapshot_dir=str(tmp_path / "pool"),
                eval_every_grad_steps=1,
                reference_eval_episodes=2,
                max_grad_steps=12,
                checkpoint_every_grad_steps=0,
                checkpoint_hook=lambda _trainer, step: saved.append(int(step)),
            )

        assert saved, (
            "the interrupted run wrote no final checkpoint: the night's weights "
            "are gone"
        )


class TestTheShippedFileSaysWhetherItWasFrozen:
    """``.best.pt`` must record whether its score came from a frozen net (S2).

    When staging fails the cycle falls back to the LIVE net and logs it — hours
    before anyone reads the file. Without a flag in ``meta`` two ``.best.pt``
    files are identical in provenance on freeze morning, even though one was
    scored against a moving target.
    """

    def _ship(self, tmp_path, monkeypatch, **overrides: Any):
        _playing_eval_stub(monkeypatch, scripted_win_rate=0.0, won=True)
        saved: List[Dict[str, Any]] = []
        warm = _write_checkpoint(tmp_path / "warm.pt")
        cfg = _selfplay_cfg(warm)
        pads = [_MirrorPadEnv() for _ in range(cfg.arenas)]
        call: Dict[str, Any] = dict(
            snapshot_dir=str(tmp_path / "pool"),
            eval_every_grad_steps=1,
            reference_eval_episodes=2,
            max_grad_steps=6,
            best_checkpoint_hook=lambda _t, _step, meta, _w: saved.append(dict(meta)),
        )
        call.update(overrides)
        _run_multi_arena(cfg, pads, **call)
        assert saved, "nothing shipped, so there is no provenance to inspect"
        return saved[0]

    def test_a_frozen_cycle_says_so(self, tmp_path, monkeypatch):
        pytest.importorskip("torch")

        meta = self._ship(tmp_path, monkeypatch)

        assert meta["candidate_frozen"] is True, (
            "a cycle that DID freeze its candidate is recorded as degraded"
        )

    def test_a_degraded_cycle_says_so(self, tmp_path, monkeypatch):
        pytest.importorskip("torch")

        import agent.train as train_module

        # Exactly what an unwritable directory produces at 4am.
        monkeypatch.setattr(
            train_module, "_freeze_eval_candidate", lambda **_kwargs: None
        )
        meta = self._ship(tmp_path, monkeypatch)

        assert meta["candidate_frozen"] is False, (
            "a checkpoint scored on the LIVE net is indistinguishable from one "
            "scored on a frozen candidate; the morning comparison cannot tell "
            "which file sat the exam"
        )


class TestTheRunSaysWhatItIsFighting:
    """The opponent banner and the EMPTY warning had NO test at all.

    ``train_multi_arena``'s ``log`` defaults to ``None`` and every fixture in
    the suite left it there, so every ``_emit`` in that function was dead code
    under test — including the three-branch opponent banner that exists because
    a self-play run once opened its log by announcing ``opponent=dummy``, which
    is also the true symptom of the wiring bug where ``opponent_for`` stays
    ``None``. The guard against silent failure was itself unguarded.
    """

    def _lines(self, cfg, tmp_path, **overrides) -> List[str]:
        pads = [_MirrorPadEnv() for _ in range(cfg.arenas)]
        lines: List[str] = []
        call: Dict[str, Any] = dict(log=lines.append, max_grad_steps=4)
        call.update(overrides)
        _run_multi_arena(cfg, pads, **call)
        return lines

    def test_a_selfplay_run_announces_its_pool(self, tmp_path):
        pytest.importorskip("torch")

        cfg = _selfplay_cfg(_write_checkpoint(tmp_path / "warm.pt"))
        lines = self._lines(cfg, tmp_path, snapshot_dir=str(tmp_path / "pool"))

        assert any("opponent=selfplay (sampling=" in line for line in lines), (
            "the run's opening banner does not say it is self-playing; the "
            "same line reading 'opponent=dummy' is the symptom of the wiring "
            "bug where opponent_for stays None and the whole night trains "
            "against a stationary bot"
        )
        assert any(
            "[multi] opponent=selfplay: snapshot pool at" in line for line in lines
        )

    def test_a_dummy_run_announces_the_dummy(self, tmp_path):
        pytest.importorskip("torch")

        cfg = dataclasses.replace(
            _selfplay_cfg(_write_checkpoint(tmp_path / "warm.pt")),
            opponent="dummy",
            warm_start=None,
        )
        lines = self._lines(cfg, tmp_path)

        assert any("opponent=dummy, " in line for line in lines)
        assert not any("opponent=selfplay" in line for line in lines)

    def test_a_scripted_run_announces_its_curriculum(self, tmp_path):
        pytest.importorskip("torch")

        cfg = dataclasses.replace(
            _selfplay_cfg(_write_checkpoint(tmp_path / "warm.pt")),
            opponent="scripted",
        )
        lines = self._lines(cfg, tmp_path)

        assert any("opponent=scripted (mix_easy" in line for line in lines)

    def test_the_eval_banner_states_the_gauntlet_cost(self, tmp_path, monkeypatch):
        # A cycle's wall clock is the scripted track PLUS one leg per pinned
        # reference. An operator sizing an overnight run must see the
        # multiplier, not infer it from a gap between eval timestamps.
        pytest.importorskip("torch")

        _playing_eval_stub(monkeypatch)
        cfg = _selfplay_cfg(_write_checkpoint(tmp_path / "warm.pt"))
        lines = self._lines(
            cfg,
            tmp_path,
            snapshot_dir=str(tmp_path / "pool"),
            eval_every_grad_steps=1,
            reference_eval_episodes=7,
        )

        assert any(
            "+ 7 eps vs EACH pinned reference" in line for line in lines
        ), "the eval banner does not state the reference gauntlet's cost"


class TestOneGradStepStillLogsOneRow:
    """Both cadences on, so a single grad step satisfies BOTH log predicates.

    The T12 test that carried this name inherited ``eval_every_grad_steps=0``
    from its fixture, so the two call sites could never collide and the latch
    could be deleted with a green suite. Here both cadences are enabled, the
    collision is asserted to have ACTUALLY happened, and only then is the
    uniqueness claim made — so deleting the latch reddens this test.

    The second assertion pins the other half of the design: the eval cycle's
    own rates are deliberately NOT behind the latch. They exist only at the eval
    boundary, so a step the checkpoint cadence happened to share would lose them
    for good rather than delay them by a cycle — which is what the pool row,
    logged at every boundary either way, costs instead.
    """

    def test_a_colliding_step_logs_exactly_one_row(self, tmp_path, monkeypatch):
        pytest.importorskip("torch")

        _playing_eval_stub(monkeypatch)
        warm = _write_checkpoint(tmp_path / "warm.pt")
        cfg = _selfplay_cfg(warm)
        pads = [_MirrorPadEnv() for _ in range(cfg.arenas)]
        logger = _RecordingLogger()
        lines: List[str] = []
        _run_multi_arena(
            cfg,
            pads,
            snapshot_dir=str(tmp_path / "pool"),
            logger=logger,
            log=lines.append,
            # BOTH cadences at 1, so a single grad step satisfies both.
            eval_every_grad_steps=1,
            checkpoint_every_grad_steps=1,
            checkpoint_hook=lambda _trainer, _step: None,
            reference_eval_episodes=1,
            max_grad_steps=6,
        )

        saved_at = {
            int(line.rsplit(" ", 1)[-1])
            for line in lines
            if "checkpoint saved (periodic) at grad_step" in line
        }
        evaluated_at = {
            int(line.split("]")[0].rsplit(" ", 1)[-1])
            for line in lines
            if "reference gauntlet:" in line
        }
        collisions = saved_at & evaluated_at
        assert collisions, (
            "no grad step satisfied BOTH cadences, so this test does not "
            "exercise the collision it exists for"
        )

        steps = logger.steps_of("elo/learner_online")
        assert steps, "the self-play row was never logged at all"
        assert len(steps) == len(set(steps)), (
            f"one grad step logged the self-play row twice: {steps}"
        )
        # And the cycle's own rates survived the very step the pool row had to
        # give up: they are logged outside the latch, at the eval boundary.
        cycle_steps = set(logger.steps_of("selfplay/scripted_win_rate"))
        assert cycle_steps >= collisions, (
            "the eval cycle's rates were dropped at a step the periodic "
            "checkpoint shared; they are only produced here, so they are gone "
            f"rather than delayed (collisions={sorted(collisions)}, "
            f"cycle rows at {sorted(cycle_steps)})"
        )
