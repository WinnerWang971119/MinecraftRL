"""Tests for opponent stepping and the EASY/HARD curriculum gate (T12, AC9/AC10).

What is pinned here, and why each pin exists:

* **The M2 stationary-dummy path is untouched** (AC9). With no opponent
  attached, ``collect_episode`` must call ``env.step(action)`` with ONE
  positional argument and never ask the env for a raw view. A regression here
  would silently poison the retrain by putting an opponent action on a wire that
  the M1/M2 path proves nothing about.
* **One step == one decision window.** The env shadow-tracks the opponent's
  attack meter by COUNTING windows, so the rollout must take exactly one
  ``env.step`` per opponent decision — never a skipped or doubled step.
* **The attack cooldown reaches the bot intact.** ``raw_opponent_view()`` clamps
  it to exactly 1.0 and ``ScriptedBot`` tests readiness at ``>= 1.0 - 1e-6``; a
  value a hair below (``1.0 - 1e-5``) makes the bot NEVER attack, which presents
  as a mysteriously passive opponent rather than as an error. Both polarities
  are pinned.
* **Per-arena isolation.** Each arena gets its OWN driver, bots, and RNG streams,
  routed through the factory the pool actually uses. The behavioral form of the
  check (arena 0's macro stream is identical solo and interleaved with arena 1)
  is what fails if anyone shares one instance across arenas.
* **The gate is a MIXTURE, not a promotion** — after it fires, EASY still arrives
  at ``opponent_mix_easy_after`` (0.2), never 0.
* **The gate cannot stall a run** (AC10) — a run whose gate never fires completes
  at the initial ratio. The stall check runs the loop on a worker thread with a
  join timeout so a blocking implementation FAILS instead of hanging.

No socket, no live server, no Minecraft: every env here is a fake.
"""

from __future__ import annotations

import dataclasses
import random
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest

from agent.actions import Macro, N_ACTIONS
from agent.train_config import TrainConfig
from env.observation_spec import OBS_DIM
from opponents.scripted_bot import OpponentView, ScriptedBot, ScriptedPreset


# Seconds a worker thread gets before a test calls it a stall. Generous enough to
# absorb a loaded CI box, short enough that a genuine block fails the run.
_THREAD_TIMEOUT = 30.0


def _completes_within(fn, timeout: float = _THREAD_TIMEOUT):
    """Run ``fn`` on a daemon thread and FAIL (never hang) if it does not finish.

    Every loop in this file that draws from the curriculum or drives a training
    run goes through here. AC10 says the curriculum must not be able to stall a
    run, and a curriculum that blocked — waiting on a gate that may never fire —
    would otherwise wedge the whole pytest process instead of failing one test.
    A hang is not a test result; this converts it into one.

    Returns ``fn``'s return value, and re-raises whatever ``fn`` raised.
    """
    box: Dict[str, Any] = {}

    def _target() -> None:
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 - re-raised on the main thread
            box["error"] = exc

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(timeout=timeout)
    assert not thread.is_alive(), (
        f"blocked for more than {timeout}s instead of completing; the curriculum "
        "must never wait on anything (AC10: a gate that never fires still trains "
        "to completion)"
    )
    if "error" in box:
        raise box["error"]
    return box.get("value")


def _macro_stream(
    driver,
    *,
    episodes: int = 1,
    steps: int = 10,
    won: bool = False,
    view: Optional[OpponentView] = None,
) -> List[int]:
    """Run whole episodes on ``driver`` and return the flat macro stream.

    Goes through :func:`_completes_within` because ``begin_episode`` and
    ``observe_outcome`` are the two calls that touch the shared curriculum.
    """
    seen = view if view is not None else _MOVING_VIEW

    def _run() -> List[int]:
        out: List[int] = []
        for _ in range(episodes):
            driver.begin_episode()
            out.extend(driver.act(seen) for _ in range(steps))
            driver.observe_outcome({"won": won})
        return out

    return _completes_within(_run)


# ===========================================================================
# Fixtures / fakes
# ===========================================================================


def _view(
    *,
    in_attack_range: bool = False,
    attack_cooldown: float = 1.0,
    self_health: float = 20.0,
    can_see_target: bool = True,
    last_known: Optional[Tuple[float, float, float]] = (3.0, 0.0, 0.0),
) -> OpponentView:
    """Hand-authored omniscient view (no env, no numpy, no bridge)."""
    return OpponentView(
        self_pos=(0.0, 0.0, 0.0),
        self_yaw=0.0,
        self_health=self_health,
        target_pos=(3.0, 0.0, 0.0),
        target_yaw=180.0,
        target_health=20.0,
        distance=3.0,
        in_attack_range=in_attack_range,
        attack_cooldown=attack_cooldown,
        can_see_target=can_see_target,
        last_known_target_pos=last_known,
    )


#: A view whose branch consumes the bot's RNG every call (visible, out of range
#: -> the strafe/jump movement draw). Used wherever a test needs the RNG stream
#: to actually advance.
_MOVING_VIEW = _view(in_attack_range=False, can_see_target=True)


class _OpponentEnv:
    """Fake env exposing the T11a opponent seam: ``raw_opponent_view`` + ``opp_action``.

    Records every ``(action, opp_action)`` pair and how many times the raw view
    was read, so a test can pin the one-step-one-window invariant directly.
    """

    def __init__(
        self,
        *,
        k: int = 5,
        view: Optional[OpponentView] = None,
        won: bool = True,
    ) -> None:
        self.k = int(k)
        self._view = view if view is not None else _MOVING_VIEW
        self._won = bool(won)
        self._t = 0
        self._rng = np.random.default_rng(0)
        self.step_calls: List[Tuple[int, Optional[int]]] = []
        self.view_calls = 0
        self.reset_seeds: List[Optional[int]] = []
        self.closed = False

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        self.reset_seeds.append(seed)
        self._rng = np.random.default_rng(0 if seed is None else int(seed))
        self._t = 0
        return self._obs()

    def raw_opponent_view(self) -> OpponentView:
        self.view_calls += 1
        return self._view

    def step(
        self, action: int, opp_action: Optional[int] = None
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        self.step_calls.append((int(action), opp_action))
        self._t += 1
        done = self._t >= self.k
        info = {
            "step": self._t,
            "won": bool(done and self._won),
            "lost": False,
            "timeout": False,
        }
        return self._obs(), 0.0, done, info

    def close(self) -> None:
        self.closed = True

    def _obs(self) -> np.ndarray:
        return self._rng.uniform(-1.0, 1.0, size=OBS_DIM).astype(np.float32)


class _DummyPathEnv:
    """Fake env that FAILS LOUDLY if the opponent seam is touched (M2 regression).

    ``step`` accepts ``*args``/``**kwargs`` deliberately: the point is to record
    what the caller passed rather than to raise a TypeError that could be mistaken
    for an unrelated signature problem.
    """

    def __init__(self, k: int = 4) -> None:
        self.k = int(k)
        self._t = 0
        self._rng = np.random.default_rng(0)
        self.extra_call_shapes: List[Tuple[tuple, dict]] = []

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        self._rng = np.random.default_rng(0 if seed is None else int(seed))
        self._t = 0
        return self._obs()

    def raw_opponent_view(self) -> OpponentView:  # pragma: no cover - must not run
        raise AssertionError(
            "raw_opponent_view() was called on the stationary-dummy path; the M2 "
            "rollout must never read privileged opponent state."
        )

    def step(self, action, *args, **kwargs):
        self.extra_call_shapes.append((args, dict(kwargs)))
        self._t += 1
        done = self._t >= self.k
        return self._obs(), 0.0, done, {"won": False}

    def _obs(self) -> np.ndarray:
        return self._rng.uniform(-1.0, 1.0, size=OBS_DIM).astype(np.float32)


class _RecordingOpponent:
    """Minimal ``EpisodeOpponent`` that records the protocol call ORDER."""

    def __init__(self, macro: Macro = Macro.IDLE) -> None:
        self._macro = macro
        self.calls: List[str] = []
        self.views: List[OpponentView] = []
        self.outcomes: List[Dict[str, Any]] = []

    def begin_episode(self) -> None:
        self.calls.append("begin")

    def act(self, view: OpponentView) -> int:
        self.calls.append("act")
        self.views.append(view)
        return int(self._macro)

    def observe_outcome(self, info) -> None:
        self.calls.append("outcome")
        self.outcomes.append(dict(info))


def _fixed_policy(torch_mod, action: int = 0):
    """A ``RolloutPolicy`` that always picks ``action`` and carries a 1x1x2 hidden."""

    class _FixedPolicy:
        arena_id = 0
        policy_version = 0
        code_version = ""

        def __init__(self) -> None:
            self.seeds: List[int] = []

        def reseed(self, episode_seed: int) -> None:
            self.seeds.append(int(episode_seed))

        def init_hidden(self):
            zeros = torch_mod.zeros(1, 1, 2)
            return (zeros, zeros.clone())

        def act(self, obs, hidden, epsilon):
            return int(action), hidden

    return _FixedPolicy()


def _cfg(**overrides) -> TrainConfig:
    """A TrainConfig with the curriculum knobs overridable per test."""
    return dataclasses.replace(TrainConfig(), **overrides)


# ===========================================================================
# TrainConfig fields (the Contracts block, verbatim)
# ===========================================================================


class TestTrainConfigOpponentFields:
    """The six new fields exist with EXACTLY the contracted names and defaults."""

    def test_defaults_match_the_contract(self):
        cfg = TrainConfig()
        assert cfg.opponent == "dummy"
        assert cfg.opponent_mix_easy == 0.8
        assert cfg.opponent_mix_easy_after == 0.2
        assert cfg.opponent_gate_winrate == 0.6
        assert cfg.opponent_gate_window == 50
        assert cfg.warm_start is None

    def test_the_default_config_is_the_m2_dummy_path(self):
        # The regression that would silently poison the retrain: a default config
        # must still describe the stationary dummy.
        assert TrainConfig().opponent == "dummy"

    def test_an_unknown_opponent_is_refused(self):
        with pytest.raises(ValueError, match="opponent must be one of"):
            _cfg(opponent="scripted_hard")

    @pytest.mark.parametrize("value", [-0.01, 1.01, float("nan")])
    def test_mix_easy_out_of_range_is_refused(self, value):
        with pytest.raises(ValueError, match="opponent_mix_easy must be in"):
            _cfg(opponent_mix_easy=value)

    @pytest.mark.parametrize("value", [-0.01, 1.01, float("nan")])
    def test_mix_easy_after_out_of_range_is_refused(self, value):
        with pytest.raises(ValueError, match="opponent_mix_easy_after must be in"):
            _cfg(opponent_mix_easy_after=value)

    @pytest.mark.parametrize("value", [-0.5, 1.5, float("nan")])
    def test_gate_winrate_out_of_range_is_refused(self, value):
        with pytest.raises(ValueError, match="opponent_gate_winrate must be in"):
            _cfg(opponent_gate_winrate=value)

    @pytest.mark.parametrize("value", [0, -1])
    def test_a_non_positive_gate_window_is_refused(self, value):
        with pytest.raises(ValueError, match="opponent_gate_window must be >= 1"):
            _cfg(opponent_gate_window=value)

    def test_an_empty_warm_start_is_refused(self):
        # `--warm-start ""` would otherwise read as "warm start requested" while
        # naming no checkpoint at all.
        with pytest.raises(ValueError, match="warm_start must be a non-empty path"):
            _cfg(warm_start="")

    def test_a_warm_start_path_is_accepted(self):
        assert _cfg(warm_start="runs/best.pt").warm_start == "runs/best.pt"


# ===========================================================================
# The gate
# ===========================================================================


class TestOpponentCurriculumGate:
    """The win-rate gate: full window, EASY-only, latching."""

    def test_a_partial_window_never_fires_the_gate(self):
        # The easiest bug in this task: 1 EASY win out of 1 is a 100% win rate.
        from agent.train import OpponentCurriculum

        cfg = _cfg(opponent_gate_window=50, opponent_gate_winrate=0.6)
        curriculum = OpponentCurriculum(cfg)

        for _ in range(10):
            assert curriculum.record_episode(ScriptedPreset.EASY, True) is False

        assert curriculum.gate_fired is False
        assert curriculum.easy_window_win_rate() is None
        assert curriculum.mix_easy() == cfg.opponent_mix_easy

    def test_the_gate_fires_when_a_full_window_clears_the_threshold(self):
        from agent.train import OpponentCurriculum

        cfg = _cfg(opponent_gate_window=10, opponent_gate_winrate=0.6)
        curriculum = OpponentCurriculum(cfg)

        # 6 wins / 4 losses == exactly the threshold, and the window is full.
        outcomes = [True] * 6 + [False] * 4
        fired = [curriculum.record_episode(ScriptedPreset.EASY, won) for won in outcomes]

        assert fired == [False] * 9 + [True], (
            "the gate must fire on the episode that COMPLETES a qualifying window, "
            "exactly once"
        )
        assert curriculum.gate_fired is True
        assert curriculum.easy_window_win_rate() == pytest.approx(0.6)
        assert curriculum.mix_easy() == cfg.opponent_mix_easy_after

    def test_a_full_window_below_the_threshold_does_not_fire(self):
        from agent.train import OpponentCurriculum

        cfg = _cfg(opponent_gate_window=10, opponent_gate_winrate=0.6)
        curriculum = OpponentCurriculum(cfg)

        for won in [True] * 5 + [False] * 5:
            curriculum.record_episode(ScriptedPreset.EASY, won)

        assert curriculum.gate_fired is False
        assert curriculum.easy_window_win_rate() == pytest.approx(0.5)
        assert curriculum.mix_easy() == cfg.opponent_mix_easy

    def test_the_window_rolls(self):
        from agent.train import OpponentCurriculum

        cfg = _cfg(opponent_gate_window=4, opponent_gate_winrate=0.75)
        curriculum = OpponentCurriculum(cfg)

        for _ in range(4):
            curriculum.record_episode(ScriptedPreset.EASY, False)
        assert curriculum.gate_fired is False

        # The four losses roll out of the window as four wins roll in.
        for _ in range(3):
            curriculum.record_episode(ScriptedPreset.EASY, True)
        assert curriculum.gate_fired is True

    def test_hard_episodes_never_enter_the_gate_window(self):
        # The gate measures the agent against EASY; HARD outcomes are counted for
        # reporting only, or a strong HARD run would fire an EASY gate.
        from agent.train import OpponentCurriculum

        cfg = _cfg(opponent_gate_window=5, opponent_gate_winrate=0.6)
        curriculum = OpponentCurriculum(cfg)

        for _ in range(100):
            curriculum.record_episode(ScriptedPreset.HARD, True)

        stats = curriculum.stats()
        assert curriculum.gate_fired is False
        assert stats["easy_window_size"] == 0
        assert stats["hard_episodes"] == 100
        assert stats["hard_wins"] == 100
        assert stats["easy_episodes"] == 0

    def test_the_gate_latches_once_fired(self):
        from agent.train import OpponentCurriculum

        cfg = _cfg(opponent_gate_window=4, opponent_gate_winrate=0.75)
        curriculum = OpponentCurriculum(cfg)

        for _ in range(4):
            curriculum.record_episode(ScriptedPreset.EASY, True)
        assert curriculum.gate_fired is True

        # A losing streak long enough to empty the window must not un-fire it: the
        # window refills ~4x slower after the shift, so an un-firing gate flaps.
        for _ in range(20):
            assert curriculum.record_episode(ScriptedPreset.EASY, False) is False
        assert curriculum.gate_fired is True
        assert curriculum.mix_easy() == cfg.opponent_mix_easy_after

    def test_counters_track_both_tiers(self):
        from agent.train import OpponentCurriculum

        curriculum = OpponentCurriculum(_cfg(opponent_gate_window=50))
        curriculum.record_episode(ScriptedPreset.EASY, True)
        curriculum.record_episode(ScriptedPreset.EASY, False)
        curriculum.record_episode(ScriptedPreset.HARD, True)

        stats = curriculum.stats()
        assert stats["episodes"] == 3
        assert (stats["easy_episodes"], stats["easy_wins"]) == (2, 1)
        assert (stats["hard_episodes"], stats["hard_wins"]) == (1, 1)

    def test_concurrent_recording_loses_no_episode(self):
        # The arenas are concurrent threads feeding one gate.
        from agent.train import OpponentCurriculum

        curriculum = OpponentCurriculum(
            _cfg(opponent_gate_window=1_000_000, opponent_gate_winrate=1.0)
        )
        n_threads, per_thread = 8, 200

        def _worker() -> None:
            for i in range(per_thread):
                preset = ScriptedPreset.EASY if i % 2 == 0 else ScriptedPreset.HARD
                curriculum.record_episode(preset, i % 3 == 0)

        threads = [threading.Thread(target=_worker) for _ in range(n_threads)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=_THREAD_TIMEOUT)
            assert not thread.is_alive(), "record_episode blocked under contention"

        stats = curriculum.stats()
        assert stats["episodes"] == n_threads * per_thread
        assert stats["easy_episodes"] + stats["hard_episodes"] == stats["episodes"]


class TestTheCurriculumIsAMixtureNotAPromotion:
    """After the gate fires the mixture SHIFTS; it does not become HARD-only."""

    def _fired_curriculum(self, cfg):
        from agent.train import OpponentCurriculum

        curriculum = OpponentCurriculum(cfg)
        for _ in range(cfg.opponent_gate_window):
            curriculum.record_episode(ScriptedPreset.EASY, True)
        assert curriculum.gate_fired is True
        return curriculum

    def test_mix_easy_shifts_to_the_after_ratio_not_to_zero(self):
        cfg = _cfg(opponent_gate_window=4, opponent_mix_easy_after=0.2)
        curriculum = self._fired_curriculum(cfg)
        assert curriculum.mix_easy() == pytest.approx(0.2)
        assert curriculum.mix_easy() != 0.0

    def test_easy_keeps_being_drawn_after_the_gate(self):
        # Fixed seed + fixed N: the counts are exact, not statistical.
        cfg = _cfg(opponent_gate_window=4, opponent_mix_easy_after=0.2)
        curriculum = self._fired_curriculum(cfg)

        rng = random.Random(0)
        draws = _completes_within(
            lambda: [curriculum.sample_preset(rng) for _ in range(4_000)]
        )
        easy = sum(1 for preset in draws if preset is ScriptedPreset.EASY)

        assert 700 <= easy <= 900, (
            f"expected ~20% EASY after the gate, got {easy}/4000. A HARD-only "
            "promotion (mix_easy == 0) is not the contracted curriculum."
        )
        assert easy < 4_000 - easy, "HARD must dominate after the shift"

    def test_the_initial_mixture_is_easy_dominated(self):
        from agent.train import OpponentCurriculum

        curriculum = OpponentCurriculum(_cfg(opponent_mix_easy=0.8))
        rng = random.Random(0)
        draws = _completes_within(
            lambda: [curriculum.sample_preset(rng) for _ in range(4_000)]
        )
        easy = sum(1 for preset in draws if preset is ScriptedPreset.EASY)

        assert 3_100 <= easy <= 3_300, f"expected ~80% EASY before the gate, got {easy}"

    def test_a_degenerate_mixture_is_honored_exactly(self):
        from agent.train import OpponentCurriculum

        rng = random.Random(1)
        all_easy = OpponentCurriculum(_cfg(opponent_mix_easy=1.0))
        all_hard = OpponentCurriculum(_cfg(opponent_mix_easy=0.0))
        assert _completes_within(
            lambda: {all_easy.sample_preset(rng) for _ in range(200)}
        ) == {ScriptedPreset.EASY}
        assert _completes_within(
            lambda: {all_hard.sample_preset(rng) for _ in range(200)}
        ) == {ScriptedPreset.HARD}


class TestAGateThatNeverFiresCannotStallTheRun:
    """AC10: the mixture simply stays put and training completes normally."""

    def test_a_never_winning_run_completes_at_the_initial_ratio(self):
        # Driven on a WORKER THREAD with a join timeout so a curriculum that
        # blocked (waiting for a gate that never fires) FAILS instead of hanging.
        from agent.train import build_scripted_opponents

        cfg = _cfg(arenas=2, opponent="scripted", opponent_gate_window=50)
        curriculum, opponent_for = build_scripted_opponents(cfg)
        driver = opponent_for(0)

        episodes = 500
        completed: List[int] = []

        def _run() -> None:
            for _ in range(episodes):
                driver.begin_episode()
                for _ in range(3):
                    driver.act(_MOVING_VIEW)
                driver.observe_outcome({"won": False})
                completed.append(1)

        _completes_within(_run)

        assert len(completed) == episodes
        assert curriculum.gate_fired is False
        assert curriculum.mix_easy() == cfg.opponent_mix_easy
        assert curriculum.stats()["episodes"] == episodes

    def test_an_impossible_gate_still_lets_every_episode_through(self):
        from agent.train import OpponentCurriculum

        # winrate 1.0 over a window longer than the run: unreachable by design.
        cfg = _cfg(opponent_gate_winrate=1.0, opponent_gate_window=10_000)
        curriculum = OpponentCurriculum(cfg)
        rng = random.Random(7)

        def _run() -> None:
            for i in range(1_000):
                preset = curriculum.sample_preset(rng)
                curriculum.record_episode(preset, i % 2 == 0)

        _completes_within(_run)

        assert curriculum.gate_fired is False
        assert curriculum.stats()["episodes"] == 1_000
        assert curriculum.mix_easy() == cfg.opponent_mix_easy


# ===========================================================================
# Per-arena isolation
# ===========================================================================


class TestPerArenaOpponentIsolation:
    """Every arena owns its driver, its bots, and its RNG streams."""

    def test_the_factory_hands_each_arena_its_own_driver_and_bots(self):
        from agent.train import build_scripted_opponents

        cfg = _cfg(arenas=4, opponent="scripted")
        _curriculum, opponent_for = build_scripted_opponents(cfg)

        drivers = [opponent_for(i) for i in range(4)]
        assert len({id(driver) for driver in drivers}) == 4, (
            "two arenas were handed the same driver; all ScriptedBot state is "
            "instance-level, so sharing one interleaves the arenas' RNG streams"
        )
        for preset in (ScriptedPreset.EASY, ScriptedPreset.HARD):
            bots = [driver.bot_for(preset) for driver in drivers]
            assert len({id(bot) for bot in bots}) == 4
        # ... and every bot object in the whole fleet is distinct.
        every_bot = [
            driver.bot_for(preset)
            for driver in drivers
            for preset in (ScriptedPreset.EASY, ScriptedPreset.HARD)
        ]
        assert len({id(bot) for bot in every_bot}) == 8

    def test_the_same_arena_is_memoized(self):
        # A fresh driver per call would re-seed the bots every episode and replay
        # one identical opponent stream forever.
        from agent.train import build_scripted_opponents

        _curriculum, opponent_for = build_scripted_opponents(
            _cfg(arenas=2, opponent="scripted")
        )
        assert opponent_for(0) is opponent_for(0)
        assert opponent_for(1) is not opponent_for(0)

    def test_arena_zero_is_unaffected_by_arena_one_interleaving(self):
        # The behavioral form of the isolation check: if the two arenas shared one
        # ScriptedBot (or one mixture RNG), interleaving arena 1's decisions would
        # change arena 0's macro stream.
        from agent.train import build_scripted_opponents

        cfg = _cfg(arenas=2, opponent="scripted")

        _c_solo, solo_for = build_scripted_opponents(cfg)
        solo_macros = _macro_stream(solo_for(0), episodes=5, steps=8)

        _c_mixed, mixed_for = build_scripted_opponents(cfg)
        arena0, arena1 = mixed_for(0), mixed_for(1)

        def _interleaved() -> List[int]:
            out: List[int] = []
            for _ in range(5):
                arena0.begin_episode()
                arena1.begin_episode()
                for _ in range(8):
                    out.append(arena0.act(_MOVING_VIEW))
                    arena1.act(_MOVING_VIEW)
                arena0.observe_outcome({"won": False})
                arena1.observe_outcome({"won": False})
            return out

        mixed_macros = _completes_within(_interleaved)

        assert mixed_macros == solo_macros, (
            "arena 0's macro stream changed when arena 1 acted alongside it; the "
            "arenas are sharing opponent RNG state"
        )

    def test_two_arenas_do_not_produce_the_same_stream(self):
        from agent.train import build_scripted_opponents

        _curriculum, opponent_for = build_scripted_opponents(
            _cfg(arenas=2, opponent="scripted")
        )
        streams = [
            _macro_stream(opponent_for(arena), steps=60) for arena in (0, 1)
        ]

        assert streams[0] != streams[1], (
            "both arenas drew the same opponent stream; the per-arena seed bands "
            "collapsed onto one seed"
        )

    def test_the_whole_run_is_reproducible_from_the_config_seed(self):
        # Same cfg -> same opponent stream. The constructor seed governs the run;
        # bot.reset() per episode is a no-op on the RNG (gym convention).
        from agent.train import build_scripted_opponents

        cfg = _cfg(arenas=2, opponent="scripted", seed=11)

        def _stream() -> List[int]:
            _curriculum, opponent_for = build_scripted_opponents(cfg)
            return _macro_stream(opponent_for(1), episodes=4, steps=10)

        assert _stream() == _stream()

    def test_a_different_seed_gives_a_different_stream(self):
        from agent.train import build_scripted_opponents

        def _stream(seed: int) -> List[int]:
            _curriculum, opponent_for = build_scripted_opponents(
                _cfg(arenas=2, opponent="scripted", seed=seed)
            )
            return _macro_stream(opponent_for(0), steps=60)

        assert _stream(0) != _stream(1)

    def test_episodes_are_decorrelated_within_one_arena(self):
        # bot.reset() must NOT re-seed: back-to-back episodes would otherwise
        # replay one identical macro sequence.
        from agent.train import build_scripted_opponents

        _curriculum, opponent_for = build_scripted_opponents(
            _cfg(arenas=2, opponent="scripted", opponent_mix_easy=1.0)
        )
        driver = opponent_for(0)
        episodes = [_macro_stream(driver, steps=40) for _ in range(2)]

        assert episodes[0] != episodes[1], (
            "consecutive episodes replayed an identical macro stream; reset() was "
            "re-seeding when it must be a no-op on the RNG"
        )


class TestOpponentSeedScheme:
    """The seed helper stays clear of the per-episode seed band."""

    def test_roles_are_distinct_within_an_arena(self):
        from agent.train import opponent_seed

        cfg = _cfg(arenas=2)
        seeds = {opponent_seed(cfg, 0, role) for role in ("mixture", "easy", "hard")}
        assert len(seeds) == 3

    def test_arenas_never_collide(self):
        from agent.train import opponent_seed

        cfg = _cfg(arenas=8)
        seeds = [
            opponent_seed(cfg, arena, role)
            for arena in range(8)
            for role in ("mixture", "easy", "hard")
        ]
        assert len(set(seeds)) == len(seeds)

    def test_opponent_seeds_sit_above_any_realistic_episode_seed(self):
        from agent.train import arena_episode_seed, opponent_seed

        cfg = _cfg(arenas=2)
        for arena in range(2):
            lowest = opponent_seed(cfg, arena, "mixture")
            # Half a stride of clearance: 500k episodes at the default stride.
            assert lowest > arena_episode_seed(cfg, arena, 100_000)

    def test_an_unknown_role_is_refused(self):
        from agent.train import opponent_seed

        with pytest.raises(ValueError, match="unknown opponent seed role"):
            opponent_seed(_cfg(), 0, "medium")


# ===========================================================================
# collect_episode: the wire path
# ===========================================================================


class TestCollectEpisodeStepsTheOpponent:
    """AC9: the loop steps a Python opponent policy and sends ``opp_action``."""

    def test_one_view_one_macro_one_step_per_decision(self):
        torch = pytest.importorskip("torch")
        from agent.train import collect_episode

        env = _OpponentEnv(k=5)
        opponent = _RecordingOpponent(macro=Macro.APPROACH)

        episode = collect_episode(
            env,
            _fixed_policy(torch),
            max_steps=None,
            episode_index=0,
            epsilon=0.0,
            episode_seed=3,
            opponent=opponent,
        )

        assert len(episode.transitions) == 5
        assert len(env.step_calls) == 5
        assert env.view_calls == 5, (
            "the opponent's shadow attack meter counts decision windows, so the "
            "raw view must be read exactly once per env.step"
        )
        assert opponent.calls == ["begin"] + ["act"] * 5 + ["outcome"]

    def test_every_step_carries_the_opponents_macro(self):
        torch = pytest.importorskip("torch")
        from agent.train import collect_episode

        env = _OpponentEnv(k=4)
        collect_episode(
            env,
            _fixed_policy(torch, action=2),
            max_steps=None,
            episode_index=0,
            epsilon=0.0,
            episode_seed=1,
            opponent=_RecordingOpponent(macro=Macro.STRAFE_L),
        )

        assert env.step_calls == [(2, int(Macro.STRAFE_L))] * 4
        for _action, opp_action in env.step_calls:
            assert opp_action is not None
            assert 0 <= opp_action < N_ACTIONS

    def test_the_final_info_is_handed_back_for_scoring(self):
        torch = pytest.importorskip("torch")
        from agent.train import collect_episode

        env = _OpponentEnv(k=3, won=True)
        opponent = _RecordingOpponent()
        collect_episode(
            env,
            _fixed_policy(torch),
            max_steps=None,
            episode_index=0,
            epsilon=0.0,
            episode_seed=1,
            opponent=opponent,
        )

        assert len(opponent.outcomes) == 1
        assert opponent.outcomes[0]["won"] is True

    def test_a_truncated_episode_is_not_scored_as_a_win(self):
        torch = pytest.importorskip("torch")
        from agent.train import collect_episode

        env = _OpponentEnv(k=50, won=True)  # never reaches its own terminal step
        opponent = _RecordingOpponent()
        collect_episode(
            env,
            _fixed_policy(torch),
            max_steps=4,
            episode_index=0,
            epsilon=0.0,
            episode_seed=1,
            opponent=opponent,
        )

        assert len(env.step_calls) == 4
        assert opponent.outcomes[0]["won"] is False

    def test_the_dummy_path_puts_nothing_extra_on_the_wire(self):
        # AC9's other half: the M2 stationary-dummy path is byte-identical.
        torch = pytest.importorskip("torch")
        from agent.train import collect_episode

        env = _DummyPathEnv(k=4)
        episode = collect_episode(
            env,
            _fixed_policy(torch),
            max_steps=None,
            episode_index=0,
            epsilon=0.0,
            episode_seed=5,
        )

        assert len(episode.transitions) == 4
        assert env.extra_call_shapes == [((), {})] * 4, (
            "env.step() must be called with exactly one positional argument on the "
            "stationary-dummy path; an opp_action there changes M2 behavior"
        )


class TestTheAttackCooldownReachesTheBotIntact:
    """A cooldown a hair under 1.0 silently disables the bot's attack entirely."""

    def _opp_actions(self, torch, view: OpponentView) -> List[int]:
        from agent.train import build_scripted_opponents, collect_episode

        env = _OpponentEnv(k=12, view=view)
        _curriculum, opponent_for = build_scripted_opponents(
            _cfg(arenas=2, opponent="scripted", opponent_mix_easy=1.0)
        )
        _completes_within(
            lambda: collect_episode(
                env,
                _fixed_policy(torch),
                max_steps=None,
                episode_index=0,
                epsilon=0.0,
                episode_seed=1,
                opponent=opponent_for(0),
            )
        )
        return [opp for _action, opp in env.step_calls]

    def test_a_clamped_cooldown_makes_the_bot_attack(self):
        torch = pytest.importorskip("torch")
        opp_actions = self._opp_actions(
            torch, _view(in_attack_range=True, attack_cooldown=1.0)
        )
        assert opp_actions == [int(Macro.ATTACK)] * 12, (
            "an in-range bot with a fully charged meter must swing; anything that "
            "perturbs attack_cooldown on the way through makes it silently passive"
        )

    def test_a_cooldown_a_hair_below_one_never_attacks(self):
        # Pinned deliberately: this is what a rounded / re-derived / stale view
        # looks like from the outside — a mysteriously passive opponent, not an
        # error. The producer clamps to exactly 1.0 for this reason.
        torch = pytest.importorskip("torch")
        opp_actions = self._opp_actions(
            torch, _view(in_attack_range=True, attack_cooldown=1.0 - 1e-5)
        )
        assert int(Macro.ATTACK) not in opp_actions

    def test_the_view_object_is_passed_through_untouched(self):
        torch = pytest.importorskip("torch")
        from agent.train import collect_episode

        view = _view(in_attack_range=True, attack_cooldown=1.0)
        env = _OpponentEnv(k=3, view=view)
        opponent = _RecordingOpponent()
        collect_episode(
            env,
            _fixed_policy(torch),
            max_steps=None,
            episode_index=0,
            epsilon=0.0,
            episode_seed=1,
            opponent=opponent,
        )

        assert len(opponent.views) == 3
        for seen in opponent.views:
            assert seen is view
            assert seen.attack_cooldown == 1.0


class TestScriptedOpponentDriver:
    """The per-arena driver's episode lifecycle."""

    def test_act_returns_a_valid_macro_index(self):
        from agent.train import build_scripted_opponents

        _curriculum, opponent_for = build_scripted_opponents(
            _cfg(arenas=2, opponent="scripted")
        )
        driver = opponent_for(0)
        _completes_within(driver.begin_episode)

        for view in (
            _MOVING_VIEW,
            _view(in_attack_range=True, attack_cooldown=1.0),
            _view(can_see_target=False, last_known=(1.0, 0.0, 1.0)),
            _view(can_see_target=False, last_known=None),
            _view(self_health=1.0),
        ):
            action = driver.act(view)
            assert isinstance(action, int)
            assert 0 <= action < N_ACTIONS
            assert action in {int(macro) for macro in Macro}

    def test_the_driver_scores_its_own_tier(self):
        from agent.train import build_scripted_opponents

        cfg = _cfg(arenas=2, opponent="scripted", opponent_mix_easy=1.0)
        curriculum, opponent_for = build_scripted_opponents(cfg)
        driver = opponent_for(0)

        _completes_within(driver.begin_episode)
        assert driver.preset is ScriptedPreset.EASY
        assert driver.name == "scripted_easy"
        _completes_within(lambda: driver.observe_outcome({"won": True}))

        stats = curriculum.stats()
        assert (stats["easy_episodes"], stats["easy_wins"]) == (1, 1)
        assert stats["hard_episodes"] == 0

    def test_a_missing_won_key_reads_as_a_loss(self):
        from agent.train import build_scripted_opponents

        curriculum, opponent_for = build_scripted_opponents(
            _cfg(arenas=2, opponent="scripted", opponent_mix_easy=1.0)
        )
        driver = opponent_for(0)
        _completes_within(driver.begin_episode)
        _completes_within(lambda: driver.observe_outcome({}))

        assert curriculum.stats()["easy_wins"] == 0
        assert curriculum.stats()["easy_episodes"] == 1

    def test_hard_bots_carry_the_hard_preset_parameters(self):
        from agent.train import build_scripted_opponents

        _curriculum, opponent_for = build_scripted_opponents(
            _cfg(arenas=2, opponent="scripted")
        )
        driver = opponent_for(0)

        assert isinstance(driver.bot_for(ScriptedPreset.HARD), ScriptedBot)
        assert driver.bot_for(ScriptedPreset.HARD).name == "scripted_hard"
        assert driver.bot_for(ScriptedPreset.EASY).name == "scripted_easy"
        # T11c owns making this reach the server; T12 only carries it.
        assert driver.bot_for(ScriptedPreset.HARD).config.knockback_immune is False


# ===========================================================================
# The MULTI-ARENA path end to end (offline, fake envs, real threads)
# ===========================================================================


class _ArenaRecorder:
    """Thread-safe record of what one arena's env was asked to do."""

    def __init__(self, arena_id: int) -> None:
        self.arena_id = int(arena_id)
        self._lock = threading.Lock()
        self.opp_actions: List[Optional[int]] = []
        self.view_calls = 0

    def record_step(self, opp_action: Optional[int]) -> None:
        with self._lock:
            self.opp_actions.append(opp_action)

    def record_view(self) -> None:
        with self._lock:
            self.view_calls += 1

    def snapshot(self) -> Tuple[List[Optional[int]], int]:
        with self._lock:
            return list(self.opp_actions), self.view_calls


class _ArenaEnv:
    """Fake pad env for the multi-arena run: reports every opp_action it sees."""

    def __init__(self, recorder: _ArenaRecorder, k: int = 4) -> None:
        self._recorder = recorder
        self.k = int(k)
        self._t = 0
        self._rng = np.random.default_rng(0)

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        self._rng = np.random.default_rng(0 if seed is None else int(seed))
        self._t = 0
        return self._obs()

    def raw_opponent_view(self) -> OpponentView:
        self._recorder.record_view()
        return _MOVING_VIEW

    def step(
        self, action: int, opp_action: Optional[int] = None
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        self._recorder.record_step(opp_action)
        self._t += 1
        done = self._t >= self.k
        # The agent never wins: the gate can never fire, so this run is also the
        # AC10 "gate never fires" case.
        return self._obs(), 0.0, done, {"won": False, "lost": done, "timeout": False}

    def close(self) -> None:
        pass

    def _obs(self) -> np.ndarray:
        return self._rng.uniform(-1.0, 1.0, size=OBS_DIM).astype(np.float32)


def _multi_arena_cfg(opponent: str, n_arenas: int = 2) -> TrainConfig:
    """Minimal multi-arena config: tiny windows, instant warm-up, no relaunches."""
    return dataclasses.replace(
        TrainConfig(),
        arenas=n_arenas,
        opponent=opponent,
        batch_size=4,
        seq_len=2,
        burn_in=1,
        n_step=1,
        min_replay=1,
        replay_capacity=2_000,
        weight_sync_every_k_steps=1,
        fault_relaunch=False,
        seed=0,
    )


def _run_multi_arena(cfg: TrainConfig, recorders: List[_ArenaRecorder]):
    """Drive ``train_multi_arena`` over fake pads to a small grad-step budget.

    Wrapped in :func:`_completes_within`: collectors that blocked (e.g. on a
    curriculum that waited for its gate) would produce no episodes, the learner
    would park forever on ``recv()``, and the driver loop would poll a grad step
    that never advances — a hang. This turns that into a failing test.
    """
    from unittest.mock import MagicMock

    from distributed.actor import GlobalEpisodeCounter
    from distributed.learner import LearnerWatchdog
    from distributed.transport import LocalTransport
    from distributed.weights import WeightStore

    from agent.train import train_multi_arena

    def env_factory_for(arena_id: int):
        recorder = recorders[arena_id]

        def _build() -> _ArenaEnv:
            return _ArenaEnv(recorder=recorder, k=4)

        return _build

    launcher = MagicMock()
    return _completes_within(
        lambda: train_multi_arena(
            cfg,
            env_factory_for=env_factory_for,
            launcher=launcher,
            transport=LocalTransport(maxsize=0),
            weight_store=WeightStore(),
            counter=GlobalEpisodeCounter(),
            max_grad_steps=20,
            eval_every_grad_steps=0,  # no eval: this test is about the collect path
            designated_arena=0,
            stop_on_pass=False,
            relaunch_backoff_seconds=0.001,
            relaunch_backoff_max_seconds=0.001,
            sleep=lambda _s: None,
            poll_interval=0.005,
            net_kwargs={"encoder_hidden": 16, "lstm_hidden": 16, "lstm_layers": 1},
            rollout_step_cap=6,
            watchdog=LearnerWatchdog(patience=500, interval_s=1.0),
        )
    )


class TestMultiArenaOpponentWiring:
    """The retrain's actual path: ``train_multi_arena`` with per-arena opponents."""

    def test_every_arena_sends_an_opp_action(self):
        pytest.importorskip("torch")

        recorders = [_ArenaRecorder(i) for i in range(2)]
        result = _run_multi_arena(_multi_arena_cfg("scripted"), recorders)

        assert result.stop_reason == "max_grad_steps"
        assert result.curriculum is not None
        for recorder in recorders:
            opp_actions, view_calls = recorder.snapshot()
            assert opp_actions, f"arena {recorder.arena_id} collected nothing"
            assert all(a is not None for a in opp_actions), (
                f"arena {recorder.arena_id} stepped without an opp_action; the "
                "collector is not driving its opponent policy"
            )
            assert all(0 <= int(a) < N_ACTIONS for a in opp_actions)
            assert view_calls == len(opp_actions), (
                "one raw view per env.step is the decision-window invariant the "
                "opponent's shadow attack meter depends on"
            )

    def test_a_run_whose_gate_never_fires_completes_at_the_initial_ratio(self):
        # AC10 end to end: the fake env never lets the agent win.
        pytest.importorskip("torch")

        cfg = _multi_arena_cfg("scripted")
        recorders = [_ArenaRecorder(i) for i in range(2)]
        result = _run_multi_arena(cfg, recorders)

        assert result.stop_reason == "max_grad_steps"
        assert result.grad_steps >= 20
        curriculum = result.curriculum
        assert curriculum is not None
        stats = curriculum.stats()
        assert stats["episodes"] >= 1
        assert stats["gate_fired"] is False
        assert curriculum.mix_easy() == cfg.opponent_mix_easy
        assert stats["easy_episodes"] + stats["hard_episodes"] == stats["episodes"]

    def test_the_dummy_path_never_touches_the_opponent_seam(self):
        # The M2 regression guard on the multi-arena path.
        pytest.importorskip("torch")

        recorders = [_ArenaRecorder(i) for i in range(2)]
        result = _run_multi_arena(_multi_arena_cfg("dummy"), recorders)

        assert result.curriculum is None
        for recorder in recorders:
            opp_actions, view_calls = recorder.snapshot()
            assert opp_actions, f"arena {recorder.arena_id} collected nothing"
            assert all(a is None for a in opp_actions), (
                "an opp_action reached the wire on the stationary-dummy path"
            )
            assert view_calls == 0


class TestTheSingleArenaPathRefusesTheScriptedOpponent:
    """It steps no opponent policy, so it must refuse rather than ignore the config."""

    def test_train_vs_dummy_refuses_a_scripted_config(self):
        pytest.importorskip("torch")
        from agent.train import train_vs_dummy

        with pytest.raises(ValueError, match="never steps an opponent policy"):
            train_vs_dummy(
                _cfg(opponent="scripted"),
                transport_factory=lambda: pytest.fail(
                    "the refusal must happen before any transport is opened"
                ),
                max_episodes=1,
            )

    def test_the_cli_refuses_scripted_on_one_arena(self, capsys):
        from agent.train import main

        assert main(["--arenas", "1", "--opponent", "scripted"]) == 1
        assert "needs --arenas >1" in capsys.readouterr().err

    def test_the_cli_defaults_to_the_dummy(self):
        from agent.train import _build_parser

        assert _build_parser().parse_args([]).opponent == "dummy"

    def test_the_cli_stamps_the_opponent_onto_the_config(self):
        from agent.train import _build_parser

        args = _build_parser().parse_args(["--arenas", "4", "--opponent", "scripted"])
        cfg = dataclasses.replace(
            TrainConfig(), arenas=int(args.arenas), opponent=str(args.opponent)
        )
        assert cfg.opponent == "scripted"
