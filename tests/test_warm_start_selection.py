"""Tests for the warm-start retrain and checkpoint selection (T13).

Every test here exists because a specific, verified failure mode would otherwise
consume a whole overnight run and REPORT SUCCESS. Each one is written so that
reverting the fix makes it fail:

* **The eval scored a stationary opponent.** ``evaluate`` called ``env.step(action)``
  bare, so a run whose training opponent moves was evaluated against an opponent
  that stands still — an easy, different fight. With ``stop_on_pass`` defaulting
  True and eval every 1000 grad steps, a warm-started agent cleared that gate at
  its FIRST eval and the retrain stopped minutes in with ``stop_reason="passed_m2"``.
  Pinned below at both levels: ``evaluate`` itself, and the multi-arena loop that
  has to hand it an opponent.
* **The only checkpoint save sat inside the eval-improvement branch.** With
  ``--eval-every-grad-steps 0`` the run trained all night and saved NOTHING
  (``Trainer._fire_hooks`` / ``cfg.checkpoint_interval`` are dead on this path —
  the learner calls ``trainer.learn()`` directly). Pinned: periodic saves, a final
  save, and both with eval switched off entirely.
* **Selection keyed on recency.** Pinned: a later, worse eval must not replace the
  best checkpoint, and a run that never wins an eval episode must not ship its
  first eval's net in a file named "best".
* **``warm_start`` was declared, validated, and read by nothing**, with no ε
  restart to go with it. Pinned: online AND target initialized from the
  checkpoint, replay left fresh, and ε restarting in the 0.2-0.3 band rather than
  at the fresh-init 1.0 that would spend the budget acting at random.
* **AC18's last inch**: ``--opponent scripted`` must route
  ``dummy_knockback_immune=False`` into the launcher, or the retrain fights a bot
  that cannot be knocked back and cannot walk.

No socket, no live server, no Minecraft: the "bridge" is a generative fake and
every pad env is a fake.
"""

from __future__ import annotations

import dataclasses
import threading
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock

import numpy as np
import pytest

from agent.actions import Macro, N_ACTIONS
from agent.train_config import TrainConfig
from bridge.messages import ResetAckMsg, StateMsg
from env.observation_spec import OBS_DIM
from opponents.scripted_bot import OpponentView, ScriptedPreset


# Seconds a worker thread gets before a test calls it a stall.
_THREAD_TIMEOUT = 30.0

#: Shrunken net so a real gradient step is cheap; obs_dim / n_actions still
#: assert against the frozen contracts inside DuelingDRQN.
_TINY_NET = {"encoder_hidden": 16, "lstm_hidden": 16, "lstm_layers": 1}


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
# Fixtures: an omniscient view, a generative bridge, fake pads.
# ===========================================================================


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


class GenerativeBridge:
    """A fake ``BridgeTransport`` generating an endless valid state stream.

    Mirrors ``tests/test_integration_m2.GenerativeBridge``: an eval opens a fresh
    env over this transport for every eval and collects many episodes, so the
    stream cannot be a pre-scripted queue. Every outbound wire dict is recorded,
    which is how the ``opp_action`` assertions below read the wire directly
    rather than trusting a mock.
    """

    def __init__(self, *, opp_pos=(0.0, 64.0, 2.0), kill_step: int = 3) -> None:
        self.opp_pos = tuple(float(v) for v in opp_pos)
        self.kill_step = int(kill_step)
        self.sent: List[Dict[str, Any]] = []
        self.connects = 0
        self.closes = 0
        self.is_open = False
        self._episode_step = 0
        self._pending_reset_ack = False
        self._tick = 0
        self._lock = threading.Lock()

    # -- BridgeTransport protocol -----------------------------------------

    def connect(self) -> None:
        self.connects += 1
        self.is_open = True

    def close(self) -> None:
        self.closes += 1
        self.is_open = False

    def send(self, obj) -> None:
        msg = dict(obj)
        with self._lock:
            self.sent.append(msg)
        if msg.get("type") == "reset":
            self._pending_reset_ack = True
            self._episode_step = 0
        elif msg.get("type") == "step":
            self._episode_step += 1

    def recv(self):
        self._tick += 1
        if self._pending_reset_ack:
            self._pending_reset_ack = False
            return ResetAckMsg.from_dict(
                {
                    "type": "reset_ack",
                    "ok": True,
                    "readback": {"self_hp": 20.0, "opp_hp": 20.0},
                }
            )
        opponent_died = self._episode_step >= self.kill_step
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
                    "pos": list(self.opp_pos),
                    "yaw": 0.0,
                    "pitch": 0.0,
                    "velocity": [0.0, 0.0, 0.0],
                    "health": 20.0,
                },
                "events": {
                    "damage_dealt": 20.0 if opponent_died else 1.0,
                    "damage_taken": 0.0,
                    "i_died": False,
                    "opponent_died": bool(opponent_died),
                },
                "arena": {"wall_distances": [8.0, 8.0, 8.0, 8.0]},
                "tick": self._tick,
                "code_version": "test",
            }
        )

    # -- assertions helpers ------------------------------------------------

    def step_messages(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [m for m in self.sent if m.get("type") == "step"]


class ScriptedGreedyPolicy:
    """A torch-free ``GreedyPolicy`` returning a fixed action every step."""

    def __init__(self, action: int = int(Macro.ATTACK)) -> None:
        self._action = int(action)
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1

    def act(self, obs) -> int:
        return self._action


class RecordingOpponent:
    """An ``EvalOpponent`` that records the views it was handed."""

    name = "recording_opponent"

    def __init__(self, action: int = int(Macro.APPROACH)) -> None:
        self._action = int(action)
        self.views: List[Any] = []
        self.begin_calls = 0

    def begin_episode(self) -> None:
        self.begin_calls += 1

    def act(self, view) -> int:
        self.views.append(view)
        return self._action


class PadEnv:
    """Fake pad env for a multi-arena run, backed by a real fake transport.

    ``_transport`` is what ``_eval_via_designated_arena`` borrows: the eval builds
    a REAL ``MCPvPEnv`` over it with ``auto_connect=False``, which is exactly the
    single-connection protocol the live run uses. Collection itself stays fake and
    cheap.
    """

    def __init__(self, transport: GenerativeBridge, k: int = 4) -> None:
        self._transport = transport
        self.k = int(k)
        self._t = 0
        self._rng = np.random.default_rng(0)
        self.opp_actions: List[Optional[int]] = []

    def reset(self, seed: Optional[int] = None):
        self._rng = np.random.default_rng(0 if seed is None else int(seed))
        self._t = 0
        return self._obs()

    def raw_opponent_view(self) -> OpponentView:
        return _view()

    def step(self, action: int, opp_action: Optional[int] = None):
        self.opp_actions.append(opp_action)
        self._t += 1
        done = self._t >= self.k
        # The agent never wins during COLLECTION, so the curriculum gate never
        # fires and this doubles as the AC10 "gate never fires" case.
        return self._obs(), 0.0, done, {"won": False, "lost": done, "timeout": False}

    def close(self) -> None:
        pass

    def _obs(self):
        return self._rng.uniform(-1.0, 1.0, size=OBS_DIM).astype(np.float32)


def _multi_cfg(**overrides: Any) -> TrainConfig:
    """Minimal multi-arena config: tiny windows, instant warm-up, no relaunches."""
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
    )
    base.update(overrides)
    return dataclasses.replace(TrainConfig(), **base)


def _run_multi_arena(cfg: TrainConfig, pads: List[PadEnv], **kwargs: Any):
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
        # SHORT on purpose. `Collector._wait_while_paused` sets its paused-idle
        # flag once, BEFORE its wait loop, so a resume() immediately followed by
        # a new pause() (which this fake fleet produces: tiny nets make the
        # grad-step budget between evals elapse in milliseconds) leaves the
        # collector in the loop with the flag cleared, and the next eval waits out
        # the whole pause timeout before skipping. That is a pre-existing race in
        # distributed/actor.py, not something these tests are pinning; a short
        # timeout keeps a lost race a skipped eval instead of a 2-minute stall.
        # The FIRST eval cannot lose it (no resume precedes it), which is the one
        # every assertion below depends on.
        eval_pause_timeout=1.0,
        watchdog=LearnerWatchdog(patience=500, interval_s=1.0),
    )
    call.update(kwargs)
    return _completes_within(lambda: train_multi_arena(cfg, **call))


def _fake_eval_report(win_rate: float, *, passed_m2: bool = False) -> MagicMock:
    """A canned ``EvalReport`` with the fields the driver loop formats/reads."""
    report = MagicMock()
    report.win_rate = float(win_rate)
    report.n_episodes = 1
    report.mean_episode_length = 10.0
    report.aim_while_invisible = 0.0
    report.passed_m2 = bool(passed_m2)
    report.opponent = "scripted_mixed"
    return report


def _canned_evals(
    monkeypatch,
    win_rates: List[float],
    seen: List[float],
    *,
    passed_m2: bool = False,
) -> None:
    """Stub the eval HANDOFF with a canned win-rate sequence.

    Returns a real ``_EvalOutcome`` — report PLUS the weight snapshot and the
    grad step the eval was taken at — because that tuple, not a bare report, is
    what the driver loop selects and saves on.
    """
    import agent.train as train_module
    from distributed.weights import clone_state_dict

    def _stub(**kwargs):
        rate = win_rates[min(len(seen), len(win_rates) - 1)]
        seen.append(rate)
        trainer = kwargs["trainer"]
        return train_module._EvalOutcome(
            report=_fake_eval_report(rate, passed_m2=passed_m2),
            weights=clone_state_dict(trainer.online.state_dict()),
            grad_step=int(trainer.grad_step),
        )

    monkeypatch.setattr(train_module, "_eval_via_designated_arena", _stub)


def _write_checkpoint(tmp_path, *, payload_key: Optional[str] = "model"):
    """Save a DuelingDRQN checkpoint with DISTINCTIVE (non-default) weights."""
    import torch

    from agent.dqn import DuelingDRQN

    net = DuelingDRQN(**_TINY_NET)
    with torch.no_grad():
        for param in net.parameters():
            # Deterministic, and far from any fresh init, so "the load happened"
            # cannot be confused with "two fresh inits happened to match".
            param.copy_(torch.full_like(param, 0.125))
    state_dict = {k: v.clone() for k, v in net.state_dict().items()}

    path = tmp_path / "warm.pt"
    if payload_key is None:
        torch.save(state_dict, path)
    else:
        torch.save(
            {payload_key: state_dict, "grad_step": 4_242, "code_version": "test"}, path
        )
    return str(path), state_dict


# ===========================================================================
# BLOCKER 2 — warm_start is declared, validated, and read by nothing.
# ===========================================================================


class TestWarmStartWeights:
    """TC24: the loaded params must equal the checkpoint's, exactly."""

    def test_the_online_net_is_initialized_from_the_checkpoint(self, tmp_path):
        pytest.importorskip("torch")
        import torch

        from agent.train import Trainer

        path, state_dict = _write_checkpoint(tmp_path)
        trainer = Trainer(
            _multi_cfg(warm_start=path), net_kwargs=dict(_TINY_NET)
        )

        loaded = trainer.online.state_dict()
        assert set(loaded) == set(state_dict)
        for key, expected in state_dict.items():
            assert torch.equal(loaded[key], expected), f"{key} was not loaded"

    def test_the_target_net_is_initialized_from_the_checkpoint_too(self, tmp_path):
        # A target left at its random init bootstraps the warm-started online net
        # toward noise -- the same "warm start thrown away" failure as eps=1.0.
        pytest.importorskip("torch")
        import torch

        from agent.train import Trainer

        path, state_dict = _write_checkpoint(tmp_path)
        trainer = Trainer(_multi_cfg(warm_start=path), net_kwargs=dict(_TINY_NET))

        target = trainer.target.state_dict()
        for key, expected in state_dict.items():
            assert torch.equal(target[key], expected), (
                f"target {key} was not initialized from the warm-start checkpoint"
            )

    def test_without_warm_start_the_nets_are_freshly_initialized(self, tmp_path):
        # Negative control: the equality above must come from the LOAD, not from
        # every DuelingDRQN happening to start at 0.125.
        pytest.importorskip("torch")
        import torch

        from agent.train import Trainer

        _path, state_dict = _write_checkpoint(tmp_path)
        trainer = Trainer(_multi_cfg(), net_kwargs=dict(_TINY_NET))

        online = trainer.online.state_dict()
        assert any(
            not torch.equal(online[key], expected)
            for key, expected in state_dict.items()
        ), "a fresh init must not equal the checkpoint"

    def test_the_replay_buffer_stays_fresh(self, tmp_path):
        # The pinned regime: weights are reused, DATA is not. The stored
        # transitions came from a different reward regime and a stationary
        # opponent.
        pytest.importorskip("torch")
        from agent.train import Trainer

        path, _ = _write_checkpoint(tmp_path)
        trainer = Trainer(_multi_cfg(warm_start=path), net_kwargs=dict(_TINY_NET))

        assert len(trainer.replay) == 0

    def test_a_missing_checkpoint_fails_immediately_and_names_the_path(self, tmp_path):
        pytest.importorskip("torch")
        from agent.train import Trainer

        missing = str(tmp_path / "nope.pt")
        with pytest.raises(FileNotFoundError, match="nope.pt"):
            Trainer(_multi_cfg(warm_start=missing), net_kwargs=dict(_TINY_NET))


class TestWarmWeightsReachTheCollectorsBeforeTheyStart:
    """The warm weights must be published BEFORE ``pool.start()``, or they miss.

    ``train_multi_arena`` publishes version 0 by hand right before starting the
    pool. Delete that line (or move it after ``pool.start()``) and the collectors
    open their first episodes on a RANDOMLY-INITIALIZED net -- the learner's own
    version-0 publish lands later, because its thread starts after the pool -- so
    the one run whose whole purpose is not starting from scratch feeds noise into
    a deliberately fresh replay. Nothing in any log says so.
    """

    def _run_with_event_log(self, cfg, pads, events):
        """Drive a run recording (publish, version) and pool-start, in order."""
        from distributed import actor as actor_module
        from distributed.weights import WeightStore, clone_state_dict

        class _SpyStore(WeightStore):
            def publish(self, state_dict, version):
                # Clone AT RECORD TIME: state_dict holds views into the live net
                # and the learner mutates them for the rest of the run, so a
                # deferred comparison against the raw reference would flake.
                events.append(("publish", int(version), clone_state_dict(state_dict)))
                super().publish(state_dict, version)

        original_start = actor_module.ActorPool.start

        def _spy_start(pool_self):
            events.append(("pool_start", -1, None))
            return original_start(pool_self)

        actor_module.ActorPool.start = _spy_start
        try:
            return _run_multi_arena(
                cfg,
                pads,
                weight_store=_SpyStore(),
                max_grad_steps=3,
                eval_every_grad_steps=0,
            )
        finally:
            actor_module.ActorPool.start = original_start

    def test_the_warm_weights_are_published_before_the_pool_starts(self, tmp_path):
        pytest.importorskip("torch")
        import torch

        path, warm_state = _write_checkpoint(tmp_path)
        events: List[Tuple[str, int, Any]] = []
        pads = [PadEnv(GenerativeBridge(), k=4) for _ in range(2)]

        self._run_with_event_log(_multi_cfg(warm_start=path), pads, events)

        assert events, "nothing was recorded; the run never got going"
        kind, version, published = events[0]
        assert (kind, version) == ("publish", 0), (
            "the collectors were started before any weights were published: "
            f"first event was {events[0][:2]}. Every arena rolls its opening "
            "episodes on a randomly-initialized net and the warm start is "
            "silently thrown away for them."
        )
        # ...and they are the WARM weights, not some other net.
        assert set(published) == set(warm_state)
        assert all(
            torch.equal(published[key], warm_state[key].cpu()) for key in warm_state
        ), "a version-0 snapshot was published, but it was not the warm start"

    def test_a_fresh_run_publishes_nothing_before_the_pool(self):
        # The hand publish is warm-start-only: a fresh run has nothing to hand
        # over, and the learner's own publish is the first one.
        pytest.importorskip("torch")

        events: List[Tuple[str, int, Any]] = []
        pads = [PadEnv(GenerativeBridge(), k=4) for _ in range(2)]

        self._run_with_event_log(_multi_cfg(), pads, events)

        assert events[0][0] == "pool_start"


class TestWarmStartCheckpointLoader:
    """Be liberal in what a checkpoint may look like, loud when it holds nothing."""

    def test_a_raw_state_dict_is_accepted(self, tmp_path):
        pytest.importorskip("torch")
        import torch

        from agent.train import load_checkpoint_state_dict

        path, state_dict = _write_checkpoint(tmp_path, payload_key=None)
        loaded = load_checkpoint_state_dict(path)

        assert set(loaded) == set(state_dict)
        assert all(torch.equal(loaded[k], v) for k, v in state_dict.items())

    @pytest.mark.parametrize(
        "key", ["model", "model_state_dict", "state_dict", "online"]
    )
    def test_every_documented_wrapper_key_is_accepted(self, tmp_path, key):
        pytest.importorskip("torch")
        from agent.train import load_checkpoint_state_dict

        path, state_dict = _write_checkpoint(tmp_path, payload_key=key)

        assert set(load_checkpoint_state_dict(path)) == set(state_dict)

    def test_a_payload_with_no_weights_is_refused(self, tmp_path):
        # Silence here would mean a "warm start" that loaded nothing and looked
        # identical to one that worked.
        pytest.importorskip("torch")
        import torch

        from agent.train import load_checkpoint_state_dict

        path = tmp_path / "empty.pt"
        torch.save({"grad_step": 7, "code_version": "test"}, path)

        with pytest.raises(ValueError, match="carries no network weights"):
            load_checkpoint_state_dict(str(path))

    def test_a_non_mapping_payload_is_refused(self, tmp_path):
        pytest.importorskip("torch")
        import torch

        from agent.train import load_checkpoint_state_dict

        path = tmp_path / "list.pt"
        torch.save([1, 2, 3], path)

        with pytest.raises(ValueError, match="not a state_dict"):
            load_checkpoint_state_dict(str(path))


class TestWarmStartEpsilonRestart:
    """AC17: eps_start is lowered when warm_start is set, or the warm start is lost."""

    def test_epsilon_restarts_in_the_pinned_band(self):
        from agent.train import epsilon_for_episode

        cfg = _multi_cfg(warm_start="runs/m2_multi.pt")

        first = epsilon_for_episode(0, cfg)
        assert first == pytest.approx(cfg.warm_start_eps_start)
        assert 0.2 <= first <= 0.3, (
            "a warm start under the fresh-init eps_start=1.0 spends the decay "
            "window acting at random and throws the checkpoint away"
        )

    def test_a_fresh_run_keeps_the_full_exploration_schedule(self):
        # Negative control: nothing about the ε schedule changes without a warm
        # start.
        from agent.train import epsilon_for_episode

        cfg = _multi_cfg()

        assert epsilon_for_episode(0, cfg) == pytest.approx(cfg.eps_start)
        assert cfg.eps_start == 1.0

    def test_the_schedule_still_decays_to_the_floor(self):
        from agent.train import epsilon_for_episode

        cfg = _multi_cfg(warm_start="runs/m2_multi.pt", eps_decay_episodes=10)

        mid = epsilon_for_episode(5, cfg)
        assert cfg.eps_end < mid < cfg.warm_start_eps_start
        assert epsilon_for_episode(10, cfg) == pytest.approx(cfg.eps_end)
        assert epsilon_for_episode(10_000, cfg) == pytest.approx(cfg.eps_end)

    def test_the_collectors_share_this_exact_schedule_function(self):
        # All N arenas compute ε through the same import; if a copy were ever
        # made in distributed.actor, the warm-start restart would apply to the
        # learner's log line and to nothing that actually acts.
        import distributed.actor as actor

        from agent.train import epsilon_for_episode

        assert actor.epsilon_for_episode is epsilon_for_episode

    def test_the_trainer_reports_the_effective_start(self, tmp_path):
        pytest.importorskip("torch")
        from agent.train import Trainer

        path, _ = _write_checkpoint(tmp_path)
        trainer = Trainer(_multi_cfg(warm_start=path), net_kwargs=dict(_TINY_NET))

        assert trainer.last_epsilon == pytest.approx(
            trainer.cfg.warm_start_eps_start
        )


class TestWarmStartConfigValidation:
    """A misconfigured ε restart must fail at construction, not at 3am."""

    def test_an_out_of_range_restart_is_refused(self):
        with pytest.raises(ValueError, match="warm_start_eps_start must be in"):
            dataclasses.replace(TrainConfig(), warm_start_eps_start=1.5)

    def test_a_restart_below_the_floor_is_refused(self):
        with pytest.raises(ValueError, match="must be >= eps_end"):
            dataclasses.replace(
                TrainConfig(), eps_end=0.3, warm_start_eps_start=0.1
            )

    def test_the_default_sits_in_the_planned_band(self):
        assert 0.2 <= TrainConfig().warm_start_eps_start <= 0.3

    def test_an_unknown_eval_opponent_preset_is_refused(self):
        with pytest.raises(ValueError, match="eval_opponent_preset must be one of"):
            dataclasses.replace(TrainConfig(), eval_opponent_preset="medium")


# ===========================================================================
# BLOCKER 1a — the eval must fight the same opponent training fights.
# ===========================================================================


class TestEvaluateStepsAnOpponent:
    """``evaluate`` puts an opp_action on the wire once per decision, or nothing."""

    def _env(self, transport):
        from env.mc_pvp_env import MCPvPEnv

        return MCPvPEnv(transport=transport, max_episode_steps=32)

    def test_every_step_carries_an_opp_action(self):
        from eval.evaluate import evaluate

        transport = GenerativeBridge(kill_step=3)
        opponent = RecordingOpponent()

        report = evaluate(
            self._env(transport),
            ScriptedGreedyPolicy(),
            n_episodes=2,
            timeout_cap=32,
            opponent=opponent,
        )

        steps = transport.step_messages()
        assert steps, "the eval took no steps at all"
        assert all("opp_action" in m for m in steps), (
            "an eval step reached the wire with no opp_action: the eval is "
            "scoring a STATIONARY opponent while training fights a moving one"
        )
        assert all(0 <= int(m["opp_action"]) < N_ACTIONS for m in steps)
        # One raw view per step -- the decision-window invariant the opponent's
        # shadow attack meter depends on.
        assert len(opponent.views) == len(steps)
        assert opponent.begin_calls == report.n_episodes == 2

    def test_the_opponents_view_reaches_it_unperturbed(self):
        # attack_cooldown is clamped to EXACTLY 1.0 by the producer and compared
        # against >= 1.0 - 1e-6; a value a hair under makes a scripted bot never
        # attack, which looks passive rather than broken.
        from eval.evaluate import evaluate

        transport = GenerativeBridge(kill_step=2)
        opponent = RecordingOpponent()

        evaluate(
            self._env(transport),
            ScriptedGreedyPolicy(),
            n_episodes=1,
            timeout_cap=32,
            opponent=opponent,
        )

        assert opponent.views
        assert all(v.attack_cooldown == 1.0 for v in opponent.views)
        assert all(isinstance(v, OpponentView) for v in opponent.views)

    def test_without_an_opponent_the_m2_wire_line_is_unchanged(self):
        # The M2 regression guard: no opponent means the field never appears.
        from eval.evaluate import evaluate

        transport = GenerativeBridge(kill_step=3)

        report = evaluate(
            self._env(transport),
            ScriptedGreedyPolicy(),
            n_episodes=2,
            timeout_cap=32,
        )

        steps = transport.step_messages()
        assert steps
        assert all("opp_action" not in m for m in steps)
        assert report.opponent == "dummy"

    def test_the_report_records_who_was_fought(self):
        from eval.evaluate import evaluate

        transport = GenerativeBridge(kill_step=2)

        report = evaluate(
            self._env(transport),
            ScriptedGreedyPolicy(),
            n_episodes=1,
            timeout_cap=32,
            opponent=RecordingOpponent(),
        )

        assert report.opponent == "recording_opponent"
        assert report.to_dict()["opponent"] == "recording_opponent"
        # passed_m2 is AC6's gate against the DUMMY; scored against anything else
        # the artifact has to say so.
        assert any("NOT the stationary dummy" in note for note in report.notes)


class TestEvalOpponentDriver:
    """The eval's opponent must be identical at every eval, or nothing compares."""

    #: Out of attack range but visible, so ``ScriptedBot.act`` takes the movement
    #: draw on EVERY step. The in-range fixture ``_view()`` short-circuits to
    #: ATTACK before touching the RNG at all, which is half of why the reseed
    #: mutant used to survive: the compared sequences were constants.
    def _drawing_view(self):
        return _view(
            in_attack_range=False,
            distance=8.0,
            target_pos=(0.0, 64.0, 8.0),
            last_known_target_pos=(0.0, 64.0, 8.0),
        )

    def _episode_sequences(self, driver, step_counts):
        """Return one macro list PER EPISODE (lengths differ between runs)."""
        episodes: List[List[Tuple[str, int]]] = []
        for n_steps in step_counts:
            driver.begin_episode()
            episodes.append(
                [
                    (driver.preset.value, int(driver.act(self._drawing_view())))
                    for _ in range(n_steps)
                ]
            )
        return episodes

    def test_two_evals_face_an_identical_opponent(self):
        # The per-episode reseed exists for exactly one condition: eval episodes
        # end on a DEATH whose timing depends on the agent, so two evals of the
        # same run consume different amounts of opponent RNG per episode. Without
        # the reseed, episode i's opponent depends on how long episodes 0..i-1
        # ran, and the win-rate series that SELECTS the shipped checkpoint
        # compares different fights. Equal-length runs cannot see that at all --
        # the streams line up either way -- so these two runs are deliberately
        # ragged.
        from agent.train import build_eval_opponent

        factory = build_eval_opponent(_multi_cfg(opponent="scripted"))
        assert factory is not None

        ragged = self._episode_sequences(factory(), [3, 9, 5, 11])
        uniform = self._episode_sequences(factory(), [14, 14, 14, 14])

        # Non-constant, or the comparison below would hold for any bot at all.
        assert len({macro for episode in uniform for _tier, macro in episode}) > 1

        for index, (short, long) in enumerate(zip(ragged, uniform)):
            assert short == long[: len(short)], (
                f"eval episode {index} faced a different opponent once the "
                "earlier episodes had different lengths: the per-episode reseed "
                "is gone, and the win rate a checkpoint is selected on now "
                "depends on how long the AGENT survived earlier episodes"
            )

    def test_mixed_alternates_easy_and_hard_by_episode_index(self):
        from agent.train import EvalOpponentDriver

        driver = EvalOpponentDriver(
            _multi_cfg(opponent="scripted"), base_seed=7, preset_choice="mixed"
        )

        presets = []
        for _ in range(4):
            driver.begin_episode()
            presets.append(driver.preset)

        assert presets == [
            ScriptedPreset.EASY,
            ScriptedPreset.HARD,
            ScriptedPreset.EASY,
            ScriptedPreset.HARD,
        ]
        assert driver.name == "scripted_mixed"

    @pytest.mark.parametrize(
        "choice,preset",
        [("easy", ScriptedPreset.EASY), ("hard", ScriptedPreset.HARD)],
    )
    def test_a_pinned_tier_never_alternates(self, choice, preset):
        from agent.train import EvalOpponentDriver

        driver = EvalOpponentDriver(
            _multi_cfg(opponent="scripted"), base_seed=7, preset_choice=choice
        )

        for _ in range(4):
            driver.begin_episode()
            assert driver.preset is preset
        assert driver.name == f"scripted_{choice}"

    def test_the_dummy_path_builds_no_eval_opponent(self):
        from agent.train import build_eval_opponent

        assert build_eval_opponent(_multi_cfg(opponent="dummy")) is None

    def test_the_eval_seed_band_belongs_to_no_arena(self):
        # The eval opponent's RNG must not coincide with any collector's.
        from agent.train import arena_episode_seed, opponent_seed

        cfg = _multi_cfg(opponent="scripted", arenas=4)
        eval_seed = opponent_seed(cfg, arena_id=cfg.arenas, role="eval")
        arena_seeds = {
            opponent_seed(cfg, arena, role)
            for arena in range(cfg.arenas)
            for role in ("mixture", "easy", "hard", "eval")
        }
        arena_seeds |= {
            arena_episode_seed(cfg, arena, ep)
            for arena in range(cfg.arenas)
            for ep in range(64)
        }

        assert eval_seed not in arena_seeds

    def test_an_unknown_preset_choice_is_refused(self):
        from agent.train import EvalOpponentDriver

        with pytest.raises(ValueError, match="must be 'mixed', 'easy' or 'hard'"):
            EvalOpponentDriver(
                _multi_cfg(opponent="scripted"), base_seed=1, preset_choice="medium"
            )


class TestTheMultiArenaEvalFightsTheScriptedOpponent:
    """The wiring blocker itself, end to end through ``train_multi_arena``."""

    def test_the_eval_puts_an_opp_action_on_the_designated_arenas_wire(self):
        pytest.importorskip("torch")

        transports = [GenerativeBridge(kill_step=3) for _ in range(2)]
        pads = [PadEnv(t, k=4) for t in transports]

        result = _run_multi_arena(
            _multi_cfg(opponent="scripted"),
            pads,
            eval_every_grad_steps=3,
            eval_episodes=1,
            max_grad_steps=9,
        )

        assert result.reports, "no eval ran; this test proves nothing"
        # Eval borrows arena 0's connection (and only that one).
        eval_steps = transports[0].step_messages()
        assert eval_steps, "the eval never stepped the borrowed env"
        assert all("opp_action" in m for m in eval_steps), (
            "the multi-arena eval stepped the env with no opp_action: under "
            "--opponent scripted it is scoring a stationary opponent, and the "
            "checkpoint it selects is selected against the wrong fight"
        )
        assert result.eval_opponent == "scripted_mixed"
        assert all(r.opponent == "scripted_mixed" for r in result.reports)
        # Arena 1 keeps collecting; the eval never touched its bridge.
        assert transports[1].step_messages() == []

    def test_the_dummy_path_eval_stays_on_the_m2_wire_line(self):
        pytest.importorskip("torch")

        transports = [GenerativeBridge(kill_step=3) for _ in range(2)]
        pads = [PadEnv(t, k=4) for t in transports]

        result = _run_multi_arena(
            _multi_cfg(opponent="dummy"),
            pads,
            eval_every_grad_steps=3,
            eval_episodes=1,
            max_grad_steps=9,
        )

        assert result.reports
        eval_steps = transports[0].step_messages()
        assert eval_steps
        assert all("opp_action" not in m for m in eval_steps)
        assert result.eval_opponent == "dummy"


# ===========================================================================
# BLOCKER 1b — the multi-arena path must save checkpoints without eval.
# ===========================================================================


class TestMultiArenaCheckpointsAreIndependentOfEval:
    """`--eval-every-grad-steps 0` used to train all night and save NOTHING."""

    def test_checkpoints_are_saved_with_eval_disabled(self):
        pytest.importorskip("torch")

        saves: List[int] = []
        pads = [PadEnv(GenerativeBridge(), k=4) for _ in range(2)]

        result = _run_multi_arena(
            _multi_cfg(),
            pads,
            # The blocker's exact configuration.
            eval_every_grad_steps=0,
            checkpoint_every_grad_steps=5,
            max_grad_steps=20,
            checkpoint_hook=lambda _t, step: saves.append(int(step)),
        )

        assert not result.reports, "eval was supposed to be disabled"
        assert len(saves) >= 2, (
            f"a run with eval disabled saved {len(saves)} checkpoint(s); the "
            "overnight run would finish with nothing on disk"
        )
        assert result.checkpoints_saved == len(saves)
        assert saves == sorted(saves)

    def test_a_final_checkpoint_is_saved_even_with_the_periodic_save_off(self):
        pytest.importorskip("torch")

        saves: List[int] = []
        pads = [PadEnv(GenerativeBridge(), k=4) for _ in range(2)]

        result = _run_multi_arena(
            _multi_cfg(),
            pads,
            eval_every_grad_steps=0,
            checkpoint_every_grad_steps=0,
            max_grad_steps=10,
            checkpoint_hook=lambda _t, step: saves.append(int(step)),
        )

        assert len(saves) == 1, "the final save is the last line of defence"
        assert saves[0] >= result.grad_steps - 1

    def test_the_cadence_falls_back_to_the_configs_checkpoint_interval(self):
        # cfg.checkpoint_interval was dead on this path (the learner calls
        # trainer.learn() directly, so Trainer._fire_hooks never runs).
        pytest.importorskip("torch")

        saves: List[int] = []
        pads = [PadEnv(GenerativeBridge(), k=4) for _ in range(2)]

        _run_multi_arena(
            _multi_cfg(checkpoint_interval=5),
            pads,
            eval_every_grad_steps=0,
            checkpoint_every_grad_steps=None,  # -> cfg.checkpoint_interval
            max_grad_steps=20,
            checkpoint_hook=lambda _t, step: saves.append(int(step)),
        )

        assert len(saves) >= 2

    def test_a_failing_checkpoint_hook_never_kills_the_run(self):
        pytest.importorskip("torch")

        pads = [PadEnv(GenerativeBridge(), k=4) for _ in range(2)]

        def _explode(_trainer, _step):
            raise OSError("disk full")

        result = _run_multi_arena(
            _multi_cfg(),
            pads,
            eval_every_grad_steps=0,
            checkpoint_every_grad_steps=5,
            max_grad_steps=15,
            checkpoint_hook=_explode,
        )

        assert result.grad_steps >= 15
        assert result.checkpoints_saved == 0


# ===========================================================================
# BLOCKER 3 — selection by win rate, not recency.
# ===========================================================================


class TestBestCheckpointSelector:
    """TC25: the highest win rate wins, and a later worse one does not."""

    def test_an_improvement_is_selected(self):
        from agent.train import _BestCheckpointSelector

        selector = _BestCheckpointSelector()

        assert selector.consider(0.4, 100) is True
        assert selector.best_win_rate == pytest.approx(0.4)
        assert selector.best_grad_step == 100

    def test_a_later_worse_eval_is_not_selected(self):
        from agent.train import _BestCheckpointSelector

        selector = _BestCheckpointSelector()
        selector.consider(0.8, 100)

        assert selector.consider(0.1, 200) is False
        assert selector.best_grad_step == 100, "selection fell back to recency"

    def test_a_tie_keeps_the_earlier_checkpoint(self):
        from agent.train import _BestCheckpointSelector

        selector = _BestCheckpointSelector()
        selector.consider(0.5, 100)

        assert selector.consider(0.5, 200) is False
        assert selector.best_grad_step == 100

    def test_a_run_that_never_wins_ships_no_best_checkpoint(self):
        # Every eval at 0.0 beats the -1.0 initial best; without the zero guard
        # the FIRST eval's barely-trained net would be shipped as "best".
        from agent.train import _BestCheckpointSelector

        selector = _BestCheckpointSelector()

        assert selector.consider(0.0, 10) is False
        assert selector.consider(0.0, 20) is False
        assert selector.best_grad_step == -1

    def test_the_first_real_win_after_zeros_is_selected(self):
        from agent.train import _BestCheckpointSelector

        selector = _BestCheckpointSelector()
        selector.consider(0.0, 10)

        assert selector.consider(0.2, 20) is True
        assert selector.best_grad_step == 20


class TestSelectionInTheMultiArenaLoop:
    """The selector, wired: the best hook must track the best win rate.

    The eval HANDOFF is stubbed here (not the eval itself): what these tests pin
    is which reports produce a save, and a canned report sequence is the only way
    to state "0.75 then 0.10" as a fact rather than hope a fake env produces it.
    The handoff proper — pause, borrow the one connection, step the opponent — is
    pinned by :class:`TestTheMultiArenaEvalFightsTheScriptedOpponent` above.
    """

    def _fake_report(self, win_rate: float):
        return _fake_eval_report(win_rate)

    def _canned_evals(self, monkeypatch, win_rates: List[float], seen: List[float]):
        _canned_evals(monkeypatch, win_rates, seen)

    def test_the_best_hook_fires_only_on_improvement(self, monkeypatch):
        pytest.importorskip("torch")

        seen: List[float] = []
        best_saves: List[Tuple[int, Dict[str, Any]]] = []
        latest_saves: List[int] = []
        # The last value repeats once the list is exhausted, so extra evals can
        # only ever TIE the high-water mark -- never add a save.
        self._canned_evals(monkeypatch, [0.25, 0.75, 0.10, 0.75], seen)

        pads = [PadEnv(GenerativeBridge(), k=4) for _ in range(2)]
        result = _run_multi_arena(
            _multi_cfg(opponent="scripted"),
            pads,
            eval_every_grad_steps=3,
            eval_episodes=1,
            max_grad_steps=15,
            checkpoint_every_grad_steps=0,
            checkpoint_hook=lambda _t, step: latest_saves.append(int(step)),
            best_checkpoint_hook=lambda _t, step, meta, _w: best_saves.append(
                (int(step), dict(meta))
            ),
        )

        assert len(seen) >= 3, f"only {len(seen)} evals ran; the test proves little"
        saved_rates = [meta["win_rate"] for _step, meta in best_saves]
        assert saved_rates == [0.25, 0.75], (
            f"best-checkpoint saves keyed on the wrong thing: {saved_rates}. A "
            "0.10 after a 0.75 must not replace the shipped checkpoint."
        )
        assert result.best_win_rate == pytest.approx(0.75)
        assert result.best_grad_step == best_saves[-1][0]
        assert result.eval_opponent == "scripted_mixed"
        # The LATEST hook is a separate file on its own schedule (here the final
        # save only), so a later worse net can never overwrite the best one.
        assert latest_saves, "the final save must still happen"

    def test_the_best_checkpoint_records_what_it_scored(self, monkeypatch):
        pytest.importorskip("torch")

        best_saves: List[Dict[str, Any]] = []
        self._canned_evals(monkeypatch, [0.5], [])

        pads = [PadEnv(GenerativeBridge(), k=4) for _ in range(2)]
        _run_multi_arena(
            _multi_cfg(opponent="scripted"),
            pads,
            eval_every_grad_steps=3,
            eval_episodes=1,
            max_grad_steps=9,
            best_checkpoint_hook=lambda _t, _s, meta, _w: best_saves.append(
                dict(meta)
            ),
        )

        assert best_saves
        meta = best_saves[0]
        assert meta["win_rate"] == pytest.approx(0.5)
        assert meta["eval_opponent"] == "scripted_mixed"
        assert "eval_episodes" in meta and "passed_m2" in meta

    def test_a_run_that_never_wins_saves_no_best_checkpoint(self, monkeypatch):
        # ...but must still leave the periodic/final checkpoint behind.
        pytest.importorskip("torch")

        best_saves: List[int] = []
        latest_saves: List[int] = []
        self._canned_evals(monkeypatch, [0.0], [])

        pads = [PadEnv(GenerativeBridge(), k=4) for _ in range(2)]
        result = _run_multi_arena(
            _multi_cfg(opponent="scripted"),
            pads,
            eval_every_grad_steps=3,
            eval_episodes=1,
            max_grad_steps=9,
            checkpoint_hook=lambda _t, step: latest_saves.append(int(step)),
            best_checkpoint_hook=lambda _t, step, _m, _w: best_saves.append(int(step)),
        )

        assert best_saves == [], (
            "a run whose agent never won an eval episode shipped its first eval's "
            "net in a file named 'best'"
        )
        assert result.best_grad_step == -1
        assert latest_saves, "the final save is what such a run has to fall back on"


# ===========================================================================
# The gate VERDICT and the stop DECISION are two different things.
# ===========================================================================


class TestTheGateVerdictIsIndependentOfStopOnPass:
    """`passed_m2` must report the gate, not whether the loop chose to stop.

    ``passed`` used to be set only inside ``if report.passed_m2 and
    stop_on_pass:``. T13 made ``stop_on_pass`` default False under ``--opponent
    scripted``, which made that branch unreachable, pinned
    ``MultiArenaResult.passed_m2`` at False forever, and left
    ``_main_multi_arena`` returning ``0 if passed_m2 else 1`` -- so every
    successful scripted run trained all night, shipped a good checkpoint, and
    exited 1.
    """

    def test_a_passing_run_that_keeps_training_still_reports_passed(
        self, monkeypatch
    ):
        pytest.importorskip("torch")

        seen: List[float] = []
        _canned_evals(monkeypatch, [1.0], seen, passed_m2=True)

        pads = [PadEnv(GenerativeBridge(), k=4) for _ in range(2)]
        result = _run_multi_arena(
            _multi_cfg(opponent="scripted"),
            pads,
            eval_every_grad_steps=3,
            eval_episodes=1,
            max_grad_steps=12,
            # The scripted default (T13): clearing the M2 gate is not the
            # milestone for a moving opponent, so the run keeps going.
            stop_on_pass=False,
        )

        assert len(seen) >= 2, (
            f"only {len(seen)} eval(s) ran; this test has to prove the run kept "
            "training AFTER a passing eval"
        )
        assert result.passed_m2 is True, (
            "a run that cleared the M2 gate reported passed_m2=False because it "
            "was configured not to stop -- the CLI turns this into exit code 1"
        )
        assert result.stop_reason == "max_grad_steps", (
            "the loop stopped on the gate despite stop_on_pass=False"
        )

    def test_stop_on_pass_still_stops_and_still_reports_passed(self, monkeypatch):
        pytest.importorskip("torch")

        seen: List[float] = []
        _canned_evals(monkeypatch, [1.0], seen, passed_m2=True)

        pads = [PadEnv(GenerativeBridge(), k=4) for _ in range(2)]
        result = _run_multi_arena(
            _multi_cfg(opponent="scripted"),
            pads,
            eval_every_grad_steps=3,
            eval_episodes=1,
            max_grad_steps=1_000_000,  # only the gate can end this run
            stop_on_pass=True,
        )

        assert result.passed_m2 is True
        assert result.stop_reason == "passed_m2"
        assert len(seen) == 1, "stop_on_pass=True must break at the FIRST pass"

    def test_a_failing_run_still_reports_not_passed(self, monkeypatch):
        pytest.importorskip("torch")

        seen: List[float] = []
        _canned_evals(monkeypatch, [0.4], seen, passed_m2=False)

        pads = [PadEnv(GenerativeBridge(), k=4) for _ in range(2)]
        result = _run_multi_arena(
            _multi_cfg(opponent="scripted"),
            pads,
            eval_every_grad_steps=3,
            eval_episodes=1,
            max_grad_steps=12,
            stop_on_pass=False,
        )

        assert seen, "no eval ran; this test proves nothing"
        assert result.passed_m2 is False


# ===========================================================================
# The startup warnings: a run that cannot save must say so BEFORE it trains.
# ===========================================================================


class TestTheStartupWarningNamesTheMissingFlag:
    """`--best-checkpoint` alone silently restored the save-nothing blocker.

    The warning was guarded on BOTH hooks being None, but only
    ``checkpoint_hook`` drives ``_save_latest`` -- which is the periodic AND the
    final save. So ``--best-checkpoint`` on its own suppressed the warning,
    disabled the periodic save, made the final save a no-op, and left the run's
    only write behind a strictly-improving eval win rate above zero.
    """

    def _warnings_for(self, **hooks) -> List[str]:
        lines: List[str] = []
        pads = [PadEnv(GenerativeBridge(), k=4) for _ in range(2)]
        _run_multi_arena(
            _multi_cfg(),
            pads,
            max_grad_steps=3,
            eval_every_grad_steps=0,
            log=lines.append,
            **hooks,
        )
        return [line for line in lines if "WARNING" in line]

    def test_best_checkpoint_without_checkpoint_warns_and_names_checkpoint(self):
        pytest.importorskip("torch")

        warnings = self._warnings_for(
            checkpoint_hook=None,
            best_checkpoint_hook=lambda _t, _s, _m, _w: None,
        )

        assert any(
            "--best-checkpoint was given WITHOUT --checkpoint" in w for w in warnings
        ), (
            "a run with --best-checkpoint and no --checkpoint started silently: "
            f"warnings were {warnings}. It saves nothing unless some eval "
            "strictly improves the win rate above zero."
        )
        # The operator has to be told what to DO, not just that something is off.
        assert any("--checkpoint" in w for w in warnings)

    def test_no_hooks_at_all_still_warns(self):
        pytest.importorskip("torch")

        warnings = self._warnings_for(checkpoint_hook=None, best_checkpoint_hook=None)

        assert any("save NOTHING" in w for w in warnings)

    def test_a_properly_configured_run_warns_about_neither(self):
        pytest.importorskip("torch")

        warnings = self._warnings_for(
            checkpoint_hook=lambda _t, _s: None,
            best_checkpoint_hook=lambda _t, _s, _m, _w: None,
        )

        assert not [w for w in warnings if "checkpoint" in w], (
            f"a fully configured run cried wolf: {warnings}"
        )


# ===========================================================================
# The "best" checkpoint must hold the network that earned the win rate.
# ===========================================================================


class TestTheBestCheckpointHoldsTheEvaluatedNet:
    """The save-best path must serialize a SNAPSHOT, never the live net.

    ``DRQNGreedyPolicy`` holds ``trainer.online`` by reference and
    ``_eval_via_designated_arena`` pauses only the designated collector -- the
    learner thread keeps stepping the optimizer on the other N-1 arenas for the
    whole eval AND the whole save. Reading ``trainer.online.state_dict()`` inside
    the hook therefore ships a net that is thousands of gradient steps past the
    one the win rate describes (and reads live tensors mid-``optimizer.step()``).
    """

    def test_the_outcome_weights_do_not_track_the_live_net(self):
        pytest.importorskip("torch")
        import torch

        from agent.train import Trainer, _eval_against_opponent

        trainer = Trainer(_multi_cfg(), net_kwargs=dict(_TINY_NET))
        outcome = _eval_against_opponent(
            trainer=trainer,
            evaluate=lambda *_a, **_k: _fake_eval_report(0.5),
            policy_cls=lambda _net, device=None: ScriptedGreedyPolicy(),
            shared_transport=GenerativeBridge(),
            n_episodes=1,
            timeout_cap=64,
            env_max_episode_steps=64,
            eval_step_cap=4,
            logger=None,
            is_live=False,
            base_seed=0,
            log=None,
        )
        taken = {key: value.clone() for key, value in outcome.weights.items()}

        # What the learner thread does for the rest of the eval and the save.
        with torch.no_grad():
            for param in trainer.online.parameters():
                param.add_(1.0)

        assert all(
            torch.equal(outcome.weights[key], taken[key]) for key in taken
        ), (
            "the eval outcome carries VIEWS into the live net, so the 'best' "
            "checkpoint is whatever the learner produced by save time"
        )
        live = trainer.online.state_dict()
        assert any(
            not torch.equal(outcome.weights[key], live[key].cpu())
            for key in outcome.weights
        ), "the live net did not move; this test cannot detect aliasing"
        assert outcome.grad_step == int(trainer.grad_step)

    def test_the_driver_hands_the_hook_the_evaluated_snapshot(self, monkeypatch):
        pytest.importorskip("torch")
        import torch

        import agent.train as train_module

        sentinel = {"marker": torch.full((2,), 3.0)}

        def _stub(**_kwargs):
            return train_module._EvalOutcome(
                report=_fake_eval_report(0.5),
                weights=sentinel,
                grad_step=4_242,
            )

        monkeypatch.setattr(train_module, "_eval_via_designated_arena", _stub)

        got: List[Tuple[int, Any]] = []
        pads = [PadEnv(GenerativeBridge(), k=4) for _ in range(2)]
        result = _run_multi_arena(
            _multi_cfg(opponent="scripted"),
            pads,
            eval_every_grad_steps=3,
            eval_episodes=1,
            max_grad_steps=9,
            best_checkpoint_hook=lambda _t, step, _m, weights: got.append(
                (int(step), weights)
            ),
        )

        assert got, "the best hook never fired; this test proves nothing"
        step, weights = got[0]
        assert weights is sentinel, (
            "the save-best hook was handed the LIVE net instead of the weights "
            "the eval scored"
        )
        assert step == 4_242, (
            "the checkpoint was stamped with the grad step at SAVE time, not the "
            "one the evaluated weights were taken at"
        )
        assert result.best_grad_step == 4_242

    def test_the_written_best_file_holds_those_weights(self, monkeypatch, tmp_path):
        """End to end through ``main()``: the closure must write ``weights``."""
        pytest.importorskip("torch")
        import torch

        import eval.logging as logging_module

        import agent.train as train_module

        class _NullLogger:
            def __init__(self, *_a, **_k) -> None:
                pass

            def log(self, *_a, **_k) -> None:
                pass

            def close(self) -> None:
                pass

        monkeypatch.setattr(logging_module, "MetricsLogger", _NullLogger)

        captured: Dict[str, Any] = {}

        def _stub_main_multi(_args, _cfg, **kwargs):
            captured.update(kwargs)
            return 0

        monkeypatch.setattr(train_module, "_main_multi_arena", _stub_main_multi)

        best_path = tmp_path / "best.pt"
        assert (
            train_module.main(
                [
                    "--arenas", "2",
                    "--opponent", "scripted",
                    "--checkpoint", str(tmp_path / "latest.pt"),
                    "--best-checkpoint", str(best_path),
                ]
            )
            == 0
        )
        hook = captured.get("best_checkpoint_hook")
        assert hook is not None, "the CLI wired no save-best hook"

        evaluated = {"marker": torch.full((2,), 7.0)}
        # What the learner thread has produced by the time the hook runs.
        live = {"marker": torch.full((2,), -1.0)}

        class _StubNet:
            def state_dict(self):
                return live

        class _StubTrainer:
            online = _StubNet()

        meta = {"win_rate": 0.5, "eval_opponent": "scripted_mixed"}
        hook(_StubTrainer(), 11, meta, evaluated)

        payload = torch.load(best_path, weights_only=False)
        assert torch.equal(payload["model"]["marker"], evaluated["marker"]), (
            "the best checkpoint on disk holds the live net, not the weights the "
            "win rate it is stamped with was measured on"
        )
        assert payload["grad_step"] == 11
        assert payload["win_rate"] == pytest.approx(0.5)


# ===========================================================================
# The CLI surface: flags that exist, and defaults that make the run survivable.
# ===========================================================================


class TestConfigFromArgs:
    """The four curriculum flags + warm start must reach TrainConfig."""

    def _cfg(self, argv: List[str]) -> TrainConfig:
        from agent.train import _build_parser, _config_from_args

        return _config_from_args(_build_parser().parse_args(argv))

    def test_the_curriculum_flags_land_on_the_config(self):
        cfg = self._cfg(
            [
                "--arenas", "4",
                "--opponent", "scripted",
                "--opponent-mix-easy", "1.0",
                "--opponent-mix-easy-after", "0.35",
                "--opponent-gate-winrate", "0.7",
                "--opponent-gate-window", "25",
            ]
        )

        assert cfg.opponent == "scripted"
        assert cfg.opponent_mix_easy == 1.0
        assert cfg.opponent_mix_easy_after == 0.35
        assert cfg.opponent_gate_winrate == 0.7
        assert cfg.opponent_gate_window == 25

    def test_the_easy_only_cut_is_reachable_from_the_cli(self):
        # The plan's declared schedule cut #3. It takes BOTH flags: mix_easy=1.0
        # alone holds only until the gate fires, after which the mixture shifts
        # to opponent_mix_easy_after (0.2) and HARD becomes the majority tier.
        import random

        from agent.train import OpponentCurriculum

        cfg = self._cfg(
            [
                "--arenas", "4",
                "--opponent", "scripted",
                "--opponent-mix-easy", "1.0",
                "--opponent-mix-easy-after", "1.0",
            ]
        )
        curriculum = OpponentCurriculum(cfg)
        rng = random.Random(0)

        for _ in range(cfg.opponent_gate_window * 2):
            preset = curriculum.sample_preset(rng)
            assert preset is ScriptedPreset.EASY
            curriculum.record_episode(preset, won=True)

        assert curriculum.gate_fired, "the gate must still be able to fire"
        assert curriculum.mix_easy() == 1.0

    def test_omitted_flags_keep_the_dataclass_defaults(self):
        cfg = self._cfg([])
        default = TrainConfig()

        assert cfg.opponent_mix_easy == default.opponent_mix_easy
        assert cfg.opponent_mix_easy_after == default.opponent_mix_easy_after
        assert cfg.opponent_gate_winrate == default.opponent_gate_winrate
        assert cfg.opponent_gate_window == default.opponent_gate_window
        assert cfg.eval_opponent_preset == default.eval_opponent_preset
        assert cfg.warm_start is None

    def test_the_warm_start_flags_land_on_the_config(self):
        cfg = self._cfg(
            [
                "--arenas", "25",
                "--opponent", "scripted",
                "--warm-start", "runs/m2_multi.pt",
                "--warm-start-eps-start", "0.3",
            ]
        )

        assert cfg.warm_start == "runs/m2_multi.pt"
        assert cfg.warm_start_eps_start == 0.3

    def test_the_eval_opponent_preset_lands_on_the_config(self):
        cfg = self._cfg(["--arenas", "2", "--eval-opponent-preset", "hard"])

        assert cfg.eval_opponent_preset == "hard"

    def test_a_bad_value_is_rejected_before_anything_starts(self):
        with pytest.raises(ValueError, match="opponent_mix_easy must be in"):
            self._cfg(["--opponent-mix-easy", "1.5"])

    def test_an_empty_warm_start_is_rejected(self):
        with pytest.raises(ValueError, match="warm_start must be a non-empty path"):
            self._cfg(["--warm-start", ""])


class TestStopOnPassResolution:
    """The M2 gate is defined against the DUMMY; it cannot end a scripted run."""

    def test_a_scripted_run_does_not_stop_on_the_m2_gate_by_default(self):
        from agent.train import _resolve_stop_on_pass

        assert _resolve_stop_on_pass(None, _multi_cfg(opponent="scripted")) is False

    def test_the_dummy_path_keeps_todays_behavior(self):
        from agent.train import _resolve_stop_on_pass

        assert _resolve_stop_on_pass(None, _multi_cfg(opponent="dummy")) is True

    def test_an_explicit_flag_wins_either_way(self):
        from agent.train import _resolve_stop_on_pass

        assert _resolve_stop_on_pass(True, _multi_cfg(opponent="scripted")) is True
        assert _resolve_stop_on_pass(False, _multi_cfg(opponent="dummy")) is False

    def test_both_flag_forms_parse(self):
        from agent.train import _build_parser

        parser = _build_parser()

        assert parser.parse_args([]).stop_on_pass is None
        assert parser.parse_args(["--stop-on-pass"]).stop_on_pass is True
        assert parser.parse_args(["--no-stop-on-pass"]).stop_on_pass is False


class TestBestCheckpointPath:
    """One path for both files means the periodic save eats the best net."""

    def test_it_is_derived_from_the_checkpoint_path(self):
        from agent.train import _best_checkpoint_path

        assert _best_checkpoint_path("runs/m3.pt", None) == "runs/m3.best.pt"

    def test_an_explicit_path_wins(self):
        from agent.train import _best_checkpoint_path

        assert _best_checkpoint_path("runs/m3.pt", "runs/pick.pt") == "runs/pick.pt"

    def test_no_checkpoint_means_no_best_checkpoint(self):
        from agent.train import _best_checkpoint_path

        assert _best_checkpoint_path(None, None) is None

    def test_an_extensionless_path_still_gets_one(self):
        from agent.train import _best_checkpoint_path

        assert _best_checkpoint_path("runs/m3", None) == "runs/m3.best.pt"


# ===========================================================================
# BLOCKER 4 — AC18's last inch: the launcher must hear about the scripted bot.
# ===========================================================================


class _StubReached(Exception):
    """Sentinel: the stubbed train_multi_arena was reached (nothing started)."""


class TestScriptedOpponentReachesTheLauncher:
    """A scripted run must not spawn immune, speed-pinned opponents."""

    def _args(self, **overrides: Any):
        import argparse

        base = dict(
            port=5555, host="127.0.0.1", mc_port=25565,
            max_episodes=1, max_grad_steps=1, eval_every_grad_steps=0,
            eval_episodes=1,
        )
        base.update(overrides)
        return argparse.Namespace(**base)

    def _run(self, monkeypatch, cfg: TrainConfig, **arg_overrides: Any):
        import distributed.launcher as launcher_module

        import agent.train as train_module

        built: Dict[str, Any] = {}

        class _FakeLauncher:
            def __init__(self, **kwargs):
                built.update(kwargs)

        monkeypatch.setattr(
            launcher_module, "SubprocessArenaLauncher", _FakeLauncher
        )

        seen: Dict[str, Any] = {}

        def _stub_train_multi_arena(_cfg, **kwargs):
            seen.update(kwargs)
            raise _StubReached()

        monkeypatch.setattr(
            train_module, "train_multi_arena", _stub_train_multi_arena
        )

        with pytest.raises(_StubReached):
            train_module._main_multi_arena(
                self._args(**arg_overrides),
                cfg,
                logger=None,
                checkpoint_hook=None,
            )
        return built, seen

    def test_a_scripted_run_turns_the_dummys_immunity_off(self, monkeypatch):
        built, _seen = self._run(
            monkeypatch, _multi_cfg(opponent="scripted", arenas=4)
        )

        assert built["dummy_knockback_immune"] is False, (
            "the retrain would fight an opponent that cannot be knocked back and "
            "(on a minecraft-data bump) cannot move -- AC18"
        )

    def test_the_dummy_path_keeps_the_stationary_target(self, monkeypatch):
        built, _seen = self._run(
            monkeypatch, _multi_cfg(opponent="dummy", arenas=4)
        )

        assert built["dummy_knockback_immune"] is True

    def test_stop_on_pass_is_forwarded_to_the_loop(self, monkeypatch):
        # _main_multi_arena forwarded NO stop_on_pass at all, so the loop used its
        # own True default and a scripted run ended at its first passing eval.
        _built, seen = self._run(
            monkeypatch, _multi_cfg(opponent="scripted", arenas=4)
        )

        assert seen["stop_on_pass"] is False

    def test_an_explicit_stop_on_pass_is_forwarded(self, monkeypatch):
        _built, seen = self._run(
            monkeypatch,
            _multi_cfg(opponent="scripted", arenas=4),
            stop_on_pass=True,
        )

        assert seen["stop_on_pass"] is True

    def test_the_checkpoint_cadence_is_forwarded(self, monkeypatch):
        _built, seen = self._run(
            monkeypatch,
            _multi_cfg(opponent="scripted", arenas=4),
            checkpoint_every_grad_steps=500,
        )

        assert seen["checkpoint_every_grad_steps"] == 500
