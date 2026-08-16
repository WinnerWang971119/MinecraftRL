"""train — n-step Double-DQN training loop for the Dueling-DRQN (T16).

Drives the full training cycle (training-spec §5 / §8):

  per episode:  env.reset(seed) -> net.init_hidden -> rollout with per-EPISODE
                ε-greedy ``act`` (collecting transitions + per-step hidden states)
                -> replay.add_episode
  per step (once ``len(replay) >= min_replay``):
                replay.sample_sequences -> n-step Double-DQN target -> Huber
                (smooth-L1) loss weighted by PER IS weights, on the post-burn-in
                scored steps only -> backward -> grad-norm clip -> optimizer step
                -> replay.update_priorities(|δ|) -> soft (Polyak) target update.

Reads every hyperparameter from :class:`agent.train_config.TrainConfig`; the
sequence/burn-in geometry and the n-step/gamma are shared with the replay buffer
so storage and learner can never disagree.

------------------------------------------------------------------------------
The exact target (training-spec §5/§8)
------------------------------------------------------------------------------
For a sampled window of ``T = burn_in + seq_len`` transitions, transition ``i``
is ``(s_i, a_i, r_i, s'_i, done_i)`` (``s'_i`` is ``next_obs[i]``, the state at
time ``i + 1``). For each SCORED step ``i`` in ``[B, T)``:

    G_i = Σ_{k=0}^{n-1} γ^k r_{i+k}        (truncated at the first done in-window)
    a*  = argmax_a Q_online(s_{i+n}, a)    (Double-DQN: ONLINE selects the action)
    y_i = G_i + bootstrap_i · γ^n · Q_target(s_{i+n}, a*)   (TARGET evaluates it)
    δ_i = y_i − Q_online(s_i, a_i)

``G_i`` and ``bootstrap_i`` come from :func:`agent.replay.compute_n_step_returns`
(the network-free reward arithmetic) so the bootstrap term is dropped exactly
where a ``done`` truncates the return or the n-step horizon runs off the window.

Because the net is RECURRENT, ``Q_target(s_{i+n}, ·)`` must be evaluated with the
LSTM hidden state that has consumed the FULL contiguous history s_0..s_{i+n}, not
a state seeded from the wrong stream. Within an episode the window is contiguous
(``next_obs[:, :-1] == obs[:, 1:]``), so we build the extended observation stream
``obs_ext = concat(obs, next_obs[:, -1:])`` of length ``T+1`` — where
``obs_ext[p] == s_p`` for ``p`` in ``[0, T]`` — and run ONE seeded forward of the
online and target nets over it (under ``no_grad``). The bootstrap state ``s_{i+n}``
then sits at ``obs_ext`` position ``i + n``, and its Q uses the correct recurrent
memory by construction.

The loss is the IS-weighted Huber (smooth-L1) of ``δ`` over the scored steps:
``L = mean(w_i · Huber(δ_i))``. The scored-step Q comes from
``DuelingDRQN.forward_with_burn_in`` so gradients flow ONLY through the
post-burn-in steps (the burn-in prefix warms the hidden state under ``no_grad``).

Owner: T16 (DQN core track)
"""

from __future__ import annotations

import random
import threading
from collections import deque
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    Tuple,
)

import numpy as np
import torch
import torch.nn.functional as F

from agent.actions import N_ACTIONS
from agent.contract_config import MAX_EPISODE_STEPS, code_version
from agent.dqn import DuelingDRQN
from agent.progress import ProgressReporter, progress_metrics
from agent.replay import PrioritizedSequenceReplay, SequenceBatch
from agent.seeding import seed_everything
from agent.train_config import TrainConfig
from distributed.serialization import Episode
from opponents.scripted_bot import OpponentView, ScriptedBot, ScriptedPreset

__all__ = [
    "EnvProtocol",
    "RolloutPolicy",
    "Trainer",
    "train",
    "epsilon_for_episode",
    "effective_eps_start",
    "load_checkpoint_state_dict",
    "arena_episode_seed",
    "opponent_seed",
    "EpisodeOpponent",
    "OpponentCurriculum",
    "ScriptedOpponentDriver",
    "build_scripted_opponents",
    "EvalOpponentDriver",
    "build_eval_opponent",
    "collect_episode",
    "hidden_snapshot",
    "LearnStats",
    "M2Result",
    "train_vs_dummy",
    "run_m2",
    "MultiArenaResult",
    "train_multi_arena",
    "main",
]


# ---------------------------------------------------------------------------
# Env seam — the minimal Gym-style surface the trainer depends on.
#
# ``env.mc_pvp_env.MCPvPEnv`` satisfies this; so does the tiny in-test fake env
# (no socket / no live server). Keeping it a Protocol lets the smoke test inject
# a fake without constructing a bridge transport.
# ---------------------------------------------------------------------------


class EnvProtocol(Protocol):
    """Structural Gym-style env the trainer rolls out against.

    Deliberately still the MINIMAL dummy-path surface. The scripted-opponent path
    (T12) additionally calls ``env.raw_opponent_view()`` and passes
    ``env.step(action, opp_action=...)``, but those are NOT declared here: this
    Protocol is what the M1/M2 path requires, and widening it would structurally
    un-satisfy every existing two-argument fake env for a capability only the
    opponent path uses. :func:`collect_episode` documents the extra surface at the
    one place that depends on it.
    """

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Start an episode and return the initial observation ``(OBS_DIM,)``."""
        ...

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Advance one decision step -> ``(obs, reward, done, info)``."""
        ...


# ---------------------------------------------------------------------------
# Hooks — call signatures only (eval is T19, full M2 training is T20).
# ---------------------------------------------------------------------------

#: Called every ``cfg.eval_interval`` gradient steps. T19 owns the body.
EvalHook = Callable[["Trainer", int], None]
#: Called every ``cfg.checkpoint_interval`` gradient steps. T20 owns the body.
CheckpointHook = Callable[["Trainer", int], None]
#: Called when an eval produces a NEW BEST win rate (T13's save-best path). Takes
#: the same ``(trainer, grad_step)`` plus the eval metadata that justified the
#: save, so the shipped file can record WHAT it scored and against WHOM — this
#: repo's documented weak spot is exactly that kind of missing run provenance.
BestCheckpointHook = Callable[["Trainer", int, Mapping[str, Any]], None]
#: Called every ``cfg.log_interval`` gradient steps with the latest learn stats.
LogHook = Callable[["Trainer", int, "LearnStats"], None]


@dataclass
class LearnStats:
    """Per-gradient-step learner diagnostics (handed to the log hook).

    Attributes:
        grad_step: 1-based index of this gradient step.
        loss: Scalar IS-weighted Huber loss for the batch.
        td_error_mean: Mean absolute TD error over the scored batch (pre-priority).
        grad_norm: Global gradient L2 norm BEFORE clipping (clip reduces it to
            at most ``cfg.grad_clip``).
        epsilon: ε used for the most recently collected episode.
        beta: Current PER importance-sampling exponent β.
        replay_size: Stored transitions at the time of this step.
    """

    grad_step: int
    loss: float
    td_error_mean: float
    grad_norm: float
    epsilon: float
    beta: float
    replay_size: int


# ---------------------------------------------------------------------------
# ε schedule — per EPISODE (NOT per step). See agent.seeding's gotcha note.
# ---------------------------------------------------------------------------


def effective_eps_start(cfg: TrainConfig) -> float:
    """Return the ε the schedule actually STARTS at for this run.

    ``cfg.eps_start`` for a fresh run; ``cfg.warm_start_eps_start`` when
    ``cfg.warm_start`` names a checkpoint.

    THIS IS THE HALF OF A WARM START THAT IS EASY TO FORGET AND FATAL TO OMIT.
    A warm start exists to keep a trained policy; the fresh-init default
    ``eps_start=1.0`` immediately discards it, acting uniformly at random for the
    first episode and mostly at random for the whole ``eps_decay_episodes``
    window — which is also the window that fills the (deliberately fresh) replay
    buffer. The run then relearns from noise having paid for a checkpoint it
    never used, and nothing about the logs looks wrong.

    It is resolved HERE, inside the one function every ε consumer calls
    (``distributed.actor.Collector`` imports it, so all N arenas are covered),
    rather than by rewriting the frozen config: there is then exactly one place
    the effective schedule is decided and no way for a caller to construct a
    config whose ε silently disagrees with the run it is driving.

    Args:
        cfg: The training config.

    Returns:
        The ε value at episode 0 for this run's schedule.
    """
    if cfg.warm_start is None:
        return float(cfg.eps_start)
    return float(cfg.warm_start_eps_start)


def epsilon_for_episode(episode: int, cfg: TrainConfig) -> float:
    """Return the ε-greedy exploration rate for episode index ``episode``.

    Linear decay from :func:`effective_eps_start` (``cfg.eps_start``, or
    ``cfg.warm_start_eps_start`` under a warm start) to ``cfg.eps_end`` over the
    FIRST ``cfg.eps_decay_episodes`` episodes, then flat at ``cfg.eps_end``. The
    decay is per EPISODE on purpose: episodes are short (tens of decisions), so a
    per-step decay would collapse ε in a handful of episodes and kill
    exploration (the documented gotcha in :mod:`agent.seeding`).

    Args:
        episode: 0-based episode index (>= 0). Clamped at 0 for safety.
        cfg: The training config holding the ε schedule.

    Returns:
        ε in ``[cfg.eps_end, effective_eps_start(cfg)]``, monotonically
        non-increasing in ``episode`` and flat within a single episode.
    """
    ep = max(0, int(episode))
    start = effective_eps_start(cfg)
    # frac goes 0 -> 1 over the first eps_decay_episodes, then saturates at 1.
    frac = min(ep / float(cfg.eps_decay_episodes), 1.0)
    return start + (cfg.eps_end - start) * frac


# ---------------------------------------------------------------------------
# Warm start — initialize a run from an existing checkpoint (T13).
#
# The regime is pinned by the plan and implemented in exactly three places:
#   1. the WEIGHTS load into the online net AND the target net (here + Trainer);
#   2. ε RESTARTS at ``cfg.warm_start_eps_start`` (``effective_eps_start`` above);
#   3. the replay buffer stays FRESH — nothing below restores one, deliberately:
#      the stored transitions were produced by a different reward regime and a
#      different (stationary) opponent, and replaying them is how a warm start
#      turns into a slow relearn of the thing being replaced.
# ---------------------------------------------------------------------------

#: Keys a checkpoint may wrap its ``state_dict`` under. ``agent.train``'s own CLI
#: writes ``"model"``; older/alternate artifacts use the others. Mirrors
#: ``eval.evaluate._load_drqn`` so a checkpoint that evals can also warm-start.
_CHECKPOINT_STATE_DICT_KEYS: Tuple[str, ...] = (
    "model",
    "model_state_dict",
    "state_dict",
    "online",
)


def load_checkpoint_state_dict(path: str, *, map_location: Any = "cpu") -> Dict[str, Any]:
    """Load ``path`` and return the bare network ``state_dict`` inside it.

    Accepts either a raw ``state_dict`` or a checkpoint dict wrapping one under
    any of :data:`_CHECKPOINT_STATE_DICT_KEYS` (liberal in what it accepts, loud
    when nothing matches — a warm start that silently loaded nothing would look
    exactly like a warm start that worked).

    Args:
        path: Filesystem path to the checkpoint.
        map_location: ``torch.load`` map location (default CPU).

    Returns:
        The extracted ``state_dict`` mapping.

    Raises:
        FileNotFoundError: if ``path`` does not exist — raised with the resolved
            path so an overnight run fails in the first second rather than at the
            first checkpoint save.
        ValueError: if the payload is not a mapping, or is a mapping in which no
            known wrapper key holds a dict and whose own values are not tensors.
    """
    import os

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"warm-start checkpoint not found: {path!r} (resolved to "
            f"{os.path.abspath(path)})"
        )

    payload = torch.load(path, map_location=map_location)
    if not isinstance(payload, dict):
        raise ValueError(
            f"warm-start checkpoint {path!r} holds a "
            f"{type(payload).__name__}, not a state_dict or a dict wrapping one"
        )

    for key in _CHECKPOINT_STATE_DICT_KEYS:
        value = payload.get(key)
        if isinstance(value, dict):
            return value

    # A raw state_dict: a flat mapping whose values are tensors.
    if payload and all(torch.is_tensor(v) for v in payload.values()):
        return payload

    raise ValueError(
        f"warm-start checkpoint {path!r} carries no network weights: expected a "
        f"raw state_dict or one of {list(_CHECKPOINT_STATE_DICT_KEYS)}, found keys "
        f"{sorted(payload)[:8]}"
    )


# ---------------------------------------------------------------------------
# Per-arena seed scheme — deterministic and collision-free across arenas.
# ---------------------------------------------------------------------------


def arena_episode_seed(cfg: TrainConfig, arena_id: int, local_ep: int) -> int:
    """Return the deterministic per-episode seed for one arena's local episode.

    The scheme is ``cfg.seed + arena_id * cfg.seed_stride + local_ep``. Each arena
    owns a contiguous block of ``cfg.seed_stride`` seeds (the stride is large
    enough that no two arenas ever collide for any realistic episode count), so
    distinct arenas draw statistically independent episode streams while a given
    ``(arena_id, local_ep)`` always reproduces the same env reset and action RNG.

    For the single-arena path (``arena_id == 0``, ``local_ep == episode_index``)
    this reduces to ``cfg.seed + episode_index`` — exactly the seed the original
    single-arena rollout used, so N=1 stays byte-identical.

    Args:
        cfg: The training config (supplies ``seed`` and ``seed_stride``).
        arena_id: 0-based arena index.
        local_ep: 0-based episode index local to that arena.

    Returns:
        The integer seed to hand to both ``env.reset`` and the action RNG.
    """
    return int(cfg.seed) + int(arena_id) * int(cfg.seed_stride) + int(local_ep)


# ---------------------------------------------------------------------------
# Opponent stepping + the win-rate-gated EASY/HARD curriculum (T12).
#
# WHAT THIS IS FOR. Through M2 the opponent was a stationary dummy served
# entirely by the bridge / Paper server, so the training loop never stepped an
# opponent policy — it only talked to the env. The demo-day retrain fights the
# omniscient :class:`~opponents.scripted_bot.ScriptedBot` instead, which lives in
# PYTHON: once per decision the loop reads ``env.raw_opponent_view()``, asks the
# bot for a ``Macro``, and threads it through ``env.step(action, opp_action=...)``
# so the bridge drives the opponent handle in the SAME window as the learner's.
#
# ONE STEP == ONE DECISION WINDOW. The env shadow-tracks the opponent's attack
# meter by COUNTING decision windows (it deliberately does not read the coarse
# server clock in ``state.tick``), so the rollout must call ``env.step`` exactly
# once per opponent decision — never skipping a step, never taking two for one
# decision. :func:`collect_episode` below is the one place that invariant lives.
#
# PASS THE VIEW STRAIGHT THROUGH. ``raw_opponent_view()`` clamps
# ``attack_cooldown`` to exactly 1.0 because ``ScriptedBot`` treats the swing as
# ready at ``>= 1.0 - 1e-6``. Rounding it, re-deriving it, or handing the bot a
# view captured before the last step yields a value a hair under 1.0 — and then
# the bot NEVER ATTACKS, presenting as a mysteriously passive opponent rather
# than as an error. Nothing on this path may touch that number.
#
# THE CURRICULUM IS A GATED MIXTURE, NOT A PROMOTION CLIFF. Every episode draws
# EASY or HARD at ``cfg.opponent_mix_easy``; once the rolling win rate over a
# FULL window of ``cfg.opponent_gate_window`` EASY episodes reaches
# ``cfg.opponent_gate_winrate``, the draw shifts to
# ``cfg.opponent_mix_easy_after`` — which is 0.2, not 0.0, so EASY episodes keep
# arriving (they are what keeps the gate's own window fed). The gate can only
# ever change one probability: it never blocks, never waits, never raises, and
# never divides by zero, so a run whose gate never fires simply trains to
# completion at the initial ratio (AC10).
# ---------------------------------------------------------------------------


#: Seed roles inside one arena's opponent band, in index order. Each per-arena
#: driver owns three independent RNG streams — the EASY/HARD mixture draw, and
#: one per ``ScriptedBot`` — and they must not collide with each other or with
#: another arena's.
_OPPONENT_SEED_ROLES: Tuple[str, ...] = ("mixture", "easy", "hard", "eval")


def opponent_seed(cfg: TrainConfig, arena_id: int, role: str) -> int:
    """Return the deterministic seed for one arena's opponent RNG ``role``.

    The scheme is ``arena_episode_seed(cfg, arena_id, 0) + cfg.seed_stride // 2 +
    role_index`` — i.e. it reuses the per-arena seed bands
    :func:`arena_episode_seed` already carves out (arena ``i`` owns
    ``cfg.seed_stride`` consecutive seeds) but sits HALF A STRIDE up, far above
    the ``cfg.seed + arena_id * stride + local_ep`` episode seeds that grow from
    the bottom of the same band. With the default stride that is 500 000
    episodes of clearance per arena, so an opponent stream can never coincide
    with an episode stream, and two arenas can never coincide at all.

    Seeding at CONSTRUCTION (rather than per episode) is the deliberate half of
    the determinism contract: ``ScriptedBot.reset()`` with no argument is a
    no-op on the RNG (the gym convention), so the constructor seed governs the
    whole run while consecutive episodes stay naturally decorrelated. Re-seeding
    per episode from entropy would destroy run reproducibility; re-seeding per
    episode from a constant would replay one identical opponent stream forever.

    Args:
        cfg: The training config (supplies ``seed`` and ``seed_stride``).
        arena_id: 0-based arena index.
        role: One of :data:`_OPPONENT_SEED_ROLES` — ``"mixture"``, ``"easy"``,
            ``"hard"``, or ``"eval"`` (the periodic eval's own opponent, which
            :func:`build_eval_opponent` seeds from a band no collector owns).

    Returns:
        The integer seed for that arena's stream.

    Raises:
        ValueError: if ``role`` is not a known role (a typo would otherwise
            silently alias two streams onto one seed).
    """
    try:
        offset = _OPPONENT_SEED_ROLES.index(role)
    except ValueError:
        raise ValueError(
            f"unknown opponent seed role {role!r}; expected one of "
            f"{list(_OPPONENT_SEED_ROLES)}"
        ) from None
    return arena_episode_seed(cfg, arena_id, 0) + int(cfg.seed_stride) // 2 + offset


class EpisodeOpponent(Protocol):
    """The per-arena opponent surface :func:`collect_episode` drives.

    Deliberately narrow: three calls, all on the collector's own thread, with the
    episode boundary made explicit so the implementation can sample a difficulty
    tier per episode and score the result. :class:`ScriptedOpponentDriver` is the
    only implementation; a test may substitute a recorder.
    """

    def begin_episode(self) -> None:
        """Called once before an episode's first decision (pick a tier, reset)."""
        ...

    def act(self, view: OpponentView) -> int:
        """Return the opponent's macro (``0..N_ACTIONS-1``) for this window."""
        ...

    def observe_outcome(self, info: Mapping[str, Any]) -> None:
        """Called once with the FINAL step's ``info`` so the episode can be scored."""
        ...


class OpponentCurriculum:
    """Thread-safe EASY/HARD mixture with a rolling win-rate gate (AC10).

    Shared by every arena in a multi-arena run — the arenas are concurrent
    threads feeding one learner, so the gate's state is guarded by a lock and
    every mutation is a short critical section. The per-arena Bernoulli draw
    itself happens OUTSIDE the lock, against that arena's own
    ``random.Random``, so the mixture stays reproducible per arena and the lock
    is never held across a call into another object.

    The gate:

      * only EASY episodes enter the window (the gate measures the agent against
        the EASY tier, so a HARD loss must not depress it and a HARD win must not
        inflate it — HARD outcomes are still counted for reporting);
      * the window must be **FULL** (``cfg.opponent_gate_window`` EASY episodes)
        before it is compared to the threshold. Evaluating a partial window would
        fire the gate on the very first EASY win (1/1 == 100% >= 0.6);
      * once fired it LATCHES. After the shift EASY episodes arrive ~4x more
        slowly, so an un-firing gate would flap on a window that refills at a
        different rate than it drained. Latching is not a promotion: the mixture
        still draws EASY at ``cfg.opponent_mix_easy_after``, which is 0.2.

    It cannot stall a run. There is no waiting, no blocking, no unbounded
    accumulation (the window is a bounded ``deque``), and no arithmetic that can
    divide by zero — a gate that never fires simply leaves
    :meth:`mix_easy` at ``cfg.opponent_mix_easy`` forever.

    Args:
        cfg: The training config supplying the four curriculum knobs.
    """

    def __init__(self, cfg: TrainConfig) -> None:
        self._cfg = cfg
        self._lock = threading.Lock()
        # Bounded by construction: the deque drops the oldest EASY outcome once
        # the window is full, so this is a rolling win rate and not a growing
        # accumulator. maxlen >= 1 is guaranteed by TrainConfig validation.
        self._easy_window: Deque[bool] = deque(maxlen=int(cfg.opponent_gate_window))
        self._gate_fired = False
        self._episodes = 0
        self._easy_episodes = 0
        self._hard_episodes = 0
        self._easy_wins = 0
        self._hard_wins = 0

    # -- read side ---------------------------------------------------------

    @property
    def gate_fired(self) -> bool:
        """True once the rolling EASY win rate has cleared the gate (latched)."""
        with self._lock:
            return self._gate_fired

    def mix_easy(self) -> float:
        """The CURRENT probability that an episode draws the EASY preset."""
        with self._lock:
            return float(
                self._cfg.opponent_mix_easy_after
                if self._gate_fired
                else self._cfg.opponent_mix_easy
            )

    def easy_window_win_rate(self) -> Optional[float]:
        """Rolling EASY win rate, or ``None`` while the window is not yet full.

        ``None`` (rather than a partial average) is what makes "the window must
        be full" a property of the data and not only of the gate's caller.
        """
        with self._lock:
            window = self._easy_window
            if len(window) < int(self._cfg.opponent_gate_window):
                return None
            return sum(1 for won in window if won) / float(len(window))

    def stats(self) -> Dict[str, Any]:
        """A snapshot of the curriculum's counters (logging / assertions)."""
        with self._lock:
            window = list(self._easy_window)
            full = len(window) >= int(self._cfg.opponent_gate_window)
            return {
                "episodes": self._episodes,
                "easy_episodes": self._easy_episodes,
                "hard_episodes": self._hard_episodes,
                "easy_wins": self._easy_wins,
                "hard_wins": self._hard_wins,
                "gate_fired": self._gate_fired,
                "easy_window_size": len(window),
                "easy_window_win_rate": (
                    sum(1 for won in window if won) / float(len(window))
                    if full and window
                    else None
                ),
                "mix_easy": float(
                    self._cfg.opponent_mix_easy_after
                    if self._gate_fired
                    else self._cfg.opponent_mix_easy
                ),
            }

    # -- write side --------------------------------------------------------

    def sample_preset(self, rng: random.Random) -> ScriptedPreset:
        """Draw this episode's preset from the current mixture.

        Args:
            rng: The CALLER's own ``random.Random`` (one per arena), so the draw
                is reproducible per arena and no two arenas share RNG state. The
                draw happens outside the curriculum's lock.

        Returns:
            ``ScriptedPreset.EASY`` with probability :meth:`mix_easy`, else
            ``ScriptedPreset.HARD``. ``mix_easy == 1.0`` always yields EASY
            (``random()`` is on ``[0, 1)``) and ``0.0`` never does.
        """
        p_easy = self.mix_easy()
        return ScriptedPreset.EASY if rng.random() < p_easy else ScriptedPreset.HARD

    def record_episode(self, preset: ScriptedPreset, won: bool) -> bool:
        """Score one finished episode; return True iff the gate fired on THIS call.

        Args:
            preset: The tier that episode was fought at.
            won: Whether the LEARNER won it (``info["won"]`` from the env's final
                step — the agent's win, which is what the gate measures).

        Returns:
            ``True`` only on the single call that flips the gate, so a caller can
            log the transition exactly once. Always ``False`` afterwards.
        """
        fired_now = False
        with self._lock:
            self._episodes += 1
            if preset is ScriptedPreset.EASY:
                self._easy_episodes += 1
                if won:
                    self._easy_wins += 1
                self._easy_window.append(bool(won))
                # FULL window only — a partial one fires on the first EASY win.
                if (
                    not self._gate_fired
                    and len(self._easy_window) >= int(self._cfg.opponent_gate_window)
                ):
                    rate = sum(1 for w in self._easy_window if w) / float(
                        len(self._easy_window)
                    )
                    if rate >= float(self._cfg.opponent_gate_winrate):
                        self._gate_fired = True
                        fired_now = True
            else:
                self._hard_episodes += 1
                if won:
                    self._hard_wins += 1
        return fired_now


class ScriptedOpponentDriver:
    """One arena's scripted opponent: per-episode tier draw + per-window macro.

    Owns TWO :class:`~opponents.scripted_bot.ScriptedBot` instances — one per
    preset — plus the mixture RNG, all seeded from this arena's own band via
    :func:`opponent_seed`. Two bots rather than one because ``ScriptedBot`` fixes
    its preset at construction and the curriculum re-draws the tier every
    episode; rebuilding a bot per episode would either replay one identical RNG
    stream (constant seed) or destroy reproducibility (entropy seed). The
    invariant that matters is that NOTHING is shared across arenas: every arena
    gets its own driver, its own bots, and its own RNG streams, because all
    ``ScriptedBot`` state is instance-level by design.

    Per episode: :meth:`begin_episode` draws the tier and calls the active bot's
    ``reset()`` with NO argument — which is deliberately a no-op on the RNG (gym
    convention: continue the stream), so episodes stay decorrelated while the
    constructor seed still governs the whole run.

    Args:
        cfg: The training config (seeds + curriculum knobs).
        curriculum: The SHARED :class:`OpponentCurriculum` (one per run).
        arena_id: 0-based arena index; selects this driver's seed band.
    """

    def __init__(
        self, cfg: TrainConfig, curriculum: OpponentCurriculum, arena_id: int
    ) -> None:
        self.arena_id = int(arena_id)
        self._curriculum = curriculum
        self._rng = random.Random(opponent_seed(cfg, self.arena_id, "mixture"))
        self._bots: Dict[ScriptedPreset, ScriptedBot] = {
            ScriptedPreset.EASY: ScriptedBot(
                ScriptedPreset.EASY, seed=opponent_seed(cfg, self.arena_id, "easy")
            ),
            ScriptedPreset.HARD: ScriptedBot(
                ScriptedPreset.HARD, seed=opponent_seed(cfg, self.arena_id, "hard")
            ),
        }
        # The tier in force. Replaced by every begin_episode(); the initial value
        # only matters if act() were somehow called first, which collect_episode
        # never does.
        self._preset = ScriptedPreset.EASY

    @property
    def preset(self) -> ScriptedPreset:
        """The preset this arena is currently fighting at."""
        return self._preset

    @property
    def name(self) -> str:
        """The active bot's name (``"scripted_easy"`` / ``"scripted_hard"``)."""
        return self._bots[self._preset].name

    def bot_for(self, preset: ScriptedPreset) -> ScriptedBot:
        """This arena's bot for ``preset`` (exposed for tests / introspection)."""
        return self._bots[preset]

    def begin_episode(self) -> None:
        """Draw this episode's tier and start the chosen bot's episode."""
        self._preset = self._curriculum.sample_preset(self._rng)
        # NO argument on purpose: ScriptedBot.reset() is a no-op on the RNG, so
        # the stream continues across episodes under the constructor's seed.
        # reset(None) does NOT re-seed either — never pass one hoping it will.
        self._bots[self._preset].reset()

    def act(self, view: OpponentView) -> int:
        """Return the active bot's macro for this decision window.

        ``view`` is passed through untouched — in particular its
        ``attack_cooldown``, which the producer clamped to exactly 1.0 and which
        the bot compares against a deliberately tight ``>= 1.0 - 1e-6``.

        Returns:
            The macro as an ``int`` in ``[0, N_ACTIONS)``. ``ScriptedBot.act``
            only ever returns ``Macro`` members, so the env's own range check on
            ``opp_action`` is unreachable from here.
        """
        return int(self._bots[self._preset].act(view))

    def observe_outcome(self, info: Mapping[str, Any]) -> None:
        """Score the finished episode into the shared curriculum.

        ``info["won"]`` is the LEARNER's win (the opponent died), which is what
        the gate measures. A missing key reads as a loss: the only way to get
        one is an episode that ended without a final env ``info``, which is not
        a win by any reading.

        An episode aborted mid-flight by a ``BridgeError`` never reaches here,
        so a lost episode is simply not scored — it is not counted as a loss
        against the gate, which would otherwise let a flaky pad depress the
        curriculum.
        """
        self._curriculum.record_episode(self._preset, bool(info.get("won", False)))


def build_scripted_opponents(
    cfg: TrainConfig,
) -> Tuple[OpponentCurriculum, Callable[[int], ScriptedOpponentDriver]]:
    """Build the shared curriculum and a per-arena driver factory.

    The factory MEMOIZES one :class:`ScriptedOpponentDriver` per ``arena_id``, so
    a given arena keeps the same bots (and therefore the same continuing RNG
    streams) for the whole run, while two arenas can never be handed the same
    driver. Memoization is lock-guarded because a pool may build its collectors
    from more than one thread.

    Args:
        cfg: The training config (seeds + curriculum knobs).

    Returns:
        ``(curriculum, opponent_for)`` — the shared gate and
        ``arena_id -> ScriptedOpponentDriver``.
    """
    curriculum = OpponentCurriculum(cfg)
    drivers: Dict[int, ScriptedOpponentDriver] = {}
    lock = threading.Lock()

    def opponent_for(arena_id: int) -> ScriptedOpponentDriver:
        key = int(arena_id)
        with lock:
            driver = drivers.get(key)
            if driver is None:
                driver = ScriptedOpponentDriver(cfg, curriculum, key)
                drivers[key] = driver
            return driver

    return curriculum, opponent_for


class EvalOpponentDriver:
    """The PERIODIC EVAL's opponent: fixed tier schedule, fully deterministic.

    Satisfies ``eval.evaluate.EvalOpponent`` (``begin_episode`` / ``act``). It is
    deliberately NOT a :class:`ScriptedOpponentDriver`: it never touches the
    curriculum and never scores anything back into it, because an eval measures
    the agent and must not move the training distribution it is measuring.

    Two properties make its win rate comparable across the whole run, which is
    the entire point — a checkpoint is SELECTED on this number:

      * **Fixed tier schedule.** ``"mixed"`` alternates EASY/HARD strictly by
        episode index (even -> EASY, odd -> HARD); ``"easy"`` / ``"hard"`` pin
        one tier. It never samples the curriculum's mixture, which SHIFTS when
        the gate fires — two evals either side of that shift would score
        different opponents under one name.
      * **Per-episode reseed from a fixed base.** Episode ``i``'s bot is reseeded
        to ``base_seed + i``. Without this, episode ``i``'s opponent RNG state
        would depend on how many decisions the earlier episodes took, which
        depends on the agent — so the "same" eval opponent would drift as the
        agent improved and the win-rate series would compare different fights.
        This is a DETERMINISTIC reseed, not the forbidden per-episode entropy
        reseed: rerunning the same eval replays the same opponent exactly.

    Args:
        cfg: The training config (supplies the preset choice).
        base_seed: Seed for episode 0; episode ``i`` uses ``base_seed + i``.
        preset_choice: One of ``"mixed"`` / ``"easy"`` / ``"hard"``; defaults to
            ``cfg.eval_opponent_preset``.
    """

    def __init__(
        self,
        cfg: TrainConfig,
        *,
        base_seed: int,
        preset_choice: Optional[str] = None,
    ) -> None:
        choice = str(
            cfg.eval_opponent_preset if preset_choice is None else preset_choice
        )
        if choice not in ("mixed", "easy", "hard"):
            raise ValueError(
                f"eval opponent preset must be 'mixed', 'easy' or 'hard', got "
                f"{choice!r}"
            )
        self._choice = choice
        self._base_seed = int(base_seed)
        self._bots: Dict[ScriptedPreset, ScriptedBot] = {
            ScriptedPreset.EASY: ScriptedBot(ScriptedPreset.EASY, seed=self._base_seed),
            ScriptedPreset.HARD: ScriptedBot(ScriptedPreset.HARD, seed=self._base_seed),
        }
        # -1 so the first begin_episode() lands on episode 0 (EASY under "mixed").
        self._episode_index = -1
        self._preset = (
            ScriptedPreset.HARD if choice == "hard" else ScriptedPreset.EASY
        )

    @property
    def name(self) -> str:
        """Stable name for the report / logs (``"scripted_mixed"`` etc.)."""
        return f"scripted_{self._choice}"

    @property
    def preset(self) -> ScriptedPreset:
        """The tier the CURRENT eval episode is being fought at."""
        return self._preset

    def begin_episode(self) -> None:
        """Advance to the next eval episode: pick its tier and reseed its bot."""
        self._episode_index += 1
        if self._choice == "easy":
            self._preset = ScriptedPreset.EASY
        elif self._choice == "hard":
            self._preset = ScriptedPreset.HARD
        else:
            # Strict alternation, not a draw: an RNG'd mixture would make two
            # evals of the same length face different tier sequences.
            self._preset = (
                ScriptedPreset.EASY
                if self._episode_index % 2 == 0
                else ScriptedPreset.HARD
            )
        # reset(<int>) DOES re-seed (reset() with no argument is a no-op on the
        # RNG — the gym convention the training driver relies on). Here the
        # re-seed is what we want: every eval replays an identical opponent.
        self._bots[self._preset].reset(self._base_seed + self._episode_index)

    def act(self, view: OpponentView) -> int:
        """Return the active bot's macro, passing ``view`` through untouched."""
        return int(self._bots[self._preset].act(view))


def build_eval_opponent(
    cfg: TrainConfig, *, preset_choice: Optional[str] = None
) -> Optional[Callable[[], EvalOpponentDriver]]:
    """Return a factory for the eval's opponent, or ``None`` for the dummy path.

    ``cfg.opponent == "dummy"`` yields ``None``: the stationary dummy is served
    by the bridge, there is no opponent policy to step, and the eval keeps its
    byte-identical M2 wire line.

    The returned callable builds a FRESH :class:`EvalOpponentDriver` per eval,
    all from the same base seed, so eval #1 and eval #40 face exactly the same
    opponent behavior and their win rates are comparable — which is what makes
    "select the checkpoint with the highest scripted win rate" a real criterion
    rather than a comparison of two different fights.

    The base seed comes from arena band ``cfg.arenas`` — one past the last
    collector — so the eval opponent's RNG can never coincide with any arena's
    training opponent or episode stream.

    Args:
        cfg: The training config.
        preset_choice: Optional override of ``cfg.eval_opponent_preset``.

    Returns:
        ``() -> EvalOpponentDriver``, or ``None`` on the dummy path.
    """
    if cfg.opponent != "scripted":
        return None

    base_seed = opponent_seed(cfg, arena_id=int(cfg.arenas), role="eval")

    def _build() -> EvalOpponentDriver:
        return EvalOpponentDriver(
            cfg, base_seed=base_seed, preset_choice=preset_choice
        )

    return _build


# ---------------------------------------------------------------------------
# Reentrant rollout — the actor-side collection path (shared by N=1 and N>1).
#
# ``collect_episode`` below is a FREE function with NO writes to any shared
# learner state: ε and the per-episode seed are passed in, and it returns an
# immutable :class:`~distributed.serialization.Episode` instead of touching a
# replay buffer or an episode counter. The single-arena ``Trainer.collect_episode``
# wrapper derives the same ε/seed the original loop did and re-adds the replay
# write, so its observable behavior is unchanged.
# ---------------------------------------------------------------------------


class RolloutPolicy(Protocol):
    """The minimal acting surface :func:`collect_episode` rolls out against.

    Both the one-shot adapter the single-arena wrapper builds around
    ``Trainer.online`` and the distributed ``SnapshotPolicy`` (T7 reconciles the
    latter to this contract) satisfy it. The free rollout depends ONLY on this
    surface, never on any concrete net or trainer attribute.

    The three Episode-stamp attributes describe the snapshot the policy is acting
    under; ``collect_episode`` copies them verbatim onto the returned Episode.

    Attributes:
        arena_id: Which arena this policy collects for (0-based).
        policy_version: Weight-snapshot version the policy is acting under.
        code_version: Train/serve build stamp for the actor/learner skew guard.
    """

    arena_id: int
    policy_version: int
    code_version: str

    def reseed(self, episode_seed: int) -> None:
        """Re-seed the policy's action RNG for the start of this episode."""
        ...

    def init_hidden(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the initial LSTM ``(h, c)`` hidden state for a single sequence."""
        ...

    def act(
        self,
        obs: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor],
        epsilon: float,
    ) -> Tuple[int, Tuple[torch.Tensor, torch.Tensor]]:
        """Select one action (no grad) and return ``(action, new_hidden)``."""
        ...


def hidden_snapshot(hidden: Tuple[torch.Tensor, torch.Tensor]) -> np.ndarray:
    """Snapshot an LSTM ``(h, c)`` pair for the current single-sequence step.

    Returns a contiguous ``float32`` array of shape
    ``(2, num_layers, lstm_hidden)`` (the (h, c) pair) detached to NumPy. The
    batch axis (== 1 during rollout) is squeezed and h/c are stacked on a new
    leading axis. This is net-agnostic — it only reshapes the carried state — so
    the free rollout can snapshot without going back through the policy. It is the
    exact arithmetic ``Trainer._hidden_snapshot`` performs.

    Args:
        hidden: The current LSTM ``(h, c)`` tuple, each ``(num_layers, 1, hidden)``.

    Returns:
        ``float32`` array of shape ``(2, num_layers, lstm_hidden)``.
    """
    h, c = hidden
    # Drop the batch axis (== 1 during rollout) and stack h/c on a new axis 0.
    h_np = h.detach().squeeze(1).cpu().numpy()  # (num_layers, lstm_hidden)
    c_np = c.detach().squeeze(1).cpu().numpy()  # (num_layers, lstm_hidden)
    return np.stack((h_np, c_np), axis=0).astype(np.float32)


def collect_episode(
    env: EnvProtocol,
    policy: RolloutPolicy,
    *,
    max_steps: Optional[int],
    episode_index: int,
    epsilon: float,
    episode_seed: int,
    opponent: Optional[EpisodeOpponent] = None,
) -> Episode:
    """Roll ONE episode against ``env`` with ``policy`` and return it as an Episode.

    Reentrant and side-effect-free with respect to any shared learner state: it
    writes nothing to a replay buffer, bumps no episode counter, and records no ε.
    ε and the per-episode seed are PASSED IN (so the caller owns the schedule and
    the per-arena seed scheme), and the collected episode is returned by value for
    the caller / learner to enqueue or store. The Episode's ``arena_id`` /
    ``policy_version`` / ``code_version`` are copied from ``policy``.

    The rollout mirrors the original single-arena loop exactly: reseed the action
    RNG and the env from ``episode_seed``, init the LSTM hidden, then for each
    decision snapshot the hidden state SEEN by this step (for R2D2 burn-in
    seeding), pick an action under ε with the policy's own RNG, step the env, and
    append the ``(obs, action, reward, next_obs, done)`` 5-tuple plus the snapshot.

    Putting the net into eval mode for the rollout (and restoring its prior mode)
    is the POLICY's responsibility, not this function's: the single-arena wrapper
    toggles ``self.online.eval()`` / ``train()`` around the call so the action
    stream stays byte-identical, and the distributed snapshot policy is always in
    eval mode.

    Args:
        env: The environment to roll out against (real or fake).
        policy: The acting surface (see :class:`RolloutPolicy`).
        max_steps: Optional hard cap on decisions this episode (defence against a
            fake env that never terminates). ``None`` relies on the env's own
            termination.
        episode_index: The episode index used to reset the env (forwarded to
            ``env.reset(seed=episode_seed)`` semantics; kept explicit so the caller
            owns numbering). Currently informational for callers / logging.
        epsilon: The ε-greedy rate to act under for the whole episode (flat within
            an episode, per the per-EPISODE schedule).
        episode_seed: Deterministic seed for BOTH the env reset and the policy's
            action RNG, so the exploration stream is replayable.
        opponent: Optional per-arena :class:`EpisodeOpponent` (T12). ``None`` (the
            default) is the M1/M2 stationary-dummy path and is BYTE-IDENTICAL to
            what this loop did before the parameter existed: ``env.step(action)``
            is called with one positional argument, no ``opp_action`` reaches the
            wire, and the env is never asked for a raw view. When given, each
            decision reads ``env.raw_opponent_view()``, asks the opponent for a
            macro, and sends both actions in ONE ``env.step`` — see the
            one-step-one-window invariant in this section's banner. The env must
            then expose ``raw_opponent_view()`` and accept ``opp_action``
            (``MCPvPEnv`` does; a fake env used with an opponent must too).

    Returns:
        An immutable :class:`~distributed.serialization.Episode` whose
        ``transitions`` / ``hidden_states`` are built exactly as the original loop
        built them (same dtypes, shapes, and tuple structure) and whose
        ``total_reward`` is the summed per-step reward.
    """
    # ``episode_index`` is accepted to keep episode numbering an explicit caller
    # responsibility (and for future per-arena logging); the deterministic stream
    # is fully pinned by ``episode_seed``, which the caller derives from it.
    del episode_index

    # Deterministic per-episode seeds: reseed the env reset and the policy's action
    # RNG so the exploration stream is replayable (reseed on the episode boundary,
    # not per step — the documented gotcha).
    policy.reseed(episode_seed)
    obs = env.reset(seed=episode_seed)
    hidden = policy.init_hidden()
    # Episode boundary for the opponent: draw this episode's difficulty tier and
    # start its bot's episode. AFTER env.reset(), because the reset re-arms the
    # opponent's shadow swing meter that the bot's ATTACK gate reads.
    if opponent is not None:
        opponent.begin_episode()

    transitions: List[Tuple[np.ndarray, int, float, np.ndarray, bool]] = []
    hidden_states: List[np.ndarray] = []

    total_reward = 0.0
    done = False
    steps = 0
    last_info: Optional[Mapping[str, Any]] = None
    while not done:
        # Capture the hidden state SEEN by this step (the LSTM state that produced
        # the action), stacked (num_layers, hidden) for burn-in seeding by the
        # replay buffer.
        snapshot = hidden_snapshot(hidden)

        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=_policy_device(hidden))
        action, hidden = policy.act(obs_tensor, hidden, epsilon)

        if opponent is None:
            # The M1/M2 line, unchanged: one positional argument, no opp_action.
            next_obs, reward, done, info = env.step(action)
        else:
            # ONE view, ONE macro, ONE step — the decision window the env counts.
            # The view is read here (not cached from an earlier step) so the
            # opponent decides from the same state the agent just acted on, and
            # its clamped attack_cooldown reaches the bot untouched.
            opp_action = opponent.act(env.raw_opponent_view())
            next_obs, reward, done, info = env.step(action, opp_action=opp_action)
        last_info = info

        transitions.append(
            (
                np.asarray(obs, dtype=np.float32),
                int(action),
                float(reward),
                np.asarray(next_obs, dtype=np.float32),
                bool(done),
            )
        )
        hidden_states.append(snapshot)

        total_reward += float(reward)
        obs = next_obs
        steps += 1
        if max_steps is not None and steps >= max_steps:
            break

    # Score the finished episode into the curriculum. The FINAL step's info holds
    # the terminal verdict (``won`` / ``lost`` / ``timeout``); an episode stopped
    # by ``max_steps`` carries won=False, which is the right reading — it did not
    # win. The loop body always runs at least once, so ``last_info`` is only None
    # in a pathological env.
    if opponent is not None:
        opponent.observe_outcome(last_info if last_info is not None else {})

    return Episode(
        transitions=transitions,
        hidden_states=hidden_states,
        arena_id=int(policy.arena_id),
        policy_version=int(policy.policy_version),
        code_version=str(policy.code_version),
        total_reward=total_reward,
    )


def _policy_device(hidden: Tuple[torch.Tensor, torch.Tensor]) -> torch.device:
    """Device the observation tensor should live on for a rollout step.

    Read off the carried LSTM hidden state so the free rollout never needs to know
    the policy's device directly. The original loop built ``obs_tensor`` on the
    trainer's device, which is the same device ``init_hidden`` placed the hidden
    state on, so this is byte-identical for the single-arena path.
    """
    return hidden[0].device


# ---------------------------------------------------------------------------
# The trainer.
# ---------------------------------------------------------------------------


class _TrainerOnlinePolicy:
    """One-shot :class:`RolloutPolicy` adapter over a Trainer's online net.

    Binds the free :func:`collect_episode` to a single ``DuelingDRQN`` and the
    trainer's per-episode action ``torch.Generator``. It is intentionally inert on
    the Episode-stamp metadata: for the single-arena path the arena id, policy
    version, and code version do not affect what lands in the replay buffer (the
    learner reads ``transitions`` / ``hidden_states`` only), so they are pinned to
    neutral constants and never participate in N=1 reproduction.

    The eval/train mode toggle around the rollout lives in the caller
    (:meth:`Trainer.collect_episode`), not here, so this adapter does not change
    the net's mode itself.
    """

    #: N=1 has a single arena; the stamp is inert for replay storage.
    arena_id: int = 0
    #: N=1 has no weight-snapshot versioning; inert for replay storage.
    policy_version: int = 0
    #: N=1 never crosses an actor/learner build boundary; inert for replay storage.
    code_version: str = ""

    def __init__(
        self,
        net: DuelingDRQN,
        generator: torch.Generator,
        device: torch.device,
    ) -> None:
        self._net = net
        self._generator = generator
        self._device = device

    def reseed(self, episode_seed: int) -> None:
        """Re-seed the shared action generator for this episode."""
        self._generator.manual_seed(int(episode_seed))

    def init_hidden(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the online net's zeroed single-sequence LSTM hidden state."""
        return self._net.init_hidden(1, device=self._device)

    def act(
        self,
        obs: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor],
        epsilon: float,
    ) -> Tuple[int, Tuple[torch.Tensor, torch.Tensor]]:
        """Select one action via the online net using the shared action RNG."""
        return self._net.act(
            obs,
            hidden,
            epsilon=epsilon,
            generator=self._generator,
        )


class Trainer:
    """n-step Double-DQN learner for the Dueling-DRQN with PER + R2D2 burn-in.

    Owns the online net, the target net (a soft-tracked copy), the optimizer, and
    the prioritized sequence replay. :meth:`collect_episode` rolls one episode and
    stores it; :meth:`learn` performs ONE gradient step from a sampled batch;
    :meth:`train` runs the full episode/step loop and fires the eval/checkpoint/log
    hooks at their configured cadences.

    The online and target nets are constructed identically (same architecture
    kwargs) so a soft update is well defined parameter-for-parameter.

    Args:
        cfg: Training hyperparameters. The sequence/burn-in geometry and
            n-step/gamma are forwarded to the replay buffer so storage and learner
            agree.
        device: Torch device for the nets/tensors. Defaults to CPU (the dev box
            has a CPU-only torch wheel).
        net_kwargs: Optional architecture overrides forwarded to BOTH
            ``DuelingDRQN`` constructions (e.g. smaller hidden sizes for a fast
            unit test). ``obs_dim``/``n_actions`` still assert against the frozen
            contracts inside the net. A ``cfg.warm_start`` checkpoint must match
            this architecture — the load is ``strict=True``.
        seed_global: When True (default) call :func:`agent.seeding.seed_everything`
            with ``cfg.seed`` at construction so the whole run is reproducible.
    """

    def __init__(
        self,
        cfg: TrainConfig,
        *,
        device: Optional[torch.device] = None,
        net_kwargs: Optional[Dict[str, int]] = None,
        seed_global: bool = True,
    ) -> None:
        self.cfg = cfg
        self.device = torch.device("cpu") if device is None else torch.device(device)

        # Reproducible run: seed Python/NumPy/torch before any net init draws RNG.
        if seed_global:
            seed_everything(cfg.seed)

        net_kwargs = dict(net_kwargs or {})
        self.online = DuelingDRQN(**net_kwargs).to(self.device)
        self.target = DuelingDRQN(**net_kwargs).to(self.device)

        # WARM START: replace the fresh init with the checkpoint's weights BEFORE
        # the target is copied, so θ_target == θ_online == the loaded policy. A
        # target left at its random init would spend the first thousands of steps
        # bootstrapping the loaded net toward noise — the same "warm start thrown
        # away" failure as leaving ε at 1.0, one layer down. strict=True: a shape
        # or key mismatch (a checkpoint from a different architecture) must abort
        # the run, never load a partial policy.
        if cfg.warm_start is not None:
            state_dict = load_checkpoint_state_dict(
                str(cfg.warm_start), map_location=self.device
            )
            self.online.load_state_dict(state_dict)
        # Target starts as an exact copy of the online net (θ_target = θ_online).
        self.target.load_state_dict(self.online.state_dict())
        # The target net is never optimized directly — only soft-updated. Freezing
        # its grads avoids building a needless autograd graph through it.
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.target.eval()

        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=cfg.lr)

        # Replay geometry MUST mirror the config (seq_len == scored L, burn_in,
        # n_step, gamma) so sampled windows match what the loss expects. Seed the
        # buffer's sampler from cfg.seed for reproducible PER draws.
        self.replay = PrioritizedSequenceReplay(
            capacity=cfg.replay_capacity,
            seq_len=cfg.seq_len,
            burn_in=cfg.burn_in,
            alpha=cfg.per_alpha,
            beta0=cfg.per_beta0,
            priority_eps=cfg.per_priority_eps,
            n_step=cfg.n_step,
            gamma=cfg.gamma,
            beta_anneal_steps=cfg.per_beta_anneal_steps,
            rng=np.random.default_rng(cfg.seed),
        )

        # Per-episode action-sampling RNG. Re-seeded deterministically each
        # episode (seed + episode) so exploration is replayable.
        self._action_generator = torch.Generator(device=self.device)

        self.grad_step = 0  # number of completed gradient steps
        self.episode_count = 0  # number of collected episodes
        # ε of the most recent episode. Seeded from the EFFECTIVE schedule start
        # (the warm-start restart when one is configured) so the very first
        # progress line and metrics row report the ε the run will actually use.
        self.last_epsilon = effective_eps_start(cfg)

    # ------------------------------------------------------------------
    # Rollout / collection
    # ------------------------------------------------------------------
    def collect_episode(
        self, env: EnvProtocol, *, max_steps: Optional[int] = None
    ) -> Tuple[int, float]:
        """Roll ONE episode with per-episode ε-greedy and store it in replay.

        Reseeds the env and the action-sampling generator deterministically from
        ``cfg.seed + episode_index`` so the rollout is reproducible. Each decision
        uses ``DuelingDRQN.act`` (no grad) with the episode's ε and the carried
        LSTM hidden state; the per-step hidden state captured at COLLECTION time is
        stored alongside the transition so the buffer can seed burn-in (R2D2).

        Args:
            env: The environment to roll out against (real or fake).
            max_steps: Optional hard cap on decisions this episode (defence
                against a fake env that never terminates). ``None`` relies on the
                env's own termination.

        Returns:
            ``(n_transitions, total_reward)`` for the collected episode.
        """
        episode_index = self.episode_count
        epsilon = epsilon_for_episode(episode_index, self.cfg)
        self.last_epsilon = epsilon

        # Deterministic per-episode seed for the env reset and the action RNG so
        # the exploration stream is replayable (the gotcha fix: reseed on the
        # episode boundary, not per step). For N=1 (arena 0, local_ep ==
        # episode_index) this is exactly ``cfg.seed + episode_index``.
        episode_seed = arena_episode_seed(self.cfg, arena_id=0, local_ep=episode_index)

        # One-shot adapter binding the free rollout to this trainer's online net
        # and its per-episode action generator. The eval-mode toggle stays HERE so
        # the inference-mode action stream is byte-identical to the original loop:
        # ``self.online.act`` short-circuits the RNG when ε == 0, so both the
        # generator reseed and the eval/train mode must match exactly.
        adapter = _TrainerOnlinePolicy(self.online, self._action_generator, self.device)

        was_training = self.online.training
        self.online.eval()  # inference mode for action selection
        try:
            episode = collect_episode(
                env,
                adapter,
                max_steps=max_steps,
                episode_index=episode_index,
                epsilon=epsilon,
                episode_seed=episode_seed,
            )
        finally:
            if was_training:
                self.online.train()

        # Re-add the original direct replay write (the free rollout deliberately
        # leaves storage to the caller). Guard on a non-empty episode as before.
        if episode.transitions:
            self.replay.add_episode(
                episode.transitions, hidden_states=episode.hidden_states
            )

        self.episode_count += 1
        return len(episode.transitions), episode.total_reward

    def _hidden_snapshot(self, hidden: Tuple[torch.Tensor, torch.Tensor]) -> np.ndarray:
        """Snapshot the LSTM ``(h, c)`` for the current single-sequence step.

        Returns a contiguous ``float32`` array of shape
        ``(2, num_layers, lstm_hidden)`` (the (h, c) pair) detached to NumPy. The
        replay buffer stores one per transition and hands the window-start snapshot
        back so T16 can seed burn-in instead of zeroing it.
        """
        h, c = hidden
        # Drop the batch axis (== 1 during rollout) and stack h/c on a new axis 0.
        h_np = h.detach().squeeze(1).cpu().numpy()  # (num_layers, lstm_hidden)
        c_np = c.detach().squeeze(1).cpu().numpy()  # (num_layers, lstm_hidden)
        return np.stack((h_np, c_np), axis=0).astype(np.float32)

    # ------------------------------------------------------------------
    # Learning — one gradient step
    # ------------------------------------------------------------------
    def ready_to_learn(self) -> bool:
        """True iff replay has >= ``min_replay`` transitions AND a sampleable window."""
        return self.replay.is_ready(self.cfg.min_replay)

    def learn(self) -> Optional[LearnStats]:
        """Run ONE n-step Double-DQN gradient step from a sampled batch.

        Returns ``None`` (a no-op) if the buffer is not yet ready (``len(replay)
        < min_replay`` or no sampleable window). Otherwise:

          1. sample a prioritized batch of ``B + L`` windows,
          2. compute the IS-weighted Huber loss on the scored ``L`` steps using
             the n-step Double-DQN target,
          3. backward, clip the global grad norm to ``cfg.grad_clip``, step,
          4. write the fresh |δ| back as priorities, and
          5. soft-update the target net.

        Returns:
            A :class:`LearnStats` for this gradient step, or ``None`` if not ready.
        """
        if not self.ready_to_learn():
            return None

        cfg = self.cfg
        # Anneal PER β by gradient step BEFORE sampling so this batch's IS weights
        # use the up-to-date β.
        self.replay.anneal_beta(self.grad_step)

        batch = self.replay.sample_sequences(cfg.batch_size, L=cfg.seq_len)

        loss, abs_td_per_sample, mean_abs_td = self._compute_loss(batch)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.online.parameters(), cfg.grad_clip
        )
        self.optimizer.step()

        # Priorities: one per sampled window, from the latest |δ| (mean over the
        # window's scored steps — a single priority per window is what the buffer
        # stores per start index).
        self.replay.update_priorities(batch.indices, abs_td_per_sample)

        # Soft target update every step: θ_target ← τ·θ_online + (1−τ)·θ_target.
        if cfg.target_soft:
            self._soft_update_target()

        self.grad_step += 1

        return LearnStats(
            grad_step=self.grad_step,
            loss=float(loss.detach().cpu().item()),
            td_error_mean=float(mean_abs_td),
            grad_norm=float(grad_norm),
            epsilon=self.last_epsilon,
            beta=float(self.replay.beta),
            replay_size=len(self.replay),
        )

    def _compute_loss(
        self, batch: SequenceBatch
    ) -> Tuple[torch.Tensor, np.ndarray, float]:
        """Build the IS-weighted Huber loss for one sampled batch.

        Implements the n-step Double-DQN target exactly (see the module docstring):
        ONLINE selects the bootstrap action, TARGET evaluates it, the burn-in
        prefix is detached, and the loss is computed on the scored ``L`` steps
        only.

        Returns:
            ``(loss, abs_td_per_sample, mean_abs_td)`` where ``loss`` is the scalar
            graph tensor, ``abs_td_per_sample`` is a ``(batch,)`` NumPy array of
            per-window mean |δ| (for ``update_priorities``), and ``mean_abs_td`` is
            the scalar batch-mean |δ| (for logging).
        """
        cfg = self.cfg
        device = self.device
        burn_in = batch.burn_in
        # Full window length T = burn_in + seq_len (the buffer guarantees this).
        obs = torch.as_tensor(batch.obs, dtype=torch.float32, device=device)
        next_obs = torch.as_tensor(batch.next_obs, dtype=torch.float32, device=device)
        actions = torch.as_tensor(batch.actions, dtype=torch.long, device=device)
        is_weights = torch.as_tensor(
            batch.is_weights, dtype=torch.float32, device=device
        )

        bsz, window, _ = obs.shape
        scored = window - burn_in  # == seq_len (L)

        # --- n-step discounted returns + bootstrap mask (network-free) -----
        # Shapes (batch, window). G_i and the mask are aligned with absolute step
        # index i in [0, window); we use only the scored slice [burn_in:].
        returns_np, bootstrap_np = self.replay.n_step_returns(
            batch, n=cfg.n_step, gamma=cfg.gamma
        )
        returns = torch.as_tensor(returns_np, dtype=torch.float32, device=device)
        bootstrap = torch.as_tensor(
            bootstrap_np.astype(np.float32), dtype=torch.float32, device=device
        )

        # --- seed hidden state from the stored collection-time hidden ------
        # batch.hidden is (batch, 2, num_layers, lstm_hidden) when present (our
        # collector stores the (h, c) pair). Seed the very first window step.
        seed_hidden = self._seed_hidden_from_batch(batch, bsz)

        # --- online Q on the scored steps (gradients flow here) ------------
        q_scored, _, _ = self.online.forward_with_burn_in(
            obs, burn_in, hidden=seed_hidden
        )  # (batch, L, N_ACTIONS)
        scored_actions = actions[:, burn_in:]  # (batch, L)
        q_taken = q_scored.gather(-1, scored_actions.unsqueeze(-1)).squeeze(-1)
        # (batch, L) — Q_online(s_i, a_i) for scored i.

        # --- bootstrap states: a SINGLE correctly-recurrent forward --------
        # The bootstrap state for scored step i (absolute window index i in
        # [B, T)) is s_{i+n}. For a DRQN the bootstrap Q must be evaluated with the
        # hidden state that incorporates the FULL contiguous history up to s_{i+n},
        # NOT a hidden seeded from the wrong stream (the old code ran the LSTM over
        # ``next_obs`` from the window-START hidden, which is shifted by one step
        # and missing obs[B] — biasing every n-step target whenever memory matters).
        #
        # Within an episode the window is contiguous, so next_obs[:, :-1] == obs[:,
        # 1:]. Build an EXTENDED stream obs_ext = concat(obs, next_obs[:, -1:]) of
        # shape (b, T+1, OBS_DIM): obs_ext[p] == s_p for p in [0, T] (obs_ext[T] ==
        # next_obs[T-1] == s_T). Running ONE seeded forward over obs_ext from the
        # collection-time window-start hidden makes the Q at position p use the
        # hidden that incorporates s_0..s_p — exactly the correct memory for s_p.
        # The bootstrap (a target) carries no gradient, so the whole pass is under
        # no_grad.
        with torch.no_grad():
            obs_ext = torch.cat([obs, next_obs[:, -1:, :]], dim=1)  # (b, T+1, OBS_DIM)
            q_ext_online, _ = self.online(obs_ext, seed_hidden)  # (b, T+1, A)
            q_ext_target, _ = self.target(obs_ext, seed_hidden)  # (b, T+1, A)
            # Double-DQN per position: ONLINE selects a*, TARGET evaluates it.
            next_actions = q_ext_online.argmax(dim=-1)  # (b, T+1)
            q_ext_eval = q_ext_target.gather(
                -1, next_actions.unsqueeze(-1)
            ).squeeze(-1)  # (b, T+1) target value of the online-greedy action at s_p

            # For scored step i the bootstrap state s_{i+n} sits at obs_ext
            # position i + n. bootstrap_i is False whenever that horizon ran off the
            # window or a done truncated the return, so the clamped (in-range)
            # gather there is masked to 0.
            boot_value = self._gather_bootstrap_values(
                q_ext_eval, burn_in, scored, cfg.n_step
            )  # (batch, L)

            returns_scored = returns[:, burn_in:]  # (batch, L)
            bootstrap_scored = bootstrap[:, burn_in:]  # (batch, L)
            gamma_n = cfg.gamma ** cfg.n_step
            target_y = returns_scored + bootstrap_scored * gamma_n * boot_value
            # (batch, L)

        # --- TD error + IS-weighted Huber loss -----------------------------
        # Derive BOTH the priority |δ| and the Huber loss from the SAME ``td_error``
        # tensor so the value that drives the update can never diverge from the one
        # that drives the priority (Huber(δ) is a deterministic function of δ).
        td_error = target_y - q_taken  # (batch, L)
        # smooth-L1 (Huber) per scored step, no reduction, computed FROM td_error.
        huber = F.smooth_l1_loss(
            td_error, torch.zeros_like(td_error), reduction="none"
        )  # (batch, L)
        # Mean Huber per window, then weight by the window's IS weight, then mean.
        per_window_huber = huber.mean(dim=1)  # (batch,)
        loss = (is_weights * per_window_huber).mean()

        # Per-window |δ| (mean over scored steps) drives the new priority; detach.
        abs_td = td_error.detach().abs()  # (batch, L)
        abs_td_per_sample = abs_td.mean(dim=1).cpu().numpy()  # (batch,)
        mean_abs_td = float(abs_td.mean().cpu().item())

        return loss, abs_td_per_sample, mean_abs_td

    def _seed_hidden_from_batch(
        self, batch: SequenceBatch, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build the LSTM seed ``(h0, c0)`` from the stored window-start hidden.

        The collector stores a per-step ``(2, num_layers, lstm_hidden)`` snapshot;
        the buffer returns the window-start one as ``batch.hidden`` of shape
        ``(batch, 2, num_layers, lstm_hidden)``. We split it back into the
        ``(num_layers, batch, lstm_hidden)`` layout ``torch.nn.LSTM`` expects. When
        no hidden was stored (``batch.hidden is None``) we fall back to a zero seed
        — an acceptable burn-in fallback per the net's docstring.
        """
        if batch.hidden is None:
            return self.online.init_hidden(batch_size, device=self.device)

        hidden = np.asarray(batch.hidden, dtype=np.float32)
        # Expected layout (batch, 2, num_layers, lstm_hidden). Be defensive about a
        # collapsed num_layers axis (batch, 2, lstm_hidden) for a single-layer net.
        if hidden.ndim == 3:
            hidden = hidden[:, :, np.newaxis, :]  # -> (batch, 2, 1, lstm_hidden)
        if hidden.ndim != 4 or hidden.shape[1] != 2:
            # Unexpected snapshot shape — fall back to a zero seed rather than risk
            # feeding a mis-shaped hidden into the LSTM.
            return self.online.init_hidden(batch_size, device=self.device)

        h = torch.as_tensor(hidden[:, 0], dtype=torch.float32, device=self.device)
        c = torch.as_tensor(hidden[:, 1], dtype=torch.float32, device=self.device)
        # (batch, num_layers, lstm_hidden) -> (num_layers, batch, lstm_hidden).
        h = h.permute(1, 0, 2).contiguous()
        c = c.permute(1, 0, 2).contiguous()
        return h, c

    @staticmethod
    def _gather_bootstrap_values(
        q_ext_eval: torch.Tensor, burn_in: int, scored: int, n_step: int
    ) -> torch.Tensor:
        """Select, per scored step ``i``, the bootstrap value at ``obs_ext`` ``i+n``.

        ``q_ext_eval`` is ``(batch, T+1)`` — the target value of the online-greedy
        action at EVERY extended-stream position (``obs_ext[p]`` == state ``s_p``).
        For scored step ``i`` (absolute index ``burn_in + s`` for ``s`` in
        ``[0, scored)``) the bootstrap state ``s_{i+n}`` is ``obs_ext[i + n]``. We
        clamp that index into ``[0, T]`` and rely on the caller's ``bootstrap`` mask
        to zero any term where the true horizon ran past the window end, so the
        clamped (in-range) gather is never actually used there.

        Returns:
            ``(batch, scored)`` bootstrap values aligned with the scored steps.
        """
        bsz, ext_len = q_ext_eval.shape  # ext_len == T + 1
        # Absolute scored indices i = burn_in + [0..scored).
        i = torch.arange(scored, device=q_ext_eval.device) + burn_in  # (scored,)
        boot_idx = i + n_step  # obs_ext position of s_{i+n}
        # Non-mutating clamp into the extended stream (max valid index is T == ext_len-1).
        boot_idx = boot_idx.clamp(max=ext_len - 1)  # keep gather in range (masked)
        boot_idx = boot_idx.unsqueeze(0).expand(bsz, scored)  # (batch, scored)
        return q_ext_eval.gather(1, boot_idx)

    def _soft_update_target(self) -> None:
        """Polyak soft update: ``θ_target ← τ·θ_online + (1−τ)·θ_target``.

        Applied to every parameter AND buffer of the target net every gradient
        step. Runs under ``no_grad`` (the target is never differentiated).
        """
        tau = self.cfg.tau
        with torch.no_grad():
            for tgt_p, src_p in zip(
                self.target.parameters(), self.online.parameters()
            ):
                tgt_p.mul_(1.0 - tau).add_(src_p, alpha=tau)
            # Buffers (none in this net, but future-proof) are copied verbatim so
            # they never go stale.
            for tgt_b, src_b in zip(self.target.buffers(), self.online.buffers()):
                tgt_b.copy_(src_b)

    # ------------------------------------------------------------------
    # Full loop
    # ------------------------------------------------------------------
    def train(
        self,
        env: EnvProtocol,
        num_episodes: int,
        *,
        updates_per_step: int = 1,
        max_episode_steps: Optional[int] = None,
        eval_hook: Optional[EvalHook] = None,
        checkpoint_hook: Optional[CheckpointHook] = None,
        log_hook: Optional[LogHook] = None,
    ) -> None:
        """Run the full episode/gradient loop for ``num_episodes`` episodes.

        For each episode: collect it (per-EPISODE ε), then — once the buffer is
        ready — run ``updates_per_step`` gradient steps. The eval/checkpoint/log
        hooks fire at their configured cadences (``cfg.eval_interval`` etc.); a
        cadence of 0 disables that hook. The hooks are call-signature stubs here:
        T19 owns eval, T20 owns checkpoint/full training.

        Args:
            env: The environment to roll out against.
            num_episodes: Number of episodes to collect/train on (>= 0).
            updates_per_step: Gradient steps to run per collected episode once
                ready (>= 0). 1 keeps the update-to-data ratio near one update per
                episode; raise for more gradient steps per env episode.
            max_episode_steps: Optional per-episode decision cap (for a fake env).
            eval_hook / checkpoint_hook / log_hook: Optional callbacks; see the
                hook type aliases. Called only when their interval is > 0 and the
                current grad step is a multiple of it.

        Raises:
            ValueError: if ``num_episodes`` or ``updates_per_step`` is negative.
        """
        if num_episodes < 0:
            raise ValueError(f"num_episodes must be >= 0, got {num_episodes}")
        if updates_per_step < 0:
            raise ValueError(
                f"updates_per_step must be >= 0, got {updates_per_step}"
            )

        for _ in range(num_episodes):
            self.collect_episode(env, max_steps=max_episode_steps)

            for _ in range(updates_per_step):
                stats = self.learn()
                if stats is None:
                    break  # not ready yet; skip remaining updates this episode
                self._fire_hooks(stats, eval_hook, checkpoint_hook, log_hook)

    def _fire_hooks(
        self,
        stats: LearnStats,
        eval_hook: Optional[EvalHook],
        checkpoint_hook: Optional[CheckpointHook],
        log_hook: Optional[LogHook],
    ) -> None:
        """Invoke eval/checkpoint/log hooks whose interval divides the grad step."""
        step = self.grad_step
        if log_hook is not None and self.cfg.log_interval > 0:
            if step % self.cfg.log_interval == 0:
                log_hook(self, step, stats)
        if eval_hook is not None and self.cfg.eval_interval > 0:
            if step % self.cfg.eval_interval == 0:
                eval_hook(self, step)
        if checkpoint_hook is not None and self.cfg.checkpoint_interval > 0:
            if step % self.cfg.checkpoint_interval == 0:
                checkpoint_hook(self, step)


# ---------------------------------------------------------------------------
# Functional entry point.
# ---------------------------------------------------------------------------


def train(
    env: EnvProtocol,
    cfg: TrainConfig,
    num_episodes: int,
    *,
    device: Optional[torch.device] = None,
    net_kwargs: Optional[Dict[str, int]] = None,
    updates_per_step: int = 1,
    max_episode_steps: Optional[int] = None,
    eval_hook: Optional[EvalHook] = None,
    checkpoint_hook: Optional[CheckpointHook] = None,
    log_hook: Optional[LogHook] = None,
) -> Trainer:
    """Construct a :class:`Trainer` and run ``num_episodes`` of the §8 loop.

    Convenience wrapper for callers that just want "train this env with this
    config". Returns the constructed :class:`Trainer` so the caller can inspect
    the nets / replay / grad-step count afterwards (eval is T19, checkpoint T20).

    Args:
        env: The environment to roll out against (real ``MCPvPEnv`` or a fake).
        cfg: The training hyperparameters.
        num_episodes: Episodes to collect/train on.
        device / net_kwargs: Forwarded to the :class:`Trainer` constructor.
        updates_per_step / max_episode_steps / *_hook: Forwarded to
            :meth:`Trainer.train`.

    Returns:
        The :class:`Trainer` after the loop completes.
    """
    trainer = Trainer(cfg, device=device, net_kwargs=net_kwargs)
    trainer.train(
        env,
        num_episodes,
        updates_per_step=updates_per_step,
        max_episode_steps=max_episode_steps,
        eval_hook=eval_hook,
        checkpoint_hook=checkpoint_hook,
        log_hook=log_hook,
    )
    return trainer


# ===========================================================================
# T20 — the M2 integration entrypoint: train the real DRQN vs the stationary
# dummy over the real perception+reward env, with periodic greedy eval against
# the M2 gate.
#
# This composes the WHOLE stack that the other tracks built:
#
#   seed_everything
#     -> MCPvPEnv(transport, ...)            # real PerceptionFilter + reward
#        over a StationaryDummy opponent     # served by the bridge/server
#     -> Trainer(cfg)                        # online/target DuelingDRQN +
#                                            # PrioritizedSequenceReplay
#     -> per-episode loop (per-EPISODE ε)    # Trainer.collect + Trainer.learn
#        with periodic evaluate(...)         # greedy ε=0, the M2 gate (AC6)
#        logging EACH reward component       # via MetricsLogger
#        + checkpoint hooks
#     -> stop at the M2 gate (passed_m2)     # win_rate>=0.95 AND
#        OR a max-step / max-episode budget. #   aim_while_invisible==0 AND
#                                            #   mean_len < cap
#
# The transport is INJECTABLE (``transport_factory``) so this exact loop runs
# OFFLINE against a fake/scripted bridge in tests — the offline end-to-end proof.
# The live M2 run wires the real ``TcpBridgeClient`` via ``main()`` below.
#
# WHAT THIS DOES NOT DO (the documented human follow-up):
#   AC6/TC13 proper — "greedy DRQN >= 95% over 100 eval eps vs the LIVE
#   stationary dummy" — needs the live Paper server + Node bridge and a real
#   training budget. That is the human M2 run on the laptop+Paper stack; see
#   server/README.md ("Live follow-up") and server/compat_check.md for the
#   live-handshake follow-ups. This task delivers the full integration WIRING
#   plus the offline end-to-end proof (tests/test_integration_m2.py).
# ===========================================================================


#: Factory that returns a fresh bridge transport (the env owns its lifecycle).
#: The real one is ``lambda: TcpBridgeClient(host, port)``; tests inject one that
#: returns a scripted fake bridge. A factory (not an instance) is taken so the M2
#: entrypoint stays symmetric with the live CLI, which constructs the transport
#: lazily, and so a future reconnect could rebuild it.
TransportFactory = Callable[[], "Any"]


@dataclass
class M2Result:
    """Outcome of an :func:`train_vs_dummy` / :func:`run_m2` integration run.

    Attributes:
        trainer: The :class:`Trainer` after the loop (online/target nets, replay,
            grad-step count — inspectable by the caller / a checkpoint).
        passed_m2: ``True`` iff the most recent eval cleared the full M2 gate
            (``win_rate >= 0.95`` AND ``aim_while_invisible == 0`` AND
            ``mean_episode_length < timeout_cap``). ``False`` if the budget ran
            out first or no eval ran.
        episodes_run: Number of episodes actually collected before stopping.
        grad_steps: Number of gradient steps completed.
        last_report: The most recent eval :class:`~eval.evaluate.EvalReport`
            (``None`` if no eval ran — e.g. ``eval_every_episodes == 0``).
        reports: Every eval report produced, in order (for plotting the win-rate
            curve over training).
        stop_reason: One of ``"passed_m2"`` / ``"max_episodes"`` / ``"max_steps"``
            — why the loop stopped.
        is_live: ``True`` for a real-bridge run, ``False`` for the offline proof.
    """

    trainer: "Trainer"
    passed_m2: bool
    episodes_run: int
    grad_steps: int
    last_report: Optional[Any] = None
    reports: List[Any] = None  # type: ignore[assignment]  # set in __post_init__
    stop_reason: str = "max_episodes"
    is_live: bool = False

    def __post_init__(self) -> None:
        if self.reports is None:
            self.reports = []


def train_vs_dummy(
    cfg: TrainConfig,
    *,
    transport_factory: TransportFactory,
    max_episodes: int = 10_000,
    max_grad_steps: Optional[int] = None,
    updates_per_step: int = 1,
    eval_every_episodes: int = 50,
    eval_episodes: int = 100,
    timeout_cap: int = MAX_EPISODE_STEPS,
    env_max_episode_steps: int = MAX_EPISODE_STEPS,
    rollout_step_cap: Optional[int] = None,
    stop_on_pass: bool = True,
    device: Optional[torch.device] = None,
    net_kwargs: Optional[Dict[str, int]] = None,
    logger: Optional[Any] = None,
    checkpoint_hook: Optional[CheckpointHook] = None,
    is_live: bool = False,
    log: Optional[Callable[[str], None]] = None,
    show_progress: bool = True,
    progress_log_interval: float = 30.0,
    progress_stream: Optional[Any] = None,
    progress_reporter: Optional[ProgressReporter] = None,
) -> M2Result:
    """Train the DRQN vs the stationary dummy until the M2 gate or a budget.

    The full M2 composition (see the section banner above). One ``Trainer`` owns
    the online/target :class:`~agent.dqn.DuelingDRQN` and the
    :class:`~agent.replay.PrioritizedSequenceReplay`; this function drives the
    episode loop, runs a GREEDY :func:`~eval.evaluate.evaluate` against the dummy
    every ``eval_every_episodes`` episodes (logging EACH reward component via the
    ``logger``), fires the ``checkpoint_hook``, and STOPS the moment an eval clears
    the gate (``report.passed_m2``) — or when the episode / gradient-step budget is
    exhausted.

    The agent and the eval BOTH face the same stage-0 opponent: the
    :class:`~opponents.dummy.StationaryDummy`. The dummy is served by the bridge /
    Paper server (its idle policy + immunity flags are enforced server-side), so
    this loop never steps an opponent policy in Python — it just talks to the env.

    Determinism: ``Trainer`` seeds Python/NumPy/torch from ``cfg.seed`` at
    construction (``seed_everything``), and per-episode env resets + ε RNG are
    reseeded from ``cfg.seed + episode_index`` inside ``collect_episode`` — so the
    whole run is replayable.

    Args:
        cfg: Training hyperparameters (net/replay geometry, ε schedule, PER, etc.).
        transport_factory: Zero-arg callable returning a fresh bridge transport
            (a real :class:`~env.mc_pvp_env.TcpBridgeClient` live; a scripted fake
            offline). The env owns the returned transport's lifecycle.
        max_episodes: Hard cap on episodes collected (the episode budget).
        max_grad_steps: Optional hard cap on gradient steps (the compute budget);
            ``None`` relies on ``max_episodes`` alone.
        updates_per_step: Gradient steps per collected episode once the replay is
            warm (forwarded to the learner).
        eval_every_episodes: Run a greedy eval after every this-many episodes
            (``0`` disables periodic eval — then the gate is never checked and the
            loop runs the full episode budget).
        eval_episodes: Episodes per greedy eval. AC6 uses 100; tests use a few.
        timeout_cap: The episode-length horizon a timeout hits, passed to
            ``evaluate`` (its run-away guard requires ``mean_len < timeout_cap``).
        env_max_episode_steps: Per-episode decision cap enforced by the env itself.
        rollout_step_cap: Optional belt-and-suspenders per-episode decision cap for
            COLLECTION (defends against a fake env that never sets ``done``);
            ``None`` relies on the env's own ``max_episode_steps``.
        stop_on_pass: When True (default) stop as soon as an eval clears the M2
            gate. Set False to run the whole budget regardless (e.g. to collect the
            full win-rate curve).
        device / net_kwargs: Forwarded to the :class:`Trainer` constructor.
        logger: Optional :class:`~eval.logging.MetricsLogger`. Per-episode training
            stats AND every periodic eval's per-component breakdown are logged
            through it. ``None`` disables logging (the report is still computed).
        checkpoint_hook: Optional ``(trainer, grad_step) -> None`` called after the
            loop ends and after each eval that improves the win rate, so the caller
            can persist the net + ``code_version``. (The cadence-driven
            ``Trainer``-internal checkpoint hook is separate; this is the M2-level
            "save best / save final" hook.)
        is_live: Marks the produced reports/result as a live (vs offline) run.
        log: Optional ``str -> None`` progress sink (``None`` silences it). When a
            progress reporter is active the loop's own eval-summary lines are
            routed through it instead (so they never garble the in-place bar).
        show_progress: When True (default) attach a
            :class:`~agent.progress.ProgressReporter` — a live status bar (on a
            TTY) plus a periodic progress LINE with throughput and a budget ETA.
            Ignored when ``progress_reporter`` is supplied.
        progress_log_interval: Seconds between persistent progress lines / metrics
            rows (the bar itself redraws faster on a TTY).
        progress_stream: Where the bar/lines are drawn (defaults to the reporter's
            own default, ``sys.stderr``). Ignored when ``progress_reporter`` is set.
        progress_reporter: Inject a pre-built reporter (mainly for tests); when
            given it overrides ``show_progress`` / the stream / the interval.

    Returns:
        An :class:`M2Result` with the trainer, the gate verdict, the episode /
        gradient-step counts, every eval report, and the stop reason.

    Raises:
        ValueError: on a non-positive ``max_episodes`` / ``eval_episodes`` or a
            negative ``eval_every_episodes``.
    """
    # Local import to keep the eval dependency at the call boundary (and to avoid
    # any import cycle between agent.train and eval.evaluate).
    from eval.evaluate import DRQNGreedyPolicy, evaluate

    from env.mc_pvp_env import MCPvPEnv

    if max_episodes <= 0:
        raise ValueError(f"max_episodes must be > 0, got {max_episodes}")
    if eval_episodes <= 0:
        raise ValueError(f"eval_episodes must be > 0, got {eval_episodes}")
    if eval_every_episodes < 0:
        raise ValueError(
            f"eval_every_episodes must be >= 0, got {eval_every_episodes}"
        )
    if cfg.opponent != "dummy":
        # REFUSE, do not ignore. This loop steps no opponent policy — the dummy is
        # served entirely by the bridge — so honoring cfg.opponent here would mean
        # silently training against a stationary dummy while the run's own config
        # said "scripted", which is exactly the kind of silent-substitution failure
        # this project keeps paying for. The scripted curriculum lives on the
        # multi-arena path (T12).
        raise ValueError(
            f"train_vs_dummy is the stationary-dummy loop and cannot fight "
            f"cfg.opponent={cfg.opponent!r}: it never steps an opponent policy. "
            "Use train_multi_arena (--arenas >= 2) for the scripted opponent."
        )

    def _emit(message: str) -> None:
        # Route standalone lines through the reporter when present so they clear
        # the in-place bar first (otherwise the bar and the line collide on a TTY).
        # ALSO forward to the `log` sink: the reporter writes to its own progress
        # stream, so a separate file/structured `log` must still receive the
        # run-start + eval-summary lines (routing to only one dropped them).
        if reporter is not None:
            reporter.message(message)
        if log is not None:
            log(message)

    # --- build the trainer (online/target DRQN + PER replay, seeded) --------
    trainer = Trainer(cfg, device=device, net_kwargs=net_kwargs)

    # --- build the real env over the injected transport ---------------------
    # MCPvPEnv applies the real PerceptionFilter (FOV+LoS+memory gating) and the
    # canonical reward (single-source-of-truth components) — exactly what the
    # learner sees in production. The opponent is the stationary dummy, served by
    # the bridge; the env owns the transport lifecycle.
    transport = transport_factory()
    env = MCPvPEnv(transport=transport, max_episode_steps=env_max_episode_steps)

    reports: List[Any] = []
    last_report: Optional[Any] = None
    best_win_rate = -1.0
    passed = False
    stop_reason = "max_episodes"
    episodes_run = 0
    total_steps = 0  # env transitions collected across episodes (for throughput)

    # --- live progress reporter (status bar + throughput/ETA log) -----------
    reporter = progress_reporter
    if reporter is None and show_progress:
        reporter = ProgressReporter(
            total_episodes=max_episodes,
            stream=progress_stream,
            log_interval=progress_log_interval,
            # On a TTY, tick the bar ~1/s so it doesn't look frozen across the
            # slow (~90 s) live episodes. Auto-disabled when output is redirected.
            heartbeat_interval=1.0,
        )
    if reporter is not None:
        # Immediate, redirect-safe confirmation the run is up — the first
        # throughput/ETA numbers can't appear until episode 1 finishes (~90 s live).
        eval_note = (
            f"eval every {eval_every_episodes} eps" if eval_every_episodes > 0
            else "eval disabled"
        )
        _emit(
            f"[m2] training started - budget {max_episodes} episodes, {eval_note}. "
            f"Throughput + ETA appear after episode 1 completes (~90 s live)."
        )
        reporter.start(epsilon=trainer.last_epsilon)

    def _log_training(_trainer: "Trainer", _step: int, stats: LearnStats) -> None:
        if logger is not None:
            logger.log(
                {
                    "train/loss": stats.loss,
                    "train/td_error_mean": stats.td_error_mean,
                    "train/grad_norm": stats.grad_norm,
                    "train/epsilon": stats.epsilon,
                    "train/beta": stats.beta,
                    "train/replay_size": stats.replay_size,
                },
                step=stats.grad_step,
            )

    try:
        for episode_index in range(max_episodes):
            # --- collect ONE episode (per-EPISODE ε, deterministic reseed) ---
            n_transitions, _ = trainer.collect_episode(env, max_steps=rollout_step_cap)
            episodes_run += 1
            total_steps += n_transitions

            # --- gradient steps once the replay is warm ----------------------
            for _ in range(updates_per_step):
                stats = trainer.learn()
                if stats is None:
                    break  # not ready yet; skip remaining updates this episode
                _log_training(trainer, stats.grad_step, stats)

            # --- live progress: status bar + periodic throughput/ETA line ----
            if reporter is not None:
                snap = reporter.update(
                    episodes_run=episodes_run,
                    steps_collected=total_steps,
                    grad_step=trainer.grad_step,
                    epsilon=trainer.last_epsilon,
                    last_win_rate=(
                        last_report.win_rate if last_report is not None else None
                    ),
                )
                if snap is not None and logger is not None:
                    logger.log(progress_metrics(snap), step=trainer.grad_step)

            # --- periodic GREEDY eval against the M2 gate --------------------
            do_eval = (
                eval_every_episodes > 0
                and (episode_index + 1) % eval_every_episodes == 0
            )
            if do_eval:
                if reporter is not None:
                    reporter.clear()  # drop the bar before the (long) eval output
                # Eval BORROWS the training env's (now idle) transport instead of
                # opening a second connection — the bridge serves exactly one
                # connection, so a fresh eval socket would steal the stream out from
                # under training and abort the run. See _eval_against_opponent.
                report = _eval_against_opponent(
                    trainer=trainer,
                    evaluate=evaluate,
                    policy_cls=DRQNGreedyPolicy,
                    shared_transport=transport,
                    n_episodes=eval_episodes,
                    timeout_cap=timeout_cap,
                    env_max_episode_steps=env_max_episode_steps,
                    eval_step_cap=rollout_step_cap,
                    logger=logger,
                    is_live=is_live,
                    base_seed=cfg.seed,
                    log=log,
                )
                reports.append(report)
                last_report = report

                # Save-best hook: checkpoint whenever the win rate improves.
                if report.win_rate > best_win_rate:
                    best_win_rate = report.win_rate
                    if checkpoint_hook is not None:
                        checkpoint_hook(trainer, trainer.grad_step)

                _emit(
                    f"[m2 ep {episode_index + 1}] grad_step={trainer.grad_step} "
                    f"win_rate={report.win_rate:.3f} "
                    f"mean_len={report.mean_episode_length:.1f} "
                    f"aim_invisible={report.aim_while_invisible:.3f} "
                    f"passed_m2={report.passed_m2}"
                )

                if report.passed_m2 and stop_on_pass:
                    passed = True
                    stop_reason = "passed_m2"
                    break

            # --- gradient-step budget ---------------------------------------
            if max_grad_steps is not None and trainer.grad_step >= max_grad_steps:
                stop_reason = "max_steps"
                break
        else:
            stop_reason = "max_episodes"
    finally:
        _close_quietly(env)
        if reporter is not None:
            final_snap = reporter.close()
            if final_snap is not None and logger is not None:
                logger.log(progress_metrics(final_snap), step=trainer.grad_step)

    # A final save (best/final) so the caller always has a persistable net.
    if checkpoint_hook is not None:
        checkpoint_hook(trainer, trainer.grad_step)

    return M2Result(
        trainer=trainer,
        passed_m2=passed,
        episodes_run=episodes_run,
        grad_steps=trainer.grad_step,
        last_report=last_report,
        reports=reports,
        stop_reason=stop_reason,
        is_live=bool(is_live),
    )


#: Alias — the plan offers ``train_vs_dummy`` OR ``run_m2`` as the entrypoint
#: name; expose both so either reads naturally at the call site.
run_m2 = train_vs_dummy


def _eval_against_opponent(
    *,
    trainer: "Trainer",
    evaluate: Callable[..., Any],
    policy_cls: Callable[..., Any],
    shared_transport: Any,  # the training env's bridge transport (BridgeTransport)
    n_episodes: int,
    timeout_cap: int,
    env_max_episode_steps: int,
    eval_step_cap: Optional[int],
    logger: Optional[Any],
    is_live: bool,
    base_seed: int,
    log: Optional[Callable[[str], None]],
    opponent: Optional[Any] = None,
) -> Any:
    """Run ONE greedy (ε=0) eval of the current online net vs the stage opponent.

    ``opponent`` is ``None`` on the stationary-dummy path (the bridge serves the
    dummy and no ``opp_action`` goes on the wire) and an
    :class:`EvalOpponentDriver` when the run fights the scripted opponent — in
    which case the eval steps it exactly as collection does, so the win rate this
    returns is a win rate against the SAME moving opponent training faces.

    The bridge serves EXACTLY ONE connection, so eval must not open a second one:
    a fresh eval socket adopts the stream and the bridge destroys the training
    socket, which then aborts the run on the next ``reset``. Eval therefore BORROWS
    the training env's ``shared_transport`` — safe because at the eval boundary the
    training env is genuinely idle: ``collect_episode`` has finished (its last
    ``step`` already got its ``state`` reply) and the gradient steps touch no
    socket, so no reply is in flight on the shared connection.

    The eval :class:`~env.mc_pvp_env.MCPvPEnv` is a SEPARATE instance built with
    ``auto_connect=False`` over that transport, so it has its OWN per-episode state
    (``_episode`` / PerceptionFilter / ``_prev_obs`` / ``_done``) and cannot corrupt
    the training env's. It does NOT own the socket: the training env opened it and
    closes it once in ``train_vs_dummy``'s ``finally``. Eval never sends ``close``
    or closes the transport (that would kill the live bridge mid-run); it leaves the
    connection idle for training to resume. (Honoring the design intent: eval still
    does not consume the training env's IN-FLIGHT stream — there is none in flight at
    the boundary — it just shares the one idle connection.)

    Wraps the online net in a greedy :class:`~eval.evaluate.DRQNGreedyPolicy`, runs
    :func:`~eval.evaluate.evaluate`, and returns its
    :class:`~eval.evaluate.EvalReport`.
    """
    from env.mc_pvp_env import MCPvPEnv

    policy = policy_cls(trainer.online, device=trainer.device)
    # auto_connect=False: the shared transport is already connected by the training
    # env; reconnecting would re-handshake the live bridge mid-run.
    eval_env = MCPvPEnv(
        transport=shared_transport,
        max_episode_steps=env_max_episode_steps,
        auto_connect=False,
    )
    # Switch back to train mode afterward: DRQNGreedyPolicy flips the net to eval()
    # for inference; the next collection/learn step expects train mode.
    was_training = trainer.online.training
    try:
        report = evaluate(
            eval_env,
            policy,
            n_episodes=n_episodes,
            logger=logger,
            timeout_cap=timeout_cap,
            base_seed=base_seed,
            is_live=is_live,
            max_episode_steps=eval_step_cap,
            log=log,
            opponent=opponent,
        )
    finally:
        # Do NOT close eval_env: it borrows training's socket and must not send
        # `close` or tear down the shared transport. The training env owns and
        # closes that socket exactly once in train_vs_dummy's finally.
        if was_training:
            trainer.online.train()
    return report


#: Historical name for :func:`_eval_against_opponent`, from when the eval could
#: only ever fight the stationary dummy. Kept because ``eval/combat_probe.py``
#: names it in prose.
_eval_against_dummy = _eval_against_opponent


def _close_quietly(env: Any) -> None:
    """Close ``env`` if it exposes a ``close()``; ignore its absence/errors.

    The trainer's :class:`EnvProtocol` only requires ``reset`` / ``step``; the
    real :class:`~env.mc_pvp_env.MCPvPEnv` also owns a transport and MUST be
    closed, but a lightweight fake env may not implement ``close``. Teardown must
    never mask the loop's own result or error, so a missing/raising ``close`` is
    swallowed here.
    """
    close = getattr(env, "close", None)
    if not callable(close):
        return
    try:
        close()
    except Exception:
        # Teardown is best-effort; the connection is going away regardless.
        pass


# ===========================================================================
# T8 — the MULTI-ARENA training entrypoint (issue #4, Route 1 / Ape-X-lite).
#
# Engaged only when ``--arenas N`` with N > 1; N == 1 stays on today's exact
# ``train_vs_dummy`` single-env path (byte-identical, AC1/TC15). This wires the
# stack the other distributed tasks built:
#
#   one learner-side Trainer (its replay is the SINGLE shared buffer)
#     + a WeightStore (learner publishes cloned snapshots; collectors read)
#     + a LocalTransport (collectors push Episodes up; the learner drains them)
#     + N SnapshotPolicy collectors, one per arena, each over its own MCPvPEnv
#       on bridge port base+i (the N bridges/servers are started by the human;
#       T8 only connects clients)
#     + an ActorPool supervising the N collector daemons under the two-tier fault
#       policy (a dead pad's bridge is restarted via the launcher; a dead shared
#       Paper JVM aborts the whole run — there is no survivor floor)
#     + a LearnerLoop on a background thread (the SOLE replay mutator)
#
# Periodic GREEDY eval runs on ONE designated arena (arena 0) via the collector
# pause/handoff protocol: pause the designated collector at its episode boundary,
# wait for it to park idle, BORROW its (idle) env/transport, run eval reusing the
# single-connection ``_eval_against_opponent`` borrow, then resume. Eval opens NO new
# connection on ANY arena — the bridge serves exactly one connection per arena (a
# recorded gotcha that aborted a live run before).
#
# The orchestration is factored so a test can drive it with FAKE envs / a fake
# transport / a fake launcher WITHOUT a live server: every collaborator is
# injectable. The live CLI path (``main`` below) constructs the real components.
# ===========================================================================


@dataclass
class MultiArenaResult:
    """Outcome of a :func:`train_multi_arena` run (the N>1 sibling of M2Result).

    Attributes:
        trainer: The learner-side :class:`Trainer` after the run (online/target
            nets, the single shared replay, grad-step count).
        passed_m2: ``True`` iff the most recent designated-arena eval cleared the
            full M2 gate (``win_rate >= 0.95`` AND ``aim_while_invisible == 0`` AND
            ``mean_episode_length < timeout_cap``). ``False`` if no eval cleared it.
        grad_steps: Gradient steps the learner completed before stopping.
        episodes_received: Episodes the learner drained off the transport.
        last_report: The most recent eval report (``None`` if no eval ran).
        reports: Every eval report produced, in order.
        stop_reason: One of ``"passed_m2"`` / ``"max_grad_steps"`` /
            ``"max_episodes"`` / ``"learner_error"`` / ``"pool_aborted"``.
        is_live: ``True`` for a real-bridge run, ``False`` for the offline proof.
        curriculum: The shared :class:`OpponentCurriculum` when the run fought the
            scripted opponent (``cfg.opponent == "scripted"``), else ``None``. Kept
            on the result so a caller can read the final mixture / gate state
            without reaching into the pool.
        best_win_rate: Highest eval win rate seen, i.e. the score of the
            SAVE-BEST checkpoint. ``-1.0`` when no eval ran.
        best_grad_step: Learner grad step at which ``best_win_rate`` was measured
            (``-1`` when nothing was ever saved as best) — the number that says
            WHICH checkpoint the best file holds.
        eval_opponent: Who the periodic eval fought (``"dummy"`` or e.g.
            ``"scripted_mixed"``). Recorded because ``best_win_rate`` cannot be
            compared across runs — or trusted at all — without it.
        checkpoints_saved: How many times the periodic/final checkpoint hook
            fired. Reported so a run that saved NOTHING is visible in the result
            rather than only discoverable on disk at 8am.
    """

    trainer: "Trainer"
    passed_m2: bool
    grad_steps: int
    episodes_received: int
    last_report: Optional[Any] = None
    reports: List[Any] = None  # type: ignore[assignment]  # set in __post_init__
    stop_reason: str = "max_grad_steps"
    is_live: bool = False
    curriculum: Optional["OpponentCurriculum"] = None
    best_win_rate: float = -1.0
    best_grad_step: int = -1
    eval_opponent: str = "dummy"
    checkpoints_saved: int = 0

    def __post_init__(self) -> None:
        if self.reports is None:
            self.reports = []


#: ``arena_id -> (zero-arg env builder)``. The builder must return a FRESH,
#: connected env bound to that arena's bridge (so a relaunch can rebuild a fresh
#: client to the same single-connection bridge). The live path builds an
#: ``MCPvPEnv`` over a ``TcpBridgeClient(host, base_port + arena_id)``; tests inject
#: a factory returning a fake env.
EnvFactoryFor = Callable[[int], Callable[[], "EnvProtocol"]]


class _BestCheckpointSelector:
    """Decide which eval report is worth saving as THE checkpoint to ship.

    Selection is by **win rate, not recency**: a later checkpoint is kept only if
    it scored strictly higher than every earlier one. (The plain periodic hook
    saves the latest net separately; that file is the fallback, not the pick.)

    Two rules, both deliberate:

      * **strictly greater** — ties keep the EARLIER checkpoint. Two evals at the
        same win rate are the same evidence, and the earlier net got there with
        less overfitting to the eval opponent's fixed seed.
      * **must beat zero** — the first eval always beats the ``-1.0`` initial
        best, so without this a run whose agent never won a single eval episode
        would ship its FIRST eval's net (barely trained) in a file named "best".
        Until something is actually won there is no evidence to select on, and
        the honest answer is that no best checkpoint exists.

    Args:
        min_win_rate: A report must strictly exceed this to be selectable
            (default 0.0 — it must have won at least one eval episode).
    """

    def __init__(self, *, min_win_rate: float = 0.0) -> None:
        self.best_win_rate: float = -1.0
        self.best_grad_step: int = -1
        self._min_win_rate = float(min_win_rate)

    def consider(self, win_rate: float, grad_step: int) -> bool:
        """Record this eval; return True iff it is the new checkpoint to ship."""
        rate = float(win_rate)
        if rate <= self.best_win_rate:
            return False
        self.best_win_rate = rate
        if rate <= self._min_win_rate:
            # Tracked as the high-water mark, but not worth shipping yet.
            return False
        self.best_grad_step = int(grad_step)
        return True


def _eval_via_designated_arena(
    *,
    trainer: "Trainer",
    pool: Any,
    designated_arena: int,
    evaluate: Callable[..., Any],
    policy_cls: Callable[..., Any],
    n_episodes: int,
    timeout_cap: int,
    env_max_episode_steps: int,
    eval_step_cap: Optional[int],
    logger: Optional[Any],
    is_live: bool,
    base_seed: int,
    log: Optional[Callable[[str], None]],
    pause_timeout: float,
    opponent: Optional[Any] = None,
) -> Optional[Any]:
    """Run ONE greedy eval on the designated arena via the pause/handoff protocol.

    Pauses the designated arena's collector at its next EPISODE BOUNDARY, waits for
    it to confirm paused-and-idle (no reply in flight on its single bridge
    connection), BORROWS its idle env's transport, runs the eval through
    :func:`_eval_against_opponent` (which builds a separate ``MCPvPEnv`` with
    ``auto_connect=False`` over that shared transport — never a second connection),
    then resumes the collector. Eval opens NO connection on any OTHER arena; they
    keep collecting throughout.

    ``opponent`` (``None`` on the dummy path) is the Python-stepped opponent
    policy the eval fights; it is threaded straight through to
    :func:`_eval_against_opponent`. The borrowed connection is the arena's, but
    the opponent is the EVAL's own — a fresh, fixed-seed driver per eval, never
    the collector's curriculum-driven one, so the eval neither perturbs training's
    opponent RNG nor inherits its drifting mixture.

    Returns the eval report, or ``None`` if the designated collector could not be
    brought to an idle boundary within ``pause_timeout`` (e.g. it is mid-relaunch),
    in which case eval is SKIPPED this cycle rather than risking a second connection.
    """
    collector = pool.collector_for(designated_arena)
    if collector is None:
        return None

    collector.pause()
    try:
        # Block until the collector parks between episodes with its connection idle.
        # If it cannot (mid-relaunch / dead), skip eval this cycle: borrowing a
        # non-idle connection would race a reply, and opening a fresh one would steal
        # the bridge's single connection.
        if not collector.wait_until_idle(timeout=pause_timeout):
            if log is not None:
                log(
                    f"[multi] eval skipped: designated arena {designated_arena} did "
                    f"not reach an idle boundary within {pause_timeout:.1f}s"
                )
            return None

        shared_env = collector.current_env()
        if shared_env is None:
            return None
        shared_transport = getattr(shared_env, "_transport", None)
        if shared_transport is None:
            return None

        return _eval_against_opponent(
            trainer=trainer,
            evaluate=evaluate,
            policy_cls=policy_cls,
            shared_transport=shared_transport,
            n_episodes=n_episodes,
            timeout_cap=timeout_cap,
            env_max_episode_steps=env_max_episode_steps,
            eval_step_cap=eval_step_cap,
            logger=logger,
            is_live=is_live,
            base_seed=base_seed,
            log=log,
            opponent=opponent,
        )
    finally:
        # Always resume the collector, even if eval raised, so a single arena is
        # never left parked forever.
        collector.resume()


def train_multi_arena(
    cfg: TrainConfig,
    *,
    env_factory_for: EnvFactoryFor,
    launcher: Optional[Any] = None,
    transport: Optional[Any] = None,
    weight_store: Optional[Any] = None,
    counter: Optional[Any] = None,
    max_episodes: Optional[int] = None,
    max_grad_steps: Optional[int] = None,
    eval_every_grad_steps: int = 1_000,
    eval_episodes: int = 100,
    designated_arena: int = 0,
    timeout_cap: int = MAX_EPISODE_STEPS,
    env_max_episode_steps: int = MAX_EPISODE_STEPS,
    rollout_step_cap: Optional[int] = None,
    stop_on_pass: bool = True,
    device: Optional[torch.device] = None,
    net_kwargs: Optional[Dict[str, int]] = None,
    logger: Optional[Any] = None,
    checkpoint_hook: Optional[CheckpointHook] = None,
    best_checkpoint_hook: Optional[BestCheckpointHook] = None,
    checkpoint_every_grad_steps: Optional[int] = None,
    eval_opponent_factory: Optional[Callable[[], Any]] = None,
    is_live: bool = False,
    log: Optional[Callable[[str], None]] = None,
    poll_interval: float = 0.05,
    eval_pause_timeout: float = 120.0,
    relaunch_backoff_seconds: Optional[float] = None,
    relaunch_backoff_max_seconds: Optional[float] = None,
    sleep: Optional[Callable[[float], None]] = None,
    watchdog: Optional[Any] = None,
    jvm_probe: Optional[Callable[[str, int], bool]] = None,
    mc_host: Optional[str] = None,
    mc_port: Optional[int] = None,
    launcher_shutdown: Optional[Any] = None,
) -> MultiArenaResult:
    """Train one learner from ``cfg.arenas`` concurrent collectors (Ape-X-lite).

    The N>1 path. One :class:`Trainer` owns the single shared replay (mutated ONLY
    by the background :class:`~distributed.learner.LearnerLoop`); N
    :class:`~distributed.weights.SnapshotPolicy` collectors act on periodically
    synced weight snapshots and push whole :class:`~distributed.serialization.Episode`
    objects onto a :class:`~distributed.transport.LocalTransport`. An
    :class:`~distributed.actor.ActorPool` supervises the collector daemons (relaunch
    via the injected ``launcher``); this function runs the learner on a background
    thread and JOINS on the episode/grad-step budget or the M2 win-rate gate, then
    stops everything cleanly. A LearnerLoop error or a pool abort surfaces loudly.

    Everything external is INJECTABLE so the whole orchestration is offline-testable
    with fakes (env factory, launcher, transport, store, counter, sleep, watchdog).
    The live CLI path constructs the real components.

    Determinism: each arena's per-episode seed is ``arena_episode_seed(cfg,
    arena_id, local_ep)`` (its own ``torch.Generator`` inside the SnapshotPolicy);
    the ε schedule advances off the GLOBAL episode counter (in the Collector); PER β
    anneals off the learner's ``grad_step`` (inside ``trainer.learn()``). The MEAN ε
    across arenas is logged (each arena carries its own ε under the global schedule).

    Args:
        cfg: Training hyperparameters. ``cfg.arenas`` (> 1) sets the collector count;
            ``cfg.weight_sync_every_k_steps`` the publish cadence;
            ``cfg.collector_queue_max`` the transport bound (0 == unbounded);
            ``cfg.fault_relaunch`` arms tier 1 of the fault policy (restart a dead
            pad's own bridge). Tier 2 — a dead shared JVM aborts the run — is armed
            by ``jvm_probe``, not by config; there is no survivor floor.
        env_factory_for: ``arena_id -> (zero-arg env builder)``; the builder returns a
            fresh connected env on that arena's bridge (see :data:`EnvFactoryFor`).
        launcher: The :class:`~distributed.actor.ArenaLauncher` the pool relaunches a
            dead arena through. The live path lazily imports
            :class:`~distributed.launcher.SubprocessArenaLauncher`; a test injects a
            fake. Required (relaunch cannot be wired without it).
        transport / weight_store / counter: Optional injected collaborators; the real
            ones (``LocalTransport`` / ``WeightStore`` / ``GlobalEpisodeCounter``) are
            built here when omitted.
        max_episodes: Optional cap on episodes the LEARNER drains before stopping
            (the episode budget); ``None`` relies on ``max_grad_steps``.
        max_grad_steps: Optional cap on learner gradient steps (the compute budget);
            ``None`` relies on ``max_episodes``. At least one budget should be set or
            the run stops only on the gate / an abort.
        eval_every_grad_steps: Run a designated-arena greedy eval each time the
            learner crosses this many grad steps (``0`` disables periodic eval).
            Indexed off the learner's grad step (not episodes) because the learner is
            the single clock under N collectors.
        eval_episodes: Episodes per greedy eval. AC6 uses 100; tests use a few.
        designated_arena: The single arena eval pauses/borrows (default 0). Eval never
            fans out across arenas.
        timeout_cap / env_max_episode_steps / rollout_step_cap: Episode-length knobs,
            forwarded to eval / the collectors (see ``train_vs_dummy``).
        stop_on_pass: Stop as soon as an eval clears the M2 gate (default True).
            NOTE for a scripted-opponent retrain: ``passed_m2`` is the M2 gate,
            which AC6 defines against the STATIONARY DUMMY. Leaving this True on a
            run whose eval fights a moving opponent stops the night the first time
            the agent has a good eval, which is not the milestone and not the
            plan. ``agent.train``'s CLI therefore defaults it to False whenever
            ``--opponent scripted`` is selected.
        device / net_kwargs: Forwarded to the learner :class:`Trainer` AND used to
            build each collector's snapshot net (same architecture as the learner).
        logger: Optional metrics logger (per-eval components + throughput).
        checkpoint_hook: Optional ``(trainer, grad_step) -> None`` for the LATEST
            net. Fired on a cadence (see ``checkpoint_every_grad_steps``) and once
            more at the end of the run — both INDEPENDENT of eval, because a run
            with eval disabled that saves nothing all night is the worst outcome
            this function can produce. Also fired on an eval improvement when no
            ``best_checkpoint_hook`` is given (the historical single-file
            behavior).
        best_checkpoint_hook: Optional ``(trainer, grad_step, meta) -> None`` for
            the SAVE-BEST net, fired only when an eval strictly improves the win
            rate (see :class:`_BestCheckpointSelector`); ``meta`` carries the
            win rate, the eval opponent and the episode count that justified the
            save. Keep it pointed at a DIFFERENT
            path from ``checkpoint_hook``: sharing one path means the next
            periodic save overwrites the best net with a more recent, worse one —
            selection by recency wearing selection-by-win-rate's clothes.
        checkpoint_every_grad_steps: Grad steps between periodic ``checkpoint_hook``
            saves. ``None`` (the default) uses ``cfg.checkpoint_interval``; ``0``
            disables the periodic save (the FINAL save still happens).
        eval_opponent_factory: Optional ``() -> EvalOpponent`` building the
            opponent policy the periodic eval steps; a FRESH one is built per eval
            so every eval faces an identical opponent. ``None`` (the default)
            means the eval is built from ``cfg`` via :func:`build_eval_opponent` —
            which yields ``None`` on the dummy path, keeping the M2 eval's wire
            line byte-identical.
        is_live: Marks reports/result as live vs offline.
        log: Optional ASCII-only ``str -> None`` sink (Windows cp1252-safe).
        poll_interval: Seconds between the driver's budget/health polls.
        eval_pause_timeout: Max seconds to wait for the designated collector to park
            idle before SKIPPING an eval cycle (an arena may be mid-relaunch).
        relaunch_backoff_seconds / relaunch_backoff_max_seconds / sleep: Forwarded to
            the ``ActorPool`` collectors (tests pass tiny / no-op values so no real
            sleeping; ``None`` uses the actor module defaults).
        watchdog: Optional :class:`~distributed.learner.LearnerWatchdog` for the
            learner loop; ``None`` uses a default watchdog so a wedged learner aborts.
        jvm_probe: Optional ``(host, port) -> bool`` liveness check for the SHARED
            Paper JVM — tier 2 of the fault policy. ``None`` (the default) means no
            JVM supervision, which is what an offline pool of fake envs wants; the
            live path passes :func:`distributed.actor.jvm_alive`. A ``False`` from it
            aborts the whole run with :class:`~distributed.actor.PoolAbortedError`.
            REQUIRED when ``is_live`` — an unsupervised live run is rejected.
        mc_host / mc_port: Where that JVM listens. Defaults come from
            ``distributed.actor``; pass the SAME values the ``launcher`` was built
            with so the probe and the launcher can never watch different ports.
        launcher_shutdown: Optional
            :class:`~distributed.actor.ShutdownSignal` whose ``sleep`` was injected
            into the ``launcher``. The pool sets it on stop/abort so a collector
            parked inside a bridge relaunch unwinds at once instead of holding
            shutdown for the launcher's full bounded wait.

    Returns:
        A :class:`MultiArenaResult` with the trainer, the gate verdict, the grad-step
        / received-episode counts, every eval report, and the stop reason.

    Raises:
        ValueError: if ``cfg.arenas`` < 2 (use ``train_vs_dummy`` for N=1), or
            ``designated_arena`` is out of range, or ``eval_episodes`` <= 0, or
            ``is_live`` is set without a ``jvm_probe``.
        LearnerError: if the background learner thread aborts (re-raised loudly).
        PoolAbortedError: if the shared Paper JVM dies mid-run (tier 2). A dead pad
            never raises: its bridge is restarted in place.
    """
    import threading as _threading
    import time

    # Local imports keep the distributed stack off agent.train's import path until
    # the N>1 branch actually runs (distributed.actor imports FROM agent.train, so a
    # top-level import here would be a cycle).
    from distributed.actor import ActorPool, GlobalEpisodeCounter, PoolAbortedError
    from distributed.learner import LearnerLoop, LearnerWatchdog
    from distributed.transport import LocalTransport
    from distributed.weights import SnapshotPolicy, WeightStore

    from agent.contract_config import code_version as _code_version
    from agent.dqn import DuelingDRQN

    if cfg.arenas < 2:
        raise ValueError(
            f"train_multi_arena requires cfg.arenas >= 2, got {cfg.arenas}; "
            "use train_vs_dummy for the single-arena (N=1) path."
        )
    if not (0 <= designated_arena < cfg.arenas):
        raise ValueError(
            f"designated_arena {designated_arena} out of range [0, {cfg.arenas})"
        )
    if eval_episodes <= 0:
        raise ValueError(f"eval_episodes must be > 0, got {eval_episodes}")
    if eval_every_grad_steps < 0:
        raise ValueError(
            f"eval_every_grad_steps must be >= 0, got {eval_every_grad_steps}"
        )
    if checkpoint_every_grad_steps is not None and checkpoint_every_grad_steps < 0:
        raise ValueError(
            "checkpoint_every_grad_steps must be >= 0 or None, got "
            f"{checkpoint_every_grad_steps}"
        )
    if is_live and jvm_probe is None:
        # A LIVE run with no tier 2 is a configuration the fault policy says must not
        # exist: the shared Paper JVM could die and every pad would sit restarting
        # bridges into nothing, forever, while the learner trained on a world that is
        # gone. Refuse at construction rather than discover it during the incident.
        raise ValueError(
            "train_multi_arena(is_live=True) requires a jvm_probe: a live fleet has "
            "a shared Paper JVM whose death must abort the run (tier 2 of the fault "
            "policy). Pass jvm_probe=distributed.actor.jvm_alive together with the "
            "mc_port the launcher was built with. jvm_probe=None is for OFFLINE runs "
            "over fake envs, which have no JVM to lose."
        )

    def _emit(message: str) -> None:
        if log is not None:
            log(message)

    # --- the single learner-side trainer (its replay is THE shared buffer) ---
    net_kwargs = dict(net_kwargs or {})
    trainer = Trainer(cfg, device=device, net_kwargs=net_kwargs)

    # --- shared collaborators (build the real ones when not injected) --------
    if transport is None:
        # Bounded iff a positive cap is configured; else unbounded (today's behavior).
        maxsize = cfg.collector_queue_max if cfg.collector_queue_max > 0 else 0
        transport = LocalTransport(maxsize=maxsize)
    if weight_store is None:
        weight_store = WeightStore()
    if counter is None:
        counter = GlobalEpisodeCounter()
    if launcher is None:
        raise ValueError(
            "train_multi_arena needs an ArenaLauncher (the pool relaunches a dead "
            "arena through it). Pass launcher=SubprocessArenaLauncher(...) for the "
            "live run, or a fake launcher in tests."
        )

    # --- the opponent curriculum (T12) --------------------------------------
    # "dummy" (the default) builds nothing at all: no opponent_for reaches the
    # pool, no opp_action reaches the wire, and the M2 path is untouched.
    # "scripted" builds ONE shared curriculum gate plus one driver per arena.
    curriculum: Optional[OpponentCurriculum] = None
    opponent_for: Optional[Callable[[int], EpisodeOpponent]] = None
    if cfg.opponent == "scripted":
        curriculum, opponent_for = build_scripted_opponents(cfg)

    # --- the EVAL's own opponent (T13) --------------------------------------
    # Separate from the collectors' drivers on purpose: the eval must fight the
    # same KIND of opponent training does (or its win rate scores a stationary
    # target and selects the wrong checkpoint) while staying identical across
    # evals (or two win rates are not comparable and "select the best" is
    # meaningless). None on the dummy path.
    if eval_opponent_factory is None:
        eval_opponent_factory = build_eval_opponent(cfg)
    eval_opponent_name = "dummy"
    if eval_opponent_factory is not None:
        eval_opponent_name = str(getattr(eval_opponent_factory(), "name", "opponent"))

    # --- per-arena snapshot policies (same net architecture as the learner) --
    # Each collector owns its own net clone (built from the SAME net_kwargs so the
    # learner's published state_dict load_state_dicts cleanly) and its own RNG seeded
    # off the per-arena seed band, plus the arena/code-version stamp for the Episode.
    code_ver = _code_version()

    def _net_factory() -> Any:
        return DuelingDRQN(**net_kwargs)

    policies: Dict[int, Any] = {}

    def _policy_for(arena_id: int) -> Any:
        policy = policies.get(arena_id)
        if policy is None:
            policy = SnapshotPolicy(
                _net_factory,
                # The generator seed is the arena's local-episode-0 seed; the policy
                # re-seeds per episode anyway, so this is just a distinct, reproducible
                # starting point per arena.
                generator_seed=arena_episode_seed(cfg, arena_id, 0),
                arena_id=arena_id,
                code_version=code_ver,
            )
            policies[arena_id] = policy
        return policy

    # --- the actor pool (N daemon collectors over the shared store/transport)
    actor_kwargs: Dict[str, Any] = dict(
        cfg=cfg,
        env_factory_for=env_factory_for,
        policy_for=_policy_for,
        transport=transport,
        weight_store=weight_store,
        launcher=launcher,
        counter=counter,
        max_episode_steps=rollout_step_cap,
    )
    if opponent_for is not None:
        actor_kwargs["opponent_for"] = opponent_for
    if relaunch_backoff_seconds is not None:
        actor_kwargs["relaunch_backoff_seconds"] = relaunch_backoff_seconds
    if relaunch_backoff_max_seconds is not None:
        actor_kwargs["relaunch_backoff_max_seconds"] = relaunch_backoff_max_seconds
    if sleep is not None:
        actor_kwargs["sleep"] = sleep
    # Two-tier fault policy (T11). Each of these is omitted rather than passed as
    # None so the pool keeps its own documented defaults; jvm_probe=None there means
    # "no JVM supervision", which is exactly right for a pool of fake envs.
    if jvm_probe is not None:
        actor_kwargs["jvm_probe"] = jvm_probe
    if mc_host is not None:
        actor_kwargs["mc_host"] = mc_host
    if mc_port is not None:
        actor_kwargs["mc_port"] = mc_port
    if launcher_shutdown is not None:
        actor_kwargs["shutdown"] = launcher_shutdown
    pool = ActorPool.build(**actor_kwargs)

    # --- the decoupled learner loop (the SOLE replay mutator) ----------------
    if watchdog is None:
        watchdog = LearnerWatchdog()
    learner = LearnerLoop(
        trainer,
        transport,
        weight_store,
        cfg,
        watchdog=watchdog,
        log=log,
    )
    learner_thread = _threading.Thread(
        target=learner.run, name="learner-loop", daemon=True
    )

    # --- eval wiring (lazy: keep the eval dependency at the call boundary) ----
    reports: List[Any] = []
    last_report: Optional[Any] = None
    selector = _BestCheckpointSelector()
    passed = False
    stop_reason = "max_grad_steps"
    do_eval = eval_every_grad_steps > 0
    if do_eval:
        from eval.evaluate import DRQNGreedyPolicy, evaluate
    next_eval_at = eval_every_grad_steps  # first eval boundary (grad steps)

    # --- checkpoint cadence (INDEPENDENT of eval) ---------------------------
    # The reason this exists: the only save on this path used to sit inside the
    # eval-improvement branch, so `--eval-every-grad-steps 0` trained all night
    # and wrote ZERO checkpoints, and `Trainer._fire_hooks` / cfg.checkpoint_interval
    # are dead here (the learner calls trainer.learn() directly, never
    # trainer.train()). Both the periodic save below and the final save in the
    # `finally` are unconditional on eval.
    checkpoint_every = (
        int(cfg.checkpoint_interval)
        if checkpoint_every_grad_steps is None
        else int(checkpoint_every_grad_steps)
    )
    do_periodic_checkpoint = checkpoint_hook is not None and checkpoint_every > 0
    next_checkpoint_at = checkpoint_every
    checkpoints_saved = 0

    def _save_latest(grad_step: int, why: str) -> None:
        """Fire the LATEST-net hook, never letting a save failure kill the run."""
        nonlocal checkpoints_saved
        if checkpoint_hook is None:
            return
        try:
            checkpoint_hook(trainer, int(grad_step))
        except Exception as exc:  # noqa: BLE001 - a bad path must not end the night
            _emit(f"[multi] checkpoint save FAILED at grad_step {grad_step}: {exc}")
            return
        checkpoints_saved += 1
        _emit(f"[multi] checkpoint saved ({why}) at grad_step {grad_step}")

    def _maybe_log_mean_epsilon(grad_step: int) -> None:
        # The logged epsilon is per-arena under N collectors; log the MEAN across the
        # arenas' current schedule positions (computed from the GLOBAL counter so it
        # reflects the combined stream the schedule actually advanced over).
        if logger is None:
            return
        # All arenas share the global episode counter, so they sit at (nearly) the
        # same schedule point; the mean over the counter's value is the representative
        # ε. We sample the schedule at the current global episode count.
        global_eps = epsilon_for_episode(max(0, counter.value - 1), cfg)
        logger.log({"train/epsilon_mean": float(global_eps)}, step=int(grad_step))

    _emit(
        f"[multi] starting {cfg.arenas} pads - "
        f"weight_sync_every_k={cfg.weight_sync_every_k_steps}, "
        f"queue_max={cfg.collector_queue_max}, "
        f"bridge_restart={'on' if cfg.fault_relaunch else 'OFF'}, "
        f"jvm_watch={'on' if jvm_probe is not None else 'off'}, "
        + (
            f"opponent=scripted (mix_easy {cfg.opponent_mix_easy:.2f} -> "
            f"{cfg.opponent_mix_easy_after:.2f} at win_rate "
            f"{cfg.opponent_gate_winrate:.2f} over {cfg.opponent_gate_window} "
            f"EASY eps), "
            if curriculum is not None
            else "opponent=dummy, "
        )
        + (
            f"eval every {eval_every_grad_steps} grad steps on arena "
            f"{designated_arena} vs {eval_opponent_name}"
            if do_eval
            else "eval disabled"
        )
        + (
            f", checkpoint every {checkpoint_every} grad steps"
            if do_periodic_checkpoint
            else ", periodic checkpoint OFF"
        )
        + (f", stop_on_pass={'on' if stop_on_pass else 'off'}")
    )
    if checkpoint_hook is None and best_checkpoint_hook is None:
        # Said at the START, loudly. A multi-hour run that saves nothing is
        # unrecoverable at 8am, and the only symptom is an empty runs/ directory.
        _emit(
            "[multi] WARNING: no checkpoint hook configured - this run will train "
            "and save NOTHING. Pass --checkpoint (and optionally "
            "--best-checkpoint) before starting an overnight run."
        )
    if cfg.warm_start is not None:
        _emit(
            f"[multi] warm start from {cfg.warm_start} - online+target initialized "
            f"from it, replay FRESH, epsilon restarts at "
            f"{effective_eps_start(cfg):.3f} (not {cfg.eps_start:.3f})"
        )
        # Hand the collectors the warm weights BEFORE they roll their first
        # episode. The learner publishes an identical version-0 snapshot the
        # instant its thread starts, but the pool starts first, so without this a
        # collector can open with a randomly-initialized net — a small amount of
        # garbage in a deliberately fresh replay, from the one run that exists to
        # not start from scratch. Re-publishing version 0 is harmless: collectors
        # reload only on a STRICTLY greater version and the weights are the same.
        weight_store.publish(trainer.online.state_dict(), 0)

    pool.start()
    learner_thread.start()

    # One-shot latch so the curriculum's shift is logged exactly once. Purely
    # observational — the driver loop NEVER waits on the gate, so a gate that
    # does not fire cannot hold the run up (AC10).
    gate_logged = False

    try:
        while True:
            # --- learner liveness: a dead learner or a pool abort stops loudly ---
            if learner.error is not None:
                stop_reason = "learner_error"
                break
            if pool.aborted():
                stop_reason = "pool_aborted"
                break
            if learner.stopped:
                # The learner ended on its own (transport closed/drained or stop()).
                stop_reason = "learner_stopped"
                break

            grad_step = int(trainer.grad_step)
            received = int(learner.received)

            # --- curriculum observability (never a control point) ---------------
            if curriculum is not None and not gate_logged and curriculum.gate_fired:
                gate_logged = True
                stats = curriculum.stats()
                _emit(
                    f"[multi grad_step {grad_step}] opponent curriculum gate FIRED: "
                    f"mix_easy {cfg.opponent_mix_easy:.2f} -> "
                    f"{cfg.opponent_mix_easy_after:.2f} after "
                    f"{stats['easy_episodes']} EASY episodes "
                    f"(rolling win_rate={stats['easy_window_win_rate']})"
                )

            # --- budget checks (the learner is the single clock) ----------------
            if max_grad_steps is not None and grad_step >= max_grad_steps:
                stop_reason = "max_grad_steps"
                break
            if max_episodes is not None and received >= max_episodes:
                stop_reason = "max_episodes"
                break

            # --- periodic checkpoint of the LATEST net (eval-independent) -------
            if do_periodic_checkpoint and grad_step >= next_checkpoint_at:
                _save_latest(grad_step, "periodic")
                # Re-read grad_step: saving takes time and the learner kept going.
                next_checkpoint_at = int(trainer.grad_step) + checkpoint_every

            # --- periodic designated-arena eval via pause/handoff ---------------
            if do_eval and grad_step >= next_eval_at:
                _maybe_log_mean_epsilon(grad_step)
                report = _eval_via_designated_arena(
                    trainer=trainer,
                    pool=pool,
                    designated_arena=designated_arena,
                    evaluate=evaluate,
                    policy_cls=DRQNGreedyPolicy,
                    n_episodes=eval_episodes,
                    timeout_cap=timeout_cap,
                    env_max_episode_steps=env_max_episode_steps,
                    eval_step_cap=rollout_step_cap,
                    logger=logger,
                    is_live=is_live,
                    base_seed=cfg.seed,
                    log=log,
                    pause_timeout=eval_pause_timeout,
                    # A FRESH opponent per eval, from the same fixed seed: every
                    # eval fights the identical opponent, so the win-rate series
                    # measures the AGENT and nothing else.
                    opponent=(
                        eval_opponent_factory()
                        if eval_opponent_factory is not None
                        else None
                    ),
                )
                # Advance the boundary past the CURRENT grad step so a long eval (the
                # learner kept stepping during the borrow) does not immediately re-fire.
                next_eval_at = int(trainer.grad_step) + eval_every_grad_steps

                if report is not None:
                    reports.append(report)
                    last_report = report
                    eval_grad_step = int(trainer.grad_step)
                    # Selection is by WIN RATE, not recency (see the selector).
                    if selector.consider(report.win_rate, eval_grad_step):
                        if best_checkpoint_hook is not None:
                            try:
                                best_checkpoint_hook(
                                    trainer,
                                    eval_grad_step,
                                    {
                                        "win_rate": float(report.win_rate),
                                        "eval_opponent": eval_opponent_name,
                                        "eval_episodes": int(report.n_episodes),
                                        "mean_episode_length": float(
                                            report.mean_episode_length
                                        ),
                                        "passed_m2": bool(report.passed_m2),
                                    },
                                )
                            except Exception as exc:  # noqa: BLE001
                                _emit(
                                    "[multi] BEST checkpoint save FAILED at "
                                    f"grad_step {eval_grad_step}: {exc}"
                                )
                            else:
                                _emit(
                                    f"[multi] best checkpoint saved: win_rate="
                                    f"{report.win_rate:.3f} vs {eval_opponent_name} "
                                    f"at grad_step {eval_grad_step}"
                                )
                        else:
                            # Legacy single-hook callers keep the old save-best
                            # behavior; the periodic/final saves are additional.
                            _save_latest(eval_grad_step, "best")
                    _emit(
                        f"[multi grad_step {eval_grad_step}] "
                        f"win_rate={report.win_rate:.3f} "
                        f"mean_len={report.mean_episode_length:.1f} "
                        f"aim_invisible={report.aim_while_invisible:.3f} "
                        f"passed_m2={report.passed_m2} "
                        f"opponent={eval_opponent_name}"
                    )
                    if report.passed_m2 and stop_on_pass:
                        passed = True
                        stop_reason = "passed_m2"
                        break

            # Park briefly; the learner and collectors run on their own threads.
            time.sleep(poll_interval)
    finally:
        # --- clean shutdown: stop collectors, close the channel, join learner ---
        # Close the transport FIRST so the learner's blocking recv() wakes and the
        # loop exits cleanly; then stop the pool (collectors wind down at their next
        # boundary) and join the learner thread.
        learner.stop()
        try:
            transport.close()
        except Exception:  # noqa: BLE001 - teardown best-effort
            pass
        pool_abort_error: Optional[BaseException] = None
        try:
            pool.stop()
        except PoolAbortedError as exc:
            pool_abort_error = exc
        learner_thread.join(timeout=5.0)

        # --- the FINAL save ------------------------------------------------
        # After the join, so the state_dict cannot be read while the learner is
        # mid-optimizer-step. Inside the `finally`, so a run that ends on a
        # learner error, a pool abort, or a Ctrl-C at 4am still leaves the night's
        # weights on disk — the old code had no final save here at all, and
        # _save_latest swallows its own failures so teardown can never mask the
        # error that got us here.
        _save_latest(int(trainer.grad_step), "final")

    # --- surface a learner error or a pool abort LOUDLY ----------------------
    if learner.error is not None:
        # The learner stores and re-raises the ORIGINAL exception (a WatchdogError on
        # a stalled drain, or any trainer/transport failure), so re-raise it verbatim
        # rather than wrap it — the run must fail loudly, never train into the void.
        _emit(f"[multi] learner aborted: {learner.error}")
        raise learner.error
    if pool_abort_error is not None:
        _emit(f"[multi] pool aborted: {pool_abort_error}")
        raise pool_abort_error

    return MultiArenaResult(
        trainer=trainer,
        passed_m2=passed,
        grad_steps=int(trainer.grad_step),
        episodes_received=int(learner.received),
        last_report=last_report,
        reports=reports,
        stop_reason=stop_reason,
        is_live=bool(is_live),
        curriculum=curriculum,
        best_win_rate=selector.best_win_rate,
        best_grad_step=selector.best_grad_step,
        eval_opponent=eval_opponent_name,
        checkpoints_saved=checkpoints_saved,
    )


# ---------------------------------------------------------------------------
# CLI — the LIVE M2 training run (AC6/TC13 prep), part of T20.
#
# Wires the REAL ``TcpBridgeClient`` transport to the started Node bridge / Paper
# server (which serves the stationary dummy), runs ``train_vs_dummy`` to the M2
# gate or the budget, logs through a ``MetricsLogger``, optionally writes a final
# checkpoint, and exits 0 iff the gate passed. The OFFLINE end-to-end logic is
# proved by tests/test_integration_m2.py; this entry point is the LIVE run.
# ---------------------------------------------------------------------------

#: The SHARED Minecraft port: ONE JVM serves every pad, so this is a port, not a
#: base. Declared here rather than imported because ``distributed.actor`` imports
#: FROM this module (a top-level import back would be a cycle) and the parser is
#: built on the single-arena path too, which must not pay for the distributed
#: stack. It MUST equal ``distributed.actor.MC_PORT`` and ``server.properties``'
#: ``server-port``; tests/test_actor_pool.py pins the two against each other.
_DEFAULT_MC_PORT: int = 25565


def _build_parser() -> "Any":
    import argparse

    parser = argparse.ArgumentParser(
        prog="train",
        description=(
            "T20 M2 integration: train the Dueling-DRQN vs the stationary dummy "
            "over the real perception+reward env, with periodic greedy eval "
            "against the M2 gate (win_rate>=95%, aim-while-invisible==0, mean "
            "length<cap). Connects to a started Node bridge / Paper server. The "
            "offline end-to-end wiring is proved by tests/test_integration_m2.py; "
            "this entry point is the LIVE M2 run (AC6/TC13) - see server/README.md."
        ),
    )
    parser.add_argument(
        "--max-episodes", type=int, default=10_000,
        help="episode budget (default: 10000).",
    )
    parser.add_argument(
        "--max-grad-steps", type=int, default=None,
        help="optional gradient-step budget (default: none - episodes only).",
    )
    parser.add_argument(
        "--eval-every-episodes", type=int, default=50,
        help="single-arena (--arenas 1) only: run a greedy eval every N episodes "
        "(default: 50; 0 disables).",
    )
    parser.add_argument(
        "--eval-every-grad-steps", type=int, default=1_000,
        help="multi-arena (--arenas >1) only: run a designated-arena greedy eval "
        "every N learner gradient steps (default: 1000; 0 disables). The learner "
        "is the single clock under N collectors, so eval is paced by grad steps "
        "rather than episodes.",
    )
    parser.add_argument(
        "--eval-episodes", type=int, default=100,
        help="episodes per greedy eval (default: 100, per AC6).",
    )
    parser.add_argument(
        "--host", type=str, default="127.0.0.1",
        help="bridge host for the live run (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port", type=int, default=5555,
        help="bridge TCP port for the live run (default: 5555). For --arenas N "
        "this is the BASE port: arena i connects to port + i.",
    )
    parser.add_argument(
        "--mc-port", type=int, default=_DEFAULT_MC_PORT,
        help=f"the SHARED Minecraft port for --arenas >1 (default: {_DEFAULT_MC_PORT}). One JVM "
        "serves every pad, so this is a single port, not a base. It is used for "
        "two things that must agree: the launcher's precondition check before it "
        "restarts a pad's bridge, and the jvm_alive() watchdog whose failure aborts "
        "the whole run.",
    )
    parser.add_argument(
        "--arenas", type=int, default=1,
        help="number of parallel Minecraft arenas to train from (default: 1). "
        "1 == today's exact single-env path (no threading, no weight sync). "
        ">1 engages the multi-arena ActorPool + decoupled learner: arena i "
        "connects to bridge port (--port)+i. The N pads must already be booted "
        "and PRIMED (server/setup/start-pads.sh --pads N, wait for FLEET READY). "
        "This is the agent.train TRAINING flag, distinct from eval.benchmark's "
        "measurement --arenas.",
    )
    parser.add_argument(
        "--opponent", type=str, default="dummy", choices=("dummy", "scripted"),
        help="who the agent fights (default: dummy). 'dummy' is the M1/M2 "
        "stationary dummy served entirely by the bridge - no opponent action ever "
        "goes on the wire. 'scripted' steps the ScriptedBot curriculum in Python "
        "and threads its macro through as opp_action; it requires --arenas >1 "
        "(the single-arena loop steps no opponent policy and refuses rather than "
        "silently fighting the dummy).",
    )
    # -- curriculum knobs (all default to TrainConfig's own values) ----------
    # Every one of these defaults to None and is applied only when given, so
    # TrainConfig stays the single source of the defaults and an argparse default
    # can never drift away from the dataclass it configures. They exist because
    # the plan's declared fallback ("train EASY-only, keep HARD as a demo knob")
    # needs opponent_mix_easy=1.0, and without a flag taking that fallback means
    # editing dataclass defaults on freeze day.
    parser.add_argument(
        "--opponent-mix-easy", type=float, default=None,
        help="probability an episode draws the EASY preset BEFORE the win-rate "
        "gate fires (default: TrainConfig's 0.8). For the plan's EASY-ONLY "
        "schedule cut pass BOTH this and --opponent-mix-easy-after as 1.0: this "
        "one alone only holds until the gate fires, after which the mixture "
        "shifts to --opponent-mix-easy-after (0.2) and HARD becomes the majority "
        "tier - the opposite of the cut.",
    )
    parser.add_argument(
        "--opponent-mix-easy-after", type=float, default=None,
        help="probability of EASY AFTER the gate fires (default: 0.2). Not 0.0 on "
        "purpose: the curriculum is a gated MIXTURE, and EASY episodes are what "
        "keep the gate's own window fed.",
    )
    parser.add_argument(
        "--opponent-gate-winrate", type=float, default=None,
        help="rolling win rate vs EASY that shifts the mixture (default: 0.6).",
    )
    parser.add_argument(
        "--opponent-gate-window", type=int, default=None,
        help="EASY episodes in the rolling gate window; it must be FULL before the "
        "gate is evaluated (default: 50).",
    )
    parser.add_argument(
        "--eval-opponent-preset", type=str, default=None,
        choices=("mixed", "easy", "hard"),
        help="which scripted tier the PERIODIC EVAL fights with --opponent "
        "scripted (default: mixed - EASY/HARD alternating by episode index). It is "
        "deliberately NOT the training mixture, which shifts when the gate fires: "
        "two evals either side of that shift would score different opponents and "
        "the checkpoint selection would compare numbers that do not mean the same "
        "thing.",
    )
    parser.add_argument(
        "--warm-start", type=str, default=None,
        help="path to a checkpoint whose weights initialize this run (online AND "
        "target). The replay buffer is deliberately NOT restored, and the epsilon "
        "schedule restarts at --warm-start-eps-start instead of eps_start.",
    )
    parser.add_argument(
        "--warm-start-eps-start", type=float, default=None,
        help="epsilon the schedule restarts at under --warm-start (default: 0.25, "
        "the plan's 0.2-0.3 band). Ignored without --warm-start. Leaving this at "
        "the fresh-init 1.0 would spend the whole decay window acting mostly at "
        "random and throw the warm start away.",
    )
    parser.add_argument(
        "--stop-on-pass", dest="stop_on_pass", action="store_true", default=None,
        help="stop the run as soon as an eval clears the M2 gate. DEFAULT: on for "
        "--opponent dummy, OFF for --opponent scripted (the M2 gate is defined "
        "against the stationary dummy, so it is not this retrain's finish line).",
    )
    parser.add_argument(
        "--no-stop-on-pass", dest="stop_on_pass", action="store_false",
        help="run the full budget even if an eval clears the M2 gate.",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="base RNG seed (overrides TrainConfig.seed) (default: 0).",
    )
    parser.add_argument(
        "--run-name", type=str, default="m2_train",
        help="logger run name (default: m2_train).",
    )
    parser.add_argument(
        "--log-backend", type=str, default="auto",
        help="metrics backend: auto|wandb|tensorboard|jsonl (default: auto).",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="optional path for the LATEST DRQN checkpoint (state_dict). On the "
        "multi-arena path it is rewritten periodically and once at the end, "
        "INDEPENDENT of eval - so a run with eval disabled still leaves weights.",
    )
    parser.add_argument(
        "--best-checkpoint", type=str, default=None,
        help="multi-arena only: path for the SAVE-BEST checkpoint, written only "
        "when an eval strictly improves the win rate against the eval opponent. "
        "Defaults to --checkpoint with a '.best' suffix. Keep it distinct from "
        "--checkpoint: one path for both means the next periodic save overwrites "
        "the best net with a more recent, worse one.",
    )
    parser.add_argument(
        "--checkpoint-every-grad-steps", type=int, default=None,
        help="multi-arena only: grad steps between periodic --checkpoint saves "
        "(default: TrainConfig.checkpoint_interval; 0 disables the periodic save, "
        "the final save still happens).",
    )
    parser.add_argument(
        "--no-progress", action="store_true",
        help="disable the live status bar / progress log (default: enabled).",
    )
    parser.add_argument(
        "--progress-interval", type=float, default=30.0,
        help="seconds between persistent progress lines / metrics rows "
        "(the on-TTY bar redraws faster) (default: 30).",
    )
    return parser


#: ``(argparse dest, TrainConfig field, cast)`` for every flag that overrides a
#: config default only when it is actually given. Kept as a table so adding a
#: knob is one line and cannot forget the ``is not None`` guard that keeps
#: TrainConfig the single source of defaults.
_CONFIG_OVERRIDE_FLAGS: Tuple[Tuple[str, str, Callable[[Any], Any]], ...] = (
    ("opponent_mix_easy", "opponent_mix_easy", float),
    ("opponent_mix_easy_after", "opponent_mix_easy_after", float),
    ("opponent_gate_winrate", "opponent_gate_winrate", float),
    ("opponent_gate_window", "opponent_gate_window", int),
    ("eval_opponent_preset", "eval_opponent_preset", str),
    ("warm_start", "warm_start", str),
    ("warm_start_eps_start", "warm_start_eps_start", float),
)


def _config_from_args(args: Any) -> TrainConfig:
    """Build the run's :class:`TrainConfig` from parsed CLI args.

    Always stamps ``seed`` / ``arenas`` / ``opponent`` (they have real argparse
    defaults); applies every entry of :data:`_CONFIG_OVERRIDE_FLAGS` only when the
    flag was actually passed, so an omitted flag keeps the dataclass default
    rather than re-declaring it in two places.

    ``TrainConfig.__post_init__`` validates the result, so a bad CLI value (a
    mixture outside [0, 1], an unknown eval preset, an empty ``--warm-start``)
    fails here — before a fleet is touched — with the field named.

    Args:
        args: The parsed argparse namespace.

    Returns:
        The validated config for this run.
    """
    import dataclasses

    overrides: Dict[str, Any] = {
        "seed": int(args.seed),
        "arenas": int(args.arenas),
        "opponent": str(args.opponent),
    }
    for dest, field_name, cast in _CONFIG_OVERRIDE_FLAGS:
        value = getattr(args, dest, None)
        if value is not None:
            overrides[field_name] = cast(value)
    return dataclasses.replace(TrainConfig(), **overrides)


def _resolve_stop_on_pass(explicit: Optional[bool], cfg: TrainConfig) -> bool:
    """Resolve ``--stop-on-pass`` / ``--no-stop-on-pass``, defaulting by opponent.

    An explicit flag always wins. With neither, the default is ``True`` for the
    dummy (today's M2 behavior, unchanged) and ``False`` for the scripted
    opponent.

    The asymmetry is not a preference, it is what the gate MEANS. ``passed_m2``
    is AC6's gate: >= 95% win rate **against the stationary dummy**. A retrain
    whose eval fights a moving opponent that clears the same arithmetic has not
    certified M2, so stopping the night on it ends a multi-hour run early on a
    verdict about a different opponent — and a warm-started agent can clear it in
    its very first eval, reporting success minutes in having learned nothing.

    Args:
        explicit: The parsed flag (``None`` when neither form was given).
        cfg: The run config, read for ``cfg.opponent``.

    Returns:
        Whether the run stops as soon as an eval clears the M2 gate.
    """
    if explicit is not None:
        return bool(explicit)
    return cfg.opponent == "dummy"


def _best_checkpoint_path(
    checkpoint: Optional[str], explicit: Optional[str]
) -> Optional[str]:
    """Return where the SAVE-BEST checkpoint goes (``None`` == no best save).

    An explicit ``--best-checkpoint`` wins; otherwise it is derived from
    ``--checkpoint`` by inserting ``.best`` before the extension
    (``runs/m3.pt`` -> ``runs/m3.best.pt``). Deriving rather than sharing is the
    point: the periodic/final hook rewrites ``--checkpoint`` on a cadence, so a
    best net saved to the same path survives only until the next periodic save —
    which is selection by recency with extra steps.
    """
    if explicit:
        return explicit
    if not checkpoint:
        return None
    import os

    root, ext = os.path.splitext(checkpoint)
    return f"{root}.best{ext or '.pt'}"


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point for the LIVE training run (T20 single-arena / T8 multi-arena).

    Wires a real :class:`~env.mc_pvp_env.TcpBridgeClient` to the bridge, runs
    :func:`train_vs_dummy` to the M2 gate or the budget, logs through a
    :class:`~eval.logging.MetricsLogger`, optionally writes a final checkpoint
    (with the ``code_version`` stamp), and EXITS 0 iff the M2 gate passed.

    Dispatch on ``--arenas``: ``1`` (the default) runs today's EXACT single-arena
    path unchanged (byte-identical, AC1); ``> 1`` engages the multi-arena
    :func:`train_multi_arena` stack (issue #4) over N bridge connections on
    ``--port + i`` (the N bridges/servers must already be started by the human).

    Args:
        argv: Argument vector (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code (0 == passed the M2 gate).
    """
    import dataclasses
    import sys

    from eval.logging import MetricsLogger

    from agent.reward_config import RewardConfig
    from env.mc_pvp_env import TcpBridgeClient

    args = _build_parser().parse_args(argv)

    # Honor the CLI seed by replacing it on the frozen config. For N>1 also stamp the
    # arena count onto the (frozen) config so the multi-arena stack reads it. The
    # opponent source, the curriculum knobs, the eval opponent and the warm start
    # are stamped the same way (TrainConfig validates every one of them).
    cfg = _config_from_args(args)

    # Refuse the one combination that cannot be honored, before anything starts:
    # the single-arena loop steps no opponent policy, so --opponent scripted there
    # would quietly train against the stationary dummy instead.
    if int(args.arenas) < 2 and cfg.opponent != "dummy":
        print(
            f"[train] FATAL: --opponent {cfg.opponent} needs --arenas >1. The "
            "single-arena loop never steps an opponent policy (the dummy is served "
            "by the bridge), so it would silently train against the stationary "
            "dummy while this run's config claimed otherwise. Re-run with "
            "--arenas N (N >= 2) against a booted pad fleet. Aborting.",
            file=sys.stderr,
        )
        return 1

    # Reward coefficients go into the run's own config so the run stays readable
    # without them. `code_version` cannot stand in for this: its SHA half has no
    # `--dirty`, and its `cfg` half hashes only version pins and timing constants,
    # so two runs with different coefficients fingerprint identically. Every
    # archived run paid for that — their coefficients had to be recovered by
    # inverting logged component values. The env builds `RewardConfig()` itself
    # (env/mc_pvp_env.py:438) and nothing overrides it, so these ARE the values
    # this run scores with; if an override is ever added, read it from there.
    reward_cfg = {
        f"reward.{k}": v for k, v in dataclasses.asdict(RewardConfig()).items()
    }

    logger = MetricsLogger(
        run_name=args.run_name,
        backend=args.log_backend,
        config={
            "host": args.host,
            "port": args.port,
            "arenas": args.arenas,
            "opponent": args.opponent,
            "seed": args.seed,
            "max_episodes": args.max_episodes,
            "eval_episodes": args.eval_episodes,
            "code_version": code_version(),
            # The run's own record of the knobs that decide what it learns and
            # what its win rate means. `code_version` cannot stand in for any of
            # them (same reasoning as the reward coefficients below).
            "opponent_mix_easy": cfg.opponent_mix_easy,
            "opponent_mix_easy_after": cfg.opponent_mix_easy_after,
            "opponent_gate_winrate": cfg.opponent_gate_winrate,
            "opponent_gate_window": cfg.opponent_gate_window,
            "eval_opponent_preset": cfg.eval_opponent_preset,
            "warm_start": cfg.warm_start,
            "eps_start_effective": effective_eps_start(cfg),
            "eps_end": cfg.eps_end,
            "eps_decay_episodes": cfg.eps_decay_episodes,
            **reward_cfg,
        },
    )

    checkpoint_path = args.checkpoint
    best_checkpoint_path = _best_checkpoint_path(
        checkpoint_path, getattr(args, "best_checkpoint", None)
    )

    def _save_checkpoint(trainer: "Trainer", grad_step: int) -> None:
        if checkpoint_path is None:
            return
        torch.save(
            {
                "model": trainer.online.state_dict(),
                "grad_step": grad_step,
                "code_version": code_version(),
            },
            checkpoint_path,
        )

    def _save_best_checkpoint(
        trainer: "Trainer", grad_step: int, meta: Mapping[str, Any]
    ) -> None:
        """Write the SAVE-BEST net, stamped with the eval that justified it."""
        if best_checkpoint_path is None:
            return
        torch.save(
            {
                "model": trainer.online.state_dict(),
                "grad_step": grad_step,
                "code_version": code_version(),
                # Freeze day has to know WHAT this file scored and against WHOM;
                # a bare state_dict is how a run's provenance gets lost.
                **dict(meta),
            },
            best_checkpoint_path,
        )

    # N>1 dispatches to the multi-arena live run; N=1 falls through to TODAY'S EXACT
    # single-arena path below (untouched so M2/TC8b cannot regress, AC1/TC15).
    if int(args.arenas) > 1:
        try:
            return _main_multi_arena(
                args,
                cfg,
                logger=logger,
                checkpoint_hook=_save_checkpoint if checkpoint_path else None,
                best_checkpoint_hook=(
                    _save_best_checkpoint if best_checkpoint_path else None
                ),
            )
        finally:
            logger.close()

    def _transport_factory() -> Any:
        return TcpBridgeClient(host=args.host, port=args.port)

    try:
        result = train_vs_dummy(
            cfg,
            transport_factory=_transport_factory,
            max_episodes=args.max_episodes,
            max_grad_steps=args.max_grad_steps,
            eval_every_episodes=args.eval_every_episodes,
            eval_episodes=args.eval_episodes,
            stop_on_pass=_resolve_stop_on_pass(
                getattr(args, "stop_on_pass", None), cfg
            ),
            logger=logger,
            checkpoint_hook=_save_checkpoint if checkpoint_path else None,
            is_live=True,
            log=lambda m: print(m, file=sys.stderr),
            show_progress=not args.no_progress,
            progress_log_interval=args.progress_interval,
            progress_stream=sys.stderr,
        )
    finally:
        logger.close()

    report = result.last_report
    print(
        f"[m2 done] reason={result.stop_reason} episodes={result.episodes_run} "
        f"grad_steps={result.grad_steps} passed_m2={result.passed_m2}",
        file=sys.stderr,
    )
    if report is not None:
        print(
            f"  last eval: win_rate={report.win_rate:.3f} "
            f"mean_len={report.mean_episode_length:.1f} "
            f"aim_invisible={report.aim_while_invisible:.3f}",
            file=sys.stderr,
        )

    return 0 if result.passed_m2 else 1


def _main_multi_arena(
    args: Any,
    cfg: TrainConfig,
    *,
    logger: Any,
    checkpoint_hook: Optional[CheckpointHook],
    best_checkpoint_hook: Optional[BestCheckpointHook] = None,
) -> int:
    """Live multi-arena (N>1) run: wire real clients + the subprocess launcher.

    Constructs, per pad ``i``, an env factory that opens a
    :class:`~env.mc_pvp_env.TcpBridgeClient` to bridge port ``--port + i`` and wraps
    it in an :class:`~env.mc_pvp_env.MCPvPEnv`, then runs :func:`train_multi_arena`.
    The N pads must ALREADY be booted AND PRIMED by the human
    (``server/setup/start-pads.sh --pads N``, which resets every pad before any pad
    may step); T8 only connects clients. The
    :class:`~distributed.launcher.SubprocessArenaLauncher` is imported lazily here so
    the import is paid only on the N>1 path; if it is unavailable the run fails with a
    clear message (the launcher is needed only to RESTART a dead pad's bridge).

    This is where the TWO-TIER FAULT POLICY is armed for a live run, and the only
    place the two tiers are wired to the same numbers:

      * ``--mc-port`` is handed to BOTH the launcher (which refuses to spawn a bridge
        against an unreachable JVM) and the pool's :func:`~distributed.actor.jvm_alive`
        watchdog (whose ``False`` aborts the run). One value, one meaning.
      * A :class:`~distributed.actor.ShutdownSignal` is created HERE, before the
        launcher, because the launcher takes its ``sleep`` at construction. The pool
        sets it on stop/abort, so a shutdown that lands while a collector is inside
        the launcher's bounded relaunch wait unwinds immediately rather than holding
        the thread for the full wait.

    Returns the process exit code (0 == passed the M2 gate).
    """
    import sys

    # Function-local, like the other distributed imports on this path:
    # distributed.actor imports FROM agent.train, so a module-level import here
    # would be a cycle.
    from distributed.actor import ShutdownSignal, jvm_alive

    from env.mc_pvp_env import MCPvPEnv, TcpBridgeClient

    base_port = int(args.port)
    host = str(args.host)
    mc_port = int(getattr(args, "mc_port", _DEFAULT_MC_PORT))

    # THE MC PORT MUST NOT BE A BRIDGE PORT. Everything that probes the mc port
    # CONNECTS to it — the tier-2 watchdog every few seconds, and the launcher's own
    # precondition check before every bridge restart — and that is safe only because
    # Paper is an ordinary multi-client server. BridgeServer is the opposite: it
    # accepts exactly ONE TCP client and resolves a second by DESTROYING the
    # incumbent. Point --mc-port at pad i's bridge and the watchdog evicts that pad's
    # collector on a timer: it faults, recovers, is evicted again, forever, with
    # nothing in the logs naming the cause. The launcher would also "confirm the JVM"
    # by connecting to a bridge. The defaults (25565 vs 5555+i) are far apart; this
    # closes the one hand-edited way to bring them together.
    bridge_ports = range(base_port, base_port + int(cfg.arenas))
    if mc_port in bridge_ports:
        print(
            f"[multi] FATAL: --mc-port {mc_port} falls inside the bridge port range "
            f"{bridge_ports.start}..{bridge_ports.stop - 1} that --arenas "
            f"{cfg.arenas} uses (pad i listens on --port + i). The Minecraft port and "
            f"the bridge ports must be disjoint: a bridge accepts ONE client and "
            f"destroys the incumbent when a second connects, so the JVM watchdog "
            f"would silently evict pad {mc_port - base_port}'s collector on every "
            f"probe. Aborting.",
            file=sys.stderr,
        )
        return 1

    def _env_factory_for(arena_id: int) -> Callable[[], Any]:
        port = base_port + arena_id

        def _build() -> Any:
            # auto_connect=True (default): the collector treats a successful return as
            # a working connection; a connect failure raises BridgeError into its
            # recovery path. One TCP connection per arena (the wire has no arena id).
            transport = TcpBridgeClient(host=host, port=port)
            return MCPvPEnv(
                transport=transport,
                max_episode_steps=MAX_EPISODE_STEPS,
            )

        return _build

    # The launcher is the only piece that cannot be exercised offline. Import it
    # lazily and fail loudly (not silently) if it is missing — it is required to
    # RELAUNCH a dead arena, which the pool may need mid-run.
    try:
        from distributed.launcher import SubprocessArenaLauncher
    except Exception as exc:  # noqa: BLE001 - surface a missing launcher clearly.
        print(
            "[multi] FATAL: could not import distributed.launcher."
            "SubprocessArenaLauncher "
            f"({type(exc).__name__}: {exc}); it is required to relaunch a dead arena "
            "in a multi-arena run. Aborting.",
            file=sys.stderr,
        )
        return 1

    # AC18's last inch. The dummy is knockback-immune and speed-pinned by the
    # datapack; the bridge undoes both ONLY when it is launched with
    # `--dummy-knockback-immune false`, and nothing was ever passing that. A
    # scripted-opponent run against an immune, speed-0 bot trains against a
    # target that cannot be knocked back and (on a minecraft-data bump) cannot
    # walk — with clean logs the whole way.
    dummy_knockback_immune = cfg.opponent == "dummy"

    # The shutdown signal must exist BEFORE the launcher: the launcher takes its
    # polling sleep as a constructor argument, and that sleep is the only hook that
    # can interrupt its bounded wait from outside.
    launcher_shutdown = ShutdownSignal()
    launcher = SubprocessArenaLauncher(
        bridge_base_port=base_port,
        mc_port=mc_port,
        sleep=launcher_shutdown.sleep,
        dummy_knockback_immune=dummy_knockback_immune,
    )

    def _log(message: str) -> None:
        # ASCII-only (Windows cp1252): escape any stray non-ASCII rather than risk a
        # console encode crash mid-run (recorded gotcha).
        safe = message.encode("ascii", "backslashreplace").decode("ascii")
        print(safe, file=sys.stderr, flush=True)

    if not dummy_knockback_immune:
        # This launcher only ever spawns a REPLACEMENT bridge for a pad that
        # died. The pads this run connects to were booted by
        # server/setup/start-pads.sh, which does not pass the flag — so say so
        # here rather than let a whole night train against immune opponents on
        # the strength of a kwarg that only covers relaunches.
        _log(
            "[multi] opponent=scripted: relaunched bridges get "
            "--dummy-knockback-immune false. The pads booted BEFORE this run were "
            "not launched by this process - boot the fleet with "
            "DUMMY_KNOCKBACK_IMMUNE=false or the opponents already running stay "
            "knockback-immune and speed-pinned (AC18). Verify by hitting one and "
            "watching it move; the datapack's tellraw cannot be trusted for this."
        )

    result = train_multi_arena(
        cfg,
        env_factory_for=_env_factory_for,
        launcher=launcher,
        max_episodes=args.max_episodes,
        max_grad_steps=args.max_grad_steps,
        eval_every_grad_steps=args.eval_every_grad_steps,
        eval_episodes=args.eval_episodes,
        stop_on_pass=_resolve_stop_on_pass(getattr(args, "stop_on_pass", None), cfg),
        checkpoint_every_grad_steps=getattr(
            args, "checkpoint_every_grad_steps", None
        ),
        logger=logger,
        checkpoint_hook=checkpoint_hook,
        best_checkpoint_hook=best_checkpoint_hook,
        is_live=True,
        log=_log,
        # Tier 2: the shared JVM, on the same port the launcher just got. The host
        # is deliberately NOT --host (that is the BRIDGE host): the launcher spawns
        # bridges as local child processes and probes the JVM on 127.0.0.1, so the
        # watchdog uses actor.MC_HOST, which is the same loopback address.
        jvm_probe=jvm_alive,
        mc_port=mc_port,
        launcher_shutdown=launcher_shutdown,
    )

    report = result.last_report
    _log(
        f"[multi done] reason={result.stop_reason} "
        f"episodes={result.episodes_received} grad_steps={result.grad_steps} "
        f"passed_m2={result.passed_m2} checkpoints_saved={result.checkpoints_saved}"
    )
    if report is not None:
        _log(
            f"  last eval: win_rate={report.win_rate:.3f} "
            f"mean_len={report.mean_episode_length:.1f} "
            f"aim_invisible={report.aim_while_invisible:.3f}"
        )
    # WHICH FILE TO SHIP. Printed at the end of the run because freeze day picks a
    # checkpoint from this line, not from file mtimes.
    if result.best_grad_step >= 0:
        _log(
            f"  best checkpoint: win_rate={result.best_win_rate:.3f} vs "
            f"{result.eval_opponent} at grad_step {result.best_grad_step}"
        )
    elif result.reports:
        _log(
            f"  no best checkpoint: no eval vs {result.eval_opponent} won a single "
            "episode, so the latest periodic/final checkpoint is all there is"
        )
    return 0 if result.passed_m2 else 1


if __name__ == "__main__":
    raise SystemExit(main())
