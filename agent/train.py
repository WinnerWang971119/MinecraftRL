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

import hashlib
import os
import random
import threading
from collections import deque
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    ClassVar,
    Deque,
    Dict,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
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
from agent.train_config import (
    ASSUMED_MEAN_EPISODE_STEPS,
    ASSUMED_RUN_HOURS,
    EPS_DECAY_FRACTION_OF_RUN,
    TrainConfig,
    eps_decay_episodes_for,
    projected_episodes,
)
from distributed.serialization import Episode
from opponents.scripted_bot import OpponentView, ScriptedBot, ScriptedPreset
from opponents.snapshot_pool import (
    DRAW_SCORE,
    INDEX_FILENAME,
    MatchResult,
    SnapshotPool,
    SnapshotRecord,
)

__all__ = [
    "EnvProtocol",
    "RolloutPolicy",
    "Trainer",
    "train",
    "epsilon_for_episode",
    "effective_eps_start",
    "eps_floor_fraction_of_run",
    "epsilon_log_row",
    "epsilon_schedule_report",
    "per_actor_eps_enabled",
    "per_actor_epsilon",
    "mean_per_actor_epsilon",
    "PerActorEpsilonPolicy",
    "maybe_wrap_per_actor_epsilon",
    "build_arena_policy",
    "load_checkpoint_state_dict",
    "arena_episode_seed",
    "opponent_seed",
    "EpisodeOpponent",
    "ObservationOpponent",
    "OpponentDriver",
    "OpponentCurriculum",
    "ScriptedOpponentDriver",
    "build_scripted_opponents",
    "SnapshotOpponentDriver",
    "build_snapshot_opponents",
    "build_rated_eval_opponent",
    "build_reference_tracks",
    "selfplay_eval_cycle_row",
    "selfplay_log_row",
    "snapshot_pool_directory",
    "SnapshotArchivist",
    "EvalOpponentDriver",
    "build_eval_opponent",
    "build_live_env_factory_for",
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
#: ``(trainer, grad_step, meta, weights)``: the eval metadata that justified the
#: save, so the shipped file can record WHAT it scored and against WHOM — this
#: repo's documented weak spot is exactly that kind of missing run provenance —
#: plus the WEIGHTS to write. The hook MUST serialize ``weights`` and must NOT
#: reach for ``trainer.online.state_dict()``: on the multi-arena path the learner
#: thread keeps stepping the optimizer throughout the eval and the save, so the
#: live net at hook time is thousands of gradient steps past the one that earned
#: the win rate (and reading it mid-``optimizer.step()`` can tear). ``trainer`` is
#: still passed for run-level context (device, grad_step, config), never weights.
BestCheckpointHook = Callable[
    ["Trainer", int, Mapping[str, Any], Mapping[str, Any]], None
]
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


#: Warn at startup when ε reaches its floor before this fraction of a run's
#: PROJECTED episodes. AC15 asks for ~15% (``EPS_DECAY_FRACTION_OF_RUN``); this
#: is a third of that, i.e. the threshold for "this is not a tuning choice, this
#: is the single-arena default on a 25-pad fleet". It only ever LOGS — a config
#: this function dislikes still runs, because a legitimately short run is a thing
#: an operator may want and a refusal at 3am is not.
EPS_FLOOR_WARN_FRACTION: float = 0.05


def eps_floor_fraction_of_run(cfg: TrainConfig) -> float:
    """Return where ε hits its floor, as a fraction of the run's projected episodes.

    ``cfg.eps_decay_episodes / projected_episodes(cfg.arenas)``. The number AC15
    is written against: it should be ~0.15, and it was ~0.001 at 25 pads under
    the old single-arena default of 200.

    Args:
        cfg: The run config (``eps_decay_episodes`` and ``arenas``).

    Returns:
        The fraction (may exceed 1.0 when the decay outlasts the projected run,
        which means ε never reaches ``eps_end`` — the opposite failure and a much
        cheaper one under a warm start).
    """
    return float(cfg.eps_decay_episodes) / projected_episodes(int(cfg.arenas))


def epsilon_log_row(global_episode_count: int, cfg: TrainConfig) -> Dict[str, float]:
    """Return the metrics row describing the fleet's exploration right now.

    All arenas share one episode counter, so they all sit at the SAME schedule
    point — but under the Ape-X spread they do not all ACT at it: arena ``i``
    acts at ``ε ** (1 + i/(N-1)*α)``, and at 25 pads the fleet's mean is roughly
    an order of magnitude below the schedule. Reporting the schedule value under
    the name ``epsilon_mean`` was true before issue #15 and would be false after,
    so the two are separate rows:

      * ``train/epsilon_mean``     — the arenas' true mean effective ε;
      * ``train/epsilon_schedule`` — the shared schedule value they derive from.

    With the spread off the two are equal.

    Args:
        global_episode_count: Episodes claimed so far (``counter.value``); the
            schedule is sampled at the LAST claimed index.
        cfg: The run config.

    Returns:
        A ``{metric_name: value}`` mapping ready for ``MetricsLogger.log``.
    """
    schedule = epsilon_for_episode(max(0, int(global_episode_count) - 1), cfg)
    return {
        "train/epsilon_mean": mean_per_actor_epsilon(schedule, cfg),
        "train/epsilon_schedule": float(schedule),
    }


def epsilon_schedule_report(cfg: TrainConfig) -> List[str]:
    """Return the startup log line(s) describing this run's exploration schedule.

    One descriptive line always, plus a WARNING line when ε reaches its floor
    before :data:`EPS_FLOOR_WARN_FRACTION` of the projected episodes — the shape
    of the bug T16 exists to fix, which a hand-built ``TrainConfig(arenas=25)``
    can still walk into because a dataclass default cannot depend on another
    field.

    It returns lines rather than logging them so the content is unit-testable
    without driving a whole multi-arena run.

    Args:
        cfg: The run config.

    Returns:
        One or two ASCII-only lines, ready for the run's log sink.
    """
    projected = projected_episodes(int(cfg.arenas))
    fraction = eps_floor_fraction_of_run(cfg)
    if per_actor_eps_enabled(cfg):
        spread = (
            f"ON (alpha={cfg.per_actor_eps_alpha:g}, arena 0 most exploratory, "
            f"arena {int(cfg.arenas) - 1} at eps**{1.0 + cfg.per_actor_eps_alpha:g})"
        )
    else:
        spread = "OFF (every arena shares one epsilon)"
    lines = [
        f"[multi] epsilon {effective_eps_start(cfg):.3f} -> {cfg.eps_end:.3f} over "
        f"{cfg.eps_decay_episodes} GLOBAL episodes shared by {cfg.arenas} pads "
        f"({fraction * 100:.1f}% of the ~{projected:,.0f} episodes projected for a "
        f"{ASSUMED_RUN_HOURS:g}h run, assuming {ASSUMED_MEAN_EPISODE_STEPS:g} "
        "steps/episode - set --eps-decay-episodes from the smoke run's MEASURED "
        f"mean); per-actor eps {spread}"
    ]
    if fraction < EPS_FLOOR_WARN_FRACTION:
        lines.append(
            f"[multi] WARNING: epsilon reaches its floor after only "
            f"{fraction * 100:.2f}% of this run's projected episodes (target "
            f"~{EPS_DECAY_FRACTION_OF_RUN * 100:.0f}%, AC15). "
            f"eps_decay_episodes={cfg.eps_decay_episodes} looks sized for far fewer "
            f"than {cfg.arenas} pads - every pad claims from the SAME episode "
            f"counter. Pass --eps-decay-episodes "
            f"{eps_decay_episodes_for(int(cfg.arenas))} (or let the CLI derive it) "
            "unless this is deliberate."
        )
    return lines


# ---------------------------------------------------------------------------
# Ape-X per-actor ε (issue #15, T16(f)).
#
# WHY IT IS NEEDED HERE SPECIFICALLY. Every collector claims its episode index
# from ONE shared ``distributed.actor.GlobalEpisodeCounter`` and derives ε from
# it (``distributed/actor.py:672-674``), so all N pads sit at the SAME schedule
# point at all times: at 25 pads the fleet is 25 copies of one explorer. Ape-X
# (Horgan et al. 2018) spreads them instead — arena ``i`` of ``N`` acts at
# ``ε ** (1 + i/(N-1) * α)`` — so the fleet covers a wide exploration band and a
# near-greedy exploit arm at every moment of the run.
#
# WHERE IT IS APPLIED. ``epsilon_for_episode`` stays the ONE global schedule
# (the Collector's call site is unchanged); the per-arena transform is applied by
# :class:`PerActorEpsilonPolicy`, which wraps each arena's ``SnapshotPolicy`` in
# :func:`build_arena_policy`. That keeps the schedule and the spread in separate
# functions, and keeps the whole feature inside this module.
#
# WHAT IT DOES NOT TOUCH: the per-arena episode SEEDS (``arena_episode_seed``)
# and the collected ``Episode`` stamp. An Episode records ``arena_id`` but no ε
# at all (``distributed/serialization.py`` has no epsilon field), so there is no
# stamped-ε provenance to drift — read the effective ε back from ``arena_id``
# plus this run's ``per_actor_eps_alpha``.
# ---------------------------------------------------------------------------


def per_actor_eps_enabled(cfg: TrainConfig) -> bool:
    """True iff this run gives each arena its own Ape-X ε.

    The single ``and`` that expresses "configured on AND actually multi-arena".
    ``cfg.arenas == 1`` is excluded explicitly even though the formula is the
    identity there, so the N=1 path provably never reaches the wrapper.

    Args:
        cfg: The training config.

    Returns:
        Whether :class:`PerActorEpsilonPolicy` should wrap the arena policies.
    """
    return bool(cfg.per_actor_eps) and int(cfg.arenas) > 1


def per_actor_epsilon(
    base_epsilon: float, arena_id: int, arenas: int, alpha: float
) -> float:
    """Return arena ``arena_id``'s ε under the Ape-X spread.

    ``ε_i = base ** (1 + i/(N-1) * α)``. Because ``base <= 1``, a LARGER exponent
    means a SMALLER ε, so:

      * arena ``0`` gets exponent 1 and therefore ``base`` itself — **the most
        exploratory arena**, which is the Ape-X convention and what TC27 pins;
      * arena ``N-1`` gets exponent ``1 + α`` (``base ** 8`` at the default
        α = 7) — the near-greedy exploit arm.

    ``arenas == 1`` returns ``base`` unchanged and is checked BEFORE the
    ``N - 1`` ratio is formed, so there is no division by zero on the
    single-arena path.

    The values are distinct for every arena only while ``0 < base < 1``. At
    ``base == 1`` (episode 0 of a fresh, non-warm-started run) every power is 1,
    and at ``base == 0`` every power is 0; both are correct, and both are
    momentary — the schedule leaves 1.0 after one episode and never reaches 0
    (``eps_end`` is the floor).

    Args:
        base_epsilon: The GLOBAL schedule's ε for this episode, in ``[0, 1]``.
        arena_id: 0-based arena index, in ``[0, arenas)``.
        arenas: Total arena count (>= 1).
        alpha: The spread exponent α (> 0).

    Returns:
        This arena's effective ε.

    Raises:
        ValueError: on ``arenas < 1``, an ``arena_id`` outside ``[0, arenas)``,
            or a ``base_epsilon`` outside ``[0, 1]``. The last one matters: a
            base above 1 would GROW with the exponent and silently invert the
            ordering, making arena 0 the least exploratory.
    """
    n = int(arenas)
    index = int(arena_id)
    if n < 1:
        raise ValueError(f"arenas must be >= 1, got {arenas}")
    if not (0 <= index < n):
        raise ValueError(f"arena_id {arena_id} out of range [0, {n})")
    # `not 0 <= x <= 1` rather than or-ed comparisons so NaN is rejected instead
    # of slipping through as an always-false predicate.
    if not 0.0 <= base_epsilon <= 1.0:
        raise ValueError(f"base_epsilon must be in [0, 1], got {base_epsilon!r}")
    if n == 1:
        # Single arena: the spread is the identity. Returned before the ratio so
        # `N - 1 == 0` can never be a denominator.
        return float(base_epsilon)
    exponent = 1.0 + (index / float(n - 1)) * float(alpha)
    return float(base_epsilon) ** exponent


def mean_per_actor_epsilon(base_epsilon: float, cfg: TrainConfig) -> float:
    """Return the MEAN ε across the fleet for a global-schedule ``base_epsilon``.

    With the spread off (or at N=1) this is ``base_epsilon`` itself. With it on
    the fleet's mean is far below the schedule value — at 25 pads, α=7 and
    base 0.25 the mean is ~0.02 — so a log line that reports the schedule value
    as "mean epsilon" is simply false once issue #15 ships. This is what
    ``train_multi_arena`` logs instead.

    Args:
        base_epsilon: The global schedule's ε for the current episode.
        cfg: The run config (arena count, spread flag, α).

    Returns:
        The arithmetic mean of the arenas' effective ε values.
    """
    if not per_actor_eps_enabled(cfg):
        return float(base_epsilon)
    n = int(cfg.arenas)
    alpha = float(cfg.per_actor_eps_alpha)
    return sum(per_actor_epsilon(base_epsilon, i, n, alpha) for i in range(n)) / n


class PerActorEpsilonPolicy:
    """Wrap one arena's :class:`RolloutPolicy` so it acts under its OWN Ape-X ε.

    A pure pass-through except for :meth:`act`, whose ``epsilon`` argument — the
    GLOBAL schedule value the Collector computed — is mapped through
    :func:`per_actor_epsilon` before it reaches the wrapped policy. Everything
    else (weight refresh, re-seeding, hidden init, and the three Episode-stamp
    attributes) is delegated to the wrapped policy so the wrapper satisfies
    ``RolloutPolicy`` structurally and nothing downstream can tell the difference.

    ``policy_version`` is delegated as a PROPERTY, never copied at construction:
    the wrapped :class:`~distributed.weights.SnapshotPolicy` mutates it inside
    ``maybe_refresh``, and a copy taken here would freeze every collected
    Episode's provenance stamp at ``-1``.

    Args:
        policy: The wrapped per-arena policy (a ``SnapshotPolicy`` live).
        arenas: Total arena count ``N`` (must be > 1 — at N=1 the transform is
            the identity and the wrapper is pointless, so the caller
            :func:`maybe_wrap_per_actor_epsilon` refuses to build one).
        alpha: The spread exponent α.

    Raises:
        ValueError: if ``arenas < 2`` or ``alpha <= 0``.
    """

    def __init__(self, policy: Any, *, arenas: int, alpha: float) -> None:
        if int(arenas) < 2:
            raise ValueError(
                "PerActorEpsilonPolicy needs arenas >= 2 (at N=1 the Ape-X "
                f"exponent is exactly 1 and the wrapper is a no-op), got {arenas}"
            )
        if not float(alpha) > 0.0:  # `not >` so NaN is rejected too
            raise ValueError(f"alpha must be > 0, got {alpha!r}")
        self._policy = policy
        self._arenas = int(arenas)
        self._alpha = float(alpha)

    # -- the one behavior this class adds ---------------------------------

    def epsilon_for(self, base_epsilon: float) -> float:
        """Map a global-schedule ε to THIS arena's ε (see :func:`per_actor_epsilon`)."""
        return per_actor_epsilon(
            base_epsilon, self._policy.arena_id, self._arenas, self._alpha
        )

    def act(
        self,
        obs: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor],
        epsilon: float,
    ) -> Tuple[int, Tuple[torch.Tensor, torch.Tensor]]:
        """Act under this arena's ε rather than the shared schedule value."""
        return self._policy.act(obs, hidden, self.epsilon_for(epsilon))

    # -- pure delegation ---------------------------------------------------

    @property
    def arena_id(self) -> int:
        return self._policy.arena_id

    @property
    def policy_version(self) -> int:
        return self._policy.policy_version

    @property
    def code_version(self) -> str:
        return self._policy.code_version

    def maybe_refresh(self, store: Any) -> None:
        self._policy.maybe_refresh(store)

    def reseed(self, episode_seed: int) -> None:
        self._policy.reseed(episode_seed)

    def init_hidden(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._policy.init_hidden()

    def __getattr__(self, name: str) -> Any:
        # Only reached when normal lookup fails. The `_policy` guard prevents the
        # infinite recursion that would follow if this fired before __init__ bound
        # it (e.g. during copy/unpickle).
        if name == "_policy":
            raise AttributeError(name)
        return getattr(self._policy, name)


def maybe_wrap_per_actor_epsilon(policy: Any, cfg: TrainConfig) -> Any:
    """Return ``policy`` wrapped for Ape-X ε, or unchanged when the spread is off.

    The single decision point, extracted from ``train_multi_arena``'s closure so
    it is directly unit-testable: a test can hand it any object and assert
    whether it comes back wrapped.

    Args:
        policy: One arena's rollout policy.
        cfg: The run config (:func:`per_actor_eps_enabled` decides).

    Returns:
        The same object when the spread is off or ``arenas == 1``; a
        :class:`PerActorEpsilonPolicy` around it otherwise.
    """
    if not per_actor_eps_enabled(cfg):
        return policy
    return PerActorEpsilonPolicy(
        policy, arenas=int(cfg.arenas), alpha=float(cfg.per_actor_eps_alpha)
    )


def build_arena_policy(
    arena_id: int,
    cfg: TrainConfig,
    *,
    net_factory: Callable[[], Any],
    code_version: str = "",
) -> Any:
    """Build arena ``arena_id``'s rollout policy for the multi-arena pool.

    A :class:`~distributed.weights.SnapshotPolicy` over its own net clone, seeded
    from that arena's local-episode-0 seed, wrapped by
    :func:`maybe_wrap_per_actor_epsilon`. Module-level (rather than inline in
    ``train_multi_arena``) so the wrapping is provable by a unit test instead of
    only by a full offline run.

    Args:
        arena_id: 0-based arena index.
        cfg: The run config.
        net_factory: Zero-arg callable returning a fresh net for the clone.
        code_version: Build stamp copied onto every Episode this policy collects.

    Returns:
        The arena's ``RolloutPolicy``.
    """
    from distributed.weights import SnapshotPolicy

    policy = SnapshotPolicy(
        net_factory,
        # The generator seed is the arena's local-episode-0 seed; the policy
        # re-seeds per episode anyway, so this is just a distinct, reproducible
        # starting point per arena.
        generator_seed=arena_episode_seed(cfg, arena_id, 0),
        arena_id=arena_id,
        code_version=code_version,
    )
    return maybe_wrap_per_actor_epsilon(policy, cfg)


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
#: driver owns independent RNG streams — for the scripted driver the EASY/HARD
#: mixture draw plus one per ``ScriptedBot``; for the self-play driver the
#: snapshot draw plus its ε-greedy generator — and they must not collide with
#: each other or with another arena's.
#:
#: APPEND ONLY. The seed for a role is its INDEX in this tuple, so inserting or
#: reordering an entry silently re-seeds every role after it and changes what a
#: previously reproducible run replays. The self-play roles are therefore at the
#: end, behind the four this project already shipped.
_OPPONENT_SEED_ROLES: Tuple[str, ...] = (
    "mixture",
    "easy",
    "hard",
    "eval",
    "snapshot_sample",
    "snapshot_epsilon",
)


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
            ``"hard"``, ``"eval"`` (the periodic eval's own opponent, which
            :func:`build_eval_opponent` seeds from a band no collector owns),
            ``"snapshot_sample"`` (the self-play driver's pool draw) or
            ``"snapshot_epsilon"`` (its ε-greedy ``torch.Generator``).

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


# ---------------------------------------------------------------------------
# Self-play: the frozen past-self opponent (T10, M4 / issues #8-#10).
#
# WHAT CHANGES. The scripted curriculum drives the second fighter from an
# omniscient hand-written bot that reads an :class:`OpponentView`. Self-play
# drives it from a FROZEN SNAPSHOT of this agent's own past policy, which is a
# ``DuelingDRQN`` and therefore needs the one thing a view is not: the 23-dim
# observation vector, computed from the OPPONENT'S seat. ``MCPvPEnv`` serves
# that from ``opponent_observation()`` when it was built with
# ``mirror_opponent=True``.
#
# TWO PROTOCOLS, ONE DISCRIMINATOR. The two opponent kinds cannot share one
# ``act`` signature without one of them silently receiving the wrong argument
# type, so :class:`ObservationOpponent` is a SEPARATE protocol and
# :func:`collect_episode` picks the branch off the ``needs_observation`` class
# attribute. :class:`EpisodeOpponent` is untouched; the scripted path keeps
# every byte of its behavior.
#
# THE MIRROR IS NOT OPTIONAL. ``opponent_observation()`` RAISES on an env built
# without ``mirror_opponent=True`` rather than returning a zeroed world, so a
# missed construction site is a loud failure on the first episode instead of a
# night spent training a frozen net on garbage. Both sites — the training env
# factory and the eval env — are wired below.
# ---------------------------------------------------------------------------


class ObservationOpponent(Protocol):
    """An opponent that decides from the OPPONENT-seat OBSERVATION, not a view.

    Deliberately a SEPARATE protocol from :class:`EpisodeOpponent` rather than a
    widening of it: the two differ in what ``act`` takes (a 23-dim
    ``np.ndarray`` here, an :class:`~opponents.scripted_bot.OpponentView` there)
    and Python would happily pass either object to either implementation.
    Splitting them makes the mismatch a routing decision
    :func:`collect_episode` takes ONCE per episode, off :attr:`needs_observation`,
    instead of a duck-typing accident discovered at the first ``act`` call.

    :class:`SnapshotOpponentDriver` is the only production implementation; a
    test may substitute a recorder.

    Attributes:
        needs_observation: Always ``True``. The discriminator
            :func:`collect_episode` reads to decide between
            ``env.opponent_observation()`` and ``env.raw_opponent_view()``. A
            ``ClassVar`` rather than an instance field so it describes the KIND
            of opponent: one driver cannot route one way this episode and the
            other way the next.
    """

    needs_observation: ClassVar[bool]

    def begin_episode(self) -> None:
        """Called once before an episode's first decision (sample + reset)."""
        ...

    def act(self, obs: np.ndarray) -> int:
        """Return the opponent's macro (``0..N_ACTIONS-1``) for this window.

        Args:
            obs: The OPPONENT seat's ``(OBS_DIM,)`` float32 observation, from
                ``env.opponent_observation()``.
        """
        ...

    def observe_outcome(self, info: Mapping[str, Any]) -> None:
        """Called once with the FINAL step's ``info`` so the episode can be scored."""
        ...


#: Either opponent kind, as :func:`collect_episode` sees it.
#: :class:`~distributed.actor.ActorPool` builds one per arena and only stores it
#: (it never calls into a driver itself); :func:`collect_episode` is the single
#: place either protocol is actually exercised.
OpponentDriver = Union[EpisodeOpponent, ObservationOpponent]


def _needs_observation(opponent: Optional[Any]) -> bool:
    """True iff ``opponent`` wants the mirrored observation rather than a view.

    Read via ``getattr`` with a ``False`` default so an
    :class:`EpisodeOpponent` — which has no such attribute, and must not grow
    one — routes to the historical view path untouched. ``None`` (the
    stationary-dummy path) reads as ``False`` for the same reason. The ``bool``
    coercion makes the return a real boolean rather than whatever the attribute
    happened to hold, so a caller may compare it with ``is``.
    """
    return bool(getattr(opponent, "needs_observation", False))


def snapshot_pool_directory(run_name: str) -> str:
    """Return the snapshot-pool directory for ``run_name`` (``runs/<run>/snapshots``).

    The one place that layout is spelled out, so the pool the training loop
    writes and the pool a restart reloads cannot drift apart by a path typo.

    Args:
        run_name: The run's logger name (``--run-name``), e.g. ``"m4_selfplay"``.

    Returns:
        The relative directory path. :class:`~opponents.snapshot_pool.SnapshotPool`
        creates it if it does not exist.

    Raises:
        ValueError: ``run_name`` is empty or whitespace — that would silently
            collapse the path to ``runs/snapshots`` and merge two runs' pools.
    """
    name = str(run_name).strip()
    if not name:
        raise ValueError(f"run_name must be a non-empty string, got {run_name!r}")
    return os.path.join("runs", name, "snapshots")


class SnapshotOpponentDriver:
    """One arena's self-play opponent: a frozen past-self policy, resampled per episode.

    Implements :class:`ObservationOpponent`. Per episode it draws a snapshot
    from the shared :class:`~opponents.snapshot_pool.SnapshotPool`, loads those
    frozen weights into its OWN net, and plays them; at the end it scores the
    match back into the pool, which is what feeds PFSP and both Elo series.

    Four properties are load-bearing, and each one is a specific failure this
    class exists to prevent:

      * **The net is a PRIVATE CPU clone in ``eval()`` mode.** One per arena —
        ``cfg.arenas`` of them, all live while the learner steps the optimizer;
        putting them on the learner's device would make every arena's episode
        boundary contend with it for the same compute. It is a SECOND frozen
        clone per collector, beside the
        :class:`~distributed.weights.SnapshotPolicy` that already acts for the
        learner, and that one is CPU-pinned too. ``DuelingDRQN.act`` is itself
        ``@torch.no_grad()``, so no autograd graph is ever built here.
      * **The LSTM hidden state is reset in :meth:`begin_episode`.** A DRQN's
        action depends on its whole carried history; a hidden state left over
        from the PREVIOUS episode (a different snapshot, a different fight) is
        not an error anywhere — the net still returns a legal macro — it just
        makes the opponent's behavior a function of the last episode's ending.
        Silent corruption, so it is reset explicitly and asserted by a test.
      * **Two private RNG streams, never the global ones.** ``self._rng`` draws
        the snapshot, ``self._generator`` drives ε-greedy; both are seeded from
        this arena's own band via :func:`opponent_seed`. A stream shared with
        another arena — or with the process-wide RNG — makes each arena's draws
        depend on how the collector threads happened to interleave, so the run
        stops being reproducible from its seed and the arenas stop being
        independent samples.
      * **It acts at ``cfg.opponent_epsilon`` (0.02 by default), not greedily.**
        Two greedy deterministic policies facing each other can lock into the
        same move sequence every episode, producing thousands of near-identical
        trajectories that teach the learner nothing. ``rated=True`` is the one
        thing IN THIS CLASS that overrides that ε, pinning both sides to a
        literal ``0.0`` — see the argument below and
        :func:`build_rated_eval_opponent`. It is for an eval cycle of a bounded
        number of episodes, where the lock-in this ε exists to prevent has
        nothing to corrupt (``eval.evaluate.evaluate`` writes nothing into the
        replay buffer) and where greedy-vs-greedy is the POINT: it is the only
        pairing ``elo/learner_rated`` accepts. A run can of course also reach
        ε=0 by configuring ``opponent_epsilon`` to 0 — which is why that field
        carries its own warning, and why it is not the fix for the eval.

    Args:
        cfg: The training config (``opponent_epsilon``, seeds).
        pool: The SHARED snapshot pool (one per run). Thread-safe; every arena's
            driver samples from and records into the same instance.
        arena_id: 0-based arena index; selects this driver's seed band.
        net_factory: Zero-arg builder for this driver's own net. MUST produce
            the same architecture the learner publishes, or
            ``load_state_dict(strict=True)`` refuses the snapshot loudly.
            Defaults to a stock :class:`~agent.dqn.DuelingDRQN`.
        learner_epsilon: The learner's ε for the FIRST match, before
            :meth:`note_learner_epsilon` reports the real per-episode value.
            Defaults to ``cfg.eps_end`` — the learner's ε FLOOR, deliberately
            not ``0.0``: a match whose ε was never reported must not read as a
            greedy one and slip into the rated Elo series.
        rated: Build the RATED-EVAL driver instead of a training one. Both
            epsilons become a literal ``0.0``, so every match this driver scores
            is ``rated_eligible`` and moves ``elo/learner_rated`` — the AC7
            series and the checkpoint-selection input. Without it that series is
            empty BY CONSTRUCTION: a training driver stamps
            ``cfg.opponent_epsilon`` (0.02) on every record and the learner
            anneals only to ``cfg.eps_end`` (0.05), so no training match can
            ever qualify. Setting ``--opponent-epsilon 0`` is NOT the same fix —
            that makes the frozen opponent greedy in TRAINING too, which is the
            deterministic lock-in the nonzero default prevents. Defaults to
            ``False``, so every existing construction site is unchanged.
        reference: PIN this driver to ONE snapshot for every episode instead of
            drawing a fresh one from the pool. ``None`` (the default) keeps the
            sampling behavior every training driver has. The reference eval
            needs the pin because ``selfplay/win_rate_vs_ref_<id>`` names a
            SPECIFIC snapshot: a driver that resampled would spread one track's
            ten episodes across whatever PFSP happened to draw, and the series
            would be labelled with a snapshot the agent mostly did not fight.
            Passed a PINNED record (which is all
            :meth:`~opponents.snapshot_pool.SnapshotPool.pinned_references`
            returns), so the pool's corruption policy for it is fatal-not-drop —
            which is the correct reading here too: a reference gauntlet that
            silently lost a reference would report a rising curve over a
            shrinking set of opponents.

    Raises:
        ValueError: ``rated=True`` together with a nonzero ``learner_epsilon`` —
            a contradiction, and the direction that fails SILENTLY (the driver
            would look rated and score nothing into the rated series).
    """

    #: The :func:`collect_episode` discriminator: this driver is fed
    #: ``env.opponent_observation()``, never ``env.raw_opponent_view()``.
    needs_observation: ClassVar[bool] = True

    def __init__(
        self,
        cfg: TrainConfig,
        pool: SnapshotPool,
        arena_id: int,
        *,
        net_factory: Optional[Callable[[], Any]] = None,
        learner_epsilon: Optional[float] = None,
        rated: bool = False,
        reference: Optional[SnapshotRecord] = None,
    ) -> None:
        self.arena_id = int(arena_id)
        self._pool = pool
        self._rated = bool(rated)
        self._reference = reference
        if self._rated:
            if learner_epsilon is not None and float(learner_epsilon) != 0.0:
                raise ValueError(
                    "a rated driver plays at epsilon 0.0 on BOTH sides, so "
                    f"learner_epsilon={learner_epsilon!r} contradicts rated=True. "
                    "Pass 0.0 or omit it."
                )
            # LITERAL zeros, not `cfg`-derived ones. `MatchResult.rated_eligible`
            # is exact float equality, so a value that arrived here as 1e-18
            # would leave every eval match unrated and `elo/learner_rated` empty
            # with no error anywhere.
            self._epsilon = 0.0
            self._learner_epsilon = 0.0
        else:
            self._epsilon = float(cfg.opponent_epsilon)
            self._learner_epsilon = float(
                cfg.eps_end if learner_epsilon is None else learner_epsilon
            )

        # CPU only, deliberately and unconditionally — see the class docstring.
        # Not derived from the learner's device: a run that ever gets a GPU
        # learner must NOT quietly move a frozen clone per arena onto it.
        self._device = torch.device("cpu")
        build = net_factory if net_factory is not None else DuelingDRQN
        self._net = build().to(self._device)
        self._net.eval()

        # Private streams, seeded from this arena's own band. `random.Random`
        # for the pool draw (SnapshotPool.sample only needs `.random()`) and a
        # torch.Generator for ε-greedy (what DuelingDRQN.act consumes).
        self._rng = random.Random(
            opponent_seed(cfg, self.arena_id, "snapshot_sample")
        )
        self._generator = torch.Generator(device=self._device)
        self._generator.manual_seed(
            opponent_seed(cfg, self.arena_id, "snapshot_epsilon")
        )

        # Per-episode state. `None` hidden is the LSTM zero-init contract
        # DuelingDRQN.act applies, and `None` record means "no snapshot loaded
        # yet", which observe_outcome treats as nothing to score.
        self._hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self._record: Optional[SnapshotRecord] = None
        self._last_match: Optional[MatchResult] = None

    # -- read side ---------------------------------------------------------

    @property
    def net(self) -> Any:
        """This driver's own frozen net (exposed for tests / introspection)."""
        return self._net

    @property
    def hidden(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """The carried LSTM state; ``None`` at every episode's first decision."""
        return self._hidden

    @property
    def snapshot_id(self) -> Optional[int]:
        """Id of the snapshot loaded for the CURRENT episode, or ``None``."""
        return None if self._record is None else int(self._record.snapshot_id)

    @property
    def record(self) -> Optional[SnapshotRecord]:
        """The :class:`~opponents.snapshot_pool.SnapshotRecord` being played."""
        return self._record

    @property
    def reference(self) -> Optional[SnapshotRecord]:
        """The snapshot this driver is PINNED to, or ``None`` if it samples.

        Set once at construction and never mutated, so a reference track cannot
        drift onto another opponent halfway through its episodes.
        """
        return self._reference

    @property
    def epsilon(self) -> float:
        """This driver's own ε — the ``opponent_epsilon`` side of a match record."""
        return self._epsilon

    @property
    def learner_epsilon(self) -> float:
        """The learner's ε for the current episode, as last reported."""
        return self._learner_epsilon

    @property
    def rated(self) -> bool:
        """True iff this driver was built for a RATED eval (both ε exactly 0.0).

        Every match a rated driver scores is ``rated_eligible``. A training
        driver's matches are not, unless the RUN itself was configured greedy on
        both sides — which the nonzero ``opponent_epsilon`` default exists to
        prevent. The flag is fixed at construction and cannot be set later:
        :meth:`note_learner_epsilon` refuses to move a rated driver off ``0.0``
        rather than silently downgrading it back to an unrated one.
        """
        return self._rated

    @property
    def name(self) -> str:
        """Stable name for logs (``"snapshot_<id>"``, or ``"snapshot"`` before one)."""
        return "snapshot" if self._record is None else f"snapshot_{self.snapshot_id}"

    @property
    def current_match(self) -> Optional[MatchResult]:
        """The most recently SCORED match, or ``None`` before the first outcome.

        Deliberately the scored result and not the in-flight one: a
        :class:`~opponents.snapshot_pool.MatchResult` carries a ``score``, and a
        match still being played has none. Fabricating a placeholder score here
        would put a value into a record whose whole purpose is to be rated. The
        in-flight match's identity is available meanwhile from
        :attr:`snapshot_id`, :attr:`learner_epsilon` and :attr:`epsilon`.
        """
        return self._last_match

    # -- write side --------------------------------------------------------

    def note_learner_epsilon(self, epsilon: float) -> None:
        """Report the LEARNER's ε for the episode about to start.

        The driver cannot derive this: ε comes from the global per-episode
        schedule (plus the Ape-X per-actor spread), which lives on the collector
        side. :func:`collect_episode` calls this immediately before
        :meth:`begin_episode` on every self-play episode; an eval that wants its
        matches to count toward ``elo/learner_rated`` must call it with EXACTLY
        ``0.0`` (see ``TrainConfig.opponent_epsilon`` — eligibility is exact
        float equality, so an ε-adjacent constant empties the rated series
        silently).

        Args:
            epsilon: The learner's exploration rate for this episode, in
                ``[0, 1]``.

        Raises:
            ValueError: ``epsilon`` is outside ``[0, 1]`` or not finite —
                :class:`~opponents.snapshot_pool.MatchResult` would reject it
                later, at scoring time, far from the caller that supplied it.
                Also when a nonzero ε is reported to a :attr:`rated` driver: that
                would silently un-rate an eval cycle, and an empty
                ``elo/learner_rated`` looks exactly like a flat one.
        """
        value = float(epsilon)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"learner epsilon must be in [0, 1], got {epsilon!r}")
        if self._rated and value != 0.0:
            raise ValueError(
                f"a rated driver plays at learner epsilon 0.0; got {epsilon!r}. "
                "Reporting a nonzero epsilon here would make every match of this "
                "eval cycle unrated, and elo/learner_rated would stay flat at "
                "its initial value with nothing reporting why."
            )
        self._learner_epsilon = value

    def begin_episode(self) -> None:
        """Sample this episode's snapshot, load it, and clear the LSTM memory.

        Three steps, all of them before the first :meth:`act`:

          1. ``pool.sample_state_dict`` draws a snapshot AND reads its weights,
             internally dropping-and-resampling past an unpinned member that
             turns out to be missing or corrupt. That retry loop is the pool's,
             not this class's — reimplementing it here would give the fleet two
             disagreeing corruption policies.
          2. ``load_state_dict(strict=True)`` (torch's default) copies the frozen
             weights in; a snapshot from a different architecture fails loudly
             rather than loading a partial net.
          3. The hidden state is cleared, so the first ``act`` of the episode
             starts from a zero LSTM state instead of the last episode's memory.

        ``exclude_id`` is passed as ``None``. It names the learner's own current
        policy version, and this driver has no way to know it: the collector
        carries a :class:`~distributed.weights.WeightStore` version, which is a
        publish counter, not a snapshot id. So the newest snapshot stays
        sampleable — a near-copy of the learner as of the last archive cadence.
        Wiring a real exclusion belongs with whatever owns that cadence, which
        is also the only thing that knows which snapshot id it just wrote.
        Either way the pool's bootstrap rule (snapshot 0 stays sampleable until
        a second distinct version exists) guarantees the very first episode has
        a legal opponent.

        A driver built with a ``reference`` skips step 1 entirely and reloads
        THAT snapshot every episode. Reloading is strictly redundant — the
        record never changes and :meth:`act` cannot write weights — and it is
        done anyway so a pinned driver and a sampling one keep ONE shape: the
        weights an episode plays always came from a pool read at the start of
        that episode, with no second lifecycle to reason about. It costs one
        snapshot read (~2.4 MB) per episode, against an eval cycle measured in
        tens of minutes. ``pool.load_state_dict`` applies the same corruption
        policy the sampling path uses, and for a PINNED record that policy is
        fatal rather than drop-and-resample — the correct reading here too: a
        gauntlet that silently lost a reference would report a rising curve
        over a shrinking set of opponents.

        Raises:
            opponents.snapshot_pool.PinnedSnapshotError: A pinned reference
                snapshot is missing or unreadable — fatal by contract.
            opponents.snapshot_pool.SnapshotPoolError: No sampleable member
                remains. Never silently falls back to an untrained net.
        """
        if self._reference is not None:
            record = self._reference
            state_dict = self._pool.load_state_dict(record)
        else:
            record, state_dict = self._pool.sample_state_dict(
                self._rng, exclude_id=None
            )
        self._net.load_state_dict(state_dict)
        self._record = record
        # AFTER the load, and unconditionally: a hidden state carried across the
        # episode boundary makes this episode's behavior depend on the last
        # one's ending, with nothing anywhere reporting it.
        self._hidden = None

    def act(self, obs: np.ndarray) -> int:
        """Return the frozen snapshot's macro for this decision window.

        Advances the driver's own LSTM by exactly one step and acts ε-greedily
        at :attr:`epsilon` off the driver's private generator.

        Args:
            obs: The OPPONENT seat's ``(OBS_DIM,)`` observation, from
                ``env.opponent_observation()``. Passed through untouched apart
                from the float32/CPU coercion the net requires.

        Returns:
            The macro as an ``int`` in ``[0, N_ACTIONS)`` — ``DuelingDRQN.act``
            can only return an index into its own Q head, so the env's range
            check on ``opp_action`` is unreachable from here.
        """
        if torch.is_tensor(obs):
            obs_tensor = obs.to(dtype=torch.float32, device=self._device)
        else:
            obs_tensor = torch.as_tensor(
                obs, dtype=torch.float32, device=self._device
            )
        action, self._hidden = self._net.act(
            obs_tensor,
            self._hidden,
            epsilon=self._epsilon,
            generator=self._generator,
        )
        return int(action)

    def observe_outcome(self, info: Mapping[str, Any]) -> None:
        """Score the finished episode into the shared pool.

        ``info["won"]`` is the LEARNER's win (this snapshot died), so the score
        is written from the LEARNER's perspective, matching
        :class:`~opponents.snapshot_pool.MatchResult`: 1.0 win, 0.0 loss,
        :data:`~opponents.snapshot_pool.DRAW_SCORE` otherwise — the shared
        constant, because ``SnapshotPool.record_result`` recognizes a draw by
        exact equality with it to count ``selfplay/draw_rate``. "Otherwise"
        covers a timeout AND an episode stopped by the rollout's ``max_steps`` —
        in both, neither fighter died, which is a draw by any reading and must
        not be recorded as a loss for the learner.

        ``lost`` is tested FIRST so a malformed pair with both flags set reads
        as a loss. ``MCPvPEnv`` already resolves them exclusively — a
        simultaneous double death is a loss there, because the learner dying can
        never count as a win — and this keeps the same rule rather than letting
        a fake env or a future producer turn a double death into a win.

        No-ops when no snapshot is loaded — the only way to reach that is an
        ``observe_outcome`` without a preceding ``begin_episode``, and inventing
        a match against no opponent would corrupt PFSP and Elo alike. An episode
        aborted mid-flight by a ``BridgeError`` never reaches here at all, so a
        lost pad is simply not scored, matching the scripted curriculum.
        """
        record = self._record
        if record is None:
            return
        won = bool(info.get("won", False))
        lost = bool(info.get("lost", False))
        score = 0.0 if lost else (1.0 if won else DRAW_SCORE)
        result = MatchResult.create(
            snapshot_id=int(record.snapshot_id),
            learner_epsilon=self._learner_epsilon,
            opponent_epsilon=self._epsilon,
            score=score,
        )
        self._pool.record_result(result)
        self._last_match = result


def build_snapshot_opponents(
    cfg: TrainConfig,
    *,
    run_name: str = "selfplay",
    snapshot_dir: Optional[str] = None,
    net_factory: Optional[Callable[[], Any]] = None,
    log: Optional[Callable[[str], None]] = None,
) -> Tuple[SnapshotPool, Callable[[int], SnapshotOpponentDriver]]:
    """Build the shared snapshot pool and a per-arena self-play driver factory.

    The self-play twin of :func:`build_scripted_opponents`, with the same
    lock-guarded MEMOIZATION: a given ``arena_id`` gets the SAME driver for the
    whole run, and two arenas can never be handed one object.
    :meth:`distributed.actor.ActorPool.build` requires that — two collectors
    sharing a driver would share one LSTM hidden state (each arena's episode
    stomping the other's memory mid-fight) and one ε-greedy generator, which is
    exactly the cross-arena correlation the per-arena seed bands exist to
    prevent. The memoization is lock-guarded, matching
    :func:`build_scripted_opponents`: nothing in this factory's contract says
    the caller builds every collector on one thread, and a torn read would hand
    two collectors two DIFFERENT drivers for the same arena.

    Seeds the pool's snapshot 0 from ``cfg.warm_start``, ``pinned=True``: the
    run's first opponent is the policy the run itself starts from, and a pinned
    member is never dropped, so PFSP always has a floor opponent and Elo always
    has a fixed yardstick.

    A directory that ALREADY holds a ``pool.json`` is RELOADED through
    :meth:`~opponents.snapshot_pool.SnapshotPool.load` instead, and no snapshot
    is seeded. Constructing a fresh pool over a populated directory would reset
    the id counter to 0 and let the next ``add`` overwrite ``snap_0.pt`` — the
    PINNED first reference — while the index still claimed the whole earlier
    history. The reload also restores the Elo series and every head-to-head
    statistic, so PFSP resumes from what the run actually measured rather than
    from a flat table.

    Args:
        cfg: The training config. Reads ``warm_start`` (snapshot 0's weights),
            ``snapshot_sampling``, ``elo_k``, ``elo_initial``,
            ``opponent_epsilon`` and the seed scheme.
        run_name: The run's name, used to derive the default pool directory via
            :func:`snapshot_pool_directory` (so the default is
            ``runs/selfplay/snapshots``). Ignored when ``snapshot_dir`` is
            given.
        snapshot_dir: Explicit pool directory, which OVERRIDES ``run_name``.
            The multi-arena loop passes its own ``snapshot_dir`` straight
            through; tests point it at a ``tmp_path``.
        net_factory: Zero-arg builder for each driver's own net. MUST match the
            learner's architecture (the multi-arena path passes the same
            ``net_kwargs`` the learner was built with) or a snapshot load fails.
        log: Sink for the pool's loud corruption/drop messages.

    Returns:
        ``(pool, opponent_for)`` — the shared pool and
        ``arena_id -> SnapshotOpponentDriver``.

    Raises:
        ValueError: ``cfg.warm_start`` is ``None``. ``TrainConfig`` already
            refuses that combination for ``opponent == "selfplay"``; this second
            check covers a caller that built the pool from some other config,
            because the alternative is seeding snapshot 0 from a randomly
            initialized net and calling it a past self.
        FileNotFoundError: ``cfg.warm_start`` does not exist.
        opponents.snapshot_pool.PinnedSnapshotError: A reloaded pool is missing
            a pinned snapshot's file — fatal, never recovered from.
        opponents.snapshot_pool.SnapshotPoolError: An existing ``pool.json`` is
            unreadable, of an unknown index version, or malformed.
    """
    if cfg.warm_start is None:
        raise ValueError(
            "build_snapshot_opponents requires cfg.warm_start: the pool's "
            "snapshot 0 IS the warm-start policy, and with no checkpoint to "
            "load the first opponent would be a freshly initialized net "
            "presented as a past self. Pass --warm-start <checkpoint>."
        )

    directory = (
        snapshot_pool_directory(run_name) if snapshot_dir is None else str(snapshot_dir)
    )
    if os.path.isfile(os.path.join(directory, INDEX_FILENAME)):
        pool = SnapshotPool.load(directory, sampling=cfg.snapshot_sampling, log=log)
    else:
        pool = SnapshotPool(
            directory,
            elo_k=cfg.elo_k,
            elo_initial=cfg.elo_initial,
            sampling=cfg.snapshot_sampling,
            log=log,
        )
        pool.add(
            load_checkpoint_state_dict(str(cfg.warm_start)),
            grad_step=0,
            elo=cfg.elo_initial,
            pinned=True,
        )

    drivers: Dict[int, SnapshotOpponentDriver] = {}
    lock = threading.Lock()

    def opponent_for(arena_id: int) -> SnapshotOpponentDriver:
        key = int(arena_id)
        with lock:
            driver = drivers.get(key)
            if driver is None:
                driver = SnapshotOpponentDriver(
                    cfg, pool, key, net_factory=net_factory
                )
                drivers[key] = driver
            return driver

    return pool, opponent_for


def build_rated_eval_opponent(
    cfg: TrainConfig,
    pool: SnapshotPool,
    *,
    arena_id: Optional[int] = None,
    net_factory: Optional[Callable[[], Any]] = None,
    reference: Optional[SnapshotRecord] = None,
) -> Callable[[], SnapshotOpponentDriver]:
    """Return a factory for a RATED-eval self-play opponent (both ε exactly 0.0).

    The counterpart of :func:`build_snapshot_opponents` for the eval side, and
    the piece that makes ``elo/learner_rated`` reachable without making the
    whole run greedy. At the shipped defaults a training driver stamps
    ``cfg.opponent_epsilon`` (0.02) on every ``MatchResult`` while the learner
    anneals only to ``cfg.eps_end`` (0.05), so no training match is ever
    ``rated_eligible`` and the AC7 series is empty by construction until
    something builds a driver like this one. Every match a driver from this
    factory scores is rated, whatever the config says, and moves the series.

    Deliberately NOT memoized and NOT shared with the training drivers. Two
    reasons, both silent if ignored: a shared object would carry one LSTM hidden
    state and one ε-greedy generator across the training/eval boundary, and
    :meth:`SnapshotOpponentDriver.note_learner_epsilon` — which
    :func:`collect_episode` calls on every TRAINING episode with the schedule's
    ε — would knock a rated driver back off ``0.0``. (It now RAISES instead, but
    a separate object is what makes the situation unreachable rather than loud.)

    A FRESH driver per call, matching :func:`build_eval_opponent`: eval #1 and
    eval #40 then start from the same seeded snapshot-sampling stream, so the
    two cycles are comparable in the one respect this side controls.

    Args:
        cfg: The training config (seed scheme; ``cfg.opponent_epsilon`` is
            deliberately NOT read — a rated driver is 0.0 whatever it says).
        pool: The SHARED snapshot pool the training drivers also use. The eval
            rates into the same Elo table and the same head-to-head statistics;
            a second pool would rate against a different set of past selves.
        arena_id: Seed band for the eval driver's private RNG streams. Defaults
            to ``cfg.arenas`` — one past the last collector, the same band
            :func:`build_eval_opponent` uses — so the eval's snapshot draws can
            never coincide with a training arena's. No collision with that
            function's own seed even at the same band: the roles differ
            (``"snapshot_sample"``/``"snapshot_epsilon"`` vs ``"eval"``), and
            :func:`opponent_seed` offsets by role.
        net_factory: Zero-arg builder for the driver's net. MUST match the
            learner's architecture or a snapshot load fails loudly.
        reference: PIN the driver to one snapshot instead of sampling — what the
            reference gauntlet passes, one track per member of
            :meth:`~opponents.snapshot_pool.SnapshotPool.pinned_references`. With
            it, neither of this driver's two RNG streams is ever drawn from: the
            pool draw is replaced by the pin, and ``DuelingDRQN.act`` at
            ``epsilon == 0.0`` short-circuits to ``argmax`` with no RNG touch at
            all. That is why every reference track can share the one eval seed
            band without correlating anything. ``None`` (the default) keeps the
            sampling driver :func:`build_snapshot_opponents` builds.

    Returns:
        ``() -> SnapshotOpponentDriver`` with :attr:`SnapshotOpponentDriver.rated`
        True.
    """
    band = int(cfg.arenas) if arena_id is None else int(arena_id)

    def _build() -> SnapshotOpponentDriver:
        return SnapshotOpponentDriver(
            cfg,
            pool,
            band,
            net_factory=net_factory,
            rated=True,
            reference=reference,
        )

    return _build


#: Episodes per pinned reference in one eval cycle. Three references at ten
#: episodes is thirty armored fights, which at the measured ~95-step bare
#: episode (longer under iron) keeps a cycle near 30-45 minutes; twenty each
#: would put it at 1-2 hours and cost the overnight run several cycles of the
#: AC7 curve it exists to draw.
DEFAULT_REFERENCE_EVAL_EPISODES: int = 10


class _ReferenceTrack(NamedTuple):
    """One leg of the reference gauntlet: WHICH past self, for how many episodes.

    Built by :func:`build_reference_tracks` and consumed by
    :func:`_eval_against_opponent`, which runs every leg over the SAME borrowed
    connection, the same eval env and the same greedy policy — the frozen
    candidate when the cycle has one.

    Attributes:
        snapshot_id: The pinned reference's id — the ``<id>`` in
            ``selfplay/win_rate_vs_ref_<id>``, so this leg's report can be
            attributed to a specific past self rather than to "the pool".
        name: What :attr:`~eval.evaluate.EvalReport.opponent` records
            (``"snapshot_<id>"``). It is passed to ``evaluate`` EXPLICITLY, and
            has to be: ``evaluate`` falls back to the driver's own ``name`` once,
            before the first episode, and :attr:`SnapshotOpponentDriver.name`
            only becomes ``"snapshot_<id>"`` after ``begin_episode`` loads the
            record — a freshly built pinned driver still reports plain
            ``"snapshot"``, which would label every leg of the gauntlet
            identically.
        n_episodes: Episodes in this leg.
        opponent_factory: Zero-arg builder for a FRESH rated driver pinned to
            this reference. Fresh per leg so no LSTM state crosses a track
            boundary, and rated so every match it scores is ``rated_eligible``.
    """

    snapshot_id: int
    name: str
    n_episodes: int
    opponent_factory: Callable[[], SnapshotOpponentDriver]


class _ReferenceOutcome(NamedTuple):
    """One reference leg's result: which past self, and how the candidate did.

    Attributes:
        snapshot_id: The pinned reference fought.
        report: That leg's :class:`~eval.evaluate.EvalReport`. Its ``win_rate``
            is THIS CYCLE's raw rate over ``n_episodes``, which is what the
            checkpoint selection compares; the ``selfplay/win_rate_vs_ref_<id>``
            METRIC is the pool's Beta-smoothed lifetime rate and is a different
            (deliberately smoother) number.
    """

    snapshot_id: int
    report: Any


def build_reference_tracks(
    cfg: TrainConfig,
    pool: SnapshotPool,
    *,
    n_episodes: int = DEFAULT_REFERENCE_EVAL_EPISODES,
    net_factory: Optional[Callable[[], Any]] = None,
    arena_id: Optional[int] = None,
) -> Tuple[_ReferenceTrack, ...]:
    """Build one rated eval track per PINNED reference in ``pool`` (AC8).

    HOWEVER MANY EXIST. The plan creates snapshot 0 as a pinned reference at
    seed and promotes two more at ``cfg.reference_promote_grad_steps``, so a
    cycle before the first promotion has one reference, a cycle after the first
    has two, and only the late run has three. Returning whatever
    :meth:`~opponents.snapshot_pool.SnapshotPool.pinned_references` holds is what
    makes the eval degrade gracefully instead of assuming three and either
    crashing or silently skipping the gauntlet.

    An EMPTY tuple is returned for an empty pool rather than raised on. It is
    unreachable through :func:`build_snapshot_opponents` (snapshot 0 is seeded
    pinned), so it means the caller passed a pool this run does not own — a
    condition the eval reports and survives, because losing a night's training
    to a bad eval argument is the worse outcome.

    Args:
        cfg: The training config (seed band, via :func:`build_rated_eval_opponent`).
        pool: The run's SHARED snapshot pool — the same one the collectors draw
            from, so the gauntlet rates into the same Elo table and the same
            head-to-head statistics ``selfplay/win_rate_vs_ref_<id>`` reads.
            ACCEPTED consequence (S7): those head-to-head stats are also the PFSP
            weighting's input and take EVERY match, so a reference's lifetime
            rate blends this cycle's ~30 ε=0 eval episodes with the thousands of
            ε>0 training episodes against the same snapshot. Sharing is the
            point — a second pool would give the run two disagreeing Elo tables —
            and at that ratio the eval's pull on PFSP is negligible; but
            ``selfplay/win_rate_vs_ref_<id>`` is a two-regime average, and the
            single-cycle rate the checkpoint is actually selected on is the
            ``_ReferenceOutcome`` report, not this series.
        n_episodes: Episodes per reference (see
            :data:`DEFAULT_REFERENCE_EVAL_EPISODES`).
        net_factory: Zero-arg builder for each driver's net; MUST match the
            learner's architecture or the snapshot load fails loudly.
        arena_id: Seed band override, forwarded to
            :func:`build_rated_eval_opponent`.

    Returns:
        One :class:`_ReferenceTrack` per pinned reference, ordered by snapshot
        id ascending (the pool's own order).

    Raises:
        ValueError: ``n_episodes`` < 1 — a zero-episode track would report a
            win rate of 0.0 for a fight that never happened, and that number
            feeds the checkpoint selection.
    """
    episodes = int(n_episodes)
    if episodes < 1:
        raise ValueError(
            f"reference eval episodes must be >= 1, got {n_episodes!r}; a track "
            "with no episodes reports win_rate 0.0 for a fight that never ran, "
            "and that value is a checkpoint-selection input"
        )
    tracks: List[_ReferenceTrack] = []
    for record in pool.pinned_references():
        tracks.append(
            _ReferenceTrack(
                snapshot_id=int(record.snapshot_id),
                name=f"snapshot_{int(record.snapshot_id)}",
                n_episodes=episodes,
                # The factory, not a driver: `_eval_against_opponent` builds one
                # per leg so a track never inherits the previous track's LSTM
                # memory, which would make reference 1's first episodes a
                # function of how reference 0's last one ended.
                opponent_factory=build_rated_eval_opponent(
                    cfg,
                    pool,
                    arena_id=arena_id,
                    net_factory=net_factory,
                    reference=record,
                ),
            )
        )
    return tuple(tracks)


class _ReferenceVerdict(NamedTuple):
    """One eval cycle's gauntlet, reduced to the two numbers selection uses.

    Attributes:
        aggregate: EPISODE-weighted win rate over every reference leg — total
            wins divided by total reference episodes. Episode-weighted rather
            than the mean of the per-leg rates so a shortened leg (a pool with a
            reference added mid-run, a future per-reference episode budget)
            cannot count as much as a full one.
        worst: The lowest per-reference win rate in the cycle. Selection reads
            this as well as ``aggregate`` because the two disagree exactly in
            the case this run is most likely to produce: a policy that grows
            decisive against the two recent references while collapsing against
            the oldest still improves the mean.
        references: How many legs were fought (1..3 in this run).
        episodes: Total reference episodes in the cycle.
    """

    aggregate: float
    worst: float
    references: int
    episodes: int


def _summarize_reference_outcomes(
    outcomes: Sequence[_ReferenceOutcome],
) -> Optional[_ReferenceVerdict]:
    """Reduce a cycle's reference legs to a :class:`_ReferenceVerdict`.

    ``None`` when there were no legs at all (a non-self-play run, or a self-play
    cycle whose pool holds no pinned reference). ``None`` rather than a zeroed
    verdict on purpose: ``None`` is what makes the caller fall back to selecting
    on the scripted yardstick, while a 0.0 aggregate would BE the selection
    number — and 0.0 never clears the selector's "must beat zero" bar, so a run
    with no reference would ship no checkpoint at all and report only that no
    eval won an episode.

    A leg with ``n_episodes == 0`` contributes nothing to either total —
    :func:`build_reference_tracks` refuses to create one, so this only guards a
    hand-built outcome — and when EVERY leg is empty the return is ``None`` for
    the same reason as above.
    """
    total_episodes = 0
    total_wins = 0.0
    worst = 1.0
    seen = 0
    for outcome in outcomes:
        episodes = int(getattr(outcome.report, "n_episodes", 0))
        if episodes <= 0:
            continue
        rate = float(outcome.report.win_rate)
        total_episodes += episodes
        total_wins += rate * episodes
        worst = min(worst, rate)
        seen += 1
    if seen == 0 or total_episodes == 0:
        return None
    return _ReferenceVerdict(
        aggregate=total_wins / total_episodes,
        worst=worst,
        references=seen,
        episodes=total_episodes,
    )


def selfplay_eval_cycle_row(
    report: Optional[Any], verdict: Optional[_ReferenceVerdict]
) -> Dict[str, float]:
    """Return the EVAL-CYCLE metrics row a self-play eval logs at its boundary.

    A row of its own, beside (not merged into) :func:`selfplay_log_row`'s: these
    values are produced only at an eval boundary, while that row is also logged
    on the checkpoint cadence and is deduplicated per grad step. Merging them
    would tie a once-per-eval measurement to the lifetime of a row that a
    checkpoint boundary can claim first, and lose it for good on the steps where
    the two cadences coincide.

    Module-level, like :func:`selfplay_log_row` and for the same reason: the
    exact metric names live in one place and are unit-testable without driving a
    multi-arena run. A typo in one of them costs the demo a curve and raises
    nothing.

    The row, and why each entry is in it:

      * ``selfplay/scripted_win_rate`` — the ABSOLUTE yardstick (T13). Elo and
        the per-reference rates are both relative to a pool of past selves and
        can drift upward together; this one is scored against a fixed scripted
        bot and is directly comparable to the M3 run's number.
      * ``selfplay/reference_win_rate`` — the cycle's episode-weighted aggregate
        across pinned references, and the headline number the checkpoint is
        selected on.
      * ``selfplay/worst_reference_win_rate`` — the cycle's weakest reference,
        the second selection criterion. Logged rather than left implicit because
        a curve where the aggregate climbs while this one falls is the
        specialization failure, and it is invisible in the aggregate alone.
      * ``selfplay/references_evaluated`` — how many legs the cycle actually
        fought. It moves 1 -> 2 -> 3 as T18 promotes references, so an aggregate
        that jumps between cycles can be read against a changing opponent set
        rather than mistaken for a change in the agent.

    Keys are OMITTED rather than zero-filled when the underlying measurement did
    not happen (no reference in the pool, no eval report because the cycle was
    skipped): a logged 0.0 reads as "lost everything", which is a different
    claim from "did not play".

    Args:
        report: The MAIN (scripted yardstick) eval report, or ``None`` if the
            cycle produced none.
        verdict: The reference summary from
            :func:`_summarize_reference_outcomes`, or ``None``.

    Returns:
        Metric name -> value, ready for ``logger.log(row, step=grad_step)``.
        EMPTY when the cycle measured neither — the caller skips the log rather
        than writing a bare row.
    """
    row: Dict[str, float] = {}
    if report is not None:
        row["selfplay/scripted_win_rate"] = float(report.win_rate)
    if verdict is not None:
        row["selfplay/reference_win_rate"] = float(verdict.aggregate)
        row["selfplay/worst_reference_win_rate"] = float(verdict.worst)
        row["selfplay/references_evaluated"] = float(verdict.references)
    return row


def selfplay_log_row(pool: SnapshotPool) -> Dict[str, float]:
    """Return the self-play metrics row for one log step (issue #10, AC7/AC8).

    A module-level builder rather than an inline dict inside the training loop,
    for the same reason :func:`epsilon_log_row` is one: what gets logged is then
    unit-testable without driving a whole multi-arena run, and the exact metric
    names live in one place instead of being retyped at each call site.

    The row, and why each entry is in it:

      * ``elo/learner_rated`` — the AC7 rising-trend series and the
        checkpoint-selection input. Moves ONLY on matches where both sides were
        at ε=0, i.e. rated eval cycles (see :func:`build_rated_eval_opponent`).
      * ``elo/learner_online`` — EVERY match, whatever the epsilons. Dense and
        noisy. It is the pool's documented input for the rating a new snapshot
        is frozen at (see ``SnapshotPool.add``'s ``elo`` argument), and it is
        deliberately NOT the AC7 series nor the checkpoint-selection input. The
        PFSP weighting itself reads ``MatchStats.win_rate()``, not this.
      * ``selfplay/pool_size`` — live snapshots. Flat at 1 all night means the
        archive cadence never fired and every episode was fought against the
        warm start.
      * ``selfplay/matches_scored`` / ``selfplay/rated_matches`` — the two
        denominators. ``rated_matches`` at 0 is what separates "the learner
        stopped improving" from "``elo/learner_rated`` has no data at all";
        both render as a flat line and only this number tells them apart.
      * ``selfplay/draw_rate`` — omitted until at least one match is scored,
        because 0/0 is not a draw rate and logging 0.0 for it would read as "no
        draws" rather than "no matches". A rate near 1.0 in the armored regime
        means episodes are hitting the 600-step cap instead of terminating.
      * ``selfplay/win_rate_vs_ref_<id>`` — one per PINNED reference, and only
        once that reference has actually been played (AC8). Unplayed ones are
        omitted rather than reported at the Beta(1, 1) prior's 0.5, which would
        put a flat "even" segment on the front of a curve that has no data
        behind it.

    Assembled from several independent reads of a pool that arena threads are
    scoring into concurrently, so it is a monitoring snapshot, not an atomic
    one: two entries may be a match apart. Neither ratio can be internally
    inconsistent, though — ``draw_rate`` is computed inside one lock
    acquisition, and ``win_rate()`` runs on the COPY ``stats_for`` took under
    the lock — so nothing here can report a rate above 1.0.

    Args:
        pool: The run's shared snapshot pool.

    Returns:
        Metric name -> value, ready for ``logger.log(row, step=grad_step)`` or
        ``logger.summary(row)``.
    """
    # Read ONCE, not once for the value and once for the guard: an arena thread
    # scoring between the two reads would otherwise emit a row saying zero
    # matches AND carrying a draw rate.
    matches_scored = pool.matches_scored
    row: Dict[str, float] = {
        "elo/learner_rated": float(pool.learner_elo_rated),
        "elo/learner_online": float(pool.learner_elo_online),
        "selfplay/pool_size": float(len(pool)),
        "selfplay/matches_scored": float(matches_scored),
        "selfplay/rated_matches": float(pool.rated_matches),
    }
    if matches_scored > 0:
        row["selfplay/draw_rate"] = float(pool.draw_rate)
    for record in pool.pinned_references():
        stats = pool.stats_for(record.snapshot_id)
        if stats.plays > 0:
            row[f"selfplay/win_rate_vs_ref_{record.snapshot_id}"] = float(
                stats.win_rate()
            )
    return row


class _StampedWeightStore:
    """Wraps a :class:`~distributed.weights.WeightStore` and DATES every publish.

    :class:`SnapshotArchivist` has to freeze the learner's PUBLISHED weights and
    label them with the gradient step those weights came out of. The store alone
    cannot say which step that was: its ``version`` is a publish COUNTER, and
    ``distributed.learner.LearnerLoop._maybe_publish`` fires on K-boundary
    CROSSINGS, so version ``v`` can belong to any grad step at or above
    ``v * cfg.weight_sync_every_k_steps``. Reading ``trainer.grad_step`` from the
    driver thread when it notices a new version is worse still — the learner
    never stopped, so that reads a LATER step than the weights. A snapshot
    labelled with the wrong step corrupts the Elo history and every later
    analysis, and nothing downstream can detect it.

    The stamp is therefore taken HERE, inside :meth:`publish`, on the publishing
    thread. ``LearnerLoop`` reads ``trainer.online.state_dict()`` and calls this
    in one unbroken stretch of the learner thread with no ``learn()`` between the
    two, so ``grad_step_of()`` returns exactly the step those tensors came out
    of.

    Composition rather than a subclass because ``train_multi_arena`` accepts an
    INJECTED ``weight_store``: a wrapper dates whatever object the caller passed,
    where a subclass could only date one this module constructed itself. The
    delegation covers the two methods anything ever calls on a store — ``publish``
    (``LearnerLoop``, plus the warm-start pre-publish below) and ``latest``
    (``distributed.weights.SnapshotPolicy.maybe_refresh``) — and no production
    path ``isinstance``-checks a store, so a duck-typed stand-in is safe here.

    Args:
        store: The real :class:`~distributed.weights.WeightStore` to wrap.
        grad_step_of: Zero-arg reader of the learner's gradient step, called on
            whichever thread publishes (``lambda: trainer.grad_step``).
    """

    def __init__(self, store: Any, grad_step_of: Callable[[], int]) -> None:
        self._store = store
        self._grad_step_of = grad_step_of
        # Guards the (stored snapshot, stamp) PAIR, and is held across the inner
        # publish so a reader can never see version v beside the stamp of v-1 —
        # the one way this class could hand out a wrong grad step. Cheap: the
        # only reader is the archive cadence (once per snapshot_every_grad_steps)
        # and all it does under this lock is read the store's stored pair. The
        # collector threads go through `latest`, which takes no lock of ours.
        self._lock = threading.Lock()
        self._version = -1
        self._grad_step: Optional[int] = None

    def publish(self, state_dict: Any, version: int) -> None:
        """Publish to the wrapped store, recording the caller's grad step."""
        # Read BEFORE the lock: `grad_step_of` is caller-supplied code, and the
        # rule SnapshotPool applies to its `log` sink applies here too — never
        # run injected code while holding a lock another thread waits on.
        stamp = int(self._grad_step_of())
        with self._lock:
            self._store.publish(state_dict, version)
            self._version = int(version)
            self._grad_step = stamp

    def latest(self) -> Any:
        """The wrapped store's ``(state_dict, version)`` — the collector path.

        Takes no lock of this wrapper's: every collector calls this at every
        episode boundary and must not serialize behind a learner publish.
        """
        return self._store.latest()

    def latest_stamped(self) -> Tuple[Any, int, Optional[int]]:
        """Return ``(state_dict, version, grad_step)`` for the newest publish.

        ``state_dict`` is ``None`` and ``version`` is ``-1`` until the first
        publish. ``grad_step`` is ``None`` when this wrapper did not observe the
        publish that produced the stored snapshot — reachable only if something
        published to the wrapped store directly, behind this object's back. The
        caller must skip such a snapshot rather than guess a step for it.
        """
        with self._lock:
            state_dict, version = self._store.latest()
            if int(version) != self._version:
                return state_dict, int(version), None
            return state_dict, int(version), self._grad_step


class SnapshotArchivist:
    """Archives the learner into the snapshot pool on a grad-step cadence (AC5).

    The thing that makes the pool GROW. Without a caller driving this, the pool
    holds exactly one member for the life of the run — snapshot 0, the warm start
    — and the failure is silent and total: PFSP weights a single candidate so its
    weighting means nothing, Elo has one opponent so the rating cannot move,
    ``selfplay/pool_size`` reads 1 all night, and every self-play unit test still
    passes because they exercise the pool directly.

    Three rules, each protecting something specific:

      * **PUBLISHED weights, never ``trainer.online``.** The learner mutates the
        live net from its own thread with no thread-safety against a reader, and
        the published snapshot is the decoupled CPU clone
        :meth:`~distributed.weights.WeightStore.publish` already made for the
        collectors — so archiving it costs no second read of the live net.
      * **The PUBLISHED grad step, not the observed one.** :meth:`maybe_archive`
        runs on the driver thread, which reads a grad step the learner has
        already moved past; only the publish itself knows which step its weights
        came from (see :class:`_StampedWeightStore`).
      * **At most one archive per PUBLISH.** A cadence boundary reached with no
        new publish behind it leaves the archive OWED, not skipped — it lands on
        a later poll. Re-archiving one published version would mint a second
        snapshot id for a single policy version, and that id IS the
        policy-version identity PFSP weights and Elo rates.

    Not thread-safe, and does not need to be: the multi-arena driver loop is the
    sole caller. It holds no lock of its own either — :meth:`SnapshotPool.add`
    does its own locking, keeps its ``torch.save`` off the state lock, and
    persists the index itself.

    Args:
        pool: The run's shared :class:`~opponents.snapshot_pool.SnapshotPool`.
        published: Zero-arg reader of the learner's published weights, returning
            ``(state_dict, version, grad_step)`` — normally
            :meth:`_StampedWeightStore.latest_stamped`. It MUST return published
            weights; handing it ``trainer.online.state_dict()`` would reintroduce
            the read-during-write race this class routes around.
        every_grad_steps: Learner grad steps between archives
            (``cfg.snapshot_every_grad_steps``).
        promote_at: Grad steps at which the archived snapshot is PINNED as a
            permanent reference (``cfg.reference_promote_grad_steps``). A
            promotion boundary is a trigger in its own right, so a threshold that
            is not a multiple of ``every_grad_steps`` still produces its
            reference instead of being silently skipped.
        log: ASCII-only ``str -> None`` sink for the archive lines.

    Attributes:
        archived: Snapshots this archivist has added to the pool.
        promoted: How many of those were pinned as references.
    """

    def __init__(
        self,
        pool: SnapshotPool,
        published: Callable[[], Tuple[Any, int, Optional[int]]],
        *,
        every_grad_steps: int,
        promote_at: Tuple[int, ...] = (),
        log: Optional[Callable[[str], None]] = None,
    ) -> None:
        every = int(every_grad_steps)
        if every < 1:
            raise ValueError(
                f"every_grad_steps must be >= 1, got {every_grad_steps!r}: a "
                "cadence of 0 would archive on every poll of the driver loop"
            )
        self._pool = pool
        self._published = published
        self._every = every
        # Sorted and de-duplicated: two equal thresholds would otherwise be
        # consumed one at a time and pin two snapshots for one intended
        # reference.
        self._promote_at: List[int] = sorted({int(step) for step in promote_at})
        self._log = log
        self._next_at = every
        # -1 so the first publish (version 0) is strictly greater. Per-PROCESS
        # state on purpose, and never seeded from the reloaded pool: a resumed
        # run's WeightStore numbers its versions from 0 again, so a watermark
        # taken from what is on disk (the pool's last grad step, say) would sit
        # above every version this process will ever publish and block every
        # archive for the rest of the run.
        self._last_version = -1
        # Latch for the undated-publish line in `maybe_archive`, holding the
        # version it has already been emitted for. That branch leaves the
        # boundary DUE on purpose (the archive is owed, not skipped), so the
        # driver loop re-enters it on every poll; unlatched, one publish behind
        # the wrapper would repeat one identical line for the rest of the run
        # and bury every other archive line in the log.
        self._undated_version = -1
        self.archived = 0
        self.promoted = 0

    def _emit(self, message: str) -> None:
        if self._log is not None:
            self._log(message)

    def maybe_archive(self, grad_step: int) -> Optional[SnapshotRecord]:
        """Archive one snapshot iff ``grad_step`` crossed a cadence/promotion boundary.

        Args:
            grad_step: The learner's CURRENT gradient step, as the driver loop
                reads it. Used ONLY to test the boundaries — the snapshot itself
                is labelled with the grad step the published weights carry, which
                is at or before this one.

        Returns:
            The new :class:`~opponents.snapshot_pool.SnapshotRecord`, or ``None``
            when no boundary was due or nothing new had been published yet.

        Raises:
            Exception: whatever :meth:`SnapshotPool.add` raises (a failed
                ``torch.save``, an unwritable index). Deliberately NOT swallowed:
                a self-play run whose pool stops growing is the exact silent
                failure this class exists to prevent, so it must end the run
                loudly instead of logging and continuing.
        """
        step = int(grad_step)
        due = step >= self._next_at
        promote = bool(self._promote_at) and step >= self._promote_at[0]
        if not (due or promote):
            return None

        state_dict, version, published_step = self._published()
        if not state_dict:
            # Nothing published yet. `_next_at` deliberately does NOT advance:
            # the archive is OWED, and lands on a later poll.
            return None
        if version <= self._last_version:
            # No publish since the last archive, so these are the very weights
            # already in the pool. A second id for one policy version would be
            # counted twice by PFSP and rated twice by Elo.
            return None
        if published_step is None:
            # The stamping wrapper did not see this publish, so there is no
            # honest grad step to label the snapshot with. Loud, because the only
            # way here is a publish that bypassed the wrapper — but ONCE per
            # undated version, because this branch advances nothing and the
            # boundary it declined stays due on every later poll.
            if version != self._undated_version:
                self._undated_version = int(version)
                self._emit(
                    "[archive] SKIPPED: the published weights carry no grad "
                    "step, so this snapshot could only be labelled with a "
                    "guess. Something published to the WeightStore behind "
                    "the stamping wrapper."
                )
            return None
        if published_step <= 0:
            # Weights published at grad step 0 have taken no optimizer step: they
            # ARE the warm start, which the pool already holds as pinned snapshot
            # 0, and a second id for it would be one policy version wearing two.
            # Reachable only when the first trigger boundary falls before the
            # learner's first real publish (``weight_sync_every_k_steps`` larger
            # than the cadence, or a promotion step below it); the archive stays
            # owed and lands as soon as a trained net is published.
            return None

        pinned = promote
        record = self._pool.add(
            state_dict,
            grad_step=published_step,
            # The learner's rating RIGHT NOW, frozen onto the snapshot forever;
            # the ONLINE series, because the rated one only moves on eval cycles
            # and would leave most snapshots stamped with the initial value.
            elo=self._pool.learner_elo_online,
            pinned=pinned,
        )
        self._last_version = int(version)
        self.archived += 1

        if due:
            # Absolute multiples of the cadence, NOT `step + every`: the driver
            # loop polls, so `step` overshoots the boundary, and compounding that
            # overshoot would stretch the cadence further on every archive across
            # a 24-hour run. A boundary the loop missed entirely collapses into
            # this one archive rather than firing a burst of near-identical
            # snapshots.
            self._next_at = (step // self._every + 1) * self._every
        if promote:
            self.promoted += 1
            # Every threshold at or below `step` is consumed by THIS snapshot. If
            # the loop lagged past two of them they name one policy version, and
            # two pinned records of one net would be two identical yardsticks
            # rather than two references — `pinned_references()` returns however
            # many exist and T13's eval degrades gracefully.
            self._promote_at = [
                threshold for threshold in self._promote_at if threshold > step
            ]

        self._emit(
            f"[archive] snapshot {record.snapshot_id} "
            + ("PINNED as a reference, " if pinned else "")
            + f"from published version {version} at grad_step {published_step} "
            f"(elo {record.elo:.1f}); pool holds {len(self._pool)} snapshot(s), "
            f"{len(self._pool.pinned_references())} pinned"
        )
        return record

    def flush(self) -> None:
        """Write the pool's index one final time. Never raises.

        :meth:`SnapshotPool.record_result` deliberately never touches disk — it
        is the per-episode hot path on every arena thread — so the match
        statistics and BOTH Elo series reach ``pool.json`` only on the next
        :meth:`SnapshotPool.add`. Whatever is scored after the last archive lives
        in memory alone, while a restart and every post-run analysis read the
        file. On a run that ends on its grad-step budget that is up to a full
        cadence of Elo history; on one killed mid-flight, more.

        Safe to call redundantly, and safe to call late. The payload is built
        immediately before the write, so it carries the pool's CURRENT generation
        stamp and ``persist`` never has cause to drop it, and a second call
        writes the newest state rather than replaying an older one — this can
        only move the index forward, never behind what the pool already wrote.

        Swallows its own failure for the same reason the run's other teardown
        steps do: the caller is about to re-raise the learner's own exception, and
        a traceback from here would replace it.
        """
        try:
            self._pool.persist(self._pool.index_payload())
        except Exception as exc:  # noqa: BLE001 - must not mask the real error
            self._emit(f"[archive] final pool index write FAILED: {exc}")


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

    A SELF-PLAY run gets this SAME scripted driver — the "one scripted track"
    of T13, and the reason it is here rather than a self-play-specific opponent.
    Elo is relative: ``elo/learner_rated`` can climb all night while the whole
    pool drifts, because a rating measures the learner against its own past
    selves and nothing anchors that ladder to the ground. A win rate against the
    fixed scripted bot cannot inflate, is the same measurement the M3 run
    produced, and is therefore the one number in this run comparable to anything
    outside it. Returning ``None`` here instead — which is what this function did
    before T13 — pointed the periodic eval at the STATIONARY dummy: a win rate
    against a target that never moves, used to select the demo checkpoint.

    The rated self-play half of the cycle is NOT built here. It is a gauntlet of
    one track per pinned reference rather than a single opponent, so it does not
    fit a ``() -> opponent`` factory at all; see :func:`build_reference_tracks`,
    which ``train_multi_arena`` passes alongside this one.

    Args:
        cfg: The training config.
        preset_choice: Optional override of ``cfg.eval_opponent_preset``.

    Returns:
        ``() -> EvalOpponentDriver``, or ``None`` on the dummy path.
    """
    if cfg.opponent not in ("scripted", "selfplay"):
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
    opponent: Optional[OpponentDriver] = None,
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
        opponent: Optional per-arena opponent driver. ``None`` (the default) is
            the M1/M2 stationary-dummy path and is BYTE-IDENTICAL to what this
            loop did before the parameter existed: ``env.step(action)`` is
            called with one positional argument, no ``opp_action`` reaches the
            wire, and the env is never asked for anything extra.

            Otherwise the driver's ``needs_observation`` attribute selects what
            it is fed, ONCE per episode, before the first decision:

              * absent/False — an :class:`EpisodeOpponent` (the scripted
                curriculum, T12). Each decision reads
                ``env.raw_opponent_view()``.
              * ``True`` — an :class:`ObservationOpponent` (the self-play
                snapshot driver, T10). Each decision reads
                ``env.opponent_observation()``, the OPPONENT seat's 23-dim
                vector, because a frozen ``DuelingDRQN`` needs the exact inputs
                it was trained on and an ``OpponentView`` is not one of them.
                That accessor RAISES unless the env was built with
                ``mirror_opponent=True``, so a missed construction site fails on
                the first episode rather than training an hour against garbage.

            Either way it is ONE read, ONE macro, ONE ``env.step`` per decision
            window — see the one-step-one-window invariant in this section's
            banner. The env must accept ``opp_action`` and expose whichever
            accessor the branch uses (``MCPvPEnv`` does; a fake env used with an
            opponent must too).

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
    # Which accessor this episode's opponent is fed. Resolved ONCE, before the
    # first decision, so the routing cannot change mid-episode and the hot loop
    # pays no attribute lookup per step.
    opponent_observes = _needs_observation(opponent)
    # Episode boundary for the opponent: draw this episode's difficulty tier (or
    # snapshot) and start its episode. AFTER env.reset(), because the reset
    # re-arms the opponent's shadow swing meter that the scripted bot's ATTACK
    # gate reads, and re-primes the mirror the snapshot driver reads.
    if opponent is not None:
        if opponent_observes:
            # Report the LEARNER's ε for this episode so the driver can build a
            # truthful MatchResult: the rated Elo series takes a match only when
            # BOTH epsilons are exactly 0.0, and ε lives on this side of the
            # seam (the schedule plus the Ape-X per-actor spread), not the
            # driver's. Guarded rather than required, because it is NOT part of
            # the ObservationOpponent protocol — a test recorder can omit it.
            # It is never looked up on the view branch, so an EpisodeOpponent
            # is untouched by this whether or not it has such an attribute.
            note_epsilon = getattr(opponent, "note_learner_epsilon", None)
            if callable(note_epsilon):
                note_epsilon(float(epsilon))
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
        elif opponent_observes:
            # ONE mirrored observation, ONE macro, ONE step. Same invariant as
            # the view branch below, different accessor: a frozen DuelingDRQN
            # decides from the OPPONENT seat's 23-dim vector. Read here rather
            # than cached from an earlier step so the snapshot decides from the
            # same window the agent just acted on; the env computed it EAGERLY
            # when it ingested that state, so this read is a cache hit and does
            # not age the mirror's perception memory.
            opp_action = opponent.act(env.opponent_observation())
            next_obs, reward, done, info = env.step(action, opp_action=opp_action)
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

    # Score the finished episode — into the curriculum gate on the scripted
    # path, into the snapshot pool's stats and Elo on the self-play one. The
    # FINAL step's info holds the terminal verdict (``won`` / ``lost`` /
    # ``timeout``); an episode stopped by ``max_steps`` carries won=False, which
    # is the right reading — it did not win. The loop body always runs at least
    # once, so ``last_info`` is only None in a pathological env.
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
                # The single-arena loop is single-threaded: nothing mutates the
                # net between the eval and this hook, so it reads the report and
                # ignores the outcome's weight snapshot.
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
                ).report
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


class _EvalOutcome(NamedTuple):
    """One eval's report plus the weights that produced it.

    ``report`` alone is not enough to select a checkpoint. On the multi-arena
    path only the DESIGNATED arena's collector is paused for an eval; the learner
    thread keeps stepping the optimizer on the other N-1 arenas the whole time,
    so by the moment the save-best hook fires ``trainer.online`` can be thousands
    of gradient steps past the net the win rate was measured on. Carrying the
    weights (and the grad step they were taken at) alongside the report is what
    makes "ship the best checkpoint" mean the net that actually earned the score.

    Attributes:
        report: The :class:`~eval.evaluate.EvalReport` of the MAIN track — the
            scripted yardstick on a scripted or self-play run, the dummy on an
            M2 one. This is the report ``passed_m2`` and the eval log line read.
        weights: A detached CPU clone of ``trainer.online.state_dict()`` taken
            immediately BEFORE the eval ran — or, when the eval ran against a
            frozen :class:`_FrozenCandidate`, that candidate's weights, which
            are the bytes the tracks were actually scored on. Safe to serialize
            from any thread.
        grad_step: ``trainer.grad_step`` as of that clone — the number that says
            which net the file holds. Read after the eval it would name a net
            that never sat the exam.
        reference_outcomes: One :class:`_ReferenceOutcome` per pinned reference
            fought in the SAME cycle, on the SAME weights (T13/AC8). Empty on
            every non-self-play path and on a self-play cycle whose pool holds
            no pinned reference, so a caller must treat "no references" as a
            real state rather than an error.
    """

    report: Any
    weights: Mapping[str, Any]
    grad_step: int
    reference_outcomes: Tuple[_ReferenceOutcome, ...] = ()


#: Filename of the frozen eval candidate inside the snapshot-pool directory.
#: One file, overwritten every cycle: it is a staging area for the CURRENT
#: cycle's exam paper, not an archive (the archive is the pool, owned by
#: :class:`SnapshotArchivist`, and this file is deliberately not a pool member —
#: it must never become a sampleable opponent).
EVAL_CANDIDATE_FILENAME: str = "eval_candidate.pt"


class _FrozenCandidate(NamedTuple):
    """ONE immutable net, on disk, that a whole eval cycle is scored on.

    The failure this closes: :class:`~eval.evaluate.DRQNGreedyPolicy` holds
    ``trainer.online`` BY REFERENCE, and on the multi-arena path only the
    designated collector is paused for an eval — the learner thread keeps
    stepping the optimizer for the whole cycle. A four-track, forty-episode
    self-play cycle therefore scores a MOVING target: reference 0 is fought by
    one net and reference 2 by a net thousands of gradient steps later, and the
    three win rates the checkpoint is selected on describe three different
    agents. Nothing raises; the numbers simply do not mean what they say.

    So the candidate is written to disk, read BACK from disk, and loaded into a
    net of its own. The round trip is the point, not ceremony: what the gauntlet
    scores is then byte-for-byte what the save-best hook would ship, rather than
    a live object that merely started out equal to it.

    Attributes:
        policy: The greedy policy over the frozen net. ONE object for the whole
            cycle — every track shares it, which is the strongest available
            statement that the tracks fought the same agent.
        weights: The candidate's ``state_dict`` (detached CPU clone), handed to
            the save-best hook so the shipped file IS the evaluated net.
        grad_step: The learner's gradient step when the clone was taken.
        path: Where the candidate was staged (see
            :data:`EVAL_CANDIDATE_FILENAME`).
    """

    policy: Any
    weights: Mapping[str, Any]
    grad_step: int
    path: str


def _freeze_eval_candidate(
    *,
    trainer: "Trainer",
    policy_cls: Callable[..., Any],
    net_factory: Callable[[], Any],
    directory: str,
    log: Optional[Callable[[str], None]] = None,
) -> Optional[_FrozenCandidate]:
    """Stage ONE immutable on-disk candidate and return a greedy policy over it.

    Clones the learner's live weights, writes them atomically into ``directory``
    under :data:`EVAL_CANDIDATE_FILENAME`, reads them back, and loads them into a
    fresh CPU net. See :class:`_FrozenCandidate` for why the disk round trip is
    load-bearing.

    CPU, unconditionally, like every other frozen clone in this module: the
    learner keeps training on its own device throughout the cycle and must not
    share it with the exam.

    Returns ``None`` — never raises any ``Exception`` — if anything goes wrong
    (an unwritable directory, a full disk, a torch failure, a staged file that
    reads back unloadable). The caller then falls back to the historical
    live-net policy: a cycle scored on a moving target is a degradation worth
    reporting loudly, while an exception here would end a multi-hour run over a
    staging file. EVERY statement that can raise is inside the ``try``,
    including the live-weight clone and the ``grad_step`` read — a promise this
    broad is only worth what its narrowest statement is.

    Args:
        trainer: The learner, read for ``online`` and ``grad_step``.
        policy_cls: The greedy-policy class (``eval.evaluate.DRQNGreedyPolicy``).
            Called as ``policy_cls(net)`` with NO device, so the policy adopts
            the frozen net's own (CPU) device rather than the learner's.
        net_factory: Zero-arg builder producing the learner's architecture; the
            multi-arena loop passes the same one the collectors use.
        directory: Where to stage the file — the run's snapshot-pool directory.
        log: Optional sink for the failure line.

    Returns:
        The :class:`_FrozenCandidate`, or ``None`` if staging failed.
    """
    # Bound BEFORE the try only so the failure line below can name a grad step
    # even when reading it is the thing that failed; the real read is the first
    # statement inside.
    grad_step = -1
    try:
        from distributed.weights import clone_state_dict

        # BEFORE anything else: from here on the learner keeps mutating the live
        # net, so both numbers must be read as close together as possible.
        # INSIDE the try (S1): the docstring promises this function never
        # raises, and a torch failure is exactly what `state_dict()` and
        # `clone_state_dict` can produce — reading them outside would have let
        # the case the promise names end the run anyway. The pairing survives
        # the move: they are still adjacent and still the first thing read.
        grad_step = int(trainer.grad_step)
        weights = clone_state_dict(trainer.online.state_dict())
        path = os.path.join(directory, EVAL_CANDIDATE_FILENAME)
        os.makedirs(directory, exist_ok=True)
        # The same checkpoint shape load_checkpoint_state_dict reads, so the
        # staged file is loadable by every tool in this repo that opens a
        # checkpoint — including a human debugging the morning after.
        _atomic_torch_save(
            {
                "model": weights,
                "grad_step": grad_step,
                "code_version": code_version(),
            },
            path,
        )
        # CPU before the load, not after: `load_state_dict` copies INTO the
        # existing parameters, so moving the module afterwards would already
        # have allocated the candidate on whatever device `net_factory` chose.
        net = net_factory().to(torch.device("cpu"))
        # strict=True (torch's default): a candidate that does not fit the
        # learner's architecture is rejected outright rather than loaded
        # partially, so this cycle can never score a half-initialized net — it
        # degrades to the live-net policy below, and says so.
        net.load_state_dict(load_checkpoint_state_dict(path, map_location="cpu"))
        net.eval()
        # Inside the try as well: `DRQNGreedyPolicy` touches the net (eval mode,
        # parameter device), and a failure there is the same kind of degradation
        # as a failed write — not a reason to end a multi-hour run.
        policy = policy_cls(net)
    except Exception as exc:  # noqa: BLE001 - a staging failure must not end the run
        if log is not None:
            log(
                f"[multi] eval candidate could NOT be frozen at grad_step "
                f"{grad_step} ({type(exc).__name__}: {exc}); this cycle is "
                "scored on the LIVE net, which the learner keeps stepping - "
                "its per-reference win rates describe slightly different "
                "networks and are weaker evidence than usual"
            )
        return None
    return _FrozenCandidate(
        policy=policy, weights=weights, grad_step=grad_step, path=path
    )


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
    mirror_opponent: bool = False,
    candidate: Optional[_FrozenCandidate] = None,
    reference_tracks: Sequence[_ReferenceTrack] = (),
) -> Any:
    """Run ONE greedy (ε=0) eval of the current online net vs the stage opponent.

    ``opponent`` is ``None`` on the stationary-dummy path (the bridge serves the
    dummy and no ``opp_action`` goes on the wire) and an
    :class:`EvalOpponentDriver` when the run fights the scripted opponent — in
    which case the eval steps it exactly as collection does, so the win rate this
    returns is a win rate against the SAME moving opponent training faces.

    ``mirror_opponent`` builds the eval env with the opponent-seat observation
    mirror, and MUST be true whenever ``opponent`` is a self-play
    :class:`SnapshotOpponentDriver` — that driver reads
    ``env.opponent_observation()``, which raises on an env without the mirror.
    It is the SECOND of the two construction sites the self-play wiring has to
    reach, and the more dangerous one: the training factory's mistake surfaces
    on the first episode, this one's only at the first eval cycle, potentially
    an hour into a run.

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
    :func:`~eval.evaluate.evaluate`, and returns an :class:`_EvalOutcome` — the
    :class:`~eval.evaluate.EvalReport` PLUS a detached CPU clone of the weights
    the eval started from, so the save-best path serializes the net that earned
    the score instead of whatever the learner thread has since produced.

    ``candidate`` (T13) replaces BOTH of those with a
    :class:`_FrozenCandidate` — a net staged on disk and read back, plus the
    weights and grad step it was taken at. With it, the eval no longer scores a
    moving target: the policy is the frozen one and the returned weights are the
    exact bytes every track was scored on. ``None`` (the default) keeps the
    historical behavior described in the paragraph below, byte for byte, on the
    dummy and scripted paths.

    ``reference_tracks`` (T13/AC8) are extra legs run over the SAME eval env,
    the SAME borrowed connection and the SAME policy, after the main track:
    ten episodes against each pinned reference, each leg driven by a FRESH rated
    :class:`SnapshotOpponentDriver` whose matches are what fill
    ``elo/learner_rated``. They are run HERE, inside the one env construction and
    the one collector pause, because each track is otherwise a separate borrow of
    a connection the bridge only grants once.

    WITH ``candidate=None``, the scope note stands: the eval runs against the
    LIVE net (the policy holds ``trainer.online`` by reference), so a long eval
    on the multi-arena path scores a moving target — episode 1 and episode 100
    can run on different weights, and the returned clone is "the weights the eval
    started from", not a net frozen for its duration.

    What the clone removes is the multi-thousand-grad-step gap between the eval
    that produced a win rate and the save that recorded it — without it the
    save-best path serializes whatever the learner had reached by save time.
    What it does NOT remove is a torn read:
    :func:`~distributed.weights.clone_state_dict` walks the net's ~12 tensors with
    the learner thread UNPAUSED, and torch releases the GIL inside a large
    ``clone()``, so the optimizer can step between two keys or partway through one
    and the snapshot can mix parameters from ADJACENT gradient steps. That residual is accepted rather than paused around
    because it is bounded to roughly one optimizer step, and every tensor is
    present at its right shape and dtype either way — so the file this writes is
    always structurally valid and loadable.
    """
    from env.mc_pvp_env import MCPvPEnv

    from distributed.weights import clone_state_dict

    if candidate is None:
        # BEFORE evaluate(): from here on the learner thread keeps mutating the
        # live net, so anything read afterwards is a different (later) network.
        eval_grad_step = int(trainer.grad_step)
        weights = clone_state_dict(trainer.online.state_dict())
        policy = policy_cls(trainer.online, device=trainer.device)
    else:
        # The frozen path: nothing here reads the live net at all, which is what
        # makes every track below comparable to every other.
        eval_grad_step = int(candidate.grad_step)
        weights = candidate.weights
        policy = candidate.policy

    # auto_connect=False: the shared transport is already connected by the training
    # env; reconnecting would re-handshake the live bridge mid-run.
    eval_env = MCPvPEnv(
        transport=shared_transport,
        max_episode_steps=env_max_episode_steps,
        auto_connect=False,
        # The self-play eval's second construction site (AC13). False on every
        # other path, which is byte-identical to the pre-M4 call.
        mirror_opponent=bool(mirror_opponent),
    )
    # Switch back to train mode afterward: DRQNGreedyPolicy flips the net to eval()
    # for inference; the next collection/learn step expects train mode. Read even
    # on the frozen path, where nothing touches trainer.online: restoring a mode
    # that was never changed is a no-op, and the guard is cheaper than a branch
    # that has to stay in step with which policy was built.
    was_training = trainer.online.training
    reference_outcomes: List[_ReferenceOutcome] = []
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
        for track in reference_tracks:
            reference_outcomes.append(
                _ReferenceOutcome(
                    snapshot_id=int(track.snapshot_id),
                    report=evaluate(
                        eval_env,
                        policy,
                        n_episodes=int(track.n_episodes),
                        # logger=None ON PURPOSE. `evaluate` logs its per-episode
                        # rows at step=episode_index and writes a run `summary`;
                        # handing it the run's logger would make every track
                        # overwrite the main track's series at steps 0..9 and
                        # end the cycle with the LAST reference's numbers in the
                        # run summary. The per-reference series that survives is
                        # `selfplay/win_rate_vs_ref_<id>`, logged by
                        # `selfplay_log_row` from the pool these matches feed.
                        logger=None,
                        timeout_cap=timeout_cap,
                        base_seed=base_seed,
                        is_live=is_live,
                        max_episode_steps=eval_step_cap,
                        log=log,
                        # A FRESH driver per track: a shared one would carry
                        # reference 0's LSTM memory into reference 1's opening
                        # episodes.
                        opponent=track.opponent_factory(),
                        opponent_name=track.name,
                    ),
                )
            )
    finally:
        # Do NOT close eval_env: it borrows training's socket and must not send
        # `close` or tear down the shared transport. The training env owns and
        # closes that socket exactly once in train_vs_dummy's finally.
        if was_training:
            trainer.online.train()
    return _EvalOutcome(
        report=report,
        weights=weights,
        grad_step=eval_grad_step,
        reference_outcomes=tuple(reference_outcomes),
    )


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
        best_win_rate: Win rate of the best checkpoint that was actually
            WRITTEN — the score of the file on disk, not of the best eval seen.
            ``-1.0`` when no best checkpoint was ever persisted.
        best_grad_step: Learner grad step the PERSISTED best checkpoint holds
            (``-1`` when nothing was ever written as best) — the number that says
            WHICH checkpoint the best file holds.
        eval_opponent: Who the periodic eval fought (``"dummy"`` or e.g.
            ``"scripted_mixed"``). Recorded because ``best_win_rate`` cannot be
            compared across runs — or trusted at all — without it.
        checkpoints_saved: How many times the periodic/final checkpoint hook
            fired. Reported so a run that saved NOTHING is visible in the result
            rather than only discoverable on disk at 8am.
        best_selected_win_rate: The selector's high-water win rate — the highest
            eval seen, whether or not it cleared the selector's "must beat zero"
            bar and whether or not the save that should have recorded it
            succeeded. ``-1.0`` when no eval ran.
        best_selected_grad_step: Grad step of the last SELECTED eval, i.e. the
            checkpoint the run MEANT to ship (``-1`` when nothing was ever
            selectable — no eval ran, or none won an episode). Diverges from
            ``best_grad_step`` exactly when a save failed, which is the only way
            to tell "never won" apart from "won, but the disk did not
            cooperate".
        best_save_failures: How many best-checkpoint saves raised. Non-zero means
            the selected peak is NOT the file on disk; see the
            ``[multi] BEST checkpoint save FAILED`` lines in the run log.
        selection_opponent: WHO ``best_win_rate`` / ``best_selected_win_rate``
            are a win rate against, when that is NOT ``eval_opponent``. Empty on
            every path where the two coincide. A self-play run sets it to a
            description of the reference gauntlet, because its checkpoint is
            selected on the AGGREGATE rate across pinned references while
            ``eval_opponent`` still names the scripted yardstick track — and a
            "best checkpoint: win_rate=0.62 vs scripted_mixed" line reporting a
            number earned against past selves is exactly the kind of false
            summary freeze day picks a checkpoint from.

    SELECTED vs PERSISTED, and why they are two fields: the save hook can raise
    (disk full at 4am, a permission error, a serialization fault) and that
    failure is deliberately swallowed so it cannot end the night. If the result
    reported the selector's high-water mark, the end-of-run "which file to ship"
    line would name a checkpoint that was never written, with a single FAILED
    line in a 12-hour log as the only counter-evidence. So ``best_win_rate`` /
    ``best_grad_step`` describe the FILE, and the ``best_selected_*`` pair
    describes the DECISION.
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
    best_selected_win_rate: float = -1.0
    best_selected_grad_step: int = -1
    best_save_failures: int = 0
    selection_opponent: str = ""

    def __post_init__(self) -> None:
        if self.reports is None:
            self.reports = []


#: ``arena_id -> (zero-arg env builder)``. The builder must return a FRESH,
#: connected env bound to that arena's bridge (so a relaunch can rebuild a fresh
#: client to the same single-connection bridge). The live path builds an
#: ``MCPvPEnv`` over a ``TcpBridgeClient(host, base_port + arena_id)``; tests inject
#: a factory returning a fake env.
EnvFactoryFor = Callable[[int], Callable[[], "EnvProtocol"]]


def build_live_env_factory_for(
    cfg: TrainConfig,
    *,
    host: str,
    base_port: int,
    max_episode_steps: Optional[int] = MAX_EPISODE_STEPS,
) -> EnvFactoryFor:
    """Build the LIVE per-arena env factory: pad ``i`` -> a fresh connected env.

    The TRAINING half of the two ``MCPvPEnv`` construction sites a self-play run
    must reach (the other is the eval env inside
    :func:`_eval_against_opponent`). Both derive their ``mirror_opponent`` flag
    from the same ``cfg.opponent == "selfplay"`` test, so a run cannot mirror on
    one side and not the other.

    Module-level rather than a closure inside :func:`_main_multi_arena` for one
    reason: this is the wiring AC13 is about, and wiring that can only be
    exercised by starting a real fleet is wiring nothing tests. Here a test
    builds the factory from a config and inspects the env it produces.

    Each returned builder opens ONE
    :class:`~env.mc_pvp_env.TcpBridgeClient` to ``base_port + arena_id``.
    ``auto_connect`` stays at its default ``True``: the collector treats a
    successful return as a working connection, and a connect failure raises
    ``BridgeError`` into its recovery path. One TCP connection per arena — the
    wire carries no arena id, and a bridge accepts exactly one client.

    Args:
        cfg: The training config; only ``opponent`` is read (for the mirror).
        host: Bridge host shared by every pad.
        base_port: Pad 0's bridge port; pad ``i`` listens on ``base_port + i``.
        max_episode_steps: Episode truncation handed to every env; ``None``
            disables truncation entirely.

    Returns:
        An :data:`EnvFactoryFor` — ``arena_id -> (() -> MCPvPEnv)``.
    """
    # Function-local, like every other env import in this module (there is no
    # module-level one), so importing agent.train never drags in the env
    # package. Resolved HERE, at factory-build time — the builders below close
    # over these two names, which is the same binding moment the closure this
    # replaced had inside _main_multi_arena.
    from env.mc_pvp_env import MCPvPEnv, TcpBridgeClient

    # Evaluated ONCE, not per arena: every pad in a run mirrors or none does.
    mirror_opponent = cfg.opponent == "selfplay"

    def _factory_for(arena_id: int) -> Callable[[], Any]:
        port = int(base_port) + int(arena_id)

        def _build() -> Any:
            transport = TcpBridgeClient(host=host, port=port)
            return MCPvPEnv(
                transport=transport,
                max_episode_steps=max_episode_steps,
                # Self-play drives the second fighter from a frozen DuelingDRQN,
                # which needs the OPPONENT seat's observation.
                # ``opponent_observation()`` RAISES without this flag, so
                # omitting it here fails on the run's first episode rather than
                # quietly feeding the snapshot nothing.
                mirror_opponent=mirror_opponent,
            )

        return _build

    return _factory_for


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

    A self-play cycle adds a THIRD rule by passing ``worst_reference``: the
    candidate must also beat the incumbent's weakest reference score — AND clear
    ``min_win_rate`` on it, because on the first cycle of a selector there is no
    incumbent to beat. A single aggregate hides the failure this run is most
    likely to produce — a policy that learns to beat the two recent references
    decisively while collapsing against the oldest one still gains on the mean.
    That is not improvement, it is specialization, and it is exactly the
    checkpoint a human challenger with an unfamiliar style would beat on demo day.

    Args:
        min_win_rate: A report must strictly exceed this to be selectable
            (default 0.0 — it must have won at least one eval episode). On a
            self-play cycle BOTH the aggregate and the worst reference must
            exceed it, so a shipped checkpoint has never been swept by any single
            past self.
    """

    def __init__(self, *, min_win_rate: float = 0.0) -> None:
        self.best_win_rate: float = -1.0
        self.best_grad_step: int = -1
        #: The SHIPPED candidate's weakest per-reference win rate. ``-1.0`` until
        #: something ships — the sentinel means "no incumbent", NOT "anything
        #: clears it": the first cycle still has to beat ``min_win_rate`` on this
        #: number, exactly as it does on the aggregate. See :meth:`consider`.
        self.best_worst_reference: float = -1.0
        self._min_win_rate = float(min_win_rate)

    def consider(
        self,
        win_rate: float,
        grad_step: int,
        *,
        worst_reference: Optional[float] = None,
    ) -> bool:
        """Record this eval; return True iff it is the new checkpoint to ship.

        Args:
            win_rate: The cycle's headline rate — the scripted win rate on a
                scripted run, the AGGREGATE across pinned references on a
                self-play one.
            grad_step: The grad step the EVALUATED weights were taken at.
            worst_reference: The cycle's weakest per-reference win rate.
                ``None`` (every non-self-play cycle, and a self-play cycle whose
                pool holds no reference) keeps the historical single-criterion
                rule byte for byte.

        Returns:
            Whether this eval is the new checkpoint to ship.
        """
        rate = float(win_rate)
        if worst_reference is None:
            if rate <= self.best_win_rate:
                return False
            self.best_win_rate = rate
            if rate <= self._min_win_rate:
                # Tracked as the high-water mark, but not worth shipping yet.
                return False
            self.best_grad_step = int(grad_step)
            return True

        # Both bars are the SHIPPED candidate's own scores, and both advance
        # only together, on an actual ship. Advancing one of them for a
        # candidate that was not shipped would raise the bar without changing
        # the incumbent, so a later candidate could be rejected for failing to
        # beat a net nobody is holding.
        floor = float(worst_reference)
        if rate <= self.best_win_rate or floor <= self.best_worst_reference:
            return False
        # BOTH numbers face the absolute bar, not just the aggregate (W1). The
        # incumbent bar above is vacuous on the FIRST cycle of a selector — it is
        # still the -1.0 sentinel — so gating only `rate` here let a cycle that
        # scored 0.60 on the mean while being SWEPT by one reference ship, and
        # then become the incumbent every later candidate is measured against. A
        # fresh run hides it (the first eval predates the first promotion, so
        # there is one reference and aggregate == worst); a RESUMED run does not:
        # this selector is rebuilt per `train_multi_arena` call, so a restart
        # against a reloaded pool of three pinned references re-opens the window
        # with the run's whole history behind it.
        if rate <= self._min_win_rate or floor <= self._min_win_rate:
            return False
        self.best_win_rate = rate
        self.best_worst_reference = floor
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
    mirror_opponent: bool = False,
    candidate: Optional[_FrozenCandidate] = None,
    reference_tracks: Sequence[_ReferenceTrack] = (),
) -> Optional[_EvalOutcome]:
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

    ``mirror_opponent`` is forwarded unchanged to :func:`_eval_against_opponent`,
    which is where the eval env is actually built; see that function for why the
    flag has to reach it. ``candidate`` and ``reference_tracks`` (T13) are
    forwarded the same way: the WHOLE self-play cycle — the scripted yardstick
    plus every pinned-reference leg — runs inside this ONE pause, on this ONE
    borrowed connection. Splitting the gauntlet into a track-per-call would
    pause and resume the designated collector once per reference and let it
    restart episodes in between, and every extra borrow is another chance to
    take a connection the bridge grants exactly once.

    Returns the eval :class:`_EvalOutcome` (report + the weight snapshot the eval
    started from + any per-reference outcomes), or ``None`` if the designated
    collector could not be brought to an idle boundary within ``pause_timeout``
    (e.g. it is mid-relaunch), in which case eval is SKIPPED this cycle rather
    than risking a second connection.

    NOTE: pausing the designated collector does NOT pause the learner — it keeps
    stepping the optimizer off the other N-1 arenas for the whole eval. That is
    why the outcome carries its own weight snapshot; see :func:`_eval_against_opponent`.
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
            mirror_opponent=mirror_opponent,
            candidate=candidate,
            reference_tracks=reference_tracks,
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
    reference_eval_episodes: int = DEFAULT_REFERENCE_EVAL_EPISODES,
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
    snapshot_dir: Optional[str] = None,
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
    anneals off the learner's ``grad_step`` (inside ``trainer.learn()``).

    Exploration: ONE global ε schedule (``epsilon_for_episode`` off the shared
    counter) plus, when ``cfg.per_actor_eps`` and ``cfg.arenas > 1``, the Ape-X
    per-actor spread — arena ``i`` acts at ``ε ** (1 + i/(N-1)*α)``, so arena 0 is
    the most exploratory and arena ``N-1`` is near-greedy
    (:func:`per_actor_epsilon`). ``train/epsilon_mean`` logs the fleet's TRUE mean
    under that spread, and ``train/epsilon_schedule`` the underlying schedule
    value; with the spread off the two are equal.

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
            On a self-play run this sizes the SCRIPTED yardstick track only; the
            reference gauntlet is sized by ``reference_eval_episodes``.
        reference_eval_episodes: Episodes against EACH pinned reference in a
            self-play eval cycle (default
            :data:`DEFAULT_REFERENCE_EVAL_EPISODES`). Ignored entirely on the
            dummy and scripted paths, which have no references. The whole cycle
            costs ``eval_episodes + len(pinned_references) * this``; three
            references at ten is thirty extra armored fights, roughly 30-45
            minutes, and twenty each would cost the night several cycles of the
            AC7 curve.
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
        best_checkpoint_hook: Optional ``(trainer, grad_step, meta, weights) ->
            None`` for the SAVE-BEST net, fired only when an eval strictly
            improves the win rate (see :class:`_BestCheckpointSelector`);
            ``meta`` carries the win rate, the eval opponent and the episode
            count that justified the save, and ``weights`` is the detached CPU
            snapshot of the net that EARNED that win rate (taken before the eval
            ran — the learner never stops, so ``trainer.online`` here is a later
            net). The hook must serialize ``weights``. Keep it pointed at a
            DIFFERENT
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
            line byte-identical. MUST NOT build a TRAINING driver: ``evaluate``
            calls ``observe_outcome`` duck-typed on whatever it is handed, and
            :meth:`ScriptedOpponentDriver.observe_outcome` scores into the shared
            :class:`OpponentCurriculum` and can FIRE its gate — so the eval would
            move the training distribution it exists to measure. Every production
            path passes :class:`EvalOpponentDriver`, which has no
            ``observe_outcome`` at all; this is a note for the next caller, not a
            live bug.
        snapshot_dir: Where the self-play snapshot pool lives. Read only when
            ``cfg.opponent == "selfplay"``; ``None`` falls back to
            ``snapshot_pool_directory("selfplay")``. The live CLI passes
            ``snapshot_pool_directory(--run-name)`` so two runs cannot share one
            pool of past selves, and a test points it at a ``tmp_path``.
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
    # SnapshotPolicy itself is constructed by the module-level build_arena_policy
    # (which also applies the Ape-X ε wrap), so only the store is needed here.
    from distributed.weights import WeightStore

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
    if reference_eval_episodes <= 0:
        # Refused here rather than at the first eval cycle, which on a live run
        # is an hour in: a zero-episode reference track reports win_rate 0.0 for
        # a fight that never happened, and that number selects the checkpoint.
        raise ValueError(
            f"reference_eval_episodes must be > 0, got {reference_eval_episodes}"
        )
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

    # --- the opponent drivers (T12 scripted / T10 self-play) ----------------
    # "dummy" (the default) builds nothing at all: no opponent_for reaches the
    # ACTOR pool, no opp_action reaches the wire, and the M2 path is untouched.
    # "scripted" builds ONE shared curriculum gate plus one driver per arena.
    # "selfplay" builds ONE shared snapshot pool plus one driver per arena.
    #
    # EVERY non-dummy choice MUST have a branch here. A choice that reaches this
    # block without one leaves `opponent_for` at None, and then the whole run
    # trains against the stationary bridge-served dummy while every log line,
    # every metric label and the logger's own config record all say otherwise —
    # a silent night, not a crash.
    curriculum: Optional[OpponentCurriculum] = None
    snapshot_pool: Optional[SnapshotPool] = None
    # None on every other path, which is what keeps the archive cadence off the
    # dummy/scripted runs entirely (they have no pool to archive into).
    snapshot_archivist: Optional[SnapshotArchivist] = None
    opponent_for: Optional[Callable[[int], OpponentDriver]] = None
    # "Does this run mirror the opponent seat?" — the eval env's flag, below.
    # `build_live_env_factory_for` applies the SAME test on the training side,
    # so the two construction sites AC13 names cannot disagree.
    mirror_opponent = cfg.opponent == "selfplay"
    if cfg.opponent == "scripted":
        curriculum, opponent_for = build_scripted_opponents(cfg)
    elif cfg.opponent == "selfplay":
        snapshot_pool, opponent_for = build_snapshot_opponents(
            cfg,
            snapshot_dir=snapshot_dir,
            # The SAME net_kwargs the learner was built with, so a published
            # state_dict loads into a frozen clone strictly. `net_kwargs` is
            # already normalized to a dict above; `_net_factory` is defined
            # further down for the collector policies and does exactly this.
            net_factory=lambda: DuelingDRQN(**net_kwargs),
            log=log,
        )
        # From here on the learner publishes THROUGH the stamping wrapper, so
        # every published snapshot carries the grad step it was produced at and
        # the archive below can label a pool member with a fact rather than an
        # estimate. Only on this branch: a dummy/scripted run keeps whatever
        # store the caller passed, untouched. It MUST be installed before every
        # PUBLISHER is handed the store — `LearnerLoop` below, and the warm-start
        # pre-publish further down — because a publish that bypasses the wrapper
        # carries no stamp, `maybe_archive` refuses to date such a version rather
        # than guess, and the pool then stops growing with only a SKIPPED line to
        # say so. NOT an ordering constraint for the collectors: `latest`
        # delegates to the wrapped store untouched, so `ActorPool.build` below
        # sees every publish whichever of the two objects it is handed.
        weight_store = _StampedWeightStore(weight_store, lambda: trainer.grad_step)
        # AC5's actual mechanism. Without it the pool never grows past snapshot 0
        # and nothing in the run reports that.
        snapshot_archivist = SnapshotArchivist(
            snapshot_pool,
            weight_store.latest_stamped,
            every_grad_steps=cfg.snapshot_every_grad_steps,
            promote_at=cfg.reference_promote_grad_steps,
            log=log,
        )
        # The pool's own line, because "selfplay" in the config record proves
        # only that the flag was parsed. This says a pool exists, where it is,
        # and how many past selves the collectors can actually draw from.
        _emit(
            f"[multi] opponent=selfplay: snapshot pool at "
            f"{snapshot_pool.directory!r} holds {len(snapshot_pool)} snapshot(s), "
            f"sampling={snapshot_pool.sampling}, opponent_epsilon="
            f"{cfg.opponent_epsilon}, archiving every "
            f"{cfg.snapshot_every_grad_steps} grad steps with pinned references "
            f"promoted at {tuple(cfg.reference_promote_grad_steps)}"
        )

    # --- the EVAL's own opponent (T13) --------------------------------------
    # Separate from the collectors' drivers on purpose: the eval must fight the
    # same KIND of opponent training does (or its win rate scores a stationary
    # target and selects the wrong checkpoint) while staying identical across
    # evals (or two win rates are not comparable and "select the best" is
    # meaningless). None on the dummy path.
    #
    # On a SELF-PLAY run this is the scripted YARDSTICK track, not the rated
    # half of the cycle: an Elo ladder measured entirely against past selves can
    # rise while the whole pool drifts, so the cycle also needs one number that
    # cannot inflate and is comparable to the M3 run. The rated half is the
    # reference gauntlet built per cycle below.
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

    # --- the RATED half of a self-play eval cycle (T13 / AC7, AC8) ----------
    # Rebuilt per cycle, deliberately, and not hoisted: `pinned_references()`
    # GROWS during the run (snapshot 0 at seed, then T18's promotions at
    # `cfg.reference_promote_grad_steps`), so a tuple captured once here would
    # pin the gauntlet to whatever existed at startup and the two promoted
    # references would never be fought — with `selfplay/win_rate_vs_ref_<id>`
    # simply never appearing for them and nothing reporting why. Defined below
    # `_net_factory` because it closes over it: the gauntlet's frozen nets must
    # be the learner's architecture or every snapshot load fails.
    def _reference_tracks_now() -> Tuple[_ReferenceTrack, ...]:
        if snapshot_pool is None:
            return ()
        return build_reference_tracks(
            cfg,
            snapshot_pool,
            n_episodes=reference_eval_episodes,
            net_factory=_net_factory,
        )

    policies: Dict[int, Any] = {}

    def _policy_for(arena_id: int) -> Any:
        policy = policies.get(arena_id)
        if policy is None:
            # build_arena_policy also applies the Ape-X per-actor ε wrap (T16(f))
            # when cfg.per_actor_eps is on AND cfg.arenas > 1. It is module-level
            # so that wrapping is unit-testable rather than only observable
            # through a full run.
            policy = build_arena_policy(
                arena_id, cfg, net_factory=_net_factory, code_version=code_ver
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

    # --- what actually reached the disk, as distinct from what was chosen ----
    # The selector above says which eval DESERVES to be the shipped checkpoint;
    # these three say what the save hook managed to write. They diverge whenever
    # a save raises — which is swallowed on purpose so a bad path cannot end the
    # night — and the end-of-run "which file to ship" line reports THESE, so it
    # can never name a checkpoint that does not exist.
    best_saved_win_rate = -1.0
    best_saved_grad_step = -1
    best_save_failures = 0
    # WHO the two win rates above were earned against, when that is not
    # `eval_opponent_name`. Set on the first self-play cycle that actually
    # fought a reference and left alone afterwards, so the end-of-run "which
    # file to ship" line can never label a reference-gauntlet aggregate as a
    # scripted win rate. Empty string, not None, so the caller's `or` fallback
    # is a plain truthiness test.
    selection_opponent = ""

    def _save_latest(grad_step: int, why: str) -> bool:
        """Fire the LATEST-net hook, never letting a save failure kill the run.

        Returns ``True`` iff the hook ran to completion (so a caller can record
        that something genuinely reached the disk); ``False`` when there was no
        hook to call or the call raised.

        KNOWN, ACCEPTED: a "periodic" save reads ``trainer.online.state_dict()``
        (tensor VIEWS) inside the hook while the learner thread may be mid
        ``optimizer.step()``, so the written file can mix pre- and post-step
        parameters. Closing that would mean threading a weight snapshot through
        the 2-arg :data:`CheckpointHook`, which every caller in the repo
        implements — out of scope here. The exposure is bounded: this file is the
        recency fallback, never the selection criterion (that is the save-best
        path, which now writes an explicit clone), and the "final" save below
        runs AFTER the learner thread is joined, so the last file of the run is
        always a clean read.
        """
        nonlocal checkpoints_saved
        if checkpoint_hook is None:
            return False
        try:
            checkpoint_hook(trainer, int(grad_step))
        except Exception as exc:  # noqa: BLE001 - a bad path must not end the night
            _emit(f"[multi] checkpoint save FAILED at grad_step {grad_step}: {exc}")
            return False
        checkpoints_saved += 1
        _emit(f"[multi] checkpoint saved ({why}) at grad_step {grad_step}")
        return True

    def _maybe_log_mean_epsilon(grad_step: int) -> None:
        # The row is built by the module-level epsilon_log_row so what gets logged
        # is unit-testable without driving a whole multi-arena run. It reports the
        # fleet's TRUE mean under the Ape-X spread alongside the shared schedule
        # value they derive from.
        if logger is None:
            return
        logger.log(epsilon_log_row(counter.value, cfg), step=int(grad_step))

    # --- self-play observability: both Elo series + the pool's shape (T12) ---
    # Emitted at grad-step boundaries the loop ALREADY has: the eval cycle (AC7
    # states the rated series is logged each eval cycle) and the periodic
    # checkpoint, so a run started with --eval-every-grad-steps 0 still leaves a
    # curve behind instead of nothing. The latch below is why both call sites are
    # safe: when the two boundaries fall in the SAME loop iteration they carry
    # the same `grad_step`, and one step must not produce two identical rows.
    last_selfplay_log_step = -1

    def _maybe_log_selfplay(grad_step: int) -> None:
        nonlocal last_selfplay_log_step
        if snapshot_pool is None:
            return
        step = int(grad_step)
        if step == last_selfplay_log_step:
            return
        last_selfplay_log_step = step
        row = selfplay_log_row(snapshot_pool)
        if logger is not None:
            logger.log(row, step=step)
        # Said on stderr too, and NOT behind `logger is not None`: this line is
        # the 3am read of whether AC7 will have any data in the morning, and a
        # run configured with no metrics backend is exactly when it matters.
        rated = int(row["selfplay/rated_matches"])
        draw_rate = row.get("selfplay/draw_rate")
        _emit(
            f"[multi grad_step {step}] selfplay: "
            f"elo_rated={row['elo/learner_rated']:.1f} "
            + (
                f"({rated} rated match(es))"
                if rated
                else "(0 rated matches - elo/learner_rated is EMPTY)"
            )
            + f" elo_online={row['elo/learner_online']:.1f} "
            f"pool={int(row['selfplay/pool_size'])} "
            f"matches={int(row['selfplay/matches_scored'])} "
            + (
                "draw_rate=n/a"
                if draw_rate is None
                else f"draw_rate={draw_rate:.3f}"
            )
        )

    def _summarize_selfplay() -> None:
        # The run's FINAL numbers, from the teardown `finally` so a night that
        # ended on a learner error or a 4am Ctrl-C still records what it measured.
        # Wrapped for the same reason `_save_latest` is: the caller re-raises the
        # learner's own exception right after this, and a logger that raises here
        # would replace it with a teardown traceback.
        if snapshot_pool is None or logger is None:
            return
        try:
            logger.summary(selfplay_log_row(snapshot_pool))
        except Exception as exc:  # noqa: BLE001 - must not mask the real error
            _emit(f"[multi] self-play summary FAILED: {exc}")

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
            # Three branches, not two: with `snapshot_pool` folded into the
            # `else` a self-play run opened its own log by announcing
            # "opponent=dummy", which is the exact false-summary this run cannot
            # afford - it is also the true symptom of the wiring bug where
            # `opponent_for` stays None.
            else (
                f"opponent=selfplay (sampling={snapshot_pool.sampling}, "
                f"opponent_epsilon={cfg.opponent_epsilon}), "
                if snapshot_pool is not None
                else "opponent=dummy, "
            )
        )
        + (
            f"eval every {eval_every_grad_steps} grad steps on arena "
            f"{designated_arena} vs {eval_opponent_name}"
            # Said out loud because the cycle's WALL CLOCK is the sum of both
            # tracks: at 25 pads a self-play cycle is one scripted track plus
            # one leg per pinned reference, and an operator sizing the night
            # needs to see the multiplier rather than infer it from a gap in
            # the eval timestamps.
            + (
                f" + {reference_eval_episodes} eps vs EACH pinned reference"
                if snapshot_pool is not None
                else ""
            )
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
    # --- the exploration schedule, said out loud (T16 / AC15) ----------------
    # All N pads draw from ONE episode counter, so eps_decay_episodes is spent N
    # times faster than a single-arena reading of it suggests. Report where the
    # floor actually lands so an operator sees it in the run's opening lines
    # rather than inferring it from a flat epsilon curve at 3am.
    for _line in epsilon_schedule_report(cfg):
        _emit(_line)
    if checkpoint_hook is None and best_checkpoint_hook is None:
        # Said at the START, loudly. A multi-hour run that saves nothing is
        # unrecoverable at 8am, and the only symptom is an empty runs/ directory.
        _emit(
            "[multi] WARNING: no checkpoint hook configured - this run will train "
            "and save NOTHING. Pass --checkpoint (and optionally "
            "--best-checkpoint) before starting an overnight run."
        )
    elif checkpoint_hook is None:
        # --best-checkpoint ALONE is the same blocker wearing a disguise: only
        # checkpoint_hook drives _save_latest, which is BOTH the periodic and the
        # final save, so with it unset the sole write left on this path sits
        # behind a strictly-improving eval win rate above zero. An agent that
        # never wins an eval episode - the likeliest outcome of a hard retrain -
        # leaves an empty runs/ in the morning and a clean log. The config is NOT
        # silently repaired: an operator who meant --checkpoint must see it.
        _emit(
            "[multi] WARNING: --best-checkpoint was given WITHOUT --checkpoint. "
            "The periodic save is DISABLED and the final save is a NO-OP; this "
            "run writes a file ONLY if some eval strictly improves the win rate "
            "above zero. If the agent never wins an eval episode, the whole run "
            "is LOST. Stop now and add --checkpoint <path> unless that is "
            "genuinely what you want."
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

            # --- the snapshot archive cadence (T18 / AC5) -----------------------
            # THE call that makes the pool grow. Delete it and a 24-hour self-play
            # run fights the frozen warm start every single episode: PFSP has one
            # candidate, Elo has one opponent so the rating cannot move, and
            # `selfplay/pool_size` reads 1 all night. Nothing raises. Placed
            # BEFORE the checkpoint/eval blocks so a snapshot archived on this
            # iteration is already counted by the `selfplay/pool_size` those
            # blocks log at this same grad step.
            if snapshot_archivist is not None:
                snapshot_archivist.maybe_archive(grad_step)

            # --- periodic checkpoint of the LATEST net (eval-independent) -------
            if do_periodic_checkpoint and grad_step >= next_checkpoint_at:
                _maybe_log_selfplay(grad_step)
                _save_latest(grad_step, "periodic")
                # Re-read grad_step: saving takes time and the learner kept going.
                next_checkpoint_at = int(trainer.grad_step) + checkpoint_every

            # --- periodic designated-arena eval via pause/handoff ---------------
            if do_eval and grad_step >= next_eval_at:
                # The WHOLE cycle is guarded (W2). It is the scripted track
                # PLUS one `evaluate` per pinned reference - four calls and tens
                # of minutes of live bridge traffic once three references are
                # pinned - and a BridgeError in the last leg (or an
                # `opponent_observation()` refusal from a mis-built env) used to
                # propagate straight through this loop into teardown and END a
                # 24-hour run, throwing away the legs already fought with it.
                # `_save_latest` and the best-checkpoint hook already swallow
                # and shout; eval is the biggest exposure of the three and was
                # the only one left unguarded.
                try:
                    _maybe_log_mean_epsilon(grad_step)
                    # ONE immutable on-disk candidate for the WHOLE cycle (T13).
                    # Built before the collector is paused so the pause window holds
                    # only bridge traffic, and `None` on every non-self-play path,
                    # where the historical live-net policy is unchanged.
                    candidate = (
                        _freeze_eval_candidate(
                            trainer=trainer,
                            policy_cls=DRQNGreedyPolicy,
                            net_factory=_net_factory,
                            directory=snapshot_pool.directory,
                            log=log,
                        )
                        if snapshot_pool is not None
                        else None
                    )
                    outcome = _eval_via_designated_arena(
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
                        # AC13's second construction site. Derived from the SAME cfg
                        # field as the training factory's flag so the two sites
                        # cannot disagree about whether this run mirrors.
                        mirror_opponent=mirror_opponent,
                        candidate=candidate,
                        # The rated gauntlet: one leg per PINNED reference, rebuilt
                        # each cycle so a reference promoted mid-run is fought from
                        # the next cycle on. Empty off the self-play path.
                        reference_tracks=_reference_tracks_now(),
                    )
                    # Advance the boundary past the CURRENT grad step so a long
                    # eval (the learner kept stepping during the borrow) does not
                    # immediately re-fire.
                    next_eval_at = int(trainer.grad_step) + eval_every_grad_steps

                    verdict = _summarize_reference_outcomes(
                        () if outcome is None else outcome.reference_outcomes
                    )
                    # AFTER the gauntlet, not before it (where T12 had to put it,
                    # having nothing to wait for): the rated matches this cycle just
                    # scored are what make `elo/learner_rated` non-empty, so a row
                    # written first reports the PREVIOUS cycle's rating and the
                    # first cycle of a run reports an empty series. Still labelled
                    # with the ITERATION's `grad_step`, never `trainer.grad_step`,
                    # which the learner has moved on during the borrow. When the
                    # periodic-checkpoint boundary above already logged this step,
                    # the latch drops this call — the pool row is then one cycle
                    # behind for that step and fresh again at the next one, which is
                    # the price of never emitting two contradictory rows at one x.
                    _maybe_log_selfplay(grad_step)
                    # The CYCLE's own rates, in their own row and NOT behind the
                    # latch. They are a different measurement with a different
                    # lifetime — one per eval, never once per checkpoint — and they
                    # exist only here, so dropping them on a step the checkpoint
                    # boundary happened to share would lose them for good rather
                    # than delay them. (`_maybe_log_mean_epsilon` above already
                    # writes a second row at this same step; two rows at one step is
                    # this loop's established shape, not a new one.)
                    cycle_row = selfplay_eval_cycle_row(
                        None if outcome is None else outcome.report, verdict
                    )
                    if logger is not None and cycle_row and snapshot_pool is not None:
                        logger.log(cycle_row, step=grad_step)
                    if verdict is not None:
                        _emit(
                            f"[multi grad_step {grad_step}] reference gauntlet: "
                            f"aggregate={verdict.aggregate:.3f} "
                            f"worst={verdict.worst:.3f} over {verdict.references} "
                            f"reference(s), {verdict.episodes} episode(s) - "
                            + " ".join(
                                f"ref{ref.snapshot_id}={ref.report.win_rate:.3f}"
                                for ref in outcome.reference_outcomes
                            )
                        )
                    elif snapshot_pool is not None and outcome is not None:
                        # A self-play cycle that fought NO reference. Reachable only
                        # through a pool with no pinned member, which
                        # `build_snapshot_opponents` cannot produce — so it means the
                        # loop is holding a pool this run does not own, and
                        # `elo/learner_rated` will stay empty all night.
                        _emit(
                            f"[multi grad_step {grad_step}] reference gauntlet ran "
                            "NO references: the snapshot pool holds no pinned "
                            "member, so no rated match was played and "
                            "elo/learner_rated cannot move"
                        )

                    if outcome is not None:
                        report = outcome.report
                        reports.append(report)
                        last_report = report
                        # The grad step the EVALUATED weights were taken at, not
                        # trainer.grad_step now: the learner never stopped, so "now"
                        # names a net this win rate says nothing about.
                        eval_grad_step = int(outcome.grad_step)
                        # WHAT the checkpoint is selected on. A self-play cycle with
                        # references selects on the reference AGGREGATE gated by the
                        # WORST reference; every other path keeps selecting on the
                        # single eval win rate, unchanged.
                        selection_rate = (
                            report.win_rate if verdict is None else verdict.aggregate
                        )
                        worst_reference = None if verdict is None else verdict.worst
                        if verdict is not None:
                            # Deliberately COUNTLESS. The gauntlet grows 1 -> 2 -> 3
                            # as references are promoted, so a label naming this
                            # cycle's count would still be on the result at the end
                            # of a run that later fought three. The count that
                            # matters rides with the checkpoint that earned it, in
                            # the save-best hook's `references_evaluated`.
                            selection_opponent = (
                                "the pinned reference gauntlet (aggregate)"
                            )
                        # Selection is by WIN RATE, not recency (see the selector).
                        if selector.consider(
                            selection_rate,
                            eval_grad_step,
                            worst_reference=worst_reference,
                        ):
                            if best_checkpoint_hook is not None:
                                # `win_rate` is the SELECTION number and
                                # `eval_opponent` names what it was earned against,
                                # so the pair is always self-consistent: the
                                # scripted yardstick on a scripted/dummy run, the
                                # reference aggregate on a self-play one. The
                                # scripted rate is carried alongside rather than
                                # dropped — it is the only number comparable to M3.
                                meta: Dict[str, Any] = {
                                    "win_rate": float(selection_rate),
                                    "eval_opponent": (
                                        selection_opponent or eval_opponent_name
                                    ),
                                    "eval_episodes": int(report.n_episodes),
                                    "mean_episode_length": float(
                                        report.mean_episode_length
                                    ),
                                    "passed_m2": bool(report.passed_m2),
                                    "scripted_win_rate": float(report.win_rate),
                                    "scripted_opponent": eval_opponent_name,
                                    # HOW MUCH the win rate above is worth (S2).
                                    # True: every track fought the same net,
                                    # staged on disk and read back, so these are
                                    # the bytes that sat the exam. False: the
                                    # cycle was scored on the LIVE net the
                                    # learner kept stepping, so the score is
                                    # approximately-these-bytes — by design off
                                    # the self-play path, and by DEGRADATION on
                                    # it (`_freeze_eval_candidate` returned None
                                    # and said so, hours before anyone reads
                                    # this file). Without the flag two
                                    # `.best.pt` files are indistinguishable in
                                    # provenance on freeze morning.
                                    "candidate_frozen": candidate is not None,
                                }
                                if verdict is not None:
                                    meta["worst_reference_win_rate"] = float(
                                        verdict.worst
                                    )
                                    meta["references_evaluated"] = int(
                                        verdict.references
                                    )
                                    meta["reference_episodes"] = int(verdict.episodes)
                                try:
                                    best_checkpoint_hook(
                                        trainer,
                                        eval_grad_step,
                                        meta,
                                        # The weights that EARNED this win rate. The
                                        # hook must write these, not the live net.
                                        outcome.weights,
                                    )
                                except Exception as exc:  # noqa: BLE001
                                    # Swallowed on purpose: one unwritable path must
                                    # not end a 12-hour run. But the high-water mark
                                    # the summary reports is recorded in the `else`
                                    # below, so nothing that failed here can be
                                    # printed at the end as a file to ship.
                                    best_save_failures += 1
                                    _emit(
                                        "[multi] BEST checkpoint save FAILED at "
                                        f"grad_step {eval_grad_step}: {exc}"
                                    )
                                else:
                                    best_saved_win_rate = float(selection_rate)
                                    best_saved_grad_step = eval_grad_step
                                    _emit(
                                        f"[multi] best checkpoint saved: win_rate="
                                        f"{selection_rate:.3f} vs "
                                        f"{selection_opponent or eval_opponent_name} "
                                        f"at grad_step {eval_grad_step}"
                                    )
                            else:
                                # Legacy single-hook callers keep the old save-best
                                # behavior; the periodic/final saves are additional.
                                # This path still writes the LIVE net (the 2-arg
                                # CheckpointHook has nowhere to put a snapshot), so it
                                # keeps the stale-weights exposure the 4-arg
                                # best_checkpoint_hook above fixes. Left as-is
                                # deliberately: the live CLI always passes a
                                # best_checkpoint_hook when a best path is configured
                                # (see _best_checkpoint_path), so tonight's run never
                                # takes this branch.
                                if _save_latest(eval_grad_step, "best"):
                                    best_saved_win_rate = float(selection_rate)
                                    best_saved_grad_step = eval_grad_step
                                elif checkpoint_hook is not None:
                                    # A hook existed and raised — same accounting as
                                    # the 4-arg path above. No hook at all is not a
                                    # failure, just a run with nowhere to save.
                                    best_save_failures += 1
                        _emit(
                            f"[multi grad_step {eval_grad_step}] "
                            f"win_rate={report.win_rate:.3f} "
                            f"mean_len={report.mean_episode_length:.1f} "
                            f"aim_invisible={report.aim_while_invisible:.3f} "
                            f"passed_m2={report.passed_m2} "
                            f"opponent={eval_opponent_name}"
                        )
                        if report.passed_m2:
                            # Two INDEPENDENT concerns, kept independent on purpose.
                            # `passed` is the verdict the CLI turns into an exit code
                            # (_main_multi_arena returns `0 if passed_m2 else 1`);
                            # `stop_on_pass` only decides whether the loop breaks.
                            # Folding the verdict into the break made a scripted run
                            # (which defaults stop_on_pass False, see T13) train all
                            # night, clear the gate, ship a good checkpoint — and
                            # still exit 1.
                            passed = True
                            if stop_on_pass:
                                stop_reason = "passed_m2"
                                break
                except Exception as exc:  # noqa: BLE001 - eval must not end the run
                    # `Exception`, never `BaseException`: a Ctrl-C at 4am and a
                    # SystemExit must still reach the `finally` below, which is
                    # what writes the night's final checkpoint.
                    #
                    # Re-armed HERE as well as on the success path, because that
                    # assignment sits AFTER the eval call: an exception raised
                    # before it would leave next_eval_at in the past and the very
                    # next poll would re-fire the same failing cycle, retrying it
                    # every poll_interval for the rest of the night.
                    next_eval_at = int(trainer.grad_step) + eval_every_grad_steps
                    # Nothing here has to unwind the pause: the collector is
                    # resumed by `_eval_via_designated_arena`'s own `finally` and
                    # train mode restored by `_eval_against_opponent`'s, both of
                    # which have already run by the time this handler is entered.
                    # A skipped cycle that left the designated arena parked would
                    # silently stall 1/N of collection - worse than the crash
                    # this replaces, because nothing would report it.
                    _emit(
                        f"[multi grad_step {grad_step}] eval cycle SKIPPED: it "
                        f"raised {type(exc).__name__}: {exc}. Training continues "
                        "and the next cycle is due at grad_step "
                        f"{next_eval_at}; this cycle selected no checkpoint and "
                        "rated no match, so both series simply have a gap here."
                    )

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
        # --- the pool's LAST index write (T18) ------------------------------
        # `SnapshotPool.record_result` never touches disk — it is the per-episode
        # hot path on every arena thread — so everything scored since the last
        # archive (both Elo series, every head-to-head statistic, the match
        # counters behind selfplay/draw_rate) exists ONLY in memory, while a
        # restart and every post-run analysis read pool.json. Here rather than
        # earlier because `pool.stop()` above has signalled and joined the
        # collectors and the learner thread has been joined too, so this writes
        # the run's settled numbers rather than a mid-flight view; inside the
        # `finally` so a run killed at 4am still flushes what it had. `flush`
        # builds its payload immediately before writing it, so this can only move
        # pool.json forward, and it swallows its own failure so teardown never
        # replaces the learner's exception with its own.
        if snapshot_archivist is not None:
            snapshot_archivist.flush()
        _summarize_selfplay()

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
        # PERSISTED, not selected: these two name the file that exists. The
        # selector's own high-water mark rides along in best_selected_* so a run
        # whose saves all failed is distinguishable from one that never won.
        best_win_rate=best_saved_win_rate,
        best_grad_step=best_saved_grad_step,
        eval_opponent=eval_opponent_name,
        checkpoints_saved=checkpoints_saved,
        best_selected_win_rate=selector.best_win_rate,
        best_selected_grad_step=selector.best_grad_step,
        best_save_failures=best_save_failures,
        selection_opponent=selection_opponent,
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
        help="episodes per greedy eval (default: 100, per AC6). With "
        "--opponent selfplay this sizes the SCRIPTED yardstick track only; the "
        "reference gauntlet is sized by --reference-eval-episodes.",
    )
    parser.add_argument(
        "--reference-eval-episodes",
        type=int,
        default=DEFAULT_REFERENCE_EVAL_EPISODES,
        help=(
            f"episodes against EACH pinned reference in a --opponent selfplay "
            f"eval cycle (default: {DEFAULT_REFERENCE_EVAL_EPISODES}). Ignored "
            "on the dummy/scripted paths, which have no references. A cycle "
            "costs --eval-episodes plus this times the number of pinned "
            "references (1 at seed, up to 3 after promotion), so 10 keeps a "
            "cycle near 30-45 min and 20 pushes it past an hour."
        ),
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
        "--opponent", type=str, default="dummy",
        choices=("dummy", "scripted", "selfplay"),
        help="who the agent fights (default: dummy). 'dummy' is the M1/M2 "
        "stationary dummy served entirely by the bridge - no opponent action ever "
        "goes on the wire. 'scripted' steps the ScriptedBot curriculum in Python "
        "and threads its macro through as opp_action. 'selfplay' steps a FROZEN "
        "past self drawn from the run's snapshot pool and threads its macro "
        "through the same way; it additionally requires --warm-start (the pool's "
        "snapshot 0 is that checkpoint) and builds every env with the "
        "opponent-seat observation mirror. Both non-dummy choices require "
        "--arenas >1 (the single-arena loop steps no opponent policy and refuses "
        "rather than silently fighting the dummy).",
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
    # -- self-play knobs (T11b; all default to None -> the config) ------------
    # Same None-default rule as the curriculum block above: TrainConfig owns
    # every default, so an argparse default can never drift away from the
    # dataclass it configures. Each of these is read by the run ONLY when
    # --opponent selfplay, and each says so, because a knob that silently does
    # nothing on the path the operator is actually running is worse than absent.
    parser.add_argument(
        "--warm-start-sha256", type=str, default=None,
        help="expected SHA-256 of the --warm-start checkpoint, as 64 lowercase hex "
        "characters. The run REFUSES to start when the file on disk hashes to "
        "anything else, naming BOTH digests. Omitted, nothing is verified. A "
        "self-play run seeds snapshot 0 - the pool's first PINNED member, never "
        "dropped and never evicted - entirely from this file, so the wrong "
        "checkpoint becomes a permanent reference opponent; and a stale path or a "
        "half-copied file loads perfectly cleanly into the same architecture.",
    )
    parser.add_argument(
        "--snapshot-every-grad-steps", type=int, default=None,
        help="--opponent selfplay only: learner gradient steps between snapshot-pool "
        "archive events, each of which freezes the PUBLISHED weights as a new past "
        "self the collectors can be drawn against (default: TrainConfig's 1000). "
        "Read by the archive hook (T18), not by this module.",
    )
    parser.add_argument(
        "--snapshot-sampling", type=str, default=None, choices=("uniform", "pfsp"),
        help="--opponent selfplay only: how each episode's opponent snapshot is "
        "drawn (default: TrainConfig's pfsp). 'pfsp' weights the pool toward the "
        "snapshots the learner is roughly even with; 'uniform' draws every live "
        "member with equal probability and is the documented fallback if PFSP has "
        "to be cut.",
    )
    # DELIBERATELY still a RUN-WIDE value, and still not the ε=0 knob a RATED
    # eval needs. That gap is closed elsewhere, not here: `rated=True` on
    # `SnapshotOpponentDriver` (via `build_rated_eval_opponent`) pins BOTH
    # epsilons to a literal 0.0 for an eval driver, which is what makes
    # `MatchResult.rated_eligible` — and so `elo/learner_rated` (AC7) — reachable
    # at all. Passing 0 to THIS flag would instead make the frozen opponent
    # greedy in every TRAINING episode, which is the deterministic lock-in the
    # nonzero default exists to prevent.
    parser.add_argument(
        "--opponent-epsilon", type=float, default=None,
        help="--opponent selfplay only: exploration epsilon applied to the FROZEN "
        "snapshot opponent, never to the learner (default: TrainConfig's 0.02). It "
        "never decays. Do not set it to 0 for a whole run: a greedy learner facing "
        "a greedy snapshot can lock the pair into one deterministic episode "
        "replayed until morning.",
    )
    parser.add_argument(
        "--reference-promote-grad-steps", type=int, nargs=2, default=None,
        metavar=("FIRST", "SECOND"),
        help="--opponent selfplay only: the two grad steps at which pinned reference "
        "snapshots 2 and 3 are promoted (default: TrainConfig's 5000 15000). Must "
        "be strictly increasing. Reference 1 is snapshot 0, pinned at creation from "
        "the warm start, and is not listed here. Read by the promotion hook (T18), "
        "not by this module.",
    )
    parser.add_argument(
        "--elo-k", type=float, default=None,
        help="--opponent selfplay only: Elo K-factor, the largest rating swing one "
        "rated match can produce (default: TrainConfig's 24).",
    )
    parser.add_argument(
        "--elo-initial", type=float, default=None,
        help="--opponent selfplay only: the learner's Elo before any rated match "
        "(default: TrainConfig's 1000). A snapshot's rating is frozen at creation "
        "from the learner's rating at that moment; only the learner's own rating "
        "moves afterwards.",
    )
    parser.add_argument(
        "--stop-on-pass", dest="stop_on_pass", action="store_true", default=None,
        help="stop the run as soon as an eval clears the M2 gate. DEFAULT: on for "
        "--opponent dummy, OFF for every other opponent (the M2 gate is defined "
        "against the STATIONARY dummy, so clearing it says nothing about a run "
        "that fights a scripted bot or a past self, and a warm-started agent can "
        "clear it in its first eval).",
    )
    parser.add_argument(
        "--no-stop-on-pass", dest="stop_on_pass", action="store_false",
        help="run the full budget even if an eval clears the M2 gate.",
    )
    # -- exploration + replay sizing (T16; all default to None -> the config) --
    parser.add_argument(
        "--eps-decay-episodes", type=int, default=None,
        help="GLOBAL episodes over which epsilon decays to its floor. Omitted, it "
        "is DERIVED from --arenas by TrainConfig.eps_decay_episodes_for (~15%% of "
        "the episodes that many pads are projected to collect overnight) - every "
        "pad claims from ONE shared episode counter, so a single-arena number is "
        "spent N times too fast. The projection assumes an UNMEASURED mean episode "
        "length; set this flag from the smoke run's measured mean (do NOT use 600, "
        "which is MAX_EPISODE_STEPS - the truncation cap, not a typical episode).",
    )
    parser.add_argument(
        "--replay-capacity", type=int, default=None,
        help="max stored transitions in the prioritized sequence replay (default: "
        "TrainConfig's 1e6, ~2.3 GB - the per-step LSTM hidden snapshot is 90%% of "
        "that). Lower it on a smaller machine.",
    )
    parser.add_argument(
        "--min-replay", type=int, default=None,
        help="transitions that must be stored before the FIRST gradient step "
        "(default: TrainConfig's 25000).",
    )
    parser.add_argument(
        "--per-actor-eps", dest="per_actor_eps", action="store_true", default=None,
        help="give each arena its own Ape-X epsilon, eps**(1 + i/(N-1)*alpha), so "
        "arena 0 is the MOST exploratory and arena N-1 is near-greedy (issue #15). "
        "ON by default, and a no-op at --arenas 1.",
    )
    parser.add_argument(
        "--no-per-actor-eps", dest="per_actor_eps", action="store_false",
        help="every arena shares one epsilon (the pre-issue-#15 behavior). The "
        "instant off switch if the smoke run looks wrong.",
    )
    parser.add_argument(
        "--per-actor-eps-alpha", type=float, default=None,
        help="Ape-X alpha: the last arena acts at eps**(1+alpha) (default: 7.0, the "
        "paper's value). Must be > 0; use --no-per-actor-eps to disable, not 0.",
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
    ("warm_start_sha256", "warm_start_sha256", str),
    ("snapshot_every_grad_steps", "snapshot_every_grad_steps", int),
    ("snapshot_sampling", "snapshot_sampling", str),
    ("opponent_epsilon", "opponent_epsilon", float),
    # `tuple`, NOT a passthrough. argparse `nargs=2` hands over a LIST, and
    # `TrainConfig.__post_init__` rejects a non-tuple outright: a list field on a
    # frozen dataclass makes `hash(cfg)` raise at whatever unrelated call site
    # hashes the config first, arbitrarily far from this line. The entries are
    # already `int` — the flag declares `type=int`.
    ("reference_promote_grad_steps", "reference_promote_grad_steps", tuple),
    ("elo_k", "elo_k", float),
    ("elo_initial", "elo_initial", float),
    ("eps_decay_episodes", "eps_decay_episodes", int),
    ("replay_capacity", "replay_capacity", int),
    ("min_replay", "min_replay", int),
    ("per_actor_eps", "per_actor_eps", bool),
    ("per_actor_eps_alpha", "per_actor_eps_alpha", float),
)


def _config_from_args(args: Any) -> TrainConfig:
    """Build the run's :class:`TrainConfig` from parsed CLI args.

    Always stamps ``seed`` / ``arenas`` / ``opponent`` (they have real argparse
    defaults); applies every entry of :data:`_CONFIG_OVERRIDE_FLAGS` only when the
    flag was actually passed, so an omitted flag keeps the dataclass default
    rather than re-declaring it in two places.

    ``eps_decay_episodes`` is the ONE field whose omitted-flag fallback is not
    "keep the dataclass default": the default is sized for
    ``DEFAULT_EPS_DECAY_ARENAS`` pad(s) and every arena claims from the SAME
    global episode counter, so at 25 pads that default floors ε about 1% into an
    overnight run. Omitted, it is therefore RE-DERIVED by the dataclass's own
    :func:`~agent.train_config.eps_decay_episodes_for` at this run's real
    ``--arenas`` — same function the default calls, so the two cannot drift.

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
    # Not expressible in the table above (its fallback is the dataclass default,
    # this one's is a formula) — an explicit --eps-decay-episodes has already been
    # applied by the loop and wins.
    if "eps_decay_episodes" not in overrides:
        overrides["eps_decay_episodes"] = eps_decay_episodes_for(int(args.arenas))
    return dataclasses.replace(TrainConfig(), **overrides)


#: Bytes read per hashing iteration in :func:`_verify_warm_start_checksum`. A
#: checkpoint is a few MB today; a fixed-size loop costs nothing now and keeps a
#: future larger net from being read into memory whole just to fingerprint it.
_HASH_CHUNK_BYTES: int = 1 << 20


def _verify_warm_start_checksum(cfg: TrainConfig) -> None:
    """Refuse to start when ``--warm-start``'s bytes disagree with its checksum.

    AC14's second half. ``TrainConfig`` validates only the SHAPE of
    ``warm_start_sha256`` (64 lowercase hex characters) and never opens the
    file; this is the only place the checkpoint's actual bytes are hashed.

    A no-op unless ``cfg.warm_start_sha256`` is set, so every run that does not
    pass ``--warm-start-sha256`` is unaffected.

    Why the gate earns its own failure mode: a self-play run seeds snapshot 0 —
    the pool's first PINNED member, which by
    :class:`~opponents.snapshot_pool.SnapshotPool`'s corruption policy is never
    dropped and never evicted — entirely from this file. A stale path, the wrong
    run's checkpoint or a half-copied file loads perfectly cleanly into the same
    architecture, and the run then spends the night measured against a permanent
    reference opponent nobody can identify afterwards.

    Args:
        cfg: The run config, read for ``warm_start`` and ``warm_start_sha256``.

    Raises:
        ValueError: ``warm_start`` is unset (unreachable through
            ``TrainConfig``, which already refuses that pair), or the file's
            SHA-256 is not ``warm_start_sha256``. The mismatch message carries
            BOTH digests: "checksum mismatch" alone cannot tell an operator
            whether they pasted the wrong checksum or aimed at the wrong file.
        FileNotFoundError: ``warm_start`` names no existing file. Raised here,
            before a logger or a fleet exists, rather than inside ``Trainer``.
    """
    expected = cfg.warm_start_sha256
    if expected is None:
        return
    if cfg.warm_start is None:
        # Unreachable via TrainConfig (it rejects a checksum with no warm start),
        # kept so this function cannot hash the literal string "None" if it is
        # ever called with a config built some other way.
        raise ValueError(
            "warm_start_sha256 is set but warm_start is None: there is no "
            f"checkpoint to hash against {expected}"
        )
    path = str(cfg.warm_start)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"--warm-start {path!r} is not an existing file, so the "
            f"--warm-start-sha256 {expected} cannot be verified against it. "
            "Fix the path (an absolute one survives a change of working "
            "directory) or drop the checksum."
        )
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        raise ValueError(
            "warm-start checksum MISMATCH - refusing to start. "
            f"--warm-start {path!r} hashes to {actual}, but "
            f"--warm-start-sha256 says {expected}. Either the checkpoint is not "
            "the one this run was sized against, or the checksum was copied "
            "from a different file; do not start until you know which."
        )


def _resolve_stop_on_pass(explicit: Optional[bool], cfg: TrainConfig) -> bool:
    """Resolve ``--stop-on-pass`` / ``--no-stop-on-pass``, defaulting by opponent.

    An explicit flag always wins. With neither, the default is ``True`` for the
    dummy (today's M2 behavior, unchanged) and ``False`` for every other
    opponent — the scripted curriculum and self-play alike.

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


def _atomic_torch_save(payload: Mapping[str, Any], path: str) -> None:
    """``torch.save`` that can never truncate the file it is replacing.

    Saving straight onto the destination writes the deliverable IN PLACE: a
    crash, a Ctrl-C, an OOM kill, or a full disk partway through leaves a
    half-written checkpoint where the previous good one used to be, and there is
    no prior generation to fall back on. ``runs/m3.best.pt`` is the single file
    the demo depends on, so that window has to close.

    So: serialize into a fresh temp file in the SAME DIRECTORY, fsync it, then
    ``os.replace`` it onto ``path``. Same directory is load-bearing — ``os.replace``
    is atomic only within one filesystem, and a temp in ``/tmp`` would make the
    rename a cross-device copy, i.e. exactly the in-place truncation this avoids.
    Until the replace lands, the destination is byte-for-byte the previous save;
    after it, it is wholly the new one. Readers never see a partial file.

    Failure handling: any exception unlinks the temp and propagates unchanged, so
    callers keep whatever error handling they already had (both call sites are
    fired through hooks whose failures the run loop logs and swallows). Temp
    names come from :func:`tempfile.mkstemp`, so they are unique per attempt —
    a stale temp left behind by a hard kill cannot collide with, block, or be
    mistaken for a later run's write; it only costs disk.

    Directories are NOT created: a missing parent must fail exactly as the plain
    ``torch.save`` it replaces did, rather than quietly inventing a run
    directory.

    Args:
        payload: The object to serialize (the checkpoint dict).
        path: Destination path, replaced atomically on success.
    """
    import os
    import tempfile

    # dirname(abspath(...)) so a bare filename ("best.pt") still resolves to a
    # real directory — dirname() alone returns "" there and mkstemp would reject it.
    directory = os.path.dirname(os.path.abspath(path))
    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.", suffix=".tmp", dir=directory
    )

    def _discard_temp() -> None:
        try:
            os.unlink(tmp_path)
        except OSError:
            # Best effort: the temp is already gone, or the directory is
            # unwritable — neither is worth masking the original failure.
            pass

    try:
        handle = os.fdopen(fd, "wb")
    except BaseException:
        # fdopen did not take ownership of the descriptor, so close it here or
        # it leaks for the life of the process.
        os.close(fd)
        _discard_temp()
        raise

    try:
        with handle:
            torch.save(payload, handle)
            handle.flush()
            # Durability: without the fsync the bytes may still be in the page
            # cache when the rename commits, so a power loss could publish an
            # empty file over a good one.
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        # BaseException, not Exception: a KeyboardInterrupt at 4am must not be
        # the one path that leaves a temp behind.
        _discard_temp()
        raise


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

    # The rest of the config validation, continued from TrainConfig into the one
    # check that has to touch the disk. FIRST, so no later early return can skip
    # it, and before the logger exists so a refused run leaves no run directory.
    _verify_warm_start_checksum(cfg)
    if cfg.warm_start_sha256 is not None:
        print(
            f"[train] warm start {cfg.warm_start} verified: "
            f"sha256={cfg.warm_start_sha256}",
            file=sys.stderr,
        )

    # Refuse the one combination that cannot be honored, before anything starts:
    # the single-arena loop steps no opponent policy, so --opponent scripted or
    # selfplay there would quietly train against the stationary dummy instead.
    if int(args.arenas) < 2 and cfg.opponent != "dummy":
        print(
            f"[train] FATAL: --opponent {cfg.opponent} needs --arenas >1. The "
            "single-arena loop never steps an opponent policy - neither the "
            "scripted curriculum nor a self-play snapshot (the dummy is served "
            "by the bridge) - so it would silently train against the stationary "
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
            # The ε window is DERIVED from --arenas when the flag is omitted, so
            # the number alone does not say whether the run explored for long
            # enough. Record where the floor lands and how the fleet was spread.
            "eps_floor_fraction_of_run": eps_floor_fraction_of_run(cfg),
            "per_actor_eps": per_actor_eps_enabled(cfg),
            "per_actor_eps_alpha": cfg.per_actor_eps_alpha,
            "replay_capacity": cfg.replay_capacity,
            "min_replay": cfg.min_replay,
            # WHO THIS RUN FOUGHT. `opponent` above proves only that the flag
            # parsed; these decide which past self was drawn, how often a new
            # one was archived, and what its rating meant. Recorded on every run
            # (they are constants either way) — the loop reads them only when
            # opponent == "selfplay".
            "warm_start_sha256": cfg.warm_start_sha256,
            "snapshot_every_grad_steps": cfg.snapshot_every_grad_steps,
            "snapshot_sampling": cfg.snapshot_sampling,
            "opponent_epsilon": cfg.opponent_epsilon,
            # `list`, not the tuple: `json.dump` writes a tuple as a JSON array
            # anyway, so converting here keeps summary.json and a W&B config the
            # same shape rather than backend-dependent.
            "reference_promote_grad_steps": list(cfg.reference_promote_grad_steps),
            "elo_k": cfg.elo_k,
            "elo_initial": cfg.elo_initial,
            # Where this run's past selves live — through the SAME helper
            # `_main_multi_arena` hands the loop, so the record and the pool
            # cannot name two directories. None off the self-play path, which
            # builds no pool at all.
            "snapshot_pool_dir": (
                snapshot_pool_directory(args.run_name)
                if cfg.opponent == "selfplay"
                else None
            ),
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
        # Atomic: a kill mid-write must not truncate the previous checkpoint,
        # which is the fallback when no best net was ever selected.
        _atomic_torch_save(
            {
                "model": trainer.online.state_dict(),
                "grad_step": grad_step,
                "code_version": code_version(),
            },
            checkpoint_path,
        )

    def _save_best_checkpoint(
        trainer: "Trainer",
        grad_step: int,
        meta: Mapping[str, Any],
        weights: Mapping[str, Any],
    ) -> None:
        """Write the SAVE-BEST net, stamped with the eval that justified it.

        ``weights`` — NOT ``trainer.online.state_dict()``. The learner thread runs
        throughout the eval and this save, so the live net here is a different,
        later network than the one the win rate in ``meta`` describes. Writing it
        would mean shipping a checkpoint selected by a score it did not earn.

        Written atomically: this is THE deliverable, it is rewritten every time a
        better eval lands, and a kill partway through would destroy the previous
        (good) generation in place with nothing to fall back on.
        """
        if best_checkpoint_path is None:
            return
        _atomic_torch_save(
            {
                "model": weights,
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

    Constructs, via :func:`build_live_env_factory_for`, an env factory that per
    pad ``i`` opens a :class:`~env.mc_pvp_env.TcpBridgeClient` to bridge port
    ``--port + i`` and wraps it in an :class:`~env.mc_pvp_env.MCPvPEnv` (with
    the self-play observation mirror when ``--opponent selfplay``), then runs
    :func:`train_multi_arena`. The N pads must ALREADY be booted AND PRIMED by the human
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
      * A self-play run's ``snapshot_dir`` is resolved HERE, from ``--run-name``
        via :func:`snapshot_pool_directory`, and handed to
        :func:`train_multi_arena`. Left unset, the loop's own fallback names
        ``runs/selfplay/snapshots`` for every run alike, and a second run would
        silently inherit the first's past selves and its Elo series.
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

    # The per-pad env factory, including the self-play observation mirror. Built
    # by a module-level helper so the mirror wiring is testable without a fleet.
    _env_factory_for = build_live_env_factory_for(
        cfg,
        host=host,
        base_port=base_port,
        max_episode_steps=MAX_EPISODE_STEPS,
    )

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

    # WHERE THIS RUN'S PAST SELVES LIVE. Derived from --run-name here rather than
    # left to train_multi_arena's `snapshot_dir=None` fallback, which resolves to
    # runs/selfplay/snapshots for EVERY run regardless of its name: a second
    # self-play run would then reload the first one's pool, resample its
    # snapshots and continue its Elo series as though the two learners were one.
    # `None` off the self-play path, where the loop never reads it — and
    # `args.run_name` is read only there, deliberately without a getattr
    # fallback, because a default would be a shared pool by another name.
    snapshot_dir = (
        snapshot_pool_directory(args.run_name) if cfg.opponent == "selfplay" else None
    )

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
        # getattr, matching every other optional flag on this call: a caller
        # that built `args` by hand (the M2 smoke harness does) must not have to
        # know about a flag only the self-play path reads.
        reference_eval_episodes=getattr(
            args, "reference_eval_episodes", DEFAULT_REFERENCE_EVAL_EPISODES
        ),
        stop_on_pass=_resolve_stop_on_pass(getattr(args, "stop_on_pass", None), cfg),
        checkpoint_every_grad_steps=getattr(
            args, "checkpoint_every_grad_steps", None
        ),
        snapshot_dir=snapshot_dir,
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
    # checkpoint from this line, not from file mtimes. It therefore reports what was
    # PERSISTED, never what was merely selected: a save that raised is swallowed so
    # it cannot end the night, and a line naming a file that was never written is
    # worse than no line at all. Every branch below keeps the "best checkpoint:"
    # prefix so grepping for it on freeze day finds the bad news too.
    #
    # WHO the number was earned against comes from `selection_opponent` when the
    # run set one — a self-play run selects on the reference-gauntlet aggregate
    # while `eval_opponent` still names the scripted yardstick, and printing the
    # aggregate under the scripted label would put a false number on the one
    # line freeze day reads.
    selection_basis = result.selection_opponent or result.eval_opponent
    if result.best_grad_step >= 0:
        _log(
            f"  best checkpoint: win_rate={result.best_win_rate:.3f} vs "
            f"{selection_basis} at grad_step {result.best_grad_step}"
        )
        if (
            result.best_save_failures
            and result.best_selected_grad_step != result.best_grad_step
        ):
            # The file above is the best that EXISTS, not the best this run
            # reached. Gated on DIVERGENCE, not on the failure count: a save can
            # fail and a LATER one succeed, and claiming the peak is missing when
            # it is on disk is the same false-summary defect, inverted.
            _log(
                f"  best checkpoint: WARNING - {result.best_save_failures} "
                "best-checkpoint save(s) FAILED; the run's best eval "
                f"(win_rate={result.best_selected_win_rate:.3f} at grad_step "
                f"{result.best_selected_grad_step}) is NOT on disk"
            )
    elif result.best_selected_grad_step >= 0 and result.best_save_failures:
        _log(
            f"  best checkpoint: NONE WRITTEN - all {result.best_save_failures} "
            "save(s) FAILED. The run selected "
            f"win_rate={result.best_selected_win_rate:.3f} vs "
            f"{selection_basis} at grad_step "
            f"{result.best_selected_grad_step}, but NO best checkpoint file "
            "exists - do not ship from this line; see the '[multi] BEST "
            "checkpoint save FAILED' lines above"
        )
    elif result.best_selected_grad_step >= 0:
        # Selected something, never even tried to save it: this run was given no
        # checkpoint path at all. Saying "no best checkpoint" without saying why
        # would read as "the agent never won".
        _log(
            "  best checkpoint: NONE WRITTEN - no checkpoint path was configured, "
            "so nothing was saved. The run selected "
            f"win_rate={result.best_selected_win_rate:.3f} vs "
            f"{selection_basis} at grad_step "
            f"{result.best_selected_grad_step}"
        )
    elif result.reports:
        _log(
            f"  no best checkpoint: no eval vs {selection_basis} won a single "
            "episode, so the latest periodic/final checkpoint is all there is"
        )
    return 0 if result.passed_m2 else 1


if __name__ == "__main__":
    raise SystemExit(main())
