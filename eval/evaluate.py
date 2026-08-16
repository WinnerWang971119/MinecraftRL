"""evaluate — M2 eval harness: greedy agent win-rate + per-reward-component logging.

Runs a GREEDY policy (ε=0) for N eval episodes against the stage opponent (the
stationary dummy at stage 0) and measures the M2 gate (AC6):

  * **win rate** — the gate proper; AC6 requires >= 95% over 100 episodes.
  * **mean / median episode length** — with two anti-reward-hacking guards:
      (a) aim-bonus accrued while ``visible == false`` must be EXACTLY 0 over the
          whole run (the spin-farming guard — the agent must not be paid for
          "aiming" at an opponent it cannot see), and
      (b) mean episode length must be BELOW the timeout cap (the run-away guard —
          a policy that stalls to the horizon is not winning).
  * **per-reward-component breakdown** — every reward term
    (``r_damage_dealt`` / ``r_damage_taken`` / ``r_step`` / ``r_aim`` /
    ``r_shaping`` / ``r_terminal``) is accumulated AND logged SEPARATELY, per the
    plan and training-spec §9.2: a single scalar reward hides reward hacking, so
    the component decomposition is the fastest way to catch it.

------------------------------------------------------------------------------
The injectable env + policy seam (offline proof)
------------------------------------------------------------------------------
:func:`evaluate` takes an injected ``env`` and ``policy`` so the harness LOGIC is
provable with no socket and no live server:

  * ``env`` is any object with the Gym-style ``reset(seed) -> obs`` /
    ``step(action) -> (obs, reward, done, info)`` surface — the real
    :class:`~env.mc_pvp_env.MCPvPEnv` over a live :class:`~env.mc_pvp_env.TcpBridgeClient`
    in production, a FAKE scripted bridge offline (``tests/test_evaluate.py``
    reuses the ``ScriptedBridge`` pattern from ``tests/test_mc_pvp_env.py``).
  * ``policy`` is any :class:`GreedyPolicy` — ``reset()`` at each episode start
    then ``act(obs) -> int`` per step. The production policy is
    :class:`DRQNGreedyPolicy`, a thin wrapper over
    :class:`~agent.dqn.DuelingDRQN` that drives ``act(obs, hidden, epsilon=0.0)``
    (the pure-greedy, no-RNG path) and carries the LSTM hidden state across the
    episode. Tests inject a tiny scripted policy with no torch dependency.

  * ``opponent`` (optional) is any :class:`EvalOpponent` — ``begin_episode()`` per
    episode then ``act(view) -> int`` per step. ``None`` (the default) is the
    stage-0 stationary dummy, served entirely by the bridge, and keeps the wire
    line byte-identical to the M2 path. Supplying one makes the eval fight the
    SAME moving opponent training fights, which is what a scripted-opponent run's
    win rate has to mean before a checkpoint can be selected on it.

The eval module imports ``torch`` LAZILY (only inside :class:`DRQNGreedyPolicy`
and the CLI), so importing ``eval.evaluate`` — and running its offline tests —
never requires torch.

------------------------------------------------------------------------------
The REAL AC6 / TC13 M2 eval is a T20 live run
------------------------------------------------------------------------------
This task (T19) delivers the harness, the offline fake-bridge proof, and the
per-component logging seam. It does NOT, and cannot offline, produce the actual
AC6 number: the greedy trained DRQN evaluated vs the LIVE stationary dummy over
100 episodes hitting >= 95% win rate. That needs a trained checkpoint, the live
Paper server, and the Node bridge, and runs as part of T20 via
``python -m eval.evaluate --checkpoint <ckpt>`` against a started bridge. The
printed :class:`EvalReport` (with ``passed_m2``) is the AC6 evidence artifact.

Owner: T19 (Eval/infra track)
"""

from __future__ import annotations

import argparse
import statistics
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

import numpy as np

from agent.contract_config import MAX_EPISODE_STEPS
from env.mc_pvp_env import REWARD_COMPONENT_KEYS
from env.observation_spec import Obs
from eval.logging import MetricsLogger

__all__ = [
    "REWARD_COMPONENT_KEYS",
    "M2_WIN_RATE_THRESHOLD",
    "DUMMY_OPPONENT_NAME",
    "GreedyPolicy",
    "EvalOpponent",
    "DRQNGreedyPolicy",
    "EpisodeOutcome",
    "EvalReport",
    "evaluate",
    "main",
]


# ---------------------------------------------------------------------------
# The M2 gate threshold (AC6): >= 95% win rate over 100 eval episodes.
# ---------------------------------------------------------------------------
M2_WIN_RATE_THRESHOLD: float = 0.95

#: What :attr:`EvalReport.opponent` says when no opponent policy was stepped —
#: i.e. the bridge-served stationary dummy, the M1/M2 stage-0 opponent.
DUMMY_OPPONENT_NAME: str = "dummy"


# ---------------------------------------------------------------------------
# Policy seam.
#
# The evaluator needs only two things of a policy: reset its per-episode state at
# the start of each episode and choose ONE action per step. Keeping this a tiny
# Protocol lets a test inject a trivial scripted policy (no torch) while the real
# greedy DRQN is wrapped by DRQNGreedyPolicy below.
# ---------------------------------------------------------------------------


class GreedyPolicy(Protocol):
    """Structural protocol for a deterministic eval policy.

    The evaluator calls :meth:`reset` once at the start of every episode (so a
    recurrent policy clears its hidden state and never carries memory across an
    episode boundary) and :meth:`act` once per decision step.
    """

    def reset(self) -> None:
        """Clear any per-episode internal state (e.g. an LSTM hidden state)."""
        ...

    def act(self, obs: np.ndarray) -> int:
        """Return the greedy action index in ``[0, N_ACTIONS)`` for ``obs``."""
        ...


class EvalOpponent(Protocol):
    """Structural protocol for an opponent policy the EVAL steps in Python.

    Mirrors ``agent.train.EpisodeOpponent`` minus ``observe_outcome``: an eval
    scores the AGENT, so nothing is fed back into a curriculum from here.

    Passing one of these to :func:`evaluate` is what makes the eval score the
    same MOVING opponent training fights. Without it the eval sends no
    ``opp_action`` and the opponent stands still for the whole episode — which
    is not "the dummy path" when the run's opponent is a scripted bot, it is a
    silently different, far easier opponent, and any checkpoint selected on that
    win rate was selected against the wrong thing.
    """

    def begin_episode(self) -> None:
        """Called once per episode, AFTER ``env.reset`` and before the first step."""
        ...

    def act(self, view: Any) -> int:
        """Return the opponent's macro (``0..N_ACTIONS-1``) for this window.

        ``view`` is whatever ``env.raw_opponent_view()`` returned, passed
        through UNTOUCHED (its ``attack_cooldown`` is clamped to exactly 1.0 by
        its producer and compared against a deliberately tight
        ``>= 1.0 - 1e-6``; perturbing it makes a scripted bot never attack).
        """
        ...


class DRQNGreedyPolicy:
    """Greedy (ε=0) adapter over :class:`~agent.dqn.DuelingDRQN` for eval.

    Wraps the recurrent Q-network so the evaluator can treat it as a plain
    :class:`GreedyPolicy`: :meth:`reset` zeroes the LSTM hidden state at each
    episode boundary, and :meth:`act` advances the net by one step under
    ``epsilon=0.0`` — the pure-greedy, NO-RNG path of ``DuelingDRQN.act`` (so the
    eval is fully deterministic and leaves the global torch RNG untouched).

    ``torch`` is imported lazily here, not at module load, so importing
    :mod:`eval.evaluate` never requires torch (the offline tests inject their own
    torch-free policy).

    Args:
        net: A trained :class:`~agent.dqn.DuelingDRQN`. It is switched to ``eval``
            mode on construction (inference, no dropout/BN drift).
        device: Optional torch device for the per-step observation tensor;
            defaults to the net's parameter device so the tensor lands on the same
            device as the weights.
    """

    def __init__(self, net: Any, device: Optional[Any] = None) -> None:
        import torch  # lazy: keep eval.evaluate importable without torch

        self._torch = torch
        # BY REFERENCE, and knowingly so. On the multi-arena path the learner
        # thread keeps stepping the optimizer throughout an eval (only the
        # designated arena's COLLECTOR is paused), so a long eval scores a moving
        # target: episode 1 and episode 100 can run on different weights. The fix
        # for that is to evaluate a frozen snapshot net, which is a bigger change
        # than the freeze-day scope allowed. What IS handled:
        # ``agent.train._eval_against_opponent`` clones the weights this eval
        # starts from and hands them to the save-best path, so the shipped
        # checkpoint is at least the net the eval began on rather than one
        # thousands of gradient steps later. This comment marks a deliberate
        # boundary, not an oversight.
        self._net = net
        self._net.eval()  # inference mode for greedy action selection
        if device is None:
            device = next(net.parameters()).device
        self._device = device
        self._hidden: Optional[Tuple[Any, Any]] = None

    def reset(self) -> None:
        """Clear the LSTM hidden state so the next episode starts from zeros."""
        # None lets DuelingDRQN.act zero-init on the first step of the episode.
        self._hidden = None

    def act(self, obs: np.ndarray) -> int:
        """Return the greedy macro index for ``obs`` and advance the LSTM state."""
        torch = self._torch
        obs_tensor = torch.as_tensor(
            np.asarray(obs, dtype=np.float32), dtype=torch.float32, device=self._device
        )
        # epsilon=0.0 is the pure-greedy short-circuit: argmax with no RNG draw.
        action, self._hidden = self._net.act(obs_tensor, self._hidden, epsilon=0.0)
        return int(action)


# ---------------------------------------------------------------------------
# Per-episode + run-level result types.
# ---------------------------------------------------------------------------


@dataclass
class EpisodeOutcome:
    """Outcome of one finished eval episode.

    Attributes:
        index: 0-based episode index within the eval run.
        result: One of ``"win"`` / ``"loss"`` / ``"timeout"`` (mutually exclusive,
            mirroring :meth:`env.mc_pvp_env.MCPvPEnv.step`).
        length: Number of decision steps taken before termination.
        total_reward: Sum of scalar step rewards over the episode.
        components: Per-component reward SUMS over the episode (keys are
            :data:`REWARD_COMPONENT_KEYS`).
        aim_while_invisible: Aim-bonus (``r_aim``) accrued on steps where the obs
            ``visible`` flag was false. Must be 0 for a healthy policy.
    """

    index: int
    result: str
    length: int
    total_reward: float
    components: Dict[str, float] = field(default_factory=dict)
    aim_while_invisible: float = 0.0


@dataclass
class EvalReport:
    """Structured result of a greedy eval run — the M2 gate (AC6) evidence.

    Attributes:
        n_episodes: Number of episodes evaluated.
        n_wins / n_losses / n_timeouts: Outcome counts (sum to ``n_episodes``).
        win_rate: ``n_wins / n_episodes`` (0.0 when no episodes ran). THE gate.
        mean_episode_length / median_episode_length: Episode-length statistics over
            the run (0.0 when no episodes ran).
        timeout_cap: The episode-length horizon (decision steps) a timeout hits.
            ``mean_episode_length`` must be strictly below this (run-away guard).
        reward_component_sums: Per-component reward summed over ALL episodes (keys
            are :data:`REWARD_COMPONENT_KEYS`).
        reward_component_means: Per-component reward averaged PER EPISODE.
        mean_total_reward: Mean scalar episode reward over the run.
        aim_while_invisible: Total aim-bonus accrued while ``visible == false``
            across the whole run. MUST be exactly 0 (the spin-farming guard).
        passed_m2: ``True`` iff ``win_rate >= 0.95`` AND ``aim_while_invisible ==
            0`` AND ``mean_episode_length < timeout_cap``.
        is_live: ``True`` for a real-bridge run, ``False`` for the offline
            fake-bridge proof (so the artifact is never mistaken for the live AC6
            number).
        opponent: WHO was fought — :data:`DUMMY_OPPONENT_NAME` when no opponent
            policy was stepped, else the name of the stepped opponent (e.g.
            ``"scripted_mixed"``). Recorded because ``win_rate`` is meaningless
            without it: the same net scores very differently against a stationary
            target and a moving one, and a checkpoint selected on the wrong one is
            selected on nothing.
        episodes: Per-episode outcomes, in order.
        notes: Free-form human-readable notes (incl. the AC6/TC13 follow-up).
    """

    n_episodes: int = 0
    n_wins: int = 0
    n_losses: int = 0
    n_timeouts: int = 0
    win_rate: float = 0.0
    mean_episode_length: float = 0.0
    median_episode_length: float = 0.0
    timeout_cap: int = MAX_EPISODE_STEPS
    reward_component_sums: Dict[str, float] = field(default_factory=dict)
    reward_component_means: Dict[str, float] = field(default_factory=dict)
    mean_total_reward: float = 0.0
    aim_while_invisible: float = 0.0
    passed_m2: bool = False
    is_live: bool = False
    opponent: str = DUMMY_OPPONENT_NAME
    episodes: List[EpisodeOutcome] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Render the report as a plain JSON-serializable dict (no per-ep detail)."""
        return {
            "n_episodes": self.n_episodes,
            "n_wins": self.n_wins,
            "n_losses": self.n_losses,
            "n_timeouts": self.n_timeouts,
            "win_rate": self.win_rate,
            "mean_episode_length": self.mean_episode_length,
            "median_episode_length": self.median_episode_length,
            "timeout_cap": self.timeout_cap,
            "reward_component_sums": dict(self.reward_component_sums),
            "reward_component_means": dict(self.reward_component_means),
            "mean_total_reward": self.mean_total_reward,
            "aim_while_invisible": self.aim_while_invisible,
            "passed_m2": self.passed_m2,
            "is_live": self.is_live,
            "opponent": self.opponent,
            "notes": list(self.notes),
        }


# ---------------------------------------------------------------------------
# Outcome classification (mirrors run_random._classify / MCPvPEnv.step).
# ---------------------------------------------------------------------------


def _classify(info: Mapping[str, Any]) -> str:
    """Map a terminal ``info`` dict to ``"win"`` / ``"loss"`` / ``"timeout"``.

    Mirrors the mutually-exclusive outcome flags set by
    :meth:`env.mc_pvp_env.MCPvPEnv.step`: a loss takes precedence (a same-step
    double-death is a loss), then a win, else timeout.
    """
    if info.get("lost"):
        return "loss"
    if info.get("won"):
        return "win"
    return "timeout"


def _is_visible(obs: np.ndarray) -> bool:
    """True iff the obs ``visible`` flag is set (the gated opponent is seen).

    Uses the same > 0.5 threshold the env/reward use so "visible" here means
    exactly what it means when the aim bonus is granted.
    """
    return bool(obs[Obs.VISIBLE] > 0.5)


# ---------------------------------------------------------------------------
# The evaluator.
# ---------------------------------------------------------------------------


def evaluate(
    env: Any,
    policy: GreedyPolicy,
    n_episodes: int = 100,
    logger: Optional[MetricsLogger] = None,
    *,
    timeout_cap: int = MAX_EPISODE_STEPS,
    base_seed: int = 0,
    is_live: bool = False,
    max_episode_steps: Optional[int] = None,
    log: Optional[Any] = None,
    opponent: Optional[EvalOpponent] = None,
    opponent_name: Optional[str] = None,
) -> EvalReport:
    """Run the GREEDY policy for ``n_episodes`` vs the stage opponent (M2 gate).

    Each episode: ``policy.reset()`` -> ``env.reset(seed)`` -> loop
    ``policy.act`` / ``env.step`` until ``done``. The policy runs greedy (ε=0)
    throughout — :class:`DRQNGreedyPolicy` drives ``DuelingDRQN.act`` with
    ``epsilon=0.0``, the deterministic no-RNG path.

    WHO THE AGENT FIGHTS IS AN ARGUMENT, NOT AN ASSUMPTION. With ``opponent ==
    None`` (the default) the opponent is the stage-0 stationary dummy: the
    env/bridge drive it, this loop steps no opponent policy, and the wire line is
    the byte-identical M2 one. Pass an :class:`EvalOpponent` and each decision
    additionally reads ``env.raw_opponent_view()``, asks that policy for a macro,
    and sends BOTH in one ``env.step(action, opp_action=...)`` — which is the
    only way a run whose training opponent MOVES gets a win rate that means
    anything. Scoring a scripted-opponent run against a stationary target reports
    a number the agent did not earn, and selecting a checkpoint on it selects
    against the wrong opponent.

    Per episode the per-reward-component breakdown carried in each ``step``
    ``info`` (keys :data:`REWARD_COMPONENT_KEYS`) is accumulated, the aim-bonus
    accrued while ``visible == false`` is summed (the spin-farming guard), and —
    when a ``logger`` is given — every component is logged SEPARATELY (per the
    plan + training-spec §9.2). Run-level summaries (win rate, mean/median length,
    component sums/means, the guards, and ``passed_m2``) are computed and, with a
    logger, written via :meth:`MetricsLogger.summary`.

    Args:
        env: A Gym-style env (``reset(seed) -> obs`` / ``step(a) -> (obs, r, done,
            info)``). The real :class:`~env.mc_pvp_env.MCPvPEnv` for the live run, a
            fake scripted bridge offline. NOT closed here — the caller owns its
            lifecycle (so a single env can be reused across episodes).
        policy: The greedy policy to evaluate (a :class:`GreedyPolicy`).
        n_episodes: Number of eval episodes (>= 1). AC6 uses 100.
        logger: Optional :class:`~eval.logging.MetricsLogger`. When given, each
            reward component is logged per episode and the run summary is recorded;
            when ``None`` the breakdown is still computed into the report.
        timeout_cap: Episode-length horizon (decision steps) a timeout reaches;
            ``mean_episode_length`` must be strictly below it (run-away guard).
            Defaults to the frozen :data:`MAX_EPISODE_STEPS`.
        base_seed: Per-episode env reset seed is ``base_seed + episode_index`` so
            the eval run is reproducible.
        is_live: Marks the report as a live (vs offline) run.
        max_episode_steps: Optional hard per-episode decision cap defending against
            a fake env that never terminates. ``None`` relies on the env's own
            ``done``. (The real env enforces its own ``max_episode_steps``.)
        log: Optional ``str -> None`` progress sink (``None`` silences it).
        opponent: Optional :class:`EvalOpponent` stepped once per decision (see
            above). ``None`` == the stationary-dummy path, unchanged. When given,
            ``env`` must also expose ``raw_opponent_view()`` and accept
            ``opp_action`` (the real :class:`~env.mc_pvp_env.MCPvPEnv` does).
        opponent_name: Name recorded on the report's ``opponent`` field. Defaults
            to :data:`DUMMY_OPPONENT_NAME` when no ``opponent`` is given, and to
            the opponent's own ``name`` attribute (else ``"opponent"``) when one
            is.

    Returns:
        A populated :class:`EvalReport`.

    Raises:
        ValueError: if ``n_episodes`` < 1 or ``timeout_cap`` < 1.
    """
    if n_episodes < 1:
        raise ValueError(f"n_episodes must be >= 1, got {n_episodes}")
    if timeout_cap < 1:
        raise ValueError(f"timeout_cap must be >= 1, got {timeout_cap}")

    def _emit(message: str) -> None:
        if log is not None:
            log(message)

    if opponent_name is None:
        opponent_name = (
            DUMMY_OPPONENT_NAME
            if opponent is None
            else str(getattr(opponent, "name", "opponent"))
        )

    n_wins = 0
    n_losses = 0
    n_timeouts = 0
    lengths: List[int] = []
    total_rewards: List[float] = []
    # Run-level per-component sums, seeded at 0.0 for every key so a component that
    # never fires still appears in the report (never a missing key).
    component_sums: Dict[str, float] = {key: 0.0 for key in REWARD_COMPONENT_KEYS}
    aim_while_invisible_total = 0.0
    episodes: List[EpisodeOutcome] = []

    for ep in range(n_episodes):
        outcome = _run_one_episode(
            env,
            policy,
            episode_index=ep,
            seed=base_seed + ep,
            max_episode_steps=max_episode_steps,
            opponent=opponent,
        )
        episodes.append(outcome)

        if outcome.result == "win":
            n_wins += 1
        elif outcome.result == "loss":
            n_losses += 1
        else:
            n_timeouts += 1

        lengths.append(outcome.length)
        total_rewards.append(outcome.total_reward)
        for key in REWARD_COMPONENT_KEYS:
            component_sums[key] += outcome.components.get(key, 0.0)
        aim_while_invisible_total += outcome.aim_while_invisible

        # Log EACH reward component separately for THIS episode (the fastest way to
        # catch reward hacking — a single scalar would hide it). The win flag and
        # length go alongside so the per-episode series is self-contained.
        if logger is not None:
            metrics: Dict[str, Any] = {
                "episode_length": outcome.length,
                "episode_reward": outcome.total_reward,
                "win": 1.0 if outcome.result == "win" else 0.0,
                "aim_while_invisible": outcome.aim_while_invisible,
            }
            for key in REWARD_COMPONENT_KEYS:
                metrics[key] = outcome.components.get(key, 0.0)
            logger.log(metrics, step=ep)

        _emit(
            f"[eval {ep:>4}] {outcome.result:<7} len={outcome.length:>4} "
            f"R={outcome.total_reward:+.2f}"
        )

    # --- assemble the run-level report -------------------------------------
    win_rate = n_wins / n_episodes
    mean_len = statistics.fmean(lengths) if lengths else 0.0
    median_len = float(statistics.median(lengths)) if lengths else 0.0
    component_means = {
        key: component_sums[key] / n_episodes for key in REWARD_COMPONENT_KEYS
    }
    mean_total_reward = statistics.fmean(total_rewards) if total_rewards else 0.0

    # The M2 gate (AC6): the win-rate threshold AND both anti-reward-hacking guards.
    passed_m2 = (
        win_rate >= M2_WIN_RATE_THRESHOLD
        and aim_while_invisible_total == 0.0
        and mean_len < float(timeout_cap)
    )

    report = EvalReport(
        n_episodes=n_episodes,
        n_wins=n_wins,
        n_losses=n_losses,
        n_timeouts=n_timeouts,
        win_rate=win_rate,
        mean_episode_length=mean_len,
        median_episode_length=median_len,
        timeout_cap=int(timeout_cap),
        reward_component_sums=dict(component_sums),
        reward_component_means=component_means,
        mean_total_reward=mean_total_reward,
        aim_while_invisible=aim_while_invisible_total,
        passed_m2=passed_m2,
        is_live=bool(is_live),
        opponent=str(opponent_name),
        episodes=episodes,
    )

    report.notes.append(
        "AC6/TC13 follow-up: the REAL M2 eval (greedy DRQN vs the LIVE stationary "
        "dummy, 100 eps, >=95% win rate) runs as part of T20 against the live "
        "Paper server + Node bridge; this report is "
        + ("a LIVE" if is_live else "an OFFLINE")
        + " measurement of the eval harness / component-logging logic."
    )
    if opponent is not None:
        # passed_m2 is the M2 gate, and the M2 gate is defined against the
        # STATIONARY DUMMY (AC6). Scored against a moving opponent the same
        # arithmetic is a useful selection signal but not that milestone, and the
        # artifact must not be read as if it were.
        report.notes.append(
            f"Scored against the stepped opponent {report.opponent!r}, NOT the "
            "stationary dummy: win_rate is a scripted-opponent win rate (the "
            "checkpoint-selection signal) and passed_m2 is therefore NOT the M2 "
            "gate, which AC6 defines against the dummy."
        )

    # Run summary: the headline gate numbers + the full per-component breakdown.
    if logger is not None:
        summary: Dict[str, Any] = {
            "n_episodes": report.n_episodes,
            "n_wins": report.n_wins,
            "n_losses": report.n_losses,
            "n_timeouts": report.n_timeouts,
            "win_rate": report.win_rate,
            "mean_episode_length": report.mean_episode_length,
            "median_episode_length": report.median_episode_length,
            "timeout_cap": report.timeout_cap,
            "mean_total_reward": report.mean_total_reward,
            "aim_while_invisible": report.aim_while_invisible,
            "passed_m2": report.passed_m2,
            "is_live": report.is_live,
            "opponent": report.opponent,
        }
        # Surface each component sum AND mean under a stable, namespaced key so the
        # breakdown is queryable in the run summary, not just the per-episode series.
        for key in REWARD_COMPONENT_KEYS:
            summary[f"sum.{key}"] = report.reward_component_sums[key]
            summary[f"mean.{key}"] = report.reward_component_means[key]
        logger.summary(summary)

    _emit(
        f"[eval done] episodes={report.n_episodes} wins={report.n_wins} "
        f"losses={report.n_losses} timeouts={report.n_timeouts} "
        f"win_rate={report.win_rate:.3f} mean_len={report.mean_episode_length:.1f} "
        f"aim_invisible={report.aim_while_invisible:.3f} "
        f"passed_m2={report.passed_m2} opponent={report.opponent}"
    )

    return report


def _run_one_episode(
    env: Any,
    policy: GreedyPolicy,
    *,
    episode_index: int,
    seed: int,
    max_episode_steps: Optional[int],
    opponent: Optional[EvalOpponent] = None,
) -> EpisodeOutcome:
    """Roll out one greedy episode, accumulating per-component reward + the guard.

    Returns an :class:`EpisodeOutcome`. The episode terminates on the env's
    ``done`` (the real env enforces its own ``max_episode_steps``); the optional
    ``max_episode_steps`` is a belt-and-suspenders cap so a fake env that never
    sets ``done`` cannot hang the harness.

    The aim-bonus guard inspects the obs handed to ``policy.act`` for THIS step:
    if that obs reports ``visible == false`` AND the step's ``info`` carries a
    nonzero ``r_aim``, that aim bonus is summed into ``aim_while_invisible`` — so a
    spin-farming policy (rewarded for "aiming" at an unseen opponent) is caught.

    With an ``opponent``, the per-decision order mirrors
    ``agent.train.collect_episode`` EXACTLY — agent action, then one fresh
    ``env.raw_opponent_view()``, then ONE ``env.step`` carrying both. One
    ``env.step`` is one decision window and the opponent's swing meter is
    shadow-tracked by counting those windows, so a skipped, doubled, or cached
    view desynchronizes the meter (and a view read before the agent acts scores
    the opponent on a state the agent has already left).
    """
    # Clear the policy's per-episode state (e.g. LSTM hidden) BEFORE the reset obs
    # so no memory leaks across the episode boundary.
    policy.reset()
    obs = env.reset(seed=seed)
    # Episode boundary for the opponent, AFTER the reset: the reset re-arms the
    # opponent's shadow swing meter that its ATTACK gate reads.
    if opponent is not None:
        opponent.begin_episode()

    components: Dict[str, float] = {key: 0.0 for key in REWARD_COMPONENT_KEYS}
    aim_while_invisible = 0.0
    total_reward = 0.0
    length = 0
    info: Dict[str, Any] = {}
    done = False

    while not done:
        # Visibility of the obs the policy is about to act on. The aim bonus for the
        # resulting step is granted on the NEXT obs (s'), but the anti-spin guard is
        # about the agent farming aim while it cannot SEE the opponent; an unseen
        # opponent at s' is the spin-farm signal. We therefore read visibility from
        # the post-step obs below, alongside r_aim, so the two always agree.
        action = policy.act(obs)
        if opponent is None:
            # The M1/M2 line, unchanged: one positional argument, no opp_action
            # on the wire at all.
            next_obs, reward, done, info = env.step(action)
        else:
            # ONE view, ONE macro, ONE step. The view is read HERE (never cached
            # from an earlier step) and passed through untouched so its clamped
            # attack_cooldown reaches the opponent policy exactly as produced.
            opp_action = opponent.act(env.raw_opponent_view())
            next_obs, reward, done, info = env.step(action, opp_action=opp_action)

        total_reward += float(reward)
        length += 1

        # Accumulate every reward component for this step (defensive .get so a
        # missing key contributes 0 rather than raising).
        for key in REWARD_COMPONENT_KEYS:
            components[key] += float(info.get(key, 0.0))

        # Spin-farming guard: if the gated opponent is NOT visible on the resulting
        # observation, any r_aim granted this step is illegitimate. A correct env
        # never grants r_aim while invisible (the aim bonus is visibility-gated), so
        # this sum stays 0; a regression that leaks aim-while-invisible is caught.
        step_aim = float(info.get("r_aim", 0.0))
        if step_aim != 0.0 and not _is_visible(next_obs):
            aim_while_invisible += step_aim

        obs = next_obs

        if max_episode_steps is not None and length >= max_episode_steps:
            break

    return EpisodeOutcome(
        index=episode_index,
        result=_classify(info),
        length=length,
        total_reward=total_reward,
        components=components,
        aim_while_invisible=aim_while_invisible,
    )


# ---------------------------------------------------------------------------
# CLI — the LIVE M2 eval (AC6/TC13), run as part of T20.
#
# Loads a trained DuelingDRQN checkpoint, builds the greedy adapter, connects to
# the LIVE Node bridge / Paper server (which serves the stationary dummy), runs
# `evaluate`, prints the EvalReport JSON, and bakes the M2 gate into the exit code
# (0 iff passed_m2). torch is imported lazily inside main so the module stays
# importable (and offline-testable) without torch.
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluate",
        description=(
            "M2 eval harness (AC6/TC13): greedy (eps=0) DuelingDRQN vs the LIVE "
            "stationary dummy. Connects to a started Node bridge / Paper server, "
            "runs N episodes, reports win rate (gate: >=95%), mean/median episode "
            "length, the anti-reward-hacking guards (aim-while-invisible == 0, "
            "mean length < timeout cap), and the per-reward-component breakdown. "
            "Exits 0 iff passed_m2. The offline harness/logging logic is proved by "
            "tests/test_evaluate.py; this entry point is the T20 LIVE run."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="path to a trained DuelingDRQN checkpoint (state_dict or {'model': sd}).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="number of eval episodes (default: 100, per AC6).",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="bridge host for the live run (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5555,
        help="bridge TCP port for the live run (default: 5555).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="base RNG seed; episode i resets with seed + i (default: 0).",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="m2_eval",
        help="logger run name (default: m2_eval).",
    )
    parser.add_argument(
        "--log-backend",
        type=str,
        default="auto",
        help="metrics backend: auto|wandb|tensorboard|jsonl (default: auto).",
    )
    return parser


def _load_drqn(checkpoint_path: str, device: Any) -> Any:
    """Load a :class:`~agent.dqn.DuelingDRQN` from a checkpoint onto ``device``.

    Accepts either a raw ``state_dict`` or a checkpoint dict that wraps it under a
    ``"model"`` / ``"model_state_dict"`` / ``"state_dict"`` key (T20's checkpoint
    shape is not yet frozen, so be liberal in what we accept and loud if none
    match).
    """
    import torch  # lazy

    from agent.dqn import DuelingDRQN

    payload = torch.load(checkpoint_path, map_location=device)
    state_dict = payload
    if isinstance(payload, dict):
        for key in ("model", "model_state_dict", "state_dict", "online"):
            if key in payload and isinstance(payload[key], dict):
                state_dict = payload[key]
                break

    net = DuelingDRQN().to(device)
    net.load_state_dict(state_dict)
    net.eval()
    return net


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point for the LIVE M2 eval run (AC6/TC13), part of T20.

    Loads the checkpoint, builds the greedy :class:`DRQNGreedyPolicy`, connects a
    real :class:`~env.mc_pvp_env.TcpBridgeClient` to the bridge (which serves the
    stationary dummy), runs :func:`evaluate` for ``--episodes`` episodes, logs
    through a :class:`~eval.logging.MetricsLogger`, prints the report JSON, and
    EXITS WITH THE M2 GATE baked into the exit code: ``0`` iff ``passed_m2`` (win
    rate >= 95% AND aim-while-invisible == 0 AND mean length < the timeout cap),
    ``1`` otherwise — so the live run is usable as the AC6 pass/fail gate.

    Args:
        argv: Argument vector (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code (0 == passed the M2 gate).
    """
    import json

    import torch  # lazy: only the LIVE CLI needs torch

    from env.mc_pvp_env import MCPvPEnv, TcpBridgeClient

    args = _build_parser().parse_args(argv)

    device = torch.device("cpu")
    net = _load_drqn(args.checkpoint, device)
    policy = DRQNGreedyPolicy(net, device=device)

    logger = MetricsLogger(
        run_name=args.run_name,
        backend=args.log_backend,
        config={
            "checkpoint": args.checkpoint,
            "episodes": args.episodes,
            "host": args.host,
            "port": args.port,
            "seed": args.seed,
        },
    )

    transport = TcpBridgeClient(host=args.host, port=args.port)
    env = MCPvPEnv(transport=transport)
    try:
        report = evaluate(
            env,
            policy,
            n_episodes=args.episodes,
            logger=logger,
            base_seed=args.seed,
            is_live=True,
            log=lambda m: print(m, file=sys.stderr),
        )
    finally:
        env.close()
        logger.close()

    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))

    if not report.passed_m2:
        reasons: List[str] = []
        if report.win_rate < M2_WIN_RATE_THRESHOLD:
            reasons.append(
                f"win rate {report.win_rate:.3f} < {M2_WIN_RATE_THRESHOLD:.2f}"
            )
        if report.aim_while_invisible != 0.0:
            reasons.append(
                f"aim-while-invisible {report.aim_while_invisible:.3f} != 0 "
                "(spin-farming)"
            )
        if report.mean_episode_length >= float(report.timeout_cap):
            reasons.append(
                f"mean length {report.mean_episode_length:.1f} >= timeout cap "
                f"{report.timeout_cap} (run-away)"
            )
        print("FAIL (M2 gate): " + "; ".join(reasons), file=sys.stderr)

    return 0 if report.passed_m2 else 1


if __name__ == "__main__":
    raise SystemExit(main())
