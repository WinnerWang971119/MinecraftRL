"""run_random — M1 tracer bullet: random policy end-to-end smoke run.

Drives episodes of :class:`~env.mc_pvp_env.MCPvPEnv` with the uniform random
policy (:class:`agent.random_policy.RandomPolicy`) vs the idle dummy and watches
for crashes, hangs, and memory growth. It is the M1 *plumbing* proof: it never
learns, but it exercises the FULL training path — rollout -> store -> sample ->
(no-op) update — so the slice that real training (T16) plugs into is shown to
work end-to-end before any network exists.

------------------------------------------------------------------------------
Two run modes (the injectable transport seam)
------------------------------------------------------------------------------
The env talks to the world only through an injectable bridge transport (the
four-method ``BridgeTransport`` protocol in ``env/mc_pvp_env.py``). This runner is
parameterized by a ``transport_factory`` — a zero-arg callable that returns a
fresh transport — so it runs in two modes with identical code:

  * LIVE (the AC3 run):  the factory builds a real
    :class:`~env.mc_pvp_env.TcpBridgeClient` pointed at the Node bridge in front
    of the Paper server. Driven from :func:`main` / the CLI.
  * OFFLINE (the unit test): the factory builds a FAKE scripted bridge that
    returns canned ``state`` / ``reset_ack`` messages, so the whole loop runs
    with no socket and no server. ``tests/test_run_random.py`` uses this to prove
    the runner offline.

------------------------------------------------------------------------------
LIVE FOLLOW-UP (TC11 / AC3) — NOT covered offline
------------------------------------------------------------------------------
TC11 / AC3 require the MEANINGFUL run: a random policy completing >= 100 full
episodes end-to-end vs the idle dummy through the REAL Paper server + REAL bridge,
with ZERO crashes and combined-process RSS (Node + Python + JVM) growth
< ~200 MB sampled every 10 episodes. That run needs a live server and is a
documented HUMAN follow-up: run ``python -m eval.run_random`` against a started
bridge/server (see ``bridge`` / ``server`` setup) and confirm the printed summary
(zero crashes, RSS growth under the budget).

This task delivers the runner plus the OFFLINE fake-bridge proof. Offline, the
RSS sampler only watches the single Python process (no Node/JVM exist), so the
< ~200 MB combined-process budget is asserted on the LIVE run; the test only
exercises the sampling + bookkeeping code path.

Owner: T10 (Eval/infra track)
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from agent.actions import N_ACTIONS
from agent.random_policy import RandomPolicy
from agent.seeding import seed_action_space, seed_everything
from env.mc_pvp_env import (
    BridgeError,
    BridgeTransport,
    MCPvPEnv,
    TcpBridgeClient,
)
from env.observation_spec import OBS_DIM

__all__ = [
    "ToyReplayBuffer",
    "noop_grad_step",
    "RssSampler",
    "EpisodeRecord",
    "RunResult",
    "run_random",
    "main",
    "DEFAULT_EPISODES",
    "DEFAULT_REPLAY_CAPACITY",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_MIN_REPLAY",
    "RSS_SAMPLE_EVERY",
    "RSS_GROWTH_BUDGET_MB",
]


# ---------------------------------------------------------------------------
# Run knobs (kept small — this is a smoke run, not a training job).
# ---------------------------------------------------------------------------

#: Episodes the live AC3 run must complete (>= 100 per TC11).
DEFAULT_EPISODES: int = 100

#: Toy replay capacity in TRANSITIONS. Small on purpose; the buffer only has to
#: prove the rollout -> store -> sample path, not hold a real training corpus.
DEFAULT_REPLAY_CAPACITY: int = 50_000

#: Transitions sampled per no-op update step.
DEFAULT_BATCH_SIZE: int = 32

#: Transitions that must be buffered before the first no-op update runs (so the
#: update path is exercised only once there is something to sample, mirroring the
#: MIN_REPLAY gate of the real trainer).
DEFAULT_MIN_REPLAY: int = 64

#: RSS is sampled every this-many episodes (AC3: "sampled every 10 episodes").
RSS_SAMPLE_EVERY: int = 10

#: Combined-process RSS growth budget for the LIVE run, in megabytes (AC3).
RSS_GROWTH_BUDGET_MB: float = 200.0


# ---------------------------------------------------------------------------
# Toy in-memory replay buffer.
#
# A deliberately minimal ring buffer of raw transitions. The REAL training buffer
# is agent/replay.py (a prioritized SEQUENCE buffer); this one exists only so the
# M1 tracer exercises store + sample without dragging in PER/sequence machinery.
# ---------------------------------------------------------------------------


#: A single stored transition: (obs, action, reward, next_obs, done).
Transition = Tuple[np.ndarray, int, float, np.ndarray, bool]


class ToyReplayBuffer:
    """A tiny fixed-capacity ring buffer of ``(obs, a, r, next_obs, done)`` tuples.

    Stores transitions in insertion order and overwrites the oldest once full
    (FIFO ring). Sampling draws ``batch_size`` transitions uniformly at random
    *with replacement* from its own seeded generator, so the M1 update path is
    reproducible and independent of the global RNG.

    This is intentionally not the real :mod:`agent.replay` buffer — it only has to
    prove the rollout->store->sample plumbing for the tracer.

    Args:
        capacity: Maximum stored transitions (> 0). Oldest are overwritten.
        rng: NumPy ``Generator`` for sampling. Defaults to a fresh ``default_rng``;
            pass a seeded one for deterministic sampling.

    Raises:
        ValueError: if ``capacity`` is not positive.
    """

    def __init__(
        self, capacity: int, rng: Optional[np.random.Generator] = None
    ) -> None:
        cap = int(capacity)
        if cap <= 0:
            raise ValueError(f"capacity must be positive, got {capacity!r}")
        self._capacity = cap
        self._rng = rng if rng is not None else np.random.default_rng()
        self._storage: List[Transition] = []
        self._cursor = 0  # next write position once the buffer is full
        self._total_added = 0  # lifetime count (for bookkeeping / assertions)

    # -- sizing ------------------------------------------------------------

    def __len__(self) -> int:
        """Number of transitions currently stored (caps at ``capacity``)."""
        return len(self._storage)

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def total_added(self) -> int:
        """Lifetime number of transitions ever pushed (ignores overwrites)."""
        return self._total_added

    def is_ready(self, min_transitions: int) -> bool:
        """True once at least ``min_transitions`` are stored."""
        return len(self._storage) >= int(min_transitions)

    # -- insertion ---------------------------------------------------------

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Store one transition, overwriting the oldest if at capacity.

        ``obs`` / ``next_obs`` are copied to float32 arrays so a later in-place
        mutation of the caller's observation can never corrupt stored history.
        """
        transition: Transition = (
            np.asarray(obs, dtype=np.float32).copy(),
            int(action),
            float(reward),
            np.asarray(next_obs, dtype=np.float32).copy(),
            bool(done),
        )
        if len(self._storage) < self._capacity:
            self._storage.append(transition)
        else:
            self._storage[self._cursor] = transition
            self._cursor = (self._cursor + 1) % self._capacity
        self._total_added += 1

    # -- sampling ----------------------------------------------------------

    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample ``batch_size`` transitions uniformly (with replacement).

        Returns a struct-of-arrays batch ready to feed a (no-op) update:
        ``(obs, actions, rewards, next_obs, dones)`` with leading axis
        ``batch_size``.

        Raises:
            ValueError: if the buffer is empty or ``batch_size`` is not positive.
        """
        bs = int(batch_size)
        if bs <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size!r}")
        n = len(self._storage)
        if n == 0:
            raise ValueError("cannot sample from an empty replay buffer")

        idx = self._rng.integers(0, n, size=bs)
        obs = np.stack([self._storage[i][0] for i in idx], axis=0)
        actions = np.asarray([self._storage[i][1] for i in idx], dtype=np.int64)
        rewards = np.asarray([self._storage[i][2] for i in idx], dtype=np.float32)
        next_obs = np.stack([self._storage[i][3] for i in idx], axis=0)
        dones = np.asarray([self._storage[i][4] for i in idx], dtype=bool)
        return obs, actions, rewards, next_obs, dones


def noop_grad_step(
    batch: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
) -> float:
    """Run a NO-OP gradient step over a sampled batch and return a scalar "loss".

    This exercises the *shape* of a learner update without a network: it does a
    trivial deterministic forward (a fixed linear reduction of the obs batch),
    forms a placeholder TD-style target from the rewards, and computes a mean
    squared "loss" — then discards it (no parameters, no autograd). Its only job
    is to prove that the rollout -> store -> sample -> update plumbing runs every
    cycle and is fed well-formed, finite arrays.

    Args:
        batch: ``(obs, actions, rewards, next_obs, dones)`` from
            :meth:`ToyReplayBuffer.sample`.

    Returns:
        A finite scalar stand-in "loss" (purely for logging/assertion).

    Raises:
        ValueError: if any array in the batch is empty or non-finite.
    """
    obs, actions, rewards, next_obs, dones = batch
    if obs.size == 0 or next_obs.size == 0:
        raise ValueError("noop_grad_step received an empty batch")
    if not (np.all(np.isfinite(obs)) and np.all(np.isfinite(next_obs))):
        raise ValueError("noop_grad_step received non-finite observations")

    # Dummy "forward": collapse each obs vector to a single scalar via a fixed
    # mean. Stand-in for Q(s, a) and max_a' Q(s', a').
    q_pred = obs.mean(axis=1)
    q_next = next_obs.mean(axis=1)
    # Placeholder one-step target: r + (1 - done) * q_next (no discount needed for
    # a no-op). Casting dones to float gives the (1 - done) bootstrap mask.
    target = rewards + (1.0 - dones.astype(np.float32)) * q_next
    # Mean squared "loss". No backward pass — there are no parameters.
    loss = float(np.mean((q_pred - target) ** 2))
    return loss


# ---------------------------------------------------------------------------
# RSS sampling (AC3): track combined process resident-set growth.
# ---------------------------------------------------------------------------


class RssSampler:
    """Samples process resident-set size (RSS) and tracks growth from a baseline.

    On the LIVE run the meaningful figure is the COMBINED Node + Python + JVM RSS;
    the cleanest portable proxy is the whole process tree rooted at this Python
    process (the bridge Node child and, transitively, the JVM it launches). This
    sampler walks that tree when :mod:`psutil` is importable. Fallbacks, in order:

      1. ``psutil`` present  -> sum RSS of this process + all descendants
         (covers Node + JVM when they are this process's children).
      2. no psutil, POSIX    -> ``resource.getrusage`` self-RSS only (no children;
         logged as a degraded measurement).
      3. no psutil, Windows  -> no portable RSS source; sampling is DISABLED and a
         one-time note is recorded. Bookkeeping (sample count, cadence) still runs.

    The first successful sample becomes the baseline; :attr:`growth_mb` reports the
    peak-minus-baseline growth in megabytes so a caller can assert it stays under
    the AC3 budget on the live run.
    """

    _BYTES_PER_MB = 1024.0 * 1024.0

    def __init__(self) -> None:
        self.samples_mb: List[float] = []
        self.baseline_mb: Optional[float] = None
        self.peak_mb: Optional[float] = None
        self.available: bool = True
        self.note: Optional[str] = None
        self._psutil = None
        self._resource = None

        # Resolve a sampling backend once, recording a note on degradation.
        try:
            import psutil  # type: ignore

            self._psutil = psutil
            self._proc = psutil.Process()
        except Exception:  # pragma: no cover - psutil import/availability is env-specific
            try:
                import resource  # POSIX-only

                self._resource = resource
                self.note = (
                    "psutil not importable; falling back to resource.getrusage "
                    "(self-RSS only, excludes Node/JVM children)"
                )
            except Exception:
                # Windows without psutil: no portable combined-RSS source.
                self.available = False
                self.note = (
                    "RSS sampling unavailable (no psutil, no POSIX resource module); "
                    "RSS growth not tracked on this platform"
                )

    def _read_rss_mb(self) -> Optional[float]:
        """Read current RSS in MB via the resolved backend, or ``None`` if disabled."""
        if self._psutil is not None:
            total = 0
            try:
                total += self._proc.memory_info().rss
                # Include the bridge Node child and the JVM it spawns. Tolerate
                # children that vanish mid-walk (a finished arena process).
                for child in self._proc.children(recursive=True):
                    try:
                        total += child.memory_info().rss
                    except Exception:
                        continue
            except Exception:
                return None
            return total / self._BYTES_PER_MB

        if self._resource is not None:
            usage = self._resource.getrusage(self._resource.RUSAGE_SELF)
            # ru_maxrss units differ by platform: kilobytes on Linux, bytes on
            # macOS. We only need a stable relative growth signal, so normalize to
            # MB assuming kilobytes (the Linux convention) — the absolute value is
            # not asserted offline, only the growth trend, and the LIVE run uses
            # psutil regardless.
            return float(usage.ru_maxrss) / 1024.0

        return None

    def sample(self) -> Optional[float]:
        """Take one RSS sample (MB), update baseline/peak, and return it.

        Returns ``None`` when sampling is unavailable on this platform (the call
        is still counted by the caller's cadence bookkeeping).
        """
        if not self.available:
            return None
        rss_mb = self._read_rss_mb()
        if rss_mb is None:
            return None
        self.samples_mb.append(rss_mb)
        if self.baseline_mb is None:
            self.baseline_mb = rss_mb
            self.peak_mb = rss_mb
        else:
            # peak_mb is always set once baseline_mb is; track the running max.
            self.peak_mb = max(self.peak_mb, rss_mb)
        return rss_mb

    @property
    def growth_mb(self) -> float:
        """Peak RSS growth above the baseline, in MB (0.0 before two samples)."""
        if self.baseline_mb is None or self.peak_mb is None:
            return 0.0
        return max(0.0, self.peak_mb - self.baseline_mb)

    def within_budget(self, budget_mb: float = RSS_GROWTH_BUDGET_MB) -> bool:
        """True if measured growth is under ``budget_mb`` (True when not measured).

        When sampling is unavailable (or only the baseline exists) there is no
        evidence of a leak, so this returns ``True`` — the LIVE run, where psutil
        is present, is where the budget is genuinely enforced.
        """
        if not self.available or len(self.samples_mb) < 2:
            return True
        return self.growth_mb < float(budget_mb)


# ---------------------------------------------------------------------------
# Per-episode + aggregate bookkeeping.
# ---------------------------------------------------------------------------


@dataclass
class EpisodeRecord:
    """Outcome of one finished episode.

    Attributes:
        index: 0-based episode index within the run.
        outcome: One of ``"win"``, ``"loss"``, ``"timeout"``.
        length: Number of decision steps taken before termination.
        total_reward: Sum of step rewards over the episode.
    """

    index: int
    outcome: str
    length: int
    total_reward: float


@dataclass
class RunResult:
    """Aggregate result of a tracer run (returned by :func:`run_random`).

    Attributes:
        episodes: Per-episode records, in order.
        wins / losses / timeouts: Outcome counts.
        crashes: Number of episodes aborted by a :class:`BridgeError` (0 is the
            AC3 success condition).
        win_rate: Wins / completed episodes (0.0 if none completed).
        replay_size: Final number of transitions stored in the toy buffer.
        replay_total_added: Lifetime transitions pushed to the buffer.
        grad_steps: Number of no-op update steps that ran.
        last_loss: The most recent no-op "loss" (``None`` if no update ran).
        rss: The :class:`RssSampler` used (for growth / budget inspection).
    """

    episodes: List[EpisodeRecord] = field(default_factory=list)
    wins: int = 0
    losses: int = 0
    timeouts: int = 0
    crashes: int = 0
    replay_size: int = 0
    replay_total_added: int = 0
    grad_steps: int = 0
    last_loss: Optional[float] = None
    rss: Optional[RssSampler] = None

    @property
    def completed(self) -> int:
        """Number of episodes that terminated normally (win + loss + timeout)."""
        return self.wins + self.losses + self.timeouts

    @property
    def win_rate(self) -> float:
        """Fraction of completed episodes that were wins (0.0 if none completed)."""
        done = self.completed
        return (self.wins / done) if done else 0.0


def _classify(info: dict) -> str:
    """Map a terminal ``info`` dict to ``"win"`` / ``"loss"`` / ``"timeout"``.

    Mirrors the mutually-exclusive outcome flags set by :meth:`MCPvPEnv.step`. A
    loss takes precedence (a double-death is a loss), then a win, else timeout.
    """
    if info.get("lost"):
        return "loss"
    if info.get("won"):
        return "win"
    return "timeout"


# ---------------------------------------------------------------------------
# The runner.
# ---------------------------------------------------------------------------


def run_random(
    transport_factory: Callable[[], BridgeTransport],
    *,
    episodes: int = DEFAULT_EPISODES,
    seed: int = 0,
    replay_capacity: int = DEFAULT_REPLAY_CAPACITY,
    batch_size: int = DEFAULT_BATCH_SIZE,
    min_replay: int = DEFAULT_MIN_REPLAY,
    rss_sample_every: int = RSS_SAMPLE_EVERY,
    rss_growth_budget_mb: float = RSS_GROWTH_BUDGET_MB,
    max_episode_steps: Optional[int] = None,
    log: Optional[Callable[[str], None]] = print,
) -> RunResult:
    """Run the M1 tracer: random policy vs the idle dummy, end to end.

    Each episode: ``env.reset()`` -> loop ``policy.act`` / ``env.step`` until
    ``done`` -> store every transition in the toy replay buffer -> once the buffer
    is warm, run a NO-OP gradient step per decision (rollout -> store -> sample ->
    update). Per-episode outcome, length, and a RUNNING win-rate are logged; RSS
    is sampled every ``rss_sample_every`` episodes and its growth tracked against
    the AC3 budget.

    The opponent is the idle dummy: in the offline fake-bridge mode the scripted
    ``state`` messages already encode an idle, immune opponent, and on the live
    run the bridge connects the learner against ``opponents.dummy.StationaryDummy``
    — so this loop does not drive the opponent policy itself; the env/bridge do.

    A :class:`BridgeError` raised by an episode is caught, counted as a CRASH, and
    the run continues to the next episode rather than aborting the whole tracer
    (so one flaky reset does not lose the other 99 episodes of evidence). The live
    AC3 success condition is ``result.crashes == 0``.

    Args:
        transport_factory: Zero-arg callable returning a fresh
            :class:`~env.mc_pvp_env.BridgeTransport` (one per episode). Real
            :class:`~env.mc_pvp_env.TcpBridgeClient` for the live run; a fake for
            tests.
        episodes: Number of episodes to run (>= 1).
        seed: Base seed. The action policy is reseeded per episode with
            ``seed + episode_index`` so exploration is reproducible.
        replay_capacity: Toy buffer capacity in transitions.
        batch_size: Transitions per no-op update.
        min_replay: Transitions buffered before the first no-op update.
        rss_sample_every: Sample RSS once every this-many episodes (>= 1).
        rss_growth_budget_mb: Growth budget passed through to the summary.
        max_episode_steps: Optional override of the env episode horizon (tests use
            a tiny value to keep episodes short); ``None`` uses the frozen default.
        log: A ``str -> None`` sink for progress lines (``print`` by default;
            pass ``None`` to silence).

    Returns:
        A :class:`RunResult` with outcome counts, win-rate, replay/grad-step
        bookkeeping, and the RSS sampler.

    Raises:
        ValueError: if ``episodes`` < 1 or ``rss_sample_every`` < 1.
    """
    if episodes < 1:
        raise ValueError(f"episodes must be >= 1, got {episodes}")
    if rss_sample_every < 1:
        raise ValueError(f"rss_sample_every must be >= 1, got {rss_sample_every}")

    def _emit(message: str) -> None:
        if log is not None:
            log(message)

    # Seed every global RNG once (Python/NumPy/torch-if-present) for the run.
    seed_everything(seed)

    # Dedicated, seeded generators so the policy and the buffer are reproducible
    # and independent of the global RNG (the seeding gotcha — see agent.seeding).
    policy = RandomPolicy(seed=seed, n_actions=N_ACTIONS)
    buffer_rng = np.random.default_rng(seed + 1)
    buffer = ToyReplayBuffer(replay_capacity, rng=buffer_rng)
    rss = RssSampler()
    result = RunResult(rss=rss)

    if rss.note:
        _emit(f"[rss] {rss.note}")

    # Env kwargs: only pass max_episode_steps when overridden (None means "frozen
    # default", which the env supplies itself).
    env_kwargs = {}
    if max_episode_steps is not None:
        env_kwargs["max_episode_steps"] = max_episode_steps

    for ep in range(episodes):
        # Per-episode reproducible exploration: reseed the policy's action stream.
        episode_seed = seed + ep
        seed_action_space(policy, episode_seed)

        transport = transport_factory()
        try:
            with MCPvPEnv(transport=transport, **env_kwargs) as env:
                outcome, length, total_reward = _run_one_episode(
                    env,
                    policy,
                    buffer,
                    episode_seed=episode_seed,
                    batch_size=batch_size,
                    min_replay=min_replay,
                    result=result,
                    obs_dim=OBS_DIM,
                )
        except BridgeError as exc:
            # A transport failure aborts THIS episode only. Count it as a crash and
            # keep going so the run still reports the surviving episodes.
            result.crashes += 1
            _emit(f"[ep {ep:>4}] CRASH (BridgeError): {exc}")
            continue

        # Tally and record the outcome.
        if outcome == "win":
            result.wins += 1
        elif outcome == "loss":
            result.losses += 1
        else:
            result.timeouts += 1
        result.episodes.append(
            EpisodeRecord(
                index=ep, outcome=outcome, length=length, total_reward=total_reward
            )
        )

        # RSS sampling cadence (every Nth episode, 1-based so ep 10/20/... sample).
        if (ep + 1) % rss_sample_every == 0:
            rss.sample()

        _emit(
            f"[ep {ep:>4}] {outcome:<7} len={length:>4} "
            f"R={total_reward:+.2f} win_rate={result.win_rate:.3f} "
            f"replay={len(buffer)} grad_steps={result.grad_steps}"
        )

    # Final bookkeeping snapshot.
    result.replay_size = len(buffer)
    result.replay_total_added = buffer.total_added

    _emit(
        f"[done] episodes={result.completed} crashes={result.crashes} "
        f"wins={result.wins} losses={result.losses} timeouts={result.timeouts} "
        f"win_rate={result.win_rate:.3f}"
    )
    _emit(
        f"[done] replay={result.replay_size} (added {result.replay_total_added}) "
        f"grad_steps={result.grad_steps} last_loss="
        f"{'n/a' if result.last_loss is None else f'{result.last_loss:.4f}'}"
    )
    if rss.available and len(rss.samples_mb) >= 2:
        ok = rss.within_budget(rss_growth_budget_mb)
        _emit(
            f"[rss] baseline={rss.baseline_mb:.1f}MB peak={rss.peak_mb:.1f}MB "
            f"growth={rss.growth_mb:.1f}MB budget={rss_growth_budget_mb:.0f}MB "
            f"{'OK' if ok else 'OVER BUDGET'}"
        )
    else:
        _emit(
            "[rss] growth not asserted offline "
            "(combined Node+Python+JVM RSS is the LIVE AC3 check)"
        )

    return result


def _run_one_episode(
    env: MCPvPEnv,
    policy: RandomPolicy,
    buffer: ToyReplayBuffer,
    *,
    episode_seed: int,
    batch_size: int,
    min_replay: int,
    result: RunResult,
    obs_dim: int,
) -> Tuple[str, int, float]:
    """Roll out one full episode, storing transitions and running no-op updates.

    Returns ``(outcome, length, total_reward)``. ``outcome`` is
    ``"win"`` / ``"loss"`` / ``"timeout"``. The episode is guaranteed to terminate
    (the env enforces ``max_episode_steps``), so this never hangs on a fake bridge
    that keeps returning non-terminal states.

    ``episode_seed`` is threaded in from the outer loop (``seed + ep``) so both
    the world reset and the action-space seeding share the same deterministic
    per-episode seed, satisfying the AC7/TC14 reset-determinism intent.
    """
    obs = env.reset(seed=episode_seed)
    if obs.shape != (obs_dim,):
        # Defensive: a malformed obs would corrupt the buffer / no-op step.
        raise BridgeError(
            f"reset returned obs of shape {obs.shape}, expected ({obs_dim},)"
        )

    total_reward = 0.0
    length = 0
    info: dict = {}
    done = False
    while not done:
        action = policy.act(obs)
        next_obs, reward, done, info = env.step(action)

        # Store the transition (rollout -> store).
        buffer.add(obs, action, reward, next_obs, done)
        total_reward += reward
        length += 1

        # Exercise the update path (sample -> no-op update) once the buffer is
        # warm. This runs EVERY decision step, not just per episode, so the full
        # plumbing is hit many times even in short episodes.
        if buffer.is_ready(min_replay):
            batch = buffer.sample(batch_size)
            loss = noop_grad_step(batch)
            result.grad_steps += 1
            result.last_loss = loss

        obs = next_obs

    return _classify(info), length, total_reward


# ---------------------------------------------------------------------------
# CLI entry point — the LIVE M1 run (needs a started bridge + Paper server).
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_random",
        description=(
            "M1 tracer bullet: random policy vs the idle dummy, end to end. "
            "This connects to a LIVE Node bridge / Paper server (the AC3 run). "
            "The offline fake-bridge path is exercised by tests/test_run_random.py, "
            "not via a CLI flag."
        ),
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=DEFAULT_EPISODES,
        help=f"number of episodes to run (default: {DEFAULT_EPISODES})",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="bridge host for the live run (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5555,
        help="bridge TCP port for the live run (default: 5555)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="base RNG seed (default: 0)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point for the LIVE M1 tracer run.

    Builds a real :class:`~env.mc_pvp_env.TcpBridgeClient` per episode pointed at
    ``--host:--port`` and runs :func:`run_random`. Exits non-zero if any episode
    crashed or the combined-process RSS growth blew the AC3 budget, so the run is
    usable as a pass/fail gate.

    Args:
        argv: Argument vector (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code: 0 on a clean run (no crashes, RSS within budget),
        1 otherwise.
    """
    args = _build_parser().parse_args(argv)

    def factory() -> BridgeTransport:
        # A fresh client per episode mirrors the one-connection-per-arena contract;
        # the env owns its connect()/close() lifecycle.
        return TcpBridgeClient(host=args.host, port=args.port)

    result = run_random(
        factory,
        episodes=args.episodes,
        seed=args.seed,
    )

    crashed = result.crashes > 0
    over_budget = result.rss is not None and not result.rss.within_budget(
        RSS_GROWTH_BUDGET_MB
    )
    if crashed:
        print(f"FAIL: {result.crashes} episode(s) crashed", file=sys.stderr)
    if over_budget:
        print(
            f"FAIL: RSS growth {result.rss.growth_mb:.1f}MB exceeded the "
            f"{RSS_GROWTH_BUDGET_MB:.0f}MB budget",
            file=sys.stderr,
        )
    return 1 if (crashed or over_budget) else 0


if __name__ == "__main__":
    raise SystemExit(main())
