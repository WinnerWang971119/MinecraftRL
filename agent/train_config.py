"""train_config — Training hyperparameters dataclass (OWNED by T16).

Centralizes every training knob for the n-step Double-DQN learner that trains
the Dueling-DRQN (``agent.dqn.DuelingDRQN``) from prioritized sequence replay
(``agent.replay.PrioritizedSequenceReplay``). Tuning the optimizer / target /
ε-schedule touches only this file; the loop in :mod:`agent.train` reads every
value from here. This is deliberately SEPARATE from
:mod:`agent.contract_config`, which holds the frozen cross-track interface
constants (versions, timing, ``MAX_EPISODE_STEPS``) — training hyperparameters
are not a contract and may be re-tuned without re-freezing the protocol.

All defaults below are STARTING POINTS marked ``TUNE`` per the training-spec
§5/§8 start table; the docstring on each field records the role and the tuning
direction so the tuner (T17/T20) knows which way to move it and why.

------------------------------------------------------------------------------
Sequence / burn-in geometry (must agree with the replay buffer)
------------------------------------------------------------------------------
The replay buffer fixes ``seq_len`` (``L``, the SCORED learn span) and
``burn_in`` (``B``) at construction, because together they decide which window
start indices are valid. A sampled window therefore has ``B + L`` timesteps:
the first ``B`` warm the LSTM hidden state (gradients detached) and the last
``L`` carry the TD loss. ``TrainConfig`` is the single place those two numbers
are chosen, and :mod:`agent.train` passes them straight into the buffer and into
``DuelingDRQN.forward_with_burn_in`` so the geometry can never drift between the
storage and the learner.

------------------------------------------------------------------------------
Gotcha encoded here: ε decays per EPISODE, not per step
------------------------------------------------------------------------------
``eps_decay_episodes`` is measured in EPISODES on purpose (see
``agent.seeding`` and ``epsilon_for_episode`` in :mod:`agent.train`). Episodes
are short (tens of decisions), so a per-STEP decay collapses ε in a handful of
episodes and silently kills exploration. Keeping the schedule on the episode
boundary is the documented fix.

Owner: T16 (DQN core track)
"""

from __future__ import annotations

from dataclasses import dataclass

from agent.replay import DEFAULT_ALPHA, DEFAULT_BETA0, DEFAULT_PRIORITY_EPS

__all__ = ["TrainConfig"]


@dataclass(frozen=True)
class TrainConfig:
    """Frozen bundle of training hyperparameters for the n-step Double-DQN loop.

    Frozen so a config is a stable, hashable record of the run (it can be logged,
    checkpointed, and compared for drift). Construct a new instance with
    ``dataclasses.replace`` to vary a knob rather than mutating in place.

    Every field is a ``TUNE`` starting point; see the per-field comments for the
    role and the direction to move it.
    """

    # -- optimizer --------------------------------------------------------
    #: TUNE — Adam learning rate. 1e-4 is the DQN-standard starting point: large
    #: enough to make progress within the kickoff budget, small enough that the
    #: bootstrapped target does not diverge. Lower it if the loss/Q-values blow
    #: up; raise it cautiously if learning stalls.
    lr: float = 1e-4

    #: TUNE — number of sequences per gradient step. 32 balances gradient noise
    #: against the per-step compute (the net is tiny — §4 — so the bottleneck is
    #: the Minecraft servers, not the batch size). Raise for a smoother gradient
    #: once throughput allows; lower if memory-bound.
    batch_size: int = 32

    # -- sequence / burn-in geometry (must match the replay buffer) -------
    #: TUNE — ``L``, the SCORED learn-span length per sampled window (R2D2, §5.5).
    #: 16 gives the LSTM a meaningful horizon of credit assignment without making
    #: windows so long that few start indices are valid in short episodes. Must be
    #: passed to the replay buffer's ``seq_len``. TUNE 8–16.
    seq_len: int = 16

    #: TUNE — ``B``, burn-in steps that warm the LSTM hidden state before the
    #: scored span (gradients detached, §5.5). 4 is enough to seed memory for the
    #: scored window; the full sampled window is ``B + L`` steps. Must be passed
    #: to the replay buffer's ``burn_in``. 0 disables burn-in. TUNE 0–8.
    burn_in: int = 4

    # -- return / discount ------------------------------------------------
    #: TUNE — n-step return horizon (§5/§8). 3 trades faster reward propagation
    #: (vs 1-step) against more target variance. Must match the replay buffer's
    #: ``n_step``. TUNE 3–5.
    n_step: int = 3

    #: TUNE — discount factor. 0.99 matches the reward config's ``gamma`` (the
    #: potential-shaping discount) so shaping stays policy-invariant. Must match
    #: the replay buffer's ``gamma``. Raise toward 0.997 for longer-horizon
    #: credit if episodes lengthen.
    gamma: float = 0.99

    # -- target network ---------------------------------------------------
    #: TUNE — soft-update interpolation rate (Polyak, §8). Every step
    #: ``θ_target ← τ·θ_online + (1−τ)·θ_target``. 0.005 tracks the online net
    #: closely enough to learn yet slowly enough to keep the target stable.
    #: Smaller == more stable / slower; larger == faster / less stable.
    tau: float = 0.005

    #: TUNE — when True (default) use the soft Polyak update above every step.
    #: A future hard-update variant (copy weights every K steps) would set this
    #: False and add an interval knob; the kickoff loop is soft-only.
    target_soft: bool = True

    # -- gradient clipping ------------------------------------------------
    #: TUNE — max global gradient L2 norm (§8). 10 caps the occasional large TD
    #: gradient from a surprising transition so a single outlier cannot destabilize
    #: the weights. Lower if training is noisy; rarely needs raising.
    grad_clip: float = 10.0

    # -- ε-greedy exploration (decayed per EPISODE — see module docstring) -
    #: TUNE — initial exploration rate. 1.0 == fully random at the very first
    #: episode so early replay is diverse.
    eps_start: float = 1.0

    #: TUNE — floor exploration rate. 0.05 keeps a small steady trickle of
    #: exploration for the rest of training (never anneal fully to 0 — the agent
    #: must keep probing alternatives in a non-stationary self-play world).
    eps_end: float = 0.05

    #: TUNE — number of EPISODES (NOT steps) over which ε decays linearly from
    #: ``eps_start`` to ``eps_end``; it is flat at ``eps_end`` afterwards. Sized
    #: at ~10–20% of the planned episode budget so exploration is rich early and
    #: largely exploitative for the bulk of training. Per-EPISODE on purpose
    #: (see module docstring / agent.seeding): a per-step decay collapses ε far
    #: too fast on short episodes. TUNE to ~15% of total episodes.
    eps_decay_episodes: int = 200

    # -- replay buffer sizing --------------------------------------------
    #: TUNE — max stored TRANSITIONS in the prioritized sequence replay. 100k is a
    #: large, diverse history that fits comfortably in RAM for the kickoff
    #: single-arena setup (the buffer is pure-NumPy). Raise toward 1e6 for the
    #: distributed scale-up.
    replay_capacity: int = 100_000

    #: TUNE — transitions that must be stored before the FIRST gradient step
    #: (warm-up). 1000 ensures the initial batches are not dominated by a handful
    #: of correlated early episodes. Gating is on ``len(replay) >= min_replay``.
    #: TUNE: keep >= a few full windows; raise for more diverse warm-up.
    min_replay: int = 1_000

    # -- PER (prioritized experience replay) knobs -----------------------
    #: TUNE — PER priority exponent α (§5.4). Mirrors the replay default; 0 ==
    #: uniform sampling, 1 == full prioritization. 0.6 is the standard middle.
    per_alpha: float = DEFAULT_ALPHA

    #: TUNE — initial importance-sampling exponent β0 (§5.4), annealed toward 1.0
    #: over ``per_beta_anneal_steps`` gradient steps. 0.4 is the standard start.
    per_beta0: float = DEFAULT_BETA0

    #: TUNE — gradient steps over which β anneals from ``per_beta0`` to 1.0. Sized
    #: to roughly the planned training length so IS correction is full by the end.
    per_beta_anneal_steps: int = 100_000

    #: TUNE — ε_p added to |TD error| so no transition ever has zero priority
    #: (§5.4). Mirrors the replay default; rarely tuned.
    per_priority_eps: float = DEFAULT_PRIORITY_EPS

    # -- seeding / cadence -----------------------------------------------
    #: TUNE — base RNG seed. The action-sampling RNG for episode ``ep`` is seeded
    #: deterministically from ``seed + ep`` so an entire run is replayable.
    seed: int = 0

    #: TUNE — gradient steps between eval hook invocations (T19 owns eval itself;
    #: this only sets the cadence at which the hook is CALLED). 0 disables.
    eval_interval: int = 5_000

    #: TUNE — gradient steps between checkpoint hook invocations (T20 owns the
    #: actual save; this only sets the cadence). 0 disables.
    checkpoint_interval: int = 10_000

    #: TUNE — gradient steps between log hook invocations. 0 disables.
    log_interval: int = 100

    # -- multi-arena / distributed (issue #4) --------------------------------
    #: Number of parallel Minecraft arena processes. 1 == today's single-env
    #: path (no threading, no weight sync, no fault handling — all overhead
    #: below is a no-op at this value). Set > 1 to engage the ActorPool (T7).
    arenas: int = 1

    #: Learner grad steps between weight-snapshot pushes to the collectors.
    #: A snapshot is a cheap state-dict copy; 50 steps keeps collectors within
    #: ~50 updates of the learner without hammering serialization. Only read
    #: when ``arenas > 1``; ignored in single-arena mode. TUNE 20–100.
    weight_sync_every_k_steps: int = 50

    #: When True, the ActorPool attempts a background relaunch of an arena
    #: process that died mid-run (network drop, JVM crash, etc.). Only read
    #: when ``arenas > 1``. Set False to surface arena failures immediately
    #: instead of masking them with retries.
    fault_relaunch: bool = True

    #: Minimum number of live arenas required to keep the run going. If live
    #: arenas drop below this floor (after relaunch attempts are exhausted),
    #: the training loop aborts. 1 means the run continues as long as at least
    #: one arena is alive. Only read when ``arenas > 1``.
    fault_min_live_arenas: int = 1

    #: Max items in the inter-thread experience queue that collectors push to
    #: and the learner pops from. 0 == unbounded ``queue.Queue`` (no
    #: backpressure). A positive value caps memory and applies backpressure to
    #: fast collectors, at the cost of potential collector stalls. Only read
    #: when ``arenas > 1``; ignored in single-arena mode. TUNE 0–1000.
    collector_queue_max: int = 0

    #: Per-arena RNG seed offset. Arena ``i`` receives seed ``seed + i *
    #: seed_stride``, keeping episodes across arenas statistically independent
    #: even when ``seed`` is small. 1_000_000 gives each arena a
    #: million-episode independent band. Only read when ``arenas > 1``.
    seed_stride: int = 1_000_000

    def __post_init__(self) -> None:
        """Validate the hyperparameters so a misconfigured run fails loudly.

        Frozen dataclasses forbid attribute assignment in ``__post_init__``, but
        validation (which only reads) is allowed and is the right place to reject
        nonsensical values before the training loop wires them into the net /
        buffer / optimizer.
        """
        if self.lr <= 0.0:
            raise ValueError(f"lr must be > 0, got {self.lr}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}")
        if self.seq_len <= 0:
            raise ValueError(f"seq_len must be > 0, got {self.seq_len}")
        if self.burn_in < 0:
            raise ValueError(f"burn_in must be >= 0, got {self.burn_in}")
        if self.burn_in >= self.seq_len + self.burn_in:
            # Defensive: B must leave at least one scored step (always true while
            # seq_len > 0, but assert the invariant the net relies on).
            raise ValueError(
                f"burn_in {self.burn_in} leaves no scored steps for seq_len "
                f"{self.seq_len}"
            )
        if self.n_step < 1:
            raise ValueError(f"n_step must be >= 1, got {self.n_step}")
        if not (0.0 <= self.gamma <= 1.0):
            raise ValueError(f"gamma must be in [0, 1], got {self.gamma}")
        if not (0.0 < self.tau <= 1.0):
            raise ValueError(f"tau must be in (0, 1], got {self.tau}")
        if self.grad_clip <= 0.0:
            raise ValueError(f"grad_clip must be > 0, got {self.grad_clip}")
        if not (0.0 <= self.eps_end <= self.eps_start <= 1.0):
            raise ValueError(
                "epsilons must satisfy 0 <= eps_end <= eps_start <= 1, got "
                f"eps_start={self.eps_start}, eps_end={self.eps_end}"
            )
        if self.eps_decay_episodes < 1:
            raise ValueError(
                f"eps_decay_episodes must be >= 1, got {self.eps_decay_episodes}"
            )
        if self.replay_capacity <= 0:
            raise ValueError(
                f"replay_capacity must be > 0, got {self.replay_capacity}"
            )
        if self.min_replay < 0:
            raise ValueError(f"min_replay must be >= 0, got {self.min_replay}")
        if self.per_alpha < 0.0:
            raise ValueError(f"per_alpha must be >= 0, got {self.per_alpha}")
        if not (0.0 <= self.per_beta0 <= 1.0):
            raise ValueError(f"per_beta0 must be in [0, 1], got {self.per_beta0}")
        if self.per_beta_anneal_steps <= 0:
            raise ValueError(
                f"per_beta_anneal_steps must be > 0, got {self.per_beta_anneal_steps}"
            )
        if self.per_priority_eps < 0.0:
            raise ValueError(
                f"per_priority_eps must be >= 0, got {self.per_priority_eps}"
            )
        if self.seed < 0:
            raise ValueError(f"seed must be >= 0, got {self.seed}")
        if self.eval_interval < 0:
            raise ValueError(f"eval_interval must be >= 0, got {self.eval_interval}")
        if self.checkpoint_interval < 0:
            raise ValueError(
                f"checkpoint_interval must be >= 0, got {self.checkpoint_interval}"
            )
        if self.log_interval < 0:
            raise ValueError(f"log_interval must be >= 0, got {self.log_interval}")
        if self.arenas < 1:
            raise ValueError(f"arenas must be >= 1, got {self.arenas}")
        if self.weight_sync_every_k_steps < 1:
            raise ValueError(
                f"weight_sync_every_k_steps must be >= 1, got "
                f"{self.weight_sync_every_k_steps}"
            )
        if self.fault_min_live_arenas < 1:
            raise ValueError(
                f"fault_min_live_arenas must be >= 1, got {self.fault_min_live_arenas}"
            )
        if self.collector_queue_max < 0:
            raise ValueError(
                f"collector_queue_max must be >= 0, got {self.collector_queue_max}"
            )
        if self.seed_stride < 1:
            raise ValueError(f"seed_stride must be >= 1, got {self.seed_stride}")
