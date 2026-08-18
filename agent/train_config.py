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

------------------------------------------------------------------------------
Second gotcha: the ε schedule is counted in GLOBAL episodes, not per-arena ones
------------------------------------------------------------------------------
Under ``arenas > 1`` every collector claims its episode index from ONE shared
``distributed.actor.GlobalEpisodeCounter`` (``distributed/actor.py:672-674``
claims the index and feeds it straight into ``epsilon_for_episode``; the counter
itself is built at ``agent/train.py:2636`` and re-read for logging at
``agent/train.py:2787``). So ``eps_decay_episodes`` is consumed N times faster at
N pads: the single-arena value of 200 that this file used to ship floored ε after
8 episodes PER ARENA at 25 pads — about 1% of a one-night run, against the
field's own ~15% guidance. :func:`eps_decay_episodes_for` is the fix: the default
is COMPUTED from the arena count rather than written down for one pad.

Owner: T16 (DQN core track)
"""

from __future__ import annotations

from dataclasses import dataclass

from agent.replay import DEFAULT_ALPHA, DEFAULT_BETA0, DEFAULT_PRIORITY_EPS

__all__ = [
    "TrainConfig",
    "MEASURED_PER_ARENA_TRANSITIONS_PER_S",
    "MEASURED_MEAN_EPISODE_STEPS",
    "ASSUMED_RUN_HOURS",
    "ASSUMED_MEAN_EPISODE_STEPS",  # deprecated alias; see its definition below
    "EPS_DECAY_FRACTION_OF_RUN",
    "DEFAULT_EPS_DECAY_ARENAS",
    "projected_episodes",
    "eps_decay_episodes_for",
]


# ---------------------------------------------------------------------------
# Sizing the ε schedule from throughput (T16(a) / AC15 / AC17).
#
# The four inputs below are the ONLY quantities the default ε-decay window is
# built from. TWO of them are measured (the per-arena rate and the mean episode
# length) and two are assumptions, and each says which on its own line. Anything
# derived from them is still a PROJECTION, never a measurement — the run length
# is a plan for the night, not an observation of one — which is why
# ``--eps-decay-episodes`` exists.
# ---------------------------------------------------------------------------

#: MEASURED (AC16, 2026-08-16). Per-arena collection rate in transitions/second,
#: from the 600 s confirm run at 25 pads: 121.95/s aggregate over 25 arenas ==
#: 4.8782/s each (``runs/confirm-n25/summary.json``). Per-arena rate varied 0.05%
#: across N = 16/20/24/25, so treating it as a constant and multiplying by the
#: arena count is what the sweep actually licenses.
MEASURED_PER_ARENA_TRANSITIONS_PER_S: float = 4.8782

#: ASSUMPTION — the length of one unattended overnight run, in hours. 12 h is
#: "start it after dinner, read it at breakfast"; it is a plan for the night, not
#: a measurement of one. A shorter run makes the projected episode budget (and so
#: the decay window) proportionally smaller.
ASSUMED_RUN_HOURS: float = 12.0

#: MEASURED (smoke run, 2026-08-16). Mean decisions per TRAINING episode: 25
#: pads against the SCRIPTED opponent, warm-started from ``runs/m2_multi.pt``,
#: eval off, clean exit at 400 grad steps with ``episodes=519`` over 1212 s.
#: That is 285 steps/episode against a cap of ``MAX_EPISODE_STEPS`` that was 400
#: AT MEASUREMENT TIME (it is 600 now, see ``agent/contract_config.py``) — most
#: episodes ran most of the way to that timeout.
#:
#: CROSS-CHECK, because a mean episode length is easy to mis-derive: 519 x 285 /
#: 1212 s = 122.0 transitions/s, which closes to within 0.1% of the 121.955/s
#: aggregate measured independently by the AC16 pad sweep
#: (``MEASURED_PER_ARENA_TRANSITIONS_PER_S`` x 25). Two separate measurements of
#: the same stream agree, so the derivation is sound.
#:
#: WHICH OPPONENT THIS IS MEASURED AGAINST, because the number is not universal:
#: the scripted HARD tier, which flees at 6 HP and drags fights toward the
#: timeout. A human or the stationary dummy would produce a different (shorter)
#: figure. That makes 285 the right basis for sizing a TRAINING run specifically
#: — which is the only thing this constant feeds — and the wrong basis for
#: reasoning about eval episodes.
#:
#: WHICH GEAR THIS IS MEASURED AGAINST — the other axis the number is not
#: universal on, and it is ASYMMETRIC. This run predates M4: neither fighter
#: wore armor, the learner held an iron sword, and the opponent held NOTHING.
#: ``bridge/bot.js`` builds ``dummyResetTemplate`` with ``inventory: []`` ("the
#: datapack gives the dummy no weapon"), so the scripted HARD policy named above
#: was steering an EMPTY-HANDED body.
#:
#: M4 changes both halves at once. Full iron is 15 armor points at 0 toughness,
#: and per ``CombatRules.getDamageAfterAbsorb`` in the pinned jar the reduction
#: is ``g / 25`` with
#: ``g = clamp(armor - damage / (2 + toughness / 4), armor * 0.2, 20)``: for a
#: 6.0-damage iron sword ``g = clamp(15 - 3, 3, 20) == 12``, so 48% is absorbed,
#: ~3.1 lands, and 20 HP takes ~7 hits instead of ~4 — a ~1.75x stretch, not the
#: doubling a flat 4%-per-point model predicts. M4 ALSO arms the opponent.
#:
#: Those two changes move 285 in OPPOSITE directions, which is why no scaling
#: factor rescues it: armor lengthens fights, while an armed opponent kills the
#: learner sooner and pushes 285 DOWN. So 285 is doubly unrepresentative of the
#: armored self-play regime — no armor on either side, AND no weapon on one —
#: and the armored mean is a DIFFERENT number that is NOT YET MEASURED. Do not
#: extrapolate it from 285 — that is exactly the mistake the section below
#: documents happening once already, in the other direction. The real armored
#: figure has to come from T19's smoke run; until it does, pass
#: ``--eps-decay-episodes`` explicitly for the armored run instead of trusting
#: ``eps_decay_episodes_for``'s default, which is still sized off the bare-handed
#: value below.
#:
#: WHY THE OLD VALUE WAS WRONG, so nobody restores it. This shipped as
#: ``ASSUMED_MEAN_EPISODE_STEPS = 30.0``, obtained by doubling
#: ``mean_episode_length: 17.0`` from ``runs/m2_multi/summary.json``. That 17.0
#: is a GREEDY eval against a STATIONARY dummy — the single easiest episode this
#: environment can produce — so it was never a lower bound worth doubling; it was
#: 9.5x low. This value is the DIVISOR of the projected episode count, so too LOW
#: a length over-estimates the episodes and the decay window comes out far too
#: long: at 25 pads the derived default was 26,342 episodes == 142% of an entire
#: 12 h run, i.e. ε would never finish decaying and would sit near its start value
#: all night. Both directions are real failures — too high floors ε a few percent
#: into the run (the bug T16 exists to fix), too low never anneals it at all — so
#: there is no safe direction to err in. RE-MEASURE instead of padding, and set
#: ``--eps-decay-episodes N`` for a run whose regime differs from the one above.
MEASURED_MEAN_EPISODE_STEPS: float = 285.0

#: DEPRECATED ALIAS, kept only because :mod:`agent.train` still imports this
#: name. It is the same measured value; the "ASSUMED" spelling is now false.
#: Delete it (and the import in ``agent/train.py``, whose startup line still says
#: "assuming N steps/episode") the next time that module is edited.
ASSUMED_MEAN_EPISODE_STEPS: float = MEASURED_MEAN_EPISODE_STEPS

#: ASSUMPTION (field convention, and the number AC15 is written against) —
#: fraction of a run's projected episodes the ε decay should span. ~15% is the
#: DQN-lineage guidance: rich exploration early, mostly exploitative for the bulk
#: of training.
EPS_DECAY_FRACTION_OF_RUN: float = 0.15

#: The arena count :attr:`TrainConfig.eps_decay_episodes`'s DATACLASS default is
#: sized for. A dataclass default is one frozen number and cannot depend on
#: ``arenas``, so it is pinned at the single-arena case (today's M1/M2 loop) and
#: the multi-arena launch path re-derives it from the real pad count via
#: :func:`eps_decay_episodes_for` (``agent.train._config_from_args``). A config
#: hand-built as ``TrainConfig(arenas=25)`` therefore keeps the N=1 window —
#: ``agent.train.train_multi_arena`` logs the resulting floor position at startup
#: and warns when it lands early, because that is the silent shape of this bug.
DEFAULT_EPS_DECAY_ARENAS: int = 1


def projected_episodes(
    arenas: int,
    *,
    hours: float = ASSUMED_RUN_HOURS,
    per_arena_transitions_per_s: float = MEASURED_PER_ARENA_TRANSITIONS_PER_S,
    mean_episode_steps: float = MEASURED_MEAN_EPISODE_STEPS,
) -> float:
    """Project how many EPISODES a run of ``arenas`` pads collects.

    The formula, in full::

        transitions = arenas * per_arena_transitions_per_s * 3600 * hours
        episodes    = transitions / mean_episode_steps

    Linear in ``arenas`` because AC16 measured the per-arena rate flat to 0.05%
    from 16 to 25 pads (no knee, no thermal decay over a 600 s confirm).

    Args:
        arenas: Pad count for the run (>= 1).
        hours: Wall-clock length of the run (see :data:`ASSUMED_RUN_HOURS`).
        per_arena_transitions_per_s: Collection rate for ONE pad (see
            :data:`MEASURED_PER_ARENA_TRANSITIONS_PER_S`).
        mean_episode_steps: Mean decisions per episode (see
            :data:`MEASURED_MEAN_EPISODE_STEPS` — measured against the SCRIPTED
            opponent; a different opponent is a different number).

    Returns:
        Projected episode count as a float (a projection; it is not rounded here
        so callers can take fractions of it without compounding rounding).

    Raises:
        ValueError: on a non-positive input, so a zero episode budget can never
            reach the ε schedule and make its denominator meaningless.
    """
    if arenas < 1:
        raise ValueError(f"arenas must be >= 1, got {arenas}")
    if not (hours > 0.0):
        raise ValueError(f"hours must be > 0, got {hours!r}")
    if not (per_arena_transitions_per_s > 0.0):
        raise ValueError(
            "per_arena_transitions_per_s must be > 0, got "
            f"{per_arena_transitions_per_s!r}"
        )
    if not (mean_episode_steps > 0.0):
        raise ValueError(f"mean_episode_steps must be > 0, got {mean_episode_steps!r}")

    transitions = int(arenas) * float(per_arena_transitions_per_s) * 3600.0 * float(hours)
    return transitions / float(mean_episode_steps)


def eps_decay_episodes_for(
    arenas: int,
    *,
    fraction: float = EPS_DECAY_FRACTION_OF_RUN,
    hours: float = ASSUMED_RUN_HOURS,
    per_arena_transitions_per_s: float = MEASURED_PER_ARENA_TRANSITIONS_PER_S,
    mean_episode_steps: float = MEASURED_MEAN_EPISODE_STEPS,
) -> int:
    """Return the ε-decay window (in GLOBAL episodes) for a run of ``arenas`` pads.

    ``round(fraction * projected_episodes(arenas, ...))``, floored at 1 so the
    result always satisfies ``TrainConfig``'s own ``eps_decay_episodes >= 1``.

    This is the ONE place the default window is computed. The dataclass default
    calls it at :data:`DEFAULT_EPS_DECAY_ARENAS`, and
    ``agent.train._config_from_args`` calls it again with the run's real
    ``--arenas`` whenever ``--eps-decay-episodes`` was not given — same function,
    so the two can never drift apart.

    Args:
        arenas: Pad count for the run (>= 1).
        fraction: Share of the projected episodes the decay spans (see
            :data:`EPS_DECAY_FRACTION_OF_RUN`).
        hours / per_arena_transitions_per_s / mean_episode_steps: Forwarded to
            :func:`projected_episodes`.

    Returns:
        The episode count over which ε decays from its start value to
        ``eps_end`` (>= 1).

    Raises:
        ValueError: if ``fraction`` is not in (0, 1], or from
            :func:`projected_episodes` on a bad projection input.
    """
    # `not 0 < x <= 1` rather than two or-ed comparisons so NaN — which fails
    # every ordered comparison — is rejected instead of slipping through as an
    # always-false predicate.
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction!r}")
    episodes = projected_episodes(
        arenas,
        hours=hours,
        per_arena_transitions_per_s=per_arena_transitions_per_s,
        mean_episode_steps=mean_episode_steps,
    )
    return max(1, int(round(fraction * episodes)))


#: The opponent sources ``TrainConfig.opponent`` accepts. ``"dummy"`` is the
#: bridge-served stationary dummy (no ``opp_action`` on the wire at all);
#: ``"scripted"`` is the Python-stepped ``ScriptedBot`` curriculum (T12). A typo
#: here would otherwise select the dummy silently and quietly invalidate a whole
#: overnight retrain, so the set is validated in ``__post_init__``.
_OPPONENT_CHOICES = frozenset({"dummy", "scripted"})

#: The tiers ``TrainConfig.eval_opponent_preset`` accepts — which scripted
#: opponent the PERIODIC EVAL fights (never the training mixture, which drifts
#: as the curriculum gate fires and would make two evals incomparable).
#: ``"mixed"`` alternates EASY/HARD by episode index, ``"easy"`` / ``"hard"``
#: pin one tier. Read only when ``opponent == "scripted"``.
_EVAL_OPPONENT_PRESET_CHOICES = frozenset({"mixed", "easy", "hard"})


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

    #: Number of GLOBAL EPISODES (NOT steps, and NOT per-arena episodes) over
    #: which ε decays linearly from ``eps_start`` to ``eps_end``; it is flat at
    #: ``eps_end`` afterwards. Per-EPISODE on purpose (see module docstring /
    #: agent.seeding): a per-step decay collapses ε far too fast on short
    #: episodes.
    #:
    #: The default is COMPUTED, not written down — see
    #: :func:`eps_decay_episodes_for` for the formula and every assumption behind
    #: it. It is ~15% of the episodes a run of
    #: :data:`DEFAULT_EPS_DECAY_ARENAS` pad(s) is projected to collect. The
    #: multi-arena CLI re-derives it from the real ``--arenas`` (all arenas share
    #: ONE episode counter, so N pads consume this budget N times faster), and
    #: ``--eps-decay-episodes N`` overrides both — THAT is the flag to set when a
    #: run's regime differs from the one :data:`MEASURED_MEAN_EPISODE_STEPS` was
    #: measured under, rather than editing any constant in this file.
    eps_decay_episodes: int = eps_decay_episodes_for(DEFAULT_EPS_DECAY_ARENAS)

    # -- replay buffer sizing --------------------------------------------
    #: TUNE — max stored TRANSITIONS in the prioritized sequence replay. 1e6 for
    #: the multi-arena scale-up: at the measured 121.95 transitions/s aggregate
    #: (AC16, 25 pads) the old 100k held ~14 minutes of experience, so an
    #: overnight run would have been sampling almost exclusively from the last
    #: quarter-hour it collected. 1e6 is ~2.3 hours of that same stream.
    #:
    #: MEMORY (measured on the pinned modules, not estimated — the plan's
    #: "~200 MB at OBS_DIM=23" is wrong by ~11x because it counts only the
    #: transition tuple): per stored transition the buffer holds
    #: ``obs`` 23xf32 (92 B) + ``next_obs`` 23xf32 (92 B) + action i64 (8 B) +
    #: reward f32 (4 B) + done bool (1 B) + **the per-step LSTM hidden snapshot
    #: (2, LSTM_LAYERS=1, LSTM_HIDDEN=256) f32 = 2048 B** == 2245 B, plus 32 B per
    #: CAPACITY SLOT for the sum-tree (2 x f64) and the two i64 leaf maps. That is
    #: 2277 B/transition => **~2.28 GB of array payload at 1e6** (~2.5 GB
    #: resident, measured RSS). The hidden-state term is 90% of it and is the part
    #: that gets forgotten. Use ``--replay-capacity`` to shrink it on a smaller
    #: machine.
    replay_capacity: int = 1_000_000

    #: TUNE — transitions that must be stored before the FIRST gradient step
    #: (warm-up). 25k for the multi-arena path: the old 1000 is ~8 seconds of
    #: fleet output (121.95/s) and — at the MEASURED 285 steps/episode — about 3
    #: and a half episodes, so the first gradients came from one or two correlated
    #: openings. 25k spreads the warm-up over ~88 episodes drawn from all N pads,
    #: which is what makes them decorrelated.
    #: Gating is on ``len(replay) >= min_replay``. TUNE: keep >= a few full
    #: windows; raise for more diverse warm-up.
    min_replay: int = 25_000

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

    #: TUNE — gradient steps between checkpoint hook invocations. 0 disables.
    #: 5k rather than 10k so a night produces twice as many late candidates for
    #: freeze-day selection.
    #:
    #: WHICH PATH READS THIS. ``Trainer._fire_hooks`` (the ``Trainer.train()``
    #: cadence) is DEAD on the multi-arena path — the learner thread calls
    #: ``trainer.learn()`` directly and never ``trainer.train()``. But the value
    #: is NOT dead there: T13 made it the default periodic-save cadence at
    #: ``agent/train.py:2756-2760``, used whenever
    #: ``--checkpoint-every-grad-steps`` is omitted. Passing that flag overrides
    #: this field entirely.
    checkpoint_interval: int = 5_000

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

    #: When True (the default), TIER 1 of the two-tier fault policy is armed: a
    #: collector whose pad's BRIDGE died restarts that one bridge in the
    #: background and reconnects, leaving every other pad running. Only read
    #: when ``arenas > 1``.
    #:
    #: Set False only as a hands-on diagnostic: a dead pad then STAYS dead and
    #: the run keeps going on the rest of the fleet, with nothing to make that
    #: loud. (It does not turn a pad fault into an abort. The one abort in the
    #: system is TIER 2 — the shared Paper JVM dying — and that fires whatever
    #: this flag says.) There is deliberately no survivor floor: a
    #: ``fault_min_live_arenas``-style knob would license training on a
    #: silently shrunken fleet, which the plan forbids.
    fault_relaunch: bool = True

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

    # -- Ape-X per-actor ε (issue #15) ---------------------------------------
    #: When True (the default) each arena acts under its OWN ε rather than the
    #: one shared schedule value: arena ``i`` of ``N`` uses
    #: ``ε_i = ε ** (1 + i/(N-1) * per_actor_eps_alpha)`` (Ape-X, Horgan et al.
    #: 2018). Since ``ε <= 1``, raising it to a larger power SHRINKS it, so
    #: **arena 0 is the most exploratory** (``ε_0 == ε`` exactly) and arena
    #: ``N-1`` is the most exploitative. One fleet then spans exploration and
    #: exploitation simultaneously instead of moving through them together.
    #:
    #: EFFECTIVE ONLY WHEN ``arenas > 1``. At ``arenas == 1`` the exponent is
    #: exactly 1 and the value is the base ε unchanged, so the single-arena
    #: (M1/M2) path is byte-identical whatever this says — and there is no
    #: ``N-1 == 0`` division anywhere, because that branch is taken before the
    #: ratio is formed. ``agent.train.per_actor_eps_enabled`` is the one place
    #: the ``and arenas > 1`` gate is expressed.
    #:
    #: Turn it off with ``--no-per-actor-eps`` — no source edit, no relaunch
    #: cost beyond the run itself. This is the plan's declared schedule cut #2.
    #:
    #: WARM-START INTERPLAY, so nobody "fixes" it on demo day: under
    #: ``warm_start`` the base ε spans 0.25 -> 0.05, so at 25 pads arena 24 acts
    #: at ``0.25 ** 8`` ~= 1.5e-5 — effectively greedy from episode 0. That is the
    #: Ape-X exploit arm doing its job (it is the arm whose episodes show what the
    #: current policy actually does), not a broken schedule.
    per_actor_eps: bool = True

    #: Ape-X's ``α``: how hard the per-arena ε spread fans out. The last arena
    #: acts at ``ε ** (1 + α)``, so 7.0 (the paper's value) spans ε down to ε**8.
    #: Lower it for a tighter fleet; ``0`` is NOT the off switch (it would flatten
    #: every arena onto the same ε, which is what ``per_actor_eps=False`` already
    #: means) and is rejected. Only read when ``per_actor_eps`` and ``arenas > 1``.
    per_actor_eps_alpha: float = 7.0

    # -- opponent + curriculum (T12; demo-day scripted-opponent track) --------
    #: Which opponent the collectors fight. ``"dummy"`` (the default) is the
    #: M1/M2 path: the stationary dummy is served entirely by the bridge / Paper
    #: server and no ``opp_action`` is ever put on the wire, so the wire line is
    #: byte-identical to what it was before the field existed. ``"scripted"``
    #: engages the :class:`~opponents.scripted_bot.ScriptedBot` curriculum —
    #: Python steps an opponent policy per decision and threads its macro through
    #: ``env.step(action, opp_action=...)``. Only the MULTI-ARENA path
    #: (``arenas >= 2``) steps an opponent policy; the single-arena loop refuses
    #: ``"scripted"`` loudly rather than silently ignoring it.
    opponent: str = "dummy"

    #: Probability an episode draws the EASY preset BEFORE the win-rate gate
    #: fires. 0.8 == the plan's 80/20 EASY/HARD starting mixture: mostly a
    #: beatable opponent early, with enough HARD episodes that the agent is never
    #: purely on-distribution for one tier. Only read when
    #: ``opponent == "scripted"``.
    opponent_mix_easy: float = 0.8

    #: Probability an episode draws EASY AFTER the gate fires — the shifted
    #: mixture, 0.2 == 20/80 EASY/HARD. Deliberately NOT 0.0: the curriculum is a
    #: gated MIXTURE, not a one-way promotion, so EASY episodes keep arriving
    #: (they are what keeps the gate's own window fed, and they stop the agent
    #: from overfitting to a single opponent tier).
    opponent_mix_easy_after: float = 0.2

    #: Rolling win rate vs the EASY preset that must be reached before the
    #: mixture shifts to ``opponent_mix_easy_after``. 0.6 == "beats EASY more
    #: often than not, with margin". Raise to demand a stronger agent before
    #: HARD dominates; lower to shift sooner.
    opponent_gate_winrate: float = 0.6

    #: Number of EASY episodes in the rolling gate window. The window must be
    #: FULL before the gate is even evaluated, so this is also the minimum EASY
    #: sample size behind the decision: 50 episodes make a 0.6 win rate mean
    #: something, where a partial window would fire the gate on the first EASY
    #: win (1/1 == 100%).
    opponent_gate_window: int = 50

    #: Optional path to a checkpoint whose weights initialize the run. ``None``
    #: == train from a fresh initialization. The load happens in
    #: ``agent.train.Trainer.__init__``: the checkpoint's weights go into BOTH
    #: the online net and the target net (a target left at a random init would
    #: bootstrap the warm-started online net toward noise), and the replay
    #: buffer is deliberately NOT restored — a warm start reuses the policy, not
    #: the stale off-policy data that produced it.
    warm_start: str | None = None

    #: ε the schedule RESTARTS at when ``warm_start`` is set (read by
    #: ``agent.train.effective_eps_start``; ``eps_start`` governs a fresh run).
    #: 0.25 is the plan's pinned 0.2-0.3 band. THIS IS THE FIELD THAT MAKES A
    #: WARM START WORTH DOING: under the fresh-init ``eps_start=1.0`` the agent
    #: spends the first ``eps_decay_episodes`` episodes acting mostly at random,
    #: which throws away the very behavior the checkpoint was loaded for and
    #: fills the fresh replay with noise. Ignored entirely when ``warm_start``
    #: is ``None``.
    warm_start_eps_start: float = 0.25

    #: Which scripted tier the PERIODIC EVAL fights (one of
    #: :data:`_EVAL_OPPONENT_PRESET_CHOICES`). Read only when ``opponent ==
    #: "scripted"``; the dummy path has no opponent policy to step.
    #:
    #: The eval opponent is deliberately NOT the training curriculum's mixture:
    #: that mixture SHIFTS when the win-rate gate fires, so two evals either side
    #: of the gate would score different opponents and the checkpoint selection
    #: would compare numbers that do not mean the same thing. ``"mixed"`` (the
    #: default) alternates EASY/HARD by episode index — deterministic, identical
    #: at every eval, and it can neither saturate at 100% early (the HARD half)
    #: nor sit pinned at 0 all run (the EASY half), which are the two ways a
    #: single-tier eval stops discriminating between checkpoints.
    eval_opponent_preset: str = "mixed"

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
        if self.collector_queue_max < 0:
            raise ValueError(
                f"collector_queue_max must be >= 0, got {self.collector_queue_max}"
            )
        if self.seed_stride < 1:
            raise ValueError(f"seed_stride must be >= 1, got {self.seed_stride}")
        # `not x > 0` (not `x <= 0`) so NaN is rejected: NaN fails every ordered
        # comparison, so `<= 0` would be False and let it through, and a NaN
        # exponent turns every arena's ε into NaN with nothing to catch it.
        if not self.per_actor_eps_alpha > 0.0:
            raise ValueError(
                "per_actor_eps_alpha must be > 0 (use per_actor_eps=False / "
                f"--no-per-actor-eps to disable the spread), got "
                f"{self.per_actor_eps_alpha!r}"
            )
        if self.opponent not in _OPPONENT_CHOICES:
            raise ValueError(
                f"opponent must be one of {sorted(_OPPONENT_CHOICES)}, got "
                f"{self.opponent!r}"
            )
        # Written as `not 0 <= x <= 1` rather than two or-ed comparisons so NaN —
        # which fails every ordered comparison — is rejected instead of slipping
        # through as an always-false predicate.
        if not 0.0 <= self.opponent_mix_easy <= 1.0:
            raise ValueError(
                f"opponent_mix_easy must be in [0, 1], got {self.opponent_mix_easy!r}"
            )
        if not 0.0 <= self.opponent_mix_easy_after <= 1.0:
            raise ValueError(
                "opponent_mix_easy_after must be in [0, 1], got "
                f"{self.opponent_mix_easy_after!r}"
            )
        if not 0.0 <= self.opponent_gate_winrate <= 1.0:
            raise ValueError(
                "opponent_gate_winrate must be in [0, 1], got "
                f"{self.opponent_gate_winrate!r}"
            )
        if self.opponent_gate_window < 1:
            raise ValueError(
                f"opponent_gate_window must be >= 1, got {self.opponent_gate_window}"
            )
        if self.warm_start is not None and not str(self.warm_start).strip():
            # An empty string is the shape a shell `--warm-start ""` produces; it
            # would read as "warm start requested" everywhere downstream while
            # naming no checkpoint at all. `None` is how "no warm start" is spelled.
            raise ValueError(
                "warm_start must be a non-empty path or None, got "
                f"{self.warm_start!r}"
            )
        # `not 0 <= x <= 1` (not two or-ed comparisons) so NaN is rejected rather
        # than slipping through as an always-false predicate.
        if not 0.0 <= self.warm_start_eps_start <= 1.0:
            raise ValueError(
                "warm_start_eps_start must be in [0, 1], got "
                f"{self.warm_start_eps_start!r}"
            )
        if self.warm_start_eps_start < self.eps_end:
            # The schedule decays FROM the start value TO eps_end; a start below
            # the floor would make ε climb over the run, which is not a schedule
            # anyone means to configure.
            raise ValueError(
                "warm_start_eps_start must be >= eps_end, got "
                f"warm_start_eps_start={self.warm_start_eps_start}, "
                f"eps_end={self.eps_end}"
            )
        if self.eval_opponent_preset not in _EVAL_OPPONENT_PRESET_CHOICES:
            raise ValueError(
                "eval_opponent_preset must be one of "
                f"{sorted(_EVAL_OPPONENT_PRESET_CHOICES)}, got "
                f"{self.eval_opponent_preset!r}"
            )
