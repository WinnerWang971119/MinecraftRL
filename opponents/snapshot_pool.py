"""opponents/snapshot_pool — frozen past-self policies for M4 self-play (T8).

Self-play needs an opponent that is neither the live learner (it would track the
learner's own ``optimizer.step()`` and the match would be a mirror of a moving
target) nor a hand-written script (M3 already exhausted that). It needs *past
selves*: policy versions frozen at a grad step, persisted to disk, and drawn
again later as rivals. This module owns that set — the registry, the on-disk
files, the per-snapshot win statistics, and the learner's Elo.

Three properties are load-bearing, and each one exists because of a specific way
the run breaks without it:

  * **Frozen means frozen.** :meth:`SnapshotPool.add` copies through
    :func:`distributed.weights.clone_state_dict`, so a snapshot never aliases the
    learner's live parameters. Storing ``net.state_dict()`` directly would store
    tensor VIEWS and every "past self" in the pool would silently become the
    present self.

  * **One lock, and disk I/O outside it.** A single :class:`threading.RLock`
    guards the registry, the statistics and the Elo table. Loading a snapshot's
    weights back off disk happens with that lock RELEASED: up to 25 arena threads
    load snapshots at episode boundaries, and holding the lock across a
    ``torch.load`` would serialize the whole fleet behind one thread's disk read.

  * **Bootstrap must not deadlock.** Snapshot 0 stays sampleable while it is the
    only version in the pool — the ``exclude_id`` same-version exclusion switches
    on only once a second distinct version exists. Applying it from the first
    episode would leave every arena with no legal opponent, and the run would
    stall before it ever took a gradient step.

Durability: both the snapshot files (``snap_<id>.pt``) and the index
(``pool.json``) are written through a temp file in the SAME directory, fsynced,
then ``os.replace``d — the pattern of ``agent.train._atomic_torch_save``, which
exists because a kill partway through an in-place write truncates the previous
good file with nothing to fall back on. Duplicated rather than imported: T10
makes ``agent.train`` import THIS module, so importing back would be a cycle.
The index additionally carries a monotone ``index_generation`` stamp and
:meth:`SnapshotPool.persist` refuses to write a payload older than the one it
last wrote: atomicity stops a write from being half-applied, but only the stamp
stops a stale payload from being wholly applied on top of a newer one.

Corruption policy (spec §Error Handling): a missing or unreadable **unpinned**
snapshot is dropped from the pool and resampled, loudly. A missing or unreadable
**PINNED** snapshot is FATAL and raises — the pinned references are the fixed
measuring sticks for ``selfplay/win_rate_vs_ref_<id>``, and quietly substituting
another opponent would corrupt that series instead of reporting the loss. In
neither case does the pool ever fall back to an untrained net.

Torch is imported lazily, inside the methods that touch tensors, so the
dataclasses, the Elo arithmetic and the index round-trip stay importable (and
testable) in an environment without torch.

Sampling (T9): ``pfsp`` (the default) weights snapshot ``i`` by
``f(p_i) + 0.05 / N`` with ``f(p) = p(1 - p)``, then normalizes — mass on the
opponents that are challenging but still winnable, since ``f`` peaks at
``p = 0.5`` and vanishes at both extremes. ``uniform`` keeps the flat
distribution so the two are A/B-comparable. The floor is not cosmetic: at
``p = 0`` or ``p = 1`` the shaping term is exactly zero, and a zero-weight member
is one the learner never meets again — which is how a self-play run forgets how
to beat its oldest ancestors.

Owner: T8 (M4 self-play track); PFSP weighting and the ``uniform | pfsp`` switch
by T9.
"""

from __future__ import annotations

import collections.abc
import dataclasses
import json
import math
import os
import sys
import tempfile
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from agent.contract_config import code_version

__all__ = [
    "DEFAULT_SAMPLING",
    "DRAW_SCORE",
    "ELO_INITIAL",
    "ELO_K",
    "INDEX_FILENAME",
    "INDEX_VERSION",
    "MatchResult",
    "MatchStats",
    "PFSP_FLOOR_MASS",
    "PinnedSnapshotError",
    "SAMPLING_MODES",
    "SNAPSHOT_FILENAME",
    "SnapshotPool",
    "SnapshotPoolError",
    "SnapshotRecord",
    "SnapshotUnavailableError",
    "expected_score",
    "updated_elo",
]


# ---------------------------------------------------------------------------
# Constants (spec §Data Model / §Naming — exact strings)
# ---------------------------------------------------------------------------

#: Elo K-factor. One match moves the learner by at most K rating points.
ELO_K = 24.0

#: The learner's starting rating. Snapshot ratings are frozen at creation to the
#: learner's rating at that moment and never move again.
ELO_INITIAL = 1000.0

#: The score a DRAW contributes, from the learner's side and the snapshot's
#: alike: an episode in which neither fighter died is half a win. Named here
#: rather than left as a literal because TWO places have to agree on it — the
#: driver that WRITES it (``agent.train.SnapshotOpponentDriver.observe_outcome``)
#: and :meth:`SnapshotPool.record_result`, which RECOGNIZES it to count
#: ``selfplay/draw_rate``. Two independent literals could drift apart, and the
#: symptom would be a draw rate reading 0.0 through a night of timeouts.
DRAW_SCORE = 0.5

#: Index file inside the pool directory (``runs/<run>/snapshots/pool.json``).
INDEX_FILENAME = "pool.json"

#: Per-snapshot weight file (``runs/<run>/snapshots/snap_<id>.pt``).
SNAPSHOT_FILENAME = "snap_{snapshot_id}.pt"

#: Bumped only on an incompatible index layout change, so a stale ``pool.json``
#: fails loudly at load instead of half-deserializing into a wrong pool.
INDEX_VERSION = 1

#: The sampling regimes a pool accepts (the ``uniform | pfsp`` switch, spec #9).
#: ``uniform`` exists so the PFSP arm has something to be measured against.
SAMPLING_MODES = frozenset({"uniform", "pfsp"})

#: Default regime. PFSP is the shipped behaviour; ``uniform`` is the fallback in
#: the plan's feature-cut order.
DEFAULT_SAMPLING = "pfsp"

#: Total UNNORMALIZED weight reserved for the PFSP floor, spread evenly over the
#: candidates (``floor = PFSP_FLOOR_MASS / N``, so the N shares sum back to this
#: constant). Its SHARE of the normalized distribution is therefore
#: ``PFSP_FLOOR_MASS / (sum(f) + PFSP_FLOOR_MASS)``, not a flat 5% — see
#: :meth:`SnapshotPool._raw_weights_locked`. It is what keeps a member with
#: ``f(p) = 0`` — a snapshot the learner always beats, or never beats — reachable
#: instead of weighted out of the pool forever.
PFSP_FLOOR_MASS = 0.05

#: Every key the snapshot payload must carry (spec §Data Model). A file missing
#: any of them is not one of ours and is treated as corrupt.
_PAYLOAD_KEYS = ("model", "grad_step", "code_version", "snapshot_id", "elo")

#: Hard ceiling on resample attempts. Each drop strictly shrinks the pool, so
#: the loops provably terminate; the cap only converts a hypothetical unforeseen
#: cycle into a diagnosable error rather than a hung arena thread.
_MAX_SAMPLE_ATTEMPTS = 1000

#: A sink for the loud corruption/eviction messages.
LogFn = Callable[[str], None]


def _default_log(message: str) -> None:
    """Print to stderr, unbuffered, tagged — the house logging shape."""
    print(f"[snapshot_pool] {message}", file=sys.stderr, flush=True)


def _validated_sampling(sampling: str) -> str:
    """Return ``sampling`` if it names a mode in :data:`SAMPLING_MODES`.

    Loud, because the silent alternative is a run that believes it is the PFSP
    arm while sampling uniformly (or the reverse), which makes the AC6
    comparison measure nothing.

    Raises:
        ValueError: ``sampling`` is not a known mode.
    """
    if sampling not in SAMPLING_MODES:
        raise ValueError(
            f"sampling must be one of {sorted(SAMPLING_MODES)}, got {sampling!r}"
        )
    return str(sampling)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class SnapshotPoolError(RuntimeError):
    """Base class for every pool failure, so callers can catch one type."""


class PinnedSnapshotError(SnapshotPoolError):
    """A PINNED snapshot is missing or unreadable — fatal, never recoverable.

    Pinned members are the fixed reference opponents behind
    ``selfplay/win_rate_vs_ref_<id>``. Silently sampling something else in place
    of one would keep the run going while making that series measure a different
    opponent than its name claims, so the pool refuses instead.
    """


class SnapshotUnavailableError(SnapshotPoolError):
    """An UNPINNED snapshot could not be loaded; it has been dropped.

    Raised by the load path after the drop has already been applied and logged.
    The caller's correct response is to sample again — which
    :meth:`SnapshotPool.sample_state_dict` does for you.
    """


# ---------------------------------------------------------------------------
# Elo (spec §Data Model): E_a = 1/(1+10**((R_b-R_a)/400)); R_a += K*(S_a-E_a)
# ---------------------------------------------------------------------------


def expected_score(rating_a: float, rating_b: float) -> float:
    """Return A's expected score against B under the standard Elo curve.

    ``E_a = 1 / (1 + 10 ** ((R_b - R_a) / 400))`` — 0.5 at equal ratings, and
    ~0.76 for a 200-point favourite. Written with the exponent on the OPPONENT's
    advantage so the formula reads exactly as the spec states it.

    Args:
        rating_a: Rating of the player whose expectation is wanted.
        rating_b: Rating of the opponent.

    Returns:
        The expected score in ``(0, 1)``.
    """
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def updated_elo(
    rating_a: float, rating_b: float, score: float, k: float = ELO_K
) -> float:
    """Return A's rating after scoring ``score`` against a rating-``rating_b`` B.

    ``R_a += K * (S_a - E_a)``. Only A's rating moves: in this project A is
    always the learner and B is always a snapshot, whose rating was frozen when
    it was created (a snapshot is a fixed measuring stick, not a competitor with
    a career).

    Args:
        rating_a: The learner's current rating.
        rating_b: The snapshot's frozen rating.
        score: 1.0 win / 0.5 draw / 0.0 loss.
        k: K-factor (default :data:`ELO_K`).

    Returns:
        The learner's new rating.
    """
    return rating_a + k * (score - expected_score(rating_a, rating_b))


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SnapshotRecord:
    """One frozen policy version: its identity, its file, and its rating.

    Frozen (immutable) on purpose. Arena threads hold references to records
    across a whole episode while the learner thread may be adding new snapshots;
    an immutable record cannot change underneath a match in progress, so the
    match is always scored against the snapshot it was actually played against.

    Attributes:
        snapshot_id: Immutable policy-version identity, allocated monotonically
            by :meth:`SnapshotPool.add`. Never reused, even after a drop.
        grad_step: The learner's gradient step when the weights were published.
        path: ``runs/<run>/snapshots/snap_<id>.pt``.
        elo: The learner's rating at creation time, frozen here forever.
        pinned: A fixed reference opponent — never dropped, never evicted, and
            its disappearance is fatal rather than recoverable.
    """

    snapshot_id: int
    grad_step: int
    path: str
    elo: float
    pinned: bool

    def __post_init__(self) -> None:
        if self.snapshot_id < 0:
            raise ValueError(f"snapshot_id must be >= 0, got {self.snapshot_id}")
        if self.grad_step < 0:
            raise ValueError(f"grad_step must be >= 0, got {self.grad_step}")
        if not math.isfinite(self.elo):
            raise ValueError(f"elo must be finite, got {self.elo!r}")


@dataclass(frozen=True)
class MatchResult:
    """The outcome of one learner-vs-snapshot match — the unit T12 rates from.

    ``rated_eligible`` carries the eligibility decision on the record itself
    rather than leaving it to be re-derived at rating time, because the two Elo
    series diverge on exactly this bit: ``elo/learner_online`` takes every match,
    while ``elo/learner_rated`` — the AC7 rising-trend series and the
    checkpoint-selection input — takes only matches where BOTH sides played
    greedily.

    :meth:`__post_init__` therefore refuses a record whose flag disagrees with
    its epsilons, in either direction. A False flag on a greedy match silently
    empties the rated series; a True flag on an exploring match silently poisons
    it with ε-noise. Both are invisible in the data and both invalidate AC7, so
    the inconsistency is rejected where it is constructed. Use
    :meth:`MatchResult.create` and the flag is computed for you.

    Attributes:
        snapshot_id: The snapshot the learner played against.
        learner_epsilon: The learner's exploration rate for this match.
        opponent_epsilon: The snapshot driver's exploration rate (0.02 in
            training, 0.0 in eval cycles).
        score: 1.0 win / 0.5 draw / 0.0 loss, from the LEARNER's perspective.
        rated_eligible: True iff both epsilons are exactly 0.0.
    """

    snapshot_id: int
    learner_epsilon: float
    opponent_epsilon: float
    score: float
    rated_eligible: bool

    def __post_init__(self) -> None:
        for name, epsilon in (
            ("learner_epsilon", self.learner_epsilon),
            ("opponent_epsilon", self.opponent_epsilon),
        ):
            if not math.isfinite(epsilon) or not 0.0 <= epsilon <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {epsilon!r}")
        # Any score in [0, 1] rates correctly under Elo; the project only ever
        # produces {0.0, 0.5, 1.0}, but the bound that matters here is keeping
        # NaN and out-of-range garbage out of the rating table.
        if not math.isfinite(self.score) or not 0.0 <= self.score <= 1.0:
            raise ValueError(f"score must be in [0, 1], got {self.score!r}")
        expected = self.learner_epsilon == 0.0 and self.opponent_epsilon == 0.0
        if bool(self.rated_eligible) is not expected:
            raise ValueError(
                "rated_eligible must be True iff BOTH epsilons are 0.0 "
                f"(learner_epsilon={self.learner_epsilon!r}, "
                f"opponent_epsilon={self.opponent_epsilon!r} => {expected}), "
                f"got rated_eligible={self.rated_eligible!r}. Use "
                "MatchResult.create() to derive it."
            )

    @classmethod
    def create(
        cls,
        snapshot_id: int,
        learner_epsilon: float,
        opponent_epsilon: float,
        score: float,
    ) -> "MatchResult":
        """Build a result with ``rated_eligible`` derived from the epsilons.

        The foolproof constructor — preferred over the raw one everywhere the
        eligibility bit is not already known.

        Args:
            snapshot_id: The snapshot the learner played against.
            learner_epsilon: The learner's exploration rate for this match.
            opponent_epsilon: The snapshot driver's exploration rate.
            score: 1.0 win / 0.5 draw / 0.0 loss, learner's perspective.

        Returns:
            The :class:`MatchResult`.
        """
        return cls(
            snapshot_id=snapshot_id,
            learner_epsilon=learner_epsilon,
            opponent_epsilon=opponent_epsilon,
            score=score,
            rated_eligible=(learner_epsilon == 0.0 and opponent_epsilon == 0.0),
        )


@dataclass
class MatchStats:
    """Head-to-head record against one snapshot: plays and learner win mass.

    Mutable, and mutated only under the pool's lock. ``learner_wins`` is a float
    because a draw contributes 0.5.

    Attributes:
        plays: Matches scored against this snapshot.
        learner_wins: Summed learner score (1.0 win / 0.5 draw / 0.0 loss).
    """

    plays: int = 0
    learner_wins: float = 0.0

    def win_rate(self) -> float:
        """Return the learner's win rate under a Beta(1, 1) prior.

        ``(wins + 1) / (plays + 2)`` — the posterior mean of a uniform prior,
        i.e. "one imagined win and one imagined loss already on the books". The
        +1/+2 is not cosmetic: it makes the rate defined (0.5) at ZERO plays,
        which is what keeps the PFSP weight ``f(p) = p(1 - p)`` finite and
        non-zero for a brand-new snapshot. With the raw ratio, a fresh snapshot
        would be 0/0 (NaN) and, once guarded to 0, would be weighted to zero and
        never sampled — so it could never accumulate the plays that would raise
        its weight. It would be dead on arrival.

        Returns:
            The smoothed win rate, always strictly inside ``(0, 1)``.
        """
        return (self.learner_wins + 1.0) / (float(self.plays) + 2.0)


# ---------------------------------------------------------------------------
# Atomic writers — the pattern of agent.train._atomic_torch_save
# ---------------------------------------------------------------------------


def _atomic_write(path: str, write_bytes: Callable[[Any], None]) -> None:
    """Write ``path`` through a temp file so it can never be seen half-written.

    Serialize into a fresh temp file in the SAME DIRECTORY, fsync it, then
    ``os.replace`` it onto ``path``. Same directory is load-bearing: ``os.replace``
    is atomic only within one filesystem, so a temp in ``/tmp`` would turn the
    rename into a cross-device copy — exactly the in-place truncation this
    avoids. Until the replace lands, the destination is byte-for-byte the
    previous write; after it, wholly the new one.

    This project has already lost a checkpoint to a kill mid-write; the pool
    index is worse, because losing ``pool.json`` orphans every snapshot file.

    Directories are NOT created here — :class:`SnapshotPool` creates its own
    directory once, at construction.

    Args:
        path: Destination path, replaced atomically on success.
        write_bytes: Callback handed the open binary temp handle.

    Raises:
        BaseException: Whatever ``write_bytes`` raises, unchanged, after
            unlinking the temp file.
    """
    # dirname(abspath(...)) so a bare filename still resolves to a real
    # directory — dirname() alone returns "" there and mkstemp would reject it.
    directory = os.path.dirname(os.path.abspath(path))
    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.", suffix=".tmp", dir=directory
    )

    def _discard_temp() -> None:
        try:
            os.unlink(tmp_path)
        except OSError:
            # Best effort: already gone, or the directory turned unwritable.
            # Neither is worth masking the original failure.
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
            write_bytes(handle)
            handle.flush()
            # Without the fsync the bytes may still sit in the page cache when
            # the rename commits, so a power loss could publish an empty file
            # over a good one.
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        # BaseException, not Exception: a KeyboardInterrupt at 4am must not be
        # the one path that leaves a temp behind.
        _discard_temp()
        raise


def _atomic_torch_save(payload: Mapping[str, Any], path: str) -> None:
    """``torch.save`` the payload onto ``path`` without ever truncating it."""
    import torch  # lazy: keeps this module importable without torch

    _atomic_write(path, lambda handle: torch.save(dict(payload), handle))


def _atomic_write_json(payload: Mapping[str, Any], path: str) -> None:
    """Serialize ``payload`` as JSON onto ``path`` without ever truncating it."""
    # Serialized BEFORE the temp file is opened: an unserializable value then
    # fails without having created (and possibly leaked) a temp.
    blob = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    _atomic_write(path, lambda handle: handle.write(blob.encode("utf-8")))


# ---------------------------------------------------------------------------
# The pool
# ---------------------------------------------------------------------------


class SnapshotPool:
    """Thread-safe registry of frozen past-self policies, persisted to a directory.

    Writers: the learner thread (:meth:`add`, on the snapshot cadence). Readers:
    every arena thread (:meth:`sample`, :meth:`load_state_dict`,
    :meth:`record_result`, at episode boundaries).

    **Locking.** ONE :class:`threading.RLock` guards the registry, the statistics
    and the Elo table — the three pieces that must agree with each other. It is
    reentrant so a public method may call another without deadlocking (T12 and
    T18 both add methods to this class). Three rules keep it cheap and
    deadlock-free:

      * No disk I/O under the lock. ``torch.load`` of a snapshot happens with the
        lock released; 25 arena threads reloading at episode boundaries must not
        serialize behind one another. Pinned by
        ``test_disk_io_never_runs_under_the_state_lock``, which fails if a future
        edit moves a read or a write inside ``with self._lock:``.
      * No caller-supplied callback under the lock either. The ``log`` sink is
        the caller's code and may take a lock of its own; invoking it while
        holding ``_lock`` turns "another thread waits for the pool" into a
        deadlock. :meth:`_retire` and :meth:`_normalized_weights_locked`
        therefore both hand their message OUT and log after the release.
      * ``_io_lock`` serializes index writes so two persists cannot interleave
        into one file. It is never taken while holding ``_lock``, and
        :meth:`persist` never acquires ``_lock`` at all — it is handed the
        payload — so the two locks are ORDER-INDEPENDENT and an ABBA cycle
        between them cannot be written, not merely avoided. Handing the payload
        in gives up the freshness that building it under the write lock used to
        guarantee, so every payload carries the monotone ``index_generation``
        stamp instead and :meth:`persist` drops a write older than its last one.
        Both properties, one lock each, and neither thread ever holds two.

    **Sampling.** ``sampling="pfsp"`` (default) weights each candidate by
    ``p(1 - p) + PFSP_FLOOR_MASS / N``; ``sampling="uniform"`` gives every
    candidate the same weight. See :meth:`_raw_weights_locked`.

    **Eviction.** None in this run. The only way a member leaves the pool is the
    corruption drop, and only if it is unpinned.

    Attributes:
        directory: The pool directory (``runs/<run>/snapshots``), created if it
            does not exist.
    """

    def __init__(
        self,
        directory: str,
        *,
        elo_k: float = ELO_K,
        elo_initial: float = ELO_INITIAL,
        sampling: str = DEFAULT_SAMPLING,
        log: Optional[LogFn] = None,
    ) -> None:
        """Create an empty pool rooted at ``directory``.

        Args:
            directory: Where ``snap_<id>.pt`` and ``pool.json`` live. Created
                (with parents) if missing — the pool owns this directory.
            elo_k: Elo K-factor (default :data:`ELO_K` = 24).
            elo_initial: The learner's starting rating (default 1000).
            sampling: ``"pfsp"`` (default) or ``"uniform"`` — the opponent
                weighting regime, fixed for the life of the pool so a run cannot
                change arm halfway through its own A/B.
            log: Sink for the loud corruption/drop messages; defaults to stderr.

        Raises:
            ValueError: ``elo_k`` is negative or non-finite, ``elo_initial`` is
                non-finite, or ``sampling`` is not a known mode.
            OSError: The directory could not be created.
        """
        if not math.isfinite(elo_k) or elo_k < 0.0:
            raise ValueError(f"elo_k must be finite and >= 0, got {elo_k!r}")
        if not math.isfinite(elo_initial):
            raise ValueError(f"elo_initial must be finite, got {elo_initial!r}")

        self.directory = directory
        self._elo_k = float(elo_k)
        self._elo_initial = float(elo_initial)
        self._sampling = _validated_sampling(sampling)
        self._log: LogFn = log if log is not None else _default_log

        # One lock over registry + stats + Elo (see the class docstring).
        self._lock = threading.RLock()
        # Serializes index writes only; never guards state, never taken while
        # holding `_lock`.
        self._io_lock = threading.Lock()

        self._records: Dict[int, SnapshotRecord] = {}
        self._stats: Dict[int, MatchStats] = {}
        # Frozen ratings of dropped snapshots, so a match that was already in
        # flight when its opponent got dropped can still be rated correctly
        # instead of being discarded or rated against a guessed number.
        self._retired_elo: Dict[int, float] = {}
        self._next_id = 0
        # Monotone version stamp over everything `index_payload` serializes —
        # the registry, the id counter, the statistics and both Elo series.
        # Bumped under `_lock` by every mutation that touches any of them, so
        # equal generations mean identical payloads and a smaller generation
        # means a strictly older payload. Guarded by `_lock`.
        self._generation = 0
        # Generation of the last payload `persist` actually wrote. Guarded by
        # `_io_lock`, alongside the write it describes. -1 means this pool has
        # written nothing yet, so its first persist always lands.
        self._persisted_generation = -1
        self._learner_elo_online = self._elo_initial
        self._learner_elo_rated = self._elo_initial
        # Match accounting behind ``selfplay/draw_rate`` and the "is the rated
        # series about to come out empty?" tell. Pool-wide rather than folded
        # into :class:`MatchStats`, because a draw is NOT recoverable from that
        # pair of numbers: ``learner_wins`` sums 1.0/0.5/0.0 into one float, so
        # two draws and one win plus one loss are both ``plays=2, wins=1.0``.
        # All three are guarded by `_lock` and carried in the index.
        self._matches_scored = 0
        self._draws_scored = 0
        self._rated_matches = 0

        os.makedirs(self.directory, exist_ok=True)

    # -- introspection ----------------------------------------------------

    @property
    def index_path(self) -> str:
        """Path of the ``pool.json`` index."""
        return os.path.join(self.directory, INDEX_FILENAME)

    @property
    def elo_k(self) -> float:
        """The Elo K-factor in force."""
        return self._elo_k

    @property
    def elo_initial(self) -> float:
        """The learner's starting rating."""
        return self._elo_initial

    @property
    def sampling(self) -> str:
        """The weighting regime in force: ``"pfsp"`` or ``"uniform"``.

        Set at construction and never mutated, so it needs no lock.
        """
        return self._sampling

    @property
    def learner_elo_online(self) -> float:
        """Learner rating from EVERY recorded match (``elo/learner_online``).

        Dense and noisy — both sides may be exploring. This is the series PFSP
        reads; it is not the AC7 trend series.
        """
        with self._lock:
            return self._learner_elo_online

    @property
    def learner_elo_rated(self) -> float:
        """Learner rating from greedy-only matches (``elo/learner_rated``).

        Sparse and clean: updated only by results whose ``rated_eligible`` is
        True, i.e. eval-cycle matches where BOTH sides sit at ε=0. This is the
        AC7 rising-trend series and the checkpoint-selection input.
        """
        with self._lock:
            return self._learner_elo_rated

    @property
    def matches_scored(self) -> int:
        """How many matches have been through :meth:`record_result`.

        The denominator of :attr:`draw_rate`, and on its own the answer to "did
        the self-play path score ANYTHING tonight?" — zero here after hours of
        episodes means no outcome ever reached the pool, which no other metric
        reports.
        """
        with self._lock:
            return self._matches_scored

    @property
    def draws_scored(self) -> int:
        """How many of those matches scored exactly :data:`DRAW_SCORE`."""
        with self._lock:
            return self._draws_scored

    @property
    def rated_matches(self) -> int:
        """How many were ``rated_eligible`` — what builds ``elo/learner_rated``.

        The number that separates "the learner stopped improving" from "nothing
        ever fed the rated series". Both look like a flat line at
        :attr:`elo_initial`, and only this counter tells them apart: it is zero
        exactly when every match so far was played with at least one side
        exploring, which at the shipped ``opponent_epsilon`` default is every
        TRAINING match.
        """
        with self._lock:
            return self._rated_matches

    @property
    def draw_rate(self) -> float:
        """Fraction of scored matches that were draws; ``0.0`` before any match.

        Numerator and denominator are read under ONE lock acquisition, so they
        describe the same instant. Computed from two separate property reads
        instead, the value could exceed 1.0 while arena threads score
        concurrently.
        """
        with self._lock:
            if self._matches_scored <= 0:
                return 0.0
            return self._draws_scored / float(self._matches_scored)

    def __len__(self) -> int:
        """Number of LIVE snapshots (dropped ones do not count)."""
        with self._lock:
            return len(self._records)

    def records(self) -> List[SnapshotRecord]:
        """Return every live snapshot, ordered by ``snapshot_id`` ascending.

        The records are immutable, so this is a safe read from any thread.
        """
        with self._lock:
            return self._sorted_records_locked()

    def get(self, snapshot_id: int) -> Optional[SnapshotRecord]:
        """Return the live record for ``snapshot_id``, or ``None`` if not held."""
        with self._lock:
            return self._records.get(snapshot_id)

    def stats_for(self, snapshot_id: int) -> MatchStats:
        """Return a COPY of the head-to-head statistics against ``snapshot_id``.

        A copy, so a caller cannot mutate pool state by accident. Unknown ids
        return a zeroed :class:`MatchStats` (whose ``win_rate()`` is the Beta
        prior's 0.5) rather than raising — "no matches played yet" is a valid
        state, not an error.
        """
        with self._lock:
            stats = self._stats.get(snapshot_id)
            return dataclasses.replace(stats) if stats is not None else MatchStats()

    def pinned_references(self) -> List[SnapshotRecord]:
        """Return the pinned reference snapshots, ordered by id ascending.

        However many exist — one at bootstrap (snapshot 0), up to three once T18
        has promoted the references at ``reference_promote_grad_steps``. Not
        truncated to three: a fourth pinned member would be a wiring bug, and
        hiding it here would make ``selfplay/win_rate_vs_ref_<id>`` quietly miss
        a series.
        """
        with self._lock:
            return [rec for rec in self._sorted_records_locked() if rec.pinned]

    # -- mutation ---------------------------------------------------------

    def add(
        self,
        state_dict: Mapping[str, Any],
        grad_step: int,
        elo: float,
        pinned: bool = False,
    ) -> SnapshotRecord:
        """Freeze ``state_dict`` as a new snapshot, persist it, and register it.

        The weights are cloned through
        :func:`distributed.weights.clone_state_dict` BEFORE anything else, so the
        snapshot owns its storage and cannot track the learner's next
        ``optimizer.step()``. The id is allocated under the lock, the
        ``torch.save`` runs with the lock RELEASED (arena threads keep sampling
        throughout), and the record is registered under the lock afterwards — so
        a snapshot becomes visible to samplers only once its file is completely
        on disk.

        The index is rewritten (atomically) before returning: a snapshot file
        with no index entry is an orphan that a restart would never find.

        Args:
            state_dict: The weights to freeze. Normally the PUBLISHED weights
                (``WeightStore.latest()``), never ``trainer.online.state_dict()``
                — see T18.
            grad_step: The learner's gradient step for those weights.
            elo: The learner's rating right now; frozen onto the snapshot
                forever. Callers pass :attr:`learner_elo_online`.
            pinned: Mark as a fixed reference opponent (never dropped; its
                disappearance is fatal).

        Returns:
            The new :class:`SnapshotRecord`.

        Raises:
            ValueError: ``state_dict`` is empty, ``grad_step`` is negative, or
                ``elo`` is not finite.
            Exception: Anything ``torch.save`` raises; the destination file and
                the index are then left untouched and the id is simply skipped.
        """
        from distributed.weights import clone_state_dict  # lazy: needs torch

        if not state_dict:
            raise ValueError(
                "refusing to snapshot an empty state_dict — an empty snapshot "
                "would load as an untrained net and silently become the opponent"
            )
        grad_step = int(grad_step)
        if grad_step < 0:
            raise ValueError(f"grad_step must be >= 0, got {grad_step}")
        elo = float(elo)
        if not math.isfinite(elo):
            raise ValueError(f"elo must be finite, got {elo!r}")

        cloned = clone_state_dict(dict(state_dict))

        with self._lock:
            snapshot_id = self._next_id
            self._next_id += 1
            # `next_snapshot_id` is index state, so allocating an id ages the
            # payload just as registering the record does.
            self._generation += 1

        path = os.path.join(
            self.directory, SNAPSHOT_FILENAME.format(snapshot_id=snapshot_id)
        )
        payload = {
            "model": cloned,
            "grad_step": grad_step,
            "code_version": code_version(),
            "snapshot_id": snapshot_id,
            "elo": elo,
        }
        # OUTSIDE the lock: this is the expensive part, and samplers must not
        # block on it.
        _atomic_torch_save(payload, path)

        record = SnapshotRecord(
            snapshot_id=snapshot_id,
            grad_step=grad_step,
            path=path,
            elo=elo,
            pinned=bool(pinned),
        )
        with self._lock:
            self._records[snapshot_id] = record
            self._stats.setdefault(snapshot_id, MatchStats())
            self._generation += 1

        # Read the state under `_lock` (inside index_payload), then write it
        # under `_io_lock` only — `persist` never touches `_lock`, so the order
        # in which a caller happens to hold the two can never matter. Built
        # immediately before the write, and stamped, so neither this payload nor
        # a concurrent one can land on top of a newer index.
        self.persist(self.index_payload())
        return record

    def record_result(self, result: MatchResult) -> None:
        """Score one finished match into the statistics and the Elo table.

        Updates, all under the one lock:

          * the head-to-head :class:`MatchStats` (``plays += 1``,
            ``learner_wins += score``) — the input to the PFSP weighting;
          * ``elo/learner_online``, from EVERY result;
          * ``elo/learner_rated``, only when ``result.rated_eligible``;
          * the pool-wide match counters behind ``selfplay/draw_rate`` and
            :attr:`rated_matches`.

        Only the learner's rating moves. The opponent's rating is the snapshot's
        frozen ``elo``; if the snapshot was dropped while the match was in
        flight, its remembered frozen rating is used, so a match is never rated
        against a guessed number.

        Deliberately does NOT touch disk: this runs once per episode on every
        arena thread, and an ``fsync`` per episode would put the whole fleet
        behind the disk. The index is rewritten on every :meth:`add` (the
        snapshot cadence), and callers may :meth:`persist` on their own cadence.

        Args:
            result: The finished match.

        Raises:
            KeyError: ``result.snapshot_id`` was never in this pool — a wiring
                bug, loud on purpose.
        """
        snapshot_id = int(result.snapshot_id)
        score = float(result.score)

        with self._lock:
            record = self._records.get(snapshot_id)
            if record is not None:
                opponent_elo = record.elo
            elif snapshot_id in self._retired_elo:
                opponent_elo = self._retired_elo[snapshot_id]
            else:
                raise KeyError(
                    f"record_result for unknown snapshot_id {snapshot_id}; "
                    f"pool holds {sorted(self._records)} "
                    f"(retired: {sorted(self._retired_elo)})"
                )

            stats = self._stats.setdefault(snapshot_id, MatchStats())
            stats.plays += 1
            stats.learner_wins += score

            self._matches_scored += 1
            # Exact equality against the shared literal, deliberately. The score
            # alphabet this project produces is {0.0, DRAW_SCORE, 1.0} and the
            # only producer writes DRAW_SCORE itself, so there is no arithmetic
            # here to round; a tolerance would start counting near-misses that
            # cannot occur as draws.
            if score == DRAW_SCORE:
                self._draws_scored += 1

            self._learner_elo_online = updated_elo(
                self._learner_elo_online, opponent_elo, score, self._elo_k
            )
            if result.rated_eligible:
                self._rated_matches += 1
                self._learner_elo_rated = updated_elo(
                    self._learner_elo_rated, opponent_elo, score, self._elo_k
                )
            # The statistics and both Elo series are in the index too, so a
            # result ages the payload: one built before this line must never be
            # written on top of one built after it.
            self._generation += 1

    # -- sampling ---------------------------------------------------------

    def sample(self, rng: Any, exclude_id: Optional[int] = None) -> SnapshotRecord:
        """Draw one opponent snapshot, weighted by :meth:`pfsp_weights`.

        **The bootstrap rule.** ``exclude_id`` (the learner's own current
        version, so it does not train against a copy of itself) is honoured ONLY
        once the pool holds two or more distinct versions. While snapshot 0 is
        the only member it stays sampleable, because the alternative is that
        every arena finds no legal opponent on episode 1 and the run deadlocks
        before taking a single gradient step.

        The chosen record's file is checked for existence with the lock
        RELEASED. A missing file gets the corruption policy: fatal if pinned,
        dropped-and-resampled if not. Whole-file corruption (a truncated write)
        is caught later, by :meth:`load_payload`.

        Args:
            rng: Any object with a ``random()`` method returning a float in
                ``[0, 1)`` — ``random.Random`` and ``numpy.random.Generator``
                both qualify. Passed in so each arena owns its own stream.
            exclude_id: Snapshot id to avoid (the learner's current version).

        Returns:
            The sampled :class:`SnapshotRecord`, whose file existed a moment ago.

        Raises:
            TypeError: ``rng`` has no ``random()`` method.
            PinnedSnapshotError: A pinned snapshot's file is missing.
            SnapshotPoolError: The pool is empty, or every candidate was dropped.
        """
        if not hasattr(rng, "random"):
            raise TypeError(
                "rng must expose random() -> float in [0, 1) "
                "(random.Random or numpy.random.Generator); "
                f"got {type(rng).__name__}"
            )

        for _attempt in range(_MAX_SAMPLE_ATTEMPTS):
            with self._lock:
                candidates = self._eligible_locked(exclude_id)
                if not candidates:
                    raise SnapshotPoolError(
                        "snapshot pool has no sampleable member "
                        f"(exclude_id={exclude_id!r}); the run cannot pick an "
                        "opponent. Seed snapshot 0 from the warm start before "
                        "collecting."
                    )
                weights, warning = self._normalized_weights_locked(candidates)
            # OUTSIDE the lock: `_log` is caller-supplied and may take a lock of
            # its own, and a callback that blocks while we hold `_lock` wedges
            # every other arena thread.
            if warning is not None:
                self._log(warning)

            record = _weighted_choice(rng, candidates, weights)

            # Disk touch, outside the lock.
            if os.path.isfile(record.path):
                return record
            # Fatal for a pinned member; for an unpinned one this drops it (so
            # the next iteration draws from a strictly smaller pool) and returns.
            self._retire(record, "file is missing")

        raise SnapshotPoolError(
            f"sample() gave up after {_MAX_SAMPLE_ATTEMPTS} attempts; the pool "
            "is churning members faster than it can draw one"
        )

    def sample_state_dict(
        self, rng: Any, exclude_id: Optional[int] = None
    ) -> Tuple[SnapshotRecord, Dict[str, Any]]:
        """Sample an opponent AND load its weights, resampling past dropped ones.

        The method the opponent driver should use: it closes the gap between
        :meth:`sample` (which only proves the file exists) and
        :meth:`load_state_dict` (which proves it actually reads back), retrying
        whenever an unpinned member turns out to be corrupt. A pinned member's
        corruption still propagates, fatally.

        The load runs outside the pool lock.

        Args:
            rng: Per-arena RNG exposing ``random()``.
            exclude_id: Snapshot id to avoid, subject to the bootstrap rule.

        Returns:
            ``(record, state_dict)`` — the snapshot and its CPU weights.

        Raises:
            PinnedSnapshotError: A pinned snapshot is missing or unreadable.
            SnapshotPoolError: No sampleable member remains.
        """
        for _attempt in range(_MAX_SAMPLE_ATTEMPTS):
            record = self.sample(rng, exclude_id=exclude_id)
            try:
                return record, self.load_state_dict(record)
            except SnapshotUnavailableError:
                # Unpinned and unreadable: already dropped and logged by the
                # loader. Draw again from the (now smaller) pool.
                continue

        raise SnapshotPoolError(
            f"sample_state_dict() gave up after {_MAX_SAMPLE_ATTEMPTS} attempts"
        )

    def pfsp_weights(self) -> Dict[int, float]:
        """Return normalized sampling probabilities over every live snapshot.

        Under ``sampling="pfsp"`` the weight of snapshot ``i`` is
        ``f(p_i) + PFSP_FLOOR_MASS / N`` with ``f(p) = p(1 - p)`` and ``p_i`` the
        Beta(1, 1)-smoothed head-to-head win rate, normalized over the pool;
        under ``sampling="uniform"`` every member gets ``1 / N``. See
        :meth:`_raw_weights_locked` for why each term is there.

        T17's launch canary reads this as-is and refuses to launch on invalid or
        NaN PFSP probabilities: ``scripts/canary_selfplay.sh`` calls it on the
        run's pool and its ``PFSP_INVALID`` condition blocks the launch on a
        non-finite, negative, non-normalized or incomplete vector. These numbers
        are checked at launch today, so a weakening here is caught there.

        Returns:
            ``{snapshot_id: probability}`` summing to 1.0, or ``{}`` when the
            pool is empty. Never contains a NaN.
        """
        with self._lock:
            weights, warning = self._normalized_weights_locked(
                self._sorted_records_locked()
            )
        # Logged after the release, for the reason given in :meth:`sample`.
        if warning is not None:
            self._log(warning)
        return weights

    # -- loading (all disk I/O happens OUTSIDE the lock) -------------------

    def load_payload(self, record: SnapshotRecord) -> Dict[str, Any]:
        """Read a snapshot's full payload off disk, applying the corruption policy.

        Runs with the pool lock RELEASED — this is the call 25 arena threads make
        at episode boundaries, and holding the lock across it would serialize the
        fleet behind one thread's disk read.

        Anything that goes wrong (missing file, truncated write, wrong keys, an
        id that does not match the record) is treated as corruption: fatal for a
        pinned snapshot, a loud drop for an unpinned one. The pool never returns
        a partial payload and never substitutes an untrained net.

        Args:
            record: The snapshot to read, from :meth:`sample` or :meth:`records`.

        Returns:
            The payload dict: ``model``, ``grad_step``, ``code_version``,
            ``snapshot_id``, ``elo``.

        Raises:
            PinnedSnapshotError: The snapshot is pinned and unreadable — fatal.
            SnapshotUnavailableError: The snapshot was unpinned and has now been
                dropped; sample again.
        """
        import torch  # lazy: keeps this module importable without torch

        try:
            # weights_only=True: a snapshot file is data, never a vector for
            # arbitrary code at load time. Our payload is tensors plus plain
            # scalars, all of which weights_only permits.
            payload = torch.load(record.path, map_location="cpu", weights_only=True)
        except Exception as exc:  # noqa: BLE001 - every read failure is corruption
            return self._unreadable(record, f"torch.load failed: {exc!r}")

        if not isinstance(payload, dict):
            return self._unreadable(
                record, f"payload is a {type(payload).__name__}, expected a dict"
            )
        missing = [key for key in _PAYLOAD_KEYS if key not in payload]
        if missing:
            return self._unreadable(record, f"payload is missing keys {missing}")
        model = payload["model"]
        if not isinstance(model, dict) or not model:
            return self._unreadable(
                record, "payload['model'] is not a non-empty state_dict"
            )
        stored_id = payload["snapshot_id"]
        if int(stored_id) != record.snapshot_id:
            # The file on disk belongs to a different policy version than the
            # index says. Loading it would drive an opponent under the wrong
            # identity and mis-attribute every match played against it.
            return self._unreadable(
                record,
                f"payload snapshot_id {stored_id!r} != record "
                f"{record.snapshot_id}",
            )
        return payload

    def load_state_dict(self, record: SnapshotRecord) -> Dict[str, Any]:
        """Return just the frozen weights of ``record`` (see :meth:`load_payload`)."""
        return self.load_payload(record)["model"]

    # -- persistence ------------------------------------------------------

    def index_payload(self) -> Dict[str, Any]:
        """Return the JSON-serializable index: the exact content of ``pool.json``.

        Also the round-trip witness — two pools with equal index payloads hold
        the same snapshots, the same statistics, the same Elo and the same match
        accounting.

        Every payload is stamped with ``index_generation``: the pool's monotone
        version counter as of the moment it was built. That stamp is what lets
        :meth:`persist` order two payloads and refuse the older one, so a
        payload built long ago cannot land on top of a newer index.
        :meth:`load` restores it, so the counter continues across a restart
        instead of starting again at zero underneath a much later ``pool.json``.

        Snapshot paths are stored as bare FILENAMES, not full paths, so a run
        directory stays relocatable (copy ``runs/<run>/`` to another box and the
        pool still loads). :meth:`load` rejoins them onto its own directory.
        """
        with self._lock:
            return {
                "index_version": INDEX_VERSION,
                "index_generation": self._generation,
                "elo_k": self._elo_k,
                "elo_initial": self._elo_initial,
                "learner_elo_online": self._learner_elo_online,
                "learner_elo_rated": self._learner_elo_rated,
                # Carried so a resumed run's `selfplay/draw_rate` describes the
                # whole run rather than restarting from the restart.
                "matches_scored": self._matches_scored,
                "draws_scored": self._draws_scored,
                "rated_matches": self._rated_matches,
                "next_snapshot_id": self._next_id,
                "snapshots": [
                    {
                        "snapshot_id": rec.snapshot_id,
                        "grad_step": rec.grad_step,
                        "filename": os.path.basename(rec.path),
                        "elo": rec.elo,
                        "pinned": rec.pinned,
                        "plays": self._stats_locked(rec.snapshot_id).plays,
                        "learner_wins": self._stats_locked(
                            rec.snapshot_id
                        ).learner_wins,
                    }
                    for rec in self._sorted_records_locked()
                ],
                "retired": [
                    {
                        "snapshot_id": snapshot_id,
                        "elo": elo,
                        "plays": self._stats_locked(snapshot_id).plays,
                        "learner_wins": self._stats_locked(snapshot_id).learner_wins,
                    }
                    for snapshot_id, elo in sorted(self._retired_elo.items())
                ],
            }

    def persist(self, payload: Mapping[str, Any]) -> None:
        """Write ``payload`` (an :meth:`index_payload` result) onto ``pool.json``.

        The payload is an ARGUMENT rather than something this method builds, and
        that is the whole point: ``persist`` never acquires ``_lock``. A thread
        inside ``persist`` holds only ``_io_lock`` and waits for nothing, so the
        two locks are order-independent and no ABBA cycle between them can be
        written — whatever a future caller happens to hold. The alternative
        (``persist`` taking ``_io_lock`` and then ``_lock`` internally) is
        correct only for as long as every caller remembers the rule, and that is
        a docstring, not an invariant.

        ``_io_lock`` still serializes the writes themselves, so two persists can
        never interleave into one file. What handing the payload in DID cost is
        freshness: building the index inside the write lock used to guarantee
        that the last writer held the newest state. The ``index_generation``
        stamp buys that back without reintroducing a lock order — every payload
        carries the pool's monotone version counter, ``persist`` remembers the
        generation it last wrote (under ``_io_lock``, with the write), and a
        payload older than that is DROPPED and logged instead of written.

        The stamp, not :meth:`load`'s floor, is what protects the id counter.
        That floor is computed from the ids in whatever index is on disk, so
        against a stale index it reports the stale ``next_snapshot_id`` back and
        protects nothing: a fresh index at ``next_snapshot_id: 2`` overwritten by
        a stale one at ``1`` hands id 1 out a second time after a restart, and
        the second policy version silently ``os.replace``s the first's
        ``snap_1.pt``, so two policy versions answer to one id.

        MEASURED, not assumed: the re-issued id does NOT inherit the first's
        statistics. ``load`` only re-issues an id the index does not contain,
        and an index missing the id is missing its stats and Elo too — the
        record comes back with zeroed ``MatchStats``, the new Elo, and
        ``pinned=False``. That is worse, not better: nothing INSIDE the pool
        looks anomalous, while everything outside it that was already keyed on
        that id (a previously logged ``selfplay/win_rate_vs_ref_<id>``, T12's
        rating history) now silently describes a different policy, and a pinned
        reference quietly stops being one. Refusing the backwards write is what
        makes all of that unreachable.

        A skipped write is NOT an error and callers need not handle it: the
        pool's state is already on disk in a newer form, which is what the caller
        wanted. Callers that follow the convention
        (``pool.persist(pool.index_payload())``) will normally never cause one.

        Args:
            payload: The index dict from :meth:`index_payload`.

        Raises:
            TypeError: ``payload`` is not a mapping — usually
                ``persist(self.index_payload)`` with the call parentheses lost —
                or it carries no usable ``index_generation``, which means it did
                not come from :meth:`index_payload` and cannot be ordered.
            OSError: The index could not be written — the previous index then
                survives intact, because the write is atomic, and so does the
                watermark, because it only advances after the write lands.
        """
        if not isinstance(payload, collections.abc.Mapping):
            raise TypeError(
                "persist(payload) takes the index dict from index_payload(); "
                f"got {type(payload).__name__}"
            )
        try:
            generation = int(payload["index_generation"])
        except (KeyError, TypeError, ValueError) as exc:
            # An unorderable payload is precisely the hole this guard closes:
            # writing one would put an index of unknown age onto pool.json.
            raise TypeError(
                "persist(payload) takes the index dict from index_payload(), "
                "which stamps 'index_generation'; got a mapping without a "
                f"usable one ({exc!r})"
            ) from exc

        stale: Optional[str] = None
        with self._io_lock:
            if generation < self._persisted_generation:
                stale = (
                    f"SKIPPED a stale index write: payload generation "
                    f"{generation} is older than generation "
                    f"{self._persisted_generation}, already on "
                    f"{self.index_path!r}. Nothing was written and the newer "
                    "index stands — writing backwards here would let a later "
                    "snapshot reuse a live id. Build the payload immediately "
                    "before persisting: pool.persist(pool.index_payload())."
                )
            else:
                _atomic_write_json(payload, self.index_path)
                # Advanced only once the replace has landed: a failed write
                # leaves the previous index in force, so the watermark that
                # describes it must stay in force too.
                self._persisted_generation = generation

        # OUTSIDE `_io_lock`: `_log` is caller-supplied code, and the rule that
        # keeps it off `_lock` (see :meth:`sample`) applies to `_io_lock` too.
        if stale is not None:
            self._log(stale)

    @classmethod
    def load(
        cls,
        directory: str,
        *,
        sampling: str,
        log: Optional[LogFn] = None,
    ) -> "SnapshotPool":
        """Rebuild a pool from ``directory``/``pool.json``.

        Restores the registry, the per-snapshot statistics, both Elo series,
        the match counters and the id counter, so a restarted run continues
        numbering where it stopped (ids are never reused, even across restarts).

        PINNED snapshots are verified to exist right here, so a run that lost a
        reference file fails at startup rather than a thousand episodes later,
        mid-training. Unpinned members are not verified: their failure mode is
        recoverable and is handled at sample time.

        Args:
            directory: The pool directory written by :meth:`persist`.
            sampling: Weighting regime for the restored pool. REQUIRED, and
                deliberately without a default: the regime is a run-config
                choice (it mirrors T11a's ``snapshot_sampling`` field), so a
                resumed run passes the value its own config carries, and a
                resume that omitted it would restart the ``uniform`` arm of the
                AC6 comparison as the ``pfsp`` arm, invisibly. A default would
                leave that exact failure expressible by omission — the one thing
                this parameter exists to prevent — so there is none. Not stored
                in the index for the same reason: ``pool.json`` describes the
                pool, not the run that resumes it.
            log: Sink for the loud corruption/drop messages.

        Returns:
            The restored pool.

        Raises:
            FileNotFoundError: No ``pool.json`` in ``directory``.
            SnapshotPoolError: The index is unreadable, of an unknown version,
                or structurally malformed.
            PinnedSnapshotError: A pinned snapshot's file is missing.
        """
        sampling = _validated_sampling(sampling)
        index_path = os.path.join(directory, INDEX_FILENAME)
        with open(index_path, "r", encoding="utf-8") as handle:
            try:
                raw = json.load(handle)
            except json.JSONDecodeError as exc:
                raise SnapshotPoolError(
                    f"pool index {index_path!r} is not valid JSON: {exc}"
                ) from exc

        if not isinstance(raw, dict):
            raise SnapshotPoolError(
                f"pool index {index_path!r} is a {type(raw).__name__}, "
                "expected an object"
            )
        version = raw.get("index_version")
        if version != INDEX_VERSION:
            raise SnapshotPoolError(
                f"pool index {index_path!r} has index_version {version!r}, "
                f"this build writes {INDEX_VERSION}"
            )

        try:
            pool = cls(
                directory,
                elo_k=float(raw.get("elo_k", ELO_K)),
                elo_initial=float(raw.get("elo_initial", ELO_INITIAL)),
                sampling=sampling,
                log=log,
            )
            entries = raw["snapshots"]
            retired = raw.get("retired", [])
            if not isinstance(entries, list) or not isinstance(retired, list):
                raise TypeError("'snapshots' and 'retired' must be lists")

            for entry in entries:
                # basename(): the index is ours, but a filename is still the one
                # field that could escape the run directory, and rejoining a
                # bare name costs nothing.
                filename = os.path.basename(str(entry["filename"]))
                record = SnapshotRecord(
                    snapshot_id=int(entry["snapshot_id"]),
                    grad_step=int(entry["grad_step"]),
                    path=os.path.join(directory, filename),
                    elo=float(entry["elo"]),
                    pinned=bool(entry["pinned"]),
                )
                pool._records[record.snapshot_id] = record
                pool._stats[record.snapshot_id] = MatchStats(
                    plays=int(entry.get("plays", 0)),
                    learner_wins=float(entry.get("learner_wins", 0.0)),
                )

            for entry in retired:
                retired_id = int(entry["snapshot_id"])
                pool._retired_elo[retired_id] = float(entry["elo"])
                pool._stats[retired_id] = MatchStats(
                    plays=int(entry.get("plays", 0)),
                    learner_wins=float(entry.get("learner_wins", 0.0)),
                )

            pool._learner_elo_online = float(
                raw.get("learner_elo_online", pool._elo_initial)
            )
            pool._learner_elo_rated = float(
                raw.get("learner_elo_rated", pool._elo_initial)
            )
            # `.get` with a zero default: an index written before these counters
            # existed is not malformed, it just has no history to restore.
            pool._matches_scored = int(raw.get("matches_scored", 0))
            pool._draws_scored = int(raw.get("draws_scored", 0))
            pool._rated_matches = int(raw.get("rated_matches", 0))
            known_ids = list(pool._records) + list(pool._retired_elo)
            # max(...)+1 as a floor: ids must stay monotone even if the stored
            # counter was written by an older build or hand-edited.
            pool._next_id = max(
                int(raw.get("next_snapshot_id", 0)),
                max(known_ids) + 1 if known_ids else 0,
            )
            # Continue the stamp where the index left off rather than restarting
            # at 0 underneath a much later pool.json. max(0, ...) because a
            # hand-edited negative would invert the ordering the stamp exists to
            # provide; an index written before the stamp existed simply has
            # none, and starts at 0. `_persisted_generation` deliberately stays
            # at -1: the restored pool was BUILT from this index, so its first
            # write is at least as fresh, and taking a write watermark from disk
            # would let one bad edit silence every persist for the rest of the
            # run.
            pool._generation = max(0, int(raw.get("index_generation", 0)))
        except (KeyError, TypeError, ValueError) as exc:
            raise SnapshotPoolError(
                f"pool index {index_path!r} is malformed: {exc!r}"
            ) from exc

        for record in pool.pinned_references():
            if not os.path.isfile(record.path):
                raise PinnedSnapshotError(
                    f"PINNED snapshot {record.snapshot_id} is missing its file "
                    f"{record.path!r} at pool load. Refusing to continue: the "
                    "pinned references are the fixed opponents behind "
                    "selfplay/win_rate_vs_ref_<id>, and substituting another "
                    "snapshot would silently redefine that metric."
                )
        return pool

    # -- internals --------------------------------------------------------

    def _stats_locked(self, snapshot_id: int) -> MatchStats:
        """The live stats object for ``snapshot_id`` (zeroed if absent).

        Caller holds ``_lock``. Read-only for callers outside this class — see
        :meth:`stats_for`, which hands out a copy.
        """
        return self._stats.get(snapshot_id) or MatchStats()

    def _sorted_records_locked(self) -> List[SnapshotRecord]:
        """Live records ordered by id. Caller holds ``_lock``.

        The ordering is not cosmetic: it makes weighted sampling reproducible for
        a given RNG stream regardless of dict insertion history.
        """
        return [self._records[key] for key in sorted(self._records)]

    def _eligible_locked(self, exclude_id: Optional[int]) -> List[SnapshotRecord]:
        """Candidates for :meth:`sample`, with the bootstrap rule applied.

        Caller holds ``_lock``. The same-version exclusion is skipped while the
        pool holds fewer than two distinct versions — see :meth:`sample`.
        """
        records = self._sorted_records_locked()
        if exclude_id is None or len(records) < 2:
            return records
        # Ids are unique, so removing one from >= 2 always leaves >= 1.
        return [rec for rec in records if rec.snapshot_id != exclude_id]

    def _raw_weights_locked(
        self, candidates: Sequence[SnapshotRecord]
    ) -> Dict[int, float]:
        """Unnormalized sampling weights. Caller holds ``_lock``.

        ``uniform``: every candidate gets 1.0 — the flat baseline the PFSP arm is
        measured against, and the fallback in the plan's feature-cut order.

        ``pfsp``: ``w_i = f(p_i) + PFSP_FLOOR_MASS / N``, with
        ``f(p) = p(1 - p)`` and ``p_i`` straight from :meth:`MatchStats.win_rate`.
        Two terms, two jobs:

          * ``f`` peaks at ``p = 0.5`` and falls to zero at both ends, so mass
            lands on the opponents that are challenging but still winnable — the
            ones the learner can actually learn from. Because ``win_rate``
            carries the Beta(1, 1) prior, a snapshot with ZERO plays reports
            ``p = 0.5`` and therefore the MAXIMUM ``f``: a brand-new past self is
            treated as maximally informative until its record says otherwise.
            The prior is not reimplemented here; that is deliberate, so there is
            exactly one place where the smoothing can be wrong.
          * The floor keeps every member reachable. ``f`` is exactly 0 at
            ``p = 0`` and ``p = 1``, and a zero-weight snapshot is one the
            learner never plays again — so it never accumulates the results that
            would move ``p`` back off the extreme. That is the mechanism by which
            a self-play run forgets how to beat its oldest ancestors, and the
            pinned references (the fixed measuring sticks behind
            ``selfplay/win_rate_vs_ref_<id>``) are exactly the members that sit
            at the extremes once the learner has outgrown them.

        Weights are unnormalized here; :meth:`_normalized_weights_locked`
        divides by the sum. In absolute terms the floor contributes
        ``PFSP_FLOOR_MASS`` of unnormalized weight in TOTAL — one N-th of it to
        each of the N candidates — so its share of the normalized distribution
        is ``PFSP_FLOOR_MASS / (sum(f) + PFSP_FLOOR_MASS)``: the WHOLE
        distribution when every ``f`` is zero, and a diminishing fraction as the
        shaping mass grows. That share is not pinned at 5%, and EXCEEDS it
        whenever ``sum(f) < 0.95`` — common, since ``f`` maxes out at 0.25. On
        TC14's table (``sum(f) = 0.66``) the floor holds 0.05 / 0.71 = 7.04%.

        Raises:
            SnapshotPoolError: A mode was added to :data:`SAMPLING_MODES` without
                a branch here — loud rather than silently uniform.
        """
        if not candidates:
            # Never reached through _normalized_weights_locked (which short-
            # circuits on empty), but a direct caller must not divide by zero.
            return {}

        if self._sampling == "uniform":
            return {rec.snapshot_id: 1.0 for rec in candidates}

        if self._sampling == "pfsp":
            floor = PFSP_FLOOR_MASS / float(len(candidates))
            weights: Dict[int, float] = {}
            for record in candidates:
                probability = self._stats_locked(record.snapshot_id).win_rate()
                weights[record.snapshot_id] = (
                    probability * (1.0 - probability) + floor
                )
            return weights

        raise SnapshotPoolError(
            f"unknown sampling mode {self._sampling!r}; __init__ validates "
            f"against {sorted(SAMPLING_MODES)}, so a mode was added there "
            "without a branch in _raw_weights_locked"
        )

    def _normalized_weights_locked(
        self, candidates: Sequence[SnapshotRecord]
    ) -> Tuple[Dict[int, float], Optional[str]]:
        """Normalize :meth:`_raw_weights_locked` into a probability distribution.

        Caller holds ``_lock``. Falls back to uniform — loudly — if the raw
        weights are non-finite or sum to zero. Neither can happen with the
        shipped formulas (``p(1 - p) + floor`` is finite and strictly positive
        for every ``p`` in ``[0, 1]``), so this is defence in depth: a NaN or an
        all-zero distribution reaching the sampler would either wedge an arena or
        silently pin it to one opponent, and both present as a training problem
        rather than as a weighting bug.

        Returns ``(weights, warning)`` rather than logging the warning itself:
        ``_log`` is a caller-supplied callback and this method runs under
        ``_lock``. A callback that waits on a lock held by a thread that is
        waiting on ``_lock`` would deadlock the pool — ``_lock`` being reentrant
        saves only the same-thread case. The CALLER logs ``warning`` (when it is
        not ``None``) after releasing, exactly as :meth:`_retire` does.

        Returns:
            ``(weights, warning)``: the normalized distribution, and either
            ``None`` or the message the caller must log once the lock is out of
            its hands.
        """
        if not candidates:
            return {}, None
        raw = self._raw_weights_locked(candidates)
        values = [raw.get(rec.snapshot_id, float("nan")) for rec in candidates]
        total = math.fsum(value for value in values if math.isfinite(value))
        if any(not math.isfinite(value) for value in values) or total <= 0.0:
            uniform = 1.0 / len(candidates)
            return (
                {rec.snapshot_id: uniform for rec in candidates},
                f"WARNING: sampling weights are unusable ({values!r}); falling "
                "back to uniform over "
                f"{[rec.snapshot_id for rec in candidates]}",
            )
        return (
            {
                rec.snapshot_id: value / total
                for rec, value in zip(candidates, values)
            },
            None,
        )

    def _retire(self, record: SnapshotRecord, reason: str) -> None:
        """Apply the corruption policy to ``record``: fatal if pinned, else drop.

        The unpinned drop keeps the snapshot's frozen rating in ``_retired_elo``
        and its statistics in ``_stats``, so a match that was already in flight
        against it can still be rated when it finishes.

        Args:
            record: The snapshot that could not be read.
            reason: Human-readable cause, quoted into the log/exception.

        Raises:
            PinnedSnapshotError: ``record`` is pinned.
        """
        remaining: List[int] = []
        with self._lock:
            # Prefer the registry's copy: it is the authority on `pinned`.
            live = self._records.get(record.snapshot_id, record)
            pinned = live.pinned
            if not pinned:
                removed = self._records.pop(record.snapshot_id, None)
                if removed is not None:
                    self._retired_elo[record.snapshot_id] = removed.elo
                    # Only on a real drop: re-retiring an already-dropped member
                    # changes nothing, and a bump there would break the
                    # "equal generation, equal payload" reading of the stamp.
                    self._generation += 1
                remaining = sorted(self._records)

        if pinned:
            raise PinnedSnapshotError(
                f"PINNED snapshot {record.snapshot_id} at {record.path!r} is "
                f"unusable ({reason}). Refusing to continue: a pinned reference "
                "is a fixed opponent behind selfplay/win_rate_vs_ref_<id>, and "
                "resampling around it would silently redefine that metric — or, "
                "worse, put an untrained net in the arena."
            )

        self._log(
            f"DROPPED unpinned snapshot {record.snapshot_id} at {record.path!r} "
            f"({reason}); resampling. Pool now holds {remaining}."
        )

    def _unreadable(self, record: SnapshotRecord, reason: str) -> Any:
        """Drop-or-die for a snapshot that failed to load; ALWAYS raises.

        Split out so :meth:`load_payload` can ``return self._unreadable(...)`` at
        each validation point and stay flat. The return annotation is a promise
        this function never has to keep — every path raises.

        Raises:
            PinnedSnapshotError: The snapshot is pinned.
            SnapshotUnavailableError: The snapshot was unpinned and is now
                dropped.
        """
        self._retire(record, reason)  # raises for a pinned member
        raise SnapshotUnavailableError(
            f"snapshot {record.snapshot_id} at {record.path!r} is unusable "
            f"({reason}); it has been dropped from the pool — resample."
        )


def _weighted_choice(
    rng: Any, candidates: Sequence[SnapshotRecord], weights: Mapping[int, float]
) -> SnapshotRecord:
    """Draw one record from ``candidates`` with the given normalized ``weights``.

    Inverse-CDF over the candidate order, from a single ``rng.random()`` draw —
    one uniform per sample, so an arena's RNG stream advances predictably and a
    replayed seed reproduces the same opponent sequence.

    Args:
        rng: Object exposing ``random() -> float`` in ``[0, 1)``.
        candidates: Records in a stable order (id-ascending), never empty.
        weights: ``{snapshot_id: probability}``, summing to ~1.

    Returns:
        The chosen record.
    """
    draw = float(rng.random())
    cumulative = 0.0
    for record in candidates:
        cumulative += weights.get(record.snapshot_id, 0.0)
        if draw < cumulative:
            return record
    # Only reachable when floating-point summation lands a hair under the draw.
    return candidates[-1]
