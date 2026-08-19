"""Tests for the M4 self-play snapshot pool (T8/T9; TC12, TC14-TC17, TC37).

What is pinned here, and why each pin exists:

* **TC12 — the round-trip.** A pool reloaded from ``pool.json`` must be the pool
  that was saved: same snapshots, same per-snapshot statistics, BOTH Elo series,
  and the id counter. A restart that silently reset the statistics would reset
  PFSP's view of the opponent field, and a restart that reset the id counter
  would REUSE an id — two different policy versions answering to one identity,
  which corrupts every match attributed to it.
* **TC17 — the bootstrap rule.** ``exclude_id`` (don't fight a copy of yourself)
  must switch on only once the pool holds two distinct versions. Applied from
  episode 1, when snapshot 0 is the only member, every arena would find no legal
  opponent and the whole run would deadlock before its first gradient step.
* **TC37 — payload shape.** The snapshot payload must load through
  ``eval.evaluate._load_drqn``, which builds ``DuelingDRQN()`` with DEFAULT
  kwargs. A payload the shared loader cannot read is a snapshot the eval and
  deploy paths cannot use, and that is only discovered on demo day.
* **Elo moves only the learner.** A snapshot's rating is frozen at creation. If
  snapshot ratings drifted, the pool would be rating a moving target against a
  moving target and ``elo/learner_rated`` (the AC7 trend series) would mean
  nothing. The two series are also pinned apart: an ε > 0 match must move
  ``online`` and NOT ``rated``.
* **The Beta(1, 1) prior.** ``win_rate()`` is defined (0.5) at zero plays and
  never reaches exactly 0 or 1. This is what stops the ``p(1 - p)`` weighting
  from being NaN, or zero, for a brand-new snapshot — a zero-weighted snapshot
  can never earn the plays that would raise its weight, so it would be dead on
  arrival.
* **TC14-TC16 + AC6 — PFSP.** ``w_i = p_i(1 - p_i) + 0.05 / N``, normalized.
  TC14 pins the exact numbers for a fixed stats table (a formula that is merely
  "monotone in the right direction" can be wrong by a factor and still pass a
  fuzzy test). TC16 pins the floor at ``p = 0`` and ``p = 1``, where the shaping
  term is exactly zero: without it a snapshot the learner always beats — an old
  pinned reference — gets weight 0, is never sampled again, and can never earn
  the results that would bring it back. That is how a self-play run forgets how
  to beat its ancestors. AC6 pins the point of the whole exercise: measurably
  more mass in the 0.4-0.6 win-rate band than uniform puts there.
* **Disk I/O outside the state lock, mechanically.** The spec's headline
  concurrency property, and until now nothing detected its regression: moving a
  ``torch.load`` inside ``with self._lock:`` keeps every other test green while
  serializing 25 arena threads behind one disk read — which presents as a
  throughput problem, not as a lock bug.
  ``test_disk_io_never_runs_under_the_state_lock`` records lock ownership at
  every read and every write instead.
* **``persist(payload)`` never takes the state lock, and never writes
  backwards.** It is handed the index rather than building it, so it holds only
  ``_io_lock`` and waits for nothing — the ABBA cycle between the two locks
  becomes unwritable rather than merely avoided. Pinned by persisting from a
  worker thread while the main thread holds ``_lock``: the old shape deadlocks
  there and the join timeout says so. Handing the payload in is exactly what
  costs freshness, so each payload carries a monotone ``index_generation`` and
  ``persist`` drops one older than its last write. That is pinned
  DETERMINISTICALLY — two payloads persisted in the wrong order, not two threads
  racing — because without it a stale index overwrites a fresh one, ``load``'s
  ``max(known ids) + 1`` floor reads the STALE ids back and re-issues a live id,
  and the next snapshot silently overwrites ``snap_<id>.pt`` under an identity
  that already carries statistics, an Elo, and possibly a pinned metric.
* **Corruption asymmetry.** A missing/corrupt UNPINNED snapshot is dropped and
  resampled; a missing/corrupt PINNED one is FATAL. Pinned members are the fixed
  opponents behind ``selfplay/win_rate_vs_ref_<id>``; quietly substituting
  another snapshot would redefine that metric instead of reporting the loss.
  Neither path may ever fall back to an untrained net.
* **One lock, no deadlock.** Concurrent scoring from many arena threads must
  lose no match, and sampling must keep working while the learner thread adds
  snapshots. Every threaded case here runs with a join timeout, so a lock-order
  bug FAILS the test instead of hanging the whole pytest process.

No sockets, no live server, no Minecraft: every filesystem here is ``tmp_path``.
"""

from __future__ import annotations

import json
import math
import os
import random
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pytest

torch = pytest.importorskip("torch", exc_type=ImportError)

import opponents.snapshot_pool as snapshot_pool  # noqa: E402
from opponents.snapshot_pool import (  # noqa: E402
    DEFAULT_SAMPLING,
    ELO_INITIAL,
    ELO_K,
    INDEX_FILENAME,
    MatchResult,
    MatchStats,
    PFSP_FLOOR_MASS,
    PinnedSnapshotError,
    SAMPLING_MODES,
    SnapshotPool,
    SnapshotPoolError,
    SnapshotRecord,
    SnapshotUnavailableError,
    expected_score,
    updated_elo,
)


# Seconds a worker thread gets before a test calls it a deadlock. Generous
# enough to absorb a loaded box, short enough that a genuine lock-order bug
# fails the run instead of wedging it.
_THREAD_TIMEOUT = 30.0


def _fake_state_dict(value: float = 0.0) -> Dict[str, Any]:
    """A minimal, valid ``state_dict``.

    The pool is architecture-agnostic — it stores whatever mapping of tensors it
    is handed — so almost every test here avoids building a real net. TC37 uses
    a real :class:`~agent.dqn.DuelingDRQN` precisely because THAT test is about
    the shared loader's expectations.
    """
    return {"weight": torch.full((4,), float(value)), "bias": torch.zeros(2)}


def _new_pool(tmp_path, log: Optional[Callable[[str], None]] = None) -> SnapshotPool:
    """A pool rooted in a fresh ``snapshots`` directory under ``tmp_path``."""
    return SnapshotPool(str(tmp_path / "snapshots"), log=log)


def _record_history(
    pool: SnapshotPool, snapshot_id: int, plays: int, wins: int
) -> None:
    """Score ``wins`` wins and ``plays - wins`` losses against ``snapshot_id``.

    Drives the PFSP tests through the PUBLIC scoring path rather than poking
    ``pool._stats``, so a weighting that reads the statistics differently than
    ``record_result`` writes them cannot pass.
    """
    assert 0 <= wins <= plays, f"impossible history {wins}/{plays}"
    for index in range(plays):
        score = 1.0 if index < wins else 0.0
        pool.record_result(MatchResult.create(snapshot_id, 0.0, 0.0, score))


def _run_threads(targets: List[Callable[[], None]]) -> None:
    """Run ``targets`` concurrently and FAIL (never hang) if one does not finish.

    A deadlock is not a test result; this turns it into one.
    """
    threads = [threading.Thread(target=fn, daemon=True) for fn in targets]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=_THREAD_TIMEOUT)
        assert not thread.is_alive(), (
            f"worker thread did not finish within {_THREAD_TIMEOUT}s — the pool "
            "lock deadlocked (disk I/O under the lock, or an _io_lock/_lock "
            "ordering inversion)"
        )


# ---------------------------------------------------------------------------
# TC12 -- add / persist / reload round-trip
# ---------------------------------------------------------------------------


def test_tc12_add_persist_reload_round_trip(tmp_path):
    """A reloaded pool equals the original: snapshots, stats, and both Elo series."""
    pool = _new_pool(tmp_path)
    reference = pool.add(_fake_state_dict(1.0), grad_step=0, elo=1000.0, pinned=True)
    later = pool.add(_fake_state_dict(2.0), grad_step=1000, elo=1000.0)

    # A mixed history: an exploring win, a greedy draw, a greedy loss.
    pool.record_result(MatchResult.create(reference.snapshot_id, 0.1, 0.02, 1.0))
    pool.record_result(MatchResult.create(reference.snapshot_id, 0.0, 0.0, 0.5))
    pool.record_result(MatchResult.create(later.snapshot_id, 0.0, 0.0, 0.0))
    pool.persist(pool.index_payload())

    reloaded = SnapshotPool.load(
        str(tmp_path / "snapshots"), sampling=DEFAULT_SAMPLING
    )

    assert reloaded.index_payload() == pool.index_payload()
    assert reloaded.records() == pool.records() == [reference, later]
    assert reloaded.pinned_references() == [reference]
    assert reloaded.learner_elo_online == pool.learner_elo_online
    assert reloaded.learner_elo_rated == pool.learner_elo_rated
    assert reloaded.learner_elo_online != ELO_INITIAL, (
        "the fixture must actually move the online rating, or this test would "
        "pass on a pool that persists no Elo at all"
    )
    for snapshot_id in (reference.snapshot_id, later.snapshot_id):
        assert reloaded.stats_for(snapshot_id) == pool.stats_for(snapshot_id)
    assert reloaded.stats_for(reference.snapshot_id) == MatchStats(
        plays=2, learner_wins=1.5
    )

    # The reloaded pool can read the weights the ORIGINAL wrote...
    assert reloaded.load_payload(later)["grad_step"] == 1000
    assert torch.equal(
        reloaded.load_state_dict(later)["weight"], torch.full((4,), 2.0)
    )
    # ...and keeps numbering where the original stopped. A reused id would make
    # two different policy versions answer to one identity.
    assert reloaded.add(_fake_state_dict(3.0), 2000, 1000.0).snapshot_id == 2


def test_add_persists_the_index_without_an_explicit_persist(tmp_path):
    """``add`` writes ``pool.json`` itself, with no separate ``persist()`` call.

    A snapshot file with no index entry is an orphan that a restart never finds.
    """
    pool = _new_pool(tmp_path)
    record = pool.add(_fake_state_dict(), grad_step=42, elo=1000.0, pinned=True)

    index_path = tmp_path / "snapshots" / INDEX_FILENAME
    assert index_path.is_file()
    index = json.loads(index_path.read_text(encoding="utf-8"))
    assert [entry["snapshot_id"] for entry in index["snapshots"]] == [0]
    assert index["snapshots"][0]["filename"] == "snap_0.pt"
    assert index["snapshots"][0]["grad_step"] == 42
    assert index["snapshots"][0]["pinned"] is True
    assert record.path == str(tmp_path / "snapshots" / "snap_0.pt")


def test_atomic_writes_leave_no_temp_files(tmp_path):
    """Every write goes temp -> fsync -> os.replace, and cleans up after itself."""
    pool = _new_pool(tmp_path)
    pool.add(_fake_state_dict(), grad_step=0, elo=1000.0, pinned=True)
    pool.persist(pool.index_payload())
    pool.persist(pool.index_payload())

    leftovers = sorted(
        entry.name
        for entry in (tmp_path / "snapshots").iterdir()
        if entry.name.endswith(".tmp") or entry.name.startswith(".")
    )
    assert leftovers == []


def test_add_freezes_the_weights_against_later_mutation(tmp_path):
    """A snapshot must not track the learner's next optimizer.step().

    ``state_dict()`` hands back tensor VIEWS; storing them directly would make
    every "past self" in the pool silently become the present self.
    """
    pool = _new_pool(tmp_path)
    live = _fake_state_dict(1.0)
    record = pool.add(live, grad_step=0, elo=1000.0, pinned=True)

    with torch.no_grad():
        live["weight"].add_(100.0)  # simulate an optimizer step

    assert torch.equal(pool.load_state_dict(record)["weight"], torch.full((4,), 1.0))


def test_load_rejects_a_foreign_index_version(tmp_path):
    """A stale/unknown ``pool.json`` fails loudly instead of half-deserializing."""
    pool = _new_pool(tmp_path)
    pool.add(_fake_state_dict(), 0, 1000.0, pinned=True)

    index_path = tmp_path / "snapshots" / INDEX_FILENAME
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["index_version"] = 99
    index_path.write_text(json.dumps(index), encoding="utf-8")

    with pytest.raises(SnapshotPoolError, match="index_version"):
        SnapshotPool.load(str(tmp_path / "snapshots"), sampling=DEFAULT_SAMPLING)


# ---------------------------------------------------------------------------
# TC17 -- same-version exclusion applies only with >= 2 distinct versions
# ---------------------------------------------------------------------------


def test_tc17_bootstrap_keeps_snapshot_zero_sampleable(tmp_path):
    """With one version in the pool, excluding it must NOT empty the candidates.

    This is the deadlock guard: on episode 1 the learner's current version IS
    snapshot 0, and honouring the exclusion would leave every arena with no
    legal opponent.
    """
    pool = _new_pool(tmp_path)
    snapshot_zero = pool.add(_fake_state_dict(), grad_step=0, elo=1000.0, pinned=True)
    rng = random.Random(1234)

    for _ in range(32):
        assert pool.sample(rng, exclude_id=snapshot_zero.snapshot_id) is snapshot_zero


def test_tc17_exclusion_engages_once_a_second_version_exists(tmp_path):
    """With >= 2 versions the excluded id is never drawn — in either direction."""
    pool = _new_pool(tmp_path)
    pool.add(_fake_state_dict(1.0), grad_step=0, elo=1000.0, pinned=True)
    pool.add(_fake_state_dict(2.0), grad_step=1000, elo=1000.0)
    rng = random.Random(7)

    assert {pool.sample(rng, exclude_id=0).snapshot_id for _ in range(64)} == {1}
    assert {pool.sample(rng, exclude_id=1).snapshot_id for _ in range(64)} == {0}
    # Without an exclusion, both versions are reachable. No history is recorded
    # in this fixture, so both sit at the Beta prior's p = 0.5, where the shaping
    # term f(p) = p(1 - p) is at its MAXIMUM (0.25) — equal, and far from zero,
    # on f alone. The floor is what rescues a member at p = 0 or p = 1; that is
    # TC16's case, not this one.
    assert {pool.sample(rng).snapshot_id for _ in range(200)} == {0, 1}


def test_sample_accepts_a_numpy_generator_and_rejects_a_non_rng(tmp_path):
    """Arena threads pass their own stream; no random() means a wiring bug."""
    pool = _new_pool(tmp_path)
    record = pool.add(_fake_state_dict(), 0, 1000.0, pinned=True)

    assert pool.sample(np.random.default_rng(0)) is record
    with pytest.raises(TypeError, match="random"):
        pool.sample(object())


def test_sample_on_an_empty_pool_raises(tmp_path):
    """No silent fallback: an empty pool has no opponent to offer."""
    with pytest.raises(SnapshotPoolError, match="no sampleable member"):
        _new_pool(tmp_path).sample(random.Random(0))


# ---------------------------------------------------------------------------
# TC37 -- the payload loads through the shared loader
# ---------------------------------------------------------------------------


def test_tc37_payload_loads_through_the_shared_drqn_loader(tmp_path):
    """``eval.evaluate._load_drqn`` must accept a snapshot written by the pool.

    ``_load_drqn`` builds ``DuelingDRQN()`` with DEFAULT kwargs and reads the
    weights from the payload's ``"model"`` key, so the snapshot must be written
    from a default-shaped net under exactly that key.
    """
    from agent.dqn import DuelingDRQN
    from eval.evaluate import _load_drqn

    net = DuelingDRQN()
    pool = _new_pool(tmp_path)
    record = pool.add(net.state_dict(), grad_step=7, elo=1000.0, pinned=True)

    loaded = _load_drqn(record.path, torch.device("cpu"))

    assert loaded.training is False, "_load_drqn must hand back an eval-mode net"
    original = net.state_dict()
    reloaded = loaded.state_dict()
    assert set(reloaded) == set(original)
    for key, value in original.items():
        assert torch.equal(reloaded[key], value), f"tensor {key!r} did not survive"


def test_snapshot_payload_carries_every_contracted_key(tmp_path):
    """The payload keys are a contract: model, grad_step, code_version, id, elo."""
    pool = _new_pool(tmp_path)
    record = pool.add(_fake_state_dict(), grad_step=1500, elo=1042.5, pinned=True)

    payload = pool.load_payload(record)
    assert set(payload) == {"model", "grad_step", "code_version", "snapshot_id", "elo"}
    assert payload["grad_step"] == 1500
    assert payload["snapshot_id"] == 0
    assert payload["elo"] == 1042.5
    assert isinstance(payload["code_version"], str) and payload["code_version"]


def test_add_refuses_an_empty_state_dict(tmp_path):
    """An empty snapshot would load as an untrained net and become the opponent."""
    with pytest.raises(ValueError, match="empty state_dict"):
        _new_pool(tmp_path).add({}, grad_step=0, elo=1000.0)


# ---------------------------------------------------------------------------
# Elo arithmetic (TC18) -- only the learner's rating moves
# ---------------------------------------------------------------------------


def test_expected_score_is_symmetric_and_half_at_parity():
    """E_a = 1/(1+10**((R_b-R_a)/400)) -- the spec's exact curve."""
    assert expected_score(1000.0, 1000.0) == pytest.approx(0.5)
    assert expected_score(1200.0, 1000.0) == pytest.approx(
        1.0 / (1.0 + 10.0 ** (-200.0 / 400.0))
    )
    assert expected_score(1000.0, 1200.0) + expected_score(1200.0, 1000.0) == (
        pytest.approx(1.0)
    )
    assert updated_elo(1000.0, 1000.0, 1.0) == pytest.approx(1000.0 + ELO_K * 0.5)


@pytest.mark.parametrize(
    "score, expected_rating",
    [(1.0, 1012.0), (0.5, 1000.0), (0.0, 988.0)],
)
def test_elo_update_at_parity_moves_by_k_times_the_surprise(
    tmp_path, score, expected_rating
):
    """Win/draw/loss against an equal-rated snapshot: +K/2, 0, -K/2 (K = 24)."""
    pool = _new_pool(tmp_path)
    record = pool.add(_fake_state_dict(), grad_step=0, elo=ELO_INITIAL, pinned=True)

    pool.record_result(MatchResult.create(record.snapshot_id, 0.0, 0.0, score))

    assert pool.learner_elo_online == pytest.approx(expected_rating)
    assert pool.learner_elo_rated == pytest.approx(expected_rating)


def test_only_the_learner_rating_moves(tmp_path):
    """Snapshot ratings are frozen at creation; the next update uses the gap."""
    pool = _new_pool(tmp_path)
    record = pool.add(_fake_state_dict(), grad_step=0, elo=1000.0, pinned=True)

    pool.record_result(MatchResult.create(record.snapshot_id, 0.0, 0.0, 1.0))
    assert pool.get(record.snapshot_id).elo == 1000.0, "snapshot rating drifted"
    assert pool.learner_elo_online == pytest.approx(1012.0)

    pool.record_result(MatchResult.create(record.snapshot_id, 0.0, 0.0, 1.0))
    # Re-derived from the spec formula, not from the implementation.
    expectation = 1.0 / (1.0 + 10.0 ** ((1000.0 - 1012.0) / 400.0))
    assert pool.learner_elo_online == pytest.approx(1012.0 + 24.0 * (1.0 - expectation))
    assert pool.get(record.snapshot_id).elo == 1000.0


def test_draw_scores_one_half_into_the_statistics(tmp_path):
    """A draw is 0.5 of a win, in the stats as well as in the rating."""
    pool = _new_pool(tmp_path)
    record = pool.add(_fake_state_dict(), grad_step=0, elo=1000.0, pinned=True)

    pool.record_result(MatchResult.create(record.snapshot_id, 0.0, 0.0, 0.5))

    assert pool.stats_for(record.snapshot_id) == MatchStats(plays=1, learner_wins=0.5)
    assert pool.learner_elo_online == pytest.approx(1000.0)


def test_the_two_elo_series_diverge_on_rated_eligibility(tmp_path):
    """``online`` takes every match; ``rated`` takes only the greedy ones (AC7)."""
    pool = _new_pool(tmp_path)
    record = pool.add(_fake_state_dict(), grad_step=0, elo=1000.0, pinned=True)

    # A training match: the learner explores at 0.05, the opponent at 0.02.
    pool.record_result(MatchResult.create(record.snapshot_id, 0.05, 0.02, 1.0))
    assert pool.learner_elo_online == pytest.approx(1012.0)
    assert pool.learner_elo_rated == ELO_INITIAL, (
        "an exploring match must never touch elo/learner_rated — that series is "
        "the AC7 trend and the checkpoint-selection input"
    )

    # An eval-cycle match: both sides greedy.
    pool.record_result(MatchResult.create(record.snapshot_id, 0.0, 0.0, 1.0))
    assert pool.learner_elo_rated == pytest.approx(1012.0)
    assert pool.stats_for(record.snapshot_id).plays == 2, (
        "PFSP statistics come from EVERY match, rated or not"
    )


def test_record_result_for_an_unknown_snapshot_raises(tmp_path):
    """Scoring against an id the pool never held is a wiring bug, not a no-op."""
    pool = _new_pool(tmp_path)
    pool.add(_fake_state_dict(), grad_step=0, elo=1000.0, pinned=True)

    with pytest.raises(KeyError, match="unknown snapshot_id 7"):
        pool.record_result(MatchResult.create(7, 0.0, 0.0, 1.0))


def test_match_result_rated_eligible_must_agree_with_the_epsilons():
    """Both directions of the inconsistency silently break an Elo series."""
    with pytest.raises(ValueError, match="rated_eligible"):
        MatchResult(
            snapshot_id=0,
            learner_epsilon=0.0,
            opponent_epsilon=0.0,
            score=1.0,
            rated_eligible=False,  # empties the rated series
        )
    with pytest.raises(ValueError, match="rated_eligible"):
        MatchResult(
            snapshot_id=0,
            learner_epsilon=0.05,
            opponent_epsilon=0.0,
            score=1.0,
            rated_eligible=True,  # poisons the rated series with ε-noise
        )

    assert MatchResult.create(0, 0.0, 0.0, 1.0).rated_eligible is True
    assert MatchResult.create(0, 0.0, 0.02, 1.0).rated_eligible is False


def test_match_result_rejects_non_finite_and_out_of_range_values():
    """NaN must never reach the rating table."""
    with pytest.raises(ValueError, match="score"):
        MatchResult.create(0, 0.0, 0.0, float("nan"))
    with pytest.raises(ValueError, match="score"):
        MatchResult.create(0, 0.0, 0.0, 1.5)
    with pytest.raises(ValueError, match="learner_epsilon"):
        MatchResult.create(0, -0.1, 0.0, 1.0)


def test_snapshot_record_rejects_impossible_fields():
    """The record IS a policy version's identity; garbage in it poisons the rest."""
    with pytest.raises(ValueError, match="snapshot_id"):
        SnapshotRecord(snapshot_id=-1, grad_step=0, path="p", elo=1000.0, pinned=False)
    with pytest.raises(ValueError, match="grad_step"):
        SnapshotRecord(snapshot_id=0, grad_step=-1, path="p", elo=1000.0, pinned=False)
    with pytest.raises(ValueError, match="elo"):
        SnapshotRecord(
            snapshot_id=0, grad_step=0, path="p", elo=float("inf"), pinned=False
        )


# ---------------------------------------------------------------------------
# Beta(1, 1) prior -- the reason a zero-play snapshot is sampleable at all
# ---------------------------------------------------------------------------


def test_beta_prior_is_defined_at_zero_plays():
    """(wins + 1) / (plays + 2): 0.5 with nothing on the books, never 0/0."""
    fresh = MatchStats()
    assert fresh.plays == 0 and fresh.learner_wins == 0.0
    assert fresh.win_rate() == 0.5
    assert math.isfinite(fresh.win_rate())


def test_beta_prior_never_reaches_zero_or_one():
    """p(1-p) must stay positive, or PFSP weights the snapshot out of existence."""
    all_losses = MatchStats(plays=20, learner_wins=0.0)
    all_wins = MatchStats(plays=20, learner_wins=20.0)

    assert all_losses.win_rate() == pytest.approx(1.0 / 22.0)
    assert all_wins.win_rate() == pytest.approx(21.0 / 22.0)
    for stats in (all_losses, all_wins):
        probability = stats.win_rate()
        assert 0.0 < probability < 1.0
        assert probability * (1.0 - probability) > 0.0

    assert MatchStats(plays=1, learner_wins=1.0).win_rate() == pytest.approx(2.0 / 3.0)


def test_a_brand_new_snapshot_reports_the_prior_through_the_pool(tmp_path):
    """The pool hands out the prior for an unplayed snapshot, not a zero."""
    pool = _new_pool(tmp_path)
    record = pool.add(_fake_state_dict(), grad_step=0, elo=1000.0, pinned=True)

    assert pool.stats_for(record.snapshot_id).win_rate() == 0.5
    assert pool.stats_for(999).win_rate() == 0.5  # unknown id: not an error


def test_stats_for_returns_a_copy(tmp_path):
    """A caller must not mutate pool state through a returned stats object."""
    pool = _new_pool(tmp_path)
    record = pool.add(_fake_state_dict(), grad_step=0, elo=1000.0, pinned=True)
    pool.record_result(MatchResult.create(record.snapshot_id, 0.0, 0.0, 1.0))

    borrowed = pool.stats_for(record.snapshot_id)
    borrowed.plays = 10_000
    borrowed.learner_wins = 10_000.0

    assert pool.stats_for(record.snapshot_id) == MatchStats(plays=1, learner_wins=1.0)


# ---------------------------------------------------------------------------
# pfsp_weights (T9) -- w = p(1-p) + 0.05/N, and the uniform arm it is measured
# against. TC14 (exact numbers), TC15 (zero plays), TC16 (the floor), AC6.
# ---------------------------------------------------------------------------


def test_pfsp_weights_are_finite_normalized_and_cover_every_member(tmp_path):
    """The invariants AC6 needs — and the guard must never have to supply them."""
    messages: List[str] = []
    pool = _new_pool(tmp_path, log=messages.append)
    for index in range(3):
        pool.add(_fake_state_dict(index), grad_step=index * 1000, elo=1000.0)
    pool.record_result(MatchResult.create(0, 0.0, 0.0, 1.0))  # a lopsided history

    weights = pool.pfsp_weights()

    assert set(weights) == {0, 1, 2}
    assert math.fsum(weights.values()) == pytest.approx(1.0)
    assert all(math.isfinite(value) and value > 0.0 for value in weights.values())
    # Snapshot 0 is the only one with a record, and it is 1-0: p = 2/3 sits
    # further from the winnable band than the prior's 0.5, where the two unplayed
    # snapshots sit, so it must draw LESS mass than they do.
    assert weights[0] < weights[1]
    assert weights[1] == pytest.approx(weights[2])
    assert messages == [], (
        "PFSP must produce a usable distribution by itself; a message here means "
        "_normalized_weights_locked fell back to uniform, and the numbers above "
        "then say nothing about the formula"
    )


def test_pfsp_weights_on_an_empty_pool_is_empty(tmp_path):
    """No members, no probabilities — and no division by zero."""
    assert _new_pool(tmp_path).pfsp_weights() == {}


def test_tc14_pfsp_weights_are_exact_for_a_fixed_stats_table(tmp_path):
    """The formula to the digit, against a table worked out by hand.

    A weighting that is merely monotone in the right direction can still be
    wrong — floor applied per candidate instead of per pool, ``f`` normalized
    twice, the prior re-applied on top of ``win_rate`` — and pass every "more
    mass on the middle" check. On a fixed table ``pfsp_weights()`` is
    deterministic and has exactly one right answer, so pin that.
    """
    pool = _new_pool(tmp_path)
    for index in range(4):
        pool.add(_fake_state_dict(index), grad_step=index * 1000, elo=1000.0)
    _record_history(pool, 1, plays=8, wins=7)
    _record_history(pool, 2, plays=8, wins=1)
    _record_history(pool, 3, plays=18, wins=17)

    # p = (wins + 1) / (plays + 2)      [MatchStats.win_rate, Beta(1, 1) prior]
    # w = p * (1 - p) + 0.05 / N,       N = 4  ->  floor = 0.0125
    #
    #   id 0:   0 plays          p = 1/2   = 0.5   f = 0.25   w = 0.2625
    #   id 1:   8 plays,  7 won  p = 8/10  = 0.8   f = 0.16   w = 0.1725
    #   id 2:   8 plays,  1 won  p = 2/10  = 0.2   f = 0.16   w = 0.1725
    #   id 3:  18 plays, 17 won  p = 18/20 = 0.9   f = 0.09   w = 0.1025
    #                                                   sum   = 0.71
    assert pool.stats_for(0).win_rate() == 0.5
    assert pool.stats_for(1).win_rate() == pytest.approx(0.8)
    assert pool.stats_for(2).win_rate() == pytest.approx(0.2)
    assert pool.stats_for(3).win_rate() == pytest.approx(0.9)

    weights = pool.pfsp_weights()

    assert weights == pytest.approx(
        {
            0: 0.2625 / 0.71,  # 0.3697183098591549
            1: 0.1725 / 0.71,  # 0.2429577464788732
            2: 0.1725 / 0.71,  # 0.2429577464788732
            3: 0.1025 / 0.71,  # 0.1443661971830986
        }
    )
    assert math.fsum(weights.values()) == pytest.approx(1.0)
    # The two 0.16 members are equally FAR from 0.5, on opposite sides: PFSP
    # weights by how informative the match-up is, not by who is winning it.
    assert weights[1] == pytest.approx(weights[2])


def test_tc15_a_zero_play_snapshot_gets_a_finite_normalized_weight(tmp_path):
    """A snapshot with no record is the MAXIMUM of f, never a NaN and never 0.

    Without the Beta prior ``p`` is 0/0 here. Guarded to zero it would be worse
    than a crash: the newest past self would never be sampled, so it could never
    earn the plays that would raise its weight — dead on arrival, silently.
    """
    pool = _new_pool(tmp_path)
    for index in range(3):
        pool.add(_fake_state_dict(index), grad_step=index * 1000, elo=1000.0)
    _record_history(pool, 0, plays=8, wins=7)  # p = 0.8 -> f = 0.16
    _record_history(pool, 1, plays=8, wins=1)  # p = 0.2 -> f = 0.16
    assert pool.stats_for(2) == MatchStats(), "snapshot 2 must be the unplayed one"

    weights = pool.pfsp_weights()

    assert set(weights) == {0, 1, 2}
    assert all(math.isfinite(value) for value in weights.values())
    assert not any(math.isnan(value) for value in weights.values())
    assert math.fsum(weights.values()) == pytest.approx(1.0)
    # floor = 0.05 / 3; raw = [0.16 + floor, 0.16 + floor, 0.25 + floor]
    floor = PFSP_FLOOR_MASS / 3.0
    total = (0.16 + floor) + (0.16 + floor) + (0.25 + floor)
    assert weights == pytest.approx(
        {
            0: (0.16 + floor) / total,
            1: (0.16 + floor) / total,
            2: (0.25 + floor) / total,
        }
    )
    assert weights[2] > weights[0] and weights[2] > weights[1]

    # And a pool where NOTHING has played is exactly flat: every member sits at
    # the prior, so no candidate is preferred and nothing divides by zero.
    fresh = SnapshotPool(str(tmp_path / "fresh"))
    for index in range(4):
        fresh.add(_fake_state_dict(index), grad_step=index, elo=1000.0)
    assert fresh.pfsp_weights() == pytest.approx({0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})


@pytest.mark.parametrize("extreme", [0.0, 1.0])
def test_tc16_the_floor_keeps_a_member_at_a_degenerate_win_rate_sampleable(
    tmp_path, monkeypatch, extreme
):
    """At p = 0 and p = 1 the shaping term is exactly 0 — only the floor is left.

    ``win_rate()``'s prior keeps the pool off these endpoints in practice, so
    they are forced here: this test is about the FLOOR, not about the prior. A
    zero-weight member is never sampled again and therefore never earns a result
    that could move its ``p`` back off the extreme — and the members that end up
    at the extremes are precisely the pinned references the learner has outgrown,
    which is how a self-play run forgets how to beat its oldest ancestors.
    """
    messages: List[str] = []
    pool = _new_pool(tmp_path, log=messages.append)
    for index in range(3):
        pool.add(
            _fake_state_dict(index),
            grad_step=index * 1000,
            elo=1000.0,
            pinned=(index == 0),
        )
    monkeypatch.setattr(MatchStats, "win_rate", lambda self: extreme)
    assert extreme * (1.0 - extreme) == 0.0, "the shaping term contributes nothing"

    weights = pool.pfsp_weights()

    assert [record.snapshot_id for record in pool.pinned_references()] == [0]
    assert weights[0] > 0.0, "a pinned member must never reach probability zero"
    assert all(value > 0.0 for value in weights.values())
    # All three sit on the floor alone, so the floor normalizes to uniform.
    assert weights == pytest.approx({0: 1 / 3, 1: 1 / 3, 2: 1 / 3})
    assert messages == [], (
        "the floor must make the distribution usable on its own — a fallback "
        "message here means the raw weights summed to zero"
    )
    # And it is reachable in practice, not just non-zero on paper.
    rng = random.Random(0)
    assert {pool.sample(rng).snapshot_id for _ in range(200)} == {0, 1, 2}


def test_ac6_pfsp_concentrates_mass_on_the_winnable_band_versus_uniform(tmp_path):
    """AC6: measurably more probability in the 0.4-0.6 win-rate band than uniform.

    Ten snapshots spanning p = 0.05 .. 0.95 in 0.10 steps, the SAME history in
    both arms, and the only difference is the sampling mode. This is the point
    of issue #9: matches the learner can still learn from, instead of a flat draw
    that spends most of its episodes on opponents it always beats or never does.
    """
    ladder = [(18, wins) for wins in range(0, 19, 2)]  # p = (wins+1)/20
    arms: Dict[str, Dict[int, float]] = {}
    for mode in ("pfsp", "uniform"):
        pool = SnapshotPool(str(tmp_path / mode), sampling=mode)
        for index, (plays, wins) in enumerate(ladder):
            pool.add(_fake_state_dict(index), grad_step=index * 1000, elo=1000.0)
            _record_history(pool, index, plays=plays, wins=wins)
        arms[mode] = pool.pfsp_weights()

    band = {
        index
        for index, (plays, wins) in enumerate(ladder)
        if 0.4 <= (wins + 1.0) / (plays + 2.0) <= 0.6
    }
    tails = {0, len(ladder) - 1}  # p = 0.05 and p = 0.95
    assert band == {4, 5}, "the fixture must actually span the band"

    pfsp_band = math.fsum(arms["pfsp"][index] for index in band)
    uniform_band = math.fsum(arms["uniform"][index] for index in band)

    # sum f(p) over the ladder = 1.675; floors add 10 * 0.005 = 0.05
    # band raw = 2 * (0.2475 + 0.005) = 0.505  ->  0.505 / 1.725 = 0.29275...
    assert uniform_band == pytest.approx(0.2)
    assert pfsp_band == pytest.approx(0.505 / 1.725)
    assert pfsp_band > uniform_band * 1.4, (
        f"PFSP put {pfsp_band:.4f} on the winnable band against uniform's "
        f"{uniform_band:.4f} — not a measurable concentration"
    )
    # ...and it comes out of the tails, not out of thin air.
    assert math.fsum(arms["pfsp"][index] for index in tails) < math.fsum(
        arms["uniform"][index] for index in tails
    )
    for weights in arms.values():
        assert math.fsum(weights.values()) == pytest.approx(1.0)
        assert all(value > 0.0 for value in weights.values())


def test_uniform_mode_is_exactly_the_pre_pfsp_behaviour(tmp_path):
    """The A/B baseline: flat weights and the same draws, whatever the history.

    If ``uniform`` drifted even slightly, the AC6 comparison would be measuring
    two versions of PFSP against each other.
    """
    pool = SnapshotPool(str(tmp_path / "snapshots"), sampling="uniform")
    for index in range(4):
        pool.add(_fake_state_dict(index), grad_step=index * 1000, elo=1000.0)
    _record_history(pool, 0, plays=8, wins=8)  # a maximally lopsided field
    _record_history(pool, 1, plays=8, wins=0)
    _record_history(pool, 2, plays=3, wins=2)

    assert pool.sampling == "uniform"
    # Exact equality, not approx: 1/4 is exact in binary, and so is 1.0/4.0.
    assert pool.pfsp_weights() == {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}

    # The draws themselves match the inverse-CDF over a flat distribution, drawn
    # from an independent copy of the same seeded stream.
    records = pool.records()
    pool_rng, reference_rng = random.Random(99), random.Random(99)
    for _ in range(64):
        drawn = pool.sample(pool_rng)
        expected = records[int(reference_rng.random() * len(records))]
        assert drawn.snapshot_id == expected.snapshot_id


def test_the_default_mode_is_pfsp_and_an_unknown_mode_is_rejected(tmp_path):
    """A typo must fail at construction, not silently pick an arm."""
    assert DEFAULT_SAMPLING == "pfsp"
    assert SAMPLING_MODES == frozenset({"uniform", "pfsp"})
    assert _new_pool(tmp_path).sampling == "pfsp"

    with pytest.raises(ValueError, match="sampling must be one of"):
        SnapshotPool(str(tmp_path / "bad"), sampling="epsilon-greedy")
    # Validated before the index is even opened, so the error names the real
    # problem instead of blaming a malformed pool.json.
    with pytest.raises(ValueError, match="sampling must be one of"):
        SnapshotPool.load(str(tmp_path / "nonexistent"), sampling="PFSP")


def test_the_weight_fallback_warning_is_logged_outside_the_lock(
    tmp_path, monkeypatch
):
    """The ``log`` sink is caller code; calling it under ``_lock`` can deadlock.

    ``_lock`` is reentrant, so a callback that only re-enters the pool is fine
    and this path looks harmless — right up until a callback that takes an
    EXTERNAL lock meets a thread holding that lock and waiting on ``_lock``.
    Then the whole fleet wedges, mid-run, unreproducibly. ``_retire`` already
    logs after releasing; the weighting guard must too.

    T8's uniform weights made this branch dead code. PFSP makes it live, so the
    unusable-weights case is forced here to exercise it.
    """
    owned_at_log: List[bool] = []
    pool = _new_pool(
        tmp_path, log=lambda message: owned_at_log.append(pool._lock._is_owned())
    )
    for index in range(2):
        pool.add(_fake_state_dict(index), grad_step=index * 1000, elo=1000.0)
    monkeypatch.setattr(
        SnapshotPool,
        "_raw_weights_locked",
        lambda self, candidates: {rec.snapshot_id: float("nan") for rec in candidates},
    )

    assert pool.pfsp_weights() == pytest.approx({0: 0.5, 1: 0.5})
    assert owned_at_log == [False], (
        f"the fallback warning was logged with _lock held ({owned_at_log!r}); "
        "a caller-supplied callback must never run under the pool's lock"
    )

    owned_at_log.clear()
    pool.sample(random.Random(0))  # the guard's other caller
    assert owned_at_log == [False]


def test_load_takes_the_sampling_mode_from_the_caller_not_the_index(tmp_path):
    """A resumed ``uniform`` run must not come back as the ``pfsp`` arm.

    The regime is a run-config choice, so it is deliberately not stored in
    ``pool.json``; the resuming caller passes the value its own config carries.
    And it is REQUIRED rather than defaulted, because a default leaves the exact
    failure the parameter exists to prevent expressible by saying nothing: a
    resume that omits it silently restarts the ``uniform`` arm of the AC6
    comparison as the ``pfsp`` arm, and the comparison then measures nothing.
    """
    directory = str(tmp_path / "snapshots")
    pool = SnapshotPool(directory, sampling="uniform")
    pool.add(_fake_state_dict(), grad_step=0, elo=1000.0, pinned=True)
    pool.persist(pool.index_payload())

    # The mode round-trips through the explicit ARGUMENT, in both directions...
    assert SnapshotPool.load(directory, sampling="uniform").sampling == "uniform"
    assert SnapshotPool.load(directory, sampling="pfsp").sampling == "pfsp"
    # ...and omitting it is not expressible.
    with pytest.raises(TypeError, match="sampling"):
        SnapshotPool.load(directory)
    assert "sampling" not in json.loads(
        (tmp_path / "snapshots" / INDEX_FILENAME).read_text(encoding="utf-8")
    )


# ---------------------------------------------------------------------------
# Corruption policy -- pinned is fatal, unpinned is dropped
# ---------------------------------------------------------------------------


def test_missing_pinned_snapshot_is_fatal_at_sample(tmp_path):
    """Never resample around a lost reference: it would redefine the metric."""
    pool = _new_pool(tmp_path)
    record = pool.add(_fake_state_dict(), grad_step=0, elo=1000.0, pinned=True)
    os.remove(record.path)

    with pytest.raises(PinnedSnapshotError, match="PINNED snapshot 0"):
        pool.sample(random.Random(0))

    assert pool.records() == [record], "a pinned snapshot must never be dropped"


def test_corrupt_pinned_snapshot_is_fatal_at_load(tmp_path):
    """A truncated write survives the existence check and must still be fatal."""
    pool = _new_pool(tmp_path)
    record = pool.add(_fake_state_dict(), grad_step=0, elo=1000.0, pinned=True)
    with open(record.path, "wb") as handle:
        handle.write(b"not a torch archive")

    with pytest.raises(PinnedSnapshotError, match="PINNED snapshot 0"):
        pool.load_state_dict(record)
    with pytest.raises(PinnedSnapshotError):
        pool.sample_state_dict(random.Random(0))

    assert pool.records() == [record]


def test_missing_pinned_snapshot_is_fatal_at_pool_load(tmp_path):
    """Fail at startup, not a thousand episodes into the run."""
    pool = _new_pool(tmp_path)
    record = pool.add(_fake_state_dict(), grad_step=0, elo=1000.0, pinned=True)
    os.remove(record.path)

    with pytest.raises(PinnedSnapshotError, match="PINNED snapshot 0"):
        SnapshotPool.load(str(tmp_path / "snapshots"), sampling=DEFAULT_SAMPLING)


def test_missing_unpinned_snapshot_is_dropped_and_resampled(tmp_path):
    """Loudly drop it, then keep playing — but never against an untrained net."""
    messages: List[str] = []
    pool = _new_pool(tmp_path, log=messages.append)
    reference = pool.add(_fake_state_dict(1.0), grad_step=0, elo=1000.0, pinned=True)
    doomed = pool.add(_fake_state_dict(2.0), grad_step=1000, elo=1000.0)
    os.remove(doomed.path)

    rng = random.Random(3)
    assert {pool.sample(rng).snapshot_id for _ in range(64)} == {
        reference.snapshot_id
    }
    assert pool.records() == [reference]
    assert len(pool) == 1
    assert any("DROPPED unpinned snapshot 1" in message for message in messages), (
        f"the drop must be loud; got {messages!r}"
    )


def test_corrupt_unpinned_snapshot_is_dropped_by_the_loader(tmp_path):
    """``sample_state_dict`` retries past a member that only fails on read."""
    messages: List[str] = []
    pool = _new_pool(tmp_path, log=messages.append)
    reference = pool.add(_fake_state_dict(1.0), grad_step=0, elo=1000.0, pinned=True)
    doomed = pool.add(_fake_state_dict(2.0), grad_step=1000, elo=1000.0)
    with open(doomed.path, "wb") as handle:
        handle.write(b"\x00\x01\x02truncated")

    with pytest.raises(SnapshotUnavailableError, match="snapshot 1"):
        pool.load_state_dict(doomed)
    assert pool.records() == [reference]

    record, state_dict = pool.sample_state_dict(random.Random(11))
    assert record == reference
    assert torch.equal(state_dict["weight"], torch.full((4,), 1.0))
    assert any("DROPPED unpinned snapshot 1" in message for message in messages)


def test_a_swapped_snapshot_file_is_treated_as_corruption(tmp_path):
    """An id mismatch means the file holds a different version than the index says."""
    pool = _new_pool(tmp_path)
    reference = pool.add(_fake_state_dict(1.0), grad_step=0, elo=1000.0, pinned=True)
    victim = pool.add(_fake_state_dict(2.0), grad_step=1000, elo=1000.0)

    payload = pool.load_payload(reference)
    torch.save(payload, victim.path)  # snapshot 0's payload under snapshot 1's name

    with pytest.raises(SnapshotUnavailableError, match="snapshot_id"):
        pool.load_state_dict(victim)
    assert pool.records() == [reference]


def test_a_dropped_snapshot_can_still_rate_an_in_flight_match(tmp_path):
    """A match already under way when its opponent is dropped is still rated.

    The frozen rating is remembered, so the result is neither discarded nor
    rated against a guessed number.
    """
    pool = _new_pool(tmp_path, log=lambda message: None)
    pool.add(_fake_state_dict(1.0), grad_step=0, elo=1000.0, pinned=True)
    doomed = pool.add(_fake_state_dict(2.0), grad_step=1000, elo=1000.0)
    os.remove(doomed.path)
    pool.sample(random.Random(5), exclude_id=0)  # triggers the drop

    pool.record_result(MatchResult.create(doomed.snapshot_id, 0.0, 0.0, 1.0))

    assert pool.learner_elo_online == pytest.approx(1012.0)
    assert pool.stats_for(doomed.snapshot_id) == MatchStats(plays=1, learner_wins=1.0)
    # Dropped members survive the round-trip, so a restart cannot resurrect the
    # id and hand it to a different policy version.
    pool.persist(pool.index_payload())
    reloaded = SnapshotPool.load(
        str(tmp_path / "snapshots"), sampling=DEFAULT_SAMPLING
    )
    assert reloaded.index_payload() == pool.index_payload()
    assert reloaded.add(_fake_state_dict(3.0), 2000, 1000.0).snapshot_id == 2


# ---------------------------------------------------------------------------
# Thread safety -- one lock over registry/stats/Elo, disk I/O outside it
# ---------------------------------------------------------------------------


def test_concurrent_scoring_loses_no_match(tmp_path):
    """8 arena threads x 50 results each: every match lands, none interleaves away."""
    pool = _new_pool(tmp_path)
    record = pool.add(_fake_state_dict(), grad_step=0, elo=1000.0, pinned=True)

    def worker() -> None:
        for _ in range(50):
            pool.record_result(MatchResult.create(record.snapshot_id, 0.1, 0.02, 1.0))

    _run_threads([worker] * 8)

    assert pool.stats_for(record.snapshot_id) == MatchStats(
        plays=400, learner_wins=400.0
    )
    assert pool.learner_elo_online > ELO_INITIAL
    assert pool.learner_elo_rated == ELO_INITIAL


def test_sampling_keeps_working_while_the_learner_adds_snapshots(tmp_path):
    """The learner's writes (and their disk I/O) must not wedge the samplers."""
    pool = _new_pool(tmp_path)
    pool.add(_fake_state_dict(0.0), grad_step=0, elo=1000.0, pinned=True)
    failures: List[BaseException] = []

    def sampler(seed: int) -> Callable[[], None]:
        def run() -> None:
            rng = random.Random(seed)
            try:
                for _ in range(100):
                    record = pool.sample(rng, exclude_id=0)
                    pool.load_state_dict(record)
            except BaseException as exc:  # noqa: BLE001 - reported, not swallowed
                failures.append(exc)

        return run

    def writer() -> None:
        try:
            for step in range(1, 9):
                pool.add(_fake_state_dict(step), grad_step=step * 1000, elo=1000.0)
        except BaseException as exc:  # noqa: BLE001 - reported, not swallowed
            failures.append(exc)

    _run_threads([sampler(1), sampler(2), sampler(3), writer])

    assert failures == []
    assert len(pool) == 9
    assert [record.snapshot_id for record in pool.records()] == list(range(9))


def test_disk_io_never_runs_under_the_state_lock(tmp_path, monkeypatch):
    """Every disk touch must happen with ``_lock`` RELEASED — read AND write.

    This is the spec's headline concurrency property and the one nothing else
    here can see: move ``torch.load`` inside ``with self._lock:`` and every other
    test in this file still passes, while 25 arena threads quietly serialize
    behind one thread's disk read. The symptom is a throughput number, and
    nobody goes looking for a lock bug in a throughput number.

    So it is checked mechanically: both I/O primitives record whether the
    CALLING thread owns ``_lock`` at the moment they are entered, and every
    record must say it does not.
    """
    pool = _new_pool(tmp_path)
    assert hasattr(pool._lock, "_is_owned"), (
        "this test reads ownership off threading.RLock._is_owned(); without it "
        "there is nothing to assert on"
    )

    io_calls: List[Tuple[str, bool]] = []
    real_torch_load = torch.load
    real_atomic_write = snapshot_pool._atomic_write

    def spy_load(*args, **kwargs):
        io_calls.append(("torch.load", pool._lock._is_owned()))
        return real_torch_load(*args, **kwargs)

    def spy_write(path, write_bytes):
        io_calls.append((f"write:{os.path.basename(path)}", pool._lock._is_owned()))
        return real_atomic_write(path, write_bytes)

    monkeypatch.setattr(torch, "load", spy_load)
    monkeypatch.setattr(snapshot_pool, "_atomic_write", spy_write)

    # Write path: the snapshot file and the index, through add() and persist().
    record = pool.add(_fake_state_dict(1.0), grad_step=0, elo=1000.0, pinned=True)
    pool.add(_fake_state_dict(2.0), grad_step=1000, elo=1000.0)
    pool.persist(pool.index_payload())
    # Read path: all three doors onto torch.load.
    pool.load_payload(record)
    pool.load_state_dict(record)
    pool.sample_state_dict(random.Random(0))

    assert [name for name, owned in io_calls if owned] == [], (
        f"disk I/O ran while holding the pool's state lock: {io_calls!r}. "
        "Every arena thread now queues behind that call."
    )
    observed = {name for name, _ in io_calls}
    assert "torch.load" in observed, "the read path never reached the spy"
    assert "write:snap_0.pt" in observed, "the snapshot write never reached the spy"
    assert f"write:{INDEX_FILENAME}" in observed, "the index write never reached it"


def test_persist_never_needs_the_state_lock(tmp_path):
    """A worker must be able to persist while another thread holds ``_lock``.

    ``persist`` is handed its payload precisely so that it acquires ``_io_lock``
    and nothing else: a thread inside it waits for no other lock, so no cycle
    with ``_lock`` can form no matter what a caller holds. Under the older shape
    — ``persist`` taking ``_io_lock`` and then building the index under ``_lock``
    — this worker blocks for as long as the main thread holds ``_lock``, which is
    one half of an ABBA deadlock. ``_run_threads``' join timeout turns that hang
    into a test failure instead of a wedged run.
    """
    directory = str(tmp_path / "snapshots")
    pool = _new_pool(tmp_path)
    pool.add(_fake_state_dict(1.0), grad_step=0, elo=1000.0, pinned=True)
    payload = pool.index_payload()
    finished: List[bool] = []

    def writer() -> None:
        pool.persist(payload)
        finished.append(True)

    with pool._lock:  # held for the whole of the worker's run
        _run_threads([writer])

    assert finished == [True], "the worker did not complete its persist"
    assert (
        SnapshotPool.load(directory, sampling=DEFAULT_SAMPLING).index_payload()
        == payload
    )


def test_persist_rejects_something_that_is_not_an_index_payload(tmp_path):
    """``persist(pool.index_payload)`` with the parentheses lost must be loud.

    Silently writing a bound method's repr — or nothing at all — would leave the
    run with an index that does not describe the pool.
    """
    pool = _new_pool(tmp_path)
    pool.add(_fake_state_dict(), grad_step=0, elo=1000.0, pinned=True)

    with pytest.raises(TypeError, match="index_payload"):
        pool.persist(pool.index_payload)  # noqa: B010 - the missing () IS the case

    # The good index the previous add() wrote is still there, unharmed.
    assert (
        SnapshotPool.load(
            str(tmp_path / "snapshots"), sampling=DEFAULT_SAMPLING
        ).records()
        == pool.records()
    )


# ---------------------------------------------------------------------------
# Index freshness -- persist never writes an older payload over a newer one
# ---------------------------------------------------------------------------


def test_persist_refuses_a_payload_older_than_the_last_write(tmp_path):
    """Persisted out of order, the NEWER payload is the one left on disk.

    Constructed directly rather than raced: the stale case is the whole point,
    and a test that has to win a scheduling race to produce it would go on
    passing on the day the guard is deleted.
    """
    messages: List[str] = []
    directory = str(tmp_path / "snapshots")
    pool = SnapshotPool(directory, log=messages.append)
    pool.add(_fake_state_dict(1.0), grad_step=0, elo=1000.0, pinned=True)
    stale = pool.index_payload()  # one snapshot, next_snapshot_id 1
    pool.add(_fake_state_dict(2.0), grad_step=1000, elo=1000.0)
    fresh = pool.index_payload()  # two snapshots, next_snapshot_id 2

    assert fresh["index_generation"] > stale["index_generation"], (
        "the fixture must actually age the payload, or this test proves nothing"
    )
    assert (stale["next_snapshot_id"], fresh["next_snapshot_id"]) == (1, 2)

    pool.persist(stale)  # `fresh` is already on disk: the second add() wrote it

    on_disk = json.loads(
        (tmp_path / "snapshots" / INDEX_FILENAME).read_text(encoding="utf-8")
    )
    assert on_disk == fresh
    assert [entry["snapshot_id"] for entry in on_disk["snapshots"]] == [0, 1]
    assert on_disk["next_snapshot_id"] == 2
    # Loud, not silent: a write dropped where nobody hears it is an index entry
    # missing for a reason nobody can reconstruct later.
    assert len(messages) == 1, (
        f"expected exactly one stale-write log, got {messages!r}"
    )
    assert "stale index write" in messages[0]
    assert str(stale["index_generation"]) in messages[0]


def test_a_stale_persist_cannot_regress_the_id_counter(tmp_path):
    """No id is ever handed out twice, even after a payload lands out of order.

    ``load``'s ``max(known ids) + 1`` floor cannot catch this by itself: it reads
    the ids out of whatever index is on disk, so against a STALE index it reports
    the stale counter straight back and protects nothing. Refusing the backwards
    write is what stops the restart from re-issuing a live id — which would
    ``os.replace`` that snapshot's weights with a different policy version while
    the new version inherits its MatchStats, its Elo and, for a pinned member,
    the meaning of ``selfplay/win_rate_vs_ref_<id>`` mid-run.
    """
    directory = str(tmp_path / "snapshots")
    pool = SnapshotPool(directory, log=lambda message: None)
    pool.add(_fake_state_dict(1.0), grad_step=0, elo=1000.0, pinned=True)
    stale = pool.index_payload()
    second = pool.add(_fake_state_dict(2.0), grad_step=1000, elo=1000.0)

    pool.persist(stale)

    reloaded = SnapshotPool.load(directory, sampling=DEFAULT_SAMPLING)
    assert (
        reloaded.index_payload()["index_generation"]
        == pool.index_payload()["index_generation"]
    ), "the stamp must survive the reload, or it restarts under a later index"
    assert [record.snapshot_id for record in reloaded.records()] == [0, 1]

    assert reloaded.add(_fake_state_dict(3.0), 2000, 1000.0).snapshot_id == 2, (
        "a reused id makes two policy versions answer to one identity"
    )
    # And snapshot 1 still holds snapshot 1's weights: nothing overwrote its file.
    assert torch.equal(
        reloaded.load_state_dict(reloaded.get(second.snapshot_id))["weight"],
        torch.full((4,), 2.0),
    )


def test_persist_still_writes_a_payload_of_the_same_generation(tmp_path):
    """Only a STRICTLY older payload is dropped.

    Two payloads of one generation describe identical state, so writing either is
    correct, and a caller persisting on a cadence with nothing new to say must
    still get a file. Made observable by deleting the index in between: under a
    ``<=`` comparison the pool would silently end up with no index at all.
    """
    messages: List[str] = []
    directory = str(tmp_path / "snapshots")
    pool = SnapshotPool(directory, log=messages.append)
    pool.add(_fake_state_dict(1.0), grad_step=0, elo=1000.0, pinned=True)
    payload = pool.index_payload()
    pool.persist(payload)

    os.remove(pool.index_path)
    pool.persist(payload)

    assert os.path.isfile(pool.index_path)
    assert (
        SnapshotPool.load(directory, sampling=DEFAULT_SAMPLING).index_payload()
        == payload
    )
    assert messages == [], f"an equally fresh payload was refused: {messages!r}"


def test_persist_rejects_a_payload_with_no_generation_stamp(tmp_path):
    """An unstampable payload cannot be ordered, so it is refused, not written.

    Writing one would put an index of unknown age onto ``pool.json`` — the very
    thing the stamp prevents, reached through the back door of a hand-built dict.
    """
    pool = _new_pool(tmp_path)
    pool.add(_fake_state_dict(), grad_step=0, elo=1000.0, pinned=True)
    payload = dict(pool.index_payload())
    del payload["index_generation"]

    with pytest.raises(TypeError, match="index_generation"):
        pool.persist(payload)

    # The good index the add() wrote is still there, unharmed.
    assert (
        SnapshotPool.load(
            str(tmp_path / "snapshots"), sampling=DEFAULT_SAMPLING
        ).records()
        == pool.records()
    )


def test_a_payload_built_mid_add_cannot_orphan_the_snapshot_it_missed(
    tmp_path, monkeypatch
):
    """``add`` ages the index twice: allocating the id, then registering the record.

    Between those two points — the window in which the ``torch.save`` runs, with
    no lock held — ``index_payload()`` returns an index whose
    ``next_snapshot_id`` already counts the new snapshot while its ``snapshots``
    list does not yet contain it. If registering did not age the index, that
    payload and the finished one would share a generation, ``persist`` would rank
    them equal, and the mid-flight one could land last: ``snap_<id>.pt`` on disk
    with no index entry, which is an orphan no restart ever finds.
    """
    directory = str(tmp_path / "snapshots")
    pool = SnapshotPool(directory, log=lambda message: None)
    pool.add(_fake_state_dict(1.0), grad_step=0, elo=1000.0, pinned=True)

    mid_add: List[Dict[str, Any]] = []
    real_save = snapshot_pool._atomic_torch_save

    def spy_save(payload, path):
        # Entered from inside add(), after the id is allocated and before the
        # record is registered. That IS the window.
        mid_add.append(pool.index_payload())
        return real_save(payload, path)

    monkeypatch.setattr(snapshot_pool, "_atomic_torch_save", spy_save)
    pool.add(_fake_state_dict(2.0), grad_step=1000, elo=1000.0)
    monkeypatch.undo()

    assert len(mid_add) == 1, f"the spy never saw the window: {mid_add!r}"
    stale = mid_add[0]
    assert stale["next_snapshot_id"] == 2, "the id was already allocated..."
    assert [entry["snapshot_id"] for entry in stale["snapshots"]] == [0], (
        "...and the record was not registered yet"
    )

    pool.persist(stale)

    reloaded = SnapshotPool.load(directory, sampling=DEFAULT_SAMPLING)
    assert [record.snapshot_id for record in reloaded.records()] == [0, 1], (
        "a payload built mid-add overwrote the finished index and orphaned "
        "snap_1.pt"
    )


def test_the_generation_advances_on_every_change_the_index_carries(tmp_path):
    """Everything the payload serializes must age it, or the stamp is a half-order.

    The stamp means "equal generation, equal payload". A mutation the counter
    ignored would leave two DIFFERENT payloads sharing a generation, and
    ``persist`` — which drops only what is strictly older — would then be free to
    write the older one over the newer, one field down from the id counter.
    """
    pool = _new_pool(tmp_path, log=lambda message: None)

    def stamp() -> int:
        return int(pool.index_payload()["index_generation"])

    start = stamp()
    pool.add(_fake_state_dict(1.0), grad_step=0, elo=1000.0, pinned=True)
    doomed = pool.add(_fake_state_dict(2.0), grad_step=1000, elo=1000.0)
    after_adds = stamp()
    pool.record_result(MatchResult.create(0, 0.0, 0.0, 1.0))
    after_result = stamp()
    os.remove(doomed.path)
    pool.sample(random.Random(5), exclude_id=0)  # triggers the drop
    after_drop = stamp()

    assert start < after_adds, "a new snapshot must age the index"
    assert after_adds < after_result, "a scored match must age the index"
    assert after_result < after_drop, "a retired snapshot must age the index"
