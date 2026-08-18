"""Tests for the M4 self-play snapshot pool (T8; TC12, TC17, TC37).

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
  never reaches exactly 0 or 1. This is what stops T9's ``p(1 - p)`` weighting
  from being NaN, or zero, for a brand-new snapshot — a zero-weighted snapshot
  can never earn the plays that would raise its weight, so it would be dead on
  arrival.
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
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pytest

torch = pytest.importorskip("torch", exc_type=ImportError)

from opponents.snapshot_pool import (  # noqa: E402
    ELO_INITIAL,
    ELO_K,
    INDEX_FILENAME,
    MatchResult,
    MatchStats,
    PinnedSnapshotError,
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
    pool.persist()

    reloaded = SnapshotPool.load(str(tmp_path / "snapshots"))

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
    pool.persist()
    pool.persist()

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
        SnapshotPool.load(str(tmp_path / "snapshots"))


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
    # Without an exclusion, both versions are reachable (uniform in T8).
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
    """p(1-p) must stay positive, or T9 weights the snapshot out of existence."""
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
# pfsp_weights -- UNIFORM in T8, but always finite and normalized
# ---------------------------------------------------------------------------


def test_pfsp_weights_are_finite_normalized_and_cover_every_member(tmp_path):
    """The invariants AC6 needs, which T9's real formula must keep."""
    pool = _new_pool(tmp_path)
    for index in range(3):
        pool.add(_fake_state_dict(index), grad_step=index * 1000, elo=1000.0)
    pool.record_result(MatchResult.create(0, 0.0, 0.0, 1.0))  # a lopsided history

    weights = pool.pfsp_weights()

    assert set(weights) == {0, 1, 2}
    assert math.fsum(weights.values()) == pytest.approx(1.0)
    assert all(math.isfinite(value) and value > 0.0 for value in weights.values())
    # T8 ships the uniform placeholder; T9 replaces _raw_weights_locked and this
    # assertion is expected to change with it.
    assert all(value == pytest.approx(1.0 / 3.0) for value in weights.values())


def test_pfsp_weights_on_an_empty_pool_is_empty(tmp_path):
    """No members, no probabilities — and no division by zero."""
    assert _new_pool(tmp_path).pfsp_weights() == {}


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
        SnapshotPool.load(str(tmp_path / "snapshots"))


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
    pool.persist()
    reloaded = SnapshotPool.load(str(tmp_path / "snapshots"))
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
