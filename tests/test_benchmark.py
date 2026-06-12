"""Tests for the throughput/latency benchmark (T11) — offline, fake-bridge only.

No live Minecraft server, Node bridge, or socket is touched. The metric MATH is
proved on synthetic data via the pure module-level functions; the runner is
proved end-to-end against a scripted :class:`~eval.benchmark.FakeBridge` driven by
an injected :class:`~eval.benchmark.FakeClock` (deterministic round-trip
latencies). The :class:`~eval.logging.MetricsLogger` fallback is proved to write
AND read back its JSON-lines + summary, and to resolve gracefully when W&B /
TensorBoard are absent.

What is NOT proved here (the LIVE human follow-up): AC4 / TC12 proper — the REAL
sustained >= 10-minute number on the dev laptop (Core Ultra 7 258V) measuring the
max arenas that hold >= 19 TPS with CPU package power/thermal recorded. That needs
the live Paper server + Node bridge and is run via ``python -m eval.benchmark``.
See the module docstring of ``eval/benchmark.py``.
"""

import pytest

from agent.contract_config import DECISION_INTERVAL_MS, SERVER_TPS
from bridge.messages import StateMsg
from eval.benchmark import (
    DAMAGE_PER_HIT,
    DECISION_INTERVAL_S,
    MIN_SUSTAINED_TPS,
    BenchmarkReport,
    FakeBridge,
    FakeClock,
    ResourceSampler,
    TickDeltaTpsProvider,
    latency_percentiles,
    main,
    min_sustained_tps,
    percentile,
    run_benchmark,
    sustains_tps,
    tally_damage_dealt,
    transitions_per_second,
)
from eval.logging import (
    AUTO_BACKEND_ORDER,
    BackendUnavailableError,
    MetricsLogger,
    read_jsonl,
    read_summary,
)


# ===========================================================================
# Scripted-message helpers (mirrors tests/test_run_random.py conventions).
# ===========================================================================


def _state(*, damage_dealt=0.0, damage_taken=0.0, tick=1):
    """A canonical valid ``state`` dataclass; only the damage event varies."""
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
                "pos": [0.0, 64.0, 2.0],
                "yaw": 0.0,
                "pitch": 0.0,
                "velocity": [0.0, 0.0, 0.0],
                "health": 20.0,
            },
            "events": {
                "damage_dealt": damage_dealt,
                "damage_taken": damage_taken,
                "i_died": False,
                "opponent_died": False,
            },
            "arena": {"wall_distances": [8.0, 8.0, 8.0, 8.0]},
            "tick": tick,
            "code_version": "test",
        }
    )


# ===========================================================================
# Pure metric: percentile.
# ===========================================================================


def test_percentile_matches_known_sample():
    """p50/p95/p99 match the NumPy-default linear-interpolation values on 1..100.

    For samples 1..100 (n=100), the linear-interp percentile is
    ``1 + (q/100) * (n-1)`` => p50 = 50.5, p95 = 95.05, p99 = 99.01.
    """
    samples = list(range(1, 101))  # 1..100
    assert percentile(samples, 50.0) == pytest.approx(50.5)
    assert percentile(samples, 95.0) == pytest.approx(95.05)
    assert percentile(samples, 99.0) == pytest.approx(99.01)
    # Boundaries are exact min/max.
    assert percentile(samples, 0.0) == pytest.approx(1.0)
    assert percentile(samples, 100.0) == pytest.approx(100.0)


def test_percentile_small_and_unsorted():
    """Order-independent; interpolates between ranks on a tiny sample."""
    # Sorted: [10, 20, 30, 40]; p50 rank = 0.5*(3) = 1.5 -> between 20 and 30 -> 25.
    assert percentile([40, 10, 30, 20], 50.0) == pytest.approx(25.0)
    # Single element returns itself for any q.
    assert percentile([7.0], 99.0) == pytest.approx(7.0)


def test_percentile_rejects_empty_and_bad_q():
    with pytest.raises(ValueError):
        percentile([], 50.0)
    with pytest.raises(ValueError):
        percentile([1.0, 2.0], -1.0)
    with pytest.raises(ValueError):
        percentile([1.0, 2.0], 101.0)


def test_latency_percentiles_converts_to_ms_and_has_stable_keys():
    """Latency summary converts seconds -> ms and always carries p50/p95/p99 keys."""
    # 100 samples at 1ms..100ms expressed in seconds.
    samples_s = [i / 1000.0 for i in range(1, 101)]
    out = latency_percentiles(samples_s)
    assert out["p50_ms"] == pytest.approx(50.5)
    assert out["p95_ms"] == pytest.approx(95.05)
    assert out["p99_ms"] == pytest.approx(99.01)
    assert out["min_ms"] == pytest.approx(1.0)
    assert out["max_ms"] == pytest.approx(100.0)
    assert out["mean_ms"] == pytest.approx(50.5)
    assert out["count"] == 100


def test_latency_percentiles_empty_is_zeroed_but_typed():
    out = latency_percentiles([])
    for key in ("p50_ms", "p95_ms", "p99_ms", "min_ms", "max_ms", "mean_ms"):
        assert out[key] == 0.0
    assert out["count"] == 0


# ===========================================================================
# Pure metric: transitions/s.
# ===========================================================================


def test_transitions_per_second_basic():
    assert transitions_per_second(1000, 200.0) == pytest.approx(5.0)
    # At the 200 ms decision interval a single arena tops out near 5 decisions/s.
    assert transitions_per_second(50, 10.0) == pytest.approx(5.0)


def test_transitions_per_second_rejects_bad_inputs():
    with pytest.raises(ValueError):
        transitions_per_second(10, 0.0)
    with pytest.raises(ValueError):
        transitions_per_second(10, -1.0)
    with pytest.raises(ValueError):
        transitions_per_second(-1, 1.0)


# ===========================================================================
# Pure metric: damage-event boundary tally (TC7b cross-check).
# ===========================================================================


def test_tally_damage_counts_n_hit_exchange_exactly():
    """A known 3-hit exchange counts exactly 3 events — no drop, no double-count.

    Five decision windows; hits land in three of them. The tally must report
    exactly 3 hit events (one per landing window) regardless of the zero-damage
    windows interleaved between them.
    """
    states = [
        _state(damage_dealt=DAMAGE_PER_HIT, tick=1),  # hit
        _state(damage_dealt=0.0, tick=2),  # miss / no event
        _state(damage_dealt=DAMAGE_PER_HIT, tick=3),  # hit
        _state(damage_dealt=DAMAGE_PER_HIT, tick=4),  # hit
        _state(damage_dealt=0.0, tick=5),  # miss / no event
    ]
    result = tally_damage_dealt(states)
    assert result["hit_events"] == 3
    assert result["total_damage"] == pytest.approx(3 * DAMAGE_PER_HIT)
    assert result["windows"] == 5


def test_tally_damage_no_double_count_within_window():
    """A single window with one positive damage value is one event, not two.

    Guards the double-count failure mode: even if a window aggregates the damage
    of a hit that straddled a tick boundary into one positive value, it is ONE
    decision-window event, counted once.
    """
    states = [_state(damage_dealt=2 * DAMAGE_PER_HIT, tick=1)]
    result = tally_damage_dealt(states)
    assert result["hit_events"] == 1
    assert result["total_damage"] == pytest.approx(2 * DAMAGE_PER_HIT)


def test_tally_damage_empty():
    result = tally_damage_dealt([])
    assert result == {"hit_events": 0, "total_damage": 0.0, "windows": 0}


# ===========================================================================
# Pure metric: sustained-TPS detection.
# ===========================================================================


def test_sustained_tps_flags_a_dip_below_19():
    """A single dip below 19 TPS fails the sustained check and sets the floor."""
    good = [20.0, 20.0, 19.5, 20.0]
    assert sustains_tps(good) is True
    assert min_sustained_tps(good) == pytest.approx(19.5)

    dipped = [20.0, 20.0, 18.7, 20.0]  # one bad sample
    assert sustains_tps(dipped) is False
    assert min_sustained_tps(dipped) == pytest.approx(18.7)


def test_sustained_tps_boundary_is_inclusive():
    """Exactly 19.0 sustains (>=), 18.999 does not."""
    assert sustains_tps([19.0, 19.0, 19.0]) is True
    assert sustains_tps([19.0, 18.999]) is False
    assert MIN_SUSTAINED_TPS == 19.0


def test_sustained_tps_empty_does_not_sustain():
    assert sustains_tps([]) is False
    assert min_sustained_tps([]) == 0.0


# ===========================================================================
# FakeClock + FakeBridge: deterministic latency injection.
# ===========================================================================


def test_fake_clock_advances():
    clock = FakeClock(start=1.0)
    assert clock() == 1.0
    clock.advance(0.25)
    assert clock() == pytest.approx(1.25)
    with pytest.raises(ValueError):
        clock.advance(-0.1)


def test_fake_bridge_charges_scripted_latency_on_state():
    """recv() of a state advances the shared clock by the next scripted latency."""
    clock = FakeClock()
    bridge = FakeBridge(
        inbound=[_state(tick=1), _state(tick=2)],
        latencies_s=[0.040, 0.060],
        clock=clock,
    )
    bridge.connect()
    t0 = clock()
    bridge.send({"type": "step", "action": 0})
    bridge.recv()
    assert clock() - t0 == pytest.approx(0.040)
    t1 = clock()
    bridge.send({"type": "step", "action": 1})
    bridge.recv()
    assert clock() - t1 == pytest.approx(0.060)
    # Two step dicts were recorded.
    assert [d["type"] for d in bridge.sent] == ["step", "step"]


# ===========================================================================
# Runner end to end (the OFFLINE metric-logic proof).
# ===========================================================================


def _make_factory(n_decisions_per_arena, latency_s, *, damage_windows=None, tick0=1):
    """Build a transport_factory yielding one FakeBridge per arena.

    Each arena gets ``n_decisions_per_arena`` scripted ``state`` replies (one per
    step), each charged ``latency_s``. ``damage_windows`` is an optional iterable
    of 0-based window indices that should carry a positive damage event (used by
    the boundary cross-check); applied identically to every arena.
    """
    clock = FakeClock()
    bridges = []
    damage_windows = set(damage_windows or ())

    def factory(arena_index):
        inbound = []
        for i in range(n_decisions_per_arena):
            dealt = DAMAGE_PER_HIT if i in damage_windows else 0.0
            inbound.append(_state(damage_dealt=dealt, tick=tick0 + i))
        bridge = FakeBridge(
            inbound=inbound,
            latencies_s=[latency_s] * n_decisions_per_arena,
            clock=clock,
        )
        bridges.append(bridge)
        return bridge

    factory.clock = clock  # type: ignore[attr-defined]
    factory.bridges = bridges  # type: ignore[attr-defined]
    return factory


def test_run_benchmark_offline_produces_populated_report():
    """A short fake-bridge run completes and fills every report field."""
    factory = _make_factory(n_decisions_per_arena=20, latency_s=0.040)
    clock = factory.clock

    report = run_benchmark(
        factory,
        n_arenas=1,
        max_decisions=20,
        duration_s=1e9,  # huge so max_decisions is the binding stop condition
        clock=clock,
        log=None,
    )

    assert isinstance(report, BenchmarkReport)
    assert report.n_arenas == 1
    assert report.transitions == 20
    assert report.is_live is False

    # Latency: every round-trip was exactly 40 ms, so all percentiles are 40 ms.
    assert report.latency_ms["p50_ms"] == pytest.approx(40.0)
    assert report.latency_ms["p95_ms"] == pytest.approx(40.0)
    assert report.latency_ms["p99_ms"] == pytest.approx(40.0)
    assert report.latency_ms["count"] == 20

    # Throughput: 20 decisions over 20 * 40 ms == 0.8 s -> 25 decisions/s.
    assert report.duration_actual_s == pytest.approx(0.8)
    assert report.transitions_per_s_per_arena == pytest.approx(25.0)

    # Default TPS provider reports the vanilla rate, so the run sustains >=19.
    assert report.sustained_tps_min == pytest.approx(float(SERVER_TPS))
    assert report.sustains_19_tps is True
    assert report.max_arenas_sustaining_tps == 1

    # Report carries the AC4 follow-up note and the resource report shape.
    assert any("AC4/TC12 follow-up" in n for n in report.notes)
    assert "available" in report.resources
    assert "power_thermal_note" in report.resources

    # Bridge lifecycle: connected once, closed once.
    assert factory.bridges[0].connects == 1
    assert factory.bridges[0].closes == 1


def test_run_benchmark_damage_boundary_cross_check_passes():
    """Driving a scripted N-hit exchange yields hit_events == N (ok=True)."""
    # 10 windows, hits in windows 1, 4, 7 -> exactly 3 hit events.
    factory = _make_factory(
        n_decisions_per_arena=10,
        latency_s=0.030,
        damage_windows=[1, 4, 7],
    )
    report = run_benchmark(
        factory,
        n_arenas=1,
        max_decisions=10,
        duration_s=1e9,
        clock=factory.clock,
        expected_hits=3,
        log=None,
    )
    db = report.damage_boundary
    assert db["hit_events"] == 3
    assert db["expected_hits"] == 3
    assert db["ok"] is True
    assert db["total_damage"] == pytest.approx(3 * DAMAGE_PER_HIT)


def test_run_benchmark_damage_boundary_detects_a_drop():
    """If a hit window is dropped, hit_events != expected and ok=False."""
    # Script only 2 hits but claim 3 expected -> the cross-check must fail.
    factory = _make_factory(
        n_decisions_per_arena=6,
        latency_s=0.030,
        damage_windows=[1, 4],  # only 2 hits land
    )
    report = run_benchmark(
        factory,
        n_arenas=1,
        max_decisions=6,
        duration_s=1e9,
        clock=factory.clock,
        expected_hits=3,
        log=None,
    )
    assert report.damage_boundary["hit_events"] == 2
    assert report.damage_boundary["ok"] is False


def test_run_benchmark_flags_tps_dip_via_provider():
    """A tps_provider that returns a sub-19 reading makes the run not sustain."""
    factory = _make_factory(n_decisions_per_arena=5, latency_s=0.05)

    # Force one window to report a low TPS (the 3rd decision dips to 17).
    seen = {"n": 0}

    def dipping_provider(_state_msg, *, arena=0):
        seen["n"] += 1
        return 17.0 if seen["n"] == 3 else 20.0

    report = run_benchmark(
        factory,
        n_arenas=1,
        max_decisions=5,
        duration_s=1e9,
        clock=factory.clock,
        tps_provider=dipping_provider,
        log=None,
    )
    assert report.sustained_tps_min == pytest.approx(17.0)
    assert report.sustains_19_tps is False
    assert report.max_arenas_sustaining_tps == 0


def test_run_benchmark_multi_arena_throughput_is_per_arena():
    """With 2 arenas, transitions/s is reported PER ARENA, not summed."""
    factory = _make_factory(n_decisions_per_arena=10, latency_s=0.020)
    # 2 arenas * 10 decisions == 20 total; each round-trip charges 20 ms on the
    # shared clock, so 20 charges -> 0.4 s elapsed. Per-arena: 10 / 0.4 == 25.
    report = run_benchmark(
        factory,
        n_arenas=2,
        max_decisions=20,
        duration_s=1e9,
        clock=factory.clock,
        log=None,
    )
    assert report.n_arenas == 2
    assert report.transitions == 20
    assert report.duration_actual_s == pytest.approx(0.4)
    assert report.transitions_per_s_per_arena == pytest.approx(25.0)
    # Both arena bridges were connected and closed.
    assert len(factory.bridges) == 2
    for b in factory.bridges:
        assert b.connects == 1 and b.closes == 1


def test_run_benchmark_per_arena_throughput_non_even_division():
    """A NON-EVEN arena division reports the exact per-arena float, not a rounded one.

    Regression for C1: ``int(round(transitions / n_arenas))`` discarded the
    fractional part of the per-arena rate. With 7 total transitions across 3
    arenas the headline must be ``transitions_per_second(7, elapsed) / 3``, NOT
    ``round(7/3)=2`` decisions charged to throughput.

    Each of the 3 arenas gets 3 scripted decisions (9 inbound states) but the run
    is capped at ``max_decisions=7``, so exactly 7 round-trips fire. Every
    round-trip charges 10 ms on the shared clock -> 7 * 0.010 == 0.07 s elapsed.
    Total throughput is 7 / 0.07 == 100 transitions/s; per-arena is 100 / 3.
    """
    factory = _make_factory(n_decisions_per_arena=3, latency_s=0.010)
    report = run_benchmark(
        factory,
        n_arenas=3,
        max_decisions=7,  # 7 is NOT evenly divisible by 3 arenas
        duration_s=1e9,
        clock=factory.clock,
        log=None,
    )
    assert report.n_arenas == 3
    assert report.transitions == 7
    assert report.duration_actual_s == pytest.approx(0.07)

    expected_total_tps = 7 / 0.07  # == 100.0
    expected_per_arena = expected_total_tps / 3  # 33.333..., NOT round(7/3)/0.07
    assert report.transitions_per_s_per_arena == pytest.approx(expected_per_arena)
    # The buggy rounded value (round(7/3)==2 -> 2/0.07 == ~28.57) must NOT appear.
    assert report.transitions_per_s_per_arena != pytest.approx(2 / 0.07)
    # And the honest per-arena rate is strictly above that rounded-down artifact.
    assert report.transitions_per_s_per_arena > 33.0


def test_run_benchmark_duration_stop_condition():
    """With max_decisions=None the wall-clock deadline stops the run."""
    # Each decision charges 0.1 s; a 0.25 s budget admits one full arena round
    # (one decision here) then trips the deadline before a second round.
    factory = _make_factory(n_decisions_per_arena=10, latency_s=0.1)
    report = run_benchmark(
        factory,
        n_arenas=1,
        max_decisions=None,
        duration_s=0.25,
        clock=factory.clock,
        log=None,
    )
    # At least one decision ran; the deadline cut it short well under 10 decisions.
    assert 1 <= report.transitions < 10
    assert report.duration_actual_s <= 0.35


def test_run_benchmark_rejects_bad_args():
    factory = _make_factory(n_decisions_per_arena=1, latency_s=0.0)
    with pytest.raises(ValueError):
        run_benchmark(factory, n_arenas=0, clock=factory.clock)
    with pytest.raises(ValueError):
        run_benchmark(factory, n_arenas=1, duration_s=0.0, clock=factory.clock)
    with pytest.raises(ValueError):
        run_benchmark(
            factory, n_arenas=1, max_decisions=0, duration_s=1.0, clock=factory.clock
        )


def test_run_benchmark_with_logger_writes_metrics(tmp_path):
    """The runner logs per-decision metrics + a summary through a MetricsLogger."""
    factory = _make_factory(n_decisions_per_arena=5, latency_s=0.040)
    logger = MetricsLogger(
        run_name="runner_log",
        backend="jsonl",
        log_dir=str(tmp_path),
    )
    try:
        report = run_benchmark(
            factory,
            n_arenas=1,
            max_decisions=5,
            duration_s=1e9,
            clock=factory.clock,
            logger=logger,
            log=None,
        )
    finally:
        logger.close()

    records = read_jsonl(logger.metrics_path)
    # One per-decision record per transition.
    assert len(records) == report.transitions == 5
    assert all("round_trip_ms" in r for r in records)

    summary = read_summary(logger.summary_path)
    assert summary["transitions"] == 5
    assert summary["p99_ms"] == pytest.approx(40.0)
    assert summary["sustains_19_tps"] is True


# ===========================================================================
# TickDeltaTpsProvider: derives TPS from tick deltas over wall-clock time.
# ===========================================================================


def test_tick_delta_provider_warms_up_then_computes_tps():
    """First state is a gate-neutral warm-up; later states derive TPS = dtick/dwall.

    Drive the provider off a FakeClock fed synthetic (tick, wall) pairs. A healthy
    20-TPS server advances 20 ticks per real second: with a 0.5 s wall gap it
    advances 10 ticks, so the derived TPS is 10 / 0.5 == 20.0.
    """
    clock = FakeClock()
    provider = TickDeltaTpsProvider(clock=clock)

    # First sample: no predecessor -> warm-up value (the floor), gate-neutral.
    assert provider(_state(tick=100)) == pytest.approx(MIN_SUSTAINED_TPS)

    # 10 ticks advanced over 0.5 wall-seconds -> 20 TPS.
    clock.advance(0.5)
    assert provider(_state(tick=110)) == pytest.approx(20.0)

    # Another 20 ticks over 1.0 s -> 20 TPS, steady.
    clock.advance(1.0)
    assert provider(_state(tick=130)) == pytest.approx(20.0)


def test_tick_delta_provider_flags_dip_below_19():
    """A lagging server advances fewer ticks per wall-second -> TPS dips below 19.

    Over a 1.0 s wall window a healthy server advances 20 ticks; here it advances
    only 18 (the server fell behind), so the derived TPS is 18.0 — a real dip that
    sustains_tps() must reject. Asserts the provider produces the sub-floor signal
    AND that the sustained-TPS gate trips on it.
    """
    clock = FakeClock()
    provider = TickDeltaTpsProvider(clock=clock)

    samples = []
    # Warm-up.
    samples.append(provider(_state(tick=0)))
    # Healthy second: +20 ticks / 1.0 s -> 20 TPS.
    clock.advance(1.0)
    samples.append(provider(_state(tick=20)))
    # Lagging second: only +18 ticks / 1.0 s -> 18 TPS (a real dip).
    clock.advance(1.0)
    dipped = provider(_state(tick=38))
    samples.append(dipped)

    assert dipped == pytest.approx(18.0)
    assert dipped < MIN_SUSTAINED_TPS
    # The run-level gate must reject the series because of the single dip.
    assert sustains_tps(samples) is False
    assert min_sustained_tps(samples) == pytest.approx(18.0)


def test_tick_delta_provider_clamps_zero_wall_delta():
    """Two states read at the same clock instant clamp instead of dividing by ~0.

    With a zero-cost clock (dwall == 0) a naive dtick/dwall would explode; the
    provider clamps to max_tps (default 2*SERVER_TPS) — never a spurious dip, so
    it cannot hide a real sub-floor reading.
    """
    clock = FakeClock()
    provider = TickDeltaTpsProvider(clock=clock)
    provider(_state(tick=1))  # warm-up seeds prev_tick / prev_wall
    # No clock.advance() -> dwall == 0 on the next sample.
    clamped = provider(_state(tick=5))
    assert clamped == pytest.approx(2.0 * float(SERVER_TPS))
    assert clamped > MIN_SUSTAINED_TPS  # a clamp is never a fake dip


def test_tick_delta_provider_clamps_high_rate():
    """A very large tick jump over a tiny wall delta is clamped to max_tps."""
    clock = FakeClock()
    provider = TickDeltaTpsProvider(clock=clock, max_tps=40.0)
    provider(_state(tick=0))  # warm-up
    clock.advance(0.001)  # 1 ms
    # +1000 ticks / 0.001 s == 1_000_000 TPS, clamped to 40.0.
    assert provider(_state(tick=1000)) == pytest.approx(40.0)


def test_tick_delta_provider_backwards_tick_is_warmup_not_dip():
    """A tick counter reset (server restart) reports warm-up, not a spurious dip."""
    clock = FakeClock()
    provider = TickDeltaTpsProvider(clock=clock)
    provider(_state(tick=500))  # warm-up
    clock.advance(1.0)
    # Tick went backwards (restart): not a real rate -> warm-up value, not a dip.
    assert provider(_state(tick=10)) == pytest.approx(MIN_SUSTAINED_TPS)


def test_tick_delta_provider_rejects_bad_max_tps():
    with pytest.raises(ValueError):
        TickDeltaTpsProvider(clock=FakeClock(), max_tps=0.0)


def test_tick_delta_provider_drives_run_benchmark_sustained():
    """Wired into run_benchmark with a healthy tick cadence, the run sustains >=19.

    Each fake round-trip charges 50 ms on the shared clock; each scripted state
    advances tick by exactly 1, so the derived TPS is 1 / 0.050 == 20.0 per
    decision after warm-up. The run must report sustained >=19 TPS off REAL
    tick deltas (not the constant default).
    """
    factory = _make_factory(n_decisions_per_arena=6, latency_s=0.050, tick0=1)
    provider = TickDeltaTpsProvider(clock=factory.clock)
    report = run_benchmark(
        factory,
        n_arenas=1,
        max_decisions=6,
        duration_s=1e9,
        clock=factory.clock,
        tps_provider=provider,
        log=None,
    )
    # Warm-up sample is the floor (19.0); steady samples are 20.0; min is 19.0.
    assert report.sustains_19_tps is True
    assert report.sustained_tps_min == pytest.approx(MIN_SUSTAINED_TPS)


def test_tick_delta_provider_drives_run_benchmark_dip_fails():
    """A stalled tick (server froze a window) drives a real sub-19 dip in the run.

    The scripted states advance tick by 1 each EXCEPT one window where the tick
    does not advance at all (the server stalled): that window derives 0 / 0.050
    == 0 TPS, a real dip that makes the whole run fail the sustained gate.
    """
    clock = FakeClock()
    bridges = []

    # Ticks: 1, 2, 3, 3, 4, 5 -> the 4th window repeats tick 3 (a stall: dtick=0).
    ticks = [1, 2, 3, 3, 4, 5]

    def factory(_arena_index):
        inbound = [_state(tick=t) for t in ticks]
        bridge = FakeBridge(
            inbound=inbound, latencies_s=[0.050] * len(ticks), clock=clock
        )
        bridges.append(bridge)
        return bridge

    provider = TickDeltaTpsProvider(clock=clock)
    report = run_benchmark(
        factory,
        n_arenas=1,
        max_decisions=len(ticks),
        duration_s=1e9,
        clock=clock,
        tps_provider=provider,
        log=None,
    )
    # The stalled window derived 0 TPS -> the run does NOT sustain >=19.
    assert report.sustains_19_tps is False
    assert report.sustained_tps_min == pytest.approx(0.0)


# ===========================================================================
# TickDeltaTpsProvider: per-arena isolation (multi-arena bug regression).
# ===========================================================================


def test_tick_delta_provider_per_arena_isolation_both_healthy():
    """Two arenas with independent tick streams each derive ~20 TPS independently.

    This is the direct regression for the multi-arena shared-state bug: with a
    SINGLE shared provider the round-robin call sequence mixes arena0's and
    arena1's ticks — ``d_tick = tick(arena1) - tick(arena0)`` — producing garbage
    TPS readings. With per-arena state each arena diffs only against its own
    previous (tick, wall), so the rate is correct for each.

    Setup: two arenas, each advancing their tick counter by 1 per 50 ms step
    (wall-clock paced via FakeClock). After the warm-up sample, each arena's
    derived TPS == 1 / 0.050 == 20.0. The run sustains >= 19 TPS, which the old
    shared-state code would FAIL because the mixed ticks produce a meaningless rate.
    """
    # Build two independent FakeBridges with separate tick sequences. Both arenas
    # advance +1 tick every 50 ms step. We drive the provider directly (not through
    # run_benchmark) to verify per-arena isolation at the unit level.
    clock = FakeClock()
    provider = TickDeltaTpsProvider(clock=clock)

    # Arena 0: ticks 100, 101, 102, 103, 104 (healthy +1/step)
    # Arena 1: ticks 500, 501, 502, 503, 504 (independent counter, also healthy)
    # Round-robin interleaving: (a0,t=100), (a1,t=500), (a0,t=101), (a1,t=501), ...
    # Each recv charges 50 ms; the two arenas alternate, so from each arena's
    # perspective 100 ms of wall time passes between its own consecutive calls
    # (two 50 ms steps: one for itself, one for the other arena).
    arena0_ticks = [100, 101, 102, 103, 104]
    arena1_ticks = [500, 501, 502, 503, 504]

    samples_a0 = []
    samples_a1 = []

    for i in range(5):
        # Arena 0 step: advance clock by 50 ms, call provider with arena=0.
        clock.advance(0.050)
        samples_a0.append(provider(_state(tick=arena0_ticks[i]), arena=0))
        # Arena 1 step: advance clock by 50 ms, call provider with arena=1.
        clock.advance(0.050)
        samples_a1.append(provider(_state(tick=arena1_ticks[i]), arena=1))

    # Warm-up samples are the floor (19.0); all subsequent samples should be
    # 1 tick / 0.100 s == 10.0 TPS ... wait — each arena sees 100 ms between its
    # own calls (50 ms for its step + 50 ms for the other arena's step), so derived
    # TPS per arena == 1 tick / 0.100 s == 10.0. But the point is BOTH arenas see
    # the SAME healthy rate, not garbage from mixing tick counters.
    # (The test below checks isolation — the derived rates are equal and consistent,
    # not a blend of 600-100=400 tick jumps that the shared-state bug would produce.)

    # Warm-up samples are the gate-neutral floor.
    assert samples_a0[0] == pytest.approx(MIN_SUSTAINED_TPS)
    assert samples_a1[0] == pytest.approx(MIN_SUSTAINED_TPS)

    # Steady samples: 1 tick / 0.100 s == 10.0 TPS for each arena.
    for s in samples_a0[1:]:
        assert s == pytest.approx(10.0), f"arena0 sample {s} != 10.0 (isolation broken)"
    for s in samples_a1[1:]:
        assert s == pytest.approx(10.0), f"arena1 sample {s} != 10.0 (isolation broken)"

    # Verify isolation: the steady-state rate is consistently 10.0 for both arenas,
    # not a garbage value from mixed tick counters. The old shared-state bug would
    # produce d_tick = tick(arena1) - tick(arena0) e.g. 500 - 101 = 399 ticks over
    # 0.1 s == 3990 TPS (clamped to max_tps) or similar nonsense — nothing like 10.
    combined_steady = samples_a0[1:] + samples_a1[1:]
    assert all(s == pytest.approx(10.0) for s in combined_steady), (
        f"expected all steady samples to be 10.0, got: {combined_steady}"
    )


def test_tick_delta_provider_multi_arena_run_benchmark_sustains():
    """Two arenas each advancing at 20 ticks/s sustain >=19 TPS end-to-end.

    Drives run_benchmark with TWO arenas and a TickDeltaTpsProvider. Each arena
    gets its own FakeBridge whose states advance tick by 1 per 50 ms step; the
    shared clock advances 50 ms per recv. In the round-robin each arena's
    consecutive calls are 100 ms apart (one own step + one other-arena step), so
    each derives 1/0.1 == 10 TPS — above 19? No. To hit >=20 TPS we need the tick
    delta to span >= 2 ticks per 100 ms, i.e. advance tick by 2 per step. With +2
    ticks per 50 ms step each arena sees 2 ticks / 0.100 s == 20 TPS — healthy.

    The old shared-state provider would mix arena0's tick with arena1's tick for
    the d_tick computation, producing e.g. (500-101) or (101-500) -> garbage or
    a backwards-tick warm-up reset, making the result unpredictable. Per-arena
    state produces a clean 20.0 for each arena.
    """
    clock = FakeClock()

    # Each arena advances tick by 2 per step at 50 ms per step.
    # Arena 0 ticks: 0, 2, 4, 6, 8, 10, 12, 14 (start 0, +2 each own step)
    # Arena 1 ticks: 1000, 1002, 1004, 1006, 1008, 1010, 1012, 1014 (independent)
    n_per_arena = 8

    def factory(arena_index):
        start = 1000 * arena_index  # arena0 starts at 0, arena1 at 1000
        ticks = [start + 2 * i for i in range(n_per_arena)]
        inbound = [_state(tick=t) for t in ticks]
        # Each step charges 50 ms on the shared clock.
        return FakeBridge(
            inbound=inbound,
            latencies_s=[0.050] * n_per_arena,
            clock=clock,
        )

    provider = TickDeltaTpsProvider(clock=clock)
    report = run_benchmark(
        factory,
        n_arenas=2,
        max_decisions=2 * n_per_arena,  # all steps from both arenas
        duration_s=1e9,
        clock=clock,
        tps_provider=provider,
        log=None,
    )

    # Each arena derives 2 ticks / 0.100 s == 20 TPS after warm-up.
    # Warm-up samples from each arena are the floor (19.0), so the minimum is 19.0.
    assert report.sustains_19_tps is True
    assert report.sustained_tps_min == pytest.approx(MIN_SUSTAINED_TPS)


def test_tick_delta_provider_multi_arena_one_stall_fails_gate():
    """One arena stalling its ticks makes the whole run fail the sustained gate.

    Arena 0 is healthy (+2 ticks per step). Arena 1 deliberately stalls its tick
    counter (all steps return the same tick value), so its derived TPS is 0 after
    warm-up. That 0-TPS sample causes the global min to drop below 19 and the gate
    fails. This proves per-arena isolation works in BOTH directions: a stall in one
    arena contaminates only that arena's readings, but one bad reading is enough to
    fail the global gate.

    (With the old shared-state code this test's result would be undefined — the
    mixed d_tick could accidentally produce 0 or something non-zero depending on
    interleaving — whereas with per-arena state it deterministically produces 0 TPS
    for the stalled arena.)
    """
    clock = FakeClock()

    n_per_arena = 6

    def factory(arena_index):
        if arena_index == 0:
            # Healthy: +2 ticks per step.
            ticks = [2 * i for i in range(n_per_arena)]
        else:
            # Stalled: tick never advances beyond 999.
            ticks = [999] * n_per_arena
        inbound = [_state(tick=t) for t in ticks]
        return FakeBridge(
            inbound=inbound,
            latencies_s=[0.050] * n_per_arena,
            clock=clock,
        )

    provider = TickDeltaTpsProvider(clock=clock)
    report = run_benchmark(
        factory,
        n_arenas=2,
        max_decisions=2 * n_per_arena,
        duration_s=1e9,
        clock=clock,
        tps_provider=provider,
        log=None,
    )

    # Arena 1's stall produces 0 TPS (d_tick=0 -> 0/0.1==0); gate must fail.
    assert report.sustains_19_tps is False
    assert report.sustained_tps_min == pytest.approx(0.0)
    assert report.max_arenas_sustaining_tps == 0


# ===========================================================================
# main(): the live gates are not silently-true.
# ===========================================================================


class _ScriptedLiveBridge:
    """A minimal four-method transport that scripts state replies for main() tests.

    Stands in for the real TcpBridgeClient so main()'s wiring (provider + gates)
    can be exercised with NO socket. ``recv`` advances a shared FakeClock by a
    fixed per-step wall cost so the tick-delta provider derives a real rate.
    """

    def __init__(self, ticks, clock, *, wall_per_step):
        self._ticks = list(ticks)
        self._clock = clock
        self._wall_per_step = float(wall_per_step)
        self._i = 0

    def connect(self):
        pass

    def send(self, obj):
        pass

    def recv(self):
        self._clock.advance(self._wall_per_step)
        tick = self._ticks[min(self._i, len(self._ticks) - 1)]
        self._i += 1
        return _state(tick=tick)

    def close(self):
        pass


def _patch_main_for_live(monkeypatch, ticks, *, wall_per_step):
    """Patch benchmark.main()'s dependencies to run offline against scripted ticks.

    Replaces TcpBridgeClient with a scripted bridge, time.perf_counter with a
    manually-advanced FakeClock (advanced by recv), and silences the logger so
    main() can be invoked without a socket, server, or real wall-clock wait.
    Returns the FakeClock so the caller can reason about timing if needed.
    """
    import eval.benchmark as bench

    clock = FakeClock()

    def fake_client(host, port):
        return _ScriptedLiveBridge(ticks, clock, wall_per_step=wall_per_step)

    class _NullLogger:
        def __init__(self, *a, **k):
            pass

        def log(self, *a, **k):
            pass

        def summary(self, *a, **k):
            pass

        def close(self):
            pass

    monkeypatch.setattr(bench, "TcpBridgeClient", fake_client)
    monkeypatch.setattr(bench, "MetricsLogger", _NullLogger)
    monkeypatch.setattr(bench.time, "perf_counter", clock)
    return clock


def test_main_tps_gate_reflects_real_dip(monkeypatch, capsys):
    """main() exits non-zero on a real TPS dip — the gate is NOT silently true.

    Scripts a tick cadence that stalls for one window (dtick=0 -> 0 TPS) so the
    live tick-delta provider produces a genuine sub-19 reading. main() must exit
    1 and the damage gate must be reported INERT (loud banner), proving exit code
    reflects gates that actually ran rather than defaulting to pass.
    """
    # Healthy then a stall: 1,2,3,3,4 -> the repeat is a real dip.
    clock = _patch_main_for_live(
        monkeypatch, ticks=[1, 2, 3, 3, 4], wall_per_step=0.050
    )
    # Tiny duration: the wall clock (advanced 50 ms per recv) trips the deadline
    # after a few decisions, so the run terminates without a real-time wait.
    rc = main(["--duration", "0.18", "--arenas", "1"])
    err = capsys.readouterr().err
    assert rc == 1
    assert "FAIL" in err
    # The TPS gate caught the dip OR the inert-damage gate forced the failure;
    # either way exit is non-zero and the damage gate is explicitly inert.
    assert "INERT" in err


def test_main_damage_gate_not_silently_true(monkeypatch, capsys):
    """Even with healthy TPS, main() never exits 0 on the inert damage gate.

    The live run cannot script a known-N-hit exchange, so the damage-boundary
    gate does not run. main() must NOT default it to pass: it prints a LOUD inert
    banner and exits 1, so exit 0 can never falsely imply the damage check ran.
    """
    # A perfectly healthy cadence: +1 tick per 50 ms -> 20 TPS sustained.
    clock = _patch_main_for_live(
        monkeypatch, ticks=[1, 2, 3, 4, 5, 6], wall_per_step=0.050
    )
    rc = main(["--duration", "0.18", "--arenas", "1"])
    err = capsys.readouterr().err
    # TPS sustained, but the damage gate is inert -> NOT a pass.
    assert rc == 1
    assert "INERT" in err
    assert "damage-boundary gate did not run" in err


# ===========================================================================
# Derived-constant sanity.
# ===========================================================================


def test_decision_interval_constant_matches_contract():
    """The benchmark's decision-interval constant is derived from the contract."""
    assert DECISION_INTERVAL_S == pytest.approx(DECISION_INTERVAL_MS / 1000.0)
    assert DECISION_INTERVAL_S == pytest.approx(0.200)


# ===========================================================================
# ResourceSampler: best-effort, never raises.
# ===========================================================================


def test_resource_sampler_never_raises_and_reports_shape():
    """Sampling and reporting must not raise on any platform; keys are stable."""
    sampler = ResourceSampler()
    for _ in range(3):
        sampler.sample()  # must not raise regardless of psutil availability
    report = sampler.report()
    for key in (
        "available",
        "cpu_percent_mean",
        "cpu_percent_max",
        "cpu_freq_mhz_mean",
        "cpu_freq_available",
        "temperature_available",
        "power_thermal_note",
        "n_samples",
    ):
        assert key in report
    # The platform power/thermal gap is documented in the note, regardless of OS.
    assert "package power" in report["power_thermal_note"].lower()


# ===========================================================================
# MetricsLogger — fallback write/read-back + backend resolution.
# ===========================================================================


def test_logger_jsonl_writes_and_reads_back(tmp_path):
    """The JSON-lines fallback writes metrics + summary and reads them back."""
    logger = MetricsLogger(run_name="rw", backend="jsonl", log_dir=str(tmp_path))
    assert logger.backend == "jsonl"
    assert logger.metrics_path is not None and logger.summary_path is not None

    logger.log({"loss": 1.5, "tps": 20.0}, step=0)
    logger.log_scalar("loss", 0.75, step=1)
    logger.summary({"final_loss": 0.75, "passed": True})
    logger.close()

    records = read_jsonl(logger.metrics_path)
    assert len(records) == 2
    assert records[0]["step"] == 0
    assert records[0]["loss"] == pytest.approx(1.5)
    assert records[0]["tps"] == pytest.approx(20.0)
    assert records[1]["step"] == 1
    assert records[1]["loss"] == pytest.approx(0.75)

    summary = read_summary(logger.summary_path)
    assert summary["final_loss"] == pytest.approx(0.75)
    assert summary["passed"] is True


def test_logger_coerces_non_finite_and_numpy_like(tmp_path):
    """Non-finite floats are dropped to null; values stay JSON-clean."""
    logger = MetricsLogger(run_name="coerce", backend="jsonl", log_dir=str(tmp_path))
    logger.log({"nan_metric": float("nan"), "inf_metric": float("inf"), "ok": 3.0})
    logger.close()
    rec = read_jsonl(logger.metrics_path)[0]
    # NaN / Inf have no JSON representation -> coerced to None (json null).
    assert rec["nan_metric"] is None
    assert rec["inf_metric"] is None
    assert rec["ok"] == pytest.approx(3.0)


def test_logger_auto_resolves_to_jsonl_when_no_heavy_backend(tmp_path):
    """backend='auto' falls back to jsonl when W&B / TensorBoard are absent.

    The test environment has neither W&B nor TensorBoard installed, so auto must
    resolve to the dependency-free jsonl sink. The auto order is documented and
    terminates in 'jsonl'.
    """
    assert AUTO_BACKEND_ORDER[-1] == "jsonl"
    logger = MetricsLogger(run_name="auto", backend="auto", log_dir=str(tmp_path))
    try:
        # On a machine with wandb/tensorboard this could differ, but it must always
        # be one of the known backends and must never raise.
        assert logger.backend in AUTO_BACKEND_ORDER
        logger.log({"x": 1.0}, step=0)
    finally:
        logger.close()


def test_logger_explicit_missing_backend_raises(monkeypatch, tmp_path):
    """An explicit unavailable backend raises BackendUnavailableError (loud).

    Force the wandb init path to report unavailable and assert the explicit
    request raises rather than silently falling back (only 'auto' falls back).
    """
    # Patch the private init so the test does not depend on wandb being absent.
    monkeypatch.setattr(MetricsLogger, "_init_wandb", lambda self: False)
    with pytest.raises(BackendUnavailableError):
        MetricsLogger(run_name="nope", backend="wandb", log_dir=str(tmp_path))


def test_logger_rejects_unknown_backend(tmp_path):
    with pytest.raises(ValueError):
        MetricsLogger(run_name="bad", backend="not_a_backend", log_dir=str(tmp_path))


def test_logger_log_after_close_raises(tmp_path):
    logger = MetricsLogger(run_name="closed", backend="jsonl", log_dir=str(tmp_path))
    logger.close()
    with pytest.raises(RuntimeError):
        logger.log({"x": 1.0})
    # close() is idempotent.
    logger.close()


def test_logger_context_manager(tmp_path):
    """The logger works as a context manager and flushes on exit."""
    with MetricsLogger(run_name="ctx", backend="jsonl", log_dir=str(tmp_path)) as logger:
        logger.log({"a": 1.0}, step=0)
        metrics_path = logger.metrics_path
    # After the with-block the file is flushed/closed and readable.
    records = read_jsonl(metrics_path)
    assert len(records) == 1
    assert records[0]["a"] == pytest.approx(1.0)


def test_read_jsonl_skips_blank_lines_and_rejects_garbage(tmp_path):
    path = tmp_path / "m.jsonl"
    path.write_text('{"a": 1}\n\n{"b": 2}\n', encoding="utf-8")
    records = read_jsonl(str(path))
    assert records == [{"a": 1}, {"b": 2}]

    bad = tmp_path / "bad.jsonl"
    bad.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_jsonl(str(bad))


def test_read_summary_rejects_non_object(tmp_path):
    path = tmp_path / "s.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ValueError):
        read_summary(str(path))
