"""benchmark — Bridge throughput and latency benchmark (M1 measured exit number).

The bridge spike's exit criterion is a **measured number**, not a green check
(plan: "AC4 — the number"). This harness drives the env/bridge and reports:

  1. **transitions/s/arena** — decisions completed per wall-second.
  2. **p99 Node→Python round-trip latency** at the 200 ms decision interval — the
     time from sending a ``step`` to receiving the matching ``state``; p50/p95/p99
     are all reported.
  3. **damage-event boundary correctness** — drive a scripted N-hit exchange and
     assert the summed ``events.damage_dealt`` count equals N, with no event
     dropped or double-counted at a decision-window boundary (a benchmark-level
     cross-check of TC7b).
  4. **max arenas sustaining ≥19 TPS** over a ≥10-minute run, with CPU package
     power / thermal recorded best-effort (see the platform note below).

------------------------------------------------------------------------------
The injectable transport/clock seam (offline metric-logic proof)
------------------------------------------------------------------------------
The measurement LOGIC must be provable without a socket or a live server, so the
harness is parameterized by:

  * a ``transport_factory`` — exactly the env's four-method
    :class:`~env.mc_pvp_env.BridgeTransport` seam. The live run passes the real
    :class:`~env.mc_pvp_env.TcpBridgeClient`; tests pass a
    :class:`FakeBridge` that returns scripted ``state`` messages with
    deterministic, pre-fed round-trip latencies — no socket, no server.
  * a ``clock`` — a zero-arg monotonic-seconds callable (defaults to
    :func:`time.perf_counter`). The :class:`FakeBridge` advances a
    :class:`FakeClock` by the scripted latency on each ``recv``, so the measured
    round-trip equals the injected number exactly and the percentile math is
    asserted against known inputs.

The metric MATH lives in pure module-level functions — :func:`percentile`,
:func:`transitions_per_second`, :func:`tally_damage_dealt`,
:func:`min_sustained_tps`, :func:`sustains_tps` — so each is unit-tested on
synthetic data independent of any bridge.

------------------------------------------------------------------------------
AC4 / TC12 — the REAL number is a documented HUMAN follow-up
------------------------------------------------------------------------------
This task delivers the harness, the offline metric-logic proof, and the shared
logging seam (``eval/logging.py``). It does NOT, and cannot offline, produce the
actual AC4 number: a SUSTAINED ≥10-minute run on the dev laptop (Intel Core Ultra
7 258V, 8c/8t) measuring the max arenas that hold ≥19 TPS with CPU package power
/ thermal recorded. That run needs the live Paper server + Node bridge and is run
by a human via ``python -m eval.benchmark`` against a started bridge (see
``server/README.md`` / the ``bridge`` setup). The printed report is the AC4
evidence artifact.

CPU power / thermal caveat: portable readings are limited on Windows. This
harness records what :mod:`psutil` exposes (per-core CPU% and, where available,
CPU frequency) and CLEARLY notes that package power (RAPL/MSR) and die temperature
are not portably available — that gap is called out in the report so the human
run can supplement it with a vendor tool (e.g. HWiNFO / Intel Power Gadget).

Owner: T11 (Eval/infra track)
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Union,
)

from agent.contract_config import DECISION_INTERVAL_MS, SERVER_TPS
from bridge.messages import ResetAckMsg, StateMsg
from env.mc_pvp_env import BridgeError, BridgeTransport, TcpBridgeClient
from eval.logging import MetricsLogger

__all__ = [
    # Constants
    "DECISION_INTERVAL_S",
    "MIN_SUSTAINED_TPS",
    "DEFAULT_BENCH_DURATION_S",
    "DEFAULT_MAX_ARENAS",
    "DAMAGE_PER_HIT",
    # Pure metric functions
    "percentile",
    "latency_percentiles",
    "transitions_per_second",
    "tally_damage_dealt",
    "min_sustained_tps",
    "sustains_tps",
    # TPS providers
    "TickDeltaTpsProvider",
    # Offline fixtures (the injectable fake bridge + clock)
    "FakeClock",
    "FakeBridge",
    # Resource sampling
    "ResourceSampler",
    # Runner + report
    "BenchmarkReport",
    "run_benchmark",
    "main",
]


# ---------------------------------------------------------------------------
# Constants (derived from the frozen contract so they can never drift).
# ---------------------------------------------------------------------------

#: The decision interval in SECONDS (== 0.200 s). The p99 round-trip latency is
#: reported "at the 200 ms decision interval"; this is that interval.
DECISION_INTERVAL_S: float = DECISION_INTERVAL_MS / 1000.0

#: Sustained-TPS floor. Below this the 200 ms decision-interval assumption breaks
#: (a tick is no longer 50 ms), so an arena that dips under it is NOT sustaining.
#: Matches the plan's "≥19 TPS" (vanilla is SERVER_TPS == 20).
MIN_SUSTAINED_TPS: float = 19.0

#: Default sustained-run duration for the LIVE AC4 run (>= 10 minutes per TC12).
DEFAULT_BENCH_DURATION_S: float = 600.0

#: Default ceiling for the arena sweep on the dev laptop. One Paper server is
#: single-threaded so arenas ~= cores; the Core Ultra 7 258V is 8c/8t, but the
#: realistic sustained count is ~2-4 (plan note), so 4 is a sane default ceiling.
DEFAULT_MAX_ARENAS: int = 4

#: Damage dealt per landed sword hit in the scripted boundary-correctness
#: exchange. Only the EVENT COUNT (one event per hit) is asserted for TC7b; the
#: magnitude is incidental. Kept as a constant so the test and the harness agree.
DAMAGE_PER_HIT: float = 5.0


# ---------------------------------------------------------------------------
# Pure metric functions (unit-tested independently of any bridge).
# ---------------------------------------------------------------------------


def percentile(samples: Sequence[float], q: float) -> float:
    """Return the ``q``-th percentile of ``samples`` (``q`` in ``[0, 100]``).

    Uses linear interpolation between the two nearest ranks (the "inclusive" /
    NumPy-default ``linear`` method), so for a sorted sample of length ``n`` the
    rank is ``(q/100) * (n - 1)``. This matches ``numpy.percentile`` /
    ``statistics.quantiles(method="inclusive")`` on the test fixtures, so a known
    sample yields a known p50/p95/p99.

    Args:
        samples: A non-empty sequence of numeric latency samples (any order).
        q: The percentile to compute, in ``[0, 100]``.

    Returns:
        The interpolated percentile value as a float.

    Raises:
        ValueError: if ``samples`` is empty or ``q`` is outside ``[0, 100]``.
    """
    if not (0.0 <= q <= 100.0):
        raise ValueError(f"percentile q must be in [0, 100], got {q!r}")
    ordered = sorted(float(s) for s in samples)
    n = len(ordered)
    if n == 0:
        raise ValueError("cannot take a percentile of an empty sample")
    if n == 1:
        return ordered[0]

    rank = (q / 100.0) * (n - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return ordered[int(rank)]
    frac = rank - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def latency_percentiles(
    samples_s: Sequence[float], percentiles: Sequence[float] = (50.0, 95.0, 99.0)
) -> Dict[str, float]:
    """Summarize round-trip latency samples (seconds) into a metrics dict (ms).

    Converts the requested percentiles plus min/max/mean to MILLISECONDS — the
    natural unit for a 200 ms decision interval. Returns an empty-but-typed dict
    of zeros when there are no samples so a report always has the keys.

    Args:
        samples_s: Round-trip latencies in SECONDS.
        percentiles: Which percentiles to compute (default p50/p95/p99).

    Returns:
        ``{"p50_ms", "p95_ms", "p99_ms", "min_ms", "max_ms", "mean_ms", "count"}``
        with latency values in milliseconds.
    """
    keys = {p: f"p{int(p)}_ms" for p in percentiles}
    if not samples_s:
        out: Dict[str, float] = {keys[p]: 0.0 for p in percentiles}
        out.update({"min_ms": 0.0, "max_ms": 0.0, "mean_ms": 0.0, "count": 0})
        return out

    samples_ms = [float(s) * 1000.0 for s in samples_s]
    out = {keys[p]: percentile(samples_ms, p) for p in percentiles}
    out["min_ms"] = min(samples_ms)
    out["max_ms"] = max(samples_ms)
    out["mean_ms"] = sum(samples_ms) / len(samples_ms)
    out["count"] = len(samples_ms)
    return out


def transitions_per_second(n_transitions: int, elapsed_s: float) -> float:
    """Throughput in transitions (decisions) per wall-second.

    Args:
        n_transitions: Number of completed ``step`` decisions (>= 0).
        elapsed_s: Wall-clock seconds the decisions spanned (> 0).

    Returns:
        ``n_transitions / elapsed_s``.

    Raises:
        ValueError: if ``elapsed_s`` <= 0 or ``n_transitions`` < 0.
    """
    if n_transitions < 0:
        raise ValueError(f"n_transitions must be >= 0, got {n_transitions}")
    if elapsed_s <= 0.0:
        raise ValueError(f"elapsed_s must be > 0, got {elapsed_s}")
    return n_transitions / elapsed_s


def tally_damage_dealt(states: Sequence[StateMsg]) -> Dict[str, Any]:
    """Sum damage-dealt EVENTS across a sequence of per-decision ``state`` messages.

    The benchmark cross-check for TC7b: a scripted N-hit exchange is driven one
    hit at a time and each landed hit must surface in exactly ONE decision
    window's ``events.damage_dealt`` — never dropped (a hit straddling a window
    boundary is lost) and never double-counted (the same hit summed into two
    windows). This counts a window as a "hit event" iff its ``damage_dealt`` is
    strictly positive, so the returned ``hit_events`` must equal the scripted N.

    Args:
        states: Per-decision ``state`` messages in order (one per ``step``).

    Returns:
        ``{"hit_events", "total_damage", "windows"}`` — number of windows with a
        positive damage event, the summed damage magnitude, and the window count.
    """
    hit_events = 0
    total_damage = 0.0
    for state in states:
        dealt = float(state.events.damage_dealt)
        if dealt > 0.0:
            hit_events += 1
        total_damage += dealt
    return {
        "hit_events": hit_events,
        "total_damage": total_damage,
        "windows": len(states),
    }


def min_sustained_tps(tps_samples: Sequence[float]) -> float:
    """Return the WORST (minimum) TPS over a run — the sustained-TPS figure.

    "Sustained ≥19 TPS" means the floor never dips below 19, so the run's
    sustained TPS is its minimum sample. Returns ``0.0`` for an empty series (an
    arena that produced no TPS reading cannot be said to sustain anything).

    Args:
        tps_samples: Per-interval server TPS readings over the run.

    Returns:
        The minimum sample, or ``0.0`` if there are none.
    """
    if not tps_samples:
        return 0.0
    return min(float(s) for s in tps_samples)


def sustains_tps(
    tps_samples: Sequence[float], floor: float = MIN_SUSTAINED_TPS
) -> bool:
    """True iff EVERY TPS sample is at or above ``floor`` (and there is >= 1).

    A single dip below ``floor`` breaks the 200 ms decision-interval assumption,
    so one bad sample fails the whole run. An empty series cannot sustain and
    returns ``False``.

    Args:
        tps_samples: Per-interval server TPS readings.
        floor: Minimum acceptable TPS (default :data:`MIN_SUSTAINED_TPS`).

    Returns:
        ``True`` if all samples >= ``floor`` and the series is non-empty.
    """
    if not tps_samples:
        return False
    return min_sustained_tps(tps_samples) >= float(floor)


# ---------------------------------------------------------------------------
# Offline fixtures: the injectable fake bridge + clock.
#
# These satisfy the env's four-method BridgeTransport seam so the measurement
# logic runs with NO socket and NO server. The fake feeds DETERMINISTIC
# round-trip latencies: recv() advances the injected clock by the scripted
# latency, so the measured round-trip equals the fed number exactly.
# ---------------------------------------------------------------------------


class FakeClock:
    """A manually-advanced monotonic clock for deterministic latency measurement.

    Call the instance (``clock()``) to read the current time in seconds; call
    :meth:`advance` to move it forward. The benchmark reads the clock around each
    ``step``/``recv`` pair, and :class:`FakeBridge` advances it by the scripted
    latency inside ``recv``, so the measured round-trip is exactly the injected
    value — making the percentile math assertable against known inputs.
    """

    def __init__(self, start: float = 0.0) -> None:
        self._now = float(start)

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        """Advance the clock by ``seconds`` (must be >= 0)."""
        if seconds < 0.0:
            raise ValueError(f"cannot advance the clock by a negative amount: {seconds}")
        self._now += float(seconds)


class FakeBridge:
    """A scripted, socket-free :class:`~env.mc_pvp_env.BridgeTransport` for the bench.

    Implements exactly the four-method transport contract the env/benchmark
    depend on (``connect`` / ``send`` / ``recv`` / ``close``). ``recv()`` pops the
    next scripted inbound message; when that message is a ``state``, it first
    advances the injected :class:`FakeClock` by the next scripted round-trip
    latency, so the time the benchmark measures from "sent step" to "got state"
    equals the fed latency exactly.

    This is the benchmark's analogue of ``tests/test_run_random.py``'s
    ``ScriptedTracerBridge`` — same contract, plus the latency/clock coupling the
    benchmark needs.

    Args:
        inbound: Ordered scripted inbound dataclasses (``ResetAckMsg`` /
            ``StateMsg``) the env/runner will ``recv()`` in protocol order:
            ``reset`` -> ``reset_ack`` (+ post-reset ``state``), then one
            ``state`` per ``step``.
        latencies_s: Round-trip latency (seconds) to charge for each inbound
            ``state`` message, consumed in order. When exhausted (or if a state is
            popped with none left) a latency of ``0.0`` is charged. Ignored for
            non-state messages (``reset_ack`` has no round-trip cost here).
        clock: The :class:`FakeClock` to advance. Required so the benchmark and the
            bridge share one clock instance.
    """

    def __init__(
        self,
        inbound: Sequence[Union[StateMsg, ResetAckMsg]],
        latencies_s: Sequence[float],
        clock: FakeClock,
    ) -> None:
        self.inbound: List[Union[StateMsg, ResetAckMsg]] = list(inbound)
        self._latencies: List[float] = [float(x) for x in latencies_s]
        self._lat_idx = 0
        self._clock = clock
        self.sent: List[Dict[str, Any]] = []
        self.connects = 0
        self.closes = 0
        self.is_open = False

    def connect(self) -> None:
        self.connects += 1
        self.is_open = True

    def send(self, obj: Mapping[str, Any]) -> None:
        self.sent.append(dict(obj))

    def recv(self) -> Union[StateMsg, ResetAckMsg]:
        if not self.inbound:
            raise BridgeError("FakeBridge.recv() called with an empty queue")
        msg = self.inbound.pop(0)
        # Charge the scripted round-trip latency only for state replies (the
        # step -> state round-trip the benchmark measures).
        if isinstance(msg, StateMsg):
            if self._lat_idx < len(self._latencies):
                latency = self._latencies[self._lat_idx]
                self._lat_idx += 1
            else:
                latency = 0.0
            self._clock.advance(latency)
        return msg

    def close(self) -> None:
        self.closes += 1
        self.is_open = False


# ---------------------------------------------------------------------------
# Resource sampling (CPU% / freq via psutil; package power/thermal noted).
# ---------------------------------------------------------------------------


class ResourceSampler:
    """Best-effort CPU utilization / frequency sampler with a clear platform note.

    On the LIVE AC4 run the meaningful figures are sustained CPU load and whether
    the laptop thermally throttles. This sampler records what :mod:`psutil`
    portably exposes — system-wide CPU% and, where the platform supports it, CPU
    frequency — and records a NOTE that CPU **package power** (RAPL/MSR) and **die
    temperature** are not portably available, most notably on Windows. The human
    AC4 run supplements those with a vendor tool (HWiNFO / Intel Power Gadget);
    this sampler makes the gap explicit rather than silently omitting it.

    All readings are best-effort: any failure degrades to "unavailable" with a
    note, never an exception, so the benchmark never crashes on a sampling error.
    """

    #: Static note: the portability gap for package power / die temperature.
    POWER_THERMAL_NOTE: str = (
        "CPU package power (RAPL/MSR) and die temperature are not portably "
        "readable from Python, especially on Windows; record them with a vendor "
        "tool (HWiNFO / Intel Power Gadget) during the live AC4 run. This sampler "
        "captures CPU%% and CPU frequency only."
    )

    def __init__(self) -> None:
        self.cpu_percent_samples: List[float] = []
        self.cpu_freq_mhz_samples: List[float] = []
        self.temperature_c_samples: List[float] = []
        self.available: bool = False
        self.freq_available: bool = False
        self.temp_available: bool = False
        self.note: str = self.POWER_THERMAL_NOTE
        self._psutil = None
        #: The first non-blocking cpu_percent() reading after import is a spurious
        #: 0.0 (it has no prior interval to diff against). We discard exactly one
        #: real sample so that leading 0.0 never dilutes cpu_percent_mean.
        self._cpu_primed: bool = False

        try:
            import psutil  # type: ignore

            self._psutil = psutil
            self.available = True
            # Prime the non-blocking cpu_percent() so the first real sample is
            # measured against this baseline rather than returning 0.0.
            try:
                psutil.cpu_percent(interval=None)
            except Exception:
                pass
        except Exception:
            self.note = (
                "psutil not importable; CPU%/frequency not sampled. "
                + self.POWER_THERMAL_NOTE
            )

    def sample(self) -> None:
        """Take one best-effort CPU%/freq/temperature sample. Never raises."""
        if not self.available or self._psutil is None:
            return
        psutil = self._psutil
        try:
            cpu = float(psutil.cpu_percent(interval=None))
            # Discard the first real reading: after import the first non-blocking
            # cpu_percent() has no prior interval to diff against and returns a
            # spurious 0.0 that would otherwise pull the mean down.
            if not self._cpu_primed:
                self._cpu_primed = True
            else:
                self.cpu_percent_samples.append(cpu)
        except Exception:
            pass
        try:
            freq = psutil.cpu_freq()
            if freq is not None and freq.current:
                self.cpu_freq_mhz_samples.append(float(freq.current))
                self.freq_available = True
        except Exception:
            # cpu_freq() is not implemented on every platform; tolerate it.
            pass
        # Temperatures are POSIX-only in psutil (no Windows support) — try, but do
        # not depend on it. This is the documented thermal gap on Windows.
        sensors = getattr(psutil, "sensors_temperatures", None)
        if callable(sensors):
            try:
                readings = sensors()
                for entries in readings.values():
                    for entry in entries:
                        if entry.current is not None:
                            self.temperature_c_samples.append(float(entry.current))
                            self.temp_available = True
            except Exception:
                pass

    def report(self) -> Dict[str, Any]:
        """Summarize the sampled resource usage into a report dict.

        Always returns the same keys (filled with ``None``/flags when a metric is
        unavailable) so the benchmark report shape is stable. The ``note`` field
        carries the package-power / thermal platform caveat verbatim.
        """

        def _mean(xs: List[float]) -> Optional[float]:
            return (sum(xs) / len(xs)) if xs else None

        def _max(xs: List[float]) -> Optional[float]:
            return max(xs) if xs else None

        return {
            "available": self.available,
            "cpu_percent_mean": _mean(self.cpu_percent_samples),
            "cpu_percent_max": _max(self.cpu_percent_samples),
            "cpu_freq_mhz_mean": _mean(self.cpu_freq_mhz_samples),
            "cpu_freq_mhz_min": (
                min(self.cpu_freq_mhz_samples) if self.cpu_freq_mhz_samples else None
            ),
            "cpu_freq_available": self.freq_available,
            "temperature_c_max": _max(self.temperature_c_samples),
            "temperature_available": self.temp_available,
            "power_thermal_note": self.note,
            "n_samples": len(self.cpu_percent_samples),
        }


# ---------------------------------------------------------------------------
# The report.
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkReport:
    """Structured result of one benchmark run (the AC4 evidence artifact).

    Attributes:
        duration_target_s: The requested run duration in seconds.
        duration_actual_s: The measured wall-clock duration in seconds.
        n_arenas: Number of arenas driven in this run.
        transitions: Total ``step`` decisions completed across all arenas.
        transitions_per_s_per_arena: Throughput per arena (decisions/wall-second).
        latency_ms: p50/p95/p99 (+ min/max/mean/count) round-trip latency, in ms.
        damage_boundary: Damage-event boundary cross-check
            (``hit_events`` / ``total_damage`` / ``windows`` / ``expected_hits`` /
            ``ok``) — ``ok`` iff ``hit_events == expected_hits`` (no drop/double).
        sustained_tps_min: The worst per-interval server TPS over the run.
        sustains_19_tps: Whether every TPS sample stayed >= 19.
        max_arenas_sustaining_tps: The largest arena count that sustained ≥19 TPS
            in this run (``n_arenas`` when it sustained, else ``0``). On the
            live sweep this is updated across arena counts.
        resources: The :class:`ResourceSampler` report (CPU%/freq + the platform
            power/thermal note).
        is_live: ``True`` for a real-bridge run, ``False`` for the offline
            fake-bridge proof (recorded so the artifact is never mistaken for the
            sustained AC4 number).
        notes: Free-form human-readable notes (incl. the AC4 follow-up reminder).
    """

    duration_target_s: float = 0.0
    duration_actual_s: float = 0.0
    n_arenas: int = 0
    transitions: int = 0
    transitions_per_s_per_arena: float = 0.0
    latency_ms: Dict[str, float] = field(default_factory=dict)
    damage_boundary: Dict[str, Any] = field(default_factory=dict)
    sustained_tps_min: float = 0.0
    sustains_19_tps: bool = False
    max_arenas_sustaining_tps: int = 0
    resources: Dict[str, Any] = field(default_factory=dict)
    is_live: bool = False
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Render the report as a plain JSON-serializable dict."""
        return {
            "duration_target_s": self.duration_target_s,
            "duration_actual_s": self.duration_actual_s,
            "n_arenas": self.n_arenas,
            "transitions": self.transitions,
            "transitions_per_s_per_arena": self.transitions_per_s_per_arena,
            "latency_ms": dict(self.latency_ms),
            "damage_boundary": dict(self.damage_boundary),
            "sustained_tps_min": self.sustained_tps_min,
            "sustains_19_tps": self.sustains_19_tps,
            "max_arenas_sustaining_tps": self.max_arenas_sustaining_tps,
            "resources": dict(self.resources),
            "is_live": self.is_live,
            "notes": list(self.notes),
        }


# ---------------------------------------------------------------------------
# The benchmark runner.
# ---------------------------------------------------------------------------

#: A scripted-bridge factory: given an arena index, return a transport for it.
#: For the offline path this hands out a fresh :class:`FakeBridge`; for the live
#: path it builds a :class:`~env.mc_pvp_env.TcpBridgeClient`.
TransportFactory = Callable[[int], BridgeTransport]


def run_benchmark(
    transport_factory: TransportFactory,
    *,
    n_arenas: int = 1,
    max_decisions: Optional[int] = None,
    duration_s: float = DEFAULT_BENCH_DURATION_S,
    clock: Callable[[], float] = time.perf_counter,
    tps_provider: Optional[Callable[[StateMsg], float]] = None,
    expected_hits: Optional[int] = None,
    logger: Optional[MetricsLogger] = None,
    resource_sampler: Optional[ResourceSampler] = None,
    is_live: bool = False,
    log: Optional[Callable[[str], None]] = None,
) -> BenchmarkReport:
    """Drive ``n_arenas`` through the bridge and return a measured report.

    Each arena is driven round-robin: a ``step`` is sent, the matching ``state``
    is awaited, and the round-trip latency (``clock`` after recv minus ``clock``
    before send) is recorded. The loop runs until ``max_decisions`` total
    decisions have completed (offline: a fixed budget) OR ``clock`` shows
    ``duration_s`` has elapsed (live: the ≥10-minute sustained run), whichever
    comes first; at least one decision per arena always runs.

    Per-arena server TPS is read from each ``state`` via ``tps_provider`` (which
    derives TPS from the ``tick`` deltas / a bridge-reported field); the offline
    fake feeds a constant so the sustained-TPS detector can be asserted. Damage
    events are tallied across the run for the boundary cross-check.

    Args:
        transport_factory: ``arena_index -> BridgeTransport``. The benchmark calls
            ``connect()`` on each transport and ``close()`` at the end.
        n_arenas: Number of arenas to drive concurrently round-robin (>= 1).
        max_decisions: Total decisions to run across all arenas before stopping
            (``None`` runs until ``duration_s`` elapses — the live mode).
        duration_s: Wall-clock budget in seconds (the ≥10-min live run uses
            :data:`DEFAULT_BENCH_DURATION_S`).
        clock: Zero-arg monotonic-seconds reader. Defaults to
            :func:`time.perf_counter`; tests inject a :class:`FakeClock`.
        tps_provider: ``state -> server_tps``. Defaults to the constant
            :func:`_default_tps_provider` (the explicit OFFLINE/test default, which
            assumes vanilla :data:`~agent.contract_config.SERVER_TPS` because the
            fake bridge has no real server load). The LIVE run passes a
            :class:`TickDeltaTpsProvider` that derives TPS from ``tick`` deltas so
            the sustained-TPS gate is honest.
        expected_hits: Ground-truth N for the damage-boundary cross-check. When
            given, ``report.damage_boundary["ok"]`` asserts ``hit_events == N``.
        logger: Optional :class:`~eval.logging.MetricsLogger`; per-decision and
            summary metrics are logged through it when provided.
        resource_sampler: Optional :class:`ResourceSampler`; sampled once per
            arena round. Defaults to a fresh one.
        is_live: Marks the report as a live (vs offline) run.
        log: Optional ``str -> None`` progress sink (``None`` silences it).

    Returns:
        A populated :class:`BenchmarkReport`.

    Raises:
        ValueError: if ``n_arenas`` < 1, ``duration_s`` <= 0, or both
            ``max_decisions`` and a finite ``duration_s`` are non-positive.
    """
    if n_arenas < 1:
        raise ValueError(f"n_arenas must be >= 1, got {n_arenas}")
    if duration_s <= 0.0:
        raise ValueError(f"duration_s must be > 0, got {duration_s}")
    if max_decisions is not None and max_decisions < 1:
        raise ValueError(f"max_decisions must be >= 1 or None, got {max_decisions}")

    def _emit(message: str) -> None:
        if log is not None:
            log(message)

    if tps_provider is None:
        tps_provider = _default_tps_provider
    if resource_sampler is None:
        resource_sampler = ResourceSampler()

    # Build + connect one transport per arena. Each arena keeps its own decision
    # action cycling deterministically through the macro indices so the run is
    # reproducible and never sticks on a single macro.
    transports: List[BridgeTransport] = []
    for arena in range(n_arenas):
        transport = transport_factory(arena)
        transport.connect()
        transports.append(transport)

    states_seen: List[StateMsg] = []
    latencies_s: List[float] = []
    tps_samples: List[float] = []
    transitions = 0
    action_cycle = 0

    start = clock()
    deadline = start + duration_s

    try:
        while True:
            # Stop conditions: hit the decision budget, or pass the wall-clock
            # deadline. At least one full arena round always runs (checked at the
            # bottom so the first round is never skipped).
            if max_decisions is not None and transitions >= max_decisions:
                break

            for arena_idx, transport in enumerate(transports):
                action = action_cycle % 8  # cycle 0..7 over the 8 frozen macros
                action_cycle += 1

                # Measure the step -> state round-trip across the injected clock.
                t0 = clock()
                transport.send({"type": "step", "action": action})
                msg = transport.recv()
                t1 = clock()

                if not isinstance(msg, StateMsg):
                    raise BridgeError(
                        f"benchmark expected a state reply to step, got "
                        f"{type(msg).__name__}"
                    )

                latency = t1 - t0
                latencies_s.append(latency)
                states_seen.append(msg)
                tps_samples.append(float(tps_provider(msg, arena=arena_idx)))
                transitions += 1

                if logger is not None:
                    logger.log(
                        {
                            "round_trip_ms": latency * 1000.0,
                            "server_tps": tps_samples[-1],
                        },
                        step=transitions,
                    )

                if max_decisions is not None and transitions >= max_decisions:
                    break

            resource_sampler.sample()

            # Guard the wall-clock deadline. This covers both the live mode
            # (max_decisions is None) and a budgeted run that must not overrun the
            # duration cap, so one unconditional check suffices.
            if clock() >= deadline:
                break
    finally:
        for transport in transports:
            try:
                transport.close()
            except (BridgeError, OSError):
                pass

    elapsed = max(clock() - start, 1e-9)  # guard divide-by-zero on a zero-cost fake

    # --- assemble the report ------------------------------------------------
    report = BenchmarkReport(
        duration_target_s=float(duration_s),
        duration_actual_s=float(elapsed),
        n_arenas=int(n_arenas),
        transitions=int(transitions),
        is_live=bool(is_live),
    )

    # transitions/s PER ARENA: total throughput divided by the arena count. Kept
    # as a true float so a non-even arena division (e.g. 7 decisions / 3 arenas)
    # reports the exact per-arena rate instead of a rounded one.
    report.transitions_per_s_per_arena = (
        transitions_per_second(transitions, elapsed) / n_arenas
    )

    report.latency_ms = latency_percentiles(latencies_s)

    damage = tally_damage_dealt(states_seen)
    if expected_hits is not None:
        damage["expected_hits"] = int(expected_hits)
        damage["ok"] = damage["hit_events"] == int(expected_hits)
    report.damage_boundary = damage

    report.sustained_tps_min = min_sustained_tps(tps_samples)
    report.sustains_19_tps = sustains_tps(tps_samples)
    report.max_arenas_sustaining_tps = n_arenas if report.sustains_19_tps else 0

    report.resources = resource_sampler.report()

    report.notes.append(
        "AC4/TC12 follow-up: the REAL sustained >=10-min number on the dev laptop "
        "(Core Ultra 7 258V) with CPU package power/thermal is a HUMAN run against "
        "the live bridge; this report is " + ("a LIVE" if is_live else "an OFFLINE")
        + " measurement of the harness/metric logic."
    )
    report.notes.append(report.resources.get("power_thermal_note", ""))

    if logger is not None:
        logger.summary(
            {
                "transitions": report.transitions,
                "transitions_per_s_per_arena": report.transitions_per_s_per_arena,
                "p50_ms": report.latency_ms.get("p50_ms", 0.0),
                "p95_ms": report.latency_ms.get("p95_ms", 0.0),
                "p99_ms": report.latency_ms.get("p99_ms", 0.0),
                "sustained_tps_min": report.sustained_tps_min,
                "sustains_19_tps": report.sustains_19_tps,
                "n_arenas": report.n_arenas,
                "damage_hit_events": damage.get("hit_events", 0),
                "is_live": report.is_live,
            }
        )

    _emit(
        f"[bench] arenas={n_arenas} transitions={transitions} "
        f"tps/arena={report.transitions_per_s_per_arena:.2f} "
        f"p99={report.latency_ms.get('p99_ms', 0.0):.1f}ms "
        f"min_tps={report.sustained_tps_min:.1f} "
        f"sustains>=19={report.sustains_19_tps}"
    )

    return report


def _default_tps_provider(state: StateMsg, *, arena: int = 0) -> float:
    """Constant TPS provider: assume the vanilla server tick rate.

    This is the explicit OFFLINE / test default only. Offline (the fake bridge)
    there is no real server under load, so the sustained-TPS path is asserted by
    feeding scripted ``state`` messages and reading a constant tick rate. It must
    NOT be used as the live default — a constant 20.0 makes ``sustains_19_tps``
    trivially true and the exit code can never catch a real TPS dip. The live run
    uses :class:`TickDeltaTpsProvider`, which derives TPS from ``tick`` deltas.

    The ``arena`` kwarg is accepted (and ignored) so both providers share the
    same call signature and can be swapped without changes at the call site.
    """
    return float(SERVER_TPS)


class TickDeltaTpsProvider:
    """Derive instantaneous server TPS from consecutive ``StateMsg.tick`` deltas.

    The Paper server advances ``StateMsg.tick`` by one game tick per server tick.
    Sampling the wall clock at each ``state`` and dividing the tick delta by the
    wall-second delta yields the server's instantaneous ticks-per-second::

        TPS = (tick_now - tick_prev) / (wall_now - wall_prev)

    A healthy 20-TPS server advances ~20 ticks per real second; a server that is
    falling behind (lag/overload) advances FEWER ticks per wall-second, so the
    derived TPS drops below the floor and the sustained-TPS gate trips. This makes
    the live ``sustains_19_tps`` gate honest, unlike the constant provider.

    The provider is a stateful callable so it can drop straight into the existing
    ``tps_provider: Callable[[StateMsg], float]`` seam. It reads wall time from an
    injected ``clock`` (the SAME monotonic clock :func:`run_benchmark` uses), so a
    test can feed synthetic ``(tick, wall_time)`` pairs and assert the exact TPS.

    The FIRST observed state has no predecessor to diff against; the provider
    reports ``warmup_tps`` for it (default :data:`MIN_SUSTAINED_TPS`, i.e. the
    floor — neutral, so the warm-up sample neither fakes a pass nor a spurious
    dip). Subsequent samples use the real tick/wall deltas.

    Args:
        clock: Zero-arg monotonic-seconds reader (defaults to
            :func:`time.perf_counter`). MUST be the same clock instance passed to
            :func:`run_benchmark` so tick deltas and wall deltas are consistent.
        warmup_tps: TPS to report for the very first state (no prior tick).
            Defaults to :data:`MIN_SUSTAINED_TPS` so warm-up is gate-neutral.
        max_tps: Upper clamp for the derived value. A zero/near-zero wall delta
            (two states read at the same clock instant — common with a zero-cost
            fake clock) would otherwise divide by ~0 and report an absurd spike;
            clamping to ``max_tps`` keeps the figure sane without masking a dip
            (a dip pushes TPS DOWN, never up). Defaults to ``2 * SERVER_TPS``.
    """

    def __init__(
        self,
        clock: Callable[[], float] = time.perf_counter,
        *,
        warmup_tps: float = MIN_SUSTAINED_TPS,
        max_tps: float = 2.0 * float(SERVER_TPS),
    ) -> None:
        if max_tps <= 0.0:
            raise ValueError(f"max_tps must be > 0, got {max_tps}")
        self._clock = clock
        self._warmup_tps = float(warmup_tps)
        self._max_tps = float(max_tps)
        # Per-arena state: keyed by arena identifier (int or any hashable).
        # Each entry is (prev_tick, prev_wall). Using a dict instead of a single
        # pair so that consecutive calls from DIFFERENT arenas in a round-robin loop
        # never corrupt each other's tick/wall baseline.
        self._prev: Dict[Any, tuple] = {}

    def __call__(self, state: StateMsg, *, arena: int = 0) -> float:
        """Derive TPS for the given arena from consecutive tick/wall deltas.

        Args:
            state: The ``StateMsg`` just received for this arena.
            arena: Arena identifier (default 0 for the single-arena case). MUST be
                the same value across consecutive calls from the same arena so the
                per-arena state is accumulated correctly.

        Returns:
            Derived instantaneous TPS for this arena, clamped and guarded.
        """
        now = float(self._clock())
        tick = int(state.tick)

        if arena not in self._prev:
            # No predecessor for this arena yet — report the gate-neutral warm-up
            # value and seed this arena's state.
            self._prev[arena] = (tick, now)
            return self._warmup_tps

        prev_tick, prev_wall = self._prev[arena]
        d_tick = tick - prev_tick
        d_wall = now - prev_wall
        self._prev[arena] = (tick, now)

        if d_wall <= 0.0:
            # Same instant (or a non-monotonic clock): cannot derive a rate. Clamp
            # to max_tps rather than divide by ~0; a dip never manifests as a
            # division-by-tiny so this cannot hide a sub-floor reading.
            return self._max_tps
        if d_tick < 0:
            # Tick went backwards (server restart / counter reset): not a
            # meaningful rate. Treat as warm-up rather than a spurious dip.
            self._prev[arena] = (tick, now)
            return self._warmup_tps

        tps = d_tick / d_wall
        if tps > self._max_tps:
            return self._max_tps
        return tps


# ---------------------------------------------------------------------------
# CLI — the LIVE AC4 run (needs a started bridge + Paper server).
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="benchmark",
        description=(
            "Throughput/latency benchmark (AC4/TC12): the bridge spike's measured "
            "exit number. Connects to a LIVE Node bridge / Paper server and reports "
            "transitions/s/arena, p99 round-trip @200ms, damage-event boundary "
            "correctness, and the max arenas sustaining >=19 TPS over the run. "
            "Offline metric-logic is proved by tests/test_benchmark.py."
        ),
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_BENCH_DURATION_S,
        help=(
            "sustained run duration in seconds for the live run "
            f"(default: {DEFAULT_BENCH_DURATION_S:.0f}s == 10 min, per TC12)"
        ),
    )
    parser.add_argument(
        "--arenas",
        type=int,
        default=1,
        help="number of arenas to drive (default: 1; sweep up for the max-arenas figure)",
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
        help=(
            "base bridge TCP port for the live run (default: 5555); arena i uses "
            "port + i (one connection per arena)"
        ),
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="bench",
        help="logger run name (default: bench)",
    )
    parser.add_argument(
        "--log-backend",
        type=str,
        default="auto",
        help="metrics backend: auto|wandb|tensorboard|jsonl (default: auto)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point for the LIVE benchmark run (the AC4 number).

    Builds one real :class:`~env.mc_pvp_env.TcpBridgeClient` per arena
    (``port + arena_index``), runs :func:`run_benchmark` for ``--duration``
    seconds, logs through a :class:`~eval.logging.MetricsLogger`, prints the
    report as JSON, and EXITS WITH THE REPORTED NUMBER baked into the exit code:
    ``0`` only when the run sustained ≥19 TPS AND the damage-boundary gate
    actually ran and passed, ``1`` otherwise — so the live run is usable as a
    pass/fail gate while the printed JSON carries the full measured numbers.

    Two gates make this exit code honest rather than trivially green:

    * **TPS** — the live run derives server TPS from real ``tick`` deltas via
      :class:`TickDeltaTpsProvider` (NOT the constant offline default), so a real
      lag dip below 19 TPS trips the gate and forces exit ``1``.
    * **Damage boundary** — a deterministic known-N-hit exchange cannot be driven
      against a live opponent from this entry point (it is an offline/scripted
      cross-check fed via ``expected_hits`` + a :class:`FakeBridge`). The live run
      therefore does NOT compute ``ok``; rather than silently defaulting that gate
      to "pass", this prints a LOUD banner that the gate is INERT and treats it as
      NOT-passed, so exit ``0`` can never falsely imply the damage check ran.

    Args:
        argv: Argument vector (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code (0 = sustained ≥19 TPS and a damage gate that ran OK).
    """
    import json

    args = _build_parser().parse_args(argv)

    def factory(arena_index: int) -> BridgeTransport:
        return TcpBridgeClient(host=args.host, port=args.port + arena_index)

    logger = MetricsLogger(
        run_name=args.run_name,
        backend=args.log_backend,
        config={
            "duration_s": args.duration,
            "arenas": args.arenas,
            "host": args.host,
            "port": args.port,
        },
    )

    # Share ONE monotonic clock between the runner and the tick-delta TPS provider
    # so the provider's wall deltas line up exactly with the runner's timing.
    clock = time.perf_counter
    tps_provider = TickDeltaTpsProvider(clock=clock)

    try:
        report = run_benchmark(
            factory,
            n_arenas=args.arenas,
            duration_s=args.duration,
            clock=clock,
            tps_provider=tps_provider,  # real tick-delta TPS, not a constant
            logger=logger,
            is_live=True,
            log=lambda m: print(m, file=sys.stderr),
        )
    finally:
        logger.close()

    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))

    # Damage-boundary gate: it only ran if expected_hits was supplied, in which
    # case tally_damage_dealt() stamped an "ok" key. A live run cannot script the
    # exchange, so the gate is INERT here. Do NOT default it to True — treat a gate
    # that did not run as NOT-passed and say so loudly.
    damage = report.damage_boundary
    damage_ran = "ok" in damage
    damage_ok = bool(damage.get("ok", False))

    if not damage_ran:
        print(
            "\n"
            "================================================================\n"
            "WARNING: the damage-boundary gate is INERT on this live run.\n"
            "No known-N-hit exchange was scripted (expected_hits was not set),\n"
            "so events.damage_dealt correctness was NOT verified. This gate is\n"
            "an OFFLINE cross-check (see tests/test_benchmark.py). Exit 0 here\n"
            "reflects the TPS gate ONLY and does NOT imply the damage check ran.\n"
            "================================================================\n",
            file=sys.stderr,
        )

    tps_passed = report.sustains_19_tps
    damage_passed = damage_ran and damage_ok
    passed = tps_passed and damage_passed

    if not passed:
        reasons = []
        if not tps_passed:
            reasons.append(
                f"did not sustain >=19 TPS (min={report.sustained_tps_min:.2f})"
            )
        if not damage_ran:
            reasons.append("damage-boundary gate did not run (inert)")
        elif not damage_ok:
            reasons.append("damage-boundary check failed")
        print("FAIL: " + "; ".join(reasons), file=sys.stderr)

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
