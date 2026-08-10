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

"TPS" here is the REAL Paper server tick rate. It is derived from the learner
bot's server world age (Mineflayer ``bot.time.age``, updated only by the
server's ``update_time`` packet, so ~1/s) and averaged over a rolling
``TPS_ROLL_WINDOW_S`` window. It is NOT the client-side ``physicsTick`` timer,
which on Windows fires at ~60 ms (Mineflayer's ``setInterval(50 ms)`` under the
~15.6 ms system timer resolution) and would otherwise masquerade as a ~16 TPS
"server" reading. That same physics-timer floor separately caps raw collection
throughput (transitions/s) on Windows; lift it later via ``timeBeginPeriod(1)``
or by running the bridge on Linux/WSL.

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
import collections
import math
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from agent.contract_config import DECISION_INTERVAL_MS, SERVER_TPS
from bridge.messages import ResetAckMsg, StateMsg
from env.mc_pvp_env import BridgeError, BridgeTransport, TcpBridgeClient
from eval.combat_probe import FULL_HEALTH, StepRecord, reconcile_against_wire
from eval.logging import MetricsLogger

__all__ = [
    # Constants
    "DECISION_INTERVAL_S",
    "MIN_SUSTAINED_TPS",
    "TPS_ROLL_WINDOW_S",
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
    "SleepingFakeBridge",
    # Resource sampling
    "ResourceSampler",
    # Runner + report
    "BenchmarkReport",
    "run_benchmark",
    "main",
    # T12 -- cross-pad isolation (AC13)
    "PadLogAnchor",
    "PadForeignSighting",
    "PadLogSummary",
    "PadReconciliation",
    "PadIsolationReport",
    "PadIsolationRecorder",
    "parse_pad_log_lines",
    "parse_pad_log_file",
    "verify_pad_log",
    "reconcile_pad_damage",
    "check_pad_isolation",
    "default_pad_log_path",
    "format_isolation_line",
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

#: Rolling window (seconds) for deriving server TPS from world-age tick deltas.
#: The live tick comes from the server world age (Mineflayer ``bot.time.age``),
#: which the server updates only ~once per second, so consecutive per-state
#: deltas read "0, 0, 0, +20" and a naive diff would report false 0-TPS dips
#: between packets. Averaging the tick advance over a multi-second window (Paper's
#: own ``/tps`` is likewise a rolling average) recovers the true ~20 TPS. 5 s is
#: long enough to span several 1/s world-age updates while still catching a
#: sustained lag dip promptly.
TPS_ROLL_WINDOW_S: float = 5.0

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

    Thread-safe: the concurrent thread-per-arena driver shares ONE clock across
    arena threads, so the read/advance pair is guarded by a lock. Without it the
    ``_now += seconds`` read-add-store could interleave across threads and silently
    drop an advance, which would make the summed wall-elapsed (and therefore the
    throughput numbers asserted by the offline tests) nondeterministic.
    """

    def __init__(self, start: float = 0.0) -> None:
        self._now = float(start)
        self._lock = threading.Lock()

    def __call__(self) -> float:
        with self._lock:
            return self._now

    def advance(self, seconds: float) -> None:
        """Advance the clock by ``seconds`` (must be >= 0)."""
        if seconds < 0.0:
            raise ValueError(f"cannot advance the clock by a negative amount: {seconds}")
        with self._lock:
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

    def _next_state_latency(self) -> float:
        """Pop the next scripted round-trip latency for a ``state`` reply.

        Consumed in order; returns ``0.0`` once the scripted list is exhausted.
        Only state replies have a round-trip cost (a ``reset_ack`` has none), so
        the caller invokes this only when the popped message is a ``state``.
        """
        if self._lat_idx < len(self._latencies):
            latency = self._latencies[self._lat_idx]
            self._lat_idx += 1
            return latency
        return 0.0

    def recv(self) -> Union[StateMsg, ResetAckMsg]:
        if not self.inbound:
            raise BridgeError("FakeBridge.recv() called with an empty queue")
        msg = self.inbound.pop(0)
        # Charge the scripted round-trip latency only for state replies (the
        # step -> state round-trip the benchmark measures).
        if isinstance(msg, StateMsg):
            self._clock.advance(self._next_state_latency())
        return msg

    def close(self) -> None:
        self.closes += 1
        self.is_open = False


class SleepingFakeBridge(FakeBridge):
    """A :class:`FakeBridge` whose ``recv()`` does a REAL :func:`time.sleep`.

    The plain :class:`FakeBridge` advances a manual :class:`FakeClock` with no
    real wall-time cost, so it proves the metric MATH but cannot prove thread
    OVERLAP — there is nothing for the GIL to release on. This sibling blocks the
    calling thread on a real ``time.sleep(latency)`` inside ``recv()`` to model the
    bridge aggregating over the ``ACTION_REPEAT`` window before it replies: that
    blocking sleep releases the GIL, so the concurrent thread-per-arena driver
    overlaps the sleeps of N arenas and the aggregate transitions/s rises with N.
    This is the fixture behind the offline overlap evidence (AC8 / TC12).

    The sleep models the SERVER side of the round-trip. To prove threads still
    overlap when there is real CPU (Python/torch) work BETWEEN the sleeps — i.e.
    that the design is I/O-bound, not GIL-bound — the driver runs a per-step work
    hook (e.g. a real ``DuelingDRQN.act`` forward); see ``run_benchmark``'s
    ``step_work`` parameter. The hook lives in the driver, not here, so this
    bridge stays a pure transport.

    The measured round-trip the benchmark records is the REAL elapsed wall time of
    the sleep (read off the real ``clock`` passed to ``run_benchmark``), so the
    latency percentiles reflect the injected sleep duration. A :class:`FakeClock`
    is still required by the base constructor; it is advanced by ``latency`` as in
    the base class, but for a real-clock run the benchmark reads its own wall clock
    and the :class:`FakeClock` is incidental.

    Args:
        inbound: Same scripted inbound messages as :class:`FakeBridge`.
        latency_s: A SINGLE real sleep duration (seconds) charged on every
            ``state`` ``recv`` — the simulated server-tick blocking window
            (~0.200 s live). A scalar, not a list, because a real-clock overlap
            test wants one steady per-step blocking cost.
        clock: A :class:`FakeClock` for the base contract (advanced by ``latency``;
            incidental on a real-clock run).
    """

    def __init__(
        self,
        inbound: Sequence[Union[StateMsg, ResetAckMsg]],
        latency_s: float,
        clock: FakeClock,
    ) -> None:
        if latency_s < 0.0:
            raise ValueError(f"latency_s must be >= 0, got {latency_s}")
        # The base consumes a per-state latency LIST; feed it the single sleep
        # value broadcast across every scripted state so FakeClock advancement
        # stays consistent with the base contract.
        super().__init__(
            inbound=inbound,
            latencies_s=[float(latency_s)] * len(inbound),
            clock=clock,
        )
        self._sleep_s = float(latency_s)

    def recv(self) -> Union[StateMsg, ResetAckMsg]:
        if not self.inbound:
            raise BridgeError("SleepingFakeBridge.recv() called with an empty queue")
        msg = self.inbound.pop(0)
        if isinstance(msg, StateMsg):
            # Advance the FakeClock for base-contract consistency, then BLOCK the
            # calling thread on a real sleep. The real sleep is what releases the
            # GIL so concurrent arena threads overlap their server-tick windows.
            self._clock.advance(self._next_state_latency())
            time.sleep(self._sleep_s)
        return msg


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
        transitions_per_s_aggregate: TOTAL throughput across all arenas
            (total decisions / wall-elapsed). This is the figure that rises with
            arena count when the concurrent driver overlaps the per-arena
            server-tick waits; the per-arena field below is invariant to overlap
            and cannot show the speedup.
        transitions_per_s_per_arena: Throughput per arena (the aggregate divided by
            ``n_arenas``). Kept exactly as defined before the concurrent driver so
            live-AC4 / log consumers are unbroken.
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
        pad_isolation: T12/AC13 per-pad isolation evidence, keyed by pad index
            as a string (JSON object keys are always strings). Empty unless
            the caller opted in (the live CLI's ``--pad-log-dir``); populated
            by the caller from :class:`PadIsolationRecorder`, not by
            ``run_benchmark`` itself, so this field is always present but
            never changes ``run_benchmark``'s own behavior or timing.
    """

    duration_target_s: float = 0.0
    duration_actual_s: float = 0.0
    n_arenas: int = 0
    transitions: int = 0
    transitions_per_s_aggregate: float = 0.0
    transitions_per_s_per_arena: float = 0.0
    latency_ms: Dict[str, float] = field(default_factory=dict)
    damage_boundary: Dict[str, Any] = field(default_factory=dict)
    sustained_tps_min: float = 0.0
    sustains_19_tps: bool = False
    max_arenas_sustaining_tps: int = 0
    resources: Dict[str, Any] = field(default_factory=dict)
    is_live: bool = False
    notes: List[str] = field(default_factory=list)
    pad_isolation: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Render the report as a plain JSON-serializable dict."""
        return {
            "duration_target_s": self.duration_target_s,
            "duration_actual_s": self.duration_actual_s,
            "n_arenas": self.n_arenas,
            "transitions": self.transitions,
            "transitions_per_s_aggregate": self.transitions_per_s_aggregate,
            "transitions_per_s_per_arena": self.transitions_per_s_per_arena,
            "latency_ms": dict(self.latency_ms),
            "damage_boundary": dict(self.damage_boundary),
            "sustained_tps_min": self.sustained_tps_min,
            "sustains_19_tps": self.sustains_19_tps,
            "max_arenas_sustaining_tps": self.max_arenas_sustaining_tps,
            "resources": dict(self.resources),
            "is_live": self.is_live,
            "notes": list(self.notes),
            "pad_isolation": dict(self.pad_isolation),
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
    step_work: Optional[Callable[[int, StateMsg], None]] = None,
) -> BenchmarkReport:
    """Drive ``n_arenas`` through the bridge CONCURRENTLY and return a report.

    Each arena runs in its OWN thread, executing the same raw
    ``send(step)``/``recv()`` per-step loop the live bridge uses. The ``recv()``
    blocks the calling thread for the full ~200 ms server-tick decision window
    (the bridge aggregates over the ``ACTION_REPEAT`` window THEN replies), and a
    blocking socket wait / ``time.sleep`` releases the GIL, so N arena threads
    overlap their waits. That overlap is the whole point: the round-robin this
    replaced sent then blocking-recv'd one arena at a time, so the per-arena waits
    serialized and the aggregate transitions/s never rose with arena count.

    The round-trip latency (``clock`` after recv minus ``clock`` before send) is
    recorded per step. Server TPS is read from each ``state`` via ``tps_provider``
    (per arena, so a :class:`TickDeltaTpsProvider`'s per-arena baseline stays
    correct). Damage events are tallied across all threads for the boundary
    cross-check. The run stops when ``max_decisions`` TOTAL decisions have
    completed across all arenas (offline: a fixed budget) OR ``clock`` shows
    ``duration_s`` has elapsed (live: the ≥10-minute sustained run).

    The global decision budget is split fairly: with ``max_decisions`` set, each
    arena thread takes at most ``ceil(max_decisions / n_arenas)`` decisions, and a
    shared atomic counter caps the TOTAL at exactly ``max_decisions`` — so the
    summed wall-elapsed on a :class:`FakeClock` equals ``max_decisions`` charges
    regardless of how the threads interleave, keeping the offline numbers
    deterministic. In duration mode (``max_decisions is None``) each thread loops
    until the wall-clock deadline.

    Args:
        transport_factory: ``arena_index -> BridgeTransport``. The benchmark calls
            ``connect()`` on each transport (in its arena thread) and ``close()``
            at the end.
        n_arenas: Number of arenas to drive concurrently, one thread each (>= 1).
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
            summary metrics are logged through it when provided. Logger calls are
            serialized under a lock since the logger is not thread-safe.
        resource_sampler: Optional :class:`ResourceSampler`; sampled once per
            completed decision (under the lock). Defaults to a fresh one.
        is_live: Marks the report as a live (vs offline) run.
        log: Optional ``str -> None`` progress sink (``None`` silences it).
        step_work: Optional ``(arena_index, state) -> None`` hook run per step
            AFTER a valid ``state`` recv and OUTSIDE the shared lock, so injected
            CPU work (e.g. a real :meth:`~agent.dqn.DuelingDRQN.act` forward)
            contends for the GIL exactly as the live collector loop would. The
            default (``None``) leaves the per-step path unchanged.

    Returns:
        A populated :class:`BenchmarkReport`.

    Raises:
        ValueError: if ``n_arenas`` < 1, ``duration_s`` <= 0, or both
            ``max_decisions`` and a finite ``duration_s`` are non-positive.
        BridgeError: if any arena thread receives a non-``state`` reply to a
            ``step`` (re-raised in the calling thread after all threads join).
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

    start = clock()
    deadline = start + duration_s

    # Per-arena fair share of the global decision budget. With max_decisions set,
    # ceil(M/N) is the most any single arena ever ran under the old round-robin,
    # and N*ceil(M/N) >= M always, so the shared counter (not the per-arena cap) is
    # what binds the total to exactly M. In duration mode there is no per-arena cap;
    # the wall-clock deadline stops each thread.
    if max_decisions is not None:
        per_arena_cap: Optional[int] = -(-int(max_decisions) // n_arenas)  # ceil
    else:
        per_arena_cap = None

    # Shared, lock-guarded state across the arena threads.
    lock = threading.Lock()
    claimed = 0  # total decisions claimed across all arenas (the global budget)
    logged_step = 0  # monotonic per-decision logging index (logger is single-writer)

    # Per-arena result slots. Each thread writes ONLY into its own slot, so the
    # per-arena lists need no lock; they are combined after every thread joins.
    arena_states: List[List[StateMsg]] = [[] for _ in range(n_arenas)]
    arena_latencies: List[List[float]] = [[] for _ in range(n_arenas)]
    arena_tps: List[List[float]] = [[] for _ in range(n_arenas)]
    arena_errors: List[Optional[BaseException]] = [None for _ in range(n_arenas)]

    def _claim_decision() -> bool:
        """Atomically claim one slot of the global budget. True iff one was taken."""
        nonlocal claimed
        with lock:
            if max_decisions is not None and claimed >= max_decisions:
                return False
            claimed += 1
            return True

    def _drive_arena(arena_idx: int, transport: BridgeTransport) -> None:
        nonlocal logged_step
        local_count = 0
        action_cycle = arena_idx  # stagger the starting macro per arena
        states = arena_states[arena_idx]
        latencies = arena_latencies[arena_idx]
        tps = arena_tps[arena_idx]
        try:
            transport.connect()
            while True:
                # Stop on the wall-clock deadline (covers duration mode and a
                # budgeted run that must not overrun the duration cap).
                if clock() >= deadline:
                    break
                # Stop once this arena has taken its fair per-arena share.
                if per_arena_cap is not None and local_count >= per_arena_cap:
                    break
                # Claim a slot of the GLOBAL budget; stop if it is exhausted.
                if not _claim_decision():
                    break

                action = action_cycle % 8  # cycle 0..7 over the 8 frozen macros
                action_cycle += 1

                # Measure the step -> state round-trip OUTSIDE the lock so the
                # blocking recv windows of the N arenas overlap.
                t0 = clock()
                transport.send({"type": "step", "action": action})
                msg = transport.recv()
                t1 = clock()

                if not isinstance(msg, StateMsg):
                    raise BridgeError(
                        f"benchmark expected a state reply to step, got "
                        f"{type(msg).__name__}"
                    )

                # Injected per-step CPU work (a real act() forward in TC12). Run
                # outside the lock so it contends for the GIL like the live loop.
                if step_work is not None:
                    step_work(arena_idx, msg)

                latency = t1 - t0
                latencies.append(latency)
                states.append(msg)
                # tps_provider keeps per-arena state; pass this arena's index.
                arena_tps_value = float(tps_provider(msg, arena=arena_idx))
                tps.append(arena_tps_value)
                local_count += 1

                # Logger + resource sampler are single-writer / not thread-safe, so
                # serialize their use. Fast (no blocking), so overlap is unaffected.
                with lock:
                    logged_step += 1
                    if logger is not None:
                        logger.log(
                            {
                                "round_trip_ms": latency * 1000.0,
                                "server_tps": arena_tps_value,
                                "arena": arena_idx,
                            },
                            step=logged_step,
                        )
                    resource_sampler.sample()
        except BaseException as exc:  # capture; re-raised in the main thread
            arena_errors[arena_idx] = exc
        finally:
            try:
                transport.close()
            except (BridgeError, OSError):
                pass

    # One transport + one daemon thread per arena. Threads connect their own
    # transport so connect()/recv()/close() never cross thread boundaries.
    transports = [transport_factory(arena) for arena in range(n_arenas)]
    threads = [
        threading.Thread(
            target=_drive_arena,
            args=(arena_idx, transports[arena_idx]),
            name=f"bench-arena-{arena_idx}",
            daemon=True,
        )
        for arena_idx in range(n_arenas)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    # Surface the first worker error (if any) now that all threads have stopped.
    for exc in arena_errors:
        if exc is not None:
            raise exc

    elapsed = max(clock() - start, 1e-9)  # guard divide-by-zero on a zero-cost fake

    # Combine the per-arena slots. Order does not matter for the percentile /
    # tally / min-TPS metrics, which are all order-independent aggregations.
    states_seen: List[StateMsg] = [s for slot in arena_states for s in slot]
    latencies_s: List[float] = [x for slot in arena_latencies for x in slot]
    tps_samples: List[float] = [x for slot in arena_tps for x in slot]
    transitions = len(states_seen)

    # --- assemble the report ------------------------------------------------
    report = BenchmarkReport(
        duration_target_s=float(duration_s),
        duration_actual_s=float(elapsed),
        n_arenas=int(n_arenas),
        transitions=int(transitions),
        is_live=bool(is_live),
    )

    # transitions/s AGGREGATE: total decisions across ALL arenas over the wall
    # elapsed. This is the figure that rises with arena count once the concurrent
    # driver overlaps the per-arena server-tick waits (AC8).
    report.transitions_per_s_aggregate = transitions_per_second(transitions, elapsed)

    # transitions/s PER ARENA: the aggregate divided by the arena count. Kept as a
    # true float (exact for a non-even division like 7 / 3) and defined EXACTLY as
    # before the concurrent driver, so live-AC4 / log consumers are unbroken.
    report.transitions_per_s_per_arena = (
        report.transitions_per_s_aggregate / n_arenas
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
                "transitions_per_s_aggregate": report.transitions_per_s_aggregate,
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
        f"tps_agg={report.transitions_per_s_aggregate:.2f} "
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
    uses :class:`TickDeltaTpsProvider`, which derives TPS from the server
    world-age tick (``bot.time.age``, updated ~1/s) averaged over a rolling window.

    The ``arena`` kwarg is accepted (and ignored) so both providers share the
    same call signature and can be swapped without changes at the call site.
    """
    return float(SERVER_TPS)


class TickDeltaTpsProvider:
    """Derive server TPS from ``StateMsg.tick`` deltas over a rolling wall window.

    ``StateMsg.tick`` now carries the REAL server world age (Mineflayer
    ``bot.time.age``, set only from the server ``update_time`` packet), so it
    tracks the server's true tick rate rather than the client-side physicsTick
    timer. But the server pushes ``update_time`` only ~once per second, so the
    tick advances COARSELY: it is flat for several states, then jumps ~20. A
    per-consecutive-state diff of that coarse signal reads ``0, 0, 0, +20`` and
    would report false 0-TPS dips between packets.

    So instead of diffing consecutive states, the provider keeps a rolling window
    of ``(tick, wall)`` samples per arena and divides the tick advance across the
    whole window by its wall span::

        TPS = (tick_now - tick_window_start) / (wall_now - wall_window_start)

    over the oldest sample that is at least ``window_s`` old. This is exactly how
    Paper's own ``/tps`` reports a rolling average. A healthy 20-TPS server
    advances ~20 ticks per real second, so the window derives ~20 regardless of
    how coarsely the packet arrives; a server that falls behind (lag/overload)
    advances FEWER ticks per wall-second, so the derived TPS drops below the floor
    and the sustained-TPS gate trips. This makes the live ``sustains_19_tps`` gate
    honest, unlike the constant provider.

    The provider is a stateful callable so it can drop straight into the existing
    ``tps_provider: Callable[[StateMsg], float]`` seam. It reads wall time from an
    injected ``clock`` (the SAME monotonic clock :func:`run_benchmark` uses), so a
    test can feed synthetic ``(tick, wall_time)`` pairs and assert the exact TPS.

    Until the window has ``window_s`` of history (warm-up), the provider reports
    ``warmup_tps`` (default :data:`MIN_SUSTAINED_TPS`, i.e. the floor -- neutral,
    so a warm-up sample neither fakes a pass nor a spurious dip). Once the window
    is full it uses the real tick/wall deltas across the window.

    Args:
        clock: Zero-arg monotonic-seconds reader (defaults to
            :func:`time.perf_counter`). MUST be the same clock instance passed to
            :func:`run_benchmark` so tick deltas and wall deltas are consistent.
        warmup_tps: TPS to report while the rolling window is not yet full (and
            after a world-age reset). Defaults to :data:`MIN_SUSTAINED_TPS` so
            warm-up is gate-neutral.
        max_tps: Upper clamp for the derived value. A zero/near-zero wall span
            would otherwise divide by ~0 and report an absurd spike; clamping to
            ``max_tps`` keeps the figure sane without masking a dip (a dip pushes
            TPS DOWN, never up). Defaults to ``2 * SERVER_TPS``.
        window_s: Rolling averaging window in seconds. Defaults to
            :data:`TPS_ROLL_WINDOW_S`; must span several ~1/s world-age updates so
            the coarse tick signal averages out to the true rate.
    """

    def __init__(
        self,
        clock: Callable[[], float] = time.perf_counter,
        *,
        warmup_tps: float = MIN_SUSTAINED_TPS,
        max_tps: float = 2.0 * float(SERVER_TPS),
        window_s: float = TPS_ROLL_WINDOW_S,
    ) -> None:
        if max_tps <= 0.0:
            raise ValueError(f"max_tps must be > 0, got {max_tps}")
        if window_s <= 0.0:
            raise ValueError(f"window_s must be > 0, got {window_s}")
        self._clock = clock
        self._warmup_tps = float(warmup_tps)
        self._max_tps = float(max_tps)
        self._window_s = float(window_s)
        # Per-arena rolling window of (tick, wall) samples, keyed by arena id.
        # A deque so old samples evict cheaply off the front; each arena keeps its
        # OWN window so consecutive calls from DIFFERENT arenas in a round-robin
        # loop never mix each other's tick/wall baseline.
        self._samples: Dict[Any, "collections.deque"] = {}

    def __call__(self, state: StateMsg, *, arena: int = 0) -> float:
        """Derive TPS for the given arena from its rolling ``(tick, wall)`` window.

        Args:
            state: The ``StateMsg`` just received for this arena.
            arena: Arena identifier (default 0 for the single-arena case). MUST be
                the same value across consecutive calls from the same arena so the
                per-arena window is accumulated correctly.

        Returns:
            Derived rolling-average TPS for this arena, clamped and guarded.
        """
        now = float(self._clock())
        tick = int(state.tick)

        dq = self._samples.get(arena)
        if dq is None:
            dq = collections.deque()
            self._samples[arena] = dq
        dq.append((tick, now))

        # Evict samples older than the window, but keep the oldest one that still
        # spans at least window_s so d_wall can reach the full window. Stop once
        # dropping the front would pull the window's start inside window_s.
        while len(dq) > 1 and (now - dq[1][1]) >= self._window_s:
            dq.popleft()

        oldest_tick, oldest_wall = dq[0]
        d_tick = tick - oldest_tick
        d_wall = now - oldest_wall

        if d_wall < self._window_s:
            # Window not yet full: not enough history to average a rate. Report the
            # gate-neutral warm-up value (neither a fake pass nor a spurious dip).
            return self._warmup_tps
        if d_tick < 0:
            # World age went backwards (server restart / age reset): not a
            # meaningful rate. Drop this arena's window to just the current sample
            # and treat as warm-up rather than a spurious huge negative dip.
            dq.clear()
            dq.append((tick, now))
            return self._warmup_tps
        if d_wall <= 0.0:
            # Degenerate span (non-monotonic clock): clamp to max_tps rather than
            # divide by ~0. A dip never manifests as a division-by-tiny, so this
            # cannot hide a sub-floor reading.
            return self._max_tps

        tps = d_tick / d_wall
        if tps > self._max_tps:
            return self._max_tps
        return tps


# ---------------------------------------------------------------------------
# T12 — Cross-pad isolation (AC13).
#
# Zero cross-pad interaction over a >=10-minute N>=8 run is proven by TWO
# independent, complementary signals, both consumed here:
#
#   1. Per-pad damage RECONCILIATION -- cumulative events.damage_dealt (the
#      frozen wire channel) against the dummy's own wire-derived health loss
#      (state.opponent.health, the free cross-check the plan's Decisions
#      describe -- "kept as a free cross-check", never a production source).
#      This proves the CHANNEL is behaving (no drop, no double-count, no
#      phantom event) at fleet scale/duration, same window-level algorithm
#      T8's combat probe already validated it with. It does NOT by itself
#      prove isolation: dummy.on('health') has no attacker attribution, so a
#      foreign learner landing a hit on this pad's dummy reconciles exactly
#      as cleanly as this pad's own learner would (see bridge/bot.js's
#      _scanForeignPlayers docstring). That is exactly why signal 2 exists.
#   2. The bridge-side foreign-username SCAN -- bridge/bot.js's
#      _scanForeignPlayers() logs (to stderr only, never the wire)
#      ``[bridge] pad <i> foreign_players <name1,name2>`` whenever the
#      learner's entity view contains a player that is neither this pad's own
#      learner nor its own dummy. T9 emits it; this module only consumes it.
#
# COVERAGE CAVEAT (read before treating a clean scan as proof of anything):
# _scanForeignPlayers has exactly ONE call site, at the end of handleReset --
# it never runs mid-episode. A window with no resets in it produces ZERO scan
# lines, and zero lines is indistinguishable from "no scan ran" -- it is NOT
# evidence of zero foreign contact during that window. The live AC13 run must
# include resets throughout its >=10 minutes (a scale-ladder run naturally
# does; a pure step-only throughput pass, as run_benchmark's own action-cycle
# driver runs today, does not reset at all and therefore has NO scan
# coverage). This module never claims coverage it cannot prove: a pad log
# with no matching anchor line for its expected pad index is an ERROR, not a
# pass -- see verify_pad_log.
#
# WHY eval.combat_probe IS REUSED, NOT REWRITTEN. reconcile_against_wire (T8)
# is win-outcome-agnostic and cycle-count-agnostic already -- it only matches
# recorded hits against wire-health drops within +/-1 decision window (the
# dummy's update_health arrives on a second connection) and flags any
# unexplained drop or off-death heal. That is exactly T12's fine-grained
# defect detector, so it is imported and used VERBATIM, unmodified, rather
# than inventing a second, differently-shaped mechanism. combat_probe.py is
# NOT edited by this task (T8's file; T13 also reads it) -- see
# _states_to_step_records below for the one adapter this requires.
#
# combat_probe.CycleRecord is deliberately NOT reused: its reset_ms /
# start_self_pos / start_opp_pos / outcome fields describe one deterministic
# reset/kill cycle and have no meaning for a continuous run (run_benchmark's
# arena driver never calls env.reset() -- see its own docstring above). T12
# needs only the per-window StepRecord shape, so that is all that is reused.
#
# WHAT reconcile_against_wire DOES NOT GIVE US: a whole-run cumulative total.
# It only ever needed to check per-cycle hits against a KNOWN expected
# sequence (AC8's fixed 6,6,6,2 / total 20). AC13 asks for something
# reconcile_against_wire was never shaped to answer -- "cumulative
# damage_dealt vs. per-pad dummy health loss" over an arbitrary, unbounded
# live run -- so _wire_health_loss below reproduces just enough of the same
# windowing (health_entering + the doImmediateRespawn masked-killing-blow
# carve-out) to sum a total, rather than asking combat_probe.py to grow a
# return shape it has no other use for.
# ---------------------------------------------------------------------------

#: Float comparison tolerance for T12's reconciliation arithmetic. Matches
#: eval.combat_probe's own (private) tolerance; damage in the live matchup is
#: exact multiples of 0.5, so 1e-6 is generous.
_RECONCILE_TOL: float = 1e-6

#: Verbatim log lines this module consumes. NEVER edit these patterns without
#: re-reading the literal template strings at their source:
#:   bridge/run.js  (the anchor line, printed once per bridge boot)
#:   bridge/bot.js  (_scanForeignPlayers -- the foreign_players line, printed
#:                   only when >= 1 foreign player was seen, once per reset)
_PAD_ANCHOR_LOG_RE = re.compile(
    r"^\[bridge\] pad (?P<pad>\d+) @ anchor (?P<x>-?\d+),(?P<z>-?\d+) "
    r"\((?P<learner>\S+) / (?P<dummy>\S+)\)\s*$"
)
_FOREIGN_PLAYERS_LOG_RE = re.compile(
    r"^\[bridge\] pad (?P<pad>\d+) foreign_players (?P<names>\S+)\s*$"
)


def _states_to_step_records(states: Sequence[StateMsg]) -> List[StepRecord]:
    """Adapt raw per-decision ``StateMsg`` into eval.combat_probe's ``StepRecord``.

    ``action`` has no meaning outside combat_probe's scripted ATTACK/IDLE
    driver (this module's own action cycling is a throughput exercise, not a
    combat script), so it is recorded as ``-1``; ``reconcile_against_wire``
    never reads it.
    """
    return [
        StepRecord(
            action=-1,
            damage_dealt=float(s.events.damage_dealt),
            opponent_died=bool(s.events.opponent_died),
            wire_health=float(s.opponent.health),
            attack_cooldown=float(s.self_state.attack_cooldown),
            tick=int(s.tick),
        )
        for s in states
    ]


def _wire_health_loss(start_health: float, steps: Sequence[StepRecord]) -> float:
    """Independently sum how much wire health the dummy actually lost.

    Mirrors the windowing ``eval.combat_probe.reconcile_against_wire`` uses
    (``health_entering`` plus the ``doImmediateRespawn`` masked-killing-blow
    carve-out: a death that snaps health back to full inside the SAME window
    it occurred in leaves no visible drop, so the window's ENTERING health is
    exactly what was lost) but returns a TOTAL rather than matching individual
    hits -- combat_probe never needed a running total since AC8's expected
    total is the fixed constant 20.0.

    A VISIBLE drop within a death window takes precedence over the masked-kill
    carve-out (checked first, per window), mirroring reconcile_against_wire's
    own precedence of matching a real drop before assuming an invisible one --
    this is what keeps a skewed-but-visible fatal drop from being counted
    twice (once as a plain drop, once as a masked kill).

    WHY THAT PRECEDENCE IS SOUND (pinned at its source, not left as an
    assertion): the dummy's ``health`` and ``death`` events are both emitted
    synchronously off ONE ``update_health`` packet handler in mineflayer
    (``mineflayer/lib/plugins/health.js``) -- there is no code path where a
    death fires before its own health-drop packet is processed. The
    respawn's health=20 packet is a SEPARATE, strictly LATER
    ``update_health`` (the server never folds a respawn into the same packet
    as the fatal hit), so "hit this window, death this window, respawn this
    window" is at minimum two packets apart even in the fastest observed
    case. That ordering is what guarantees a visible drop at window w is
    always caught by THIS window's own ``delta > _RECONCILE_TOL`` branch
    before the masked-kill branch could ever double-add it a window away --
    so a future reader chasing a double-count regression needs three minutes
    with this docstring and health.js, not an hour of live packet tracing.

    Args:
        start_health: Wire opponent health entering the FIRST recorded window.
        steps: Per-decision-window records in order.

    Returns:
        Total positive wire-health loss across ``steps`` (never negative).
    """
    n = len(steps)
    health_entering = [float(start_health)] + [float(s.wire_health) for s in steps[:-1]]
    death_windows = {w for w in range(n) if steps[w].opponent_died}

    total = 0.0
    for w in range(n):
        delta = health_entering[w] - float(steps[w].wire_health)
        if delta > _RECONCILE_TOL:
            total += delta
        elif w in death_windows:
            total += health_entering[w]
        # A negative delta outside a death window is a heal; that is a defect
        # eval.combat_probe.reconcile_against_wire already flags separately
        # (regeneration is off), and it contributes no loss here either way.
    return total


@dataclass
class PadReconciliation:
    """T12/AC13's per-pad damage reconciliation for one pad's recorded run.

    Attributes:
        pad_index: 0-based pad index this reconciliation covers.
        n_windows: Number of decision windows reconciled.
        start_health: The wire opponent health assumed entering window 0.
        cumulative_damage_dealt: Sum of ``events.damage_dealt`` across the run
            (the frozen production channel T2 repaired).
        cumulative_wire_health_loss: Independently wire-derived total health
            lost (see :func:`_wire_health_loss`) -- the free cross-check.
        trailing_residual_allowance: The single trailing-window hit amount (if
            any) exempted from the cumulative-total check, because its wire
            confirmation may not have arrived by the time recording stopped
            (there is no next window in a truncated recording to catch a
            +/-1-window-skewed drop). Zero when the last window had no hit.
        errors: Human-readable failures (empty == clean reconciliation).
    """

    pad_index: int
    n_windows: int
    start_health: float
    cumulative_damage_dealt: float
    cumulative_wire_health_loss: float
    trailing_residual_allowance: float
    errors: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def reconcile_pad_damage(
    pad_index: int,
    states: Sequence[StateMsg],
    *,
    start_health: float = FULL_HEALTH,
) -> PadReconciliation:
    """Reconcile one pad's recorded ``events.damage_dealt`` against its wire health.

    ``start_health`` defaults to :data:`~eval.combat_probe.FULL_HEALTH`
    (20.0) rather than the first recorded sample: this driver never calls
    ``env.reset()`` (see ``run_benchmark``), so there is no post-reset state to
    read a baseline from the way ``eval.combat_probe`` does. The default is
    sound only if the pad's dummy was already at a clean, stationary baseline
    when recording started -- true for a PRIMED fleet per AC18 (see
    ``server/setup/start-pads.sh``). An unprimed or mid-episode pad then fails
    LOUDLY here (a spurious window-0 drop/heal), which is the correct failure
    mode: never silently assume a clean start.

    Args:
        pad_index: 0-based pad index (used only to label error messages).
        states: Raw per-decision ``StateMsg`` for this pad, in order.
        start_health: Wire opponent health assumed entering the first window.

    Returns:
        A populated :class:`PadReconciliation`.
    """
    steps = _states_to_step_records(states)

    if not steps:
        return PadReconciliation(
            pad_index=pad_index,
            n_windows=0,
            start_health=float(start_health),
            cumulative_damage_dealt=0.0,
            cumulative_wire_health_loss=0.0,
            trailing_residual_allowance=0.0,
            errors=[
                f"pad {pad_index}: no decision windows recorded; damage cannot "
                f"be reconciled (no evidence, not proof of isolation)"
            ],
        )

    # T8's window-level defect detector, reused verbatim (see the section
    # docstring above): unrecorded hits, phantom hits, and off-death heals.
    errors = list(reconcile_against_wire(start_health, steps))

    cumulative_damage = sum(s.damage_dealt for s in steps)
    cumulative_loss = _wire_health_loss(start_health, steps)

    trailing_hit = 0.0
    if steps[-1].damage_dealt > _RECONCILE_TOL:
        trailing_hit = float(steps[-1].damage_dealt)

    # S1: the trailing-window allowance is ONE-SIDED on purpose. A POSITIVE
    # residual (damage_dealt ahead of wire-derived loss) has a legitimate
    # boundary excuse: the last window's hit may not have its wire
    # confirmation yet (see the docstring). A NEGATIVE residual (wire loss
    # ahead of damage_dealt) has no such excuse at any magnitude -- it means
    # the wire lost health with no recorded event to explain it, which is
    # exactly the unrecorded-hit / under-counting failure this whole plan
    # exists to catch, and hiding a small one behind the same allowance would
    # reopen that hole. So only a positive residual gets the trailing-hit
    # slack; any negative residual beyond float tolerance is always flagged.
    residual = cumulative_damage - cumulative_loss
    if residual > trailing_hit + _RECONCILE_TOL or residual < -_RECONCILE_TOL:
        errors.append(
            f"pad {pad_index}: cumulative damage_dealt {cumulative_damage:g} != "
            f"cumulative wire-derived health loss {cumulative_loss:g} "
            f"(residual {residual:g}; positive residuals are allowed up to the "
            f"{trailing_hit:g} trailing-window allowance, negative residuals are "
            f"never allowed)"
        )

    return PadReconciliation(
        pad_index=pad_index,
        n_windows=len(steps),
        start_health=float(start_health),
        cumulative_damage_dealt=cumulative_damage,
        cumulative_wire_health_loss=cumulative_loss,
        trailing_residual_allowance=trailing_hit,
        errors=errors,
    )


@dataclass(frozen=True)
class PadLogAnchor:
    """One ``[bridge] pad <i> @ anchor <x>,<z> (<learner> / <dummy>)`` line."""

    pad_index: int
    anchor_x: int
    anchor_z: int
    learner: str
    dummy: str


@dataclass(frozen=True)
class PadForeignSighting:
    """One ``[bridge] pad <i> foreign_players <names>`` line (>= 1 foreign name)."""

    pad_index: int
    names: Tuple[str, ...]


@dataclass
class PadLogSummary:
    """Parsed evidence from one pad's bridge stdout+stderr log file.

    ``server/setup/start-pads.sh`` redirects each pad bridge's combined
    stdout+stderr into one file (``>pad_log 2>&1``), so both the anchor line
    (stderr) and any foreign_players lines (stderr) live in the same stream
    this parses. Any other line -- bridge lifecycle logs, JS stack traces, a
    malformed write, a mid-write truncation -- simply matches neither pattern
    and is skipped; the parser never raises on unrecognized content.
    """

    lines_scanned: int = 0
    anchors: List[PadLogAnchor] = field(default_factory=list)
    foreign_sightings: List[PadForeignSighting] = field(default_factory=list)


def parse_pad_log_lines(lines: Iterable[str]) -> PadLogSummary:
    """Parse an iterable of raw log lines into a :class:`PadLogSummary`.

    Pure and socket-free so it is fully unit-testable against fixtures: a
    clean line, a foreign-player line, a malformed/garbled line, and a
    truncated final line all pass through this the same way -- match one of
    the two known patterns, or skip.
    """
    summary = PadLogSummary()
    for raw in lines:
        summary.lines_scanned += 1
        line = raw.rstrip("\r\n")

        m = _PAD_ANCHOR_LOG_RE.match(line)
        if m is not None:
            summary.anchors.append(
                PadLogAnchor(
                    pad_index=int(m.group("pad")),
                    anchor_x=int(m.group("x")),
                    anchor_z=int(m.group("z")),
                    learner=m.group("learner"),
                    dummy=m.group("dummy"),
                )
            )
            continue

        m = _FOREIGN_PLAYERS_LOG_RE.match(line)
        if m is not None:
            names = tuple(n for n in m.group("names").split(",") if n)
            summary.foreign_sightings.append(
                PadForeignSighting(pad_index=int(m.group("pad")), names=names)
            )
            continue

        # Neither pattern matched (bridge lifecycle noise, a malformed or
        # truncated line, a decode-mangled line) -- not evidence, not an
        # error. See the module-level note: absence of a foreign_players line
        # is never treated as proof of a clean scan on its own.
    return summary


def parse_pad_log_file(path: Union[str, Path]) -> PadLogSummary:
    """Read and parse one pad's bridge log file.

    Args:
        path: Path to the per-pad log file (see :func:`default_pad_log_path`
            for the ``server/setup/start-pads.sh`` naming convention).

    Returns:
        A populated :class:`PadLogSummary`.

    Raises:
        FileNotFoundError: if ``path`` does not exist. A missing log is an
            ERROR, never a silent "zero foreign players" pass -- absence of
            evidence is not evidence of absence, and that is the entire point
            of AC13.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"pad log not found: {p}")
    # errors="replace": the live bridge process may still be appending to this
    # file (or was killed mid multi-byte UTF-8 sequence at EOF); a decode
    # error must degrade that one trailing line to unmatched noise, never
    # crash the whole parse.
    with p.open("r", encoding="utf-8", errors="replace") as f:
        return parse_pad_log_lines(f)


def default_pad_log_path(log_dir: Union[str, Path], pad_index: int) -> Path:
    """The per-pad bridge log path, mirroring ``server/setup/start-pads.sh``.

    That script writes ``${LOG_DIR:-<server>/logs/pads}/pad-<i>.log`` per pad
    (combined stdout+stderr); this is the read-side mirror of that convention,
    not a second definition of it -- the write side is start-pads.sh's alone.
    """
    return Path(log_dir) / f"pad-{pad_index}.log"


def verify_pad_log(summary: PadLogSummary, expected_pad_index: int) -> List[str]:
    """Assert this log can be trusted as isolation evidence for ``expected_pad_index``.

    A pad-log-dir mistake (wrong directory, off-by-one pad numbering, an
    empty or rotated-away file) must never silently read as "clean" -- AC13
    exists to prove ABSENCE of foreign contact, and absence of a
    foreign_players line proves nothing unless this log can first be shown to
    be genuinely THIS pad's own boot log. A log with no matching anchor line
    is therefore always an error, never a pass.

    Args:
        summary: Parsed log evidence (see :func:`parse_pad_log_file`).
        expected_pad_index: The pad index this log is supposed to belong to.

    Returns:
        Human-readable failures (empty == this log verifiably belongs to
        ``expected_pad_index`` and only that pad).
    """
    errors: List[str] = []
    matching = [a for a in summary.anchors if a.pad_index == expected_pad_index]
    if not matching:
        errors.append(
            f"pad {expected_pad_index}: no matching '@ anchor' boot line found in "
            f"this log ({summary.lines_scanned} line(s) scanned) -- cannot trust "
            f"this file as isolation evidence for this pad"
        )
    for other in summary.anchors:
        if other.pad_index != expected_pad_index:
            errors.append(
                f"pad {expected_pad_index}: log also contains a boot line for pad "
                f"{other.pad_index} -- wrong file, or pads sharing one log"
            )
    return errors


@dataclass
class PadIsolationReport:
    """T12/AC13's combined verdict for one pad: reconciliation + foreign scan.

    Attributes:
        pad_index: 0-based pad index.
        reconciliation: The damage-vs-wire-health reconciliation (signal 1).
        log_summary: The parsed bridge log (signal 2's raw evidence).
        log_errors: Failures from :func:`verify_pad_log` (an untrustworthy log
            file, distinct from an actual foreign-player sighting).
    """

    pad_index: int
    reconciliation: PadReconciliation
    log_summary: PadLogSummary
    log_errors: List[str] = field(default_factory=list)

    @property
    def foreign_sightings(self) -> List[PadForeignSighting]:
        return [
            s for s in self.log_summary.foreign_sightings if s.pad_index == self.pad_index
        ]

    @property
    def ok(self) -> bool:
        return not self.reconciliation.errors and not self.log_errors and not self.foreign_sightings

    def violations(self) -> List[str]:
        """All human-readable failures, reconciliation first, then log issues."""
        out = list(self.reconciliation.errors)
        out.extend(self.log_errors)
        out.extend(
            f"pad {self.pad_index}: foreign player(s) seen in this pad's entity "
            f"view: {', '.join(s.names)}"
            for s in self.foreign_sightings
        )
        return out


#: Grace period (seconds) subtracted from ``check_pad_isolation``'s
#: ``min_mtime`` freshness threshold, to absorb filesystem mtime granularity
#: and small clock skew between the OS and Python's ``time.time()``. This is
#: NOT slack against a genuinely stale (previous-boot) log -- a real leftover
#: predates the run by far more than this; it exists only so a log a live
#: bridge touches within the same instant this check runs is never flagged on
#: a rounding artifact.
_MTIME_FRESHNESS_SLACK_S: float = 2.0


def check_pad_isolation(
    pad_index: int,
    states: Sequence[StateMsg],
    log_path: Union[str, Path],
    *,
    start_health: float = FULL_HEALTH,
    min_mtime: Optional[float] = None,
) -> PadIsolationReport:
    """Run T12/AC13's full per-pad check: reconciliation + log consumption.

    Args:
        pad_index: 0-based pad index.
        states: This pad's raw per-decision ``StateMsg`` sequence, in order.
        log_path: Path to this pad's bridge log file.
        start_health: Wire opponent health assumed entering window 0 (see
            :func:`reconcile_pad_damage`).
        min_mtime: If given (epoch seconds, e.g. ``time.time()`` captured
            just before the live run started), the log file's mtime must be
            at or after ``min_mtime - _MTIME_FRESHNESS_SLACK_S`` or the log is
            rejected as STALE. ``verify_pad_log`` proves IDENTITY (this log
            genuinely belongs to this pad) but not FRESHNESS: a leftover
            ``pad-<i>.log`` from a previous, smaller/different boot can carry
            a perfectly valid anchor line and no ``foreign_players`` lines,
            reporting a silent clean pass for a pad that no longer exists.
            This check closes that gap using a real side effect: connecting
            this run's transport to a genuinely live, correctly-addressed
            bridge always appends an ``env connected`` line to that bridge's
            own log (``bridge/run.js``), so a live pad's log is always fresh
            by the time this runs. ``None`` (the default) skips the check
            entirely -- callers that cannot supply a trustworthy ``min_mtime``
            (e.g. offline tests) get identity verification only, same as
            before this parameter existed.

    Returns:
        A populated :class:`PadIsolationReport`.

    Raises:
        FileNotFoundError: propagated from :func:`parse_pad_log_file` if
            ``log_path`` does not exist.
    """
    reconciliation = reconcile_pad_damage(pad_index, states, start_health=start_health)
    summary = parse_pad_log_file(log_path)
    log_errors = verify_pad_log(summary, pad_index)

    if min_mtime is not None:
        mtime = Path(log_path).stat().st_mtime
        threshold = float(min_mtime) - _MTIME_FRESHNESS_SLACK_S
        if mtime < threshold:
            log_errors.append(
                f"pad {pad_index}: log file {log_path} was last modified "
                f"{threshold - mtime:.1f}s before this run started (mtime "
                f"{mtime:.3f} < run-start threshold {threshold:.3f}) -- STALE "
                f"evidence, likely a leftover from a previous boot; refusing "
                f"to treat it as current"
            )

    return PadIsolationReport(
        pad_index=pad_index,
        reconciliation=reconciliation,
        log_summary=summary,
        log_errors=log_errors,
    )


def format_isolation_line(report: PadIsolationReport) -> str:
    """One human-readable summary line per pad, in combat_probe's log style."""
    r = report.reconciliation
    return (
        f"[isolation] pad {report.pad_index}: windows={r.n_windows} "
        f"dealt={r.cumulative_damage_dealt:g} wire_loss={r.cumulative_wire_health_loss:g} "
        f"foreign_events={len(report.foreign_sightings)} "
        f"{'OK' if report.ok else 'FAIL'}"
    )


class PadIsolationRecorder:
    """Records live per-decision states, per pad, for T12 reconciliation.

    Drop-in as ``run_benchmark``'s ``step_work`` hook (its exact
    ``(arena_index, state) -> None`` signature) to capture the raw per-pad
    ``StateMsg`` sequence WITHOUT modifying ``run_benchmark`` at all: the
    concurrent driver already flattens ``arena_states`` across every arena
    before assembling the fleet-wide report (``report.damage_boundary`` sums
    across ALL arenas), so per-pad identity would otherwise be lost by the
    time a caller sees the report.

    Each arena thread appends only to its OWN pre-sized slot (mirroring
    ``run_benchmark``'s own ``arena_states`` pattern), so no lock is needed --
    this keeps the hook cheap enough to run inside ``step_work``'s per-step,
    outside-the-shared-lock hot path without adding contention.
    """

    def __init__(self, n_arenas: int) -> None:
        if n_arenas < 1:
            raise ValueError(f"n_arenas must be >= 1, got {n_arenas}")
        self._states: List[List[StateMsg]] = [[] for _ in range(n_arenas)]

    def record(self, arena_index: int, state: StateMsg) -> None:
        """Matches ``run_benchmark``'s ``step_work`` signature exactly."""
        self._states[arena_index].append(state)

    def states_for(self, arena_index: int) -> List[StateMsg]:
        return list(self._states[arena_index])

    def check_all(
        self,
        log_paths: Mapping[int, Union[str, Path]],
        *,
        start_health: float = FULL_HEALTH,
        pad_index_for_arena: Optional[Callable[[int], int]] = None,
        min_mtime: Optional[float] = None,
    ) -> Dict[int, PadIsolationReport]:
        """Run :func:`check_pad_isolation` for every recorded arena.

        Args:
            log_paths: ``arena_index -> log file path``, must cover every
                arena this recorder was constructed with.
            start_health: Forwarded to :func:`reconcile_pad_damage`.
            pad_index_for_arena: ``arena_index -> TRUE pad index``, used to
                label each report AND as ``verify_pad_log``'s expected pad
                index. Defaults to the identity function (arena i IS pad i) --
                true only when this run's arena 0 is genuinely connected to
                pad 0's bridge port. A caller that points ``--port`` at a
                SUBSET or offset of the fleet (e.g. pads 2..5) MUST supply a
                resolver, or this call would compare each log's anchor line
                against the wrong expected pad index -- ``verify_pad_log``
                would then fail loudly rather than silently misattribute, but
                it is better not to construct that mismatch in the first
                place. See ``eval.benchmark.main``'s ``--bridge-base-port``.
            min_mtime: Forwarded to :func:`check_pad_isolation` for every pad
                (freshness check; ``None`` skips it).

        Returns:
            ``{arena_index: PadIsolationReport}`` (keyed by ARENA index, same
            as ``log_paths``; each report's own ``.pad_index`` carries the
            resolved TRUE pad index).

        Raises:
            KeyError: if ``log_paths`` is missing an entry for a recorded
                arena -- never silently skip a pad's isolation evidence.
        """
        resolve = pad_index_for_arena or (lambda arena_index: arena_index)
        reports: Dict[int, PadIsolationReport] = {}
        for arena_index, states in enumerate(self._states):
            if arena_index not in log_paths:
                raise KeyError(f"no log path supplied for arena {arena_index}")
            reports[arena_index] = check_pad_isolation(
                resolve(arena_index),
                states,
                log_paths[arena_index],
                start_health=start_health,
                min_mtime=min_mtime,
            )
        return reports


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
    parser.add_argument(
        "--pad-log-dir",
        type=str,
        default=None,
        metavar="DIR",
        help=(
            "T12/AC13 cross-pad isolation: directory of per-pad bridge logs "
            "(server/setup/start-pads.sh's --log-dir, default <server>/logs/pads; "
            "pad i's file is DIR/pad-<i>.log, where i is (--port - "
            "--bridge-base-port) + arena_index -- see --bridge-base-port). When "
            "given, this run ALSO reconciles cumulative damage_dealt against "
            "each pad's own wire-derived dummy health loss and consumes each "
            "pad's foreign-username scan, printing one [isolation] line per "
            "pad and folding isolation violations into the exit code. Omit to "
            "leave this run's behavior BEHAVIORALLY identical to today (the "
            "printed JSON always carries a pad_isolation key; it is just "
            "empty). NOTE: the foreign-player scan fires only on reset "
            "(bridge/bot.js _scanForeignPlayers) -- a run with no resets in "
            "the window has zero SCAN COVERAGE, not zero violations."
        ),
    )
    parser.add_argument(
        "--bridge-base-port",
        type=int,
        default=5555,
        metavar="PORT",
        help=(
            "the FLEET's bridge base port -- pad 0's port (matches "
            "server/setup/start-pads.sh's --bridge-base-port / "
            "distributed/launcher.py's DEFAULT_BRIDGE_BASE_PORT, default "
            "5555). Only meaningful with --pad-log-dir: this run's arena i "
            "connects to --port + i, and its TRUE pad index is "
            "(--port - --bridge-base-port) + i, so pointing --port at a "
            "SUBSET of the fleet (e.g. --port 5557 to benchmark pads 2..5) "
            "still reads and labels the correct per-pad logs instead of "
            "silently reading pad-0.log for what is actually pad 2. A --port "
            "below --bridge-base-port is refused loudly at startup rather "
            "than resolving to a negative pad index."
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point for the LIVE benchmark run (the AC4 number).

    Builds one real :class:`~env.mc_pvp_env.TcpBridgeClient` per arena
    (``port + arena_index``), runs :func:`run_benchmark` for ``--duration``
    seconds, logs through a :class:`~eval.logging.MetricsLogger`, prints the
    report as JSON, and EXITS WITH THE REPORTED NUMBER baked into the exit
    code: ``0`` when the run sustained ≥19 TPS, the damage-boundary gate
    either ran and passed OR is inert (never scripted from this live entry
    point), and, if requested, isolation is clean on every pad; ``1``
    otherwise — so the live run is usable as a pass/fail gate while the
    printed JSON carries the full measured numbers.

    Gates that make this exit code honest rather than trivially green:

    * **TPS** — the live run derives server TPS from real ``tick`` deltas via
      :class:`TickDeltaTpsProvider` (NOT the constant offline default), so a real
      lag dip below 19 TPS trips the gate and forces exit ``1``.
    * **Damage boundary** — a deterministic known-N-hit exchange cannot be
      driven against a live opponent from this entry point (it is an
      offline/scripted cross-check fed via ``expected_hits`` + a
      :class:`FakeBridge`, and this entry point never sets ``expected_hits`` —
      a live run cannot know N a priori). So this gate is always INERT here: a
      LOUD banner says so, the printed JSON stamps ``damage_boundary.inert``
      explicitly, and — because a gate that never ran cannot be graded — it is
      EXCLUDED from the exit code rather than forced to fail (exit ``0`` can
      still happen; it truthfully means "TPS passed and the damage gate did
      not run", exactly what the banner says). A gate that DID run (a
      caller-supplied ``expected_hits``, e.g. a future harness) and FAILED
      still fails the exit code — inert and failed are different outcomes.

    With ``--pad-log-dir``, this run ALSO performs T12/AC13's cross-pad
    isolation check over the SAME live run (see :class:`PadIsolationRecorder`):
    per-pad reconciliation folds into the exit code alongside the TPS/damage
    gates, and one ``[isolation]`` line per pad is printed to stderr. Omitting
    the flag leaves this function's behavior BEHAVIORALLY identical to before
    T12 (not byte-identical: the printed JSON always carries a
    ``pad_isolation`` key now, empty when unused).

    Args:
        argv: Argument vector (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code (0 = sustained ≥19 TPS, the damage gate did not
        fail (ran-and-passed or inert), and, if requested, isolation clean on
        every pad).
    """
    import json

    args = _build_parser().parse_args(argv)

    def factory(arena_index: int) -> BridgeTransport:
        return TcpBridgeClient(host=args.host, port=args.port + arena_index)

    # T12/AC13, opt-in only (--pad-log-dir). Validated and constructed BEFORE
    # any resource (the MetricsLogger, the live run) is acquired, so a bad
    # --arenas or a --port/--bridge-base-port mismatch fails loudly with
    # nothing left over to leak or tear down.
    recorder: Optional[PadIsolationRecorder] = None
    pad_offset = 0
    if args.pad_log_dir is not None:
        # W3: this run's arena i connects to --port + i, but a --pad-log-dir
        # read keyed by raw arena index silently assumes arena i IS pad i.
        # --port and the fleet's --bridge-base-port are independently
        # configurable (both default 5555), so e.g. --port 5557 --arenas 4
        # drives pads 2..5 while a raw-index read would fetch logs 0..3 --
        # misrouted evidence, worse than none, and verify_pad_log cannot
        # catch it (pad-0.log legitimately contains pad 0's own anchor). The
        # offset below derives the TRUE pad index instead of assuming one.
        pad_offset = args.port - args.bridge_base_port
        if pad_offset < 0:
            print(
                f"[isolation] ABORT: --port {args.port} is below --bridge-base-port "
                f"{args.bridge_base_port}; arena 0 would resolve to a negative pad "
                f"index. Misrouted evidence is worse than none -- refusing to "
                f"start. Pass the fleet's real --bridge-base-port (default 5555), "
                f"or point --port at (or above) it.",
                file=sys.stderr,
            )
            return 1
        try:
            recorder = PadIsolationRecorder(args.arenas)
        except ValueError as exc:
            print(f"[isolation] ABORT: {exc}", file=sys.stderr)
            return 1

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

    # Real wall-clock run-start marker (deliberately NOT the injectable
    # `clock` above, which can be fake in tests) -- the freshness check below
    # compares it against each log file's real filesystem mtime.
    run_start_wall = time.time()

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
            step_work=(recorder.record if recorder is not None else None),
        )
    finally:
        logger.close()

    isolation_passed = True
    if recorder is not None:
        print(
            "\n"
            "[isolation] NOTE: the foreign-player scan fires ONLY on reset "
            "(bridge/bot.js _scanForeignPlayers); a window with no resets in "
            "it has ZERO scan coverage, not zero violations.\n",
            file=sys.stderr,
        )
        pad_index_for_arena = lambda arena_index: pad_offset + arena_index  # noqa: E731
        log_paths = {
            i: default_pad_log_path(args.pad_log_dir, pad_index_for_arena(i))
            for i in range(args.arenas)
        }
        try:
            isolation_reports = recorder.check_all(
                log_paths,
                pad_index_for_arena=pad_index_for_arena,
                min_mtime=run_start_wall,
            )
        except FileNotFoundError as exc:
            print(f"[isolation] ABORT: {exc}", file=sys.stderr)
            isolation_reports = {}
            isolation_passed = False
        for i in sorted(isolation_reports):
            r = isolation_reports[i]
            print(format_isolation_line(r), file=sys.stderr)
            for violation in r.violations():
                print(f"    FAIL: {violation}", file=sys.stderr)
        if isolation_reports:
            isolation_passed = isolation_passed and all(
                r.ok for r in isolation_reports.values()
            )
        report.pad_isolation = {
            str(r.pad_index): {
                "ok": r.ok,
                "windows": r.reconciliation.n_windows,
                "cumulative_damage_dealt": r.reconciliation.cumulative_damage_dealt,
                "cumulative_wire_health_loss": r.reconciliation.cumulative_wire_health_loss,
                "foreign_events": len(r.foreign_sightings),
                "violations": r.violations(),
            }
            for r in isolation_reports.values()
        }

    # Damage-boundary gate: it only ran if expected_hits was supplied, in which
    # case tally_damage_dealt() stamped an "ok" key. A live run cannot script the
    # exchange, so the gate is INERT here. Do NOT default it to "passed" -- but
    # do NOT force it to "failed" either: a gate that never ran cannot be
    # graded, and forcing it to fail made exit 0 unreachable on this entry
    # point, contradicting the banner's own text. Computed and stamped into the
    # report BEFORE the JSON print below, so the artifact records it explicitly.
    damage = report.damage_boundary
    damage_ran = "ok" in damage
    damage_ok = bool(damage.get("ok", False))
    damage["inert"] = not damage_ran  # explicit, machine-readable, in the JSON artifact

    if not damage_ran:
        print(
            "\n"
            "================================================================\n"
            "WARNING: the damage-boundary gate is INERT on this live run.\n"
            "No known-N-hit exchange was scripted (expected_hits was not set),\n"
            "so events.damage_dealt correctness was NOT verified. This gate is\n"
            "an OFFLINE cross-check (see tests/test_benchmark.py). It is EXCLUDED\n"
            "from the exit code (inert, not failed) -- exit 0 here reflects the\n"
            "TPS gate (and isolation, if requested) and does NOT imply the damage\n"
            "check ran.\n"
            "================================================================\n",
            file=sys.stderr,
        )

    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))

    tps_passed = report.sustains_19_tps
    damage_passed = damage_ok if damage_ran else True  # inert never forces a failure
    passed = tps_passed and damage_passed and isolation_passed

    if not passed:
        reasons = []
        if not tps_passed:
            reasons.append(
                f"did not sustain >=19 TPS (min={report.sustained_tps_min:.2f})"
            )
        if damage_ran and not damage_ok:
            reasons.append("damage-boundary check failed")
        if not isolation_passed:
            reasons.append("cross-pad isolation check failed (see [isolation] lines above)")
        print("FAIL: " + "; ".join(reasons), file=sys.stderr)

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
