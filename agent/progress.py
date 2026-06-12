"""progress — a live status bar + periodic progress log for the M2 train loop.

The T20 live loop (:func:`agent.train.train_vs_dummy`) collects MANY episodes
against the live bridge but only prints a line at each periodic eval (every
``eval_every_episodes`` episodes). Live episodes are slow — on the dev box each
one is ~90-100 s — so between evals the run was silent and there was no way to
gauge throughput or remaining time.

:class:`ProgressReporter` closes that gap. Fed one :meth:`~ProgressReporter.update`
per collected episode it:

  * renders a live, in-place status BAR on a TTY (carriage-return redraw,
    throttled to ``redraw_interval`` s so a fast loop never thrashes the
    terminal), and
  * emits a periodic full progress LINE (every ``log_interval`` s) that persists
    in scrollback / a redirected log file, returning a :class:`ProgressSnapshot`
    so the caller can also write ``progress/*`` rows to the run's MetricsLogger.

Throughput is measured over a SLIDING window of the most recent episodes (see
``rate_window_s``), not the whole run, so the rate — and therefore the ETA —
tracks the CURRENT pace instead of being dragged down by the slow replay warm-up.
The ETA is the time to exhaust the EPISODE BUDGET (``total_episodes``); it is an
honest worst-case upper bound, because the run usually stops earlier the moment a
greedy eval clears the M2 gate.

Reliability first: this is monitoring code wrapped around a multi-hour run, so a
formatting/encoding hiccup must NEVER take the run down. All terminal writes go
through :meth:`~ProgressReporter._safe_write`, which falls back to an ASCII-safe
re-encode rather than letting a ``UnicodeEncodeError`` escape (the Windows console
code page can't always encode the block glyphs). Glyphs auto-degrade to ASCII when
the stream's encoding can't represent them.

Time is read through an injected ``clock`` (default :func:`time.monotonic`) so
tests drive it deterministically without sleeping.

Owner: T20 (M2 integration track).
"""

from __future__ import annotations

import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, replace
from typing import Any, Callable, Deque, Dict, Optional, Tuple

__all__ = [
    "ProgressSnapshot",
    "ProgressReporter",
    "Glyphs",
    "UNICODE_GLYPHS",
    "ASCII_GLYPHS",
    "format_duration",
    "render_bar",
    "render_line",
    "progress_metrics",
]


# ---------------------------------------------------------------------------
# Glyph sets — the bar fill characters. Unicode where the terminal supports it,
# ASCII everywhere else (the persistent log LINE is always ASCII).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Glyphs:
    """Characters used to draw the bar (``fill``/``empty``) and the ε label."""

    fill: str = "#"
    empty: str = "-"
    eps: str = "eps"


#: Pretty glyphs for a UTF-8-capable terminal.
UNICODE_GLYPHS = Glyphs(fill="█", empty="░", eps="ε")  # █ ░ ε
#: Portable fallback for a console whose code page can't encode the block glyphs.
ASCII_GLYPHS = Glyphs()


# ---------------------------------------------------------------------------
# Duration formatting — compact, human, and ALWAYS defined (None -> "?").
# ---------------------------------------------------------------------------


def format_duration(seconds: Optional[float]) -> str:
    """Format a span of seconds compactly (``"4d 22h"``, ``"2h41m"``, ``"45s"``).

    Args:
        seconds: Non-negative span in seconds, or ``None`` for "unknown".

    Returns:
        A short human string. ``None`` -> ``"?"``; sub-second / negative -> ``"0s"``.
        Two units of precision at most (days+hours, hours+minutes, ...).
    """
    if seconds is None:
        return "?"
    total = int(seconds)
    if total <= 0:
        return "0s"
    days, rem = divmod(total, 86_400)
    hours, rem = divmod(rem, 3_600)
    minutes, secs = divmod(rem, 60)
    if days > 0:
        return f"{days}d {hours}h"
    if hours > 0:
        return f"{hours}h{minutes:02d}m"
    if minutes > 0:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


# ---------------------------------------------------------------------------
# The computed view handed to the renderers / metrics sink.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProgressSnapshot:
    """A point-in-time view of training progress (everything the bar/line need).

    Attributes:
        episodes_run: Episodes collected so far (1-based count).
        total_episodes: The episode budget (``max_episodes``) — the ETA target.
        steps_collected: Total environment transitions collected across episodes.
        grad_step: Completed gradient steps at this point.
        epsilon: ε of the most recently collected episode.
        last_win_rate: Win rate of the most recent greedy eval (``None`` if no
            eval has run yet).
        elapsed_s: Wall seconds since :meth:`ProgressReporter.start`.
        eps_per_min: Episodes/minute over the recent sliding window.
        steps_per_s: Transitions/second over the recent sliding window.
        eta_s: Estimated seconds to reach ``total_episodes`` at the windowed rate;
            ``0.0`` when already at the budget, ``None`` when the rate is 0 (so the
            ETA is genuinely unknown rather than "infinite").
        frac: ``episodes_run / total_episodes`` clamped to ``[0, 1]``.
    """

    episodes_run: int
    total_episodes: int
    steps_collected: int
    grad_step: int
    epsilon: float
    last_win_rate: Optional[float]
    elapsed_s: float
    eps_per_min: float
    steps_per_s: float
    eta_s: Optional[float]
    frac: float


def progress_metrics(snap: ProgressSnapshot) -> Dict[str, Any]:
    """Flatten a snapshot into ``progress/*`` metric keys for the MetricsLogger.

    ``eta_s`` may be ``None`` (unknown rate); the logger coerces that to JSON
    ``null``, so a reader can distinguish "unknown" from a real number.
    """
    return {
        "progress/episodes": snap.episodes_run,
        "progress/total_episodes": snap.total_episodes,
        "progress/steps": snap.steps_collected,
        "progress/frac": snap.frac,
        "progress/eps_per_min": snap.eps_per_min,
        "progress/steps_per_s": snap.steps_per_s,
        "progress/elapsed_s": snap.elapsed_s,
        "progress/eta_budget_s": snap.eta_s,
        "progress/epsilon": snap.epsilon,
    }


# ---------------------------------------------------------------------------
# Renderers — pure string builders (no I/O), so they unit-test trivially.
# ---------------------------------------------------------------------------


def render_bar(
    snap: ProgressSnapshot,
    width: int,
    *,
    glyphs: Glyphs = ASCII_GLYPHS,
    label: str = "m2",
) -> str:
    """Build the one-line, in-place status BAR (no leading ``\\r`` / trailing NL).

    ``[m2] ep 234/10000 [####------] 2.3%  0.63 ep/min  ETA 4d 22h  eps=0.81 ...``
    """
    width = max(1, int(width))
    filled = max(0, min(width, int(round(snap.frac * width))))
    bar = glyphs.fill * filled + glyphs.empty * (width - filled)
    parts = [
        f"[{label}]",
        f"ep {snap.episodes_run}/{snap.total_episodes}",
        f"[{bar}]",
        f"{snap.frac * 100:.1f}%",
        f"{snap.eps_per_min:.2f} ep/min",
        f"ETA {format_duration(snap.eta_s)}",
        f"{glyphs.eps}={snap.epsilon:.2f}",
    ]
    if snap.last_win_rate is not None:
        parts.append(f"win {snap.last_win_rate * 100:.0f}%")
    return "  ".join(parts)


def render_line(snap: ProgressSnapshot, *, label: str = "m2") -> str:
    """Build the persistent, ASCII-only progress LINE (one per ``log_interval``).

    This is what lands in a redirected log file / scrollback, so it carries more
    detail than the bar and never uses glyphs that a log viewer might mangle.
    """
    fields = [
        f"ep {snap.episodes_run}/{snap.total_episodes} ({snap.frac * 100:.1f}%)",
        f"{snap.eps_per_min:.2f} ep/min",
        f"{snap.steps_per_s:.2f} steps/s",
        f"{snap.steps_collected:,} steps",
        f"grad {snap.grad_step}",
        f"elapsed {format_duration(snap.elapsed_s)}",
        f"ETA(budget) {format_duration(snap.eta_s)}",
        f"eps={snap.epsilon:.3f}",
    ]
    if snap.last_win_rate is not None:
        fields.append(f"last_win={snap.last_win_rate:.3f}")
    return f"[{label} progress] " + " | ".join(fields)


# ---------------------------------------------------------------------------
# The reporter — owns the stream, the clock, throttling, and bar bookkeeping.
# ---------------------------------------------------------------------------


def _stream_supports(stream: Any, sample: str) -> bool:
    """True iff ``sample`` round-trips through ``stream``'s encoding."""
    encoding = getattr(stream, "encoding", None)
    if not encoding:
        return False
    try:
        sample.encode(encoding)
    except (LookupError, UnicodeError):
        return False
    return True


class ProgressReporter:
    """Live status bar + throttled progress log for the M2 training loop.

    Call :meth:`start` once before the episode loop, :meth:`update` after every
    collected episode, :meth:`message` to print a standalone line (e.g. an eval
    summary) without garbling the bar, :meth:`clear` to erase the bar before a
    long sub-task (an eval), and :meth:`close` at the end.

    On a TTY the bar is redrawn in place (carriage return) at most every
    ``redraw_interval`` s. Whether or not it's a TTY, a full progress LINE is
    emitted every ``log_interval`` s; :meth:`update` returns the
    :class:`ProgressSnapshot` exactly when it emitted one (so the caller can mirror
    it to the metrics logger), else ``None``.

    Args:
        total_episodes: The episode budget (must be > 0) — the ETA target.
        stream: Where to draw (defaults to ``sys.stderr``).
        clock: Zero-arg monotonic seconds source (defaults to ``time.monotonic``);
            injected in tests.
        redraw_interval: Min seconds between in-place bar redraws (TTY only).
        log_interval: Min seconds between persistent progress lines.
        bar_width: Width of the bar's fill region in characters.
        rate_window_s: Sliding window (seconds) over which throughput/ETA are
            measured, so the rate reflects the current pace.
        heartbeat_interval: When > 0 AND the bar is enabled (a TTY), a daemon
            thread redraws the bar every this-many seconds so the elapsed/ETA TICK
            between episodes (live episodes are ~90 s, so a once-per-episode bar
            would otherwise look frozen). ``0`` (default) disables the thread — the
            bar then redraws only on :meth:`update`. Off automatically when not a
            TTY (a redirected log gets the periodic LINE instead).
        label: Short tag shown in the bar/line (default ``"m2"``).
        enabled: Force the in-place bar on/off. ``None`` (default) auto-detects
            from ``stream.isatty()`` — off when output is redirected to a file.
    """

    def __init__(
        self,
        total_episodes: int,
        *,
        stream: Optional[Any] = None,
        clock: Optional[Callable[[], float]] = None,
        redraw_interval: float = 0.5,
        log_interval: float = 30.0,
        bar_width: int = 24,
        rate_window_s: float = 180.0,
        heartbeat_interval: float = 0.0,
        label: str = "m2",
        enabled: Optional[bool] = None,
    ) -> None:
        if total_episodes <= 0:
            raise ValueError(f"total_episodes must be > 0, got {total_episodes}")
        self.total_episodes = int(total_episodes)
        self._stream = stream if stream is not None else sys.stderr
        self._clock = clock if clock is not None else time.monotonic
        self.redraw_interval = float(redraw_interval)
        self.log_interval = float(log_interval)
        self.bar_width = int(bar_width)
        self.rate_window_s = float(rate_window_s)
        self.heartbeat_interval = float(heartbeat_interval)
        self.label = str(label)

        if enabled is None:
            isatty = getattr(self._stream, "isatty", None)
            try:
                enabled = bool(isatty()) if callable(isatty) else False
            except Exception:
                enabled = False
        self.enabled = bool(enabled)

        # Pretty glyphs only when the stream can actually encode them.
        self._glyphs = (
            UNICODE_GLYPHS
            if _stream_supports(self._stream, UNICODE_GLYPHS.fill + UNICODE_GLYPHS.eps)
            else ASCII_GLYPHS
        )

        self._t0: Optional[float] = None
        self._last_redraw = 0.0
        self._last_log = 0.0
        self._last_update_t = 0.0  # wall time of the most recent update() (heartbeat)
        # (wall_time, episodes_run, steps_collected) samples for the rate window.
        self._samples: Deque[Tuple[float, int, int]] = deque()
        self._bar_len = 0  # printed length of the current bar (for in-place clear)
        self._last_snapshot: Optional[ProgressSnapshot] = None
        # The bar is "suppressed" between clear() (before a long eval) and the next
        # update(), so the heartbeat thread doesn't redraw over the eval's output.
        self._suppressed = False

        # All stream writes + bar bookkeeping happen under this lock so the
        # heartbeat thread and the training thread never interleave a half-drawn
        # line. RLock: update() draws AND writes while already holding it.
        self._lock = threading.RLock()
        self._hb_thread: Optional[threading.Thread] = None
        self._hb_stop = threading.Event()

    # -- lifecycle --------------------------------------------------------

    def start(self, *, epsilon: float = 0.0) -> "ProgressReporter":
        """Mark the loop start (the elapsed/ETA origin) and show INSTANT feedback.

        Draws an ``ep 0/N`` bar right away (on a TTY) so the run visibly comes up
        the moment it launches — before the first ~90 s episode finishes — and
        starts the heartbeat thread so the bar ticks while that episode runs. Arms
        the first :meth:`update` to emit a persistent progress line.

        Args:
            epsilon: ε of the first episode, shown in the initial bar (cosmetic).
        """
        with self._lock:
            now = self._clock()
            self._t0 = now
            self._last_update_t = now
            self._last_redraw = now
            # Arm the first update() to emit a persistent line (and return a
            # snapshot the caller logs) regardless of the throttle interval.
            self._last_log = now - self.log_interval
            self._samples.clear()
            self._samples.append((now, 0, 0))
            self._suppressed = False
            # An initial ep-0 snapshot so the heartbeat has something to redraw and
            # the bar appears immediately (rate 0 / ETA "?" reads as "warming up").
            self._last_snapshot = self._snapshot(now, 0, 0, 0, epsilon, None)
            if self.enabled:
                self._draw_bar(self._last_snapshot)
        self._start_heartbeat()
        return self

    def update(
        self,
        *,
        episodes_run: int,
        steps_collected: int,
        grad_step: int,
        epsilon: float,
        last_win_rate: Optional[float] = None,
    ) -> Optional[ProgressSnapshot]:
        """Record progress; redraw the bar and maybe emit a line.

        Returns the :class:`ProgressSnapshot` iff a persistent progress line was
        emitted this call (i.e. ``log_interval`` elapsed) — the caller logs that to
        the metrics sink — otherwise ``None``.
        """
        if self._t0 is None:
            self.start(epsilon=epsilon)
        with self._lock:
            now = self._clock()
            self._record_sample(now, episodes_run, steps_collected)
            snap = self._snapshot(
                now, episodes_run, steps_collected, grad_step, epsilon, last_win_rate
            )
            self._last_snapshot = snap
            self._last_update_t = now
            # New data un-suppresses the bar (e.g. after an eval cleared it).
            self._suppressed = False

            if self.enabled and (now - self._last_redraw) >= self.redraw_interval:
                self._draw_bar(snap)
                self._last_redraw = now

            emitted: Optional[ProgressSnapshot] = None
            if (now - self._last_log) >= self.log_interval:
                self._write_line(render_line(snap, label=self.label))
                self._last_log = now
                emitted = snap
            return emitted

    def message(self, text: str) -> None:
        """Print a standalone line (e.g. an eval summary), clearing the bar first."""
        with self._lock:
            self._write_line(text)

    def clear(self) -> None:
        """Erase the in-place bar (call before a long sub-task such as an eval).

        Also suppresses the heartbeat redraw until the next :meth:`update`, so the
        bar can't reappear in the middle of the eval's own output.
        """
        with self._lock:
            self._suppressed = True
            self._erase_bar()
            self._flush()

    def close(self, *, emit_final: bool = True) -> Optional[ProgressSnapshot]:
        """Finalize: stop the heartbeat, drop below the bar, emit one last line.

        Returns the final snapshot (so the caller can log it) or ``None`` if no
        update ever ran.
        """
        # Stop the heartbeat FIRST, without the lock held, so the thread (which
        # grabs the lock each tick) can wake from its wait and exit without a
        # deadlock on join().
        self._stop_heartbeat()
        with self._lock:
            if self._t0 is None:
                return None
            # Keep the last bar visible by moving to a fresh line before the summary.
            if self.enabled and self._bar_len:
                self._safe_write("\n")
                self._bar_len = 0
                self._flush()
            snap = self._last_snapshot
            if emit_final and snap is not None:
                self._write_line(render_line(snap, label=self.label))
            return snap if emit_final else None

    # -- heartbeat thread (keeps the bar ticking between slow episodes) ----

    def _start_heartbeat(self) -> None:
        """Launch the daemon redraw thread (only on a TTY with a positive interval)."""
        if not self.enabled or self.heartbeat_interval <= 0.0:
            return
        if self._hb_thread is not None:
            return
        self._hb_stop.clear()
        thread = threading.Thread(
            target=self._heartbeat_loop, name="m2-progress-heartbeat", daemon=True
        )
        self._hb_thread = thread
        thread.start()

    def _stop_heartbeat(self) -> None:
        """Signal the heartbeat thread to exit and join it briefly (best-effort)."""
        self._hb_stop.set()
        thread = self._hb_thread
        self._hb_thread = None
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(0.1, self.heartbeat_interval) + 0.5)

    def _heartbeat_loop(self) -> None:
        """Redraw the bar every ``heartbeat_interval`` s with extrapolated time.

        Between episodes no new data arrives, so the bar would sit still. Here we
        re-render the last snapshot with the elapsed clock ADVANCED and the ETA
        wound DOWN by the same delta — so the bar visibly ticks and the user can
        see the run is alive and roughly how the ETA is moving. Never raises.
        """
        # ``Event.wait`` returns True only when stop is set, so the loop exits
        # promptly on close() instead of sleeping out a final interval.
        while not self._hb_stop.wait(self.heartbeat_interval):
            try:
                with self._lock:
                    snap = self._last_snapshot
                    if snap is None or self._suppressed or not self.enabled:
                        continue
                    self._draw_bar(self._extrapolate(snap, self._clock()))
            except Exception:
                # A redraw must never take the run down; drop this tick.
                pass

    def _extrapolate(self, snap: ProgressSnapshot, now: float) -> ProgressSnapshot:
        """Advance ``snap``'s elapsed clock and wind its ETA down to ``now``."""
        dt = max(0.0, now - self._last_update_t)
        eta = None if snap.eta_s is None else max(0.0, snap.eta_s - dt)
        return replace(snap, elapsed_s=snap.elapsed_s + dt, eta_s=eta)

    # -- internals --------------------------------------------------------

    def _record_sample(self, now: float, episodes_run: int, steps_collected: int) -> None:
        """Append a sample and evict any older than the rate window (keep >= 2)."""
        self._samples.append((now, episodes_run, steps_collected))
        cutoff = now - self.rate_window_s
        # Drop the oldest while the next-oldest is still beyond the window, so the
        # retained head is the last sample at/just before the cutoff.
        while len(self._samples) > 2 and self._samples[1][0] < cutoff:
            self._samples.popleft()

    def _snapshot(
        self,
        now: float,
        episodes_run: int,
        steps_collected: int,
        grad_step: int,
        epsilon: float,
        last_win_rate: Optional[float],
    ) -> ProgressSnapshot:
        elapsed = max(0.0, now - (self._t0 or now))
        t_old, ep_old, st_old = self._samples[0]
        wdt = now - t_old

        # Prefer the windowed rate; fall back to the whole-run average when the
        # window hasn't seen progress yet (e.g. the very first episode).
        eps_per_s = self._rate(episodes_run - ep_old, wdt)
        if eps_per_s <= 0.0:
            eps_per_s = self._rate(episodes_run, elapsed)
        steps_per_s = self._rate(steps_collected - st_old, wdt)
        if steps_per_s <= 0.0:
            steps_per_s = self._rate(steps_collected, elapsed)

        remaining = max(0, self.total_episodes - episodes_run)
        if remaining == 0:
            eta_s: Optional[float] = 0.0
        elif eps_per_s > 0.0:
            eta_s = remaining / eps_per_s
        else:
            eta_s = None  # unknown rate -> honest "?" rather than a bogus number

        frac = 0.0
        if self.total_episodes > 0:
            frac = min(1.0, max(0.0, episodes_run / self.total_episodes))

        return ProgressSnapshot(
            episodes_run=int(episodes_run),
            total_episodes=self.total_episodes,
            steps_collected=int(steps_collected),
            grad_step=int(grad_step),
            epsilon=float(epsilon),
            last_win_rate=None if last_win_rate is None else float(last_win_rate),
            elapsed_s=elapsed,
            eps_per_min=eps_per_s * 60.0,
            steps_per_s=steps_per_s,
            eta_s=eta_s,
            frac=frac,
        )

    @staticmethod
    def _rate(delta: float, dt: float) -> float:
        """A non-negative per-second rate, or 0 when undefined."""
        if dt <= 0.0 or delta <= 0.0:
            return 0.0
        return delta / dt

    def _draw_bar(self, snap: ProgressSnapshot) -> None:
        text = render_bar(snap, self.bar_width, glyphs=self._glyphs, label=self.label)
        # Pad with spaces to overwrite any leftover from a previously longer bar,
        # then a trailing CR-free write keeps the cursor parked on the bar line.
        pad = " " * max(0, self._bar_len - len(text))
        self._safe_write("\r" + text + pad)
        self._flush()
        self._bar_len = len(text)

    def _erase_bar(self) -> None:
        if self.enabled and self._bar_len:
            self._safe_write("\r" + " " * self._bar_len + "\r")
            self._bar_len = 0

    def _write_line(self, text: str) -> None:
        """Clear the bar (if any) and write ``text`` as its own line."""
        self._erase_bar()
        self._safe_write(text + "\n")
        self._flush()

    def _safe_write(self, text: str) -> None:
        """Write to the stream, NEVER raising — fall back to an ASCII re-encode.

        Monitoring output must not be able to kill a multi-hour run, so an encoding
        error (Windows console code page can't encode a glyph) degrades to a
        lossy ASCII write instead of propagating.
        """
        try:
            self._stream.write(text)
        except UnicodeError:
            encoding = getattr(self._stream, "encoding", None) or "ascii"
            try:
                self._stream.write(text.encode(encoding, "replace").decode(encoding, "replace"))
            except Exception:
                try:
                    self._stream.write(text.encode("ascii", "replace").decode("ascii"))
                except Exception:
                    pass  # last resort: drop this write rather than crash the run
        except Exception:
            pass  # a broken pipe / closed stream must not take the run down

    def _flush(self) -> None:
        flush = getattr(self._stream, "flush", None)
        if callable(flush):
            try:
                flush()
            except Exception:
                pass
