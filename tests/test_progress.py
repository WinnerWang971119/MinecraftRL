"""Unit tests for the live training progress reporter (agent.progress).

The reporter is pure-Python and stdlib-only, so these tests drive it with an
injected fake clock and an in-memory stream — no sleeping, no torch, no terminal.
They pin the three things the M2 loop relies on:

  * the throughput / ETA arithmetic (windowed rate, budget ETA, the honest
    ``None`` ETA when the rate is unknown);
  * the throttling contract (one persistent line per ``log_interval``, the bar
    redrawn at most once per ``redraw_interval`` on a TTY); and
  * reliability — a write that raises (encoding / broken pipe) must NEVER escape,
    because this wraps a multi-hour run.
"""

from __future__ import annotations

import io
import time

import pytest

from agent.progress import (
    ASCII_GLYPHS,
    UNICODE_GLYPHS,
    ProgressReporter,
    ProgressSnapshot,
    format_duration,
    progress_metrics,
    render_bar,
    render_line,
)


class FakeClock:
    """A deterministic monotonic clock the tests advance by hand."""

    def __init__(self, t: float = 0.0) -> None:
        self.t = float(t)

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += float(dt)


def _snap(**overrides) -> ProgressSnapshot:
    """A fully-populated snapshot with sensible defaults, for renderer tests."""
    base = dict(
        episodes_run=234,
        total_episodes=10_000,
        steps_collected=5_210,
        grad_step=234,
        epsilon=0.81,
        last_win_rate=None,
        elapsed_s=9_660.0,  # 2h41m
        eps_per_min=0.63,
        steps_per_s=0.42,
        eta_s=421_200.0,  # ~4d 21h
        frac=0.0234,
    )
    base.update(overrides)
    return ProgressSnapshot(**base)


# ---------------------------------------------------------------------------
# format_duration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "seconds,expected",
    [
        (None, "?"),
        (0, "0s"),
        (-5, "0s"),
        (0.4, "0s"),
        (45, "45s"),
        (125, "2m05s"),
        (3_600, "1h00m"),
        (9_660, "2h41m"),
        (90_000, "1d 1h"),
        (421_200, "4d 21h"),
    ],
)
def test_format_duration(seconds, expected):
    assert format_duration(seconds) == expected


# ---------------------------------------------------------------------------
# Renderers (pure strings)
# ---------------------------------------------------------------------------


def test_render_bar_has_fields_and_fixed_width():
    snap = _snap()
    bar = render_bar(snap, width=24, glyphs=ASCII_GLYPHS)
    assert "[m2]" in bar
    assert "ep 234/10000" in bar
    assert "2.3%" in bar
    assert "0.63 ep/min" in bar
    assert "ETA 4d 21h" in bar
    assert "eps=0.81" in bar
    # The fill region is exactly `width` cells (fill + empty), framed by brackets.
    fill_region = bar.split("[", 2)[2].split("]", 1)[0]
    assert len(fill_region) == 24
    assert set(fill_region) <= {ASCII_GLYPHS.fill, ASCII_GLYPHS.empty}


def test_render_bar_includes_win_rate_when_present():
    bar = render_bar(_snap(last_win_rate=0.62), width=10, glyphs=ASCII_GLYPHS)
    assert "win 62%" in bar


def test_render_bar_unicode_glyphs():
    bar = render_bar(_snap(frac=0.5), width=10, glyphs=UNICODE_GLYPHS)
    assert UNICODE_GLYPHS.fill in bar  # at half progress some cells are filled


def test_render_line_is_ascii_and_detailed():
    line = render_line(_snap(last_win_rate=0.62))
    assert line.startswith("[m2 progress]")
    assert "ep 234/10000 (2.3%)" in line
    assert "0.42 steps/s" in line
    assert "5,210 steps" in line
    assert "ETA(budget) 4d 21h" in line
    assert "last_win=0.620" in line
    # Persistent log lines stay ASCII-clean (no block glyphs).
    line.encode("ascii")


# ---------------------------------------------------------------------------
# progress_metrics flattening
# ---------------------------------------------------------------------------


def test_progress_metrics_keys_and_none_eta_passthrough():
    metrics = progress_metrics(_snap(eta_s=None))
    assert metrics["progress/episodes"] == 234
    assert metrics["progress/total_episodes"] == 10_000
    assert metrics["progress/eps_per_min"] == pytest.approx(0.63)
    # An unknown ETA stays None so a reader can tell it apart from a real number.
    assert metrics["progress/eta_budget_s"] is None


# ---------------------------------------------------------------------------
# Snapshot arithmetic via the reporter (rate / ETA / frac)
# ---------------------------------------------------------------------------


def test_windowed_rate_and_budget_eta():
    clock = FakeClock()
    rep = ProgressReporter(total_episodes=100, stream=io.StringIO(), clock=clock,
                           enabled=False)
    rep.start()
    clock.advance(10.0)
    snap = rep.update(episodes_run=1, steps_collected=10, grad_step=1, epsilon=0.9)
    # 1 episode in 10 s -> 0.1 ep/s -> 6 ep/min; 99 left -> 990 s ETA.
    assert snap is not None  # forced first emit
    assert snap.eps_per_min == pytest.approx(6.0)
    assert snap.steps_per_s == pytest.approx(1.0)
    assert snap.eta_s == pytest.approx(990.0)
    assert snap.frac == pytest.approx(0.01)


def test_eta_is_none_when_rate_unknown():
    clock = FakeClock()
    rep = ProgressReporter(total_episodes=100, stream=io.StringIO(), clock=clock,
                           enabled=False)
    rep.start()
    clock.advance(5.0)
    # No episode progressed -> rate 0 -> ETA genuinely unknown (None, not inf).
    snap = rep.update(episodes_run=0, steps_collected=0, grad_step=0, epsilon=1.0)
    assert snap.eta_s is None
    assert format_duration(snap.eta_s) == "?"


def test_eta_zero_at_budget():
    clock = FakeClock()
    rep = ProgressReporter(total_episodes=5, stream=io.StringIO(), clock=clock,
                           enabled=False)
    rep.start()
    clock.advance(50.0)
    snap = rep.update(episodes_run=5, steps_collected=50, grad_step=5, epsilon=0.5)
    assert snap.frac == pytest.approx(1.0)
    assert snap.eta_s == pytest.approx(0.0)


def test_rate_window_evicts_old_samples():
    clock = FakeClock()
    rep = ProgressReporter(total_episodes=100, stream=io.StringIO(), clock=clock,
                           rate_window_s=180.0, enabled=False)
    rep.start()
    clock.advance(10.0)
    rep.update(episodes_run=1, steps_collected=10, grad_step=1, epsilon=0.9)
    # Jump past the window: the (t=0) origin sample is evicted, so the rate is
    # measured from the t=10 head (1 episode over the last 190 s), not from t=0.
    clock.advance(190.0)
    snap = rep.update(episodes_run=2, steps_collected=20, grad_step=2, epsilon=0.8)
    assert snap.eps_per_min == pytest.approx(60.0 / 190.0, rel=1e-6)


# ---------------------------------------------------------------------------
# Throttling: one line per log_interval; bar at most once per redraw_interval.
# ---------------------------------------------------------------------------


def test_periodic_line_is_throttled_to_log_interval():
    clock = FakeClock()
    out = io.StringIO()
    rep = ProgressReporter(total_episodes=100, stream=out, clock=clock,
                           log_interval=30.0, enabled=False)
    rep.start()
    # First update fires immediately (start() arms it) ...
    clock.advance(1.0)
    assert rep.update(episodes_run=1, steps_collected=5, grad_step=1, epsilon=0.9)
    # ... the next one inside the interval does NOT emit ...
    clock.advance(5.0)
    assert rep.update(episodes_run=2, steps_collected=10, grad_step=2, epsilon=0.9) is None
    # ... and once the interval elapses, it emits again.
    clock.advance(30.0)
    assert rep.update(episodes_run=3, steps_collected=15, grad_step=3, epsilon=0.9)
    # Exactly two persistent lines were written.
    assert out.getvalue().count("[m2 progress]") == 2


def test_disabled_bar_writes_lines_but_no_carriage_return():
    clock = FakeClock()
    out = io.StringIO()
    rep = ProgressReporter(total_episodes=100, stream=out, clock=clock,
                           enabled=False)
    rep.start()
    clock.advance(1.0)
    rep.update(episodes_run=1, steps_collected=5, grad_step=1, epsilon=0.9)
    text = out.getvalue()
    assert "[m2 progress]" in text
    assert "\r" not in text  # no in-place bar when output is not a TTY


def test_enabled_bar_uses_carriage_return_and_message_clears_it():
    clock = FakeClock()
    out = io.StringIO()
    rep = ProgressReporter(total_episodes=100, stream=out, clock=clock,
                           redraw_interval=0.5, log_interval=1e9, enabled=True)
    rep.start(epsilon=0.9)
    # start() draws an instant ep-0 bar so the run visibly comes up at launch.
    assert "ep 0/100" in out.getvalue()
    clock.advance(0.5)  # past redraw_interval so the new data redraws
    rep.update(episodes_run=1, steps_collected=5, grad_step=1, epsilon=0.9)
    drawn = out.getvalue()
    assert "\r" in drawn  # the in-place bar redraws with a carriage return
    assert "ep 1/100" in drawn
    # A standalone message clears the bar (CR + blanking) before its own line.
    rep.message("[m2 ep 1] eval summary")
    assert "[m2 ep 1] eval summary\n" in out.getvalue()


def test_start_draws_instant_bar_on_tty_before_any_episode():
    out = io.StringIO()
    rep = ProgressReporter(total_episodes=10, stream=out, clock=FakeClock(),
                           enabled=True)
    rep.start(epsilon=0.99)
    text = out.getvalue()
    assert "ep 0/10" in text  # immediate feedback, no episode completed yet
    assert "ETA ?" in text  # rate unknown at launch -> honest "?", not a number


# ---------------------------------------------------------------------------
# Heartbeat thread — keeps the bar ticking between slow episodes.
# ---------------------------------------------------------------------------


def test_extrapolate_advances_elapsed_and_winds_eta_down():
    rep = ProgressReporter(total_episodes=100, stream=io.StringIO(), clock=FakeClock(),
                           enabled=False)
    rep._last_update_t = 100.0
    ex = rep._extrapolate(_snap(elapsed_s=10.0, eta_s=50.0), now=105.0)
    assert ex.elapsed_s == pytest.approx(15.0)  # +5 s of wall time
    assert ex.eta_s == pytest.approx(45.0)  # ETA wound down by the same 5 s
    # An unknown ETA stays unknown.
    assert rep._extrapolate(_snap(eta_s=None), now=105.0).eta_s is None
    # ETA never goes negative.
    assert rep._extrapolate(_snap(eta_s=2.0), now=200.0).eta_s == pytest.approx(0.0)


def test_heartbeat_not_started_when_not_a_tty():
    rep = ProgressReporter(total_episodes=10, stream=io.StringIO(), clock=FakeClock(),
                           heartbeat_interval=1.0, enabled=False)
    rep.start()
    assert rep._hb_thread is None  # no live bar to tick when output is redirected
    rep.close()


def test_heartbeat_thread_starts_ticks_and_stops():
    out = io.StringIO()
    # Real monotonic clock so the heartbeat extrapolates real elapsed time.
    rep = ProgressReporter(total_episodes=100, stream=out, heartbeat_interval=0.01,
                           enabled=True)
    rep.start(epsilon=0.9)
    assert rep._hb_thread is not None and rep._hb_thread.is_alive()
    time.sleep(0.05)  # let the daemon tick the bar a few times
    rep.update(episodes_run=1, steps_collected=10, grad_step=1, epsilon=0.9)
    rep.close()
    assert rep._hb_thread is None  # close() joined/cleared the thread
    assert "\r" in out.getvalue()  # the bar was actually drawn


def test_close_emits_final_line_and_returns_snapshot():
    clock = FakeClock()
    out = io.StringIO()
    rep = ProgressReporter(total_episodes=100, stream=out, clock=clock,
                           log_interval=1e9, enabled=False)
    rep.start()
    clock.advance(20.0)
    rep.update(episodes_run=4, steps_collected=40, grad_step=4, epsilon=0.7)
    final = rep.close()
    assert final is not None
    assert final.episodes_run == 4
    assert out.getvalue().strip().endswith("eps=0.700")


def test_close_without_any_update_is_a_noop():
    rep = ProgressReporter(total_episodes=10, stream=io.StringIO(), clock=FakeClock())
    assert rep.close() is None  # never started/updated


# ---------------------------------------------------------------------------
# Reliability + validation.
# ---------------------------------------------------------------------------


class _BrokenStream:
    """A stream whose write always raises — the reporter must swallow it."""

    encoding = "utf-8"

    def write(self, _text):
        raise UnicodeEncodeError("utf-8", "x", 0, 1, "boom")

    def flush(self):
        raise OSError("broken pipe")

    def isatty(self):
        return True


def test_write_failures_never_propagate():
    clock = FakeClock()
    rep = ProgressReporter(total_episodes=10, stream=_BrokenStream(), clock=clock,
                           enabled=True)
    rep.start()
    clock.advance(1.0)
    # None of these may raise even though every underlying write/flush throws.
    rep.update(episodes_run=1, steps_collected=1, grad_step=1, epsilon=0.9)
    rep.message("hello")
    rep.clear()
    rep.close()


def test_ascii_glyphs_when_stream_cannot_encode_unicode():
    # A StringIO has no usable `.encoding`, so the reporter must pick ASCII glyphs.
    rep = ProgressReporter(total_episodes=10, stream=io.StringIO(), clock=FakeClock(),
                           enabled=True)
    assert rep._glyphs is ASCII_GLYPHS


def test_rejects_non_positive_budget():
    with pytest.raises(ValueError, match="total_episodes must be > 0"):
        ProgressReporter(total_episodes=0)
