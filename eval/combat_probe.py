"""combat_probe — T8 deterministic combat gate (AC8, the go/no-go gate).

Drives ``--cycles`` reset/kill cycles against the LIVE stack (Paper -> bridge ->
this driver): face the dummy from the spawn posture, issue an ATTACK whenever
the wire reports the swing fully cooled (``state.self.attack_cooldown == 1.0``),
IDLE otherwise, and record every decision window. Per cycle it asserts:

  * the recorded per-hit ``events.damage_dealt`` sequence is exactly
    ``6, 6, 6, 2`` (iron sword vs a 20 HP dummy, regeneration off);
  * cumulative dealt damage is exactly 20;
  * ``events.opponent_died`` fires in exactly one window and the episode ends
    as a win;
  * the episode starts from a clean baseline (wire opponent health == 20 at the
    first post-reset state — which also proves the first post-respawn hit of
    the NEXT cycle measures from a clean baseline);
  * every recorded damage value RECONCILES against the wire's privileged
    ``state.opponent.health`` at +/-1 window (the dummy's ``update_health``
    arrives on a second connection, so a one-window skew is legal);
  * no unexplained wire-health drop (an unrecorded hit) and no wire-health
    increase outside the death/respawn window (regeneration is off, so any
    heal is a defect);
  * TC16: cumulative dealt damage > 20 in any episode is a defect, not noise.

THE WIRE HEALTH IS THE ORACLE, NOT THE PRODUCTION PATH. ``state.opponent.health``
is used here strictly as the free, independent cross-check of the repaired
``events`` channel (see the plan's Decisions). Deriving ``damage_dealt`` from it
in production was considered and rejected — this probe must never be read as a
template for that.

SINGLE-CONNECTION DISCIPLINE. ``BridgeServer`` accepts exactly ONE TCP client
and a second connection silently destroys the first. The probe therefore opens
ONE :class:`~env.mc_pvp_env.TcpBridgeClient` (wrapped in a recording shim so the
raw ``state`` messages stay inspectable), constructs ONE
:class:`~env.mc_pvp_env.MCPvPEnv` over it with ``auto_connect=False``, and runs
every cycle through that single env/connection — the same borrow pattern as
``agent.train._eval_against_dummy``.

KNOWN FALSE-FAIL MODE (D2, issue #28) — read this before rationalizing a red
run. The wire's ``attack_cooldown`` reads 1.0 at episode start (the bridge's
swing tracker is cleared by the reset), but the reset's ``/clear`` + ``/give``
re-equip resets the SERVER-side attack meter, which the bridge does not model.
On the FIRST cycle after a fresh bridge boot the resulting w0 swing can land a
weak partial-cooldown hit (observed live: ``1.269, 6, 6, 6, 0.731``), failing
the 6,6,6,2 sequence assertion ON A CORRECT DAMAGE CHANNEL (1 of 3 fresh-boot
runs observed). The fingerprint: cycle 0 only, every value still reconciles
window-for-window against the wire, and the total is still exactly 20. That is
D2 (a bridge cooldown observable defect), NOT a damage-channel fault — re-run
before concluding anything, and do NOT weaken the probe to tolerate it: once D2
is fixed bridge-side, w0 legitimately reads not-ready, the probe IDLEs there,
and the arithmetic holds for an understood reason. In steady state (every cycle
after the first) the sequence was exact across 48/48 live cycles.

Usage (Paper and the bridge already running, in that order):

    python -m eval.combat_probe --cycles 10
    python -m eval.combat_probe --cycles 5 --expect-anchor 512,0   # non-zero pad

Exit codes: 0 = AC8 PASS across all cycles, 1 = FAIL (any assertion, any
non-win outcome), 2 = aborted on a :class:`BridgeError` — either the probe
could not connect at all, or a mid-episode transport abort ended the run early
(step() never silently retries; a lost in-flight reply is an unrecoverable
desync). A 2 therefore means "no verdict", never "pass".

Owner: T8 (Eval/infra track)
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from agent.actions import Macro
from bridge.messages import ResetAckMsg, StateMsg
from env.mc_pvp_env import BridgeError, MCPvPEnv, TcpBridgeClient

__all__ = [
    "RecordingTransport",
    "StepRecord",
    "CycleRecord",
    "extract_hits",
    "reconcile_against_wire",
    "analyze_cycle",
    "run_probe",
    "main",
    "EXPECTED_HITS",
    "EXPECTED_TOTAL",
    "FULL_HEALTH",
]

# ---------------------------------------------------------------------------
# Expected combat arithmetic (AC8).
#
# The dummy has 20 HP; an iron sword deals exactly 6.0 per fully-cooled swing
# against an unarmored player; the fourth hit is capped by remaining health:
# 6 + 6 + 6 + 2 == 20. A hit inside the cooldown deals less, so the driver only
# ever swings at attack_cooldown == 1.0.
# ---------------------------------------------------------------------------

EXPECTED_HITS: Tuple[float, ...] = (6.0, 6.0, 6.0, 2.0)
EXPECTED_TOTAL: float = 20.0
FULL_HEALTH: float = 20.0

#: Float comparison tolerance for damage amounts / health values. Damage in
#: vanilla is exact multiples of 0.5 for this matchup, so 1e-6 is generous.
_TOL: float = 1e-6

#: Position tolerance for the spawn-anchor assertion — the same posEpsilon the
#: bridge's own read-back gate uses (bridge/bot.js DEFAULT_READBACK).
_POS_EPS: float = 0.25

#: Swing readiness threshold on the wire's [0,1] attack_cooldown.
_COOLDOWN_READY: float = 1.0 - _TOL


# ---------------------------------------------------------------------------
# Recording transport shim.
# ---------------------------------------------------------------------------


class RecordingTransport:
    """Wraps a :class:`BridgeTransport` and records the raw inbound messages.

    The env consumes parsed ``StateMsg``/``ResetAckMsg`` dataclasses through the
    transport seam but only exposes gated observations; the probe needs the RAW
    wire state (privileged ``opponent.health``, ``self.attack_cooldown``,
    positions). This shim is transparent to the env and keeps the last seen
    message of each type inspectable.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.last_state: Optional[StateMsg] = None
        self.last_reset_ack: Optional[ResetAckMsg] = None

    def connect(self) -> None:
        self._inner.connect()

    def send(self, obj: Mapping[str, Any]) -> None:
        self._inner.send(obj)

    def recv(self) -> Union[StateMsg, ResetAckMsg]:
        msg = self._inner.recv()
        if isinstance(msg, StateMsg):
            self.last_state = msg
        elif isinstance(msg, ResetAckMsg):
            self.last_reset_ack = msg
        return msg

    def close(self) -> None:
        self._inner.close()


# ---------------------------------------------------------------------------
# Per-cycle records (plain data so the analysis below stays pure/unit-testable).
# ---------------------------------------------------------------------------


@dataclass
class StepRecord:
    """One decision window as the probe observed it.

    Attributes:
        action: The action index the probe issued for this window.
        damage_dealt: ``events.damage_dealt`` reported for this window.
        opponent_died: ``events.opponent_died`` reported for this window.
        wire_health: Raw ``state.opponent.health`` at the END of this window.
        attack_cooldown: Raw ``state.self.attack_cooldown`` at window end.
        tick: ``state.tick`` at window end.
    """

    action: int
    damage_dealt: float
    opponent_died: bool
    wire_health: float
    attack_cooldown: float
    tick: int


@dataclass
class CycleRecord:
    """Everything recorded for one reset/kill cycle."""

    index: int
    reset_ms: float
    start_health: float
    start_self_pos: Tuple[float, float, float]
    start_opp_pos: Tuple[float, float, float]
    outcome: str  # "win" | "loss" | "timeout"
    steps: List[StepRecord] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pure analysis (unit-tested in tests/test_combat_probe.py).
# ---------------------------------------------------------------------------


def extract_hits(steps: Sequence[StepRecord]) -> List[Tuple[int, float]]:
    """Return ``(window_index, amount)`` for every window with nonzero damage."""
    return [
        (i, float(s.damage_dealt)) for i, s in enumerate(steps) if s.damage_dealt > _TOL
    ]


def reconcile_against_wire(
    start_health: float, steps: Sequence[StepRecord]
) -> List[str]:
    """Cross-check recorded damage events against the wire's opponent health.

    Rules (all violations are returned as human-readable error strings):

      * Every recorded hit must match a wire-health DROP of the same amount in
        its own window or an adjacent one (+/-1 — the dummy's ``update_health``
        arrives on a second connection). Each drop can satisfy only one hit.
      * The killing blow may be invisible as a drop: with ``doImmediateRespawn``
        the window-end snapshot can already read the respawned 20, so a hit
        whose windows +/-1 contain ``opponent_died`` reconciles instead against
        the health ENTERING its window (the blow equals remaining health).
      * A wire drop that no recorded hit explains is an UNRECORDED hit — the
        exact under-counting failure the repair must not have.
      * A wire-health INCREASE outside the death/respawn neighbourhood is a
        heal; with regeneration off it is a defect.
    """
    errors: List[str] = []
    n = len(steps)
    # health_entering[w] is the wire health at the START of window w;
    # steps[w].wire_health is the health at its end.
    health_entering = [float(start_health)] + [float(s.wire_health) for s in steps[:-1]]

    drops: Dict[int, float] = {}
    increases: Dict[int, float] = {}
    for w in range(n):
        delta = health_entering[w] - float(steps[w].wire_health)
        if delta > _TOL:
            drops[w] = delta
        elif delta < -_TOL:
            increases[w] = -delta

    death_windows = {w for w in range(n) if steps[w].opponent_died}

    def _near_death(w: int) -> bool:
        return any(u in death_windows for u in (w - 1, w, w + 1))

    hits = extract_hits(steps)
    used_drops: set = set()
    for w, amount in hits:
        matched = False
        for u in (w, w - 1, w + 1):  # exact window first, then the skew cases
            if u in drops and u not in used_drops and abs(drops[u] - amount) <= _TOL:
                used_drops.add(u)
                matched = True
                break
        if matched:
            continue
        # Death path: the fatal drop can be masked by the immediate respawn.
        if _near_death(w) and abs(health_entering[w] - amount) <= _TOL:
            continue
        errors.append(
            f"recorded hit of {amount:g} in window {w} has no matching wire-health "
            f"drop within +/-1 window (wire drops: "
            f"{ {k: round(v, 3) for k, v in sorted(drops.items())} })"
        )

    for u in sorted(drops):
        if u not in used_drops:
            errors.append(
                f"wire health dropped {drops[u]:g} in window {u} with no recorded "
                f"damage event to explain it (unrecorded hit)"
            )

    for u in sorted(increases):
        if not _near_death(u):
            errors.append(
                f"wire health INCREASED {increases[u]:g} in window {u} outside the "
                f"death/respawn neighbourhood — a heal with regeneration off is a "
                f"defect"
            )

    return errors


def analyze_cycle(
    record: CycleRecord, expected_hits: Sequence[float] = EXPECTED_HITS
) -> List[str]:
    """All AC8 assertions for one cycle. Returns a list of failures (empty == pass)."""
    errors: List[str] = []

    if record.outcome != "win":
        errors.append(f"episode ended as {record.outcome!r}, expected a win")

    if abs(record.start_health - FULL_HEALTH) > _TOL:
        errors.append(
            f"episode started from wire opponent health {record.start_health:g}, "
            f"expected a clean {FULL_HEALTH:g} baseline"
        )

    hits = extract_hits(record.steps)
    amounts = [amount for _, amount in hits]
    expected = [float(v) for v in expected_hits]
    if len(amounts) != len(expected) or any(
        abs(a - e) > _TOL for a, e in zip(amounts, expected)
    ):
        errors.append(
            f"per-hit sequence {[round(a, 3) for a in amounts]} != expected "
            f"{[round(e, 3) for e in expected]}"
        )

    total = sum(s.damage_dealt for s in record.steps)
    if abs(total - EXPECTED_TOTAL) > _TOL:
        errors.append(f"cumulative dealt damage {total:g} != {EXPECTED_TOTAL:g}")
    if total > EXPECTED_TOTAL + _TOL:
        errors.append(
            f"TC16: cumulative dealt damage {total:g} > {EXPECTED_TOTAL:g} with "
            f"regeneration off is a defect, not noise"
        )

    deaths = sum(1 for s in record.steps if s.opponent_died)
    if deaths != 1:
        errors.append(f"opponent_died fired in {deaths} windows, expected exactly 1")

    errors.extend(reconcile_against_wire(record.start_health, record.steps))
    return errors


def check_anchor(
    record: CycleRecord, anchor: Tuple[int, int], eps: float = _POS_EPS
) -> List[str]:
    """Assert both bots start on THIS pad's anchor (learner +0.5, dummy +3.5)."""
    ax, az = anchor
    errors: List[str] = []
    expectations = (
        ("learner", record.start_self_pos, (ax + 0.5, 64.0, az + 0.5)),
        ("dummy", record.start_opp_pos, (ax + 3.5, 64.0, az + 0.5)),
    )
    for name, actual, expected in expectations:
        if any(abs(a - e) > eps for a, e in zip(actual, expected)):
            errors.append(
                f"{name} start pos {tuple(round(v, 3) for v in actual)} is not at "
                f"the pad anchor expectation {expected} (eps {eps})"
            )
    return errors


# ---------------------------------------------------------------------------
# The live driver.
# ---------------------------------------------------------------------------


def _classify(info: Mapping[str, Any]) -> str:
    if info.get("lost"):
        return "loss"
    if info.get("won"):
        return "win"
    return "timeout"


def _run_cycle(
    env: MCPvPEnv,
    transport: RecordingTransport,
    cycle: int,
    seed: int,
    max_steps: int,
) -> CycleRecord:
    """Run one reset/kill cycle and record every window from the raw wire."""
    t0 = time.monotonic()
    env.reset(seed=seed)
    reset_ms = (time.monotonic() - t0) * 1000.0

    initial = transport.last_state
    if initial is None:
        raise BridgeError("no post-reset state message was recorded")

    record = CycleRecord(
        index=cycle,
        reset_ms=reset_ms,
        start_health=float(initial.opponent.health),
        start_self_pos=tuple(float(v) for v in initial.self_state.pos),
        start_opp_pos=tuple(float(v) for v in initial.opponent.pos),
        outcome="timeout",
    )

    cooldown = float(initial.self_state.attack_cooldown)
    info: Dict[str, Any] = {}
    for _ in range(max_steps):
        # Fully-cooled swings ONLY: a swing inside the cooldown deals reduced
        # damage and would break the 6,6,6,2 arithmetic by construction.
        action = Macro.ATTACK if cooldown >= _COOLDOWN_READY else Macro.IDLE
        _, _, done, info = env.step(int(action))

        state = transport.last_state
        if state is None:  # pragma: no cover - recv() always records on success
            raise BridgeError("step returned but no state message was recorded")
        events = info["events"]
        record.steps.append(
            StepRecord(
                action=int(action),
                damage_dealt=float(events["damage_dealt"]),
                opponent_died=bool(events["opponent_died"]),
                wire_health=float(state.opponent.health),
                attack_cooldown=float(state.self_state.attack_cooldown),
                tick=int(state.tick),
            )
        )
        cooldown = float(state.self_state.attack_cooldown)
        if done:
            record.outcome = _classify(info)
            break

    return record


def _format_cycle_line(record: CycleRecord, errors: Sequence[str]) -> str:
    hits = extract_hits(record.steps)
    hit_text = ", ".join(f"{amount:g}@w{w}" for w, amount in hits) or "none"
    total = sum(s.damage_dealt for s in record.steps)
    deaths = sum(1 for s in record.steps if s.opponent_died)
    return (
        f"[cycle {record.index:>2}] reset={record.reset_ms:6.0f}ms "
        f"start_hp={record.start_health:g} hits=[{hit_text}] total={total:g} "
        f"deaths={deaths} steps={len(record.steps)} outcome={record.outcome} "
        f"{'OK' if not errors else 'FAIL'}"
    )


def run_probe(
    *,
    host: str,
    port: int,
    cycles: int,
    seed: int,
    max_steps: int,
    anchor: Tuple[int, int],
    log=print,
) -> bool:
    """Run the full probe. Returns True iff every cycle passed (AC8 PASS)."""
    if cycles < 1:
        raise ValueError(f"cycles must be >= 1, got {cycles}")

    transport = RecordingTransport(TcpBridgeClient(host=host, port=port))
    # ONE connection for the whole probe (the bridge serves exactly one client);
    # connect explicitly, then hand the transport to the env with
    # auto_connect=False so the ownership is unambiguous.
    transport.connect()

    all_ok = True
    reset_times: List[float] = []
    env = MCPvPEnv(transport=transport, auto_connect=False, max_episode_steps=max_steps)
    try:
        for cycle in range(cycles):
            record = _run_cycle(env, transport, cycle, seed + cycle, max_steps)
            errors = analyze_cycle(record) + check_anchor(record, anchor)
            reset_times.append(record.reset_ms)
            log(_format_cycle_line(record, errors))
            if errors:
                all_ok = False
                for err in errors:
                    log(f"    FAIL: {err}")
            # Full per-window trace so the evidence is verbatim, not summarized.
            log(
                "    windows: "
                + " ".join(
                    f"(w{i} a={s.action} d={s.damage_dealt:g} hp={s.wire_health:g}"
                    f"{' DIED' if s.opponent_died else ''})"
                    for i, s in enumerate(record.steps)
                )
            )
    finally:
        env.close()

    if reset_times:
        log(
            f"[resets] n={len(reset_times)} min={min(reset_times):.0f}ms "
            f"median={statistics.median(reset_times):.0f}ms "
            f"max={max(reset_times):.0f}ms"
        )
    log(f"[combat_probe] AC8 {'PASS' if all_ok else 'FAIL'} over {cycles} cycle(s)")
    return all_ok


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def _parse_anchor(raw: str) -> Tuple[int, int]:
    # Negative anchors are rejected DELIBERATELY: T10's padAnchor(i) only ever
    # produces non-negative anchors, and the datapack's textual "$(x).5" macro
    # concatenation is unsafe for negatives (it would yield anchor MINUS half a
    # block). Relax this only if pads ever legitimately go negative — together
    # with the macro plumbing, never alone.
    parts = raw.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f'--expect-anchor must be "<x>,<z>", got {raw!r}'
        )
    try:
        x, z = (int(p.strip()) for p in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f'--expect-anchor must be two integers "<x>,<z>", got {raw!r}'
        ) from exc
    if x < 0 or z < 0:
        raise argparse.ArgumentTypeError(
            f"--expect-anchor coordinates must be non-negative, got {raw!r}"
        )
    return (x, z)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="combat_probe",
        description=(
            "T8 deterministic combat gate (AC8): fully-cooled ATTACKs vs the "
            "stationary dummy, asserting the exact 6,6,6,2 per-hit sequence and "
            "reconciling every recorded value against the wire's privileged "
            "opponent health. Requires a LIVE Paper server + bridge."
        ),
    )
    parser.add_argument(
        "--cycles", type=int, default=10, help="reset/kill cycles to run (default: 10)"
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="bridge host")
    parser.add_argument("--port", type=int, default=5555, help="bridge TCP port")
    parser.add_argument("--seed", type=int, default=0, help="base reset seed")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=80,
        help=(
            "per-cycle decision-step cap (default: 80; the healthy kill takes "
            "~13 windows, so hitting this cap is itself a failure)"
        ),
    )
    parser.add_argument(
        "--expect-anchor",
        type=_parse_anchor,
        default=(0, 0),
        metavar="X,Z",
        help=(
            'pad anchor the bots must spawn on, as "<x>,<z>" (default 0,0). '
            "Must match the bridge's --pad-origin."
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        ok = run_probe(
            host=args.host,
            port=args.port,
            cycles=args.cycles,
            seed=args.seed,
            max_steps=args.max_steps,
            anchor=args.expect_anchor,
        )
    except BridgeError as exc:
        print(f"[combat_probe] ABORT (bridge error): {exc}", file=sys.stderr)
        return 2
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
