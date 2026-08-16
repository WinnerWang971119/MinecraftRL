"""mc_pvp_env — Gym-style Minecraft PvP environment wrapper.

Implements the ``reset()`` / ``step()`` / ``done`` interface over the
Node<->Python TCP bridge. Translates raw JSON-lines bridge messages into frozen
observation vectors (via :mod:`env.observation_spec`, gated by the
:class:`~env.perception_filter.PerceptionFilter`) and scalar rewards (via
:mod:`env.reward`). Manages the episode lifecycle, enforces
``MAX_EPISODE_STEPS``, and drives the bridge reset RPC with the read-back gate.

------------------------------------------------------------------------------
The injectable bridge seam (the unit-test contract)
------------------------------------------------------------------------------
The env never touches a socket directly. It talks to a **bridge transport**
object that satisfies a tiny protocol (see :class:`BridgeTransport`):

    transport.send(obj: Mapping)          # serialize obj + "\n", write to wire
    transport.recv() -> dataclass         # block for ONE inbound message line,
                                          #   parse via bridge.messages.parse_line
                                          #   -> StateMsg | ResetAckMsg
    transport.connect()                   # (re)establish the connection
    transport.close()                     # tear down

The REAL transport is :class:`TcpBridgeClient`, a small TCP JSON-lines client to
the Node bridge on ``localhost:<port>``. It buffers partial reads across TCP
packet boundaries and only yields complete ``"\n"``-terminated lines, exactly as
the wire contract (``bridge/schema.md``) requires.

Unit tests inject a **fake** transport that returns scripted ``StateMsg`` /
``ResetAckMsg`` objects, so the test suite never opens a socket and no live
Minecraft server is needed. The fake-bridge contract is documented at the bottom
of this module (the ``FAKE-BRIDGE CONTRACT`` block) so T10 (the train loop) and
T20 (integration smoke) can reuse the exact same seam.

------------------------------------------------------------------------------
Episode protocol (mirrors bridge/schema.md)
------------------------------------------------------------------------------
reset:
    Python -> ``reset(episode, seed)``
    Node   -> ``reset_ack(ok, readback)``   # read-back gate result
    Node   -> ``state(...)``                # the post-reset first observation

    If ``ok`` is False (read-back gate failed) the episode must NOT start from an
    unverified world; the env RETRIES the reset ONCE and RAISES on a second
    failure (protects the MDP / AC7 — never learn from a corrupt initial state).

step:
    Python -> ``step(action, opp_action?)``
    Node   -> ``state(...)``                # raw aggregated state for the interval

    ``opp_action`` is optional: omitted (the default) the opponent takes no
    action of its own and the wire line is byte-identical to the M1/M2
    stationary-dummy path; supplied, the bridge drives the opponent through the
    same decision window. The policy that chooses it reads
    :meth:`MCPvPEnv.raw_opponent_view` — raw, ungated state that must NEVER
    reach the agent's observation.

    The raw state is gated by the PerceptionFilter, packed by
    ``build_observation``, validated, and scored by ``compute_reward`` against
    the previous observation. ``done`` fires on a death event or at
    ``MAX_EPISODE_STEPS`` (timeout) — unless the env was built with
    ``max_episode_steps=None``, the exhibition form (see
    :class:`ExhibitionConfig`), where only a death ends the match.

Privileged-data discipline (spec §5): opponent raw ``pos`` / ``yaw`` / ``vel``
reach the obs ONLY through the gating filter; opponent ``health`` never reaches
the obs at all (the obs has no slot for it). The reward consumes the privileged
``events`` block, never raw opponent health here.

Owner: T9 (Environment/bridge track)
"""

from __future__ import annotations

import json
import math
import re
import socket
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Protocol, Tuple, Union

import numpy as np

from agent.actions import Macro, N_ACTIONS
from agent.contract_config import (
    ACTION_REPEAT,
    DECISION_INTERVAL_MS,
    MAX_EPISODE_STEPS,
    SERVER_TPS,
)
from agent.reward_config import RewardConfig
from bridge.messages import (
    CloseMsg,
    ResetAckMsg,
    ResetMsg,
    SchemaError,
    StateMsg,
    StepMsg,
    parse_line,
)
from env.observation_spec import (
    OBS_DIM,
    OBS_DTYPE,
    DerivedState,
    SelfState,
    build_observation,
    held_item_id,  # noqa: F401  (re-exported convenience for callers/logging)
    validate,
)
from env.perception_filter import ATTACK_RANGE, PerceptionFilter
from env.reward import (
    TermInfo,
    compute_reward,
    compute_reward_components,
)
from opponents.scripted_bot import OpponentView

__all__ = [
    "BridgeError",
    "BridgeTransport",
    "TcpBridgeClient",
    "ExhibitionConfig",
    "MCPvPEnv",
    "DECISION_DT_SECONDS",
    "OPPONENT_ATTACK_SPEED_TICKS",
    "REWARD_COMPONENT_KEYS",
]


# ---------------------------------------------------------------------------
# Timing.
#
# Per-step wall-clock advance handed to the PerceptionFilter so its memory aging
# (``time_since_seen``) matches the real decision cadence. Derived from the
# frozen contract constants so it can never drift from ACTION_REPEAT / SERVER_TPS
# (== DECISION_INTERVAL_MS / 1000 == 0.2 s).
# ---------------------------------------------------------------------------
DECISION_DT_SECONDS: float = ACTION_REPEAT / SERVER_TPS


# ---------------------------------------------------------------------------
# Opponent swing period (the shadow cooldown's only constant).
#
# Ticks for the opponent's weapon to fully recharge. The wire carries NO
# opponent cooldown — ``state.self.attack_cooldown`` is the learner's, and
# ``state.opponent`` has no such field — so ``raw_opponent_view()`` reconstructs
# the meter in Python from the decision windows elapsed since the opponent's
# last swing (one window == ACTION_REPEAT gate ticks). NEVER from the coarse
# ``state.tick``, which rides the server world age and is flat-then-jumping —
# see _track_opponent_swing() for why that clock cannot carry this meter.
#
# MIRRORS ``IRON_SWORD_ATTACK_SPEED_TICKS`` in ``bridge/actions.js`` (the same
# ``SERVER_TPS / 1.6`` == 12.5 ticks at vanilla 1.9+ attack speed), and is
# written as that expression rather than the literal so it cannot drift from
# ``SERVER_TPS``. Both sides must agree: the bridge's ``MacroExecutor.canSwing``
# gate allows the next swing at ``elapsed >= 12.5`` ticks of ITS clock, and the
# shadow meter here reaches 1.0 at exactly the same elapsed tick count.
# ---------------------------------------------------------------------------
OPPONENT_ATTACK_SPEED_TICKS: float = SERVER_TPS / 1.6


# ---------------------------------------------------------------------------
# Reward component keys exposed in ``info`` (for T19/T11 logging).
#
# These are the per-component decomposition of the scalar ``compute_reward``
# output. As of T20 the breakdown in ``info`` comes from the SINGLE source of
# truth — ``env.reward.compute_reward_components`` — so the env and the reward
# function can never disagree (and the canonical version's ``isfinite`` guard on
# the shaping fields prevents the logged ``r_shaping`` from drifting to NaN while
# the scalar stays finite). Mirrors ``env.reward.REWARD_COMPONENT_KEYS``.
# ---------------------------------------------------------------------------
REWARD_COMPONENT_KEYS: Tuple[str, ...] = (
    "r_damage_dealt",
    "r_damage_taken",
    "r_step",
    "r_aim",
    "r_shaping",
    "r_terminal",
)


# ---------------------------------------------------------------------------
# Errors.
# ---------------------------------------------------------------------------


class BridgeError(RuntimeError):
    """Raised when the bridge transport disconnects or yields an unparseable line.

    A loud, explicit failure: the env catches a low-level transport/parse error,
    wraps it in this type, attempts exactly ONE reconnect, and re-raises (with a
    clear message) if the reconnect or the retried exchange also fails. The run
    must abort rather than silently hang or learn from corrupt state.
    """


class _ResetGateError(BridgeError):
    """Internal: the read-back gate failed twice (NOT a transport failure).

    Subclasses :class:`BridgeError` so callers still see a single error type, but
    lets :meth:`MCPvPEnv.reset` distinguish a verified-bad world (never retry — the
    bridge answered, it just refused) from a *transport* drop (reconnect + retry
    the whole exchange). The reset transport-retry loop must NOT swallow this.
    """


class _ResetProtocolError(BridgeError):
    """Internal: the bridge replied during reset, but with the WRONG message.

    An out-of-order / wrong-type reply (e.g. a ``state`` where a ``reset_ack`` was
    due) is a protocol violation, not a dropped socket. Reconnecting and re-sending
    cannot fix a bridge that speaks the protocol wrong, so :meth:`MCPvPEnv.reset`
    must propagate this immediately rather than burning its transport retries on it.
    Subclasses :class:`BridgeError` so external callers still catch one type.
    """


# ---------------------------------------------------------------------------
# Transport seam.
# ---------------------------------------------------------------------------


class BridgeTransport(Protocol):
    """Structural protocol the env requires of any bridge client (real or fake).

    The env depends only on these four methods, so unit tests can inject a fake
    that returns scripted messages and never opens a socket. The real
    implementation is :class:`TcpBridgeClient`.
    """

    def connect(self) -> None:
        """Establish (or re-establish) the connection. Idempotent if already open."""
        ...

    def send(self, obj: Mapping[str, Any]) -> None:
        """Serialize ``obj`` as a JSON line (+ trailing ``"\n"``) and write it."""
        ...

    def recv(self) -> Union[StateMsg, ResetAckMsg]:
        """Block for ONE inbound message line and return the parsed dataclass."""
        ...

    def close(self) -> None:
        """Tear down the connection. Safe to call multiple times."""
        ...


class TcpBridgeClient:
    """Real TCP JSON-lines client to the Node bridge on ``localhost:<port>``.

    One connection per arena (the wire contract has no arena id). Outbound:
    ``json.dumps(obj) + "\n"`` written to the socket. Inbound: TCP is a byte
    stream, so :meth:`recv` BUFFERS partial reads across packet boundaries and
    only parses a complete ``"\n"``-terminated line, exactly as the framing
    contract requires (``bridge/schema.md`` "Transport"). Parsing/dispatch is
    delegated to :func:`bridge.messages.parse_line`.

    This class is deliberately behind the :class:`BridgeTransport` seam so unit
    tests never construct it (no socket is opened in the test suite).

    Args:
        host: Bridge host. Defaults to ``"127.0.0.1"`` (loopback only).
        port: Bridge TCP port.
        timeout: Per-recv socket timeout in seconds (``None`` blocks forever).
            A timeout surfaces as a :class:`BridgeError` from the env, never a
            silent hang.
    """

    _RECV_CHUNK = 4096

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5555,
        timeout: Optional[float] = 30.0,
    ) -> None:
        self.host = str(host)
        self.port = int(port)
        self.timeout = timeout
        self._sock: Optional[socket.socket] = None
        # Inbound byte buffer holding bytes received but not yet framed into a
        # complete line. Persisted across recv() calls so a message split across
        # TCP packets is reassembled correctly.
        self._rbuf = bytearray()

    # -- lifecycle ---------------------------------------------------------

    def connect(self) -> None:
        """Open the TCP connection (no-op if already connected)."""
        if self._sock is not None:
            return
        try:
            sock = socket.create_connection(
                (self.host, self.port), timeout=self.timeout
            )
        except OSError as exc:
            raise BridgeError(
                f"failed to connect to bridge at {self.host}:{self.port}: {exc}"
            ) from exc
        # Disable Nagle so single small command lines are sent immediately
        # (the bridge is request/response per decision step; latency matters
        # more than coalescing tiny writes).
        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            # Not fatal — some platforms/sockets reject the option; proceed.
            pass
        sock.settimeout(self.timeout)
        self._sock = sock
        self._rbuf.clear()

    def close(self) -> None:
        """Close the socket and drop any buffered partial line. Idempotent."""
        sock = self._sock
        self._sock = None
        self._rbuf.clear()
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                sock.close()
            except OSError:
                pass

    # -- I/O ---------------------------------------------------------------

    def send(self, obj: Mapping[str, Any]) -> None:
        """Encode ``obj`` as one compact JSON line + ``"\n"`` and send it all."""
        if self._sock is None:
            raise BridgeError("cannot send on a closed bridge connection")
        line = json.dumps(dict(obj), separators=(",", ":"), ensure_ascii=False) + "\n"
        payload = line.encode("utf-8")
        try:
            self._sock.sendall(payload)
        except OSError as exc:
            raise BridgeError(f"bridge send failed: {exc}") from exc

    def recv(self) -> Union[StateMsg, ResetAckMsg]:
        """Return the next complete inbound message, buffering partial reads.

        Reads from the socket until a ``"\n"`` is buffered, splits off exactly
        one line, and parses it with :func:`bridge.messages.parse_line`. A peer
        disconnect (empty recv), a socket timeout, or an unparseable line all
        surface as :class:`BridgeError`.
        """
        if self._sock is None:
            raise BridgeError("cannot recv on a closed bridge connection")

        # Drain any complete line already sitting in the buffer first, then read
        # more bytes until a newline appears.
        while True:
            newline = self._rbuf.find(b"\n")
            if newline >= 0:
                raw_line = bytes(self._rbuf[:newline])
                # Drop the line AND its trailing newline from the buffer.
                del self._rbuf[: newline + 1]
                try:
                    text = raw_line.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise BridgeError(f"bridge sent non-UTF-8 bytes: {exc}") from exc
                if text.strip() == "":
                    # Tolerate blank keep-alive lines: keep reading.
                    continue
                try:
                    return parse_line(text)
                except SchemaError as exc:
                    raise BridgeError(f"bridge sent an invalid message: {exc}") from exc

            try:
                chunk = self._sock.recv(self._RECV_CHUNK)
            except socket.timeout as exc:
                raise BridgeError("timed out waiting for a bridge message") from exc
            except OSError as exc:
                raise BridgeError(f"bridge recv failed: {exc}") from exc

            if not chunk:
                # Orderly peer shutdown mid-stream.
                raise BridgeError("bridge closed the connection")
            self._rbuf.extend(chunk)


# ---------------------------------------------------------------------------
# Local-frame helpers.
#
# The bridge reports world-frame self velocity, but observation_spec.SelfState
# wants velocity in the agent's yaw-aligned LOCAL frame (the same frame the
# PerceptionFilter uses for the opponent). The yaw rotation convention is the
# single one documented in env/perception_filter (yaw 0 looks toward +z;
# +Z_local forward, +X_local right, +Y_local world-up).
# ---------------------------------------------------------------------------


def _world_to_local_yaw(vx: float, vy: float, vz: float, yaw: float) -> Tuple[float, float, float]:
    """Rotate a world-frame vector into the agent's yaw-aligned LOCAL frame."""
    s = math.sin(yaw)
    c = math.cos(yaw)
    x_local = vx * c + vz * s   # right  (+X_local)
    y_local = vy                # up     (+Y_local), preserved
    z_local = -vx * s + vz * c  # forward(+Z_local)
    return (float(x_local), float(y_local), float(z_local))


# ---------------------------------------------------------------------------
# EXHIBITION MODE (T3) — the config for a match against a live human.
#
# Owned here because :class:`MCPvPEnv` is what ``no_timeout`` actually acts on;
# the launcher, the reset command and the reflex shield (T5/T6/T7) consume it.
# It is deliberately NOT a bridge concern: the bridge already carries the two
# process-local keys it needs (``opponentMode`` / ``challengerUsername``, wired
# through ``bridge/run.js``), and it has no notion of an episode horizon at all.
# ---------------------------------------------------------------------------

#: Minecraft usernames as this project uses them (offline mode, ops.json).
#: Mirrors ``MACRO_USERNAME_RE`` in ``bridge/bot.js``.
_USERNAME_RE = re.compile(r"^[A-Za-z0-9_]{1,16}$")


@dataclass(frozen=True)
class ExhibitionConfig:
    """Settings for one human-exhibition match.

    Frozen: the launcher builds it once and hands the same object to the bridge
    wiring, the env and the reflex shield, so no consumer can retune the match
    out from under another.

    Args:
        challenger_username: The pinned challenger's Minecraft username, or
            ``None`` for "the first non-agent player who enters the pad claims
            the slot" (the bridge's first-claimant latch). **Prefer a pinned
            name on demo day**: it is the operator naming the opponent up front,
            which no heuristic can beat.
        no_timeout: Disable the episode horizon while a human is the opponent.
            A person deciding what to do is not a stalling agent, and a match
            that ends mid-fight because 400 decision steps elapsed reads to a
            classroom as the agent giving up. See :attr:`env_max_episode_steps`
            for the form this takes — there is no sentinel and no large integer.
            Must be exactly ``True`` or ``False``: a truthy stand-in such as the
            string ``"false"`` would disable the horizon while reading, to
            whoever passed it, as if it had been kept.
        auto_reset: Whether a finished match restarts by itself. **Must be
            exactly False** (AC4): the match ends, the result is reported, and
            the operator arms the next challenger with the separate reset
            command. Any other value raises rather than silently promising a
            restart nothing implements.
        reflex_blind_steps: Consecutive blind decision steps before the
            exhibition-only reflex shield overrides the action with
            ``TURN_TO_LAST_SEEN``. Config only here — T7 owns the behavior. The
            default of 8 is ~1.6 s at the frozen 200 ms decision interval.

    Raises:
        ValueError: on a malformed username, a ``no_timeout`` that is not
            exactly ``True``/``False``, an ``auto_reset`` that is not exactly
            ``False``, or a non-int / negative ``reflex_blind_steps``.
    """

    challenger_username: Optional[str] = None
    no_timeout: bool = True
    auto_reset: bool = False
    reflex_blind_steps: int = 8

    def __post_init__(self) -> None:
        name = self.challenger_username
        if name is not None and (
            not isinstance(name, str) or _USERNAME_RE.match(name) is None
        ):
            # A pin that cannot match a real player is an exhibition in which
            # nobody is ever the opponent — indistinguishable, from the operator's
            # side, from nobody having joined. Refuse it at construction.
            raise ValueError(
                "challenger_username must be a Minecraft username matching "
                f"{_USERNAME_RE.pattern} or None, got {name!r}"
            )
        if self.no_timeout is not True and self.no_timeout is not False:
            # `is not True/False` rather than a truthiness test: the string
            # "false" is truthy, so a launcher that forwarded an unparsed flag
            # would disable the horizon while its operator believed they had
            # kept it — and a horizon that is silently off looks like a match
            # that simply has not ended yet.
            raise ValueError(
                f"no_timeout must be exactly True or False, got {self.no_timeout!r}"
            )
        if self.auto_reset is not False:
            # The message names the VALUE requirement, not just the True case:
            # this branch also rejects 0, None and "" — falsy non-bools that a
            # "True is not implemented" message would describe wrongly.
            raise ValueError(
                "auto_reset must be exactly False, got "
                f"{self.auto_reset!r}: an auto-restart is not implemented and "
                "would violate AC4 (after a death the match must not "
                "auto-restart); the operator runs the separate reset command "
                "between challengers"
            )
        if not isinstance(self.reflex_blind_steps, int) or isinstance(
            self.reflex_blind_steps, bool
        ):
            raise ValueError(
                f"reflex_blind_steps must be an int, got {self.reflex_blind_steps!r}"
            )
        if self.reflex_blind_steps < 0:
            raise ValueError(
                f"reflex_blind_steps must be >= 0, got {self.reflex_blind_steps}"
            )

    @property
    def env_max_episode_steps(self) -> Optional[int]:
        """The ``max_episode_steps`` an env for this match must be built with.

        ``None`` when :attr:`no_timeout` is set — **the one and only form**
        "disabled" takes. Not a sentinel, and emphatically not a large integer:
        a large integer is a timeout that has merely been moved somewhere less
        convenient to notice, and it would fire in the middle of a long match in
        front of an audience.
        """
        return None if self.no_timeout else MAX_EPISODE_STEPS


# ---------------------------------------------------------------------------
# The environment.
# ---------------------------------------------------------------------------


class MCPvPEnv:
    """Gym-style Minecraft PvP environment over the injectable bridge transport.

    Standard ``reset()`` / ``step()`` loop. The opponent block of every
    observation is produced by an internal :class:`PerceptionFilter` so the
    agent only ever sees fair (FOV + LoS + memory gated) opponent features;
    opponent raw health never enters the observation.

    Args:
        transport: A bridge client satisfying :class:`BridgeTransport` (the real
            :class:`TcpBridgeClient` in production, a fake in unit tests). The env
            owns its lifecycle (calls ``connect`` on construction and ``close``
            on :meth:`close`).
        reward_config: Reward coefficients. Defaults to the frozen
            :class:`RewardConfig` starting table.
        perception_filter: Optional pre-built filter (tests inject one with a
            custom ``los_clear`` / FOV). Defaults to a fresh :class:`PerceptionFilter`.
        max_episode_steps: Decision steps before a timeout truncation. Defaults to
            the frozen :data:`MAX_EPISODE_STEPS`. Pass ``None`` to disable
            truncation entirely — the exhibition path (T3/AC4), where the
            opponent is a person and an episode horizon would end a live match
            mid-fight. ``None`` is the ONLY spelling of "disabled": a sentinel
            or a very large integer would be a timeout that still fires, just
            somewhere harder to notice. Build one with
            :attr:`ExhibitionConfig.env_max_episode_steps`.
        dt: Per-step seconds handed to the filter's memory aging. Defaults to the
            derived :data:`DECISION_DT_SECONDS`.
        auto_connect: Connect the transport during construction. Defaults to True;
            tests with a fake that needs no connect can leave it True (a fake
            ``connect`` is a harmless no-op).

    Action space:
        Discrete ``N_ACTIONS`` (== 8), the frozen :class:`~agent.actions.Macro`
        enum. :meth:`step` accepts an ``int`` (or ``Macro``) in ``[0, N_ACTIONS)``.

    Observation space:
        ``np.ndarray`` of shape ``(OBS_DIM,)`` dtype float32, always passing
        :func:`env.observation_spec.validate`.
    """

    #: Number of discrete actions (frozen). Mirrors ``agent.actions.N_ACTIONS``.
    n_actions: int = N_ACTIONS
    #: Observation vector length (frozen). Mirrors ``observation_spec.OBS_DIM``.
    obs_dim: int = OBS_DIM

    #: Max times reset() re-attempts the FULL reset exchange after a *transport*
    #: drop (reconnect + re-send from scratch). reset() is idempotent and carries
    #: no in-flight episode state, so a dropped socket here — e.g. a periodic eval
    #: stole the bridge's single connection mid-run — is recoverable. Bounded so a
    #: genuinely-down bridge still aborts LOUDLY instead of spinning forever. This
    #: is layered AROUND the read-back-gate retry, which is a separate concern.
    _RESET_MAX_TRANSPORT_ATTEMPTS: int = 3

    def __init__(
        self,
        transport: BridgeTransport,
        reward_config: Optional[RewardConfig] = None,
        perception_filter: Optional[PerceptionFilter] = None,
        max_episode_steps: Optional[int] = MAX_EPISODE_STEPS,
        dt: float = DECISION_DT_SECONDS,
        auto_connect: bool = True,
    ) -> None:
        # ``None`` is "no truncation" and is handled BEFORE the range check —
        # the check reads ``<= 0``, and ``None <= 0`` is a TypeError on Python 3.
        if max_episode_steps is not None and max_episode_steps <= 0:
            raise ValueError(
                f"max_episode_steps must be > 0 or None, got {max_episode_steps}"
            )
        if dt < 0.0:
            raise ValueError(f"dt must be >= 0, got {dt}")

        self._transport = transport
        self._cfg = reward_config if reward_config is not None else RewardConfig()
        self._filter = (
            perception_filter if perception_filter is not None else PerceptionFilter()
        )
        #: Episode horizon in decision steps, or None for no truncation at all.
        self._max_steps: Optional[int] = (
            None if max_episode_steps is None else int(max_episode_steps)
        )
        self._dt = float(dt)

        # Per-episode mutable state (initialized by reset()).
        self._episode: int = -1
        self._step_count: int = 0
        self._prev_obs: Optional[np.ndarray] = None
        # The most recent RAW state message, kept ONLY to serve
        # raw_opponent_view(). It is never consulted by the observation path —
        # _state_to_obs() gates the state it is handed, and this attribute is
        # not one of its inputs.
        self._last_state: Optional[StateMsg] = None
        # Shadow attack-cooldown tracker for the opponent: the 0-based index of
        # the decision window in which its last swing actually fired, or None
        # for "no swing yet this episode" (== fully charged). A WINDOW COUNT,
        # never a wire tick — see _track_opponent_swing() for why the coarse
        # ``state.tick`` cannot carry this meter.
        self._opp_last_swing_window: Optional[int] = None
        # True only while reset() is doing wire I/O. Switches _send/_recv from
        # their step-time semantics (self-recover on send; reconnect-then-re-raise
        # on recv) to "propagate the raw transport error" so reset()'s own bounded
        # transport-retry loop owns the reconnect-and-redo decision. step() never
        # sets this, so mid-episode desync protection is untouched.
        self._in_reset_exchange: bool = False
        self._done: bool = True  # no episode in progress until reset()

        if auto_connect:
            self._transport.connect()

    # -- public properties -------------------------------------------------

    @property
    def episode(self) -> int:
        """Index of the current/last episode (-1 before the first reset)."""
        return self._episode

    @property
    def step_count(self) -> int:
        """Number of decision steps taken in the current episode."""
        return self._step_count

    @property
    def max_episode_steps(self) -> Optional[int]:
        """Episode horizon in decision steps, or ``None`` for no truncation.

        ``None`` is the exhibition form (T3/AC4): only a death ends the match.
        Exposed so a caller can assert what it built rather than infer the
        horizon from how long an episode happened to run.
        """
        return self._max_steps

    # -- lifecycle ---------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Start a new episode and return its initial observation vector.

        Sends a ``reset`` and awaits the ``reset_ack`` read-back gate. If the gate
        reports ``ok == False`` the world is unverified, so the reset is RETRIED
        ONCE; a second ``ok == False`` RAISES :class:`BridgeError` rather than
        starting an episode from a corrupt state (protects the MDP / AC7). On
        success the PerceptionFilter memory and step counter are reset and the
        post-reset ``state`` message is gated/packed into the initial observation.

        Transport resilience: unlike :meth:`step`, ``reset`` is idempotent and has
        no in-flight episode state to lose, so a *transport* drop during the
        exchange (e.g. a concurrent periodic eval transiently stole the bridge's
        single connection) is recoverable — the env reconnects and re-sends the
        whole reset from scratch, up to :data:`_RESET_MAX_TRANSPORT_ATTEMPTS`
        times, then aborts LOUDLY. The read-back-gate retry above is a separate
        concern layered INSIDE each transport attempt; a verified-bad world
        (``ok == False`` twice) is never retried at the transport level.

        Args:
            seed: Per-episode RNG seed forwarded to the bridge (spawn jitter, gear,
                opponent choice). ``None`` becomes ``0`` on the wire (the schema
                requires an integer seed).

        Returns:
            The initial observation ``np.ndarray`` of shape ``(OBS_DIM,)``.

        Raises:
            BridgeError: if the bridge disconnects and cannot be recovered within
                the bounded transport retries, sends an out-of-order/invalid
                message, or fails the read-back gate twice.
        """
        self._episode += 1
        wire_seed = 0 if seed is None else int(seed)

        # Bounded transport-retry loop around the WHOLE reset exchange. Each
        # attempt does a full reset->reset_ack (with the read-back-gate retry) plus
        # the post-reset state read. A raw transport BridgeError here means the
        # socket died (the eval-stole-the-connection race); reconnect and redo from
        # scratch. A _ResetGateError (the bridge answered but refused) is NOT a
        # transport fault — it propagates immediately and is never retried here.
        last_transport_exc: Optional[BridgeError] = None
        for attempt in range(self._RESET_MAX_TRANSPORT_ATTEMPTS):
            self._in_reset_exchange = True
            try:
                return self._reset_protocol(wire_seed)
            except (_ResetGateError, _ResetProtocolError):
                # NOT a transport fault: a verified-bad world (gate refused) or a
                # wrong/out-of-order reply. Reconnecting cannot fix either — abort.
                raise
            except BridgeError as exc:
                # Transport drop mid-reset. Reconnect on a fresh socket and retry
                # the full exchange — reset has no in-flight state to corrupt.
                last_transport_exc = exc
                if attempt + 1 >= self._RESET_MAX_TRANSPORT_ATTEMPTS:
                    break
                self._reconnect_or_abort("reset")
            finally:
                self._in_reset_exchange = False

        raise BridgeError(
            "bridge reset failed: the connection dropped during the reset "
            f"exchange and {self._RESET_MAX_TRANSPORT_ATTEMPTS} reconnect+retry "
            f"attempts did not recover (episode {self._episode}, seed={wire_seed}); "
            "the bridge appears down. Last transport error: "
            f"{last_transport_exc}"
        ) from last_transport_exc

    def _reset_protocol(self, wire_seed: int) -> np.ndarray:
        """One full reset exchange: gate (with its own retry) + post-reset state.

        Runs the read-back gate (send ``reset``, await ``reset_ack``; retry ONCE on
        ``ok == False`` then raise :class:`_ResetGateError`), then reads the
        post-reset ``state`` and packs it into the initial observation. Any raw
        transport :class:`BridgeError` here propagates to :meth:`reset`'s bounded
        transport-retry loop; a :class:`_ResetGateError` is NOT retried there.
        """
        # Read-back gate with a single retry. Each attempt is a full
        # reset -> reset_ack exchange; on ok==False we try once more, then raise.
        ack: Optional[ResetAckMsg] = None
        last_readback: Dict[str, Any] = {}
        for _attempt in range(2):  # at most one retry
            ack = self._reset_exchange(wire_seed)
            last_readback = dict(ack.readback)
            if ack.ok:
                break
        if ack is None or not ack.ok:
            raise _ResetGateError(
                "bridge reset read-back gate failed twice for "
                f"episode {self._episode} (seed={wire_seed}); refusing to start "
                f"an episode from an unverified state. last readback={last_readback!r}"
            )

        # Gate confirmed: clear per-episode memory BEFORE consuming the first
        # state so the opponent starts ABSENT (no stale memory across episodes).
        self._filter.reset()
        self._step_count = 0
        self._done = False
        # Re-arm the opponent's shadow swing meter alongside the filter memory:
        # the bridge calls executor.resetCooldown() on every reset, so carrying
        # last episode's swing window across would leave a fresh opponent unable
        # to attack for its first few decisions of the new episode.
        self._opp_last_swing_window = None
        # Drop the previous episode's latched raw state alongside it: if the
        # post-reset state read below dies (and the transport retries run out),
        # raw_opponent_view() must raise its "before any state" error, not
        # silently serve the PREVIOUS episode's world as if it were current.
        self._last_state = None

        # The post-reset first observation is the next inbound `state` message.
        state = self._recv_state()
        obs = self._state_to_obs(state)
        self._prev_obs = obs
        return obs

    def step(
        self, action: int, opp_action: Optional[int] = None
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Advance one decision step.

        Sends a ``step(action)``, awaits one ``state`` reply, gates + packs it
        into the next observation, validates it, and scores the transition with
        :func:`env.reward.compute_reward`.

        ``opp_action`` is the optional opponent-acts path: pass the macro an
        opponent policy chose and the bridge drives the opponent handle through
        a second executor in the same window. Leave it ``None`` (the default)
        and the wire line is byte-identical to what it was before the field
        existed — the M1/M2 stationary-dummy path is untouched.

        ``done`` fires when the bridge reports a death (``events.i_died`` →
        terminal loss, ``events.opponent_died`` → terminal win) OR when the step
        count reaches ``max_episode_steps`` (timeout / draw) — the latter only
        when a horizon is set at all; ``max_episode_steps=None`` (exhibition,
        AC4) never truncates. The three terminal outcomes are mutually
        exclusive; a same-step double-death is resolved as a loss (the learner
        dying is never a win).

        Nothing here restarts an episode. ``done`` leaves the env finished and
        the next :meth:`step` raises until :meth:`reset` is called explicitly —
        which is exactly AC4's "after a death the match does not auto-restart",
        and the reason :class:`ExhibitionConfig` refuses ``auto_reset=True``.

        Args:
            action: Discrete action index (or :class:`~agent.actions.Macro`) in
                ``[0, n_actions)``.
            opp_action: Optional discrete action index for the OPPONENT, in the
                same ``[0, n_actions)`` range and validated identically.
                ``None`` (the default) means the opponent takes no action of its
                own and the field is omitted from the wire line entirely.

        Returns:
            ``(obs, reward, done, info)`` where ``obs`` is a validated
            ``(OBS_DIM,)`` float32 vector, ``reward`` is the scalar step reward,
            ``done`` is the terminal flag, and ``info`` is a dict that includes
            the per-reward-component breakdown (keys in
            :data:`REWARD_COMPONENT_KEYS`) plus the raw events and win/loss/timeout
            flags.

        Raises:
            ValueError: if ``action`` is out of range or no episode is in progress.
            BridgeError: if the bridge disconnects or sends an invalid/out-of-order
                message (after one reconnect attempt).
        """
        if self._done:
            raise ValueError(
                "step() called on a finished/unstarted episode; call reset() first"
            )
        act = int(action)
        if not (0 <= act < N_ACTIONS):
            raise ValueError(
                f"action must be in [0, {N_ACTIONS}), got {action!r}"
            )
        # Same validation as `action` — N_ACTIONS is frozen at 8, and this field
        # widens WHO acts, never the action space. None stays None: it is the
        # "opponent does nothing" case, not a zeroth macro.
        opp_act: Optional[int] = None
        if opp_action is not None:
            opp_act = int(opp_action)
            if not (0 <= opp_act < N_ACTIONS):
                raise ValueError(
                    f"opp_action must be in [0, {N_ACTIONS}) or None, got {opp_action!r}"
                )
        if self._prev_obs is None:
            # Defensive: reset() always sets _prev_obs before clearing _done.
            raise ValueError("internal error: no previous observation; call reset()")

        # Send the action and await exactly one state reply.
        self._send(StepMsg(action=act, opp_action=opp_act))
        state = self._recv_state()

        # Fold this window into the opponent's shadow swing meter. MUST stay
        # ABOVE the `self._step_count += 1` below: the stamp is the 0-based
        # index of the window that swung, read from the PRE-increment counter
        # (see _track_opponent_swing's ordering invariant) — and it must land
        # before step() returns so a raw_opponent_view() taken after this step
        # sees the swing that just happened.
        self._track_opponent_swing(opp_act, state)

        # Gate + pack the new observation (s').
        obs = self._state_to_obs(state)
        prev_obs = self._prev_obs

        # --- termination ---------------------------------------------------
        events = state.events
        self._step_count += 1

        died = bool(events.i_died)
        opp_died = bool(events.opponent_died)
        # A None horizon NEVER truncates (T3/AC4): against a human the only
        # things that end a match are a death and the operator's reset. Written
        # as an explicit None test rather than a comparison against a stand-in
        # ceiling so there is no number here that could ever be reached.
        timed_out = self._max_steps is not None and self._step_count >= self._max_steps

        done = died or opp_died or timed_out
        # Resolve mutually-exclusive outcome flags. A simultaneous double-death is
        # a loss (the learner dying can never count as a win). Timeout only counts
        # when no death occurred this step.
        lost = died
        won = opp_died and not died
        timeout = timed_out and not died and not opp_died

        terminal = TermInfo(done=done, won=won, lost=lost, timeout=timeout)

        # --- reward + per-component breakdown ------------------------------
        # The scalar reward and its per-component decomposition both flow from the
        # SINGLE source of truth in env.reward, so the logged components can never
        # drift from the scalar (and they carry the canonical isfinite guard on the
        # shaping fields — a non-finite opponent position no longer poisons the
        # logged r_shaping while the scalar stays finite).
        reward = compute_reward(events, obs, prev_obs, terminal, self._cfg)
        components = compute_reward_components(
            events, obs, prev_obs, terminal, self._cfg
        )

        # Advance episode state.
        self._prev_obs = obs
        self._done = done

        info: Dict[str, Any] = {
            "episode": self._episode,
            "step": self._step_count,
            "won": won,
            "lost": lost,
            "timeout": timeout,
            "tick": state.tick,
            "code_version": state.code_version,
            "events": {
                "damage_dealt": float(events.damage_dealt),
                "damage_taken": float(events.damage_taken),
                "i_died": bool(events.i_died),
                "opponent_died": bool(events.opponent_died),
            },
        }
        info.update(components)

        return obs, float(reward), done, info

    # -- the raw opponent view (scripted-opponent seam) --------------------

    def raw_opponent_view(self) -> OpponentView:
        """RAW, ungated combat state as the *opponent* sees it. **Never the obs.**

        This is the ONLY sanctioned raw-state accessor and it exists for one
        consumer: the omniscient :class:`~opponents.scripted_bot.ScriptedBot`,
        which cannot be driven from :meth:`step`'s return value because that
        observation is deliberately gated (FOV cone + line of sight + memory).
        The view is built from the most recent ``state`` message with **no
        gating applied at all**.

        **It must never be routed into the agent's observation.** The obs is
        built solely by :meth:`_state_to_obs` from the ``state`` message it is
        handed, through the :class:`~env.perception_filter.PerceptionFilter`;
        this method is not one of its inputs and calling it has no effect on any
        observation. Feeding this return value into the agent would leak
        privileged state into training and quietly invalidate every result.

        **TRAINING ONLY — not valid in exhibition mode.** ``self_health`` can
        only come from ``state.opponent.health``, which the bridge reports as
        ``0`` when the opponent is a *human* (mineflayer never populates
        ``entity.health`` for another player, so there is no source). Against a
        person this view would read ``self_health == 0`` — a HARD-preset bot
        would flee forever, and anything scoring on it would see a permanent
        corpse. That is not a live hazard today, because exhibition mode has no
        scripted opponent at all: the human *replaces* the opponent bot. It is
        recorded here so nobody wires the two together later.

        Field mapping (the polarity is the thing to get right — ``self_*`` is
        the OPPONENT, ``target_*`` is the LEARNER):

        ==========================  ==========================================
        ``OpponentView`` field      Source
        ==========================  ==========================================
        ``self_pos/yaw/health``     ``state.opponent`` (the scripted bot itself)
        ``target_pos/yaw/health``   ``state.self`` (the learner it is fighting)
        ``distance``                horizontal (XZ) separation, per the type
        ``in_attack_range``         ``distance <= ATTACK_RANGE``
        ``attack_cooldown``         the Python shadow tracker (see below)
        ``can_see_target``          always ``True`` — this bot is omniscient
        ``last_known_target_pos``   the target's position now; with
                                    ``can_see_target`` always ``True`` the
                                    most-recently-seen position IS the current
                                    one
        ==========================  ==========================================

        Yaw is converted to **degrees**: the wire carries radians, and
        ``OpponentView`` documents its ``self_yaw`` / ``target_yaw`` as degrees.
        This method is the only place that conversion happens.

        ``attack_cooldown`` is clamped to **exactly** ``1.0`` (never ``1.0``
        minus a float hair) because ``ScriptedBot`` tests readiness with a
        deliberately tight ``>= 1.0 - 1e-6``: a value a hair under 1.0 makes it
        never attack at all, which presents as a mysteriously passive opponent
        rather than as an error.

        Returns:
            A fresh :class:`~opponents.scripted_bot.OpponentView`.

        Raises:
            ValueError: if no ``state`` has been received yet (``reset()`` has
                not run). There is no raw state to report and inventing a
                zeroed one would hand the bot a view in which both fighters are
                at the origin and dead.
        """
        state = self._last_state
        if state is None:
            raise ValueError(
                "raw_opponent_view() called before any state was received; "
                "call reset() first"
            )

        opp_raw = state.opponent
        self_raw = state.self_state

        # The opponent is the "self" of this view; the learner is its target.
        view_self_pos = (
            float(opp_raw.pos[0]),
            float(opp_raw.pos[1]),
            float(opp_raw.pos[2]),
        )
        view_target_pos = (
            float(self_raw.pos[0]),
            float(self_raw.pos[1]),
            float(self_raw.pos[2]),
        )

        # Horizontal (XZ-plane) separation, matching OpponentView's documented
        # `distance` and the PerceptionFilter's own in-range test — a jumping
        # fighter is not out of melee range.
        distance = math.hypot(
            view_target_pos[0] - view_self_pos[0],
            view_target_pos[2] - view_self_pos[2],
        )

        return OpponentView(
            self_pos=view_self_pos,
            self_yaw=math.degrees(float(opp_raw.yaw)),
            # PRIVILEGED and training-only — see the exhibition note above.
            self_health=float(opp_raw.health),
            target_pos=view_target_pos,
            target_yaw=math.degrees(float(self_raw.yaw)),
            target_health=float(self_raw.health),
            distance=distance,
            # The module-level ATTACK_RANGE, deliberately NOT this env's filter
            # instance: tests (and a future tuned filter) override the LEARNER's
            # perception, and an omniscient opponent must not inherit those.
            in_attack_range=distance <= ATTACK_RANGE,
            attack_cooldown=self._opponent_attack_cooldown(),
            # Omniscient by design: no FOV, no LoS, no memory. The field is on
            # the type for a future filtered mode that does not exist yet.
            can_see_target=True,
            last_known_target_pos=view_target_pos,
        )

    def close(self) -> None:
        """Send ``close`` (best-effort) and tear down the transport.

        Safe to call multiple times. A transport error during teardown is
        swallowed — the connection is going away regardless.
        """
        try:
            self._transport.send(CloseMsg().to_dict())
        except (BridgeError, OSError):
            # The peer may already be gone; closing the socket is what matters.
            pass
        finally:
            self._done = True
            try:
                self._transport.close()
            except OSError:
                pass

    # -- context-manager sugar --------------------------------------------

    def __enter__(self) -> "MCPvPEnv":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # -- transport helpers (with one-reconnect resilience) ----------------

    def _reset_exchange(self, seed: int) -> ResetAckMsg:
        """Send one ``reset`` and await the matching ``reset_ack`` dataclass.

        A wrong reply TYPE is a protocol violation (``_ResetProtocolError``), kept
        distinct from a dropped socket so reset() does not waste a transport retry
        reconnecting against a bridge that just spoke out of order.
        """
        self._send(ResetMsg(episode=self._episode, seed=seed))
        msg = self._recv()
        if not isinstance(msg, ResetAckMsg):
            raise _ResetProtocolError(
                f"expected a reset_ack after reset, got {type(msg).__name__}"
            )
        return msg

    def _recv_state(self) -> StateMsg:
        """Await one inbound message and require it to be a ``state``.

        During reset() a wrong reply TYPE raises ``_ResetProtocolError`` (a
        non-retryable protocol violation); during step() it is the plain
        ``BridgeError`` below — both subclass ``BridgeError`` for external callers.
        """
        msg = self._recv()
        if not isinstance(msg, StateMsg):
            if self._in_reset_exchange:
                raise _ResetProtocolError(
                    f"expected a state message after reset_ack, got "
                    f"{type(msg).__name__}"
                )
            raise BridgeError(
                f"expected a state message, got {type(msg).__name__}"
            )
        # Latch the RAW state here — the one place both the reset and the step
        # path funnel through — so raw_opponent_view() can never be served a
        # stale snapshot because a caller forgot to record one.
        self._last_state = msg
        return msg

    def _send(self, message: Union[ResetMsg, StepMsg, CloseMsg]) -> None:
        """Send an outbound dataclass, reconnecting once on a transport failure.

        During a reset exchange (``_in_reset_exchange``) a transport failure is
        propagated RAW so reset()'s own bounded reconnect+retry loop redoes the
        whole exchange from scratch; the in-helper single-reconnect-retry is for
        the step path only.
        """
        try:
            self._transport.send(message.to_dict())
        except BridgeError:
            if self._in_reset_exchange:
                raise
            self._reconnect_or_abort("send")
            # Retry the send exactly once on the fresh connection.
            try:
                self._transport.send(message.to_dict())
            except BridgeError as exc:
                raise BridgeError(
                    f"bridge send failed after one reconnect: {exc}"
                ) from exc

    def _recv(self) -> Union[StateMsg, ResetAckMsg]:
        """Receive one inbound message, reconnecting once on a transport failure.

        STEP semantics (the default): a reconnect cannot recover the in-flight
        reply that was lost with the old socket, so after a successful reconnect
        this still raises — the caller (step) aborts the run loudly rather than
        silently desyncing the request/response stream. The reconnect exists so
        the NEXT episode can proceed and so the failure carries a clear, actionable
        message.

        RESET semantics (``_in_reset_exchange``): the raw transport error is
        propagated WITHOUT reconnecting here, because reset() is idempotent and its
        own bounded retry loop owns the reconnect-and-redo-from-scratch decision.
        A lost reply mid-reset desyncs nothing — there is no in-flight episode.
        """
        try:
            return self._transport.recv()
        except BridgeError as first_exc:
            if self._in_reset_exchange:
                raise
            self._reconnect_or_abort("recv")
            raise BridgeError(
                "bridge connection dropped mid-exchange; reconnected, but the "
                f"in-flight reply is lost and the run must restart this episode: {first_exc}"
            ) from first_exc

    def _reconnect_or_abort(self, where: str) -> None:
        """Attempt exactly ONE reconnect; abort loudly if it fails."""
        try:
            self._transport.close()
        except OSError:
            pass
        try:
            self._transport.connect()
        except BridgeError as exc:
            raise BridgeError(
                f"bridge {where} failed and the single reconnect attempt also "
                f"failed; aborting the run: {exc}"
            ) from exc

    # -- opponent shadow attack cooldown ----------------------------------

    def _track_opponent_swing(
        self, opp_action: Optional[int], state: StateMsg
    ) -> None:
        """Fold one window's ``opp_action`` into the opponent's shadow swing meter.

        Records a swing only when this window actually asked the opponent to
        ``ATTACK`` **and** the bridge did not report that the swing failed to
        fire (``state.opp_action_executed is False``). An absent/``None`` report
        is treated as "it fired": the opposite default would leave the meter
        pinned at 1.0 forever, and a permanently-charged meter is exactly the
        flailing opponent the readiness epsilon exists to prevent.

        The stamp is a **WINDOW COUNT**, deliberately never ``state.tick``.
        The wire tick rides the learner's server world age (``bot.js
        _serverTick``), which the server's ``update_time`` packet refreshes only
        ~once per second — so it is FLAT for several decision windows, then
        jumps ~20 (``eval/benchmark.py`` TickDeltaTpsProvider records the same
        measured shape). The bridge's swing gate rides a different clock,
        ``_currentTick``, which advances exactly ``ACTION_REPEAT`` per window.
        A meter fed the coarse clock reads ready 1-5 windows off the gate, and
        the ready-early case LOCKS IN: ``ScriptedBot`` returns ``ATTACK``,
        ``canSwing`` blocks it, ``opp_action_executed=false`` suppresses the
        stamp, the meter stays pinned at 1.0, and the opponent mashes ``ATTACK``
        every window instead of strafing or jumping. Counting decision windows
        reconstructs the gate's clock by construction: one :meth:`step` is one
        window is ``ACTION_REPEAT`` gate ticks.

        ORDERING INVARIANT: this runs BEFORE :meth:`step` increments
        ``_step_count``, so the stamp is the 0-based index of the window that
        swung, and ``_step_count - stamp`` afterwards counts the windows elapsed
        since that window's start. Moving the increment above this call would
        shift the meter one whole window slow (ready at ``T+20``, not ``T+16``);
        the worked-example gate test pins that direction, and the coarse-clock
        tests pin the fast one.
        """
        if opp_action != int(Macro.ATTACK):
            return
        if state.opp_action_executed is False:
            # Explicitly reported as not fired (gate-blocked, or nothing to
            # swing at). Mirrors the executor's own rule that a swing at nothing
            # does not start the cooldown.
            return
        self._opp_last_swing_window = self._step_count

    def _opponent_attack_cooldown(self) -> float:
        """The opponent's shadow swing **METER** in ``[0, 1]`` — charge, not a gate.

        The quantity is the one :class:`~opponents.scripted_bot.OpponentView`
        documents for ``attack_cooldown`` (1.0 == fully charged), reconstructed
        on the swing gate's own clock: ``elapsed = (decision windows since the
        swing) * ACTION_REPEAT`` gate ticks, which equals the bridge's
        ``currentTick - lastSwingTick`` by construction (one :meth:`step` is
        one window). Its SATURATION therefore coincides with the gate at window
        granularity: the meter reads 1.0 exactly when ``MacroExecutor.canSwing``
        would allow the next window's swing (``elapsed >=
        OPPONENT_ATTACK_SPEED_TICKS``), and reads the true partial charge in
        between.

        RESET SEEDING — deliberately none. "No swing yet this episode" returns
        1.0, with no analogue of the bridge's ``_meterResetTick`` regear anchor
        (``bot.js`` issue #28), because that anchor models a hazard the
        opponent's channel does not have:

        * The thing this meter must mirror — the opponent-side swing gate —
          starts every episode OPEN: the bridge clears ``executor.lastSwingTick``
          on reset. A shadow seeded below the gate would hold a fresh opponent
          out of a fight it is allowed to swing in.
        * The server's own meter is NOT re-zeroed by the opponent's reset.
          ``Player.tick()`` resets it only on a main-hand item-TYPE change,
          compared once per tick (decompiled from the pinned jar; quoted at
          ``bot.js:2102-2113``), and ``arena:spawn_dummy_pad`` runs ``$clear``
          with **no** ``give`` at all, inside one single-tick function — the
          dummy's hand goes air -> air, so the comparison never sees a change,
          the empty-handed JOIN case included. The LEARNER needed the anchor
          precisely because its regear is a same-tick clear+give of a sword
          and its join is air -> sword.
        * The residual — the opponent's final kill swing of the previous
          episode zeroing the server-side ticker across a reset — affects hit
          STRENGTH only, never gate agreement, and the bridge's own gate is
          equally blind to it by design.

        Returns **exactly** ``1.0`` (via ``min``, never an arithmetic result)
        whenever the meter is full, including the "no swing yet this episode"
        case. ``ScriptedBot`` reads readiness as ``>= 1.0 - 1e-6``, so a value a
        hair under 1.0 would silently stop it attacking altogether.
        """
        last_swing_window = self._opp_last_swing_window
        if last_swing_window is None:
            return 1.0
        if OPPONENT_ATTACK_SPEED_TICKS <= 0.0:
            # Degenerate weapon speed: never block, matching the bridge's own
            # defensive branches in canSwing()/computeAttackCooldown().
            return 1.0
        elapsed = (self._step_count - last_swing_window) * ACTION_REPEAT
        if elapsed <= 0:
            return 0.0
        return min(1.0, elapsed / OPPONENT_ATTACK_SPEED_TICKS)

    # -- state -> observation ---------------------------------------------

    def _state_to_obs(self, state: StateMsg) -> np.ndarray:
        """Gate + pack a raw ``state`` message into a validated observation vector.

        The opponent block goes through the PerceptionFilter (FOV + LoS + memory)
        so only fair features reach the obs. Self velocity is rotated from the
        bridge's world frame into the agent's local frame. Opponent ``health`` is
        never read here — it is privileged and obs-forbidden.
        """
        self_raw = state.self_state
        opp_raw = state.opponent

        # Gate the opponent: PerceptionFilter consumes the RAW world-frame self +
        # opponent (it reads pos/yaw/pitch/velocity; opponent health is ignored).
        gated_opp, derived = self._filter.filter(self_raw, opp_raw, dt=self._dt)

        # Build the FULL self block. Bridge velocity is world-frame; the obs wants
        # it in the agent's yaw-aligned local frame (build_observation divides by
        # MAX_SPEED but does not rotate).
        vel_local = _world_to_local_yaw(
            float(self_raw.velocity[0]),
            float(self_raw.velocity[1]),
            float(self_raw.velocity[2]),
            float(self_raw.yaw),
        )
        self_state = SelfState(
            health=float(self_raw.health),
            yaw=float(self_raw.yaw),
            pitch=float(self_raw.pitch),
            vel_local=vel_local,
            on_ground=bool(self_raw.on_ground),
            held_item=self_raw.held_item,
            attack_cooldown=float(self_raw.attack_cooldown),
        )

        obs = build_observation(self_state, gated_opp, derived)
        # Fail loudly on a malformed vector rather than feeding the agent garbage.
        validate(obs)
        return obs.astype(OBS_DTYPE, copy=False)


# ===========================================================================
# FAKE-BRIDGE CONTRACT (reuse this in T10 train-loop tests and T20 smoke tests)
# ===========================================================================
#
# The env depends ONLY on the four-method `BridgeTransport` protocol above, so a
# unit test injects a fake that returns scripted inbound dataclasses and records
# what the env sent. There is no socket and no live server.
#
# A conformant fake transport must:
#
#   * connect()      -> no-op (or flip an `is_open` flag); the env calls it once
#                       at construction and again on each reconnect attempt.
#   * send(obj)      -> append `dict(obj)` to a `.sent` list. `obj` is the wire
#                       dict from a ResetMsg/StepMsg/CloseMsg `.to_dict()`.
#   * recv()         -> pop and return the NEXT scripted inbound dataclass, in the
#                       exact order the protocol expects:
#                         reset():  ResetAckMsg, then (if ok) StateMsg
#                         step():   StateMsg
#                       Return real `bridge.messages.StateMsg` / `ResetAckMsg`
#                       instances (build them with `.from_dict(...)` off a wire
#                       dict, or construct directly).
#   * close()        -> no-op (or flip `is_open`).
#
# To simulate a disconnect, have `recv()` (or `send()`) raise `BridgeError`. The
# env catches it, attempts ONE reconnect via `connect()`, and re-raises a clear
# BridgeError if recovery is impossible (recv) or after the retried send fails.
#
# `tests/test_mc_pvp_env.py` ships a reference `ScriptedBridge` implementing
# exactly this contract; copy it for T10/T20.
# ===========================================================================
