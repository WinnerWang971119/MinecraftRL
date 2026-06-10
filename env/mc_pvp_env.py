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
    Python -> ``step(action)``
    Node   -> ``state(...)``                # raw aggregated state for the interval

    The raw state is gated by the PerceptionFilter, packed by
    ``build_observation``, validated, and scored by ``compute_reward`` against
    the previous observation. ``done`` fires on a death event or at
    ``MAX_EPISODE_STEPS`` (timeout).

Privileged-data discipline (spec §5): opponent raw ``pos`` / ``yaw`` / ``vel``
reach the obs ONLY through the gating filter; opponent ``health`` never reaches
the obs at all (the obs has no slot for it). The reward consumes the privileged
``events`` block, never raw opponent health here.

Owner: T9 (Environment/bridge track)
"""

from __future__ import annotations

import json
import math
import socket
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
from env.perception_filter import PerceptionFilter
from env.reward import (
    TermInfo,
    compute_reward,
    compute_reward_components,
)

__all__ = [
    "BridgeError",
    "BridgeTransport",
    "TcpBridgeClient",
    "MCPvPEnv",
    "DECISION_DT_SECONDS",
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
            the frozen :data:`MAX_EPISODE_STEPS`.
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

    def __init__(
        self,
        transport: BridgeTransport,
        reward_config: Optional[RewardConfig] = None,
        perception_filter: Optional[PerceptionFilter] = None,
        max_episode_steps: int = MAX_EPISODE_STEPS,
        dt: float = DECISION_DT_SECONDS,
        auto_connect: bool = True,
    ) -> None:
        if max_episode_steps <= 0:
            raise ValueError(
                f"max_episode_steps must be > 0, got {max_episode_steps}"
            )
        if dt < 0.0:
            raise ValueError(f"dt must be >= 0, got {dt}")

        self._transport = transport
        self._cfg = reward_config if reward_config is not None else RewardConfig()
        self._filter = (
            perception_filter if perception_filter is not None else PerceptionFilter()
        )
        self._max_steps = int(max_episode_steps)
        self._dt = float(dt)

        # Per-episode mutable state (initialized by reset()).
        self._episode: int = -1
        self._step_count: int = 0
        self._prev_obs: Optional[np.ndarray] = None
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

    # -- lifecycle ---------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Start a new episode and return its initial observation vector.

        Sends a ``reset`` and awaits the ``reset_ack`` read-back gate. If the gate
        reports ``ok == False`` the world is unverified, so the reset is RETRIED
        ONCE; a second ``ok == False`` RAISES :class:`BridgeError` rather than
        starting an episode from a corrupt state (protects the MDP / AC7). On
        success the PerceptionFilter memory and step counter are reset and the
        post-reset ``state`` message is gated/packed into the initial observation.

        Args:
            seed: Per-episode RNG seed forwarded to the bridge (spawn jitter, gear,
                opponent choice). ``None`` becomes ``0`` on the wire (the schema
                requires an integer seed).

        Returns:
            The initial observation ``np.ndarray`` of shape ``(OBS_DIM,)``.

        Raises:
            BridgeError: if the bridge disconnects, sends an out-of-order/invalid
                message, or fails the read-back gate twice.
        """
        self._episode += 1
        wire_seed = 0 if seed is None else int(seed)

        # Read-back gate with a single retry. Each attempt is a full
        # reset -> reset_ack exchange; on ok==False we try once more, then raise.
        ack: Optional[ResetAckMsg] = None
        last_readback: Dict[str, Any] = {}
        for attempt in range(2):  # at most one retry
            ack = self._reset_exchange(wire_seed)
            last_readback = dict(ack.readback)
            if ack.ok:
                break
        if ack is None or not ack.ok:
            raise BridgeError(
                "bridge reset read-back gate failed twice for "
                f"episode {self._episode} (seed={wire_seed}); refusing to start "
                f"an episode from an unverified state. last readback={last_readback!r}"
            )

        # Gate confirmed: clear per-episode memory BEFORE consuming the first
        # state so the opponent starts ABSENT (no stale memory across episodes).
        self._filter.reset()
        self._step_count = 0
        self._done = False

        # The post-reset first observation is the next inbound `state` message.
        state = self._recv_state()
        obs = self._state_to_obs(state)
        self._prev_obs = obs
        return obs

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Advance one decision step.

        Sends a ``step(action)``, awaits one ``state`` reply, gates + packs it
        into the next observation, validates it, and scores the transition with
        :func:`env.reward.compute_reward`.

        ``done`` fires when the bridge reports a death (``events.i_died`` →
        terminal loss, ``events.opponent_died`` → terminal win) OR when the step
        count reaches ``max_episode_steps`` (timeout / draw). The three terminal
        outcomes are mutually exclusive; a same-step double-death is resolved as a
        loss (the learner dying is never a win).

        Args:
            action: Discrete action index (or :class:`~agent.actions.Macro`) in
                ``[0, n_actions)``.

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
        if self._prev_obs is None:
            # Defensive: reset() always sets _prev_obs before clearing _done.
            raise ValueError("internal error: no previous observation; call reset()")

        # Send the action and await exactly one state reply.
        self._send(StepMsg(action=act))
        state = self._recv_state()

        # Gate + pack the new observation (s').
        obs = self._state_to_obs(state)
        prev_obs = self._prev_obs

        # --- termination ---------------------------------------------------
        events = state.events
        self._step_count += 1

        died = bool(events.i_died)
        opp_died = bool(events.opponent_died)
        timed_out = self._step_count >= self._max_steps

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
        """Send one ``reset`` and await the matching ``reset_ack`` dataclass."""
        self._send(ResetMsg(episode=self._episode, seed=seed))
        msg = self._recv()
        if not isinstance(msg, ResetAckMsg):
            raise BridgeError(
                f"expected a reset_ack after reset, got {type(msg).__name__}"
            )
        return msg

    def _recv_state(self) -> StateMsg:
        """Await one inbound message and require it to be a ``state``."""
        msg = self._recv()
        if not isinstance(msg, StateMsg):
            raise BridgeError(
                f"expected a state message, got {type(msg).__name__}"
            )
        return msg

    def _send(self, message: Union[ResetMsg, StepMsg, CloseMsg]) -> None:
        """Send an outbound dataclass, reconnecting once on a transport failure."""
        try:
            self._transport.send(message.to_dict())
        except BridgeError:
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

        NOTE: a reconnect cannot recover the in-flight reply that was lost with
        the old socket, so after a successful reconnect this still raises — the
        caller (reset/step) aborts the run loudly rather than silently desyncing
        the request/response stream. The reconnect exists so the NEXT episode can
        proceed and so the failure carries a clear, actionable message.
        """
        try:
            return self._transport.recv()
        except BridgeError as first_exc:
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
