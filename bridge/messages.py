"""messages — Python side of the Node<->Python bridge wire contract.

This module is **one of the four FROZEN CONTRACT artifacts (AC1)**. It is the
Python binding for the bridge wire schema: frozen dataclasses for every message
type, JSON-lines (de)serialization, a ``type``-dispatched inbound parser, and a
**dependency-free** validator that mirrors ``bridge/schema.json``.

The contract has three mutually-consistent forms — change one, change all:

    bridge/schema.md    human-readable doc
    bridge/schema.json  canonical machine-readable JSON Schema (draft-07)
    bridge/messages.py  this module (Python bindings + validator)

Altering any of them is a contract change and requires a PR visible to all
tracks. ``mc_pvp_env.py`` imports this module to parse inbound ``state`` /
``reset_ack`` messages and to serialize outbound ``reset`` / ``step`` / ``close``
commands.

Transport (documented here, enforced by the Node transport T7a)
---------------------------------------------------------------
Messages are **newline-delimited JSON** over a raw TCP socket, **one connection
per arena**. Each message is one UTF-8 JSON object terminated by ``"\n"``. TCP is
a byte stream, so the reader MUST buffer partial reads across packet boundaries
and only parse complete lines — that framing is the Node transport's job. The
helpers here operate on already-framed single lines (:func:`parse_line`) and
append the trailing newline on send (:meth:`OutboundMessage.to_json_line`).

PRIVILEGED INFORMATION (read this)
----------------------------------
``state.opponent.health`` is the opponent's **RAW true health**. It is on the
wire and is parsed into :class:`OpponentState` here, but it is **privileged**:
the reward may read it, but it **MUST NEVER reach the observation** (the
PerceptionFilter, T12, gates the opponent block, and ``observation_spec`` has no
opponent-health slot at all). Downstream code must route ``opponent.health``
**only to the reward**, never to the observation builder.

Owner: T3 (Environment/bridge track)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Tuple

__all__ = [
    "SchemaError",
    "ACTION_MIN",
    "ACTION_MAX",
    "MESSAGE_TYPES",
    "OUTBOUND_TYPES",
    "INBOUND_TYPES",
    # Outbound (Python -> Node)
    "ResetMsg",
    "StepMsg",
    "CloseMsg",
    "OutboundMessage",
    # Inbound (Node -> Python)
    "SelfState",
    "OpponentState",
    "Events",
    "Arena",
    "StateMsg",
    "ResetAckMsg",
    "InboundMessage",
    # Helpers
    "validate",
    "parse_line",
    "from_dict",
]


# ---------------------------------------------------------------------------
# Error type and frozen constants.
# ---------------------------------------------------------------------------


class SchemaError(ValueError):
    """Raised when a message dict violates the bridge wire contract.

    A ``ValueError`` subclass so callers can catch either. Carries a clear,
    field-level message describing the violation.
    """


#: Inclusive bounds of the discrete action index. FROZEN: the 8 macros are
#: indices 0..7 (matches ``agent/actions.py``). Mirrors ``step.action`` in
#: ``schema.json`` (``minimum``/``maximum``).
ACTION_MIN: int = 0
ACTION_MAX: int = 7

#: The full set of wire ``type`` discriminators, split by direction. These are
#: the canonical message list and mirror the ``oneOf`` branches in schema.json.
OUTBOUND_TYPES: Tuple[str, ...] = ("reset", "step", "close")
INBOUND_TYPES: Tuple[str, ...] = ("state", "reset_ack")
MESSAGE_TYPES: Tuple[str, ...] = OUTBOUND_TYPES + INBOUND_TYPES


# ---------------------------------------------------------------------------
# Outbound messages: Python -> Node.
#
# Each is a frozen dataclass with a ``TYPE`` class constant, ``to_dict`` (adds
# the discriminator), and ``to_json_line`` (compact JSON + trailing "\n").
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResetMsg:
    """``reset`` — request a new episode (Python -> Node).

    Attributes:
        episode: Monotonic episode counter; lets Node correlate the reset_ack.
        seed: Per-episode RNG seed; logged for reproducibility.
    """

    TYPE = "reset"

    episode: int
    seed: int

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.TYPE, "episode": self.episode, "seed": self.seed}

    def to_json_line(self) -> str:
        """Serialize to a single newline-terminated JSON line."""
        return _to_json_line(self.to_dict())


@dataclass(frozen=True)
class StepMsg:
    """``step`` — run one action macro for the decision interval (Python -> Node).

    Attributes:
        action: Discrete action index in ``[ACTION_MIN, ACTION_MAX]`` (0..7),
            matching the 8 frozen action macros in ``agent/actions.py``.
    """

    TYPE = "step"

    action: int

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.TYPE, "action": self.action}

    def to_json_line(self) -> str:
        """Serialize to a single newline-terminated JSON line."""
        return _to_json_line(self.to_dict())


@dataclass(frozen=True)
class CloseMsg:
    """``close`` — tear down the arena and close the connection (Python -> Node)."""

    TYPE = "close"

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.TYPE}

    def to_json_line(self) -> str:
        """Serialize to a single newline-terminated JSON line."""
        return _to_json_line(self.to_dict())


#: Union alias for the outbound (Python -> Node) message dataclasses.
OutboundMessage = "ResetMsg | StepMsg | CloseMsg"  # typing alias (string form)


# ---------------------------------------------------------------------------
# Inbound messages: Node -> Python.
#
# Nested raw-state blocks are typed dataclasses so callers read fields by name.
# Vector fields (``pos``/``velocity``) are kept as plain length-3 lists of
# floats — the same world-frame triples that are on the wire.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SelfState:
    """Learner-bot RAW state (FULL; always real, never gated).

    Attributes:
        pos: ``[x, y, z]`` world-frame position.
        yaw: Yaw in radians.
        pitch: Pitch in radians.
        velocity: ``[x, y, z]`` world-frame velocity.
        on_ground: Whether the bot is on the ground.
        health: Self current health (``0..MAX_HEALTH``); allowed in the obs.
        held_item: Held-item identifier string (resolved to a vocab id by
            ``observation_spec``).
        attack_cooldown: Bridge-computed swing progress in ``[0, 1]``.
    """

    pos: List[float]
    yaw: float
    pitch: float
    velocity: List[float]
    on_ground: bool
    health: float
    held_item: str
    attack_cooldown: float

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "SelfState":
        return cls(
            pos=[float(v) for v in d["pos"]],
            yaw=float(d["yaw"]),
            pitch=float(d["pitch"]),
            velocity=[float(v) for v in d["velocity"]],
            on_ground=bool(d["on_ground"]),
            health=float(d["health"]),
            held_item=str(d["held_item"]),
            attack_cooldown=float(d["attack_cooldown"]),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pos": list(self.pos),
            "yaw": self.yaw,
            "pitch": self.pitch,
            "velocity": list(self.velocity),
            "on_ground": self.on_ground,
            "health": self.health,
            "held_item": self.held_item,
            "attack_cooldown": self.attack_cooldown,
        }


@dataclass(frozen=True)
class OpponentState:
    """Opponent RAW true state (gated upstream before reaching the obs).

    PRIVILEGED: ``health`` is the opponent's RAW true health. It is on the wire
    and parsed here, but it MUST NEVER reach the observation — route it to the
    reward only. Position/facing/velocity are also raw here and are gated
    upstream by the PerceptionFilter before any of them reach the obs.

    Attributes:
        pos: ``[x, y, z]`` world-frame position.
        yaw: Yaw in radians.
        pitch: Pitch in radians.
        velocity: ``[x, y, z]`` world-frame velocity.
        health: RAW true opponent health. PRIVILEGED -> reward only, NEVER obs.
    """

    pos: List[float]
    yaw: float
    pitch: float
    velocity: List[float]
    health: float

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "OpponentState":
        return cls(
            pos=[float(v) for v in d["pos"]],
            yaw=float(d["yaw"]),
            pitch=float(d["pitch"]),
            velocity=[float(v) for v in d["velocity"]],
            # PRIVILEGED: parsed because it is on the wire; downstream must route
            # this ONLY to the reward, never to the observation builder.
            health=float(d["health"]),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pos": list(self.pos),
            "yaw": self.yaw,
            "pitch": self.pitch,
            "velocity": list(self.velocity),
            "health": self.health,
        }


@dataclass(frozen=True)
class Events:
    """Damage/death events aggregated over the decision interval.

    Source of the reward's damage anchors (privileged and fair).

    Attributes:
        damage_dealt: Total damage dealt to the opponent this interval (>= 0).
        damage_taken: Total damage taken this interval (>= 0).
        i_died: Learner bot died this interval.
        opponent_died: Opponent died this interval.
    """

    damage_dealt: float
    damage_taken: float
    i_died: bool
    opponent_died: bool

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Events":
        return cls(
            damage_dealt=float(d["damage_dealt"]),
            damage_taken=float(d["damage_taken"]),
            i_died=bool(d["i_died"]),
            opponent_died=bool(d["opponent_died"]),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "damage_dealt": self.damage_dealt,
            "damage_taken": self.damage_taken,
            "i_died": self.i_died,
            "opponent_died": self.opponent_died,
        }


@dataclass(frozen=True)
class Arena:
    """Arena geometry sensed this interval.

    Attributes:
        wall_distances: Distances to surrounding arena walls (fixed probe order).
    """

    wall_distances: List[float]

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Arena":
        return cls(wall_distances=[float(v) for v in d["wall_distances"]])

    def to_dict(self) -> Dict[str, Any]:
        return {"wall_distances": list(self.wall_distances)}


@dataclass(frozen=True)
class StateMsg:
    """``state`` — RAW aggregated state for one decision interval (Node -> Python).

    Attributes:
        self_state: Learner-bot raw state (FULL).
        opponent: Opponent raw state (incl. PRIVILEGED true health).
        events: Aggregated damage/death events.
        arena: Arena geometry.
        tick: Server game tick at end of the interval.
        code_version: Env+filter code-version stamp. Kickoff LOGS a mismatch;
            the distributed future REJECTS it.

    Note:
        The wire field is named ``self``; it is exposed as ``self_state`` here
        because ``self`` is a reserved parameter name in Python methods.
    """

    TYPE = "state"

    self_state: SelfState
    opponent: OpponentState
    events: Events
    arena: Arena
    tick: int
    code_version: str

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "StateMsg":
        return cls(
            self_state=SelfState.from_dict(d["self"]),
            opponent=OpponentState.from_dict(d["opponent"]),
            events=Events.from_dict(d["events"]),
            arena=Arena.from_dict(d["arena"]),
            tick=int(d["tick"]),
            code_version=str(d["code_version"]),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.TYPE,
            "self": self.self_state.to_dict(),
            "opponent": self.opponent.to_dict(),
            "events": self.events.to_dict(),
            "arena": self.arena.to_dict(),
            "tick": self.tick,
            "code_version": self.code_version,
        }

    def to_json_line(self) -> str:
        """Serialize to a single newline-terminated JSON line."""
        return _to_json_line(self.to_dict())


@dataclass(frozen=True)
class ResetAckMsg:
    """``reset_ack`` — acknowledges a reset after the read-back gate (Node -> Python).

    Attributes:
        ok: True if the read-back gate confirmed; ``False`` signals a read-back
            timeout (the episode must be treated as failed-to-start).
        readback: Free-form post-reset read-back snapshot used to verify gates.
    """

    TYPE = "reset_ack"

    ok: bool
    readback: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ResetAckMsg":
        return cls(ok=bool(d["ok"]), readback=dict(d["readback"]))

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.TYPE, "ok": self.ok, "readback": dict(self.readback)}

    def to_json_line(self) -> str:
        """Serialize to a single newline-terminated JSON line."""
        return _to_json_line(self.to_dict())


#: Union alias for the inbound (Node -> Python) message dataclasses.
InboundMessage = "StateMsg | ResetAckMsg"  # typing alias (string form)


# ---------------------------------------------------------------------------
# Serialization helper.
# ---------------------------------------------------------------------------


def _to_json_line(d: Mapping[str, Any]) -> str:
    """Encode ``d`` as a compact JSON object plus a single trailing newline.

    ``separators`` strips spaces for a compact line, and ``ensure_ascii=False``
    keeps any item names readable. The trailing ``"\n"`` is the JSON-lines frame
    delimiter the Node transport splits on.
    """
    return json.dumps(d, separators=(",", ":"), ensure_ascii=False) + "\n"


# ---------------------------------------------------------------------------
# Dependency-free validator.
#
# Mirrors the rules in bridge/schema.json without importing ``jsonschema`` (a
# deliberate choice: no new runtime dependency). schema.json stays the canonical
# reference; this validator must be kept in lock-step with it.
# ---------------------------------------------------------------------------

# A "number" on the wire is an int or float, but NOT a bool (in Python ``bool``
# is a subclass of ``int``, so we exclude it explicitly).
def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _require(cond: bool, message: str) -> None:
    if not cond:
        raise SchemaError(message)


def _check_keys(d: Mapping[str, Any], required: Tuple[str, ...], where: str) -> None:
    """Require exactly ``required`` keys in ``d`` (no missing, no extras).

    Mirrors ``required`` + ``additionalProperties: false`` from schema.json.
    """
    _require(isinstance(d, Mapping), f"{where} must be an object, got {type(d).__name__}")
    keys = set(d.keys())
    req = set(required)
    missing = req - keys
    _require(not missing, f"{where} missing required field(s): {sorted(missing)}")
    extra = keys - req
    _require(not extra, f"{where} has unexpected field(s): {sorted(extra)}")


def _check_vec3(v: Any, where: str) -> None:
    _require(
        isinstance(v, (list, tuple)),
        f"{where} must be a [x,y,z] array, got {type(v).__name__}",
    )
    _require(len(v) == 3, f"{where} must have exactly 3 elements, got {len(v)}")
    for i, component in enumerate(v):
        _require(_is_number(component), f"{where}[{i}] must be a number")


def _validate_reset(d: Mapping[str, Any]) -> None:
    _check_keys(d, ("type", "episode", "seed"), "reset")
    _require(_is_int(d["episode"]), "reset.episode must be an integer")
    _require(d["episode"] >= 0, "reset.episode must be >= 0")
    _require(_is_int(d["seed"]), "reset.seed must be an integer")


def _validate_step(d: Mapping[str, Any]) -> None:
    _check_keys(d, ("type", "action"), "step")
    action = d["action"]
    _require(_is_int(action), "step.action must be an integer")
    _require(
        ACTION_MIN <= action <= ACTION_MAX,
        f"step.action must be in [{ACTION_MIN}, {ACTION_MAX}], got {action}",
    )


def _validate_close(d: Mapping[str, Any]) -> None:
    _check_keys(d, ("type",), "close")


def _validate_self(d: Any) -> None:
    _check_keys(
        d,
        ("pos", "yaw", "pitch", "velocity", "on_ground", "health", "held_item", "attack_cooldown"),
        "state.self",
    )
    _check_vec3(d["pos"], "state.self.pos")
    _require(_is_number(d["yaw"]), "state.self.yaw must be a number")
    _require(_is_number(d["pitch"]), "state.self.pitch must be a number")
    _check_vec3(d["velocity"], "state.self.velocity")
    _require(isinstance(d["on_ground"], bool), "state.self.on_ground must be a boolean")
    _require(_is_number(d["health"]), "state.self.health must be a number")
    _require(isinstance(d["held_item"], str), "state.self.held_item must be a string")
    _require(
        _is_number(d["attack_cooldown"]), "state.self.attack_cooldown must be a number"
    )


def _validate_opponent(d: Any) -> None:
    _check_keys(d, ("pos", "yaw", "pitch", "velocity", "health"), "state.opponent")
    _check_vec3(d["pos"], "state.opponent.pos")
    _require(_is_number(d["yaw"]), "state.opponent.yaw must be a number")
    _require(_is_number(d["pitch"]), "state.opponent.pitch must be a number")
    _check_vec3(d["velocity"], "state.opponent.velocity")
    # PRIVILEGED: validated as present/number, but it is reward-only downstream.
    _require(_is_number(d["health"]), "state.opponent.health must be a number")


def _validate_events(d: Any) -> None:
    _check_keys(
        d, ("damage_dealt", "damage_taken", "i_died", "opponent_died"), "state.events"
    )
    _require(_is_number(d["damage_dealt"]), "state.events.damage_dealt must be a number")
    _require(d["damage_dealt"] >= 0, "state.events.damage_dealt must be >= 0")
    _require(_is_number(d["damage_taken"]), "state.events.damage_taken must be a number")
    _require(d["damage_taken"] >= 0, "state.events.damage_taken must be >= 0")
    _require(isinstance(d["i_died"], bool), "state.events.i_died must be a boolean")
    _require(
        isinstance(d["opponent_died"], bool), "state.events.opponent_died must be a boolean"
    )


def _validate_arena(d: Any) -> None:
    _check_keys(d, ("wall_distances",), "state.arena")
    wd = d["wall_distances"]
    _require(isinstance(wd, list), "state.arena.wall_distances must be an array")
    for i, dist in enumerate(wd):
        _require(_is_number(dist), f"state.arena.wall_distances[{i}] must be a number")


def _validate_state(d: Mapping[str, Any]) -> None:
    _check_keys(
        d,
        ("type", "self", "opponent", "events", "arena", "tick", "code_version"),
        "state",
    )
    _validate_self(d["self"])
    _validate_opponent(d["opponent"])
    _validate_events(d["events"])
    _validate_arena(d["arena"])
    _require(_is_int(d["tick"]), "state.tick must be an integer")
    _require(d["tick"] >= 0, "state.tick must be >= 0")
    _require(isinstance(d["code_version"], str), "state.code_version must be a string")


def _validate_reset_ack(d: Mapping[str, Any]) -> None:
    _check_keys(d, ("type", "ok", "readback"), "reset_ack")
    _require(isinstance(d["ok"], bool), "reset_ack.ok must be a boolean")
    _require(isinstance(d["readback"], Mapping), "reset_ack.readback must be an object")


#: Dispatch table: wire ``type`` -> validator. Mirrors the ``oneOf`` in
#: schema.json (keyed on the ``type`` discriminator).
_VALIDATORS = {
    "reset": _validate_reset,
    "step": _validate_step,
    "close": _validate_close,
    "state": _validate_state,
    "reset_ack": _validate_reset_ack,
}


def validate(msg: Mapping[str, Any]) -> None:
    """Validate a message dict against the bridge wire contract. **Raises** on invalid.

    Dependency-free mirror of ``bridge/schema.json``: dispatches on the ``type``
    discriminator and enforces required fields, field types, and value bounds
    (notably ``step.action`` in ``[0, 7]``). Returns ``None`` on success.

    Args:
        msg: The candidate message as a plain dict.

    Raises:
        SchemaError: if ``msg`` is not a valid message of any known type.
    """
    _require(isinstance(msg, Mapping), f"message must be an object, got {type(msg).__name__}")
    _require("type" in msg, "message missing required 'type' discriminator")
    mtype = msg["type"]
    _require(isinstance(mtype, str), "message 'type' must be a string")
    validator = _VALIDATORS.get(mtype)
    _require(
        validator is not None,
        f"unknown message type {mtype!r}; expected one of {list(_VALIDATORS)}",
    )
    validator(msg)


# ---------------------------------------------------------------------------
# Inbound parsing: dispatch on ``type``.
# ---------------------------------------------------------------------------

#: Dispatch table: inbound wire ``type`` -> dataclass factory.
_INBOUND_FACTORIES = {
    "state": StateMsg.from_dict,
    "reset_ack": ResetAckMsg.from_dict,
}


def from_dict(msg: Mapping[str, Any]):
    """Validate ``msg`` and build the matching inbound dataclass.

    Dispatches on the ``type`` discriminator. Only inbound (Node -> Python)
    message types — ``state`` and ``reset_ack`` — produce dataclasses here;
    outbound types are constructed directly by the caller.

    Args:
        msg: A decoded inbound message dict.

    Returns:
        A :class:`StateMsg` or :class:`ResetAckMsg`.

    Raises:
        SchemaError: if the message is invalid or is not an inbound type.
    """
    validate(msg)
    mtype = msg["type"]
    factory = _INBOUND_FACTORIES.get(mtype)
    _require(
        factory is not None,
        f"{mtype!r} is not an inbound (Node -> Python) message; "
        f"expected one of {list(_INBOUND_FACTORIES)}",
    )
    return factory(msg)


def parse_line(line: str):
    """Parse one already-framed JSON line into an inbound message dataclass.

    The Node transport (T7a) is responsible for splitting the TCP byte stream on
    ``"\n"`` and handing whole lines here; this function decodes one such line,
    validates it, and dispatches on ``type``.

    Args:
        line: A single message as a JSON string (a trailing newline is tolerated).

    Returns:
        A :class:`StateMsg` or :class:`ResetAckMsg`.

    Raises:
        SchemaError: if the line is not valid JSON or not a valid inbound message.
    """
    text = line.strip()
    _require(text != "", "cannot parse an empty line")
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SchemaError(f"invalid JSON line: {exc}") from exc
    _require(
        isinstance(decoded, Mapping),
        f"a message line must decode to an object, got {type(decoded).__name__}",
    )
    return from_dict(decoded)
