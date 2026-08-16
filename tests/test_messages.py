"""Tests for the frozen bridge wire contract (T3 / AC1).

These tests are the **freeze guard** for the Node<->Python message schema. They
assert (AC1: importable, round-trips a validation test, altering requires a PR):

  - the module imports and the canonical message list is frozen,
  - every outbound message round-trips dict <-> dataclass and produces a valid,
    newline-terminated JSON line,
  - ``parse_line`` decodes and dispatches inbound ``state`` / ``reset_ack``,
  - ``validate`` rejects a bad action (8, -1), a missing required field, and a
    wrong field type,
  - a full sample ``state`` message validates,
  - schema.json (the canonical machine-readable contract) stays consistent with
    this module's discriminators.

The optional fields (``step.opp_action``, ``state.opp_action_executed``) are
covered here in BOTH forms on purpose. The contract exists in three
mutually-consistent forms and ``schema.json`` is the canonical one; a field that
reaches the prose and the Python but not the JSON is rejected on the wire
precisely when something first tries to use it, which no test of the Python
shape alone would catch. The ``test_schema_json_*`` tests below read the real
file so removing a property there fails a test by name.
"""

import json
from pathlib import Path

import pytest

from bridge import messages as msg
from bridge.messages import (
    ACTION_MAX,
    ACTION_MIN,
    INBOUND_TYPES,
    MESSAGE_TYPES,
    OUTBOUND_TYPES,
    Arena,
    CloseMsg,
    Events,
    OpponentState,
    ResetAckMsg,
    ResetMsg,
    SchemaError,
    SelfState,
    StateMsg,
    StepMsg,
    from_dict,
    parse_line,
    validate,
)


# ---------------------------------------------------------------------------
# Sample message builders (the canonical valid shapes).
# ---------------------------------------------------------------------------


def _sample_state_dict():
    return {
        "type": "state",
        "self": {
            "pos": [0.5, 64.0, 0.5],
            "yaw": 0.0,
            "pitch": 0.0,
            "velocity": [0.0, 0.0, 0.0],
            "on_ground": True,
            "health": 20.0,
            "held_item": "iron_sword",
            "attack_cooldown": 1.0,
        },
        "opponent": {
            "pos": [3.5, 64.0, 1.5],
            "yaw": 3.14,
            "pitch": -0.2,
            "velocity": [0.1, 0.0, -0.1],
            "health": 18.0,  # PRIVILEGED true health (reward only, never obs).
        },
        "events": {
            "damage_dealt": 2.0,
            "damage_taken": 0.0,
            "i_died": False,
            "opponent_died": False,
        },
        "arena": {"wall_distances": [8.0, 8.0, 8.0, 8.0]},
        "tick": 12345,
        "code_version": "abc123",
    }


def _sample_reset_ack_dict():
    return {
        "type": "reset_ack",
        "ok": True,
        "readback": {"self_hp": 20.0, "opp_hp": 20.0},
    }


# ---------------------------------------------------------------------------
# AC1: importable + frozen message list.
# ---------------------------------------------------------------------------


def test_module_importable_and_message_list_frozen():
    assert MESSAGE_TYPES == ("reset", "step", "close", "state", "reset_ack")
    assert OUTBOUND_TYPES == ("reset", "step", "close")
    assert INBOUND_TYPES == ("state", "reset_ack")
    assert (ACTION_MIN, ACTION_MAX) == (0, 7)


# ---------------------------------------------------------------------------
# Outbound round-trips: dataclass -> dict -> validate, and JSON-line framing.
# ---------------------------------------------------------------------------


def test_reset_roundtrips_and_validates():
    m = ResetMsg(episode=7, seed=12345)
    d = m.to_dict()
    assert d == {"type": "reset", "episode": 7, "seed": 12345}
    validate(d)  # must not raise

    line = m.to_json_line()
    assert line.endswith("\n")
    assert "\n" not in line[:-1]  # exactly one trailing newline, none inside
    assert json.loads(line) == d


def test_step_roundtrips_and_validates():
    m = StepMsg(action=3)
    d = m.to_dict()
    assert d == {"type": "step", "action": 3}
    validate(d)

    line = m.to_json_line()
    assert line.endswith("\n")
    assert json.loads(line) == d


def test_close_roundtrips_and_validates():
    m = CloseMsg()
    d = m.to_dict()
    assert d == {"type": "close"}
    validate(d)

    line = m.to_json_line()
    assert line.endswith("\n")
    assert json.loads(line) == d


@pytest.mark.parametrize("action", list(range(ACTION_MIN, ACTION_MAX + 1)))
def test_step_accepts_every_valid_action(action):
    validate(StepMsg(action=action).to_dict())


# ---------------------------------------------------------------------------
# step.opp_action — the optional opponent-acts field.
# ---------------------------------------------------------------------------


def test_step_without_opp_action_is_byte_identical_to_before_the_field():
    """The M1/M2 dummy path must not gain a key. Absent, not an explicit null."""
    m = StepMsg(action=3)
    assert m.opp_action is None
    assert m.to_dict() == {"type": "step", "action": 3}
    assert "opp_action" not in m.to_dict()
    assert m.to_json_line() == '{"type":"step","action":3}\n'


def test_step_with_opp_action_carries_it_on_the_wire():
    m = StepMsg(action=3, opp_action=5)
    d = m.to_dict()
    assert d == {"type": "step", "action": 3, "opp_action": 5}
    validate(d)
    assert json.loads(m.to_json_line()) == d


def test_step_with_explicit_none_opp_action_omits_the_key():
    """`opp_action=None` means "no opponent action", not "macro 0"."""
    assert StepMsg(action=1, opp_action=None).to_dict() == {"type": "step", "action": 1}


@pytest.mark.parametrize("opp_action", list(range(ACTION_MIN, ACTION_MAX + 1)))
def test_step_accepts_every_valid_opp_action(opp_action):
    validate(StepMsg(action=0, opp_action=opp_action).to_dict())


def test_validate_accepts_explicit_null_opp_action_on_the_wire():
    """A producer may spell "no opponent action" as an explicit null."""
    validate({"type": "step", "action": 2, "opp_action": None})


@pytest.mark.parametrize("bad_opp_action", [8, -1, 100, -100])
def test_validate_rejects_out_of_range_opp_action(bad_opp_action):
    """N_ACTIONS is FROZEN at 8: opp_action widens WHO acts, not the space."""
    with pytest.raises(SchemaError):
        validate({"type": "step", "action": 0, "opp_action": bad_opp_action})


def test_validate_rejects_non_integer_opp_action():
    with pytest.raises(SchemaError):
        validate({"type": "step", "action": 0, "opp_action": 3.5})
    with pytest.raises(SchemaError):
        validate({"type": "step", "action": 0, "opp_action": "3"})
    # bool is a subclass of int in Python but is NOT a valid action index.
    with pytest.raises(SchemaError):
        validate({"type": "step", "action": 0, "opp_action": True})


def test_validate_still_rejects_other_extra_step_fields():
    """Allowing one optional key must not open the message to arbitrary ones."""
    with pytest.raises(SchemaError):
        validate({"type": "step", "action": 1, "opp_action": 2, "extra": 7})


# ---------------------------------------------------------------------------
# state.opp_action_executed — the optional swing report (feeds the shadow
# cooldown in MCPvPEnv.raw_opponent_view()).
# ---------------------------------------------------------------------------


def test_state_without_swing_report_roundtrips_unchanged():
    original = _sample_state_dict()
    parsed = StateMsg.from_dict(original)
    assert parsed.opp_action_executed is None
    assert parsed.to_dict() == original  # no key materialized out of nowhere


@pytest.mark.parametrize("executed", [True, False])
def test_state_carries_the_swing_report_both_ways(executed):
    d = _sample_state_dict()
    d["opp_action_executed"] = executed
    validate(d)
    parsed = StateMsg.from_dict(d)
    assert parsed.opp_action_executed is executed
    assert parsed.to_dict() == d


def test_state_swing_report_none_is_distinct_from_false():
    """Tri-state: "not reported" must not collapse into "did not fire"."""
    unreported = StateMsg.from_dict(_sample_state_dict())
    reported_false = StateMsg.from_dict(
        {**_sample_state_dict(), "opp_action_executed": False}
    )
    assert unreported.opp_action_executed is None
    assert reported_false.opp_action_executed is False


def test_validate_accepts_explicit_null_swing_report():
    validate({**_sample_state_dict(), "opp_action_executed": None})


@pytest.mark.parametrize("bad", [1, 0, "true", [], {}])
def test_validate_rejects_non_boolean_swing_report(bad):
    with pytest.raises(SchemaError):
        validate({**_sample_state_dict(), "opp_action_executed": bad})


def test_validate_still_rejects_other_extra_state_fields():
    with pytest.raises(SchemaError):
        validate({**_sample_state_dict(), "opp_action_executed": True, "extra": 1})


# ---------------------------------------------------------------------------
# Inbound parse_line dispatch.
# ---------------------------------------------------------------------------


def test_parse_line_dispatches_state():
    line = json.dumps(_sample_state_dict()) + "\n"
    parsed = parse_line(line)
    assert isinstance(parsed, StateMsg)
    assert parsed.tick == 12345
    assert parsed.code_version == "abc123"
    assert parsed.self_state.health == pytest.approx(20.0)
    assert parsed.self_state.held_item == "iron_sword"
    # PRIVILEGED field is parsed (it is on the wire) but is reward-only downstream.
    assert parsed.opponent.health == pytest.approx(18.0)
    assert parsed.events.damage_dealt == pytest.approx(2.0)
    assert parsed.arena.wall_distances == [8.0, 8.0, 8.0, 8.0]


def test_parse_line_dispatches_reset_ack():
    line = json.dumps(_sample_reset_ack_dict()) + "\n"
    parsed = parse_line(line)
    assert isinstance(parsed, ResetAckMsg)
    assert parsed.ok is True
    assert parsed.readback == {"self_hp": 20.0, "opp_hp": 20.0}


def test_reset_ack_ok_false_signals_readback_timeout():
    d = {"type": "reset_ack", "ok": False, "readback": {}}
    validate(d)
    parsed = from_dict(d)
    assert isinstance(parsed, ResetAckMsg)
    assert parsed.ok is False


def test_parse_line_tolerates_no_trailing_newline():
    line = json.dumps(_sample_reset_ack_dict())  # no "\n"
    assert isinstance(parse_line(line), ResetAckMsg)


def test_parse_line_rejects_invalid_json():
    with pytest.raises(SchemaError):
        parse_line("{not valid json")


def test_parse_line_rejects_empty_line():
    with pytest.raises(SchemaError):
        parse_line("   \n")


def test_from_dict_rejects_outbound_type():
    """Outbound types are not built by the inbound parser."""
    with pytest.raises(SchemaError):
        from_dict({"type": "step", "action": 1})


# ---------------------------------------------------------------------------
# Full inbound round-trips: dataclass -> dict -> dataclass.
# ---------------------------------------------------------------------------


def test_state_dataclass_roundtrips():
    original = _sample_state_dict()
    rebuilt = StateMsg.from_dict(original).to_dict()
    assert rebuilt == original
    validate(rebuilt)


def test_reset_ack_dataclass_roundtrips():
    original = _sample_reset_ack_dict()
    rebuilt = ResetAckMsg.from_dict(original).to_dict()
    assert rebuilt == original
    validate(rebuilt)


def test_state_msg_to_json_line_validates():
    state = StateMsg(
        self_state=SelfState(
            pos=[0.0, 64.0, 0.0],
            yaw=0.0,
            pitch=0.0,
            velocity=[0.0, 0.0, 0.0],
            on_ground=True,
            health=20.0,
            held_item="diamond_sword",
            attack_cooldown=0.5,
        ),
        opponent=OpponentState(
            pos=[1.0, 64.0, 1.0],
            yaw=0.0,
            pitch=0.0,
            velocity=[0.0, 0.0, 0.0],
            health=20.0,
        ),
        events=Events(damage_dealt=0.0, damage_taken=0.0, i_died=False, opponent_died=False),
        arena=Arena(wall_distances=[5.0, 5.0]),
        tick=1,
        code_version="v1",
    )
    line = state.to_json_line()
    assert line.endswith("\n")
    validate(json.loads(line))


# ---------------------------------------------------------------------------
# validate(): a full sample state validates.
# ---------------------------------------------------------------------------


def test_full_sample_state_validates():
    validate(_sample_state_dict())


# ---------------------------------------------------------------------------
# validate(): rejects bad action, missing field, wrong type.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_action", [8, -1, 100, -100])
def test_validate_rejects_out_of_range_action(bad_action):
    with pytest.raises(SchemaError):
        validate({"type": "step", "action": bad_action})


def test_validate_rejects_non_integer_action():
    with pytest.raises(SchemaError):
        validate({"type": "step", "action": 3.5})
    with pytest.raises(SchemaError):
        validate({"type": "step", "action": "3"})
    # bool is a subclass of int in Python but is NOT a valid action.
    with pytest.raises(SchemaError):
        validate({"type": "step", "action": True})


def test_validate_rejects_missing_required_field():
    # reset without 'seed'.
    with pytest.raises(SchemaError):
        validate({"type": "reset", "episode": 0})
    # step without 'action'.
    with pytest.raises(SchemaError):
        validate({"type": "step"})
    # state missing 'opponent'.
    bad = _sample_state_dict()
    del bad["opponent"]
    with pytest.raises(SchemaError):
        validate(bad)


def test_validate_rejects_wrong_type():
    # episode as a string.
    with pytest.raises(SchemaError):
        validate({"type": "reset", "episode": "0", "seed": 1})
    # self.health as a string.
    bad = _sample_state_dict()
    bad["self"]["health"] = "full"
    with pytest.raises(SchemaError):
        validate(bad)
    # on_ground as an int instead of bool.
    bad = _sample_state_dict()
    bad["self"]["on_ground"] = 1
    with pytest.raises(SchemaError):
        validate(bad)


def test_validate_rejects_unexpected_field():
    """additionalProperties: false — extra fields are a contract violation."""
    with pytest.raises(SchemaError):
        validate({"type": "step", "action": 1, "extra": 7})


def test_validate_rejects_unknown_type():
    with pytest.raises(SchemaError):
        validate({"type": "teleport"})


def test_validate_rejects_non_object():
    with pytest.raises(SchemaError):
        validate("not a dict")
    with pytest.raises(SchemaError):
        validate([1, 2, 3])


def test_validate_rejects_bad_vec3():
    bad = _sample_state_dict()
    bad["self"]["pos"] = [0.0, 64.0]  # only 2 components
    with pytest.raises(SchemaError):
        validate(bad)


def test_validate_rejects_negative_damage():
    bad = _sample_state_dict()
    bad["events"]["damage_dealt"] = -1.0
    with pytest.raises(SchemaError):
        validate(bad)


# ---------------------------------------------------------------------------
# Consistency with the canonical machine-readable schema.json.
# ---------------------------------------------------------------------------


SCHEMA_PATH = Path(__file__).resolve().parents[1] / "bridge" / "schema.json"


def _schema_branch(mtype):
    """The ``oneOf`` branch of the REAL schema.json file for one message type."""
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    return next(
        b for b in schema["oneOf"] if b["properties"]["type"]["const"] == mtype
    )


def _fully_populated(mtype):
    """A valid message of ``mtype`` carrying EVERY field, optional ones included.

    The cross-form tests below use these to check both directions at once: this
    module must accept every declared property, and must not accept any field
    schema.json does not declare.
    """
    return {
        "reset": {"type": "reset", "episode": 0, "seed": 12345},
        "step": {"type": "step", "action": 3, "opp_action": 5},
        "close": {"type": "close"},
        "state": {**_sample_state_dict(), "opp_action_executed": True},
        "reset_ack": _sample_reset_ack_dict(),
    }[mtype]


def test_schema_json_discriminators_match_module():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    consts = {branch["properties"]["type"]["const"] for branch in schema["oneOf"]}
    assert consts == set(MESSAGE_TYPES)


def test_schema_json_action_bounds_match_module():
    action = _schema_branch("step")["properties"]["action"]
    assert action["minimum"] == ACTION_MIN
    assert action["maximum"] == ACTION_MAX


def test_schema_json_declares_optional_opp_action():
    """The canonical form must carry opp_action — schema.json is what the wire enforces.

    Both validators are ``additionalProperties: false``. A field that reaches
    only the prose and ``messages.py`` is rejected on the wire *precisely* when
    something first sends it, so this asserts against the real file rather than
    against the Python shape.
    """
    step = _schema_branch("step")
    assert step["additionalProperties"] is False
    assert "opp_action" in step["properties"], (
        "step.opp_action is missing from bridge/schema.json — the canonical "
        "form of the contract. Adding it to schema.md and messages.py only "
        "leaves it rejected on the wire the moment it is used."
    )
    opp_action = step["properties"]["opp_action"]
    # Nullable: absent and null both mean "the opponent takes no action".
    assert opp_action["type"] == ["integer", "null"]
    # Same frozen range as `action` — N_ACTIONS stays 8.
    assert opp_action["minimum"] == ACTION_MIN
    assert opp_action["maximum"] == ACTION_MAX
    # Optional: a step without it is still valid (the M1/M2 dummy path).
    assert "opp_action" not in step["required"]
    assert step["required"] == ["type", "action"]


def test_schema_json_declares_optional_swing_report():
    state = _schema_branch("state")
    assert state["additionalProperties"] is False
    assert "opp_action_executed" in state["properties"], (
        "state.opp_action_executed is missing from bridge/schema.json — the "
        "bridge could not report the opponent's swing, and the shadow attack "
        "cooldown has no other source."
    )
    assert state["properties"]["opp_action_executed"]["type"] == ["boolean", "null"]
    assert "opp_action_executed" not in state["required"]


@pytest.mark.parametrize("mtype", MESSAGE_TYPES)
def test_schema_json_declares_every_field_this_module_accepts(mtype):
    """No field may exist in the Python form without existing in the canonical one."""
    message = _fully_populated(mtype)
    validate(message)  # this module accepts the fully-populated form...
    declared = set(_schema_branch(mtype)["properties"])
    undeclared = set(message) - declared
    assert not undeclared, (
        f"{mtype} accepts field(s) {sorted(undeclared)} that bridge/schema.json "
        "does not declare; additionalProperties:false means the wire rejects them"
    )
    assert _schema_branch(mtype)["additionalProperties"] is False


@pytest.mark.parametrize("mtype", MESSAGE_TYPES)
def test_schema_json_required_fields_are_required_by_this_module(mtype):
    for key in _schema_branch(mtype)["required"]:
        if key == "type":
            continue  # dropping the discriminator is a different failure mode
        incomplete = {k: v for k, v in _fully_populated(mtype).items() if k != key}
        with pytest.raises(SchemaError):
            validate(incomplete)


@pytest.mark.parametrize("mtype", MESSAGE_TYPES)
def test_schema_json_optional_fields_are_optional_in_this_module(mtype):
    branch = _schema_branch(mtype)
    optional = set(branch["properties"]) - set(branch["required"])
    for key in optional:
        without = {k: v for k, v in _fully_populated(mtype).items() if k != key}
        validate(without)  # must not raise
