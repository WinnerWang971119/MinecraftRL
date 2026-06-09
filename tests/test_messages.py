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


def test_schema_json_discriminators_match_module():
    schema_path = Path(__file__).resolve().parents[1] / "bridge" / "schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    consts = {branch["properties"]["type"]["const"] for branch in schema["oneOf"]}
    assert consts == set(MESSAGE_TYPES)


def test_schema_json_action_bounds_match_module():
    schema_path = Path(__file__).resolve().parents[1] / "bridge" / "schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    step = next(
        b for b in schema["oneOf"] if b["properties"]["type"]["const"] == "step"
    )
    action = step["properties"]["action"]
    assert action["minimum"] == ACTION_MIN
    assert action["maximum"] == ACTION_MAX
