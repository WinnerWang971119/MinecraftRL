"""Tests for the frozen action macro contract (T4 / AC1).

These tests are the freeze guard for agent/actions.py.  They assert:
  - the enum has exactly 8 members with the exact names and integer values 0..7,
  - N_ACTIONS == 8 == len(Macro) (derived constant never drifts),
  - every Macro has an entry in MACRO_SEMANTICS (mapping is complete),
  - values are contiguous 0..N_ACTIONS-1 (lines up with a Q-head of width N_ACTIONS).
"""

import pytest

from agent.actions import Macro, N_ACTIONS, MACRO_SEMANTICS


# ---------------------------------------------------------------------------
# Frozen name/value table — any rename or reorder breaks these explicitly.
# ---------------------------------------------------------------------------

_EXPECTED: list[tuple[str, int]] = [
    ("IDLE", 0),
    ("APPROACH", 1),
    ("RETREAT", 2),
    ("STRAFE_L", 3),
    ("STRAFE_R", 4),
    ("ATTACK", 5),
    ("JUMP", 6),
    ("TURN_TO_LAST_SEEN", 7),
]


def test_enum_has_exactly_eight_members():
    """AC1: the enum contains exactly 8 members — no more, no less."""
    assert len(Macro) == 8


@pytest.mark.parametrize("name, value", _EXPECTED)
def test_enum_member_name_and_value(name: str, value: int):
    """Each member exists with the exact frozen name and integer value."""
    member = Macro[name]          # KeyError if name missing
    assert member.value == value  # wrong value breaks the freeze


def test_n_actions_equals_eight_and_len_macro():
    """N_ACTIONS == 8 == len(Macro) — the derived constant never drifts."""
    assert N_ACTIONS == 8
    assert N_ACTIONS == len(Macro)


# ---------------------------------------------------------------------------
# Contiguity guard — values must tile [0, N_ACTIONS) exactly.
# ---------------------------------------------------------------------------

def test_values_are_contiguous_from_zero():
    """Values form the set {0, 1, …, N_ACTIONS-1} with no gaps or duplicates.

    This is a precondition for using the enum as a Q-head index directly.
    """
    values = sorted(m.value for m in Macro)
    assert values == list(range(N_ACTIONS)), (
        f"Action values are not contiguous 0..{N_ACTIONS - 1}: {values}"
    )


# ---------------------------------------------------------------------------
# Semantic mapping completeness.
# ---------------------------------------------------------------------------

def test_macro_semantics_covers_every_member():
    """Every Macro member has an entry in MACRO_SEMANTICS — no macro undocumented."""
    for member in Macro:
        assert member in MACRO_SEMANTICS, (
            f"MACRO_SEMANTICS missing entry for Macro.{member.name}"
        )


def test_macro_semantics_has_no_extra_keys():
    """MACRO_SEMANTICS contains no keys that are not Macro members."""
    for key in MACRO_SEMANTICS:
        assert isinstance(key, Macro), (
            f"MACRO_SEMANTICS has unexpected key: {key!r}"
        )


def test_macro_semantics_descriptions_are_non_empty():
    """Each semantic description is a non-empty string."""
    for member, description in MACRO_SEMANTICS.items():
        assert isinstance(description, str) and description.strip(), (
            f"MACRO_SEMANTICS[{member.name}] is empty or not a string"
        )


# ---------------------------------------------------------------------------
# IntEnum contract — values must be usable as plain integers.
# ---------------------------------------------------------------------------

def test_macro_members_are_int_instances():
    """Macro members are instances of int (IntEnum guarantee)."""
    for member in Macro:
        assert isinstance(member, int), (
            f"Macro.{member.name} is not an int instance"
        )


def test_macro_usable_as_list_index():
    """Members index a list of length N_ACTIONS without raising."""
    q_values = list(range(N_ACTIONS))
    for member in Macro:
        _ = q_values[member]  # must not raise IndexError
