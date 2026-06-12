"""Tests for the Opponent interface and StationaryDummy (T18).

Coverage:
  - ``StationaryDummy.act`` returns ``Macro.IDLE`` for a variety of inputs.
  - ``StationaryDummy.config`` has all four immunity flags set to ``True``.
  - ``StationaryDummy.name`` is the expected string identifier.
  - ``Opponent`` is abstract and cannot be instantiated directly.
  - A minimal concrete subclass that implements ``act`` (and optionally
    overrides ``reset``) satisfies the interface.
"""

from __future__ import annotations

import pytest

from agent.actions import Macro
from opponents.base import Opponent, OpponentConfig
from opponents.dummy import StationaryDummy


# ---------------------------------------------------------------------------
# StationaryDummy.act — always IDLE regardless of input
# ---------------------------------------------------------------------------

_OBS_SAMPLES = [
    None,
    {},
    {"health": 18.0, "pos": [1.0, 64.0, 1.0]},
    [0.0] * 32,
    42,
    "some_string_obs",
]


@pytest.mark.parametrize("obs", _OBS_SAMPLES)
def test_dummy_always_returns_idle(obs):
    """act() returns Macro.IDLE for every observation shape, including None."""
    dummy = StationaryDummy()
    result = dummy.act(obs)
    assert result is Macro.IDLE
    assert result == Macro.IDLE
    assert int(result) == 0


def test_dummy_act_is_deterministic_across_resets():
    """act() stays IDLE before and after reset() calls."""
    dummy = StationaryDummy()
    assert dummy.act(None) is Macro.IDLE
    dummy.reset(seed=0)
    assert dummy.act({"pos": [0.0, 64.0, 0.0]}) is Macro.IDLE
    dummy.reset(seed=None)
    assert dummy.act(None) is Macro.IDLE


# ---------------------------------------------------------------------------
# StationaryDummy.config — all immunity flags True
# ---------------------------------------------------------------------------

def test_dummy_config_is_opponent_config_instance():
    dummy = StationaryDummy()
    assert isinstance(dummy.config, OpponentConfig)


def test_dummy_config_knockback_immune():
    assert StationaryDummy().config.knockback_immune is True


def test_dummy_config_fall_immune():
    assert StationaryDummy().config.fall_immune is True


def test_dummy_config_void_immune():
    assert StationaryDummy().config.void_immune is True


def test_dummy_config_fixed_spawn():
    assert StationaryDummy().config.fixed_spawn is True


def test_dummy_config_all_flags_true():
    """Single guard: all four bridge-consumed flags are True simultaneously."""
    cfg = StationaryDummy().config
    assert cfg.knockback_immune and cfg.fall_immune and cfg.void_immune and cfg.fixed_spawn


def test_dummy_config_is_frozen():
    """OpponentConfig is a frozen dataclass — mutation must raise."""
    cfg = StationaryDummy().config
    with pytest.raises((AttributeError, TypeError)):
        cfg.knockback_immune = False  # type: ignore[misc]


def test_dummy_config_same_object_on_repeated_calls():
    """config property returns the shared singleton on every access."""
    dummy = StationaryDummy()
    assert dummy.config is dummy.config


# ---------------------------------------------------------------------------
# StationaryDummy.name
# ---------------------------------------------------------------------------

def test_dummy_name():
    assert StationaryDummy().name == "stationary_dummy"


# ---------------------------------------------------------------------------
# Opponent is abstract — cannot be instantiated directly
# ---------------------------------------------------------------------------

def test_opponent_is_abstract():
    """Opponent has abstract methods; direct instantiation must raise TypeError."""
    with pytest.raises(TypeError):
        Opponent()  # type: ignore[abstract]


def test_opponent_subclass_missing_act_is_abstract():
    """A subclass that omits act() is still abstract."""
    class IncompleteOpponent(Opponent):
        @property
        def name(self) -> str:
            return "incomplete"

        @property
        def config(self) -> OpponentConfig:
            return OpponentConfig(
                knockback_immune=False,
                fall_immune=False,
                void_immune=False,
                fixed_spawn=False,
            )
        # act() intentionally NOT implemented

    with pytest.raises(TypeError):
        IncompleteOpponent()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# A minimal concrete subclass satisfies the interface
# ---------------------------------------------------------------------------

class _AttackAlways(Opponent):
    """Minimal concrete opponent used only in tests."""

    @property
    def name(self) -> str:
        return "attack_always"

    @property
    def config(self) -> OpponentConfig:
        return OpponentConfig(
            knockback_immune=False,
            fall_immune=False,
            void_immune=False,
            fixed_spawn=False,
        )

    def act(self, observation) -> Macro:
        return Macro.ATTACK


def test_concrete_subclass_instantiates():
    opp = _AttackAlways()
    assert isinstance(opp, Opponent)


def test_concrete_subclass_act():
    opp = _AttackAlways()
    assert opp.act(None) is Macro.ATTACK


def test_concrete_subclass_reset_is_noop():
    """Inherited reset() accepts any seed without raising."""
    opp = _AttackAlways()
    opp.reset()
    opp.reset(seed=0)
    opp.reset(seed=None)


def test_concrete_subclass_name_and_config():
    opp = _AttackAlways()
    assert opp.name == "attack_always"
    cfg = opp.config
    assert isinstance(cfg, OpponentConfig)
    assert cfg.knockback_immune is False
