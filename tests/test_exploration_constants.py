"""Tests for the exploration schedule and the multi-arena training constants (T16).

Every test here exists because a wrong CONSTANT is invisible: the run starts, the
loss falls, checkpoints appear, and the only symptom is an agent that stopped
exploring an hour in. Each test is written to fail if the constant or the formula
were wrong, not merely to re-state the shipped default:

* **The headline bug (AC15).** ``eps_decay_episodes=200`` was single-arena sizing.
  Every collector claims its episode index from ONE shared
  ``distributed.actor.GlobalEpisodeCounter`` (``distributed/actor.py:672-674``;
  the counter is built at ``agent/train.py`` in ``train_multi_arena`` and re-read
  for the epsilon log), so at 25 pads that value floors epsilon after ~8 episodes
  PER ARENA — ~0.1% of a night, against the field's ~15% guidance. Pinned below
  as a PROPERTY of the schedule evaluated at the chosen pad count, and pinned
  again by asserting the OLD value still violates it.
* **The projection is a formula, not a magic number.** Pinned with hand-computed
  arithmetic on explicit inputs, so a changed coefficient or a swapped
  numerator/denominator fails even if the module constants move.
* **Ape-X per-actor epsilon (issue #15).** Pinned: arena 0 is the MOST
  exploratory (the convention the whole scheme rests on), the ordering is strict,
  the exact exponent is ``i/(N-1)`` and not ``i/N``, N==1 does not divide by zero,
  and the wrap actually reaches the collectors' policies.
* **Every new knob is reachable from the CLI.** The plan's declared cut #2 has to
  be executable at 3am without editing source, so each flag is pinned at its
  parse/validate boundary.

No socket, no live server, no Minecraft.
"""

from __future__ import annotations

import dataclasses
from typing import Any, List, Tuple

import pytest

from agent.train_config import (
    ASSUMED_MEAN_EPISODE_STEPS,
    ASSUMED_RUN_HOURS,
    DEFAULT_EPS_DECAY_ARENAS,
    EPS_DECAY_FRACTION_OF_RUN,
    MEASURED_PER_ARENA_TRANSITIONS_PER_S,
    TrainConfig,
    eps_decay_episodes_for,
    projected_episodes,
)

#: AC16's measured choice: 25 pads, 121.95 transitions/s aggregate.
CHOSEN_PADS = 25

#: What ``eps_decay_episodes`` used to be — single-arena sizing.
OLD_SINGLE_ARENA_DEFAULT = 200


def _cfg(**overrides: Any) -> TrainConfig:
    return dataclasses.replace(TrainConfig(), **overrides)


def _parse(argv: List[str]) -> Any:
    from agent.train import _build_parser

    return _build_parser().parse_args(argv)


def _cfg_from_argv(argv: List[str]) -> TrainConfig:
    from agent.train import _config_from_args

    return _config_from_args(_parse(argv))


# ===========================================================================
# (a) The epsilon schedule sizing — AC15 / AC17.
# ===========================================================================


class TestProjectedEpisodesFormula:
    """The projection is arithmetic on named inputs, and it is checkable by hand."""

    def test_the_formula_is_arenas_times_rate_times_seconds_over_episode_length(self):
        # Hand-computed on EXPLICIT inputs so this survives (and catches) any
        # change to the module's own assumption constants:
        #   10 pads * 5 transitions/s * 3600 s/h * 10 h = 1,800,000 transitions
        #   1,800,000 / 25 steps-per-episode           =    72,000 episodes
        assert projected_episodes(
            10,
            hours=10.0,
            per_arena_transitions_per_s=5.0,
            mean_episode_steps=25.0,
        ) == pytest.approx(72_000.0)

    def test_it_is_linear_in_the_pad_count(self):
        # AC16 measured the per-arena rate flat to 0.05% from 16 to 25 pads, which
        # is exactly what licenses multiplying instead of re-measuring.
        one = projected_episodes(1)
        assert projected_episodes(2) == pytest.approx(2.0 * one)
        assert projected_episodes(CHOSEN_PADS) == pytest.approx(CHOSEN_PADS * one)

    def test_a_longer_episode_means_fewer_episodes(self):
        # The mean episode length is a DIVISOR. Getting this backwards is how the
        # 400-step timeout would sneak in and shrink the decay window 13x.
        short = projected_episodes(CHOSEN_PADS, mean_episode_steps=30.0)
        long = projected_episodes(CHOSEN_PADS, mean_episode_steps=400.0)
        assert long < short
        assert short / long == pytest.approx(400.0 / 30.0)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"arenas": 0},
            {"arenas": 1, "hours": 0.0},
            {"arenas": 1, "per_arena_transitions_per_s": 0.0},
            {"arenas": 1, "mean_episode_steps": -5.0},
            {"arenas": 1, "mean_episode_steps": float("nan")},
        ],
    )
    def test_a_degenerate_projection_input_is_rejected(self, kwargs):
        arenas = kwargs.pop("arenas")
        with pytest.raises(ValueError):
            projected_episodes(arenas, **kwargs)

    def test_the_measured_rate_matches_the_ac16_confirm_run(self):
        # 121.95/s aggregate over 25 pads (runs/confirm-n25/summary.json). If
        # someone re-measures, this is the line that has to change with it.
        assert MEASURED_PER_ARENA_TRANSITIONS_PER_S * CHOSEN_PADS == pytest.approx(
            121.955, abs=0.01
        )

    def test_the_decay_fraction_is_the_fifteen_percent_ac15_is_written_against(self):
        # Pinned as a LITERAL, not against itself: every other assertion in this
        # file compares the schedule to EPS_DECAY_FRACTION_OF_RUN, so moving the
        # constant would otherwise move the whole target with it and AC15 ("~15%")
        # would be satisfied by definition at any value.
        assert EPS_DECAY_FRACTION_OF_RUN == pytest.approx(0.15)

    def test_the_run_length_assumption_is_one_unattended_night(self):
        assert 8.0 <= ASSUMED_RUN_HOURS <= 16.0

    def test_the_assumed_episode_length_is_not_the_timeout(self):
        from agent.contract_config import MAX_EPISODE_STEPS

        # The plan says this explicitly: 400 is a TIMEOUT, not a typical episode.
        # It also must stay above the one measured figure (17.0, a greedy eval vs
        # a stationary dummy) which is a lower bound on a real training episode.
        assert ASSUMED_MEAN_EPISODE_STEPS != MAX_EPISODE_STEPS
        assert 17.0 <= ASSUMED_MEAN_EPISODE_STEPS < MAX_EPISODE_STEPS


class TestEpsDecayEpisodesFor:
    """The default decay window is derived, and derived from the arena count."""

    def test_it_is_the_configured_fraction_of_the_projected_episodes(self):
        # Hand-computed again: 72,000 projected episodes (above) * 0.15 = 10,800.
        assert (
            eps_decay_episodes_for(
                10,
                fraction=0.15,
                hours=10.0,
                per_arena_transitions_per_s=5.0,
                mean_episode_steps=25.0,
            )
            == 10_800
        )

    def test_it_scales_with_the_pad_count(self):
        # The whole point: 25 pads burn the shared counter 25x faster.
        one = eps_decay_episodes_for(1)
        many = eps_decay_episodes_for(CHOSEN_PADS)
        assert many > one
        assert many == pytest.approx(CHOSEN_PADS * one, rel=0.01)

    def test_it_never_returns_a_window_trainconfig_would_reject(self):
        # TrainConfig requires eps_decay_episodes >= 1; a tiny projection must
        # floor at 1 rather than produce a 0 denominator in epsilon_for_episode.
        tiny = eps_decay_episodes_for(
            1, fraction=1e-9, hours=0.001, mean_episode_steps=400.0
        )
        assert tiny >= 1
        _cfg(eps_decay_episodes=tiny)  # must validate

    @pytest.mark.parametrize("fraction", [0.0, -0.1, 1.5, float("nan")])
    def test_a_fraction_outside_zero_to_one_is_rejected(self, fraction):
        with pytest.raises(ValueError, match="fraction must be in"):
            eps_decay_episodes_for(4, fraction=fraction)

    def test_the_dataclass_default_calls_this_same_function(self):
        # Two places computing the same thing is how they drift. The default must
        # BE the function's output, not a copy of it that a re-tune forgets.
        assert TrainConfig().eps_decay_episodes == eps_decay_episodes_for(
            DEFAULT_EPS_DECAY_ARENAS
        )

    def test_the_default_is_no_longer_the_single_arena_literal(self):
        assert TrainConfig().eps_decay_episodes != OLD_SINGLE_ARENA_DEFAULT


class TestAC15EpsilonStillExploresAtTheChosenPadCount:
    """TC26: at 25 pads epsilon must still be above its floor at ~15% of the run."""

    def _floor_fraction(self, cfg: TrainConfig) -> float:
        from agent.train import eps_floor_fraction_of_run

        return eps_floor_fraction_of_run(cfg)

    def test_the_derived_window_puts_the_floor_at_the_target_fraction(self):
        cfg = _cfg(
            arenas=CHOSEN_PADS, eps_decay_episodes=eps_decay_episodes_for(CHOSEN_PADS)
        )
        fraction = self._floor_fraction(cfg)

        assert fraction == pytest.approx(EPS_DECAY_FRACTION_OF_RUN, rel=0.01)
        # AC15's own words, as a literal band: "does not reach its floor before
        # ~15% of the run's episodes". Written out so re-tuning the module
        # constant cannot move the acceptance target with it.
        assert 0.10 <= fraction <= 0.20

    def test_epsilon_is_still_above_the_floor_just_before_that_point(self):
        from agent.train import epsilon_for_episode

        cfg = _cfg(
            arenas=CHOSEN_PADS,
            eps_decay_episodes=eps_decay_episodes_for(CHOSEN_PADS),
            warm_start="runs/m2_multi.pt",  # the retrain's real regime
        )
        # The decay is linear and saturates EXACTLY at eps_decay_episodes, so the
        # last exploring episode is the one before it.
        assert epsilon_for_episode(cfg.eps_decay_episodes - 1, cfg) > cfg.eps_end
        assert epsilon_for_episode(cfg.eps_decay_episodes, cfg) == pytest.approx(
            cfg.eps_end
        )

    def test_the_old_default_would_still_violate_ac15_at_this_pad_count(self):
        # The mutation guard. If someone re-pins eps_decay_episodes to a
        # single-arena literal, THIS is the assertion that says so out loud.
        cfg = _cfg(arenas=CHOSEN_PADS, eps_decay_episodes=OLD_SINGLE_ARENA_DEFAULT)
        fraction = self._floor_fraction(cfg)
        assert fraction < 0.01, (
            "the old default was supposed to floor epsilon ~1% into a 25-pad run; "
            f"got {fraction:.4%} - the projection changed and this test's premise "
            "with it"
        )
        assert fraction < EPS_DECAY_FRACTION_OF_RUN

    def test_a_run_at_one_pad_is_unaffected_by_the_multi_arena_sizing(self):
        cfg = _cfg(arenas=1)
        assert self._floor_fraction(cfg) == pytest.approx(
            EPS_DECAY_FRACTION_OF_RUN, rel=0.01
        )


class TestEpsilonScheduleReport:
    """The startup line is the operator's only chance to see this before 3am."""

    def test_a_correctly_sized_run_reports_one_line_and_no_warning(self):
        from agent.train import epsilon_schedule_report

        cfg = _cfg(
            arenas=CHOSEN_PADS, eps_decay_episodes=eps_decay_episodes_for(CHOSEN_PADS)
        )
        lines = epsilon_schedule_report(cfg)

        assert len(lines) == 1
        assert "WARNING" not in lines[0]
        assert str(cfg.eps_decay_episodes) in lines[0]
        assert "GLOBAL episodes" in lines[0]

    def test_a_hand_built_multi_arena_config_gets_the_warning(self):
        # A dataclass default cannot depend on `arenas`, so TrainConfig(arenas=25)
        # keeps the N=1 window. The CLI re-derives it; a hand-built config does
        # not, and this line is what makes that loud instead of silent.
        from agent.train import epsilon_schedule_report

        lines = epsilon_schedule_report(_cfg(arenas=CHOSEN_PADS))

        assert len(lines) == 2
        assert "WARNING" in lines[1]
        # It must name the value to pass, not just complain.
        assert str(eps_decay_episodes_for(CHOSEN_PADS)) in lines[1]

    def test_the_warning_threshold_is_a_real_threshold(self):
        from agent.train import EPS_FLOOR_WARN_FRACTION, epsilon_schedule_report

        projected = projected_episodes(CHOSEN_PADS)
        just_under = _cfg(
            arenas=CHOSEN_PADS,
            eps_decay_episodes=int(projected * EPS_FLOOR_WARN_FRACTION) - 1,
        )
        just_over = _cfg(
            arenas=CHOSEN_PADS,
            eps_decay_episodes=int(projected * EPS_FLOOR_WARN_FRACTION) + 1,
        )

        assert len(epsilon_schedule_report(just_under)) == 2
        assert len(epsilon_schedule_report(just_over)) == 1

    def test_it_names_the_spread_state_either_way(self):
        from agent.train import epsilon_schedule_report

        on = epsilon_schedule_report(
            _cfg(arenas=4, eps_decay_episodes=eps_decay_episodes_for(4))
        )[0]
        off = epsilon_schedule_report(
            _cfg(
                arenas=4,
                per_actor_eps=False,
                eps_decay_episodes=eps_decay_episodes_for(4),
            )
        )[0]

        assert "per-actor eps ON" in on
        assert "per-actor eps OFF" in off

    def test_every_line_is_ascii(self):
        # The run log is cp1252-safe by contract elsewhere in this module.
        from agent.train import epsilon_schedule_report

        for cfg in (_cfg(arenas=CHOSEN_PADS), _cfg(arenas=1)):
            for line in epsilon_schedule_report(cfg):
                line.encode("ascii")


# ===========================================================================
# (f) Ape-X per-actor epsilon — issue #15 / TC27.
# ===========================================================================


class TestPerActorEpsilonFormula:
    """arena 0 is the MOST exploratory; the exponent is i/(N-1), not i/N."""

    def test_arena_zero_gets_the_base_epsilon_unchanged(self):
        from agent.train import per_actor_epsilon

        # The Ape-X convention the whole scheme rests on. Reversing the ordering
        # would still produce a monotone spread, which is why this is its own test.
        assert per_actor_epsilon(0.4, 0, 8, 7.0) == pytest.approx(0.4)

    def test_arena_zero_is_the_largest_epsilon_in_the_fleet(self):
        from agent.train import per_actor_epsilon

        values = [per_actor_epsilon(0.4, i, 8, 7.0) for i in range(8)]
        assert values[0] == max(values)
        assert values[0] > values[1]

    def test_the_last_arena_is_the_base_raised_to_one_plus_alpha(self):
        from agent.train import per_actor_epsilon

        # Hand-computed: 0.4 ** 8 == 0.00065536. Monotonicity alone would survive
        # an i/N vs i/(N-1) typo; this exact value would not.
        assert per_actor_epsilon(0.4, 7, 8, 7.0) == pytest.approx(0.00065536)

    def test_a_middle_arena_uses_i_over_n_minus_one(self):
        from agent.train import per_actor_epsilon

        # N=5, i=2, alpha=8 -> exponent 1 + (2/4)*8 = 5 -> 0.5**5 = 0.03125.
        # With the off-by-one i/N the exponent would be 4.2 (0.0543...).
        assert per_actor_epsilon(0.5, 2, 5, 8.0) == pytest.approx(0.03125)

    def test_the_spread_is_strictly_monotone_across_arenas(self):
        from agent.train import per_actor_epsilon

        values = [per_actor_epsilon(0.3, i, CHOSEN_PADS, 7.0) for i in range(CHOSEN_PADS)]
        assert len(set(values)) == CHOSEN_PADS, "arenas must explore at DISTINCT rates"
        assert values == sorted(values, reverse=True)

    def test_one_arena_returns_the_base_without_dividing_by_zero(self):
        from agent.train import per_actor_epsilon

        # N-1 == 0. The guard must come BEFORE the ratio is formed.
        assert per_actor_epsilon(0.4, 0, 1, 7.0) == pytest.approx(0.4)
        assert per_actor_epsilon(0.0, 0, 1, 7.0) == 0.0
        assert per_actor_epsilon(1.0, 0, 1, 7.0) == 1.0

    def test_alpha_controls_how_far_the_fleet_fans_out(self):
        from agent.train import per_actor_epsilon

        tight = per_actor_epsilon(0.5, 3, 4, 1.0)
        wide = per_actor_epsilon(0.5, 3, 4, 7.0)
        assert wide < tight, "a larger alpha must push the exploit arm further down"

    @pytest.mark.parametrize("base", [0.0, 1.0])
    def test_the_degenerate_bases_collapse_the_spread_and_that_is_correct(self, base):
        from agent.train import per_actor_epsilon

        # Documented caveat: distinctness holds only for 0 < eps < 1. base==1.0 is
        # episode 0 of a fresh run (one episode later it is not), base==0 never
        # happens because eps_end is the floor.
        values = {per_actor_epsilon(base, i, 6, 7.0) for i in range(6)}
        assert values == {base}

    @pytest.mark.parametrize(
        "args",
        [
            (0.4, 0, 0, 7.0),  # arenas < 1
            (0.4, 8, 8, 7.0),  # arena_id == arenas
            (0.4, -1, 8, 7.0),  # negative arena_id
            (1.5, 0, 8, 7.0),  # base above 1 would INVERT the ordering
            (-0.1, 0, 8, 7.0),  # negative base
            (float("nan"), 0, 8, 7.0),
        ],
    )
    def test_bad_inputs_are_rejected(self, args):
        from agent.train import per_actor_epsilon

        with pytest.raises(ValueError):
            per_actor_epsilon(*args)


class TestPerActorEpsilonGating:
    """Default ON, but only for multi-arena runs, and instantly disableable."""

    @pytest.mark.parametrize(
        "arenas,flag,expected",
        [
            (1, True, False),  # N==1: never, even though the flag defaults on
            (1, False, False),
            (2, True, True),
            (25, True, True),
            (25, False, False),  # the CLI off switch
        ],
    )
    def test_the_gate_is_flag_and_pad_count(self, arenas, flag, expected):
        from agent.train import per_actor_eps_enabled

        assert per_actor_eps_enabled(_cfg(arenas=arenas, per_actor_eps=flag)) is expected

    def test_it_is_on_by_default_for_a_multi_arena_run(self):
        from agent.train import per_actor_eps_enabled

        assert TrainConfig().per_actor_eps is True
        assert per_actor_eps_enabled(_cfg(arenas=CHOSEN_PADS)) is True

    def test_the_single_arena_path_is_untouched_by_the_default(self):
        from agent.train import per_actor_eps_enabled

        assert per_actor_eps_enabled(TrainConfig()) is False


class TestMeanPerActorEpsilon:
    """The logged mean has to be the fleet's mean, not the schedule value."""

    def test_with_the_spread_off_it_is_the_schedule_value(self):
        from agent.train import mean_per_actor_epsilon

        cfg = _cfg(arenas=8, per_actor_eps=False)
        assert mean_per_actor_epsilon(0.5, cfg) == pytest.approx(0.5)

    def test_at_one_arena_it_is_the_schedule_value(self):
        from agent.train import mean_per_actor_epsilon

        assert mean_per_actor_epsilon(0.5, _cfg(arenas=1)) == pytest.approx(0.5)

    def test_with_the_spread_on_it_is_the_hand_computed_fleet_mean(self):
        from agent.train import mean_per_actor_epsilon

        # N=2, alpha=7: arenas act at 0.5 and 0.5**8 == 0.00390625.
        cfg = _cfg(arenas=2, per_actor_eps=True, per_actor_eps_alpha=7.0)
        assert mean_per_actor_epsilon(0.5, cfg) == pytest.approx(
            (0.5 + 0.00390625) / 2.0
        )

    def test_it_is_far_below_the_schedule_value_at_the_chosen_pad_count(self):
        from agent.train import mean_per_actor_epsilon

        # This is the number that made the old `train/epsilon_mean` line false.
        cfg = _cfg(arenas=CHOSEN_PADS)
        assert mean_per_actor_epsilon(0.25, cfg) < 0.25 / 4.0


class TestEpsilonLogRow:
    """`train/epsilon_mean` must be the fleet's mean, not the schedule value."""

    def test_the_mean_is_below_the_schedule_once_the_spread_is_on(self):
        from agent.train import epsilon_log_row, mean_per_actor_epsilon

        cfg = _cfg(
            arenas=CHOSEN_PADS, eps_decay_episodes=eps_decay_episodes_for(CHOSEN_PADS)
        )
        row = epsilon_log_row(1_000, cfg)

        assert row["train/epsilon_mean"] < row["train/epsilon_schedule"]
        assert row["train/epsilon_mean"] == pytest.approx(
            mean_per_actor_epsilon(row["train/epsilon_schedule"], cfg)
        )

    def test_with_the_spread_off_the_two_rows_agree(self):
        from agent.train import epsilon_log_row

        cfg = _cfg(arenas=CHOSEN_PADS, per_actor_eps=False)
        row = epsilon_log_row(1_000, cfg)

        assert row["train/epsilon_mean"] == pytest.approx(
            row["train/epsilon_schedule"]
        )

    def test_the_schedule_row_samples_the_last_claimed_episode(self):
        from agent.train import epsilon_for_episode, epsilon_log_row

        cfg = _cfg(arenas=4, eps_decay_episodes=1_000)

        assert epsilon_log_row(500, cfg)["train/epsilon_schedule"] == pytest.approx(
            epsilon_for_episode(499, cfg)
        )
        # A fresh run (nothing claimed yet) must not index -1.
        assert epsilon_log_row(0, cfg)["train/epsilon_schedule"] == pytest.approx(
            epsilon_for_episode(0, cfg)
        )

    def test_the_row_decays_as_the_run_advances(self):
        from agent.train import epsilon_log_row

        cfg = _cfg(arenas=4, eps_decay_episodes=1_000)
        early = epsilon_log_row(10, cfg)
        late = epsilon_log_row(900, cfg)

        assert late["train/epsilon_schedule"] < early["train/epsilon_schedule"]
        assert late["train/epsilon_mean"] < early["train/epsilon_mean"]

    def test_the_multi_arena_loop_logs_this_row(self):
        # The row builder is module-level precisely so it can be tested; this
        # pins that train_multi_arena actually uses it rather than rebuilding a
        # (differently wrong) dict inline.
        import inspect

        from agent.train import train_multi_arena

        source = inspect.getsource(train_multi_arena)
        assert "epsilon_log_row(counter.value, cfg)" in source
        assert '"train/epsilon_mean"' not in source, (
            "the metric name is built inside epsilon_log_row; a second inline "
            "copy is how the two drift"
        )


class _RecordingPolicy:
    """Minimal RolloutPolicy stand-in that records the epsilon it was handed."""

    def __init__(self, arena_id: int = 0) -> None:
        self.arena_id = arena_id
        self.policy_version = -1
        self.code_version = "test"
        self.seen_epsilons: List[float] = []
        self.reseeded: List[int] = []
        self.refreshed = 0

    def maybe_refresh(self, store: Any) -> None:
        self.refreshed += 1
        self.policy_version += 1  # mutates, exactly like SnapshotPolicy

    def reseed(self, episode_seed: int) -> None:
        self.reseeded.append(int(episode_seed))

    def init_hidden(self) -> Tuple[str, str]:
        return ("h", "c")

    def act(self, obs: Any, hidden: Any, epsilon: float):
        self.seen_epsilons.append(float(epsilon))
        return 0, hidden


class TestPerActorEpsilonPolicyWrapper:
    """The wrapper must transform epsilon and pass EVERYTHING else through."""

    def test_act_receives_this_arenas_epsilon_not_the_schedule_value(self):
        from agent.train import PerActorEpsilonPolicy, per_actor_epsilon

        inner = _RecordingPolicy(arena_id=3)
        wrapped = PerActorEpsilonPolicy(inner, arenas=4, alpha=7.0)

        wrapped.act(None, ("h", "c"), 0.5)

        assert inner.seen_epsilons == [pytest.approx(per_actor_epsilon(0.5, 3, 4, 7.0))]
        assert inner.seen_epsilons[0] != pytest.approx(0.5)

    def test_arena_zero_still_acts_at_the_schedule_value(self):
        from agent.train import PerActorEpsilonPolicy

        inner = _RecordingPolicy(arena_id=0)
        PerActorEpsilonPolicy(inner, arenas=4, alpha=7.0).act(None, ("h", "c"), 0.5)

        assert inner.seen_epsilons == [pytest.approx(0.5)]

    def test_the_episode_stamp_is_delegated_live_not_copied(self):
        from agent.train import PerActorEpsilonPolicy

        inner = _RecordingPolicy(arena_id=2)
        wrapped = PerActorEpsilonPolicy(inner, arenas=4, alpha=7.0)

        assert wrapped.policy_version == -1
        wrapped.maybe_refresh(object())
        # A copy taken at construction would freeze every Episode's provenance
        # stamp at -1.
        assert wrapped.policy_version == 0
        assert wrapped.arena_id == 2
        assert wrapped.code_version == "test"

    def test_reseed_init_hidden_and_refresh_pass_straight_through(self):
        from agent.train import PerActorEpsilonPolicy

        inner = _RecordingPolicy(arena_id=1)
        wrapped = PerActorEpsilonPolicy(inner, arenas=4, alpha=7.0)

        wrapped.reseed(1234)
        assert wrapped.init_hidden() == ("h", "c")
        wrapped.maybe_refresh(object())

        assert inner.reseeded == [1234]
        assert inner.refreshed == 1

    def test_unknown_attributes_fall_through_to_the_wrapped_policy(self):
        from agent.train import PerActorEpsilonPolicy

        inner = _RecordingPolicy()
        inner.net = "the-clone"  # type: ignore[attr-defined]
        wrapped = PerActorEpsilonPolicy(inner, arenas=2, alpha=7.0)

        assert wrapped.net == "the-clone"
        with pytest.raises(AttributeError):
            wrapped.nonexistent_attribute

    @pytest.mark.parametrize("bad", [{"arenas": 1}, {"alpha": 0.0}, {"alpha": -1.0}])
    def test_a_pointless_or_invalid_wrapper_is_refused(self, bad):
        from agent.train import PerActorEpsilonPolicy

        kwargs = {"arenas": 4, "alpha": 7.0}
        kwargs.update(bad)
        with pytest.raises(ValueError):
            PerActorEpsilonPolicy(_RecordingPolicy(), **kwargs)


class TestPerActorEpsilonWiring:
    """The transform has to actually reach the collectors, not just exist."""

    def test_the_wrap_happens_only_when_the_gate_says_so(self):
        from agent.train import PerActorEpsilonPolicy, maybe_wrap_per_actor_epsilon

        inner = _RecordingPolicy(arena_id=1)

        assert maybe_wrap_per_actor_epsilon(inner, _cfg(arenas=1)) is inner
        assert (
            maybe_wrap_per_actor_epsilon(inner, _cfg(arenas=8, per_actor_eps=False))
            is inner
        )
        wrapped = maybe_wrap_per_actor_epsilon(inner, _cfg(arenas=8))
        assert isinstance(wrapped, PerActorEpsilonPolicy)
        assert wrapped is not inner

    def test_the_wrapper_carries_the_configured_alpha(self):
        from agent.train import maybe_wrap_per_actor_epsilon, per_actor_epsilon

        inner = _RecordingPolicy(arena_id=3)
        wrapped = maybe_wrap_per_actor_epsilon(
            inner, _cfg(arenas=4, per_actor_eps_alpha=2.0)
        )
        wrapped.act(None, ("h", "c"), 0.5)

        # alpha=2.0, not the default 7.0.
        assert inner.seen_epsilons == [pytest.approx(per_actor_epsilon(0.5, 3, 4, 2.0))]

    def test_build_arena_policy_wraps_on_the_multi_arena_path(self):
        pytest.importorskip("torch")
        from agent.train import PerActorEpsilonPolicy, build_arena_policy

        built = [
            build_arena_policy(
                i, _cfg(arenas=4), net_factory=_tiny_net_factory(), code_version="cv"
            )
            for i in range(4)
        ]

        assert all(isinstance(p, PerActorEpsilonPolicy) for p in built)
        assert [p.arena_id for p in built] == [0, 1, 2, 3]
        assert all(p.code_version == "cv" for p in built)
        # The arenas must end up at DISTINCT, ordered exploration rates (TC27).
        spread = [p.epsilon_for(0.4) for p in built]
        assert spread == sorted(spread, reverse=True)
        assert len(set(spread)) == 4

    def test_build_arena_policy_leaves_the_single_arena_path_bare(self):
        pytest.importorskip("torch")
        from distributed.weights import SnapshotPolicy

        from agent.train import PerActorEpsilonPolicy, build_arena_policy

        policy = build_arena_policy(
            0, _cfg(arenas=1), net_factory=_tiny_net_factory(), code_version="cv"
        )

        assert isinstance(policy, SnapshotPolicy)
        assert not isinstance(policy, PerActorEpsilonPolicy)

    def test_build_arena_policy_seeds_each_arena_from_its_own_band(self):
        pytest.importorskip("torch")
        from agent.train import build_arena_policy

        cfg = _cfg(arenas=3)
        policies = [
            build_arena_policy(i, cfg, net_factory=_tiny_net_factory()) for i in range(3)
        ]

        # Distinct RNG streams per arena survive the wrapping (the wrapper must
        # not swallow reseed).
        for i, policy in enumerate(policies):
            policy.reseed(i)
        assert len({id(p) for p in policies}) == 3


def _tiny_net_factory():
    """A cheap real DRQN factory (the frozen OBS_DIM/N_ACTIONS still assert)."""
    from agent.dqn import DuelingDRQN

    return lambda: DuelingDRQN(encoder_hidden=8, lstm_hidden=8, lstm_layers=1)


# ===========================================================================
# (b)(c)(d)(e) The replay / checkpoint constants.
# ===========================================================================


class TestReplaySizing:
    """1e6 is ~2.3 hours of the measured stream, and it costs ~2.3 GB."""

    def test_the_capacity_holds_hours_of_the_measured_stream_not_minutes(self):
        aggregate_per_s = MEASURED_PER_ARENA_TRANSITIONS_PER_S * CHOSEN_PADS
        minutes = TrainConfig().replay_capacity / aggregate_per_s / 60.0

        assert minutes > 120.0, (
            f"the buffer holds only {minutes:.0f} minutes of collection at the "
            "measured 25-pad rate; an overnight run would train almost entirely "
            "on the last few minutes it collected"
        )
        # And the value it replaced would not have.
        assert 100_000 / aggregate_per_s / 60.0 < 30.0

    def test_the_documented_per_transition_memory_cost_is_measured_not_guessed(self):
        # The plan estimated ~200 MB at OBS_DIM=23. That counts the transition
        # tuple and forgets the per-step LSTM hidden snapshot, which is 90% of the
        # real cost. Measure it rather than restate it.
        pytest.importorskip("torch")
        import numpy as np

        from agent.dqn import LSTM_HIDDEN, LSTM_LAYERS
        from agent.replay import PrioritizedSequenceReplay
        from env.observation_spec import OBS_DIM

        capacity, ep_len = 2_000, 50
        buf = PrioritizedSequenceReplay(capacity, seq_len=16, burn_in=4, n_step=3)
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        hidden = np.zeros((2, LSTM_LAYERS, LSTM_HIDDEN), dtype=np.float32)
        stored = 0
        while stored + ep_len <= capacity:
            buf.add_episode(
                [(obs.copy(), 1, 0.5, obs.copy(), False) for _ in range(ep_len)],
                hidden_states=[hidden.copy() for _ in range(ep_len)],
            )
            stored += ep_len

        payload = sum(
            arr.nbytes
            for episode in buf._episodes.values()
            for arr in (
                episode.obs,
                episode.actions,
                episode.rewards,
                episode.next_obs,
                episode.dones,
                episode.hidden,
            )
            if arr is not None
        )
        per_transition = payload / stored
        hidden_bytes = 2 * LSTM_LAYERS * LSTM_HIDDEN * 4

        # The hidden snapshot dominates; the field comment says so.
        assert hidden_bytes == 2048
        assert per_transition == pytest.approx(2245.0, abs=1.0)
        assert hidden_bytes / per_transition > 0.85

        gigabytes = TrainConfig().replay_capacity * per_transition / 1e9
        assert 2.0 < gigabytes < 2.6, (
            f"the shipped replay_capacity projects to {gigabytes:.2f} GB; the "
            "field comment claims ~2.3 GB and one of the two is now wrong"
        )
        assert gigabytes > 1.0, "the plan's ~200 MB estimate was off by ~11x"

    @pytest.mark.parametrize("min_replay,ready", [(10, True), (500, False)])
    def test_min_replay_is_the_live_gate_on_the_first_gradient_step(
        self, min_replay, ready
    ):
        # Not an assertion about the shipped default: two DIFFERENT non-default
        # values driven through the Trainer, each of which must actually decide
        # whether a gradient step happens.
        pytest.importorskip("torch")
        import numpy as np

        from env.observation_spec import OBS_DIM

        from agent.train import Trainer

        cfg = _cfg(
            min_replay=min_replay,
            replay_capacity=2_000,
            seq_len=2,
            burn_in=1,
            n_step=1,
            batch_size=2,
        )
        trainer = Trainer(cfg, net_kwargs={"encoder_hidden": 8, "lstm_hidden": 8})
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        trainer.replay.add_episode(
            [(obs.copy(), 0, 0.0, obs.copy(), False) for _ in range(50)]
        )

        assert trainer.ready_to_learn() is ready
        assert (trainer.learn() is not None) is ready

    def test_min_replay_covers_many_episodes_worth_of_warm_up(self):
        cfg = TrainConfig()
        episodes = cfg.min_replay / ASSUMED_MEAN_EPISODE_STEPS

        assert episodes > 100, (
            "the warm-up must span enough episodes that the first gradients are "
            f"not a handful of correlated openings; got {episodes:.0f}"
        )
        # And it must leave room for at least one sampleable window.
        assert cfg.min_replay > cfg.burn_in + cfg.seq_len

    def test_the_sequence_geometry_is_unchanged(self):
        # T16(e): seq_len/burn_in/n_step stay put. Changing seq_len without
        # changing the buffer's would silently break the R2D2 window contract.
        cfg = TrainConfig()
        assert (cfg.seq_len, cfg.burn_in, cfg.n_step) == (16, 4, 3)


class TestCheckpointInterval:
    """checkpoint_interval is NOT dead — T13 made it the multi-arena default."""

    def test_the_default_cadence_halved(self):
        assert TrainConfig().checkpoint_interval == 5_000

    def test_it_is_the_live_cadence_wherever_fire_hooks_runs(self):
        # Two DIFFERENT non-default intervals, so this fails if the hook is ever
        # rewired to a literal. (The multi-arena path's use of the same field as
        # the periodic-save default is pinned by
        # tests/test_warm_start_selection.py::
        # TestMultiArenaCheckpointsAreIndependentOfEval.)
        pytest.importorskip("torch")
        from agent.train import LearnStats, Trainer

        for interval, expected in ((3, [3, 6, 9]), (5, [5, 10])):
            cfg = _cfg(
                checkpoint_interval=interval,
                replay_capacity=2_000,
                min_replay=1,
                seq_len=2,
                burn_in=1,
                n_step=1,
                batch_size=2,
                log_interval=0,
                eval_interval=0,
            )
            trainer = Trainer(cfg, net_kwargs={"encoder_hidden": 8, "lstm_hidden": 8})
            fired: List[int] = []
            stats = LearnStats(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)
            for step in range(1, 11):
                trainer.grad_step = step
                trainer._fire_hooks(
                    stats, None, lambda _t, s: fired.append(int(s)), None
                )
            assert fired == expected


# ===========================================================================
# The CLI surface — every knob reachable without editing source.
# ===========================================================================


class TestExplorationFlags:
    """Cut #2 has to be executable at 3am, and the epsilon window overridable."""

    def test_eps_decay_episodes_is_derived_from_the_pad_count_when_omitted(self):
        derived = _cfg_from_argv(["--arenas", str(CHOSEN_PADS)])

        assert derived.eps_decay_episodes == eps_decay_episodes_for(CHOSEN_PADS)
        # The load-bearing half: it must NOT be the dataclass default, or the
        # derivation has been deleted and nothing else would notice.
        assert derived.eps_decay_episodes != TrainConfig().eps_decay_episodes

    def test_the_derivation_tracks_the_pad_count(self):
        four = _cfg_from_argv(["--arenas", "4"]).eps_decay_episodes
        twenty_five = _cfg_from_argv(["--arenas", "25"]).eps_decay_episodes

        assert four == eps_decay_episodes_for(4)
        assert twenty_five > four

    def test_the_single_arena_default_is_untouched_by_the_derivation(self):
        assert _cfg_from_argv([]).eps_decay_episodes == TrainConfig().eps_decay_episodes

    def test_an_explicit_window_beats_the_derivation(self):
        # THE flag the operator sets from the smoke run's measured mean episode
        # length, instead of editing ASSUMED_MEAN_EPISODE_STEPS.
        cfg = _cfg_from_argv(
            ["--arenas", str(CHOSEN_PADS), "--eps-decay-episodes", "12345"]
        )

        assert cfg.eps_decay_episodes == 12_345
        assert cfg.eps_decay_episodes != eps_decay_episodes_for(CHOSEN_PADS)

    def test_a_nonsensical_window_is_rejected_before_anything_starts(self):
        with pytest.raises(ValueError, match="eps_decay_episodes must be >= 1"):
            _cfg_from_argv(["--eps-decay-episodes", "0"])

    def test_the_replay_flags_land_on_the_config(self):
        cfg = _cfg_from_argv(
            ["--arenas", "4", "--replay-capacity", "250000", "--min-replay", "7500"]
        )

        assert cfg.replay_capacity == 250_000
        assert cfg.min_replay == 7_500

    def test_the_replay_flags_keep_the_defaults_when_omitted(self):
        cfg = _cfg_from_argv(["--arenas", "4"])

        assert cfg.replay_capacity == TrainConfig().replay_capacity
        assert cfg.min_replay == TrainConfig().min_replay

    @pytest.mark.parametrize(
        "argv,match",
        [
            (["--replay-capacity", "0"], "replay_capacity must be > 0"),
            (["--min-replay", "-1"], "min_replay must be >= 0"),
        ],
    )
    def test_bad_replay_values_are_rejected(self, argv, match):
        with pytest.raises(ValueError, match=match):
            _cfg_from_argv(argv)

    def test_both_per_actor_eps_flag_forms_parse(self):
        assert _parse([]).per_actor_eps is None
        assert _parse(["--per-actor-eps"]).per_actor_eps is True
        assert _parse(["--no-per-actor-eps"]).per_actor_eps is False

    def test_the_off_switch_reaches_the_config_and_the_gate(self):
        from agent.train import per_actor_eps_enabled

        off = _cfg_from_argv(["--arenas", str(CHOSEN_PADS), "--no-per-actor-eps"])
        on = _cfg_from_argv(["--arenas", str(CHOSEN_PADS)])

        assert off.per_actor_eps is False
        assert per_actor_eps_enabled(off) is False
        assert on.per_actor_eps is True
        assert per_actor_eps_enabled(on) is True

    def test_alpha_lands_on_the_config_and_changes_the_spread(self):
        from agent.train import per_actor_epsilon

        cfg = _cfg_from_argv(
            ["--arenas", "4", "--opponent", "scripted", "--per-actor-eps-alpha", "3.5"]
        )

        assert cfg.per_actor_eps_alpha == 3.5
        assert per_actor_epsilon(
            0.5, 3, 4, cfg.per_actor_eps_alpha
        ) != pytest.approx(per_actor_epsilon(0.5, 3, 4, 7.0))

    @pytest.mark.parametrize("bad", ["0", "-2"])
    def test_alpha_zero_is_not_an_off_switch(self, bad):
        # alpha=0 would flatten every arena onto one epsilon, which is what
        # --no-per-actor-eps already means. Two spellings of one behavior is how
        # a config ends up meaning something nobody intended.
        with pytest.raises(ValueError, match="per_actor_eps_alpha must be > 0"):
            _cfg_from_argv(["--per-actor-eps-alpha", bad])

    def test_the_documented_launch_command_produces_a_sane_schedule(self):
        # STATUS-2026-08-16's overnight command, end to end through the parser.
        from agent.train import eps_floor_fraction_of_run, per_actor_eps_enabled

        cfg = _cfg_from_argv(
            [
                "--arenas", "25",
                "--opponent", "scripted",
                "--warm-start", "runs/m2_multi.pt",
                "--checkpoint-every-grad-steps", "1000",
            ]
        )

        assert eps_floor_fraction_of_run(cfg) == pytest.approx(
            EPS_DECAY_FRACTION_OF_RUN, rel=0.01
        )
        assert per_actor_eps_enabled(cfg) is True
        assert cfg.warm_start == "runs/m2_multi.pt"

    def test_the_run_records_the_exploration_regime_it_used(self):
        # Run provenance is a documented weak spot here: code_version has no
        # --dirty and the config fingerprint ignores these knobs, so a run that
        # does not log them cannot be reconstructed afterwards.
        import inspect

        from agent import train as train_module

        source = inspect.getsource(train_module.main)
        for key in (
            "eps_decay_episodes",
            "eps_floor_fraction_of_run",
            "per_actor_eps",
            "replay_capacity",
            "min_replay",
        ):
            assert f'"{key}"' in source, f"the run never logs {key}"
