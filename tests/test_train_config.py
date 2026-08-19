"""Tests for the self-play config schema T11a adds to ``TrainConfig``.

T11a lands EARLY in the M4 self-play plan on purpose: T9 and T10 build on these
fields and defaults before any CLI plumbing (T11b) exists to set them from the
command line, so this file pins the DATACLASS contract in isolation — no
snapshot pool, no driver, no bridge, no Minecraft.

What is pinned here, and why each test exists:

* **``"selfplay"`` joins ``_OPPONENT_CHOICES`` without disturbing the existing
  two.** ``dummy`` and ``scripted`` are every training path that exists today;
  a regression in the choice set would break them silently for every run
  already in flight. Pinned as an explicit non-regression check, not just an
  incidental pass.
* **AC14 — the cross-field rule that matters most.** ``opponent == "selfplay"``
  requires ``warm_start is not None``. Without it the snapshot pool has no
  policy to seed snapshot 0 from, and — absent this check — a launched run
  would only discover that hours in, as "pool empty: refuse to start" at
  runtime instead of at config construction (TC28).
* **TC29's SHAPE half, plus the converse of AC14.** The plan splits the
  warm-start checksum feature in two: T11a owns the field and validates its
  SHAPE (a well-formed SHA-256 hex digest or ``None``); T11b owns comparing it
  against the checkpoint's actual digest. Only the shape half is exercised
  here. A digest set WITHOUT a ``warm_start`` is refused for the same reason
  AC14 exists: there would be no checkpoint to compare against, so T11b's gate
  would silently skip and an operator who pasted a checksum but forgot
  ``--warm-start`` would get zero verification and zero error.
* **Types, not just values.** ``reference_promote_grad_steps`` must be a
  ``tuple`` of ``int``. An ``argparse`` flag with ``nargs=2`` yields a LIST,
  which clears every value check and then makes this ``frozen=True`` dataclass
  unhashable (``hash(cfg)`` raises far from the config); float entries clear
  every value check and then never match T18's ``grad_step == promote_first``
  equality, so references 2 and 3 are silently never created. A hash test
  pins the first failure mode directly.
* **``elo_k`` / ``elo_initial`` must be FINITE, not merely ordered.** ``+inf``
  passes ``not x > 0`` — it really is greater than zero — so ``math.isfinite``
  is the only guard that catches it, and an infinite K-factor turns the first
  rated match into inf/nan and empties ``elo/learner_rated``. The int fields
  deliberately keep the file's plain ordered-comparison idiom.
* **The exact-zero Elo footgun (flagged in T8 review).** ``opponent_epsilon``
  is the FROZEN opponent's exploration rate, not the eval-time epsilon that
  ``MatchResult.rated_eligible`` gates on. A denormal value here (e.g.
  ``1e-18``) is a legal *training* epsilon — the field's own default, 0.02, is
  nonzero and correct — so validation does not reject it. What matters is
  documented on the field itself: eval cycles must pass a literal ``0.0``,
  never an epsilon-adjacent constant, or the entire Elo curve comes out empty
  with no error anywhere. A test below pins that current, intentional
  behavior so nobody "fixes" it into a range check that would also reject the
  legitimate default.
* **Every new numeric/tuple field gets its own boundary test**, matching the
  density of validation already in ``TrainConfig.__post_init__`` — every
  non-bool field there is guarded, and a config schema task with a gap in it
  is exactly the kind of thing that stays invisible until 3am.

No socket, no live server, no Minecraft: everything here is
``dataclasses.replace`` on a plain ``TrainConfig``.
"""

from __future__ import annotations

import dataclasses
import math

import pytest

from agent.train_config import TrainConfig

#: A checkpoint path good enough to satisfy ``warm_start``'s own non-empty
#: check; these tests never touch the filesystem, so it need not exist.
_WARM_START_PATH = "runs/m2_multi.pt"

#: A syntactically valid SHA-256 hex digest (64 lowercase hex chars) for
#: exercising the "well-formed" side of the shape check.
_VALID_SHA256 = "a" * 64


def _cfg(**overrides) -> TrainConfig:
    """A ``TrainConfig`` with any field overridable per test."""
    return dataclasses.replace(TrainConfig(), **overrides)


# ===========================================================================
# _OPPONENT_CHOICES gains "selfplay" without disturbing dummy/scripted
# ===========================================================================


class TestOpponentChoicesRegression:
    # The "default config is still dummy" and "an unknown opponent is refused"
    # cases live in tests/test_opponent_curriculum.py
    # (TestTrainConfigOpponentFields) and are not duplicated here; what is new
    # in T11a — and only pinned here — is that "selfplay" JOINED the choice set
    # without displacing the two paths already in flight.
    def test_dummy_still_constructs_unchanged(self):
        cfg = _cfg(opponent="dummy")
        assert cfg.opponent == "dummy"

    def test_scripted_still_constructs_unchanged(self):
        cfg = _cfg(opponent="scripted")
        assert cfg.opponent == "scripted"

    def test_selfplay_is_now_accepted_given_a_warm_start(self):
        cfg = _cfg(opponent="selfplay", warm_start=_WARM_START_PATH)
        assert cfg.opponent == "selfplay"

    def test_the_rejected_value_names_all_three_choices(self):
        with pytest.raises(ValueError) as excinfo:
            _cfg(opponent="bogus")
        message = str(excinfo.value)
        assert "'dummy'" in message
        assert "'scripted'" in message
        assert "'selfplay'" in message


# ===========================================================================
# AC14 / TC28 — selfplay requires a warm start
# ===========================================================================


class TestSelfplayRequiresWarmStart:
    def test_selfplay_without_warm_start_raises(self):
        with pytest.raises(ValueError, match="warm_start"):
            _cfg(opponent="selfplay", warm_start=None)

    def test_the_error_names_the_opponent_and_the_missing_field(self):
        with pytest.raises(ValueError, match="opponent='selfplay'"):
            _cfg(opponent="selfplay")

    def test_selfplay_with_a_warm_start_is_accepted(self):
        cfg = _cfg(opponent="selfplay", warm_start=_WARM_START_PATH)
        assert cfg.warm_start == _WARM_START_PATH

    def test_dummy_needs_no_warm_start(self):
        # Regression: the cross-field rule must not leak onto the other
        # opponents, which have always been legal with warm_start=None.
        cfg = _cfg(opponent="dummy", warm_start=None)
        assert cfg.warm_start is None

    def test_scripted_needs_no_warm_start(self):
        cfg = _cfg(opponent="scripted", warm_start=None)
        assert cfg.warm_start is None


# ===========================================================================
# New field defaults (the Contracts block, verbatim)
# ===========================================================================


class TestNewFieldDefaults:
    def test_defaults_match_the_plan(self):
        cfg = TrainConfig()
        assert cfg.snapshot_every_grad_steps == 1000
        assert cfg.snapshot_sampling == "pfsp"
        assert cfg.opponent_epsilon == 0.02
        assert cfg.reference_promote_grad_steps == (5000, 15000)
        assert cfg.elo_k == 24
        assert cfg.elo_initial == 1000
        assert cfg.warm_start_sha256 is None

    def test_the_defaults_alone_construct_a_valid_config(self):
        # No override needed: opponent defaults to "dummy", so none of the
        # selfplay-only fields are even read, and every field must still pass
        # its own boundary check.
        TrainConfig()


# ===========================================================================
# snapshot_every_grad_steps
# ===========================================================================


class TestSnapshotEveryGradStepsValidation:
    @pytest.mark.parametrize("value", [0, -1, -1000])
    def test_non_positive_is_refused(self, value):
        with pytest.raises(
            ValueError, match="snapshot_every_grad_steps must be >= 1"
        ):
            _cfg(snapshot_every_grad_steps=value)

    def test_one_is_the_accepted_floor(self):
        cfg = _cfg(snapshot_every_grad_steps=1)
        assert cfg.snapshot_every_grad_steps == 1


# ===========================================================================
# snapshot_sampling
# ===========================================================================


class TestSnapshotSamplingValidation:
    @pytest.mark.parametrize("value", ["uniform", "pfsp"])
    def test_both_documented_modes_are_accepted(self, value):
        cfg = _cfg(snapshot_sampling=value)
        assert cfg.snapshot_sampling == value

    @pytest.mark.parametrize("value", ["random", "PFSP", "", "Uniform"])
    def test_anything_else_is_refused(self, value):
        with pytest.raises(ValueError, match="snapshot_sampling must be one of"):
            _cfg(snapshot_sampling=value)


# ===========================================================================
# opponent_epsilon
# ===========================================================================


class TestOpponentEpsilonValidation:
    @pytest.mark.parametrize("value", [-0.01, 1.01, float("nan")])
    def test_out_of_range_is_refused(self, value):
        with pytest.raises(ValueError, match="opponent_epsilon must be in"):
            _cfg(opponent_epsilon=value)

    @pytest.mark.parametrize("value", [0.0, 0.02, 1.0])
    def test_in_range_values_are_accepted(self, value):
        cfg = _cfg(opponent_epsilon=value)
        assert cfg.opponent_epsilon == value

    def test_a_denormal_nonzero_value_is_a_legal_training_epsilon(self):
        # Documents the deliberate design (T8 review): opponent_epsilon governs
        # the FROZEN opponent's TRAINING-time exploration, where any value in
        # [0, 1] — including a tiny one — is a legitimate epsilon; the field's
        # own default, 0.02, is itself nonzero. The exact-zero requirement
        # belongs to whatever passes epsilons into eval-time MatchResults
        # (T12), not to this field's own bound, so 1e-18 is accepted here
        # exactly as any other in-range float would be.
        cfg = _cfg(opponent_epsilon=1e-18)
        assert cfg.opponent_epsilon == 1e-18
        assert cfg.opponent_epsilon != 0.0


# ===========================================================================
# reference_promote_grad_steps
# ===========================================================================


class TestReferencePromoteGradStepsValidation:
    def test_the_default_is_a_strictly_increasing_pair(self):
        cfg = TrainConfig()
        first, second = cfg.reference_promote_grad_steps
        assert first < second

    @pytest.mark.parametrize(
        "value", [(5000,), (5000, 10000, 15000), (), (1,)]
    )
    def test_anything_other_than_a_pair_is_refused(self, value):
        with pytest.raises(
            ValueError, match="must have exactly 2 entries"
        ):
            _cfg(reference_promote_grad_steps=value)

    @pytest.mark.parametrize("value", [(0, 15000), (-5000, 15000), (5000, 0)])
    def test_a_non_positive_entry_is_refused(self, value):
        with pytest.raises(
            ValueError, match="reference_promote_grad_steps must all be positive"
        ):
            _cfg(reference_promote_grad_steps=value)

    @pytest.mark.parametrize("value", [(15000, 5000), (5000, 5000)])
    def test_a_non_increasing_pair_is_refused(self, value):
        with pytest.raises(
            ValueError, match="must be strictly increasing"
        ):
            _cfg(reference_promote_grad_steps=value)

    @pytest.mark.parametrize("value", [[5000, 15000], [], [1, 2, 3]])
    def test_a_list_is_refused(self, value):
        # Not hypothetical: `agent/train.py`'s _CONFIG_OVERRIDE_FLAGS applies a
        # per-field `cast`, and an argparse flag with `nargs=2` produces a
        # LIST. A list passes every value check (length, sign, ordering) and
        # then breaks somewhere else entirely — see the hash test below.
        with pytest.raises(
            ValueError, match="reference_promote_grad_steps must be a tuple"
        ):
            _cfg(reference_promote_grad_steps=value)

    @pytest.mark.parametrize(
        "value",
        [
            (5000.5, 15000.5),
            (5000.0, 15000.0),  # int-VALUED but still float
            (5000, 15000.0),  # only the second entry is wrong
            ("5000", "15000"),
            (5000, None),
        ],
    )
    def test_a_non_int_entry_is_refused(self, value):
        # T18 promotes on `grad_step == promote_first` — an equality against an
        # integer counter — so a float entry never fires and pinned references
        # 2 and 3 silently never exist, taking T13's Elo yardsticks with them.
        with pytest.raises(
            ValueError,
            match="reference_promote_grad_steps entries must all be int",
        ):
            _cfg(reference_promote_grad_steps=value)

    def test_the_offending_entry_is_named_in_the_message(self):
        with pytest.raises(ValueError) as excinfo:
            _cfg(reference_promote_grad_steps=(5000, 15000.5))
        message = str(excinfo.value)
        assert "reference_promote_grad_steps" in message
        assert "15000.5" in message

    def test_the_tuple_check_runs_before_the_length_check(self):
        # A list of the RIGHT length must still be refused as a list, so the
        # error points at the actual defect instead of passing silently.
        with pytest.raises(
            ValueError, match="reference_promote_grad_steps must be a tuple"
        ):
            _cfg(reference_promote_grad_steps=[5000, 15000])

    def test_a_valid_pair_is_accepted(self):
        cfg = _cfg(reference_promote_grad_steps=(100, 200))
        assert cfg.reference_promote_grad_steps == (100, 200)


# ===========================================================================
# The frozen dataclass stays hashable — what the tuple check protects
# ===========================================================================


class TestTrainConfigStaysHashable:
    def test_the_default_config_hashes(self):
        # `TrainConfig` is `@dataclass(frozen=True)`, so it hashes by field
        # values; one list-valued field is enough to make `hash(cfg)` raise
        # `TypeError: unhashable type: 'list'` at some far-away call site,
        # long after the config was accepted as valid.
        assert isinstance(hash(TrainConfig()), int)

    def test_a_fully_specified_selfplay_config_hashes(self):
        cfg = _cfg(
            opponent="selfplay",
            warm_start=_WARM_START_PATH,
            warm_start_sha256=_VALID_SHA256,
            reference_promote_grad_steps=(1000, 2000),
        )
        assert isinstance(hash(cfg), int)

    def test_equal_configs_hash_equal(self):
        assert hash(TrainConfig()) == hash(TrainConfig())


# ===========================================================================
# elo_k
# ===========================================================================


class TestEloKValidation:
    @pytest.mark.parametrize("value", [0.0, -1.0, -24.0, float("nan")])
    def test_non_positive_or_nan_is_refused(self, value):
        with pytest.raises(ValueError, match="elo_k must be finite and > 0"):
            _cfg(elo_k=value)

    @pytest.mark.parametrize("value", [float("inf"), float("-inf")])
    def test_an_infinite_k_factor_is_refused(self, value):
        # `+inf` PASSES `not x > 0` — it is genuinely greater than zero — so
        # only the finiteness half catches it. An infinite K-factor turns the
        # first rated match's rating into inf/nan and empties
        # `elo/learner_rated`, the headline deliverable of issue #10.
        with pytest.raises(ValueError, match="elo_k must be finite and > 0"):
            _cfg(elo_k=value)

    def test_the_default_matches_the_snapshot_pool_constant(self):
        # opponents.snapshot_pool.ELO_K = 24.0; kept as a literal here (not an
        # import) so this test does not itself depend on that module — only
        # the VALUE needs to agree.
        assert TrainConfig().elo_k == 24.0

    def test_a_positive_value_is_accepted(self):
        cfg = _cfg(elo_k=10.0)
        assert cfg.elo_k == 10.0


# ===========================================================================
# elo_initial
# ===========================================================================


class TestEloInitialValidation:
    @pytest.mark.parametrize("value", [-1.0, -1000.0, float("nan")])
    def test_negative_or_nan_is_refused(self, value):
        with pytest.raises(ValueError, match="elo_initial must be finite and >= 0"):
            _cfg(elo_initial=value)

    @pytest.mark.parametrize("value", [float("inf"), float("-inf")])
    def test_an_infinite_starting_rating_is_refused(self, value):
        # Same asymmetry as elo_k: `-inf` is caught by the ordered comparison,
        # `+inf` only by `math.isfinite`. A rating that starts at infinity
        # never moves and every Elo delta computed from it is nan.
        with pytest.raises(ValueError, match="elo_initial must be finite and >= 0"):
            _cfg(elo_initial=value)

    def test_zero_is_the_accepted_floor(self):
        cfg = _cfg(elo_initial=0.0)
        assert cfg.elo_initial == 0.0

    def test_the_default_is_the_conventional_baseline(self):
        assert TrainConfig().elo_initial == 1000.0


# ===========================================================================
# warm_start_sha256 (shape only — T11b owns the byte comparison, TC29)
# ===========================================================================


class TestWarmStartSha256ShapeValidation:
    def test_none_is_accepted(self):
        cfg = _cfg(warm_start_sha256=None)
        assert cfg.warm_start_sha256 is None

    def test_a_well_formed_digest_is_accepted(self):
        # `warm_start` comes along because a digest without one is refused
        # (see TestWarmStartSha256RequiresAWarmStart); the shape is what is
        # under test here.
        cfg = _cfg(warm_start_sha256=_VALID_SHA256, warm_start=_WARM_START_PATH)
        assert cfg.warm_start_sha256 == _VALID_SHA256

    def test_a_real_sha256_hexdigest_is_accepted(self):
        import hashlib

        digest = hashlib.sha256(b"warm start checkpoint bytes").hexdigest()
        cfg = _cfg(warm_start_sha256=digest, warm_start=_WARM_START_PATH)
        assert cfg.warm_start_sha256 == digest

    @pytest.mark.parametrize(
        "value",
        [
            "",
            "abc123",  # too short
            "a" * 63,  # one short of 64
            "a" * 65,  # one over 64
            "A" * 64,  # uppercase — hexdigest() never produces this
            "g" * 64,  # not a hex character
            ("a" * 63) + " ",  # trailing whitespace
        ],
    )
    def test_a_malformed_digest_is_refused(self, value):
        with pytest.raises(ValueError, match="warm_start_sha256 must be"):
            _cfg(warm_start_sha256=value, warm_start=_WARM_START_PATH)

    def test_the_shape_check_reports_before_the_missing_warm_start(self):
        # Both defects at once: the message must name the malformed digest,
        # not send the operator chasing the absent warm start.
        with pytest.raises(ValueError, match="warm_start_sha256 must be"):
            _cfg(warm_start_sha256="nope", warm_start=None)


# ===========================================================================
# warm_start_sha256 without a warm start — the converse of AC14
# ===========================================================================


class TestWarmStartSha256RequiresAWarmStart:
    def test_a_digest_without_a_warm_start_is_refused(self):
        # The digest exists to verify the checkpoint at `warm_start`. With no
        # warm start there is no file to hash, so T11b's gate would silently
        # skip: an operator who pastes a checksum and forgets `--warm-start`
        # would get zero verification and zero error.
        with pytest.raises(ValueError, match="warm_start_sha256 requires warm_start"):
            _cfg(warm_start_sha256=_VALID_SHA256, warm_start=None)

    def test_the_error_names_both_fields(self):
        with pytest.raises(ValueError) as excinfo:
            _cfg(warm_start_sha256=_VALID_SHA256, warm_start=None)
        message = str(excinfo.value)
        assert "warm_start_sha256" in message
        assert "warm_start=None" in message

    def test_a_digest_with_a_warm_start_is_accepted(self):
        cfg = _cfg(warm_start_sha256=_VALID_SHA256, warm_start=_WARM_START_PATH)
        assert cfg.warm_start_sha256 == _VALID_SHA256
        assert cfg.warm_start == _WARM_START_PATH

    def test_a_warm_start_without_a_digest_is_still_accepted(self):
        # Regression: the converse guard must not make the checksum mandatory
        # — plain `--warm-start` runs predate this field and stay legal.
        cfg = _cfg(warm_start=_WARM_START_PATH, warm_start_sha256=None)
        assert cfg.warm_start_sha256 is None

    def test_neither_field_set_is_the_untouched_default(self):
        cfg = TrainConfig()
        assert cfg.warm_start is None
        assert cfg.warm_start_sha256 is None


# ===========================================================================
# Integration across the new fields — a fully-specified selfplay config
# ===========================================================================


class TestFullySpecifiedSelfplayConfig:
    def test_every_new_field_together_constructs_cleanly(self):
        cfg = _cfg(
            opponent="selfplay",
            warm_start=_WARM_START_PATH,
            warm_start_sha256=_VALID_SHA256,
            snapshot_every_grad_steps=500,
            snapshot_sampling="uniform",
            opponent_epsilon=0.05,
            reference_promote_grad_steps=(1000, 2000),
            elo_k=32.0,
            elo_initial=1200.0,
        )
        assert cfg.opponent == "selfplay"
        assert cfg.snapshot_every_grad_steps == 500
        assert cfg.snapshot_sampling == "uniform"
        assert cfg.opponent_epsilon == 0.05
        assert cfg.reference_promote_grad_steps == (1000, 2000)
        assert cfg.elo_k == 32.0
        assert cfg.elo_initial == 1200.0
        assert cfg.warm_start_sha256 == _VALID_SHA256

    def test_finite_check_does_not_reject_the_plain_default_path(self):
        # Sanity: none of the new validation reaches into the dummy/scripted
        # paths and none of it trips on math.isnan for ordinary floats.
        cfg = TrainConfig()
        assert math.isfinite(cfg.elo_k)
        assert math.isfinite(cfg.elo_initial)
        assert math.isfinite(cfg.opponent_epsilon)
