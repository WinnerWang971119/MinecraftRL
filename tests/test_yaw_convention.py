"""The yaw/pitch wire convention — the pin that keeps the FOV cone pointing forward.

WHY THIS FILE EXISTS. Mineflayer and the Minecraft protocol measure yaw in
frames that are MIRRORED along ``z``, and mineflayer's pitch is sign-flipped too.
``bridge/bot.js`` shipped ``entity.yaw`` onto the wire raw, while
``env/perception_filter.py`` assumed the protocol convention it documents. The
result was not a subtle inaccuracy: the field-of-view gate ran back to front, so
``visible == 1`` held exactly when the agent was facing 180 degrees AWAY from the
opponent, and an opponent 8.4 degrees off the true look axis was reported at
171.6 degrees and zeroed out of the observation entirely.

Nothing in either language's test suite covered the seam, which is how it
shipped. These tests are that coverage. If a change makes them fail, the frame
has flipped back — do not "fix" the expected values to match the new output.

THE CONTRACT (``bridge/schema.md`` "the angle convention"):

    protocol_yaw   = normalizeYaw(pi - mineflayer_yaw)   # folded to (-pi, pi]
    protocol_pitch = -mineflayer_pitch

The four rows of :data:`CARDINAL_DIRECTIONS` are REAL MEASURED DATA, read off a
live bot driven with ``bot.lookAt`` toward each cardinal direction in turn. They
are the ground truth this whole module is built on; they are not derived from
the formula, so they can contradict it (and would, if the formula were wrong).
"""

import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from env.perception_filter import PerceptionFilter, RawState, _look_vector

REPO_ROOT = Path(__file__).resolve().parent.parent
BRIDGE_DIR = REPO_ROOT / "bridge"


# ---------------------------------------------------------------------------
# The measured ground truth.
#
# (label, mineflayer entity.yaw, protocol wire yaw, expected unit look vector)
#
# Read live, one `bot.lookAt` per row. Deliberately written as literals rather
# than computed from the conversion: a test that derives its own expectation
# from the code under test cannot catch that code being wrong.
# ---------------------------------------------------------------------------
CARDINAL_DIRECTIONS = [
    ("+z (south)", -math.pi, 0.0, (0.0, 0.0, 1.0)),
    ("-z (north)", 0.0, math.pi, (0.0, 0.0, -1.0)),
    ("+x (east)", -math.pi / 2, -math.pi / 2, (1.0, 0.0, 0.0)),
    ("-x (west)", math.pi / 2, math.pi / 2, (-1.0, 0.0, 0.0)),
]


class TestLookVectorAgainstMeasuredCardinals:
    """The four measured directions, pinned against ``_look_vector``."""

    @pytest.mark.parametrize(
        "label,mineflayer_yaw,protocol_yaw,expected_look",
        CARDINAL_DIRECTIONS,
        ids=[row[0] for row in CARDINAL_DIRECTIONS],
    )
    def test_converted_yaw_produces_the_measured_look_vector(
        self, label, mineflayer_yaw, protocol_yaw, expected_look
    ):
        """THE test. The wire value must aim `_look_vector` the measured way.

        If this fails, the perception filter's idea of "forward" no longer
        matches where the bot is actually looking, and every FOV decision the
        agent makes is wrong. There is no cosmetic way for this to break.
        """
        look = _look_vector(protocol_yaw, 0.0)

        np.testing.assert_allclose(
            look,
            np.asarray(expected_look, dtype=np.float64),
            atol=1e-9,
            err_msg=(
                f"facing {label}: the wire yaw {protocol_yaw!r} (converted from "
                f"mineflayer's {mineflayer_yaw!r}) must look toward "
                f"{expected_look}. A mismatch means the yaw convention flipped "
                f"— see bridge/schema.md 'the angle convention'. Do NOT update "
                f"this expectation to match the new output."
            ),
        )

    @pytest.mark.parametrize(
        "label,mineflayer_yaw,protocol_yaw,expected_look",
        CARDINAL_DIRECTIONS,
        ids=[row[0] for row in CARDINAL_DIRECTIONS],
    )
    def test_raw_mineflayer_yaw_mirrors_the_z_axis(
        self, label, mineflayer_yaw, protocol_yaw, expected_look
    ):
        """TEETH. Feeding the RAW mineflayer yaw mirrors z — the shipped bug.

        This is the failure mode itself, asserted rather than described, so the
        cardinal test above cannot be satisfied by a conversion that happens to
        be a no-op.
        """
        mirrored = _look_vector(mineflayer_yaw, 0.0)
        expected_mirrored = np.asarray(
            [expected_look[0], expected_look[1], -expected_look[2]],
            dtype=np.float64,
        )

        np.testing.assert_allclose(mirrored, expected_mirrored, atol=1e-9)


class TestOpponentInFrontIsVisible:
    """The regression test for the actual demo-blocking bug."""

    @staticmethod
    def _filter():
        # `los_clear` left at the default (open flat arena, nothing occludes),
        # so visibility here is decided by the FOV cone alone — which is the
        # thing under test.
        return PerceptionFilter()

    def test_opponent_dead_ahead_is_visible_and_lands_forward(self):
        """Converted wire yaw + opponent straight ahead => seen, and in front.

        Before the bridge conversion this exact case produced ``visible == 0``
        (pinned by the companion test below), which is what zeroed the whole
        opponent block for the entire demo.
        """
        me = RawState(pos=(0.0, 0.0, 0.0), yaw=0.0, pitch=0.0, velocity=(0.0, 0.0, 0.0))
        # Straight ahead along +z, comfortably inside the melee range.
        opponent = RawState(
            pos=(0.0, 0.0, 3.0), yaw=math.pi, pitch=0.0, velocity=(0.0, 0.0, 0.0)
        )

        gated, derived = self._filter().filter(me, opponent, dt=0.2)

        assert gated.visible is True, (
            "an opponent directly in front of the agent must be visible; "
            "visible == 0 here is the mirrored-FOV bug"
        )
        assert gated.pos_local[2] > 0.0, (
            "an opponent in front must land with a POSITIVE forward component "
            f"in pos_local, got {gated.pos_local!r}"
        )
        # Dead ahead means no lateral or vertical offset at all.
        assert gated.pos_local[0] == pytest.approx(0.0, abs=1e-9)
        assert gated.pos_local[1] == pytest.approx(0.0, abs=1e-9)
        # ...and dead ahead is, by definition, inside the crosshair cone.
        assert derived.in_crosshair is True
        assert derived.in_range is True

    def test_the_unconverted_wire_value_zeroes_the_same_opponent(self):
        """TEETH: the identical geometry, with the RAW mineflayer yaw, is blind.

        This pins WHY the bridge must convert. ``-pi`` is what mineflayer
        reports for a bot facing ``+z`` — the opponent is genuinely dead ahead,
        and the filter reports it absent. If this test ever passes with
        ``visible is True``, the conversion has been applied twice somewhere.
        """
        me = RawState(
            pos=(0.0, 0.0, 0.0), yaw=-math.pi, pitch=0.0, velocity=(0.0, 0.0, 0.0)
        )
        opponent = RawState(
            pos=(0.0, 0.0, 3.0), yaw=0.0, pitch=0.0, velocity=(0.0, 0.0, 0.0)
        )

        gated, derived = self._filter().filter(me, opponent, dt=0.2)

        assert gated.visible is False
        assert derived.in_crosshair is False
        assert gated.pos_local == (0.0, 0.0, 0.0)

    def test_pitch_sign_puts_a_lower_opponent_below_not_above(self):
        """The pitch half of the conversion, at the level the filter sees it.

        Protocol pitch is positive looking DOWN, so a positive pitch must aim
        the look vector at a NEGATIVE y. Mineflayer's is positive looking up;
        shipping it raw tilts every FOV decision the wrong way vertically.
        """
        look = _look_vector(0.0, math.pi / 4)

        assert look[1] < 0.0, (
            "protocol pitch is positive looking DOWN, so pitch=+pi/4 must give "
            f"a negative y component; got {look!r}"
        )
        np.testing.assert_allclose(look[1], -math.sin(math.pi / 4), atol=1e-9)


@pytest.mark.skipif(
    shutil.which("node") is None, reason="node is not installed on this machine"
)
class TestBridgeAgreesWithPython:
    """Cross-language pin: the REAL ``bot.js`` conversion, fed to the REAL filter.

    Every other test in this file asserts one side of the wire. This one closes
    the loop by calling the shipped JavaScript, because the bug lived in neither
    side alone — it lived in the seam, where a Python-only and a Node-only suite
    both stayed green.
    """

    @staticmethod
    def _convert_in_node(mineflayer_yaws, mineflayer_pitches):
        """Run the ACTUAL bot.js exports and return their output as floats."""
        script = (
            "const bot = require('./bot.js');"
            "const yaws = JSON.parse(process.argv[1]);"
            "const pitches = JSON.parse(process.argv[2]);"
            "process.stdout.write(JSON.stringify({"
            "  yaw: yaws.map(bot.toProtocolYaw),"
            "  pitch: pitches.map(bot.toProtocolPitch),"
            "}));"
        )
        completed = subprocess.run(
            [
                "node",
                "-e",
                script,
                json.dumps(list(mineflayer_yaws)),
                json.dumps(list(mineflayer_pitches)),
            ],
            cwd=str(BRIDGE_DIR),
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )
        return json.loads(completed.stdout)

    def test_bot_js_conversion_matches_the_measured_table(self):
        converted = self._convert_in_node(
            [row[1] for row in CARDINAL_DIRECTIONS], [0.0, 0.5, -0.5, math.pi / 2]
        )

        for (label, _mine, protocol_yaw, expected_look), actual_yaw in zip(
            CARDINAL_DIRECTIONS, converted["yaw"]
        ):
            assert actual_yaw == pytest.approx(protocol_yaw, abs=1e-9), (
                f"bot.js must convert the {label} reading to {protocol_yaw!r}"
            )
            np.testing.assert_allclose(
                _look_vector(actual_yaw, 0.0),
                np.asarray(expected_look, dtype=np.float64),
                atol=1e-9,
                err_msg=f"bot.js output must aim the filter toward {label}",
            )

        assert converted["pitch"] == pytest.approx(
            [0.0, -0.5, 0.5, -math.pi / 2], abs=1e-9
        )

    def test_bot_js_output_always_lands_in_the_canonical_range(self):
        # Several turns' worth of accumulated yaw in both directions, plus both
        # wrap boundaries, all of which must fold into (-pi, pi].
        raw = [
            0.0,
            math.pi,
            -math.pi,
            math.pi - 1e-12,
            -math.pi + 1e-12,
            7.0,
            -7.0,
            3 * math.pi,
            -3 * math.pi,
            100.0,
            -100.0,
        ]

        converted = self._convert_in_node(raw, [0.0])["yaw"]

        for source, value in zip(raw, converted):
            assert -math.pi < value <= math.pi + 1e-12, (
                f"bot.js emitted {value!r} for mineflayer yaw {source!r}, "
                "outside the canonical (-pi, pi] wire range"
            )
            # ...and the fold must not change which way the bot is facing.
            np.testing.assert_allclose(
                _look_vector(value, 0.0),
                _look_vector(math.pi - source, 0.0),
                atol=1e-9,
                err_msg=f"normalizing {source!r} changed the look direction",
            )
