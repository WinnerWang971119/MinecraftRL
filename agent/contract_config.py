"""contract_config — Frozen version pins, timing constants, and the code_version stamp.

Written only after the day-1 compat check (Tv) confirmed compatible versions
(see ``server/compat_check.md``). This module is the **single source of truth**
for the cross-cutting constants that every workstream must agree on:

  - the pinned Minecraft / Paper / Node / Python / npm versions,
  - the decision/episode timing constants (ACTION_REPEAT, DECISION_INTERVAL_MS,
    MAX_EPISODE_STEPS),
  - the ``code_version()`` run stamp written onto bridge ``state`` messages and
    checkpoints so train/deploy can detect drift.

Importing this module costs nothing and pulls in **no heavy dependencies** — it
is stdlib-only (``subprocess``, ``hashlib``) on purpose so the bridge, the env,
the agent, eval, and any future distributed actor can all import it without
dragging in ``torch``/``numpy``. Do NOT add such imports here.

------------------------------------------------------------------------------
Ownership / non-duplication contract
------------------------------------------------------------------------------
This module owns **versions + timing + code_version only**. It deliberately does
NOT redefine constants that belong to other frozen artifacts:

  - ``env/observation_spec.py`` owns the observation layout (``OBS_DIM``,
    ``MEMORY_TTL_SECONDS``, ``POS_SCALE``, ``MAX_HEALTH``, the held-item vocab, …).
  - ``env/perception_filter.py`` owns the field-of-view / visibility knobs (FOV).
  - ``agent/actions.py`` owns the discrete action macro enum.
  - ``agent/train_config.py`` owns the trainable hyperparameters (lr, ε schedule…).

If you need an obs/FOV/training constant, import it from its owner — never copy
it here. Duplicating a constant across two modules is how train and deploy
silently drift apart.

Owner: T6 (DQN core track / shared contract)
"""

from __future__ import annotations

import hashlib
import subprocess
from typing import Dict, Tuple

__all__ = [
    "MINECRAFT_VERSION",
    "PAPER_VERSION",
    "PAPER_BUILD",
    "PAPER_JAR",
    "JAVA_MIN",
    "NODE_VERSION",
    "NODE_ENGINE_MIN",
    "PYTHON_MIN",
    "NPM_PINS",
    "ACTION_REPEAT",
    "DECISION_INTERVAL_MS",
    "SERVER_TPS",
    "MAX_EPISODE_STEPS",
    "code_version",
    "config_fingerprint",
]


# ---------------------------------------------------------------------------
# Version pins — CONFIRMED by the Tv compat check (server/compat_check.md).
#
# These are FROZEN. Changing any of them is a contract change: update this
# module, re-run the compat check, and re-freeze. The bridge and the agent both
# assert against these at startup so a mismatch is a hard error, not a silent
# protocol/handshake bug.
# ---------------------------------------------------------------------------

#: Minecraft protocol/data version the whole stack targets. All four Node
#: dependencies (mineflayer, minecraft-data, mineflayer-pvp, mineflayer-pathfinder)
#: support this; it is well past the 1.9 attack-cooldown combat cutover that the
#: PvP mechanics depend on.
MINECRAFT_VERSION: str = "1.21.1"

#: Paper server line (matches MINECRAFT_VERSION) and the specific build to pin.
#: The jar is ``paper-1.21.1-133.jar`` from PaperMC channel STABLE.
PAPER_VERSION: str = "1.21.1"
PAPER_BUILD: int = 133

#: Convenience: the exact server jar filename T8 installs under ``server/``.
PAPER_JAR: str = f"paper-{PAPER_VERSION}-{PAPER_BUILD}.jar"

#: Minimum Java major version Paper 1.21.1 requires. The dev machine runs Java 25.
JAVA_MIN: int = 21

#: Node version installed and verified on the dev machine (Node 24 "Krypton" LTS).
#: ``code_version`` / startup checks compare against this exact string for logging.
NODE_VERSION: str = "v24.13.0"

#: mineflayer's declared floor (``engines.node``). v24.13.0 is comfortably above it.
NODE_ENGINE_MIN: int = 22

#: Minimum supported Python (major, minor). The dev machine runs 3.14.2; the
#: project floor is 3.11 (matches ``pyproject.toml`` ``requires-python``).
PYTHON_MIN: Tuple[int, int] = (3, 11)

#: Exact npm pins for the bridge. These are the versions T7a tightens into
#: ``bridge/package.json``; recorded here so the contract has one authoritative
#: copy. minecraft-data is transitive via mineflayer but pinned for reproducibility.
NPM_PINS: Dict[str, str] = {
    "mineflayer": "4.37.1",
    "minecraft-data": "3.110.2",
    "mineflayer-pvp": "1.3.2",
    "mineflayer-pathfinder": "2.4.5",
}


# ---------------------------------------------------------------------------
# Decision / episode timing constants.
#
# The Paper server runs at a fixed 20 ticks per second (50 ms/tick). The agent
# does NOT act every tick — it holds each macro for ACTION_REPEAT ticks, giving a
# coarser, more stable decision cadence (frame-skip / action-repeat). These
# constants are shared by the env (step pacing), the bridge (control-state hold
# duration), and the trainer (episode length).
# ---------------------------------------------------------------------------

#: Vanilla server tick rate (ticks/second). Fixed by Minecraft, recorded so the
#: timing math below is self-documenting rather than relying on a magic 20.
SERVER_TPS: int = 20

#: Ticks each chosen macro is held before the next decision (frame-skip).
#: 4 ticks @ 20 TPS == 200 ms, the agent's decision interval. Movement macros
#: hold their control state for this many ticks (see ``agent/actions.py``).
ACTION_REPEAT: int = 4

#: Decision interval in milliseconds, derived from ACTION_REPEAT and SERVER_TPS so
#: it can never drift from them. == 4 / 20 * 1000 == 200 ms.
DECISION_INTERVAL_MS: int = ACTION_REPEAT * 1000 // SERVER_TPS

#: Max decisions (NOT ticks) per episode before truncation.
#:
#: TUNE — sizing math: at a sword DPS of ~5 health/s and a 20-health opponent, a
#: kill takes ~4 s of *landed* hits; allowing for missed swings, repositioning,
#: and a stationary/evasive dummy, a comfortable episode horizon is ~80 s. At
#: DECISION_INTERVAL_MS == 200 ms (5 decisions/s), 80 s == 400 decisions.
#:   400 decisions * 0.200 s/decision == 80 s of wall-clock combat.
#: Raise this if early curricula time out before a kill; lower it once the agent
#: reliably wins fast, to keep episodes short and throughput high.
MAX_EPISODE_STEPS: int = 400


# ---------------------------------------------------------------------------
# code_version — the run/version stamp.
#
# A stable string identifying *exactly* this build of the contract + code:
#
#     "<git-short-sha>+cfg<config-fingerprint>"   e.g. "c98b532+cfg1a2b3c4d"
#
# It is stamped onto every bridge ``state`` message and every checkpoint. The
# distributed future (deferred) will REJECT actors whose code_version does not
# match the learner's — that is how train/serve skew is caught. The kickoff stack
# does NOT reject; it only logs the value so mismatches are visible in the logs.
#
# Two components, so both kinds of drift are caught:
#   - the git short SHA catches any committed code change, and
#   - the config fingerprint (a hash of the frozen values in THIS module) catches
#     a constant being edited in a dirty/uncommitted tree, where the SHA would
#     otherwise be unchanged.
# ---------------------------------------------------------------------------

#: Falls back to this when the source tree is not a git checkout (e.g. a release
#: tarball) or git is not installed. The config fingerprint still varies, so the
#: stamp remains useful for detecting constant changes even without a SHA.
_NOGIT_SENTINEL: str = "nogit"


def _git_short_sha() -> str:
    """Return the short git SHA of HEAD, or ``"nogit"`` if unavailable.

    Captured via ``subprocess`` so this module stays dependency-free. Any failure
    mode — git not installed, not a repository, a detached/empty tree, a timeout —
    is swallowed and reported as the ``"nogit"`` sentinel rather than raising, so
    importing/stamping never crashes a run just because git is missing.
    """
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        # git binary missing, not executable, or the call itself blew up.
        return _NOGIT_SENTINEL

    if completed.returncode != 0:
        # Not a git repo, no commits yet, etc.
        return _NOGIT_SENTINEL

    sha = completed.stdout.strip()
    return sha if sha else _NOGIT_SENTINEL


def config_fingerprint() -> str:
    """Return a short, stable hash of the frozen constants defined in this module.

    The fingerprint is a truncated SHA-256 over a canonical text rendering of the
    version pins and timing constants. It is deterministic across processes and
    machines (no dict-ordering or address dependence) so the same config always
    yields the same fingerprint. Editing any frozen value below changes it.
    """
    # A canonical, order-stable rendering. NPM_PINS is sorted so dict insertion
    # order can never perturb the hash. Everything is stringified explicitly.
    parts = [
        f"minecraft={MINECRAFT_VERSION}",
        f"paper={PAPER_VERSION}",
        f"paper_build={PAPER_BUILD}",
        f"java_min={JAVA_MIN}",
        f"node={NODE_VERSION}",
        f"node_engine_min={NODE_ENGINE_MIN}",
        f"python_min={PYTHON_MIN[0]}.{PYTHON_MIN[1]}",
        f"server_tps={SERVER_TPS}",
        f"action_repeat={ACTION_REPEAT}",
        f"decision_interval_ms={DECISION_INTERVAL_MS}",
        f"max_episode_steps={MAX_EPISODE_STEPS}",
    ]
    parts.extend(f"npm.{name}={version}" for name, version in sorted(NPM_PINS.items()))
    canonical = "\n".join(parts).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()[:8]


def code_version() -> str:
    """Return the stable run/version stamp for this build.

    Format: ``"<git-short-sha>+cfg<config-fingerprint>"`` (e.g.
    ``"c98b532+cfg1a2b3c4d"``). The SHA half is ``"nogit"`` when the tree is not a
    git checkout. This is the value written onto bridge ``state`` messages and
    saved into checkpoints. In the (deferred) distributed setup a learner will
    reject actors whose ``code_version`` differs; the kickoff stack only logs it.
    """
    return f"{_git_short_sha()}+cfg{config_fingerprint()}"
