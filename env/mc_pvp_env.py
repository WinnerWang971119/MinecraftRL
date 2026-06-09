"""mc_pvp_env — Gym-style Minecraft PvP environment wrapper.

Implements the reset()/step()/done interface over the Node↔Python TCP bridge.
Translates raw JSON-lines bridge events into observation vectors (via
observation_spec) and scalar rewards (via reward).  Manages episode lifecycle,
enforces MAX_EPISODE_STEPS, and calls the bridge reset RPC with the read-back
gate.

Owner: T9 (Environment/bridge track)
# TODO(T9): implemented by task T9
"""

# TODO(T9): implemented by task T9
