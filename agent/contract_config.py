"""contract_config — Frozen version pins and interface constants.

Written only after the day-1 compat check (Tv) confirms compatible versions.
Stores: Minecraft version string, ACTION_REPEAT ticks, MAX_EPISODE_STEPS,
code_version stamp, Node/Python version assertions, and any other constant
that all workstreams must agree on.  Importing this module at startup makes
a version mismatch a hard error rather than a silent bug.

Owner: T6 (DQN core track / shared contract)
# TODO(T6): implemented by task T6
"""

# TODO(T6): implemented by task T6
