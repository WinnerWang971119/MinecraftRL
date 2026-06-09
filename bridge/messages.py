"""messages — Python-side message schema helpers for the Node↔Python TCP bridge.

Provides dataclasses (or TypedDicts) for every JSON-lines message type that
the bridge exchanges: step events, reset requests, reset confirmations, and
action commands.  Must round-trip cleanly with bridge/schema.json.  Imported
by mc_pvp_env.py to parse incoming events and serialize outgoing actions.

Owner: T3 (Environment/bridge track)
# TODO(T3): implemented by task T3
"""

# TODO(T3): implemented by task T3
