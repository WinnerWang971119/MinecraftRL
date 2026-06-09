"""observation_spec — Frozen observation vector contract (single source of truth).

Defines the fixed-length float observation vector (~30-40 dims), the frozen
index map, and helper functions for packing/unpacking observations.  Every
component that reads or writes an observation (net, env, filter, actors) must
import from here rather than hardcode indices.

Owner: T2 (Environment/bridge track)
# TODO(T2): implemented by task T2
"""

# TODO(T2): implemented by task T2
