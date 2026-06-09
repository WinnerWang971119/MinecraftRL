"""perception_filter — FOV cone + raycast LoS + memory gating for opponent features.

Filters raw opponent state so that position, facing, and velocity are only
exposed when the opponent is within the agent's FOV cone AND the raycast
line-of-sight is clear.  When the opponent is occluded, the filter substitutes
the last-known position with a decaying time_since_seen counter.  Includes the
leak-detection battery required by AC5.

Owner: T12 (Environment/bridge track)
# TODO(T12): implemented by task T12
"""

# TODO(T12): implemented by task T12
