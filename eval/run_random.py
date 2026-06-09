"""run_random — M1 tracer bullet: random policy end-to-end smoke run.

Drives at least 100 full episodes through the real Paper server and bridge
using the uniform random policy (agent/random_policy.py) vs the idle dummy.
Monitors for crashes and tracks combined Node + Python + JVM RSS growth.
Satisfies AC3 when it completes 100 episodes with zero crashes and RSS growth
< ~200 MB across the run.

Owner: T10 (Eval/infra track)
# TODO(T10): implemented by task T10
"""

# TODO(T10): implemented by task T10
