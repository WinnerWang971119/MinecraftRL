"""dist_config — Distributed-backend configuration for the multi-arena setup.

Holds the backend selector and any transport-level settings needed to wire
collectors to the learner. Right now only ``backend="local"`` is supported:
all arenas run as threads in the same process, sharing objects directly via
``queue.Queue`` and ``threading.Event``. No network, no serialization beyond
PyTorch state-dicts.

Route-2 fields (Redis, ZeroMQ, sharding) are declared here but left ``None``
or at their stub defaults so the dataclass stays forward-compatible. They are
intentionally unused until the Route-2 networked-transport milestone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

__all__ = ["DistConfig"]


@dataclass(frozen=True)
class DistConfig:
    """Frozen bundle of distributed-backend settings.

    Frozen for the same reason as :class:`agent.train_config.TrainConfig`:
    a config is a stable record of a run. Use ``dataclasses.replace`` to
    vary a field rather than mutating in place.
    """

    # -- backend selector -------------------------------------------------
    #: Transport backend. ``"local"`` == same-process threads sharing a
    #: ``queue.Queue``; no serialization beyond weight state-dicts. Only
    #: ``"local"`` is supported now. ``"redis"`` and ``"zeromq"`` are
    #: Route-2 future work.
    backend: Literal["local"] = "local"

    # -- Route-2 reserved fields (unused in local mode) -------------------
    # These fields are RESERVED for the Route-2 networked transport (Redis /
    # ZeroMQ). They are documented here so the dataclass interface is stable
    # when Route 2 lands; none of them are read by any current code path.

    #: Redis connection URL (e.g. ``"redis://localhost:6379"``). Only used
    #: when ``backend="redis"`` (Route 2). ``None`` == disabled.
    redis_url: str | None = None

    #: Redis pub/sub channel name on which the learner broadcasts weight
    #: snapshots to collectors. Only used when ``backend="redis"`` (Route 2).
    #: ``None`` == disabled.
    weight_sync_channel: str | None = None

    #: Number of replay-buffer shards spread across collector nodes. 1 ==
    #: no sharding (single buffer). Only meaningful when ``backend`` is a
    #: networked transport (Route 2); ignored in local mode.
    num_shards: int = 1

    def __post_init__(self) -> None:
        """Reject unsupported backends immediately so misconfiguration fails loudly."""
        if self.backend != "local":
            raise ValueError(
                f'backend must be "local" (the only supported backend now); '
                f"got {self.backend!r}. Redis/ZeroMQ support is Route-2 future work."
            )
        if self.num_shards < 1:
            raise ValueError(f"num_shards must be >= 1, got {self.num_shards}")
