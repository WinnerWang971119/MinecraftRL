"""transport — ExperienceTransport interface seam for distributed actor/learner.

Defines the abstract transport contract that decouples actor workers from the
learner.  Only LocalTransport (single-process, in-memory queue) is built
during kickoff; ZeroMQ/Redis transports are deferred to the post-M2 scaling
phase.

DEFERRED: full distributed transport is out of scope for kickoff.
Only the interface definition and LocalTransport stub are provided here.
"""

from __future__ import annotations

import abc
from typing import Any


class ExperienceTransport(abc.ABC):
    """Abstract base class for experience transport between actors and learner.

    All implementations must be thread-safe (or process-safe, if using separate
    processes).  The unit of transfer is a single experience batch dict as
    produced by the replay buffer's sample() method.
    """

    @abc.abstractmethod
    def send(self, batch: dict[str, Any]) -> None:
        """Send an experience batch from an actor to the learner.

        Args:
            batch: A dict of numpy arrays representing a sampled sequence
                   batch (keys: obs, actions, rewards, dones, weights, etc.).

        Raises:
            TransportError: If the underlying channel is closed or full.
        """

    @abc.abstractmethod
    def recv(self) -> dict[str, Any]:
        """Receive the next experience batch on the learner side.

        Blocks until a batch is available.

        Returns:
            A dict of numpy arrays as produced by send().

        Raises:
            TransportError: If the underlying channel is closed.
        """

    @abc.abstractmethod
    def close(self) -> None:
        """Shut down the transport and release any underlying resources."""


class TransportError(RuntimeError):
    """Raised when a transport operation fails."""


# NOTE: Only LocalTransport (single-process, in-memory) will be built
# during kickoff.  Multi-process / networked transports are deferred.
# DEFERRED(distributed): LocalTransport and remote transports implemented post-M2
