"""dqn — Dueling-DRQN network architecture (T14).

Implements the Dueling Deep Recurrent Q-Network from the training spec §4:

    obs (OBS_DIM) ─► MLP encoder: Linear(OBS_DIM,256)→ReLU→Linear(256,256)→ReLU
                  ─► LSTM(input=256, hidden=256, num_layers=1)   (DRQN — memory)
                  ─► Dueling heads:  V(s): Linear(256,1)
                                     A(s,a): Linear(256,N_ACTIONS)
                  ─► Q(s,a) = V(s) + (A(s,a) − mean_a' A(s,a'))   (dueling aggregation)

The network is intentionally small (§4, §11): the throughput bottleneck is the
Minecraft servers, not the gradient step, so a tiny MLP+LSTM saturates the GPU.

------------------------------------------------------------------------------
Recurrence is mandatory (§ partial observability)
------------------------------------------------------------------------------
The observation is partially observed: the opponent block is gated out by the
PerceptionFilter once the opponent leaves the FOV cone / line-of-sight. Whether
the agent should re-acquire, predict, or retreat depends on *how long ago* and
*where* it last saw the opponent — information that only survives in the LSTM
hidden state. The MLP encoder alone cannot solve this; TC8b is the unit fixture
that proves the LSTM (not the MLP) carries that memory.

------------------------------------------------------------------------------
Burn-in (R2D2, spec §5.5)
------------------------------------------------------------------------------
When training on a length-``L`` sequence with a burn-in prefix of ``B`` steps,
the first ``B`` steps only **warm** the LSTM hidden state; the TD loss is
computed on the remaining ``L−B`` steps. The recommended recipe:

  1. Seed the hidden state from the stored hidden state captured at collection
     time (R2D2), or from zeros (acceptable fallback — see ``init_hidden``).
  2. Run the first ``B`` steps under ``torch.no_grad()`` to produce a *detached*
     seed hidden state — those steps contribute no gradient.
  3. Run the remaining ``L−B`` steps **with** grad enabled, starting from that
     detached seed; the loss/backward sees only these scored steps.

``forward_with_burn_in`` implements exactly this and returns Q **only** for the
scored steps, so the caller cannot accidentally backprop through the burn-in
prefix. ``forward`` is the plain full-sequence pass used when no burn-in split
is wanted (e.g. inference, or computing targets over the whole window).

Owner: T14 (DQN core track)
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from agent.actions import N_ACTIONS
from env.observation_spec import OBS_DIM

__all__ = ["DuelingDRQN", "ENCODER_HIDDEN", "LSTM_HIDDEN", "LSTM_LAYERS"]


# ---------------------------------------------------------------------------
# Architecture constants (spec §4). Small by design.
# ---------------------------------------------------------------------------
#: Width of both MLP encoder layers.
ENCODER_HIDDEN: int = 256
#: LSTM hidden size (also the dueling-head input width).
LSTM_HIDDEN: int = 256
#: Number of stacked LSTM layers (DRQN uses a single layer).
LSTM_LAYERS: int = 1


# A (h0, c0) pair of shape (num_layers, batch, hidden) each.
HiddenState = Tuple[torch.Tensor, torch.Tensor]


class DuelingDRQN(nn.Module):
    """Dueling Deep Recurrent Q-Network (spec §4).

    Forward signature operates on **sequences**: an input of shape
    ``(B, T, OBS_DIM)`` produces Q-values of shape ``(B, T, N_ACTIONS)`` plus the
    LSTM hidden state after the last step, so a caller can chain windows.

    Attributes:
        obs_dim: Frozen input width — asserted equal to ``OBS_DIM``.
        n_actions: Q-head width — asserted equal to ``N_ACTIONS``.
        encoder_hidden / lstm_hidden / lstm_layers: architecture sizes.
    """

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        n_actions: int = N_ACTIONS,
        encoder_hidden: int = ENCODER_HIDDEN,
        lstm_hidden: int = LSTM_HIDDEN,
        lstm_layers: int = LSTM_LAYERS,
    ) -> None:
        super().__init__()

        # Freeze guard: the net must agree with the frozen contracts. A drift in
        # OBS_DIM or N_ACTIONS is a contract change and must fail loudly here.
        if obs_dim != OBS_DIM:
            raise ValueError(
                f"DuelingDRQN input dim {obs_dim} != OBS_DIM ({OBS_DIM}); "
                "the net must match the frozen observation contract."
            )
        if n_actions != N_ACTIONS:
            raise ValueError(
                f"DuelingDRQN output width {n_actions} != N_ACTIONS ({N_ACTIONS}); "
                "the Q head must match the frozen action contract."
            )

        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.encoder_hidden = encoder_hidden
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers

        # --- MLP encoder: Linear→ReLU→Linear→ReLU -------------------------
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, encoder_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(encoder_hidden, encoder_hidden),
            nn.ReLU(inplace=True),
        )

        # --- LSTM head (DRQN, batch_first so input is (B, T, feat)) -------
        self.lstm = nn.LSTM(
            input_size=encoder_hidden,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
        )

        # --- Dueling heads -----------------------------------------------
        self.value_head = nn.Linear(lstm_hidden, 1)
        self.advantage_head = nn.Linear(lstm_hidden, n_actions)

    # ------------------------------------------------------------------
    # Hidden-state helpers
    # ------------------------------------------------------------------
    def init_hidden(
        self, batch_size: int, device: Optional[torch.device] = None
    ) -> HiddenState:
        """Return a zeroed LSTM hidden state ``(h0, c0)``.

        Each tensor has shape ``(lstm_layers, batch_size, lstm_hidden)``, matching
        ``torch.nn.LSTM``'s expected ``(num_layers, N, H)`` layout. Use this for
        zero-init burn-in (the acceptable fallback when no stored hidden state is
        available) or as the start of an episode rollout.

        Args:
            batch_size: Number of independent sequences (the LSTM's ``N``).
            device: Target device; defaults to the module's parameter device so
                the state lands on the same device as the weights.

        Returns:
            ``(h0, c0)`` tuple of zero tensors.

        Raises:
            ValueError: if ``batch_size`` is not a positive integer.
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        if device is None:
            device = next(self.parameters()).device

        shape = (self.lstm_layers, batch_size, self.lstm_hidden)
        h0 = torch.zeros(shape, device=device)
        c0 = torch.zeros(shape, device=device)
        return h0, c0

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------
    @staticmethod
    def _dueling_aggregate(value: torch.Tensor, advantage: torch.Tensor) -> torch.Tensor:
        """Combine value/advantage streams via the dueling identity.

        ``Q(s,a) = V(s) + (A(s,a) − mean_a' A(s,a'))``. Subtracting the mean
        advantage removes the unidentifiable V/A offset (the standard dueling
        aggregation; mean is more stable than max).

        Args:
            value: ``(..., 1)`` state value.
            advantage: ``(..., N_ACTIONS)`` per-action advantage.

        Returns:
            ``(..., N_ACTIONS)`` Q-values.
        """
        advantage_mean = advantage.mean(dim=-1, keepdim=True)
        return value + (advantage - advantage_mean)

    def _heads(self, features: torch.Tensor) -> torch.Tensor:
        """Apply the dueling heads to LSTM features ``(..., lstm_hidden)``."""
        value = self.value_head(features)
        advantage = self.advantage_head(features)
        return self._dueling_aggregate(value, advantage)

    def forward(
        self, obs_seq: torch.Tensor, hidden: Optional[HiddenState] = None
    ) -> Tuple[torch.Tensor, HiddenState]:
        """Run the full sequence and return ``(q_seq, new_hidden)``.

        Args:
            obs_seq: Observations of shape ``(B, T, OBS_DIM)``.
            hidden: Optional ``(h0, c0)`` seed. ``None`` zero-inits (per the LSTM
                default), which is equivalent to ``init_hidden(B)``.

        Returns:
            ``q_seq`` of shape ``(B, T, N_ACTIONS)`` and the LSTM hidden state
            ``(h_n, c_n)`` after the final step (for chaining the next window).

        Raises:
            ValueError: if ``obs_seq`` is not 3-D or its last dim != ``OBS_DIM``.
        """
        if obs_seq.dim() != 3:
            raise ValueError(
                f"obs_seq must be (B, T, OBS_DIM), got shape {tuple(obs_seq.shape)}"
            )
        if obs_seq.shape[-1] != self.obs_dim:
            raise ValueError(
                f"obs_seq last dim {obs_seq.shape[-1]} != OBS_DIM ({self.obs_dim})"
            )

        batch, seq_len, _ = obs_seq.shape

        # Encode every timestep. Flatten (B, T) → (B*T) for the MLP, then restore.
        encoded = self.encoder(obs_seq.reshape(batch * seq_len, self.obs_dim))
        encoded = encoded.reshape(batch, seq_len, self.encoder_hidden)

        if hidden is None:
            hidden = self.init_hidden(batch, device=obs_seq.device)

        lstm_out, new_hidden = self.lstm(encoded, hidden)  # (B, T, lstm_hidden)
        q_seq = self._heads(lstm_out)  # (B, T, N_ACTIONS)
        return q_seq, new_hidden

    def forward_with_burn_in(
        self,
        obs_seq: torch.Tensor,
        burn_in: int,
        hidden: Optional[HiddenState] = None,
    ) -> Tuple[torch.Tensor, HiddenState, HiddenState]:
        """R2D2 burn-in forward (spec §5.5).

        Splits a length-``L`` sequence into a ``burn_in`` prefix and the scored
        ``L − burn_in`` suffix. The prefix runs under ``torch.no_grad()`` to
        produce a **detached** seed hidden state — its steps contribute no
        gradient — and the suffix runs with grad from that detached seed. Q is
        returned for the **scored steps only**, so the caller physically cannot
        backprop through the burn-in steps.

        Seeding: pass the stored collection-time hidden state via ``hidden``
        (R2D2 recipe), or leave it ``None`` for the zero-init fallback.

        Args:
            obs_seq: Observations of shape ``(B, L, OBS_DIM)``.
            burn_in: Number of warm-up steps ``B`` with ``0 <= B < L``.
            hidden: Optional ``(h0, c0)`` seed for the very first step.

        Returns:
            Tuple ``(q_scored, seed_hidden, final_hidden)`` where:
              - ``q_scored`` has shape ``(B, L − burn_in, N_ACTIONS)`` and carries
                gradient only through the scored steps,
              - ``seed_hidden`` is the detached hidden state after the burn-in
                prefix (what seeded the scored pass),
              - ``final_hidden`` is the hidden state after the final scored step.

        Raises:
            ValueError: on a bad shape or an out-of-range ``burn_in``.
        """
        if obs_seq.dim() != 3:
            raise ValueError(
                f"obs_seq must be (B, L, OBS_DIM), got shape {tuple(obs_seq.shape)}"
            )
        if obs_seq.shape[-1] != self.obs_dim:
            raise ValueError(
                f"obs_seq last dim {obs_seq.shape[-1]} != OBS_DIM ({self.obs_dim})"
            )

        batch, seq_len, _ = obs_seq.shape
        if not 0 <= burn_in < seq_len:
            raise ValueError(
                f"burn_in must satisfy 0 <= burn_in < L ({seq_len}), got {burn_in}"
            )

        if hidden is None:
            hidden = self.init_hidden(batch, device=obs_seq.device)

        # --- Burn-in prefix: warm the hidden state, no gradient ----------
        if burn_in > 0:
            with torch.no_grad():
                _, seed_hidden = self.forward(obs_seq[:, :burn_in], hidden)
            # Detach so the scored pass starts from a constant seed (no graph
            # leaks back into the burn-in steps even if grad were enabled).
            seed_hidden = (seed_hidden[0].detach(), seed_hidden[1].detach())
        else:
            # No burn-in: the seed is the (already-detached/zeroed) input hidden.
            seed_hidden = (hidden[0].detach(), hidden[1].detach())

        # --- Scored suffix: full gradient from the detached seed ---------
        q_scored, final_hidden = self.forward(obs_seq[:, burn_in:], seed_hidden)
        return q_scored, seed_hidden, final_hidden

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    @torch.no_grad()
    def act(
        self,
        obs: torch.Tensor,
        hidden: Optional[HiddenState],
        epsilon: float,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[int, HiddenState]:
        """ε-greedy single-step action selection (no gradient).

        Advances the LSTM by exactly one step. With probability ``epsilon`` a
        uniform-random macro is chosen; otherwise the greedy ``argmax_a Q``.

        Args:
            obs: A single observation. Accepts shape ``(OBS_DIM,)``,
                ``(1, OBS_DIM)`` or ``(1, 1, OBS_DIM)``; all are treated as one
                step for one sequence.
            hidden: Current LSTM hidden state, or ``None`` to zero-init (start of
                an episode).
            epsilon: Exploration rate in ``[0, 1]``.
            generator: Optional ``torch.Generator`` for deterministic sampling of
                both the explore/exploit coin flip and the random action.

        Returns:
            ``(action, new_hidden)`` — ``action`` is a Python ``int`` in
            ``[0, N_ACTIONS)``; ``new_hidden`` is the advanced LSTM state.

        Raises:
            ValueError: if ``obs`` cannot be coerced to a single step of width
                ``OBS_DIM``, or if ``epsilon`` is outside ``[0, 1]``.
        """
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError(f"epsilon must be in [0, 1], got {epsilon}")

        obs_seq = self._coerce_single_step(obs)  # (1, 1, OBS_DIM)
        device = obs_seq.device

        if hidden is None:
            hidden = self.init_hidden(1, device=device)

        q_seq, new_hidden = self.forward(obs_seq, hidden)  # (1, 1, N_ACTIONS)
        q_values = q_seq[0, 0]  # (N_ACTIONS,)

        # Short-circuit for pure-greedy: no RNG draw at all, so the global torch
        # RNG state is left completely untouched (important for test isolation).
        if epsilon == 0.0:
            action = int(torch.argmax(q_values).item())
            return action, new_hidden

        # Explore/exploit coin flip — generator-driven for reproducibility.
        explore = (
            torch.rand((), generator=generator, device=device).item() < epsilon
        )
        if explore:
            action = int(
                torch.randint(
                    self.n_actions, (), generator=generator, device=device
                ).item()
            )
        else:
            action = int(torch.argmax(q_values).item())

        return action, new_hidden

    def _coerce_single_step(self, obs: torch.Tensor) -> torch.Tensor:
        """Coerce ``obs`` to ``(1, 1, OBS_DIM)`` for a single inference step.

        Accepts ``(OBS_DIM,)``, ``(1, OBS_DIM)`` or ``(1, 1, OBS_DIM)``. Any other
        shape (e.g. a real batch or sequence) is a caller error for single-step
        ``act`` and raises.
        """
        if not torch.is_tensor(obs):
            obs = torch.as_tensor(obs, dtype=torch.float32)

        if obs.dim() == 1:
            view = obs.reshape(1, 1, -1)
        elif obs.dim() == 2 and obs.shape[0] == 1:
            view = obs.reshape(1, 1, -1)
        elif obs.dim() == 3 and obs.shape[0] == 1 and obs.shape[1] == 1:
            view = obs
        else:
            raise ValueError(
                "act expects a single observation of shape (OBS_DIM,), "
                f"(1, OBS_DIM) or (1, 1, OBS_DIM); got {tuple(obs.shape)}"
            )

        if view.shape[-1] != self.obs_dim:
            raise ValueError(
                f"obs last dim {view.shape[-1]} != OBS_DIM ({self.obs_dim})"
            )
        return view.to(dtype=torch.float32)
