"""train — n-step Double-DQN training loop for the Dueling-DRQN (T16).

Drives the full training cycle (training-spec §5 / §8):

  per episode:  env.reset(seed) -> net.init_hidden -> rollout with per-EPISODE
                ε-greedy ``act`` (collecting transitions + per-step hidden states)
                -> replay.add_episode
  per step (once ``len(replay) >= min_replay``):
                replay.sample_sequences -> n-step Double-DQN target -> Huber
                (smooth-L1) loss weighted by PER IS weights, on the post-burn-in
                scored steps only -> backward -> grad-norm clip -> optimizer step
                -> replay.update_priorities(|δ|) -> soft (Polyak) target update.

Reads every hyperparameter from :class:`agent.train_config.TrainConfig`; the
sequence/burn-in geometry and the n-step/gamma are shared with the replay buffer
so storage and learner can never disagree.

------------------------------------------------------------------------------
The exact target (training-spec §5/§8)
------------------------------------------------------------------------------
For a sampled window of ``T = burn_in + seq_len`` transitions, transition ``i``
is ``(s_i, a_i, r_i, s'_i, done_i)`` (``s'_i`` is ``next_obs[i]``, the state at
time ``i + 1``). For each SCORED step ``i`` in ``[B, T)``:

    G_i = Σ_{k=0}^{n-1} γ^k r_{i+k}        (truncated at the first done in-window)
    a*  = argmax_a Q_online(s_{i+n}, a)    (Double-DQN: ONLINE selects the action)
    y_i = G_i + bootstrap_i · γ^n · Q_target(s_{i+n}, a*)   (TARGET evaluates it)
    δ_i = y_i − Q_online(s_i, a_i)

``G_i`` and ``bootstrap_i`` come from :func:`agent.replay.compute_n_step_returns`
(the network-free reward arithmetic) so the bootstrap term is dropped exactly
where a ``done`` truncates the return or the n-step horizon runs off the window.

Because the net is RECURRENT, ``Q_target(s_{i+n}, ·)`` must be evaluated with the
LSTM hidden state that has consumed the FULL contiguous history s_0..s_{i+n}, not
a state seeded from the wrong stream. Within an episode the window is contiguous
(``next_obs[:, :-1] == obs[:, 1:]``), so we build the extended observation stream
``obs_ext = concat(obs, next_obs[:, -1:])`` of length ``T+1`` — where
``obs_ext[p] == s_p`` for ``p`` in ``[0, T]`` — and run ONE seeded forward of the
online and target nets over it (under ``no_grad``). The bootstrap state ``s_{i+n}``
then sits at ``obs_ext`` position ``i + n``, and its Q uses the correct recurrent
memory by construction.

The loss is the IS-weighted Huber (smooth-L1) of ``δ`` over the scored steps:
``L = mean(w_i · Huber(δ_i))``. The scored-step Q comes from
``DuelingDRQN.forward_with_burn_in`` so gradients flow ONLY through the
post-burn-in steps (the burn-in prefix warms the hidden state under ``no_grad``).

Owner: T16 (DQN core track)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Protocol, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from agent.actions import N_ACTIONS
from agent.dqn import DuelingDRQN
from agent.replay import PrioritizedSequenceReplay, SequenceBatch
from agent.seeding import seed_everything
from agent.train_config import TrainConfig

__all__ = [
    "EnvProtocol",
    "Trainer",
    "train",
    "epsilon_for_episode",
    "LearnStats",
]


# ---------------------------------------------------------------------------
# Env seam — the minimal Gym-style surface the trainer depends on.
#
# ``env.mc_pvp_env.MCPvPEnv`` satisfies this; so does the tiny in-test fake env
# (no socket / no live server). Keeping it a Protocol lets the smoke test inject
# a fake without constructing a bridge transport.
# ---------------------------------------------------------------------------


class EnvProtocol(Protocol):
    """Structural Gym-style env the trainer rolls out against."""

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Start an episode and return the initial observation ``(OBS_DIM,)``."""
        ...

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Advance one decision step -> ``(obs, reward, done, info)``."""
        ...


# ---------------------------------------------------------------------------
# Hooks — call signatures only (eval is T19, full M2 training is T20).
# ---------------------------------------------------------------------------

#: Called every ``cfg.eval_interval`` gradient steps. T19 owns the body.
EvalHook = Callable[["Trainer", int], None]
#: Called every ``cfg.checkpoint_interval`` gradient steps. T20 owns the body.
CheckpointHook = Callable[["Trainer", int], None]
#: Called every ``cfg.log_interval`` gradient steps with the latest learn stats.
LogHook = Callable[["Trainer", int, "LearnStats"], None]


@dataclass
class LearnStats:
    """Per-gradient-step learner diagnostics (handed to the log hook).

    Attributes:
        grad_step: 1-based index of this gradient step.
        loss: Scalar IS-weighted Huber loss for the batch.
        td_error_mean: Mean absolute TD error over the scored batch (pre-priority).
        grad_norm: Global gradient L2 norm BEFORE clipping (clip reduces it to
            at most ``cfg.grad_clip``).
        epsilon: ε used for the most recently collected episode.
        beta: Current PER importance-sampling exponent β.
        replay_size: Stored transitions at the time of this step.
    """

    grad_step: int
    loss: float
    td_error_mean: float
    grad_norm: float
    epsilon: float
    beta: float
    replay_size: int


# ---------------------------------------------------------------------------
# ε schedule — per EPISODE (NOT per step). See agent.seeding's gotcha note.
# ---------------------------------------------------------------------------


def epsilon_for_episode(episode: int, cfg: TrainConfig) -> float:
    """Return the ε-greedy exploration rate for episode index ``episode``.

    Linear decay from ``cfg.eps_start`` to ``cfg.eps_end`` over the FIRST
    ``cfg.eps_decay_episodes`` episodes, then flat at ``cfg.eps_end``. The decay
    is per EPISODE on purpose: episodes are short (tens of decisions), so a
    per-step decay would collapse ε in a handful of episodes and kill
    exploration (the documented gotcha in :mod:`agent.seeding`).

    Args:
        episode: 0-based episode index (>= 0). Clamped at 0 for safety.
        cfg: The training config holding the ε schedule.

    Returns:
        ε in ``[cfg.eps_end, cfg.eps_start]``, monotonically non-increasing in
        ``episode`` and flat within a single episode.
    """
    ep = max(0, int(episode))
    # frac goes 0 -> 1 over the first eps_decay_episodes, then saturates at 1.
    frac = min(ep / float(cfg.eps_decay_episodes), 1.0)
    return cfg.eps_start + (cfg.eps_end - cfg.eps_start) * frac


# ---------------------------------------------------------------------------
# The trainer.
# ---------------------------------------------------------------------------


class Trainer:
    """n-step Double-DQN learner for the Dueling-DRQN with PER + R2D2 burn-in.

    Owns the online net, the target net (a soft-tracked copy), the optimizer, and
    the prioritized sequence replay. :meth:`collect_episode` rolls one episode and
    stores it; :meth:`learn` performs ONE gradient step from a sampled batch;
    :meth:`train` runs the full episode/step loop and fires the eval/checkpoint/log
    hooks at their configured cadences.

    The online and target nets are constructed identically (same architecture
    kwargs) so a soft update is well defined parameter-for-parameter.

    Args:
        cfg: Training hyperparameters. The sequence/burn-in geometry and
            n-step/gamma are forwarded to the replay buffer so storage and learner
            agree.
        device: Torch device for the nets/tensors. Defaults to CPU (the dev box
            has a CPU-only torch wheel).
        net_kwargs: Optional architecture overrides forwarded to BOTH
            ``DuelingDRQN`` constructions (e.g. smaller hidden sizes for a fast
            unit test). ``obs_dim``/``n_actions`` still assert against the frozen
            contracts inside the net.
        seed_global: When True (default) call :func:`agent.seeding.seed_everything`
            with ``cfg.seed`` at construction so the whole run is reproducible.
    """

    def __init__(
        self,
        cfg: TrainConfig,
        *,
        device: Optional[torch.device] = None,
        net_kwargs: Optional[Dict[str, int]] = None,
        seed_global: bool = True,
    ) -> None:
        self.cfg = cfg
        self.device = torch.device("cpu") if device is None else torch.device(device)

        # Reproducible run: seed Python/NumPy/torch before any net init draws RNG.
        if seed_global:
            seed_everything(cfg.seed)

        net_kwargs = dict(net_kwargs or {})
        self.online = DuelingDRQN(**net_kwargs).to(self.device)
        self.target = DuelingDRQN(**net_kwargs).to(self.device)
        # Target starts as an exact copy of the online net (θ_target = θ_online).
        self.target.load_state_dict(self.online.state_dict())
        # The target net is never optimized directly — only soft-updated. Freezing
        # its grads avoids building a needless autograd graph through it.
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.target.eval()

        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=cfg.lr)

        # Replay geometry MUST mirror the config (seq_len == scored L, burn_in,
        # n_step, gamma) so sampled windows match what the loss expects. Seed the
        # buffer's sampler from cfg.seed for reproducible PER draws.
        self.replay = PrioritizedSequenceReplay(
            capacity=cfg.replay_capacity,
            seq_len=cfg.seq_len,
            burn_in=cfg.burn_in,
            alpha=cfg.per_alpha,
            beta0=cfg.per_beta0,
            priority_eps=cfg.per_priority_eps,
            n_step=cfg.n_step,
            gamma=cfg.gamma,
            beta_anneal_steps=cfg.per_beta_anneal_steps,
            rng=np.random.default_rng(cfg.seed),
        )

        # Per-episode action-sampling RNG. Re-seeded deterministically each
        # episode (seed + episode) so exploration is replayable.
        self._action_generator = torch.Generator(device=self.device)

        self.grad_step = 0  # number of completed gradient steps
        self.episode_count = 0  # number of collected episodes
        self.last_epsilon = cfg.eps_start  # ε of the most recent episode

    # ------------------------------------------------------------------
    # Rollout / collection
    # ------------------------------------------------------------------
    def collect_episode(
        self, env: EnvProtocol, *, max_steps: Optional[int] = None
    ) -> Tuple[int, float]:
        """Roll ONE episode with per-episode ε-greedy and store it in replay.

        Reseeds the env and the action-sampling generator deterministically from
        ``cfg.seed + episode_index`` so the rollout is reproducible. Each decision
        uses ``DuelingDRQN.act`` (no grad) with the episode's ε and the carried
        LSTM hidden state; the per-step hidden state captured at COLLECTION time is
        stored alongside the transition so the buffer can seed burn-in (R2D2).

        Args:
            env: The environment to roll out against (real or fake).
            max_steps: Optional hard cap on decisions this episode (defence
                against a fake env that never terminates). ``None`` relies on the
                env's own termination.

        Returns:
            ``(n_transitions, total_reward)`` for the collected episode.
        """
        episode_index = self.episode_count
        epsilon = epsilon_for_episode(episode_index, self.cfg)
        self.last_epsilon = epsilon

        # Deterministic per-episode seeds for the env reset and the ε RNG so the
        # exploration stream is replayable (the gotcha fix: reseed on the episode
        # boundary, not per step).
        episode_seed = self.cfg.seed + episode_index
        self._action_generator.manual_seed(episode_seed)

        obs = env.reset(seed=episode_seed)
        hidden = self.online.init_hidden(1, device=self.device)

        transitions: List[Tuple[np.ndarray, int, float, np.ndarray, bool]] = []
        hidden_states: List[np.ndarray] = []

        total_reward = 0.0
        done = False
        steps = 0
        was_training = self.online.training
        self.online.eval()  # inference mode for action selection
        try:
            while not done:
                # Capture the hidden state SEEN by this step (the LSTM state that
                # produced the action), stacked (num_layers, hidden) for burn-in
                # seeding by the replay buffer.
                hidden_snapshot = self._hidden_snapshot(hidden)

                obs_tensor = torch.as_tensor(
                    obs, dtype=torch.float32, device=self.device
                )
                action, hidden = self.online.act(
                    obs_tensor,
                    hidden,
                    epsilon=epsilon,
                    generator=self._action_generator,
                )

                next_obs, reward, done, _info = env.step(action)

                transitions.append(
                    (
                        np.asarray(obs, dtype=np.float32),
                        int(action),
                        float(reward),
                        np.asarray(next_obs, dtype=np.float32),
                        bool(done),
                    )
                )
                hidden_states.append(hidden_snapshot)

                total_reward += float(reward)
                obs = next_obs
                steps += 1
                if max_steps is not None and steps >= max_steps:
                    break
        finally:
            if was_training:
                self.online.train()

        if transitions:
            self.replay.add_episode(transitions, hidden_states=hidden_states)

        self.episode_count += 1
        return len(transitions), total_reward

    def _hidden_snapshot(self, hidden: Tuple[torch.Tensor, torch.Tensor]) -> np.ndarray:
        """Snapshot the LSTM ``(h, c)`` for the current single-sequence step.

        Returns a contiguous ``float32`` array of shape
        ``(2, num_layers, lstm_hidden)`` (the (h, c) pair) detached to NumPy. The
        replay buffer stores one per transition and hands the window-start snapshot
        back so T16 can seed burn-in instead of zeroing it.
        """
        h, c = hidden
        # Drop the batch axis (== 1 during rollout) and stack h/c on a new axis 0.
        h_np = h.detach().squeeze(1).cpu().numpy()  # (num_layers, lstm_hidden)
        c_np = c.detach().squeeze(1).cpu().numpy()  # (num_layers, lstm_hidden)
        return np.stack((h_np, c_np), axis=0).astype(np.float32)

    # ------------------------------------------------------------------
    # Learning — one gradient step
    # ------------------------------------------------------------------
    def ready_to_learn(self) -> bool:
        """True iff replay has >= ``min_replay`` transitions AND a sampleable window."""
        return self.replay.is_ready(self.cfg.min_replay)

    def learn(self) -> Optional[LearnStats]:
        """Run ONE n-step Double-DQN gradient step from a sampled batch.

        Returns ``None`` (a no-op) if the buffer is not yet ready (``len(replay)
        < min_replay`` or no sampleable window). Otherwise:

          1. sample a prioritized batch of ``B + L`` windows,
          2. compute the IS-weighted Huber loss on the scored ``L`` steps using
             the n-step Double-DQN target,
          3. backward, clip the global grad norm to ``cfg.grad_clip``, step,
          4. write the fresh |δ| back as priorities, and
          5. soft-update the target net.

        Returns:
            A :class:`LearnStats` for this gradient step, or ``None`` if not ready.
        """
        if not self.ready_to_learn():
            return None

        cfg = self.cfg
        # Anneal PER β by gradient step BEFORE sampling so this batch's IS weights
        # use the up-to-date β.
        self.replay.anneal_beta(self.grad_step)

        batch = self.replay.sample_sequences(cfg.batch_size, L=cfg.seq_len)

        loss, abs_td_per_sample, mean_abs_td = self._compute_loss(batch)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.online.parameters(), cfg.grad_clip
        )
        self.optimizer.step()

        # Priorities: one per sampled window, from the latest |δ| (mean over the
        # window's scored steps — a single priority per window is what the buffer
        # stores per start index).
        self.replay.update_priorities(batch.indices, abs_td_per_sample)

        # Soft target update every step: θ_target ← τ·θ_online + (1−τ)·θ_target.
        if cfg.target_soft:
            self._soft_update_target()

        self.grad_step += 1

        return LearnStats(
            grad_step=self.grad_step,
            loss=float(loss.detach().cpu().item()),
            td_error_mean=float(mean_abs_td),
            grad_norm=float(grad_norm),
            epsilon=self.last_epsilon,
            beta=float(self.replay.beta),
            replay_size=len(self.replay),
        )

    def _compute_loss(
        self, batch: SequenceBatch
    ) -> Tuple[torch.Tensor, np.ndarray, float]:
        """Build the IS-weighted Huber loss for one sampled batch.

        Implements the n-step Double-DQN target exactly (see the module docstring):
        ONLINE selects the bootstrap action, TARGET evaluates it, the burn-in
        prefix is detached, and the loss is computed on the scored ``L`` steps
        only.

        Returns:
            ``(loss, abs_td_per_sample, mean_abs_td)`` where ``loss`` is the scalar
            graph tensor, ``abs_td_per_sample`` is a ``(batch,)`` NumPy array of
            per-window mean |δ| (for ``update_priorities``), and ``mean_abs_td`` is
            the scalar batch-mean |δ| (for logging).
        """
        cfg = self.cfg
        device = self.device
        burn_in = batch.burn_in
        # Full window length T = burn_in + seq_len (the buffer guarantees this).
        obs = torch.as_tensor(batch.obs, dtype=torch.float32, device=device)
        next_obs = torch.as_tensor(batch.next_obs, dtype=torch.float32, device=device)
        actions = torch.as_tensor(batch.actions, dtype=torch.long, device=device)
        is_weights = torch.as_tensor(
            batch.is_weights, dtype=torch.float32, device=device
        )

        bsz, window, _ = obs.shape
        scored = window - burn_in  # == seq_len (L)

        # --- n-step discounted returns + bootstrap mask (network-free) -----
        # Shapes (batch, window). G_i and the mask are aligned with absolute step
        # index i in [0, window); we use only the scored slice [burn_in:].
        returns_np, bootstrap_np = self.replay.n_step_returns(
            batch, n=cfg.n_step, gamma=cfg.gamma
        )
        returns = torch.as_tensor(returns_np, dtype=torch.float32, device=device)
        bootstrap = torch.as_tensor(
            bootstrap_np.astype(np.float32), dtype=torch.float32, device=device
        )

        # --- seed hidden state from the stored collection-time hidden ------
        # batch.hidden is (batch, 2, num_layers, lstm_hidden) when present (our
        # collector stores the (h, c) pair). Seed the very first window step.
        seed_hidden = self._seed_hidden_from_batch(batch, bsz)

        # --- online Q on the scored steps (gradients flow here) ------------
        q_scored, _, _ = self.online.forward_with_burn_in(
            obs, burn_in, hidden=seed_hidden
        )  # (batch, L, N_ACTIONS)
        scored_actions = actions[:, burn_in:]  # (batch, L)
        q_taken = q_scored.gather(-1, scored_actions.unsqueeze(-1)).squeeze(-1)
        # (batch, L) — Q_online(s_i, a_i) for scored i.

        # --- bootstrap states: a SINGLE correctly-recurrent forward --------
        # The bootstrap state for scored step i (absolute window index i in
        # [B, T)) is s_{i+n}. For a DRQN the bootstrap Q must be evaluated with the
        # hidden state that incorporates the FULL contiguous history up to s_{i+n},
        # NOT a hidden seeded from the wrong stream (the old code ran the LSTM over
        # ``next_obs`` from the window-START hidden, which is shifted by one step
        # and missing obs[B] — biasing every n-step target whenever memory matters).
        #
        # Within an episode the window is contiguous, so next_obs[:, :-1] == obs[:,
        # 1:]. Build an EXTENDED stream obs_ext = concat(obs, next_obs[:, -1:]) of
        # shape (b, T+1, OBS_DIM): obs_ext[p] == s_p for p in [0, T] (obs_ext[T] ==
        # next_obs[T-1] == s_T). Running ONE seeded forward over obs_ext from the
        # collection-time window-start hidden makes the Q at position p use the
        # hidden that incorporates s_0..s_p — exactly the correct memory for s_p.
        # The bootstrap (a target) carries no gradient, so the whole pass is under
        # no_grad.
        with torch.no_grad():
            obs_ext = torch.cat([obs, next_obs[:, -1:, :]], dim=1)  # (b, T+1, OBS_DIM)
            q_ext_online, _ = self.online(obs_ext, seed_hidden)  # (b, T+1, A)
            q_ext_target, _ = self.target(obs_ext, seed_hidden)  # (b, T+1, A)
            # Double-DQN per position: ONLINE selects a*, TARGET evaluates it.
            next_actions = q_ext_online.argmax(dim=-1)  # (b, T+1)
            q_ext_eval = q_ext_target.gather(
                -1, next_actions.unsqueeze(-1)
            ).squeeze(-1)  # (b, T+1) target value of the online-greedy action at s_p

            # For scored step i the bootstrap state s_{i+n} sits at obs_ext
            # position i + n. bootstrap_i is False whenever that horizon ran off the
            # window or a done truncated the return, so the clamped (in-range)
            # gather there is masked to 0.
            boot_value = self._gather_bootstrap_values(
                q_ext_eval, burn_in, scored, cfg.n_step
            )  # (batch, L)

            returns_scored = returns[:, burn_in:]  # (batch, L)
            bootstrap_scored = bootstrap[:, burn_in:]  # (batch, L)
            gamma_n = cfg.gamma ** cfg.n_step
            target_y = returns_scored + bootstrap_scored * gamma_n * boot_value
            # (batch, L)

        # --- TD error + IS-weighted Huber loss -----------------------------
        # Derive BOTH the priority |δ| and the Huber loss from the SAME ``td_error``
        # tensor so the value that drives the update can never diverge from the one
        # that drives the priority (Huber(δ) is a deterministic function of δ).
        td_error = target_y - q_taken  # (batch, L)
        # smooth-L1 (Huber) per scored step, no reduction, computed FROM td_error.
        huber = F.smooth_l1_loss(
            td_error, torch.zeros_like(td_error), reduction="none"
        )  # (batch, L)
        # Mean Huber per window, then weight by the window's IS weight, then mean.
        per_window_huber = huber.mean(dim=1)  # (batch,)
        loss = (is_weights * per_window_huber).mean()

        # Per-window |δ| (mean over scored steps) drives the new priority; detach.
        abs_td = td_error.detach().abs()  # (batch, L)
        abs_td_per_sample = abs_td.mean(dim=1).cpu().numpy()  # (batch,)
        mean_abs_td = float(abs_td.mean().cpu().item())

        return loss, abs_td_per_sample, mean_abs_td

    def _seed_hidden_from_batch(
        self, batch: SequenceBatch, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build the LSTM seed ``(h0, c0)`` from the stored window-start hidden.

        The collector stores a per-step ``(2, num_layers, lstm_hidden)`` snapshot;
        the buffer returns the window-start one as ``batch.hidden`` of shape
        ``(batch, 2, num_layers, lstm_hidden)``. We split it back into the
        ``(num_layers, batch, lstm_hidden)`` layout ``torch.nn.LSTM`` expects. When
        no hidden was stored (``batch.hidden is None``) we fall back to a zero seed
        — an acceptable burn-in fallback per the net's docstring.
        """
        if batch.hidden is None:
            return self.online.init_hidden(batch_size, device=self.device)

        hidden = np.asarray(batch.hidden, dtype=np.float32)
        # Expected layout (batch, 2, num_layers, lstm_hidden). Be defensive about a
        # collapsed num_layers axis (batch, 2, lstm_hidden) for a single-layer net.
        if hidden.ndim == 3:
            hidden = hidden[:, :, np.newaxis, :]  # -> (batch, 2, 1, lstm_hidden)
        if hidden.ndim != 4 or hidden.shape[1] != 2:
            # Unexpected snapshot shape — fall back to a zero seed rather than risk
            # feeding a mis-shaped hidden into the LSTM.
            return self.online.init_hidden(batch_size, device=self.device)

        h = torch.as_tensor(hidden[:, 0], dtype=torch.float32, device=self.device)
        c = torch.as_tensor(hidden[:, 1], dtype=torch.float32, device=self.device)
        # (batch, num_layers, lstm_hidden) -> (num_layers, batch, lstm_hidden).
        h = h.permute(1, 0, 2).contiguous()
        c = c.permute(1, 0, 2).contiguous()
        return h, c

    @staticmethod
    def _gather_bootstrap_values(
        q_ext_eval: torch.Tensor, burn_in: int, scored: int, n_step: int
    ) -> torch.Tensor:
        """Select, per scored step ``i``, the bootstrap value at ``obs_ext`` ``i+n``.

        ``q_ext_eval`` is ``(batch, T+1)`` — the target value of the online-greedy
        action at EVERY extended-stream position (``obs_ext[p]`` == state ``s_p``).
        For scored step ``i`` (absolute index ``burn_in + s`` for ``s`` in
        ``[0, scored)``) the bootstrap state ``s_{i+n}`` is ``obs_ext[i + n]``. We
        clamp that index into ``[0, T]`` and rely on the caller's ``bootstrap`` mask
        to zero any term where the true horizon ran past the window end, so the
        clamped (in-range) gather is never actually used there.

        Returns:
            ``(batch, scored)`` bootstrap values aligned with the scored steps.
        """
        bsz, ext_len = q_ext_eval.shape  # ext_len == T + 1
        # Absolute scored indices i = burn_in + [0..scored).
        i = torch.arange(scored, device=q_ext_eval.device) + burn_in  # (scored,)
        boot_idx = i + n_step  # obs_ext position of s_{i+n}
        # Non-mutating clamp into the extended stream (max valid index is T == ext_len-1).
        boot_idx = boot_idx.clamp(max=ext_len - 1)  # keep gather in range (masked)
        boot_idx = boot_idx.unsqueeze(0).expand(bsz, scored)  # (batch, scored)
        return q_ext_eval.gather(1, boot_idx)

    def _soft_update_target(self) -> None:
        """Polyak soft update: ``θ_target ← τ·θ_online + (1−τ)·θ_target``.

        Applied to every parameter AND buffer of the target net every gradient
        step. Runs under ``no_grad`` (the target is never differentiated).
        """
        tau = self.cfg.tau
        with torch.no_grad():
            for tgt_p, src_p in zip(
                self.target.parameters(), self.online.parameters()
            ):
                tgt_p.mul_(1.0 - tau).add_(src_p, alpha=tau)
            # Buffers (none in this net, but future-proof) are copied verbatim so
            # they never go stale.
            for tgt_b, src_b in zip(self.target.buffers(), self.online.buffers()):
                tgt_b.copy_(src_b)

    # ------------------------------------------------------------------
    # Full loop
    # ------------------------------------------------------------------
    def train(
        self,
        env: EnvProtocol,
        num_episodes: int,
        *,
        updates_per_step: int = 1,
        max_episode_steps: Optional[int] = None,
        eval_hook: Optional[EvalHook] = None,
        checkpoint_hook: Optional[CheckpointHook] = None,
        log_hook: Optional[LogHook] = None,
    ) -> None:
        """Run the full episode/gradient loop for ``num_episodes`` episodes.

        For each episode: collect it (per-EPISODE ε), then — once the buffer is
        ready — run ``updates_per_step`` gradient steps. The eval/checkpoint/log
        hooks fire at their configured cadences (``cfg.eval_interval`` etc.); a
        cadence of 0 disables that hook. The hooks are call-signature stubs here:
        T19 owns eval, T20 owns checkpoint/full training.

        Args:
            env: The environment to roll out against.
            num_episodes: Number of episodes to collect/train on (>= 0).
            updates_per_step: Gradient steps to run per collected episode once
                ready (>= 0). 1 keeps the update-to-data ratio near one update per
                episode; raise for more gradient steps per env episode.
            max_episode_steps: Optional per-episode decision cap (for a fake env).
            eval_hook / checkpoint_hook / log_hook: Optional callbacks; see the
                hook type aliases. Called only when their interval is > 0 and the
                current grad step is a multiple of it.

        Raises:
            ValueError: if ``num_episodes`` or ``updates_per_step`` is negative.
        """
        if num_episodes < 0:
            raise ValueError(f"num_episodes must be >= 0, got {num_episodes}")
        if updates_per_step < 0:
            raise ValueError(
                f"updates_per_step must be >= 0, got {updates_per_step}"
            )

        for _ in range(num_episodes):
            self.collect_episode(env, max_steps=max_episode_steps)

            for _ in range(updates_per_step):
                stats = self.learn()
                if stats is None:
                    break  # not ready yet; skip remaining updates this episode
                self._fire_hooks(stats, eval_hook, checkpoint_hook, log_hook)

    def _fire_hooks(
        self,
        stats: LearnStats,
        eval_hook: Optional[EvalHook],
        checkpoint_hook: Optional[CheckpointHook],
        log_hook: Optional[LogHook],
    ) -> None:
        """Invoke eval/checkpoint/log hooks whose interval divides the grad step."""
        step = self.grad_step
        if log_hook is not None and self.cfg.log_interval > 0:
            if step % self.cfg.log_interval == 0:
                log_hook(self, step, stats)
        if eval_hook is not None and self.cfg.eval_interval > 0:
            if step % self.cfg.eval_interval == 0:
                eval_hook(self, step)
        if checkpoint_hook is not None and self.cfg.checkpoint_interval > 0:
            if step % self.cfg.checkpoint_interval == 0:
                checkpoint_hook(self, step)


# ---------------------------------------------------------------------------
# Functional entry point.
# ---------------------------------------------------------------------------


def train(
    env: EnvProtocol,
    cfg: TrainConfig,
    num_episodes: int,
    *,
    device: Optional[torch.device] = None,
    net_kwargs: Optional[Dict[str, int]] = None,
    updates_per_step: int = 1,
    max_episode_steps: Optional[int] = None,
    eval_hook: Optional[EvalHook] = None,
    checkpoint_hook: Optional[CheckpointHook] = None,
    log_hook: Optional[LogHook] = None,
) -> Trainer:
    """Construct a :class:`Trainer` and run ``num_episodes`` of the §8 loop.

    Convenience wrapper for callers that just want "train this env with this
    config". Returns the constructed :class:`Trainer` so the caller can inspect
    the nets / replay / grad-step count afterwards (eval is T19, checkpoint T20).

    Args:
        env: The environment to roll out against (real ``MCPvPEnv`` or a fake).
        cfg: The training hyperparameters.
        num_episodes: Episodes to collect/train on.
        device / net_kwargs: Forwarded to the :class:`Trainer` constructor.
        updates_per_step / max_episode_steps / *_hook: Forwarded to
            :meth:`Trainer.train`.

    Returns:
        The :class:`Trainer` after the loop completes.
    """
    trainer = Trainer(cfg, device=device, net_kwargs=net_kwargs)
    trainer.train(
        env,
        num_episodes,
        updates_per_step=updates_per_step,
        max_episode_steps=max_episode_steps,
        eval_hook=eval_hook,
        checkpoint_hook=checkpoint_hook,
        log_hook=log_hook,
    )
    return trainer
