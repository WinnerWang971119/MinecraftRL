"""Tests for the n-step Double-DQN training loop (T16).

Exercises ``agent.train`` end-to-end against a tiny in-file FAKE env (random
observations, terminates after ``k`` steps) so NO socket / live Minecraft server
is needed. The checks mirror the kickoff acceptance criteria for T16:

  * a few gradient steps run without error and the loss is finite,
  * the online parameters CHANGE after an update,
  * the target net MOVES TOWARD the online net (soft Polyak update) but is NOT
    equal to it after a single step,
  * ε decays per EPISODE (``epsilon_for_episode`` is monotonically non-increasing
    across episodes and flat within an episode), and
  * PER priorities get updated by a gradient step.

Speed: every test uses a SHRUNKEN net (tiny LSTM/encoder via ``net_kwargs``),
a tiny replay capacity, a 1-step ``min_replay`` warm-up, and only a handful of
episodes/updates. The single longer convergence-style check is marked
``@pytest.mark.slow`` and is deselected by the default ``-m 'not slow'`` addopts.

------------------------------------------------------------------------------
torch availability
------------------------------------------------------------------------------
The dev machine runs Python 3.14 where a torch wheel may be absent, so every
torch-dependent test guards with ``pytest.importorskip("torch")`` at the top of
the body — the suite stays GREEN (tests SKIP, not fail) when torch is missing.
"""

from __future__ import annotations

import numpy as np
import pytest

from agent.train_config import TrainConfig
from env.observation_spec import OBS_DIM
from agent.actions import N_ACTIONS


# ===========================================================================
# Tiny FAKE env — no socket, no live server.
# ===========================================================================


class FakeEnv:
    """Minimal Gym-style env: random obs of shape (OBS_DIM,), terminates after k.

    Satisfies ``agent.train.EnvProtocol`` (``reset(seed) -> obs`` and
    ``step(a) -> (obs, reward, done, info)``). Observations are uniform in
    ``[-1, 1]`` (the valid obs range) and seeded per episode from ``reset``'s seed
    so rollouts are reproducible. Rewards are small deterministic-from-RNG scalars;
    the episode terminates after exactly ``k`` steps (``done`` on the k-th step).
    """

    def __init__(self, k: int = 6, obs_dim: int = OBS_DIM) -> None:
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        self.k = int(k)
        self.obs_dim = int(obs_dim)
        self._rng = np.random.default_rng(0)
        self._t = 0

    def reset(self, seed=None) -> np.ndarray:
        self._rng = np.random.default_rng(0 if seed is None else int(seed))
        self._t = 0
        return self._obs()

    def step(self, action: int):
        if not (0 <= int(action) < N_ACTIONS):
            raise ValueError(f"action out of range: {action!r}")
        self._t += 1
        obs = self._obs()
        reward = float(self._rng.uniform(-1.0, 1.0))
        done = self._t >= self.k
        info = {"step": self._t}
        return obs, reward, done, info

    def _obs(self) -> np.ndarray:
        return self._rng.uniform(-1.0, 1.0, size=self.obs_dim).astype(np.float32)


# ===========================================================================
# Config / Trainer builders kept tiny for speed.
# ===========================================================================

# Episode length for the fake env. Must exceed burn_in + seq_len + n_step so the
# stored episodes yield sampleable windows and the n-step horizon fits.
_EPISODE_K = 14


def _tiny_cfg(**overrides) -> TrainConfig:
    """A fast TrainConfig: short windows, tiny warm-up, small replay."""
    base = dict(
        lr=1e-3,
        batch_size=4,
        seq_len=4,
        burn_in=2,
        n_step=2,
        gamma=0.99,
        tau=0.1,  # large-ish so the soft update is clearly visible in one step
        grad_clip=10.0,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay_episodes=10,
        replay_capacity=2_000,
        min_replay=1,  # learn almost immediately for a fast smoke
        per_beta_anneal_steps=100,
        eval_interval=0,
        checkpoint_interval=0,
        log_interval=1,
        seed=0,
    )
    base.update(overrides)
    return TrainConfig(**base)


# A shrunken net so forward/backward is cheap. obs_dim / n_actions still assert
# against the frozen contracts inside DuelingDRQN, so only the hidden sizes shrink.
_TINY_NET = {"encoder_hidden": 16, "lstm_hidden": 16, "lstm_layers": 1}


def _make_trainer(torch, cfg=None):
    from agent.train import Trainer

    return Trainer(cfg or _tiny_cfg(), net_kwargs=dict(_TINY_NET))


# ===========================================================================
# ε schedule — per EPISODE, monotone across episodes, flat within an episode.
# ===========================================================================


def test_epsilon_decays_per_episode_monotone():
    """epsilon_for_episode is non-increasing across episodes and hits the floor."""
    from agent.train import epsilon_for_episode

    cfg = _tiny_cfg(eps_start=1.0, eps_end=0.05, eps_decay_episodes=10)

    eps = [epsilon_for_episode(ep, cfg) for ep in range(20)]

    # Starts at eps_start, ends at the floor.
    assert eps[0] == pytest.approx(cfg.eps_start)
    assert eps[-1] == pytest.approx(cfg.eps_end)
    # Monotonically non-increasing across episodes.
    for earlier, later in zip(eps, eps[1:]):
        assert later <= earlier + 1e-12
    # Strictly decreasing during the decay window (not flat there).
    assert eps[1] < eps[0]
    # Flat at the floor after the decay window.
    for ep in range(cfg.eps_decay_episodes, 20):
        assert epsilon_for_episode(ep, cfg) == pytest.approx(cfg.eps_end)


def test_epsilon_is_flat_within_an_episode():
    """ε is a function of the EPISODE index only — querying it twice for the same
    episode (i.e. 'within' an episode) returns the identical value."""
    from agent.train import epsilon_for_episode

    cfg = _tiny_cfg()
    for ep in range(0, 15):
        a = epsilon_for_episode(ep, cfg)
        b = epsilon_for_episode(ep, cfg)
        assert a == b  # no per-step component; flat within the episode


# ===========================================================================
# Collection / learning smoke.
# ===========================================================================


def test_collect_episode_stores_transitions_and_hidden():
    """A rollout stores a full episode (with per-step hidden) into the replay."""
    torch = pytest.importorskip("torch", exc_type=ImportError)
    trainer = _make_trainer(torch)
    env = FakeEnv(k=_EPISODE_K)

    n, total_reward = trainer.collect_episode(env)
    assert n == _EPISODE_K  # one transition per decision until done
    assert np.isfinite(total_reward)
    assert len(trainer.replay) == _EPISODE_K
    # The window (burn_in + seq_len) fits inside the episode, so it is sampleable.
    assert trainer.replay.n_sampleable > 0
    assert trainer.episode_count == 1


def test_training_iterations_run_and_loss_is_finite():
    """A few gradient steps run without error and report a finite loss."""
    torch = pytest.importorskip("torch", exc_type=ImportError)
    trainer = _make_trainer(torch)
    env = FakeEnv(k=_EPISODE_K)

    # Collect a couple of episodes so the buffer has several sampleable windows.
    trainer.collect_episode(env)
    trainer.collect_episode(env)
    assert trainer.ready_to_learn()

    for _ in range(5):
        stats = trainer.learn()
        assert stats is not None
        assert np.isfinite(stats.loss)
        assert np.isfinite(stats.grad_norm)
        assert np.isfinite(stats.td_error_mean)
    assert trainer.grad_step == 5


def test_online_params_change_after_update():
    """One gradient step moves the online parameters (learning actually happens)."""
    torch = pytest.importorskip("torch", exc_type=ImportError)
    trainer = _make_trainer(torch)
    env = FakeEnv(k=_EPISODE_K)
    trainer.collect_episode(env)
    trainer.collect_episode(env)

    before = [p.detach().clone() for p in trainer.online.parameters()]
    stats = trainer.learn()
    assert stats is not None
    after = list(trainer.online.parameters())

    changed = any(
        not torch.equal(b, a.detach()) for b, a in zip(before, after)
    )
    assert changed, "online parameters did not change after a gradient step"


def test_target_moves_toward_online_but_not_equal():
    """Soft update nudges the target toward the online net without copying it."""
    torch = pytest.importorskip("torch", exc_type=ImportError)
    trainer = _make_trainer(torch)
    env = FakeEnv(k=_EPISODE_K)
    trainer.collect_episode(env)
    trainer.collect_episode(env)

    # Snapshot target params before the step (target == online at construction).
    target_before = [p.detach().clone() for p in trainer.target.parameters()]
    stats = trainer.learn()
    assert stats is not None

    online_after = [p.detach().clone() for p in trainer.online.parameters()]
    target_after = list(trainer.target.parameters())

    moved_toward = False
    not_equal = False
    for t_before, t_after, o_after in zip(
        target_before, target_after, online_after
    ):
        t_after = t_after.detach()
        # The target must have moved (online changed, so the soft update shifts it).
        if not torch.equal(t_before, t_after):
            moved_toward = True
            # After ONE soft step with tau < 1, target != online (not a full copy).
            if not torch.allclose(t_after, o_after):
                not_equal = True
            # And the move is exactly the Polyak interpolation toward online.
            expected = (1.0 - trainer.cfg.tau) * t_before + trainer.cfg.tau * o_after
            assert torch.allclose(t_after, expected, atol=1e-6), (
                "target update is not the configured Polyak interpolation"
            )
    assert moved_toward, "target net did not move after a soft update"
    assert not_equal, "target net equals online after one soft step (tau too large?)"


def test_priorities_get_updated_after_learn():
    """A gradient step writes fresh PER priorities back into the sum-tree."""
    torch = pytest.importorskip("torch", exc_type=ImportError)
    trainer = _make_trainer(torch)
    env = FakeEnv(k=_EPISODE_K)
    trainer.collect_episode(env)
    trainer.collect_episode(env)

    # New windows enter at MAX priority; capture the pre-learn total.
    total_before = trainer.replay._tree.total  # noqa: SLF001 (test-internal probe)

    # Spy on update_priorities to confirm it is called with a full batch of |δ|.
    seen = {}
    real_update = trainer.replay.update_priorities

    def spy(indices, td_errors):
        seen["indices"] = np.asarray(indices)
        seen["td"] = np.asarray(td_errors, dtype=np.float64)
        return real_update(indices, td_errors)

    trainer.replay.update_priorities = spy  # type: ignore[assignment]
    stats = trainer.learn()
    assert stats is not None

    assert "indices" in seen, "learn() did not call update_priorities"
    assert seen["indices"].shape[0] == trainer.cfg.batch_size
    assert seen["td"].shape[0] == trainer.cfg.batch_size
    assert np.all(np.isfinite(seen["td"]))
    # Updating priorities away from the initial max changes the tree total.
    total_after = trainer.replay._tree.total  # noqa: SLF001
    assert total_after != total_before


def test_train_loop_runs_with_hooks():
    """The full ``Trainer.train`` loop runs and fires the log hook by cadence."""
    torch = pytest.importorskip("torch", exc_type=ImportError)
    from agent.train import LearnStats

    trainer = _make_trainer(torch)
    env = FakeEnv(k=_EPISODE_K)

    log_calls = []

    def log_hook(tr, step, stats):
        assert isinstance(stats, LearnStats)
        assert step == stats.grad_step
        log_calls.append(step)

    trainer.train(env, num_episodes=4, updates_per_step=2, log_hook=log_hook)
    # At least one gradient step ran (min_replay == 1) and the hook fired.
    assert trainer.grad_step > 0
    assert log_calls, "log hook never fired despite gradient steps"


def test_functional_train_entrypoint_returns_trainer():
    """``train(env, cfg, n)`` constructs a Trainer, runs the loop, returns it."""
    torch = pytest.importorskip("torch", exc_type=ImportError)
    from agent.train import Trainer, train as train_fn

    env = FakeEnv(k=_EPISODE_K)
    trainer = train_fn(
        env,
        _tiny_cfg(),
        num_episodes=3,
        net_kwargs=dict(_TINY_NET),
        updates_per_step=1,
    )
    assert isinstance(trainer, Trainer)
    assert trainer.episode_count == 3


def test_n_step_target_truncates_at_done():
    """The n-step Double-DQN target drops the bootstrap term at a done boundary.

    Build a 1-window batch by hand whose only ``done`` falls inside the n-step
    horizon of a scored step; the trainer's loss path must zero that step's
    bootstrap (relying on ``compute_n_step_returns``' mask) so the target equals
    the discounted reward sum alone there.
    """
    torch = pytest.importorskip("torch", exc_type=ImportError)
    from agent.replay import Transition

    cfg = _tiny_cfg(batch_size=1, seq_len=4, burn_in=0, n_step=2, gamma=0.5)
    trainer = _make_trainer(torch, cfg)

    # Episode of exactly window length (burn_in + seq_len == 4). Make the LAST
    # transition terminal so the n-step horizon of earlier steps truncates.
    obs = [np.full(OBS_DIM, 0.01 * t, dtype=np.float32) for t in range(5)]
    episode = [
        Transition(
            obs=obs[t],
            action=t % N_ACTIONS,
            reward=1.0,
            next_obs=obs[t + 1],
            done=(t == 3),
        )
        for t in range(4)
    ]
    trainer.replay.add_episode(episode)
    assert trainer.replay.n_sampleable == 1

    # Confirm the n-step return + bootstrap mask the loss relies on: the LAST
    # scored step (t=3, done) and the second-to-last (t=2, whose 2-step horizon
    # crosses the done at t=3) must have bootstrap == False.
    batch = trainer.replay.sample_sequences(1, L=cfg.seq_len)
    returns, bootstrap = trainer.replay.n_step_returns(batch, n=cfg.n_step, gamma=cfg.gamma)
    # bootstrap shape (1, window). Steps 2 and 3 cannot bootstrap (done in horizon
    # / off-window); step 0 and 1 can.
    assert not bool(bootstrap[0, 3])
    assert not bool(bootstrap[0, 2])
    assert bool(bootstrap[0, 0])

    # And a learn() over this batch runs and yields a finite loss.
    stats = trainer.learn()
    assert stats is not None
    assert np.isfinite(stats.loss)


# ===========================================================================
# Regression: the n-step bootstrap must use the CORRECT recurrent hidden state.
#
# The bug (C1): the bootstrap Q was computed by running the LSTM over the
# ``next_obs`` stream seeded from the WINDOW-START hidden. That stream is shifted
# by one step (it starts at obs[B+1], missing obs[B]) so, for a recurrent net,
# the hidden used to evaluate Q(s_{i+n}) encoded the wrong history. The fix runs
# a single seeded forward over the EXTENDED contiguous stream obs_ext =
# concat(obs, next_obs[:, -1:]) and reads position i+n.
#
# This is invisible on an untrained net (the LSTM barely uses its hidden state),
# which is why the smoke tests miss it — so this test scales the recurrent
# weights up until memory dominates, then checks the trainer's bootstrap value
# against an explicit per-position reference recurrence over the SAME contiguous
# stream from the SAME collection hidden. It FAILS against the mis-seeded code.
# ===========================================================================


def _amplify_recurrence(torch, net) -> None:
    """Make the LSTM's recurrence non-trivial so memory dominates the output.

    Scales up the recurrent (hidden->hidden) weights and zeroes the input
    biases so the hidden state — i.e. the accumulated history — is the primary
    driver of each step's output. On the default near-zero-init net the hidden
    contribution is negligible and the mis-seeded bootstrap looks correct; after
    this amplification the seeding error produces a large, detectable difference.
    """
    with torch.no_grad():
        lstm = net.lstm
        for layer in range(lstm.num_layers):
            weight_hh = getattr(lstm, f"weight_hh_l{layer}")
            weight_ih = getattr(lstm, f"weight_ih_l{layer}")
            # Large recurrent weights => the hidden state strongly shapes the gates.
            weight_hh.copy_(torch.randn_like(weight_hh) * 3.0)
            weight_ih.copy_(torch.randn_like(weight_ih) * 1.0)
        # Strong, varied head weights so distinct hidden states map to distinct Q.
        net.value_head.weight.copy_(torch.randn_like(net.value_head.weight) * 2.0)
        net.advantage_head.weight.copy_(
            torch.randn_like(net.advantage_head.weight) * 2.0
        )


def _reference_bootstrap(torch, net_online, net_target, batch, seed_hidden, n_step):
    """Compute the Double-DQN bootstrap value per scored step by EXPLICIT recurrence.

    Independently of the trainer's batched path, step the LSTM one position at a
    time over the extended contiguous stream ``obs_ext = concat(obs,
    next_obs[:, -1:])`` from ``seed_hidden``; at each position ``p`` record the
    online-greedy action's TARGET Q (so position ``p`` holds Q(s_p) under the
    correct history s_0..s_p). For scored step ``i`` in ``[B, T)`` the bootstrap
    state ``s_{i+n}`` is at position ``i+n`` (clamped to ``T``).

    Returns a ``(batch, scored)`` tensor aligned with the scored steps.
    """
    with torch.no_grad():
        obs = torch.as_tensor(batch.obs, dtype=torch.float32)
        next_obs = torch.as_tensor(batch.next_obs, dtype=torch.float32)
        obs_ext = torch.cat([obs, next_obs[:, -1:, :]], dim=1)  # (b, T+1, OBS_DIM)
        bsz, ext_len, _ = obs_ext.shape  # ext_len == T + 1
        burn_in = batch.burn_in
        window = obs.shape[1]
        scored = window - burn_in

        # Per-position Q via single-step LSTM advances (the explicit reference
        # path). Online SELECTS the greedy action, target EVALUATES it
        # (Double-DQN); both nets carry their own hidden trajectory advanced one
        # position at a time from the SAME collection seed, so position p holds
        # Q(s_p) under history s_0..s_p.
        per_pos_eval = torch.empty(bsz, ext_len, dtype=torch.float32)
        hidden_o = (seed_hidden[0].clone(), seed_hidden[1].clone())
        hidden_t = (seed_hidden[0].clone(), seed_hidden[1].clone())
        for p in range(ext_len):
            step_obs = obs_ext[:, p : p + 1, :]  # (b, 1, OBS_DIM)
            q_online_p, hidden_o = net_online.forward(step_obs, hidden_o)  # (b,1,A)
            q_target_p, hidden_t = net_target.forward(step_obs, hidden_t)  # (b,1,A)
            a_star = q_online_p[:, 0, :].argmax(dim=-1, keepdim=True)  # (b, 1)
            per_pos_eval[:, p] = q_target_p[:, 0, :].gather(-1, a_star).squeeze(-1)

        # Gather bootstrap value at obs_ext position i+n for each scored step i.
        i = torch.arange(scored) + burn_in  # (scored,)
        boot_idx = (i + n_step).clamp(max=ext_len - 1)  # (scored,)
        boot_idx = boot_idx.unsqueeze(0).expand(bsz, scored)  # (b, scored)
        return per_pos_eval.gather(1, boot_idx)  # (b, scored)


def test_bootstrap_uses_correct_recurrent_hidden_state():
    """REGRESSION (C1): the n-step bootstrap Q must use the right recurrent memory.

    With amplified recurrence, the trainer's bootstrap value (captured from
    ``_gather_bootstrap_values``) must equal an explicit per-position reference
    recurrence over the extended contiguous stream from the collection hidden.
    The old mis-seeded code (LSTM over ``next_obs`` from the window-start hidden,
    indexed at ``i+n-1``) disagrees with this reference and FAILS.
    """
    torch = pytest.importorskip("torch", exc_type=ImportError)

    # A config where recurrence genuinely spans the bootstrap horizon: a non-zero
    # burn-in (so the seed hidden has real history to carry) and n_step > 1.
    cfg = _tiny_cfg(
        batch_size=4,
        seq_len=4,
        burn_in=2,
        n_step=2,
        gamma=0.99,
        min_replay=1,
    )
    trainer = _make_trainer(torch, cfg)

    # Make memory matter for BOTH nets identically (target is a copy of online),
    # so the only thing under test is which hidden state seeds the bootstrap.
    _amplify_recurrence(torch, trainer.online)
    trainer.target.load_state_dict(trainer.online.state_dict())

    env = FakeEnv(k=_EPISODE_K)
    trainer.collect_episode(env)
    trainer.collect_episode(env)
    assert trainer.ready_to_learn()

    # Capture the trainer's bootstrap value by wrapping the (instance) gather.
    captured = {}
    real_gather = trainer._gather_bootstrap_values  # noqa: SLF001

    def spy_gather(q_ext_eval, burn_in, scored, n_step):
        out = real_gather(q_ext_eval, burn_in, scored, n_step)
        captured["boot_value"] = out.detach().clone()
        return out

    trainer._gather_bootstrap_values = spy_gather  # type: ignore[assignment]

    # Compute the independent reference INSIDE the loss spy, while the online /
    # target nets still hold the EXACT weights the bootstrap was computed with.
    # (``learn()`` runs an optimizer step + soft target update AFTER ``_compute_loss``
    # returns, so the post-``learn`` weights would no longer match the captured
    # bootstrap — the reference must be taken on the live, pre-update nets.)
    real_compute_loss = trainer._compute_loss  # noqa: SLF001

    def spy_compute_loss(batch):
        captured["batch"] = batch
        seed_hidden = trainer._seed_hidden_from_batch(  # noqa: SLF001
            batch, batch.obs.shape[0]
        )
        captured["seed_hidden"] = seed_hidden
        captured["reference"] = _reference_bootstrap(
            torch,
            trainer.online,
            trainer.target,
            batch,
            seed_hidden,
            cfg.n_step,
        )
        return real_compute_loss(batch)

    trainer._compute_loss = spy_compute_loss  # type: ignore[assignment]

    stats = trainer.learn()
    assert stats is not None
    assert "boot_value" in captured, "the loss path did not gather bootstrap values"

    seed_hidden = captured["seed_hidden"]
    # Sanity: the batch actually carried collection-time hidden states (so the
    # seed is non-zero history, not a zero fallback that would hide the bug).
    assert captured["batch"].hidden is not None
    assert not torch.allclose(
        seed_hidden[0], torch.zeros_like(seed_hidden[0])
    ), "seed hidden is all zeros — recurrence would not exercise the bug"

    reference = captured["reference"]
    trainer_boot = captured["boot_value"]
    assert trainer_boot.shape == reference.shape

    # The amplified recurrence must make the reference meaningfully non-trivial,
    # otherwise the test could pass for the wrong reason (everything ~0).
    assert reference.abs().max().item() > 1e-3, (
        "reference bootstrap is ~0 — recurrence not amplified enough to test C1"
    )

    assert torch.allclose(trainer_boot, reference, atol=1e-5), (
        "trainer bootstrap Q does not match the correct per-position recurrence "
        "over the contiguous stream — the bootstrap is using a mis-seeded hidden "
        "state (C1 regression)"
    )


# ===========================================================================
# Slow convergence-style smoke (deselected by default).
# ===========================================================================


@pytest.mark.slow
def test_slow_many_updates_keep_loss_finite():
    """SLOW: many episodes + updates stay numerically stable (no NaN/Inf blowup)."""
    torch = pytest.importorskip("torch", exc_type=ImportError)
    trainer = _make_trainer(torch)
    env = FakeEnv(k=_EPISODE_K)

    losses = []

    def log_hook(tr, step, stats):
        losses.append(stats.loss)

    trainer.train(env, num_episodes=40, updates_per_step=4, log_hook=log_hook)
    assert losses, "no gradient steps ran"
    assert all(np.isfinite(loss) for loss in losses)
    # Target tracked online throughout without diverging (params stay finite).
    for p in trainer.online.parameters():
        assert torch.isfinite(p).all()
    for p in trainer.target.parameters():
        assert torch.isfinite(p).all()
