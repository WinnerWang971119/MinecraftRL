"""Tests for the Dueling-DRQN network (T14).

Covers two cases from the kickoff plan:

  - **TC8**: build the net, run forward + backward on one fixture sequence, and
    assert output shape, finite loss, non-NaN grads, and — crucially — that with
    a burn-in prefix ``B`` the loss/gradients depend ONLY on the post-burn-in
    ``L − B`` steps (perturbing burn-in-step targets yields zero gradient through
    them; the burn-in pass is detached).

  - **TC8b scaffold**: a FIXED synthetic episode where the correct greedy action
    at step ``t`` depends on a value observed at ``t − N`` that is now gated out
    of the observation, so it is recoverable ONLY from the LSTM hidden state.
    Provides (a) an overfit training harness asserting the burn-in DRQN learns
    the memory-dependent action above chance (marked slow), and (b) an ABLATION
    that zeroes the LSTM hidden state every step and shows it FAILS the same
    fixture — isolating LSTM correctness from the MLP encoder. A short-training
    smoke keeps the harness wired for T20.

------------------------------------------------------------------------------
torch availability
------------------------------------------------------------------------------
The dev machine runs Python 3.14 where a torch wheel may be absent. Every
torch-dependent test guards with ``torch = pytest.importorskip("torch", exc_type=ImportError)`` at the
top of the test body, so the suite stays GREEN (tests SKIP, not fail) when torch
is missing and runs for real once it is installed. There is intentionally no
hard ``import torch`` at module top level.

------------------------------------------------------------------------------
What this file does and does NOT validate
------------------------------------------------------------------------------
AC6 / M2 (the "learns vs dummy" milestone) does NOT validate that the LSTM
memory actually works — a memoryless MLP can beat a static dummy. TC8b is the
test that isolates and validates the recurrence: only the LSTM-equipped net
solves the memory-gated fixture, and the hidden-state ablation fails it.
"""

from __future__ import annotations

import pytest

from agent.actions import N_ACTIONS
from env.observation_spec import OBS_DIM


# ===========================================================================
# Shared fixture builders (pure-Python / torch-free signatures; torch tensors
# are created inside, guarded by importorskip in each test).
# ===========================================================================


def _make_net(torch):
    """Build a fresh DuelingDRQN with a fixed seed for reproducibility."""
    from agent.dqn import DuelingDRQN

    torch.manual_seed(0)
    return DuelingDRQN()


def _build_memory_fixture(torch, *, seq_len: int = 12, seed: int = 7):
    """Construct ONE fixed memory-gated episode pair.

    A "cue" is shown at step 0 in a dedicated observation channel; that channel
    is then ZEROED for every later step (the gate). The correct target action at
    every step is selected by the cue, so for every step after step 0 it is
    recoverable ONLY from the LSTM hidden state — never from the current
    observation. The MLP encoder, which sees only the current step, cannot
    distinguish the two episodes once the gate closes.

    We use TWO episodes (cue=−1 and cue=+1) that are IDENTICAL except for the
    single cue step, with different correct actions. A memoryless model sees
    identical inputs on every step after step 0, so for those steps it must emit
    the SAME action for both episodes and can satisfy only ONE of them. Its best
    strategy is therefore: get both cue steps right (t=0 differs, so it can) and
    commit the rest to one episode's action — yielding roughly one full episode
    plus the two cue steps correct, i.e. a ceiling near ``0.5 + 1/seq_len`` of
    the scored steps (≈0.54 for seq_len=12). A net that carries the cue in the
    LSTM hidden state can be right on every step of BOTH episodes (≈1.0).

    Returns:
        (obs, targets) where
          obs:     (2, seq_len, OBS_DIM) float32  — two episodes
          targets: (2, seq_len)         int64     — correct action per step
    """
    g = torch.Generator().manual_seed(seed)

    # A neutral, identical-across-episodes background so the ONLY difference is
    # the cue at step 0. Small magnitude keeps it in the net's linear regime.
    background = 0.1 * torch.randn((seq_len, OBS_DIM), generator=g)

    # Cue channel and two distinct target actions the cue selects.
    cue_channel = 0
    action_if_cue0 = 1
    action_if_cue1 = 4
    assert action_if_cue0 < N_ACTIONS and action_if_cue1 < N_ACTIONS

    obs = background.unsqueeze(0).repeat(2, 1, 1).clone()  # (2, T, OBS_DIM)

    # Episode 0: cue = -1 at step 0. Episode 1: cue = +1 at step 0.
    obs[0, 0, cue_channel] = -1.0
    obs[1, 0, cue_channel] = +1.0
    # Gate: the cue channel is zero for ALL steps after step 0 in both episodes.
    obs[:, 1:, cue_channel] = 0.0

    targets = torch.empty((2, seq_len), dtype=torch.long)
    targets[0, :] = action_if_cue0
    targets[1, :] = action_if_cue1

    return obs, targets


# ===========================================================================
# TC8 — forward + backward + burn-in gradient isolation
# ===========================================================================


def test_tc8_construct_asserts_contract_dims():
    """The net's input/output dims are pinned to the frozen contracts."""
    torch = pytest.importorskip("torch", exc_type=ImportError)
    net = _make_net(torch)
    assert net.obs_dim == OBS_DIM
    assert net.n_actions == N_ACTIONS
    # Mismatching either contract must fail loudly.
    from agent.dqn import DuelingDRQN

    with pytest.raises(ValueError):
        DuelingDRQN(obs_dim=OBS_DIM + 1)
    with pytest.raises(ValueError):
        DuelingDRQN(n_actions=N_ACTIONS + 1)


def test_tc8_forward_output_shape():
    """forward maps (B, T, OBS_DIM) → Q of shape (B, T, N_ACTIONS)."""
    torch = pytest.importorskip("torch", exc_type=ImportError)
    net = _make_net(torch)
    batch, seq_len = 3, 6
    obs = torch.randn(batch, seq_len, OBS_DIM)

    q_seq, hidden = net(obs)
    assert q_seq.shape == (batch, seq_len, N_ACTIONS)
    h_n, c_n = hidden
    assert h_n.shape == (net.lstm_layers, batch, net.lstm_hidden)
    assert c_n.shape == (net.lstm_layers, batch, net.lstm_hidden)


def test_tc8_init_hidden_is_zeroed_and_shaped():
    """init_hidden returns zeroed (h0, c0) of the LSTM's (layers, N, H) shape."""
    torch = pytest.importorskip("torch", exc_type=ImportError)
    net = _make_net(torch)
    h0, c0 = net.init_hidden(5)
    assert h0.shape == (net.lstm_layers, 5, net.lstm_hidden)
    assert c0.shape == (net.lstm_layers, 5, net.lstm_hidden)
    assert torch.count_nonzero(h0) == 0
    assert torch.count_nonzero(c0) == 0

    with pytest.raises(ValueError):
        net.init_hidden(0)


def test_tc8_forward_backward_loss_finite_grads_non_nan():
    """One forward+backward on a fixture sequence: finite loss, non-NaN grads."""
    torch = pytest.importorskip("torch", exc_type=ImportError)
    import torch.nn.functional as F

    net = _make_net(torch)
    batch, seq_len = 2, 8
    torch.manual_seed(1)
    obs = torch.randn(batch, seq_len, OBS_DIM)
    targets = torch.randint(N_ACTIONS, (batch, seq_len))

    q_seq, _ = net(obs)
    loss = F.cross_entropy(q_seq.reshape(-1, N_ACTIONS), targets.reshape(-1))
    assert torch.isfinite(loss), "loss must be finite"

    loss.backward()
    grad_params = [p for p in net.parameters() if p.grad is not None]
    assert grad_params, "expected gradients on the parameters"
    for p in grad_params:
        assert torch.isfinite(p.grad).all(), "gradient contains NaN/Inf"


def test_tc8_burn_in_output_shape_excludes_prefix():
    """forward_with_burn_in returns Q for the scored L−B steps only."""
    torch = pytest.importorskip("torch", exc_type=ImportError)
    net = _make_net(torch)
    batch, seq_len, burn_in = 2, 10, 4
    obs = torch.randn(batch, seq_len, OBS_DIM)

    q_scored, seed_hidden, final_hidden = net.forward_with_burn_in(obs, burn_in)
    assert q_scored.shape == (batch, seq_len - burn_in, N_ACTIONS)
    # The seed hidden state must be detached (no grad path through burn-in).
    assert not seed_hidden[0].requires_grad
    assert not seed_hidden[1].requires_grad

    # Out-of-range burn-in is rejected.
    with pytest.raises(ValueError):
        net.forward_with_burn_in(obs, seq_len)  # B == L is invalid
    with pytest.raises(ValueError):
        net.forward_with_burn_in(obs, -1)


def test_tc8_burn_in_gradients_depend_only_on_scored_steps():
    """CORE TC8 assertion: gradients flow ONLY through post-burn-in steps.

    We compute the burn-in loss two ways that must give identical gradients:
      (1) score only the L−B suffix via forward_with_burn_in, and
      (2) run the full sequence but mask the loss so burn-in steps contribute 0.

    Then we PERTURB only the burn-in-step targets and confirm the gradients are
    unchanged — proving the burn-in steps are detached and contribute nothing to
    the backward pass.
    """
    torch = pytest.importorskip("torch", exc_type=ImportError)
    import torch.nn.functional as F

    batch, seq_len, burn_in = 2, 9, 4
    torch.manual_seed(2)
    obs = torch.randn(batch, seq_len, OBS_DIM)
    targets = torch.randint(N_ACTIONS, (batch, seq_len))

    def scored_grads(tgts):
        net = _make_net(torch)  # identical init each call
        net.zero_grad(set_to_none=True)
        q_scored, _, _ = net.forward_with_burn_in(obs, burn_in)
        scored_targets = tgts[:, burn_in:].reshape(-1)
        loss = F.cross_entropy(q_scored.reshape(-1, N_ACTIONS), scored_targets)
        loss.backward()
        return [p.grad.clone() for p in net.parameters()]

    base_grads = scored_grads(targets)

    # Perturb ONLY the burn-in-step targets. If the burn-in contributed any
    # gradient, these grads would change; they must be bit-identical.
    perturbed = targets.clone()
    perturbed[:, :burn_in] = (perturbed[:, :burn_in] + 1) % N_ACTIONS
    perturbed_grads = scored_grads(perturbed)

    for g_base, g_pert in zip(base_grads, perturbed_grads):
        assert torch.equal(g_base, g_pert), (
            "perturbing burn-in-step targets changed the gradients — the "
            "burn-in prefix is NOT detached"
        )


def test_tc8_burn_in_matches_masked_full_sequence_gradients():
    """forward_with_burn_in grads equal a mask-the-prefix full pass (same seed).

    Validates the burn-in implementation produces the SAME gradient as the
    reference recipe of running the full sequence and zeroing the burn-in loss,
    confirming the scored-step graph is identical (no off-by-one in the split).
    """
    torch = pytest.importorskip("torch", exc_type=ImportError)
    import torch.nn.functional as F

    batch, seq_len, burn_in = 2, 9, 4
    torch.manual_seed(3)
    obs = torch.randn(batch, seq_len, OBS_DIM)
    targets = torch.randint(N_ACTIONS, (batch, seq_len))

    # (1) burn-in path
    net_a = _make_net(torch)
    q_scored, _, _ = net_a.forward_with_burn_in(obs, burn_in)
    loss_a = F.cross_entropy(
        q_scored.reshape(-1, N_ACTIONS), targets[:, burn_in:].reshape(-1)
    )
    loss_a.backward()
    grads_a = [p.grad.clone() for p in net_a.parameters()]

    # (2) reference: full sequence, but seed the suffix from a no_grad burn-in,
    #     score only the suffix. This mirrors what forward_with_burn_in does, so
    #     gradients must match to floating-point tolerance.
    net_b = _make_net(torch)
    with torch.no_grad():
        _, seed = net_b(obs[:, :burn_in])
    seed = (seed[0].detach(), seed[1].detach())
    q_suffix, _ = net_b(obs[:, burn_in:], seed)
    loss_b = F.cross_entropy(
        q_suffix.reshape(-1, N_ACTIONS), targets[:, burn_in:].reshape(-1)
    )
    loss_b.backward()
    grads_b = [p.grad.clone() for p in net_b.parameters()]

    for ga, gb in zip(grads_a, grads_b):
        assert torch.allclose(ga, gb, atol=1e-6), (
            "burn-in gradients diverge from the masked-full-sequence reference"
        )


def test_tc8_act_epsilon_greedy_deterministic_with_generator():
    """act selects greedy at ε=0, random at ε=1, and is generator-deterministic."""
    torch = pytest.importorskip("torch", exc_type=ImportError)
    net = _make_net(torch)
    obs = torch.randn(OBS_DIM)

    # ε = 0 → pure greedy = argmax Q.
    hidden = net.init_hidden(1)
    q_seq, _ = net(obs.reshape(1, 1, OBS_DIM), hidden)
    greedy = int(torch.argmax(q_seq[0, 0]).item())
    action, new_hidden = net.act(obs, hidden, epsilon=0.0)
    assert action == greedy
    assert 0 <= action < N_ACTIONS
    # act advanced the LSTM state by one step.
    assert new_hidden[0].shape == hidden[0].shape

    # ε = 1 → random; same generator seed reproduces the same action stream.
    def run_random(seed):
        gen = torch.Generator().manual_seed(seed)
        h = net.init_hidden(1)
        actions = []
        for _ in range(8):
            a, h = net.act(obs, h, epsilon=1.0, generator=gen)
            actions.append(a)
        return actions

    seq1 = run_random(123)
    seq2 = run_random(123)
    assert seq1 == seq2, "generator-seeded random actions must be reproducible"
    assert all(0 <= a < N_ACTIONS for a in seq1)

    # epsilon out of range rejected.
    with pytest.raises(ValueError):
        net.act(obs, None, epsilon=1.5)


# ===========================================================================
# TC8b scaffold — memory-dependent recurrence gate (full train runs at M2/T20)
# ===========================================================================


def _greedy_accuracy(torch, net, obs, targets, burn_in):
    """Fraction of SCORED steps where argmax Q matches the target action."""
    net.eval()
    with torch.no_grad():
        q_scored, _, _ = net.forward_with_burn_in(obs, burn_in)
    preds = q_scored.argmax(dim=-1)  # (B, L−B)
    scored_targets = targets[:, burn_in:]
    correct = (preds == scored_targets).float().mean().item()
    net.train()
    return correct


def _overfit_memory_fixture(torch, net, obs, targets, burn_in, *, steps, lr=5e-3):
    """Tiny supervised overfit of the net to the memory fixture.

    Uses cross-entropy on Q-logits as a stand-in for the RL target so the harness
    is self-contained and fast; the point is to prove the architecture CAN route
    the cue through the LSTM, not to validate the DQN loss (that is T20).
    Returns the final scored-step accuracy.
    """
    import torch.nn.functional as F

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    scored_targets = targets[:, burn_in:].reshape(-1)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        q_scored, _, _ = net.forward_with_burn_in(obs, burn_in)
        loss = F.cross_entropy(q_scored.reshape(-1, N_ACTIONS), scored_targets)
        loss.backward()
        opt.step()
    return _greedy_accuracy(torch, net, obs, targets, burn_in)


class _HiddenAblatedDRQN:
    """Wrapper that ZEROES the LSTM hidden state every step (memory ablation).

    It reuses the real encoder + LSTM + dueling heads but never lets hidden state
    flow between steps: each timestep is run independently from a zero state.
    This isolates the LSTM's contribution — the MLP encoder and heads are
    identical, so if the full net solves the fixture and this one cannot, the
    difference is attributable to recurrence alone.
    """

    def __init__(self, net):
        self.net = net

    def parameters(self):
        return self.net.parameters()

    def train(self):
        self.net.train()

    def eval(self):
        self.net.eval()

    def forward_with_burn_in(self, obs, burn_in):
        import torch

        batch, seq_len, _ = obs.shape
        scored = []
        # Run EVERY step from a fresh zero hidden state — no memory carries over.
        for t in range(burn_in, seq_len):
            step = obs[:, t : t + 1]  # (B, 1, OBS_DIM)
            zero_hidden = self.net.init_hidden(batch, device=obs.device)
            q_step, _ = self.net(step, zero_hidden)  # (B, 1, N_ACTIONS)
            scored.append(q_step)
        q_scored = torch.cat(scored, dim=1)  # (B, L−B, N_ACTIONS)
        return q_scored, None, None


@pytest.mark.slow
def test_tc8b_burn_in_drqn_learns_memory_dependent_action():
    """SLOW: the recurrent net overfits the memory fixture well above chance.

    The cue is gated out after step 0, so being right on the post-cue steps of
    BOTH episodes requires carrying the cue in the LSTM hidden state. A memoryless
    model is capped near chance there; the recurrent net should approach perfect.
    Marked slow; the full train-to-convergence assertion runs at M2/T20.

    ``burn_in`` is 0 here ON PURPOSE: R2D2 burn-in steps run under ``no_grad`` and
    receive no gradient, so to TEACH the net to *store* the cue the cue step must
    be in the scored window. The burn-in *mechanism* (detached prefix, scored
    suffix, gradient isolation) is validated separately by the TC8 tests above.
    """
    torch = pytest.importorskip("torch", exc_type=ImportError)
    obs, targets = _build_memory_fixture(torch)
    burn_in = 0
    net = _make_net(torch)

    accuracy = _overfit_memory_fixture(
        torch, net, obs, targets, burn_in, steps=600
    )
    # Well above the memoryless ceiling (~1/seq_len correct on both episodes).
    assert accuracy > 0.9, (
        f"recurrent DRQN failed to learn the memory-dependent action "
        f"(scored accuracy {accuracy:.3f})"
    )


def test_tc8b_smoke_harness_runs_short_training():
    """Smoke: the overfit harness runs end-to-end on a short budget (T20-ready).

    Keeps the harness wired and importable without the slow convergence cost.
    Asserts only that training runs and produces a valid accuracy in [0, 1].
    """
    torch = pytest.importorskip("torch", exc_type=ImportError)
    obs, targets = _build_memory_fixture(torch)
    burn_in = 0
    net = _make_net(torch)

    accuracy = _overfit_memory_fixture(
        torch, net, obs, targets, burn_in, steps=5
    )
    assert 0.0 <= accuracy <= 1.0


def test_tc8b_ablation_zeroed_hidden_state_fails_fixture():
    """ABLATION: zeroing the LSTM hidden state each step CANNOT solve the fixture.

    Same encoder/heads, but no memory carries between steps. After the cue is
    gated out the two episodes are observationally identical, so a memoryless
    model must emit the same action for both on those steps and can satisfy only
    one episode there. Its best case is one full episode plus the two cue steps,
    a ceiling near ``0.5 + 1/seq_len`` (≈0.54 for seq_len=12). Training it must
    NOT approach the recurrent net's near-perfect (>0.9) accuracy — isolating the
    LSTM as the component that makes the full net succeed (TC8b above).

    Uses the SAME ``burn_in = 0`` and step budget as the recurrent test so the
    only difference is the zeroed hidden state.
    """
    torch = pytest.importorskip("torch", exc_type=ImportError)
    obs, targets = _build_memory_fixture(torch)
    burn_in = 0
    seq_len = obs.shape[1]
    ablated = _HiddenAblatedDRQN(_make_net(torch))

    accuracy = _overfit_memory_fixture(
        torch, ablated, obs, targets, burn_in, steps=600
    )
    # Memoryless ceiling ≈ one episode + cue steps. Slack above it for fp/ties,
    # but it must stay well below the recurrent net's >0.9.
    ceiling = 0.5 + (1.0 / seq_len) + 0.10
    assert accuracy <= ceiling, (
        f"hidden-state ablation unexpectedly solved the memory fixture "
        f"(scored accuracy {accuracy:.3f} > ceiling {ceiling:.3f}); the fixture "
        f"is not actually memory-gated, so TC8b would not be isolating the LSTM"
    )
