# Training Specification (Complete) — Minecraft PvP DQN

**This is the authoritative, self-contained spec for *training* the agent.** Everything needed to implement the full training pipeline is here: environment contract, observation/reward, network, the exact DQN learning rule, exploration, opponent curriculum and self-play, the training loop, distributed training, evaluation/checkpointing, reproducibility, and a debugging playbook.

**Relation to the other docs**
- `minecraft-pvp-dqn-spec.md` — the broader system (tech stack, bridge, deployment/exhibition, repo, team). Read it for the full observation table and the Node↔Python contract.
- `minecraft-pvp-distributed-training.md` — wire-level details of the distributed transport. §10 here summarizes it.

**Conventions:** `TUNE` = starting value, tune empirically. All equations are written for direct implementation.

---

## 0. Training objective & definition of "trained"

We learn a policy π that wins **1-v-1 melee PvP** under a human-equivalent perception constraint (the `PerceptionFilter`). Training success is defined by milestone gates, each with a concrete, measurable pass criterion:

| Gate | Meaning | Pass criterion (`TUNE`) |
|---|---|---|
| **M1** | bridge + loop integrate | random policy completes full episodes vs a stationary dummy, end-to-end, no crashes |
| **M2** | learns basic combat | ≥ 95% win rate vs **stationary dummy** over 100 eval episodes |
| **M3** | beats fixed strategy | ≥ 70% win rate vs the **scripted bot** over 200 eval episodes |
| **M4** | self-play improves | **Elo rises** monotonically (trend) across self-play snapshots |

"Trained" = M3 reached with a stable training curve; M4 is the research-grade target. **We do not need to beat SOTA or train to convergence** — a focused, well-controlled result is the goal.

---

## 1. The training environment as an MDP

### 1.1 Episode

- **Reset:** teleport both agents to fixed spawn points, restore full health and identical gear, clear the `PerceptionFilter` memory.
- **Termination (`done`):** either agent's health reaches 0, **or** a step/time cap is hit (`MAX_EPISODE_STEPS`, `TUNE` e.g. 400 decisions ≈ 80 s at 5 dps). A timeout is a draw (no terminal win/loss reward; see §3).
- **Decision interval:** the agent acts every `ACTION_REPEAT` game ticks (`TUNE` e.g. 4 ticks ≈ 200 ms). The chosen macro runs for that interval; events are aggregated over it.

### 1.2 Observation, action, reward (summary; full observation table in the main spec)

- **Observation** — structured vector (~30–40 floats) of self-state + **gated** opponent-state, produced by the `PerceptionFilter` (§2). Angles as `(sin, cos)`; positions in the agent's **local frame**; normalized to ≈ `[-1, 1]`.
- **Action** — discrete tactical macros (§ main spec): `{idle, approach, retreat, strafe-L, strafe-R, attack, jump, turn-to-last-seen}`. Discrete → DQN-native.
- **Reward** — dense, damage-anchored (§3).

### 1.3 Determinism

Seed the env per episode where possible; log every seed. Minecraft has inherent stochasticity (physics, timing) — accept it, but control everything you can.

---

## 2. Observation pipeline (PerceptionFilter) — training-relevant essentials

The fairness rule: **perceptual parity, not state parity.** Remove omniscience, not numbers. (Full per-feature table in the main spec.) Training-critical points:

1. **Self features** (`FULL`): health, facing (yaw/pitch), local velocity, on-ground, held item, attack-cooldown progress.
2. **Opponent features** (`GATED`): position/facing/velocity given **only** when the opponent is inside the FOV cone (≈70°, `TUNE`) **and** a raycast line-of-sight is clear; otherwise the position degrades to `MEMORY` (last-seen value + `time_since_seen`) for `MEMORY_TTL` (≈5 s, `TUNE`), then to absent. A `visible` flag and `time_since_seen` scalar are always present so the net distinguishes "unseen" from "value 0".
3. **Opponent health:** `NONE` by default — infer from landed hits. (Scoreboard-nametag option in main spec §16.)
4. **Derived features** (`in_range`, `in_crosshair`) inherit the same gating — compute them *after* gating, from gated values only. **Leaking the opponent's position through a derived feature is the most common fairness bug.**
5. **Privileged info rule:** the **reward** may read raw privileged values (true opponent health → damage). The **observation** may not. Keep this boundary clean.
6. **Memory is mandatory:** because information expires, the policy must be recurrent (LSTM/DRQN) or frame-stacked (§5).

`observation_spec.py` is the single source of truth for the vector layout and is imported by the net, the env, and (critically) every distributed actor (§10).

---

## 3. Reward function

Dense shaping anchored to damage; minimal, to avoid reward hacking. Suggested form per decision step:

```
r =  c_dmg_out * damage_dealt        # primary objective
   - c_dmg_in  * damage_taken        # survive
   - c_step                          # small per-step penalty (decisiveness)
   + c_aim * 1[opponent_visible AND in_crosshair]   # gated on visibility
   + R_terminal                      # at episode end only
```

| Coefficient | Role | Start (`TUNE`) |
|---|---|---|
| `c_dmg_out` | reward per point of damage dealt | 1.0 (per HP) |
| `c_dmg_in` | penalty per point of damage taken | 1.0 (symmetric or slightly < out) |
| `c_step` | per-step penalty | small, e.g. 0.005 |
| `c_aim` | aiming shaping (visibility-gated) | small, e.g. 0.01 |
| `R_terminal` | +W on win, −L on loss, 0 on timeout | W = L = 5–10 |

**Prefer potential-based shaping** for any positional term (e.g. closing distance) so it provably does not change the optimal policy: `F(s,s') = γ·Φ(s') − Φ(s)`.

**Anti-hacking review (re-check during tuning):**
- `c_step` too large → suicide-rushing; too small → runs away forever. The single most important coefficient to tune.
- An always-on aiming reward → spin-to-farm. That is why `c_aim` is gated on `opponent_visible` and is tiny.
- `damage_dealt`/`damage_taken` come from the raw `events` block — privileged and fair (§2.5).

---

## 4. Network architecture

```
obs vector (≈30-40)
      │
   [MLP encoder]   Linear→ReLU→Linear→ReLU      (hidden 256, 256)   TUNE
      │
   [LSTM head]     hidden 256, 1 layer          (DRQN — required for partial obs)
      │
   ┌──┴───────────────┐
[Value V(s)]   [Advantage A(s,a)]                (Dueling: two heads)
   └──────┬───────────┘
   Q(s,a) = V(s) + (A(s,a) − mean_a' A(s,a'))     (dueling aggregation)
      │
   Q over the discrete macro-actions
```

- If not using DRQN, replace the LSTM with **frame-stacking** (concatenate last `k`=4 observations into the MLP). DRQN is preferred here because memory length is variable ("how long since I saw them").
- Small net by design → **GPU is not the bottleneck** (the bottleneck is environment throughput; see §11).

---

## 5. The DQN learning rule (exact)

Implement the ladder incrementally (§ main spec): vanilla → Double → Dueling → PER → DRQN → n-step. Final target form:

### 5.1 n-step Double-DQN target

For a stored n-step segment starting at t (terminate the sum early if `done`):

```
G        = Σ_{k=0}^{n-1} γ^k · r_{t+k}
a*       = argmax_a Q_online(s_{t+n}, a)
y        = G + (1 − done) · γ^n · Q_target(s_{t+n}, a*)
```

`n` (`TUNE`) = 3–5. Double DQN (action selection by online net, evaluation by target net) curbs overestimation.

### 5.2 Loss (Huber on TD error)

```
δ_i  = y_i − Q_online(s_i, a_i)
L    = (1/B) Σ_i  w_i · Huber(δ_i)        # w_i = PER importance weight (=1 if uniform)
```

Huber (smooth-L1) is more robust to outlier TD errors than MSE. **Clip gradient norm** (e.g. max-norm 10, `TUNE`).

### 5.3 Target network

Either:
- **Hard:** copy `θ_target ← θ_online` every `TARGET_UPDATE` steps (`TUNE` 1k–10k), or
- **Soft:** `θ_target ← τ·θ_online + (1−τ)·θ_target` every step (`τ` ≈ 0.005, `TUNE`).

Soft updates are smoother and a good default here.

### 5.4 Prioritized Experience Replay (PER)

```
priority   p_i = |δ_i| + ε_p            (ε_p small, e.g. 1e-6)
sample P(i)    = p_i^α / Σ_k p_k^α      (α ≈ 0.6, TUNE)
IS weight w_i  = ( 1 / (N · P(i)) )^β   then normalize by max_i w_i
β anneal       0.4 → 1.0 over training
```

Update priorities with the latest `|δ_i|` after each gradient step. New transitions enter at max priority.

### 5.5 DRQN specifics (R2D2-style)

- **Store sequences**, not shuffled singletons: ordered chunks per episode (the distributed actors push sequences; §10).
- **Sample** sequences of length `L` (`TUNE` 8–16) for training.
- **Burn-in:** run the first `B` steps (`TUNE` ≈ 4) only to warm the LSTM hidden state; compute the loss on the remaining `L−B` steps. Store the hidden state at collection time and use it as the burn-in seed (R2D2 recipe). Zero-init burn-in is an acceptable simpler fallback.

### 5.6 Hyperparameters (consolidated)

| Param | Start (`TUNE`) |
|---|---|
| Discount γ | 0.99 |
| Optimizer | Adam, lr 1e-4 … 2.5e-4 |
| Batch size | 32–64 (sequences for DRQN) |
| Replay capacity | 1e5 – 1e6 transitions |
| Min replay before training | 10k–50k |
| n-step | 3–5 |
| Target update | soft τ ≈ 0.005 (or hard 1k–10k) |
| Grad-norm clip | 10 |
| PER α / β | 0.6 / 0.4→1.0 |
| DRQN seq len / burn-in | 8–16 / 4 |

---

## 6. Exploration

- **ε-greedy** with ε annealed `1.0 → 0.05` over the first ~10–20% of training, then held.
- **Distributed (Ape-X-style) per-actor ε:** give each actor a *fixed, different* ε so the pool covers a spread of exploration rates simultaneously:

```
ε_i = ε_base ^ ( 1 + (i / (N−1)) · α_exp )     for actor i in 0..N−1
ε_base ≈ 0.4 ,  α_exp ≈ 7        (TUNE)
```

This yields some near-greedy actors (good data near the current policy) and some highly exploratory ones (coverage), without an annealing schedule per actor. The learner stays purely greedy for its target computation.

---

## 7. Opponent curriculum & self-play

A single learner throughout. **Never train two policies simultaneously** — concurrent learners are highly non-stationary and unstable.

### 7.1 Stage 0 — stationary dummy
Validates bridge + reward + "can it approach and land hits." **Exit:** M2 (≥95% vs dummy / 100 eps).

### 7.2 Stage 1 — scripted bot
A fixed heuristic opponent. Reference behavior:

```
if low_health and c_flee:        retreat
elif in_attack_range:            attack (respect cooldown)
elif can_see_player:             approach; with prob p_strafe, strafe; with prob p_jump, jump
else:                            move toward last-known position / search
```

`p_strafe`, `p_jump`, `c_flee` are `TUNE`. Keep it competent but beatable. **Exit:** M3 (≥70% vs scripted / 200 eps).

### 7.3 Stage 2 — self-play with frozen snapshots
- The learner trains; the **opponent is a frozen snapshot** of a past policy, drawn from a **snapshot pool**.
- **Snapshot cadence:** push a new frozen copy every `SNAPSHOT_EVERY` learner updates (`TUNE`).
- **Opponent sampling:**
  - *Uniform* over the pool (simple, robust baseline), or
  - **PFSP** (prioritized fictitious self-play): sample opponents weighted toward those the current policy beats ~40–60% of the time — focuses training on challenging-but-winnable matchups and prevents both forgetting and stagnation.
- **Elo** to measure progress (round-robin among snapshots):

```
E_a = 1 / (1 + 10^((R_b − R_a)/400))
R_a ← R_a + K · (S_a − E_a)        S_a ∈ {1, 0.5, 0},  K ≈ 16–32
```

**Exit:** M4 (Elo trend rising).

### 7.4 Fairness curriculum (optional)
The `PerceptionFilter` knobs (FOV width, `MEMORY_TTL`) double as difficulty: start looser so the agent *can* learn, then tighten toward human limits as competence grows. Train first, blind down second.

---

## 8. The training loop (single-machine reference)

```python
seed_everything(SEED)
env      = MCPvPEnv(observation_spec, perception_filter, reward_fn)
online, target = build_dueling_drqn(); target.load_state_dict(online.state_dict())
replay   = PrioritizedSequenceReplay(capacity)
opt      = Adam(online.parameters(), lr=LR)
stage    = Stage0_Dummy()
steps    = 0

while not training_complete():
    obs, h = env.reset(), online.init_hidden()
    ep = new_episode_buffer()
    done = False
    while not done:
        a, h = online.act(obs, h, epsilon=eps(steps))     # ε-greedy
        obs2, r, done, info = env.step(a)
        ep.append(obs, a, r, obs2, done)
        obs = obs2

        if len(replay) >= MIN_REPLAY:
            seqs, idx, w = replay.sample_sequences(BATCH, L)
            y    = nstep_double_target(seqs, online, target, γ, n)    # §5.1
            δ    = y - online.q(seqs)                                 # with burn-in §5.5
            loss = (w * huber(δ)).mean()
            opt.zero_grad(); loss.backward()
            clip_grad_norm_(online.parameters(), 10); opt.step()
            replay.update_priorities(idx, abs(δ))                     # §5.4
            soft_update(target, online, τ)                           # §5.3
            steps += 1
            if steps % SNAPSHOT_EVERY == 0: snapshot_pool.add(online) # §7.3
            if steps % EVAL_EVERY     == 0: run_eval(online, stage)   # §9
            if steps % CKPT_EVERY     == 0: checkpoint(online, target, opt, steps)
            if steps % LOG_EVERY      == 0: log_metrics(...)

    replay.add_episode(ep)                       # store as sequences
    stage = maybe_advance_stage(stage, latest_eval)   # gate transitions §7
```

In distributed mode this loop is split: the inner `env` rollout becomes the **actor**, and the gradient block becomes the **learner** (§10).

---

## 9. Evaluation, logging & checkpointing

### 9.1 Evaluation protocol
- Periodically (`EVAL_EVERY`) evaluate the **greedy** policy (ε=0) against the current stage's reference opponent over a **fixed** number of episodes (dummy: 100; scripted: 200). Report win rate (the gate metric).
- For self-play, maintain **Elo** via round-robin among snapshots.
- Evaluation must use the **same `PerceptionFilter`** as training (and as deployment) — no privileged eval.

### 9.2 Logging (TensorBoard / W&B)
Log: loss, mean/max Q, mean `|TD error|`, ε, replay size, episode length, **each reward component separately** (to catch hacking), win rate, Elo, and per-actor throughput (distributed). Plotting reward components separately is the fastest way to see "it's farming the aim bonus" or "it learned to run."

### 9.3 Checkpointing
- Save every `CKPT_EVERY`: `online`, `target`, optimizer state, `steps`, RNG state, `config`, `code_version`, and the snapshot pool index. (Replay buffer optionally — large; usually skip and warm up on resume.)
- Resume must be exact enough to continue training; record the git SHA.

---

## 10. Distributed training (summary; wire details in the companion doc)

When arenas live on team members' PCs:

- **One learner, many actors, Redis broker.** Actors run env + a **copy** of the policy + a local opponent; they batch experience **up** and pull weights **down**. The learner trains on the pooled replay and republishes weights periodically.
- **Why DQN fits:** it is **off-policy**, so actors may use slightly stale weights — no tight sync, tolerant of home-internet lag. Embrace policy lag; the learner can drop only extremely stale batches (`MAX_STALENESS`).
- **Per-actor ε** (§6) gives exploration diversity for free.
- **Self-play distributes cleanly:** each actor runs learner-bot vs a local frozen snapshot; snapshot versions are broadcast with weights. Opponent matches cost the center nothing.
- **The distributed-specific hazard — code-version skew:** env + `PerceptionFilter` are replicated across machines; if versions differ, the replay buffer mixes inconsistent state definitions. Stamp every batch with `code_version` and have the learner **reject mismatches** (companion §8).
- **Migration:** hide the network behind an `ExperienceTransport` interface (`LocalTransport` → `RedisTransport`); single-machine → distributed is a config switch, not a rewrite. **Build single-machine first.**

See `minecraft-pvp-distributed-training.md` for Redis schema, message formats, heartbeat/reconnect, and the actor/learner pseudocode.

---

## 11. Compute & throughput expectations

- **Bottleneck is CPU/RAM (running Minecraft servers), not GPU.** The net is tiny (structured vector → MLP+LSTM); a modest GPU or even CPU saturates the gradient step. Replay of 1e7 structured transitions ≈ ~1.6 GB — fits in RAM.
- **No fast-forward** (server ≈ 20 TPS). At `ACTION_REPEAT`=4 → ~5 decisions/s/arena → ~0.43 M/day/arena. Recover throughput with **N parallel arenas** (8 arenas ≈ 3.2 M/day).
- **Rough sample budget per gate:** M2 ~0.05–0.2 M; M3 ~1–3 M; M4 ~5–20 M+. With ~8 arenas, M3 in a few days, self-play a few more.
- **Measure, don't trust estimates:** once M1 runs, log real transitions/s and steps-to-M2, then extrapolate from your own numbers.

---

## 12. Reproducibility

- **Seed everything** (Python, NumPy, PyTorch, env) and log seeds.
- For the paper: run **3–5 seeds per condition** across `{omniscient, perception-limited} × {scripted, self-play}`; report mean ± std with significance.
- Record `code_version` (git SHA + config hash) with every run and every checkpoint; pin the **Minecraft version** (main spec §16).

---

## 13. Debugging playbook (training-specific)

| Symptom | Likely cause | Action |
|---|---|---|
| Reward flat at M2; agent never approaches | reward/observation bug; opponent not in obs; bridge stale state | Verify random policy gets nonzero `damage_dealt` events; print the obs vector; confirm reset works |
| Q-values explode / diverge | lr too high, no grad clip, target updates too fast, exploding n-step | Lower lr, clip grad norm, slow target (τ↓ or C↑), reduce n |
| Agent spins in place | aiming reward farmed | Gate `c_aim` on visibility (already specified); lower or remove it |
| Agent runs away forever | `c_step` too small / flee not penalized | Increase per-step penalty; check terminal loss magnitude |
| Agent suicide-rushes | `c_step` too large or `c_dmg_in` too small | Lower `c_step`; raise damage-taken penalty |
| Learns vs dummy, fails vs scripted | overfit to a static target; no memory | Confirm DRQN/frame-stack active; check it re-acquires after losing sight |
| Self-play Elo stalls / oscillates | training both at once, or opponent pool too narrow | Ensure only one learner; widen pool; switch to PFSP |
| "It cheats / sees through walls" | derived feature leaks position; FOV/raycast wrong | Audit that derived features are computed post-gating; unit-test the raycast |
| Distributed: garbage learning after a redeploy | code-version skew across actors | Check `code_version` rejection counter; redeploy all actors |
| Replay full of stale data | dead actor reconnected with ancient weights | Confirm `MAX_STALENESS` filter active |

**General method:** plot reward components separately, sanity-check with a random policy first, and unit-test the `PerceptionFilter` (FOV, raycast, memory expiry) in isolation before blaming the learner.

---

## 14. Training-specific open/tunable decisions

1. `ACTION_REPEAT` / decision interval (reactivity vs sample count).
2. Reward coefficients — especially `c_step` (the make-or-break knob).
3. n-step `n`, target-update style (soft vs hard), PER on/off for the first runs.
4. DRQN vs frame-stacking; sequence length and burn-in.
5. Self-play opponent sampling (uniform vs PFSP) and snapshot cadence.
6. Fairness curriculum schedule (how aggressively to tighten FOV/`MEMORY_TTL`).
7. Whether to run the **PPO comparison arm** (recommended for the paper; same env, A/B).

---

### TL;DR for the implementer

Train a **Dueling DRQN** with **n-step Double-DQN** targets, **PER**, **Huber loss**, and **soft target updates**, on a structured observation produced by the shared `PerceptionFilter` (fairness = gated perception; privileged info only in reward, never in obs; derived features inherit gating). Reward is **damage-anchored** with a carefully tuned per-step penalty. Opponents progress **dummy → scripted → frozen-snapshot self-play (PFSP, Elo-tracked)**, one learner only. Throughput's sole lever is **parallel arenas**; the net is tiny so **CPU/RAM, not GPU, is the limit**. For multi-machine runs, split the loop into off-policy **actors + learner over Redis**, stamp `code_version` to defeat skew, and build single-machine first. Gate progress on **M1→M2→M3→M4** with the concrete win-rate/Elo criteria above, run **3–5 seeds per condition**, and **measure real throughput instead of trusting estimates**.
