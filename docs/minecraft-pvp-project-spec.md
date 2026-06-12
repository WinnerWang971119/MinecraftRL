# Minecraft PvP × Deep Q-Learning — Master Project Specification

**This is the umbrella document for the entire project.** It consolidates every decision and design from our planning: what we use and *why*, how we train, the fairness philosophy, hardware and time, deployment, and the accompanying research paper. Two companion docs hold the deepest implementation detail; this one ties it all together.

### Document map
| Doc | Scope |
|---|---|
| **this file** — `minecraft-pvp-project-spec.md` | the whole project: decisions + rationale, system, training, hardware, deployment, research plan, execution checklist |
| `minecraft-pvp-training-spec.md` | complete training internals (DQN equations, loss, PER, DRQN, training loop, debugging playbook) |
| `minecraft-pvp-distributed-training.md` | distributed transport (Redis schema, actor/learner pseudocode, reconnect/version-stamping) |

**Audience:** Claude Code (implementation) + the 8-person team.
**Conventions:** `TUNE` = empirical starting value.

---

## 0. What we are building

A reinforcement-learning agent that fights **1-v-1 melee PvP in real Minecraft (Java Edition)**. At the final exhibition, **visitors join a server and fight the agent in person**, so the production environment must be real Minecraft. Alongside the system we run a **human-subjects study** on *perceptual fairness* (see §12) — the research contribution that turns "we built a bot" into a paper.

### Three principles that drive everything
1. **One interface for training and deployment.** Train and demo through the *same* bot interface (Mineflayer). What we train is exactly what visitors fight — no sim-to-real gap, no second codebase.
2. **Perceptual parity, not state parity.** A protocol-reading bot is omniscient by default (knows exact enemy position/health through walls). That is a perception superpower, not skill. We strip *omniscience*, not *numbers*, via a single `PerceptionFilter` enforcing field-of-view + line-of-sight. **This is the soul of the project** — and also the manipulated variable in the research (§12).
3. **Tactical macro-actions, not raw control.** Discrete high-level actions (engage, strafe, retreat, attack…); low-level execution handled by Mineflayer plugins. Discrete → DQN-native; orders of magnitude fewer samples; invariant to engine details.

---

## 1. Design decisions & rationale (the journey)

This section records *why* each choice was made, including approaches we rejected, so the team and Claude Code don't re-litigate settled questions.

### 1.1 Interface: **Mineflayer** — chosen
We compared the Minecraft-RL landscape and it had shifted; there is **no turnkey PvP environment**, so the PvP scenario is custom work regardless. Decisions:

| Option | Verdict | Why |
|---|---|---|
| **MineRL** | ✗ rejected | Effectively unmaintained; pinned to old Minecraft + JDK 8. |
| **MineDojo** | ✗ rejected | Has combat tasks but old Minecraft, VLM-reward-oriented; too heavy and off-axis. |
| **Project Malmo** | ✗ rejected for us | Genuine gym-shaped platform with *native multi-agent*, but newest release is years old, pinned to old Minecraft. Using it for training re-introduces a train/deploy split and doesn't fit a "public joins a server" exhibition. |
| **CraftGround** | reserve only | Modern (latest Minecraft, fast, JDK 21, Gym+SB3), but multi-agent / custom-scenario support is its weak point — exactly what PvP needs. Keep as an optional fast-pretraining sim (§11). |
| **Mineflayer** | ✓ **chosen** | Connects as a player to a real server; supports Minecraft 1.8–1.21.11; gives **structured state** (entity tracking, blocks, own health) — no pixels/CV; has `mineflayer-pvp` + pathfinder + movement plugins; Python bridge exists. Multi-agent is *free* (spawn two bots). **Collapses train and deploy into one interface.** |

### 1.2 Observation: **structured state**, not pixels — chosen
Pixels (first-person CNN) is the "purest" fairness but blows up sample complexity and forces partial-observability learning from scratch — research-grade difficulty for a one-semester team. We get comparable fairness far more cheaply by feeding a **structured vector** and removing omniscience in software (the `PerceptionFilter`, §5). Pixels would also re-introduce a train/deploy domain gap (textures/HUD/resolution).

### 1.3 Rejected: **external screen-capture + simulated mouse/keyboard** for *training*
Tempting idea: treat Minecraft as a black box, read the screen, send virtual key/mouse events (this is conceptually how OpenAI's **VPT** acts). For *training* it is a trap, because it switches off nearly every lever that makes RL tractable:
- No faster-than-real-time (game runs 1×).
- Hard to parallelize (one screen + one input device per machine).
- Forces pixels → CNN → sample explosion.
- Real-time world doesn't wait while the agent "thinks."

Order-of-magnitude estimate (10 M steps): structured sim ≈ minutes–hours; CraftGround (~300 TPS) ≈ ~9 h; **external real-time capture ≈ ~185 h (~8 days) per run** — and pixels usually need far more than 10 M steps. **Why VPT could do it:** it pre-trained on ~70,000 hours of human video (behavior cloning) then RL-fine-tuned, with industrial compute — a precondition we can't replicate, so "VPT did it this way" is not evidence of feasibility for us. **Verdict:** screen-capture is great for an *optional demo wrapper*, never for training.

### 1.4 Action: **discrete tactical macros** — chosen (see §6)
Discrete → DQN-native; far fewer samples than low-level/pixel control; Mineflayer plugins execute the low level, shrinking any sim-to-real gap.

### 1.5 Algorithm: **DQN family** (with optional PPO comparison) — chosen
DQN is mandated-friendly and discrete-action-native. Honest caveat: for adversarial self-play, **PPO is often more stable**; DQN is fine for vs-fixed-opponent and tractable with macros. DQN's **off-policy** property is also what makes distributed training easy (§10). We additionally run an optional **PPO arm** as an A/B comparison — a strong result for the paper and a way to spread work.

---

## 2. System architecture

Same Python brain + `PerceptionFilter` wrapped around a Mineflayer bot, in two run modes.

**Training mode**
```
                       × N parallel arenas
        Minecraft Server (Paper) ── flat arena · fixed gear
                 ▲ actions          │ raw protocol state
        Mineflayer (Node) ── learner bot + opponent bot (scripted / frozen snapshot)
                 ▲ actions          │ raw state + events   (socket/RPC)
        Python:  PerceptionFilter (FOV + raycast)   ← fairness
                 Gym env (reset/step/reward/done)
                 DQN (replay · target · LSTM) · logging · eval
```

**Deployment / exhibition mode**
```
        Minecraft Server (offline-mode) ── visitors type a name and join
                 ▲ actions          │ raw state
        Mineflayer:  agent bot  +  human players
                 ▲ actions          │ raw state   (socket)
        Python:  PerceptionFilter (the SAME module)   ← zero gap
                 trained policy (inference only, real-time)
```

**Shared core = `PerceptionFilter` + policy network.** That sharing is the source of "zero gap." Real-time is fine for *inference* (a PvP match is real-time; a forward pass is sub-ms); the real-time cap only limits *training throughput* (§11).

---

## 3. Technology stack

| Layer | Choice | Notes |
|---|---|---|
| Game server | **Paper** (or Spigot) | flat arena, fixed gear, set spawns; **pin one Minecraft version** for the whole project; exhibition server in **offline-mode** (visitors join with just a username). |
| Bot interface | **Mineflayer** (Node.js) | player over protocol; MC 1.8–1.21.11; structured state. |
| Plugins | `mineflayer-pvp`, `mineflayer-pathfinder`, movement | execute macro-actions. |
| Bridge | local **socket / JSON-RPC** (TCP/ZeroMQ) | Node ↔ Python; one per arena. |
| Brain | **Python 3.11+, PyTorch** | DQN family. |
| Distributed | **Redis** broker (companion doc) | actor/learner experience + weights. |
| Logging | TensorBoard / W&B | loss, win rate, Elo, throughput. |

---

## 4. The Node ↔ Python bridge

Node emits **raw** state; Python owns fairness, reward, learning. **Privileged information is allowed in the reward but forbidden in the observation** (§5). Mineflayer is event-driven with no native synchronous `step()`, so Node aggregates events over a fixed **decision interval** (`TUNE` ≈ 4 ticks ≈ 200 ms) and replies per `step`.

Messages (Python→Node): `reset`, `step{action}`, `close`.
Raw state (Node→Python): `self{pos, yaw, pitch, velocity, on_ground, health, held_item, attack_cooldown}`, `opponent{...raw, incl. true health}`, `events{damage_dealt, damage_taken, i_died, opponent_died}`, `arena{wall_distances}`, `tick`. (Full schema in the training/companion docs.)

---

## 5. Observation & PerceptionFilter (the soul module)

**Rule for every feature:** *"Could a human at the screen, right now, perceive this?"* If yes → expose; how the agent infers/predicts from what it saw is **skill, and skill is fair**. Remove omniscience, not numbers.

**Per-feature degradation knobs:** `FULL` / `GATED` (only in FOV + clear line-of-sight) / `MEMORY` (last-seen value + seconds-since) / `NOISY` / `COARSE` / `NONE`.

**Assignment:**
- **Self** (`FULL`): health, facing (yaw/pitch as sin/cos), local velocity, on-ground, held item, attack-cooldown progress.
- **Opponent** (`GATED → MEMORY`): position/facing/velocity only when visible; else last-seen pos + age, then absent. Always include a `visible` flag + `time_since_seen`.
- **Opponent health:** `NONE` by default (vanilla shows no enemy health bar) — infer from hits. Optional: expose via a server **scoreboard nametag** (then it's on-screen and fair).
- **Derived** (`in_range`, `in_crosshair`): inherit the same gating; compute **after** gating. *Leaking position through a derived feature is the most common fairness bug.*

**Implementation:** FOV cone ≈ 70° (`TUNE`); line-of-sight by raycast for solid blocks; `MEMORY_TTL` ≈ 5 s. Encode angles as sin/cos, positions in local frame, normalize ≈ `[-1,1]`. **Memory is mandatory** (info expires) → use an **LSTM/DRQN** head or frame-stacking. `observation_spec.py` is the single source of truth, imported by the net, the env, and every distributed actor.

**Fairness-as-curriculum (optional):** start with looser FOV/longer memory so the agent *can* learn, then tighten toward human limits. Train first, blind down second.

---

## 6. Action space (tactical macros)

Discrete set (DQN-native); each maps to Mineflayer plugin calls:
`{0 idle, 1 approach, 2 retreat, 3 strafe-left, 4 strafe-right, 5 attack, 6 jump, 7 turn-to-last-seen}`. Keep it small (≈6–8) initially; add finer macros only if the agent plateaus.

---

## 7. Reward

Dense, damage-anchored, minimal (to avoid hacking):
`r = c_dmg_out·damage_dealt − c_dmg_in·damage_taken − c_step + c_aim·1[visible & in_crosshair] + R_terminal`.
Starts: `c_dmg_out=1, c_dmg_in≈1, c_step≈0.005, c_aim≈0.01, R_terminal=±(5–10)`, timeout = draw (0). Prefer **potential-based shaping** for positional terms. Watch: `c_step` too big → suicide-rush, too small → runs away; always-on aim reward → spin-to-farm (hence visibility-gated, tiny). (Full detail: training spec §3.)

---

## 8. Agent / algorithm

**Network:** MLP encoder (256,256) → **LSTM** head (DRQN; or frame-stack) → **Dueling** value/advantage heads → Q over macros. Small net by design.
**DQN ladder:** vanilla → Double + Dueling → Prioritized Replay → DRQN(LSTM) → n-step. Final target: **n-step Double-DQN**, **Huber** loss, **soft target update** (τ≈0.005), grad-norm clip. ε-greedy 1.0→0.05; in distributed mode, **per-actor ε** for exploration diversity (Ape-X style).
**Optional PPO arm** on the same env for an A/B comparison.
*(Exact equations, PER weights, DRQN burn-in, hyperparameter table → training spec §5–6.)*

---

## 9. Opponent curriculum & self-play

One learner throughout — **never train two policies at once** (non-stationary, unstable).
- **Stage 0 — stationary dummy.** Validates bridge + reward. Exit: ≥95% win / 100 eps (**M2**).
- **Stage 1 — scripted bot** (chase + attack-in-range + probabilistic strafe/jump + flee-when-low). Exit: ≥70% win / 200 eps (**M3**).
- **Stage 2 — self-play, frozen snapshots.** Opponent = a frozen past copy from a snapshot pool; refresh on a cadence. **PFSP** sampling (favor opponents beaten ~40–60% of the time). Track **Elo**. Exit: Elo trend rising (**M4**).

Humans enter only at the exhibition — pure inference, real-time-safe.

---

## 10. Distributed training (arenas on members' PCs)

**One learner, many actors, Redis broker.** Actors run env + a policy copy + a local opponent; batch experience **up**, pull weights **down**. The **off-policy** learner trains on pooled replay and republishes weights periodically — so actors tolerate stale weights and flaky home internet (embrace policy lag). Actors **dial out** (NAT-friendly); the learner never blocks on any actor; heartbeats reap dead ones; at-most-once delivery is fine (a lost batch is just less data). Self-play distributes cleanly (each actor runs a local frozen-snapshot opponent).

**The distributed-specific hazard — code-version skew:** env + `PerceptionFilter` are replicated across machines; differing versions silently mix inconsistent state definitions. Stamp every batch with `code_version`; the learner **rejects mismatches**.

**Migration:** hide the network behind an `ExperienceTransport` interface (`LocalTransport` → `RedisTransport`); single-machine → distributed is a config switch. **Build single-machine multi-arena first.** *(Redis schema, message formats, reconnect logic → distributed companion doc.)*

---

## 11. Hardware, compute & time

**The bottleneck is CPU/RAM (running Minecraft servers), not GPU.** The net is tiny (structured vector → MLP+LSTM, a few MB); a modest GPU or even CPU saturates the gradient step. Replay of 1e7 structured transitions ≈ ~1.6 GB → fits in RAM. **You are buying core count to host arenas, not a graphics card.**

**No fast-forward** (server ≈ 20 TPS). At `ACTION_REPEAT`=4 → ~5 decisions/s/arena → ~0.43 M/day/arena. **Parallel arenas** are the only throughput lever:

| Parallel arenas | transitions/day | reach ~5 M | reach ~10 M |
|---|---|---|---|
| 1 | ~0.4 M | ~12 days | ~25 days |
| 8 | ~3.2 M | ~1.5 days | ~3 days |
| 16 | ~6.4 M | <1 day | ~1.5 days |

**Sample budget per gate** (structured + macros buy 1–2 orders of magnitude vs pixels): M2 ~0.05–0.2 M; M3 ~1–3 M; M4 ~5–20 M+. So **~8 arenas → M3 in a few days, self-play a few more**; a single arena is week-scale (too slow to iterate).

**Arenas per machine** (`TUNE` ≈ 1–2 cores + 2–4 GB each; minimal view/sim distance, flat world, no mob spawning):
- workstation (8–16 cores, 32–64 GB): ~4–8 arenas
- 32-core / 128 GB server: ~16–32 arenas
- **arenas distribute across machines** — members' PCs each run 1–2; a central learner collects (§10).

**Three hardware tiers (pick one):**
- **Budget:** learner on any modern-GPU machine (even a laptop 3060 / Colab); arenas spread across members' PCs to total 4–8. Works; slow; don't expect many full retrains.
- **Comfortable (recommended):** rent a **CPU-heavy cloud VM** (16–32 vCPU) for the sprint weeks to host 8–16 arenas; ordinary GPU. Spot/preemptible is cheap and spares laptops.
- **Lab cluster (luxury):** 32–64 cores. Overkill for one run, but valuable for the paper.

**Time is calendar, not just compute hours.** The real long pole is **debugging and iteration** (bridge, reward, PerceptionFilter) — many short runs to tune. Long runs (days) come last. Schedule heavy compute for the **final third** of the semester, once long runs won't be wasted on bugs.

**The paper multiplies compute.** Seeds × conditions ≈ 3–5 seeds × {omniscient, limited} × {scripted, self-play} ≈ up to ~20 full runs. A single run fits the budget tier; **20 runs is the real reason to want more cores** (parallel runs), not single-run difficulty.

**Measure, don't guess.** Once M1 runs, log real transitions/s and steps-to-M2; extrapolate wall-clock from your own numbers, not these estimates.

---

## 12. Research / paper plan (the human-subjects study)

This is what makes it a paper, not just an artifact: a falsifiable claim + controlled experiment + human data. **You don't run a separate study — you *instrument the exhibition*.** The booth is the lab; visitors are the participants.

### 12.1 Research question & hypotheses
> *Does constraining an RL agent to human-equivalent perception (FOV + line-of-sight, no enemy-health, expiring memory) change (a) its combat performance and (b) how human opponents perceive it — fairness, difficulty, fun, human-likeness?*

- **H1:** the perception-limited agent has lower raw win rate but higher human-rated fairness/fun.
- **H2:** it exhibits human-like behaviors the omniscient agent lacks (search after losing sight, re-acquiring line-of-sight). *Hypotheses must be falsifiable.*

### 12.2 Conditions (the ablation backbone)
Matrix: **{omniscient vs perception-limited} × {scripted-bot vs self-play}**. The omniscient/limited contrast is a single `PerceptionFilter` flag — the cheapest, most important figure. **3–5 seeds per cell**, report mean ± std + significance.

### 12.3 Metrics
- **Automatic (free, logged):** win rate, Elo, time-to-kill, damage-dealt/taken ratio, hit rate, mean engagement distance, frequency of "lost-sight → re-acquire" search behavior.
- **Human (collected at the booth):** per-version Likert (fairness, difficulty, fun, human-likeness) + win/loss vs the human.

### 12.4 The killer experiment — blind, within-subjects A/B
- **Within-subjects:** each visitor fights **both** versions (omniscient & limited) in randomized order. Far higher statistical power than between-subjects (each person is their own control) → works with limited foot traffic, and enables a direct **forced-choice**: *"Which one felt fairer / more fun / more human?"*
- **Blind:** the visitor is **not told** which version they're fighting (removes expectation bias).
- **Counterbalancing:** ~half play omniscient-first, ~half limited-first, to cancel order/warm-up effects.

### 12.5 Confound to control: "I won, therefore I liked it"
The limited agent is weaker and easier to beat, which can inflate its preference scores. Mitigate by (a) recording **match outcome as a covariate** and controlling for it in analysis, and (b) **asking "fair/human-like" separately from "fun"** so you can disentangle *winning* from *human-likeness*. Demonstrating this separation is what makes reviewers trust the result.

### 12.6 Ethics / consent (proportionate)
Anonymous; collect **no personal data**; one short consent line ("course project collecting anonymous feedback on an AI opponent; you may skip anytime"); a random session id per participant. **Confirm whether your department/school requires ethics/IRB sign-off** for anonymous game feedback (usually light or exempt) — settle this before Demo Day.

### 12.7 Booth instrumentation
Match length capped (60–120 s or first to X); ~5 min/person (two matches + 30 s survey on a tablet). The booth app **auto-assigns version, randomizes order, hides all labels, and logs everything**. **Pilot** the full flow with classmates first. Collect as much N as foot traffic allows.

### 12.8 Statistics (light, honest)
Paired Likert → **Wilcoxon signed-rank** (ordinal) or paired t-test; forced-choice → **binomial/sign test**; report mean ± std + effect size + p; **don't over-claim significance**. Limitations to write: a noisy booth, skill variance, self-selection — acceptable at workshop level; state them, don't hide them.

### 12.9 Paper structure
Abstract → Intro (motivation + contributions) → Related Work (Minecraft RL platforms; competitive self-play incl. failure modes; POMDP/DRQN; **believable / human-like / fair game AI** — the line you extend) → Method (environment, PerceptionFilter, agent) → Experimental Setup (conditions, metrics, seeds, human protocol) → Results (answer each question) → Discussion & Limitations → Conclusion & Future Work. Don't claim "no one has done this"; claim you advance toward **perceptually-fair adversarial agents**.

> A dedicated companion (`experiment-and-paper-plan.md`) with the full questionnaire, consent template, blinding procedure, and per-section bullet outline can be produced next.

---

## 13. Repository structure

```
mc-pvp-dqn/
├── README.md                 # pinned MC version, setup, run commands
├── bridge/                   # Node.js + Mineflayer
│   ├── bot.js · raw_state.js · actions.js · arena.js
├── env/                      # Python gym side
│   ├── mc_pvp_env.py
│   ├── perception_filter.py  # FOV + raycast + memory   ← SOUL MODULE
│   ├── observation_spec.py   # single source of truth for the vector
│   └── reward.py
├── agent/                    # dqn.py · replay.py · train.py · config.py
├── opponents/                # scripted_bot.py · snapshot_pool.py
├── distributed/              # actor.py · learner.py · transport.py · serialization.py · dist_config.py
├── eval/                     # evaluate.py · logging.py
├── server/                   # Paper setup, arena, offline-mode demo
├── deploy/                   # exhibition.py (inference loop for public PvP)
└── study/                    # booth app, survey, version assignment, logging  (research)
```

`observation_spec.py` is imported by `perception_filter.py`, `agent/dqn.py`, `deploy/exhibition.py`, and every distributed actor — so training, deployment, and all machines can never drift.

---

## 14. Roadmap (milestone gates)

- **Phase 1 → M1:** Paper server + arena; Mineflayer connects two bots; socket bridge round-trips; **random policy** runs end-to-end vs a dummy. *Everyone has working code.*
- **Phase 2 → M2:** observation_spec + PerceptionFilter + reward; vanilla DQN ≥95% vs stationary dummy.
- **Phase 3 → M3:** Double/Dueling (+PER) + DRQN; ≥70% vs scripted bot.
- **Phase 4 → M4:** frozen-snapshot self-play (PFSP); Elo rising.
- **Demo Day:** offline-mode exhibition, open public PvP, **blind A/B study running**, plus an analysis overlay (learning curves, live Q-values / chosen macro).

(Research workstream runs in parallel with Phases 2–4.)

---

## 15. Team workstreams (8 people)

| Workstream | People | Owns |
|---|---|---|
| Environment / bridge | 2 | `bridge/`, `server/`, arena, socket protocol, **distributed transport** (highest risk — staff well) |
| Observation / PerceptionFilter | 1–2 | `perception_filter.py`, `observation_spec.py`, FOV+raycast, degradation knobs |
| DQN core | 2 | `agent/` net, replay, target, LSTM |
| Reward + opponent curriculum | 1–2 | `reward.py`, `opponents/`, self-play schedule |
| Training infra / eval | 1 | parallel arenas, logging, win-rate/Elo, curves |
| Research / study + demo / viz | 1 | `study/` booth app, A/B + survey, `deploy/`, Q-value overlay, write-up |

Integrate weekly so workstreams don't diverge.

---

## 16. Risks & mitigations

| Risk | Mitigation |
|---|---|
| **Training throughput** (no fast-forward) — biggest | macro-actions + N parallel arenas |
| **Self-play instability** | freeze the opponent (snapshots); never train both at once |
| **Reward hacking** (spin/run-away) | minimal damage-anchored shaping; visibility-gated aim reward; potential-based; tune `c_step` |
| **Perception leakage** via derived features | compute after gating; all fairness logic in one `PerceptionFilter`; unit-test raycast |
| **Train/deploy drift** | one interface (Mineflayer) + shared `observation_spec` + shared filter; pin one MC version |
| **Code-version skew** (distributed) | stamp `code_version`; learner rejects mismatches |
| **Mineflayer event model** (no native step) | fixed decision interval; Node aggregates events |
| **Win-vs-likeness confound** (study) | outcome as covariate; ask fairness/human-likeness separately from fun |
| **Compute for the paper** (20 runs) | comfortable/cluster tier for parallel seed runs |

---

## 17. Open decisions (settle before/early in build)

1. **Pin one Minecraft version** (≤1.21.11) for training *and* exhibition; record in README.
2. Opponent-health exposure: default `NONE`, or scoreboard-nametag (then fair).
3. Decision interval / `ACTION_REPEAT` (≈4 ticks).
4. FOV angle (~70°) and `MEMORY_TTL` (~5 s).
5. Optional fast-pretraining sim (CraftGround) only if real-time training is too slow even parallelized.
6. PPO comparison arm: yes/no (recommended for the paper).
7. Self-play opponent sampling (uniform vs PFSP) and snapshot cadence.
8. Ethics/IRB requirement for the booth study.

---

## 18. Consolidated execution checklist

**Decide first (gates everything)**
- [ ] Confirm paper axis = perceptual fairness + blind human preference (Condition A)
- [ ] Pin one Minecraft version; opponent-health choice; decision interval
- [ ] Ethics/IRB check for anonymous booth feedback
- [ ] Assign 8 people to workstreams; set weekly integration

**Phase 1 → M1 (bridge works)**
- [ ] Paper server, flat arena, fixed gear, spawns
- [ ] Mineflayer + plugins; two bots; socket bridge (raw state + events)
- [ ] Python Gym skeleton; **random policy** end-to-end vs dummy
- [ ] Scaffold the repo

**Phase 2–3 → M2, M3 (learns to fight)**
- [ ] `observation_spec.py` (single source of truth)
- [ ] `PerceptionFilter` (FOV + raycast + memory); unit-test it
- [ ] Verify derived features inherit gating (no leak)
- [ ] `reward.py` (damage-anchored; check run-away / spin)
- [ ] Macro-action → Mineflayer mapping
- [ ] DQN core (MLP+LSTM, replay, target) → Double/Dueling → PER → DRQN
- [ ] Scripted bot; reach M2 then M3

**Phase 4 → M4 (self-play)**
- [ ] Frozen-snapshot pool + PFSP self-play (one learner only)
- [ ] N parallel arenas; Elo tracking
- [ ] (optional) PPO arm, same env, A/B

**Distributed (when scaling beyond one machine)**
- [ ] `ExperienceTransport` interface; LocalTransport works single-machine first
- [ ] Redis broker (cloud, password-protected); actor/learner split
- [ ] Heartbeat/reconnect; `code_version` stamping + rejection
- [ ] Central logging of per-actor throughput

**Research / study (parallel with Phases 2–4)**
- [ ] Research question + falsifiable H1/H2
- [ ] Two conditions via the `PerceptionFilter` flag
- [ ] **Within-subjects + counterbalanced + blind** design
- [ ] Questionnaire (Likert + forced-choice); separate fairness/human-likeness from fun
- [ ] Record outcome as covariate (win-vs-likeness confound)
- [ ] Booth app: auto-assign version, randomize order, hide labels, log all
- [ ] Consent template; random session id; no personal data
- [ ] Auto-metric definitions + extraction from logs
- [ ] **Pilot** the full flow with classmates
- [ ] Analysis plan: Wilcoxon/paired-t; binomial for forced-choice; effect size + p

**Demo Day**
- [ ] Offline-mode exhibition server
- [ ] Inference-only deploy; same `PerceptionFilter`
- [ ] Q-value / chosen-macro overlay
- [ ] Open public PvP; run blind A/B; collect surveys + match data

**Writing**
- [ ] Related-work positioning (precise citations — can be assisted)
- [ ] Draft sections per the outline (§12.9)
- [ ] `experiment-and-paper-plan.md` companion (can be produced next)

---

### TL;DR

Build everything on **Mineflayer** (one interface for train + deploy; structured state, no pixels; rejected MineRL/MineDojo/Malmo and screen-capture-for-training, all for documented reasons). Route observations through a single **PerceptionFilter** that enforces FOV + line-of-sight — **perceptual parity, not state parity** — which is both the fairness mechanism and the study's manipulated variable. Actions are **discrete macros**; the learner is a **Dueling DRQN** with n-step Double-DQN + PER; opponents go **dummy → scripted → frozen-snapshot self-play (PFSP, Elo)**. Throughput's only lever is **parallel arenas**; **CPU/RAM, not GPU, is the limit**; multi-machine runs use **off-policy actors + learner over Redis** (watch code-version skew); schedule heavy compute for the final third and **measure real throughput**. Turn it into a paper by **instrumenting the exhibition** as a **blind, within-subjects A/B** on perceptual fairness, controlling the win-vs-likeness confound. Gate on **M1→M2→M3→M4**. Deepest detail lives in the training and distributed companion docs.
