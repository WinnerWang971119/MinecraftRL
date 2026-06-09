# Brainstorm: Minecraft PvP RL — how to start

**Seed (verbatim):** How do I start

**Context summary:**
- Two complete specs already exist in-repo (`minecraft-pvp-project-spec.md`, `minecraft-pvp-training-spec.md`); a third distributed companion doc is referenced but not present. No code yet — only docs.
- Project: 8-person, ~one-semester RL agent that fights 1v1 melee PvP in *real* Minecraft (Java) via Mineflayer. The PerceptionFilter (FOV + line-of-sight, no omniscience, expiring memory) is both the technical soul and the manipulated variable in a blind, within-subjects human-subjects study (the paper).
- All major decisions are made & justified: Mineflayer (one interface for train+deploy); structured state over pixels; discrete tactical macros; Dueling DRQN + n-step Double-DQN + PER; opponent curriculum dummy→scripted→PFSP self-play; parallel arenas as the ONLY throughput lever; CPU/RAM-bound not GPU; gates M1→M2→M3→M4.
- mem0: user has built DQN demos before (CartPole, gridworld for AI Club Lesson 3) and hit gotchas (per-EPISODE ε decay, action_space.seed, vanilla-DQN instability). Algorithm familiar; real-Minecraft plumbing + scale is the new ground.
- The spec PRESCRIBES a start (Phase 1 → M1: Paper server + arena, two Mineflayer bots, socket bridge round-trips, random policy end-to-end). Question is whether that prescribed start is the highest-leverage first move given an 8-person team and a real-time throughput ceiling.

**Clarified framing:** The project has a complete, decision-locked spec and 8 people ready to begin immediately. The user wants the smartest FIRST move — one that front-loads the project's scariest unknown (the thing most likely to sink it) and de-risks/kills it before weeks are sunk, while ALSO giving 8 people meaningful parallel work from day one so nobody is bottlenecked waiting on a single risk spike. Core tension: de-risking is usually a narrow 1–2 person spike, but 8 idle people need parallel tracks. Identify what the scariest unknown actually is, how to probe it fastest, and how to structure day-one parallel work around (or independent of) that probe.

## Perspectives (round 1)

### Critic (strongest first)
1. **Scariest unknown is mis-identified.** M1 (bridge round-trips) is plumbing — a *known* unknown. The project-killer is whether DRQN+PER+n-step on a 30-40 float POMDP actually *learns* to win within the real-time sample budget (M3 ~1-3M, M4 ~5-20M+ at ~0.43M/arena/day) with a beginner team. Spec even admits PPO is more stable for self-play yet locks DQN. A go-fast kickoff celebrating green M1 de-risks the cheap unknown, leaves the expensive one untouched — and the learning-feasibility spike is itself gated behind the bridge, so day-one parallelization is partly fiction.
2. **Can't parallelize 8 people against an observation contract that doesn't exist.** `observation_spec.py` is imported by net, env, filter, deploy, every distributed actor — most load-bearing file, absent on day one. DQN/filter/reward people would write against an *imagined* contract; when the real Mineflayer raw-state schema lands, their modules need rework = integration debt, not parallel work.
3. **Brooks's Law: 8 RL-beginners, one serial pipeline, 28 communication paths.** Architecture is serial (Mineflayer→bridge→filter→env→DQN→eval); a defect anywhere blocks everyone downstream. Weekly integration is too slack to catch drift; no senior to adjudicate contract disputes.
4. **Throughput math is optimistic and compounds.** Each Paper server ≈ 2-4 single-thread cores; 8 arenas = 8 servers + 16 protocol clients + Python, on students' PCs. When an arena drops below 20 TPS under contention, the 200ms interval silently stretches, MDP time-homogeneity breaks, transitions/day collapses — discovered only *after* building the harness.
5. **Bridge is a single point of failure + bus-factor-of-one.** Mineflayer has no synchronous step (continuous `physicsTick`); the bridge must manufacture step boundaries via event aggregation — the deepest dark art (damage aggregation, 1.9+ attack cooldown, reset semantics). Multi-bot-in-one-process has reported crashes/connection failures, amplifying the hardest least-shareable component.
6. **Study/deploy track produces throwaway work.** Booth validity hinges on the omniscient-vs-limited `PerceptionFilter` flag whose semantics the filter team is still discovering; needs two working policies that only exist post-M4. Front-loading booth infra builds a measurement apparatus for an experiment months away.

### Feasibility — VERDICT: **Conditional** (no unconditional hard blockers)
Must-be-true: (1) Author the frozen interface contracts on **day 1** before any track writes implementation — `observation_spec.py` + bridge JSON schema + action enum + reward signature (pure data, zero deps, a few hours, one person → unblocks the other 7). (2) The de-risk spike's exit criterion is a **measured number** (transitions/s, p99 Node→Python round-trip at 200ms, TPS under N arenas), NOT a passing test. (3) Downstream tracks build against **mocks/stubs** (LocalTransport + fake-env emitting spec-shaped vectors) for 1-2 weeks. (4) **Pin one MC version** on day 1.
Solvable challenges: Mineflayer no-step desync (spec's event-aggregation pattern is correct; spike must verify a macro completes within 200ms and damage events aren't dropped/double-counted at boundaries — the single most important thing to validate). Throughput plausible but needs per-arena process isolation + core pinning + measured TPS. The spike-vs-idle tension resolves into a **contract-gated vertical slice (= M1 itself)**: 1 track probes the unknown, 6 build mergeable components against stable stubs.

### Thinker (branches)
1. **Freeze the interface contract on day one** — `observation_spec.py` + bridge JSON schema as the first committed, versioned artifact; 7 build against a schema-valid fake bridge while 1-2 spike the real connection.
2. **Tracer-bullet a vertical slice through every layer** — thinnest hollow end-to-end path (random policy→bridge→Paper→dummy→reward→toy replay→no-op grad→logged win-rate); M1 as a skeleton each workstream stubs.
3. **Throughput-first: measure real transitions/s before writing the agent** — one arena + do-nothing 200ms loop; log decisions/s, RAM/arena, arenas-before-TPS-drop. Kills/confirms the calendar in week one.
4. **Build the off-Minecraft mock env as the real product** — fast pure-Python PvP toy (two dots, FOV, melee, health) implementing the exact env contract; validate the whole DQN ladder + filter + self-play at thousands of steps/s; survives as the CI fixture + debugging harness. Reframes scariest unknown as "does our RL stack even learn?"
5. **Attack the PerceptionFilter raycast as the soul-and-study risk** — build + adversarially unit-test the filter in isolation, with a leak-detection battery asserting no derived feature reveals position when `visible=false`. Protects the research contribution.
6. **Start from the booth and work backward (research-first)** — build the study harness end-to-end, pilot with classmates fighting a *scripted* bot (no RL); forces the omniscient/limited flag, metric schema, and IRB question in week one.
7. **One-week timeboxed kill-the-project spike** — deliberately try to *disprove* feasibility in 5 days (events readable at 200ms? 10k steps no leak/desync? N>4 arenas hold 20 TPS?). Adversarial probe teams. Success = "found no blocker"; failure = "pivot now, not week 8."
8. **Assign ownership by risk gradient, not by component** — rank unknowns by scariness × time-to-discover-broken; staff top risks heaviest; risk register *is* the work-breakdown structure.
9. **Pin the MC version + lock determinism scaffolding first** — version pin, Paper build, fixed gear/arena/spawns, `code_version` stamping, seed logging, repo scaffold. Boring, possibly highest-leverage; everyone builds inside a fixed reproducible substrate.
10. **Build the bridge-resilience/chaos harness, not the bridge (speculative)** — start with the fault injector (drop messages, stall Node, kill server mid-episode, stale weights, corrupt code_version) + recovery assertions; design the bridge for resilience before a fragile happy-path locks in.

## Live direction
**CHOSEN: Contract + real vertical slice (no mock).**

Day 0/1 — freeze 4 contract artifacts before any implementation: `observation_spec.py` (§5 slots+indices), bridge JSON schema (§4 raw state), `actions.py` macro enum (§6), `reward.py` signature (§7). Same session: pin ONE MC version (§17 #1), ACTION_REPEAT≈4 ticks (§17 #3), `code_version` convention. Output = one frozen contract PR everyone branches from.

Spine — tracer-bullet vertical slice (= M1, §14 Phase 1): random policy → bridge → real Paper → stationary dummy → reward → toy replay → no-op grad → logged win-rate. Real stack from first commit; no mock.

Day-1 allocation (contract-gated fan-out; 6 of 7 tracks non-blocking):
- Env & bridge — 2 (highest risk): Paper+arena, two bots, socket schema, event-aggregation step boundary. EXIT = measured number (transitions/s, p99 round-trip @200ms, damage events not dropped/double-counted at boundaries).
- PerceptionFilter — 1-2: FOV+raycast+memory vs synthetic geometry + leak battery (no derived feature leaks pos when visible=false). No bridge dep.
- DQN core — 2: Dueling-DRQN + replay + target + loop vs one hardcoded spec-shaped fixture vector. Unit fixture, not mock env.
- Reward + scripted opponent — 1-2: reward.py (§7 c_step knob) vs hand-authored event dicts; Stage-1 scripted bot (§7.2).
- Eval/infra + research scaffold — 1: logging, win-rate/Elo, repo scaffold (§13), seed+code_version, booth-app skeleton stub.

Two early gates = joint risk probe: Week 1 M1 + the number; Week ~2 M2 (≥95% vs dummy, ~0.05-0.2M samples ≈ ~1 day on one real arena) answers Critic's real unknown "does our stack learn?" at real-time speed, no mock.

Kill-risks (Critic carried fwd): freeze (not "mostly agree") the contract & version it; spike returns a NUMBER not a green check; per-arena process isolation + core pinning + measured TPS; adversarially unit-test damage-event aggregation at step boundaries (silent reward corruption).

## Killed / parked
- **Branch 4 — off-Minecraft mock/sim env as a deliverable** — KILLED by user. Violates the spec's core principle: ONE interface (Mineflayer) for train AND deploy, no second codebase, no sim-to-real gap (§0 principle 1; §1.2; §1.3). CraftGround is reserved ONLY as an optional fast-pretraining sim (§1.1, §11), not a development mock.
- **Branch 1+4+2 "mock-env combo" and Feasibility's "build against a mock env" resolution** — PARKED/adjusted. Contract-first (branch 1) survives; "build against mocks" is replaced by "build real components tested with tiny hand-authored fixtures + integrate fast on the real bridge."

## Decisions & debate verdicts
- **Handoff:** user chose **Hand to /plan** (kickoff plan, contract→vertical-slice→M1→M2). Plan in progress.
- **Plan Q&A decisions:** Claude builds the foundation (scaffold + 4 contract artifacts + slice skeleton) then humans take tracks; git init + GitHub repo (T0); full §13 skeleton; pin MC **1.21.1** (verify minecraft-data/Paper/plugins day 1); keep **1.9+ attack-cooldown** combat; benchmark host = Diego's laptop (**Intel Core Ultra 7 258V, 8c/8t, ~32GB → ~2-4 arena baseline**, not 8).
- **/debate — reset mechanism (custom Paper plugin vs Mineflayer commands), kickoff scope:** VERDICT **Lean No (4.5/10)** on building the plugin for the kickoff. Use Mineflayer server commands + a read-back verification gate (health==20/pos==spawn/inventory==template before episode start), behind a clean reset interface (env.reset() → bridge `reset` RPC). Adopt the plugin at N-parallel-arena scaling or if the gate proves flaky/slow. Resolved: a reset plugin needs only Bukkit API (no NMS/paperweight); atomicity unnecessary for a paused/unobserved reset window; spam/latency negligible at kickoff scale.
