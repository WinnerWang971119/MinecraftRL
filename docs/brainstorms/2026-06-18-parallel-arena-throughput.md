# Brainstorm: Parallel-arena training throughput (issue #4)

**Seed (verbatim):** issue#4 — "A: Parallel-arena training throughput (THE BLOCKER - approach open)".
Phase-1 critical path. A single arena does ~0.4M transitions/day → M3 ~12 days, M4 weeks.
Parallel arenas are the ONLY throughput lever (project spec §11). Target a measured
multi-arena speedup. The HOW is open — start with a design spike picking the approach.
Route 1 (do first): single-machine multi-arena — N Paper servers on N ports + N bridges
+ a vectorized env/actor pool feeding ONE learner via LocalTransport. Route 2 (later):
distributed actor/learner over Redis. EXIT = a measured number: transitions/s vs arena
count, and max arenas sustained at ≥19 TPS over a ≥10-min run. Not a green check.

**Context summary:**
- `eval/benchmark.py` ALREADY accepts `--arenas N` and drives N `TcpBridgeClient`s on
  `port+i` — but its driver loop is a **synchronous round-robin** (send→recv blocks per
  arena), so it does NOT overlap the 200 ms waits. As written it would show ~no speedup.
- `agent/train.py` builds ONE `MCPvPEnv` + one eval env; the rollout loop is single-env.
  Vectorizing = N envs feeding one replay buffer + one learner.
- `MCPvPEnv`/`TcpBridgeClient` are blocking (send→recv). Running N concurrently needs
  threads / asyncio / processes, OR a batched send-all-then-recv-all vector env.
- `bridge/run.js` hard-codes `BridgeServer` on 5555 + one learner_bot + one dummy_bot.
  GOTCHA (confirmed in `bridge/transport.js`): one TCP connection per server → multi-arena
  = N bridge procs on N ports, not one bridge multiplexing.
- `distributed/{transport,actor,learner,dist_config}.py` and `bridge/arena.js` are
  DEFERRED stubs. `ExperienceTransport` ABC exists; `LocalTransport` not built.
- Hidden design fork the issue glosses: **N Paper servers** vs **1 Paper server hosting N
  spatially-separated arenas** (shared 20-TPS tick budget — more entities = TPS drop).
- Resilience contract: `step()` raises on mid-episode reply loss → one arena dying would
  abort the whole pool unless the vector loop isolates per-arena faults.
- Project = ~4-5-student learning project, training-only through M4; favor simplicity over
  production distributed systems. Live MC stack is NOT runnable in-session (design + offline
  harness only). Dev laptop = Core Ultra 7 258V, 8c/8t → ~2-4 arenas; cloud VM tier = 16-32.

**Clarified framing:** Build single-machine multi-arena (Route 1) optimized for the dev
laptop (Core Ultra 7 258V, 8c/8t → ~2-4 arenas), but architect it so the same abstractions
scale to a distributed actor/learner later WITHOUT a rewrite — the `LocalTransport` /
vectorized-actor-pool seam should be the production seam, not throwaway scaffolding.
Deliver a measured number: transitions/s vs arena count + max arenas sustaining ≥19 TPS over
a ≥10-min run. The OPEN decisions: the concurrency model (threads vs asyncio vs
multiprocessing actor pool vs batched send-all/recv-all vector env) and the Paper topology
(N servers vs 1 server hosting N arenas). Redis/Route 2 is out of scope to BUILD now, but the
design must not preclude it being a later config switch. Priority order: (1) build-it-once
seam quality, (2) measured speedup on the laptop, (3) team learnability.

## Perspectives (round 1)
### Critic
1. **Laptop can't host enough arenas to reach M3** — spec §11 says M3 needs ~8 arenas; laptop yields 2-4 (~0.9-1.7M/day), leaving M3 ~a week out. Route 1 on this HW doesn't truly unblock; spec already said "works; slow."
2. **"Build scalable seam once" contradicts the actual `ExperienceTransport` shape** — the ABC ships pre-sampled replay *batches* (Ape-X style), but laptop-optimal is a vectorized env → one in-process replay + learner needing whole episode sequences w/ LSTM hidden snapshots. In-process queue vs ship-sampled-batches are different seams; one of the two goals gives.
3. **Frozen single-connection wire forces N Paper servers — the costliest topology where resources are scarcest.** The cheaper 1-server-N-arenas needs an arena id the frozen wire lacks; N bot-pairs on one single-threaded tick also drags TPS under 19.
4. **N actors → one learner adds off-policy staleness + a fault-isolation hole.** No weight versioning / per-actor ε today; and `step()` aborts on mid-episode reply loss → one arena's hiccup raises BridgeError that kills the shared loop unless new per-arena supervision is written.
5. **AC4 number isn't measurable as defined on a Windows laptop** — package power/die temp not portable on Windows (benchmark.py documents this); thermal throttle is the dominant failure mode and exactly what the harness can't read. Real number is a live human follow-up; live stack not runnable in-session.
6. **`run_benchmark` round-robin makes the headline metric a lie** — send→blocking-recv per arena serializes the 200 ms waits; and per-arena rate = total/N is invariant to overlap, so it looks healthy even when aggregate is flat. True speedup needs real concurrency, and GIL throttles learner + PerceptionFilter packing.

### Feasibility — VERDICT: Conditional
- Must be true: enough RAM for N Paper JVMs (~2-3 GB each flat-world → 2-3 servers fine on 16 GB, tight at 4); cores are binding (each Paper loop single-threaded, 8c/8t no SMT); concurrency model MUST overlap I/O (round-robin serializes; thread-per-arena releases GIL on recv, or batched vector env); thermal throttle on a thin-and-light is real and the ≥10-min gate is the right instrument.
- Hard blockers: NONE that stop the build.
- Solvable: blocking client → thread-per-arena pool behind the existing 4-method `BridgeTransport` seam (no contract change); N-separate-servers is what the code already targets (`TcpBridgeClient(host, port+i)`); single-connection ≠ rewrite because multi-arena = N independent 1:1 connections. Two latent rewrite-forcers to design in NOW: (1) `code_version` rejection on the batch/replay path (replay.py has no version field), (2) replay-buffer thread-safety (no locks today) — cleanest: actors push through transport, one learner thread owns the buffer exclusively. `Trainer.learn()` is already factored out so an async learner loop is additive.

### Thinker (branches)
1. **Async overlap, not more arenas** — batched send-all/recv-all (EnvPool/AsyncVectorEnv style) overlaps the 200 ms tick wait across 2-4 arenas; more samples/wall-second before adding a server.
2. **One Paper server, N spatially-separated arenas** — share one JVM + one tick budget; entity tick cost scales with loaded chunks/entities not logical arenas; needs a plugin + arena-id multiplexing the wire lacks.
3. **Decouple decision interval from tick — sub-stepping / staggered phase** — stagger arena decision phases across the 200 ms window for a near-continuous stream; (spec.) shrink ACTION_REPEAT for collection only.
4. **Actor/learner split by serialization seam (IMPALA/SEED-RL)** — actors ship raw transition sequences over LocalTransport (mp.Queue→ZeroMQ/Redis later) to one learner owning replay+grad. (a) Ape-X: synced policy copies + per-actor ε; (b) **SEED RL: centralize inference** — batch N arenas' obs into one forward pass (nearly free for a tiny net, keeps policy fresh, sidesteps GIL).
5. **Reframe bottleneck: the wire/serialization tax, not arena count** — profile whether the laptop is idle-waiting-on-tick (arenas help) or CPU-bound on JSON+framing+PerceptionFilter (arenas won't scale → need binary/batched wire). Deliverable becomes transitions/s per CPU-second.
6. **Arenas need not be Paper — fast-forward sim tier (speculative)** — schema-identical mock-arena (kinematic PvP, no 20-TPS cap) as a 2nd transport_factory behind the same wire; bulk off-policy samples then fine-tune on Paper.
7. **Replay reuse / sample-efficiency as throughput multiplier (speculative)** — raise replay ratio (more SGD/transition) so otherwise-idle cores do extra gradient steps while arenas wait out 200 ms; attacks wall-clock-to-M3 from the consumption side.
8. **Many learner bots per arena (speculative)** — M policy-sharing learner bots in one arena/tick budget → M streams + free self-play data; needs reward attribution + "nearest-of-many" opponent slot on the wire.

## Live direction
**Topology A — N separate Paper servers (one JVM/world/port per arena + N bridge procs on
port+i), each holding the single TCP connection the frozen wire allows.**

**Concurrency model: #1 thread-per-arena collectors + learner-side replay, WITH the Ape-X
weight-snapshot pattern** (collectors hold a periodically-synced net copy, NOT the live learner
net). `queue.Queue` of whole episodes = the LocalTransport seam (→ Redis later). Promote the
learner to its own thread (branch #6) only if the collect-then-learn lockstep shows in the numbers.

### Concurrency model (round 2 — Thinker)
1. **Thread-per-arena collectors, shared learner (Ape-X-lite)** — N daemon threads each own one
   `MCPvPEnv`+`collect_episode`; `socket.recv` releases the GIL so the 200 ms waits overlap.
   Push whole episodes to one `queue.Queue` (= LocalTransport); main thread = learner owns replay.
   DRQN seq: trivial (thread owns its episode+hidden). Dead arena: thread marks down, survivors continue. **Simplest, best learnability, lowest blast radius.**
2. **Centralized batched inference, thin env threads (SEED-RL)** — N threads do ONLY bridge I/O;
   one inference thread batches pending obs into one `forward((N,1,OBS_DIM))` + stacked hidden,
   scatters actions. GIL-heavy tensor op once/tick, policy always fresh, no per-thread net lock.
   Dead arena: mask its row, zero its hidden column on reconnect. **Elegant build-once seam.**
3. **Batched vector-env send-all/recv-all (EnvPool/AsyncVectorEnv)** — `VecMCPvPEnv.step(actions)`
   sends all N then recvs all N → wall cost ~max(latency) not sum. DRQN: needs N independent
   hidden states + N partial-episode accumulators, flush-on-done (never ragged seq to buffer).
4. **Multiprocessing collectors + shared-memory weights (true Ape-X)** — 1 proc/arena, GIL-free,
   stale-weight sync. Strongest fault isolation + the literal distributed topology, BUT pickle/
   RAM/N-net-copies cost may lose the LOCAL speedup race on 8 cores. The build-for-distributed branch.
5. **asyncio event loop over async sockets (speculative)** — same overlap as #3, no threads, but
   forces an async rewrite of the frozen sync `TcpBridgeClient`/env — highest cost, lowest marginal benefit vs #1.
6. **Two-tier: vector-env collector + decoupled learner thread (IMPALA-shaped)** — #3 + a learner
   in its OWN thread on a bounded episode queue, breaking the collect-then-learn lockstep that
   structurally caps speedup. Most scalable; new failure mode = dead learner thread needs a watchdog.

**Cross-cutting (must resolve before building LocalTransport):** `ExperienceTransport.send(batch)`
is typed to carry a *sampled batch dict* (replay on the ACTOR side, Ape-X). Every branch above keeps
the single replay on the LEARNER side and ships *raw episodes*. Change the transfer unit to
`Episode` (transitions + per-step hidden snapshots) NOW, or in-process and distributed paths disagree on who owns replay.

## Killed / parked
- Topology B (1 server / N arenas + arena-id on wire) — parked: needs a frozen-contract PR
  cascade, serializes all arenas' tick work into one main loop (worse for the ≥19-TPS bar),
  and has no distributed-actor analogue. RAM (its headline argument) is non-binding here.
- Thinker #6 (fast-forward sim tier), #7 (replay-reuse), #8 (many bots/arena) — parked as
  out of Route-1 scope; revisit if laptop throughput proves insufficient.

## Decisions & debate verdicts
- **Topology fork (A vs B): A wins, Lean Yes 7.5/10** (Feasibility 8, Impact 7, Risk 7,
  Effort 8). Decisive fact: machine is **31.6 GB** (not 16) → RAM non-binding; cores+thermal
  bind, and A's per-core TPS isolation beats B's shared-tick-budget. A needs zero contract
  change and is the build-once distributed seam. **Build requirements A inherits from the
  Critic:** (1) per-arena fault supervision — one arena's BridgeError must not kill the pool;
  (2) a launcher to parameterize `run.js`/server trees off the hard-coded 5555/25565.
- **Corrected fact:** dev laptop RAM = ~32 GB (31.6), NOT 16 GB (an earlier note was wrong).
- **Concurrency fork (#1 thread-per-arena vs #2 SEED-RL): #1 wins, Lean Yes 7.75/10**
  (Feasibility 8, Impact 8, Risk 7, Effort 8). Hinge fact: `recv` blocks the full ~200 ms
  server-tick window (bot.js aggregates over ACTION_REPEAT=4 then replies), so GIL-held CPU
  work (PerceptionFilter/reward/no_grad `act`) is single-digit ms ≪ 200 ms at 2-4 arenas →
  thread overlap is real and captures ~the full speedup. **Build requirement (resolves the one
  surviving Critic objection — the torch read-during-write race on the shared `online` net):
  collectors use a periodically-synced weight SNAPSHOT, not the live net** — which also sharpens
  the distributed seam (ship weights down, ship episodes up = Ape-X, the spec's stated Route-2).
  SEED (#2) deferred: its centralized-inference batching only pays off with a GPU inference
  server + fast RPC; at CPU/2-4-arena scale it builds complexity the roadmap would discard.
- **Cross-cutting build requirement (all branches):** change `ExperienceTransport`'s transfer
  unit from a sampled-batch dict to a raw `Episode` (transitions + per-step hidden snapshots)
  BEFORE building `LocalTransport`, keeping the single replay on the learner side.
- **Build requirements inherited from critics (carry into /plan):** (1) per-arena fault
  supervision — one arena's `BridgeError` must not kill the pool; (2) a launcher parameterizing
  `run.js`/server trees off the hard-coded 5555/25565; (3) replace `benchmark.py`'s synchronous
  round-robin so it actually measures aggregate-transitions/s overlap; (4) weight-snapshot sync.
