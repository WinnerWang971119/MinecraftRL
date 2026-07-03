# Multi-Arena Training Throughput (issue #4, Route 1) Plan

**Goal:** Lift training throughput off the single-arena floor (~0.4M transitions/day) by running N arenas concurrently into one learner, so M3/M4 become reachable — and build the seam so single-machine → distributed (Route 2) is a config switch, not a rewrite.

**Approach:** Topology A (N separate Paper servers + N bridge processes on `port+i`, one TCP connection each — the frozen wire unchanged) with concurrency model #1 (Ape-X-lite): N daemon collector threads each own one `MCPvPEnv` and run a reentrant `collect_episode` against a periodically-synced weight **snapshot** (their own net clone, not the live net — avoids a torch read-during-write race). Collectors push whole `Episode`s onto a `LocalTransport` (`queue.Queue` behind the existing `ExperienceTransport` ABC); a separate decoupled learner thread is the **sole** mutator of the single `PrioritizedSequenceReplay`, draining episodes, stepping `learn()`, and republishing weights every K steps. Multi-arena is opt-in behind `--arenas N`; `N=1` preserves today's exact single-env path, M2 win-rate gate, and TC8b recurrence gate. Build order is transport-first / launcher-last so every layer is offline-testable before the only unverifiable-in-session piece.

## Scope

- **In scope:**
  - `LocalTransport` over `queue.Queue` behind the existing `ExperienceTransport` ABC, with the transfer unit corrected to a raw `Episode`.
  - `Episode` value object (transitions + per-step LSTM hidden snapshots + metadata) with `to_dict`/`from_dict` serialization (the future Redis boundary).
  - `SnapshotPolicy` (per-collector net clone + own RNG) and a lock-guarded `WeightStore` publishing cloned `state_dict()`s every K learner steps.
  - Reentrant `collect_episode` returning an `Episode`, with a byte-identical `N=1` wrapper and an explicit per-arena deterministic seed scheme.
  - `LearnerLoop` (decoupled learner thread; sole replay mutator) in `distributed/learner.py`.
  - `Collector` + `ActorPool` supervisor with per-arena drop-out + background restart-with-backoff behind an injectable `ArenaLauncher`.
  - `agent.train --arenas N` wiring (default 1 unchanged); `train_config`/`dist_config` fields.
  - `eval/benchmark.py`: a real-latency `SleepingFakeBridge` + concurrent thread-per-arena driver reporting **aggregate** transitions/s; loose offline overlap assertion.
  - The N-`(Paper server, bridge)` launcher: `bridge/run.js` port-from-argv/env override + a multi-instance launch orchestrator (distinct world dirs/ops/usernames).
  - Offline unit/integration tests for every layer above; RUNBOOK procedure for the human's full 10-min AC4 live run; README update (distributed/ now built out).

- **Out of scope:**
  - Route 2 (Redis/ZeroMQ transport, cross-machine actors) — the seam is built for it, but no networked transport is implemented now.
  - Any change to the frozen wire contract (`observation_spec.py`, `bridge/schema.json`+`messages.py`, `actions.py`, `compute_reward` signature) — no arena id added.
  - SEED-RL centralized inference (rejected by debate; only wins with a GPU inference server).
  - The actual live measured AC4 number (heavy, first-ever live boot, WU-reboot risk) — documented as a human follow-up in the RUNBOOK.
  - Moving fairness/PerceptionFilter out of Python.

## Decisions

- **Topology A over B** — N separate Paper servers, not one shared server. B shares one 20-TPS main-loop tick budget (worse for the ≥19-TPS gate) and needs an arena-id the frozen wire lacks. RAM is non-binding (32 GB measured); cores + thermal bind. (Debate: Lean Yes 7.5/10.)
- **Concurrency #1 (thread-per-arena Ape-X-lite) over #2 (SEED-RL)** — `recv` blocks the full ~200 ms server-tick window (bridge aggregates over `ACTION_REPEAT` then replies), so GIL-held CPU work *should be* negligible at 2-4 arenas and thread overlap *should be* real. (Debate: Lean Yes 7.75/10.) **This is the central HYPOTHESIS the route rests on, not a settled fact** — it is confirmed only by the offline overlap test (loosely) and the human AC4 live run. **Fallback if the live run shows the collector workload is GIL-bound rather than I/O-bound: switch collectors to multiprocessing** (each arena its own process, weights shared via the same `WeightStore` contract over shared memory) — the `Collector`/`LocalTransport`/`Episode` seams are designed so this is a transport/worker swap, not a redesign.
- **Off-policy hidden-state staleness is accepted, bounded by K** — collectors store LSTM `hidden_states` produced by their (possibly stale) snapshot net; the learner later seeds burn-in from those stored states into its *current* nets (R2D2 "stored state", `Trainer._seed_hidden_from_batch`). This representational drift is bounded by `weight_sync_every_k_steps`. Accepted as standard for off-policy recurrent collection; do NOT switch to recompute-from-scratch burn-in (that would change the learner path and risk TC8b). NOTE: TC8b itself is a learner-side n-step-bootstrap-over-`obs_ext` gate and is untouched by collectors.
- **Global counters under N collectors** — ε decay and PER β annealing must index off GLOBAL counters, not a single shared `episode_count`: ε is computed per collector from a global episode counter (atomically incremented across arenas) so the schedule still advances monotonically; β anneals off the learner's `grad_step` (already global on the single learner thread). Each arena may optionally carry its own ε offset for exploration diversity (Ape-X style), but the *schedule progression* is global. Logged `last_epsilon` becomes a per-arena value; log the mean.
- **Fault relaunch granularity** — a dead arena's `BridgeError` first triggers the env's existing idempotent `reset()` reconnect against the *same* bridge; only if that fails does the supervisor relaunch the OS processes. **Relaunching a Paper server is slow (30-60s+: world-gen, plugin load, bot re-op/re-teleport)**, so backoff is on the order of seconds-to-tens-of-seconds and a relaunched arena may be down a long time. `fault_min_live_arenas` abort timing must tolerate this (survivors keep feeding meanwhile).
- **Collectors use a weight snapshot, not the live net** — eliminates the `act()`-vs-`optimizer.step()` torch race; snapshot taken at the learner (single writer), applied at collector **episode boundaries** so the within-episode LSTM trajectory is from one coherent weight set (protects TC8b). This is also the distributed seam (ship weights down, episodes up).
- **`ExperienceTransport` transfer unit = `Episode`, not a sampled batch dict** — the ABC docstring is currently wrong for this design; replay + PER stay centralized on the learner. (Consult Trap 1.)
- **Per-arena deterministic seed offsets** — `cfg.seed + arena_id * SEED_STRIDE + local_ep`, each collector owning its own `torch.Generator`. Avoids both nondeterministic shared-counter interleaving and identical correlated episodes across arenas. (Consult Trap 2.)
- **Decoupled learner thread (IMPALA-shaped) over synchronous drain-then-learn** — collectors produce continuously; the learner steps at its own rate. Exactly one mutator of the non-thread-safe replay (the learner thread).
- **Fault policy: drop-out + background relaunch** — a dead arena's `BridgeError` removes it from the pool; survivors keep feeding; a supervisor relaunches it out-of-band. Learner liveness is independent of arena count.
- **Opt-in `--arenas N`, default 1** — N=1 runs today's exact code path (a thin wrapper) so M2/TC8b cannot regress.
- **Periodic eval runs on ONE designated arena in multi-arena mode** — never fan out. The N=1 eval-borrow works because the single training env is genuinely idle at the eval boundary; under N arenas the chosen arena's `Collector` is a continuously-running daemon, so there is no natural idle boundary. Required protocol: at an eval boundary the learner signals the designated arena's `Collector` to PAUSE after its current episode, the paused collector hands its (idle) transport/connection to the eval routine, eval runs, then the collector resumes. Eval never opens a second connection on any arena. (Consult watch + critic Axis-1.)
- **New code lives in `distributed/`** — `transport.py`, `serialization.py`, `weights.py`, `actor.py`, `learner.py`, `launcher.py`, `dist_config.py`. Issue #4 supersedes the README's "do not build out" note (now past kickoff).
- **Done bar: offline tests + live procedure doc** — no live run in this plan.

## Acceptance Criteria

- [ ] **AC1 — N=1 is a no-op.** `python -m agent.train --arenas 1 ...` runs today's exact single-env path; `tests/test_integration_m2.py` and the eval single-connection test (`peak == 1`) stay green. (TC15)
- [ ] **AC2 — TC8b unchanged.** The DRQN recurrence gate (memory fixture green, ablation fails) still passes. (TC14)
- [ ] **AC3 — Episodes flow N→1 correctly.** N collectors push `Episode`s through `LocalTransport`; the single learner thread is the sole replay mutator; replay grows and gradient steps run. (TC2, TC7, TC8)
- [ ] **AC4 — Snapshot isolation.** Collectors act on a periodically-synced weight snapshot; mutating the learner net after a publish does not change a collector's snapshot. (TC3, TC4)
- [ ] **AC5 — Deterministic, distinct per-arena streams.** With env held identical, two arenas with the same base seed produce DIFFERENT episodes; a single arena+seed reproduces its own stream; N=1 reproduces the pre-refactor action stream. (TC5, TC6)
- [ ] **AC6 — Fault tolerance.** A dead arena drops out without killing the pool; survivors keep feeding; relaunch is attempted via the injectable launcher; the run aborts loudly below `fault_min_live_arenas`; the learner watchdog aborts on a stalled drain. (TC9, TC10, TC11, TC17)
- [ ] **AC7 — Backpressure.** A bounded queue blocks `send` when full and unblocks when the learner drains, without deadlock. (TC18)
- [ ] **AC8 — Measured overlap (offline).** The benchmark reports an AGGREGATE transitions/s that rises with arena count (alongside the unchanged per-arena field), and a real-latency fake shows N=4 wall-time < 2× single-arena. (TC12, TC13)
- [ ] **AC9 — Route-2 seam present.** `ExperienceTransport`'s unit is `Episode`; `Episode.to_dict/from_dict` round-trips; `dist_config.backend` selects `local` now with Redis fields reserved. (TC1)
- [ ] **AC10 — Launcher + docs.** The N-`(server, bridge)` launcher exists with a `--dry-run` plan; the RUNBOOK documents the full 10-min AC4 live procedure; the README `distributed/` note and `--arenas` usage are updated. (non-test: review)

## Data Model

```python
# distributed/serialization.py
@dataclass(frozen=True)
class Episode:
    transitions: list[tuple]          # the 5-tuples collect_episode already builds
    hidden_states: list[np.ndarray]   # parallel list, each (2, num_layers, lstm_hidden)
    arena_id: int
    policy_version: int               # WeightStore version the snapshot came from
    code_version: str                 # train/serve skew guard (from state messages)
    total_reward: float               # logging convenience
    def to_dict(self) -> dict: ...    # numpy -> lists; the Redis boundary
    @classmethod
    def from_dict(cls, d: dict) -> "Episode": ...
```

```python
# agent/train_config.py additions (all defaulted so N=1 config is unchanged)
arenas: int = 1                       # --arenas; 1 = today's single-env path
weight_sync_every_k_steps: int = 50   # collector snapshot refresh cadence (learner grad steps)
fault_relaunch: bool = True           # attempt background relaunch of a dead arena
fault_min_live_arenas: int = 1        # abort the run if live arenas drop below this
collector_queue_max: int = 0          # 0 = unbounded queue.Queue; >0 = bounded backpressure
seed_stride: int = 1_000_000          # per-arena seed offset stride
```

```python
# distributed/dist_config.py
backend: Literal["local"] = "local"   # "local" now; "redis"/"zeromq" later (Route 2)
# arena addresses, weight-sync, sharding fields reserved for Route 2 (documented, unused now)
```

## API Contracts (internal interfaces)

```python
# distributed/transport.py  (ExperienceTransport ABC docstring rewritten: unit = one Episode)
# LocalTransport passes the Episode object BY REFERENCE through queue.Queue — it must NOT call
# to_dict/from_dict (those are dormant until Route 2's networked transport; calling them in-process
# would pay a numpy->list->numpy copy per episode for nothing).
class LocalTransport(ExperienceTransport):
    def __init__(self, maxsize: int = 0): ...      # wraps queue.Queue; maxsize>0 = bounded backpressure
    def send(self, episode: Episode) -> None: ...  # actor side; blocks if bounded+full; raises if closed
    def recv(self) -> Episode: ...                 # learner side; blocks; raises on close (sentinel)
    def close(self) -> None: ...

# distributed/weights.py
class WeightStore:                                 # lock-guarded; learner publishes, collectors read
    def publish(self, state_dict: dict, version: int) -> None: ...   # learner clones+detaches tensors
    def latest(self) -> tuple[dict, int]: ...                        # (cloned state_dict, version)

class SnapshotPolicy:                              # one per collector
    def __init__(self, net_factory, generator_seed: int): ...        # owns its net clone + torch.Generator
    def maybe_refresh(self, store: WeightStore) -> None: ...          # load_state_dict if version advanced
    def act(self, obs, hidden, epsilon) -> tuple[int, Any]: ...       # @no_grad on the clone

# distributed/actor.py
class ArenaLauncher(Protocol):                     # injectable; fake in tests, subprocess shim in prod
    def launch(self, arena_id: int) -> None: ...
    def terminate(self, arena_id: int) -> None: ...

class Collector:                                   # one daemon thread
    # loop: maybe_refresh -> collect_episode(env, policy) -> transport.send(episode)
    # on BridgeError: mark_dead, request relaunch, backoff, reconnect via fresh TcpBridgeClient

class ActorPool:                                   # supervises N Collectors + relaunch-with-backoff
    def start(self) -> None: ...
    def live_count(self) -> int: ...
    def stop(self) -> None: ...

# distributed/learner.py
class LearnerLoop:                                 # the ONLY replay mutator
    def __init__(self, trainer, transport, weight_store, cfg): ...
    def run(self) -> None: ...                     # drain episodes -> add_episode -> learn() -> publish every K
```

```python
# agent/train.py refactor (back-compat preserved)
# The free function takes epsilon + episode_index EXPLICITLY (today's method derives both from shared
# Trainer state: epsilon_for_episode(self.episode_count) and manual_seed(cfg.seed + self.episode_count)).
# For N=1 byte-identical reproduction, the wrapper must pass episode_seed == cfg.seed + episode_index and
# epsilon == epsilon_for_episode(episode_index). Per-arena: episode_seed = cfg.seed + arena_id*seed_stride + local_ep.
def collect_episode(env, policy, *, max_steps, episode_index, epsilon, episode_seed) -> Episode: ...
class Trainer:
    def collect_episode(self, ...):                # N=1 wrapper — NOT a one-liner: it must (a) build a
        ...                                        # one-shot policy from self.online, (b) call the free
                                                   # function, (c) re-add the OLD direct self.replay.add_episode
                                                   # write, and (d) return the OLD (n_transitions, total_reward)
                                                   # tuple — so today's single-env behavior is byte-identical.
```

## Error Handling

- **Mid-episode reply loss in an arena (`BridgeError`):** the Collector aborts that episode (per the resilience contract, `step()` desync is unrecoverable), marks the arena dead, requests a background relaunch, backs off, and reconnects with a fresh `TcpBridgeClient`. The learner and the other collectors are unaffected.
- **Arena relaunch fails repeatedly:** the arena stays dead; if `live_count()` drops below `fault_min_live_arenas`, the `ActorPool` aborts the run loudly.
- **Learner thread dies:** a watchdog detects the stalled drain (queue growing, no grad progress) and aborts the run loudly — never silently collect into a buffer no one drains.
- **`reset()` read-back gate fails / transport drop during reset:** unchanged from today (idempotent reconnect-and-retry inside the env).
- **Bounded queue full (`collector_queue_max > 0`):** collectors block on `send` (natural backpressure) rather than growing memory without limit; default unbounded preserves today's behavior.
- **Weight snapshot during `optimizer.step()`:** safe — the learner publishes a **cloned/detached** `state_dict()`; collectors apply only at episode boundaries.

## Testing Strategy

**Levels:** Unit + Integration (all offline against fakes; no live server). The live AC4 run is a documented human procedure, not an automated test.

| ID  | Test Case | Type | Expected Behavior |
|-----|-----------|------|-------------------|
| TC1 | `Episode.to_dict`/`from_dict` round-trip (incl. numpy hidden states) | Unit | Reconstructed Episode equals original (arrays allclose) |
| TC2 | `LocalTransport` send/recv ordering + `close()` unblocks `recv` | Unit | FIFO episodes returned; `recv` after close raises/sentinels cleanly |
| TC3 | `WeightStore.publish` then `latest()` returns an isolated copy | Unit | Mutating the learner net after publish does NOT change the stored snapshot (clone proven) |
| TC4 | `SnapshotPolicy.maybe_refresh` only reloads when version advances | Unit | No reload at same version; reload on bump; `act` runs under no_grad |
| TC5 | `collect_episode` N=1 wrapper reproduces the pre-refactor seed/action stream | Integration | Action sequence + stored episode identical to the old `Trainer.collect_episode` for a fixed seed |
| TC6 | Per-arena seed offsets give distinct, reproducible streams | Unit | With the env held IDENTICAL across both arenas (same scripted fake) so the per-arena seed offset is the only varying input: two arenas with the same base seed produce DIFFERENT episodes; same arena+seed reproduces its own |
| TC7 | `LearnerLoop` drains episodes → `add_episode` + `learn()` + publishes every K | Integration | Replay grows; grad steps run; `WeightStore` version bumps every K steps |
| TC8 | `LearnerLoop` is the sole replay mutator | Unit | Collectors hold no replay ref; only the learner thread calls `add_episode`/`sample` |
| TC9 | `ActorPool`: one arena raises `BridgeError`, survivors keep producing | Integration | Dead arena drops; remaining collectors keep sending; `live_count` decremented |
| TC10 | `ActorPool` relaunch via fake `ArenaLauncher` | Integration | `launch(arena_id)` invoked with backoff; collector reconnects and resumes on success |
| TC11 | `ActorPool` aborts when live arenas < `fault_min_live_arenas` | Unit | Run stops loudly with a clear error |
| TC12 | Benchmark overlap with `SleepingFakeBridge` (real `time.sleep`) + a real `act()` forward in the collector loop | Integration | Wall-time for N=4 arenas < 2× single-arena (loose bound). The real `act()` forward makes the test exercise GIL contention, not just sleep overlap. HARNESS-LEVEL evidence only — the real overlap verdict is the human AC4 run. |
| TC13 | Benchmark reports AGGREGATE transitions/s ALONGSIDE the existing per-arena field | Unit | New `transitions_per_s_aggregate` scales with arena count; the existing `transitions_per_s_per_arena` field is preserved (live-AC4/log consumers unbroken) |
| TC14 | TC8b recurrence gate still passes (memory fixture green, ablation fails) | Integration | Unchanged from today |
| TC15 | M2 single-env path unchanged: `--arenas 1` runs the wrapper | Integration | `test_integration_m2.py` + eval-borrow single-connection test (`peak == 1`) stay green |
| TC16 | Multi-arena eval runs on ONE designated arena via the pause/handoff protocol | Unit | Periodic eval pauses the designated collector, reuses its connection, opens NO connection on other arenas |
| TC17 | Learner watchdog fires on a stalled drain | Unit | With the learner wedged / not draining and the queue growing, the watchdog aborts the run loudly |
| TC18 | Bounded-queue backpressure (`collector_queue_max > 0`) | Unit | A full bounded queue blocks `send`; it unblocks when the learner drains; no deadlock with a slow learner |

**Test data:** reuse `ScriptedBridge`/fake-transport fixtures from `tests/test_mc_pvp_env.py` and the benchmark's `transport_factory`/`FakeBridge`; add a `SleepingFakeBridge` (real `time.sleep`). A fake `ArenaLauncher` records `launch`/`terminate` calls.
**Run command:** `pytest` (Python) · `cd bridge && npm test` (Node run.js port override).

## Tasks

| ID | Task | Blocked By | Risk | Files | Description |
|----|------|------------|------|-------|-------------|
| T1 | `Episode` + serialization | — | med | `distributed/serialization.py` | Replaces the empty `DEFERRED` stub. Define the `Episode` frozen dataclass (fields per Data Model) + `to_dict`/`from_dict` with numpy round-trip. AC9. TC1. |
| T2 | `LocalTransport` + ABC fix | T1 | med | `distributed/transport.py` | Rewrite the `ExperienceTransport` ABC docstring (currently "unit = sampled batch dict from sample()"): transfer unit is one `Episode`. Change `send(batch: dict)` → `send(episode: Episode)`. Implement `LocalTransport` over `queue.Queue` passing the `Episode` BY REFERENCE (no to_dict/from_dict in-process), `maxsize` for backpressure, a sentinel that unblocks `recv` on `close()`. AC9. TC2. |
| T3 | `WeightStore` + `SnapshotPolicy` | — | med | `distributed/weights.py` | Lock-guarded `WeightStore.publish` (learner clones+detaches `state_dict`) / `latest()`. `SnapshotPolicy` owns a net clone (via a net factory) + its own `torch.Generator`, `maybe_refresh` reloads only on version bump, `act` is `@no_grad`. TC3, TC4. |
| T4 | `train_config`/`dist_config` fields | — | low | `agent/train_config.py`, `distributed/dist_config.py` | Add the defaulted config fields (Data Model). `dist_config.backend="local"` with Route-2 fields reserved/documented. Defaults must leave N=1 config identical. |
| T5 | Reentrant `collect_episode` + N=1 wrapper + seed/ε scheme | T1 | high | `agent/train.py` | Extract `collect_episode(env, policy, *, max_steps, episode_index, epsilon, episode_seed) -> Episode` with NO writes to shared `Trainer` state (ε and seed passed in, not derived from `self.episode_count`). Keep `Trainer.collect_episode` as the N=1 wrapper — NOT a one-liner: it re-adds the old direct `self.replay.add_episode` write and returns the old `(n_transitions, total_reward)` tuple. Per-arena seed `cfg.seed + arena_id*seed_stride + local_ep`; ε schedule advances off a GLOBAL atomic episode counter. Satisfies AC1, AC5. TC5, TC6. |
| T6 | `LearnerLoop` | T2, T3 | high | `distributed/learner.py` | Decoupled learner thread (replaces the empty `DEFERRED` stub): drain up to M episodes from transport → `trainer.replay.add_episode` → `trainer.learn()` loop → `WeightStore.publish` every K grad steps. Sole mutator of replay. Watchdog: abort loudly if the queue grows while no grad progress. β anneals off the global `grad_step`. AC3, AC6. TC7, TC8, TC17. |
| T7 | `Collector` + `ActorPool` + `ArenaLauncher` | T2, T3, T5 | high | `distributed/actor.py` | Replaces the empty `DEFERRED` stub. Daemon `Collector` (maybe_refresh → `collect_episode` → `transport.send`; on `BridgeError` first try env idempotent `reset()` reconnect, else mark dead + request relaunch + seconds-scale backoff + reconnect). `ActorPool` supervises N collectors via injectable `ArenaLauncher`, aborts below `fault_min_live_arenas`. AC6. TC9, TC10, TC11. |
| T8 | Wire `train.py --arenas N` + main() | T4, T5, T6, T7 | high | `agent/train.py` | Add `--arenas` (default 1). N=1 → today's exact path. N>1 → build `Trainer` + `WeightStore` + `LocalTransport` + `ActorPool` + `LearnerLoop`, start threads, join on budget/gate. Implement the eval pause/handoff protocol on ONE designated arena. AC1, AC2. TC15, TC16. (Same file as T5 → sequenced after it.) |
| T9 | Benchmark concurrent driver + `SleepingFakeBridge` | — | med | `eval/benchmark.py` | Replace the synchronous round-robin with a thread-per-arena driver around the EXISTING raw `send(step)`/`recv()` loop (reuse the threading PATTERN, NOT the `Collector` class — the bench needs raw `StateMsg`/per-step latency/tick-delta TPS/damage-boundary metrics the Collector doesn't expose). ADD a `transitions_per_s_aggregate` field ALONGSIDE the existing `transitions_per_s_per_arena` (do not break it). Add `SleepingFakeBridge` (real `time.sleep(latency)` + a real `act()` forward) to exercise GIL contention. AC8. TC12, TC13. |
| T10 | `run.js` argv/env config (ports + usernames) | — | med | `bridge/run.js` | Today `run.js` constructs `new ArenaBots()` with NO config — there is zero CLI→config path. Read argv/env and forward `{port, bridgePort, learnerUsername, dummyUsername}` into `ArenaBots(config)` (the `DEFAULT_BOT_CONFIG` fields already exist in `bot.js`). Defaults = current 5555/25565/`learner_bot`/`dummy_bot`. Node-test the override; live wiring is human-verified. |
| T11 | Multi-instance launcher + N server roots | T7, T10 | high | `distributed/launcher.py`, `server/setup/start-arenas.ps1`, `server/setup/setup.ps1` | Real `ArenaLauncher` subprocess shim + a PowerShell orchestrator: materialize N Paper server roots (distinct world dir, `server.properties` port `25565+i`, `ops.json` listing that arena's distinct bot usernames) and start N servers + N bridges (`5555+i`, usernames via T10). Unverifiable in-session — ship a `--dry-run` that prints the launch plan. |
| T12 | Tests: transport/serialization/weights/backpressure | T2, T3 | low | `tests/test_local_transport.py`, `tests/test_snapshot_policy.py` | TC1-TC4, TC18 (bounded-queue backpressure). AC4, AC7, AC9. |
| T13 | Tests: collect_episode equivalence + seeds | T5 | low | `tests/test_collect_episode_multiarena.py` | TC5, TC6 — incl. the byte-identical N=1 assertion. MUST be re-run after T8 (T8 re-touches the same `train.py` path). AC5. |
| T14 | Tests: learner loop + actor pool faults | T6, T7 | low | `tests/test_learner_loop.py`, `tests/test_actor_pool.py` | TC7-TC11, TC17 (watchdog) — fake transport + fake `ArenaLauncher`. AC3, AC6. |
| T15 | Tests: benchmark overlap + M2/TC8b regression | T8, T9 | low | `tests/test_benchmark_overlap.py` | TC12, TC13, TC14, TC15, TC16 — loose overlap bound; confirm `--arenas 1` and TC8b stay green. AC1, AC2, AC8. |
| T16 | Docs: RUNBOOK live procedure + README | T8, T11 | low | `RUNBOOK.md`, `README.md` | Document the full 10-min AC4 multi-arena live run (launch order, sweep `--arenas`, WU-reboot + first-boot caveats). Update the README `distributed/` "do not build out" note + add `--arenas` usage, and reconcile that the pre-existing benchmark `--arenas` (on `eval.benchmark`) and the NEW training `--arenas` (on `agent.train`) are two different flags on two entrypoints. README + commit/PR prose must follow the user's voice rules (no AI tells, no em-dashes, none of the banned words). AC10. |

## Notes for Implementer

- **The N=1 path is sacred.** `--arenas 1` must run the existing `Trainer.collect_episode` wrapper with identical seeds and the identical eval-borrow single-connection mechanism. TC15 (`test_integration_m2.py` + the `peak == 1` connection test) and TC14 (TC8b) are the regression tripwires — run them after T5 and again after T8.
- **Exactly one replay mutator.** Only `LearnerLoop` may call `add_episode`/`sample_sequences`/`update_priorities`. `PrioritizedSequenceReplay` is pure-NumPy with no locks — a stray collector reference to it corrupts the sum-tree silently. Enforce in review.
- **Clone on publish, not on read.** `state_dict()` returns views; `WeightStore.publish` must `.detach().clone()` each tensor (CPU) or collectors read memory the learner is mutating. Apply snapshots at collector **episode boundaries** only (coherent within-episode LSTM trajectory → TC8b).
- **Seeds:** per-arena offset stride must exceed the max episodes-per-arena so streams never collide; each collector owns its own `torch.Generator` (no shared generator, no lock on the hot path).
- **Benchmark overlap is a loose bound.** A real-`time.sleep` fake proves GIL handoff / overlap exists; assert `N=4 < 2× single` (generous margin), not an exact speedup — the real number is the human AC4 run. Log what the offline bench cannot measure (thermal/package power on Windows is already documented as a gap).
- **Launcher is the only unverifiable piece** — keep `ArenaLauncher` injectable so all supervisor/fault logic is tested against a fake offline; the subprocess shim (T11) is the thin untested layer. Ship it with `--dry-run`.
- **Do not touch the frozen wire contract.** No arena id on the wire; multi-arena is N independent connections. Fairness/PerceptionFilter stays in Python.
- **Empty stubs being replaced:** `distributed/serialization.py` (T1), `transport.py` (T2), `learner.py` (T6), `actor.py` (T7), `weights.py` (new), `launcher.py` (new), `dist_config.py` (T4) are one-line `# DEFERRED(distributed)` stubs today — replace wholesale and remove that comment. `bridge/arena.js` stays a deferred stub (it was for topology B multiplexing, which we rejected).
- **Two `--arenas` flags exist.** `eval.benchmark --arenas` (pre-existing, the AC4 measurement) and the NEW `agent.train --arenas` (training). They are separate; don't conflate them in code or docs.
- **Rollback:** every change is behind `--arenas` (default 1) and new `distributed/` modules; reverting to single-arena is dropping the flag. The launcher/server-tree changes are additive scripts.
