# Minecraft PvP RL — Kickoff Plan (Contract → Vertical Slice → M1 → M2)

**Goal:** Get an 8-person team from zero code to a *measured*, de-risked foundation: a frozen interface contract, a real end-to-end Mineflayer→bridge→Paper vertical slice (M1), and a Dueling-DRQN agent that beats a stationary dummy ≥95% (M2) — answering both "does the plumbing work?" and "does our RL stack learn?" on the real stack, no mock.

**Approach:** Freeze four pure-data contract artifacts on day one so all 8 people parallelize without blocking. Claude scaffolds the repo + contracts + the tracer-bullet vertical slice (the foundation); the human workstreams build their real components offline against tiny hand-authored fixtures and integrate on the live bridge as it lands. One paired track owns the highest-risk bridge and must exit with a *measured number*, not a passing test. Transport is raw TCP + JSON-lines behind a swappable interface; episode reset uses Mineflayer server commands + a read-back verification gate (the custom Paper plugin is deferred per the `/debate` verdict). Single arena on the dev laptop; multi-arena, distributed, M3/M4, self-play, and the study are out of scope.

## Scope

- **In scope:**
  - Repo setup: `git init`, `.gitignore`, initial commit, private GitHub repo, push (the contract-PR baseline).
  - A day-1 version-compatibility check that *precedes* the version pin.
  - Full §13 repo skeleton (all directories with stubs + per-dir README).
  - The four frozen contract artifacts: `env/observation_spec.py`, the `bridge` JSON schema, `agent/actions.py` (macro enum), `env/reward.py` signature.
  - Day-1 pins (written only after the compat check): Minecraft **1.21.1**, `ACTION_REPEAT = 4` ticks, `MAX_EPISODE_STEPS`, `code_version` stamping, seed utilities.
  - Paper server: flat arena, fixed gear/spawns, offline-mode, minimal view/sim distance, no mob spawning; opped bot accounts.
  - Node↔Python bridge (raw TCP + JSON-lines): connect two bots (learner + idle dummy), event aggregation over `ACTION_REPEAT`, macro execution, `reset` RPC with read-back gate.
  - Gym-style env (`env/mc_pvp_env.py`) with `reset()/step()/done`, reward, observation.
  - The M1 tracer bullet: random policy end-to-end vs the idle dummy, no crashes.
  - The throughput/latency benchmark (the bridge spike's measured exit number).
  - PerceptionFilter (FOV + raycast LoS + memory) + the leak-detection battery.
  - Dueling-DRQN net + prioritized **sequence** replay + n-step Double-DQN training loop (DRQN from the start, per user override).
  - The M2 gate: greedy agent ≥95% win over 100 eval episodes vs the stationary dummy, with per-reward-component logging.
- **Out of scope (explicitly deferred):**
  - N parallel arenas, distributed actor/learner, Redis transport (`ExperienceTransport` interface is stubbed but only `LocalTransport` is built).
  - M3 (scripted bot), M4 (self-play, PFSP, Elo), the PPO comparison arm.
  - The study/booth app, survey, deploy/exhibition (booth skeleton dir only).
  - The custom Paper reset plugin (adopt later per the `/debate` verdict).
  - Fairness curriculum, opponent-health scoreboard exposure.
- **Note on DRQN ahead of need:** Recurrence is built at M2 by user override, but the M2 opponent is a stationary, always-visible dummy — a fully-observed, degenerate MDP with no expiring information. M2 therefore *cannot* validate the LSTM (a feed-forward encoder alone could pass it). LSTM correctness is gated separately by a memory-dependent unit fixture (TC8b), not by M2.

## Decisions

- **Contract-first, frozen + versioned** — the four artifacts merge before any track writes implementation; changing one requires a PR everyone sees. Counters Brooks's-law drift on an 8-beginner team and the "can't parallelize against a nonexistent contract" risk.
- **No mock environment** — components unit-test against tiny hand-authored fixtures, not a parallel sim. Honors the spec's one-interface / no-second-codebase / no-sim-to-real-gap principle (§0.1, §1.2-1.3).
- **Reset = Mineflayer server commands + read-back gate**, behind a clean `reset` RPC — per `/debate` (Lean No, 4.5/10, on the plugin for kickoff). Bot accounts are opped (ops.json by username, offline-mode) so `/tp`, `/effect clear`, regear commands are authorized; the read-back gate is required precisely because command execution is async and not acknowledged. Adopt the Bukkit plugin at N-arena scaling or if the gate proves flaky/slow.
- **Transport = raw TCP + newline-delimited JSON** behind the transport seam — zero deps, eyeball-debuggable, swappable to ZeroMQ/Redis later. Framing must buffer partial reads across TCP packet boundaries.
- **Pin Minecraft 1.21.1** — but only after a day-1 compat check (Tv) confirms `minecraft-data`, the Paper build, **and** `mineflayer-pvp` + `mineflayer-pathfinder` all support it; if any plugin lags, move the pin to the highest version all four support. Keep **1.9+ attack-cooldown** combat (matches the attack-cooldown observation feature).
- **Dueling-DRQN from the start (user override)** — implies sequence replay + burn-in immediately, not frame-stacking. NOTE: this deviates from the source-spec incremental DQN ladder (where DRQN normally arrives at M3) and builds recurrence ahead of need. Because the M2 dummy is stationary and always visible, M2 CANNOT validate that the LSTM works — a green M2 only proves the MLP encoder + value head learn, and a silently-broken LSTM would still pass. Recurrence correctness must be gated by unit test (TC8b), not by M2 acceptance.
- **Stationary dummy = an idle Mineflayer bot** at a fixed spawn, made knockback-immune (knockback-resistance attribute ≈ 1.0, or teleported back each reset) and void/fall-immune, so "stationary" actually holds and M2 stays a fully-observed MDP that doesn't accidentally exercise the gating/memory path.
- **Single arena on the dev laptop** (Intel Core Ultra 7 258V, 8c/8t, ~32 GB, Lunar Lake mobile) — the benchmark is a *lower-bound smoke figure* for M1 de-risking, NOT fleet capacity planning (which targets members' PCs + a cloud VM per source §11).
- **Bridge spike exit = a measured number**, not a green check — transitions/s, p99 round-trip at 200 ms, damage-event boundary correctness, max stable arenas at ≥19 TPS sustained.
- **Claude builds the foundation** (T0-T11 + Tv); human workstreams own T12-T20.

## Acceptance Criteria

- [ ] **AC1 (contract frozen):** `observation_spec.py`, the bridge JSON schema, `actions.py`, and the `reward.py` signature are merged to `main`; each is importable and round-trips a validation test; altering any requires a PR.
- [ ] **AC2 (repo):** Full §13 skeleton exists; a documented setup (Python 3.11+/PyTorch, the verified Node LTS, Paper) runs from a fresh clone; private GitHub repo pushed.
- [ ] **AC3 (M1 — plumbing):** A random policy completes ≥100 full episodes end-to-end vs the idle dummy through the real Paper server + real bridge with zero crashes, and combined process RSS (Node + Python + JVM) grows < ~200 MB across the run (sampled every 10 episodes).
- [ ] **AC4 (M1 — the number):** Benchmark reports transitions/s/arena, p99 Node→Python round-trip at the 200 ms decision interval, a verified damage-event count at step boundaries (no drop/double-count vs a ground-truth scripted exchange), and the max arenas the laptop sustains at ≥19 TPS over a ≥10-minute run.
- [ ] **AC5 (fairness integrity):** PerceptionFilter unit tests pass, including a leak battery asserting `in_range`/`in_crosshair` (and every derived feature) never reveal opponent position when `visible == false`.
- [ ] **AC6 (M2 — learning):** The greedy (ε=0) Dueling-DRQN agent wins ≥95% of 100 eval episodes vs the stationary dummy; aim-bonus reward accrued while `visible == false` is exactly 0 over the eval run, and the mean episode does not hit the timeout cap (guards spin-farming / run-away). AC6 validates the end-to-end RL stack but does NOT validate recurrence — LSTM correctness is gated separately by TC8b.
- [ ] **AC7 (reset correctness):** Episode start is gated on a read-back check (health==max, position==spawn within ε, inventory==gear template, no active effects); given a fixed seed the post-reset readback matches the template within ε across runs (bit-identical is not required — Minecraft has inherent physics/timing stochasticity).

## Data Model

### `env/observation_spec.py` — the single source of truth (imported by net, env, filter, every future actor)

A fixed-length float vector (~30-40 dims) with a frozen index map. Angles encoded as `(sin, cos)`; positions in the agent's **local frame**; normalized to ≈ `[-1, 1]`.

```python
# Indicative layout — exact indices frozen in the file, asserted by a test.
SELF = {
  "health": 1,            # normalized /max
  "yaw_sin": 1, "yaw_cos": 1, "pitch_sin": 1, "pitch_cos": 1,
  "vel_local": 3,         # vx, vy, vz in local frame
  "on_ground": 1,
  "held_item": 1,         # small categorical -> normalized id or one-hot (frozen choice)
  "attack_cooldown": 1,   # 0..1 progress (1.9+ combat). Computed by the BRIDGE from
                          # (current_tick - last_swing_tick) / weapon_attack_speed_ticks.
                          # Bridge owns cooldown; ATTACK uses raw bot.attack so this is observable.
}
OPPONENT = {              # GATED: real values only when visible & LoS clear, else MEMORY then absent
  "pos_local": 3, "facing_sin_cos": 2, "vel_local": 3,
  "visible": 1,           # 1 if in FOV cone + clear raycast this step
  "time_since_seen": 1,   # seconds, normalized; large/absent when never seen
}
DERIVED = {               # computed AFTER gating, from gated values only
  "in_range": 1, "in_crosshair": 1,
}
# Opponent health: NOT in the observation (NONE by default). Reward may read it (privileged).
OBS_DIM = <sum>           # frozen constant; net input dim asserts against it
```

Provides: `OBS_DIM`, an index enum, `build_observation(...) -> np.ndarray`, `validate(vec)`, normalization constants.

### Bridge wire schema (raw state — Node side; fairness applied later in Python)

```
Python -> Node:
  {"type":"reset", "episode": <int>, "seed": <int>}
  {"type":"step",  "action": <int 0..7>}
  {"type":"close"}

Node -> Python:
  {"type":"state",
   "self":     {"pos":[x,y,z], "yaw":f, "pitch":f, "velocity":[x,y,z],
                "on_ground":bool, "health":f, "held_item":str, "attack_cooldown":f},
   "opponent": {"pos":[x,y,z], "yaw":f, "pitch":f, "velocity":[x,y,z], "health":f},  # RAW true health,
                # carried for FUTURE stages only; never reaches the observation (gated by PerceptionFilter).
   "events":   {"damage_dealt":f, "damage_taken":f, "i_died":bool, "opponent_died":bool},
   "arena":    {"wall_distances":[...]},
   "tick": <int>, "code_version": <str>}
  {"type":"reset_ack", "ok":bool, "readback":{...}}   # after read-back gate
```

### `agent/actions.py`

```python
class Macro(IntEnum):
    IDLE=0; APPROACH=1; RETREAT=2; STRAFE_L=3; STRAFE_R=4; ATTACK=5; JUMP=6; TURN_TO_LAST_SEEN=7
N_ACTIONS = 8
```

### `env/reward.py` signature (§7)

```python
def compute_reward(events: Events, gated_obs: Obs, prev_obs: Obs, terminal: TermInfo,
                   cfg: RewardConfig) -> float:
    # r = c_dmg_out*dealt - c_dmg_in*taken - c_step
    #     + c_aim*1[visible and in_crosshair] + R_terminal
    # potential-based shaping for positional terms; coeffs in RewardConfig (TUNE).
```

## API Contracts

```
Node TCP server (localhost:<port>), newline-delimited JSON (buffer partial reads across packets),
one connection per arena. Bot accounts are opped on the offline-mode server.
  reset:  Python sends {"type":"reset",...}; Node teleports both bots (/tp), /effect clear, regear,
          then READ-BACK gate (poll bot.health/position/inventory until they match the template
          or a timeout); replies {"type":"reset_ack","ok":true,"readback":{...}}.
  step:   Python sends {"type":"step","action":a}; Node runs the macro for ACTION_REPEAT ticks,
          aggregates events over that window, replies one {"type":"state",...}.
  close:  graceful disconnect.
Errors: reset_ack.ok=false on read-back timeout; state messages carry code_version (learner
        rejects mismatch in the distributed future — kickoff just logs it).
```

## Error Handling

- **Bridge disconnect / Node crash:** Python `step()` raises `BridgeError`; the env runner logs, attempts one reconnect, and aborts the run with a clear message if it fails (no silent hang). Kickoff does not need auto-recovery — it needs loud failure.
- **Bots not opped / command rejected:** reset fails loudly with the rejected command — bot usernames must be in `ops.json` before T7a runs.
- **Reset read-back timeout:** `reset_ack.ok=false` → env retries reset once, then raises; never starts an episode from an unverified state (protects AC7 and the MDP).
- **Damage-event boundary drift:** the aggregation window must sum damage events landing across tick boundaries exactly once; covered by an adversarial unit test (TC7) AND a live integration test (TC7b) because this silently corrupts the reward.
- **Command spam kick:** reset issues a small, fixed command set per ~80 s episode (well under spam thresholds); if Paper kicks, space commands by one tick — do not remove the read-back gate.
- **Server below 20 TPS under load:** the benchmark records actual sustained TPS; if an arena dips below ~19 TPS the 200 ms interval assumption breaks — flag it in the number, reduce arenas.
- **Q-value divergence at M2:** lower lr, confirm grad-norm clip, slow target (τ↓); plot reward components separately to catch hacking before blaming the learner.

## Testing Strategy

**Levels:** Unit (Python components, offline), Integration (bridge smoke + M1 slice + live damage exchange), Benchmark (the number), Eval (M2 gate).

| ID  | Test Case | Type | Expected Behavior |
|-----|-----------|------|-------------------|
| TC1 | `observation_spec` build+validate round-trips; `OBS_DIM` matches the index map | Unit | Vector length == `OBS_DIM`; out-of-range values rejected |
| TC2 | PerceptionFilter gates opponent features outside FOV/LoS | Unit | `visible=0`, opponent pos fields zeroed, `time_since_seen` grows |
| TC3 | **Leak battery:** derived features when `visible=false` | Unit | `in_range`/`in_crosshair` and all derived features reveal NO position info (AC5) |
| TC4 | Memory expiry after `MEMORY_TTL` | Unit | last-seen value held then dropped to absent at TTL |
| TC5 | Reward: damage dealt/taken, step penalty, gated aim bonus | Unit | matches formula on hand-authored event dicts; aim bonus only when visible+in_crosshair |
| TC6 | Reward anti-hacking: spinning while opponent unseen | Unit | no aim reward accrues (guards spin-to-farm) |
| TC7 | **Damage-event aggregation across step boundary (fixtures)** | Unit | a known 3-hit exchange straddling a window counts exactly 3, no drop/double |
| TC7b | **Damage-event aggregation on the LIVE bridge** | Integration | drive a scripted N-hit exchange through the real server; summed events == N (AC4) |
| TC8 | Dueling-DRQN forward + backward on one fixture sequence | Unit | output shape == `N_ACTIONS`; loss finite; grads non-NaN; loss computed ONLY on post-burn-in `L−B` steps (burn-in step gradients detached/zero) |
| TC8b | **Memory-dependent recurrence gate (runs at M2)** | Unit | A fixed synthetic episode where the correct greedy action at step t depends on a value seen at t−N and now gated out: (a) the trained DRQN with burn-in picks the memory-dependent action above chance; (b) an ablation that zeroes/detaches the LSTM hidden state FAILS the same fixture. Isolates LSTM correctness from the MLP encoder. |
| TC9 | Prioritized sequence replay: store/sample/priority update | Unit | sequences length L sampled; IS weights normalized; priorities update |
| TC10 | Bridge smoke: reset→step→state round-trip on a live server | Integration | one `state` per `step`; `reset_ack.ok=true`; read-back matches template |
| TC11 | M1 slice: random policy, ≥100 episodes vs idle dummy | Integration | no crash; RSS growth < ~200 MB sampled per 10 eps; episodes terminate on death/timeout (AC3) |
| TC12 | Throughput/latency benchmark (sustained ≥10 min) | Benchmark | reports transitions/s, p99 round-trip @200 ms, max arenas ≥19 TPS, with CPU package power/thermal recorded (AC4) |
| TC13 | M2 eval: greedy agent vs stationary dummy, 100 eps | Eval | win rate ≥95%; aim-bonus-while-invisible == 0; mean episode < timeout cap (AC6) |
| TC14 | Reset determinism under fixed seed | Integration | post-reset readback matches template within ε across runs (AC7) |

**Test data:** hand-authored observation vectors and event dicts (fixtures); a scripted ground-truth hit exchange for TC7/TC7b; a memory-dependent synthetic episode for TC8b; a fixed seed list for TC14.
**Run command:** `pytest` (Python); `npm test` (Node bridge smoke); a `make bench` / script for the benchmark.

## Tasks

| ID | Task | Owner | Blocked By | Risk | Files | Description |
|----|------|-------|------------|------|-------|-------------|
| T0 | Repo + git + GitHub | Claude | — | low | repo root, `.gitignore`, `README.md` | `git init`; `.gitignore` (Python, Node, Java, Minecraft worlds/logs); initial commit; create private GitHub repo via `gh repo create`; push `main`. Satisfies AC2. |
| T1 | §13 skeleton | Claude | T0 | low | `bridge/ env/ agent/ opponents/ distributed/ eval/ server/ deploy/ study/` + per-dir `README.md` + `__init__.py`/stub | Create every directory from spec §13 with stub files and a one-line README per dir stating its owner workstream. Satisfies AC2. |
| Tv | Day-1 compat check (precedes the pin) | Claude | T1 | high | `server/compat_check.md` (+ throwaway probe) | Spin up Paper 1.21.1 + connect Mineflayer on the candidate Node LTS + load `mineflayer-pvp` + `mineflayer-pathfinder` + confirm `minecraft-data` has 1.21.1. If any of the four lags 1.21.1, pick the highest version all four support. Output = the confirmed MC + Node versions, consumed by T6/T8. |
| T2 | observation_spec | Claude | T1 | high | `env/observation_spec.py`, `tests/test_observation_spec.py` | Frozen vector layout + index enum + `OBS_DIM` + `build_observation`/`validate` + normalization consts (Data Model); `attack_cooldown` documented as bridge-computed. Satisfies AC1; TC1. |
| T3 | Bridge schema | Claude | T1 | high | `bridge/schema.md`, `bridge/schema.json`, `bridge/messages.py` | Author the wire schema (Data Model/API Contracts); Python dataclasses + a JSON validator; Node-side schema doc; note `opponent.health` is future-only and never reaches obs. Satisfies AC1. |
| T4 | Action enum | Claude | T1 | low | `agent/actions.py`, `tests/test_actions.py` | `Macro` IntEnum + `N_ACTIONS` + semantic mapping table. Satisfies AC1. |
| T5 | Reward signature + config | Claude | T2 | med | `env/reward.py`, `agent/reward_config.py` | `compute_reward` signature + `RewardConfig` coeffs (§7 starts) + initial implementation. Owns `reward_config.py`. Satisfies AC1 (signature). |
| T6 | Pins + determinism | Claude | Tv | med | `agent/contract_config.py`, `agent/seeding.py`, `README.md` | Write the verified MC version + Node LTS, `ACTION_REPEAT=4`, `MAX_EPISODE_STEPS`; `code_version` (git SHA + config hash); `seed_everything` (incl. `action_space.seed`). Sole owner of `contract_config.py`. |
| T7a | Bridge transport + reset | Claude | T3, T6 | high | `bridge/bot.js`, `bridge/transport.js` | Mineflayer connect two opped bots (learner + idle dummy); TCP JSON-lines server per schema (buffer partial reads); `reset` RPC = server commands + **read-back gate**. Satisfies AC7; TC10, TC14. |
| T7b | Macro exec + event aggregation | Claude | T7a, T4 | high | `bridge/actions.js`, `bridge/bot.js` | ATTACK maps to low-level `bot.attack(entity)` (single swing, manually cooldown-gated by the bridge); movement macros (approach/retreat/strafe-L/strafe-R/turn-to-last-seen) map to time-bounded `bot.setControlState` held for ACTION_REPEAT ticks — NOT async pathfinder/pvp goals. Do NOT use `bot.pvp.attack` for ATTACK (it owns cooldown, drives pathfinder pursuit, swings across boundaries). Aggregate `damage_dealt/taken/i_died/opponent_died` over ACTION_REPEAT ticks, exactly once at boundaries. TC7, TC7b. |
| T8 | Paper server + arena | Claude | T6 | high | `server/README.md`, `server/setup.*`, `server/arena.*`, `server/ops.json` | Pin Paper to the verified version; flat world, offline-mode, fixed gear/spawns, minimal view/sim distance, no mobs, anti-spam config; op the bot usernames. Plugin note: `mineflayer-pvp` is OPTIONAL (cooldown reference only, NOT used to drive ATTACK); `mineflayer-pathfinder` demoted from day-1 blocker (movement uses `setControlState`). |
| T9 | Gym env | Claude | T2, T3, T5 | high | `env/mc_pvp_env.py` | `reset()/step(action)/done` over the bridge; applies PerceptionFilter (stub at first), computes reward, returns `observation_spec` vector; reset retries once then raises. |
| T10 | M1 tracer bullet | Claude | T9, T7b, T8 | med | `eval/run_random.py`, `agent/random_policy.py` | Random policy vs idle dummy end-to-end; logs episodes + win-rate; a toy in-memory replay buffer lives in `eval/run_random.py` + a no-op grad step to exercise the full path. Satisfies AC3; TC11. |
| T11 | Throughput/latency benchmark | Claude | T10 | high | `eval/benchmark.py`, `eval/logging.py` | Sole writer of `eval/logging.py`. Measure transitions/s/arena, p99 Node→Python round-trip @200 ms, damage-event boundary correctness, max arenas at ≥19 TPS sustained ≥10 min with CPU power/thermal recorded; log via W&B/TensorBoard. Satisfies AC4; TC12. |
| T12 | PerceptionFilter | PerceptionFilter track | T2 | high | `env/perception_filter.py`, `tests/test_perception_filter.py` | FOV cone (~70°) + raycast LoS + memory expiry (`MEMORY_TTL`); gating + `visible`/`time_since_seen`. Build vs synthetic geometry. TC2, TC4. |
| T13 | Leak-detection battery | PerceptionFilter track | T12 | high | `tests/test_perception_leak.py` | Assert no derived feature (`in_range`/`in_crosshair`/any) reveals position when `visible=false`; derived computed post-gating. Satisfies AC5; TC3. |
| T14 | Dueling-DRQN net | DQN core track | T2 | high | `agent/dqn.py`, `tests/test_dqn.py` | MLP encoder (256,256) → LSTM(256) → dueling value/advantage → Q over `N_ACTIONS`; init hidden; burn-in support. Test vs fixture sequences. TC8, TC8b. |
| T15 | Prioritized sequence replay | DQN core track | T2 | high | `agent/replay.py`, `tests/test_replay.py` | Store episode sequences; sample length-L sequences; PER (α/β anneal); n-step segments; priority update. TC9. |
| T16 | Training loop | DQN core track | T14, T15 | high | `agent/train.py`, `agent/train_config.py` | n-step Double-DQN target + Huber + soft target (τ≈0.005) + grad-norm clip + burn-in; ε-greedy schedule (per-EPISODE decay). Owns `train_config.py`; READS `agent/contract_config.py` (read-only, never edits). |
| T17 | Reward + tests | Reward/opponent track | T5 | med | `env/reward.py`, `tests/test_reward.py` | Finalize damage-anchored reward; tune `c_step`; potential-based positional shaping; anti-hacking tests. TC5, TC6. |
| T18 | Dummy opponent + interface | Reward/opponent track | T1 | low | `opponents/dummy.py`, `opponents/base.py` | Stationary dummy (Stage 0): idle, knockback-immune, void/fall-immune + an `Opponent` interface the future scripted/snapshot opponents implement. |
| T19 | Eval + reward-component logging | Eval/infra track | T11, T16 | med | `eval/evaluate.py` | Greedy (ε=0) win-rate over 100 eps vs the stage opponent (M2 gate metric); log each reward component separately. Imports `eval/logging.py` read-only; does NOT edit it. |
| T20 | Integration → M2 | Environment/bridge track (named lead) | T9, T12, T13, T16, T17, T19 | high | `env/mc_pvp_env.py`, `agent/train.py` | Wire real DRQN + PerceptionFilter + reward into the env/slice; train vs stationary dummy to ≥95% win / 100 eval eps. Blocked By T9 makes the two writers of `env/mc_pvp_env.py` strictly sequential. Satisfies AC6; TC13. |

**Parallelism (honest):** The human tracks T12, T14, T15, T17, T18 are all gated on the contract-freeze merge (T2, and T5 for T17) — they do NOT start at literal hour zero. Day 1 the human workstreams set up their environments and read the specs while Claude lands T0→T1→Tv→T2/T3/T4/T5/T6. Once the contract PR merges (end of day 1 at earliest), the five human tracks fan out in parallel and offline. Claude's slice (T7a→T7b, T8, T9→T10→T11) runs in parallel with them. T16 waits on T14+T15; T13 on T12; T19 on T11+T16; T20 integrates everything for M2.

## Risks (kickoff-specific)

| Risk | Mitigation |
|---|---|
| Cooldown ownership contradiction (`attack_cooldown` obs vs execution model) | RESOLVED before freeze: bridge owns cooldown via raw `bot.attack` swing-timing; `mineflayer-pvp`/pathfinder pursuit disabled for all macros (T7b). |
| Throughput optimistic / laptop thermal throttling skews the number | Benchmark sustained ≥10 min, record CPU package power/thermal; treat the laptop figure as a lower-bound smoke number, not fleet capacity (T11). |
| Bridge is bus-factor-of-one (Mineflayer has no native step) | Pair the bridge track; adversarial TC7 + live TC7b on damage aggregation; event-aggregation window is the most-tested code. |
| Contract drift on an 8-beginner team | Freeze + version the four artifacts; every change is a PR everyone sees. |
| DRQN recurrence unvalidated at M2 | Gate LSTM correctness via TC8b (memory-dependent fixture), independent of the M2 milestone. |
| Day-1 version pin wrong (plugin lag) | Tv runs the four-way compat check BEFORE T6 writes the pin. |

## Notes for Implementer

- **Freeze the contract before anything else.** A renamed field in `observation_spec.py` or the bridge schema mid-week cascades through 6 tracks. Version it; all changes go through a PR everyone sees. This is the single most important guardrail.
- **The bridge spike (T11) must return numbers, not a green check.** "It runs" is not the exit criterion; transitions/s, p99 round-trip, damage-boundary correctness, and max-arenas-at-TPS are.
- **Damage-event aggregation (T7b) is the deepest silent risk** — it corrupts the reward with no error. TC7 (fixtures) AND TC7b (live exchange) are both required; the real boundary race only shows up on the live bridge.
- **Reset stays behind the `reset` RPC interface.** Commands + read-back gate now; the Bukkit plugin is a backend swap later (per `/debate`). Do not inline reset logic into the env. Bots must be opped.
- **ATTACK uses raw `bot.attack`, movement uses `setControlState`** — not `bot.pvp.attack`/pathfinder goals. The bridge computes `attack_cooldown` so the frozen self-feature is genuinely observable.
- **Per-arena process isolation + core pinning** when the benchmark explores >1 arena; one Paper server is single-threaded, so arenas ≈ cores. Expect ~2-4 arenas on the Core Ultra 7 258V (8c/8t, ~32 GB), not 8 — record the real sustained number, not a cold spot reading.
- **DRQN from the start (user choice):** replay stores sequences (not singletons), training does burn-in; do not shuffle singleton transitions. M2 cannot validate the LSTM — lean on TC8 + TC8b before integration, and remember a green M2 says nothing about recurrence.
- **ε decays per EPISODE, not per step** (short episodes otherwise never decay), and seed `action_space` explicitly — both are known gotchas from prior DQN work.
- **Plot reward components separately** from the first M2 run to catch spin-farming / run-away before blaming the learner.
- **Out of scope creep:** distributed/Redis, scripted bot, self-play, study, and the reset plugin are deferred — stub their dirs, don't build them.
- **Rollback:** all foundation work is on a branch; if the bridge spike reveals an unrecoverable throughput blocker, the measured number is itself the deliverable (it informs a pivot), not a failure.
