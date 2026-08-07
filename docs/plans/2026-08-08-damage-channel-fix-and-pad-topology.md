# Damage-Channel Repair + One-JVM Pad Topology Plan

**Goal:** Repair the opponent-damage reward channel — which has been identically zero for the
entire life of the project — enclose the arena so the agent cannot leave it, and replace the
N-JVM multi-arena topology with N enclosed pads inside a single Paper JVM, sized empirically
against the ≥19 TPS gate.

**Approach:** Merge PR #21 into `main` first (clean 21-commit fast-forward), then land a tight
bugfix branch that (a) sources opponent damage from the dummy bot's OWN connection via
`dummy.on('health')` instead of the learner's entity view, (b) deletes the dead
`_onEntityHurt` damage path so the two can never double-count, (c) encloses each arena in
bedrock walls, and (d) restores the `loss < timeout < win` terminal ordering. Prove the repair
with a **deterministic combat gate** (a scripted face-and-attack probe asserting exact per-hit
health deltas in the correct decision window) before any throughput work. Then retopologize:
N enclosed pads in one flat world, ≥512 blocks apart, one Node bridge process per pad, scaled
through a measured 1/2/4/8/12/16/20/25 ladder with promotion on TPS **and** p99 latency, RSS/GC
stability, reset reliability, and zero cross-pad interaction.

---

## Background — why this plan exists

Five successive reward-coefficient variants failed to teach the agent to attack a stationary,
knockback-immune dummy spawned three blocks in front of it. The cause is not the reward
function.

**Root cause (verified, not hypothesis).** `bridge/bot.js` `_onEntityHurt(entity)` is the only
path converting a landed hit into `damage_dealt`. It computes
`drop = _prevOpponentHealth - entity.health`, where `entity` is the **learner's view** of the
dummy player entity. Mineflayer never populates `health` on non-self entities:
`prismarine-entity`'s `Entity` class defines no `health` field, and the only health assignment
in all of mineflayer is `lib/plugins/health.js:18` — `bot.health = packet.health`, fed by the
`update_health` packet the server sends only about the receiving client's own player.
Therefore `entity.health` is `undefined`, `now` falls back to `_prevOpponentHealth`, `drop` is
always `0`, and **`recordDamageDealt()` has never been called.** Confirmed empirically by
replaying the handler arithmetic against a real `prismarine-entity` for MC 1.21.1.

**Why it shipped.** Every damage unit test calls `recordDamageDealt()` directly on the
aggregator. The event-handler path is untested; the test that would have caught it — "TC7b, the
real damage exchange, real `entityHurt` timing" — exists only as a comment at
`bridge/actions.test.js:35`.

**Compounding factor: the arena has no walls.** `server/arena/data/arena/function/setup.mcfunction`
builds a 25x25 platform (floor `y=63`, bedrock sub-floor `y=62`) and no perimeter. From the
learner's spawn at `x=0.5` the −X edge is 8.5 blocks away — roughly ten `RETREAT` steps. The
codebase knows there is void below: `spawn_learner.mcfunction` carries an *"Anti-void safety"*
comment, and `StationaryDummy` declares `void_immune=True` ("bridge teleports the dummy to
spawn if it reaches the void"). **The dummy is protected from the void; the learner is not.**
Void damage is not covered by the `fallDamage false` gamerule.

**The resulting incentive structure explains every observation:**

| Agent behavior | Old reward (win 8 / timeout 0 / loss −8) | PR #21 reward (win 50 / timeout −30 / loss −8) |
|---|---|---|
| Stand still | timeout → **0** (optimal) | timeout → **−30** |
| Walk off the edge | void death → **−8** | void death → **−8** (now optimal) |
| Land hits | **invisible** (`damage_dealt` ≡ 0) | **invisible** (`damage_dealt` ≡ 0) |

Under the original reward, standing still was strictly optimal. Under the reshape, walking off
the edge became strictly better than standing still — and "it keeps getting away" is what an
edge-walk looks like from outside. The timeout penalty worked; it steered toward the only other
outcome the agent could perceive, because the action that should have mattered was invisible.

---

## Scope

- **In scope:**
  - Merge PR #21 (`feat/multi-arena-throughput`) into `main`.
  - Rewire opponent-damage accounting in `bridge/bot.js` to `dummy.on('health')`; **delete** the
    `_onEntityHurt` damage-recording path.
  - A regression test that drives the **real handler path** with a fake dummy bot whose own
    `health` drops and emits `'health'` — never a direct `recordDamageDealt()` call.
  - Enclosed arena geometry: bedrock perimeter walls + bedrock sub-floor, pad-parameterized.
  - Terminal-reward ordering repair: `loss < timeout < win`.
  - A deterministic combat gate (`eval/combat_probe.py`) asserting exact per-hit health deltas,
    correct decision-window attribution, cumulative 20 HP to a kill, and clean post-respawn
    baselines.
  - Mac bring-up: venv + torch, `setup.sh` on macOS, Java 26 boot verification.
  - Analysis of the Windows `runs/` archive against the falsifiable prediction below.
  - Retopology: N enclosed pads in one world in one JVM; bash launcher; per-pad
    `resetTemplate` coordinates via a new `bridge/run.js` flag; per-bot `/spawnpoint`;
    `max-players` raise; staggered bot joins; two-tier fault policy.
  - Scale ladder 1/2/4/8/12/16/20/25 with multi-criteria promotion.
  - M2 re-baseline across the fleet.
  - Follow-up issues filed (Folia, `c_aim`/`c_step`, `naturalRegeneration`, `wall_distances`
    in the observation, M3 scripted bot).

- **Out of scope:**
  - **Folia.** Regionized multithreading is the only option that would use all 18 cores from one
    JVM, and it is the natural escape hatch if one Paper main thread cannot hold 20–25 pads.
    Explicitly deferred by user decision; filed as an issue.
  - **`c_aim` (0.01) > `c_step` (0.005) inversion.** Looking at a visible opponent is net-positive
    per step. Real, but not this change — filed.
  - **`naturalRegeneration=true`.** The dummy heals ~1 HP/4s, so spread-out hits can never kill
    it. Left alone deliberately; filed, with a farm-watch metric added instead.
  - **Adding `wall_distances` to the observation vector.** Walls will contain the agent
    physically, but it will not perceive them directly — only their consequences via position and
    velocity, which are observed. Adding the field changes `OBS_DIM`, a frozen-contract change
    needing its own PR. Filed.
  - **M3 scripted bot.** `opponents/scripted_bot.py` is a 13-line stub, so M3 has no opponent and
    every downstream algorithm task's "A/B on the M3 scripted benchmark" is blocked. Named as the
    next milestone; filed.
  - **Any change to the frozen wire.** No arena id is added to `bridge/schema.json` —
    per-pad configuration is process-local via bridge argv.
  - Multi-JVM sharding (the contingency if one JVM misses the gate — documented, not built).

---

## Decisions

- **Merge PR #21 first, do not split it** — Codex vetoed this, Fable endorsed it, user
  reconfirmed. The fix touches `bridge/bot.js` and so does PR #21; landing the fix on `main`
  first would destroy a clean 21-commit fast-forward and force a rebase. Codex's attribution
  concern is answered without git surgery: the reward reshape is six values in a frozen
  dataclass, so it can be A/B'd after the fix by flipping the config.
- **Opponent damage comes from the dummy's own connection** — `dummy.on('health')`, mirroring
  the existing `_onSelfHealth` handler. The `update_health` packet is the only health mineflayer
  ever populates, and `_snapshotOpponent()` already reads `this.dummy.health` correctly, so this
  makes the events channel consistent with the rest of the file.
- **Delete `_onEntityHurt`'s damage recording rather than leaving both paths live** — if a future
  mineflayer ever populates `entity.health`, two live recorders would double-count silently.
  `recordOpponentDied` remains covered by the already-wired `dummy.on('death')`.
- **Enclosed pads, not open platforms** — demanded independently by the user, Fable, and Codex.
  Bedrock walls: unbreakable by an iron sword, visible for debugging, consistent with the
  existing sub-floor material.
- **Pad spacing ≥512 blocks, not 128** — the original 128 figure was justified against
  entity-tracking range (32 blocks at `view-distance=2`), which is the wrong bound. A bot at
  ~4.3 m/s over an 80-second episode covers ~344 blocks. Walls make this moot, but spacing is
  free in a superflat world and is defense in depth if a wall ever fails.
- **Cross-pad contact is a correctness hazard, not just a nuisance** — `dummy.on('health')`
  records a health *drop* with **no attacker attribution**. A learner reaching a neighbouring pad
  would silently credit its damage to that pad's policy. This is why walls + spacing + a
  zero-cross-pad promotion criterion are all required together.
- **Terminal ordering repaired now: `loss < timeout < win`** — user overrode the minimal-scope
  decision after Codex flagged it. With `timeout(−30) < loss(−8)`, deliberate death beats running
  out the clock; harmless against a dummy that cannot attack, actively harmful the moment M3's
  opponent can kill. New defaults `win +50 / timeout −15 / loss −30`, all `TUNE`.
- **Deterministic combat gate over a random-policy smoke** — a scripted face-and-attack probe with
  exact assertions, not 20 episodes hoping a random policy stays in reach. The random run remains
  as a secondary smoke.
- **Pads configured bridge-side via argv** — `bridge/run.js` already reads ports and usernames
  from argv/env, and `handleReset` already issues its own `/tp` and regear. Pad origin becomes one
  more argv value flowing into `resetTemplate`, keeping the wire free of an arena id.
- **One Node bridge process per pad** — required twice over: `BridgeServer` accepts exactly one
  TCP client (a second connection destroys the first), and the damage fix needs both bots of a
  pad in the same Node process for `dummy.health` to be readable. Consolidating bridges was
  considered and rejected: one unhandled bot promise is process-fatal in Node, so consolidation
  would turn a one-pad incident into a fleet-wide kill and destroy the fault-isolation domain the
  fault policy depends on.
- **Two-tier fault policy** — a dead pad restarts only its own bridge process and re-runs that
  pad's reset (JVM alive, seconds). A dead JVM is unrecoverable: abort the run loudly rather than
  silently training on fewer arenas.
- **N is an empirical result, not a target** — scaled on the ladder against the existing world-age
  TPS metric (commit `19fd8e5`), with multi-criteria promotion.
- **Minimal-scope rationale corrected** — the original justification ("keep the A/B against
  training history clean") is unsound: the entire history came from a zero-gradient regime and
  PR #21's reshape lands first regardless. The real rule is **one variable per change going
  forward**, with the post-fix re-baseline as the new reference.

---

## Falsifiable prediction (checked in T5, before any code is trusted)

In **every episode of every run** in the Windows `runs/` archive, `r_damage_dealt` is exactly
`0.0`, and `r_damage_taken` is ≈0 except where the learner walked off the edge. If any nonzero
`r_damage_dealt` appears, the diagnosis is incomplete — **stop and reconcile rather than explain
it away.** A high loss-rate against a dummy that cannot attack is direct confirmation of the
edge-walk failure mode.

---

## Data Model

```python
# agent/reward_config.py — terminal block after this change (all TUNE)
R_terminal_win: float = 50.0      # unchanged
R_terminal_timeout: float = -15.0 # was -30.0; now strictly better than a loss
R_terminal_loss: float = 30.0     # was 8.0; stored POSITIVE, applied as -R_terminal_loss
# Invariant: -R_terminal_loss < R_terminal_timeout < R_terminal_win
```

```js
// bridge/run.js argv additions (process-local; NOT on the wire)
--pad-origin "<x>,<y>,<z>"   // pad floor origin; default "0,64,0" == today's single arena
--pad-index  <i>             // 0-based, for logging and username derivation only
```

```js
// bridge/bot.js — ArenaBots.resetTemplate, derived from pad origin
resetTemplate = {
  health: 20.0,
  position: { x: padOrigin.x + 0.5, y: padOrigin.y, z: padOrigin.z + 0.5 },
  inventory: ['iron_sword'],
}
// dummy spawns at position.x + 3 (unchanged relative offset)
```

---

## Contracts & Interfaces

### Signatures

- `wireDamageEvents(): void` — owner: T2. Subscribes `learner.on('health')`,
  `learner.on('death')`, `dummy.on('health')`, `dummy.on('death')`. Idempotent: removes prior
  bound handlers before re-adding. Consumers: T3 (test), T9 (pad-aware construction).
- `_onOpponentHealth(): void` — owner: T2. Replaces `_onEntityHurt`'s damage recording. Reads
  `this.dummy.health`, records a positive drop via `this.events.recordDamageDealt(drop)`,
  re-seeds `_prevOpponentHealth` on a health increase (heal/respawn), and calls
  `recordOpponentDied()` at `<= 0`.
- `padOrigin(index: int) -> {x, y, z}` — owner: T10 (bash launcher) and mirrored in T9.
  Grid layout, `PAD_SPACING = 512`. Single source of truth for pad coordinates.

### File ownership

| File | Owner task | Consumer tasks |
|------|-----------|----------------|
| `bridge/bot.js` | T2 | T9 |
| `bridge/bot.test.js` | T3 | — |
| `agent/reward_config.py` | T4 | — |
| `server/arena/data/arena/function/*.mcfunction` | T6 | T10 |
| `bridge/run.js` | T9 | T10 |
| `server/setup/start-pads.sh` (new) | T10 | T13 |
| `distributed/actor.py` | T12 | — |
| `distributed/launcher.py` | T10 | T12 |
| `eval/combat_probe.py` (new) | T8 | T13 |

### Naming

- `PAD_SPACING = 512` (blocks), `PAD_GRID_COLS = 5`.
- Bot usernames: `learner_<i>` / `dummy_<i>`; `i == 0` keeps `learner_bot` / `dummy_bot` so the
  single-pad path is byte-identical to today.
- Bridge TCP port: `5555 + i`. Minecraft port stays `25565` (one JVM).
- Datapack functions: `arena:setup_pad`, `arena:reset_pad` (coordinate-parameterized).

---

## Acceptance Criteria

- [ ] **AC1** — PR #21 is merged to `main` as a fast-forward; `pytest` and `npm test` are green
      on the merged tree. (TC0)
- [ ] **AC2** — A landed hit records nonzero `damage_dealt` through the real handler path.
      Driving a fake dummy bot's own `health` from 20 → 14 and emitting `'health'` produces
      `damage_dealt == 6` in the current window. (TC1)
- [ ] **AC3** — `_onEntityHurt` no longer records damage; an `entityHurt` event with a
      health-bearing entity records nothing, so the two paths cannot double-count. (TC2)
- [ ] **AC4** — Opponent death still resolves: `dummy.on('death')` sets `opponent_died` exactly
      once, and a health drop to `<= 0` does not double-record it. (TC3)
- [ ] **AC5** — Baseline re-seeding survives heal and respawn: a health *increase* re-seeds
      `_prevOpponentHealth` and records no damage; the first hit after a respawn measures from
      the post-respawn value. (TC4)
- [ ] **AC6** — Terminal ordering holds: `-R_terminal_loss < R_terminal_timeout <
      R_terminal_win`, asserted as a config invariant. (TC5)
- [ ] **AC7** — Each pad is fully enclosed: no reachable path off the floor. A bot issued 400
      consecutive `RETREAT` macros ends the episode inside the pad with `y >= floor`. (TC6, live)
- [ ] **AC8 — the go/no-go gate.** Deterministic combat probe over ≥10 reset/kill cycles: each
      fully-cooled hit produces the dummy's exact health delta **once**, attributed to the correct
      decision window; cumulative dealt damage reaches 20; death fires exactly once; the first
      post-respawn hit measures from a clean baseline. (TC7, live)
- [ ] **AC9** — Windows `runs/` analysis reports the `r_damage_dealt` distribution per run and
      explicitly confirms or refutes the prediction above. (TC8)
- [ ] **AC10** — `python -m eval.run_random --episodes 20` vs the dummy on the Mac completes with
      zero crashes and nonzero mean `r_damage_dealt`. (TC9, live, secondary)
- [ ] **AC11** — N=1 on the new topology is behaviorally identical to today: same ports, same
      usernames, same spawn coordinates; `tests/test_integration_m2.py` and the single-connection
      eval test stay green. (TC10)
- [ ] **AC12** — The pad fleet boots: N pads, 2N bots joined and opped, staggered, all reset
      read-back gates passing, `max-players` sufficient. (TC11, live)
- [ ] **AC13** — Zero cross-pad interaction: over a ≥10-minute N≥8 run, no pad observes an entity
      belonging to another pad, and per-pad cumulative `damage_dealt` reconciles against per-pad
      dummy health loss. (TC12, live)
- [ ] **AC14** — Scale ladder measured at 1/2/4/8/12/16/20/25 with TPS, p99 step latency, RSS, GC
      pause, and reset success rate recorded per rung; the promoted N is the largest rung passing
      **all** criteria. (TC13, live)
- [ ] **AC15** — Two-tier fault policy: killing one pad's bridge restarts only that bridge and the
      pad resumes; killing the JVM aborts the run loudly with a clear message. (TC14)
- [ ] **AC16** — M2 re-baseline on the fleet reaches greedy win-rate ≥95% over 100 eval episodes
      vs the stationary dummy, with aim-bonus-while-invisible == 0. (TC15, live)
- [ ] **AC17** — Follow-up issues filed: Folia, `c_aim`/`c_step`, `naturalRegeneration`,
      `wall_distances` in observation, M3 scripted bot. (review)

---

## Error Handling

- **Dummy `health` unavailable at wire time** (bot not yet spawned): treat as no signal, record
  nothing, leave `_prevOpponentHealth` untouched. Never record a phantom drop from an undefined
  value — that is the class of bug this plan exists to fix.
- **Health increase** (heal, regen, respawn): re-seed the baseline and record zero. Never record
  negative damage.
- **Pad bridge process dies:** ActorPool restarts that bridge only, re-runs `arena:reset_pad` for
  its coordinates, and resumes. Backoff in seconds.
- **Paper JVM dies:** unrecoverable — every pad is gone. Abort the run loudly with an explicit
  message naming the JVM as the cause. Do not silently continue on survivors.
- **Bot join storm:** stagger joins; on join failure, retry that bot with backoff before failing
  the pad.
- **`max-players` exceeded:** fail loudly at launch with the computed requirement
  (`2N + slack`), not at the 21st bot's silent rejection.
- **Java 26 incompatibility:** if Paper refuses to boot or warns, install Temurin 21 and pin
  `JAVA_HOME` in `start.sh`. Documented fallback, decided by the user.
- **Combat probe fails (AC8):** stop. Do not proceed to topology work on an unproven repair.

---

## Testing Strategy

**Levels:** Unit (Node + Python), Integration, Live.

| ID | Test Case | Type | Expected Behavior |
|-----|-----------|------|-------------------|
| TC0 | Merged tree runs both suites | Integration | `pytest` 353+ passed; `npm test` green |
| TC1 | Fake dummy bot health 20→14, emits `'health'` | Unit (Node) | `damage_dealt == 6` in the current window, via the real handler |
| TC2 | `entityHurt` fired with an entity carrying `health` | Unit (Node) | Records nothing — the old path is gone, no double-count |
| TC3 | Dummy `'death'` event | Unit (Node) | `opponent_died == true` exactly once; a `<=0` health drop does not re-record |
| TC4 | Dummy health 20→14, then →20 (respawn), then →14 | Unit (Node) | `damage_dealt` totals 6 then 6; no negative, no phantom 14 |
| TC5 | RewardConfig terminal invariant | Unit (Py) | `-R_terminal_loss < R_terminal_timeout < R_terminal_win`; violating values raise |
| TC6 | 400 consecutive RETREAT macros | Live | Bot remains inside the pad, `y >= floor`, episode ends by timeout not death |
| TC7 | Deterministic combat probe, ≥10 reset/kill cycles | Live | Exact per-hit delta once, correct window, cumulative 20 HP, one death, clean post-respawn baseline |
| TC8 | Windows `runs/` archive parsed | Analysis | `r_damage_dealt` distribution reported per run; prediction confirmed or refuted explicitly |
| TC9 | `eval.run_random --episodes 20` | Live | 0 crashes, nonzero mean `r_damage_dealt` |
| TC10 | N=1 on the new topology | Integration | Byte-identical ports/usernames/coords; existing integration + single-connection tests green |
| TC11 | Fleet boot at N=8 | Live | 16 bots joined, opped, staggered; all read-back gates pass |
| TC12 | Cross-pad isolation at N≥8 over ≥10 min | Live | No foreign entity observed; per-pad `damage_dealt` reconciles with per-pad dummy health loss |
| TC13 | Scale ladder 1/2/4/8/12/16/20/25 | Live | Per-rung TPS, p99 latency, RSS, GC pause, reset success recorded |
| TC14 | Kill one bridge; separately kill the JVM | Live | Bridge case: that pad only, restarts and resumes. JVM case: loud abort |
| TC15 | M2 re-baseline on the fleet | Live | Greedy win ≥95%/100 eps; aim-while-invisible == 0 |
| TC16 | Damage-farm watch | Analysis | Episodes with `damage_dealt > 20` HP logged as a regen-farming indicator |

**Test data:** Node tests use a fake dummy *bot* object (an EventEmitter with a mutable `health`
property), never a fake entity and never a direct `recordDamageDealt()` call — driving the real
handler is the entire point. Python tests use the existing fixtures.
**Run commands:** `pytest` · `cd bridge && npm test` · `python -m eval.combat_probe --cycles 10`

---

## Tasks

| ID | Task | Blocked By | Risk | Files | Description |
|----|------|------------|------|-------|-------------|
| T1 | Merge PR #21 to main | — | med | *(git)* | Fast-forward merge `feat/multi-arena-throughput` (21 commits, 0 conflicts, no review comments). Run `pytest` and `cd bridge && npm test` on the merged tree. Satisfies AC1. Do not squash — preserve the commit history. |
| T2 | Repair the damage channel | T1 | high | `bridge/bot.js` | In `wireDamageEvents()`, add `dummy.on('health')` → new `_onOpponentHealth()` mirroring `_onSelfHealth`: read `this.dummy.health`, record positive drops via `recordDamageDealt`, re-seed `_prevOpponentHealth` on an increase, `recordOpponentDied()` at `<=0`. **Delete** the damage-recording body of `_onEntityHurt` and its registration. Keep the idempotent off/on re-wiring and bound-reference bookkeeping exactly as-is. Keep `dummy.on('death')`. Satisfies AC2, AC3, AC4, AC5. |
| T3 | Regression tests for the handler path | T2 | low | `bridge/bot.test.js` | Add TC1–TC4 driving the REAL handler with a fake dummy bot (EventEmitter, mutable `health`). Never call `recordDamageDealt()` directly — that hole is what shipped this bug. Satisfies AC2–AC5. |
| T4 | Repair terminal reward ordering | T1 | med | `agent/reward_config.py`, `tests/test_reward.py` | Set `R_terminal_win=50.0`, `R_terminal_timeout=-15.0`, `R_terminal_loss=30.0`. Add a `__post_init__` invariant that `-R_terminal_loss < R_terminal_timeout < R_terminal_win` and raises otherwise. Rewrite the docstring rationale: the old "timeout is the worst outcome" reasoning is superseded. Satisfies AC6, TC5. |
| T5 | Analyze the Windows runs/ archive | — | low | *(scratchpad script)* | Parse every `runs/*/metrics.jsonl`. Report per-run `r_damage_dealt` distribution, win/loss/timeout split, and episode-end `y` where available. Explicitly confirm or refute the falsifiable prediction. **If any nonzero `r_damage_dealt` appears, halt and report — do not proceed.** Satisfies AC9, TC8. |
| T6 | Enclose the arena, parameterize by pad | T1 | med | `server/arena/data/arena/function/*.mcfunction` | Rewrite `setup.mcfunction` into `setup_pad` taking a coordinate origin (`execute positioned`): bedrock sub-floor, smooth-stone floor, air interior, and a **closed bedrock perimeter wall** 5 blocks tall around the 25x25 footprint. Split `reset.mcfunction` into `reset_pad`. Keep global gamerules in a one-shot `setup` function. Satisfies AC7. |
| T7 | Mac bring-up | — | low | `server/setup/setup.sh`, *(env)* | Create a Python venv (system Python is 3.9.6 — needs 3.11+), install `-e .` plus torch. Run `setup.sh` on macOS and fix any bashism/`curl` issues. Boot Paper on Java 26; if it refuses or warns, install Temurin 21 and pin `JAVA_HOME` in `start.sh`. Verify both bots join and are opped. |
| T8 | Deterministic combat probe | T2, T6, T7 | high | `eval/combat_probe.py` *(new)* | Scripted probe: face the dummy, issue fully-cooled ATTACK macros, assert each hit produces the dummy's exact health delta once in the correct decision window, cumulative 20 HP kills, death fires once, first post-respawn hit uses a clean baseline. `--cycles N`, default 10. **This is the go/no-go gate.** Satisfies AC8, TC7. |
| T9 | Pad-aware bridge configuration | T2 | med | `bridge/run.js`, `bridge/bot.js` | Add `--pad-origin "x,y,z"` and `--pad-index i` to argv/env parsing (mirroring the existing port/username handling). Derive `resetTemplate.position` from the origin; dummy stays at `+3` on X. Default `0,64,0` so N=1 is byte-identical. **No arena id on the wire.** Satisfies AC11. |
| T10 | Bash pad-fleet launcher | T6, T9 | high | `server/setup/start-pads.sh` *(new)*, `distributed/launcher.py` | Replace Topology A. One `setup.sh`-provisioned server root; boot one JVM; run `arena:setup_pad` per pad at `padOrigin(i)` (grid, `PAD_SPACING=512`); write `ops.json` for all `learner_i`/`dummy_i`; raise `max-players` to `2N + 10`; issue per-bot `/spawnpoint` at its own pad; launch N bridge processes on `5555+i` with **staggered joins**. Delete `start-arenas.ps1` and the N-JVM path in `launcher.py`. Satisfies AC12. |
| T11 | Two-tier fault policy | T10 | high | `distributed/actor.py` | Replace per-arena JVM relaunch. Dead bridge → restart that bridge, re-run `arena:reset_pad` for its origin, resume. Dead JVM (detect via MC port unreachable) → abort the whole run loudly naming the JVM. Remove `fault_min_live_arenas` silent-degradation behavior. Satisfies AC15, TC14. |
| T12 | Cross-pad isolation verification | T10 | med | `eval/benchmark.py` | Add a per-pad reconciliation check: cumulative `damage_dealt` vs per-pad dummy health loss, plus an assertion that no pad observes a foreign entity. This guards the no-attacker-attribution hazard. Satisfies AC13, TC12. |
| T13 | Scale ladder measurement | T10, T11, T12 | med | `eval/benchmark.py`, `RUNBOOK.md` | Run 1/2/4/8/12/16/20/25. Record per rung: world-age TPS, p99 step latency, RSS, GC pause, reset success rate. Promoted N = largest rung passing all criteria. Document the procedure and the measured table. Satisfies AC14, TC13. |
| T14 | M2 re-baseline on the fleet | T8, T13 | med | *(run)* | Train vs the stationary dummy at the promoted N until the greedy gate passes. Watch per-component reward from episode 1, especially `r_damage_dealt` and the TC16 farm-watch metric. Satisfies AC16. |
| T15 | File follow-up issues | T1 | low | *(GitHub)* | Folia; `c_aim` > `c_step` inversion; `naturalRegeneration`; `wall_distances` in the observation (frozen-contract change); M3 scripted bot (blocks every downstream algorithm A/B). Satisfies AC17. |
| T16 | Update docs | T13, T14 | low | `README.md`, `RUNBOOK.md`, `server/README.md` | Replace the N-JVM/PowerShell procedure with the macOS pad-fleet procedure. Record the measured ladder. Remove the stale "Don't run this until parallel arena is finished" note. Document the new arena geometry. |

**Parallelism:** T5 and T7 have no blockers and can start immediately alongside T1. T2, T4, and
T6 fan out from T1 with no file overlap. T3 and T9 both wait on T2 (both touch or test
`bridge/bot.js`). T8 is the gate — nothing downstream of it should start until it passes.

---

## Notes for Implementer

- **T2 is the whole point of this plan.** Read `_onSelfHealth` first and mirror it exactly; the
  asymmetry between the two handlers is what caused this bug.
- **Never derive another bot's internal state from the entity view.** Health, hunger, XP, and
  effects come only from that bot's own connection. Position, yaw, velocity, and equipment are
  fine from the entity view. This is the generalized lesson.
- **The frozen contract is PR-gated:** `env/observation_spec.py`, `bridge/schema.json` +
  `messages.py`, `agent/actions.py`, and the `compute_reward` signature. This plan touches none
  of them. T4 changes reward *values*, not the signature — allowed.
- **`server.properties` is regenerated by `setup.sh`.** Any `max-players` change must go into the
  script, not the generated file, or it will be silently overwritten on the next setup run.
- **`doImmediateRespawn` is true and world spawn is shared.** Without a per-bot `/spawnpoint`,
  every death teleports the bot to pad 0 — which would be a cross-pad contamination event.
- Both bots of a pad must live in one Node process. The plan depends on it twice over; a future
  refactor that splits them silently breaks damage accounting again.
- **Rollback:** T2 is a small, self-contained diff in one function. If the combat probe (T8) fails
  in a way that implicates the fix rather than the environment, revert T2's commit and re-run the
  probe against the old path to confirm the delta before iterating.
- Live steps (T7, T8, T10–T14) need Paper running; run order is always **Paper → bridges →
  Python driver**.
- Git: feature branches only, never commit to `main` directly. No AI attribution in commits or
  PRs — match the repo's existing commit voice (`git log -20 --oneline` first).
