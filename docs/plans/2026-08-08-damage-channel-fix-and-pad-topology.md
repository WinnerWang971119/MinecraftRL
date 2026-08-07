# Damage-Channel Repair + One-JVM Pad Topology Plan

**Goal:** Repair the opponent-damage reward channel — which has been identically zero for the
entire life of the project — enclose the arena so the agent cannot leave it, and replace the
N-JVM multi-arena topology with N enclosed pads inside a single Paper JVM, sized empirically
against the ≥19 TPS gate.

**Approach:** Merge PR #21 into `main` first (clean 21-commit fast-forward), then land a tight
bugfix branch that (a) sources opponent damage from the dummy bot's OWN connection via
`dummy.on('health')` instead of the learner's entity view, (b) deletes the dead `_onEntityHurt`
damage path so the two can never double-count, (c) encloses each arena in bedrock walls,
(d) restores `loss < timeout < win` on the terminal rewards, and (e) makes the opponent's
health stationary across episodes (regeneration off, hunger reset). Prove the repair with a
**deterministic combat gate** before any throughput work. Then retopologize: N enclosed pads in
one flat world, ≥512 blocks apart, addressed by **datapack macro functions**, one Node bridge
process per pad, scaled through a measured 1/2/4/8/12/16/20/25 ladder.

---

## Background — why this plan exists

Five successive reward-coefficient variants failed to teach the agent to attack a stationary,
knockback-immune dummy spawned three blocks in front of it. The cause is not the reward function.

**Root cause (verified at primary source, and independently confirmed by both reviewers).**
`bridge/bot.js` `_onEntityHurt(entity)` is the only path converting a landed hit into
`damage_dealt`. It computes `drop = _prevOpponentHealth - entity.health`, where `entity` is the
**learner's view** of the dummy player entity. Mineflayer never populates `health` on non-self
entities: `prismarine-entity`'s `Entity` class defines no `health` field, and the only health
assignment in all of mineflayer is `lib/plugins/health.js:18` — `bot.health = packet.health`,
fed by the `update_health` packet the server sends only about the receiving client's own player.
Therefore `entity.health` is `undefined`, `drop` is always `0`, and **`recordDamageDealt()` has
never been called.** Verified empirically by replaying the handler arithmetic against a real
`prismarine-entity` for MC 1.21.1.

**Why it shipped — mock infidelity, not missing coverage.** `bridge/actions.test.js:591` and
`:685` *do* drive the real handler (`learner.emit('entityHurt', dummy.entity)`) and assert
`damage_dealt === 5`. They pass because the fake entity is given a `health` property that real
`prismarine-entity` objects never have. **A mock more capable than reality tests nothing.** No
amount of additional tests against the same fake would have caught this. The lesson for T3 is
therefore *fakes must only populate fields the library actually populates* — not "add more
tests."

**What was and wasn't invisible.** The *per-hit* signal was dead. The *kill outcome* was not:
`dummy.on('death') → recordOpponentDied` was always correctly wired (`bridge/bot.js:649-652`),
so a completed kill did pay the terminal win reward. **This gives T5 its sharpest single
confirmation: any won episode with `damage_dealt == 0` is a direct fingerprint of the bug.**

**Compounding factor: the arena has no walls.** `setup.mcfunction` builds a 25x25 platform
(floor `y=63` spanning `x=-8..16, z=-12..12`, bedrock sub-floor `y=62`) and no perimeter. From
the learner's spawn at `(0.5, 64, 0.5)` the −X edge is 8.5 blocks away — roughly ten `RETREAT`
steps.

**Correction to an earlier reading of this plan: falling off is almost certainly NOT a death.**
Both setup scripts write `level-type=minecraft:flat` with `generator-settings={}`, whose 1.18+
default preset places solid ground at y≈−60 (bedrock at −64). With `fallDamage false`, a learner
that walks off the y=63 platform **lands alive ~123 blocks below and is stranded there for the
rest of the episode.** The void is unreachable. The `void_immune` flag on the dummy and the
"anti-void safety" comment in `spawn_learner.mcfunction` are defensive leftovers, not evidence
of reachable void. **This inference is from the generation config and MUST be verified at first
boot (T7) before T5's analysis is interpreted.**

**The corrected incentive structure:**

| Agent behavior | Old reward (win 8 / timeout 0 / loss −8) | PR #21 reward (win 50 / timeout −30 / loss −8) |
|---|---|---|
| Stand still | timeout → **0** | timeout → **−30** |
| Walk off the edge | stranded → timeout → **0** (tied) | stranded → timeout → **−30** (tied) |
| Land hits | per-hit reward **invisible** | per-hit reward **invisible** |
| Complete a kill | win **+8** (paid correctly) | win **+50** (paid correctly) |

Under the original reward, standing still and wandering off were both worth exactly 0, and the
only positive dense term was the aim bonus (`c_aim` 0.01 > `c_step` 0.005), making "stare at the
opponent" a net-positive absorbing behavior. Under the reshape, every non-kill outcome collapsed
to the same −30 constant — **zero gradient between stalling and wandering**. In both regimes the
one action that should have mattered paid nothing. That is why five coefficient variants all
failed, and why no sixth would have worked.

**A third source of nonstationarity: the dummy's health is not reset-stable.** Nothing in
`handleReset` or either spawn mcfunction restores food or saturation — only `instant_health`,
which heals HP and never hunger. A fresh full-saturation player regenerates 1 HP per 0.5 s,
decaying to 1 HP per 4 s as saturation drains, then to zero once food drops below 18. Since
regeneration costs exhaustion and food is never replenished, **the dummy's regeneration rate
silently decays across episodes** — uncontrolled cross-episode state contaminating every run.

---

## Scope

- **In scope:**
  - Merge PR #21 (`feat/multi-arena-throughput`) into `main`.
  - Rewire opponent-damage accounting in `bridge/bot.js` to `dummy.on('health')`; **delete** the
    `_onEntityHurt` damage-recording path and rewrite the tests that assert the old behavior.
  - A regression suite driving the real handler with a **faithful** fake (a dummy *bot* whose own
    `health` changes), including negative paths.
  - Enclosed arena geometry: bedrock perimeter walls, addressed by macro functions.
  - Terminal-reward ordering repair: `loss < timeout < win`, with sign-semantics validation.
  - **Opponent-health stationarity:** `naturalRegeneration` off, and food/saturation restored at
    reset. Promoted from out-of-scope — see Decisions.
  - `bukkit.yml` with `connection-throttle: -1` (the join-storm root cause).
  - A deterministic combat gate (`eval/combat_probe.py`) cross-checked against the wire's
    privileged `state.opponent.health`.
  - Mac bring-up: venv + torch, `setup.sh` on macOS, Java 26 boot verification, **and empirical
    verification of what lies below the platform**.
  - Analysis of the Windows `runs/` archive (blocked on the user supplying it).
  - Retopology: N enclosed pads in one world in one JVM; bash launcher; per-pad configuration;
    per-bot `/spawnpoint`; `max-players`; staggered joins; two-tier fault policy.
  - Scale ladder with quantified multi-criteria promotion.
  - M2 re-baseline across the fleet.
  - Follow-up issues filed.

- **Out of scope:**
  - **Folia.** The only option that would use all 18 cores from one JVM, and the natural escape
    hatch if one Paper main thread cannot hold 20–25 pads. Deferred by user decision; filed.
  - **`c_aim` (0.01) > `c_step` (0.005) inversion.** Real (it makes staring net-positive), but
    the dense damage signal now dominates it. Filed.
  - **Adding `wall_distances` to the observation.** Changes `OBS_DIM` — a frozen-contract change
    needing its own PR. Filed. **Note:** the observation contains self *velocity* but **no self
    position** (`env/observation_spec.py`, indices 0..10), so an agent pressed against a wall
    perceives it only as velocity going to zero, not as a position bound.
  - **M3 scripted bot.** `opponents/scripted_bot.py` is a 13-line stub; every downstream
    algorithm task's "A/B on the M3 scripted benchmark" is blocked on it. Filed as next milestone.
  - **Any change to the frozen wire.** No arena id is added; per-pad config is process-local.
  - Multi-JVM sharding (documented contingency, not built).

---

## Decisions

- **Merge PR #21 first, do not split it** — Codex vetoed, Fable endorsed, user reconfirmed. The
  fix touches `bridge/bot.js` and so does PR #21; fix-first would destroy a clean 21-commit
  fast-forward. Codex's attribution concern is answered without git surgery: the reward reshape
  is a handful of values in a frozen dataclass and can be A/B'd by flipping the config.
- **The reshape inventory is win 8→50, timeout 0→−30, AND `c_dmg_in` 1.0→0.5.** The third is
  easy to overlook and is the only one that had real historical effect — `damage_taken` worked
  the whole time, unlike `damage_dealt`.
- **Opponent damage comes from the dummy's own connection** — `dummy.on('health')`, mirroring
  `_onSelfHealth`. `_snapshotOpponent()` already reads `this.dummy.health` correctly, so this
  makes the events channel consistent with the rest of the file.
- **Delete `_onEntityHurt`'s damage recording rather than leaving both live** — if a future
  mineflayer ever populates `entity.health`, two recorders would double-count silently.
- **Wire health is the probe's independent oracle, not the production path** — `state.opponent.health`
  already carries the dummy's true health every step. Deriving `damage_dealt` from it Python-side
  was considered and rejected for production: window-boundary sampling conflates damage with
  regeneration, and it would create a second source of truth competing with the frozen `events`
  channel. It is kept as a **free cross-check** inside the combat probe.
- **Enclosed pads, not open platforms** — demanded independently by the user and both reviewers.
  Bedrock: unbreakable by an iron sword, visible for debugging, consistent with the sub-floor.
- **Pad addressing uses 1.20.2+ datapack macro functions, NOT `execute positioned`** — this was a
  factual error in the first draft, caught by both reviewers. `execute positioned` relocates only
  `~`-relative coordinates; every command in today's arena is absolute (`fill -8 63 -12 …`,
  `tp learner_bot 0.5 64 0.5`). Worse, entity-selector `x/y/z` arguments cannot be relative at
  all, and player-name commands (`tp`/`clear`/`attribute` on `learner_<i>`) cannot be positionally
  parameterized by any mechanism. Macro functions take `$(x)`, `$(z)`, `$(learner)`, `$(dummy)`
  and solve coordinates and usernames in one mechanism.
- **Server commands are delivered by the pad's own opped bridge bots, not by the launcher** —
  RCON is disabled in the generated `server.properties`, and the bash launcher has no console
  channel. `handleReset` already issues `/tp`, `/clear`, `/give`, and `/effect` via chat as an
  opped bot; extending that to `/function arena:setup_pad {…}` and `/spawnpoint` needs no new
  transport, no RCON, and no stdin plumbing. **The bridge is the sole reset authority.**
- **Pad spacing ≥512 blocks, not 128** — the 128 figure was justified against entity-tracking
  range (32 blocks at `view-distance=2`), the wrong bound. A bot at ~4.3 m/s over an 80-second
  episode covers ~344 blocks. Walls make this moot; spacing is free and is defense in depth.
- **Cross-pad contact is a correctness hazard** — `dummy.on('health')` records a health *drop*
  with **no attacker attribution**. A learner reaching a neighbouring pad would silently credit
  its damage to that pad's policy. Walls + spacing + reconciliation are required together.
- **`naturalRegeneration` off, and hunger reset — promoted into scope.** Both reviewers showed the
  original defer was untenable. Codex: with regeneration on, `20 HP dealt − 15 timeout − 2 step =
  **+3**`, so farming a dummy that cannot die is *net-positive* — directly defeating the user's
  goal of making timeouts strongly undesirable. Fable: interleaved `+1` heals between cooled hits
  make AC8's exact-delta assertions **false-negative on a correct implementation**, and the dummy's
  regeneration rate decays across episodes because food is never restored. This is a second
  override of the original minimal-scope decision, on the same grounds as the first.
- **Terminal ordering repaired: `loss < timeout < win`** — user decision after Codex flagged that
  `timeout(−30) < loss(−8)` makes deliberate death beat running out the clock; harmless against a
  dummy that cannot attack, actively harmful the moment M3's opponent can kill. New defaults
  `win +50 / timeout −15 / loss −30`, all `TUNE`. The invariant must also pin **sign semantics**
  (`R_terminal_loss > 0`, `R_terminal_timeout < 0`, `R_terminal_win > 0`, all finite), since the
  ordering alone admits nonsense configurations.
- **One Node bridge process per pad** — required by `BridgeServer`'s single-TCP-client rule and by
  `dummy.health` locality. **Correction:** the first draft justified this partly on "an unhandled
  promise is process-fatal in Node." That is false — `bridge/run.js:27` installs an
  `unhandledRejection` handler that logs and continues. The decision stands on fault-isolation
  grounds (a per-pad process is the restart unit the fault policy depends on), and that isolation
  is to be validated by crash fault injection in TC14, not asserted.
- **Two-tier fault policy** — dead pad restarts only its own bridge; dead JVM aborts the run
  loudly. `fault_min_live_arenas` is **deleted**, not reconfigured: a survivor floor directly
  contradicts "abort rather than silently train on fewer arenas."
- **N is an empirical result, not a target** — promoted against quantified criteria (below).
- **Minimal-scope rationale corrected** — "keep the A/B against training history clean" is
  unsound: the history came from a zero-gradient regime and PR #21's reshape lands first
  regardless. The rule is **one variable per change going forward**, with the post-fix
  re-baseline as the new reference.

---

## Falsifiable prediction (T5 — checked before the repair is trusted)

In every episode of every run in the Windows archive:

1. **`r_damage_dealt` is exactly `0.0`.** The primary prediction.
2. **Won episodes show `damage_dealt == 0`** — the sharpest fingerprint, since kills were paid
   correctly while per-hit damage was not.
3. **Losses are near-absent.** With a passive dummy, `fallDamage`/fire/drowning/freeze off, and
   (predicted) solid ground below, there is essentially no loss mechanism; starvation on normal
   difficulty stops at 1 HP. A meaningful loss rate would **refute** the geometry analysis.
4. **Wander-offs appear as timeouts with episode-end `y ≈ −60`**, not as deaths.

If any nonzero `r_damage_dealt` appears, **halt and reconcile — do not explain it away.**

---

## Data Model

```python
# agent/reward_config.py — terminal block (all TUNE)
R_terminal_win: float = 50.0      # unchanged
R_terminal_timeout: float = -15.0 # was -30.0; now strictly better than a loss
R_terminal_loss: float = 30.0     # was 8.0; stored POSITIVE, applied as -R_terminal_loss

# __post_init__ must enforce ALL of:
#   all three finite
#   R_terminal_loss > 0 and R_terminal_timeout < 0 and R_terminal_win > 0
#   -R_terminal_loss < R_terminal_timeout < R_terminal_win
```

```js
// bridge/run.js argv additions (process-local; NOT on the wire)
--pad-origin "<x>,<z>"   // pad ANCHOR (see below); default "0,0" == today's single arena
--pad-index  <i>         // 0-based; used for usernames and logging only
```

```mcfunction
# arena:setup_pad — macro function. Called with {x:<int>, z:<int>}
# Coordinates below are relative to the pad ANCHOR, matching today's absolute layout:
#   floor      y=63, x anchor-8..anchor+16, z anchor-12..anchor+12
#   sub-floor  y=62, same footprint
#   interior   y=64..71 air
#   walls      y=64..71 bedrock, forming a CLOSED ring including all four corners
$fill $(x)-8 62 $(z)-12 $(x)16 62 $(z)12 minecraft:bedrock replace
# ... (full command set pinned by T6)
```

```mcfunction
# arena:reset_pad — macro function. Called with {x, z, learner, dummy}
$tp $(learner) $(x)0.5 64 $(z)0.5 90 0
$tp $(dummy)   $(x)3.5 64 $(z)0.5 -90 0
# ... regear, health, effects, hunger, spawnpoint
```

**Anchor vs. floor origin.** `(0, 64, 0)` is the **spawn anchor**, not the floor origin. The real
floor spans `x = anchor−8 … anchor+16`, `z = anchor−12 … anchor+12` at `y = 63`; learner feet sit
at `(anchor+0.5, 64, anchor+0.5)`. The first draft conflated these. All pad math is expressed
relative to the anchor.

---

## Contracts & Interfaces

### Signatures

- `wireDamageEvents(): void` — owner: T2. Subscribes `learner.on('health'|'death')` and
  `dummy.on('health'|'death')`. Idempotent: removes prior bound handlers before re-adding.
- `_onOpponentHealth(): void` — owner: T2. Reads `this.dummy.health`; records positive drops via
  `this.events.recordDamageDealt(drop)`; re-seeds `_prevOpponentHealth` on an increase; calls
  `recordOpponentDied()` at `<= 0`. Ignores `undefined`/non-finite health without touching the
  baseline.
- `padAnchor(index: int) -> {x: int, z: int}` — **owner: T10, sole implementation.** Formula:
  `x = (i % PAD_GRID_COLS) * PAD_SPACING`, `z = (i // PAD_GRID_COLS) * PAD_SPACING`. T9 only
  *parses* the value handed to it on argv; `distributed/actor.py` receives it via config. There is
  no mirror.
- `restart_bridge(pad_index: int) -> None` and `jvm_alive() -> bool` — owner: T11. The T10↔T11
  seam. `jvm_alive()` probes the Minecraft port; a False result is the abort trigger.

### File ownership

| File | Owner task | Consumer tasks |
|------|-----------|----------------|
| `bridge/bot.js` | T2 | T9 (sequential co-owner) |
| `bridge/bot.test.js` | T3 | — |
| `bridge/actions.test.js` | T3 | — |
| `agent/reward_config.py` | T4 | — |
| `tests/test_reward.py` | T4 | — |
| `server/arena/data/arena/function/*.mcfunction` | T6 | T10 |
| `server/setup/setup.sh` | T7 | T10 (sequential; T10 blocked by T7) |
| `bridge/run.js` | T9 | T10 |
| `eval/combat_probe.py` (new) | T8 | T13 |
| `server/setup/start-pads.sh` (new) | T10 | T13 |
| `distributed/launcher.py` | T10 | T11 |
| `distributed/actor.py` | T11 | — |
| `agent/train.py` | T11 | — |
| `agent/train_config.py` | T11 | — |
| `tests/test_actor_pool.py` | T11 | — |
| `tests/test_learner_loop.py` | T11 | — |
| `tests/test_benchmark_overlap.py` | T11 | — |
| `eval/benchmark.py` | T12 | T13 |

### Naming

- `PAD_SPACING = 512`, `PAD_GRID_COLS = 5`.
- Usernames `learner_<i>` / `dummy_<i>`. **`i == 0` uses `learner_bot` / `dummy_bot`** so the
  manual single-arena path is byte-identical. Note this *changes* PR #21's launcher default,
  which used `learner_0` at `i == 0`.
- Bridge TCP port `5555 + i`; Minecraft port stays `25565` (one JVM).
- Datapack macro functions `arena:setup_pad`, `arena:reset_pad`.

---

## Acceptance Criteria

- [ ] **AC1** — PR #21 merged to `main` as a fast-forward; both suites green on the merged tree
      at the count observed during T1. (TC0)
- [ ] **AC2** — A landed hit records nonzero `damage_dealt` through the real handler: a fake dummy
      *bot* whose own `health` drops 20 → 14 and emits `'health'` yields `damage_dealt == 6`. (TC1)
- [ ] **AC3** — `_onEntityHurt` records nothing; an `entityHurt` event with a health-bearing entity
      is inert, so the paths cannot double-count. (TC2)
- [ ] **AC4** — Opponent death resolves exactly once, and the `opponent_died` latch clears across
      `drain()`. (TC3)
- [ ] **AC5** — Baseline re-seeding survives heal and respawn; `undefined`/NaN health records
      nothing and leaves the baseline untouched; double-`wireDamageEvents()` does not double-count.
      Asserted with a recorder spy on **call counts**, not just final values. (TC4)
- [ ] **AC6** — Terminal invariant holds including sign semantics and finiteness; violating configs
      raise. (TC5)
- [ ] **AC7** — Each pad is fully enclosed on **all four walls and all four corners**, with exact
      x/z bounds asserted and wall continuity and height verified. (TC6, live)
- [ ] **AC8 — the go/no-go gate.** With regeneration disabled, ≥10 reset/kill cycles: the recorded
      per-hit sequence is exactly `6, 6, 6, 2`; cumulative dealt damage is exactly 20; death fires
      once; the first post-respawn hit measures from a clean baseline; and every recorded value
      reconciles against the wire's `state.opponent.health`. Window attribution accepted at ±1
      window (the dummy's `update_health` arrives on a second connection). (TC7, live)
- [ ] **AC9** — Windows archive analysis reports, per run, the `r_damage_dealt` distribution,
      win/loss/timeout split, won-episodes-with-zero-damage count, and episode-end `y`; and
      explicitly confirms or refutes all four predictions. Unreadable or unrecognized artifacts are
      listed, not silently skipped. (TC8)
- [ ] **AC10** — `eval.run_random --episodes 20` on the Mac: 0 crashes, nonzero mean
      `r_damage_dealt`. (TC9, secondary)
- [ ] **AC11** — N=1 is byte-identical to the manual single-arena path: same ports, usernames,
      coordinates; existing integration and single-connection tests green. (TC10)
- [ ] **AC12** — Fleet boots: N pads, 2N bots joined, opped, staggered, all read-back gates
      passing, `max-players` sufficient, no connection-throttle rejections. (TC11, live)
- [ ] **AC13** — Zero cross-pad interaction over a ≥10-minute N≥8 run, proven by **per-pad damage
      reconciliation** (cumulative `damage_dealt` vs. per-pad dummy health loss) plus a bridge-side
      `bot.entities` foreign-username scan logged per pad. (TC12, live)
- [ ] **AC14** — Ladder measured at 1/2/4/8/12/16/20/25. Promoted N = largest rung meeting **all**:
      world-age TPS ≥ 19.0; p99 step round-trip ≤ 250 ms (25% over the 200 ms budget); Paper RSS
      growth < 200 MB over the rung; max GC pause < 50 ms (one tick); reset success ≥ 99.5%.
      (TC13, live)
- [ ] **AC15** — Two-tier fault policy, validated by injection: killing one bridge restarts only
      that bridge; killing the JVM aborts loudly. `fault_min_live_arenas` is fully removed with no
      dangling references. (TC14)
- [ ] **AC16** — From a pinned seed set (3 seeds) and a pinned step budget, the *fixed* final
      checkpoint scores greedy win-rate ≥95% over 100 eval episodes vs. the stationary dummy, with
      aim-bonus-while-invisible == 0, reported per seed. "Training completed" and "gate passed" are
      recorded separately. (TC15, live)
- [ ] **AC17** — Mac bring-up verified: Python ≥3.11 in a venv, `torch` imports, Java selection
      recorded, Paper boots warning-free, both bots join opped, **and the block composition below
      the platform is confirmed empirically** (settles the void-vs-ground question). (TC17)
- [ ] **AC18** — Opponent health is stationary across episodes: `naturalRegeneration` is off and
      food/saturation are restored at reset, so the dummy's health is identical at every episode
      start over ≥20 consecutive episodes. (TC18, live)
- [ ] **AC19** — Follow-up issues filed: Folia; `c_aim`/`c_step`; `wall_distances` in observation;
      M3 scripted bot. (review)

---

## Error Handling

- **Dummy `health` undefined or non-finite** (bot not yet spawned): record nothing, leave the
  baseline untouched. Never record a phantom drop from an unpopulated value — that is precisely
  the bug class this plan exists to fix.
- **Health increase** (heal, respawn, reset): re-seed the baseline, record zero, never negative.
- **Reset-generated health events:** `handleReset` heals the dummy asynchronously and forces
  `_prevOpponentHealth = 20` while gating only the learner. A first hit against a not-yet-healed
  dummy would produce phantom damage. **T2 must add a dummy read-back gate** (health and position)
  and seed `_prevOpponentHealth` from the confirmed read-back, discarding reset-generated events
  before acknowledging the reset.
- **Pad bridge dies:** restart that bridge only, re-run `arena:reset_pad` for its anchor, resume.
- **Paper JVM dies** (detected by `jvm_alive()` port probe): abort the whole run loudly, naming
  the JVM. Never continue on survivors.
- **Join storm:** `bukkit.yml` sets `connection-throttle: -1`; joins are additionally staggered,
  with per-bot retry and backoff before failing the pad.
- **`max-players` exceeded:** fail loudly at launch with the computed requirement (`2N + 10`).
- **Malformed `--pad-origin`:** fail at startup with the offending string, never default silently.
- **Java 26 incompatibility:** install Temurin 21, pin `JAVA_HOME` in `start.sh`.
- **Combat probe fails (AC8):** stop. Do not proceed to topology work on an unproven repair.

---

## Testing Strategy

| ID | Test Case | Type | Expected Behavior |
|-----|-----------|------|-------------------|
| TC0 | Merged tree runs both suites | Integration | Both green at the count observed in T1 |
| TC1 | Fake dummy *bot* health 20→14, emits `'health'` | Unit (Node) | `damage_dealt == 6` via the real handler |
| TC2 | `entityHurt` fired with a health-bearing entity | Unit (Node) | Records nothing |
| TC3 | Dummy `'death'`; then `drain()`; then another window | Unit (Node) | `opponent_died` true once; latch clears on drain |
| TC4 | health `undefined`; NaN; increase; double-`wireDamageEvents`; late reset event | Unit (Node) | Recorder spy shows exact call counts; no phantom, no double-count |
| TC5 | Terminal config invariants | Unit (Py) | Ordering, signs, finiteness all enforced; bad configs raise |
| TC6 | 400 RETREAT in each of 4 directions, plus 4 diagonal corner runs | Live | Bot stays within exact x/z bounds; wall continuity and height verified |
| TC7 | Combat probe, ≥10 cycles, regen off | Live | Per-hit `6,6,6,2`; cumulative exactly 20; one death; clean post-respawn baseline; reconciles with wire health; ±1 window |
| TC8 | Windows archive parsed | Analysis | All four predictions confirmed or refuted; skipped artifacts listed |
| TC9 | `eval.run_random --episodes 20` | Live | 0 crashes, nonzero mean `r_damage_dealt` |
| TC10 | N=1 on the new topology | Integration | Byte-identical ports/usernames/coords; existing tests green |
| TC11 | Fleet boot at N=8 | Live | 16 bots joined, opped, staggered; no throttle rejections; all gates pass |
| TC12 | Cross-pad isolation, N≥8, ≥10 min | Live | Per-pad damage reconciles; zero foreign usernames in `bot.entities` |
| TC13 | Ladder 1/2/4/8/12/16/20/25 | Live | All five AC14 metrics recorded per rung |
| TC14 | Kill one bridge; separately kill the JVM | Live | Bridge: that pad only. JVM: loud abort |
| TC15 | M2 re-baseline, 3 pinned seeds | Live | Greedy win ≥95%/100 eps per seed; aim-while-invisible == 0 |
| TC16 | Damage-farm watch | Analysis | With regen off, any episode with `damage_dealt > 20` is a defect, not noise |
| TC17 | Mac bring-up + sub-platform block probe | Live | Python/torch/Java/Paper/bots verified; block at y≈−60 identified |
| TC18 | 20 consecutive episode starts | Live | Dummy health and food identical at every start |
| TC19 | `max-players` guard; malformed `--pad-origin` | Unit | Both fail loudly with the offending value |

**Test data:** Node tests use a fake dummy *bot* (EventEmitter with mutable `health`) and a spy on
`recordDamageDealt` asserting call counts. **Fakes must not carry fields mineflayer does not
populate** — that is what shipped this bug.

---

## Tasks

| ID | Task | Blocked By | Risk | Files | Description |
|----|------|------------|------|-------|-------------|
| T1 | Merge PR #21 to main | — | med | *(git)* | Fast-forward merge (21 commits, 0 conflicts, no review comments). Do not squash. Record the observed suite counts. Satisfies AC1. |
| T2 | Repair the damage channel | T1 | xHigh | `bridge/bot.js` | Add `dummy.on('health')` → `_onOpponentHealth()` mirroring `_onSelfHealth`. **Delete** `_onEntityHurt`'s recording body and registration. Add a dummy read-back gate (health + position) and seed `_prevOpponentHealth` from it, discarding reset-generated events before the reset ack. Ignore undefined/non-finite health. Keep idempotent re-wiring and `dummy.on('death')`. Satisfies AC2–AC5. **xHigh: this is the defect the whole plan exists to fix, and a silent regression is undetectable from outside.** |
| T3 | Rewrite the damage test suite | T2 | med | `bridge/bot.test.js`, `bridge/actions.test.js` | Add TC1–TC4 with a faithful fake dummy bot and a recorder spy. **Rewrite or delete the existing `actions.test.js` cases at ~587, ~682-693, ~721-738 that emit `entityHurt` with a health-bearing entity** — T2 makes them fail, and they are the mocks that hid the bug. Satisfies AC2–AC5. |
| T4 | Repair terminal reward ordering | T1 | med | `agent/reward_config.py`, `tests/test_reward.py` | Set win 50 / timeout −15 / loss 30. Add `__post_init__` enforcing ordering, sign semantics, and finiteness. Rewrite the docstring rationale — the "timeout is the worst outcome" reasoning is superseded. Satisfies AC6, TC5. |
| T5 | Analyze the Windows runs/ archive | **user supplies archive** | low | *(scratchpad)* | **Blocked until the user copies the archive in — it is not on this machine.** Inventory every artifact type first (nested runs, configs, checkpoints, console logs), not just `metrics.jsonl`. Report all four predictions in the Falsifiable Prediction section; list unreadable files. **Halt on any nonzero `r_damage_dealt`.** Satisfies AC9. |
| T6 | Enclose the arena, macro-parameterize | T1 | high | `server/arena/data/arena/function/*.mcfunction` | Convert `setup`/`reset` into **macro functions** `arena:setup_pad {x,z}` and `arena:reset_pad {x,z,learner,dummy}`. Pin every command: `$`-prefixed macro lines, `~`-relative or `$(x)`-composed coordinates, distance-only selectors (selector `x/y/z` cannot be relative). Add a **closed bedrock perimeter** `y=64..71` including all four corners. Keep global gamerules in a one-shot `arena:setup`. Satisfies AC7. |
| T7 | Mac bring-up + world verification | — | med | `server/setup/setup.sh`, *(env)* | Create a venv (system Python is 3.9.6; project needs ≥3.11), install `-e .` + torch. Run `setup.sh` on macOS. **Write `bukkit.yml` with `connection-throttle: -1`** from `setup.sh` (note it is regenerated, like `server.properties`). Set `naturalRegeneration false`. Boot Paper on Java 26; fall back to Temurin 21 and pin `JAVA_HOME` if it refuses. **Probe the block composition below y=62 and record it** — this settles whether an edge-walk is a death or a stranding, and T5's interpretation depends on it. Satisfies AC17, AC18 (partial). |
| T8 | Deterministic combat probe | T2, T6, T7 | xHigh | `eval/combat_probe.py` *(new)* | Face the dummy, issue fully-cooled ATTACKs, assert the `6,6,6,2` sequence, cumulative 20, one death, clean post-respawn baseline, ±1-window attribution, and reconciliation against `state.opponent.health`. `--cycles N`, default 10. **This is the go/no-go gate.** Satisfies AC8. **xHigh: it is the sole evidence the repair works; a probe that passes wrongly sends the whole fleet down a bad path.** |
| T9 | Pad-aware bridge configuration | T2 | med | `bridge/run.js`, `bridge/bot.js` | Add `--pad-origin "x,z"` and `--pad-index i` (mirroring existing port/username parsing); fail loudly on malformed input. Derive `resetTemplate` from the anchor. Issue `/function arena:setup_pad` and `/spawnpoint` **as the opped bot** — the bridge is the sole command channel. Default `0,0` keeps N=1 byte-identical. Satisfies AC11, TC19. |
| T10 | Bash pad-fleet launcher | T6, T7, T8, T9 | high | `server/setup/start-pads.sh` *(new)*, `distributed/launcher.py`, `server/setup/setup.sh` | One server root, one JVM. Implement `padAnchor(i)` as the **sole** coordinate source. Raise `max-players` to `2N + 10` **in `setup.sh`** (regenerated file). Write `ops.json` for all `learner_i`/`dummy_i`. Launch N bridges on `5555+i` with staggered joins, each passed its anchor. Delete `start-arenas.ps1` and the N-JVM path. Satisfies AC12. |
| T11 | Two-tier fault policy | T10 | high | `distributed/actor.py`, `agent/train.py`, `agent/train_config.py`, `tests/test_actor_pool.py`, `tests/test_learner_loop.py`, `tests/test_benchmark_overlap.py` | Implement `restart_bridge(pad_index)` and `jvm_alive()`. **Delete `fault_min_live_arenas` entirely** — the field (`train_config.py:203`), its validation (`:293-295`), the abort-floor logic in `ActorPool`, and the live f-string fragment at `train.py:1839` which raises `AttributeError` if the field is dropped. Update all **17 references** across the three test files. Satisfies AC15. |
| T12 | Cross-pad isolation verification | T10 | med | `eval/benchmark.py` | Per-pad reconciliation of cumulative `damage_dealt` against per-pad dummy health loss, plus consumption of the bridge-side foreign-username scan (emitted by T9's logging, not the frozen wire). Satisfies AC13. |
| T13 | Scale ladder measurement | T10, T11, T12 | med | `eval/benchmark.py`, `RUNBOOK.md` | Run the ladder; record all five AC14 metrics per rung (identify the Paper PID for RSS/GC). Promoted N = largest rung passing all. Document the measured table. Satisfies AC14. |
| T14 | M2 re-baseline on the fleet | T8, T13 | med | *(run)* | Train at the promoted N across 3 pinned seeds and a pinned step budget. Evaluate the fixed final checkpoint separately. Watch `r_damage_dealt` and TC16 from episode 1. Satisfies AC16. |
| T15 | File follow-up issues | T1 | low | *(GitHub)* | Folia; `c_aim` > `c_step`; `wall_distances` in observation; M3 scripted bot. Satisfies AC19. |
| T16 | Update docs | T13, T14 | low | `README.md`, `RUNBOOK.md`, `server/README.md` | Replace the N-JVM/PowerShell procedure with the macOS pad-fleet procedure. Record the measured ladder. Remove the stale "Don't run this until parallel arena is finished" note. Document the new geometry and the regen/hunger change. |

**Parallelism:** T7 starts immediately alongside T1; T5 starts whenever the user supplies the
archive. T2, T4, T6 fan out from T1 with no file overlap. T3 and T9 both wait on T2. **T8 gates
all topology work** — T10 and everything downstream are blocked on it, so the graph now matches
the prose.

---

## Notes for Implementer

- **T2 is the whole point.** Read `_onSelfHealth` first and mirror it; the asymmetry between the
  two handlers is what caused this bug.
- **Never derive another bot's internal state from the entity view.** Health, hunger, XP, effects
  come only from that bot's own connection. Position, yaw, velocity, equipment are fine from the
  entity view.
- **When writing a fake, check the upstream source for the field first.** A mock more capable than
  reality tests nothing. This is the actual lesson of this bug.
- **The frozen contract is PR-gated:** `env/observation_spec.py`, `bridge/schema.json` +
  `messages.py`, `agent/actions.py`, `compute_reward` signature. This plan touches none. T4
  changes reward *values*, not the signature — allowed.
- **`server.properties` and `bukkit.yml` are regenerated by `setup.sh`.** Any change must go into
  the script or it is silently overwritten on the next setup run.
- **`doImmediateRespawn` is true with a shared world spawn.** Without per-bot `/spawnpoint`, every
  death teleports the bot to pad 0 — a cross-pad contamination event.
- **Selector `x/y/z` arguments cannot be relative.** Use `distance=` from the execution position.
- **Rollback:** T2 is a small diff in one function. If T8 fails in a way implicating the fix rather
  than the environment, revert T2 and re-run the probe against the old path to confirm the delta.
- Run order for anything live is always **Paper → bridges → Python driver**.
- Git: feature branches only. No AI attribution in commits or PRs — run `git log -20 --oneline`
  and match the repo's voice.
