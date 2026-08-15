# Demo-Day Push: Scripted Opponent + Human Exhibition Mode

**Base:** `origin/main` @ `4fcf93e` (PR #32, "repair the damage channel and rebuild the arena as pads"). Branch `feat/demo-turn-actions-scripted-opponent` is rebased onto it. **All line citations below are post-#32 coordinates** — an earlier draft cited a stale pre-#32 `main` and every number was wrong. `feat/damage-channel-fix-and-pad-topology` is merged and dead; do not branch from it. The branch name still says "turn-actions" although this plan *cuts* them — noted in the PR body rather than renamed mid-flight.

**Goal:** By 2026-08-20, classmates can join over LAN and play a visibly competent RL agent one-on-one, launched with a single command.

**Approach:** KEEP-8 DEMO-FIRST. Do **not** touch the frozen 8-action `Macro` contract, so the existing trained checkpoint stays loadable and a working demo exists from day 1. **This cuts issue #23 (turn macros), which the user explicitly asked to do first** — see Decisions for the verified reason and the disclosure obligation it creates. Build the never-yet-exercised human-exhibition wiring first and rehearse it against a real person immediately; then land the scripted opponent (#5) and the opponent-acts wire path (#6) and warm-start a retrain against a moving opponent, so training can only improve a demo that already works.

---

## Scope

- **In scope:**
  - Bridge opponent-source seam so the "opponent" can be either the dummy bot or a human player entity
  - Human win/loss detection that does not depend on the opponent bot's own connection
  - Exhibition mode: no episode timeout vs a human, no auto-reset after death, one challenger at a time
  - `opponents/scripted_bot.py` — Stage-1 heuristic opponent (#5), EASY/HARD presets, omniscient vision, deterministic given a seed
  - Opponent-acts path (#6): optional `opp_action` on the wire; bridge drives the 2nd bot from a Python policy
  - Per-opponent knockback immunity toggle (the datapack currently hard-pins it on)
  - Curriculum as a **win-rate-gated EASY/HARD mixture** (not a one-way promotion cliff)
  - Warm-start retrain from the existing checkpoint; checkpoint selection by scripted win-rate
  - Pad-count stress sweep; training constants derived from its measurement
  - Exploration fixes: ε schedule resized for multi-arena, Ape-X per-actor ε (#15)
  - Exhibition-only reflex shield
  - Java 21 pin in `server/setup/start.sh`
  - `deploy/exhibition.py` one-command launcher; a **separate** reset command
  - Docs: `README.md`, `RUNBOOK.md`, new demo-day guide

- **Out of scope:**
  - **#23 turn macros (`FACE_LEFT`/`FACE_RIGHT`/`FACE_BACK`) — explicitly requested by the user, deliberately cut.** See Decisions. `N_ACTIONS` stays 8.
  - **Resolving `TODO(T12)` in `bridge/bot.js:_updateLastSeen()`.** FROZEN through 2026-08-20 — see Contracts.
  - The formal M3 number (#7, ≥70% win / 200 eps). We select a checkpoint by win-rate; we do not certify the milestone.
  - IRB / human-subject study (`study/booth_app.py` stays a stub)
  - Cross-machine distributed training (#22), PPO/SAC/Rainbow (#18–#20)
  - Spectator UI, leaderboards, match history
  - **Changing `ACTION_REPEAT` (the 200 ms decision interval).** A post-demo experiment. Human *reaction* latency is ~200 ms so the agent is near parity; what it lacks is sub-window control granularity, partly offset by the assisted turn. Dropping to 50 ms needs `seq_len` 16→~64 to preserve the 3.2 s memory horizon (~4× learner compute), `MAX_EPISODE_STEPS` 400→1600, and a 4× message rate whose sustainability is **unmeasured** (p99 step time ~208 ms today, i.e. paced by the game tick). It also behaviorally invalidates the checkpoint.

---

## Decisions

- **Keep `N_ACTIONS == 8`; cut the #23 expansion the user asked for first** — Verification killed the premise. `_updateLastSeen()` (`bridge/bot.js:1740-1761`) writes the opponent's **live** position unconditionally, and `handleStep` calls it after every window (`bot.js:1709`). Its own `TODO(T12)` says so: "stores the opponent's LIVE position unconditionally (perfect tracking)". `_turnToLastSeen` no-ops only on null memory, i.e. step 1 after reset. So `TURN_TO_LAST_SEEN` is an omniscient ~200 ms-stale aim-snap from step 2 onward. The agent is blind in its **observation** (correctly gated by `env/perception_filter.py`) but not in its **actions**; new facing macros would be strictly dominated by action 7. Independently re-verified by an adversarial reviewer against post-#32 code.
- **Ship the assisted turn, and disclose it** — Action 7 aims at the opponent's live position regardless of line of sight, so the agent's turn is assisted in a way a human's is not. Two alternatives were offered and the user chose this one:
  - *Scan fallback* — gate the memory to the genuinely last-seen position and make action 7 rotate ~45° when stale. Honest, keeps `N_ACTIONS == 8`. Rejected on time.
  - *Full honest version* — resolve `TODO(T12)` plus real facing macros. Correct, but invalidates checkpoints and leaves one training run with no fallback.

  **Chosen deliberately, not by omission.** The obligation is disclosure (AC14). The agent's *observation* remains honestly gated; only its turning is assisted. Revisit with the scan fallback immediately after the demo, alongside #23.
- **Demo-first, not train-first** — The human exhibition path is the only deliverable the classroom sees and the only code training never exercises. Building it day 1 converts "one retrain must succeed" into "the demo works on day 1 and training only improves it," and yields two overnight runs instead of one.
- **Warm-start the retrain** — `N_ACTIONS` unchanged, so initialize from the existing checkpoint. Requires lowering `eps_start` (T13/T16) or the warm start is thrown away.
- **Curriculum = gated mixture, not a promotion cliff** — Sample EASY/HARD per episode, shifting 80/20 → 20/80 once rolling win-rate vs EASY clears the gate. Cannot stall, cannot collapse.
- **Scripted bot sees raw truth (omniscient)** — No FOV/LoS/memory gating on its own input. It is a training target, not a fair rival.
- **Human replaces the opponent bot** — Reuses the existing two-combatant arena instead of modelling a third entity the observation has no slot for.
- **No algorithm change** — The stack is already dueling + double + n-step(3) + PER + recurrent. The agent is weak because it has only ever fought a stationary dummy under a reward changed in `8b4c151` and never retrained; that is what T9-T13 fix. PPO/SAC would discard sample efficiency for no gain on a discrete space.
- **Declared cut line** — If the schedule slips, cut in this order: **(1) T7 reflex shield, (2) T16(f) per-actor ε, (3) the HARD tier (train EASY-only and keep HARD as a demo knob).** T8 must land day 1 regardless; it is the plan's risk-retirement step.

---

## Acceptance Criteria

- [ ] **AC1** A human joins over LAN, and the agent visibly tracks them, closes distance, and lands hits — including when the human circles behind it.
- [ ] **AC2** `ATTACK` and `TURN_TO_LAST_SEEN` work against a **human player entity**, not only the dummy bot (they must not silently no-op).
- [ ] **AC3** A human death ends the match and is reported as an agent win, without depending on an opponent bot connection.
- [ ] **AC4** No episode timeout while a human is the opponent; after a death the match does **not** auto-restart.
- [ ] **AC5** One command starts Paper, the bridge, and the agent playing greedily from a checkpoint; a **separate** command resets for the next challenger. Refusal paths (port taken, checkpoint missing) exit with an actionable message.
- [ ] **AC6** `server/setup/start.sh` runs Paper on Java 21 even when `java` on PATH is 26, and the server survives >5 minutes.
- [ ] **AC7** `ScriptedBot.act()` is deterministic given a seed: same seed + same fixture sequence ⇒ identical `Macro` sequence.
- [ ] **AC8** Every branch of the spec-7.2 ladder (flee / attack / approach-strafe-jump / search) is unit-tested on fixtures.
- [ ] **AC9** The training loop steps a Python opponent policy and sends `opp_action`; the M2 stationary-dummy path is unchanged and its tests still pass.
- [ ] **AC10** The EASY/HARD mixture shifts on the win-rate gate, and a run where the gate never fires still trains to completion.
- [ ] **AC11** A guard test fails if `_updateLastSeen()` is made visibility-gated before the demo, and the two comments that falsely assert memory-gating are corrected.
- [ ] **AC12** Full offline suite green: `pytest` ≥ **647** passed, 2 deselected (the branch baseline post-#32; an earlier draft said 450, measured before the rebase), **and** the bridge suite reports **143** tests (see Testing Strategy — the naive command runs 1).
- [ ] **AC13** `README.md`, `RUNBOOK.md`, and the demo-day guide describe the one-command flow accurately on macOS.
- [ ] **AC14** The demo-day guide and `README.md` state plainly that the agent's turn is assisted, that its observation is nonetheless honestly FOV/LoS-gated, and that this is a placeholder scheduled for removal. Written to be said out loud to a classroom, not buried in a footnote.
- [ ] **AC15** ε does not reach its floor before ~15% of the run's episodes at the chosen pad count, and the arenas explore at distinct rates rather than one shared schedule.
- [x] **AC16 — MEASURED 2026-08-16. Chosen pad count: 25.**

  | N | aggregate/s | per-arena/s | p99 ms | min TPS | ≥19 gate |
  |---|---|---|---|---|---|
  | 16 | 78.04 | 4.8772 | 208.7 | 19.00 | pass |
  | 20 | 97.57 | 4.8787 | 208.7 | 19.00 | pass |
  | 24 | 117.06 | 4.8776 | 208.8 | 19.00 | pass |
  | 25 | 121.91 | 4.8763 | 208.9 | 19.00 | pass |

  **600s confirm at N=25: 121.95/s aggregate, 4.8782 per-arena, p99 208.8 ms, max 305.9 ms, 73,197 transitions, gate passes.** Throughput came out marginally *higher* over the longer window than the 240s rung, so there is no sustained-load or thermal decay.

  Scaling is linear with no knee: 16→25 is 1.5625× the arenas for 1.562× the throughput, per-arena rate varies 0.05% across the whole range, and p99 moves 0.2 ms. **No ceiling was found** — the binding constraint at 25 is `max-players=60` (`REQUIRED_PLAYERS = 2N + 10` = exactly 60), not the machine. Going beyond 25 needs a `setup.sh` re-run, not more hardware.

  **The ≥19 TPS gate now PASSES at every rung**, where issue #4's sweep reported `max_arenas_sustaining_tps: 0`. That artifact appears to have been fixed by `19fd8e5` ("measure real server TPS from world age, not the physics timer"), which is on this branch. The old advice to disregard the gate as a minimum-over-run artifact no longer applies — it can be read at face value.

  **Budget consequence:** 121.95/s ≈ **10.5M transitions/day**, up 55% from the 6.8M the plan was sized against.

  **Operational note:** `start-pads.sh` rewrites the git-tracked `server/ops.json` for all 2N bots on every fleet boot (issue #29). Restore it (`git checkout -- server/ops.json`) after any sweep, or `tests/test_pad_launcher.py::TestOpsJson::test_one_pad_is_byte_identical_to_the_committed_file` fails and looks like a code regression. The test suite itself does **not** touch the file — verified.

- [ ] ~~**AC16** The pad ceiling is a measured number:~~ transitions/s and p99 vs pad count at N = 16/20/24/25, a 600 s confirm at the chosen N, Paper console TPS flat at 20.0 there, and learner-limited throughput reported alongside collector throughput.
- [ ] **AC17** Training constants are derived from AC16 and the smoke run's *measured* mean episode length — not the 400-step ceiling — and `eps_start` is lowered when `warm_start` is set.
- [ ] **AC18** The scripted opponent takes knockback (its `knockback_immune=False` reaches the server), verified by hitting it and observing displacement.

---

## Contracts & Interfaces

### FROZEN — do not change during this plan

| Contract | Where | Rule |
|---|---|---|
| `Macro` / `N_ACTIONS == 8` | `agent/actions.py` | No members added or renumbered. `tests/test_actions.py:45` stays `== 8`. |
| Wire `action` range `0..7` | `bridge/schema.md`, `bridge/schema.json`, `bridge/messages.py` | Unchanged. |
| Observation spec (`OBS_DIM == 23`) | `env/observation_spec.py` | Unchanged — no new slots. |
| Reward components | `agent/reward_config.py` | Unchanged (`8b4c151` ships as-is). |
| **`TODO(T12)` un-gated memory** | `bridge/bot.js:1740-1761` | **MUST stay unconditional through 2026-08-20.** Gating it removes the agent's only re-acquire ability. T14 guard-tests it. |

#### Known contract violation: `TURN_TO_LAST_SEEN` does not turn to the last *seen* position

The macro's name states the **intended** contract; the implementation does not honor it. Two comments assert the memory-gating that does not exist, which is why this went unnoticed:

1. `agent/actions.py` `MACRO_SEMANTICS[Macro.TURN_TO_LAST_SEEN]` — "the last-known opponent position stored in bridge memory";
2. `bridge/bot.js:838-839` — "Updated from perception when the opponent is visible; null until the opponent has been seen at least once this episode" — contradicted by the implementation ~900 lines below at `bot.js:1740-1761`.

(The macro *name* is a third assertion but is not a comment.) **Decision: do NOT rename the macro** — the name describes the contract we intend to honor once `TODO(T12)` is resolved; renaming to match a placeholder bug means renaming back later and churns the frozen `agent/actions.py`. T14 corrects both comments instead.

### The three forms of the wire contract

`bridge/messages.py` states the contract exists in **three mutually-consistent forms** — `bridge/schema.md` (prose), `bridge/schema.json` (canonical, what `transport.validateOutbound` enforces), and `bridge/messages.py` (Python). **Change one, change all three.** Any task touching the wire lists all three plus `bridge/transport.js`.

### New — `opp_action` wire field (owner: T11a; consumers: T11b, T12)

```
Python → Node   step   { "type":"step", "action": <int 0..7>, "opp_action": <int 0..7 | null> }
```

- Absent or `null` ⇒ opponent takes no action (today's M2 path, byte-identical)
- Present ⇒ bridge runs it through a second `MacroExecutor` bound to the opponent handle
- Same validation as `action`: integer, `0 <= v < N_ACTIONS`
- **`schema.json` is `additionalProperties:false`** — the field must be added there or a present `opp_action` is rejected at runtime and the "backwards compatible" claim fails exactly when it matters

### New — env API seam (owner: T11a; consumers: T3, T12)

Pinned here because T11a and T12 otherwise meet at an unpinned seam:

```python
MCPvPEnv.step(action: int, opp_action: int | None = None) -> tuple[obs, reward, done, info]
MCPvPEnv.raw_opponent_view() -> OpponentView   # ungated raw state for the scripted opponent
```

`env.step()` returns a **gated** observation; the scripted bot is omniscient and cannot be built from it. `raw_opponent_view()` is the only sanctioned raw-state accessor and must never be routed into the agent's observation.

**`no_timeout` mechanism:** `MCPvPEnv` raises on `max_episode_steps <= 0`, so "disabled" is expressed as `max_episode_steps=None` meaning no truncation. Three consumers (T3, T5, TC16) use exactly this — no sentinels, no large integers.

### New — bridge opponent handle (owner: T1; consumers: T2, T3, T11b)

`this.dummy` is read at ~20 sites in `bridge/bot.js`. Route every **behavioral** read through one accessor:

```js
_opponentHandle()  // -> { entity, isBot, username, healthSource } | null
```

Behavioral sites that MUST be converted:
- `_opponentEntity()` (`bot.js:1732-1734`) — ATTACK's target
- `_updateLastSeen()` (`bot.js:1751`) — `TURN_TO_LAST_SEEN`'s memory
- `_snapshotOpponent()` (`bot.js:1780-1781`) — the observation source
- opponent health reads (`bot.js:974-975`, `1060-1061`) — see below
- opponent listeners (`bot.js:998-1003` on; `953-961` off)

Lifecycle sites (`872` spawn, `1140` role dispatch, `1340`/`1360` reset readback, `1846-1847` quit) stay bot-specific and must **no-op** when the opponent is a human.

**Damage channel caveat:** PR #32 repaired `damage_dealt` by reading the opponent's health from *its own* mineflayer connection (`dummy.on('health')`, `bot.js:1002`) — because mineflayer never populates `entity.health` for non-self players. In exhibition mode there is no such connection, so `damage_dealt` against a human has no source. This is accepted: the demo runs greedy with no learning, so reward is unused. Only **win detection** matters, and it comes from the scoreboard (below).

### New — human death detection (owner: T2)

**Two commands are required, not one.** Both ship together:

```
/scoreboard objectives add rl_deaths deathCount
/scoreboard objectives setdisplay list rl_deaths      <-- MANDATORY, not cosmetic
```

Read the objective from **raw client packets** on the learner's connection — `scoreboard_score`, `reset_score`, `scoreboard_objective`, `scoreboard_display_objective` — not from mineflayer's scoreboard plugin, and never from chat scraping. An increment on the challenger's entry ⇒ `opponent_died`.

**Three corrections to an earlier draft of this section, each verified at primary source during T2 (do not "clean up" back toward the old form):**

1. `bot.on('scoreboardUpdate')` **does not exist** — the mineflayer plugin emits `scoreUpdated`.
2. `scoreUpdated` is **dead on 1.21.1**: the plugin gates the score path on `packet.action === 0` (`node_modules/mineflayer/lib/plugins/scoreboard.js:41-46`), but the 1.21.1 `scoreboard_score` packet has no `action` field — it is `{itemName, scoreName, value, display_name, number_format, styling}`, the field having been split into a separate `reset_score` packet in 1.20.3. The branch never runs, so the event never fires and `itemsMap` is never populated by score updates.
3. **`objectives add` alone broadcasts nothing.** Decompile **`server/versions/1.21.1/paper-1.21.1.jar`** — NOT `server/paper-1.21.1-133.jar`, which is the Paperclip *bundler* and contains **zero** `net/minecraft` classes (verified: `unzip -l | grep -c net/minecraft` returns 0). Anyone pointed at the bundler gets nothing and may wrongly conclude the claim is unverifiable. In the real jar, `ServerScoreboard.onScoreChanged` sends `ClientboundSetScorePacket` only inside `if (trackedObjectives.contains(objective))`, and the jar's sole caller of `startTrackingObjective` is `setDisplayObjective`. Without the `setdisplay`, the server counts deaths correctly and tells the bridge **nothing, with no error anywhere** — the project's signature silent-failure mode. Side effect for T15's docs: `rl_deaths` is visible in the tab player list during exhibitions.

**Entity-gone is deliberately NOT used, even as a secondary signal** — it cannot distinguish a leaver from a death, and TC22 requires a mid-match leaver to hold IDLE. Using it would fabricate wins from disconnects.

**What no fake can prove (for T8's live rehearsal):** that a live Paper actually broadcasts these packets to the learner's client. The live check is cheap — on first exhibition boot, the **absence** of `[bridge] pad N rl_deaths objective NOT confirmed` is the read-back confirming, and one real kill of a test human confirms the increment path end to end.

**Status reporting:** the `state` message is `additionalProperties:false` on both validators and the env blocks on exactly one `state` per `step`. There is no wire slot for a status string. "Waiting for challenger" therefore means: **state keeps flowing with a zeroed opponent block; the status goes to the bridge log only.**

### New — `OpponentView` (owner: T11a for the accessor, T9 for the type; consumers: T10, T12)

```python
@dataclass(frozen=True)
class OpponentView:
    self_pos: tuple[float, float, float]
    self_yaw: float
    self_health: float
    target_pos: tuple[float, float, float]
    target_yaw: float
    target_health: float
    distance: float
    in_attack_range: bool
    attack_cooldown: float     # 0.0..1.0, 1.0 == fully charged
    can_see_target: bool       # always True for the omniscient bot; reserved for a future filtered mode
    last_known_target_pos: tuple[float, float, float] | None
```

**`attack_cooldown` MUST be clamped to exactly 1.0 by its producer (T11a).** `ScriptedBot` treats the swing as ready at `attack_cooldown >= 1.0 - 1e-6` — a deliberately tight epsilon, so a lenient threshold cannot reintroduce flailing. Consequence: if the shadow tracker ever yields a value a hair below 1.0 (e.g. `1.0 - 1e-5`, measured to produce `IDLE`), **the bot never attacks at all** — a total behavior failure that presents as a mysteriously passive opponent, not as an error. Values above 1.0 are safe. Clamp in `raw_opponent_view()`.

**`attack_cooldown` source (pinned — it has none today):** the wire's `state.self.attack_cooldown` (`schema.md:135`) is the *learner's*; `state.opponent` carries only pos/yaw/pitch/velocity/health, and `messages.py:_check_keys` rejects extras. The opponent's swing gate lives in T11b's second `MacroExecutor` on the Node side. **Decision: Python shadow-tracks it** — `raw_opponent_view()` derives `attack_cooldown` from ticks elapsed since the last `opp_action == ATTACK` that the bridge reported as *executed*, using the same cooldown constant. This avoids a four-form wire change. T11b must therefore report whether the opponent's swing actually fired. TC2/TC3 depend on this.

### New — `ScriptedBot` (owner: T9; consumers: T10, T12)

```python
class ScriptedBot(Opponent):
    def __init__(self, preset: ScriptedPreset = ScriptedPreset.EASY,
                 *, p_strafe: float | None = None, p_jump: float | None = None,
                 c_flee: float | None = None, seed: int | None = None) -> None: ...
    @property
    def name(self) -> str: ...          # "scripted_easy" | "scripted_hard"
    @property
    def config(self) -> OpponentConfig: ...
    def reset(self, seed: int | None = None) -> None: ...
    def act(self, observation: OpponentView) -> Macro: ...
```

| Preset | p_strafe | p_jump | c_flee | flee_health |
|---|---|---|---|---|
| EASY | 0.15 | 0.05 | 0.0 (never flees) | — |
| HARD | 0.40 | 0.20 | 1.0 | 6.0 |

`OpponentConfig` for both: `knockback_immune=False, fall_immune=True, void_immune=True, fixed_spawn=True`.

**`knockback_immune=False` has no implementation path today (owner: T11c).** The dummy's immunity is applied server-side by the tracked datapack — `server/arena/data/arena/function/spawn_dummy_pad.mcfunction:36` sets `knockback_resistance = 1.0` — and no wire field carries `OpponentConfig`. Without T11c the scripted opponent is still immune and, in the spec's own words, the fight is unreal.

### New — `TrainConfig` fields (owner: T12; consumers: T13, T16)

```python
opponent: str = "dummy"              # "dummy" | "scripted"
opponent_mix_easy: float = 0.8
opponent_mix_easy_after: float = 0.2
opponent_gate_winrate: float = 0.6
opponent_gate_window: int = 50
warm_start: str | None = None
```

**Which training loop:** the retrain uses the **multi-arena** path (`train_multi_arena`, `agent/train.py:1592`) at the pad count T17 selects — not `train_vs_dummy` (`train.py:1099`, the single-arena loop whose comment records that it never steps an opponent policy). That means per-arena `ScriptedBot` instances (each with its own RNG), `opp_action` threaded through every collector, and a **thread-safe** win-rate gate. `distributed/actor.py` and `distributed/learner.py` are therefore in T12's file list.

### New — exhibition config (owner: T3; consumers: T5, T6, T7)

```
challenger_username : str | None   # None => first non-agent player who enters the pad
no_timeout          : bool = True  # max_episode_steps=None
auto_reset          : bool = False
reflex_blind_steps  : int = 8      # exhibition-only, ~1.6s
```

### Naming

- Scoreboard objective: `rl_deaths`
- Branch: `feat/demo-turn-actions-scripted-opponent` (misnomer, see Base)
- Launcher: `python -m deploy.exhibition`; reset: `python -m deploy.exhibition --reset`

---

## Error Handling

- **Human leaves mid-match** — `_opponentHandle()` returns `null`; agent held in IDLE, zeroed opponent block keeps flowing, status to the bridge log. No termination, no crash.
- **Two people in the pad** — First matching player takes the `challenger` slot; later joiners ignored until reset. Documented as a one-challenger-at-a-time protocol.
- **Agent dies vs a human** — Match ends, result reported, **no auto-restart** (AC4). Operator runs the reset command.
- **`opp_action` for an absent opponent** — Silently ignored, never throws.
- **Java on PATH is not 21** — `start.sh` resolves Java 21 via `/usr/libexec/java_home -v 21`; if unavailable, exit with the `brew install --cask temurin@21` instruction rather than booting a JVM that SIGSEGVs 20 s later.
- **Bridge TCP already in use** — Only one client may connect (a second connect destroys the first). The launcher checks the port and refuses with a clear message.
- **Curriculum gate never fires** — Mixture stays at its initial ratio; the run completes (AC10).
- **Checkpoint missing / unloadable** — Exit with the expected path and the list of available checkpoints; never start a randomly-initialized agent for a demo.

---

## Testing Strategy

**Levels:** Unit (Python + Node), Integration (fake bridge), Manual (live rehearsal — the only way to cover AC1–AC5).

> **Run command — the naive one gives a false green.** From the repo root `node --test bridge/` runs **1 test**; `cd bridge && node --test` runs **143** (verified empirically post-rebase; an earlier draft said 100, measured on the pre-#32 checkout). Six tasks edit `bot.js`, so the wrong command would skip >99% of the suite.

```bash
.venv/bin/python -m pytest -q && (cd bridge && node --test)
```

| ID | Test Case | Type | Expected Behavior |
|-----|-----------|------|-------------------|
| TC1 | Low health + `c_flee=1.0` | Unit | Returns `Macro.RETREAT` |
| TC2 | In range, cooldown charged | Unit | Returns `Macro.ATTACK` |
| TC3 | In range, cooldown NOT charged | Unit | Does not return `ATTACK` |
| TC4 | Visible, far, `p_strafe=1.0, p_jump=0.0` | Unit | Returns `STRAFE_L` or `STRAFE_R`. **Must pass `p_jump=0.0` explicitly** — `p_strafe=1.0` alone inherits EASY's `p_jump=0.05`, sums to 1.05, and correctly raises `ValueError` |
| TC5 | Visible, far, `p_strafe=0, p_jump=0` | Unit | Returns `Macro.APPROACH` |
| TC6 | Not visible, last-known set | Unit | Returns `TURN_TO_LAST_SEEN` or `APPROACH` toward memory |
| TC7 | Same seed, same fixtures, two instances | Unit | Identical `Macro` sequences (AC7) |
| TC8 | `reset(seed)` re-seeds | Unit | Post-reset sequence repeats the original |
| TC9 | EASY vs HARD, **N=2000 samples, seed=0** | Unit | HARD strafes/jumps/flees strictly more often (fixed N and seed — do not leave statistical) |
| TC10 | `act` output domain | Unit | Every output is a valid `Macro` member |
| TC11 | `step` with `opp_action` omitted | Unit (Node) | Opponent takes no action; M2 behavior byte-identical |
| TC12 | `step` with `opp_action` out of range | Unit (Node) | Protocol error, bridge does not crash |
| TC13 | `_opponentHandle()` in exhibition mode | Unit (Node) | Resolves to the player entity; `_opponentEntity`/`_updateLastSeen`/`_snapshotOpponent` all use it (AC2) |
| TC14 | `_updateLastSeen()` still unconditional | Unit (Node) | Writes memory even when the opponent is outside the FOV cone — **fails if anyone gates it** (AC11) |
| TC15 | Human death via `rl_deaths` | Unit (Node) | Scoreboard increment ⇒ `opponent_died`, with no opponent-bot connection (AC3) |
| TC16 | Exhibition timeout | Unit | `max_episode_steps=None` ⇒ no truncation (AC4) |
| TC17 | Curriculum gate fires | Unit | Mixture shifts to `opponent_mix_easy_after` |
| TC18 | Curriculum gate never fires | Unit | Run completes at the initial ratio (AC10) |
| TC19 | Reflex shield | Unit | After `reflex_blind_steps` blind steps, action overridden to `TURN_TO_LAST_SEEN`; unchanged when visible |
| TC20 | Launcher refuses: port taken | Unit | Exits non-zero with an actionable message, starts nothing (AC5) |
| TC21 | Launcher refuses: checkpoint missing | Unit | Exits non-zero listing available checkpoints; never random-inits (AC5) |
| TC22 | Human leaves mid-match | Unit (Node) | `_opponentHandle()` null ⇒ IDLE held, zeroed opponent block, no throw |
| TC23 | `opp_action` with no opponent bot | Unit (Node) | Silently ignored, no throw |
| TC24 | `warm_start` weight load | Unit | Loaded params equal the checkpoint's exactly |
| TC25 | Checkpoint selection | Unit | Picks highest scripted win-rate, **not** the most recent |
| TC26 | `epsilon_for_episode` at the chosen pad count | Unit | ε still above floor at 15% of the planned episode budget (AC15) |
| TC27 | Per-actor ε spread | Unit | With N arenas, ε values are distinct and monotonically ordered; **arena 0 is the most exploratory** (ε_0 = ε, largest — Ape-X convention) |
| TC28 | Scripted opponent takes knockback | Manual/live | Hitting it displaces it (AC18) — the datapack cannot be trusted from logs alone |
| TC29 | Full regression | Integration | `pytest` ≥ **647** passed, 2 deselected; bridge suite reports **143** tests (AC12) |
| TC30 | Live human rehearsal | Manual | AC1–AC5 confirmed by a real person on the existing checkpoint |

**Test data:** Hand-authored `OpponentView` fixtures (no server, no numpy). Node tests use the existing fake-bot harness in `bridge/actions.test.js`. Seeded RNG for all probabilistic branches.

---

## Tasks

| ID | Task | Blocked By | Risk | Files | Description |
|----|------|-----------|------|-------|-------------|
| T4 | Pin Java 21 in start.sh | — | low | `server/setup/start.sh` | Resolve `JAVA_HOME="$(/usr/libexec/java_home -v 21)"` before exec (explicit env still wins); fail with the temurin@21 hint if absent. Fix the stale "machine has Java 25" message. Satisfies AC6. |
| T17 | Pad-count stress sweep | T4 | med | — (measurement) | Sweep `eval.benchmark --arenas` at N = 16, 20, 24, 25 (`max-players=60` allows 25; never measured), 240 s/rung, then a 600 s confirm at the winner. Record aggregate + per-arena transitions/s, p99, and Paper's **console** TPS — not `sustained_tps_min`, a minimum-over-run that reads ~1% low by construction. **Report learner-limited throughput too:** collectors outrunning the learner produce stale off-policy data, so the useful ceiling is where added pads stop increasing gradient steps on fresh data. Satisfies AC16. |
| T1 | Bridge opponent-handle seam | — | high | `bridge/bot.js` | Add `_opponentHandle()`; route the five behavioral read groups in Contracts through it. Lifecycle sites stay bot-only and no-op for humans. Do NOT change `_updateLastSeen`'s unconditional write. Satisfies AC2. |
| T2 | Human death detection | T1 | high | `bridge/bot.js` | `rl_deaths` deathCount objective read via scoreboard packet events (not chat scraping); increment ⇒ `opponent_died`. Entity-gone secondary. Keep the dummy's `death`/`health` path intact. Satisfies AC3. |
| T3 | Exhibition mode | T2 | high | `bridge/bot.js`, `bridge/run.js`, `env/mc_pvp_env.py` | Exhibition config per Contracts; `max_episode_steps=None` disables truncation; no auto-reset on death; hold IDLE with a zeroed opponent block and log-only status when no challenger. **Also owns three items T1 deliberately deferred, all verified during T1's review:** (a) **the first-claimant latch — now the highest-severity item in T3, and it MUST land before the live rehearsal (T8).** `_resolveChallengerEntity()` is stateless and with `challengerUsername=null` returns the lowest entity ID, NOT "first to enter the pad"; the plan's Error Handling promises a latch and nothing implements it. T2's review escalated the consequence: once scoreboard death detection is live, the live-resolve fallback means **a bystander in the learner's entity view who dies to anything gets credited as the agent's win** — the failure moved from "we aim at the wrong player" (visible, recoverable) to "we declare a win we did not earn" (silent, AC3-relevant). Pinning `challengerUsername` closes it today; the latch closes it properly. (b) `_resetPadCommand()` (`bot.js:1337-1345`) still passes `dummy: dummyUsername` unconditionally, so exhibition resets emit ~10 "no player found" console errors per reset — a runtime no-op, not a macro abort (the `$`-substitutions stay syntactically valid, so the function does not void), but suppress it. (c) `_scanForeignPlayers()` (`bot.js:1359`) excludes only the pad's two bots, so it logs the challenger as `foreign_players` every reset — and `eval/benchmark.py` reads that line as cross-pad contamination, so exhibition runs will look contaminated. Satisfies AC4. |
| T11a | `opp_action` wire + env seam | T3 | high | `bridge/schema.md`, `bridge/schema.json`, `bridge/messages.py`, `bridge/transport.js`, `env/mc_pvp_env.py` | Add optional `opp_action` to `step` in **all three contract forms** plus transport validation (`schema.json` is `additionalProperties:false`). Add `MCPvPEnv.step(action, opp_action=None)` and `raw_opponent_view()` per Contracts. Blocked by T3 for the shared `env/mc_pvp_env.py`. Satisfies AC9. |
| T11b | Bridge applies `opp_action` | T11a | high | `bridge/bot.js`, `bridge/actions.js` | Second `MacroExecutor` bound to the opponent handle, applied in the same window as the learner's. Report whether the opponent's swing actually fired via the new `state.opp_action_executed` (bool\|null) — absent means "assume fired", `false` suppresses the cooldown stamp. Silently ignore when no opponent bot. **Two items assigned by T11a's review:** (a) **nothing validates inbound `opp_action` on the receive path.** `handleStep` validates `msg.action` inline but has no equivalent for `opp_action`, and `transport.js`'s `validateInbound` — the one thing that would check it — is exported and tested but deliberately never called (wiring it into `_onData` would change live M2 behavior). **TC12 is not satisfied by anything that actually runs.** T11b must either call `validateInbound` in its handler or extend the inline guard; do not leave two implementations of the same rule, one enforced and one decorative. (b) Construct the opponent's `MacroExecutor` with the **default** `weaponAttackSpeedTicks` — Python's shadow tracker hard-codes `SERVER_TPS / 1.6` to match, and a non-default here drifts the two with nothing to catch it. Satisfies AC9. |
| T11c | Per-opponent knockback toggle | T11b | med | `server/arena/data/arena/function/spawn_dummy_pad.mcfunction`, `bridge/bot.js` | `spawn_dummy_pad.mcfunction:36` hard-pins `knockback_resistance = 1.0`. Make it conditional so `OpponentConfig.knockback_immune=False` reaches the server. Per project history, verify by hitting the bot and watching displacement — a clean boot log proves nothing. Satisfies AC18. |
| T14 | Guard test + truthful comments | T11c | low | `bridge/actions.test.js`, `bridge/bot.js`, `agent/actions.py` | TC14, framed as documenting a KNOWN contract violation frozen until after the demo — not as asserting correctness. Correct the two false comments (`agent/actions.py` MACRO_SEMANTICS; `bot.js:838-839`). Do NOT rename the macro. Last in the bot.js chain. Satisfies AC11. |
| T9 | ScriptedBot + OpponentView | — | med | `opponents/scripted_bot.py` | Implement `OpponentView`, `ScriptedPreset`, `ScriptedBot` per Contracts. Spec-7.2 ladder, owned seeded RNG re-seeded in `reset()`. Satisfies AC7. |
| T10 | ScriptedBot unit tests | T9 | low | `tests/test_scripted_bot.py` | TC1-TC10, mirroring the `tests/test_opponents.py` idiom. Satisfies AC8. |
| T5 | One-command launcher | T3, T4 | med | `deploy/exhibition.py` | Start Paper, wait for readiness, start the bridge, connect the agent greedily from a checkpoint. Refusal paths per TC20/TC21. Print LAN IP/port + pad coords. Satisfies AC5. |
| T6 | Separate reset command | T5 | low | `deploy/exhibition.py` | `--reset`: heal/reposition both sides, arm the next challenger. Manual only. **Interface gap found in T3's review — design for it rather than discovering it:** `formatHumanResetCommands` resets the **learner only**; there is no heal or reposition for the human side. After the human dies they respawn at full health anyway, but a match ending in the **agent's** death leaves the human carrying partial health into the next round. Also, because `handleReset` releases the challenger slot at its tail, the bridge may not know the challenger's name at reset time unless `challengerUsername` is pinned. Satisfies AC5. |
| T7 | Exhibition reflex shield | T6 | med | `deploy/exhibition.py` | After `reflex_blind_steps` consecutive blind steps, override the action with `TURN_TO_LAST_SEEN`. Never active during training. **Cut #1 if the schedule slips.** |
| T8 | Day-1 live human rehearsal | T5, T6 | **high** | — | Run the launcher; a second person joins over LAN; confirm AC1–AC5 on the **existing** checkpoint. **External dependency: needs a second human.** The plan's risk-retirement step — must land day 1. |
| T12 | Opponent stepping + curriculum | T9, T11b | high | `agent/train.py`, `agent/train_config.py`, `distributed/actor.py`, `distributed/learner.py` | TrainConfig fields; step the opponent policy per arena and thread `opp_action` through every collector on the **multi-arena** path (`train.py:1592`). Per-arena `ScriptedBot` instances; thread-safe win-rate gate. Must not stall if the gate never fires. Satisfies AC9, AC10. |
| T13 | Warm-start retrain + selection | T12 | med | `agent/train.py`, `eval/evaluate.py` | Honor `warm_start`; **pin the retrain's ε restart (~0.2-0.3), fresh replay, and target-net init from the loaded weights** — a warm start under `eps_start=1.0` is thrown away. Select the shipped checkpoint by scripted win-rate, not recency. Kick off overnight runs. |
| T16 | Exploration + constants | T13, T17 | med | `agent/train_config.py`, `agent/train.py` | Blocked by T13 (shared `agent/train.py`) and T17 (needs measured throughput). (a) `eps_decay_episodes=200` is single-arena sizing; all arenas share the GLOBAL counter (`distributed/actor.py:455`, `train.py:1829`), so at 16 arenas ε floors after ~12 episodes/arena — ~1% of a one-day run vs the field's own "~15%" guidance. Set to 15% of projected episodes, recomputed from T17 and the smoke run's **measured** mean episode length (do NOT assume 400 — that is a timeout, not a typical episode). (b) `replay_capacity` 100k → 1e6 (100k is ~21 min of data; ~200 MB at OBS_DIM=23). (c) `min_replay` 1k → ~25k. (d) `checkpoint_interval` 10k → 5k for more late candidates. (e) Keep `seq_len=16`/`burn_in=4` and `n_step=3`. (f) Ape-X per-actor ε (#15): arena *i* of *N* at ε_i = ε^(1 + i/(N-1)·α), α≈7 — **arena 0 is the most exploratory**. **Cut #2 if the schedule slips.** Satisfies AC15, AC17. |
| T15 | Docs | T8 | low | `README.md`, `RUNBOOK.md`, `docs/demo-day.md` | One-command flow on macOS, join instructions, one-challenger protocol, reset command, Java 21 note. Fix RUNBOOK's stale PowerShell/`pip install -e .`/Java-version content. **Include the assisted-turn disclosure per AC14 in plain language near the top — not a footnote.** Two operational notes from T2: (a) **prefer a pinned `challengerUsername` on demo day** rather than the `null` auto-resolve — with `null`, a bystander who dies to anything can be credited as the agent's win; (b) `rl_deaths` is visible in the tab player list during exhibitions, which is expected, not a bug; (c) the **first** reset of an exhibition logs the challenger on the `foreign_players` line, because the exclusion only covers a *claimed* name and nothing has claimed yet — expected, and `eval/benchmark.py` reads that line as contamination evidence, so don't let it alarm anyone mid-demo. Also document the one-line live health check: the **absence** of `[bridge] pad N rl_deaths objective NOT confirmed` on first exhibition boot is the read-back confirming. Satisfies AC13, AC14. |

**Parallelism.** `bridge/bot.js` chain, strictly serial: **T1 → T2 → T3 → T11b → T11c → T14** (T11a sits between T3 and T11b for `env/mc_pvp_env.py`). `agent/train.py` chain, strictly serial: **T12 → T13 → T16**. `deploy/exhibition.py` chain: **T5 → T6 → T7**. Independent starters: **T4, T9**. T17 follows T4; T10 follows T9; T8 follows T6.

---

## Notes for Implementer

- **Do not "fix" `TODO(T12)`.** The unconditional write is what gives the agent any ability to re-acquire an opponent it cannot see. T14 makes gating it fail loudly.
- **`pip install -e .` installs nothing** — `pyproject.toml` declares no dependencies. Use `requirements.txt` and the `.venv` (3.11.15); system python is 3.9.6.
- **Other players' health is invisible.** mineflayer never sets `entity.health` for non-self players; PR #32's damage fix works only because it reads the dummy's *own* connection. Human-mode logic must use the scoreboard.
- **Datapack failures are silent** — a `$`-macro or `/fill` failure voids the whole function with nothing in the log. Verify geometry and attributes by *hitting things*, never by reading the boot log (T11c depends on this).
- **The bridge accepts exactly one TCP connection.** A second connect destroys the first — never run eval tooling alongside the demo.
- **`state.opponent.health` is a live hazard in exhibition mode.** It is documented as "PRIVILEGED raw true health" (`bridge/messages.py:30`) and `_snapshotOpponent` emits `0` when a human's health is unreadable — which reads as "dead" on a channel labelled "true health". Nothing enforces that consumers ignore it. Today Python does ignore it (`env/mc_pvp_env.py:820` reads only pos/yaw/pitch/velocity; `opp_died` comes from `events.opponent_died` at `mc_pvp_env.py:635`, sourced solely from the opponent bot's own `death` event, which is never wired in human mode). **T2, T11b, and any future reward work must not start reading `state.opponent.health`** or they will fabricate kills against humans.
- **Rollback:** every change is additive and behind config (`opponent="dummy"`, exhibition off). The frozen enum means the pre-existing checkpoint always loads.
- **Freeze day (19th):** no code changes. Pick the checkpoint, rehearse, fix only what rehearsal breaks.
