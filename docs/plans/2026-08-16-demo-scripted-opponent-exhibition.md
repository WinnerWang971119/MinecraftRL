# Demo-Day Push: Scripted Opponent + Human Exhibition Mode

**Goal:** By 2026-08-20, classmates can join over LAN and play a visibly competent RL agent one-on-one, launched with a single command.

**Approach:** KEEP-8 DEMO-FIRST. Do **not** touch the frozen 8-action `Macro` contract, so the existing trained checkpoint stays loadable and a working demo exists from day 1. Build the never-yet-exercised human-exhibition wiring first and rehearse it against a real person immediately; then land the scripted opponent (#5) and the opponent-acts wire path (#6) and warm-start a retrain against a moving opponent, so training can only improve a demo that already works.

---

## Scope

- **In scope:**
  - Bridge opponent-source seam so the "opponent" can be either the dummy bot or a human player entity
  - Human win/loss detection that does not depend on `entity.health` (invisible for other players)
  - Exhibition mode: no episode timeout vs a human, no auto-reset after death, one challenger at a time
  - `opponents/scripted_bot.py` — Stage-1 heuristic opponent (#5), EASY/HARD presets, omniscient vision, deterministic given a seed
  - Opponent-acts path (#6): optional `opp_action` on the wire; bridge drives the 2nd bot from a Python policy
  - Curriculum as a **win-rate-gated EASY/HARD mixture** (not a one-way promotion cliff)
  - Warm-start retrain from the existing checkpoint; checkpoint selection by scripted win-rate
  - Exhibition-only reflex shield: force re-acquire when the agent has been blind too long
  - Java 21 pin in `server/setup/start.sh`
  - `deploy/exhibition.py` one-command launcher; a **separate** reset command
  - Docs: `README.md`, `RUNBOOK.md`, new demo-day guide

- **Out of scope:**
  - **#23 turn macros (`FACE_LEFT`/`FACE_RIGHT`/`FACE_BACK`).** Cut after verification showed the premise was false — see Decisions. `N_ACTIONS` stays 8.
  - **Resolving `TODO(T12)` in `bridge/bot.js:_updateLastSeen()`.** Explicitly FROZEN through 2026-08-20 — see Contracts.
  - The formal M3 number (#7, ≥70% win / 200 eps). We select a checkpoint by win-rate; we do not certify the milestone.
  - IRB / human-subject study (`study/booth_app.py` stays a stub)
  - Cross-machine distributed training (#22), PPO/SAC/Rainbow (#18–#20)
  - Spectator UI, leaderboards, match history

---

## Decisions

- **Keep `N_ACTIONS == 8`; drop the 11-action expansion** — The plan originally added ±90°/180° facing macros to fix "the agent can never search for an enemy behind it." Verification killed that premise: `bridge/bot.js:951-976` `_updateLastSeen()` writes the opponent's **live** position unconditionally every step (its own `TODO(T12)` says "stores the opponent's LIVE position unconditionally (perfect tracking)"), and `bot.js:925` calls it after every step window. So `TURN_TO_LAST_SEEN` is an omniscient ~200 ms-stale aim-snap from step 2 onward. The agent is blind in its **observation** (correctly gated by `env/perception_filter.py`) but not in its **actions**. New facing macros would be strictly dominated by action 7, so the enum change would break a frozen contract and invalidate every checkpoint for no measurable gain.
- **Demo-first, not train-first** — The human exhibition path is the only deliverable the classroom sees and the only code training never exercises. Building it day 1 converts "one retrain must succeed" into "the demo works on day 1 and training only improves it," and yields two overnight runs instead of one.
- **Warm-start the retrain** — Because `N_ACTIONS` is unchanged, initialize from the existing checkpoint rather than cold. Spends the limited training budget on adapting to a moving opponent instead of rediscovering locomotion.
- **Curriculum = gated mixture, not a promotion cliff** — Sample EASY/HARD per episode, shifting 80/20 → 20/80 once rolling win-rate vs EASY clears the gate. A one-way promotion has a threshold and window we have no time to tune and a collapse mode we have no time to recover from. Mixture cannot stall and cannot collapse.
- **Scripted bot sees raw truth (omniscient)** — No FOV/LoS/memory gating on the opponent's own input. It is a training target, not a fair rival; gating it would make it weaker and flakier and buys nothing before the demo.
- **Human replaces the opponent bot** — Reuses the existing two-combatant arena instead of modelling a third entity the observation has no slot for.
- **Reward/obs contracts unchanged** — The reward reshape from `8b4c151` ships as-is. Retraining against a moving opponent already co-varies two factors (opponent + reward); adding a third would make any result unattributable.

---

## Acceptance Criteria

- [ ] **AC1** A human joins over LAN, and the agent visibly tracks them, closes distance, and lands hits — including when the human circles behind it.
- [ ] **AC2** `ATTACK` and `TURN_TO_LAST_SEEN` work against a **human player entity**, not only against the dummy bot (regression: they must not silently no-op).
- [ ] **AC3** A human death ends the match and is reported as an agent win, without reading `entity.health`.
- [ ] **AC4** No episode timeout while a human is the opponent; after a death the match does **not** auto-restart.
- [ ] **AC5** One command starts Paper, the bridge, and the agent playing greedily from a checkpoint; a **separate** command resets for the next challenger.
- [ ] **AC6** `server/setup/start.sh` runs Paper on Java 21 even when `java` on PATH is 26, and the server survives >5 minutes.
- [ ] **AC7** `ScriptedBot.act()` is deterministic given a seed: same seed + same fixture sequence ⇒ identical `Macro` sequence.
- [ ] **AC8** Every branch of the spec-7.2 ladder (flee / attack / approach-strafe-jump / search) is unit-tested on fixtures.
- [ ] **AC9** The training loop steps a Python opponent policy and sends `opp_action`; the M2 stationary-dummy path is unchanged and its tests still pass.
- [ ] **AC10** The EASY/HARD mixture shifts on the win-rate gate, and a run where the gate never fires still trains to completion.
- [ ] **AC11** A guard test fails if `_updateLastSeen()` is made visibility-gated before the demo.
- [ ] **AC12** Full offline suite green (baseline: 450 passed, 2 deselected).
- [ ] **AC13** `README.md`, `RUNBOOK.md`, and the demo-day guide describe the one-command flow accurately on macOS.

---

## Contracts & Interfaces

### FROZEN — do not change during this plan

| Contract | Where | Rule |
|---|---|---|
| `Macro` / `N_ACTIONS == 8` | `agent/actions.py` | No members added or renumbered. `tests/test_actions.py:45` stays `== 8`. |
| Wire `action` range `0..7` | `bridge/schema.md`, `bridge/messages.py` | Unchanged. |
| Observation spec | `env/observation_spec.py` | Unchanged — no new slots. |
| Reward components | `agent/reward_config.py` | Unchanged (`8b4c151` ships as-is). |
| **`TODO(T12)` un-gated memory** | `bridge/bot.js:951-976` | **MUST stay unconditional through 2026-08-20.** Gating it removes the agent's only re-acquire ability and breaks the demo. T15 adds a guard test. Revisit only after the demo, together with #23. |

### New — `opp_action` wire field (owner: T11a; consumers: T11b, T12)

Extends the existing `step` message. Backwards compatible: absent or `null` ⇒ opponent idles, which is exactly today's M2 dummy behavior.

```
Python → Node   step   { "type":"step", "action": <int 0..7>, "opp_action": <int 0..7 | null> }
```

- `opp_action` absent/`null` — opponent takes no action (M2 path, unchanged)
- `opp_action` present — bridge runs it through a second `MacroExecutor` bound to the opponent handle
- Validation mirrors `action`: integer, `0 <= v < N_ACTIONS`, else protocol error

### New — bridge opponent handle (owner: T1; consumers: T2, T3, T11b)

`this.dummy` is read in ten places in `bridge/bot.js` (502, 568, 575, 620-622, 633-634, 649-651, 791-803, 949-950, 967, 997, 1074-1075). Human mode must not patch them one by one. Introduce a single accessor and route **every** behavioral read through it.

```js
// Resolves the current opponent: the dummy bot in training mode, or the
// challenger's player entity in exhibition mode. Never returns this.dummy directly.
_opponentHandle()  // -> { entity, isBot, username } | null
```

Call sites that MUST be converted (behavioral):
- `_opponentEntity()` (949-950) — ATTACK's target
- `_updateLastSeen()` (967) — `TURN_TO_LAST_SEEN`'s memory
- `_snapshotOpponent()` (997) — the observation source
- opponent health read (633-634) — see death detection below
- death listener wiring (649-651) — see death detection below

Lifecycle sites (568 spawn, 791-803 reset teleports, 1074 quit) stay bot-specific and must no-op when the opponent is a human.

### New — `OpponentView` (owner: T9; consumers: T10, T12)

The omniscient state the scripted bot reads. Plain dataclass, no numpy dependency, hand-authorable in tests.

```python
@dataclass(frozen=True)
class OpponentView:
    self_pos: tuple[float, float, float]
    self_yaw: float
    self_health: float
    target_pos: tuple[float, float, float]
    target_yaw: float
    target_health: float
    distance: float            # horizontal distance to target
    in_attack_range: bool
    attack_cooldown: float     # 0.0..1.0, 1.0 == fully charged
    can_see_target: bool       # always True for the omniscient bot; reserved for a future filtered mode
    last_known_target_pos: tuple[float, float, float] | None
```

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

`ScriptedPreset` presets (explicit constructor args override):

| Preset | p_strafe | p_jump | c_flee | flee_health |
|---|---|---|---|---|
| EASY | 0.15 | 0.05 | 0.0 (never flees) | — |
| HARD | 0.40 | 0.20 | 1.0 | 6.0 |

`OpponentConfig` for both presets: `knockback_immune=False, fall_immune=True, void_immune=True, fixed_spawn=True` — unlike the dummy it must take knockback, or the fight is unreal.

### New — `TrainConfig` fields (owner: T12; consumers: T13)

```python
opponent: str = "dummy"              # "dummy" | "scripted"
opponent_mix_easy: float = 0.8       # P(EASY) before the gate fires
opponent_mix_easy_after: float = 0.2 # P(EASY) after the gate fires
opponent_gate_winrate: float = 0.6   # rolling win-rate vs EASY that opens the gate
opponent_gate_window: int = 50       # episodes in the rolling window
warm_start: str | None = None        # checkpoint path to initialize from
```

### New — exhibition config (owner: T3; consumers: T5, T6, T7)

```
challenger_username : str | None   # None => first non-agent player who enters the pad
no_timeout          : bool = True  # env max_episode_steps effectively disabled
auto_reset          : bool = False # deaths do NOT restart the match
reflex_blind_steps  : int = 8      # exhibition-only: force TURN_TO_LAST_SEEN after N blind steps (~1.6s)
```

### Naming

- Scoreboard objective for human deaths: `rl_deaths` (`/scoreboard objectives add rl_deaths deathCount`)
- Branch: `feat/demo-turn-actions-scripted-opponent`
- Launcher entry: `python -m deploy.exhibition`; reset: `python -m deploy.exhibition --reset`

---

## Error Handling

- **Human leaves mid-match** — `_opponentHandle()` returns `null`; the bridge holds the agent in IDLE and reports "waiting for challenger" rather than erroring. No episode termination, no crash.
- **Two people walk into the pad** — First matching player wins the `challenger` slot; later joiners are ignored until reset. Documented in the demo guide as a one-challenger-at-a-time protocol.
- **Agent dies vs a human** — Match ends, result reported, **no auto-restart** (per AC4). Operator runs the reset command.
- **`opp_action` for an absent opponent** — Bridge ignores it silently (does not throw); training never sends it in dummy mode.
- **Java on PATH is not 21** — `start.sh` resolves Java 21 via `/usr/libexec/java_home -v 21`; if unavailable it exits with the `brew install --cask temurin@21` instruction rather than booting a JVM that SIGSEGVs 20 s later.
- **Bridge TCP already in use** — Only one client may connect (a second connect destroys the first). The launcher checks the port and refuses to start a second bridge with a clear message.
- **Curriculum gate never fires** — Mixture simply stays at its initial ratio; the run completes normally (AC10).
- **Checkpoint missing / unloadable** — Launcher exits with the expected path and the list of available checkpoints; it never starts a randomly-initialized agent for a demo.

---

## Testing Strategy

**Levels:** Unit (Python + Node), Integration (fake bridge), Manual (live rehearsal — the only way to cover AC1–AC5).

| ID | Test Case | Type | Expected Behavior |
|-----|-----------|------|-------------------|
| TC1 | `ScriptedBot.act` low health + `c_flee=1.0` | Unit | Returns `Macro.RETREAT` |
| TC2 | `act` in range, cooldown charged | Unit | Returns `Macro.ATTACK` |
| TC3 | `act` in range, cooldown NOT charged | Unit | Does not return `ATTACK` |
| TC4 | `act` visible, far, `p_strafe=1.0` | Unit | Returns `STRAFE_L` or `STRAFE_R` |
| TC5 | `act` visible, far, `p_strafe=0, p_jump=0` | Unit | Returns `Macro.APPROACH` |
| TC6 | `act` target not visible, last-known set | Unit | Returns `TURN_TO_LAST_SEEN` or `APPROACH` toward memory |
| TC7 | Same seed, same fixture sequence, two instances | Unit | Identical `Macro` sequences (AC7) |
| TC8 | `reset(seed)` re-seeds | Unit | Post-reset sequence repeats the original |
| TC9 | EASY vs HARD presets on one fixture set | Unit | HARD strafes/jumps/flees strictly more often over N samples |
| TC10 | `act` returns only members of `Macro` | Unit | Every output is a valid `Macro`; never an int outside 0..7 |
| TC11 | `step` with `opp_action` omitted | Unit (Node) | Opponent takes no action; M2 behavior byte-identical |
| TC12 | `step` with `opp_action` out of range | Unit (Node) | Protocol error, bridge does not crash |
| TC13 | `_opponentHandle()` in exhibition mode | Unit (Node) | Resolves to the player entity; `_opponentEntity`/`_updateLastSeen`/`_snapshotOpponent` all use it (AC2) |
| TC14 | `_updateLastSeen()` still unconditional | Unit (Node) | Writes memory even when the opponent is outside the FOV cone — **fails if anyone gates it** (AC11) |
| TC15 | Human death detection without `entity.health` | Unit (Node) | `rl_deaths` increment ⇒ `opponent_died` event (AC3) |
| TC16 | Exhibition mode timeout | Unit | Episode does not truncate at `max_episode_steps` (AC4) |
| TC17 | Curriculum gate fires | Unit | Mixture shifts to `opponent_mix_easy_after` after the window clears the gate |
| TC18 | Curriculum gate never fires | Unit | Run completes at the initial ratio (AC10) |
| TC19 | Reflex shield | Unit | After `reflex_blind_steps` with `visible==0`, action is overridden to `TURN_TO_LAST_SEEN`; unchanged when visible |
| TC20 | Full regression | Integration | `pytest` ≥ 450 passed; `node --test bridge/` green (AC12) |
| TC21 | Live human rehearsal | Manual | AC1–AC5 confirmed by a real person on the existing checkpoint |

**Test data:** Hand-authored `OpponentView` fixtures (no server, no numpy). Node tests use the existing fake-bot harness in `bridge/actions.test.js`. Seeded RNG for all probabilistic branches.

**Run command:**
```bash
.venv/bin/python -m pytest -q && node --test bridge/
```

---

## Tasks

| ID | Task | Blocked By | Risk | Files | Description |
|----|------|-----------|------|-------|-------------|
| T1 | Bridge opponent-handle seam | — | high | `bridge/bot.js` | Add `_opponentHandle()` returning `{entity,isBot,username}｜null`. Route `_opponentEntity` (949), `_updateLastSeen` (967), `_snapshotOpponent` (997) through it. Leave lifecycle sites (568/791-803/1074) bot-only, no-op for humans. Do NOT change `_updateLastSeen`'s unconditional write. Satisfies AC2. |
| T2 | Human death detection | T1 | high | `bridge/bot.js` | Register `rl_deaths` deathCount objective; detect challenger death from it plus entity-gone, not `entity.health` (mineflayer never populates it for other players). Emit the existing `opponent_died` event. Keep the dummy's `death` listener path intact. Satisfies AC3. |
| T3 | Exhibition mode in the bridge/env | T2 | high | `bridge/bot.js`, `bridge/run.js`, `env/mc_pvp_env.py` | Add exhibition config (challenger_username, no_timeout, auto_reset=False). Disable timeout truncation when set; do not auto-reset on death; hold IDLE when no challenger. Satisfies AC4. |
| T4 | Pin Java 21 in start.sh | — | low | `server/setup/start.sh` | Resolve `JAVA_HOME="$(/usr/libexec/java_home -v 21)"` before exec (explicit env still wins); fail with the temurin@21 install hint if absent. Fix the stale "machine has Java 25" message. Satisfies AC6. |
| T5 | One-command launcher | T3, T4 | med | `deploy/exhibition.py` | Start Paper, wait for readiness, start the bridge, connect the agent greedily from a checkpoint. Refuse to start if the bridge port is taken, or the checkpoint is missing (list what's available). Print LAN IP/port + pad coords. Satisfies AC5. |
| T6 | Separate reset command | T5 | low | `deploy/exhibition.py` | `--reset`: heal/reposition both sides and arm the next challenger. Manually triggered only — never automatic. Satisfies AC5. |
| T7 | Exhibition reflex shield | T5 | med | `deploy/exhibition.py` | Exhibition-only wrapper: after `reflex_blind_steps` consecutive steps with `visible==0`, override the policy action with `TURN_TO_LAST_SEEN`. Must NOT be active during training. |
| T8 | Day-1 live human rehearsal | T5, T6, T7 | med | — | Run the launcher, have a second person join over LAN, and confirm AC1–AC5 on the **existing** checkpoint. Record what breaks; this is the plan's main risk-retirement step. |
| T9 | ScriptedBot + OpponentView | — | med | `opponents/scripted_bot.py` | Implement `OpponentView`, `ScriptedPreset`, `ScriptedBot` per Contracts. Spec-7.2 ladder, owned seeded RNG re-seeded in `reset()`. Returns only `Macro` members. Satisfies AC7. |
| T10 | ScriptedBot unit tests | T9 | low | `tests/test_scripted_bot.py` | TC1-TC10 on hand-authored fixtures, mirroring the `tests/test_opponents.py` idiom. Satisfies AC8. |
| T11a | `opp_action` wire schema | — | high | `bridge/schema.md`, `bridge/messages.py`, `env/mc_pvp_env.py` | Add optional `opp_action` to `step` per Contracts. Absent/null ⇒ opponent idle (M2 unchanged). Same range validation as `action`. Satisfies AC9. |
| T11b | Bridge applies `opp_action` | T1, T11a | high | `bridge/bot.js`, `bridge/actions.js` | Second `MacroExecutor` bound to the opponent handle; apply `opp_action` in the same window as the learner's. Silently ignore when no opponent bot. Satisfies AC9. |
| T12 | Opponent stepping + curriculum | T9, T11b | high | `agent/train.py`, `agent/train_config.py` | Add the TrainConfig fields; step the opponent policy in Python and pass `opp_action` down (the loop currently never steps one — `train.py:1099`). Per-episode EASY/HARD mixture with the rolling win-rate gate; must not stall if the gate never fires. Satisfies AC9, AC10. |
| T13 | Warm-start retrain + selection | T12 | med | `agent/train.py`, `eval/evaluate.py` | Honor `warm_start` to init from the existing checkpoint. Select the shipped checkpoint by scripted win-rate, not recency. Kick off overnight runs. |
| T14 | Guard test for TODO(T12) | T1 | low | `bridge/actions.test.js` (or bot test) | TC14: assert `_updateLastSeen()` writes memory even when the opponent is out of FOV, with a comment explaining the freeze. Satisfies AC11. |
| T15 | Docs | T8 | low | `README.md`, `RUNBOOK.md`, `docs/demo-day.md` | One-command flow on macOS, join instructions, one-challenger protocol, reset command, Java 21 note. Fix RUNBOOK's stale PowerShell/`pip install -e .`/Java-version content. Satisfies AC13. |

**Parallelism:** `bridge/bot.js` is edited by T1, T2, T3, T11b — strictly serial in that order. T4, T9, T11a start immediately in parallel with T1. T10 follows T9. T14 follows T1.

---

## Notes for Implementer

- **Do not "fix" `TODO(T12)`.** `_updateLastSeen()` writing the live position unconditionally is what gives the agent any ability to re-acquire an opponent it cannot see. Gating it before the demo removes that and the agent will stare at walls. T14 exists to make this fail loudly.
- **`pip install -e .` installs nothing** — `pyproject.toml` declares no dependencies. Use `requirements.txt` and the `.venv` (Python 3.11.15); system python is 3.9.6.
- **Other players' health is invisible.** mineflayer never sets `entity.health` for anyone but the connected bot. Any human HP/damage logic must go through the scoreboard, not the entity.
- **A `/fill` or `$`-macro datapack failure is silent** — a reset ack does not prove the geometry exists. Verify the pad by hitting the opponent and watching knockback, not by reading the boot log.
- **The bridge accepts exactly one TCP connection.** A second connect destroys the first — never run eval tooling alongside the demo.
- **Attack cooldown is a real gate.** The scripted bot must respect it or it will flail; `attack_cooldown` is on `OpponentView` for exactly this.
- **Rollback:** every change is additive and behind config (`opponent="dummy"`, exhibition off). Reverting to the M2 path is a config change, not a code revert. The frozen enum means the pre-existing checkpoint always loads.
- **Freeze day (19th):** no code changes. Pick the checkpoint, rehearse the demo, fix only what rehearsal breaks.
