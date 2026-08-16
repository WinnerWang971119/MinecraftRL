# Bridge Wire Schema

**One of the four FROZEN CONTRACT artifacts (AC1).** This is the Node↔Python
message contract: the exact set of JSON messages exchanged over the bridge, in
both directions, with every field's type and meaning.

- Machine-readable, canonical form: [`schema.json`](./schema.json) (JSON Schema
  draft-07). Both sides validate against it.
- Python bindings + a dependency-free validator that mirrors `schema.json`:
  [`messages.py`](./messages.py).

These three files are **the same contract in three forms** and must stay
mutually consistent. **Changing any of them is a contract change** and requires a
PR visible to all tracks.

**Owner:** T3 (Environment/bridge track)

---

## Transport

- **Newline-delimited JSON** (JSON-lines): each message is one UTF-8 JSON object
  followed by a single `\n`. No `\n` appears inside a message.
- **Raw TCP socket**, **one connection per arena**. Multi-arena = multiple
  independent connections; there is no arena id on the wire.
- **Framing buffers partial reads across TCP packet boundaries.** TCP is a byte
  stream: a single `recv` may contain half a message, several messages, or a
  message split across packets. The reader MUST accumulate bytes and only parse
  up to each `\n`. This framing is the **Node transport's responsibility
  (T7a)**; the Python side reads already-split lines (`parse_line`).
- **Raw state.** Node emits raw state; Python owns fairness, reward, and
  learning. Mineflayer has no native synchronous `step()`, so Node aggregates
  events over the `ACTION_REPEAT` decision interval and replies once per `step`.

---

## Privileged information (READ THIS)

`state.opponent.health` is the opponent's **RAW true health**. It is on the wire
because the **reward** is allowed to use privileged information, and because
future stages need it. It **MUST NEVER reach the observation.**

- The **PerceptionFilter (T12)** gates the opponent block. The observation
  contract (`env/observation_spec.py`) has **no slot for opponent health at
  all** — it is intentionally absent.
- `messages.py` parses `opponent.health` into the inbound `StateMsg` (it is on
  the wire), but downstream code must route it **only to the reward**, never to
  the observation builder.
- The opponent's `pos`/`yaw`/`pitch`/`velocity` are also raw here and are gated
  upstream (FOV cone + raycast LoS + memory) before any of them reach the obs.

The same discipline applies to anything else privileged the bridge ever adds:
raw → reward only, gated → observation.

---

## Messages: Python → Node

### `reset`

Request a new episode: stop bots, teleport both agents to fixed spawns, restore
full health and identical gear, and clear the PerceptionFilter memory. Node
replies with a [`reset_ack`](#reset_ack) after the read-back gate.

| Field     | Type    | Meaning                                                                 |
|-----------|---------|-------------------------------------------------------------------------|
| `type`    | `"reset"` (const) | Discriminator.                                                |
| `episode` | int ≥ 0 | Monotonic episode counter; lets Node correlate the `reset_ack`.         |
| `seed`    | int     | Per-episode RNG seed (spawn jitter, gear, opponent choice); logged.     |

```json
{"type":"reset","episode":0,"seed":12345}
```

### `step`

Execute one discrete action macro for the `ACTION_REPEAT` decision interval.
Node aggregates events over the interval and replies with one
[`state`](#state) message.

| Field    | Type            | Meaning                                                                    |
|----------|-----------------|----------------------------------------------------------------------------|
| `type`   | `"step"` (const) | Discriminator.                                                            |
| `action` | int `0..7`      | Discrete action index, **must be in [0, 7]**, matching the 8 frozen action macros in `agent/actions.py`. |
| `opp_action` | int `0..7` \| `null` \| *absent* | **Optional.** The same discrete action index, for the **opponent**. Same validation as `action`: integer, `0 <= v < N_ACTIONS`. |

```json
{"type":"step","action":3}
{"type":"step","action":3,"opp_action":5}
```

**`opp_action` — the opponent-acts path.** `N_ACTIONS` is frozen at 8; this field
widens *who* acts, never the action space.

- **Absent or `null` ⇒ the opponent takes no action.** This is the M1/M2
  stationary-dummy path, and it stays byte-identical: `StepMsg` **omits** the key
  when it is `None`, so a dummy-path step line is exactly what it was before the
  field existed.
- **Present ⇒ the bridge drives the opponent handle through a second
  `MacroExecutor`** in the same decision window as the learner's. With no
  opponent bot to drive, it is **silently ignored** — never an error.
- Out of range, non-integer, or boolean ⇒ a protocol error, exactly as for
  `action`.

### `close`

End this client's session: the bridge closes this connection but keeps both
bots in-game and keeps listening — the env opens a fresh connection per
episode, and `reset` re-establishes all bot state. Bots disconnect only when
the bridge process exits. No reply.

| Field  | Type             | Meaning        |
|--------|------------------|----------------|
| `type` | `"close"` (const) | Discriminator. |

```json
{"type":"close"}
```

---

## Messages: Node → Python

### `state`

RAW aggregated state for one decision interval. Replies to a `step` (and is the
post-`reset` first observation once the episode is running).

| Field          | Type    | Meaning                                                             |
|----------------|---------|---------------------------------------------------------------------|
| `type`         | `"state"` (const) | Discriminator.                                            |
| `self`         | object  | Learner-bot raw state (FULL; always real, never gated). See below.  |
| `opponent`     | object  | Opponent RAW true state, **including privileged true health**. See below. |
| `events`       | object  | Damage/death events aggregated over the interval. See below.        |
| `arena`        | object  | Arena geometry sensed this interval. See below.                     |
| `tick`         | int ≥ 0 | Server game tick at end of the interval.                            |
| `code_version` | string  | Env+filter code-version stamp (git SHA + config hash). Kickoff **logs** a mismatch; the distributed future **rejects** it. |
| `opp_action_executed` | bool \| `null` \| *absent* | **Optional.** Did this window's `opp_action` actually take effect? See [the swing report](#the-swing-report). |

**`self`** (FULL — always real, never gated):

| Field             | Type        | Meaning                                                       |
|-------------------|-------------|---------------------------------------------------------------|
| `pos`             | `[x,y,z]` floats | World-frame position.                                    |
| `yaw`             | float       | Yaw in radians.                                               |
| `pitch`           | float       | Pitch in radians.                                             |
| `velocity`        | `[x,y,z]` floats | World-frame velocity.                                    |
| `on_ground`       | bool        | Whether the bot is on the ground.                             |
| `health`          | float       | Self current health (`0..MAX_HEALTH`). Allowed in the obs.    |
| `held_item`       | string      | Held-item identifier (e.g. `"iron_sword"`); resolved to a vocab id by `observation_spec`. |
| `attack_cooldown` | float       | Bridge-computed swing progress in `[0, 1]` (`1.0` == ready).  |

**`opponent`** (RAW; gated upstream before reaching the obs):

| Field      | Type        | Meaning                                                              |
|------------|-------------|----------------------------------------------------------------------|
| `pos`      | `[x,y,z]` floats | World-frame position. Gated upstream.                           |
| `yaw`      | float       | Yaw in radians. Gated upstream.                                      |
| `pitch`    | float       | Pitch in radians. Gated upstream.                                   |
| `velocity` | `[x,y,z]` floats | World-frame velocity. Gated upstream.                          |
| `health`   | float       | **RAW true opponent health. PRIVILEGED → reward only, NEVER the observation.** |

**`events`** (aggregated over the interval; source of the reward's damage anchors):

| Field           | Type      | Meaning                                            |
|-----------------|-----------|----------------------------------------------------|
| `damage_dealt`  | float ≥ 0 | Total damage dealt to the opponent this interval.  |
| `damage_taken`  | float ≥ 0 | Total damage taken this interval.                  |
| `i_died`        | bool      | Learner bot died this interval.                    |
| `opponent_died` | bool      | Opponent died this interval.                       |

**`arena`**:

| Field            | Type            | Meaning                                                          |
|------------------|-----------------|------------------------------------------------------------------|
| `wall_distances` | array of floats | Distances to surrounding arena walls (fixed bridge probe order). |

<a id="the-swing-report"></a>
**`opp_action_executed` — the swing report.** The opponent has **no
`attack_cooldown` channel on the wire**: `self.attack_cooldown` is the
*learner's*, and the `opponent` block carries only pos/yaw/pitch/velocity/health.
Python therefore **shadow-tracks** the opponent's swing meter
(`MCPvPEnv.raw_opponent_view()`), counting **decision windows** since the last
`opp_action == ATTACK` that actually fired — and this field is the only way it
can know whether one did.

**The tracker must NOT use `state.tick`.** That field is the *server* tick
(`bot.time.age`), which the server sends only ~once per second: it reads flat for
several states and then jumps ~20 (see `eval/benchmark.py`'s coarse-tick note).
The swing gate rides a different clock — `bot.js`'s `_currentTick`, which advances
exactly `ACTION_REPEAT` per window — and `bot.js`'s `_serverTick()` docstring states
the two are decoupled. Deriving the meter from `state.tick` makes it read ready
1-5 windows after a swing instead of 4, and then **locks into a flail**: the shadow
says ready, `canSwing` blocks, `swung=false` suppresses the stamp, the meter stays
pinned at `1.0`, and the opponent mashes ATTACK every window without ever strafing.

| Value | Meaning | Tracker behavior |
|-------|---------|------------------|
| `true` | The opponent's macro produced its side effect; for `ATTACK`, the swing really went out. | Stamp the 0-based **decision-window index** of the swing, and compute `elapsed = (windows_since_swing) * ACTION_REPEAT`. One `step()` is one window is `ACTION_REPEAT` gate ticks, so this reconstructs `_currentTick - _lastSwingTick` by construction. Never derive it from `state.tick` — see above. |
| `false` | The swing did **not** fire: gate-blocked, or no entity to swing at. | Do **not** stamp — mirrors the executor's own "a swing at nothing does not start the cooldown" rule. |
| absent / `null` | The bridge does not report (no `opp_action` was sent, or a pre-T11b bridge). | Assume it fired. The opposite assumption pins the shadow meter at `1.0` forever and reintroduces the flailing the tight readiness epsilon exists to prevent. |

The bridge already has this value: `MacroExecutor.begin(...)` returns `{swung}`.
It is threaded onto the wire, not recomputed.

```json
{"type":"state",
 "self":{"pos":[0.5,64.0,0.5],"yaw":0.0,"pitch":0.0,"velocity":[0.0,0.0,0.0],
         "on_ground":true,"health":20.0,"held_item":"iron_sword","attack_cooldown":1.0},
 "opponent":{"pos":[3.5,64.0,1.5],"yaw":3.14,"pitch":0.0,"velocity":[0.0,0.0,0.0],"health":20.0},
 "events":{"damage_dealt":0.0,"damage_taken":0.0,"i_died":false,"opponent_died":false},
 "arena":{"wall_distances":[8.0,8.0,8.0,8.0]},
 "tick":12345,"code_version":"abc123"}
```

### `reset_ack`

Acknowledges a [`reset`](#reset) **after the post-reset read-back gate**.

| Field      | Type    | Meaning                                                                       |
|------------|---------|-------------------------------------------------------------------------------|
| `type`     | `"reset_ack"` (const) | Discriminator.                                                  |
| `ok`       | bool    | `true` if the read-back gate confirmed; **`false` signals a read-back timeout** (the env must treat the episode as failed-to-start). |
| `readback` | object  | Post-reset read-back snapshot used to verify spawn/health/gear gates. Free-form by design (gate fields evolve). |

```json
{"type":"reset_ack","ok":true,"readback":{"self_hp":20.0,"opp_hp":20.0}}
```

---

## Message list (frozen)

| Direction      | `type`        | Required fields                                                | Optional fields |
|----------------|---------------|----------------------------------------------------------------|-----------------|
| Python → Node  | `reset`       | `type`, `episode`, `seed`                                      | — |
| Python → Node  | `step`        | `type`, `action` (int 0..7)                                   | `opp_action` (int 0..7 \| null) |
| Python → Node  | `close`       | `type`                                                        | — |
| Node → Python  | `state`       | `type`, `self`, `opponent`, `events`, `arena`, `tick`, `code_version` | `opp_action_executed` (bool \| null) |
| Node → Python  | `reset_ack`   | `type`, `ok`, `readback`                                      | — |

Both validators are `additionalProperties: false`, so a field is accepted **only**
if it is listed above — an optional field that reaches only the prose and the
Python is rejected on the wire precisely when something first tries to use it.
