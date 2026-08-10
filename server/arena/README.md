# server/arena/ — training arena datapack (task T8, retopologized by T6)

A small datapack that defines the **enclosed PvP training pads**: bedrock
geometry, fixed spawns, a **stationary, knockback-immune dummy**, and the
gamerules that keep the MDP **fully observed and deterministic**.

**This datapack is the sole reset authority.** The bridge's reset RPC
(`bridge/bot.js` `handleReset`) sends exactly **one** command per episode —
`/function arena:reset_pad {…}` — and then runs its read-back gate. It no longer
issues its own `/tp`, `/effect clear` and regear sequence; all of that lives
here. The bots can run the commands because they are **opped**
(`server/ops.json`), and an opped bot's chat is the only command channel there
is: RCON is disabled and the fleet launcher has no console.

One datapack serves any number of pads because the per-pad functions are
**1.20.2+ macro functions**, parameterized by anchor and by username.

## Layout

```
arena/
├─ pack.mcmeta                          # pack_format 48 (Minecraft 1.21.1)
└─ data/
   ├─ arena/function/
   │  ├─ load.mcfunction               # auto-run on load -> calls arena:setup
   │  ├─ setup.mcfunction              # ONCE per boot: gamerules, time, world spawn, pad 0
   │  ├─ setup_pad.mcfunction          # MACRO {x,z}: build + enclose one pad
   │  ├─ reset_pad.mcfunction          # MACRO {x,z,learner,dummy,nonce}: per-episode reset
   │  ├─ spawn_learner_pad.mcfunction  # MACRO {x,z,learner,nonce}
   │  ├─ spawn_dummy_pad.mcfunction    # MACRO {x,z,dummy,nonce}
   │  ├─ reset.mcfunction              # pad-0 wrapper -> reset_pad {x:0,z:0,…}
   │  ├─ spawn_learner.mcfunction      # pad-0 wrapper
   │  └─ spawn_dummy.mcfunction        # pad-0 wrapper
   └─ minecraft/tags/function/
      └─ load.json                     # registers arena:load on datapack load
```

The four `*_pad` files hold all the logic. The three unsuffixed wrappers exist
only so the single-arena path (`/function arena:reset`) stays byte-compatible
with what it did before the pad topology existed — they take no arguments and
call the macro at anchor `(0, 0)` with `learner_bot` / `dummy_bot`. **Multi-pad
callers must not use them.**

> **1.21 directory note:** Minecraft 1.21 renamed the datapack folders from
> `functions/` → `function/` and `tags/functions/` → `tags/function/`. This pack
> uses the **1.21 (singular)** names, matching `pack_format` 48.

## Pad geometry

The **anchor** `(x, z)` is the **learner's spawn cell**, not the floor origin.
Everything is expressed relative to it, so `arena:setup_pad {x,z}` builds an
identical pad anywhere. Anchors come from `padAnchor(i)` in
`distributed/launcher.py`, which is the sole coordinate source in the repo — no
formula and no literal anchor is reproduced in this datapack, deliberately.

| Layer | Y | Extent (relative to anchor `A`) | Block |
|---|---|---|---|
| sub-floor | 62 | `x ∈ [−8, +16]`, `z ∈ [−12, +12]` — 25×25 | bedrock |
| floor | 63 | same footprint | `smooth_stone` |
| interior | 64–71 | same footprint | air |
| perimeter ring | 64–71 | the ring of that footprint, **all four corners included** | bedrock |

There is deliberately **no ceiling**. Net reachable interior is
`x ∈ [A.x−7, A.x+15]`, `z ∈ [A.z−11, A.z+11]` (23×23), standing on floor at
`y=63`. Wall height is 8 blocks above the feet level; a player jumps ~1.25 blocks
and the bots cannot place blocks, so a pad is closed.

The wall is built as four slabs each given its **full** span, so every corner
column is filled twice rather than zero times — the classic four-fills-with-corner-
holes bug is avoided by construction. Cross-check: `(25² − 23²) × 8 = 768`
distinct wall blocks. The header comment in `setup_pad.mcfunction` carries the
arithmetic.

Pads are ≥512 blocks apart. Walls plus spacing exist together for one reason:
`dummy.on('health')` records a health **drop with no attacker attribution**, so a
learner that reached a neighbouring pad would silently credit its damage to that
pad's policy.

**Known gap — issue #27.** A `/fill` into an unloaded chunk no-ops without an
error, so a pad's geometry could be silently absent while every reset still acks.
`setup_pad` now forceloads the footprint before building and releases it after.
That fix has **not been live-verified**; AC7 (walls and corners present, exact
bounds asserted) is still an unrun live test.

## Spawn / gear template (MUST match `bridge/bot.js`)

| Entity | Position | Gear | Health / food | Attributes |
|---|---|---|---|---|
| learner (`learner_bot` at pad 0) | `A.x+0.5, 64, A.z+0.5` | `iron_sword` | 20 / full saturation | — |
| dummy (`dummy_bot` at pad 0) | `A.x+3.5, 64, A.z+0.5` | none | 20 / full saturation | `knockback_resistance = 1.0`, `movement_speed = 0.0` |

This mirrors `bridge/bot.js` `ArenaBots.resetTemplate`, which at anchor `(0,0)` is:

```js
resetTemplate = {
  health: 20.0,
  position: { x: 0.5, y: 64.0, z: 0.5 },
  inventory: ['iron_sword'],
  requireNoEffects: true,
};
// dummy /tp uses spawn.x + 3  ->  x = 3.5
```

Both bots also get a per-bot `/spawnpoint` on their own pad at the end of every
reset. This is not optional: `doImmediateRespawn` is on and the world spawn is
pad 0's anchor, so without it **every death teleports the bot into pad 0** — a
cross-pad contamination event.

Because the read-back gate enforces `requireNoEffects`, the spawn functions
**clear all effects first, then grant the instant effects** — never the reverse,
since a trailing `effect clear` in the same tick can strip an instant effect
before its first tick applies it. Nothing lingers: `/effect give`'s duration is
*"in seconds (or in gameticks for `instant_damage`, `instant_health`, and
`saturation`)"* (minecraft.wiki, `Commands/effect`), so the `1` on those lines is
**one gametick**. Knockback immunity and fall safety are implemented WITHOUT
effects:

- **Knockback immunity** is an **attribute**
  (`minecraft:generic.knockback_resistance base set 1.0`), not an effect, so the
  dummy never gets shoved off its spawn and its position stays observable. Set
  every reset in `spawn_dummy_pad` to survive respawns.
  *(The `generic.` infix is **required** on the pinned Paper 1.21.1 stack — the
  flattening that removed it landed in 1.21.2. Verified live and against this
  repo's boot logs; see `server/README.md` and the header of
  `spawn_dummy_pad.mcfunction` before changing it. Inside a macro function a bad
  attribute id aborts the **whole** function at instantiation, so the entire
  reset silently does nothing.)*
- **Fall safety** comes from the bedrock sub-floor at `y=62` plus
  `gamerule fallDamage false` — again, no stored effect, so the gate's "no active
  effects" check still passes. Note that with the walls in place an edge-walk is
  no longer reachable at all; before them, it stranded the agent alive at
  `y ≈ −60` rather than killing it (`server/compat_check.md`).

## Gamerules (set once by `arena:setup`)

`doMobSpawning false`, `doDaylightCycle false`, `doWeatherCycle false`,
`keepInventory true` (the four required by the plan), plus `doFireTick false`,
`mobGriefing false`, `doImmediateRespawn true`, `announceAdvancements false`,
`showDeathMessages false`, `randomTickSpeed 0`, `spawnRadius 0`, and the
damage-off rules (`fallDamage`, `drowningDamage`, `fireDamage`, `freezeDamage`)
so the only damage source is PvP. Time is pinned to day and weather to clear so
observations never vary with the world clock.

**`naturalRegeneration false`** is the one to know about. It is set here, not in
`server.properties`, and it must stay off:

- with regeneration on, `20 HP dealt − 15 timeout − 2 step = +3`, so farming a
  dummy that cannot die is **net-positive** — the opposite of what the terminal
  rewards are shaped to do;
- interleaved `+1` heals between cooled hits make the combat probe's exact
  per-hit assertions **false-negative on a correct implementation**.

Regeneration off is only half of opponent-health stationarity. The other half is
that the reset restores **food and saturation**, not just health: regeneration
costs exhaustion, so a dummy whose food was never replenished had a regeneration
rate that silently decayed across episodes. Both halves together are AC18.

`arena:setup` also runs `setworldspawn 0 64 0`, which is pad 0's anchor — so a
player who joins before any reset lands **inside pad 0**, in survival, next to a
live combat bot. That matters for humans too; see `docs/spectate.md`.

## How the bridge invokes these

Per episode, one command, composed by `bridge/bot.js` from this pad's anchor and
usernames:

```
/function arena:reset_pad {x:<int>,z:<int>,learner:"<name>",dummy:"<name>",nonce:<int>}
```

Once per pad per boot, before any reset:

```
/function arena:setup_pad {x:<int>,z:<int>}
```

`arena:setup` (gamerules + pad 0) auto-runs on datapack load, so pad 0 needs no
explicit setup call.

**Every key is required.** A macro function errors if a referenced key is absent,
and the failure is total: not one command in the function runs. `nonce` is the
bridge's monotonic reset epoch, forwarded to both spawn functions and stamped
into their causality beacons so a beacon delayed past its own reset cannot
satisfy the next one. `x` and `z` must be **non-negative plain integers** with no
NBT type suffix — `$(x)` is a textual substitution and the call chain builds
`$(x).5`, so `-340` would yield `-340.5` and `340L` would yield a non-coordinate.
Neither is checked at runtime.

The pad-0 wrappers take no arguments and are byte-compatible with the pre-macro
path:

```
/function arena:reset            # == arena:reset_pad {x:0,z:0,learner:"learner_bot",dummy:"dummy_bot",nonce:0}
/function arena:spawn_learner
/function arena:spawn_dummy
```

Either way the commands are **async/unacked**, so the bridge MUST keep running
its read-back gate and only reply `reset_ack{ok:true}` once the observed
health/position/inventory/effects match the template — **and** once both spawn
functions' causality beacons for that `nonce` have arrived. If the gates match
but no beacon does, the bridge replies `ok:false` and names it loudly on stderr
(`reset NOT confirmed by the datapack … arena:reset_pad may have aborted at
instantiation`). That is the classic silent failure this pack is defended
against: a macro that aborted at instantiation while a post-kill state happened
to look exactly like a fresh reset.

## Enabling the pack

`server/setup/setup.sh` (or `setup.ps1`) copies this folder into
`world/datapacks/arena/`. **Paper loads only that copy** — editing files here
changes nothing until setup is re-run, and `server/setup/start-pads.sh` diffs the
two and refuses to launch on a stale world copy. On a fresh world the pack is
enabled automatically; if added to a running server, enable it live:

```
/datapack enable "file/arena"
/reload
/datapack list            # expect: file/arena (enabled)
/function arena:setup
```
