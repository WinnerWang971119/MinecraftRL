# server/arena/ — training arena datapack (task T8)

A small datapack that defines the **flat PvP training arena**: fixed spawns, a
**stationary, knockback-immune dummy**, and the gamerules that keep the MDP
**fully observed and deterministic**. The bridge's reset RPC (`bridge/bot.js`
`handleReset` / T7a) drives the per-episode functions; the bots run the commands
because they are **opped** (`server/ops.json`).

## Layout

```
arena/
├─ pack.mcmeta                          # pack_format 48 (Minecraft 1.21.1)
└─ data/
   ├─ arena/function/
   │  ├─ load.mcfunction               # auto-run on load -> calls setup
   │  ├─ setup.mcfunction              # ONCE per boot: floor, gamerules, clear
   │  ├─ reset.mcfunction              # per-episode: spawn_learner + spawn_dummy
   │  ├─ spawn_learner.mcfunction      # learner -> [0.5,64,0.5], iron_sword
   │  └─ spawn_dummy.mcfunction        # dummy -> [3.5,64,0.5], kb_resist=1.0
   └─ minecraft/tags/function/
      └─ load.json                     # registers arena:load on datapack load
```

> **1.21 directory note:** Minecraft 1.21 renamed the datapack folders from
> `functions/` → `function/` and `tags/functions/` → `tags/function/`. This pack
> uses the **1.21 (singular)** names, matching `pack_format` 48.

## Spawn / gear template (MUST match `bridge/bot.js`)

| Entity        | Position (block center) | Facing            | Gear         | Health |
|---------------|-------------------------|-------------------|--------------|--------|
| `learner_bot` | `0.5, 64, 0.5`          | +X (toward dummy) | `iron_sword` | 20     |
| `dummy_bot`   | `3.5, 64, 0.5`          | -X (toward learner) | (none)     | 20     |

This mirrors `bridge/bot.js` `ArenaBots.resetTemplate`:

```js
resetTemplate = {
  health: 20.0,
  position: { x: 0.5, y: 64.0, z: 0.5 },
  inventory: ['iron_sword'],
  requireNoEffects: true,
};
// dummy /tp uses spawn.x + 3  ->  x = 3.5
```

Because the read-back gate enforces `requireNoEffects`, the spawn functions
**clear all effects** and avoid leaving any stored buff active. Knockback
immunity and void safety are therefore implemented WITHOUT effects:

- **Knockback immunity** is an **attribute**
  (`minecraft:knockback_resistance base set 1.0`), not an effect, so the
  dummy never gets shoved off its spawn and its position stays observable. Set
  every reset in `spawn_dummy` to survive respawns.
  *(1.21 flattened attribute IDs — `generic.` prefix removed; use `minecraft:knockback_resistance`
  not `minecraft:generic.knockback_resistance`.)*
- **Void / fall immunity** comes from the **bedrock under-floor** laid by
  `arena:setup` (y=62) plus `gamerule fallDamage false` — again, no stored
  effect, so the gate's "no active effects" check still passes.

## Gamerules (set once by `arena:setup`)

`doMobSpawning false`, `doDaylightCycle false`, `doWeatherCycle false`,
`keepInventory true` (the four required by the plan), plus `doFireTick false`,
`mobGriefing false`, `doImmediateRespawn true`, `showDeathMessages false`,
`randomTickSpeed 0`, and damage-off rules (`fallDamage`, `drowningDamage`,
`fireDamage`, `freezeDamage`) so the only damage source is PvP. Time is pinned to
day and weather to clear so observations never vary with the world clock.

## How the bridge invokes these (T7a)

The bridge already issues its own `/tp`, `/effect clear`, and regear in
`handleReset` and then runs the read-back gate. It can instead (or additionally)
call the opped functions as single commands:

```
/function arena:setup           # once, at server boot (also auto-runs on load)
/function arena:reset            # every episode: re-places + re-gears both bots
# or the individual halves:
/function arena:spawn_learner
/function arena:spawn_dummy
```

Either way the commands are **async/unacked**, so the bridge MUST keep running
its read-back gate (`runReadbackGate`) and only reply `reset_ack{ok:true}` once
the learner's observed health/position/inventory/effects match the template.

## Enabling the pack

`server/setup/setup.ps1` (or `setup.sh`) copies this folder into
`world/datapacks/arena/`. On a fresh world it is enabled automatically; if added
to a running server, enable it live:

```
/datapack enable "file/arena"
/reload
/datapack list            # expect: file/arena (enabled)
/function arena:setup
```
