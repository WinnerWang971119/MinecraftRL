# server/ — Paper training server + arena (task T8)

Paper Minecraft server for the PvP RL training arena: an **offline-mode**,
**flat-world** server with two **opped bot accounts**, a tiny view/sim distance,
no mob spawning, and a datapack that defines fixed spawns and a stationary
dummy. The bridge bots (`bridge/bot.js`) connect as players and issue `/tp`,
`/effect clear`, and regear commands on every reset — which only works because
the accounts are opped.

**Owner workstream:** Environment/bridge track.

## Pinned versions (from `compat_check.md` / Tv)

| Thing | Pin |
|-------|-----|
| Paper | **1.21.1 build 133**, channel `STABLE` (`paper-1.21.1-133.jar`) |
| Java  | **21+** required; this machine has **Java 25** |
| Minecraft protocol | 1.21.1 (matches `bridge/package.json` mineflayer 4.37.1) |

## Prerequisites

- **Java 21+** on `PATH` (`java -version`). Paper 1.21.1 requires it.
- Network access to `api.papermc.io` for the one-time jar download.
- (Windows) PowerShell 7+ (`pwsh`). The setup/start scripts also have bash
  equivalents for Linux/macOS cloud VMs.

## Quick start

```powershell
# 1. Download the jar + write eula.txt, server.properties, install the datapack.
pwsh -NoProfile -File server/setup/setup.ps1

# 2. Launch the server (--nogui console; 'stop' to shut down).
pwsh -NoProfile -File server/setup/start.ps1
```

Linux/macOS:

```bash
bash server/setup/setup.sh
bash server/setup/start.sh
```

`setup.*` is **idempotent**: re-running skips the jar download if it is already
present (pass `-Force` / `FORCE=1` to refresh) and only rewrites the config files
it owns. **Neither script runs git, and you must run `setup` before `start`.**

## What setup writes

- **`paper-1.21.1-133.jar`** — downloaded from
  `https://api.papermc.io/v2/projects/paper/versions/1.21.1/builds/133/downloads/paper-1.21.1-133.jar`.
- **`eula.txt`** — `eula=true` (accepting the Mojang EULA).
- **`server.properties`** — training-tuned (see rationale below).
- **`world/datapacks/arena/`** — a copy of `server/arena/` (the arena datapack).

`ops.json` is **committed** (it is the source of truth for the opped bots), so
setup does not generate it.

## `server.properties` rationale

| Key | Value | Why |
|-----|-------|-----|
| `online-mode` | `false` | Bots join with just a username (offline UUIDs). Also how the demo-day exhibition lets visitors join. |
| `level-type` | `minecraft:flat` | Flat arena — no terrain, no surprises, cheap to load. |
| `level-seed` | `8675309` | Fixed seed -> reproducible world (the spec asks to seed and log everything). |
| `generate-structures` | `false` | No villages/strongholds near spawn. |
| `spawn-monsters` / `spawn-animals` / `spawn-npcs` | `false` | No ambient entities -> the only entity in the obs is the opponent (fully-observed MDP). |
| `view-distance` / `simulation-distance` | `2` | Both bots and the dummy stay at spawn; loading more chunks is pure CPU/RAM waste, and **CPU/RAM is the throughput limit** (spec §9). 2 is the practical floor that still keeps the arena ticking. |
| `allow-nether` | `false` | No other dimensions to load. |
| `difficulty` | `normal` | Standard PvP damage numbers. |
| `pvp` | `true` | Required — the bots fight each other. |
| `gamemode` / `force-gamemode` | `survival` / `true` | Survival so health/damage/combat behave normally; forced so a rejoin can't drift to creative. |
| `spawn-protection` | `0` | Opped bots act at spawn without protection blocking commands. |
| `max-chat-message-length` | `2048` | Anti-spam friendliness — the bridge fires rapid `/tp` + `/effect` bursts per reset; never throttle or kick them. |
| `enforce-secure-profile` | `false` | Required for offline-mode bots (no Mojang chat signing). |
| `op-permission-level` | `4` | Bots need full op (level 4) for `/tp`, `/effect`, `/attribute`, `/give`. |

## The two bot accounts (opped, offline UUIDs)

`server/ops.json` ops both accounts at **level 4**. The usernames MUST match
`bridge/bot.js` `DEFAULT_BOT_CONFIG` (`learnerUsername` / `dummyUsername`):

| Username | Offline UUID | Level |
|----------|--------------|:-----:|
| `learner_bot` | `904ab765-0884-3b39-af00-9cdbf8d5f528` | 4 |
| `dummy_bot`   | `08809ff7-fb5e-3b8d-bdd1-a867b604ded8` | 4 |

### Offline-mode UUID note (why these exact values)

On an **offline-mode** server there is no Mojang account, so the server derives a
deterministic UUID from the username. Bukkit/Paper match `ops.json` entries by
**UUID**, not name — so these must be the correct offline UUIDs or the ops won't
apply. The derivation mirrors Mojang's:

```
offlineUUID(name) = Java UUID.nameUUIDFromBytes(("OfflinePlayer:" + name).getBytes(UTF_8))
                  = RFC-4122 version-3 (MD5) UUID over those bytes
```

Regenerate them with this Python snippet if you ever rename a bot:

```python
import hashlib, uuid
def offline_uuid(name: str) -> str:
    b = bytearray(hashlib.md5(("OfflinePlayer:" + name).encode("utf-8")).digest())
    b[6] = (b[6] & 0x0f) | 0x30  # version 3
    b[8] = (b[8] & 0x3f) | 0x80  # IETF variant
    return str(uuid.UUID(bytes=bytes(b)))

print(offline_uuid("learner_bot"))  # 904ab765-0884-3b39-af00-9cdbf8d5f528
print(offline_uuid("dummy_bot"))    # 08809ff7-fb5e-3b8d-bdd1-a867b604ded8
```

(The `3` in each UUID's third group is the version nibble — that's the tell that
it is a correct name-based v3 UUID.)

## Arena datapack

The flat arena, fixed spawns, stationary dummy, and MDP gamerules live in
**`server/arena/`** (a datapack). `setup.*` installs it into
`world/datapacks/arena/`. The per-episode reset functions (`arena:spawn_learner`,
`arena:spawn_dummy`, `arena:reset`) are invoked by the bridge's reset RPC (T7a);
the once-per-boot scaffolding (`arena:setup`) auto-runs on datapack load. See
**`server/arena/README.md`** for the spawn/gear table, the knockback-immunity
attribute, and how the bridge calls the functions.

Spawn template (matches `bridge/bot.js` `resetTemplate`): learner at
`[0.5, 64, 0.5]` with an `iron_sword`; dummy at `[3.5, 64, 0.5]` with
`knockback_resistance = 1.0`.

**Attribute ID note — this stack REQUIRES the `generic.` infix.** The arena functions use
`minecraft:generic.knockback_resistance` and `minecraft:generic.movement_speed`. The flattening
that *removed* the `generic.` infix landed in **Minecraft 1.21.2**; Paper here is pinned to
**1.21.1 build 133**, which predates it, so the un-prefixed ids simply do not exist.

This was verified two ways after an earlier revision of this file asserted the opposite:

1. Live console round-trips against the booted server (`server/logs/latest.log`) — the
   un-prefixed probes were immediately retried with the `generic.` prefix, and only the
   prefixed form was used thereafter.
2. This repo's own boot logs, which had been failing on it for three consecutive boots
   (`server/logs/2026-08-08-{1,2,3}.log.gz`):

   ```
   [ServerMain/ERROR]: Failed to load function arena:spawn_dummy
   IllegalArgumentException: Whilst parsing command on line 41:
     Can't find element 'minecraft:knockback_resistance' of type 'minecraft:attribute'
   ```

Do not re-flip this from memory. A wrong id is not a soft failure: inside a **macro** function a
parse error aborts instantiation of the entire function, so *no* command in it runs — the dummy
gets no teleport, heal, food, knockback immunity or spawnpoint, and nothing appears in the boot
log. If the pinned Paper version is ever moved to 1.21.2+, flatten both ids in the same commit.

## Plugins

- **`mineflayer-pvp` is OPTIONAL** — used only as a **cooldown reference**, NOT
  to drive ATTACK. The bridge calls raw `bot.attack(entity)` and computes the
  1.9+ attack cooldown itself (`bridge/bot.js` `computeAttackCooldown`). Do not
  let the pvp plugin pace or issue swings.
- **`mineflayer-pathfinder` is demoted** from a day-1 blocker — movement uses
  `bot.setControlState(...)`, not pathfinder goals.
- **No server-side reset plugin.** Resets are driven by the bridge's opped
  commands + the read-back gate (the `/debate` verdict deferred a reset plugin).
  Nothing server-side needs to be installed beyond the arena datapack.

## Compatibility check

Version pins and the live-handshake follow-up are documented in
**`server/compat_check.md`** (Tv). That file is the authority for the Paper
build, Java floor, and the mineflayer/plugin versions; this README only consumes
its decisions.

## Live follow-up (not done here)

This task authored config + scripts only — **no jar was downloaded and no server
was started**. The first real boot (download, EULA, world gen, datapack enable,
two-bot handshake) is the human follow-up in `compat_check.md`.
