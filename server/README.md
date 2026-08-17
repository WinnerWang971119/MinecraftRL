# server/ — Paper training server + arena (task T8)

Paper Minecraft server for the PvP RL training arena: an **offline-mode**,
**flat-world** server holding **N enclosed bedrock pads** in a single JVM, with
two **opped bot accounts per pad**, a tiny view/sim distance, no mob spawning,
and a datapack that owns pad geometry, fixed spawns, and a stationary dummy.

**The datapack is the sole reset authority.** The bridge bots (`bridge/bot.js`)
connect as players and send exactly **one** command per episode —
`/function arena:reset_pad {x,z,learner,dummy,nonce}` — and then run their
read-back gate. They no longer issue their own `/tp`, `/effect clear` and regear
sequence; that is now inside the datapack's macro functions. The single command
still only works because the accounts are opped: RCON is disabled and the
launcher has no console, so an opped bot's chat is the only command channel.

**Owner workstream:** Environment/bridge track.

## Pinned versions (from `compat_check.md` / Tv)

| Thing | Pin |
|-------|-----|
| Paper | **1.21.1 build 133**, channel `STABLE` (`paper-1.21.1-133.jar`) |
| Java  | **21 exactly** — see below; `start.sh` refuses anything else |
| Minecraft protocol | 1.21.1 (matches `bridge/package.json` mineflayer 4.37.1) |

**Java 21, not "21 or newer".** Measured on macOS: Temurin 26 reaches
`Done (…)! For help, type "help"` and then the JVM aborts with a native SIGSEGV
inside `libasyncProfiler` — spark, bundled in Paper and auto-started as the
background profiler, reads JVM internals that Java 26 relocated. The crash is
*delayed*, so it presents as a bridge fault rather than a JVM fault. That is why
`server/setup/start.sh` selects Java 21 via `/usr/libexec/java_home -v 21` on
macOS, asserts the resolved JVM's major version everywhere else, and **aborts**
on a mismatch (`ALLOW_JAVA_MISMATCH=1` overrides deliberately). Full evidence:
`server/compat_check.md`.

## Prerequisites

- **Java 21** on `PATH` or via `JAVA_HOME` (`java -version`). macOS:
  `brew install --cask temurin@21`.
- Network access to **`fill.papermc.io` / `fill-data.papermc.io`** for the
  one-time jar download. The old `api.papermc.io/v2` endpoint is dead (410 Gone).
- `bash` (macOS/Linux). PowerShell 7+ (`pwsh`) equivalents exist for Windows.

## Quick start

```bash
# 1. Download + verify the jar, write eula.txt / server.properties / bukkit.yml,
#    install the datapack. PADS sizes max-players; omit it for the single pad.
bash server/setup/setup.sh
PADS=8 bash server/setup/setup.sh     # or, for an 8-pad fleet

# 2. Launch the server (--nogui console; 'stop' to shut down).
bash server/setup/start.sh
```

Windows:

```powershell
pwsh -NoProfile -File server/setup/setup.ps1
pwsh -NoProfile -File server/setup/start.ps1
```

For a multi-pad fleet, do not launch `start.sh` by hand — use
`server/setup/start-pads.sh --pads N`, which owns `ops.json`, the staggered
joins, and the reset-before-step prime barrier. See `RUNBOOK.md`.

`setup.*` is **idempotent**: it re-verifies the jar's sha256 on every run and
re-downloads only on a mismatch (pass `-Force` / `FORCE=1` to force), and only
rewrites the config files it owns. **Neither script runs git, and you must run
`setup` before `start`.**

## What setup writes

- **`paper-1.21.1-133.jar`** — downloaded through PaperMC's **v3** "fill" API,
  which resolves to a content-addressed URL:
  `https://fill-data.papermc.io/v1/objects/39bd8c00b9e18de91dcabd3cc3dcfa5328685a53b7187a2f63280c22e2d287b9/paper-1.21.1-133.jar`.
  That sha256 is **pinned in the setup scripts and verified on every run**,
  including runs that skip the download — an already-present jar can be corrupt,
  hand-copied, or a leftover from the dead v2 URL, and `start.sh` would execute
  it either way.
- **`eula.txt`** — `eula=true` (accepting the Mojang EULA).
- **`server.properties`** — training-tuned (see rationale below). `max-players`
  is sized `max(20, 2N+10)` for the pad count.
- **`bukkit.yml`** — every value is the Bukkit default except
  `settings.connection-throttle: -1`, defense in depth against a join storm.
  Measured: CraftBukkit exempts `127.0.0.1` from the throttle anyway, so this
  only becomes load-bearing when bridges run on a different host than the JVM.
- **`world/datapacks/arena/`** — a copy of `server/arena/` (the arena datapack).
  Paper loads **only** this copy, so re-run setup after any datapack change;
  `start-pads.sh` compares the two and refuses to launch on a stale world copy.

**`server.properties` and `bukkit.yml` are regenerated on every setup run.** A
hand edit to either is silently overwritten — change the script instead.

`ops.json` is **generated, not committed** — it is git-ignored (issue #29). Paper
owns the file as much as we do: it reads the op list only at startup and rewrites
it on shutdown, and its contents depend on the pad count, so tracking it left the
tree dirty after every boot-and-stop cycle. `setup` does not write it either (the
op list depends on N). Two entry points do, before Paper boots:
`start-pads.sh --pads N` (2N bots) and `deploy/exhibition.py` (one pad). Write it
by hand with:

```bash
.venv/bin/python -m distributed.launcher --pads 1 --write-ops
```

**`start.sh` refuses to launch** unless that file opps `learner_bot` and
`dummy_bot` at level 4, so a fresh clone cannot boot into unopped bots — a failure
that is otherwise silent (an unopped bot cannot run `/function` at all, so the
arena is never built and every reset quietly does nothing).

## `server.properties` rationale

| Key | Value | Why |
|-----|-------|-----|
| `online-mode` | `false` | Bots join with just a username (offline UUIDs). It is also why **your own Minecraft client connects to `localhost:25565` unmodified** — see `docs/spectate.md`. |
| `level-type` | `minecraft:flat` | Flat arena — no terrain, no surprises, cheap to load. |
| `generator-settings` | `{}` | **Do not "fix" this.** It does not parse; Paper logs `ERROR: No key layers in MapLike[{}]` at world creation and falls back to the default flat preset — grass at `y=-61`, dirt at `-62/-63`, bedrock at `-64`, everywhere. That fallback **is** the intended, empirically verified world. Supplying real layers would change topology and invalidate the analysis in `compat_check.md`. |
| `level-seed` | `8675309` | Fixed seed -> reproducible world (the spec asks to seed and log everything). |
| `generate-structures` | `false` | No villages/strongholds near spawn. |
| `spawn-monsters` / `spawn-animals` / `spawn-npcs` | `false` | No ambient entities -> the only entity in the obs is the opponent (fully-observed MDP). |
| `view-distance` / `simulation-distance` | `2` | Bots stay on their pads; loading more chunks is pure CPU/RAM waste, and **CPU/RAM is the throughput limit** (spec §9). 2 is the practical floor that still keeps a pad ticking, and it is an input to the ≥19 TPS scale gate. A human joining to watch sees ~32 blocks and blackness past it; **raise these for a nicer view and you invalidate the ladder** (and the edit is overwritten on the next setup run anyway). |
| `allow-nether` | `false` | No other dimensions to load. |
| `difficulty` | `normal` | Standard PvP damage numbers. |
| `pvp` | `true` | Required — the bots fight each other, and the repaired damage channel depends on real PvP damage landing. |
| `gamemode` / `force-gamemode` | `survival` / `true` | Survival so health/damage/combat behave normally; forced so a rejoin can't drift to creative. **This forcing applies to humans too:** every join puts you in survival at the world spawn (`0 64 0`), which is pad 0's learner spawn cell — inside the arena, next to a bot swinging an iron sword. Hits you take land in the learner's `damage_taken` reward term. Switch to spectator from the console the instant you join, and again after every reconnect. Procedure: `docs/spectate.md`. |
| `spawn-protection` | `0` | Opped bots act at spawn without protection blocking commands. |
| `max-players` | `max(20, 2N+10)` | 2 bots per pad plus headroom for a ghost session after a bridge restart, the reconnect overlap during a restart, and a human joining to look. `start-pads.sh` reads this value back and refuses to launch below `2N+10`, so raising N without re-running setup fails loudly instead of losing bots to a full server. |
| `max-chat-message-length` | `2048` | Anti-spam friendliness — the bridge issues commands through chat as an opped bot; never throttle or kick it. |
| `enforce-secure-profile` | `false` | Required for offline-mode bots (no Mojang chat signing). |
| `op-permission-level` | `4` | Bots need full op (level 4) for `/function`, `/tp`, `/effect`, `/attribute`, `/give`. |

## The bot accounts (opped, offline UUIDs)

Pad `i` gets one learner and one dummy. **Pad 0 uses `learner_bot` / `dummy_bot`**
so the single-pad path is byte-identical to the historical one; pads above 0 use
`learner_<i>` / `dummy_<i>`. `server/ops.json` ops every account at **level 4**;
`start-pads.sh` regenerates it for 2N bots before Paper boots (Paper reads the op
list only at startup, and will not re-read it on a running server).

The pad-0 usernames MUST match `bridge/bot.js` `DEFAULT_BOT_CONFIG`
(`learnerUsername` / `dummyUsername`):

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

Pad geometry, fixed spawns, the stationary dummy, and the MDP gamerules live in
**`server/arena/`** (a datapack). `setup.*` installs it into
`world/datapacks/arena/`.

Two layers, addressed by **1.20.2+ macro functions** so one datapack serves any
number of pads:

| Function | When | Arguments |
|---|---|---|
| `arena:setup` | once per boot; auto-runs on datapack load | none — world-wide gamerules, time, weather, world spawn, and pad 0 |
| `arena:setup_pad` | once per pad per boot | `{x,z}` — builds and encloses one pad at that anchor |
| `arena:reset_pad` | every episode, from the bridge | `{x,z,learner,dummy,nonce}` — re-places, re-gears, heals, feeds, and re-pins both bots' spawnpoints |

Macro functions are used rather than `execute positioned` because every command
in the arena is absolute, entity-selector `x/y/z` arguments cannot be relative at
all, and player-name commands (`/tp`, `/clear`, `/attribute` on `learner_<i>`)
cannot be positionally parameterized by any mechanism. `$(x)`, `$(z)`,
`$(learner)`, `$(dummy)` solve coordinates and usernames in one mechanism.

**Pad geometry.** The anchor is the **learner's spawn cell**, not the floor
origin. For a pad at anchor `A`:

| Layer | Y | Extent |
|---|---|---|
| bedrock sub-floor | 62 | `x ∈ [A.x−8, A.x+16]`, `z ∈ [A.z−12, A.z+12]` (25×25) |
| `smooth_stone` floor | 63 | same footprint |
| air interior | 64–71 | same footprint |
| **closed bedrock ring** | 64–71 | the perimeter of that footprint, **including all four corners** |

No ceiling. Reachable interior is `x ∈ [A.x−7, A.x+15]`,
`z ∈ [A.z−11, A.z+11]`. Wall height is 8 blocks above the floor; a player jumps
~1.25 blocks and the bots cannot place blocks, so a pad is closed. Pads are 512
blocks apart, and `padAnchor(i)` in `distributed/launcher.py` is the **sole**
implementation of that spacing — no coordinate formula is duplicated in this
datapack or in any shell script.

Spawn template (matches `bridge/bot.js` `resetTemplate`): learner at
`[A.x+0.5, 64, A.z+0.5]` with an `iron_sword`; dummy at `[A.x+3.5, 64, A.z+0.5]`
with `knockback_resistance = 1.0`.

**`naturalRegeneration` is off**, set by `arena:setup` (not by
`server.properties`), and the reset restores **food and saturation** as well as
health. Both are required for the dummy's health to be stationary across
episodes: with regeneration on, a dummy that cannot die is net-positive to farm,
and the combat probe's exact per-hit deltas become false negatives on a correct
implementation.

See **`server/arena/README.md`** for the full function list, the gear table, the
knockback-immunity attribute, and the macro argument contracts.

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
- **No server-side reset plugin.** A reset is one `/function arena:reset_pad`
  issued by the pad's own opped bot, plus the bridge's read-back gate. Nothing
  server-side needs to be installed beyond the arena datapack.

## Compatibility check

Version pins **and everything measured on the live macOS bring-up** are in
**`server/compat_check.md`** (Tv / T7). That file is the authority for the Paper
build, the Java pin, the mineflayer/plugin versions, the PaperMC v3 download
URL, the block composition below the pads, the connection-throttle behavior, and
the attribute IDs. This README consumes its decisions rather than restating the
evidence — go there when you need the numbers.

## What is live-verified, and what is not

The server **has** been booted on macOS: jar downloaded and digest-verified,
world generated, datapack loaded, `naturalRegeneration` read back as `false`,
both bots joined and confirmed opped (with a negative control), the sub-pad block
composition scanned column by column, and the connection throttle measured. Re-run
any of it yourself with `node server/tools/probe_world.js` against a running
server — it is a standalone mineflayer client and does not touch the bridge's TCP
port, so it is safe to run alongside a training session.

Still open, and worth knowing before you trust a live result:

- **Issue #27 — a pad's geometry can be silently absent.** `/fill` into an
  unloaded chunk no-ops without an error, and a reset ack proves only that the
  bots were placed, never that walls exist. `arena:setup_pad` now forceloads the
  pad footprint before building; that fix is **not yet live-verified**. AC7 (all
  four walls and all four corners, exact bounds) is a live test that has not run.
- **Issue #28 — the episode-start attack-cooldown observable.** Fixed
  bridge-side; **pending live verification**. See the docstring of
  `eval/combat_probe.py` for the fingerprint of the pre-fix false-fail.
- The **bridge-driven** join path (`bridge/bot.js` → `BridgeServer`) is a
  separate assertion from the standalone probe's join and is not covered by it.

Both are easy to check by eye from inside the world — see `docs/spectate.md`.
