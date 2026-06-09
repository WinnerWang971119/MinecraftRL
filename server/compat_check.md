# Day-1 Version Compatibility Check (Tv)

**Status:** DONE — run 2026-06-10, before the version pin lands in `agent/contract_config.py`.

**Owner:** Tv (Environment/bridge track)

This check confirms that `mineflayer`, `minecraft-data`, the Paper server build,
`mineflayer-pvp`, and `mineflayer-pathfinder` all support the candidate Minecraft
version (target **1.21.1**) before that version is frozen. If any of the four
lagged 1.21.1, the rule was to pin the highest Minecraft version all of them
support. **All four support 1.21.1, so the target holds.**

## Method (package metadata, no live server)

Verified from package metadata and release-channel APIs — a full live Paper +
Mineflayer + plugin handshake was **not** run here and is the human follow-up
listed at the bottom. Evidence gathered via:

- `npm view <pkg> version time.modified peerDependencies dependencies engines`
- Unpacking `minecraft-data@3.110.2` and reading
  `data/pc/common/versions.json` + `data/pc/common/protocolVersions.json`
- PaperMC downloads API: `GET https://api.papermc.io/v2/projects/paper/versions/1.21.1`
- GitHub repo + raw `package.json` / `README.md` for the two plugins and mineflayer

## Compatibility matrix

| Component | Supports 1.21.1? | Latest version | Last update | Evidence |
|-----------|:---------------:|----------------|-------------|----------|
| **mineflayer** | YES | `4.37.1` | npm published 2026-05-03 | README: "Supports Minecraft 1.8 to 1.21.11" with `1.21` explicitly listed. `engines.node` = `>=22`. Pins `minecraft-data ^3.108.0`, satisfied by 3.110.2. 1.21.1 was in the first 1.21 line to land, so it is firmly supported (the rocky 1.21.x patches were 1.21.5/1.21.6/1.21.7/1.21.11, all *later* than our target). |
| **minecraft-data** | YES | `3.110.2` | npm published 2026-05-13 | `data/pc/1.21.1/` data directory exists; `pc/common/versions.json` includes `1.21.1`. Protocol entry: `{minecraftVersion:"1.21.1", version:767, dataVersion:3955, usesNetty:true, majorVersion:"1.21", releaseType:"release"}`. Version list runs far past 1.21.1 (up to 26.x snapshots), so 1.21.1 is mature, not bleeding-edge. |
| **mineflayer-pvp** | YES | `1.3.2` (npm) | npm 2022-07-03; GitHub `master` pushed 2026-04-01, **not archived**, 13 open issues | `package.json` declares `mineflayer ^4.0.0` and `mineflayer-pathfinder ^2.0.0` — both satisfied by the current stack. The plugin is MC-version-agnostic: it has no hardcoded version list and delegates all version handling to mineflayer / minecraft-data, so it works on any MC version mineflayer supports, including 1.21.1. Provides the 1.9+ attack-cooldown-aware attack solver we need. |
| **mineflayer-pathfinder** | YES | `2.4.5` (npm) | npm 2023-09-04; GitHub `master` pushed 2026-04-17, **not archived**, 47 open issues | `package.json` deps: `minecraft-data ^3.5.1` (satisfied by 3.110.2), devDep `mineflayer ^4.3.0`. Also MC-version-agnostic — no hardcoded version list; rides on mineflayer + minecraft-data, so 1.21.1 is supported. |
| **Paper** | YES | `1.21.1` build **133** (channel `STABLE`) | n/a (server jar) | PaperMC API returns 130+ builds for 1.21.1 (2..133, latest 133, jar `paper-1.21.1-133.jar`, channel `STABLE`). Paper 1.21.1 requires **Java 21+**; this machine has Java 25 → OK. |

### Why the stale npm dates on the plugins are not a blocker

`mineflayer-pvp` and `mineflayer-pathfinder` have old *npm publish* dates (2022 /
2023) but both GitHub repos were pushed in **April 2026** and are not archived.
Neither plugin hardcodes a Minecraft version table — they consume mineflayer and
minecraft-data through semver ranges and operate on whatever version mineflayer
negotiates. Because mineflayer 4.37.1 + minecraft-data 3.110.2 fully support
1.21.1, the plugins do too. The npm `latest` tags are still the versions T8/T6
should install; if a live regression surfaces, install the plugins from the
`master` git refs instead.

## DECISION

- **Minecraft version to pin: `1.21.1`** — all four dependencies and mineflayer
  itself support it. No downgrade to a "highest common version" is needed; 1.21.1
  is the confirmed common target. (It is well past the 1.9 attack-cooldown combat
  cutover, so the cooldown-based PvP mechanics this project depends on are present.)
- **Paper build to pin: `1.21.1` build `133`** (channel `STABLE`,
  `paper-1.21.1-133.jar`). Requires Java 21+; satisfied by the installed Java 25.
- **Node version: `v24.13.0`** — this is on the **Node 24 "Krypton" LTS** line
  (entered Active LTS 2025-05-06, in maintenance through 2028-04-30), i.e. a
  maintained LTS. mineflayer's minimum is `engines.node >=22`, so v24.13.0 is
  comfortably above the floor.

### npm versions to install (for T6 pin + T8 bridge `package.json`)

| Package | Pin |
|---------|-----|
| mineflayer | `4.37.1` |
| minecraft-data | `3.110.2` (transitive via mineflayer; pin for reproducibility) |
| mineflayer-pvp | `1.3.2` |
| mineflayer-pathfinder | `2.4.5` |

### Installed toolchain on this machine (recorded, unchanged)

| Tool | Version | Note |
|------|---------|------|
| Node | v24.13.0 | Node 24 "Krypton" LTS (maintenance); ≥ mineflayer min `>=22` |
| npm | 11.6.2 | — |
| Java | OpenJDK 25.0.3 | ≥ Paper 1.21.1 requirement of Java 21+ |
| Python | 3.14.2 | agent/training side; ≥ project floor of 3.11 |

## Consumers of this decision

- **T6** writes these confirmed pins into `agent/contract_config.py`: the
  Minecraft version string `1.21.1`, the Node version assertion (`>=22`, observed
  v24.13.0), and the Python version assertion. Importing that module at startup
  turns a version mismatch into a hard error instead of a silent bug.
- **T8** (Paper server setup) installs **Paper 1.21.1 build 133**
  (`paper-1.21.1-133.jar`) under `server/` and the bridge installs
  `mineflayer@4.37.1`, `mineflayer-pvp@1.3.2`, `mineflayer-pathfinder@2.4.5`.

## Live handshake — human follow-up

This compat check is metadata + release-channel only. Before relying on the pin,
run a one-off live handshake to confirm the full stack actually connects and the
two plugins load against a real 1.21.1 server:

1. **Download the server jar** (matches the pin above):
   ```bash
   curl -L -o server/paper-1.21.1-133.jar \
     https://api.papermc.io/v2/projects/paper/versions/1.21.1/builds/133/downloads/paper-1.21.1-133.jar
   ```
2. **First boot to generate `eula.txt`**, then accept it:
   ```bash
   cd server
   java -jar paper-1.21.1-133.jar --nogui   # exits asking for EULA
   # set eula=true in server/eula.txt
   ```
3. **Set `server.properties` for offline test arena**: `online-mode=false`,
   `spawn-protection=0`, a low `view-distance`/`simulation-distance`, and op the
   bot account so it can `/tp`, `/effect clear`, regear.
4. **Launch the server** and wait for "Done":
   ```bash
   java -Xms2G -Xmx2G -jar paper-1.21.1-133.jar --nogui
   ```
5. **Connect a Mineflayer bot and load both plugins** (minimal probe):
   ```js
   const mineflayer = require('mineflayer')          // 4.37.1
   const { pathfinder } = require('mineflayer-pathfinder') // 2.4.5
   const { plugin: pvp } = require('mineflayer-pvp')       // 1.3.2

   const bot = mineflayer.createBot({
     host: 'localhost', port: 25565,
     username: 'TestBot', version: '1.21.1', auth: 'offline',
   })
   bot.loadPlugin(pathfinder)
   bot.loadPlugin(pvp)
   bot.once('spawn', () => {
     console.log('spawned on', bot.version)   // expect 1.21.1
     console.log('pvp loaded:', !!bot.pvp, '| pathfinder loaded:', !!bot.pathfinder)
     process.exit(0)
   })
   bot.on('error', (e) => { console.error('handshake error:', e); process.exit(1) })
   bot.on('kicked', (r) => { console.error('kicked:', r); process.exit(1) })
   ```
6. **Confirm combat mechanics**: spawn a second bot (or a dummy), call
   `bot.pvp.attack(target)`, and verify the attack solver respects the **1.9+
   attack cooldown** (swings pace to the cooldown, not spammed every tick). This
   is the behavior the project's combat model depends on.

If step 5 reports a version/protocol mismatch or either plugin fails to attach,
re-pin to the highest 1.21.x that mineflayer cleanly supports (do **not** drop
below 1.9 — pre-1.9 has no attack cooldown) and update T6/T8 accordingly.
