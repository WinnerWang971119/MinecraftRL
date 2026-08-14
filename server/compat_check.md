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
  — **historical record only; this v2 endpoint now returns 410 Gone. Do not
  copy it.** Use the v3 "fill" API instead (see the macOS bring-up section).
- GitHub repo + raw `package.json` / `README.md` for the two plugins and mineflayer

## Compatibility matrix

| Component | Supports 1.21.1? | Latest version | Last update | Evidence |
|-----------|:---------------:|----------------|-------------|----------|
| **mineflayer** | YES | `4.37.1` | npm published 2026-05-03 | README: "Supports Minecraft 1.8 to 1.21.11" with `1.21` explicitly listed. `engines.node` = `>=22`. Pins `minecraft-data ^3.108.0`, satisfied by 3.110.2. 1.21.1 was in the first 1.21 line to land, so it is firmly supported (the rocky 1.21.x patches were 1.21.5/1.21.6/1.21.7/1.21.11, all *later* than our target). |
| **minecraft-data** | YES | `3.110.2` | npm published 2026-05-13 | `data/pc/1.21.1/` data directory exists; `pc/common/versions.json` includes `1.21.1`. Protocol entry: `{minecraftVersion:"1.21.1", version:767, dataVersion:3955, usesNetty:true, majorVersion:"1.21", releaseType:"release"}`. Version list runs far past 1.21.1 (up to 26.x snapshots), so 1.21.1 is mature, not bleeding-edge. |
| **mineflayer-pvp** | YES | `1.3.2` (npm) | npm 2022-07-03; GitHub `master` pushed 2026-04-01, **not archived**, 13 open issues | `package.json` declares `mineflayer ^4.0.0` and `mineflayer-pathfinder ^2.0.0` — both satisfied by the current stack. The plugin is MC-version-agnostic: it has no hardcoded version list and delegates all version handling to mineflayer / minecraft-data, so it works on any MC version mineflayer supports, including 1.21.1. Provides the 1.9+ attack-cooldown-aware attack solver we need. |
| **mineflayer-pathfinder** | YES | `2.4.5` (npm) | npm 2023-09-04; GitHub `master` pushed 2026-04-17, **not archived**, 47 open issues | `package.json` deps: `minecraft-data ^3.5.1` (satisfied by 3.110.2), devDep `mineflayer ^4.3.0`. Also MC-version-agnostic — no hardcoded version list; rides on mineflayer + minecraft-data, so 1.21.1 is supported. |
| **Paper** | YES | `1.21.1` build **133** (channel `STABLE`) | n/a (server jar) | PaperMC API returns 130+ builds for 1.21.1 (2..133, latest 133, jar `paper-1.21.1-133.jar`, channel `STABLE`). Paper 1.21.1 needs **exactly Java 21** in practice — see the Java section below; "21+" is wrong. |

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
  `paper-1.21.1-133.jar`). **Run it on Java 21, not "21+"** — a newer JDK boots
  and then crashes natively. See "Java version — measured, supersedes 21+" below.
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
| Java | OpenJDK 25.0.3 | **Superseded.** Recorded on the old Windows box. Do not read this as "OK" — see the Java section below. |
| Python | 3.14.2 | agent/training side; ≥ project floor of 3.11 |

> The table above was recorded on the original Windows machine before anything
> had been booted. The sections below record what was actually measured on the
> macOS bring-up (task T7) and supersede it wherever they conflict.

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

1. **Download the server jar** (matches the pin above). Just run
   `server/setup/setup.sh`, which does this and verifies the digest. The raw
   equivalent — note the **v2 API is dead (410 Gone)**, so this is the v3
   content-addressed URL:
   ```bash
   curl -fL -o server/paper-1.21.1-133.jar \
     https://fill-data.papermc.io/v1/objects/39bd8c00b9e18de91dcabd3cc3dcfa5328685a53b7187a2f63280c22e2d287b9/paper-1.21.1-133.jar
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

### TC7b live assertions (damage-event feed — must verify by hand)

After T7b wires both bots, two additional live checks are required before the
damage-event feed is considered production-ready:

**TC7b-W1a — reconnect/re-wire does not double-count damage.**
Simulate a reconnect: call `arena.wireDamageEvents()` a second time while both
bots are already connected (e.g. after a bot drop and rejoin in the same
process). Then land one real hit on the dummy bot and observe the `state`
message the bridge emits at the next step boundary. `events.damage_dealt` must
equal the actual hit amount **exactly once** — not twice. A double-count here
means the reconnect re-registered the handler without removing the previous one,
which silently corrupts every reward signal going forward.

**TC7b-W1b — opponent respawn mid-observation does not drop the next hit.**
Let the dummy bot die (opponent health → 0) and wait for it to respawn to full
health. Without resetting, land one real hit on the (now-full-health) dummy and
observe the `state` message. `events.damage_dealt` must equal the actual hit
amount, **not zero or a truncated value**. A dropped or under-counted hit means
`_prevOpponentHealth` was left at the stale post-death value and the health
increase on respawn was measured as a negative delta that silently zeroed the
damage. The same scenario applies symmetrically to the learner bot (self death
→ respawn → take a hit): `events.damage_taken` must equal the actual hit, not
zero.

If step 5 reports a version/protocol mismatch or either plugin fails to attach,
re-pin to the highest 1.21.x that mineflayer cleanly supports (do **not** drop
below 1.9 — pre-1.9 has no attack cooldown) and update T6/T8 accordingly.

---

# macOS bring-up, measured (task T7)

Everything below was **observed on a live Paper `1.21.1-133` server** on macOS
26.5 (arm64, Apple Silicon), not inferred from metadata. Where it conflicts with
the pre-boot sections above, this section wins.

## Java version — measured, supersedes "21+"

**Run Paper on Java 21. Not "21 or newer".**

Temurin **26.0.1** reaches `Done (6.430s)! For help, type "help"` and then, a few
seconds later, the JVM aborts:

```
# A fatal error has been detected by the Java Runtime Environment:
#  SIGSEGV (0xb) at pc=0x0000000109fb8b80
# Problematic frame:
# C  [libasyncProfiler.so.tmp+0x10b80]  Lookup::fillJavaMethodInfo(MethodInfo*, _jmethodID*, bool)+0x3c
# The crash happened outside the Java Virtual Machine in native code.
```

spark — bundled in Paper and auto-started as the background profiler — ships a
native async-profiler that reads JVM-internal structures Java 26 relocated.
Java 26 additionally emits restricted-method and `sun.misc.Unsafe`
terminal-deprecation warnings, so the boot is not warning-free there either.

**The crash is delayed, which is the dangerous part.** The server reports a
successful startup first, so the failure presents as a bridge fault rather than
a JVM fault. `server/setup/start.sh` therefore does two things: it selects
Java 21 via `/usr/libexec/java_home -v 21` on macOS, and — because that helper
does not exist on Linux — it **asserts the resolved JVM's major version and
refuses to launch on a mismatch** (override with `ALLOW_JAVA_MISMATCH=1`).

Temurin **21.0.11** boots clean and stable. On Java 21 the only remaining
console noise is expected and intended:

| Line | Why it is not a problem |
|---|---|
| `**** SERVER IS RUNNING IN OFFLINE/INSECURE MODE!` (×4) | Intended: `online-mode=false` for offline bot joins |
| `Advanced terminal features are not available` | No TTY (launched headless/non-interactive) |
| `*** Warning, you've not updated in a while! ***` | The build is deliberately pinned to 133 |
| `ERROR: No key layers in MapLike[{}]` | `generator-settings={}`; see below. World creation only |

## PaperMC download API — the v2 URL is dead

`https://api.papermc.io/v2/.../downloads/<jar>` now returns **410 Gone**. Builds
resolve through the v3 "fill" API instead:

```
https://fill.papermc.io/v3/projects/paper/versions/1.21.1/builds/133
```

which reports a content-addressed
`https://fill-data.papermc.io/v1/objects/<sha256>/<jar>` URL plus the digest.
Same project / version / build 133 / STABLE channel / jar name; metadata commit
`3cb8529bd…` matches the server's own `1.21.1-133-ver/1.21.1@3cb8529` banner.
Both `setup.sh` and `setup.ps1` pin `sha256
39bd8c00b9e18de91dcabd3cc3dcfa5328685a53b7187a2f63280c22e2d287b9` and verify it
on **every** run, including when the download is skipped.

## Block composition below the arena — AC17, CONFIRMED

**This settles whether walking off the platform is a death or a stranding. It is
a stranding.** T5's archive analysis depends on this result.

**Reproduce it yourself** — the probe is committed, so this is a re-runnable
measurement rather than a claim. With the server running:

```bash
server/setup/start.sh              # terminal 1
node server/tools/probe_world.js   # terminal 2
```

It is a standalone mineflayer client (it does not touch the bridge's TCP port,
so it cannot disturb a training run) and adds no dependency beyond the
mineflayer already in `bridge/node_modules`. `--host/--port/--pad-x/--pad-z`
let you point it at another server or a non-zero pad anchor.

Output below is verbatim from that script: full-height columns from `y=319` down
to `y=-64`, 0 of 384 levels unloaded.

```
--- column x=0, z=0 : INSIDE pad footprint (learner spawn column) ---
    y=  63  minecraft:smooth_stone
    y=  62  minecraft:bedrock
    y= -61  minecraft:grass_block
    y= -62  minecraft:dirt
    y= -63  minecraft:dirt
    y= -64  minecraft:bedrock

--- column x=20, z=0 : OUTSIDE footprint, +X (past the x=16 edge) ---
    y= -61  minecraft:grass_block
    y= -62  minecraft:dirt
    y= -63  minecraft:dirt
    y= -64  minecraft:bedrock
```

Column `x=0, z=20` was identical to `x=20, z=0`. `y=63 smooth_stone` and
`y=62 bedrock` are the datapack's pad and exist only inside the footprint;
everything from `y=61` down to `y=-60` is air.

**Mechanism.** `server.properties` sets `level-type=minecraft:flat` with
`generator-settings={}`. That value does not parse — Paper logs
`ERROR: No key layers in MapLike[{}]` at world creation and falls back to the
**default flat preset**, which is exactly the grass/dirt/dirt/bedrock stack at
`y=-61..-64`. **Do not "fix" `generator-settings`**: the fallback IS the
intended world, and supplying real layers would change topology and invalidate
this analysis. Both setup scripts carry a comment saying so.

**Consequence.** The damage gamerules were read back live from the running
server (`node server/tools/probe_world.js`), verbatim:

```
/gamerule fallDamage      -> Gamerule fallDamage is currently set to: false
/gamerule drowningDamage  -> Gamerule drowningDamage is currently set to: false
/gamerule fireDamage      -> Gamerule fireDamage is currently set to: false
/gamerule freezeDamage    -> Gamerule freezeDamage is currently set to: false
```

With `fallDamage` off, an agent that walks off the
`y=63` platform therefore falls ~124 blocks, **lands alive standing at `y=-60`,
and is stranded there for the remainder of the episode**. The void is
unreachable and an edge-walk is **never** a death — it appears in the data as a
timeout whose episode-end `y ≈ -60`, not as a loss. Any analysis reading
wander-offs as deaths is misreading them. The `void_immune` flag on the dummy
and the "anti-void safety" comment in the spawn functions are defensive
leftovers, not evidence of reachable void.

## Connection throttle — measured

Four back-to-back joins from a single source IP:

| `connection-throttle` | Source | Result |
|---|---|---|
| `-1` | `127.0.0.1` | 4/4 joined |
| `-1` | LAN address | 4/4 joined |
| `4000` (default) | `127.0.0.1` | **4/4 joined — loopback is exempt** |
| `4000` (default) | LAN address | 1 joined, 3 kicked `"Connection throttled! Please wait before reconnecting."` |

CraftBukkit exempts `127.0.0.1` from the throttle, so a fleet whose bridges all
run on the Paper host was never exposed. `bukkit.yml`'s `connection-throttle: -1`
(written by `setup.sh`) is defense in depth, and becomes load-bearing only when
bridges run on a different host than the JVM. Paper leaves the generated
`bukkit.yml` **byte-identical** across a boot — no rewrite, no key normalization.

## Attribute IDs — the `generic.` prefix is REQUIRED on 1.21.1

The flattening that removed `generic.` landed in **1.21.2**, so the pinned
1.21.1 predates it. Measured against the live server:

```
/attribute learner_bot minecraft:knockback_resistance base get
  -> Can't find element 'minecraft:knockback_resistance' of type 'minecraft:attribute'
/attribute learner_bot minecraft:generic.knockback_resistance base get
  -> Base value of attribute Knockback Resistance for entity learner_bot is 0
```

Same for `movement_speed`. **In a macro function (`$`-prefixed lines) a bad ID
is not caught at load time**, and at invocation the failure is total — not one
command in the function runs:

```
Failed to instantiate function arena:spawn_dummy_pad: While instantiating macro
arena:spawn_dummy_pad: Command 'attribute dummy_bot minecraft:knockback_resistance
base set 1.0' caused error: Can't find element ...
```

So a single wrong attribute ID silently voids an entire reset (no tp, no heal,
no regear, knockback resistance left at 0). Verify any `/attribute` id with
`/attribute <bot> <id> base get` as an opped bot before relying on it.

## Bots join opped — verified live (AC17)

Both bot accounts join and are opped. A non-opped player cannot even see
`/gamerule` in its command tree, so a real value coming back is itself the proof:

```
[join] learner_bot spawned at (0.5, 64, 0.5) (+292ms)
[join] dummy_bot   spawned at (0.5, 64, 0.5) (+49ms later)
  op check learner_bot    -> OPPED
  op check dummy_bot      -> OPPED
```

Negative control, same server, an account absent from `ops.json`:

```
not_opped_bot -> "Unknown or incomplete command, see below for error
                  gamerule naturalRegeneration<--[HERE]"
```

Re-run with `node server/tools/probe_world.js`. **Scope note:** this is a
standalone mineflayer client. The bridge-driven join path
(`bridge/bot.js` → `BridgeServer`) is a separate assertion and is not covered
here.

## `naturalRegeneration` off — verified live (AC18, gamerule half)

Read back from the running server:

```
/gamerule naturalRegeneration  ->  Gamerule naturalRegeneration is currently set to: false
```

The gamerule is set by the datapack in
`server/arena/data/arena/function/setup.mcfunction`, not by `server.properties`
or either setup script. Earlier boots in this bring-up reported `true`; the
value above was read after the datapack change landed. AC18's remaining half —
food/saturation restored at reset, so the dummy's health is identical at every
episode start — belongs to the reset functions and is not asserted here.

## Python

`pyproject.toml` declares `dependencies = []`, so **`pip install -e .` installs
nothing**. `pip install -r requirements.txt` is mandatory — otherwise you get an
importable package with no torch and no numpy. System Python is 3.9.6, below the
project floor of 3.11.

Verified in the `.venv/` used for bring-up (AC17):

```
$ .venv/bin/python -c "import sys, torch, numpy; \
    print('python', sys.version.split()[0]); \
    print('torch', torch.__version__); print('numpy', numpy.__version__)"
python 3.11.15
torch 2.13.0
numpy 2.4.6
```

`torch` imports and computes (`torch.zeros(3).sum()` returns `0.0`), and
`sys.prefix != sys.base_prefix` confirms it really is a virtualenv rather than
the system interpreter.
