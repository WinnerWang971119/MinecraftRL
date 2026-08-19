# Watching the arena live — joining as a spectator

**What this is for:** connecting your normal Minecraft Java client to the training
server so you can *see* whether the two bots behave correctly, before committing a
machine to a training run.

**Read the whole "Before you join" section first.** Three of the four traps below
are only avoidable *before* you press Join, and one of them costs you a corrupted
episode if you get it wrong.

Everything here is the single-pad (`--pads 1`) path. It works the same at any pad
count; you just pick which anchor to fly to.

---

## Before you join — four things that will bite you

**1. You will spawn in survival, inside pad 0, next to a bot holding an iron sword and
wearing a full iron set.**

`server.properties` sets `gamemode=survival` **and** `force-gamemode=true`, so every
join forces you to survival regardless of what you were last time. The world spawn is
`0 64 0` (`arena:setup` runs `setworldspawn 0 64 0`), which is pad 0's **learner spawn
cell** — the exact block the combat bot is teleported onto every reset. You will be
standing in the fight.

That is not just uncomfortable, it corrupts the run — and worse than you would guess,
because **neither damage channel records who caused the damage.** `damage_dealt` is
"the dummy's own health went down", from any cause. `damage_taken` is "the learner's
own health went down", from any cause. So a survival human inside pad 0 writes
directly into the reward function: punch the dummy and you **pay the learner a reward
it did not earn**; hit the learner and you write a penalty it did not deserve. Every
episode you are present for in survival is suspect and should be thrown away.

**So: have this typed into the Paper console before you press Join, and hit Enter the
moment you are in.**

```
gamemode spectator <your-mc-username>
```

You cannot pre-issue it — the command needs an online player. Type it, leave the
cursor at the end of the line, join, then Enter.

`force-gamemode` applies **at join**, so this is not a one-time fix: **re-issue it
after every reconnect**, including after a client crash or a `/kick`.

Spectator is the right mode for three more reasons: spectators cannot be hit, cannot
hit, and pass through blocks — so you can fly straight through the bedrock ring
instead of looking for a door (there isn't one), and, less obviously, **you stop
shoving the learner around**. A survival player standing on the world spawn is
standing *in* the learner's spawn cell, and player-player collision will push the
learner off its mark with no damage recorded anywhere. That is precisely the symptom
checklist item 2 below teaches you to read as a reset bug.

**2. The bots do not move on their own.**

Paper plus the bridge gives you two bots standing still. The bridge only executes
actions it is told to execute; the *policy* lives on the Python side. With nothing
driving it you will watch two idle players do nothing forever and conclude the stack
is broken when it isn't.

The driver for this purpose is the random-policy tracer:

```bash
.venv/bin/python -m eval.run_random --episodes 20 --host 127.0.0.1 --port 5555
```

This is also **acceptance criterion AC10** (TC9): 20 episodes, **zero crashes _and_ a
nonzero mean `r_damage_dealt`**. So the run you watch is the run that collects the
acceptance — do not treat it as throwaway. `run_random` reports the crash half itself
and says nothing about damage; the damage half comes from what you see in the world
and, definitively, from `eval.combat_probe` (below).

**Budget about 40–45 minutes for those 20 episodes**, not a few minutes. An episode is
capped at `MAX_EPISODE_STEPS = 600` decisions at `DECISION_INTERVAL_MS = 200`, i.e.
**120 seconds**, and a random policy mostly times out rather than killing — so
20 × 120 s = **40 minutes**, plus a reset between each.
Drop `--episodes` if you only want a look; keep it at 20 if you want the acceptance.

**Do not Ctrl-C at minute 27.** This page used to say the cap was 400 decisions and
budget 27 minutes, which was right when `MAX_EPISODE_STEPS` was 400. It is 600 now, so
a healthy run is still going a good 13 minutes after the old budget says it should
have finished. Killing it there voids AC10 and you get to run it again.

**3. Your normal Minecraft client works. You do not need a cracked client.**

`online-mode=false`, so the server accepts a plain username and derives an offline
UUID from it. Point the vanilla launcher at:

```
localhost:25565
```

Client version must be **1.21.1** (the pinned server build). Loopback is also exempt
from CraftBukkit's connection throttle — measured, see
[`server/compat_check.md`](../server/compat_check.md#connection-throttle--measured) —
so reconnecting repeatedly from this machine will not get you throttled.

**4. You will not see much terrain, and that is deliberate.**

`view-distance=2` and `simulation-distance=2` are tuned for training throughput, not
for looking around: roughly 32 blocks of loaded world and blackness past it. A 25×25
pad fills most of what you can see, which is fine — the pad is the only thing worth
looking at. Everything outside it is empty flat world.

**Do not raise these values to get a better view.** They are load-bearing for the
≥19 TPS gate the scale ladder is measured against, and `setup.sh` regenerates
`server.properties` anyway, so a hand edit silently disappears on the next setup run.

---

## Boot sequence

The run order in this project is always **Paper → bridge → Python driver**, and it is
a hard rule: the bridge's bots connect to Paper *before* the bridge opens its TCP
port, and the driver connects to that port. Start them out of order and you get a
bridge that exits with `ECONNREFUSED`.

You need **four terminals plus the game client** — every numbered block below is a
separate window, and each one is self-contained (paste it anywhere). Terminal 0 exits
when setup finishes, so you can reuse that window for anything later. Terminal 1 is
special: it must be the *interactive* Paper console, because that is where you type
`gamemode spectator`. Join the world **before** you start terminal 3 — the client step
below sits between terminals 2 and 3 on purpose.

### Terminal 0 — one-time setup (idempotent, run it anyway)

```bash
cd /Users/diego/Documents/MinecraftRL
bash server/setup/setup.sh
```

It re-verifies the Paper jar's sha256 on every run, rewrites `server.properties` and
`bukkit.yml`, and **re-copies `server/arena/` into `world/datapacks/arena/`**. That
last one matters: Paper only ever loads the copy under `world/`, and the arena
functions changed on this branch. If you skip this and the world copy is stale,
terminal 2's preflight refuses to launch and tells you so.

### Terminal 1 — Paper, with a console you can type into

```bash
cd /Users/diego/Documents/MinecraftRL
bash server/setup/start.sh
```

Wait for `Done (…)! For help, type "help"`.

`start.sh` pins **Java 21** and refuses to launch on anything else. That refusal is
deliberate: Paper 1.21.1 boots fine on Java 26 and *then* dies with a native SIGSEGV
inside the bundled spark profiler seconds later, which looks like a bridge fault.
If it refuses, `brew install --cask temurin@21`.

> **Do not start Paper through `start-pads.sh` for this.** That script runs Paper with
> its output redirected to `server/logs/pads/paper.log` and its stdin on `/dev/null` —
> there is no console to type into, and `gamemode spectator` is the first thing you
> need. Start Paper here, and attach the bridge to it in the next step.

The console noise on a healthy boot (offline-mode banner ×4, "no advanced terminal
features", the "not updated in a while" nag, and `ERROR: No key layers in MapLike[{}]`
at world creation) is all expected and documented in
[`server/compat_check.md`](../server/compat_check.md#java-version--measured-supersedes-21).

### Terminal 2 — the bridge, attached to the running JVM

```bash
cd /Users/diego/Documents/MinecraftRL
bash server/setup/start-pads.sh --pads 1 --no-server
```

`--no-server` means "Paper is already up, do not start one" — which is exactly the
situation, since terminal 1 owns it. The script still runs its full preflight (node,
datapack currency, `max-players`, ops, port occupancy), starts the pad's bridge, waits
for both bots to join, resets the pad once, and prints:

```
[start-pads] FLEET READY: 1 pad(s) primed, 2 bots placed at their anchors.
```

**Wait for `FLEET READY` before going any further.** The driver connecting mid-prime
steals the bridge's single TCP client slot, and the bridge accepts exactly one.

> `FLEET READY` means "the bots were placed", **not** "the arena exists". A reset ack
> cannot see walls. That is open issue **#27** — a pad's geometry could be silently
> absent (a `/fill` into an unloaded chunk no-ops without an error). The fix
> (forceloading the pad footprint during `arena:setup_pad`) has landed but has **not
> been live-verified**, which is one of the things you are in the world to check.
> There is a checklist item for it below.

### The game client — join, and get out of survival (before terminal 3)

**Do this before you start the driver, not after.** Right now both bots are standing
idle and nothing is swinging, so the window where you are a survival player inside pad
0 is as harmless as it will ever be. Once the driver is running it is not.

1. Launch the Minecraft Java **1.21.1** client, Multiplayer → Direct Connection →
   `localhost:25565`.
2. **Immediately** press Enter on the pre-typed `gamemode spectator <your-mc-username>`
   in terminal 1.
3. Optionally turn on the health readout and fly to a vantage point (both below).

### Terminal 3 — the driver

```bash
cd /Users/diego/Documents/MinecraftRL
.venv/bin/python -m eval.run_random --episodes 20 --host 127.0.0.1 --port 5555
```

One line per episode: outcome, length, total reward, running win-rate. At the end it
prints two `[done]` lines and an `[rss]` line — `crashes=` is on the **first** of the
two. The process exits non-zero if anything crashed or RSS growth blew the budget.

This runs for roughly 25–30 minutes (see trap 2). Leave it alone.

---

## Where to stand

Pad 0's **anchor** is `(0, 0)`. The anchor is the learner's spawn cell, not the floor
origin. For any pad at anchor `A`:

| Thing | Position |
|---|---|
| Learner feet | `A.x + 0.5`, `64`, `A.z + 0.5` |
| Dummy feet | `A.x + 3.5`, `64`, `A.z + 0.5` |
| Floor (`smooth_stone`) | `y = 63`, `x ∈ [A.x−8, A.x+16]`, `z ∈ [A.z−12, A.z+12]` |
| Bedrock sub-floor | `y = 62`, same footprint |
| Bedrock ring | `y = 64…71`, the perimeter of that footprint, **all four corners closed** |
| Reachable interior | `x ∈ [A.x−7, A.x+15]`, `z ∈ [A.z−11, A.z+11]` |
| Ceiling | none |

So the fight happens along the x axis, from `x = 0.5` to `x = 3.5`, at `z = 0.5`,
standing on `y = 64`.

Two vantage points, both issued from the terminal 1 console. They use
`teleport … facing <x> <y> <z>`, which aims the camera at a point instead of at a yaw
angle, so there is nothing to get backwards:

```
tp <your-mc-username> 2 68 10 facing 2 65 1
```
Side-on, from inside the pad's `+z` half at eye level above the floor, looking along
the fight line. This is the one you want for judging swings and movement. Stay at
`z ≤ 11`; `z = 12` is the wall column, and standing inside bedrock is a black screen.

```
tp <your-mc-username> 2 80 0 facing 2 64 0
```
Straight down from above the wall line (the walls top out at `y = 71`). Good for
confirming the ring is closed and for seeing where the learner actually wanders.

For a pad other than 0, add the anchor: pad `i` sits at `((i % 5) * 512, (i // 5) * 512)`,
so pad 1's learner is at `(512.5, 64, 0.5)` and the side-on vantage is
`tp <your-mc-username> 514 68 10 facing 514 65 1`.

### Make the bots' health visible

Minecraft does not render other players' health bars, so "is the dummy taking damage"
is not something you can read off the screen. Turn on the vanilla health readout in
the tab list — from the Paper console, once:

```
scoreboard objectives add hp health
scoreboard objectives setdisplay list hp
```

Now hold Tab and every player's current health shows next to their name, live. The
arena datapack uses no scoreboard objectives, so this cannot collide with anything.
Remove it when you are done with `scoreboard objectives remove hp`.

For a single spot reading instead, from the console:

```
data get entity dummy_bot Health
```

---

## What "working correctly" looks like

Go through this list while the driver runs. Item 5 was the open question; it has now
been checked, it was a real defect, and the item is the confirmation that the fix took.
Item 6 is still the one thing nobody has ever checked by eye.

**1. Both bots are placed, three blocks apart, at every episode start.**
Learner at `x ≈ 0.5`, dummy at `x ≈ 3.5`, both `y = 64`, both `z ≈ 0.5`. Press F3 and
read the coordinates off the debug screen if you want to be exact. If a bot is
anywhere else at episode start, the reset did not take.

**2. The dummy never moves.**
It has no policy, and its `knockback_resistance` is set to `1.0` every reset, so
landed hits do not shove it. A dummy that drifts off its spawn cell is a defect — it
is one of the two bugs this branch exists to fix, so it is worth watching for.

**3. The learner moves, and stays inside the walls.**
Under a random policy it strafes, jumps, spins, and occasionally swings. It should
never leave the pad. The bedrock ring is 8 blocks tall (`y = 64…71`) with no gaps and
no corner holes; a player jumps ~1.25 blocks and the bots cannot place blocks.
**If you ever see the learner outside the ring or falling, that is issue #27** —
the geometry was silently not built.

**4. The dummy takes damage and dies.**
A fully-cooled iron-sword hit does **6** damage against an *unarmored* target, so a
clean kill is `6, 6, 6, 2` — four hits, exactly 20 HP. With the tab-list readout on,
the dummy's number should drop in steps and **never tick back up**:
`naturalRegeneration` is off, so any mid-episode healing is a defect. You will also see
the hurt flash and hear the hit, which is the signal to watch for if you skipped the
scoreboard setup.

> **The dummy is no longer unarmored, so the number you *watch* is not 6.** Since M4
> (issue #33) `spawn_dummy_pad.mcfunction` gives it a full iron set — 15 armor points,
> about 48% off an incoming 6, so roughly **3.12** a hit and about **7** hits to a kill.
> `eval.combat_probe` **has** been recalibrated (T23): it now derives the expectation
> from the target's loadout and expects `3.12` six times then `1.28`. It prints what it
> expects before the first cycle. An earlier revision of this box said the tool had not
> been recalibrated — that was true when written and is not now.

Expect a **mix, weighted toward timeouts**: a random policy has to still be standing
in melee range at the moment it happens to pick ATTACK, and it wanders. Kills happen;
they are not the common case, and that is not a fault. If you want certainty rather
than an impression, do not squint at this — run the combat probe, which counts every
hit and asserts exact deltas. It is the real damage-channel gate:

```bash
.venv/bin/python -m eval.combat_probe --cycles 10
```

**Stop the driver first.** The bridge accepts exactly one TCP client and resolves a
second connection by destroying the first, so the probe and `run_random` cannot both
be attached. Run it from the same terminal 3, at the repo root.

(Read the module docstring in `eval/combat_probe.py` before interpreting a red run;
there is a known first-cycle-after-boot false-fail tied to issue **#28**, the
episode-start attack-cooldown observable. That has been fixed bridge-side but is
itself **pending live verification**.)

**5. Which way do the bots face? — answered, defect confirmed, fixed. Verify the fix.**
This was an open question. It was checked, and both bots really were spawning facing
directly **away** from each other. Both spawn yaws were inverted; **T22 fixed them**,
and this item is now the live confirmation that the fix took.

The reading that found it (server console, both bots at rest after a reset, on pad 2):

```
learner_bot Rotation: [90.0f, 0.0f]   Pos: [1024.5d, 64.0d, 0.5d]
dummy_bot   Rotation: [-90.0f, 0.0f]  Pos: [1027.5d, 64.0d, 0.5d]
```

Minecraft's look vector is `(x, z) = (−sin yaw, cos yaw)`, so yaw `90` looks `−X` and
yaw `−90` looks `+X`. The learner was looking `−X` with the dummy `+X` of it, and the
dummy `+X` with the learner `−X` of it: each pointed exactly 180° from its opponent.
The datapack now uses learner yaw **−90** (looks `+X`, toward the dummy) and dummy yaw
**+90** (looks `−X`, toward the learner).

**What to check.** From the console, right after a reset and before the driver moves
anything — this is the whole test, and it does not need the game window:

```
data get entity learner_bot Rotation
data get entity dummy_bot Rotation
```

Expected, exactly: `learner_bot` → `[-90.0f, 0.0f]`, `dummy_bot` → `[90.0f, 0.0f]`.
Anything else — and especially the old `[90.0f, ...]` / `[-90.0f, ...]` pair — means
the server is running a stale datapack copy; re-run `server/setup/setup.sh`'s datapack
install step and `/reload`. By eye, the dummy should be looking back down the fight
line at the learner's spawn cell, not out over the `+X` wall.

Nothing in the automated suite can confirm this for you. `bot.attack()` does not
require the attacker to be facing its target, so the combat probe passed the entire
time the bots were back-to-back. What the bug actually cost: `r_aim` pays out only
when the opponent is both visible **and** in the crosshair, so every episode opened
with the one dense shaping term reading zero until the agent turned ~180°, and the
dummy — which never turns on its own — faced away for the whole episode, every
episode. With the fix, both bots start eye-to-eye at 0° of crosshair error.

**6. Is the bedrock ring actually closed?**
Fly to each of the four corners — `(−8, −12)`, `(−8, +12)`, `(+16, −12)`, `(+16, +12)`
relative to the anchor — and check there is bedrock from `y = 64` up to `y = 71` in
each corner column. This is the live half of AC7 that has never been run, and closed
corners are the classic thing four-wall builders get wrong.

**7. Both bots snap back on reset.**
At the end of an episode the dummy respawns instantly (`doImmediateRespawn`) at its
own pinned spawnpoint, not at the world spawn, and both bots are teleported back to
their spawn cells with full health and full hunger for the next episode. Watch two or
three episode boundaries; the placement should look identical every time.

**8. Things you should NOT see.**
- Any player other than you and the two bots.
- A bot at `y ≈ −60`. That means it left the pad and landed on the flat world's ground
  (`grass_block` at `y = −61`). With `fallDamage` off this is **never a death** — the
  agent is simply stranded alive for the rest of the episode. Full column scan and
  consequences in
  [`server/compat_check.md`](../server/compat_check.md#block-composition-below-the-arena--ac17-confirmed).
- Any mob, item drop, weather, or day/night change.

One thing you will only find in a log, not in a terminal: once per reset the bridge
writes any player in the learner's view that is not one of this pad's two bots as
`[bridge] pad 0 foreign_players <name>`. Under the procedure above that goes to
`server/logs/pads/pad-0.log`, not to terminal 2 — `start-pads.sh` redirects each
bridge's output to its own file. It will probably list you: the scan reads
`learner.entities`, and the server still tracks and transmits a spectator's player
entity (clients hide them; the server does not withhold them). Either way it is the
cross-pad isolation observable noticing you, not a bug.

---

## Shutting down cleanly

In this order:

1. **Terminal 3** — let the 20 episodes finish, or Ctrl-C.
2. **Your client** — disconnect (Esc → Disconnect). Do this before the server goes
   down, so you leave cleanly rather than timing out.
3. **Terminal 2** — Ctrl-C. Its teardown stops the bridge it started. In `--no-server`
   mode it never started Paper, so it will not stop Paper.
4. **Terminal 1** — type `stop` in the Paper console and wait for it to save the world
   and exit. Do not Ctrl-C or `kill -9` the JVM; that risks an unsaved world.

---

## Do not do these

- **Do not load a checkpoint from `runs/` to "show the trained bot".** Every archived
  checkpoint was trained with the opponent-damage channel dead — `r_damage_dealt` was
  exactly `0.0` across all 453 recorded episodes in the Windows archive
  ([`docs/analysis/2026-08-10-windows-archive.md`](analysis/2026-08-10-windows-archive.md))
  — and against a dummy whose health drifted across episodes. Whatever those policies
  learned, they learned in a regime where landing hits paid nothing. Showing one would
  misrepresent what "the bots working correctly" means. **A random policy against a
  repaired damage channel is the honest demonstration right now.** There will be
  something worth showing after the post-repair re-baseline (T14).

- **Do not join during a benchmark or ladder run.** Your client is another player the
  server ticks and streams chunks to, and TPS / round-trip latency are the entire
  point of those runs. Watch during `run_random` or `combat_probe`, never during
  `eval.benchmark`.

- **Do not edit `view-distance` / `simulation-distance` / `max-players` by hand.**
  `setup.sh` regenerates `server.properties`, so the edit vanishes on the next setup
  run, and the first two are inputs to the TPS gate.

---

## When it goes wrong

| Symptom | Cause and fix |
|---|---|
| `start.sh`: `REFUSING TO LAUNCH: Java 26 is not the pinned Java 21` | Working as intended. `brew install --cask temurin@21`. |
| Terminal 2: `--no-server was given but nothing answers on mc port 25565` | Terminal 1 has not reached `Done` yet. Wait, then re-run. |
| Terminal 2: `the installed datapack differs from server/arena` | The world's copy is stale. `bash server/setup/setup.sh`, then restart Paper (terminal 1: `stop`, then `bash server/setup/start.sh`). |
| Terminal 2: `bridge port(s) already in use: 5555` | A previous fleet is still up. Find and stop it before relaunching. |
| Terminal 3: `ConnectionRefusedError` / `BridgeError` on the first reset | The bridge is not listening yet. Wait for `FLEET READY`. |
| Client: "Connection refused" | Paper is not up, or you typed a port other than 25565. |
| Client: connects, then you are in survival taking hits | The `gamemode spectator` command did not run. Re-issue it now. Any episode you stood in has polluted `damage_taken` and/or `damage_dealt` — discard it and restart the driver. |
| A bot is missing from the world | Read `server/logs/pads/pad-0.log` — that is the bridge's own log, and a bot that failed to join or was kicked says so there. |

---

## Related

- [`RUNBOOK.md`](../RUNBOOK.md) — the full go-live procedure this is a side door into.
- [`server/compat_check.md`](../server/compat_check.md) — the authority for the Java
  pin, the world's block composition, the connection throttle, and the attribute IDs.
  Every "measured" claim above traces back to it.
- [`server/arena/README.md`](../server/arena/README.md) — pad geometry, the spawn and
  gear template, and the datapack macro functions the reset is made of.
