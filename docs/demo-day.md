# Demo day — running the human exhibition

**What this is:** a classmate joins the Minecraft server over the school LAN and
fights the trained agent one-on-one. One command starts everything. After that,
**anyone types `reset` in Minecraft chat** to arm the next challenger — no
terminal, no alt-tab, which is what makes a queue of people workable.

**Who this is for:** the person operating the demo, standing up, with a queue of
people waiting. Read it once the night before. On the day, work down
[The run](#the-run) and use [When something looks wrong](#when-something-looks-wrong)
as a lookup table.

---

## Say this out loud before the first match

The agent's **senses are fair. Its turning is assisted.** Both halves matter and
both are true.

- **Fair:** what the agent *sees* goes through a 70° field-of-view cone, a
  line-of-sight raycast, and a memory timeout ([`env/perception_filter.py`](../env/perception_filter.py)).
  It genuinely cannot see behind itself. Circle behind it and it loses you.
- **Assisted:** one of its eight actions is called `TURN_TO_LAST_SEEN`, and the
  name is wrong. It does not recall where you were last *seen*. The bridge
  (`bridge/bot.js`, `_updateLastSeen()`) writes your **live** world position
  into that memory on every decision window, whether or not the agent can see
  you. So when the agent picks that action it snaps its aim onto where you
  actually are, about 200 ms ago.

In short: the agent can be blinded, but it cannot be *dodged* once it decides to
turn.

This was a deliberate call made under deadline, not something discovered
afterwards. The two honest alternatives (gate the memory on real visibility, or
resolve it properly and add real facing actions) both invalidate the trained
checkpoint, and there was one training window left before the 20th. It is
written down in the action's own docstring in
[`agent/actions.py`](../agent/actions.py), frozen through 2026-08-20, and
scheduled for removal right after the demo.

If a teacher asks "is it cheating?", the accurate answer is: *its perception is
honestly limited, its aim is not, and that is documented in the source.*

---

## The night before

Everything here is offline. None of it needs a person to fight.

**1. Check the toolchain.**

```bash
/usr/libexec/java_home -v 21
```

Must print a path. Paper 1.21.1 needs **Java 21**. This Mac's default `java` is
26, and that is fine: `server/setup/start.sh` resolves 21 itself. Java 26 boots
Paper and then kills the JVM with a SIGSEGV inside spark's profiler about 20
seconds later, long after the console says the server is ready, so it reads like
a bridge fault. If the command above prints nothing:

```bash
brew install --cask temurin@21
```

**2. Check Python.**

```bash
.venv/bin/python --version
```

Must be 3.11.x. The system python on this Mac is 3.9.6 and will not run this
code. If the venv is missing:

```bash
python3.11 -m venv .venv
```

```bash
.venv/bin/pip install -r requirements.txt
```

`pip install -e .` installs **nothing** on its own. `pyproject.toml` declares no
dependencies, so `requirements.txt` is the only thing that pulls numpy, torch
and pytest. Running `-e .` afterwards is optional and only puts the packages on
the path by name.

**3. Install the server and the datapack** (idempotent, safe to re-run):

```bash
bash server/setup/setup.sh
```

**4. Rehearse the launch without launching anything.** `--dry-run` runs every
refusal gate, prints the resolved plan, and starts no process. Exit code 0 means
the checkpoint loads, both ports are free, and the Node toolchain resolves.

```bash
.venv/bin/python -m deploy.exhibition --challenger-username demo_player --dry-run
```

**5. Gear the pinned name once, for the first match only. Five commands, not one.**

Every reset arms the challenger with an iron sword **and a full iron set**, so
from the second match on this is handled. Match 1 is the exception: it starts
before anybody has joined, so there is nobody to arm yet. Join under the pinned
name during a standalone Paper boot and run all five lines — one `give` for the
sword and four `item replace` lines for the armor, because **`give` cannot
equip armor**. The exact commands, and why that distinction is the difference
between an armored challenger and one carrying armor in their bag, are in
[Gear](#gear-an-iron-sword-and-a-full-iron-set-both-sides). Do it tonight:
during an exhibition there is no server console to type into.

**6. Rehearse the chat keyword once, in game.**

Start the exhibition for real, join under the pinned name, and type `reset` in
chat. You are looking for one line back:

```
<learner_bot> reset armed - next match starting
```

Worth the two minutes. Everything else in this guide is covered by automated
tests, but "the bot receives a player's chat line" can only be proved against a
live server — mineflayer's chat plugin has silently gone dead on this Minecraft
version before (the `rl_deaths` scoreboard workaround exists for exactly that
reason). Confirm it tonight and the keyword is a known quantity tomorrow. If the
line never comes, nothing is broken for the demo itself: fall back to
`.venv/bin/python -m deploy.exhibition --reset` from a second terminal.

---

## The run

**One challenger at a time, start to finish.** The bridge accepts exactly one
TCP client and the exhibition claims exactly one challenger slot. This is not a
queueing system.

### 1. Start it

```bash
.venv/bin/python -m deploy.exhibition --challenger-username <their_mc_name>
```

This starts Paper, waits for the Minecraft port, starts the bridge in human
opponent mode, connects the agent playing greedily from
`runs/m2_multi.pt` (override with `--checkpoint`), and prints the join address.
It runs in the **foreground for the whole exhibition**. Leave the terminal open.

**Pin `--challenger-username`. It is the recommended default, for two separate
reasons:**

- Without it, the bridge credits the **first non-agent player who enters the
  pad**. A bystander who dies to anything at all, fall damage, another player,
  their own mistake, gets reported as the agent's win. The launcher prints a
  `WARNING` at startup when you leave it unset.
- Without it, the challenger **cannot be healed between matches**. Nothing on
  the wire reports who claimed the slot, so the launcher does not know whose
  name to heal. It says so explicitly at reset time and skips the heal. That
  goes wrong exactly when it hurts most: after a match the *agent* lost, the
  human carries their leftover health into the next round.

### 2. Get the challenger in

The launcher prints a banner:

```
[exhibition] JOIN AT: 192.168.x.x:25565  (same LAN, offline-mode -- any username)
[exhibition] ARENA: pad 0, anchor (0, 0) -- the world spawn, so a fresh join should land right there.
```

The server is offline-mode, so any username works. It must be **exactly** the
name you pinned. They join in **survival** (`force-gamemode=true`, so it is
forced on every join) and land at the world spawn, which is inside the arena.

Match 1 starts the moment the agent connects, before anybody has joined. Until
someone claims the slot the agent stands in the pad with a zeroed opponent
reading. That is normal.

### 3. Watch the match

The match ends on a death, either side. There is **no timeout** against a human
and **no auto-restart** after a death. The launcher prints the result and waits:

```
[exhibition] match finished after N decision step(s): AGENT WIN (opponent died)
```

### 4. Arm the next challenger

**Type this in Minecraft chat:**

```
reset
```

Press `T`, type `reset`, press Enter. `!reset` works too, and case does not
matter (`RESET` is fine). The bot answers in chat:

```
<learner_bot> reset armed - next match starting
```

If you do not see that reply, nothing was armed. Type it again.

**Anyone can type it** — the challenger, the next person in the queue, a
spectator. There is no permission check, on purpose: the whole point is that
nobody has to reach a keyboard outside the game.

Two rules that surprise people:

- **It must be the whole message.** "how do I reset?" does nothing. That is
  deliberate: a chat line that merely *contains* the word must not end a live
  match.
- **Type it after the match ends**, not during. A request filed mid-match is
  discarded when that match ends (see the rules below), and you would have to
  type it again. Wait for `match finished` in the launcher terminal, or simply
  for the fight to be over.

Typing it several times in a row is harmless — a 5-second cooldown collapses a
burst (four people typing it at once, or one person impatient) into exactly one
reset and one reply.

**The terminal command is the fallback**, and does exactly the same thing. From
a second terminal, in the repo:

```bash
.venv/bin/python -m deploy.exhibition --reset
```

Use it when nobody is in game yet, or when chat is not cooperating. It starts
nothing and never connects to the bridge.

**Both triggers are one mechanism.** Each writes the same one-shot request file
at `server/logs/exhibition/reset.request`, which the running launcher picks up
within about a second — the chat keyword makes the *bridge* write it, since the
bridge is already in game. The launcher then heals, repositions and re-gears the
human through Paper's console — sword plus all four worn armor pieces — reads
the gear back off the server and reports what it found, resets the learner (the
datapack re-arms that side), releases the challenger slot, and plays **exactly
one more match**. It cannot tell which trigger fired.

**The next person joins under the same pinned username.** The pin is baked into
the bridge's command line for the life of the launcher, and `--reset` refuses to
change it. The server is offline-mode, so anyone can type any name at the join
screen, and the bridge matches it exactly, case included. Running a queue means
everyone takes their turn as the same name. Changing the pin means Ctrl-C and
relaunch.

Rules worth knowing before you are standing in front of people:

- **One reset, one match.** It never chains.
- `--reset` takes **only** `--log-dir`. Every other flag is refused, because
  everything else was decided when the launcher started. Changing the pinned
  challenger means restarting the exhibition.
- A reset filed **while a match is still running** is discarded when that match
  ends, loudly, and you trigger it again. A death must never be what restarts
  the match. This applies to the chat keyword exactly as it does to `--reset`.
- If you pass a non-default `--log-dir` to the launcher, pass the same one to
  `--reset`. The launcher prints the exact command to use. The chat keyword
  needs nothing: the launcher hands the bridge the path it is polling.
- The chat keyword is **exhibition-only**. Training runs the bridge in bot
  opponent mode, where the keyword is ignored outright, so a stray `reset` typed
  into a training server cannot perturb a run.

### 5. Shut down

`Ctrl-C` in the launcher terminal. That tears down the agent connection, the
bridge, and the Paper JVM, in that order, and exits **130**. A second `Ctrl-C`
during the wait is absorbed on purpose, so double-tapping will not orphan the
JVM on port 25565.

---

## Is it working? The one-line check

On the **first boot** of an exhibition, look in `server/logs/exhibition/bridge.log`
for this substring:

```
rl_deaths objective NOT confirmed
```

**Its absence is the confirmation.** A healthy read-back says nothing. Human win
detection reads deaths off the `rl_deaths` scoreboard objective on raw client
packets, and this line is the bridge telling you the server never acknowledged
the objective, meaning detection is armed but unverified. If you see it, the
named causes are that the learner is not opped or that the two scoreboard
commands were not accepted.

That check plus one real kill covers the whole death-detection path.

**The gear has a separate, per-reset check, and it works the other way round —
it speaks up when it is happy.** Every reset prints either a
`server-authoritative read: ... all four iron armor pieces WORN` confirmation or
a `COULD NOT CONFIRM n of 5 gear slot(s)` block. It is best-effort and never
stops a match, so read it every time:
[Gear](#gear-an-iron-sword-and-a-full-iron-set-both-sides).

---

## When something looks wrong

### Expected. Not bugs. Do not react to these mid-demo.

| What you see | Why |
|---|---|
| `rl_deaths` sitting in the tab player list, counting up | The objective must be display-bound or the server broadcasts no score packets at all. The tab entry is the price of death detection working. |
| The **first** reset logs the challenger on a `foreign_players` line | The exclusion only covers a *claimed* challenger, and on the first reset nothing has claimed yet. `eval/benchmark.py` reads that line as cross-pad contamination evidence, so an exhibition log looks contaminated to that tool. It is not. |
| `No player was found` in the console after a reset | A pinned challenger who is not online right now. All ten heal, reposition and gear commands run against a name nobody is holding. Harmless. |
| `COULD NOT CONFIRM 5 of 5 gear slot(s)` on a reset where nobody was online | The same cause as the row above: the five gear read-backs also find nobody, so none of the five can be confirmed. Expected while the pad is empty. On a reset where the challenger *is* in game, it is not — see the next table. |
| `reflex shield overrode the action on 4192/4200 decision step(s)` | The shield counts the idle wait before anyone joins, where the opponent reading is zeroed and therefore "blind". The number is dominated by waiting, not by the fight. The summary line is noise, not a verdict on the match. |
| The agent stands still before a challenger joins | No claimant means a zeroed opponent block and no last-seen memory to turn toward. It wakes up when someone claims the slot. |

### Real problems

| Symptom | Fix |
|---|---|
| `bridge port 5555 ... is already in use` and nothing started | Something else holds the port: another exhibition, a training run, eval tooling. Stop it. The bridge accepts one client and a second connect destroys the first, which is why the launcher refuses instead of half-starting. |
| `something is already listening on Minecraft port 25565` | Usually a Paper JVM left behind by an interrupted launcher. The refusal prints the recovery command: `lsof -ti:25565 \| xargs kill`. |
| `checkpoint not found` plus a list of what does exist | Pass `--checkpoint` with a real path. The launcher never falls back to an untrained agent for a demo. |
| `cannot start: the bridge toolchain does not resolve` | `node` is not on PATH, or `bridge/run.js` / `server/setup/start.sh` is missing. Pass `--node <path>` if Node is installed somewhere unusual. |
| Paper reports Done and then dies ~20 s later | Wrong Java. See [The night before](#the-night-before). |
| Challenger joined but the agent ignores them | Name mismatch against `--challenger-username`. Offline-mode names are case-sensitive and exact. Restart the exhibition with the right name. |
| Match will not end | Correct. There is no timeout against a human. Somebody has to die. |
| Typing `reset` in chat gets no reply | Check it was the whole message and nothing else — "reset?" and "reset now" do not count. If a plain `reset` still gets nothing, `server/logs/exhibition/bridge.log` says `in-game chat reset armed:` at startup when the feature is on; use `.venv/bin/python -m deploy.exhibition --reset` and carry on. |
| The reply came but the match did not restart | The request was filed while the previous match was still running, so it was discarded when that match ended (the launcher terminal says so). Type `reset` again now that it has. |
| `COULD NOT CONFIRM n of 5 gear slot(s)` while the challenger **is** in game | The server did not confirm those pieces. It is not proof they are missing, but it is the case where a human fights an armored agent unarmored, so spend the ten seconds: the block prints the exact `data get entity` lines — run them from the console or an opped account. If a piece really is absent, type `reset` in chat again; `item replace` overwrites, so re-issuing is safe. This never blocks the match on its own — see [Gear](#gear-an-iron-sword-and-a-full-iron-set-both-sides). |
| The challenger says they are unarmored and it is match 1 | Pre-gearing was skipped or only the `give` was run. `give` puts armor in the bag, it does not wear it. Finish the match, then type `reset` — the reset equips all five properly. |

---

## Gear: an iron sword and a full iron set, both sides

**Both fighters carry one iron sword and wear a full set of iron armor** —
helmet, chestplate, leggings, boots. Five items each. The two sides are armed by
different machinery that emits the same five lines.

The agent's comes from the datapack, every reset
([`spawn_learner_pad.mcfunction`](../server/arena/data/arena/function/spawn_learner_pad.mcfunction);
the opponent bot's [`spawn_dummy_pad.mcfunction`](../server/arena/data/arena/function/spawn_dummy_pad.mcfunction)
carries the identical block):

```
$give $(learner) minecraft:iron_sword 1
$item replace entity $(learner) armor.head with minecraft:iron_helmet
$item replace entity $(learner) armor.chest with minecraft:iron_chestplate
$item replace entity $(learner) armor.legs with minecraft:iron_leggings
$item replace entity $(learner) armor.feet with minecraft:iron_boots
```

The challenger's comes from the reset itself — whichever way it was triggered —
which sends the same five, plus a scoped `clear` in front of the sword, through
Paper's console (`human_gear_commands()` in
[`deploy/exhibition.py`](../deploy/exhibition.py)):

```
clear <their_mc_name> minecraft:iron_sword
give <their_mc_name> minecraft:iron_sword 1
item replace entity <their_mc_name> armor.head with minecraft:iron_helmet
item replace entity <their_mc_name> armor.chest with minecraft:iron_chestplate
item replace entity <their_mc_name> armor.legs with minecraft:iron_leggings
item replace entity <their_mc_name> armor.feet with minecraft:iron_boots
```

**`give` cannot equip armor. That is the one thing to remember on this page.**
`give` pushes a stack into the first free *inventory* slot and stops there;
nothing in Minecraft moves a piece from a player's bag onto their body. Four
`give`s leave a challenger *carrying* a full iron set at **zero** armor points —
armored in the chat log, naked in the fight, taking 6 a hit where the agent
takes 3.12 through its own iron. `item replace entity <name> armor.<slot> with
<item>` writes the equipment slot itself, so the piece is worn the instant the
command returns.

The two halves are repeat-safe for different reasons, which is why the `clear`
sits in front of the sword and nowhere else:

- The four armor lines **overwrite** the slot rather than appending to a bag, so
  an evening of resets leaves exactly one set, always at full durability. They
  need no `clear`.
- A bare `give` **stacks**, so the sword needs `clear <name>
  minecraft:iron_sword` immediately before it. That clear is scoped to the sword
  on purpose: a blanket `clear <name>` would empty a person's unrelated items,
  which is no part of healing them between matches.

Earlier versions of this page said neither side gets armor, on the grounds that
the agent had never trained against an armored opponent. That reasoning now runs
the other way: the agent is re-geared with a full iron set on every reset, so it
is the *unarmored human* who is the mismatch.

| Symptom | Fix |
|---|---|
| A challenger breaks their sword mid-match | Nothing to do. Arm the next challenger; the reset hands out a fresh one. |
| A challenger's armor breaks | Same. `item replace` writes a fresh piece into the slot on the next reset. |
| A challenger is empty-handed, or in their own clothes | Match 1, no pinned name, or the console off (`--no-paper-console`). The first two are below. |

**Arming needs the pinned `--challenger-username`.** Nothing on the wire tells
the launcher who claimed the slot, so an unpinned exhibition prints a warning at
reset time and arms nobody — the same reason [step 2](#2-get-the-challenger-in)
wants the pin.

**Match 1 of a launch is not armed, and that now costs five items rather than
one.** The launcher arms on a reset, and match 1 starts the moment the agent
connects, before anybody has joined — arming a name nobody is holding would only
print `No player was found`. A reset filed while match 1 is still running is
discarded on purpose, so it cannot be used to catch up. Either the first
challenger fights an armored agent barehanded and unarmored, or you gear the
pinned name ahead of time from the standalone Paper console (`bash
server/setup/start.sh`, RUNBOOK Step 1), which is interactive:

```
give <pinned_name> minecraft:iron_sword 1
item replace entity <pinned_name> armor.head with minecraft:iron_helmet
item replace entity <pinned_name> armor.chest with minecraft:iron_chestplate
item replace entity <pinned_name> armor.legs with minecraft:iron_leggings
item replace entity <pinned_name> armor.feet with minecraft:iron_boots
```

**Run all five.** The `give` is the sword only, and it mirrors the datapack's own
`$give`. If you stop there — as this page used to tell you to — match 1 is an
unarmored human against an armored agent, which is the exact failure this page
exists to prevent. Do not substitute `give <pinned_name> minecraft:iron_helmet 1`
for the second line: it puts a helmet in their bag and leaves their head bare.

`give` and `item replace` both need an **online** player, the same way `tp` and
`effect` do, so someone has to be joined under that name at the time — you, the
night before, is fine. `keepInventory` is on and inventories are stored per
username, so the sword and the worn set are still there when the exhibition
starts and survive every death after that.

Confirm it before you log off, from the same console:

```
data get entity <pinned_name> Inventory[{Slot:103b}].id
```

That is the head slot. `103, 102, 101, 100` are head, chest, legs, feet — a
player's worn pieces live in the `Inventory` NBT list at `slot + 100`, not in
`ArmorItems`, which is a mob tag and would read empty on a perfectly armored
person. A reply naming `minecraft:iron_helmet` means it is on their head; no
reply, or an empty one, means it is not.

There is still no way to hand out gear **live**. During an exhibition Paper's
console stdin belongs to the launcher process, and `server/ops.json` is
rewritten to exactly `learner_bot` and `dummy_bot` before Paper boots, so no
human account can be opped and nobody in the world can run a command. A reset is
the only gear channel once the exhibition is up, which is exactly why it arms on
every single one — all five items, every time.

### Every reset says whether the gear went on. Read the line.

After the gear commands, the reset asks the server what the challenger is
actually wearing: five `data get entity` reads down the same console pipe, then
a scan of Paper's log for the answers. Paper runs console input in order, so the
reads observe the world the re-gear left behind rather than racing it. You get
one of exactly two things in the launcher terminal.

Good — all five confirmed:

```
[exhibition] server-authoritative read: <name> has all four iron armor pieces WORN and an iron sword in their inventory.
```

Bad — one or more unconfirmed:

```
[exhibition] COULD NOT CONFIRM 2 of 5 gear slot(s) for <name>: armor.head (minecraft:iron_helmet), armor.chest (minecraft:iron_chestplate).
```

followed by the exact `data get entity` lines for the pieces it could not
confirm, so you can check them by hand from the console.

**This check is best-effort, not fail-closed. It does not stop the match.**
The two bots have a real gate: the bridge polls each bot's own connection for
its worn armor and main-hand item on every reset and will not confirm a reset
that does not match (`loadoutFailures` in [`bridge/bot.js`](../bridge/bot.js)).
A human has no bot connection, so there is nothing to gate on — and a demo that
refuses to start because a helmet is unconfirmed is worse than one that starts
and says so. This check therefore logs loudly and plays anyway, which means
**you have to read the line**. Nothing downstream will stop you if you don't.

**"Could not confirm" is not "is missing."** A piece that silently failed to
equip and a reply that was late, lost, or worded differently by some future
Paper build look identical from here, and the message says so. Treat it as
"unknown, and worth ten seconds", not as proof. If a hand-run `data get` shows
the piece really is absent, type `reset` in chat again — `item replace`
overwrites, so re-issuing is safe.

**One honest limit: the sword is confirmed as OWNED, not HELD.** `give` picks a
free inventory slot, not the selected one, and which slot a person holds is
theirs to choose. The read asks "does the server say they own an iron sword",
which is the strongest true claim available. If a challenger is swinging their
fist, tell them to select the sword.

---

## Exit codes

| Code | Meaning |
|---|---|
| `0` | `--dry-run` passed every gate and started nothing, or `--reset` armed the request. |
| `1` | Any refusal or boot failure: port in use, checkpoint missing or unloadable, toolchain missing, `--reset` with no launcher to talk to, `--reset` when one is already armed, `--reset` passed flags it cannot honor. |
| `130` | `Ctrl-C`. The normal way an exhibition ends. |

---

## Flags that exist

Confirmed against `deploy/exhibition.py`'s parser. Everything has a working
default; on a normal demo day you pass `--challenger-username` and nothing else.

| Flag | Default | What it does |
|---|---|---|
| `--challenger-username <name>` | unset | Pins the human opponent. **Pass this.** |
| `--checkpoint <path>` | `runs/m2_multi.pt` | The trained network to play from. |
| `--checkpoints-dir <dir>` | `runs` | Directory listed in the refusal message when the checkpoint is missing. |
| `--reset` | off | Arm the next challenger in a running exhibition, then exit. Takes only `--log-dir`. |
| `--dry-run` | off | Run every gate, print the plan, start nothing. |
| `--no-paper-console` | console on | Do not open Paper's stdin. Resets then print the heal, gear and gear-check commands instead of running them, for you to run from an opped account. An escape hatch. |
| `--mc-host` / `--mc-port` | `127.0.0.1` / `25565` | Minecraft server address. |
| `--bridge-host` / `--bridge-port` | `127.0.0.1` / `5555` | Bridge address. |
| `--node <path>` | `node` | Node executable. |
| `--log-dir <dir>` | `server/logs/exhibition` | Where `paper.log`, `bridge.log` and `reset.request` live. |
| `--server-timeout` / `--bridge-timeout` | `300` / `120` seconds | Bounded waits for each port to open. |
| `--xms` / `--xmx` | JVM default (`2G`) | Heap override passed to `start.sh`. |

The launcher's own printed reset hint says bare `python -m deploy.exhibition
--reset`. Use `.venv/bin/python` unless you have the venv activated.

---

## What to expect from the agent

Set expectations before the first match rather than after it. The first three
bullets are the honest answer to "how good is it?".

- **Check which checkpoint you are running.** The launcher prints it at startup;
  the default is `runs/m2_multi.pt`. Everything below depends on which one it is:
  a pre-M4 checkpoint and one trained by self-play are different agents, and the
  honest description of each is different.
- **Nobody has played this matchup yet.** The iron loadout is new. As this was
  written, **no armored match had been fought by anyone — bot or human.** Every
  checkpoint in `runs/` was trained bare-handed. If yours is one of those, it has
  never seen an opponent in armor and never had armor on itself, and the first
  armored fight in this project's history may be the one you are about to run.
  That is worth saying out loud rather than discovering in front of a teacher.
- **What training now points at, and what it does not yet prove.** The pipeline
  trains against a frozen *past self* that is armed and armored, and the
  opponent-velocity channel — dead for the life of the project, the same defect
  class that killed `damage_dealt` — has been repaired, so a policy trained from
  here can perceive opponent motion for the first time. **That run has not
  happened yet.** Do not describe the agent as self-play-trained unless the
  checkpoint you loaded came out of such a run. A checkpoint from the older
  regime fought a stationary dummy, under a reward repaired late, blind to
  opponent motion: it closes distance and swings. It is not a PvP expert.
- **Your movement does reach it now, but an old checkpoint will not use it.**
  A human has no bot connection to read a velocity off, so the bridge finite-
  differences your motion from two server-sent positions and puts that on the
  wire (`_challengerVelocity` in [`bridge/bot.js`](../bridge/bot.js)) — three of
  the agent's 23 observation features. A network trained while that channel was
  a stale constant learned to ignore those three inputs, so do not expect a
  pre-repair checkpoint to lead your movement or read a charge. It sees your
  motion; it was never taught to care.
- **Armor makes fights longer, not just tougher.** A fully-cooled iron-sword hit
  loses about 48% to a full iron set: roughly **3.12** through instead of 6, so a
  kill takes about **7** clean hits rather than 4. Expect matches to run
  noticeably longer than a bare-handed one would. Training caps an episode at 600
  decisions (**120 s**) for that reason — but there is **no timeout against a
  human**, so somebody still has to die.
- It cannot see behind itself, and it has a memory timeout. Circling works.
- Once it commits to turning, it turns straight at you. See
  [the disclosure](#say-this-out-loud-before-the-first-match).
- It decides once every 200 ms. Human reaction time is roughly the same, so the
  two are near parity on reaction and the agent is worse on fine control inside
  a window.
