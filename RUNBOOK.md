# Go-Live Runbook — from green tests to M1 + M2

**What we have:** the full kickoff foundation plus the damage-channel repair, and
the offline suite is green. But every offline test runs against fakes and
fixtures. The two questions the project exists to answer are still open:

- **M1 — does the plumbing work?** (AC3 zero-crash slice + AC4 the measured number)
- **M2 — does the RL stack learn?** (AC6 greedy win-rate ≥95% vs the dummy)

This runbook takes the stack live and collects those acceptances, in dependency
order. Everything below the line needs the real game running; none of it is
covered by `pytest`.

> **Just want to look at the bots with your own eyes?** That is a different, shorter
> procedure with its own traps (you join in **survival**, at **world spawn**, which is
> **inside pad 0**). Follow [`docs/spectate.md`](docs/spectate.md) instead, and read
> its "Before you join" section before you press Join.

> **Running the classroom demo?** That is [Step 7](#step-7--the-human-exhibition) here
> and, in full, [`docs/demo-day.md`](docs/demo-day.md). It is one command and it does
> not need any of the milestones below to have been collected.

---

## Read this first — what changed on this branch

**The opponent-damage channel was dead for the entire life of the project, and is
now repaired.** `damage_dealt` was computed from the *learner's entity view* of the
dummy (`entity.health`), and mineflayer never populates `health` on a non-self
entity — only on the bot's own connection. So the value was always `undefined`, the
drop was always `0`, and **landing a hit had never once paid a reward**. Opponent
damage now comes from the dummy bot's **own** connection (`dummy.on('health')`), and
the old entity-view path is deleted so the two can never double-count. This is the
single most important thing to know about this branch: every number from before it
was collected in a regime where the one action that mattered paid nothing. The
archive analysis that confirms it is
[`docs/analysis/2026-08-10-windows-archive.md`](docs/analysis/2026-08-10-windows-archive.md).

Riding along with the repair:

- **Terminal rewards are re-ordered:** win **+50**, timeout **−15**, loss **30**
  (stored positive, applied as `−30`). `RewardConfig.__post_init__` now enforces
  `−R_terminal_loss < R_terminal_timeout < R_terminal_win` plus sign semantics and
  finiteness, so a config where running out the clock beats dying is rejected rather
  than trained on.
- **Every arena is enclosed.** A pad is a 25×25 floor at `y=63` over a bedrock
  sub-floor at `y=62`, ringed by bedrock from `y=64` to `y=71` including all four
  corners. No ceiling. Reachable interior is `x ∈ [anchor−7, anchor+15]`,
  `z ∈ [anchor−11, anchor+11]`.
- **The dummy's health is stationary across episodes.** `naturalRegeneration` is off
  (set by the datapack, not `server.properties`), and the reset restores food and
  saturation as well as health — previously only health was restored, so the dummy's
  regeneration rate silently decayed run over run.
- **The topology is one Paper JVM hosting N enclosed pads**, not N JVMs. The
  PowerShell N-JVM orchestrator is gone. Pads are 512 blocks apart and addressed by
  datapack **macro functions** (`arena:setup_pad {x,z}`,
  `arena:reset_pad {x,z,learner,dummy,nonce}`).
- **The datapack is the sole reset authority.** The bridge no longer issues its own
  `/tp` + `/effect clear` + regear sequence; it sends exactly one
  `/function arena:reset_pad {…}` and then runs its read-back gate.

Two open defects to keep in mind while reading live results:

- **Issue #27 — a pad's geometry can be silently absent.** A `/fill` into an unloaded
  chunk no-ops without an error, and a reset ack does not prove walls exist. Fixed by
  forceloading the pad footprint inside `arena:setup_pad`; **not yet live-verified**.
- **Issue #28 — the episode-start attack-cooldown observable.** The wire used to read
  `attack_cooldown = 1.0` at episode start even though the reset's regear re-zeroed
  the server-side attack meter. Fixed bridge-side (the reported value is now the
  minimum of the last-swing ramp and the reset-boundary ramp); **also pending live
  verification**. It is the known first-cycle false-fail in `eval/combat_probe.py` —
  read that module's docstring before rationalizing a red probe run.

---

## Prereqs (one-time)

| Need | Check | Source |
|------|-------|--------|
| **Java 21 installed** | `/usr/libexec/java_home -v 21` | Must print a path. Paper 1.21.1 boots on newer JDKs and then SIGSEGVs inside spark's native profiler ~20 s after it reports Done. **Do not check `java -version`**: this Mac's PATH `java` is 26 and that is fine, because `start.sh` resolves 21 through `java_home` and refuses to launch on anything else. That pin is the only thing standing between you and the SIGSEGV. |
| Node ≥22 (v24.13.0 pinned) | `node -v` | `bridge/package.json` |
| Python ≥3.11 in a venv | `.venv/bin/python --version` | System python on this Mac is 3.9.6, below the floor |
| Python deps incl. torch | `.venv/bin/python -c "import torch"` | `requirements.txt` |

```bash
# Python side. NOTE: `pip install -e .` alone installs NOTHING — pyproject declares
# no dependencies. requirements.txt is mandatory.
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e .          # optional: puts the packages on the path by name

# Node side.
cd bridge && npm install && cd ..   # mineflayer + plugins (pinned)
```

Missing Java 21 on macOS: `brew install --cask temurin@21`.

Prove the checkout before going live:

```bash
.venv/bin/python -m pytest          # Python unit + integration
cd bridge && npm test && cd ..      # Node bridge suite
```

---

## Step 0 — the bridge launcher

**DONE — `bridge/run.js` + `npm start` exist.** The Python side connects to a
Node TCP bridge on `127.0.0.1:5555`. The bridge logic all lives in
`bridge/bot.js` as the `ArenaBots` class; `bridge/run.js` is the process entry
that does the four-call wiring (construct → `wireTransport()` → `connect()` →
`transport.listen()`). Two deviations from the original sketch, both deliberate:

- **Bots connect before the port opens** (`connect()` before `listen()`), so
  the Python env can only ever reach a bridge that is ready to serve a reset.
  If Paper is down, `npm start` exits 1 with `ECONNREFUSED 127.0.0.1:25565`
  instead of opening a port that leads nowhere. This is also what makes "pad `i`'s
  port is open" mean exactly "pad `i`'s two bots joined and spawned", which is how
  `start-pads.sh` gates its stagger.
- **`'error'` listeners on the transport and both bots** — an unlistened
  `'error'` event on an EventEmitter kills the Node process, and the M1 bar
  is zero crashes. Socket/bot errors are logged; the Python side retries.

`run.js` also takes `--pad-origin "<x>,<z>"` and `--pad-index <i>` (process-local,
never on the wire). Both default to pad 0 at anchor `0,0`, so a bare `npm start` is
byte-identical to the historical single-arena path.

> Run order matters and is a hard rule: **Paper first** (Step 1), then the bridge
> (Step 2) so the two bots have a server to join, then the Python driver (Steps 3+).

---

## Step 1 — boot the Paper server

```bash
bash server/setup/setup.sh    # downloads + sha256-verifies the jar, writes config, installs the datapack (idempotent)
bash server/setup/start.sh    # --nogui console; type 'stop' to shut down
```

`setup.sh` re-copies `server/arena/` into `world/datapacks/arena/` on every run.
Paper only ever loads that copy, so **re-run setup after any datapack change** or the
world keeps running the old functions. `start-pads.sh` checks for this and refuses to
launch on a stale copy.

First-boot checklist (from `server/README.md` / `compat_check.md`):
- world generates flat, no mobs, `online-mode=false`.
- the arena datapack loads (`arena:setup` runs on load, sets the gamerules including
  `naturalRegeneration false`, and builds pad 0).
- `learner_bot` and `dummy_bot` are opped at level 4 (`server/ops.json`).
- after a bot joins, `/attribute dummy_bot minecraft:generic.knockback_resistance get`
  resolves and the dummy stays put when hit. The `generic.` infix is **required**
  on the pinned Paper 1.21.1 stack — the flattening that removed it landed in
  1.21.2. A wrong attribute id inside a macro function aborts the **whole** function
  at instantiation, silently voiding the entire reset with nothing in the boot log.
  See `server/README.md` before changing this form.

The console noise on a healthy Java 21 boot (offline-mode banner ×4, no advanced
terminal features, the pinned-build nag, and `ERROR: No key layers in MapLike[{}]` at
world creation) is expected and itemized in `server/compat_check.md`.

**Do not "fix" `generator-settings={}`.** It does not parse, Paper falls back to the
default flat preset, and that fallback **is** the intended, verified world: solid
ground at `y=−61` with bedrock at `y=−64` everywhere. Combined with
`fallDamage false`, walking off a pad **strands the agent alive at `y≈−60` and is
never a death** — it shows up as a timeout, not a loss. Column scan and consequences:
`server/compat_check.md`.

## Step 2 — start the bridge

```bash
cd bridge && npm start   # the Step-0 launcher, single arena at anchor 0,0
```

Both bots should spawn and the bridge should print
`[bridge] listening on 127.0.0.1:5555, both bots spawned`.

For anything multi-pad, use `server/setup/start-pads.sh` instead (Step 4b) — it owns
`ops.json`, the stagger, and the prime barrier.

---

## Step 2b — the damage-channel gate (AC8) before you trust anything

This is the go/no-go gate for the whole branch. It drives deterministic reset/kill
cycles and asserts the exact per-hit sequence against a 20 HP dummy with regeneration
off, cross-checked against the wire's privileged `state.opponent.health`:

```bash
.venv/bin/python -m eval.combat_probe --cycles 10
```

Pass is exit code 0. Per cycle it requires the recorded per-hit `damage_dealt`
sequence `6, 6, 6, 2`, cumulative exactly 20, exactly one death, a clean post-respawn
baseline, and reconciliation with the wire at ±1 window. Exit code 2 means "no
verdict" (a transport abort), never "pass".

`--expect-anchor 512,0` points it at a non-zero pad. Read the module docstring before
concluding anything from a red first cycle — see issue #28 above.

### Step 2c — the opponent's mobility, when you run a scripted opponent

Only for runs launched with `--dummy-knockback-immune false` (the flag
`distributed/launcher.py` appends when `dummy_knockback_immune=False`). The default —
the M2 stationary dummy — needs none of this.

**Do not read `kb_resist=1.0` off the arena debug line.** `spawn_dummy_pad.mcfunction`
prints it as a hard-coded literal on every reset, and it is telling you what the
datapack just did, not what the opponent ends up with: on a scripted-opponent run the
bridge's `/attribute` override lands moments later and sets knockback resistance to
`0.0` and movement speed to `0.1`. The same caveat covers the word `idle` on that line.

Two checks, in this order:

- **The bridge read-back.** After each reset the bridge asks the server what it
  actually applied. Like the `rl_deaths` objective, **the absence of the line is the
  confirmation** — a healthy override says nothing. What you must never see:
  `[bridge] pad N minecraft:generic.movement_speed override NOT confirmed` (or its
  `knockback_resistance` twin, or an `override REJECTED` line). If one appears, the
  named causes are: the attribute id changed with the Minecraft version, the dummy is
  not opped, or the server has `sendCommandFeedback` off. It is deliberately log-only
  and never fails a reset, so nothing else will stop the run for you.
- **Hit it (AC18).** Datapack and attribute failures are silent here by tradition, so
  the read-back is corroboration, not proof. Join the pad, hit the opponent, and watch
  it get displaced; then watch it walk. A clean boot log has never proved either.

---

## Step 3 — M1 plumbing (AC3 / TC11)

Random policy, ≥100 full episodes vs the idle dummy, end to end. The bar is
**zero crashes** and combined-process RSS growth **< ~200 MB** (sampled every 10
episodes).

```bash
.venv/bin/python -m eval.run_random --episodes 100 --host 127.0.0.1 --port 5555
```

Read the printed summary: episodes completed, win/loss/timeout split, RSS growth
(`crashes=` is on the first of the two `[done]` lines). This is the first proof the
whole loop (rollout → store → sample → no-op update) survives the real bridge.

**Budget the time.** An episode is capped at `MAX_EPISODE_STEPS = 400` decisions at
`DECISION_INTERVAL_MS = 200` — 80 seconds — and a random policy mostly times out, so
100 episodes is on the order of **2¼ hours** plus reset overhead. The 20-episode
version (~25–30 min) is **AC10**, and is what
[`docs/spectate.md`](docs/spectate.md) uses to give you something to watch.

## Step 4 — M1 the number (AC4 / TC12)

The bridge spike's exit criterion is a *measured number*, not a green check:
transitions/s, p99 Node→Python round-trip at the 200 ms interval, damage-event
boundary correctness, and max pads sustaining ≥19 TPS over a ≥10-min run.

```bash
.venv/bin/python -m eval.benchmark --duration 600 --arenas 1
```

> **What "TPS" measures here.** The benchmark's server TPS is the REAL server
> tick rate, read from the learner bot's world age (`bot.time.age`, set only by
> the server's `update_time` packet) and averaged over a rolling 5 s window
> (Paper's own `/tps` is likewise a rolling average). It is decoupled from the
> client-side Mineflayer `physicsTick` timer, so a healthy ~20 TPS server reads
> ~20 even though `update_time` arrives only ~once per second. Raw collection
> throughput (transitions/s) is a SEPARATE number.

---

## Step 4b — the pad fleet (one JVM, N enclosed pads)

**NOTE on `--arenas` flags.** Two different commands have an `--arenas` flag that
means different things. `eval.benchmark --arenas N` (Step 4) **measures** throughput
from N pads and does not train. `agent.train --arenas N` (below) runs **training**
from N pads. Do not confuse them.

**Topology.** One Paper JVM, one Minecraft port (`25565`), N enclosed pads in one flat
world. Pad `i` gets bridge port `5555+i`, anchor `((i % 5) * 512, (i // 5) * 512)`,
and bots `learner_bot`/`dummy_bot` at `i == 0`, `learner_<i>`/`dummy_<i>` above.
`padAnchor(i)` has exactly one implementation, in `distributed/launcher.py`; nothing
else in the repo may compute an anchor.

### Caveats (read before a long run)

- **First boot is slow.** World generation, datapack load, bot joins, and the prime
  barrier each take real time. The first reset per pad is noticeably slower than
  steady state; let it settle before judging throughput.
- **`FLEET READY` does not mean the arena was verified.** It means 2N bots were
  placed at their anchors. A reset ack cannot see walls (issue #27).
- **Do not shrink the JVM heap to fit more pads.** RAM is not the ceiling; the
  single-threaded Paper tick is. Cutting heap only invites GC pauses that burn CPU and
  *drop* TPS. Leave `-Xms2G -Xmx2G` (kept equal). Only change it if a GC log shows
  real pressure, and then *raise* it.
- **Nothing else should be running.** The AC14 gate is a TPS floor, and background
  contention on a laptop is what moves it.

### Launch procedure

**1. Dry-run first** to print the plan and sanity-check ports, usernames, and anchors
before anything goes live:

```bash
bash server/setup/start-pads.sh --pads 2 --dry-run
```

Check the output: one shared MC port (25565), distinct bridge ports (5555, 5556, …),
distinct anchors (`0,0`, `512,0`, …) and distinct usernames (`learner_bot`/`dummy_bot`,
`learner_1`/`dummy_1`, …).

You can also verify the Python-side plan alone:

```bash
.venv/bin/python -m distributed.launcher --pads 2 --dry-run
```

**2. Run `setup.sh` once with the pad count** (idempotent; sizes `max-players` to
`2N+10` and refreshes the installed datapack):

```bash
PADS=2 bash server/setup/setup.sh
```

**3. Start the fleet** (Paper + N bridges + the prime barrier). This blocks and keeps
every process alive until Ctrl-C:

```bash
bash server/setup/start-pads.sh --pads 2
```

`--check` runs only the preflight gates (node, datapack currency, `max-players`,
ports, ops) and exits. `--help` lists every flag. The script starts Paper, then each
bridge one at a time — a pad's bridge port opens only after both of its bots have
joined, so the port coming up *is* the join gate — then resets every pad once before
printing `FLEET READY`.

**Do not start the Python driver until `FLEET READY` is printed.** Until every pad has
been reset, all 2N bots are stacked at the shared world spawn inside pad 0, and a
stepping pad will hit foreign bots and credit the damage to the wrong policy.

**4. Confirm each bridge is listening.** Use the non-connecting check — it answers the
same question and is safe at any time, including mid-run:

```bash
lsof -nP -iTCP:5555 -sTCP:LISTEN
lsof -nP -iTCP:5556 -sTCP:LISTEN
```

**Do not reach for a connect probe here.** `BridgeServer` accepts exactly **one** TCP
client and resolves a second connection by destroying the incumbent, so a
`socket.create_connection` against a bridge a driver is attached to silently kills
that driver's connection while the port stays open and everything looks fine. The
window between `FLEET READY` and starting the driver is the only one where it is
harmless, and `lsof` already covers that window too.

**5. Measure the ladder** (see the next section for what to record). Because the live
benchmark opens one real `TcpBridgeClient` on `port + i` per arena, N pads must be up
first — relaunch `start-pads.sh --pads N` to match before each rung:

```bash
.venv/bin/python -m eval.benchmark --duration 600 --arenas 4 --pad-log-dir server/logs/pads
```

`--pad-log-dir` engages the AC13 cross-pad isolation check: per-pad reconciliation of
cumulative `damage_dealt` against each pad's own dummy health loss, plus consumption
of the bridge-side `foreign_players` scan. Any foreign username in a pad's log, or a
reconciliation mismatch, means bots are reaching each other and the run is void.

> **A benchmark-only run has ZERO foreign-scan coverage.** The scan fires from exactly
> one call site — the end of `handleReset` — and `run_benchmark` never resets; it drives
> a continuous step loop. So a benchmark run collects the *damage-reconciliation* half
> of AC13 and proves **nothing** about the scan half, while still looking clean.
> **Collect AC13 against a fleet that is actually resetting** — alongside or as part of
> a training run — with `--pad-log-dir` pointed at the same per-pad logs.

> **One spurious pad failure at the tail of a timed run is expected and benign.** The
> run stops on a wall-clock deadline, so if a pad's final decision window happens to
> contain a landed hit, there is no window+1 for that hit's ±1-skewed wire-health drop
> to arrive in, and per-window reconciliation flags it as "no matching wire-health
> drop" on a perfectly healthy fleet. At 8–25 pads with one shot per rung this will
> happen. The **only** benign signature is: a single unmatched hit in a pad's **very
> last** recorded window, with the aggregate residual within that one hit's own size.
> Re-run the rung. Anything else — an unmatched hit in a non-final window, more than
> one unmatched hit, or a negative residual — is real, and the run is void. The
> per-window flag is deliberately not trimmed (trimming would drop a window of genuine
> coverage); the aggregate check already forgives exactly this case and only this case.

**6. Run multi-arena training** with the promoted N. Keep `start-pads.sh` running (or
relaunch it), wait for `FLEET READY`, then in a separate terminal:

```bash
.venv/bin/python -m agent.train --arenas 2 --port 5555 --max-episodes 10000 \
  --eval-every-grad-steps 1000 --eval-episodes 100 \
  --checkpoint runs/m2_multi.pt --run-name m2_multi
```

The `--arenas N` flag on `agent.train` engages the multi-arena `ActorPool` +
decoupled learner. Arena `i` connects to bridge port `--port + i`. The N bridges must
already be listening and primed before this command runs.

**Fault policy.** Two tiers, and the tier matters. One pad's bridge dying affects that
pad only — it is reported with its log tail, and the driver's `ActorPool` restarts
exactly that bridge. The **JVM** dying aborts the whole run loudly, because every pad
went with it. There is no survivor floor: the run never silently continues on fewer
arenas.

**Stopping.** Ctrl-C on `agent.train` cleanly stops the collector threads. The
multi-arena path writes a checkpoint during the run whenever a greedy eval improves
the win-rate (not on exit), so the latest best-eval checkpoint is already on disk.
Then Ctrl-C on `start-pads.sh` tears down every bridge and the Paper JVM.

**Fast relaunch** (restart the bridges against a Paper JVM that is already up):

```bash
bash server/setup/start-pads.sh --pads 2 --no-server
```

`--no-server` attaches to the running JVM instead of starting one, so you skip the
world load. It requires `ops.json` to already op all 2N bots (a running server will
not re-read that file) and still runs the prime barrier. This is also the mode
[`docs/spectate.md`](docs/spectate.md) uses, so that Paper keeps an interactive
console.

---

## The measured scale ladder (AC14) — NOT YET RUN

> **TODO(T13): fill this table from `eval.benchmark` runs on the pad fleet.**
> Nothing below has been measured. **Do not put a number in this table that did not
> come out of a run**, and do not carry over any figure from the Windows machine —
> that hardware is gone and its arena topology no longer exists.

Rungs: **1, 2, 4, 8, 12, 16, 20, 25** pads. Each rung is a ≥10-minute run at
`--duration 600`. Record all five metrics per rung. **Promoted N is the largest rung
that meets every one of them** — not four out of five:

| Metric | Gate | Where it comes from |
|---|---|---|
| World-age server TPS | **≥ 19.0** | `eval.benchmark` (real world age, not the physics timer) |
| p99 step round-trip | **≤ 250 ms** | `eval.benchmark` (25% over the 200 ms decision budget) |
| Paper RSS growth over the rung | **< 200 MB** | sample the Paper JVM's PID; identify it before the run starts |
| Max GC pause | **< 50 ms** | one tick; JVM GC log |
| Reset success rate | **≥ 99.5%** | `eval.benchmark` / per-pad bridge logs |

| N | world-age TPS | p99 round-trip (ms) | Paper RSS growth (MB) | max GC pause (ms) | reset success | verdict |
|---:|---|---|---|---|---|---|
| 1 | TODO | TODO | TODO | TODO | TODO | |
| 2 | TODO | TODO | TODO | TODO | TODO | |
| 4 | TODO | TODO | TODO | TODO | TODO | |
| 8 | TODO | TODO | TODO | TODO | TODO | |
| 12 | TODO | TODO | TODO | TODO | TODO | |
| 16 | TODO | TODO | TODO | TODO | TODO | |
| 20 | TODO | TODO | TODO | TODO | TODO | |
| 25 | TODO | TODO | TODO | TODO | TODO | |

**Promoted N: TODO.** N is an empirical result, not a target. Stop climbing once a
rung fails, and record the failing metric — a rung that fails on GC pause tells you
something different from one that fails on TPS.

Also worth recording alongside each rung, though not gated: aggregate transitions/s,
CPU package power, and thermals. A thin-and-light throttles minutes in, so an N that
looks fine at second 30 can fall under 19 TPS by minute 8. Keep them in view for the
whole window, not just the start.

---

## Step 5 — gate recurrence BEFORE trusting M2 (TC8b)

M2's dummy is stationary and always visible, so a green M2 says **nothing** about
whether the LSTM works — a feed-forward encoder alone would pass it. Confirm the
memory-dependent fixture is green so recurrence is actually exercised:

```bash
.venv/bin/python -m pytest tests/test_dqn.py -k "memory or burn_in or recurr" -v
```

## Step 6 — M2 learning (AC6 / TC13)

Train the Dueling-DRQN vs the stationary dummy until the greedy (ε=0) eval clears
the gate: **win-rate ≥95% over 100 eps, aim-bonus-while-invisible == 0, mean
episode length < timeout cap**. The process exits `0` iff the gate passes.

Single arena:

```bash
.venv/bin/python -m agent.train --max-episodes 10000 --eval-every-episodes 50 \
  --eval-episodes 100 --checkpoint runs/m2.pt --run-name m2_train
```

For the multi-pad version, run Step 4b first to get the promoted N, then pass
`--arenas N`.

A **live status bar** (on a TTY) plus a periodic **progress line** report
throughput and an ETA so you can estimate how long the run will take:

```
[m2]  ep 234/10000  [███████░░░░░░░░░░░░]  2.3%  0.63 ep/min  ETA 4d 21h  ε=0.81  win 62%
[m2 progress] ep 234/10000 (2.3%) | 0.63 ep/min | 1.45 steps/s | 89,000 steps | grad 234 | elapsed 2h41m | ETA(budget) 4d 21h | eps=0.810 | last_win=0.620
```

The ETA is to the full `--max-episodes` budget — a worst-case upper bound, since
the loop stops the moment a greedy eval clears the gate. The progress line lands in
the redirected log every `--progress-interval` seconds (default 30; `--no-progress`
silences it), and the same numbers are written to `runs/<run-name>/metrics.jsonl`
under `progress/*` keys.

**Watch `r_damage_dealt` from episode 1.** It was identically zero for the entire
history of this project; a run where it is still zero means the repair regressed, and
the whole reward shape is back to having no gradient on the one action that matters.
The complementary watch is TC16: with regeneration off, **any episode with
`damage_dealt > 20` is a defect, not noise** — 20 HP is all the dummy has.

Also watch the per-component reward log — if win-rate stalls or the agent spins or
runs away, the components catch reward hacking before you blame the learner. If Q
diverges: lower lr, confirm grad-norm clip, slow the target (τ↓).

**Every checkpoint from before the damage-channel repair is unusable** as a baseline,
a warm start, or a demo: it was trained in a regime where landing a hit paid nothing.
The archived pre-repair set is catalogued in
[`docs/analysis/2026-08-10-windows-archive.md`](docs/analysis/2026-08-10-windows-archive.md).

`runs/m2_multi.pt` is the exception and the only checkpoint this branch has trained
post-repair. It is what `deploy.exhibition` loads by default and what the demo runs
on. It is a working checkpoint, not a certified one: no milestone has been signed off
against it. Check the date on any other `.pt` before trusting it.

---

## Training (flags TBD)

The retrain flags (warm start, opponent selection, the EASY/HARD curriculum, eval and
checkpoint cadence) are documented separately and land here shortly.

---

## Step 7 — the human exhibition

This is the demo-day path, not a milestone. It is the only mode where a **person**
is the opponent instead of a bot, and it is the only mode where the episode horizon
is off.

**The full operating procedure lives in [`docs/demo-day.md`](docs/demo-day.md).** Read
that one, not this section, when you are actually running a demo. What follows is the
runbook-level summary and the handful of facts that surprise people.

```bash
.venv/bin/python -m deploy.exhibition --challenger-username <their_mc_name>
```

One command: it runs its refusal gates, writes `server/ops.json` for this one pad,
starts Paper, waits for the Minecraft port, starts the bridge in human-opponent mode,
waits for the bridge port, prints the LAN join address and the pad anchor, and
connects the agent playing greedily from `runs/m2_multi.pt`. Foreground for the whole
exhibition. Every gate runs **before** anything is spawned, so a refusal leaves
nothing running.

Arm the next challenger **from inside Minecraft** — press `T`, type `reset`, Enter:

```
reset
```

`!reset` also works, case is ignored, and the message must be exactly that (a line
merely containing the word does nothing). The bot replies `reset armed - next match
starting` in chat; no reply means nothing was armed. Anyone may type it, there is no
permission check, and a burst of them inside 5 seconds collapses into one reset. This
is the primary path on demo day: with a queue of people waiting, alt-tabbing to a
terminal between matches is not workable.

The terminal command is the fallback and does exactly the same thing:

```bash
.venv/bin/python -m deploy.exhibition --reset
```

It starts nothing and never connects to the bridge. It files
`server/logs/exhibition/reset.request`, which the running launcher consumes within
about a second. Exit codes: `0` armed, `1` refused. The launcher itself exits `130` on
Ctrl-C and `1` on any refusal or boot failure.

**One mechanism, two triggers.** The chat keyword makes the BRIDGE write that same
request file — it is already in game and already reading chat, so it needs no terminal
and opens no socket. `deploy.exhibition` passes it the path with `--reset-request-path`,
the same one the launcher polls, so the two cannot disagree about where the file lives.
Everything downstream (the heal, the re-arm, one-trigger-one-match, and the discard of
a request filed mid-match) is identical and cannot tell which trigger fired. The keyword
is gated on `--opponent-mode human`, so training bridges ignore it outright.

**Do not try to reset by connecting a second client.** `BridgeServer` accepts exactly
one TCP client and resolves a second connection by destroying the incumbent, so a
process that connected would evict the live agent mid-exhibition. That single-client
rule is also why the exhibition is one challenger at a time, start to finish.

Things that bite, all of them documented at length in the demo-day guide:

- **Pin `--challenger-username`.** Unpinned, the bridge credits the first non-agent
  player in the pad, so a bystander who dies to anything gets reported as the agent's
  win; and the launcher cannot heal the human between matches, because nothing on the
  wire says who claimed the slot. It warns you at startup and again at reset time.
- **A reset restores health, food, position and the challenger's sword.** Both
  fighters get exactly one iron sword: the learner from the datapack, the human from
  the launcher's reset commands (a `clear` scoped to `minecraft:iron_sword`, then a
  `give`, so repeated resets never pile up duplicates). Neither side gets armor.
  **Match 1 of a launch is the exception** — a reset is what arms the human, and none
  has happened yet, so the first challenger needs gear handed out before the
  exhibition starts. There is no live command channel once it is running (Paper's
  stdin belongs to the launcher, and `ops.json` is rewritten to the two bots before
  Paper boots).
- **The health check is an absence.** On first exhibition boot, `rl_deaths objective
  NOT confirmed` must **not** appear in `server/logs/exhibition/bridge.log`. Silence
  there is the read-back confirming that human death detection works.
- **`rl_deaths` shows up in the tab player list.** Expected. The objective has to be
  display-bound or the server broadcasts no score packets at all.
- **The first reset of an exhibition logs the challenger on a `foreign_players`
  line.** Expected: the exclusion covers a *claimed* challenger and nothing has
  claimed yet. `eval/benchmark.py` reads that line as cross-pad contamination
  evidence, so an exhibition log looks contaminated to that tool and is not.
- **`No player was found` after a reset** means a pinned challenger who is offline
  right now. Harmless.

Rehearse the whole gate chain with `--dry-run`, which starts nothing and exits `0`:

```bash
.venv/bin/python -m deploy.exhibition --challenger-username demo_player --dry-run
```

---

## Done = the milestones, not the tests

| Milestone | Command | Pass condition | AC |
|-----------|---------|----------------|-----|
| Damage-channel gate | `eval.combat_probe --cycles 10` | per-hit `6,6,6,2`, cumulative 20, one death, reconciles with wire health | AC8 |
| M1 plumbing | `eval.run_random --episodes 100` | ≥100 eps, 0 crashes, RSS < 200 MB | AC3 |
| M1 smoke + spectate | `eval.run_random --episodes 20` | 0 crashes, damage actually lands | AC10 |
| M1 number | `eval.benchmark --duration 600` | transitions/s, p99, damage-exact, TPS | AC4 |
| Pad fleet boot | `start-pads.sh --pads N` | 2N bots joined, opped, staggered, primed | AC12 |
| Cross-pad isolation | `eval.benchmark --arenas N --pad-log-dir …` **against a resetting fleet** | per-pad damage reconciles, zero foreign usernames — the scan half needs resets, so a benchmark-only run does not collect it | AC13 |
| Scale ladder | the eight rungs above | all five metrics per rung; promoted N recorded | AC14 |
| Recurrence | `pytest tests/test_dqn.py -k memory` | memory fixture green, ablation fails | TC8b |
| M2 learning | `agent.train --checkpoint runs/m2.pt` | win ≥95%, aim-while-invisible == 0, len < cap | AC6 |

After M2 passes, the M3/M4 ladder (scripted bot → self-play/PFSP/Elo) is the
next horizon. `distributed/` is built (issue #4) and `deploy/exhibition.py` is the
human-exhibition path (Step 7); `study/` remains skeleton-only and out of kickoff
scope.
