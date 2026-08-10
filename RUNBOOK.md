# Go-Live Runbook — from green tests to M1 + M2

**What we have:** the full kickoff foundation (T0–T20) is built and the offline
suite is green — `pytest` reports **353 passed**. But every offline test runs
against fakes and fixtures. Nothing has touched a live Paper server yet, so the
two questions the project exists to answer are still open:

- **M1 — does the plumbing work?** (AC3 zero-crash slice + AC4 the measured number)
- **M2 — does the RL stack learn?** (AC6 greedy win-rate ≥95% vs the dummy)

This runbook takes the stack live and collects those acceptances, in dependency
order. Everything below the line needs the real game running; none of it is
covered by `pytest`.

---

## Prereqs (one-time)

| Need | Check | Source |
|------|-------|--------|
| Java 21+ (machine has 25) | `java -version` | Paper requires it |
| Node v24 (≥22) | `node -v` | `bridge/package.json` |
| Python 3.11+ (machine has 3.14) | `python --version` | `torch` works here per memory |
| Python deps incl. torch | `python -c "import torch"` | `pip install -e .` |

```powershell
python -m pip install -e .     # agent package + deps
cd bridge; npm install; cd ..  # mineflayer + plugins (pinned)
```

---

## Step 0 — close the one gap: the bridge launcher

**DONE — `bridge/run.js` + `npm start` exist.** The Python side connects to a
Node TCP bridge on `127.0.0.1:5555`. The bridge logic all lives in
`bridge/bot.js` as the `ArenaBots` class; `bridge/run.js` is the process entry
that does the four-call wiring (construct → `wireTransport()` → `connect()` →
`transport.listen()`). Two deviations from the original sketch, both deliberate:

- **Bots connect before the port opens** (`connect()` before `listen()`), so
  the Python env can only ever reach a bridge that is ready to serve a reset.
  If Paper is down, `npm start` exits 1 with `ECONNREFUSED 127.0.0.1:25565`
  instead of opening a port that leads nowhere (verified offline).
- **`'error'` listeners on the transport and both bots** — an unlistened
  `'error'` event on an EventEmitter kills the Node process, and the M1 bar
  is zero crashes. Socket/bot errors are logged; the Python side retries.

This is the first real integration point — expect to debug the live handshake
here (bot spawn order, op timing), which is exactly what `server/compat_check.md`
flags as the human follow-up.

> Run order matters: **start Paper first** (Step 1), then the bridge (Step 2) so
> the two bots have a server to join, then the Python driver (Steps 3+).

---

## Step 1 — boot the Paper server

```powershell
pwsh -NoProfile -File server/setup/setup.ps1   # downloads jar, writes config, installs datapack (idempotent)
pwsh -NoProfile -File server/setup/start.ps1   # --nogui console; type 'stop' to shut down
```

First-boot checklist (from `server/README.md` / `compat_check.md`):
- world generates flat, no mobs, `online-mode=false`.
- the arena datapack loads (`arena:setup` runs on load).
- `learner_bot` and `dummy_bot` are opped at level 4 (`server/ops.json`).
- after a bot joins, `/attribute dummy_bot minecraft:generic.knockback_resistance get`
  resolves and the dummy stays put when hit. The `generic.` infix is **required**
  on the pinned Paper 1.21.1 stack — the flattening that removed it landed in
  1.21.2. See `server/README.md` before changing this form.

## Step 2 — start the bridge

```powershell
cd bridge; npm start   # the Step-0 launcher
```

Both bots should spawn and the bridge should print `listening on 5555`.

---

## Step 3 — M1 plumbing (AC3 / TC11)

Random policy, ≥100 full episodes vs the idle dummy, end to end. The bar is
**zero crashes** and combined-process RSS growth **< ~200 MB** (sampled every 10
episodes).

```powershell
python -m eval.run_random --episodes 100 --host 127.0.0.1 --port 5555
```

Read the printed summary: episodes completed, win/loss/timeout split, RSS growth.
This is the first proof the whole loop (rollout → store → sample → no-op update)
survives the real bridge.

## Step 4 — M1 the number (AC4 / TC12)

The bridge spike's exit criterion is a *measured number*, not a green check:
transitions/s, p99 Node→Python round-trip at the 200 ms interval, damage-event
boundary correctness, and max arenas sustaining ≥19 TPS over a ≥10-min run.

```powershell
python -m eval.benchmark --duration 600 --arenas 1   # then sweep --arenas 2,3,4 for the max-arenas figure
```

Record CPU package power / thermals alongside — the Core Ultra 7 figure is a
lower-bound smoke number, not fleet capacity. Expect ~2–4 stable arenas, not 8.

> **What "TPS" measures here.** The benchmark's server TPS is the REAL server
> tick rate, read from the learner bot's world age (`bot.time.age`, set only by
> the server's `update_time` packet) and averaged over a rolling 5 s window
> (Paper's own `/tps` is likewise a rolling average). It is decoupled from the
> client-side Mineflayer `physicsTick` timer, so a healthy ~20 TPS server reads
> ~20 even though `update_time` arrives only ~once per second. Raw collection
> throughput (transitions/s) is a SEPARATE number and is still capped on Windows
> by Mineflayer's `setInterval(50ms)` physics loop firing at ~60ms under the
> ~15.6ms system timer resolution; raise it later with `timeBeginPeriod(1)` or by
> running the bridge on Linux/WSL.

---

## Step 4b — AC4 multi-arena live run (issue #4, ~10 min)

This step confirms `distributed/` is wired correctly and measures peak throughput
under N concurrent arenas. It is a follow-up to Step 4, run after the single-arena
benchmark already passes.

**NOTE on `--arenas` flags.** Two different commands have an `--arenas` flag that
means different things. `eval.benchmark --arenas N` (Step 4 above) measures
throughput from N arenas as a **benchmark** (it does not train). The
`agent.train --arenas N` flag below runs **training** from N concurrent arenas.
Do not confuse them.

### Caveats (read before starting a long run)

- **Windows Update auto-reboots this box at ~03:30** with no warning and no
  checkpoint save. Pause Windows Update (Start → Windows Update → Advanced options →
  Pause, or set Active Hours to cover your run window) before any run expected to
  last more than a few hours. Prefer launching in the morning so Active Hours carry
  you through.

- **First boot of N fresh Paper worlds is slow.** World generation, plugin load,
  bot re-op, and re-teleport each take tens of seconds per arena. The first reset per
  arena will be noticeably slower than steady-state; let it settle before judging
  throughput. Relaunch of a dead arena (fault recovery) is similarly slow (30–60 s+).

- **Cores and thermals bind before RAM** on this 8c/8t / 32 GB box. Around 2–4
  arenas is the realistic range; 8 is not achievable. The throughput goal is
  ≥19 TPS sustained across all running arenas.

- **Do not shrink the JVM heap to fit more arenas.** RAM is not the ceiling here:
  4 arenas at the default `-Xms2G -Xmx2G` is ~10–12 GB of 32 GB. The single-threaded
  Paper tick runs out of cores first, and cutting heap only invites GC pauses that
  burn CPU and *drop* TPS — the opposite of what you want. Leave heap at 2G. Only
  touch `-Xms/-Xmx` (both, kept equal) if a GC log shows real pressure, and then
  *raise* it, never lower it.

### Launch procedure

> **Topology note (T10).** This is now **one Paper JVM hosting N enclosed pads**,
> not N JVMs. The Minecraft port stays `25565`; pad `i` gets bridge port `5555+i`,
> anchor `((i % 5) * 512, (i // 5) * 512)`, and bots `learner_bot`/`dummy_bot` at
> `i == 0`, `learner_<i>`/`dummy_<i>` above. `server/setup/start-pads.sh` is the
> launcher. The old PowerShell N-JVM orchestrator is gone.

**1. Dry-run first** to print the full plan and sanity-check ports, usernames, and
anchors before anything goes live:

```bash
bash server/setup/start-pads.sh --pads 2 --dry-run
```

Check the output: one shared MC port (25565), distinct bridge ports (5555, 5556,
...), distinct anchors (`0,0`, `512,0`, ...) and distinct usernames
(`learner_bot`/`dummy_bot`, `learner_1`/`dummy_1`, ...).

You can also verify the Python-side plan without touching any process:

```bash
python -m distributed.launcher --pads 2 --dry-run
```

**2. Run `setup.sh` once with the pad count** (idempotent; downloads the Paper jar
and sizes `max-players` to `2N+10`):

```bash
PADS=2 bash server/setup/setup.sh
```

**3. Start the fleet** (Paper + N bridges + the prime barrier). This blocks and
keeps every process alive until Ctrl-C:

```bash
bash server/setup/start-pads.sh --pads 2
```

Add `--check` to run only the preflight gates (max-players, ports, datapack, ops)
and exit. The script starts Paper, then each bridge one at a time — a pad's bridge
port opens only after both of its bots have joined, so the port coming up *is* the
join gate — then resets every pad once before printing `FLEET READY`.

**Do not start the Python driver until `FLEET READY` is printed.** Until every pad
has been reset, all 2N bots are stacked at the shared world spawn inside pad 0, and
a stepping pad will hit foreign bots and credit the damage to the wrong policy.

**4. Confirm each bridge is listening** (from a second terminal, while
`start-pads.sh` is still running):

```powershell
# Arena 0 (bridge port 5555)
python -c "import socket; s=socket.create_connection(('127.0.0.1',5555),2); s.close(); print('5555 up')"
# Arena 1 (bridge port 5556)
python -c "import socket; s=socket.create_connection(('127.0.0.1',5556),2); s.close(); print('5556 up')"
```

**5. Sweep `--arenas` to find the max** (in a third terminal). The ceiling is set
by cores + thermal, not RAM, so this is the measurement that actually finds it.
The answer is the highest N where **every** arena holds ≥19 TPS for the **full**
10 min *and* aggregate transitions/s is still rising. Because the live benchmark
opens one real `TcpBridgeClient` on `port + i` per arena, N pads must be up first —
relaunch `start-pads.sh --pads N` (step 3) to match before each run:

```powershell
python -m eval.benchmark --duration 600 --arenas 1
python -m eval.benchmark --duration 600 --arenas 2
python -m eval.benchmark --duration 600 --arenas 3
python -m eval.benchmark --duration 600 --arenas 4
# keep stepping (5, 6, ...) until the stop rule below trips
```

**Watch thermals for the whole run, not just the start.** A thin-and-light
throttles minutes in, so an N that looks fine at second 30 can fall under 19 TPS
by minute 8. Keep CPU package power / core temps (HWiNFO or Task Manager) in view
next to the per-arena TPS the benchmark prints.

**Stop rule — back off one when either trips at N:**
- any arena's sustained TPS drops below 19 across the 10-min window, or
- aggregate transitions/s stops rising (or falls) versus N−1.

The max usable arena count is the last N *before* the trip. Record each run:

| N | aggregate transitions/s | min sustained per-arena TPS | peak pkg power / temp | verdict |
|---|-------------------------|-----------------------------|-----------------------|---------|
| 1 |                         |                             |                       | base    |
| 2 |                         |                             |                       |         |
| 3 |                         |                             |                       |         |
| 4 |                         |                             |                       |         |

The real AC4 number is measured live here; it could not be measured in-session.

**6. Run multi-arena training** with the winning N (once you know it from step 5).
Keep `start-pads.sh` running (or relaunch it), wait for `FLEET READY`, then in a
separate terminal:

```bash
python -m agent.train --arenas 2 --max-episodes 10000 \
  --eval-every-grad-steps 1000 --eval-episodes 100 \
  --checkpoint runs/m2_multi.pt --run-name m2_multi
```

The `--arenas N` flag on `agent.train` (not on `eval.benchmark`) is what engages
the multi-arena `ActorPool` + decoupled learner. Arena `i` connects to bridge port
`--port + i` (default base 5555, so arena 0 → 5555, arena 1 → 5556, ...). The
N bridges must already be listening before this command runs.

**Stopping.** Ctrl-C on `agent.train` cleanly stops the collector threads. The
multi-arena path writes a checkpoint during the run whenever a greedy eval
improves the win-rate (not on exit), so the latest best-eval checkpoint is
already on disk. Then Ctrl-C on `start-pads.sh` tears down every bridge and the
Paper JVM.

**Fast relaunch** (restart the bridges against a Paper JVM that is already up):

```bash
bash server/setup/start-pads.sh --pads 2 --no-server
```

`--no-server` attaches to the running JVM instead of starting one, so you skip the
world load. It requires `ops.json` to already op all 2N bots (a running server will
not re-read that file) and still runs the prime barrier.

---

## Step 5 — gate recurrence BEFORE trusting M2 (TC8b)

M2's dummy is stationary and always visible, so a green M2 says **nothing** about
whether the LSTM works — a feed-forward encoder alone would pass it. Confirm the
memory-dependent fixture is green so recurrence is actually exercised:

```powershell
python -m pytest tests/test_dqn.py -k "memory or burn_in or recurr" -v
```

## Step 6 — M2 learning (AC6 / TC13)

### Before you run this

Multi-arena training (`distributed/`, issue #4) is now built. Run Step 4b first
to find the max stable N on this machine, then pass `--arenas N` here for faster
throughput. If you are short on time or skipping 4b, the single-arena path below
works as-is.

Train the Dueling-DRQN vs the stationary dummy until the greedy (ε=0) eval clears
the gate: **win-rate ≥95% over 100 eps, aim-bonus-while-invisible == 0, mean
episode length < timeout cap**. The process exits `0` iff the gate passes.

```powershell
python -m agent.train --max-episodes 10000 --eval-every-episodes 50 `
  --eval-episodes 100 --checkpoint runs/m2.pt --run-name m2_train
```

A **live status bar** (on a TTY) plus a periodic **progress line** report
throughput and an ETA so you can estimate how long the run will take:

```
[m2]  ep 234/10000  [███████░░░░░░░░░░░░]  2.3%  0.63 ep/min  ETA 4d 21h  ε=0.81  win 62%
[m2 progress] ep 234/10000 (2.3%) | 0.63 ep/min | 1.45 steps/s | 89,000 steps | grad 234 | elapsed 2h41m | ETA(budget) 4d 21h | eps=0.810 | last_win=0.620
```

The ETA is to the full `--max-episodes` budget — a worst-case upper bound, since
the loop stops the moment a greedy eval clears the gate. Throughput is measured
over a sliding window so it tracks the *current* pace, not the slow replay
warm-up. The progress line lands in the redirected log every `--progress-interval`
seconds (default 30; pass `--no-progress` to silence it), and the same numbers are
written to `runs/<run-name>/metrics.jsonl` under `progress/*` keys for plotting.

Watch the **per-component reward log** from episode 1 — if win-rate stalls or the
agent spins/runs away, the components catch hacking before you blame the learner.
If Q diverges: lower lr, confirm grad-norm clip, slow the target (τ↓).

---

## Done = the milestones, not the tests

| Milestone | Command | Pass condition | AC |
|-----------|---------|----------------|-----|
| M1 plumbing | `eval.run_random --episodes 100` | ≥100 eps, 0 crashes, RSS < 200 MB | AC3 |
| M1 number | `eval.benchmark --duration 600` | transitions/s, p99, damage-exact, max-arenas@19TPS | AC4 |
| Multi-arena live run | `start-pads.sh --pads N` + `eval.benchmark --arenas N` | max N sustaining ≥19 TPS (live-measured); `agent.train --arenas N` runs training | AC4 follow-up |
| Recurrence | `pytest tests/test_dqn.py -k memory` | memory fixture green, ablation fails | TC8b |
| M2 learning | `agent.train --checkpoint runs/m2.pt` | win ≥95%, aim-while-invisible == 0, len < cap | AC6 |

After M2 passes, the M3/M4 ladder (scripted bot → self-play/PFSP/Elo) is the
next horizon. `distributed/` is now built (issue #4); `deploy/` and `study/`
remain skeleton-only and out of kickoff scope.
