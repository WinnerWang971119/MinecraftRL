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
- after a bot joins, `/attribute dummy_bot minecraft:knockback_resistance get`
  resolves and the dummy stays put when hit (the 1.21 attribute-ID fix).

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

---

## Step 5 — gate recurrence BEFORE trusting M2 (TC8b)

M2's dummy is stationary and always visible, so a green M2 says **nothing** about
whether the LSTM works — a feed-forward encoder alone would pass it. Confirm the
memory-dependent fixture is green so recurrence is actually exercised:

```powershell
python -m pytest tests/test_dqn.py -k "memory or burn_in or recurr" -v
```

## Step 6 — M2 learning (AC6 / TC13)

### Important!!!
Don't run this until parrel arena is finished or it will take you too ling to finish.

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
| Recurrence | `pytest tests/test_dqn.py -k memory` | memory fixture green, ablation fails | TC8b |
| M2 learning | `agent.train --checkpoint runs/m2.pt` | win ≥95%, aim-while-invisible == 0, len < cap | AC6 |

After M2 passes, the deferred dirs (`distributed/`, `deploy/`, `study/`) and the
M3/M4 ladder (scripted bot → self-play/PFSP/Elo) are the next horizon — all
explicitly out of kickoff scope.
