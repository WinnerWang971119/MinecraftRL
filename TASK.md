# TASK — what's left to do

Snapshot after kickoff go-live (2026-06-11). The kickoff foundation (T0–T20) is
built and the **offline suite is green (353 passed)**. The live runbook
(`RUNBOOK.md`) ran overnight; results are in `RUNBOOK_RESULTS.md`. This file is
the open-items list distilled from those two docs + the plan
(`docs/plans/2026-06-09-minecraft-pvp-kickoff.md`).

## Where the milestones actually stand

| AC | Milestone | State | Note |
|----|-----------|-------|------|
| AC1 | Contract frozen | ✅ done | 4 artifacts merged, round-trip tested |
| AC2 | Repo + setup | ✅ done | full skeleton, README setup |
| AC3 | M1 plumbing (zero-crash) | 🟡 passed, number not captured | bridge survived 100 eps = zero-crash PASS; **RSS-growth + win/loss/timeout split scrolled off** (`run_random` persists no log) |
| AC4 | M1 the number | 🟡 partial | latency p50/p95/p99 = 247.8/250.5/251.5 ms, 4.03 transitions/s captured; **TPS gate unmeasurable** (synthetic tick) and **damage-boundary gate INERT live** |
| AC5 | Fairness (PerceptionFilter leak battery) | ✅ done | offline tests green |
| AC6 | **M2 learning (≥95% win)** | ❌ NOT achieved | run killed by Windows Update reboot ~16 min in; no checkpoint written |
| AC7 | Reset correctness | ✅ done | offline + exercised live during M1 |

The kickoff is **not finished**: AC6 (the "does our RL stack learn?" question)
is still open, and AC3/AC4 each have a measurement gap.

---

## P0 — finish the kickoff

### 1. Commit the uncommitted go-live work
The working tree has finished, runbook-referenced work that is not yet committed.
Land it first so the M2 re-run is reproducible and the tree is clean.
- `agent/train.py` (+85) — wires the live progress bar / throughput-ETA reporter.
- `agent/progress.py` + `tests/test_progress.py` — the `ProgressReporter` (new, untracked).
- `server/ops.json` — bot accounts opped (the 1.21 knockback-resistance fix path).
- `RUNBOOK.md`, `RUNBOOK_RESULTS.md` — the go-live runbook + results (untracked).
- `server/` config dump (`bukkit.yml`, `spigot.yml`, `server.properties`, `config/`,
  `plugins/`, etc.) — decide what belongs in git vs `.gitignore` (worlds/logs/jars out).
- `bridge/package-lock.json` — pin the bridge deps in git.

### 2. Re-run M2 training to completion — the headline deliverable (AC6 / TC13)
The whole point of the kickoff. Pause Windows Update first (this box auto-reboots
~03:30 for updates and that is exactly what killed the last run — no checkpoint
survived).
```powershell
# pause WU / set Active Hours / NoAutoRebootWithLoggedOnUsers=1 BEFORE launching
python -m agent.train --max-episodes 10000 --eval-every-episodes 50 `
  --eval-episodes 100 --checkpoint runs/m2.pt --run-name m2_train
```
Pass = greedy (ε=0) win-rate ≥95% / 100 eps, aim-bonus-while-invisible == 0, mean
episode length < timeout cap. Watch the per-component reward log from ep 1 for
spin-farming / run-away before blaming the learner. Note: a single timeout episode
is 80 s wall-clock, so the 10k budget will not finish in one night — the loop
stops early the moment a greedy eval clears the gate.

---

## P1 — close the M1 measurement gaps

### 3. Capture the AC3 number (RSS growth + win/loss/timeout split)
`eval.run_random` has no `MetricsLogger`, so the AC3 figures (combined-process RSS
growth < ~200 MB, episode outcome split) were never persisted — they scrolled off
in the user's terminal. Either re-read that scrollback, or add a persisted log to
`eval/run_random.py` and re-run a shorter pass to record the number on disk.

### 4. Report the real server tick so the TPS gate means something (AC4)
The bridge emits a **synthetic** tick (`handleStep` sets `_currentTick += ACTION_REPEAT`
in `bridge/bot.js`), so `sustains_19_tps` is a logical decision counter, not Paper's
real TPS — it structurally cannot pass for one synchronous arena even though the
server is healthy (~20 TPS, 13% CPU). Make the bridge report the **real** server
tick / MSPT so the AC4 TPS gate is honest.

### 5. Drive the live damage-boundary exchange (TC7b)
The AC4 damage-boundary gate is **INERT** on a live run — it needs a scripted,
known-N-hit exchange driven through the real server, which the benchmark doesn't do
yet (the CLI prints an INERT banner and exits 1 by design). Add a scripted N-hit
driver so `summed events == N` is actually verified live, not just in fixtures (TC7).

---

## P2 — deferred, on file (next horizon, explicitly out of kickoff scope)

These have skeleton dirs/stubs already; do not build during kickoff.
- **Multi-arena bridge** (`bridge/arena.js`) — so `eval.benchmark --arenas N` connects
  to more than port 5555 and the "max arenas @≥19 TPS" figure becomes measurable.
- **M3 — scripted bot opponent** (`opponents/scripted_bot.py` stub) — heuristic chase/attack.
- **M4 — self-play / PFSP / Elo** (`opponents/snapshot_pool.py` stub) + the PPO comparison arm.
- **distributed/** — `LocalTransport` + ZeroMQ/Redis actor/learner (interface stubbed only).
- **deploy/** booth + **study/** human-subject app — skeleton dirs only.
- **Custom Paper reset plugin** — adopt at N-arena scaling or if the read-back gate
  proves flaky/slow (per the `/debate` verdict; commands + read-back gate for now).
- **Simulator parity** — memory flags exact sim↔robot parity is required for the
  Phase 5/6 mission delivery horizon.

---

## Operational reminder
- **Pause Windows Update before any multi-hour unattended run.** The last M2 attempt
  died to a planned servicing reboot at 03:33; the run had no checkpoint yet, so
  everything was lost. Use Active Hours covering the window or
  `NoAutoRebootWithLoggedOnUsers=1`.
