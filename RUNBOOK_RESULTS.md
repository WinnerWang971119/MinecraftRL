# Go-Live Results — RUNBOOK Steps 3 → 6

Overnight run recorded by Claude while the live stack (Paper + bridge) stayed up.
Started from a live stack already running: Paper (java) on `:25565`, bridge
(`node run.js`, PID 33688) on `127.0.0.1:5555`. Date context: 2026-06-11.

**How to read this:** each step lists the command, the pass condition from the
RUNBOOK "Done" table, and what was actually observed. Raw captures live under
`runs/`.

---

## Step 3 — M1 plumbing (AC3 / TC11) — RUN BY USER

`python -m eval.run_random --episodes 100 --host 127.0.0.1 --port 5555` (PID 30480).

This was running in the user's own terminal at handoff, so its **printed summary
(episodes / win-loss-timeout split / RSS growth) is in that terminal's
scrollback** — the `run_random` CLI does not persist a log file (no MetricsLogger
on this entry point), so it cannot be recovered here.

What can be verified independently is recorded below once the run releases the
bridge (process exit + whether the Paper server and bridge survived = the
zero-crash signal).

- Status: **COMPLETED** (process exited after ~2 h 10 m wall-clock; 100
  episodes of 80 s-cap random-policy play).
- **Zero-crash signal (the AC3 bar): PASS.** Immediately after PID 30480 exited,
  the live stack was still up: bridge `node run.js` (PID 33688) **alive and
  listening on 5555**, Paper still listening on 25565. A mid-run crash would have
  killed the bridge process (an unhandled `error`/rejection is process-fatal in
  Node — the exact M1 failure mode), so the bridge surviving all 100 episodes is
  the core zero-crash evidence. (Recorded to `runs/step3_wait.txt`.)
- **What is NOT captured here:** the printed per-run summary — episode win/loss/
  timeout split and the combined-process RSS-growth number — scrolled in the
  user's own terminal and `run_random`'s CLI persists no log, so those exact
  figures must be read from that terminal. The runtime (~2 h 10 m for 100 eps)
  is consistent with most episodes hitting the 80 s timeout cap, i.e. a random
  policy rarely killing the dummy (a low-but-ideally-nonzero win count — if it is
  exactly 0, check the damage-event path per the standing note).

---

## Step 4 — M1 the number (AC4 / TC12)

`python -m eval.benchmark --duration 600 --arenas 1` then a sweep `--arenas 2,3,4`.

- Status: **COMPLETED** for `--arenas 1` (full 600.1 s live run, 2420 decisions).
  Sweep `--arenas 2,3,4` not possible on this bridge (single arena — see below);
  user confirmed single arena is acceptable. Captures: `runs/step4_bench_arena1.json`
  (full report), `runs/step4_bench_arena1.log` (summary + gate banner),
  `runs/bench_arena1/metrics.jsonl` (per-decision series).

**Measured AC4 numbers (live, 1 arena, 600 s):**

| Metric | Value |
|--------|-------|
| transitions/s/arena | **4.03** (2420 decisions / 600.1 s) |
| round-trip p50 / p95 / p99 | **247.8 / 250.5 / 251.5 ms** (min 178.8, max 252.7, mean 247.7) |
| sustained_tps_min / sustains ≥19 | **15.81 / FALSE** |
| max_arenas_sustaining_tps | 0 |
| CPU mean / max | 13.2% / 75% (freq pinned 2200 MHz) |
| damage-boundary gate | INERT (not scripted live) → exit 1 |

**Reading the numbers:**
- **Latency is excellent and stable** — p99 is only ~4 ms above p50 over a 10-minute
  run, i.e. essentially no jitter or drift. Each decision is the 200 ms (4-tick)
  window + ~48 ms of socket/Python/bridge round-trip overhead ≈ 248 ms, which is
  exactly the 4.03 decisions/s observed.
- **The sub-19 "TPS" is a metric artifact, NOT server lag.** The bridge emits a
  **synthetic** tick: `handleStep` sets `this._currentTick = windowStartTick +
  ACTION_REPEAT` (+4 every step, `bridge/bot.js`), so it is a logical decision
  counter, not Paper's real server tick. `TickDeltaTpsProvider` then computes
  `4 ticks ÷ wall_per_step = 4 / 0.248 s ≈ 16`. For a single **synchronous**
  arena this is structurally capped below 19 (it would need total per-step
  overhead < ~10.5 ms to clear the floor; even zero overhead gives exactly 20.0
  at the edge). Meanwhile real Paper advanced ~5 ticks in that 248 ms (~20 TPS)
  and CPU sat at ~13% mean — the **server is keeping up fine**; the gate just
  cannot see the ticks that elapse during the inter-step overhead gap.
- **Implication:** the `sustains_19_tps` gate, as wired against the bridge's
  synthetic tick counter, effectively cannot pass for one synchronous arena and
  the "max arenas @≥19 TPS" figure is not meaningfully measurable until (a) the
  bridge reports the **real** server tick (or MSPT) and (b) the multi-arena
  `arena.js` exists so arenas run concurrently rather than one-at-a-time. Both are
  the deferred follow-ups already on file. The honest AC4 evidence tonight is the
  throughput + the tight, stable latency distribution + low CPU, all of which say
  the single-arena live loop is healthy with headroom.

> Note on exit code: the live `benchmark` entry point drives raw `step`s and
> derives TPS from real `tick` deltas (the honest gate), but the **damage-boundary
> gate is inert** on a live run (it needs a scripted known-N-hit exchange, which
> can't be driven live). The CLI therefore prints a LOUD "INERT" banner and exits
> **1 by design even when TPS is fine** — exit 1 here is NOT a TPS failure. The
> measured numbers (transitions/s, p50/p95/p99 round-trip, `sustained_tps_min`,
> `sustains_19_tps`) come from the printed JSON and are the actual AC4 evidence.

> ⚠️ Sweep limitation found before running: the live bridge (`bridge/run.js`)
> serves a **single arena on a single port (5555)**. `eval.benchmark --arenas N`
> opens one client per arena at `port + i` (5556, 5557, 5558 for a 2/3/4 sweep),
> and nothing is listening there, so `--arenas >1` cannot connect to this bridge.
> The multi-arena `arena.js` work is explicitly deferred / out of kickoff scope
> (RUNBOOK Step 0 note + project memory). The `--arenas 1` run is therefore the
> AC4 evidence available tonight; the max-arenas figure needs the multi-arena
> bridge and is left as a documented follow-up rather than a code change made
> while unattended.

---

## Step 5 — gate recurrence BEFORE trusting M2 (TC8b) — DONE ✅

Offline / CPU-only (no bridge), so run immediately.

**RUNBOOK command** — `pytest tests/test_dqn.py -k "memory or burn_in or recurr"`:
`3 passed, 8 deselected` (the three TC8 burn-in *geometry* tests). Capture:
`runs/step5_recurrence_pytest.txt`.

**Caveat found:** that exact `-k` filter matches only the burn-in geometry tests.
The tests that actually *isolate recurrence* (the project's stated TC8b gate) were
missed by it — the memory-dependent learning test is `@pytest.mark.slow` (and
`pyproject.toml` sets `addopts = "-m 'not slow'"`, deselecting it by default), and
the ablation test's name matches none of `memory|burn_in|recurr`. So the recurrence
gate was run explicitly:

- `test_tc8b_burn_in_drqn_learns_memory_dependent_action` (slow) — **PASSED**: the
  recurrent DRQN overfits the memory-gated fixture to scored accuracy > 0.9, i.e.
  it must carry the cue in the LSTM hidden state.
- `test_tc8b_ablation_zeroed_hidden_state_fails_fixture` — **PASSED**: the same net
  with the hidden state zeroed each step **cannot** solve the fixture (stays under
  the memoryless ceiling). This is the "ablation fails" half of the gate — it
  proves the LSTM is the component responsible.
- `test_tc8b_smoke_harness_runs_short_training` — **PASSED**.
- `test_bootstrap_uses_correct_recurrent_hidden_state` (test_train.py) — **PASSED**:
  the n-step bootstrap recurs over the contiguous `obs_ext` stream (the known
  silent-bias trap from project memory is guarded).

Captures: `runs/step5_tc8b_recurrence.txt`, `runs/step5_bootstrap_recurrence.txt`.

**Verdict:** recurrence is genuinely exercised — memory fixture green AND the
ablation fails. Safe to trust an M2 pass as evidence the LSTM works.

---

## Step 6 — M2 learning (AC6 / TC13)

`python -m agent.train --max-episodes 10000 --eval-every-episodes 50 --eval-episodes 100 --checkpoint runs/m2.pt --run-name m2_train`

Pass condition: greedy (ε=0) eval win-rate ≥95% over 100 eps, aim-bonus-while-
invisible == 0, mean episode length < timeout cap. Process exits 0 iff the gate
passes. Logs to `runs/m2_train/` (jsonl fallback).

- Status: _pending Step 3 + Step 4 (shares the single bridge)._

**Timing reality (frozen contract):** `MAX_EPISODE_STEPS = 400`,
`DECISION_INTERVAL_MS = 200`, `SERVER_TPS = 20`. A timeout episode is therefore
`400 × 0.2 s = 80 s` of wall-clock. With the default `--eval-episodes 100`, an
early greedy eval (policy still loses/times out) can cost up to ~`100 × 80 s ≈
2.2 h`; a 50-episode training block adds more. So the **10000-episode budget will
not complete overnight** — but the stationary dummy is the easiest possible
opponent (always visible, never moves, wins terminate early as soon as it dies),
so win-rate should climb and episodes should shorten as the agent learns. The
plan: launch the runbook command verbatim, log to `runs/m2_train/`
(`metrics.jsonl` + `summary.json`, jsonl fallback), monitor the per-eval
win-rate trajectory, and record whatever gate state is reached by morning plus
the full curve. Hyperparameters are left exactly as the runbook specifies (not
tuned while unattended). `runs/m2.pt` holds the best/final checkpoint for a later
standalone `eval.evaluate --checkpoint runs/m2.pt` re-check.

- Status: **INTERRUPTED — killed by a Windows Update reboot ~16 min in (see
  "Incident" below). Needs a re-run; nothing learned survived.**
- Was launched detached at 03:17:33 (PID 35452, command exactly as the runbook).
  Reached only **5 gradient steps / ~7–8 episodes** (replay 2801, ε still 0.967)
  before the 03:33:59 reboot. No greedy eval ran (first is at episode 50) and
  **`runs/m2.pt` was never written** (checkpoints save only on an eval win-rate
  improvement), so there is no recoverable checkpoint — a clean re-run is needed.
- (Original launch confirmation, for the record:) it held an
  ESTABLISHED socket to the bridge and the logger initialized
  (`runs/m2_train/summary.json` has the config). Progress sources:
  - `runs/m2_train/metrics.jsonl` — per-grad-step training stats (starts once the
    replay passes `min_replay = 1000` transitions, ~3 episodes in) and per-episode
    eval-component series.
  - `runs/m2_train/summary.json` — rolling run-level summary (rewritten each eval).
  - `runs/step6_m2_train.log` — stderr: one `[m2 ep N] … win_rate=… passed_m2=…`
    line per greedy eval (every 50 episodes) and a final `[m2 done] …` line.
  - `runs/m2.pt` — best/final checkpoint (saved on each win-rate improvement and at
    the end), re-checkable later with `eval.evaluate --checkpoint runs/m2.pt`.
  - A persistent monitor is tailing the stderr log for eval lines, completion, and
    crash signatures. The per-eval results + final gate verdict are appended below
    as they arrive.

### Step 6 eval trajectory (appended as evals land)

_(no eval ran — the run was killed at 03:33:59, before the episode-50 eval.)_

---

## Incident — Windows Update auto-reboot at 03:33:59 (2026-06-11)

The overnight machine rebooted itself, killing the live stack (Paper + bridge)
and the in-flight Step 6 training. **Root cause: a scheduled Windows Update
servicing reboot — NOT a crash, power loss, or hardware fault.** System event log:

- 03:29:31 — Event **1074**, `MoUsoCoreWorker.exe` (Windows Update Orchestrator)
  initiated restart, reason *"Operating System: Service pack (Planned)"*.
- 03:30–03:34 — multi-phase servicing reboots (`TrustedInstaller.exe`,
  *"Upgrade (Planned)"*); back up at 03:33:59.
- 03:34:35 — Event **19**: installed **2026-06 Security Update KB5094126
  (build 26200.8655)** (+ KB5094135). No Kernel-Power 41, no BugCheck 1001, no
  6008 — a clean planned reboot, so the filesystem (and Steps 3–5 results) is
  intact.

**Why it cost the whole training run:** Step 3 took ~2 h 10 m, so Step 6 didn't
start until 03:17 — and the update reboot landed 16 minutes later, before the
first eval/checkpoint. To avoid a repeat, pause Windows Update (or set Active
Hours / no-auto-restart-with-logged-on-users) before the next multi-hour run.

Steps 3, 4, 5 above are unaffected (completed and on disk before the reboot).
