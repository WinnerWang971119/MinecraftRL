# Minecraft PvP RL

Reinforcement-learning agents that fight in Minecraft, trained on the **real game**
(no simulator) through a Mineflayer → bridge → Paper stack.

This repo is the **kickoff foundation** for an 8-person team: a frozen interface
contract, a real end-to-end vertical slice (M1), and a Dueling-DRQN agent that
beats a stationary dummy ≥95% (M2). It answers the two questions the project
exists to answer — *does the plumbing work?* and *does our RL stack learn?* — on
the real stack, no mock.

> **New here? Read this file top to bottom, then run [Your first 30 minutes](#your-first-30-minutes).**
> When you're ready to take the stack live, follow [`RUNBOOK.md`](RUNBOOK.md).

---

## How it fits together

One decision every 200 ms: Python picks a macro action, the Node bridge runs it on
a real bot for `ACTION_REPEAT` (4) ticks, and ships back an aggregated game state.

```
  agent/ (DRQN)                  env/                         bridge/ (Node)            server/ (Paper)
  ┌──────────────┐   action int  ┌──────────────────┐  JSON   ┌────────────────┐ MC    ┌───────────────┐
  │ Q-net + replay├──────────────►│ MCPvPEnv         ├────────►│ Mineflayer bots├──────►│ flat arena,   │
  │ + train loop  │◄──────────────┤ + PerceptionFilter│◄────────┤ macros, events │◄──────┤ learner+dummy │
  └──────────────┘  obs + reward  │ + reward.py      │  state  └────────────────┘       └───────────────┘
                                  └──────────────────┘
                         raw TCP, newline-delimited JSON on 127.0.0.1:5555
```

- **Python → Node:** `{"type":"step","action":<0..7>}` / `{"type":"reset",...}` / `{"type":"close"}`
- **Node → Python:** a `state` message per step (self, opponent **raw**, events, arena, tick, `code_version`).
- **Fairness is applied in Python, not Node.** The bridge sends raw opponent state;
  `env/perception_filter.py` gates it (FOV + line-of-sight + memory) **before** it
  reaches the observation. The agent never sees what it shouldn't. Do not move
  gating into the bridge.

The full wire contract is the plan's *API Contracts* section and `bridge/schema.md`.

---

## Where we are right now

The whole foundation (tasks T0–T20) is **built**, and the **offline test suite is
green** (`pytest` → 353 passing). But every offline test runs against fakes and
fixtures — **nothing has touched a live Paper server yet.** The two milestones that
need the real game are still open:

| Milestone | Question | Bar | Status |
|-----------|----------|-----|:------:|
| **M1 plumbing** (AC3) | does the loop survive the real bridge? | ≥100 episodes, 0 crashes, RSS growth < ~200 MB | ⬜ live run pending |
| **M1 the number** (AC4) | how fast / how many arenas? | transitions/s, p99 round-trip, damage-exact, max arenas @≥19 TPS | ⬜ live run pending |
| **Recurrence gate** (TC8b) | does the LSTM actually work? | memory fixture green, ablation fails | ✅ offline, re-confirm before trusting M2 |
| **M2 learning** (AC6) | does the RL stack learn? | greedy win ≥95% / 100 eps, no spin-farming | ⬜ live training pending |

Taking the stack live and collecting these acceptances is exactly what
[`RUNBOOK.md`](RUNBOOK.md) walks through, in dependency order. **Done = the
milestones, not green tests.**

---

## Your first 30 minutes

**Prereqs** (the dev laptop has all of these; check yours):

| Need | Check | Why |
|------|-------|-----|
| Java 21+ | `java -version` | Paper 1.21.1 requires it (dev box runs 25) |
| Node v24 (≥22) | `node -v` | mineflayer floor is `engines.node >=22` |
| Python 3.11+ | `python --version` | agent package (dev box runs 3.14) |

```powershell
# 1. Clone, then install both sides.
python -m pip install -e .        # agent package + deps (numpy, torch, pytest)
cd bridge; npm install; cd ..     # mineflayer + plugins (pinned in package.json)

# 2. Prove your checkout is healthy — this is the green light to start working.
pytest                            # Python unit + integration -> 353 passed
cd bridge; npm test; cd ..        # Node bridge smoke
```

`torch` is the one heavy dependency — install the CPU or CUDA wheel for your
machine (see the note in [`requirements.txt`](requirements.txt)). The determinism
helpers in [`agent/seeding.py`](agent/seeding.py) degrade gracefully and still seed
Python + NumPy if `torch` is missing, so most of the suite runs without it.

If `pytest` is green you have a working offline checkout. **Going live** (Paper +
bridge + a real episode) is a separate, ordered process — see [`RUNBOOK.md`](RUNBOOK.md).

---

## Who owns what

The repo is split by workstream so 8 people parallelize without colliding. Find
your track, read that directory's `README.md`, and start against the frozen
contract + hand-authored fixtures.

| Dir | Owner workstream | What lives here |
|-----|------------------|-----------------|
| [`env/`](env/) | Environment/bridge, PerceptionFilter, Reward | Observation spec, Gym env, reward, perception filter (per-file owners in [`env/README.md`](env/README.md)) |
| [`bridge/`](bridge/) | Environment/bridge | Node↔Python TCP bridge, Mineflayer bots, macro execution, event aggregation, reset RPC |
| [`agent/`](agent/) | DQN core | Dueling-DRQN net, sequence replay, training loop, action enum, configs |
| [`opponents/`](opponents/) | Reward/opponent | Stationary dummy + the `Opponent` interface future opponents implement |
| [`eval/`](eval/) | Eval/infra | Random tracer (M1), benchmark (the number), eval harness (M2), logging |
| [`server/`](server/) | Environment/bridge | Paper setup scripts, flat arena datapack, ops, `server.properties` rationale |
| [`distributed/`](distributed/) | *(deferred)* | Transport seam stub only — do not build out |
| [`deploy/`](deploy/) | *(deferred)* | Booth skeleton only |
| [`study/`](study/) | *(deferred)* | Study skeleton only |
| [`tests/`](tests/) | all | Unit + integration tests (one per component) |

---

## The four rules that keep 8 people from breaking each other

1. **The contract is frozen — change it only by PR.** The four contract artifacts —
   [`env/observation_spec.py`](env/observation_spec.py), the bridge JSON schema
   ([`bridge/schema.json`](bridge/schema.json) / [`bridge/messages.py`](bridge/messages.py)),
   [`agent/actions.py`](agent/actions.py), and the `compute_reward` signature in
   [`env/reward.py`](env/reward.py) — are imported by everyone. A renamed field
   mid-week cascades through 6 tracks. Touch one → PR everyone sees.

2. **Fairness lives in Python, never in the bridge.** Opponent state arrives raw;
   `PerceptionFilter` gates it before the observation. The leak battery
   ([`tests/test_perception_leak.py`](tests/test_perception_leak.py)) exists to
   catch any feature that reveals position when `visible == false`. Keep it green.

3. **M1's exit is a measured number, not a green check.** "It runs" is not the bar —
   transitions/s, p99 round-trip @200 ms, exact damage-event boundaries, and
   max-arenas-at-TPS are. The bridge + its event-aggregation window are the highest
   silent risk in the repo; that's why they're the most-tested code.

4. **A green M2 says *nothing* about the LSTM.** The M2 dummy is stationary and
   always visible — a fully-observed MDP a feed-forward net could solve. Recurrence
   is built ahead of need (user override) and is gated **separately** by the
   memory-dependent fixture TC8b ([`tests/test_dqn.py`](tests/test_dqn.py)), not by
   M2. Re-confirm TC8b before you trust any win-rate.

---

## Pinned versions (frozen)

Frozen in [`agent/contract_config.py`](agent/contract_config.py) (stdlib-only, no
heavy imports) after the day-1 compatibility check; evidence behind each pin is in
[`server/compat_check.md`](server/compat_check.md).

| Component | Pinned version | Notes |
|-----------|----------------|-------|
| Minecraft | `1.21.1` | Past the 1.9 attack-cooldown cutover the PvP combat model needs |
| Paper | `1.21.1` build `133` (channel STABLE) | Requires **Java 21+**; dev machine runs Java 25 |
| Node | `v24.13.0` | Node 24 "Krypton" LTS; mineflayer floor is `engines.node >=22` |
| Python | `3.11+` | Dev machine runs 3.14.2 |
| `mineflayer` | `4.37.1` | |
| `minecraft-data` | `3.110.2` | Transitive via mineflayer; pinned for reproducibility |
| `mineflayer-pvp` | `1.3.2` | Cooldown **reference only** — does *not* drive ATTACK |
| `mineflayer-pathfinder` | `2.4.5` | Demoted; movement uses `setControlState`, not pathfinder goals |

Every run computes a `code_version()` stamp (short git SHA + a hash of the frozen
config), written onto bridge `state` messages and saved into checkpoints. The
distributed future will reject actors whose `code_version` mismatches the learner;
the kickoff stack only logs it so train/serve skew stays visible.

---

## Running things

| Goal | Command | Notes |
|------|---------|-------|
| Offline tests | `pytest` · `cd bridge && npm test` | Fakes + fixtures; no game needed |
| Boot Paper | `pwsh -NoProfile -File server/setup/setup.ps1` then `start.ps1` | `setup` is idempotent; bash equivalents in [`server/setup/`](server/setup/) |
| Start the bridge | `cd bridge && npm start` | Start Paper **first** — bots connect before the port opens |
| M1 slice | `python -m eval.run_random --episodes 100 --host 127.0.0.1 --port 5555` | ≥100 eps, zero crashes |
| M1 benchmark | `python -m eval.benchmark --duration 600 --arenas 1` | Sweep `--arenas 2,3,4` for the max |
| M2 training | `python -m agent.train --max-episodes 10000 --eval-every-episodes 50 --eval-episodes 100 --checkpoint runs/m2.pt --run-name m2_train` | Live status bar + ETA; stops early when the gate passes |

**Run order for anything live: Paper → bridge → Python driver.** Full ordered
procedure, pass conditions, and what to watch (reward components, Q divergence) are
in [`RUNBOOK.md`](RUNBOOK.md).

---

## Documentation map

| Read this | When |
|-----------|------|
| **This README** | First. The mental model + where to start. |
| [`RUNBOOK.md`](RUNBOOK.md) | Taking the stack live and collecting M1 + M2. |
| [`docs/plans/2026-06-09-minecraft-pvp-kickoff.md`](docs/plans/2026-06-09-minecraft-pvp-kickoff.md) | The full picture: scope, decisions, data model, task table (T0–T20), acceptance criteria, risks. |
| [`docs/minecraft-pvp-project-spec.md`](docs/minecraft-pvp-project-spec.md) · [`docs/minecraft-pvp-training-spec.md`](docs/minecraft-pvp-training-spec.md) | The source specs the plan derives from. |
| Each `*/README.md` | Your workstream's specifics + per-file ownership. |
| [`server/compat_check.md`](server/compat_check.md) | Authority for version pins and the live-handshake follow-up. |

After M2 passes, the deferred dirs (`distributed/`, `deploy/`, `study/`) and the
M3/M4 ladder (scripted bot → self-play / PFSP / Elo) are the next horizon — all
explicitly out of kickoff scope.
