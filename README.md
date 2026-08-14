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
> Just want to watch the bots fight with your own Minecraft client? That has its own
> short procedure and its own traps: [`docs/spectate.md`](docs/spectate.md).

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

The whole foundation is **built** and the **offline test suite is green**. The stack
has booted on macOS and both bots have joined a real Paper server, but the milestones
that need a sustained live run are still open.

> **The one thing to know before reading any historical result.** The
> opponent-damage channel was dead for the entire life of the project: `damage_dealt`
> was computed from the *learner's entity view* of the dummy, and mineflayer never
> populates `health` on a non-self entity — only on that bot's own connection. So
> landing a hit had never once paid a reward. It is now sourced from the dummy bot's
> own connection and the old path is deleted. Every checkpoint and metric in `runs/`
> predates the repair and was collected in a regime with no gradient on the one action
> that mattered — see
> [`docs/analysis/2026-08-10-windows-archive.md`](docs/analysis/2026-08-10-windows-archive.md).

| Milestone | Question | Bar | Status |
|-----------|----------|-----|:------:|
| **M1 plumbing** (AC3) | does the loop survive the real bridge? | ≥100 episodes, 0 crashes, RSS growth < ~200 MB | ⬜ live run pending |
| **M1 the number** (AC4) | how fast / how many pads? | transitions/s, p99 round-trip, damage-exact, max pads @≥19 TPS | ⬜ live run pending — the scale ladder in [`RUNBOOK.md`](RUNBOOK.md) is an empty table on purpose |
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
| **Java 21 exactly** | `java -version` | Paper 1.21.1 boots on newer JDKs and then SIGSEGVs in spark's native profiler; `server/setup/start.sh` pins 21 and refuses anything else. macOS: `brew install --cask temurin@21` |
| Node v24 (≥22) | `node -v` | mineflayer floor is `engines.node >=22` |
| Python 3.11+ **in a venv** | `.venv/bin/python --version` | System python on the dev Mac is 3.9.6, below the floor |

```bash
# 1. Clone, then install both sides.
#    NOTE: `pip install -e .` alone installs NOTHING — pyproject declares no
#    dependencies. requirements.txt is what actually pulls numpy/torch/pytest.
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e .        # optional: puts the packages on the path by name
cd bridge && npm install && cd .. # mineflayer + plugins (pinned in package.json)

# 2. Prove your checkout is healthy — this is the green light to start working.
.venv/bin/python -m pytest        # Python unit + integration
cd bridge && npm test && cd ..    # Node bridge smoke
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
| [`eval/`](eval/) | Eval/infra | Random tracer (M1), combat probe (the damage-channel gate), benchmark (the number), eval harness (M2), logging |
| [`server/`](server/) | Environment/bridge | Paper setup scripts, enclosed-pad arena datapack, ops, `server.properties` rationale |
| [`distributed/`](distributed/) | Multi-pad (issue #4) | N-pad collection into one learner: `Episode`+serialization, `LocalTransport`, `WeightStore`, `LearnerLoop`, `ActorPool`, `SubprocessArenaLauncher`, and `padAnchor(i)` — the **sole** source of pad coordinates; wired into `agent.train --arenas N` |
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
| Paper | `1.21.1` build `133` (channel STABLE) | **Java 21 exactly, not "21+"** — measured. Newer JDKs boot and then die with a native SIGSEGV in spark's profiler. Downloaded through PaperMC's **v3** API with a sha256 pin verified on every setup run; the old v2 endpoint returns 410 Gone |
| Node | `v24.13.0` | Node 24 "Krypton" LTS; mineflayer floor is `engines.node >=22` |
| Python | `3.11+` | Dev Mac runs 3.11.15 in `.venv`; its system python is 3.9.6 |
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

Commands assume the venv (`.venv/bin/python`); plain `python` works if you activate it.

| Goal | Command | Notes |
|------|---------|-------|
| Offline tests | `.venv/bin/python -m pytest` · `cd bridge && npm test` | Fakes + fixtures; no game needed |
| Boot Paper | `bash server/setup/setup.sh` then `bash server/setup/start.sh` | `setup` is idempotent and re-copies the datapack into the world; `start.sh` pins Java 21. PowerShell equivalents in [`server/setup/`](server/setup/) |
| Start the bridge | `cd bridge && npm start` | Start Paper **first** — bots connect before the port opens |
| Damage-channel gate | `.venv/bin/python -m eval.combat_probe --cycles 10` | AC8, the go/no-go: per-hit `6,6,6,2`, cumulative 20, reconciled against the wire's opponent health |
| M1 slice | `.venv/bin/python -m eval.run_random --episodes 100 --host 127.0.0.1 --port 5555` | ≥100 eps, zero crashes |
| M1 benchmark | `.venv/bin/python -m eval.benchmark --duration 600 --arenas 1` | Climb the rungs for the max; this is the AC4 **measurement** flag on `eval.benchmark` |
| M2 training (single pad) | `.venv/bin/python -m agent.train --max-episodes 10000 --eval-every-episodes 50 --eval-episodes 100 --checkpoint runs/m2.pt --run-name m2_train` | Live status bar + ETA; stops early when the gate passes |
| Multi-pad training | `PADS=N bash server/setup/setup.sh`, `bash server/setup/start-pads.sh --pads N` (wait for `FLEET READY`), then `.venv/bin/python -m agent.train --arenas N --port 5555 --max-episodes 10000 --checkpoint runs/m2_multi.pt --run-name m2_multi` | **One** Paper JVM, N enclosed pads 512 blocks apart, bridge port `5555+i`. `--arenas N` on `agent.train` is the **training** flag; distinct from `eval.benchmark --arenas` above |
| Watch it live | [`docs/spectate.md`](docs/spectate.md) | Join with your own client. Read its "Before you join" first — you spawn in **survival**, inside pad 0 |

**Run order for anything live: Paper → bridge → Python driver.** Full ordered
procedure, pass conditions, and what to watch (reward components, Q divergence) are
in [`RUNBOOK.md`](RUNBOOK.md).

---

## Documentation map

| Read this | When |
|-----------|------|
| **This README** | First. The mental model + where to start. |
| [`RUNBOOK.md`](RUNBOOK.md) | Taking the stack live and collecting M1 + M2, plus the scale-ladder table. |
| [`docs/spectate.md`](docs/spectate.md) | Joining the world with your own Minecraft client to check the bots by eye. |
| [`docs/plans/2026-08-08-damage-channel-fix-and-pad-topology.md`](docs/plans/2026-08-08-damage-channel-fix-and-pad-topology.md) | Why the damage channel was dead, the enclosed-pad topology, and the reward re-ordering. Current branch. |
| [`docs/analysis/2026-08-10-windows-archive.md`](docs/analysis/2026-08-10-windows-archive.md) | What the pre-repair training archive actually contains, and why no checkpoint in `runs/` is usable. |
| [`docs/plans/2026-06-09-minecraft-pvp-kickoff.md`](docs/plans/2026-06-09-minecraft-pvp-kickoff.md) | The full picture: scope, decisions, data model, task table (T0–T20), acceptance criteria, risks. |
| [`docs/minecraft-pvp-project-spec.md`](docs/minecraft-pvp-project-spec.md) · [`docs/minecraft-pvp-training-spec.md`](docs/minecraft-pvp-training-spec.md) | The source specs the plan derives from. |
| Each `*/README.md` | Your workstream's specifics + per-file ownership. |
| [`server/compat_check.md`](server/compat_check.md) | Authority for version pins, the Java pin, the world's block composition, and the connection throttle — all measured live on macOS. |

After M2 passes, the M3/M4 ladder (scripted bot → self-play / PFSP / Elo) is the
next horizon. `distributed/` is now built (issue #4); `deploy/` and `study/`
remain skeleton-only and out of kickoff scope.
