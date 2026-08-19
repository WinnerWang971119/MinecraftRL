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
> Running the classroom exhibition? [`docs/demo-day.md`](docs/demo-day.md).
> Just want to watch the bots fight with your own Minecraft client? That has its own
> short procedure and its own traps: [`docs/spectate.md`](docs/spectate.md).

---

## Disclosure: the agent's senses are fair, its turning is assisted

If you play against this agent, or watch someone play against it, you should know
this up front rather than find it in a source comment later.

**Fair.** Everything the agent *perceives* is honestly limited. Its observation
passes through a 70° field-of-view cone, a line-of-sight raycast, and a memory
timeout ([`env/perception_filter.py`](env/perception_filter.py)). It genuinely
cannot see behind itself, and circling out of its cone does make it lose you.

**Assisted.** One of its eight actions is named `TURN_TO_LAST_SEEN`, as if it
recalls where the opponent was last *seen*. It does not. `_updateLastSeen()` in
[`bridge/bot.js`](bridge/bot.js) writes the opponent's **live** world position
into that memory on every decision window, whether or not the agent can see them.
So that one action is an aim-snap onto where the opponent actually is, about
200 ms stale, regardless of line of sight.

The practical upshot: the agent can be blinded, but once it decides to turn it
cannot be dodged.

This was a deliberate, disclosed decision made under deadline. The two honest
fixes (gate the memory on real visibility, or resolve it properly and add real
facing actions) each invalidate the trained checkpoint, and there was one
training window left before the 2026-08-20 demo. It is written into the action's
own docstring in [`agent/actions.py`](agent/actions.py), frozen through
2026-08-20, guard-tested so nobody silently changes it before then, and recorded
in the plan's decision log
([`docs/plans/2026-08-16-demo-scripted-opponent-exhibition.md`](docs/plans/2026-08-16-demo-scripted-opponent-exhibition.md)).
It comes out after the demo.

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

> **The loadout changed, and no armored match has been played yet.** Both fighters —
> learner, opponent bot, and the human challenger at the exhibition — now get an iron
> sword **and a full iron set** (`spawn_learner_pad.mcfunction`,
> `spawn_dummy_pad.mcfunction`, and `human_gear_commands()` in
> [`deploy/exhibition.py`](deploy/exhibition.py)). Armor is applied with
> `item replace entity … armor.<slot>`, never `give`: **`give` does not equip**, it
> drops the piece in the inventory at zero armor points, which looks identical in a
> chat log. A full iron set takes ~48% off an incoming iron-sword hit (6 → ~3.12), so
> fights run ~1.75× longer and `MAX_EPISODE_STEPS` moved 400 → **600** (120 s). Every
> checkpoint in `runs/` predates the change and was trained bare-handed.

| Milestone | Question | Bar | Status |
|-----------|----------|-----|:------:|
| **M1 plumbing** (AC3) | does the loop survive the real bridge? | ≥100 episodes, 0 crashes, RSS growth < ~200 MB | ⬜ live run pending |
| **M1 the number** (AC4) | how fast / how many pads? | transitions/s, p99 round-trip, damage-exact, max pads @≥19 TPS | ⬜ live run pending — the scale ladder in [`RUNBOOK.md`](RUNBOOK.md) is an empty table on purpose |
| **Recurrence gate** (TC8b) | does the LSTM actually work? | memory fixture green, ablation fails | ✅ offline, re-confirm before trusting M2 |
| **M2 learning** (AC6) | does the RL stack learn? | greedy win ≥95% / 100 eps, no spin-farming | ⬜ live training pending |
| **M4 self-play** (AC7) | does it improve against itself? | `elo/learner_rated` rising, win rate vs pinned references not collapsing | ⬜ live run pending — `--opponent selfplay` exists; no self-play run has been done |

Taking the stack live and collecting these acceptances is exactly what
[`RUNBOOK.md`](RUNBOOK.md) walks through, in dependency order. **Done = the
milestones, not green tests.**

---

## Your first 30 minutes

**Prereqs** (the dev laptop has all of these; check yours):

| Need | Check | Why |
|------|-------|-----|
| **Java 21 installed** | `/usr/libexec/java_home -v 21` | Must print a path. Paper 1.21.1 boots on newer JDKs and then SIGSEGVs in spark's native profiler ~20 s after it reports Done. Your PATH `java` being 26 is fine and expected on this Mac: `server/setup/start.sh` resolves 21 itself and refuses to launch on anything else. macOS: `brew install --cask temurin@21` |
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
| [`deploy/`](deploy/) | Demo | `exhibition.py`: the one-command human-exhibition launcher and its separate reset command ([`docs/demo-day.md`](docs/demo-day.md)). Booth skeleton otherwise |
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
| Boot Paper | `bash server/setup/setup.sh` then `bash server/setup/start.sh` | `setup` is idempotent and re-copies the datapack into the world; `start.sh` resolves and pins Java 21. The `.ps1` files in [`server/setup/`](server/setup/) are Windows leftovers, unmaintained since the move to macOS and **not** kept in step with the shell scripts (their Java advice is wrong). macOS is the supported path |
| Start the bridge | `cd bridge && npm start` | Start Paper **first** — bots connect before the port opens |
| Damage-channel gate | `.venv/bin/python -m eval.combat_probe --cycles 10` | AC8, the go/no-go. Per-hit expectation is DERIVED from the target's loadout, not hardcoded: against the iron set both fighters now wear it is `3.12` x6 then `1.28`, cumulative 20. `--target-armor none` reproduces the historical `6,6,6,2` |
| M1 slice | `.venv/bin/python -m eval.run_random --episodes 100 --host 127.0.0.1 --port 5555` | ≥100 eps, zero crashes |
| M1 benchmark | `.venv/bin/python -m eval.benchmark --duration 600 --arenas 1` | Climb the rungs for the max; this is the AC4 **measurement** flag on `eval.benchmark` |
| M2 training (single pad) | `.venv/bin/python -m agent.train --max-episodes 10000 --eval-every-episodes 50 --eval-episodes 100 --checkpoint runs/m2.pt --run-name m2_train` | Live status bar + ETA; stops early when the gate passes |
| Multi-pad training | `PADS=N bash server/setup/setup.sh`, `bash server/setup/start-pads.sh --pads N` (wait for `FLEET READY`), then `.venv/bin/python -m agent.train --arenas N --port 5555 --max-episodes 10000 --checkpoint runs/m2_multi.pt --run-name m2_multi` | **One** Paper JVM, N enclosed pads 512 blocks apart, bridge port `5555+i`. `--arenas N` on `agent.train` is the **training** flag; distinct from `eval.benchmark --arenas` above |
| M4 self-play night | The ordered sequence in [`RUNBOOK.md`](RUNBOOK.md) (*The self-play night*): fleet, `scripts/canary_selfplay.sh`, `scripts/launch_selfplay.sh smoke`/`plan`/`launch` | A 12-hour unattended run. Two prerequisites are load-bearing and neither is a flag on the scripts: `export PYTHON=/Users/diego/Documents/MinecraftRL/.venv/bin/python` (the M4 worktree has no `.venv`, and `canary_selfplay.sh`, `launch_selfplay.sh` and `start-pads.sh` all default to one), and `caffeinate -dimsu` on the **fleet boot**, not the launch, because this Mac sleeps after a minute idle and `launch` detaches and returns in ~20 s |
| Check a live run | `bash scripts/watch_selfplay.sh --run-name m4_selfplay` | Read-only: writes nothing, signals nothing, opens no socket. Five badged signals; exit `0` OK/WARN, `1` ALARM, `2` usage, `3` UNKNOWN. Needs only a stdlib `python3` |
| Watch it live | [`docs/spectate.md`](docs/spectate.md) | Join with your own client. Read its "Before you join" first — you spawn in **survival**, inside pad 0 |
| Human exhibition | `.venv/bin/python -m deploy.exhibition --challenger-username <name>` | One command: Paper + bridge + the agent playing greedily from a checkpoint. Plays **one** match, then waits. Full procedure, the one-challenger protocol and the failure lookup table are in [`docs/demo-day.md`](docs/demo-day.md) |
| Arm the next challenger | type `reset` in Minecraft chat, or `.venv/bin/python -m deploy.exhibition --reset` | One mechanism, two triggers; chat is the demo-day path. Heals, repositions and re-gears both sides (sword + full iron set), reads the human's gear back off the server, and plays exactly one more match. Never automatic |

**Run order for anything live: Paper → bridge → Python driver.** Full ordered
procedure, pass conditions, and what to watch (reward components, Q divergence) are
in [`RUNBOOK.md`](RUNBOOK.md).

---

## Documentation map

| Read this | When |
|-----------|------|
| **This README** | First. The mental model + where to start. |
| [`RUNBOOK.md`](RUNBOOK.md) | Taking the stack live and collecting M1 + M2, plus the scale-ladder table. |
| [`docs/demo-day.md`](docs/demo-day.md) | Running the classroom exhibition: the one command, the one-challenger protocol, the reset, and what to do when something looks wrong mid-demo. |
| [`docs/spectate.md`](docs/spectate.md) | Joining the world with your own Minecraft client to check the bots by eye. |
| [`docs/plans/2026-08-19-m4-selfplay.md`](docs/plans/2026-08-19-m4-selfplay.md) | The M4 self-play design: the snapshot pool, PFSP, the two Elo series, the T17 canary and the T19 launch gate, and the decided warm start with its digest. |
| [`docs/plans/2026-08-08-damage-channel-fix-and-pad-topology.md`](docs/plans/2026-08-08-damage-channel-fix-and-pad-topology.md) | Why the damage channel was dead, the enclosed-pad topology, and the reward re-ordering. Current branch. |
| [`docs/analysis/2026-08-10-windows-archive.md`](docs/analysis/2026-08-10-windows-archive.md) | What the pre-repair training archive actually contains, and why no checkpoint in `runs/` is usable. |
| [`docs/plans/2026-06-09-minecraft-pvp-kickoff.md`](docs/plans/2026-06-09-minecraft-pvp-kickoff.md) | The full picture: scope, decisions, data model, task table (T0–T20), acceptance criteria, risks. |
| [`docs/minecraft-pvp-project-spec.md`](docs/minecraft-pvp-project-spec.md) · [`docs/minecraft-pvp-training-spec.md`](docs/minecraft-pvp-training-spec.md) | The source specs the plan derives from. |
| Each `*/README.md` | Your workstream's specifics + per-file ownership. |
| [`server/compat_check.md`](server/compat_check.md) | Authority for version pins, the Java pin, the world's block composition, and the connection throttle — all measured live on macOS. |

After M2 passes, the M3/M4 ladder (scripted bot → self-play / PFSP / Elo) is the
next horizon. `distributed/` is built (issue #4), and `deploy/exhibition.py` is
the human-exhibition path built for the 2026-08-20 demo. `study/` remains
skeleton-only and out of kickoff scope.
