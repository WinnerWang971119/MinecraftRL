# Minecraft PvP RL

Reinforcement-learning agents that fight in Minecraft, trained on the real game
(no simulator) through a Mineflayer→bridge→Paper stack.

This repository is the **kickoff foundation**: a frozen interface contract, a real
end-to-end vertical slice (M1), and a Dueling-DRQN agent that beats a stationary
dummy ≥95% (M2). See `docs/plans/2026-06-09-minecraft-pvp-kickoff.md` for the plan
and `minecraft-pvp-project-spec.md` / `minecraft-pvp-training-spec.md` for the
source specs.

## Layout

| Dir | Owner workstream | What lives here |
|-----|------------------|-----------------|
| `env/` | Environment/bridge | Observation spec, Gym env, reward, perception filter |
| `bridge/` | Environment/bridge | Node↔Python TCP bridge, Mineflayer bots, macro execution |
| `agent/` | DQN core | Dueling-DRQN net, replay, training loop, action enum, config |
| `opponents/` | Reward/opponent | Dummy + opponent interface |
| `eval/` | Eval/infra | Random tracer, benchmark, eval harness, logging |
| `server/` | Environment/bridge | Paper server setup, arena, ops |
| `distributed/` | (deferred) | Transport seam stub only |
| `deploy/` | (deferred) | Booth skeleton only |
| `study/` | (deferred) | Study skeleton only |
| `tests/` | all | Unit + integration tests |

## Versions (pinned)

These versions are **frozen** in `agent/contract_config.py` (stdlib-only, no heavy
imports) after the day-1 compatibility check; see `server/compat_check.md` for the
evidence behind each pin.

| Component | Pinned version | Notes |
|-----------|----------------|-------|
| Minecraft | `1.21.1` | Past the 1.9 attack-cooldown cutover the PvP combat model needs |
| Paper | `1.21.1` build `133` (`paper-1.21.1-133.jar`, channel STABLE) | Requires **Java 21+**; dev machine runs Java 25 |
| Node | `v24.13.0` | Node 24 "Krypton" LTS; mineflayer floor is `engines.node >=22` |
| Python | `3.11+` | Dev machine runs 3.14.2 |
| `mineflayer` | `4.37.1` | |
| `minecraft-data` | `3.110.2` | Transitive via mineflayer; pinned for reproducibility |
| `mineflayer-pvp` | `1.3.2` | 1.9+ attack-cooldown-aware attack solver |
| `mineflayer-pathfinder` | `2.4.5` | |

Every run computes a `code_version()` stamp (short git SHA + a hash of the frozen
config) which is written onto bridge `state` messages and saved into checkpoints.
The distributed future will reject actors whose `code_version` does not match the
learner; the kickoff stack only logs it so train/serve skew stays visible.

## Setup

**Python agent** (3.11+, including 3.14):

```
python -m pip install -e .        # editable install of the agent package
# or, to install the listed runtime deps directly:
python -m pip install -r requirements.txt
```

(`torch` is the heavy dependency — install the CUDA/CPU variant for your machine,
see the note in `requirements.txt`. The determinism helpers in `agent/seeding.py`
degrade gracefully and still seed Python + NumPy if `torch` is not installed.)

**Node bridge** (Node `v24.13.0`):

```
cd bridge && npm install
```

The exact npm pins (`mineflayer@4.37.1`, `minecraft-data@3.110.2`,
`mineflayer-pvp@1.3.2`, `mineflayer-pathfinder@2.4.5`) live in
`bridge/package.json`; the authoritative copy is also recorded in
`agent/contract_config.py`.

**Paper server** (Paper `1.21.1` build `133`, Java 21+): download the jar and do
the first-boot/EULA/`server.properties` setup described in `server/README.md`,
following the live-handshake steps in `server/compat_check.md`.

## Tests

```
pytest                 # Python unit + integration
cd bridge && npm test  # Node bridge smoke
```

The throughput/latency benchmark and the M2 eval are run on the dev laptop against
a live Paper server; see `eval/` and `server/README.md`.
