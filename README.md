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

## Setup (filled in by T6 once versions are pinned)

- **Python** 3.11+ with PyTorch — `pip install -r requirements.txt`
- **Node** LTS (verified by the day-1 compat check) — `cd bridge && npm install`
- **Paper** Minecraft server — see `server/README.md`

Pinned versions and determinism helpers land in `agent/contract_config.py`.

## Tests

```
pytest                 # Python unit + integration
cd bridge && npm test  # Node bridge smoke
```

The throughput/latency benchmark and the M2 eval are run on the dev laptop against
a live Paper server; see `eval/` and `server/README.md`.
