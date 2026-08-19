# M4 self-play — the overnight run (2026-08-19 18:36 → 2026-08-20 06:01)

Ran to its budget and stopped itself. No intervention, no restarts, no arena lost.

```
[multi done] reason=max_grad_steps episodes=125106 grad_steps=124885 passed_m2=True checkpoints_saved=22
  last eval: win_rate=1.000 mean_len=55.6 aim_invisible=0.000
  best checkpoint: win_rate=0.900 vs the pinned reference gauntlet (aggregate) at grad_step 93712
```

| | This run | M3 retry (bare-handed) |
|---|---|---|
| Gradient steps | **124,885** | 30,000 |
| Episodes | 125,106 | 30,503 |
| Rate | ~10,900/hour | 4,570/hour |
| Opponent | itself, armed and armored | unarmed, knockback-immune |
| Draw rate | 0.003 | — |

## Ship this one

**`runs/m4_selfplay.best.pt`, grad step 93,712** — scripted 1.000, reference aggregate
0.900, **reference worst 0.800**, rated Elo 1852.

It beats the FINAL net (124,885, Elo 1789) by 63 points, so the last 30k steps drifted
rather than improved. Judge by the gauntlet, not by recency.

## What the night actually looked like

| grad step | ref0 | ref5 | ref12 | aggregate | rated Elo |
|---|---|---|---|---|---|
| 6,500 | 1.000 | 0.700 | — | 0.850 | 1247.5 |
| 16,568 | 0.900 | 1.000 | **0.000** | 0.633 | 1426.0 |
| 27,598 | 0.800 | 0.600 | **0.100** | 0.500 | 1459.2 |
| 38,691 | 1.000 | 0.800 | 0.400 | 0.733 | 1611.1 |
| 52,697 | 1.000 | 0.900 | 0.500 | 0.800 | 1714.9 |
| 93,712 | — | — | — | **0.900** | 1852 |

`ref12` is the learner frozen at grad step 15,000. It beat the learner **10-0**, then
9-1, and finished losing 4 out of 5. That row is the run: the agent got stuck against a
version of itself and learned its way out.

Two things this corrects, both called wrong in real time and recorded here so the mistake
is not repeated:

- The mid-run slide (aggregate 0.850 → 0.633 → 0.500) read as collapse. It was
  **non-transitive cycling**, which is what PFSP exists to damp, and it resolved without
  intervention. The `ref0` decline of 1.000 → 0.900 → 0.800, which looked like forgetting,
  went straight back to 1.000.
- The aggregate is **not comparable across cycles**: it was scored over 2 references
  before grad step 15,000 and 3 after. Adding a harder reference lowers it mechanically.
  Compare per-reference rates, or compare only cycles with the same reference set.

## For the demo

The scripted opponent is **saturated** at a 1.000 win rate and cannot rank tonight's
checkpoints against each other. Select on the reference gauntlet and rated Elo.

Rated Elo is **pool-local**. 1852 means "beats snapshot 0 about 98% of the time" and
nothing else; there is no human anywhere in that pool, so it does not predict a human
match. To give the number a human anchor, play a fixed set of matches and rate the human
in the same pool.

The strongest net is not automatically the best demo. There are 22 periodic checkpoints
and 78 snapshots; if the 93,712 net fights cagily, a livelier earlier net makes a better
exhibition. Pick on match quality after a rehearsal, not on the leaderboard.

## Operational notes

- The watchdog (5-minute poll, exits only on ALARM or a dead driver) fired once, at
  06:01:56, on normal completion. `CronCreate` never fired at all across ~5 scheduled
  times: session cron only runs while the REPL is idle, which did not happen. A polling
  background task that exits on failure is the mechanism that worked.
- `--expect-sha256` verified the warm start against the recorded digest at both the canary
  and the launch, and the automatic canary/launch cross-check agreed. First real use of
  both gates, written the same afternoon.
- Throughput ran 31% above the smoke's projection (10,900 vs 8,326 grad steps/hour) with
  the machine 86% idle. The learner, not the fleet, is the ceiling: ~51 environment
  transitions were collected per gradient step, against a typical DQN ratio of 4 to 8.
  More arenas would add data nothing consumes.
