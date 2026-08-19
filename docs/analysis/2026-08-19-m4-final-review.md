# M4 self-play — final pre-launch review (2026-08-19)

Branch `feat/m4-selfplay`, 33 commits off `bc8fd42`, +32,453 / −670 across 50 files.
Reviewed at `f164ee6`. Suites at review time: **2033 passed / 6 skipped / 2 deselected**
(Python), **305 passed / 1 skipped** (bridge).

Five independent reviewers, one per integration seam. Per-task review had already
passed for all 24 tasks, so this pass looked only for **cross-task drift** — two
commits each correct alone that disagree with each other.

**Verdict: no CRITICAL findings in any seam.**

## What each seam covered

| Seam | Acceptance criteria | Result |
|---|---|---|
| Opponent wire contract | AC1, AC10 | 0 critical, 1 warning, 3 notes |
| Mirrored observation / env | AC2, AC3, AC4 | 0 critical, 0 warnings, 3 notes |
| Snapshot pool / PFSP / Elo | AC5–AC8 | 0 critical, 1 warning, 4 notes |
| Train-loop wiring | AC5, AC7, AC11, AC13, AC14 | 0 critical, 2 warnings, 1 note |
| Launch-night ops | AC9, AC11, AC12, AC14 | 0 critical, 4 warnings, 4 notes |

## The one finding both an implementation seam and the ops seam raised independently

**AC14 was unenforced.** Both scripts hashed whatever file `--warm-start` named and
passed that self-computed digest to the driver as the expected value — a tautology
that catches a file changing between hash and load, never the wrong file. The decided
digest existed in exactly one place, this repo's plan document, and nothing machine-read
it.

The failure it permits: tab-complete `runs/m4.pt` instead of `runs/m4.best.pt` — same
directory, one keystroke apart, and `m4.pt` is the explicitly rejected 30k-step net.
Canary green, smoke green, plan cleared, launch runs all night on the wrong net and pins
it as permanently-pinned reference 0.

Closed by `--expect-sha256` on both scripts plus an automatic canary↔launch digest
cross-check. Details in the commit.

Verified digests:

| File | sha256 |
|---|---|
| `runs/m4.best.pt` (chosen) | `1d3d0c600e2ad76e49f2a6be10859f492ae2877c40f186b14f8924102774d5b2` |
| `runs/m4.pt` (rejected) | `c4afabf60ec7b88135c6339817999d449da1725b0c5e195da93eff3c123ac369` |

## Findings accepted without a code change

**A dead collector thread does not abort the run.** The collector loop catches only
`BridgeError` / `TransportError`, and `ActorPool`'s sole abort trigger is the JVM probe
(`distributed/actor.py:648-659`, `:1195-1233`). Any other exception silently retires that
arena with `aborted()` still False, so over 12 unattended hours the fleet can dwindle
from 25 toward zero while training "continues". Deliberately not patched before the run —
editing the actor pool hours before a 12-hour window trades a known, unlikely fault for
an unknown one. Mitigated instead by `scripts/watch_selfplay.sh`, a read-only checker.

**`PinnedSnapshotError` is downgraded at both layers above the pool.** The eval guard is
a blanket `except Exception`, so a corrupt pinned reference becomes "eval cycle SKIPPED …
Training continues" and `elo/learner_rated` freezes. Trigger requires an external fault:
`SnapshotPool.load` verifies pinned files at startup and `_atomic_write` replaces rather
than truncates.

**The human challenger's gear check is fail-open by design.** `confirm_human_loadout` is
server-authoritative but annotated "NEVER FAIL-CLOSED, per T15" — it logs
`COULD NOT CONFIRM n of 5` and plays anyway. This deviates from AC9's "failure aborts",
and the deviation was re-ratified during this review: a glitched NBT read must not
hard-stop a live demo in front of an audience. **Both bots remain fail-closed**, which is
what protects training. The operator reads the confirm line.

**The warm start's velocity columns are untrained, not neutral.** Indices 16-18
(`opp_vel_local`) were dead until this branch, so the plan's argument that "the net
learned to ignore those inputs" is mechanically wrong — zero input yields zero gradient,
leaving those encoder columns near random init. The learner trains them within the run;
**snapshot 0 cannot**, and it is a pinned reference all night. Consequence is that
`win_rate_vs_ref_0` measures M3-plus-noise rather than M3's true strength. Layout itself
is unchanged (`OBS_DIM` 23, `observation_spec.py` and `perception_filter.py` have zero
diff on this branch), and `m4.best.pt` loads `strict=True`.

## Reading tomorrow's curves

- **Not every row in `metrics.jsonl` carries a grad step.** `agent/train.py:4240` hands
  the run's logger to the main eval track and `eval/evaluate.py:525` logs `step=ep`, so
  each eval cycle appends one row per eval EPISODE numbered `0..n-1` into the same file,
  interleaved with training rows numbered by grad step. `agent/train.py:4256-4264` passes
  `logger=None` for the REFERENCE tracks with a comment naming this exact collision; the
  main track never got the same treatment. Nothing in training depends on the file
  (checkpoint selection reads in-memory `_ReferenceOutcome`), so this is an analysis trap
  rather than a defect: **select rows by key presence, never by `step` alone.** Found
  while reviewing the run watcher, which had assumed every row was a grad step.
- **`selfplay/win_rate_vs_ref_<id>` is the lifetime Beta-smoothed rate**, so after
  thousands of plays it moves ~1/N per match and a late collapse surfaces sluggishly.
  The collapse-sensitive series is **`selfplay/worst_reference_win_rate`** (per-cycle,
  raw), plus the per-cycle rates in the stderr gauntlet line.
- **Exit code 1 is not failure.** `_main_multi_arena` returns `0 if passed_m2 else 1`,
  and `passed_m2` is a bar set for a different opponent. Both scripts already treat the
  code as a non-signal and parse the `[multi done]` line instead.
- **`MatchResult.learner_epsilon` records the schedule ε, not the per-actor ε** the
  collector acts at. With per-actor ε on by default at α=7, that value is wrong for 24 of
  25 arenas. It does not affect training or rated-Elo eligibility — the rated path pins
  literal `0.0` — but do not condition any analysis on it.

## Environment facts confirmed at review time

- Java on PATH is 26.0.1, which SIGSEGVs in spark seconds after Paper reports Done.
  `start.sh` pins 21, `JAVA_HOME` is unset so it resolves via `java_home -v 21`, and
  Temurin 21.0.11 is installed. The mismatch path **aborts**, with `ALLOW_JAVA_MISMATCH=1`
  as the deliberate override.
- `ops.json` is generated by `start-pads.sh` for all 2N bots at boot; the issue #29
  restore chore no longer exists.
- 602 GiB free disk, 64 GiB RAM against a ~2.28 GB replay buffer.
- **Nothing in the repo prevents the host from sleeping.** `pmset` reports `sleep 1` and
  `disksleep 10`, held off only by transient assertions. A 12-hour unattended run needs
  `caffeinate`.
- The worktree has **no `.venv`** and no `runs/m4.best.pt`; both live in the main
  checkout. Every script defaults `PYTHON_BIN` to `${REPO_ROOT}/.venv/bin/python`, so
  `PYTHON` must be exported. The main tree's interpreter with the worktree as cwd imports
  `agent`/`env` from the worktree, so the canary's checkout-identity gate still passes.

## Method note

Reviewers were required to prove findings by mutation — break the guard, show the test
fails, restore — rather than by inspection alone. That standard is why the report is
short: several plausible-looking defects died on contact with a mutation that the suite
caught. The inverse also held, and is the more useful result: every load-bearing
invariant that was mutation-tested killed a test, which is the property that separates a
test from decoration.
