# Windows `runs/` archive analysis (T5 / AC9)

**Plan:** `docs/plans/2026-08-08-damage-channel-fix-and-pad-topology.md` — task T5, satisfies AC9, test case TC8.
**Archive:** `~/Downloads/runs_data_20260810.zip` (13,534,432 bytes, 20 entries).
**Analysed:** 2026-08-10, on macOS, entirely in a scratchpad. No run data was copied into the repository.

## Verdict summary

All verdicts are over the **453 recorded episodes, which are eval episodes only** — the archive
contains no per-episode training records (§1).

| # | Prediction | Verdict |
|---|---|---|
| 1 | `r_damage_dealt` is exactly `0.0` | **CONFIRMED** over all 453 recorded (eval) episodes |
| 2 | Won episodes show `damage_dealt == 0` | **CONFIRMED** — 4 of 4 won episodes among the 453 recorded (eval) episodes, n=4 |
| 3 | Losses are near-absent | **CONFIRMED** — 0 losses in all 453 recorded (eval) episodes |
| 4 | Wander-offs appear as timeouts with episode-end `y ≈ −60`, not deaths | **SPLIT** — "not deaths" confirmed over the 453 recorded (eval) episodes; `y ≈ −60` **NOT TESTABLE** (no positional field is logged anywhere in the archive) |

**Halt rule: NOT triggered.** Zero nonzero, NaN, or infinite `r_damage_dealt` values were found anywhere in the archive.

---

## 1. Scope — what this archive can and cannot answer

This is the single most important framing for reading the numbers below, and it narrows the
plan's quantifier.

**The archive contains eval episodes only.** Every per-episode row in every `metrics.jsonl`
carries the exact key set written by `eval/evaluate.py` (see §4). Training episodes appear in the
archive **only as `progress/*` heartbeat counters** — aggregate episode/step/epsilon counts sampled
every ~30 s, with no per-episode record of reward components or outcomes. `m2_train` alone ran
1,498 training episodes; not one of them has a recoverable `r_damage_dealt`.

Consequently every verdict in this report is stated as **confirmed or refuted over all *recorded*
episodes (453 eval episodes)**, not over the literal "every episode of every run" in the plan's
prediction text. Training episodes are untested because this era's writer logged no per-episode
training metrics. This is a limitation of the archive, not a finding about the runs.

**The scope split is in the evidence, not in the mechanism.** Training and eval episodes ran the
same bridge process and therefore the same `bridge/bot.js` recorder — the dead `_onEntityHurt`
damage path is upstream of the train/eval distinction and cannot behave differently between them.
So the 8,780 untested training episodes counted by the heartbeats (plus `m2_train_5a`'s, which logged
no heartbeats at all) were produced by the identical broken channel. That is an argument from the
code's structure, not archive evidence, and it is offered as
mechanism only: this analysis measures 453 episodes and claims nothing more.

**The archive contains no configs and no console logs.** All 20 entries are `metrics.jsonl`,
`summary.json`, or `.pt`. There is no `train_config`, no `reward_config`, no server log, no bridge
log, and no nested run directories. The reward coefficients that produced each run are therefore
**not recorded anywhere** and had to be inferred from the logged component values (§5).

---

## 2. Archive integrity

The zip was written on Windows and uses `\` path separators, so an extractor that does not translate
them produces literal files named `m2_train\metrics.jsonl` instead of nested directories. The
extraction was therefore re-done into a fresh directory and compared against the pre-existing one.

- `ZipFile.testzip()` → `None` (no CRC failures).
- 20/20 entries extracted; every extracted size equals the central-directory `file_size`.
- SHA-256 of all 20 files is **identical** between the fresh extraction and the pre-existing
  `unzip` output, and the pre-existing directory contains no literal-backslash filenames.

The pre-existing extraction is sound. All numbers below come from verified bytes.

**Method correction — do not reuse the shortcut.** Python's `zipfile` does **not** normalize the
separator for you. `os.altsep` is `None` on POSIX, so a plain `ZipFile.extractall()` on macOS
produces literal-backslash filenames — verified here by running it into a temp directory, which
yielded exactly 14 such names and zero subdirectories. The fresh extraction used above is correct
only because each member name was normalized manually (`info.filename.replace('\\', '/')`) before
writing. Every byte matched regardless, so nothing downstream is affected, but a reader reusing
"just use `zipfile`" on a Windows-authored archive would get the broken layout.

**Timezone note.** Zip entry mtimes are local Windows time; `wall_time` inside the files is UTC.
The two differ by exactly +8 h throughout (e.g. `m2_train_5a` last row `2026-07-05 01:09 UTC`,
mtime `2026-07-05 09:09`), so the source machine was UTC+8. Dates in this report are labelled.

---

## 3. Artifact inventory (all 20 entries)

### 3.1 Run directories (7)

| Run | `metrics.jsonl` bytes | rows | `summary.json` bytes | Complete? |
|---|---:|---:|---:|---|
| `m2_train` | 278,331 | 813 | 909 | yes |
| `m2_train_5a` | 27,991 | 101 | 904 | yes (oldest writer schema — see §4) |
| `m2_train_5a_4` | 310,795 | 881 | 938 | yes |
| `m2_train_5a_5` | 437,650 | 1,208 | 904 | yes |
| `m2_train_6a` | 378,072 | 1,118 | 874 | yes (3 eval passes; summary covers only the last) |
| `m2_train_6a_2` | 75,894 | 214 | 213 | **INCOMPLETE** — eval aborted after 3 episodes |
| `m2_train_8a` | 587 | 2 | 214 | **INCOMPLETE** — no eval; two heartbeats 30 s apart |

Every line in every `metrics.jsonl` parsed as JSON. **0 parse errors, 0 unreadable files, 0
unrecognized artifact types** across the archive.

### 3.2 Checkpoints (6)

All six load cleanly with `torch.load(..., map_location='cpu', weights_only=True)` and contain
exactly three top-level keys: `model`, `grad_step`, `code_version`. Architecture is **identical**
across all six — a 23-dim input (matching the frozen `OBS_DIM = 23`), a 256-unit encoder, a
256-unit LSTM, and dueling value/advantage heads with 8 actions; 600,585 parameters each.
No checkpoint was evaluated (out of scope for T5).

| Checkpoint | bytes | `grad_step` | `code_version` | mtime (local) | Mapping to a run |
|---|---:|---:|---|---|---|
| `m2_5arenas.pt` | 2,407,221 | 1,401 | `nogit+cfg403300e3` | 2026-07-05 09:09 | `m2_train_5a` — **timestamp-exact** (same minute as its `summary.json`) |
| `m2_5a_2.pt` | 2,406,591 | 6 | `nogit+cfg403300e3` | 2026-07-05 16:10 | falls inside `m2_train`'s window; near-initialization checkpoint |
| `m2_5a_3.pt` | 2,406,591 | 1,404 | `nogit+cfg403300e3` | 2026-07-06 00:17 | adjacent to `m2_train` (final step 1,496; ~30 min before its last heartbeat) — **not exact** |
| `m2_5a_4.pt` | 2,406,591 | 1,930 | `8b4c151+cfg403300e3` | 2026-07-07 05:11 | `m2_train_5a_4` — **exact** (`grad_step` == final step 1,930) |
| `m2_5a_5.pt` | 2,406,591 | 2,110 | `8b4c151+cfg403300e3` | 2026-07-07 16:18 | `m2_train_5a_5` — **exact** (`grad_step` == final step 2,110) |
| `m2_6a.pt` | 2,406,491 | 779 | `nogit+cfg403300e3` | 2026-07-06 04:44 | mid-run periodic checkpoint inside `m2_train_6a`'s window (final step 2,730) |

**`m2_5a_2.pt` and `m2_5a_3.pt` have no run directory of their own** in the archive — they are
listed here rather than skipped. Only two of six checkpoints (`m2_5a_4`, `m2_5a_5`) map to a run by
an exact `grad_step` identity; the rest are matched by timestamp containment only, which is
suggestive but not proof.

The 630-byte size difference of `m2_5arenas.pt` is container/metadata overhead, not architecture —
its tensor shapes and parameter count are byte-for-byte the same as the others'.

### 3.3 What is absent

No configs, no console/server/bridge logs, no nested run directories, no replay buffers, no
TensorBoard event files. The plan's T5 description asks for these to be inventoried; they are
**not present in the archive**.

---

## 4. Schema — verified against the code, then against the data

Read from `eval/logging.py` (`_METRICS_FILENAME`, `_SUMMARY_FILENAME`), `eval/evaluate.py`
(the row it emits), and `env/reward.py` (`REWARD_COMPONENT_KEYS`). The complete set of keys ever
observed in the whole archive is closed at three row shapes:

| Row shape | Keys | Source |
|---|---|---|
| **Eval episode** | `step`, `wall_time`, `episode_length`, `episode_reward`, `win`, `aim_while_invisible`, `r_damage_dealt`, `r_damage_taken`, `r_step`, `r_aim`, `r_shaping`, `r_terminal` | `eval/evaluate.py` |
| **Progress heartbeat** | `step`, `wall_time`, `progress/{episodes,total_episodes,steps,frac,eps_per_min,steps_per_s,elapsed_s,eta_budget_s,epsilon}` | training loop |
| **Epsilon** | `step`, `wall_time`, `train/epsilon_mean` | training loop |

**Schema divergence (reported as AC9 requires).** `m2_train_5a` is the only run with **no progress
heartbeat rows at all** — its file is one `train/epsilon_mean` row followed by 100 eval rows, and
its first line is a `step: 1000` epsilon row. It is the oldest run in the archive (2026-07-04/05)
and evidently predates the progress-heartbeat writer. Its eval-row schema is identical to every
other run's, so it is fully comparable for this analysis. No other run diverges.

**Three facts that follow from the schema and matter for AC9:**

1. **There is no positional field.** No `y`, no position, no coordinate, no height — in any of the
   three row shapes, or in any `summary.json`. This closes prediction 4's positional half as
   untestable from this archive (§7, prediction 4).
2. **`damage_dealt` itself is never logged** — only `r_damage_dealt = c_dmg_out · damage_dealt`.
   `c_dmg_out = 1.0` at every commit in the project's history (verified in §8), so
   `r_damage_dealt == 0.0` ⟺ `damage_dealt == 0`.
3. **Loss vs. timeout is not a per-episode field.** The row carries only `win`. Outcomes were
   therefore reconstructed from `r_terminal`, whose value set is disjoint across the three
   outcomes in both reward regimes (§5), and every reconstruction was cross-checked against
   `summary.json`'s `n_wins`/`n_losses`/`n_timeouts` (§4.1).

### 4.1 Reconciliation of the reconstruction

`metrics.jsonl` is opened in **append mode across process restarts**, so a file may contain several
eval passes and several training launches while `summary.json` is rewritten to reflect only the
**last** pass. Passes were separated by `step` resets; launches by `progress/episodes` resets and
`progress/total_episodes` changes.

For every run with eval results, the reconstructed last pass matches `summary.json` exactly:
episode count, `sum.r_terminal`, all six `mean.r_*` components, and `mean_episode_length`. Worked
example for `m2_train`: 99 timeouts × (−30) + 1 win × (+50) = **−2,920**, equal to its recorded
`sum.r_terminal` to the last digit.

**A discrepancy worth recording:** `m2_train_6a` contains **3 eval passes** (150 episodes) and
**3 wins**, all in the first pass — but its `summary.json` reports `n_wins: 0`, `win_rate: 0.0`,
because it describes only the final pass. Reading summaries alone would have hidden three quarters
of the archive's wins, and with them most of prediction 2's evidence. This report scans all passes.

### 4.2 The additivity check kills the "metric was never logged" hypothesis

This is the one alternative explanation that would produce **byte-identical data** to a dead damage
channel, so it deserves to be dispatched explicitly rather than assumed away.

**Why key-presence proves nothing.** `eval/evaluate.py:531` accumulates components as
`components[key] += float(info.get(key, 0.0))`, with every component pre-initialized to `0.0` at
`:509`. That is a literal silent-zero path: if the env stopped putting `r_damage_dealt` into `info`,
the key would still appear in the JSONL, still read `0.0`, and be **indistinguishable** from a
correctly-plumbed channel reporting genuine zero damage. Seeing the key in the output is not
evidence that the metric was live.

**What discriminates them.** The same loop accumulates the env's scalar reward *independently* at
`:528` — `total_reward += float(reward)` — and `compute_reward` returns
`float(sum(components.values()))`. That return statement was verified as the actual code at both
era commits (`8b4c151` and `136ff0d`), not merely asserted by the docstring. So the scalar and the
component breakdown come from the same sum by construction, and **a dropped key would leave a
residual in `episode_reward` exactly equal to the missing damage reward**.

**Result:** for every one of the 453 episode rows, the six components sum to `episode_reward` to
within **1.847 × 10⁻¹³** (worst case — pure floating-point noise, ~10 orders of magnitude below the
smallest reward term `c_step = 0.005`). No key was dropped on any step of any episode. The
`r_damage_dealt` zeros are **reported zeros from a live logging path**, not absent measurements.

The channel was plumbed and reporting. It was reporting zero because `recordDamageDealt()` was
never called — which is the diagnosis, not an artifact of the logger.

---

## 5. Reward regimes — the runs are not one population

`config.code_version` is identical (`cfg403300e3`) for every run in the archive, but **this proves
nothing about the reward**: `agent/contract_config.config_fingerprint()` hashes only the version
pins and timing constants (Minecraft/Paper/Java/Node/Python versions, `ACTION_REPEAT`,
`DECISION_INTERVAL_MS`, `MAX_EPISODE_STEPS`, npm pins). Reward coefficients are not in the hash, so
the fingerprint is blind to PR #21's reshape. The regime had to be recovered from the data.

`r_terminal` takes exactly three distinct values across the entire archive — `0.0`, `−30.0`, `+50.0`
— which partitions the runs cleanly:

| Regime | Terminal values observed | Runs | Chronology (local) |
|---|---|---|---|
| **A — pre-reshape** | win n/a, timeout `0.0` | `m2_train_5a` | 2026-07-05 06:26 → 09:09 |
| **B — post-reshape** | win `+50.0`, timeout `−30.0` | `m2_train`, `m2_train_6a`, `m2_train_6a_2`, `m2_train_5a_4`, `m2_train_5a_5`, (`m2_train_8a`, no episodes) | 2026-07-05 14:01 → 2026-07-07 16:18 |

So the reshape landed between `m2_train_5a` (ends 2026-07-05 09:09) and `m2_train`
(starts 2026-07-05 14:01), and six of the seven runs are post-reshape.

**Confirmed against the commit record.** The inference above was made from the logged values alone;
git then confirms it exactly. The reshape is commit **`8b4c151` — "reshape the combat reward so the
agent stops kiting", `Sun Jul 5 13:04:30 2026 +0800`** — which falls inside the inferred
09:09 → 14:01 window. Its diff to `agent/reward_config.py` changes exactly three coefficients:

| Coefficient | Before | After |
|---|---:|---:|
| `c_dmg_in` | 1.0 | **0.5** |
| `R_terminal_win` | 8.0 | **50.0** |
| `R_terminal_timeout` | 0.0 | **−30.0** |

That is the plan's three-item reshape inventory verbatim, and it confirms both the regime partition
and the `−19.0` / `−9.5` reading below. `c_dmg_out` is not touched by that commit.

**Independent confirmation of the `c_dmg_in` 1.0 → 0.5 half of the reshape.** `m2_train_5a`
(regime A) books `r_damage_taken = −19.0` in every one of its 100 episodes; `m2_train`
(regime B) books `−9.5` in every one of its 100 — both with zero variance. Given `8b4c151`'s
verified `c_dmg_in` 1.0 → 0.5, these are the *same physical 19 HP*, scored at 1.0 and 0.5
respectively. The third and most easily-overlooked item in the reshape inventory is therefore
recoverable from the logged data alone, and the commit record confirms the reading.

**The 19 HP signature.** 19 HP is exactly `20 − 1`, i.e. damage that stops at 1 HP. The plan's
background notes that reset heals HP but never restores food or saturation, and that starvation on
normal difficulty stops at 1 HP. The deterministic, zero-variance 19 HP is **consistent with**
starvation-to-1-HP. Labelled as a hypothesis, not a finding — nothing in the archive identifies the
damage source.

**Unexplained cross-run difference (reported, not explained).** The four later runs
(`m2_train_5a_4`, `m2_train_5a_5`, `m2_train_6a`, `m2_train_6a_2`) record `r_damage_taken = 0.0`
in every episode — the 19 HP disappears entirely. The archive contains no config or log that would
say what changed. Recorded here as an open discrepancy.

### 5.1 A third regime marker: `r_shaping` reveals uncommitted config drift

The A/B partition above is by terminal values only, and it is **incomplete**. A third coefficient
varied across these runs, and it is the sharpest provenance finding in the archive.

`r_shaping` is exactly `0.0` in three runs and multi-valued in three others:

| Run | Distinct `r_shaping` values | Range | `config.seed` | `config.code_version` |
|---|---:|---|---:|---|
| `m2_train` | 1 | `0.0` only | 0 | `nogit+…` |
| `m2_train_5a` | 1 | `0.0` only | 0 | `nogit+…` |
| `m2_train_6a` | 1 | `0.0` only | 0 | `nogit+…` |
| `m2_train_5a_4` | **17** | −0.337583 … +1.729412 | **20260706** | `8b4c151+…` |
| `m2_train_5a_5` | **15** | 0.0 … +0.208071 | 0 | `8b4c151+…` |
| `m2_train_6a_2` | **2** | 0.0 and +0.122437 | 0 | `nogit+…` |

`r_shaping = γ·Φ(s′) − Φ(s)` is identically zero whenever `c_approach == 0.0` (`env/reward.py:248`
short-circuits). A nonzero value therefore proves `c_approach != 0` for those three runs.

**But `c_approach` is `0.0` in every commit that has ever touched `agent/reward_config.py`** — all
three of them (`cf0a4b8`, `8b4c151`, `136ff0d`), verified directly. The only non-default assignments
anywhere in the tree are `dataclasses.replace(RewardConfig(), c_approach=…)` calls inside
`tests/test_reward.py`. So `m2_train_5a_4`, `m2_train_5a_5`, and `m2_train_6a_2` ran with a reward
configuration **unreachable from any committed code path** — uncommitted local config drift.

**This is worse than it looks, because the stamp cannot detect it.** `code_version()` builds its
SHA half from `git rev-parse --short HEAD` (`agent/contract_config.py:170-180`) with **no
`--dirty` flag**. A recorded commit therefore does not imply a clean tree. `m2_train_5a_4` and
`m2_train_5a_5` both stamp `8b4c151` — a commit whose `c_approach` is `0.0` — while their own data
proves it was not. **The run stamp actively contradicts the run's behavior, and nothing in the
recording pipeline would have flagged it.**

Note also that five of the seven runs stamp `nogit`, meaning `git rev-parse` failed and the code
state for those runs is **entirely unrecoverable**. Combined with §5's finding that the `cfg`
fingerprint is blind to reward coefficients, the archive has three independent provenance gaps: no
config files, a fingerprint that omits the reward, and a SHA that is either absent or not
dirty-checked.

`m2_train_5a_4` is also the only run with a non-zero seed (`20260706`; every other run used `0`),
so it differs from its siblings on at least two axes.

**Scope note:** this is the same class of finding as the 19 HP → 0 flip — an environment difference
the archive records but does not explain. It does not affect any prediction verdict:
`r_damage_dealt` is `0.0` in all three of these runs as in every other.

---

## 6. Per-run results (AC9)

`c_dmg_out` is a positive constant, so `r_damage_dealt` and `damage_dealt` are the same statement
(§8). "Distribution" is reported as the observed value set over all recorded episodes of the run.

| Run | Regime | Recorded eval episodes (passes) | `r_damage_dealt` distribution | Win / Loss / Timeout | Won episodes with zero damage | Episode-end `y` |
|---|---|---:|---|---|---|---|
| `m2_train` | B | 100 (1) | **all exactly `0.0`** (min = max = 0.0, no NaN/inf) | 1 / 0 / 99 | **1 of 1** | not logged |
| `m2_train_5a` | A | 100 (1) | **all exactly `0.0`** | 0 / 0 / 100 | n/a (no wins) | not logged |
| `m2_train_5a_4` | B | 50 (1) | **all exactly `0.0`** | 0 / 0 / 50 | n/a (no wins) | not logged |
| `m2_train_5a_5` | B | 50 (1) | **all exactly `0.0`** | 0 / 0 / 50 | n/a (no wins) | not logged |
| `m2_train_6a` | B | 150 (3) | **all exactly `0.0`** | 3 / 0 / 147 | **3 of 3** | not logged |
| `m2_train_6a_2` | B | 3 (1, aborted) | **all exactly `0.0`** | 0 / 0 / 3 | n/a (no wins) | not logged |
| `m2_train_8a` | B (assumed) | **0** | n/a — no episodes recorded | — | — | not logged |
| **TOTAL** | — | **453** | **all exactly `0.0`** | **4 / 0 / 449** | **4 of 4** | not logged |

### Supporting per-run detail

| Run | Episode lengths | `aim_while_invisible` | `r_aim` range | `r_damage_taken` | Training episodes reached |
|---|---|---|---|---|---|
| `m2_train` | 99 × 400 (cap), 1 × 310 (the win) | all `0.0` | 0.00 – 3.99 | −9.5 (all) | 1,498 |
| `m2_train_5a` | 100 × 400 | all `0.0` | 0.00 – 3.74 | −19.0 (all) | not recorded (no heartbeats) |
| `m2_train_5a_4` | 50 × 400 | all `0.0` | 0.00 – 3.71 | 0.0 (all) | 1,935 |
| `m2_train_5a_5` | 50 × 400 | all `0.0` | 0.00 – 0.00 | 0.0 (all) | 2,113 |
| `m2_train_6a` | 147 × 400, + 88 / 104 / 158 (the 3 wins) | all `0.0` | 0.00 – 0.22 | 0.0 (all) | 2,732 |
| `m2_train_6a_2` | 3 × 400 | all `0.0` | 0.00 – 0.00 | 0.0 (all) | 501 |
| `m2_train_8a` | — | — | — | — | 1 |

Two facts stand out. **Every single non-win episode in the archive ran to exactly the 400-step
cap** — 449 of 449. No episode ever ended early for any reason other than a win. And
`aim_while_invisible` is exactly `0.0` in all 453 episodes, so the anti-spin-farm invariant
(AC6/TC6) held throughout the archived history.

---

## 7. The four predictions

### Prediction 1 — `r_damage_dealt` is exactly `0.0`. **CONFIRMED.**

Every one of the 453 recorded episode rows was scanned individually for an exact `!= 0.0`
comparison plus a finiteness check. **Result: 0 nonzero, 0 NaN, 0 infinite.** The value set of
`r_damage_dealt` over the entire archive is the single element `{0.0}`.

The row-level scan is the evidence, not the aggregates. (`sum.r_damage_dealt == 0.0` in every
summary corroborates it — since `r_damage_dealt = c_dmg_out · damage_dealt ≥ 0` by construction, a
zero sum forces every term to zero — but a NaN would not have shown up in a sum, which is why the
row scan was run.)

This is exactly what the root-cause diagnosis predicts: `recordDamageDealt()` was never called,
because `_onEntityHurt` read `entity.health` on a non-self entity, which mineflayer never populates.

### Prediction 2 — Won episodes show `damage_dealt == 0`. **CONFIRMED.**

The archive contains **4 won episodes** in total, and all 4 have `r_damage_dealt == 0.0`:

| Run | Pass | `step` | `episode_length` | `r_terminal` | `r_damage_dealt` | `episode_reward` |
|---|---|---:|---:|---:|---:|---:|
| `m2_train` | 1 of 1 | 57 | 310 | +50.0 | **0.0** | 38.95 |
| `m2_train_6a` | 1 of 3 | 0 | 88 | +50.0 | **0.0** | 49.56 |
| `m2_train_6a` | 1 of 3 | 4 | 158 | +50.0 | **0.0** | 49.23 |
| `m2_train_6a` | 1 of 3 | 12 | 104 | +50.0 | **0.0** | 49.48 |

This is the plan's sharpest fingerprint, and it is present exactly as described. Each of these
episodes ended well short of the 400-step cap because the dummy actually **died** — at least 20 HP
of damage was demonstrably delivered in-world (at least, because `naturalRegeneration` was still on
during these runs; disabling it is new scope in this plan, so interleaved heals mean the true total
can only be higher) — and the per-hit channel recorded **none of it**. The kill
outcome was paid (`dummy.on('death') → recordOpponentDied`, always correctly wired) while every hit
that produced it scored zero. The two halves of the diagnosis are visible in the same four rows.

**Sample size stated plainly: n = 4.** Three of the four are in an eval pass that
`m2_train_6a/summary.json` does not describe. The prediction is confirmed on every won episode that
exists in the archive, but four episodes is a small base, and this is corroborating evidence for a
root cause already established at primary source — not independent proof of it.

### Prediction 3 — Losses are near-absent. **CONFIRMED (stronger than "near").**

**0 losses in 453 episodes.** Not near-absent — absent. `r_terminal`'s entire observed value set
across the archive is `{0.0, −30.0, +50.0}`, and every `summary.json` with eval results reports
`n_losses: 0`. Combined with the fact that all 449 non-win episodes ran to the exact 400-step cap,
no episode in the archive ended in the agent's death by any mechanism.

The commit record sharpens this from "no loss-shaped value appeared" to a positive test:
`R_terminal_loss = 8.0` at both era commits (`cf0a4b8` and `8b4c151`), and the reward applies it as
`−R_terminal_loss`, so **a loss would have logged `r_terminal = −8.0` exactly**. That value does
not occur once in 453 episodes. The absent outcome has a known signature, and the signature is
absent.

This does not refute the geometry analysis. It is what that analysis predicts.

### Prediction 4 — Wander-offs appear as timeouts with episode-end `y ≈ −60`, not deaths. **SPLIT.**

**"Not deaths" — CONFIRMED.** 0 deaths in 453 episodes; all 449 non-win episodes terminated by
reaching the step cap. Whatever wandering occurred, it never produced a terminal event.

**"`y ≈ −60`" — NOT TESTABLE from this archive.** There is no `y`, position, or coordinate field in
any of the three row shapes or in any `summary.json` (§4). The writer of this era logged no
positional data at all, so the archive cannot confirm or refute where episodes ended. **This is
reported as untestable rather than confirmed** — the prediction's positional half remains open on
this evidence.

What the archive *is* consistent with: an agent that walks off the platform, is stranded, and runs
out the clock produces exactly the observed signature — a timeout at the full 400-step cap with no
death. The interpretation that such an agent lands alive rests on the **live column scan performed
in this session** (recorded in `server/compat_check.md`, reproducible via
`server/tools/probe_world.js`), which found solid ground at y = −61 (grass/dirt/dirt) with bedrock
at y = −64 everywhere including outside the pad footprint, and verified `fallDamage false` live.
That live evidence establishes the mechanism; this archive establishes only that no deaths
occurred. Confirming `y ≈ −60` end-of-episode would require re-instrumenting the logger, which is
outside T5.

---

## 8. Halt rule

**The halt rule did not trigger.**

The scan compared every `r_damage_dealt` value in all 453 episode rows against `0.0` exactly (not
by tolerance), and separately checked each for NaN and infinity. Zero rows failed. There is no
nonzero value in this archive to reconcile, and nothing in the data contradicts the root-cause
diagnosis the plan rests on.

**The one load-bearing dependency — now verified, not assumed.** `r_damage_dealt == 0` implies
`damage_dealt == 0` only if `c_dmg_out ≠ 0`; a zero coefficient would make the whole result vacuous.
`c_dmg_out` is not recorded anywhere in the archive, so this was initially carried as an explicit
assumption. It has since been checked against the commit record:

**`c_dmg_out = 1.0` at every commit that has ever touched `agent/reward_config.py`** — `cf0a4b8`
(1.0), `8b4c151` (1.0), `136ff0d` (1.0). The reshape commit's diff does not touch it. There is no
point in the project's history at which `c_dmg_out` was zero, or anything other than 1.0.

The `c_dmg_out = 0` alternative is therefore **dead, not merely argued against**. Under
`c_dmg_out = 1.0`, `r_damage_dealt` *is* `damage_dealt` in HP, and the 453 zeros mean 453 episodes
in which the agent landed no measured damage at all.

Together with §4.2 (no key was ever dropped, so the zeros are reported values from a live path),
the two ways this result could have been an artifact — a vacuous coefficient, or an absent
measurement — are both closed.

---

## 9. Anomalies, incomplete artifacts, and caveats

Listed rather than skipped, per AC9.

**Incomplete runs (2):**

- **`m2_train_6a_2`** — `summary.json` is 213 bytes and **config-only**: it has the seven
  `config.*` keys and none of the 24 eval-result keys. Its eval aborted after 3 of 50 episodes.
  The summary's mtime (2026-07-06 16:42 local) *precedes* its `metrics.jsonl`'s last row
  (18:32 local), consistent with the config block being written at launch and the results block
  never being written. The 3 episodes it did record are included above and all show
  `r_damage_dealt == 0.0`.
- **`m2_train_8a`** — `summary.json` is 214 bytes and config-only; `metrics.jsonl` is 2 progress
  heartbeats 30 s apart, reaching 1 training episode. **No eval episodes at all**, so it
  contributes nothing to any prediction. Not skipped: reported as contributing zero rows.

**Other anomalies:**

- **`m2_train_6a`'s summary undercounts its own wins** — 3 wins exist in eval pass 1 of 3; the
  summary reports 0 because it describes only the last pass (§4.1). Any future analysis that reads
  `summary.json` alone will miss them.
- **`m2_train_5a` has a divergent schema** — no progress heartbeat rows at all; the oldest writer
  version in the archive. Its eval rows are schema-identical to every other run's.
- **Two checkpoints have no run directory** — `m2_5a_2.pt` (`grad_step` 6) and `m2_5a_3.pt`
  (`grad_step` 1,404).
- **Only 2 of 6 checkpoints map to a run by exact `grad_step` identity**; the other four are
  matched by timestamp containment, which is suggestive but not proof.
- **`m2_train` and `m2_train_6a` each contain two training launches**, with `m2_train_6a`'s first
  launch configured for `total_episodes: 500` and its second for `10,000`. Episode counters reset
  between launches; the archive preserves no boundary marker other than that reset.
- **The 19 HP → 0 HP flip** in `r_damage_taken` between the two earliest runs and the four later
  ones is unexplained by anything in the archive (§5).
- **Three runs used an uncommitted reward config** — `m2_train_5a_4`, `m2_train_5a_5` and
  `m2_train_6a_2` log nonzero `r_shaping`, which requires `c_approach != 0`, a value that appears in
  no commit of `agent/reward_config.py` (§5.1).
- **`m2_train_5a_4` is the only run with a non-zero seed** (`20260706`; all others used `0`).
- **Provenance is broken three independent ways** (§5, §5.1): the archive carries no config files;
  `config.code_version`'s `cfg403300e3` fingerprint hashes only version pins and timing constants,
  so it is identical across two materially different reward functions; and its SHA half comes from
  `git rev-parse --short HEAD` with **no `--dirty`**, so `8b4c151` is stamped on two runs whose own
  data proves the tree was modified. Five of seven runs stamp `nogit`, leaving their code state
  unrecoverable. Worth a follow-up issue: a stamp that cannot detect either a reward change or a
  dirty tree is not a provenance stamp.

**Two interpretive caveats from live session context, flagged as such:**

- **The dummy was not stationary.** Established live this session: the dummy's `/attribute` calls
  used IDs that MC 1.21.1 rejects, and because a bad value in a `$`-macro function aborts the whole
  function, the dummy never received knockback resistance or `movement_speed 0` and drifted off
  spawn every episode. The wide `r_aim` spread across runs (per-episode totals ranging from 0.00 to
  3.99, and whole runs at 0.0) must be read against a *moving* opponent, not the stationary one
  these runs assumed. No opponent-distance or opponent-position metric in this archive should be
  taken at face value.
- **Timing figures are not agent behavior.** These runs came from the Windows machine, where Paper
  ticked ~15–19 TPS rather than 20. `progress/steps_per_s`, `eps_per_min`, and wall-clock episode
  spacing carry that artifact.

---

## 10. Bottom line

Across 453 recorded eval episodes spanning seven runs, two reward regimes, and three days of
training on the Windows machine, the agent's per-hit damage channel recorded **exactly zero, every
single time** — including in all four episodes where it killed the dummy outright. The damage
channel was dead for the entire archived history, and the archive contains no counter-evidence.

The result is not an artifact of the measurement. The two ways it could have been are both closed:
`c_dmg_out = 1.0` at every commit in the project's history, so the coefficient is not vacuous
(§8); and the six components sum to `episode_reward` on all 453 rows, so no key was ever silently
dropped and the zeros are reported values from a live logging path (§4.2).

Predictions 1, 2 and 3 are confirmed over all recorded episodes. Prediction 4 is confirmed in its
"not a death" half and untestable in its "`y ≈ −60`" half, because this era's writer logged no
positional data. The halt rule did not trigger.

**AC9 status: satisfied except for its `episode-end y` clause, which this archive cannot supply** —
no positional field exists in any row shape, so prediction 4 is reported as split (confirmed /
untestable) rather than confirmed or refuted. All other AC9 requirements are met, with the scope
narrowed throughout to recorded (eval) episodes. Testing the positional half would require
re-instrumenting the logger, which is outside T5.
