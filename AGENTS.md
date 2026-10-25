# Agent notes

Notes for coding agents and new contributors working in this repo.
This file is the single source of truth.

## What this is

Rolling-window XGBoost trainer + evaluator for intraday equity alpha
studies on the shared eif datasets
(`/data/CN-EQ/weight_target/eif_v{k}`, see that directory's README).
Pure-Python scripts, no package/build step, no test suite. I/O is polars;
numeric core is numpy + xgboost; results are xarray `.nc` files and
matplotlib PDFs.

## Before any factor-evaluation or training task

Read `.claude/skills/evaluate-factors/SKILL.md` — the end-to-end pipeline
guide (auto-discovered as a skill by Claude Code; plain Markdown for
everyone else). It is the source of truth for CLI
flags, work-dir layout, date/embargo handling and report sanity checks;
do not duplicate it here.

**The one hard rule:** run `trainer.py` first, then `evaluator.py`
(then `write_report.py`). Never an evaluator-only factor screen unless
explicitly asked.

## Pipeline at a glance

```bash
python trainer.py        --task T --work-dir ROOT --study NAME [--x-loader SPEC] ...
python evaluator.py      --task T --work-dir ROOT --study NAME
python write_report.py   --eval ROOT/NAME/long_short/eval.nc --pdf report_NAME_long_short.pdf
python factor_importance.py --work-dir ROOT --study NAME          # optional, after training
```

Pass the SAME `--task`, `--work-dir` and `--study` to every stage.
`--work-dir` has no default and is required.

## Files

| File | Role |
|---|---|
| `trainer.py` | `Trainer`: rolling XGBoost (SRT/END months coverage-derived, `LOOK_BACK` 24, `CKPT_FREQ` 3), one model over all groups with one-hot `group_{g}` features, and for task 1 one model on the 09:30 instant (`open`) plus one on the other instants (`rest`), each predicting its own instants: `{work-dir}/{study}/group0123_{open,rest}_{strategy}/{ckpt,pred}/`. Also exports `DATES`/`ID_COLS`/`model_dirs` used by the other modules. |
| `evaluator.py` | `Evaluator`: stitches the study's model dirs' preds (`trainer.model_dirs`), daily weighted moments sliced side x group, writes `{work-dir}/{study}/{strategy}/eval.nc` (15 buckets: side all/pos/neg x group all/0/1/2/3). `--lock-mask unfillable|locked|none` (default unfillable) drops limit-locked slots from the moments; non-default modes write `eval.{mode}.nc`. Also `turnover` (date, fitter, book, tod): daily turnover of the gross-1 linear/decile books per decision slot, one portfolio per slot rebalanced once a day. |
| `write_report.py` | PDF from `eval.nc`. Convention: `report_{study}_{strategy}.pdf` in repo root (PDFs are untracked). |
| `factor_importance.py` | Gain-based importance/concentration from each model dir's `ckpt/*.json` (one page per model, labelled e.g. `0123_open`); writes `importance.nc` + `report_importance_{study}_{strategy}.pdf`. |
| `targets.py` | eif data access: task grids (`TASK_SECONDS`), calendar, latest `eif_v{k}` resolution (requires `weight/` + `target.{task}/` + `limit_mask/`, the eif_v11 layout), target loader, the daily `(ukey, weight, group)` loader broadcast to the task's instants, and the per-instant limit-lock mask (`{eif_dir}/limit_mask`, `join_limit_mask`). |
| `xloader_contract.py` | The `x_loader(date) -> pl.DataFrame[ukey, time, *features]` contract; `resolve()` CLI specs, `conform()` validates a day's frame. |
| `xcodec.py` | `Float16Codec`: float16 storage of the trainer's day cache (default; `--x-cache-dtype float32` opts out), `(x - center) / scale` with float32 center/scale per factor, decoded on read. |
| `xloader_example.py` | Reference loader over `/data/FACTORS/*` daily sets (all but `REVERSAL32`); the `--x-loader` default. |

## Conventions and gotchas

- **Tasks:** `--task 0` 盘前 = one decision/day at 09:24:59; `--task 1`
  盘中 = 48 five-minute decisions. A full-grid task-1 window is 48x the
  daily sample count, so by default task 1 trains on 09:30 plus
  `--rest-instants` (default 8) of the other instants drawn per day;
  `--times` fixes the TRAINING instants instead (one model on exactly
  those). Predictions are always written on the full task grid
  (every finite-weight row, target or not), whatever `--times` trained
  on (2026-10-06), so eval.nc is comparable across studies and the
  evaluator's full-grid W never dilutes a subset run.
- **One model over the groups (2026-10-15):** the `--group` labels are
  one-hot `group_{g}` features of a single booster, never separate
  models, and they name the work dir (`group0123_...`), so `--group`
  must match across trainer, evaluator and factor_importance. Task 1
  without `--times` additionally fits the 09:30 instant (`open`) and the
  other instants (`rest`, on a per-day random draw of them) as two
  models; `trainer.model_dirs` is the one place that
  knows which dirs a study has — consumers go through it, never glob.
- **Crashes and smoke runs:** `--resume` reruns the same trainer command
  reusing the checkpoints and pred files already written (no config
  check); `--look-back N` with `--end-month` is the minutes-long smoke
  test of a loader or code change (SKILL.md, section 1).
- **Limit-locked rows (2026-10-06):** `{eif_dir}/limit_mask/{t}` (the eif
  root beside `weight/`, schema in its README) flags,
  per instant, a mark price on 涨停 (+1, buys cannot fill) or 跌停 (-1,
  sells cannot fill); null = no mark. They are <1% of rows but ~40% of the
  pooled y_1 cross-moment (limit continuation a book cannot fill). The
  trainer's `--lock-mask` defaults to `unfillable`: the rows stay in the
  sample with a sign-aware loss (the prediction is clipped to the
  fillable half-line at locked samples, via the same custom objective as
  the one-sided strategies). Why: in the three-arm head-to-head it ties
  `none` (keep, plain loss) on the masked corr (6.17%) while `locked`
  (drop) loses a third (4.40%) — the rows' split structure transfers to
  free names — but unlike `none` it does not learn the unfillable
  continuation, so the raw predictions stay honest (unmasked corr 7.6%
  vs 11.3%), which matters to anything reading pred/ directly.
  Predictions cover every row; the evaluator masks the unfillable slots
  from the moments by its own `--lock-mask` (default `unfillable`:
  sign-aware, plus no-mark slots). Never fold the mask into the weight file — the weight
  is the point-in-time sizing weight shared with downstream consumers,
  and the mask is one-sided and realized day-t data.
- **Feature lag lives in the x_loader, not the evaluator.** Day t's
  instants must see the PREVIOUS trading day's daily file. A same-date
  join leaks the session (measured task-0 OOS corr 0.72 vs ~0.03). The
  evaluator applies no shift; pred/target/weight for day t describe the
  same instants.
- **Keys:** `ukey` Int32, `time` tz-aware Datetime (normalized to ns UTC
  by `conform`). Feature columns are set-prefixed (`DAILY:feat_1`).
  Contract violations should fail the day loudly — don't soften them.
- **Study naming:** name studies after the feature set / experiment,
  never after the weight/target version. Every run records its factor
  list in `{work-dir}/{study}/factors.txt`; reuse it via `--factors FILE`.
- **Dates are derived, not hand-set.** Data root, SRT/END months and the
  evaluator's range are resolved at startup from coverage and the embargo
  cutoff. Don't bump dates manually. `--y-shift` no longer exists.
- **Shared EFS is required:** `/data/CN-EQ/calendar.csv` and
  `/data/CN-EQ/weight_target/eif_v*`. Nothing runs without them; there
  is no repo-local calendar copy. The default x_loader additionally
  needs `/data/FACTORS`; any other loader brings its own data.
- **eif_v11 layout (2026-10-04):** one daily `weight/{t}` =
  `(ukey, weight, group)`; the per-instant intraday weight (`weight.{task}/`)
  and the separate `group/` are retired, and `targets.latest_version_dir`
  no longer resolves roots with that layout. The sample weight is the
  **daily** weight at every instant.
- **Report checks:** all 15 buckets populated, legends show 15 labels,
  spnl/curve values in basis points (single digits to tens).

## Working style

- Prefer small, focused commits with imperative subjects (see `git log`).
- Don't commit generated artifacts (`*.pdf`, `*.nc`, work dirs).
- When changing a CLI flag, output path or report layout, update
  `SKILL.md` (and `AGENTS.md`/this file if the summary changes) in the
  same commit.
