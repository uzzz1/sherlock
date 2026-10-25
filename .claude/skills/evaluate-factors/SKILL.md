---
name: evaluate-factors
description: Evaluate a daily factor dataset end to end with the sherlock pipeline — trainer.py (per-group rolling XGBoost models), evaluator.py (stitched preds -> eval.nc), write_report.py (PDF), optionally factor_importance.py (importance/concentration report from the checkpoints). Use when asked to evaluate factors/features, train and evaluate a dataset, regenerate the evaluation report, or analyze factor importance/concentration.
---

# Evaluate a factor dataset (trainer -> evaluator -> write_report)

The pipeline turns a directory of daily feature files into a per-group
model study with a PDF report. Always run the three stages in this order —
never an evaluator-only factor screen unless explicitly asked.

## 0. Configure

Both `trainer.py` and `evaluator.py` read their config from `__main__`:

- `--feature-dirs` (trainer only): factor-set directory names under
  `/data/efs/FACTORS/ASTOCK`, each holding one `{YYYYMMDD}.parquet` per
  trading day with an id column (`ukey`, `security_id` or `ticker` —
  detected per file, in that priority) plus one column per factor.
  Defaults to every set except `REVERSAL32`. Columns are prefixed with
  the set name (`DAILY:feat_1`) and rename-duplicate ukeys are deduped.
- Factor selection (trainer): by default a greedy pairwise-correlation
  filter (`--corr-threshold`, default 0.8; 1 keeps everything) on a
  quarterly sample of the training range. `--factors FILE` (one name per
  line, set-prefixed) trains on exactly that list instead — the way to
  refit another study's factor set unchanged. Every run records the set
  it trained on in `{work-dir}/{study}/factors.txt` (baseline2 -> 386
  factors, reused verbatim by baseline3).
- `--y-shift K` (trainer AND evaluator, default 0): take the target from
  the eif file K trading days EARLIER while weight/group stay those of
  file t. With K = 1, day t's `y_1` is the return close(t) -> close(t+1)
  — i.e. the return earned by trading at the very close the feature file
  was computed from, an execution-lag-free upper bound on the signal
  (baseline3). Pass the same K to both scripts. The day whose shifted
  file does not exist (the first calendar day) yields no samples.
- `--study` (all scripts, default `baseline`): the study name — pass the
  SAME value to trainer, evaluator and factor_importance. It names every
  work dir and report. Never name anything after the weight/target
  version.
- `--work-dir` (both scripts, required — no default): root under which
  `{study}/...` work dirs live. Pass the SAME value to trainer and
  evaluator.
- `DATA_DIR`: the weight/target data, auto-resolved at startup to the
  largest `/data/efs/CN-EQ/weight_target/eif_v{k}`. Files key by `ukey`
  and supply the `weight`, the responses `y_1`/`y_3`/`y_5`, and the
  `group` label 0..3 (pred files are `(ukey, pred)` too). Dating (eif_v2+):
  the file dated t holds the weights OF date t, traded on t+1, with targets
  covering t+1 -> t+1+i (see the dataset README).
  Dates from the embargo cutoff on are owner-only (first day of the
  latest quarter starting >= 1 year before the run date; 20250701 as of
  2026-08). The trainer's `END_MONTH` is set at startup to the last month
  with FULL coverage (every trading day present) across all feature sets
  and the weight/target dir — this can run past the embargo cutoff, which
  works when the run owns the embargoed files. The evaluator's `END_DATE`
  follows the trainer's output: the last pred date present in every
  evaluated group. No manual date bumping.
- Trading calendar: both scripts read `/data/efs/CN-EQ/calendar.csv`
  (one YYYYMMDD row per trading day, generated from the weight-folder
  listing). There is no repo-local copy anymore — the shared EFS file is
  the single source, so it must be reachable to run anything.
- The trainer's `x_loader` joins features on the SAME date and drops
  non-feature columns such as `timestamp`: under the eif_v2 dating, day
  t's feature file is generated from data up to day t's market close,
  matching the eif file whose windows also end at t — pred/{t} is then
  the signal for trade day t+1, and the evaluator applies no extra shift.

## 1. Train

```bash
python trainer.py --work-dir ROOT [--feature-dirs SET ...] [--study NAME] \
    [--group 0 1 2 3] [--strategy long_short|long_only|short_only] \
    [--corr-threshold 0.8 | --factors FILE] [--y-shift K]
```

Trains one rolling-window XGBoost model per group (default all four) in
lockstep, sharing one cached read per file. Rolling config (class vars):
months SRT_MONTH (201701) .. END_MONTH (dynamic, see above; 202506 as of
2026-08), LOOK_BACK 24, CKPT_FREQ 3. One-sided strategies train against the
clipped prediction (see `_one_sided_obj`). Outputs per group:
`{work-dir}/{study}/group{g}_{strategy}/{ckpt,pred}/` with daily preds
starting LOOK_BACK months after SRT_MONTH (201901) through the end month.

## 2. Evaluate

```bash
python evaluator.py --work-dir ROOT [--study NAME] \
    [--group 0 1 2 3] [--strategy ...] [--side all pos neg] [--y-shift K]
```

Stitches the groups' pred files (they partition the universe) into one
fitter "pred", computes daily weighted moments sliced by side x group,
attaches the pooled metrics, and writes
`{work-dir}/{study}/{strategy}/eval.nc` from the first pred month
(derived: Trainer.SRT_MONTH + LOOK_BACK -> 20190101) through the dynamic
`END_DATE` (last day of the trainer's end month).
Keep the default full side/group sets when a report will follow — the
report expects the 15 (side = all/pos/neg, group = all/0/1/2/3) buckets.

## 3. Report

```bash
python write_report.py --eval {work-dir}/{study}/{strategy}/eval.nc \
    --pdf report_{study}_{strategy}.pdf \
    [--fitter pred] [--title "..."]
```

Pages: per response (y_1/y_3/y_5), a page of seven metric tables and a
cumulative-spnl curve page; then one spnl-vs-horizon plot. Layout and units
follow the "Report guidance" section of evaluator.md — keep the two in
sync. Convention: `report_{study}_{strategy}.pdf` in the repo root.
Requires matplotlib.

Sanity checks after generating: page-1 tables populated for all 15
buckets; curve/horizon legends show all 15 (side, group) labels; spnl and
curve values in basis points (single digits to tens, not 1e-4-scale
decimals).

## 4. Factor importance (optional, after training)

```bash
python factor_importance.py --work-dir ROOT [--study NAME] \
    [--group 0 1 2 3] [--strategy ...] [--top 10] [--stack 8] [--pdf ...]
```

Reads the groups' `ckpt/*.json` boosters (no retraining), normalizes each
booster's total gain into per-factor importance shares, saves the
(month, group, factor) panel to
`{work-dir}/{study}/{strategy}/importance.nc`, and writes
`report_importance_{study}_{strategy}.pdf`: per group a stacked area of the
top `--stack` factors, effective-number-of-factors (1/HHI) and top-K-share
series, and turnover (consecutive Spearman rho + top-K Jaccard); last page
is a cross-group summary table with metric definitions. Key readings:
rotation = pooled/within N_eff (~1 = same factors dominate every window,
>1 = importance rotates); judge rank persistence by the non-overlap-lag rho
column, not the consecutive one (training windows overlap). Gain splits
credit arbitrarily among correlated factors — read individual rankings
loosely, the concentration metrics are robust to it. Pass the same
`--study`/`--work-dir` as the trainer run.
