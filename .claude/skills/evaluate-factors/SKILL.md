---
name: evaluate-factors
description: Evaluate a factor dataset end to end with the sherlock pipeline — trainer.py (rolling XGBoost on an intraday prediction task, one model over the eif groups), evaluator.py (stitched preds -> eval.nc), write_report.py (PDF), optionally factor_importance.py. Use when asked to evaluate factors/features, train and evaluate a dataset, regenerate the evaluation report, or analyze factor importance/concentration.
---

# Evaluate a factor dataset (trainer -> evaluator -> write_report)

The pipeline turns a feature dataset — daily files served through an
x_loader — into a model study on one intraday prediction task,
with a PDF report. Always run the three stages in this order — never an
evaluator-only factor screen unless explicitly asked.

## 0. Configure

Both `trainer.py` and `evaluator.py` read their config from `__main__`:

- `--task` (trainer AND evaluator, required): the prediction task —
  0 盘前 (one decision per day at 09:24:59) or 1 盘中 (48 five-minute
  decisions). Pass the same value to both. `--times` (trainer) trains
  ONE model on the listed instants of task 1's grid (e.g. '09:30,10:00',
  09:30 included if listed); without it task 1 fits the 09:30 instant
  and the other instants as two models, the latter on `--rest-instants`
  (default 8) instants drawn per day (section 1). Predictions
  are always written on the full grid (2026-10-06), so the evaluator
  sees every instant whatever the model trained on.
- `--group` (trainer, evaluator, factor_importance; default `0 1 2 3`):
  the eif groups the ONE model trains over; their labels name the model
  dirs (`group0123_...`), so pass the same value to every stage — the
  evaluator and factor_importance do not select among groups, they
  locate the study's dirs.
- `--x-loader` (trainer, default `xloader_example`): feature loader spec
  'module[:attr]' or 'file.py[:attr]' per xloader_contract —
  `x_loader(date) -> polars [ukey, time, *features]`. The default reads
  the daily factor sets under `/data/FACTORS` (every set except
  `REVERSAL32`; edit FEATURE_DIRS or copy the file to change), prefixes
  columns with the set name (`DAILY:feat_1`), dedups rename-duplicate
  ukeys, and joins the PREVIOUS trading day's file to day t's instants —
  every decision instant fires before day t's close, so a same-date join
  leaks the day's own session (measured: task-0 OOS corr 0.72 vs ~0.03).
- Factor selection (trainer): by default a greedy pairwise-correlation
  filter (`--corr-threshold`, default 0.8; 1 keeps everything) on every
  63rd trading day of the training range (~quarterly). `--factors FILE`
  (one name per line, set-prefixed) trains on exactly that list instead
  — the way to refit another study's factor set unchanged, and the way
  to keep two arms of an experiment on identical features. Every run
  records the set it trained on in `{work-dir}/{study}/factors.txt`.
- `--y-shift` is GONE (2026-09-19): the intraday targets measure from
  the decision instant itself, so the execution-lag question the flag
  probed no longer exists — task 0's target IS the open-to-open return
  of a 09:24:59 decision.
- `--study` (all scripts, default `baseline`): the study name — pass the
  SAME value to trainer, evaluator and factor_importance. It names every
  work dir and report. Never name anything after the weight/target
  version.
- `--work-dir` (both scripts, required — no default): root under which
  `{study}/...` work dirs live. Pass the SAME value to trainer and
  evaluator.
- Data root (`targets.py`): auto-resolved at startup to the largest
  `/data/CN-EQ/weight_target/eif_v{k}` holding `weight/` and
  `target.{task}/` — the eif_v11 layout (2026-10-04): `weight/{t}` is ONE
  daily file `(ukey, weight, group)` = universe + label + daily
  weight, broadcast by `load_weights` to the task's instants;
  `target.{task}/{t}` is keyed by ukey + timestamp; `limit_mask/{t}` is
  the per-instant limit-lock flag. eif_v10's `group/` +
  per-instant `weight.{task}/` layout no longer resolves (v10 and v11
  targets are built differently, so studies on them are not
  comparable). `eif_v11/` is owner-only (0700): run as its owner.
  Coverage from 2021. Dates from the embargo cutoff on are owner-only
  (first day of the latest quarter starting >= 1 year before the run
  date). The trainer's SRT/END months are set at startup to the
  first/last months with FULL target+weight coverage — the end can run
  past the embargo cutoff, which works when the run owns the embargoed
  files. The evaluator's date range follows the trainer's output: the
  first/last pred dates present in every evaluated group. No manual
  date bumping.
- Trading calendar: both scripts read `/data/CN-EQ/calendar.csv`
  (one YYYYMMDD row per trading day, generated from the weight-folder
  listing). There is no repo-local copy anymore — the shared EFS file is
  the single source, so it must be reachable to run anything.
- Alignment: pred/{t} and target.{task}/{t} describe the same decision
  instants of day t, and weight/{t} is the day's constant weight at each
  of them, so the evaluator applies no shift; the feature lag lives
  inside the x_loader (previous-day files, above).
- Target semantics (eif_v11, 2026-10-04): y_n is the model-facing
  transform of the raw forward return r_n at the same day and instant
  (what the models are trained to predict); r_n is kept alongside as
  the untransformed reference. Group is a universe partition and a
  one-hot feature of the model only — it plays no role in y. The eif README
  is the reference for both.

## 1. Train

```bash
python trainer.py --task T --work-dir ROOT [--x-loader SPEC] [--times ...] \
    [--study NAME] [--group 0 1 2 3] \
    [--strategy long_short|long_only|short_only] \
    [--corr-threshold 0.8 | --factors FILE] [--end-month YYYYMM] \
    [--lock-mask unfillable|locked|none] [--x-cache-dtype float16|float32] \
    [--rest-instants K] [--resume] [--look-back MONTHS]
```

`--look-back` overrides the 24-month training window, together with
`--end-month` the way to smoke-test a loader or a code change in minutes
(e.g. `--look-back 2 --end-month 202103`: one fit on 202101-202102,
predicting 202103) — never for a study to be reported.

`--resume` picks a crashed run up where it stopped (rerun the SAME command
with the flag added): a window whose checkpoint is already in the work
dir is loaded instead of refit, and only the days still missing a pred
file are predicted (plus each model's last written day, which the crash
may have truncated). It checks nothing about the configuration — a
different factor list or flag set against existing work dirs silently
mixes runs, so start a fresh study instead.

`--x-cache-dtype` sets how the trainer's day cache of features is stored
(the resident memory of a rolling fit). The cache holds only the factors
the model trains on (after the correlation filter / `--factors`).
`float16` (default) halves the memory of `float32` (which stores them as
is; pass `--x-cache-dtype float32` to get it), storing each factor as
`(x - center) / scale` in float16 plus a float32 center/scale per factor (`xcodec.py`), decoded to float32 on read, so the model still sees the
original values (level included) — plain float16 would collapse a factor
sitting far from zero. It picks, per factor and from the sample days of the
correlation filter, a study-wide center/scale, else the day's own, else keeps
float32, and prints the split at startup (`float16 x cache over N factors: ...`).
Peak memory while fitting (the window frame and matrix) is unchanged.

`--lock-mask` is the trainer's counterpart of the evaluator option of
the same name, for training rows whose mark price sits on 涨停/跌停 at
the instant (or have no mark; `{eif_dir}/limit_mask` joined in the
training weight loader via `targets.join_limit_mask`):

- `unfillable` (default): keep the rows but make the loss sign-aware —
  at a +1 lock only a non-positive prediction is deployable (buys cannot
  fill), at -1 only a non-negative one, so the loss sees the prediction
  clipped to that half-line (same custom objective as the one-sided
  strategies); no-mark rows are dropped. A locked sample on its
  unfillable side earns no gradient, on its fillable side (the reversal)
  it trains normally.
- `none`: keep every row, plain loss.
- `locked`: drop every locked and no-mark row.

Why `unfillable` is the default (three-arm head-to-head on
kline1m_lb2_lb5, same factors and instants, 2026-10-06): locked rows are
<1% of rows but ~40% of the pooled y_1 cross-moment. On the
fillable-masked eval `unfillable` and `none` tie (6.17% corr, spnl 13.4
vs 13.3 bp) and `locked` loses a third (4.40%) — the rows carry split
structure that transfers to free names, which clipping keeps and
dropping destroys. But `none` also learns to predict the limit
continuation a book cannot trade, so its raw predictions overstate what
is realizable (unmasked corr 11.3% vs 6.2% masked); `unfillable` stops
that (raw 7.6%), so pred/ files are honest for anything that consumes
them directly. Same realizable signal, less misleading output.

Predictions cover every row whatever the mode; the evaluator's
`--lock-mask` handles fillability there.

Trains ONE rolling-window XGBoost model over the `--group` values
(default all four): the sample is the union of the groups' rows and the
group label enters as one-hot features `group_{g}` (1.0 on the sample's
group, 0.0 elsewhere) appended after the x_loader's factors, so the
trees can still split on it (one model per group was dropped
2026-10-15). Rolling config (class
vars): months SRT_MONTH .. END_MONTH (coverage-derived, see above),
LOOK_BACK 24, CKPT_FREQ 3, HALF_LIFE 12 (sample-weight decay in months).
One-sided strategies train against the clipped prediction (see
`_clipped_obj`). Output: `{work-dir}/{study}/group0123_{strategy}/`
(the groups' labels concatenated) with ckpt/ and per-day
`(ukey, timestamp, pred)` pred/ files over all of the groups' rows,
starting LOOK_BACK months after SRT_MONTH through the end month;
`factors.txt` lists the x_loader factors only (the dummies are implied).
Pred files cover the task's FULL grid over the finite-weight universe
(rows without a target included; the evaluator counts their NaN y as
zero), even when `--times` restricted training.

Task 1 without `--times` trains two models (2026-10-15): `open` on the
09:30 instant and `rest` on the other 47, each predicting its own
instants of the full grid — work dirs `group0123_open_{strategy}/`
(pred/ = the 09:30 rows) and `group0123_rest_{strategy}/` (the other 47
instants), fitted in lockstep (predictions are read date-major after
each window's fits so a day's full-grid features are loaded once for
both). The `rest` model does not see every instant of every day
(2026-10-16): each training day contributes 09:30 plus `--rest-instants`
(default 8) of the other 47 instants, drawn at random per day with the
date as seed (a rerun or `--resume` draws the same instants), so the
window and the day cache hold 9/48 of the grid; `--rest-instants 0`
trains on all 47 (a full-grid window is 48x the daily sample count —
mind the memory). With `--times` a single model, in the dir without the
suffix, trains on exactly the listed instants (09:30 included if listed)
and predicts the whole grid, as for task 0; `--rest-instants` is then
ignored. The evaluator, factor_importance.py and predict.py find
whichever model dirs the study has (`trainer.model_dirs`). The sample weights are
median-normalized over the model's window and `min_child_weight` scales
with its row count, so a leaf holds the same FRACTION of the data
whatever the window holds.

## 2. Evaluate

```bash
python evaluator.py --task T --work-dir ROOT [--study NAME] \
    [--group 0 1 2 3] [--strategy ...] [--side all pos neg] \
    [--lock-mask unfillable|locked|none]
```

Stitches the study's model dirs' pred files
(`group0123_{open,rest}_{strategy}/pred/`, found by `trainer.model_dirs`
from `--group` and `--strategy`; they partition the universe)
into one fitter "pred", computes daily weighted moments sliced by side x group,
attaches the pooled metrics, and writes
`{work-dir}/{study}/{strategy}/eval.nc` over exactly the dates every
model dir has pred files for. `--lock-mask` (default `unfillable`)
removes limit-locked slots from every moment before the side/group
masks: `unfillable` drops a slot where the prediction would buy into
涨停 or sell into 跌停, plus no-mark slots; `locked` drops every locked
or no-mark slot whatever the sign; `none` keeps all. The mode is in the
file's attrs (the report prints it in its title) and the non-default
modes write `eval.{mode}.nc` beside `eval.nc`, so `none` is the
"paper" number and the default the fillable one — report both when
they differ materially.
Keep the default full side/group sets when a report will follow — the
report expects the 15 (side = all/pos/neg, group = all/0/1/2/3) buckets.

eval.nc also carries `turnover` (date, fitter, book, tod): how much a
book built on the prediction trades. Each decision slot (clock time) is
its own portfolio, rebalanced once a day at the same slot — never slot
to slot within a day. Two gross-1 books: `linear` (weight x weighted
z-score of the prediction) and `decile` (top decile long / bottom decile
short, weight-sized legs of 0.5). turnover = 0.5 * sum |dp| over the
union of names, so 0 = untouched, 1 = fully replaced; universe
entries/exits count as trades. Only slots the lock mask keeps hold a
position, so the books match the moments. Read it with the spnl: a
strategy's cost per day is roughly turnover x round-trip cost, and the
spnl is per unit of prediction, so a high-turnover slot needs a
proportionally higher spnl to clear costs.

## 3. Report

```bash
python write_report.py --eval {work-dir}/{study}/{strategy}/eval.nc \
    --pdf report_{study}_{strategy}.pdf \
    [--fitter pred] [--title "..."] [--trade raw]
```

`--title` defaults to the eval path + fitter, with the eval's lock-mask
mode appended; `--trade` selects a trade coordinate only when the eval
has one (the standard evaluator writes none).

Pages: per response (y_1/y_3/y_5), a page of seven metric tables and a
cumulative-spnl curve page; then one spnl-vs-horizon plot; then a
turnover page (per slot, median and IQR over days of the daily turnover
of the linear and decile books; the all-slot median is in the legend
and printed to stdout). Layout and units are described here and in
write_report.py's docstring — keep the two in sync. Convention: `report_{study}_{strategy}.pdf` in the repo root.
Requires matplotlib.

Sanity checks after generating: page-1 tables populated for all 15
buckets; curve/horizon legends show all 15 (side, group) labels; spnl and
curve values in basis points (single digits to tens, not 1e-4-scale
decimals); turnover bars between 0 and 1 for every slot of the task's
grid.

## 4. Factor importance (optional, after training)

```bash
python factor_importance.py --work-dir ROOT [--study NAME] \
    [--group 0 1 2 3] [--strategy ...] [--top 10] [--stack 8] [--pdf ...]
```

Reads the model's `ckpt/*.json` boosters (no retraining), normalizes each
booster's total gain into per-factor importance shares, saves the
(month, group, factor) panel to
`{work-dir}/{study}/{strategy}/importance.nc`, and writes
`report_importance_{study}_{strategy}.pdf`: per model a stacked area of the
top `--stack` factors, effective-number-of-factors (1/HHI) and top-K-share
series, and turnover (consecutive Spearman rho + top-K Jaccard); last page
is a summary table with metric definitions. One page per model dir
(`trainer.model_dirs` from `--group`/`--strategy`, as the evaluator),
labelled by its group labels and instant slice (e.g. `0123_open`,
`0123_rest`; the `group_{g}` dummies are among the factors, and the
`group` coordinate of importance.nc holds those labels). Key readings:
rotation = pooled/within N_eff (~1 = same factors dominate every window,
>1 = importance rotates); judge rank persistence by the non-overlap-lag rho
column, not the consecutive one (training windows overlap). Gain splits
credit arbitrarily among correlated factors — read individual rankings
loosely, the concentration metrics are robust to it. Pass the same
`--study`/`--work-dir` as the trainer run.
