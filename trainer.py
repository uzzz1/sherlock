"""Rolling-window XGBoost trainer for the sherlock quantitative research study.

Intraday edition (2026-09-19): samples are keyed by (ukey, time) — the
prediction instants of one eif task (targets.py; task 0 盘前 = 09:24:59,
task 1 盘中 = the 48 five-minute points) — with targets from
target.{task}/, the daily (ukey, weight, group) file weight/ broadcast
to each instant (eif_v11; the per-instant intraday weight is retired),
and features through the xloader_contract (an x_loader(date) ->
polars frame of [ukey, time, *features]). I/O is polars; the numeric
core is numpy + xgboost. See .claude/skills/evaluate-factors/SKILL.md
for the dataset layout and the rolling-fit procedure.
"""

import datetime
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import polars as pl
import xgboost as xgb

# Trading calendar (YYYYMMDD per row), generated from the weight-folder
# listing.
CALENDAR_FILE = "/data/CN-EQ/calendar.csv"
DATES = pl.read_csv(CALENDAR_FILE, schema_overrides={"date": pl.String})["date"]
POS = {d: i for i, d in enumerate(DATES)}

ID_COLS = ("ukey", "time")


def group_dir(groups: list[int]) -> str:
    """Name of the model work dir (before the _{strategy} suffix) of the
    model trained over the groups: their labels concatenated, e.g.
    group0123."""
    return "group" + "".join(map(str, groups))


# Task 1 without --times trains two models per group set: `open` on the
# 09:30 instant and `rest` on the other instants, each predicting its own
# instants of the full grid. OPEN_SEC is 09:30 as seconds of the
# Asia/Shanghai day (targets.TASK_SECONDS).
OPEN_SEC = 9 * 3600 + 30 * 60
TIME_SLICES = ("open", "rest")


def time_slices(task: int, train_secs: list[int], full_secs: list[int],
                split: bool = True) -> list[tuple[str, list[int], list[int]]]:
    """The models of a task over its instants, as (name, training instants,
    prediction instants): with `split` (the CLI: task 1 without --times),
    for task 1 with 09:30 and at least one other instant among the
    training instants, `open` (09:30 only) and `rest` (the others,
    predicting every other instant of the full grid); else one unnamed
    model over all the training instants predicting the full grid."""
    rest = [s for s in train_secs if s != OPEN_SEC]
    if split and task == 1 and OPEN_SEC in train_secs and rest:
        return [("open", [OPEN_SEC], [OPEN_SEC]),
                ("rest", rest, [s for s in full_secs if s != OPEN_SEC])]
    return [("", list(train_secs), list(full_secs))]


def model_dirs(work_root: str, groups: list[int], strategy: str) -> list[str]:
    """The model work dirs (names under work_root) a study has: with gd =
    group_dir(groups), either `{gd}_{strategy}` (one model over the task's
    instants) or its time slices `{gd}_open_{strategy}` + `{gd}_rest_{strategy}`
    (time_slices). Their pred files partition the prediction universe."""
    gd = group_dir(groups)
    found = [d for d in [f"{gd}_{strategy}"] + [f"{gd}_{t}_{strategy}" for t in TIME_SLICES]
             if os.path.isdir(f"{work_root}/{d}")]
    assert found, f"no model dir {gd}[_open|_rest]_{strategy} under {work_root}"
    return found


def slice_secs(dir_name: str, strategy: str, full_secs: list[int]) -> list[int]:
    """The prediction instants of a model dir named by model_dirs."""
    tag = dir_name.removesuffix(f"_{strategy}").rsplit("_", 1)[-1]
    if tag == "open":
        return [OPEN_SEC]
    if tag == "rest":
        return [s for s in full_secs if s != OPEN_SEC]
    return list(full_secs)


def _end_month() -> str:
    """Last month before the quarter that started at least one year ago.

    Matches the upstream eif embargo cutoff (files from that quarter on are
    owner-only): e.g. run on 2026-08-13 -> embargo 20250701 -> "202506".
    """
    today = datetime.date.today()
    try:
        cutoff = today.replace(year=today.year - 1)
    except ValueError:                          # Feb 29
        cutoff = today.replace(year=today.year - 1, day=28)
    q_month = (cutoff.month - 1) // 3 * 3 + 1
    y, m = (cutoff.year, q_month - 1) if q_month > 1 else (cutoff.year - 1, 12)
    return f"{y}{m:02d}"


class Trainer:
    """Rolling-window XGBoost trainer.

    Users configure work_dir, the w/y/x loaders and the execution strategy.
    Each loader maps a date "YYYYMMDD" to a polars frame keyed by
    (ukey Int32, time Datetime[ns, UTC]): W with a single `weight` column,
    Y with a single target column, X with one column per feature. The
    rolling-fit configuration and model hyper-parameters are class variables
    maintained by the Trainer admin team. run() writes model checkpoints to
    {work_dir}/ckpt/ and per-day predictions to {work_dir}/pred/.

    Training and prediction may run on different instants: the w/x loaders
    define the TRAINING samples (e.g. a --times subset of the task grid),
    pred_w_loader / pred_x_loader the PREDICTION rows (default: the same).
    Predictions cover every finite-weight row of pred_w_loader whether or
    not it has a target, so the evaluator always sees the full grid
    (decided 2026-10-06: however many instants train the model, predict all
    of the task's instants).

    The execution strategy sets the training objective. A long_short book
    deploys the prediction as-is, so it trains with plain squarederror. A
    one-sided book deploys the clipped prediction — ReLU(pred) for
    long_only, -ReLU(-pred) for short_only — so it trains against the
    clipped value instead: (y - ReLU(pred))^2 resp. (y + ReLU(-pred))^2;
    see _clipped_obj.

    Group dummies: with `group_dummies` set to a list of group labels, W
    also carries the `group` column and the trainer appends one-hot
    features group_{g} (1.0 on the sample's group, else 0.0) after the
    x_loader's features, so one booster trains and predicts over all of
    those groups and can still split on the label; the CLI always trains
    this way (one model over its --group values). Without it W must not
    carry `group` (a Trainer over one universe with no label).

    Limit-locked samples: W may carry a `lock` column (+1 mark on 涨停,
    -1 on 跌停, 0 free, null no mark), consumed per `lock_mask`, the
    trainer's counterpart of the Evaluator's option of the same name:
    "unfillable" (default) keeps the rows but is sign-aware — a book
    cannot buy into 涨停 or sell into 跌停, so at a +1 lock only a
    non-positive prediction is deployable and at -1 only a non-negative
    one, and the loss sees that clipped prediction (the same half-line
    clip the one-sided strategies apply to every sample, composed with
    it; no-mark rows are dropped). A locked sample on its unfillable side
    then contributes no gradient — the training analogue of the
    Evaluator's "unfillable" mask — while on the fillable side (the
    reversal) it trains normally. "none" trains on every row and ignores
    the column; "locked" drops every locked and no-mark row.

    Why "unfillable" is the default (three-arm head-to-head, 2026-10-06):
    on the fillable-masked evaluation it ties "none" (the locked rows'
    value is split structure that transfers to free names, which the
    clip preserves) and beats "locked" by a third; but unlike "none" it
    does not learn to predict the limit continuation a book cannot
    trade, so the raw prediction files are honest about what is
    realizable — raw corr close to the masked figure instead of nearly
    double it — which matters to anything consuming pred/ directly.
    """

    # Execution strategies (see class docstring).
    STRATEGIES = ("long_short", "long_only", "short_only")
    # Lock-mask modes (class docstring); the first is the default.
    LOCK_MASKS = ("unfillable", "locked", "none")

    # Rolling-fit configuration, maintained by the Trainer admin team.
    SRT_MONTH = "201701"  # YYYYMM, inclusive
    END_MONTH = _end_month()  # YYYYMM, inclusive; tracks the eif embargo
    LOOK_BACK = 24  # training-window length in months
    HALF_LIFE = 12  # sample-weight decay half-life in months
    CKPT_FREQ = 3   # months predicted per fit; window shift size
    READ_THREADS = 16  # days of a month read concurrently

    # Model hyper-parameters, maintained by the Trainer admin team.
    XGB_PARAMS = {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "max_depth": 4,
        "eta": 0.05,
        "subsample": 0.8,
        # Leave most of the box to others; past ~32 threads the window
        # fits see little speedup anyway.
        "nthread": 32,
        # Hessian units are median-normalized sample weights (see _fit), so
        # this means: each leaf must hold >= ~1000 median-sized samples.
        # Reference value at REF_SAMPLES; scaled per fit to the actual
        # training sample count (see _min_child_weight).
        "min_child_weight": 1000,
    }
    NUM_BOOST_ROUND = 200
    # colsample_bytree is set per fit from the user's feature count (see
    # _colsample): wide, redundant feature sets need stronger per-tree
    # decorrelation than narrow ones.
    COLSAMPLE_REF_FEATURES = 50
    # min_child_weight was tuned on full-universe daily windows of ~2.4M
    # samples (~4800 stocks x 24 months); _min_child_weight scales it with
    # the actual window size so a leaf always holds >= ~1/2400th of the
    # data. This also absorbs the task-1 grid: 48 instants/day multiply the
    # sample count, and the floor scales with it.
    MIN_CHILD_REF_SAMPLES = 2_400_000

    @classmethod
    def _colsample(cls, n_features: int) -> float:
        """Feature fraction per tree: 1.0 at <= REF features, then a cube-root
        falloff (~0.48 at 464 features, ~0.29 at 2000)."""
        return min(1.0, (cls.COLSAMPLE_REF_FEATURES / n_features) ** (1 / 3))

    @classmethod
    def _min_child_weight(cls, n_samples: int) -> float:
        """Leaf-size floor in median-weight samples, proportional to the
        window's sample count (~61 at 145k samples, ~270 at 650k, the
        reference 1000 at 2.4M); never below 1."""
        return max(1.0, cls.XGB_PARAMS["min_child_weight"] * n_samples / cls.MIN_CHILD_REF_SAMPLES)

    def __init__(
        self,
        work_dir,  # root directory for outputs (ckpt/, pred/)
        w_loader,  # callable: date "YYYYMMDD" -> [ukey, time, weight]
        y_loader,  # callable: date "YYYYMMDD" -> [ukey, time, <target>]
        x_loader,  # callable: date "YYYYMMDD" -> [ukey, time, *features]
        strategy="long_short",  # execution strategy, one of STRATEGIES
        pred_w_loader=None,  # like w_loader, over the instants to PREDICT (default: w_loader)
        pred_x_loader=None,  # like x_loader, over the instants to PREDICT (default: x_loader)
        lock_mask=LOCK_MASKS[0],  # how W's `lock` column enters training, one of LOCK_MASKS
        group_dummies=None,  # group labels to one-hot from W's `group` column into features (class docstring)
    ):
        assert strategy in self.STRATEGIES, f"strategy must be one of {self.STRATEGIES}, got {strategy!r}"
        assert lock_mask in self.LOCK_MASKS, f"lock_mask must be one of {self.LOCK_MASKS}, got {lock_mask!r}"
        self.lock_mask = lock_mask
        self.group_dummies = list(group_dummies) if group_dummies else None
        self.work_dir = work_dir
        self.w_loader = w_loader
        self.y_loader = y_loader
        self.x_loader = x_loader
        self.pred_w_loader = pred_w_loader or w_loader
        self.pred_x_loader = pred_x_loader or x_loader
        self.strategy = strategy
        # Month -> its trading dates in [SRT_MONTH, END_MONTH], from the calendar.
        span = DATES.filter(DATES.str.slice(0, 6).is_between(self.SRT_MONTH, self.END_MONTH))
        self._date_group: dict[str, list[str]] = {
            m[0] if isinstance(m, tuple) else m: g["date"].to_list()
            for m, g in span.rename("date").to_frame()
            .group_by(pl.col("date").str.slice(0, 6), maintain_order=True)
        }
        # x_loader features (fixed by the first day read) + the group dummies.
        self._x_features: list[str] | None = None
        self.features: list[str] | None = None

    def _dummy_exprs(self) -> list[pl.Expr]:
        """The one-hot group columns from a frame's `group` column."""
        return [(pl.col("group") == g).cast(pl.Float32).alias(f"group_{g}")
                for g in self.group_dummies or []]

    # ------------------------------------------------------------- data
    def _load_daily_data(self, date: str) -> pl.DataFrame:
        """One day's joined sample frame: [date, ukey, time, weight, y,
        *features], float32 values, plus `lock` (float32, 0/+1/-1) under
        lock_mask "unfillable". The features are the x_loader's columns
        followed by the group dummies, when configured.

        Y and X are joined onto W's (ukey, time) universe; samples missing
        the weight or the target are dropped; missing/NaN features are 0.
        W's optional `lock` column is applied per lock_mask (class
        docstring): "none" ignores it, "locked" and "unfillable" drop the
        no-mark (null) rows and "locked" the +1/-1 rows as well.
        """
        w = self.w_loader(date)
        y = self.y_loader(date)
        x = self.x_loader(date)
        assert w.columns[:3] == [*ID_COLS, "weight"] and set(w.columns[3:]) <= {"lock", "group"}, \
            f"W columns {w.columns}"
        assert ("group" in w.columns) == bool(self.group_dummies), \
            f"W carries a `group` column iff group_dummies is set; W columns {w.columns}"
        if self.lock_mask == "none":
            w = w.drop("lock", strict=False)
        else:
            assert "lock" in w.columns, f"lock_mask {self.lock_mask!r} needs a `lock` column in W"
            w = w.filter(pl.col("lock") == 0 if self.lock_mask == "locked"
                         else pl.col("lock").is_not_null())
            if self.lock_mask == "locked":
                w = w.drop("lock")
        ycol = [c for c in y.columns if c not in ID_COLS]
        assert len(ycol) == 1, f"Y must have a unique value column, got {y.columns}"
        feats = [c for c in x.columns if c not in ID_COLS]
        if self.features is None:
            self._x_features = feats
            self.features = feats + [f"group_{g}" for g in self.group_dummies or []]
        assert feats == self._x_features, f"feature columns changed on {date}"
        df = (w.join(y.rename({ycol[0]: "y"}), on=list(ID_COLS), how="inner")
              .filter(pl.col("weight").is_finite() & pl.col("y").is_finite())
              .join(x, on=list(ID_COLS), how="left"))
        return df.select(
            pl.lit(date).alias("date"), *ID_COLS,
            pl.col("weight").cast(pl.Float32), pl.col("y").cast(pl.Float32),
            *([pl.col("lock").cast(pl.Float32)] if "lock" in df.columns else []),
            *[pl.col(c).fill_null(0.0).fill_nan(0.0).cast(pl.Float32)
              for c in feats],
            *self._dummy_exprs())

    def _load_window_data(self, months: list[str]) -> pl.DataFrame:
        """The window's daily frames concatenated, read on READ_THREADS threads (polars releases the GIL
        while reading). The first day of the study is read alone, since it fixes the feature order.
        Nothing is cached here: the x_loader keeps the raw daily reads shared across the lockstep trainers,
        and a second per-trainer copy of the window would double the resident data. rechunk=False keeps
        the result a view of the day frames instead of copying the window once more."""
        dates = [d for m in months for d in self._date_group[m]]
        frames = [self._load_daily_data(dates.pop(0))] if self.features is None else []
        with ThreadPoolExecutor(self.READ_THREADS) as pool:
            frames += pool.map(self._load_daily_data, dates)
        return pl.concat(frames, rechunk=False)

    # ------------------------------------------------------------ train
    def _clipped_obj(self, pred: np.ndarray, dmat: xgb.DMatrix,
                     lock: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Gradient/hessian of the clipped loss (y - clip(pred))^2 / 2,
        where clip keeps pred on its deployable half-line and is 0
        elsewhere: ReLU for long_only, -ReLU(-.) for short_only, and per
        sample under lock_mask "unfillable" (lock = the samples' 0/+1/-1
        lock), -ReLU(-.) at a +1 lock and ReLU at -1; the clips compose.

        On the clipped side the loss is flat, so grad = hess = 0 there; the
        boundary pred = 0 counts as active, which keeps the base_score 0
        start trainable. The 1/2 convention gives hess = 1 per unit weight,
        matching reg:squarederror so min_child_weight keeps its meaning.
        Custom objectives bypass XGBoost's sample weighting, so the weights
        are applied here.
        """
        y = dmat.get_label()
        w = dmat.get_weight()
        active = np.ones_like(pred, dtype=bool)
        if self.strategy != "long_short":
            active &= pred >= 0.0 if self.strategy == "long_only" else pred <= 0.0
        if lock is not None:
            active &= ((lock == 0.0) | ((lock > 0.0) & (pred <= 0.0))
                       | ((lock < 0.0) & (pred >= 0.0)))
        return np.where(active, pred - y, 0.0) * w, active * w

    def _fit(self, months: list[str]) -> xgb.Booster:
        """Train a booster on the given months with decayed sample weights;
        checkpoint it as {work_dir}/ckpt/{last train month}.json."""
        df = self._load_window_data(months)
        n_days = df["date"].n_unique()
        print(f"{self.work_dir}: {len(df)} rows over {n_days} days, "
              f"{df.select(pl.struct('date', 'time').n_unique()).item() / n_days:.1f} instants/day",
              flush=True)
        # Exponential decay: a month HALF_LIFE months before the end of the
        # window carries half the weight of the last month.
        decay = {m: 2.0 ** (i / self.HALF_LIFE) for i, m in enumerate(months)}
        weight = (df["weight"].to_numpy()
                  * df["date"].str.slice(0, 6).replace_strict(decay).to_numpy())
        # Normalize so the hessian unit is "one median-weight sample": raw
        # weights are liquidity-scaled and drift over the study, which
        # would silently retune min_child_weight across windows.
        weight = weight / np.median(weight)
        x = df.select(self.features).to_numpy()
        y = df["y"].to_numpy()
        n_rows = len(df)
        lock = df["lock"].to_numpy() if self.lock_mask == "unfillable" else None
        # x is a full copy of the features, so the window frame is dead
        # weight from here on: free it before QuantileDMatrix allocates.
        del df
        # QuantileDMatrix pre-bins the features for tree_method "hist":
        # faster to build and ~half the memory of a full DMatrix, with
        # identical training. Inference (_predict) keeps a plain DMatrix.
        dmat = xgb.QuantileDMatrix(x, label=y, weight=weight,
                                   feature_names=self.features)
        # The DMatrix holds the binned copy; free the raw matrix before the
        # training buffers are allocated.
        del x
        params = self.XGB_PARAMS | {
            "colsample_bytree": self._colsample(len(self.features)),
            "min_child_weight": self._min_child_weight(n_rows),
        }
        if self.strategy == "long_short" and lock is None:
            booster = xgb.train(params, dmat, self.NUM_BOOST_ROUND)
        else:
            # One-sided books and the sign-aware lock mask train against
            # the clipped prediction (_clipped_obj). base_score 0 starts on
            # the active boundary — the 0.5 default would put every
            # short_only sample on the flat side and stall training at the
            # first round.
            params = {k: v for k, v in params.items() if k != "objective"} | {"base_score": 0.0}
            booster = xgb.train(params, dmat, self.NUM_BOOST_ROUND,
                                obj=lambda p, d: self._clipped_obj(p, d, lock))
        booster.save_model(f"{self.work_dir}/ckpt/{months[-1]}.json")
        return booster

    # ---------------------------------------------------------- predict
    def _predict_day(self, booster: xgb.Booster, date: str,
                     x: pl.DataFrame | None = None) -> None:
        """Predict one trading day over pred_w_loader's finite-weight rows
        (no target needed) and write {work_dir}/pred/{date}.parquet =
        [ukey, timestamp, pred], keyed like target.{task}. `x` is the day's
        pred_x_loader frame when the caller already holds it (run_lockstep
        loads each date once and shares it across the groups)."""
        w = self.pred_w_loader(date)
        assert w.columns == [*ID_COLS, "weight", *(["group"] if self.group_dummies else [])], \
            f"W columns {w.columns}"
        if x is None:
            x = self.pred_x_loader(date)
        df = (w.filter(pl.col("weight").is_finite()).select(pl.exclude("weight"))
              .with_columns(self._dummy_exprs()).drop("group", strict=False)
              .join(x, on=list(ID_COLS), how="left"))
        X = df.select([pl.col(c).fill_null(0.0).fill_nan(0.0).cast(pl.Float32)
                       if c in df.columns else pl.lit(0.0, dtype=pl.Float32).alias(c)
                       for c in self.features]).to_numpy()
        pred = booster.predict(xgb.DMatrix(X, feature_names=self.features)).astype(np.float64)
        (df.select("ukey", pl.col("time").alias("timestamp"))
         .with_columns(pl.Series("pred", pred))
         .write_parquet(f"{self.work_dir}/pred/{date}.parquet"))

    # -------------------------------------------------------------- run
    @classmethod
    def _pred_months(cls, months: list[str], i: int) -> list[str]:
        """The CKPT_FREQ months window i predicts (empty for the last fit)."""
        j = i + cls.LOOK_BACK
        return months[j:j + cls.CKPT_FREQ]

    def _fit_window(self, months: list[str], i: int) -> xgb.Booster:
        """One rolling step's fit: train on months[i : i+LOOK_BACK] and
        checkpoint; the window's predictions are written by run_lockstep."""
        train_months = months[i:i + self.LOOK_BACK]
        pred_months = self._pred_months(months, i)
        pred = f"{pred_months[0]}-{pred_months[-1]}" if pred_months else "none"
        print(f"{self.work_dir}: window {train_months[0]}-{train_months[-1]} -> predict {pred}", flush=True)
        return self._fit(train_months)

    def run(self):
        """Rolling fit over the calendar months.

        Per window: train on LOOK_BACK months, checkpoint the model as
        {work_dir}/ckpt/{last train month}.json, predict every trading day
        of the next CKPT_FREQ months on the prediction instants, then shift
        the window by CKPT_FREQ months and repeat until the months are
        exhausted.
        """
        self.run_lockstep([self])

    def _load_ckpt(self, path: str) -> xgb.Booster:
        """A saved checkpoint, fixing the feature list from it when no day
        has been read yet (a resumed run may never fit)."""
        booster = xgb.Booster()
        booster.load_model(path)
        if self.features is None:
            self.features = list(booster.feature_names)
            n_dummies = len(self.group_dummies or [])
            self._x_features = self.features[:len(self.features) - n_dummies]
        assert list(booster.feature_names) == self.features, f"{path}: features differ from the run's"
        return booster

    @classmethod
    def run_lockstep(cls, trainers: list["Trainer"], resume: bool = False) -> None:
        """Run several trainers over the same calendar in lockstep: every
        trainer fits window 0, then the window's dates are predicted
        date-major (each date for every trainer), then window 1, and so on.

        resume picks a crashed run of the SAME configuration up where it
        stopped: a window whose checkpoint exists in a trainer's work dir
        is loaded instead of refit, and only the window's dates missing a
        pred file in some trainer are predicted (plus each trainer's last
        written date, which the crash may have truncated). Nothing checks
        that the configuration matches the files found.

        The point is IO sharing: with training loaders that cache the raw
        daily reads (e.g. an lru_cache sized to one window's dates), each
        file is read once for the first trainer and served from cache to
        the rest. Prediction frames are not cached that way — a full-grid
        task-1 day is large — so each date's pred_x_loader frame is read
        once here (READ_THREADS dates at a time, when every trainer shares
        the same loader) and handed to all trainers before the next chunk.
        """
        for tr in trainers:
            os.makedirs(f"{tr.work_dir}/ckpt", exist_ok=True)
            os.makedirs(f"{tr.work_dir}/pred", exist_ok=True)
        months = list(trainers[0]._date_group)
        shared_x = all(tr.pred_x_loader is trainers[0].pred_x_loader for tr in trainers)
        for i in range(0, len(months) - cls.LOOK_BACK + 1, cls.CKPT_FREQ):
            dates = [d for m in cls._pred_months(months, i) for d in trainers[0]._date_group[m]]
            if resume:
                ckpt = months[i + cls.LOOK_BACK - 1]
                have = [{f[:8] for f in os.listdir(f"{tr.work_dir}/pred") if f.endswith(".parquet")}
                        for tr in trainers]
                last = {max(h) for h in have if h}
                dates = [d for d in dates if d in last or any(d not in h for h in have)]
                paths = [f"{tr.work_dir}/ckpt/{ckpt}.json" for tr in trainers]
                if all(os.path.exists(p) for p in paths) and not dates:
                    print(f"resume: window {months[i]}-{ckpt} done, skipped", flush=True)
                    continue
                boosters = [tr._load_ckpt(p) if os.path.exists(p) else tr._fit_window(months, i)
                            for tr, p in zip(trainers, paths)]
            else:
                boosters = [tr._fit_window(months, i) for tr in trainers]
            with ThreadPoolExecutor(cls.READ_THREADS) as pool:
                for c in range(0, len(dates), cls.READ_THREADS):
                    chunk = dates[c:c + cls.READ_THREADS]
                    xs = (list(pool.map(trainers[0].pred_x_loader, chunk)) if shared_x
                          else [None] * len(chunk))
                    for date, x in zip(chunk, xs):
                        for tr, booster in zip(trainers, boosters):
                            tr._predict_day(booster, date, x)


def filter_correlated(panel: pl.DataFrame, threshold: float = 0.8) -> list[str]:
    """Greedy factor subset with pairwise |correlation| below `threshold`.

    Factors are visited in column order; one is kept iff its absolute
    Pearson correlation with every factor already kept is < threshold, so
    the earliest member of each correlated cluster survives. threshold >= 1
    disables filtering (every column is kept). NaNs are counted as 0
    (the training convention); a constant column has undefined correlation,
    which is treated as 0, so it is kept.
    """
    cols = list(panel.columns)
    if threshold >= 1.0:
        return cols
    # one float32 copy, centered in place; correlations from a single
    # F x F gram (normalized after the matmul, not on the N x F panel)
    X = np.nan_to_num(panel.to_numpy().astype("float32"), copy=False)
    X -= X.mean(axis=0)
    G = X.T @ X
    norm = np.sqrt(np.diag(G).copy())
    norm[norm == 0.0] = 1.0
    C = np.abs(G / np.outer(norm, norm))
    alive = np.ones(len(cols), dtype=bool)
    keep_idx: list[int] = []
    for j in range(len(cols)):
        if alive[j]:
            keep_idx.append(j)
            alive &= C[j] < threshold   # kill everything too close to j
    return [cols[j] for j in keep_idx]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train one rolling model over the eif groups under an "
                    "execution strategy."
    )
    parser.add_argument("--task", type=int, required=True, choices=[0, 1],
                        help="prediction task: 0 盘前 (09:24:59), 1 盘中 "
                             "(the 48 five-minute points)")
    parser.add_argument("--times", default=None,
                        help="subset of the task's prediction times to TRAIN "
                             "on, e.g. '09:30,10:00' (default: the full grid); "
                             "predictions are always written on the full grid. "
                             "Listed, ONE model trains on exactly these instants "
                             "(09:30 included if listed); without it task 1 trains "
                             "the 09:30 instant as its own model (`open`) and the "
                             "other instants as another (`rest`, see --rest-instants), "
                             "each predicting its own instants")
    parser.add_argument("--rest-instants", type=int, default=8, metavar="K",
                        help="task 1 without --times: the `rest` model trains, per "
                             "day, on K of the 47 non-09:30 instants drawn at random "
                             "(seeded by the date, so reruns and --resume draw the "
                             "same); 0 = all 47 (default %(default)s)")
    parser.add_argument("--x-loader", default="xloader_example",
                        help="feature loader spec, 'module[:attr]' or "
                             "'file.py[:attr]' (xloader_contract)")
    parser.add_argument("--x-cache-dtype", default="float16", choices=["float32", "float16"],
                        help="storage of the day cache of features: float16 (default) with a "
                             "per-factor float32 center/scale (xcodec), ~half the resident "
                             "memory, factors a float16 cannot hold stay float32; or float32 "
                             "to store them as is")
    parser.add_argument("--group", type=int, nargs="+", default=[0, 1, 2, 3],
                        choices=[0, 1, 2, 3],
                        help="group column values: one model is trained over all of "
                             "them, with one-hot dummies of the group label as "
                             "features (group_{g}, after the x_loader's factors); "
                             "work dir group{labels}_{strategy}, e.g. group0123_long_short")
    parser.add_argument("--strategy", default="long_short", choices=Trainer.STRATEGIES,
                        help="execution strategy (sets the XGB objective)")
    parser.add_argument("--study", default="baseline",
                        help="study name (names the work dirs)")
    parser.add_argument("--work-dir", required=True,
                        help="root under which {study}/group{g}_{strategy}/ "
                             "work dirs are created")
    parser.add_argument("--corr-threshold", type=float, default=0.8,
                        help="drop factors whose |pairwise correlation| with "
                             "an earlier kept factor is >= this (1 = keep all)")
    parser.add_argument("--factors", default=None,
                        help="text file listing the factors to train on, one "
                             "per line (e.g. another study's factors.txt); "
                             "replaces the correlation filter")
    parser.add_argument("--resume", action="store_true",
                        help="pick a crashed run of the SAME configuration up where it "
                             "stopped: reuse the checkpoints found in the work dirs and "
                             "predict only the days still missing a pred file (plus each "
                             "model's last written day); nothing checks the config matches")
    parser.add_argument("--end-month", default=None,
                        help="override the coverage-derived END_MONTH (YYYYMM)")
    parser.add_argument("--look-back", type=int, default=None,
                        help=f"override the training-window length in months "
                             f"(class default {Trainer.LOOK_BACK}), e.g. 2 with "
                             f"--end-month for a quick smoke run")
    parser.add_argument("--lock-mask", default=Trainer.LOCK_MASKS[0],
                        choices=Trainer.LOCK_MASKS,
                        help="limit-locked rows ({eif_dir}/limit_mask: mark on "
                             "涨停/跌停, or no mark) in the TRAINING sample: "
                             "'unfillable' (default) keeps them but clips "
                             "the loss to the fillable sign (buy into 涨停 / "
                             "sell into 跌停 earns no gradient: same masked "
                             "corr as 'none', honest raw predictions), "
                             "'none' keeps them all, 'locked' drops them "
                             "(cost a third of the fillable correlation, "
                             "2026-10-06). Predictions always cover every row")
    args = parser.parse_args()

    from functools import lru_cache

    import targets
    import xcodec
    import xloader_contract

    EIF_DIR = targets.latest_version_dir(args.task)
    SECS = targets.eval_times(args.task, args.times)
    FULL_SECS = targets.TASK_SECONDS[args.task]
    SLICES = time_slices(args.task, SECS, FULL_SECS, split=args.times is None)
    # Training instants of a day: --times fixes them; else task 1 draws
    # --rest-instants of the non-09:30 instants per day, keeping 09:30 for
    # the `open` model (0 = every instant). The day cache and the training
    # universe hold only those instants.
    DRAW_K = args.rest_instants if args.task == 1 and args.times is None else 0
    assert 0 <= DRAW_K < len(FULL_SECS), f"--rest-instants must be 0..{len(FULL_SECS) - 1}, got {DRAW_K}"

    # The study name comes from --study; the weight/target version (latest
    # eif_v{k}) does not appear in work dirs or model/report names.
    STUDY = args.study
    XNAME, X_FN = xloader_contract.resolve(args.x_loader)
    print(f"study {STUDY}: task {args.task}, {len(SECS)} timepoints, "
          f"x_loader {XNAME}, eif {EIF_DIR}, training rows: "
          f"lock mask {args.lock_mask}; models per group set: "
          + ", ".join(f"{name or 'all'} ("
                      + (f"{DRAW_K} of {len(tr)} drawn per day" if DRAW_K and name == "rest"
                         else f"{len(tr)} train")
                      + f" / {len(pr)} pred instants)"
                      for name, tr, pr in SLICES), flush=True)

    # Coverage window: the eif data starts 2021-01-04, so SRT_MONTH is
    # lifted to the first month whose every trading day has target+weight
    # files, and END_MONTH is the last (target_dates already intersects the
    # two; the trailing dates whose horizon crosses the kline end carry
    # all-NaN y and are dropped per sample, not per month). Feature coverage
    # is the loader's concern — a missing feature file fails loudly.
    _have = set(targets.target_dates(args.task, EIF_DIR))
    _full = [m[0] if isinstance(m, tuple) else m
             for m, g in DATES.rename("date").to_frame()
             .group_by(pl.col("date").str.slice(0, 6).alias("month"),
                       maintain_order=True)
             .agg(pl.col("date"))
             .iter_rows()
             if set(g) <= _have]
    Trainer.SRT_MONTH = max(Trainer.SRT_MONTH, min(_full))
    Trainer.END_MONTH = args.end_month or max(_full)
    if args.look_back:
        Trainer.LOOK_BACK = args.look_back
    print(f"months {Trainer.SRT_MONTH} .. {Trainer.END_MONTH}: full "
          f"target+weight coverage in {EIF_DIR.name}", flush=True)

    # Raw daily reads, cached so the lockstep trainers hit each file once
    # per date instead of once per group. 600 dates covers a full window +
    # pred span (~570 trading days).
    @lru_cache(maxsize=600)
    def day_secs(date: str) -> tuple[int, ...]:
        """The day's training instants: SECS, or with DRAW_K 09:30 plus
        DRAW_K of the other instants drawn with the date as seed."""
        if not DRAW_K:
            return tuple(SECS)
        rest = [s for s in FULL_SECS if s != OPEN_SEC]
        drawn = np.random.default_rng(int(date)).choice(rest, DRAW_K, replace=False)
        return (OPEN_SEC, *sorted(int(s) for s in drawn))

    @lru_cache(maxsize=600)
    def times_ns(date: str) -> tuple[int, ...]:
        return tuple(targets.times_to_utc_ns(date, list(day_secs(date))))

    @lru_cache(maxsize=600)
    def read_universe(date: str) -> pl.DataFrame:
        """(ukey, time, weight, group[, lock]) at the TRAINING instants —
        the daily weight broadcast to each instant (targets.load_weights),
        with the limit-mask `lock` column joined unless --lock-mask none
        (the Trainer applies the mode). Locked rows are <1% of the sample
        but ~40% of the pooled y_1 cross-moment (limit continuation a book
        cannot fill), yet the head-to-head on kline1m_lb2_lb5 (2026-10-06,
        same factors and instants, fillable-masked eval) went 6.2% corr
        keeping them vs 4.4% dropping them: the rows carry signal that
        transfers to free names, so they stay in; the default
        "unfillable" keeps that 6.2% while clipping the loss so the
        predictions do not chase the continuation (Trainer docstring),
        and the evaluator's --lock-mask handles fillability."""
        df = targets.load_weights(date, EIF_DIR, args.task)
        if args.times or DRAW_K:
            df = df.filter(pl.col("time").cast(pl.Int64).is_in(times_ns(date)))
        if args.lock_mask != "none":
            df = targets.join_limit_mask(df, date, EIF_DIR)
        return df

    def read_universe_full(date: str) -> pl.DataFrame:
        """(ukey, time, weight, group) at ALL of the task's instants — the
        prediction universe, whatever --times trained on and locked rows
        included (the evaluator applies its own mask)."""
        return targets.load_weights(date, EIF_DIR, args.task)

    @lru_cache(maxsize=600)
    def read_targets(date: str) -> pl.DataFrame:
        return targets.load_targets(date, args.task, EIF_DIR, ["y_1"])

    def read_x(date: str) -> pl.DataFrame:
        return xloader_contract.conform(X_FN(date), date, list(times_ns(date)))

    def read_x_full(date: str) -> pl.DataFrame:
        """The day's features at ALL of the task's instants (prediction);
        uncached — run_lockstep reads each date once and shares it."""
        return xloader_contract.conform(
            X_FN(date), date, targets.times_to_utc_ns(date, targets.TASK_SECONDS[args.task]))

    @lru_cache(maxsize=2048)
    def slice_ns(date: str, secs: tuple[int, ...]) -> list[int]:
        return targets.times_to_utc_ns(date, list(secs))

    def make_w_loader(groups: list[int], secs: list[int], full: bool = False):
        """W over the given groups' rows at the given instants (a model's
        training instants — of those the day holds, see day_secs — or its
        prediction instants of the full grid when full), `group` column
        included (the Trainer one-hots it)."""
        read = read_universe_full if full else read_universe
        secs = tuple(secs)

        def w_loader(date: str) -> pl.DataFrame:
            return read(date).filter(pl.col("group").is_in(groups)
                                     & pl.col("time").cast(pl.Int64).is_in(slice_ns(date, secs)))
        return w_loader

    def y_loader(date: str) -> pl.DataFrame:
        return read_targets(date)

    # Factor pre-filter: greedy pairwise-correlation dedup on a quarterly
    # sample of the training range (see filter_correlated). Computed once,
    # applied to every date's frame.
    span = DATES.filter(DATES.str.slice(0, 6)
                        .is_between(Trainer.SRT_MONTH, Trainer.END_MONTH))
    if args.factors:
        keep = [l.strip() for l in open(args.factors) if l.strip()]
        print(f"factors: {len(keep)} listed in {args.factors} "
              f"(correlation filter skipped)", flush=True)
    elif args.corr_threshold < 1.0:
        sample = span.to_list()[::63]
        panel = pl.concat([read_x(d) for d in sample]).drop("ukey", "time")
        keep = filter_correlated(panel, args.corr_threshold)
        print(f"factor filter (|corr| < {args.corr_threshold}, "
              f"{len(sample)} sample days): kept {len(keep)}/{panel.width}",
              flush=True)
        del panel
    else:
        keep = None
    # Record the factor set actually trained on, so another study can
    # reuse it verbatim via --factors.
    os.makedirs(f"{args.work_dir}/{STUDY}", exist_ok=True)
    with open(f"{args.work_dir}/{STUDY}/factors.txt", "w") as fh:
        cols = keep if keep is not None else [
            c for c in read_x(span[0]).columns if c not in ID_COLS]
        fh.write("\n".join(cols) + "\n")

    def _reindex(df: pl.DataFrame) -> pl.DataFrame:
        # column-reindex, not select: the factor set can drift across dates,
        # and a kept column absent on some day must yield null (-> 0
        # downstream)
        if keep is None:
            return df
        return df.select("ukey", "time",
                         *[pl.col(c) if c in df.columns
                           else pl.lit(None, dtype=pl.Float64).alias(c)
                           for c in keep])

    # The day cache holds only the factors the model trains on (after the
    # filter / --factors list), as float16 by default (see xcodec), or as
    # float32, the precision the samples are built at, with
    # --x-cache-dtype float32.
    codec = None
    if args.x_cache_dtype == "float16":
        codec = xcodec.Float16Codec.fit([_reindex(read_x(d)) for d in span.to_list()[::63]])
        print(codec.summary(), flush=True)

    @lru_cache(maxsize=600)
    def cached_x(date: str):
        df = _reindex(read_x(date))
        if codec is None:
            return df.with_columns(pl.exclude(*ID_COLS).cast(pl.Float32))
        return codec.encode(df)

    def x_loader(date: str) -> pl.DataFrame:
        return cached_x(date) if codec is None else codec.decode(*cached_x(date))

    def x_loader_full(date: str) -> pl.DataFrame:
        return _reindex(read_x_full(date))

    # One trainer over all the groups per time slice (one, or task 1's
    # `open` + `rest`).
    trainers = [
        Trainer(
            work_dir=f"{args.work_dir}/{STUDY}/{group_dir(args.group)}{'_' + sl if sl else ''}_{args.strategy}",
            w_loader=make_w_loader(args.group, train_secs),
            x_loader=x_loader,
            y_loader=y_loader,
            strategy=args.strategy,
            pred_w_loader=make_w_loader(args.group, pred_secs, full=True),
            pred_x_loader=x_loader_full,
            lock_mask=args.lock_mask,
            group_dummies=args.group,
        )
        for sl, train_secs, pred_secs in SLICES
    ]
    # Windows outer, models inner: each date's files are read once and
    # shared across the models through the caches above.
    Trainer.run_lockstep(trainers, resume=args.resume)
