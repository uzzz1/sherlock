"""Rolling-window XGBoost trainer for the sherlock quantitative research study.

See trainer.md for the dataset layout and the rolling-fit procedure.
"""

import datetime
import os

import numpy as np
import pandas as pd
import xgboost as xgb

# Trading calendar (YYYYMMDD per row), generated from the weight-folder
# listing.
CALENDAR_FILE = "/data/efs/CN-EQ/calendar.csv"
DATES = pd.read_csv(CALENDAR_FILE, dtype=str)["date"]
POS = {d: i for i, d in enumerate(DATES)}


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

    Users configure work_dir, the w/y/x loaders and the execution strategy;
    W and Y must each yield a single value column besides ukey. The
    rolling-fit configuration and model hyper-parameters are class variables
    maintained by the Trainer admin team. run() writes model checkpoints to
    {work_dir}/ckpt/ and monthly predictions to {work_dir}/pred/.

    The execution strategy sets the training objective. A long_short book
    deploys the prediction as-is, so it trains with plain squarederror. A
    one-sided book deploys the clipped prediction — ReLU(pred) for
    long_only, -ReLU(-pred) for short_only — so it trains against the
    clipped value instead: (y - ReLU(pred))^2 resp. (y + ReLU(-pred))^2;
    see _one_sided_obj.
    """

    # Execution strategies (see class docstring).
    STRATEGIES = ("long_short", "long_only", "short_only")

    # Rolling-fit configuration, maintained by the Trainer admin team.
    SRT_MONTH = "201701"  # YYYYMM, inclusive
    END_MONTH = _end_month()  # YYYYMM, inclusive; tracks the eif embargo
    LOOK_BACK = 24  # training-window length in months
    HALF_LIFE = 12  # sample-weight decay half-life in months
    CKPT_FREQ = 3   # months predicted per fit; window shift size

    # Model hyper-parameters, maintained by the Trainer admin team.
    XGB_PARAMS = {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "max_depth": 4,
        "eta": 0.05,
        "subsample": 0.8,
        # Leave most of the box to others; past ~32 threads the small
        # per-group datasets see little speedup anyway.
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
    # min_child_weight was tuned on full-universe windows of ~2.4M samples
    # (~4800 stocks x 24 months); _min_child_weight scales it with the
    # actual window size so a leaf always holds >= ~1/2400th of the data.
    # Small universes — eif groups run ~300 (group 0) to ~1400 (group 3)
    # stocks, i.e. ~145k-650k samples per window — would otherwise be
    # over-regularized into near-constant trees.
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
        w_loader,  # callable: date "YYYYMMDD" -> W frame indexed by ukey, columns = [weight]
        y_loader,  # callable: date "YYYYMMDD" -> Y frame indexed by ukey, columns = [target]
        x_loader,  # callable: date "YYYYMMDD" -> X frame indexed by ukey, columns = [features]
        strategy="long_short",  # execution strategy, one of STRATEGIES
    ):
        assert strategy in self.STRATEGIES, f"strategy must be one of {self.STRATEGIES}, got {strategy!r}"
        self.work_dir = work_dir
        self.w_loader = w_loader
        self.y_loader = y_loader
        self.x_loader = x_loader
        self.strategy = strategy
        # Month -> its trading dates in [SRT_MONTH, END_MONTH], from the calendar.
        dates = DATES[DATES.str[:6].between(self.SRT_MONTH, self.END_MONTH)]
        self._date_group: dict[str, list[str]] = (
            dates.groupby(dates.str[:6]).agg(list).to_dict()
        )
        # Monthly frames cached across overlapping windows.
        self._cache = {}

    # ------------------------------------------------------------- data
    def _load_daily_data(self, date: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load one day's (w, y, x) frames as float32, indexed by ukey.

        y and x are reindexed onto W's ukey universe.
        Only samples with non-NaN weight and target are kept.
        """
        w: pd.DataFrame = self.w_loader(date)
        y: pd.DataFrame = self.y_loader(date).reindex(index = w.index)
        x: pd.DataFrame = self.x_loader(date).reindex(index = w.index).fillna(0.0)
        assert len(w.columns) == 1, f"W must have a unique value column, got {list(w.columns)}"
        assert len(y.columns) == 1, f"Y must have a unique value column, got {list(y.columns)}"
        keep = w.iloc[:,0].notna() & y.iloc[:,0].notna()
        return (
            w[keep].astype(np.float32),
            y[keep].astype(np.float32),
            x[keep].astype(np.float32),
        )

    def _load_monthly_data(self, month: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Concat the month's daily (w, y, x) frames, indexed by (date, ukey)."""
        if month in self._cache:
            return self._cache[month]
        dates = self._date_group[month]
        data = [self._load_daily_data(date) for date in dates]
        w, y, x = (
            pd.concat(wyx, keys=pd.to_datetime(dates), names=["date"])
            for wyx in zip(*data)
        )
        self._cache[month] = (w, y, x)
        return w, y, x

    # ------------------------------------------------------------ train
    def _one_sided_obj(self, pred: np.ndarray, dmat: xgb.DMatrix) -> tuple[np.ndarray, np.ndarray]:
        """Gradient/hessian of the one-sided loss (y - clip(pred))^2 / 2,
        clip = ReLU for long_only and -ReLU(-.) for short_only.

        On the clipped side the loss is flat, so grad = hess = 0 there; the
        boundary pred = 0 counts as active, which keeps the base_score 0
        start trainable. The 1/2 convention gives hess = 1 per unit weight,
        matching reg:squarederror so min_child_weight keeps its meaning.
        Custom objectives bypass XGBoost's sample weighting, so the weights
        are applied here.
        """
        y = dmat.get_label()
        w = dmat.get_weight()
        active = pred >= 0.0 if self.strategy == "long_only" else pred <= 0.0
        return np.where(active, pred - y, 0.0) * w, active * w

    def _fit(self, months: list[str]) -> xgb.Booster:
        """Train a booster on the given months with decayed sample weights;
        checkpoint it as {work_dir}/ckpt/{last train month}.json."""
        data = [self._load_monthly_data(month) for month in months]
        w, y, x = (pd.concat(wyx) for wyx in zip(*data))
        # Exponential decay: a month HALF_LIFE months before the end of the
        # window carries half the weight of the last month.
        decay = {m: 2.0 ** (i / self.HALF_LIFE) for i, m in enumerate(months)}
        month = w.index.get_level_values("date").strftime("%Y%m")
        weight = w.iloc[:, 0] * np.asarray(month.map(decay))
        # Normalize so the hessian unit is "one median-weight sample": raw
        # weights are liquidity-scaled (~1e6, drifting ~2x over the study),
        # which would silently retune min_child_weight across windows.
        weight = weight / weight.median()
        # QuantileDMatrix pre-bins the features for tree_method "hist":
        # faster to build and ~half the memory of a full DMatrix, with
        # identical training. Inference (_predict) keeps a plain DMatrix.
        dmat = xgb.QuantileDMatrix(x, label=y.iloc[:, 0], weight=weight)
        params = self.XGB_PARAMS | {
            "colsample_bytree": self._colsample(x.shape[1]),
            "min_child_weight": self._min_child_weight(len(x)),
        }
        if self.strategy == "long_short":
            booster = xgb.train(params, dmat, self.NUM_BOOST_ROUND)
        else:
            # One-sided books train against the clipped prediction
            # (_one_sided_obj). base_score 0 starts on the active boundary —
            # the 0.5 default would put every short_only sample on the flat
            # side and stall training at the first round.
            params = {k: v for k, v in params.items() if k != "objective"} | {"base_score": 0.0}
            booster = xgb.train(params, dmat, self.NUM_BOOST_ROUND, obj=self._one_sided_obj)
        booster.save_model(f"{self.work_dir}/ckpt/{months[-1]}.json")
        return booster

    # ---------------------------------------------------------- predict
    def _predict(self, booster: xgb.Booster, month: str) -> pd.DataFrame:
        """Predict one month; save one (ukey, pred) file per trading
        day under {work_dir}/pred/, in the same format as the weight files."""
        _, _, x = self._load_monthly_data(month)
        pred = x.index.to_frame(index=False)
        pred["pred"] = booster.predict(xgb.DMatrix(x)).astype(np.float64)
        for date, day in pred.groupby("date"):
            day.assign(ukey=day["ukey"].astype("int32"))[["ukey", "pred"]].to_parquet(
                f"{self.work_dir}/pred/{date:%Y%m%d}.parquet", index=False
            )
        return pred

    # -------------------------------------------------------------- run
    def _run_window(self, months: list[str], i: int) -> None:
        """One rolling step: train on months[i : i+LOOK_BACK], checkpoint,
        predict the next CKPT_FREQ months, then evict the months the window
        rolls past."""
        j = i + self.LOOK_BACK
        k = j + self.CKPT_FREQ
        train_months = months[i:j]
        pred_months = months[j:k]
        pred = f"{pred_months[0]}-{pred_months[-1]}" if pred_months else "none"
        print(f"{self.work_dir}: window {train_months[0]}-{train_months[-1]} -> predict {pred}", flush=True)
        booster = self._fit(train_months)
        for month in pred_months:
            self._predict(booster, month)
        for month in months[i:i+self.CKPT_FREQ]:
            self._cache.pop(month, None)

    def run(self):
        """Rolling fit over the calendar months.

        Per window: train on LOOK_BACK months, checkpoint the model as
        {work_dir}/ckpt/{last train month}.json, predict the next CKPT_FREQ
        months, then shift the window by CKPT_FREQ months and repeat until
        the months are exhausted.
        """
        os.makedirs(f"{self.work_dir}/ckpt", exist_ok=True)
        os.makedirs(f"{self.work_dir}/pred", exist_ok=True)
        months = list(self._date_group)
        for i in range(0, len(months) - self.LOOK_BACK + 1, self.CKPT_FREQ):
            self._run_window(months, i)

    @classmethod
    def run_lockstep(cls, trainers: list["Trainer"]) -> None:
        """Run several trainers over the same calendar in lockstep: every
        trainer fits window 0, then every trainer fits window 1, and so on.

        The point is IO sharing: with loaders that cache the raw daily
        reads (e.g. an lru_cache sized to one window's dates), each file is
        read once for the first trainer and served from cache to the rest —
        instead of once per trainer as with sequential run() calls.
        """
        for tr in trainers:
            os.makedirs(f"{tr.work_dir}/ckpt", exist_ok=True)
            os.makedirs(f"{tr.work_dir}/pred", exist_ok=True)
        months = list(trainers[0]._date_group)
        for i in range(0, len(months) - cls.LOOK_BACK + 1, cls.CKPT_FREQ):
            for tr in trainers:
                tr._run_window(months, i)


def filter_correlated(panel: pd.DataFrame, threshold: float = 0.8) -> list[str]:
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
    X = np.nan_to_num(panel.to_numpy("float32", copy=True), copy=False)
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
        description="Train one model per weight_target group under an execution strategy."
    )
    parser.add_argument("--group", type=int, nargs="+", default=[0, 1, 2, 3],
                        choices=[0, 1, 2, 3],
                        help="group column values; one model is trained per group")
    parser.add_argument("--strategy", default="long_short", choices=Trainer.STRATEGIES,
                        help="execution strategy (sets the XGB objective)")
    parser.add_argument("--feature-dirs", nargs="+", default=None,
                        help="factor-set directories under FACTORS/ASTOCK "
                             "(default: all except REVERSAL32)")
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
    parser.add_argument("--y-shift", type=int, default=0,
                        help="take the target from the eif file this many "
                             "trading days EARLIER (weight/group stay those "
                             "of file t): 1 makes day t's y_1 the return "
                             "close(t)->close(t+1) instead of "
                             "close(t+1)->close(t+2)")
    args = parser.parse_args()

    from functools import lru_cache

    WT_ROOT = "/data/efs/CN-EQ/weight_target"
    DATA_DIR = f"{WT_ROOT}/eif_v" + str(max(
        int(d[len("eif_v"):]) for d in os.listdir(WT_ROOT)
        if d.startswith("eif_v") and d[len("eif_v"):].isdigit()))
    FACTOR_ROOT = "/data/efs/FACTORS/ASTOCK"
    FEATURE_DIRS = args.feature_dirs or sorted(
        d for d in os.listdir(FACTOR_ROOT)
        if os.path.isdir(f"{FACTOR_ROOT}/{d}") and d != "REVERSAL32")
    # The study name comes from --study; the weight/target version (latest
    # eif_v{k}) does not appear in work dirs or model/report names.
    STUDY = args.study
    print(f"study {STUDY}: feature sets {FEATURE_DIRS}", flush=True)

    # End month override: the last month whose every trading day has a file
    # in every feature set AND in the weight/target dir (replaces the
    # embargo-based default; embargoed eif files are owner-readable here).
    _have = None
    for _root in [f"{FACTOR_ROOT}/{n}" for n in FEATURE_DIRS] + [DATA_DIR]:
        _files = {f[:8] for f in os.listdir(_root)
                  if len(f) == 16 and f.endswith(".parquet") and f[:8].isdigit()}
        _have = _files if _have is None else _have & _files
    _by_month = DATES.groupby(DATES.str[:6]).agg(set)
    Trainer.END_MONTH = max(m for m, days in _by_month.items() if days <= _have)
    print(f"end month {Trainer.END_MONTH}: last full-coverage month across "
          f"feature sets + {DATA_DIR.rstrip('/').rsplit('/', 1)[-1]}", flush=True)

    # Raw daily reads, cached so the lockstep trainers hit each file once
    # per date instead of once per group (and w/y once instead of twice).
    # 600 dates covers a full window + pred span (~570 trading days);
    # the concatenated float32 features run ~25 MB/date, so the cache tops
    # out around 15 GB.
    @lru_cache(maxsize=600)
    def read_eif(date: str) -> pd.DataFrame:
        df = pd.read_parquet(f"{DATA_DIR}/{date}.parquet")
        # ukey int32 since 2026-09-02; older eif versions stored strings
        return df.set_axis(pd.to_numeric(df["ukey"]).astype("int32"))

    # Feature sets differ in id column; detect per file. Columns are
    # prefixed with the set name: sets reuse generic names (feat_1 in both
    # DAILY and jeffli_minute_features) and XGBoost needs them unique.
    KEY_COLS = ("ukey", "security_id", "ticker")

    @lru_cache(maxsize=600)
    def read_features(date: str) -> pd.DataFrame:
        parts = []
        for name in FEATURE_DIRS:
            df = pd.read_parquet(f"{FACTOR_ROOT}/{name}/{date}.parquet")
            key = next((c for c in KEY_COLS if c in df.columns), None)
            assert key is not None, f"{name}/{date}: no id column among {KEY_COLS}"
            df = df.set_index(pd.to_numeric(df[key]).astype("int32")).rename_axis("ukey")
            df = df.drop(columns=[c for c in ("timestamp", *KEY_COLS) if c in df.columns])
            # retroactively rekeyed renames (e.g. 000022->001872) appear
            # twice under one ukey in DAILY/DAILY_DICTIONARY; keep first
            df = df[~df.index.duplicated()]
            parts.append(df.astype(np.float32).add_prefix(f"{name}:"))
        return pd.concat(parts, axis=1)

    def make_w_loader(group: int):
        def w_loader(date: str) -> pd.DataFrame:
            df = read_eif(date)
            return df[df.group == group][["weight"]]
        return w_loader

    def make_y_loader(group: int):
        def y_loader(date: str) -> pd.DataFrame:
            df = read_eif(date)
            if args.y_shift:
                # eif file t-k: its y_i spans close(t+1-k) -> close(t+1-k+i);
                # taken on file t's group universe (the Trainer reindexes
                # onto W anyway); names absent from file t-k get NaN and
                # drop out of the day's sample
                i = POS[date] - args.y_shift
                if i < 0 or not os.path.exists(f"{DATA_DIR}/{DATES.iloc[i]}.parquet"):
                    return pd.DataFrame({"y_1": np.nan}, index=df.index[df.group == group])
                src = read_eif(DATES.iloc[i])
                return src[["y_1"]].reindex(df.index[df.group == group])
            return df[df.group == group][["y_1"]]
        return y_loader

    # Factor pre-filter: greedy pairwise-correlation dedup on a quarterly
    # sample of the training range (see filter_correlated). Computed once,
    # applied to every date's frame.
    span = DATES[DATES.str[:6].between(Trainer.SRT_MONTH, Trainer.END_MONTH)]
    if args.factors:
        keep = [l.strip() for l in open(args.factors) if l.strip()]
        print(f"factors: {len(keep)} listed in {args.factors} "
              f"(correlation filter skipped)", flush=True)
    elif args.corr_threshold < 1.0:
        sample = list(span)[::63]
        panel = pd.concat([read_features(d) for d in sample])
        keep = filter_correlated(panel, args.corr_threshold)
        print(f"factor filter (|corr| < {args.corr_threshold}, "
              f"{len(sample)} sample days): kept {len(keep)}/{panel.shape[1]}",
              flush=True)
        del panel
    else:
        keep = None
    # Record the factor set actually trained on, so another study can
    # reuse it verbatim via --factors.
    os.makedirs(f"{args.work_dir}/{STUDY}", exist_ok=True)
    with open(f"{args.work_dir}/{STUDY}/factors.txt", "w") as fh:
        fh.write("\n".join(keep if keep is not None
                           else list(read_features(span.iloc[0]).columns)) + "\n")
    if args.y_shift:
        print(f"y shift {args.y_shift}: day t's target is y_1 of eif file "
              f"t-{args.y_shift}", flush=True)

    def x_loader(date: str) -> pd.DataFrame:
        # reindex, not [keep]: the factor set drifts across dates, and a
        # kept column absent on some day must yield NaN (-> 0 downstream)
        df = read_features(date)
        return df.reindex(columns=keep) if keep is not None else df

    trainers = [
        Trainer(
            work_dir=f"{args.work_dir}/{STUDY}/group{group}_{args.strategy}",
            w_loader=make_w_loader(group),
            x_loader=x_loader,
            y_loader=make_y_loader(group),
            strategy=args.strategy,
        )
        for group in args.group
    ]
    # Windows outer, groups inner: each date's files are read once and
    # shared across the groups through the caches above.
    Trainer.run_lockstep(trainers)
