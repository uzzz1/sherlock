"""Per-day second moments of predictors against targets, for correlation
studies.

Intraday edition (2026-09-19): samples are keyed by (ukey, time) — one
eif task's prediction instants — with responses from target.{task}/ and
the daily (ukey, weight, group) file weight/ broadcast to each instant
(targets.py; eif_v11 — the per-instant intraday weight is retired); a
day's moments pool its instants, so the date dim and the
15-bucket eval.nc format are unchanged. I/O is polars.

For evaluating predictors — Trainer predictions or raw factors — only
correlations are needed, so only the diagonal of xx is kept. Each x
column is one "fitter". Per day:
- xx: weighted second moment of each fitter, dims (fitter, side, group)
- xy: weighted cross-moments of fitters and responses, dims (fitter, side, group, response)
- yy: weighted second moments of the responses, dims (fitter, side, group, response)
- sw: sum of weights, dims (fitter, side, group)
where "side" restricts the slots to all / x > 0 / x < 0 and "group" to
all / one of the upstream data's group labels 0..3. Either coordinate can
be limited to a subset; the full 3 x 5 side/group grid is the default and
forms the report buckets.
Moments are about zero and NaN contributes zero. run() stacks the days
along the outer dim "date", attaches the summary metrics — e.g. the
weighted correlation xy / sqrt(xx * yy) pooled over days; see
gen_metric — and writes {work_dir}/eval.nc.

Limit-locked rows (2026-10-06): when W carries a `lock` column
(targets.join_limit_mask: +1 mark on 涨停, -1 on 跌停, 0 free, null no
mark) the `lock_mask` mode removes the unfillable slots from every
moment before the side/group masks — "unfillable" (default) drops a
slot where the fitter's x would buy into 涨停 (x > 0, lock +1) or sell
into 跌停 (x < 0, lock -1), plus null-lock slots; "locked" drops every
lock != 0 or null slot regardless of x; "none" keeps all. Locked names
are <1% of the rows but ~40% of the pooled y_1 cross-moment (limit
continuation a book cannot fill), so the unmasked numbers overstate
what is realizable; the mode is recorded in eval.nc's attrs and the
non-default modes write eval.{mode}.nc.

Turnover (2026-10-06): eval.nc also carries `turnover` (date, fitter,
book, tod) — how much a book built on the fitter trades. Each decision
slot (tod, Asia/Shanghai clock time) is its own portfolio, rebalanced
ONCE A DAY at the same slot (slot s of day d -> slot s of day d+1; never
slot to slot within a day). Two gross-1 books per fitter: "linear" =
weight x weighted z-score of the fitter, "decile" = top decile +weight /
bottom decile -weight, each leg gross 0.5. turnover = 0.5 * sum |p_{d+1}
- p_d| over the union of names, so 0 = untouched, 1 = fully replaced,
and universe entries/exits count as trades. Only slots the lock mask
keeps hold a position (an unfillable slot holds nothing), so the books
are the fillable ones the moments describe; a (slot, fitter) with fewer
than MIN_ROWS such rows is NaN, as is the first date.
"""

import os

import numpy as np
import polars as pl
import xarray as xr

from targets import fmt_time
from trainer import DATES, ID_COLS, model_dirs


class Evaluator:
    """Daily x/response second moments, one "fitter" per x column.

    Users configure work_dir and the w/y/x loaders, as in Trainer: each
    maps a date "YYYYMMDD" to a polars frame keyed by (ukey Int32, time
    Datetime[ns, UTC]). W carries `weight` and the day's `group` label
    (0..3, constant across a name's instants); Y one column per response
    (e.g. y_1/y_3/y_5); X one column per fitter — Trainer daily pred files
    to evaluate model predictions, or a wide factor frame to screen
    factors. The side and group coordinates default to the full
    SIDES/GROUPS but can be restricted to any non-empty subset. W may
    carry a `lock` column, consumed per `lock_mask` (module docstring).
    run() writes {work_dir}/eval.nc.
    """

    # The full side/group coordinate sets (see _calculate_stat).
    SIDES = ("all", "pos", "neg")
    GROUPS = ("all", "0", "1", "2", "3")
    # Lock-mask modes (module docstring); the first is the default.
    LOCK_MASKS = ("unfillable", "locked", "none")
    # Turnover books (module docstring), gross 1 each; MIN_ROWS = rows a
    # (slot, fitter) book needs to be scored.
    BOOKS = ("linear", "decile")
    MIN_ROWS = 100

    # Date range, maintained by the Trainer admin team.
    SRT_DATE = "20190102"
    END_DATE = "20260228"

    def __init__(
        self,
        work_dir,  # root directory for the output (eval.nc)
        w_loader,  # callable: date "YYYYMMDD" -> [ukey, time, weight, group]
        y_loader,  # callable: date "YYYYMMDD" -> [ukey, time, *responses]
        x_loader,  # callable: date "YYYYMMDD" -> [ukey, time, *fitters]
        side=SIDES,  # side coordinates to compute, non-empty subset of SIDES
        group=GROUPS,  # group coordinates to compute, non-empty subset of GROUPS
        lock_mask=LOCK_MASKS[0],  # how W's `lock` column masks the slots, one of LOCK_MASKS
    ):
        assert side and set(side) <= set(self.SIDES), f"side must be a non-empty subset of {self.SIDES}, got {side!r}"
        assert group and set(group) <= set(self.GROUPS), f"group must be a non-empty subset of {self.GROUPS}, got {group!r}"
        assert lock_mask in self.LOCK_MASKS, f"lock_mask must be one of {self.LOCK_MASKS}, got {lock_mask!r}"
        self.lock_mask = lock_mask
        self.work_dir = work_dir
        self.w_loader = w_loader
        self.y_loader = y_loader
        self.x_loader = x_loader
        self.side = list(side)
        self.group = list(group)

    def _load_daily_data(self, date: str) -> tuple[pl.DataFrame, list[str], list[str]]:
        """One day's joined frame [ukey, time, weight, group[, lock],
        *responses, *fitters] plus the fitter/response column names.

        Y and X are joined onto W's (ukey, time) universe; only samples
        with an available weight are kept. NaN values elsewhere contribute
        zero to the moments (a NaN group label falls outside every 0..3
        slice but still counts in group "all")."""
        w = self.w_loader(date)
        y = self.y_loader(date)
        x = self.x_loader(date)
        assert w.columns[:4] == [*ID_COLS, "weight", "group"] and set(w.columns[4:]) <= {"lock"}, \
            f"W columns {w.columns}"
        if "lock" not in w.columns:
            assert self.lock_mask == "none", f"lock_mask {self.lock_mask!r} needs a `lock` column in W"
        responses = [c for c in y.columns if c not in ID_COLS]
        fitters = [c for c in x.columns if c not in ID_COLS]
        df = (w.filter(pl.col("weight").is_finite())
              .join(y, on=list(ID_COLS), how="left")
              .join(x, on=list(ID_COLS), how="left"))
        return df, fitters, responses

    @staticmethod
    def _lock(df: pl.DataFrame) -> np.ndarray:
        """W's lock column as float (NaN = no mask row), zeros without one."""
        return (df["lock"].cast(pl.Float64).to_numpy() if "lock" in df.columns
                else np.zeros(len(df)))

    def _arrays(self, df: pl.DataFrame, fitters: list[str], responses: list[str]
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """The day's aligned (w, y, x, g, lock) arrays for _calculate_stat;
        NaN responses/fitters become 0."""
        return (df["weight"].to_numpy(),
                np.nan_to_num(df.select(responses).to_numpy()),
                np.nan_to_num(df.select(fitters).to_numpy()),
                df["group"].to_numpy(),
                self._lock(df))

    def _fill_mask(self, x: np.ndarray, lock: np.ndarray) -> np.ndarray:
        """(n, fitter) bool: the slots the lock_mask mode keeps (module
        docstring). NaN lock (no mark) is unfillable in every masking mode."""
        if self.lock_mask == "none":
            return np.ones_like(x, dtype=bool)
        lk = lock[:, None]
        if self.lock_mask == "locked":
            return np.broadcast_to(lk == 0.0, x.shape)
        return ~(((lk == 1.0) & (x > 0.0)) | ((lk == -1.0) & (x < 0.0)) | np.isnan(lk))

    def _calculate_stat(self, w: np.ndarray, y: np.ndarray, x: np.ndarray, g: np.ndarray,
                        lock: np.ndarray | None = None) -> dict[str, np.ndarray]:
        """One day's second moments as plain numpy arrays — xx (fitter,
        side, group), xy (fitter, side, group, response), yy like xy, sw
        like xx. run() stacks the days and builds the Dataset once; a
        per-day xr.Dataset would make run() pay for thousands of tiny
        xarray constructions and a giant concat.

        "side" masks the slots to every slot ("all") or those where the
        fitter's x is positive ("pos") / negative ("neg") — the rest
        contribute zero, like NaN — so for yy and sw the side is
        fitter-dependent. "group" masks the slots to every slot ("all") or
        those whose group label g — the upstream data's group column —
        equals 0..3. Only the configured side/group coordinates (self.side,
        self.group) are computed. The lock mask (_fill_mask) multiplies
        into every side mask, so unfillable slots leave xx, xy, yy and sw
        alike.
        """
        w = w[:, None] if w.ndim == 1 else w
        fill = self._fill_mask(x, np.zeros(len(x)) if lock is None else lock)
        # Side masks per slot and fitter, (n, fitter).
        masks = [
            (np.ones_like(x, dtype=bool) if s == "all" else x > 0.0 if s == "pos" else x < 0.0) & fill
            for s in self.side
        ]
        # Per-group weights: each mask zeroes the slots outside its group.
        wg = [w if k == "all" else w * (g == int(k))[:, None] for k in self.group]
        # Stacking yields (side, group, fitter, ...); move fitter first.
        return {
            "xx": np.moveaxis(np.stack([np.stack([((x * m) ** 2 * v).sum(axis=0) for v in wg]) for m in masks]), 2, 0),
            "xy": np.moveaxis(np.stack([np.stack([(x * m * v).T @ y for v in wg]) for m in masks]), 2, 0),
            "yy": np.moveaxis(np.stack([np.stack([(m * v).T @ y**2 for v in wg]) for m in masks]), 2, 0),
            "sw": np.moveaxis(np.stack([np.stack([(m * v).sum(axis=0) for v in wg]) for m in masks]), 2, 0),
        }

    # ----------------------------------------------------------- turnover
    @staticmethod
    def _ranks(x: np.ndarray, keep: np.ndarray) -> np.ndarray:
        """Per-column ranks (1..n_kept) among the kept rows; others last."""
        order = np.argsort(np.where(keep, x, np.inf), axis=0, kind="stable")
        rank = np.empty_like(order)
        np.put_along_axis(rank, order, np.arange(len(x))[:, None], axis=0)
        return (rank + 1).astype(float)

    def _books(self, w: np.ndarray, x: np.ndarray, keep: np.ndarray) -> np.ndarray:
        """(book, n, fitter) positions of one slot's BOOKS per fitter, gross
        1 per column (module docstring). `keep` (n, fitter) marks the rows
        that may hold a position; a column with fewer than MIN_ROWS kept
        rows is NaN throughout."""
        Wv = keep * w[:, None]
        sw = Wv.sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            mu = (np.where(keep, x, 0.0) * Wv).sum(axis=0) / sw
            dev = np.where(keep, x - mu, 0.0)
            sd = np.sqrt((dev ** 2 * Wv).sum(axis=0) / sw)
            lin = dev / sd * w[:, None]
            lin = lin / np.abs(lin).sum(axis=0)
            q = self._ranks(x, keep) / keep.sum(axis=0)
            top = keep & (q > 0.9)
            bot = keep & (q <= 0.1)
            dec = (np.where(top, w[:, None], 0.0) / (2 * (top * w[:, None]).sum(axis=0))
                   - np.where(bot, w[:, None], 0.0) / (2 * (bot * w[:, None]).sum(axis=0)))
        bad = keep.sum(axis=0) < self.MIN_ROWS
        lin[:, bad] = np.nan
        dec[:, bad] = np.nan
        return np.stack([lin, dec])

    def _slot_books(self, df: pl.DataFrame, fitters: list[str]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """{seconds of day: (ukeys, books)} of one day's frame, slot by
        slot. A row may hold a position when its fitter is finite and the
        lock mask keeps it (_fill_mask), so unfillable slots hold nothing."""
        x = df.select(fitters).to_numpy()
        keep = np.isfinite(x) & self._fill_mask(np.nan_to_num(x), self._lock(df))
        tod = (df["time"].cast(pl.Int64).to_numpy() // 1_000_000_000 + 8 * 3600) % 86400
        w = df["weight"].to_numpy()
        u = df["ukey"].to_numpy()
        return {int(s): (u[i], self._books(w[i], x[i], keep[i]))
                for s in np.unique(tod) for i in [tod == s]}

    def _turnover(self, a: pl.DataFrame, b: pl.DataFrame, fitters: list[str]) -> dict[int, np.ndarray]:
        """{seconds of day: (fitter, book)} turnover of each fitter's books
        from day a to day b at the same slot: 0.5 * sum |p_b - p_a| over the
        union of names, a name missing from one side holding 0 there. Slots
        absent on either day are skipped."""
        pa, pb = self._slot_books(a, fitters), self._slot_books(b, fitters)
        out = {}
        for s in sorted(pa.keys() & pb.keys()):
            (ua, Pa), (ub, Pb) = pa[s], pb[s]
            u = np.union1d(ua, ub)
            P = np.zeros((2, len(self.BOOKS), len(u), len(fitters)))
            P[0][:, np.searchsorted(u, ua)] = Pa
            P[1][:, np.searchsorted(u, ub)] = Pb
            out[s] = (0.5 * np.abs(P[1] - P[0]).sum(axis=1)).T
        return out

    def gen_metric(self, ds: xr.Dataset) -> xr.Dataset:
        """Attach the summary metrics (corr, spnl, sharpe, curve, win) to
        the daily-moment Dataset."""
        ds_sum = ds.sum(dim = "date", skipna = True)
        ds["sumw"] = ds_sum.sw
        ds["stdx"] = np.sqrt(ds_sum.xx / ds_sum.sw)
        ds["stdy"] = np.sqrt(ds_sum.yy / ds_sum.sw)
        ds["corr"] = ds_sum.xy / np.sqrt(ds_sum.xx * ds_sum.yy)
        ds["spnl"] = ds_sum.xy / np.sqrt(ds_sum.xx * ds_sum.sw)
        ds["win"] = (ds.xy >= 0.0).mean(dim = "date", skipna = True)
        ds["sharpe"] = ds.xy.mean(dim = "date", skipna = True) / ds.xy.std(dim = "date", skipna = True) * np.sqrt(252)
        ds["curve"] = ds.xy.cumsum(dim = "date", skipna = True) / np.sqrt(ds_sum.xx * ds_sum.sw)
        return ds

    def eval_path(self) -> str:
        """{work_dir}/eval.nc for the default lock_mask, else
        {work_dir}/eval.{lock_mask}.nc — so the modes coexist."""
        tag = "" if self.lock_mask == self.LOCK_MASKS[0] else f".{self.lock_mask}"
        return f"{self.work_dir}/eval{tag}.nc"

    def run(self) -> xr.Dataset:
        """Compute the moments for every calendar date in range and save
        eval_path(). Daily numpy stats are stacked into a single Dataset
        at the end (see _calculate_stat)."""
        dates = [d for d in DATES.to_list() if self.SRT_DATE <= d <= self.END_DATE]
        days, turns, prev, fitters, responses = [], [], None, None, None
        for date in dates:
            df, fit_d, resp_d = self._load_daily_data(date)
            if fitters is None:
                fitters, responses = fit_d, resp_d
            assert fit_d == fitters and resp_d == responses, \
                f"fitter/response columns changed on {date}"
            days.append(self._calculate_stat(*self._arrays(df, fitters, responses)))
            # Turnover needs yesterday's books at the same slots.
            turns.append(self._turnover(prev, df, fitters) if prev is not None else {})
            prev = df
        dims = {
            "xx": ("fitter", "side", "group"),
            "xy": ("fitter", "side", "group", "response"),
            "yy": ("fitter", "side", "group", "response"),
            "sw": ("fitter", "side", "group"),
        }
        ds = xr.Dataset(
            {v: (("date", *dim), np.stack([d[v] for d in days])) for v, dim in dims.items()},
            coords={
                "date": np.array(
                    [f"{d[:4]}-{d[4:6]}-{d[6:]}" for d in dates],
                    dtype="datetime64[ns]"),
                "fitter": fitters,
                "side": self.side,
                "group": self.group,
                "response": responses,
            },
        )
        tods = sorted(set().union(*(t.keys() for t in turns)))
        to = np.full((len(dates), len(fitters), len(self.BOOKS), len(tods)), np.nan)
        for i, t in enumerate(turns):
            for j, s in enumerate(tods):
                if s in t:
                    to[i, :, :, j] = t[s]
        ds = ds.assign_coords(book=list(self.BOOKS), tod=[fmt_time(s) for s in tods])
        ds["turnover"] = (("date", "fitter", "book", "tod"), to)
        ds = self.gen_metric(ds)
        ds.attrs["lock_mask"] = self.lock_mask
        ds.to_netcdf(self.eval_path())
        print(f"wrote {self.eval_path()} (lock_mask {self.lock_mask}, {len(dates)} dates)", flush=True)
        return ds


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate the stitched models of one execution strategy."
    )
    parser.add_argument("--task", type=int, required=True, choices=[0, 1],
                        help="prediction task the study was trained on")
    parser.add_argument("--group", type=int, nargs="+", default=[0, 1, 2, 3],
                        choices=[0, 1, 2, 3],
                        help="groups the study was trained over (names its "
                             "group{labels}_... model dirs)")
    parser.add_argument("--strategy", default="long_short",
                        choices=("long_short", "long_only", "short_only"),
                        help="execution strategy whose work dirs to evaluate")
    parser.add_argument("--side", nargs="+", default=list(Evaluator.SIDES),
                        choices=Evaluator.SIDES,
                        help="side coordinates to compute")
    parser.add_argument("--study", default="baseline",
                        help="study name of the trainer run to evaluate "
                             "(names the work dirs)")
    parser.add_argument("--work-dir", required=True,
                        help="root of the trainer's work dirs")
    parser.add_argument("--lock-mask", default=Evaluator.LOCK_MASKS[0],
                        choices=Evaluator.LOCK_MASKS,
                        help="limit-locked slots to drop from the moments: "
                             "'unfillable' (default) = buys into 涨停 / sells "
                             "into 跌停 per the fitter's sign plus no-mark "
                             "slots, 'locked' = every locked/no-mark slot, "
                             "'none' = keep all (writes eval.{mode}.nc when "
                             "not the default)")
    args = parser.parse_args()

    from functools import lru_cache

    import targets

    EIF_DIR = targets.latest_version_dir(args.task)
    # The study name comes from --study (matching the trainer run); the
    # weight/target version (latest eif_v{k}) does not appear in work dirs
    # or report names.
    STUDY = args.study
    WORK_ROOT = f"{args.work_dir}/{STUDY}"
    PRED_DIRS = [f"{WORK_ROOT}/{d}/pred"
                 for d in model_dirs(WORK_ROOT, args.group, args.strategy)]

    @lru_cache(maxsize=8)
    def w_loader(date: str) -> pl.DataFrame:
        w = targets.load_weights(date, EIF_DIR, args.task)
        return w if args.lock_mask == "none" else targets.join_limit_mask(w, date, EIF_DIR)

    @lru_cache(maxsize=8)
    def y_loader(date: str) -> pl.DataFrame:
        return targets.load_targets(date, args.task, EIF_DIR,
                                    ["y_1", "y_3", "y_5"])

    def x_loader(date: str) -> pl.DataFrame:
        # The study's model dirs (task 1's open/rest instant slices) each
        # predict their own (ukey, time) rows, so their pred files
        # partition the universe; stitch them into one fitter.
        # Pred/{date} and target/weight files of the same date all
        # describe the same decision instants — no shift here.
        return (pl.concat([pl.read_parquet(f"{d}/{date}.parquet") for d in PRED_DIRS])
                .rename({"timestamp": "time"}))

    os.makedirs(f"{WORK_ROOT}/{args.strategy}", exist_ok=True)
    evaluator = Evaluator(
        work_dir=f"{WORK_ROOT}/{args.strategy}",
        w_loader=w_loader,
        x_loader=x_loader,
        y_loader=y_loader,
        side=args.side,
        # Slice the stats by the groups being evaluated (plus the whole book).
        group=["all"] + [str(g) for g in args.group],
        lock_mask=args.lock_mask,
    )
    # Evaluate exactly the dates the trainer produced: [first, last] pred
    # date present in every evaluated group (the trainer's SRT/END months
    # are coverage-derived, so the pred listing is the ground truth).
    preds = [sorted(f[:8] for f in os.listdir(d) if f.endswith(".parquet"))
             for d in PRED_DIRS]
    evaluator.SRT_DATE = max(p[0] for p in preds)
    evaluator.END_DATE = min(p[-1] for p in preds)
    evaluator.run()
