"""Per-day second moments of predictors against targets, for correlation
studies.

For evaluating predictors — Trainer predictions or raw factors — only
correlations are needed, so only the diagonal of xx is kept. Each x
column is one "fitter". Per day:
- xx: weighted second moment of each fitter, dims (fitter, side, group)
- xy: weighted cross-moments of fitters and responses, dims (fitter, side, group, response)
- yy: weighted second moments of the responses, dims (fitter, side, group, response)
- sw: sum of weights, dims (fitter, side, group)
where "side" restricts the slots to all / x > 0 / x < 0 and "group" to
all / one of the upstream data's group labels 0..3, supplied per
ukey by the g_loader (see _calculate_stat). Either coordinate can be limited
to a subset; the full 3 x 5 side/group grid is the default and forms the
report buckets.
Moments are about zero and NaN contributes zero. run() stacks the days
along the outer dim "date", attaches the summary metrics — e.g. the
weighted correlation xy / sqrt(xx * yy) pooled over days; see
gen_metric — and writes {work_dir}/eval.nc.
"""

import os

import numpy as np
import pandas as pd
import xarray as xr

from trainer import Trainer, POS

# Trading calendar (YYYYMMDD per row), as in trainer.py.
CALENDAR_FILE = "/data/efs/CN-EQ/calendar.csv"
DATES = pd.read_csv(CALENDAR_FILE, dtype=str)["date"]

class Evaluator:
    """Daily x/response second moments, one "fitter" per x column.

    Users configure work_dir and the w/y/x/g loaders, as in Trainer. W must
    yield a single value column besides ukey; Y yields one column per
    response (e.g. y_1/y_3/y_5) and X one column per fitter — a raw feature
    frame to screen factors, or Trainer daily pred files to evaluate model
    predictions. G yields each ukey's group label (the upstream data's
    group column, 0..3) for the "group" slices. The side and group
    coordinates default to the full SIDES/GROUPS but can be restricted to
    any non-empty subset. run() writes {work_dir}/eval.nc.
    """

    # The full side/group coordinate sets (see _calculate_stat).
    SIDES = ("all", "pos", "neg")
    GROUPS = ("all", "0", "1", "2", "3")

    # Date range, maintained by the Trainer admin team.
    SRT_DATE = "20190102"
    END_DATE = "20260228"

    def __init__(
        self,
        work_dir,  # root directory for the output (eval.nc)
        w_loader,  # callable: date "YYYYMMDD" -> W frame indexed by ukey, columns = [weight]
        y_loader,  # callable: date "YYYYMMDD" -> Y frame indexed by ukey, columns = [target]
        x_loader,  # callable: date "YYYYMMDD" -> X frame indexed by ukey, columns = [features]
        g_loader,  # callable: date "YYYYMMDD" -> Series of group labels (0..3) indexed by ukey
        side=SIDES,  # side coordinates to compute, non-empty subset of SIDES
        group=GROUPS,  # group coordinates to compute, non-empty subset of GROUPS
    ):
        assert side and set(side) <= set(self.SIDES), f"side must be a non-empty subset of {self.SIDES}, got {side!r}"
        assert group and set(group) <= set(self.GROUPS), f"group must be a non-empty subset of {self.GROUPS}, got {group!r}"
        self.work_dir = work_dir
        self.w_loader = w_loader
        self.y_loader = y_loader
        self.x_loader = x_loader
        self.g_loader = g_loader
        self.side = list(side)
        self.group = list(group)

    def _load_daily_data(self, date: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
        """Load one day's (w, y, x, g) data, indexed by ukey.

        y, x and g are reindexed onto W's ukey universe; only samples
        with an available weight are kept. NaN values elsewhere contribute
        zero to the moments (a NaN group label falls outside every 0..3
        slice but still counts in group "all").
        """
        w: pd.DataFrame = self.w_loader(date)
        y: pd.DataFrame = self.y_loader(date).reindex(index = w.index)
        x: pd.DataFrame = self.x_loader(date).reindex(index = w.index).fillna(0.0)
        g: pd.Series = self.g_loader(date).reindex(index = w.index)
        assert len(w.columns) == 1, f"W must have a unique value column, got {list(w.columns)}"
        keep = w.iloc[:,0].notna()
        return w[keep], y[keep], x[keep], g[keep]

    def _calculate_stat(self, w: pd.DataFrame, y: pd.DataFrame, x: pd.DataFrame, g: pd.Series) -> dict[str, np.ndarray]:
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
        self.group) are computed.
        """
        w = np.nan_to_num(w.to_numpy())
        y = np.nan_to_num(y.to_numpy())
        x = np.nan_to_num(x.to_numpy())
        g = g.to_numpy()
        # Side masks per slot and fitter, (n, fitter).
        masks = [
            np.ones_like(x, dtype=bool) if s == "all" else x > 0.0 if s == "pos" else x < 0.0
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

    def run(self) -> xr.Dataset:
        """Compute the moments for every calendar date in range and save
        eval.nc. Daily numpy stats are stacked into a single Dataset at
        the end (see _calculate_stat)."""
        dates: list[str] = list(DATES[DATES.between(self.SRT_DATE, self.END_DATE)])
        days, fitters, responses = [], None, None
        for date in dates:
            w, y, x, g = self._load_daily_data(date)
            if fitters is None:
                fitters, responses = list(x.columns), list(y.columns)
            assert list(x.columns) == fitters and list(y.columns) == responses, \
                f"fitter/response columns changed on {date}"
            days.append(self._calculate_stat(w, y, x, g))
        dims = {
            "xx": ("fitter", "side", "group"),
            "xy": ("fitter", "side", "group", "response"),
            "yy": ("fitter", "side", "group", "response"),
            "sw": ("fitter", "side", "group"),
        }
        ds = xr.Dataset(
            {v: (("date", *dim), np.stack([d[v] for d in days])) for v, dim in dims.items()},
            coords={
                "date": pd.Index(pd.to_datetime(dates), name="date"),
                "fitter": fitters,
                "side": self.side,
                "group": self.group,
                "response": responses,
            },
        )
        ds = self.gen_metric(ds)
        ds.to_netcdf(f"{self.work_dir}/eval.nc")
        return ds


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate the stitched per-group models of one execution strategy."
    )
    parser.add_argument("--group", type=int, nargs="+", default=[0, 1, 2, 3],
                        choices=[0, 1, 2, 3],
                        help="groups whose pred files are stitched into the fitter")
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
    parser.add_argument("--y-shift", type=int, default=0,
                        help="take the responses from the eif file this many "
                             "trading days earlier (match the trainer run)")
    args = parser.parse_args()

    from functools import lru_cache

    WT_ROOT = "/data/efs/CN-EQ/weight_target"
    DATA_DIR = f"{WT_ROOT}/eif_v" + str(max(
        int(d[len("eif_v"):]) for d in os.listdir(WT_ROOT)
        if d.startswith("eif_v") and d[len("eif_v"):].isdigit()))
    # The study name comes from --study (matching the trainer run); the
    # weight/target version (latest eif_v{k}) does not appear in work dirs
    # or report names.
    STUDY = args.study
    WORK_ROOT = f"{args.work_dir}/{STUDY}"

    # One read per date, shared by the w/y/g loaders (the run loop asks
    # for the same date three times in a row).
    @lru_cache(maxsize=8)
    def read_eif(date: str) -> pd.DataFrame:
        df = pd.read_parquet(f"{DATA_DIR}/{date}.parquet")
        # ukey int32 since 2026-09-02; older eif versions stored strings
        return df.set_axis(pd.to_numeric(df["ukey"]).astype("int32"))

    def w_loader(date: str) -> pd.DataFrame:
        return read_eif(date)[["weight"]]

    def y_loader(date: str) -> pd.DataFrame:
        # with --y-shift k the responses come from eif file t-k (y_i then
        # spans close(t+1-k) -> close(t+1-k+i)), as in the trainer
        src = read_eif(DATES.iloc[POS[date] - args.y_shift]) if args.y_shift else read_eif(date)
        return src[["y_1", "y_3", "y_5"]]

    def g_loader(date: str) -> pd.Series:
        return read_eif(date)["group"]

    def x_loader(date: str) -> pd.DataFrame:
        # Each group's trainer predicts only its own ukeys, so the group
        # pred files partition the universe; stitch them into one fitter.
        # eif file {date} (w/y here) and pred/{date} both describe the trade
        # on {date}+1 — the trainer builds pred/{date} from the same-date
        # feature file — so w, y and pred of the same date are aligned;
        # no extra shift here.
        preds = pd.concat(
            pd.read_parquet(f"{WORK_ROOT}/group{g}_{args.strategy}/pred/{date}.parquet")
            for g in args.group
        )
        return preds.set_axis(pd.to_numeric(preds["ukey"]).astype("int32"))

    os.makedirs(f"{WORK_ROOT}/{args.strategy}", exist_ok=True)
    evaluator = Evaluator(
        work_dir=f"{WORK_ROOT}/{args.strategy}",
        w_loader=w_loader,
        x_loader=x_loader,
        y_loader=y_loader,
        g_loader=g_loader,
        side=args.side,
        # Slice the stats by the groups being evaluated (plus the whole book).
        group=["all"] + [str(g) for g in args.group],
    )
    # Preds start LOOK_BACK months after the trainer's SRT_MONTH (the first
    # in-window months produce no predictions). The end tracks what the
    # trainer actually produced — its END_MONTH is coverage-based (last
    # month with full feature + weight/target coverage), so take the last
    # pred date present in every evaluated group.
    m0 = int(Trainer.SRT_MONTH[:4]) * 12 + int(Trainer.SRT_MONTH[4:]) - 1
    m1 = m0 + Trainer.LOOK_BACK
    evaluator.SRT_DATE = f"{m1 // 12}{m1 % 12 + 1:02d}01"
    evaluator.END_DATE = min(
        max(f[:8] for f in os.listdir(f"{WORK_ROOT}/group{g}_{args.strategy}/pred")
            if f.endswith(".parquet"))
        for g in args.group)
    evaluator.run()