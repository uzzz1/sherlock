"""Reference x_loader (xloader_contract): the FACTORS/ASTOCK daily sets.

x_loader(date) -> pl.DataFrame [ukey Int32, time Datetime[ns, UTC],
*features]: the date's daily feature files from every set under
/data/FACTORS (all except REVERSAL32 by default — edit
FEATURE_DIRS or copy this file for another selection), cross-joined to
the union of both tasks' prediction instants (from targets.TASK_SECONDS,
so the grid can never drift from the targets; conform() keeps only the
instants under evaluation, so the same loader serves --task 0 and 1).

A daily feature file of day s carries information through day s's
market close, but every prediction instant of day t — 09:24:59
included — fires BEFORE day t's close, so day t's instants see the
PREVIOUS trading day's file (held constant across the day; the
intraday-staleness question is the model's to learn). Joining the
same-date file instead leaks the day's own session into the pre-open
and intraday decisions — measured on a task-0 smoke: OOS corr jumps
from ~0.03 to 0.72. Columns are prefixed with "{set}:" (sets reuse
generic names like feat_1); the id column is detected per file
(ukey / security_id / ticker); rename-duplicate ukeys keep their
first row.
"""

import os

import polars as pl

import targets

FACTOR_ROOT = "/data/FACTORS"
FEATURE_DIRS: list[str] | None = None       # None = all sets except REVERSAL32
KEY_COLS = ("ukey", "security_id", "ticker")


def feature_dirs() -> list[str]:
    if FEATURE_DIRS is not None:
        return FEATURE_DIRS
    return sorted(d for d in os.listdir(FACTOR_ROOT)
                  if os.path.isdir(f"{FACTOR_ROOT}/{d}") and d != "REVERSAL32")


def gen_times(date: str) -> pl.Series:
    """Union of both tasks' prediction instants on `date`."""
    secs = sorted(set(targets.TASK_SECONDS[0]) | set(targets.TASK_SECONDS[1]))
    return (pl.Series("time", targets.times_to_utc_ns(date, secs))
            .cast(pl.Datetime("ns", "UTC")))


def read_daily(date: str) -> pl.DataFrame:
    """One row per ukey: the date's feature files, concatenated and
    prefixed."""
    out = None
    for name in feature_dirs():
        df = pl.read_parquet(f"{FACTOR_ROOT}/{name}/{date}.parquet")
        key = next((c for c in KEY_COLS if c in df.columns), None)
        assert key is not None, f"{name}/{date}: no id column among {KEY_COLS}"
        feats = [c for c in df.columns if c not in ("timestamp", *KEY_COLS)]
        df = (df.with_columns(pl.col(key).cast(pl.Int64).cast(pl.Int32)
                              .alias("ukey"))
              .unique(subset="ukey", keep="first", maintain_order=True)
              .select("ukey", *[pl.col(c).cast(pl.Float64)
                                .alias(f"{name}:{c}") for c in feats]))
        out = df if out is None else out.join(df, on="ukey", how="full",
                                              coalesce=True)
    return out


CAL = targets.load_calendar()
POS = {d: i for i, d in enumerate(CAL)}


def x_loader(date: str) -> pl.DataFrame:
    daily = read_daily(CAL[POS[date] - 1])      # previous trading day's file
    return (daily.join(pl.DataFrame({"time": gen_times(date)}), how="cross")
            .select("ukey", "time",
                    *[c for c in daily.columns if c != "ukey"]))
