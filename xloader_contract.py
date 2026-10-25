"""The x_loader contract: how users plug their feature files into the pipeline.

An x_loader is any callable

    x_loader(date: str) -> pl.DataFrame        # date "YYYYMMDD"

returning one row per (ukey, time) with the day's feature values:

    ukey     integer-like security key (int32 convention)
    time     the prediction instant the features are KNOWN BY — tz-aware
             datetime (any unit; Asia/Shanghai or UTC both fine)
    <f1>..   one numeric column per feature; NaN/null = feature not
             available for that (ukey, time)

It may cover any superset of ukeys and times; the pipeline intersects with the
task's universe and the evaluated timepoints. Sparse-in-time loaders
(e.g. a daily factor replicated to every slot) and dense intraday
loaders both fit the same contract — see xloader_example.py.

`resolve` turns a CLI spec into the callable; `conform` normalizes and
validates one day's frame so the evaluator can rely on exact dtypes.
"""

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Callable

import polars as pl

XLoader = Callable[[str], pl.DataFrame]

ID_COLS = ("ukey", "time")


def resolve(spec: str) -> tuple[str, XLoader]:
    """'module:attr', 'module' (attr defaults to x_loader), or
    'path/to/file.py[:attr]' -> (display name, callable)."""
    target, _, attr = spec.partition(":")
    attr = attr or "x_loader"
    if target.endswith(".py"):
        path = Path(target).resolve()
        mod_name = path.stem
        module_spec = importlib.util.spec_from_file_location(mod_name, path)
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[mod_name] = module
        module_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(target)
    fn = getattr(module, attr)
    if not callable(fn):
        raise TypeError(f"{spec}: {attr} is not callable")
    return f"{module.__name__}:{attr}", fn


def conform(x: pl.DataFrame, date: str, times_ns: list[int]) -> pl.DataFrame:
    """Normalize one day's x_loader output for the evaluator.

    Casts ukey -> Int32 and time -> Datetime[ns, UTC], keeps only the
    evaluated timepoints, casts features to Float64, and asserts
    (ukey, time) uniqueness. Raises with the date in the message —
    contract violations should fail the day loudly, not skew metrics."""
    missing = [c for c in ID_COLS if c not in x.columns]
    if missing:
        raise ValueError(f"{date}: x_loader output lacks {missing}; "
                         f"columns = {x.columns}")
    features = [c for c in x.columns if c not in ID_COLS]
    if not features:
        raise ValueError(f"{date}: x_loader output has no feature columns")
    if x["time"].dtype.time_zone is None:
        raise ValueError(f"{date}: x_loader 'time' must be tz-aware")
    x = (x.with_columns(
            pl.col("ukey").cast(pl.Int32),
            pl.col("time").dt.convert_time_zone("UTC")
              .dt.cast_time_unit("ns"),
            *[pl.col(f).cast(pl.Float64) for f in features])
         .filter(pl.col("time").cast(pl.Int64).is_in(times_ns)))
    if x.select(pl.struct(ID_COLS).is_duplicated().any()).item():
        raise ValueError(f"{date}: duplicate (ukey, time) rows in x_loader "
                         f"output")
    return x.select([*ID_COLS, *features])
