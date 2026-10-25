"""Access to the 5-minute intraday targets and universe under
/data/CN-EQ/weight_target/eif_v{k} (see that directory's README).

Two prediction tasks share one universe and one target construction:

    task 0  盘前   one decision per day at 09:24:59 (fills at the 09:25
                   auction; its target equals the 09:30 point's)
    task 1  盘中   48 decisions: 09:30, 09:35 .. 11:25, 13:00 .. 14:55

Targets live under the eif version root as target.{task}/{YYYYMMDD}.parquet.
The universe, its group label and the sample weight are ONE daily file,
weight/{YYYYMMDD}.parquet = (ukey Int32, weight Float64, group Int64): one
DAILY weight per name — the sample weight for training and the sizing
weight for portfolio construction. load_weights broadcasts it to every
decision instant of the task, so consumers still see one
(ukey, time, weight, group) row per instant.

eif_v11 layout (2026-10-04). The per-instant weight files (weight.{task}/)
and the separate universe file (group/) of eif_v10 are RETIRED; roots
with that layout are not resolved. v10 and v11 targets are built
differently, so studies on the two are not comparable.

All times below are seconds of the Asia/Shanghai day (grid entries print
as HH:MM or, off the whole minute, HH:MM:SS); timestamps in files are UTC
instants.
"""

import datetime
import zoneinfo
from pathlib import Path

import polars as pl

WEIGHT_TARGET_ROOT = Path("/data/CN-EQ/weight_target")
CALENDAR_FILE = Path("/data/CN-EQ/calendar.csv")
CST = zoneinfo.ZoneInfo("Asia/Shanghai")

TASKS = (0, 1)
TASK_SECONDS: dict[int, list[int]] = {
    0: [9 * 3600 + 24 * 60 + 59],
    1: [m * 60 for m in list(range(9 * 60 + 30, 11 * 60 + 26, 5))
        + list(range(13 * 60, 14 * 60 + 56, 5))],
}
RESPONSES = ["y_1", "y_3", "y_5", "r_1", "r_3", "r_5"]


def load_calendar() -> list[str]:
    return pl.read_csv(CALENDAR_FILE,
                       schema_overrides={"date": pl.String})["date"].to_list()


def latest_version_dir(task: int) -> Path:
    """The largest WEIGHT_TARGET_ROOT/eif_v{k} holding weight/,
    target.{task}/ and limit_mask/ — the default data root. Roots with
    the retired v10 layout (group/ + weight.{task}/) do not qualify."""
    best = None
    for p in WEIGHT_TARGET_ROOT.glob("eif_v*"):
        k = p.name[len("eif_v"):]
        if (k.isdigit() and (p / "weight").is_dir()
                and (p / f"target.{task}").is_dir()
                and (p / "limit_mask").is_dir()):
            if best is None or int(k) > int(best.name[len("eif_v"):]):
                best = p
    if best is None:
        raise RuntimeError(f"no eif_v* under {WEIGHT_TARGET_ROOT} with "
                           f"weight/, target.{task}/ and limit_mask/ "
                           "(eif_v11 layout)")
    return best


def fmt_time(sec: int) -> str:
    """Seconds of day -> 'HH:MM', or 'HH:MM:SS' off the whole minute."""
    h, m, s = sec // 3600, sec % 3600 // 60, sec % 60
    return f"{h:02d}:{m:02d}" + (f":{s:02d}" if s else "")


def parse_times(spec: str) -> list[int]:
    """'09:30,13:05[,09:24:59]' -> seconds of day, validated HH:MM[:SS]."""
    out = []
    for tok in spec.split(","):
        h, m, *rest = tok.strip().split(":")
        out.append(int(h) * 3600 + int(m) * 60 + (int(rest[0]) if rest else 0))
    return out


def eval_times(task: int, subset: str | None) -> list[int]:
    """The task's prediction times (seconds of day), optionally intersected
    with a user subset ('09:30,10:00,...'). Subset entries outside the
    task's grid are reported, not silently dropped."""
    base = TASK_SECONDS[task]
    if subset is None:
        return base
    want = parse_times(subset)
    outside = sorted(set(want) - set(base))
    if outside:
        pretty = ", ".join(fmt_time(t) for t in outside)
        raise ValueError(f"timepoints not in task {task}'s grid: {pretty}")
    return [t for t in base if t in set(want)]


def times_to_utc_ns(date: str, seconds: list[int]) -> list[int]:
    """Epoch-ns UTC instants of the given Asia/Shanghai seconds-of-day."""
    day = datetime.date(int(date[:4]), int(date[4:6]), int(date[6:]))
    return [int(datetime.datetime.combine(
        day, datetime.time(s // 3600, s % 3600 // 60, s % 60),
        tzinfo=CST).timestamp() * 1e9) for s in seconds]


def load_targets(date: str, task: int, eif_dir: Path,
                 responses: list[str]) -> pl.DataFrame:
    """(ukey Int32, time Datetime[ns, UTC], *responses) for one day."""
    df = pl.read_parquet(eif_dir / f"target.{task}" / f"{date}.parquet",
                         columns=["ukey", "timestamp"] + responses)
    return df.rename({"timestamp": "time"})


def load_weights(date: str, eif_dir: Path, task: int) -> pl.DataFrame:
    """(ukey Int32, time Datetime[ns, UTC], weight Float64, group Int64):
    the day's DAILY weight and group (weight/{date}) broadcast to
    each of the task's decision instants — the same weight at every
    instant of the day, by design (targets.py module docstring)."""
    w = pl.read_parquet(eif_dir / "weight" / f"{date}.parquet",
                        columns=["ukey", "weight", "group"])
    times = (pl.Series("time", times_to_utc_ns(date, TASK_SECONDS[task]),
                       dtype=pl.Int64)
             .cast(pl.Datetime("ns")).dt.replace_time_zone("UTC"))
    return (w.join(times.to_frame(), how="cross")
            .select("ukey", "time", "weight", "group"))


def target_dates(task: int, eif_dir: Path) -> list[str]:
    """Dates with BOTH a target.{task} and a weight/ file."""
    t = {p.stem for p in
         (eif_dir / f"target.{task}").glob("[0-9]" * 8 + ".parquet")}
    w = {p.stem for p in (eif_dir / "weight").glob("[0-9]" * 8 + ".parquet")}
    return sorted(t & w)


def load_limit_mask(date: str, eif_dir: Path) -> pl.DataFrame:
    """(ukey Int32, time Datetime[ns, UTC], lock Int8) for one day from
    eif_dir/limit_mask/ — the per-instant 涨停/跌停 flag at the 52 grid
    points + 09:24:59 (schema in the eif README): +1 = on the high limit
    (no ask: buys cannot fill), -1 = on the low limit (no bid: sells
    cannot fill), 0 = free, null = no price at the instant (suspended /
    untraded) or no limit record. Realized day-t data: a
    training/evaluation mask, never a feature."""
    return (pl.read_parquet(eif_dir / "limit_mask" / f"{date}.parquet")
            .rename({"timestamp": "time"}))


def join_limit_mask(w: pl.DataFrame, date: str, eif_dir: Path) -> pl.DataFrame:
    """W with the day's `lock` column (Int8) joined on (ukey, time); null
    where the instant has no mask row — no mark price, i.e. the name did
    not trade at or before that instant (suspended) or is absent from the
    bar data — which consumers treat as unfillable, like a locked row."""
    return w.join(load_limit_mask(date, eif_dir), on=["ukey", "time"],
                  how="left")
