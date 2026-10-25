"""Float16 storage for the trainer's day cache (the default; `trainer.py --x-cache-dtype float32` opts out).

A float16 keeps ~3 significant digits relative to the value, so a factor
that sits far from zero (a [-1, 1] signal with a stray +1000) collapses to a
handful of levels. The codec therefore stores each factor as
`(x - center) / scale` in float16 and keeps the float32 center/scale beside
it; `decode` returns `x` itself, so the model still sees the original factor
(level included) to ~5e-4 of the factor's own spread. Nothing is demeaned for
the model; only the storage is shifted.

center/scale are the median and IQR (std, then 1, as fallbacks), robust to
the outliers that make mean/std useless. Per factor, `fit` picks the first
encoding whose hist bins barely move on a sample of days:

  global  one (center, scale) for the whole study. Equal values stay equal on
          every day, which a discrete factor needs.
  day     the day's own (center, scale), for a level that drifts far more
          than the cross-section is wide. Rounding then differs per day, so
          it breaks ties; it is only taken when `global` fails.
  f32     kept float32 when neither passes.

"Bins move" = the share of sample rows whose 256-quantile hist bin (as
xgboost's hist would cut it) differs by 2 or more between the original and the
decoded values (see _bins_moved). Values beyond +-CLIP scales saturate: they
stay in the top/bottom bin, which is all hist can tell them apart by anyway.
"""

import numpy as np
import polars as pl

ID_COLS = ("ukey", "time")
CLIP = 6e4          # float16 tops out at 65504
BIN_TOL = 0.01      # max share of rows whose hist bin may move by 2+
SAMPLE_ROWS = 200_000   # rows used per factor for the check, across the sample days
GLOBAL, DAY, F32 = "global", "day", "f32"


def _center_scale(a: np.ndarray) -> tuple[float, float]:
    center = float(np.median(a))
    q1, q3 = np.quantile(a, [0.25, 0.75])
    scale = float(q3 - q1)
    if not scale > 0:
        scale = float(a.std())
    if not scale > 0:
        scale = 1.0
    return center, scale


def _roundtrip(a: np.ndarray, center: float, scale: float) -> np.ndarray:
    z = np.clip((a - center) / scale, -CLIP, CLIP).astype(np.float16)
    return z.astype(np.float64) * scale + center


def _bins(a: np.ndarray) -> np.ndarray:
    return np.searchsorted(np.quantile(a, np.linspace(0, 1, 257)[1:-1]), a)


def _bins_moved(a: np.ndarray, b: np.ndarray) -> float:
    """Share of rows whose bin moves by 2+. A row flipping to the neighbouring
    bin is just rounding noise at a bin edge (~1-2% of rows even for a clean
    factor); lost resolution (levels collapsing, ties splitting) jumps further."""
    return float((np.abs(_bins(a) - _bins(b)) >= 2).mean())


class Float16Codec:
    def __init__(self, columns: list[str], modes: list[str], stats: list[tuple[float, float]]):
        self.columns, self.modes, self.stats = columns, modes, stats   # stats: global (center, scale)

    @classmethod
    def fit(cls, frames: list[pl.DataFrame], seed: int = 0) -> "Float16Codec":
        """Choose an encoding per factor from sample days (frames: ukey, time, *factors)."""
        columns = [c for c in frames[0].columns if c not in ID_COLS]
        rng = np.random.default_rng(seed)
        per_day = max(1, SAMPLE_ROWS // len(frames))
        days: dict[str, list[np.ndarray]] = {c: [] for c in columns}
        for df in frames:
            for c in columns:
                a = df[c].cast(pl.Float64).to_numpy()
                a = a[np.isfinite(a)]
                if len(a) > per_day:
                    a = rng.choice(a, per_day, replace=False)
                days[c].append(a)
        modes, stats = [], []
        for c in columns:
            pooled = np.concatenate(days[c])
            if len(pooled) == 0:
                modes.append(GLOBAL)
                stats.append((0.0, 1.0))
                continue
            center, scale = _center_scale(pooled)
            stats.append((center, scale))
            if _bins_moved(pooled, _roundtrip(pooled, center, scale)) <= BIN_TOL:
                modes.append(GLOBAL)
                continue
            by_day = np.concatenate([_roundtrip(a, *_center_scale(a)) if len(a) else a for a in days[c]])
            modes.append(DAY if _bins_moved(pooled, by_day) <= BIN_TOL else F32)
        return cls(columns, modes, stats)

    def summary(self) -> str:
        n = {m: self.modes.count(m) for m in (GLOBAL, DAY, F32)}
        return (f"float16 x cache over {len(self.columns)} factors: {n[GLOBAL]} global, "
                f"{n[DAY]} per-day, {n[F32]} kept float32")

    def encode(self, df: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, tuple[float, float]]]:
        """One day's frame -> (float16 frame, {factor: (center, scale)} to decode it)."""
        day_cols = [c for c, m in zip(self.columns, self.modes) if m == DAY]
        day_stats: dict[str, tuple[float, float]] = {}
        if day_cols:
            row = df.select(*[
                e for c in day_cols
                for e in (pl.col(c).filter(pl.col(c).is_finite()).median().alias(f"{c}\0m"),
                          pl.col(c).filter(pl.col(c).is_finite()).quantile(0.25).alias(f"{c}\0a"),
                          pl.col(c).filter(pl.col(c).is_finite()).quantile(0.75).alias(f"{c}\0b"),
                          pl.col(c).filter(pl.col(c).is_finite()).std().alias(f"{c}\0s"))]).row(0)
            for c, (m, q1, q3, sd) in zip(day_cols, zip(*[iter(row)] * 4)):
                scale = (q3 - q1) if (q3 is not None and q3 - q1 > 0) else (sd if sd and sd > 0 else 1.0)
                day_stats[c] = (float(m if m is not None else 0.0), float(scale))
        exprs, used = [], {}
        for c, mode, glob in zip(self.columns, self.modes, self.stats):
            if mode == F32:
                exprs.append(pl.col(c).cast(pl.Float32))
                continue
            center, scale = day_stats[c] if mode == DAY else glob
            used[c] = (center, scale)
            exprs.append(((pl.col(c) - center) / scale).clip(-CLIP, CLIP).cast(pl.Float16))
        return df.select(*ID_COLS, *exprs), used

    @staticmethod
    def decode(df: pl.DataFrame, used: dict[str, tuple[float, float]]) -> pl.DataFrame:
        """Inverse of encode: float32 frame holding the original factor values."""
        return df.with_columns(
            pl.col(c).cast(pl.Float32) * pl.lit(s, dtype=pl.Float32) + pl.lit(m, dtype=pl.Float32)
            for c, (m, s) in used.items())
