"""Standard sherlock evaluation PDF report from an Evaluator eval.nc.

Per response (y_1/y_3/y_5), a page of seven metric tables and a
cumulative-spnl-curve page; then one spnl-vs-horizon plot across all
responses. The 15 rows/lines are the native side x group buckets, keyed
by (side = all/pos/neg, group = all/0/1/2/3). See evaluator.md "Report
guidance" for units and encoding.

Usage:
    python write_report.py --eval {work_dir}/eval.nc \
        --pdf report_{study}_{strategy}.pdf \
        [--fitter pred] [--title "..."]

e.g. eval.nc from ~/sherlock_work/astock_daily/long_short/ (the stitched
per-group models; study named after the feature dir, never the
weight/target version).
"""

import argparse
from itertools import product

import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--eval", required=True, help="path to an Evaluator eval.nc")
parser.add_argument("--pdf", required=True, help="output PDF path")
parser.add_argument("--fitter", default="pred", help="fitter to report on")
parser.add_argument("--title", default=None,
                    help="report title; defaults to eval path + fitter")
parser.add_argument("--trade", default="raw",
                    help="trade coord to report on, if the eval has one")
args = parser.parse_args()

ds = xr.open_dataset(args.eval).sel(fitter=args.fitter)
if "trade" in ds.dims:
    ds = ds.sel(trade=args.trade)

# The 15 native (side, group) buckets, every cell precomputed by the Evaluator.
BUCKETS = list(product(["all", "pos", "neg"], ["all", "0", "1", "2", "3"]))
LABEL = {b: f"({b[0]}, {b[1]})" for b in BUCKETS}

SCALE = {"sumw": 1, "stdy": 1e4, "stdx": 1e4, "corr": 100, "spnl": 1e4, "win": 100, "sharpe": 1}
FMT = {"sumw": "{:,.0f}", "stdy": "{:.1f}", "stdx": "{:.1f}",
       "corr": "{:.2f}", "spnl": "{:.2f}", "win": "{:.1f}", "sharpe": "{:.2f}"}
UNIT = {"sumw": "", "stdy": " (bps)", "stdx": " (bps)",
        "corr": " (%)", "spnl": " (bps)", "win": " (%)", "sharpe": ""}

RESPONSES = list(ds.response.values)
tables = {r: {} for r in RESPONSES}  # response -> {bucket label -> metric row}
curves, spnl_h = {}, {}
for b in BUCKETS:
    cell = ds.sel(side=b[0], group=b[1])
    for r in RESPONSES:
        row = {}
        for m in FMT:
            v = cell[m]
            row[m] = float(v.sel(response=r) if "response" in v.dims else v) * SCALE[m]
        tables[r][LABEL[b]] = row
    # (date x response) frame; one curve page is drawn per response.
    curves[b] = cell.curve.to_pandas() * 1e4
    spnl_h[b] = (cell.spnl * 1e4).to_pandas()
tables = {r: pd.DataFrame(tables[r]).T for r in RESPONSES}
# Horizon in days from the response name: "r01" -> 1, "y_3" -> 3, ...
horizons = [int("".join(c for c in r if c.isdigit())) for r in ds.response.values]

INK, MUTED = "#1a1a1a", "#666666"
RAMP = {
    "pos": ["#6f1414", "#a52222", "#d05050", "#eb9c94"],
    "neg": ["#143a5e", "#24598f", "#4f88c2", "#93bade"],
    "all": ["#14522a", "#2c8a3f", "#5cb46b", "#a0d3a8"],
}
STYLE = {("all", "all"): ("#111111", 2.6), ("pos", "all"): ("#c62828", 2.4),
         ("neg", "all"): ("#2b6cb0", 2.4)}
STYLE |= {(s, str(g)): (RAMP[s][g], 1.4) for s in RAMP for g in range(4)}


def polish(ax):
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color("#cccccc")
    ax.grid(True, color="#e6e6e6", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=8)


def lineplot(title, ylabel, xlabel, series, x=None):
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    for b in BUCKETS:
        c, lw = STYLE[b]
        v = series[b]
        ax.plot(v.index if x is None else x, v.to_numpy(), color=c, lw=lw,
                label=LABEL[b])
    polish(ax)
    ax.axhline(0, color="#bbbbbb", lw=0.8)
    ax.set_title(title, color=INK, fontsize=12, loc="left", pad=12)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False,
              fontsize=8, labelcolor=INK)
    fig.tight_layout(rect=(0, 0, 0.99, 1))
    return fig


figs = []
d0, d1 = pd.Timestamp(ds.date.values[0]), pd.Timestamp(ds.date.values[-1])
title = args.title or f"{args.eval} — fitter '{args.fitter}'"
for r in RESPONSES:
    fig, axes = plt.subplots(1, 7, figsize=(16.53, 8.27))
    fig.suptitle(f"{title} — response {r}, {d0:%Y-%m-%d} to {d1:%Y-%m-%d}",
                 x=0.02, ha="left", color=INK, fontsize=13)
    fig.subplots_adjust(left=0.06, right=0.98, wspace=1.1, top=0.86, bottom=0.06)
    for ax, metric in zip(axes, FMT):
        ax.axis("off")
        cells = [[FMT[metric].format(v)] for v in tables[r][metric]]
        tb = ax.table(cellText=cells, rowLabels=list(tables[r].index),
                      colLabels=[args.fitter], loc="center", cellLoc="right",
                      colWidths=[0.95])
        tb.auto_set_font_size(False)
        tb.set_fontsize(8.5)
        tb.scale(1.0, 1.65)
        for (row, c), cell in tb.get_celld().items():
            cell.set_edgecolor("#dddddd")
            cell.get_text().set_color(INK if row > 0 else MUTED)
        ax.set_title(metric + UNIT[metric], color=INK, fontsize=10)
    figs.append(fig)

for r in RESPONSES:
    figs.append(lineplot(
        f"Cumulative spnl curve by bucket (response {r})",
        "curve = cumsum(xy) / sqrt(xx x sw) (bps)", "date",
        {b: curves[b][r] for b in BUCKETS}))
figs.append(lineplot(
    f"spnl by horizon and bucket ({RESPONSES[0]}..{RESPONSES[-1]})",
    "spnl (bps)", "horizon (days)", spnl_h, x=horizons))

with PdfPages(args.pdf) as pdf:
    for fig in figs:
        pdf.savefig(fig)
print(f"wrote {args.pdf}")
for r in RESPONSES:
    print(f"--- response {r}")
    print(tables[r].round(3).to_string())
