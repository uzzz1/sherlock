"""Factor-importance and concentration report from the rolling checkpoints.

For each group's ckpt/{month}.json booster, per-factor importance is the
booster's total gain, normalized to shares summing to 1 — what the deployed
model actually used in that 24-month window, at zero extra compute. The
(month, group, factor) share panel is saved to {work_dir}/{study}/{strategy}/
importance.nc and summarized in a PDF:

* per group: a stacked area of the top factors' shares over time, the
  concentration series (effective number of factors N_eff = 1/sum(p^2) and
  top-K share), and the turnover series (Spearman rank correlation and
  top-K Jaccard overlap between consecutive checkpoints);
* a summary table comparing, per group, the mean within-window N_eff to the
  N_eff of the time-averaged shares ("pooled"). Pooled >> within means
  importance rotates across factors over time; pooled ~ within means the
  same few factors dominate every window.

Consecutive windows overlap LOOK_BACK - CKPT_FREQ months, so consecutive
checkpoints are mechanically correlated; the table's "rho @ lag L" column is
the rank correlation at the first non-overlapping lag (L = ceil(LOOK_BACK /
CKPT_FREQ) checkpoints) — judge persistence against that, not against the
consecutive number. Gain splits credit arbitrarily among correlated factors,
so read individual rankings loosely; the concentration metrics are robust to
this.

Usage
-----
    python factor_importance.py --study NAME --work-dir ROOT
        [--group 0 1 2 3] [--strategy long_short] [--top 10] [--stack 8]
        [--pdf report_importance_{study}_{strategy}.pdf]
"""

import argparse
import os

import numpy as np
import pandas as pd
import xarray as xr
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from trainer import Trainer

# Categorical palette (validated reference order) + neutral for "Other";
# recessive text/grid inks.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
OTHER = "#b7b5ad"
TEXT, SUBTEXT, GRID = "#0b0b0b", "#52514e", "#e3e2dd"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--group", type=int, nargs="+", default=[0, 1, 2, 3],
                    choices=[0, 1, 2, 3],
                    help="groups whose checkpoints are analyzed")
parser.add_argument("--strategy", default="long_short",
                    choices=("long_short", "long_only", "short_only"),
                    help="execution strategy whose work dirs to analyze")
parser.add_argument("--study", default="baseline",
                    help="study name of the trainer run (names the work dirs)")
parser.add_argument("--work-dir", required=True,
                    help="root of the trainer's work dirs")
parser.add_argument("--top", type=int, default=10,
                    help="K for top-K share and top-K turnover")
parser.add_argument("--stack", type=int, default=8,
                    help="factors shown individually in the stacked area")
parser.add_argument("--pdf", default=None,
                    help="output PDF (default report_importance_{study}_{strategy}.pdf)")
args = parser.parse_args()

STUDY = args.study
WORK_ROOT = f"{args.work_dir}/{STUDY}"
PDF = args.pdf or f"report_importance_{STUDY}_{args.strategy}.pdf"
# first checkpoint lag with no training-window overlap
LAG = -(-Trainer.LOOK_BACK // Trainer.CKPT_FREQ)


def gain_shares(ckpt_dir: str) -> pd.DataFrame:
    """(month x factor) frame of total-gain shares, rows summing to 1."""
    months, rows, names = [], [], None
    for f in sorted(os.listdir(ckpt_dir)):
        if not f.endswith(".json"):
            continue
        b = xgb.Booster()
        b.load_model(f"{ckpt_dir}/{f}")
        if names is None:
            names = b.feature_names
        elif b.feature_names != names:
            raise RuntimeError(f"{ckpt_dir}/{f}: feature names differ")
        g = b.get_score(importance_type="total_gain")
        s = pd.Series(g, dtype="float64").reindex(names).fillna(0.0)
        rows.append(s / s.sum())
        months.append(f[:-5])
    return pd.DataFrame(rows, index=pd.Index(months, name="month"))


def n_eff(p: pd.DataFrame) -> pd.Series:
    """Effective number of factors 1/HHI per row."""
    return 1.0 / (p ** 2).sum(axis=1)


def rank_corr(p: pd.DataFrame, lag: int) -> pd.Series:
    """Spearman correlation of importance vectors lag checkpoints apart."""
    r = p.rank(axis=1)
    out = {}
    for i in range(lag, len(r)):
        a, b = r.iloc[i - lag], r.iloc[i]
        out[p.index[i]] = np.corrcoef(a, b)[0, 1]
    return pd.Series(out, dtype="float64")


def topk_sets(p: pd.DataFrame, k: int) -> list[frozenset]:
    return [frozenset(row.nlargest(k).index) for _, row in p.iterrows()]


def style(ax, title):
    ax.set_title(title, color=TEXT, fontsize=9, loc="left")
    ax.tick_params(colors=SUBTEXT, labelsize=7)
    ax.grid(color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


shares = {g: gain_shares(f"{WORK_ROOT}/group{g}_{args.strategy}/ckpt")
          for g in args.group}

# aligned (month, group, factor) panel -> importance.nc beside eval.nc
months = sorted(set().union(*(s.index for s in shares.values())))
factors = list(next(iter(shares.values())).columns)
panel = xr.Dataset(
    {"share": (("month", "group", "factor"),
               np.stack([shares[g].reindex(months).to_numpy()
                         for g in args.group], axis=1))},
    coords={"month": months, "group": args.group, "factor": factors},
    attrs={"importance": "total_gain share", "study": STUDY,
           "strategy": args.strategy, "top_k": args.top},
)
os.makedirs(f"{WORK_ROOT}/{args.strategy}", exist_ok=True)
panel.to_netcdf(f"{WORK_ROOT}/{args.strategy}/importance.nc")

summary = []
with PdfPages(PDF) as pdf:
    for g in args.group:
        p = shares[g]
        t = pd.to_datetime(p.index, format="%Y%m")
        pooled = p.mean(axis=0)
        within = n_eff(p)
        topk = p.apply(lambda r: r.nlargest(args.top).sum(), axis=1)
        rho1, rhoL = rank_corr(p, 1), rank_corr(p, LAG)
        sets = topk_sets(p, args.top)
        jac = pd.Series(
            [len(a & b) / len(a | b) for a, b in zip(sets, sets[1:])],
            index=p.index[1:], dtype="float64")
        summary.append({
            "group": g, "factors": int((p.sum() > 0).sum()),
            "N_eff within": within.mean(),
            "N_eff pooled": 1.0 / (pooled ** 2).sum(),
            "rotation": (1.0 / (pooled ** 2).sum()) / within.mean(),
            f"top-{args.top} share": topk.mean(),
            "rho consec": rho1.mean(),
            f"rho lag {LAG}": rhoL.mean(),
            f"top-{args.top} Jaccard": jac.mean(),
        })

        fig = plt.figure(figsize=(11.69, 8.27))
        fig.suptitle(f"{STUDY} {args.strategy} — factor importance, group {g}",
                     color=TEXT, fontsize=11, x=0.06, ha="left")
        gs = fig.add_gridspec(2, 3, height_ratios=[1.6, 1],
                              hspace=0.35, wspace=0.3,
                              left=0.06, right=0.97, top=0.9, bottom=0.08)

        # stacked area of the top --stack factors at their own scale (the
        # remainder is far larger but structureless; its share is in the
        # title so the scale isn't mistaken for the whole)
        ax = fig.add_subplot(gs[0, :])
        top = list(pooled.nlargest(args.stack).index)
        stack = p[top]
        ax.stackplot(t, stack.to_numpy().T, labels=stack.columns,
                     colors=PALETTE[:len(top)],
                     edgecolor="white", linewidth=0.5)
        ax.set_ylim(0, stack.sum(axis=1).max() * 1.15)
        style(ax, f"importance share — top {args.stack} factors by pooled "
                  f"share (their combined share averages "
                  f"{stack.sum(axis=1).mean():.0%} of the total)")
        ax.legend(loc="center left", bbox_to_anchor=(1.001, 0.5), fontsize=7,
                  frameon=False, labelcolor=SUBTEXT)

        ax = fig.add_subplot(gs[1, 0])
        ax.plot(t, within, color=PALETTE[0], linewidth=2)
        style(ax, "effective number of factors (1/HHI)")

        ax = fig.add_subplot(gs[1, 1])
        ax.plot(t, topk, color=PALETTE[1], linewidth=2)
        ax.set_ylim(0, topk.max() * 1.3)
        style(ax, f"top-{args.top} share")

        ax = fig.add_subplot(gs[1, 2])
        ax.plot(t[1:], rho1, color=PALETTE[2], linewidth=2, label="Spearman ρ")
        ax.plot(t[1:], jac, color=PALETTE[6], linewidth=2,
                label=f"top-{args.top} Jaccard")
        ax.set_ylim(0, 1)
        style(ax, "turnover vs previous checkpoint")
        ax.legend(fontsize=7, frameon=False, labelcolor=SUBTEXT)
        pdf.savefig(fig)
        plt.close(fig)

    # summary table page
    tab = pd.DataFrame(summary).set_index("group")
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.axis("off")
    fig.suptitle(f"{STUDY} {args.strategy} — concentration summary "
                 f"({months[0]}..{months[-1]}, {len(months)} checkpoints)",
                 color=TEXT, fontsize=11, x=0.06, ha="left")
    cell = tab.map(lambda v: f"{v:.2f}" if isinstance(v, float) else str(v))
    mt = ax.table(cellText=cell.to_numpy(), rowLabels=[f"group {i}" for i in tab.index],
                  colLabels=tab.columns, loc="upper center", cellLoc="center")
    mt.auto_set_font_size(False)
    mt.set_fontsize(8)
    mt.scale(1, 1.6)
    for c in mt.get_celld().values():
        c.set_edgecolor(GRID)
    ax.text(0.0, 0.62, "\n".join([
        "N_eff = 1/sum(share^2): the model behaves as if using that many factors.",
        "'within' averages per-checkpoint N_eff; 'pooled' is the N_eff of the time-averaged shares.",
        "rotation = pooled/within: ~1 means the same factors dominate every window; >1 means importance rotates over time.",
        f"rho = Spearman correlation of importance vectors between checkpoints; training windows overlap until lag {LAG},",
        f"so judge persistence by 'rho lag {LAG}', not the consecutive value.",
    ]), transform=ax.transAxes, fontsize=8, color=SUBTEXT, va="top")
    pdf.savefig(fig)
    plt.close(fig)

print(tab.to_string(float_format=lambda v: f"{v:.2f}"))
print(f"\nwrote {WORK_ROOT}/{args.strategy}/importance.nc and {PDF}")
