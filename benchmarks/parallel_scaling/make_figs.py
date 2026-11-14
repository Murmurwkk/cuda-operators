"""Figures for the ParScale post: refit curves, iso-loss contours, serving cost,
and the aggregation probe."""
import json
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from fit_law import N, P, PILE, STACK

OUT = Path(__file__).with_name('results')
FIG = Path(__file__).with_name('figures')
FIT = None

BLUE = "#2f5d8a"
ORANGE = "#c97b3c"
GREEN = "#5f8f5a"
RED = "#a64b4b"
GREY = "#6b6b6b"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "legend.frameon": False,
    "figure.dpi": 200,
    "savefig.dpi": 200,
})


def fmt_b(x, _):
    return f"{x / 1e9:.1f}B"


def practical(res, n, p):
    return (res["A"] / (n * (1 + res["k"] * np.log(p)))) ** res["alpha"] + res["E"]


# ---------------------------------------------------------------- figure 1
def fig_fit():
    fig, axes = plt.subplots(1, 2, figsize=(7.02, 2.95), sharey=False)
    for ax, key, table, title in (
        (axes[0], "stack", STACK, "Stack-V2-Python (code)"),
        (axes[1], "pile", PILE, "Pile (general text)"),
    ):
        res = FIT[key]
        cmap = plt.get_cmap("Blues")
        for i, n in enumerate(N):
            c = cmap(0.42 + 0.55 * i / (len(N) - 1))
            pg = np.linspace(1, 8, 60)
            ax.plot(pg, practical(res, n, pg), color=c, lw=1.4, zorder=2)
            ax.scatter(P, table[:, i], s=14, facecolor="white", edgecolor=c,
                       linewidth=1.0, zorder=3)
            if i in (0, len(N) - 1):
                ax.annotate(fmt_b(n, 0), (8.15, table[-1, i]), color=c,
                            fontsize=8, va="center")
        ax.set_xscale("log", base=2)
        ax.set_xticks(P)
        ax.set_xticklabels([str(int(p)) for p in P])
        ax.set_xlim(0.9, 11)
        ax.set_xlabel("parallel streams $P$")
        ax.set_title(title, loc="left")
        ax.annotate(f"$k$={res['k']:.2f}, $R^2$={res['r2']:.4f}",
                    xy=(0.03, 0.03), xycoords="axes fraction", fontsize=8,
                    color="#333333")
        ax.grid(axis="y", color="#e4e4e4", lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("validation loss")
    fig.tight_layout(w_pad=2.0)
    fig.savefig(FIG / "loss-vs-p.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- figure 2
def fig_contours():
    res = FIT["stack"]
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    pgrid = np.linspace(0.8, 64, 300)
    ngrid = np.geomspace(7e8, 5.5e9, 300)
    PP, NN = np.meshgrid(pgrid, ngrid)
    LL = practical(res, NN, PP)
    levels = [0.99, 1.00, 1.02, 1.04, 1.06, 1.08, 1.10]
    ax.contourf(PP, NN, LL, levels=levels, cmap="Blues_r", alpha=0.28)
    ax.contour(PP, NN, LL, levels=levels[1:-1], colors=BLUE, linewidths=0.6,
               alpha=0.5)
    pts = [
        (1, 1.571e9, "1.6B, P=1\nloss 1.0817", ORANGE, "o", (1.15, 1.30e9)),
        (8, 1.5957e9, "1.6B, P=8\nloss 1.0383", GREEN, "s", (9.4, 1.52e9)),
        (1, 4.353e9, "4.4B, P=1\nloss 1.0213", RED, "^", (1.15, 3.95e9)),
    ]
    for p, n, label, c, m, off in pts:
        ax.scatter([p], [n], s=48, color=c, marker=m, zorder=4, edgecolor="white",
                   linewidth=0.8)
        ax.annotate(label, (p, n), xytext=off, textcoords="data", fontsize=7.2,
                    color=c, ha="left", va="center",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none",
                              alpha=0.75))
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32, 64])
    ax.set_xticklabels(["1", "2", "4", "8", "16", "32", "64"])
    ax.set_xlabel("parallel streams $P$")
    ax.set_ylabel("non-embedding parameters")
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_b))
    ax.set_xlim(0.75, 64)
    ax.set_ylim(7e8, 5.5e9)
    ax.grid(color="#ececec", lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIG / "iso-loss.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- figure 3
def fig_serving():
    lat1 = json.loads((OUT / "decode_p1.json").read_text())["latency"]
    lat8 = json.loads((OUT / "decode_p8.json").read_text())["latency"]
    bs = [1, 2, 4, 8]
    fig, axes = plt.subplots(1, 2, figsize=(7.02, 2.7))
    for lat, c, label in ((lat1, BLUE, "P=1"), (lat8, ORANGE, "P=8")):
        ys = [lat[str(b)]["summary"]["cached_decode_tokens_per_second"] for b in bs]
        lo = [b * 1000 / lat[str(b)]["summary"]["decode_step_range_ms"][1] for b in bs]
        hi = [b * 1000 / lat[str(b)]["summary"]["decode_step_range_ms"][0] for b in bs]
        axes[0].errorbar(bs, ys, yerr=[[y-l for y,l in zip(ys,lo)], [h-y for y,h in zip(ys,hi)]],
                         fmt="o-", color=c, lw=1.4, ms=4, capsize=3, label=label)
        axes[1].plot(bs, [lat[str(b)]["summary"]["peak_allocated_gb"] for b in bs], "o-", color=c,
                     lw=1.4, ms=4, label=label)
    axes[0].set_ylabel("cached-decode tokens per second")
    axes[1].set_ylabel("peak GPU memory (GB)")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(bs)
        ax.set_xticklabels([str(b) for b in bs])
        ax.set_xlabel("batch size")
        ax.grid(color="#ececec", lw=0.6)
        ax.set_axisbelow(True)
    axes[0].legend(loc="upper left")
    fig.tight_layout(w_pad=2.0)
    fig.savefig(FIG / "serving-cost.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- figure 4
def fig_probe():
    pr = json.loads((OUT / "probe_p8.json").read_text())["streams"]
    nll = pr["nll_per_stream"]
    w = pr["weight_mean_per_stream"]
    n = len(nll)
    fig, axes = plt.subplots(1, 2, figsize=(7.02, 2.6))
    axes[0].bar(range(1, n + 1), nll, color=BLUE, alpha=0.85, width=0.7)
    axes[0].axhline(pr["agg_logits_self_nll"], color=ORANGE, lw=1.2,
                    label=f"aggregate {pr['agg_logits_self_nll']:.2f}")
    axes[0].axhline(pr["nll_uniform_weights"], color=GREY, lw=1.0, ls="--",
                    label=f"plain average {pr['nll_uniform_weights']:.2f}")
    axes[0].set_xlabel("stream")
    axes[0].set_ylabel("NLL on 204 tokens")
    axes[0].set_ylim(3.0, 6.9)
    axes[0].legend(loc="upper left", fontsize=7.5)
    axes[1].bar(range(1, n + 1), w, color=GREEN, alpha=0.85, width=0.7)
    axes[1].axhline(1 / n, color=GREY, lw=1.0, ls="--", label="uniform 1/8")
    axes[1].set_xlabel("stream")
    axes[1].set_ylabel("mean gate weight")
    axes[1].set_ylim(0, 0.2)
    axes[1].legend(loc="upper right", fontsize=7.5)
    axes[1].annotate(f"entropy {pr['weight_entropy']:.2f} / {np.log(n):.2f}",
                     xy=(0.03, 0.86), xycoords="axes fraction", fontsize=7.5,
                     color="#333333")
    for ax in axes:
        ax.grid(axis="y", color="#ececec", lw=0.6)
        ax.set_axisbelow(True)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(FIG / "streams.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=OUT)
    parser.add_argument('--out-dir', type=Path, default=FIG)
    parser.add_argument('--only', choices=['fit', 'contours', 'serving', 'probe'])
    args = parser.parse_args()
    OUT, FIG = args.data_dir, args.out_dir
    FIG.mkdir(parents=True, exist_ok=True)
    FIT = json.loads((OUT / 'fit.json').read_text())
    for name, function in [('fit', fig_fit), ('contours', fig_contours), ('serving', fig_serving), ('probe', fig_probe)]:
        if args.only in (None, name):
            function()
    print("wrote", sorted(p.name for p in FIG.glob("*.png")))
