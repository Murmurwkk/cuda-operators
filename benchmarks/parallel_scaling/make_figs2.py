"""Figures for the rewritten ParScale essay: exchange rate, prefill, ensemble."""

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA = Path(__file__).with_name('results')
OUT = Path(__file__).with_name('figures')

BLUE = "#2f5d8a"
ORANGE = "#c97b3c"
GREEN = "#5f8f5a"
RED = "#a64b4b"
GRAY = "#8a8a8a"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.7,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color="#e8e8e8", linewidth=0.7)
    ax.set_axisbelow(True)


def exchange():
    fit = json.loads((DATA / "fit.json").read_text())
    P = np.logspace(0, np.log10(256), 400)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
    for ax, key, title, color in [
        (axes[0], "STACK", "Stack-V2-Python (code)", BLUE),
        (axes[1], "PILE", "Pile (general text)", ORANGE),
    ]:
        d = fit[key.lower()]
        k, a = d["k"], d["alpha"]
        rho, a_th = d["theoretical"]["rho"], d["theoretical"]["alpha"]
        practical = 1 + k * np.log(P)
        theory = (P / ((P - 1) * rho + 1)) ** (1 / a_th)
        ax.plot(P, practical, color=color, lw=1.8, label="practical law")
        ax.plot(P, theory, color=color, lw=1.5, ls="--", label="correlation model")
        ax.axhline(2.77, color=GRAY, lw=0.9, ls=":")
        ax.text(1.05, 2.70, "4.4B / 1.6B", fontsize=7, color=GRAY, va="top")
        ax.axvline(8, color="#555555", lw=0.7, ls=(0, (1, 2)))
        ax.text(8.6, 1.03, "data ends", fontsize=7, color="#555555", rotation=90, va="bottom")
        ax.plot([8], [1.845 if key == "STACK" else 1.720], "o", ms=4, color=color)
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8, 16, 32, 64, 128, 256])
        ax.set_xticklabels(["1", "2", "4", "8", "16", "32", "64", "128", "256"])
        ax.set_xlim(1, 256)
        ax.set_ylim(1, 2.9)
        ax.set_xlabel("parallel streams $P$")
        ax.set_title(title, fontsize=9.5, pad=6)
        style(ax)
    axes[0].set_ylabel("parameter-equivalent multiplier")
    axes[0].annotate("78 streams", xy=(78, 2.77), xytext=(20, 2.35), fontsize=7.5,
                     color=BLUE, arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.8))
    axes[0].annotate("ceiling 1.92", xy=(200, 1.922), xytext=(90, 1.60), fontsize=7.5,
                     color=BLUE, arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.8))
    axes[0].legend(loc="lower right", fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "exchange-rate.png", dpi=200)
    plt.close(fig)


def prefill():
    p1 = json.loads((DATA / "prefill_p1.json").read_text())
    p8 = json.loads((DATA / "prefill_p8.json").read_text())
    x1 = [r["tokens"] for r in p1["rows"]]
    y1 = [r["median_ms"] for r in p1["rows"]]
    x8 = [r["tokens"] for r in p8["rows"]]
    y8 = [r["median_ms"] for r in p8["rows"]]
    ratio = [b / a for a, b in zip(y1, y8)]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    ax = axes[0]
    ax.plot(x1, y1, "o-", color=BLUE, ms=4, lw=1.5, label="$P=1$")
    ax.plot(x8, y8, "o-", color=ORANGE, ms=4, lw=1.5, label="$P=8$")
    ax.set_xscale("log", base=2)
    ax.set_xticks([128, 512, 1024, 2048])
    ax.set_xticklabels(["128", "512", "1k", "2k"])
    ax.set_xlabel("prompt tokens")
    ax.set_ylabel("full-forward latency (ms)")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    style(ax)

    ax = axes[1]
    ax.plot(x8, ratio, "o-", color=GREEN, ms=4, lw=1.5)
    ax.axhline(1, color=GRAY, lw=0.8, ls=":")
    ax.set_xscale("log", base=2)
    ax.set_xticks([128, 512, 1024, 2048])
    ax.set_xticklabels(["128", "512", "1k", "2k"])
    ax.set_ylim(1, 8.6)
    ax.set_xlabel("prompt tokens")
    ax.set_ylabel("$P=8$ / $P=1$ full-forward time")
    for xx, yy in zip(x8, ratio):
        ax.annotate(f"{yy:.1f}×", (xx, yy), textcoords="offset points", xytext=(3, 5), fontsize=7)
    style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "prefill.png", dpi=200)
    plt.close(fig)


def ensemble(json_path=None, out_dir=None):
    d = json.loads(Path(json_path or DATA / "ensemble_p8.json").read_text())
    out_dir = Path(out_dir or OUT)
    out_dir.mkdir(parents=True, exist_ok=True)
    ks = list(range(1, 9))
    mean = [d["subset_curve"][str(k)]["mean"] for k in ks]
    lo = [d["subset_curve"][str(k)]["min"] for k in ks]
    hi = [d["subset_curve"][str(k)]["max"] for k in ks]

    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    ax.fill_between(ks, lo, hi, color=BLUE, alpha=0.14, lw=0, label="range over subsets")
    ax.plot(ks, mean, "o-", color=BLUE, ms=4, lw=1.6, label="uniform average")
    ax.axhline(d["gate_nll"], color=ORANGE, lw=1.2, ls="--", label="learned gate")
    ax.axhline(d["oracle_nll"], color=RED, lw=1.2, ls=":", label="per-token oracle")
    ax.set_xticks(ks)
    ax.set_xlabel("streams in the average")
    ax.set_ylabel("mean NLL per token")
    ax.set_ylim(min(lo + [d["oracle_nll"]]) - 0.3, max(hi) + 0.4)
    ax.legend(frameon=False, fontsize=7.2, loc="upper right")
    ax.annotate(f"gate {d['gate_nll']:.2f}", xy=(8, d["gate_nll"]), xytext=(1.05, d["gate_nll"] - 0.22),
                fontsize=7, color=ORANGE)
    ax.annotate(f"oracle {d['oracle_nll']:.2f}", xy=(8, d["oracle_nll"]), xytext=(1.05, d["oracle_nll"] + 0.08),
                fontsize=7, color=RED)
    style(ax)
    fig.tight_layout()
    fig.savefig(out_dir / "ensemble.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["exchange", "prefill", "ensemble"])
    ap.add_argument("--ensemble-json", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--data-dir", type=Path, default=DATA)
    args = ap.parse_args()
    DATA = args.data_dir
    if args.out_dir:
        OUT = Path(args.out_dir)
    if args.only in (None, "exchange") or args.only == "prefill":
        OUT.mkdir(parents=True, exist_ok=True)
    if args.only in (None, "exchange"):
        exchange()
    if args.only in (None, "prefill"):
        prefill()
    if args.only in (None, "ensemble"):
        ensemble(args.ensemble_json, args.out_dir)
    print("done")
