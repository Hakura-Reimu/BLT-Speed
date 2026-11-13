"""Figures for REPORT.md (static PNG, light surface). Reads the mirrored results under results/.
usage: python scripts/diag/report_figures.py [RESULTS_DIR]
Palette: the documented reference palette slots 1-4 (dataviz skill references/palette.md); slots 1-3 are validated
for all pairs (scatter), 1-4 for adjacent pairs (lines). Identity is also carried by marker shape and direct labels."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
OUT = R / "figures"
OUT.mkdir(exist_ok=True)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "font.size": 10, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
                     "axes.spines.top": False, "axes.spines.right": False})


def bpb(run: str, milestone: str = "3000000000") -> float:
    return json.loads((R / "main-scores" / f"{run}-bytes-{milestone}.json").read_text())["bpb"]


def solo_speed() -> dict[tuple[str, int], float]:
    data = json.loads((R / "final-timing" / "summary-end_to_end.json").read_text())
    speeds = {}
    for name, cells in data["solo"].items():
        for key, cell in cells.items():
            kib, variant = key.split("/")
            if variant == "end_to_end":
                c = cell["combined"]
                speeds[(name.removeprefix("solo-"), int(kib))] = tuple(c[k] / 1000 for k in ("value", "ci95_low", "ci95_high"))  # kB/s
    return speeds


def frontier() -> None:
    speeds = solo_speed()
    models = {
        "xs-bpe50k": ("XS-BPE50K*", AQUA, "^"), "xs-bpe16k": ("XS-BPE16K*", AQUA, "^"),
        "s-bpe50k": ("S-BPE50K", BLUE, "o"), "s-bpe16k": ("S-BPE16K", BLUE, "o"),
        "m-bpe50k": ("M-BPE50K", BLUE, "o"), "m-bpe16k": ("M-BPE16K", BLUE, "o"),
        "s-p6": ("S-P6", ORANGE, "s"), "s-p8": ("S-P8", ORANGE, "s"),
        "m-p6": ("M-P6", ORANGE, "s"), "m-p8": ("M-P8", ORANGE, "s"), "m-p6-nohash": ("M-P6-nohash", ORANGE, "s"),
    }
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=150)
    best = bpb("m-bpe16k-seed0")
    ax.axhspan(best - 0.03, best + 0.03, color=BLUE, alpha=0.08, lw=0)
    ax.text(2.6, best - 0.022, "M-BPE16K +/-0.03 BPB (iso-quality band)", color=INK2, fontsize=8, ha="left", va="center")
    points = []
    # (x factor, y offset in BPB, alignment); the y axis is inverted, so a negative offset draws above the point.
    offsets = {"M-P6": (0.95, 0.0, "right"), "M-P6-nohash": (1.06, -0.016, "left"), "S-BPE16K": (0.95, 0.012, "right"),
               "XS-BPE16K*": (1.05, 0.013, "left"), "XS-BPE50K*": (1.05, -0.012, "left"),
               "M-BPE50K": (1.04, 0.012, "left")}
    for run, (label, color, marker) in models.items():
        value, low, high = (v / 1000 for v in speeds[(run, 8)])
        x, y = value, bpb(f"{run}-seed0")
        points.append((x, y, label))
        ax.errorbar([x], [y], xerr=[[x - low], [high - x]], fmt="none", ecolor=color, elinewidth=1.2, capsize=0, zorder=2)
        ax.scatter([x], [y], s=46, color=color, marker=marker, edgecolors=SURFACE, linewidths=1.5, zorder=3)
        fx, dy, ha = offsets.get(label, (1.06, 0.0, "left"))
        ax.annotate(label, (x, y), xytext=(x * fx, y + dy), fontsize=8, color=INK, ha=ha, va="center")
    pareto, best_y = [], float("inf")
    for x, y, _ in sorted(points, key=lambda p: -p[0]):
        if y < best_y:
            pareto.append((x, y)); best_y = y
    pareto.sort()
    ax.plot([p[0] for p in pareto], [p[1] for p in pareto], color=INK2, lw=1.2, ls="--", zorder=2)
    ax.set_xscale("log")
    ax.set_xticks([0.8, 1, 2, 3, 4, 5]); ax.set_xticklabels(["0.8", "1", "2", "3", "4", "5"])
    ax.set_xlabel("8 KiB prefill throughput, MB/s (batch 1, end-to-end; higher is faster)")
    ax.set_ylabel("validation BPB at 3 GB (lower is better)")
    ax.invert_yaxis()
    handles = [plt.Line2D([], [], color=c, marker=m, ls="", markersize=7, label=l) for l, c, m in
               (("BPE (pre-registered)", BLUE, "o"), ("byte-latent", ORANGE, "s"), ("BPE XS (exploratory)*", AQUA, "^"))]
    handles.append(plt.Line2D([], [], color=INK2, ls="--", label="observed-point frontier (guide)"))
    ax.legend(handles=handles, loc="lower right", fontsize=8, frameon=False)
    ax.set_title("Quality vs prefill speed (seed 0, 3 GB; bars: 95% CI of throughput)", color=INK, fontsize=11, loc="left")
    fig.tight_layout(); fig.savefig(OUT / "frontier_8k.png"); plt.close(fig)


def length_curve() -> None:
    """Read the ratios from the timing summary: 8/16 KiB are three-session combined estimates (filled markers, with
    their 95% intervals); 2/4/32 KiB are one sweep (hollow markers, prompt-bootstrap intervals)."""
    data = json.loads((R / "final-timing" / "summary-end_to_end.json").read_text())
    series = {"M-P6 / M-BPE16K": ("m-p6--m-bpe16k", ORANGE, "s"), "S-P6 / S-BPE50K": ("s-p6--s-bpe50k", BLUE, "o"),
              "M-P6 / XS-BPE50K*": ("m-p6--xs-bpe50k", AQUA, "^")}
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
    ax.axhline(1.0, color=INK2, lw=1)
    ax.axhline(1.2, color=INK2, lw=1, ls=":")
    ax.text(2.05, 1.215, "1.20x speed threshold (the quality condition also failed)", fontsize=8, color=INK2, va="bottom")
    ax.text(2.05, 1.015, "equal speed", fontsize=8, color=INK2, va="bottom")
    for label, (pair, color, marker) in series.items():
        points = []
        for kib in (8, 16):
            c = data["pairs"][pair][f"{kib}/end_to_end"]["combined"]
            points.append((kib, c["value"], c["ci95_low"], c["ci95_high"], True))
        for key, cell in data["curve"].get(pair, {}).items():
            kib, variant = key.split("/")
            if variant == "end_to_end":
                b = cell["prompt_bootstrap"]
                points.append((int(kib), b["speed_a_over_b"], b["ci95_low"], b["ci95_high"], False))
        points.sort()
        xs, ys = [p[0] for p in points], [p[1] for p in points]
        ax.plot(xs, ys, color=color, lw=2, zorder=2)
        for kib, y, low, high, combined in points:
            ax.errorbar([kib], [y], yerr=[[y - low], [high - y]], fmt="none", ecolor=color, elinewidth=1.2, zorder=2)
            ax.scatter([kib], [y], s=46, marker=marker, zorder=3, linewidths=1.5,
                       color=color if combined else SURFACE, edgecolors=color if not combined else SURFACE)
        ax.annotate(label, (xs[-1], ys[-1]), xytext=(8, 0), textcoords="offset points", fontsize=8, va="center")
    handles = [plt.Line2D([], [], color=INK2, marker="o", ls="", markersize=7, label="three sessions (8, 16 KiB)"),
               plt.Line2D([], [], color=INK2, marker="o", ls="", markersize=7, markerfacecolor=SURFACE,
                          label="one sweep (2, 4, 32 KiB)")]
    ax.legend(handles=handles, loc="center right", fontsize=8, frameon=False)
    ax.set_xscale("log", base=2); ax.set_xticks([2, 4, 8, 16, 32]); ax.set_xticklabels(["2", "4", "8", "16", "32"])
    ax.set_xlim(1.8, 64)
    ax.set_xlabel("prompt length, KiB")
    ax.set_ylabel("prefill speed ratio, byte / BPE (>1: byte faster)")
    ax.set_title("Prefill speed ratio vs prompt length (batch 1, 95% CI)", color=INK, fontsize=11, loc="left")
    fig.tight_layout(); fig.savefig(OUT / "prefill_length.png"); plt.close(fig)


def training_curve() -> None:
    gb = [0.375, 0.75, 1.5, 3.0]
    names = {"M-BPE16K": ("m-bpe16k-seed0", BLUE, "o"), "S-BPE50K": ("s-bpe50k-seed0", AQUA, "o"),
             "M-P6": ("m-p6-seed0", ORANGE, "s"), "S-P6": ("s-p6-seed0", YELLOW, "s")}
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
    for label, (run, color, marker) in names.items():
        ys = [bpb(run, str(int(g * 1e9))) for g in gb]
        ax.plot(gb, ys, color=color, lw=2, marker=marker, markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.annotate(f"{label}  {ys[-1]:.3f}", (gb[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                    fontsize=8, va="center")
    ax.set_xscale("log", base=2); ax.set_xticks(gb); ax.set_xticklabels(["0.375", "0.75", "1.5", "3"])
    ax.set_xlim(0.33, 6)
    ax.set_xlabel("training bytes, GB")
    ax.set_ylabel("validation BPB (lower is better)")
    ax.set_title("Quality vs training data (seed 0)", color=INK, fontsize=11, loc="left")
    fig.tight_layout(); fig.savefig(OUT / "training_curve.png"); plt.close(fig)


if __name__ == "__main__":
    frontier(); length_curve(); training_curve()
    print("written", sorted(p.name for p in OUT.iterdir()))
