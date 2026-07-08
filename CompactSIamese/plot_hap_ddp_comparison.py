
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

plt.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})

# ── baselines (Experiment 2, FIVR-50K) ───────────────────────────────────────
BASELINES = {
    "TMK":            {"online": 16.5797, "fbeta": 0.3243, "color": "#4C72B0"},
    "HAP-VR":         {"online": 67.610,  "fbeta": 0.8103, "color": "#DD8452"},
    "LINE-VSC":       {"online": 17.29,   "fbeta": 0.8508, "color": "#55A868"},
    "VVS":            {"online":  0.4971, "fbeta": 0.6979, "color": "#C44E52"},
    "CompactSiamese": {"online": 20.2382, "fbeta": 0.1167, "color": "#8172B2"},
}

# ── HAP-DPP at k_div=2 (Experiment 5, FIVR-50K) ─────────────────────────────
HAP_DDP = {"online": 43.175, "fbeta": 0.8051}
HAP_DDP_COLOR = "#DD8452"   # same orange as HAP-VR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=str, default=None)
    args = parser.parse_args()
    save_dir = Path(args.save_dir) if args.save_dir else None

    fig, ax = plt.subplots(figsize=(8, 5))

    # ── baselines ─────────────────────────────────────────────────────────────
    for name, d in BASELINES.items():
        ax.scatter(d["online"], d["fbeta"], s=180, color=d["color"],
                   alpha=0.85, edgecolor="white", linewidth=1.2, zorder=4)
        offset = (8, 6)
        if name == "HAP-VR":
            offset = (8, -16)
        elif name == "LINE-VSC":
            offset = (-90, 6)
        ax.annotate(name, (d["online"], d["fbeta"]),
                    textcoords="offset points", xytext=offset, fontsize=9)

    # ── HAP-DPP ───────────────────────────────────────────────────────────────
    ax.scatter(HAP_DDP["online"], HAP_DDP["fbeta"], s=180, color="white",
               edgecolor=HAP_DDP_COLOR, linewidth=2.0, marker="^", zorder=5)
    ax.annotate("HAP-DPP", (HAP_DDP["online"], HAP_DDP["fbeta"]),
                textcoords="offset points", xytext=(-75, 6),
                fontsize=9, color="#994400",
                arrowprops=dict(arrowstyle="-", lw=0.8,
                                color="#DD8452", alpha=0.6))

    ax.set_xscale("log")
    ax.set_xlabel("Online time per query (s, log scale)")
    ax.set_ylabel(r"Detection $F_{\beta=0.5}$ (ND+DS)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Detection accuracy vs. online cost on FIVR-50K")

    # ── legend ────────────────────────────────────────────────────────────────
    baseline_handle = mlines.Line2D([], [], marker="o", color="w",
                                    markerfacecolor="grey", markersize=9,
                                    label="Baseline methods")
    hapdp_handle = mlines.Line2D([], [], marker="^", color="w",
                                 markerfacecolor="white",
                                 markeredgecolor=HAP_DDP_COLOR,
                                 markeredgewidth=2, markersize=9,
                                 label="HAP-DPP")
    ax.legend(handles=[baseline_handle, hapdp_handle],
              loc="lower right", frameon=False, fontsize=9)

    fig.tight_layout()

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / "hap_ddp_comparison.png"
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved: {path}")
    else:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
