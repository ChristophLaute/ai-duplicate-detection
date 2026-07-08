

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})

METHOD_COLORS = {
    "TMK": "#4C72B0",
    "HAP-VR": "#DD8452",
    "LINE-VSC": "#55A868",
    "VVS": "#C44E52",
    "CompactSiamese": "#8172B2",
}

# FIVR-50K
METHODS = ["TMK", "HAP-VR", "LINE-VSC", "VVS", "CompactSiamese"]
ONLINE_TIME_50K = [16.5797, 67.610, 17.29, 0.4971, 20.2382]   # s / query
OFFLINE_TIME = [1.4858, 1.1307, 2.7690, 1.369, 1.730]          # s / video
DSVR_MAP_50K = [0.1797, 0.9225, 0.9008, 0.7093, 0.0710]
DET_FBETA_50K = [0.3243, 0.8103, 0.8508, 0.6979, 0.1167]  # ND+DS, overall


def offline_to_marker_size(offline_times, lo=80, hi=500):
    offline_times = np.asarray(offline_times)
    norm = (offline_times - offline_times.min()) / (offline_times.max() - offline_times.min())
    return lo + norm * (hi - lo)


def plot_panel(ax, x, y, sizes, ylabel, title):
    for method, xi, yi, si in zip(METHODS, x, y, sizes):
        ax.scatter(xi, yi, s=si, color=METHOD_COLORS[method], alpha=0.75,
                   edgecolor="white", linewidth=1.2, zorder=3, label=method)
        ax.annotate(method, (xi, yi), textcoords="offset points", xytext=(8, 6), fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("Online time per query (s, log scale)")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.05)
    ax.set_title(title)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Save figure here instead of showing interactively")
    args = parser.parse_args()
    save_dir = Path(args.save_dir) if args.save_dir else None

    sizes = offline_to_marker_size(OFFLINE_TIME)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_panel(axes[0], ONLINE_TIME_50K, DSVR_MAP_50K, sizes,
               "DSVR mAP", "Retrieval accuracy vs. online cost (FIVR-50K)")
    plot_panel(axes[1], ONLINE_TIME_50K, DET_FBETA_50K, sizes,
               r"Detection $F_{\beta=0.5}$ (ND+DS)", "Detection accuracy vs. online cost (FIVR-50K)")

    # Shared legend for marker size (offline cost)
    handles = []
    for offline_val, label in zip([min(OFFLINE_TIME), max(OFFLINE_TIME)], ["lower offline cost", "higher offline cost"]):
        size = offline_to_marker_size(np.array(OFFLINE_TIME + [offline_val]))[-1]
        handles.append(axes[0].scatter([], [], s=size, color="grey", alpha=0.5,
                                        edgecolor="white", label=f"{label} ({offline_val:.2f} s/video)"))
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.08), title="Marker size: offline cost (s/video)")

    fig.suptitle("Accuracy-Efficiency Frontier on FIVR-50K", y=1.02)
    fig.tight_layout()

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / "fivr_efficiency_frontier.png"
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved: {path}")
    else:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
