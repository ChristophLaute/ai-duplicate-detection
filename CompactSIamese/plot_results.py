"""
Generate plots for any CompactSiamese experiment output directory.

Automatically detects what files are present and plots accordingly:
  - metrics.json        → bar chart of scalar metrics
  - per_query_metrics   → AP distribution, recall@k bars, first-relevant-rank histogram
  - rankings.csv        → precision-recall curve, score distributions
  - pair_predictions.csv → score histogram by label, ROC curve

Usage:
    # Single experiment:
    python plot_results.py --dir evaluation_results/retrieval_chamfer_20260402_203234

    # Compare multiple experiments side by side:
    python plot_results.py --dir eval_results/exp1 eval_results/exp2 eval_results/exp3 --compare

    # Save plots to a directory instead of showing interactively:
    python plot_results.py --dir evaluation_results/retrieval_chamfer_20260402_203234 --save-dir plots/
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve

plt.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})

METRIC_COLORS = {
    "accuracy": "#4C72B0",
    "precision": "#DD8452",
    "recall": "#55A868",
    "f1": "#C44E52",
    "fbeta": "#E377C2",
    "mAP": "#8172B2",
    "recall@1": "#937860",
    "recall@5": "#DA8BC3",
    "recall@10": "#8C8C8C",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json(path):
    with open(path) as f:
        return json.load(f)


def find_csv(directory: Path, *stems):
    """Return first matching CSV file for any of the given stems."""
    for stem in stems:
        for p in directory.glob(f"*{stem}*"):
            if p.suffix == ".csv":
                return p
    return None


def savefig(fig, save_dir: Path | None, name: str):
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / name
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved: {path}")
    else:
        plt.show()
    plt.close(fig)


def exp_label(directory: Path) -> str:
    return directory.name


# ---------------------------------------------------------------------------
# Single-experiment plots
# ---------------------------------------------------------------------------

def plot_scalar_metrics(metrics: dict, title: str, save_dir: Path | None, prefix: str = ""):
    keys = ["accuracy", "precision", "recall", "f1", "fbeta", "mAP", "recall@1", "recall@5", "recall@10"]
    present = {k: metrics[k] for k in keys if k in metrics and metrics[k] is not None}
    if not present:
        return

    fig, ax = plt.subplots(figsize=(max(6, len(present) * 0.9), 4))
    bars = ax.bar(
        list(present.keys()),
        list(present.values()),
        color=[METRIC_COLORS.get(k, "#aaaaaa") for k in present],
        width=0.6,
        zorder=3,
    )
    for bar, val in zip(bars, present.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title(title)
    savefig(fig, save_dir, f"{prefix}metrics_bar.png")


def plot_ap_distribution(query_df: pd.DataFrame, title: str, save_dir: Path | None, prefix: str = ""):
    if "average_precision" not in query_df.columns:
        return
    ap = query_df["average_precision"].dropna()
    if len(ap) == 0:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(ap, bins=20, color="#4C72B0", edgecolor="white", zorder=3)
    ax.axvline(ap.mean(), color="#C44E52", linestyle="--", label=f"mean AP = {ap.mean():.3f}")
    ax.axvline(ap.median(), color="#55A868", linestyle="--", label=f"median AP = {ap.median():.3f}")
    ax.set_xlabel("Average Precision")
    ax.set_ylabel("Number of queries")
    ax.set_title(title)
    ax.legend()
    savefig(fig, save_dir, f"{prefix}ap_distribution.png")


def plot_recall_at_k_bars(query_df: pd.DataFrame, title: str, save_dir: Path | None, prefix: str = ""):
    cols = [c for c in ["recall@1", "recall@5", "recall@10"] if c in query_df.columns]
    if not cols:
        return

    means = [query_df[c].mean() for c in cols]
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(cols, means, color=[METRIC_COLORS.get(c, "#aaaaaa") for c in cols], width=0.5, zorder=3)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Recall")
    ax.set_title(title)
    savefig(fig, save_dir, f"{prefix}recall_at_k.png")


def plot_first_relevant_rank(query_df: pd.DataFrame, title: str, save_dir: Path | None, prefix: str = ""):
    if "first_relevant_rank" not in query_df.columns:
        return
    ranks = query_df["first_relevant_rank"].dropna()
    if len(ranks) == 0:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(ranks, bins=30, color="#8172B2", edgecolor="white", zorder=3)
    ax.axvline(ranks.mean(), color="#C44E52", linestyle="--", label=f"mean = {ranks.mean():.1f}")
    ax.axvline(ranks.median(), color="#55A868", linestyle="--", label=f"median = {ranks.median():.1f}")
    ax.set_xlabel("Rank of first relevant result")
    ax.set_ylabel("Number of queries")
    ax.set_title(title)
    ax.legend()
    savefig(fig, save_dir, f"{prefix}first_relevant_rank.png")


def plot_precision_recall_curve_from_rankings(rankings_df: pd.DataFrame, title: str,
                                               save_dir: Path | None, prefix: str = ""):
    """Micro-averaged PR curve from the ranked list (is_relevant column)."""
    if "is_relevant" not in rankings_df.columns or "score" not in rankings_df.columns:
        return

    y_true = rankings_df["is_relevant"].values
    y_score = rankings_df["score"].values

    if y_true.sum() == 0:
        return

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = auc(recall, precision)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color="#4C72B0", lw=2, label=f"AP = {ap:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.legend()
    savefig(fig, save_dir, f"{prefix}pr_curve.png")


def plot_score_distribution(rankings_df: pd.DataFrame, title: str, save_dir: Path | None, prefix: str = ""):
    if "score" not in rankings_df.columns or "is_relevant" not in rankings_df.columns:
        return

    rel = rankings_df[rankings_df["is_relevant"] == 1]["score"].dropna()
    irrel = rankings_df[rankings_df["is_relevant"] == 0]["score"].dropna()

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(rankings_df["score"].min(), rankings_df["score"].max(), 40)
    ax.hist(irrel, bins=bins, alpha=0.6, color="#DD8452", label=f"Non-relevant (n={len(irrel)})", zorder=3)
    ax.hist(rel, bins=bins, alpha=0.8, color="#55A868", label=f"Relevant (n={len(rel)})", zorder=4)
    ax.set_xlabel("Retrieval score")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    savefig(fig, save_dir, f"{prefix}score_distribution.png")


def plot_pair_score_histogram(pairs_df: pd.DataFrame, title: str, save_dir: Path | None, prefix: str = ""):
    """For classification-style pair prediction results."""
    score_col = next((c for c in ["logit", "max_output", "score"] if c in pairs_df.columns), None)
    if score_col is None or "label" not in pairs_df.columns:
        return

    pos = pairs_df[pairs_df["label"] == 1][score_col].dropna()
    neg = pairs_df[pairs_df["label"] == 0][score_col].dropna()

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(pairs_df[score_col].min(), pairs_df[score_col].max(), 40)
    ax.hist(neg.values, bins=bins, alpha=0.6, color="#DD8452", label=f"Non-duplicate (n={len(neg)})", zorder=3)
    ax.hist(pos.values, bins=bins, alpha=0.8, color="#55A868", label=f"Duplicate (n={len(pos)})", zorder=4)
    ax.axvline(0, color="black", linestyle="--", linewidth=1, label="Decision boundary (logit=0)")
    ax.set_xlabel(score_col)
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    savefig(fig, save_dir, f"{prefix}score_histogram.png")


def plot_roc_curve(pairs_df: pd.DataFrame, title: str, save_dir: Path | None, prefix: str = ""):
    score_col = next((c for c in ["logit", "max_output", "score"] if c in pairs_df.columns), None)
    if score_col is None or "label" not in pairs_df.columns:
        return

    y_true = pairs_df["label"].values
    y_score = pairs_df[score_col].values
    mask = ~pd.isnull(y_score)
    y_true, y_score = y_true[mask], y_score[mask]

    if len(np.unique(y_true)) < 2:
        return

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, color="#4C72B0", lw=2, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend()
    savefig(fig, save_dir, f"{prefix}roc_curve.png")


def plot_experiment(directory: Path, save_dir: Path | None):
    directory = Path(directory)
    title_base = exp_label(directory)
    prefix = ""

    print(f"\n=== Plotting: {directory} ===")

    # Scalar metrics
    metrics_path = directory / "metrics.json"
    if metrics_path.exists():
        metrics = load_json(metrics_path)
        plot_scalar_metrics(metrics, f"Metrics — {title_base}", save_dir, prefix)

    # Per-query metrics
    query_csv = find_csv(directory, "per_query", "query_metrics", "query_rows")
    if query_csv is not None:
        df = pd.read_csv(query_csv)
        plot_ap_distribution(df, f"AP Distribution — {title_base}", save_dir, prefix)
        plot_recall_at_k_bars(df, f"Recall@k — {title_base}", save_dir, prefix)
        plot_first_relevant_rank(df, f"First Relevant Rank — {title_base}", save_dir, prefix)

    # Rankings
    rankings_csv = find_csv(directory, "rankings", "ranking")
    if rankings_csv is not None:
        df = pd.read_csv(rankings_csv)
        plot_precision_recall_curve_from_rankings(df, f"PR Curve — {title_base}", save_dir, prefix)
        plot_score_distribution(df, f"Score Distribution — {title_base}", save_dir, prefix)

    # Pair predictions (classification)
    pairs_csv = find_csv(directory, "pair_predictions", "pair_pred")
    if pairs_csv is not None:
        df = pd.read_csv(pairs_csv)
        plot_pair_score_histogram(df, f"Score Histogram — {title_base}", save_dir, prefix)
        plot_roc_curve(df, f"ROC Curve — {title_base}", save_dir, prefix)


# ---------------------------------------------------------------------------
# Multi-experiment comparison plots
# ---------------------------------------------------------------------------

def plot_comparison(directories: list[Path], save_dir: Path | None):
    """Compare scalar metrics and mAP across multiple experiments."""
    scalar_keys = ["accuracy", "precision", "recall", "f1", "fbeta", "mAP", "recall@1", "recall@5", "recall@10"]
    rows = []

    for d in directories:
        metrics_path = d / "metrics.json"
        if not metrics_path.exists():
            print(f"  No metrics.json in {d}, skipping comparison entry")
            continue
        m = load_json(metrics_path)
        row = {"experiment": exp_label(d)}
        row.update({k: m[k] for k in scalar_keys if k in m})
        rows.append(row)

    if len(rows) < 2:
        print("Need at least 2 experiments with metrics.json to compare.")
        return

    df = pd.DataFrame(rows).set_index("experiment")
    present_cols = [c for c in scalar_keys if c in df.columns]
    df = df[present_cols].dropna(axis=1, how="all")

    # Grouped bar chart
    n_exp = len(df)
    n_metrics = len(df.columns)
    x = np.arange(n_metrics)
    width = 0.8 / n_exp

    fig, ax = plt.subplots(figsize=(max(8, n_metrics * 1.2), 5))
    colors = plt.cm.tab10(np.linspace(0, 1, n_exp))

    for i, (exp_name, row) in enumerate(df.iterrows()):
        offset = (i - n_exp / 2 + 0.5) * width
        bars = ax.bar(x + offset, row.values, width=width * 0.9, color=colors[i], label=exp_name, zorder=3)
        for bar, val in zip(bars, row.values):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=7, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels(df.columns, rotation=15, ha="right")
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("Score")
    ax.set_title("Experiment Comparison")
    ax.legend(loc="upper right", fontsize=9)
    savefig(fig, save_dir, "comparison_metrics.png")

    # mAP-only comparison if present
    if "mAP" in df.columns:
        fig, ax = plt.subplots(figsize=(max(5, n_exp * 0.8), 4))
        vals = df["mAP"].dropna()
        bars = ax.bar(vals.index, vals.values,
                      color=colors[:len(vals)], width=0.5, zorder=3)
        for bar, val in zip(bars, vals.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("mAP")
        ax.set_title("mAP Comparison")
        ax.tick_params(axis="x", rotation=15)
        savefig(fig, save_dir, "comparison_mAP.png")

    # AP distribution overlay across experiments
    ap_data = {}
    for d in directories:
        query_csv = find_csv(d, "per_query", "query_metrics", "query_rows")
        if query_csv and "average_precision" in pd.read_csv(query_csv, nrows=1).columns:
            ap_data[exp_label(d)] = pd.read_csv(query_csv)["average_precision"].dropna().values

    if len(ap_data) >= 2:
        fig, ax = plt.subplots(figsize=(8, 4))
        bins = np.linspace(0, 1, 25)
        colors_iter = iter(plt.cm.tab10(np.linspace(0, 1, len(ap_data))))
        for name, ap_vals in ap_data.items():
            c = next(colors_iter)
            ax.hist(ap_vals, bins=bins, alpha=0.5, color=c,
                    label=f"{name} (mean={ap_vals.mean():.3f})", zorder=3)
        ax.set_xlabel("Average Precision")
        ax.set_ylabel("Queries")
        ax.set_title("AP Distribution Comparison")
        ax.legend(fontsize=8)
        savefig(fig, save_dir, "comparison_ap_distribution.png")


# ---------------------------------------------------------------------------
# Training plots
# ---------------------------------------------------------------------------

def plot_training_log(log_path: Path, save_dir: Path | None):
    df = pd.read_csv(log_path)
    if df.empty:
        return

    best_epoch = df[df["best"] == True]["epoch"].values if "best" in df.columns else []

    def mark_best(ax):
        for ep in best_epoch:
            ax.axvline(ep, color="#C44E52", alpha=0.25, linewidth=1)

    # --- Loss curves ---
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["epoch"], df["train_loss"], label="Train loss", color="#4C72B0", lw=2)
    ax.plot(df["epoch"], df["val_loss"], label="Val loss", color="#DD8452", lw=2)
    mark_best(ax)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    savefig(fig, save_dir, "training_loss.png")

    # --- Val metrics curves ---
    metric_cols = [c for c in ["val_accuracy", "val_precision", "val_recall", "val_f1", "val_fbeta"] if c in df.columns]
    colors_cycle = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#E377C2"]
    fig, ax = plt.subplots(figsize=(8, 4))
    for col, color in zip(metric_cols, colors_cycle):
        label = col.replace("val_", "").capitalize()
        ax.plot(df["epoch"], df[col], label=label, color=color, lw=2)
    mark_best(ax)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Validation Metrics over Training")
    ax.legend()
    savefig(fig, save_dir, "training_val_metrics.png")

    # --- Train loss vs val F1 on dual axis ---
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax2 = ax1.twinx()
    ax1.plot(df["epoch"], df["train_loss"], color="#4C72B0", lw=2, label="Train loss")
    ax2.plot(df["epoch"], df["val_f1"], color="#C44E52", lw=2, linestyle="--", label="Val F1")
    if "val_fbeta" in df.columns:
        ax2.plot(df["epoch"], df["val_fbeta"], color="#E377C2", lw=2, linestyle=":", label="Val Fbeta")
    mark_best(ax1)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss", color="#4C72B0")
    ax2.set_ylabel("Val F1 / Fbeta", color="#C44E52")
    ax2.set_ylim(0, 1.05)
    ax1.tick_params(axis="y", labelcolor="#4C72B0")
    ax2.tick_params(axis="y", labelcolor="#C44E52")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
    ax1.set_title("Loss vs Val F1 / Fbeta")
    # remove grid from twinx to avoid double grid
    ax2.grid(False)
    savefig(fig, save_dir, "training_loss_vs_f1.png")

    best_col = "val_fbeta" if "val_fbeta" in df.columns else "val_f1"
    best_idx = df[best_col].idxmax()
    print(f"Best epoch: {df.loc[best_idx, 'epoch']}  "
          f"val_fbeta={df.loc[best_idx, best_col]:.4f}  "
          f"val_f1={df.loc[best_idx, 'val_f1']:.4f}  "
          f"val_acc={df.loc[best_idx, 'val_accuracy']:.4f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", nargs="+", required=True,
                        help="Experiment output directory/directories")
    parser.add_argument("--training-log", type=str, default=None,
                        help="Path to training_log.csv (or directory containing it)")
    parser.add_argument("--compare", action="store_true",
                        help="Plot comparison across all provided directories")
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Save plots here instead of showing interactively")
    args = parser.parse_args()

    directories = [Path(d) for d in args.dir]
    save_dir = Path(args.save_dir) if args.save_dir else None

    # Training log
    if args.training_log:
        log_path = Path(args.training_log)
        if log_path.is_dir():
            log_path = log_path / "training_log.csv"
        if log_path.exists():
            print(f"\n=== Plotting training log: {log_path} ===")
            plot_training_log(log_path, save_dir)
        else:
            print(f"WARNING: training log not found at {log_path}")
    else:
        # Auto-detect training_log.csv in the first --dir
        for d in directories:
            candidate = Path(d) / "training_log.csv"
            if candidate.exists():
                print(f"\n=== Auto-detected training log: {candidate} ===")
                plot_training_log(candidate, save_dir)
                break

    for d in directories:
        if not d.exists():
            print(f"WARNING: {d} does not exist, skipping")
            continue
        exp_save_dir = (save_dir / d.name) if save_dir and len(directories) > 1 else save_dir
        plot_experiment(d, exp_save_dir)

    if args.compare and len(directories) > 1:
        compare_save_dir = save_dir / "_comparison" if save_dir else None
        plot_comparison(directories, compare_save_dir)
