import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))

from runtime_config import VCDB_CORE_DIR
from data.FramePrep import (
    interframe_difference_curve,
    smooth_curve_from_values,
    detect_local_maxima,
    select_frame_indices,
)
SAVE_DIR = VCDB_CORE_DIR.parent.parent / "experiments" / "fragment_analysis" / "plots"
SAVE_DIR.mkdir(parents=True, exist_ok=True)




VIDEO_EXTS = ("*.mp4", "*.flv", "*.avi", "*.mov", "*.mkv")


def list_videos(video_dir):
    videos = []
    for ext in VIDEO_EXTS:
        videos.extend(
            p for p in video_dir.rglob(ext)
            if p.is_file() and not p.name.startswith("._")
        )
    return sorted(videos)


def analyze_video(video_path):
    curve, n_frames = interframe_difference_curve(video_path)
    smoothed = smooth_curve_from_values(curve, window_size=50)
    maxima = detect_local_maxima(smoothed, n_frames=n_frames)
    fragments = select_frame_indices(curve, n_frames=n_frames, len_frames=15, window_size=50)

    return {
        "video_rel_path": str(video_path.relative_to(VCDB_CORE_DIR)),
        "event": video_path.parent.name,
        "stem": video_path.stem,
        "n_frames": n_frames,
        "curve_len": len(curve),
        "num_maxima": len(maxima),
        "num_fragments": len(fragments),
        "is_empty": int(len(fragments) == 0),
    }

def plot_video_analysis(df):
    #histogram of maxima
    num_maxima = df["num_maxima"].dropna()
    sns.set_theme(style="whitegrid")
    sns.histplot(num_maxima, bins=30, kde=True)
    plt.title("Distribution of Number of Maxima Detected in Videos")
    plt.axvline(15, color='red', linestyle='--', label='Threshold for Fragments')
    plt.legend()
    plt.xlabel("Number of Maxima")
    plt.ylabel("Number of Videos")
    plt.savefig(SAVE_DIR / "num_maxima_histogram.png", dpi=200, bbox_inches="tight")
    # scatter plot of num_maxima vs num_fragments
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="num_maxima", y="num_fragments", hue="is_empty", alpha=0.7)
    plt.title("Scatter Plot of Maxima vs Kept Fragments")
    plt.axvline(15, color='red', linestyle='--', label='Threshold for Fragments')
    plt.xlabel("Number of Maxima")
    plt.ylabel("Number of Fragments")
    plt.savefig(SAVE_DIR / "num_maxima_vs_fragments.png", dpi=200, bbox_inches="tight")
    plt.close()



def main():
    videos = list_videos(VCDB_CORE_DIR)
    rows = []

    for video_path in videos:
        try:
            rows.append(analyze_video(video_path))
        except Exception as e:
            rows.append({
                "video_rel_path": str(video_path.relative_to(VCDB_CORE_DIR)),
                "event": video_path.parent.name,
                "stem": video_path.stem,
                "n_frames": None,
                "curve_len": None,
                "num_maxima": None,
                "num_fragments": None,
                "is_empty": None,
                "error": str(e),
            })

    df = pd.DataFrame(rows)
    out_path = VCDB_CORE_DIR.parent / "fragment_analysis.csv"
    df.to_csv(out_path, index=False)

    print(f"Saved: {out_path}")
    print(df["is_empty"].value_counts(dropna=False))




if __name__ == "__main__":
    main()
