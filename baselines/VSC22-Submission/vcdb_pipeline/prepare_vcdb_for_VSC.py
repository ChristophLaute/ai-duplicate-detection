from pathlib import Path
import pandas as pd
import os
import csv

# =========================
# CONFIG
# =========================
VCDB_DIR = Path("/Volumes/LaCie/DuplicateDetectionThesis/VCDB")
ANNOTATION_DIR = VCDB_DIR / "annotation"
VIDEO_DIR = VCDB_DIR / "core_dataset"

ADAPTER_DIR = VCDB_DIR / "vsc_adapter"
TRAIN_VIDEOS_DIR = ADAPTER_DIR / "videos" / "train"
TRAIN_META_DIR = ADAPTER_DIR / "meta" / "train"
TRAIN_JPG_ZIPS_DIR = ADAPTER_DIR / "jpg_zips"
TRAIN_LMDB_DIR = ADAPTER_DIR / "lmdb"
SHARED_DIR = ADAPTER_DIR / "meta"

START_Q_ID = 100001
START_R_ID = 100001
SKIP_SELF_PAIRS = True


# =========================
# HELPERS
# =========================
def ensure_dirs():
    for p in [
        TRAIN_META_DIR,
        TRAIN_VIDEOS_DIR / "query",
        TRAIN_VIDEOS_DIR / "reference",
        TRAIN_JPG_ZIPS_DIR,
        TRAIN_LMDB_DIR,
        SHARED_DIR,
    ]:
        p.mkdir(parents=True, exist_ok=True)


def hhmmss_to_seconds(s: str) -> int:
    h, m, sec = map(int, s.strip().split(":"))
    return h * 3600 + m * 60 + sec


def make_symlink(src: Path, dst: Path):
    if dst.is_symlink() or dst.exists():
        if dst.is_dir() and not dst.is_symlink():
            raise ValueError(f"Destination exists as directory: {dst}")
        dst.unlink()
    os.symlink(src, dst)


def index_videos_by_name(video_dir: Path) -> dict[str, list[Path]]:
    index = {}
    for p in video_dir.rglob("*"):
        if p.is_file():
            index.setdefault(p.name, []).append(p)
    return index


def load_vcdb_annotations(annotation_dir: Path) -> pd.DataFrame:
    rows = []

    for txt_file in sorted(annotation_dir.glob("*.txt")):
        with txt_file.open("r", encoding="latin-1") as f:
            reader = csv.reader(f)
            for parts in reader:
                if not parts:
                    continue

                parts = [p.strip() for p in parts]
                if len(parts) != 6:
                    continue

                video_a, video_b, a_start, a_end, b_start, b_end = parts

                rows.append({
                    "source_file": txt_file.name,
                    "video_a": video_a,
                    "video_b": video_b,
                    "a_start": a_start,
                    "a_end": a_end,
                    "b_start": b_start,
                    "b_end": b_end,
                })

    return pd.DataFrame(rows)


# =========================
# MAIN
# =========================
def main():
    ensure_dirs()

    print("Loading VCDB annotations...")
    ann_df = load_vcdb_annotations(ANNOTATION_DIR)
    print(f"Total raw annotation rows: {len(ann_df)}")

    if SKIP_SELF_PAIRS:
        ann_df = ann_df[ann_df["video_a"] != ann_df["video_b"]].copy()
        print(f"Rows after removing self-pairs: {len(ann_df)}")

    print("Indexing source videos...")
    video_index = index_videos_by_name(VIDEO_DIR)

    # BETTER DESIGN:
    # Every original video gets both:
    # - one query ID
    # - one reference ID
    #
    # This means if VCDB has N unique originals,
    # you will have N Q-IDs and N R-IDs.

    all_orig_names = sorted(set(ann_df["video_a"].tolist()) | set(ann_df["video_b"].tolist()))
    print(f"Unique original VCDB videos involved: {len(all_orig_names)}")

    q_map = {}
    r_map = {}

    q_counter = START_Q_ID
    r_counter = START_R_ID

    for name in all_orig_names:
        q_map[name] = f"Q{q_counter:06d}"
        r_map[name] = f"R{r_counter:06d}"
        q_counter += 1
        r_counter += 1

    missing = []
    ambiguous = []

    def resolve_unique_path(orig_name: str):
        matches = video_index.get(orig_name, [])
        if len(matches) == 1:
            return matches[0]
        elif len(matches) == 0:
            missing.append(orig_name)
            return None
        else:
            ambiguous.append((orig_name, matches))
            return None

    # Create BOTH query and reference symlinks for every original video
    query_rows = []
    ref_rows = []

    print("Creating dual-role symlinks...")
    for orig_name in all_orig_names:
        src = resolve_unique_path(orig_name)
        if src is None:
            continue

        qid = q_map[orig_name]
        rid = r_map[orig_name]

        q_dst = TRAIN_VIDEOS_DIR / "query" / f"{qid}.mp4"
        r_dst = TRAIN_VIDEOS_DIR / "reference" / f"{rid}.mp4"

        make_symlink(src, q_dst)
        make_symlink(src, r_dst)

        query_rows.append({
            "video_id": qid,
            "original_name": orig_name,
            "original_path": str(src),
        })

        ref_rows.append({
            "video_id": rid,
            "original_name": orig_name,
            "original_path": str(src),
        })

    if missing:
        missing_path = SHARED_DIR / "vcdb_missing_videos.txt"
        missing_path.write_text("\n".join(sorted(set(missing))), encoding="utf-8")
        print(f"Missing videos written to: {missing_path}")

    if ambiguous:
        ambiguous_path = SHARED_DIR / "vcdb_ambiguous_videos.txt"
        with ambiguous_path.open("w", encoding="utf-8") as f:
            for name, paths in ambiguous:
                f.write(f"{name}\n")
                for p in paths:
                    f.write(f"  {p}\n")
        print(f"Ambiguous videos written to: {ambiguous_path}")

    valid_orig_names = {row["original_name"] for row in query_rows}

    ann_df = ann_df[
        ann_df["video_a"].isin(valid_orig_names) &
        ann_df["video_b"].isin(valid_orig_names)
    ].copy()

    print(f"Annotation rows after path resolution filter: {len(ann_df)}")

    # Convert VCDB rows to VSC-style GT:
    # video_a always uses its Q-ID
    # video_b always uses its R-ID
    gt_rows = []
    for _, row in ann_df.iterrows():
        gt_rows.append({
            "query_id": q_map[row["video_a"]],
            "ref_id": r_map[row["video_b"]],
            "query_start": hhmmss_to_seconds(row["a_start"]),
            "query_end": hhmmss_to_seconds(row["a_end"]),
            "ref_start": hhmmss_to_seconds(row["b_start"]),
            "ref_end": hhmmss_to_seconds(row["b_end"]),
        })

    gt_df = pd.DataFrame(gt_rows).drop_duplicates().sort_values(
        by=["query_id", "ref_id", "query_start", "ref_start"]
    )

    coarse_gt_df = gt_df[["query_id", "ref_id"]].drop_duplicates().sort_values(
        by=["query_id", "ref_id"]
    )

    query_ids = sorted([row["video_id"] for row in query_rows])
    ref_ids = sorted([row["video_id"] for row in ref_rows])

    positive_query_ids = sorted(gt_df["query_id"].unique().tolist())

    # Minimal metadata sufficient for your current preprocessing code
    train_query_metadata_df = pd.DataFrame({"video_id": query_ids})
    train_reference_metadata_df = pd.DataFrame({"video_id": ref_ids})

    # train_query_vids.txt = all Q IDs
    train_query_vids = query_ids

    # train_vids.txt = all Q and R IDs
    train_vids = sorted(set(query_ids) | set(ref_ids))

    out_train = TRAIN_META_DIR

    gt_df.to_csv(out_train / "train_matching_ground_truth.csv", index=False)
    coarse_gt_df.to_csv(out_train / "train_ground_truth.csv", index=False, header=False)

    pd.Series(query_ids).to_csv(out_train / "train_query_id.csv", index=False, header=False)
    pd.Series(ref_ids).to_csv(out_train / "train_ref_vids.txt", index=False, header=False)
    pd.Series(positive_query_ids).to_csv(out_train / "train_positive_query.txt", index=False, header=False)

    train_query_metadata_df.to_csv(out_train / "train_query_metadata.csv", index=False)
    train_reference_metadata_df.to_csv(out_train / "train_reference_metadata.csv", index=False)

    (out_train / "train_query_vids.txt").write_text("\n".join(train_query_vids) + "\n", encoding="utf-8")
    (out_train / "train_vids.txt").write_text("\n".join(train_vids) + "\n", encoding="utf-8")
    (SHARED_DIR / "vids.txt").write_text("\n".join(train_vids) + "\n", encoding="utf-8")

    # Mapping table: one row for Q role, one row for R role
    query_map_df = pd.DataFrame([
        {"original_name": k, "video_id": v, "role": "query"}
        for k, v in q_map.items() if k in valid_orig_names
    ]).sort_values(by="video_id")

    ref_map_df = pd.DataFrame([
        {"original_name": k, "video_id": v, "role": "reference"}
        for k, v in r_map.items() if k in valid_orig_names
    ]).sort_values(by="video_id")

    mapping_df = pd.concat([query_map_df, ref_map_df], ignore_index=True)
    mapping_df.to_csv(SHARED_DIR / "vcdb_to_vsc_train_mapping.csv", index=False)

    print("\nDone.")
    print(f"Unique original videos resolved: {len(valid_orig_names)}")
    print(f"Query-role videos linked: {len(query_rows)}")
    print(f"Reference-role videos linked: {len(ref_rows)}")
    print(f"Ground-truth rows: {len(gt_df)}")
    print(f"Positive query IDs in GT: {len(positive_query_ids)}")
    print(f"Mapping saved to: {SHARED_DIR / 'vcdb_to_vsc_train_mapping.csv'}")
    print(f"Train metadata written to: {out_train}")


if __name__ == "__main__":
    main()