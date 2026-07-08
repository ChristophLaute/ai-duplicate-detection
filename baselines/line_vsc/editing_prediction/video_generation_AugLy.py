import argparse
import csv
import glob
import json
import os
import random
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from augly.video import (
    BlendVideos,
    Brightness,
    ColorJitter,
    Contrast,
    Crop,
    Grayscale,
    HFlip,
    HStack,
    InsertInBackground,
    Loop,
    Overlay,
    OverlayDots,
    OverlayEmoji,
    OverlayShapes,
    OverlayText,
    Pad,
    PerspectiveTransformAndShake,
    RandomAspectRatio,
    RandomBlur,
    RandomEncodingQuality,
    RandomNoise,
    RandomPixelization,
    RandomVideoSpeed,
    Rotate,
    VFlip,
    VStack,
)


class RotateWithLabel:
    """Wrapper around augly.video.Rotate that records the correct angle label.

    AugLy's Rotate always stores {"name": "rotate"} regardless of the angle,
    so prepare.py's AUG_LABEL_MAPPING for rotate90_270 / rotate180_vflip would
    never fire.  This wrapper bypasses AugLy metadata and appends the right
    label directly.
    """

    def __init__(self, degrees, label):
        self._aug = Rotate(degrees=degrees)
        self._label = label

    def __call__(self, input_path, output_path, metadata=None):
        # Call without metadata so AugLy doesn't record its own "rotate" entry.
        self._aug(input_path, output_path)
        if metadata is not None:
            metadata.append({"name": self._label, "sub_name": self._label})


class VStackHStack:
    """Tile four videos in a 2×2 grid using ffmpeg: [tl tr / bl br].

    AugLy has no VStackHStack — this replicates the LINE VSC paper's training
    augmentation via a custom ffmpeg filter_complex.
    """

    def __init__(
        self,
        video2,
        video3,
        video4,
        width,
        height,
        ffmpeg_preset="ultrafast",
        ffmpeg_crf=18,
        ffmpeg_threads=None,
    ):
        self._v2 = str(video2)
        self._v3 = str(video3)
        self._v4 = str(video4)
        self._ffmpeg_preset = ffmpeg_preset
        self._ffmpeg_crf = ffmpeg_crf
        self._ffmpeg_threads = ffmpeg_threads
        hw = width // 2
        hh = height // 2
        scale = (
            f"scale={hw}:{hh}:force_original_aspect_ratio=decrease,"
            f"pad={hw}:{hh}:(ow-iw)/2:(oh-ih)/2,setsar=1"
        )
        self._filter = (
            f"[0:v]{scale}[tl];[1:v]{scale}[tr];"
            f"[2:v]{scale}[bl];[3:v]{scale}[br];"
            "[tl][tr]hstack[top];[bl][br]hstack[bottom];[top][bottom]vstack[out]"
        )

    def __call__(self, input_path, output_path, metadata=None):
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", input_path,
            "-i", self._v2,
            "-i", self._v3,
            "-i", self._v4,
            "-filter_complex", self._filter,
            "-map", "[out]",
            "-c:v", "libx264", "-preset", self._ffmpeg_preset,
            "-crf", str(self._ffmpeg_crf),
        ]
        if self._ffmpeg_threads is not None and self._ffmpeg_threads > 0:
            cmd.extend(["-threads", str(self._ffmpeg_threads)])
        cmd.append(output_path)
        subprocess.run(cmd, check=True)
        if metadata is not None:
            metadata.append({"name": "vstack_hstack", "sub_name": "vstack_hstack"})


def select_emoji_path():
    import augly as _augly_pkg
    twemoji_root = Path(_augly_pkg.__file__).resolve().parent / "assets" / "twemojis"
    emoji_dirs = list(twemoji_root.glob("*"))
    emoji_dir = random.choice(emoji_dirs)
    emoji_pics = list(emoji_dir.glob("*.png"))
    return str(random.choice(emoji_pics))


def collect_emoji_paths():
    import augly as _augly_pkg

    twemoji_root = Path(_augly_pkg.__file__).resolve().parent / "assets" / "twemojis"
    emoji_paths = [str(p) for p in twemoji_root.glob("*/*.png")]
    return emoji_paths


def ffmpeg_assemble_video(
    ref_id, query_id, r_start, r_dura_copy, q_dura_org,
    q_insert_time, q_fps, q_w, q_h, out_path,
    ffmpeg_preset="ultrafast", ffmpeg_crf=18, ffmpeg_threads=None,
):
    """Insert a ref clip into query video at q_insert_time using ffmpeg.

    Replaces the decord-based assembly.  decord.VideoReader hangs indefinitely
    on DAS-6 nodes due to /dev/random entropy starvation.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Extract ref clip and re-encode to match query dimensions/fps
        ref_clip = os.path.join(tmpdir, "ref_clip.mp4")
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", str(r_start), "-t", str(r_dura_copy),
            "-i", str(ref_id),
            "-vf", (
                f"scale={q_w}:{q_h}:force_original_aspect_ratio=decrease,"
                f"pad={q_w}:{q_h}:(ow-iw)/2:(oh-ih)/2,setsar=1"
            ),
            "-r", str(q_fps),
            "-c:v", "libx264", "-preset", ffmpeg_preset,
            "-crf", str(ffmpeg_crf), "-an",
        ]
        if ffmpeg_threads is not None and ffmpeg_threads > 0:
            cmd.extend(["-threads", str(ffmpeg_threads)])
        cmd.append(ref_clip)
        subprocess.run(cmd, check=True)

        parts = []

        # 2. Query segment before insert point
        if q_insert_time > 0:
            before = os.path.join(tmpdir, "before.mp4")
            cmd = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-t", str(q_insert_time),
                "-i", str(query_id),
                "-c:v", "libx264", "-preset", ffmpeg_preset,
                "-crf", str(ffmpeg_crf), "-an",
            ]
            if ffmpeg_threads is not None and ffmpeg_threads > 0:
                cmd.extend(["-threads", str(ffmpeg_threads)])
            cmd.append(before)
            subprocess.run(cmd, check=True)
            parts.append(before)

        parts.append(ref_clip)

        # 3. Query segment after insert point
        if q_insert_time < q_dura_org:
            after = os.path.join(tmpdir, "after.mp4")
            cmd = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-ss", str(q_insert_time),
                "-i", str(query_id),
                "-c:v", "libx264", "-preset", ffmpeg_preset,
                "-crf", str(ffmpeg_crf), "-an",
            ]
            if ffmpeg_threads is not None and ffmpeg_threads > 0:
                cmd.extend(["-threads", str(ffmpeg_threads)])
            cmd.append(after)
            subprocess.run(cmd, check=True)
            parts.append(after)

        # 4. Concatenate
        concat_list = os.path.join(tmpdir, "concat.txt")
        with open(concat_list, "w") as f:
            for p in parts:
                f.write(f"file '{p}'\n")

        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "concat", "-safe", "0",
            "-i", concat_list,
            "-c:v", "libx264", "-preset", ffmpeg_preset,
            "-crf", str(ffmpeg_crf),
        ]
        if ffmpeg_threads is not None and ffmpeg_threads > 0:
            cmd.extend(["-threads", str(ffmpeg_threads)])
        cmd.append(out_path)
        subprocess.run(cmd, check=True)


def main(args):
    gt_match_csv = f"{args.output}/cvl_train_matching_ground_truth.csv"
    edit_query_meta_csv = f"{args.output}/cvl_train_query_metadata.csv"
    if not Path(gt_match_csv).is_file():
        os.makedirs(f"{args.output}/query", exist_ok=True)
        os.makedirs(f"{args.output}/cvl_train_augly_metadata", exist_ok=True)
        with open(gt_match_csv, "a") as f_gt:
            writer_gt = csv.writer(f_gt, delimiter=",")
            writer_gt.writerow(
                ["query_id", "ref_id", "query_start", "query_end", "ref_start", "ref_end"]
            )
        with open(edit_query_meta_csv, "a") as f_meta:
            writer_meta = csv.writer(f_meta, delimiter=",")
            writer_meta.writerow(
                ["video_id", "duration_sec", "frames_per_sec", "width", "height", "rn", "base_query_id"]
            )

    ref_meta = pd.read_csv(args.meta_path)
    meta_by_id = {
        str(row["video_id"]): row
        for _, row in ref_meta.iterrows()
    }

    video_files = list(Path(args.data_path).glob("*.mp4"))
    if not video_files:
        raise FileNotFoundError(f"No .mp4 files found in data_path: {args.data_path}")

    emoji_paths = collect_emoji_paths()
    if not emoji_paths:
        raise FileNotFoundError("No AugLy emoji assets found.")

    loop_started_at = time.time()
    generated_count = 0

    for i in range(args.sample_range[0], args.sample_range[1]):
        sample_id = f"Q4{str(i).zfill(5)}"
        out_vid_path = f"{args.output}/query/{sample_id}.mp4"
        out_meta_path = f"{args.output}/cvl_train_augly_metadata/{sample_id}.json"

        # Skip already-generated samples (resume support)
        if Path(out_vid_path).exists() and Path(out_meta_path).exists():
            continue

        try:
            ref_id = random.choice(video_files)
            r_meta = meta_by_id.get(ref_id.stem)
            if r_meta is None:
                continue
            r_dura_org = float(r_meta["duration_sec"])
            if r_dura_org <= 5:
                continue
            r_dura_copy = random.uniform(4, min(r_dura_org - 1, 60))
            r_start = random.uniform(0, r_dura_org - r_dura_copy)
            r_end = r_start + r_dura_copy

            query_id = random.choice(video_files)
            q_meta = meta_by_id.get(query_id.stem)
            if q_meta is None:
                continue
            q_dura_org = float(q_meta["duration_sec"])
            q_fps = float(q_meta["frames_per_sec"])
            q_w = int(q_meta["width"])
            q_h = int(q_meta["height"])

            q_n_frames = int(q_fps * q_dura_org)
            q_insert_frame = random.choice(range(0, q_n_frames))
            q_insert_time = q_insert_frame / q_fps

            r_frame_count = int(q_fps * r_dura_copy)
            frameNo = q_n_frames + r_frame_count

            video_in2 = random.choice(video_files)
            video_in3 = random.choice(video_files)
            video_in4 = random.choice(video_files)
            emoji_path = random.choice(emoji_paths)
            brightness_level = random.uniform(-0.6, 0.6)
            contrast_level = random.uniform(-2.0, 2.0)
            saturation_factor = random.uniform(0.0, 3.0)
            rotate_degrees = random.choice([d for d in range(-30, 30, 5)])
            crop_left = random.uniform(0, 0.3)
            crop_top = random.uniform(0, 0.3)
            crop_right = random.uniform(0.7, 1.0)
            crop_bottom = random.uniform(0.7, 1.0)
            pad_w_factor = random.uniform(0, 0.25)
            pad_h_factor = random.uniform(0, 0.25)
            pad_color_r = random.randint(0, 255)
            pad_color_g = random.randint(0, 255)
            pad_color_b = random.randint(0, 255)
            overlay_num_dots = random.randint(50, 250)
            overlay_dot_type = random.choice(["colored", "blur"])
            overlay_random_movement = random.choice([True, False])
            overlay_num_shapes = random.randint(1, 3)
            overlay_text_len = random.randint(5, 15)
            shake_sigma = random.uniform(10, 50)
            shake_radius = random.uniform(0, 1.0)
            emoji_x_factor = random.uniform(0.1, 0.6)
            emoji_y_factor = random.uniform(0.1, 0.6)
            emoji_opacity = random.uniform(0.5, 1.0)
            emoji_size = random.uniform(0.2, 0.6)
            opacity = random.uniform(0.1, 0.5)
            overlay_size = random.choice([1.0, 1.0, 1.0, 0.8, 0.9])
            overlay_xy = [[0.2, 0], [0.3, 0], [0.4, 0], [0.5, 0]]
            overlay_xy_i = random.randint(0, len(overlay_xy) - 1)
            overlay_x_factor = overlay_xy[overlay_xy_i][0]
            overlay_y_factor = overlay_xy[overlay_xy_i][1]

            augly_method = [
                Brightness(level=brightness_level),
                Contrast(level=contrast_level),
                ColorJitter(saturation_factor=2.5),
                Grayscale(),
                RandomBlur(min_sigma=4.0, max_sigma=12.0),
                RandomPixelization(min_ratio=0.1, max_ratio=0.7),
                RandomAspectRatio(min_ratio=0.5, max_ratio=2.0),
                RandomVideoSpeed(min_factor=0.2, max_factor=5.0),
                Rotate(degrees=rotate_degrees),
                # Explicit angle labels so prepare.py's AUG_LABEL_MAPPING fires correctly
                RotateWithLabel(90, "rotate90"),
                RotateWithLabel(270, "rotate270"),
                RotateWithLabel(180, "rotate180"),
                Crop(left=crop_left, top=crop_top, right=crop_right, bottom=crop_bottom),
                Pad(
                    w_factor=pad_w_factor,
                    h_factor=pad_w_factor,
                    color=(pad_color_r, pad_color_g, pad_color_b),
                ),
                RandomNoise(min_level=15, max_level=60),
                RandomEncodingQuality(min_quality=10, max_quality=45),
                HFlip(),
                VFlip(),
                Loop(),
                OverlayDots(
                    num_dots=overlay_num_dots,
                    dot_type=overlay_dot_type,
                    random_movement=overlay_random_movement,
                ),
                OverlayShapes(num_shapes=overlay_num_shapes),
                OverlayText(text_len=overlay_text_len),
                PerspectiveTransformAndShake(sigma=shake_sigma, shake_radius=shake_radius),
                OverlayEmoji(
                    emoji_path=emoji_path,
                    x_factor=emoji_x_factor,
                    y_factor=emoji_y_factor,
                    opacity=emoji_opacity,
                    emoji_size=emoji_size,
                ),
                BlendVideos(str(video_in2), opacity=opacity, overlay_size=overlay_size),
                HStack(str(video_in2)),
                VStack(str(video_in2)),
                Overlay(str(video_in2), x_factor=overlay_x_factor, y_factor=overlay_y_factor),
                InsertInBackground(str(video_in2), offset_factor=0.1),
                VStackHStack(
                    str(video_in2),
                    str(video_in3),
                    str(video_in4),
                    q_w,
                    q_h,
                    ffmpeg_preset=args.ffmpeg_preset,
                    ffmpeg_crf=args.ffmpeg_crf,
                    ffmpeg_threads=args.ffmpeg_threads,
                ),
            ]
            edit_method = random.choice(augly_method)

            # Assemble copy_video and apply augmentation
            meta_list = []
            with tempfile.TemporaryDirectory() as tmpdir:
                copy_vid_path = os.path.join(tmpdir, "copy_video.mp4")
                ffmpeg_assemble_video(
                    ref_id=ref_id,
                    query_id=query_id,
                    r_start=r_start,
                    r_dura_copy=r_dura_copy,
                    q_dura_org=q_dura_org,
                    q_insert_time=q_insert_time,
                    q_fps=q_fps,
                    q_w=q_w,
                    q_h=q_h,
                    out_path=copy_vid_path,
                    ffmpeg_preset=args.ffmpeg_preset,
                    ffmpeg_crf=args.ffmpeg_crf,
                    ffmpeg_threads=args.ffmpeg_threads,
                )
                edit_method(copy_vid_path, out_vid_path, metadata=meta_list)

            factor = 1
            if meta_list[0]["name"] == "change_video_speed":
                factor = meta_list[0]["factor"]

            with open(gt_match_csv, "a") as f_gt:
                writer_gt = csv.writer(f_gt, delimiter=",")
                q_start_sec = q_insert_frame / q_fps / factor
                q_end_sec = q_start_sec + r_frame_count / q_fps / factor
                row = [
                    sample_id,
                    ref_id.stem,
                    q_start_sec,
                    q_end_sec,
                    r_start,
                    r_end,
                ]
                writer_gt.writerow(row)

            with open(edit_query_meta_csv, "a") as f_meta:
                writer_meta = csv.writer(f_meta, delimiter=",")
                duration_sec = frameNo / q_fps / factor
                row = [
                    sample_id,
                    duration_sec,
                    q_fps,
                    q_w,
                    q_h,
                    i,
                    query_id.stem,
                ]
                writer_meta.writerow(row)

            with open(out_meta_path, "w") as f:
                json.dump(meta_list[0], f)

            generated_count += 1
            if generated_count % 25 == 0:
                elapsed = time.time() - loop_started_at
                rate = generated_count / elapsed if elapsed > 0 else 0.0
                print(
                    f"[INFO] generated={generated_count} "
                    f"elapsed_sec={elapsed:.1f} rate={rate:.3f} samples/sec"
                )

        except Exception as e:
            print(f"[WARN] sample {i} failed: {e}")
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, default="competition_data/train/reference/"
    )
    parser.add_argument(
        "--meta_path",
        type=str,
        default="competition_data/train/train_reference_metadata.csv",
    )
    parser.add_argument("--output", type=str, default="gen_data/train_reference")
    parser.add_argument("--sample_range", nargs="*", type=int, required=True)
    parser.add_argument("--ffmpeg_preset", type=str, default="ultrafast")
    parser.add_argument("--ffmpeg_crf", type=int, default=28)
    parser.add_argument("--ffmpeg_threads", type=int, default=0)
    args = parser.parse_args()
    if args.ffmpeg_threads is not None and args.ffmpeg_threads <= 0:
        args.ffmpeg_threads = None
    main(args)
