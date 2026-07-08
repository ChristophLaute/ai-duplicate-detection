#!/usr/bin/env python3
import csv
import subprocess
import time
from pathlib import Path
from tqdm import tqdm

# -------- settings --------
BASE_DIR = Path(__file__).resolve().parent

CSV_PATH = BASE_DIR.parent / "VCSL" / "data" / "videos_url_uuid.csv"
OUTPUT_DIR = Path("/Volumes/LaCie/DuplicateDetectionThesis/VCSL/vcsl_videos")
ARCHIVE_FILE = BASE_DIR / "downloaded_ids.txt"
FAILED_FILE = BASE_DIR / "failed_downloads.csv"

SLEEP_BETWEEN = 0.0
VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".flv", ".mov", ".avi", ".m4v"}
PARTIAL_EXTS = {".part", ".ytdl"}
SUBPROCESS_TIMEOUT = 20
# -------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_cmd(cmd, timeout=SUBPROCESS_TIMEOUT):
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 999, "", f"timeout after {timeout} seconds"


def try_download(url: str, outtmpl: str):
    cmd = [
        "yt-dlp",
        "--no-playlist",
        "--ignore-errors",
        "--no-overwrites",
        "--continue",
        "--restrict-filenames",
        "--download-archive", str(ARCHIVE_FILE),
        "--socket-timeout", "10",
        "-o", outtmpl,
        url,
    ]
    return run_cmd(cmd)


def try_download_fallback(url: str, outtmpl: str):
    cmd = [
        "yt-dlp",
        "--no-playlist",
        "--ignore-errors",
        "--no-overwrites",
        "--continue",
        "--restrict-filenames",
        "--download-archive", str(ARCHIVE_FILE),
        "--socket-timeout", "10",
        "-f", "mp4/best",
        "-o", outtmpl,
        url,
    ]
    return run_cmd(cmd)


def should_try_fallback(stderr_text: str, stdout_text: str) -> bool:
    msg = f"{stdout_text}\n{stderr_text}".lower()

    helpful_cases = [
        "requested format is not available",
        "format not available",
        "unsupported format",
    ]

    hopeless_cases = [
        "video unavailable",
        "private video",
        "this video is private",
        "sign in to confirm your age",
        "404",
        "not found",
        "removed",
        "deleted",
        "forbidden",
        "http error 403",
        "http error 404",
        "unable to download webpage",
        "timeout after",
    ]

    if any(x in msg for x in hopeless_cases):
        return False

    if any(x in msg for x in helpful_cases):
        return True

    return False


def infer_columns(fieldnames):
    uuid_col = None
    url_col = None

    for c in ["uuid", "video_id", "id"]:
        if c in fieldnames:
            uuid_col = c
            break

    for c in ["url", "video_url", "video_urls", "src_url"]:
        if c in fieldnames:
            url_col = c
            break

    return uuid_col, url_col


def build_existing_indexes():
    complete = set()
    partial = set()

    for p in OUTPUT_DIR.iterdir():
        if not p.is_file():
            continue

        suffix = p.suffix.lower()

        if suffix in VIDEO_EXTS:
            complete.add(p.stem)
        elif suffix in PARTIAL_EXTS:
            partial.add(Path(p.stem).stem)

    return complete, partial


def load_failed_ids():
    failed_ids = set()

    if not FAILED_FILE.exists():
        return failed_ids

    try:
        with FAILED_FILE.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                vid = str(row.get("uuid", "")).strip()
                if vid:
                    failed_ids.add(vid)
    except Exception:
        pass

    return failed_ids


def append_failed_row(vid: str, url: str, reason: str):
    file_exists = FAILED_FILE.exists()

    with FAILED_FILE.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["uuid", "url", "reason"])
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "uuid": vid,
            "url": url,
            "reason": reason[:1000],
        })


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print("No rows found in CSV.")
        return

    fieldnames = set(rows[0].keys())
    uuid_col, url_col = infer_columns(fieldnames)

    if uuid_col is None or url_col is None:
        raise ValueError(
            f"Could not infer uuid/url columns from CSV headers: {sorted(fieldnames)}"
        )

    print(f"Using uuid column: uuid")
    print(f"Using url column:  url")
    print(f"Found {len(rows)} rows")
    print(f"CSV path:          {CSV_PATH}")
    print(f"Output dir:        {OUTPUT_DIR}")

    complete_videos, partial_videos = build_existing_indexes()
    failed_ids = load_failed_ids()

    print(f"Already completed: {len(complete_videos)}")
    print(f"Partial files:     {len(partial_videos)}")
    print(f"Known failed IDs:  {len(failed_ids)}")

    skipped_complete = 0
    skipped_failed = 0
    attempted_count = 0
    success_count = 0
    fail_count = 0
    fallback_count = 0

    progress = tqdm(rows, desc="Downloading videos", unit="video")

    for row in progress:
        vid = str(row[uuid_col]).strip()
        url = str(row[url_col]).strip()

        if not vid or not url:
            fail_count += 1
            continue

        if vid in complete_videos:
            skipped_complete += 1
            continue

        if vid in failed_ids:
            skipped_failed += 1
            continue

        attempted_count += 1
        outtmpl = str(OUTPUT_DIR / f"{vid}.%(ext)s")

        code, out, err = try_download(url, outtmpl)

        if code != 0 and should_try_fallback(err, out):
            fallback_count += 1
            code, out, err = try_download_fallback(url, outtmpl)

        if code != 0:
            reason = (err or out or "unknown_error").strip().replace("\n", " | ")
            append_failed_row(vid, url, reason)
            failed_ids.add(vid)
            fail_count += 1
        else:
            complete_videos.add(vid)
            partial_videos.discard(vid)
            success_count += 1

        progress.set_postfix({
            "skip_ok": skipped_complete,
            "skip_failed": skipped_failed,
            "attempted": attempted_count,
            "ok": success_count,
            "failed": fail_count,
            "fallbacks": fallback_count,
        })

        if SLEEP_BETWEEN > 0:
            time.sleep(SLEEP_BETWEEN)

    print("\nDone.")
    print(f"Skipped completed: {skipped_complete}")
    print(f"Skipped known failed: {skipped_failed}")
    print(f"Attempted this run: {attempted_count}")
    print(f"Successful this run: {success_count}")
    print(f"Failed this run: {fail_count}")


if __name__ == "__main__":
    main()