#!/usr/bin/env python3
import csv
import subprocess
import time
from pathlib import Path
from tqdm import tqdm
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# -------- settings --------
BASE_DIR = Path(__file__).resolve().parent

CSV_PATH = BASE_DIR.parent / "VCSL" / "data" / "videos_url_uuid.csv"
OUTPUT_DIR = Path("/Volumes/LaCie/DuplicateDetectionThesis/VCSL/vcsl_videos")
ARCHIVE_FILE = BASE_DIR / "downloaded_ids.txt"
FAILED_FILE = BASE_DIR / "failed_downloads.csv"
RESUME_FILE = BASE_DIR / "resume_state.txt"

SLEEP_BETWEEN = 0.0
VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".flv", ".mov", ".avi", ".m4v"}
PARTIAL_EXTS = {".part", ".ytdl"}

SUBPROCESS_TIMEOUT = 120
SOCKET_TIMEOUT = "20"
RETRIES = "5"
FRAGMENT_RETRIES = "5"
MAX_WORKERS = 4
# -------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
state_lock = threading.Lock()


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
        "--socket-timeout", SOCKET_TIMEOUT,
        "--retries", RETRIES,
        "--fragment-retries", FRAGMENT_RETRIES,
        "--concurrent-fragments", "4",
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
        "--socket-timeout", SOCKET_TIMEOUT,
        "--retries", RETRIES,
        "--fragment-retries", FRAGMENT_RETRIES,
        "--concurrent-fragments", "4",
        "-f", "mp4/best",
        "-o", outtmpl,
        url,
    ]
    return run_cmd(cmd)


def normalize_reason(stdout_text: str, stderr_text: str) -> str:
    msg = f"{stdout_text}\n{stderr_text}".strip().replace("\n", " | ")
    return msg[:1000] if msg else "unknown_error"


def classify_failure(reason: str) -> str:
    r = (reason or "").lower()

    permanent_patterns = [
        "video unavailable",
        "private video",
        "this video is private",
        "404",
        "not found",
        "removed",
        "deleted",
        "sign in to confirm your age",
    ]

    retryable_patterns = [
        "timeout after",
        "timed out",
        "requested format is not available",
        "format not available",
        "unsupported format",
        "unable to download webpage",
        "http error 403",
        "forbidden",
        "403",
        "geo",
        "country",
        "temporarily unavailable",
        "connection",
        "reset by peer",
        "network",
        "ssl",
    ]

    if any(p in r for p in permanent_patterns):
        return "permanent"

    if any(p in r for p in retryable_patterns):
        return "retryable"

    return "retryable"


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


def load_failed_statuses():
    failed_statuses = {}

    if not FAILED_FILE.exists():
        return failed_statuses

    try:
        with FAILED_FILE.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                vid = str(row.get("uuid", "")).strip()
                reason = str(row.get("reason", "")).strip()
                status = str(row.get("status", "")).strip()
                url = str(row.get("url", "")).strip()

                if not vid:
                    continue

                if status not in {"permanent", "retryable"}:
                    status = classify_failure(reason)

                failed_statuses[vid] = {
                    "url": url,
                    "reason": reason,
                    "status": status,
                }
    except Exception:
        pass

    return failed_statuses


def write_failed_statuses(failed_statuses):
    with FAILED_FILE.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["uuid", "url", "reason", "status"])
        writer.writeheader()
        for vid, info in failed_statuses.items():
            writer.writerow({
                "uuid": vid,
                "url": info.get("url", ""),
                "reason": info.get("reason", "")[:1000],
                "status": info.get("status", "retryable"),
            })


def load_resume_index(total_rows: int) -> int:
    if total_rows <= 0 or not RESUME_FILE.exists():
        return 0
    try:
        idx = int(RESUME_FILE.read_text(encoding="utf-8").strip())
        if idx < 0:
            return 0
        return idx % total_rows
    except Exception:
        return 0


def save_resume_index(idx: int):
    RESUME_FILE.write_text(str(idx), encoding="utf-8")


def ask_for_initial_pointer(total_rows: int) -> int:
    print("\nNo saved resume pointer found.")
    print(f"Valid pointer range: 0 to {total_rows - 1}")
    print("Press Enter to start from 0, or type a row index to start from there.")
    raw = input("Start pointer: ").strip()

    if raw == "":
        return 0

    try:
        idx = int(raw)
    except ValueError:
        print("Invalid input. Starting from 0.")
        return 0

    if idx < 0 or idx >= total_rows:
        print("Pointer out of range. Starting from 0.")
        return 0

    return idx


def parse_args():
    parser = argparse.ArgumentParser(description="Resume-capable parallel VCSL downloader")
    parser.add_argument(
        "--start-pointer",
        type=int,
        default=None,
        help="Override resume pointer with a CSV row index (0-based).",
    )
    parser.add_argument(
        "--reset-pointer",
        action="store_true",
        help="Ignore saved pointer and ask for a new one.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=MAX_WORKERS,
        help="Number of parallel downloads to run.",
    )
    return parser.parse_args()


def process_row(
    task,
    uuid_col,
    url_col,
    complete_videos,
    partial_videos,
    permanent_failed_ids,
    retryable_failed_ids,
    failed_statuses,
):
    original_idx, row = task

    vid = str(row.get(uuid_col, "")).strip()
    url = str(row.get(url_col, "")).strip()

    result = {
        "original_idx": original_idx,
        "vid": vid,
        "url": url,
        "final_status": None,
        "used_fallback": False,
        "failure_reason": "",
        "failure_class": "",
        "was_retryable_retry": False,
    }

    if not vid or not url:
        result["final_status"] = "bad_row"
        return result

    with state_lock:
        if vid in complete_videos:
            result["final_status"] = "skip_complete"
            return result

        if vid in permanent_failed_ids:
            result["final_status"] = "skip_permanent"
            return result

        if vid in retryable_failed_ids:
            result["was_retryable_retry"] = True

    outtmpl = str(OUTPUT_DIR / f"{vid}.%(ext)s")

    code, out, err = try_download(url, outtmpl)

    if code != 0 and should_try_fallback(err, out):
        result["used_fallback"] = True
        code, out, err = try_download_fallback(url, outtmpl)

    with state_lock:
        if code != 0:
            reason = normalize_reason(out, err)
            status = classify_failure(reason)

            failed_statuses[vid] = {
                "url": url,
                "reason": reason,
                "status": status,
            }

            if status == "permanent":
                permanent_failed_ids.add(vid)
                retryable_failed_ids.discard(vid)
            else:
                retryable_failed_ids.add(vid)
                permanent_failed_ids.discard(vid)

            result["final_status"] = "failed"
            result["failure_reason"] = reason
            result["failure_class"] = status
        else:
            complete_videos.add(vid)
            partial_videos.discard(vid)
            failed_statuses.pop(vid, None)
            retryable_failed_ids.discard(vid)
            permanent_failed_ids.discard(vid)
            result["final_status"] = "ok"

    return result


def main():
    args = parse_args()

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

    print(f"Using uuid column: {uuid_col}")
    print(f"Using url column:  {url_col}")
    print(f"Found {len(rows)} rows")
    print(f"CSV path:          {CSV_PATH}")
    print(f"Output dir:        {OUTPUT_DIR}")
    print(f"Parallel workers:  {args.workers}")

    complete_videos, partial_videos = build_existing_indexes()
    failed_statuses = load_failed_statuses()

    permanent_failed_ids = {
        vid for vid, info in failed_statuses.items() if info["status"] == "permanent"
    }
    retryable_failed_ids = {
        vid for vid, info in failed_statuses.items() if info["status"] == "retryable"
    }

    print(f"Already completed:      {len(complete_videos)}")
    print(f"Partial files:          {len(partial_videos)}")
    print(f"Known permanent failed: {len(permanent_failed_ids)}")
    print(f"Known retryable failed: {len(retryable_failed_ids)}")

    total_rows = len(rows)

    if args.start_pointer is not None:
        resume_idx = args.start_pointer % total_rows
        print(f"Using command-line start pointer: {resume_idx}")
    elif args.reset_pointer:
        resume_idx = ask_for_initial_pointer(total_rows)
        print(f"Using manually entered pointer:   {resume_idx}")
    elif RESUME_FILE.exists():
        resume_idx = load_resume_index(total_rows)
        print(f"Loaded saved start pointer:       {resume_idx}")
    else:
        resume_idx = ask_for_initial_pointer(total_rows)
        print(f"Using manually entered pointer:   {resume_idx}")

    save_resume_index(resume_idx)

    indexed_rows = list(enumerate(rows))
    indexed_rows = indexed_rows[resume_idx:] + indexed_rows[:resume_idx]

    tasks_to_run = []
    pre_skipped_complete = 0
    pre_skipped_permanent = 0
    pre_bad_rows = 0

    for original_idx, row in indexed_rows:
        vid = str(row.get(uuid_col, "")).strip()
        url = str(row.get(url_col, "")).strip()

        if not vid or not url:
            pre_bad_rows += 1
            continue

        if vid in complete_videos:
            pre_skipped_complete += 1
            continue

        if vid in permanent_failed_ids:
            pre_skipped_permanent += 1
            continue

        tasks_to_run.append((original_idx, row))

    print("First 5 original row indices in this run:", [idx for idx, _ in tasks_to_run[:5]])
    print(f"Pre-skipped completed:            {pre_skipped_complete}")
    print(f"Pre-skipped permanent failed:     {pre_skipped_permanent}")
    print(f"Pre-skipped bad rows:             {pre_bad_rows}")
    print(f"Tasks that still need action:     {len(tasks_to_run)}")

    skipped_complete = pre_skipped_complete
    skipped_permanent_failed = pre_skipped_permanent
    retried_failed = 0
    attempted_count = 0
    success_count = 0
    fail_count = 0
    fallback_count = 0
    bad_row_count = pre_bad_rows

    progress = tqdm(
        total=len(tasks_to_run),
        desc=f"Downloading remaining videos (scan@{resume_idx})",
        unit="video",
    )

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                process_row,
                task,
                uuid_col,
                url_col,
                complete_videos,
                partial_videos,
                permanent_failed_ids,
                retryable_failed_ids,
                failed_statuses,
            )
            for task in tasks_to_run
        ]

        for future in as_completed(futures):
            result = future.result()
            status = result["final_status"]

            if status == "skip_complete":
                skipped_complete += 1
            elif status == "skip_permanent":
                skipped_permanent_failed += 1
            elif status == "bad_row":
                bad_row_count += 1
            elif status == "ok":
                attempted_count += 1
                success_count += 1
            elif status == "failed":
                attempted_count += 1
                fail_count += 1

            if result["was_retryable_retry"]:
                retried_failed += 1

            if result["used_fallback"]:
                fallback_count += 1

            progress.update(1)
            done = progress.n
            left = progress.total - progress.n
            progress.set_postfix({
                "done": done,
                "left": left,
                "ok": success_count,
                "fail": fail_count,
                "retry": retried_failed,
                "fb": fallback_count,
            })

            if SLEEP_BETWEEN > 0:
                time.sleep(SLEEP_BETWEEN)

    progress.close()

    write_failed_statuses(failed_statuses)

    next_start = (resume_idx + len(indexed_rows)) % total_rows
    save_resume_index(next_start)

    print("\nDone.")
    print(f"Skipped completed: {skipped_complete}")
    print(f"Skipped permanent failed: {skipped_permanent_failed}")
    print(f"Retried old retryable failures: {retried_failed}")
    print(f"Attempted this run: {attempted_count}")
    print(f"Successful this run: {success_count}")
    print(f"Failed this run: {fail_count}")
    print(f"Bad rows: {bad_row_count}")
    print(f"Saved next start pointer to: {RESUME_FILE}")


if __name__ == "__main__":
    main()