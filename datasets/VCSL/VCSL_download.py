#!/usr/bin/env python3
import csv
import subprocess
import time
from pathlib import Path

# -------- settings --------
CSV_PATH = Path("data/videos_url_uuid.csv")
OUTPUT_DIR = Path("vcsl_videos")
ARCHIVE_FILE = Path("downloaded_ids.txt")
FAILED_FILE = Path("failed_downloads.csv")
SLEEP_BETWEEN = 1.0   # seconds between downloads
# -------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def run_cmd(cmd):
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return result.returncode, result.stdout, result.stderr

def try_download(url: str, outtmpl: str):
    # First try normal download
    cmd = [
        "yt-dlp",
        "--no-playlist",
        "--ignore-errors",
        "--no-overwrites",
        "--continue",
        "--restrict-filenames",
        "-o", outtmpl,
        url,
    ]
    return run_cmd(cmd)

def try_download_fallback(url: str, outtmpl: str):
    # Fallback: ask yt-dlp for a simpler mp4 when possible
    cmd = [
        "yt-dlp",
        "--no-playlist",
        "--ignore-errors",
        "--no-overwrites",
        "--continue",
        "--restrict-filenames",
        "-f", "mp4/best",
        "-o", outtmpl,
        url,
    ]
    return run_cmd(cmd)

def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    rows = []
    with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print("No rows found in CSV.")
        return

    # Try to infer column names
    fieldnames = set(rows[0].keys())
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

    if uuid_col is None or url_col is None:
        raise ValueError(
            f"Could not infer uuid/url columns from CSV headers: {sorted(fieldnames)}"
        )

    print(f"Using uuid column: {uuid_col}")
    print(f"Using url column:  {url_col}")
    print(f"Found {len(rows)} rows")

    failed = []

    for i, row in enumerate(rows, start=1):
        vid = str(row[uuid_col]).strip()
        url = str(row[url_col]).strip()

        if not vid or not url:
            failed.append({"uuid": vid, "url": url, "reason": "missing_uuid_or_url"})
            continue

        # We let yt-dlp choose extension automatically
        outtmpl = str(OUTPUT_DIR / f"{vid}.%(ext)s")

        # Skip if already downloaded in any common format
        already = list(OUTPUT_DIR.glob(f"{vid}.*"))
        if any(p.suffix.lower() in {".mp4", ".mkv", ".webm", ".flv", ".mov"} for p in already):
            print(f"[{i}/{len(rows)}] Skip existing: {vid}")
            continue

        print(f"[{i}/{len(rows)}] Downloading {vid}")
        code, out, err = try_download(url, outtmpl)

        if code != 0:
            print(f"  normal attempt failed for {vid}, trying fallback...")
            code, out, err = try_download_fallback(url, outtmpl)

        if code != 0:
            reason = (err or out or "unknown_error").strip().replace("\n", " | ")
            print(f"  FAILED: {vid}")
            failed.append({"uuid": vid, "url": url, "reason": reason[:1000]})
        else:
            with ARCHIVE_FILE.open("a", encoding="utf-8") as af:
                af.write(vid + "\n")
            print(f"  OK: {vid}")

        time.sleep(SLEEP_BETWEEN)

    if failed:
        with FAILED_FILE.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["uuid", "url", "reason"])
            writer.writeheader()
            writer.writerows(failed)
        print(f"\nDone. Failed downloads saved to: {FAILED_FILE}")
    else:
        print("\nDone. No failures.")

if __name__ == "__main__":
    main()