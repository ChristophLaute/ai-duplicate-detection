"""
Downloads FIVR-200K videos from temporary URLs provided by the dataset authors
using aria2c. Skips videos already on disk so aria2c never sees a file-exists error.
Respects the authors' request of max 4 parallel connections.

Usage:
    python download_from_urls.py -v /Volumes/LaCie/FIVR-200K/videos -u temporary_urls.txt
"""

import os
import subprocess
import argparse
from pathlib import Path

DOWNLOADED_FILE = 'downloaded_videos.txt'
VIDEO_EXTENSIONS = ('.mp4', '.mkv', '.webm', '.flv', '.mov')


def load_id_set(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return set()
    with open(path) as f:
        return {line.strip() for line in f if line.strip()}


def exists_on_disk(video_id, video_dir):
    for ext in VIDEO_EXTENSIONS:
        if (video_dir / f'{video_id}{ext}').exists():
            return True
    return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video_dir', required=True,
                        help='Directory where videos will be stored')
    parser.add_argument('-u', '--urls_file', required=True,
                        help='Path to temporary_urls.txt from the authors')
    parser.add_argument('-c', '--cores', default=4, type=int,
                        help='Parallel downloads (authors request max 4)')
    parser.add_argument('--status', action='store_true',
                        help='Print counts and exit without downloading')
    args = parser.parse_args()

    if args.cores > 4:
        print('WARNING: Authors requested max 4 threads. Capping at 4.')
        args.cores = 4

    video_dir = Path(args.video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)

    with open(args.urls_file) as f:
        urls = [line.strip() for line in f if line.strip()]

    video_ids = [Path(url).stem for url in urls]
    downloaded = load_id_set(DOWNLOADED_FILE)

    pending = []
    already_have = 0
    for video_id, url in zip(video_ids, urls):
        if video_id in downloaded or exists_on_disk(video_id, video_dir):
            already_have += 1
        else:
            pending.append((video_id, url))

    print(f'Total in URL list:   {len(urls)}')
    print(f'Already on disk:     {already_have}')
    print(f'Pending download:    {len(pending)}')

    if args.status or not pending:
        raise SystemExit(0)

    # Write filtered aria2 input file
    input_file = video_dir / '_aria2_input.txt'
    with open(input_file, 'w') as f:
        for video_id, url in pending:
            f.write(f'{url}\n')
            f.write(f'  out={video_id}.mp4\n')

    cmd = [
        'aria2c',
        f'--dir={video_dir}',
        f'--input-file={input_file}',
        f'--max-concurrent-downloads={args.cores}',
        '--max-connection-per-server=1',
        '--split=1',
        '--continue=true',
        '--allow-overwrite=false',
        '--auto-file-renaming=false',
        '--console-log-level=warn',
        '--summary-interval=30',
    ]
    print(f'Starting aria2c for {len(pending)} videos ...')
    result = subprocess.run(cmd)

    # Log newly downloaded videos
    newly_done = [vid for vid, _ in pending
                  if exists_on_disk(vid, video_dir) and vid not in downloaded]
    if newly_done:
        with open(DOWNLOADED_FILE, 'a') as f:
            f.writelines(f'{vid}\n' for vid in newly_done)
        print(f'Logged {len(newly_done)} newly downloaded videos to {DOWNLOADED_FILE}.')

    input_file.unlink(missing_ok=True)
    print('Done!')
    raise SystemExit(result.returncode)
