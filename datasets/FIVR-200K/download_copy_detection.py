"""
Downloads only the videos needed for copy detection evaluation on FIVR-200K.
That is: the 100 query videos + all ND and DS annotated videos.
"""

import os
import json
import argparse
from pathlib import Path

import yt_dlp
from tqdm import tqdm


LABELS = {'ND', 'DS'}
DOWNLOADED_FILE = 'downloaded_videos.txt'
MISSING_FILE = 'missing_videos.txt'


def load_id_set(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return set()
    with open(path) as f:
        return {line.strip() for line in f if line.strip()}


def get_existing_video_path(video_id, video_dir):
    video_dir = Path(video_dir)
    for ext in ('.mp4', '.mkv', '.webm', '.flv', '.mov'):
        candidate = video_dir / f'{video_id}{ext}'
        if candidate.exists():
            return candidate
    return None


def collect_needed_ids(annotation_path):
    with open(annotation_path) as f:
        ann = json.load(f)
    needed = set(ann.keys())  # 100 query videos
    for rels in ann.values():
        for label, vids in rels.items():
            if label in LABELS:
                needed.update(vids)
    return needed


def classify_download_error(error):
    message = str(error).lower()
    if "confirm you're not a bot" in message or 'bot_detection' in message:
        return 'bot_check'
    if 'rate' in message or 'too many requests' in message or '429' in message or 'quota' in message:
        return 'rate_limited'
    if 'private video' in message or 'this video is private' in message:
        return 'private'
    if 'video unavailable' in message or 'unable to extract' in message:
        return 'unavailable'
    return 'failed'


def build_ydl_opts(args):
    ydl_opts = {
        'format': 'best[height<={}][ext=mp4]/best[ext=mp4]/best[height<={}]/best'
                  .format(args.resolution, args.resolution),
        'outtmpl': '{}/%(id)s.%(ext)s'.format(args.video_dir),
        'quiet': True,
        'no_warnings': True,
    }
    if args.cookies:
        ydl_opts['cookiefile'] = args.cookies
    if args.cookies_from_browser:
        ydl_opts['cookiesfrombrowser'] = (args.cookies_from_browser,)
    return ydl_opts


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video_dir', required=True,
                        help='Directory where videos will be stored')
    parser.add_argument('-a', '--annotation', default='dataset/annotation.json',
                        help='Path to annotation.json')
    parser.add_argument('-r', '--resolution', default=480, type=int)
    parser.add_argument('--cookies', help='Path to cookies.txt')
    parser.add_argument('--cookies-from-browser',
                        choices=['brave', 'chrome', 'chromium', 'edge', 'firefox', 'opera', 'safari', 'vivaldi'])
    parser.add_argument('--status', action='store_true',
                        help='Print counts and exit')
    args = parser.parse_args()

    os.makedirs(args.video_dir, exist_ok=True)

    needed_ids = collect_needed_ids(args.annotation)
    downloaded = load_id_set(DOWNLOADED_FILE)
    missing = load_id_set(MISSING_FILE)

    pending = [
        vid for vid in needed_ids
        if vid not in downloaded
        and vid not in missing
        and get_existing_video_path(vid, args.video_dir) is None
    ]

    print(f'Needed for copy detection (ND+DS): {len(needed_ids)}')
    print(f'Already downloaded:                {len(needed_ids) - len(pending) - len(missing & needed_ids)}')
    print(f'Missing/unavailable:               {len(missing & needed_ids)}')
    print(f'Pending:                           {len(pending)}')

    if args.status:
        raise SystemExit(0)

    ydl = yt_dlp.YoutubeDL(build_ydl_opts(args))

    with open(MISSING_FILE, 'a') as missing_f, open(DOWNLOADED_FILE, 'a') as downloaded_f:
        for video_id in tqdm(pending):
            try:
                if get_existing_video_path(video_id, args.video_dir) is not None:
                    downloaded_f.write(f'{video_id}\n')
                    downloaded_f.flush()
                    continue
                url = f'https://www.youtube.com/watch?v={video_id}'
                ydl.download([url])
                downloaded_f.write(f'{video_id}\n')
                downloaded_f.flush()
            except Exception as e:
                status = classify_download_error(e)
                if status in {'rate_limited', 'bot_check'}:
                    tqdm.write(f'Rate limited on {video_id}, stopping.')
                    break
                elif video_id not in missing:
                    missing_f.write(f'{video_id}\n')
                    missing_f.flush()

    print('Done!')
