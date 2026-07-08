# Copyright 2019 Giorgos Kordopatis-Zilos. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Script that downloads the videos in the FIVR-200K dataset
"""

from __future__ import division
from __future__ import print_function

import os
import argparse
from pathlib import Path
import numpy as np

from multiprocessing import Pool

DOWNLOADED_FILE = 'downloaded_videos.txt'
MISSING_FILE = 'missing_videos.txt'
WORKER_ARGS = None
WORKER_YDL = None


def load_id_set(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return set()

    entries = np.loadtxt(path, dtype=str)
    entries = np.atleast_1d(entries)
    return {str(entry) for entry in entries}


def get_existing_video_path(video_id, video_dir):
    video_dir = Path(video_dir)
    for ext in ('.mp4', '.mkv', '.webm', '.flv', '.mov'):
        candidate = video_dir / f'{video_id}{ext}'
        if candidate.exists():
            return candidate
    return None


def build_pending_videos(dataset, video_dir, downloaded_videos, missing_videos, retry_missing):
    pending_videos = []
    for video in dataset:
        video = str(video)
        if video in downloaded_videos:
            continue
        if not retry_missing and video in missing_videos:
            continue
        if get_existing_video_path(video, video_dir) is not None:
            downloaded_videos.add(video)
            continue
        pending_videos.append(video)
    return pending_videos


def print_status(dataset, downloaded_videos, missing_videos, pending_videos):
    total_videos = len(dataset)
    print('Dataset size: {}'.format(total_videos))
    print('Downloaded: {}'.format(len(downloaded_videos)))
    print('Missing: {}'.format(len(missing_videos)))
    print('Pending: {}'.format(len(pending_videos)))


def classify_download_error(error):
    message = str(error).lower()
    if "confirm you're not a bot" in message or "bot_detection" in message:
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
        'outtmpl': '{}/{}.%(ext)s'.format(args.video_dir, video_id_placeholder()),
        'quiet': True,
        'no_warnings': True,
    }
    if args.cookies:
        ydl_opts['cookiefile'] = args.cookies
    if args.cookies_from_browser:
        ydl_opts['cookiesfrombrowser'] = (args.cookies_from_browser,)
    return ydl_opts


def video_id_placeholder():
    return '%(id)s'


def create_downloader(args):
    import yt_dlp

    return yt_dlp.YoutubeDL(build_ydl_opts(args))


def init_worker(args):
    global WORKER_ARGS, WORKER_YDL
    WORKER_ARGS = args
    WORKER_YDL = create_downloader(args)


def download_video(video_id, args=None, ydl=None):
    """
      Download the provided video using the yt-dlp library

      Args:
        video_id: Youtube ID of the video
        args: arguments provided by the user
      Returns:
        a flag that indicates whether there was an error during the downloading
    """
    try:
        if args is None:
            args = WORKER_ARGS
        if ydl is None:
            ydl = WORKER_YDL if args is WORKER_ARGS else create_downloader(args)

        if get_existing_video_path(video_id, args.video_dir) is not None:
            return 'exists'

        video_url = (
            'https://www.youtube.com/watch?v={}'.format(video_id)
            if 'http' not in video_id
            else video_id
        )
        ydl.download([video_url])
        return 'downloaded'
    except Exception as error:
        return classify_download_error(error)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video_dir',
                        required=True,
                        help='Directory where the downloaded videos will be stored')
    parser.add_argument('-d', '--dataset_ids',
                        default='dataset/youtube_ids.txt',
                        help='File that contains the Youtube IDs of the videos in FIVR-200K dataset')
    parser.add_argument('-c', '--cores',
                        default=8, type=int,
                        help='Number of cores that will be used for downloading.')
    parser.add_argument('-r', '--resolution',
                        default=480, type=int,
                        help='Preferred resolution to download videos.')
    parser.add_argument('--cookies',
                        help='Path to a cookies.txt file exported for yt-dlp.')
    parser.add_argument('--cookies-from-browser',
                        choices=['brave', 'chrome', 'chromium', 'edge', 'firefox', 'opera', 'safari', 'vivaldi'],
                        help='Read YouTube cookies directly from a supported browser.')
    parser.add_argument('--retry-missing',
                        action='store_true',
                        help='Retry IDs already listed in missing_videos.txt.')
    parser.add_argument('--status',
                        action='store_true',
                        help='Print downloaded/missing/pending counts and exit.')
    args = parser.parse_args()

    # create root dir
    if not os.path.exists(args.video_dir):
        os.makedirs(args.video_dir)

    dataset = np.loadtxt(args.dataset_ids, dtype=str)
    dataset = np.atleast_1d(dataset)

    missing_videos = load_id_set(MISSING_FILE)
    downloaded_videos = load_id_set(DOWNLOADED_FILE)
    pending_videos = build_pending_videos(
        dataset,
        args.video_dir,
        downloaded_videos,
        missing_videos,
        args.retry_missing,
    )

    if args.status:
        print_status(dataset, downloaded_videos, missing_videos, pending_videos)
        raise SystemExit(0)

    from tqdm import tqdm

    with open(MISSING_FILE, 'a') as missing_f, open(DOWNLOADED_FILE, 'a') as downloaded_f:
        progress_bar = tqdm(pending_videos)
        if args.cores > 1:
            future = []
            pool = Pool(args.cores, initializer=init_worker, initargs=(args,))
            for video in pending_videos:
                future.append(pool.apply_async(download_video, args=[video],
                                               callback=(lambda *a: progress_bar.update())))

            for video, flag in zip(pending_videos, future):
                status = flag.get()
                if status in {'downloaded', 'exists'}:
                    if video not in downloaded_videos:
                        downloaded_f.write('{}\n'.format(video))
                        downloaded_f.flush()
                        downloaded_videos.add(video)
                elif status in {'rate_limited', 'bot_check'}:
                    print('\nRate limited by YouTube, stopping.')
                    pool.terminate()
                    raise SystemExit(1)
                elif video not in missing_videos:
                    missing_f.write('{}\n'.format(video))
                    missing_f.flush()
                    missing_videos.add(video)
            pool.terminate()
        else:
            downloader = create_downloader(args)
            for video in progress_bar:
                status = download_video(video, args, downloader)
                if status in {'downloaded', 'exists'}:
                    if video not in downloaded_videos:
                        downloaded_f.write('{}\n'.format(video))
                        downloaded_f.flush()
                        downloaded_videos.add(video)
                elif status in {'rate_limited', 'bot_check'}:
                    tqdm.write('Rate limited by YouTube, stopping.')
                    raise SystemExit(1)
                elif video not in missing_videos:
                    missing_f.write('{}\n'.format(video))
                    missing_f.flush()
                    missing_videos.add(video)
    print('Downloading completed!')
