import cv2
import numpy as np
from scipy.signal import find_peaks

def interframe_difference_curve_from_frames(frames):
    """Computes interframe difference curve from frames directly in case its faster"""
    if len(frames) < 2:
        return np.array([])

    H, W, C = frames[0].shape
    diffs = []

    for i in range(1, len(frames)):
        diff = interframe_difference(frames[i-1], frames[i])
        value = np.sum(diff) / (H * W)
        diffs.append(value)

    return np.array(diffs)



def interframe_difference(frame1, frame2):
    """Does infterframe difference curve by taking absolute pixel differences"""
    return np.abs(frame1.astype(np.float32) - frame2.astype(np.float32))


def interframe_difference_curve(video_path, frame_skip=1, max_frames=None, start_sec=None, end_sec=None):
    """Computes interframe difference curve for video"""
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return np.array([], dtype=np.float32), 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-6:
        fps = 25.0

    differences = []
    prev_frame = None
    end_frame = None

    if start_sec is not None:
        start_frame = int(start_sec * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    if end_sec is not None:
        end_frame = int(end_sec * fps)

    raw_idx = 0
    kept_idx = 0

    while True:
        current_frame_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if end_frame is not None and current_frame_pos >= end_frame:
            break

        ret, frame = cap.read()
        if not ret:
            break

        if raw_idx % frame_skip == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if prev_frame is not None:
                h, w = frame.shape[:2]
                diff = interframe_difference(prev_frame, frame)
                value = np.sum(diff) / (h * w)
                differences.append(value)

            prev_frame = frame
            kept_idx += 1

            if max_frames is not None and kept_idx >= max_frames:
                break

        raw_idx += 1

    cap.release()
    return np.array(differences, dtype=np.float32), kept_idx


def smooth_curve_from_values(curve, window_size=50):
    if len(curve) == 0:
        return curve

    window_size = min(window_size, len(curve))
    if window_size < 3:
        return curve

    hanning_window = np.hanning(window_size)
    smoothed = np.convolve(curve, hanning_window / hanning_window.sum(), mode="same")
    return smoothed


def detect_local_maxima(curve, n_frames=None):
    if len(curve) == 0:
        return np.array([], dtype=int)

    peaks, _ = find_peaks(curve)
    peaks = peaks + 1

    if n_frames is not None:
        peaks = peaks[peaks < n_frames]

    return peaks


def select_frame_indices(curve, n_frames, len_frames=15, window_size=50):
    if n_frames < len_frames:
        return []

    smoothed = smooth_curve_from_values(curve, window_size=window_size)
    maxima = detect_local_maxima(smoothed)

    maxima = maxima[(maxima >= 0) & (maxima < n_frames)]

    if len(maxima) < len_frames:
        return []

    fragments = []
    for i in range(len(maxima) - len_frames + 1):
        fragment_indices = maxima[i:i + len_frames]
        fragments.append(fragment_indices)

    return fragments


def load_specific_frames(video_path, target_indices, frame_skip=1, max_frames=None, start_sec=None, end_sec=None):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return {}

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-6:
        fps = 25.0

    end_frame = None

    if start_sec is not None:
        start_frame = int(start_sec * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    if end_sec is not None:
        end_frame = int(end_sec * fps)

    target_indices = sorted(set(int(i) for i in target_indices))
    target_set = set(target_indices)

    frames = {}
    raw_idx = 0
    kept_idx = 0

    while True:
        current_frame_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if end_frame is not None and current_frame_pos >= end_frame:
            break

        ret, frame = cap.read()
        if not ret:
            break

        if raw_idx % frame_skip == 0:
            if kept_idx in target_set:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames[kept_idx] = frame

                if len(frames) == len(target_indices):
                    break

            kept_idx += 1

            if max_frames is not None and kept_idx >= max_frames:
                break

        raw_idx += 1

    cap.release()
    return frames


def resize_frames(frames, target_size=(32, 32)):
    resized_frames = []
    for frame in frames:
        resized_frame = cv2.resize(frame, target_size)
        resized_frames.append(resized_frame)
    return np.array(resized_frames, dtype=np.uint8)


def build_fragments_from_indices(
    video_path,
    fragment_indices,
    frame_skip=1,
    max_frames=None,
    target_size=(32, 32),
    start_sec=None,
    end_sec=None,
):
    if len(fragment_indices) == 0:
        return np.array([], dtype=np.uint8)

    all_needed = np.concatenate(fragment_indices)
    frame_dict = load_specific_frames(
        video_path,
        all_needed,
        frame_skip=frame_skip,
        max_frames=max_frames,
        start_sec=start_sec,
        end_sec=end_sec,
    )

    fragments = []
    for frag_idxs in fragment_indices:
        frag = []
        for idx in frag_idxs:
            frame = frame_dict.get(int(idx))
            if frame is None:
                frag = []
                break
            frame = cv2.resize(frame, target_size)
            frag.append(frame)

        if len(frag) == len(frag_idxs):
            fragments.append(np.array(frag, dtype=np.uint8))

    return np.array(fragments, dtype=np.uint8)


def compute_fragments_streaming(video_path,frame_skip=1,max_frames=None,len_frames=15,target_size=(32, 32),window_size=50,start_sec=None,end_sec=None,return_metadata=False
):
    curve, n_frames = interframe_difference_curve(
        video_path,
        frame_skip=frame_skip,
        max_frames=max_frames,
        start_sec=start_sec,
        end_sec=end_sec,
    )

    fragment_indices = select_frame_indices(
        curve,
        n_frames=n_frames,
        len_frames=len_frames,
        window_size=window_size,
    )

    if len(fragment_indices) == 0:
        if return_metadata:
            return np.array([], dtype=np.uint8), []
        return np.array([], dtype=np.uint8)


    fragments = build_fragments_from_indices(
        video_path,
        fragment_indices,
        frame_skip=frame_skip,
        max_frames=max_frames,
        target_size=target_size,
        start_sec=start_sec,
        end_sec=end_sec,
    )
    if return_metadata:
        metadata = []
        base_start = start_sec if start_sec is not None else 0.0
        fps = 25.0
        for i, frag_idxs in enumerate(fragment_indices):
            frag_idxs = np.asarray(frag_idxs, dtype=int)
            frag_start_sec = base_start + float(frag_idxs[0]) * frame_skip / fps
            frag_end_sec = base_start + float(frag_idxs[-1]) * frame_skip / fps

            metadata.append({
                "frag_id": i,
                "first_frame_idx": int(frag_idxs[0]),
                "last_frame_idx": int(frag_idxs[-1]),
                "start_sec": frag_start_sec,
                "end_sec": frag_end_sec,
                "frame_indices": frag_idxs.tolist(),
            })
        return fragments, metadata

        

    return fragments