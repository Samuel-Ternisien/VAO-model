from __future__ import annotations
import numpy as np
from typing import List, Tuple

def frame_to_label_array(num_frames: int, segments: List[Tuple[int,int,int]]) -> np.ndarray:
    """
    Creates a label array frame_labels: (num_frames,) with 0 = unknown
    segments: (start_frame, end_frame, class_id) where end_frame is exclusive or inclusive?
    In your Events, segments are consecutive (end == next start). We treat end as EXCLUSIVE.
    """
    y = np.zeros((num_frames,), dtype=np.int32)
    for s, e, c in segments:
        s = max(0, int(s))
        e = min(num_frames, int(e))
        if e > s:
            y[s:e] = int(c)
    return y

def make_windows(X: np.ndarray, y_frame: np.ndarray, win: int, stride: int, purity: float):
    """
    X: (T, F)
    y_frame: (T,) labels 0..C
    Returns:
      windows_X: (N, win, F)
      windows_y: (N,)
    """
    T = X.shape[0]
    xs = []
    ys = []
    for start in range(0, T - win + 1, stride):
        end = start + win
        yf = y_frame[start:end]
        # ignore windows with too much unknown
        valid = yf[yf > 0]
        if len(valid) == 0:
            continue
        # majority class among valid frames
        cls, counts = np.unique(valid, return_counts=True)
        idx = np.argmax(counts)
        maj = int(cls[idx])
        maj_ratio = counts[idx] / float(win)
        if maj_ratio >= purity:
            xs.append(X[start:end])
            ys.append(maj)
    if not xs:
        return np.empty((0, win, X.shape[1]), dtype=np.float32), np.empty((0,), dtype=np.int32)
    return np.stack(xs).astype(np.float32), np.array(ys, dtype=np.int32)