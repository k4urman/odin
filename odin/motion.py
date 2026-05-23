"""
Region motion from frame differencing on a downscaled grayscale stream.
Cheap cues for 'waving' vs 'still' without running pose every frame.
"""

from __future__ import annotations

import cv2
import numpy as np


class MotionAnalyzer:
    def __init__(self, width: int = 320):
        self.width = width
        self._prev: np.ndarray | None = None

    def _small_gray(self, frame_bgr: np.ndarray) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        nw = self.width
        nh = max(1, int(h * (nw / max(w, 1))))
        return cv2.cvtColor(cv2.resize(frame_bgr, (nw, nh)), cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _map_box_to_small(
        box: tuple[int, int, int, int], frame_w: int, frame_h: int, sw: int, sh: int
    ) -> tuple[int, int, int, int]:
        x, y, bw, bh = box
        sx = int(x * sw / frame_w)
        sy = int(y * sh / frame_h)
        sw_b = max(2, int(bw * sw / frame_w))
        sh_b = max(2, int(bh * sh / frame_h))
        sx = min(max(0, sx), sw - 2)
        sy = min(max(0, sy), sh - 2)
        sw_b = min(sw_b, sw - sx)
        sh_b = min(sh_b, sh - sy)
        return sx, sy, sw_b, sh_b

    def tick(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray | None, int, int]:
        g = self._small_gray(frame_bgr)
        prev = self._prev
        self._prev = g
        return g, prev, g.shape[1], g.shape[0]

    def roi_motion(
        self,
        box: tuple[int, int, int, int],
        frame_shape: tuple[int, int, int],
        prev_small: np.ndarray | None,
        curr_small: np.ndarray,
        sw: int,
        sh: int,
    ) -> tuple[float, float, float]:
        fh, fw = frame_shape[0], frame_shape[1]
        sx, sy, sw_b, sh_b = self._map_box_to_small(box, fw, fh, sw, sh)
        cur = curr_small[sy : sy + sh_b, sx : sx + sw_b]
        if prev_small is None or cur.size == 0:
            return 0.0, 0.0, 0.0
        prev = prev_small[sy : sy + sh_b, sx : sx + sw_b]
        if cur.shape != prev.shape:
            return 0.0, 0.0, 0.0
        diff = cv2.absdiff(cur, prev).astype(np.float32) / 255.0
        mid = diff.shape[0] // 2
        if mid < 1:
            m = float(np.mean(diff))
            return m, m, m
        upper = diff[:mid, :]
        lower = diff[mid:, :]
        return float(np.mean(upper)), float(np.mean(lower)), float(np.mean(diff))


def classify_person_motion(upper: float, lower: float, whole: float) -> str:
    if whole < 0.012:
        return "still"
    if upper > 0.035 and upper > lower * 1.9:
        return "arms_moving"
    if whole > 0.028:
        return "moving"
    if lower > upper * 1.6:
        return "lower_body_moving"
    return "subtle_motion"
