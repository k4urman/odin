"""
Optional YOLO-pose keypoints for finer cues (raised arms / hands low).
Runs throttled — heavy compared to frame differencing.
"""

from __future__ import annotations

from typing import Any

import numpy as np

KP_L_SH, KP_R_SH = 5, 6
KP_L_WR, KP_R_WR = 9, 10
KP_L_HIP, KP_R_HIP = 11, 12


def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(area_a + area_b - inter, 1e-6)


def _xyxy_from_xywh(box: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = box
    return np.array([x, y, x + w, y + h], dtype=np.float32)


class PoseHintEstimator:
    def __init__(self, model_name: str = "yolo11n-pose.pt"):
        from ultralytics import YOLO

        print(f"[Odin] Loading pose model {model_name}...")
        self.model = YOLO(model_name)
        self._prev_wy: list[float | None] = []
        print("[Odin] Pose model ready.")

    def estimate(
        self,
        frame: np.ndarray,
        person_boxes_xywh: list[tuple[int, int, int, int]],
        table_boxes_xywh: list[tuple[int, int, int, int]],
    ) -> list[dict[str, Any]]:
        if not person_boxes_xywh:
            return []

        if len(self._prev_wy) != len(person_boxes_xywh):
            self._prev_wy = [None] * len(person_boxes_xywh)

        results = self.model(frame, verbose=False)[0]
        if results.keypoints is None or results.boxes is None or len(results.boxes) == 0:
            return [{} for _ in person_boxes_xywh]

        kpts_xy = results.keypoints.xy.cpu().numpy()
        kpts_conf = None
        if hasattr(results.keypoints, "conf") and results.keypoints.conf is not None:
            kpts_conf = results.keypoints.conf.cpu().numpy()

        boxes_xyxy = results.boxes.xyxy.cpu().numpy()
        out: list[dict[str, Any]] = [{} for _ in person_boxes_xywh]

        for pi, pbox in enumerate(person_boxes_xywh):
            p_xyxy = _xyxy_from_xywh(pbox)
            best_i, best_iou = -1, 0.0
            for i in range(len(boxes_xyxy)):
                iou = _iou_xyxy(boxes_xyxy[i], p_xyxy)
                if iou > best_iou:
                    best_iou, best_i = iou, i
            if best_i < 0 or best_iou < 0.15:
                continue

            xy = kpts_xy[best_i]
            conf = kpts_conf[best_i] if kpts_conf is not None else np.ones(17)

            def ok(idx: int) -> bool:
                return conf[idx] > 0.35 and xy[idx, 0] > 0 and xy[idx, 1] > 0

            tags: dict[str, Any] = {}
            shoulders_y = []
            if ok(KP_L_SH):
                shoulders_y.append(float(xy[KP_L_SH, 1]))
            if ok(KP_R_SH):
                shoulders_y.append(float(xy[KP_R_SH, 1]))
            wrists_y = []
            if ok(KP_L_WR):
                wrists_y.append(float(xy[KP_L_WR, 1]))
            if ok(KP_R_WR):
                wrists_y.append(float(xy[KP_R_WR, 1]))
            hips_y = []
            if ok(KP_L_HIP):
                hips_y.append(float(xy[KP_L_HIP, 1]))
            if ok(KP_R_HIP):
                hips_y.append(float(xy[KP_R_HIP, 1]))

            if shoulders_y and wrists_y:
                mean_sh = float(np.mean(shoulders_y))
                mean_wr = float(np.mean(wrists_y))
                if mean_wr < mean_sh - 12:
                    tags["wrists_elevated"] = True
                if mean_wr > mean_sh + 25:
                    tags["hands_low"] = True

            if wrists_y:
                mean_wy = float(np.mean(wrists_y))
                prev = self._prev_wy[pi]
                if prev is not None:
                    dy = abs(mean_wy - prev)
                    if dy > 18 and tags.get("wrists_elevated"):
                        tags["possible_wave"] = True
                self._prev_wy[pi] = mean_wy

            px, py, pw, ph = pbox
            person_bottom = py + ph
            for tbox in table_boxes_xywh:
                tx, ty, tw, th = tbox
                table_mid_y = ty + th * 0.35
                if wrists_y:
                    mw = float(np.mean(wrists_y))
                    if (
                        mw >= table_mid_y - 25
                        and mw <= ty + th + 15
                        and abs((px + pw / 2) - (tx + tw / 2)) < (pw + tw) * 0.65
                        and person_bottom >= ty
                    ):
                        tags["possible_hands_on_surface"] = True

            if hips_y and wrists_y and not tags.get("wrists_elevated"):
                mean_h = float(np.mean(hips_y))
                mean_w = float(np.mean(wrists_y))
                if mean_w > mean_h - 20:
                    tags["hands_near_lap"] = True

            out[pi] = tags

        return out
