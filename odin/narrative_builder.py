"""
Rich, spoken-style phrases from enriched detections (distance, motion, pose).
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from odin.depth import distance_bucket, estimate_distance_feet, feet_to_phrase
from odin.narration import SceneFingerprint, pose_tag_signature


def lighting_from_frame(frame_bgr: np.ndarray) -> tuple[str, str]:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    mean_lum = float(np.mean(gray)) / 255.0
    if mean_lum < 0.2:
        return "The scene looks quite dark.", "dark"
    if mean_lum < 0.42:
        return "Lighting is on the dim side.", "dim"
    if mean_lum > 0.78:
        return "The view is very bright.", "bright"
    return "Lighting looks fairly even.", "normal"


def region_phrase(cx: float, frame_w: int) -> tuple[str, str]:
    if cx < frame_w / 3:
        return "toward your left", "L"
    if cx < 2 * frame_w / 3:
        return "in the center of the view", "C"
    return "toward your right", "R"


def motion_fingerprint_tag(motion: str) -> str:
    if motion in ("still", "subtle_motion"):
        return "calm"
    if motion == "arms_moving":
        return "arms"
    return "active"


def person_activity_phrase(motion: str, pose: dict[str, Any]) -> str:
    p = pose or {}
    if p.get("possible_wave"):
        return "and their arms suggest they may be waving"
    if p.get("possible_hands_on_surface"):
        return "with hands that may be resting on a surface such as a table"
    if p.get("wrists_elevated") and p.get("hands_near_lap"):
        return "with arms in a mixed position"
    if p.get("wrists_elevated"):
        return "with arms raised or reaching upward"
    if p.get("hands_near_lap"):
        return "with hands lowered near lap height"
    if p.get("hands_low"):
        return "with hands held low"

    if motion == "arms_moving":
        return "with noticeable arm movement—possibly gesturing or waving"
    if motion == "lower_body_moving":
        return "with movement lower in the frame, as if shifting or walking"
    if motion == "moving":
        return "with visible body movement"
    if motion == "subtle_motion":
        return "with only slight movement"
    if motion == "still":
        return "appearing relatively still"
    return ""


def object_clause(
    label: str,
    distance_phrase: str,
    region_spoken: str,
    is_person: bool,
    motion: str | None,
    pose: dict[str, Any],
) -> str:
    label_nice = label.replace("_", " ")
    if is_person:
        base = f"someone roughly {distance_phrase}, positioned {region_spoken}"
        act = person_activity_phrase(motion or "still", pose)
        if act:
            return f"{base}, {act}"
        return base
    return f"a {label_nice} roughly {distance_phrase}, {region_spoken}"


def compose_narration(
    enriched: list[dict[str, Any]],
    lighting_phrase: str,
) -> str:
    if not enriched:
        return f"I am not picking up labeled objects clearly right now. {lighting_phrase}"

    ordered = sorted(enriched, key=lambda e: e["distance_ft"])
    clauses: list[str] = []
    for e in ordered:
        clauses.append(
            object_clause(
                e["label"],
                e["distance_phrase"],
                e["region_spoken"],
                e["is_person"],
                e.get("motion"),
                e.get("pose") or {},
            )
        )

    if len(clauses) == 1:
        body = clauses[0][0].upper() + clauses[0][1:]
    else:
        joined = "; ".join(clauses[:-1]) + f"; and {clauses[-1]}"
        body = joined[0].upper() + joined[1:]

    return f"{body}. {lighting_phrase}"


def build_enriched_detections(
    detections: list[dict[str, Any]],
    frame_shape: tuple[int, int, int],
    person_motion: list[str],
    person_pose: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    fh, fw = frame_shape[0], frame_shape[1]
    p_idx = 0
    out: list[dict[str, Any]] = []
    for d in detections:
        box = d["box"]
        x, y, bw, bh = box
        cx = x + bw / 2
        label = d["label"]
        region_spoken, region_key = region_phrase(cx, fw)
        feet = estimate_distance_feet(label, box, fh, fw)
        is_person = label.lower() == "person"
        motion = None
        pose: dict[str, Any] = {}
        if is_person:
            motion = person_motion[p_idx] if p_idx < len(person_motion) else "still"
            pose = dict(person_pose[p_idx]) if p_idx < len(person_pose) else {}
            p_idx += 1
        out.append(
            {
                "label": label,
                "confidence": d["confidence"],
                "box": box,
                "distance_ft": feet,
                "distance_phrase": feet_to_phrase(feet),
                "dist_bucket": distance_bucket(feet),
                "region_spoken": region_spoken,
                "region_key": region_key,
                "is_person": is_person,
                "motion": motion,
                "pose": pose,
            }
        )
    return out


def format_enriched_for_llm(enriched: list[dict[str, Any]]) -> str:
    if not enriched:
        return "No objects detected above the confidence threshold."
    rows = []
    for e in enriched:
        bits = [
            f"{e['label']}",
            e["distance_phrase"],
            f"region={e['region_key']}",
        ]
        if e["is_person"]:
            bits.append(f"motion={e.get('motion')}")
            if e.get("pose"):
                bits.append(f"pose_hints={e['pose']}")
        rows.append(" | ".join(bits))
    return "\n".join(rows)


def build_scene_fingerprint(enriched: list[dict[str, Any]], lighting_bucket: str) -> SceneFingerprint:
    items = []
    for e in enriched:
        motion_fp = (
            motion_fingerprint_tag(e.get("motion") or "still") if e["is_person"] else ""
        )
        pose_sig = pose_tag_signature(e.get("pose") or {}) if e["is_person"] else ""
        items.append(
            (
                e["label"].lower(),
                e["region_key"],
                e["dist_bucket"],
                motion_fp,
                pose_sig,
            )
        )
    return SceneFingerprint(items=tuple(sorted(items)), lighting=lighting_bucket)
