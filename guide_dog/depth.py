"""
Approximate distance from a single camera using bounding-box scale.

This is *not* true metric depth — it assumes objects fill more of the frame when
closer. Tune PERSON_SCALE_CALIBRATION for your webcam if readings feel off.
"""

from __future__ import annotations

# Tuned so a ~40%–frame-tall person reads as roughly 3–4 feet on a typical laptop webcam.
PERSON_SCALE_CALIBRATION = 2.85
OBJECT_AREA_CALIBRATION = 4.2


def bbox_height_fraction(box: tuple[int, int, int, int], frame_h: int) -> float:
    _x, _y, _w, h = box
    return h / max(frame_h, 1)


def bbox_area_fraction(box: tuple[int, int, int, int], frame_h: int, frame_w: int) -> float:
    x, y, w, h = box
    return (w * h) / max(frame_h * frame_w, 1)


def estimate_distance_feet(label: str, box: tuple[int, int, int, int], frame_h: int, frame_w: int) -> float:
    """
    Return a rough distance in feet. Person-like classes use vertical extent;
    other objects use apparent size (area).
    """
    label_l = label.lower()
    people = {"person"}
    large_objects = {"chair", "couch", "bed", "dining table", "table", "bench", "tv", "refrigerator"}

    hf = bbox_height_fraction(box, frame_h)
    af = bbox_area_fraction(box, frame_h, frame_w)

    if label_l in people:
        raw = PERSON_SCALE_CALIBRATION / max(hf, 0.07)
    elif label_l in large_objects:
        raw = OBJECT_AREA_CALIBRATION / max(af**0.45, 0.04)
    else:
        raw = OBJECT_AREA_CALIBRATION * 0.85 / max(af**0.5, 0.025)

    return float(min(max(raw, 1.0), 45.0))


def feet_to_phrase(feet: float) -> str:
    """Spoken rounding: 'about three feet', 'roughly six and a half feet'."""
    if feet < 1.5:
        return "within about a foot"
    r = round(feet * 2) / 2
    if abs(r - round(r)) < 0.01:
        n = int(round(r))
        if n == 1:
            return "about one foot away"
        return f"about {n} feet away"
    whole = int(r)
    half = " and a half" if r - whole >= 0.4 else ""
    return f"about {whole}{half} feet away"


def distance_bucket(feet: float) -> int:
    """Bucket for change-detection (6 inch resolution under 15 ft)."""
    return int(round(feet * 2))
