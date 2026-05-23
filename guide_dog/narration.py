"""
De-duplicate spoken updates: only narrate when the scene fingerprint changes.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SceneFingerprint:
    items: tuple[tuple[str, str, int, str, str], ...]
    lighting: str


class NarrationGate:
    def __init__(self) -> None:
        self._last: SceneFingerprint | None = None

    def is_new(self, fp: SceneFingerprint) -> bool:
        if self._last is None:
            self._last = fp
            return True
        if fp == self._last:
            return False
        self._last = fp
        return True

    def reset(self) -> None:
        self._last = None


def pose_tag_signature(tags: dict) -> str:
    if not tags:
        return ""
    parts = []
    if tags.get("possible_wave"):
        parts.append("wave")
    elif tags.get("wrists_elevated"):
        parts.append("arms_up")
    if tags.get("possible_hands_on_surface"):
        parts.append("hands_surface")
    elif tags.get("hands_near_lap"):
        parts.append("lap")
    elif tags.get("hands_low"):
        parts.append("hands_low")
    return "|".join(sorted(parts))
