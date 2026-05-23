from __future__ import annotations

import json
import os
import queue
import threading
import tempfile
import time

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import cv2
import numpy as np
import pygame
import requests
from gtts import gTTS
from skimage import color
from skimage.filters import sobel
from ultralytics import YOLO

from guide_dog.motion import MotionAnalyzer, classify_person_motion
from guide_dog.narration import NarrationGate
from guide_dog.narrative_builder import (
    build_enriched_detections,
    build_scene_fingerprint,
    compose_narration,
    format_enriched_for_llm,
    lighting_from_frame,
)
from guide_dog.pose_hints import PoseHintEstimator

# --- Configuration ---
YOLO_MODEL = os.getenv("YOLO_MODEL", "yolo26n.pt")
POSE_MODEL = os.getenv("POSE_MODEL", "yolo11n-pose.pt")
YOLO_CONFIDENCE = 0.5
CAMERA_INDEX = 0
ENABLE_POSE = True
POSE_EVERY_N_FRAMES = 8
MIN_DESCRIBE_INTERVAL_SEC = 2.0

PURDUE_GENAI_API_KEY = os.getenv("PURDUE_GENAI_API_KEY", "")
PURDUE_GENAI_URL = os.getenv(
    "PURDUE_GENAI_URL",
    "https://genai.rcac.purdue.edu/api/chat/completions",
)
PURDUE_GENAI_MODEL = os.getenv("PURDUE_GENAI_MODEL", "llama3.1:latest")

LANGUAGE = "en"
TABLE_LABELS = {"dining table", "table", "bench"}


def interpret_scene_layout(frame_bgr: np.ndarray) -> str:
    """
    Turn raw pixels into a text briefing for Purdue GenAI (no image upload).
    Uses scikit-image luminance and edge structure plus quadrant brightness.
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    gray = color.rgb2gray(rgb)
    mean_lum = float(np.mean(gray))
    edge_activity = float(np.mean(sobel(gray)))
    h, w = gray.shape
    mid_h, mid_w = h // 2, w // 2
    quadrants = {
        "top-left": gray[:mid_h, :mid_w],
        "top-right": gray[:mid_h, mid_w:],
        "bottom-left": gray[mid_h:, :mid_w],
        "bottom-right": gray[mid_h:, mid_w:],
    }
    quad_bits = [f"{name} brightness {float(np.mean(q)):.2f}" for name, q in quadrants.items()]
    busy = "high" if edge_activity > 0.12 else "moderate" if edge_activity > 0.06 else "low"
    return (
        f"Frame layout: overall brightness {mean_lum:.2f} (0=dark, 1=bright); "
        f"structural busyness {busy} (edge mean {edge_activity:.3f}). "
        f"Quadrants: {', '.join(quad_bits)}."
    )


class ObjectDetector:
    def __init__(self, model_name: str = YOLO_MODEL, conf: float = YOLO_CONFIDENCE):
        print(f"[Guide Dog] Loading detection model {model_name}...")
        self.model = YOLO(model_name)
        self.conf = conf
        print("[Guide Dog] Detection model ready.")

    def detect(self, frame: np.ndarray) -> list[dict]:
        result = self.model(frame, conf=self.conf, verbose=False)[0]
        detections: list[dict] = []
        if result.boxes is None:
            return detections
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = result.names[cls_id]
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            detections.append(
                {
                    "label": label,
                    "confidence": round(float(box.conf[0]), 2),
                    "box": (int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
                }
            )
        return detections


class AIDescriber:
    """Purdue GenAI Studio — OpenAI-compatible chat completions (text only)."""

    def __init__(
        self,
        api_key: str,
        language: str = "en",
        base_url: str | None = None,
        model: str | None = None,
    ):
        self.api_key = (api_key or "").strip()
        self.is_configured = bool(self.api_key)
        self.language = language
        self.base_url = base_url or PURDUE_GENAI_URL
        self.model = model or PURDUE_GENAI_MODEL

    def describe(
        self,
        scene_layout: str,
        detections_text: str,
        local_draft: str,
    ) -> str | None:
        if not self.is_configured:
            return None

        system = (
            "You help blind and low-vision users understand their surroundings from "
            "sensor summaries. Reply in one or two short, natural sentences suitable "
            "for text-to-speech. Do not use markdown or bullet lists. "
            f"Respond in language code: {self.language}."
        )
        user = (
            f"Scene layout (from image analysis):\n{scene_layout}\n\n"
            f"Detected objects (YOLO + heuristics):\n{detections_text}\n\n"
            f"Local draft narration:\n{local_draft}\n\n"
            "Polish the draft into a clear spoken update. Keep facts from the sensors; "
            "do not invent objects that are not listed."
        )
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            resp = requests.post(
                self.base_url, headers=headers, json=body, timeout=60
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except (requests.RequestException, KeyError, json.JSONDecodeError) as exc:
            print(f"[Guide Dog] GenAI request failed: {exc}")
            return None


class TextToSpeech:
    def __init__(self) -> None:
        pygame.mixer.init()
        self.language = LANGUAGE
        self._q: queue.Queue[str | None] = queue.Queue()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self) -> None:
        while True:
            text = self._q.get()
            if text is None:
                break
            try:
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                    path = tmp.name
                gTTS(text=text, lang=self.language).save(path)
                pygame.mixer.music.load(path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)
                try:
                    os.unlink(path)
                except OSError:
                    pass
            except Exception as exc:
                print(f"[Guide Dog] TTS error: {exc}")
            finally:
                self._q.task_done()

    def speak(self, text: str) -> None:
        if text and text.strip():
            self._q.put(text.strip())

    def set_language(self, lang: str) -> None:
        self.language = lang


def _person_and_table_boxes(
    detections: list[dict],
) -> tuple[list[tuple[int, int, int, int]], list[tuple[int, int, int, int]]]:
    people: list[tuple[int, int, int, int]] = []
    tables: list[tuple[int, int, int, int]] = []
    for d in detections:
        label = d["label"].lower()
        if label == "person":
            people.append(d["box"])
        elif label in TABLE_LABELS:
            tables.append(d["box"])
    return people, tables


def _draw_overlay(
    frame: np.ndarray,
    detections: list[dict],
    spoken: str,
) -> None:
    for d in detections:
        x, y, w, h = d["box"]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{d['label']}: {d['confidence']}",
            (x, max(y - 8, 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
    if spoken:
        y0 = 28
        for line in _wrap_text(spoken, 70):
            cv2.putText(
                frame,
                line,
                (10, y0),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (200, 230, 255),
                2,
            )
            y0 += 22


def _wrap_text(text: str, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        trial = " ".join(current + [word])
        if len(trial) <= width:
            current.append(word)
        else:
            if current:
                lines.append(" ".join(current))
            current = [word]
    if current:
        lines.append(" ".join(current))
    return lines[:4]


def main() -> None:
    global LANGUAGE

    detector = ObjectDetector()
    tts = TextToSpeech()
    describer = AIDescriber(api_key=PURDUE_GENAI_API_KEY, language=LANGUAGE)
    motion = MotionAnalyzer()
    narration_gate = NarrationGate()
    pose_estimator: PoseHintEstimator | None = None
    if ENABLE_POSE:
        try:
            pose_estimator = PoseHintEstimator(POSE_MODEL)
        except Exception as exc:
            print(f"[Guide Dog] Pose model disabled: {exc}")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {CAMERA_INDEX}")

    last_spoken_text = [""]
    describe_lock = threading.Lock()
    last_describe_started = 0.0
    frame_idx = 0
    cached_pose: list[dict] = []

    if describer.is_configured:
        tts.speak("Guide Dog is ready. I will speak when something meaningful changes in view.")
    else:
        print(
            "[Guide Dog] No PURDUE_GENAI_API_KEY — using local narration from YOLO and scene cues.\n"
            "  Add a key in .env or set PURDUE_GENAI_API_KEY in your environment."
        )
        tts.speak("Guide Dog is ready. I will describe changes using the camera.")

    print("[Guide Dog] Running. Press 'q' to quit, 'l' to change language.")

    def run_describe(
        scene_layout: str,
        enriched: list[dict],
        lighting_phrase: str,
        local_draft: str,
    ) -> None:
        nonlocal last_describe_started
        detections_text = format_enriched_for_llm(enriched)
        polished = describer.describe(scene_layout, detections_text, local_draft)
        phrase = polished if polished else local_draft
        tts.speak(phrase)
        last_spoken_text[0] = phrase

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        detections = detector.detect(frame)
        curr_small, prev_small, sw, sh = motion.tick(frame)

        people, tables = _person_and_table_boxes(detections)
        person_motion: list[str] = []
        for box in people:
            upper, lower, whole = motion.roi_motion(
                box, frame.shape, prev_small, curr_small, sw, sh
            )
            person_motion.append(classify_person_motion(upper, lower, whole))

        if (
            pose_estimator is not None
            and people
            and frame_idx % POSE_EVERY_N_FRAMES == 0
        ):
            cached_pose = pose_estimator.estimate(frame, people, tables)

        lighting_phrase, lighting_bucket = lighting_from_frame(frame)
        enriched = build_enriched_detections(
            detections, frame.shape, person_motion, cached_pose
        )
        fingerprint = build_scene_fingerprint(enriched, lighting_bucket)

        if narration_gate.is_new(fingerprint):
            local_draft = compose_narration(enriched, lighting_phrase)
            scene_layout = interpret_scene_layout(frame)
            now = time.time()
            with describe_lock:
                if now - last_describe_started >= MIN_DESCRIBE_INTERVAL_SEC:
                    last_describe_started = now
                    threading.Thread(
                        target=run_describe,
                        args=(scene_layout, enriched, lighting_phrase, local_draft),
                        daemon=True,
                    ).start()

        _draw_overlay(frame, detections, last_spoken_text[0])
        cv2.imshow("Guide Dog", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("l"):
            lang = input("Language code (e.g. en, es): ").strip() or LANGUAGE
            LANGUAGE = lang
            tts.set_language(lang)
            describer.language = lang
            narration_gate.reset()
            print(f"[Guide Dog] Language set to {lang}")

    cap.release()
    cv2.destroyAllWindows()
    tts._q.put(None)


if __name__ == "__main__":
    main()
