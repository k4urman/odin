# ODIN - Object Detection Interpretive Narrator

Odin captures your webcam, runs **Ultralytics YOLO** object detection, estimates **rough distance** from a single camera, infers **motion** and **pose-related cues** (e.g. possible waving, hands near a surface), and speaks a **natural-language summary** with **gTTS** + **pygame**. Optional **Purdue GenAI Studio** can polish the same structured context over an OpenAI-compatible HTTP API.

Narration is **deduplicated**: it only speaks again when the scene “fingerprint” changes (objects, position, distance bucket, motion/pose hints, or lighting)—not on a fixed repeat loop.

## Features

| Area | What it does |
|------|----------------|
| **Detection** | `yolo26n.pt` (auto-downloaded on first run). Configurable via `YOLO_MODEL`. |
| **Scene interpretation** | **scikit-image** luminance, edges, and quadrant brightness → text briefing for GenAI |
| **Depth (approximate)** | Maps bounding-box size to spoken ranges like “about three feet away.” Tune `odin/depth.py` for your webcam if needed. |
| **Motion** | Frame differencing on a downscaled gray stream for person ROI movement cues. |
| **Pose** | Optional `yolo11n-pose.pt` (throttled) for arm/hand hints. |
| **Speech** | gTTS + pygame with a **queue** so updates are not dropped. |
| **Purdue GenAI** | If `PURDUE_GENAI_API_KEY` is set, structured sensor lines + local draft are sent to [GenAI Studio](https://www.rcac.purdue.edu/knowledge/genaistudio/api). Without a key, **local** narration still runs. |

## Project layout

- **`main.py`** — Camera loop, YOLO, scikit-image layout, TTS, optional GenAI.
- **`odin/`** — `depth.py`, `motion.py`, `pose_hints.py`, `narration.py`, `narrative_builder.py`.
- **`tests/`** — `python -m unittest discover -s tests -v`

## Setup

1. **Python 3.10+** recommended.

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. **Optional — Purdue GenAI Studio**

   API key: GenAI Studio UI → avatar → **Settings** → **Account** → **API Keys**.

   Copy `.env.example` to `.env`:

   ```
   PURDUE_GENAI_API_KEY=your-key-here
   ```

   Or in PowerShell:

   ```powershell
   $env:PURDUE_GENAI_API_KEY = "your-key"
   ```

   Optional: `PURDUE_GENAI_URL`, `PURDUE_GENAI_MODEL` (default `llama3.1:latest`).

## Usage

```bash
python main.py
```

- **q** — Quit  
- **l** — Change TTS language (e.g. `en`, `es`)

Adjust `YOLO_CONFIDENCE`, `CAMERA_INDEX`, `POSE_EVERY_N_FRAMES` at the top of `main.py`.

## Limitations

- **Distance** is a monocular heuristic, not a depth sensor.
- **Gesture hints** are best-effort and can false-positive in clutter.
- **gTTS** needs network access to synthesize speech.

## Dependencies

See **`requirements.txt`**: `ultralytics`, `opencv-python`, `numpy`, `scikit-image`, `requests`, `python-dotenv`, `gTTS`, `pygame`.

---

Maninder Kaur
