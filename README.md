<![CDATA[# Train and Deploy YOLO Models on Edge Devices

> From dataset → training → real-time inference on Raspberry Pi. One pipeline, no cloud needed.

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![YOLO](https://img.shields.io/badge/YOLO-Ultralytics-00FFFF?logo=yolo&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv&logoColor=white)
![Raspberry Pi](https://img.shields.io/badge/Raspberry%20Pi-Supported-C51A4A?logo=raspberrypi&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## What This Does

Most YOLO tutorials stop at training. This repo takes you all the way to **real-time detection running on actual hardware** — USB cameras, Raspberry Pi cameras, with servo tracking for physical object following.

**The full pipeline:**

```
Raw Images → Train/Val Split → YOLO Training → Real-Time Inference → Servo Tracking
```

## Features

| Feature | Description |
|---------|-------------|
| Dataset Split | Randomly splits images + labels into train/val folders (configurable ratio) |
| Multi-Source Inference | Images, videos, USB cameras, Raspberry Pi cameras |
| Servo Tracking | Physical object tracking with YOLO + servo motors |
| Confidence Filtering | Configurable detection threshold |
| FPS Counter | Real-time performance monitoring |
| Recording | Save detection results as video |

## Quick Start

```bash
# Install dependencies
pip install ultralytics opencv-python numpy

# Clone the repo
git clone https://github.com/evahsudo/Train-and-Deploy-YOLO-Models.git
cd Train-and-Deploy-YOLO-Models
```

### 1. Prepare Your Dataset

```bash
python train_val_split.py --datapath /path/to/labeled/data --train_pct 0.8
```

This splits your YOLO-format dataset (images + .txt labels) into `data/train/` and `data/validation/` folders.

### 2. Train (using Ultralytics)

```bash
yolo detect train data=data.yaml model=yolov8n.pt epochs=100 imgsz=640
```

### 3. Run Inference

```bash
# Webcam (real-time)
python yolo_detect.py --model best.pt --source usb0 --resolution 640x480

# Raspberry Pi Camera
python yolo_detect.py --model best.pt --source picamera0 --resolution 640x480

# Video file
python yolo_detect.py --model best.pt --source video.mp4 --thresh 0.5

# Single image
python yolo_detect.py --model best.pt --source test.jpg

# Record output
python yolo_detect.py --model best.pt --source usb0 --resolution 640x480 --record
```

## Supported Input Sources

| Source | Syntax | Example |
|--------|--------|---------|
| Image | filename | `--source photo.jpg` |
| Image folder | directory | `--source test_images/` |
| Video | filename | `--source clip.mp4` |
| USB Camera | `usb` + index | `--source usb0` |
| Pi Camera | `picamera` + index | `--source picamera0` |

## Hardware Setup (Servo Tracking)

The repo includes code for combining YOLO inference with servo motor control — point a camera at objects and physically track them in real-time.

**You'll need:**
- Raspberry Pi (3B+ or newer)
- USB camera or Pi Camera module
- Servo motors (SG90 or similar)
- Trained YOLO model (.pt file)

## Project Structure

```
├── yolo_detect.py              # Real-time inference (all sources)
├── train_val_split.py          # Dataset preparation utility
├── This code combines YOLO...  # Servo tracking integration
└── README.md
```

## Why This Exists

I couldn't find a clean, single-repo solution that goes from raw dataset to running on a Raspberry Pi with servo tracking. Everything was either "train only" or "inference only" or locked behind some paid course. So I built this.

## Requirements

- Python 3.8+
- OpenCV (`opencv-python`)
- Ultralytics (`ultralytics`)
- NumPy
- `picamera2` (only if using Pi Camera)

## License

MIT — use it however you want.

---

**If this saved you time, drop a star.** It helps others find it.
]]>