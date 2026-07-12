# Train and Deploy YOLO Models

End-to-end pipeline for training, validating, and deploying YOLO object detection models — from dataset preparation to real-time inference on edge devices.

## Features

- **Dataset preparation** — Train/validation split utility for YOLO-format datasets
- **Real-time inference** — Run YOLO detection on images, videos, USB cameras, or Raspberry Pi cameras
- **Servo tracking** — Combine YOLO inference with hardware servo control for object tracking
- **Edge deployment** — Optimized for Raspberry Pi and embedded Linux

## Usage

### Split dataset into train/validation sets

```bash
python train_val_split.py --datapath /path/to/data --train_pct 0.8
```

### Run inference

```bash
# On an image
python yolo_detect.py --model runs/detect/train/weights/best.pt --source test.jpg

# On a video
python yolo_detect.py --model best.pt --source video.mp4 --thresh 0.5

# USB camera
python yolo_detect.py --model best.pt --source usb0 --resolution 640x480

# Raspberry Pi camera
python yolo_detect.py --model best.pt --source picamera0 --resolution 640x480
```

## Requirements

- Python 3.8+
- OpenCV
- Ultralytics (YOLO)
- NumPy

```bash
pip install ultralytics opencv-python numpy
```

## Project Structure

```
├── yolo_detect.py          # Real-time inference script
├── train_val_split.py      # Dataset train/val splitting utility
└── README.md
```

## License

MIT
