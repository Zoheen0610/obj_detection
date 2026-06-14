# 🎯 YOLOv8 Real-Time Object Detection

Real-time object detection using YOLOv8 and OpenCV with live per-class counting, annotated video saving, and image inference.

## Setup
```bash
pip install ultralytics opencv-python
```

## Usage
```bash
# Webcam
python detect.py --source 0

# Video file + save output
python detect.py --source traffic.mp4 --save output.mp4

# Single image
python detect.py --source street.jpg --save result.jpg --image
```
Press **`q`** to quit. Detection summary prints to terminal on exit.

## Config
```python
MODEL_PATH = "yolov8n.pt"  # swap for yolov8s/m for better accuracy
CONF_THRESHOLD = 0.4
```

## Stack
Python · YOLOv8 · OpenCV · PyTorch