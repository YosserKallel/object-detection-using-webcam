# YOLOv8 Webcam Object Detection

## Description

This project runs a real-time YOLOv8 object detector from a webcam feed. It opens your default camera, detects common objects, draws bounding boxes and labels on the video, and shows the total number of detected objects on screen.

The current script uses the pretrained `yolov8s.pt` checkpoint and is configured to detect all COCO classes, not just people.

## What It Does

- Opens your default webcam.
- Runs YOLOv8 object detection on each frame.
- Shows object names and confidence scores on the video.
- Displays a running `Total: X` count for detected objects.
- Lets you quit with the `q` key.

## Requirements

- Python 3.12 is recommended for this project.
- OpenCV
- Ultralytics

If the YOLOv8s checkpoint is not already available locally, Ultralytics may download it automatically the first time you run the app.

## How To Run

1. Open a terminal in this folder.
2. Run:

```powershell
& "C:\Program Files\Python312\python.exe" app.py
```

3. Allow webcam access if Windows asks for permission.
4. Point the camera at an object and wait for the label and bounding box to appear.
5. Press `q` to exit.

## Current Settings In The App

- Model: `yolov8s.pt`
- Confidence threshold: `0.35`
- IoU threshold: `0.5`
- Image size: `640`

## Notes

- The older RealSense example and snapshot-saving notes are not active in the current script.
- If detections feel noisy, raise the confidence threshold a little.
- If detections feel too strict, lower the confidence threshold a little.
