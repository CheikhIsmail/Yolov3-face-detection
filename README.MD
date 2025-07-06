# YOLOv3 Face Detection

This project implements a YOLOv3-based face detector using PyTorch for training and ONNX+OpenCV for real-time inference.

---

## Features

- YOLOv3 architecture for face detection
- Training and evaluation in PyTorch (`YOLO_v3_ONNX_exp.py`)
- Result visualization and saving
- ONNX export for fast inference
- Real-time webcam detection with OpenCV (`detect_vid.py`)
- Real-time image detection with OpenCV (`detect_img.py`)


---


## Getting Started

### 1. Install Requirements

```sh
pip install torch torchvision opencv-python numpy matplotlib tqdm pillow
```

### 2. Prepare Dataset

- Organize your dataset in YOLO format:
  - `images/` (input images)
  - `labels/` (YOLO-format label files: normalized `x_center y_center width height` per line)

### 3. Training

```sh
python YOLO_v3_ONNX_exp.py
```
- Adjust paths and hyperparameters as needed in the script.
- Checkpoints and ONNX model will be saved automatically.

### 4. Testing/Evaluation

- The script will save result images with predictions in the `results/` folder.

### 5. Real time image Detection
```sh
python detect_img.py
```
-Make sure to add your image path 
-Adjust the conf treshold if needed


### 6. Real-Time Webcam Detection

```sh
python detect_vid.py
```
- Make sure `YOLO_v3_faces_fixed.onnx` is present in the project folder.
- adjust the treshold if needed
- A webcam window will open with live face detection. Press `Esc` to exit.

---

## Customization

- **Anchors:**  
  Adjust anchors in both scripts if your dataset has different object sizes.
- **Image Size:**  
  Default is 160x160. Change `image_size` in both scripts if needed.
- **Thresholds:**  
  Tune `conf_threshold` and `nms_threshold` for best results.

---

## Troubleshooting

- **High Loss:**  
  Check your data, anchors, and learning rate.
- **No Detections:**  
  Lower the confidence threshold or check if the ONNX model matches your training.
- **Multiple Boxes:**  
  Increase the confidence or NMS threshold.

---
