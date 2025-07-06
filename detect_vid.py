import os
import cv2
import numpy as np
import math

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# === Load YOLOv3 ONNX model ===
model_path = 'YOLO_v3_faces_fixed.onnx'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")
model = cv2.dnn.readNetFromONNX(model_path)

# === Anchors and config ===
anchors_per_scale = [
    [(0.4375, 0.4375), (0.6125, 0.6125), (0.475, 0.475)],
    [(0.50625, 0.50625), (0.375, 0.375), (0.5625, 0.5625)],
    [(0.5875, 0.5875), (0.6625, 0.6625), (0.5375, 0.5375)],
]
flat_anchors = sum(anchors_per_scale, [])
grid_sizes = [5, 10, 20]

input_size = 160  # Model expects 160x160
conf_threshold = 0.38
nms_threshold = 0.5
scale_factor = 4.0

def iou_np(box1, box2):
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]
    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    box1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    return inter_area / (box1_area + box2_area - inter_area + 1e-6)

cap = cv2.VideoCapture(0)

while True:
    ret, image = cap.read()
    if not ret:
        break

    orig_height, orig_width = image.shape[:2]

    # Resize image to 160x160 for the model
    resized_image = cv2.resize(image, (input_size, input_size))
    blob = cv2.dnn.blobFromImage(resized_image, 1/255.0, (input_size, input_size), mean=(0,0,0), swapRB=True, crop=False)
    model.setInput(blob)
    outputs = model.forward()
    detections = outputs[0]

    # Debug: print output shape and a few confidences
    print("Model output shape:", detections.shape)
    print("First 10 confidences:", [sigmoid(d[0]) for d in detections[:10]])

    boxes, confidences = [], []
    index = 0

    for scale_idx, S in enumerate(grid_sizes):
        for anchor_idx in range(3):
            anchor = flat_anchors[scale_idx * 3 + anchor_idx]
            for i in range(S):
                for j in range(S):
                    if index >= detections.shape[0]:
                        break

                    det = detections[index]
                    conf = sigmoid(det[0])
                    if conf < conf_threshold:
                        index += 1
                        continue

                    tx, ty, tw, th = det[1:5]
                    x = (j + sigmoid(tx)) / S
                    y = (i + sigmoid(ty)) / S

                    tw = min(tw, 4)
                    th = min(th, 4)
                    w = anchor[0] * scale_factor * math.exp(tw)
                    h = anchor[1] * scale_factor * math.exp(th)

                    # Map box coordinates back to original image size
                    x0 = max(0, int((x - w / 2) * orig_width))
                    y0 = max(0, int((y - h / 2) * orig_height))
                    w_pixel = min(orig_width - x0, int(w * orig_width))
                    h_pixel = min(orig_height - y0, int(h * orig_height))

                    boxes.append([x0, y0, w_pixel, h_pixel])
                    confidences.append(float(conf))
                    index += 1

    print("Detections this frame:", len(confidences))

    if len(confidences) > 0:
        # Convert boxes to [x, y, x2, y2] format for NMS
        boxes_xyxy = [[x, y, x + w, y + h] for (x, y, w, h) in boxes]
        indices = cv2.dnn.NMSBoxes(boxes_xyxy, confidences, conf_threshold, nms_threshold)

        for i in indices:
            i = i[0] if isinstance(i, (list, np.ndarray)) else i
            x, y, w, h = boxes[i]
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = "face"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            label_size, baseline = cv2.getTextSize(label, font, font_scale, thickness)
            label_x = x
            label_y = y - 10 if y - 10 > 10 else y + label_size[1] + 10
            cv2.rectangle(
                image,
                (label_x, label_y - label_size[1] - baseline),
                (label_x + label_size[0], label_y + baseline),
                (0, 255, 0),
                cv2.FILLED
            )
            cv2.putText(
                image,
                label,
                (label_x, label_y),
                font,
                font_scale,
                (0, 0, 0),
                thickness
            )

    cv2.imshow('YOLOv3 Face Detection - Live', image)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
