import cv2
import numpy as np
import math

# === Load YOLOv3 ONNX model ===
model_path = 'YOLO_v3_faces_fixed.onnx'
model = cv2.dnn.readNetFromONNX(model_path)

# === Load image ===
image = cv2.imread('380358.jpg')
image_height, image_width = image.shape[:2]

# === Preprocess image ===
input_size = 160  # Model was trained on 160x160 images
blob = cv2.dnn.blobFromImage(
    image,
    scalefactor=1/255.0,
    size=(input_size, input_size),
    mean=(0, 0, 0),
    swapRB=True,
    crop=False
)
model.setInput(blob)

# === Run inference ===
outputs = model.forward()  # Output shape: (1, N, 6)
detections = outputs[0]  # (N, 6)

# === Anchors: must match training
anchors_per_scale = [
    [(0.4375, 0.4375), (0.6125, 0.6125), (0.475, 0.475)],   # 5x5
    [(0.50625, 0.50625), (0.375, 0.375), (0.5625, 0.5625)], # 10x10
    [(0.5875, 0.5875), (0.6625, 0.6625), (0.5375, 0.5375)], # 20x20
]
flat_anchors = sum(anchors_per_scale, [])  # Flattened anchor list
grid_sizes = [5, 10, 20]  # Low to high resolution

# === Post-processing settings ===
conf_threshold = 0.425 # Start low to debug
nms_threshold = 0.5

scale_factor = 4.0  # Increase box sizes by this factor (tweak as needed)

# === Decode outputs ===
boxes = []
confidences = []

index = 0
for scale_idx, S in enumerate(grid_sizes):
    for anchor_idx in range(3):
        anchor = flat_anchors[scale_idx * 3 + anchor_idx]
        for i in range(S):  # y
            for j in range(S):  # x
                if index >= detections.shape[0]:
                    break

                det = detections[index]
                conf = 1 / (1 + math.exp(-det[0]))
                if conf < conf_threshold:
                    index += 1
                    continue

                tx, ty, tw, th = det[1:5]

                # Decode position (sigmoid)
                x = (j + 1 / (1 + math.exp(-tx))) / S
                y = (i + 1 / (1 + math.exp(-ty))) / S

                # Clamp sizes and apply scale factor
                tw = min(tw, 4)
                th = min(th, 4)
                w = anchor[0] * scale_factor * math.exp(tw)
                h = anchor[1] * scale_factor * math.exp(th)

                # Scale to image size
                x0 = int((x - w / 2) * image_width)
                y0 = int((y - h / 2) * image_height)
                w_pixel = int(w * image_width)
                h_pixel = int(h * image_height)

                # Debug print for box size
                print(f"Box (w,h) in pixels: ({w_pixel}, {h_pixel})")

                boxes.append([x0, y0, w_pixel, h_pixel])
                confidences.append(float(conf))

                index += 1

# === Apply Non-Max Suppression ===
indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

# === Draw results ===
for i in indices:
    i = i[0] if isinstance(i, (list, tuple, np.ndarray)) else i
    x, y, w, h = boxes[i]
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(image, "face", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

if len(indices) == 0:
    print("No detections passed the confidence/NMS threshold.")

# === Show and save result ===
cv2.imshow('YOLOv3 Face Detection', image)
cv2.imwrite('../../outputs/yolov3_face_result.jpg', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
