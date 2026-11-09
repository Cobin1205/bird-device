import onnxruntime as ort
from torchvision import transforms
from PIL import Image
import numpy as np
import onnx
import json
import os
import cv2

# ----------------------------
# CONFIG & MODEL LOADING
# ----------------------------

classifier_path = "bird_model.onnx"
detector_path = "yolo11n.onnx"

DETECT_SIZE = 320
BIRD_CLASS_ID = 14        # COCO "bird"
DETECT_CONF = 0.40        # min confidence for a bird detection
NMS_IOU = 0.45            

# Load Classifier (ONNX Runtime)
session = ort.InferenceSession(
    classifier_path,
    providers=["CPUExecutionProvider"]
)

# Extract class names directly from the ONNX model metadata
onnx_model = onnx.load(classifier_path)
metadata = {meta.key: meta.value for meta in onnx_model.metadata_props}

if "class_names" in metadata:
    classes = json.loads(metadata["class_names"])
    print(f"Successfully loaded {len(classes)} classes from ONNX model metadata.\n")
else:
    raise ValueError("Class names metadata not found in the ONNX model! Ensure you exported it using the updated pt_to_onnx.py script.")

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Load Detector (OpenCV DNN, matching Raspberry Pi main.py)
print("Loading YOLO detector...")
detector = cv2.dnn.readNetFromONNX(detector_path)
detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

print("Models loaded successfully.")
print("-" * 40)

# ----------------------------
# HELPER FUNCTIONS (YOLO Detection & Cropping)
# ----------------------------

def letterbox(img, size):
    w, h = img.size
    scale = min(size / w, size / h)
    new_w, new_h = round(w * scale), round(h * scale)
    resized = img.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2
    canvas.paste(resized, (pad_x, pad_y))
    return canvas, scale, pad_x, pad_y

def nms(boxes, scores, iou_thresh):
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / (area_i + area_r - inter + 1e-9)
        order = rest[iou < iou_thresh]
    return keep

def detect_and_crop_bird(detector, image):
    """Runs YOLO to find the highest-confidence bird and crops it. Fallback to full image if none found."""
    canvas, scale, pad_x, pad_y = letterbox(image, DETECT_SIZE)
    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    arr = np.ascontiguousarray(np.transpose(arr, (2, 0, 1))[None, ...])

    detector.setInput(arr)
    out = detector.forward()[0]
    if out.shape[0] < out.shape[1]:
        out = out.T

    scores = out[:, 4 + BIRD_CLASS_ID]
    mask = scores > DETECT_CONF
    if not mask.any():
        return image  # Fallback to uncropped if YOLO misses

    out, scores = out[mask], scores[mask]
    cx, cy, w, h = out[:, 0], out[:, 1], out[:, 2], out[:, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)

    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale
    img_w, img_h = image.size
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, img_w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, img_h)

    keep = nms(boxes, scores, NMS_IOU)
    if not keep:
        return image

    # Take the top detection box
    best_box = boxes[keep[0]]
    x1, y1, x2, y2 = (int(v) for v in best_box)
    
    if x2 - x1 < 10 or y2 - y1 < 10:
        return image

    return image.crop((x1, y1, x2, y2))

# ----------------------------
# IMAGE PREPROCESSING FOR CLASSIFIER
# ----------------------------

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ----------------------------
# LOOP OVER TEST IMAGES
# ----------------------------

test_dir = "model/testBirds"

if not os.path.exists(test_dir):
    print(f"Directory not found: {test_dir}")
else:
    valid_extensions = (".jpg", ".jpeg", ".png")
    image_files = [f for f in os.listdir(test_dir) if f.lower().endswith(valid_extensions)]

    total = len(image_files)
    wrong = 0

    if not image_files:
        print(f"No valid images found in {test_dir}")
    else:
        for img_file in image_files:
            image_path = os.path.join(test_dir, img_file)
            
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                print(f"Could not open {img_file}: {e}")
                continue

            # CRITICAL FIX: Detect and crop to the bird first (matching Pi behavior)
            cropped_bird = detect_and_crop_bird(detector, image)

            # Preprocess cropped bird image
            tensor_image = transform(cropped_bird)
            
            # Convert PyTorch tensor to NumPy array and add batch dimension
            np_image = np.expand_dims(tensor_image.numpy(), axis=0).astype(np.float32)

            # Run classifier model
            outputs = session.run(
                [output_name],
                {input_name: np_image}
            )

            output = outputs[0]
            predicted_index = np.argmax(output, axis=1)[0]
            predicted_species = classes[predicted_index]

            expected_species = img_file[0:len(img_file)-9]

            if expected_species != predicted_species:
                wrong += 1

            print(f"{'WRONG!!!' if expected_species != predicted_species else ''} Image: {expected_species} --> Predicted: {predicted_species}")

        correct = total - wrong
        accuracy = (correct / total) * 100 if total > 0 else 0

        print("-" * 40)
        print("Finished processing all test images.")
        print(f'Incorrect: {wrong}    Total: {total}    Accuracy: {accuracy:.1f}%')