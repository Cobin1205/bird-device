import io
import json
import time
import subprocess
from datetime import datetime

import cv2
import numpy as np
import requests
from PIL import Image, ImageDraw

# =====================================================
# CONFIG
# =====================================================

VERBOSE = True

DETECTOR_PATH = "yolo11n.onnx"       # exported YOLO11n (imgsz must match DETECT_SIZE)
CLASSIFIER_PATH = "bird_model.onnx"  # your species model

DETECT_SIZE = 320         # must match the imgsz used when exporting YOLO to ONNX
BIRD_CLASS_ID = 14        # COCO "bird"
DETECT_CONF = 0.40        # min confidence for a bird detection
NMS_IOU = 0.45            # overlap threshold for removing duplicate boxes
LOOP_DELAY = 0.5          # seconds between frames

WEBHOOK_URL = "https://webhook.site/a32c1a03-0dd5-4ef2-a876-baae164519dd"

CLASSES = [
    'American_Crow', 'American_Goldfinch', 'American_Robin', 'Baltimore_Oriole', 'Barn_Swallow',
    'Black_Capped_Chickadee', 'Blue_Grosbeak', 'Blue_Jay', 'Bobolink', 'Brewer_Blackbird',
    'Brown_Creeper', 'Brown_Thrasher', 'Brownheaded_Cowbird', 'Cardinal', 'Carolina_Wren',
    'Cedar_Waxwing', 'Chipping_Sparrow', 'Common_Grackle', 'Common_Raven', 'Common_Yellowthroat',
    'Dark_eyed_Junco', 'Downy_Woodpecker', 'Eastern_Bluebird', 'Eastern_Towhee', 'European_Starling',
    'Field_Sparrow', 'Fox_Sparrow', 'Gray_Catbird', 'Hairy_Woodpecker', 'House_Finch',
    'House_Sparrow', 'House_Wren', 'Indigo_Bunting', 'Lincoln_Sparrow', 'Magnolia_Warbler',
    'Mockingbird', 'Mourning_Dove', 'Northern_Flicker', 'Orchard_Oriole', 'Pileated_Woodpecker',
    'Pine_Warbler', 'Purple_Finch', 'Red_bellied_Woodpecker', 'Red_headed_Woodpecker',
    'Red_winged_Blackbird', 'Rose_breasted_Grosbeak', 'Ruby_throated_Hummingbird', 'Rusty_Blackbird',
    'Savannah_Sparrow', 'Scarlet_Tanager', 'Song_Sparrow', 'Spotted_Catbird', 'Tree_Sparrow',
    'Tree_Swallow', 'Tufted_Titmouse', 'White_breasted_Nuthatch', 'White_crowned_Sparrow',
    'White_throated_Sparrow', 'Yellow_Warbler', 'Yellow_headed_Blackbird',
]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def vprint(msg):
    if VERBOSE:
        print(msg)


# =====================================================
# MODEL LOADING (OpenCV DNN, no onnxruntime needed)
# =====================================================

def load_net(path):
    net = cv2.dnn.readNetFromONNX(path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net


# =====================================================
# CAMERA
# =====================================================

def capture_image():
    """Take a photo with the Pi camera and return it as an RGB PIL image (or None on failure)."""
    try:
        result = subprocess.run(
            ["rpicam-still", "-n", "-t", "500",
             "--width", "640", "--height", "480",
             "-e", "jpg", "-o", "-"],       # "-o -" writes the JPEG to stdout
            capture_output=True,
            timeout=30,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"Failed to capture image: {e}")
        return None

    if result.returncode != 0 or not result.stdout:
        print("Failed to capture image:", result.stderr.decode(errors="ignore")[-300:])
        return None

    vprint('successfully captured image')
    return Image.open(io.BytesIO(result.stdout)).convert("RGB")

# =====================================================
# BIRD DETECTION (YOLO11n via cv2.dnn)
# =====================================================

def letterbox(img, size):
    """Resize keeping aspect ratio and pad to a square. Returns padded image + mapping info."""
    w, h = img.size
    scale = min(size / w, size / h)
    new_w, new_h = round(w * scale), round(h * scale)
    resized = img.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2
    canvas.paste(resized, (pad_x, pad_y))
    vprint('resized image')
    return canvas, scale, pad_x, pad_y


def nms(boxes, scores, iou_thresh):
    """Plain numpy non-max suppression. boxes are [x1, y1, x2, y2]."""
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


def detect_birds(detector, image):
    """Return a list of [x1, y1, x2, y2] boxes (in original image pixels) for each bird."""
    canvas, scale, pad_x, pad_y = letterbox(image, DETECT_SIZE)
    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    arr = np.ascontiguousarray(np.transpose(arr, (2, 0, 1))[None, ...])  # (1, 3, H, W)

    detector.setInput(arr)
    out = detector.forward()[0]          # (84, N) for YOLO11
    if out.shape[0] < out.shape[1]:
        out = out.T                      # -> (N, 84)

    scores = out[:, 4 + BIRD_CLASS_ID]
    mask = scores > DETECT_CONF
    if not mask.any():
        return []

    out, scores = out[mask], scores[mask]
    cx, cy, w, h = out[:, 0], out[:, 1], out[:, 2], out[:, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)

    # Undo letterboxing -> original image coordinates
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale
    img_w, img_h = image.size
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, img_w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, img_h)

    keep = nms(boxes, scores, NMS_IOU)
    vprint('detected birds and obtained boxes')
    return [boxes[i].tolist() for i in keep]


# =====================================================
# SPECIES CLASSIFICATION (your bird_model.onnx via cv2.dnn)
# =====================================================

def preprocess_for_classifier(crop):
    """Same as Resize(256) -> CenterCrop(224) -> ToTensor -> Normalize, using only PIL + numpy."""
    crop = crop.convert("RGB")
    w, h = crop.size
    if w < h:
        new_w, new_h = 256, max(224, round(h * 256 / w))
    else:
        new_w, new_h = max(224, round(w * 256 / h)), 256
    crop = crop.resize((new_w, new_h), Image.BILINEAR)

    left, top = (new_w - 224) // 2, (new_h - 224) // 2
    crop = crop.crop((left, top, left + 224, top + 224))

    arr = np.asarray(crop, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))[None, ...]  # (1, 3, 224, 224)
    vprint('preprocessed')
    return np.ascontiguousarray(arr, dtype=np.float32)


def predict(classifier, image, boxes):
    """Classify each cropped bird. Returns a list of species names."""
    species = []
    for box in boxes:
        x1, y1, x2, y2 = (int(v) for v in box)
        if x2 - x1 < 2 or y2 - y1 < 2:
            species.append("Unknown")
            continue
        crop = image.crop((x1, y1, x2, y2))
        tensor = preprocess_for_classifier(crop)
        classifier.setInput(tensor)
        output = classifier.forward()
        idx = int(np.argmax(output, axis=1)[0])
        name = CLASSES[idx] if idx < len(CLASSES) else f"class_{idx}"
        species.append(name)
        vprint(f"species predicted: {name}")
    if len(species) == 0:
        vprint('no birds found')
    return species


# =====================================================
# DRAWING + UPLOAD
# =====================================================

def annotate(image, boxes, labels):
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    for box, label in zip(boxes, labels):
        draw.rectangle(box, outline="red", width=3)
        draw.text((box[0] + 4, box[1] + 4), label, fill=(255, 255, 255),
                  stroke_width=2, stroke_fill="red")
    vprint('boxes drawn on image')
    return annotated


def send_to_server(image, labels):
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    buf.seek(0)

    data = {
        "bird_names": json.dumps(labels),
        "timestamp": datetime.now().isoformat(),
    }
    files = {"image": ("bird.jpg", buf, "image/jpeg")}
    try:
        response = requests.post(WEBHOOK_URL, data=data, files=files, timeout=15)
        print(response)
    except requests.RequestException as e:
        print(f"Failed to send to server: {e}")


# =====================================================
# MAIN
# =====================================================

def main():
    print("=====================================")
    print("Starting Bird Device Application")
    print("=====================================")

    # Load models ONCE
    print("Loading models...")
    try:
        detector = load_net(DETECTOR_PATH)
        classifier = load_net(CLASSIFIER_PATH)
    except cv2.error as e:
        print(f"OpenCV could not load a model: {e}")
        return

    # Sanity check: does the classifier's class count match CLASSES?
    try:
        classifier.setInput(np.zeros((1, 3, 224, 224), dtype=np.float32))
        n_out = classifier.forward().shape[-1]
        if n_out != len(CLASSES):
            print(f"WARNING: model outputs {n_out} classes but CLASSES has {len(CLASSES)}. "
                  f"Predicted names will be wrong!")
    except cv2.error as e:
        print(f"Classifier test run failed: {e}")
        return

    num_birds = 0
    print("Running. Press Ctrl+C to stop.")

    try:
        while True:
            image = capture_image()
            if image is None:
                break

            boxes = detect_birds(detector, image)
            new_num_birds = len(boxes)
            vprint(f"Num of Birds: {new_num_birds}")

            # Only report when more birds appear than were there before
            if new_num_birds > num_birds:
                species = predict(classifier, image, boxes)
                annotated = annotate(image, boxes, species)
                send_to_server(annotated, species)

            num_birds = new_num_birds
            time.sleep(LOOP_DELAY)
    except KeyboardInterrupt:
        print("Exiting Program Loop")
    finally:
        print("Program Stopped")


if __name__ == "__main__":
    main()
