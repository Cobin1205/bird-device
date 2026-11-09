import numpy as np
import torch
import torch.nn as nn
from torchvision import models
import json
import os
import sys
import ast

# ----------------------------
# 1. SETTINGS
# ----------------------------

checkpoint_path = "bird_checkpoint.pth"
traced_model_path = "bird_model.pt"
onnx_path = "bird_model.onnx"

device = torch.device("cpu")
print("Using device:", device)

# ----------------------------
# 2. LOAD CLASSES FROM TRACED MODEL (.pt) EXTRA FILES
# ----------------------------

classes = []
if os.path.exists(traced_model_path):
    print(f"Extracting classes from {traced_model_path} extra files...")
    try:
        extra_files = {"classes.txt": ""}
        # Load the TorchScript model to retrieve extra files
        torch.jit.load(traced_model_path, map_location=device, _extra_files=extra_files)
        
        classes_raw = extra_files["classes.txt"]
        if isinstance(classes_raw, bytes):
            classes_raw = classes_raw.decode("utf-8")
            
        classes = ast.literal_eval(classes_raw)
        print(f"Successfully loaded {len(classes)} classes from bird_model.pt extra files.")
    except Exception as e:
        print(f"Failed to parse classes.txt from extra files: {e}")

if not classes:
    print("FAILED: Could not load class list from bird_model.pt! Ensure build_model.py has run successfully.")
    sys.exit(1)

num_classes = len(classes)

# ----------------------------
# 3. LOAD CHECKPOINT & CREATE MODEL
# ----------------------------

if not os.path.exists(checkpoint_path):
    print(f"FAILED: Checkpoint file not found at {checkpoint_path}")
    sys.exit(1)

print("Loading checkpoint weights...")
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

# Verify shape matches
expected_num_classes = checkpoint["model_state"]["classifier.1.weight"].shape[0]
if expected_num_classes != num_classes:
    print(f"WARNING: Checkpoint expects {expected_num_classes} classes, but .pt extra files gave {num_classes}!")

print("Creating MobileNetV2...")
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
model = model.to(device)

print("Loading model state dict...")
model.load_state_dict(checkpoint["model_state"])
print(f"Successfully loaded model from epoch {checkpoint['epoch'] + 1}")

# ----------------------------
# 4. PREPARE FOR EXPORT
# ----------------------------

model.eval()
example_input = torch.randn(1, 3, 224, 224, device=device)

# ----------------------------
# 5. EXPORT TO ONNX (legacy exporter, fixed batch size of 1)
# ----------------------------

print("Exporting ONNX model...")

export_kwargs = dict(
    input_names=["input"],
    output_names=["output"],
    opset_version=12,
    do_constant_folding=True,   # folds BatchNorm into Conv, which OpenCV likes
)

try:
    # Newer torch: dynamo=False forces the legacy TorchScript exporter
    torch.onnx.export(model, example_input, onnx_path, dynamo=False, **export_kwargs)
except TypeError:
    # Older torch: the legacy exporter is the only one, and has no dynamo argument
    torch.onnx.export(model, example_input, onnx_path, **export_kwargs)

print(f"ONNX model saved as: {onnx_path}")

# ----------------------------
# 6. EMBED CLASSES INTO ONNX MODEL METADATA & DESCRIPTION
# ----------------------------

try:
    import onnx
    
    onnx_model = onnx.load(onnx_path)
    classes_json = json.dumps(classes)

    # Method A: Standard ONNX metadata props (works for onnxruntime)
    meta = onnx_model.metadata_props.add()
    meta.key = "class_names"
    meta.value = classes_json

    # Method B: Model doc_string (Crucial: OpenCV's DNN module can read this field!)
    onnx_model.doc_string = classes_json

    onnx.save(onnx_model, onnx_path)
    print("Successfully embedded class names into ONNX model metadata and doc_string.")

except ImportError:
    print("onnx package not installed; could not embed class names into the model file.")

# ----------------------------
# 7. OPTIONAL: SIMPLIFY (pip install onnx onnxsim)
# ----------------------------

try:
    from onnxsim import simplify

    simplified, ok = simplify(onnx.load(onnx_path))
    if ok:
        onnx.save(simplified, onnx_path)
        print("Simplified ONNX model with onnxsim")
    else:
        print("onnxsim could not validate the simplified model; keeping original")
except ImportError:
    print("onnxsim not installed; skipping simplification")

# ----------------------------
# 8. VERIFY WITH OPENCV (the same loader the Pi uses)
# ----------------------------

try:
    import cv2

    net = cv2.dnn.readNetFromONNX(onnx_path)
    test = np.random.randn(1, 3, 224, 224).astype(np.float32)

    net.setInput(test)
    cv_out = net.forward()

    with torch.no_grad():
        torch_out = model(torch.from_numpy(test)).numpy()

    diff = np.abs(cv_out - torch_out).max()
    print(f"OpenCV loaded the model. Output shape: {cv_out.shape}")
    print(f"Max difference vs PyTorch: {diff:.6f} (should be tiny, e.g. < 0.001)")
    print("Same top class:", int(cv_out.argmax()) == int(torch_out.argmax()))
except ImportError:
    print("cv2 not installed on this machine; run: pip install opencv-python")
except cv2.error as e:
    print("OpenCV still cannot load this model:")
    print(e)