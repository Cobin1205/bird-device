import numpy as np
import torch
import torch.nn as nn
from torchvision import models

# ----------------------------
# 1. SETTINGS
# ----------------------------

checkpoint_path = "bird_checkpoint.pth"
onnx_path = "bird_model.onnx"
num_classes = 60

device = torch.device("cpu")
print("Using device:", device)

# ----------------------------
# 2. CREATE MODEL
# ----------------------------

print("Creating MobileNetV2...")
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
model = model.to(device)

# ----------------------------
# 3. LOAD CHECKPOINT
# ----------------------------

print("Loading checkpoint...")
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
model.load_state_dict(checkpoint["model_state"])
print(f"Loaded model from epoch {checkpoint['epoch'] + 1}")

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
# 6. OPTIONAL: SIMPLIFY (pip install onnx onnxsim)
# ----------------------------

try:
    import onnx
    from onnxsim import simplify

    simplified, ok = simplify(onnx.load(onnx_path))
    if ok:
        onnx.save(simplified, onnx_path)
        print("Simplified ONNX model with onnxsim")
    else:
        print("onnxsim could not validate the simplified model; keeping original")
except ImportError:
    print("onnx/onnxsim not installed; skipping simplification")

# ----------------------------
# 7. VERIFY WITH OPENCV (the same loader the Pi uses)
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