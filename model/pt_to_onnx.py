import torch
import torch.nn as nn
from torchvision import models

# ----------------------------
# 1. SETTINGS
# ----------------------------

checkpoint_path = "bird_checkpoint.pth"
onnx_path = "bird_model.onnx"

num_classes = 60

# ----------------------------
# 2. DEVICE
# ----------------------------

device = torch.device("cpu")

print("Using device:", device)

# ----------------------------
# 3. CREATE MODEL
# ----------------------------

print("Creating MobileNetV2...")

model = models.mobilenet_v2(pretrained=False)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    num_classes
)

model = model.to(device)

# ----------------------------
# 4. LOAD CHECKPOINT
# ----------------------------

print("Loading checkpoint...")

checkpoint = torch.load(
    checkpoint_path,
    map_location=device
)

model.load_state_dict(checkpoint["model_state"])

print(
    f"Loaded model from epoch "
    f"{checkpoint['epoch'] + 1}"
)

# ----------------------------
# 5. PREPARE FOR EXPORT
# ----------------------------

model.eval()

example_input = torch.randn(
    1, 3, 224, 224,
    device=device
)


# ----------------------------
# 6. EXPORT TO ONNX
# ----------------------------

print("Exporting ONNX model...")

torch.onnx.export(
    model,
    example_input,
    onnx_path,

    input_names=["input"],
    output_names=["output"],

    dynamic_axes={
        "input": {
            0: "batch_size"
        },
        "output": {
            0: "batch_size"
        }
    },

    opset_version=12
)

print(f"\nONNX model saved as: {onnx_path}")