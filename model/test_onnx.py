import onnxruntime as ort
from torchvision import transforms
from PIL import Image
import numpy as np
import onnx
import json

# ----------------------------
# LOAD MODEL & EXTRACT CLASSES
# ----------------------------

onnx_path = "bird_model.onnx"

session = ort.InferenceSession(
    onnx_path,
    providers=["CPUExecutionProvider"]
)

# Extract class names directly from the ONNX model metadata
onnx_model = onnx.load(onnx_path)
metadata = {meta.key: meta.value for meta in onnx_model.metadata_props}

if "class_names" in metadata:
    classes = json.loads(metadata["class_names"])
    print(f"Successfully loaded {len(classes)} classes from ONNX model metadata.")
else:
    raise ValueError("Class names metadata not found in the ONNX model! Ensure you exported it using the updated pt_to_onnx.py script.")

# Get model input/output names
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

print("Model input:", input_name)
print("Model output:", output_name)

# ----------------------------
# IMAGE PREPROCESSING
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
# LOAD TEST IMAGE
# ----------------------------

image_path = "model/testBirds/testBlackbird.jpg"

image = Image.open(image_path).convert("RGB")

# Preprocess image
image = transform(image)

# Convert PyTorch tensor to NumPy array
image = image.numpy()

# Add batch dimension
image = np.expand_dims(image, axis=0)

# Make sure ONNX receives float32
image = image.astype(np.float32)

# ----------------------------
# RUN MODEL
# ----------------------------

outputs = session.run(
    [output_name],
    {input_name: image}
)

output = outputs[0]

# Get predicted class
predicted_index = np.argmax(output, axis=1)[0]

# ----------------------------
# PRINT RESULT
# ----------------------------

predicted_species = classes[predicted_index]

print("\nPredicted bird species:")
print(predicted_species)