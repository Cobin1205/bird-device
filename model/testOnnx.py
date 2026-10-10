import onnxruntime as ort
from torchvision import transforms
from PIL import Image
import numpy as np
import os

# ----------------------------
# LOAD MODEL
# ----------------------------

session = ort.InferenceSession(
    "bird_model.onnx",
    providers=["CPUExecutionProvider"]
)

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
# CLASS NAMES
# ----------------------------

dataset_path = "model/CUB_200_2011/images"

classes = sorted([
    d for d in os.listdir(dataset_path)
    if os.path.isdir(os.path.join(dataset_path, d))
])

# ----------------------------
# LOAD TEST IMAGE
# ----------------------------

image_path = "model/testBirds/starling.jpg"

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
