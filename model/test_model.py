import torch
from torchvision import transforms
from PIL import Image
import os
from bird_classes import ClassObject
import ast

# ----------------------------
# LOAD MODEL
# ----------------------------

extra_files = {"classes.txt": ""}

# Load the model while passing the extra_files dictionary
model = torch.jit.load("bird_model.pt", _extra_files=extra_files)
model.eval() # switch from training mode to evaluation mode

# Retrieve and decode the classes list from the loaded bytes
classes_str = extra_files["classes.txt"].decode("utf-8")
classes = ast.literal_eval(classes_str)

print(f"Loaded {len(classes)} classes from model checkpoint.")

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
classes = ClassObject.get_class_list()

print(classes)
print(len(classes))

# ----------------------------
# LOAD TEST IMAGE
# ----------------------------
image_path = "model/testBirds/testBlackbird.jpg"   # change this to your image
image = Image.open(image_path).convert("RGB")
image = transform(image)
image = image.unsqueeze(0)

# ----------------------------
# RUN MODEL
# ----------------------------
with torch.no_grad():
    output = model(image)
    predicted_index = torch.argmax(output, dim=1).item()

# ----------------------------
# PRINT RESULT
# ----------------------------
predicted_species = classes[predicted_index]

print("Predicted bird species:")
print(predicted_species)