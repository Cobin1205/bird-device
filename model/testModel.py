import torch
from torchvision import transforms
from PIL import Image
import os

# ----------------------------
# LOAD MODEL
# ----------------------------

model = torch.jit.load("bird_model.pt")
model.eval() #switch from training mode to evaluation mode

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

print(classes)

# ----------------------------
# LOAD TEST IMAGE
# ----------------------------
image_path = "model/testBirds/starling.jpg"   # change this to your image
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