import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, random_split

# ----------------------------
# 1. DEVICE
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ----------------------------
# 2. TRANSFORMS
# ----------------------------
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ----------------------------
# 3. DATASET
# ----------------------------
full_dataset = datasets.ImageFolder(
    root="model/CUB_200_2011/cropped_images",
    transform=train_transform
)

num_classes = len(full_dataset.classes)

print("Classes:", num_classes)
print("Total Images:", len(full_dataset))

# ----------------------------
# 4. TRAIN / VALIDATION SPLIT
# ----------------------------
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(
    full_dataset,
    [train_size, val_size]
)

# Validation should NOT use augmentation
val_dataset.dataset = copy.deepcopy(full_dataset)
val_dataset.dataset.transform = val_transform

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)

# ----------------------------
# 5. MODEL
# ----------------------------
model = models.mobilenet_v2(pretrained=True)

# Freeze feature extractor initially
for param in model.features.parameters():
    param.requires_grad = False

# Replace classifier
model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    num_classes
)

model = model.to(device)

# ----------------------------
# 6. LOSS / OPTIMIZER
# ----------------------------
criterion = nn.CrossEntropyLoss()

# Only train classifier at first
optimizer = optim.Adam(
    model.classifier.parameters(),
    lr=0.001
)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=2
)

# ----------------------------
# 7. CHECKPOINT LOADING
# ----------------------------
checkpoint_path = "bird_checkpoint.pth"

start_epoch = 0
best_val_loss = float('inf')

if os.path.exists(checkpoint_path):

    print("Loading previous checkpoint...")

    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint['model_state'])
    #optimizer.load_state_dict(checkpoint['optimizer_state'])

    start_epoch = checkpoint['epoch'] + 1
    best_val_loss = checkpoint['best_val_loss']

    print(f"Resuming from epoch {start_epoch}")

# ----------------------------
# 8. EARLY STOPPING
# ----------------------------
patience = 5
patience_counter = 0

# ----------------------------
# 9. TRAINING LOOP
# ----------------------------
epochs = 60

print("\nBeginning Training...\n")

for epoch in range(start_epoch, epochs):

    # ----------------------------
    # UNFREEZE AFTER 5 EPOCHS
    # ----------------------------
    if epoch == 5:

        print("\nUnfreezing feature extractor...\n")

        for param in model.features.parameters():
            param.requires_grad = True

        optimizer = optim.Adam(
            model.parameters(),
            lr=1e-5
        )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=2
        )

    # ----------------------------
    # TRAINING
    # ----------------------------
    model.train()

    running_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = outputs.max(1)

        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total

    # ----------------------------
    # VALIDATION
    # ----------------------------
    model.eval()

    val_loss_total = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            val_loss_total += loss.item()

            _, predicted = outputs.max(1)

            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

    val_loss = val_loss_total / len(val_loader)
    val_acc = 100 * val_correct / val_total

    scheduler.step(val_loss)

    # ----------------------------
    # PRINT METRICS
    # ----------------------------
    print(f"Epoch {epoch+1}/{epochs}")

    print(f"Train Loss: {train_loss:.4f}")
    print(f"Train Accuracy: {train_acc:.2f}%")

    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_acc:.2f}%")

    print("-" * 40)

    # ----------------------------
    # SAVE BEST MODEL
    # ----------------------------
    if val_loss < best_val_loss:

        print("Validation improved. Saving checkpoint...\n")

        best_val_loss = val_loss
        patience_counter = 0

        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'best_val_loss': best_val_loss
        }, checkpoint_path)

    else:

        patience_counter += 1

        print(
            f"No significant improvement "
            f"({patience_counter}/{patience})"
        )

        if patience_counter >= patience:

            print("\nEarly stopping triggered.")
            break

# ----------------------------
# 10. EXPORT FOR RASPBERRY PI
# ----------------------------
print("\nExporting TorchScript model...\n")

model.eval()

example = torch.randn(1, 3, 224, 224).to(device)

traced_model = torch.jit.trace(model, example)

traced_model.save("bird_model.pt")

print("Saved model as bird_model.pt")