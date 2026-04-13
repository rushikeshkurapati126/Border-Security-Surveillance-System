import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split

# ==============================
# CONFIG
# ==============================
DATA_DIR = r"C:\Users\rushi\Downloads\security\archive\Dataset\train"
BATCH_SIZE = 32
EPOCHS = 10
LR = 0.001
MODEL_SAVE_PATH = "gun_classifier.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================
# TRANSFORMS
# ==============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

# ==============================
# LOAD DATASET
# ==============================
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)

print("Classes:", dataset.classes)  # should be ['gun', 'normal']

# ==============================
# TRAIN / VALID SPLIT
# ==============================
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# ==============================
# MODEL (Transfer Learning)
# ==============================
model = models.resnet18(pretrained=True)

# Replace final layer for 2 classes
model.fc = nn.Linear(model.fc.in_features, 2)

model = model.to(DEVICE)

# ==============================
# LOSS + OPTIMIZER
# ==============================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ==============================
# TRAINING LOOP
# ==============================
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0
    correct = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()

    train_acc = correct / len(train_dataset)

    # ==============================
    # VALIDATION
    # ==============================
    model.eval()
    val_correct = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            val_correct += (preds == labels).sum().item()

    val_acc = val_correct / len(val_dataset)

    print(f"Epoch [{epoch+1}/{EPOCHS}] "
          f"Loss: {running_loss:.4f} "
          f"Train Acc: {train_acc:.4f} "
          f"Val Acc: {val_acc:.4f}")

# ==============================
# SAVE MODEL
# ==============================
torch.save(model.state_dict(), MODEL_SAVE_PATH)

print(f"\n✅ Model saved at: {MODEL_SAVE_PATH}")