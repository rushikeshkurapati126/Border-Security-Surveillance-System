import os
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image

# ==============================
# CONFIG
# ==============================
MODEL_PATH = "gun_classifier.pth"
TEST_DIR = r"C:\Users\rushi\Downloads\security\archive\Dataset\test"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================
# LOAD MODEL
# ==============================
model = resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# ==============================
# TRANSFORM
# ==============================
weights = ResNet18_Weights.DEFAULT
transform = weights.transforms()

# ==============================
# CLASS NAMES
# ==============================
classes = ["gun", "normal"]

# ==============================
# DEBUG: CHECK PATH
# ==============================
if not os.path.exists(TEST_DIR):
    print("❌ Test folder not found!")
    exit()

print("✅ Test folder found")
print("Folders:", os.listdir(TEST_DIR))

# ==============================
# TEST LOOP
# ==============================
correct = 0
total = 0

for class_name in os.listdir(TEST_DIR):
    class_path = os.path.join(TEST_DIR, class_name)

    if not os.path.isdir(class_path):
        continue

    print(f"\n📂 Class: {class_name}")

    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)

        try:
            image = Image.open(img_path).convert("RGB")
            image = transform(image).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                outputs = model(image)
                _, pred = torch.max(outputs, 1)

            predicted_class = classes[pred.item()]

            print(f"{img_name} -> {predicted_class}")

            if class_name == predicted_class:
                correct += 1

            total += 1

        except Exception as e:
            print(f"❌ Error with {img_name}: {e}")

# ==============================
# ACCURACY
# ==============================
if total > 0:
    print(f"\n✅ Accuracy: {(correct/total)*100:.2f}%")
else:
    print("❌ No images found!")