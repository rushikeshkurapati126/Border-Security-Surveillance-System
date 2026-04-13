import os
import random
import shutil

# Paths
source_folder = r"C:\Users\rushi\Downloads\security\archive\Dataset\images"
train_folder = r"C:\Users\rushi\Downloads\security\archive\Dataset\train"
test_folder = r"C:\Users\rushi\Downloads\security\archive\Dataset\test"

# Create folders
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# Get all images
images = [f for f in os.listdir(source_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Shuffle images
random.shuffle(images)

# Split ratio
split_ratio = 0.8
split_index = int(len(images) * split_ratio)

train_images = images[:split_index]
test_images = images[split_index:]

# Copy files
for img in train_images:
    shutil.copy(os.path.join(source_folder, img), os.path.join(train_folder, img))

for img in test_images:
    shutil.copy(os.path.join(source_folder, img), os.path.join(test_folder, img))

print("Total Images:", len(images))
print("Train Images:", len(train_images))
print("Test Images:", len(test_images))