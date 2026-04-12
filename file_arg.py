import os
import shutil
import random

# Paths
base_path = r"D:\Programming\Python\Python_Projects\Traffic_Management_System\data"
images_path = os.path.join(base_path, "images")
labels_path = os.path.join(base_path, "labels")

train_images = os.path.join(base_path, "train", "images")
train_labels = os.path.join(base_path, "train", "labels")
val_images = os.path.join(base_path, "val", "images")
val_labels = os.path.join(base_path, "val", "labels")

# Create directories
for path in [train_images, train_labels, val_images, val_labels]:
    os.makedirs(path, exist_ok=True)

# Get all images
image_files = [f for f in os.listdir(images_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
random.shuffle(image_files)

# Train/val split ratio
split_ratio = 0.8
train_count = int(len(image_files) * split_ratio)

train_files = image_files[:train_count]
val_files = image_files[train_count:]

def move_files(file_list, target_img_dir, target_lbl_dir):
    for file in file_list:
        img_src = os.path.join(images_path, file)
        lbl_src = os.path.join(labels_path, file.replace(".jpg", ".txt").replace(".png", ".txt").replace(".jpeg", ".txt"))

        img_dst = os.path.join(target_img_dir, file)
        lbl_dst = os.path.join(target_lbl_dir, os.path.basename(lbl_src))

        # Move image
        shutil.copy(img_src, img_dst)

        # Move label if exists
        if os.path.exists(lbl_src):
            shutil.copy(lbl_src, lbl_dst)

# Move training files
move_files(train_files, train_images, train_labels)

# Move validation files
move_files(val_files, val_images, val_labels)

print(f"✅ Done! {len(train_files)} training and {len(val_files)} validation images prepared.")
