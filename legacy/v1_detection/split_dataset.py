import os
import shutil
import random

# Repo root (this file lives in legacy/v1_detection/)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA = os.path.join(ROOT, "data")
FINAL = os.path.join(DATA, "final_dataset")
SPLIT = os.path.join(DATA, "split_dataset")

# Create folder structure YOLOv8 expects
for folder in ["train/images", "train/labels", "val/images", "val/labels"]:
    os.makedirs(os.path.join(SPLIT, folder), exist_ok=True)

# Get all images
images = [f for f in os.listdir(os.path.join(FINAL, "images")) 
          if f.endswith('.jpg')]

# Shuffle randomly so split is not biased
random.seed(42)
random.shuffle(images)

# 80% train, 20% val
split_point = int(len(images) * 0.8)
train_images = images[:split_point]
val_images = images[split_point:]

def copy_split(file_list, split_name):
    copied = 0
    for img_file in file_list:
        label_file = os.path.splitext(img_file)[0] + ".txt"
        
        img_src = os.path.join(FINAL, "images", img_file)
        lbl_src = os.path.join(FINAL, "labels", label_file)
        
        if os.path.exists(lbl_src):
            shutil.copy(img_src, os.path.join(SPLIT, split_name, "images", img_file))
            shutil.copy(lbl_src, os.path.join(SPLIT, split_name, "labels", label_file))
            copied += 1
    print(f"{split_name}: {copied} images")

copy_split(train_images, "train")
copy_split(val_images, "val")

print(f"\n✅ Dataset split complete!")
print(f"Total: {len(images)} images")
print(f"Train: {len(train_images)} | Val: {len(val_images)}")