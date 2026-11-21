import os
import cv2
import shutil
import numpy as np
from pathlib import Path

# Repo root (this file lives in legacy/v1_detection/)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Paths
RAW = os.path.join(ROOT, "data", "raw")
OUT = os.path.join(ROOT, "data", "processed")
SIZE = 640  # resize all images to 640x640

# Create output folders
categories = ["impact_beating", "gunshot", "drip"]
for cat in categories:
    os.makedirs(os.path.join(OUT, cat), exist_ok=True)

def resize_and_save(src_path, dst_path):
    img = cv2.imread(src_path)
    if img is None:
        print(f"Could not read: {src_path}")
        return
    img_resized = cv2.resize(img, (SIZE, SIZE))
    cv2.imwrite(dst_path, img_resized)

count = {"impact_beating": 0, "gunshot": 0, "drip": 0}

for cat in categories:
    src_folder = os.path.join(RAW, cat)
    dst_folder = os.path.join(OUT, cat)
    
    for root, dirs, files in os.walk(src_folder):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
                src = os.path.join(root, f)
                dst = os.path.join(dst_folder, f)
                # Handle duplicate filenames across subfolders
                if os.path.exists(dst):
                    base, ext = os.path.splitext(f)
                    dst = os.path.join(dst_folder, f"{base}_{count[cat]}{ext}")
                resize_and_save(src, dst)
                count[cat] += 1

    print(f"{cat}: {count[cat]} images processed → {dst_folder}")

print("\nDone! All images resized to 640x640")