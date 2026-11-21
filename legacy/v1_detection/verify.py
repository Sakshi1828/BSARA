import os
import cv2

# Repo root (this file lives in legacy/v1_detection/)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

base = os.path.join(ROOT, "data", "processed")
categories = ["impact_beating", "gunshot", "drip"]

for cat in categories:
    folder = os.path.join(base, cat)
    images = [f for f in os.listdir(folder) 
              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # Check first image dimensions
    if images:
        sample = cv2.imread(os.path.join(folder, images[0]))
        h, w, c = sample.shape
        print(f"{cat}:")
        print(f"  Total images: {len(images)}")
        print(f"  Sample size: {w}x{h} pixels")
        print(f"  Channels: {c} (3 = colour RGB)")
        print()