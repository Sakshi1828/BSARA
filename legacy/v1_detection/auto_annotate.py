import cv2
import os
import numpy as np

# Repo root (this file lives in legacy/v1_detection/)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

INPUT = os.path.join(ROOT, "data", "to_annotate")
OUTPUT = os.path.join(ROOT, "data", "auto_annotated")

# Class mapping
CLASS_MAP = {
    "impact_beating": 1,
    "gunshot": 2
}

os.makedirs(OUTPUT, exist_ok=True)
# Create subfolders for images and labels
os.makedirs(os.path.join(OUTPUT, "images"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT, "labels"), exist_ok=True)

images = [f for f in os.listdir(INPUT) if f.endswith('.jpg')]
success = 0
failed = 0

for img_file in images:
    # Get class from filename prefix
    class_name = img_file.split("__")[0]
    if class_name not in CLASS_MAP:
        print(f"Skipping unknown class: {img_file}")
        continue

    class_id = CLASS_MAP[class_name]

    # Read image
    img_path = os.path.join(INPUT, img_file)
    img = cv2.imread(img_path)
    if img is None:
        continue

    h, w = img.shape[:2]

    # Convert to grayscale and threshold to find dark stain
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Invert so dark stains become bright
    inverted = cv2.bitwise_not(gray)

    # Threshold — anything darker than 200 becomes white
    _, thresh = cv2.threshold(inverted, 55, 255, cv2.THRESH_BINARY)

    # Find contours (blobs)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print(f"No stain found: {img_file}")
        failed += 1
        continue

    # Get the largest contour — that's our stain
    largest = max(contours, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(largest)

    # Add small padding around the box
    pad = 10
    x = max(0, x - pad)
    y = max(0, y - pad)
    bw = min(w - x, bw + 2 * pad)
    bh = min(h - y, bh + 2 * pad)

    # Convert to YOLO format (normalized center x, center y, width, height)
    cx = (x + bw / 2) / w
    cy = (y + bh / 2) / h
    nw = bw / w
    nh = bh / h

    # Save image
    cv2.imwrite(os.path.join(OUTPUT, "images", img_file), img)

    # Save label
    label_file = os.path.splitext(img_file)[0] + ".txt"
    with open(os.path.join(OUTPUT, "labels", label_file), "w") as f:
        f.write(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

    success += 1

print(f"\n✅ Done!")
print(f"Successfully annotated: {success} images")
print(f"Failed (no stain found): {failed} images")
print(f"Labels saved to: {OUTPUT}/labels")