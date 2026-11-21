import os
import shutil

# Repo root (this file lives in legacy/v1_detection/)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA = os.path.join(ROOT, "data")

# Source folders
DRIP_IMAGES  = os.path.join(DATA, "images")
DRIP_LABELS  = os.path.join(DATA, "labels")
AUTO_IMAGES  = os.path.join(DATA, "auto_annotated", "images")
AUTO_LABELS  = os.path.join(DATA, "auto_annotated", "labels")

# Destination
FINAL_IMAGES = os.path.join(DATA, "final_dataset", "images")
FINAL_LABELS = os.path.join(DATA, "final_dataset", "labels")

os.makedirs(FINAL_IMAGES, exist_ok=True)
os.makedirs(FINAL_LABELS, exist_ok=True)

def copy_files(src_img, src_lbl, label):
    images = [f for f in os.listdir(src_img) if f.endswith('.jpg')]
    count = 0
    for img in images:
        lbl = os.path.splitext(img)[0] + ".txt"
        lbl_path = os.path.join(src_lbl, lbl)
        if os.path.exists(lbl_path):
            shutil.copy(os.path.join(src_img, img), os.path.join(FINAL_IMAGES, img))
            shutil.copy(lbl_path, os.path.join(FINAL_LABELS, lbl))
            count += 1
    print(f"{label}: {count} files copied")

copy_files(DRIP_IMAGES, DRIP_LABELS, "drip")
copy_files(AUTO_IMAGES, AUTO_LABELS, "impact_beating + gunshot")

total = len(os.listdir(FINAL_IMAGES))
print(f"\n✅ Final dataset: {total} images ready")
print(f"Saved to: {DATA}\\final_dataset")