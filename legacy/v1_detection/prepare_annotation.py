# prepare_annotation_batch.py
import os
import shutil

# Repo root (this file lives in legacy/v1_detection/)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

AUGMENTED = os.path.join(ROOT, "data", "augmented")
ANNOTATION_BATCH = os.path.join(ROOT, "data", "to_annotate")

categories = ["drip", "impact_beating", "gunshot"]

# We want ~50 originals per class
LIMIT = 50

os.makedirs(ANNOTATION_BATCH, exist_ok=True)

for cat in categories:
    folder = os.path.join(AUGMENTED, cat)
    files = [f for f in os.listdir(folder) if f.endswith("_original.jpg")]
    
    selected = files[:LIMIT]
    
    for f in selected:
        src = os.path.join(folder, f)
        # Prefix filename with category so you know what class it is
        dst = os.path.join(ANNOTATION_BATCH, f"{cat}__{f}")
        shutil.copy(src, dst)
    
    print(f"{cat}: copied {len(selected)} originals")

print(f"\nAll originals ready in: {ANNOTATION_BATCH}")