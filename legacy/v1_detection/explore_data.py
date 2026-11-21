import os

# Repo root (this file lives in legacy/v1_detection/)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

base = os.path.join(ROOT, "data", "raw")

for category in ["impact_beating", "gunshot", "drip"]:
    path = os.path.join(base, category)
    count = 0
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
                count += 1
    print(f"{category}: {count} images")
