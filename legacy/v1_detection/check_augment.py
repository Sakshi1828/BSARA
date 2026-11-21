import os

# Repo root (this file lives in legacy/v1_detection/)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

base = os.path.join(ROOT, "data", "augmented")
categories = ["impact_beating", "gunshot", "drip"]

total = 0
for cat in categories:
    folder = os.path.join(base, cat)
    images = [f for f in os.listdir(folder)
              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"{cat}: {len(images)} images")
    total += len(images)

print(f"\nTotal dataset size: {total} images")