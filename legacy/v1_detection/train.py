import os
from ultralytics import YOLO

# Repo root (this file lives in legacy/v1_detection/)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# On Windows, multiprocessing requires this wrapper
# Without it, Python tries to spawn worker processes before
# the main program is ready — causing the error you saw
if __name__ == '__main__':
    
    import yaml

    # Resolve the dataset path in dataset.yaml against the repo root
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset.yaml")) as fh:
        cfg = yaml.safe_load(fh)
    cfg["path"] = os.path.join(ROOT, cfg["path"])
    resolved_yaml = os.path.join(ROOT, "runs", "dataset_resolved.yaml")
    os.makedirs(os.path.dirname(resolved_yaml), exist_ok=True)
    with open(resolved_yaml, "w") as fh:
        yaml.safe_dump(cfg, fh)

    # Load pretrained YOLOv8 nano model
    model = YOLO(os.path.join(ROOT, "yolov8n.pt"))

    # Start training
    results = model.train(
        data=resolved_yaml,
        epochs=50,
        imgsz=640,
        batch=16,
        project=os.path.join(ROOT, "runs"),
        name="bsara_v1",
        device=0,
        workers=4,      # reduced from 8 to be safer on Windows
        verbose=True
    )

    print("\n✅ Training complete!")
    print(f"Results saved to: runs/bsara_v1")