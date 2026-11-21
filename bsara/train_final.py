"""Train the demo classifier on ALL images (used by app.py).

Evaluation numbers come from cv_classify.py, never from this model; it has
seen every image, so its own validation score is meaningless.

Usage:  python -m bsara.train_final  ->  models/bsara_cls_v2.pt
"""
import shutil

import pandas as pd

from bsara.cv_classify import materialise_fold
from bsara.paths import CV_DIR, INDEX_CSV, PROCESSED, ROOT


def main(epochs=15):
    from ultralytics import YOLO

    df = pd.read_csv(INDEX_CSV)
    # Ultralytics requires a val split, so 2 token images per class go to
    # "val" (fold 0) and are then also copied into train.
    token_val = df.groupby("cls").head(2).index
    fold = pd.Series(-1, index=df.index)
    fold.loc[token_val] = 0
    fold_dir = CV_DIR / "final"
    materialise_fold(df, fold, 0, PROCESSED, fold_dir)
    for r in df.loc[token_val].itertuples():
        src = fold_dir / "val" / r.cls / r.path.replace("/", "__")
        shutil.copy2(src, fold_dir / "train" / r.cls / src.name)

    model = YOLO("yolov8n-cls.yaml").load(str(ROOT / "yolov8n.pt"))
    model.train(data=str(fold_dir), epochs=epochs, imgsz=224, batch=16, device=0, workers=0,
                amp=False, plots=False, seed=42, deterministic=True,
                project=str(ROOT / "runs" / "v2_final"), name="final", exist_ok=True, verbose=False)
    out = ROOT / "models" / "bsara_cls_v2.pt"
    out.parent.mkdir(exist_ok=True)
    shutil.copy2(ROOT / "runs" / "v2_final" / "final" / "weights" / "last.pt", out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
