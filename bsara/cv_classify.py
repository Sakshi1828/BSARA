"""4-fold cross-validated image-level pattern classification with YOLOv8n-cls.

Conditions (same model, same hyper-parameters, same folds where applicable):
  leaky_full            v1-style image-level split, original images
  grouped_full          group-aware split, original images
  grouped_stain_removed group-aware split, blood inpainted out (shortcut test)

Usage:  python -m bsara.cv_classify [--epochs 20] [--conditions ...]
Writes: results/predictions_<condition>.csv
"""
import argparse
import os
import shutil

import cv2
import pandas as pd

from bsara.folds import assign_folds, leaked_fraction
from bsara.paths import CLASSES, CV_DIR, INDEX_CSV, PROCESSED, RESULTS, ROOT
from bsara.stain_removal import remove_stains

CONDITIONS = {
    "leaky_full": ("leaky", "full"),
    "grouped_full": ("grouped", "full"),
    "grouped_stain_removed": ("grouped", "stain_removed"),
}
MAX_OVERSAMPLE = 40  # minority classes are repeated (and randomly augmented) up to this factor


STAIN_REMOVED = CV_DIR / "_stain_removed"


def image_source(variant: str):
    return PROCESSED if variant == "full" else STAIN_REMOVED


def ensure_stain_removed(df):
    for p in df["path"]:
        dst = STAIN_REMOVED / p
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst), remove_stains(cv2.imread(str(PROCESSED / p)))[0])


def link(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def materialise_fold(df, fold, k, src_root, fold_dir):
    if fold_dir.exists():
        shutil.rmtree(fold_dir)
    train, val = df[fold != k], df[fold == k]
    n_major = train["cls"].value_counts().max()
    for _, r in train.iterrows():
        n_cls = (train["cls"] == r["cls"]).sum()
        reps = min(MAX_OVERSAMPLE, max(1, round(n_major / n_cls)))
        name = r["path"].replace("/", "__")
        for i in range(reps):
            stem, ext = os.path.splitext(name)
            link(src_root / r["path"], fold_dir / "train" / r["cls"] / f"{stem}__r{i}{ext}")
    for _, r in val.iterrows():
        link(src_root / r["path"], fold_dir / "val" / r["cls"] / r["path"].replace("/", "__"))
    for split in ("train", "val"):
        for c in CLASSES:
            (fold_dir / split / c).mkdir(parents=True, exist_ok=True)
    return val


def train_and_predict(fold_dir, val, src_root, run_name, epochs):
    from ultralytics import YOLO

    # Classification head on the COCO-pretrained YOLOv8n backbone already in
    # the repo (no extra weight download needed).
    model = YOLO("yolov8n-cls.yaml").load(str(ROOT / "yolov8n.pt"))
    model.train(data=str(fold_dir), epochs=epochs, imgsz=224, batch=16, device=0,
                workers=0, amp=False, plots=False, seed=42, deterministic=True,
                project=str(ROOT / "runs" / "v2_cv"), name=run_name, exist_ok=True,
                verbose=False)
    best = YOLO(str(ROOT / "runs" / "v2_cv" / run_name / "weights" / "best.pt"))
    names = best.names
    rows = []
    for _, r in val.iterrows():
        res = best.predict(str(src_root / r["path"]), imgsz=224, verbose=False)[0]
        probs = res.probs.data.cpu().numpy()
        row = dict(path=r["path"], group=r["group"], true=r["cls"],
                   pred=names[int(probs.argmax())])
        row.update({f"p_{names[i]}": float(probs[i]) for i in range(len(probs))})
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    args = ap.parse_args()

    df = pd.read_csv(INDEX_CSV)
    RESULTS.mkdir(exist_ok=True)
    if any(CONDITIONS[c][1] == "stain_removed" for c in args.conditions):
        ensure_stain_removed(df)

    for cond in args.conditions:
        split_mode, variant = CONDITIONS[cond]
        fold = assign_folds(df, split_mode)
        print(f"\n=== {cond}: {leaked_fraction(df, fold):.1%} of validation images share a drop/scan with training")
        src_root = image_source(variant)
        preds = []
        for k in sorted(fold.unique()):
            fold_dir = CV_DIR / cond / f"fold{k}"
            done = CV_DIR / cond / f"fold{k}_predictions.csv"  # resume support
            if done.exists():
                rows = pd.read_csv(done).to_dict("records")
            else:
                val = materialise_fold(df, fold, k, src_root, fold_dir)
                rows = train_and_predict(fold_dir, val, src_root, f"{cond}_fold{k}", args.epochs)
                for r in rows:
                    r["fold"] = k
                pd.DataFrame(rows).to_csv(done, index=False)
            preds += rows
            acc = sum(r["true"] == r["pred"] for r in rows) / len(rows)
            print(f"  fold {k}: n_val={len(rows)} acc={acc:.3f}")
        pd.DataFrame(preds).to_csv(RESULTS / f"predictions_{cond}.csv", index=False)


if __name__ == "__main__":
    main()
