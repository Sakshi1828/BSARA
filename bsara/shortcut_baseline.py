"""How far can you get WITHOUT looking at bloodstain shape?

Fits a logistic regression on 'dumb' global image statistics (per-channel mean,
std and percentiles, border brightness, colour histogram) using the same
grouped folds as the CNN. Run on both the original and the stain-removed
images. High scores here mean each class is identifiable from photography
conditions alone, i.e. the dataset itself contains a source shortcut.

Usage:  python -m bsara.shortcut_baseline     (after cv_classify has created
        the stain-removed images)
"""
import cv2
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from bsara.cv_classify import STAIN_REMOVED, ensure_stain_removed
from bsara.folds import assign_folds
from bsara.metrics import summarise
from bsara.paths import INDEX_CSV, PROCESSED, RESULTS


def global_features(img):
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA).astype(np.float32)
    feats = []
    for c in range(3):
        ch = img[..., c]
        border = np.concatenate([ch[:16].ravel(), ch[-16:].ravel(), ch[:, :16].ravel(), ch[:, -16:].ravel()])
        feats += [ch.mean(), ch.std(), *np.percentile(ch, [5, 50, 95]), border.mean(), border.std()]
        feats += list(np.histogram(ch, bins=8, range=(0, 256))[0] / ch.size)
    return np.array(feats)


def run(df, root, fold, name):
    X = np.stack([global_features(cv2.imread(str(root / p))) for p in df["path"]])
    rows = []
    for k in sorted(fold.unique()):
        tr, va = fold != k, fold == k
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced"))
        clf.fit(X[tr], df.loc[tr, "cls"])
        for (_, r), pred in zip(df[va].iterrows(), clf.predict(X[va])):
            rows.append(dict(path=r["path"], group=r["group"], true=r["cls"], pred=pred, fold=k))
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / f"predictions_{name}.csv", index=False)
    return out


def main():
    df = pd.read_csv(INDEX_CSV)
    ensure_stain_removed(df)
    fold = assign_folds(df, "grouped")
    for name, root in [("stats_full", PROCESSED), ("stats_stain_removed", STAIN_REMOVED)]:
        print(name, summarise(run(df, root, fold, name)).round(3).to_dict())


if __name__ == "__main__":
    main()
