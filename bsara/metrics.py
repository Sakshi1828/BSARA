"""Metrics for imbalanced 3-class evaluation, reported as mean +/- std over folds.

Plain accuracy is misleading here (94% drip), so the headline metrics are
balanced accuracy, macro-F1 and per-class recall.

Usage:  python -m bsara.metrics      -> results/metrics.md, results/confusion_*.png
"""
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, recall_score

from bsara.paths import CLASSES, RESULTS

LABELS = {
    "leaky_full": "CNN, v1-style image split (leaky)",
    "grouped_full": "CNN, group-aware split",
    "grouped_stain_removed": "CNN, group-aware split, **blood removed**",
    "stats_full": "Global colour stats + logistic regression",
    "stats_stain_removed": "Global colour stats, **blood removed**",
}


def fold_metrics(d):
    rec = recall_score(d["true"], d["pred"], labels=CLASSES, average=None, zero_division=0)
    m = dict(accuracy=(d["true"] == d["pred"]).mean(),
             balanced_acc=balanced_accuracy_score(d["true"], d["pred"]),
             macro_f1=f1_score(d["true"], d["pred"], labels=CLASSES, average="macro", zero_division=0))
    m.update({f"recall_{c}": r for c, r in zip(CLASSES, rec)})
    return m


def summarise(preds: pd.DataFrame) -> pd.Series:
    per_fold = pd.DataFrame([fold_metrics(d) for _, d in preds.groupby("fold")])
    return per_fold.mean()


def fmt(per_fold, col):
    return f"{per_fold[col].mean():.2f} ± {per_fold[col].std(ddof=0):.2f}"


def plot_confusion(preds, name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = confusion_matrix(preds["true"], preds["pred"], labels=CLASSES)
    norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]}\n({norm[i, j]:.0%})", ha="center", va="center",
                    color="white" if norm[i, j] > 0.6 else "black", fontsize=8)
    ax.set_xticks(range(3), CLASSES, rotation=20)
    ax.set_yticks(range(3), CLASSES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(name.replace("_", " "), fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULTS / f"confusion_{name}.png", dpi=150)
    plt.close(fig)


def main():
    lines = ["| Condition | Balanced acc. | Macro-F1 | Recall drip | Recall impact | Recall gunshot | Accuracy |",
             "|---|---|---|---|---|---|---|"]
    for name, label in LABELS.items():
        f = RESULTS / f"predictions_{name}.csv"
        if not f.exists():
            continue
        preds = pd.read_csv(f)
        pf = pd.DataFrame([fold_metrics(d) for _, d in preds.groupby("fold")])
        lines.append(f"| {label} | {fmt(pf, 'balanced_acc')} | {fmt(pf, 'macro_f1')} | "
                     f"{fmt(pf, 'recall_drip')} | {fmt(pf, 'recall_impact_beating')} | "
                     f"{fmt(pf, 'recall_gunshot')} | {fmt(pf, 'accuracy')} |")
        plot_confusion(preds, name)
    text = "\n".join(lines) + "\n\nMean ± std over 4 cross-validation folds.\n"
    (RESULTS / "metrics.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
