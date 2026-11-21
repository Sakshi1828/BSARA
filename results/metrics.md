| Condition | Balanced acc. | Macro-F1 | Recall drip | Recall impact | Recall gunshot | Accuracy |
|---|---|---|---|---|---|---|
| CNN, v1-style image split (leaky) | 0.99 ± 0.01 | 0.98 ± 0.03 | 1.00 ± 0.00 | 0.98 ± 0.03 | 1.00 ± 0.00 | 1.00 ± 0.00 |
| CNN, group-aware split | 0.99 ± 0.01 | 0.98 ± 0.03 | 1.00 ± 0.00 | 0.98 ± 0.03 | 1.00 ± 0.00 | 1.00 ± 0.00 |
| CNN, group-aware split, **blood removed** | 0.95 ± 0.07 | 0.97 ± 0.05 | 1.00 ± 0.00 | 0.98 ± 0.03 | 0.88 ± 0.22 | 1.00 ± 0.00 |
| Global colour stats + logistic regression | 0.74 ± 0.13 | 0.69 ± 0.08 | 1.00 ± 0.00 | 0.72 ± 0.08 | 0.50 ± 0.35 | 0.98 ± 0.01 |
| Global colour stats, **blood removed** | 0.86 ± 0.08 | 0.54 ± 0.03 | 0.84 ± 0.01 | 0.85 ± 0.15 | 0.88 ± 0.22 | 0.85 ± 0.01 |

Mean ± std over 4 cross-validation folds.
