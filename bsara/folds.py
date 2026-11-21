"""Train/validation fold assignment.

grouped : StratifiedGroupKFold - every version of a drop / every scan stays on
          one side of the split (the correct protocol).
leaky   : StratifiedKFold on individual images - reproduces the v1 protocol,
          where edited copies of the same drop land in train AND validation.
"""
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

N_SPLITS = 4  # only 8 gunshot scans exist -> 2 held out per fold
SEED = 42


def assign_folds(df: pd.DataFrame, mode: str) -> pd.Series:
    if mode == "grouped":
        splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        splits = splitter.split(df, df["cls"], groups=df["group"])
    elif mode == "leaky":
        splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        splits = splitter.split(df, df["cls"])
    else:
        raise ValueError(mode)
    fold = pd.Series(-1, index=df.index)
    for k, (_, val_idx) in enumerate(splits):
        fold.iloc[val_idx] = k
    return fold


def leaked_fraction(df: pd.DataFrame, fold: pd.Series) -> float:
    """Share of validation images whose group also appears in that fold's training set."""
    leaked = 0
    for k in sorted(fold.unique()):
        train_groups = set(df.loc[fold != k, "group"])
        leaked += df.loc[fold == k, "group"].isin(train_groups).sum()
    return leaked / len(df)
