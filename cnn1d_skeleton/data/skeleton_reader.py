from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path

def read_skeleton_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Skeleton file is tab-separated (TSV):
    columns: Frame, Time, then joint features (X,Y,Z,QX,QY,QZ,QW)...
    Returns:
      times: (T,)
      X: (T, F)
      feat_names: list[str]
    """
    df = pd.read_csv(path, sep=None, engine="python")

    if "Time" not in df.columns:
        raise ValueError(f"Missing 'Time' column in skeleton file: {path}")

    times = df["Time"].to_numpy(dtype=np.float32)

    drop_cols = [c for c in ["Frame", "Time"] if c in df.columns]
    feat_df = df.drop(columns=drop_cols)

    # Ensure numeric, fill gaps safely
    feat_df = feat_df.apply(pd.to_numeric, errors="coerce")
    feat_df = feat_df.ffill().bfill().fillna(0.0)

    X = feat_df.to_numpy(dtype=np.float32)
    feat_names = feat_df.columns.tolist()
    return times, X, feat_names