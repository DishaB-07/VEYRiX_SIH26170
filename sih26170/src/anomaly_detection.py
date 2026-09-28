"""
Module A -- Dynamic Lot-Relative Anomaly Detection.

Primary signal: robust Z-score built from each lot's own median and MAD
(Median Absolute Deviation), computed independently per checkpoint. This is
the direct, explainable, peer-reviewed approach for exactly this problem
(lot-relative Iddq/leakage outlier screening).

Secondary signal: Isolation Forest over the lot-normalised multi-checkpoint
signature (each value expressed as a ratio to its own lot's median), so that
correlated drift across multiple checkpoints -- which a single-checkpoint
Z-score could miss -- is also caught. The Isolation Forest is fit ONLY on
the training-lot rows passed in, and reused (never refit) on test/unseen
rows, so it cannot leak information from held-out lots.

Both signals are fused into one transparent 0-1 risk score. The fusion is a
plain weighted maximum, not another model, so it stays auditable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .preprocessing import VALUE_COLS, EPSILON

Z_THRESHOLD_DEFAULT = 3.5  # documented statistical engineering threshold, not an ISRO limit
Z_CAP_FOR_SCORING = 10.0   # for normalising the 0-1 risk score display only


def fit_isolation_forest(train_df: pd.DataFrame, random_state: int = 0) -> IsolationForest:
    """Fit Isolation Forest on TRAIN rows only, using lot-normalised ratios
    (value / that row's own lot median) at each checkpoint so the model
    compares shape-of-trajectory, not raw magnitude, across different
    parameter types and lot baselines.

    contamination is left at 'auto' rather than a hand-picked global
    fraction, per the instruction not to assume a fixed contamination rate
    that may not match reality.
    """
    X = _isolation_features(train_df)
    model = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=random_state,
    )
    model.fit(X)
    return model


def _isolation_features(df: pd.DataFrame) -> np.ndarray:
    cols = []
    for col in VALUE_COLS:
        median_col = f"{col}_lot_median"
        ratio = df[col] / (df[median_col] + EPSILON)
        cols.append(ratio.to_numpy())
    return np.column_stack(cols)


def score_module_a(
    df: pd.DataFrame,
    if_model: IsolationForest,
    z_threshold: float = Z_THRESHOLD_DEFAULT,
) -> pd.DataFrame:
    """Score every row in `df` (already feature-engineered, see
    preprocessing.engineer_features) using the fitted Isolation Forest plus
    the live robust-Z statistics already attached to each row.

    Returns df with added columns:
      max_abs_robust_z, driving_checkpoint, if_anomaly_score (0-1, higher = more anomalous),
      module_a_flag (bool), module_a_score (0-1 fused risk score)
    """
    df = df.copy()

    z_cols = [f"{c}_robust_z" for c in VALUE_COLS]
    z_matrix = df[z_cols].to_numpy()
    abs_z = np.abs(z_matrix)
    df["max_abs_robust_z"] = abs_z.max(axis=1)
    driving_idx = abs_z.argmax(axis=1)
    checkpoint_names = [c.replace("_robust_z", "") for c in z_cols]
    df["driving_checkpoint"] = [checkpoint_names[i] for i in driving_idx]

    X = _isolation_features(df)
    # decision_function: higher = more normal. Flip and min-max scale to 0-1.
    raw = -if_model.decision_function(X)
    lo, hi = raw.min(), raw.max()
    if hi - lo < EPSILON:
        if_score = np.zeros_like(raw)
    else:
        if_score = (raw - lo) / (hi - lo)
    df["if_anomaly_score"] = if_score
    df["if_flag"] = if_model.predict(X) == -1

    z_score_norm = np.clip(df["max_abs_robust_z"] / Z_CAP_FOR_SCORING, 0, 1)
    # Weighted-max fusion: a strong single signal is not diluted away, but
    # agreement between both signals still pushes the score higher.
    fused = 0.7 * np.maximum(z_score_norm, df["if_anomaly_score"]) + \
            0.3 * np.minimum(z_score_norm, df["if_anomaly_score"])
    df["module_a_score"] = fused.clip(0, 1)

    df["module_a_flag"] = (df["max_abs_robust_z"] > z_threshold) | df["if_flag"]

    return df
