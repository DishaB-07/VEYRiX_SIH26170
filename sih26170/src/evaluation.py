"""
Evaluation utilities: lot-wise (never row-wise) train/test split, plus
classification metrics for the anomaly/QA decision and regression metrics
for the 168h drift forecast.

False negatives are treated as the headline concern throughout, per the PS:
recall / false-negative rate are surfaced first and most prominently.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix,
    average_precision_score, mean_absolute_error, mean_squared_error,
)


def split_by_lot(df: pd.DataFrame, test_fraction: float = 0.3, seed: int = 0) -> tuple[list[str], list[str]]:
    """Split by LOT, not by individual component, so no component from a
    test lot has ever contributed to fitting any model or statistic used to
    judge it. Deterministic given `seed`.
    """
    lot_ids = sorted(df["lot_id"].unique())
    if len(lot_ids) < 2:
        # Degenerate case: can't hold out a whole lot. Everything is both
        # train and test, and this is surfaced to the user as a warning by
        # the caller -- metrics in this situation are optimistic/invalid.
        return lot_ids, lot_ids

    rng = np.random.default_rng(seed)
    shuffled = lot_ids.copy()
    rng.shuffle(shuffled)
    n_test = max(1, int(round(len(shuffled) * test_fraction)))
    test_lots = sorted(shuffled[:n_test])
    train_lots = sorted(shuffled[n_test:])
    return train_lots, test_lots


@dataclass
class ClassificationReport:
    recall: float | None
    precision: float | None
    f1: float | None
    fnr: float | None
    pr_auc: float | None
    confusion: np.ndarray | None
    n_positive: int
    n_negative: int
    note: str = ""


def evaluate_classification(y_true: pd.Series, y_pred_flag: pd.Series, y_score: pd.Series) -> ClassificationReport:
    """y_true: ground_truth_anomaly (bool). y_pred_flag: model's flag
    (bool). y_score: continuous risk score in [0,1], used for PR-AUC."""
    y_true = y_true.astype(bool).to_numpy()
    y_pred = y_pred_flag.astype(bool).to_numpy()
    y_score = y_score.to_numpy()

    n_pos = int(y_true.sum())
    n_neg = int((~y_true).sum())

    if n_pos == 0:
        return ClassificationReport(
            None, None, None, None, None, None, n_pos, n_neg,
            note="No positive (anomalous) components in this set -- recall/precision/PR-AUC are undefined.",
        )

    recall = recall_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    fnr = 1.0 - recall
    cm = confusion_matrix(y_true, y_pred, labels=[False, True])

    pr_auc = None
    note = ""
    if n_neg == 0:
        note = "No normal components in this set -- PR-AUC is undefined."
    else:
        try:
            pr_auc = average_precision_score(y_true, y_score)
        except ValueError:
            note = "PR-AUC could not be computed for this set."

    return ClassificationReport(recall, precision, f1, fnr, pr_auc, cm, n_pos, n_neg, note)


@dataclass
class RegressionReport:
    mae: float
    rmse: float
    mae_baseline: float
    rmse_baseline: float
    n: int


def evaluate_regression(y_true: pd.Series, y_pred: pd.Series, y_pred_baseline: pd.Series) -> RegressionReport:
    y_true = y_true.to_numpy()
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae_b = mean_absolute_error(y_true, y_pred_baseline)
    rmse_b = float(np.sqrt(mean_squared_error(y_true, y_pred_baseline)))
    return RegressionReport(mae, rmse, mae_b, rmse_b, n=len(y_true))
