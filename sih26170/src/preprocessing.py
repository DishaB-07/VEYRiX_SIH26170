"""
Data validation, lot-wise statistics, and feature engineering.

These utilities are used by BOTH Module A (anomaly detection) and Module B
(drift prediction), so they live in one shared place.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data_generator import PARAMETER_PROFILES

REQUIRED_COLUMNS = [
    "component_id", "lot_id", "component_type", "parameter_type",
    "Value_0h", "Value_24h", "Value_96h", "Value_168h", "datasheet_limit",
]
VALUE_COLS = ["Value_0h", "Value_24h", "Value_96h", "Value_168h"]

MIN_LOT_SIZE_FOR_STATS = 5  # below this, per-lot MAD/median is unstable
EPSILON = 1e-6


def validate_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Validate schema and basic sanity of the burn-in dataset.

    Returns (cleaned_df, issues) where `issues` is a list of human-readable
    warnings/errors. Rows that cannot be salvaged (missing required values,
    non-positive readings) are dropped and reported, rather than silently
    breaking downstream stats.
    """
    issues: list[str] = []
    df = df.copy()

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        issues.append(f"Missing required columns: {missing_cols}")
        return df, issues

    n_before = len(df)

    # duplicate component ids
    dup_mask = df["component_id"].duplicated(keep=False)
    if dup_mask.any():
        issues.append(f"{dup_mask.sum()} duplicate component_id rows found and de-duplicated (kept first).")
        df = df[~df["component_id"].duplicated(keep="first")]

    # required numeric columns must be present and finite
    for col in VALUE_COLS + ["datasheet_limit"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    bad_mask = df[VALUE_COLS + ["datasheet_limit"]].isna().any(axis=1)
    if bad_mask.any():
        issues.append(f"{bad_mask.sum()} rows dropped: missing/non-numeric parametric values.")
        df = df[~bad_mask]

    non_positive_mask = (df[VALUE_COLS] <= 0).any(axis=1) | (df["datasheet_limit"] <= 0)
    if non_positive_mask.any():
        issues.append(f"{non_positive_mask.sum()} rows dropped: non-positive parametric reading or limit.")
        df = df[~non_positive_mask]

    # lots that are too small for reliable statistics (warn, don't drop)
    lot_sizes = df.groupby("lot_id").size()
    small_lots = lot_sizes[lot_sizes < MIN_LOT_SIZE_FOR_STATS].index.tolist()
    if small_lots:
        issues.append(
            f"{len(small_lots)} lot(s) have < {MIN_LOT_SIZE_FOR_STATS} components "
            f"({small_lots}); their lot-relative statistics will fall back to a "
            f"population-level reference and should be treated as low-confidence."
        )

    if "ground_truth_anomaly" not in df.columns:
        df["ground_truth_anomaly"] = np.nan
    if "anomaly_type" not in df.columns:
        df["anomaly_type"] = "unknown"

    n_after = len(df)
    if n_after == 0:
        issues.append("No valid rows remain after validation.")
    elif n_after < n_before:
        issues.append(f"Validation kept {n_after}/{n_before} rows.")

    return df.reset_index(drop=True), issues


def compute_lot_stats(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Per-lot median & MAD (Median Absolute Deviation) for a given
    checkpoint column. Lots smaller than MIN_LOT_SIZE_FOR_STATS still get a
    median/MAD, but are flagged `low_confidence=True` so the anomaly logic
    can widen its tolerance instead of overreacting to noisy small-n stats.
    """
    def _mad(x: pd.Series) -> float:
        med = x.median()
        return float((x - med).abs().median())

    grouped = df.groupby("lot_id")[value_col]
    stats = grouped.agg(median="median", n="count").reset_index()
    stats["mad"] = grouped.apply(_mad).reset_index(drop=True)
    stats["low_confidence"] = stats["n"] < MIN_LOT_SIZE_FOR_STATS

    # Population-level fallback reference, used to stabilise tiny lots.
    pop_median = df[value_col].median()
    pop_mad = float((df[value_col] - pop_median).abs().median())
    stats.loc[stats["low_confidence"], "median"] = (
        0.5 * stats.loc[stats["low_confidence"], "median"] + 0.5 * pop_median
    )
    stats.loc[stats["low_confidence"], "mad"] = (
        0.5 * stats.loc[stats["low_confidence"], "mad"] + 0.5 * pop_mad
    )
    # Guard against a degenerate MAD (e.g. every part in a lot reads
    # identically) which would otherwise blow up the robust Z-score.
    fallback_mad = max(pop_mad, EPSILON)
    stats["mad"] = stats["mad"].apply(lambda m: m if m > EPSILON else 0.15 * fallback_mad + EPSILON)
    stats = stats.rename(columns={"median": f"{value_col}_lot_median", "mad": f"{value_col}_lot_mad",
                                   "n": f"{value_col}_lot_n",
                                   "low_confidence": f"{value_col}_low_confidence"})
    return stats


def robust_z_score(value: pd.Series, median: pd.Series, mad: pd.Series) -> pd.Series:
    """0.6745 * (value - median) / (MAD + epsilon).

    0.6745 rescales MAD to be comparable to a normal-distribution standard
    deviation, so the same '3.5' style threshold used for classic Z-scores
    remains meaningful here.
    """
    return 0.6745 * (value - median) / (mad + EPSILON)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Attach lot statistics and engineered delta/slope/lot-relative
    features needed by both Module A and Module B."""
    df = df.copy()

    for col in VALUE_COLS:
        lot_stats = compute_lot_stats(df, col)
        df = df.merge(lot_stats, on="lot_id", how="left")
        df[f"{col}_robust_z"] = robust_z_score(
            df[col], df[f"{col}_lot_median"], df[f"{col}_lot_mad"]
        )

    df["delta_24_0"] = df["Value_24h"] - df["Value_0h"]
    df["early_slope"] = df["delta_24_0"] / 24.0
    df["relative_early_slope"] = df["delta_24_0"] / (df["Value_0h"].abs() + EPSILON) / 24.0
    df["pct_of_datasheet_limit_0h"] = df["Value_0h"] / df["datasheet_limit"]
    df["pct_of_datasheet_limit_24h"] = df["Value_24h"] / df["datasheet_limit"]

    # One-hot parameter-type indicators. Different parameters (leakage vs.
    # Iddq vs. propagation delay) live on wildly different absolute scales,
    # so the drift model needs an explicit signal for which one it's
    # looking at, and downstream safety-slope logic is computed relative
    # (fractional) rather than in raw units for the same reason.
    for pname in PARAMETER_PROFILES.keys():
        df[f"is_{pname}"] = (df["parameter_type"] == pname).astype(int)

    return df
