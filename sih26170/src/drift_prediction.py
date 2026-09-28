"""
Module B -- Time-Series Drift Predictor.

Predicts Value_168h from Value_0h + Value_24h (plus engineered features),
using gradient boosting. XGBoost is used if it is importable; otherwise the
code transparently falls back to scikit-learn's HistGradientBoostingRegressor
(no LSTM/GRU, per the implementation rules -- with only two real input
readings per part, tree-based tabular models are the better-justified,
faster, more explainable choice; see the accompanying research notes).

A plain linear-regression baseline is also trained and reported so the
gradient-boosting model's improvement is visible and honest rather than an
unverifiable number in isolation.

The "safety slope" is NOT a hard-coded number. It is derived from the
TRAINING lots' own drift-rate distribution using a robust
(median + k * MAD) statistic, which tolerates the small fraction of
anomalous parts already present in the training mix without needing ground
truth labels at deployment time.

Drift rate is expressed as a FRACTIONAL (relative) rate -- (predicted_168h -
Value_24h) / Value_24h / hours -- rather than a raw unit rate, and the
safety criterion is derived SEPARATELY PER parameter_type. Different
parameters (leakage current in uA, Iddq in uA, propagation delay in ns) sit
on very different absolute scales; pooling their raw drift rates into one
global threshold would systematically over- or under-flag whichever
parameter type happens to have larger raw numbers. This is displayed
everywhere as a prototype statistical criterion, never as an official ISRO
limit.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor

try:
    from xgboost import XGBRegressor
    _HAS_XGBOOST = True
except Exception:  # pragma: no cover - exercised when xgboost isn't installed
    _HAS_XGBOOST = False

FEATURE_BASE_COLS = [
    "Value_0h",
    "Value_24h",
    "delta_24_0",
    "early_slope",
    "relative_early_slope",
    "Value_0h_robust_z",
    "Value_24h_robust_z",
    "pct_of_datasheet_limit_0h",
    "pct_of_datasheet_limit_24h",
]
TARGET_COL = "Value_168h"
SAFETY_K_MAD = 3.0  # documented prototype multiplier, not an ISRO figure
LATE_WINDOW_H = 168.0 - 24.0
EPS = 1e-6


@dataclass
class DriftModelBundle:
    model: object
    baseline_model: object
    feature_cols: list
    model_name: str
    safety_slope_by_type: dict
    safety_slope_global: float
    safety_reference_mad_by_type: dict
    safety_reference_mad_global: float


def _make_model(random_state: int = 0):
    if _HAS_XGBOOST:
        return XGBRegressor(
            n_estimators=250,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=random_state,
        ), "XGBoost"
    return HistGradientBoostingRegressor(
        max_depth=4,
        learning_rate=0.06,
        max_iter=300,
        random_state=random_state,
    ), "HistGradientBoostingRegressor (XGBoost unavailable)"


def _relative_drift_rate(value_24h: pd.Series, value_168h: pd.Series) -> pd.Series:
    """Fractional change per hour between the 24h and 168h checkpoints.
    Scale-free, so it is comparable across different parameter types and
    different lot baselines."""
    return (value_168h - value_24h) / (value_24h.abs() + EPS) / LATE_WINDOW_H


def fit_drift_model(train_df: pd.DataFrame, random_state: int = 0) -> DriftModelBundle:
    """Fit the drift regressor AND derive the safety-slope criterion, using
    ONLY the rows in `train_df` (i.e. training-lot rows). Nothing here may
    ever see a test-lot row.
    """
    dummy_cols = sorted(c for c in train_df.columns if c.startswith("is_"))
    feature_cols = FEATURE_BASE_COLS + dummy_cols
    missing = [c for c in feature_cols if c not in train_df.columns]
    if missing:
        raise ValueError(f"train_df is missing engineered feature columns: {missing}")

    X = train_df[feature_cols].to_numpy()
    # Model the RELATIVE growth from 24h to 168h rather than the absolute
    # 168h value. Absolute-value targets tie the model to the specific
    # baseline magnitudes seen during training, which generalises poorly to
    # unseen lots with a different process baseline (tree models in
    # particular do not extrapolate outside the ranges they were trained
    # on). A relative growth-factor target is scale-free across lots and
    # parameter types, consistent with how the safety slope itself is
    # defined below.
    y_ratio = (train_df[TARGET_COL] - train_df["Value_24h"]) / (train_df["Value_24h"].abs() + EPS)

    # Cost-sensitive training: give known-anomalous training rows more
    # weight in the regression loss. This is a deliberate, documented
    # asymmetric-cost design choice, mirroring the PS's own instruction
    # that false negatives are the metric that matters most -- without
    # it, a gradient-boosted model trained on a population that is >90%
    # normal parts tends to under-fit the rare, nonlinear latent-drift
    # pattern precisely because it is rare. This is only possible because
    # the synthetic training data carries labels; a real deployment
    # without labels would need a different weighting signal (e.g. prior
    # Module-A anomaly scores) and this is called out explicitly in the UI.
    if "ground_truth_anomaly" in train_df.columns and train_df["ground_truth_anomaly"].notna().all():
        sample_weight = np.where(train_df["ground_truth_anomaly"].to_numpy(), 6.0, 1.0)
    else:
        sample_weight = None

    model, model_name = _make_model(random_state)
    if sample_weight is not None:
        model.fit(X, y_ratio.to_numpy(), sample_weight=sample_weight)
    else:
        model.fit(X, y_ratio.to_numpy())

    baseline = LinearRegression()
    if sample_weight is not None:
        baseline.fit(X, y_ratio.to_numpy(), sample_weight=sample_weight)
    else:
        baseline.fit(X, y_ratio.to_numpy())

    # Reference drift-rate distribution, computed on the ACTUAL training
    # values (not predictions), per parameter_type since different
    # parameters have very different natural fractional drift.
    rel_drift = _relative_drift_rate(train_df["Value_24h"], train_df["Value_168h"])
    tmp = pd.DataFrame({"parameter_type": train_df["parameter_type"].to_numpy(), "rel_drift": rel_drift.to_numpy()})

    slope_by_type, mad_by_type = {}, {}
    for ptype, grp in tmp.groupby("parameter_type"):
        med = float(grp["rel_drift"].median())
        mad = float((grp["rel_drift"] - med).abs().median())
        mad = max(mad, 1e-8)
        slope_by_type[ptype] = med + SAFETY_K_MAD * mad
        mad_by_type[ptype] = mad

    global_med = float(tmp["rel_drift"].median())
    global_mad = float((tmp["rel_drift"] - global_med).abs().median())
    global_mad = max(global_mad, 1e-8)
    slope_global = global_med + SAFETY_K_MAD * global_mad

    return DriftModelBundle(
        model=model,
        baseline_model=baseline,
        feature_cols=feature_cols,
        model_name=model_name,
        safety_slope_by_type=slope_by_type,
        safety_slope_global=slope_global,
        safety_reference_mad_by_type=mad_by_type,
        safety_reference_mad_global=global_mad,
    )


def score_module_b(df: pd.DataFrame, bundle: DriftModelBundle) -> pd.DataFrame:
    """Apply the already-fitted model/baseline/safety-slope to any dataframe
    (train or test) -- fitting never happens here, only inference."""
    df = df.copy()
    X = df[bundle.feature_cols].to_numpy()

    predicted_ratio = bundle.model.predict(X)
    predicted_ratio_baseline = bundle.baseline_model.predict(X)
    df["predicted_168h"] = df["Value_24h"] * (1 + predicted_ratio)
    df["predicted_168h_baseline_linear"] = df["Value_24h"] * (1 + predicted_ratio_baseline)

    df["projected_drift_rate"] = _relative_drift_rate(df["Value_24h"], df["predicted_168h"])

    df["safety_slope"] = df["parameter_type"].map(bundle.safety_slope_by_type).fillna(bundle.safety_slope_global)
    ref_mad = df["parameter_type"].map(bundle.safety_reference_mad_by_type).fillna(bundle.safety_reference_mad_global)

    df["module_b_flag"] = df["projected_drift_rate"] > df["safety_slope"]

    # 0-1 risk score: how far past the safety slope, relative to the
    # reference MAD for that parameter type, saturating softly so one wild
    # point can't dominate. 0.5 = exactly at the safety slope.
    excess = (df["projected_drift_rate"] - df["safety_slope"]) / (ref_mad * SAFETY_K_MAD + EPS)
    df["module_b_score"] = (1 / (1 + np.exp(-3 * excess))).clip(0, 1)

    return df
