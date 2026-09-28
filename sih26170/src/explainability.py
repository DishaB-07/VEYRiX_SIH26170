"""
Turns the numeric outputs of Module A and Module B into plain-language
explanations a QA inspector can act on, and combines both modules into one
PASS / WATCH / REJECT decision with a transparent, auditable fusion rule
(no second model sits between the scores and the decision).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WATCH_THRESHOLD_DEFAULT = 0.33
REJECT_THRESHOLD_DEFAULT = 0.66


def combine_risk_score(module_a_score: pd.Series, module_b_score: pd.Series) -> pd.Series:
    """Weighted-max fusion so a single strongly-flagged module is never
    diluted away by the other module looking calm -- deliberately biased
    toward recall, per the PS's false-negative emphasis."""
    a = module_a_score.to_numpy()
    b = module_b_score.to_numpy()
    fused = 0.75 * np.maximum(a, b) + 0.25 * np.minimum(a, b)
    return pd.Series(fused, index=module_a_score.index).clip(0, 1)


def decide(
    risk_score: pd.Series,
    watch_threshold: float = WATCH_THRESHOLD_DEFAULT,
    reject_threshold: float = REJECT_THRESHOLD_DEFAULT,
) -> pd.Series:
    def _band(s):
        if s >= reject_threshold:
            return "REJECT"
        if s >= watch_threshold:
            return "WATCH"
        return "PASS"
    return risk_score.apply(_band)


def explain_module_a(row: pd.Series, z_threshold: float) -> str:
    if not row.get("module_a_flag", False):
        return (
            f"No lot-relative anomaly detected. {row['driving_checkpoint']} reads "
            f"robust Z = {row['max_abs_robust_z']:.2f} against this lot's own median "
            f"(threshold {z_threshold:.1f})."
        )
    value_col = row["driving_checkpoint"]
    value = row[value_col]
    median_col = f"{value_col}_lot_median"
    lot_median = row[median_col]
    z = row["max_abs_robust_z"]
    limit = row["datasheet_limit"]
    under_limit = value < limit
    limit_clause = (
        f"Although the value is below the datasheet limit of {limit:.1f}, "
        if under_limit else
        f"The value also exceeds the datasheet limit of {limit:.1f}. "
    )
    return (
        f"FLAGGED (Module A): {value_col} = {value:.2f}. "
        f"Lot median = {lot_median:.2f}. Robust Z-score = {z:.2f} "
        f"(threshold {z_threshold:.1f}). {limit_clause}"
        f"it is statistically abnormal relative to its manufacturing lot."
    )


def explain_module_b(row: pd.Series) -> str:
    predicted = row["predicted_168h"]
    slope_pct = row["safety_slope"] * 100
    projected_pct = row["projected_drift_rate"] * 100
    early_slope = row["early_slope"]
    if not row.get("module_b_flag", False):
        return (
            f"Predicted Value_168h = {predicted:.2f}. Projected relative drift rate "
            f"{projected_pct:.3f}%/h is within the prototype safety criterion "
            f"for this parameter type ({slope_pct:.3f}%/h)."
        )
    return (
        f"FLAGGED (Module B): Predicted Value_168h = {predicted:.2f}. "
        f"Projected relative drift rate ({projected_pct:.3f}%/h) exceeds the "
        f"prototype safety criterion for this parameter type ({slope_pct:.3f}%/h). "
        f"Early 0-24h slope was {early_slope:.4f} units/h, indicating abnormal "
        f"early degradation."
    )


def explain_decision(row: pd.Series, z_threshold: float) -> str:
    a_text = explain_module_a(row, z_threshold)
    b_text = explain_module_b(row)
    decision = row["qa_decision"]
    header = {
        "REJECT": "QA DECISION: REJECT",
        "WATCH": "QA DECISION: WATCH -- recommend extended burn-in / engineering review",
        "PASS": "QA DECISION: PASS",
    }[decision]
    return f"{header}\n\n[Module A] {a_text}\n\n[Module B] {b_text}"
