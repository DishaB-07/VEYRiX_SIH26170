"""
Orchestrates the full pipeline:

Raw df -> validate -> engineer_features -> lot-wise split ->
fit Module A (Isolation Forest) + Module B (drift model, safety slope) on
TRAIN lots only -> score TRAIN and TEST rows with the frozen fitted
artifacts -> combine into risk score -> PASS/WATCH/REJECT -> explanations.

This module has no Streamlit dependency so it can be imported and smoke-
tested from a plain Python script.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .preprocessing import validate_dataset, engineer_features
from .anomaly_detection import fit_isolation_forest, score_module_a, Z_THRESHOLD_DEFAULT
from .drift_prediction import fit_drift_model, score_module_b
from .evaluation import split_by_lot, evaluate_classification, evaluate_regression
from .explainability import combine_risk_score, decide, WATCH_THRESHOLD_DEFAULT, REJECT_THRESHOLD_DEFAULT


@dataclass
class PipelineResult:
    scored_df: pd.DataFrame
    train_lots: list
    test_lots: list
    if_model: object
    drift_bundle: object
    z_threshold: float
    watch_threshold: float
    reject_threshold: float
    validation_issues: list
    classification_report: object
    regression_report: object
    train_classification_report: object


def run_pipeline(
    raw_df: pd.DataFrame,
    test_fraction: float = 0.3,
    z_threshold: float = Z_THRESHOLD_DEFAULT,
    watch_threshold: float = WATCH_THRESHOLD_DEFAULT,
    reject_threshold: float = REJECT_THRESHOLD_DEFAULT,
    split_seed: int = 0,
) -> PipelineResult:
    clean_df, issues = validate_dataset(raw_df)
    if any("Missing required columns" in i for i in issues):
        raise ValueError("Dataset validation failed: " + "; ".join(issues))
    if len(clean_df) == 0:
        raise ValueError("Dataset has no valid rows after validation: " + "; ".join(issues))

    train_lots, test_lots = split_by_lot(clean_df, test_fraction=test_fraction, seed=split_seed)
    if train_lots == test_lots:
        issues.append(
            "Only one lot is present -- a genuine held-out test split is not possible. "
            "Metrics below are computed on the SAME lot used for fitting and are optimistic; "
            "do not present them as unseen-lot performance."
        )

    featured_df = engineer_features(clean_df)
    train_df = featured_df[featured_df["lot_id"].isin(train_lots)].copy()
    test_df = featured_df[featured_df["lot_id"].isin(test_lots)].copy()

    if_model = fit_isolation_forest(train_df)
    drift_bundle = fit_drift_model(train_df)

    def _score_all(sub_df: pd.DataFrame) -> pd.DataFrame:
        sub_df = score_module_a(sub_df, if_model, z_threshold=z_threshold)
        sub_df = score_module_b(sub_df, drift_bundle)
        sub_df["module_a_score"] = sub_df["module_a_score"].fillna(0.0)
        sub_df["module_b_score"] = sub_df["module_b_score"].fillna(0.0)
        sub_df["risk_score"] = combine_risk_score(sub_df["module_a_score"], sub_df["module_b_score"])
        sub_df["qa_decision"] = decide(sub_df["risk_score"], watch_threshold, reject_threshold)
        return sub_df

    train_scored = _score_all(train_df)
    train_scored["split"] = "train"
    test_scored = _score_all(test_df)
    test_scored["split"] = "test"

    scored_df = pd.concat([train_scored, test_scored], ignore_index=True)

    # Evaluation is reported on the TEST split only (held-out lots).
    has_labels = scored_df["ground_truth_anomaly"].notna().all()
    classification_report = None
    train_classification_report = None
    regression_report = None
    if has_labels and len(test_scored) > 0:
        classification_report = evaluate_classification(
            test_scored["ground_truth_anomaly"],
            test_scored["qa_decision"] != "PASS",
            test_scored["risk_score"],
        )
        train_classification_report = evaluate_classification(
            train_scored["ground_truth_anomaly"],
            train_scored["qa_decision"] != "PASS",
            train_scored["risk_score"],
        )
        regression_report = evaluate_regression(
            test_scored["Value_168h"],
            test_scored["predicted_168h"],
            test_scored["predicted_168h_baseline_linear"],
        )

    return PipelineResult(
        scored_df=scored_df,
        train_lots=train_lots,
        test_lots=test_lots,
        if_model=if_model,
        drift_bundle=drift_bundle,
        z_threshold=z_threshold,
        watch_threshold=watch_threshold,
        reject_threshold=reject_threshold,
        validation_issues=issues,
        classification_report=classification_report,
        regression_report=regression_report,
        train_classification_report=train_classification_report,
    )
