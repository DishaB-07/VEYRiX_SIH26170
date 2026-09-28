"""
AI-Driven Component Burn-In Intelligence -- SIH 26170 QA Screening Assistant.

A Streamlit dashboard wrapping the full pipeline:
  raw data -> validation -> lot statistics -> Module A (dynamic lot-relative
  anomaly detection) -> Module B (0h+24h -> 168h drift prediction) ->
  prototype safety criterion -> combined risk score -> PASS/WATCH/REJECT ->
  explainable QA decision.

Run locally with:  streamlit run app.py
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from src.data_generator import (
    generate_synthetic_dataset, inject_demo_cases, DATASET_LABEL,
    PARAMETER_PROFILES,
)
from src.preprocessing import REQUIRED_COLUMNS, VALUE_COLS
from src.pipeline import run_pipeline
from src.anomaly_detection import Z_THRESHOLD_DEFAULT
from src.explainability import (
    explain_module_a, explain_module_b, explain_decision,
    WATCH_THRESHOLD_DEFAULT, REJECT_THRESHOLD_DEFAULT,
)
from src.evaluation import split_by_lot

# --------------------------------------------------------------------------
# Page config + styling
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="AI-Driven Component Burn-In Intelligence | SIH 26170",
    page_icon="\U0001F6F0\uFE0F",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
:root {
    --navy: #0B1D3A;
    --navy2: #13294F;
    --accent: #FFB300;
    --ok: #1E7B4D;
    --watch: #B9770E;
    --reject: #A93226;
}
.main .block-container {padding-top: 1.2rem; max-width: 1300px;}
.app-header {
    background: linear-gradient(120deg, var(--navy) 0%, var(--navy2) 100%);
    padding: 1.4rem 1.8rem; border-radius: 10px; margin-bottom: 1rem;
}
.app-header h1 {color: white; margin: 0; font-size: 1.6rem; font-weight: 700;}
.app-header p {color: #B9C7E6; margin: 0.2rem 0 0 0; font-size: 0.95rem;}
.data-banner {
    background: #FFF4E0; border-left: 4px solid var(--accent);
    padding: 0.55rem 0.9rem; border-radius: 6px; font-size: 0.85rem;
    color: #6B4A00; margin-bottom: 0.8rem;
}
.metric-card {
    background: white; border: 1px solid #E4ECF8; border-left: 4px solid var(--navy);
    border-radius: 8px; padding: 0.7rem 0.9rem; text-align: left;
}
.metric-card .label {font-size: 0.72rem; color: #5A6C8C; font-weight: 600; text-transform: uppercase; letter-spacing: 0.03em;}
.metric-card .value {font-size: 1.55rem; color: #16213E; font-weight: 700; line-height: 1.3;}
.badge {padding: 0.15rem 0.6rem; border-radius: 999px; font-size: 0.75rem; font-weight: 700; color: white; display: inline-block;}
.badge-pass {background: var(--ok);}
.badge-watch {background: var(--watch);}
.badge-reject {background: var(--reject);}
.explain-card {background: #0B1D3A; color: #E8EEF7; border-radius: 8px; padding: 1rem 1.2rem; font-size: 0.9rem; line-height: 1.5; white-space: pre-wrap;}
.section-title {font-size: 1.1rem; font-weight: 700; color: #16213E; margin: 0.4rem 0 0.6rem 0;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

DECISION_COLORS = {"PASS": "#1E7B4D", "WATCH": "#B9770E", "REJECT": "#A93226"}


def badge(decision: str) -> str:
    cls = {"PASS": "badge-pass", "WATCH": "badge-watch", "REJECT": "badge-reject"}[decision]
    return f'<span class="badge {cls}">{decision}</span>'


def metric_card(label: str, value: str) -> str:
    return f'<div class="metric-card"><div class="label">{label}</div><div class="value">{value}</div></div>'


# --------------------------------------------------------------------------
# Session state defaults
# --------------------------------------------------------------------------
DEFAULTS = dict(
    raw_df=None, demo_ids=None, nav="Dashboard",
    z_threshold=Z_THRESHOLD_DEFAULT,
    watch_threshold=WATCH_THRESHOLD_DEFAULT,
    reject_threshold=REJECT_THRESHOLD_DEFAULT,
    test_fraction=0.30, split_seed=1, n_lots=14,
)
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


def load_synthetic(n_lots: int, seed: int = 42):
    st.session_state.raw_df = generate_synthetic_dataset(n_lots=n_lots, seed=seed)
    st.session_state.demo_ids = None


def load_demo():
    df = generate_synthetic_dataset(n_lots=14, seed=42)
    train_lots, test_lots = split_by_lot(df, test_fraction=st.session_state.test_fraction, seed=st.session_state.split_seed)
    target_lot = test_lots[0] if test_lots else sorted(df["lot_id"].unique())[-1]
    df = inject_demo_cases(df, target_lot_id=target_lot)
    st.session_state.raw_df = df
    st.session_state.demo_ids = ["DEMO-CASE-A", "DEMO-CASE-B", "DEMO-CASE-C"]
    st.session_state.nav = "QA Decisions"


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### \U0001F6F0\uFE0F Burn-In Intelligence")
    st.caption("SIH 26170 | QA Screening Assistant")

    nav_options = ["Dashboard", "Dataset", "Lot Analysis", "Anomaly Detection",
                   "Drift Prediction", "QA Decisions", "Model Performance"]
    nav = st.radio("Navigate", nav_options, index=nav_options.index(st.session_state.nav), label_visibility="collapsed")
    st.session_state.nav = nav

    st.divider()
    st.markdown("#### Data source")
    if st.button("\U0001F3AC Load Demo Scenario", width='stretch', type="primary"):
        load_demo()
    n_lots = st.slider("Number of lots (synthetic)", 6, 24, st.session_state.n_lots)
    st.session_state.n_lots = n_lots
    if st.button("\U0001F504 Generate new synthetic dataset", width='stretch'):
        load_synthetic(n_lots)

    uploaded = st.file_uploader("...or upload a CSV", type=["csv"])
    if uploaded is not None:
        try:
            st.session_state.raw_df = pd.read_csv(uploaded)
            st.session_state.demo_ids = None
            st.success(f"Loaded {uploaded.name}")
        except Exception as e:
            st.error(f"Could not read CSV: {e}")

    st.divider()
    st.markdown("#### Prototype thresholds")
    st.caption("Statistical engineering thresholds for this prototype -- not official ISRO limits.")
    st.session_state.z_threshold = st.slider("Module A: robust Z-score threshold", 2.0, 6.0, float(st.session_state.z_threshold), 0.1)
    st.session_state.watch_threshold = st.slider("WATCH risk-score threshold", 0.05, 0.9, float(st.session_state.watch_threshold), 0.01)
    reject_min = max(st.session_state.watch_threshold + 0.01, 0.1)
    st.session_state.reject_threshold = st.slider("REJECT risk-score threshold", reject_min, 0.99, max(float(st.session_state.reject_threshold), reject_min), 0.01)
    st.session_state.test_fraction = st.slider("Held-out test lot fraction", 0.15, 0.5, float(st.session_state.test_fraction), 0.05)
    st.session_state.split_seed = st.number_input("Lot split seed", 0, 999, int(st.session_state.split_seed))

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.markdown(
    '<div class="app-header"><h1>AI-Driven Component Burn-In Intelligence</h1>'
    '<p>SIH 26170 | QA Screening Assistant &mdash; Dynamic anomaly detection '
    '+ early drift prediction for high-reliability burn-in screening</p></div>',
    unsafe_allow_html=True,
)

if st.session_state.raw_df is None:
    st.info(
        "No dataset loaded yet. Click **Load Demo Scenario** in the sidebar for the fastest way to "
        "see the pipeline in action, or **Generate new synthetic dataset**, or upload your own CSV."
    )
    st.markdown(f"**Required columns:** `{', '.join(REQUIRED_COLUMNS)}` (+ optional `ground_truth_anomaly`, `anomaly_type`)")
    st.stop()

st.markdown(f'<div class="data-banner">\u26A0\uFE0F {DATASET_LABEL if st.session_state.demo_ids or True else ""}'
            f' This prototype has not been validated against real ISRO burn-in data.</div>', unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Run pipeline
# --------------------------------------------------------------------------
try:
    result = run_pipeline(
        st.session_state.raw_df,
        test_fraction=st.session_state.test_fraction,
        z_threshold=st.session_state.z_threshold,
        watch_threshold=st.session_state.watch_threshold,
        reject_threshold=st.session_state.reject_threshold,
        split_seed=st.session_state.split_seed,
    )
except Exception as e:
    st.error(f"Pipeline failed: {e}")
    st.stop()

for issue in result.validation_issues:
    st.warning(issue)

scored = result.scored_df

# ==========================================================================
# PAGE: Dashboard
# ==========================================================================
if nav == "Dashboard":
    decision_counts = scored["qa_decision"].value_counts()
    n_total = len(scored)
    n_lots_total = scored["lot_id"].nunique()
    n_normal = int(decision_counts.get("PASS", 0))
    n_watch = int(decision_counts.get("WATCH", 0))
    n_reject = int(decision_counts.get("REJECT", 0))
    recall_txt = f"{result.classification_report.recall*100:.1f}%" if result.classification_report and result.classification_report.recall is not None else "n/a"
    mae_txt = f"{result.regression_report.mae:.2f}" if result.regression_report else "n/a"

    cols = st.columns(7)
    for c, (label, value) in zip(cols, [
        ("Components Screened", f"{n_total}"),
        ("Lots", f"{n_lots_total}"),
        ("Normal (PASS)", f"{n_normal}"),
        ("Watch", f"{n_watch}"),
        ("Reject", f"{n_reject}"),
        ("Recall (test lots)", recall_txt),
        ("168h MAE (test lots)", mae_txt),
    ]):
        c.markdown(metric_card(label, value), unsafe_allow_html=True)

    st.write("")
    left, right = st.columns([1, 1])
    with left:
        st.markdown('<div class="section-title">Decision breakdown</div>', unsafe_allow_html=True)
        fig = px.pie(
            names=decision_counts.index, values=decision_counts.values,
            color=decision_counts.index, color_discrete_map=DECISION_COLORS, hole=0.55,
        )
        fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=320)
        st.plotly_chart(fig, width='stretch')
    with right:
        st.markdown('<div class="section-title">Average risk score by lot</div>', unsafe_allow_html=True)
        by_lot = scored.groupby("lot_id")["risk_score"].mean().reset_index().sort_values("lot_id")
        fig2 = px.bar(by_lot, x="lot_id", y="risk_score", color="risk_score", color_continuous_scale="Oranges")
        fig2.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=320, coloraxis_showscale=False)
        st.plotly_chart(fig2, width='stretch')

    if st.session_state.demo_ids:
        st.info("Demo scenario loaded -- open **QA Decisions** to see CASE A / B / C highlighted with full explanations.")

# ==========================================================================
# PAGE: Dataset
# ==========================================================================
elif nav == "Dataset":
    st.markdown('<div class="section-title">Synthetic burn-in dataset</div>', unsafe_allow_html=True)
    st.caption(DATASET_LABEL)

    c1, c2, c3 = st.columns(3)
    c1.markdown(metric_card("Rows", f"{len(scored)}"), unsafe_allow_html=True)
    c2.markdown(metric_card("Lots", f"{scored['lot_id'].nunique()}"), unsafe_allow_html=True)
    c3.markdown(metric_card("Labelled anomalies", f"{int(scored['ground_truth_anomaly'].sum())}"), unsafe_allow_html=True)

    st.write("")
    lot_filter = st.multiselect("Filter by lot", sorted(scored["lot_id"].unique()))
    type_filter = st.multiselect("Filter by anomaly type", sorted(scored["anomaly_type"].unique()))
    view = scored.copy()
    if lot_filter:
        view = view[view["lot_id"].isin(lot_filter)]
    if type_filter:
        view = view[view["anomaly_type"].isin(type_filter)]

    display_cols = ["component_id", "lot_id", "component_type", "parameter_type",
                     "Value_0h", "Value_24h", "Value_96h", "Value_168h",
                     "datasheet_limit", "ground_truth_anomaly", "anomaly_type", "split", "qa_decision"]
    st.dataframe(view[display_cols], width='stretch', height=420)
    st.download_button("Download current dataset (CSV)", scored[display_cols].to_csv(index=False), "burnin_dataset.csv", "text/csv")

    with st.expander("Column reference"):
        st.markdown("""
| Column | Meaning |
|---|---|
| `component_id` / `lot_id` | Identifiers; lot grouping drives all lot-relative statistics |
| `parameter_type` | Which screening parameter this row represents (leakage / Iddq / propagation delay) |
| `Value_0h/24h/96h/168h` | Parametric reading at each burn-in checkpoint |
| `datasheet_limit` | Absolute static pass/fail limit for this parameter |
| `ground_truth_anomaly` / `anomaly_type` | Synthetic labels, used only for evaluation and for cost-sensitive training weighting -- never fed to the models as a feature |
| `split` | train / test lot, from the lot-wise split |
| `qa_decision` | PASS / WATCH / REJECT, this prototype's output |
""")

# ==========================================================================
# PAGE: Lot Analysis
# ==========================================================================
elif nav == "Lot Analysis":
    st.markdown('<div class="section-title">Lot-wise statistics</div>', unsafe_allow_html=True)

    lot_summary = scored.groupby("lot_id").agg(
        n=("component_id", "count"),
        parameter_type=("parameter_type", "first"),
        datasheet_limit=("datasheet_limit", "first"),
        median_0h=("Value_0h", "median"),
        flagged=("qa_decision", lambda s: (s != "PASS").sum()),
    ).reset_index().sort_values("lot_id")
    st.dataframe(lot_summary, width='stretch', height=280)

    st.write("")
    sel_lot = st.selectbox("Inspect a lot", sorted(scored["lot_id"].unique()))
    checkpoint = st.selectbox("Checkpoint", VALUE_COLS, index=0)
    lot_df = scored[scored["lot_id"] == sel_lot]
    median_col = f"{checkpoint}_lot_median"
    mad_col = f"{checkpoint}_lot_mad"

    fig = go.Figure()
    fig.add_trace(go.Box(y=lot_df[checkpoint], name=sel_lot, boxpoints="all", jitter=0.4, marker=dict(color="#1C4E80")))
    fig.add_hline(y=lot_df[median_col].iloc[0], line_dash="dot", line_color="#16213E", annotation_text="lot median")
    fig.add_hline(y=lot_df["datasheet_limit"].iloc[0], line_dash="dash", line_color="#A93226", annotation_text="datasheet limit")
    fig.update_layout(height=420, title=f"{sel_lot} -- {checkpoint} distribution", margin=dict(t=50))
    st.plotly_chart(fig, width='stretch')

    c1, c2, c3 = st.columns(3)
    c1.markdown(metric_card(f"{checkpoint} median", f"{lot_df[median_col].iloc[0]:.2f}"), unsafe_allow_html=True)
    c2.markdown(metric_card(f"{checkpoint} MAD", f"{lot_df[mad_col].iloc[0]:.3f}"), unsafe_allow_html=True)
    c3.markdown(metric_card("Datasheet limit", f"{lot_df['datasheet_limit'].iloc[0]:.1f}"), unsafe_allow_html=True)

# ==========================================================================
# PAGE: Anomaly Detection (Module A)
# ==========================================================================
elif nav == "Anomaly Detection":
    st.markdown('<div class="section-title">Module A -- Dynamic Lot-Relative Anomaly Detection</div>', unsafe_allow_html=True)
    with st.expander("Method", expanded=False):
        st.markdown(f"""
Primary signal: robust Z-score built from each lot's own median and MAD (Median Absolute Deviation):

`robust_z = 0.6745 * (value - lot_median) / (MAD + epsilon)`

flagged if `max(|robust_z|)` across checkpoints exceeds **{st.session_state.z_threshold:.1f}** (adjustable in the sidebar;
a documented statistical engineering threshold, not an ISRO limit).

Secondary/backstop signal: Isolation Forest over lot-normalised, multi-checkpoint ratios (fit on training lots only,
`contamination="auto"` -- no hand-picked global contamination fraction).
""")

    checkpoint = st.selectbox("Checkpoint to visualise", VALUE_COLS, index=1)
    view = scored.copy()
    view["Decision"] = view["qa_decision"]
    fig = px.scatter(
        view, x="lot_id", y=checkpoint, color="module_a_flag",
        color_discrete_map={True: "#A93226", False: "#1C4E80"},
        hover_data=["component_id", "max_abs_robust_z", "driving_checkpoint", "datasheet_limit"],
    )
    limit_val = view["datasheet_limit"].iloc[0] if view["parameter_type"].nunique() == 1 else None
    fig.update_layout(height=460, legend_title_text="Module A flag")
    st.plotly_chart(fig, width='stretch')

    flagged = scored[scored["module_a_flag"]].sort_values("max_abs_robust_z", ascending=False)
    st.markdown(f"**{len(flagged)} components flagged by Module A**")
    st.dataframe(
        flagged[["component_id", "lot_id", "driving_checkpoint", "max_abs_robust_z",
                 "if_anomaly_score", "module_a_score", "datasheet_limit", "ground_truth_anomaly", "qa_decision"]],
        width='stretch', height=350,
    )

# ==========================================================================
# PAGE: Drift Prediction (Module B)
# ==========================================================================
elif nav == "Drift Prediction":
    st.markdown('<div class="section-title">Module B -- 0h + 24h &rarr; 168h Drift Prediction</div>', unsafe_allow_html=True)
    st.caption(f"Regression model in use: **{result.drift_bundle.model_name}**")

    with st.expander("Safety criterion (prototype, not an ISRO limit)", expanded=False):
        st.markdown(
            "Safety slope is derived per parameter type from the **training** lots' own relative "
            "(fractional) drift-rate distribution: `median + 3xMAD`. This is a prototype statistical "
            "criterion, not an official ISRO limit."
        )
        slope_tbl = pd.DataFrame([
            {"parameter_type": k, "safety_slope_%_per_hour": v * 100}
            for k, v in result.drift_bundle.safety_slope_by_type.items()
        ])
        st.dataframe(slope_tbl, width='stretch')

    test_scored = scored[scored["split"] == "test"]
    fig = px.scatter(
        test_scored, x="Value_168h", y="predicted_168h", color="module_b_flag",
        color_discrete_map={True: "#A93226", False: "#1C4E80"},
        hover_data=["component_id", "lot_id", "projected_drift_rate", "safety_slope"],
        labels={"Value_168h": "Actual 168h (ground truth, held-out test lots)", "predicted_168h": "Predicted 168h"},
    )
    lo = min(test_scored["Value_168h"].min(), test_scored["predicted_168h"].min())
    hi = max(test_scored["Value_168h"].max(), test_scored["predicted_168h"].max())
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(dash="dash", color="gray"), name="perfect prediction"))
    fig.update_layout(height=460, legend_title_text="Module B flag")
    st.plotly_chart(fig, width='stretch')

    flagged_b = scored[scored["module_b_flag"]].sort_values("projected_drift_rate", ascending=False)
    st.markdown(f"**{len(flagged_b)} components flagged by Module B**")
    st.dataframe(
        flagged_b[["component_id", "lot_id", "parameter_type", "Value_0h", "Value_24h",
                   "predicted_168h", "Value_168h", "projected_drift_rate", "safety_slope",
                   "ground_truth_anomaly", "qa_decision"]],
        width='stretch', height=350,
    )

# ==========================================================================
# PAGE: QA Decisions
# ==========================================================================
elif nav == "QA Decisions":
    if st.session_state.demo_ids:
        st.markdown('<div class="section-title">\U0001F3AC Demo scenario</div>', unsafe_allow_html=True)
        demo_rows = scored[scored["component_id"].isin(st.session_state.demo_ids)]
        demo_labels = {
            "DEMO-CASE-A": "CASE A -- passes the absolute limit, fails lot-relative check",
            "DEMO-CASE-B": "CASE B -- looks fine at 0h/24h, unsafe by 168h (early forecast)",
            "DEMO-CASE-C": "CASE C -- clean, normal component",
        }
        for cid in st.session_state.demo_ids:
            row_matches = demo_rows[demo_rows["component_id"] == cid]
            if len(row_matches) == 0:
                continue
            row = row_matches.iloc[0]
            with st.expander(f"{demo_labels.get(cid, cid)}  {badge(row['qa_decision'])}", expanded=True):
                st.markdown(badge(row["qa_decision"]), unsafe_allow_html=True)
                st.markdown(f'<div class="explain-card">{explain_decision(row, result.z_threshold)}</div>', unsafe_allow_html=True)
        st.divider()

    st.markdown('<div class="section-title">All QA decisions</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    decision_filter = c1.multiselect("Decision", ["PASS", "WATCH", "REJECT"], default=["WATCH", "REJECT"])
    lot_filter2 = c2.multiselect("Lot", sorted(scored["lot_id"].unique()))
    search = c3.text_input("Search component_id")

    view = scored.copy()
    if decision_filter:
        view = view[view["qa_decision"].isin(decision_filter)]
    if lot_filter2:
        view = view[view["lot_id"].isin(lot_filter2)]
    if search:
        view = view[view["component_id"].str.contains(search, case=False, na=False)]

    st.dataframe(
        view[["component_id", "lot_id", "qa_decision", "risk_score", "module_a_flag", "module_b_flag",
              "ground_truth_anomaly", "split"]].sort_values("risk_score", ascending=False),
        width='stretch', height=350,
    )

    st.write("")
    st.markdown("#### Inspect a component")
    options = view["component_id"].tolist() if len(view) else scored["component_id"].tolist()
    if options:
        chosen = st.selectbox("Component", options)
        row = scored[scored["component_id"] == chosen].iloc[0]
        st.markdown(badge(row["qa_decision"]), unsafe_allow_html=True)
        st.markdown(f'<div class="explain-card">{explain_decision(row, result.z_threshold)}</div>', unsafe_allow_html=True)

# ==========================================================================
# PAGE: Model Performance
# ==========================================================================
elif nav == "Model Performance":
    st.markdown('<div class="section-title">Test-set performance (held-out lots only)</div>', unsafe_allow_html=True)
    st.caption(f"Train lots: {', '.join(result.train_lots)}  |  Test lots: {', '.join(result.test_lots)}")

    cr = result.classification_report
    rr = result.regression_report
    if cr is None or rr is None:
        st.warning("Ground-truth labels are not available (or the test split has no positives) -- classification/regression metrics cannot be computed for this dataset.")
    else:
        cols = st.columns(6)
        for c, (label, value) in zip(cols, [
            ("Recall / Sensitivity", f"{cr.recall*100:.1f}%"),
            ("False Negative Rate", f"{cr.fnr*100:.1f}%"),
            ("Precision", f"{cr.precision*100:.1f}%"),
            ("F1", f"{cr.f1:.2f}"),
            ("PR-AUC", f"{cr.pr_auc:.2f}" if cr.pr_auc is not None else "n/a"),
            ("168h MAE", f"{rr.mae:.2f}"),
        ]):
            c.markdown(metric_card(label, value), unsafe_allow_html=True)
        if cr.note:
            st.caption(cr.note)

        left, right = st.columns(2)
        with left:
            st.markdown('<div class="section-title">Confusion matrix (test lots)</div>', unsafe_allow_html=True)
            cm = cr.confusion
            fig = px.imshow(
                cm, x=["Predicted Normal", "Predicted Anomalous"], y=["Actual Normal", "Actual Anomalous"],
                text_auto=True, color_continuous_scale="Blues",
            )
            fig.update_layout(height=350, coloraxis_showscale=False)
            st.plotly_chart(fig, width='stretch')
        with right:
            st.markdown('<div class="section-title">168h MAE: model vs. linear baseline</div>', unsafe_allow_html=True)
            bar_df = pd.DataFrame({
                "model": ["Gradient boosting", "Linear regression (baseline)"],
                "MAE": [rr.mae, rr.mae_baseline],
                "RMSE": [rr.rmse, rr.rmse_baseline],
            })
            fig2 = px.bar(bar_df, x="model", y=["MAE", "RMSE"], barmode="group", color_discrete_sequence=["#1C4E80", "#FFB300"])
            fig2.update_layout(height=350)
            st.plotly_chart(fig2, width='stretch')

        if result.train_classification_report is not None:
            tr = result.train_classification_report
            st.caption(
                f"For reference (train lots, NOT a held-out evaluation -- optimistic by construction): "
                f"recall {tr.recall*100:.1f}%, precision {tr.precision*100:.1f}%. The gap between this and "
                f"the test-lot numbers above is the honest measure of how well this prototype generalises "
                f"to genuinely unseen lots."
            )

    st.divider()
    st.markdown("#### Validation log")
    if result.validation_issues:
        for issue in result.validation_issues:
            st.write(f"- {issue}")
    else:
        st.write("No validation issues.")
